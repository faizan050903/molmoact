#!/usr/bin/env bash
# 8-GPU LoRA fine-tune of MolmoAct-7B-D-0812 on the cleaned spraying dataset.
#
# Usage on a100gpu8 (or any 8x80GB GCP A3-Ultra node) after the merged
# processed dataset is at $MOLMOACT_FINETUNE_PATH:
#
#   cd ~/molmoact && source .venv/bin/activate
#   bash examples/spraying/run_train.sh
#
# Or, to override common knobs:
#
#   DURATION=30000 GLOBAL_BATCH=32 bash examples/spraying/run_train.sh
#   RUN_NAME=molmoact_spraying_cleaned_lora_v2 bash examples/spraying/run_train.sh
#
# Notes:
# - Uses FSDP1 (--fsdp.fsdp2=False) because PEFT's LoRA wrapper sees per-rank
#   shard sizes under FSDP2 and constructs lora_B with out_dim = base/world_size,
#   causing a 'tensor size 1152 vs 144' mismatch on the first forward.
# - NCCL_NET=Socket bypasses GCP's gIB plugin (built for NCCL 2.27.5 but
#   PyTorch 2.7.1+cu126 ships 2.26.2 — version skew segfaults gIB on this VM
#   image). For single-node training NVLink P2P handles intra-node bandwidth
#   anyway, so socket transport is functionally equivalent.
# - --fsdp.sharding_strategy=NO_SHARD: with FULL_SHARD, the FSDP state_dict
#   hook tries to gather LoRA-wrapped parameters and SIGSEGVs in PyTorch's
#   C++ FSDP code (any of the state_dict APIs trigger it). NO_SHARD makes
#   FSDP behave like DDP — each rank keeps a full replica — eliminating the
#   cross-rank gather entirely. Memory cost: ~16 GB bf16 model per GPU
#   instead of sharded ~2 GB; A100 80 GB has plenty of headroom.
# - The save path in olmo/train/distributed_checkpointing.py is also patched
#   to detect PEFT-wrapped models and bypass dist_cp.state_dict_saver
#   entirely. Rank 0 iterates named_parameters() and writes a plain
#   `lora_state.pt` via torch.save — no FSDP state_dict() call at all.
# - --save_intermediate_unsharded_checkpoint / --save_final_unsharded_checkpoint:
#   still kept on so the trainer also writes an unsharded checkpoint copy
#   (cheap given the LoRA bypass writes the only thing that matters).

set -euo pipefail

# --- Overridable knobs ---
RUN_NAME=${RUN_NAME:-molmoact_spraying_cleaned_lora_v1}
DATASET_NAME=${DATASET_NAME:-spraying-v1-cleaned-processed}
DURATION=${DURATION:-20000}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
DEVICE_TRAIN_BATCH=${DEVICE_TRAIN_BATCH:-2}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
SAVE_KEEP=${SAVE_KEEP:-2}
WANDB_ENTITY=${WANDB_ENTITY:-10xconstruction}
WANDB_PROJECT=${WANDB_PROJECT:-vla-v0}
LR=${LR:-5e-4}
NUM_GPUS=${NUM_GPUS:-8}

MOLMOACT_FINETUNE_PATH=${MOLMOACT_FINETUNE_PATH:-$HOME/data/molmoact/${DATASET_NAME}}
SAVE_FOLDER=${SAVE_FOLDER:-checkpoints/${RUN_NAME}}
LOG_FILE=${LOG_FILE:-$HOME/molmoact-logs/train_${RUN_NAME}.log}

# --- Pre-flight ---
cd "$HOME/molmoact"

# CRITICAL: strip GCP's gIB-NCCL paths from LD_LIBRARY_PATH. Those point at
# /usr/local/gib/lib{,64}/libnccl.so.2.27.5 (built for CUDA 12.8) which the
# dynamic linker prefers over PyTorch's bundled NCCL 2.26.2 (built for CUDA
# 12.6). The version skew causes every NCCL call to SIGSEGV. The GCP image's
# startup scripts re-add /usr/local/gib/lib64 on every login, so we filter at
# launch time.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | \
        grep -vE '/usr/local/gib' | grep -v '^$' | tr '\n' ':' | sed 's/:$//')
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: WANDB_API_KEY not set in environment" >&2
    exit 1
fi
if [ ! -f "${MOLMOACT_FINETUNE_PATH}/dataset_statistics.json" ]; then
    echo "ERROR: dataset_statistics.json not found at ${MOLMOACT_FINETUNE_PATH}/" >&2
    exit 1
fi
if [ ! -d "$HOME/molmoact/.venv" ]; then
    echo "ERROR: venv missing at $HOME/molmoact/.venv (run setup_8gpu_box.sh first)" >&2
    exit 1
fi

# Make sure the venv is active for the python torchrun spawns
# shellcheck disable=SC1091
source "$HOME/molmoact/.venv/bin/activate"

mkdir -p "$(dirname "$LOG_FILE")"

export MOLMOACT_FINETUNE_PATH

echo "================================================================"
echo "Launching MolmoAct LoRA fine-tune"
echo "  Run name:        ${RUN_NAME}"
echo "  Dataset:         ${MOLMOACT_FINETUNE_PATH}"
echo "  Save folder:     ${SAVE_FOLDER}"
echo "  Log file:        ${LOG_FILE}"
echo "  GPUs:            ${NUM_GPUS}"
echo "  Duration:        ${DURATION} steps"
echo "  Global batch:    ${GLOBAL_BATCH}"
echo "  Per-device batch:${DEVICE_TRAIN_BATCH}"
echo "  LoRA rank/alpha: ${LORA_RANK}/${LORA_ALPHA}"
echo "  Save interval:   ${SAVE_INTERVAL} (keep ${SAVE_KEEP})"
echo "  LR (all groups): ${LR}"
echo "  Wandb:           ${WANDB_ENTITY}/${WANDB_PROJECT}/${RUN_NAME}"

# --- Optional: knockknock Slack notification on success/failure ---
# Set KNOCKKNOCK_SLACK_WEBHOOK (and optionally KNOCKKNOCK_SLACK_CHANNEL) in env
# to get a Slack message when training finishes or errors out. Requires:
#   uv pip install knockknock
KK_PREFIX=()
if [ -n "${KNOCKKNOCK_SLACK_WEBHOOK:-}" ]; then
    if command -v knockknock >/dev/null 2>&1; then
        KK_PREFIX=(knockknock slack --webhook-url "${KNOCKKNOCK_SLACK_WEBHOOK}")
        if [ -n "${KNOCKKNOCK_SLACK_CHANNEL:-}" ]; then
            KK_PREFIX+=(--channel "${KNOCKKNOCK_SLACK_CHANNEL}")
        fi
        KK_PREFIX+=(--)
        echo "  Slack notify:    ENABLED${KNOCKKNOCK_SLACK_CHANNEL:+ (channel: ${KNOCKKNOCK_SLACK_CHANNEL})}"
    else
        echo "  Slack notify:    SKIPPED (KNOCKKNOCK_SLACK_WEBHOOK set but 'knockknock' not installed — run: uv pip install knockknock)"
    fi
else
    echo "  Slack notify:    off (set KNOCKKNOCK_SLACK_WEBHOOK to enable)"
fi
echo "================================================================"

# --- Launch ---
NCCL_NET=Socket \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
NCCL_SHM_DISABLE=0 \
PYTHONPATH=. \
WANDB_API_KEY=${WANDB_API_KEY} \
"${KK_PREFIX[@]}" torchrun \
    --nnodes=1 \
    --nproc-per-node="${NUM_GPUS}" \
    launch_scripts/train_multitask_model.py \
    robot-finetune allenai/MolmoAct-7B-D-0812 \
    --wandb.name="${RUN_NAME}" \
    --wandb.entity="${WANDB_ENTITY}" \
    --wandb.project="${WANDB_PROJECT}" \
    --norm_stats_path "${MOLMOACT_FINETUNE_PATH}/dataset_statistics.json" \
    --save_folder="${SAVE_FOLDER}" \
    --save_overwrite \
    --duration "${DURATION}" \
    --ft_embedding all \
    --depth_tokens \
    --global_batch_size "${GLOBAL_BATCH}" \
    --device_train_batch_size "${DEVICE_TRAIN_BATCH}" \
    --lr_connector "${LR}" \
    --lr_vit "${LR}" \
    --lr_llm "${LR}" \
    --save_interval "${SAVE_INTERVAL}" \
    --save_num_checkpoints_to_keep "${SAVE_KEEP}" \
    --max_images 2 \
    --lora_enable \
    --lora_rank "${LORA_RANK}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout 0.0 \
    --img_aug \
    --fsdp.fsdp2=False \
    2>&1 | tee "${LOG_FILE}"
# NOTE: --save_intermediate_unsharded_checkpoint and --save_final_unsharded_checkpoint
# are intentionally OFF. Both flags trigger trainer.save_unsharded_checkpoint()
# which calls olmo/train/checkpointer.py:save_unsharded → dist_cp_sd.get_model_state_dict
# (a separate code path we don't patch), and that path hits PEFT's renamed
# keys → KeyError mid-save. Our patched sharded save in
# olmo/train/distributed_checkpointing.py already writes lora_state.pt with
# all trainable params on rank 0 — that's the only file we need to reload
# the LoRA adapter at inference time.
