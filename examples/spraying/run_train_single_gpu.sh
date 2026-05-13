#!/usr/bin/env bash
# Single-GPU LoRA fine-tune of MolmoAct-7B-D-0812 on the cleaned spraying
# dataset. Reliable production path: no FSDP cross-rank gather, no PEFT
# state-dict hook bugs, no NCCL collectives during save.
#
# Multi-GPU (run_train.sh) hit a wall in this PyTorch 2.7.1 + nvidia-nccl-cu12
# 2.26.2 environment: FSDP1's gather code SIGSEGVs when PEFT-wrapped modules
# are present, regardless of sharding_strategy / state_dict API. Single-GPU
# sidesteps every one of those bugs and trains reliably.
#
# Throughput vs the 8-GPU recipe:
#   - Smoke test on 1 GPU @ batch=1: ~0.58 step/s, ~1300 tok/s
#   - 8 GPU @ global_batch=16:       ~0.34 step/s, ~1500 tok/s/device (12k total)
# Single-GPU at batch=4 is roughly 6-7x slower per useful example, but the
# wandb run from the earlier 8-GPU attempt showed loss plateauing around
# step 3-5k, so 5000-10000 steps single-GPU is sufficient.
#
# Usage:
#   cd ~/molmoact && source .venv/bin/activate
#   bash examples/spraying/run_train_single_gpu.sh
#
# Override knobs:
#   DURATION=20000 bash examples/spraying/run_train_single_gpu.sh
#   GLOBAL_BATCH=8 bash examples/spraying/run_train_single_gpu.sh   # if memory permits
#   RUN_NAME=molmoact_spraying_single_gpu_v2 bash examples/spraying/run_train_single_gpu.sh

set -euo pipefail

# --- Knobs ---
RUN_NAME=${RUN_NAME:-molmoact_spraying_single_gpu_v1}
DATASET_NAME=${DATASET_NAME:-spraying-v1-cleaned-processed}
DURATION=${DURATION:-10000}              # 10k steps at single-GPU rate ~7h
GLOBAL_BATCH=${GLOBAL_BATCH:-4}          # single device; conservative
DEVICE_TRAIN_BATCH=${DEVICE_TRAIN_BATCH:-4}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-16}
SAVE_INTERVAL=${SAVE_INTERVAL:-2500}     # 4 mid-training saves
SAVE_KEEP=${SAVE_KEEP:-2}
WANDB_ENTITY=${WANDB_ENTITY:-10xconstruction}
WANDB_PROJECT=${WANDB_PROJECT:-vla-v0}
LR=${LR:-5e-4}

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

# shellcheck disable=SC1091
source "$HOME/molmoact/.venv/bin/activate"

mkdir -p "$(dirname "$LOG_FILE")"
export MOLMOACT_FINETUNE_PATH

echo "================================================================"
echo "Launching MolmoAct LoRA fine-tune (SINGLE GPU)"
echo "  Run name:        ${RUN_NAME}"
echo "  Dataset:         ${MOLMOACT_FINETUNE_PATH}"
echo "  Save folder:     ${SAVE_FOLDER}"
echo "  Log file:        ${LOG_FILE}"
echo "  GPUs:            1 (no FSDP, no NCCL gather, no PEFT save bugs)"
echo "  Duration:        ${DURATION} steps"
echo "  Global batch:    ${GLOBAL_BATCH}"
echo "  Per-device batch:${DEVICE_TRAIN_BATCH}"
echo "  LoRA rank/alpha: ${LORA_RANK}/${LORA_ALPHA}"
echo "  Save interval:   ${SAVE_INTERVAL} (keep ${SAVE_KEEP})"
echo "  LR (all groups): ${LR}"
echo "  Wandb:           ${WANDB_ENTITY}/${WANDB_PROJECT}/${RUN_NAME}"

# --- Optional: knockknock Slack notification ---
# Set KNOCKKNOCK_SLACK_WEBHOOK (+ optionally KNOCKKNOCK_SLACK_CHANNEL) to enable.
# Requires: uv pip install knockknock
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

# Single-GPU launch via torchrun --nproc-per-node=1.
# We still need NCCL env vars set in case any code path probes them, but
# with world_size=1 there are no real collectives, just no-op AllReduces.
NCCL_NET=Socket \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
NCCL_SHM_DISABLE=0 \
PYTHONPATH=. \
CUDA_VISIBLE_DEVICES=0 \
WANDB_API_KEY=${WANDB_API_KEY} \
"${KK_PREFIX[@]}" torchrun \
    --nnodes=1 \
    --nproc-per-node=1 \
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
# NOTE: --save_intermediate_unsharded_checkpoint / --save_final_unsharded_checkpoint
# intentionally OFF. They trigger save_unsharded_checkpoint() which calls
# olmo/train/checkpointer.py:save_unsharded — a different code path that
# uses dist_cp_sd.get_model_state_dict and hits PEFT's renamed keys.
# Our patched sharded save writes lora_state.pt which is all we need.
