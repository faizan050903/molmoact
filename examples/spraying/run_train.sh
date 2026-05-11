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
echo "================================================================"

# --- Launch ---
NCCL_NET=Socket \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=0 \
NCCL_SHM_DISABLE=0 \
PYTHONPATH=. \
WANDB_API_KEY=${WANDB_API_KEY} \
torchrun \
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
