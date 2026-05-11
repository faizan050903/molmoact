#!/usr/bin/env bash
# Parallel action-reasoning preprocessing across N GPUs by sharding episodes.
# Each shard runs as its own Python process pinned to one GPU via CUDA_VISIBLE_DEVICES.
#
# Run AFTER setup_8gpu_box.sh has finished. Run from inside an active uv venv:
#   source ~/molmoact/.venv/bin/activate
#   bash ~/molmoact/examples/spraying/preprocess_parallel.sh
#
# Optionally override:
#   NUM_GPUS=4 bash preprocess_parallel.sh
#   DATASET_REPO_ID=rishi-10x/spraying-v1-local bash preprocess_parallel.sh

set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
DATASET_REPO_ID=${DATASET_REPO_ID:-rishi-10x/spraying-v1-cleaned}
OUT_BASE=${OUT_BASE:-$HOME/data/molmoact/$(basename "$DATASET_REPO_ID")-shards}
MERGED_OUT=${MERGED_OUT:-$HOME/data/molmoact/$(basename "$DATASET_REPO_ID")-processed}
LOG_DIR=${LOG_DIR:-$HOME/molmoact-logs}
POINT_PROMPT=${POINT_PROMPT:-"point to the spray nozzle"}

# Optional: subset of shard IDs to run (default: all NUM_GPUS).
#   SHARDS="0 1 2 3 4 6" bash preprocess_parallel.sh
# Each ID in SHARDS still pins to the GPU of the same numeric index.
SHARDS=${SHARDS:-}

# Optional: dir containing per-shard gripper-point cache JSONs (shard0.json,
# shard1.json, ...). When set and the file exists, that shard skips the Molmo
# call for cached frames. Build via extract_gripper_points_from_log.py.
GRIPPER_POINTS_CACHE_DIR=${GRIPPER_POINTS_CACHE_DIR:-}

# When SHARDS is set we typically want to resume an existing run rather than
# regenerate everything. The merge step would then drop the unprocessed shards.
# Default to skipping the merge so the caller can run it once everything is done.
RUN_MERGE=${RUN_MERGE:-auto}

cd "$HOME/molmoact"

export DEPTH_CHECKPOINT_DIR="$HOME/Depth-Anything-V2/checkpoints"
export VQVAE_MODEL_PATH="$HOME/molmoact/vae-final.pt"

mkdir -p "$OUT_BASE" "$LOG_DIR"

# Discover episode count from the LeRobot dataset metadata
N_EPS=$(python - <<PYEOF
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="${DATASET_REPO_ID}")
n = getattr(ds.meta, "total_episodes", None) or len(ds.meta.episodes)
print(n)
PYEOF
)
echo "Dataset $DATASET_REPO_ID has $N_EPS episodes; sharding across $NUM_GPUS GPUs"

PER_GPU=$(( (N_EPS + NUM_GPUS - 1) / NUM_GPUS ))

# Resolve which shard IDs to actually launch this run.
if [ -z "$SHARDS" ]; then
    SHARD_IDS=( $(seq 0 $((NUM_GPUS-1))) )
else
    read -r -a SHARD_IDS <<< "$SHARDS"
fi
echo "Running shards: ${SHARD_IDS[*]}"
if [ -n "$GRIPPER_POINTS_CACHE_DIR" ]; then
    echo "Looking for gripper-point caches in: $GRIPPER_POINTS_CACHE_DIR"
fi

PIDS=()
for gpu in "${SHARD_IDS[@]}"; do
    start=$(( gpu * PER_GPU ))
    end=$(( start + PER_GPU - 1 ))
    if [ $end -ge $N_EPS ]; then end=$(( N_EPS - 1 )); fi
    if [ $start -gt $end ]; then
        echo "  shard $gpu: no episodes assigned, skipping"
        continue
    fi
    eps=$(seq -s, "$start" "$end")

    log="$LOG_DIR/preprocess_shard${gpu}.log"
    out="$OUT_BASE/shard${gpu}"

    cache_args=()
    if [ -n "$GRIPPER_POINTS_CACHE_DIR" ] && [ -f "$GRIPPER_POINTS_CACHE_DIR/shard${gpu}.json" ]; then
        cache_args=(--gripper-points-cache "$GRIPPER_POINTS_CACHE_DIR/shard${gpu}.json")
        echo "  shard $gpu (eps $start-$end) on GPU $gpu -> $out  [resume from $GRIPPER_POINTS_CACHE_DIR/shard${gpu}.json]"
    else
        echo "  shard $gpu (eps $start-$end) on GPU $gpu -> $out"
    fi

    CUDA_VISIBLE_DEVICES=$gpu nohup python preprocess/action_reasoning_data.py \
        --dataset-path "$DATASET_REPO_ID" \
        --output-path "$out" \
        --depth-encoder vitb \
        --line-length 5 \
        --process-actions \
        --action-bins 256 \
        --action-chunk-size 8 \
        --normalize-dims 8 \
        --point-prompt "$POINT_PROMPT" \
        --episodes "$eps" \
        "${cache_args[@]}" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo "Launched ${#PIDS[@]} shard PIDs: ${PIDS[*]}"
echo "Waiting for all shards to complete..."
echo "Tail any log with: tail -f $LOG_DIR/preprocess_shard0.log"

# Block until all done
FAILED=()
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED+=("$pid")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED PIDs: ${FAILED[*]}" >&2
    echo "Inspect logs in $LOG_DIR/" >&2
    exit 1
fi

# Skip the merge step when only a subset of shards was requested (partial run).
# Caller can re-invoke this script with no SHARDS set, or call merge_shards.py
# directly, once all shards exist in $OUT_BASE.
should_merge() {
    case "$RUN_MERGE" in
        yes|true|1) return 0 ;;
        no|false|0) return 1 ;;
        auto)
            # auto: merge only if we ran all NUM_GPUS shards this invocation
            [ ${#SHARD_IDS[@]} -ge "$NUM_GPUS" ]
            ;;
        *) return 1 ;;
    esac
}

if ! should_merge; then
    echo
    echo "Skipping merge (partial run or RUN_MERGE=$RUN_MERGE)."
    echo "When all shards are on disk in $OUT_BASE, run:"
    echo "  python examples/spraying/merge_shards.py --shards-root $OUT_BASE --output-path $MERGED_OUT --dataset-name $(basename "$MERGED_OUT")"
    exit 0
fi

echo "All shards complete. Merging into $MERGED_OUT..."
python examples/spraying/merge_shards.py \
    --shards-root "$OUT_BASE" \
    --output-path "$MERGED_OUT" \
    --dataset-name "$(basename "$MERGED_OUT")"

echo
echo "Done. Merged dataset + dataset_statistics.json at: $MERGED_OUT"
echo "Push back to S3 with:"
echo "  aws s3 sync $MERGED_OUT/ s3://vla-data-collection/arranged_dummy_data/$(basename "$MERGED_OUT")/"
