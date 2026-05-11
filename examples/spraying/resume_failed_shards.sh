#!/usr/bin/env bash
# Resume a subset of shards using gripper-point caches built from their
# previous-run logs. Skips the Molmo calls for frames the prior run already
# processed; only the missing tail needs new Molmo inference.
#
# Usage:
#   bash examples/spraying/resume_failed_shards.sh "0 1 2 3 4 6"
#
# Reads logs from $LOG_DIR (default ~/molmoact-logs/), writes caches to
# $CACHE_DIR (default ~/molmoact-logs/gripper_points_cache/), then invokes
# preprocess_parallel.sh with SHARDS + GRIPPER_POINTS_CACHE_DIR set.

set -euo pipefail

SHARDS=${1:-"0 1 2 3 4 6"}
LOG_DIR=${LOG_DIR:-$HOME/molmoact-logs}
CACHE_DIR=${CACHE_DIR:-$HOME/molmoact-logs/gripper_points_cache}

cd "$HOME/molmoact"

mkdir -p "$CACHE_DIR"
echo "Extracting gripper-point caches for shards: $SHARDS"
for s in $SHARDS; do
    log="$LOG_DIR/preprocess_shard${s}.log"
    cache="$CACHE_DIR/shard${s}.json"
    if [ ! -f "$log" ]; then
        echo "  shard $s: log missing ($log) — will run from scratch" >&2
        continue
    fi
    python examples/spraying/extract_gripper_points_from_log.py \
        --log "$log" --output "$cache"
done

echo
echo "Launching preprocess_parallel.sh in resume mode..."
SHARDS="$SHARDS" \
GRIPPER_POINTS_CACHE_DIR="$CACHE_DIR" \
RUN_MERGE=no \
bash examples/spraying/preprocess_parallel.sh
