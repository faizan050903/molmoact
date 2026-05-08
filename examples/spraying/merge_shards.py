"""Concatenate per-GPU preprocessing shards into one HF dataset and
recompute global action statistics.

Each shard is the output dir of a `preprocess/action_reasoning_data.py` run on
a slice of episodes. Shards have identical schemas; we concat them with
`datasets.concatenate_datasets` and write a single `dataset_statistics.json`
that contains stats over the union of all actions.

Per-shard `dataset_statistics.json` files are ignored — they were keyed to
each shard's local episode subset and would mismatch the merged dataset's
distribution at inference time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import concatenate_datasets, load_from_disk


def compute_stats(merged, dataset_name: str, normalize_dims: int) -> dict:
    actions = np.array([np.asarray(a, dtype=np.float32) for a in merged["actions"]])
    if actions.ndim != 2:
        raise ValueError(f"expected actions shape (T, D), got {actions.shape}")
    n_dims = actions.shape[1]
    mask = [True] * min(normalize_dims, n_dims) + [False] * max(0, n_dims - normalize_dims)

    episode_ids = set(int(e) for e in merged["episode_index"]) if "episode_index" in merged.column_names else set()

    proprio_dim = n_dims
    return {
        dataset_name: {
            "action": {
                "mean": actions.mean(0).tolist(),
                "std": actions.std(0).tolist(),
                "min": actions.min(0).tolist(),
                "max": actions.max(0).tolist(),
                "q01": np.quantile(actions, 0.01, 0).tolist(),
                "q99": np.quantile(actions, 0.99, 0).tolist(),
                "mask": mask,
            },
            "num_transitions": int(actions.shape[0]),
            "num_trajectories": len(episode_ids),
            "proprio": {
                "mean": [0.0] * proprio_dim,
                "std": [0.0] * proprio_dim,
                "min": [0.0] * proprio_dim,
                "max": [0.0] * proprio_dim,
                "q01": [0.0] * proprio_dim,
                "q99": [0.0] * proprio_dim,
            },
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", required=True, help="Dir containing shard0, shard1, ...")
    parser.add_argument("--output-path", required=True, help="Where to write the merged dataset")
    parser.add_argument("--dataset-name", required=True, help="Key under which stats are saved (also used as --unnorm_key at inference)")
    parser.add_argument("--normalize-dims", type=int, default=8, help="Match what each shard was preprocessed with")
    parser.add_argument("--shard-glob", default="shard*", help="Glob pattern under --shards-root")
    args = parser.parse_args()

    shard_paths = sorted(Path(args.shards_root).glob(args.shard_glob))
    if not shard_paths:
        print(f"No shards matched {args.shards_root}/{args.shard_glob}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading {len(shard_paths)} shards from {args.shards_root}")
    shards = [load_from_disk(str(p)) for p in shard_paths]
    for p, ds in zip(shard_paths, shards):
        print(f"  {p.name}: {len(ds)} frames")

    merged = concatenate_datasets(shards)
    print(f"Merged: {len(merged)} frames")

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving merged dataset to {args.output_path}")
    merged.save_to_disk(args.output_path)

    print("Computing global action statistics...")
    stats = compute_stats(merged, args.dataset_name, args.normalize_dims)

    stats_path = Path(args.output_path) / "dataset_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {stats_path}")

    sample = merged[0]
    print("\nSample frame 0:")
    for k in ("episode_index", "frame_index", "language_instruction"):
        if k in sample:
            print(f"  {k}: {sample[k]}")
    print(f"  depth (head): {sample.get('depth', '')[:80]}")
    print(f"  trace: {sample.get('trace', '')}")


if __name__ == "__main__":
    main()
