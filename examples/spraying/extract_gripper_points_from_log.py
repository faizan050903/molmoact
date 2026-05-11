"""Parse a preprocess_shard{N}.log file and emit a JSON list of per-frame
gripper points, in iteration order. Used to resume a crashed shard without
re-running Molmo for the frames we already processed.

Each `Debug gripper result {dict}` line in the log corresponds to one frame in
`collect_gripper_points` iteration order. We extract the `x_px`/`y_px` fields
and produce a JSON list `[[x_px, y_px], ...]` where `None` is allowed for
frames where Molmo failed to locate a point.

Handles two log quirks defensively:

1. tqdm's progress bar overwrites stdout on the same line via `\r`, which
   `tee` preserves as a carriage return rather than a newline. We split on
   `\r` and `\n`.
2. The final `Debug gripper result {...` may be truncated if the process
   crashed mid-write (e.g. ENOSPC). We catch parse errors and skip that line.

Usage:
    python extract_gripper_points_from_log.py \
        --log ~/molmoact-logs/preprocess_shard0.log \
        --output ~/molmoact-logs/gripper_points_cache/shard0.json
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


PREFIX = "Debug gripper result "


def parse_log(log_path: Path) -> list[list[int] | None]:
    points: list[list[int] | None] = []
    raw = log_path.read_text(errors="ignore")
    # tqdm intersperses \r; treat both \r and \n as record separators.
    for line in raw.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line.startswith(PREFIX):
            continue
        try:
            d = ast.literal_eval(line[len(PREFIX):])
        except (SyntaxError, ValueError):
            # Likely a truncated tail line from a crash; skip it.
            continue
        if not isinstance(d, dict):
            continue
        x_px = d.get("x_px")
        y_px = d.get("y_px")
        if x_px is None or y_px is None:
            points.append(None)
        else:
            try:
                points.append([int(x_px), int(y_px)])
            except (TypeError, ValueError):
                points.append(None)
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    points = parse_log(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(points))
    n_total = len(points)
    n_valid = sum(1 for p in points if p is not None)
    print(f"{args.log.name}: parsed {n_total} frames ({n_valid} with valid points) -> {args.output}")


if __name__ == "__main__":
    main()
