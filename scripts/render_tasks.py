#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from route_env.io import read_jsonl, write_jsonl
from route_env.render import render_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out-dir", default="data/experiments/manual/maps")
    parser.add_argument("--write-updated-tasks", help="optional JSONL path with image paths updated to --out-dir")
    args = parser.parse_args()
    tasks = read_jsonl(args.tasks)
    for task in tqdm(tasks, desc="render"):
        out_path = Path(args.out_dir) / task["task_id"] / "map.png"
        render_task(task, out_path)
        task["images"]["map"] = str(out_path)
    if args.write_updated_tasks:
        write_jsonl(args.write_updated_tasks, tasks)
    print(f"rendered {len(tasks)} tasks under {args.out_dir}")


if __name__ == "__main__":
    main()
