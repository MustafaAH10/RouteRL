#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from route_env.io import read_jsonl
from route_env.render import render_debug_overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", default="data/debug")
    args = parser.parse_args()
    tasks = {task["task_id"]: task for task in read_jsonl(args.tasks)}
    preds = {pred["task_id"]: pred for pred in read_jsonl(args.predictions)}
    results = {result["task_id"]: result for result in read_jsonl(args.results)}
    for task_id, task in tqdm(tasks.items(), desc="debug"):
        render_debug_overlay(task, preds.get(task_id), results.get(task_id), Path(args.out_dir) / f"{task_id}.png")
    print(f"wrote overlays to {args.out_dir}")


if __name__ == "__main__":
    main()

