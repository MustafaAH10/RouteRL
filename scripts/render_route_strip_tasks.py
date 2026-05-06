#!/usr/bin/env python
from __future__ import annotations

import argparse

from tqdm import tqdm

from route_env.io import read_jsonl, write_jsonl
from route_env.render import render_route_strip_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--write-updated-tasks", help="optional JSONL path with route-strip image paths updated")
    args = parser.parse_args()

    tasks = read_jsonl(args.tasks)
    for task in tqdm(tasks, desc="strip-render"):
        render_route_strip_task(task, args.out_dir)
    if args.write_updated_tasks:
        write_jsonl(args.write_updated_tasks, tasks)
    print(f"rendered {len(tasks)} route-strip tasks under {args.out_dir}")


if __name__ == "__main__":
    main()
