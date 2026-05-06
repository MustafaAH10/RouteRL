#!/usr/bin/env python
from __future__ import annotations

import argparse

from route_env.io import read_jsonl, write_jsonl
from route_env.strip_tasks import make_route_strip_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="flat RouteRL tasks.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-segment-distance-m", type=float, default=2500)
    parser.add_argument("--max-segment-checkpoints", type=int, default=32)
    parser.add_argument("--segment-margin-m", type=float, default=260)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    parents = read_jsonl(args.tasks)
    if args.limit:
        parents = parents[: args.limit]
    strips = [
        make_route_strip_task(
            task,
            target_segment_distance_m=args.target_segment_distance_m,
            max_segment_checkpoints=args.max_segment_checkpoints,
            segment_margin_m=args.segment_margin_m,
            seed=args.seed + index,
        )
        for index, task in enumerate(parents)
    ]
    write_jsonl(args.out, strips)
    print(f"wrote {len(strips)} route-strip tasks to {args.out}")


if __name__ == "__main__":
    main()
