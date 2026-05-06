#!/usr/bin/env python
from __future__ import annotations

import argparse

from route_env.graph_tasks import GenerateConfig, generate_tasks
from route_env.io import write_jsonl


def parse_bbox(value: str) -> list[float]:
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", type=parse_bbox, required=True, help="west,south,east,north")
    parser.add_argument("--city", default="Singapore")
    parser.add_argument("--network-type", default="drive")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--min-distance-m", type=float, default=500)
    parser.add_argument("--max-distance-m", type=float, default=2000)
    parser.add_argument("--max-checkpoints", type=int, default=24)
    parser.add_argument("--route-margin-m", type=float, default=140)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = GenerateConfig(
        bbox=args.bbox,
        city=args.city,
        network_type=args.network_type,
        min_distance_m=args.min_distance_m,
        max_distance_m=args.max_distance_m,
        max_checkpoints=args.max_checkpoints,
        route_margin_m=args.route_margin_m,
        seed=args.seed,
    )
    tasks = generate_tasks(config, args.n)
    write_jsonl(args.out, tasks)
    print(f"wrote {len(tasks)} tasks to {args.out}")


if __name__ == "__main__":
    main()
