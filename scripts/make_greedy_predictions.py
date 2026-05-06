#!/usr/bin/env python
from __future__ import annotations

import argparse

from route_env.geometry import haversine_m
from route_env.io import read_jsonl, write_jsonl


def point_for_checkpoint(task, label):
    checkpoint = task["turn_checkpoints"][label]
    return {"lat": checkpoint["lat"], "lon": checkpoint["lon"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-len", type=int, default=12)
    args = parser.parse_args()
    records = []
    for task in read_jsonl(args.tasks):
        dest = {"lat": task["destination"]["lat"], "lon": task["destination"]["lon"]}
        current = {"lat": task["origin"]["lat"], "lon": task["origin"]["lon"]}
        unused = set(task["turn_checkpoints"])
        turns = []
        current_distance = haversine_m(current, dest)
        for _ in range(args.max_len):
            if not unused:
                break
            best = min(unused, key=lambda label: haversine_m(point_for_checkpoint(task, label), dest))
            best_point = point_for_checkpoint(task, best)
            best_distance = haversine_m(best_point, dest)
            if best_distance >= current_distance:
                break
            turns.append(best)
            unused.remove(best)
            current = best_point
            current_distance = best_distance
        records.append(
            {
                "task_id": task["task_id"],
                "prediction": {"turns": turns, "confidence": 0.3, "reason": "Greedy checkpoint-distance baseline."},
            }
        )
    write_jsonl(args.out, records)
    print(f"wrote {len(records)} greedy predictions to {args.out}")


if __name__ == "__main__":
    main()
