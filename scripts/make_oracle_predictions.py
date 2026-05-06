#!/usr/bin/env python
from __future__ import annotations

import argparse

from route_env.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = []
    for task in read_jsonl(args.tasks):
        if task.get("task_type") == "route_strip":
            prediction = {
                "segments": [
                    {
                        "segment_id": segment["segment_id"],
                        "turns": segment["oracle"]["gold_turn_route"],
                    }
                    for segment in task["segments"]
                ],
            }
        else:
            prediction = {
                "turns": task["oracle"]["gold_turn_route"],
            }
        records.append(
            {
                "task_id": task["task_id"],
                "prediction": prediction,
            }
        )
    write_jsonl(args.out, records)
    print(f"wrote {len(records)} oracle predictions to {args.out}")


if __name__ == "__main__":
    main()
