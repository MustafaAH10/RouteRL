#!/usr/bin/env python
from __future__ import annotations

import argparse
import random

from route_env.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-len", type=int, default=8)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    records = []
    for task in read_jsonl(args.tasks):
        labels = list(task["turn_checkpoints"])
        k = rng.randint(0, min(args.max_len, len(labels))) if labels else 0
        records.append(
            {
                "task_id": task["task_id"],
                "prediction": {
                    "turns": rng.sample(labels, k),
                    "confidence": 0.1,
                    "reason": "Random turn-checkpoint baseline.",
                },
            }
        )
    write_jsonl(args.out, records)
    print(f"wrote {len(records)} random predictions to {args.out}")


if __name__ == "__main__":
    main()
