#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from route_env.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = []
    for task in read_jsonl(args.tasks):
        records.append(
            {
                "id": task["task_id"],
                "image": task["images"]["map"],
                "messages": [
                    {"role": "user", "content": task["prompt"]},
                    {"role": "assistant", "content": json.dumps({"turns": task["oracle"]["gold_turn_route"]})},
                ],
            }
        )
    write_jsonl(args.out, records)
    print(f"wrote {len(records)} SFT records to {args.out}")


if __name__ == "__main__":
    main()
