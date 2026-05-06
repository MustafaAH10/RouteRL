#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from route_env.io import read_jsonl, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]

FLAT_TEMPLATE = """You are a routing model.

Given the driving map image, return JSON only:
{"turns":["T1","T2"]}

Most visible labels are distractors. Do not list every label unless the route
really passes through every one.

Allowed turn checkpoint labels: {allowed}
"""


def strip_prompt(task: dict[str, Any]) -> str:
    lines = ["Images are ordered as:", "Image 0: overview corridor map."]
    allowed = []
    for index, segment in enumerate(task["segments"], start=1):
        lines.append(f"Image {index}: segment {segment['segment_id']} local driving map.")
        allowed.append(f"{segment['segment_id']}: " + ", ".join(segment["turn_checkpoints"].keys()))
    return (
        (REPO_ROOT / "prompts/drive_route_strip_prompt.txt").read_text(encoding="utf-8")
        + "\n\n"
        + "\n".join(lines)
        + "\n\nAllowed checkpoints by segment:\n"
        + "\n".join(allowed)
    )


def record_for_task(task: dict[str, Any], split: str) -> dict[str, Any]:
    if task.get("task_type") == "route_strip":
        images = [task["images"]["overview"]] + list(task["images"]["segments"])
        prompt = strip_prompt(task)
    else:
        images = [task["images"]["map"]]
        prompt = FLAT_TEMPLATE.format(allowed=", ".join(task["turn_checkpoints"].keys()))
    return {
        "data_source": "routerl",
        "task_id": task["task_id"],
        "task_type": task.get("task_type", "flat"),
        "split": split,
        "images": images,
        "prompt": prompt,
        "reward_model": {"style": "rule", "ground_truth": task["task_id"]},
        "extra_info": {
            "city": task.get("city"),
            "distance_m": task.get("oracle", {}).get("distance_m"),
            "tasks_jsonl": None,
        },
    }


def write_parquet(path: str, records: list[dict[str, Any]]) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--format", choices=["auto", "jsonl", "parquet"], default="auto")
    args = parser.parse_args()

    records = [record_for_task(task, args.split) for task in read_jsonl(args.tasks)]
    for record in records:
        record["extra_info"]["tasks_jsonl"] = args.tasks
    wrote_parquet = False
    if args.format in {"auto", "parquet"} and args.out.endswith(".parquet"):
        wrote_parquet = write_parquet(args.out, records)
        if args.format == "parquet" and not wrote_parquet:
            raise RuntimeError("pyarrow is required for parquet output")
    if not wrote_parquet:
        write_jsonl(args.out, records)
    print(f"wrote {len(records)} VeRL dataset records to {args.out}")


if __name__ == "__main__":
    main()
