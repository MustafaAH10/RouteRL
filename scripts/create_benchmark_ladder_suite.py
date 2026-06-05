#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from route_env.io import iter_jsonl, write_jsonl


DIFFICULTIES = {
    "easy": "short_500m_2km",
    "medium": "mid_2_6km",
    "hard": "long_8_25km",
}


def prepare_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    for child in ("maps", "rerank_maps", "rollouts", "predictions", "results", "viewports"):
        (path / child).mkdir(parents=True, exist_ok=True)


def selected_tasks(source_path: Path, count: int, out_dir: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for task in iter_jsonl(source_path):
        if len(tasks) >= count:
            break
        task = dict(task)
        task["difficulty"] = out_dir.name
        task["images"] = dict(task.get("images", {}))
        task["images"]["map"] = str(out_dir / "maps" / task["task_id"] / "map.png")
        tasks.append(task)
    if len(tasks) < count:
        raise RuntimeError(f"{source_path} only had {len(tasks)} tasks, needed {count}")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 500/500/500 benchmark ladder suite from Singapore tasks.")
    parser.add_argument("--source-root", default="data/experiments/singapore_trace_choice_4k")
    parser.add_argument("--out-root", default="data/experiments/singapore_benchmark_ladder_1500")
    parser.add_argument("--count-per-difficulty", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_root": str(source_root),
        "count_per_difficulty": args.count_per_difficulty,
        "difficulties": {},
    }

    for difficulty, source_band in DIFFICULTIES.items():
        out_dir = out_root / difficulty
        prepare_dir(out_dir, overwrite=args.overwrite)
        source_path = source_root / source_band / "tasks.jsonl"
        tasks = selected_tasks(source_path, args.count_per_difficulty, out_dir)
        write_jsonl(out_dir / "tasks.jsonl", tasks)
        metadata["difficulties"][difficulty] = {
            "source_band": source_band,
            "source_path": str(source_path),
            "tasks_path": str(out_dir / "tasks.jsonl"),
            "count": len(tasks),
        }
        print(f"{difficulty}: wrote {len(tasks)} tasks to {out_dir / 'tasks.jsonl'}")

    with (out_root / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote metadata to {out_root / 'metadata.json'}")


if __name__ == "__main__":
    main()
