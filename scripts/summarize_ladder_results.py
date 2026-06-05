#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from route_env.io import iter_jsonl


DIFFICULTIES = ("easy", "medium", "hard")


def _metric(record: dict[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    return record.get("metrics", {}).get(key)


def _finite(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, int | float) and math.isfinite(float(value))]


def summarize_file(path: Path) -> dict[str, Any]:
    records = list(iter_jsonl(path))
    scores = _finite([_metric(record, "score") for record in records])
    valid_route_values = [_metric(record, "valid_route") for record in records]
    valid_choice_values = [_metric(record, "valid_choice") for record in records]
    exact_best_values = [_metric(record, "exact_best") for record in records]
    summary = {
        "path": str(path),
        "records": len(records),
        "score_mean": statistics.mean(scores) if scores else 0.0,
        "score_median": statistics.median(scores) if scores else 0.0,
        "score_min": min(scores) if scores else 0.0,
        "score_max": max(scores) if scores else 0.0,
    }
    if any(value is not None for value in valid_route_values):
        summary["valid_route"] = sum(bool(value) for value in valid_route_values) / len(records) if records else 0.0
    if any(value is not None for value in valid_choice_values):
        summary["valid_choice"] = sum(bool(value) for value in valid_choice_values) / len(records) if records else 0.0
    if any(value is not None for value in exact_best_values):
        summary["exact_best"] = sum(bool(value) for value in exact_best_values) / len(records) if records else 0.0
    if any("regret" in record for record in records):
        regrets = _finite([record.get("regret") for record in records])
        summary["regret_mean"] = statistics.mean(regrets) if regrets else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark ladder result files.")
    parser.add_argument("--root", default="data/experiments/singapore_benchmark_ladder_1500")
    parser.add_argument("--pattern", required=True, help="Relative glob below each difficulty, e.g. results/foo.jsonl")
    parser.add_argument("--out")
    args = parser.parse_args()

    root = Path(args.root)
    by_difficulty = {}
    all_records = 0
    weighted_score = 0.0
    for difficulty in DIFFICULTIES:
        matches = sorted((root / difficulty).glob(args.pattern))
        if not matches:
            continue
        summary = summarize_file(matches[0])
        by_difficulty[difficulty] = summary
        all_records += int(summary["records"])
        weighted_score += float(summary["score_mean"]) * int(summary["records"])

    cumulative = {
        "records": all_records,
        "score_mean": weighted_score / all_records if all_records else 0.0,
        "by_difficulty": by_difficulty,
    }
    print(json.dumps(cumulative, indent=2, sort_keys=True))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(cumulative, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
