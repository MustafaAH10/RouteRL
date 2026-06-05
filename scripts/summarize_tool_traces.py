#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from route_env.io import iter_jsonl


def _score(record: dict[str, Any]) -> float:
    return float(record.get("metrics", {}).get("score", 0.0) or 0.0)


def _finite_metric(record: dict[str, Any], key: str) -> float | None:
    value = record.get("metrics", {}).get(key)
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    return None


def _direction_edges(task: dict[str, Any]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for edge in task.get("graph", {}).get("edges", []):
        edges.add((str(edge["u"]), str(edge["v"])))
    for segment in task.get("segments", []):
        for edge in segment.get("graph", {}).get("edges", []):
            edges.add((str(edge["u"]), str(edge["v"])))
    return edges


def _direction_audit(task: dict[str, Any], record: dict[str, Any]) -> tuple[int, int]:
    directed = _direction_edges(task)
    route = [str(node) for node in record.get("metrics", {}).get("agent_osm_route_expanded", [])]
    checked = 0
    missing = 0
    for u, v in zip(route, route[1:]):
        checked += 1
        if (u, v) not in directed:
            missing += 1
    return checked, missing


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CUA-lite/HF tool-agent trace JSONL records.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--worst", type=int, default=6)
    args = parser.parse_args()

    with Path(args.trace).open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise SystemExit("no records found")
    task_ids = {record["task_id"] for record in records}
    tasks = {task["task_id"]: task for task in iter_jsonl(args.tasks) if task.get("task_id") in task_ids}

    scores = [_score(record) for record in records]
    length_ratios = [value for record in records if (value := _finite_metric(record, "length_ratio")) is not None]
    mean_distances = [
        value for record in records if (value := _finite_metric(record, "mean_route_distance_m")) is not None
    ]
    valid_count = sum(bool(record.get("metrics", {}).get("valid_route")) for record in records)
    direction_checked = 0
    direction_missing = 0
    for record in records:
        task = tasks.get(record["task_id"])
        if not task:
            continue
        checked, missing = _direction_audit(task, record)
        direction_checked += checked
        direction_missing += missing

    print(f"records={len(records)}")
    print(f"valid_route={valid_count}/{len(records)}")
    print(
        "score mean={:.3f} median={:.3f} min={:.3f} max={:.3f}".format(
            _mean(scores),
            _median(scores),
            min(scores),
            max(scores),
        )
    )
    if length_ratios:
        print(
            "length_ratio mean={:.3f} median={:.3f} min={:.3f} max={:.3f}".format(
                _mean(length_ratios),
                _median(length_ratios),
                min(length_ratios),
                max(length_ratios),
            )
        )
    if mean_distances:
        print(
            "mean_route_distance_m mean={:.1f} median={:.1f} min={:.1f} max={:.1f}".format(
                _mean(mean_distances),
                _median(mean_distances),
                min(mean_distances),
                max(mean_distances),
            )
        )
    print(f"direction_check missing_edges={direction_missing}/{direction_checked}")

    print("\nworst_by_score")
    for record in sorted(records, key=_score)[: args.worst]:
        metrics = record["metrics"]
        print(
            "{} score={:.3f} valid={} turns={}/{} length_ratio={:.3f} mean_distance_m={:.1f} prediction={}".format(
                record["task_id"],
                float(metrics.get("score", 0.0) or 0.0),
                metrics.get("valid_route"),
                metrics.get("num_predicted_turns"),
                metrics.get("num_gold_turns"),
                float(metrics.get("length_ratio", 0.0) or 0.0),
                float(metrics.get("mean_route_distance_m", 0.0) or 0.0),
                json.dumps(record.get("prediction", {}), separators=(",", ":")),
            )
        )


if __name__ == "__main__":
    main()
