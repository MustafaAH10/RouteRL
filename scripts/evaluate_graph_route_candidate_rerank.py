#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any

from route_env.io import iter_jsonl, write_jsonl


def choose_route(record: dict[str, Any], policy: str, rng: random.Random) -> str | None:
    candidates = record.get("candidates", [])
    if not candidates:
        return None
    if policy == "first":
        return candidates[0]["route_id"]
    if policy == "random":
        return rng.choice(candidates)["route_id"]
    if policy == "shortest_distance":
        valid = [candidate for candidate in candidates if isinstance(candidate.get("distance_m"), int | float)]
        return min(valid, key=lambda candidate: candidate["distance_m"])["route_id"] if valid else candidates[0]["route_id"]
    if policy == "fewest_nodes":
        return min(candidates, key=lambda candidate: candidate.get("node_count", 10**9))["route_id"]
    if policy == "fewest_turns":
        return min(candidates, key=lambda candidate: candidate.get("turn_count", 10**9))["route_id"]
    if policy == "oracle":
        return record.get("target", {}).get("route_id")
    raise ValueError(f"unknown policy: {policy}")


def metric_for(record: dict[str, Any], route_id: str | None) -> dict[str, Any]:
    hidden = record.get("hidden_metrics", {})
    if route_id is None or route_id not in hidden:
        return {
            "score": 0.0,
            "valid_route": False,
            "length_ratio": math.inf,
            "mean_route_distance_m": math.inf,
            "checkpoint_coverage": 0.0,
            "checkpoint_precision": 0.0,
            "checkpoint_order": 0.0,
        }
    return hidden[route_id]


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if isinstance(value, int | float) and math.isfinite(value)]
    return mean(finite) if finite else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_policy.setdefault(record["policy"], []).append(record)
    summary: dict[str, Any] = {}
    for policy, items in sorted(by_policy.items()):
        metrics = [item["selected_metrics"] for item in items]
        scores = [metric["score"] for metric in metrics]
        summary[policy] = {
            "count": len(items),
            "mean_score": mean(scores) if scores else 0.0,
            "exact_best": mean([item["selected_route_id"] == item["target_route_id"] for item in items]) if items else 0.0,
            "valid_route_rate": mean([metric["valid_route"] for metric in metrics]) if metrics else 0.0,
            "success_at_0_75": mean([score >= 0.75 for score in scores]) if scores else 0.0,
            "mean_length_ratio": finite_mean([metric.get("length_ratio", math.inf) for metric in metrics]),
            "mean_route_distance_m": finite_mean([metric.get("mean_route_distance_m", math.inf) for metric in metrics]),
            "mean_checkpoint_coverage": mean([metric.get("checkpoint_coverage", 0.0) for metric in metrics]) if metrics else 0.0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["first", "shortest_distance", "fewest_nodes", "fewest_turns", "random", "oracle"],
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = []
    for task in iter_jsonl(args.candidate_tasks):
        for policy in args.policies:
            selected = choose_route(task, policy, rng)
            target = task.get("target", {}).get("route_id")
            records.append(
                {
                    "task_id": task["task_id"],
                    "policy": policy,
                    "selected_route_id": selected,
                    "target_route_id": target,
                    "selected_metrics": metric_for(task, selected),
                    "candidate_count": task.get("candidate_count", len(task.get("candidates", []))),
                }
            )

    write_jsonl(Path(args.out), records)
    summary = summarize(records)
    summary_path = Path(args.summary_out) if args.summary_out else Path(args.out).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(records)} rerank evaluation record(s) to {args.out}")
    print(f"wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
