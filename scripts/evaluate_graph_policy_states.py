#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from route_env.io import iter_jsonl, write_jsonl


PolicyFn = Callable[[dict[str, Any], random.Random], str | None]


def _candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(record.get("candidates", []))


def _candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate["candidate_id"])


def _finite_number(value: Any, default: float) -> float:
    return float(value) if isinstance(value, int | float) and math.isfinite(float(value)) else default


def choose_oracle(record: dict[str, Any], _rng: random.Random) -> str | None:
    return record.get("target", {}).get("candidate_id")


def choose_first(record: dict[str, Any], _rng: random.Random) -> str | None:
    candidates = _candidates(record)
    return _candidate_id(candidates[0]) if candidates else None


def choose_random(record: dict[str, Any], rng: random.Random) -> str | None:
    candidates = _candidates(record)
    return _candidate_id(rng.choice(candidates)) if candidates else None


def choose_shortest_edge(record: dict[str, Any], _rng: random.Random) -> str | None:
    candidates = _candidates(record)
    if not candidates:
        return None
    return _candidate_id(min(candidates, key=lambda item: _finite_number(item.get("edge_length_m"), math.inf)))


def choose_max_progress(record: dict[str, Any], _rng: random.Random) -> str | None:
    candidates = _candidates(record)
    if not candidates:
        return None
    return _candidate_id(max(candidates, key=lambda item: _finite_number(item.get("progress_m"), -math.inf)))


def choose_closest_to_goal(record: dict[str, Any], _rng: random.Random) -> str | None:
    candidates = _candidates(record)
    if not candidates:
        return None
    return _candidate_id(min(candidates, key=lambda item: _finite_number(item.get("straight_to_goal_m"), math.inf)))


def choose_progress_per_meter(record: dict[str, Any], _rng: random.Random) -> str | None:
    candidates = _candidates(record)
    if not candidates:
        return None

    def score(candidate: dict[str, Any]) -> tuple[float, float]:
        progress = _finite_number(candidate.get("progress_m"), -math.inf)
        length = max(1.0, _finite_number(candidate.get("edge_length_m"), math.inf))
        return progress / length, progress

    return _candidate_id(max(candidates, key=score))


def choose_nonvisited_closest_goal(record: dict[str, Any], _rng: random.Random) -> str | None:
    candidates = _candidates(record)
    if not candidates:
        return None

    def score(candidate: dict[str, Any]) -> tuple[int, float, float]:
        visited_penalty = 1 if candidate.get("visited") else 0
        return (
            visited_penalty,
            _finite_number(candidate.get("straight_to_goal_m"), math.inf),
            _finite_number(candidate.get("edge_length_m"), math.inf),
        )

    return _candidate_id(min(candidates, key=score))


POLICIES: dict[str, PolicyFn] = {
    "first": choose_first,
    "random": choose_random,
    "shortest_edge": choose_shortest_edge,
    "max_progress": choose_max_progress,
    "closest_to_goal": choose_closest_to_goal,
    "progress_per_meter": choose_progress_per_meter,
    "nonvisited_closest_goal": choose_nonvisited_closest_goal,
    "oracle": choose_oracle,
}


def evaluate_record(record: dict[str, Any], policy: str, rng: random.Random) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    candidates = _candidates(record)
    selected_id = POLICIES[policy](record, rng)
    target_id = record.get("target", {}).get("candidate_id")
    by_id = {_candidate_id(candidate): candidate for candidate in candidates}
    return {
        "task_id": record.get("task_id"),
        "state_id": record.get("state_id"),
        "split": record.get("split"),
        "policy": policy,
        "selected_candidate_id": selected_id,
        "target_candidate_id": target_id,
        "correct": selected_id == target_id,
        "candidate_count": len(candidates),
        "branching": len(candidates) > 1,
        "selected_candidate": by_id.get(str(selected_id)),
        "target_candidate": by_id.get(str(target_id)),
        "current_node": record.get("current_node"),
        "goal_node": record.get("goal_node"),
        "route_so_far_m": record.get("route_so_far_m"),
        "remaining_teacher_steps": record.get("remaining_teacher_steps"),
    }


def _bool_mean(values: list[bool]) -> float:
    return mean([1.0 if value else 0.0 for value in values]) if values else 0.0


def _number_mean(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if isinstance(value, int | float) and math.isfinite(float(value))]
    return mean(finite) if finite else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_policy.setdefault(record["policy"], []).append(record)

    def candidate_metric(item: dict[str, Any], key: str, field: str) -> Any:
        candidate = item.get(key)
        return candidate.get(field) if isinstance(candidate, dict) else None

    summary: dict[str, Any] = {}
    for policy, items in sorted(by_policy.items()):
        branching = [item for item in items if item["branching"]]
        selected_progress = [candidate_metric(item, "selected_candidate", "progress_m") for item in items]
        target_progress = [candidate_metric(item, "target_candidate", "progress_m") for item in items]
        selected_goal_distance = [candidate_metric(item, "selected_candidate", "straight_to_goal_m") for item in items]
        target_goal_distance = [candidate_metric(item, "target_candidate", "straight_to_goal_m") for item in items]
        summary[policy] = {
            "count": len(items),
            "accuracy": _bool_mean([item["correct"] for item in items]),
            "branching_count": len(branching),
            "branching_accuracy": _bool_mean([item["correct"] for item in branching]),
            "single_candidate_rate": _bool_mean([not item["branching"] for item in items]),
            "mean_candidate_count": _number_mean([item["candidate_count"] for item in items]),
            "mean_selected_progress_m": _number_mean(selected_progress),
            "mean_target_progress_m": _number_mean(target_progress),
            "mean_selected_straight_to_goal_m": _number_mean(selected_goal_distance),
            "mean_target_straight_to_goal_m": _number_mean(target_goal_distance),
        }
    return summary


def dataset_summary(states: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = {str(state.get("task_id")) for state in states}
    candidate_counts = [len(_candidates(state)) for state in states]
    branching_counts = [count > 1 for count in candidate_counts]
    return {
        "states": len(states),
        "tasks": len(task_ids),
        "single_candidate_rate": _bool_mean([not value for value in branching_counts]),
        "branching_rate": _bool_mean(branching_counts),
        "mean_candidate_count": _number_mean(candidate_counts),
        "max_candidate_count": max(candidate_counts) if candidate_counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument("--policies", nargs="+", default=sorted(POLICIES))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    states = list(iter_jsonl(args.states))
    records = []
    for state in states:
        for policy in args.policies:
            records.append(evaluate_record(state, policy, rng))

    write_jsonl(Path(args.out), records)
    summary = {
        "states_path": args.states,
        "dataset": dataset_summary(states),
        "policies": summarize(records),
    }
    summary_path = Path(args.summary_out) if args.summary_out else Path(args.out).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(records)} graph-policy evaluation record(s) to {args.out}")
    print(f"wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
