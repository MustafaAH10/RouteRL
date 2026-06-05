#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from route_env.io import iter_jsonl, write_jsonl


def _prediction_route_id(record: dict[str, Any]) -> str | None:
    prediction = record.get("prediction", record)
    if isinstance(prediction, dict):
        value = prediction.get("route_id", prediction.get("choice", prediction.get("selected_route")))
        return str(value).upper() if value is not None else None
    if isinstance(prediction, str):
        return prediction.upper()
    return None


def evaluate_record(task: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    route_id = _prediction_route_id(prediction)
    candidates = {candidate["route_id"]: candidate for candidate in task["candidates"]}
    selected = candidates.get(route_id or "")
    valid_choice = selected is not None
    best_route_id = task["best_route_id"]
    best = candidates[best_route_id]
    selected_metrics = selected["hidden_metrics"] if selected else {}
    best_score = float(best["hidden_metrics"]["score"])
    selected_score = float(selected_metrics.get("score", 0.0)) if selected else 0.0
    regret = max(0.0, best_score - selected_score)
    return {
        "task_id": task["task_id"],
        "difficulty": task.get("difficulty"),
        "valid_schema": isinstance(prediction.get("prediction", prediction), dict),
        "valid_choice": valid_choice,
        "route_id": route_id,
        "best_route_id": best_route_id,
        "exact_best": valid_choice and route_id == best_route_id,
        "oracle_selected": bool(selected and selected.get("is_oracle")),
        "score": selected_score,
        "best_score": best_score,
        "regret": regret,
        "length_ratio": selected_metrics.get("length_ratio") if selected else math.inf,
        "mean_route_distance_m": selected_metrics.get("mean_route_distance_m") if selected else math.inf,
        "checkpoint_reward": selected_metrics.get("checkpoint_reward") if selected else 0.0,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(result["score"]) for result in results]
    regrets = [float(result["regret"]) for result in results]
    return {
        "records": len(results),
        "valid_choice": sum(bool(result["valid_choice"]) for result in results) / len(results) if results else 0.0,
        "exact_best": sum(bool(result["exact_best"]) for result in results) / len(results) if results else 0.0,
        "oracle_selected": sum(bool(result["oracle_selected"]) for result in results) / len(results) if results else 0.0,
        "score_mean": statistics.mean(scores) if scores else 0.0,
        "score_median": statistics.median(scores) if scores else 0.0,
        "score_min": min(scores) if scores else 0.0,
        "score_max": max(scores) if scores else 0.0,
        "regret_mean": statistics.mean(regrets) if regrets else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one-shot route-reranking predictions.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tasks = {task["task_id"]: task for task in iter_jsonl(args.tasks)}
    results = []
    for prediction in iter_jsonl(args.predictions):
        task = tasks[prediction["task_id"]]
        results.append(evaluate_record(task, prediction))
    write_jsonl(args.out, results)
    summary = summarize(results)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
