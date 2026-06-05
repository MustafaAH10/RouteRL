#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from route_env.io import read_jsonl, write_jsonl
from route_env.map_env import (
    CuaLiteMapEnv,
    all_visible_label_actions_for_task,
    empty_actions_for_task,
    load_task,
    oracle_actions_for_task,
)


DEFAULT_TASKS = "data/experiments/long_8_25km_route_strip_probe/tasks.jsonl"


def _tool(action: dict[str, Any]) -> str:
    tool = action.get("tool", action.get("action"))
    return tool if isinstance(tool, str) else "<missing>"


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.3f}"
    if value is None:
        return "-"
    return str(value)


def actions_for_mode(mode: str, task: dict[str, Any]) -> list[dict[str, Any]]:
    if mode == "oracle":
        return oracle_actions_for_task(task)
    if mode == "all-labels":
        return all_visible_label_actions_for_task(task)
    if mode == "empty":
        return empty_actions_for_task(task)
    raise ValueError(f"unsupported mode: {mode}")


def run_trajectory(
    task: dict[str, Any],
    *,
    mode: str,
    max_actions: int,
    keep_trace: bool,
) -> dict[str, Any]:
    actions = actions_for_mode(mode, task)
    env = CuaLiteMapEnv(task, max_actions=max_actions)
    trace = []
    for action in actions:
        outcome = env.step(action)
        if keep_trace:
            trace.append(
                {
                    "action": action,
                    "step_reward": outcome.step_reward,
                    "error": outcome.error,
                    "observation": outcome.observation,
                }
            )
        if outcome.done:
            break
    if not env.done:
        outcome = env.step({"tool": "finish"})
        if keep_trace:
            trace.append(
                {
                    "action": {"tool": "finish"},
                    "step_reward": outcome.step_reward,
                    "error": outcome.error,
                    "observation": outcome.observation,
                }
            )
    counts = Counter(_tool(action) for action in actions)
    metrics = env.last_metrics or {}
    return {
        "mode": mode,
        "task_id": task["task_id"],
        "action_count": env.action_count,
        "requested_action_count": len(actions),
        "tool_counts": dict(sorted(counts.items())),
        "prediction": env.prediction(),
        "metrics": metrics,
        "trace": trace,
    }


def print_task_header(task: dict[str, Any]) -> None:
    if task.get("task_type") == "route_strip":
        segment_counts = {
            segment["segment_id"]: {
                "visible_labels": len(segment.get("turn_checkpoints", {})),
                "gold_turns": len(segment.get("oracle", {}).get("gold_turn_route", [])),
                "image": segment.get("images", {}).get("map"),
            }
            for segment in task.get("segments", [])
        }
        payload = {
            "task_id": task["task_id"],
            "task_type": "route_strip",
            "overview": task.get("images", {}).get("overview"),
            "segments": segment_counts,
        }
    else:
        payload = {
            "task_id": task["task_id"],
            "task_type": "flat",
            "image": task.get("images", {}).get("map"),
            "visible_labels": len(task.get("turn_checkpoints", {})),
            "gold_turns": len(task.get("oracle", {}).get("gold_turn_route", [])),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))


def print_summary(results: list[dict[str, Any]]) -> None:
    columns = [
        ("mode", "mode"),
        ("action_count", "actions"),
        ("mark", "marks"),
        ("score", "score"),
        ("valid_schema", "schema"),
        ("valid_route", "route"),
        ("checkpoint_reward", "ckpt"),
        ("num_predicted_turns", "pred"),
        ("num_gold_turns", "gold"),
        ("length_ratio", "len_ratio"),
    ]
    rows = []
    for result in results:
        metrics = result["metrics"]
        tools = result["tool_counts"]
        row = {
            "mode": result["mode"],
            "action_count": result["action_count"],
            "mark": tools.get("mark", 0),
            **metrics,
        }
        rows.append(row)

    widths = {
        key: max(len(label), *(len(_fmt(row.get(key))) for row in rows))
        for key, label in columns
    }
    print(" ".join(label.rjust(widths[key]) for key, label in columns))
    print(" ".join("-" * widths[key] for key, _ in columns))
    for row in rows:
        print(" ".join(_fmt(row.get(key)).rjust(widths[key]) for key, _ in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic CUA-lite RouteRL map-tool demos.")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--task-id")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--mode", choices=["compare", "empty", "all-labels", "oracle"], default="compare")
    parser.add_argument("--max-actions", type=int, default=256)
    parser.add_argument("--write-trace")
    parser.add_argument("--print-prediction", action="store_true")
    args = parser.parse_args()

    task = load_task(args.tasks, task_id=args.task_id, task_index=args.task_index)
    print_task_header(task)

    modes = ["empty", "all-labels", "oracle"] if args.mode == "compare" else [args.mode]
    keep_trace = bool(args.write_trace)
    results = [
        run_trajectory(task, mode=mode, max_actions=args.max_actions, keep_trace=keep_trace)
        for mode in modes
    ]
    print()
    print_summary(results)

    if args.print_prediction:
        print()
        for result in results:
            print(f"{result['mode']} prediction:")
            print(json.dumps(result["prediction"], indent=2, sort_keys=True))

    if args.write_trace:
        records = [
            {
                "mode": result["mode"],
                "task_id": result["task_id"],
                "action_count": result["action_count"],
                "tool_counts": result["tool_counts"],
                "prediction": result["prediction"],
                "metrics": result["metrics"],
                "trace": result["trace"],
            }
            for result in results
        ]
        write_jsonl(args.write_trace, records)
        print()
        print(f"wrote trace to {Path(args.write_trace)}")


if __name__ == "__main__":
    main()
