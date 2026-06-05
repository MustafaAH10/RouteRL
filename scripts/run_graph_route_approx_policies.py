#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

from route_env.graph_route_approx import POLICIES
from route_env.io import iter_jsonl, write_jsonl
from route_env.render import render_debug_overlay


def finite_mean(values: list[float]) -> float | None:
    finite = [value for value in values if isinstance(value, int | float) and math.isfinite(value)]
    return mean(finite) if finite else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_policy.setdefault(record["policy"], []).append(record)
    out: dict[str, Any] = {}
    for policy, items in sorted(by_policy.items()):
        metrics = [item["metrics"] for item in items]
        scores = [metric["score"] for metric in metrics]
        out[policy] = {
            "count": len(items),
            "mean_score": mean(scores) if scores else 0.0,
            "success_at_0_75": mean([score >= 0.75 for score in scores]) if scores else 0.0,
            "valid_route_rate": mean([metric["valid_route"] for metric in metrics]) if metrics else 0.0,
            "mean_length_ratio": finite_mean([metric["length_ratio"] for metric in metrics]),
            "mean_route_distance_m": finite_mean([metric["mean_route_distance_m"] for metric in metrics]),
            "mean_checkpoint_coverage": mean([metric["checkpoint_coverage"] for metric in metrics]) if metrics else 0.0,
            "mean_num_route_nodes": mean([metric["num_route_nodes"] for metric in metrics]) if metrics else 0.0,
        }
    return out


def selected_tasks(path: Path, *, limit: int | None, start_index: int) -> list[dict[str, Any]]:
    tasks = []
    for index, task in enumerate(iter_jsonl(path)):
        if index < start_index:
            continue
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def run_policy(policy: str, task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if policy == "local_greedy":
        result = POLICIES[policy](task, max_steps=args.max_steps)
    elif policy == "heuristic_beam":
        result = POLICIES[policy](task, beam_width=args.beam_width, max_expansions=args.max_expansions)
    elif policy == "hill_climb_beam":
        result = POLICIES[policy](
            task,
            lookahead_depth=args.lookahead_depth,
            branch_width=args.branch_width,
            max_steps=args.max_steps,
        )
    else:
        raise ValueError(f"unknown policy: {policy}")
    return {
        "task_id": task["task_id"],
        "policy": result.policy,
        "prediction": {"turns": result.turns},
        "route_nodes": result.route_nodes,
        "metrics": result.metrics,
        "diagnostics": result.diagnostics,
    }


def render_examples(records: list[dict[str, Any]], tasks: dict[str, dict[str, Any]], out_dir: Path, max_examples: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for record in records:
        if rendered >= max_examples:
            break
        task = tasks[record["task_id"]]
        prediction = {"task_id": record["task_id"], "prediction": record["prediction"]}
        render_debug_overlay(task, prediction, record["metrics"], out_dir / f"{record['task_id']}__{record['policy']}.png")
        rendered += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument("--policies", nargs="+", default=sorted(POLICIES))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--max-expansions", type=int, default=2000)
    parser.add_argument("--lookahead-depth", type=int, default=4)
    parser.add_argument("--branch-width", type=int, default=6)
    parser.add_argument("--render-dir")
    parser.add_argument("--render-examples", type=int, default=0)
    args = parser.parse_args()

    tasks_list = selected_tasks(Path(args.tasks), limit=args.limit, start_index=args.start_index)
    records = []
    for task in tqdm(tasks_list, desc="graph-route-approx"):
        for policy in args.policies:
            records.append(run_policy(policy, task, args))

    write_jsonl(args.out, records)
    summary = summarize(records)
    summary_path = Path(args.summary_out) if args.summary_out else Path(args.out).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(records)} records to {args.out}")
    print(f"wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.render_dir and args.render_examples:
        tasks = {task["task_id"]: task for task in tasks_list}
        render_examples(records, tasks, Path(args.render_dir), args.render_examples)
        print(f"rendered {args.render_examples} example overlays per record order to {args.render_dir}")


if __name__ == "__main__":
    main()
