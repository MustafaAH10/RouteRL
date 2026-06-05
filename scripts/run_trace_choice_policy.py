#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from route_env.io import iter_jsonl, write_jsonl
from route_env.trace_choice_env import TraceChoiceEnv


def select_action(env: TraceChoiceEnv, policy: str, rng: random.Random) -> dict[str, Any]:
    if env.current_node == env.destination_node:
        return {"tool": "finish"}
    if policy == "oracle":
        candidate_id = env.oracle_candidate_id() or env.shortest_candidate_id()
    elif policy == "shortest":
        candidate_id = env.shortest_candidate_id()
    elif policy == "heading":
        candidate_id = env.heading_candidate_id()
    elif policy == "random":
        candidates = env.candidates()
        candidate_id = rng.choice(candidates).candidate_id if candidates else None
    else:
        raise ValueError(f"unknown policy: {policy}")
    if not candidate_id:
        return {"tool": "finish"}
    return {"tool": "choose", "candidate_id": candidate_id}


def selected_candidate(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    if action.get("tool") != "choose":
        return None
    candidate_id = str(action.get("candidate_id", "")).upper()
    for candidate in observation["candidates"]:
        if candidate["candidate_id"] == candidate_id:
            return candidate
    return None


def run_task(
    task: dict[str, Any],
    *,
    policy: str,
    max_steps: int,
    trace_length_m: float,
    max_candidates: int,
    render_dir: str | None,
    render_context: str,
    compact: bool,
    rng: random.Random,
) -> dict[str, Any]:
    env = TraceChoiceEnv(
        task,
        max_steps=max_steps,
        trace_length_m=trace_length_m,
        max_candidates=max_candidates,
        render_dir=render_dir,
        render_context=render_context,
    )
    trace = []
    for _ in range(max_steps):
        observation = env.observe()
        action = select_action(env, policy, rng)
        candidate = selected_candidate(observation, action)
        outcome = env.step(action)
        if not compact:
            trace.append(
                {
                    "action": action,
                    "pre_observation": observation,
                    "selected_candidate": candidate,
                    "step_reward": outcome.step_reward,
                    "error": outcome.error,
                    "observation": outcome.observation,
                }
            )
        if outcome.done:
            break
    if not env.done:
        observation = env.observe()
        outcome = env.step({"tool": "finish"})
        if not compact:
            trace.append(
                {
                    "action": {"tool": "finish"},
                    "pre_observation": observation,
                    "selected_candidate": None,
                    "step_reward": outcome.step_reward,
                    "error": outcome.error,
                    "observation": outcome.observation,
                }
            )
    record = {
        "mode": f"trace-choice:{policy}",
        "task_id": task["task_id"],
        "policy": policy,
        "action_count": env.action_count,
        "prediction": env.scored_prediction(),
        "observed_prediction": env.prediction(),
        "route_nodes": env.route_nodes,
        "metrics": env.last_metrics or env.direct_metrics(),
    }
    if compact:
        record["metrics"].pop("agent_geometry", None)
    else:
        record["trace"] = trace
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run graph-native trace-choice routing policies.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy", choices=["oracle", "shortest", "heading", "random"], default="shortest")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--trace-length-m", type=float, default=350.0)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--render-dir")
    parser.add_argument("--render-context", choices=["local", "local_overview"], default="local")
    parser.add_argument("--compact", action="store_true", help="Do not write per-step trace observations.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final output path, not every record.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    written = 0

    def records() -> Any:
        nonlocal written
        stop = args.task_index + args.num_tasks
        for index, task in enumerate(iter_jsonl(args.tasks)):
            if index < args.task_index:
                continue
            if index >= stop:
                break
            record = run_task(
                task,
                policy=args.policy,
                max_steps=args.max_steps,
                trace_length_m=args.trace_length_m,
                max_candidates=args.max_candidates,
                render_dir=args.render_dir,
                render_context=args.render_context,
                compact=args.compact,
                rng=rng,
            )
            written += 1
            if not args.quiet:
                metrics = record["metrics"]
                print(
                    "{} score={:.3f} valid_route={} actions={} turns={}/{} length_ratio={:.3f}".format(
                        record["task_id"],
                        float(metrics.get("score", 0.0)),
                        metrics.get("valid_route"),
                        record["action_count"],
                        metrics.get("num_predicted_turns"),
                        metrics.get("num_gold_turns"),
                        float(metrics.get("length_ratio", 0.0)),
                    )
                )
            yield record

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, records())
    if not written:
        raise ValueError(f"no tasks selected from {args.tasks}")
    print(f"wrote {written} trace-choice records to {args.out}")


if __name__ == "__main__":
    main()
