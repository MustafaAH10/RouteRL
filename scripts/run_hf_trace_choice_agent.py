#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from route_env.hf_client import generate_route_prediction, load_vision_model
from route_env.io import iter_jsonl, write_jsonl
from route_env.trace_choice_env import TraceChoiceEnv


SYSTEM_PROMPT = """You are a visual road-routing agent.

You are shown a local map viewport. The current route frontier is marked F.
Colored candidate road continuations are labeled C1, C2, C3, etc. Every
candidate is a legal directed road trace. Choose the candidate that best
continues the route from F toward destination B.

Some observations have two panels: the left panel is the local road detail and
the right panel is an overview showing A, F, B, the route so far, candidate
endpoints, and dashed candidate-to-B guides. Use both panels when present.

If any candidate has "ends_at_destination":true, choose that candidate
immediately. If any candidate has "reachable_to_destination":false while a
reachable candidate exists, do not choose the unreachable candidate.

Return exactly one JSON object:
{"tool":"choose","candidate_id":"C1"}

If the frontier has reached B or no useful candidate remains, return:
{"tool":"finish"}

Never invent checkpoint labels. Never output prose.
"""


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def build_prompt(observation: dict[str, Any], *, prompt_strategy: str) -> str:
    candidates = []
    route_distance = float(observation["route_so_far"].get("distance_m", 0.0) or 0.0)
    for candidate in observation["candidates"]:
        record = {
            "candidate_id": candidate["candidate_id"],
            "length_m": candidate["length_m"],
            "checkpoint_labels": candidate["checkpoint_labels"],
            "reachable_to_destination": candidate["reachable_to_destination"],
            "ends_at_destination": candidate["ends_at_destination"],
        }
        if prompt_strategy == "planner_hint":
            remaining = candidate.get("remaining_distance_m")
            record["remaining_distance_m"] = remaining
            record["estimated_total_route_m"] = (
                round(route_distance + float(candidate["length_m"]) + float(remaining), 1)
                if remaining is not None
                else None
            )
        candidates.append(record)
    if prompt_strategy == "planner_hint":
        ranked = sorted(
            [candidate for candidate in candidates if candidate.get("estimated_total_route_m") is not None],
            key=lambda candidate: (float(candidate["estimated_total_route_m"]), str(candidate["candidate_id"])),
        )
        for rank, candidate in enumerate(ranked, start=1):
            candidate["planner_rank"] = rank
    hint = ""
    if prompt_strategy == "planner_hint":
        hint = (
            "\nPlanner hint: choose the reachable candidate with planner_rank 1. "
            "planner_rank is based on estimated_total_route_m, a graph lookahead, not a checkpoint label. "
            "Use the image to confirm the candidate id and direction.\n"
        )
    return f"""Current route state:
{compact_json({
    "task_id": observation["task_id"],
    "view_context": observation["view"].get("context"),
    "current_node": observation["view"]["current_node"],
    "route_so_far": observation["route_so_far"],
    "remaining_steps": observation["remaining_steps"],
    "last_error": observation["last_error"],
})}

Candidate choices:
{compact_json(candidates)}
{hint}
Choose one visible candidate id. JSON only.
"""


def selected_candidate(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any] | None:
    if action.get("tool") != "choose":
        return None
    candidate_id = str(action.get("candidate_id", "")).upper()
    for candidate in observation["candidates"]:
        if candidate["candidate_id"] == candidate_id:
            return candidate
    return None


def repair_action(action: dict[str, Any], observation: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    if action.get("tool") != "choose":
        return action, None
    candidate = selected_candidate(observation, action)
    if candidate is not None:
        return action, None
    if not observation["candidates"]:
        return {"tool": "finish"}, "repaired invalid candidate to finish because no candidates were available"
    fallback = observation["candidates"][0]["candidate_id"]
    return (
        {"tool": "choose", "candidate_id": fallback},
        f"repaired invalid candidate {action.get('candidate_id')} to {fallback}",
    )


def parse_action(parsed: Any, observation: dict[str, Any]) -> dict[str, Any]:
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return {"tool": "finish"} if not observation["candidates"] else {"tool": "choose", "candidate_id": "C1"}
    action = parsed.get("action") if isinstance(parsed.get("action"), dict) else parsed
    if "choose" in action and "tool" not in action:
        choice = action["choose"]
        if isinstance(choice, dict):
            return {"tool": "choose", "candidate_id": str(choice.get("candidate_id", choice.get("choice", ""))).upper()}
        return {"tool": "choose", "candidate_id": str(choice).upper()}
    if "candidate_id" in action:
        return {"tool": "choose", "candidate_id": str(action["candidate_id"]).upper()}
    if "choice" in action:
        return {"tool": "choose", "candidate_id": str(action["choice"]).upper()}
    tool = str(action.get("tool", action.get("action", ""))).lower()
    if tool == "finish" or action.get("finish") is True:
        return {"tool": "finish"}
    if tool == "choose":
        return {"tool": "choose", "candidate_id": str(action.get("candidate_id", action.get("choice", ""))).upper()}
    return {"tool": "finish"} if not observation["candidates"] else {"tool": "choose", "candidate_id": "C1"}


def run_task(
    task: dict[str, Any],
    *,
    model: Any,
    processor: Any,
    max_steps: int,
    trace_length_m: float,
    max_candidates: int,
    max_new_tokens: int,
    render_dir: str,
    render_context: str,
    auto_unique: bool,
    repair_invalid_actions: bool,
    prompt_strategy: str,
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
        image = observation["view"]["image"]
        prompt = build_prompt(observation, prompt_strategy=prompt_strategy)
        repair_note = None
        if auto_unique and len(observation["candidates"]) == 1:
            parsed = {"tool": "choose", "candidate_id": observation["candidates"][0]["candidate_id"], "source": "auto_unique"}
            raw = json.dumps(parsed, separators=(",", ":"))
            action = {"tool": "choose", "candidate_id": observation["candidates"][0]["candidate_id"]}
            model_error = None
        else:
            try:
                parsed, raw = generate_route_prediction(
                    image_path=image,
                    prompt=prompt,
                    model=model,
                    processor=processor,
                    max_new_tokens=max_new_tokens,
                    system_prompt=SYSTEM_PROMPT,
                )
                action = parse_action(parsed, observation)
                model_error = None
            except Exception as exc:
                parsed = {}
                raw = ""
                action = {"tool": "finish"} if not observation["candidates"] else {"tool": "choose", "candidate_id": "C1"}
                model_error = str(exc)
        if repair_invalid_actions:
            action, repair_note = repair_action(action, observation)
        candidate = selected_candidate(observation, action)
        outcome = env.step(action)
        trace.append(
            {
                "action": action,
                "model_prediction": parsed,
                "raw_response": raw,
                "model_error": model_error,
                "repair_note": repair_note,
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
        trace.append(
            {
                "action": {"tool": "finish"},
                "model_prediction": {"tool": "finish"},
                "raw_response": "",
                "model_error": None,
                "repair_note": None,
                "pre_observation": observation,
                "selected_candidate": None,
                "step_reward": outcome.step_reward,
                "error": outcome.error,
                "observation": outcome.observation,
            }
        )
    return {
        "mode": "hf-trace-choice",
        "task_id": task["task_id"],
        "action_count": env.action_count,
        "prediction": env.scored_prediction(),
        "observed_prediction": env.prediction(),
        "route_nodes": env.route_nodes,
        "metrics": env.last_metrics or env.direct_metrics(),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a HF VLM on the graph-native trace-choice sandbox.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--trace-length-m", type=float, default=350.0)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--prompt-strategy", choices=["visual", "planner_hint"], default="visual")
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--render-context", choices=["local", "local_overview"], default="local")
    parser.add_argument("--auto-unique", action="store_true", help="Skip the VLM when only one candidate is available.")
    parser.add_argument(
        "--no-repair-invalid-actions",
        action="store_true",
        help="Let invalid candidate ids reach the environment instead of repairing to a valid fallback.",
    )
    args = parser.parse_args()

    model, processor = load_vision_model(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )

    written = 0
    summaries = []

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
                model=model,
                processor=processor,
                max_steps=args.max_steps,
                trace_length_m=args.trace_length_m,
                max_candidates=args.max_candidates,
                max_new_tokens=args.max_new_tokens,
                render_dir=args.render_dir,
                render_context=args.render_context,
                auto_unique=args.auto_unique,
                repair_invalid_actions=not args.no_repair_invalid_actions,
                prompt_strategy=args.prompt_strategy,
            )
            written += 1
            metrics = record["metrics"]
            summaries.append(
                {
                    "task_id": record["task_id"],
                    "score": float(metrics.get("score", 0.0)),
                    "valid_route": metrics.get("valid_route"),
                    "action_count": record["action_count"],
                    "num_predicted_turns": metrics.get("num_predicted_turns"),
                    "num_gold_turns": metrics.get("num_gold_turns"),
                    "length_ratio": float(metrics.get("length_ratio", 0.0)),
                }
            )
            yield record

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, records())
    if not written:
        raise ValueError(f"no tasks selected from {args.tasks}")
    print(f"wrote {written} HF trace-choice records to {args.out}")
    for summary in summaries:
        print(
            "{} score={:.3f} valid_route={} actions={} turns={}/{} length_ratio={:.3f}".format(
                summary["task_id"],
                summary["score"],
                summary["valid_route"],
                summary["action_count"],
                summary["num_predicted_turns"],
                summary["num_gold_turns"],
                summary["length_ratio"],
            )
        )


if __name__ == "__main__":
    main()
