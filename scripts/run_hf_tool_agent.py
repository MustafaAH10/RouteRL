#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from route_env.hf_client import generate_route_prediction, load_vision_model
from route_env.io import read_jsonl, write_jsonl
from route_env.map_env import CuaLiteMapEnv, load_task


DEFAULT_TASKS = "data/experiments/long_8_25km_route_strip_probe/tasks.jsonl"

SYSTEM_PROMPTS = {
    "default": """You are a map-using routing agent operating a deterministic RouteRL tool environment.

Your job is to inspect the current map image and choose one next tool action.
You are not writing the final route directly. You must build it with tool calls.

Critical behavior:
- Return exactly one JSON object, never an array and never prose.
- Use only labels listed in visible_labels for the current observation.
- Never mark a label that is already present in prediction_so_far.
- Do not repeat the previous action if it did not change prediction_so_far.
- Prefer a small number of useful route checkpoints over many labels.
- If labels are crowded or the next turn is unclear, use pan/zoom tools before marking.
- Do not finish until the current prediction has enough marks to plausibly describe the A-to-B route.
""",
    "path_prefix": """You are a map-using routing agent operating a deterministic RouteRL tool environment.

Treat prediction_so_far as an ordered route prefix from blue A toward red B.
At each step, choose the next checkpoint after the last marked checkpoint, not
just any nearby label.

Critical behavior:
- Return exactly one JSON object, never an array and never prose.
- Use only labels listed in visible_labels for the current observation.
- Never mark a label that is already present in prediction_so_far.
- Do not repeat the previous action if it did not change prediction_so_far.
- Follow connected roads and one-way arrows visually.
- If labels are crowded or the next route continuation is unclear, zoom in or pan before marking.
- Avoid jumping to a cluster of labels near B until the route has actually reached that cluster.
- Prefer sparse useful checkpoints over side-road or parallel-road labels.
- Do not finish until the ordered prefix plausibly connects A to B.
""",
    "sparse_planner": """You are a careful visual route planner using deterministic map tools.

Before choosing an action, mentally trace the shortest-looking legal driving
path from blue A to red B. Mark only checkpoints on that path. Most labels are
distractors and many nearby labels are not part of the route.

Critical behavior:
- Return exactly one JSON object, never an array and never prose.
- Use only labels listed in visible_labels for the current observation.
- Never mark a label that is already present in prediction_so_far.
- Mark at most one new checkpoint per step.
- If labels overlap the road or the route continuation is unclear, zoom in or pan before marking.
- If several labels are in a dense cluster, choose the one that continues the route, not all of them.
- If the marked route has enough checkpoints and adding another label would be a guess, finish.
""",
    "inspect_first": """You are a map-inspection agent using deterministic viewport tools.

Your first job is to reduce visual clutter before marking turns. If many labels
are visible, inspect a smaller area with zoom or pan. Only mark a checkpoint
after the current viewport is local enough to reason about the road geometry.

Critical behavior:
- Return exactly one JSON object, never an array and never prose.
- Use only labels listed in visible_labels for the current observation.
- If more than 12 labels are visible, do not mark; use zoom_in, zoom_to_label, or pan.
- Never mark a label that is already present in prediction_so_far.
- Mark at most one new checkpoint per step.
- Do not finish until the marked route plausibly connects A to B.
""",
    "frontier": """You are a local map-use routing agent.

You operate a moving viewport. Treat the current prediction as an ordered route
frontier from blue A toward red B. Your job is to advance that frontier through
local observations, not to collect every nearby label.

Operating loop:
- Start from the local viewport near A.
- Mark one visible checkpoint only if it is clearly on the driving path from the current frontier toward B.
- After marking, keep the viewport near the frontier with center_on_last_mark or pan_toward_destination.
- If the next route continuation is unclear, inspect with pan, zoom_in, zoom_out, or zoom_to_label.
- Finish only when the ordered marks plausibly connect A to B.

Critical behavior:
- Return exactly one JSON object, never an array and never prose.
- Use only labels listed in visible_labels for the current observation.
- Never mark a label that is already present in prediction_so_far.
- Mark at most one new checkpoint per step.
- Do not mark several labels from the same dense cluster unless the road path clearly passes through them in order.
""",
    "road_follow": """You are a local visual road-following agent.

You operate a moving viewport. Your route is an ordered path from blue A to red B.
The destination direction is only a weak hint: roads bend, ramps split, and a
straight-line pan toward B can jump onto the wrong road.

Operating loop:
- Use frontier_candidates as the ordered local candidate list.
- At the start, mark the first frontier_candidate unless the image clearly shows it is off-route.
- After marking local start labels, inspect the road continuation with pan or zoom instead of harvesting a dense cluster.
- When there are no markable labels, never return an error. Pan along the visible road continuation.
- Prefer pan west/east/north/south/northwest/northeast/southwest/southeast over pan_toward_destination unless B is visible.
- Finish only when the ordered marks plausibly connect A to B.

Critical behavior:
- Return exactly one JSON object, never an array and never prose.
- Use only labels listed in frontier_candidates for mark actions.
- Never mark a label that is already present in prediction_so_far.
- Mark at most one new checkpoint per step.
""",
}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def action_from_model(parsed: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return {"tool": "observe"}
    action = parsed.get("action") if isinstance(parsed.get("action"), dict) else parsed
    if "mark" in action and "tool" not in action and "action" not in action:
        mark = action["mark"]
        action = {"tool": "mark", "turn": mark.get("turn", mark.get("label")) if isinstance(mark, dict) else mark}
    elif "unmark" in action and "tool" not in action and "action" not in action:
        unmark = action["unmark"]
        action = {
            "tool": "unmark",
            "turn": unmark.get("turn", unmark.get("label")) if isinstance(unmark, dict) else unmark,
        }
    elif "open_segment" in action and "tool" not in action and "action" not in action:
        action = {"tool": "open_segment", "segment_id": action["open_segment"]}
    elif "pan" in action and "tool" not in action and "action" not in action:
        pan = action["pan"]
        action = {"tool": "pan", **pan} if isinstance(pan, dict) else {"tool": "pan", "direction": pan}
    elif "zoom_to_label" in action and "tool" not in action and "action" not in action:
        zoom = action["zoom_to_label"]
        action = {"tool": "zoom_to_label", "turn": zoom.get("turn", zoom.get("label"))} if isinstance(zoom, dict) else {
            "tool": "zoom_to_label",
            "turn": zoom,
        }
    elif "center_on_last_mark" in action and "tool" not in action and "action" not in action:
        action = {"tool": "center_on_last_mark"}
    elif "pan_toward_destination" in action and "tool" not in action and "action" not in action:
        action = {"tool": "pan_toward_destination"}
    tool = action.get("tool", action.get("action"))
    if not isinstance(tool, str):
        return {"tool": "observe"}
    normalized: dict[str, Any] = {"tool": tool}
    if isinstance(action.get("segment_id"), str):
        normalized["segment_id"] = action["segment_id"]
    elif observation.get("view", {}).get("segment_id") and tool in {"mark", "unmark", "clear_segment"}:
        normalized["segment_id"] = observation["view"]["segment_id"]
    if isinstance(action.get("turn"), str):
        normalized["turn"] = action["turn"]
    elif isinstance(action.get("label"), str):
        normalized["turn"] = action["label"]
    if isinstance(action.get("direction"), str):
        normalized["direction"] = action["direction"]
    for key in ("dx", "dy"):
        if isinstance(action.get(key), int | float):
            normalized[key] = float(action[key])
    return normalized


def build_action_prompt(observation: dict[str, Any], *, strategy: str) -> str:
    view = observation["view"]
    prediction_so_far = observation["prediction_so_far"]
    markable_labels = observation.get("markable_labels", observation["visible_labels"])
    frontier_candidates = observation.get("frontier_candidates", markable_labels)
    if observation["task_type"] == "route_strip":
        if view["kind"] == "overview":
            allowed_tools = {
                "open_segment": {"segment_id": "one visible segment id"},
                "finish": "only after useful turns have been marked for the route",
            }
            instruction = "Open a segment that still needs inspection. Usually start with S01."
            visible = observation["visible_segments"]
            already_marked: list[str] = []
        else:
            segment_id = view["segment_id"]
            segment_predictions = prediction_so_far.get("segments", [])
            current_segment = next(
                (segment for segment in segment_predictions if segment.get("segment_id") == segment_id),
                {"turns": []},
            )
            already_marked = list(current_segment.get("turns", []))
            allowed_tools = {
                "mark": {
                    "segment_id": segment_id,
                    "turn": "one label from markable_labels",
                },
                "unmark": {
                    "segment_id": segment_id,
                    "turn": "one already marked label if you need to revise",
                },
                "open_segment": {"segment_id": "another visible segment id"},
                "open_overview": {},
                "pan": {"direction": "north|south|east|west|northwest|northeast|southwest|southeast"},
                "zoom_in": {},
                "zoom_out": {},
                "zoom_to_label": {"turn": "one visible label"},
                "center_on_start": {},
                "center_on_end": {},
                "center_on_last_mark": {},
                "pan_toward_destination": {},
                "finish": "only after enough useful route turns have been marked",
            }
            instruction = (
                "Inspect the segment image. If a visible label appears to lie on the A-to-B driving route and "
                "is not already marked, mark one new useful label. If this segment already has enough marks, "
                "open the next segment. Do not repeat an already-marked label."
            )
            visible = frontier_candidates
    else:
        already_marked = list(prediction_so_far.get("turns", []))
        allowed_tools = {
            "mark": {"turn": "one label from markable_labels"},
            "unmark": {"turn": "one already marked label if you need to revise"},
            "pan": {"direction": "north|south|east|west|northwest|northeast|southwest|southeast"},
            "zoom_in": {},
            "zoom_out": {},
            "zoom_to_label": {"turn": "one visible label"},
            "center_on_start": {},
            "center_on_end": {},
            "center_on_last_mark": {},
            "pan_toward_destination": {},
            "finish": "only after enough useful route turns have been marked",
        }
        instruction = (
            "Inspect the map image. If a visible label lies on the A-to-B driving route and is not already "
            "marked, mark one new useful label. Finish only when the sparse route looks complete."
        )
        visible = frontier_candidates

    if strategy == "path_prefix":
        if already_marked:
            instruction += (
                f" The current route prefix ends at {already_marked[-1]}; choose the next checkpoint after "
                "that point along the road path toward B. Do not choose another label merely because it is "
                "near the destination cluster."
            )
        else:
            instruction += " Choose the first checkpoint after A along the road path toward B."
    elif strategy == "sparse_planner":
        instruction += (
            " Prefer finishing over adding a low-confidence label. Dense label clusters are dangerous: "
            "only mark a cluster label if the route physically passes through it."
        )
    elif strategy == "inspect_first":
        if len(observation["visible_labels"]) > 12:
            instruction = (
                f"There are {len(observation['visible_labels'])} visible labels, which is too cluttered. Do not mark a turn yet. "
                "Use zoom_in, zoom_to_label, or pan to inspect a smaller area first."
            )
        else:
            instruction += (
                " The viewport is local enough; mark one useful unmarked checkpoint if the route clearly "
                "passes through it, otherwise pan or zoom again."
            )
    elif strategy == "frontier":
        if not markable_labels:
            instruction = (
                "No unmarked checkpoint labels are currently markable. Use center_on_last_mark, "
                "pan_toward_destination, zoom_out, or pan to inspect the next local route area."
            )
        elif already_marked:
            instruction = (
                f"The route frontier is the last marked checkpoint {already_marked[-1]}. Choose one action "
                "that advances from that frontier toward B: mark exactly one next visible checkpoint if the "
                "road clearly continues through it, otherwise use center_on_last_mark or pan_toward_destination "
                "to inspect the next local area. Do not jump backward to earlier labels."
            )
        else:
            instruction = (
                "The route has no checkpoint yet. frontier_candidates are sorted nearest-first from blue A. "
                "Mark the first frontier_candidate if it lies on the driving path toward red B. If the first "
                "checkpoint is unclear, use zoom_in or pan_toward_destination instead of guessing."
            )
    elif strategy == "road_follow":
        if not markable_labels:
            if len(already_marked) >= 5:
                instruction = (
                    "No unmarked checkpoint is markable in this viewport, and the route already has several "
                    "ordered marks. Return finish unless a specific pan direction is clearly needed to reach B."
                )
            else:
                instruction = (
                    "No unmarked checkpoint is markable in this viewport. Return a pan action, not observe and "
                    "not an error. Pan along the visible road continuation from the current route frontier; use "
                    "one of north, south, east, west, northwest, northeast, southwest, southeast."
                )
        elif already_marked:
            instruction = (
                f"The route frontier is {already_marked[-1]}. frontier_candidates are sorted by distance "
                "from that frontier, but distance alone can be wrong. Mark one candidate only if the visible "
                "road continuation clearly passes through it; otherwise pan along the road continuation. "
                "Do not harvest multiple labels from a side-road cluster."
            )
        else:
            instruction = (
                "The route has no checkpoint yet. frontier_candidates are sorted nearest-first from blue A. "
                "Mark the first frontier_candidate if it lies on the road leaving A toward B."
            )

    return f"""Current observation:
{compact_json({
    "task_id": observation["task_id"],
    "task_type": observation["task_type"],
    "view": observation["view"],
    "visible_labels": observation["visible_labels"],
    "markable_labels": observation.get("markable_labels", observation["visible_labels"]),
    "frontier_candidates": observation.get("frontier_candidates", observation.get("markable_labels", [])),
    "visible_segments": observation["visible_segments"],
    "prediction_so_far": observation["prediction_so_far"],
    "remaining_actions": observation["remaining_actions"],
    "last_error": observation["last_error"],
})}

Visible choices now:
{compact_json(visible)}

Already marked in this view:
{compact_json(already_marked)}

Allowed tool schemas:
{compact_json(allowed_tools)}

Next-step instruction:
{instruction}
"""


def view_signature(observation: dict[str, Any]) -> tuple[float, ...]:
    bbox = observation.get("view", {}).get("bbox") or []
    return tuple(round(float(value), 7) for value in bbox)


def has_marked_turns(observation: dict[str, Any]) -> bool:
    return count_marked_turns(observation) > 0


def count_marked_turns(observation: dict[str, Any]) -> int:
    prediction = observation.get("prediction_so_far", {})
    if "turns" in prediction:
        return len(prediction.get("turns") or [])
    return sum(len(segment.get("turns") or []) for segment in prediction.get("segments", []))


def movement_repair_action(observation: dict[str, Any]) -> dict[str, Any]:
    if has_marked_turns(observation):
        return {"tool": "pan_toward_destination"}
    return {"tool": "zoom_out"}


def repair_action(
    action: dict[str, Any],
    observation: dict[str, Any],
    *,
    enabled: bool,
    marks_in_current_view: int,
    max_marks_per_view: int | None,
    finish_on_empty_markable_after: int | None,
) -> tuple[dict[str, Any], str | None]:
    if not enabled:
        return action, None

    tool = action.get("tool", action.get("action"))
    markable = list(observation.get("markable_labels", observation.get("visible_labels", [])))
    frontier_candidates = list(observation.get("frontier_candidates", markable))
    marked_count = count_marked_turns(observation)

    if not markable and finish_on_empty_markable_after is not None and marked_count >= finish_on_empty_markable_after:
        if tool in {"observe", "pan", "zoom_in", "zoom_out", "zoom_to_label", "pan_toward_destination"}:
            return {"tool": "finish"}, "finish_on_empty_markable"

    if tool == "mark":
        turn = action.get("turn", action.get("label"))
        if max_marks_per_view is not None and marks_in_current_view >= max_marks_per_view:
            return movement_repair_action(observation), "max_marks_per_view"
        if turn not in markable:
            if frontier_candidates:
                return {"tool": "mark", "turn": frontier_candidates[0]}, "mark_not_markable"
            return movement_repair_action(observation), "no_markable_labels"
        if not has_marked_turns(observation) and frontier_candidates and turn != frontier_candidates[0]:
            return {"tool": "mark", "turn": frontier_candidates[0]}, "first_frontier_candidate"
        return action, None

    if tool in {"observe", ""} and not markable:
        return movement_repair_action(observation), "empty_or_non_tool_response"

    return action, None


def rollout_record(task: dict[str, Any], env: CuaLiteMapEnv, trace: list[dict[str, Any]], strategy: str) -> dict[str, Any]:
    return {
        "mode": f"hf-tool-agent:{strategy}",
        "task_id": task["task_id"],
        "strategy": strategy,
        "action_count": env.action_count,
        "prediction": env.prediction(),
        "metrics": env.last_metrics or {},
        "trace": trace,
    }


def run_rollout(
    task: dict[str, Any],
    *,
    model: Any,
    processor: Any,
    max_actions: int,
    max_new_tokens: int,
    strategy: str,
    viewport_dir: str | None = None,
    initial_view: str = "full",
    viewport_scale: float = 0.42,
    max_visible_labels: int | None = None,
    enforce_prefix_validity: bool = False,
    repair_actions: bool = False,
    max_marks_per_view: int | None = None,
    finish_on_empty_markable_after: int | None = None,
    out_path: str | None = None,
) -> dict[str, Any]:
    env = CuaLiteMapEnv(
        task,
        max_actions=max_actions,
        viewport_dir=viewport_dir,
        initial_view=initial_view,
        viewport_scale=viewport_scale,
        max_visible_labels=max_visible_labels,
        enforce_prefix_validity=enforce_prefix_validity,
    )
    trace = []
    marks_by_view: dict[tuple[float, ...], int] = {}
    for _ in range(max_actions):
        observation = env.observe()
        signature = view_signature(observation)
        image = observation.get("view", {}).get("image")
        prompt = build_action_prompt(observation, strategy=strategy)
        try:
            parsed, raw = generate_route_prediction(
                image_path=image,
                prompt=prompt,
                model=model,
                processor=processor,
                max_new_tokens=max_new_tokens,
                system_prompt=SYSTEM_PROMPTS[strategy],
            )
            action = action_from_model(parsed, observation)
            model_error = None
        except Exception as exc:
            parsed = {}
            raw = ""
            action = {"tool": "observe"}
            model_error = str(exc)
        action_before_repair = dict(action)
        action, action_repair = repair_action(
            action,
            observation,
            enabled=repair_actions,
            marks_in_current_view=marks_by_view.get(signature, 0),
            max_marks_per_view=max_marks_per_view,
            finish_on_empty_markable_after=finish_on_empty_markable_after,
        )
        outcome = env.step(action)
        if action.get("tool") == "mark" and outcome.error is None:
            marks_by_view[signature] = marks_by_view.get(signature, 0) + 1
        trace.append(
            {
                "action": action,
                "action_before_repair": action_before_repair,
                "action_repair": action_repair,
                "model_prediction": parsed,
                "raw_response": raw,
                "model_error": model_error,
                "step_reward": outcome.step_reward,
                "error": outcome.error,
                "observation": outcome.observation,
            }
        )
        if out_path:
            write_jsonl(out_path, [rollout_record(task, env, trace, strategy)])
        if outcome.done:
            break
    if not env.done:
        outcome = env.step({"tool": "finish"})
        trace.append(
            {
                "action": {"tool": "finish"},
                "model_prediction": {"tool": "finish"},
                "raw_response": "",
                "model_error": None,
                "step_reward": outcome.step_reward,
                "error": outcome.error,
                "observation": outcome.observation,
            }
        )
        if out_path:
            write_jsonl(out_path, [rollout_record(task, env, trace, strategy)])
    return rollout_record(task, env, trace, strategy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Hugging Face VLM as a one-action-at-a-time CUA-lite map agent.")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--task-id")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--strategy", choices=sorted(SYSTEM_PROMPTS), default="default")
    parser.add_argument("--viewport-dir", help="Render dynamic CUA viewport images here for pan/zoom rollouts.")
    parser.add_argument("--initial-view", choices=["full", "start", "end"], default="full")
    parser.add_argument("--viewport-scale", type=float, default=0.42)
    parser.add_argument("--max-visible-labels", type=int)
    parser.add_argument("--enforce-prefix-validity", action="store_true")
    parser.add_argument("--repair-actions", action="store_true")
    parser.add_argument("--max-marks-per-view", type=int)
    parser.add_argument("--finish-on-empty-markable-after", type=int)
    args = parser.parse_args()

    if args.task_id:
        tasks = [load_task(args.tasks, task_id=args.task_id)]
    elif args.num_tasks == 1:
        tasks = [load_task(args.tasks, task_index=args.task_index)]
    else:
        all_tasks = read_jsonl(args.tasks)
        stop = min(len(all_tasks), args.task_index + args.num_tasks)
        tasks = all_tasks[args.task_index : stop]
        if not tasks:
            raise ValueError(f"no tasks selected from index {args.task_index}")

    model, processor = load_vision_model(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    records = []
    stream_single_trace = len(tasks) == 1
    for task in tasks:
        record = run_rollout(
            task,
            model=model,
            processor=processor,
            max_actions=args.max_actions,
            max_new_tokens=args.max_new_tokens,
            strategy=args.strategy,
            viewport_dir=args.viewport_dir,
            initial_view=args.initial_view,
            viewport_scale=args.viewport_scale,
            max_visible_labels=args.max_visible_labels,
            enforce_prefix_validity=args.enforce_prefix_validity,
            repair_actions=args.repair_actions,
            max_marks_per_view=args.max_marks_per_view,
            finish_on_empty_markable_after=args.finish_on_empty_markable_after,
            out_path=args.out if stream_single_trace else None,
        )
        records.append(record)
        if not stream_single_trace:
            write_jsonl(args.out, records)

    write_jsonl(args.out, records)
    print(f"wrote HF tool trace to {args.out}")
    for record in records:
        metrics = record["metrics"]
        print(
            "{} score={:.3f} valid_schema={} valid_route={} actions={} turns={}/{}".format(
                record["task_id"],
                float(metrics.get("score", 0.0)),
                metrics.get("valid_schema"),
                metrics.get("valid_route"),
                record["action_count"],
                metrics.get("num_predicted_turns"),
                metrics.get("num_gold_turns"),
            )
        )


if __name__ == "__main__":
    main()
