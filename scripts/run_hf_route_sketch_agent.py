#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from route_env.geometry import bbox_from_points, dedupe_consecutive, haversine_m, lonlat_to_latlon
from route_env.hf_client import extract_json, load_vision_model
from route_env.io import iter_jsonl, write_jsonl
from route_env.render import (
    _draw_checkpoint_labels,
    _draw_direction_arrow,
    _draw_panel_badge,
    _edge_style,
    _graph_edges,
    _line_xy,
    render_debug_overlay,
    render_task,
)
from route_env.verify import verify_prediction


SYSTEM_PROMPT = """You are RouteRL-MapUse, a careful vision-language routing agent.

You operate a deterministic CUA-lite map environment. Your job is to build an
ordered sparse checkpoint route from blue A to red B using tool calls.

Map semantics:
- Blue A is the start. Red B is the destination.
- Gray/black roads are drivable graph edges.
- Blue arrowheads on roads show known one-way direction. Never choose a route
  that visually goes against a one-way arrow.
- Black T-labels are candidate turn checkpoints. Their numbers are arbitrary.
  T17 is not necessarily after T16. Labels are sparse decision markers, not a
  complete path.
- Most T-labels are distractors. A good answer usually contains only the few
  checkpoints that lie on the route.
- Labels may cover roads. If a label hides a junction, inspect that area rather
  than guessing.

Available tools. Return exactly one JSON object using one tool:

1. Inspect a local area:
{"tool":"inspect","target":"T6","reason":"why this local crop is needed"}
{"tool":"inspect","target":"B2","reason":"why this grid cell is ambiguous"}

2. Replace the current draft route:
{"tool":"edit_route","turns":["T6","T10"],"reason":"why this ordered draft is plausible"}
The environment may reject this if you add too many new checkpoints at once.
Extend incrementally and preview between extensions.

3. Preview the current or proposed draft:
{"tool":"preview_route","turns":["T6","T10"],"reason":"what you want checked visually"}

4. Finish with the final route:
{"tool":"finish","turns":["T6","T10"],"reason":"why this is final"}

Operating policy:
- Work as a route planner, not a label collector.
- Think of the route as a prefix from A toward B. Every added checkpoint should
  be reachable from the previous frontier by following connected roads and
  respecting visible arrows.
- Do not jump to a far destination-side label until the road path has actually
  reached that area.
- Prefer inspecting before editing when the next junction is dense, occluded, or
  has one-way arrows.
- Use preview_route before finish whenever possible. The preview image shows
  only your draft route in orange; it does not show the hidden oracle route.
- Add only a few new checkpoints per edit. If you need a long route, build it
  through several inspect/edit/preview cycles.
- Finish only with the same draft that was just previewed as graph-valid.
- Preview validity means the graph can connect your chosen waypoints. It does
  not mean the route is optimal or correct. Still use the map image and arrows.
- A and B are implicit endpoints. Never include A or B in turns.
- Use only allowed T-labels. Do not invent road names, coordinates, or labels.

Output rules:
- JSON only. No markdown, no prose outside JSON.
- One tool call only.
- Keep reasons short but specific enough to audit your decision.
"""


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def finite_round(value: Any, digits: int = 1) -> float | None:
    if isinstance(value, int | float) and math.isfinite(value):
        return round(float(value), digits)
    return None


def label_number(label: str) -> int:
    try:
        return int(label.removeprefix("T"))
    except ValueError:
        return 10_000


def sorted_labels(task: dict[str, Any]) -> list[str]:
    return sorted(task["turn_checkpoints"], key=label_number)


def grid_cell_for_point(task: dict[str, Any], point: dict[str, float], cols: int = 4, rows: int = 4) -> str:
    west, south, east, north = task["task_bbox"]
    lon_frac = 0.0 if east == west else (point["lon"] - west) / (east - west)
    lat_frac = 0.0 if north == south else (north - point["lat"]) / (north - south)
    col = min(cols - 1, max(0, int(lon_frac * cols)))
    row = min(rows - 1, max(0, int(lat_frac * rows)))
    return f"{chr(ord('A') + col)}{row + 1}"


def point_for_grid_cell(task: dict[str, Any], cell: str, cols: int = 4, rows: int = 4) -> dict[str, float] | None:
    match = re.fullmatch(r"([A-Da-d])([1-4])", cell.strip())
    if not match:
        return None
    col = ord(match.group(1).upper()) - ord("A")
    row = int(match.group(2)) - 1
    west, south, east, north = task["task_bbox"]
    lon = west + (east - west) * (col + 0.5) / cols
    lat = north - (north - south) * (row + 0.5) / rows
    return {"lon": lon, "lat": lat}


def normalize_label(value: Any, allowed: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    if text in allowed:
        return text
    match = re.fullmatch(r"T0*([0-9]+)", text)
    if match:
        candidate = f"T{int(match.group(1))}"
        if candidate in allowed:
            return candidate
    return None


def sanitize_turns(values: Any, allowed: set[str]) -> tuple[list[str], list[Any]]:
    if not isinstance(values, list):
        return [], [values]
    accepted: list[str] = []
    rejected: list[Any] = []
    seen: set[str] = set()
    for value in values:
        label = normalize_label(value, allowed)
        if label is None:
            rejected.append(value)
            continue
        if label in seen:
            continue
        accepted.append(label)
        seen.add(label)
    return dedupe_consecutive(accepted), rejected


def normalize_action(parsed: Any, allowed: set[str], current_draft: list[str]) -> dict[str, Any]:
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return {"tool": "observe", "parse_note": "model did not return a JSON object"}
    action = parsed.get("action") if isinstance(parsed.get("action"), dict) else parsed
    if not isinstance(action, dict):
        return {"tool": "observe", "parse_note": "action was not an object"}
    tool = action.get("tool", action.get("action", "observe"))
    if not isinstance(tool, str):
        tool = "observe"
    tool = tool.strip().lower().replace("-", "_")
    aliases = {
        "preview": "preview_route",
        "route_preview": "preview_route",
        "draft": "edit_route",
        "edit": "edit_route",
        "set_route": "edit_route",
        "submit": "finish",
        "final": "finish",
    }
    tool = aliases.get(tool, tool)
    if tool not in {"inspect", "edit_route", "preview_route", "finish", "observe"}:
        tool = "observe"

    normalized: dict[str, Any] = {"tool": tool}
    if isinstance(action.get("reason"), str):
        normalized["reason"] = action["reason"].strip()
    target = action.get("target", action.get("label", action.get("turn", action.get("cell"))))
    if isinstance(target, str):
        normalized["target"] = target.strip().upper()

    raw_turns = action.get("turns", action.get("route", action.get("draft", action.get("prediction"))))
    if tool in {"edit_route", "preview_route", "finish"}:
        if raw_turns is None:
            turns = current_draft
            rejected: list[Any] = []
        else:
            turns, rejected = sanitize_turns(raw_turns, allowed)
        normalized["turns"] = turns
        normalized["rejected_turns"] = rejected
    return normalized


def point_for_target(task: dict[str, Any], target: str) -> dict[str, float] | None:
    target = target.strip().upper()
    if target == "A":
        return {"lon": task["origin"]["lon"], "lat": task["origin"]["lat"]}
    if target == "B":
        return {"lon": task["destination"]["lon"], "lat": task["destination"]["lat"]}
    if target in task["turn_checkpoints"]:
        point = task["turn_checkpoints"][target]
        return {"lon": point["lon"], "lat": point["lat"]}
    return point_for_grid_cell(task, target)


def clip_bbox(task: dict[str, Any], bbox: list[float]) -> list[float]:
    west, south, east, north = task["task_bbox"]
    clipped = [
        max(west, min(east, bbox[0])),
        max(south, min(north, bbox[1])),
        max(west, min(east, bbox[2])),
        max(south, min(north, bbox[3])),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return task["task_bbox"]
    return clipped


def render_inspection_crop(
    task: dict[str, Any],
    target: str,
    out_path: Path,
    *,
    margin_m: float,
) -> dict[str, Any]:
    point = point_for_target(task, target)
    if point is None:
        return {"ok": False, "target": target, "error": "unknown target"}
    bbox = clip_bbox(task, bbox_from_points([{"lon": point["lon"], "lat": point["lat"]}], margin_m=margin_m))
    crop_task = dict(task)
    crop_task["task_bbox"] = bbox
    crop_task["images"] = dict(task.get("images", {}))
    render_task(crop_task, out_path, show_labels=True, panel_label=f"Inspect {target}", show_grid=True)
    return {
        "ok": True,
        "target": target,
        "image": str(out_path),
        "bbox": bbox,
        "grid_cell": grid_cell_for_point(task, point),
    }


def render_agent_preview(
    task: dict[str, Any],
    turns: list[str],
    diagnostics: dict[str, Any],
    out_path: Path,
    *,
    panel_label: str = "Preview: agent route only",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = task["task_bbox"]
    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    ax.set_facecolor("#f8f8f4")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")

    for edge in _graph_edges(task):
        xs, ys = _line_xy(edge["geometry"])
        color, width, alpha = _edge_style(edge)
        ax.plot(xs, ys, color=color, linewidth=max(0.7, width * 0.8), alpha=alpha * 0.50, solid_capstyle="round", zorder=1)
        if edge.get("oneway"):
            _draw_direction_arrow(ax, edge["geometry"], alpha=0.42)

    if diagnostics.get("agent_geometry"):
        agent = [lonlat_to_latlon(point) for point in diagnostics["agent_geometry"]]
        ax.plot([p["lon"] for p in agent], [p["lat"] for p in agent], color="#f27a1a", linewidth=4.2, alpha=0.88, zorder=3)

    origin = task["origin"]
    dest = task["destination"]
    _draw_checkpoint_labels(
        fig,
        ax,
        task["turn_checkpoints"],
        fontsize=7.8,
        marker_size=36,
        label_alpha=0.88,
        occupied_points=[origin, dest],
    )
    for index, turn in enumerate(turns, start=1):
        point = task["turn_checkpoints"].get(turn)
        if not point:
            continue
        ax.scatter(point["lon"], point["lat"], s=112, facecolors="none", edgecolors="#f27a1a", linewidth=2.2, zorder=8)
        ax.text(
            point["lon"],
            point["lat"],
            str(index),
            ha="center",
            va="center",
            fontsize=7,
            weight="bold",
            color="#f27a1a",
            zorder=9,
            bbox={"boxstyle": "circle,pad=0.16", "facecolor": "white", "edgecolor": "#f27a1a", "alpha": 0.95},
        )

    ax.scatter(origin["lon"], origin["lat"], s=170, color="#1664d9", edgecolor="white", linewidth=1.4, zorder=7)
    ax.scatter(dest["lon"], dest["lat"], s=170, color="#d92525", edgecolor="white", linewidth=1.4, zorder=7)
    ax.text(origin["lon"], origin["lat"], "A", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=8)
    ax.text(dest["lon"], dest["lat"], "B", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=8)
    _draw_panel_badge(ax, panel_label)
    ax.set_title(f"Draft turns: {json.dumps(turns)}", fontsize=9)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def preview_feedback(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "valid_schema": bool(diagnostics.get("valid_schema")),
        "valid_route": bool(diagnostics.get("valid_route")),
        "unknown_turns": diagnostics.get("unknown_turns", []),
        "unknown_turn_count": diagnostics.get("unknown_turn_count", 0),
        "num_predicted_turns": diagnostics.get("num_predicted_turns", 0),
        "num_expanded_graph_nodes": diagnostics.get("num_expanded_nodes", 0),
        "agent_distance_m": finite_round(diagnostics.get("agent_distance_m"), 1),
        "max_segment_length_m": finite_round(diagnostics.get("max_segment_length_m"), 1),
        "note": "Preview feedback hides oracle route, score, length ratio, and gold checkpoints.",
    }


def render_and_score_preview(
    task: dict[str, Any],
    turns: list[str],
    out_path: Path,
    *,
    panel_label: str = "Preview: agent route only",
) -> tuple[dict[str, Any], dict[str, Any]]:
    prediction = {"task_id": task["task_id"], "prediction": {"turns": turns}}
    diagnostics = verify_prediction(task, prediction)
    render_agent_preview(task, turns, diagnostics, out_path, panel_label=panel_label)
    return diagnostics, preview_feedback(diagnostics)


def new_turns_since(current: list[str], proposed: list[str]) -> list[str]:
    current_set = set(current)
    return [turn for turn in proposed if turn not in current_set]


def incremental_guard(
    current: list[str],
    proposed: list[str],
    *,
    max_new_turns: int,
) -> dict[str, Any] | None:
    added = new_turns_since(current, proposed)
    if len(added) <= max_new_turns:
        return None
    return {
        "ok": False,
        "error": "too_many_new_turns_in_one_tool_call",
        "max_new_turns_per_edit": max_new_turns,
        "proposed_new_turns": added,
        "draft_unchanged": current,
        "note": "Add checkpoints incrementally. Inspect or preview before extending further.",
    }


def checkpoint_table(task: dict[str, Any], *, max_labels: int) -> str:
    origin = task["origin"]
    dest = task["destination"]
    rows = []
    for label in sorted_labels(task)[:max_labels]:
        point = task["turn_checkpoints"][label]
        rows.append(
            {
                "label": label,
                "cell": grid_cell_for_point(task, point),
                "from_A_m": round(haversine_m(origin, point), 1),
                "from_B_m": round(haversine_m(dest, point), 1),
            }
        )
    suffix = ""
    total = len(task["turn_checkpoints"])
    if total > max_labels:
        suffix = f"\n... {total - max_labels} labels omitted by --max-label-table."
    return "\n".join(compact_json(row) for row in rows) + suffix


def history_text(history: list[dict[str, Any]], *, max_entries: int) -> str:
    if not history:
        return "No tool calls yet."
    tail = history[-max_entries:]
    lines = []
    offset = len(history) - len(tail)
    for i, event in enumerate(tail, start=offset + 1):
        lines.append(f"{i}. action={compact_json(event.get('action'))} observation={compact_json(event.get('observation'))}")
    return "\n".join(lines)


def build_user_prompt(
    task: dict[str, Any],
    state: dict[str, Any],
    *,
    step: int,
    max_steps: int,
    max_label_table: int,
    history_entries: int,
    max_new_turns_per_edit: int,
) -> str:
    origin_cell = grid_cell_for_point(task, task["origin"])
    dest_cell = grid_cell_for_point(task, task["destination"])
    straight_line_m = haversine_m(task["origin"], task["destination"])
    last_preview = state.get("last_preview_feedback") or "No preview yet."
    last_inspection = state.get("last_inspection") or "No local inspection yet."
    return f"""Current task: {task['task_id']}
Step: {step} of {max_steps}

Objective:
Build a sparse ordered list of checkpoint labels that a driver should pass from
blue A to red B. You can inspect, edit a draft, preview it, or finish.

Non-oracle task context:
- Start A grid cell: {origin_cell}
- Destination B grid cell: {dest_cell}
- Straight-line A-to-B distance: {straight_line_m:.1f} m
- Allowed labels, with non-oracle grid/straight-line position hints:
{checkpoint_table(task, max_labels=max_label_table)}

Images in this request:
- Image 0 is always the full overview map with labels and road arrows.
- If present, Image 1 is the latest inspection crop around a label/grid cell.
- If present, Image 2 is the latest orange preview of your own draft route only.

Current draft route:
{compact_json(state['draft_turns'])}

Incremental editing rule:
- You may add at most {max_new_turns_per_edit} new checkpoint label(s) beyond
  the current draft in one edit_route or preview_route call.
- If the route needs more labels, extend it over multiple turns.
- finish must use the exact same draft that was already previewed as valid.

Latest inspection:
{compact_json(last_inspection)}

Latest preview feedback:
{compact_json(last_preview)}

Tool history:
{history_text(state['history'], max_entries=history_entries)}

Decision checklist before choosing a tool:
1. If the next route segment is hidden by labels or a dense junction, inspect a
   label/cell instead of guessing.
2. If your draft is empty, identify the first checkpoint after A along connected
   roads, not the closest arbitrary label.
3. If your draft has labels, extend from the last draft label toward B. Avoid
   jumping to a distant cluster unless the visible road actually connects there.
4. If you have a plausible draft and have not previewed it recently, preview it.
5. Finish only when the preview is graph-valid and the orange route visually
   follows the road direction from A to B.

Return exactly one JSON tool call now."""


def generate_action(
    image_paths: list[Path],
    captions: list[str],
    prompt: str,
    model: Any,
    processor: Any,
    *,
    max_new_tokens: int,
) -> tuple[dict[str, Any] | None, str, str | None]:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    content: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        if index < len(captions):
            content.append({"type": "text", "text": captions[index]})
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    target_device = getattr(model, "device", None)
    if target_device is not None:
        inputs = inputs.to(target_device)
    try:
        import torch

        context = torch.inference_mode()
    except ImportError:
        context = nullcontext()
    with context:
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    try:
        return extract_json(raw), raw, None
    except Exception as exc:
        return None, raw, str(exc)


def image_bundle(
    overview_path: Path,
    state: dict[str, Any],
) -> tuple[list[Path], list[str]]:
    paths = [overview_path]
    captions = [
        "Image 0: full overview map. Blue A=start, red B=destination, black T-labels=candidate checkpoints.",
    ]
    if state.get("last_inspection_image"):
        paths.append(Path(state["last_inspection_image"]))
        captions.append("Image 1: latest local inspection crop with grid and labels.")
    if state.get("last_preview_image"):
        paths.append(Path(state["last_preview_image"]))
        captions.append("Image 2: latest preview overlay. Orange is only your draft route, not the hidden oracle.")
    return paths, captions


def _load_font(name: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


def render_input_composite(
    image_paths: list[Path],
    captions: list[str],
    out_path: Path,
    *,
    max_panel_px: int = 920,
) -> None:
    title_font = _load_font("DejaVuSans-Bold.ttf", 30)
    body_font = _load_font("DejaVuSans.ttf", 22)
    panels = []
    for index, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((max_panel_px, max_panel_px), Image.Resampling.LANCZOS)
        header_h = 112
        panel = Image.new("RGB", (image.width, image.height + header_h), "white")
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, image.width, header_h), fill=(245, 247, 250))
        draw.text((16, 12), f"Image {index}: {path.name}", fill=(20, 27, 38), font=title_font)
        caption = captions[index] if index < len(captions) else ""
        caption = caption.replace(f"Image {index}: ", "")
        words = caption.split()
        lines = []
        line = ""
        max_chars = max(24, image.width // 13)
        for word in words:
            candidate = f"{line} {word}".strip()
            if len(candidate) > max_chars and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
        y = 54
        for line in lines[:2]:
            draw.text((16, y), line, fill=(65, 75, 88), font=body_font)
            y += 26
        panel.paste(image, (0, header_h))
        panels.append(panel)

    gap = 24
    width = sum(panel.width for panel in panels) + gap * max(0, len(panels) - 1)
    height = max(panel.height for panel in panels)
    composite = Image.new("RGB", (width, height), (232, 235, 239))
    x = 0
    for panel in panels:
        composite.paste(panel, (x, 0))
        x += panel.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path, quality=92)


def run_task(
    task: dict[str, Any],
    *,
    model: Any,
    processor: Any,
    render_root: Path,
    max_steps: int,
    max_new_tokens: int,
    inspect_margin_m: float,
    max_label_table: int,
    history_entries: int,
    max_new_turns_per_edit: int,
    send_composite_image: bool,
    composite_max_panel_px: int,
) -> dict[str, Any]:
    task_dir = render_root / task["task_id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    overview_path = task_dir / "step_00_overview.png"
    render_task(task, overview_path, show_labels=True, panel_label="Overview", show_grid=True)

    allowed = set(task["turn_checkpoints"])
    state: dict[str, Any] = {
        "draft_turns": [],
        "history": [],
        "last_inspection": None,
        "last_inspection_image": None,
        "last_preview_feedback": None,
        "last_preview_image": None,
        "last_preview_turns": None,
    }
    steps: list[dict[str, Any]] = []
    final_prediction = {"task_id": task["task_id"], "prediction": {"turns": []}}
    final_result: dict[str, Any] | None = None
    finish_reason = "max_steps"

    for step in range(1, max_steps + 1):
        prompt = build_user_prompt(
            task,
            state,
            step=step,
            max_steps=max_steps,
            max_label_table=max_label_table,
            history_entries=history_entries,
            max_new_turns_per_edit=max_new_turns_per_edit,
        )
        prompt_path = task_dir / f"step_{step:02d}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        image_paths, captions = image_bundle(overview_path, state)
        raw_path = task_dir / f"step_{step:02d}_raw_response.txt"
        if send_composite_image:
            source_image_paths = list(image_paths)
            source_image_captions = list(captions)
            composite_path = task_dir / f"step_{step:02d}_input_bundle.png"
            render_input_composite(
                source_image_paths,
                source_image_captions,
                composite_path,
                max_panel_px=composite_max_panel_px,
            )
            image_paths = [composite_path]
            captions = [
                "Image 0: exact composite input bundle. Read panels left to right; each panel has its own title and caption.",
            ]
        else:
            source_image_paths = []
            source_image_captions = []
        parsed, raw, parse_error = generate_action(
            image_paths,
            captions,
            prompt,
            model,
            processor,
            max_new_tokens=max_new_tokens,
        )
        raw_path.write_text(raw, encoding="utf-8")
        action = normalize_action(parsed, allowed, state["draft_turns"]) if parse_error is None else {
            "tool": "observe",
            "parse_error": parse_error,
        }
        observation: dict[str, Any]
        tool = action["tool"]

        if tool == "inspect":
            target = action.get("target")
            if not isinstance(target, str):
                observation = {"ok": False, "error": "inspect requires target label or grid cell"}
            else:
                inspect_path = task_dir / f"step_{step:02d}_inspect_{target}.png"
                observation = render_inspection_crop(task, target, inspect_path, margin_m=inspect_margin_m)
                if observation.get("ok"):
                    state["last_inspection"] = observation
                    state["last_inspection_image"] = str(inspect_path)
        elif tool == "edit_route":
            proposed_turns = list(action.get("turns", []))
            guard = incremental_guard(state["draft_turns"], proposed_turns, max_new_turns=max_new_turns_per_edit)
            if guard:
                observation = guard
            else:
                state["draft_turns"] = proposed_turns
                state["last_preview_turns"] = None
                observation = {
                    "ok": True,
                    "draft_turns": state["draft_turns"],
                    "rejected_turns": action.get("rejected_turns", []),
                    "note": "Draft replaced. Use preview_route to check graph connectivity.",
                }
        elif tool == "preview_route":
            proposed_turns = list(action.get("turns", state["draft_turns"]))
            guard = incremental_guard(state["draft_turns"], proposed_turns, max_new_turns=max_new_turns_per_edit)
            if guard:
                observation = guard
            else:
                state["draft_turns"] = proposed_turns
                preview_path = task_dir / f"step_{step:02d}_preview.png"
                diagnostics, feedback = render_and_score_preview(task, state["draft_turns"], preview_path)
                state["last_preview_feedback"] = feedback
                state["last_preview_image"] = str(preview_path)
                state["last_preview_turns"] = list(state["draft_turns"])
                observation = {
                    "ok": True,
                    "draft_turns": state["draft_turns"],
                    "preview": feedback,
                    "rejected_turns": action.get("rejected_turns", []),
                }
        elif tool == "finish":
            proposed_turns = list(action.get("turns", state["draft_turns"]))
            if proposed_turns != state["draft_turns"]:
                observation = {
                    "ok": False,
                    "error": "finish_turns_do_not_match_current_draft",
                    "current_draft": state["draft_turns"],
                    "proposed_turns": proposed_turns,
                    "note": "Use edit_route then preview_route before finish.",
                }
            elif state.get("last_preview_turns") != state["draft_turns"] or not (state.get("last_preview_feedback") or {}).get("valid_route"):
                observation = {
                    "ok": False,
                    "error": "finish_requires_latest_valid_preview",
                    "current_draft": state["draft_turns"],
                    "last_preview_turns": state.get("last_preview_turns"),
                    "last_preview_feedback": state.get("last_preview_feedback"),
                    "note": "Run preview_route on the current draft, then finish if it is valid and visually plausible.",
                }
            else:
                final_prediction = {"task_id": task["task_id"], "prediction": {"turns": state["draft_turns"]}}
                final_result = verify_prediction(task, final_prediction)
                final_agent_path = task_dir / f"step_{step:02d}_final_agent_preview.png"
                render_agent_preview(task, state["draft_turns"], final_result, final_agent_path, panel_label="Final: agent route only")
                final_oracle_path = task_dir / "final_with_oracle_overlay.png"
                render_debug_overlay(task, final_prediction, final_result, final_oracle_path)
                state["last_preview_image"] = str(final_agent_path)
                observation = {
                    "ok": True,
                    "finished": True,
                    "draft_turns": state["draft_turns"],
                    "score": finite_round(final_result.get("score"), 4),
                    "valid_route": bool(final_result.get("valid_route")),
                    "final_agent_image": str(final_agent_path),
                    "final_oracle_overlay": str(final_oracle_path),
                }
                finish_reason = "model_finish"
        else:
            observation = {
                "ok": True,
                "note": "Observed the same images again. Prefer inspect/edit/preview/finish next.",
            }

        step_record = {
            "step": step,
            "prompt_path": str(prompt_path),
            "image_paths": [str(path) for path in image_paths],
            "image_captions": captions,
            "sent_as_composite": send_composite_image,
            "source_image_paths": [str(path) for path in source_image_paths],
            "source_image_captions": source_image_captions,
            "raw_response_path": str(raw_path),
            "raw_response": raw,
            "parse_error": parse_error,
            "parsed": parsed,
            "action": action,
            "observation": observation,
        }
        steps.append(step_record)
        state["history"].append({"action": action, "observation": observation})
        if tool == "finish" and final_result is not None:
            break

    if final_result is None:
        final_prediction = {"task_id": task["task_id"], "prediction": {"turns": state["draft_turns"]}}
        final_result = verify_prediction(task, final_prediction)
        final_agent_path = task_dir / "forced_final_agent_preview.png"
        final_oracle_path = task_dir / "forced_final_with_oracle_overlay.png"
        render_agent_preview(task, state["draft_turns"], final_result, final_agent_path, panel_label="Forced final: agent route only")
        render_debug_overlay(task, final_prediction, final_result, final_oracle_path)

    return {
        "task_id": task["task_id"],
        "prediction": final_prediction["prediction"],
        "result": final_result,
        "finish_reason": finish_reason,
        "render_dir": str(task_dir),
        "overview_image": str(overview_path),
        "steps": steps,
    }


def selected_tasks(path: Path, *, start_index: int, num_tasks: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, task in enumerate(iter_jsonl(path)):
        if index < start_index:
            continue
        out.append(task)
        if num_tasks is not None and len(out) >= num_tasks:
            break
    return out


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    scores = [record["result"]["score"] for record in records]
    valid = [record["result"]["valid_route"] for record in records]
    return {
        "count": len(records),
        "mean_score": sum(scores) / len(scores),
        "success_at_0_75": sum(score >= 0.75 for score in scores) / len(scores),
        "valid_route_rate": sum(bool(item) for item in valid) / len(valid),
        "mean_turn_count": sum(len(record["prediction"].get("turns", [])) for record in records) / len(records),
        "finish_reasons": {
            reason: sum(1 for record in records if record["finish_reason"] == reason)
            for reason in sorted({record["finish_reason"] for record in records})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--inspect-margin-m", type=float, default=260.0)
    parser.add_argument("--max-label-table", type=int, default=120)
    parser.add_argument("--history-entries", type=int, default=24)
    parser.add_argument("--max-new-turns-per-edit", type=int, default=3)
    parser.add_argument(
        "--send-composite-image",
        action="store_true",
        help="Send a single stitched input-bundle image instead of separate images.",
    )
    parser.add_argument("--composite-max-panel-px", type=int, default=920)
    args = parser.parse_args()

    tasks = selected_tasks(Path(args.tasks), start_index=args.start_index, num_tasks=args.num_tasks)
    model, processor = load_vision_model(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    records = []
    for task in tqdm(tasks, desc="interactive-route-sketch"):
        records.append(
            run_task(
                task,
                model=model,
                processor=processor,
                render_root=Path(args.render_dir),
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                inspect_margin_m=args.inspect_margin_m,
                max_label_table=args.max_label_table,
                history_entries=args.history_entries,
                max_new_turns_per_edit=args.max_new_turns_per_edit,
                send_composite_image=args.send_composite_image,
                composite_max_panel_px=args.composite_max_panel_px,
            )
        )

    out_path = Path(args.out)
    write_jsonl(out_path, records)
    summary = summarize(records)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(records)} interactive rollouts to {out_path}")
    print(f"wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
