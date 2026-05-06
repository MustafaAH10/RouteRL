from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from route_env.geometry import lonlat_to_latlon


def _line_xy(geometry: list[list[float]]) -> tuple[list[float], list[float]]:
    return [p[0] for p in geometry], [p[1] for p in geometry]


def _draw_direction_arrow(ax: Any, geometry: list[list[float]], color: str = "#2f5fbd", alpha: float = 0.72) -> None:
    if len(geometry) < 2:
        return
    mid = max(1, len(geometry) // 2)
    start = geometry[mid - 1]
    end = geometry[mid]
    if start == end:
        return
    ax.add_patch(
        FancyArrowPatch(
            (start[0], start[1]),
            (end[0], end[1]),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.0,
            color=color,
            alpha=alpha,
            zorder=3,
        )
    )


def _edge_style(edge: dict[str, Any]) -> tuple[str, float, float]:
    highway = edge.get("highway", "")
    if isinstance(highway, list):
        highway = highway[0] if highway else ""
    major = {"motorway", "trunk", "primary", "secondary"}
    medium = {"tertiary", "motorway_link", "trunk_link", "primary_link", "secondary_link"}
    if highway in major:
        return "#4c566a", 1.8, 0.9
    if highway in medium:
        return "#677489", 1.35, 0.86
    return "#9aa1a8", 0.9, 0.74


def _graph_edges(task: dict[str, Any]) -> list[dict[str, Any]]:
    return task.get("graph", {}).get("edges", [])


def _label_number(label: str) -> int:
    try:
        return int(label.removeprefix("T"))
    except ValueError:
        return 10_000


def _overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    bottom = max(a[1], b[1])
    right = min(a[2], b[2])
    top = min(a[3], b[3])
    if right <= left or top <= bottom:
        return 0.0
    return (right - left) * (top - bottom)


def _label_candidates() -> list[tuple[int, int]]:
    return [
        (0, 14),
        (18, 8),
        (-18, 8),
        (18, -8),
        (-18, -8),
        (0, -16),
        (28, 0),
        (-28, 0),
        (32, 16),
        (-32, 16),
        (32, -16),
        (-32, -16),
        (0, 30),
        (0, -32),
        (44, 0),
        (-44, 0),
    ]


def _draw_checkpoint_labels(
    fig: Any,
    ax: Any,
    checkpoints: dict[str, Any],
    *,
    fontsize: float,
    marker_size: float,
    label_alpha: float,
) -> None:
    fig.canvas.draw()
    dpi_scale = fig.dpi / 72.0
    placed: list[tuple[float, float, float, float]] = []
    candidates = _label_candidates()

    for label, point in sorted(checkpoints.items(), key=lambda item: _label_number(item[0])):
        ax.scatter(point["lon"], point["lat"], s=marker_size, color="#111111", edgecolor="white", linewidth=0.7, zorder=5)
        px, py = ax.transData.transform((point["lon"], point["lat"]))
        # Conservative text box estimate in display pixels. The actual bbox is
        # drawn by matplotlib, this is only for collision avoidance.
        width = max(24.0, 9.0 * len(label) + 12.0)
        height = 21.0

        best_offset = candidates[0]
        best_box: tuple[float, float, float, float] | None = None
        best_cost = float("inf")
        for dx_pt, dy_pt in candidates:
            dx_px = dx_pt * dpi_scale
            dy_px = dy_pt * dpi_scale
            cx = px + dx_px
            cy = py + dy_px
            box = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
            overlap = sum(_overlap_area(box, existing) for existing in placed)
            distance_cost = (dx_pt * dx_pt + dy_pt * dy_pt) ** 0.5
            cost = overlap * 10.0 + distance_cost
            if cost < best_cost:
                best_cost = cost
                best_offset = (dx_pt, dy_pt)
                best_box = box

        if best_box is not None:
            placed.append(best_box)
        ax.annotate(
            label,
            xy=(point["lon"], point["lat"]),
            xytext=best_offset,
            textcoords="offset points",
            fontsize=fontsize,
            color="#111111",
            ha="center",
            va="center",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#cfcfcf", "alpha": label_alpha},
            arrowprops={
                "arrowstyle": "-",
                "color": "#505050",
                "alpha": 0.45,
                "linewidth": 0.55,
                "shrinkA": 2,
                "shrinkB": 3,
            }
            if abs(best_offset[0]) + abs(best_offset[1]) > 18
            else None,
        )


def render_task(task: dict[str, Any], out_path: str | Path, show_labels: bool = True) -> None:
    out_path = Path(out_path)
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
        ax.plot(xs, ys, color=color, linewidth=width, alpha=alpha, solid_capstyle="round", zorder=1)
        if edge.get("oneway"):
            _draw_direction_arrow(ax, edge["geometry"])

    if show_labels:
        _draw_checkpoint_labels(fig, ax, task["turn_checkpoints"], fontsize=8.5, marker_size=42, label_alpha=0.92)
    else:
        for point in task["turn_checkpoints"].values():
            ax.scatter(point["lon"], point["lat"], s=42, color="#111111", edgecolor="white", linewidth=0.7, zorder=5)

    origin = task["origin"]
    dest = task["destination"]
    ax.scatter(origin["lon"], origin["lat"], s=190, color="#1664d9", edgecolor="white", linewidth=1.6, zorder=7)
    ax.scatter(dest["lon"], dest["lat"], s=190, color="#d92525", edgecolor="white", linewidth=1.6, zorder=7)
    ax.text(origin["lon"], origin["lat"], "A", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=8)
    ax.text(dest["lon"], dest["lat"], "B", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=8)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_debug_overlay(
    task: dict[str, Any],
    prediction: dict[str, Any] | None,
    diagnostics: dict[str, Any] | None,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
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
        ax.plot(xs, ys, color=color, linewidth=max(0.7, width * 0.8), alpha=alpha * 0.55, solid_capstyle="round", zorder=1)
        if edge.get("oneway"):
            _draw_direction_arrow(ax, edge["geometry"], alpha=0.45)

    oracle = [lonlat_to_latlon(p) for p in task["oracle"]["geometry"]]
    ax.plot([p["lon"] for p in oracle], [p["lat"] for p in oracle], color="#11823b", linewidth=4, alpha=0.72, zorder=2)

    if diagnostics and diagnostics.get("agent_geometry"):
        agent = [lonlat_to_latlon(p) for p in diagnostics["agent_geometry"]]
        ax.plot([p["lon"] for p in agent], [p["lat"] for p in agent], color="#f27a1a", linewidth=3.3, alpha=0.86, zorder=3)

    _draw_checkpoint_labels(fig, ax, task["turn_checkpoints"], fontsize=7.8, marker_size=36, label_alpha=0.88)

    if prediction:
        turns = prediction.get("prediction", prediction).get("turns", [])
        title = f"{task['task_id']} predicted {json.dumps(turns)}"
        if diagnostics:
            title += f" score={diagnostics.get('score', 0):.3f}"
        ax.set_title(title, fontsize=9)

    origin = task["origin"]
    dest = task["destination"]
    ax.scatter(origin["lon"], origin["lat"], s=170, color="#1664d9", edgecolor="white", linewidth=1.4, zorder=7)
    ax.scatter(dest["lon"], dest["lat"], s=170, color="#d92525", edgecolor="white", linewidth=1.4, zorder=7)
    ax.text(origin["lon"], origin["lat"], "A", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=8)
    ax.text(dest["lon"], dest["lat"], "B", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=8)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
