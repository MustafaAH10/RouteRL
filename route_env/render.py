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


def render_task(task: dict[str, Any], out_path: str | Path, show_labels: bool = True) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = task["task_bbox"]
    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    ax.set_facecolor("#f8f8f4")

    for edge in _graph_edges(task):
        xs, ys = _line_xy(edge["geometry"])
        color, width, alpha = _edge_style(edge)
        ax.plot(xs, ys, color=color, linewidth=width, alpha=alpha, solid_capstyle="round", zorder=1)
        if edge.get("oneway"):
            _draw_direction_arrow(ax, edge["geometry"])

    for label, point in task["turn_checkpoints"].items():
        ax.scatter(point["lon"], point["lat"], s=42, color="#111111", edgecolor="white", linewidth=0.7, zorder=5)
        if show_labels:
            ax.text(
                point["lon"],
                point["lat"],
                label,
                fontsize=8.0,
                color="#111111",
                ha="left",
                va="bottom",
                zorder=6,
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "#d6d6d6", "alpha": 0.9},
            )

    origin = task["origin"]
    dest = task["destination"]
    ax.scatter(origin["lon"], origin["lat"], s=190, color="#1664d9", edgecolor="white", linewidth=1.6, zorder=7)
    ax.scatter(dest["lon"], dest["lat"], s=190, color="#d92525", edgecolor="white", linewidth=1.6, zorder=7)
    ax.text(origin["lon"], origin["lat"], "A", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=8)
    ax.text(dest["lon"], dest["lat"], "B", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=8)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
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

    for label, point in task["turn_checkpoints"].items():
        ax.scatter(point["lon"], point["lat"], s=36, color="#111111", edgecolor="white", linewidth=0.7, zorder=5)
        ax.text(
            point["lon"],
            point["lat"],
            label,
            fontsize=7.5,
            color="#111111",
            ha="left",
            va="bottom",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#d6d6d6", "alpha": 0.88},
        )

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

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
