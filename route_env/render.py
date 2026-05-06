from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

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
            mutation_scale=13,
            linewidth=0.55,
            color=color,
            alpha=min(1.0, alpha + 0.12),
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
    candidates: list[tuple[int, int]] = []
    for radius in (18, 32, 48, 68, 92, 120, 150):
        candidates.extend(
            [
                (0, radius),
                (radius, 0),
                (0, -radius),
                (-radius, 0),
                (radius, radius // 2),
                (-radius, radius // 2),
                (radius, -(radius // 2)),
                (-radius, -(radius // 2)),
                (radius // 2, radius),
                (-(radius // 2), radius),
                (radius // 2, -radius),
                (-(radius // 2), -radius),
            ]
        )
    return candidates


def _draw_checkpoint_labels(
    fig: Any,
    ax: Any,
    checkpoints: dict[str, Any],
    *,
    fontsize: float,
    marker_size: float,
    label_alpha: float,
    occupied_points: list[dict[str, float]] | None = None,
) -> None:
    fig.canvas.draw()
    dpi_scale = fig.dpi / 72.0
    axes_box = ax.get_window_extent()
    placed: list[tuple[float, float, float, float]] = []
    for point in occupied_points or []:
        px, py = ax.transData.transform((point["lon"], point["lat"]))
        placed.append((px - 34, py - 34, px + 34, py + 34))
    for point in checkpoints.values():
        px, py = ax.transData.transform((point["lon"], point["lat"]))
        placed.append((px - 9, py - 9, px + 9, py + 9))
    candidates = _label_candidates()

    for label, point in sorted(checkpoints.items(), key=lambda item: _label_number(item[0])):
        ax.scatter(point["lon"], point["lat"], s=marker_size, color="#111111", edgecolor="white", linewidth=0.7, zorder=5)
        px, py = ax.transData.transform((point["lon"], point["lat"]))
        # Conservative text box estimate in display pixels. The actual bbox is
        # drawn by matplotlib, this is only for collision avoidance.
        width = max(38.0, 9.5 * len(label) + 22.0)
        height = 27.0

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
            outside = (
                max(0.0, axes_box.x0 - box[0])
                + max(0.0, box[2] - axes_box.x1)
                + max(0.0, axes_box.y0 - box[1])
                + max(0.0, box[3] - axes_box.y1)
            )
            distance_cost = (dx_pt * dx_pt + dy_pt * dy_pt) ** 0.5
            cost = overlap * 150.0 + outside * 60.0 + distance_cost
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


def _draw_grid_overlay(ax: Any, bbox: list[float], cols: int = 4, rows: int = 4) -> None:
    west, south, east, north = bbox
    width = east - west
    height = north - south
    for i in range(1, cols):
        x = west + width * i / cols
        ax.plot([x, x], [south, north], color="#708090", linewidth=0.45, alpha=0.25, zorder=0)
    for j in range(1, rows):
        y = south + height * j / rows
        ax.plot([west, east], [y, y], color="#708090", linewidth=0.45, alpha=0.25, zorder=0)
    for i in range(cols):
        x = west + width * (i + 0.5) / cols
        ax.text(x, north - height * 0.012, chr(ord("A") + i), ha="center", va="top", fontsize=7, color="#596575", zorder=8)
    for j in range(rows):
        y = north - height * (j + 0.5) / rows
        ax.text(west + width * 0.012, y, str(j + 1), ha="left", va="center", fontsize=7, color="#596575", zorder=8)


def _draw_panel_badge(ax: Any, label: str) -> None:
    ax.text(
        0.018,
        0.982,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        weight="bold",
        color="#172033",
        zorder=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bdc4cf", "alpha": 0.94},
    )


def render_task(
    task: dict[str, Any],
    out_path: str | Path,
    show_labels: bool = True,
    *,
    panel_label: str | None = None,
    show_grid: bool = False,
    grid_cols: int = 4,
    grid_rows: int = 4,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = task["task_bbox"]
    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)
    ax.set_facecolor("#f8f8f4")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
    if show_grid:
        _draw_grid_overlay(ax, task["task_bbox"], cols=grid_cols, rows=grid_rows)

    for edge in _graph_edges(task):
        xs, ys = _line_xy(edge["geometry"])
        color, width, alpha = _edge_style(edge)
        ax.plot(xs, ys, color=color, linewidth=width, alpha=alpha, solid_capstyle="round", zorder=1)
        if edge.get("oneway"):
            _draw_direction_arrow(ax, edge["geometry"])

    origin = task["origin"]
    dest = task["destination"]
    if show_labels:
        _draw_checkpoint_labels(
            fig,
            ax,
            task["turn_checkpoints"],
            fontsize=7.8,
            marker_size=38,
            label_alpha=0.92,
            occupied_points=[origin, dest],
        )
    else:
        for point in task["turn_checkpoints"].values():
            ax.scatter(point["lon"], point["lat"], s=42, color="#111111", edgecolor="white", linewidth=0.7, zorder=5)

    ax.scatter(origin["lon"], origin["lat"], s=190, color="#1664d9", edgecolor="white", linewidth=1.6, zorder=7)
    ax.scatter(dest["lon"], dest["lat"], s=190, color="#d92525", edgecolor="white", linewidth=1.6, zorder=7)
    ax.text(origin["lon"], origin["lat"], "A", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=8)
    ax.text(dest["lon"], dest["lat"], "B", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=8)
    if panel_label:
        _draw_panel_badge(ax, panel_label)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_route_strip_overview(task: dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    west, south, east, north = task["task_bbox"]
    fig, ax = plt.subplots(figsize=(12, 8), dpi=180)
    ax.set_facecolor("#f8f8f4")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
    _draw_grid_overlay(ax, task["task_bbox"], cols=6, rows=4)

    for edge in _graph_edges(task):
        xs, ys = _line_xy(edge["geometry"])
        color, width, alpha = _edge_style(edge)
        ax.plot(xs, ys, color=color, linewidth=max(0.65, width * 0.72), alpha=alpha * 0.62, solid_capstyle="round", zorder=1)

    for segment in task.get("segments", []):
        sw, ss, se, sn = segment["task_bbox"]
        ax.add_patch(
            Rectangle(
                (sw, ss),
                se - sw,
                sn - ss,
                fill=False,
                edgecolor="#e07828",
                linewidth=1.15,
                alpha=0.84,
                zorder=4,
            )
        )
        ax.text(
            sw,
            sn,
            segment["segment_id"],
            ha="left",
            va="bottom",
            fontsize=7.5,
            weight="bold",
            color="#8f3f0d",
            zorder=5,
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "#e2b58f", "alpha": 0.92},
        )

    origin = task["origin"]
    dest = task["destination"]
    ax.scatter(origin["lon"], origin["lat"], s=190, color="#1664d9", edgecolor="white", linewidth=1.6, zorder=7)
    ax.scatter(dest["lon"], dest["lat"], s=190, color="#d92525", edgecolor="white", linewidth=1.6, zorder=7)
    ax.text(origin["lon"], origin["lat"], "A", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=8)
    ax.text(dest["lon"], dest["lat"], "B", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=8)
    _draw_panel_badge(ax, "Overview")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_route_strip_task(task: dict[str, Any], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    base = out_dir / task["task_id"]
    overview_path = base / "overview.png"
    task["images"]["overview"] = str(overview_path)
    render_route_strip_overview(task, overview_path)
    segment_paths = []
    for segment in task.get("segments", []):
        segment_path = base / f"{segment['segment_id'].lower()}.png"
        segment["images"]["map"] = str(segment_path)
        render_task(segment, segment_path, panel_label=segment["segment_id"], show_grid=True, grid_cols=4, grid_rows=4)
        segment_paths.append(str(segment_path))
    task["images"]["segments"] = segment_paths


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

    if prediction:
        turns = prediction.get("prediction", prediction).get("turns", [])
        title = f"{task['task_id']} predicted {json.dumps(turns)}"
        if diagnostics:
            title += f" score={diagnostics.get('score', 0):.3f}"
        ax.set_title(title, fontsize=9)

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
