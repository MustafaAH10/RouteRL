from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import networkx as nx

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from route_env.geometry import (
    bbox_from_points,
    latlon_to_lonlat,
    lonlat_to_latlon,
    mean_bidirectional_distance_m,
)
from route_env.verify import checkpoint_alignment_reward, geometry_for_osm_path, task_graph


TRACE_COLORS = ["#f97316", "#2563eb", "#16a34a", "#c026d3", "#dc2626", "#0891b2"]


@dataclass
class TraceCandidate:
    candidate_id: str
    node_path: list[str]
    geometry: list[dict[str, float]]
    length_m: float
    checkpoint_labels: list[str]
    reachable_to_destination: bool
    remaining_distance_m: float | None

    @property
    def to_node(self) -> str:
        return self.node_path[-1]

    @property
    def ends_at_destination(self) -> bool:
        return self.remaining_distance_m == 0.0


@dataclass
class TraceChoiceStep:
    observation: dict[str, Any]
    step_reward: float
    done: bool
    error: str | None = None


def _node_point(graph: nx.DiGraph, node_id: str) -> dict[str, float]:
    node = graph.nodes[str(node_id)]
    return {"lat": float(node["lat"]), "lon": float(node["lon"])}


def _edge_geometry(graph: nx.DiGraph, u: str, v: str) -> list[dict[str, float]]:
    data = graph.edges[str(u), str(v)]
    geometry = data.get("geometry") or []
    if geometry:
        return [lonlat_to_latlon(point) for point in geometry]
    return [_node_point(graph, str(u)), _node_point(graph, str(v))]


def _path_geometry(graph: nx.DiGraph, path: list[str]) -> list[dict[str, float]]:
    return geometry_for_osm_path(graph, path)


def _path_length_m(graph: nx.DiGraph, path: list[str]) -> float:
    return sum(float(graph.edges[u, v].get("length", 0.0)) for u, v in zip(path, path[1:]))


def _dedupe_consecutive(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def _edge_style(edge_data: dict[str, Any]) -> tuple[str, float, float]:
    highway = edge_data.get("highway", "")
    if isinstance(highway, list):
        highway = highway[0] if highway else ""
    if highway in {"motorway", "trunk", "primary", "secondary"}:
        return "#4b5563", 1.7, 0.78
    if highway in {"tertiary", "motorway_link", "trunk_link", "primary_link", "secondary_link"}:
        return "#6b7280", 1.25, 0.68
    return "#9ca3af", 0.85, 0.58


def _bbox_contains(bbox: list[float], point: dict[str, float]) -> bool:
    west, south, east, north = bbox
    return west <= point["lon"] <= east and south <= point["lat"] <= north


class TraceChoiceEnv:
    """Graph-native routing sandbox.

    The agent never picks arbitrary checkpoint labels. It chooses among valid
    directed road continuations from the current graph frontier. This makes
    reachability and one-way direction structural properties of the action
    space, while still leaving route quality to the policy/reward.
    """

    def __init__(
        self,
        task: dict[str, Any],
        *,
        max_steps: int = 128,
        trace_length_m: float = 350.0,
        max_candidates: int = 6,
        render_dir: str | Path | None = None,
        render_context: str = "local",
        avoid_visited: bool = True,
        hide_unreachable_candidates: bool = True,
        force_destination_candidate: bool = True,
    ) -> None:
        if task.get("task_type") == "route_strip":
            raise ValueError("TraceChoiceEnv currently expects a flat task, not a route_strip task")
        if render_context not in {"local", "local_overview"}:
            raise ValueError(f"unknown render_context: {render_context}")
        self.task = deepcopy(task)
        self.graph = task_graph(self.task)
        self.max_steps = max_steps
        self.trace_length_m = trace_length_m
        self.max_candidates = max_candidates
        self.render_dir = Path(render_dir) if render_dir else None
        self.render_context = render_context
        self.avoid_visited = avoid_visited
        self.hide_unreachable_candidates = hide_unreachable_candidates
        self.force_destination_candidate = force_destination_candidate
        self.origin_node = str(int(self.task["origin"]["osm_id"]))
        self.destination_node = str(int(self.task["destination"]["osm_id"]))
        self.checkpoint_by_node = {
            str(int(point["osm_id"])): label for label, point in self.task.get("turn_checkpoints", {}).items()
        }
        self._distance_to_destination = self._build_distance_to_destination()
        self._edge_render_cache = self._build_edge_render_cache() if self.render_dir else []
        self._overview_bbox = (
            bbox_from_points(
                [_node_point(self.graph, str(node_id)) for node_id in self.graph.nodes],
                margin_m=180,
            )
            if self.render_dir
            else []
        )
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.current_node = self.origin_node
        self.route_nodes = [self.origin_node]
        self.action_count = 0
        self.done = False
        self.last_error: str | None = None
        self.last_metrics: dict[str, Any] | None = None
        return self.observe()

    def observe(self) -> dict[str, Any]:
        candidates = [] if self.done or self.current_node == self.destination_node else self.candidates()
        image = self._render_observation(candidates) if self.render_dir else None
        return {
            "task_id": self.task["task_id"],
            "task_type": "trace_choice",
            "view": {
                "kind": "trace_choice",
                "context": self.render_context,
                "image": image,
                "current_node": self.current_node,
            },
            "current": _node_point(self.graph, self.current_node),
            "destination": _node_point(self.graph, self.destination_node),
            "route_so_far": {
                "node_count": len(self.route_nodes),
                "distance_m": _path_length_m(self.graph, self.route_nodes),
                "turns": self.prediction()["turns"],
            },
            "candidates": [self._candidate_record(candidate) for candidate in candidates],
            "remaining_steps": max(0, self.max_steps - self.action_count),
            "done": self.done,
            "last_error": self.last_error,
            "metrics": self.last_metrics,
        }

    def prediction(self) -> dict[str, list[str]]:
        turns = [self.checkpoint_by_node[node] for node in self.route_nodes if node in self.checkpoint_by_node]
        return {"turns": _dedupe_consecutive(turns)}

    def scored_prediction(self) -> dict[str, list[str]]:
        gold_turns = set(self.task.get("oracle", {}).get("gold_turn_route", []))
        turns = [label for label in self.prediction()["turns"] if label in gold_turns]
        return {"turns": turns}

    def step(self, action: dict[str, Any]) -> TraceChoiceStep:
        if self.done:
            self.last_error = "episode is already done"
            return TraceChoiceStep(self.observe(), 0.0, True, self.last_error)

        self.action_count += 1
        self.last_error = None
        tool = str(action.get("tool", action.get("action", "")))

        try:
            if tool == "finish":
                reward = self._finish()
            elif tool == "choose":
                candidate_id = str(action.get("candidate_id", action.get("choice", ""))).upper()
                reward = self._choose(candidate_id)
            else:
                raise ValueError(f"unknown tool: {tool or '<missing>'}")
        except ValueError as exc:
            self.last_error = str(exc)
            reward = -0.05

        if not self.done and self.current_node == self.destination_node:
            reward = self._finish()
        elif not self.done and self.action_count >= self.max_steps:
            self.last_error = self.last_error or "step budget exhausted; finishing episode"
            reward = self._finish()

        return TraceChoiceStep(self.observe(), reward, self.done, self.last_error)

    def candidates(self) -> list[TraceCandidate]:
        raw = []
        successors = list(self.graph.successors(self.current_node))
        for successor in successors:
            candidate = self._candidate_from_successor(str(successor))
            if candidate:
                raw.append(candidate)

        if self.avoid_visited:
            visited = set(self.route_nodes[:-1])
            unvisited = [candidate for candidate in raw if not any(node in visited for node in candidate.node_path[1:])]
            if unvisited:
                raw = unvisited

        if self.hide_unreachable_candidates:
            reachable = [candidate for candidate in raw if candidate.reachable_to_destination]
            if reachable:
                raw = reachable

        if self.force_destination_candidate:
            destination_candidates = [candidate for candidate in raw if candidate.to_node == self.destination_node]
            if destination_candidates:
                raw = destination_candidates

        raw.sort(
            key=lambda candidate: (
                not candidate.reachable_to_destination,
                candidate.to_node != self.destination_node,
                candidate.length_m,
                candidate.to_node,
            )
        )
        return [
            TraceCandidate(
                candidate_id=f"C{index}",
                node_path=candidate.node_path,
                geometry=candidate.geometry,
                length_m=candidate.length_m,
                checkpoint_labels=candidate.checkpoint_labels,
                reachable_to_destination=candidate.reachable_to_destination,
                remaining_distance_m=candidate.remaining_distance_m,
            )
            for index, candidate in enumerate(raw[: self.max_candidates], start=1)
        ]

    def oracle_candidate_id(self) -> str | None:
        gold = [str(int(node)) for node in self.task.get("oracle", {}).get("gold_osm_route", [])]
        if self.current_node not in gold:
            return None
        current_index = gold.index(self.current_node)
        candidates = self.candidates()
        best: tuple[int, str] | None = None
        for candidate in candidates:
            overlap = 0
            for offset, node in enumerate(candidate.node_path[1:], start=1):
                if current_index + offset >= len(gold) or gold[current_index + offset] != node:
                    break
                overlap += 1
            if overlap and (best is None or overlap > best[0]):
                best = (overlap, candidate.candidate_id)
        return best[1] if best else None

    def shortest_candidate_id(self) -> str | None:
        best: tuple[float, str] | None = None
        for candidate in self.candidates():
            tail = self._distance_to_destination_m(candidate.to_node)
            if tail is None:
                continue
            cost = candidate.length_m + tail
            if best is None or cost < best[0]:
                best = (cost, candidate.candidate_id)
        return best[1] if best else None

    def heading_candidate_id(self) -> str | None:
        destination = _node_point(self.graph, self.destination_node)
        best: tuple[float, str] | None = None
        for candidate in self.candidates():
            end = _node_point(self.graph, candidate.to_node)
            score = (end["lon"] - destination["lon"]) ** 2 + (end["lat"] - destination["lat"]) ** 2
            if best is None or score < best[0]:
                best = (score, candidate.candidate_id)
        return best[1] if best else None

    def _choose(self, candidate_id: str) -> float:
        candidates = {candidate.candidate_id: candidate for candidate in self.candidates()}
        candidate = candidates.get(candidate_id)
        if not candidate:
            raise ValueError(f"unknown candidate_id: {candidate_id}")
        self.route_nodes.extend(candidate.node_path[1:])
        self.current_node = candidate.to_node
        return 0.0

    def _candidate_from_successor(self, successor: str) -> TraceCandidate | None:
        if not self.graph.has_edge(self.current_node, successor):
            return None
        path = [self.current_node, successor]
        length = float(self.graph.edges[self.current_node, successor].get("length", 0.0))
        node = successor
        seen = {self.current_node}

        while length < self.trace_length_m and node != self.destination_node:
            if node in self.checkpoint_by_node and node != successor:
                break
            successors = list(self.graph.successors(node))
            if len(successors) != 1:
                break
            nxt = str(successors[0])
            if nxt in seen or not self.graph.has_edge(node, nxt):
                break
            seen.add(node)
            path.append(nxt)
            length += float(self.graph.edges[node, nxt].get("length", 0.0))
            node = nxt

        geometry = _path_geometry(self.graph, path) if self.render_dir else []
        labels = [self.checkpoint_by_node[item] for item in path if item in self.checkpoint_by_node]
        remaining_distance_m = self._distance_to_destination_m(path[-1])
        reachable = remaining_distance_m is not None
        return TraceCandidate(
            candidate_id="",
            node_path=path,
            geometry=geometry,
            length_m=length,
            checkpoint_labels=_dedupe_consecutive(labels),
            reachable_to_destination=reachable,
            remaining_distance_m=remaining_distance_m,
        )

    def _build_distance_to_destination(self) -> dict[str, float]:
        if self.destination_node not in self.graph:
            return {}
        reverse_graph = self.graph.reverse(copy=False)
        return {
            str(node): float(distance)
            for node, distance in nx.single_source_dijkstra_path_length(
                reverse_graph,
                self.destination_node,
                weight="length",
            ).items()
        }

    def _distance_to_destination_m(self, node: str) -> float | None:
        return self._distance_to_destination.get(str(node))

    def _candidate_record(self, candidate: TraceCandidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "to_node": candidate.to_node,
            "length_m": round(candidate.length_m, 1),
            "checkpoint_labels": candidate.checkpoint_labels,
            "reachable_to_destination": candidate.reachable_to_destination,
            "ends_at_destination": candidate.to_node == self.destination_node,
            "remaining_distance_m": round(candidate.remaining_distance_m, 1)
            if candidate.remaining_distance_m is not None
            else None,
        }

    def _finish(self) -> float:
        self.done = True
        self.last_metrics = self.direct_metrics()
        return float(self.last_metrics.get("score", 0.0))

    def direct_metrics(self) -> dict[str, Any]:
        valid_route = self.current_node == self.destination_node and len(self.route_nodes) >= 2
        agent_geometry = _path_geometry(self.graph, self.route_nodes) if len(self.route_nodes) >= 2 else []
        oracle_geometry = [lonlat_to_latlon(point) for point in self.task["oracle"]["geometry"]]
        agent_distance = _path_length_m(self.graph, self.route_nodes) if len(self.route_nodes) >= 2 else math.inf
        oracle_distance = float(self.task["oracle"]["distance_m"])
        length_ratio = agent_distance / oracle_distance if valid_route and oracle_distance > 0 else math.inf
        mean_distance = mean_bidirectional_distance_m(agent_geometry, oracle_geometry) if valid_route else math.inf
        observed_turns = self.prediction()["turns"]
        turns = self.scored_prediction()["turns"]
        gold_turns = list(self.task.get("oracle", {}).get("gold_turn_route", []))
        checkpoint_reward, coverage, precision, order = checkpoint_alignment_reward(turns, gold_turns)
        distance_ratio_reward = math.exp(-abs(math.log(length_ratio))) if math.isfinite(length_ratio) and length_ratio > 0 else 0.0
        similarity_reward = math.exp(-mean_distance / 100) if math.isfinite(mean_distance) else 0.0
        endpoint_reward = 1.0 if valid_route else 0.0
        loop_count = len(self.route_nodes) - len(set(self.route_nodes))
        loop_penalty = max(0.0, 1.0 - 0.05 * loop_count)
        score = (
            0.25 * endpoint_reward
            + 0.25 * distance_ratio_reward
            + 0.20 * similarity_reward
            + 0.30 * checkpoint_reward
        ) * loop_penalty
        return {
            "task_id": self.task["task_id"],
            "valid_schema": True,
            "valid_route": valid_route,
            "num_predicted_turns": len(turns),
            "num_observed_turns": len(observed_turns),
            "num_gold_turns": len(gold_turns),
            "agent_distance_m": agent_distance,
            "oracle_distance_m": oracle_distance,
            "length_ratio": length_ratio,
            "mean_route_distance_m": mean_distance,
            "checkpoint_reward": checkpoint_reward,
            "checkpoint_coverage": coverage,
            "checkpoint_precision": precision,
            "checkpoint_order": order,
            "loop_penalty": loop_penalty,
            "score": max(0.0, min(1.0, score)),
            "agent_osm_route_expanded": list(self.route_nodes),
            "agent_geometry": [latlon_to_lonlat(point) for point in agent_geometry],
        }

    def _build_edge_render_cache(self) -> list[tuple[list[dict[str, float]], str, float, float]]:
        return [
            (*(_edge_geometry(self.graph, str(u), str(v)),), *_edge_style(data))
            for u, v, data in self.graph.edges(data=True)
        ]

    def _local_bbox(self, candidates: list[TraceCandidate]) -> list[float]:
        points = [_node_point(self.graph, self.current_node)]
        points.extend(point for candidate in candidates for point in candidate.geometry)
        if self.current_node == self.destination_node:
            points.append(_node_point(self.graph, self.destination_node))
        return bbox_from_points(points, margin_m=110)

    def _prepare_axis(self, ax: Any, bbox: list[float], *, facecolor: str = "#f8fafc") -> None:
        west, south, east, north = bbox
        ax.set_facecolor(facecolor)
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_edges(self, ax: Any, bbox: list[float] | None, *, width_scale: float = 1.0, alpha_scale: float = 1.0) -> None:
        for geometry, color, width, alpha in self._edge_render_cache:
            if bbox is not None and not any(_bbox_contains(bbox, point) for point in geometry):
                continue
            ax.plot(
                [point["lon"] for point in geometry],
                [point["lat"] for point in geometry],
                color=color,
                linewidth=width * width_scale,
                alpha=min(1.0, alpha * alpha_scale),
                solid_capstyle="round",
                zorder=1,
            )

    def _draw_route_so_far(self, ax: Any, *, linewidth: float, alpha: float, zorder: int) -> None:
        if len(self.route_nodes) >= 2:
            route_geometry = _path_geometry(self.graph, self.route_nodes)
            ax.plot(
                [point["lon"] for point in route_geometry],
                [point["lat"] for point in route_geometry],
                color="#f97316",
                linewidth=linewidth,
                alpha=alpha,
                solid_capstyle="round",
                zorder=zorder,
            )

    def _draw_candidates(self, ax: Any, candidates: list[TraceCandidate], *, linewidth: float, label_size: int) -> None:
        for index, candidate in enumerate(candidates):
            color = TRACE_COLORS[index % len(TRACE_COLORS)]
            ax.plot(
                [point["lon"] for point in candidate.geometry],
                [point["lat"] for point in candidate.geometry],
                color=color,
                linewidth=linewidth,
                alpha=0.92,
                solid_capstyle="round",
                zorder=4,
            )
            if len(candidate.geometry) >= 2:
                start = candidate.geometry[-2]
                end = candidate.geometry[-1]
                ax.annotate(
                    "",
                    xy=(end["lon"], end["lat"]),
                    xytext=(start["lon"], start["lat"]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "lw": 0,
                        "mutation_scale": 15 + linewidth,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                    zorder=6,
                )
            end = candidate.geometry[-1]
            label_bbox = {"boxstyle": "circle,pad=0.25", "facecolor": color, "edgecolor": "white", "linewidth": 1.2}
            if candidate.to_node == self.destination_node:
                ax.annotate(
                    candidate.candidate_id,
                    xy=(end["lon"], end["lat"]),
                    xytext=(18, 18),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=label_size,
                    weight="bold",
                    color="white",
                    zorder=10,
                    bbox=label_bbox,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 1.3},
                )
            else:
                ax.text(
                    end["lon"],
                    end["lat"],
                    candidate.candidate_id,
                    ha="center",
                    va="center",
                    fontsize=label_size,
                    weight="bold",
                    color="white",
                    zorder=7,
                    bbox=label_bbox,
                )

    def _draw_markers(self, ax: Any, bbox: list[float], *, marker_size: int = 150, font_size: int = 9) -> None:
        current = _node_point(self.graph, self.current_node)
        dest = _node_point(self.graph, self.destination_node)
        origin = _node_point(self.graph, self.origin_node)
        if _bbox_contains(bbox, origin):
            ax.scatter(origin["lon"], origin["lat"], s=marker_size * 0.58, color="#111827", edgecolor="white", linewidth=1.0, zorder=8)
            ax.text(origin["lon"], origin["lat"], "A", color="white", weight="bold", ha="center", va="center", fontsize=max(6, font_size - 2), zorder=9)
        ax.scatter(current["lon"], current["lat"], s=marker_size, color="#1664d9", edgecolor="white", linewidth=1.5, zorder=8)
        ax.text(current["lon"], current["lat"], "F", color="white", weight="bold", ha="center", va="center", fontsize=font_size, zorder=9)
        if _bbox_contains(bbox, dest):
            ax.scatter(dest["lon"], dest["lat"], s=marker_size, color="#dc2626", edgecolor="white", linewidth=1.5, zorder=8)
            ax.text(dest["lon"], dest["lat"], "B", color="white", weight="bold", ha="center", va="center", fontsize=font_size, zorder=9)

    def _draw_candidate_to_destination_guides(self, ax: Any, candidates: list[TraceCandidate]) -> None:
        dest = _node_point(self.graph, self.destination_node)
        for index, candidate in enumerate(candidates):
            if candidate.to_node == self.destination_node:
                continue
            color = TRACE_COLORS[index % len(TRACE_COLORS)]
            end = _node_point(self.graph, candidate.to_node)
            ax.plot(
                [end["lon"], dest["lon"]],
                [end["lat"], dest["lat"]],
                color=color,
                linewidth=1.0,
                linestyle=(0, (4, 4)),
                alpha=0.5,
                zorder=2,
            )

    def _draw_local_observation(self, ax: Any, candidates: list[TraceCandidate]) -> None:
        bbox = self._local_bbox(candidates)
        self._prepare_axis(ax, bbox)
        self._draw_edges(ax, bbox)
        self._draw_route_so_far(ax, linewidth=4.0, alpha=0.78, zorder=3)
        self._draw_candidates(ax, candidates, linewidth=5.0, label_size=10)
        self._draw_markers(ax, bbox, marker_size=150, font_size=9)

    def _draw_overview_observation(self, ax: Any, candidates: list[TraceCandidate]) -> None:
        bbox = self._overview_bbox
        self._prepare_axis(ax, bbox, facecolor="#f1f5f9")
        self._draw_edges(ax, None, width_scale=0.42, alpha_scale=0.45)
        self._draw_candidate_to_destination_guides(ax, candidates)
        self._draw_route_so_far(ax, linewidth=2.2, alpha=0.84, zorder=3)
        self._draw_candidates(ax, candidates, linewidth=3.0, label_size=8)
        self._draw_markers(ax, bbox, marker_size=90, font_size=7)

    def _render_observation(self, candidates: list[TraceCandidate]) -> str:
        out_dir = self.render_dir / self.task["task_id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"step_{self.action_count:03d}.png"

        if self.render_context == "local_overview":
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(12, 6),
                dpi=170,
                gridspec_kw={"width_ratios": [1.08, 0.92], "wspace": 0.015},
            )
            self._draw_local_observation(axes[0], candidates)
            self._draw_overview_observation(axes[1], candidates)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.015)
        else:
            fig, ax = plt.subplots(figsize=(8, 8), dpi=170)
            self._draw_local_observation(ax, candidates)
            fig.tight_layout(pad=0)

        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return str(out_path)
