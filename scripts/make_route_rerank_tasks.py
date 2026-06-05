#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import matplotlib
import networkx as nx

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from route_env.geometry import latlon_to_lonlat, lonlat_to_latlon, mean_bidirectional_distance_m
from route_env.io import iter_jsonl, write_jsonl
from route_env.verify import checkpoint_alignment_reward, geometry_for_osm_path, task_graph


ROUTE_COLORS = ["#f97316", "#2563eb", "#16a34a", "#c026d3", "#dc2626", "#0891b2", "#7c3aed"]


def _path_length_m(graph: nx.DiGraph, path: list[str]) -> float:
    return sum(float(graph.edges[u, v].get("length", 0.0)) for u, v in zip(path, path[1:]))


def _node_point(graph: nx.DiGraph, node_id: str) -> dict[str, float]:
    node = graph.nodes[str(node_id)]
    return {"lat": float(node["lat"]), "lon": float(node["lon"])}


def _edge_style(edge: dict[str, Any]) -> tuple[str, float, float]:
    highway = edge.get("highway", "")
    if isinstance(highway, list):
        highway = highway[0] if highway else ""
    if highway in {"motorway", "trunk", "primary", "secondary"}:
        return "#4b5563", 1.15, 0.55
    if highway in {"tertiary", "motorway_link", "trunk_link", "primary_link", "secondary_link"}:
        return "#6b7280", 0.9, 0.5
    return "#a0a7b1", 0.65, 0.42


def _turns_for_path(task: dict[str, Any], path: list[str]) -> list[str]:
    checkpoint_by_node = {
        str(int(point["osm_id"])): label for label, point in task.get("turn_checkpoints", {}).items()
    }
    turns: list[str] = []
    for node in path:
        label = checkpoint_by_node.get(str(node))
        if label and (not turns or turns[-1] != label):
            turns.append(label)
    return turns


def route_score(task: dict[str, Any], graph: nx.DiGraph, path: list[str]) -> dict[str, Any]:
    oracle_geometry = [lonlat_to_latlon(point) for point in task["oracle"]["geometry"]]
    agent_geometry = geometry_for_osm_path(graph, path)
    agent_distance = _path_length_m(graph, path)
    oracle_distance = float(task["oracle"]["distance_m"])
    length_ratio = agent_distance / oracle_distance if oracle_distance > 0 else math.inf
    mean_distance = mean_bidirectional_distance_m(agent_geometry, oracle_geometry)
    turns = _turns_for_path(task, path)
    gold_turns = list(task.get("oracle", {}).get("gold_turn_route", []))
    checkpoint_reward, coverage, precision, order = checkpoint_alignment_reward(
        [turn for turn in turns if turn in set(gold_turns)],
        gold_turns,
    )
    distance_reward = math.exp(-abs(math.log(length_ratio))) if math.isfinite(length_ratio) and length_ratio > 0 else 0.0
    similarity_reward = math.exp(-mean_distance / 120) if math.isfinite(mean_distance) else 0.0
    score = 0.40 * distance_reward + 0.35 * similarity_reward + 0.25 * checkpoint_reward
    return {
        "score": max(0.0, min(1.0, score)),
        "distance_m": agent_distance,
        "length_ratio": length_ratio,
        "mean_route_distance_m": mean_distance,
        "checkpoint_reward": checkpoint_reward,
        "checkpoint_coverage": coverage,
        "checkpoint_precision": precision,
        "checkpoint_order": order,
        "turns": turns,
        "num_turns": len(turns),
    }


def penalized_shortest_path(
    graph: nx.DiGraph,
    origin: str,
    destination: str,
    penalized_edges: set[tuple[str, str]],
    penalty: float,
) -> list[str] | None:
    def weight(u: str, v: str, data: dict[str, Any]) -> float:
        value = float(data.get("length", 0.0))
        return value * penalty if (str(u), str(v)) in penalized_edges else value

    try:
        return [str(node) for node in nx.shortest_path(graph, origin, destination, weight=weight)]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def via_route(graph: nx.DiGraph, origin: str, via: str, destination: str) -> list[str] | None:
    try:
        first = [str(node) for node in nx.shortest_path(graph, origin, via, weight="length")]
        second = [str(node) for node in nx.shortest_path(graph, via, destination, weight="length")]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return first + second[1:]


def candidate_paths(task: dict[str, Any], *, num_candidates: int, rng: random.Random) -> list[tuple[str, list[str]]]:
    graph = task_graph(task)
    origin = str(int(task["origin"]["osm_id"]))
    destination = str(int(task["destination"]["osm_id"]))
    oracle = [str(int(node)) for node in task["oracle"]["gold_osm_route"]]
    oracle_length = _path_length_m(graph, oracle)
    out: list[tuple[str, list[str]]] = [("oracle", oracle)]
    seen = {tuple(oracle)}
    oracle_edges = list(zip(oracle, oracle[1:]))

    attempts = 0
    while len(out) < num_candidates and attempts < 80:
        attempts += 1
        path: list[str] | None = None
        if attempts % 3 == 0:
            nodes = list(graph.nodes)
            rng.shuffle(nodes)
            for via in nodes[:12]:
                via = str(via)
                if via in {origin, destination} or via in set(oracle):
                    continue
                path = via_route(graph, origin, via, destination)
                if path:
                    break
        else:
            if not oracle_edges:
                continue
            block_size = rng.choice([1, 1, 2, 3])
            start = rng.randrange(0, max(1, len(oracle_edges) - block_size + 1))
            penalized = set(oracle_edges[start : start + block_size])
            penalty = rng.choice([8.0, 15.0, 30.0, 60.0])
            path = penalized_shortest_path(graph, origin, destination, penalized, penalty)

        if not path or len(path) < 2:
            continue
        key = tuple(path)
        if key in seen:
            continue
        length = _path_length_m(graph, path)
        if not math.isfinite(length) or length > max(oracle_length * 2.75, oracle_length + 2500):
            continue
        seen.add(key)
        out.append((f"alt_{len(out)}", path))

    return out[:num_candidates]


def render_rerank_task(task: dict[str, Any], candidates: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    graph = task_graph(task)
    west, south, east, north = task["task_bbox"]
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")

    for edge in task.get("graph", {}).get("edges", []):
        xs = [point[0] for point in edge["geometry"]]
        ys = [point[1] for point in edge["geometry"]]
        color, width, alpha = _edge_style(edge)
        ax.plot(xs, ys, color=color, linewidth=width, alpha=alpha, solid_capstyle="round", zorder=1)

    for candidate in candidates:
        geometry = geometry_for_osm_path(graph, [str(node) for node in candidate["route_nodes"]])
        color = candidate["color"]
        ax.plot(
            [point["lon"] for point in geometry],
            [point["lat"] for point in geometry],
            color=color,
            linewidth=3.2,
            alpha=0.72,
            solid_capstyle="round",
            zorder=4,
        )
        if len(geometry) >= 2:
            mid = geometry[len(geometry) // 2]
            ax.text(
                mid["lon"],
                mid["lat"],
                candidate["route_id"],
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="white",
                zorder=7,
                bbox={"boxstyle": "circle,pad=0.28", "facecolor": color, "edgecolor": "white", "linewidth": 1.1},
            )

    origin = task["origin"]
    dest = task["destination"]
    ax.scatter(origin["lon"], origin["lat"], s=190, color="#1664d9", edgecolor="white", linewidth=1.6, zorder=8)
    ax.scatter(dest["lon"], dest["lat"], s=190, color="#dc2626", edgecolor="white", linewidth=1.6, zorder=8)
    ax.text(origin["lon"], origin["lat"], "A", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=9)
    ax.text(dest["lon"], dest["lat"], "B", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def build_rerank_task(task: dict[str, Any], *, out_dir: Path, num_candidates: int, rng: random.Random) -> dict[str, Any]:
    graph = task_graph(task)
    paths = candidate_paths(task, num_candidates=num_candidates, rng=rng)
    if not paths:
        raise RuntimeError(f"could not build candidates for {task['task_id']}")

    candidates: list[dict[str, Any]] = []
    shuffled = []
    for kind, path in paths:
        metrics = route_score(task, graph, path)
        shuffled.append((kind, path, metrics))
    rng.shuffle(shuffled)
    for index, (kind, path, metrics) in enumerate(shuffled, start=1):
        route_id = f"R{index}"
        candidates.append(
            {
                "route_id": route_id,
                "color": ROUTE_COLORS[(index - 1) % len(ROUTE_COLORS)],
                "kind": kind,
                "is_oracle": kind == "oracle",
                "route_nodes": path,
                "hidden_metrics": metrics,
            }
        )

    best = max(candidates, key=lambda item: (float(item["hidden_metrics"]["score"]), item["is_oracle"]))
    image_path = out_dir / "rerank_maps" / task["task_id"] / "candidates.png"
    render_rerank_task(task, candidates, image_path)
    return {
        "task_id": task["task_id"],
        "difficulty": task.get("difficulty"),
        "base_task_id": task["task_id"],
        "image": str(image_path),
        "origin": task["origin"],
        "destination": task["destination"],
        "oracle_distance_m": task["oracle"]["distance_m"],
        "candidate_ids": [candidate["route_id"] for candidate in candidates],
        "best_route_id": best["route_id"],
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one-shot route-reranking tasks from flat OSM tasks.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)
    records = []
    for index, task in enumerate(iter_jsonl(args.tasks), start=1):
        if args.limit and index > args.limit:
            break
        records.append(build_rerank_task(task, out_dir=out_dir, num_candidates=args.num_candidates, rng=rng))
        if index % 50 == 0:
            print(f"built {index} rerank tasks", flush=True)
    write_jsonl(args.out, records)
    print(f"wrote {len(records)} rerank tasks to {args.out}")


if __name__ == "__main__":
    main()
