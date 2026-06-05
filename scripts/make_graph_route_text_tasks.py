#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tqdm import tqdm

from route_env.geometry import haversine_m
from route_env.io import iter_jsonl, write_jsonl


def bearing_deg(a: dict[str, float], b: dict[str, float]) -> float:
    lat1 = math.radians(a["lat"])
    lat2 = math.radians(b["lat"])
    dlon = math.radians(b["lon"] - a["lon"])
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def node_record(node_id: str, point: dict[str, Any], destination: dict[str, Any]) -> dict[str, Any]:
    p = {"lat": float(point["lat"]), "lon": float(point["lon"])}
    return {
        "id": str(int(node_id)),
        "lat": round(p["lat"], 7),
        "lon": round(p["lon"], 7),
        "straight_to_goal_m": round(haversine_m(p, destination), 1),
    }


def edge_record(edge: dict[str, Any], nodes: dict[str, Any]) -> dict[str, Any]:
    u = str(int(edge["u"]))
    v = str(int(edge["v"]))
    up = nodes[u]
    vp = nodes[v]
    highway = edge.get("highway", "")
    if isinstance(highway, list):
        highway = highway[0] if highway else ""
    return {
        "u": u,
        "v": v,
        "length_m": round(float(edge["length_m"]), 1),
        "bearing_deg": round(bearing_deg(up, vp), 1),
        "highway": highway,
    }


def graph_text_prompt(record: dict[str, Any], *, max_edges_in_prompt: int) -> str:
    graph = record["graph"]
    edge_lines = []
    for edge in graph["edges"][:max_edges_in_prompt]:
        edge_lines.append(
            f'{edge["u"]}->{edge["v"]} len={edge["length_m"]}m bearing={edge["bearing_deg"]} highway={edge["highway"]}'
        )
    omitted = max(0, len(graph["edges"]) - len(edge_lines))
    omitted_text = f"\n... {omitted} edges omitted from prompt preview." if omitted else ""
    return f"""You are a graph routing model.

Given a directed road graph, predict a short valid route from START to GOAL.
Use directed edges only. Prefer low total distance and avoid loops.
Return JSON only: {{"route_nodes":["node_id", "..."]}}

START: {record['start_node']}
GOAL: {record['goal_node']}

Each node has straight-line distance to goal in meters. Each edge is directed.
Graph summary:
nodes={len(graph['nodes'])}
edges={len(graph['edges'])}

Edges:
{chr(10).join(edge_lines)}{omitted_text}
"""


def convert_task(task: dict[str, Any], *, include_prompt: bool, max_edges_in_prompt: int) -> dict[str, Any]:
    start = str(int(task["origin"]["osm_id"]))
    goal = str(int(task["destination"]["osm_id"]))
    destination = {"lat": float(task["destination"]["lat"]), "lon": float(task["destination"]["lon"])}
    nodes = {
        str(int(node_id)): {"lat": float(point["lat"]), "lon": float(point["lon"])}
        for node_id, point in task["graph"]["nodes"].items()
    }
    record = {
        "task_id": task["task_id"],
        "city": task.get("city"),
        "difficulty": task.get("difficulty"),
        "start_node": start,
        "goal_node": goal,
        "graph": {
            "nodes": [node_record(node_id, point, destination) for node_id, point in nodes.items()],
            "edges": [edge_record(edge, nodes) for edge in task["graph"]["edges"]],
        },
        "target": {
            "route_nodes": [str(int(node)) for node in task["oracle"]["gold_osm_route"]],
            "turns": list(task["oracle"]["gold_turn_route"]),
            "distance_m": round(float(task["oracle"]["distance_m"]), 1),
        },
        "notes": {
            "inference_constraint": "The model should predict route_nodes using only graph.nodes and graph.edges.",
            "teacher": "target.route_nodes is the hidden Dijkstra/OSM shortest route used only for training/eval.",
        },
    }
    if include_prompt:
        record["prompt"] = graph_text_prompt(record, max_edges_in_prompt=max_edges_in_prompt)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--max-edges-in-prompt", type=int, default=300)
    args = parser.parse_args()

    records = []
    for task in tqdm(iter_jsonl(args.tasks), desc="graph-text"):
        records.append(
            convert_task(
                task,
                include_prompt=args.include_prompt,
                max_edges_in_prompt=args.max_edges_in_prompt,
            )
        )
        if args.limit is not None and len(records) >= args.limit:
            break
    write_jsonl(Path(args.out), records)
    print(f"wrote {len(records)} graph text route task(s) to {args.out}")


if __name__ == "__main__":
    main()
