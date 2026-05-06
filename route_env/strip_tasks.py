from __future__ import annotations

import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import networkx as nx

from route_env.geometry import bbox_from_points, haversine_m, latlon_to_lonlat, lonlat_to_latlon
from route_env.verify import geometry_for_osm_path, task_graph


def _node_point(graph: nx.DiGraph, node: str) -> dict[str, float]:
    data = graph.nodes[node]
    return {"lat": float(data["lat"]), "lon": float(data["lon"])}


def _route_length_m(graph: nx.DiGraph, route: list[str]) -> float:
    total = 0.0
    for u, v in zip(route, route[1:]):
        total += float(graph.edges[u, v].get("length", haversine_m(_node_point(graph, u), _node_point(graph, v))))
    return total


def _turn_angle_deg(a: dict[str, float], b: dict[str, float], c: dict[str, float]) -> float:
    v1 = (b["lon"] - a["lon"], b["lat"] - a["lat"])
    v2 = (c["lon"] - b["lon"], c["lat"] - b["lat"])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return abs(180 - math.degrees(math.acos(cosang)))


def _important_nodes(graph: nx.DiGraph, route: list[str], min_angle_deg: float = 30) -> list[str]:
    if len(route) <= 2:
        return route[:]
    important = [route[0]]
    for prev_node, node, next_node in zip(route, route[1:], route[2:]):
        if _turn_angle_deg(_node_point(graph, prev_node), _node_point(graph, node), _node_point(graph, next_node)) >= min_angle_deg:
            important.append(node)
        elif graph.out_degree(node) + graph.in_degree(node) >= 4:
            important.append(node)
    important.append(route[-1])
    return list(dict.fromkeys(important))


def _nodes_in_bbox(graph: nx.DiGraph, bbox: list[float]) -> list[str]:
    west, south, east, north = bbox
    return [
        str(node)
        for node, data in graph.nodes(data=True)
        if west <= float(data["lon"]) <= east and south <= float(data["lat"]) <= north
    ]


def _edge_records(task: dict[str, Any], crop_nodes: set[str]) -> list[dict[str, Any]]:
    return [
        deepcopy(edge)
        for edge in task["graph"]["edges"]
        if str(edge["u"]) in crop_nodes and str(edge["v"]) in crop_nodes
    ]


def _select_segment_checkpoints(
    graph: nx.DiGraph,
    route: list[str],
    crop_nodes: list[str],
    max_checkpoints: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    endpoint_nodes = {route[0], route[-1]}
    route_important = [node for node in _important_nodes(graph, route)[1:-1] if node not in endpoint_nodes]
    selected = route_important[:max_checkpoints]
    selected_set = set(selected)
    route_points = [_node_point(graph, node) for node in route]

    def dist_to_route(node: str) -> float:
        point = _node_point(graph, node)
        return min(haversine_m(point, route_point) for route_point in route_points)

    candidates = [
        node
        for node in crop_nodes
        if node not in selected_set
        and node not in endpoint_nodes
        and graph.out_degree(node) + graph.in_degree(node) >= 3
    ]
    candidates.sort(key=dist_to_route)
    near = candidates[: max(max_checkpoints * 4, 40)]
    rng.shuffle(near)
    for node in near:
        if len(selected) >= max_checkpoints:
            break
        selected.append(node)
    return selected, route_important


def split_route_by_distance(graph: nx.DiGraph, route: list[str], target_distance_m: float) -> list[list[str]]:
    if len(route) <= 2:
        return [route]
    segments: list[list[str]] = []
    current = [route[0]]
    current_length = 0.0
    for u, v in zip(route, route[1:]):
        current.append(v)
        current_length += float(graph.edges[u, v].get("length", haversine_m(_node_point(graph, u), _node_point(graph, v))))
        if current_length >= target_distance_m and len(current) >= 3 and v != route[-1]:
            segments.append(current)
            current = [v]
            current_length = 0.0
    if current:
        current_is_too_short = _route_length_m(graph, current) < target_distance_m * 0.45
        if segments and (len(current) <= 2 or current_is_too_short):
            segments[-1].extend(current[1:])
        else:
            segments.append(current)
    return segments


def _build_segment_task(
    parent: dict[str, Any],
    graph: nx.DiGraph,
    segment_id: str,
    route: list[str],
    *,
    max_checkpoints: int,
    margin_m: float,
    rng: random.Random,
) -> dict[str, Any]:
    oracle_points = geometry_for_osm_path(graph, route)
    task_bbox = bbox_from_points(oracle_points, margin_m=margin_m)
    crop_nodes = _nodes_in_bbox(graph, task_bbox)
    checkpoint_nodes, important_nodes = _select_segment_checkpoints(graph, route, crop_nodes, max_checkpoints, rng)
    label_order = checkpoint_nodes[:]
    rng.shuffle(label_order)
    labels = {node: f"T{i + 1:02d}" for i, node in enumerate(label_order)}
    checkpoints = {
        labels[node]: {
            "lat": _node_point(graph, node)["lat"],
            "lon": _node_point(graph, node)["lon"],
            "osm_id": int(node),
        }
        for node in label_order
    }
    gold_turn_route = [labels[node] for node in important_nodes[1:-1] if node in labels]
    crop_set = set(crop_nodes)
    graph_nodes = {
        node: {
            "lat": _node_point(graph, node)["lat"],
            "lon": _node_point(graph, node)["lon"],
        }
        for node in crop_nodes
    }

    return {
        "task_id": f"{parent['task_id']}_{segment_id.lower()}",
        "segment_id": segment_id,
        "city": parent["city"],
        "mode": parent.get("mode", "drive"),
        "network_type": parent.get("network_type", "drive"),
        "bbox": task_bbox,
        "task_bbox": task_bbox,
        "origin": {**_node_point(graph, route[0]), "osm_id": int(route[0]), "label": "A"},
        "destination": {**_node_point(graph, route[-1]), "osm_id": int(route[-1]), "label": "B"},
        "images": {"map": f"data/experiments/manual/maps/{parent['task_id']}/{segment_id}.png"},
        "turn_checkpoints": checkpoints,
        "graph": {
            "nodes": graph_nodes,
            "edges": _edge_records(parent, crop_set),
        },
        "oracle": {
            "provider": "osmnx_drive_segment",
            "distance_m": _route_length_m(graph, route),
            "geometry": [latlon_to_lonlat(point) for point in oracle_points],
            "gold_turn_route": gold_turn_route,
            "gold_osm_route": [int(node) for node in route],
            "turn_count": len(gold_turn_route),
        },
        "prompt": (
            f"Segment {segment_id}: trace a valid driving route from blue A to red B "
            "using sparse local turn checkpoints."
        ),
    }


def make_route_strip_task(
    parent: dict[str, Any],
    *,
    target_segment_distance_m: float = 2500,
    max_segment_checkpoints: int = 32,
    segment_margin_m: float = 260,
    seed: int = 7,
) -> dict[str, Any]:
    graph = task_graph(parent)
    route = [str(int(node)) for node in parent["oracle"]["gold_osm_route"]]
    route_segments = split_route_by_distance(graph, route, target_segment_distance_m)
    rng = random.Random(seed)
    segments = [
        _build_segment_task(
            parent,
            graph,
            f"S{i + 1:02d}",
            segment_route,
            max_checkpoints=max_segment_checkpoints,
            margin_m=segment_margin_m,
            rng=rng,
        )
        for i, segment_route in enumerate(route_segments)
    ]

    return {
        "task_id": f"{parent['task_id']}_strip",
        "source_task_id": parent["task_id"],
        "task_type": "route_strip",
        "city": parent["city"],
        "mode": parent.get("mode", "drive"),
        "network_type": parent.get("network_type", "drive"),
        "bbox": parent["bbox"],
        "task_bbox": parent["task_bbox"],
        "origin": parent["origin"],
        "destination": parent["destination"],
        "images": {
            "overview": f"data/experiments/manual/maps/{parent['task_id']}_strip/overview.png",
            "segments": [
                f"data/experiments/manual/maps/{parent['task_id']}_strip/{segment['segment_id'].lower()}.png"
                for segment in segments
            ],
        },
        "segments": segments,
        "graph": parent["graph"],
        "oracle": {
            "provider": "osmnx_drive_route_strip",
            "distance_m": parent["oracle"]["distance_m"],
            "geometry": parent["oracle"]["geometry"],
            "gold_osm_route": parent["oracle"]["gold_osm_route"],
            "segment_gold_routes": {
                segment["segment_id"]: segment["oracle"]["gold_turn_route"]
                for segment in segments
            },
            "segment_count": len(segments),
        },
        "render": {
            "overview": {"grid": {"cols": 6, "rows": 4}},
            "segments": [
                {
                    "segment_id": segment["segment_id"],
                    "grid": {"cols": 4, "rows": 4},
                    "bbox": segment["task_bbox"],
                }
                for segment in segments
            ],
        },
        "prompt": "Use the overview and local segment maps to return per-segment driving turns.",
    }


def update_route_strip_image_paths(task: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    task["images"]["overview"] = str(out_dir / task["task_id"] / "overview.png")
    segment_paths = []
    for segment in task["segments"]:
        path = out_dir / task["task_id"] / f"{segment['segment_id'].lower()}.png"
        segment["images"]["map"] = str(path)
        segment_paths.append(str(path))
    task["images"]["segments"] = segment_paths
    return task
