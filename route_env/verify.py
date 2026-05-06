from __future__ import annotations

import math
from typing import Any

import networkx as nx

from route_env.geometry import (
    dedupe_consecutive,
    hausdorff_distance_m,
    haversine_m,
    latlon_to_lonlat,
    lonlat_to_latlon,
    mean_bidirectional_distance_m,
)


def task_graph(task: dict[str, Any]) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node_id, point in task["graph"]["nodes"].items():
        graph.add_node(node_id, lat=point["lat"], lon=point["lon"])
    for edge in task["graph"]["edges"]:
        graph.add_edge(edge["u"], edge["v"], length=float(edge["length_m"]), geometry=edge["geometry"])
    return graph


def _node_point(graph: nx.DiGraph, node_id: str) -> dict[str, float]:
    p = graph.nodes[node_id]
    return {"lat": float(p["lat"]), "lon": float(p["lon"])}


def _edge_points(graph: nx.DiGraph, u: str, v: str) -> list[dict[str, float]]:
    data = graph.edges[u, v]
    return [lonlat_to_latlon(p) for p in data.get("geometry", [])]


def geometry_for_osm_path(graph: nx.DiGraph, path: list[str]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for u, v in zip(path, path[1:]):
        segment = _edge_points(graph, u, v)
        if not segment:
            segment = [_node_point(graph, u), _node_point(graph, v)]
        if points:
            segment = segment[1:]
        points.extend(segment)
    if not points and path:
        points.append(_node_point(graph, path[0]))
    return points


def checkpoint_to_osm(task: dict[str, Any], label: str) -> str:
    return str(int(task["turn_checkpoints"][label]["osm_id"]))


def route_through_waypoints(graph: nx.DiGraph, waypoint_osm_ids: list[str]) -> tuple[bool, list[str], float, float]:
    expanded: list[str] = []
    total_length = 0.0
    max_segment_length = 0.0
    for u, v in zip(waypoint_osm_ids, waypoint_osm_ids[1:]):
        try:
            segment = nx.shortest_path(graph, u, v, weight="length")
            segment_length = float(nx.shortest_path_length(graph, u, v, weight="length"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False, [], math.inf, math.inf
        if expanded:
            segment = segment[1:]
        expanded.extend(segment)
        total_length += segment_length
        max_segment_length = max(max_segment_length, segment_length)
    if len(waypoint_osm_ids) == 1:
        expanded = waypoint_osm_ids[:]
    return True, expanded, total_length, max_segment_length


def verify_prediction(task: dict[str, Any], prediction_record: dict[str, Any]) -> dict[str, Any]:
    prediction = prediction_record.get("prediction", prediction_record)
    valid_schema = isinstance(prediction, dict) and isinstance(prediction.get("turns"), list)
    raw_turns = prediction.get("turns", []) if valid_schema else []
    turns = dedupe_consecutive([str(n) for n in raw_turns])
    checkpoints = set(task["turn_checkpoints"])
    unknown = [turn for turn in turns if turn not in checkpoints]
    known_turns = [turn for turn in turns if turn in checkpoints]

    graph = task_graph(task)
    origin_osm = str(int(task["origin"]["osm_id"]))
    destination_osm = str(int(task["destination"]["osm_id"]))
    waypoint_osm_ids = [origin_osm] + [checkpoint_to_osm(task, turn) for turn in known_turns] + [destination_osm]
    valid_route, expanded_path, agent_distance_m, max_gap_m = (
        route_through_waypoints(graph, waypoint_osm_ids) if not unknown else (False, [], math.inf, math.inf)
    )

    agent_geometry = geometry_for_osm_path(graph, expanded_path) if valid_route else []
    oracle_geometry = [lonlat_to_latlon(p) for p in task["oracle"]["geometry"]]
    oracle_distance = float(task["oracle"]["distance_m"])
    start_error_m = 0.0 if valid_route else math.inf
    end_error_m = 0.0 if valid_route else math.inf
    length_ratio = agent_distance_m / oracle_distance if valid_route and oracle_distance > 0 else math.inf
    hausdorff_m = hausdorff_distance_m(agent_geometry, oracle_geometry) if valid_route else math.inf
    mean_distance_m = mean_bidirectional_distance_m(agent_geometry, oracle_geometry) if valid_route else math.inf

    format_reward = 1.0 if valid_schema and not unknown else 0.0
    endpoint_reward = 1.0 if valid_route else 0.0
    route_validity_reward = 1.0 if valid_route else 0.0
    distance_ratio_reward = math.exp(-abs(math.log(length_ratio))) if math.isfinite(length_ratio) and length_ratio > 0 else 0.0
    similarity_reward = math.exp(-mean_distance_m / 100) if math.isfinite(mean_distance_m) else 0.0
    loop_penalty = max(0.0, 1.0 - (len(turns) - len(set(turns))) * 0.15)
    turn_count_penalty = math.exp(-max(0, len(turns) - 12) / 8)
    score = (
        0.10 * format_reward
        + 0.20 * endpoint_reward
        + 0.25 * route_validity_reward
        + 0.25 * distance_ratio_reward
        + 0.20 * similarity_reward
    ) * loop_penalty * turn_count_penalty

    return {
        "task_id": task["task_id"],
        "valid_schema": valid_schema,
        "valid_route": valid_route,
        "unknown_turn_count": len(unknown),
        "unknown_turns": unknown,
        "num_predicted_turns": len(turns),
        "num_expanded_nodes": len(expanded_path),
        "start_error_m": start_error_m,
        "end_error_m": end_error_m,
        "agent_distance_m": agent_distance_m,
        "oracle_distance_m": oracle_distance,
        "length_ratio": length_ratio,
        "max_segment_length_m": max_gap_m,
        "hausdorff_distance_m": hausdorff_m,
        "mean_route_distance_m": mean_distance_m,
        "format_reward": format_reward,
        "endpoint_reward": endpoint_reward,
        "route_validity_reward": route_validity_reward,
        "distance_ratio_reward": distance_ratio_reward,
        "similarity_reward": similarity_reward,
        "score": max(0.0, min(1.0, score)),
        "agent_osm_route_expanded": expanded_path,
        "agent_geometry": [latlon_to_lonlat(p) for p in agent_geometry],
    }
