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


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for left in a:
        current = [0]
        for j, right in enumerate(b, start=1):
            if left == right:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def checkpoint_alignment_reward(turns: list[str], gold_turns: list[str]) -> tuple[float, float, float, float]:
    if not gold_turns:
        reward = 1.0 if not turns else 0.5
        return reward, reward, reward, reward
    unique_turns = set(turns)
    gold_set = set(gold_turns)
    overlap = unique_turns & gold_set
    coverage = len(overlap) / len(gold_set)
    precision = len(overlap) / len(unique_turns) if unique_turns else 0.0
    order = _lcs_length(turns, gold_turns) / len(gold_turns)
    reward = 0.45 * coverage + 0.35 * order + 0.20 * precision
    return reward, coverage, precision, order


def verify_flat_prediction(task: dict[str, Any], prediction_record: dict[str, Any]) -> dict[str, Any]:
    prediction = prediction_record.get("prediction", prediction_record)
    raw_turns = prediction.get("turns") if isinstance(prediction, dict) else None
    valid_turn_list = isinstance(raw_turns, list) and all(isinstance(turn, str) for turn in raw_turns)
    valid_schema = isinstance(prediction, dict) and valid_turn_list
    turns = dedupe_consecutive(raw_turns if valid_turn_list else [])
    checkpoints = set(task["turn_checkpoints"])
    unknown = [turn for turn in turns if turn not in checkpoints]
    known_turns = [turn for turn in turns if turn in checkpoints]
    gold_turns = list(task.get("oracle", {}).get("gold_turn_route", []))
    has_required_checkpoint_evidence = bool(known_turns) or not gold_turns

    graph = task_graph(task)
    origin_osm = str(int(task["origin"]["osm_id"]))
    destination_osm = str(int(task["destination"]["osm_id"]))
    gold_waypoint_osm_ids = [origin_osm] + [checkpoint_to_osm(task, turn) for turn in gold_turns] + [destination_osm]
    _, _, _, oracle_checkpoint_max_gap_m = route_through_waypoints(graph, gold_waypoint_osm_ids)
    waypoint_osm_ids = [origin_osm] + [checkpoint_to_osm(task, turn) for turn in known_turns] + [destination_osm]
    valid_route, expanded_path, agent_distance_m, max_gap_m = (
        route_through_waypoints(graph, waypoint_osm_ids)
        if valid_turn_list and not unknown and has_required_checkpoint_evidence
        else (False, [], math.inf, math.inf)
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
    checkpoint_reward, checkpoint_coverage, checkpoint_precision, checkpoint_order = checkpoint_alignment_reward(
        turns, gold_turns
    )
    expected_gap_m = max(250.0, oracle_checkpoint_max_gap_m * 1.25 if math.isfinite(oracle_checkpoint_max_gap_m) else 0.0)
    max_gap_reward = (
        math.exp(-max(0.0, max_gap_m - expected_gap_m) / expected_gap_m)
        if math.isfinite(max_gap_m)
        else 0.0
    )
    loop_penalty = max(0.0, 1.0 - (len(turns) - len(set(turns))) * 0.15)
    allowed_turns = max(12, len(gold_turns) + 3)
    turn_count_penalty = math.exp(-max(0, len(turns) - allowed_turns) / 8)
    score = (
        0.05 * format_reward
        + 0.10 * endpoint_reward
        + 0.10 * route_validity_reward
        + 0.20 * distance_ratio_reward
        + 0.15 * similarity_reward
        + 0.35 * checkpoint_reward
        + 0.05 * max_gap_reward
    ) * loop_penalty * turn_count_penalty

    return {
        "task_id": task["task_id"],
        "valid_schema": valid_schema,
        "valid_turn_list": valid_turn_list,
        "valid_route": valid_route,
        "unknown_turn_count": len(unknown),
        "unknown_turns": unknown,
        "num_predicted_turns": len(turns),
        "num_gold_turns": len(gold_turns),
        "num_expanded_nodes": len(expanded_path),
        "start_error_m": start_error_m,
        "end_error_m": end_error_m,
        "agent_distance_m": agent_distance_m,
        "oracle_distance_m": oracle_distance,
        "length_ratio": length_ratio,
        "max_segment_length_m": max_gap_m,
        "expected_max_segment_length_m": expected_gap_m,
        "hausdorff_distance_m": hausdorff_m,
        "mean_route_distance_m": mean_distance_m,
        "format_reward": format_reward,
        "endpoint_reward": endpoint_reward,
        "route_validity_reward": route_validity_reward,
        "distance_ratio_reward": distance_ratio_reward,
        "similarity_reward": similarity_reward,
        "checkpoint_reward": checkpoint_reward,
        "checkpoint_coverage": checkpoint_coverage,
        "checkpoint_precision": checkpoint_precision,
        "checkpoint_order": checkpoint_order,
        "max_gap_reward": max_gap_reward,
        "loop_penalty": loop_penalty,
        "turn_count_penalty": turn_count_penalty,
        "score": max(0.0, min(1.0, score)),
        "agent_osm_route_expanded": expanded_path,
        "agent_geometry": [latlon_to_lonlat(p) for p in agent_geometry],
    }


def verify_strip_prediction(task: dict[str, Any], prediction_record: dict[str, Any]) -> dict[str, Any]:
    prediction = prediction_record.get("prediction", prediction_record)
    raw_segments = prediction.get("segments") if isinstance(prediction, dict) else None
    valid_segment_list = isinstance(raw_segments, list) and all(isinstance(segment, dict) for segment in raw_segments)
    by_segment = {
        str(segment.get("segment_id")): segment
        for segment in raw_segments
        if isinstance(segment, dict) and isinstance(segment.get("segment_id"), str)
    } if valid_segment_list else {}
    segment_results = []
    stitched_geometry: list[dict[str, float]] = []
    stitched_route: list[str] = []
    all_valid = valid_segment_list
    for segment_task in task.get("segments", []):
        segment_id = segment_task["segment_id"]
        segment_prediction = by_segment.get(segment_id, {"turns": []})
        result = verify_flat_prediction(segment_task, {"task_id": task["task_id"], "prediction": segment_prediction})
        segment_results.append(result)
        all_valid = all_valid and result["valid_route"]
        if result["agent_geometry"]:
            points = [lonlat_to_latlon(point) for point in result["agent_geometry"]]
            if stitched_geometry:
                points = points[1:]
            stitched_geometry.extend(points)
        expanded = result.get("agent_osm_route_expanded", [])
        if expanded:
            if stitched_route:
                expanded = expanded[1:]
            stitched_route.extend(expanded)

    oracle_geometry = [lonlat_to_latlon(point) for point in task["oracle"]["geometry"]]
    oracle_distance = float(task["oracle"]["distance_m"])
    agent_distance_m = sum(
        result["agent_distance_m"]
        for result in segment_results
        if isinstance(result.get("agent_distance_m"), int | float) and math.isfinite(result["agent_distance_m"])
    )
    length_ratio = agent_distance_m / oracle_distance if all_valid and oracle_distance > 0 else math.inf
    mean_distance_m = mean_bidirectional_distance_m(stitched_geometry, oracle_geometry) if all_valid else math.inf
    distance_ratio_reward = math.exp(-abs(math.log(length_ratio))) if math.isfinite(length_ratio) and length_ratio > 0 else 0.0
    similarity_reward = math.exp(-mean_distance_m / 100) if math.isfinite(mean_distance_m) else 0.0
    segment_score = sum(result["score"] for result in segment_results) / len(segment_results) if segment_results else 0.0
    route_validity_reward = 1.0 if all_valid else 0.0
    score = 0.70 * segment_score + 0.15 * route_validity_reward + 0.10 * distance_ratio_reward + 0.05 * similarity_reward
    unknown_turns = [
        {"segment_id": result["task_id"].rsplit("_", 1)[-1].upper(), "turns": result["unknown_turns"]}
        for result in segment_results
        if result["unknown_turns"]
    ]

    return {
        "task_id": task["task_id"],
        "task_type": "route_strip",
        "valid_schema": isinstance(prediction, dict) and valid_segment_list,
        "valid_turn_list": valid_segment_list,
        "valid_route": all_valid,
        "unknown_turn_count": sum(result["unknown_turn_count"] for result in segment_results),
        "unknown_turns": unknown_turns,
        "num_predicted_turns": sum(result["num_predicted_turns"] for result in segment_results),
        "num_gold_turns": sum(result["num_gold_turns"] for result in segment_results),
        "num_expanded_nodes": len(stitched_route),
        "start_error_m": 0.0 if all_valid else math.inf,
        "end_error_m": 0.0 if all_valid else math.inf,
        "agent_distance_m": agent_distance_m if all_valid else math.inf,
        "oracle_distance_m": oracle_distance,
        "length_ratio": length_ratio,
        "max_segment_length_m": max((result["max_segment_length_m"] for result in segment_results), default=math.inf),
        "expected_max_segment_length_m": max((result["expected_max_segment_length_m"] for result in segment_results), default=math.inf),
        "hausdorff_distance_m": hausdorff_distance_m(stitched_geometry, oracle_geometry) if all_valid else math.inf,
        "mean_route_distance_m": mean_distance_m,
        "format_reward": 1.0 if isinstance(prediction, dict) and valid_segment_list and not unknown_turns else 0.0,
        "endpoint_reward": route_validity_reward,
        "route_validity_reward": route_validity_reward,
        "distance_ratio_reward": distance_ratio_reward,
        "similarity_reward": similarity_reward,
        "checkpoint_reward": segment_score,
        "checkpoint_coverage": sum(result["checkpoint_coverage"] for result in segment_results) / len(segment_results) if segment_results else 0.0,
        "checkpoint_precision": sum(result["checkpoint_precision"] for result in segment_results) / len(segment_results) if segment_results else 0.0,
        "checkpoint_order": sum(result["checkpoint_order"] for result in segment_results) / len(segment_results) if segment_results else 0.0,
        "max_gap_reward": sum(result["max_gap_reward"] for result in segment_results) / len(segment_results) if segment_results else 0.0,
        "loop_penalty": min((result["loop_penalty"] for result in segment_results), default=0.0),
        "turn_count_penalty": min((result["turn_count_penalty"] for result in segment_results), default=0.0),
        "score": max(0.0, min(1.0, score)),
        "agent_osm_route_expanded": stitched_route,
        "agent_geometry": [latlon_to_lonlat(point) for point in stitched_geometry],
        "segment_results": segment_results,
    }


def verify_prediction(task: dict[str, Any], prediction_record: dict[str, Any]) -> dict[str, Any]:
    if task.get("task_type") == "route_strip":
        return verify_strip_prediction(task, prediction_record)
    return verify_flat_prediction(task, prediction_record)
