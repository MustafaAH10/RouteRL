from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import networkx as nx

from route_env.geometry import (
    haversine_m,
    latlon_to_lonlat,
    lonlat_to_latlon,
    mean_bidirectional_distance_m,
)
from route_env.verify import checkpoint_alignment_reward, geometry_for_osm_path, task_graph


@dataclass(frozen=True)
class GraphRouteResult:
    policy: str
    route_nodes: list[str]
    turns: list[str]
    metrics: dict[str, Any]
    diagnostics: dict[str, Any]


def node_point(graph: nx.DiGraph, node_id: str) -> dict[str, float]:
    node = graph.nodes[str(node_id)]
    return {"lat": float(node["lat"]), "lon": float(node["lon"])}


def edge_length(graph: nx.DiGraph, u: str, v: str) -> float:
    return float(graph.edges[str(u), str(v)].get("length", 0.0))


def path_length_m(graph: nx.DiGraph, path: list[str]) -> float:
    return sum(edge_length(graph, u, v) for u, v in zip(path, path[1:]))


def dedupe_consecutive(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def checkpoint_by_node(task: dict[str, Any]) -> dict[str, str]:
    return {str(int(point["osm_id"])): label for label, point in task.get("turn_checkpoints", {}).items()}


def turns_for_path(task: dict[str, Any], path: list[str]) -> list[str]:
    by_node = checkpoint_by_node(task)
    return dedupe_consecutive([by_node[node] for node in path if node in by_node])


def direct_route_metrics(task: dict[str, Any], route_nodes: list[str], *, policy: str) -> dict[str, Any]:
    graph = task_graph(task)
    destination = str(int(task["destination"]["osm_id"]))
    valid_route = len(route_nodes) >= 2 and route_nodes[-1] == destination
    agent_geometry = geometry_for_osm_path(graph, route_nodes) if len(route_nodes) >= 2 else []
    oracle_geometry = [lonlat_to_latlon(point) for point in task["oracle"]["geometry"]]
    agent_distance = path_length_m(graph, route_nodes) if len(route_nodes) >= 2 else math.inf
    oracle_distance = float(task["oracle"]["distance_m"])
    length_ratio = agent_distance / oracle_distance if valid_route and oracle_distance > 0 else math.inf
    mean_distance = mean_bidirectional_distance_m(agent_geometry, oracle_geometry) if valid_route else math.inf
    observed_turns = turns_for_path(task, route_nodes)
    gold_turns = list(task.get("oracle", {}).get("gold_turn_route", []))
    checkpoint_reward, coverage, precision, order = checkpoint_alignment_reward(observed_turns, gold_turns)
    distance_ratio_reward = (
        math.exp(-abs(math.log(length_ratio))) if math.isfinite(length_ratio) and length_ratio > 0 else 0.0
    )
    similarity_reward = math.exp(-mean_distance / 100) if math.isfinite(mean_distance) else 0.0
    endpoint_reward = 1.0 if valid_route else 0.0
    loop_count = len(route_nodes) - len(set(route_nodes))
    loop_penalty = max(0.0, 1.0 - 0.05 * loop_count)
    score = (
        0.25 * endpoint_reward
        + 0.25 * distance_ratio_reward
        + 0.20 * similarity_reward
        + 0.30 * checkpoint_reward
    ) * loop_penalty
    return {
        "task_id": task["task_id"],
        "policy": policy,
        "valid_route": valid_route,
        "num_route_nodes": len(route_nodes),
        "num_predicted_turns": len(observed_turns),
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
        "agent_osm_route_expanded": route_nodes,
        "agent_geometry": [latlon_to_lonlat(point) for point in agent_geometry],
    }


def route_result(
    task: dict[str, Any],
    *,
    policy: str,
    route_nodes: list[str],
    diagnostics: dict[str, Any] | None = None,
) -> GraphRouteResult:
    return GraphRouteResult(
        policy=policy,
        route_nodes=route_nodes,
        turns=turns_for_path(task, route_nodes),
        metrics=direct_route_metrics(task, route_nodes, policy=policy),
        diagnostics=diagnostics or {},
    )


def _origin_dest(task: dict[str, Any]) -> tuple[str, str]:
    return str(int(task["origin"]["osm_id"])), str(int(task["destination"]["osm_id"]))


def local_greedy_route(
    task: dict[str, Any],
    *,
    max_steps: int = 512,
    edge_weight: float = 0.35,
    dest_weight: float = 1.0,
    progress_weight: float = 2.0,
    revisit_penalty_m: float = 2_500.0,
) -> GraphRouteResult:
    graph = task_graph(task)
    origin, destination = _origin_dest(task)
    dest_point = node_point(graph, destination)
    current = origin
    path = [origin]
    visited = {origin}
    dead_ends = 0

    for _ in range(max_steps):
        if current == destination:
            break
        current_dist = haversine_m(node_point(graph, current), dest_point)
        successors = [str(node) for node in graph.successors(current)]
        if not successors:
            dead_ends += 1
            break

        def score(successor: str) -> tuple[float, float, str]:
            succ_dist = haversine_m(node_point(graph, successor), dest_point)
            progress_penalty = max(0.0, succ_dist - current_dist)
            revisit = revisit_penalty_m if successor in visited else 0.0
            value = edge_weight * edge_length(graph, current, successor)
            value += dest_weight * succ_dist
            value += progress_weight * progress_penalty
            value += revisit
            return value, succ_dist, successor

        nxt = min(successors, key=score)
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return route_result(
        task,
        policy="local_greedy",
        route_nodes=path,
        diagnostics={"max_steps": max_steps, "dead_ends": dead_ends},
    )


@dataclass(order=True)
class _BeamItem:
    rank_score: float
    length_m: float
    tie: int
    path: list[str]


def heuristic_beam_route(
    task: dict[str, Any],
    *,
    beam_width: int = 32,
    max_expansions: int = 2_000,
    heuristic_weight: float = 1.25,
    loop_penalty_m: float = 5_000.0,
) -> GraphRouteResult:
    graph = task_graph(task)
    origin, destination = _origin_dest(task)
    dest_point = node_point(graph, destination)
    counter = 0

    def item_for(path: list[str], length_m: float) -> _BeamItem:
        nonlocal counter
        counter += 1
        last = path[-1]
        heuristic = haversine_m(node_point(graph, last), dest_point)
        loop_count = len(path) - len(set(path))
        rank_score = length_m + heuristic_weight * heuristic + loop_penalty_m * loop_count
        return _BeamItem(rank_score=rank_score, length_m=length_m, tie=counter, path=path)

    beam = [item_for([origin], 0.0)]
    completed: list[_BeamItem] = []
    expansions = 0
    pruned_loops = 0

    while beam and expansions < max_expansions:
        next_items: list[_BeamItem] = []
        for item in beam:
            last = item.path[-1]
            if last == destination:
                completed.append(item)
                continue
            for successor in graph.successors(last):
                successor = str(successor)
                if successor in item.path[:-1]:
                    pruned_loops += 1
                    continue
                path = item.path + [successor]
                length_m = item.length_m + edge_length(graph, last, successor)
                next_items.append(item_for(path, length_m))
                expansions += 1
                if expansions >= max_expansions:
                    break
            if expansions >= max_expansions:
                break
        if completed:
            best = min(completed, key=lambda item: item.length_m)
            return route_result(
                task,
                policy="heuristic_beam",
                route_nodes=best.path,
                diagnostics={
                    "beam_width": beam_width,
                    "max_expansions": max_expansions,
                    "expansions": expansions,
                    "completed": len(completed),
                    "pruned_loops": pruned_loops,
                },
            )
        beam = heapq.nsmallest(beam_width, next_items)

    fallback = min(beam, key=lambda item: item.rank_score) if beam else item_for([origin], 0.0)
    return route_result(
        task,
        policy="heuristic_beam",
        route_nodes=fallback.path,
        diagnostics={
            "beam_width": beam_width,
            "max_expansions": max_expansions,
            "expansions": expansions,
            "completed": len(completed),
            "pruned_loops": pruned_loops,
            "fallback": True,
        },
    )


def hill_climb_beam_route(
    task: dict[str, Any],
    *,
    lookahead_depth: int = 4,
    branch_width: int = 6,
    max_steps: int = 512,
    heuristic_weight: float = 1.15,
    revisit_penalty_m: float = 4_000.0,
) -> GraphRouteResult:
    graph = task_graph(task)
    origin, destination = _origin_dest(task)
    dest_point = node_point(graph, destination)
    current = origin
    path = [origin]
    visited = {origin}
    decisions = 0

    def rollout_score(prefix: list[str], prefix_length: float) -> float:
        last = prefix[-1]
        loop_count = len(prefix) - len(set(prefix))
        return prefix_length + heuristic_weight * haversine_m(node_point(graph, last), dest_point) + loop_count * revisit_penalty_m

    for _ in range(max_steps):
        if current == destination:
            break
        successors = [str(node) for node in graph.successors(current)]
        if not successors:
            break
        candidate_items: list[tuple[float, list[str], float]] = []
        for successor in successors:
            first_path = [current, successor]
            first_length = edge_length(graph, current, successor)
            beam: list[tuple[float, list[str], float]] = [(rollout_score(first_path, first_length), first_path, first_length)]
            for _depth in range(max(0, lookahead_depth - 1)):
                expanded: list[tuple[float, list[str], float]] = []
                for _score, candidate_path, length_m in beam:
                    last = candidate_path[-1]
                    if last == destination:
                        expanded.append((_score, candidate_path, length_m))
                        continue
                    for nxt in graph.successors(last):
                        nxt = str(nxt)
                        new_path = candidate_path + [nxt]
                        new_length = length_m + edge_length(graph, last, nxt)
                        expanded.append((rollout_score(new_path, new_length), new_path, new_length))
                if not expanded:
                    break
                beam = heapq.nsmallest(branch_width, expanded, key=lambda item: item[0])
            candidate_items.extend(beam)
        candidate_items.sort(key=lambda item: item[0])
        chosen = candidate_items[0][1][1]
        if chosen in visited and len(successors) > 1:
            nonvisited = [item for item in candidate_items if item[1][1] not in visited]
            if nonvisited:
                chosen = nonvisited[0][1][1]
        path.append(chosen)
        visited.add(chosen)
        current = chosen
        decisions += 1

    return route_result(
        task,
        policy="hill_climb_beam",
        route_nodes=path,
        diagnostics={
            "lookahead_depth": lookahead_depth,
            "branch_width": branch_width,
            "max_steps": max_steps,
            "decisions": decisions,
        },
    )


POLICIES = {
    "local_greedy": local_greedy_route,
    "heuristic_beam": heuristic_beam_route,
    "hill_climb_beam": hill_climb_beam_route,
}
