from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

from route_env.geometry import haversine_m
from route_env.graph_route_approx import (
    GraphRouteResult,
    edge_length,
    node_point,
    path_length_m,
    route_result,
)
from route_env.verify import task_graph


Edge = tuple[str, str]


@dataclass(order=True)
class _CandidateBeamItem:
    rank_score: float
    penalized_length_m: float
    actual_length_m: float
    tie: int
    path: list[str]


def candidate_beam_route(
    task: dict[str, Any],
    *,
    edge_penalties_m: dict[Edge, float] | None = None,
    blocked_edges: set[Edge] | None = None,
    beam_width: int = 64,
    max_expansions: int = 12_000,
    heuristic_weight: float = 1.25,
    loop_penalty_m: float = 5_000.0,
    policy_name: str = "candidate_beam",
) -> GraphRouteResult:
    """Find one plausible route using beam search with optional diversity penalties.

    The penalties affect search ranking only. The returned route distance and
    metrics use true graph edge lengths.
    """

    graph = task_graph(task)
    origin = str(int(task["origin"]["osm_id"]))
    destination = str(int(task["destination"]["osm_id"]))
    dest_point = node_point(graph, destination)
    penalties = edge_penalties_m or {}
    blocked = blocked_edges or set()
    counter = 0

    def item_for(path: list[str], actual_length_m: float, penalized_length_m: float) -> _CandidateBeamItem:
        nonlocal counter
        counter += 1
        last = path[-1]
        heuristic = haversine_m(node_point(graph, last), dest_point)
        loop_count = len(path) - len(set(path))
        rank_score = penalized_length_m + heuristic_weight * heuristic + loop_penalty_m * loop_count
        return _CandidateBeamItem(
            rank_score=rank_score,
            penalized_length_m=penalized_length_m,
            actual_length_m=actual_length_m,
            tie=counter,
            path=path,
        )

    beam = [item_for([origin], 0.0, 0.0)]
    completed: list[_CandidateBeamItem] = []
    expansions = 0
    pruned_loops = 0

    while beam and expansions < max_expansions:
        next_items: list[_CandidateBeamItem] = []
        for item in beam:
            last = item.path[-1]
            if last == destination:
                completed.append(item)
                continue
            for successor in graph.successors(last):
                successor = str(successor)
                if (last, successor) in blocked:
                    continue
                if successor in item.path[:-1]:
                    pruned_loops += 1
                    continue
                length = edge_length(graph, last, successor)
                penalty = penalties.get((last, successor), 0.0)
                path = item.path + [successor]
                next_items.append(
                    item_for(
                        path,
                        actual_length_m=item.actual_length_m + length,
                        penalized_length_m=item.penalized_length_m + length + penalty,
                    )
                )
                expansions += 1
                if expansions >= max_expansions:
                    break
            if expansions >= max_expansions:
                break
        if completed:
            best = min(completed, key=lambda item: item.penalized_length_m)
            return route_result(
                task,
                policy=policy_name,
                route_nodes=best.path,
                diagnostics={
                    "beam_width": beam_width,
                    "max_expansions": max_expansions,
                    "expansions": expansions,
                    "completed": len(completed),
                    "pruned_loops": pruned_loops,
                    "penalized_length_m": best.penalized_length_m,
                },
            )
        beam = heapq.nsmallest(beam_width, next_items)

    fallback = min(beam, key=lambda item: item.rank_score) if beam else item_for([origin], 0.0, 0.0)
    return route_result(
        task,
        policy=policy_name,
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


def edge_set(path: list[str]) -> set[Edge]:
    return {(u, v) for u, v in zip(path, path[1:])}


def route_signature(path: list[str]) -> tuple[str, ...]:
    return tuple(path)


def generate_route_candidates(
    task: dict[str, Any],
    *,
    num_candidates: int = 8,
    beam_width: int = 64,
    max_expansions: int = 12_000,
    diversity_penalty_m: float = 900.0,
    max_attempts: int | None = None,
) -> list[GraphRouteResult]:
    """Generate diverse valid route candidates using penalized heuristic beams."""

    attempts = max_attempts or max(num_candidates * 5, 24)
    heuristic_weights = [1.00, 1.15, 1.25, 1.40, 1.65, 0.85, 1.90, 0.70]
    loop_penalties = [5_000.0, 8_000.0, 3_000.0, 12_000.0]
    candidates: list[GraphRouteResult] = []
    seen: set[tuple[str, ...]] = set()
    edge_penalties: dict[Edge, float] = {}

    def maybe_add(result: GraphRouteResult) -> bool:
        signature = route_signature(result.route_nodes)
        valid = bool(result.metrics.get("valid_route"))
        if valid and signature not in seen:
            candidates.append(result)
            seen.add(signature)
            return True
        return False

    for attempt in range(attempts):
        heuristic_weight = heuristic_weights[attempt % len(heuristic_weights)]
        loop_penalty_m = loop_penalties[(attempt // len(heuristic_weights)) % len(loop_penalties)]
        result = candidate_beam_route(
            task,
            edge_penalties_m=edge_penalties,
            beam_width=beam_width,
            max_expansions=max_expansions,
            heuristic_weight=heuristic_weight,
            loop_penalty_m=loop_penalty_m,
            policy_name=f"candidate_beam_h{heuristic_weight:g}",
        )
        if maybe_add(result):
            if len(candidates) >= num_candidates:
                break

        multiplier = 1.0 + 0.35 * attempt
        for edge in edge_set(result.route_nodes):
            edge_penalties[edge] = edge_penalties.get(edge, 0.0) + diversity_penalty_m * multiplier

        # If soft penalties keep finding the same route, hard-block one edge
        # from existing valid routes to discover alternate corridors.
        for base in list(candidates):
            edges = list(edge_set(base.route_nodes))
            if not edges:
                continue
            stride = max(1, len(edges) // 8)
            for edge in edges[::stride]:
                blocked_result = candidate_beam_route(
                    task,
                    edge_penalties_m=edge_penalties,
                    blocked_edges={edge},
                    beam_width=beam_width,
                    max_expansions=max_expansions,
                    heuristic_weight=heuristic_weight,
                    loop_penalty_m=loop_penalty_m,
                    policy_name=f"candidate_beam_blocked_h{heuristic_weight:g}",
                )
                maybe_add(blocked_result)
                if len(candidates) >= num_candidates:
                    break
            if len(candidates) >= num_candidates:
                break
        if len(candidates) >= num_candidates:
            break

    candidates.sort(key=lambda candidate: path_length_m(task_graph(task), candidate.route_nodes))
    return candidates[:num_candidates]


def candidate_record(candidate: GraphRouteResult, index: int) -> dict[str, Any]:
    metrics = candidate.metrics
    return {
        "route_id": f"R{index}",
        "source_policy": candidate.policy,
        "distance_m": round(float(metrics["agent_distance_m"]), 1)
        if isinstance(metrics.get("agent_distance_m"), int | float) and math.isfinite(metrics["agent_distance_m"])
        else None,
        "node_count": len(candidate.route_nodes),
        "turn_count": len(candidate.turns),
        "turns": candidate.turns,
        "route_nodes": candidate.route_nodes,
    }


def hidden_candidate_metrics(candidate: GraphRouteResult) -> dict[str, Any]:
    metrics = candidate.metrics
    return {
        "score": metrics["score"],
        "valid_route": metrics["valid_route"],
        "length_ratio": metrics["length_ratio"],
        "mean_route_distance_m": metrics["mean_route_distance_m"],
        "checkpoint_coverage": metrics["checkpoint_coverage"],
        "checkpoint_precision": metrics["checkpoint_precision"],
        "checkpoint_order": metrics["checkpoint_order"],
    }
