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
from route_env.verify import task_graph


def bearing_deg(a: dict[str, float], b: dict[str, float]) -> float:
    lat1 = math.radians(a["lat"])
    lat2 = math.radians(b["lat"])
    dlon = math.radians(b["lon"] - a["lon"])
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def node_point(graph: Any, node_id: str) -> dict[str, float]:
    node = graph.nodes[str(node_id)]
    return {"lat": float(node["lat"]), "lon": float(node["lon"])}


def edge_highway(graph: Any, u: str, v: str) -> str:
    highway = graph.edges[u, v].get("highway", "")
    if isinstance(highway, list):
        highway = highway[0] if highway else ""
    return str(highway)


def edge_features(graph: Any, current: str, successor: str, goal_point: dict[str, float]) -> dict[str, Any]:
    current_point = node_point(graph, current)
    successor_point = node_point(graph, successor)
    current_to_goal = haversine_m(current_point, goal_point)
    successor_to_goal = haversine_m(successor_point, goal_point)
    return {
        "to_node": successor,
        "edge_length_m": round(float(graph.edges[current, successor].get("length", 0.0)), 1),
        "bearing_deg": round(bearing_deg(current_point, successor_point), 1),
        "straight_to_goal_m": round(successor_to_goal, 1),
        "progress_m": round(current_to_goal - successor_to_goal, 1),
        "highway": edge_highway(graph, current, successor),
    }


def compact_prompt(record: dict[str, Any]) -> str:
    lines = [
        "You are a graph-routing policy.",
        "Choose the next directed road edge from CURRENT toward GOAL.",
        "Use only one candidate_id. Avoid loops and prefer efficient progress.",
        'Return JSON only: {"candidate_id":"C1"}',
        "",
        f"task_id: {record['task_id']}",
        f"current_node: {record['current_node']}",
        f"goal_node: {record['goal_node']}",
        f"route_so_far_m: {record['route_so_far_m']}",
        f"remaining_teacher_steps: {record['remaining_teacher_steps']}",
        "",
        "Candidates:",
    ]
    for candidate in record["candidates"]:
        lines.append(
            f'{candidate["candidate_id"]}: to={candidate["to_node"]} '
            f'len={candidate["edge_length_m"]}m progress={candidate["progress_m"]}m '
            f'goal={candidate["straight_to_goal_m"]}m bearing={candidate["bearing_deg"]} '
            f'highway={candidate["highway"]}'
        )
    return "\n".join(lines)


def route_distance_prefix(graph: Any, route: list[str], end_index: int) -> float:
    return sum(float(graph.edges[u, v].get("length", 0.0)) for u, v in zip(route[:end_index], route[1 : end_index + 1]))


def records_for_task(
    task: dict[str, Any],
    *,
    include_prompt: bool,
    max_states_per_task: int | None,
) -> list[dict[str, Any]]:
    graph = task_graph(task)
    teacher_route = [str(int(node)) for node in task["oracle"]["gold_osm_route"]]
    goal = str(int(task["destination"]["osm_id"]))
    goal_point = node_point(graph, goal)
    records = []
    if len(teacher_route) < 2:
        return records

    for index, current in enumerate(teacher_route[:-1]):
        teacher_next = teacher_route[index + 1]
        successors = [str(node) for node in graph.successors(current)]
        if teacher_next not in successors:
            continue
        candidates = []
        target_candidate_id = None
        for candidate_index, successor in enumerate(sorted(successors), start=1):
            candidate = edge_features(graph, current, successor, goal_point)
            candidate["candidate_id"] = f"C{candidate_index}"
            candidate["visited"] = successor in teacher_route[: index + 1]
            candidates.append(candidate)
            if successor == teacher_next:
                target_candidate_id = candidate["candidate_id"]
        record = {
            "task_id": task["task_id"],
            "state_id": f"{task['task_id']}::step_{index:04d}",
            "split": task.get("difficulty"),
            "current_node": current,
            "goal_node": goal,
            "route_so_far_m": round(route_distance_prefix(graph, teacher_route, index), 1),
            "remaining_teacher_steps": len(teacher_route) - index - 1,
            "candidates": candidates,
            "target": {
                "candidate_id": target_candidate_id,
                "next_node": teacher_next,
            },
            "teacher": {
                "route_nodes": teacher_route,
                "distance_m": round(float(task["oracle"]["distance_m"]), 1),
            },
        }
        if include_prompt:
            record["prompt"] = compact_prompt(record)
        records.append(record)
        if max_states_per_task is not None and len(records) >= max_states_per_task:
            break
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--max-states-per-task", type=int)
    parser.add_argument("--include-prompt", action="store_true")
    args = parser.parse_args()

    records = []
    for task_index, task in enumerate(tqdm(iter_jsonl(args.tasks), desc="graph-policy")):
        if args.limit_tasks is not None and task_index >= args.limit_tasks:
            break
        records.extend(
            records_for_task(
                task,
                include_prompt=args.include_prompt,
                max_states_per_task=args.max_states_per_task,
            )
        )

    write_jsonl(Path(args.out), records)
    print(f"wrote {len(records)} graph policy state(s) to {args.out}")


if __name__ == "__main__":
    main()
