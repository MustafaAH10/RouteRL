#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from route_env.graph_route_candidates import candidate_record, generate_route_candidates, hidden_candidate_metrics
from route_env.io import iter_jsonl, write_jsonl


def build_prompt(record: dict[str, Any]) -> str:
    lines = [
        "You are a route-reranking model.",
        "",
        "Choose the best route candidate from START to GOAL.",
        "Every candidate is a valid directed route unless marked otherwise.",
        "Prefer the shortest efficient route, avoid detours and loops, and return JSON only.",
        "",
        f"START: {record['start_node']}",
        f"GOAL: {record['goal_node']}",
        "",
        "Candidates:",
    ]
    for candidate in record["candidates"]:
        turns = ",".join(candidate["turns"][:24])
        if len(candidate["turns"]) > 24:
            turns += ",..."
        lines.append(
            f'{candidate["route_id"]}: distance_m={candidate["distance_m"]} '
            f'nodes={candidate["node_count"]} turns=[{turns}]'
        )
    lines.extend(["", 'Return exactly: {"route_id":"R1"}'])
    return "\n".join(lines)


def best_route_id(hidden_metrics: dict[str, dict[str, Any]]) -> str | None:
    valid_items = [
        (route_id, metrics)
        for route_id, metrics in hidden_metrics.items()
        if metrics.get("valid_route")
    ]
    if not valid_items:
        return None
    return max(valid_items, key=lambda item: item[1].get("score", 0.0))[0]


def convert_task(
    task: dict[str, Any],
    *,
    num_candidates: int,
    beam_width: int,
    max_expansions: int,
    diversity_penalty_m: float,
    include_prompt: bool,
    shuffle_candidates: bool,
) -> dict[str, Any]:
    candidates = generate_route_candidates(
        task,
        num_candidates=num_candidates,
        beam_width=beam_width,
        max_expansions=max_expansions,
        diversity_penalty_m=diversity_penalty_m,
    )
    if shuffle_candidates:
        seed_bytes = hashlib.sha256(task["task_id"].encode("utf-8")).digest()[:8]
        rng = random.Random(int.from_bytes(seed_bytes, byteorder="big"))
        rng.shuffle(candidates)
    visible = [candidate_record(candidate, index) for index, candidate in enumerate(candidates, start=1)]
    hidden = {
        visible_candidate["route_id"]: hidden_candidate_metrics(candidate)
        for visible_candidate, candidate in zip(visible, candidates, strict=True)
    }
    record = {
        "task_id": task["task_id"],
        "city": task.get("city"),
        "difficulty": task.get("difficulty"),
        "start_node": str(int(task["origin"]["osm_id"])),
        "goal_node": str(int(task["destination"]["osm_id"])),
        "candidate_count": len(visible),
        "candidates": visible,
        "hidden_metrics": hidden,
        "target": {
            "route_id": best_route_id(hidden),
            "teacher_distance_m": round(float(task["oracle"]["distance_m"]), 1),
            "teacher_turns": list(task["oracle"]["gold_turn_route"]),
        },
        "notes": {
            "candidate_generation": "penalized heuristic beam search over directed OSM graph",
            "hidden": "hidden_metrics and target are for training/evaluation, not model prompt",
        },
    }
    if include_prompt:
        record["prompt"] = build_prompt(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--max-expansions", type=int, default=12_000)
    parser.add_argument("--diversity-penalty-m", type=float, default=900.0)
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--shuffle-candidates", action="store_true")
    args = parser.parse_args()

    records = []
    for task in tqdm(iter_jsonl(args.tasks), desc="graph-candidates"):
        records.append(
            convert_task(
                task,
                num_candidates=args.num_candidates,
                beam_width=args.beam_width,
                max_expansions=args.max_expansions,
                diversity_penalty_m=args.diversity_penalty_m,
                include_prompt=args.include_prompt,
                shuffle_candidates=args.shuffle_candidates,
            )
        )
        if args.limit is not None and len(records) >= args.limit:
            break
    write_jsonl(Path(args.out), records)
    print(f"wrote {len(records)} candidate route task(s) to {args.out}")


if __name__ == "__main__":
    main()
