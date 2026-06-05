#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from typing import Any

from route_env.io import iter_jsonl, write_jsonl


def select_route(task: dict[str, Any], policy: str, rng: random.Random) -> str:
    candidates = list(task["candidates"])
    if policy == "oracle":
        return str(task["best_route_id"])
    if policy == "random":
        return str(rng.choice(candidates)["route_id"])
    if policy == "first":
        return str(candidates[0]["route_id"])
    if policy == "shortest_hidden":
        return str(
            min(
                candidates,
                key=lambda candidate: (
                    float(candidate["hidden_metrics"]["distance_m"]),
                    str(candidate["route_id"]),
                ),
            )["route_id"]
        )
    raise ValueError(f"unknown policy: {policy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple route-reranking policies.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy", choices=["oracle", "random", "first", "shortest_hidden"], default="random")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    def records() -> Any:
        for index, task in enumerate(iter_jsonl(args.tasks), start=1):
            if args.limit and index > args.limit:
                break
            route_id = select_route(task, args.policy, rng)
            yield {
                "task_id": task["task_id"],
                "mode": f"route-rerank:{args.policy}",
                "prediction": {"route_id": route_id},
                "rollout": {
                    "candidate_ids": task["candidate_ids"],
                    "selected_route_id": route_id,
                    "policy": args.policy,
                },
            }

    write_jsonl(args.out, records())
    print(f"wrote route-rerank {args.policy} predictions to {args.out}")


if __name__ == "__main__":
    main()
