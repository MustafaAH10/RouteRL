#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from statistics import mean

from route_env.io import read_jsonl, write_jsonl
from route_env.verify import verify_prediction


def finite_mean(values: list[float]) -> float:
    vals = [v for v in values if isinstance(v, int | float) and math.isfinite(v)]
    return mean(vals) if vals else math.inf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tasks = {task["task_id"]: task for task in read_jsonl(args.tasks)}
    predictions = read_jsonl(args.predictions)
    results = []
    for pred in predictions:
        task = tasks[pred["task_id"]]
        results.append(verify_prediction(task, pred))
    write_jsonl(args.out, results)
    scores = [r["score"] for r in results]
    print(f"wrote {len(results)} results to {args.out}")
    print(f"mean_score={mean(scores):.3f}")
    print(f"success@0.75={mean([s >= 0.75 for s in scores]):.3f}")
    print(f"valid_schema={mean([r['valid_schema'] for r in results]):.3f}")
    print(f"valid_route={mean([r['valid_route'] for r in results]):.3f}")
    print(f"mean_length_ratio={finite_mean([r['length_ratio'] for r in results]):.3f}")
    print(f"mean_route_distance_m={finite_mean([r['mean_route_distance_m'] for r in results]):.1f}")


if __name__ == "__main__":
    main()

