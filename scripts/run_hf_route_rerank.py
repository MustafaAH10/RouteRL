#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from route_env.hf_client import generate_route_prediction, load_vision_model
from route_env.io import iter_jsonl, write_jsonl


SYSTEM_PROMPT = """You are a visual route-selection model.

You are shown one map image with start A, destination B, and several colored
candidate driving routes labeled R1, R2, R3, etc. Every candidate is a valid
directed route from A to B. Choose the best route: shortest plausible driving
route with minimal detour, no unnecessary loops, and good agreement with the
visible road geometry.

Return exactly one JSON object:
{"route_id":"R1"}

Never output prose.
"""


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def build_prompt(task: dict[str, Any]) -> str:
    candidates = [
        {
            "route_id": candidate["route_id"],
            "color": candidate["color"],
        }
        for candidate in task["candidates"]
    ]
    return f"""Candidate routes:
{compact_json(candidates)}

The labels and colors identify complete valid routes from A to B. The route
lengths, scores, and oracle route are hidden. Choose the best route by looking
at the map. JSON only.
"""


def parse_route_id(parsed: Any, task: dict[str, Any]) -> str | None:
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if isinstance(parsed, dict):
        value = parsed.get("route_id", parsed.get("choice", parsed.get("selected_route")))
        if value is not None:
            route_id = str(value).upper()
            if route_id in set(task["candidate_ids"]):
                return route_id
    if isinstance(parsed, str):
        route_id = parsed.upper()
        if route_id in set(task["candidate_ids"]):
            return route_id
    return None


def run_task(
    task: dict[str, Any],
    *,
    model: Any,
    processor: Any,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = build_prompt(task)
    try:
        parsed, raw = generate_route_prediction(
            image_path=task["image"],
            prompt=prompt,
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
            system_prompt=SYSTEM_PROMPT,
        )
        model_error = None
    except Exception as exc:
        parsed = {}
        raw = ""
        model_error = str(exc)
    route_id = parse_route_id(parsed, task)
    prediction = {"route_id": route_id} if route_id else {}
    return {
        "task_id": task["task_id"],
        "mode": "hf-route-rerank",
        "prediction": prediction,
        "raw_prediction": parsed,
        "raw_response": raw,
        "model_error": model_error,
        "rollout": {
            "image": task["image"],
            "candidate_ids": task["candidate_ids"],
            "selected_route_id": route_id,
            "prompt": prompt,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a HF VLM on one-shot route-reranking tasks.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    model, processor = load_vision_model(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )

    written = 0

    def records() -> Any:
        nonlocal written
        for index, task in enumerate(iter_jsonl(args.tasks), start=1):
            if args.limit and index > args.limit:
                break
            record = run_task(task, model=model, processor=processor, max_new_tokens=args.max_new_tokens)
            written += 1
            yield record
            if written % 25 == 0:
                print(f"wrote {written} route-rerank records", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, records())
    print(f"wrote {written} HF route-rerank records to {args.out}")


if __name__ == "__main__":
    main()
