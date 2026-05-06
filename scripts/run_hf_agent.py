#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from route_env.hf_client import generate_route_prediction, load_vision_model
from route_env.io import read_jsonl, write_jsonl


PROMPT = """You are a routing model.

You are given a real driving map image. Roads are drawn with hierarchy styling, one-way arrows show directed streets where known, the blue marker A is the start, and the red marker B is the destination.

Black T-labels mark sparse turn checkpoints and decision points. The T numbers are arbitrary labels, not route order. Infer a plausible short driving route from A to B by choosing only the checkpoints that the driver should physically pass through, in driving order. Do not invent labels.

Return JSON only with exactly this shape:
{"turns":["T1","T2"],"confidence":0.0,"reason":"brief"}
"""


def sanitize_prediction(prediction: dict, allowed_labels: set[str]) -> tuple[dict, dict]:
    turns = prediction.get("turns", []) if isinstance(prediction, dict) else []
    sanitized_turns = []
    rejected_turns = []
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, str) and turn in allowed_labels:
                sanitized_turns.append(turn)
            else:
                rejected_turns.append(turn)
    else:
        rejected_turns.append(turns)
    sanitized = dict(prediction) if isinstance(prediction, dict) else {}
    sanitized["turns"] = sanitized_turns
    return sanitized, {
        "rejected_turns": rejected_turns,
        "num_rejected_turns": len(rejected_turns),
        "num_sanitized_turns": len(sanitized_turns),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--dtype", default="auto", help="auto, float16, bfloat16, or float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sanitize-labels", action="store_true")
    args = parser.parse_args()

    model, processor = load_vision_model(
        model_id=args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )

    records = []
    tasks = read_jsonl(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    for task in tqdm(tasks, desc=f"hf:{args.model}"):
        image_path = Path(task["images"]["map"])
        allowed_labels = set(task["turn_checkpoints"])
        prompt = PROMPT + "\nAllowed turn checkpoint labels: " + ", ".join(task["turn_checkpoints"].keys())
        try:
            parsed, raw = generate_route_prediction(
                image_path=image_path,
                prompt=prompt,
                model=model,
                processor=processor,
                max_new_tokens=args.max_new_tokens,
            )
            record = {"task_id": task["task_id"], "prediction": parsed, "raw_response": raw}
            if args.sanitize_labels:
                sanitized, diagnostics = sanitize_prediction(parsed, allowed_labels)
                record["raw_prediction"] = parsed
                record["prediction"] = sanitized
                record["sanitization"] = diagnostics
            records.append(record)
        except Exception as exc:
            records.append(
                {
                    "task_id": task["task_id"],
                    "prediction": {"turns": None, "confidence": 0.0, "reason": "inference_error"},
                    "error": str(exc),
                }
            )
    write_jsonl(args.out, records)
    print(f"wrote {len(records)} Hugging Face predictions to {args.out}")


if __name__ == "__main__":
    main()
