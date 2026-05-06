#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from route_env.hf_client import generate_route_prediction, load_vision_model
from route_env.io import read_jsonl, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]

PROMPT = """You are a routing model.

You are given a real driving map image. Roads are drawn with hierarchy styling, one-way arrows show directed streets where known, the blue marker A is the start, and the red marker B is the destination.

Black T-labels mark sparse turn checkpoints and decision points. The T numbers are arbitrary labels, not route order. Infer a plausible short driving route from A to B by choosing only the checkpoints that the driver should physically pass through, in driving order. Do not invent labels.

Most visible labels are distractors. Do not list every label unless the route really passes through every one.

Return JSON only with exactly this shape:
{"turns":["T1","T2"]}
"""


STRIP_PROMPT = (REPO_ROOT / "prompts/drive_route_strip_prompt.txt").read_text(encoding="utf-8")


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


def sanitize_strip_prediction(prediction: dict, allowed_by_segment: dict[str, set[str]]) -> tuple[dict, dict]:
    raw_segments = prediction.get("segments", []) if isinstance(prediction, dict) else []
    rejected: dict[str, list] = {}
    sanitized_segments = []
    if isinstance(raw_segments, list):
        for segment in raw_segments:
            if not isinstance(segment, dict) or not isinstance(segment.get("segment_id"), str):
                rejected.setdefault("_schema", []).append(segment)
                continue
            segment_id = segment["segment_id"]
            allowed = allowed_by_segment.get(segment_id, set())
            turns = segment.get("turns", [])
            kept = []
            if isinstance(turns, list):
                for turn in turns:
                    if isinstance(turn, str) and turn in allowed:
                        kept.append(turn)
                    else:
                        rejected.setdefault(segment_id, []).append(turn)
            else:
                rejected.setdefault(segment_id, []).append(turns)
            sanitized_segments.append({"segment_id": segment_id, "turns": kept})
    else:
        rejected["_schema"] = [raw_segments]
    sanitized = dict(prediction) if isinstance(prediction, dict) else {}
    sanitized["segments"] = sanitized_segments
    return sanitized, {
        "rejected_turns": rejected,
        "num_rejected_turns": sum(len(values) for values in rejected.values()),
        "num_sanitized_segments": len(sanitized_segments),
    }


def build_flat_prompt(task: dict) -> tuple[str, Path, list[str] | None, set[str] | None, dict[str, set[str]] | None]:
    prompt = PROMPT + "\nAllowed turn checkpoint labels: " + ", ".join(task["turn_checkpoints"].keys())
    return prompt, Path(task["images"]["map"]), None, set(task["turn_checkpoints"]), None


def build_strip_prompt(task: dict) -> tuple[str, list[Path], list[str], set[str] | None, dict[str, set[str]]]:
    captions = ["Image 0: overview corridor map."]
    image_paths = [Path(task["images"]["overview"])]
    allowed_lines = []
    for index, segment in enumerate(task["segments"], start=1):
        image_paths.append(Path(segment["images"]["map"]))
        captions.append(f"Image {index}: segment {segment['segment_id']} local driving map.")
        allowed_lines.append(f"{segment['segment_id']}: " + ", ".join(segment["turn_checkpoints"].keys()))
    prompt = STRIP_PROMPT + "\n\nImage order:\n" + "\n".join(captions)
    prompt += "\n\nAllowed checkpoints by segment:\n" + "\n".join(allowed_lines)
    allowed_by_segment = {segment["segment_id"]: set(segment["turn_checkpoints"]) for segment in task["segments"]}
    return prompt, image_paths, captions, None, allowed_by_segment


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
        if task.get("task_type") == "route_strip":
            prompt, image_path, image_captions, allowed_labels, allowed_by_segment = build_strip_prompt(task)
        else:
            prompt, image_path, image_captions, allowed_labels, allowed_by_segment = build_flat_prompt(task)
        try:
            parsed, raw = generate_route_prediction(
                image_path=image_path,
                prompt=prompt,
                model=model,
                processor=processor,
                max_new_tokens=args.max_new_tokens,
                image_captions=image_captions,
            )
            record = {"task_id": task["task_id"], "prediction": parsed, "raw_response": raw}
            if args.sanitize_labels:
                if allowed_by_segment is not None:
                    sanitized, diagnostics = sanitize_strip_prediction(parsed, allowed_by_segment)
                else:
                    sanitized, diagnostics = sanitize_prediction(parsed, allowed_labels or set())
                record["raw_prediction"] = parsed
                record["prediction"] = sanitized
                record["sanitization"] = diagnostics
            records.append(record)
        except Exception as exc:
            records.append(
                {
                    "task_id": task["task_id"],
                    "prediction": {"turns": None},
                    "error": str(exc),
                }
            )
    write_jsonl(args.out, records)
    print(f"wrote {len(records)} Hugging Face predictions to {args.out}")


if __name__ == "__main__":
    main()
