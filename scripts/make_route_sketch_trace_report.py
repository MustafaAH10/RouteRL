#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from route_env.io import iter_jsonl


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def load_font(name: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(name, size)
    except Exception:
        return ImageFont.load_default()


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width_chars: int,
    line_height: int,
) -> None:
    x, y = xy
    for line in textwrap.wrap(text, width=width_chars):
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height


def make_composite(
    image_paths: list[Path],
    captions: list[str],
    out_path: Path,
    *,
    max_panel_px: int = 920,
) -> None:
    title_font = load_font("DejaVuSans-Bold.ttf", 30)
    body_font = load_font("DejaVuSans.ttf", 22)
    panels = []
    for index, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((max_panel_px, max_panel_px), Image.Resampling.LANCZOS)
        header_h = 112
        panel = Image.new("RGB", (image.width, image.height + header_h), "white")
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, image.width, header_h), fill=(245, 247, 250))
        draw.text((16, 12), f"Image {index}: {path.name}", fill=(20, 27, 38), font=title_font)
        caption = captions[index] if index < len(captions) else ""
        caption = caption.replace(f"Image {index}: ", "")
        draw_wrapped(
            draw,
            caption,
            (16, 54),
            font=body_font,
            fill=(65, 75, 88),
            width_chars=max(24, image.width // 13),
            line_height=26,
        )
        panel.paste(image, (0, header_h))
        panels.append(panel)

    gap = 24
    width = sum(panel.width for panel in panels) + gap * max(0, len(panels) - 1)
    height = max(panel.height for panel in panels)
    composite = Image.new("RGB", (width, height), (232, 235, 239))
    x = 0
    for panel in panels:
        composite.paste(panel, (x, 0))
        x += panel.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path, quality=92)


def safe_metric(value: Any, digits: int = 4) -> str:
    if isinstance(value, int | float) and math.isfinite(value):
        return f"{float(value):.{digits}f}"
    return "n/a"


def load_tasks(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {record["task_id"]: record for record in iter_jsonl(path)}


def report_for_record(
    record: dict[str, Any],
    *,
    task: dict[str, Any] | None,
    overwrite: bool,
    max_panel_px: int,
) -> Path:
    render_dir = Path(record["render_dir"])
    report_path = render_dir / "full_agent_trace_report.md"
    if report_path.exists() and not overwrite:
        return report_path

    steps = record.get("steps", [])
    for step in steps:
        if step.get("sent_as_composite"):
            source_paths = [Path(path) for path in step.get("source_image_paths", [])]
            source_captions = list(step.get("source_image_captions", []))
            if source_paths:
                out_name = f"trace_step_{int(step['step']):02d}_source_bundle.png"
                make_composite(source_paths, source_captions, render_dir / out_name, max_panel_px=max_panel_px)
        else:
            image_paths = [Path(path) for path in step.get("image_paths", [])]
            captions = list(step.get("image_captions", []))
            if image_paths:
                out_name = f"trace_step_{int(step['step']):02d}_vlm_seen.png"
                make_composite(image_paths, captions, render_dir / out_name, max_panel_px=max_panel_px)

    result = record.get("result", {})
    prediction = record.get("prediction", {})
    gold_turns = task.get("oracle", {}).get("gold_turn_route", []) if task else []
    input_mode = "exact single composite image" if any(step.get("sent_as_composite") for step in steps) else "separate images with captions"
    report_lines: list[str] = [
        "# Full Handheld VLM Agent Trace",
        "",
        f"Task: `{record['task_id']}`",
        "",
        f"Input mode: `{input_mode}`",
        "",
        "Final prediction:",
        "",
        "```json",
        compact_json({"turns": prediction.get("turns", [])}),
        "```",
        "",
    ]
    if gold_turns:
        report_lines.extend(["Gold checkpoint route:", "", "```json", compact_json(gold_turns), "```", ""])
    report_lines.extend(
        [
            "Final score:",
            "",
            "```text",
            f"score:                 {safe_metric(result.get('score'))}",
            f"valid_route:           {bool(result.get('valid_route'))}",
            f"length_ratio:          {safe_metric(result.get('length_ratio'))}",
            f"mean_route_distance_m: {safe_metric(result.get('mean_route_distance_m'), 1)}",
            f"checkpoint_coverage:   {safe_metric(result.get('checkpoint_coverage'))}",
            f"checkpoint_precision:  {safe_metric(result.get('checkpoint_precision'))}",
            f"checkpoint_order:      {safe_metric(result.get('checkpoint_order'))}",
            "```",
            "",
            "## What The Agent Actually Received",
            "",
        ]
    )
    if input_mode == "exact single composite image":
        report_lines.extend(
            [
                "At each turn, the model received one stitched composite PNG as its single image input.",
                "The source panels are preserved in each step record for auditing.",
                "",
            ]
        )
    else:
        report_lines.extend(
            [
                "At each turn, the model received the images as separate image inputs with captions.",
                "The large step image below is a human-readable composite of that same image bundle.",
                "It is faithful to the image order and captions, but it was not the literal single PNG sent to the model.",
                "",
            ]
        )

    report_lines.extend(
        [
            "Available tools:",
            "",
            "```text",
            "inspect(target)",
            "edit_route(turns)",
            "preview_route(turns)",
            "finish(turns)",
            "```",
            "",
        ]
    )

    for step in steps:
        step_num = int(step["step"])
        report_lines.extend([f"## Step {step_num}", ""])
        if step.get("sent_as_composite"):
            image_name = Path(step.get("image_paths", [""])[0]).name
            source_name = f"trace_step_{step_num:02d}_source_bundle.png"
            report_lines.extend(
                [
                    "Exact single image sent to the VLM:",
                    "",
                    f'<img src="{image_name}" width="1400">',
                    "",
                    "Source panel composite for audit:",
                    "",
                    f'<img src="{source_name}" width="1400">',
                    "",
                ]
            )
        else:
            bundle_name = f"trace_step_{step_num:02d}_vlm_seen.png"
            report_lines.extend(
                [
                    "Human composite of the separate images sent to the VLM:",
                    "",
                    f'<img src="{bundle_name}" width="1400">',
                    "",
                ]
            )
        report_lines.extend(
            [
                "Tool call:",
                "",
                "```json",
                compact_json(step.get("action", {})),
                "```",
                "",
                "Environment result:",
                "",
                "```json",
                compact_json(step.get("observation", {})),
                "```",
                "",
            ]
        )

    final_agent = render_dir / "step_08_final_agent_preview.png"
    if not final_agent.exists():
        final_agent = render_dir / "forced_final_agent_preview.png"
    final_oracle = render_dir / "final_with_oracle_overlay.png"
    if not final_oracle.exists():
        final_oracle = render_dir / "forced_final_with_oracle_overlay.png"
    if final_agent.exists():
        report_lines.extend(
            [
                "## Final Agent-Only Preview",
                "",
                "This is the final route preview. Orange is only the agent route.",
                "",
                f'<img src="{final_agent.name}" width="1200">',
                "",
            ]
        )
    if final_oracle.exists():
        report_lines.extend(
            [
                "## Final Oracle Comparison",
                "",
                "This image is for after-the-fact evaluation only. It was not shown to the model during interaction.",
                "",
                f'<img src="{final_oracle.name}" width="1200">',
                "",
            ]
        )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--tasks")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-panel-px", type=int, default=920)
    args = parser.parse_args()

    tasks = load_tasks(Path(args.tasks) if args.tasks else None)
    reports = []
    for index, record in enumerate(iter_jsonl(args.rollout)):
        if args.limit is not None and index >= args.limit:
            break
        reports.append(
            report_for_record(
                record,
                task=tasks.get(record["task_id"]),
                overwrite=args.overwrite,
                max_panel_px=args.max_panel_px,
            )
        )
    for path in reports:
        print(path)
    print(f"wrote {len(reports)} trace report(s)")


if __name__ == "__main__":
    main()
