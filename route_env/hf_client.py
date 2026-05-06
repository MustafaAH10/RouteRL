from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _resolve_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def load_vision_model(
    model_id: str,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
) -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face inference dependencies are missing. Run: "
            "routerl/bin/python -m pip install -e ."
        ) from exc

    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelClass

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    resolved_dtype = _resolve_dtype(torch, dtype)
    if resolved_dtype != "auto":
        kwargs["torch_dtype"] = resolved_dtype
    else:
        kwargs["torch_dtype"] = "auto"

    if device == "auto":
        kwargs["device_map"] = "auto"

    model = ModelClass.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, local_files_only=local_files_only)
    if device != "auto":
        model = model.to(device)
    model.eval()
    return model, processor


def generate_route_prediction(
    image_path: str | Path,
    prompt: str,
    model: Any,
    processor: Any,
    max_new_tokens: int = 256,
) -> tuple[dict[str, Any], str]:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    target_device = getattr(model, "device", None)
    if target_device is not None:
        inputs = inputs.to(target_device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return extract_json(raw), raw

