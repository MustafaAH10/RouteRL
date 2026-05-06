#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing huggingface_hub. On a GPU instance, run: bash scripts/setup_gpu_instance.sh"
        ) from exc

    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        resume_download=True,
    )
    print(f"downloaded {args.model} to {path}")


if __name__ == "__main__":
    main()
