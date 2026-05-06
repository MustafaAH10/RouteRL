#!/usr/bin/env bash
set -euo pipefail

MODEL="${HF_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
REVISION="${HF_MODEL_REVISION:-main}"
CACHE_DIR="${HF_HOME:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --revision)
      REVISION="$2"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

ARGS=(--model "${MODEL}" --revision "${REVISION}")
if [[ -n "${CACHE_DIR}" ]]; then
  ARGS+=(--cache-dir "${CACHE_DIR}")
fi

python scripts/download_hf_model.py "${ARGS[@]}"
echo "prefetched ${MODEL}@${REVISION}"
