#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_gpu_instance.sh
#
# Optional overrides:
#   PYTHON_BIN=python3.11 bash scripts/setup_gpu_instance.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_gpu_instance.sh
#   HF_MODEL=Qwen/Qwen3-VL-4B-Instruct bash scripts/setup_gpu_instance.sh
#   HF_MODEL_REVISION=<commit-or-tag> bash scripts/setup_gpu_instance.sh
#   EXPECTED_GPU_COUNT=8 bash scripts/setup_gpu_instance.sh
#   INSTALL_FLASH_ATTN=1 bash scripts/setup_gpu_instance.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-routerl}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
HF_MODEL_REVISION="${HF_MODEL_REVISION:-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-}}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-0}"

echo "Creating venv: ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo "Installing RouteRL base package"
python -m pip install -e .

echo "Installing PyTorch from ${TORCH_INDEX_URL}"
python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"

echo "Installing Hugging Face VLM dependencies"
python -m pip install \
  "accelerate" \
  "huggingface_hub" \
  "qwen-vl-utils" \
  "safetensors" \
  "transformers"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  echo "Installing flash-attn. This can fail if CUDA/toolkit headers are not available."
  python -m pip install flash-attn --no-build-isolation
fi

echo "Verifying GPU visibility"
python - <<'PY'
import os
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_0:", torch.cuda.get_device_name(0))
expected = int(os.environ.get("EXPECTED_GPU_COUNT", "0"))
if expected and torch.cuda.device_count() < expected:
    raise SystemExit(f"expected at least {expected} CUDA devices, found {torch.cuda.device_count()}")
PY

echo "Downloading Hugging Face model: ${HF_MODEL}"
DOWNLOAD_ARGS=(--model "${HF_MODEL}")
if [[ -n "${HF_MODEL_REVISION}" ]]; then
  DOWNLOAD_ARGS+=(--revision "${HF_MODEL_REVISION}")
fi
if [[ -n "${HF_CACHE_DIR}" ]]; then
  DOWNLOAD_ARGS+=(--cache-dir "${HF_CACHE_DIR}")
fi
python scripts/download_hf_model.py "${DOWNLOAD_ARGS[@]}"

python -m unittest discover -s tests
python -m pip check

cat <<EOF

Setup complete.

Run a one-task prediction:

  source ${VENV_DIR}/bin/activate
  EXP=data/experiments/long_8_25km_route_strip_probe
  python scripts/run_hf_agent.py \\
    --tasks "\${EXP}/tasks.jsonl" \\
    --model ${HF_MODEL} \\
    --out "\${EXP}/predictions/hf_predictions.jsonl" \\
    --limit 1 \\
    --device auto \\
    --dtype bfloat16 \\
    --max-new-tokens 1536 \\
    --sanitize-labels

Evaluate:

  python scripts/evaluate_predictions.py \\
    --tasks "\${EXP}/tasks.jsonl" \\
    --predictions "\${EXP}/predictions/hf_predictions.jsonl" \\
    --out "\${EXP}/results/hf_predictions.jsonl"
EOF
