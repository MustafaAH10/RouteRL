#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_gpu_instance.sh
#
# Optional overrides:
#   PYTHON_BIN=python3.11 bash scripts/setup_gpu_instance.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_gpu_instance.sh
#   HF_MODEL=Qwen/Qwen3-VL-4B-Instruct bash scripts/setup_gpu_instance.sh
#   INSTALL_FLASH_ATTN=1 bash scripts/setup_gpu_instance.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-routerl}"
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

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
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_0:", torch.cuda.get_device_name(0))
PY

echo "Downloading Hugging Face model: ${HF_MODEL}"
python scripts/download_hf_model.py --model "${HF_MODEL}"

cat <<EOF

Setup complete.

Run a one-task prediction:

  source ${VENV_DIR}/bin/activate
  python scripts/run_hf_agent.py \\
    --tasks data/tasks/demo.jsonl \\
    --model ${HF_MODEL} \\
    --out data/predictions/qwen3_vl_4b_hf.jsonl \\
    --limit 1 \\
    --device auto \\
    --dtype bfloat16 \\
    --max-new-tokens 512

Evaluate:

  python scripts/evaluate_predictions.py \\
    --tasks data/tasks/demo.jsonl \\
    --predictions data/predictions/qwen3_vl_4b_hf.jsonl \\
    --out data/results/qwen3_vl_4b_hf.jsonl
EOF

