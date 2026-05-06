# Fresh GPU Instance Setup

Use this when you get a new GPU box and need to bring RouteRL up from scratch.
This page is only about setting up the repo for generation, rendering,
evaluation, and Hugging Face VLM testing.

## 1. Machine Requirements

Recommended minimums:

```text
GPU: 24GB VRAM for Qwen3-VL-4B; 32GB+ for Qwen3-VL-8B
Disk: 80GB+ free if using the checked-in experiment bundle and one model
OS: Linux with a working NVIDIA driver
Python: 3.10 or 3.11
```

Check the box:

```bash
nvidia-smi
python3 --version
git --version
```

Install basic system packages if the image is very bare:

```bash
apt-get update
apt-get install -y git python3 python3-venv python3-dev build-essential
```

## 2. Clone The Repo

```bash
git clone <YOUR_REPO_URL> RouteRL
cd RouteRL
git checkout <PINNED_COMMIT>
```

If the repo is already present:

```bash
cd RouteRL
git pull --ff-only
```

The checked-in experiment bundle lives under `data/experiments/`, so a normal
clone should already include the current route-strip probe, maps, predictions,
and results.

## 3. Set Cache Locations

Use persistent disk for Hugging Face model downloads when possible:

```bash
export HF_HOME=/mnt/hf
mkdir -p "$HF_HOME"
```

If you do not have mounted storage, skip this and Hugging Face will use the
default cache under your home directory.

## 4. Create The Python Environment

The setup script creates `routerl/`, installs RouteRL, installs CUDA PyTorch,
installs the Hugging Face VLM dependencies, verifies CUDA, downloads the model,
and runs the tests.

Default model:

```text
Qwen/Qwen3-VL-4B-Instruct
```

Run:

```bash
bash scripts/setup_gpu_instance.sh
source routerl/bin/activate
```

Useful overrides:

```bash
HF_MODEL=Qwen/Qwen3-VL-8B-Instruct bash scripts/setup_gpu_instance.sh
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_gpu_instance.sh
PYTHON_BIN=python3.11 bash scripts/setup_gpu_instance.sh
EXPECTED_GPU_COUNT=8 bash scripts/setup_gpu_instance.sh
```

Only use `INSTALL_FLASH_ATTN=1` if the machine has CUDA build headers and you
actually need it.

## 5. Smoke Check

```bash
source routerl/bin/activate
python -m py_compile route_env/*.py scripts/*.py
python -m unittest discover -s tests
python -m pip check
```

For a GPU visibility and oracle-verifier smoke check:

```bash
bash scripts/smoke_h100_instance.sh --expected-gpus 1
```

On an 8xH100 box:

```bash
bash scripts/smoke_h100_instance.sh --expected-gpus 8
```

## 6. First Functional Test

Use the checked-in route-strip probe:

```bash
EXP=data/experiments/long_8_25km_route_strip_probe
```

Re-evaluate the checked-in oracle baseline:

```bash
python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/oracle.jsonl" \
  --out "$EXP/results/oracle.jsonl"
```

Expected summary:

```text
mean_score=1.000
valid_schema=1.000
valid_route=1.000
```

Run one VLM route-strip sample:

```bash
python scripts/run_hf_agent.py \
  --tasks "$EXP/tasks.jsonl" \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --out "$EXP/predictions/qwen3_vl_8b_strip_limit1.jsonl" \
  --limit 1 \
  --device auto \
  --dtype bfloat16 \
  --max-new-tokens 1536 \
  --sanitize-labels

python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/qwen3_vl_8b_strip_limit1.jsonl" \
  --out "$EXP/results/qwen3_vl_8b_strip_limit1.jsonl"
```

## 7. View Results

```bash
bash scripts/serve_overlay.sh --port 8000 --bind 0.0.0.0
```

Open with SSH or IDE port forwarding:

```text
http://localhost:8000/renderer/route_overlay.html?tasks=/data/experiments/long_8_25km_route_strip_probe/tasks.jsonl&predictions=/data/experiments/long_8_25km_route_strip_probe/predictions/qwen3_vl_8b_strip_limit1.jsonl&results=/data/experiments/long_8_25km_route_strip_probe/results/qwen3_vl_8b_strip_limit1.jsonl
```

If the public IP refuses port `8000`, use port forwarding. `localhost` means the
machine where the browser is running.
