# Fresh GPU Instance Setup

Use this only on a new GPU machine.

## 1. Machine Check

```bash
nvidia-smi
python3 --version
git --version
```

Prefer Python 3.10 or 3.11.

```bash
apt-get update
apt-get install -y git python3.10-venv python3.10-dev build-essential
```

## 2. Environment

For this 96 GB RTX PRO 6000 Blackwell box, use CUDA 13 PyTorch wheels. CUDA 12.4 wheels can report `cuda_available=True` but still fail to support `sm_120` compute.

```bash
python3.10 -m venv routerl
source routerl/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install accelerate huggingface_hub qwen-vl-utils safetensors transformers
```

## 3. Verify

```bash
bash scripts/smoke_h100_instance.sh --expected-gpus 1
```

Expected: CUDA visible, tests pass, oracle `mean_score=1.000`, and `RouteRL H100 smoke check passed`.

## 4. Model Download

```bash
source routerl/bin/activate
python scripts/download_hf_model.py --model Qwen/Qwen3-VL-8B-Instruct
```

If auth is needed:

```bash
hf auth login
```

## 5. What This Sets Up

This sets up generation, rendering, verification, and Hugging Face VLM inference. It does not fully install or run VeRL training. See `docs/verl_integration.md` for the reward adapter and remaining VeRL wiring.
