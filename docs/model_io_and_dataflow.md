# GPU Setup, Model I/O, And Test Flow

This is the practical GPU runbook. It keeps only the setup and testing path:
clone the repo, create the Python environment, verify the checked-in data,
run a small VLM probe, and view the result.

Read `docs/start_here.md` first if the task format is unfamiliar.

Use the repo directly on the GPU instance.

## What The Repo Contains

Current generated artifacts are committed under named experiment folders:

```text
data/experiments/<experiment_name>/
  tasks.jsonl
  maps/
  predictions/
  results/
  overlays/
```

Important folders:

```text
data/experiments/short_500m_2km/
data/experiments/medium_2_6km/
data/experiments/long_8_25km_80cp_probe/
data/experiments/long_8_25km_200cp_probe/
data/experiments/long_8_25km_route_strip_probe/
```

Use named experiment folders. Do not use flat generated folders like
`data/rendered`, `data/debug`, `data/predictions`, or `data/tasks`; task IDs
repeat across experiments.

## 1. Fresh GPU Setup

Start on a Linux GPU box with a working NVIDIA driver:

```bash
nvidia-smi
python3 --version
git --version
```

Clone and enter the repo:

```bash
git clone <YOUR_REPO_URL> RouteRL
cd RouteRL
git checkout <PINNED_COMMIT>
```

For a frequently replaced instance, use persistent Hugging Face cache storage if
available:

```bash
export HF_HOME=/mnt/hf
mkdir -p "$HF_HOME"
```

Create the environment:

```bash
bash scripts/setup_gpu_instance.sh
source routerl/bin/activate
```

Useful setup overrides:

```bash
HF_MODEL=Qwen/Qwen3-VL-8B-Instruct bash scripts/setup_gpu_instance.sh
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_gpu_instance.sh
PYTHON_BIN=python3.11 bash scripts/setup_gpu_instance.sh
EXPECTED_GPU_COUNT=8 bash scripts/setup_gpu_instance.sh
```

What `scripts/setup_gpu_instance.sh` does:

```text
creates routerl/
installs RouteRL editable package
installs CUDA PyTorch
installs Hugging Face VLM dependencies
checks CUDA visibility
downloads the selected HF model
runs unit tests and pip check
```

## 2. Verify The Instance

```bash
source routerl/bin/activate
python -m py_compile route_env/*.py scripts/*.py
python -m unittest discover -s tests
python -m pip check
```

Quick GPU and verifier smoke check:

```bash
bash scripts/smoke_h100_instance.sh --expected-gpus 1
```

Use `--expected-gpus 8` on an 8xH100 box.

## 3. Model I/O

The model never sees the hidden graph. It sees rendered image files and a list
of allowed checkpoint labels.

Flat task input:

```text
image: task["images"]["map"]
prompt: prompts/drive_turn_prompt.txt
allowed labels: task["turn_checkpoints"].keys()
```

Flat task output:

```json
{"turns":["T03","T11","T14"]}
```

Route-strip task input:

```text
image 1: task["images"]["overview"]
image 2..N: task["images"]["segments"]
prompt: prompts/drive_route_strip_prompt.txt
allowed labels: per segment
```

Route-strip task output:

```json
{"segments":[{"segment_id":"S01","turns":["T001","T011"]}]}
```

Route-strip labels are unique across the whole strip, while the JSON is still
grouped by segment. The model should return route labels only. It should not
invent labels.

## 4. Core Test: Route-Strip Probe

Use the checked-in long route-strip probe:

```bash
EXP=data/experiments/long_8_25km_route_strip_probe
```

Re-evaluate oracle. This should score perfectly:

```bash
python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/oracle.jsonl" \
  --out "$EXP/results/oracle.jsonl"
```

Run the cheap non-model baseline:

```bash
python scripts/make_greedy_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --out "$EXP/predictions/greedy.jsonl"

python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/greedy.jsonl" \
  --out "$EXP/results/greedy.jsonl"
```

Run one Qwen3-VL-8B sample:

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

## 5. View The Result

Serve the repo:

```bash
bash scripts/serve_overlay.sh --port 8000 --bind 0.0.0.0
```

Open with SSH or IDE port forwarding:

```text
http://localhost:8000/renderer/route_overlay.html?tasks=/data/experiments/long_8_25km_route_strip_probe/tasks.jsonl&predictions=/data/experiments/long_8_25km_route_strip_probe/predictions/qwen3_vl_8b_strip_limit1.jsonl&results=/data/experiments/long_8_25km_route_strip_probe/results/qwen3_vl_8b_strip_limit1.jsonl
```

If `http://<server-ip>:8000` refuses to connect, the server can still be fine;
the cloud firewall is probably blocking the port. `localhost` means the machine
where your browser is running.

## 6. Regenerate Data Only When Needed

Task generation and rendering are separate:

```text
scripts/generate_tasks.py             -> flat tasks.jsonl only
scripts/render_tasks.py               -> flat map PNGs and image paths
scripts/make_route_strip_tasks.py     -> long flat tasks to route-strip tasks
scripts/render_route_strip_tasks.py   -> route-strip overview/panel PNGs
```

If you need a tiny fresh flat experiment:

```bash
EXP=data/experiments/short_500m_2km
mkdir -p "$EXP"/{maps,predictions,results,overlays}

python scripts/generate_tasks.py \
  --bbox 103.845,1.285,103.855,1.295 \
  --city Singapore \
  --network-type drive \
  --n 20 \
  --min-distance-m 500 \
  --max-distance-m 2000 \
  --max-checkpoints 24 \
  --out "$EXP/tasks.jsonl"

python scripts/render_tasks.py \
  --tasks "$EXP/tasks.jsonl" \
  --out-dir "$EXP/maps" \
  --write-updated-tasks "$EXP/tasks.jsonl"
```

If you need to rebuild the route-strip probe from the checked-in long flat
tasks:

```bash
EXP=data/experiments/long_8_25km_route_strip_probe
mkdir -p "$EXP"/{maps,predictions,results,overlays}

python scripts/make_route_strip_tasks.py \
  --tasks data/experiments/long_8_25km_80cp_probe/tasks.jsonl \
  --out "$EXP/tasks.jsonl" \
  --target-segment-distance-m 2500 \
  --max-segment-checkpoints 32 \
  --segment-margin-m 260 \
  --limit 4

python scripts/render_route_strip_tasks.py \
  --tasks "$EXP/tasks.jsonl" \
  --out-dir "$EXP/maps" \
  --write-updated-tasks "$EXP/tasks.jsonl"
```

Always set `EXP` in the same shell as the commands. If `EXP` is empty,
`"$EXP/tasks.jsonl"` becomes `/tasks.jsonl`, which writes outside the repo when
running as root.

## 7. Script Responsibilities

Core scripts:

```text
scripts/setup_gpu_instance.sh       set up Python, CUDA PyTorch, HF deps, model
scripts/smoke_h100_instance.sh      check GPU visibility, tests, oracle scoring
scripts/run_hf_agent.py             run a Hugging Face VLM on task images
scripts/evaluate_predictions.py     score predictions against hidden OSM graph
scripts/serve_overlay.sh            serve the browser route overlay
```

Baseline scripts:

```text
scripts/make_oracle_predictions.py  hidden teacher answer, should score 1.000
scripts/make_greedy_predictions.py  cheap checkpoint-distance baseline
scripts/make_random_predictions.py  random checkpoint baseline
```

Rendering scripts:

```text
scripts/render_tasks.py             render flat task images
scripts/render_route_strip_tasks.py render overview and segment panel images
scripts/render_debug_overlays.py    render static PNG overlays
```

## 8. Troubleshooting

CUDA not visible:

```bash
nvidia-smi
source routerl/bin/activate
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

Wrong PyTorch wheel:

```bash
rm -rf routerl
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_gpu_instance.sh
```

Hugging Face download is slow or rate-limited:

```bash
huggingface-cli login
```

Frontend does not load on a remote box:

```bash
ssh -L 8000:127.0.0.1:8000 root@<server-ip>
```

Then open the `localhost:8000` overlay URL on your laptop.

## 9. Read Results

Important result fields:

```text
score: final reward, 0 to 1
valid_schema: parseable JSON with the expected fields
valid_route: hidden directed graph can route through predicted checkpoints
length_ratio: predicted route length / oracle route length
mean_route_distance_m: geometry distance from oracle route
checkpoint_reward: overlap/order agreement with hidden oracle checkpoints
unknown_turn_count: labels invented by the model
```

The latest narrative report is `docs/model_test_report.md`.
