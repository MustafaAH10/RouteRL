# RouteRL GPU Setup, Model I/O, And Experiment Flow

This is the canonical runbook for taking a fresh GPU instance from zero to
generated RouteRL tasks, rendered maps, baseline evaluations, Hugging Face VLM
predictions, and debug overlays.

If this repo feels disorienting, read `docs/start_here.md` first. It explains
the model input/output, `oracle`, `greedy`, result metrics, and the current
recommended route-strip workflow.

The current generated-data layout is:

```text
data/experiments/<experiment_name>/
  tasks.jsonl
  maps/
  predictions/
  results/
  overlays/
```

Use named experiment folders. Do not use flat folders like `data/rendered`,
`data/debug`, `data/predictions`, or `data/tasks`; task IDs repeat across
experiments.

Always set `EXP` in the same shell where you run the commands. If `EXP` is
empty, `"$EXP/tasks.jsonl"` becomes `/tasks.jsonl`, which writes outside the
repo when running as root. `scripts/generate_tasks.py` now fails fast for that
specific path, but setting `EXP` explicitly is still the habit to keep.

Generation and rendering are separate:

```text
scripts/generate_tasks.py -> writes tasks.jsonl only
scripts/render_tasks.py   -> writes map PNGs and updates image paths
```

## 1. Rent And Enter A GPU Instance

Recommended starting point:

```text
GPU: NVIDIA 4090/A5000/A6000/A100/H100 class
VRAM: 24GB minimum for 4B; 32GB+ preferred for 8B
Disk: 80GB+ free
OS: Linux with recent NVIDIA driver
Python: 3.10 or 3.11
```

After SSHing into the instance:

```bash
pwd
nvidia-smi
python3 --version
git --version
```

## 2. Get The Repo

Clone the repo, or pull it if the instance already has a copy:

```bash
git clone <YOUR_REPO_URL> RouteRL
cd RouteRL
```

If the repo is already present:

```bash
cd RouteRL
git status --short
git pull --ff-only
```

If you are copying this workspace manually rather than cloning, just make sure
you are at the repo root:

```bash
pwd
ls
```

You should see files such as `pyproject.toml`, `scripts/`, `route_env/`, and
`docs/`.

## 3. Create The Python Environment

The GPU setup script creates `routerl`, installs the package, installs PyTorch
and Hugging Face dependencies, verifies CUDA, and downloads a model.

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
INSTALL_FLASH_ATTN=1 bash scripts/setup_gpu_instance.sh
```

Verify the environment:

```bash
source routerl/bin/activate
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## 4. Optional Browser Setup For Streets-GL

The core benchmark does not require Streets-GL. It uses the 2D OSM renderer and
hidden graph verifier.

For optional Streets-GL screenshots:

```bash
npm install
npx playwright install chromium
```

Smoke test:

```bash
npm run capture:streets-gl -- \
  --lat=1.2966 \
  --lon=103.7764 \
  --pitch=65 \
  --yaw=0 \
  --distance=700 \
  --out=data/experiments/manual/streets_gl.png
```

## 5. Generate A Short Driving Experiment

Create the experiment folder. Run these two lines in the same shell as the
commands below:

```bash
EXP=data/experiments/short_500m_2km
mkdir -p "$EXP"/{maps,predictions,results,overlays}
```

Generate 20 Singapore driving tasks around central Singapore. This writes only
`tasks.jsonl`; it does not create map images yet.

```bash
python scripts/generate_tasks.py \
  --bbox 103.845,1.285,103.855,1.295 \
  --city Singapore \
  --network-type drive \
  --n 20 \
  --min-distance-m 500 \
  --max-distance-m 2000 \
  --max-checkpoints 24 \
  --out "$EXP/tasks.jsonl"
```

Render model-facing maps and update `tasks.jsonl` so every task points at this
experiment's own map path:

```bash
python scripts/render_tasks.py \
  --tasks "$EXP/tasks.jsonl" \
  --out-dir "$EXP/maps" \
  --write-updated-tasks "$EXP/tasks.jsonl"
```

Inspect counts:

```bash
python - <<'PY'
import json, statistics
tasks = [json.loads(line) for line in open("data/experiments/short_500m_2km/tasks.jsonl") if line.strip()]
print("tasks:", len(tasks))
print("distance min/mean/max:",
      round(min(t["oracle"]["distance_m"] for t in tasks), 1),
      round(statistics.mean(t["oracle"]["distance_m"] for t in tasks), 1),
      round(max(t["oracle"]["distance_m"] for t in tasks), 1))
print("turn count min/mean/max:",
      min(t["oracle"]["turn_count"] for t in tasks),
      round(statistics.mean(t["oracle"]["turn_count"] for t in tasks), 1),
      max(t["oracle"]["turn_count"] for t in tasks))
print("first image:", tasks[0]["images"]["map"])
PY
```

## 6. Run Baselines

Oracle should score perfectly. If it does not, the verifier/task format is
wrong.

```bash
python scripts/make_oracle_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --out "$EXP/predictions/oracle.jsonl"

python scripts/make_random_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --out "$EXP/predictions/random.jsonl"

python scripts/make_greedy_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --out "$EXP/predictions/greedy.jsonl"
```

Evaluate:

```bash
python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/oracle.jsonl" \
  --out "$EXP/results/oracle.jsonl"

python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/random.jsonl" \
  --out "$EXP/results/random.jsonl"

python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/greedy.jsonl" \
  --out "$EXP/results/greedy.jsonl"
```

Create oracle overlays:

```bash
python scripts/render_debug_overlays.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/oracle.jsonl" \
  --results "$EXP/results/oracle.jsonl" \
  --out-dir "$EXP/overlays/oracle"
```

## 7. Run Qwen3-VL On The Short Experiment

Recommended current comparison:

```text
Qwen/Qwen3-VL-4B-Instruct
Qwen/Qwen3-VL-8B-Instruct
```

Run 8B on the first 5 short tasks:

```bash
python scripts/run_hf_agent.py \
  --tasks "$EXP/tasks.jsonl" \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --out "$EXP/predictions/qwen3_vl_8b_clean_sanitized_limit5.jsonl" \
  --limit 5 \
  --device auto \
  --dtype bfloat16 \
  --max-new-tokens 512 \
  --sanitize-labels
```

`--sanitize-labels` keeps the raw model output in `raw_prediction`, writes
rejected labels under `sanitization`, and evaluates only labels visible in the
current task.

Evaluate:

```bash
python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/qwen3_vl_8b_clean_sanitized_limit5.jsonl" \
  --out "$EXP/results/qwen3_vl_8b_clean_sanitized_limit5.jsonl"
```

Render overlays:

```bash
python scripts/render_debug_overlays.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/qwen3_vl_8b_clean_sanitized_limit5.jsonl" \
  --results "$EXP/results/qwen3_vl_8b_clean_sanitized_limit5.jsonl" \
  --out-dir "$EXP/overlays/qwen3_vl_8b_clean_sanitized_limit5"
```

Run the same experiment with 4B:

```bash
python scripts/run_hf_agent.py \
  --tasks "$EXP/tasks.jsonl" \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --out "$EXP/predictions/qwen3_vl_4b_clean_sanitized_limit5.jsonl" \
  --limit 5 \
  --device auto \
  --dtype bfloat16 \
  --max-new-tokens 512 \
  --sanitize-labels

python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/qwen3_vl_4b_clean_sanitized_limit5.jsonl" \
  --out "$EXP/results/qwen3_vl_4b_clean_sanitized_limit5.jsonl"
```

## 8. Generate A Medium 2-6km Experiment

Use a larger bbox and more checkpoints. As above, set `EXP`, generate
`tasks.jsonl`, then render maps into that same experiment folder:

```bash
EXP=data/experiments/medium_2_6km
mkdir -p "$EXP"/{maps,predictions,results,overlays}

python scripts/generate_tasks.py \
  --bbox 103.75,1.25,103.90,1.36 \
  --city Singapore \
  --network-type drive \
  --n 5 \
  --min-distance-m 2000 \
  --max-distance-m 6000 \
  --max-checkpoints 40 \
  --route-margin-m 300 \
  --out "$EXP/tasks.jsonl"

python scripts/render_tasks.py \
  --tasks "$EXP/tasks.jsonl" \
  --out-dir "$EXP/maps" \
  --write-updated-tasks "$EXP/tasks.jsonl"
```

Run oracle and empty-baseline sanity checks:

```bash
python scripts/make_oracle_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --out "$EXP/predictions/oracle.jsonl"

python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/oracle.jsonl" \
  --out "$EXP/results/oracle.jsonl"

python - <<'PY'
from route_env.io import read_jsonl, write_jsonl
from route_env.verify import verify_prediction
tasks = read_jsonl("data/experiments/medium_2_6km/tasks.jsonl")
preds = [
    {"task_id": t["task_id"], "prediction": {"turns": []}}
    for t in tasks
]
results = [verify_prediction(t, p) for t, p in zip(tasks, preds)]
write_jsonl("data/experiments/medium_2_6km/predictions/empty.jsonl", preds)
write_jsonl("data/experiments/medium_2_6km/results/empty.jsonl", results)
PY
```

Run a tiny 8B stress check:

```bash
python scripts/run_hf_agent.py \
  --tasks "$EXP/tasks.jsonl" \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --out "$EXP/predictions/qwen3_vl_8b_clean_sanitized_limit2.jsonl" \
  --limit 2 \
  --device auto \
  --dtype bfloat16 \
  --max-new-tokens 768 \
  --sanitize-labels

python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/qwen3_vl_8b_clean_sanitized_limit2.jsonl" \
  --out "$EXP/results/qwen3_vl_8b_clean_sanitized_limit2.jsonl"

python scripts/render_debug_overlays.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/qwen3_vl_8b_clean_sanitized_limit2.jsonl" \
  --results "$EXP/results/qwen3_vl_8b_clean_sanitized_limit2.jsonl" \
  --out-dir "$EXP/overlays/qwen3_vl_8b_clean_sanitized_limit2"
```

## 9. Long-Distance Probe Commands

These probes are useful for confirming that single-panel long routes are a bad
interface. They are not the recommended final long-route training format.

80-checkpoint 8-25km probe:

```bash
EXP=data/experiments/long_8_25km_80cp_probe
mkdir -p "$EXP"/{maps,predictions,results,overlays}

python scripts/generate_tasks.py \
  --bbox 103.60,1.20,104.05,1.48 \
  --city Singapore \
  --network-type drive \
  --n 10 \
  --min-distance-m 8000 \
  --max-distance-m 25000 \
  --max-checkpoints 80 \
  --route-margin-m 800 \
  --seed 23 \
  --out "$EXP/tasks.jsonl"

python scripts/make_oracle_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --out "$EXP/predictions/oracle.jsonl"

python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/oracle.jsonl" \
  --out "$EXP/results/oracle.jsonl"
```

200-checkpoint render probe:

```bash
EXP=data/experiments/long_8_25km_200cp_probe
mkdir -p "$EXP"/{maps,predictions,results,overlays}

python scripts/generate_tasks.py \
  --bbox 103.60,1.20,104.05,1.48 \
  --city Singapore \
  --network-type drive \
  --n 2 \
  --min-distance-m 8000 \
  --max-distance-m 25000 \
  --max-checkpoints 200 \
  --route-margin-m 800 \
  --seed 29 \
  --out "$EXP/tasks.jsonl"

python scripts/render_tasks.py \
  --tasks "$EXP/tasks.jsonl" \
  --out-dir "$EXP/maps" \
  --write-updated-tasks "$EXP/tasks.jsonl"
```

The expected finding is label clutter: hundreds of visible `T` labels are not a
good VLM input. Use route strips for long trips.

## 10. Build A Long Route-Strip Probe

Route strips convert long flat tasks into one overview image plus several local
segment images. Each segment has its own local `T01...` labels.

```bash
EXP=data/experiments/long_8_25km_route_strip_probe
mkdir -p "$EXP"/{maps,predictions,results,overlays}

python scripts/make_route_strip_tasks.py \
  --tasks data/experiments/long_8_25km_80cp_probe/tasks.jsonl \
  --out "$EXP/tasks.jsonl" \
  --target-segment-distance-m 2500 \
  --max-segment-checkpoints 32 \
  --segment-margin-m 260 \
  --limit 3

python scripts/render_route_strip_tasks.py \
  --tasks "$EXP/tasks.jsonl" \
  --out-dir "$EXP/maps" \
  --write-updated-tasks "$EXP/tasks.jsonl"

python scripts/make_oracle_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --out "$EXP/predictions/oracle.jsonl"

python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/oracle.jsonl" \
  --out "$EXP/results/oracle.jsonl"
```

Run the VLM with a larger token budget because route-strip JSON contains one
object per segment:

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
```

## 11. Browser Route Overlay

Serve the repo and open the zero-bundler overlay viewer:

```bash
bash scripts/serve_overlay.sh --port 8000 --bind 0.0.0.0
```

Then open:

```text
http://localhost:8000/renderer/route_overlay.html?tasks=/data/experiments/long_8_25km_route_strip_probe/tasks.jsonl&predictions=/data/experiments/long_8_25km_route_strip_probe/predictions/oracle.jsonl&results=/data/experiments/long_8_25km_route_strip_probe/results/oracle.jsonl
```

This viewer draws RouteRL graph edges, oracle paths, agent paths, checkpoints,
scores, and route-strip segment boxes directly from JSONL files. It does not
need Node. Streets-GL remains optional 3D context; its source can be cloned under
`external/streets-gl`, but the RouteRL verifier-aligned overlay is the browser
viewer above.

If the GPU box is remote and the public IP refuses port `8000`, use SSH or IDE
port forwarding. `localhost` in your browser means your laptop unless the
browser itself is running on the GPU box.

## 12. Reproducible GPU And VeRL Prep

On frequently replaced 8xH100 boxes, prefer the reproducible flow in:

```text
docs/reproducible_gpu_infra.md
docs/verl_integration.md
```

Prepare route-strip trainer records:

```bash
python scripts/prepare_verl_dataset.py \
  --tasks data/experiments/long_8_25km_route_strip_probe/tasks.jsonl \
  --out data/experiments/long_8_25km_route_strip_probe/verl_train.jsonl \
  --split train
```

Smoke-check a GPU node:

```bash
EXPECTED_GPU_COUNT=8 bash scripts/smoke_h100_instance.sh --expected-gpus 8
```

## 13. What The Model Receives

The Hugging Face runner gives the model:

For flat tasks:

1. One rendered image from `task["images"]["map"]`.
2. The driving prompt.
3. A list of allowed visible sparse turn checkpoint labels.

For route-strip tasks:

1. One overview image from `task["images"]["overview"]`.
2. One local segment image per segment from `task["images"]["segments"]`.
3. The route-strip prompt.
4. A per-segment list of allowed checkpoint labels.

The model does **not** receive:

```text
graph adjacency
edge lengths
hidden shortest path
OSM metadata
Google Maps directions
OSRM output
```

The prompt says:

```text
You are a routing model.

You are given a real driving map image. Roads are drawn with hierarchy styling, one-way arrows show directed streets where known, the blue marker A is the start, and the red marker B is the destination.

Black T-labels mark sparse turn checkpoints and decision points. The T numbers are arbitrary labels, not route order. Infer a plausible short driving route from A to B by choosing only the checkpoints that the driver should physically pass through, in driving order. Do not invent labels.

Return JSON only with exactly this shape:
{"turns":["T1","T2"]}
```

Expected model output:

```json
{
  "turns": ["T6", "T11", "T4"]
}
```

## 14. How OSM Becomes An Image

OpenStreetMap is structured geographic data, not just raster map tiles. OSMnx
downloads a driving network and turns it into a directed graph:

```text
nodes = real OSM road points
edges = directed drivable road segments between points
```

The task generator stores:

```text
origin/destination OSM IDs
turn checkpoint OSM IDs
hidden graph nodes and directed edges
oracle shortest path geometry
oracle gold_turn_route
```

The renderer draws:

```text
gray/dark roads by hierarchy
one-way arrows
black checkpoint dots with displaced labels and leader lines
blue A
red B
```

The hidden graph is used for generation and evaluation, but not as model input.

## 15. How Verification Works

Evaluation expands the checkpoint prediction through the hidden graph:

```text
A -> T6 -> T11 -> T4 -> B
```

becomes the shortest directed OSM path through those waypoints. If any segment
does not exist because of one-way constraints or graph disconnection, the route
is invalid.

Current verifier rewards include:

```text
valid JSON/schema
known checkpoint labels
directed route exists through predicted checkpoints
checkpoint overlap/order against oracle gold_turn_route
route length ratio
geometry similarity
loop penalty
turn-count penalty relative to oracle complexity
max segment gap compared with oracle checkpoint spacing
```

Empty `turns` is deliberately not a valid route proposal when a task has gold
checkpoints.

## 16. Existing Results To Compare Against

Current kept artifacts:

```text
data/experiments/short_500m_2km/
data/experiments/medium_2_6km/
data/experiments/long_8_25km_80cp_probe/
data/experiments/long_8_25km_200cp_probe/
```

The latest report is:

```text
docs/model_test_report.md
```

Known short-run reference:

```text
Qwen3-VL-4B clean sanitized, limit 5:
  mean_score=0.235
  success=0/5
  valid_route=1/5

Qwen3-VL-8B clean sanitized, limit 5:
  mean_score=0.402
  success=1/5
  valid_route=2/5
```

Known medium-run reference:

```text
Qwen3-VL-8B clean sanitized, limit 2:
  mean_score=0.620
  success=0/2
  valid_route=2/2
```

## 17. Troubleshooting

If CUDA is not visible:

```bash
nvidia-smi
source routerl/bin/activate
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

If the wrong PyTorch wheel was installed, rebuild the environment with the
correct `TORCH_INDEX_URL`.

If Hugging Face downloads are slow or rate-limited:

```bash
huggingface-cli login
```

If overlays render more tasks than expected, make sure you are using the updated
`scripts/render_debug_overlays.py`; it now renders only tasks with predictions
or results unless `--all-tasks` is passed.

If model output contains impossible labels, use `--sanitize-labels` and inspect
the `sanitization.rejected_turns` field in the prediction JSONL.

## 18. Real-World Target Flow

For a query like:

```text
How do I get from City Hall MRT to Orchard Road?
```

the intended system is:

```text
natural language query
  -> deterministic geocoder resolves A/B coordinates
  -> snap A/B to Singapore OSM driving graph
  -> render driving map observation(s)
  -> VLM predicts sparse turns or route waypoints
  -> verifier checks prediction on hidden directed OSM graph
  -> render RouteRL route in Streets-GL
  -> optionally compare against Google/OSRM route for evaluation
```

The model is not being asked to know Singapore from memory. It is being asked to
look at rendered real map observations and infer route proposals visually.

## 19. Current Caveat

Single-panel short routes are useful for debugging, but single-panel long routes
become OCR/clutter tasks. The next interface should add:

- overview corridor images;
- local segment maps;
- multi-image VLM input;
- segment stitching rewards;
- later, normalized screen waypoints instead of text labels.

See `docs/task_design.md` and `routesight_rl_mvp_spec.md`.
