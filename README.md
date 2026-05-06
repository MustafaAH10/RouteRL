# RouteRL

RouteRL is a small infrastructure stack for testing whether a VLM can learn
driving-route behavior from real map observations without calling a router at
inference time.

The first milestone is intentionally narrow:

- generate real OSM driving graph crops;
- render a model-facing map image with A/B markers, road hierarchy, one-way
  arrows, and sparse turn checkpoints;
- ask a Hugging Face VLM to output turn checkpoints;
- verify the route against the hidden directed OSM driving graph shortest path;
- keep Streets-GL as an optional visualization backend, not the core verifier.

## Setup

```bash
python3 -m venv routerl
routerl/bin/python -m pip install --upgrade pip setuptools wheel
routerl/bin/python -m pip install -e .
```

The current CPU machine is not the intended place for VLM inference. Use this
local setup only for code review, task generation, rendering, and lightweight
script checks.

Optional Streets-GL screenshot support:

```bash
npm install
npx playwright install chromium
```

Streets-GL is a WebGL2 browser app. Its upstream README says Chrome/WebGL2 is
required and a discrete GPU is recommended, so expect headless CPU screenshots
to be less reliable than the 2D OSM renderer.

## Generate Real OSM Driving Tasks

Use a small bbox first. This example is around central Singapore and uses the
OSM driving network.

```bash
routerl/bin/python scripts/generate_tasks.py \
  --bbox 103.845,1.285,103.855,1.295 \
  --city Singapore \
  --network-type drive \
  --n 5 \
  --min-distance-m 500 \
  --max-distance-m 2000 \
  --max-checkpoints 24 \
  --out data/tasks/demo.jsonl
```

Render the model-facing images:

```bash
routerl/bin/python scripts/render_tasks.py \
  --tasks data/tasks/demo.jsonl \
  --out-dir data/rendered
```

## Baselines And Evaluation

```bash
routerl/bin/python scripts/make_oracle_predictions.py \
  --tasks data/tasks/demo.jsonl \
  --out data/predictions/oracle.jsonl

routerl/bin/python scripts/make_random_predictions.py \
  --tasks data/tasks/demo.jsonl \
  --out data/predictions/random.jsonl

routerl/bin/python scripts/make_greedy_predictions.py \
  --tasks data/tasks/demo.jsonl \
  --out data/predictions/greedy.jsonl

routerl/bin/python scripts/evaluate_predictions.py \
  --tasks data/tasks/demo.jsonl \
  --predictions data/predictions/oracle.jsonl \
  --out data/results/oracle.jsonl
```

Create visual debug overlays:

```bash
routerl/bin/python scripts/render_debug_overlays.py \
  --tasks data/tasks/demo.jsonl \
  --predictions data/predictions/oracle.jsonl \
  --results data/results/oracle.jsonl \
  --out-dir data/debug
```

## Hugging Face VLM Trial

Do this on a GPU instance, not on the current CPU-only machine:

```bash
bash scripts/setup_gpu_instance.sh
```

The script creates the `routerl` venv, installs PyTorch/Transformers/Hugging
Face dependencies, checks CUDA visibility, and downloads the default model:

```text
Qwen/Qwen3-VL-4B-Instruct
```

If your GPU image needs a different PyTorch CUDA wheel, override the index URL:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  bash scripts/setup_gpu_instance.sh
```

Then run predictions:

```bash
routerl/bin/python scripts/run_hf_agent.py \
  --tasks data/tasks/demo.jsonl \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --out data/predictions/qwen3_vl_4b_hf.jsonl \
  --limit 1 \
  --device auto \
  --dtype auto
```

Evaluate:

```bash
routerl/bin/python scripts/evaluate_predictions.py \
  --tasks data/tasks/demo.jsonl \
  --predictions data/predictions/qwen3_vl_4b_hf.jsonl \
  --out data/results/qwen3_vl_4b_hf.jsonl
```

CPU inference for VLMs can be very slow. For practical experiments, run this on
a GPU machine and use `--device auto` or `--device cuda`.

## Current Task Interface

The model sees only the rendered map image and a short list of allowed visible
turn labels. It does not receive adjacency, edge lengths, or the hidden shortest
path.

Expected prediction:

```json
{"turns":["T3","T8","T12"],"confidence":0.7,"reason":"brief"}
```

The verifier expands:

```text
A -> T3 -> T8 -> T12 -> B
```

through the hidden directed OSM driving graph, then scores validity, length
ratio, route geometry similarity, loops, and schema quality.

For long trips, the intended next step is a hierarchical route-strip setup:
overview corridor map, segment maps, local route predictions, then stitched
graph verification.

See `routesight_rl_mvp_spec.md` and `docs/task_design.md` for the proposed next
interface. See `docs/progress_and_goals.md` for the current implementation
status and GPU continuation checklist.

## Optional Streets-GL Capture

```bash
npm run capture:streets-gl -- --lat=1.2966 --lon=103.7764 --out=data/rendered/streets_gl.png
```

This is intentionally separate from the core route task images. The route
learning benchmark should not depend on browser/GPU availability.
