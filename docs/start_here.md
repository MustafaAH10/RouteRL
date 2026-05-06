# Start Here

RouteRL tests whether a vision-language model can infer driving routes from map
images without calling a router at inference time.

## Mental Model

There are three separate worlds:

```text
1. Hidden OSM graph
   Used by scripts to generate tasks and score answers. The model never sees it.

2. Rendered map images
   What the model sees: roads, one-way arrows, A/B markers, and sparse T labels.

3. Model JSON
   What the model returns: checkpoint labels in driving order.
```

For a flat task the model returns:

```json
{"turns":["T03","T11","T14"]}
```

For a route-strip task the model returns:

```json
{"segments":[{"segment_id":"S01","turns":["T001","T011"]}]}
```

Route-strip checkpoint labels are unique across the whole strip, but the answer
is still grouped by segment.

No extra score or explanation is required.

## Baselines

`oracle` means the hidden teacher answer. It uses the OSM shortest path generated
by OSMnx and should score `1.000`. If oracle fails, the task/verifier is broken.
Oracle is not a model and is never shown to the model.

`greedy` is a dumb baseline. It repeatedly chooses a checkpoint that is closer
to B by straight-line distance. It does not understand one-way streets or map
semantics. It is useful as a cheap sanity check: if a VLM cannot beat greedy,
the prompt/image format probably needs work.

`random` samples visible checkpoint labels randomly. It should usually be bad.

`Qwen...` files are actual VLM predictions.

## Current Recommended Flow

Start with the route-strip probe:

```bash
source routerl/bin/activate

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

python scripts/make_oracle_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --out "$EXP/predictions/oracle.jsonl"

python scripts/evaluate_predictions.py \
  --tasks "$EXP/tasks.jsonl" \
  --predictions "$EXP/predictions/oracle.jsonl" \
  --out "$EXP/results/oracle.jsonl"
```

Then run one VLM sample:

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

## Experiment Names

The names are descriptive, not magic:

```text
short_500m_2km
  Flat single-image routes, 500m to 2km.

medium_2_6km
  Flat single-image routes, 2km to 6km.

long_8_25km_80cp_probe
  Long flat routes, capped at 80 visible checkpoints.
  This exists to show why flat long maps are bad.

long_8_25km_200cp_probe
  Same idea with 200 checkpoints.
  This confirms more labels make OCR clutter worse.

long_8_25km_route_strip_probe
  Recommended current long-route format.
  One overview image plus several local segment images.
```

Generated files inside each experiment:

```text
tasks.jsonl       task definitions and hidden verifier data
maps/             rendered map images
predictions/      model or baseline JSON outputs
results/          verifier scores for predictions
overlays/         optional rendered debug overlays
```

The `_smoke/` experiment folder contains tiny generator sanity fixtures. Use the
named short, medium, long, and route-strip folders for real analysis.

## View Results

Start the local overlay server from the repo root:

```bash
bash scripts/serve_overlay.sh --port 8000 --bind 0.0.0.0
```

Open, with port forwarding if the GPU box is remote:

```text
http://localhost:8000/renderer/route_overlay.html?tasks=/data/experiments/long_8_25km_route_strip_probe/tasks.jsonl&predictions=/data/experiments/long_8_25km_route_strip_probe/predictions/qwen3_vl_8b_strip_limit1.jsonl&results=/data/experiments/long_8_25km_route_strip_probe/results/qwen3_vl_8b_strip_limit1.jsonl
```

If `http://<server-ip>:8000` does not load, the cloud firewall probably blocks
port `8000`. Use SSH/IDE port forwarding instead.

Example SSH forwarding from your laptop:

```bash
ssh -L 8000:127.0.0.1:8000 root@<server-ip>
```

Then open the `localhost:8000` URL on your laptop. `localhost` means whichever
machine your browser is running on, not automatically the GPU box.

## How To Read Scores

Useful fields in a result JSONL:

```text
score: final reward, 0 to 1
valid_schema: model returned parseable JSON with the expected fields
valid_route: hidden graph could route through the predicted checkpoints
length_ratio: predicted route length / oracle route length
mean_route_distance_m: average geometry distance from oracle route
checkpoint_reward: overlap and order agreement with hidden oracle checkpoints
unknown_turn_count: labels invented by the model
```

Common failure: the model outputs every visible `T` label. That can be
schema-valid but route-bad because it creates loops and a very long path.
