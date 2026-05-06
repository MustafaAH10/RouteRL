# RouteRL Progress And Goals

## Current Goal

Build a driving-first VLM/RL routing benchmark where the model infers a route
from real map observations without calling a routing API at inference time.

The target real-world flow is:

```text
natural language query
  -> deterministic geocoder resolves A/B coordinates
  -> snap A/B to the Singapore OSM driving graph
  -> render model-facing driving map observations
  -> VLM predicts sparse turn checkpoints or segment waypoints
  -> hidden directed OSM verifier expands and scores the route
  -> Streets-GL renders the predicted route for visualization
```

## What Is Built

- OSMnx task generation uses `network_type=drive`.
- Tasks contain a hidden directed OSM graph crop with node coordinates, directed
  edges, edge lengths, road hierarchy metadata, and one-way metadata.
- The model-facing renderer draws road hierarchy, one-way arrows, blue A, red B,
  and sparse `T` turn checkpoints instead of dense node labels.
- The expected prediction schema is now:

```json
{"turns":["T3","T8","T12"],"confidence":0.7,"reason":"brief"}
```

- The verifier maps predicted `T` checkpoints back to hidden OSM node IDs,
  expands `A -> turns -> B` with directed shortest paths, and rewards validity,
  length ratio, route geometry similarity, loop avoidance, and schema quality.
- Oracle, random, greedy, SFT export, Hugging Face inference, rendering, debug
  overlays, and evaluation scripts are aligned to the `turns` schema.
- The local model-server path has been removed from the project direction. GPU
  inference is via Hugging Face/Transformers.
- Streets-GL remains a visualization layer, not the verifier or source of truth.

## What Is Not Built Yet

- Route-strip/segment-map generation for long or cross-island trips.
- Multi-image VLM input for overview + local panels.
- Segment stitching rewards.
- Turn restriction handling beyond directed one-way graph constraints.
- Geocoding from natural-language locations such as `City Hall MRT` and
  `Orchard Road`.
- Snapping arbitrary A/B coordinates to the driving graph.
- Streets-GL route overlay for predicted versus external comparison routes.
- RL training loop. The rewardable verifier exists, but policy optimization is
  still a GPU-side next phase.

## GPU Next Steps

On the GPU instance:

```bash
bash scripts/setup_gpu_instance.sh
```

Generate a small Singapore driving set:

```bash
routerl/bin/python scripts/generate_tasks.py \
  --bbox 103.845,1.285,103.855,1.295 \
  --city Singapore \
  --network-type drive \
  --n 20 \
  --min-distance-m 500 \
  --max-distance-m 2000 \
  --max-checkpoints 24 \
  --out data/tasks/singapore_drive_smoke.jsonl
```

Render images:

```bash
routerl/bin/python scripts/render_tasks.py \
  --tasks data/tasks/singapore_drive_smoke.jsonl \
  --out-dir data/rendered
```

Run oracle and one VLM pass:

```bash
routerl/bin/python scripts/make_oracle_predictions.py \
  --tasks data/tasks/singapore_drive_smoke.jsonl \
  --out data/predictions/oracle.jsonl

routerl/bin/python scripts/run_hf_agent.py \
  --tasks data/tasks/singapore_drive_smoke.jsonl \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --out data/predictions/qwen3_vl_4b_hf.jsonl \
  --limit 5 \
  --device auto \
  --dtype auto
```

Evaluate:

```bash
routerl/bin/python scripts/evaluate_predictions.py \
  --tasks data/tasks/singapore_drive_smoke.jsonl \
  --predictions data/predictions/qwen3_vl_4b_hf.jsonl \
  --out data/results/qwen3_vl_4b_hf.jsonl
```

## Design Notes

Single huge maps will fail for real cross-island driving because minor roads,
ramps, and turn choices need local detail. The better setup is hierarchical:

```text
overview corridor map
  + segment map 1
  + segment map 2
  + segment map 3
  -> per-segment turns
  -> stitched route verification
```

That gives the VLM enough visual context to make local choices while preserving
a rewardable global route objective.
