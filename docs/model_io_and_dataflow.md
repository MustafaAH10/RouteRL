# RouteRL Model Input, Output, OSM Data, And Hugging Face Flow

This note explains the current Hugging Face model path.

## GPU Command

Do this on a GPU instance:

```bash
bash scripts/setup_gpu_instance.sh
```

Run one prediction:

```bash
routerl/bin/python scripts/run_hf_agent.py \
  --tasks data/tasks/demo.jsonl \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --out data/predictions/qwen3_vl_4b_hf.jsonl \
  --limit 1 \
  --device auto \
  --dtype bfloat16 \
  --max-new-tokens 512
```

Evaluate:

```bash
routerl/bin/python scripts/evaluate_predictions.py \
  --tasks data/tasks/demo.jsonl \
  --predictions data/predictions/qwen3_vl_4b_hf.jsonl \
  --out data/results/qwen3_vl_4b_hf.jsonl
```

## What Data Is This?

The intended next demo data should be generated from OpenStreetMap driving
network data in Singapore.

The first demo task is:

```text
task_id: singapore_000001
city: Singapore
network_type: drive
bbox: [103.845, 1.285, 103.855, 1.295]
image: data/rendered/singapore_000001/map.png
```

The bbox is:

```text
west longitude:  103.845
south latitude:    1.285
east longitude:  103.855
north latitude:    1.295
```

## What The Model Receives

The Hugging Face runner gives the model:

1. One rendered image: `data/rendered/singapore_000001/map.png`
2. A prompt describing the task.
3. A list of allowed visible sparse turn checkpoint labels, such as `T1`, `T2`,
   `T3`.

The model does **not** receive:

```text
graph adjacency
edge lengths
hidden shortest path
OSM metadata
Google Maps directions
OSRM output
```

The current driving-checkpoint prompt says:

```text
You are given a real driving map image.
Roads are styled by hierarchy and one-way arrows show directed streets.
Black T-labels mark sparse turn checkpoints.

Infer a plausible short driving route from A to B using only visible
turn checkpoints.
Return JSON only.
```

Current model output:

```json
{
  "turns": ["T6", "T11", "T4"],
  "confidence": 0.7,
  "reason": "brief"
}
```

## How OSM Becomes An Image

OpenStreetMap is structured geographic data, not just raster map tiles. OSMnx
downloads a driving network and turns it into a directed graph:

```text
nodes = real OSM road points
edges = directed drivable road segments between points
```

The renderer draws those graph geometries:

```text
OSM data
  -> OSMnx mode-specific graph
  -> rendered top-down map image
```

In the current image:

```text
gray lines = real OSM roads
black dots/labels = sparse turn checkpoints and decision points
blue A = start
red B = destination
```

The hidden graph is used for generation and evaluation, but not as model input.

Evaluation expands the checkpoint prediction through the hidden graph:

```text
A -> T6 -> T11 -> T4 -> B
```

becomes the shortest directed OSM path through those waypoints. If any segment
does not exist because of one-way constraints or graph disconnection, the route
is invalid.

## How Hugging Face Inference Works

`scripts/run_hf_agent.py` loads:

```text
Qwen/Qwen3-VL-4B-Instruct
```

using `transformers` and `AutoProcessor`. It formats the image and prompt as a
multimodal chat message, runs `model.generate`, extracts JSON from the model
text, and writes prediction JSONL.

The default Hugging Face model id comes from the official Qwen3-VL model family.
Hugging Face documents Qwen3-VL usage with
`Qwen3VLForConditionalGeneration` and `AutoProcessor`.

## Real-World Target Flow

For a query like:

```text
How do I get from City Hall MRT to Orchard Road?
```

the intended system is:

```text
natural language query
  -> deterministic geocoder resolves A/B coordinates
  -> snap A/B to Singapore OSM driving graph
  -> render driving map observation
  -> VLM predicts sparse turns or route waypoints
  -> verifier checks prediction on hidden directed OSM graph
  -> render RouteRL route in Streets-GL
  -> optionally compare against Google/OSRM route for evaluation
```

The model is not being asked to know Singapore from memory. It is being asked to
look at a rendered real map observation and infer a route visually.

## Current Caveat

The current local task is short-range and single-panel. It is aligned with the
driving-first direction, but it is not yet enough for cross-island routing.
The next interface should add:

- normalized screen waypoints,
- route-strip segment maps for long trips.

See `docs/task_design.md`.
