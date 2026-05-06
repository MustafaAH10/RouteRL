# RouteRL-Drive MVP Spec

## Purpose

Build a driving-first benchmark/training environment for model-native route
planning on real OpenStreetMap data.

The model should learn to infer a valid driving route from rendered map
observations. It should **not** call Google Maps, OSRM, NetworkX, OSMnx, or any
routing API at inference time.

Routing algorithms and OSM graphs are allowed only for:

- dataset generation;
- hidden oracle route generation;
- reward/evaluation;
- debug visualization.

## Core Claim

The target capability is:

```text
Given a real driving map observation, infer a valid route from A to B without
calling a routing API.
```

This is not intended to beat Dijkstra or OSRM. Those systems are the hidden
teacher/verifier. The research question is whether a VLM can internalize enough
visual driving-map behavior to produce valid route proposals from map images.

## Mode Decision

The project should now be **driving-first**.

Initial mode:

```text
mode = drive
```

Use OSMnx with a driving network:

```python
ox.graph_from_bbox(..., network_type="drive")
```

Driving mode matters because the graph and renderer need to respect:

- one-way streets;
- road hierarchy;
- ramps and slip roads;
- non-drivable roads/paths;
- major roads and expressways;
- later: turn restrictions where available.

Walking mode can be added later as a separate environment. Do not mix walking
and driving in the same task format.

## Current Problem With Dense Node Labels

The current smoke-test render with many labels like `N1...N60` is not a good
long-term model input.

Problems:

- labels overlap;
- OCR becomes harder than routing;
- visual clutter hides minor-road structure;
- route outputs become long and brittle;
- real-world route maps do not expose every graph node as a label.

Keep dense labels only as a quick infrastructure smoke test.

## Recommended MVP

Implement the next MVP as:

```text
RouteRL-Drive v1
```

Scope:

```text
city: Singapore
mode: drive
distance: 500m to 2000m
input: clean top-down driving map
output: sparse turn checkpoint sequence
verifier: hidden directed OSM driving graph
visualization: debug overlays first, Streets-GL later
```

## Model Input

Use a clean driving map rather than dense node labels.

Image should include:

- blue `A` start marker;
- red `B` destination marker;
- drivable roads only or strongly emphasized;
- road hierarchy styling;
- one-way arrows on directed roads;
- expressways/major roads visually distinct;
- sparse candidate turn checkpoints, e.g. `T1...T20`;
- no hidden oracle route.

Recommended visual style:

```text
major roads / expressways: thick blue or dark strokes
arterial roads: medium dark strokes
minor roads: thin gray strokes
one-way roads: small arrows in travel direction
ramps/slip roads: orange or distinct thin strokes
non-drivable roads/paths: hidden or very faint
turn checkpoints: sparse high-contrast markers
```

Do not show adjacency lists or edge lengths as text.

## Model Output

### Phase 1: Sparse Turn Checkpoints

For the first serious driving benchmark, ask the model to output sparse turn
checkpoints:

```json
{
  "turns": ["T2", "T5", "T9"]
}
```

The verifier converts this into:

```text
A -> T2 -> T5 -> T9 -> B
```

and routes through those checkpoints on the hidden directed driving graph.

### Phase 2: Normalized Waypoints

After sparse checkpoints work, remove labels and ask for normalized route
waypoints on the canonical map:

```json
{
  "route_points": [
    {"x": 0.14, "y": 0.78},
    {"x": 0.33, "y": 0.62},
    {"x": 0.61, "y": 0.41}
  ]
}
```

The verifier maps screen points to lat/lon, snaps them to the hidden driving
graph, routes through the snapped points, and scores the result.

## Long Trips And Cross-Island Trips

Do not use one giant map image for long trips. A cross-island route needs too
much context for one screenshot.

Use a hierarchical route-strip setup.

### Level 1: Overview Corridor

The model sees a simplified overview map:

```text
wide city map
A and B markers
major roads and expressways
coarse corridor/checkpoint grid
```

Output:

```json
{
  "corridor": ["C4", "D4", "E5", "F5"]
}
```

This chooses the broad route corridor, not every turn.

### Level 2: Local Segment Routing

Split the corridor into local segment panels. Each panel is visually manageable
and has its own start/end markers:

```text
S1_A -> S1_B
S2_A -> S2_B
S3_A -> S3_B
```

The model outputs per-segment turns:

```json
{
  "segments": [
    {"segment_id": "S1", "turns": ["T2", "T5"]},
    {"segment_id": "S2", "turns": ["T1", "T4", "T8"]},
    {"segment_id": "S3", "turns": ["T3"]}
  ]
}
```

The verifier stitches segment routes on the hidden directed graph.

## Reward And Verification

Use the hidden OSM driving graph to score predictions.

Reward components:

```text
valid JSON/schema
known checkpoint IDs or valid waypoint coordinates
starts near A
ends near B
directed route exists through predicted checkpoints/waypoints
respects one-way direction because graph is directed
route length close to hidden optimal route
geometry close to hidden optimal route
no excessive loops
reasonable number of turns/waypoints
```

For route-strip tasks:

```text
total_reward =
  average(segment_rewards)
  + stitched_route_reward
  + corridor_reward
```

This gives denser RL feedback than a single all-or-nothing route score.

## Dataset Format

Use JSONL. One task per line.

Example local driving task:

```json
{
  "task_id": "sg_drive_000001",
  "city": "Singapore",
  "mode": "drive",
  "bbox": [103.845, 1.285, 103.855, 1.295],
  "origin": {"lat": 1.2868, "lon": 103.8458, "label": "A"},
  "destination": {"lat": 1.2919, "lon": 103.8509, "label": "B"},
  "images": {
    "map": "data/experiments/short_500m_2km/maps/sg_drive_000001/map.png"
  },
  "turn_checkpoints": {
    "T1": {"lat": 1.2871, "lon": 103.8460},
    "T2": {"lat": 1.2880, "lon": 103.8467}
  },
  "oracle": {
    "provider": "osmnx_drive",
    "distance_m": 1320.4,
    "geometry": [[103.8458, 1.2868], [103.8460, 1.2871]],
    "gold_turn_route": ["T1", "T2"]
  },
  "prompt": "Trace a valid driving route from A to B using the sparse turn checkpoints."
}
```

Example route-strip task:

```json
{
  "task_id": "sg_drive_strip_000001",
  "city": "Singapore",
  "mode": "drive",
  "images": {
    "overview": "data/experiments/route_strip_demo/maps/sg_drive_strip_000001_strip/overview.png",
    "segments": [
      "data/experiments/route_strip_demo/maps/sg_drive_strip_000001_strip/s01.png",
      "data/experiments/route_strip_demo/maps/sg_drive_strip_000001_strip/s02.png"
    ]
  },
  "segments": [
    {"segment_id": "S01", "start": "S01_A", "end": "S01_B"},
    {"segment_id": "S02", "start": "S02_A", "end": "S02_B"}
  ],
  "oracle": {
    "distance_m": 9300.0,
    "segment_gold_routes": {
      "S01": ["T02", "T05"],
      "S02": ["T01", "T08"]
    }
  }
}
```

## Baselines

Implement before RL:

- oracle route;
- random checkpoint route;
- greedy checkpoint route;
- straight-line waypoint route;
- base Hugging Face VLM prompt-only route;
- later: SFT model;
- later: RL model.

## Hugging Face GPU Workflow

Use Hugging Face models on a GPU instance. The canonical setup and experiment
commands live in `docs/model_io_and_dataflow.md`.

## Streets-GL Role

Streets-GL is a visualization layer, not the first verifier.

Use it to render:

- RouteRL predicted route;
- Google/OSRM comparison route;
- final demos;
- optional extra visual context later.

The canonical route-output frame should remain a top-down 2D map because it is
easier to score and align with OSM graph geometry.

## Immediate Next Steps

1. Move to a GPU instance and run the Hugging Face setup script.
2. Generate a small Singapore `drive` task set with sparse turn checkpoints.
3. Render and inspect the checkpoint maps for readability.
4. Run `Qwen/Qwen3-VL-4B-Instruct` prompt-only predictions.
5. Evaluate oracle, random, greedy, and VLM predictions with the directed graph
   verifier.
6. Add route-strip task generation for longer trips.
7. Add RL training around segment-level and stitched-route rewards.
