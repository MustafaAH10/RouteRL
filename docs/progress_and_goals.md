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

The canonical GPU setup and experiment runbook is now
`docs/model_io_and_dataflow.md`. Keep command snippets there so operational
instructions do not drift across multiple docs.

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
