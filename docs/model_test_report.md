# RouteRL Model Test Report

This report summarizes the current generated experiments after the data cleanup.
Generated artifacts now live under:

```text
data/experiments/<experiment_name>/
  tasks.jsonl
  maps/
  predictions/
  results/
  overlays/
```

The old flat folders such as `data/debug`, `data/debug_prompt2`,
`data/rendered`, `data/rendered_clean`, `data/report_rendered`,
`data/predictions`, `data/results`, and `data/tasks` were removed because task
IDs repeat across experiments and those folders made outputs ambiguous.

## Prompt Used For Qwen

The Hugging Face runner sends one map image plus this prompt:

```text
You are a routing model.

You are given a real driving map image. Roads are drawn with hierarchy styling, one-way arrows show directed streets where known, the blue marker A is the start, and the red marker B is the destination.

Black T-labels mark sparse turn checkpoints and decision points. The T numbers are arbitrary labels, not route order. Infer a plausible short driving route from A to B by choosing only the checkpoints that the driver should physically pass through, in driving order. Do not invent labels.

Return JSON only with exactly this shape:
{"turns":["T1","T2"],"confidence":0.0,"reason":"brief"}

Allowed turn checkpoint labels: T1, T2, ...
```

The final line is generated per task from that task's visible checkpoint labels.

## Current Experiments

| Experiment | Path | Count | Distance | Gold turns | Checkpoints | Kept outputs |
|---|---|---:|---:|---:|---:|---|
| Short | `data/experiments/short_500m_2km/` | 20 | 613m-1714m | 6-16 | 24 | oracle, empty, random, greedy, Qwen 4B/8B |
| Medium | `data/experiments/medium_2_6km/` | 5 | 2654m-5435m | 15-40 | 40 | oracle, empty, Qwen 8B limit 2 |
| Long 80cp probe | `data/experiments/long_8_25km_80cp_probe/` | 10 | 8622m-23515m | 80 capped | 80 | oracle only |
| Long 200cp probe | `data/experiments/long_8_25km_200cp_probe/` | 2 | 21593m-23276m | 98-136 | 200 | maps only |

## Short 500m-2km

Example map:

![short task 1](../data/experiments/short_500m_2km/maps/singapore_drive_000001/map.png)

Kept prediction/result files:

```text
data/experiments/short_500m_2km/predictions/oracle.jsonl
data/experiments/short_500m_2km/predictions/empty.jsonl
data/experiments/short_500m_2km/predictions/random.jsonl
data/experiments/short_500m_2km/predictions/greedy.jsonl
data/experiments/short_500m_2km/predictions/qwen3_vl_4b_clean_sanitized_limit5.jsonl
data/experiments/short_500m_2km/predictions/qwen3_vl_8b_clean_sanitized_limit5.jsonl
```

Current short-run summary:

| Predictor | Mean score | Success | Valid routes |
|---|---:|---:|---:|
| Oracle | 1.000 | 20/20 | 20/20 |
| Empty baseline | 0.050 | 0/20 | 0/20 |
| Random | 0.277 | 0/20 | 7/20 |
| Greedy | 0.469 | 0/20 | 14/20 |
| Qwen3-VL-4B clean sanitized, limit 5 | 0.235 | 0/5 | 1/5 |
| Qwen3-VL-8B clean sanitized, limit 5 | 0.402 | 1/5 | 2/5 |

The 8B model improved over 4B, but it is still not reliably routing. Its best
short result was:

```text
task: singapore_drive_000003
score: 0.769
valid_route: true
predicted: ["T4", "T11", "T14", "T2"]
oracle: ["T4", "T10", "T11", "T12", "T19", "T24", "T13", "T1", "T20", "T7", "T14", "T2"]
```

Debug overlays are intentionally named after the run:

```text
data/experiments/short_500m_2km/overlays/qwen3_vl_4b_clean_sanitized_limit5/
data/experiments/short_500m_2km/overlays/qwen3_vl_8b_clean_sanitized_limit5/
```

Each overlay folder now contains only the tasks that actually have predictions.

## Medium 2-6km

Example map:

![medium task 1](../data/experiments/medium_2_6km/maps/singapore_drive_000001/map.png)

Current medium-run summary:

| Predictor | Mean score | Success | Valid routes |
|---|---:|---:|---:|
| Oracle | 1.000 | 5/5 | 5/5 |
| Empty baseline | 0.050 | 0/5 | 0/5 |
| Qwen3-VL-8B clean sanitized, limit 2 | 0.620 | 0/2 | 2/2 |

The medium 8B run produced legal graph routes, but they were still too loose:

```text
mean_length_ratio=1.906
mean_route_distance_m=32.5
```

Overlay path:

```text
data/experiments/medium_2_6km/overlays/qwen3_vl_8b_clean_sanitized_limit2/
```

## Long 8-25km

The 80-checkpoint long probe lives at:

```text
data/experiments/long_8_25km_80cp_probe/
```

It generated 10 tasks from `8622m` to `23515m`, but every task hit the
checkpoint cap:

```text
gold turns: 80 min / 80 mean / 80 max
checkpoints: 80
oracle: mean_score=1.000, success=10/10, valid_route=10/10
```

The 200-checkpoint probe lives at:

```text
data/experiments/long_8_25km_200cp_probe/
```

Example:

![long 200cp task 1](../data/experiments/long_8_25km_200cp_probe/maps/singapore_drive_000001/map.png)

Probe stats:

```text
task 1: distance=23276m, gold_turns=136, checkpoints=200, graph_edges=11220
task 2: distance=21593m, gold_turns=98, checkpoints=200, graph_edges=6627
```

This confirms that removing the checkpoint cap is not the real fix. It creates
a huge OCR/clutter task rather than a useful navigation observation.

## Interpretation

Cleaner labels help, and Qwen3-VL-8B is better than Qwen3-VL-4B on the small
sample. But the remaining failure mode is still clear: single-panel routing
forces the model to solve OCR, label association, route planning, and long
sequence output all at once.

The next interface should be route strips:

```text
overview corridor image
  -> local segment image 1
  -> local segment image 2
  -> local segment image 3
  -> stitched verifier
```

Each local segment should keep a small local label set, e.g. `T1`-`T24` or
`T1`-`T40`, and the model should output per-segment turns instead of one massive
global turn list.
