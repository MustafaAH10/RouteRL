# CUA-Lite Map Environment

## Status

This checkpoint-marker CUA prototype is diagnostic, not the recommended
training surface. Expanded-budget runs can make every verifier-expanded route
directed and reachable, but they still wander badly because the model is allowed
to mark arbitrary checkpoint labels. Use `docs/trace_choice_env.md` for the
current graph-native sandbox.

This is a deterministic prototype for map-using routing agents. It is not a
browser controller and it does not train a model. It shows what a future RL
environment can expose while reusing the current RouteRL tasks and verifier.

## Idea

The static VLM baseline does this:

```text
all images -> one JSON route prediction
```

The CUA-lite environment does this:

```text
observe viewport -> pan/zoom -> mark one local checkpoint -> repeat -> finish
```

The final `finish` action converts marked labels into the same prediction JSON
already scored by `route_env.verify`.

## Actions

Route-strip actions:

```json
{"tool":"open_overview"}
{"tool":"open_segment","segment_id":"S03"}
{"tool":"mark","segment_id":"S03","turn":"T071"}
{"tool":"unmark","segment_id":"S03","turn":"T071"}
{"tool":"clear_segment","segment_id":"S03"}
{"tool":"finish"}
```

Flat task actions:

```json
{"tool":"open_map"}
{"tool":"pan","direction":"northwest"}
{"tool":"zoom_in"}
{"tool":"zoom_out"}
{"tool":"zoom_to_label","turn":"T07"}
{"tool":"center_on_start"}
{"tool":"center_on_end"}
{"tool":"center_on_last_mark"}
{"tool":"pan_toward_destination"}
{"tool":"mark","turn":"T07"}
{"tool":"unmark","turn":"T07"}
{"tool":"finish"}
```

For current CUA experiments, prefer flat tasks with `--initial-view start`.
That starts the rendered viewport near blue A instead of dumping the whole map
into the first observation.

## Observation Shape

Each step returns a dictionary like:

```json
{
  "task_id": "singapore_drive_000001_strip",
  "task_type": "route_strip",
  "view": {
    "kind": "segment",
    "segment_id": "S01",
    "image": "data/experiments/.../s01.png"
  },
  "visible_labels": ["T001", "T002"],
  "markable_labels": ["T002"],
  "frontier_candidates": ["T002"],
  "visible_segments": [],
  "prediction_so_far": {
    "segments": [{"segment_id": "S01", "turns": ["T001"]}]
  },
  "remaining_actions": 42,
  "done": false,
  "last_error": null,
  "reward": null,
  "metrics": null
}
```

`visible_labels` are labels rendered in the current image. `markable_labels`
remove labels already in the prediction. `frontier_candidates` are markable
labels sorted from the current route frontier. The model should use
`frontier_candidates` for `mark` actions.

After `finish`, `reward` is the verifier score and `metrics` includes fields
such as `valid_route`, `checkpoint_reward`, `length_ratio`, and
`num_predicted_turns`.

## Demo

Compare three deterministic trajectories on the first route-strip task:

```bash
source routerl/bin/activate
python scripts/demo_map_env.py --mode compare
```

The demo runs:

```text
empty       opens the task and finishes with no marks
all-labels marks every visible checkpoint, reproducing the common VLM failure
oracle      opens each segment and marks the hidden teacher labels
```

Write full traces:

```bash
python scripts/demo_map_env.py \
  --mode compare \
  --write-trace data/experiments/long_8_25km_route_strip_probe/agent_traces/cua_lite_compare.jsonl
```

View the deterministic trace in the browser:

```bash
bash scripts/serve_overlay.sh --port 8000 --bind 0.0.0.0
```

```text
http://localhost:8000/renderer/route_overlay.html?tasks=/data/experiments/long_8_25km_route_strip_probe/tasks.jsonl&trace=/data/experiments/long_8_25km_route_strip_probe/agent_traces/cua_lite_compare.jsonl&mode=oracle
```

Run and view a short real VLM tool rollout. Open this URL first if you want to
watch the trace file update while the rollout runs:

```text
http://localhost:8000/renderer/route_overlay.html?tasks=/data/experiments/long_8_25km_route_strip_probe/tasks.jsonl&trace=/data/experiments/long_8_25km_route_strip_probe/agent_traces/qwen3_vl_8b_tool_live.jsonl&poll=1
```

Then run:

```bash
python scripts/run_hf_tool_agent.py \
  --tasks data/experiments/long_8_25km_route_strip_probe/tasks.jsonl \
  --out data/experiments/long_8_25km_route_strip_probe/agent_traces/qwen3_vl_8b_tool_live.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --local-files-only \
  --max-actions 8 \
  --max-new-tokens 128 \
  --device auto \
  --dtype bfloat16
```

The runner writes the trace after each action. The browser polls it every two
seconds when `poll=1` is set.

```text
http://localhost:8000/renderer/route_overlay.html?tasks=/data/experiments/long_8_25km_route_strip_probe/tasks.jsonl&trace=/data/experiments/long_8_25km_route_strip_probe/agent_traces/qwen3_vl_8b_tool_limit8.jsonl
```

## Current Expanded-Budget Probe

The current cleaned short-route sandbox artifact uses:

- local start viewport;
- at most 8 visible labels;
- `road_follow` prompting;
- action repair for obvious illegal/model-error actions;
- prefix-validity masking so a new mark cannot make the ordered waypoint route
  unroutable on the road graph;
- a 40-action budget;
- finish guard once no local labels remain after 10 marked turns.

Run it:

```bash
routerl/bin/python scripts/run_hf_tool_agent.py \
  --tasks data/experiments/short_500m_2km/tasks.jsonl \
  --num-tasks 20 \
  --out data/experiments/short_500m_2km/agent_traces/strategy_road_follow_prefix_valid_budget40_all20.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --local-files-only \
  --max-actions 40 \
  --max-new-tokens 128 \
  --device auto \
  --dtype bfloat16 \
  --strategy road_follow \
  --viewport-dir data/experiments/short_500m_2km/agent_viewports_road_follow_prefix_valid_budget40_all20 \
  --initial-view start \
  --viewport-scale 0.42 \
  --max-visible-labels 8 \
  --repair-actions \
  --enforce-prefix-validity \
  --finish-on-empty-markable-after 10
```

View it:

```text
http://localhost:8000/renderer/route_overlay.html?tasks=/data/experiments/short_500m_2km/tasks.jsonl&trace=/data/experiments/short_500m_2km/agent_traces/strategy_road_follow_prefix_valid_budget40_all20.jsonl
```

Summarize it:

```bash
routerl/bin/python scripts/summarize_tool_traces.py \
  --tasks data/experiments/short_500m_2km/tasks.jsonl \
  --trace data/experiments/short_500m_2km/agent_traces/strategy_road_follow_prefix_valid_budget40_all20.jsonl
```

Observed summary:

```text
records=20
valid_route=20/20
score mean=0.614 median=0.613 min=0.476 max=0.802
length_ratio mean=3.415 median=3.258 min=1.000 max=6.995
direction_check missing_edges=0/829
```

The direction check confirms that verifier-expanded agent routes use directed
edges from the task graph. Remaining weakness is route quality: detours,
under-marking, and occasional over-marking. The expanded budget did not improve
mean score; it mostly gave the model more room to wander after it had already
found a legal route.

When viewing a trace without a separate `results=` file, the overlay now draws
the final agent route from `trace.metrics.agent_geometry`. This matters because
checkpoint rings alone can make a legal verifier path look like an illegal
straight-line jump.

## What This Is For

Use this to design curriculum RL infrastructure before training:

- action schema;
- observation schema;
- action budgets;
- invalid-action handling;
- action masking / repair;
- sparse marking behavior;
- local viewport curriculum;
- final reward from the hidden graph verifier.

The next infra step is not SFT or RL. It is wiring a model-facing loop that asks
a VLM to emit one tool action at a time against this environment.
