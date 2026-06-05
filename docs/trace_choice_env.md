# Trace-Choice Routing Sandbox

The checkpoint-marker CUA loop is not the right main training surface. It lets
the model pick arbitrary checkpoint labels, so predictions can look jagged, skip
across roads, or depend on graph masks that are invisible in the image.

Trace choice changes the action space:

```text
current OSM frontier node -> choose one valid outgoing road trace C1/C2/C3 -> repeat
```

The model no longer emits `T17` or `T42` directly. It chooses a colored road
continuation. Checkpoint labels are derived afterward from the driven graph path
for scoring and compatibility with the existing verifier.

## What Is Solved

- Every action follows directed OSM graph edges.
- One-way road validity is structural.
- Unreachable dead-end candidates are hidden when reachable options exist.
- If a candidate reaches destination `B`, the environment offers that
  destination-ending candidate as the useful continuation.
- Done observations expose no new candidates.
- Long routes become tens or hundreds of local branch decisions instead of one
  global label-ordering guess.

This means the current failures are no longer illegal jumps. They are local
policy mistakes among legal branches, which is the problem RL should actually
work on.

## Observation And Action

Observation:

```json
{
  "task_type": "trace_choice",
  "view": {
    "kind": "trace_choice",
    "context": "local_overview",
    "image": "data/experiments/.../step_005.png",
    "current_node": "74460245"
  },
  "route_so_far": {
    "node_count": 7,
    "distance_m": 532.8,
    "turns": ["T11", "T13", "T4"]
  },
  "candidates": [
    {
      "candidate_id": "C1",
      "to_node": "237170380",
      "length_m": 20.7,
      "checkpoint_labels": ["T22"],
      "reachable_to_destination": true,
      "ends_at_destination": false,
      "remaining_distance_m": 282.7
    }
  ]
}
```

Action:

```json
{"tool":"choose","candidate_id":"C1"}
```

Finish is still available, but the environment auto-finishes when the current
frontier reaches destination `B`:

```json
{"tool":"finish"}
```

## Prompt Modes

`visual` mode gives the VLM the viewport, candidate IDs, lengths, checkpoint
labels, reachability, and whether a candidate ends at `B`. This tests whether
the VLM can make local map decisions from the rendered view.

`planner_hint` mode additionally includes:

```json
{
  "remaining_distance_m": 188.4,
  "estimated_total_route_m": 879.2,
  "planner_rank": 1
}
```

This is a graph lookahead hint. `planner_rank: 1` is the shortest legal
continuation under the task graph. It is useful as a strong baseline and
guardrail, but it should be treated as planner assistance, not pure visual
routing.

## Render Modes

`--render-context local` writes the original single local decision viewport.

`--render-context local_overview` writes a two-panel observation:

```text
left:  local road detail around F and the candidate traces
right: graph overview with A, F, B, route-so-far, candidate endpoints, and
       dashed candidate-to-B guides
```

The overview panel improves inspectability and gives visual-only policies more
global context, but by itself it does not fully solve branch choice. Ranked
planner hints are still the stable baseline.

## Commands

Short-route graph upper bound:

```bash
routerl/bin/python scripts/run_trace_choice_policy.py \
  --tasks data/experiments/short_500m_2km/tasks.jsonl \
  --num-tasks 20 \
  --out data/experiments/short_500m_2km/agent_traces/trace_choice_shortest_all20.jsonl \
  --policy shortest \
  --max-steps 256 \
  --trace-length-m 350
```

Long-route graph upper bound:

```bash
routerl/bin/python scripts/run_trace_choice_policy.py \
  --tasks data/experiments/long_8_25km_80cp_probe/tasks.jsonl \
  --num-tasks 10 \
  --out data/experiments/long_8_25km_80cp_probe/trace_choice_shortest_all10.jsonl \
  --policy shortest \
  --max-steps 512 \
  --trace-length-m 800
```

Qwen ranked planner-hint smoke:

```bash
routerl/bin/python scripts/run_hf_trace_choice_agent.py \
  --tasks data/experiments/short_500m_2km/tasks.jsonl \
  --num-tasks 20 \
  --out data/experiments/short_500m_2km/agent_traces/hf_trace_choice_qwen_all20_overview_planner_rank.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --local-files-only \
  --device auto \
  --dtype bfloat16 \
  --max-steps 96 \
  --trace-length-m 350 \
  --max-candidates 6 \
  --max-new-tokens 96 \
  --auto-unique \
  --prompt-strategy planner_hint \
  --render-context local_overview \
  --render-dir data/experiments/short_500m_2km/hf_trace_choice_viewports_qwen_all20_overview_planner_rank
```

Summarize:

```bash
routerl/bin/python scripts/summarize_tool_traces.py \
  --tasks data/experiments/short_500m_2km/tasks.jsonl \
  --trace data/experiments/short_500m_2km/agent_traces/hf_trace_choice_qwen_all20_overview_planner_rank.jsonl
```

## Current Results

Short routes, graph-shortest policy:

```text
records=20
valid_route=20/20
score mean=1.000
length_ratio mean=1.000
direction_check missing_edges=0/223
```

Long 8-25 km routes, graph-shortest policy:

```text
records=10
valid_route=10/10
score mean=1.000
length_ratio mean=1.000
direction_check missing_edges=0/1203
```

Qwen 8B, short 500 m-2 km tasks, local+overview render, ranked planner hints:

```text
records=20
valid_route=20/20
perfect=20/20
score mean=1.000 median=1.000 min=1.000 max=1.000
length_ratio mean=1.000 median=1.000 min=1.000 max=1.000
direction_check missing_edges=0/223
```

Qwen 8B, long 8-25 km tasks, local render, ranked planner hints:

```text
records=10
valid_route=10/10
score mean=1.000
length_ratio mean=1.000
direction_check missing_edges=0/1203
```

Qwen 8B, first three short tasks, local+overview render, visual-only:

```text
valid_route=3/3
score mean=0.952
length_ratio mean=1.136
```

Visual-only now stays legal, but it can still choose a valid detour when the
destination is outside the local viewport. The overview panel makes those
decisions easier to inspect, but the reliable baseline is ranked planner hints.

## GUI

Serve the overlay:

```bash
bash scripts/serve_overlay.sh --port 8000 --bind 0.0.0.0
```

View the Qwen all-20 trace:

```text
http://127.0.0.1:8000/renderer/route_overlay.html?tasks=/data/experiments/short_500m_2km/tasks.jsonl&trace=/data/experiments/short_500m_2km/agent_traces/hf_trace_choice_qwen_all20_overview_planner_rank.jsonl
```

View the long-route Qwen trace:

```text
http://127.0.0.1:8000/renderer/route_overlay.html?tasks=/data/experiments/long_8_25km_80cp_probe/tasks.jsonl&trace=/data/experiments/long_8_25km_80cp_probe/hf_trace_choice_qwen_all10_planner_rank.jsonl
```

View a rendered long-route graph trace:

```text
http://127.0.0.1:8000/renderer/route_overlay.html?tasks=/data/experiments/long_8_25km_80cp_probe/tasks.jsonl&trace=/data/experiments/long_8_25km_80cp_probe/trace_choice_shortest_task1_rendered.jsonl
```

## Recommendation

The best path forward is not browser CUA first and not raw checkpoint-label
prediction. Use this stack:

```text
directed OSM graph action space
  + local rendered candidate viewport
  + optional planner/lookahead hints
  + hidden graph verifier reward
```

For curriculum RL, train the candidate chooser. Start with short routes and
branch-only decisions, then scale to 8-25 km by increasing trace length and task
distance. Keep planner hints as a guardrail/baseline, then ablate them when the
policy is strong enough.

The next useful infra upgrade is an RL-facing curriculum wrapper around this
same action space: short branch-only episodes first, then longer tasks, with
planner-rank behavior as the teacher/guardrail and visual-only behavior as the
ablation to improve.
