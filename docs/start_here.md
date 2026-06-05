# Start Here

RouteRL trains/evaluates visual routing agents on OSM-derived Singapore driving
tasks. The old baseline asks a VLM to output checkpoint labels from rendered map
images. The current recommended sandbox is graph-native trace choice: the agent
chooses among valid directed road continuations instead of inventing arbitrary
checkpoint jumps.

## First Command

From the repo root:

```bash
source routerl/bin/activate
bash scripts/smoke_h100_instance.sh --expected-gpus 1
```

Use `--expected-gpus 1` on this machine: it has one 96 GB RTX PRO 6000 Blackwell GPU, not eight H100s.

A good setup prints CUDA visibility, unit tests passing, oracle scoring at `mean_score=1.000`, and `RouteRL H100 smoke check passed`.

## Current Recommended Sandbox

```bash
source routerl/bin/activate
routerl/bin/python scripts/run_trace_choice_policy.py \
  --tasks data/experiments/short_500m_2km/tasks.jsonl \
  --num-tasks 20 \
  --out data/experiments/short_500m_2km/agent_traces/trace_choice_shortest_all20.jsonl \
  --policy shortest \
  --max-steps 256 \
  --trace-length-m 350
```

Expected: `20/20` valid routes, score `1.000`, length ratio `1.000`.

Run the same upper-bound check on 8-25 km tasks:

```bash
routerl/bin/python scripts/run_trace_choice_policy.py \
  --tasks data/experiments/long_8_25km_80cp_probe/tasks.jsonl \
  --num-tasks 10 \
  --out data/experiments/long_8_25km_80cp_probe/trace_choice_shortest_all10.jsonl \
  --policy shortest \
  --max-steps 512 \
  --trace-length-m 800
```

Expected: `10/10` valid routes, score `1.000`, length ratio `1.000`.

## One VLM Harness Sanity Probe

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

Observed on this machine: `20/20` valid, `20/20` exact, mean score `1.000`,
mean length ratio `1.000`.

The same ranked trace-choice harness also scored `10/10` exact on
`data/experiments/long_8_25km_80cp_probe/tasks.jsonl`; the trace is
`data/experiments/long_8_25km_80cp_probe/hf_trace_choice_qwen_all10_planner_rank.jsonl`.

This is a model-I/O and environment sanity check, not a pure routing benchmark.
`planner_hint` exposes graph lookahead through estimated remaining distance and
planner rank, and the prompt tells the model to choose rank 1.

For an honest next-continuation probe, remove planner rank, automatic
single-candidate skipping, and invalid-action repair:

```bash
routerl/bin/python scripts/run_hf_trace_choice_agent.py \
  --tasks data/experiments/singapore_trace_choice_4k/short_500m_2km/tasks.jsonl \
  --num-tasks 10 \
  --out data/experiments/singapore_trace_choice_4k/short_500m_2km/benchmarks/qwen_trace_choice_visual_norepair_sample10.jsonl \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --local-files-only \
  --device auto \
  --dtype bfloat16 \
  --max-steps 96 \
  --trace-length-m 350 \
  --max-candidates 6 \
  --max-new-tokens 96 \
  --prompt-strategy visual \
  --render-context local \
  --render-dir data/experiments/singapore_trace_choice_4k/short_500m_2km/benchmarks/qwen_viewports_visual_norepair_sample10 \
  --no-repair-invalid-actions
```

Observed on this machine for the first 10 short tasks: `7/10` valid, mean score
`0.615`. This is still interactive next-continuation, not one-shot route
estimation.

## Mental Model

```text
OSM directed graph -> environment action space and hidden verifier
Rendered local viewport -> model input
Candidate JSON -> legal road continuations C1/C2/C3
Model JSON -> {"tool":"choose","candidate_id":"C2"}
Final route -> scored against the oracle graph route
```

The old checkpoint-label interface still exists, but it is not the recommended
baseline for RL. It lets the model create jagged, globally inconsistent routes.
Trace choice makes one-way validity and reachability structural.

## GUI

The static overlay server is already enough:

```bash
bash scripts/serve_overlay.sh --port 8000 --bind 0.0.0.0
```

From your laptop, forward the port:

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<gpu-host>
```

Then open:

```text
http://127.0.0.1:8000/renderer/route_overlay.html?tasks=/data/experiments/short_500m_2km/tasks.jsonl&trace=/data/experiments/short_500m_2km/agent_traces/hf_trace_choice_qwen_all20_overview_planner_rank.jsonl
```

See `docs/trace_choice_env.md` for the action schema, results, and recommended
project direction.

## Training Status

Generation, rendering, verification, reward code, trace-choice rollout, and HF
inference are present. Full VeRL training is not plug-and-play yet because VeRL
is not vendored or pinned here. Do not start with RL until the trace-choice
baseline is the interface you want to optimize.
