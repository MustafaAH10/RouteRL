# RouteRL

RouteRL is an infrastructure stack for testing whether a multimodal agent can
learn driving-route behavior from OSM-derived map observations without calling a
closed routing API at inference time.

The current milestone is driving-first:

- generate real OSM driving graph crops;
- render model-facing map images with A/B markers, road hierarchy, one-way
  arrows, and sparse turn checkpoints;
- test both static checkpoint prediction and graph-native trace-choice agents;
- verify predictions against the hidden directed OSM driving graph;
- keep Streets-GL as an optional visualization layer, not the core verifier.
- split long routes into overview + local segment route-strip images.

## Current Recommended Interface

The recommended baseline is now trace choice:

```text
current OSM frontier node -> choose one valid directed road continuation C1/C2/C3 -> repeat
```

The model sees a rendered local viewport plus candidate JSON and returns:

```json
{"tool":"choose","candidate_id":"C2"}
```

This keeps one-way validity and reachability inside the action space. The old
static checkpoint interface still exists for comparison, but it is not the
right surface for curriculum RL because it lets the model create jagged,
unreachable waypoint sequences.

Current harness sanity result with Qwen3-VL-8B plus ranked planner hints:

```text
short 500m-2km: 20/20 valid, 20/20 exact
long 8-25km:    10/10 valid, 10/10 exact
```

This result is not a pure routing-capability benchmark: ranked planner hints
expose graph lookahead and tell the model which candidate is shortest. Use it to
verify model I/O and environment validity, not to claim the model can route.

The more honest next-continuation probe removes planner rank, estimated
remaining distance, automatic single-candidate skipping, and invalid-action
repair. On the generated Singapore 4k suite, the first short 10-task sample was:

```text
Qwen3-VL-8B visual no-repair: 7/10 valid, mean score 0.615
```

Legacy static checkpoint prediction looks like:


```json
{"turns":["T3","T8","T12"]}
```

The verifier expands:

```text
A -> T3 -> T8 -> T12 -> B
```

through the hidden directed driving graph and scores schema validity, checkpoint
alignment, route validity, length ratio, geometry similarity, loop behavior, and
segment gaps.

## Data Layout

Generated artifacts live under named experiment folders:

```text
data/experiments/<experiment_name>/
  tasks.jsonl
  maps/
  predictions/
  results/
  overlays/
```

Avoid flat generated folders such as `data/rendered` or `data/debug`; task IDs
repeat across experiments, so named experiment folders prevent accidental
overwrites.

Before running any command that writes to `"$EXP/tasks.jsonl"`, set `EXP` in
that same shell and create the experiment directories:

```bash
EXP=data/experiments/short_500m_2km
mkdir -p "$EXP"/{maps,predictions,results,overlays}
```

Task generation writes only `tasks.jsonl`. Run `scripts/render_tasks.py`
afterward to create the model-facing map PNGs and update each task's image path.

## Documentation

- [Start here: concepts, baselines, and workflow](docs/start_here.md)
- [Trace-choice routing sandbox](docs/trace_choice_env.md)
- [GPU setup, model I/O, and test flow](docs/model_io_and_dataflow.md)
- [Current progress and goals](docs/progress_and_goals.md)
- [Task design notes](docs/task_design.md)
- [Experiment report](docs/model_test_report.md)
- [CUA-lite map environment prototype](docs/cua_lite_map_env.md)
- [Drive MVP spec](routesight_rl_mvp_spec.md)
- [Fresh GPU instance setup](docs/reproducible_gpu_infra.md)
- [VeRL integration notes](docs/verl_integration.md)

## Project Direction

The project direction is graph-native map interaction first, RL second. Use the
trace-choice sandbox to make action validity structural, then train or guard the
local candidate chooser. Browser CUA and Streets-GL are better treated as demos
after the deterministic map environment is stable.
