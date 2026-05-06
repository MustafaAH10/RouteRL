# RouteRL

RouteRL is an infrastructure stack for testing whether a VLM can learn
driving-route behavior from real map observations without calling a router at
inference time.

The current milestone is driving-first:

- generate real OSM driving graph crops;
- render model-facing map images with A/B markers, road hierarchy, one-way
  arrows, and sparse turn checkpoints;
- ask a Hugging Face VLM to output checkpoint sequences;
- verify predictions against the hidden directed OSM driving graph;
- keep Streets-GL as an optional visualization layer, not the core verifier.
- split long routes into overview + local segment route-strip images.

## Current Interface

The model sees a rendered map image plus a list of allowed visible checkpoint
labels. It does not receive graph adjacency, edge lengths, OSM metadata, or the
hidden shortest path.

Expected prediction:

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
- [GPU setup, dataflow, and experiment commands](docs/model_io_and_dataflow.md)
- [Current progress and goals](docs/progress_and_goals.md)
- [Task design notes](docs/task_design.md)
- [Experiment report](docs/model_test_report.md)
- [Drive MVP spec](routesight_rl_mvp_spec.md)
- [Reproducible GPU instance flow](docs/reproducible_gpu_infra.md)
- [VeRL integration notes](docs/verl_integration.md)

## Project Direction

Single-panel short routes are useful for debugging, but longer routes should use
a hierarchical route-strip interface: overview corridor image, local segment
images, per-segment predictions, and stitched graph verification.
