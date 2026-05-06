# VeRL Integration Notes

RouteRL has a verifiable reward: model JSON is parsed, expanded through the
hidden directed OSM driving graph, and scored by `route_env.verify`. The VeRL
work is therefore mostly an adapter problem.

Upstream VeRL currently lives at `https://github.com/verl-project/verl`; its
trainer schemas can change, so this repo keeps the RouteRL side explicit and
small.

## RouteRL Pieces Added

- `route_env.reward.RouteRewarder`: loads `tasks.jsonl` and scores raw model
  responses with `verify_prediction`.
- `scripts/prepare_verl_dataset.py`: exports flat or route-strip tasks into
  trainer records with `task_id`, image paths, prompt text, and reward metadata.
- `configs/verl/routerl_qwen_vl_smoke.yaml`: names the intended model, data,
  reward module, and 8xH100 runtime assumptions.
- `scripts/run_verl_routerl.sh`: guardrail launcher that refuses to pretend VeRL
  is installed when it is not.

## Prepare A Route-Strip Dataset

```bash
EXP=data/experiments/long_8_25km_route_strip_probe

python scripts/prepare_verl_dataset.py \
  --tasks "$EXP/tasks.jsonl" \
  --out "$EXP/verl_train.jsonl" \
  --split train
```

If `pyarrow` is installed, parquet output also works:

```bash
python scripts/prepare_verl_dataset.py \
  --tasks "$EXP/tasks.jsonl" \
  --out "$EXP/verl_train.parquet" \
  --format parquet
```

## Reward Adapter Contract

The reward worker needs the full hidden task file, but the model prompt must not
include hidden graph metadata.

```python
from route_env.reward import RouteRewarder

rewarder = RouteRewarder("data/experiments/long_8_25km_route_strip_probe/tasks.jsonl")
result = rewarder.score_response(
    "singapore_drive_000001_strip",
    '{"segments":[{"segment_id":"S01","turns":["T01"]}]}',
)
print(result.reward)
print(result.metrics["valid_route"])
```

Recommended logged metrics:

```text
score
valid_schema
valid_route
checkpoint_reward
length_ratio
mean_route_distance_m
unknown_turn_count
num_predicted_turns
```

## Minimal RL Roadmap

1. Start with frozen short-route tasks and route-strip tasks.
2. Use SFT or oracle JSON as a warm start if the base VLM cannot emit valid JSON
   reliably.
3. Train with a verifiable reward over JSON responses only.
4. Keep image paths/prompts in the dataset record; keep hidden graphs only in the
   reward worker.
5. Evaluate every checkpoint with the existing prediction/evaluation scripts,
   not just the trainer's reward logs.

## What Is Still Not Automatic

The repo does not yet vendor or pin VeRL. Clone/install it into the GPU image or
use an existing VeRL image, then wire its trainer entrypoint to:

- the dataset emitted by `scripts/prepare_verl_dataset.py`;
- the model named in `configs/verl/routerl_qwen_vl_smoke.yaml`;
- `route_env.reward.RouteRewarder` as the reward function.
