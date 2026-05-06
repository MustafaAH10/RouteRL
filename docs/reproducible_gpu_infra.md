# Reproducible GPU Instance Flow

This repo is designed for machines that may disappear often. Treat generated
tasks, maps, model caches, and run outputs as artifacts mounted outside the
checkout.

## Recommended 8xH100 Flow

```bash
git clone <YOUR_REPO_URL> RouteRL
cd RouteRL
git checkout <PINNED_COMMIT>

docker build -f infra/Dockerfile.h100 -t routerl-h100:<PINNED_COMMIT> .

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /mnt/hf:/mnt/hf \
  -v /mnt/routerl_artifacts:/mnt/routerl_artifacts \
  -e HF_HOME=/mnt/hf \
  -e ROUTERL_ARTIFACT_ROOT=/mnt/routerl_artifacts \
  routerl-h100:<PINNED_COMMIT> \
  bash scripts/smoke_h100_instance.sh --expected-gpus 8
```

If you are not using Docker:

```bash
EXPECTED_GPU_COUNT=8 \
HF_HOME=/mnt/hf \
HF_MODEL=Qwen/Qwen3-VL-8B-Instruct \
bash scripts/setup_gpu_instance.sh

source routerl/bin/activate
bash scripts/smoke_h100_instance.sh --expected-gpus 8
```

## Artifact Layout

Use mounted storage for anything expensive to recreate:

```text
/mnt/routerl_artifacts/
  experiments/
  verl/
  logs/

/mnt/hf/
  hub/
```

Then point experiment folders at the mounted path:

```bash
EXP=/mnt/routerl_artifacts/experiments/long_8_25km_route_strip_probe
mkdir -p "$EXP"/{maps,predictions,results,overlays}
```

## Prefetch Models

Pin model revisions when possible:

```bash
HF_HOME=/mnt/hf \
bash scripts/prefetch_hf_assets.sh \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --revision <MODEL_COMMIT_OR_TAG>
```

For fully repeatable runs, record:

- git commit;
- Docker image tag or Python lockfile;
- `HF_MODEL` and `HF_MODEL_REVISION`;
- generated `tasks.jsonl`;
- rendered maps;
- OSMnx cache snapshot or artifact bundle;
- prediction/result JSONL files.

## Current Caveats

The repo now has smoke scripts and a Dockerfile, but dependency pinning is still
coarser than ideal. Before large paid runs, freeze Python packages into a lock
file or immutable Docker image and keep that image tag with the experiment
manifest.
