#!/usr/bin/env bash
set -euo pipefail

EXPECTED_GPUS="${EXPECTED_GPUS:-8}"
TASKS="${TASKS:-data/experiments/long_8_25km_route_strip_probe/tasks.jsonl}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --expected-gpus)
      EXPECTED_GPUS="$2"
      shift 2
      ;;
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

python - <<PY
import json
import subprocess
import sys

expected = int("${EXPECTED_GPUS}")
try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch import failed: {exc}")

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"gpu_{index}:", torch.cuda.get_device_name(index))
if torch.cuda.device_count() < expected:
    raise SystemExit(f"expected at least {expected} GPUs, found {torch.cuda.device_count()}")
PY

python -m unittest discover -s tests

if [[ -f "${TASKS}" ]]; then
  TMP_DIR="$(mktemp -d)"
  python scripts/make_oracle_predictions.py --tasks "${TASKS}" --out "${TMP_DIR}/oracle.jsonl"
  python scripts/evaluate_predictions.py --tasks "${TASKS}" --predictions "${TMP_DIR}/oracle.jsonl" --out "${TMP_DIR}/results.jsonl"
  rm -rf "${TMP_DIR}"
else
  echo "skipping oracle smoke: ${TASKS} not found"
fi

python -m pip check
echo "RouteRL H100 smoke check passed"
