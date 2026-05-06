#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/verl/routerl_qwen_vl_smoke.yaml}"

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("verl") else 1)
PY
then
  cat >&2 <<'EOF'
The `verl` package is not installed in this environment.

Install or run from a pinned VeRL checkout first, for example:
  git clone https://github.com/verl-project/verl external/verl
  python -m pip install -e external/verl

Then re-run this script.
EOF
  exit 1
fi

echo "RouteRL VeRL config: ${CONFIG}"
echo "This repo provides the dataset/reward adapter; launch the matching VeRL trainer with this config."
echo "For now, inspect docs/verl_integration.md before starting a paid multi-GPU run."
