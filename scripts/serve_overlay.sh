#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
BIND="${BIND:-0.0.0.0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --bind)
      BIND="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."
echo "Serving RouteRL overlay from $(pwd)"
echo "Local URL: http://127.0.0.1:${PORT}/renderer/route_overlay.html?tasks=/data/experiments/long_8_25km_route_strip_probe/tasks.jsonl"
exec python -m http.server "${PORT}" --bind "${BIND}"
