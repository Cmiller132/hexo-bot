#!/usr/bin/env bash
# Launch the :8080 dashboard (hexo_frontend.web, stdlib HTTP server) detached. It scans
# $HEXO_DEBUG_RUN_ROOT/runs (default: repo root), so a fresh run dir shows up
# automatically. Single-instance: no-op if one is already listening.
#
# Env: HEXO_VENV (default .venv at repo root), HEXO_DEBUG_RUN_ROOT (default repo
# root), SEALBOT_PATH (optional external SealBot; passed through if set),
# HEXO_HOST (default 127.0.0.1 - set 0.0.0.0 to expose on the LAN).
# Usage: scripts/dashboard.sh [PORT]   (default 8080)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEXO_VENV="${HEXO_VENV:-$ROOT/.venv}"
RUNROOT="${HEXO_DEBUG_RUN_ROOT:-$ROOT}"
PORT="${1:-8080}"

if pgrep -f "hexo_frontend[.]web.*--port $PORT" >/dev/null 2>&1; then
  echo "dashboard already running on :$PORT: $(pgrep -f "hexo_frontend[.]web.*--port $PORT" | tr '\n' ' ')"
  exit 0
fi

# hexo_engine/utils/runner/train resolve from the venv, but hexfield and the
# frontend are imported from the source tree.
export PYTHONPATH="$ROOT/packages/hexfield/python:$ROOT/packages/hexo_frontend/python:$ROOT/packages/hexo_engine/python:$ROOT/packages/hexo_utils/python:$ROOT/packages/hexo_runner/python:$ROOT/packages/hexo_train/python"
# Featurization radius must match the loaded weights (shipped main_7 weights
# trained at radius 4; the code default is 8). Override for your own runs.
export HEXFIELD_SUPPORT_RADIUS="${HEXFIELD_SUPPORT_RADIUS:-4}"
export HEXO_DEBUG_RUN_ROOT="$RUNROOT"
cd "$RUNROOT"
LOG="$RUNROOT/dashboard.out.log"

SEALBOT_ARGS=()
if [ -n "${SEALBOT_PATH:-}" ]; then
  SEALBOT_ARGS=(--sealbot-path "$SEALBOT_PATH")
fi

setsid "$HEXO_VENV/bin/python" -u -m hexo_frontend.web --host "${HEXO_HOST:-127.0.0.1}" --port "$PORT" \
  "${SEALBOT_ARGS[@]}" >"$LOG" 2>&1 &
echo "launched dashboard pid=$! port=$PORT log=$LOG"
sleep 4
echo "--- log tail ---"; tail -12 "$LOG" 2>/dev/null
echo "--- listening? ---"; ss -ltn 2>/dev/null | grep ":$PORT" || echo "NOT yet listening on :$PORT"
