#!/usr/bin/env bash
# Launch hexfield training. Sets the LOAD-BEARING architecture env (a checkpoint
# only loads into a net built with the same CHANNELS/HEADS/TRUNK — these match
# the shipped weights in models/) plus the parity-gated perf kernels, then either
# starts the auto-relaunch supervisor detached (default) or runs one training
# process in the foreground (--foreground).
#
# Usage:
#   scripts/launch_training.sh              # detached supervisor
#   scripts/launch_training.sh --foreground # single run, attached
# Env overrides: CONFIG, RUNDIR, HEXO_VENV (default .venv at repo root).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEXO_VENV="${HEXO_VENV:-$ROOT/.venv}"
CONFIG="${CONFIG:-$ROOT/configs/hexfield_main_7.toml}"

# Architecture (LOAD-BEARING — read at import; must match the shipped weights).
export HEXFIELD_CHANNELS="${HEXFIELD_CHANNELS:-192}"
export HEXFIELD_ATTENTION_HEADS="${HEXFIELD_ATTENTION_HEADS:-3}"
export HEXFIELD_TRUNK="${HEXFIELD_TRUNK:-CCACCACCACCACCA}"
export HEXFIELD_SUPPORT_RADIUS="${HEXFIELD_SUPPORT_RADIUS:-4}"

# Parity-gated serve/train perf kernels (safe on GPU; ignored on CPU/eager).
# NOTE: ASYNC_EVAL is presence-gated in Rust — setting it to ANY value (even 0)
# enables it; unset the variable entirely to disable.
export HEXFIELD_ASYNC_EVAL="${HEXFIELD_ASYNC_EVAL:-1}"
export HEXFIELD_DEFER_DECODE="${HEXFIELD_DEFER_DECODE:-1}"
export HEXFIELD_SERVE_FLEX="${HEXFIELD_SERVE_FLEX:-1}"
export HEXFIELD_TRITON_CONV="${HEXFIELD_TRITON_CONV:-1}"
export HEXFIELD_TRITON_CONV_LN="${HEXFIELD_TRITON_CONV_LN:-1}"
export HEXFIELD_TRITON_ATTN="${HEXFIELD_TRITON_ATTN:-1}"
export HEXFIELD_FLEX_PAIR="${HEXFIELD_FLEX_PAIR:-1}"
export HEXFIELD_SERVE_HALF="${HEXFIELD_SERVE_HALF:-1}"
export HEXFIELD_RUST_PACK="${HEXFIELD_RUST_PACK:-1}"
export HEXFIELD_COPY_STREAM="${HEXFIELD_COPY_STREAM:-1}"
export HEXFIELD_TRAIN_FLEX="${HEXFIELD_TRAIN_FLEX:-1}"
export HEXFIELD_TRAIN_COMPILE="${HEXFIELD_TRAIN_COMPILE:-1}"
# glibc malloc tunables — trim/mmap thresholds that reduce host RSS churn.
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-536870912}"
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-536870912}"
export MALLOC_TOP_PAD_="${MALLOC_TOP_PAD_:-134217728}"

if [[ "${1:-}" == "--foreground" ]]; then
  export PYTHONPATH="$ROOT/packages/hexfield/python${PYTHONPATH:+:$PYTHONPATH}"
  exec "$HEXO_VENV/bin/python" -u -m hexo_train.cli.train_model "$CONFIG"
fi

# Detached supervisor (survives the parent shell). Env exported above is
# inherited by supervise.sh.
export CONFIG
setsid nohup bash "$ROOT/scripts/supervise.sh" >/dev/null 2>&1 < /dev/null &
echo "launched supervisor pid=$! config=$CONFIG"
