#!/usr/bin/env bash
# Unattended training supervisor: auto-relaunch + circuit breaker +
# single-instance lock + halt flag. Drives the config-driven CLI
# (hexo_train.cli.train_model) and RESUMES from the latest epoch checkpoint
# (resume_from is injected into [checkpoint]; hexo_train prefers resume_from
# over initialize_from, and the hexfield loader restores model+optimizer+epoch
# when resume_from is set).
#
# Config and run dir are env-overridable:
#   CONFIG   (default configs/hexfield_main_7.toml)
#   RUNDIR   (default runs/hexfield_main_7)
#   HEXO_VENV (default .venv at repo root) — the venv whose python is used
# hexfield itself is imported via PYTHONPATH (never pip-installed; see README).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEXO_VENV="${HEXO_VENV:-$ROOT/.venv}"
CONFIG="${CONFIG:-$ROOT/configs/hexfield_main_7.toml}"
RUNDIR="${RUNDIR:-$ROOT/runs/hexfield_main_7}"

CKPTS="$RUNDIR/checkpoints"
SUPLOG="$RUNDIR/supervisor.log"; LOCK="$RUNDIR/supervisor.lock"
HALT="$RUNDIR/supervisor_halted.flag"; DONE="$RUNDIR/supervisor_completed.flag"
PY="$HEXO_VENV/bin/python"
FAST_CRASH_SECONDS=300; MAX_CONSEC_FAST=3; MAX_PER_HOUR=8

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# expandable_segments eases caching-allocator fragmentation under high
# GPU-memory pressure.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# hexfield is imported from the source tree, never installed into the venv.
export PYTHONPATH="$ROOT/packages/hexfield/python${PYTHONPATH:+:$PYTHONPATH}"
# GPU/host overlap in the self-play serve loop (parity-gated; set 0 for sync).
export HEXFIELD_ASYNC_EVAL="${HEXFIELD_ASYNC_EVAL:-1}"
# Serve-only FlexAttention: compute the rel-pos bias inside the attention
# kernel instead of materializing it (parity-gated; set 0 to revert).
export HEXFIELD_SERVE_FLEX="${HEXFIELD_SERVE_FLEX:-1}"
# Deferred-decode: hold per-group decode/softmax out of submit so the serve
# select pass overlaps the forwards (parity-gated; set 0 to keep syncs inside).
export HEXFIELD_DEFER_DECODE="${HEXFIELD_DEFER_DECODE:-1}"
# Optional external SealBot reference opponent (see README + config
# multi_stage_eval). Honored only if set; the eval fails open without it.
# export SEALBOT_PATH=/path/to/SealBot

mkdir -p "$RUNDIR" "$CKPTS"
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$SUPLOG" >&2; }

if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  log "ABORT: another supervisor running (pid $(cat "$LOCK"))"; exit 1
fi
echo $$ > "$LOCK"
[[ -f "$HALT" ]] && { log "ABORT: halt flag present ($HALT). Clear to resume."; rm -f "$LOCK"; exit 1; }
rm -f "$DONE"
trap 'rm -f "$LOCK"' EXIT

latest_ckpt(){ ls -1 "$CKPTS"/epoch_*.pt 2>/dev/null | sort -V | tail -1; }

log "SUPERVISOR start (pid=$$) run=$RUNDIR config=$CONFIG"
log "breaker: fast<${FAST_CRASH_SECONDS}s x${MAX_CONSEC_FAST} OR >${MAX_PER_HOUR}/hr -> halt"

declare -a crash_times=(); consec_fast=0
while :; do
  lc="$(latest_ckpt)"
  if [[ -n "$lc" ]]; then
    USE="$RUNDIR/_resume_config.toml"
    # Inject resume_from right after [checkpoint]; hexo_train prefers it over
    # initialize_from, and the hexfield loader then loads model+optimizer+epoch.
    # If the config has no [checkpoint] table, append one (awk sets found=1
    # only when the anchor exists), so resume still works for minimal configs.
    awk -v c="$lc" '/^\[checkpoint\]/{print; print "resume_from = \"" c "\""; found=1; next} {print} END{if(!found){print "[checkpoint]"; print "resume_from = \"" c "\""}}' "$CONFIG" > "$USE"
    log "RESUME from $(basename "$lc")"
  else
    USE="$CONFIG"; log "FIRST LAUNCH (init per config)"
  fi
  stamp="$(date -u +%Y%m%d_%H%M%S)"
  t0=$(date +%s)
  log "LAUNCH out=$RUNDIR/train.$stamp.out.log"
  "$PY" -u -m hexo_train.cli.train_model "$USE" >"$RUNDIR/train.$stamp.out.log" 2>&1 &
  cpid=$!; echo "$cpid" > "$RUNDIR/driver.pid"
  wait "$cpid"; code=$?; t1=$(date +%s); up=$((t1-t0))
  log "EXIT pid=$cpid code=$code uptime=${up}s"
  if (( code != 0 )); then
    # Surface the crash reason in supervisor.log so diagnosis never requires
    # hunting through per-launch train logs: last error-ish lines of the out log.
    tail -n 40 "$RUNDIR/train.$stamp.out.log" 2>/dev/null \
      | grep -E 'Error|Traceback|raise |CUDA|assert' | tail -n 3 \
      | while IFS= read -r line; do log "CRASH| $line"; done
  fi
  if (( code == 0 )); then echo "exit 0 at $(date -u +%FT%TZ)" > "$DONE"; log "DONE (exit 0)"; break; fi
  crash_times+=("$t1"); now=$(date +%s); kept=(); for ct in "${crash_times[@]}"; do (( now-ct < 3600 )) && kept+=("$ct"); done; crash_times=("${kept[@]}")
  if (( up < FAST_CRASH_SECONDS )); then consec_fast=$((consec_fast+1)); else consec_fast=0; fi
  log "breaker: consecFast=$consec_fast crashesLastHour=${#crash_times[@]}"
  if (( consec_fast >= MAX_CONSEC_FAST || ${#crash_times[@]} > MAX_PER_HOUR )); then
    echo "halt: consecFast=$consec_fast crashesLastHour=${#crash_times[@]}" > "$HALT"
    log "HALT: breaker tripped. Wrote $HALT. Not relaunching."; break
  fi
  log "RELAUNCH (resume from latest) in 3s"; sleep 3
done
log "SUPERVISOR exit."
