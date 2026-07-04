#!/usr/bin/env bash
# Behavior-cloning prefit pipeline (optional warm start for training).
# Three stages:
#   1. fetch_corpus.py         -> downloads a human game corpus (Hugging Face)
#   2. bootstrap_from_corpus.py-> replays it through hexo_engine into
#                                 hexfield_compact_v1 shards (train/ + val/)
#   3. scripts/prefit.py       -> trains a BC checkpoint at the main_7 arch
#
# The architecture env is LOAD-BEARING (the prefit checkpoint must be built at
# the same CHANNELS/HEADS/TRUNK as the run that will initialize_from it).
#
# Env overrides:
#   HEXO_VENV   (default .venv at repo root)
#   CORPUS_DIR  (default data/hexo-bootstrap-corpus)
#   DATA_DIR    (default data/prefit)          — the shard dataset
#   OUT_DIR     (default runs/hexfield_main_7_prefit)
#   PREFIT_EPOCHS (default 4)
#   SKIP_FETCH=1 to reuse an already-downloaded corpus
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HEXO_VENV="${HEXO_VENV:-$ROOT/.venv}"
PY="$HEXO_VENV/bin/python"

CORPUS_DIR="${CORPUS_DIR:-$ROOT/data/hexo-bootstrap-corpus}"
DATA_DIR="${DATA_DIR:-$ROOT/data/prefit}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/hexfield_main_7_prefit}"
PREFIT_EPOCHS="${PREFIT_EPOCHS:-4}"

# Architecture (must match the config the checkpoint will warm-start).
export HEXFIELD_SUPPORT_RADIUS="${HEXFIELD_SUPPORT_RADIUS:-4}"
export HEXFIELD_CHANNELS="${HEXFIELD_CHANNELS:-192}"
export HEXFIELD_ATTENTION_HEADS="${HEXFIELD_ATTENTION_HEADS:-3}"
export HEXFIELD_TRUNK="${HEXFIELD_TRUNK:-CCACCACCACCACCA}"
# Prefit trains eagerly; the serve/train perf kernels are intentionally not set.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT/packages/hexfield/python${PYTHONPATH:+:$PYTHONPATH}"

# 1. Fetch corpus.
if [ "${SKIP_FETCH:-0}" != "1" ]; then
  "$PY" -c "import huggingface_hub" 2>/dev/null || {
    echo "huggingface_hub is required to fetch the corpus: pip install huggingface_hub" >&2
    exit 1
  }
  echo "=== fetch corpus -> $CORPUS_DIR ==="
  "$PY" "$ROOT/scripts/fetch_corpus.py" --out "$CORPUS_DIR"
fi

# 2. Bootstrap the corpus into training shards.
echo "=== bootstrap corpus -> $DATA_DIR ==="
"$PY" "$ROOT/scripts/bootstrap_from_corpus.py" \
  --corpus "$CORPUS_DIR/hexo_human_corpus.jsonl" \
  --out "$DATA_DIR"

# 3. Run the BC prefit.
test -d "$DATA_DIR/train" || { echo "no prefit shards under $DATA_DIR — stage 2 failed" >&2; exit 1; }
mkdir -p "$OUT_DIR"
DEVICE="${DEVICE:-$("$PY" -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')}"
echo "=== prefit ($PREFIT_EPOCHS epochs, device=$DEVICE) -> $OUT_DIR ==="
"$PY" "$ROOT/scripts/prefit.py" \
  --data "$DATA_DIR" --out "$OUT_DIR" --epochs "$PREFIT_EPOCHS" \
  --workers "${PREFIT_WORKERS:-6}" --device "$DEVICE"
echo "=== prefit done. Point checkpoint.initialize_from at a checkpoint in $OUT_DIR ==="
