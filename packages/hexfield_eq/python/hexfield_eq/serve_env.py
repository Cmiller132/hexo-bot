"""Serve-env profile: force a standalone-eval process env to match the
self-play serve profile so isolated eval runs at self-play throughput.

Serve flags fall in two classes:

  * IMPORT-TIME kernel gates (flex / triton) are read ONCE at
    ``import hexfield.model`` (model.py:62-113) and CANNOT be flipped after
    import. ``prime_serve_env()`` sets them and MUST run BEFORE the first
    ``import hexfield.model`` (standalone eval CLIs call it first).
  * EVALUATOR-TIME serve flags (serve-half / rust-pack / copy-stream, plus the
    f32-feats opt-out) are re-read on EVERY ``HexfieldEvaluator.__init__``
    (inference.py ~348-429), so ``apply_serve_env_profile()`` can force them
    in-code at any point before construction.

This module intentionally imports NOTHING from hexfield (no torch, no model),
so it is safe to import and call before ``hexfield.model`` is imported.
"""

from __future__ import annotations

import os

# Import-time kernel gates (model.py reads each once at import). Forced to "1".
IMPORT_TIME_FLAGS = (
    "HEXFIELD_SERVE_FLEX",
    "HEXFIELD_FLEX_PAIR",
    "HEXFIELD_TRITON_CONV",
    "HEXFIELD_TRITON_ATTN",
    "HEXFIELD_TRITON_CONV_LN",
)

# Evaluator-time serve flags forced ON to match the self-play serve profile.
EVALUATOR_TIME_ON = (
    "HEXFIELD_SERVE_HALF",
    "HEXFIELD_RUST_PACK",
    "HEXFIELD_COPY_STREAM",
)
# Evaluator-time opt-out that DISABLES the fast serve paths (rust-pack /
# serve-half both require the f16 feats path); ensured unset by the profile.
EVALUATOR_TIME_UNSET = ("HEXFIELD_F32_FEATS",)


def apply_serve_env_profile(*, force=False):
    """Force the EVALUATOR-TIME serve flags to the self-play profile.

    Sets ``HEXFIELD_SERVE_HALF`` / ``HEXFIELD_RUST_PACK`` /
    ``HEXFIELD_COPY_STREAM`` to ``"1"`` and ensures ``HEXFIELD_F32_FEATS`` is
    unset. Flags already present in the environment are left untouched (an
    explicit user value wins) unless ``force``, in which case the profile is
    applied unconditionally. These flags are re-read on every
    ``HexfieldEvaluator`` construction, so forcing them in-code is sufficient
    (no pre-import ordering constraint). Returns the dict of flags this call
    changed (value ``"1"`` for a set flag, ``None`` for one that was unset).
    """
    applied: dict[str, str | None] = {}
    for name in EVALUATOR_TIME_ON:
        if force or name not in os.environ:
            os.environ[name] = "1"
            applied[name] = "1"
    for name in EVALUATOR_TIME_UNSET:
        # Only actively unset when forcing; otherwise respect an explicit user
        # value (if it is already unset there is nothing to do).
        if force and name in os.environ:
            del os.environ[name]
            applied[name] = None
    return applied


def prime_serve_env(*, force=False):
    """Set the IMPORT-TIME kernel-gate flags to the self-play serve profile.

    MUST be called BEFORE the first ``import hexfield.model`` — these flags are
    read once at model import and cannot be flipped afterward. Flags already
    present are left untouched unless ``force``. Returns the dict of flags this
    call set.
    """
    applied: dict[str, str] = {}
    for name in IMPORT_TIME_FLAGS:
        if force or name not in os.environ:
            os.environ[name] = "1"
            applied[name] = "1"
    return applied
