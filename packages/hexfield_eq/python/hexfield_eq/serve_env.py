"""Serve-env profile: force a standalone-eval process env to match the
self-play serve profile so isolated eval runs at self-play throughput.

Serve flags fall in two classes:

  * IMPORT-TIME kernel gates (flex / triton) are read ONCE at
    ``import hexfield_eq.model`` (model.py) and CANNOT be flipped after
    import. ``prime_serve_env()`` sets them and MUST run BEFORE the first
    ``import hexfield_eq.model`` (standalone eval CLIs call it first).
  * EVALUATOR-TIME serve flags (serve-half / rust-pack / copy-stream, plus the
    f32-feats opt-out) are re-read on EVERY ``HexfieldEvaluator.__init__``
    (inference.py ~348-429), so ``apply_serve_env_profile()`` can force them
    in-code at any point before construction.

This module intentionally imports NOTHING from hexfield (no torch, no model),
so it is safe to import and call before ``hexfield_eq.model`` is imported.
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
XPU_FLEX_FLAGS = (
    "HEXFIELD_SERVE_FLEX",
    "HEXFIELD_FLEX_PAIR",
)

# Evaluator-time serve flags forced ON to match the self-play serve profile.
EVALUATOR_TIME_ON = (
    "HEXFIELD_SERVE_HALF",
    "HEXFIELD_RUST_PACK",
    "HEXFIELD_COPY_STREAM",
)
# Evaluator-time opt-out that disables CUDA's f16 rust-pack/serve-half paths.
# XPU's Rust pack widens wire-f16 to fp32 and does not read this opt-out.
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

    MUST be called BEFORE the first ``import hexfield_eq.model`` — these flags are
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


def prime_serve_env_for_device(device: str, *, force=False):
    """Prime only import-time kernels that are candidates for ``device``.

    CUDA receives the historical full self-play profile. The custom Triton
    conv/attention kernels are explicitly CUDA-only in ``model.py`` and are
    therefore never enabled for XPU here. PyTorch FlexAttention has an XPU
    backend in recent torch builds, but backend/version/card performance and
    parity must be measured; it remains an explicit ``HEXFIELD_XPU_FLEX=1``
    experiment rather than a deploy default.

    CPU and an XPU without that opt-in change no import-time flags. Explicit
    per-kernel environment values still win unless ``force``.
    """
    kind = str(device).split(":", 1)[0].lower()
    if kind == "cuda":
        return prime_serve_env(force=force)
    if kind != "xpu" or os.environ.get("HEXFIELD_XPU_FLEX") != "1":
        return {}
    applied: dict[str, str] = {}
    for name in XPU_FLEX_FLAGS:
        if force or name not in os.environ:
            os.environ[name] = "1"
            applied[name] = "1"
    return applied
