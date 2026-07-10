"""Inference device selection for the showcase workers (`SHOWCASE_DEVICE`).

Values: ``auto | cpu | xpu | cuda`` (default ``auto``). ``auto`` prefers xpu,
then cuda, then cpu. An explicit accelerator request that cannot be satisfied
falls back to cpu with a warning note rather than crashing — a mis-provisioned
deployment should serve slow moves, not no moves.

XPU backend notes: torch >= 2.5 installed from the PyTorch xpu wheel index
ships NATIVE Intel-GPU support (``torch.xpu`` is a module attribute, no extra
package). The older Intel stack registered ``torch.xpu`` only after
``import intel_extension_for_pytorch`` (ipex — discontinued upstream once the
support landed in core torch; kept here as a legacy fallback). Resolution
therefore tries native ``torch.xpu`` first and attempts the ipex import only
when the attribute is missing entirely. CPU-only torch builds have no
``torch.xpu`` attribute at all, hence the getattr guards everywhere.

Only the model forward moves to the accelerator: the Rust MCTS session calls
back into the Python evaluator for every batch and is device-agnostic, and
shrimp's fast Triton kernels are ``x.is_cuda``-gated, so XPU tensors take
the eager fp32 paths automatically (correct by construction; speed is what
the deploy benchmark decides). Because eager-on-XPU is an untested-in-CI
combination, workers run a one-position CPU-vs-device parity self-check at
startup (``SHOWCASE_DEVICE_SELFCHECK``, default on whenever the resolved
device is not cpu) and fall back to cpu if it fails — never serve wrong moves.

This module is worker-side: torch/shrimp imports stay inside functions so
the web process can import the showcase package without the model stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

VALID_DEVICES = ("auto", "cpu", "xpu", "cuda")

# Max abs difference tolerated between the CPU and accelerator forwards on the
# self-check position, over the value logits and the policy logits. Generous
# for an fp32-vs-fp32 eager comparison (expected agreement ~1e-5); a genuine
# backend miscompilation is orders of magnitude off.
SELFCHECK_TOL = 1e-2
# Tolerance when the accelerator forward runs under fp16 autocast (the serve
# configuration — see shrimp.inference.serve_autocast). fp16 rounding puts
# honest policy-logit diffs in the 1e-3..1e-2 band; a miscompiled backend is
# still orders of magnitude past this.
SELFCHECK_TOL_AUTOCAST = 5e-2

# Deterministic self-check/warmup position: a legal 5-stone middlegame prefix
# (p0 opening single, then two stones per turn), same script the unit tests
# drive. Non-trivial enough to exercise the trunk, attention and every head.
_SELFCHECK_MOVES = ((0, 0), (0, 2), (1, 2), (1, 0), (2, 0))


@dataclass(frozen=True, slots=True)
class ResolvedDevice:
    """Outcome of `resolve_device`: the torch device string to use, what was
    asked for, and human-readable notes about any fallback taken."""

    device: str
    requested: str
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelfCheckResult:
    ok: bool
    value_diff: float   # max abs diff over the value-head logits
    policy_diff: float  # max abs diff over the policy logits
    error: str | None   # device-forward exception text, when one was raised


def xpu_available() -> bool:
    """True when a usable torch XPU device exists.

    Tries native ``torch.xpu`` first (xpu-index torch wheels); only when the
    attribute is missing entirely does it attempt the legacy ipex import,
    which registers the backend as a side effect on old Intel stacks. A torch
    that HAS ``torch.xpu`` but reports unavailable (e.g. missing level-zero
    runtime, no /dev/dri) is simply unavailable — ipex cannot help there.
    """
    import torch

    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        try:
            import intel_extension_for_pytorch  # noqa: F401  (legacy backend registration)
        except Exception:
            return False
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            return False
    try:
        return bool(xpu.is_available())
    except Exception:
        return False


def resolve_device(requested: str) -> ResolvedDevice:
    """Map a SHOWCASE_DEVICE value to the device this process will use.

    ``auto`` -> xpu if available, else cuda if available, else cpu. Explicit
    ``xpu``/``cuda`` requests degrade to cpu with a note when unavailable
    (the caller logs notes as warnings). Unknown values raise ValueError —
    that is a configuration typo, not a runtime condition to paper over.
    """
    req = (requested or "auto").strip().lower()
    if req not in VALID_DEVICES:
        raise ValueError(
            f"SHOWCASE_DEVICE must be one of {'|'.join(VALID_DEVICES)}, got {requested!r}"
        )
    import torch

    if req == "cpu":
        return ResolvedDevice("cpu", req, ())
    if req in ("auto", "xpu"):
        if xpu_available():
            return ResolvedDevice("xpu", req, ())
        if req == "xpu":
            return ResolvedDevice(
                "cpu", req,
                (
                    "SHOWCASE_DEVICE=xpu but no usable XPU (torch has no xpu "
                    "backend, or no Intel GPU / level-zero runtime is visible); "
                    "falling back to cpu",
                ),
            )
    if torch.cuda.is_available():
        return ResolvedDevice("cuda", req, ())
    if req == "cuda":
        return ResolvedDevice(
            "cpu", req,
            ("SHOWCASE_DEVICE=cuda but CUDA is not available; falling back to cpu",),
        )
    return ResolvedDevice("cpu", req, ())


def selfcheck_wanted(setting: bool | None, device: str) -> bool:
    """SHOWCASE_DEVICE_SELFCHECK gate: explicit setting wins; the default
    (unset -> None) is on exactly when the resolved device is not cpu."""
    if setting is not None:
        return bool(setting)
    return device != "cpu"


def _selfcheck_batch() -> dict[str, Any]:
    """Featurized fixed synthetic position -> one collated model batch (CPU)."""
    import hexo_engine as engine
    from hexo_engine.types import AxialCoord, PlacementAction

    from shrimp.batching import collate_rows

    from .analysis import featurize

    state = engine.new_game()
    for q, r in _SELFCHECK_MOVES:
        engine.apply_action(state, PlacementAction(AxialCoord(q, r)))
    return collate_rows([featurize(state)])


def _forward(
    model: Any, batch: dict[str, Any], device: Any, *, autocast_on: bool = False,
) -> dict[str, Any]:
    """One SERVE-path forward of `batch` on `device`; outputs come back as
    detached fp32 CPU tensors.

    This deliberately mirrors the evaluator's hot path — no-grad
    `forward_policy_value` under (optional) fp16 autocast — rather than the
    grad-enabled full-head forward: `build_attn_bias` branches on
    `torch.is_grad_enabled()`, and only the no-grad branch runs the fp16
    gather + chunked deep-board XPU workaround that live moves execute. A
    parity check on the grad branch would vouch for kernels serving never
    touches."""
    import torch

    feats = batch["feats"].to(device)
    nbr = batch["nbr"].to(device)
    mask = batch["mask"].to(device)
    coords = batch["coords"].to(device)
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    with torch.no_grad(), torch.autocast(
        device_type=dev.type, dtype=torch.float16, enabled=autocast_on
    ):
        out = model.forward_policy_value(
            feats, nbr, mask, coords, request_moves_left=True
        )
    return {k: v.detach().float().cpu() for k, v in out.items()}


def verify_device(model: Any, device: str, *, tol: float | None = None) -> SelfCheckResult:
    """Startup parity self-check: the same fixed position forwarded on CPU
    (fp32 reference) and on `device` UNDER THE SERVE CONFIGURATION (no-grad
    forward_policy_value, fp16 autocast when the evaluator would use it) must
    agree within `tol` (max abs diff) on the value logits and the policy
    logits, and both must be finite. `tol` defaults per configuration:
    SELFCHECK_TOL for an fp32 device forward, SELFCHECK_TOL_AUTOCAST when the
    serve path autocasts.

    On success the model is LEFT ON `device`; on any failure (mismatch,
    non-finite output, or the device forward raising) the model is moved back
    to CPU before returning, so the caller can construct a cpu evaluator
    directly. `device == "cpu"` degenerates to a trivial cpu-vs-cpu compare.
    """
    import math

    import torch

    from shrimp.inference import serve_autocast

    autocast_on = serve_autocast(device)
    if tol is None:
        tol = SELFCHECK_TOL_AUTOCAST if autocast_on else SELFCHECK_TOL
    batch = _selfcheck_batch()
    ref = _forward(model.to("cpu"), batch, torch.device("cpu"))
    dev = torch.device(device)
    try:
        out = _forward(model.to(dev), batch, dev, autocast_on=autocast_on)
    except Exception as exc:  # broken backend/runtime: fall back, don't crash
        model.to("cpu")
        return SelfCheckResult(False, math.inf, math.inf, f"{type(exc).__name__}: {exc}")
    value_diff = float((out["value"] - ref["value"]).abs().max().item())
    policy_diff = float((out["policy"] - ref["policy"]).abs().max().item())
    finite = all(
        bool(torch.isfinite(t).all()) for t in (out["value"], out["policy"])
    )
    ok = finite and math.isfinite(value_diff) and math.isfinite(policy_diff) \
        and value_diff < tol and policy_diff < tol
    if not ok:
        model.to("cpu")
    return SelfCheckResult(ok, value_diff, policy_diff, None)


def _synthetic_batch(batch_size: int, npad: int) -> dict[str, Any]:
    """A synthetic collated batch of shape (batch_size, npad) for kernel
    warmup: zero feats/coords, all-pad neighbor pointers, a plausible live
    mask. Numerically meaningless but structurally identical to a real serve
    batch, which is all a shape-keyed kernel JIT cares about."""
    import torch

    from shrimp.constants import NUM_FEATURES

    n_live = max(1, npad - 1)
    mask = torch.zeros(batch_size, npad, dtype=torch.bool)
    mask[:, :n_live] = True
    return {
        "feats": torch.zeros(batch_size, npad, NUM_FEATURES),
        "nbr": torch.full((batch_size, npad, 6), npad, dtype=torch.int64),
        "mask": mask,
        "coords": torch.zeros(batch_size, npad, 2, dtype=torch.int64),
    }


def warmup(model: Any, device: str, iters: int = 2) -> None:
    """Trigger the accelerator backend's lazy init / kernel JIT with dummy
    SERVE-path forwards so the first real moves aren't pathologically slow.

    torch.xpu JIT-compiles SYCL kernels per (batch, Npad) SHAPE and per code
    path, so warming one tiny position leaves every other serve shape cold —
    a deep-board first move then pays a multi-second (observed tens of
    seconds) JIT inside a live search, indistinguishable from a wedge to the
    pool's hang detector. Warm the shapes that bracket the serve regimes: a
    small opening-sized batch, and one past the deep-board chunked-bias
    threshold so the chunked-gather kernels (the XPU workaround path) JIT
    here instead of mid-game. No-op on cpu. The model must already be
    resident on `device`."""
    import torch

    from shrimp.inference import serve_autocast
    from shrimp.model import _BIAS_GATHER_CHUNK_THRESHOLD

    dev = torch.device(device)
    if dev.type == "cpu":
        return
    autocast_on = serve_autocast(device)
    shapes = (
        (1, 128),   # opening/shallow flush (batch-1 group, small support)
        (2, 256),   # ordinary mid-game group
        (2, _BIAS_GATHER_CHUNK_THRESHOLD + 128),  # deep board, chunked gather
    )
    for batch_size, npad in shapes:
        batch = _synthetic_batch(batch_size, npad)
        for _ in range(max(1, iters)):
            _forward(model, batch, dev, autocast_on=autocast_on)
    backend = getattr(torch, dev.type, None)
    sync = getattr(backend, "synchronize", None)
    if callable(sync):
        try:
            sync()
        except Exception:
            pass
