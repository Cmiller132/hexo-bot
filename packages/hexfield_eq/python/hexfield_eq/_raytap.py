"""Ray-tap conv support: generated tap geometry LUTs + the masked, per-orbit-
channel distance-weighted pre-aggregation (SPEC_RAYTAP_CONV.md §2.2/§2.5).

For a conv in ray-tap mode the direction-tap inputs are redefined as

    in_d(i)[c] = sum_{k=1..5} alpha[k, c] * 1[k <= raylen_{s(c)}(i, d)] * x_{i,d,k}[c]

with x_{i,d,k} the trunk features of the cell at offset ``k * DIRECTIONS[d]``
(zero when absent from the support), ``s(c)`` the visibility side of orbit
channel ``c`` (own = first orbit half, opp = second — the same orbit-index
split as the L-block sub-heads, spec §2.6), and ``alpha`` shared across the 6
directions and tiled slot-constant over the fiber.

Geometry is GENERATED, never hand-coded (spec §2.5): conv taps are indexed in
``constants.DIRECTIONS`` order; raylen slots are ``side*6 + axis*2 + dir`` with
axis in [Q, R, QR] (``features.AXIS_DELTAS``) and dir in {+ = 0, - = 1}; the
ray-gather slots follow ``_triton_ray``'s packing ``1 + axis*10 + dir*5 +
(k-1)``. :func:`tap_axis_dir` derives the tap -> (axis, dir) bijection from the
two tables and T7 asserts it against ``slot_offset_table`` geometrically.

The reference aggregation (:func:`ray_tap_taps_naive`) defines numerics for all
tests and is the serve fallback; the K2 memory path rides the same math. All
arithmetic runs in ``x.dtype`` via mul/sum (never einsum/matmul, which autocast
would down-cast): fp32 on the train stream, fp16 on half serve. At init
``alpha[:, c] = (1, 0, 0, 0, 0)`` the k=1 term is an exact multiply-by-1.0 of
the distance-1 neighbour and every other term is an exact 0.0, so the ray-tap
conv reproduces the baseline 7-tap conv bit-for-bit (T4 init-equivalence;
relies on the terminal-blocker raylen convention: raylen >= 1 whenever the
distance-1 cell is on the support).
"""

from __future__ import annotations

import functools

import torch

from .constants import DIRECTIONS, RAY_REACH, RAYLEN_SLOTS
from .features import AXIS_DELTAS
from ._triton_ray import RAY_GATHER_SLOTS, build_ray_gather_index  # noqa: F401

# Canonical axis order [Q, R, QR] — matches features._AXES, the raylen slot
# layout, and _triton_ray._AXIS_VECS.
_AXIS_ORDER = ("Q", "R", "QR")


@functools.lru_cache(maxsize=1)
def tap_axis_dir() -> tuple[tuple[int, int], ...]:
    """(axis, dir) of each direction tap t = 1..6, generated from
    ``constants.DIRECTIONS`` and the axis delta table (``features.AXIS_DELTAS``):
    tap vector == +axis delta -> dir 0, == -axis delta -> dir 1. Every hex
    direction lies on exactly one win axis (T7 asserts the bijection)."""

    axes = tuple(AXIS_DELTAS[name] for name in _AXIS_ORDER)
    out: list[tuple[int, int]] = []
    for dq, dr in DIRECTIONS:
        hit: tuple[int, int] | None = None
        for ai, (aq, ar) in enumerate(axes):
            if (dq, dr) == (aq, ar):
                hit = (ai, 0)
            elif (dq, dr) == (-aq, -ar):
                hit = (ai, 1)
        if hit is None:  # pragma: no cover - geometric invariant
            raise AssertionError(f"direction {(dq, dr)} lies on no win axis")
        out.append(hit)
    return tuple(out)


@functools.lru_cache(maxsize=1)
def tap_raylen_slots() -> torch.Tensor:
    """(2, 6) long CPU tensor: the raylen wire slot consulted by (side, tap-1):
    ``side*6 + axis*2 + dir`` (features.ray_lengths_for_cell layout)."""

    ad = tap_axis_dir()
    out = torch.empty(2, 6, dtype=torch.long)
    for side in range(2):
        for t, (axis, direc) in enumerate(ad):
            out[side, t] = side * 6 + axis * 2 + direc
    assert out.max() < RAYLEN_SLOTS
    return out


@functools.lru_cache(maxsize=1)
def tap_ray_slot_lut() -> torch.Tensor:
    """(6, 5) long CPU tensor: the ray-gather slot (``_triton_ray`` packing)
    holding the cell at offset ``k * DIRECTIONS[t]`` for (tap-1, k-1):
    ``1 + axis*10 + dir*5 + (k-1)``. T7 checks each slot's offset against
    ``slot_offset_table`` — the geometric ground truth, not this arithmetic."""

    ad = tap_axis_dir()
    out = torch.empty(6, RAY_REACH, dtype=torch.long)
    for t, (axis, direc) in enumerate(ad):
        for k in range(1, RAY_REACH + 1):
            out[t, k - 1] = 1 + axis * 10 + direc * 5 + (k - 1)
    assert out.max() < RAY_GATHER_SLOTS
    return out


def build_tap_reach(raylen: torch.Tensor) -> torch.Tensor:
    """(B, N, 2, 6) uint8 per-(side, tap) visibility reach, gathered from the
    (B, N, RAYLEN_SLOTS) u8 raylen wire buffer by the generated slot LUT.
    ``vis(i, d, k, s) = k <= reach[i, s, d]``; pad rows carry raylen 0 (the
    D-S13 pad convention), so every pad-tap contribution masks to zero."""

    slots = tap_raylen_slots().to(raylen.device)
    return raylen[:, :, slots]  # advanced index -> (B, N, 2, 6)


def _tap_flat_index(idx_taps: torch.Tensor, d: int, c: int) -> torch.Tensor:
    """The (B, N*5, C) gather/scatter index for direction tap ``d`` (an
    expanded view; no materialization)."""

    b, n = idx_taps.shape[0], idx_taps.shape[1]
    return idx_taps[:, :, d, :].reshape(b, n * RAY_REACH, 1).expand(-1, -1, c)


def _tap_vis(reach: torch.Tensor, d: int) -> torch.Tensor:
    """(B, N, 5, 2 sides) bool visibility for direction tap ``d``:
    ``k <= reach[i, s, d]``. ``reach`` must be an integer dtype comparable to
    arange (the callers pass ``.to(torch.long)`` once)."""

    k_vec = torch.arange(1, RAY_REACH + 1, device=reach.device)
    return k_vec.view(1, 1, RAY_REACH, 1) <= reach[:, :, None, :, d]


def _masked_gather(
    x_ext: torch.Tensor,
    idx_taps: torch.Tensor,
    reach: torch.Tensor,
    corb: int,
    d: int,
) -> torch.Tensor:
    """(B, N, 5, C) gathered ray rows for direction tap ``d`` with the
    per-side visibility mask applied (zero where k > reach or the cell is
    absent from the support) — the shared intermediate the forward sum and
    the alpha/x gradients both consume.

    x_ext (B, N+1, C) with the zero row appended at index N; idx_taps
    (B, N, 6, 5) long; reach (B, N, 2, 6) long; corb = the orbit width whose
    halves the own/opp visibility split rides. All ops are mul/sum in
    ``x.dtype`` (autocast-neutral)."""

    b = x_ext.shape[0]
    n = x_ext.shape[1] - 1
    c = x_ext.shape[2]
    half = corb // 2
    g = c // corb
    xg = x_ext.gather(1, _tap_flat_index(idx_taps, d, c)).reshape(
        b, n, RAY_REACH, c
    )
    vis = _tap_vis(reach, d)
    return (
        xg.view(b, n, RAY_REACH, g, 2, half)
        * vis.view(b, n, RAY_REACH, 1, 2, 1).to(x_ext.dtype)
    ).reshape(b, n, RAY_REACH, c)


class _RayTapTaps(torch.autograd.Function):
    """K2 — memory-bounded pre-aggregation (spec §2.4, blocking for training
    at conv2/both).

    A naive implementation saves the gathered (B, N, 30, C) intermediate for
    backward (~717 MB fp32 per equipped conv at B=48, S=648 — ~+7.2 GB at
    `both`, an OOM / forced batch cut on the 12 GB training card). This
    Function saves only ``x`` (alive anyway as the conv input), the tap
    gather index, the per-(side, tap) reach (u8, the visibility masks'
    compressed form), and ``alpha_full``, recomputing the gather per
    direction in backward:

        grad_alpha[k, c] = sum_{b,i,d} mask * x_gathered * grad_out
        grad_x = scatter-add of alpha * mask * grad_out into source rows

    Numerics are identical to :func:`ray_tap_taps_naive` (same
    ``_masked_gather`` + mul/sum ops in the same order); T8's small-shape
    oracle pins the gradients to <= 1e-5 rel."""

    @staticmethod
    def forward(ctx, x, idx_taps, reach, alpha_full, corb):
        b, n, c = x.shape
        reach_l = reach.to(torch.long)
        x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)
        taps = torch.empty(b, n, 6, c, dtype=x.dtype, device=x.device)
        a = alpha_full.view(1, 1, RAY_REACH, c)
        for d in range(6):
            xgv = _masked_gather(x_ext, idx_taps, reach_l, corb, d)
            taps[:, :, d] = (xgv * a).sum(dim=2)
        ctx.save_for_backward(x, idx_taps, reach, alpha_full)
        ctx.corb = corb
        return taps

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        x, idx_taps, reach, alpha_full = ctx.saved_tensors
        corb = ctx.corb
        b, n, c = x.shape
        half = corb // 2
        g = c // corb
        reach_l = reach.to(torch.long)
        x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)
        need_x, _, _, need_alpha, _ = ctx.needs_input_grad
        grad_x_ext = torch.zeros_like(x_ext) if need_x else None
        grad_alpha = torch.zeros_like(alpha_full) if need_alpha else None
        a = alpha_full.view(1, 1, RAY_REACH, c)
        for d in range(6):
            go_d = grad_out[:, :, d]  # (B, N, C)
            if need_alpha:
                xgv = _masked_gather(x_ext, idx_taps, reach_l, corb, d)
                grad_alpha += (xgv * go_d.unsqueeze(2)).sum(dim=(0, 1))
                del xgv
            if need_x:
                vis = _tap_vis(reach_l, d)
                gxg = (
                    (a * go_d.unsqueeze(2)).view(b, n, RAY_REACH, g, 2, half)
                    * vis.view(b, n, RAY_REACH, 1, 2, 1).to(x.dtype)
                ).reshape(b, n * RAY_REACH, c)
                grad_x_ext.scatter_add_(1, _tap_flat_index(idx_taps, d, c), gxg)
        grad_x = grad_x_ext[:, :n] if need_x else None
        return grad_x, None, None, grad_alpha, None


def ray_tap_taps(
    x: torch.Tensor,
    idx_taps: torch.Tensor,
    reach: torch.Tensor,
    alpha_full: torch.Tensor,
    corb: int,
) -> torch.Tensor:
    """(B, N, 6, C) direction-tap inputs — the production entry point every
    equipped conv calls: the K2 custom-autograd pre-aggregation (identical
    numerics to :func:`ray_tap_taps_naive`, memory-bounded backward)."""

    return _RayTapTaps.apply(x, idx_taps, reach, alpha_full, corb)


def ray_tap_taps_naive(
    x: torch.Tensor,
    idx_taps: torch.Tensor,
    reach: torch.Tensor,
    alpha_full: torch.Tensor,
    corb: int,
) -> torch.Tensor:
    """(B, N, 6, C) direction-tap inputs, plain autograd (the numerics
    reference and the T8 gradient oracle). Saves the gathered intermediates
    for backward — the memory profile K2 exists to avoid; production uses
    :func:`ray_tap_taps` (the ``_RayTapTaps`` Function) instead."""

    b, n, c = x.shape
    x_ext = torch.cat([x, x.new_zeros(b, 1, c)], dim=1)
    reach_l = reach.to(torch.long)
    a = alpha_full.view(1, 1, RAY_REACH, c)
    taps = [
        (_masked_gather(x_ext, idx_taps, reach_l, corb, d) * a).sum(dim=2)
        for d in range(6)
    ]
    return torch.stack(taps, dim=2)
