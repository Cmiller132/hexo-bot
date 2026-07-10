"""Serve-time coset-perm fold gate (hexfield_eq).

The D6-equivariant net applies a fixed coset channel permutation (docs/DERIVATION
§4) around every attention: ``q/k/v[..., head_perm]`` after the projections and
the inverse perm on the output before out_proj, in RelPosAttention, RayAttention,
and RegisterRefresh. On the FROZEN serve path those runtime gathers are pure
overhead — EquivLinear already regenerates+caches the dense weight once per
checkpoint, so the perm can be folded straight into that cached weight and the
runtime gathers skipped. This suite pins the fold's correctness:

  1. EquivLinear fold identities in isolation — output fold ``W[out_perm]`` is
     bit-identical to the grad-path forward + a runtime output perm; input fold
     ``W[:, in_perm]`` matches the runtime input perm + forward (tight fp32).
  2. MODULE parity — RelPosAttention / RayAttention / RegisterRefresh, same
     weights + inputs: the no-grad (folded) forward == the grad-enabled (runtime
     perm) forward.
  3. FULL-NET parity — a CLA-layout lane-carrying HexfieldNet, no-grad vs
     grad-enabled on a random board.
  4. CACHE correctness — two no-grad forwards hit the fold cache identically; an
     in-place weight bump (``_version``) invalidates it.

The fold is gated on ``self.equivariant and not torch.is_grad_enabled()`` — the
SAME condition EquivLinear._materialize uses to bake the perm in — so serve and
train can never disagree. Runs under the equivariant default build
(GROUP_ORDER == 12); self-skips otherwise. Do NOT set HEXFIELD_EQ_* here (this is
the C=96 default arch). CPU-only, fp32.
"""

from __future__ import annotations

import pytest
import torch

from hexfield_eq import constants as C
from hexfield_eq.constants import DIRECTIONS
from hexfield_eq.geometry import apply_d6, disk_offsets
from hexfield_eq.model import EquivLinear, HexfieldNet, RayAttention, RelPosAttention
from hexfield_eq.register import RegisterRefresh

pytestmark = pytest.mark.skipif(
    C.GROUP_ORDER != 12,
    reason="serve-fold gate; run under the default HEXFIELD_EQ_GROUP_ORDER=12 build",
)

# fp32 module/full-net parity tolerance. The q/k/v output fold is a row
# permutation (exact, bit-identical); only out_proj's INPUT fold (W[:, hp] vs the
# runtime out[..., hp_inv]) reorders the GEMM accumulation, so the divergence is
# pure fp32 round-off on that one matmul — measured <= 1.5e-5 through the trunk
# with N(0, 0.3) params, far under the 3e-3 serve parity gate. 1e-4 matches the
# equivariance suites' fp32 structural tolerance.
ATOL = 1e-4
# The round-off scales with the accumulated ROW magnitudes (the GEMM terms), not
# with each element's own value — so elementwise rtol is the wrong model (a
# reorder diff can land on a near-zero element). The N(0, 0.3) fat weights
# inflate the scalar-head outputs to O(1e3-1e4) at C=192 (the arm-4 gate env),
# where a pure atol is width-dependent (measured 1.1e-3 abs = ~2e-7 of the head's
# scale). The full-net check therefore scales its atol by each head's max
# magnitude (floor ATOL); modules keep pure ATOL (outputs O(10) at every width).
NET_SCALE_TOL = 2e-6


def _randomize(module: torch.nn.Module, seed: int, scale: float = 0.3) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * scale)


def _inv_perm(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    return inv


def _disk_board(radius: int = 3):
    """A G-closed hex disk with row-local neighbour gather + axial coords + a
    live mask (the Phase-3b/L1 probe board, cell-sig unused here)."""

    cells = disk_offsets(radius)
    n = len(cells)
    cidx = {c: i for i, c in enumerate(cells)}
    nbr = torch.full((1, n, 6), n, dtype=torch.long)
    for i, c in enumerate(cells):
        for d, off in enumerate(DIRECTIONS):
            nb = (c[0] + off[0], c[1] + off[1])
            if nb in cidx:
                nbr[0, i, d] = cidx[nb]
    coords = torch.tensor([[list(c) for c in cells]], dtype=torch.long)
    mask = torch.ones(1, n, dtype=torch.bool)
    return n, nbr, coords, mask


# --- 1. EquivLinear fold identities in isolation --------------------------------


def test_equivlinear_output_fold_is_bit_identical() -> None:
    """``(W x + b)[out_perm] == W[out_perm] x + b[out_perm]``: a row permutation
    does not touch any dot product, so the folded no-grad forward equals the
    grad-path (unfolded) forward with the output perm applied, BITWISE."""

    torch.manual_seed(0)
    lin = EquivLinear(C.CHANNELS, C.CHANNELS)
    _randomize(lin, 0)
    perm = torch.randperm(C.CHANNELS)
    lin.set_serve_perms(out_perm=perm)
    x = torch.randn(4, 7, C.CHANNELS)

    with torch.enable_grad():
        base = lin(x)  # grad -> unfolded W x + b (EquivLinear never perms itself)
    with torch.no_grad():
        folded = lin(x)  # folds W[perm] / b[perm]
    assert torch.equal(folded, base[..., perm])


def test_equivlinear_input_fold_matches_runtime_perm() -> None:
    """out_proj's input fold: the runtime ``out_proj(out[..., hp_inv])`` equals
    the folded ``W[:, hp] out + b`` (in_perm = head_perm itself, NOT its inverse).
    Same products, accumulation reordered -> allclose at a tight fp32 atol."""

    torch.manual_seed(1)
    lin = EquivLinear(C.CHANNELS, C.CHANNELS)
    _randomize(lin, 1)
    perm = torch.randperm(C.CHANNELS)  # the head_perm (in_perm)
    inv = _inv_perm(perm)  # the runtime pre-perm the owner applies (hp_inv)
    lin.set_serve_perms(in_perm=perm)
    x = torch.randn(4, 7, C.CHANNELS)

    with torch.enable_grad():
        runtime = lin(x[..., inv])  # grad -> W (x[hp_inv]) + b
    with torch.no_grad():
        folded = lin(x)  # W[:, hp] x + b
    # Same products, reordered accumulation over the C_in fold -> tight fp32
    # round-off (measured ~4e-6 at C=96), not bit-identity.
    torch.testing.assert_close(folded, runtime, atol=1e-5, rtol=0)


# --- 2. module-level parity -----------------------------------------------------


def test_relpos_attention_fold_parity() -> None:
    torch.manual_seed(2)
    attn = RelPosAttention(C.CHANNELS).eval()
    _randomize(attn, 2)
    s = 4 + 11  # tokens + a few cells; the module is agnostic to the split
    seq = torch.randn(1, s, C.CHANNELS)
    bias = torch.randn(1, attn.heads, s, s)
    with torch.no_grad():
        folded = attn(seq, bias)
    with torch.enable_grad():
        runtime = attn(seq, bias)
    torch.testing.assert_close(folded, runtime, atol=ATOL, rtol=0)


def test_ray_attention_fold_parity() -> None:
    torch.manual_seed(3)
    attn = RayAttention(C.CHANNELS).eval()
    _randomize(attn, 3)
    n = 13
    x = torch.randn(1, n, C.CHANNELS)
    bias = torch.randn(1, attn.heads, n, n)
    with torch.no_grad():
        folded = attn(x, bias)
    with torch.enable_grad():
        runtime = attn(x, bias)
    torch.testing.assert_close(folded, runtime, atol=ATOL, rtol=0)


def test_register_refresh_fold_parity() -> None:
    torch.manual_seed(4)
    reg = RegisterRefresh(C.CHANNELS).eval()
    _randomize(reg, 4)
    n = 13
    tokens = torch.randn(1, C.NUM_TOKENS, C.CHANNELS)
    cells = torch.randn(1, n, C.CHANNELS)
    mask = torch.ones(1, n, dtype=torch.bool)
    with torch.no_grad():
        folded = reg(tokens, cells, mask)
    with torch.enable_grad():
        runtime = reg(tokens, cells, mask)
    torch.testing.assert_close(folded, runtime, atol=ATOL, rtol=0)


# --- 3. full-net parity ---------------------------------------------------------


def test_full_net_fold_parity() -> None:
    """A CLA lane-carrying net exercises all three perm-folding modules
    (RelPosAttention in A, RayAttention in L, RegisterRefresh in the lane): the
    no-grad (folded) forward matches the grad-enabled (runtime perm) forward."""

    n, nbr, coords, mask = _disk_board(3)
    model = HexfieldNet(trunk_layout="CLA", reg_lane=True, reg_tok_read=True).eval()
    _randomize(model, 5)
    torch.manual_seed(15)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    raylen = torch.randint(0, C.RAY_REACH + 1, (1, n, C.RAYLEN_SLOTS), dtype=torch.uint8)

    with torch.no_grad():
        folded = model(feats, nbr, mask, coords, raylen=raylen)
    with torch.enable_grad():
        runtime = model(feats, nbr, mask, coords, raylen=raylen)
    for key in folded:
        tol = max(ATOL, NET_SCALE_TOL * float(runtime[key].abs().max()))
        torch.testing.assert_close(
            folded[key], runtime[key], atol=tol, rtol=0, msg=key
        )


# --- 4. fold-cache correctness --------------------------------------------------


def test_fold_cache_hit_and_version_invalidation() -> None:
    """Two no-grad forwards return the identical cached folded weight; an
    in-place weight bump (``wb._version``) invalidates the fold cache and the
    output tracks the new weight."""

    torch.manual_seed(6)
    lin = EquivLinear(C.CHANNELS, C.CHANNELS)
    _randomize(lin, 6)
    lin.set_serve_perms(out_perm=torch.randperm(C.CHANNELS))
    x = torch.randn(4, C.CHANNELS)

    with torch.no_grad():
        y1 = lin(x)
        y2 = lin(x)  # cache hit
        assert torch.equal(y1, y2)
        assert lin._cache_w is not None
        # A no-op data touch (param.data.add_(0)) would NOT bump _version; a real
        # in-place add does, invalidating the (wb._version, bias_base._version)
        # cache key so the next forward regenerates+refolds.
        lin.wb.add_(1e-3)
        y3 = lin(x)
    assert not torch.allclose(y3, y1)
