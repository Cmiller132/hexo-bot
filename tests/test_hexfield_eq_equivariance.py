"""Phase-3b equivariant-trunk gate (hexfield_eq).

The sole correctness guarantee behind the augmentation-free expand path: the
full network is EXACTLY D6-equivariant. For every g in D6,

    f(g . board).policy == g . f(board).policy      (per-cell covariance)
    f(g . board).value  ==     f(board).value        (invariance)

to fp32 tolerance, where the input transforms by the typed permutation rep the
stem lifts (13 scalar planes fixed, 12 axis planes permuted by the coset/axis
action; docs/DERIVATION_D6_EQUIVARIANT_ATTENTION.md §8). This exercises the tied
HexNodeConv, group-norm + orbit-tied LayerScale, the tied Q/K/V/out + MLP, the
coset head split, the jointly (row, head)-tied relative bias, the invariant
summary tokens, and the group-pooled heads — the whole Phase-3b contract.

Also asserts: reference (sdpa) == materialized attention parity and, on CUDA,
the tied-generated weight is consumed identically by the fused Triton conv
kernel; grads reach every tied base param; and a smoke train step is NaN-free.

Runs under the equivariant default build (GROUP_ORDER == 12). The passthrough
gates (test_hexfield_eq_smoke / _orbit_bias) run under HEXFIELD_EQ_GROUP_ORDER=1
and self-skip here, so a single ``-k hexfield_eq`` invocation stays green either
way. CPU-only for the equivariance proof; the Triton parity is CUDA-gated.
"""

from __future__ import annotations

import pytest
import torch

from hexfield_eq import constants as C
from hexfield_eq import equivariant as eq
from hexfield_eq.constants import DIRECTIONS
from hexfield_eq.geometry import apply_d6, disk_offsets
from hexfield_eq.model import HexfieldNet

pytestmark = pytest.mark.skipif(
    C.GROUP_ORDER != 12,
    reason="equivariant gate; run under the default HEXFIELD_EQ_GROUP_ORDER=12 build",
)

# fp32 tolerance. The equivariance is structural (fixed tied weights + the input
# rep), so the residual is fp32 round-off through the trunk: measured <= 2e-6 with
# every param randomized to N(0, 0.3); 1e-4 is a comfortable margin.
ATOL = 1e-4

# fp16 SERVE-path tolerance. The serve fast path forces fp16 conv GEMMs under
# autocast, so the residual is fp16 round-off through the tied trunk (the
# structure still commutes exactly with M(g)). Measured max deviation with every
# param randomized to N(0, 0.3): cov 2.2e-3 / inv 9.8e-4 (value/policy logits
# ~1.2 in magnitude); 5e-3 matches the fp16 serve-grade tolerance of the
# conv-parity test.
ATOL_SERVE = 5e-3

# Per-cell covariant heads permute with the board; the rest are D6-invariant.
COVARIANT_HEADS = ("policy", "opp_policy", "soft_policy", "cell_q")
INVARIANT_HEADS = (
    "value",
    "stvalue_2",
    "stvalue_6",
    "stvalue_16",
    "moves_left",
)


def _disk_board(radius: int = 3):
    """A G-closed hex disk (closed under all 12 D6 transforms about the origin),
    with its row-local neighbour gather, axial coords, all-live mask, and the
    cell-permutation SIG[g] each board symmetry induces."""

    cells = disk_offsets(radius)
    n = len(cells)
    cidx = {c: i for i, c in enumerate(cells)}
    nbr = torch.full((1, n, 6), n, dtype=torch.long)  # missing -> pad row (index n)
    for i, c in enumerate(cells):
        for d, off in enumerate(DIRECTIONS):
            nb = (c[0] + off[0], c[1] + off[1])
            if nb in cidx:
                nbr[0, i, d] = cidx[nb]
    coords = torch.tensor([[list(c) for c in cells]], dtype=torch.long)
    mask = torch.ones(1, n, dtype=torch.bool)
    sig = [
        torch.tensor([cidx[apply_d6(g, cells[i][0], cells[i][1])] for i in range(n)])
        for g in range(12)
    ]
    return cells, n, nbr, coords, mask, sig


_RIN = eq._in_rep_matrix()  # (12, NF, NF) input rep


def _transform_feats(feats: torch.Tensor, g: int, sig: torch.Tensor) -> torch.Tensor:
    """(T_g feats): permute cells by g AND apply the input rep rho_in(g) to the
    plane axis — feats_g[sig[i]] = rho_in(g) @ feats[i]."""

    fg = torch.zeros_like(feats)
    fg[0, sig] = feats[0] @ _RIN[g].T
    return fg


def _assert_equivariant(model: HexfieldNet, feats, nbr, mask, coords, sig) -> None:
    with torch.no_grad():
        base = model(feats, nbr, mask, coords)
        for g in range(12):
            og = model(_transform_feats(feats, g, sig[g]), nbr, mask, coords)
            for head in COVARIANT_HEADS:
                # policy(g.board)[g.x] == policy(board)[x]  (permute cells)
                lhs = og[head][0].index_select(0, sig[g])
                torch.testing.assert_close(
                    lhs, base[head][0], atol=ATOL, rtol=0, msg=f"{head} covariance g={g}"
                )
            for head in INVARIANT_HEADS:
                torch.testing.assert_close(
                    og[head], base[head], atol=ATOL, rtol=0,
                    msg=f"{head} invariance g={g}",
                )


# --- the gate: exact D6 equivariance ------------------------------------------


def test_equivariance_from_scratch_init() -> None:
    """A freshly-constructed net (zero bias tables, gamma=1/beta=0) is already
    exactly equivariant."""

    torch.manual_seed(0)
    _, n, nbr, coords, mask, sig = _disk_board(3)
    model = HexfieldNet().eval()
    feats = torch.randn(1, n, C.NUM_FEATURES)
    _assert_equivariant(model, feats, nbr, mask, coords, sig)


def test_equivariance_with_randomized_params() -> None:
    """Equivariance survives NON-trivial params: with every param randomized the
    jointly (row, head)-tied bias, the tied Q/K/V/out, the coset head split, the
    group-norm affine, the LayerScale, and the invariant tokens are all exercised
    away from their symmetric init. (An untied bias, per-head-free, would break
    here — see docs/DERIVATION §5 negative control.)"""

    torch.manual_seed(1)
    _, n, nbr, coords, mask, sig = _disk_board(3)
    model = HexfieldNet().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    _assert_equivariant(model, feats, nbr, mask, coords, sig)


# --- reference-path vs alternate-path parity ----------------------------------


def test_attention_impl_parity_sdpa_vs_materialized() -> None:
    """The equivariant attention (tied Q/K/V + coset head split + joint bias)
    agrees between the sdpa and the materialized reference impls."""

    torch.manual_seed(2)
    _, n, nbr, coords, mask, _ = _disk_board(3)
    model = HexfieldNet().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    with torch.no_grad():
        model.set_attention_impl("sdpa")
        a = model(feats, nbr, mask, coords)
        model.set_attention_impl("materialized")
        b = model(feats, nbr, mask, coords)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=2e-4, rtol=0, msg=key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + triton")
def test_reference_vs_triton_conv_parity() -> None:
    """The tied-generated dense weight is consumed identically by the reference
    GEMM and the fused Triton conv (docs/DERIVATION §2.3: 'the fp16 Triton
    conv/conv+LN kernels consume the generated weight unchanged'). If the kernel
    cannot compile for this shape it internally falls back to the reference path,
    so parity holds either way."""

    from hexfield_eq.model import HexNodeConv
    from hexfield_eq._triton_conv import hex_conv as tconv

    torch.manual_seed(3)
    dev = "cuda"
    ch = C.CHANNELS
    conv = HexNodeConv(ch, ch).to(dev).eval()
    _, n, nbr, coords, mask, _ = _disk_board(4)
    self_idx = torch.arange(n).reshape(1, n, 1)
    gidx = torch.cat([self_idx, nbr], dim=2).to(dev)
    mask = mask.to(dev)
    x = torch.randn(1, n, ch, device=dev, dtype=torch.float16)
    with torch.no_grad():
        w, b = conv._materialize()
        w16, b16 = w.half().contiguous(), b.half().contiguous()
        xe = torch.cat([x, x.new_zeros(1, 1, ch)], 1)
        flat = gidx.reshape(1, n * 7, 1).expand(-1, -1, ch)
        gathered = xe.gather(1, flat).reshape(1, n, 7 * ch)
        ref = (gathered.float() @ w.reshape(7 * ch, ch) + b) * mask.unsqueeze(-1)
        out = tconv(x, gidx, mask, w16, b16).float()
    # fp16 kernel vs fp32-accumulated reference: serve-grade tolerance.
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="serve path is CUDA-only")
def test_serve_path_equivariance(monkeypatch) -> None:
    """The SERVE fast path stays exactly D6-equivariant (to fp16 serve tolerance).

    Closes the two serve-path gaps from docs/BUGS_FOUND.md:

    * fp8 conv REMOVED — the id(weight)-keyed fp8 weight cache was a miss-every-
      forward leak under the tied trunk's per-forward weight regeneration, so the
      hex_conv_fp8 / hex_conv_ln_fp8 ops are gone; ConvBlock's serve path now
      always uses the plain fp16 conv+LN fusion.
    * fused conv+LN epilogue — the fused Triton kernel applies a per-CHANNEL
      affine in its LN epilogue. It is fed GroupAffineNorm's orbit-tied
      .weight/.bias, i.e. gamma/beta broadcast over the 12 slots
      (weight[slot*C_ORBIT + a] = gamma[a], docs/DERIVATION §3), so the epilogue
      is EXACTLY the equivariant full-fiber group-norm, not a free per-channel
      affine. No kernel change is needed and the fusion is kept for v1.

    Enables the serve fast paths by monkeypatching the module globals (otherwise
    gated off at import by HEXFIELD_TRITON_CONV*) and runs the net under autocast
    fp16 on CUDA, so ConvBlock takes the fused conv+LN branch and the head convs
    take the serve custom op. At C=96 the fused conv+LN kernel COMPILES and is
    exercised for real; the plain hex_conv head kernel currently fails to compile
    under triton 3.7.0 (C=96) and the custom op transparently falls back to its
    reference GEMM (the same equivariant serve path). It compiles at the D5 target
    C=192 (C_orbit=16). Either path must be equivariant, which is what we assert.
    """

    from hexfield_eq import model as M
    from hexfield_eq._triton_conv import hex_conv, hex_conv_ln

    if hex_conv_ln is None or hex_conv is None:  # torch built without triton
        pytest.skip("triton unavailable")

    torch.manual_seed(1)
    _, n, nbr, coords, mask, sig = _disk_board(3)
    dev = "cuda"
    model = M.HexfieldNet().to(dev).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3)

    # Transformed inputs are exact permutations, built in fp32 on CPU (the input
    # rep matrix _RIN is CPU) then moved to the device.
    feats = torch.randn(1, n, C.NUM_FEATURES)
    tfeats = [_transform_feats(feats, g, sig[g]).to(dev) for g in range(12)]
    feats = feats.to(dev)
    nbr, coords, mask = nbr.to(dev), coords.to(dev), mask.to(dev)
    sig = [s.to(dev) for s in sig]

    # Flip the serve fast paths ON (restored after the test by monkeypatch): the
    # ConvBlock fused conv+LN branch and the HexNodeConv serve conv branch both
    # read these module globals, which default to None (feature off).
    monkeypatch.setattr(M, "_hex_conv_fused", hex_conv)
    monkeypatch.setattr(M, "_hex_conv_ln_fused", hex_conv_ln)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        base = model(feats, nbr, mask, coords)
        for g in range(12):
            og = model(tfeats[g], nbr, mask, coords)
            for head in COVARIANT_HEADS:
                lhs = og[head][0].index_select(0, sig[g]).float()
                torch.testing.assert_close(
                    lhs, base[head][0].float(), atol=ATOL_SERVE, rtol=0,
                    msg=f"serve {head} covariance g={g}",
                )
            for head in INVARIANT_HEADS:
                torch.testing.assert_close(
                    og[head].float(), base[head].float(), atol=ATOL_SERVE, rtol=0,
                    msg=f"serve {head} invariance g={g}",
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="serve path is CUDA-only")
def test_serve_path_equivariance_with_register_lane(monkeypatch) -> None:
    """The fused serve fast path stays D6-equivariant with the REGISTER LANE on
    (spec D-S22/D-S27: near-zero out_proj + the fp32 token-lane carry): same
    harness as test_serve_path_equivariance, lane-carrying net."""

    from hexfield_eq import model as M
    from hexfield_eq._triton_conv import hex_conv, hex_conv_ln

    if hex_conv_ln is None or hex_conv is None:  # torch built without triton
        pytest.skip("triton unavailable")

    torch.manual_seed(6)
    _, n, nbr, coords, mask, sig = _disk_board(3)
    dev = "cuda"
    model = M.HexfieldNet(reg_lane=True, reg_tok_read=True).to(dev).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3)

    feats = torch.randn(1, n, C.NUM_FEATURES)
    tfeats = [_transform_feats(feats, g, sig[g]).to(dev) for g in range(12)]
    feats = feats.to(dev)
    nbr, coords, mask = nbr.to(dev), coords.to(dev), mask.to(dev)
    sig = [s.to(dev) for s in sig]

    monkeypatch.setattr(M, "_hex_conv_fused", hex_conv)
    monkeypatch.setattr(M, "_hex_conv_ln_fused", hex_conv_ln)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        base = model(feats, nbr, mask, coords)
        for g in range(12):
            og = model(tfeats[g], nbr, mask, coords)
            for head in COVARIANT_HEADS:
                lhs = og[head][0].index_select(0, sig[g]).float()
                torch.testing.assert_close(
                    lhs, base[head][0].float(), atol=ATOL_SERVE, rtol=0,
                    msg=f"serve+lane {head} covariance g={g}",
                )
            for head in INVARIANT_HEADS:
                torch.testing.assert_close(
                    og[head].float(), base[head].float(), atol=ATOL_SERVE, rtol=0,
                    msg=f"serve+lane {head} invariance g={g}",
                )


# --- grads reach every tied base param ----------------------------------------


def test_grads_reach_every_base_param() -> None:
    torch.manual_seed(4)
    _, n, nbr, coords, mask, _ = _disk_board(3)
    model = HexfieldNet()
    feats = torch.randn(1, n, C.NUM_FEATURES)
    out = model(feats, nbr, mask, coords)
    loss = sum(v.float().pow(2).mean() for v in out.values())
    loss.backward()

    missing = [nm for nm, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"params with no grad: {missing[:10]}"

    # The tied orbit base params specifically must receive a NONZERO gradient
    # (the index-gather touches every stored block; docs/DERIVATION §2.2).
    tied_markers = ("w_base", ".w0", ".wb", "bias_theta")
    tied = {
        nm: p
        for nm, p in model.named_parameters()
        if any(mk in nm for mk in tied_markers)
    }
    assert tied, "no tied base params found"
    for nm, p in tied.items():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, nm


def test_smoke_train_step_no_nan() -> None:
    torch.manual_seed(5)
    _, n, nbr, coords, mask, _ = _disk_board(3)
    model = HexfieldNet().train()
    feats = torch.randn(1, n, C.NUM_FEATURES)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        out = model(feats, nbr, mask, coords)
        loss = sum(v.float().pow(2).mean() for v in out.values())
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        opt.step()
    for nm, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite param after step: {nm}"
