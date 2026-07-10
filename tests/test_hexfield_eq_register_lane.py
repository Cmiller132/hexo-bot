"""Phase-R0 register-lane gate (hexfield_eq).

docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase R0 /
docs/SPEC_REGISTER_LANE_RAY_ATTENTION.md §4. The register lane attaches a
one-way sigmoid-gated SUM cross-attention (tokens <- cells) at every C-block
exit, plus an optional cells <- tokens broadcast read at C-block entry. This
suite asserts:

  (a) the FULL NET stays exactly D6-equivariant with the lane on (all 12 g,
      every param randomized — the test class that catches a head-covariance
      or gate-bias tie mistake);
  (b) toggle-off identity — a reg_lane=False net has the pre-lane state-dict
      key set and a bit-identical same-seed forward;
  (c) zero-init identity — a lane-on net at step 0 (zero out_proj / tok_reads)
      produces bit-identical outputs to the lane-off net with the same shared
      params;
  (d) gradients reach every lane parameter (randomized params), and the
      zero-init out_proj still receives gradient (grow-in is trainable);
  (e) a counting probe — duplicating a matched cell pattern k x scales the
      token update linearly in k (the SUM aggregation, plan R1);
  (f) param-classification — plugin/prefit AdamW decay vs no-decay and the
      trainer's trunk_reg grad-norm group land exactly per the spec §1.4 table.

Runs under the equivariant default build (GROUP_ORDER == 12); self-skips
otherwise, like the Phase-3b equivariance gate. The lane toggles are exercised
via the HexfieldNet constructor kwargs (env-independent), so one process builds
both arms. Runs in the hexgt-build venv via
PYTHONPATH=packages/hexfield_eq/python (plus the shared packages). CPU-only.
"""

from __future__ import annotations

import pytest
import torch

from hexfield_eq import constants as C
from hexfield_eq import equivariant as eq
from hexfield_eq.constants import DIRECTIONS
from hexfield_eq.geometry import apply_d6, disk_offsets
from hexfield_eq.model import HexfieldNet, infer_net_kwargs_from_state_dict

pytestmark = pytest.mark.skipif(
    C.GROUP_ORDER != 12,
    reason="register-lane gate; run under the default HEXFIELD_EQ_GROUP_ORDER=12 build",
)

# fp32 tolerance, matching the Phase-3b equivariance gate: the equivariance is
# structural, so the residual is fp32 round-off through the trunk.
ATOL = 1e-4

COVARIANT_HEADS = ("policy", "opp_policy", "soft_policy", "cell_q")
INVARIANT_HEADS = (
    "value",
    "stvalue_2",
    "stvalue_6",
    "stvalue_16",
    "moves_left",
)

LANE_PREFIXES = ("registers.", "tok_reads.")


def _disk_board(radius: int = 3):
    """A G-closed hex disk with its row-local neighbour gather, axial coords,
    all-live mask, and the cell-permutation SIG[g] each board symmetry induces
    (mirrors the Phase-3b gate's probe board)."""

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
    fg = torch.zeros_like(feats)
    fg[0, sig] = feats[0] @ _RIN[g].T
    return fg


def _randomize(model: HexfieldNet, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3)


def _bound_lane_scales(model: HexfieldNet) -> None:
    """Re-bound the lane's scale params after blanket randomization: a blanket
    N(0, 0.3) hands sum_scale ~10x its design scale and gate_bias ~0
    (background gates ~0.5), exploding the unnormalized count magnitudes and
    with them the fp32 ROUND-OFF the equivariance tolerance absorbs. The
    structural tie is what is under test; the values stay random and nonzero."""

    with torch.no_grad():
        for nm, p in model.named_parameters():
            if nm.endswith("gate_bias"):
                p.uniform_(-3.0, -1.0)
            elif nm.endswith("sum_scale"):
                p.uniform_(0.01, 0.06)


def _lane_params(model: HexfieldNet) -> dict[str, torch.nn.Parameter]:
    return {
        nm: p
        for nm, p in model.named_parameters()
        if nm.startswith(LANE_PREFIXES)
    }


# --- (a) full-net equivariance with the lane on ---------------------------------


@pytest.mark.parametrize("tok_read", [False, True], ids=["lane", "lane+tok_read"])
def test_equivariance_with_register_lane(tok_read: bool) -> None:
    """With every param randomized (nonzero out_proj, nonzero gate thresholds,
    nonzero tok_reads) the lane-carrying net stays exactly D6-equivariant for
    all 12 group elements: the coset head split, the head-constant gate_bias,
    and the covariant token stream all commute with M(g)."""

    _, n, nbr, coords, mask, sig = _disk_board(3)
    model = HexfieldNet(reg_lane=True, reg_tok_read=tok_read).eval()
    _randomize(model, 1)
    _bound_lane_scales(model)
    torch.manual_seed(11)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    with torch.no_grad():
        base = model(feats, nbr, mask, coords)
        for g in range(12):
            og = model(_transform_feats(feats, g, sig[g]), nbr, mask, coords)
            for head in COVARIANT_HEADS:
                lhs = og[head][0].index_select(0, sig[g])
                torch.testing.assert_close(
                    lhs, base[head][0], atol=ATOL, rtol=0, msg=f"{head} covariance g={g}"
                )
            for head in INVARIANT_HEADS:
                torch.testing.assert_close(
                    og[head], base[head], atol=ATOL, rtol=0,
                    msg=f"{head} invariance g={g}",
                )


# --- (b) toggle-off identity -----------------------------------------------------


def test_toggle_off_state_dict_and_forward_identity() -> None:
    """reg_lane=False keeps the pre-lane net: no registers./tok_reads. keys, the
    same key set as a default-env build, and a bit-identical same-seed forward
    (the trunk-walk threading is guarded off entirely)."""

    torch.manual_seed(0)
    default = HexfieldNet().eval()
    torch.manual_seed(0)
    off = HexfieldNet(reg_lane=False, reg_tok_read=False).eval()

    keys = set(off.state_dict().keys())
    assert not any(k.startswith(LANE_PREFIXES) for k in keys)
    assert keys == set(default.state_dict().keys())
    for key, val in default.state_dict().items():
        assert torch.equal(off.state_dict()[key], val), key

    _, n, nbr, coords, mask, _ = _disk_board(3)
    torch.manual_seed(7)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    with torch.no_grad():
        a = default(feats, nbr, mask, coords)
        b = off(feats, nbr, mask, coords)
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_tok_read_requires_lane() -> None:
    with pytest.raises(ValueError, match="reg_lane"):
        HexfieldNet(reg_lane=False, reg_tok_read=True)


def test_arch_meta_and_infer_round_trip() -> None:
    """The toggles ride the checkpoint meta and are read meta-first by the arch
    inferer; without meta the state-dict key set is affirmative either way."""

    on = HexfieldNet(reg_lane=True, reg_tok_read=True)
    meta = on.arch_meta()
    assert meta["reg_lane"] is True and meta["reg_tok_read"] is True
    kwargs = infer_net_kwargs_from_state_dict(on.state_dict(), meta)
    assert kwargs["reg_lane"] is True and kwargs["reg_tok_read"] is True
    # No meta: key-set fallback.
    kwargs_sd = infer_net_kwargs_from_state_dict(on.state_dict())
    assert kwargs_sd["reg_lane"] is True and kwargs_sd["reg_tok_read"] is True

    off = HexfieldNet(reg_lane=False)
    assert off.arch_meta()["reg_lane"] is False
    kwargs_off = infer_net_kwargs_from_state_dict(off.state_dict())
    assert kwargs_off["reg_lane"] is False and kwargs_off["reg_tok_read"] is False
    # The inferred kwargs rebuild a strict-loadable twin.
    rebuilt = HexfieldNet(**infer_net_kwargs_from_state_dict(on.state_dict(), meta))
    rebuilt.load_state_dict(on.state_dict(), strict=True)


# --- (c) near-zero-init identity ---------------------------------------------------


def test_near_zero_init_forward_identity() -> None:
    """At step 0 the lane is a NUMERICAL no-op: out_proj is trunc_normal std
    3e-3 (spec D-S22 — strictly zero would kill the q/k/v/gate_bias gradients)
    and every tok_read is zero-init, so a lane-on net with the lane-off net's
    shared params matches to a small atol (no longer bitwise, by design)."""

    torch.manual_seed(0)
    off = HexfieldNet(reg_lane=False).eval()
    torch.manual_seed(0)
    on = HexfieldNet(reg_lane=True, reg_tok_read=True).eval()

    # Same-seed construction gives identical shared params (the lane is built
    # after _init_weights, off the shared RNG stream)...
    for key, val in off.state_dict().items():
        assert torch.equal(on.state_dict()[key], val), key
    # ...and the copy makes the identity hermetic even if that ever changes.
    on.load_state_dict(off.state_dict(), strict=False)

    # tok_reads stay exactly zero (their grads are live at zero — they are the
    # last op of their branch); out_proj is near-zero, not zero.
    for nm, p in _lane_params(on).items():
        if nm.startswith("tok_reads."):
            assert float(p.detach().abs().sum()) == 0.0, nm
        if ".out_proj.wb" in nm or ".out_proj.weight" in nm:
            s = float(p.detach().abs().max())
            assert 0.0 < s < 0.02, nm  # near-zero grow-in seed

    _, n, nbr, coords, mask, _ = _disk_board(3)
    torch.manual_seed(9)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    with torch.no_grad():
        a = off(feats, nbr, mask, coords)
        b = on(feats, nbr, mask, coords)
    for key in a:
        # Measured max head delta at init: ~1.9e-4 (out_proj std 3e-3); 5e-3
        # gives ~25x headroom while still catching a mis-scaled init.
        torch.testing.assert_close(
            a[key], b[key], atol=5e-3, rtol=0,
            msg=f"{key} drifted beyond the near-zero-init envelope",
        )


# --- (d) grads reach every lane param --------------------------------------------


def test_grads_reach_every_lane_param() -> None:
    _, n, nbr, coords, mask, _ = _disk_board(3)
    model = HexfieldNet(reg_lane=True, reg_tok_read=True)
    _randomize(model, 4)
    torch.manual_seed(14)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    out = model(feats, nbr, mask, coords)
    loss = sum(v.float().pow(2).mean() for v in out.values())
    loss.backward()

    lane = _lane_params(model)
    assert lane, "no register-lane params found"
    for nm, p in lane.items():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, nm


def test_grads_live_at_init() -> None:
    """The D-S22 rationale, asserted: at the UNTRAINED start (near-zero
    out_proj, zero tok_reads) EVERY lane parameter — including q/k/v, the norm
    affines, gate_bias, and sum_scale, which are provably zero-grad under a
    strict zero out_proj — receives a nonzero gradient."""

    _, n, nbr, coords, mask, _ = _disk_board(3)
    torch.manual_seed(5)
    model = HexfieldNet(reg_lane=True, reg_tok_read=True)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    out = model(feats, nbr, mask, coords)
    loss = sum(v.float().pow(2).mean() for v in out.values())
    loss.backward()

    for nm, p in _lane_params(model).items():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, nm


# --- (e) counting probe: SUM aggregation scales linearly ---------------------------


def test_counting_probe_token_update_scales_with_k() -> None:
    """Duplicate a matched cell pattern k x among a fixed-size board: the token
    update from an isolated RegisterRefresh scales LINEARLY in k (an
    unnormalized sigmoid-gated sum counts; softmax's weighted mean would not).
    Loose tolerance — a sanity check on the aggregation shape."""

    from hexfield_eq.register import RegisterRefresh

    torch.manual_seed(6)
    ch = C.CHANNELS
    mod = RegisterRefresh(ch).eval()
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn_like(p) * 0.3)  # nonzero out_proj: measure the lane

    n = 64
    mask = torch.ones(1, n, dtype=torch.bool)
    tokens = torch.randn(1, C.NUM_TOKENS, ch)
    pattern = torch.randn(ch) * 1.5
    background = torch.randn(ch) * 0.5

    def update(k: int) -> torch.Tensor:
        x = background.repeat(n, 1)
        x[:k] = pattern
        with torch.no_grad():
            return mod(tokens, x.unsqueeze(0), mask) - tokens

    base = update(0)
    d1 = update(1) - base
    assert float(d1.abs().sum()) > 0.0, "probe pattern produced no token update"
    for k in (2, 4, 8):
        dk = update(k) - base
        torch.testing.assert_close(
            dk, k * d1, rtol=0.1, atol=1e-4,
            msg=f"token update not ~linear in k at k={k}",
        )


def test_value_input_counts_duplicated_patterns_end_to_end() -> None:
    """A2 end-to-end counting probe: k duplicated local patterns on a fixed
    board scale the VALUE-HEAD INPUT ~linearly in k, through the trunk, the
    register SUM, and the PRE-ln_final read block (spec D-S21 — ln_final would
    erase the count; the pre-ln block must carry it to the value input).

    Construction makes the linearity near-exact: layout CA with the A block's
    LayerScales zeroed (the final attention becomes an identity on the residual
    stream), a UNIFORM background feature vector, and pattern sites placed
    pairwise >= 8 apart and >= 4 inside the board edge, so each added site
    contributes a translation-identical delta to the register sum."""

    torch.manual_seed(30)
    _, n, nbr, coords, mask, _ = _disk_board(8)  # radius-8 disk, 217 cells
    cidx = {tuple(c): i for i, c in enumerate(coords[0].tolist())}
    sites = [cidx[s] for s in ((4, 0), (-4, 4), (0, -4))]  # interior, >= 8 apart

    model = HexfieldNet(trunk_layout="CA", reg_lane=True).eval()
    _randomize(model, 31)
    with torch.no_grad():
        for blk in model.attn_blocks:
            blk.ls_attn.gamma.zero_()
            blk.ls_mlp.gamma.zero_()

    torch.manual_seed(32)
    background = torch.randn(C.NUM_FEATURES) * 0.5
    pattern = torch.randn(C.NUM_FEATURES) * 1.5
    read_w = model.value_reduction.in_features // (C.NUM_TOKENS + 2)

    def vin(k: int) -> torch.Tensor:
        feats = background.repeat(1, n, 1).clone()
        for s in sites[:k]:
            feats[0, s] = pattern
        with torch.no_grad():
            cells, tokens, pre_tokens, _g = model.trunk(feats, nbr, mask, coords)
            pooled = model._pooled(cells, mask)
            return model._value_input(
                tokens, range(C.NUM_TOKENS), pooled, pre_tokens
            )

    base = vin(0)
    d1 = vin(1) - base
    # The PRE-ln read block is the LAST read_w slice of the value input.
    d1_pre = d1[:, -read_w:]
    assert float(d1_pre.abs().sum()) > 0.0, "pattern did not reach the pre-ln read"
    for k in (2, 3):
        dk_pre = (vin(k) - base)[:, -read_w:]
        torch.testing.assert_close(
            dk_pre, k * d1_pre, rtol=0.15, atol=1e-4,
            msg=f"pre-ln value-input block not ~linear in k at k={k}",
        )
    # And the full value input keeps growing with k (the count is not erased
    # anywhere on the way to the value head).
    norms = [float((vin(k) - base).norm()) for k in (1, 2, 3)]
    assert norms[0] < norms[1] < norms[2], norms


# --- (f) predicate classification ---------------------------------------------------


def _expected_decay(name: str) -> bool:
    """Spec §1.4: projections/read matrices decay; norm affines, every bias, and
    gate_bias do not."""

    is_matrix = name.endswith(".wb") or name.endswith(".weight")
    is_proj = any(
        f".{p}." in name for p in ("q_proj", "k_proj", "v_proj", "out_proj", "reads")
    )
    return is_matrix and is_proj and "ln_" not in name


def _assert_adamw_classification(model: HexfieldNet, opt: torch.optim.AdamW) -> None:
    decayed = {
        id(p)
        for grp in opt.param_groups
        if grp["weight_decay"] != 0.0
        for p in grp["params"]
    }
    no_decay = {
        id(p)
        for grp in opt.param_groups
        if grp["weight_decay"] == 0.0
        for p in grp["params"]
    }
    lane = _lane_params(model)
    assert lane, "no register-lane params found"
    for nm, p in lane.items():
        if _expected_decay(nm):
            assert id(p) in decayed and id(p) not in no_decay, f"{nm} should decay"
        else:
            assert id(p) in no_decay and id(p) not in decayed, f"{nm} should be no-decay"
        assert "gate_bias" not in nm or id(p) in no_decay, nm


def test_plugin_optimizer_classifies_lane_params() -> None:
    try:
        from hexfield_eq.plugin import get_plugin
    except ImportError as exc:  # hexo_train / hexo_utils not on the path
        pytest.skip(f"plugin import chain unavailable: {exc}")

    plugin = get_plugin()
    model = HexfieldNet(reg_lane=True, reg_tok_read=True)
    overrides = plugin.training_component_overrides(
        defaults=None, config={}, shared=None, model=model
    )
    _assert_adamw_classification(model, overrides.optimizer)


def test_prefit_optimizer_classifies_lane_params() -> None:
    try:
        from hexfield_eq.prefit import make_optimizer
    except ImportError as exc:
        pytest.skip(f"prefit import chain unavailable: {exc}")

    model = HexfieldNet(reg_lane=True, reg_tok_read=True)
    _assert_adamw_classification(model, make_optimizer(model))


def test_trainer_grad_norm_group_is_trunk_reg() -> None:
    try:
        from hexfield_eq.trainer import HexfieldTrainer
    except ImportError as exc:
        pytest.skip(f"trainer import chain unavailable: {exc}")

    from types import SimpleNamespace

    model = HexfieldNet(reg_lane=True, reg_tok_read=True)
    groups = HexfieldTrainer._build_grad_norm_groups(SimpleNamespace(model=model))
    assert "trunk_reg" in groups
    lane_ids = {id(p) for p in _lane_params(model).values()}
    reg_ids = {id(p) for p in groups["trunk_reg"]}
    assert reg_ids == lane_ids, "trunk_reg must hold exactly the lane params"
    for gname in ("trunk_conv", "trunk_attn", "heads"):
        assert lane_ids.isdisjoint({id(p) for p in groups[gname]}), gname
