"""Phase-L1 ray-attention gate (hexfield_eq).

docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase L1 /
docs/SPEC_REGISTER_LANE_RAY_ATTENTION.md §3. 'L' blocks are cells-only
attention masked to game-live rays, 6 heads = 3 win-axis cosets x {own, opp}
orbit-halves. This suite asserts:

  (a) head_perm6 STRUCTURE — the permutation conjugates the regular action to
      a block form over the 6 heads with the predicted head map
      ``2c+s -> 2*cosp[g](c)+s`` (own/opp halves preserved), and a deliberate
      K-SLOT split FAILS the same assertion (the plan's sharpest silent
      equivariance trap — K acts simply transitively on itself);
  (b) FULL-NET equivariance under an L layout (CCLACCLACLA, all 12 g,
      randomized params, register lane both off and on) with the raylen input
      transported by the covariance relation;
  (c) the joint (row, head) x side bias tie is exactly D6-invariant;
  (d) materialized == flex parity — the _FlexRayBias score_mod reproduces the
      materialized additive bias bitwise on a full (h, q, k) grid, and the
      eager flex_attention forward matches the materialized block output;
  (e) empty-ray softmax safety (all-zero raylen; lone-cell board);
  (f) layout/meta plumbing — L key set, arch_meta ray fields, meta-first
      rebuild + strict load, the RAY_BLOCKERS toggle (geometric rays ignore
      raylen; blockers-on without raylen fails loudly);
  (g) grads reach every ray-block param and BOTH side columns of bias_theta_l;
  (h) predicate classification (plugin/prefit AdamW + trainer trunk_attn).

Equivariance tests self-skip under GROUP_ORDER != 12; the structural tests
(d)-(h) run under both builds (passthrough L = plain 6-head masked attention).
Runs in the hexgt-build venv via PYTHONPATH=packages/hexfield_eq/python (plus
the shared packages). CPU-only.
"""

from __future__ import annotations

import pytest
import torch

from hexfield_eq import constants as C
from hexfield_eq.constants import DIRECTIONS
from hexfield_eq.geometry import apply_d6, disk_offsets, rel_bias_index
from hexfield_eq.model import HexfieldNet, infer_net_kwargs_from_state_dict

eq_only = pytest.mark.skipif(
    C.GROUP_ORDER != 12,
    reason="equivariant gate; run under the default HEXFIELD_EQ_GROUP_ORDER=12 build",
)

ATOL = 1e-4
RL = C.RAYLEN_SLOTS
L_LAYOUT = "CCLACCLACLA"  # the plan T2 primary arm (5C + 3L + 3A)
SMALL_LAYOUT = "CLA"

COVARIANT_HEADS = ("policy", "opp_policy", "soft_policy", "cell_q")
INVARIANT_HEADS = ("value", "stvalue_2", "stvalue_6", "stvalue_16", "moves_left")

_AXIS_VECS = ((1, 0), (0, 1), (1, -1))  # Q, R, QR


def _disk_board(radius: int = 3):
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
    sig = [
        torch.tensor([cidx[apply_d6(g, cells[i][0], cells[i][1])] for i in range(n)])
        for g in range(12)
    ]
    return cells, n, nbr, coords, mask, sig


def _slot_perm(g: int) -> list[int]:
    """perm[slot] = the raylen slot the value at ``slot`` lands on after g (the
    L0 covariance relation: side invariant, axis by sigma_g, direction by the
    transported ray vector)."""

    perm = [0] * RL
    for ai, (dq, dr) in enumerate(_AXIS_VECS):
        for di, sign in ((0, 1), (1, -1)):
            tq, tr = apply_d6(g, sign * dq, sign * dr)
            for aj, (aq, ar) in enumerate(_AXIS_VECS):
                if (tq, tr) == (aq, ar):
                    tgt = aj * 2 + 0
                    break
                if (tq, tr) == (-aq, -ar):
                    tgt = aj * 2 + 1
                    break
            else:  # pragma: no cover
                raise AssertionError(f"g={g} maps an axis dir off-axis")
            for side in range(2):
                perm[side * 6 + ai * 2 + di] = side * 6 + tgt
    return perm


def _transform_raylen(raylen: torch.Tensor, g: int, sig: torch.Tensor) -> torch.Tensor:
    """raylen_g[g.x, perm[slot]] = raylen[x, slot]."""

    perm = _slot_perm(g)
    inv = [0] * RL
    for s, t in enumerate(perm):
        inv[t] = s
    out = torch.zeros_like(raylen)
    out[0, sig] = raylen[0][:, inv]
    return out


def _randomize(model: HexfieldNet, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3)


def _bound_lane_scales(model: HexfieldNet) -> None:
    """Re-bound the register lane's scale params after blanket randomization
    (mirrors test_hexfield_eq_register_lane): blanket N(0, 0.3) explodes the
    unnormalized count magnitudes (sum_scale ~10x design, background gates
    ~0.5) and with them the fp32 round-off the tolerance absorbs; the
    structural tie is what is under test."""

    with torch.no_grad():
        for nm, p in model.named_parameters():
            if nm.endswith("gate_bias"):
                p.uniform_(-3.0, -1.0)
            elif nm.endswith("sum_scale"):
                p.uniform_(0.01, 0.06)


# --- (a) head_perm6 structure + the K-slot trap ----------------------------------


@eq_only
def test_head_perm6_structure_and_k_slot_trap() -> None:
    from hexfield_eq import equivariant as eq

    G = eq.build_group()
    mult, cosets, cosp = G["mult"], G["cosets"], G["cosp"]
    co = C.C_ORBIT
    n_ch = 12 * co
    head_dim = 2 * co

    perm6 = eq.head_perm6().tolist()
    inv6 = eq.head_perm6_inv()
    assert torch.equal(inv6[eq.head_perm6()], torch.arange(n_ch))

    def check_block_system(order: list[int]) -> list[tuple]:
        """Violations of 'the head partition is a D6-block system with head map
        2c+s -> 2*cosp[g](c)+s' for the given channel order."""

        pos_of = [0] * n_ch
        for p, ch in enumerate(order):
            pos_of[ch] = p
        violations = []
        for g in range(12):
            for p in range(n_ch):
                slot, a = divmod(order[p], co)
                dst_pos = pos_of[mult[g][slot] * co + a]
                h_src = p // head_dim
                h_dst = dst_pos // head_dim
                want = 2 * cosp[g][h_src // 2] + h_src % 2
                if h_dst != want:
                    violations.append((g, p, h_dst, want))
        return violations

    # head_perm6 (orbit-half split): a perfect block system, halves preserved.
    assert check_block_system(perm6) == []

    # NEGATIVE CONTROL — the K-slot split (first 2 slots vs last 2 slots of
    # each coset): K acts simply transitively on itself, so some P_K(g) crosses
    # every 2+2 slot partition and the head map breaks SILENTLY (same shapes).
    bad_order = [
        slot * co + o
        for c in range(3)
        for half_slots in (cosets[c][:2], cosets[c][2:])
        for slot in half_slots
        for o in range(co)
    ]
    assert len(bad_order) == n_ch
    assert check_block_system(bad_order), "K-slot split should break the block system"


# --- (b) full-net equivariance under an L layout ----------------------------------


@eq_only
@pytest.mark.parametrize("reg_lane", [False, True], ids=["plain", "reg_lane"])
def test_full_net_equivariance_L_layout(reg_lane: bool) -> None:
    _, n, nbr, coords, mask, sig = _disk_board(3)
    model = HexfieldNet(
        trunk_layout=L_LAYOUT, reg_lane=reg_lane, reg_tok_read=reg_lane
    ).eval()
    _randomize(model, 2)
    if reg_lane:
        _bound_lane_scales(model)
    torch.manual_seed(21)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    raylen = torch.randint(0, C.RAY_REACH + 1, (1, n, RL), dtype=torch.uint8)
    with torch.no_grad():
        base = model(feats, nbr, mask, coords, raylen=raylen)
        for g in range(12):
            fg = torch.zeros_like(feats)
            from hexfield_eq import equivariant as eq

            fg[0, sig[g]] = feats[0] @ eq._in_rep_matrix()[g].T
            og = model(
                fg, nbr, mask, coords, raylen=_transform_raylen(raylen, g, sig[g])
            )
            for head in COVARIANT_HEADS:
                lhs = og[head][0].index_select(0, sig[g])
                torch.testing.assert_close(
                    lhs, base[head][0], atol=ATOL, rtol=0,
                    msg=f"{head} covariance g={g}",
                )
            for head in INVARIANT_HEADS:
                torch.testing.assert_close(
                    og[head], base[head], atol=ATOL, rtol=0,
                    msg=f"{head} invariance g={g}",
                )


# --- (c) the joint (row, head) x side tie is D6-invariant ---------------------------


@eq_only
def test_ray_bias_table_joint_side_tie_is_d6_invariant() -> None:
    from hexfield_eq import equivariant as eq

    torch.manual_seed(3)
    model = HexfieldNet(trunk_layout=SMALL_LAYOUT)
    with torch.no_grad():
        for p in model.bias_theta_l:
            p.copy_(torch.randn_like(p))
    table = model._ray_bias_table(0)  # (BIAS_ROWS, 6)
    cosp = eq.build_group()["cosp"]

    for dq, dr in disk_offsets(C.BIAS_DISK_RADIUS):
        row = rel_bias_index(dq, dr)
        for g in range(12):
            gq, gr = apply_d6(g, dq, dr)
            row_g = rel_bias_index(gq, gr)
            for c in range(3):
                for s in range(2):
                    assert table[row_g, 2 * cosp[g][c] + s] == table[row, 2 * c + s], (
                        (dq, dr), g, c, s,
                    )


# --- (d) materialized == flex parity -------------------------------------------------


def test_materialized_matches_flex(monkeypatch) -> None:
    torch.manual_seed(4)
    _, n, nbr, coords, mask, _ = _disk_board(3)
    model = HexfieldNet(trunk_layout=SMALL_LAYOUT).eval()
    _randomize(model, 4)
    raylen = torch.randint(0, C.RAY_REACH + 1, (1, n, RL), dtype=torch.uint8)

    # (d1) the score_mod's additive term reproduces the materialized bias
    # BITWISE over the full (h, q, k) grid (score = 0 probe; same float ops).
    bias_mat = model._build_ray_bias(coords, mask, raylen, 0)  # fp32, grad mode
    carrier = model._build_ray_flex_bias(coords, mask, raylen, 0)  # fp32 table
    score_mod = carrier.make_score_mod()
    hg = torch.arange(C.RAY_HEADS).view(-1, 1, 1)
    qg = torch.arange(n).view(1, -1, 1)
    kg = torch.arange(n).view(1, 1, -1)
    probed = score_mod(torch.zeros(C.RAY_HEADS, n, n), 0, hg, qg, kg)
    assert torch.equal(probed, bias_mat[0]), "flex score_mod != materialized bias"

    # (d2) the real flex op (eager) through the block matches the materialized
    # block output.
    try:
        from torch.nn.attention.flex_attention import flex_attention
    except ImportError:
        pytest.skip("torch without flex_attention")
    from hexfield_eq import model as M

    monkeypatch.setattr(
        M, "_flex_call", lambda q, k, v, score_mod: flex_attention(q, k, v, score_mod=score_mod)
    )
    block = model.ray_blocks[0]
    torch.manual_seed(14)
    x = torch.randn(1, n, C.CHANNELS)
    with torch.no_grad():
        ref = block(x, model._build_ray_bias(coords, mask, raylen, 0), mask)
        out_flex = block(x, model._build_ray_flex_bias(coords, mask, raylen, 0), mask)
    torch.testing.assert_close(out_flex, ref, atol=2e-4, rtol=0)


# --- (e) empty-ray softmax safety ----------------------------------------------------


def test_empty_ray_softmax_safety() -> None:
    torch.manual_seed(5)
    _, n, nbr, coords, mask, _ = _disk_board(3)
    model = HexfieldNet(trunk_layout=SMALL_LAYOUT).eval()
    _randomize(model, 5)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    # All-zero raylen: every ray is dead, only the diagonal is live.
    zero_rl = torch.zeros(1, n, RL, dtype=torch.uint8)
    with torch.no_grad():
        out = model(feats, nbr, mask, coords, raylen=zero_rl)
    for key, val in out.items():
        assert torch.isfinite(val).all(), key

    # Lone-cell board (disk radius 0): 1 cell, no neighbours, self-only rays.
    _, n1, nbr1, coords1, mask1, _ = _disk_board(0)
    assert n1 == 1
    feats1 = torch.randn(1, 1, C.NUM_FEATURES)
    with torch.no_grad():
        out1 = model(
            feats1, nbr1, mask1, coords1,
            raylen=torch.zeros(1, 1, RL, dtype=torch.uint8),
        )
    for key, val in out1.items():
        assert torch.isfinite(val).all(), key


# --- (f) layout / meta plumbing + the blockers toggle ---------------------------------


def test_L_layout_key_set_and_meta_round_trip() -> None:
    plain = HexfieldNet(trunk_layout="CCA")
    assert not any(
        k.startswith(("ray_blocks.", "bias_theta_l.", "ray_bias_free_tables."))
        for k in plain.state_dict()
    )
    assert "ray_heads" not in plain.arch_meta()

    model = HexfieldNet(trunk_layout=SMALL_LAYOUT)
    keys = set(model.state_dict())
    assert any(k.startswith("ray_blocks.") for k in keys)
    if C.GROUP_ORDER == 12:
        assert any(k.startswith("bias_theta_l.") for k in keys)
        assert model.bias_theta_l[0].shape == (model._n_joint_classes, 2)
    else:
        assert any(k.startswith("ray_bias_free_tables.") for k in keys)

    meta = model.arch_meta()
    assert meta["trunk_layout"] == SMALL_LAYOUT
    assert meta["ray_heads"] == C.RAY_HEADS == 6
    assert meta["ray_blockers"] is True  # env default

    kwargs = infer_net_kwargs_from_state_dict(model.state_dict(), meta)
    assert kwargs["trunk_layout"] == SMALL_LAYOUT
    assert kwargs["ray_blockers"] is True
    rebuilt = HexfieldNet(**kwargs)
    rebuilt.load_state_dict(model.state_dict(), strict=True)

    geo = HexfieldNet(trunk_layout=SMALL_LAYOUT, ray_blockers=False)
    assert geo.arch_meta()["ray_blockers"] is False
    kwargs_geo = infer_net_kwargs_from_state_dict(geo.state_dict(), geo.arch_meta())
    assert kwargs_geo["ray_blockers"] is False


def test_ray_blockers_toggle() -> None:
    torch.manual_seed(6)
    _, n, nbr, coords, mask, _ = _disk_board(3)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    raylen = torch.randint(0, C.RAY_REACH + 1, (1, n, RL), dtype=torch.uint8)

    # Geometric rays (blockers off): raylen is ignored — with and without the
    # input the outputs are bit-identical.
    torch.manual_seed(7)
    geo = HexfieldNet(trunk_layout=SMALL_LAYOUT, ray_blockers=False).eval()
    with torch.no_grad():
        a = geo(feats, nbr, mask, coords)
        b = geo(feats, nbr, mask, coords, raylen=raylen)
    for key in a:
        assert torch.equal(a[key], b[key]), key
    # ...and the geometric mask differs from the blocker mask for a truncating
    # raylen (the toggle is a real semantic change).
    dq = coords[:, None, :, 0] - coords[:, :, None, 0]
    dr = coords[:, None, :, 1] - coords[:, :, None, 1]
    torch.manual_seed(7)
    blk = HexfieldNet(trunk_layout=SMALL_LAYOUT, ray_blockers=True).eval()
    m_geo = geo._ray_live_mask(dq, dr, None)
    m_blk = blk._ray_live_mask(dq, dr, torch.zeros(1, n, RL, dtype=torch.uint8))
    assert not torch.equal(m_geo, m_blk)

    # Blockers on without a raylen input fails loudly.
    with pytest.raises(ValueError, match="raylen"):
        with torch.no_grad():
            blk(feats, nbr, mask, coords)


# --- (g) grads reach every ray param ---------------------------------------------------


def test_grads_reach_ray_params_and_both_sides() -> None:
    _, n, nbr, coords, mask, _ = _disk_board(3)
    model = HexfieldNet(trunk_layout=L_LAYOUT)
    _randomize(model, 8)
    torch.manual_seed(18)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    raylen = torch.randint(0, C.RAY_REACH + 1, (1, n, RL), dtype=torch.uint8)
    out = model(feats, nbr, mask, coords, raylen=raylen)
    loss = sum(v.float().pow(2).mean() for v in out.values())
    loss.backward()

    ray_params = {
        nm: p
        for nm, p in model.named_parameters()
        if nm.startswith(("ray_blocks.", "bias_theta_l.", "ray_bias_free_tables."))
    }
    assert ray_params, "no ray params found"
    for nm, p in ray_params.items():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, nm
    if C.GROUP_ORDER == 12:
        for i, p in enumerate(model.bias_theta_l):
            for side in range(2):
                assert float(p.grad[:, side].abs().sum()) > 0.0, (
                    f"bias_theta_l.{i} side {side} got no gradient"
                )


# --- (h) predicate classification --------------------------------------------------------


def _expected_ray_decay(name: str) -> bool:
    is_matrix = name.endswith(".wb") or name.endswith(".weight")
    is_proj = any(
        f".{p}." in name
        for p in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2")
    )
    return is_matrix and is_proj and ".ln" not in name


def _adamw_groups(opt: torch.optim.AdamW) -> tuple[set, set]:
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
    return decayed, no_decay


def _assert_ray_classification(model: HexfieldNet, opt: torch.optim.AdamW) -> None:
    decayed, no_decay = _adamw_groups(opt)
    checked = 0
    for nm, p in model.named_parameters():
        if nm.startswith("ray_blocks."):
            if _expected_ray_decay(nm):
                assert id(p) in decayed, f"{nm} should decay"
            else:
                assert id(p) in no_decay, f"{nm} should be no-decay"
            checked += 1
        elif nm.startswith(("bias_theta_l.", "ray_bias_free_tables.")):
            # The joint-tied / orbit-tied ray bias params are no-decay by the
            # existing named predicates ("bias_theta" / "bias_free_table").
            assert id(p) in no_decay, f"{nm} should be no-decay"
            checked += 1
    assert checked > 0


def test_prefit_optimizer_classifies_ray_params() -> None:
    try:
        from hexfield_eq.prefit import make_optimizer
    except ImportError as exc:
        pytest.skip(f"prefit import chain unavailable: {exc}")

    model = HexfieldNet(trunk_layout=SMALL_LAYOUT)
    _assert_ray_classification(model, make_optimizer(model))


def test_plugin_optimizer_classifies_ray_params() -> None:
    try:
        from hexfield_eq.plugin import get_plugin
    except ImportError as exc:
        pytest.skip(f"plugin import chain unavailable: {exc}")

    plugin = get_plugin()
    model = HexfieldNet(trunk_layout=SMALL_LAYOUT)
    overrides = plugin.training_component_overrides(
        defaults=None, config={}, shared=None, model=model
    )
    _assert_ray_classification(model, overrides.optimizer)


def test_trainer_grad_norm_groups_put_ray_params_in_trunk_attn() -> None:
    try:
        from hexfield_eq.trainer import HexfieldTrainer
    except ImportError as exc:
        pytest.skip(f"trainer import chain unavailable: {exc}")

    from types import SimpleNamespace

    model = HexfieldNet(trunk_layout=L_LAYOUT, reg_lane=True, reg_tok_read=True)
    groups = HexfieldTrainer._build_grad_norm_groups(SimpleNamespace(model=model))
    attn_ids = {id(p) for p in groups["trunk_attn"]}
    reg_ids = {id(p) for p in groups["trunk_reg"]}
    for nm, p in model.named_parameters():
        if nm.startswith(("ray_blocks.", "bias_theta_l.", "ray_bias_free_tables.")):
            assert id(p) in attn_ids, f"{nm} should be trunk_attn"
        if nm.startswith(("registers_l.", "tok_reads_l.")):
            assert id(p) in reg_ids, f"{nm} should be trunk_reg"
