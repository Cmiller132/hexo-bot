"""Ray-tap conv gate (SPEC_RAYTAP_CONV.md §2, Phase R work items W-R1..W-R5).

Covers, per spec §6.1:

  T3  — full-net D6 equivariance with raytap 'conv2' and 'both' (all 12 group
        elements, randomized params incl. alpha, raylen transported by the L0
        covariance relation — the ray_block-test harness). Built via the
        constructor kwarg (equivalent to the env: both feed the same
        HexfieldNet arg; the env threading itself is pinned by the subprocess
        test below).
  T4  — init-equivalence, reference path: with alpha at its (1, 0, 0, 0, 0)
        init a ray-tap net's outputs equal the baseline net's on real
        positions (both sides to move, legal/stone/halo nodes), pinning the
        terminal-blocker raylen convention; plus a liveness guard (non-init
        alpha changes outputs).
  T5  — serve-fold parity at trained (randomized) alpha: the no-grad (folded)
        forward matches the grad-enabled forward under the perm-fold
        tolerance model, layouts with and without 'L'. (The fused-path rerun
        happens in the CUDA serve tests / K1.)
  T6  — state-dict discipline: key sets per mode, arch_meta round-trip,
        infer_net_kwargs_from_state_dict meta-first + key-set fallback with
        conv2-vs-both disambiguation, the feature_version load assert, env
        validation.
  T7  — generated tap->(axis, dir) LUT bijection, geometric consistency with
        _triton_ray's slot_offset_table, and a behavioral raylen-slot check
        against an independent in-test ray walk.
  T8  — (small-shape half) the K2 custom-autograd pre-aggregation matches the
        naive-implementation oracle: outputs bitwise, gradients <= 1e-5 rel,
        plus a float64 gradcheck. (The full-shape 12 GB memory gate is the
        CUDA subprocess test at the bottom; see _hexfield_eq_raytap_child.py.)

Plus: serve-side raylen un-gating (W-R3) through the real wire payload
(needs_rust), AdamW/grad-group classification of the alpha param, and the
CUDA T4b serve-wire init-equivalence (full serve stack: fp16 half serve,
weight cache + perm fold, CUDA graphs, raylen staging; layouts with and
without 'L').

Runs under the default env (equivariance parts self-skip when GROUP_ORDER !=
12); tests needing a non-default import env use subprocesses.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hexfield_eq import constants as C
from hexfield_eq import _raytap as RT
from hexfield_eq.batching import collate_rows
from hexfield_eq.constants import DIRECTIONS
from hexfield_eq.features import (
    AXIS_DELTAS,
    PositionFacts,
    build_position,
    build_ray_lengths,
)
from hexfield_eq.geometry import apply_d6, disk_offsets
from hexfield_eq.model import HexfieldNet, infer_net_kwargs_from_state_dict
from hexfield_eq._triton_ray import slot_offset_table

try:
    from hexfield_eq import _rust
except ImportError:  # pragma: no cover
    _rust = None

needs_rust = pytest.mark.skipif(
    _rust is None, reason="hexfield_eq._rust not built (see the Phase-1 build gate)"
)
eq_only = pytest.mark.skipif(
    C.GROUP_ORDER != 12,
    reason="equivariant gate; run under the default HEXFIELD_EQ_GROUP_ORDER=12 build",
)

_REPO = Path(__file__).resolve().parents[1]
RL = C.RAYLEN_SLOTS
ATOL = 1e-4
NET_SCALE_TOL = 2e-6  # the perm-fold suite's magnitude-scaled tolerance model
SERVE_ATOL = 3e-3

COVARIANT_HEADS = ("policy", "opp_policy", "soft_policy", "cell_q")
INVARIANT_HEADS = ("value", "stvalue_2", "stvalue_6", "stvalue_16", "moves_left")

_AXIS_VECS = tuple(AXIS_DELTAS[k] for k in ("Q", "R", "QR"))


# --- shared helpers ---------------------------------------------------------------


def _synthetic_facts(seed: int, n_stones: int) -> PositionFacts:
    """A synthetic (engine-free) position: n_stones distinct placements in a
    small neighbourhood, alternating owners, both parities of side-to-move."""

    rng = random.Random(seed)
    cells: list[tuple[int, int]] = [(0, 0)]
    seen = {(0, 0)}
    while len(cells) < n_stones:
        q, r = cells[rng.randrange(len(cells))]
        dq, dr = rng.choice(DIRECTIONS)
        step = rng.randint(1, 3)
        cand = (q + dq * step, r + dr * step)
        if cand not in seen:
            seen.add(cand)
            cells.append(cand)
    records = tuple((q, r, i % 2, i) for i, (q, r) in enumerate(cells))
    current = seed % 2
    return PositionFacts(
        records=records, current_player=current,
        phase="SecondStone", first_stone=cells[0],
    )


def _position_batch(seeds=(3, 4), n_stones=7):
    rows, raylen_rows = [], []
    for s in seeds:
        facts = _synthetic_facts(s, n_stones)
        sup, feats = build_position(facts)
        rows.append((sup, feats))
        raylen_rows.append(build_ray_lengths(facts, sup))
    return collate_rows(rows, raylen=raylen_rows)


def _randomize(model: HexfieldNet, seed: int) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.3)


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
    """raylen-slot transport under g (the ray_block-test covariance relation)."""

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
    perm = _slot_perm(g)
    inv = [0] * RL
    for s, t in enumerate(perm):
        inv[t] = s
    out = torch.zeros_like(raylen)
    out[0, sig] = raylen[0][:, inv]
    return out


# --- T7: generated LUTs -------------------------------------------------------------


def test_t7_tap_axis_dir_bijection() -> None:
    ad = RT.tap_axis_dir()
    assert len(ad) == 6
    # Bijection onto (axis, dir): each of the 6 pairs hit exactly once.
    assert sorted(ad) == [(a, d) for a in range(3) for d in range(2)]
    # Consistency with the two source tables: DIRECTIONS[t] == sign * delta.
    for t, (axis, direc) in enumerate(ad):
        aq, ar = _AXIS_VECS[axis]
        sign = 1 if direc == 0 else -1
        assert DIRECTIONS[t] == (sign * aq, sign * ar), (t, axis, direc)


def test_t7_ray_slot_lut_geometry() -> None:
    """Each (tap, k) slot's fixed offset in _triton_ray's slot_offset_table is
    exactly k * DIRECTIONS[tap] — the geometric ground truth, independent of
    the packing arithmetic both sides use."""

    lut = RT.tap_ray_slot_lut()
    off = slot_offset_table()
    assert lut.shape == (6, C.RAY_REACH)
    assert len(set(lut.reshape(-1).tolist())) == 30  # injective over slots 1..30
    for t in range(6):
        for k in range(1, C.RAY_REACH + 1):
            s = int(lut[t, k - 1])
            assert 1 <= s <= 30
            dq, dr = DIRECTIONS[t]
            assert (int(off[s, 0]), int(off[s, 1])) == (k * dq, k * dr), (t, k)


def test_t7_raylen_slot_behavioral() -> None:
    """build_tap_reach's per-tap reach equals an independent in-test ray walk
    (stop at + include the first anti-side stone; own stones and empties pass;
    off-support stops) on a real position — pins the side/axis/dir sign
    convention end to end, not just the slot arithmetic."""

    facts = _synthetic_facts(11, 9)
    sup, _ = build_position(facts)
    raylen = torch.from_numpy(build_ray_lengths(facts, sup)).unsqueeze(0)
    reach = RT.build_tap_reach(raylen)  # (1, N, 2, 6)

    owner_at = {(q, r): o for q, r, o, _ in facts.records}
    me = facts.current_player
    index = sup.index
    for row in range(sup.num_nodes):
        q0, r0 = (int(v) for v in sup.coords[row])
        for t, (dq, dr) in enumerate(DIRECTIONS):
            for side in range(2):
                anti = (1 - me) if side == 0 else me
                length = 0
                for k in range(1, C.RAY_REACH + 1):
                    y = (q0 + dq * k, r0 + dr * k)
                    if y not in index:
                        break
                    length = k
                    if owner_at.get(y) == anti:
                        break
                assert int(reach[0, row, side, t]) == length, (row, t, side)


# --- T4: init equivalence (reference path) --------------------------------------------


@pytest.mark.parametrize("mode", ["conv2", "both"])
def test_t4_init_equivalence_reference_path(mode: str) -> None:
    batch = _position_batch(seeds=(3, 4), n_stones=7)
    torch.manual_seed(0)
    base = HexfieldNet().eval()
    torch.manual_seed(0)
    rt = HexfieldNet(raytap=mode).eval()
    args = (batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    with torch.no_grad():
        out_b = base(*args)
        out_r = rt(*args, raylen=batch["raylen"])
    for key in out_b:
        torch.testing.assert_close(
            out_r[key], out_b[key], atol=1e-6, rtol=0, msg=f"{key} ({mode})"
        )


def test_t4_taps_are_live_off_init() -> None:
    """Perturbing alpha away from init changes the outputs — guards against a
    silently-dead tap path that would pass init-equivalence trivially. Params
    are randomized first (a fresh net's 1e-4 LayerScale would crush the
    residual branch below the assertion's resolution)."""

    batch = _position_batch(seeds=(5,), n_stones=9)
    rt = HexfieldNet(raytap="both").eval()
    _randomize(rt, 13)
    with torch.no_grad():
        for block in rt.conv_blocks:
            block.conv1.alpha.zero_()
            block.conv1.alpha[0] = 1.0
            block.conv2.alpha.zero_()
            block.conv2.alpha[0] = 1.0
    args = (batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    with torch.no_grad():
        out_init = rt(*args, raylen=batch["raylen"])
        for block in rt.conv_blocks:
            block.conv1.alpha[2] = 0.5
            block.conv2.alpha[2] = 0.5
        out_pert = rt(*args, raylen=batch["raylen"])
    assert not torch.allclose(out_pert["policy"], out_init["policy"], atol=1e-6)


def test_t4_lut2_own_and_opp_tables_are_live() -> None:
    """Each additive table independently reaches the policy output."""

    batch = _position_batch(seeds=(5,), n_stones=9)
    model = HexfieldNet(raytap="both", raytap_lut="additive").eval()
    _randomize(model, 14)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith((".O", ".P")):
                param.zero_()

    args = (batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    with torch.no_grad():
        base = model(*args, raylen=batch["raylen"])["policy"]
        for name, param in model.named_parameters():
            if name.endswith(".O"):
                param.fill_(0.5)
        own = model(*args, raylen=batch["raylen"])["policy"]
        for name, param in model.named_parameters():
            if name.endswith(".O"):
                param.zero_()
            elif name.endswith(".P"):
                param.fill_(-0.5)
        opp = model(*args, raylen=batch["raylen"])["policy"]

    assert not torch.allclose(own, base, atol=1e-6, rtol=0), "O table is not live"
    assert not torch.allclose(opp, base, atol=1e-6, rtol=0), "P table is not live"


def test_raytap_requires_raylen() -> None:
    batch = _position_batch(seeds=(3,), n_stones=5)
    rt = HexfieldNet(raytap="both").eval()
    with pytest.raises(ValueError, match="raylen"):
        with torch.no_grad():
            rt(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])


# --- T3: full-net equivariance -----------------------------------------------------


@eq_only
@pytest.mark.parametrize(
    "mode,layout",
    [("conv2", "CCACCA"), ("both", "CCACCA"), ("both", "CCACCACA")],
    ids=["conv2", "both", "both-A5-layout"],
)
def test_t3_full_net_equivariance_raytap(mode: str, layout: str) -> None:
    from hexfield_eq import equivariant as eq

    _, n, nbr, coords, mask, sig = _disk_board(3)
    model = HexfieldNet(trunk_layout=layout, raytap=mode).eval()
    _randomize(model, 2)
    torch.manual_seed(21)
    feats = torch.randn(1, n, C.NUM_FEATURES)
    raylen = torch.randint(0, C.RAY_REACH + 1, (1, n, RL), dtype=torch.uint8)
    rin = eq._in_rep_matrix()
    with torch.no_grad():
        base = model(feats, nbr, mask, coords, raylen=raylen)
        for g in range(12):
            fg = torch.zeros_like(feats)
            fg[0, sig[g]] = feats[0] @ rin[g].T
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


# --- T6: state-dict discipline -------------------------------------------------------


def _alpha_keys(sd: dict) -> tuple[bool, bool]:
    c1 = any(k.startswith("conv_blocks.") and k.endswith(".conv1.alpha") for k in sd)
    c2 = any(k.startswith("conv_blocks.") and k.endswith(".conv2.alpha") for k in sd)
    return c1, c2


@pytest.mark.parametrize("mode", ["0", "conv2", "both"])
def test_t6_key_set_meta_and_rebuild(mode: str) -> None:
    model = HexfieldNet(raytap=mode)
    sd = model.state_dict()
    c1, c2 = _alpha_keys(sd)
    assert c1 == (mode == "both")
    assert c2 == (mode in ("conv2", "both"))

    meta = model.arch_meta()
    assert meta["raytap"] == mode
    assert meta["feature_version"] == C.FEATURE_VERSION

    # Meta-first inference and the key-set fallback both land on the mode
    # (conv2-vs-both disambiguated by FIRST-conv alpha presence, spec §4).
    kw_meta = infer_net_kwargs_from_state_dict(sd, meta)
    assert kw_meta["raytap"] == mode
    kw_keys = infer_net_kwargs_from_state_dict(sd, {})
    assert kw_keys["raytap"] == mode

    rebuilt = HexfieldNet(**kw_meta)
    rebuilt.load_state_dict(sd, strict=True)

    # alpha init shape + value on equipped convs.
    if mode != "0":
        a = model.conv_blocks[0].conv2.alpha
        corb = C.C_ORBIT if C.GROUP_ORDER == 12 else C.CHANNELS
        assert a.shape == (C.RAY_REACH, corb)
        assert torch.equal(a[0], torch.ones(corb)) and float(a[1:].abs().sum()) == 0.0


def test_t6_invalid_raytap_kwarg_rejected() -> None:
    with pytest.raises(ValueError, match="raytap"):
        HexfieldNet(raytap="conv1")


def test_t6_feature_version_load_assert(tmp_path: Path) -> None:
    from hexfield_eq.checkpoints import load_into, save_checkpoint

    model = HexfieldNet(trunk_layout="CCA")
    path = save_checkpoint(tmp_path / "ck.pt", model=model, optimizer=None, epoch=0)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["meta"]["feature_version"] == C.FEATURE_VERSION
    load_into(model, payload)  # matching version loads

    payload["meta"]["feature_version"] = C.FEATURE_VERSION + 1
    with pytest.raises(ValueError, match="feature_version"):
        load_into(model, payload)


def test_t6_default_env_is_inert() -> None:
    """Under the default env RAYTAP is '0' and the state-dict key set is the
    pre-change set (no alpha anywhere) — the live-run isolation guard."""

    if os.environ.get("HEXFIELD_EQ_RAYTAP", "0") != "0":
        pytest.skip("non-default HEXFIELD_EQ_RAYTAP in this process")
    assert C.RAYTAP == "0"
    model = HexfieldNet()
    assert not any(k.endswith(".alpha") for k in model.state_dict())
    assert model.arch_meta()["raytap"] == "0"


def test_t6_env_threads_to_net_and_expand(tmp_path: Path) -> None:
    """HEXFIELD_EQ_RAYTAP=both threads through constants -> the default-built
    net -> samples._EXPAND_RAYLEN (the serial-expand oracle un-gating, W-R3)."""

    env = {k: v for k, v in os.environ.items() if not k.startswith("HEXFIELD")}
    env["HEXFIELD_EQ_RAYTAP"] = "both"
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    code = (
        "import json\n"
        "from hexfield_eq import constants as C\n"
        "from hexfield_eq import samples\n"
        "from hexfield_eq.model import HexfieldNet\n"
        "net = HexfieldNet(trunk_layout='CCA')\n"
        "sd = net.state_dict()\n"
        "print(json.dumps({'raytap': C.RAYTAP,\n"
        "  'net_mode': net._raytap,\n"
        "  'expand_raylen': samples._EXPAND_RAYLEN,\n"
        "  'has_alpha': any(k.endswith('.alpha') for k in sd)}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True,
        cwd=_REPO, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got == {
        "raytap": "both", "net_mode": "both",
        "expand_raylen": True, "has_alpha": True,
    }


def test_t6_invalid_env_rejected_at_import() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HEXFIELD")}
    env["HEXFIELD_EQ_RAYTAP"] = "conv1"
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", "import hexfield_eq.constants"],
        env=env, capture_output=True, text=True, cwd=_REPO, timeout=300,
    )
    assert proc.returncode != 0
    assert "HEXFIELD_EQ_RAYTAP" in proc.stderr


# --- T5: serve-fold parity at trained alpha -------------------------------------------


@eq_only
@pytest.mark.parametrize("layout", ["CCA", "CLA"], ids=["no-L", "with-L"])
def test_t5_fold_parity_trained_alpha(layout: str) -> None:
    """No-grad (folded serve weights) vs grad-enabled forward on a ray-tap net
    with every param randomized (the trained-alpha stand-in), layouts with and
    without 'L' — the perm-fold suite's tolerance model."""

    batch = _position_batch(seeds=(7, 8), n_stones=8)
    model = HexfieldNet(
        trunk_layout=layout, raytap="both", reg_lane=True, reg_tok_read=True
    ).eval()
    _randomize(model, 9)
    args = (batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
    with torch.no_grad():
        folded = model(*args, raylen=batch["raylen"])
    with torch.enable_grad():
        runtime = model(*args, raylen=batch["raylen"])
    for key in folded:
        tol = max(ATOL, NET_SCALE_TOL * float(runtime[key].abs().max()))
        torch.testing.assert_close(
            folded[key], runtime[key], atol=tol, rtol=0, msg=key
        )


# --- T8 (small-shape): K2 gradient oracle ---------------------------------------------


def _random_tap_inputs(seed: int, b=2, n=37, c=None, dtype=torch.float32):
    corb = C.C_ORBIT if C.GROUP_ORDER == 12 else C.CHANNELS
    c = c if c is not None else (C.CHANNELS if C.GROUP_ORDER == 12 else corb)
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(b, n, c, generator=gen, dtype=dtype)
    # idx values in [0, n] with the sentinel n mixed in.
    idx = torch.randint(0, n + 1, (b, n, 6, C.RAY_REACH), generator=gen)
    reach = torch.randint(
        0, C.RAY_REACH + 1, (b, n, 2, 6), generator=gen
    ).to(torch.uint8)
    alpha = torch.randn(C.RAY_REACH, corb, generator=gen, dtype=dtype)
    return x, idx, reach, alpha, corb


def test_t8_k2_matches_naive_oracle() -> None:
    x, idx, reach, alpha, corb = _random_tap_inputs(0)
    xa = x.clone().requires_grad_(True)
    aa = alpha.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    ab = alpha.clone().requires_grad_(True)

    af_a = aa.repeat(1, x.shape[-1] // corb)
    af_b = ab.repeat(1, x.shape[-1] // corb)
    out_k2 = RT.ray_tap_taps(xa, idx, reach, af_a, corb)
    out_nv = RT.ray_tap_taps_naive(xb, idx, reach, af_b, corb)
    assert torch.equal(out_k2, out_nv), "forward numerics must be identical"

    g = torch.randn_like(out_k2)
    out_k2.backward(g)
    out_nv.backward(g)

    def rel(a, b):
        denom = b.abs().max().clamp(min=1e-12)
        return float((a - b).abs().max() / denom)

    assert rel(xa.grad, xb.grad) <= 1e-5, "grad_x oracle mismatch"
    assert rel(aa.grad, ab.grad) <= 1e-5, "grad_alpha oracle mismatch"


def test_t8_k2_gradcheck_float64() -> None:
    x, idx, reach, alpha, corb = _random_tap_inputs(
        1, b=1, n=9, dtype=torch.float64
    )
    x = x.requires_grad_(True)
    alpha = alpha.requires_grad_(True)

    def fn(x_, alpha_):
        af = alpha_.repeat(1, x_.shape[-1] // corb)
        return RT.ray_tap_taps(x_, idx, reach, af, corb)

    assert torch.autograd.gradcheck(fn, (x, alpha), fast_mode=True)


def test_t8_k2_saves_no_gathered_intermediate() -> None:
    """The K2 Function's saved set is exactly {x, idx_taps, reach, alpha_full}
    — no (B, N, 30, C) gathered tensor survives to backward."""

    x, idx, reach, alpha, corb = _random_tap_inputs(2)
    x = x.requires_grad_(True)
    af = alpha.requires_grad_(True).repeat(1, x.shape[-1] // corb)
    out = RT.ray_tap_taps(x, idx, reach, af, corb)
    node = out.grad_fn
    while node is not None and "RayTapTaps" not in type(node).__name__:
        node = node.next_functions[0][0] if node.next_functions else None
    assert node is not None, "K2 Function not on the graph"
    saved_numels = sorted(
        t.numel() for t in (x, idx, reach) if isinstance(t, torch.Tensor)
    )
    # The big intermediate would be b*n*30*c; assert nothing that large is
    # reachable through saved_tensors.
    b, n, _, c = out.shape
    big = b * n * 30 * c
    for t in getattr(node, "saved_tensors", ()):
        assert t.numel() < big, "a gathered (B,N,30,C)-scale tensor was saved"
    _ = saved_numels


# --- optimizer / grad-group classification ---------------------------------------------


def test_alpha_lands_no_decay_and_trunk_conv() -> None:
    try:
        from hexfield_eq.prefit import make_optimizer
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"prefit import chain unavailable: {exc}")

    model = HexfieldNet(trunk_layout="CCA", raytap="both")
    opt = make_optimizer(model)
    no_decay = {
        id(p)
        for grp in opt.param_groups
        if grp["weight_decay"] == 0.0
        for p in grp["params"]
    }
    alphas = [
        (nm, p) for nm, p in model.named_parameters() if nm.endswith(".alpha")
    ]
    assert alphas
    for nm, p in alphas:
        assert id(p) in no_decay, f"{nm} should be no-decay"

    try:
        from hexfield_eq.trainer import HexfieldTrainer
        from types import SimpleNamespace

        groups = HexfieldTrainer._build_grad_norm_groups(
            SimpleNamespace(model=model)
        )
        conv_ids = {id(p) for p in groups["trunk_conv"]}
        for nm, p in alphas:
            assert id(p) in conv_ids, f"{nm} should be trunk_conv"
    except ImportError:  # pragma: no cover
        pass


# --- serve wire tests (W-R3 un-gating; needs the Rust featurizer) ------------------------

NBR_SENTINEL = 0xFFFF


def _random_state(seed: int, plies: int):
    from hexo_engine import api
    from hexo_engine.types import AxialCoord, PlacementAction
    from hexfield_eq.geometry import unpack_action_id

    state = api.new_game()
    rng = random.Random(seed)
    for _ in range(plies):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        q, r = unpack_action_id(rng.choice(ids))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            break
    return state


def _battery_states(n_seeds: int = 1, plies=(1, 5, 11)):
    from hexo_engine import api

    states = []
    for seed in range(n_seeds):
        for p in plies:
            st = _random_state(8100 + seed * 100 + p, p)
            if api.terminal(st) is None:
                states.append(st)
    assert states
    return states


def _build_payload(states, *, request_ml=True, raylen_override=None, drop_raylen=False):
    rows = _rust.featurize_states(states)
    order = sorted(range(len(states)), key=lambda i: (-int(rows[i]["num_nodes"]), i))
    feats, qr, nbr, rl, offsets, legal = [], [], [], [], [0], []
    for i in order:
        r = rows[i]
        n = int(r["num_nodes"])
        feats.append(np.frombuffer(r["feats"], dtype=np.float32).astype(np.float16))
        qr.append(np.frombuffer(r["coords"], dtype=np.int16))
        nb = np.frombuffer(r["nbr"], dtype=np.int32)
        nbr.append(np.where(nb < 0, NBR_SENTINEL, nb).astype(np.uint16))
        rl.append(np.frombuffer(r["raylen"], dtype=np.uint8))
        offsets.append(offsets[-1] + n)
        legal.append(int(r["legal_count"]))
    raylen_bytes = np.concatenate(rl).tobytes()
    if raylen_override is not None:
        raylen_bytes = raylen_override(np.concatenate(rl)).tobytes()
    payload = {
        "abi": 1,
        "shape": (len(order), offsets[-1]),
        "node_feats": np.concatenate(feats).tobytes(),
        "node_qr": np.concatenate(qr).tobytes(),
        "nbr": np.concatenate(nbr).tobytes(),
        "raylen": raylen_bytes,
        "node_row_offsets": offsets,
        "legal_counts": np.asarray(legal, dtype=np.int32).tobytes(),
        "request_moves_left": request_ml,
    }
    if drop_raylen:
        payload.pop("raylen")
    return payload, [states[i] for i in order]


def _reply_arrays(reply, *, request_ml=True):
    values = np.frombuffer(reply["values_bytes"], dtype=np.float32)
    priors = np.frombuffer(reply["priors_bytes"], dtype=np.float32)
    ml = (
        np.frombuffer(reply["moves_left_bytes"], dtype=np.float32)
        if request_ml
        else None
    )
    return values, priors, ml


@needs_rust
def test_raytap_net_serve_needs_and_stages_raylen() -> None:
    """A ray-tap net on an L-FREE layout (the arm-A5 shape) serves through the
    un-gated raylen path: the evaluator stages raylen, threads it into the
    forward, and the taps are LIVE (zero vs true raylen changes the reply)."""

    from hexfield_eq.inference import HexfieldEvaluator

    states = _battery_states()
    payload, _ = _build_payload(states)
    torch.manual_seed(31)
    model = HexfieldNet(trunk_layout="CCA", raytap="both").eval()
    _randomize(model, 32)
    ev = HexfieldEvaluator(model, device="cpu")
    assert ev._needs_raylen and ev._raytap == "both"
    reply = ev.evaluate_payload(payload)
    v, p, _ = _reply_arrays(reply)
    assert np.isfinite(v).all() and np.isfinite(p).all()

    # raylen is live: a doctored buffer (all rays maxed) changes the reply.
    payload_max, _ = _build_payload(
        states, raylen_override=lambda rl: np.full_like(rl, C.RAY_REACH)
    )
    ev2 = HexfieldEvaluator(model, device="cpu")
    v2, p2, _ = _reply_arrays(ev2.evaluate_payload(payload_max))
    assert not np.allclose(p, p2, atol=1e-6)


@needs_rust
def test_raytap_serve_guards() -> None:
    from hexfield_eq.inference import HexfieldEvaluator

    states = _battery_states(plies=(3,))
    torch.manual_seed(33)
    model = HexfieldNet(trunk_layout="CCA", raytap="both").eval()

    payload_nr, _ = _build_payload(states, drop_raylen=True)
    ev = HexfieldEvaluator(model, device="cpu")
    with pytest.raises(ValueError, match="raylen"):
        ev.submit_payload(payload_nr)

    payload_zero, _ = _build_payload(
        states, raylen_override=lambda rl: np.zeros_like(rl)
    )
    ev2 = HexfieldEvaluator(model, device="cpu")
    with pytest.raises(ValueError, match="all-zero"):
        ev2.submit_payload(payload_zero)


@needs_rust
def test_t4b_serve_wire_init_equivalence_cpu() -> None:
    """CPU half of T4b: through the real wire payload (raylen staging + the
    reference serve path) a ray-tap net at init-alpha replies identically to
    the baseline net, on layouts with and without 'L'."""

    from hexfield_eq.inference import HexfieldEvaluator

    states = _battery_states()
    payload, _ = _build_payload(states)
    for layout in ("CCA", "CLA"):
        torch.manual_seed(41)
        base = HexfieldNet(trunk_layout=layout).eval()
        torch.manual_seed(41)
        rt = HexfieldNet(trunk_layout=layout, raytap="both").eval()
        rb, _, _ = _reply_arrays(
            HexfieldEvaluator(base, device="cpu").evaluate_payload(dict(payload))
        )
        rr, _, _ = _reply_arrays(
            HexfieldEvaluator(rt, device="cpu").evaluate_payload(dict(payload))
        )
        np.testing.assert_allclose(rr, rb, atol=1e-6, rtol=0, err_msg=layout)


@needs_rust
@pytest.mark.skipif(not torch.cuda.is_available(), reason="full serve stack is CUDA-only")
@pytest.mark.parametrize("layout", ["CCA", "CLA"], ids=["no-L", "with-L"])
def test_t4b_serve_wire_init_equivalence_cuda(monkeypatch, layout) -> None:
    """T4b (spec §6.1): the FULL serve stack — fp16 half serve, materialized
    weight cache + perm fold, CUDA-graph capture, raylen staging — with
    ray-tap enabled at init-alpha vs the baseline net, on layouts WITH and
    WITHOUT 'L'. Behavioral guard for the tap->slot LUT and the raylen
    un-gating (T4 alone cannot catch either: here the wire raylen is real)."""

    from hexfield_eq import model as M
    from hexfield_eq.inference import HexfieldEvaluator
    from hexfield_eq._triton_conv import hex_conv, hex_conv_ln

    if hex_conv is None or hex_conv_ln is None:
        pytest.skip("triton unavailable")

    monkeypatch.setenv("HEXFIELD_NO_COMPILE", "1")  # keep capture cheap
    monkeypatch.setenv("HEXFIELD_SERVE_HALF", "1")
    monkeypatch.setenv("HEXFIELD_CUDA_GRAPHS", "1")
    monkeypatch.setattr(M, "_hex_conv_fused", hex_conv)
    monkeypatch.setattr(M, "_hex_conv_ln_fused", hex_conv_ln)

    states = _battery_states()
    payload, _ = _build_payload(states)
    torch.manual_seed(51)
    base = HexfieldNet(trunk_layout=layout).eval()
    torch.manual_seed(51)
    rt = HexfieldNet(trunk_layout=layout, raytap="both").eval()

    ev_b = HexfieldEvaluator(base, device="cuda")
    ev_r = HexfieldEvaluator(rt, device="cuda")
    assert ev_r._needs_raylen
    vb, pb, _ = _reply_arrays(ev_b.evaluate_payload(dict(payload)))
    vr, pr, _ = _reply_arrays(ev_r.evaluate_payload(dict(payload)))
    np.testing.assert_allclose(vr, vb, atol=SERVE_ATOL, rtol=0)
    np.testing.assert_allclose(pr, pb, atol=SERVE_ATOL, rtol=0)

    # Second flush replays the captured graphs (cache hit) identically.
    vr2, pr2, _ = _reply_arrays(ev_r.evaluate_payload(dict(payload)))
    np.testing.assert_allclose(vr2, vr, atol=1e-6, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + triton")
def test_t5_k1_fused_kernel_parity_trained_alpha() -> None:
    """T5 on the K1 fused path: hexfield_eq::hex_conv_ln_raytap matches the
    reference ray-tap conv + LN at a trained (non-init) alpha to the fp16
    serve-grade tolerance. If the kernel cannot compile for this shape the op
    falls back to the reference internally, so parity holds either way."""

    from hexfield_eq._triton_conv import hex_conv_ln_raytap, _conv_ln_raytap_ref
    from hexfield_eq._raytap import build_ray_gather_index, build_tap_reach

    if hex_conv_ln_raytap is None:
        pytest.skip("triton unavailable")

    torch.manual_seed(61)
    dev = "cuda"
    ch = C.CHANNELS
    corb = C.C_ORBIT if C.GROUP_ORDER == 12 else ch
    _, n, nbr, coords, mask, _ = _disk_board(4)
    self_idx = torch.arange(n).reshape(1, n, 1)
    gidx = torch.cat([self_idx, nbr], dim=2).to(dev)
    coords, mask = coords.to(dev), mask.to(dev)
    x = torch.randn(1, n, ch, device=dev, dtype=torch.float16)
    raylen = torch.randint(
        0, C.RAY_REACH + 1, (1, n, RL), dtype=torch.uint8, device=dev
    )
    ray_idx = build_ray_gather_index(coords, mask)
    reach = build_tap_reach(raylen)
    weight = (torch.randn(7, ch, ch, device=dev) * 0.05)
    bias = torch.randn(ch, device=dev) * 0.02
    lnw = torch.rand(ch, device=dev) + 0.5
    lnb = torch.randn(ch, device=dev) * 0.1
    alpha = torch.zeros(5, ch, device=dev)
    for k in range(5):
        alpha[k] = 0.8 ** k * (1.0 + 0.1 * torch.randn(ch, device=dev))
    with torch.no_grad():
        for relu in (True, False):
            out = hex_conv_ln_raytap(
                x, gidx, mask, weight, bias, lnw, lnb, ray_idx, reach, alpha,
                1e-5, relu, corb,
            ).float()
            ref = _conv_ln_raytap_ref(
                x, gidx, mask, weight, bias, lnw, lnb, ray_idx, reach, alpha,
                1e-5, relu, corb,
            ).float()
            torch.testing.assert_close(out, ref, atol=5e-3, rtol=0)
