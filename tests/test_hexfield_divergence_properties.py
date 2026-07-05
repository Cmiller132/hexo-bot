"""Property tests for the search divergences.

- LCB pick vs closed-form on synthetic visit/Q/sigma tables, including the
  eligibility fallback (None when no child qualifies).
- Moves-left utility: zero at/below the |Q| gate, correct sign
  (shorter-when-winning preferred), monotone in (M_e - M_node), bounded by w.
- Early-stop with LCB selection off compared against early-stop off: chosen-move
  equality over greedy searches, and a check that the stop fires at least once.
- Reused-root policy temperature is applied once per root lifetime.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from hexfield_testkit import api, sample_decision_states
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield.geometry import unpack_action_id
from test_hexfield_search_parity import HexfieldStub, _corpus

try:
    from hexfield import _rust as hexfield_rust
except ImportError:  # pragma: no cover
    hexfield_rust = None

needs_native = pytest.mark.skipif(hexfield_rust is None, reason="native module not built")


@needs_native
def test_lcb_matches_closed_form_tables() -> None:
    rng = random.Random(5)
    for _ in range(300):
        n_edges = rng.randint(1, 12)
        stats = []
        for i in range(n_edges):
            visits = rng.randint(0, 200)
            delta = rng.randint(0, visits) if visits else 0
            q = rng.uniform(-1, 1)
            value_sum = q * visits
            # sigma^2 in [0, 0.5]
            var = rng.uniform(0, 0.5)
            value_sq_sum = (var + q * q) * visits
            stats.append((1000 + i, delta, visits, value_sum, value_sq_sum))
        z, minv, frac = 1.6, 8, 0.1
        got = hexfield_rust.debug_lcb_pick(stats, z, minv, frac)

        max_delta = max((s[1] for s in stats), default=0)
        if max_delta == 0:
            assert got is None
            continue
        threshold = max(minv, frac * max_delta)
        best = None
        for aid, delta, visits, vsum, vsq in stats:
            if delta < threshold or visits == 0:
                continue
            q = np.float32(vsum) / np.float32(visits)
            var = max(np.float32(vsq) / np.float32(visits) - q * q, np.float32(0.0))
            lcb = float(q - np.float32(z) * math.sqrt(var) / math.sqrt(visits))
            if best is None or lcb > best[0] + 1e-9 or (abs(lcb - best[0]) <= 1e-9 and aid < best[1]):
                best = (lcb, aid)
        expected = best[1] if best else None
        assert got == expected, (stats, got, expected)


@needs_native
def test_lcb_eligibility_fallback() -> None:
    # All children below the visit threshold -> None.
    stats = [(1, 3, 3, 1.5, 1.0), (2, 2, 2, -0.5, 0.5)]
    assert hexfield_rust.debug_lcb_pick(stats, 1.6, 8, 0.1) is None


@needs_native
def test_ml_bonus_properties() -> None:
    w, scale, gate = 0.03, 32.0, 0.6
    bonus = lambda q, me, mn: hexfield_rust.debug_ml_bonus(q, me, mn, w, scale, gate)
    # Zero in the dead zone (|q| <= gate) and, one-sided, for q < -gate.
    for q in (-1.0, -0.7, 0.0, 0.3, 0.6):
        assert bonus(q, 10.0, 50.0) == 0.0
        assert bonus(q, 50.0, 10.0) == 0.0
    # Winning + child predicts fewer moves left -> positive bonus.
    assert bonus(0.9, 20.0, 40.0) > 0.0
    # Winning + child predicts more moves left -> negative bonus.
    assert bonus(0.9, 60.0, 40.0) < 0.0
    # Monotone decreasing in (M_e - M_node); bounded by w.
    last = None
    for me in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0):
        b = bonus(0.9, me, 35.0)
        assert abs(b) <= w + 1e-9
        if last is not None:
            assert b < last
        last = b
    # Invariant to a shared additive offset on both moves-left values.
    assert bonus(0.9, 20.0, 40.0) == pytest.approx(bonus(0.9, 120.0, 140.0), abs=1e-7)


@needs_native
def test_ml_bonus_two_sided() -> None:
    w, scale, gate = 0.03, 32.0, 0.6
    one = lambda q, me, mn: hexfield_rust.debug_ml_bonus(q, me, mn, w, scale, gate, False)
    two = lambda q, me, mn: hexfield_rust.debug_ml_bonus(q, me, mn, w, scale, gate, True)
    # Zero for |q| <= gate under both modes.
    for q in (-0.6, -0.3, 0.0, 0.3, 0.6):
        assert one(q, 10.0, 50.0) == 0.0
        assert two(q, 10.0, 50.0) == 0.0
    # One-sided: the losing side (q < -gate) yields zero.
    assert one(-0.9, 60.0, 40.0) == 0.0
    assert one(-0.9, 20.0, 40.0) == 0.0
    # Two-sided losing side: more moves left -> positive bonus,
    # fewer moves left -> negative bonus.
    assert two(-0.9, 60.0, 40.0) > 0.0
    assert two(-0.9, 20.0, 40.0) < 0.0
    # Symmetric with the winning side at equal |q| and mirrored delta.
    assert two(-0.9, 60.0, 40.0) == pytest.approx(two(0.9, 20.0, 40.0), abs=1e-7)
    # Bounded by w on the losing side.
    assert abs(two(-0.95, 500.0, 0.0)) <= w + 1e-9


def test_build_divergence_overrides() -> None:
    from hexfield.config import SelfplayConfig, build_divergence_overrides

    sp = SelfplayConfig()  # default config: moves-left on, two-sided, final-pick
    on = build_divergence_overrides(sp)
    assert on["moves_left_utility"] is True
    assert on["ml_two_sided"] is True
    assert on["ml_final_pick"] is True
    assert on["ml_weight"] == pytest.approx(0.03)
    assert on["ml_scale"] == pytest.approx(32.0)
    assert on["ml_q_gate"] == pytest.approx(0.6)
    # disabled=True sets the enable flags off while leaving the constants unchanged.
    off = build_divergence_overrides(sp, disabled=True)
    assert off["moves_left_utility"] is False
    assert off["ml_two_sided"] is False
    assert off["ml_final_pick"] is False
    assert off["ml_weight"] == pytest.approx(0.03)
    # Every override value is a concrete bool/float/int, never None.
    # resolve_divergences extracts each into bool/f32/u32; u32 knobs such as
    # gumbel_m and gumbel_target_min_visits are emitted as concrete int.
    for key, value in on.items():
        assert value is not None, key
        assert isinstance(value, (bool, float, int)), (key, type(value))
    # gumbel_draw_temperature is always emitted (default 1.0 = today's draw); the
    # export-only gumbel_target_c_scale is OMITTED while unset (its Rust default
    # keeps the target on gumbel_c_scale).
    assert on["gumbel_draw_temperature"] == pytest.approx(1.0)
    assert "gumbel_target_c_scale" not in on

    # When set, both fields flow into the overrides dict as concrete floats.
    from dataclasses import replace

    sp2 = replace(sp, gumbel_target_c_scale=0.35, gumbel_draw_temperature=4.0)
    on2 = build_divergence_overrides(sp2)
    assert on2["gumbel_target_c_scale"] == pytest.approx(0.35)
    assert on2["gumbel_draw_temperature"] == pytest.approx(4.0)
    assert isinstance(on2["gumbel_target_c_scale"], float)


def test_gumbel_target_c_scale_and_draw_temperature_toml_round_trip() -> None:
    """Both new selfplay levers parse from a toml-shaped mapping and survive the
    round-trip: the export-only gumbel_target_c_scale (float | None, default None)
    and gumbel_draw_temperature (float, default 1.0)."""
    from hexfield.config import parse_hexfield_config

    # Defaults when absent: target override unset, draw temperature 1.0 (off).
    default_sp = parse_hexfield_config({}).selfplay
    assert default_sp.gumbel_target_c_scale is None
    assert default_sp.gumbel_draw_temperature == pytest.approx(1.0)

    # Explicit values round-trip through the [selfplay] section verbatim.
    cfg = {
        "selfplay": {
            "gumbel_target_c_scale": 0.35,
            "gumbel_draw_temperature": 4.0,
        }
    }
    sp = parse_hexfield_config(cfg).selfplay
    assert sp.gumbel_target_c_scale == pytest.approx(0.35)
    assert sp.gumbel_draw_temperature == pytest.approx(4.0)


@needs_native
def test_early_stop_without_lcb_is_exact() -> None:
    """Visit-overtake early-stop compared against early-stop off, with the raw
    greedy argmax and tss_enabled=False. Asserts the chosen move matches and the
    early stop fires at least once across the corpus."""

    states = _corpus(60)
    on = hexfield_rust.HexfieldMctsSession(max_states=65536)
    off = hexfield_rust.HexfieldMctsSession(max_states=65536)
    overrides_on = {
        "lcb_move_selection": False,
        "early_stop": True,
        "moves_left_utility": False,
        # Set the remaining divergences to the search_parity_mode values so this
        # test isolates early_stop against the parity session below.
        "nucleus_f64": False,
        "new_child_fpu": False,
        "lazy_widening": False,
        "clean_root_prior_cache": False,
    }
    stops = 0
    for index, state in enumerate(states):
        kwargs = dict(
            visits=192, c_puct=1.5, temperature=0.0, seed=7 + index,
            evaluator=HexfieldStub(), virtual_batch_size=8,
            widening_policy_mass=0.95, widening_max_children=96,
            widening_min_children=2, fpu_reduction=0.2, tss_enabled=False,
        )
        a = on.search([index], (state,), divergence_overrides=overrides_on, **kwargs)[0]
        b = off.search([index], (state,), search_parity_mode=True, **kwargs)[0]
        assert a["action_id"] == b["action_id"], index
        stops += int(a["early_stopped"])
        on.discard(index)
        off.discard(index)
    assert stops > 0, "early-stop never fired"


@needs_native
def test_reused_root_temperature_applied_once() -> None:
    """A promoted (reused) root applies the root-policy temperature once: its
    exported prior equals normalize(normalize(raw_eval_prior) ** (1/T))."""

    state = api.new_game()
    for q, r in [(0, 0), (1, 1), (-1, 0)]:
        api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    stub = HexfieldStub()
    session = hexfield_rust.HexfieldMctsSession(max_states=65536)
    temp = 1.5
    kwargs = dict(
        visits=24, c_puct=1.5, temperature=0.0, evaluator=stub,
        virtual_batch_size=8, widening_policy_mass=0.95,
        widening_max_children=96, widening_min_children=2,
        fpu_reduction=0.2, tss_enabled=False, search_parity_mode=True,
        root_policy_temperature=temp,
    )
    r0 = session.search([7], (state,), seed=1, **kwargs)[0]
    q, rr = unpack_action_id(int(r0["action_id"]))
    api.apply_action(state, PlacementAction(AxialCoord(q=q, r=rr)))

    r1 = session.search([7], (state,), seed=2, **kwargs)[0]
    ids = np.frombuffer(bytes(r1["root_prior_policy_action_ids_bytes"]), dtype=np.uint32)
    weights = np.frombuffer(bytes(r1["root_prior_policy_weights_bytes"]), dtype=np.float32)

    # Expected: the stub's raw priors for this state, normalized, then ^(1/T),
    # renormalized — computed via the Rust featurizer path.
    payload_rows = hexfield_rust.featurize_states([state])
    row = payload_rows[0]
    # Build the evaluator payload from the featurizer output.
    feats_f16 = (
        np.frombuffer(row["feats"], dtype=np.float32).astype(np.float16).tobytes()
    )  # featurize_states emits f32; the evaluator wire carries f16
    fake_payload = {
        "shape": (1, row["num_nodes"]),
        "legal_counts": np.asarray([row["legal_count"]], dtype=np.int32).tobytes(),
        "node_row_offsets": [0, row["num_nodes"]],
        "node_qr": row["coords"],
        "node_feats": feats_f16,
        "abi": 1,
    }
    reply = stub(fake_payload)
    raw = np.frombuffer(reply["priors_bytes"], dtype=np.float32).astype(np.float64)
    coords = np.frombuffer(row["coords"], dtype=np.int16).reshape(-1, 2)
    legal_ids = np.asarray(
        [((int(q) + (1 << 15)) << 16) | (int(r) + (1 << 15)) for q, r in coords[: row["legal_count"]]],
        dtype=np.uint64,
    )
    norm = raw / raw.sum()
    softened = np.where(norm > 0, norm ** (1.0 / temp), 0.0)
    softened = softened / softened.sum()

    expected = dict(zip(legal_ids.astype(np.uint32).tolist(), softened.tolist()))
    got = dict(zip(ids.tolist(), weights.astype(np.float64).tolist()))
    # zero-prior cells are dropped from the export on both sides
    expected = {k: v for k, v in expected.items() if v > 0}
    assert set(got) == set(expected)
    for aid, w in got.items():
        assert w == pytest.approx(expected[aid], rel=2e-3), aid