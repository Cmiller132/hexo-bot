"""Continuous-scheduler self-parity for the hexfield session, plus an
end-to-end ABI round-trip through the real model.

Originally a differential harness against the (now retired) dense_cnn session;
restructured to a hexfield-vs-hexfield parity net on the public native module.
It drives ``run_continuous`` over a fixed set of games with the full per-move
machinery (PCR full/fast coin, policy-init openings, temperature schedules, TSS)
and pins:

- determinism: two runs with the same base_seed produce identical on_move
  streams (move classes, chosen actions, visit counts, visit-policy bytes) and
  identical scheduler stats;
- ``search_parity_mode=True`` equals an explicit all-divergences-off override;
- an end-to-end payload -> HexfieldNet -> reply -> tree round-trip completes with
  divergences ON.
"""

from __future__ import annotations

import pytest

from hexfield_testkit import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield.geometry import unpack_action_id
from test_hexfield_search_parity import HexfieldStub

try:
    from hexfield import _rust as hexfield_rust
except ImportError:  # pragma: no cover
    hexfield_rust = None

needs_native = pytest.mark.skipif(
    hexfield_rust is None,
    reason="hexfield native module not built",
)

# Ply cap that keeps every searched state inside dense's crop so both sides share
# an identical legal move vocabulary. The Recorder asserts the in-crop constraint
# at every decision state.
MAX_PLIES = 6


class Recorder:
    """on_move driver: applies actions to its own engine states and records
    the full decision stream."""

    def __init__(self):
        self.states = {}
        self.records = []

    def add_game(self, key: int):
        self.states[key] = api.new_game()

    def __call__(self, game_key: int, payload: dict):
        from test_hexfield_search_parity import _fully_in_crop

        state = self.states[game_key]
        # Deep-leaf out-of-disk legals get a zero prior from the stub, matching
        # dense; every decision state must stay fully in-crop so the two sides
        # share an identical legal move vocabulary.
        assert _fully_in_crop(state, margin=0), (
            "decision-state legal set left dense's crop — lower MAX_PLIES "
            "to keep the corpus in-crop"
        )
        action_id = payload["action_id"]
        self.records.append(
            (
                game_key,
                action_id,
                bool(payload["pcr_full"]),
                bool(payload["policy_init"]),
                int(payload["visits"]),
                bytes(payload["visit_policy_action_ids_bytes"]),
                bytes(payload["visit_policy_weights_bytes"]),
                round(float(payload["root_value"]), 6),
            )
        )
        q, r = unpack_action_id(action_id)
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        ply = sum(1 for rec in self.records if rec[0] == game_key)
        if result.terminal or ply >= MAX_PLIES:
            return None
        return ("advance", state)


def _continuous_kwargs():
    return dict(
        visits=48,
        c_puct=1.5,
        base_seed=20260613,
        virtual_batch_size=8,
        flush_target=24,
        active_root_limit=16,
        temperature_by_ply=[1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
        root_policy_temperature=1.1,
        root_policy_temperature_early=1.4,
        root_policy_temperature_halflife=12.0,
        fpu_reduction=0.2,
        virtual_loss=1.0,
        widening_policy_mass=0.95,
        widening_max_children=96,
        widening_min_children=2,
        pcr_full_proportion=0.33,
        pcr_fast_visits=16,
        policy_init_fraction=0.25,
        policy_init_avg_plies=4.0,
        policy_init_max_plies=8,
        policy_init_temperature=1.4,
        tss_enabled=True,
    )


def _run_continuous(keys, **overrides):
    """Drive run_continuous over `keys` with the standard kwargs (plus any
    overrides) and return (stats, recorded on_move stream)."""

    recorder = Recorder()
    for key in keys:
        recorder.add_game(key)
    session = hexfield_rust.HexfieldMctsSession(max_states=65536)
    kwargs = {**_continuous_kwargs(), **overrides}
    stats = session.run_continuous(
        keys,
        tuple(recorder.states[k] for k in keys),
        evaluator=HexfieldStub(),
        on_move=recorder,
        **kwargs,
    )
    return stats, recorder.records


@needs_native
def test_continuous_is_deterministic_full_pipeline() -> None:
    """Two independent run_continuous passes over the same games with the same
    base_seed and full per-move machinery produce identical scheduler stats and
    identical on_move streams — the reproducibility contract the differential
    harness relied on for both engines."""

    keys = list(range(700, 706))
    stats_a, records_a = _run_continuous(keys, search_parity_mode=True)
    stats_b, records_b = _run_continuous(keys, search_parity_mode=True)

    for field in ("moves_decided", "full_moves", "fast_moves", "init_moves"):
        assert stats_a[field] == stats_b[field], field
    assert stats_a["moves_decided"] > 0  # the pipeline actually decided moves

    assert len(records_a) == len(records_b)
    mismatches = [
        (i, a, b) for i, (a, b) in enumerate(zip(records_a, records_b)) if a != b
    ]
    assert not mismatches, f"{len(mismatches)} record mismatches; first: {mismatches[:3]}"


@needs_native
def test_divergences_off_equals_parity_mode() -> None:
    """All divergences off via overrides produces the same records as
    search_parity_mode."""

    keys = [900, 901, 902]
    rec_a = Recorder()
    rec_b = Recorder()
    for key in keys:
        rec_a.add_game(key)
        rec_b.add_game(key)
    kwargs = _continuous_kwargs()

    session_a = hexfield_rust.HexfieldMctsSession(max_states=65536)
    session_a.run_continuous(
        keys, tuple(rec_a.states[k] for k in keys), evaluator=HexfieldStub(),
        on_move=rec_a, search_parity_mode=True, **kwargs,
    )
    session_b = hexfield_rust.HexfieldMctsSession(max_states=65536)
    session_b.run_continuous(
        keys, tuple(rec_b.states[k] for k in keys), evaluator=HexfieldStub(),
        on_move=rec_b,
        divergence_overrides={
            "lcb_move_selection": False,
            "early_stop": False,
            "moves_left_utility": False,
            # These divergences default ON in production(); this test compares
            # all-divergences-off against search_parity_mode, so they are set to
            # their parity() value here.
            "nucleus_f64": False,
            "new_child_fpu": False,
            "lazy_widening": False,
            "clean_root_prior_cache": False,
        },
        **kwargs,
    )
    assert rec_a.records == rec_b.records


@needs_native
def test_e2e_abi_roundtrip_with_real_model() -> None:
    """Payload -> packer -> HexfieldNet -> reply -> tree with divergences ON,
    exercising moves_left_bytes end to end. Checks that search completes, visits
    land, and the chosen move is legal."""

    import torch

    from hexfield.inference import HexfieldEvaluator
    from hexfield.model import HexfieldNet

    torch.manual_seed(0)
    evaluator = HexfieldEvaluator(HexfieldNet(), device="cpu")
    session = hexfield_rust.HexfieldMctsSession(max_states=4096)

    state = api.new_game()
    for q, r in [(0, 0), (1, 1), (-1, 0), (2, -1), (0, 2)]:
        api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    legal = set(api.legal_action_ids(state))

    results = session.search(
        [4242],
        (state,),
        visits=24,
        c_puct=1.5,
        temperature=0.0,
        seed=5,
        evaluator=evaluator,
        virtual_batch_size=8,
        widening_policy_mass=0.95,
        widening_max_children=96,
        widening_min_children=2,
    )
    result = results[0]
    assert result["visits"] == 24
    assert result["action_id"] in legal
    assert result["visit_policy_count"] >= 1
    # With divergences ON (production defaults), the result payload carries the
    # LCB and early-stop fields.
    assert "lcb_override" in result
    assert "early_stopped" in result