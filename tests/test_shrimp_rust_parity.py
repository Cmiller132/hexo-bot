"""Rust/Python featurizer parity checks.

Compares shrimp._rust.featurize_states against the Python featurizer on
sampled engine states: node order, counts, BFS distances, neighbour tables,
and features. Feature floats are compared in f32 with a <= 1e-6 tolerance;
all other outputs are compared exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from shrimp_testkit import api, sample_decision_states

from shrimp import constants as C
from shrimp.engine_facts import facts_from_engine
from shrimp.features import build_position

try:
    from shrimp import _rust
except ImportError:  # pragma: no cover
    _rust = None

needs_rust = pytest.mark.skipif(
    _rust is None, reason="shrimp._rust not built (scripts/_rebuild_shrimp.sh)"
)


@needs_rust
def test_capabilities() -> None:
    caps = _rust.capabilities()
    assert caps["model_family"] == "shrimp"
    assert caps["num_features"] == C.NUM_FEATURES == 15


@needs_rust
def test_rust_python_featurizer_parity() -> None:
    states = sample_decision_states(range(10), (0, 1, 3, 7, 12, 19, 28, 41))
    assert len(states) >= 50
    payloads = _rust.featurize_states(states)
    assert len(payloads) == len(states)

    for state, payload in zip(states, payloads):
        facts = facts_from_engine(api.to_python_state(state))
        sup, feats = build_position(facts)
        n = sup.num_nodes

        assert payload["num_nodes"] == n
        assert payload["legal_count"] == sup.legal_count
        assert payload["stone_count"] == sup.stone_count
        assert payload["halo_count"] == sup.halo_count

        coords = np.frombuffer(payload["coords"], dtype=np.int16).reshape(n, 2)
        assert np.array_equal(coords.astype(np.int32), sup.coords)

        dist = np.frombuffer(payload["dist"], dtype=np.int32)
        assert np.array_equal(dist, sup.dist)

        nbr = np.frombuffer(payload["nbr"], dtype=np.int32).reshape(n, 6)
        assert np.array_equal(nbr, sup.nbr)

        rust_feats = np.frombuffer(payload["feats"], dtype=np.float32).reshape(
            n, C.NUM_FEATURES
        )
        # All features within 1e-6; recency planes may differ within this
        # tolerance, the remaining planes are checked for exact equality below.
        diff = np.abs(rust_feats - feats)
        assert diff.max() <= 1e-6, (
            f"feature mismatch: max diff {diff.max()} at "
            f"{np.unravel_index(diff.argmax(), diff.shape)}"
        )
        exact_planes = [p for p in range(C.NUM_FEATURES) if p not in (C.F_OWN_RECENCY, C.F_OPP_RECENCY)]
        assert np.array_equal(rust_feats[:, exact_planes], feats[:, exact_planes])


@needs_rust
def test_ply0_payload() -> None:
    state = api.new_game()
    payload = _rust.featurize_states([state])[0]
    assert payload["num_nodes"] == 7
    assert payload["legal_count"] == 1
    assert payload["halo_count"] == 6
    dist = np.frombuffer(payload["dist"], dtype=np.int32)
    assert dist.tolist() == [0] * 7
