"""Lab endpoints (/api/lab/eval, /api/lab/search): payload schemas, sequence
vs free-edit featurization (history features zeroed in free mode), attention
and activation payload shapes, validation errors, sims-cap enforcement,
per-checkpoint (PUCT) profile routing, rate limiting, and the enable flag.

The parity tests rebuild the server's featurization with shrimp directly (the
same modules the worker imports), so the endpoint output is checked against
ground truth, not against itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from test_showcase_api import fresh_ip
from test_showcase_units import SIX_IN_LINE_MOVES

# A short legal opening: p0 (0,0); p1 (1,1), (0,2). Player 0 to move next
# (FirstStone phase).
SEQ3 = [(0, 0), (1, 1), (0, 2)]
SEQ3_P0 = [(0, 0)]
SEQ3_P1 = [(1, 1), (0, 2)]


def cells(pairs) -> list[dict]:
    return [{"q": q, "r": r} for q, r in pairs]


def lab_eval(client, body: dict, expect: int = 200) -> dict:
    resp = client.post("/api/lab/eval", json=body, headers=fresh_ip())
    assert resp.status_code == expect, resp.text
    return resp.json()


def lab_search(client, body: dict, expect: int = 200) -> dict:
    resp = client.post("/api/lab/search", json=body, headers=fresh_ip())
    assert resp.status_code == expect, resp.text
    return resp.json()


def _assert_sparse_policy(rows, legal_count):
    probs = [row["p"] for row in rows]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    floor_mass = legal_count * 1e-4
    rounding = legal_count * 5e-7 + 1e-9
    assert 1.0 - floor_mass - rounding <= sum(probs) <= 1.0 + rounding


# ---------------------------------------------------------------------------
# eval: sequence mode
# ---------------------------------------------------------------------------


def test_lab_eval_sequence_payload(client):
    payload = lab_eval(client, {"checkpoint_id": "tiny", "actions": cells(SEQ3)})
    assert payload["checkpoint_id"] == "tiny"
    assert payload["mode"] == "sequence"
    assert payload["ply"] == 3
    assert payload["to_move"] == 0
    assert payload["phase"] == "FirstStone"
    assert "synthesized_history" not in payload

    sup = payload["support"]
    assert sup["stone_count"] == 3
    assert sup["legal_count"] == payload["legal_count"] > 0
    assert len(sup["coords"]) == sup["legal_count"] + sup["stone_count"] + sup["halo_count"]

    assert -1.0 <= payload["value"] <= 1.0
    assert len(payload["value_dist"]) == 65
    assert math.isclose(sum(payload["value_dist"]), 1.0, abs_tol=1e-3)
    assert set(payload["stv"]) == {"2", "6", "16"}
    assert all(-1.0 <= v <= 1.0 for v in payload["stv"].values())
    assert 0.0 <= payload["moves_left"] <= 209.0

    for head in ("policy", "opp_policy", "soft_policy"):
        _assert_sparse_policy(payload[head], payload["legal_count"])
    assert payload["top_k"] == payload["policy"][:5]

    # Internals not requested, not present.
    assert "attention" not in payload
    assert "activations" not in payload
    assert "features" not in payload


def test_lab_eval_empty_board(client):
    payload = lab_eval(client, {"checkpoint_id": "tiny", "actions": []})
    assert payload["ply"] == 0
    assert payload["legal_count"] == 1  # opening forced to the origin
    assert payload["support"]["stone_count"] == 0
    assert payload["policy"] == [{"q": 0, "r": 0, "p": 1.0}]


def test_lab_eval_features_match_shrimp_featurizer(client):
    """The endpoint's feature planes equal build_features over the same
    engine-replayed position (exact serve featurization)."""
    import hexo_engine as engine
    from hexo_engine.types import AxialCoord, PlacementAction
    from shrimp.engine_facts import facts_from_state
    from shrimp.features import build_features
    from shrimp.support import build_support

    state = engine.new_game()
    for q, r in SEQ3:
        engine.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    facts = facts_from_state(state)
    support = build_support(facts.stones())
    expected = build_features(facts, support)

    payload = lab_eval(
        client,
        {"checkpoint_id": "tiny", "actions": cells(SEQ3), "wants": {"features": True}},
    )
    feats = payload["features"]
    assert len(feats["names"]) == 15
    got = np.asarray(feats["planes"], dtype=np.float32).T  # (N, 15)
    assert got.shape == expected.shape
    assert np.allclose(got, expected, atol=1e-6)
    assert payload["support"]["coords"] == support.coords.tolist()


def test_lab_eval_sequence_validation(client):
    # Neither / both encodings.
    lab_eval(client, {"checkpoint_id": "tiny"}, expect=422)
    lab_eval(
        client,
        {"checkpoint_id": "tiny", "actions": [], "stones": {"p0": [], "p1": []}},
        expect=422,
    )
    # Opening anywhere but the origin is illegal.
    resp = client.post(
        "/api/lab/eval",
        json={"checkpoint_id": "tiny", "actions": cells([(5, 5)])},
        headers=fresh_ip(),
    )
    assert resp.status_code == 422
    assert "illegal" in resp.json()["detail"]
    # Terminal positions are not decision states.
    resp = client.post(
        "/api/lab/eval",
        json={"checkpoint_id": "tiny", "actions": cells(SIX_IN_LINE_MOVES)},
        headers=fresh_ip(),
    )
    assert resp.status_code == 422
    assert "terminal" in resp.json()["detail"]
    # Moves after a terminal placement are unreachable.
    resp = client.post(
        "/api/lab/eval",
        json={"checkpoint_id": "tiny", "actions": cells(SIX_IN_LINE_MOVES + [(9, 9)])},
        headers=fresh_ip(),
    )
    assert resp.status_code == 422
    # Unknown checkpoint.
    lab_eval(client, {"checkpoint_id": "nope", "actions": []}, expect=404)
    # Oversized action list is rejected by the schema.
    lab_eval(
        client,
        {"checkpoint_id": "tiny", "actions": cells([(0, 0)] * 401)},
        expect=422,
    )


# ---------------------------------------------------------------------------
# eval: free-edit mode
# ---------------------------------------------------------------------------


def test_lab_eval_free_mode_zeroes_history_features(client):
    """The same stones through both modes: history-derived planes (own/opp
    recency, opp last turn) are live in sequence mode and zeroed in free mode;
    every other plane is identical."""
    seq = lab_eval(
        client,
        {"checkpoint_id": "tiny", "actions": cells(SEQ3), "wants": {"features": True}},
    )
    free = lab_eval(
        client,
        {
            "checkpoint_id": "tiny",
            "stones": {"p0": cells(SEQ3_P0), "p1": cells(SEQ3_P1)},
            "to_move": 0,
            "wants": {"features": True},
        },
    )
    assert free["mode"] == "free"
    assert free["synthesized_history"] is True
    assert free["zeroed_features"] == ["own_recency", "opp_recency", "opp_last_turn"]
    assert free["to_move"] == seq["to_move"] == 0
    assert free["phase"] == seq["phase"] == "FirstStone"
    # Same stones -> same support, same node order.
    assert free["support"] == seq["support"]

    names = seq["features"]["names"]
    seq_planes = {n: seq["features"]["planes"][i] for i, n in enumerate(names)}
    free_planes = {n: free["features"]["planes"][i] for i, n in enumerate(names)}
    zeroed = set(free["zeroed_features"])
    for name in zeroed:
        assert any(v > 0 for v in seq_planes[name]), f"{name} unexpectedly empty in sequence mode"
        assert all(v == 0 for v in free_planes[name]), f"{name} not zeroed in free mode"
    for name in set(names) - zeroed:
        assert free_planes[name] == seq_planes[name], f"{name} differs between modes"


def test_lab_eval_free_default_to_move(client):
    # p0 ahead by one stone -> p1 to move by default.
    payload = lab_eval(
        client,
        {"checkpoint_id": "tiny", "stones": {"p0": cells([(0, 0), (1, 0)]), "p1": cells([(0, 1)])}},
    )
    assert payload["to_move"] == 1
    # Explicit to_move wins.
    payload = lab_eval(
        client,
        {
            "checkpoint_id": "tiny",
            "stones": {"p0": cells([(0, 0), (1, 0)]), "p1": cells([(0, 1)])},
            "to_move": 0,
        },
    )
    assert payload["to_move"] == 0


def test_lab_eval_free_validation(client):
    # Overlapping stones.
    resp = client.post(
        "/api/lab/eval",
        json={"checkpoint_id": "tiny", "stones": {"p0": cells([(0, 0)]), "p1": cells([(0, 0)])}},
        headers=fresh_ip(),
    )
    assert resp.status_code == 422
    assert "more than one stone" in resp.json()["detail"]
    # Implausible count parity (5 vs 0).
    resp = client.post(
        "/api/lab/eval",
        json={
            "checkpoint_id": "tiny",
            "stones": {"p0": cells([(i, 0) for i in range(5)]), "p1": []},
        },
        headers=fresh_ip(),
    )
    assert resp.status_code == 422
    assert "parity" in resp.json()["detail"]
    # Out-of-range coordinate (schema bound).
    lab_eval(
        client,
        {"checkpoint_id": "tiny", "stones": {"p0": cells([(9999, 0)]), "p1": []}},
        expect=422,
    )
    # Empty free board is a valid sandbox state.
    payload = lab_eval(client, {"checkpoint_id": "tiny", "stones": {"p0": [], "p1": []}})
    assert payload["legal_count"] == 1


# ---------------------------------------------------------------------------
# eval: attention + activations
# ---------------------------------------------------------------------------


def test_lab_eval_attention_rows(client):
    query = {"q": 0, "r": 2}  # the last stone
    payload = lab_eval(
        client,
        {
            "checkpoint_id": "tiny",
            "actions": cells(SEQ3),
            "wants": {"attention_query": query},
        },
    )
    attn = payload["attention"]
    assert attn["query"]["q"] == 0 and attn["query"]["r"] == 2
    coords = payload["support"]["coords"]
    assert coords[attn["query"]["node"]] == [0, 2]
    # The tiny test net is trunk CCA: 1 attention block, 4 heads.
    assert attn["blocks"] == 1
    assert attn["heads"] == 4
    assert len(attn["rows"]) == attn["blocks"]
    for block_rows in attn["rows"]:
        assert len(block_rows) == attn["heads"]
        for row in block_rows:
            assert len(row["tokens"]) == 8
            weights = row["tokens"] + list(row["cells"].values())
            assert all(0.0 <= w <= 1.0 for w in weights)
            # Below-floor cells are dropped (sum <= 1) and each kept weight
            # is quantized to 4 decimals (up to 5e-5 upward per entry).
            assert sum(weights) <= 1.0 + len(weights) * 5e-5 + 1e-6
            assert all(0 <= int(node) < len(coords) for node in row["cells"])
            assert all(w >= attn["floor"] for w in row["cells"].values())


def test_lab_eval_attention_query_outside_support_is_422(client):
    resp = client.post(
        "/api/lab/eval",
        json={
            "checkpoint_id": "tiny",
            "actions": cells(SEQ3),
            "wants": {"attention_query": {"q": 100, "r": 100}},
        },
        headers=fresh_ip(),
    )
    assert resp.status_code == 422
    assert "support" in resp.json()["detail"]


def test_lab_eval_activation_norms(client):
    payload = lab_eval(
        client,
        {"checkpoint_id": "tiny", "actions": cells(SEQ3), "wants": {"activations": True}},
    )
    blocks = payload["activations"]["blocks"]
    # Tiny trunk CCA: stem + conv1 + conv2 + attn1, in execution order.
    assert [(b["label"], b["kind"]) for b in blocks] == [
        ("stem", "stem"), ("conv1", "conv"), ("conv2", "conv"), ("attn1", "attn"),
    ]
    n_nodes = len(payload["support"]["coords"])
    for block in blocks:
        assert len(block["norms"]) == n_nodes
        assert all(v >= 0.0 for v in block["norms"])
    # The trunk does real work: some stage must produce nonzero activations.
    assert any(v > 0.0 for b in blocks for v in b["norms"])


def test_lab_eval_attention_and_activations_in_free_mode(client):
    payload = lab_eval(
        client,
        {
            "checkpoint_id": "tiny",
            "stones": {"p0": cells(SEQ3_P0), "p1": cells(SEQ3_P1)},
            "wants": {"attention_query": {"q": 1, "r": 1}, "activations": True},
        },
    )
    assert payload["mode"] == "free"
    assert payload["attention"]["rows"]
    assert len(payload["activations"]["blocks"]) == 4


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_lab_search_payload(client):
    payload = lab_search(
        client, {"checkpoint_id": "tiny", "actions": cells(SEQ3), "sims": 8}
    )
    assert payload["checkpoint_id"] == "tiny"
    assert payload["sims"] == 8
    assert 1 <= payload["visits"] <= 8
    assert -1.0 <= payload["root_value"] <= 1.0
    assert set(payload["best"]) == {"q", "r"}
    rows = payload["visit_policy"]
    assert rows
    probs = [row["p"] for row in rows]
    assert probs == sorted(probs, reverse=True)
    assert abs(sum(probs) - 1.0) < 1e-2
    # w is the raw wire weight (visit counts under PUCT, improved-policy mass
    # under Gumbel) — nonnegative, and proportional to p.
    assert all(row["w"] >= 0 for row in rows)
    # The chosen move is one of the visited moves.
    assert any(row["q"] == payload["best"]["q"] and row["r"] == payload["best"]["r"] for row in rows)


def test_lab_search_sims_cap_enforced(client, settings):
    assert settings.lab_search_visit_cap == 256  # code default
    resp = client.post(
        "/api/lab/search",
        json={"checkpoint_id": "tiny", "actions": cells(SEQ3), "sims": 257},
        headers=fresh_ip(),
    )
    assert resp.status_code == 422
    assert "256" in resp.json()["detail"]
    # sims <= 0 fails schema validation.
    lab_search(client, {"checkpoint_id": "tiny", "actions": cells(SEQ3), "sims": 0}, expect=422)


def test_lab_search_requires_actions(client):
    # Free-edit positions cannot be searched (no engine state): the schema
    # takes `actions` only.
    resp = client.post(
        "/api/lab/search",
        json={"checkpoint_id": "tiny", "stones": {"p0": [], "p1": []}, "sims": 8},
        headers=fresh_ip(),
    )
    assert resp.status_code == 422


def test_lab_search_rejects_illegal_and_terminal(client):
    lab_search(
        client, {"checkpoint_id": "tiny", "actions": cells([(3, 3)]), "sims": 8},
        expect=422,
    )
    lab_search(
        client,
        {"checkpoint_id": "tiny", "actions": cells(SIX_IN_LINE_MOVES), "sims": 8},
        expect=422,
    )


def test_lab_search_on_legacy_puct_checkpoint(client):
    """The lab search runs under the selected checkpoint's own as-trained
    profile — the tiny-puct entry routes through the main_5 PUCT profile
    (binding covered in test_showcase_profiles)."""
    payload = lab_search(
        client, {"checkpoint_id": "tiny-puct", "actions": cells(SEQ3), "sims": 8}
    )
    assert payload["checkpoint_id"] == "tiny-puct"
    assert 1 <= payload["visits"] <= 8
    assert abs(sum(row["p"] for row in payload["visit_policy"]) - 1.0) < 1e-2


# ---------------------------------------------------------------------------
# rate limiting + enable flag
# ---------------------------------------------------------------------------


def test_lab_eval_rate_limit(client, settings):
    """One IP gets `lab_eval_per_minute` burst tokens; the flood then 429s.
    The bucket refills at per_minute/60 per second, so a couple of extra
    successes during the loop are tolerated."""
    headers = {"CF-Connecting-IP": "10.99.0.1"}
    body = {"checkpoint_id": "tiny", "stones": {"p0": [], "p1": []}}  # cheapest eval
    statuses = [
        client.post("/api/lab/eval", json=body, headers=headers).status_code
        for _ in range(settings.lab_eval_per_minute + 15)
    ]
    assert statuses[0] == 200
    assert 429 in statuses
    assert statuses.count(200) >= settings.lab_eval_per_minute


def test_lab_search_rate_limit(client, settings):
    headers = {"CF-Connecting-IP": "10.99.0.2"}
    body = {"checkpoint_id": "tiny", "actions": cells(SEQ3), "sims": 1}
    statuses = [
        client.post("/api/lab/search", json=body, headers=headers).status_code
        for _ in range(settings.lab_search_per_minute + 8)
    ]
    assert statuses[0] == 200
    assert 429 in statuses
    assert statuses.count(200) >= settings.lab_search_per_minute
    # The eval bucket is independent of the search bucket.
    ok = client.post(
        "/api/lab/eval",
        json={"checkpoint_id": "tiny", "stones": {"p0": [], "p1": []}},
        headers=headers,
    )
    assert ok.status_code == 200


def test_lab_disabled_flag(client, settings):
    """SHOWCASE_LAB_ENABLED=0 turns both endpoints into 404s."""
    object.__setattr__(settings, "lab_enabled", False)
    try:
        lab_eval(client, {"checkpoint_id": "tiny", "actions": []}, expect=404)
        lab_search(
            client, {"checkpoint_id": "tiny", "actions": cells(SEQ3), "sims": 8},
            expect=404,
        )
    finally:
        object.__setattr__(settings, "lab_enabled", True)
    lab_eval(client, {"checkpoint_id": "tiny", "actions": []})


def test_lab_eval_payload_scrubs_non_finite(monkeypatch):
    """Forged non-finite decodes become None in the lab payload, which then
    survives a strict (allow_nan=False) encode — lab responses ride the same
    NaN-intolerant response path as analysis."""
    import json

    import torch
    from shrimp.model import ShrimpNet

    from showcase import lab

    monkeypatch.setattr(
        lab, "decode_binned_value",
        lambda logits: torch.full((logits.shape[0],), float("nan")),
    )
    facts, support, feats = lab.build_sequence_position([(0, 0)])
    payload = lab.eval_payload(
        ShrimpNet().eval(), facts, support, feats,
        policy_floor=1e-4, attention_cell=None,
        want_activations=True, want_features=False,
    )
    assert payload["value"] is None
    assert all(v is None for v in payload["stv"].values())
    assert payload["moves_left"] is not None  # untouched head still decodes
    json.dumps(payload, allow_nan=False)  # must not raise
