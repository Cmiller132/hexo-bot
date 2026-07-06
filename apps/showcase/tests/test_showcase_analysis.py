"""Analysis + summary endpoints: payload sanity, caching (incl. schema-version
recompute), searched eval, bounds, the public six_in_line archive read, and
the per-checkpoint analysis selector (?checkpoint_id=)."""

from __future__ import annotations

import gzip
import json

import pytest

from showcase.db import encode_payload
from showcase.game import encode_hxr, now_iso
from test_showcase_api import create_game, fresh_ip, poll_until, resign
from test_showcase_units import SIX_IN_LINE_MOVES, drive_six_in_line_session


@pytest.fixture(scope="module")
def finished_game(client) -> dict:
    """One short finished game (a few plies then resignation)."""
    headers = fresh_ip()
    snap = create_game(client, headers)
    game_id = snap["id"]
    for _ in range(3):
        snap = poll_until(client, game_id)
        if snap["status"] != "your_turn":
            break
        cell = snap["legal"][len(snap["legal"]) // 2]
        client.post(f"/api/game/{game_id}/move", json=cell, headers=headers)
    poll_until(client, game_id)
    final = resign(client, game_id, headers)
    return final


def _analysis(
    client, game_id: str, ply: int, search: bool = False, ckpt: str | None = None,
) -> dict:
    resp = client.get(
        f"/api/game/{game_id}/analysis",
        params={
            "ply": ply,
            **({"search": "1"} if search else {}),
            **({"checkpoint_id": ckpt} if ckpt is not None else {}),
        },
        headers=fresh_ip(),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_analysis_payload_sanity(client, finished_game):
    payload = _analysis(client, finished_game["id"], ply=2)
    assert payload["cached"] is False
    assert payload["ply"] == 2
    assert payload["checkpoint_id"] == finished_game["bot"]["checkpoint_id"]
    assert -1.0 <= payload["value"] <= 1.0
    # stv: short-horizon value head, same POV/range as value; moves_left: plies.
    assert -1.0 <= payload["stv"] <= 1.0
    assert 0.0 <= payload["moves_left"] <= 209.0
    assert payload["to_move"] in (0, 1)
    assert payload["legal_count"] > 0

    policy = payload["policy"]
    assert policy
    probs = [row["p"] for row in policy]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    # The dense policy is a softmax over the legal prefix (sums to 1). Two
    # exact error sources for the reported sum: the sparse floor (1e-4) trims
    # at most legal_count * 1e-4 of tail mass, and each p is rounded to 6
    # decimals (up to 5e-7 per cell, either direction). The random-weight test
    # net feeds ~250 legal cells here, where guessed constants flake.
    floor_mass = payload["legal_count"] * 1e-4
    rounding = payload["legal_count"] * 5e-7 + 1e-9
    assert 1.0 - floor_mass - rounding <= sum(probs) <= 1.0 + rounding

    top_k = payload["top_k"]
    assert 1 <= len(top_k) <= 5
    assert [row["p"] for row in top_k] == probs[: len(top_k)]
    assert "search" not in payload


def test_analysis_cache_hit(client, finished_game):
    first = _analysis(client, finished_game["id"], ply=3)
    assert first["cached"] is False
    second = _analysis(client, finished_game["id"], ply=3)
    assert second["cached"] is True
    for key in ("value", "stv", "moves_left", "policy", "top_k", "legal_count", "to_move"):
        assert second[key] == first[key]


def test_analysis_cache_version_mismatch_recomputes(client, finished_game):
    """A cached payload from an older schema (no/old version stamp) is treated
    as a miss and recomputed with the new fields."""
    game_id = finished_game["id"]
    db = client.app.state.db
    bot_db_id = db.get_game(game_id)["bot_id"]
    _analysis(client, game_id, ply=1)
    assert _analysis(client, game_id, ply=1)["cached"] is True
    # Simulate a stale pre-stv cache entry (schema v1: no "v" stamp).
    db.analysis_put(game_id, 1, bot_db_id, encode_payload({"value": 0.0, "ply": 1}))
    recomputed = _analysis(client, game_id, ply=1)
    assert recomputed["cached"] is False
    assert "stv" in recomputed and "moves_left" in recomputed
    assert _analysis(client, game_id, ply=1)["cached"] is True


def test_analysis_searched_eval(client, finished_game, settings):
    payload = _analysis(client, finished_game["id"], ply=2, search=True)
    search = payload["search"]
    assert 1 <= search["visits"] <= settings.analysis_search_visit_cap
    assert -1.0 <= search["root_value"] <= 1.0
    assert set(search["best"]) == {"q", "r"}
    visit_probs = [row["p"] for row in search["visit_policy"]]
    assert visit_probs == sorted(visit_probs, reverse=True)
    assert abs(sum(visit_probs) - 1.0) < 1e-2
    # The upgraded payload (net eval + search) is now the cached artifact.
    again = _analysis(client, finished_game["id"], ply=2, search=True)
    assert again["cached"] is True
    assert again["search"]["visits"] == search["visits"]


def test_analysis_initial_position_and_bounds(client, finished_game):
    empty = _analysis(client, finished_game["id"], ply=0)
    assert empty["legal_count"] == 1  # opening placement is forced
    assert empty["to_move"] == 0

    resp = client.get(
        f"/api/game/{finished_game['id']}/analysis",
        params={"ply": 999},
        headers=fresh_ip(),
    )
    assert resp.status_code == 422


def test_analysis_gates(client):
    assert client.get(
        "/api/game/no-such-game/analysis", params={"ply": 0}, headers=fresh_ip()
    ).status_code == 404

    headers = fresh_ip()
    active = create_game(client, headers)
    resp = client.get(
        f"/api/game/{active['id']}/analysis", params={"ply": 0}, headers=headers
    )
    assert resp.status_code == 409  # finished games only
    assert client.get(
        f"/api/game/{active['id']}/summary", headers=headers
    ).status_code == 409
    resign(client, active["id"], headers)


@pytest.fixture(scope="module")
def archived_six_game(client) -> dict:
    """A scripted six_in_line game inserted straight into the DB (no live
    session) — the archive/shareable-URL shape."""
    session = drive_six_in_line_session()
    db = client.app.state.db
    bot_db_id = db.upsert_bot(
        slug="tiny", label="Tiny test bot", run="showcase_tiny_test", epoch=0,
        visits=8, weights_sha="scripted-six", active_from=now_iso(),
    )
    db.create_game(
        game_id=session.game_id, bot_id=bot_db_id, human_color=0,
        started_at=now_iso(), client_hash="scripted",
    )
    db.finalize_game(
        game_id=session.game_id, finished_at=now_iso(), status="finished",
        result=1, termination="six_in_line", ply_count=len(session.actions),
        duration_s=1.0,
        record=encode_hxr(
            game_id=session.game_id, bot_slug="tiny", human_color=0,
            action_ids=session.actions, winner=0, termination="six_in_line",
            seed=session.seed,
        ),
    )
    return {"id": session.game_id, "ply_count": len(session.actions)}


def test_public_read_of_archived_six_in_line_game(client, archived_six_game):
    client.cookies.clear()
    snap = client.get(f"/api/game/{archived_six_game['id']}", headers=fresh_ip())
    assert snap.status_code == 200  # public: no cookie, no live session
    snap = snap.json()
    assert snap["status"] == "finished"
    assert snap["result"] == {"winner": 0, "termination": "six_in_line", "human_result": 1}
    assert snap["winning_line"] == [{"q": q, "r": 0} for q in range(6)]
    assert [(s["q"], s["r"]) for s in snap["stones"]] == SIX_IN_LINE_MOVES  # placement order
    assert snap["bot"] == {
        "checkpoint_id": "tiny", "label": "Tiny test bot", "epoch": 0, "sims": 8,
    }
    assert snap["to_move"] is None
    assert snap["finished_at"]


def test_summary_shape_and_cache(client, archived_six_game):
    game_id = archived_six_game["id"]
    ply_count = archived_six_game["ply_count"]
    first = client.get(f"/api/game/{game_id}/summary", headers=fresh_ip())
    assert first.status_code == 200, first.text
    first = first.json()
    assert first["cached"] is False
    assert first["ply_count"] == ply_count
    # Index i = the position AFTER ply i: ply_count + 1 entries, empty board first.
    for key in ("value", "stv", "moves_left", "to_move"):
        assert len(first[key]) == ply_count + 1
    assert all(-1.0 <= v <= 1.0 for v in first["value"])
    assert all(-1.0 <= v <= 1.0 for v in first["stv"])
    assert all(0.0 <= v <= 209.0 for v in first["moves_left"])
    assert first["to_move"][0] == 0  # opening: player0 to move
    assert first["to_move"][-1] is None  # terminal position: nobody to move

    second = client.get(f"/api/game/{game_id}/summary", headers=fresh_ip()).json()
    assert second["cached"] is True
    for key in ("value", "stv", "moves_left", "to_move", "ply_count"):
        assert second[key] == first[key]


# ---------------------------------------------------------------------------
# checkpoint selector (?checkpoint_id=)
# ---------------------------------------------------------------------------


def test_selector_unknown_checkpoint_404(client, finished_game):
    game_id = finished_game["id"]
    resp = client.get(
        f"/api/game/{game_id}/analysis",
        params={"ply": 0, "checkpoint_id": "nope"},
        headers=fresh_ip(),
    )
    assert resp.status_code == 404
    assert "nope" in resp.json()["detail"]
    resp = client.get(
        f"/api/game/{game_id}/summary",
        params={"checkpoint_id": "nope"},
        headers=fresh_ip(),
    )
    assert resp.status_code == 404


def test_selector_per_checkpoint_cache_and_analysis(client, finished_game, settings):
    """Two checkpoints analyzing the same (game, ply) cache independently, and
    the default cache row keeps serving after a selector round trip."""
    import sqlite3

    game_id = finished_game["id"]
    default = _analysis(client, game_id, ply=0)
    assert default["checkpoint_id"] == "tiny"
    assert _analysis(client, game_id, ply=0)["cached"] is True

    other = _analysis(client, game_id, ply=0, ckpt="tiny-puct")
    assert other["cached"] is False  # the default row did not serve this
    assert other["checkpoint_id"] == "tiny-puct"
    assert -1.0 <= other["value"] <= 1.0
    assert other["legal_count"] == default["legal_count"]
    assert _analysis(client, game_id, ply=0, ckpt="tiny-puct")["cached"] is True
    assert _analysis(client, game_id, ply=0)["cached"] is True  # default intact

    conn = sqlite3.connect(settings.db_path)
    rows = conn.execute(
        "SELECT DISTINCT c.bot_id, b.slug, b.visits FROM analysis_cache c"
        " JOIN bots b ON b.id = c.bot_id WHERE c.game_id = ? AND c.ply = 0",
        (game_id,),
    ).fetchall()
    assert len(rows) == 2  # one row per analyzing checkpoint
    by_slug = {slug: visits for _, slug, visits in rows}
    assert by_slug["tiny"] == 8  # the game's own bot row
    assert by_slug["tiny-puct"] == 0  # analysis-only sentinel row

    # Analysis-only bot rows never surface in the stats views (no games).
    stats = client.get("/api/stats").json()
    assert all(row["visits"] != 0 for row in stats["bots"])


def test_selector_of_the_games_own_checkpoint_shares_the_default_cache(client, finished_game):
    game_id = finished_game["id"]
    ply = finished_game["ply"]  # a ply no other test analyzes for this game
    assert _analysis(client, game_id, ply=ply)["cached"] is False
    explicit = _analysis(client, game_id, ply=ply, ckpt="tiny")
    assert explicit["cached"] is True  # same cache row as the default request
    assert explicit["checkpoint_id"] == "tiny"


def test_selector_searched_eval_uses_the_selected_bot(client, finished_game, settings):
    """?checkpoint_id=tiny-puct&search=1 on a game played vs tiny runs the
    searched eval on the tiny-puct worker bot, i.e. under its as-trained PUCT
    profile (profile binding per slug is covered in test_showcase_profiles)."""
    payload = _analysis(client, finished_game["id"], ply=2, search=True, ckpt="tiny-puct")
    assert payload["checkpoint_id"] == "tiny-puct"
    search = payload["search"]
    assert 1 <= search["visits"] <= settings.analysis_search_visit_cap
    assert -1.0 <= search["root_value"] <= 1.0
    again = _analysis(client, finished_game["id"], ply=2, search=True, ckpt="tiny-puct")
    assert again["cached"] is True
    assert again["search"]["visits"] == search["visits"]


def test_selector_summary_per_checkpoint(client, archived_six_game):
    game_id = archived_six_game["id"]
    ply_count = archived_six_game["ply_count"]
    client.get(f"/api/game/{game_id}/summary", headers=fresh_ip())  # default cached
    first = client.get(
        f"/api/game/{game_id}/summary",
        params={"checkpoint_id": "tiny-puct"},
        headers=fresh_ip(),
    )
    assert first.status_code == 200, first.text
    first = first.json()
    assert first["cached"] is False  # keyed apart from the default summary
    assert first["checkpoint_id"] == "tiny-puct"
    assert len(first["value"]) == ply_count + 1
    second = client.get(
        f"/api/game/{game_id}/summary",
        params={"checkpoint_id": "tiny-puct"},
        headers=fresh_ip(),
    ).json()
    assert second["cached"] is True
    assert second["value"] == first["value"]
    assert client.get(
        f"/api/game/{game_id}/summary", headers=fresh_ip()
    ).json()["cached"] is True  # default row intact


def test_selector_covers_games_vs_retired_checkpoints(client):
    """A feed game against a checkpoint that left the catalogue: the default
    analysis 409s (unchanged), but any current checkpoint can analyze it."""
    session = drive_six_in_line_session()
    db = client.app.state.db
    bot_db_id = db.upsert_bot(
        slug="retired", label="Retired bot", run="old_run", epoch=99,
        visits=8, weights_sha="retired-sha", active_from=now_iso(),
    )
    db.create_game(
        game_id=session.game_id, bot_id=bot_db_id, human_color=0,
        started_at=now_iso(), client_hash="scripted",
    )
    db.finalize_game(
        game_id=session.game_id, finished_at=now_iso(), status="finished",
        result=1, termination="six_in_line", ply_count=len(session.actions),
        duration_s=1.0,
        record=encode_hxr(
            game_id=session.game_id, bot_slug="retired", human_color=0,
            action_ids=session.actions, winner=0, termination="six_in_line",
            seed=session.seed,
        ),
    )
    assert client.get(
        f"/api/game/{session.game_id}/analysis", params={"ply": 0}, headers=fresh_ip()
    ).status_code == 409
    assert client.get(
        f"/api/game/{session.game_id}/summary", headers=fresh_ip()
    ).status_code == 409
    payload = _analysis(client, session.game_id, ply=0, ckpt="tiny")
    assert payload["checkpoint_id"] == "tiny"
    summary = client.get(
        f"/api/game/{session.game_id}/summary",
        params={"checkpoint_id": "tiny"},
        headers=fresh_ip(),
    )
    assert summary.status_code == 200, summary.text
    assert summary.json()["checkpoint_id"] == "tiny"


# ---------------------------------------------------------------------------
# non-finite float hygiene (NaN in a payload used to 500, permanently once
# a NaN literal landed in analysis_cache)
# ---------------------------------------------------------------------------


def _pre_fix_blob(payload: dict) -> bytes:
    """A cache blob as the pre-fix encoder wrote it: json.dumps(allow_nan=True)
    persisted bare `NaN` literals into the gzip payload."""
    return gzip.compress(json.dumps(payload, separators=(",", ":")).encode())


def test_version_bump_purges_nan_poisoned_summary_row(client, archived_six_game):
    """The production incident: a v2 summary row with NaN in the value series
    made every /summary read 500 (Starlette encodes with allow_nan=False).
    The v3 stamp treats that row as a miss, recomputes, and overwrites it."""
    game_id = archived_six_game["id"]
    ply_count = archived_six_game["ply_count"]
    db = client.app.state.db
    bot_db_id = db.get_game(game_id)["bot_id"]
    db.analysis_put(game_id, -1, bot_db_id, _pre_fix_blob({
        "v": 2,
        "value": [float("nan")] + [0.0] * ply_count,
        "stv": [0.0] * (ply_count + 1),
        "moves_left": [1.0] * (ply_count + 1),
        "to_move": [0] * ply_count + [None],
        "ply_count": ply_count,
    }))
    resp = client.get(f"/api/game/{game_id}/summary", headers=fresh_ip())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cached"] is False  # the poisoned v2 row was a version miss
    assert all(v is None or -1.0 <= v <= 1.0 for v in body["value"])
    # The recomputed row serves cleanly from then on.
    again = client.get(f"/api/game/{game_id}/summary", headers=fresh_ip())
    assert again.status_code == 200
    assert again.json()["cached"] is True


def test_fresh_non_finite_summary_serves_and_caches_as_null(client, archived_six_game, monkeypatch):
    """A worker payload carrying NaN/Inf must reach the client as JSON null
    (200), and the cache row must be written clean. Workers are spawned
    processes, so the forgery happens at the web-side pool boundary."""
    game_id = archived_six_game["id"]
    ply_count = archived_six_game["ply_count"]
    db = client.app.state.db
    bot_db_id = db.get_game(game_id)["bot_id"]
    # Force a miss: stamp the existing summary row with a stale version.
    db.analysis_put(game_id, -1, bot_db_id, encode_payload({"v": 0}))

    async def forged_summary(**kwargs):
        return {
            "value": [float("nan")] + [0.5] * ply_count,
            "stv": [float("inf")] + [0.0] * ply_count,
            "moves_left": [float("-inf")] + [1.0] * ply_count,
            "to_move": [0] * ply_count + [None],
            "ply_count": ply_count,
        }

    monkeypatch.setattr(client.app.state.pool, "summary", forged_summary)
    resp = client.get(f"/api/game/{game_id}/summary", headers=fresh_ip())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cached"] is False
    assert body["value"][0] is None and body["value"][1] == 0.5
    assert body["stv"][0] is None
    assert body["moves_left"][0] is None
    # The cached artifact is the sanitized payload (no pool call: monkeypatch
    # is still active, a recompute would return non-finite floats again).
    again = client.get(f"/api/game/{game_id}/summary", headers=fresh_ip())
    assert again.status_code == 200
    body = again.json()
    assert body["cached"] is True
    assert body["value"][0] is None and body["stv"][0] is None


def test_net_eval_scrubs_non_finite_readouts(monkeypatch):
    """Forged non-finite value/stv/moves_left decodes come out as None and the
    payload survives a strict (allow_nan=False) encode — the response contract."""
    import torch

    import hexo_engine as engine
    from hexo_engine.types import AxialCoord, PlacementAction
    from shrimp.model import ShrimpNet

    from showcase import analysis

    monkeypatch.setattr(
        analysis, "decode_binned_value",
        lambda logits: torch.full((logits.shape[0],), float("nan")),
    )
    monkeypatch.setattr(
        analysis, "decode_moves_left",
        lambda logits: torch.full((logits.shape[0],), float("inf")),
    )
    model = ShrimpNet().eval()
    state = engine.new_game()
    engine.apply_action(state, PlacementAction(AxialCoord(q=0, r=0)))
    payload = analysis.net_eval(model, state, policy_floor=1e-4)
    assert payload["value"] is None
    assert payload["stv"] is None
    assert payload["moves_left"] is None
    assert payload["policy"]  # the (real) policy softmax is untouched
    json.dumps(payload, allow_nan=False)  # must not raise


def test_summary_eval_scrubs_non_finite_series(monkeypatch):
    import torch

    import hexo_engine as engine
    from shrimp.model import ShrimpNet

    from showcase import analysis

    monkeypatch.setattr(
        analysis, "decode_binned_value",
        lambda logits: torch.full((logits.shape[0],), float("-inf")),
    )
    monkeypatch.setattr(
        analysis, "decode_moves_left",
        lambda logits: torch.full((logits.shape[0],), float("nan")),
    )
    model = ShrimpNet().eval()
    out = analysis.summary_eval(model, [analysis.featurize(engine.new_game())])
    assert out == {"value": [None], "stv": [None], "moves_left": [None]}
