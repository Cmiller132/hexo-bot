"""Analysis endpoint: payload sanity, caching, searched eval, bounds."""

from __future__ import annotations

import pytest

from test_showcase_api import create_game, fresh_ip, poll_until, resign


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


def _analysis(client, game_id: str, ply: int, search: bool = False) -> dict:
    resp = client.get(
        f"/api/game/{game_id}/analysis",
        params={"ply": ply, **({"search": "1"} if search else {})},
        headers=fresh_ip(),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_analysis_payload_sanity(client, finished_game):
    payload = _analysis(client, finished_game["id"], ply=2)
    assert payload["cached"] is False
    assert payload["ply"] == 2
    assert payload["bot_id"] == finished_game["bot"]["id"]
    assert -1.0 <= payload["value"] <= 1.0
    assert payload["to_move"] in (0, 1)
    assert payload["legal_count"] > 0

    policy = payload["policy"]
    assert policy
    probs = [row["p"] for row in policy]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    # The dense policy is a softmax over the legal prefix; the sparse floor
    # (1e-4) only trims negligible tail mass.
    assert 0.98 <= sum(probs) <= 1.0 + 1e-6

    top_k = payload["top_k"]
    assert 1 <= len(top_k) <= 5
    assert [row["p"] for row in top_k] == probs[: len(top_k)]
    assert "search" not in payload


def test_analysis_cache_hit(client, finished_game):
    first = _analysis(client, finished_game["id"], ply=3)
    assert first["cached"] is False
    second = _analysis(client, finished_game["id"], ply=3)
    assert second["cached"] is True
    for key in ("value", "policy", "top_k", "legal_count", "to_move"):
        assert second[key] == first[key]


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
    resign(client, active["id"], headers)
