"""End-to-end API tests over the session-scoped TestClient (one worker pool
for the whole suite; the tiny random-weight net keeps bot turns fast)."""

from __future__ import annotations

import itertools
import sqlite3
import time

import hexo_engine as engine
from hexo_engine.types import PlacementAction, pack_coord_id, unpack_coord_id

from showcase.game import decode_hxr_actions

_IP_COUNTER = itertools.count(1)
_COOKIE = "showcase_token"


def fresh_ip() -> dict[str, str]:
    """A unique CF-Connecting-IP per call so per-IP caps never leak across tests."""
    n = next(_IP_COUNTER)
    return {"CF-Connecting-IP": f"10.7.{n // 250}.{n % 250}"}


def create_game(client, headers, bot_id: str = "tiny-8", human_color: int = 0) -> dict:
    resp = client.post(
        "/api/game", json={"bot_id": bot_id, "human_color": human_color}, headers=headers
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def poll_until(client, game_id: str, want=("your_turn", "finished"), timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = client.get(f"/api/game/{game_id}").json()
        if snap["status"] in want:
            return snap
        time.sleep(0.05)
    raise AssertionError(f"game {game_id} never reached {want}")


def resign(client, game_id: str, headers) -> dict:
    resp = client.post(f"/api/game/{game_id}/resign", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------


def test_healthz_and_bots(client):
    health = client.get("/healthz").json()
    assert health["ok"] is True
    assert health["bots"] == 2

    bots = client.get("/api/bots").json()
    assert {b["id"] for b in bots} == {"tiny-8", "tiny-16"}
    for bot in bots:
        assert bot["run"] == "showcase_tiny_test"
        assert bot["visits"] in (8, 16)
        assert bot["label"]


def test_full_game_to_terminal_or_20_plies(client):
    headers = fresh_ip()
    snap = create_game(client, headers)
    game_id = snap["id"]
    assert snap["status"] == "your_turn"
    assert snap["phase"] == "Opening"
    assert snap["stones_left_this_turn"] == 1
    assert snap["legal"] == [{"q": 0, "r": 0}]  # opening placement is forced

    while snap["status"] != "finished" and snap["ply"] < 20:
        if snap["status"] == "your_turn":
            assert snap["to_move"] == snap["human_color"]
            assert snap["stones_left_this_turn"] == (2 if snap["phase"] == "FirstStone" else 1)
            cell = snap["legal"][(snap["ply"] * 7) % len(snap["legal"])]
            resp = client.post(f"/api/game/{game_id}/move", json=cell, headers=headers)
            assert resp.status_code == 200, resp.text
            snap = resp.json()
            assert len(snap["stones"]) == snap["ply"]
            assert snap["last_move"] == {**cell, "color": snap["human_color"]}
        else:
            snap = poll_until(client, game_id)

    if snap["status"] != "finished":
        snap = resign(client, game_id, headers)
    assert snap["status"] == "finished"
    assert snap["result"]["termination"] in ("six_in_line", "resign")
    assert snap["legal"] == []
    assert len(snap["stones"]) == snap["ply"]


def test_bot_moves_first_when_human_is_player1(client):
    headers = fresh_ip()
    snap = create_game(client, headers, human_color=1)
    assert snap["status"] == "bot_thinking"
    snap = poll_until(client, snap["id"])
    assert snap["status"] == "your_turn"
    assert snap["ply"] == 1  # the bot's opening turn is a single stone
    assert snap["stones"] == [{"q": 0, "r": 0, "color": 0}]
    resign(client, snap["id"], headers)


def test_illegal_move_rejected(client):
    headers = fresh_ip()
    snap = create_game(client, headers)
    resp = client.post(f"/api/game/{snap['id']}/move", json={"q": 3, "r": 3}, headers=headers)
    assert resp.status_code == 422
    assert "illegal move" in resp.json()["detail"]
    resign(client, snap["id"], headers)


def test_mutating_routes_require_the_session_cookie(client):
    headers = fresh_ip()
    snap = create_game(client, headers)
    game_id = snap["id"]
    token = client.cookies.get(_COOKIE)
    assert token

    client.cookies.clear()
    assert client.post(f"/api/game/{game_id}/move", json={"q": 0, "r": 0}, headers=headers).status_code == 403
    assert client.post(f"/api/game/{game_id}/resign", headers=headers).status_code == 403
    client.cookies.set(_COOKIE, "wrong-token")
    assert client.post(f"/api/game/{game_id}/resign", headers=headers).status_code == 403

    client.cookies.set(_COOKIE, token)
    resign(client, game_id, headers)


def test_per_ip_active_game_cap(client):
    headers = fresh_ip()
    first = create_game(client, headers)
    second = create_game(client, headers)
    resp = client.post("/api/game", json={"bot_id": "tiny-8", "human_color": 0}, headers=headers)
    assert resp.status_code == 429
    resign(client, first["id"], headers)
    third = create_game(client, headers)  # capacity freed by the resignation
    resign(client, second["id"], headers)
    resign(client, third["id"], headers)


def test_resign_scores_for_the_bot(client):
    headers = fresh_ip()
    snap = create_game(client, headers)
    client.post(f"/api/game/{snap['id']}/move", json={"q": 0, "r": 0}, headers=headers)
    poll_until(client, snap["id"])
    snap = resign(client, snap["id"], headers)
    assert snap["status"] == "finished"
    assert snap["result"] == {
        "winner": 1 - snap["human_color"],
        "termination": "resign",
        "human_result": -1,
    }
    # Finished games reject further moves.
    resp = client.post(f"/api/game/{snap['id']}/move", json={"q": 1, "r": 0}, headers=headers)
    assert resp.status_code == 409


def test_unknown_game_and_bot_are_404(client):
    assert client.get("/api/game/no-such-game").status_code == 404
    resp = client.post(
        "/api/game", json={"bot_id": "no-such-bot", "human_color": 0}, headers=fresh_ip()
    )
    assert resp.status_code == 404


def test_nickname_set_and_sanitized(client, settings):
    headers = fresh_ip()
    snap = create_game(client, headers)
    game_id = snap["id"]
    # Not before the game is finished.
    assert client.post(f"/api/game/{game_id}/nickname", json={"nickname": "Bob"}, headers=headers).status_code == 409
    resign(client, game_id, headers)

    resp = client.post(
        f"/api/game/{game_id}/nickname",
        json={"nickname": "  Bob<script>alert(1)</script> Q._- "},
        headers=headers,
    )
    assert resp.status_code == 200
    stored = resp.json()["nickname"]
    assert stored == "Bobscriptalert1script Q."  # allowlist + 24-char cap
    assert len(stored) <= 24

    # Junk-only nicknames are rejected, not stored empty.
    assert client.post(f"/api/game/{game_id}/nickname", json={"nickname": "<<<>>>"}, headers=headers).status_code == 422

    row = sqlite3.connect(settings.db_path).execute(
        "SELECT nickname FROM games WHERE id = ?", (game_id,)
    ).fetchone()
    assert row[0] == stored


def test_db_row_and_hxr_record_roundtrip(client, settings):
    headers = fresh_ip()
    snap = create_game(client, headers)
    game_id = snap["id"]
    client.post(f"/api/game/{game_id}/move", json={"q": 0, "r": 0}, headers=headers)
    snap = poll_until(client, game_id)
    if snap["status"] == "your_turn":
        cell = snap["legal"][0]
        client.post(f"/api/game/{game_id}/move", json=cell, headers=headers)
        snap = poll_until(client, game_id)
    final = resign(client, game_id, headers)

    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM games WHERE id = ?", (game_id,)).fetchone()
    assert row["status"] == "finished"
    assert row["termination"] == "resign"
    assert row["result"] == -1
    assert row["ply_count"] == final["ply"]
    assert row["duration_s"] >= 0.0
    assert row["client_hash"]

    # The record blob is a real .hxr: decodes via hexo_utils and replays
    # legally through the engine to exactly the final board.
    actions = decode_hxr_actions(row["record"])
    assert len(actions) == final["ply"]
    assert actions[0] == pack_coord_id(unpack_coord_id(actions[0]))  # packed ids
    state = engine.new_game()
    for aid in actions:
        engine.apply_action(state, PlacementAction(unpack_coord_id(aid)))
    mirror = engine.to_python_state(state)
    replayed = {(c.q, c.r) for c, _ in mirror.board.stones}
    assert replayed == {(s["q"], s["r"]) for s in final["stones"]}


def test_stats_views(client):
    stats = client.get("/api/stats").json()
    assert {"bots", "daily", "hall_of_fame"} <= set(stats)
    by_slug = {row["slug"]: row for row in stats["bots"]}
    assert "tiny-8" in by_slug  # finished games exist from earlier tests
    entry = by_slug["tiny-8"]
    assert entry["games"] >= 1
    assert 0.0 <= entry["bot_winrate"] <= 1.0
    assert stats["daily"] and stats["daily"][0]["games"] >= 1
    assert isinstance(stats["hall_of_fame"], list)
