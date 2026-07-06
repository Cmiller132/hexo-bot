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


def create_game(
    client, headers, checkpoint_id: str = "tiny", sims: int = 8,
    human_color: int | str = 0,
) -> dict:
    resp = client.post(
        "/api/game",
        json={"checkpoint_id": checkpoint_id, "sims": sims, "human_color": human_color},
        headers=headers,
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
    assert health["checkpoints"] == 2

    bots = client.get("/api/bots").json()
    assert bots["sims"] == [8, 16]
    assert len(bots["checkpoints"]) == 2
    by_id = {entry["id"]: entry for entry in bots["checkpoints"]}

    entry = by_id["tiny"]
    assert entry["run"] == "showcase_tiny_test"
    assert entry["epoch"] == 0
    assert entry["label"]
    assert entry["games_trained"] == 12345  # passthrough display metadata
    assert "group" not in entry and "search" not in entry

    # group/search ride the same metadata passthrough; search_profile is a
    # server-side key and never reaches the API.
    legacy = by_id["tiny-puct"]
    assert legacy["group"] == "earlier runs"
    assert legacy["search"] == "puct"
    assert "search_profile" not in legacy


def test_catalogue_times_sims_creates_bot_rows_lazily(client, settings):
    """One bots row per PLAYED (checkpoint, sims) combination, and the stats
    views keep per-strength granularity."""
    headers = fresh_ip()
    for sims in (8, 16):
        snap = create_game(client, headers, sims=sims)
        assert snap["bot"] == {
            "checkpoint_id": "tiny", "label": "Tiny test bot", "epoch": 0, "sims": sims,
        }
        resign(client, snap["id"], headers)

    conn = sqlite3.connect(settings.db_path)
    rows = conn.execute(
        "SELECT weights_sha, visits, COUNT(*) FROM bots WHERE slug = 'tiny'"
        " GROUP BY weights_sha, visits"
    ).fetchall()
    assert all(count == 1 for _, _, count in rows)  # one row per combination
    assert {visits for _, visits, _ in rows} >= {8, 16}

    stats = client.get("/api/stats").json()
    by_visits = {row["visits"]: row for row in stats["bots"] if row["slug"] == "tiny"}
    assert {8, 16} <= set(by_visits)  # the views split stats per strength


def test_invalid_sims_is_422_and_unknown_checkpoint_404(client):
    headers = fresh_ip()
    resp = client.post(
        "/api/game", json={"checkpoint_id": "tiny", "sims": 7, "human_color": 0},
        headers=headers,
    )
    assert resp.status_code == 422
    assert "sims" in resp.json()["detail"]
    resp = client.post(
        "/api/game", json={"checkpoint_id": "nope", "sims": 8, "human_color": 0},
        headers=headers,
    )
    assert resp.status_code == 404


def test_full_game_to_terminal_or_20_plies(client):
    headers = fresh_ip()
    snap = create_game(client, headers)
    game_id = snap["id"]
    # The forced opening single is pre-placed at creation; the bot (player1)
    # owns the first real turn when the human is player0.
    assert snap["status"] == "bot_thinking"
    assert snap["phase"] == "FirstStone"
    assert snap["ply"] == 1
    assert snap["stones"] == [{"q": 0, "r": 0, "color": 0}]

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


def test_no_bot_opening_search_when_human_is_player1(client):
    """The pre-placed origin single is player0's whole turn, so a human
    playing player1 gets the move immediately — the bot never searches the
    forced opening."""
    headers = fresh_ip()
    snap = create_game(client, headers, human_color=1)
    assert snap["human_color"] == 1
    assert snap["status"] == "your_turn"  # no bot turn was enqueued
    assert snap["ply"] == 1  # the pre-placed opening single
    assert snap["stones"] == [{"q": 0, "r": 0, "color": 0}]
    assert snap["to_move"] == 1
    assert snap["legal"]
    resign(client, snap["id"], headers)


def test_random_human_color_resolves_and_echoes(client, settings):
    """human_color="random" resolves server-side to a fair 0/1, echoed in the
    snapshot and recorded in the games row; junk values are rejected."""
    seen: set[int] = set()
    for i in range(16):
        headers = fresh_ip()
        snap = create_game(client, headers, human_color="random")
        assert snap["human_color"] in (0, 1)
        if i == 0:  # the DB row records the resolved color, not "random"
            row = sqlite3.connect(settings.db_path).execute(
                "SELECT human_color FROM games WHERE id = ?", (snap["id"],)
            ).fetchone()
            assert row[0] == snap["human_color"]
        seen.add(snap["human_color"])
        resign(client, snap["id"], headers)
        if seen == {0, 1} and i >= 1:
            break
    assert seen == {0, 1}  # 16 coin flips landing one-sided: ~0.003%

    resp = client.post(
        "/api/game",
        json={"checkpoint_id": "tiny", "sims": 8, "human_color": "coin"},
        headers=fresh_ip(),
    )
    assert resp.status_code == 422  # only 0 | 1 | "random"


def test_full_game_as_player2_to_terminal_or_10_plies(client):
    """A whole game with the human as player 1 (second to act): the opening
    is pre-placed, the human moves first, turn alternation stays consistent,
    and the game reaches a terminal state or a clean resignation after ~10
    plies."""
    headers = fresh_ip()
    snap = create_game(client, headers, human_color=1)
    game_id = snap["id"]
    assert snap["status"] == "your_turn"
    assert snap["ply"] == 1

    while snap["status"] != "finished" and snap["ply"] < 10:
        if snap["status"] == "your_turn":
            assert snap["to_move"] == snap["human_color"] == 1
            cell = snap["legal"][(snap["ply"] * 7) % len(snap["legal"])]
            resp = client.post(f"/api/game/{game_id}/move", json=cell, headers=headers)
            assert resp.status_code == 200, resp.text
            snap = resp.json()
            assert snap["last_move"] == {**cell, "color": 1}
            assert len(snap["stones"]) == snap["ply"]
        else:
            snap = poll_until(client, game_id)

    if snap["status"] != "finished":
        snap = resign(client, game_id, headers)
    assert snap["status"] == "finished"
    assert snap["result"]["termination"] in ("six_in_line", "resign")
    assert len(snap["stones"]) == snap["ply"]


def test_illegal_move_rejected(client):
    headers = fresh_ip()
    snap = create_game(client, headers)
    snap = poll_until(client, snap["id"])  # the bot's player1 turn lands first
    # The origin is occupied by the pre-placed opening stone.
    resp = client.post(f"/api/game/{snap['id']}/move", json={"q": 0, "r": 0}, headers=headers)
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
    resp = client.post(
        "/api/game", json={"checkpoint_id": "tiny", "sims": 8, "human_color": 0},
        headers=headers,
    )
    assert resp.status_code == 429
    resign(client, first["id"], headers)
    third = create_game(client, headers)  # capacity freed by the resignation
    resign(client, second["id"], headers)
    resign(client, third["id"], headers)


def test_resign_scores_for_the_bot(client):
    headers = fresh_ip()
    snap = create_game(client, headers)
    poll_until(client, snap["id"])  # let the bot's opening-reply turn land
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


def test_unknown_game_is_404(client):
    assert client.get("/api/game/no-such-game", headers=fresh_ip()).status_code == 404


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
    snap = poll_until(client, game_id)  # the bot's player1 turn
    if snap["status"] == "your_turn":
        cell = snap["legal"][0]
        client.post(f"/api/game/{game_id}/move", json=cell, headers=headers)
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


def test_active_game_gated_finished_game_public(client):
    """Active game state is owner-only (403 without the cookie); finished
    games are public — both from the live session and from the DB after the
    session is evicted."""
    client.cookies.clear()  # a clean jar so the token below is unambiguous
    headers = fresh_ip()
    snap = create_game(client, headers)
    game_id = snap["id"]
    token = client.cookies.get(_COOKIE)

    client.cookies.clear()
    assert client.get(f"/api/game/{game_id}", headers=headers).status_code == 403
    client.cookies.set(_COOKIE, token)
    assert client.get(f"/api/game/{game_id}", headers=headers).status_code == 200

    resign(client, game_id, headers)
    client.cookies.clear()
    public = client.get(f"/api/game/{game_id}", headers=fresh_ip())
    assert public.status_code == 200  # finished: no cookie needed
    from_session = public.json()
    assert from_session["status"] == "finished"

    # Evict the in-memory session; the DB record now serves the read.
    client.app.state.sessions.pop(game_id)
    from_db = client.get(f"/api/game/{game_id}", headers=fresh_ip())
    assert from_db.status_code == 200
    from_db = from_db.json()
    for key in ("id", "status", "bot", "human_color", "ply", "stones", "last_move",
                "winning_line", "result", "nickname"):
        assert from_db[key] == from_session[key], key
    assert from_db["legal"] == []
    assert from_db["finished_at"]
    client.cookies.set(_COOKIE, token)


def test_recent_games_feed_pagination(client):
    """/api/games: finished-only, newest first, stable keyset pagination."""
    headers = fresh_ip()
    ids = []
    for _ in range(3):
        snap = create_game(client, headers)
        resign(client, snap["id"], headers)
        ids.append(snap["id"])
        if len(ids) == 2:  # free a per-IP active slot before the third game
            headers = fresh_ip()
    active = create_game(client, fresh_ip())  # must NOT appear in the feed

    client.cookies.clear()  # the feed is public
    first = client.get("/api/games", params={"limit": 2}, headers=fresh_ip())
    assert first.status_code == 200
    page1 = first.json()
    assert len(page1["games"]) == 2
    assert page1["next"]
    feed_ids = [g["id"] for g in page1["games"]]

    item = page1["games"][0]
    assert set(item) == {"id", "bot", "human_color", "result", "ply_count",
                         "finished_at", "nickname"}
    assert set(item["bot"]) == {"checkpoint_id", "label", "epoch", "sims"}
    assert item["result"]["termination"] in ("six_in_line", "resign")
    assert item["result"]["human_result"] in (-1, 1)

    seen = set(feed_ids)
    cursor = page1["next"]
    stamps = [g["finished_at"] for g in page1["games"]]
    while cursor:
        page = client.get(
            "/api/games", params={"limit": 2, "before": cursor}, headers=fresh_ip()
        ).json()
        for game in page["games"]:
            assert game["id"] not in seen  # no duplicates across pages
            seen.add(game["id"])
            stamps.append(game["finished_at"])
        cursor = page["next"]
    assert stamps == sorted(stamps, reverse=True)  # newest first throughout
    assert set(ids) <= seen  # nothing skipped (same-second ties included)
    assert active["id"] not in seen  # active games are absent

    assert client.get(
        "/api/games", params={"before": "garbage-no-separator"}, headers=fresh_ip()
    ).status_code == 422


def test_stats_views(client):
    stats = client.get("/api/stats").json()
    assert {"bots", "daily", "hall_of_fame"} <= set(stats)
    tiny_rows = [row for row in stats["bots"] if row["slug"] == "tiny"]
    assert tiny_rows  # finished games exist from earlier tests
    for entry in tiny_rows:
        assert entry["visits"] in (8, 16)
        assert entry["games"] >= 1
        assert 0.0 <= entry["bot_winrate"] <= 1.0
    assert stats["daily"] and stats["daily"][0]["games"] >= 1
    assert isinstance(stats["hall_of_fame"], list)
