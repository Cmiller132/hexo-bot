"""DB-layer tests for list_games_filtered — filters, the sort whitelist,
offset/total paging, and the anonymous-nickname bucket. Imports ONLY
showcase.db (stdlib sqlite); never touches app/analysis/bots (they pull
torch). Runs against an in-memory ShowcaseDB built through the public API."""

from __future__ import annotations

import pytest

from showcase.db import ShowcaseDB


def _mk_bot(db, *, slug, visits, label="L", run="r", epoch=1):
    return db.upsert_bot(
        slug=slug, label=label, run=run, epoch=epoch, visits=visits,
        weights_sha=f"sha-{slug}-{visits}", active_from="2026-01-01T00:00:00",
    )


def _finish(db, *, game_id, bot_id, nickname, result, ply_count, duration_s,
            finished_at, human_color=0):
    db.create_game(
        game_id=game_id, bot_id=bot_id, human_color=human_color,
        started_at=finished_at, client_hash="h",
    )
    if nickname is not None:
        db.set_nickname(game_id, nickname)
    db.finalize_game(
        game_id=game_id, finished_at=finished_at, status="finished",
        result=result, termination="six_in_line", ply_count=ply_count,
        duration_s=duration_s, record=b"rec",
    )


@pytest.fixture()
def db():
    d = ShowcaseDB(":memory:")
    yield d
    d.close()


@pytest.fixture()
def seeded(db):
    """Two bots (main7 @16 sims, main7 @64 sims) and five finished games with a
    spread of nicknames, results, plies, and durations."""
    b16 = _mk_bot(db, slug="main7-ep70", visits=16)
    b64 = _mk_bot(db, slug="main7-ep70", visits=64)
    # id, bot, nick, result, plies, dur, when
    _finish(db, game_id="g1", bot_id=b16, nickname="alice", result=+1,
            ply_count=30, duration_s=10.0, finished_at="2026-01-01T00:01:00")
    _finish(db, game_id="g2", bot_id=b16, nickname="bob", result=-1,
            ply_count=50, duration_s=40.0, finished_at="2026-01-01T00:02:00")
    _finish(db, game_id="g3", bot_id=b64, nickname="alice", result=0,
            ply_count=10, duration_s=5.0, finished_at="2026-01-01T00:03:00")
    _finish(db, game_id="g4", bot_id=b64, nickname=None, result=-1,
            ply_count=70, duration_s=90.0, finished_at="2026-01-01T00:04:00")
    _finish(db, game_id="g5", bot_id=b64, nickname="", result=+1,
            ply_count=20, duration_s=20.0, finished_at="2026-01-01T00:05:00")
    return {"b16": b16, "b64": b64}


def _ids(rows):
    return [r["id"] for r in rows]


def test_no_filters_returns_all_recent_first(db, seeded):
    rows, total = db.list_games_filtered(limit=50)
    assert total == 5
    assert _ids(rows) == ["g5", "g4", "g3", "g2", "g1"]  # finished_at DESC
    # duration_s is carried on the filtered rows.
    assert rows[0]["duration_s"] == 20.0
    assert rows[0]["bot_run"] == "r"


def test_filter_nickname_exact(db, seeded):
    rows, total = db.list_games_filtered(nickname="alice", limit=50)
    assert total == 2
    assert set(_ids(rows)) == {"g1", "g3"}


def test_filter_nickname_anonymous_bucket(db, seeded):
    """nickname='' matches NULL and empty-string rows together."""
    rows, total = db.list_games_filtered(nickname="", limit=50)
    assert total == 2
    assert set(_ids(rows)) == {"g4", "g5"}


def test_filter_checkpoint(db, seeded):
    rows, total = db.list_games_filtered(checkpoint_id="main7-ep70", limit=50)
    assert total == 5  # both sims share the slug


def test_filter_sims(db, seeded):
    rows, total = db.list_games_filtered(sims=64, limit=50)
    assert total == 3
    assert set(_ids(rows)) == {"g3", "g4", "g5"}


def test_filter_result_each_bucket(db, seeded):
    win, _ = db.list_games_filtered(result="win", limit=50)
    loss, _ = db.list_games_filtered(result="loss", limit=50)
    draw, _ = db.list_games_filtered(result="draw", limit=50)
    assert set(_ids(win)) == {"g1", "g5"}    # result == +1
    assert set(_ids(loss)) == {"g2", "g4"}   # result == -1
    assert set(_ids(draw)) == {"g3"}         # result == 0


def test_combined_filters_and(db, seeded):
    rows, total = db.list_games_filtered(sims=64, result="loss", limit=50)
    assert total == 1 and _ids(rows) == ["g4"]


def test_sort_whitelist_orderings(db, seeded):
    assert _ids(db.list_games_filtered(sort="recent", limit=50)[0]) == \
        ["g5", "g4", "g3", "g2", "g1"]
    assert _ids(db.list_games_filtered(sort="oldest", limit=50)[0]) == \
        ["g1", "g2", "g3", "g4", "g5"]
    # longest / shortest by ply_count (g4=70 > g2=50 > g1=30 > g5=20 > g3=10).
    assert _ids(db.list_games_filtered(sort="longest", limit=50)[0]) == \
        ["g4", "g2", "g1", "g5", "g3"]
    assert _ids(db.list_games_filtered(sort="shortest", limit=50)[0]) == \
        ["g3", "g5", "g1", "g2", "g4"]
    # slowest / fastest by duration_s (g4=90 > g2=40 > g5=20 > g1=10 > g3=5).
    assert _ids(db.list_games_filtered(sort="slowest", limit=50)[0]) == \
        ["g4", "g2", "g5", "g1", "g3"]
    assert _ids(db.list_games_filtered(sort="fastest", limit=50)[0]) == \
        ["g3", "g1", "g5", "g2", "g4"]


def test_unknown_sort_falls_back_to_recent(db, seeded):
    rows, _ = db.list_games_filtered(sort="'; DROP TABLE games;--", limit=50)
    assert _ids(rows) == ["g5", "g4", "g3", "g2", "g1"]  # == recent


def test_offset_and_total_paging(db, seeded):
    page1, total = db.list_games_filtered(sort="oldest", limit=2, offset=0)
    page2, total2 = db.list_games_filtered(sort="oldest", limit=2, offset=2)
    page3, total3 = db.list_games_filtered(sort="oldest", limit=2, offset=4)
    assert total == total2 == total3 == 5
    assert _ids(page1) == ["g1", "g2"]
    assert _ids(page2) == ["g3", "g4"]
    assert _ids(page3) == ["g5"]  # last, partial page


def test_active_games_excluded(db, seeded):
    """A still-active game never appears in the finished feed."""
    b = seeded["b16"]
    db.create_game(game_id="live", bot_id=b, human_color=0,
                   started_at="2026-01-01T01:00:00", client_hash="h")
    _, total = db.list_games_filtered(limit=50)
    assert total == 5  # unchanged


def test_elo_helpers_shape(db, seeded):
    """finished_games_for_elo / bots_index / finished_count line up with the
    ELO fold's contract."""
    assert db.finished_count() == 5
    rows = db.finished_games_for_elo()
    assert [r["id"] for r in rows] == ["g1", "g2", "g3", "g4", "g5"]  # chrono
    assert set(rows[0].keys()) == {"id", "bot_id", "result", "nickname", "finished_at"}
    idx = db.bots_index()
    assert idx[seeded["b64"]]["sims"] == 64
    assert idx[seeded["b64"]]["checkpoint_id"] == "main7-ep70"
