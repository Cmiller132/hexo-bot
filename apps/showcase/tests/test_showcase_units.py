"""Unit tests for the client-free pieces: sanitizer, rate bucket, DB layer,
payload codec, and the `.hxr` encode/decode helpers."""

from __future__ import annotations

import pytest

import hexo_engine as engine
from hexo_engine.types import AxialCoord, PlacementAction, pack_coord_id, unpack_coord_id

from showcase.app import TokenBucket, sanitize_nickname
from showcase.db import ShowcaseDB, decode_payload, encode_payload
from showcase.game import (
    GameSession,
    decode_hxr_actions,
    encode_hxr,
    find_winning_line,
    now_iso,
)

# A scripted six_in_line game: p0 builds (0,0)..(5,0) along the q axis while
# p1 plays non-line cells. Turn shape: p0 opening single, then two per turn.
SIX_IN_LINE_MOVES = [
    (0, 0),                    # p0 opening (forced)
    (0, 2), (1, 2),            # p1
    (1, 0), (2, 0),            # p0
    (2, 2), (3, 2),            # p1
    (3, 0), (4, 0),            # p0
    (0, 3), (1, 3),            # p1
    (5, 0),                    # p0 completes six -> terminal mid-turn
]


def drive_six_in_line_session() -> GameSession:
    """A GameSession driven from both sides to a p0 six_in_line win."""
    session = GameSession.create(
        bot_slug="tiny", bot_db_id=1, bot_label="Tiny", bot_epoch=0, sims=8,
        human_color=0, client_hash="h",
    )
    for q, r in SIX_IN_LINE_MOVES:
        session.apply_human_move(q, r)
    assert session.engine_winner() == 0
    session.finalize(termination="six_in_line", winner=0)
    return session


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Bob", "Bob"),
        ("  spaced   out  ", "spaced out"),
        ("<b>tags</b>&entities;", "btagsbentities"),
        ("dots.and-dashes_ok", "dots.and-dashes_ok"),
        ("x" * 40, "x" * 24),
        ("@!#$%^", None),
        ("   ", None),
    ],
)
def test_sanitize_nickname(raw, expected):
    assert sanitize_nickname(raw) == expected


def test_token_bucket_burst_then_deny():
    bucket = TokenBucket(per_minute=60, burst=3)
    assert [bucket.allow("k") for _ in range(4)] == [True, True, True, False]
    assert bucket.allow("other-key") is True  # independent keys


def test_payload_codec_roundtrip():
    payload = {"value": -0.25, "policy": [{"q": 0, "r": 0, "p": 1.0}], "ply": 3}
    assert decode_payload(encode_payload(payload)) == payload


def test_db_bot_upsert_identity(tmp_path):
    db = ShowcaseDB(tmp_path / "t.db")
    kwargs = dict(slug="ep1", run="r", epoch=1, active_from=now_iso())
    first = db.upsert_bot(label="A", weights_sha="sha-1", visits=16, **kwargs)
    # Same (slug, weights_sha, visits): same row, refreshed label.
    assert db.upsert_bot(label="B", weights_sha="sha-1", visits=16, **kwargs) == first
    # Another sims budget for the same weights: its own identity row.
    second = db.upsert_bot(label="B", weights_sha="sha-1", visits=64, **kwargs)
    assert second != first
    # New weights under the same slug: a new identity row too.
    assert db.upsert_bot(label="B", weights_sha="sha-2", visits=16, **kwargs) not in (first, second)
    assert db.get_bot(first)["visits"] == 16
    assert db.get_bot(second)["visits"] == 64
    db.close()


def test_db_abandon_stale_active(tmp_path):
    db = ShowcaseDB(tmp_path / "t.db")
    bot_id = db.upsert_bot(
        slug="s", label="l", run="r", epoch=0, visits=8,
        weights_sha="sha", active_from=now_iso(),
    )
    db.create_game(
        game_id="g1", bot_id=bot_id, human_color=0,
        started_at=now_iso(), client_hash="h",
    )
    assert db.abandon_stale_active(now_iso()) == 1
    assert db.get_game("g1")["status"] == "abandoned"
    assert db.abandon_stale_active(now_iso()) == 0
    db.close()


def _legal_prefix(n: int) -> list[int]:
    state = engine.new_game()
    actions = []
    for _ in range(n):
        aid = int(engine.legal_action_ids(state)[0])
        engine.apply_action(state, PlacementAction(unpack_coord_id(aid)))
        actions.append(aid)
    return actions


def test_hxr_roundtrip_completed_and_aborted():
    actions = _legal_prefix(5)
    completed = encode_hxr(
        game_id="g-done", bot_slug="tiny", human_color=0, action_ids=actions,
        winner=1, termination="resign", seed=7,
    )
    assert decode_hxr_actions(completed) == actions

    aborted = encode_hxr(
        game_id="g-idle", bot_slug="tiny", human_color=1, action_ids=actions[:3],
        winner=None, termination="timeout", seed=7,
    )
    assert decode_hxr_actions(aborted) == actions[:3]


def test_find_winning_line_exact_six_and_joined_run():
    # Exact six along the r axis, last stone at the far end.
    stones = [{"q": 0, "r": i, "color": 0} for i in range(6)]
    line = find_winning_line(stones, winner=0)
    assert line == [{"q": 0, "r": i} for i in range(6)]

    # Placement into a gap joins two runs: the whole 7-run is the line.
    cells = [(0, 0), (1, 0), (2, 0), (4, 0), (5, 0), (6, 0), (3, 0)]  # (3,0) last
    stones = [{"q": q, "r": r, "color": 1} for q, r in cells]
    stones.insert(3, {"q": 9, "r": 9, "color": 0})  # loser stone is ignored
    line = find_winning_line(stones, winner=1)
    assert line == [{"q": q, "r": 0} for q in range(7)]

    # No six anywhere -> None.
    assert find_winning_line([{"q": 0, "r": 0, "color": 0}], winner=0) is None


def test_six_in_line_session_snapshot():
    """A driven six_in_line game exposes the winning line and placement-order
    stones in its snapshot."""
    session = drive_six_in_line_session()
    snap = session.snapshot()
    assert snap["status"] == "finished"
    assert snap["result"] == {"winner": 0, "termination": "six_in_line", "human_result": 1}
    assert snap["winning_line"] == [{"q": q, "r": 0} for q in range(6)]
    # stones is in placement order (the client derives last-two from the tail).
    assert [(s["q"], s["r"]) for s in snap["stones"]] == SIX_IN_LINE_MOVES
    assert snap["last_move"] == {"q": 5, "r": 0, "color": 0}
    assert snap["to_move"] is None
    assert session.db_status == "finished"
    assert session.result == 1


def test_session_finalize_result_convention():
    session = GameSession.create(
        bot_slug="tiny", bot_db_id=1, bot_label="Tiny", bot_epoch=0, sims=8,
        human_color=0, client_hash="h",
    )
    session.apply_human_move(0, 0)
    assert session.actions == [pack_coord_id(AxialCoord(0, 0))]

    session.finalize(termination="resign", winner=1)
    assert session.db_status == "finished"
    assert session.result == -1  # bot (color 1) beat the color-0 human
    assert session.winning_line is None  # resignation completes no line

    timed_out = GameSession.create(
        bot_slug="tiny", bot_db_id=1, bot_label="Tiny", bot_epoch=0, sims=8,
        human_color=0, client_hash="h",
    )
    timed_out.finalize(termination="timeout", winner=None)
    assert timed_out.db_status == "abandoned"
    assert timed_out.result == 0
