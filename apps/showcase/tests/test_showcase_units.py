"""Unit tests for the client-free pieces: sanitizer, rate bucket, DB layer,
payload codec, and the `.hxr` encode/decode helpers."""

from __future__ import annotations

import pytest

import hexo_engine as engine
from hexo_engine.types import AxialCoord, PlacementAction, pack_coord_id, unpack_coord_id

from showcase.app import TokenBucket, sanitize_nickname
from showcase.db import ShowcaseDB, decode_payload, encode_payload
from showcase.game import GameSession, decode_hxr_actions, encode_hxr, now_iso


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
    kwargs = dict(slug="ep1-16", run="r", epoch=1, visits=16, active_from=now_iso())
    first = db.upsert_bot(label="A", weights_sha="sha-1", **kwargs)
    # Same (slug, weights_sha): same row, refreshed label.
    assert db.upsert_bot(label="B", weights_sha="sha-1", **kwargs) == first
    # New weights under the same slug: a new identity row.
    assert db.upsert_bot(label="B", weights_sha="sha-2", **kwargs) != first
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


def test_session_finalize_result_convention():
    session = GameSession.create(
        bot_slug="tiny", bot_db_id=1, bot_label="Tiny", bot_visits=8,
        human_color=0, client_hash="h",
    )
    session.apply_human_move(0, 0)
    assert session.actions == [pack_coord_id(AxialCoord(0, 0))]

    session.finalize(termination="resign", winner=1)
    assert session.db_status == "finished"
    assert session.result == -1  # bot (color 1) beat the color-0 human

    timed_out = GameSession.create(
        bot_slug="tiny", bot_db_id=1, bot_label="Tiny", bot_visits=8,
        human_color=0, client_hash="h",
    )
    timed_out.finalize(termination="timeout", winner=None)
    assert timed_out.db_status == "abandoned"
    assert timed_out.result == 0
