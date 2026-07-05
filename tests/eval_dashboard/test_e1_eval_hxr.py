"""E1 — eval .hxr must persist real game actions (not header-only stubs).

Proof that eval games WILL populate the dashboard History: build synthetic eval
``_Game`` objects WITH a real action list, run them through the actual writer
``_write_eval_hxr`` (the exact function the live eval calls), and assert the
written .hxr decodes with ``num_records > 0`` and the actions round-trip through
the real Rust-backed ``HexoRecordFile`` codec.

Also proves the E1 hardening: a 0-record write (all games had empty .actions) is
now LOUD (warning logged) + machine-visible (``stats['games_written'] == 0``)
instead of being silently swallowed.

CPU-only, no torch, no GPU.

Run:
  PYTHONPATH=packages/shrimp/python python -m pytest tests/eval_dashboard/test_e1_eval_hxr.py
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from hexo_runner.records import HexoRecordFile

from shrimp.eval_arena import _write_eval_hxr
from shrimp.geometry import pack_action_id


class _StubGame:
    """Minimal stand-in for eval_arena._Game with exactly the attrs the writer
    reads: index, a_is_p0, seed, winner, plies, actions (packed action ids)."""

    def __init__(self, index, a_is_p0, seed, winner, actions):
        self.index = index
        self.a_is_p0 = a_is_p0
        self.seed = seed
        self.winner = winner  # "A" | "B" | None
        self.actions = list(actions)
        self.plies = len(actions)


class _ConcurrentStubGame:
    """Stand-in for the CONCURRENT play_multi_checkpoint_match._Game, which has
    NO ``index`` attribute — its game index is exposed as ``local_index`` only
    (eval_arena.py _Game __slots__ at ~1072). This is the exact shape the LIVE
    run feeds to ``_write_eval_hxr``. It must NOT expose ``index`` so this test
    locks the regression where the writer hard-referenced ``g.index`` and threw
    AttributeError on every concurrent eval game (header-only, num_records=0)."""

    __slots__ = ("local_index", "a_is_p0", "seed", "winner", "actions", "plies")

    def __init__(self, local_index, a_is_p0, seed, winner, actions):
        self.local_index = local_index
        self.a_is_p0 = a_is_p0
        self.seed = seed
        self.winner = winner  # "A" | "B" | None
        self.actions = list(actions)
        self.plies = len(actions)


def _make_actions(coords):
    return [pack_action_id(q, r) for (q, r) in coords]


def test_actions_roundtrip_to_hxr():
    diag = Path(tempfile.mkdtemp()) / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    a_coords = [(0, 0), (1, 0), (-1, 1)]
    b_coords = [(0, 0), (0, 1)]
    games = [
        _StubGame(0, a_is_p0=True, seed=11, winner="A", actions=_make_actions(a_coords)),
        _StubGame(1, a_is_p0=False, seed=11, winner="B", actions=_make_actions(b_coords)),
    ]

    stats: dict[str, int] = {}
    path = _write_eval_hxr(games, diag, "ep35", "ep30", stats=stats)
    assert path is not None, "writer returned None despite 2 games with real actions"
    assert Path(path).is_file(), f"hxr file not written: {path}"
    assert stats.get("games_written") == 2, stats
    assert stats.get("games_skipped") == 0, stats

    rf = HexoRecordFile.open(path)
    recs = list(rf.iter_records())
    assert len(recs) == 2, f"expected 2 records, got {len(recs)}"

    # Round-trip: action_ids non-empty, lengths match, game_id pattern, winner set.
    by_id = {r.game_id: r for r in recs}
    g0 = next(r for gid, r in by_id.items() if gid.endswith("g0-candP0"))
    g1 = next(r for gid, r in by_id.items() if gid.endswith("g1-candP1"))
    assert list(g0.action_ids), "game 0 has no action_ids"
    assert len(list(g0.action_ids)) == len(a_coords), "game 0 action count mismatch"
    assert len(list(g1.action_ids)) == len(b_coords), "game 1 action count mismatch"
    assert g0.game_id.startswith("ep35-ep35-vs-ep30-"), g0.game_id
    assert all(r.winner is not None for r in recs), "winner not set on decided games"
    print(f"[E1] PASS roundtrip: {len(recs)} records, g0={g0.game_id} "
          f"actions={len(list(g0.action_ids))} winner={g0.winner}")


def test_concurrent_path_game_writes_records():
    """Regression lock for the empty-.hxr bug: the CONCURRENT eval _Game (the
    shape the live run uses) exposes ``local_index`` but NO ``index``. The
    writer previously hard-referenced ``g.index`` and threw AttributeError on
    the first game, producing a header-only num_records=0 file. With the
    getattr(index)->local_index fallback the games must round-trip with the
    game index taken from ``local_index``."""
    diag = Path(tempfile.mkdtemp()) / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    a_coords = [(0, 0), (1, 0), (-1, 1)]
    b_coords = [(0, 0), (0, 1)]
    games = [
        _ConcurrentStubGame(0, a_is_p0=True, seed=7, winner="A",
                            actions=_make_actions(a_coords)),
        _ConcurrentStubGame(1, a_is_p0=False, seed=7, winner="B",
                            actions=_make_actions(b_coords)),
    ]
    # Prove the precondition: these objects truly lack ``index`` (so a bare
    # ``g.index`` reference would raise — the original bug).
    assert not hasattr(games[0], "index"), "stub must NOT expose .index"

    stats: dict[str, int] = {}
    path = _write_eval_hxr(games, diag, "ep39", "ep5", stats=stats)
    assert path is not None, "writer returned None for concurrent-path games"
    assert stats.get("games_written") == 2, stats
    assert stats.get("games_skipped") == 0, stats

    rf = HexoRecordFile.open(path)
    recs = list(rf.iter_records())
    assert len(recs) == 2, f"expected 2 records, got {len(recs)} (empty-.hxr bug)"
    by_id = {r.game_id: r for r in recs}
    # local_index 0/1 must surface in the game_id (proves the fallback path ran).
    g0 = next(r for gid, r in by_id.items() if gid.endswith("g0-candP0"))
    g1 = next(r for gid, r in by_id.items() if gid.endswith("g1-candP1"))
    assert len(list(g0.action_ids)) == len(a_coords), "g0 action count mismatch"
    assert len(list(g1.action_ids)) == len(b_coords), "g1 action count mismatch"
    assert all(r.winner is not None for r in recs), "winner not set"
    print(f"[E1] PASS concurrent-path: {len(recs)} records via local_index "
          f"g0={g0.game_id} actions={len(list(g0.action_ids))}")


def test_zero_record_is_loud():
    diag = Path(tempfile.mkdtemp()) / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    # All games have empty .actions (the exact regression that emptied live .hxr).
    games = [
        _StubGame(0, a_is_p0=True, seed=1, winner="A", actions=[]),
        _StubGame(1, a_is_p0=False, seed=1, winner="B", actions=[]),
    ]

    handler = _CaptureHandler()
    logger = logging.getLogger("shrimp.eval")
    logger.addHandler(handler)
    prev_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        stats: dict[str, int] = {}
        path = _write_eval_hxr(games, diag, "ep35", "ep30", stats=stats)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)

    assert path is None, "expected None when 0 of N games written"
    assert stats.get("games_written") == 0, stats
    assert stats.get("games_skipped") == 2, stats
    warned = [r for r in handler.records if r.levelno >= logging.WARNING]
    assert any("wrote 0 of 2 games" in r.getMessage() for r in warned), (
        "0-record write was NOT loud; warnings=" + repr([r.getMessage() for r in warned])
    )
    print(f"[E1] PASS loudness: 0-record write logged WARNING "
          f"({[r.getMessage() for r in warned]})")


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


if __name__ == "__main__":
    test_actions_roundtrip_to_hxr()
    test_concurrent_path_game_writes_records()
    test_zero_record_is_loud()
    print("E1 ALL GREEN")
