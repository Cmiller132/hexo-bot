"""Restart-graceful diagnostics + selfplay telemetry (PART 1-3 of the overhaul).

Three concerns, all pure/IO-on-tmp (no GPU, no live run, no run_continuous):

1. Merge logic (``_merge_epoch_diag``): two synthetic segments merge to correct
   additive sums, key-wise scheduler int merge, weighted means, approx flags,
   and a capped ``segments`` trail across repeated crashes.
2. Skip-path preservation (``_skip_path_result``): a non-trivial prior diag is
   preserved with a bumped ``resumed_skip_count`` instead of being zeroed (the
   epoch-13 incident); a missing/trivial prior yields a marked skip record.
3. New telemetry: the driver's ``stats()`` and epoch-end aggregation surface the
   new keys with correct values on a small synthetic driver state, the derived
   scheduler rates guard div-by-zero, and the one-line summary formats.

The instrumentation is additive: every existing top-level diagnostic key name
and type is preserved, which these tests pin.
"""

from __future__ import annotations

import json

import pytest

from shrimp.selfplay import (
    ContinuousDriver,
    _atomic_write_json,
    _derive_scheduler_rates,
    _format_epoch_summary,
    _load_prior_diag,
    _merge_epoch_diag,
    _phase_of,
    _skip_path_result,
)


# --- helpers ------------------------------------------------------------------


def _segment(**overrides):
    """A full-shaped epoch diagnostic segment with sensible defaults; override
    any field for the specific case under test."""
    base = {
        "status": "completed",
        "epoch": 7,
        "elapsed_seconds": 100.0,
        "search_visits": 200,
        "scheduler": {
            "moves_decided": 1000,
            "full_moves": 330,
            "fast_moves": 500,
            "init_moves": 170,
            "lcb_overrides": 40,
            "gumbel_play_moves": 300,
            "gumbel_play_winner_moves": 210,
            "gumbel_play_moves_early": 100,
            "gumbel_play_winner_early": 60,
        },
        "games_started": 128,
        "games_finished": 128,
        "truncated_games": 2,
        "rows_written": 11000,
        "total_decisions": 1000,
        "full_decisions": 330,
        "mean_game_length": 80.0,
        "p90_game_length": 140.0,
        "game_length_p10": 40.0,
        "game_length_p50": 78.0,
        "game_length_p90": 140.0,
        "game_length_max": 200.0,
        "root_policy_entropy_mean": 2.6,
        "root_policy_entropy_by_phase": {
            "opening": {"mean": 3.0, "n": 100},
            "mid": {"mean": 2.5, "n": 150},
            "late": {"mean": 1.9, "n": 80},
        },
        "root_value_mean": 0.10,
        "root_value_abs_mean": 0.40,
        "root_value_std": 0.5,
        "root_value_by_phase": {
            "opening": {"mean": 0.05, "n": 200},
            "mid": {"mean": 0.15, "n": 500},
            "late": {"mean": 0.30, "n": 300},
        },
        "decided_fraction": 0.20,
        "policy_surprise_mean": 0.05,
        "policy_surprise_p90": 0.12,
        "policy_surprise_max": 0.40,
        "wins_by_player": {"0": 66, "1": 60},
        "unique_openings_10ply": 120,
        "unique_openings": {"10": 120, "16": 125, "20": 127},
    }
    base.update(overrides)
    return base


# --- PART 1a: skip-path preservation ------------------------------------------


def test_skip_path_preserves_nontrivial_prior():
    """A completed epoch's real diag (games_finished>0) survives the skip path
    verbatim, gaining only a resumed_skip annotation — never zeroed."""
    prior = _segment(games_finished=128, rows_written=11000)
    skip = {
        "status": "completed", "epoch": 7, "elapsed_seconds": 0.0,
        "search_visits": 200, "scheduler": {},
        "resumed_existing_games": 128,
        "games_started": 0, "games_finished": 0, "truncated_games": 0,
        "rows_written": 0, "root_policy_entropy_mean": None,
    }
    out = _skip_path_result(prior, skip)

    # Real content preserved.
    assert out["games_finished"] == 128
    assert out["rows_written"] == 11000
    assert out["root_policy_entropy_mean"] == 2.6
    assert out["scheduler"]["moves_decided"] == 1000
    # Annotation added.
    assert out["resumed_skip"] is True
    assert out["resumed_skip_count"] == 1
    assert "prior_diag_missing" not in out


def test_skip_path_bumps_resumed_count_on_repeat():
    """A second skip over an already-annotated diag bumps the counter."""
    prior = _segment(resumed_skip=True, resumed_skip_count=1)
    out = _skip_path_result(prior, {"scheduler": {}, "games_finished": 0})
    assert out["resumed_skip_count"] == 2
    assert out["games_finished"] == prior["games_finished"]


def test_skip_path_marks_missing_prior():
    """No prior diag -> the skip record is written but flagged so the loss is
    visible downstream."""
    skip = {
        "status": "completed", "epoch": 7, "scheduler": {},
        "games_finished": 0, "rows_written": 0,
    }
    out = _skip_path_result(None, skip)
    assert out["resumed_skip"] is True
    assert out["prior_diag_missing"] is True
    assert out["games_finished"] == 0


def test_skip_path_trivial_prior_is_not_preserved():
    """A prior diag that itself recorded nothing (games_finished=0, empty
    scheduler) is trivial and does not count as content to preserve."""
    trivial = {"status": "completed", "epoch": 7, "games_finished": 0, "scheduler": {}}
    skip = {"scheduler": {}, "games_finished": 0}
    out = _skip_path_result(trivial, skip)
    assert out["prior_diag_missing"] is True


# --- PART 1b: merge logic -----------------------------------------------------


def test_merge_additive_counters_and_scheduler_ints():
    """Additive top-level counters and every integer scheduler counter sum
    key-wise across two segments."""
    a = _segment(games_started=100, games_finished=100, truncated_games=1,
                 rows_written=9000, total_decisions=800, full_decisions=260)
    b = _segment(games_started=28, games_finished=28, truncated_games=1,
                 rows_written=2000, total_decisions=200, full_decisions=70)
    m = _merge_epoch_diag([a, b])

    assert m["games_started"] == 128
    assert m["games_finished"] == 128
    assert m["truncated_games"] == 2
    assert m["rows_written"] == 11000
    assert m["total_decisions"] == 1000
    assert m["full_decisions"] == 330
    assert m["elapsed_seconds"] == pytest.approx(200.0)
    # Scheduler ints summed key-wise.
    assert m["scheduler"]["moves_decided"] == 2000
    assert m["scheduler"]["lcb_overrides"] == 80
    assert m["scheduler"]["gumbel_play_winner_moves"] == 420


def test_merge_weighted_means():
    """Entropy is full_decision-weighted; value is total_decision-weighted."""
    a = _segment(full_decisions=100, total_decisions=100,
                 root_policy_entropy_mean=3.0, root_value_mean=0.0)
    b = _segment(full_decisions=300, total_decisions=300,
                 root_policy_entropy_mean=2.0, root_value_mean=0.4)
    m = _merge_epoch_diag([a, b])

    # (3.0*100 + 2.0*300) / 400 = 2.25
    assert m["root_policy_entropy_mean"] == pytest.approx(2.25)
    # (0.0*100 + 0.4*300) / 400 = 0.30
    assert m["root_value_mean"] == pytest.approx(0.30)


def test_merge_phase_means_weighted_by_n():
    """Per-phase means recombine weighted by each segment's own phase n, and the
    n's sum."""
    a = _segment(root_value_by_phase={
        "opening": {"mean": 0.0, "n": 100},
        "mid": {"mean": 0.2, "n": 100},
        "late": {"mean": 0.5, "n": 0},
    })
    b = _segment(root_value_by_phase={
        "opening": {"mean": 0.4, "n": 300},
        "mid": {"mean": 0.6, "n": 100},
        "late": {"mean": 0.9, "n": 50},
    })
    m = _merge_epoch_diag([a, b])
    op = m["root_value_by_phase"]["opening"]
    assert op["n"] == 400
    assert op["mean"] == pytest.approx((0.0 * 100 + 0.4 * 300) / 400)
    # late: only b contributes (a had n=0).
    assert m["root_value_by_phase"]["late"]["n"] == 50
    assert m["root_value_by_phase"]["late"]["mean"] == pytest.approx(0.9)


def test_merge_approx_flags_and_unique_openings_sum():
    """Percentile/std/unique-opening merges set merged_approx; unique openings
    report the SUM."""
    a = _segment(unique_openings_10ply=100,
                 unique_openings={"10": 100, "16": 104, "20": 106})
    b = _segment(unique_openings_10ply=28,
                 unique_openings={"10": 28, "16": 29, "20": 30})
    m = _merge_epoch_diag([a, b])

    assert m["merged_approx"] is True
    assert m["unique_openings_10ply"] == 128
    assert m["unique_openings"] == {"10": 128, "16": 133, "20": 136}
    # Winner tally is exact (still summed key-wise).
    assert m["wins_by_player"] == {"0": 132, "1": 120}
    # max is exact under merge.
    assert m["game_length_max"] == 200.0
    assert m["policy_surprise_max"] == 0.40


def test_merge_preserves_existing_keys_and_type():
    """Every existing top-level key name/type survives the merge; merged diags
    additionally carry segments + merged_approx."""
    a = _segment()
    b = _segment()
    m = _merge_epoch_diag([a, b])
    for key in a:
        assert key in m, f"merge dropped existing key {key!r}"
    assert type(m["wins_by_player"]) is dict
    assert type(m["unique_openings"]) is dict
    assert isinstance(m["status"], str)
    assert "segments" in m and isinstance(m["segments"], list)
    assert "merged_approx" in m


def test_merge_passthrough_single_segment_only_keys():
    """A key present in only one segment passes through from the last segment
    that has it."""
    a = _segment(perf_trace={"gpu_busy_s": 12.3})
    b = _segment()
    m = _merge_epoch_diag([a, b])
    assert m["perf_trace"] == {"gpu_busy_s": 12.3}


def test_merge_segments_capped():
    """Across many crash-resumes the stored raw segments are capped, while the
    top-level counters keep summing correctly."""
    # Simulate repeated re-merges: each round folds the running merged diag (its
    # top level is the true aggregate) with a fresh segment, as generate_selfplay
    # does. 12 rounds > the cap of 8.
    running = _segment(games_finished=10, games_started=10, rows_written=1000)
    for _ in range(11):
        seg = _segment(games_finished=10, games_started=10, rows_written=1000)
        # emulate the driver's "prior top-level as one segment" fold
        prior_subs = running.get("segments")
        merged = _merge_epoch_diag([{k: v for k, v in running.items()
                                     if k not in ("segments", "merged_approx")}, seg])
        if isinstance(prior_subs, list) and prior_subs:
            merged["segments"] = (prior_subs + merged["segments"])[-8:]
        running = merged

    assert len(running["segments"]) <= 8
    # 12 folds of 10 finished each = 120.
    assert running["games_finished"] == 120
    assert running["rows_written"] == 12000


# --- PART 1c: atomic writes + prior load --------------------------------------


def test_atomic_write_and_load_roundtrip(tmp_path):
    """_atomic_write_json writes via tmp+replace (no leftover tmp) and
    _load_prior_diag reads it back."""
    path = tmp_path / "shrimp.selfplay.epoch_000007.json"
    payload = _segment()
    _atomic_write_json(path, payload)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp")), "left a .tmp behind"
    back = _load_prior_diag(path)
    assert back["games_finished"] == 128
    # matches a direct json load
    assert back == json.loads(path.read_text(encoding="utf-8"))


def test_load_prior_diag_missing_and_corrupt(tmp_path):
    """A missing or garbage prior diag reads back as None (treated as absent)."""
    assert _load_prior_diag(tmp_path / "nope.json") is None
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert _load_prior_diag(corrupt) is None


# --- PART 2: new telemetry via the driver -------------------------------------


def _driver():
    return ContinuousDriver(epoch=7, games_target=4, max_plies=200, out_dir=None)


def test_phase_of_boundaries():
    assert _phase_of(0) == "opening"
    assert _phase_of(19) == "opening"
    assert _phase_of(20) == "mid"
    assert _phase_of(59) == "mid"
    assert _phase_of(60) == "late"
    assert _phase_of(300) == "late"


def test_driver_stats_new_keys_and_backcompat():
    """Poke the driver's aggregation lists directly and assert the new stats()
    keys compute correctly while every legacy key keeps its name/type."""
    d = _driver()
    d.games_started = 4
    d.games_finished = 4
    d.games_truncated = 1
    d.rows_written = 300
    d.decisions = 10
    d.full_decisions = 6
    d.game_lengths = [10, 30, 70, 130]
    # (ply, entropy) across all three phases.
    d.policy_entropies = [(5, 3.0), (10, 3.2), (30, 2.5), (70, 1.9), (65, 1.7), (15, 3.1)]
    # (ply, root_value): decided (|v|>0.8) in two of them.
    d.root_values = [(5, 0.1), (25, -0.9), (70, 0.95), (10, -0.2)]
    d.policy_surprises = [0.01, 0.05, 0.2, 0.5]
    d.wins_by_player = {0: 2, 1: 1}
    d.opening_lines = {("a",), ("b",), ("c",)}
    d.opening_lines_16 = {("a",), ("b",), ("c",), ("d",)}
    d.opening_lines_20 = {("a",), ("b",), ("c",), ("d",), ("e",)}

    s = d.stats()

    # Legacy keys unchanged in name/type.
    assert s["mean_game_length"] == pytest.approx(60.0)
    assert isinstance(s["p90_game_length"], float)
    assert s["unique_openings_10ply"] == 3
    assert s["root_policy_entropy_mean"] == pytest.approx(
        (3.0 + 3.2 + 2.5 + 1.9 + 1.7 + 3.1) / 6
    )
    assert s["root_value_mean"] == pytest.approx((0.1 - 0.9 + 0.95 - 0.2) / 4)

    # New: unique openings extended, legacy key retained.
    assert s["unique_openings"] == {"10": 3, "16": 4, "20": 5}

    # New: per-phase entropy over full decisions.
    ent = s["root_policy_entropy_by_phase"]
    assert ent["opening"]["n"] == 3  # plies 5,10,15
    assert ent["opening"]["mean"] == pytest.approx((3.0 + 3.2 + 3.1) / 3)
    assert ent["mid"]["n"] == 1 and ent["mid"]["mean"] == pytest.approx(2.5)
    assert ent["late"]["n"] == 2 and ent["late"]["mean"] == pytest.approx((1.9 + 1.7) / 2)

    # New: root value stats.
    assert s["root_value_abs_mean"] == pytest.approx((0.1 + 0.9 + 0.95 + 0.2) / 4)
    assert s["root_value_std"] >= 0.0
    assert s["decided_fraction"] == pytest.approx(2 / 4)  # -0.9 and 0.95
    val = s["root_value_by_phase"]
    assert val["opening"]["n"] == 2  # plies 5,10
    assert val["mid"]["n"] == 1 and val["mid"]["mean"] == pytest.approx(-0.9)
    assert val["late"]["n"] == 1 and val["late"]["mean"] == pytest.approx(0.95)

    # New: winner balance.
    assert s["wins_by_player"] == {"0": 2, "1": 1}

    # New: policy-surprise stats.
    assert s["policy_surprise_mean"] == pytest.approx(sum([0.01, 0.05, 0.2, 0.5]) / 4)
    assert s["policy_surprise_max"] == pytest.approx(0.5)

    # New: game-length distribution (mean/p90 legacy kept above).
    assert s["game_length_max"] == pytest.approx(130.0)
    assert s["game_length_p50"] == pytest.approx(50.0)


def test_driver_stats_empty_is_none_not_crash():
    """A driver that finished no games returns None-valued stats (not a crash),
    with the stable schema still present."""
    d = _driver()
    s = d.stats()
    assert s["root_policy_entropy_mean"] is None
    assert s["root_value_mean"] is None
    assert s["policy_surprise_mean"] is None
    assert s["decided_fraction"] is None
    assert s["unique_openings"] == {"10": 0, "16": 0, "20": 0}
    assert s["wins_by_player"] == {"0": 0, "1": 0}
    # per-phase schema always present
    assert set(s["root_value_by_phase"]) == {"opening", "mid", "late"}


def test_finish_tallies_openings_and_wins():
    """_finish (through the real code path, writer stubbed) records opening lines
    at 10/16/20 plies and the winner tally."""
    d = _driver()
    d._write_queue = _NoopQueue()  # swallow the enqueue; we only test aggregation

    tape = _FakeTape(records=[(i, 0, i % 2, i + 1) for i in range(25)], ply=25)
    d._finish(tape, winner=0, truncated=False)
    tape2 = _FakeTape(records=[(i, 1, i % 2, i + 1) for i in range(25)], ply=25)
    d._finish(tape2, winner=1, truncated=False)
    trunc = _FakeTape(records=[(i, 2, 0, i + 1) for i in range(25)], ply=25)
    d._finish(trunc, winner=None, truncated=True)

    assert d.games_finished == 3
    assert d.games_truncated == 1
    assert d.wins_by_player == {0: 1, 1: 1}  # truncated game has no winner
    # Three distinct openings at each depth.
    assert len(d.opening_lines) == 3
    assert len(d.opening_lines_16) == 3
    assert len(d.opening_lines_20) == 3


# --- PART 2: derived rates + summary ------------------------------------------


def test_derive_scheduler_rates_and_div_by_zero_guard():
    result = _segment()
    _derive_scheduler_rates(result)
    # winner-rate = 210/300, early = 60/100
    assert result["gumbel_play_winner_rate"] == pytest.approx(210 / 300)
    assert result["gumbel_play_winner_early_rate"] == pytest.approx(60 / 100)
    # lcb_override_rate = 40/1000
    assert result["lcb_override_rate"] == pytest.approx(40 / 1000)
    assert result["fast_fraction"] == pytest.approx(500 / 1000)
    assert result["full_fraction"] == pytest.approx(330 / 1000)
    assert result["init_fraction"] == pytest.approx(170 / 1000)

    # Div-by-zero guard: empty scheduler -> all rates None, no crash.
    zeroed = {"scheduler": {}, "total_decisions": 0}
    _derive_scheduler_rates(zeroed)
    assert zeroed["gumbel_play_winner_rate"] is None
    assert zeroed["lcb_override_rate"] is None
    assert zeroed["fast_fraction"] is None


def test_epoch_summary_one_line_format():
    result = _segment(epoch=14, games_finished=256, truncated_games=2,
                      rows_written=23000, game_length_p50=88.0, game_length_p90=148.0,
                      root_policy_entropy_mean=2.64)
    result["root_policy_entropy_by_phase"] = {
        "opening": {"mean": 3.1, "n": 1}, "mid": {"mean": 2.5, "n": 1},
        "late": {"mean": 1.9, "n": 1},
    }
    _derive_scheduler_rates(result)
    line = _format_epoch_summary(result)

    assert "\n" not in line
    assert line.startswith("selfplay epoch 14: 256 games (2 trunc) 23000 rows")
    assert "len p50 88 p90 148" in line
    assert "ent 2.64 (open 3.1/mid 2.5/late 1.9)" in line
    assert "uniq10/16/20 120/125/127" in line
    assert "winner-rate 0.70" in line  # 210/300
    assert "P0 wins 0.52" in line  # 66/(66+60)


def test_epoch_summary_guards_missing_values():
    """The summary never crashes on a sparse/None-filled result."""
    line = _format_epoch_summary({"epoch": 3})
    assert "\n" not in line
    assert line.startswith("selfplay epoch 3:")
    assert "?" in line  # missing fields rendered as ?


# --- tiny fakes ---------------------------------------------------------------


class _NoopQueue:
    def put(self, item):  # noqa: D401 - stub
        pass


class _FakeTape:
    __slots__ = ("records", "ply", "key")

    def __init__(self, records, ply, key=0):
        self.records = records
        self.ply = ply
        self.key = key
