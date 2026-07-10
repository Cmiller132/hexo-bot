"""hexfield_eq equal-TIME eval budget (ray plan §3 L2/L3, spec D-S35).

The Rust search core has no native wall-clock budget, so the equal-time A/B
leg is the documented cheap approximation: per-arm visit calibration from a
measured probe-search latency. This suite pins:

  1. the ``visits_for_time_budget`` math (linear scaling + the starve/runaway
     clamps);
  2. ``calibrate_time_budget_visits`` — warmup dropped, median over probe
     positions, probe trees keyed off the game-key space and discarded, RNG
     streams untouched (deterministic injected clock, fake session);
  3. threading — ``HexfieldCheckpointAdapter.start`` and
     ``eval_driver.play_eval_match`` re-key their search kwargs to the
     calibrated visits and surface the calibration record; a zero/absent
     budget leaves the fixed-visit path byte-identical (no calibration call).

CPU-only; the engine is exercised only for probe-state construction. Runs in
the hexgt-build venv via PYTHONPATH=packages/hexfield_eq/python (plus the
shared packages).
"""

from __future__ import annotations

import pytest

from hexfield_eq import eval_driver
from hexfield_eq.config import parse_hexfield_config
from hexfield_eq.eval_driver import (
    HexfieldCheckpointAdapter,
    calibrate_time_budget_visits,
    play_eval_match,
    visits_for_time_budget,
)


# --- fakes -------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class FakeSession:
    """Records search/discard calls; advances the injected clock by a fixed
    ms-per-visit when given one (deterministic 'latency')."""

    def __init__(self, clock: FakeClock | None = None, ms_per_visit: float = 0.1):
        self.clock = clock
        self.ms_per_visit = ms_per_visit
        self.search_calls: list[dict] = []
        self.discarded: list[int] = []

    def search(self, keys, states, **kwargs):
        self.search_calls.append({"keys": list(keys), **kwargs})
        if self.clock is not None:
            self.clock.t += (self.ms_per_visit * kwargs["visits"]) / 1000.0
        return [{"action_id": 0} for _ in keys]

    def discard(self, key):
        self.discarded.append(key)


# --- 1. the budget math --------------------------------------------------------------


def test_visits_for_time_budget_math() -> None:
    # Linear scaling: half the budget -> half the visits; double -> double.
    assert visits_for_time_budget(512, measured_ms=100.0, budget_ms=50.0) == 256
    assert visits_for_time_budget(512, measured_ms=100.0, budget_ms=200.0) == 1024
    # Starve clamp (default floor 16) and runaway clamp (default 8x base).
    assert visits_for_time_budget(512, measured_ms=1000.0, budget_ms=1.0) == 16
    assert visits_for_time_budget(512, measured_ms=1.0, budget_ms=1000.0) == 8 * 512
    assert visits_for_time_budget(512, 1.0, 1000.0, max_visits=600) == 600
    # Degenerate measurements fall back to the base budget.
    assert visits_for_time_budget(512, measured_ms=0.0, budget_ms=50.0) == 512
    assert visits_for_time_budget(512, measured_ms=-1.0, budget_ms=50.0) == 512


# --- 2. calibration ------------------------------------------------------------------


def test_calibrate_measures_median_and_discards_probes() -> None:
    clock = FakeClock()
    session = FakeSession(clock, ms_per_visit=0.1)  # 512 visits -> 51.2 ms
    kwargs = {"visits": 512, "c_puct": 1.5, "temperature": 0.0}
    visits, info = calibrate_time_budget_visits(
        session, evaluator=object(), search_kwargs=kwargs, overrides={},
        time_ms_per_move=25.6, seed=3, _clock=clock,
    )
    # measured 51.2 ms/move, budget 25.6 -> exactly half the visits.
    assert visits == 256
    assert info["measured_ms_per_move"] == pytest.approx(51.2, abs=0.01)
    assert info["base_visits"] == 512 and info["calibrated_visits"] == 256
    # warmup + one probe per position (default 3 depths), all single-root at
    # the BASE budget, all discarded, keyed far above the game-key space.
    assert len(session.search_calls) == 4  # 1 warmup + 3 probes
    assert all(c["visits"] == 512 and len(c["keys"]) == 1 for c in session.search_calls)
    probe_keys = [c["keys"][0] for c in session.search_calls]
    assert session.discarded == probe_keys
    assert min(probe_keys) >= eval_driver._TIME_PROBE_KEY_BASE
    # kwargs are read, never mutated.
    assert kwargs["visits"] == 512


# --- 3. threading --------------------------------------------------------------------


def _adapter(time_ms):
    cfg = parse_hexfield_config({"device": "cpu"})
    session = FakeSession()
    return HexfieldCheckpointAdapter(
        "b.ckpt",
        config=cfg,
        label="B",
        overrides_b={},
        make_session=lambda: session,
        max_states=1024,
        visits=None,
        virtual_batch_size=None,
        active_root_limit=None,
        paired_openings=True,
        batch_openings=False,
        build_evaluator=lambda: object(),
        time_ms_per_move=time_ms,
    )


def test_checkpoint_adapter_calibrates_its_own_budget(monkeypatch) -> None:
    calls = []

    def fake_calibrate(session, evaluator, **kw):
        calls.append(kw)
        return 777, {"calibrated_visits": 777}

    monkeypatch.setattr(eval_driver, "calibrate_time_budget_visits", fake_calibrate)
    adapter = _adapter(time_ms=40.0)
    adapter.start([])
    assert adapter._search_kwargs["visits"] == 777
    assert adapter.time_calibration == {"calibrated_visits": 777}
    assert calls and calls[0]["time_ms_per_move"] == 40.0


def test_zero_budget_is_the_fixed_visit_path(monkeypatch) -> None:
    def boom(*a, **k):  # the off path must never calibrate
        raise AssertionError("calibrate called with budget off")

    monkeypatch.setattr(eval_driver, "calibrate_time_budget_visits", boom)
    adapter = _adapter(time_ms=None)  # config default eval_time_budget_ms=0.0
    adapter.start([])
    assert adapter._search_kwargs["visits"] == 512  # sp.search_visits default
    assert adapter.time_calibration is None


class _NullOpponent:
    label = "null"

    def start(self, games, *, seed_base=0):
        return None

    def advance(self, batch, **kwargs):
        return 0

    def close(self):
        return None


def test_play_eval_match_threads_the_budget(monkeypatch) -> None:
    def fake_calibrate(session, evaluator, **kw):
        return 999, {"calibrated_visits": 999}

    monkeypatch.setattr(eval_driver, "calibrate_time_budget_visits", fake_calibrate)
    seen = {}

    def meta_extra(games, tel):
        seen["tel"] = tel
        return {}

    result = play_eval_match(
        "a.ckpt",
        _NullOpponent(),
        0,  # no games: the round loop exits immediately; calibration still ran
        config=parse_hexfield_config({"device": "cpu"}),
        label_a="A",
        label_b="B",
        meta_extra_fn=meta_extra,
        build_candidate_evaluator=lambda: object(),
        make_session=lambda: FakeSession(),
        time_ms_per_move=30.0,
    )
    tel = seen["tel"]
    assert tel.eval_visits == 999
    assert tel.time_calibration == {"calibrated_visits": 999}
    assert result["meta"]["label_a"] == "A"

    # Config-knob resolution: the explicit kwarg absent, the toml knob drives it.
    cfg = parse_hexfield_config(
        {"device": "cpu", "multi_stage_eval": {"eval_time_budget_ms": 12.5}}
    )
    assert eval_driver._resolve_time_budget(cfg, None) == 12.5
    assert eval_driver._resolve_time_budget(cfg, 0.0) == 0.0  # explicit off wins
    off = parse_hexfield_config({"device": "cpu"})
    assert eval_driver._resolve_time_budget(off, None) == 0.0
