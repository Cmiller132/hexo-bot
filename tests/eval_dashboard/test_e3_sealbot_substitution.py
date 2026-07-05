"""E3 — a SealBot DEATH must FAIL LOUDLY: when SealBot was expected (config-
enabled) but unavailable, the BT zero-point silently re-pins to bc_prefit / the
lowest checkpoint, shifting every ABSOLUTE Elo. The fix marks Stage-D DEGRADED +
machine-flagged (``anchor_substituted`` / ``degraded``) and logs a WARNING.

A merely config-DISABLED SealBot (roster.sealbot is None) is NOT a degradation —
the anchor falls to bc_prefit but the verdict stays normal.

CPU-only, no torch, no GPU, no games played (we feed _stage_d_pool a pre-built
pool and append=False).

Run:
  PYTHONPATH=packages/shrimp/python python -m pytest tests/eval_dashboard/test_e3_sealbot_substitution.py
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from shrimp.config import parse_shrimp_config
from shrimp.multistage_eval import (
    SEALBOT_LABEL,
    Opponent,
    Roster,
    _stage_d_pool,
)


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _cfg():
    return parse_shrimp_config({}).multi_stage_eval


def _pool_row(a, b, wa, wb, kind="checkpoint", weight=1.0, epoch=35):
    return {
        "epoch": epoch,
        "a": a,
        "b": b,
        "wins_a": float(wa),
        "wins_b": float(wb),
        "weight": float(weight),
        "kind": kind,
        "raw": {},
    }


def _roster(*, sealbot_expected: bool):
    """Roster with a bc_prefit anchor + an ep30 champion; candidate cand_ep35."""
    sealbot = (
        Opponent(label=SEALBOT_LABEL, role="sealbot", ckpt=None, epoch=None)
        if sealbot_expected
        else None
    )
    return Roster(
        candidate_label="cand_ep35",
        candidate_epoch=35,
        sealbot=sealbot,
        champion=Opponent(label="ep30", role="champion", ckpt=Path("x"), epoch=30),
        opponents=(
            Opponent(label="bc_prefit", role="anchor", ckpt=Path("bc"), epoch=2),
            Opponent(label="ep30", role="champion", ckpt=Path("x"), epoch=30),
        ),
    )


def _pool_no_sealbot():
    # candidate edges vs bc_prefit + ep30, NO sealbot edge -> anchor falls back.
    return {
        "edges": [
            _pool_row("cand_ep35", "bc_prefit", 8, 2),
            _pool_row("cand_ep35", "ep30", 5, 5),
        ]
    }


def _pool_with_sealbot():
    return {
        "edges": [
            _pool_row("cand_ep35", SEALBOT_LABEL, 6, 4, kind="sealbot", weight=0.5),
            _pool_row("cand_ep35", "bc_prefit", 8, 2),
            _pool_row("cand_ep35", "ep30", 5, 5),
        ]
    }


def _run(roster, pool, reason):
    tmp = Path(tempfile.mkdtemp())
    return _stage_d_pool(
        _cfg(), roster, [], tmp,
        pool_doc=pool, append=False,
        sealbot_expected_but_unavailable=reason,
    )


def test_sealbot_death_degrades():
    handler = _CaptureHandler()
    logger = logging.getLogger("shrimp.eval")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        stage_d, ratings, verdict, _ = _run(
            _roster(sealbot_expected=True),
            _pool_no_sealbot(),
            "SealBotUnavailableError: worker died",
        )
    finally:
        logger.removeHandler(handler)

    assert stage_d.status == "degraded", f"expected degraded, got {stage_d.status!r}"
    assert verdict.get("anchor_substituted") is True, verdict
    assert verdict.get("substituted_from") == SEALBOT_LABEL, verdict
    # The substituted anchor is whatever _choose_anchor falls back to. Under the
    # live radius-4 process bc_prefit is E4-OOD-excluded, so it falls to ep30; at
    # radius-8 it would be bc_prefit. Assert it substituted to SOME non-SealBot
    # checkpoint anchor (the precise label is radius-dependent, not load-bearing).
    sub_to = verdict.get("substituted_to")
    assert sub_to is not None and sub_to != SEALBOT_LABEL, verdict
    assert sub_to in {"bc_prefit", "ep30"}, verdict
    assert "worker died" in verdict.get("sealbot_unavailable_reason", ""), verdict
    assert verdict.get("degraded") is True
    assert stage_d.detail.get("sealbot_substituted") is True
    warned = [r for r in handler.records if r.levelno >= logging.WARNING]
    assert any("SealBot expected but unavailable" in r.getMessage() for r in warned), (
        "death was not loud: " + repr([r.getMessage() for r in warned])
    )
    # BT fit still converged on the substituted anchor (it is degraded, not broken)
    assert ratings["fit"].get("anchor") == sub_to, ratings["fit"]
    print(f"[E3] PASS death-degrades: status={stage_d.status} "
          f"substituted_to={sub_to}")


def test_config_disabled_sealbot_is_not_degraded():
    # roster.sealbot is None -> sealbot_expected_but_unavailable threaded as None.
    stage_d, ratings, verdict, _ = _run(
        _roster(sealbot_expected=False),
        _pool_no_sealbot(),
        None,  # callers pass None when roster.sealbot is None
    )
    assert stage_d.status == "completed", f"config-disabled should NOT degrade: {stage_d.status}"
    assert "anchor_substituted" not in verdict, verdict
    assert not verdict.get("degraded"), verdict
    assert stage_d.detail.get("sealbot_substituted") is False
    # anchor falls to a non-SealBot checkpoint (bc_prefit at r8, ep30 when bc is
    # E4-OOD-excluded at r4), but the pool is treated as NORMAL (no degrade).
    assert ratings["fit"].get("anchor") in {"bc_prefit", "ep30"}, ratings["fit"]
    print(f"[E3] PASS config-disabled: status={stage_d.status} (no degrade, "
          f"anchor={ratings['fit'].get('anchor')})")


def test_sealbot_present_no_flag():
    stage_d, ratings, verdict, _ = _run(
        _roster(sealbot_expected=True),
        _pool_with_sealbot(),
        None,  # SealBot edge present -> reason is None
    )
    assert stage_d.status == "completed", stage_d.status
    assert "anchor_substituted" not in verdict, verdict
    assert ratings["fit"].get("anchor") == SEALBOT_LABEL, ratings["fit"]
    assert ratings["fit"].get("anchor_is_sealbot") is True
    print(f"[E3] PASS sealbot-present: anchor={ratings['fit']['anchor']}")


if __name__ == "__main__":
    test_sealbot_death_degrades()
    test_config_disabled_sealbot_is_not_degraded()
    test_sealbot_present_no_flag()
    print("E3 ALL GREEN")
