"""E2 — a missing PERMANENT anchor must FAIL LOUDLY (logged + recorded), not be
silently dropped with a bare ``continue``.

This is the bug that made bc_prefit vanish from the live roster mid-run. The fix:
``select_opponents`` records every unresolved anchor in ``roster.dropped_anchors``
(surfaced in ``_roster_summary``) AND logs a WARNING. ``HEXFIELD_ANCHOR_ROOTS``
(+ absolute config paths) lets the anchor resolve again without a code change.

CPU-only, no torch, no GPU.

Run:
  PYTHONPATH=packages/hexfield/python python -m pytest tests/eval_dashboard/test_e2_anchor_drop.py
"""
from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
from pathlib import Path

from hexfield.config import MultiStageEvalOpponents, parse_hexfield_config
from hexfield.multistage_eval import _roster_summary, select_opponents


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _base_cfg():
    return parse_hexfield_config({}).multi_stage_eval


def _make_run(tmp: Path) -> Path:
    run = tmp / "run"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    # an in-run checkpoint exists so the candidate/champion machinery is happy
    for ep in (5, 30, 35):
        (run / "checkpoints" / f"epoch_{ep:06d}.pt").write_bytes(b"stub")
    return run


def _with_anchors(cfg, anchors):
    opp = dataclasses.replace(cfg.opponents, permanent_anchors=tuple(anchors))
    return dataclasses.replace(cfg, opponents=opp)


def test_missing_anchor_is_loud_and_recorded():
    tmp = Path(tempfile.mkdtemp())
    run = _make_run(tmp)
    cand = run / "checkpoints" / "epoch_000035.pt"

    # bc_prefit points at a file that does NOT exist anywhere reachable.
    cfg = _with_anchors(
        _base_cfg(),
        [
            ("bc_prefit", "runs/hexfield_bc_1/checkpoint_epoch2.pt"),
            ("ep5", "epoch_000005.pt"),  # this one resolves (bare filename -> ckpt dir)
        ],
    )

    handler = _CaptureHandler()
    logger = logging.getLogger("hexfield.eval")
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    # Ensure no stray env root accidentally resolves bc_prefit.
    prev = os.environ.pop("HEXFIELD_ANCHOR_ROOTS", None)
    try:
        roster = select_opponents(run, cand, cfg, candidate_epoch=35)
    finally:
        logger.removeHandler(handler)
        if prev is not None:
            os.environ["HEXFIELD_ANCHOR_ROOTS"] = prev

    labels = {a["label"] for a in roster.dropped_anchors}
    assert "bc_prefit" in labels, f"bc_prefit not recorded as dropped: {roster.dropped_anchors}"
    assert all(o.label != "bc_prefit" for o in roster.opponents), "dropped anchor still in roster"

    summary = _roster_summary(roster)
    assert summary.get("dropped_anchors"), "drop not surfaced in _roster_summary"
    assert any(a["label"] == "bc_prefit" for a in summary["dropped_anchors"])

    warned = [r for r in handler.records if r.levelno >= logging.WARNING]
    assert any("bc_prefit" in r.getMessage() and "unresolved" in r.getMessage() for r in warned), (
        "missing anchor was NOT loud; warnings=" + repr([r.getMessage() for r in warned])
    )
    print(f"[E2] PASS loud-drop: dropped={labels} warned={[r.getMessage() for r in warned]}")


def test_env_root_resolves_anchor():
    tmp = Path(tempfile.mkdtemp())
    run = _make_run(tmp)
    cand = run / "checkpoints" / "epoch_000035.pt"

    # Place the bc checkpoint in a custom tree and point HEXFIELD_ANCHOR_ROOTS at it.
    custom_root = tmp / "canonical"
    (custom_root / "runs" / "hexfield_bc_1").mkdir(parents=True, exist_ok=True)
    (custom_root / "runs" / "hexfield_bc_1" / "checkpoint_epoch2.pt").write_bytes(b"stub")

    cfg = _with_anchors(
        _base_cfg(),
        [("bc_prefit", "runs/hexfield_bc_1/checkpoint_epoch2.pt")],
    )

    prev = os.environ.get("HEXFIELD_ANCHOR_ROOTS")
    os.environ["HEXFIELD_ANCHOR_ROOTS"] = str(custom_root)
    try:
        roster = select_opponents(run, cand, cfg, candidate_epoch=35)
    finally:
        if prev is None:
            os.environ.pop("HEXFIELD_ANCHOR_ROOTS", None)
        else:
            os.environ["HEXFIELD_ANCHOR_ROOTS"] = prev

    anchors = {o.label: o for o in roster.opponents if o.role == "anchor"}
    assert "bc_prefit" in anchors, f"env root did not resolve bc_prefit: {[o.label for o in roster.opponents]}"
    assert roster.dropped_anchors == (), f"unexpected drops: {roster.dropped_anchors}"
    print(f"[E2] PASS env-root: bc_prefit resolved to {anchors['bc_prefit'].ckpt}")


def test_absolute_anchor_path_resolves():
    tmp = Path(tempfile.mkdtemp())
    run = _make_run(tmp)
    cand = run / "checkpoints" / "epoch_000035.pt"

    abs_ckpt = tmp / "elsewhere" / "bc.pt"
    abs_ckpt.parent.mkdir(parents=True, exist_ok=True)
    abs_ckpt.write_bytes(b"stub")

    cfg = _with_anchors(_base_cfg(), [("bc_prefit", str(abs_ckpt))])
    prev = os.environ.pop("HEXFIELD_ANCHOR_ROOTS", None)
    try:
        roster = select_opponents(run, cand, cfg, candidate_epoch=35)
    finally:
        if prev is not None:
            os.environ["HEXFIELD_ANCHOR_ROOTS"] = prev

    anchors = {o.label for o in roster.opponents if o.role == "anchor"}
    assert "bc_prefit" in anchors, "absolute anchor path did not resolve"
    assert roster.dropped_anchors == ()
    print("[E2] PASS absolute-path: bc_prefit resolved via absolute config path")


if __name__ == "__main__":
    test_missing_anchor_is_loud_and_recorded()
    test_env_root_resolves_anchor()
    test_absolute_anchor_path_resolves()
    print("E2 ALL GREEN")
