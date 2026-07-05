"""Integration tests for shrimp.evaluation.evaluate_epoch.

The multistage eval is monkeypatched so these run without GPU games. Covered
behaviors:

  * A multistage-eval exception is caught and recorded in the result, not
    propagated.
  * The deep eval short-circuits when disabled, when epoch does not match
    every_n_epochs, or when the candidate checkpoint file is missing.
  * The runner receives the on-disk epoch checkpoint path and ctx.output_dir
    as run_dir.
  * The moves-left head audit runs whether or not the deep eval ran.

Imports shrimp.evaluation, which imports torch and the native engine at module
import time, so this runs in the torch venv rather than the pure-CPU suite.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "hexo_engine" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "shrimp" / "python"))

from shrimp import evaluation  # noqa: E402


class _StubModel:
    def __init__(self):
        self.train_calls = 0

    def train(self):
        self.train_calls += 1


def _make_ctx(tmp_path: Path, *, enabled=True, every=1):
    run = tmp_path / "run"
    ckpt = run / "checkpoints"
    diag = run / "diagnostics"
    samples = run / "samples"
    for d in (run, ckpt, diag, samples):
        d.mkdir(parents=True, exist_ok=True)
    model_config = {
        "device": "cpu",
        "multi_stage_eval": {
            "enabled": enabled,
            "games_budget": 72,
            "every_n_epochs": every,
            "full_search_visits": 512,
            "eval_virtual_batch_size": 16,
            "opponents": {"sealbot_enabled": False},
            "sprt": {"enabled": False},
        },
    }
    ctx = types.SimpleNamespace(
        config=types.SimpleNamespace(model=types.SimpleNamespace(config=model_config)),
        output_dir=run,
        checkpoint_dir=ckpt,
        diagnostics_dir=diag,
        samples_dir=samples,
    )
    components = types.SimpleNamespace(model=types.SimpleNamespace(model=_StubModel()))
    return ctx, components


@pytest.fixture(autouse=True)
def _stub_head_audit(monkeypatch):
    # Replace the head audit with a torch-free stub returning a fixed result.
    monkeypatch.setattr(
        "shrimp.head_audit.audit_moves_left_head",
        lambda *a, **k: {"passed": True, "stub": True},
    )


def test_fail_soft_does_not_propagate_and_head_audit_still_runs(tmp_path, monkeypatch):
    ctx, components = _make_ctx(tmp_path)
    (ctx.checkpoint_dir / "epoch_000005.pt").write_bytes(b"x")  # candidate exists

    def _boom(*a, **k):
        raise RuntimeError("eval blew up")

    monkeypatch.setattr(
        "shrimp.multistage_eval.run_multistage_eval_concurrent", _boom
    )
    result = evaluation.evaluate_epoch(ctx=ctx, components=components, epoch=5)
    assert result["multistage"]["status"] == "error"
    assert "eval blew up" in result["multistage"]["error"]
    # Head audit ran despite the multistage-eval error.
    assert "moves_left_head_audit" in result
    assert result["moves_left_head_audit"].get("stub") is True


def test_passes_candidate_path_and_run_dir(tmp_path, monkeypatch):
    ctx, components = _make_ctx(tmp_path)
    (ctx.checkpoint_dir / "epoch_000007.pt").write_bytes(b"x")
    captured = {}

    def _fake(run_dir, candidate_ckpt, config, **kw):
        captured["run_dir"] = Path(run_dir)
        captured["candidate"] = Path(candidate_ckpt)
        captured["kw"] = kw
        return {"meta": {"anchor": "bc_prefit", "elapsed_seconds": 1.0}, "verdict": {"label": "INCONCLUSIVE"}}

    monkeypatch.setattr("shrimp.multistage_eval.run_multistage_eval_concurrent", _fake)
    result = evaluation.evaluate_epoch(ctx=ctx, components=components, epoch=7)
    assert result["multistage"]["status"] == "completed"
    assert result["multistage"]["verdict"] == "INCONCLUSIVE"
    assert captured["candidate"] == ctx.checkpoint_dir / "epoch_000007.pt"
    assert captured["run_dir"] == ctx.output_dir
    assert captured["kw"]["candidate_epoch"] == 7


def test_gating_disabled(tmp_path):
    ctx, components = _make_ctx(tmp_path, enabled=False)
    (ctx.checkpoint_dir / "epoch_000005.pt").write_bytes(b"x")
    result = evaluation.evaluate_epoch(ctx=ctx, components=components, epoch=5)
    assert result["multistage"]["status"] == "disabled"
    assert "moves_left_head_audit" in result  # audit still runs


def test_gating_every_n_epochs(tmp_path, monkeypatch):
    ctx, components = _make_ctx(tmp_path, every=5)
    (ctx.checkpoint_dir / "epoch_000007.pt").write_bytes(b"x")
    called = {"n": 0}
    monkeypatch.setattr(
        "shrimp.multistage_eval.run_multistage_eval_concurrent",
        lambda *a, **k: called.__setitem__("n", called["n"] + 1) or {"meta": {}, "verdict": {}},
    )
    result = evaluation.evaluate_epoch(ctx=ctx, components=components, epoch=7)  # 7 % 5 != 0
    assert result["multistage"]["status"] == "skipped"
    assert called["n"] == 0


def test_gating_missing_candidate_checkpoint(tmp_path):
    ctx, components = _make_ctx(tmp_path)  # no epoch file written
    result = evaluation.evaluate_epoch(ctx=ctx, components=components, epoch=5)
    assert result["multistage"]["status"] == "skipped"
    assert "missing" in result["multistage"]["reason"]
