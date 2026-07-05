from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package in (
    "hexo_frontend",
    "hexo_runner",
    "hexo_engine",
    "hexo_utils",
):
    path = ROOT / "packages" / package / "python"
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# web.py pulls in hexo_runner -> hexo_utils._rust at import time; the Rust
# extension is only built in the WSL venv, so the whole module skips elsewhere.
web = pytest.importorskip("hexo_frontend.web", reason="needs hexo_runner/engine build")


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_training_live_payload_shape(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    web._training_live_cache.clear()
    run_dir = tmp_path / "runs" / "dense_cnn_live_shape"
    _write_jsonl(
        run_dir / "diagnostics" / "events.jsonl",
        [
            {"event": "stage_started", "payload": {"stage": "epoch_000003"}},
        ],
    )

    payload = web._training_live_cached("dense_cnn_live_shape")

    assert set(payload) == {"run", "status", "ts"}
    assert payload["run"] == "dense_cnn_live_shape"
    assert isinstance(payload["ts"], float)
    assert payload["ts"] > 0
    status = payload["status"]
    assert isinstance(status, dict)
    # The status block is exactly _training_live_status(run_dir).
    assert status == web._training_live_status(run_dir)
    assert status["stage"] == "epoch_000003"
    assert status["current_epoch"] == 3


def test_training_live_rejects_unknown_run(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    web._training_live_cache.clear()
    (tmp_path / "runs").mkdir()

    with pytest.raises(ValueError):
        web._training_live_cached("no_such_run")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _shrimp_run(tmp_path: Path, name: str, epoch: int) -> Path:
    """Minimal shrimp-lineage run dir with an active epoch stage."""
    run_dir = tmp_path / "runs" / name
    _write_json(run_dir / "manifest.json", {"model": {"name": "shrimp"}})
    _write_jsonl(
        run_dir / "diagnostics" / "events.jsonl",
        [
            {"event": "stage_started", "payload": {"stage": "run_epochs"}},
            {"event": "stage_started", "payload": {"stage": f"epoch_{epoch:06d}"}},
        ],
    )
    return run_dir


def test_shrimp_sub_phases_from_segment_files(tmp_path: Path, monkeypatch: Any) -> None:
    """shrimp epoch tail: selfplay completed -> selecting window -> training
    (with elapsed/typical progress) -> evaluating, from the per-segment files."""
    monkeypatch.chdir(tmp_path)
    run_dir = _shrimp_run(tmp_path, "shrimp_phase", 32)
    diagnostics = run_dir / "diagnostics"
    _write_json(
        diagnostics / "shrimp.selfplay.live.json",
        {"status": "completed", "epoch": 32, "timestamp": 0.0},
    )
    _write_json(diagnostics / "shrimp.selfplay.epoch_000032.json", {"epoch": 32, "status": "completed"})

    # No select output yet -> the brief window selection.
    status = web._training_live_status(run_dir)
    assert status["sub_phase"] == "selecting window"

    # Select file present -> training, with elapsed + typical from epoch 31.
    _write_json(diagnostics / "shrimp.select.epoch_000032.json", {"epoch": 32})
    _write_json(diagnostics / "shrimp.training.epoch_000031.json", {"epoch": 31, "train_seconds": 441.5})
    status = web._training_live_status(run_dir)
    assert status["sub_phase"] == "training"
    assert "typical" in status["sub_phase_detail"]
    progress = status["phase_progress"]
    assert progress["phase"] == "training"
    assert progress["typical_seconds"] == 441.5
    assert progress["elapsed_seconds"] >= 0.0

    # Training file present -> the audit/eval/checkpoint tail.
    _write_json(diagnostics / "shrimp.training.epoch_000032.json", {"epoch": 32, "train_seconds": 430.0})
    status = web._training_live_status(run_dir)
    assert status["sub_phase"] == "evaluating"
    assert "phase_progress" not in status


def test_supervisor_halted_reads_stalled_not_running(tmp_path: Path, monkeypatch: Any) -> None:
    """A tripped breaker (halted flag; EXIT/HALT tail in supervisor.log) must not
    read as "running": stage_status becomes stalled and no sub-phase is derived."""
    monkeypatch.chdir(tmp_path)
    run_dir = _shrimp_run(tmp_path, "shrimp_halted", 31)
    _write_json(
        run_dir / "diagnostics" / "shrimp.selfplay.live.json",
        {"status": "completed", "epoch": 31, "timestamp": 0.0},
    )
    _write_json(run_dir / "diagnostics" / "shrimp.selfplay.epoch_000031.json", {"epoch": 31})
    (run_dir / "supervisor.log").write_text(
        "[2026-07-04T18:53:57Z] RESUME from epoch_000030.pt\n"
        "[2026-07-04T18:53:57Z] LAUNCH out=train.out.log\n"
        "[2026-07-04T18:54:02Z] EXIT pid=74331 code=1 uptime=5s\n"
        "[2026-07-04T18:54:02Z] CRASH| Traceback (most recent call last):\n"
        "[2026-07-04T18:54:02Z] CRASH| ValueError: unknown divergence override key\n"
        "[2026-07-04T18:54:08Z] HALT: breaker tripped. Not relaunching.\n",
        encoding="utf-8",
    )
    (run_dir / "supervisor_halted.flag").write_text("tripped", encoding="utf-8")

    status = web._training_live_status(run_dir)
    supervisor = status["supervisor"]
    assert supervisor["halted"] is True
    assert supervisor["trainer_presumed_up"] is False
    assert supervisor["last_resume"] == "epoch_000030.pt"
    assert supervisor["last_crash"].startswith("ValueError")
    assert status["stage_status"] == "stalled"
    assert "sub_phase" not in status


def test_supervisor_up_after_launch(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = _shrimp_run(tmp_path, "shrimp_up", 32)
    (run_dir / "supervisor.log").write_text(
        "[2026-07-04T18:54:02Z] EXIT pid=74331 code=1 uptime=5s\n"
        "[2026-07-04T19:12:26Z] RESUME from epoch_000030.pt\n"
        "[2026-07-04T19:12:26Z] LAUNCH out=train.out.log\n",
        encoding="utf-8",
    )
    status = web._training_live_status(run_dir)
    supervisor = status["supervisor"]
    assert supervisor["halted"] is False
    assert supervisor["trainer_presumed_up"] is True
    assert status["stage_status"] == "running"


def test_latest_checkpoint_summary(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = _shrimp_run(tmp_path, "shrimp_ckpt", 32)
    ckpts = run_dir / "checkpoints"
    ckpts.mkdir(parents=True)
    (ckpts / "epoch_000030.pt").write_bytes(b"0")
    (ckpts / "epoch_000031.pt").write_bytes(b"1")
    os.utime(ckpts / "epoch_000030.pt", (1.0, 1.0))  # force ordering by mtime

    status = web._training_live_status(run_dir)
    checkpoint = status["latest_checkpoint"]
    assert checkpoint["name"] == "epoch_000031.pt"
    assert checkpoint["epoch"] == 31
    assert checkpoint["age_seconds"] >= 0.0


def test_epoch_history_marks_in_flight_epoch(tmp_path: Path, monkeypatch: Any) -> None:
    """The currently-running epoch gets a provisional row (status in_progress +
    live self-play counters); without live_status behavior is unchanged."""
    monkeypatch.chdir(tmp_path)
    run_dir = _shrimp_run(tmp_path, "shrimp_inflight", 32)
    import time as _time
    _write_json(
        run_dir / "diagnostics" / "shrimp.selfplay.live.json",
        {
            "status": "running",
            "epoch": 32,
            "timestamp": _time.time(),
            "games_finished": 90,
            "requested_games": 256,
            "search_positions_per_second": 14.4,
        },
    )
    status = web._training_live_status(run_dir)
    assert status["sub_phase"] == "self-play"

    rows = web._epoch_history(run_dir, status)
    row = next(r for r in rows if r["epoch"] == 32)
    assert row["status"] == "in_progress"
    assert row["in_progress"]["phase"] == "self-play"
    assert row["in_progress"]["selfplay_live"]["games_finished"] == 90

    # Without live_status the synthetic row is absent (legacy behavior).
    assert all(r.get("status") != "in_progress" for r in web._epoch_history(run_dir))


def test_training_epochs_strip_marks_in_flight_epoch(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = _shrimp_run(tmp_path, "shrimp_strip", 32)
    diagnostics = run_dir / "diagnostics"
    import time as _time
    _write_json(
        diagnostics / "shrimp.selfplay.live.json",
        {"status": "running", "epoch": 32, "timestamp": _time.time(), "games_finished": 5, "requested_games": 256},
    )
    # Epoch 31 finished: segment file + merged epoch json -> never marked.
    _write_json(diagnostics / "shrimp.selfplay.epoch_000031.json", {"epoch": 31, "status": "completed"})
    _write_json(
        diagnostics / "epoch_000031.json",
        {"status": "completed", "metadata": {"result": {"epoch": 31}}},
    )

    payload = web._training_epochs(run_dir)
    by_epoch = {rec["epoch"]: rec for rec in payload["epochs"]}
    assert by_epoch[32]["in_progress"] is True
    assert by_epoch[32]["live"]["games_finished"] == 5
    assert "in_progress" not in by_epoch[31]


def test_training_live_micro_cache_returns_same_object(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.chdir(tmp_path)
    web._training_live_cache.clear()
    run_dir = tmp_path / "runs" / "dense_cnn_live_cache"
    (run_dir / "diagnostics").mkdir(parents=True)

    first = web._training_live_cached("dense_cnn_live_cache")
    second = web._training_live_cached("dense_cnn_live_cache")

    # Within the ~1s micro-cache window the SAME payload object comes back
    # (no second disk pass), matching the _training_runs_cached pattern.
    assert second is first

    # Expiring the cached entry forces a rebuild -> a fresh object.
    key = "dense_cnn_live_cache"
    stamp, payload = web._training_live_cache[key]
    web._training_live_cache[key] = (
        stamp - web.TRAINING_LIVE_CACHE_TTL_SECONDS - 1.0,
        payload,
    )
    third = web._training_live_cached(key)
    assert third is not first
    assert third["run"] == first["run"]
