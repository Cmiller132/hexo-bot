from __future__ import annotations

import json
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
