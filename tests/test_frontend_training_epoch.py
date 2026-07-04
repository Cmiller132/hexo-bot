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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _epoch_result_json(epoch: int, *, checkpoint: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "epoch": epoch,
        "selfplay": {
            "status": "completed",
            "epoch": epoch,
            "games_finished": 16,
            "completed_games": 15,
            "truncated_games": 1,
            "effective_samples": 2400,
            "searched_positions": 1200,
            "mcts_simulations": 153600,
            "search_positions_per_second": 41.5,
            "elapsed_seconds": 300.0,
        },
        "training": {
            "status": "completed",
            "loss": 2.5 - 0.1 * epoch,
            "loss_components": {"policy": 1.5, "value": 0.6},
            "steps": 100,
            "samples": 6400,
            "batch_size": 64,
            "samples_per_second": 800.0,
            "elapsed_seconds": 8.0,
        },
    }
    if checkpoint:
        result["checkpoint"] = {
            "checkpoint_path": f"checkpoints/epoch_{epoch:06d}.pt",
            "name": f"epoch_{epoch:06d}.pt",
        }
    return {
        "status": "completed",
        "elapsed_seconds": 321.0,
        "metadata": {"result": result},
    }


def test_training_epoch_returns_curated_full_payload(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEXO_DEBUG_RUN_ROOT", raising=False)
    run_dir = tmp_path / "runs" / "dense_cnn_epoch_full"
    diagnostics = run_dir / "diagnostics"

    _write_json(diagnostics / "epoch_000001.json", _epoch_result_json(1))
    _write_json(diagnostics / "epoch_000002.json", _epoch_result_json(2))
    _write_json(
        diagnostics / "dense_cnn.selfplay.epoch_000002.json",
        {
            "status": "completed",
            "epoch": 2,
            "scheduler": "balanced",
            "raw_samples": 2600,
            "effective_samples": 2400,
            "total_decisions": 2500,
            "active_games": 64,
            "mcts_virtual_batch_size": 24,
            "elapsed_seconds": 300.0,
            "mcts_search_elapsed_seconds": 250.0,
            "temperature_control": {
                "expected_game_length": 150.0,
                "halflife_plies": 30.0,
                "halflife_fraction": 0.2,
            },
            "pcr": {
                "enabled": True,
                "full_proportion": 0.25,
                "fast_visits": 32,
                "full_search_count": 600,
                "fast_search_count": 1800,
                "fast_rows_excluded": 100,
                "npz_skipped_empty": 3,
            },
            "policy_init": {
                "enabled": True,
                "fraction": 0.5,
                "avg_plies": 4.0,
                "max_plies": 8,
                "temperature": 1.0,
                "moves": 512,
            },
            "root_policy_temperature_control": {
                "base": 1.1,
                "early": 1.25,
                "halflife_plies": 20.0,
            },
            "npz_writes": [{"path": "shard_a.npz"}, {"path": "shard_b.npz"}],
            "mcts_diagnostics": {"huge": "blob"},
            "scheduler_diagnostics": {"huge": "blob"},
            "spill": {"events": 0},
            "selfplay_npz_files": ["shard_a.npz"],
        },
    )
    _write_json(
        diagnostics / "dense_cnn.evaluation.epoch_000002.json",
        {
            "status": "completed",
            "epoch": 2,
            "games": 2,
            "completed": 2,
            "wins": 1,
            "losses": 1,
            "mean_turns": 12.0,
        },
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "model": {
                "name": "hexo_models.dense_cnn",
                "config": {
                    "architecture": {
                        "input_channels": 14,
                        "channels": 96,
                        "blocks_type": "restnet",
                        "attention_heads": 4,
                        "short_term_value_horizons": [2, 4, 8],
                        "moves_left_head": True,
                        "internal_extra": 1,
                    },
                    "selfplay": {
                        "search_visits": 128,
                        "active_games": 64,
                        "c_puct": 1.5,
                        "root_policy_temperature": 1.1,
                        "fpu_reduction": 0.2,
                        "temperature": 1.0,
                        "internal_extra": 1,
                    },
                    "evaluation": {"games_per_epoch": 32, "eval_every": 5, "sealbot_variant": "best"},
                    "training": {"batch_size": 64, "learning_rate": 0.001, "train_samples_per_epoch": 6400},
                },
            }
        },
    )
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "epoch_000002.pt").write_bytes(b"ckpt-bytes")

    payload = web._training_epoch("dense_cnn_epoch_full", 2)

    assert set(payload) == {
        "run",
        "epoch",
        "history",
        "prev_epoch",
        "evaluation",
        "diagnostics",
        "selfplay_extras",
        "manifest",
        "checkpoint",
        "multistage_eval",
    }
    assert payload["run"] == "dense_cnn_epoch_full"
    assert payload["epoch"] == 2
    assert payload["history"]["epoch"] == 2
    assert payload["prev_epoch"]["epoch"] == 1
    assert payload["evaluation"]["epoch"] == 2
    assert payload["evaluation"]["wins"] == 1
    assert isinstance(payload["diagnostics"], dict)
    assert "selfplay" in payload["diagnostics"]

    extras = payload["selfplay_extras"]
    assert extras["scheduler"] == "balanced"
    assert extras["raw_samples"] == 2600
    assert extras["effective_samples"] == 2400
    assert extras["mcts_search_elapsed_seconds"] == 250.0
    assert extras["temperature_control"]["expected_game_length"] == 150.0
    assert extras["pcr"]["full_search_count"] == 600
    assert extras["policy_init"]["moves"] == 512
    assert extras["root_policy_temperature_control"]["base"] == 1.1
    # Curation is the size cap: the bulk/memory-internal keys must never pass through.
    for excluded in ("npz_writes", "mcts_diagnostics", "scheduler_diagnostics", "spill", "selfplay_npz_files"):
        assert excluded not in extras
    assert "npz_skipped_empty" not in extras["pcr"]

    manifest = payload["manifest"]
    assert manifest["model_name"] == "hexo_models.dense_cnn"
    assert manifest["architecture"]["channels"] == 96
    assert manifest["architecture"]["short_term_value_horizons"] == [2, 4, 8]
    assert "internal_extra" not in manifest["architecture"]
    assert manifest["selfplay"]["search_visits"] == 128
    assert "internal_extra" not in manifest["selfplay"]
    assert manifest["evaluation"]["eval_every"] == 5
    assert manifest["training"]["batch_size"] == 64

    checkpoint = payload["checkpoint"]
    assert checkpoint["name"] == "epoch_000002.pt"
    assert checkpoint["size"] > 0
    assert checkpoint["mtime"] > 0


def test_training_epoch_missing_epoch_returns_all_null_data_fields(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEXO_DEBUG_RUN_ROOT", raising=False)
    run_dir = tmp_path / "runs" / "dense_cnn_epoch_bare"
    (run_dir / "diagnostics").mkdir(parents=True)

    payload = web._training_epoch("dense_cnn_epoch_bare", 99)

    assert payload["run"] == "dense_cnn_epoch_bare"
    assert payload["epoch"] == 99
    for key in ("history", "prev_epoch", "evaluation", "diagnostics", "selfplay_extras", "manifest", "checkpoint"):
        assert payload[key] is None


def test_training_epoch_rejects_unknown_run_and_missing_epoch_param(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEXO_DEBUG_RUN_ROOT", raising=False)
    run_dir = tmp_path / "runs" / "dense_cnn_epoch_known"
    (run_dir / "diagnostics").mkdir(parents=True)

    with pytest.raises(ValueError):
        web._training_epoch("no_such_run", 1)
    with pytest.raises(ValueError):
        web._training_epoch("dense_cnn_epoch_known", None)


def test_training_epoch_checkpoint_name_falls_back_to_canonical(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.chdir(tmp_path)

    # Variant A: history row gains its checkpoint from the checkpoints-dir scan
    # ({path, bytes, modified} -- no "name" key); name derives from the path.
    monkeypatch.delenv("HEXO_DEBUG_RUN_ROOT", raising=False)
    local_run = tmp_path / "runs" / "dense_cnn_epoch_local"
    _write_json(local_run / "diagnostics" / "epoch_000002.json", _epoch_result_json(2, checkpoint=False))
    (local_run / "checkpoints").mkdir(parents=True)
    (local_run / "checkpoints" / "epoch_000002.pt").write_bytes(b"local")

    payload = web._training_epoch("dense_cnn_epoch_local", 2)
    assert payload["checkpoint"]["name"] == "epoch_000002.pt"
    assert payload["checkpoint"]["size"] == len(b"local")

    # Variant B: no checkpoint entry at all (mirror cwd lacks checkpoints/);
    # the canonical epoch_{N:06d}.pt name resolves under HEXO_DEBUG_RUN_ROOT.
    mirror_run = tmp_path / "runs" / "dense_cnn_epoch_mirror"
    _write_json(mirror_run / "diagnostics" / "epoch_000002.json", _epoch_result_json(2, checkpoint=False))
    worktree = tmp_path / "worktree"
    wt_ckpts = worktree / "runs" / "dense_cnn_epoch_mirror" / "checkpoints"
    wt_ckpts.mkdir(parents=True)
    (wt_ckpts / "epoch_000002.pt").write_bytes(b"worktree-ckpt")
    monkeypatch.setenv("HEXO_DEBUG_RUN_ROOT", str(worktree))

    payload = web._training_epoch("dense_cnn_epoch_mirror", 2)
    assert payload["history"]["epoch"] == 2
    assert payload["history"].get("checkpoint") is None
    assert payload["checkpoint"]["name"] == "epoch_000002.pt"
    assert payload["checkpoint"]["size"] == len(b"worktree-ckpt")
