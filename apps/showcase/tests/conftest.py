"""Showcase test fixtures.

Arch env is pinned BEFORE any shrimp import (the constants are read once at
import time): a smoke-size c=32 / 4-head / CCA net at support radius 4 — the
same radius the shipped main_7 weights use, so the radius-sensitive featurizer
paths are exercised as deployed.

The tiny checkpoint is written both to a tmp dir (used by the generated
bots.toml) and to `tests/data/tiny_bot.pt`, the path `bots.example.toml`
references, so the committed example config works after one test run.

One session-scoped app/client pair keeps the worker-pool spawn (torch import)
to a single cost for the whole suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Load-bearing: set the arch env before shrimp (imported transitively by the
# showcase worker code and by tests that build the tiny net) is first imported.
os.environ["SHRIMP_CHANNELS"] = "32"
os.environ["SHRIMP_ATTENTION_HEADS"] = "4"
os.environ["SHRIMP_TRUNK"] = "CCA"
os.environ["SHRIMP_SUPPORT_RADIUS"] = "4"

_REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (
    _REPO_ROOT / "packages" / "shrimp" / "python",
    _REPO_ROOT / "apps" / "showcase" / "server",
):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import pytest

from showcase.config import Settings


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """A real smoke-size shrimp checkpoint (env-default arch, random weights)."""
    import torch

    from shrimp.model import ShrimpNet

    payload = {
        "meta": {"lineage": "shrimp", "epoch": 0, "run": "showcase_tiny_test"},
        "model": ShrimpNet().state_dict(),
        "optimizer": None,
    }
    path = tmp_path_factory.mktemp("ckpt") / "tiny_bot.pt"
    torch.save(payload, path)
    # Materialize the copy bots.example.toml points at (gitignored).
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)
    torch.save(payload, data_dir / "tiny_bot.pt")
    return path


@pytest.fixture(scope="session")
def bots_toml(tiny_checkpoint, tmp_path_factory) -> Path:
    """Two-checkpoint catalogue over the tiny net, allowed sims {8, 16}.

    The second entry serves the same weights through the as-trained main_5
    PUCT profile with the `group`/`search` display keys, so the suite
    exercises per-checkpoint profile routing and the picker metadata end to
    end (each worker parses two profiles).
    """
    path = tmp_path_factory.mktemp("cfg") / "bots.toml"
    path.write_text(
        f"""sims = [8, 16]

[[checkpoint]]
id = "tiny"
checkpoint = '{tiny_checkpoint.as_posix()}'
label = "Tiny test bot"
run = "showcase_tiny_test"
epoch = 0
games_trained = 12345

[[checkpoint]]
id = "tiny-puct"
checkpoint = '{tiny_checkpoint.as_posix()}'
label = "Tiny PUCT bot"
run = "showcase_tiny_test"
epoch = 0
search_profile = "shrimp_main_5"
group = "earlier runs"
search = "puct"
"""
    )
    return path


@pytest.fixture(scope="session")
def settings(bots_toml, tmp_path_factory) -> Settings:
    return Settings(
        db_path=tmp_path_factory.mktemp("db") / "showcase.db",
        bots_toml=bots_toml,
        search_config=_REPO_ROOT / "configs" / "shrimp_main_7.toml",
        static_dir=_REPO_ROOT / "apps" / "showcase" / "web",
        workers=1,
        max_active_games=16,
        max_games_per_ip=2,
        moves_per_minute=100_000,
        analysis_per_minute=100_000,
        games_per_hour=100_000,
        idle_timeout_s=3600.0,
        bot_timeout_s=60.0,
        finished_ttl_s=3600.0,
        sweep_interval_s=3600.0,
        analysis_search_visit_cap=32,
        policy_floor=1e-4,
        torch_threads=2,
        ip_salt="test-salt",
    )


@pytest.fixture(scope="session")
def client(settings):
    from fastapi.testclient import TestClient

    from showcase.app import create_app

    with TestClient(create_app(settings)) as test_client:
        yield test_client
