"""Runner game and batch specifications.

`GameSpec` is the per-game request consumed by `hexo_runner/loop.py` and
`hexo_runner/modes/match.py`; the external constructor is
packages/hexo_frontend/python/hexo_frontend/web.py (Match-v2 Arena). `BatchSpec`
is a retained batch-request dataclass with no in-repo caller (the local
multiprocessing batch runner it fed was removed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .player import PlayerFactory


@dataclass(frozen=True, slots=True)
class GameSpec:
    """Inputs needed to create one engine-backed game.

    The runner passes `seed` through to `hexo_engine.new_game`, but note the
    engine currently DISCARDS it (hexo_engine/rust/src/pybridge.rs new_game
    ignores both seed and scenario); the seed's real effects are that it is
    persisted in the .hxr record header and exposed to players via
    `GameContext.seed`. Durable runner records do not persist scenarios yet,
    so recorded runs require `scenario=None` (`run_match_loop` raises
    otherwise).
    """

    game_id: str
    seed: int | None = None
    scenario: object | None = None
    mode: str = "match"
    is_evaluation: bool = False
    max_actions: int = 1024
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_actions <= 0:
            raise ValueError("GameSpec.max_actions must be positive.")


@dataclass(frozen=True, slots=True)
class BatchSpec:
    """Local multiprocessing batch request for this machine.

    `worker_count=None` was intended to auto-pick the pool size; games are
    dealt round-robin in `chunk_size` blocks, one `{batch_id}-worker-{id}.hxr`
    file per worker under `output_dir`. Retained as a stable dataclass; the
    batch runner that consumed it has been removed.
    """

    batch_id: str
    games: Sequence[GameSpec]
    player_factories: tuple[PlayerFactory, PlayerFactory]
    output_dir: str | Path = Path("data/replay")
    worker_count: int | None = None
    chunk_size: int = 32
    metadata: Mapping[str, Any] = field(default_factory=dict)
