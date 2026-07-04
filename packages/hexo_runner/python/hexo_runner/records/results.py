"""Runner result summaries.

In-memory return values of hexo_runner/loop.py and the modes; not persisted
themselves (the durable artifact is the .hxr record). hexo_frontend/web.py
consumes GameResult for live Arena match status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence

from .record import AbortRecord


class GameStatus(StrEnum):
    """Runner-level game completion status."""

    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class GameResult:
    """Summary for one game.

    `turns` counts recorded actions (stone placements), not two-stone turns;
    `duration_ms` is wall-clock; `terminal`/`winner` are populated only for
    COMPLETED games; `abort` only for ABORTED ones.
    """

    game_id: str
    status: GameStatus
    terminal: Mapping[str, Any] | None = None
    winner: object | None = None
    record_ref: object | None = None
    turns: int = 0
    duration_ms: float = 0.0
    abort: AbortRecord | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Summary for a local batch run."""

    batch_id: str
    total_games: int
    completed: int
    aborted: int
    worker_count: int
    duration_ms: float
    record_refs: Sequence[object] = ()
    aborts: Sequence[AbortRecord] = ()
    results: Sequence[GameResult] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
