"""Python runner package for headless Hexo execution.

The runner owns sessions, player lifecycle, single-game loops, run modes, and
record/result emission. It applies actions through `hexo_engine` and treats
model/search internals as player-owned details.

Active surfaces (see README.md in this package for the full status table):
the RunnerPlayer contracts (player.py), the single-game loop (loop.py) and
match mode, the .hxr records facade (records/, re-exporting hexo_utils), and
the SealBot adapter (adapters/sealbot.py). All four model packages, the
hexo_frontend dashboard, and the test suite depend on these; batch mode and
the evaluation stub are test-only/never-built respectively.
"""

__version__ = "0.1.0"

from .player import (
    DecisionResult,
    FinalSummary,
    GameContext,
    PlayerFactory,
    PlayerIdentity,
    RunnerPlayer,
    TransitionEvent,
    WorkerContext,
)
from .records import BatchResult, GameResult, GameStatus, HexoRecord, HexoRecordFile
from .session import BatchSpec, GameSpec

__all__ = [
    "BatchResult",
    "BatchSpec",
    "DecisionResult",
    "FinalSummary",
    "GameContext",
    "GameResult",
    "GameSpec",
    "GameStatus",
    "HexoRecord",
    "HexoRecordFile",
    "PlayerFactory",
    "PlayerIdentity",
    "RunnerPlayer",
    "TransitionEvent",
    "WorkerContext",
]
