"""Runner player contracts.

Players are opaque synchronous adapters. The runner owns game execution and
authoritative state; players receive cloned mutable engine states and return
actions for the runner to submit to the engine.

This is THE cross-package player contract. Implementers:
- packages/dense_cnn_restnet/python/dense_cnn_restnet/player.py
- packages/hexo_models/dense_cnn/python/hexo_models/dense_cnn/player.py
- packages/hexo_models/hexgt/python/hexo_models/hexgt/player.py
- packages/hexgnn/python/hexgnn/player.py
- hexo_runner/adapters/sealbot.py (SealBotPlayer)
- packages/hexo_frontend/python/hexo_frontend/web.py (human/checkpoint bot
  wrappers for the Match-v2 Arena)
Consumed by hexo_runner/loop.py (lifecycle order: setup_worker -> start_game
-> decide/observe_transition* -> finish_game -> close).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from hexo_engine import Action, HexoState


@dataclass(frozen=True, slots=True)
class PlayerIdentity:
    """Stable identity for a participant in runner output."""

    player_id: str
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkerContext:
    """Long-lived worker setup context.

    This intentionally contains no game state.
    """

    worker_id: int
    engine_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GameContext:
    """Per-game player context.

    This intentionally contains no authoritative state handle.
    """

    game_id: str
    seed: int | None
    player_index: int
    player_role: str
    opponent: PlayerIdentity
    mode: str = "match"
    is_evaluation: bool = False
    engine_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Participant response consumed by the runner.

    Players receive only a cloned `HexoState` in `decide`, query whatever they
    need from `hexo_engine`, and return one action plus optional diagnostics.
    There is no refusal/forfeit path; errors abort the game.
    """

    # The action the runner will submit to `hexo_engine.apply_action`.
    action: Action
    # Player-owned debug data transported into the position record.
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action is None:
            raise ValueError("DecisionResult.action is required.")


@dataclass(frozen=True, slots=True)
class TransitionEvent:
    """Notification sent to participants after an engine transition.

    This is sent after the action has been accepted and applied to the primary
    engine state.
    """

    game_id: str
    action_index: int
    player_id: str
    player_role: str
    action_id: int
    action: Action
    transition: object
    terminal: object | None
    state: HexoState


@dataclass(frozen=True, slots=True)
class FinalSummary:
    """Final runner summary passed to players during cleanup."""

    game_id: str
    result: object
    metadata: Mapping[str, Any] = field(default_factory=dict)


class RunnerPlayer(Protocol):
    """Protocol implemented by all runner participants."""

    identity: PlayerIdentity

    def setup_worker(self, context: WorkerContext) -> None:
        """Prepare long-lived resources once per worker."""

    def start_game(self, context: GameContext) -> None:
        """Reset per-game mutable state before the first decision."""

    def decide(self, state: HexoState) -> DecisionResult:
        """Choose an action from a cloned, player-owned engine state."""

    def observe_transition(self, transition: TransitionEvent) -> None:
        """Observe an accepted engine transition."""

    def finish_game(self, final_summary: FinalSummary) -> None:
        """Observe the final result and clear per-game state."""

    def close(self) -> None:
        """Release long-lived resources when the worker is done."""


class PlayerFactory(Protocol):
    """Pickleable factory used by batch workers to build reusable players."""

    def create_player(self) -> RunnerPlayer:
        """Create one long-lived runner player inside a worker process."""
