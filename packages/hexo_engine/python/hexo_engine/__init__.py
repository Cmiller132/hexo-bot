"""Python package boundary for the Rust Hexo engine.

The engine owns canonical rules, legal transitions, tactical state, and stable
action identity. Python callers should come through this package instead of
duplicating game logic.

Re-exports the typed wrappers from `.api` (backed by the maturin-built
`hexo_engine._rust` extension, rust/src/pybridge.rs), the transport types from
`.types`, and the error hierarchy from `.errors`. Consumers: hexo_runner
(engine adapter + sealbot adapter), hexo_frontend (board replay/rendering),
and all model packages' selfplay/eval/player glue. See README.md in this
package for the full contract map.
"""

from .api import (
    HexoState,
    action_id,
    apply_action,
    clone_state,
    current_player,
    engine_metadata,
    is_legal_action,
    legal_action_count,
    legal_action_ids,
    legal_actions,
    new_game,
    terminal,
    to_python_state,
)
from .errors import (
    EngineUnavailableError,
    HexoEngineError,
    IllegalActionError,
)
from .types import (
    Action,
    ActionId,
    AxialCoord,
    LegalActionId,
    LegalActions,
    PlacementAction,
    Player,
    PythonBoard,
    PythonHexoState,
    PythonMoveRecord,
    PythonPlacementRecord,
    PythonTerminal,
    PythonWindowEntry,
    PythonWindowKey,
    PythonWindowStore,
    TerminalResult,
    TransitionResult,
    TurnPhase,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "ActionId",
    "AxialCoord",
    "EngineUnavailableError",
    "HexoState",
    "HexoEngineError",
    "IllegalActionError",
    "LegalActionId",
    "LegalActions",
    "PlacementAction",
    "Player",
    "PythonBoard",
    "PythonHexoState",
    "PythonMoveRecord",
    "PythonPlacementRecord",
    "PythonTerminal",
    "PythonWindowEntry",
    "PythonWindowKey",
    "PythonWindowStore",
    "TerminalResult",
    "TransitionResult",
    "TurnPhase",
    "action_id",
    "apply_action",
    "clone_state",
    "current_player",
    "engine_metadata",
    "is_legal_action",
    "legal_action_count",
    "legal_action_ids",
    "legal_actions",
    "new_game",
    "terminal",
    "to_python_state",
]
