"""One human-vs-bot game: engine session, turn phases, terminal detection,
client serialization, and `.hxr` record encode/decode.

The Rust engine (`hexo_engine`) is the rules authority — this module drives it
directly (apply move -> check terminal), never through a match controller.
The `.hxr` codec (`hexo_utils.records`) is file-based, so blobs round-trip
through a temp file; a full game is well under a kilobyte.

Colors are 0/1 (player0 moves first). Turn shape: an Opening turn places one
stone (forced to the origin — auto-applied at session creation, so live games
start at ply 1), every later turn places two — the engine owns the phase, this
module only maps it to a stones-left count for the client.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import hexo_engine as engine
from hexo_engine.types import AxialCoord, PlacementAction, TurnPhase, pack_coord_id, unpack_coord_id
from hexo_utils.records import AbortRecord, HexoRecordFile, HexoRecordPlayer

# Stones the mover still places this turn, given the engine phase (the engine
# owns phase transitions; this is presentation only).
PHASE_STONES_LEFT: dict[TurnPhase, int] = {
    TurnPhase.OPENING: 1,
    TurnPhase.FIRST_STONE: 2,
    TurnPhase.SECOND_STONE: 1,
}

TERMINATION_SIX_IN_LINE = "six_in_line"
TERMINATION_RESIGN = "resign"
TERMINATION_TIMEOUT = "timeout"

# The three line axes of the board (opposite directions are the same line).
_LINE_AXES = ((1, 0), (0, 1), (1, -1))


def now_iso() -> str:
    """UTC wall-clock timestamp for DB rows."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _player_index(player: object) -> int:
    """Engine Player ('player0'/'player1') -> 0/1."""
    return 1 if str(getattr(player, "value", player)).endswith("1") else 0


def stones_in_placement_order(mirror: Any) -> list[dict[str, int]]:
    """The client's `stones` list, in PLACEMENT ORDER (the engine's
    `board.stones` is (q, r)-sorted; `placement_history` is chronological, and
    stones are never removed, so the two hold the same cells). The client
    derives "last two moves" from the tail of this list."""
    return [
        {"q": rec.coord.q, "r": rec.coord.r, "color": _player_index(rec.player)}
        for rec in mirror.placement_history
    ]


def find_winning_line(stones: list[dict[str, int]], winner: int) -> list[dict[str, int]] | None:
    """The cells of the completed line in a six_in_line game.

    The engine detects the win (a full six-cell window) but does not export
    the cells, so this rescans: the game ends on the placement that completed
    the line, so the winning run passes through the LAST stone of the winner's
    color; walk the three axes through it and return the longest contiguous
    run of length >= 6 (a placement that joins two runs can make it longer
    than six). Cells are ordered along the line. Falls back to a full scan
    over the winner's stones (defensive; unreachable for engine-produced
    games), and returns None if no run of six exists.
    """
    cells = {(s["q"], s["r"]) for s in stones if s["color"] == winner}
    winner_stones = [s for s in stones if s["color"] == winner]
    candidates = [(winner_stones[-1]["q"], winner_stones[-1]["r"])] if winner_stones else []

    def _run_through(q0: int, r0: int) -> list[tuple[int, int]] | None:
        best: list[tuple[int, int]] | None = None
        for dq, dr in _LINE_AXES:
            q, r = q0, r0
            while (q - dq, r - dr) in cells:
                q, r = q - dq, r - dr
            run: list[tuple[int, int]] = []
            while (q, r) in cells:
                run.append((q, r))
                q, r = q + dq, r + dr
            if len(run) >= 6 and (best is None or len(run) > len(best)):
                best = run
        return best

    for q0, r0 in candidates + sorted(cells):
        run = _run_through(q0, r0)
        if run is not None:
            return [{"q": q, "r": r} for q, r in run]
    return None


def encode_hxr(
    *, game_id: str, bot_slug: str, human_color: int, action_ids: list[int],
    winner: int | None, termination: str | None, seed: int,
    human_role: str = "human",
) -> bytes:
    """Encode a finished game as `.hxr` bytes (the repo's binary game codec).

    Completed games (`six_in_line`/`resign`) close with the winner's player
    label; timed-out games close with an abort record — both keep the record
    replayable by the dev tools. The codec writes files, so this stages
    through a temp file.
    """
    roles = {human_color: human_role, 1 - human_color: f"bot:{bot_slug}"}
    players = tuple(
        HexoRecordPlayer(roles[color], f"player{color}", roles[color])
        for color in (0, 1)
    )
    with tempfile.TemporaryDirectory(prefix="showcase-hxr-") as tmp:
        path = os.path.join(tmp, "game.hxr")
        with HexoRecordFile.create(path, engine.engine_metadata(), players) as record_file:
            writer = record_file.begin_game(game_id, seed=seed)
            for aid in action_ids:
                writer.record_action(PlacementAction(unpack_coord_id(aid)))
            if winner is None:
                writer.finish_aborted(
                    AbortRecord(
                        stage="showcase",
                        exception_type="Abandoned",
                        message=f"game abandoned ({termination or 'bot_error'})",
                    )
                )
            else:
                writer.finish_completed(f"player{winner}", len(action_ids))
        with open(path, "rb") as fh:
            return fh.read()


def decode_hxr_actions(blob: bytes) -> list[int]:
    """Packed action ids of the (single) game stored in an `.hxr` blob."""
    with tempfile.TemporaryDirectory(prefix="showcase-hxr-") as tmp:
        path = os.path.join(tmp, "game.hxr")
        with open(path, "wb") as fh:
            fh.write(blob)
        record_file = HexoRecordFile.open(path)
        records = list(record_file.iter_records())
        record_file.close()
    if len(records) != 1:
        raise ValueError(f"showcase .hxr blob holds {len(records)} games, expected 1")
    return [int(a) for a in records[0].action_ids]


@dataclass
class GameSession:
    """In-memory state of one live (or recently finished) game.

    The engine `state` handle is authoritative; `actions` mirrors it as packed
    action ids in move order (the future `.hxr` record). All mutation happens
    on the event loop under `lock` — the bot-turn task and the idle sweeper
    both take it, so a game can never be finalized mid-application.
    """

    game_id: str
    token: str
    bot_slug: str
    bot_db_id: int
    bot_label: str
    bot_epoch: int
    sims: int
    human_color: int
    client_hash: str
    # Who occupies the non-server seat: "human" (browser play, cookie auth) or
    # "match" (external agent via /api/match, bearer auth). Endpoints of one
    # kind never operate on sessions of the other.
    kind: str = "human"
    # Role label written into the .hxr record for the non-server seat
    # ("agent:<name>" for matches).
    human_role: str = "human"
    game_key: int = field(default_factory=lambda: int.from_bytes(os.urandom(6), "big"))
    seed: int = field(default_factory=lambda: int.from_bytes(os.urandom(6), "big"))
    state: Any = field(default_factory=engine.new_game)
    actions: list[int] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    bot_busy: bool = False
    bot_failed: bool = False  # transient bot-turn failure; retryable, not terminal
    db_status: str = "active"  # active | finished | abandoned
    result: int | None = None  # +1 human, -1 bot, 0 none (schema convention)
    termination: str | None = None
    started_at: str = field(default_factory=now_iso)
    started_mono: float = field(default_factory=time.monotonic)
    last_activity: float = field(default_factory=time.monotonic)
    nickname: str | None = None
    winning_line: list[dict[str, int]] | None = None  # set at six_in_line finalize

    @classmethod
    def create(
        cls, *, bot_slug: str, bot_db_id: int, bot_label: str, bot_epoch: int,
        sims: int, human_color: int, client_hash: str,
    ) -> "GameSession":
        session = cls(
            game_id=str(uuid.uuid4()),
            token=uuid.uuid4().hex,
            bot_slug=bot_slug,
            bot_db_id=bot_db_id,
            bot_label=bot_label,
            bot_epoch=bot_epoch,
            sims=sims,
            human_color=human_color,
            client_hash=client_hash,
        )
        # The opening single is forced to the origin, so place it at creation
        # rather than searching it (a 1-legal-move search is degenerate) or
        # making the human click it. Every game therefore starts at ply 1 with
        # player1 to move.
        engine.apply_action(session.state, PlacementAction(AxialCoord(q=0, r=0)))
        session.actions.append(pack_coord_id(AxialCoord(q=0, r=0)))
        return session

    # -- state queries ---------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.db_status == "active"

    @property
    def to_move(self) -> int | None:
        """Color to move, or None once terminal."""
        if engine.terminal(self.state) is not None:
            return None
        return _player_index(engine.current_player(self.state))

    @property
    def bot_to_move(self) -> bool:
        return self.active and self.to_move == 1 - self.human_color

    def touch(self) -> None:
        self.last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_activity

    # -- moves -------------------------------------------------------------------

    def apply_human_move(self, q: int, r: int) -> None:
        """Validate and apply one human placement. Raises `IllegalActionError`
        if the engine rejects the cell; callers must have checked turn
        ownership and liveness first."""
        engine.apply_action(self.state, PlacementAction(AxialCoord(q=int(q), r=int(r))))
        self.actions.append(pack_coord_id(AxialCoord(q=int(q), r=int(r))))
        self.touch()

    def apply_bot_actions(self, action_ids: list[int]) -> None:
        """Apply a bot turn computed by a worker. The engine re-validates each
        placement (the worker searched a replay of the same actions, so a
        rejection here means a server bug, not user input)."""
        for aid in action_ids:
            engine.apply_action(self.state, PlacementAction(unpack_coord_id(int(aid))))
            self.actions.append(int(aid))
        self.touch()

    def engine_winner(self) -> int | None:
        """Winning color if the engine says the game is over, else None."""
        terminal = engine.terminal(self.state)
        if terminal is None or terminal.winner is None:
            return None
        return _player_index(terminal.winner)

    # -- finalization ------------------------------------------------------------

    def finalize(self, *, termination: str | None, winner: int | None) -> bytes:
        """Mark the session finished and return the `.hxr` record blob.

        `winner` is a color (0/1) or None (abandonment). `termination` is
        `six_in_line`/`resign` (status finished) or `timeout`/None (status
        abandoned; None is the internal bot-failure path). Sets the schema's
        result convention: +1 human win, -1 bot win, 0 none.
        """
        self.termination = termination
        self.db_status = (
            "finished"
            if termination in (TERMINATION_SIX_IN_LINE, TERMINATION_RESIGN)
            else "abandoned"
        )
        if winner is None:
            self.result = 0
        else:
            self.result = 1 if winner == self.human_color else -1
        if termination == TERMINATION_SIX_IN_LINE and winner is not None:
            mirror = engine.to_python_state(self.state)
            self.winning_line = find_winning_line(
                stones_in_placement_order(mirror), winner
            )
        self.touch()
        return encode_hxr(
            game_id=self.game_id,
            bot_slug=self.bot_slug,
            human_color=self.human_color,
            action_ids=self.actions,
            winner=winner,
            termination=termination,
            seed=self.seed,
            human_role=self.human_role,
        )

    def duration_s(self) -> float:
        return time.monotonic() - self.started_mono

    # See also `finished_snapshot` below: the DB-backed twin of `snapshot()`
    # for games whose in-memory session is gone.

    # -- client serialization ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """The poll payload for GET /api/game/{id} (the Phase-2 client contract).

        `stones` is in PLACEMENT ORDER (see `stones_in_placement_order`);
        `winning_line` is non-null only on six_in_line finishes.
        """
        mirror = engine.to_python_state(self.state)
        finished = not self.active
        if finished:
            status = "finished"
        elif self.bot_failed:
            # bot turn hiccuped mid-search; position intact, awaiting a retry
            status = "bot_failed"
        elif self.bot_busy or self.bot_to_move:
            status = "bot_thinking"
        else:
            status = "your_turn"
        stones = stones_in_placement_order(mirror)
        result = None
        if finished:
            winner = None
            if self.result:
                winner = self.human_color if self.result == 1 else 1 - self.human_color
            result = {
                "winner": winner,
                "termination": self.termination,
                "human_result": self.result,
            }
        return {
            "id": self.game_id,
            "status": status,
            "bot": {
                "checkpoint_id": self.bot_slug,
                "label": self.bot_label,
                "epoch": self.bot_epoch,
                "sims": self.sims,
            },
            "human_color": self.human_color,
            "to_move": self.to_move,
            "phase": str(mirror.phase.value),
            "stones_left_this_turn": PHASE_STONES_LEFT[mirror.phase],
            "ply": len(self.actions),
            "stones": stones,
            "legal": (
                [{"q": coord.q, "r": coord.r} for coord in mirror.board.legal]
                if status == "your_turn"
                else []
            ),
            "last_move": stones[-1] if stones else None,
            "winning_line": self.winning_line,
            "result": result,
            "nickname": self.nickname,
        }


def finished_snapshot(
    *, game_id: str, actions: list[int], bot: dict[str, Any], human_color: int,
    result: int | None, termination: str | None, nickname: str | None,
    finished_at: str | None,
) -> dict[str, Any]:
    """GET /api/game/{id} payload for a finished game served FROM THE DB (its
    in-memory session is gone: evicted after the finished TTL or lost to a
    restart). Replays the `.hxr` actions through the engine and emits the same
    shape as `GameSession.snapshot()` plus `finished_at`, so the client renders
    live and archived games identically. `bot` is the bots-table row dict.
    """
    state = engine.new_game()
    for aid in actions:
        engine.apply_action(state, PlacementAction(unpack_coord_id(int(aid))))
    mirror = engine.to_python_state(state)
    stones = stones_in_placement_order(mirror)
    winner = None
    if result:
        winner = human_color if result == 1 else 1 - human_color
    winning_line = None
    if termination == TERMINATION_SIX_IN_LINE and winner is not None:
        winning_line = find_winning_line(stones, winner)
    return {
        "id": game_id,
        "status": "finished",
        "bot": {
            "checkpoint_id": bot["slug"],
            "label": bot["label"],
            "epoch": bot["epoch"],
            "sims": bot["visits"],
        },
        "human_color": human_color,
        "to_move": None,
        "phase": str(mirror.phase.value),
        "stones_left_this_turn": PHASE_STONES_LEFT[mirror.phase],
        "ply": len(actions),
        "stones": stones,
        "legal": [],
        "last_move": stones[-1] if stones else None,
        "winning_line": winning_line,
        "result": {
            "winner": winner,
            "termination": termination,
            "human_result": result,
        },
        "nickname": nickname,
        "finished_at": finished_at,
    }
