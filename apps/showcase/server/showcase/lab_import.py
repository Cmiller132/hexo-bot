"""hexo.did.science import: fetch and normalize community positions.

The community site serves two JSON shapes the lab can hold:

- sandbox positions (``GET /api/sandbox-positions/{id}``):
  ``gamePosition.cells`` with a ``moveId`` placement order, ``player-1`` /
  ``player-2`` owners, and the side to move in ``currentTurnPlayer``;
- finished games (``GET /api/finished-games/{id}``): ``moves`` with a
  ``moveNumber`` order and a per-move ``playerId`` — the first mover is the
  origin opener.

Their API sends no CORS headers, so a browser on this site's origin cannot
fetch it directly; the showcase proxies exactly the two endpoints above and
nothing else (fixed host, fixed path templates, allowlisted ids). Their
``(x, y)`` axial coordinates map to the engine's ``(q, r)`` unchanged —
engine-verified against captured fixtures in the tests.

Normalization prefers a legal action sequence: the record order must follow
the engine's 1-2-2 turn structure and replay clean (a finished game may end
on the winning placement). Either site label may be the origin opener, so the
seat mapping comes from the first record, never from the label. A sandbox
whose record does not reconstruct — turn parity broken, an illegal replay, or
a stated side-to-move that contradicts the record (their editor allows
hand-built and hand-edited positions) — falls back to a free-edit stone set.
A game record that breaks the turn structure is corrupt and an error, never a
guess.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

import hexo_engine as engine
from hexo_engine.types import AxialCoord, PlacementAction

from .lab_rules import LabPositionError, validate_free_stones

HOST = "hexo.did.science"
_PATHS = {
    "sandbox": "/api/sandbox-positions/",
    "game": "/api/finished-games/",
}
_ID_RE = re.compile(r"[A-Za-z0-9-]{1,64}")
# Their longest real games are tens of KB; the cap only bounds a misbehaving
# response, it is not a tuning knob.
_BODY_MAX = 4 * 1024 * 1024
_TIMEOUT_S = 10.0


class LabImportError(ValueError):
    """str(exc) is the client-facing message; ``status`` picks the HTTP code."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def _record_owner(i: int) -> int:
    """Owner (0/1) of placement-record index ``i`` under the 1-2-2 structure."""
    if i == 0:
        return 0
    return 1 - (((i - 1) // 2) % 2)


def fetch_source(kind: str, source_id: str) -> dict:
    """One GET against the community API; JSON dict or LabImportError."""
    url = f"https://{HOST}{_PATHS[kind]}{source_id}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "hexo-showcase-lab-import",
    })
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            body = resp.read(_BODY_MAX + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise LabImportError(f"{HOST} has no {kind} {source_id!r}", 404) from exc
        raise LabImportError(f"{HOST} answered {exc.code}", 502) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LabImportError(f"{HOST} did not answer: {exc}", 502) from exc
    if len(body) > _BODY_MAX:
        raise LabImportError(f"{HOST} sent an oversized response", 502)
    try:
        doc = json.loads(body)
    except ValueError as exc:
        raise LabImportError(f"{HOST} sent malformed JSON", 502) from exc
    if not isinstance(doc, dict):
        raise LabImportError(f"{HOST} sent an unexpected payload shape", 502)
    return doc


def _replay(cells: list[tuple[int, int]], *, what: str) -> bool:
    """Engine replay; True when the final state is terminal. Terminal is legal
    only at the last placement (a finished game's winning move)."""
    state = engine.new_game()
    for i, (q, r) in enumerate(cells):
        action = PlacementAction(AxialCoord(q=int(q), r=int(r)))
        if not engine.is_legal_action(state, action):
            raise LabImportError(f"{what}: placement {i} at ({q}, {r}) is illegal")
        result = engine.apply_action(state, action)
        if result.terminal and i < len(cells) - 1:
            raise LabImportError(f"{what}: the game ends at placement {i} but moves continue")
    return engine.terminal(state) is not None


def _int_pair(row: dict, what: str) -> tuple[int, int]:
    x, y = row.get("x"), row.get("y")
    if not isinstance(x, int) or not isinstance(y, int):
        raise LabImportError(f"{what}: a cell is missing integer x/y")
    return (x, y)


def _normalize_sandbox(doc: dict, source_id: str) -> dict:
    position = doc.get("gamePosition")
    if not isinstance(position, dict) or not isinstance(position.get("cells"), list):
        raise LabImportError(f"{HOST} sandbox {source_id!r} has no gamePosition.cells", 502)
    rows = sorted(position["cells"], key=lambda c: c.get("moveId", 0))
    if not rows:
        raise LabImportError(f"{HOST} sandbox {source_id!r} is empty")
    cells = [_int_pair(row, "sandbox") for row in rows]
    players = []
    for row in rows:
        player = row.get("player")
        if player not in ("player-1", "player-2"):
            raise LabImportError(f"sandbox: unknown owner {player!r}")
        players.append(player)
    turn = position.get("currentTurnPlayer")
    if turn not in ("player-1", "player-2"):
        raise LabImportError(f"sandbox: unknown currentTurnPlayer {turn!r}")
    remaining = position.get("placementsRemaining")
    name = doc.get("name") if isinstance(doc.get("name"), str) else source_id

    sequence = _sandbox_sequence(cells, players, turn, remaining)
    if sequence is not None:
        return {"name": name, **sequence}

    # The record does not reconstruct a game: free-edit is the one honest
    # representation. The side to move comes from the sandbox itself, and the
    # site labels keep their color slots (there is no opener to anchor them).
    p0 = [c for c, p in zip(cells, players) if p == "player-1"]
    p1 = [c for c, p in zip(cells, players) if p == "player-2"]
    try:
        validate_free_stones(p0, p1)
    except LabPositionError as exc:
        raise LabImportError(f"sandbox: {exc}") from exc
    return {
        "name": name,
        "stones": {"p0": [[q, r] for q, r in p0], "p1": [[q, r] for q, r in p1]},
        "to_move": 0 if turn == "player-1" else 1,
    }


def _sandbox_sequence(
    cells: list[tuple[int, int]], players: list[str], turn: str, remaining,
) -> dict | None:
    """The sequence normalization of a sandbox record, or ``None`` when the
    record does not reconstruct a game (the free-edit fallback decides then).

    The opener — the owner of the first record — fixes the seat mapping, as in
    ``_normalize_game``. A non-terminal record must also agree with the
    sandbox's own turn state: a stated mover or placements-remaining that
    contradicts the record order means the position was hand-edited past it,
    and the record order is no longer the position's history.
    """
    owners = [0 if p == players[0] else 1 for p in players]
    if any(_record_owner(i) != o for i, o in enumerate(owners)):
        return None
    try:
        terminal = _replay(cells, what="sandbox")
    except LabImportError:
        return None
    if not terminal:
        next_owner = _record_owner(len(cells))
        opener, other = players[0], ("player-1", "player-2")[players[0] == "player-1"]
        if turn != (opener if next_owner == 0 else other):
            return None
        if remaining != (2 if len(cells) % 2 == 1 else 1):
            return None
    return {"moves": [[q, r] for q, r in cells], "terminal": terminal}


def _normalize_game(doc: dict, source_id: str) -> dict:
    moves = doc.get("moves")
    if not isinstance(moves, list) or not moves:
        raise LabImportError(f"{HOST} game {source_id!r} has no moves", 502)
    rows = sorted(moves, key=lambda m: m.get("moveNumber", 0))
    cells = [_int_pair(row, "game") for row in rows]
    first = rows[0].get("playerId")
    if not isinstance(first, str):
        raise LabImportError("game: the first move names no playerId")
    owners = [0 if row.get("playerId") == first else 1 for row in rows]
    if any(_record_owner(i) != o for i, o in enumerate(owners)):
        raise LabImportError("game: the move record breaks the 1-2-2 turn structure")
    terminal = _replay(cells, what="game")
    players = doc.get("players")
    name = source_id
    if isinstance(players, list) and len(players) == 2:
        names = [p.get("displayName") for p in players if isinstance(p, dict)]
        if len(names) == 2 and all(isinstance(n, str) for n in names):
            name = f"{names[0]} vs {names[1]}"
    return {
        "name": name,
        "moves": [[q, r] for q, r in cells],
        "terminal": terminal,
    }


def import_position(kind: str, source_id: str, fetch=None) -> dict:
    """Fetch + normalize one community position; LabImportError on anything
    that is not a clean import."""
    if kind not in _PATHS:
        raise LabImportError(f"kind must be sandbox or game, not {kind!r}")
    if not _ID_RE.fullmatch(source_id):
        raise LabImportError(f"{source_id!r} is not a {HOST} id")
    doc = (fetch or fetch_source)(kind, source_id)
    if kind == "sandbox":
        payload = _normalize_sandbox(doc, source_id)
    else:
        payload = _normalize_game(doc, source_id)
    return {"source": HOST, "kind": kind, "id": source_id, **payload}
