"""External bot match API: /api/match/*.

Lets third-party bot developers play RATED matches against any catalogue
checkpoint over plain HTTP — no browser, no cookie. A match is the same
in-memory `GameSession` (and the same DB row, stats, feed and ELO fold) as a
human game; the external agent simply occupies the "human" seat, identified by
an agent name that is recorded as the game's nickname up front. Everything the
existing robustness machinery does for human games (bot-turn retries, worker
failover, idle sweeping, active-game caps) applies to matches unchanged.

Protocol (full walkthrough in web/bot-api.md — also served on the site's API
tab with a download link; Python SDK in sdk/, mirrored at web/ for download):

    POST /api/match                    create a match -> {match_id, token, state}
    GET  /api/match/{id}?wait=25       state; long-polls until it's your turn,
                                       the game ends, or `wait` seconds elapse
    POST /api/match/{id}/move          place ONE stone {q, r} (turns after the
                                       opening place two stones = two calls)
    POST /api/match/{id}/resign
    POST /api/match/{id}/retry         re-run a hiccuped server-bot turn

Auth: the `token` returned by create rides every later call as
`Authorization: Bearer <token>`. It is shown exactly once — losing it forfeits
the match to the idle sweeper.

The router is built by `build_match_router` from a `MatchDeps` bundle of the
app's session/bucket/helper closures, so this module stays import-light and
`app.py` keeps sole ownership of the shared machinery.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from hexo_engine import IllegalActionError

from .game import GameSession

# Cap on the ?wait= long-poll, kept under common proxy/read timeouts
# (cloudflared defaults to 30s upstream) so a poll never 502s at the edge.
MAX_WAIT_S = 25.0
# Long-poll re-check cadence. 0.2s keeps worst-case turn-notification latency
# well under a search's think time while costing ~5 loop wakeups per waiting
# match per second — noise at the active-game cap.
_POLL_INTERVAL_S = 0.2

KIND_MATCH = "match"


class CreateMatchRequest(BaseModel):
    # The agent's public name: it becomes the game's nickname, so it is the
    # identity the feed, stats and ELO leaderboard aggregate under.
    agent: str = Field(min_length=1, max_length=64)
    checkpoint_id: str = Field(max_length=128)
    sims: int
    # Which color the AGENT plays: 0 moves first. Same convention and wire
    # values as the human create endpoint.
    agent_color: Literal[0, 1, "random"] = 0


class MatchMoveRequest(BaseModel):
    q: int = Field(ge=-(1 << 15), le=(1 << 15) - 1)
    r: int = Field(ge=-(1 << 15), le=(1 << 15) - 1)


@dataclass(frozen=True)
class MatchDeps:
    """The slice of app.py's closures the match endpoints drive.

    Everything here is the SAME machinery the human endpoints use — matches
    and human games share one session dict, one pool, one set of abuse caps.
    Lifespan-created state (sessions, catalogue, db) is read off
    `request.app.state` at call time, not captured here: the router is built
    before the lifespan runs.
    """

    settings: Any
    game_bucket: Any            # TokenBucket for match creation
    move_bucket: Any            # TokenBucket for move/resign/retry
    client_key: Callable[[Request], str]
    client_hash: Callable[[str], str]
    bot_db_id: Callable[[Any, int], int]
    start_bot_turn: Callable[[GameSession], None]
    finalize: Callable[..., None]
    resolve_color: Callable[[int | str], int]
    sanitize_agent: Callable[[str], str | None]
    termination_resign: str


def match_snapshot(session: GameSession) -> dict[str, Any]:
    """The match-state payload: the proven game snapshot, re-keyed for agents.

    `history` is the chronological move list (== the human payload's `stones`,
    which is in placement order); `you` is the agent's color; `legal` is
    non-empty exactly when `status == "your_turn"`.
    """
    snap = session.snapshot()
    return {
        "match_id": session.game_id,
        "agent": session.nickname,
        "you": session.human_color,
        "status": snap["status"],
        "to_move": snap["to_move"],
        "ply": snap["ply"],
        "phase": snap["phase"],
        "stones_left_this_turn": snap["stones_left_this_turn"],
        "history": snap["stones"],
        "legal": snap["legal"],
        "last_move": snap["last_move"],
        "winning_line": snap["winning_line"],
        "result": snap["result"],
        "bot": snap["bot"],
    }


def build_match_router(deps: MatchDeps) -> APIRouter:
    router = APIRouter(prefix="/api/match")

    def _bearer_token(request: Request) -> str:
        header = request.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(401, "missing bearer token (Authorization: Bearer <token>)")
        return token.strip()

    def _match_or_404(match_id: str, request: Request) -> GameSession:
        session = request.app.state.sessions.get(match_id)
        if session is None or session.kind != KIND_MATCH:
            raise HTTPException(404, "unknown or expired match")
        if _bearer_token(request) != session.token:
            raise HTTPException(403, "not your match")
        return session

    @router.post("")
    async def create_match(body: CreateMatchRequest, request: Request):
        state = request.app.state
        key = deps.client_key(request)
        if not deps.game_bucket.allow(key):
            raise HTTPException(429, "too many matches created; slow down")
        agent = deps.sanitize_agent(body.agent)
        if agent is None:
            raise HTTPException(422, "agent name has no allowed characters (A-Za-z0-9 _.-)")
        spec = state.catalogue.get(body.checkpoint_id)
        if spec is None:
            raise HTTPException(404, f"unknown checkpoint {body.checkpoint_id!r}")
        if body.sims not in state.sims_allowed:
            raise HTTPException(422, f"sims must be one of {list(state.sims_allowed)}")
        active = [s for s in state.sessions.values() if s.active]
        if len(active) >= deps.settings.max_active_games:
            raise HTTPException(429, "server is full; try again in a minute")
        client_hash = deps.client_hash(key)
        if sum(1 for s in active if s.client_hash == client_hash) >= deps.settings.max_games_per_ip:
            raise HTTPException(429, "active-game limit reached; finish or resign first")

        agent_color = deps.resolve_color(body.agent_color)
        bot_db_id = deps.bot_db_id(spec, body.sims)
        session = GameSession.create(
            bot_slug=spec.slug, bot_db_id=bot_db_id, bot_label=spec.label,
            bot_epoch=spec.epoch, sims=body.sims, human_color=agent_color,
            client_hash=client_hash,
        )
        session.kind = KIND_MATCH
        session.human_role = f"agent:{agent}"
        session.nickname = agent
        state.sessions[session.game_id] = session
        state.db.create_game(
            game_id=session.game_id, bot_id=bot_db_id,
            human_color=agent_color, started_at=session.started_at,
            client_hash=client_hash,
        )
        # Record the agent identity immediately: the feed/ELO read nicknames
        # off finished rows, and a match must count even if the agent never
        # bothers to sign it afterwards.
        state.db.set_nickname(session.game_id, agent)
        if session.bot_to_move:
            deps.start_bot_turn(session)
        return {
            "match_id": session.game_id,
            "token": session.token,
            "idle_timeout_s": deps.settings.idle_timeout_s,
            "state": match_snapshot(session),
        }

    @router.get("/{match_id}")
    async def get_match(
        match_id: str, request: Request,
        wait: float = Query(default=0.0, ge=0.0, le=MAX_WAIT_S),
    ):
        """Match state; with `wait`, long-polls until the agent can act.

        Returns as soon as `status` is anything other than "bot_thinking"
        (your_turn / bot_failed / finished), or after `wait` seconds. The
        loop reads flags the event loop itself mutates, so no lock is needed —
        same contract as the human GET.
        """
        session = _match_or_404(match_id, request)
        deadline = time.monotonic() + wait
        while (
            session.active
            and (session.bot_busy or session.bot_to_move)
            and not session.bot_failed
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(_POLL_INTERVAL_S)
        return match_snapshot(session)

    @router.post("/{match_id}/move")
    async def match_move(match_id: str, body: MatchMoveRequest, request: Request):
        """Place ONE stone. Post-opening turns place two stones: after the
        first, `status` stays "your_turn" and `stones_left_this_turn` is 1 —
        call again. The server bot's reply turn starts automatically once the
        agent's turn completes."""
        session = _match_or_404(match_id, request)
        if not deps.move_bucket.allow(deps.client_key(request)):
            raise HTTPException(429, "too many moves; slow down")
        async with session.lock:
            if not session.active:
                raise HTTPException(409, "match is finished")
            if session.bot_busy or session.to_move != session.human_color:
                raise HTTPException(409, "not your turn")
            try:
                session.apply_human_move(body.q, body.r)
            except IllegalActionError as exc:
                raise HTTPException(422, f"illegal move: {exc}") from exc
            winner = session.engine_winner()
            if winner is not None:
                deps.finalize(session, termination="six_in_line", winner=winner)
            elif session.bot_to_move:
                deps.start_bot_turn(session)
            return match_snapshot(session)

    @router.post("/{match_id}/resign")
    async def match_resign(match_id: str, request: Request):
        session = _match_or_404(match_id, request)
        if not deps.move_bucket.allow(deps.client_key(request)):
            raise HTTPException(429, "too many requests; slow down")
        async with session.lock:
            if not session.active:
                raise HTTPException(409, "match is finished")
            deps.finalize(
                session, termination=deps.termination_resign,
                winner=1 - session.human_color,
            )
            return match_snapshot(session)

    @router.post("/{match_id}/retry")
    async def match_retry(match_id: str, request: Request):
        """Re-run a server-bot turn that failed (`status == "bot_failed"`).
        The failed search never mutated the position, so this re-enqueues the
        same turn. Idempotent while a retry is already in flight."""
        session = _match_or_404(match_id, request)
        if not deps.move_bucket.allow(deps.client_key(request)):
            raise HTTPException(429, "too many requests; slow down")
        async with session.lock:
            if not session.active:
                raise HTTPException(409, "match is finished")
            if session.bot_busy:
                return match_snapshot(session)
            if not session.bot_failed or not session.bot_to_move:
                raise HTTPException(409, "no failed bot turn to retry")
            deps.start_bot_turn(session)
            return match_snapshot(session)

    return router
