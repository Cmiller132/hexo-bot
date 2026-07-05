"""FastAPI surface for the showcase.

Complete API (docs/showcase 01):

    POST /api/game                    create game
    GET  /api/game/{id}               state (poll)
    POST /api/game/{id}/move          human move
    POST /api/game/{id}/resign
    POST /api/game/{id}/nickname      set nickname on the finished game
    GET  /api/game/{id}/analysis      per-ply model insight (cached)
    GET  /api/bots                    ladder metadata
    GET  /api/stats                   public aggregates
    GET  /healthz                     liveness

No admin endpoints. Identity is a per-client httpOnly cookie token checked on
mutating routes; abuse control is in-process (global/per-IP active-game caps
plus token buckets keyed by CF-Connecting-IP or the peer address). Live games
are in-memory `GameSession`s; the bot plays via the `BotPool` worker
processes; every finished game lands in SQLite with its `.hxr` record. A
background sweeper finalizes idle games (race-safe: it takes the same per-game
lock as the move path and skips games with a search in flight).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hexo_engine import IllegalActionError

from .bots import BotPool, BotSpec, BotPoolError, BotPoolTimeout, load_bots_toml
from .config import Settings
from .db import ShowcaseDB, decode_payload, encode_payload
from .game import (
    TERMINATION_RESIGN,
    TERMINATION_SIX_IN_LINE,
    TERMINATION_TIMEOUT,
    GameSession,
    decode_hxr_actions,
    now_iso,
)

log = logging.getLogger("showcase")

_COOKIE = "showcase_token"
_NICK_STRIP = re.compile(r"[^A-Za-z0-9 _.\-]")
_NICK_MAX = 24


class CreateGameRequest(BaseModel):
    bot_id: str
    human_color: Literal[0, 1] = 0


class MoveRequest(BaseModel):
    q: int = Field(ge=-(1 << 15), le=(1 << 15) - 1)
    r: int = Field(ge=-(1 << 15), le=(1 << 15) - 1)


class NicknameRequest(BaseModel):
    nickname: str = Field(min_length=1, max_length=200)


def sanitize_nickname(raw: str) -> str | None:
    """Charset-allowlist + length-cap sanitizer; None when nothing survives."""
    cleaned = re.sub(r"\s+", " ", _NICK_STRIP.sub("", raw)).strip()
    cleaned = cleaned[:_NICK_MAX].strip()
    return cleaned or None


class TokenBucket:
    """Per-key in-process token bucket (steady rate + burst ceiling)."""

    def __init__(self, per_minute: float, burst: int | None = None) -> None:
        self.rate = per_minute / 60.0
        self.burst = float(burst if burst is not None else max(1, int(per_minute)))
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, stamp)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, stamp = self._state.get(key, (self.burst, now))
        tokens = min(self.burst, tokens + (now - stamp) * self.rate)
        if tokens < 1.0:
            self._state[key] = (tokens, now)
            return False
        self._state[key] = (tokens - 1.0, now)
        if len(self._state) > 10_000:  # bound memory under key churn
            cutoff = now - 3600.0
            self._state = {k: v for k, v in self._state.items() if v[1] >= cutoff}
        return True


def _client_key(request: Request) -> str:
    """Rate-limit identity: the Cloudflare-reported IP, else the peer address."""
    header = request.headers.get("CF-Connecting-IP")
    if header:
        return header.strip()
    return request.client.host if request.client else "unknown"


def create_app(settings: Settings) -> FastAPI:
    """Wire the app. All startup work (DB, ladder, worker pool, sweeper) runs
    in the lifespan so constructing the app object stays side-effect free."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.db = ShowcaseDB(settings.db_path)
        swept = app.state.db.abandon_stale_active(now_iso())
        if swept:
            log.info("swept %d stale active games from a previous run", swept)
        specs = load_bots_toml(settings.bots_toml)
        ladder: dict[str, tuple[BotSpec, int]] = {}
        for spec in specs:
            bot_db_id = app.state.db.upsert_bot(
                slug=spec.slug, label=spec.label, run=spec.run, epoch=spec.epoch,
                visits=spec.visits, weights_sha=spec.weights_sha,
                active_from=now_iso(),
            )
            ladder[spec.slug] = (spec, bot_db_id)
        app.state.ladder = ladder
        app.state.ladder_by_db_id = {db_id: spec for spec, db_id in ladder.values()}
        app.state.sessions = {}
        app.state.pool = BotPool(specs, settings)
        await app.state.pool.start()
        sweeper = asyncio.create_task(_sweeper(app))
        try:
            yield
        finally:
            sweeper.cancel()
            try:
                await sweeper
            except asyncio.CancelledError:
                pass
            await app.state.pool.stop()
            app.state.db.close()

    app = FastAPI(title="hexo-bot showcase", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.move_bucket = TokenBucket(settings.moves_per_minute)
    app.state.analysis_bucket = TokenBucket(settings.analysis_per_minute)
    app.state.game_bucket = TokenBucket(settings.games_per_hour / 60.0, burst=settings.games_per_hour)

    # -- helpers ---------------------------------------------------------------

    def _client_hash(key: str) -> str:
        return hashlib.sha256(f"{settings.ip_salt}:{key}".encode()).hexdigest()[:32]

    def _session_or_404(game_id: str) -> GameSession:
        session = app.state.sessions.get(game_id)
        if session is None:
            raise HTTPException(404, "unknown or expired game")
        return session

    def _authorize(session: GameSession, request: Request) -> None:
        if request.cookies.get(_COOKIE) != session.token:
            raise HTTPException(403, "not your game")

    def _finalize(session: GameSession, *, termination: str | None, winner: int | None) -> None:
        """Persist a finished game (record blob + games row) and reclaim the
        worker-side search tree. Callers hold `session.lock`."""
        record = session.finalize(termination=termination, winner=winner)
        app.state.db.finalize_game(
            game_id=session.game_id,
            finished_at=now_iso(),
            status=session.db_status,
            result=session.result,
            termination=session.termination,
            ply_count=len(session.actions),
            duration_s=round(session.duration_s(), 3),
            record=record,
        )
        app.state.pool.discard(game_key=session.game_key, bot_slug=session.bot_slug)

    async def _run_bot_turn(session: GameSession) -> None:
        """Background bot turn: search in the pool, apply under the game lock.

        The game may resign or time out while the search runs; the post-await
        liveness check makes the stale result a no-op. Any pool failure
        finalizes the game as abandoned rather than leaving it stuck."""
        try:
            out = await app.state.pool.bot_turn(
                game_key=session.game_key,
                bot_slug=session.bot_slug,
                actions=session.actions,
                seed=session.seed,
            )
        except Exception:
            log.exception("bot turn failed for game %s", session.game_id)
            async with session.lock:
                session.bot_busy = False
                if session.active:
                    _finalize(session, termination=None, winner=None)
            return
        async with session.lock:
            session.bot_busy = False
            if not session.active:
                return  # resigned/timed out mid-search; result discarded
            try:
                session.apply_bot_actions([move["action_id"] for move in out["actions"]])
            except IllegalActionError:
                log.exception("bot produced an illegal move in game %s", session.game_id)
                _finalize(session, termination=None, winner=None)
                return
            winner = session.engine_winner()
            if winner is not None:
                _finalize(session, termination=TERMINATION_SIX_IN_LINE, winner=winner)

    def _start_bot_turn(session: GameSession) -> None:
        session.bot_busy = True
        asyncio.get_running_loop().create_task(_run_bot_turn(session))

    async def _sweeper(app: FastAPI) -> None:
        """Idle-game finalizer + finished-session eviction."""
        while True:
            await asyncio.sleep(settings.sweep_interval_s)
            for session in list(app.state.sessions.values()):
                if session.active:
                    if session.bot_busy or session.idle_seconds() < settings.idle_timeout_s:
                        continue
                    async with session.lock:
                        if session.active and not session.bot_busy and (
                            session.idle_seconds() >= settings.idle_timeout_s
                        ):
                            _finalize(session, termination=TERMINATION_TIMEOUT, winner=None)
                elif session.idle_seconds() >= settings.finished_ttl_s:
                    app.state.sessions.pop(session.game_id, None)

    # -- game lifecycle ----------------------------------------------------------

    @app.post("/api/game")
    async def create_game(body: CreateGameRequest, request: Request, response: Response):
        key = _client_key(request)
        if not app.state.game_bucket.allow(key):
            raise HTTPException(429, "too many games created; slow down")
        entry = app.state.ladder.get(body.bot_id)
        if entry is None:
            raise HTTPException(404, f"unknown bot {body.bot_id!r}")
        spec, bot_db_id = entry
        sessions = app.state.sessions
        active = [s for s in sessions.values() if s.active]
        if len(active) >= settings.max_active_games:
            raise HTTPException(429, "server is full; try again in a minute")
        client_hash = _client_hash(key)
        if sum(1 for s in active if s.client_hash == client_hash) >= settings.max_games_per_ip:
            raise HTTPException(429, "active-game limit reached; finish or resign first")

        session = GameSession.create(
            bot_slug=spec.slug, bot_db_id=bot_db_id, bot_label=spec.label,
            bot_visits=spec.visits, human_color=body.human_color,
            client_hash=client_hash,
        )
        # One token per client: reuse the cookie so a client's games all
        # authenticate with the same value.
        existing = request.cookies.get(_COOKIE)
        if existing:
            session.token = existing
        sessions[session.game_id] = session
        app.state.db.create_game(
            game_id=session.game_id, bot_id=bot_db_id,
            human_color=session.human_color, started_at=session.started_at,
            client_hash=client_hash,
        )
        if session.bot_to_move:
            _start_bot_turn(session)
        response.set_cookie(
            _COOKIE, session.token, httponly=True, samesite="lax",
            max_age=7 * 24 * 3600,
        )
        return session.snapshot()

    @app.get("/api/game/{game_id}")
    async def get_game(game_id: str):
        return _session_or_404(game_id).snapshot()

    @app.post("/api/game/{game_id}/move")
    async def move(game_id: str, body: MoveRequest, request: Request):
        session = _session_or_404(game_id)
        _authorize(session, request)
        if not app.state.move_bucket.allow(_client_key(request)):
            raise HTTPException(429, "too many moves; slow down")
        async with session.lock:
            if not session.active:
                raise HTTPException(409, "game is finished")
            if session.bot_busy or session.to_move != session.human_color:
                raise HTTPException(409, "not your turn")
            try:
                session.apply_human_move(body.q, body.r)
            except IllegalActionError as exc:
                raise HTTPException(422, f"illegal move: {exc}") from exc
            winner = session.engine_winner()
            if winner is not None:
                _finalize(session, termination=TERMINATION_SIX_IN_LINE, winner=winner)
            elif session.bot_to_move:
                _start_bot_turn(session)
            return session.snapshot()

    @app.post("/api/game/{game_id}/resign")
    async def resign(game_id: str, request: Request):
        session = _session_or_404(game_id)
        _authorize(session, request)
        if not app.state.move_bucket.allow(_client_key(request)):
            raise HTTPException(429, "too many requests; slow down")
        async with session.lock:
            if not session.active:
                raise HTTPException(409, "game is finished")
            _finalize(
                session, termination=TERMINATION_RESIGN,
                winner=1 - session.human_color,
            )
            return session.snapshot()

    @app.post("/api/game/{game_id}/nickname")
    async def set_nickname(game_id: str, body: NicknameRequest, request: Request):
        session = _session_or_404(game_id)
        _authorize(session, request)
        if session.active:
            raise HTTPException(409, "nickname can be set after the game finishes")
        nickname = sanitize_nickname(body.nickname)
        if nickname is None:
            raise HTTPException(422, "nickname has no allowed characters (A-Za-z0-9 _.-)")
        session.nickname = nickname
        app.state.db.set_nickname(session.game_id, nickname)
        return {"nickname": nickname}

    # -- analysis -----------------------------------------------------------------

    @lru_cache(maxsize=256)
    def _record_actions(game_id: str) -> tuple[int, ...]:
        row = app.state.db.get_game(game_id)
        if row is None or row["record"] is None:
            raise HTTPException(404, "no record for this game")
        return tuple(decode_hxr_actions(row["record"]))

    @app.get("/api/game/{game_id}/analysis")
    async def analysis(
        game_id: str, request: Request,
        ply: int = Query(ge=0), search: bool = False,
    ):
        if not app.state.analysis_bucket.allow(_client_key(request)):
            raise HTTPException(429, "too many analysis requests; slow down")
        row = app.state.db.get_game(game_id)
        if row is None:
            raise HTTPException(404, "unknown game")
        if row["status"] == "active":
            raise HTTPException(409, "analysis is available after the game finishes")
        spec = app.state.ladder_by_db_id.get(row["bot_id"])
        if spec is None:
            raise HTTPException(409, "this game's bot is no longer in the ladder")
        actions = _record_actions(game_id)
        if ply > len(actions):
            raise HTTPException(422, f"ply {ply} out of range (game has {len(actions)} plies)")

        bot_db_id = int(row["bot_id"])
        cached_blob = app.state.db.analysis_get(game_id, ply, bot_db_id)
        payload: dict[str, Any] | None = decode_payload(cached_blob) if cached_blob else None
        cached = payload is not None
        if payload is None or (search and "search" not in payload):
            route_key = int(hashlib.sha256(game_id.encode()).hexdigest()[:12], 16)
            try:
                fresh = await app.state.pool.analyze(
                    route_key=route_key,
                    bot_slug=spec.slug,
                    actions=list(actions[:ply]),
                    want_search=search,
                    search_visits=settings.analysis_search_visit_cap,
                    seed=route_key * 5003 + ply,
                )
            except (BotPoolError, BotPoolTimeout) as exc:
                raise HTTPException(503, "analysis backend unavailable") from exc
            if payload is not None and "search" in fresh:
                payload["search"] = fresh["search"]
            else:
                payload = fresh
            app.state.db.analysis_put(game_id, ply, bot_db_id, encode_payload(payload))
            cached = False
        return {"game_id": game_id, "bot_id": spec.slug, "cached": cached, **payload}

    # -- metadata / stats -----------------------------------------------------------

    @app.get("/api/bots")
    async def bots():
        return [
            {
                "id": spec.slug, "label": spec.label, "visits": spec.visits,
                "run": spec.run, "epoch": spec.epoch,
            }
            for spec, _ in app.state.ladder.values()
        ]

    @app.get("/api/stats")
    async def stats():
        db: ShowcaseDB = app.state.db
        return {
            "bots": db.bot_stats(),
            "daily": db.daily(),
            "hall_of_fame": db.hall_of_fame(),
        }

    @app.get("/healthz")
    async def healthz():
        return {
            "ok": True,
            "bots": len(app.state.ladder),
            "active_games": sum(1 for s in app.state.sessions.values() if s.active),
        }

    if settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="web")

    return app


app = create_app(Settings.from_env())
