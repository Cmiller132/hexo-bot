"""FastAPI surface for the showcase.

Complete API (docs/showcase 01):

    POST /api/game                    create game (checkpoint_id x sims)
    GET  /api/game/{id}               state (owner-only while active; public once finished)
    POST /api/game/{id}/move          human move
    POST /api/game/{id}/resign
    POST /api/game/{id}/nickname      set nickname on the finished game
    GET  /api/game/{id}/analysis      per-ply model insight (cached; public, finished only)
    GET  /api/game/{id}/summary       per-ply value/stv/moves_left series (cached)
                                      (both take ?checkpoint_id= to analyze under
                                      any catalogue checkpoint; default is the
                                      game's own bot)
    GET  /api/games                   recent finished games feed (public, paginated)
    GET  /api/bots                    catalogue metadata (checkpoints + allowed sims)
    GET  /api/stats                   public aggregates
    POST /api/lab/eval                lab sandbox: net readout + requested
                                      internals (attention row for one query
                                      cell, per-block activation norms, input
                                      feature planes) for a visitor-built
                                      position (legal sequence or free-edit)
    POST /api/lab/search              lab sandbox: one capped real search on a
                                      legal-sequence position
    GET  /healthz                     liveness

No admin endpoints. Identity is a per-client httpOnly cookie token checked on
mutating routes and on reads of ACTIVE games; finished games are public by
default (shareable URLs, the feed) and their public reads ride the analysis
token bucket. Abuse control is in-process (global/per-IP active-game caps plus
token buckets keyed by CF-Connecting-IP or the peer address). Live games are
in-memory `GameSession`s; the bot plays via the `BotPool` worker processes;
every finished game lands in SQLite with its `.hxr` record. A background
sweeper finalizes idle games (race-safe: it takes the same per-game lock as
the move path and skips games with a search in flight).

Bots table mapping: bots.toml is a catalogue of CHECKPOINTS plus one global
allowed `sims` set; a DB bots row is one played (checkpoint, sims) combination
(identity (slug, weights_sha, visits)), created lazily on the first game with
that combination — the stats views keep per-strength granularity for free.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hexo_engine import IllegalActionError

from .bots import BotPool, CheckpointSpec, BotPoolError, BotPoolTimeout, load_bots_toml
from .config import Settings
from .lab_rules import (
    MAX_ACTIONS,
    MAX_COORD,
    MAX_FREE_STONES,
    LabPositionError,
    default_free_to_move,
    validate_actions,
    validate_free_stones,
)
from .db import ShowcaseDB, decode_payload, encode_payload
from .elo import compute_ratings
from .jsonsafe import sanitize_json
from .game import (
    TERMINATION_RESIGN,
    TERMINATION_SIX_IN_LINE,
    TERMINATION_TIMEOUT,
    GameSession,
    decode_hxr_actions,
    finished_snapshot,
    now_iso,
)

log = logging.getLogger("showcase")

_COOKIE = "showcase_token"
_NICK_STRIP = re.compile(r"[^A-Za-z0-9 _.\-]")
_NICK_MAX = 24

# Version stamp ("v") on cached analysis/summary payloads. Bump whenever the
# payload schema changes: cached entries with a different (or missing) stamp
# are treated as misses and recomputed. v2 added stv + moves_left; v3
# scrubs non-finite floats to null (and thereby retires rows poisoned with
# bare NaN literals, which 500 on every read).
_ANALYSIS_VERSION = 3

# analysis_cache "ply" slot for the whole-game summary payload (real plies are
# always >= 0, so -1 can never collide).
_SUMMARY_PLY = -1

# analysis_cache rows key on a bots-table id. Default analysis (no selector)
# keys under the game's own bot row, so pre-selector cache entries stay valid.
# Analyzing under a DIFFERENT catalogue checkpoint keys under an analysis-only
# bots row for that checkpoint at this sentinel visit count: 0 is never a
# playable sims value, and the stats views only surface bots with games, so
# these rows stay invisible. A checkpoint refresh changes the weights sha and
# therefore the row, invalidating those cached entries automatically.
_ANALYSIS_BOT_VISITS = 0

_FEED_LIMIT_MAX = 50
_FEED_LIMIT_DEFAULT = 20


class CreateGameRequest(BaseModel):
    checkpoint_id: str
    sims: int
    # 0 = human moves first (blue), 1 = human moves second (red), "random" =
    # the server flips a coin; the resolved 0/1 is echoed as `human_color` in
    # the game-state payload. Plain 0/1 stays the wire format for back-compat.
    human_color: Literal[0, 1, "random"] = 0


class MoveRequest(BaseModel):
    q: int = Field(ge=-(1 << 15), le=(1 << 15) - 1)
    r: int = Field(ge=-(1 << 15), le=(1 << 15) - 1)


class NicknameRequest(BaseModel):
    nickname: str = Field(min_length=1, max_length=200)


class LabCell(BaseModel):
    q: int = Field(ge=-MAX_COORD, le=MAX_COORD)
    r: int = Field(ge=-MAX_COORD, le=MAX_COORD)


class LabStones(BaseModel):
    p0: list[LabCell] = Field(default_factory=list, max_length=MAX_FREE_STONES)
    p1: list[LabCell] = Field(default_factory=list, max_length=MAX_FREE_STONES)


class LabWants(BaseModel):
    """Which internals ride the eval response beyond the net readout."""

    # Attention rows (per block x head) for this one support cell as the query.
    attention_query: LabCell | None = None
    # Per-block per-cell activation norms over the whole trunk.
    activations: bool = False
    # The (15, N) input feature planes as the server featurizer built them.
    features: bool = False


class LabEvalRequest(BaseModel):
    checkpoint_id: str = Field(max_length=128)
    # Exactly one of `actions` (chronological legal placements) or `stones`
    # (free-edit position) must be present.
    actions: list[LabCell] | None = Field(default=None, max_length=MAX_ACTIONS)
    stones: LabStones | None = None
    # Free-edit only: side to move (defaults from the stone counts).
    to_move: Literal[0, 1] | None = None
    wants: LabWants = Field(default_factory=LabWants)


class LabSearchRequest(BaseModel):
    checkpoint_id: str = Field(max_length=128)
    actions: list[LabCell] = Field(max_length=MAX_ACTIONS)
    sims: int = Field(ge=1)


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
    """Wire the app. All startup work (DB, catalogue, worker pool, sweeper)
    runs in the lifespan so constructing the app object stays side-effect free."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.db = ShowcaseDB(settings.db_path)
        swept = app.state.db.abandon_stale_active(now_iso())
        if swept:
            log.info("swept %d stale active games from a previous run", swept)
        catalogue = load_bots_toml(settings.bots_toml)
        app.state.catalogue = {spec.slug: spec for spec in catalogue.checkpoints}
        app.state.sims_allowed = catalogue.sims
        # (checkpoint slug, sims) -> bots-table row id; rows are created lazily
        # by _bot_db_id on the first game with that combination.
        app.state.bot_ids = {}
        app.state.sessions = {}
        app.state.pool = BotPool(list(catalogue.checkpoints), settings)
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

    app = FastAPI(title="Shrimp — a Hexo bot", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.move_bucket = TokenBucket(settings.moves_per_minute)
    app.state.analysis_bucket = TokenBucket(settings.analysis_per_minute)
    app.state.game_bucket = TokenBucket(settings.games_per_hour / 60.0, burst=settings.games_per_hour)
    # Lab limiters: eval is one cheap forward (generous); search shares the
    # worker pool with live game moves (tight). Both per client IP.
    app.state.lab_eval_bucket = TokenBucket(settings.lab_eval_per_minute)
    app.state.lab_search_bucket = TokenBucket(settings.lab_search_per_minute)

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

    def _bot_db_id(spec: CheckpointSpec, sims: int) -> int:
        """The bots-table row for a (checkpoint, sims) combination, upserted
        lazily on first use and memoized."""
        key = (spec.slug, sims)
        db_id = app.state.bot_ids.get(key)
        if db_id is None:
            db_id = app.state.db.upsert_bot(
                slug=spec.slug, label=spec.label, run=spec.run, epoch=spec.epoch,
                visits=sims, weights_sha=spec.weights_sha, active_from=now_iso(),
            )
            app.state.bot_ids[key] = db_id
        return db_id

    def _bot_row_and_spec(bot_id: int) -> tuple[dict[str, Any], CheckpointSpec]:
        """A finished game's bots row plus its catalogue checkpoint (needed to
        route worker jobs). 409s when the checkpoint left the catalogue."""
        bot_row = app.state.db.get_bot(int(bot_id))
        if bot_row is None:  # pragma: no cover - FK guarantees the row
            raise HTTPException(409, "this game's bot is no longer known")
        spec = app.state.catalogue.get(bot_row["slug"])
        if spec is None:
            raise HTTPException(409, "this game's checkpoint is no longer in the catalogue")
        return bot_row, spec

    def _analysis_target(
        row: dict[str, Any], checkpoint_id: str | None,
    ) -> tuple[CheckpointSpec, int]:
        """Resolve which catalogue checkpoint analyzes a finished game, plus
        the bots-table id its analysis_cache rows key under.

        No selector means the game's own checkpoint (existing behavior,
        including the 409 when it left the catalogue). An explicit selector
        may be ANY catalogue entry — a feed game against a retired checkpoint
        stays analyzable under a current one. Selecting the game's own
        checkpoint shares the default cache rows; any other selection keys
        under the `_ANALYSIS_BOT_VISITS` analysis-only bot row."""
        if checkpoint_id is None:
            _, spec = _bot_row_and_spec(row["bot_id"])
            return spec, int(row["bot_id"])
        spec = app.state.catalogue.get(checkpoint_id)
        if spec is None:
            raise HTTPException(404, f"unknown checkpoint {checkpoint_id!r}")
        bot_row = app.state.db.get_bot(int(row["bot_id"]))
        if bot_row is not None and bot_row["slug"] == spec.slug:
            return spec, int(row["bot_id"])
        return spec, _bot_db_id(spec, _ANALYSIS_BOT_VISITS)

    def _public_read_allowed(request: Request) -> None:
        """Token-bucket gate for public (cookieless) reads of finished games."""
        if not app.state.analysis_bucket.allow(_client_key(request)):
            raise HTTPException(429, "too many requests; slow down")

    def _versioned_cache_get(game_id: str, ply: int, bot_db_id: int) -> dict[str, Any] | None:
        """analysis_cache read that treats entries from an older payload schema
        as misses (recomputed and overwritten by the caller)."""
        blob = app.state.db.analysis_get(game_id, ply, bot_db_id)
        if blob is None:
            return None
        payload = decode_payload(blob)
        return payload if payload.get("v") == _ANALYSIS_VERSION else None

    def _versioned_cache_put(game_id: str, ply: int, bot_db_id: int, payload: dict[str, Any]) -> None:
        payload["v"] = _ANALYSIS_VERSION
        app.state.db.analysis_put(game_id, ply, bot_db_id, encode_payload(payload))

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
                visits=session.sims,
            )
        except Exception:
            log.exception("bot turn failed for game %s", session.game_id)
            async with session.lock:
                session.bot_busy = False
                # Recoverable: the search raised before mutating the position
                # (no actions applied), so the game is intact. Hold it in
                # `bot_failed` — NOT abandoned — so the human can retry the
                # bot's move. touch() resets the idle clock so the sweeper gives
                # them the full idle window to click Retry.
                if session.active:
                    session.bot_failed = True
                    session.touch()
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
        session.bot_failed = False  # a fresh attempt clears any prior hiccup
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
        spec = app.state.catalogue.get(body.checkpoint_id)
        if spec is None:
            raise HTTPException(404, f"unknown checkpoint {body.checkpoint_id!r}")
        if body.sims not in app.state.sims_allowed:
            raise HTTPException(
                422,
                f"sims must be one of {list(app.state.sims_allowed)}",
            )
        sessions = app.state.sessions
        active = [s for s in sessions.values() if s.active]
        if len(active) >= settings.max_active_games:
            raise HTTPException(429, "server is full; try again in a minute")
        client_hash = _client_hash(key)
        if sum(1 for s in active if s.client_hash == client_hash) >= settings.max_games_per_ip:
            raise HTTPException(429, "active-game limit reached; finish or resign first")

        # "random" resolves server-side to a fair 0/1; color stays keyed to
        # player index (player0 = blue moves first). When the human resolves
        # to 1 the bot owns the opening: `session.bot_to_move` is true below
        # and the bot turn is enqueued right away.
        human_color: int = (
            os.urandom(1)[0] & 1 if body.human_color == "random" else body.human_color
        )
        bot_db_id = _bot_db_id(spec, body.sims)
        session = GameSession.create(
            bot_slug=spec.slug, bot_db_id=bot_db_id, bot_label=spec.label,
            bot_epoch=spec.epoch, sims=body.sims, human_color=human_color,
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
    async def get_game(game_id: str, request: Request):
        """Owner-only while the game is live (session cookie); public once
        finished — first from the in-memory session, then from the DB record
        (evicted sessions, restarts). Public reads are rate-limited."""
        session = app.state.sessions.get(game_id)
        if session is not None:
            if session.active:
                _authorize(session, request)
            elif request.cookies.get(_COOKIE) != session.token:
                _public_read_allowed(request)
            return session.snapshot()
        row = app.state.db.get_game(game_id)
        if row is None or row["status"] == "active" or row["record"] is None:
            # An active row without a session only exists mid-crash; treat as gone.
            raise HTTPException(404, "unknown or expired game")
        _public_read_allowed(request)
        bot_row = app.state.db.get_bot(int(row["bot_id"]))
        return finished_snapshot(
            game_id=game_id,
            actions=list(_record_actions(game_id)),
            bot=bot_row,
            human_color=int(row["human_color"]),
            result=row["result"],
            termination=row["termination"],
            nickname=row["nickname"],
            finished_at=row["finished_at"],
        )

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

    @app.post("/api/game/{game_id}/retry")
    async def retry_bot(game_id: str, request: Request):
        """Re-run a bot turn that hiccuped. The failed search never mutated the
        position (the game was held in `bot_failed`, not abandoned), so this
        simply re-enqueues the same turn. Idempotent while a retry is already
        in flight."""
        session = _session_or_404(game_id)
        _authorize(session, request)
        if not app.state.move_bucket.allow(_client_key(request)):
            raise HTTPException(429, "too many requests; slow down")
        async with session.lock:
            if not session.active:
                raise HTTPException(409, "game is finished")
            if session.bot_busy:
                return session.snapshot()  # a retry is already running
            if not session.bot_failed or not session.bot_to_move:
                raise HTTPException(409, "no failed bot turn to retry")
            _start_bot_turn(session)
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
        # Signing a finished game re-buckets it under the new nickname in the
        # ELO fold, but does not change finished_count (the leaderboard cache
        # key), so drop the cache to force a recompute on the next /api/stats.
        app.state.leaderboard_cache = None
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
        checkpoint_id: str | None = Query(default=None, max_length=128),
    ):
        """Per-position model insight. `checkpoint_id` selects which catalogue
        checkpoint's net (and as-trained search profile) analyzes the position;
        default is the game's own bot."""
        if not app.state.analysis_bucket.allow(_client_key(request)):
            raise HTTPException(429, "too many analysis requests; slow down")
        row = app.state.db.get_game(game_id)
        if row is None:
            raise HTTPException(404, "unknown game")
        if row["status"] == "active":
            raise HTTPException(409, "analysis is available after the game finishes")
        spec, bot_db_id = _analysis_target(row, checkpoint_id)
        actions = _record_actions(game_id)
        if ply > len(actions):
            raise HTTPException(422, f"ply {ply} out of range (game has {len(actions)} plies)")

        payload = _versioned_cache_get(game_id, ply, bot_db_id)
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
            # Web-boundary scrub: whatever the worker sent, the fresh response
            # and the cache row stay strictly JSON-encodable (NaN/Inf -> null).
            fresh = sanitize_json(fresh)
            if payload is not None and "search" in fresh:
                payload["search"] = fresh["search"]
            else:
                payload = fresh
            _versioned_cache_put(game_id, ply, bot_db_id, payload)
            cached = False
        return {"game_id": game_id, "checkpoint_id": spec.slug, "cached": cached, **payload}

    @app.get("/api/game/{game_id}/summary")
    async def summary(
        game_id: str, request: Request,
        checkpoint_id: str | None = Query(default=None, max_length=128),
    ):
        """Whole-game per-ply {value, stv, moves_left, to_move} series for the
        value/ply chart. Index i is the position AFTER ply i (arrays have
        ply_count + 1 entries; entry 0 is the empty board). Computed lazily on
        first request — one chunked batched forward over every position — and
        cached in analysis_cache under the ply = -1 slot. `checkpoint_id`
        selects which catalogue checkpoint's net computes the series; default
        is the game's own bot."""
        if not app.state.analysis_bucket.allow(_client_key(request)):
            raise HTTPException(429, "too many analysis requests; slow down")
        row = app.state.db.get_game(game_id)
        if row is None:
            raise HTTPException(404, "unknown game")
        if row["status"] == "active":
            raise HTTPException(409, "summary is available after the game finishes")
        spec, bot_db_id = _analysis_target(row, checkpoint_id)
        actions = _record_actions(game_id)

        payload = _versioned_cache_get(game_id, _SUMMARY_PLY, bot_db_id)
        cached = payload is not None
        if payload is None:
            route_key = int(hashlib.sha256(game_id.encode()).hexdigest()[:12], 16)
            try:
                payload = await app.state.pool.summary(
                    route_key=route_key, bot_slug=spec.slug, actions=list(actions),
                )
            except (BotPoolError, BotPoolTimeout) as exc:
                raise HTTPException(503, "analysis backend unavailable") from exc
            payload = sanitize_json(payload)  # web-boundary scrub (see analysis)
            _versioned_cache_put(game_id, _SUMMARY_PLY, bot_db_id, payload)
        return {"game_id": game_id, "checkpoint_id": spec.slug, "cached": cached, **payload}

    # -- lab (live model-internals sandbox) --------------------------------------------

    def _lab_gate(request: Request, bucket: TokenBucket) -> None:
        """Enable flag + per-IP limiter for the lab endpoints."""
        if not settings.lab_enabled:
            raise HTTPException(404, "the lab is not enabled on this server")
        if not bucket.allow(_client_key(request)):
            raise HTTPException(429, "too many lab requests; slow down")

    def _lab_checkpoint(checkpoint_id: str) -> CheckpointSpec:
        spec = app.state.catalogue.get(checkpoint_id)
        if spec is None:
            raise HTTPException(404, f"unknown checkpoint {checkpoint_id!r}")
        return spec

    def _lab_route_key() -> int:
        # Lab jobs have no tree to keep sticky (eval never searches; lab
        # search trees are throwaway) — a random key spreads them across
        # workers so they interleave with, rather than queue behind, one
        # worker's game moves.
        return int.from_bytes(os.urandom(4), "big")

    @app.post("/api/lab/eval")
    async def lab_eval(body: LabEvalRequest, request: Request):
        """Net readout + requested internals for a visitor-built position.

        Body carries exactly one of `actions` (a legal placement sequence,
        replayed and validated here) or `stones` (a free-edit position; the
        worker synthesizes history and zeroes the history-derived features —
        the response names them in `zeroed_features`)."""
        _lab_gate(request, app.state.lab_eval_bucket)
        spec = _lab_checkpoint(body.checkpoint_id)
        if (body.actions is None) == (body.stones is None):
            raise HTTPException(422, "provide exactly one of 'actions' or 'stones'")
        actions = stones = to_move = None
        try:
            if body.actions is not None:
                actions = validate_actions([(c.q, c.r) for c in body.actions])
            else:
                stones = validate_free_stones(
                    [(c.q, c.r) for c in body.stones.p0],
                    [(c.q, c.r) for c in body.stones.p1],
                )
                to_move = (
                    body.to_move
                    if body.to_move is not None
                    else default_free_to_move(len(stones[0]), len(stones[1]))
                )
        except LabPositionError as exc:
            raise HTTPException(422, str(exc)) from exc
        wants = body.wants
        try:
            payload = await app.state.pool.lab_eval(
                route_key=_lab_route_key(),
                bot_slug=spec.slug,
                actions=actions,
                stones=stones,
                to_move=to_move,
                attention_cell=(
                    (wants.attention_query.q, wants.attention_query.r)
                    if wants.attention_query is not None
                    else None
                ),
                want_activations=wants.activations,
                want_features=wants.features,
            )
        except (BotPoolError, BotPoolTimeout) as exc:
            raise HTTPException(503, "lab backend unavailable") from exc
        if "reject" in payload:
            raise HTTPException(422, payload["reject"])
        return {"checkpoint_id": spec.slug, **payload}

    @app.post("/api/lab/search")
    async def lab_search(body: LabSearchRequest, request: Request):
        """One real search on a legal-sequence position, visit budget capped
        by `SHOWCASE_LAB_SEARCH_VISIT_CAP`. Free-edit positions cannot be
        searched: the engine cannot represent an unreachable position, so the
        endpoint takes `actions` only."""
        _lab_gate(request, app.state.lab_search_bucket)
        spec = _lab_checkpoint(body.checkpoint_id)
        if body.sims > settings.lab_search_visit_cap:
            raise HTTPException(
                422, f"sims must be <= {settings.lab_search_visit_cap}"
            )
        try:
            actions = validate_actions([(c.q, c.r) for c in body.actions])
        except LabPositionError as exc:
            raise HTTPException(422, str(exc)) from exc
        route_key = _lab_route_key()
        try:
            payload = await app.state.pool.lab_search(
                route_key=route_key,
                bot_slug=spec.slug,
                actions=actions,
                visits=body.sims,
                seed=route_key * 5003 + len(actions),
            )
        except (BotPoolError, BotPoolTimeout) as exc:
            raise HTTPException(503, "lab backend unavailable") from exc
        return {"checkpoint_id": spec.slug, "sims": body.sims, **payload}

    # -- public feed / metadata / stats -----------------------------------------------

    def _feed_item(row: dict[str, Any]) -> dict[str, Any]:
        """One /api/games list item. `winner` is null on a draw, else the color
        that won. `duration_s` is present only on the filtered path (the cursor
        feed row omits it) and defaults to None there for a stable shape."""
        winner = None
        if row["result"]:
            winner = (
                row["human_color"] if row["result"] == 1 else 1 - row["human_color"]
            )
        return {
            "id": row["id"],
            "bot": {
                "checkpoint_id": row["bot_slug"],
                "label": row["bot_label"],
                "epoch": row["bot_epoch"],
                "sims": row["bot_visits"],
            },
            "human_color": row["human_color"],
            "result": {
                "winner": winner,
                "termination": row["termination"],
                "human_result": row["result"],
            },
            "ply_count": row["ply_count"],
            "duration_s": row["duration_s"] if "duration_s" in row.keys() else None,
            "finished_at": row["finished_at"],
            "nickname": row["nickname"],
        }

    @app.get("/api/games")
    async def games_feed(
        request: Request,
        limit: int = Query(default=_FEED_LIMIT_DEFAULT, ge=1, le=_FEED_LIMIT_MAX),
        before: str | None = Query(default=None, max_length=128),
        nickname: str | None = Query(default=None, max_length=64),
        checkpoint_id: str | None = Query(default=None, max_length=128),
        sims: int | None = Query(default=None, ge=1),
        result: Literal["win", "loss", "draw"] | None = Query(default=None),
        sort: Literal[
            "recent", "oldest", "longest", "shortest", "slowest", "fastest"
        ] = Query(default="recent"),
        offset: int = Query(default=0, ge=0),
    ):
        """Public finished-games feed.

        Two shapes share the route. With NO filter/sort/offset param (the
        Analysis feed's calls — it may still pass `limit`/`before`), it stays the
        exact keyset-cursor response {"games": [...], "next": cursor} — `before`
        pages it, `next` is null on a short page. As soon as any of
        nickname/checkpoint_id/sims/result is set, sort != 'recent', offset > 0,
        or the client explicitly sends a sort/offset param, it switches to
        server-side LIMIT/OFFSET filtering and returns {"games", "total",
        "offset", "limit", "next": null} instead."""
        if not app.state.analysis_bucket.allow(_client_key(request)):
            raise HTTPException(429, "too many requests; slow down")

        # A paginating client (stats.js) always sends an explicit sort+offset, so
        # their presence in the query string also triggers the filtered
        # {games,total,offset,limit,duration_s} shape — even at the default
        # sort=='recent' and offset==0. The Analysis feed sends neither (it may
        # send `limit`+`before`, which stay on the keyset-cursor path), so it
        # keeps the cursor response. `limit` is deliberately NOT a trigger: the
        # cursor pagination uses it too.
        qp = request.query_params
        filtered = (
            nickname is not None
            or checkpoint_id is not None
            or sims is not None
            or result is not None
            or sort != "recent"
            or offset > 0
            or "sort" in qp
            or "offset" in qp
        )
        if filtered:
            rows, total = app.state.db.list_games_filtered(
                nickname=nickname, checkpoint_id=checkpoint_id, sims=sims,
                result=result, sort=sort, limit=limit, offset=offset,
            )
            return {
                "games": [_feed_item(row) for row in rows],
                "total": total,
                "offset": offset,
                "limit": limit,
                "next": None,
            }

        before_finished_at = before_id = None
        if before is not None:
            before_finished_at, sep, before_id = before.partition("~")
            if not sep or not before_finished_at:
                raise HTTPException(422, "malformed 'before' cursor")
        rows = app.state.db.list_finished(
            limit=limit, before_finished_at=before_finished_at, before_id=before_id,
        )
        items = [_feed_item(row) for row in rows]
        next_cursor = None
        if len(items) == limit and items:
            last = rows[-1]
            next_cursor = f"{last['finished_at']}~{last['id']}"
        return {"games": items, "next": next_cursor}

    @app.get("/api/bots")
    async def bots():
        return {
            "checkpoints": [
                {
                    "id": spec.slug, "label": spec.label,
                    "run": spec.run, "epoch": spec.epoch,
                    **spec.meta,
                }
                for spec in app.state.catalogue.values()
            ],
            "sims": list(app.state.sims_allowed),
        }

    def _leaderboard_payload(db: ShowcaseDB) -> dict[str, Any]:
        """ELO leaderboard + totals, memoized on the finished-game count.

        The count is monotone and only finished games feed the fold, so it is a
        sound self-invalidating generation key: unchanged count -> reuse the
        cached payload, changed count -> one recompute. Stored on app.state as
        {"gen": int, "payload": dict}."""
        gen = db.finished_count()
        cached = getattr(app.state, "leaderboard_cache", None)
        if cached is not None and cached["gen"] == gen:
            return cached["payload"]
        games_rows = db.finished_games_for_elo()
        bots_index = db.bots_index()
        ratings = compute_ratings(games_rows, bots_index)
        draws = sum(1 for r in games_rows if r["result"] == 0)
        payload = {
            "totals": {
                "games": gen,
                "players": len(ratings["players"]),
                "bots": len(ratings["bots"]),
                "draws": draws,
            },
            "leaderboard": ratings,
        }
        app.state.leaderboard_cache = {"gen": gen, "payload": payload}
        return payload

    @app.get("/api/stats")
    async def stats():
        db: ShowcaseDB = app.state.db
        return {
            "bots": db.bot_stats(),
            "daily": db.daily(),
            "hall_of_fame": db.hall_of_fame(),
            **_leaderboard_payload(db),
        }

    @app.get("/healthz")
    async def healthz():
        return {
            "ok": True,
            "checkpoints": len(app.state.catalogue),
            "active_games": sum(1 for s in app.state.sessions.values() if s.active),
        }

    if settings.static_dir.is_dir():
        app.mount("/", _RevalidatingStaticFiles(directory=settings.static_dir, html=True), name="web")

    return app


class _RevalidatingStaticFiles(StaticFiles):
    """StaticFiles that forces cache revalidation on every asset.

    Without a Cache-Control header, browsers and the CDN cache index.html and
    app.js on independent heuristic schedules, so successive deploys can serve
    a mixed pair (fresh markup with stale script or the reverse) and controls
    silently stop working. `no-cache` keeps caching but requires an ETag
    revalidation round-trip, so every asset is always from the same deploy.
    The 304 path keeps repeat loads cheap.
    """

    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


app = create_app(Settings.from_env())
