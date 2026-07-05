"""Bot catalogue: `bots.toml` config, per-worker checkpoint residency, and the
process pool that executes search and analysis jobs.

`bots.toml` defines a CATALOGUE of checkpoints plus one global set of allowed
search budgets (`sims`); a playable bot is any (checkpoint, sims) combination,
chosen per game at `POST /api/game` time. Workers hold one loaded net + one
search session per CHECKPOINT — the visit budget is a per-job parameter, so
the catalogue stays small in memory no matter how many strengths are offered.

Process model: `SHOWCASE_WORKERS` spawned processes each load the FULL
catalogue once (checkpoints are small) plus one `HexfieldMctsSession` per
checkpoint, and serve jobs from a per-worker queue. Jobs for a game are routed
sticky by `game_key % workers`, so a game's search trees live in exactly one
worker and `discard(game_key)` at game end reclaims them there. Results flow
back over a shared queue drained by a reader thread that resolves asyncio
futures; an abandoned or timed-out job simply resolves a future nobody awaits,
so the pool can never deadlock on a dead game.

This module keeps torch/hexfield imports out of module scope: the web process
imports it for `BotSpec`/`load_bots_toml`/`BotPool` without paying (or
depending on) the model stack. Workers do the heavy imports in
`_WorkerRuntime` after multiprocessing spawn.

Search behavior mirrors the training run: the as-trained knobs (Gumbel root /
sequential-halving profile, widening, FPU, TSS) are parsed from the training
TOML's `[model.config.selfplay]` section (`SHOWCASE_SEARCH_CONFIG`,
default `configs/hexfield_main_7.toml`); only the visit budget varies per
game. Opening plies are temperature-sampled exactly like the eval arena, so
games do not all open identically.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import multiprocessing as mp
import os
import threading
import tomllib
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings

log = logging.getLogger("showcase.bots")

_READY = "__ready__"
_SEED_MASK = (1 << 63) - 1


class BotPoolError(RuntimeError):
    """A worker job raised; carries the worker-side traceback text."""


class BotPoolTimeout(TimeoutError):
    """A worker job did not finish inside the configured deadline."""


# ---------------------------------------------------------------------------
# Catalogue config (web-process side, no torch)
# ---------------------------------------------------------------------------

DEFAULT_SIMS = (16, 64, 256, 512)

# Keys of a [[checkpoint]] table the server itself consumes; anything else is
# passed through verbatim as display metadata (e.g. games_trained).
_CHECKPOINT_REQUIRED = {"id", "checkpoint", "label", "run", "epoch"}


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    """One catalogue entry from bots.toml (a checkpoint, not a strength)."""

    slug: str
    label: str
    run: str
    epoch: int
    checkpoint: Path
    weights_sha: str
    meta: dict  # optional display metadata (scalars only), served verbatim


@dataclass(frozen=True, slots=True)
class Catalogue:
    """The parsed bots.toml: checkpoints x one global allowed-sims set."""

    checkpoints: tuple[CheckpointSpec, ...]
    sims: tuple[int, ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bots_toml(path: Path | str) -> Catalogue:
    """Parse and validate bots.toml.

    Schema: repeated `[[checkpoint]]` tables with `id`, `checkpoint` (relative
    paths resolve against the toml's directory), `label`, `run`, `epoch`; any
    extra scalar keys become display metadata. One optional top-level
    `sims = [...]` array is the global allowed search-budget set (default
    `DEFAULT_SIMS`). Checkpoints must exist (the sha is part of the bots-table
    identity key).
    """
    path = Path(path)
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    entries = raw.get("checkpoint", [])
    if not entries:
        raise ValueError(f"no [[checkpoint]] entries in {path}")
    sims_raw = raw.get("sims", list(DEFAULT_SIMS))
    if not isinstance(sims_raw, list) or not sims_raw:
        raise ValueError(f"top-level 'sims' in {path} must be a non-empty array")
    sims = tuple(sorted({int(s) for s in sims_raw}))
    if sims[0] < 1:
        raise ValueError(f"'sims' entries in {path} must be >= 1")
    specs: list[CheckpointSpec] = []
    seen: set[str] = set()
    for entry in entries:
        missing = _CHECKPOINT_REQUIRED - set(entry)
        if missing:
            raise ValueError(
                f"checkpoint entry {entry.get('id', '?')!r} missing keys: {sorted(missing)}"
            )
        slug = str(entry["id"])
        if slug in seen:
            raise ValueError(f"duplicate checkpoint id {slug!r} in {path}")
        seen.add(slug)
        checkpoint = Path(entry["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = (path.parent / checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint {slug!r}: file not found: {checkpoint}")
        meta = {key: value for key, value in entry.items() if key not in _CHECKPOINT_REQUIRED}
        for key, value in meta.items():
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"checkpoint {slug!r}: metadata key {key!r} must be a scalar"
                )
        specs.append(
            CheckpointSpec(
                slug=slug,
                label=str(entry["label"]),
                run=str(entry["run"]),
                epoch=int(entry["epoch"]),
                checkpoint=checkpoint,
                weights_sha=_file_sha256(checkpoint),
                meta=meta,
            )
        )
    return Catalogue(checkpoints=tuple(specs), sims=sims)


# ---------------------------------------------------------------------------
# Worker side (heavy imports happen here, post-spawn)
# ---------------------------------------------------------------------------


class SearchProfile:
    """As-trained search invocation, parsed once per worker from the training TOML.

    Everything except the per-bot visit budget comes from the run config:
    c_puct, widening, FPU, TSS, and the divergence overrides (which carry the
    Gumbel root/sequential-halving levers main_7 trains with). The virtual
    batch size and opening sampling mirror the eval arena's single-root CPU
    settings rather than the GPU self-play pipeline depth.
    """

    def __init__(self, config_path: Path | str) -> None:
        from hexfield.config import build_divergence_overrides, parse_hexfield_config

        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        model_cfg = raw.get("model", {}).get("config", {})
        cfg = parse_hexfield_config(
            {
                "device": "cpu",
                "selfplay": model_cfg.get("selfplay", {}),
                "multi_stage_eval": model_cfg.get("multi_stage_eval", {}),
            }
        )
        self.selfplay = cfg.selfplay
        self.overrides = build_divergence_overrides(cfg.selfplay)
        self.virtual_batch_size = int(cfg.multi_stage_eval.eval_virtual_batch_size or 32)
        self.opening_plies = int(cfg.multi_stage_eval.opening_plies)
        self.opening_temperature = float(cfg.multi_stage_eval.opening_temperature)

    def move_temperature(self, ply: int) -> float:
        """Eval-arena selection protocol: sampled opening prefix, then greedy."""
        if ply < self.opening_plies and self.opening_temperature > 0.0:
            return self.opening_temperature
        return 0.0

    def search_one(
        self, session: Any, evaluator: Any, state: Any, *,
        game_key: int, visits: int, seed: int, temperature: float,
    ) -> dict:
        """One single-root search; returns the raw per-root result dict.

        Selection happens IN-SEARCH via `move_temperatures` (the scalar
        `temperature` argument must stay 0.0; per-root behavior rides the
        list), so callers play `result["action_id"]` directly.
        """
        sp = self.selfplay
        return session.search(
            [int(game_key)],
            (state,),
            visits=int(visits),
            c_puct=sp.c_puct,
            temperature=0.0,
            seed=int(seed) & _SEED_MASK,
            evaluator=evaluator,
            move_temperatures=[float(temperature)],
            divergence_overrides=self.overrides,
            virtual_batch_size=self.virtual_batch_size,
            active_root_limit=sp.active_root_limit,
            widening_policy_mass=sp.widening_policy_mass,
            widening_max_children=sp.widening_max_children,
            widening_min_children=sp.widening_min_children,
            fpu_reduction=sp.fpu_reduction,
            tss_enabled=sp.tss_enabled,
            search_parity_mode=sp.search_parity_mode,
        )[0]


def _load_checkpoint(path: Path) -> Any:
    """Strict-load a hexfield checkpoint into a fresh eval-mode HexfieldNet.

    Arch (width / head count / trunk layout) is inferred from the state dict
    where determinable, falling back to the env-driven module globals — the
    same contract as the eval arena's loader. A mismatch fails the strict load
    with a clear error rather than serving a half-random net.
    """
    import torch

    from hexfield.model import HexfieldNet, infer_net_kwargs_from_state_dict

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError(f"checkpoint payload has no 'model' state dict: {path}")
    state_dict = payload["model"]
    model = HexfieldNet(**infer_net_kwargs_from_state_dict(state_dict))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@dataclass
class _LoadedBot:
    spec: CheckpointSpec
    model: Any
    evaluator: Any
    session: Any


class _WorkerRuntime:
    """Everything one worker process holds resident: the catalogue, one search
    session per checkpoint, and the shared search profile.

    Device: `settings.device` (SHOWCASE_DEVICE) is resolved once per worker at
    init (see showcase.device). When it resolves to an accelerator, a startup
    CPU-vs-device parity self-check runs on the FIRST checkpoint's net — the
    device either serves the same numbers as CPU within tolerance or the whole
    worker falls back to cpu (never serve wrong moves). The check is per
    device, not per checkpoint: all catalogue nets share the arch code paths,
    so one parity pass vouches for the backend. Only the model/evaluator move;
    the Rust search session is device-agnostic (it calls back into the
    evaluator for every batch).
    """

    def __init__(self, specs: list[CheckpointSpec], settings: Settings) -> None:
        import torch

        threads = settings.torch_threads or max(
            1, (os.cpu_count() or 2) // max(1, settings.workers)
        )
        torch.set_num_threads(threads)

        from hexfield import _rust
        from hexfield.inference import HexfieldEvaluator

        from . import device as devmod

        resolved = devmod.resolve_device(settings.device)
        device = resolved.device
        for note in resolved.notes:
            log.warning("%s", note)

        self.policy_floor = settings.policy_floor
        self.profile = SearchProfile(settings.search_config)
        self.bots: dict[str, _LoadedBot] = {}
        checked = False
        for spec in specs:
            model = _load_checkpoint(spec.checkpoint)
            if device != "cpu" and not checked:
                checked = True
                if devmod.selfcheck_wanted(settings.device_selfcheck, device):
                    check = devmod.verify_device(model, device)
                    if check.ok:
                        log.info(
                            "device self-check passed on %s: max |dvalue|=%.2e "
                            "|dpolicy|=%.2e (tol %.0e)",
                            device, check.value_diff, check.policy_diff,
                            devmod.SELFCHECK_TOL,
                        )
                    else:
                        log.warning(
                            "DEVICE SELF-CHECK FAILED on %s (|dvalue|=%.3g "
                            "|dpolicy|=%.3g%s) — FALLING BACK TO CPU so no "
                            "wrong moves are served; investigate the %s stack",
                            device, check.value_diff, check.policy_diff,
                            f", error: {check.error}" if check.error else "",
                            device,
                        )
                        device = "cpu"
                if device != "cpu":
                    # Trigger lazy backend init / kernel JIT off the hot path.
                    model.to(device)
                    devmod.warmup(model, device)
            self.bots[spec.slug] = _LoadedBot(
                spec=spec,
                model=model,
                evaluator=HexfieldEvaluator(model, device=device),
                session=_rust.HexfieldMctsSession(max_states=65_536),
            )
        self.device = device
        log.info(
            "showcase worker ready: device=%s (requested %r), %d checkpoint(s), "
            "torch_threads=%d",
            device, settings.device, len(self.bots), threads,
        )

    @staticmethod
    def _replay(actions: list[int]) -> Any:
        import hexo_engine as engine
        from hexo_engine.types import PlacementAction, unpack_coord_id

        state = engine.new_game()
        for aid in actions:
            engine.apply_action(state, PlacementAction(unpack_coord_id(int(aid))))
        return state

    def bot_turn(
        self, *, bot_slug: str, game_key: int, actions: list[int], seed: int, visits: int,
    ) -> dict:
        """Play the bot's whole turn (1-2 stones) from the given move history
        at the given visit budget.

        Replays the history into a fresh engine state, then searches and
        applies placements until the turn passes or the game ends. Returns the
        packed action ids played plus per-move diagnostics.
        """
        import hexo_engine as engine
        from hexo_engine.types import PlacementAction, unpack_coord_id
        from hexfield.geometry import unpack_action_id

        bot = self.bots[bot_slug]
        state = self._replay(actions)
        entry_player = engine.current_player(state)
        played: list[dict] = []
        ply = len(actions)
        while engine.terminal(state) is None and engine.current_player(state) == entry_player:
            result = self.profile.search_one(
                bot.session, bot.evaluator, state,
                game_key=game_key,
                visits=int(visits),
                seed=seed * 5003 + ply,
                temperature=self.profile.move_temperature(ply),
            )
            action_id = int(result["action_id"])
            q, r = unpack_action_id(action_id)
            engine.apply_action(state, PlacementAction(unpack_coord_id(action_id)))
            played.append(
                {
                    "action_id": action_id,
                    "q": q,
                    "r": r,
                    "root_value": round(float(result["root_value"]), 6),
                    "visits": int(result["visits"]),
                }
            )
            ply += 1
        return {"actions": played}

    def analyze(
        self, *, bot_slug: str, actions: list[int], want_search: bool,
        search_visits: int, seed: int,
    ) -> dict:
        """Net-only readout (plus optional small searched eval) for the
        position after `actions`."""
        import hexo_engine as engine

        from . import analysis

        bot = self.bots[bot_slug]
        state = self._replay(actions)
        payload = analysis.net_eval(bot.model, state, policy_floor=self.policy_floor)
        payload["ply"] = len(actions)
        terminal = engine.terminal(state)
        payload["to_move"] = (
            None if terminal is not None
            else (1 if str(engine.current_player(state).value).endswith("1") else 0)
        )
        if want_search and terminal is None:
            payload["search"] = analysis.searched_eval(
                bot.session, bot.evaluator, self.profile, state,
                # Throwaway tree key: high bit set so it can never collide with
                # a live game's 48-bit key.
                game_key=(1 << 63) | (seed & ((1 << 48) - 1)),
                visits=search_visits,
                seed=seed,
            )
        return payload

    def summary(self, *, bot_slug: str, actions: list[int]) -> dict:
        """Per-ply {value, stv, moves_left} series for a finished game: one
        chunked batched forward over the positions after ply 0..N (see
        `analysis.summary_eval`). CPU cost is one forward per position — a
        full game is a few search-batch equivalents, fine at showcase volume."""
        import hexo_engine as engine
        from hexo_engine.types import PlacementAction, unpack_coord_id

        from . import analysis

        bot = self.bots[bot_slug]
        state = engine.new_game()
        rows: list[Any] = []
        to_move: list[int | None] = []
        for index in range(len(actions) + 1):
            rows.append(analysis.featurize(state))
            terminal = engine.terminal(state)
            to_move.append(
                None if terminal is not None
                else (1 if str(engine.current_player(state).value).endswith("1") else 0)
            )
            if index < len(actions):
                engine.apply_action(state, PlacementAction(unpack_coord_id(int(actions[index]))))
        payload = analysis.summary_eval(bot.model, rows)
        payload["to_move"] = to_move
        payload["ply_count"] = len(actions)
        return payload

    def discard(self, *, bot_slug: str, game_key: int) -> None:
        """Drop a finished game's search tree (no-op if never searched here)."""
        bot = self.bots.get(bot_slug)
        if bot is not None:
            bot.session.discard(int(game_key))


def _worker_main(
    worker_index: int, specs: list[CheckpointSpec], settings: Settings,
    job_queue: Any, result_queue: Any,
) -> None:
    """Worker process entry point: load the catalogue, then serve jobs forever.

    Jobs are `(job_id, kind, kwargs)`; replies are `(job_id, payload)` where
    payload is `{"ok": ...}` or `{"error": traceback}`. Discard jobs carry
    `job_id=None` and get no reply. `None` on the job queue shuts down.
    """
    # Fresh spawned process: give it a stderr log handler so the one-time
    # device-resolution/self-check lines land in `docker compose logs`.
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s worker-{worker_index} %(name)s %(levelname)s %(message)s",
    )
    try:
        runtime = _WorkerRuntime(specs, settings)
    except Exception:
        result_queue.put((_READY, worker_index, traceback.format_exc()))
        return
    result_queue.put((_READY, worker_index, None))
    while True:
        job = job_queue.get()
        if job is None:
            return
        job_id, kind, kwargs = job
        try:
            if kind == "move":
                out = runtime.bot_turn(**kwargs)
            elif kind == "analyze":
                out = runtime.analyze(**kwargs)
            elif kind == "summary":
                out = runtime.summary(**kwargs)
            elif kind == "discard":
                runtime.discard(**kwargs)
                continue
            else:
                raise ValueError(f"unknown job kind {kind!r}")
        except Exception:
            result_queue.put((job_id, {"error": traceback.format_exc()}))
        else:
            result_queue.put((job_id, {"ok": out}))


# ---------------------------------------------------------------------------
# Web-process side: the pool
# ---------------------------------------------------------------------------


class BotPool:
    """Spawned worker pool with sticky per-game routing and asyncio results.

    `start()` blocks until every worker has the catalogue loaded (a checkpoint
    that fails to load surfaces at startup, not on the first move). Job
    results resolve futures on the event loop via a reader thread; a job whose
    caller has gone away (client abandoned the game, `wait_for` timed out)
    resolves nothing and is dropped.
    """

    def __init__(self, specs: list[CheckpointSpec], settings: Settings) -> None:
        if settings.workers < 1:
            raise ValueError("SHOWCASE_WORKERS must be >= 1")
        self._specs = specs
        self._settings = settings
        self._ctx = mp.get_context("spawn")
        self._job_queues: list[Any] = []
        self._procs: list[Any] = []
        self._result_queue = self._ctx.Queue()
        self._futures: dict[int, asyncio.Future] = {}
        self._job_ids = itertools.count(1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader: threading.Thread | None = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        for index in range(self._settings.workers):
            queue = self._ctx.Queue()
            proc = self._ctx.Process(
                target=_worker_main,
                args=(index, self._specs, self._settings, queue, self._result_queue),
                daemon=True,
                name=f"showcase-bot-{index}",
            )
            proc.start()
            self._job_queues.append(queue)
            self._procs.append(proc)
        for _ in self._procs:
            tag, index, error = await self._loop.run_in_executor(
                None, self._result_queue.get, True, 300.0
            )
            if tag != _READY:  # pragma: no cover - protocol guard
                raise RuntimeError(f"unexpected pre-ready pool message: {tag!r}")
            if error is not None:
                raise RuntimeError(f"bot worker {index} failed to start:\n{error}")
        self._reader = threading.Thread(
            target=self._reader_loop, name="showcase-pool-reader", daemon=True
        )
        self._reader.start()

    async def stop(self) -> None:
        for queue in self._job_queues:
            queue.put(None)
        self._result_queue.put(None)
        for proc in self._procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        if self._reader is not None:
            self._reader.join(timeout=5)
        for fut in self._futures.values():
            if not fut.done():
                fut.cancel()
        self._futures.clear()

    def _reader_loop(self) -> None:
        while True:
            item = self._result_queue.get()
            if item is None:
                return
            job_id, payload = item
            future = self._futures.pop(job_id, None)
            if future is not None:
                self._loop.call_soon_threadsafe(self._resolve, future, payload)

    @staticmethod
    def _resolve(future: asyncio.Future, payload: dict) -> None:
        if not future.done():
            future.set_result(payload)

    def _route(self, key: int) -> int:
        return int(key) % len(self._job_queues)

    async def _submit(self, worker: int, kind: str, kwargs: dict, timeout: float) -> Any:
        assert self._loop is not None, "BotPool.start() was not awaited"
        job_id = next(self._job_ids)
        future: asyncio.Future = self._loop.create_future()
        self._futures[job_id] = future
        self._job_queues[worker].put((job_id, kind, kwargs))
        try:
            payload = await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            self._futures.pop(job_id, None)
            raise BotPoolTimeout(f"{kind} job exceeded {timeout:.0f}s") from None
        except asyncio.CancelledError:
            self._futures.pop(job_id, None)
            raise
        if "error" in payload:
            raise BotPoolError(payload["error"])
        return payload["ok"]

    # -- public jobs ------------------------------------------------------------

    async def bot_turn(
        self, *, game_key: int, bot_slug: str, actions: list[int], seed: int, visits: int,
    ) -> dict:
        return await self._submit(
            self._route(game_key), "move",
            {
                "bot_slug": bot_slug, "game_key": game_key, "actions": list(actions),
                "seed": seed, "visits": int(visits),
            },
            self._settings.bot_timeout_s,
        )

    async def analyze(
        self, *, route_key: int, bot_slug: str, actions: list[int],
        want_search: bool, search_visits: int, seed: int,
    ) -> dict:
        return await self._submit(
            self._route(route_key), "analyze",
            {
                "bot_slug": bot_slug, "actions": list(actions),
                "want_search": want_search, "search_visits": search_visits,
                "seed": seed,
            },
            self._settings.bot_timeout_s,
        )

    async def summary(self, *, route_key: int, bot_slug: str, actions: list[int]) -> dict:
        return await self._submit(
            self._route(route_key), "summary",
            {"bot_slug": bot_slug, "actions": list(actions)},
            self._settings.bot_timeout_s,
        )

    def discard(self, *, game_key: int, bot_slug: str) -> None:
        """Fire-and-forget tree reclamation on the game's sticky worker."""
        self._job_queues[self._route(game_key)].put(
            (None, "discard", {"bot_slug": bot_slug, "game_key": game_key})
        )
