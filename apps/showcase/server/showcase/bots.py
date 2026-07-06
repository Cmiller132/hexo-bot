"""Bot catalogue: `bots.toml` config, per-worker checkpoint residency, and the
process pool that executes search and analysis jobs.

`bots.toml` defines a CATALOGUE of checkpoints plus one global set of allowed
search budgets (`sims`); a playable bot is any (checkpoint, sims) combination,
chosen per game at `POST /api/game` time. Workers hold one loaded net + one
search session per CHECKPOINT — the visit budget is a per-job parameter, so
the catalogue stays small in memory no matter how many strengths are offered.

Process model: `SHOWCASE_WORKERS` spawned processes each load the FULL
catalogue once (checkpoints are small) plus one `ShrimpMctsSession` per
checkpoint, and serve jobs from a per-worker queue. Jobs for a game are routed
sticky by `game_key % workers`, so a game's search trees live in exactly one
worker and `discard(game_key)` at game end reclaims them there. Results flow
back over a shared queue drained by a reader thread that resolves asyncio
futures; an abandoned or timed-out job simply resolves a future nobody awaits,
so the pool can never deadlock on a dead game.

This module keeps torch/shrimp imports out of module scope: the web process
imports it for `BotSpec`/`load_bots_toml`/`BotPool` without paying (or
depending on) the model stack. Workers do the heavy imports in
`_WorkerRuntime` after multiprocessing spawn.

Search behavior mirrors each checkpoint's training run: the as-trained knobs
(Gumbel root / sequential-halving profile, widening, FPU, TSS) are parsed from
a profile TOML's `[model.config.selfplay]` section; only the visit budget
varies per game. A `[[checkpoint]]` entry may name its own profile via
`search_profile` (a bare name resolves against the built-in profiles dir,
`apps/showcase/profiles/`), so PUCT-era checkpoints are served with the search
they trained under; entries without one share the global default
(`SHOWCASE_SEARCH_CONFIG`, default `configs/shrimp_main_7.toml`). Opening
plies are temperature-sampled exactly like the eval arena, so games do not all
open identically.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import multiprocessing as mp
import os
import threading
import time
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


# Substrings in a worker-side traceback that mean the worker's inference device
# (SYCL/CUDA queue, etc.) is corrupted, not just that one position was bad. Such
# a worker keeps accepting jobs but every later search hangs, so we recycle the
# process even though the job "merely" raised. Matched case-insensitively.
_WEDGE_ERROR_MARKERS = (
    "index out of bounds",
    "device-side assert",
    "sycl",
    "xpu",
    "cuda error",
    "cudnn",
    "illegal memory access",
    "corrupt",
)


def _error_indicates_wedge(error: str) -> bool:
    low = error.lower()
    return any(marker in low for marker in _WEDGE_ERROR_MARKERS)


# ---------------------------------------------------------------------------
# Catalogue config (web-process side, no torch)
# ---------------------------------------------------------------------------

DEFAULT_SIMS = (16, 64, 256, 512)

# Built-in search profiles shipped alongside the server (bare `search_profile`
# names in bots.toml resolve here).
PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"

# Keys of a [[checkpoint]] table the server itself consumes; anything else is
# passed through verbatim as display metadata (e.g. games_trained, group).
_CHECKPOINT_REQUIRED = {"id", "checkpoint", "label", "run", "epoch"}
_CHECKPOINT_SERVER_KEYS = _CHECKPOINT_REQUIRED | {"search_profile"}


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
    # Resolved profile toml this checkpoint searches with; None selects the
    # global settings.search_config default.
    search_profile: Path | None = None


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


def _resolve_search_profile(ref: str, bots_dir: Path) -> Path:
    """Resolve a checkpoint's `search_profile` reference to a profile toml.

    Resolution order: a bare name (a single path component with no .toml
    suffix, e.g. "shrimp_main_5") resolves against the built-in
    `PROFILES_DIR`; otherwise the reference is a path, taken relative to the
    bots.toml directory unless absolute. The file must exist so a bad
    reference fails at catalogue load, not on the first move.
    """
    candidate = Path(ref)
    if len(candidate.parts) == 1 and candidate.suffix != ".toml":
        candidate = PROFILES_DIR / f"{ref}.toml"
    elif not candidate.is_absolute():
        candidate = (bots_dir / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"search_profile {ref!r}: profile not found: {candidate}"
        )
    return candidate


def load_bots_toml(path: Path | str) -> Catalogue:
    """Parse and validate bots.toml.

    Schema: repeated `[[checkpoint]]` tables with `id`, `checkpoint` (relative
    paths resolve against the toml's directory), `label`, `run`, `epoch`, and
    an optional `search_profile` (see `_resolve_search_profile`); any extra
    scalar keys become display metadata (e.g. `group` for picker grouping,
    `search = "puct"` for the legacy-search tag). One optional top-level
    `sims = [...]` array is the global allowed search-budget set (default
    `DEFAULT_SIMS`). Checkpoints and referenced profiles must exist (the sha
    is part of the bots-table identity key).
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
        profile_ref = entry.get("search_profile")
        search_profile = (
            _resolve_search_profile(str(profile_ref), path.parent)
            if profile_ref is not None
            else None
        )
        meta = {key: value for key, value in entry.items() if key not in _CHECKPOINT_SERVER_KEYS}
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
                search_profile=search_profile,
            )
        )
    return Catalogue(checkpoints=tuple(specs), sims=sims)


# ---------------------------------------------------------------------------
# Worker side (heavy imports happen here, post-spawn)
# ---------------------------------------------------------------------------


class SearchProfile:
    """As-trained search invocation, parsed from a profile TOML.

    The source is either a full training config or a distilled profile from
    `PROFILES_DIR`; both carry the same `[model.config.selfplay]` and
    `[model.config.multi_stage_eval]` sections. Everything except the per-bot
    visit budget comes from there: c_puct, widening, FPU, TSS, and the
    divergence overrides (which carry the Gumbel root/sequential-halving
    levers for a Gumbel-era profile, and leave them off for a PUCT-era one).
    The virtual batch size and opening sampling mirror the eval arena's
    single-root CPU settings rather than the GPU self-play pipeline depth.
    Parsed once per unique profile per worker (see `_WorkerRuntime`).
    """

    def __init__(self, config_path: Path | str) -> None:
        from shrimp.config import build_divergence_overrides, parse_shrimp_config

        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        model_cfg = raw.get("model", {}).get("config", {})
        cfg = parse_shrimp_config(
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

        The kwarg set mirrors the training repo's eval arena exactly.
        root_policy_temperature and root_fpu_reduction are deliberately NOT
        passed even though the Rust signature accepts them: only the self-play
        driver threads them into search, while the eval arena leaves them at
        the Rust defaults for every profile, Gumbel and PUCT alike. As-trained
        serving here means the eval-arena invocation.
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
    """Strict-load a shrimp checkpoint into a fresh eval-mode ShrimpNet.

    Arch (width / head count / trunk layout) is inferred from the state dict
    where determinable, falling back to the env-driven module globals — the
    same contract as the eval arena's loader. A mismatch fails the strict load
    with a clear error rather than serving a half-random net.
    """
    import torch

    from shrimp.model import ShrimpNet, infer_net_kwargs_from_state_dict

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError(f"checkpoint payload has no 'model' state dict: {path}")
    state_dict = payload["model"]
    model = ShrimpNet(**infer_net_kwargs_from_state_dict(state_dict))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@dataclass
class _LoadedBot:
    spec: CheckpointSpec
    model: Any
    evaluator: Any
    session: Any
    profile: SearchProfile


class _WorkerRuntime:
    """Everything one worker process holds resident: the catalogue, one search
    session per checkpoint, and one parsed SearchProfile per unique profile
    toml (checkpoints without a `search_profile` share the
    settings.search_config default).

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

    def __init__(
        self, specs: list[CheckpointSpec], settings: Settings,
        *, device_override: str | None = None,
    ) -> None:
        import torch

        threads = settings.torch_threads or max(
            1, (os.cpu_count() or 2) // max(1, settings.workers)
        )
        torch.set_num_threads(threads)

        from shrimp import _rust
        from shrimp.inference import ShrimpEvaluator

        from . import device as devmod

        # `device_override` is set when the pool forces a specific serving device
        # on a (re)spawn — the GPU->CPU failover path passes "cpu" after this
        # shard's accelerator wedged. An override of "cpu" skips accelerator
        # resolution and the parity self-check entirely (there is nothing to
        # verify and no fallback to take); a `None` override is the happy path
        # that resolves SHOWCASE_DEVICE as before.
        if device_override:
            device = device_override
            log.warning(
                "worker device forced to %r by pool (accelerator failover)",
                device_override,
            )
        else:
            resolved = devmod.resolve_device(settings.device)
            device = resolved.device
            for note in resolved.notes:
                log.warning("%s", note)

        self.policy_floor = settings.policy_floor
        # One parse per unique profile toml; the settings.search_config
        # default is parsed only if some checkpoint actually uses it.
        profiles: dict[Path, SearchProfile] = {}

        def profile_for(spec: CheckpointSpec) -> SearchProfile:
            key = spec.search_profile or Path(settings.search_config)
            profile = profiles.get(key)
            if profile is None:
                profile = profiles[key] = SearchProfile(key)
            return profile

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
                evaluator=ShrimpEvaluator(model, device=device),
                session=_rust.ShrimpMctsSession(max_states=65_536),
                profile=profile_for(spec),
            )
        self.device = device
        log.info(
            "showcase worker ready: device=%s (requested %r), %d checkpoint(s), "
            "%d search profile(s), torch_threads=%d",
            device, settings.device, len(self.bots), len(profiles), threads,
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
        from shrimp.geometry import unpack_action_id

        bot = self.bots[bot_slug]
        state = self._replay(actions)
        entry_player = engine.current_player(state)
        played: list[dict] = []
        ply = len(actions)
        while engine.terminal(state) is None and engine.current_player(state) == entry_player:
            result = bot.profile.search_one(
                bot.session, bot.evaluator, state,
                game_key=game_key,
                visits=int(visits),
                seed=seed * 5003 + ply,
                temperature=bot.profile.move_temperature(ply),
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
                bot.session, bot.evaluator, bot.profile, state,
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

    def lab_eval(
        self, *, bot_slug: str, actions: list[tuple[int, int]] | None,
        stones: tuple[list[tuple[int, int]], list[tuple[int, int]]] | None,
        to_move: int | None, attention_cell: tuple[int, int] | None,
        want_activations: bool, want_features: bool,
    ) -> dict:
        """One hooked lab forward (net readout + requested internals) for a
        sequence or free-edit position. Positions were validated web-side
        (lab_rules); the one remaining user error — an attention query outside
        the support — comes back as a ``reject`` payload so the endpoint can
        422 instead of treating it as a worker failure."""
        from . import lab

        bot = self.bots[bot_slug]
        if actions is not None:
            facts, support, feats = lab.build_sequence_position(actions)
            mode = "sequence"
        else:
            p0, p1 = stones or ([], [])
            mover = to_move if to_move is not None else 0
            facts, support, feats = lab.build_free_position(p0, p1, mover)
            mode = "free"
        try:
            payload = lab.eval_payload(
                bot.model, facts, support, feats,
                policy_floor=self.policy_floor,
                attention_cell=attention_cell,
                want_activations=want_activations,
                want_features=want_features,
            )
        except ValueError as exc:
            return {"reject": str(exc)}
        payload["mode"] = mode
        if mode == "free":
            payload["synthesized_history"] = True
            payload["zeroed_features"] = list(lab.FREE_ZEROED)
        return payload

    def lab_search(
        self, *, bot_slug: str, actions: list[tuple[int, int]], visits: int, seed: int,
    ) -> dict:
        """One capped lab search from a validated placement sequence, under
        the checkpoint's own as-trained profile. Throwaway tree key (high bit
        set, like analysis) — the tree is discarded inside search_payload."""
        from . import lab

        bot = self.bots[bot_slug]
        state = lab.replay_state(actions)
        return lab.search_payload(
            bot.session, bot.evaluator, bot.profile, state,
            game_key=(1 << 63) | (seed & ((1 << 48) - 1)),
            visits=int(visits),
            seed=seed,
        )


def _worker_main(
    worker_index: int, specs: list[CheckpointSpec], settings: Settings,
    job_queue: Any, result_queue: Any, device_override: str | None = None,
) -> None:
    """Worker process entry point: load the catalogue, then serve jobs forever.

    Jobs are `(job_id, kind, kwargs)`; replies are `(job_id, payload)` where
    payload is `{"ok": ...}` or `{"error": traceback}`. Discard jobs carry
    `job_id=None` and get no reply. `None` on the job queue shuts down.

    `device_override` forces the serving device (the pool passes "cpu" when
    respawning a shard whose accelerator wedged); `None` resolves
    SHOWCASE_DEVICE normally.
    """
    # Fresh spawned process: give it a stderr log handler so the one-time
    # device-resolution/self-check lines land in `docker compose logs`.
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s worker-{worker_index} %(name)s %(levelname)s %(message)s",
    )
    try:
        runtime = _WorkerRuntime(specs, settings, device_override=device_override)
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
            elif kind == "lab_eval":
                out = runtime.lab_eval(**kwargs)
            elif kind == "lab_search":
                out = runtime.lab_search(**kwargs)
            elif kind == "discard":
                runtime.discard(**kwargs)
                continue
            else:
                raise ValueError(f"unknown job kind {kind!r}")
        except Exception:
            result_queue.put((job_id, {"error": traceback.format_exc()}))
        else:
            result_queue.put((job_id, {"ok": out}))


def _gpu_probe_main(
    spec: CheckpointSpec, settings: Settings, requested_device: str, out_queue: Any,
) -> None:
    """Throwaway GPU health probe: resolve the requested accelerator, load ONE
    checkpoint, run the CPU-vs-device parity self-check, and report the result.

    Runs in its own short-lived subprocess so that if the probe itself wedges
    the corrupted SYCL/CUDA queue dies with this process — it can never poison a
    serving worker. Posts ``(True, note)`` when the accelerator resolved and the
    parity self-check passed, else ``(False, reason)``. Any exception is caught
    and reported as unhealthy; a hard hang is handled by the parent timing the
    process out and killing it.
    """
    try:
        from . import device as devmod

        resolved = devmod.resolve_device(requested_device)
        if resolved.device == "cpu":
            # The accelerator the shard failed over from is no longer even
            # resolvable (driver gone, etc.) — definitively not healthy.
            out_queue.put((False, f"accelerator unavailable: {resolved.notes}"))
            return
        model = _load_checkpoint(spec.checkpoint)
        check = devmod.verify_device(model, resolved.device)
        if check.ok:
            out_queue.put((True, f"parity ok on {resolved.device}"))
        else:
            out_queue.put(
                (False, f"parity failed on {resolved.device}: {check.error or 'mismatch'}")
            )
    except Exception:  # pragma: no cover - reported as unhealthy
        out_queue.put((False, traceback.format_exc()))


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
        # job_id -> worker index it was routed to, so a recycle can reject every
        # in-flight future stuck on the process it is about to kill.
        self._job_worker: dict[int, int] = {}
        self._job_ids = itertools.count(1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader: threading.Thread | None = None
        # Serializes recycles so two concurrent timeouts on the same shard don't
        # both spawn a replacement; per-worker locks keep independent shards free
        # to recycle in parallel. Created in start() (needs the running loop).
        self._recycle_locks: list[asyncio.Lock] = []
        # Rolling-window recycle counts per worker, to break a respawn loop on a
        # genuinely poisonous game (see _note_recycle).
        self._recycle_times: list[list[float]] = []
        self._poisoned: list[bool] = []
        # Per-shard runtime serving device. `None` means "use the resolved
        # SHOWCASE_DEVICE" (the happy path — the worker resolves it itself);
        # a string (e.g. "cpu") is a forced override threaded into the next
        # (re)spawn. On the GPU->CPU failover this flips to "cpu" for the shard.
        self._worker_devices: list[str | None] = []
        # Rolling-window accelerator-fault timestamps per shard, used to decide
        # when a wedging accelerator shard should fail over to CPU instead of
        # respawning on the accelerator again (see _note_gpu_fault).
        self._gpu_fault_times: list[list[float]] = []
        # Re-promotion (GPU health) state, only touched on the loop thread. The
        # probe tests the shared accelerator, so the healthy-probe streak is
        # pool-wide, not per-shard. A background reprobe task promotes a
        # CPU-downgraded shard back once the streak clears the configured
        # threshold.
        self._reprobe_streak = 0
        self._reprobe_task: asyncio.Task | None = None
        # The accelerator device the pool would serve on absent any failover,
        # resolved once web-side so the reprobe subprocess and downgrade logic
        # know what "the GPU" is. `None` == "cpu was requested / resolved", so
        # there is no accelerator to fail over from or re-promote to.
        self._accel_device: str | None = None
        # worker index -> future awaiting that worker's READY sentinel. The
        # reader thread delivers READY messages here so both startup and recycle
        # can wait for a (re)spawned worker to finish loading the catalogue. A
        # READY that arrives before its waiter is registered is stashed in
        # `_ready_pending` so the waiter picks it up (no lost wakeups).
        self._ready_waiters: dict[int, asyncio.Future] = {}
        self._ready_pending: dict[int, Any] = {}
        self._stopping = False

    def _spawn_proc(self, index: int, queue: Any) -> Any:
        # Thread the shard's current device override (None == resolve
        # SHOWCASE_DEVICE) so a failover respawn comes up forced on CPU.
        override = self._worker_devices[index] if self._worker_devices else None
        proc = self._ctx.Process(
            target=_worker_main,
            args=(
                index, self._specs, self._settings, queue, self._result_queue,
                override,
            ),
            daemon=True,
            name=f"showcase-bot-{index}",
        )
        proc.start()
        return proc

    def _resolve_accel_device(self) -> str | None:
        """The accelerator this deploy would serve on, or None when cpu.

        Derived from the SHOWCASE_DEVICE *request string* only (no torch import
        web-side, preserving the module's no-model-stack guarantee): "cpu" ->
        None; "auto"/"xpu"/"cuda" -> that request, which the worker/probe
        subprocess resolves for real (an accelerator request that no hardware
        can satisfy simply falls back to cpu inside the worker, and the reprobe
        subprocess reports unhealthy, so no shard is ever wrongly promoted).
        """
        req = (self._settings.device or "auto").strip().lower()
        return None if req == "cpu" else req

    async def _await_ready(self, index: int) -> None:
        """Block until worker `index` posts its READY sentinel (delivered by the
        reader thread into `_ready_waiters`)."""
        assert self._loop is not None
        if index in self._ready_pending:  # READY already arrived
            error = self._ready_pending.pop(index)
        else:
            waiter: asyncio.Future = self._loop.create_future()
            self._ready_waiters[index] = waiter
            try:
                error = await asyncio.wait_for(waiter, 300.0)
            except asyncio.TimeoutError:
                self._ready_waiters.pop(index, None)
                raise RuntimeError(f"bot worker {index} did not report ready in time")
        if error is not None:
            raise RuntimeError(f"bot worker {index} failed to start:\n{error}")

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        # Reader thread runs first so it can route the workers' READY sentinels
        # (and, later, recycled workers' READY) to `_ready_waiters`.
        self._reader = threading.Thread(
            target=self._reader_loop, name="showcase-pool-reader", daemon=True
        )
        self._reader.start()
        # Resolve what "the accelerator" is for this deploy (web-side, cheap —
        # no torch model is touched). When SHOWCASE_DEVICE resolves to cpu there
        # is no accelerator to fail over from, so failover/re-promotion are inert.
        self._accel_device = self._resolve_accel_device()
        for index in range(self._settings.workers):
            queue = self._ctx.Queue()
            self._job_queues.append(queue)
            self._recycle_locks.append(asyncio.Lock())
            self._recycle_times.append([])
            self._poisoned.append(False)
            # Start every shard on the resolved device (override None) — a
            # forced "cpu" is only set later, on failover.
            self._worker_devices.append(None)
            self._gpu_fault_times.append([])
            # _spawn_proc reads _worker_devices[index], so append it first.
            self._procs.append(self._spawn_proc(index, queue))
        for index in range(self._settings.workers):
            await self._await_ready(index)
        # Background GPU re-promotion loop (no-op unless an accelerator is in
        # use and re-promotion is enabled). Started after workers are ready so
        # it never races startup.
        if self._accel_device is not None and self._settings.gpu_reprobe_s > 0:
            self._reprobe_task = asyncio.ensure_future(self._reprobe_loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._reprobe_task is not None:
            self._reprobe_task.cancel()
            try:
                await self._reprobe_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reprobe_task = None
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
        self._job_worker.clear()

    def _reader_loop(self) -> None:
        while True:
            item = self._result_queue.get()
            if item is None:
                return
            # READY sentinels are 3-tuples (_READY, worker_index, error); job
            # replies are 2-tuples (job_id, payload). Route the former to the
            # pending ready-waiter, the latter to the job's future.
            if len(item) == 3 and item[0] is _READY:
                _, index, error = item
                self._loop.call_soon_threadsafe(self._deliver_ready, index, error)
                continue
            job_id, payload = item
            # All _futures / _job_worker mutation happens on the loop thread so a
            # recycle (which scans _job_worker) never races the reader's pop.
            self._loop.call_soon_threadsafe(self._resolve, job_id, payload)

    def _deliver_ready(self, index: int, error: Any) -> None:
        waiter = self._ready_waiters.pop(index, None)
        if waiter is not None and not waiter.done():
            waiter.set_result(error)
        else:  # waiter not yet registered — stash for _await_ready to pick up
            self._ready_pending[index] = error

    def _resolve(self, job_id: int, payload: dict) -> None:
        """Deliver a worker reply to its future (loop thread only)."""
        self._job_worker.pop(job_id, None)
        future = self._futures.pop(job_id, None)
        if future is not None and not future.done():
            future.set_result(payload)

    def _route(self, key: int) -> int:
        return int(key) % len(self._job_queues)

    def _note_gpu_fault(self, worker: int) -> int:
        """Record an accelerator fault for `worker` and return the count within
        the rolling window. Used to decide when to fail the shard over to CPU
        (loop thread only)."""
        now = time.monotonic()
        window = self._settings.recycle_window_s
        recent = [t for t in self._gpu_fault_times[worker] if now - t < window]
        recent.append(now)
        self._gpu_fault_times[worker] = recent
        return len(recent)

    def _note_recycle(self, worker: int) -> bool:
        """Record a recycle for `worker`; return True if it is still under the
        rolling-window cap (safe to respawn), False if the shard has recycled
        too often and should be left dead so a poisonous game cannot loop."""
        now = time.monotonic()
        window = self._settings.recycle_window_s
        recent = [t for t in self._recycle_times[worker] if now - t < window]
        recent.append(now)
        self._recycle_times[worker] = recent
        return len(recent) <= self._settings.max_recycles_per_window

    async def _recycle_worker(
        self, worker: int, reason: str, *, device_fault: bool = False,
    ) -> None:
        """Kill and respawn the subprocess backing shard `worker`.

        Every future currently routed to that worker is failed with a
        BotPoolError (its process is gone, so its reply will never come), the
        old process is terminated, and — unless the shard has hit its recycle
        cap — a fresh process is spawned in its place with a brand-new job queue.
        Subsequent jobs routed to `worker` reach the fresh process. Serialized
        by a per-worker lock so racing timeouts recycle once.

        `device_fault` marks a recycle triggered by an accelerator-wedge
        signature (a wedge-marker traceback or a move/search hang on an
        accelerator shard). Once such faults cross `gpu_fault_threshold` inside
        the rolling window, the shard is failed over to CPU (respawned with a
        forced "cpu" device) instead of being poisoned: a slower CPU shard that
        keeps serving beats a dead one. A shard already on CPU is never
        downgraded further — the XPU kernel bug cannot fire there — so a CPU
        fault falls through to the ordinary poison cap.
        """
        assert self._loop is not None
        async with self._recycle_locks[worker]:
            if self._stopping or self._poisoned[worker]:
                return

            # GPU->CPU failover decision (loop thread; inside the recycle lock so
            # it can't race a concurrent recycle of the same shard). Only applies
            # when this recycle was an accelerator fault AND the shard is still
            # running on the accelerator (override None while an accel device is
            # configured). Crossing the threshold flips the shard's override to
            # "cpu" so the respawn below comes up on CPU and stays there until a
            # health re-probe promotes it back.
            on_accelerator = (
                self._accel_device is not None
                and self._worker_devices[worker] is None
            )
            just_failed_over = False
            if device_fault and on_accelerator:
                faults = self._note_gpu_fault(worker)
                if faults >= self._settings.gpu_fault_threshold:
                    just_failed_over = True
                    self._worker_devices[worker] = "cpu"
                    # A fresh downgrade invalidates any in-progress healthy
                    # streak: the accelerator just proved unhealthy.
                    self._reprobe_streak = 0
                    log.error(
                        "shard %d hit %d accelerator faults in %.0fs — failing "
                        "over to CPU (slower but reliable); will re-probe the "
                        "%s backend before promoting back (reason: %s)",
                        worker, faults, self._settings.recycle_window_s,
                        self._accel_device, reason,
                    )

            old_proc = self._procs[worker]
            # Reject in-flight futures pinned to this worker: their process is
            # about to die and no reply will ever arrive for them.
            stranded = [jid for jid, w in self._job_worker.items() if w == worker]
            for jid in stranded:
                self._job_worker.pop(jid, None)
                fut = self._futures.pop(jid, None)
                if fut is not None and not fut.done():
                    fut.set_exception(
                        BotPoolError(f"worker {worker} recycled: {reason}")
                    )
            # Terminate the wedged process off the event loop (join can block).
            def _kill(proc: Any) -> None:
                try:
                    proc.terminate()
                    proc.join(timeout=10)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=5)
                except Exception:  # pragma: no cover - best-effort teardown
                    pass

            await self._loop.run_in_executor(None, _kill, old_proc)

            # The poison cap breaks a respawn loop on a genuinely poisonous
            # accelerator shard. A failover respawn (to CPU) is exempt: CPU
            # cannot re-wedge, so it cannot loop, and poisoning it would defeat
            # the whole point of the fail-over (a dead shard instead of a slow
            # one). We still record the recycle so the window reflects reality.
            under_cap = self._note_recycle(worker)
            if not under_cap and not just_failed_over:
                self._poisoned[worker] = True
                log.error(
                    "shard %d exceeded %d recycles in %.0fs — leaving it dead; "
                    "jobs for this shard will fail fast until restart (reason: %s)",
                    worker, self._settings.max_recycles_per_window,
                    self._settings.recycle_window_s, reason,
                )
                return

            # Fresh queue so any job the dead process never consumed is dropped
            # rather than inherited by the replacement.
            queue = self._ctx.Queue()
            self._job_queues[worker] = queue
            self._procs[worker] = self._spawn_proc(worker, queue)
            log.warning(
                "recycled shard %d (reason: %s); awaiting fresh worker ready",
                worker, reason,
            )
            try:
                await self._await_ready(worker)
            except Exception:
                self._poisoned[worker] = True
                log.exception(
                    "fresh worker for shard %d failed to load; leaving it dead", worker
                )
                return
            log.warning("shard %d back online", worker)

    def _downgraded_shards(self) -> list[int]:
        """Shards currently forced onto CPU by failover (loop thread only)."""
        return [
            w for w in range(len(self._worker_devices))
            if self._worker_devices[w] == "cpu" and not self._poisoned[w]
        ]

    async def _probe_gpu_health(self) -> tuple[bool, str]:
        """Run the device self-check in a short-lived throwaway subprocess and
        return (healthy, note). Bounded by a timeout so a hard GPU hang inside
        the probe is treated as unhealthy (the probe process is then killed),
        never wedging the pool. Never touches a serving worker."""
        assert self._loop is not None and self._accel_device is not None
        probe_queue = self._ctx.Queue()
        proc = self._ctx.Process(
            target=_gpu_probe_main,
            args=(self._specs[0], self._settings, self._accel_device, probe_queue),
            daemon=True,
            name="showcase-gpu-probe",
        )
        proc.start()

        # Read the (blocking) queue and join the probe off the event loop; cap
        # the wait so a wedged probe can't stall re-promotion forever.
        deadline = max(10.0, self._settings.move_timeout_s)

        def _collect() -> tuple[bool, str]:
            try:
                healthy, note = probe_queue.get(timeout=deadline)
            except Exception:
                healthy, note = False, f"probe produced no result in {deadline:g}s"
            try:
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.kill()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
            return healthy, note

        return await self._loop.run_in_executor(None, _collect)

    async def _reprobe_loop(self) -> None:
        """Background GPU re-promotion loop. While any shard is failed over to
        CPU, periodically probe the accelerator; require
        `gpu_reprobe_healthy_streak` consecutive healthy probes (anti-flap),
        then recycle one downgraded shard back onto the accelerator. Runs only
        when an accelerator is configured and re-promotion is enabled."""
        interval = self._settings.gpu_reprobe_s
        need = max(1, self._settings.gpu_reprobe_healthy_streak)
        try:
            while not self._stopping:
                await asyncio.sleep(interval)
                if self._stopping:
                    return
                downgraded = self._downgraded_shards()
                if not downgraded:
                    # Nothing to promote — reset the streak so a promotion later
                    # requires a fresh run of healthy probes.
                    self._reprobe_streak = 0
                    continue
                healthy, note = await self._probe_gpu_health()
                if not healthy:
                    self._reprobe_streak = 0
                    log.info("GPU re-probe unhealthy, staying on CPU: %s", note)
                    continue
                self._reprobe_streak += 1
                log.info(
                    "GPU re-probe healthy (%d/%d): %s",
                    self._reprobe_streak, need, note,
                )
                if self._reprobe_streak < need:
                    continue
                self._reprobe_streak = 0
                # Promote the lowest-indexed downgraded shard back to the
                # accelerator. Clearing the override before recycle makes the
                # respawn come up on the resolved SHOWCASE_DEVICE; its own
                # startup self-check is the final gate before it serves.
                worker = downgraded[0]
                self._worker_devices[worker] = None
                self._gpu_fault_times[worker] = []
                log.warning(
                    "promoting shard %d back to %s after %d healthy probe(s)",
                    worker, self._accel_device, need,
                )
                await self._recycle_worker(worker, "GPU healthy again — re-promote")
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise

    async def _submit(
        self, worker: int, kind: str, kwargs: dict, timeout: float,
        *, recycle_on_hang: bool = False,
    ) -> Any:
        assert self._loop is not None, "BotPool.start() was not awaited"
        if self._poisoned[worker]:
            raise BotPoolError(f"shard {worker} is out of service")
        if self._recycle_locks[worker].locked():
            # Shard is mid-recycle: its process is being replaced, so a job put
            # on the (soon-dead) queue would just hang. Fail fast; the caller
            # already handles BotPoolError as a backend-unavailable outcome.
            raise BotPoolError(f"shard {worker} is recycling")
        job_id = next(self._job_ids)
        future: asyncio.Future = self._loop.create_future()
        self._futures[job_id] = future
        self._job_worker[job_id] = worker
        self._job_queues[worker].put((job_id, kind, kwargs))
        try:
            payload = await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            self._futures.pop(job_id, None)
            self._job_worker.pop(job_id, None)
            if recycle_on_hang:
                # A hung move/search means the worker's device queue is wedged;
                # respawn it so the next job for this shard runs on clean state.
                # A hang on an accelerator shard is a device fault (drives the
                # GPU->CPU failover); on a CPU shard `device_fault` is inert
                # because there is no accelerator to downgrade from.
                await self._recycle_worker(
                    worker, f"{kind} job hung >{timeout:g}s", device_fault=True,
                )
            raise BotPoolTimeout(f"{kind} job exceeded {timeout:.0f}s") from None
        except asyncio.CancelledError:
            self._futures.pop(job_id, None)
            self._job_worker.pop(job_id, None)
            raise
        if "error" in payload:
            if recycle_on_hang and _error_indicates_wedge(payload["error"]):
                # The job raised with a device-fault signature (e.g. an XPU
                # out-of-bounds kernel assert): the SYCL queue is corrupted and
                # every later job on this process would hang. Recycle now, and
                # mark it a device fault so repeated accelerator wedges fail the
                # shard over to CPU rather than poison it.
                await self._recycle_worker(
                    worker, "worker reported device fault", device_fault=True,
                )
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
            self._settings.move_timeout_s,
            recycle_on_hang=True,
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
            recycle_on_hang=True,
        )

    async def summary(self, *, route_key: int, bot_slug: str, actions: list[int]) -> dict:
        return await self._submit(
            self._route(route_key), "summary",
            {"bot_slug": bot_slug, "actions": list(actions)},
            self._settings.bot_timeout_s,
            recycle_on_hang=True,
        )

    async def lab_eval(
        self, *, route_key: int, bot_slug: str,
        actions: list[tuple[int, int]] | None,
        stones: tuple[list[tuple[int, int]], list[tuple[int, int]]] | None,
        to_move: int | None, attention_cell: tuple[int, int] | None,
        want_activations: bool, want_features: bool,
    ) -> dict:
        """Lab hooked forward. `route_key` is a per-request spreader — lab
        trees do not exist (eval never searches), so there is no stickiness to
        preserve and jobs balance across workers."""
        return await self._submit(
            self._route(route_key), "lab_eval",
            {
                "bot_slug": bot_slug, "actions": actions, "stones": stones,
                "to_move": to_move, "attention_cell": attention_cell,
                "want_activations": want_activations, "want_features": want_features,
            },
            self._settings.bot_timeout_s,
            recycle_on_hang=True,
        )

    async def lab_search(
        self, *, route_key: int, bot_slug: str, actions: list[tuple[int, int]],
        visits: int, seed: int,
    ) -> dict:
        """Lab capped search; the tree key is throwaway, so route_key only
        spreads load."""
        return await self._submit(
            self._route(route_key), "lab_search",
            {"bot_slug": bot_slug, "actions": actions, "visits": visits, "seed": seed},
            self._settings.bot_timeout_s,
            recycle_on_hang=True,
        )

    def discard(self, *, game_key: int, bot_slug: str) -> None:
        """Fire-and-forget tree reclamation on the game's sticky worker."""
        self._job_queues[self._route(game_key)].put(
            (None, "discard", {"bot_slug": bot_slug, "game_key": game_key})
        )
