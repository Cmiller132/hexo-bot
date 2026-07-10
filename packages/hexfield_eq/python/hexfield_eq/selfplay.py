"""Continuous self-play epoch driver: the Python side of the on_move protocol.

Per game the driver tracks placement history incrementally (ordinal
phase/player; the graded per-axis window planes are recomputed from history at
expand time), records FULL-search decisions as pending samples, applies the
chosen action through hexo_engine,
and at game end finalizes targets (hard z, opp policy with fast-masking, STV,
moves_left) and writes one hexfield_compact_v1 shard. Truncated games
(max_game_plies reached, no engine winner) are also written: their
outcome-independent heads (policy, opp_policy) train normally while the
value/stvalue/cell_q/moves_left heads are masked via the truncated flag
(outcome_valid=0 column -> value_mask=0 at expand).
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction
from hexo_runner.records import AbortRecord, HexoRecordFile, HexoRecordPlayer

from . import _rust
from .blunder_seeds import mine_blunder_seeds
from .config import (
    ML_AUTO_DISABLED_FLAG,
    build_divergence_overrides,
    build_fast_divergence_overrides,
    parse_hexfield_config,
)
from .engine_facts import player_int
from .features import record_phase, record_player
from .geometry import unpack_action_id
from .inference import build_serve_evaluator
from .samples import (
    STV_HORIZONS,
    HexfieldSampleData,
    _policy_surprise_kl,
    finalize_game_samples,
)
from .shards import write_compact_shard


class _GameTape:
    __slots__ = ("key", "state", "records", "pending", "ply", "seed_ply")

    def __init__(self, key: int):
        self.key = key
        self.state = api.new_game()
        self.records: list[tuple[int, int, int, int]] = []
        self.pending: list[tuple[int, HexfieldSampleData, float]] = []
        self.ply = 0
        # 0 for an ordinary empty-board game; > 0 for a blunder-seeded game
        # whose first seed_ply placements were replayed from a stored position
        # before search began. seed_ply>0 flags the game as seeded (rows carry
        # metadata["seeded"], and its shared opening is excluded from the
        # unique-openings tripwire).
        self.seed_ply = 0


def seed_game_tape(tape: _GameTape, move_prefix) -> None:
    """Replay ``move_prefix`` into ``tape`` so it reaches a stored mid-game
    position via the EXACT path a live game uses.

    For each ``(q, r)`` this mirrors the per-move bookkeeping in
    ``ContinuousDriver.__call__`` (the block at the tail of that method): apply
    the placement through the engine, append ``(q, r, record_player(ply),
    ply+1)`` to ``tape.records`` with ``record_player`` computed from the SAME
    ply counter, and increment ``tape.ply``. No pending sample is created for a
    seed move — the seeded prefix is training-invisible except as history, and
    the .hxr writer records the prefix as leading placements so the game replays
    cleanly.

    Because ``record_player`` / ``record_phase`` are pure functions of the ply
    counter and the records list (NOT the engine state), the resulting tape is
    indistinguishable from one that reached the same position by live play: the
    next row emitted carries identical current_player / phase / first_stone
    labels (the graded window planes derive from the same history). ``tape.seed_ply``
    is set to the prefix length so the game is flagged as seeded and
    ``tape.ply`` starts there (so max_game_plies applies to TOTAL ply)."""

    for q, r in move_prefix:
        current = record_player(tape.ply)
        api.apply_action(tape.state, PlacementAction(AxialCoord(q=int(q), r=int(r))))
        tape.records.append((int(q), int(r), current, tape.ply + 1))
        tape.ply += 1
    tape.seed_ply = tape.ply


class ContinuousDriver:
    def __init__(self, *, epoch: int, games_target: int, max_plies: int, out_dir,
                 horizons=STV_HORIZONS, record_file=None, diag_dir=None, active_limit=0,
                 blunder_seeds=None, blunder_seed_fraction=0.0, blunder_base_seed=0):
        self.epoch = epoch
        self.games_target = games_target
        self.max_plies = max_plies
        self.out_dir = out_dir
        self.horizons = horizons
        # Blunder-seed pool + seeding controls. An empty pool or fraction<=0
        # disables seeding entirely: start_games then takes a path bit-identical
        # to current behavior (no RNG is drawn — the seeding decision short-
        # circuits before the mix_seed call), so fraction=0.0 is a no-op.
        self.blunder_seeds = list(blunder_seeds or [])
        self.blunder_seed_fraction = float(blunder_seed_fraction)
        self.blunder_base_seed = int(blunder_base_seed)
        self.games_seeded = 0
        self.seed_plies: list[int] = []
        # .hxr game-record file for the epoch; None disables recording.
        # Set by generate_selfplay_epoch.
        self.record_file = record_file
        # Directory for the live progress file
        # <diag_dir>/hexfield.selfplay.live.json, written every LIVE_INTERVAL_S
        # while running. None disables the live file.
        self.diag_dir = diag_dir
        self.active_limit = int(active_limit)
        self._t0 = time.time()
        self._last_live = 0.0
        self.games: dict[int, _GameTape] = {}
        self.games_started = 0
        self.games_finished = 0
        self.games_truncated = 0
        self.rows_written = 0
        # main_9: Fast (PCR value-only) + Init decisions still search and stay in
        # tape.pending (targets iterate the full decision list), but are never
        # written -- only FULL rows reach disk. This counts the rows the writer
        # filter dropped across the epoch (parity with dense_cnn_restnet's
        # pcr.fast_rows_excluded telemetry).
        self.fast_rows_excluded = 0
        self.decisions = 0
        self.full_decisions = 0
        self.game_lengths: list[int] = []
        # (ply, entropy) over FULL decisions; ply drives the per-phase split.
        # The bare-entropy mean (root_policy_entropy_mean) is unchanged.
        self.policy_entropies: list[tuple[int, float]] = []
        # (ply, root_value) over every recorded decision; ply drives per-phase
        # value means. root_value_mean stays the mean of all values.
        self.root_values: list[tuple[int, float]] = []
        # KL(visit || root prior) per full row (mirrors the per-row surprise the
        # writer stores); aggregated at epoch end into policy_surprise_* stats.
        self.policy_surprises: list[float] = []
        # Finished-game winner tally (completed games only; truncated games have
        # no engine winner and are counted separately by games_truncated).
        self.wins_by_player: dict[int, int] = {0: 0, 1: 0}
        # Opening-diversity tripwire: distinct first-N-ply lines across the
        # epoch's finished games. Should stay near games_finished; a collapse
        # means the play sampler has become too exploitative. opening_lines is
        # the legacy 10-ply set (kept for unique_openings_10ply); 16/20-ply sets
        # extend the tripwire deeper into the opening book.
        self.opening_lines: set[tuple] = set()
        self.opening_lines_16: set[tuple] = set()
        self.opening_lines_20: set[tuple] = set()
        # Seeded games share stored openings; counted separately so they do not
        # dilute the diversity tripwire above (unique_openings_seeded in stats).
        self.opening_lines_seeded: set[tuple] = set()
        self.next_key = epoch * 1_000_000
        # Background shard writer: the per-game finalize + .hxr record + zlib npz
        # write runs off the on_move callback thread.
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_errors: list[BaseException] = []
        self._writer_failed = threading.Event()
        self._writer_thread: threading.Thread | None = None

    LIVE_INTERVAL_S = 3.0

    def _write_live(self, status: str) -> None:
        """Write hexfield.selfplay.live.json with epoch progress and
        positions/second. Throttled to LIVE_INTERVAL_S while status=="running";
        other statuses always write. No-op when diag_dir is None."""

        if self.diag_dir is None:
            return
        now = time.time()
        if status == "running" and (now - self._last_live) < self.LIVE_INTERVAL_S:
            return
        self._last_live = now
        elapsed = max(now - self._t0, 1e-9)
        pps = self.decisions / elapsed
        payload = {
            "status": status,
            "epoch": self.epoch,
            "timestamp": now,
            "requested_games": self.games_target,
            "games_started": self.games_started,
            "completed_games": self.games_finished - self.games_truncated,
            "truncated_games": self.games_truncated,
            "games_finished": self.games_finished,
            "active_games": len(self.games),
            "active_limit": self.active_limit,
            "searched_positions": self.decisions,
            "elapsed_seconds": elapsed,
            "search_positions_per_second": pps,
            "positions_per_second": pps,
            "full_decisions": self.full_decisions,
            "scheduler": "continuous",
        }
        # Write failures here are swallowed so they cannot interrupt self-play.
        try:
            path = self.diag_dir / "hexfield.selfplay.live.json"
            tmp = self.diag_dir / "hexfield.selfplay.live.json.tmp"
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(path)  # atomic replace
        except Exception:
            pass

    # mix_seed stream index reserved for the blunder-seed decision. Chosen well
    # outside the range of stream ids the Rust search uses for its own per-game
    # streams, so this draw cannot collide with / perturb any existing stream.
    _BLUNDER_STREAM = 0x0B1D_0000

    def _maybe_seed(self, tape: _GameTape) -> None:
        """With probability ``blunder_seed_fraction`` (a deterministic per-game
        Bernoulli drawn from a fresh mix_seed stream), replay a mined seed
        prefix into ``tape`` before search sees it.

        The decision and the seed pick are derived from ``mix_seed(base_seed,
        game_key, 0, _BLUNDER_STREAM)`` — a NEW stream keyed by the game key, so
        the choice is reproducible for a given epoch/game and independent of the
        Rust search streams. No RNG is drawn when seeding is disabled (empty
        pool or fraction<=0), so the fraction=0.0 path is bit-identical to the
        pre-feature driver."""

        if not self.blunder_seeds or self.blunder_seed_fraction <= 0.0:
            return
        h = int(_rust.mix_seed(self.blunder_base_seed, int(tape.key), 0, self._BLUNDER_STREAM))
        # Two independent uniforms from one 64-bit hash: the high 32 bits gate
        # the Bernoulli draw, the low 32 bits pick the seed index.
        u_gate = (h >> 32) / float(1 << 32)
        if u_gate >= self.blunder_seed_fraction:
            return
        idx = (h & 0xFFFF_FFFF) % len(self.blunder_seeds)
        seed = self.blunder_seeds[idx]
        seed_game_tape(tape, seed.move_prefix)
        self.games_seeded += 1
        self.seed_plies.append(int(tape.seed_ply))

    def start_games(self, count: int) -> list[_GameTape]:
        tapes = []
        for _ in range(count):
            tape = _GameTape(self.next_key)
            self.next_key += 1
            self._maybe_seed(tape)
            self.games_started += 1
            self.games[tape.key] = tape
            tapes.append(tape)
        return tapes

    def __call__(self, game_key: int, payload: dict[str, Any]):
        tape = self.games[game_key]
        action_id = int(payload["action_id"])
        full = bool(payload["pcr_full"])
        init = bool(payload["policy_init"])
        self.decisions += 1
        self._write_live("running")

        current = record_player(tape.ply)
        # Seed provenance stamped on every row of a blunder-seeded game. Empty
        # for ordinary games, so their metadata is unchanged (fraction=0.0 keeps
        # rows bit-identical to the pre-feature writer).
        seed_meta = (
            {"seeded": True, "seed_ply": int(tape.seed_ply)} if tape.seed_ply else {}
        )
        if full and not init:
            self.full_decisions += 1
            ids = np.frombuffer(bytes(payload["visit_policy_action_ids_bytes"]), dtype=np.uint32)
            weights = np.frombuffer(bytes(payload["visit_policy_weights_bytes"]), dtype=np.float32)
            # Per-cell Q: one Q per recorded action, same set and order as the
            # visit policy. Feeds the cell_q head.
            qs = np.frombuffer(bytes(payload["visit_policy_q_bytes"]), dtype=np.float32)
            # Policy-surprise = KL(visit || root prior).
            prior_ids = np.frombuffer(
                bytes(payload["root_prior_policy_action_ids_bytes"]), dtype=np.uint32
            )
            prior_weights = np.frombuffer(
                bytes(payload["root_prior_policy_weights_bytes"]), dtype=np.float32
            )
            surprise = _policy_surprise_kl(ids, weights, prior_ids, prior_weights)
            # Improved-policy target π' and raw root logits are present only when
            # gumbel_target is enabled; otherwise the keys are absent from the
            # payload and these stay empty (visit policy is used as the target).
            gumbel_pairs: tuple[tuple[int, float], ...] = ()
            prior_logit_pairs: tuple[tuple[int, float], ...] = ()
            if "gumbel_policy_action_ids_bytes" in payload:
                g_ids = np.frombuffer(
                    bytes(payload["gumbel_policy_action_ids_bytes"]), dtype=np.uint32
                )
                g_weights = np.frombuffer(
                    bytes(payload["gumbel_policy_weights_bytes"]), dtype=np.float32
                )
                gumbel_pairs = tuple(
                    zip((int(a) for a in g_ids), (float(w) for w in g_weights))
                )
                if "root_prior_logits_bytes" in payload:
                    g_logits = np.frombuffer(
                        bytes(payload["root_prior_logits_bytes"]), dtype=np.float32
                    )
                    prior_logit_pairs = tuple(
                        zip((int(a) for a in g_ids), (float(l) for l in g_logits))
                    )
            phase = record_phase(tape.ply)
            first_stone = (
                (tape.records[-1][0], tape.records[-1][1]) if phase == "SecondStone" else None
            )
            sample = HexfieldSampleData(
                game_id=str(game_key),
                turn_index=tape.ply,
                current_player=current,
                phase=phase,
                records=tuple(tape.records),
                first_stone=first_stone,
                policy=tuple(zip((int(a) for a in ids), (float(w) for w in weights))),
                q_policy=tuple(zip((int(a) for a in ids), (float(q) for q in qs))),
                gumbel_policy=gumbel_pairs,
                prior_logit=prior_logit_pairs,
                policy_surprise=float(surprise),
                metadata={"pcr_full": True, **seed_meta},
            )
            tape.pending.append((current, sample, float(payload["root_value"])))
            probs = weights[weights > 0]
            if probs.size:
                self.policy_entropies.append(
                    (tape.ply, float(-(probs * np.log(probs)).sum()))
                )
            # Aggregate the per-row policy-surprise (KL(visit || prior)) computed
            # above; epoch-end stats() reduces this to mean/p90/max.
            self.policy_surprises.append(float(surprise))
            self.root_values.append((tape.ply, float(payload["root_value"])))
        elif not full and not init:
            # Fast (PCR value-only) decisions are NEVER written to training shards
            # (main_9 records only full-search turns, matching KataGo). They are
            # still appended to tape.pending so full rows' opp-policy lookup,
            # moves_left counts, and short-term-value targets keep iterating the
            # complete decision list; the writer filter drops them before any row
            # is written. Feature planes are left empty (fast rows never persist).
            sample = HexfieldSampleData(
                game_id=str(game_key), turn_index=tape.ply, current_player=current,
                phase=record_phase(tape.ply), records=tuple(tape.records),
                first_stone=None,
                policy=(), metadata={"pcr_full": False, **seed_meta},
            )
            tape.pending.append((current, sample, float(payload["root_value"])))
        else:
            sample = HexfieldSampleData(
                game_id=str(game_key), turn_index=tape.ply, current_player=current,
                phase=record_phase(tape.ply), records=tuple(tape.records),
                first_stone=None,
                policy=(), metadata={"pcr_full": False, "policy_init": True, **seed_meta},
            )
            tape.pending.append((current, sample, float(payload["root_value"])))

        q, r = unpack_action_id(action_id)
        result = api.apply_action(tape.state, PlacementAction(AxialCoord(q=q, r=r)))
        tape.records.append((q, r, current, tape.ply + 1))
        tape.ply += 1

        if result.terminal:
            terminal = api.terminal(tape.state)
            self._finish(tape, winner=player_int(terminal.winner), truncated=False)
        elif tape.ply >= self.max_plies:
            self._finish(tape, winner=None, truncated=True)
        else:
            return ("advance", tape.state)

        del self.games[game_key]
        if self.games_started < self.games_target:
            fresh = self.start_games(1)[0]
            return ("replace", fresh.key, fresh.state)
        return None

    def _write_record(self, tape: _GameTape, *, winner, truncated: bool) -> None:
        """Write one ``.hxr`` game record. Every finished game is recorded
        (completed and truncated). Records the placement sequence in move order,
        then closes the game with the engine winner label (``player0``/``player1``)
        for completed games or an abort record for truncated games. No-op when
        record_file is None."""

        if self.record_file is None:
            return
        writer = self.record_file.begin_game(
            f"epoch-{self.epoch:06d}-game-{tape.key}", seed=tape.key
        )
        for q, r, _player, _ply in tape.records:
            writer.record_action(PlacementAction(AxialCoord(q=int(q), r=int(r))))
        if truncated:
            writer.finish_aborted(
                AbortRecord(
                    stage="selfplay",
                    exception_type="MaxPliesReached",
                    message=f"hexfield self-play reached max_plies={self.max_plies}",
                )
            )
        else:
            writer.finish_completed(f"player{int(winner)}", tape.ply)

    def _finish(self, tape: _GameTape, *, winner, truncated: bool) -> None:
        self.games_finished += 1
        self.game_lengths.append(tape.ply)
        line = tuple((q, r) for q, r, _o, _p in tape.records)
        # Opening-diversity tripwire: seeded games share their STORED opening
        # (many games replay the same mined prefix), which would falsely collapse
        # the unique-openings counts and blunt the diversity tripwire. Stratify:
        # only self-generated (empty-board) openings feed the main counters; the
        # seeded games' openings are tallied separately as unique_openings_seeded.
        # getattr keeps _finish tolerant of tape-likes lacking the attribute
        # (unseeded default), matching the defensive metadata handling elsewhere.
        if getattr(tape, "seed_ply", 0):
            self.opening_lines_seeded.add(line[:10])
        else:
            self.opening_lines.add(line[:10])
            self.opening_lines_16.add(line[:16])
            self.opening_lines_20.add(line[:20])
        if truncated:
            self.games_truncated += 1
        else:
            # winner is None only on the truncated path.
            assert winner is not None, "non-truncated finish requires an engine winner"
            # Winner-side balance over completed games (winner is 0/1).
            self.wins_by_player[int(winner)] = self.wins_by_player.get(int(winner), 0) + 1
        # Surface a prior writer-thread failure before queueing more work.
        if self._writer_failed.is_set():
            raise self._writer_errors[0]
        # Hand the finished tape to the background writer. The tape is not mutated
        # after the game ends; __call__ deletes it from self.games after this
        # returns.
        self._write_queue.put((tape, winner, truncated))

    def _writer_loop(self) -> None:
        """Background shard writer. Drains finished games from _write_queue and
        does the I/O -- .hxr record, finalize, and the zlib `write_compact_shard`
        -- off the search-callback thread. A write failure is captured in
        _writer_errors and _writer_failed is set. Exits on a None sentinel."""

        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return
                tape, winner, truncated = item
                # Record the game (completed and truncated) before finalizing.
                self._write_record(tape, winner=winner, truncated=truncated)
                # Truncated games (max_game_plies hit, no engine winner) still
                # have rows written: the outcome-independent heads (policy,
                # opp_policy) train on them, while the value / stvalue / cell_q /
                # moves_left heads are masked downstream (truncated metadata flag
                # -> outcome_valid=0 shard column -> value_mask=0 + zeroed
                # stvalue/cell_q masks at expand).
                finalized = finalize_game_samples(
                    tape.pending, winner, self.horizons,
                    truncated=truncated, mask_opp_from_fast=True,
                )
                # main_9: match KataGo -- record/train ONLY on FULL-search rows.
                # Fast (PCR value-only) and Init decisions still play their search
                # and stay in tape.pending (so full rows' STV / opp-policy /
                # moves_left targets iterate the COMPLETE decision list), but they
                # are never WRITTEN. pcr_full defaults False so a row lacking the
                # tag is treated as non-full and excluded.
                rows = [s for s in finalized if s.metadata.get("pcr_full", False)]
                # Fast + Init rows dropped from the written shard this game.
                self.fast_rows_excluded += len(finalized) - len(rows)
                if rows:
                    path = self.out_dir / f"game_{tape.key}.npz"
                    # Seeded games carry their provenance in the sidecar so a
                    # blunder-seeded shard is identifiable on disk. Ordinary
                    # games omit these keys, keeping the sidecar byte-identical
                    # to the pre-feature writer (seed_ply == 0).
                    _seed_ply = getattr(tape, "seed_ply", 0)
                    seed_sidecar = (
                        {"seeded": True, "seed_ply": int(_seed_ply)}
                        if _seed_ply
                        else {}
                    )
                    self.rows_written += write_compact_shard(
                        path, rows, short_term_value_horizons=self.horizons,
                        sidecar={
                            "epoch": self.epoch, "game_key": tape.key,
                            "winner": winner, "truncated": bool(truncated),
                            **seed_sidecar,
                        },
                    )
            except BaseException as exc:  # noqa: BLE001
                self._writer_errors.append(exc)
                self._writer_failed.set()
            finally:
                self._write_queue.task_done()

    def _start_writer(self) -> None:
        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="hexfield-selfplay-writer", daemon=True
        )
        self._writer_thread.start()

    def _stop_writer(self) -> None:
        """Enqueue the None sentinel, join the writer thread, then re-raise any
        writer error. No-op when no writer thread is running."""

        if self._writer_thread is None:
            return
        self._write_queue.put(None)
        self._writer_thread.join()
        self._writer_thread = None
        if self._writer_errors:
            raise self._writer_errors[0]

    def stats(self) -> dict[str, Any]:
        lengths = np.asarray(self.game_lengths or [0], dtype=np.float64)
        # Per-phase entropy over full decisions and per-phase value over all
        # decisions; both split on the decision ply (opening<20, 20<=mid<60,
        # late>=60). Bare means (root_policy_entropy_mean / root_value_mean) are
        # unchanged so existing readers keep working.
        entropies = [e for _ply, e in self.policy_entropies]
        values = [v for _ply, v in self.root_values]
        vals_arr = np.asarray(values, dtype=np.float64) if values else None
        return {
            "games_started": self.games_started,
            "games_finished": self.games_finished,
            "truncated_games": self.games_truncated,
            "rows_written": self.rows_written,
            "fast_rows_excluded": self.fast_rows_excluded,
            "total_decisions": self.decisions,
            "full_decisions": self.full_decisions,
            "mean_game_length": float(lengths.mean()),
            "p90_game_length": float(np.percentile(lengths, 90)),
            # Game-length distribution (mean_game_length + p90_game_length above
            # are kept unchanged; these extend it).
            "game_length_p10": float(np.percentile(lengths, 10)),
            "game_length_p50": float(np.percentile(lengths, 50)),
            "game_length_p90": float(np.percentile(lengths, 90)),
            "game_length_max": float(lengths.max()),
            "root_policy_entropy_mean": float(np.mean(entropies)) if entropies else None,
            "root_policy_entropy_by_phase": _phase_means(self.policy_entropies),
            "root_value_mean": float(vals_arr.mean()) if vals_arr is not None else None,
            "root_value_abs_mean": float(np.abs(vals_arr).mean()) if vals_arr is not None else None,
            "root_value_std": float(vals_arr.std()) if vals_arr is not None else None,
            "root_value_by_phase": _phase_means(self.root_values),
            "decided_fraction": (
                float(np.mean(np.abs(vals_arr) > 0.8)) if vals_arr is not None else None
            ),
            "policy_surprise_mean": (
                float(np.mean(self.policy_surprises)) if self.policy_surprises else None
            ),
            "policy_surprise_p90": (
                float(np.percentile(self.policy_surprises, 90)) if self.policy_surprises else None
            ),
            "policy_surprise_max": (
                float(np.max(self.policy_surprises)) if self.policy_surprises else None
            ),
            "wins_by_player": {
                "0": int(self.wins_by_player.get(0, 0)),
                "1": int(self.wins_by_player.get(1, 0)),
            },
            "unique_openings_10ply": len(self.opening_lines),
            "unique_openings": {
                "10": len(self.opening_lines),
                "16": len(self.opening_lines_16),
                "20": len(self.opening_lines_20),
            },
            # Blunder-seeded self-play telemetry. games_seeded counts games that
            # started from a mined mid-game position; seed_ply_mean is the mean
            # seed depth (None when none seeded). unique_openings_seeded is the
            # distinct 10-ply seeded-opening count, tracked apart from the
            # diversity tripwire above (seeded games share stored openings).
            "games_seeded": int(self.games_seeded),
            "seed_ply_mean": (
                float(np.mean(self.seed_plies)) if self.seed_plies else None
            ),
            "unique_openings_seeded": len(self.opening_lines_seeded),
        }


# --- Telemetry helpers ------------------------------------------------------
# Diagnostic-phase boundaries (decision ply): opening < 20 <= mid < 60 <= late.
_PHASE_BOUNDS = (20, 60)
# Segment cap for merged (crash-resumed) epoch diagnostics. Bounds "segments"
# growth if a single epoch is crashed and resumed many times; only the counters
# and weighted means below need every segment, and they are already summed into
# the top level, so keeping the last N raw payloads is enough for forensics.
_MAX_SEGMENTS = 8


def _phase_of(ply: int) -> str:
    """Classify a decision ply into opening / mid / late."""

    if ply < _PHASE_BOUNDS[0]:
        return "opening"
    if ply < _PHASE_BOUNDS[1]:
        return "mid"
    return "late"


def _phase_means(pairs: list[tuple[int, float]]) -> dict[str, dict[str, Any]]:
    """Per-phase mean + count from (ply, value) pairs. Always returns all three
    phase keys; empty phases carry {"mean": None, "n": 0} so the schema is
    stable across epochs regardless of game lengths seen."""

    buckets: dict[str, list[float]] = {"opening": [], "mid": [], "late": []}
    for ply, value in pairs:
        buckets[_phase_of(int(ply))].append(float(value))
    out: dict[str, dict[str, Any]] = {}
    for phase, vals in buckets.items():
        out[phase] = {
            "mean": float(np.mean(vals)) if vals else None,
            "n": len(vals),
        }
    return out


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write pretty JSON to path via a tmp file + os.replace (atomic on the same
    filesystem), so a crash mid-write never leaves a truncated diagnostic."""

    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_prior_diag(path: Path) -> dict[str, Any] | None:
    """Load a prior epoch diagnostic if present and parseable, else None. A
    corrupt/partial prior diag is treated as absent (we would rather write a
    fresh record than crash the epoch on a garbage file)."""

    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _diag_is_nontrivial(diag: dict[str, Any] | None) -> bool:
    """A prior diag is worth preserving when it recorded real self-play:
    games_finished>0 or a non-empty scheduler dict."""

    if not isinstance(diag, dict):
        return False
    if int(diag.get("games_finished", 0) or 0) > 0:
        return True
    scheduler = diag.get("scheduler")
    return isinstance(scheduler, dict) and bool(scheduler)


# Additive integer counters merged by summation across segments. Every integer
# in the scheduler sub-dict is summed key-wise as well (handled separately).
_ADDITIVE_KEYS = (
    "games_started", "games_finished", "truncated_games", "rows_written",
    "fast_rows_excluded",
    "total_decisions", "full_decisions", "searched_positions", "elapsed_seconds",
    "games_seeded",
)


def _segment_payload(diag: dict[str, Any]) -> dict[str, Any]:
    """Extract the raw payloads a merged diag stores per segment: everything
    except the recursive merge bookkeeping (avoids nesting segments-in-segments
    when a merged diag is itself a prior for a later resume)."""

    return {k: v for k, v in diag.items() if k not in ("segments", "merged_approx")}


def _merge_scheduler(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Key-wise sum of integer scheduler counters across segments. Non-int
    values (floats like ratios/seconds) take the last segment's value — they are
    rates, not additive counters, and the resumed segment is the freshest."""

    merged: dict[str, Any] = {}
    for seg in segments:
        sched = seg.get("scheduler")
        if not isinstance(sched, dict):
            continue
        for key, value in sched.items():
            if isinstance(value, bool):
                merged[key] = value
            elif isinstance(value, int):
                merged[key] = merged.get(key, 0) + value
            else:
                merged[key] = value
    return merged


def _weighted_mean(pairs: list[tuple[float | None, float]]) -> float | None:
    """Weighted mean of (value, weight) pairs, skipping None values and
    non-positive weights. Returns None if no usable pair remains."""

    num = 0.0
    den = 0.0
    for value, weight in pairs:
        if value is None or weight <= 0:
            continue
        num += float(value) * float(weight)
        den += float(weight)
    return (num / den) if den > 0 else None


def _merge_epoch_diag(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-segment epoch diagnostics (prior crashes first, resumed run
    last) into one whole-epoch record.

    Rules (see PART 1b of the telemetry spec):
    - additive counters (games_*, rows_written, *decisions, searched_positions,
      elapsed_seconds) and every integer scheduler counter are summed;
    - means are recombined as weighted means (entropy by full_decisions, value
      by total_decisions, game-length percentiles by games_finished, marking
      "merged_approx" for percentiles that cannot be merged exactly);
    - unique-opening counts are summed and flagged merged_approx (they cannot be
      recovered exactly from per-segment counts);
    - every existing top-level key/type from the last segment is preserved so
      downstream readers keep working; merged epochs additionally carry
      "segments" and "merged_approx". Keys present in only one segment pass
      through from the last segment that has them."""

    # Start from the union of every segment (last wins), so single-segment keys
    # survive and the schema/type of each existing key is preserved.
    merged: dict[str, Any] = {}
    for seg in segments:
        for key, value in seg.items():
            if key in ("segments", "merged_approx"):
                continue
            merged[key] = value

    # Additive top-level counters.
    for key in _ADDITIVE_KEYS:
        present = [seg[key] for seg in segments if isinstance(seg.get(key), (int, float))]
        if present:
            total = sum(present)
            merged[key] = int(total) if all(isinstance(v, int) for v in present) else float(total)

    # Scheduler: key-wise integer sum.
    if any(isinstance(seg.get("scheduler"), dict) for seg in segments):
        merged["scheduler"] = _merge_scheduler(segments)

    def _w(key: str, weight_key: str) -> list[tuple[float | None, float]]:
        return [
            (seg.get(key), float(seg.get(weight_key, 0) or 0))
            for seg in segments
        ]

    # Weighted means. Weights: entropy by full_decisions, value by
    # total_decisions, lengths by games_finished.
    approx = False
    if any("root_policy_entropy_mean" in seg for seg in segments):
        merged["root_policy_entropy_mean"] = _weighted_mean(
            _w("root_policy_entropy_mean", "full_decisions")
        )
    for vkey in ("root_value_mean", "root_value_abs_mean", "decided_fraction"):
        if any(vkey in seg for seg in segments):
            merged[vkey] = _weighted_mean(_w(vkey, "total_decisions"))
    # std cannot be merged exactly from per-segment std; games-weighted by
    # total_decisions as an approximation.
    if any("root_value_std" in seg for seg in segments):
        merged["root_value_std"] = _weighted_mean(_w("root_value_std", "total_decisions"))
        approx = True

    # Per-phase means: recombine each phase's mean weighted by its own n, and
    # sum the n's. Both phase dicts share the {phase: {mean, n}} shape.
    for pkey in ("root_policy_entropy_by_phase", "root_value_by_phase"):
        phase_dicts = [seg[pkey] for seg in segments if isinstance(seg.get(pkey), dict)]
        if not phase_dicts:
            continue
        out: dict[str, dict[str, Any]] = {}
        for phase in ("opening", "mid", "late"):
            pairs = [
                (pd[phase].get("mean"), float(pd[phase].get("n", 0) or 0))
                for pd in phase_dicts
                if isinstance(pd.get(phase), dict)
            ]
            n_total = int(sum(w for _v, w in pairs))
            out[phase] = {"mean": _weighted_mean(pairs), "n": n_total}
        merged[pkey] = out

    # Game-length percentiles/mean: games-weighted average, approximate.
    length_keys = [
        "mean_game_length", "p90_game_length",
        "game_length_p10", "game_length_p50", "game_length_p90",
    ]
    if any(k in seg for seg in segments for k in length_keys):
        for lkey in length_keys:
            if any(lkey in seg for seg in segments):
                merged[lkey] = _weighted_mean(_w(lkey, "games_finished"))
        approx = True
    # max is exact under merge.
    if any("game_length_max" in seg for seg in segments):
        maxes = [seg["game_length_max"] for seg in segments if seg.get("game_length_max") is not None]
        merged["game_length_max"] = float(max(maxes)) if maxes else None

    # Policy-surprise mean is decision-weighted (approx); p90/max approximated by
    # max across segments for max and full-decision-weighted mean for p90.
    if any("policy_surprise_mean" in seg for seg in segments):
        merged["policy_surprise_mean"] = _weighted_mean(
            _w("policy_surprise_mean", "full_decisions")
        )
    if any("policy_surprise_p90" in seg for seg in segments):
        merged["policy_surprise_p90"] = _weighted_mean(
            _w("policy_surprise_p90", "full_decisions")
        )
        approx = True
    if any("policy_surprise_max" in seg for seg in segments):
        smaxes = [
            seg["policy_surprise_max"] for seg in segments
            if seg.get("policy_surprise_max") is not None
        ]
        merged["policy_surprise_max"] = float(max(smaxes)) if smaxes else None

    # Winner tally: key-wise sum (exact).
    if any(isinstance(seg.get("wins_by_player"), dict) for seg in segments):
        wins = {"0": 0, "1": 0}
        for seg in segments:
            wb = seg.get("wins_by_player")
            if isinstance(wb, dict):
                for k in ("0", "1"):
                    wins[k] += int(wb.get(k, 0) or 0)
        merged["wins_by_player"] = wins

    # Unique-opening counts cannot be merged exactly from per-segment counts;
    # report the SUM and flag approximate.
    if any("unique_openings_10ply" in seg for seg in segments):
        merged["unique_openings_10ply"] = sum(
            int(seg.get("unique_openings_10ply", 0) or 0) for seg in segments
        )
        approx = True
    if any(isinstance(seg.get("unique_openings"), dict) for seg in segments):
        uo = {"10": 0, "16": 0, "20": 0}
        for seg in segments:
            d = seg.get("unique_openings")
            if isinstance(d, dict):
                for k in ("10", "16", "20"):
                    uo[k] += int(d.get(k, 0) or 0)
        merged["unique_openings"] = uo
        approx = True
    # Seeded-opening count sums like the other unique-opening counts (approx).
    if any("unique_openings_seeded" in seg for seg in segments):
        merged["unique_openings_seeded"] = sum(
            int(seg.get("unique_openings_seeded", 0) or 0) for seg in segments
        )
        approx = True
    # Mean seed depth is games_seeded-weighted across segments.
    if any("seed_ply_mean" in seg for seg in segments):
        merged["seed_ply_mean"] = _weighted_mean(_w("seed_ply_mean", "games_seeded"))

    # Store raw segment payloads (prior first, resumed last), capped.
    raw = [_segment_payload(seg) for seg in segments]
    merged["segments"] = raw[-_MAX_SEGMENTS:]
    merged["merged_approx"] = approx
    return merged


def _skip_path_result(
    prior: dict[str, Any] | None, skip_record: dict[str, Any]
) -> dict[str, Any]:
    """Decide what the remaining==0 skip path writes.

    - If a non-trivial prior diag exists (games_finished>0 or a non-empty
      scheduler), PRESERVE it verbatim and merge in a resumed-skip annotation
      (bumping resumed_skip_count) rather than replacing content.
    - Otherwise write the skip record, marked resumed_skip + prior_diag_missing
      so the lost telemetry is visible."""

    if _diag_is_nontrivial(prior):
        preserved = dict(prior)  # copy; do not mutate the loaded object
        preserved["resumed_skip"] = True
        preserved["resumed_skip_count"] = int(preserved.get("resumed_skip_count", 0) or 0) + 1
        return preserved
    result = dict(skip_record)
    result["resumed_skip"] = True
    result["prior_diag_missing"] = True
    return result


def _derive_scheduler_rates(result: dict[str, Any]) -> None:
    """Add div-by-zero-guarded derived rates into result, in place, from the
    scheduler counters and total_decisions. No-op-safe on missing keys."""

    sched = result.get("scheduler") or {}
    total = int(result.get("total_decisions", 0) or 0)

    def _rate(num_key: str, den: float) -> float | None:
        num = sched.get(num_key)
        if num is None or den <= 0:
            return None
        return float(num) / float(den)

    result["gumbel_play_winner_rate"] = _rate(
        "gumbel_play_winner_moves", sched.get("gumbel_play_moves", 0) or 0
    )
    result["gumbel_play_winner_early_rate"] = _rate(
        "gumbel_play_winner_early", sched.get("gumbel_play_moves_early", 0) or 0
    )
    result["lcb_override_rate"] = _rate("lcb_overrides", total)
    decided = float(sched.get("moves_decided", 0) or 0)
    result["fast_fraction"] = _rate("fast_moves", decided)
    result["full_fraction"] = _rate("full_moves", decided)
    result["init_fraction"] = _rate("init_moves", decided)


def _fmt(value: Any, spec: str = ".2f", default: str = "?") -> str:
    """Format a possibly-None numeric for the one-line summary."""

    if value is None:
        return default
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return default


def _format_epoch_summary(result: dict[str, Any]) -> str:
    """Compact one-line human-readable epoch summary for the supervisor's
    train.out log. All fields guarded for missing values."""

    epoch = result.get("epoch", "?")
    finished = int(result.get("games_finished", 0) or 0)
    trunc = int(result.get("truncated_games", 0) or 0)
    rows = int(result.get("rows_written", 0) or 0)
    by_phase = result.get("root_policy_entropy_by_phase") or {}

    def _ph(name: str) -> str:
        cell = by_phase.get(name) if isinstance(by_phase, dict) else None
        return _fmt(cell.get("mean") if isinstance(cell, dict) else None, ".1f")

    uo = result.get("unique_openings") or {}
    wins = result.get("wins_by_player") or {}
    w0 = int(wins.get("0", 0) or 0)
    w1 = int(wins.get("1", 0) or 0)
    p0_share = (w0 / (w0 + w1)) if (w0 + w1) > 0 else None
    seeded = int(result.get("games_seeded", 0) or 0)
    # Seed segment appears only when the epoch actually seeded games, so the
    # default (fraction=0) summary line is unchanged.
    seed_frag = (
        f"| seeded {seeded} (ply {_fmt(result.get('seed_ply_mean'), '.0f')}) "
        if seeded
        else ""
    )
    return (
        f"selfplay epoch {epoch}: {finished} games ({trunc} trunc) {rows} rows "
        f"| len p50 {_fmt(result.get('game_length_p50'), '.0f')} "
        f"p90 {_fmt(result.get('game_length_p90'), '.0f')} "
        f"| ent {_fmt(result.get('root_policy_entropy_mean'))} "
        f"(open {_ph('opening')}/mid {_ph('mid')}/late {_ph('late')}) "
        f"| uniq10/16/20 {uo.get('10', '?')}/{uo.get('16', '?')}/{uo.get('20', '?')} "
        f"{seed_frag}"
        f"| winner-rate {_fmt(result.get('gumbel_play_winner_rate'))} "
        f"| P0 wins {_fmt(p0_share)}"
    )


def generate_selfplay_epoch(*, ctx, components, epoch: int, games_per_epoch: int) -> dict[str, Any]:
    cfg = parse_hexfield_config(ctx.config.model.config)
    sp = cfg.selfplay
    model = components.model.model
    # Self-play already runs under the full serve env from the supervisor;
    # auto_match_serve_env=False keeps this construction byte-identical (no env
    # mutation, no import-gate warning), so epoch diagnostics are unchanged.
    evaluator = build_serve_evaluator(
        model, cfg, role="selfplay", auto_match_serve_env=False
    )

    out_dir = ctx.samples_dir / f"epoch_{epoch:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    games_target = max(int(games_per_epoch), 1)
    # Resume support: completed games already wrote their shards. On a restart,
    # keep the existing shards and generate only the remainder, using keys past
    # any the interrupted run assigned. In-flight (unfinished) games are not
    # recovered; the remainder replaces them with fresh games.
    existing = sorted(out_dir.glob("game_*.npz"))
    # A game counts as done only when both its npz AND its .json sidecar exist:
    # the sidecar is the commit marker (the buffer manifest skips sidecar-less
    # shards forever), so a power-cut npz with no sidecar is not a completed
    # game and must not shrink the epoch's remaining count. Sidecar-less npz
    # files are left on disk and still feed the next-key logic below so keys are
    # never reused.
    already_done = sum(1 for p in existing if p.with_suffix(".json").exists())
    remaining = max(games_target - already_done, 0)
    resuming = already_done > 0

    diag_path = ctx.diagnostics_dir / f"hexfield.selfplay.epoch_{epoch:06d}.json"

    if remaining == 0:
        # All of the epoch's self-play is already on disk; skip regeneration.
        # Restart-graceful: a mid-epoch supervisor restart that finds every game
        # committed must NOT clobber the completed epoch's real diagnostic with a
        # zeroed skip record (the epoch-13 incident). Preserve any non-trivial
        # prior diag; otherwise write the skip record marked so the loss is
        # visible downstream.
        driver = ContinuousDriver(
            epoch=epoch, games_target=0, max_plies=sp.max_game_plies, out_dir=out_dir,
            diag_dir=ctx.diagnostics_dir, active_limit=0,
        )
        skip_record = {
            "status": "completed", "epoch": epoch, "elapsed_seconds": 0.0,
            "search_visits": sp.search_visits, "scheduler": {},
            "resumed_existing_games": already_done, **driver.stats(),
        }
        result = _skip_path_result(_load_prior_diag(diag_path), skip_record)
        _atomic_write_json(diag_path, result)
        print(_format_epoch_summary(result), flush=True)
        return result

    # Partial resume (remaining>0): capture the prior diagnostic NOW, before the
    # resumed run's driver.stats() (which covers only the resumed segment)
    # overwrites it. The final write merges prior+resumed so the epoch diag
    # reflects the WHOLE epoch, not just the resumed portion (the epoch-7 case).
    prior_diag = _load_prior_diag(diag_path) if resuming else None

    # Blunder-seed mining. Off by default (fraction<=0 -> empty pool -> the
    # driver's start_games path is bit-identical to current behavior). When on,
    # mine seed prefixes ONCE per epoch from the run's own recent shards; the
    # mine is deterministic given the on-disk data + config, and cheap (a single
    # pass over the recent epochs' npz metadata). Mining is guarded so a failure
    # (or missing data) degrades gracefully to no seeding rather than crashing
    # the epoch.
    blunder_seeds: list = []
    if sp.blunder_seed_fraction > 0.0:
        try:
            blunder_seeds = mine_blunder_seeds(
                ctx.samples_dir,
                current_epoch=epoch,
                recent_epochs=sp.blunder_seed_recent_epochs,
                max_ply=sp.blunder_seed_max_ply,
                surprise_quantile=sp.blunder_seed_surprise_quantile,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"blunder_seeds: mining failed for epoch {epoch} ({exc!r}); "
                "falling back to no seeding",
                RuntimeWarning,
                stacklevel=2,
            )
            blunder_seeds = []
    # Base seed for the per-game blunder decision stream. Same run-seed/epoch
    # mixing as the Rust search base_seed below, on a distinct mix_seed stream,
    # so the seeding decisions are reproducible and stream-isolated.
    blunder_base_seed = (ctx.config.run.seed or 1) * 1_000_003 + epoch

    slots = min(sp.active_games, remaining)
    driver = ContinuousDriver(
        epoch=epoch, games_target=remaining, max_plies=sp.max_game_plies, out_dir=out_dir,
        diag_dir=ctx.diagnostics_dir, active_limit=slots,
        blunder_seeds=blunder_seeds,
        blunder_seed_fraction=sp.blunder_seed_fraction,
        blunder_base_seed=blunder_base_seed,
    )
    # Advance next_key past every npz already on disk -- including sidecar-less
    # (uncommitted) ones, which do not count as done but whose keys must not be
    # reused -- so a restart never overwrites a prior key.
    if existing:
        existing_keys = []
        for p in existing:
            try:
                existing_keys.append(int(p.stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                pass
        driver.next_key = (max(existing_keys) + 1) if existing_keys else epoch * 1_000_000
    tapes = driver.start_games(slots)

    # Per-epoch .hxr game records under <run>/selfplay.
    record_dir = ctx.output_dir / "selfplay"
    record_dir.mkdir(parents=True, exist_ok=True)
    # On resume, write to a separate .hxr path (HexoRecordFile.create overwrites
    # an existing file). Fresh epochs use the canonical path.
    record_path = record_dir / (
        f"epoch_{epoch:06d}_resume{already_done:03d}.hxr" if resuming
        else f"epoch_{epoch:06d}.hxr"
    )
    players = (
        HexoRecordPlayer("hexfield-a", "player0", "Hexfield A"),
        HexoRecordPlayer("hexfield-b", "player1", "Hexfield B"),
    )

    session = _rust.HexfieldMctsSession(max_states=sp.cache_max_states)
    started = time.time()
    driver._t0 = started  # anchor live pos/s to self-play start
    driver._write_live("running")  # initial progress before the first move
    noise_kwargs = {}
    if sp.root_dirichlet_noise_fraction > 0:
        noise_kwargs = dict(
            root_dirichlet_total_alpha=sp.root_dirichlet_total_alpha,
            root_dirichlet_noise_fraction=sp.root_dirichlet_noise_fraction,
        )
    # Context-managed so the .hxr is finalized even if run_continuous raises.
    with HexoRecordFile.create(record_path, api.engine_metadata(), players) as record_file:
        driver.record_file = record_file
        driver._start_writer()
        try:
            scheduler_stats = session.run_continuous(
                [tape.key for tape in tapes],
                tuple(tape.state for tape in tapes),
                evaluator=evaluator,
                on_move=driver,
                visits=sp.search_visits,
                c_puct=sp.c_puct,
                base_seed=(ctx.config.run.seed or 1) * 1_000_003 + epoch,
                virtual_batch_size=sp.virtual_batch_size,
                flush_target=sp.flush_target,
                active_root_limit=sp.active_root_limit,
                temperature_by_ply=cfg.temperature_by_ply(),
                root_policy_temperature=sp.root_policy_temperature,
                root_policy_temperature_early=sp.root_policy_temperature_early or None,
                root_policy_temperature_halflife=sp.root_policy_temperature_halflife or None,
                fpu_reduction=sp.fpu_reduction,
                virtual_loss=sp.virtual_loss,
                widening_policy_mass=sp.widening_policy_mass,
                widening_max_children=sp.widening_max_children,
                widening_min_children=sp.widening_min_children,
                forced_playout_k=sp.forced_playout_k,
                pcr_full_proportion=sp.pcr_full_proportion,
                pcr_fast_visits=sp.pcr_fast_visits,
                pcr_fast_temperature=sp.pcr_fast_temperature,
                policy_init_fraction=sp.policy_init_fraction,
                policy_init_avg_plies=sp.policy_init_avg_plies,
                policy_init_max_plies=sp.policy_init_max_plies,
                policy_init_temperature=sp.policy_init_temperature,
                tss_enabled=sp.tss_enabled,
                # Root FPU reduction. root_fpu_zero_under_noise and search_parity_mode
                # gate how this interacts with root Dirichlet noise (handled in Rust).
                root_fpu_reduction=sp.root_fpu_reduction,
                root_fpu_zero_under_noise=sp.root_fpu_zero_under_noise,
                search_parity_mode=sp.search_parity_mode,
                divergence_overrides=build_divergence_overrides(
                    sp, disabled=(ctx.diagnostics_dir / ML_AUTO_DISABLED_FLAG).exists()
                ),
                # Fast-class (main_8: Gumbel Fast) override map. Equals the base map
                # when no fast_* levers are set, so absent = today's single profile
                # and Rust's divergences_fast == divergences_full (golden invariant).
                fast_divergence_overrides=build_fast_divergence_overrides(
                    sp, disabled=(ctx.diagnostics_dir / ML_AUTO_DISABLED_FLAG).exists()
                ),
                **noise_kwargs,
            )
        except BaseException:
            # run_continuous crashed: still drain and join the writer while the
            # .hxr file is open, so already-queued finished games land on disk
            # instead of dying with the daemon thread. The original exception
            # wins; a secondary writer error is swallowed here (it would have
            # been re-raised by the success-path _stop_writer below).
            try:
                driver._stop_writer()
            except Exception:
                pass
            raise
        # Drain and join the writer while the .hxr file is still open; re-raises
        # any write error so all finished games are on disk before the epoch closes.
        driver._stop_writer()
    driver.record_file = None
    driver._write_live("completed")  # final progress marking the epoch done

    elapsed = time.time() - started
    result = {
        "status": "completed",
        "epoch": epoch,
        "elapsed_seconds": elapsed,
        "search_visits": sp.search_visits,
        "scheduler": {k: v for k, v in scheduler_stats.items() if not isinstance(v, dict)},
        **driver.stats(),
    }
    # Derived rates from the scheduler counters (guarded div-by-zero). Computed
    # on this segment's own scheduler before any merge.
    _derive_scheduler_rates(result)
    # Attach the cuda.Event GPU-busy report (None unless HEXFIELD_PERF_TRACE=1).
    # getattr defaults to None for evaluators without perf_trace_report.
    perf_report = getattr(evaluator, "perf_trace_report", lambda: None)()
    if perf_report is not None:
        result["perf_trace"] = perf_report

    # Merge with the prior (pre-resume) diagnostic so the written epoch diag
    # covers the whole epoch. When not resuming (or no usable prior existed),
    # this is a single-segment pass-through.
    if prior_diag is not None:
        # Treat the prior diag's TOP LEVEL as one segment: if it was itself a
        # merged (repeatedly-resumed) diag, its top level already holds the
        # correct whole-history aggregate, whereas its stored "segments" list is
        # capped and would drop early counters on re-merge. We still surface the
        # prior's raw sub-segments for forensics by prepending them (capped) into
        # the merged "segments" below.
        prior_segment = _segment_payload(prior_diag)
        merged = _merge_epoch_diag([prior_segment, result])
        # Preserve the prior's forensic sub-segments (older crashes) ahead of the
        # two we just merged, capped, so the segment trail is not lost.
        prior_subs = prior_diag.get("segments")
        if isinstance(prior_subs, list) and prior_subs:
            merged["segments"] = (prior_subs + merged["segments"])[-_MAX_SEGMENTS:]
        # Rederive rates from the merged (summed) scheduler counters.
        _derive_scheduler_rates(merged)
        result = merged

    _atomic_write_json(diag_path, result)
    print(_format_epoch_summary(result), flush=True)
    return result
