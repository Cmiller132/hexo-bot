"""hexo_train trainer: per-epoch replay-window selection and single-pass training.

``select_training_samples`` builds the window from an mtime-free
``(generation, game_key)`` manifest: power-law taper, recent-window cut, md5
train/val split, keep_prob subsample, overshoot-skip file selection, and a
train-bucket reuse governor, producing an in-RAM packed columnar
:class:`~shrimp.window.PackedWindow`.

``train_passes`` drains that PackedWindow in a single pass (no within-epoch
repeat):

1. Pre-draw, on the main thread before expansion, a per-row D6 vector from
   ``_aug_seed(run_seed, epoch)`` and a survivor permutation from
   ``_perm_seed(run_seed, epoch)``. No per-row rng call occurs inside the loop.
2. Expand all rows through ``expand_backends.expand_rows`` under the configured
   backend (``serial`` | ``rust`` rayon kernel). Each returns a per-row validity
   mask; off-legal rows are flagged invalid, not dropped. Results are in
   original row order.
3. Filter survivors (validity mask), permute, truncate to ``effective_rows``.
4. Micro-bucket (``pair_budget_microbuckets``).
5. loss / optimizer / AMP / grad-clip.

Backend: ``config.training.expand_backend`` (env ``SHRIMP_EXPAND`` overrides).

``effective_rows`` is threaded from ``select_training_samples`` via
``self._last_select``; a ``train_passes`` call without a prior selection
recomputes it from the window and config.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import numpy as np
import torch
import torch._dynamo  # noqa: F401  (train-compile config + mark_* helpers)

from .batching import (
    PAD_QUANTUM,
    PAIR_BUDGET,
    collate_training,
    pair_budget_microbuckets,
    policy_surprise_weights,
    split_stvalue_columns,
    step_global_denominators,
)
from .buffer_manifest import scan_or_update_manifest
from .config import ShrimpConfig
from .expand_backends import (
    _row_view_to_sample,  # re-exported here; imported from this module by tests
    expand_rows,
)
from .losses import shrimp_loss
from .samples import STV_HORIZONS
from .train_state import ShrimpTrainState
from .window import (
    PackedWindow,
    _select_files_for_rows,
    _split_by_md5,
    build_window_split,
    compute_katago_window_rows,
    keep_prob as _keep_prob,
    select_recent_window,
)

# D6 augmentation cardinality (geometry.apply_d6 accepts 0-11).
D6_SIZE = 12


def _aug_seed(run_seed: int, epoch: int) -> int:
    """Deterministic per-(run, epoch) seed for the D6 augmentation draw.

    A fold of ``(run_seed, epoch)``. All D6 randomness is drawn from this seed
    on the main thread before expansion.
    """
    return (int(run_seed) * 1_000_003 + int(epoch) * 9_176 + 1) & 0x7FFFFFFF


def _perm_seed(run_seed: int, epoch: int) -> int:
    """Deterministic per-(run, epoch) seed for the survivor permutation. A
    distinct fold from :func:`_aug_seed`, so the permutation stream is
    independent of the D6 stream. Both are pure functions of ``(run_seed,
    epoch)``."""
    return (int(run_seed) * 2_654_435_761 + int(epoch) * 40_503 + 7) & 0x7FFFFFFF


def _atomic_write_json(path, payload: dict[str, Any]) -> None:
    """Serialize ``payload`` to ``path`` via tmp + ``os.replace``.

    The host takes real power cuts (see tests/test_shrimp_durability.py); an
    interrupted plain ``write_text`` can leave a torn diagnostics json. Writing to
    ``<path>.tmp`` then atomically renaming guarantees a reader sees either the old
    file or the complete new one, never a partial. Same-directory tmp so the
    rename stays on one filesystem.
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


class ShrimpTrainer:
    def __init__(self, *, model, config: ShrimpConfig, optimizer):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.device = torch.device(config.device)
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        self.global_step = 0
        # EMA of the pre-clip grad-norm for adaptive grad-clip. Cross-epoch, not
        # checkpointed; seeded from the first observed norm and updated every step
        # (including warmup).
        self._grad_norm_ema: float | None = None
        # Param-group partition for per-group grad-norm logging.
        self._grad_norm_groups = self._build_grad_norm_groups()
        # Train-bucket governor + window bookkeeping. Serialized into the
        # checkpoint meta by the saver and restored by the loader on resume;
        # starts fresh here.
        self.train_state = ShrimpTrainState()
        # Per-epoch selection bookkeeping stashed by select_training_samples and
        # read back by train_passes on the same trainer instance. Carries
        # effective_rows / window_start / reuse_ratio / train_bucket_level.
        self._last_select: dict[int, dict[str, Any]] = {}
        # Training pair budget (SHRIMP_TRAIN_PAIR_BUDGET, default batching's
        # PAIR_BUDGET=2e7). PAIR_BUDGET was sized for the materialized fp16
        # (B, 4, S, S) bias transient; with SHRIMP_TRAIN_FLEX the attention
        # never builds an S^2 tensor, so a larger budget packs more rows per
        # microbucket (fewer, fatter fwd+bwd launches per optimizer step) with
        # gradient-identical math (step-global denominators make the step
        # gradient independent of the bucket split, modulo fp reassociation).
        self._pair_budget = float(
            os.environ.get("SHRIMP_TRAIN_PAIR_BUDGET", 0) or PAIR_BUDGET
        )
        # Compiled training forward (SHRIMP_TRAIN_COMPILE=1, CUDA only).
        # The batch dim is marked dynamic and Npad is marked STATIC per call:
        # microbucket row counts vary freely (one symbolic graph covers them)
        # while Npad only takes the few PAD_QUANTUM multiples, each getting its
        # own specialization. Marking BOTH dims dynamic trips Inductor's
        # CantSplit on the attention head-merge reshape (B*(Npad+8) with two
        # free symbols) — the same failure the serve path dodges by other means.
        self._train_compile = (
            self.device.type == "cuda"
            and os.environ.get("SHRIMP_TRAIN_COMPILE") == "1"
        )
        self._compiled_train_fwd = None
        if self._train_compile:
            try:
                torch._dynamo.config.cache_size_limit = max(
                    64, torch._dynamo.config.cache_size_limit
                )
                self._compiled_train_fwd = torch.compile(self.model.forward)
            except Exception:
                self._compiled_train_fwd = None

    def close(self) -> None:
        """Run-end teardown hook.

        Called by the pipeline's run-end teardown. The trainer now expands rows
        via the serial or Rust backend (no owned process pool), so there is
        nothing to release; kept for the pipeline lifecycle contract.
        """
        return

    def _build_grad_norm_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """Partition model params into trunk_conv / trunk_attn / heads.

        Used for per-group pre-clip grad-norm logging. ``stem*`` / ``conv_blocks*``
        -> trunk_conv; ``attn_blocks*``, ``tokens``, and ``bias_tables*`` ->
        trunk_attn; everything else -> heads.
        """
        groups: dict[str, list[torch.nn.Parameter]] = {
            "trunk_conv": [],
            "trunk_attn": [],
            "heads": [],
        }
        for name, p in self.model.named_parameters():
            if name.startswith("stem") or name.startswith("conv_blocks"):
                groups["trunk_conv"].append(p)
            elif (
                name.startswith("attn_blocks")
                or name == "tokens"
                or name.startswith("bias_tables")
            ):
                groups["trunk_attn"].append(p)
            else:
                groups["heads"].append(p)
        return groups

    def _group_grad_norms(self) -> dict[str, float]:
        """This step's per-group L2 grad-norm (pre-clip).

        Computed after ``unscale_`` and before ``clip_grad_norm_``. The caller
        merges the result into the running totals only on finite steps.
        """
        out: dict[str, float] = {}
        for gname, params in self._grad_norm_groups.items():
            sq = 0.0
            for p in params:
                if p.grad is not None:
                    sq += float(p.grad.detach().norm(2).item()) ** 2
            out[gname] = sq ** 0.5
        return out

    def _update_train_bucket(self, total_rows: int, window_start: int) -> None:
        """Accrue / clamp the train-bucket reuse governor.

        ``total_rows`` is the monotone ``cumulative_rows_ever`` from the manifest,
        not the live total.

        * ``cap = max(max_train_bucket_size, train_samples_per_epoch)``.
        * Each new row credits the bucket by ``max_train_bucket_per_new_data``,
          clamped at ``cap``, advancing ``level_at_row`` to ``total_rows``.
        * A decrease in ``total_rows`` re-bases the watermark, zeroes
          ``steps_since_last_reload``, and re-clamps the level.
        """
        cap = max(
            float(self.config.training.max_train_bucket_size),
            float(self.config.training.train_samples_per_epoch),
        )
        if total_rows > self.train_state.train_bucket_level_at_row:
            new_rows = total_rows - self.train_state.train_bucket_level_at_row
            self.train_state.train_bucket_level = min(
                cap,
                self.train_state.train_bucket_level
                + new_rows * self.config.training.max_train_bucket_per_new_data,
            )
            self.train_state.train_bucket_level_at_row = int(total_rows)
        elif total_rows < self.train_state.train_bucket_level_at_row:
            self.train_state.train_bucket_level_at_row = int(total_rows)
            self.train_state.train_steps_since_last_reload = 0
            self.train_state.train_bucket_level = min(self.train_state.train_bucket_level, cap)
        self.train_state.total_num_data_rows = int(total_rows)
        self.train_state.window_start_data_row_idx = int(window_start)

    def select_training_samples(self, *, ctx, components, epoch: int) -> dict[str, Any]:
        """Window selection into an in-RAM ``PackedWindow`` (no disk re-shard).

        1. ``scan_or_update_manifest`` -> the mtime-free ``(generation,game_key)``
           ordered shard manifest. Live ``total_rows`` drives window selection;
           the monotone ``cumulative_rows_ever`` drives the governor.
        2. ``compute_katago_window_rows`` power-law taper, clamped to
           ``max(_, min_rows)``; ``select_recent_window`` newest->oldest
           whole-shard cut.
        3. ``_update_train_bucket(cumulative_rows_ever, window_start)`` accrues on
           the monotone counter; ``window_start = max(0, total_rows - used)``.
        4. ``_split_by_md5`` per-file train/val partition.
        5. ``_select_files_for_rows`` overshoot-skip selection capped at
           ``train_samples_per_epoch`` (``no_repeat_files`` applied first; default
           off).
        6. ``effective_rows = min(requested, selected)``; then either the bucket
           throttle (``train_bucket_limited``) or a debit by ``effective_rows``.
        7. ``build_window_split`` keep_prob-subsamples and concats the survivors
           into one ``PackedWindow`` -> ``components.shared.sample_window``.

        Determinism: ``np.random.default_rng(seed + epoch)`` drives keep_prob; a
        separate ``np.random.default_rng(seed + epoch*65537)`` drives file
        selection. ``seed = (ctx.config.run.seed or 0)``.
        """
        cfg = self.config.training
        seed = int(ctx.config.run.seed or 0)
        # Wall-clock for the whole selection phase (manifest scan + window build).
        # time.monotonic is immune to wall-clock steps; recorded in the diag and
        # stashed so train_passes can print it in the epoch summary line.
        select_t0 = time.monotonic()

        # (1) manifest: live total drives window selection; the monotone counter
        # drives the governor.
        manifest = scan_or_update_manifest(ctx.samples_dir)
        entries = manifest.entries
        total_rows = int(manifest.total_rows)
        cumulative_rows_ever = int(manifest.cumulative_rows_ever)
        # New rows credited to the governor since the last accrual (for the
        # diagnostic reuse_ratio); captured before _update_train_bucket mutates it.
        prev_level_at_row = int(self.train_state.train_bucket_level_at_row)
        new_rows_this_epoch = max(0, cumulative_rows_ever - prev_level_at_row)

        # (2) taper window + recent-window cut.
        desired = compute_katago_window_rows(
            total_rows,
            min_rows=cfg.shuffle_min_rows,
            expand_window_per_row=cfg.shuffle_expand_window_per_row,
            taper_window_exponent=cfg.shuffle_taper_window_exponent,
            taper_window_scale=cfg.shuffle_taper_window_scale,
        )
        desired = max(int(desired), int(cfg.shuffle_min_rows))
        selected_window, used = select_recent_window(entries, desired)
        window_start = max(0, total_rows - used)

        # (3) governor accrual on the monotone counter.
        self._update_train_bucket(cumulative_rows_ever, window_start)

        def _skip(status: str, reason: str, **extra) -> dict[str, Any]:
            components.shared.sample_window = PackedWindow.empty()
            base = {
                "status": status,
                "epoch": epoch,
                "reason": reason,
                "total_rows": cumulative_rows_ever,
                "live_total_rows": total_rows,
                "desired_rows": int(desired),
                "used_rows": int(used),
                "window_start": window_start,
                "train_bucket_level": float(self.train_state.train_bucket_level),
            }
            base.update(extra)
            # Stash for the consumer. An empty/limited selection trains nothing,
            # so effective_rows is 0 and reuse_ratio is carried through.
            self._last_select[epoch] = {
                "effective_rows": int(base.get("effective_rows", 0) or 0),
                "window_start": int(window_start),
                "reuse_ratio": float(base.get("reuse_ratio", 0.0) or 0.0),
                "train_bucket_level": float(self.train_state.train_bucket_level),
            }
            self._write_select_diag(ctx, epoch, base)
            return base

        if not selected_window:
            return _skip("skipped", "no files selected for the window",
                         keep_prob=1.0, effective_rows=0, window_rows=0, reuse_ratio=0.0)

        kp = _keep_prob(used, int(cfg.shuffle_keep_target_rows))

        # (4) md5 train/val split (default validation_fraction=0.0 -> all-train).
        train_entries, _val_entries = _split_by_md5(
            selected_window, validation_fraction=float(cfg.validation_fraction)
        )
        if not train_entries:
            return _skip("skipped", "no train files after md5 validation split",
                         keep_prob=kp, effective_rows=0, window_rows=0, reuse_ratio=0.0)

        # (5) overshoot-skip selection, capped at train_samples_per_epoch.
        # no_repeat_files (default off) filters first.
        candidate_entries = train_entries
        if cfg.no_repeat_files:
            candidate_entries = [
                e for e in train_entries if str(e.rel_path) not in self.train_state.data_files_used
            ]
        requested_rows = int(cfg.train_samples_per_epoch)
        # build_window_split then per-row Bernoulli(kp)-thins the SELECTED files, so
        # the epoch trains ~selected_rows*kp rows. Inflate the file-selection request
        # by 1/kp so post-thinning survivors land near requested_rows; kp==1.0 leaves
        # the request (and the whole path) bit-identical.
        select_request = requested_rows if kp >= 1.0 else int(math.ceil(requested_rows / kp))
        sel_rng = np.random.default_rng(seed + epoch * 65_537)
        selected_files, selected_rows = _select_files_for_rows(
            candidate_entries, select_request, sel_rng
        )
        if selected_rows <= 0:
            return _skip("skipped", "no new training files",
                         keep_prob=kp, effective_rows=0, window_rows=0, reuse_ratio=0.0,
                         requested=requested_rows, select_request=select_request,
                         selected_rows=selected_rows)

        # (6) effective_rows = the rows that will actually be trained in expectation:
        # min(requested, selected*kp). Debit/accounting match the post-thin window,
        # not the inflated pre-thin selection. kp==1.0 -> min(requested, selected)
        # (bit-identical to the pre-fix accounting).
        effective_rows = min(requested_rows, int(round(selected_rows * kp)))
        if self.train_state.train_bucket_level + 1.0e-9 < effective_rows:
            return _skip("train_bucket_limited", "train_bucket_limited",
                         keep_prob=kp, effective_rows=int(effective_rows), window_rows=0,
                         reuse_ratio=effective_rows / max(1, new_rows_this_epoch),
                         requested=requested_rows, select_request=select_request,
                         selected_rows=selected_rows)
        # Debit effective_rows from the bucket at selection time; a later short
        # pass does not refund. steps_since_last_reload increments each selection.
        self.train_state.train_bucket_level = max(
            0.0, self.train_state.train_bucket_level - effective_rows
        )
        self.train_state.train_steps_since_last_reload += 1

        # (7) build the packed in-RAM window: per-row keep_prob subsample + concat,
        # consumed via a single per-epoch rng in (generation, game_key) order.
        # ``window_diag`` collects load/skip telemetry (shards_skipped etc.) that
        # build_window_split only surfaced via a RuntimeWarning before.
        keep_rng = np.random.default_rng(seed + epoch)
        window_diag: dict[str, Any] = {}
        window = build_window_split(
            selected_files, keep_prob=kp, rng=keep_rng, samples_dir=ctx.samples_dir,
            diag=window_diag,
        )
        components.shared.sample_window = window

        # Window depth telemetry: how many distinct producing epochs the selected
        # files span, and what fraction come from the newest epoch. The selected
        # entries carry ``generation``; both are O(len(selected_files)) and run at
        # the phase boundary (negligible cost).
        sel_gens = [int(e.generation) for e in selected_files]
        window_epoch_span: dict[str, int] = {}
        shards_from_latest_epoch = 0.0
        if sel_gens:
            gmin, gmax = min(sel_gens), max(sel_gens)
            window_epoch_span = {"min": gmin, "max": gmax, "epochs": gmax - gmin + 1}
            shards_from_latest_epoch = sum(1 for g in sel_gens if g == gmax) / len(sel_gens)

        result = {
            "status": "completed",
            "epoch": epoch,
            "total_rows": cumulative_rows_ever,
            "live_total_rows": total_rows,
            "desired_rows": int(desired),
            "used_rows": int(used),
            "keep_prob": float(kp),
            "effective_rows": int(effective_rows),
            "window_rows": int(window.n),
            "window_start": window_start,
            "train_bucket_level": float(self.train_state.train_bucket_level),
            "reuse_ratio": effective_rows / max(1, new_rows_this_epoch),
            "selected_files": len(selected_files),
            "select_request": int(select_request),
            "selected_rows": int(selected_rows),
            # Load/skip telemetry from build_window_split (torn-shard visibility).
            "shards_skipped": int(window_diag.get("shards_skipped", 0)),
            "skipped_paths": window_diag.get("skipped_paths", []),
            "rows_loaded": int(window_diag.get("rows_loaded", 0)),
            "rows_post_thin": int(window_diag.get("rows_post_thin", int(window.n))),
            # Window-depth telemetry (how many epochs deep the window reaches).
            "window_epoch_span": window_epoch_span,
            "shards_from_latest_epoch": float(shards_from_latest_epoch),
            # Selection-phase wall-clock (seconds).
            "select_seconds": round(time.monotonic() - select_t0, 1),
        }
        # Stash effective_rows / window_start / reuse_ratio / train_bucket_level
        # for train_passes, threaded via the trainer instance (SharedComponents
        # only carries the PackedWindow). select_seconds / shards_skipped /
        # window_epoch_span are carried for the train-phase summary line.
        self._last_select[epoch] = {
            "effective_rows": int(effective_rows),
            "window_start": int(window_start),
            "reuse_ratio": float(result["reuse_ratio"]),
            "train_bucket_level": float(self.train_state.train_bucket_level),
            "select_seconds": float(result["select_seconds"]),
            "shards_skipped": int(result["shards_skipped"]),
            "window_epoch_span": window_epoch_span,
            "keep_prob": float(kp),
        }
        self._write_select_diag(ctx, epoch, result)
        return result

    def _write_select_diag(self, ctx, epoch: int, result: dict[str, Any]) -> None:
        """Persist the per-epoch selection summary next to the training diag
        (best-effort; never raises into the dispatch path)."""
        try:
            diag_dir = getattr(ctx, "diagnostics_dir", None)
            if diag_dir is None:
                return
            path = diag_dir / f"shrimp.select.epoch_{epoch:06d}.json"
            _atomic_write_json(path, result)
        except OSError:
            pass

    def _effective_rows_for(self, window: PackedWindow, epoch: int) -> int:
        """Row cap for this epoch's single pass.

        Returns the ``effective_rows`` value ``select_training_samples`` stashed
        for ``epoch`` (the bucket-debited count). When ``train_passes`` is called
        without a prior selection, returns ``min(window.n,
        train_samples_per_epoch)``.
        """
        stashed = self._last_select.get(epoch)
        if stashed is not None:
            return int(stashed["effective_rows"])
        return min(int(window.n), int(self.config.training.train_samples_per_epoch))

    def train_passes(self, *, passes, sample_window, sample_symmetries, ctx, components, epoch) -> dict[str, Any]:
        # D6 augmentation is drawn from the window itself; the framework's
        # sample_symmetries argument is ignored.
        _ = sample_symmetries
        window = sample_window if isinstance(sample_window, PackedWindow) else None
        if window is None or window.n <= 0:
            return {"status": "skipped", "epoch": epoch, "reason": "empty sample window"}

        seed = int(ctx.config.run.seed or 0)
        # Pre-draw all randomness on the main thread: (a) a per-row D6 vector in
        # window row order, and (b) the survivor permutation (drawn below). Both
        # are pure functions of (seed, epoch), consumed positionally.
        d6 = np.random.default_rng(_aug_seed(seed, epoch)).integers(
            0, D6_SIZE, size=int(window.n), dtype=np.int64
        )

        self.model.train().to(self.device)
        # Move any optimizer state tensors to the model's device before the first
        # step(). No-op once state already lives on the device.
        for _st in self.optimizer.state.values():
            for _k, _v in _st.items():
                if isinstance(_v, torch.Tensor) and _v.device != self.device:
                    _st[_k] = _v.to(self.device)
        batch_rows = self.config.training.batch_rows
        comp_totals: dict[str, float] = {}
        grad_norms: list[float] = []
        clip_values: list[float] = []
        group_norm_totals: dict[str, float] = {}
        group_norm_steps = 0
        steps = 0
        # Per-row surprise-weight telemetry, aggregated at the phase boundary from
        # the per-batch weight vectors already computed below (no hot-path cost).
        surprise_weight_sum = 0.0
        surprise_weight_count = 0
        surprise_weight_max = 0.0
        surprise_weight_clamped_count = 0
        surprise_clamp_thresh = float(self.config.training.policy_surprise_max_weight)
        started = time.time()
        # Monotonic wall-clock for the train phase (train_seconds); ``started``
        # above stays for the existing ``seconds`` field (unchanged).
        train_t0 = time.monotonic()
        # When SHRIMP_SUPPORT_RADIUS < 8, tolerate (flag invalid, skip) replay
        # samples whose policy targets fall off the smaller legal set. At the
        # default radius, off-legal rows hard-error instead.
        tolerate_off_legal = int(os.environ.get("SHRIMP_SUPPORT_RADIUS", "8")) < 8

        # (1) Expand all window rows under their pre-drawn D6 via the configured
        # backend (serial | rust). The backend returns a per-row ExpandedRow list
        # aligned to range(window.n) plus a `valid` mask; off-legal rows are
        # flagged invalid, not dropped.
        backend = str(os.environ.get("SHRIMP_EXPAND", self.config.training.expand_backend))
        expanded_rows, valid = expand_rows(
            window,
            None,  # expand all rows; survivor filter + truncation happen below
            d6,
            STV_HORIZONS,
            tolerate_off_legal=tolerate_off_legal,
            backend=backend,
        )

        # (2) Filter survivors on the main thread using the validity mask.
        survivors: list = [row for row, ok in zip(expanded_rows, valid) if ok]
        rows_skipped_off_legal = int((~np.asarray(valid, dtype=bool)).sum())

        # (3) Permute the survivor index (drawn over the post-skip set), then
        # (4) truncate to effective_rows: a single pass, no within-epoch repeat,
        # capped at the bucket-debited effective_rows.
        n_surv = len(survivors)
        perm = np.random.default_rng(_perm_seed(seed, epoch)).permutation(n_surv)
        effective_rows = self._effective_rows_for(window, epoch)
        keep = perm[: max(0, int(effective_rows))]
        ordered_rows = [survivors[int(j)] for j in keep]

        # (5) Micro-bucket via pair_budget_microbuckets. One optimizer step per
        # nominal batch of ``batch_rows`` survivors; the (6) loss / optimizer /
        # AMP / grad-clip block follows below.
        for start in range(0, len(ordered_rows), batch_rows):
            expanded = ordered_rows[start : start + batch_rows]
            if not expanded:
                continue
            tcfg = self.config.training
            # Step-global denominators (mean-over-rows / -cells over the nominal
            # batch, including cell_q and the policy-surprise self-CE weight sum).
            # The matching per-row self-CE weights are computed once here over the
            # same nominal batch and keyed by id(row), so collate packs the correct
            # value even when pair_budget_microbuckets reorders rows within the
            # batch.
            denoms = step_global_denominators(
                expanded, STV_HORIZONS,
                policy_surprise_uniform_fraction=tcfg.policy_surprise_uniform_fraction,
                policy_surprise_max_weight=tcfg.policy_surprise_max_weight,
            )
            # Compute the self-policy surprise weights over the full subset only
            # (policy_valid != 0), matching the full-row weight-sum denominator in
            # step_global_denominators. Rows with policy_valid == 0 get weight 0
            # (they are policy-masked, and losses also multiply by policy_valid).
            full_rows = [r for r in expanded if getattr(r, "policy_valid", 1.0) != 0.0]
            full_weights, _ = policy_surprise_weights(
                [row.policy_surprise for row in full_rows],
                tcfg.policy_surprise_uniform_fraction,
                tcfg.policy_surprise_max_weight,
            )
            weight_by_row = {id(r): 0.0 for r in expanded}
            weight_by_row.update({id(r): w for r, w in zip(full_rows, full_weights)})
            # Telemetry: aggregate the per-row surprise weights over the policy-valid
            # rows (the ones that carry a nontrivial weight). Cheap running reduce;
            # matches the weights actually applied by collate_training. clamped ==
            # weights that hit policy_surprise_max_weight.
            if full_weights:
                surprise_weight_sum += float(sum(full_weights))
                surprise_weight_count += len(full_weights)
                surprise_weight_max = max(surprise_weight_max, max(full_weights))
                surprise_weight_clamped_count += sum(
                    1 for w in full_weights if w >= surprise_clamp_thresh - 1e-9
                )
            self.optimizer.zero_grad(set_to_none=True)
            for bucket in pair_budget_microbuckets(
                expanded, budget=self._pair_budget, quantize=PAD_QUANTUM
            ):
                # Pad to the same PAD_QUANTUM the budget split assumed.
                pad_to = -(-max(r.support.num_nodes for r in bucket) // PAD_QUANTUM) * PAD_QUANTUM
                batch = split_stvalue_columns(
                    collate_training(
                        bucket, pad_to=pad_to,
                        row_weights=[weight_by_row[id(r)] for r in bucket],
                    ),
                    STV_HORIZONS,
                )
                batch = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                fwd = self.model
                if self._compiled_train_fwd is not None:
                    # B symbolic (hint, not constraint: the flex graph break
                    # specializes B in a sub-graph, which a hard mark_dynamic
                    # turns into a ConstraintViolation), Npad pinned static:
                    # one graph per PAD_QUANTUM multiple.
                    for t in (batch["feats"], batch["nbr"], batch["mask"], batch["coords"]):
                        torch._dynamo.maybe_mark_dynamic(t, 0)
                        torch._dynamo.mark_static(t, 1)
                    fwd = self._compiled_train_fwd
                with torch.autocast(
                    device_type=self.device.type, dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    out = fwd(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
                loss, comps = shrimp_loss(
                    out, batch,
                    policy_weight=tcfg.policy_weight,
                    value_weight=tcfg.value_weight,
                    opp_policy_weight=tcfg.opp_policy_weight,
                    soft_policy_weight=tcfg.soft_policy_weight,
                    short_term_value_weight=tcfg.short_term_value_weight,
                    moves_left_weight=tcfg.moves_left_weight,
                    q_head_weight=tcfg.q_head_weight,
                    policy_target=tcfg.policy_target,
                    denominators=denoms,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite loss at epoch {epoch} step {steps}: "
                        f"{ {k: float(v) for k, v in comps.items()} }"
                    )
                self.scaler.scale(loss).backward()
                for key, val in comps.items():
                    comp_totals[key] = comp_totals.get(key, 0.0) + float(val.detach())
            self.scaler.unscale_(self.optimizer)
            # Adaptive grad-clip: during warmup, when adaptive_clip is off, or
            # before any EMA exists, clip at the static grad_clip; otherwise clip
            # at clip_c * EMA(pre-clip grad-norm). Per-group norms are accumulated
            # before clipping to reflect pre-clip magnitudes.
            tcfg = self.config.training
            if (
                not tcfg.adaptive_clip
                or self.global_step < tcfg.clip_warmup_steps
                or self._grad_norm_ema is None
            ):
                clip_value = float(tcfg.grad_clip)
            else:
                clip_value = float(tcfg.clip_c) * float(self._grad_norm_ema)
            step_group_norms = self._group_grad_norms()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            if torch.isfinite(norm):
                grad_norms.append(float(norm))
                clip_values.append(clip_value)
                group_norm_steps += 1
                for _g, _v in step_group_norms.items():
                    group_norm_totals[_g] = group_norm_totals.get(_g, 0.0) + _v
                # Update the pre-clip-norm EMA every step, warmup included.
                d = float(tcfg.clip_ema_decay)
                self._grad_norm_ema = (
                    float(norm)
                    if self._grad_norm_ema is None
                    else d * self._grad_norm_ema + (1.0 - d) * float(norm)
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            steps += 1
            self.global_step += 1

        trained_rows = len(ordered_rows)
        self.train_state.global_step_samples += trained_rows

        if steps <= 0:
            return {
                "status": "skipped",
                "epoch": epoch,
                "reason": "no optimizer steps (all rows skipped off-legal or empty)",
                "window_rows": int(window.n),
                "rows_skipped_off_legal": int(rows_skipped_off_legal),
            }

        stashed = self._last_select.get(epoch, {})
        grads = np.asarray(grad_norms or [0.0])
        # clip_fraction compared against the effective per-step clip threshold,
        # aligned positionally to the recorded norms.
        clips = np.asarray(clip_values or [self.config.training.grad_clip])
        n_clip = min(len(grads), len(clips))
        clip_fraction = float((grads[:n_clip] > clips[:n_clip]).mean()) if n_clip else 0.0
        result = {
            "status": "completed",
            "epoch": epoch,
            # Single pass, no within-epoch repeat; the generic ``passes`` request
            # is reported but not multiplied.
            "passes": 1,
            "generic_passes_requested": passes,
            "window_rows": int(window.n),
            "trained_rows": int(trained_rows),
            "steps": steps,
            "seconds": round(time.time() - started, 1),
            **{f"loss_{k}": v / max(steps, 1) for k, v in comp_totals.items()},
            "grad_norm_mean": float(grads.mean()),
            "grad_norm_p95": float(np.percentile(grads, 95)),
            "clip_fraction": clip_fraction,
            "clip_value_mean": float(clips.mean()),
            "grad_norm_ema": float(self._grad_norm_ema) if self._grad_norm_ema is not None else 0.0,
            **{
                f"grad_norm_{g}": float(group_norm_totals.get(g, 0.0) / max(group_norm_steps, 1))
                for g in ("trunk_conv", "trunk_attn", "heads")
            },
            "amp_scale": float(self.scaler.get_scale()) if self.device.type == "cuda" else None,
            # Replay-buffer diagnostics.
            "reuse_ratio": float(stashed.get("reuse_ratio", 0.0)),
            "train_bucket_level": float(
                stashed.get("train_bucket_level", self.train_state.train_bucket_level)
            ),
            "train_steps_since_last_reload": int(self.train_state.train_steps_since_last_reload),
            "rows_skipped_off_legal": int(rows_skipped_off_legal),
            # Per-row surprise-weight telemetry (mean over policy-valid rows == ~1
            # when no clamp fires; max and clamped-count expose weight skew).
            "surprise_weight_mean": (
                surprise_weight_sum / surprise_weight_count if surprise_weight_count else 0.0
            ),
            "surprise_weight_max": float(surprise_weight_max),
            "surprise_weight_clamped_count": int(surprise_weight_clamped_count),
            # Per-epoch wall-clock split (select phase timed in
            # select_training_samples and stashed; train phase timed here).
            "select_seconds": float(stashed.get("select_seconds", 0.0)),
            "train_seconds": round(time.monotonic() - train_t0, 1),
        }
        diag_path = ctx.diagnostics_dir / f"shrimp.training.epoch_{epoch:06d}.json"
        _atomic_write_json(diag_path, result)

        # Human-readable one-line summary to stdout (lands in the supervisor's
        # train.out). Every field is guarded for absence so a partial epoch still
        # prints something useful.
        span = stashed.get("window_epoch_span") or {}
        print(
            f"train epoch {epoch}: "
            f"{result['trained_rows']}/{result['window_rows']} rows "
            f"(reuse {result['reuse_ratio']:.1f}, "
            f"kp {float(stashed.get('keep_prob', 1.0)):.2f}) | "
            f"pol {result.get('loss_policy', 0.0):.2f} "
            f"soft {result.get('loss_soft_policy', 0.0):.2f} "
            f"val {result.get('loss_value', 0.0):.2f} | "
            f"surprise mean {result['surprise_weight_mean']:.1f} "
            f"max {result['surprise_weight_max']:.1f} | "
            f"select {result['select_seconds']:.0f}s "
            f"train {result['train_seconds']:.0f}s | "
            f"window {int(span.get('epochs', 0))} epochs, "
            f"{int(stashed.get('shards_skipped', 0))} skipped",
            flush=True,
        )
        return result
