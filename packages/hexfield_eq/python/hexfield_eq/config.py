"""hexfield run configuration (the [model.config] sections of a run toml).

Dataclass defaults are overridden by a run's toml; read the run toml for the
authoritative values. Defaults have moves-left utility ON and the root-policy
temperature knobs off (root_policy_temperature 1.0, no ramp; no FPU
noise-zeroing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .constants import SOFT_POLICY_WEIGHT


@dataclass(frozen=True)
class SelfplayConfig:
    search_visits: int = 512
    pcr_full_proportion: float = 0.33
    pcr_fast_visits: int = 128
    # Play temperature for the Fast (value-only) PCR class in the continuous
    # selfplay driver ONLY. Default 0.0 = exactly current behavior (Fast moves
    # take the greedy LCB pick, T==0). At T>0 the played move is a temperature
    # sample (exponent 1/T) of the guard-filtered delta-visit histogram, and the
    # LCB pick + ml_final_pick no longer fire for Fast moves (they require T==0).
    # Init stays 0.0/prior-sampled; Full is unaffected. Eval/lockstep/parity
    # paths do not read this lever.
    pcr_fast_temperature: float = 0.0
    active_games: int = 128
    c_puct: float = 1.5
    virtual_batch_size: int = 4
    flush_target: int = 256
    active_root_limit: int = 256
    root_dirichlet_total_alpha: float = 10.83
    root_dirichlet_noise_fraction: float = 0.25
    # --- Dynamic c_puct + LCB -------------------------------------------------
    # Read by resolve_divergences. c_for(N) = c_puct + c_scale*ln((N+c_base)
    # /c_base); visit_scaled_c_puct gates the log term; lcb_z is the LCB z-score.
    c_scale: float = 0.45
    c_base: float = 500.0
    visit_scaled_c_puct: bool = True
    lcb_z: float = 1.6
    # --- Search divergences ---------------------------------------------------
    # Six boolean search behaviors emitted from config and applied on top of the
    # base divergences selected by search_parity_mode (see
    # build_divergence_overrides). Each is individually controllable.
    nucleus_f64: bool = True
    new_child_fpu: bool = True
    lazy_widening: bool = True
    clean_root_prior_cache: bool = True
    dirichlet_shaped: bool = True
    pruned_dynamic_cpuct: bool = True
    # Root-policy temperature knobs. Defaults leave temperature at 1.0 with no
    # ramp (early value 0.0, halflife 0.0).
    root_policy_temperature: float = 1.0
    root_policy_temperature_early: float = 0.0
    root_policy_temperature_halflife: float = 0.0
    # root_fpu_reduction is the root FPU reduction (default 0.0).
    # root_fpu_zero_under_noise gates a noise-conditioned FPU-zeroing branch used
    # only by the parity path; default False leaves it inactive.
    root_fpu_reduction: float = 0.0
    root_fpu_zero_under_noise: bool = False
    fpu_reduction: float = 0.2
    virtual_loss: float = 1.0
    widening_policy_mass: float = 0.95
    widening_max_children: int = 96
    widening_min_children: int = 2
    forced_playout_k: float = 2.0
    policy_init_fraction: float = 0.25
    policy_init_avg_plies: float = 4.0
    policy_init_max_plies: int = 8
    policy_init_temperature: float = 1.4
    temperature: float = 1.0
    temperature_floor: float = 0.1
    temperature_halflife_plies: float = 30.0
    max_game_plies: int = 512
    tss_enabled: bool = True
    # Deep-TSS solver knobs (ported for main_5 serve). All default OFF so a
    # checkpoint config that omits them (e.g. main_2 ep70) keeps the light
    # tactical-guard-only path, byte-identical to before this port. Consumed by
    # build_divergence_overrides -> Rust divergence_overrides; defaults mirror the
    # trainer SelfplayConfig. The serve profile (configs/hexfield_eq_main_5.toml)
    # turns them on (mode 3, async, park, cap 500, 6 threads, all-leaves).
    tss_interior_guard: bool = False
    tss_solver_mode: int = 0
    tss_solver_node_cap: int = 2000
    tss_solver_sample_16: int = 16
    tss_solver_root_guard: bool = False
    tss_solver_async: bool = False
    tss_solver_async_threads: int = 8
    tss_solver_async_threads_max: int = 0
    tss_solver_park: bool = False
    tss_solver_all_leaves: bool = False
    tss_solver_park_timeout_ms: int = 100
    tss_solver_async_inline_16: int = 0
    tss_zone: bool = False
    tss_zone_stale_filter: bool = False
    tss_zone_count2: bool = False
    tss_pair_commutation: bool = False
    tss_solver_horizon: int = 16
    tss_solver_dual_pass: bool = False
    tss_solver_loss_reserve_nodes: int = 0
    tss_solver_group2: bool = False
    tss_solver_j2near: bool = False
    tss_solver_horizon_ladder: bool = False
    search_parity_mode: bool = False
    # Moves-left utility. Defaults: enabled, two-sided, with the final-move
    # tie-break. Passed to Rust as divergence_overrides. moves_left_utility=False
    # (or search_parity_mode=True) selects the no-MLH baseline. ml_auto_disabled
    # (config field) and the run-dir ml_auto_disabled.flag both force the lever
    # off; see build_divergence_overrides for the gating.
    moves_left_utility: bool = True
    ml_weight: float = 0.03
    ml_scale: float = 32.0
    ml_q_gate: float = 0.6
    ml_two_sided: bool = True
    ml_final_pick: bool = True
    ml_final_pick_band: float = 0.05
    ml_auto_disabled: bool = False
    cache_max_states: int = 262_144
    # --- Gumbel AlphaZero (Danihelka et al. 2022) -----------------------------
    # All flags default OFF; when off the corresponding Rust paths are inactive.
    # Three mechanisms, each gated by its own enable flag:
    #   root  : gumbel_root_enabled (Gumbel-Top-k sampling of m candidates)
    #           + gumbel_sequential_halving (root-only SH visit allocation).
    #   select: gumbel_nonroot_select (deterministic argmax[logits+σ(Q)]).
    #   target: gumbel_target_enabled (π'=softmax(logits+σ(completedQ))).
    # σ(q)=(c_visit+max_b N(b))·c_scale·q ; gumbel_m = candidate count (clamped to
    # n_legal at the root); gumbel_target_min_visits = target support floor.
    # Size gumbel_m for the FULL selfplay visit budget: under SH the tree
    # budget-calibrates it per move, walking m down the halving ladder
    # (m -> ceil(m/2) -> ...) until round 0 affords >= 4 visits per candidate,
    # so smaller budgets (eval matches, quick-gate evals) shrink the candidate
    # set instead of starving the round quota.
    gumbel_target_enabled: bool = False
    gumbel_root_enabled: bool = False
    gumbel_sequential_halving: bool = False
    gumbel_nonroot_select: bool = False
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0
    # Export-only σ softening for the improved-policy target π'. When set, this
    # c_scale overrides gumbel_c_scale in the exported target's σ call ONLY (the
    # in-search SH ranking and interior selection keep gumbel_c_scale). None =>
    # the target uses gumbel_c_scale, so behavior is unchanged.
    gumbel_target_c_scale: float | None = None
    gumbel_m: int = 16
    # Draw temperature τ applied to the LOGIT only in the Gumbel-Top-m draw sort
    # (candidate set ~ softmax(logits/τ) without replacement). 1.0 (or <=0) = the
    # current logit+g draw; τ>1 widens the sampled candidate set. Affects ONLY the
    # draw — the SH σ ranking, exported target, and TSS force-include use raw logits.
    gumbel_draw_temperature: float = 1.0
    gumbel_target_min_visits: int = 1
    # Play-policy quota prune: sample the PLAYED move (Full moves, T>0) from the
    # delta-visit histogram with round-0 quota losers zeroed — removes the SH
    # schedule mass (~budget/(R·m) per eliminated candidate) from move sampling
    # while leaving every recorded training target untouched.
    gumbel_play_prune: bool = False
    # --- Per-class Gumbel for the Fast PCR class (main_8) ----------------------
    # main_8 runs PUCT for Full turns and Gumbel+SH for Fast turns. These OPTIONAL
    # fast_* levers describe the Fast class's Gumbel view; the base
    # gumbel_*_enabled levers keep describing the Full/Init view. All default
    # None/False so ABSENT = today's single-profile behavior (the fast override
    # map equals the base map, and Rust's divergences_fast == divergences_full).
    # build_divergence_overrides folds a set fast_* value onto its base gumbel
    # counterpart in the SECOND (fast) override map only:
    #   fast_gumbel_root_enabled       -> gumbel_root
    #   fast_gumbel_sequential_halving -> gumbel_sequential_halving
    #   fast_gumbel_nonroot_select     -> gumbel_nonroot_select
    #   fast_gumbel_c_visit            -> gumbel_c_visit   (when not None)
    #   fast_gumbel_c_scale            -> gumbel_c_scale   (when not None)
    #   fast_gumbel_m                  -> gumbel_m         (when not None)
    #   fast_gumbel_play_prune         -> gumbel_play_prune
    fast_gumbel_root_enabled: bool = False
    fast_gumbel_sequential_halving: bool = False
    fast_gumbel_nonroot_select: bool = False
    fast_gumbel_c_visit: float | None = None
    fast_gumbel_c_scale: float | None = None
    fast_gumbel_m: int | None = None
    fast_gumbel_play_prune: bool = False
    # --- Blunder-seeded self-play (KataGo-style off-policy state coverage) -----
    # A fraction of games per epoch start from a stored mid-game position where
    # the net was recently "surprised" (high policy_surprise), instead of an
    # empty board. All flags default to a no-op: blunder_seed_fraction=0.0 means
    # zero games are seeded and the driver takes a path bit-identical to current
    # behavior (the seeding RNG is drawn from a NEW mix_seed stream only when
    # fraction>0, so existing streams are never perturbed). Seeds are mined from
    # the run's own recent npz shards (route (b): the ordered move prefix is
    # recovered from each row's hist_* placement history + policy_surprise +
    # turn_index — no .hxr cross-reference).
    blunder_seed_fraction: float = 0.0
    # Mine seeds from the last N epochs' sample shards.
    blunder_seed_recent_epochs: int = 5
    # Never seed deeper than this ply (late-game seeds make degenerate short
    # games).
    blunder_seed_max_ply: int = 40
    # Rows above this quantile of policy_surprise (over the mined full rows)
    # qualify as seed candidates.
    blunder_seed_surprise_quantile: float = 0.9


@dataclass(frozen=True)
class TrainingSection:
    batch_rows: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # --- Adaptive grad-clip --------------------------------------------------
    adaptive_clip: bool = True
    clip_c: float = 1.75
    clip_ema_decay: float = 0.99
    clip_warmup_steps: int = 50
    # --- Loss weights; defaults match the losses.py constants ----------------
    policy_weight: float = 1.0
    value_weight: float = 1.0
    opp_policy_weight: float = 0.25
    short_term_value_weight: float = 0.1
    moves_left_weight: float = 0.1
    q_head_weight: float = 0.1
    # Loss weight for the train-only soft_policy head. The soft target is
    # built in batching.collate_training: the row's base policy-improvement
    # distribution (the gumbel improved policy pi' when the row carries one,
    # else the visit policy), row-normalized then raised to pow(0.5) — T=2
    # softening — over its support only. constants.SOFT_POLICY_WEIGHT is the
    # single source for the default (losses.py re-exports it).
    soft_policy_weight: float = SOFT_POLICY_WEIGHT
    # --- Policy-surprise self-CE reweight ------------------------------------
    policy_surprise_uniform_fraction: float = 0.5
    policy_surprise_max_weight: float = 8.0
    warmup_steps: int = 0
    # --- Optimizer LR schedule (config-gated; default OFF) --------------------
    # Wired into the trainer's per-step optimizer update (see trainer.scheduled_lr).
    # ``lr_schedule="none"`` (the default) is a NO-OP: the trainer never touches
    # the optimizer LR, so a run that omits these keys behaves byte-identically to
    # before this feature (constant ``learning_rate``). When set to "cosine" or
    # "linear", the LR does a linear warmup over ``lr_warmup_steps`` OPTIMIZER
    # STEPS (0 -> learning_rate), then decays from ``learning_rate`` to ``lr_final``
    # over ``lr_decay_epochs`` EPOCHS, holding ``lr_final`` afterward. The step
    # counter that drives warmup is derived from the persisted
    # ``train_state.global_step_samples`` so warmup survives supervisor restarts
    # (and is fresh on an ``initialize_from`` warm start).
    lr_schedule: str = "none"
    lr_warmup_steps: int = 0
    lr_final: float = 0.0
    lr_decay_epochs: int = 0
    shuffle_keep_target_rows: int = 300_000
    # Replay-buffer shuffle-window knobs.
    shuffle_min_rows: int = 20_000
    shuffle_taper_window_exponent: float = 0.65
    shuffle_expand_window_per_row: float = 0.4
    shuffle_taper_window_scale: float = 20_000.0
    validation_fraction: float = 0.0
    train_samples_per_epoch: int = 100_000
    max_train_bucket_per_new_data: float = 8.0
    max_train_bucket_size: float = 500_000.0
    no_repeat_files: bool = False
    expand_backend: str = "serial"
    expand_workers: int = 0
    # Training-side policy-target selector, read by losses.py: "visit" (default;
    # visit-count target) or "gumbel" (the π'=softmax(logits+σ(completedQ))
    # target). A row falls back to "visit" when no gumbel target is present.
    policy_target: str = "visit"


# --- Multi-stage standalone evaluation ---------------------------------------
#
# An opt-in evaluator run by a standalone script (scripts/), not inside the
# training pipeline. Its product is a verdict label (PROMOTE / REGRESS / INCONCLUSIVE)
# plus rolling ratings; it does not gate, promote, or halt the run. The
# ``*_gating_*`` / ``*_promotion_*`` knobs below default OFF and are not wired
# to anything that changes training.
#
# Statistical design:
#  - SealBot is the cross-lineage zero-point (pinned at 0 Elo). Its edge is
#    down-weighted in difference inference via ``sealbot_overdispersion``.
#  - Permanent anchors (BC prefit + ep5) never slide; the sliding bracket is the
#    nearest two fixed log-grid rungs below the current epoch.
#  - The 128 games/epoch are paired (shared openings / common random numbers)
#    and scored pentanomially; pair-level SE + paired/effective counts feed the
#    Bradley-Terry likelihood. The BT fit must converge (max|grad| < tol) before
#    any covariance is computed.
#  - Resolution: a single 128-game epoch resolves roughly 100-120 Elo
#    (single-epoch SE(r_L - r_B) ~= 40-55 Elo); tighter resolution is a
#    multi-epoch rolling asymptote of the persisted pool. Stage B (SPRT) is a
#    gross-regression triage, not a calibrated 5%/5% test.


@dataclass(frozen=True)
class MultiStageEvalOpponents:
    """Opponent roster for the deep (Stage C) eval and the rolling pool.

    Three roles:
      * SealBot  -- cross-lineage zero-point / calibrator, pinned at 0 Elo.
      * permanent anchors -- never slide (BC prefit + ep5 by default).
      * sliding bracket -- the nearest ``bracket_size`` rungs of ``log_grid``
        strictly below the current epoch.
    """

    # SealBot zero-point. ``sealbot_path`` falls back to $SEALBOT_PATH when None
    # (matches hexo_runner SealBotConfig.resolved_path). When disabled, the pool
    # floats relative to the permanent anchors.
    sealbot_enabled: bool = True
    sealbot_path: str | None = None
    sealbot_variant: str = "current"
    sealbot_time_limit: float = 0.05
    # Strix zero-point (SootyOwl/hexo-strix GNN at fixed sims). Deterministic /
    # GPU-batched => STABLE strength, so it is the PREFERRED pinned 0-Elo anchor
    # (above SealBot) and enters difference inference at FULL weight. When disabled
    # the anchor falls back to SealBot, then the permanent anchors. ``strix_ckpt``
    # is a hexo-strix HeXONet ``checkpoint_*.pt``; None => disabled in practice.
    strix_enabled: bool = False
    strix_ckpt: str | None = None
    strix_sims: int = 256                   # hexo-strix EvalConfig default (DEFAULT_SIMS)
    strix_m: int = 16                       # m_actions (DEFAULT_M_ACTIONS)
    strix_device: str = "cuda"
    strix_label: str = "strix"              # stable pool/rating label (edges compound)
    # Permanent anchors, as (label, checkpoint-path) pairs. These never slide.
    # Relative paths use forward slashes and are resolved by the runner against
    # the repo/run root; ``_resolve_anchor_path`` short-circuits an absolute
    # path. The env var ``HEXFIELD_ANCHOR_ROOTS`` (os.pathsep-separated dirs)
    # prepends extra search roots. An unresolved anchor is recorded in
    # roster.dropped_anchors.
    #
    # EMPTY by default (decision D4). The old hexfield-lineage BC-prefit / ep5
    # anchors are a 15-plane net; the equivariant rewrite is a 25-plane net, so
    # a strict ``load_state_dict`` on the stem shape (7, 25, c) vs (7, 15, c)
    # ALWAYS fails. The re-anchor set for this lineage is Strix + SealBot + self
    # (the cross-lineage zero-points), configured per run — not a stale
    # checkpoint path baked into the defaults.
    permanent_anchors: tuple[tuple[str, str], ...] = ()
    # Fixed log-grid of epochs the sliding bracket is drawn from. The bracket is
    # the nearest ``bracket_size`` rungs strictly below the current epoch.
    log_grid: tuple[int, ...] = (5, 10, 20, 40, 80, 160)
    bracket_size: int = 2
    # Labels of opponents trained under the radius-8 legality regime. The support
    # radius is a process-global read once per process, so every opponent is
    # featurized at the live HEXFIELD_EQ_SUPPORT_RADIUS; a radius-8-trained net
    # forced to a different radius is out-of-distribution. Edges to these
    # opponents are annotated ``featurized_ood`` and excluded from the pinned BT
    # zero-point, but still participate descriptively. EMPTY by default: the old
    # ``bc_prefit`` radius-8 anchor is gone with ``permanent_anchors`` (D4).
    radius8_opponents: tuple[str, ...] = ()


@dataclass(frozen=True)
class MultiStageEvalSprt:
    """Stage B SPRT screen parameters -- a one-sided gross-regression triage.

    A one-sided sequential filter that tests only whether the candidate grossly
    regressed. Two simple hypotheses:

      * H0 (``elo0 = 0``): Elo gap ~0 vs the screen opponent.
      * H1 (``elo1 = -50``): a large negative Elo gap. ``winrate_from_elo`` makes
        ``p1 < 0.5 < p0``, so a loss-dominated record drives the LLR up to
        ``upper`` and accepts H1.

    Label mapping (implemented in multistage_eval._stage_b_sprt):

      * ``accept_h1`` -> ``"regress_suspected"``.
      * ``accept_h0`` -> ``"ok"`` / escalate-to-deep.
      * ``continue``  -> ``"escalate"`` (undecided under the cap -> deep eval).

    The screen does not short-circuit Stage C and does not gate/promote; Stage
    C/D (paired games + BT pool) is the authoritative measurement. With a small
    ``max_games`` cap and an expected-N near the indifference region of order
    ~285 decided games, most non-gross candidates ``escalate`` rather than
    resolve here. ``elo0``/``elo1`` are the H0/H1 Elo bounds; ``alpha``/``beta``
    the nominal error rates (advisory, given the cap); ``max_games`` caps the
    screen.
    """

    enabled: bool = True
    # H0: Elo gap ~0. H1: ~-50 Elo. See class docstring for the
    # accept_h0/accept_h1 -> label map.
    elo0: float = 0.0
    elo1: float = -50.0
    alpha: float = 0.05
    beta: float = 0.05
    max_games: int = 64


@dataclass(frozen=True)
class MultiStageEvalSection:
    """Standalone, opt-in multi-stage strength eval.

    Emits a verdict label and updates a persisted, SealBot-pinned Bradley-Terry
    pool; does not gate/promote/halt the run. Disabled by default
    (``enabled=False``) and invoked only by a standalone script.
    """

    # Master switch. Off by default.
    enabled: bool = False
    # Stage C budget: paired games per epoch (shared openings). These accumulate
    # into the rolling pool across epochs.
    games_budget: int = 128
    # Run the standalone eval against every Nth produced checkpoint/epoch.
    every_n_epochs: int = 5
    # Reduced eval search budget. The orchestrator runs eval at
    # ``full_search_visits`` (below) by default; this value is used only as an
    # explicit reduced-budget override.
    eval_visits: int = 128
    # Eval search budget. ``None`` -> use the production ``selfplay.search_visits``
    # (512); an int pins a specific budget. The deep eval (and the SPRT screen,
    # when enabled) play at this budget, threaded into Stage B + Stage C by the
    # orchestrator's _eval_visits.
    full_search_visits: int | None = None
    # Eval-only MCTS leaf-parallelism / virtual-loss batch. Threaded into the
    # eval search calls via the orchestrator's _eval_virtual_batch_size; does not
    # affect SelfplayConfig.virtual_batch_size (=4).
    eval_virtual_batch_size: int = 16
    # --- Equal-TIME eval budget (ray plan §3 L2/L3, review F7) -----------------
    # Per-move wall-clock budget in ms; 0.0 (default) = off (fixed-visit eval,
    # byte-identical to before this knob). The Rust search core has no native
    # time budget, so this is the documented cheap approximation
    # (eval_driver.calibrate_time_budget_visits): each hexfield net measures its
    # own nps with a few probe searches at the configured visit budget, then
    # plays the match at visits scaled so one move costs ~eval_time_budget_ms of
    # wall clock. Arm 4's L blocks run the slow flex path — a fixed-visit arena
    # hides that throughput cost; this leg charges it. Honored by
    # eval_driver.play_eval_match (net A) and HexfieldCheckpointAdapter (net B);
    # Strix (fixed sims, the pinned anchor) and SealBot (already time-limited)
    # ignore it.
    eval_time_budget_ms: float = 0.0
    # Opening plies temperature-sampled to diversify paired lines (shared opening
    # seed per pair => common random numbers across the two seat-swapped games).
    opening_plies: int = 8
    opening_temperature: float = 1.0
    # Primary hypothesis: candidate L vs reference B, via the BT difference-CI
    # (includes the Cov_LB term). Other opponent edges are descriptive only
    # (Wilson/Elo CIs, no significance verdict). With ``bonferroni_correction``,
    # per-edge alpha = 0.05/k when more than one edge carries a verdict.
    primary_alpha: float = 0.05
    bonferroni_correction: bool = True
    # Scale SealBot's edge effective count by this over-dispersion factor
    # (< 1 down-weights) in difference inference. It stays the pinned zero-point.
    sealbot_overdispersion: float = 0.5
    # Fraction of ``games_budget`` allocated to the SealBot zero-point pairing
    # (the rest is split evenly across the checkpoint opponents). Threaded into
    # ``allocate_budget`` by the orchestrator.
    sealbot_share: float = 0.25
    # Strix edge likelihood weight in the BT fit (full weight; contrast SealBot's
    # 0.5 over-dispersion down-weight). Strix at fixed sims is stable, so its edge
    # enters difference inference undeflated.
    strix_weight: float = 1.0
    # Fraction of ``games_budget`` for the Strix zero-point pairing.
    strix_share: float = 0.25
    # Bradley-Terry convergence guard: assert max|grad| < this before computing
    # covariance.
    bt_grad_tol: float = 1e-6
    bt_max_iters: int = 200
    # Persisted rolling pool, relative to the run diagnostics dir. Per-epoch
    # edges accumulate here.
    pool_path: str = "diagnostics/eval_pool.json"
    # Verdict thresholds (Elo, on the BT difference r_L - r_B). The CI must clear
    # these to label PROMOTE / REGRESS; otherwise INCONCLUSIVE.
    promote_elo_threshold: float = 0.0
    regress_elo_threshold: float = 0.0
    # The primary verdict compares the candidate to the highest checkpoint at
    # least ``verdict_reference_lag`` epochs below it; 0 uses the
    # immediately-prior checkpoint. The immediately-prior checkpoint still
    # appears as a descriptive bracket edge and is pooled into the BT fit; this
    # setting only chooses the reported verdict target and gates nothing.
    verdict_reference_lag: int = 5
    opponents: MultiStageEvalOpponents = field(default_factory=MultiStageEvalOpponents)
    sprt: MultiStageEvalSprt = field(default_factory=MultiStageEvalSprt)
    # --- Gating / promotion hooks: off by default ----------------------------
    # Not wired to anything that alters the run in this feature.
    eval_gating_enabled: bool = False
    eval_promotion_enabled: bool = False


@dataclass(frozen=True)
class HexfieldConfig:
    device: str = "cuda"
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    training: TrainingSection = field(default_factory=TrainingSection)
    multi_stage_eval: MultiStageEvalSection = field(default_factory=MultiStageEvalSection)

    def temperature_by_ply(self) -> list[float]:
        sp = self.selfplay
        out = []
        for ply in range(self.selfplay.max_game_plies):
            t = sp.temperature * (0.5 ** (ply / max(sp.temperature_halflife_plies, 1e-9)))
            out.append(max(sp.temperature_floor, t))
        return out


def _merge(cls, section: Mapping[str, Any]):
    known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**section)


def _merge_multi_stage_eval(section: Mapping[str, Any]) -> "MultiStageEvalSection":
    """Merge the multi-stage eval section, recursing into the ``opponents`` and
    ``sprt`` sub-tables. A flat ``_merge`` cannot handle these because the
    dataclass fields hold dataclass instances, not dicts; missing sub-tables
    fall back to their dataclass defaults so an absent toml -> all defaults."""
    section = dict(section)
    nested = {
        "opponents": MultiStageEvalOpponents,
        "sprt": MultiStageEvalSprt,
    }
    merged: dict[str, Any] = {}
    for key, sub_cls in nested.items():
        if key in section:
            merged[key] = _merge(sub_cls, dict(section.pop(key)))
    # ``section`` now holds only the scalar fields; reuse the flat merge for the
    # unknown-key guard, then overlay the parsed sub-sections.
    return _merge(MultiStageEvalSection, {**section, **merged})


ML_AUTO_DISABLED_FLAG = "ml_auto_disabled.flag"


def build_divergence_overrides(
    sp: SelfplayConfig, *, disabled: bool = False, fast: bool = False
) -> dict:
    """Build the Rust ``divergence_overrides`` dict from a SelfplayConfig.

    With ``fast=True`` this returns the SECOND (Fast-class) override map for the
    main_8 per-class refactor: the base map with its Gumbel fields replaced by
    the ``fast_*`` values (see ``build_fast_divergence_overrides``). With
    ``fast=False`` (default) it returns the base map exactly as before, so every
    existing caller and eval path is unchanged.

    Groups of levers:
      - moves-left knobs: the boolean levers are gated by ``off`` (the
        ``ml_auto_disabled`` config field or the ``disabled`` argument); the
        numeric constants are always passed.
      - dynamic c_puct / LCB knobs (c_scale, c_base, visit_scaled_c_puct,
        lcb_z), read by ``resolve_divergences``.
      - the six search divergences (nucleus_f64, new_child_fpu, lazy_widening,
        clean_root_prior_cache, dirichlet_shaped, pruned_dynamic_cpuct).
      - the Gumbel levers (default OFF).

    ``resolve_divergences`` applies this dict on top of the base selected by
    ``search_parity_mode`` (production() when False).

    All values are concrete bool/float/int (never None) because
    ``resolve_divergences`` calls ``.extract()``."""
    off = bool(disabled or sp.ml_auto_disabled)
    base = {
        # Moves-left utility (boolean levers gated by ml_auto_disabled/disabled).
        "moves_left_utility": bool(sp.moves_left_utility) and not off,
        "ml_weight": float(sp.ml_weight),
        "ml_scale": float(sp.ml_scale),
        "ml_q_gate": float(sp.ml_q_gate),
        "ml_two_sided": bool(sp.ml_two_sided) and not off,
        "ml_final_pick": bool(sp.ml_final_pick) and not off,
        "ml_final_pick_band": float(sp.ml_final_pick_band),
        # Dynamic c_puct + LCB.
        "c_scale": float(sp.c_scale),
        "c_base": float(sp.c_base),
        "visit_scaled_c_puct": bool(sp.visit_scaled_c_puct),
        "lcb_z": float(sp.lcb_z),
        # Search divergences.
        "nucleus_f64": bool(sp.nucleus_f64),
        "new_child_fpu": bool(sp.new_child_fpu),
        "lazy_widening": bool(sp.lazy_widening),
        "clean_root_prior_cache": bool(sp.clean_root_prior_cache),
        "dirichlet_shaped": bool(sp.dirichlet_shaped),
        "pruned_dynamic_cpuct": bool(sp.pruned_dynamic_cpuct),
        # TSS interior forced-move guard (Lever 0, default OFF).
        "tss_interior_guard": bool(sp.tss_interior_guard),
        # TSS deep-solver ladder (Stage 4, default OFF).
        "tss_solver_mode": int(sp.tss_solver_mode),
        "tss_solver_node_cap": int(sp.tss_solver_node_cap),
        "tss_solver_sample_16": int(sp.tss_solver_sample_16),
        "tss_solver_root_guard": bool(sp.tss_solver_root_guard),
        # TSS async solve pool (async rung, default OFF).
        "tss_solver_async": bool(sp.tss_solver_async),
        "tss_solver_async_threads": int(sp.tss_solver_async_threads),
        "tss_solver_async_threads_max": int(sp.tss_solver_async_threads_max),
        "tss_solver_park": bool(sp.tss_solver_park),
        "tss_solver_all_leaves": bool(sp.tss_solver_all_leaves),
        "tss_solver_park_timeout_ms": int(sp.tss_solver_park_timeout_ms),
        "tss_solver_async_inline_16": int(sp.tss_solver_async_inline_16),
        "tss_zone": bool(sp.tss_zone),
        "tss_zone_stale_filter": bool(sp.tss_zone_stale_filter),
        "tss_zone_count2": bool(sp.tss_zone_count2),
        "tss_pair_commutation": bool(sp.tss_pair_commutation),
        # TSS deep-solve semantic horizon (h16 floor, or 0 = unbounded) + ladder.
        "tss_solver_horizon": int(sp.tss_solver_horizon),
        "tss_solver_dual_pass": bool(sp.tss_solver_dual_pass),
        "tss_solver_loss_reserve_nodes": int(sp.tss_solver_loss_reserve_nodes),
        "tss_solver_group2": bool(sp.tss_solver_group2),
        "tss_solver_j2near": bool(sp.tss_solver_j2near),
        "tss_solver_horizon_ladder": bool(sp.tss_solver_horizon_ladder),
        # Gumbel AlphaZero levers (default OFF).
        "gumbel_target": bool(sp.gumbel_target_enabled),
        "gumbel_root": bool(sp.gumbel_root_enabled),
        "gumbel_sequential_halving": bool(sp.gumbel_sequential_halving),
        "gumbel_nonroot_select": bool(sp.gumbel_nonroot_select),
        "gumbel_c_visit": float(sp.gumbel_c_visit),
        "gumbel_c_scale": float(sp.gumbel_c_scale),
        "gumbel_m": int(sp.gumbel_m),
        # Draw temperature is always concrete (default 1.0 = today's draw).
        "gumbel_draw_temperature": float(sp.gumbel_draw_temperature),
        "gumbel_target_min_visits": int(sp.gumbel_target_min_visits),
        "gumbel_play_prune": bool(sp.gumbel_play_prune),
        # Export-only target σ override: emitted ONLY when set so an unset field
        # leaves the Rust default (target keeps gumbel_c_scale) untouched. This is
        # the sole override key that may be absent from the dict.
        **(
            {"gumbel_target_c_scale": float(sp.gumbel_target_c_scale)}
            if sp.gumbel_target_c_scale is not None
            else {}
        ),
    }
    if not fast:
        return base
    # Fast-class map: start from the base map and replace its Gumbel fields with
    # the Fast view's values. Booleans always apply; the optional numeric levers
    # (c_visit/c_scale/m) only override when set (None => keep the base value).
    fast_map = dict(base)
    fast_map["gumbel_root"] = bool(sp.fast_gumbel_root_enabled)
    fast_map["gumbel_sequential_halving"] = bool(sp.fast_gumbel_sequential_halving)
    fast_map["gumbel_nonroot_select"] = bool(sp.fast_gumbel_nonroot_select)
    fast_map["gumbel_play_prune"] = bool(sp.fast_gumbel_play_prune)
    if sp.fast_gumbel_c_visit is not None:
        fast_map["gumbel_c_visit"] = float(sp.fast_gumbel_c_visit)
    if sp.fast_gumbel_c_scale is not None:
        fast_map["gumbel_c_scale"] = float(sp.fast_gumbel_c_scale)
    if sp.fast_gumbel_m is not None:
        fast_map["gumbel_m"] = int(sp.fast_gumbel_m)
    return fast_map


def build_fast_divergence_overrides(
    sp: SelfplayConfig, *, disabled: bool = False
) -> dict:
    """The Fast-class (SECOND) override map for the continuous driver: the base
    map with its Gumbel fields folded to the ``fast_*`` values. Thin alias for
    ``build_divergence_overrides(sp, disabled=disabled, fast=True)``."""
    return build_divergence_overrides(sp, disabled=disabled, fast=True)


def build_eval_search_kwargs(
    sp: SelfplayConfig,
    *,
    visits: int,
    virtual_batch_size: int,
    active_root_limit: int,
    temperature: float = 0.0,
) -> dict:
    """Build the canonical eval ``session.search`` keyword bundle.

    The single source of truth for the search-knob dict every eval arena passes
    to ``HexfieldMctsSession.search``: the greedy-after-opening, no-Dirichlet
    profile shared by ``play_checkpoint_match`` / ``play_multi_checkpoint_match``
    / ``play_sealbot_match`` / ``play_strix_match``. ``visits`` /
    ``virtual_batch_size`` / ``active_root_limit`` are the already-resolved eval
    budget (see ``_resolve_eval_budget``); the remaining knobs read from ``sp``.

    Returns the 11 canonical fields (visits, c_puct, temperature,
    virtual_batch_size, active_root_limit, widening_policy_mass,
    widening_max_children, widening_min_children, fpu_reduction, tss_enabled,
    search_parity_mode). The PER-CALL args (seed, move_temperatures,
    divergence_overrides, evaluator, and the positional roots/states) stay at the
    call site — they are not part of this dict. A new self-play search knob is
    added here once and every arena inherits it."""
    return dict(
        visits=int(visits),
        c_puct=sp.c_puct,
        temperature=temperature,
        virtual_batch_size=int(virtual_batch_size),
        active_root_limit=int(active_root_limit),
        widening_policy_mass=sp.widening_policy_mass,
        widening_max_children=sp.widening_max_children,
        widening_min_children=sp.widening_min_children,
        fpu_reduction=sp.fpu_reduction,
        tss_enabled=sp.tss_enabled,
        search_parity_mode=sp.search_parity_mode,
    )


def parse_hexfield_config(config: Mapping[str, Any]) -> HexfieldConfig:
    config = dict(config or {})
    # Reject unknown TOP-LEVEL keys, not just unknown sub-keys: a typo'd section
    # (e.g. [model.config.slfplay]) would otherwise be silently dropped, leaving
    # production knobs at their defaults with no error.
    known_top = set(HexfieldConfig.__dataclass_fields__)
    unknown_top = set(config) - known_top
    if unknown_top:
        raise ValueError(f"unknown HexfieldConfig keys: {sorted(unknown_top)}")
    return HexfieldConfig(
        device=str(config.get("device", "cuda")),
        selfplay=_merge(SelfplayConfig, dict(config.get("selfplay", {}))),
        training=_merge(TrainingSection, dict(config.get("training", {}))),
        multi_stage_eval=_merge_multi_stage_eval(dict(config.get("multi_stage_eval", {}))),
    )
