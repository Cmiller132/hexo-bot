"""Unit tests for the hexfield multi-stage eval orchestrator (CPU-only, no GPU).

Exercises ``packages/hexfield/python/hexfield/multistage_eval.py``, the layer
that wires the game-running arena to the statistics core and emits a verdict
label. The statistics themselves are covered in ``test_hexfield_eval_stats.py``.

Sections:
  1. Opponent-ladder selection (``select_opponents``): roles/anchors/bracket/
     champion, the verdict_reference_lag stable target, anchor path resolution
     (repo-tree + run-data-tree), and budget allocation (even split + small-budget
     floor).
  2. Rolling BT pool persistence (``_save_pool`` / ``_load_pool`` /
     ``_bt_edges_from_pool``): round-trip stability, graceful degradation, and
     cross-epoch compounding.
  3. Verdict logic from synthetic edges: PROMOTE / REGRESS / INCONCLUSIVE, the
     primary-only invariant, descriptive-edge CI shape, custom thresholds.
  4. Stage flow + SPRT triage (label mapping; advisory, never a gate) + the
     full-sims wiring (Stage B/C at the production budget).
  5. Orchestrator robustness: SealBot fail-open (incl. import-error path) +
     always-pin-an-anchor.
  6. Pure-eval invariant: gating/promotion off, verdict mutates no run state.
  7. Run-in-parts: per-opponent parts, resume, durability, parts == monolithic.

The arena (which needs the GPU + the SealBot checkout) is mocked throughout via
the ``play_checkpoint_match`` / ``play_sealbot_match`` injection seams (the shared
``_FakeArena`` / ``_StubArena`` from ``hexfield_eval_kit``), so this collects and
runs on a CPU-only interpreter. ``multistage_eval`` imports torch lazily, so
importing it here does not require torch.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "packages" / "hexo_engine" / "python"))
sys.path.insert(0, str(_REPO / "packages" / "hexfield" / "python"))

from hexfield import multistage_eval as mse  # noqa: E402
from hexfield.config import (  # noqa: E402
    MultiStageEvalSection,
    parse_hexfield_config,
)

from hexfield_eval_kit import (  # noqa: E402
    _FakeArena,
    _StubArena,
    _make_run,
    _scores_for_winrate,
)


# --------------------------------------------------------------------------- #
# Local helpers.
# --------------------------------------------------------------------------- #
def _no_sprt_config(**overrides):
    """Production-default config with Stage B SPRT disabled."""

    return parse_hexfield_config({"multi_stage_eval": {"sprt": {"enabled": False}, **overrides}})


def _run(run_dir: Path, candidate_epoch: int, arena, *, config=None, **kw) -> dict:
    cfg = config if config is not None else _no_sprt_config()
    return mse.run_multistage_eval(
        run_dir,
        run_dir / "checkpoints" / f"epoch_{candidate_epoch:06d}.pt",
        cfg,
        candidate_epoch=candidate_epoch,
        write_diagnostics=False,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
        **kw,
    )


def _candidate(run: Path, epoch: int) -> Path:
    return run / "checkpoints" / f"epoch_{epoch:06d}.pt"


# =========================================================================== #
# 1. Opponent-ladder selection.
# =========================================================================== #
def test_roster_roles_anchors_bracket_champion(tmp_path: Path) -> None:
    """At a mid-ladder epoch the roster has every role: permanent anchors
    (BC + ep5), a sliding bracket of the nearest log-grid rungs strictly below,
    and the single highest-prior-epoch champion."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = MultiStageEvalSection()
    roster = mse.select_opponents(run, run / "checkpoints" / "epoch_000040.pt", cfg, candidate_epoch=40)

    assert roster.candidate_label == "cand_ep40"
    assert roster.candidate_epoch == 40
    assert roster.sealbot is not None and roster.sealbot.role == "sealbot"
    assert roster.sealbot.ckpt is None  # SealBot is an external engine, not a ckpt

    by_label = {o.label: o for o in roster.opponents}
    # Permanent anchors: BC prefit + ep5.
    assert by_label["bc_prefit"].role == "anchor"
    assert by_label["ep5"].role == "anchor"
    # Sliding bracket: nearest log-grid rungs strictly below 40 -> {10, 20}. 20
    # is also the champion (highest prior epoch) so it is de-duped to "champion".
    assert by_label["ep10"].role == "bracket"
    # Champion: highest existing epoch strictly below the candidate (ep20, since
    # no ep30 exists on disk).
    assert roster.champion is not None
    assert roster.champion.label == "ep20"
    assert roster.champion.epoch == 20
    assert by_label["ep20"].role == "champion"


def test_roster_dedupes_anchor_that_is_also_champion(tmp_path: Path) -> None:
    """When the prior champion is a permanent anchor (ep5 at candidate ep10), it
    appears once in the roster, flagged champion."""

    run = _make_run(tmp_path, epochs=(5, 10))
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000010.pt", MultiStageEvalSection(), candidate_epoch=10
    )
    ep5_entries = [o for o in roster.opponents if o.label == "ep5"]
    assert len(ep5_entries) == 1
    assert ep5_entries[0].role == "champion"
    assert roster.champion is not None and roster.champion.label == "ep5"
    # No duplicate labels anywhere in the roster.
    labels = [o.label for o in roster.opponents]
    assert len(labels) == len(set(labels))


def test_roster_first_eligible_epoch_has_no_champion(tmp_path: Path) -> None:
    """The lowest epoch (no prior checkpoint below it) yields champion=None: no
    primary hypothesis exists, and the downstream verdict is INCONCLUSIVE."""

    run = _make_run(tmp_path, epochs=(5, 10, 20))
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000005.pt", MultiStageEvalSection(), candidate_epoch=5
    )
    assert roster.champion is None
    # Bracket is empty too (no grid rung strictly below 5).
    assert all(o.role != "bracket" for o in roster.opponents)


def test_bracket_is_bounded_and_slides_as_epochs_grow(tmp_path: Path) -> None:
    """The bracket window is the nearest ``bracket_size`` log-grid rungs strictly
    below the candidate; it tracks the candidate epoch upward, and its top rung is
    de-duped into the champion role."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40, 80, 160))
    cfg = MultiStageEvalSection()  # bracket_size=2, log_grid=(5,10,20,40,80,160)
    grid = sorted(cfg.opponents.log_grid)

    def roster_for(cand_epoch: int):
        return mse.select_opponents(
            run, run / "checkpoints" / f"epoch_{cand_epoch:06d}.pt", cfg, candidate_epoch=cand_epoch
        )

    for cand in (40, 80, 160):
        roster = roster_for(cand)
        bracket = {o.label for o in roster.opponents if o.role == "bracket"}
        # Bracket window = the nearest <=2 grid rungs strictly below the candidate.
        window = [f"ep{g}" for g in grid if g < cand][-cfg.opponents.bracket_size:]
        # The literal "bracket" role is the window minus the champion (top rung).
        assert roster.champion is not None and roster.champion.label == window[-1]
        assert bracket == set(window) - {roster.champion.label}
        # Bounded: the bracket window never exceeds bracket_size.
        assert len({*bracket, roster.champion.label}) <= cfg.opponents.bracket_size

    # The window slides up as the candidate climbs (nearest-below, not cumulative).
    assert roster_for(40).champion.label == "ep20"
    assert roster_for(80).champion.label == "ep40"
    assert roster_for(160).champion.label == "ep80"


def test_bracket_size_zero_yields_no_bracket(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = MultiStageEvalSection(
        opponents=dataclasses.replace(MultiStageEvalSection().opponents, bracket_size=0)
    )
    roster = mse.select_opponents(run, run / "checkpoints" / "epoch_000040.pt", cfg, candidate_epoch=40)
    assert all(o.role != "bracket" for o in roster.opponents)
    # Champion still resolves (it is independent of the bracket).
    assert roster.champion is not None and roster.champion.label == "ep20"


def test_missing_anchor_checkpoints_are_skipped(tmp_path: Path) -> None:
    """A permanent anchor whose file is absent is silently skipped, not an error.

    The default ``bc_prefit`` anchor path resolves against the repo tree as well
    as the run-data tree, so this test uses anchor paths that resolve to nothing
    under any root.
    """

    run = _make_run(tmp_path, epochs=(10, 20), bc=False)  # no BC file, no ep5 file
    cfg = MultiStageEvalSection(
        opponents=dataclasses.replace(
            MultiStageEvalSection().opponents,
            permanent_anchors=(
                # Repo-relative under a uniquely-named dir that exists nowhere.
                ("bc_prefit", "runs/__hexfield_no_such_bc__/checkpoint_epoch2.pt"),
                # Bare filename absent from the tmp checkpoints dir.
                ("ep5", "epoch_000005.pt"),
            ),
        )
    )
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000020.pt", cfg, candidate_epoch=20
    )
    labels = {o.label for o in roster.opponents}
    assert "bc_prefit" not in labels  # file resolves nowhere -> skipped
    assert "ep5" not in labels        # ep5 file absent -> skipped
    # The ep10 champion still resolves from what is on disk.
    assert roster.champion is not None and roster.champion.label == "ep10"


def test_sealbot_disabled_drops_zero_point(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = MultiStageEvalSection(
        opponents=dataclasses.replace(MultiStageEvalSection().opponents, sealbot_enabled=False)
    )
    roster = mse.select_opponents(run, run / "checkpoints" / "epoch_000040.pt", cfg, candidate_epoch=40)
    assert roster.sealbot is None
    assert mse.SEALBOT_LABEL not in roster.all_labels()


def test_verdict_reference_lag_decorrelates_target_on_contiguous_ladder(tmp_path: Path) -> None:
    """On a contiguous ladder the verdict target is the highest checkpoint at or
    below (candidate - lag), not the immediately-prior checkpoint; lag=0 selects
    the immediately-prior checkpoint."""

    run = _make_run(tmp_path, epochs=(10, 11, 12, 13, 14, 15))
    cfg = MultiStageEvalSection(verdict_reference_lag=5)
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000015.pt", cfg, candidate_epoch=15
    )
    assert roster.champion is not None
    assert roster.champion.label == "ep10"  # 15 - 5, not ep14
    assert roster.champion.epoch == 10

    cfg0 = MultiStageEvalSection(verdict_reference_lag=0)
    roster0 = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000015.pt", cfg0, candidate_epoch=15
    )
    assert roster0.champion is not None and roster0.champion.label == "ep14"


def test_verdict_reference_lag_falls_back_when_no_old_enough_prior(tmp_path: Path) -> None:
    """When nothing is >= lag epochs below the candidate, the target falls back to
    the nearest prior checkpoint so a hypothesis exists."""

    run = _make_run(tmp_path, epochs=(1, 2, 3))
    cfg = MultiStageEvalSection(verdict_reference_lag=5)  # nothing 5 below ep3
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000003.pt", cfg, candidate_epoch=3
    )
    assert roster.champion is not None and roster.champion.label == "ep2"


def test_immediately_prior_still_pooled_as_descriptive_edge(tmp_path: Path) -> None:
    """The immediately-prior checkpoint still appears as a descriptive (bracket)
    opponent and is pooled into the BT fit; only the reported verdict target rests
    on the lagged reference."""

    run = _make_run(tmp_path, epochs=(10, 11, 12, 13, 14, 15))
    cfg = _no_sprt_config(
        verdict_reference_lag=5,
        opponents={"log_grid": (10, 12, 14), "bracket_size": 2, "sealbot_enabled": False},
    )
    arena = _StubArena(per_score=2)
    rep = _run(run, 15, arena, config=cfg)

    # Verdict target is the highest epoch <= 15-5=10 -> ep10.
    assert rep["verdict"]["primary"] is not None
    assert rep["verdict"]["primary"]["champion"] == "ep10"
    # The near-candidate bracket rungs (ep12, ep14) are played and pooled as
    # descriptive edges even though they are not the verdict target.
    played = {label for label, _ in arena.ckpt_calls}
    assert {"ep12", "ep14"} <= played
    pooled = {e["opponent"] for e in rep["edges"]}
    assert {"ep12", "ep14"} <= pooled


def test_verdict_reference_lag_in_config_summary(tmp_path: Path) -> None:
    """The reference lag is surfaced in the report's config summary for audit."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.6, n))
    rep = _run(run, 40, arena)
    assert rep["meta"]["config"]["verdict_reference_lag"] == 5


def test_bc_prefit_resolves_from_repo_tree(tmp_path: Path, monkeypatch) -> None:
    """bc_prefit path resolution: the run-data tree has a ``hexfield_bc_1/`` dir
    but no ``checkpoint_epoch2.pt``. The resolver tries the repo-tree root (derived
    from the module's own location) and returns the file that exists there."""

    # Run-data tree with a hexfield_bc_1 DIR but no BC FILE.
    data_tree = tmp_path / "run-data"
    run = data_tree / "runs" / "hexfield_main_1"
    (run / "checkpoints").mkdir(parents=True)
    (data_tree / "runs" / "hexfield_bc_1").mkdir(parents=True)  # dir only, no file
    for epoch in (5, 10, 20, 40):
        (run / "checkpoints" / f"epoch_{epoch:06d}.pt").write_text("stub", encoding="utf-8")

    # Repo tree that holds the BC file; make the resolver treat it as the repo
    # root by monkeypatching the module file location.
    repo_tree = tmp_path / "repo"
    bc_file = repo_tree / "runs" / "hexfield_bc_1" / "checkpoint_epoch2.pt"
    bc_file.parent.mkdir(parents=True)
    bc_file.write_text("stub", encoding="utf-8")
    # _resolve_anchor_path derives the repo root as Path(__file__).parents[4]; make
    # that resolve to our fake repo tree (parents[4] == repo_tree).
    fake_module_file = repo_tree / "packages" / "hexfield" / "python" / "hexfield" / "multistage_eval.py"
    fake_module_file.parent.mkdir(parents=True)
    fake_module_file.write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(mse, "__file__", str(fake_module_file))

    resolved = mse._resolve_anchor_path(
        run, run / "checkpoints", "runs/hexfield_bc_1/checkpoint_epoch2.pt"
    )
    # Resolved to the repo-tree file (which exists), not the run-data path.
    assert resolved == bc_file
    assert resolved.is_file()
    # And select_opponents therefore pins bc_prefit as an anchor opponent.
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000040.pt", MultiStageEvalSection(), candidate_epoch=40
    )
    by_label = {o.label: o for o in roster.opponents}
    assert "bc_prefit" in by_label
    assert by_label["bc_prefit"].ckpt == bc_file


def test_bc_prefit_prefers_run_data_tree_when_present(tmp_path: Path) -> None:
    """When the BC file exists under the run-data tree, that root is preferred, so
    the repo-tree fallback only fires when the run-data tree lacks the file. Also
    exercised end-to-end via select_opponents pinning."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40), bc=True)  # builds the BC file
    resolved = mse._resolve_anchor_path(
        run, run / "checkpoints", "runs/hexfield_bc_1/checkpoint_epoch2.pt"
    )
    assert resolved == tmp_path / "runs" / "hexfield_bc_1" / "checkpoint_epoch2.pt"
    assert resolved.is_file()
    roster = mse.select_opponents(
        run, run / "checkpoints" / "epoch_000040.pt", MultiStageEvalSection(), candidate_epoch=40
    )
    by_label = {o.label: o for o in roster.opponents}
    assert by_label["bc_prefit"].ckpt == resolved


def test_allocate_budget_even_split_and_sealbot_share() -> None:
    """Budget allocation: SealBot gets a fixed share, the rest is split evenly
    across checkpoint opponents and rounded to an even per-pairing count."""

    alloc = mse.allocate_budget(128, n_checkpoint_opponents=4, has_sealbot=True)
    assert alloc[mse.SEALBOT_LABEL] == 32  # 25% default share
    assert alloc["per_checkpoint"] % 2 == 0  # even pairings
    assert alloc["per_checkpoint"] == 24     # (128-32)//4 = 24, already even

    # No SealBot -> the whole budget goes to checkpoints.
    alloc2 = mse.allocate_budget(120, n_checkpoint_opponents=3, has_sealbot=False)
    assert alloc2[mse.SEALBOT_LABEL] == 0
    assert alloc2["per_checkpoint"] == 40

    # No checkpoint opponents (first epoch) -> everything to SealBot if present.
    alloc3 = mse.allocate_budget(100, n_checkpoint_opponents=0, has_sealbot=True)
    assert alloc3[mse.SEALBOT_LABEL] == 100
    assert alloc3["per_checkpoint"] == 0


def test_allocate_budget_floors_each_opponent_to_one_pair() -> None:
    """At a small positive budget every selected opponent gets >=1 CRN pair
    (2 games), so the champion edge is always played; a zero budget stays
    all-zeros."""

    # budget 4, 3 checkpoint opponents, SealBot on.
    alloc = mse.allocate_budget(4, n_checkpoint_opponents=3, has_sealbot=True)
    assert alloc["per_checkpoint"] == 2
    assert alloc["per_checkpoint"] % 2 == 0  # stays an even pair count
    assert alloc[mse.SEALBOT_LABEL] >= 2

    # budget 4 with many opponents.
    alloc2 = mse.allocate_budget(4, n_checkpoint_opponents=10, has_sealbot=True)
    assert alloc2["per_checkpoint"] == 2
    assert alloc2[mse.SEALBOT_LABEL] >= 2

    # No SealBot: per_checkpoint still floored.
    assert mse.allocate_budget(4, n_checkpoint_opponents=3, has_sealbot=False)["per_checkpoint"] == 2

    # Zero budget -> all-zeros.
    assert mse.allocate_budget(0, n_checkpoint_opponents=4, has_sealbot=True) == {
        mse.SEALBOT_LABEL: 0, "per_checkpoint": 0,
    }

    # Production budgets are unchanged by the floor.
    assert mse.allocate_budget(128, n_checkpoint_opponents=4, has_sealbot=True)["per_checkpoint"] == 24
    assert mse.allocate_budget(120, n_checkpoint_opponents=3, has_sealbot=False)["per_checkpoint"] == 40


def test_small_budget_run_pins_anchor_and_produces_primary(tmp_path: Path) -> None:
    """End-to-end of the budget floor + anchor pin together: with a tiny
    games_budget the champion is played, the pool anchors, and a primary
    hypothesis block is produced."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _StubArena(per_score=2, sealbot_winrate=0.6)
    rep = _run(run, 40, arena, config=_no_sprt_config(games_budget=4))

    assert rep["ratings"]["fit"]["converged"] is True
    assert rep["verdict"]["primary"] is not None
    assert rep["verdict"]["primary"]["champion"] == "ep20"  # lag-5 reference of ep40
    # Every checkpoint opponent got at least one pair (>=2 games).
    champ_calls = [n for lb, n in arena.ckpt_calls if lb == "ep20"]
    assert champ_calls and all(n >= 2 for n in champ_calls)


# =========================================================================== #
# 2. Rolling BT pool persistence.
# =========================================================================== #
def test_pool_roundtrip_is_stable(tmp_path: Path) -> None:
    """Write a pool, reload it, and confirm the edges + anchor survive verbatim
    and project to the same BT edges."""

    pool = tmp_path / "diagnostics" / "eval_pool.json"
    doc = {
        "format": mse.POOL_FORMAT,
        "version": mse.POOL_VERSION,
        "anchor": mse.SEALBOT_LABEL,
        "edges": [
            {"epoch": 10, "a": "cand_ep10", "b": "ep5", "wins_a": 41.0, "wins_b": 23.0,
             "weight": 1.0, "kind": "checkpoint", "raw": {}},
            {"epoch": 10, "a": "cand_ep10", "b": "sealbot", "wins_a": 30.0, "wins_b": 10.0,
             "weight": 0.5, "kind": "sealbot", "raw": {}},
        ],
    }
    mse._save_pool(pool, doc)
    back = mse._load_pool(pool)
    assert back["anchor"] == mse.SEALBOT_LABEL
    assert back["edges"] == doc["edges"]

    edges = {(e.a, e.b, e.weight): (e.wins_a, e.wins_b) for e in mse._bt_edges_from_pool(back)}
    assert edges[("cand_ep10", "ep5", 1.0)] == (41.0, 23.0)
    # SealBot edge keeps its down-weight (0.5), separate from full-weight edges.
    assert edges[("cand_ep10", "sealbot", 0.5)] == (30.0, 10.0)


def test_pool_load_degrades_gracefully(tmp_path: Path) -> None:
    """Missing / corrupt / foreign-format pools yield a fresh empty pool rather
    than raising."""

    missing = tmp_path / "nope.json"
    fresh = mse._load_pool(missing)
    assert fresh["format"] == mse.POOL_FORMAT and fresh["edges"] == []

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{ not valid json", encoding="utf-8")
    assert mse._load_pool(corrupt)["edges"] == []

    foreign = tmp_path / "foreign.json"
    foreign.write_text(json.dumps({"format": "something.else", "edges": [1, 2, 3]}), encoding="utf-8")
    fresh3 = mse._load_pool(foreign)
    assert fresh3["format"] == mse.POOL_FORMAT and fresh3["edges"] == []


def test_bt_edges_aggregate_across_epochs_and_keep_weights_separate() -> None:
    """The append-only edge log projects to one BT edge per (unordered pair,
    weight): repeated epochs of a pairing pool their counts, reversed directions
    canonicalise together, and the down-weighted SealBot edge does not merge with
    a full-weight edge."""

    doc = {
        "edges": [
            {"a": "cand", "b": "ep5", "wins_a": 40, "wins_b": 20, "weight": 1.0},
            {"a": "cand", "b": "ep5", "wins_a": 30, "wins_b": 10, "weight": 1.0},   # same pair -> pooled
            {"a": "ep5", "b": "cand", "wins_a": 5, "wins_b": 5, "weight": 1.0},      # reversed -> canonicalised
            {"a": "cand", "b": "sealbot", "wins_a": 30, "wins_b": 10, "weight": 0.5},  # down-weighted, distinct edge
        ]
    }
    edges = {(e.a, e.b, e.weight): (e.wins_a, e.wins_b) for e in mse._bt_edges_from_pool(doc)}
    # cand-ep5 pooled: 40+30 + (reversed b-wins) 5 = 75 cand wins; 20+10 + 5 = 35 ep5 wins.
    assert edges[("cand", "ep5", 1.0)] == (75.0, 35.0)
    assert edges[("cand", "sealbot", 0.5)] == (30.0, 10.0)
    assert len(edges) == 2  # exactly two distinct (pair, weight) edges


def test_pool_persists_and_compounds_across_runs(tmp_path: Path) -> None:
    """End-to-end: two eval runs of the same epoch append to the persisted pool
    and the primary difference SE shrinks as the games compound."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)

    rep1 = mse.run_multistage_eval(
        run, run / "checkpoints" / "epoch_000040.pt", _no_sprt_config(),
        candidate_epoch=40, write_diagnostics=True,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
    )
    pool_path = run / "diagnostics" / "eval_pool.json"
    edges_after_1 = len(json.loads(pool_path.read_text(encoding="utf-8"))["edges"])
    assert edges_after_1 > 0

    rep2 = mse.run_multistage_eval(
        run, run / "checkpoints" / "epoch_000040.pt", _no_sprt_config(),
        candidate_epoch=40, write_diagnostics=True,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
    )
    doc2 = json.loads(pool_path.read_text(encoding="utf-8"))
    # Append-only: the second run doubles the edge rows.
    assert len(doc2["edges"]) == 2 * edges_after_1
    assert doc2["format"] == mse.POOL_FORMAT and doc2["anchor"] == mse.SEALBOT_LABEL

    # More pooled games -> a strictly tighter primary difference SE.
    se1 = rep1["verdict"]["primary"]["se_elo"]
    se2 = rep2["verdict"]["primary"]["se_elo"]
    assert se2 < se1
    assert rep1["ratings"]["fit"]["converged"] and rep2["ratings"]["fit"]["converged"]
    assert rep2["ratings"]["fit"]["anchor"] == mse.SEALBOT_LABEL


# =========================================================================== #
# 3. Verdict logic from synthetic edge results.
# =========================================================================== #
def test_verdict_promote_when_candidate_dominates_champion(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 40, arena)
    assert rep["verdict"]["label"] == "PROMOTE"
    lo, hi = rep["verdict"]["primary"]["elo_diff_ci95"]
    assert lo > 0.0  # whole difference CI above the promote threshold (0)
    assert rep["verdict"]["primary"]["champion"] == "ep20"
    assert rep["verdict"]["primary"]["candidate"] == "cand_ep40"


def test_verdict_regress_when_candidate_loses_to_champion(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.25, n), sealbot_winrate=0.3)
    rep = _run(run, 40, arena)
    assert rep["verdict"]["label"] == "REGRESS"
    lo, hi = rep["verdict"]["primary"]["elo_diff_ci95"]
    assert hi < 0.0  # whole difference CI below the regress threshold (0)


def test_verdict_inconclusive_when_even(tmp_path: Path) -> None:
    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.5, n), sealbot_winrate=0.5)
    rep = _run(run, 40, arena)
    assert rep["verdict"]["label"] == "INCONCLUSIVE"
    lo, hi = rep["verdict"]["primary"]["elo_diff_ci95"]
    assert lo < 0.0 < hi  # CI straddles 0


def test_only_primary_hypothesis_drives_the_verdict(tmp_path: Path) -> None:
    """The verdict rests on one primary edge (candidate vs prior champion).
    Crushing every non-champion (descriptive) edge while staying even with the
    champion leaves the label at INCONCLUSIVE, and the descriptive edges carry no
    significance verdict."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))

    def scorer(label_b: str, n_pairs: int) -> list[int]:
        target = 0.5 if label_b == "ep20" else 0.75
        return _scores_for_winrate(target, n_pairs)

    arena = _FakeArena(ckpt_scorer=scorer, sealbot_winrate=0.95)
    rep = _run(run, 40, arena)

    assert rep["verdict"]["label"] == "INCONCLUSIVE"

    primary = [e for e in rep["edges"] if e.get("primary")]
    descriptive = [e for e in rep["edges"] if not e.get("primary")]
    assert [e["opponent"] for e in primary] == ["ep20"]
    assert {e["opponent"] for e in descriptive} == {"sealbot", "bc_prefit", "ep5", "ep10"}
    for e in descriptive:
        assert "label" not in e and "verdict" not in e
        assert e["primary"] is False


def test_no_champion_yields_inconclusive_with_no_primary(tmp_path: Path) -> None:
    """First eligible epoch (no prior champion): verdict INCONCLUSIVE and the
    primary block is absent."""

    run = _make_run(tmp_path, epochs=(5, 10, 20))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 5, arena)
    assert rep["verdict"]["label"] == "INCONCLUSIVE"
    assert rep["verdict"].get("primary") is None
    assert all(not e.get("primary") for e in rep["edges"])


def test_custom_thresholds_shift_the_verdict(tmp_path: Path) -> None:
    """Verdict thresholds are configurable: a demanding promote threshold turns a
    small edge INCONCLUSIVE. The thresholds only relabel."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)

    assert _run(run, 40, arena)["verdict"]["label"] == "PROMOTE"

    rep = _run(run, 40, arena, config=_no_sprt_config(promote_elo_threshold=10_000.0))
    assert rep["verdict"]["label"] == "INCONCLUSIVE"
    assert rep["verdict"]["primary"]["promote_threshold_elo"] == 10_000.0


def test_descriptive_edges_have_pairlevel_cis_not_pergame(tmp_path: Path) -> None:
    """Paired (checkpoint) edges report a pair-level CI block: the descriptive edge
    carries ``elo_ci95_pairlevel`` and ``winrate_ci95`` and is flagged paired."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 40, arena)
    ckpt_edges = [e for e in rep["edges"] if e["kind"] == "checkpoint"]
    assert ckpt_edges  # anchors + bracket + champion
    for e in ckpt_edges:
        assert e["paired"] is True
        assert "elo_ci95_pairlevel" in e
        lo, hi = e["winrate_ci95"]
        assert 0.0 <= lo <= hi <= 1.0
    # The SealBot edge is unpaired and reports its Wilson win-rate CI + down-weight.
    sb_edge = next(e for e in rep["edges"] if e["kind"] == "sealbot")
    assert sb_edge["paired"] is False
    assert sb_edge["down_weight"] == pytest.approx(0.5)
    assert rep["sealbot_winrate_ci95"] is not None


# =========================================================================== #
# 4. Stage flow + SPRT triage + full-sims wiring.
# =========================================================================== #
def test_stage_flow_runs_all_four_stages(tmp_path: Path) -> None:
    """The orchestrator returns the staged A-D structure with the deep eval and
    pool completing (Stage B skipped here since SPRT is disabled)."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 40, arena)
    stages = {s["stage"]: s["status"] for s in rep["stages"]}
    assert stages["A_bridge"] == "ok"
    assert stages["B_sprt"] == "skipped"
    assert stages["C_deep"] == "completed"
    assert stages["D_pool"] == "completed"
    opponents_played = {label for label, _ in arena.ckpt_calls}
    assert {"ep20", "ep10", "ep5", "bc_prefit"} <= opponents_played
    assert arena.sealbot_calls  # SealBot zero-point edge played


def test_sprt_config_defaults_are_gross_regression_framed() -> None:
    """The config defaults frame H0=fine (elo0=0) vs H1=grossly-regressed
    (elo1 negative)."""

    sprt = MultiStageEvalSection().sprt
    assert sprt.elo0 == 0.0
    assert sprt.elo1 < 0.0  # gross-regression alternative


def test_sprt_triage_maps_accept_h1_to_regress_and_h0_to_ok(tmp_path: Path, monkeypatch) -> None:
    """Label mapping: accept_h1 -> 'regress_suspected'; accept_h0 -> 'ok';
    continue -> 'escalate'. Stubs eval_stats.sprt's verdict so the test is
    independent of the SPRT arithmetic."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _StubArena(per_score=1)  # champion match content unused here
    cfg = parse_hexfield_config({"multi_stage_eval": {"sprt": {"enabled": True, "max_games": 8}}})

    from hexfield import eval_stats

    cases = {
        "accept_h1": "regress_suspected",
        "accept_h0": "ok",
        "continue": "escalate",
    }
    for sprt_verdict, expected_label in cases.items():
        def fake_sprt(wins, losses, **kw):
            return eval_stats.SPRTResult(llr=0.0, lower=-1.0, upper=1.0, verdict=sprt_verdict)

        monkeypatch.setattr(mse.eval_stats, "sprt", fake_sprt)
        rep = _run(run, 40, arena, config=cfg)
        stage_b = next(s for s in rep["stages"] if s["stage"] == "B_sprt")
        assert stage_b["sprt_verdict"] == sprt_verdict
        assert stage_b["triage"] == expected_label, (sprt_verdict, stage_b["triage"])


def test_sprt_stage_b_is_advisory_and_never_short_circuits(tmp_path: Path, monkeypatch) -> None:
    """Stage B runs as triage (its own verdict is reported and screened against the
    prior champion) but does not short-circuit the deep eval: even a strong
    accept_h1 still runs Stage C/D, which produce the authoritative verdict."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = parse_hexfield_config({"multi_stage_eval": {"sprt": {"enabled": True, "max_games": 8}}})
    from hexfield import eval_stats

    def fake_sprt(wins, losses, **kw):
        return eval_stats.SPRTResult(llr=99.0, lower=-1.0, upper=1.0, verdict="accept_h1")

    monkeypatch.setattr(mse.eval_stats, "sprt", fake_sprt)
    rep = _run(run, 40, _StubArena(per_score=2), config=cfg)
    stage_b = next(s for s in rep["stages"] if s["stage"] == "B_sprt")
    assert stage_b["status"] == "completed"
    assert stage_b["vs"] == "ep20"  # screened against the prior champion
    assert stage_b["triage"] == "regress_suspected"
    # Deep eval + pool still ran and produced the authoritative verdict.
    assert any(s["stage"] == "C_deep" and s["status"] == "completed" for s in rep["stages"])
    assert any(s["stage"] == "D_pool" and s["status"] == "completed" for s in rep["stages"])
    assert rep["verdict"]["label"] in {"PROMOTE", "REGRESS", "INCONCLUSIVE"}


def test_full_sims_threaded_into_stage_b_and_c_by_default(tmp_path: Path) -> None:
    """By default the orchestrator threads selfplay.search_visits (512) into both
    the SPRT screen (Stage B) and the deep eval (Stage C checkpoint + SealBot),
    not eval_visits (128)."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    cfg = parse_hexfield_config({"multi_stage_eval": {"sprt": {"enabled": True, "max_games": 16}}})
    assert cfg.selfplay.search_visits == 512
    assert cfg.multi_stage_eval.eval_visits == 128

    rep = _run(run, 40, arena, config=cfg)

    assert arena.ckpt_visits, "no checkpoint matches were played"
    assert all(v == 512 for v in arena.ckpt_visits), arena.ckpt_visits
    assert arena.sealbot_visits and all(v == 512 for v in arena.sealbot_visits)
    for e in rep["edges"]:
        assert e["provenance"].get("eval_visits") == 512, (e["opponent"], e["provenance"])


def test_full_search_visits_knob_overrides_default(tmp_path: Path) -> None:
    """``full_search_visits``, when set, is threaded everywhere instead of
    selfplay.search_visits."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    cfg = parse_hexfield_config(
        {"multi_stage_eval": {"full_search_visits": 256, "sprt": {"enabled": True, "max_games": 16}}}
    )
    _run(run, 40, arena, config=cfg)
    assert all(v == 256 for v in arena.ckpt_visits), arena.ckpt_visits
    assert all(v == 256 for v in arena.sealbot_visits)
    assert cfg.multi_stage_eval.eval_visits == 128  # not used for the eval budget


# =========================================================================== #
# 5. Orchestrator robustness: SealBot fail-open + always-pin-an-anchor.
# =========================================================================== #
def test_sealbot_unavailable_drops_edge_and_completes_with_ratings(tmp_path: Path) -> None:
    """When SealBot's compiled extension is missing, ``play_sealbot_match`` raises.
    The orchestrator drops that one edge, records the reason, and still completes
    with ratings from the checkpoint opponents; the exception does not propagate.
    """

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))

    def boom():
        raise RuntimeError("Compiled minimax_cpp extension not found")

    arena = _StubArena(per_score=2, sealbot_raises=boom)
    # Drive the run directly so a propagating exception fails the test.
    rep = mse.run_multistage_eval(
        run, run / "checkpoints" / "epoch_000040.pt", _no_sprt_config(),
        candidate_epoch=40, write_diagnostics=False,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
    )

    assert rep["ratings"]["fit"]["converged"] is True
    assert rep["verdict"]["label"] in {"PROMOTE", "REGRESS", "INCONCLUSIVE"}
    stage_c = next(s for s in rep["stages"] if s["stage"] == "C_deep")
    assert stage_c["status"] == "completed"
    assert "minimax_cpp" in stage_c["sealbot_unavailable"]
    assert mse.SEALBOT_LABEL not in {e["opponent"] for e in rep["edges"]}
    assert rep["sealbot_winrate_ci95"] is None
    # The pool anchored on a checkpoint, not on the missing SealBot.
    assert rep["ratings"]["fit"]["anchor"] != mse.SEALBOT_LABEL
    assert rep["ratings"]["fit"]["anchor_is_sealbot"] is False


def test_sealbot_import_error_is_also_failed_open(tmp_path: Path) -> None:
    """The adapter imports ``hexo_engine`` at module top, so a CPU-only box can
    raise ImportError/ModuleNotFoundError before the availability check. The
    fail-open catches that too, not just SealBotUnavailableError.
    """

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))

    def boom_import():
        raise ModuleNotFoundError("No module named 'hexo_engine'")

    arena = _StubArena(per_score=2, sealbot_raises=boom_import)
    rep = _run(run, 40, arena)
    assert rep["ratings"]["fit"]["converged"] is True
    stage_c = next(s for s in rep["stages"] if s["stage"] == "C_deep")
    assert stage_c["status"] == "completed"
    assert "ModuleNotFoundError" in stage_c["sealbot_unavailable"]


def test_anchor_pins_bc_prefit_when_sealbot_disabled(tmp_path: Path) -> None:
    """With SealBot disabled, the BT fit anchors on bc_prefit; ratings exist."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _no_sprt_config(opponents={"sealbot_enabled": False})
    rep = _run(run, 40, _StubArena(per_score=2), config=cfg)

    assert rep["ratings"]["fit"]["converged"] is True
    assert rep["ratings"]["fit"]["anchor"] == "bc_prefit"
    assert rep["ratings"]["fit"]["anchor_is_sealbot"] is False
    assert rep["ratings"]["players"], "no pooled ratings produced"
    assert rep["verdict"]["label"] in {"PROMOTE", "REGRESS", "INCONCLUSIVE"}


def test_anchor_falls_back_to_lowest_checkpoint_when_no_anchors(tmp_path: Path) -> None:
    """SealBot off and no resolvable bc_prefit/ep5: the lowest available checkpoint
    opponent (by epoch) anchors the pool. The pool has an anchor whenever any
    checkpoint edge exists."""

    run = _make_run(tmp_path, epochs=(10, 20, 40), bc=False)  # no BC, no ep5 file
    cfg = _no_sprt_config(
        opponents={
            "sealbot_enabled": False,
            "permanent_anchors": (
                ("bc_prefit", "runs/__no_such_bc_dir__/checkpoint_epoch2.pt"),
                ("ep5", "epoch_000005.pt"),
            ),
        }
    )
    rep = _run(run, 40, _StubArena(per_score=2), config=cfg)

    assert rep["ratings"]["fit"]["converged"] is True
    # Nearest log-grid rungs below 40 are {10, 20}; the lowest played is ep10.
    assert rep["ratings"]["fit"]["anchor"] == "ep10"
    assert rep["ratings"]["players"]


def test_choose_anchor_preference_order() -> None:
    """Preference order SealBot > bc_prefit > lowest-epoch checkpoint > any, and
    only ever a label that appears in an edge. Returns None only when there is no
    non-candidate edge."""

    cand = "cand_ep40"
    roster = mse.Roster(
        candidate_label=cand, candidate_epoch=40,
        sealbot=mse.Opponent(mse.SEALBOT_LABEL, "sealbot", None, None),
        champion=mse.Opponent("ep20", "champion", Path("e"), 20),
        opponents=(
            mse.Opponent("bc_prefit", "anchor", Path("b"), 2),
            mse.Opponent("ep5", "anchor", Path("c"), 5),
            mse.Opponent("ep10", "bracket", Path("d"), 10),
            mse.Opponent("ep20", "champion", Path("e"), 20),
        ),
    )
    BT = mse.eval_stats.BTEdge

    # SealBot present -> SealBot.
    assert mse._choose_anchor(
        [BT(cand, mse.SEALBOT_LABEL, 10, 10, 0.5), BT(cand, "bc_prefit", 10, 10, 1.0)], roster
    ) == mse.SEALBOT_LABEL
    # No SealBot edge -> bc_prefit before other anchors.
    assert mse._choose_anchor(
        [BT(cand, "bc_prefit", 10, 10, 1.0), BT(cand, "ep5", 10, 10, 1.0)], roster
    ) == "bc_prefit"
    # No SealBot/bc_prefit -> lowest-epoch checkpoint with an edge.
    assert mse._choose_anchor(
        [BT(cand, "ep20", 10, 10, 1.0), BT(cand, "ep10", 10, 10, 1.0)], roster
    ) == "ep10"
    # No edges at all -> None.
    assert mse._choose_anchor([], roster) is None
    # Only a candidate self-edge (no non-candidate label) -> None.
    assert mse._choose_anchor([BT(cand, cand, 1, 1, 1.0)], roster) is None


# =========================================================================== #
# 6. Pure-eval invariant: gating/promotion OFF, verdict mutates no run state.
# =========================================================================== #
def test_gating_and_promotion_default_off() -> None:
    cfg = MultiStageEvalSection()
    assert cfg.eval_gating_enabled is False
    assert cfg.eval_promotion_enabled is False
    # The tripwire passes for the default config.
    mse._assert_no_run_mutation(cfg)


def test_assert_no_run_mutation_fires_when_a_knob_is_flipped() -> None:
    """The tripwire fires when a gating/promotion knob is turned on, before any
    game runs."""

    base = MultiStageEvalSection()
    with pytest.raises(AssertionError, match="eval_gating_enabled must be False"):
        mse._assert_no_run_mutation(dataclasses.replace(base, eval_gating_enabled=True))
    with pytest.raises(AssertionError, match="eval_promotion_enabled must be False"):
        mse._assert_no_run_mutation(dataclasses.replace(base, eval_promotion_enabled=True))


def test_report_advertises_pure_eval_and_off_knobs(tmp_path: Path) -> None:
    """Every report states it is pure eval with gating/promotion off."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 40, arena)
    assert rep["meta"]["pure_eval"] is True
    assert rep["meta"]["gating_enabled"] is False
    assert rep["meta"]["promotion_enabled"] is False
    assert "gates nothing" in rep["verdict"]["note"]


def test_write_diagnostics_false_writes_nothing(tmp_path: Path) -> None:
    """With ``write_diagnostics=False`` the run writes no pool, no diagnostics
    JSON, and no directory anywhere under the run tree."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    arena = _FakeArena(ckpt_scorer=lambda lb, n: _scores_for_winrate(0.75, n), sealbot_winrate=0.7)
    rep = _run(run, 40, arena)  # _run passes write_diagnostics=False
    assert rep["verdict"]["label"] == "PROMOTE"  # verdict is still computed
    assert not (run / "diagnostics").exists()     # ...but nothing is written
    assert "diagnostics_path" not in rep["meta"]


def test_changing_the_verdict_touches_no_run_state(tmp_path: Path) -> None:
    """Flipping the verdict from PROMOTE to REGRESS leaves the run tree unchanged
    apart from the eval-only pool/diagnostics: no checkpoint write, no flag, no
    run-state edit keyed on the verdict."""

    def run_with(strength: float, winrate: float, subdir: str) -> tuple[str, set[Path]]:
        run = _make_run(tmp_path / subdir, epochs=(5, 10, 20, 40))
        before = {p.relative_to(run) for p in run.rglob("*")}
        arena = _FakeArena(
            ckpt_scorer=lambda lb, n, _s=strength: _scores_for_winrate(_s, n), sealbot_winrate=winrate
        )
        rep = mse.run_multistage_eval(
            run, run / "checkpoints" / "epoch_000040.pt", _no_sprt_config(),
            candidate_epoch=40, write_diagnostics=True,
            play_checkpoint_match=arena.play_checkpoint_match,
            play_sealbot_match=arena.play_sealbot_match,
        )
        after = {p.relative_to(run) for p in run.rglob("*")}
        return rep["verdict"]["label"], after - before

    promote_label, promote_new = run_with(0.75, 0.7, "promote_run")
    regress_label, regress_new = run_with(0.25, 0.3, "regress_run")

    assert promote_label == "PROMOTE"
    assert regress_label == "REGRESS"
    eval_only = {
        Path("diagnostics"),
        Path("diagnostics") / "eval_pool.json",
        Path("diagnostics") / "hexfield.multistage_eval.epoch_000040.json",
    }
    assert promote_new == eval_only
    assert regress_new == eval_only
    for new_set in (promote_new, regress_new):
        assert not any(p.suffix == ".pt" for p in new_set)
        assert not any(p.suffix == ".flag" for p in new_set)


# =========================================================================== #
# 7. Run-in-parts: per-opponent parts, resume, durability, parts == monolithic.
# =========================================================================== #
def _cfg_parts(**overrides):
    """A no-SPRT config (Stage B off) so the parts tests isolate Stage C/D."""

    return parse_hexfield_config({"multi_stage_eval": {"sprt": {"enabled": False}, **overrides}})


def _pool_doc_for(run: Path, cfg) -> dict:
    section = mse._coerce_section(cfg)
    return mse._load_pool(mse._pool_path(run, section))


def _pooled_edge_keys(pool_doc: dict) -> set:
    """The (epoch, a, b) identity of every edge row in the pool."""

    return {
        (row.get("epoch"), row.get("a"), row.get("b"))
        for row in pool_doc.get("edges", [])
    }


def _opponent_labels(run: Path, candidate_epoch: int, cfg) -> list[str]:
    """The checkpoint-opponent labels of the roster (the per-opponent parts)."""

    section = mse._coerce_section(cfg)
    roster = mse.select_opponents(
        run, _candidate(run, candidate_epoch), section, candidate_epoch=candidate_epoch
    )
    return [o.label for o in roster.opponents]


def test_parts_api_surface_exists() -> None:
    """The three parts entry points + the resume predicate are public."""

    for name in (
        "run_eval_part",
        "aggregate_pool",
        "run_multistage_eval_in_parts",
        "_epoch_edge_exists",
    ):
        assert hasattr(mse, name), f"multistage_eval is missing {name}"
        assert callable(getattr(mse, name)), f"{name} is not callable"


def test_run_eval_part_appends_single_edge(tmp_path: Path) -> None:
    """One part = one opponent -> exactly one edge row appended to the pool, keyed
    by (epoch, candidate_label, opponent_label), persisted to disk immediately."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _cfg_parts()
    arena = _StubArena(per_score=2)
    opp_labels = _opponent_labels(run, 40, cfg)
    assert opp_labels, "expected at least one checkpoint opponent"
    target = opp_labels[0]

    pool_before = _pool_doc_for(run, cfg)
    assert pool_before["edges"] == []

    out = mse.run_eval_part(
        run, _candidate(run, 40), target, cfg,
        candidate_epoch=40,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
    )

    assert out["status"] == "played", out
    assert {label for label, _ in arena.ckpt_calls} == {target}

    pool_after = _pool_doc_for(run, cfg)
    new_keys = _pooled_edge_keys(pool_after) - _pooled_edge_keys(pool_before)
    roster = mse.select_opponents(run, _candidate(run, 40), mse._coerce_section(cfg), candidate_epoch=40)
    assert new_keys == {(40, roster.candidate_label, target)}, new_keys
    assert len(pool_after["edges"]) == len(pool_before["edges"]) + 1


def test_run_eval_part_sealbot_appends_sealbot_edge(tmp_path: Path) -> None:
    """The SealBot zero-point is also a part (label == SEALBOT_LABEL): it plays
    the unpaired SealBot match and appends a down-weighted SealBot edge."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _cfg_parts()
    arena = _StubArena(per_score=2, sealbot_winrate=0.6)

    out = mse.run_eval_part(
        run, _candidate(run, 40), mse.SEALBOT_LABEL, cfg,
        candidate_epoch=40,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
    )

    assert out["status"] == "played", out
    assert arena.sealbot_calls, "SealBot part played no SealBot games"
    assert not arena.ckpt_calls, "SealBot part should not play checkpoint games"

    pool = _pool_doc_for(run, cfg)
    roster = mse.select_opponents(run, _candidate(run, 40), mse._coerce_section(cfg), candidate_epoch=40)
    assert (40, roster.candidate_label, mse.SEALBOT_LABEL) in _pooled_edge_keys(pool)
    sb_rows = [r for r in pool["edges"] if r["b"] == mse.SEALBOT_LABEL]
    assert len(sb_rows) == 1
    assert sb_rows[0]["weight"] < 1.0  # down-weighted


def test_part_skipped_when_edge_already_in_pool(tmp_path: Path) -> None:
    """Re-running a part whose (epoch, a, b) edge is already pooled is SKIPPED on
    resume: it returns status='skipped' and plays ZERO games."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _cfg_parts()
    target = _opponent_labels(run, 40, cfg)[0]

    arena1 = _StubArena(per_score=2)
    first = mse.run_eval_part(
        run, _candidate(run, 40), target, cfg, candidate_epoch=40,
        play_checkpoint_match=arena1.play_checkpoint_match,
        play_sealbot_match=arena1.play_sealbot_match,
    )
    assert first["status"] == "played"
    pooled_after_first = _pooled_edge_keys(_pool_doc_for(run, cfg))

    arena2 = _StubArena(per_score=2)
    second = mse.run_eval_part(
        run, _candidate(run, 40), target, cfg, candidate_epoch=40, resume=True,
        play_checkpoint_match=arena2.play_checkpoint_match,
        play_sealbot_match=arena2.play_sealbot_match,
    )
    assert second["status"] == "skipped", second
    assert arena2.ckpt_calls == [], "skipped part must play NO checkpoint games"
    assert arena2.sealbot_calls == [], "skipped part must play NO sealbot games"
    assert _pooled_edge_keys(_pool_doc_for(run, cfg)) == pooled_after_first


def test_epoch_edge_exists_predicate_is_exact() -> None:
    """The resume predicate matches on ALL THREE of (epoch, a, b)."""

    cand = "cand_ep40"
    pool_doc = {
        "format": mse.POOL_FORMAT, "version": mse.POOL_VERSION, "anchor": mse.SEALBOT_LABEL,
        "edges": [
            {"epoch": 40, "a": cand, "b": "ep20", "wins_a": 10.0, "wins_b": 6.0, "weight": 1.0},
        ],
    }
    assert mse._epoch_edge_exists(pool_doc, 40, cand, "ep20") is True
    assert mse._epoch_edge_exists(pool_doc, 41, cand, "ep20") is False  # diff epoch
    assert mse._epoch_edge_exists(pool_doc, 40, cand, "ep10") is False  # diff opponent
    empty = {"format": mse.POOL_FORMAT, "version": mse.POOL_VERSION, "edges": []}
    assert mse._epoch_edge_exists(empty, 40, cand, "ep20") is False


def test_resume_false_replays_already_pooled_part(tmp_path: Path) -> None:
    """``resume=False`` forces a re-play even when the edge is already pooled; the
    edge then appears twice and compounds in the BT fit."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _cfg_parts()
    target = _opponent_labels(run, 40, cfg)[0]

    arena = _StubArena(per_score=2)
    mse.run_eval_part(
        run, _candidate(run, 40), target, cfg, candidate_epoch=40,
        play_checkpoint_match=arena.play_checkpoint_match,
        play_sealbot_match=arena.play_sealbot_match,
    )
    n_after_first = len(_pool_doc_for(run, cfg)["edges"])

    arena2 = _StubArena(per_score=2)
    out = mse.run_eval_part(
        run, _candidate(run, 40), target, cfg, candidate_epoch=40, resume=False,
        play_checkpoint_match=arena2.play_checkpoint_match,
        play_sealbot_match=arena2.play_sealbot_match,
    )
    assert out["status"] == "played"
    assert arena2.ckpt_calls, "resume=False must re-play the opponent"
    assert len(_pool_doc_for(run, cfg)["edges"]) == n_after_first + 1


def test_interrupted_parts_keep_completed_edges_and_aggregate_fits(tmp_path: Path) -> None:
    """Durability: if a LATER part raises, every EARLIER part's edge is already
    persisted, and ``aggregate_pool`` still fits the BT ratings + a verdict."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _cfg_parts(opponents={"sealbot_enabled": False})  # checkpoint-only roster
    opp_labels = _opponent_labels(run, 40, cfg)
    assert len(opp_labels) >= 2

    for label in opp_labels[:2]:
        arena = _StubArena(per_score=2)
        out = mse.run_eval_part(
            run, _candidate(run, 40), label, cfg, candidate_epoch=40,
            play_checkpoint_match=arena.play_checkpoint_match,
            play_sealbot_match=arena.play_sealbot_match,
        )
        assert out["status"] == "played"

    def boom(*a, **k):
        raise RuntimeError("simulated arena failure mid-part")

    with pytest.raises(RuntimeError):
        mse.run_eval_part(
            run, _candidate(run, 40), opp_labels[2], cfg, candidate_epoch=40,
            play_checkpoint_match=boom,
            play_sealbot_match=boom,
        )

    roster = mse.select_opponents(run, _candidate(run, 40), mse._coerce_section(cfg), candidate_epoch=40)
    keys = _pooled_edge_keys(_pool_doc_for(run, cfg))
    for label in opp_labels[:2]:
        assert (40, roster.candidate_label, label) in keys
    assert (40, roster.candidate_label, opp_labels[2]) not in keys

    agg = mse.aggregate_pool(run, _candidate(run, 40), cfg, candidate_epoch=40)
    assert agg["ratings"]["fit"].get("converged") is True
    assert agg["verdict"]["label"] in {"PROMOTE", "REGRESS", "INCONCLUSIVE"}


def test_aggregate_only_plays_no_games_and_fits(tmp_path: Path) -> None:
    """``aggregate_pool`` reads the persisted pool and runs ONLY the BT fit +
    verdict — it plays NO games (no arena seam), and is idempotent."""

    run = _make_run(tmp_path, epochs=(5, 10, 20, 40))
    cfg = _cfg_parts(opponents={"sealbot_enabled": False})

    for label in _opponent_labels(run, 40, cfg):
        arena = _StubArena(per_score=2)
        mse.run_eval_part(
            run, _candidate(run, 40), label, cfg, candidate_epoch=40,
            play_checkpoint_match=arena.play_checkpoint_match,
            play_sealbot_match=arena.play_sealbot_match,
        )

    pool_before = _pooled_edge_keys(_pool_doc_for(run, cfg))

    # Structural proof it plays NOTHING: the signature has no game-playing seam.
    agg_params = set(inspect.signature(mse.aggregate_pool).parameters)
    assert "play_checkpoint_match" not in agg_params and "play_sealbot_match" not in agg_params, (
        "aggregate_pool must be a pure fit pass with no game-playing seam"
    )

    agg = mse.aggregate_pool(run, _candidate(run, 40), cfg, candidate_epoch=40)
    assert agg["ratings"]["fit"].get("converged") is True
    assert agg["ratings"]["players"], "no pooled ratings produced"
    assert agg["verdict"]["primary"] is not None
    assert agg["verdict"]["primary"]["champion"] == "ep20"  # lag-5 reference of ep40

    agg2 = mse.aggregate_pool(run, _candidate(run, 40), cfg, candidate_epoch=40)
    assert _pooled_edge_keys(_pool_doc_for(run, cfg)) == pool_before
    assert agg2["verdict"]["label"] == agg["verdict"]["label"]


def test_parts_path_pool_equals_monolithic_pool(tmp_path: Path) -> None:
    """Running the roster as a sequence of parts (``run_multistage_eval_in_parts``)
    pools the SAME edge set, with the same effective counts, as the monolithic
    ``run_multistage_eval`` over the same roster + stub arena."""

    cfg = _cfg_parts(opponents={"sealbot_enabled": False})

    run_mono = _make_run(tmp_path / "mono", epochs=(5, 10, 20, 40))
    arena_mono = _StubArena(per_score=2)
    mse.run_multistage_eval(
        run_mono, _candidate(run_mono, 40), cfg, candidate_epoch=40,
        write_diagnostics=True,
        play_checkpoint_match=arena_mono.play_checkpoint_match,
        play_sealbot_match=arena_mono.play_sealbot_match,
    )
    mono_pool = _pool_doc_for(run_mono, cfg)

    run_parts = _make_run(tmp_path / "parts", epochs=(5, 10, 20, 40))
    arena_parts = _StubArena(per_score=2)
    mse.run_multistage_eval_in_parts(
        run_parts, _candidate(run_parts, 40), cfg, candidate_epoch=40,
        play_checkpoint_match=arena_parts.play_checkpoint_match,
        play_sealbot_match=arena_parts.play_sealbot_match,
    )
    parts_pool = _pool_doc_for(run_parts, cfg)

    assert _pooled_edge_keys(parts_pool) == _pooled_edge_keys(mono_pool)

    def _counts_by_key(doc):
        out = {}
        for r in doc["edges"]:
            out[(r["epoch"], r["a"], r["b"])] = (
                round(float(r["wins_a"]), 6), round(float(r["wins_b"]), 6), round(float(r["weight"]), 6)
            )
        return out

    assert _counts_by_key(parts_pool) == _counts_by_key(mono_pool)


def test_parts_orchestrator_resumes_after_partial_completion(tmp_path: Path) -> None:
    """End-to-end resume: a parts orchestration that completes only some opponents
    (one raises) leaves the rest pooled; a SECOND in-parts run skips the done
    opponents and finishes the remaining one, and the final pool equals a clean
    single in-parts run."""

    cfg = _cfg_parts(opponents={"sealbot_enabled": False})

    run_ref = _make_run(tmp_path / "ref", epochs=(5, 10, 20, 40))
    arena_ref = _StubArena(per_score=2)
    mse.run_multistage_eval_in_parts(
        run_ref, _candidate(run_ref, 40), cfg, candidate_epoch=40,
        play_checkpoint_match=arena_ref.play_checkpoint_match,
        play_sealbot_match=arena_ref.play_sealbot_match,
    )
    ref_keys = _pooled_edge_keys(_pool_doc_for(run_ref, cfg))

    run = _make_run(tmp_path / "interrupted", epochs=(5, 10, 20, 40))
    opp_labels = _opponent_labels(run, 40, cfg)
    last = opp_labels[-1]

    class _PartialArena(_StubArena):
        def play_checkpoint_match(self, a, b, n, **kw):
            if kw["label_b"] == last:
                raise RuntimeError(f"simulated failure on {last}")
            return super().play_checkpoint_match(a, b, n, **kw)

    arena_fail = _PartialArena(per_score=2)
    # The in-parts orchestrator may surface the failure or fail-open per part;
    # either way the COMPLETED parts must be durably pooled.
    try:
        mse.run_multistage_eval_in_parts(
            run, _candidate(run, 40), cfg, candidate_epoch=40,
            play_checkpoint_match=arena_fail.play_checkpoint_match,
            play_sealbot_match=arena_fail.play_sealbot_match,
        )
    except RuntimeError:
        pass

    roster = mse.select_opponents(run, _candidate(run, 40), mse._coerce_section(cfg), candidate_epoch=40)
    keys_after_partial = _pooled_edge_keys(_pool_doc_for(run, cfg))
    for label in opp_labels[:-1]:
        assert (40, roster.candidate_label, label) in keys_after_partial
    assert (40, roster.candidate_label, last) not in keys_after_partial

    arena_resume = _StubArena(per_score=2)
    mse.run_multistage_eval_in_parts(
        run, _candidate(run, 40), cfg, candidate_epoch=40, resume=True,
        play_checkpoint_match=arena_resume.play_checkpoint_match,
        play_sealbot_match=arena_resume.play_sealbot_match,
    )
    assert {label for label, _ in arena_resume.ckpt_calls} == {last}, arena_resume.ckpt_calls
    assert _pooled_edge_keys(_pool_doc_for(run, cfg)) == ref_keys


def test_sealbot_edge_candidate_searches_as_trained_profile(tmp_path) -> None:
    """The SealBot edge measures the candidate under its own as-trained
    searcher — no PUCT override even on a gumbel run. Budget calibration of
    the candidate count happens in-tree (init_gumbel_root walks gumbel_m down
    the halving ladder for the eval budget), not via a per-match profile, so
    the searcher regime is recorded on the edge's provenance instead."""
    from hexfield_eval_kit import _sealbot_match

    cfg = parse_hexfield_config(
        {
            "selfplay": {
                "gumbel_target_enabled": True,
                "gumbel_root_enabled": True,
                "gumbel_sequential_halving": True,
                "gumbel_nonroot_select": True,
            }
        }
    )
    captured: dict = {}

    def fake_sealbot(ckpt, n, **kw):
        captured.update(kw)
        return _sealbot_match("cand", n, 0.9)

    roster = mse.Roster(
        candidate_label="cand_ep50",
        candidate_epoch=50,
        sealbot=mse.Opponent(label="sealbot", role="sealbot", ckpt=None),
        champion=None,
        opponents=(),
    )
    ck = tmp_path / "run" / "checkpoints" / "epoch_000050.pt"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"")
    edge, _ci, unavail = mse._play_sealbot_opponent(
        cfg.multi_stage_eval, roster, ck, cfg, 8,
        play_sealbot_match=fake_sealbot, diagnostics_dir=tmp_path,
    )
    assert unavail is None and edge is not None
    assert "divergence_overrides" not in captured, (
        "candidate must search SealBot with its as-trained profile"
    )
    prov = edge["descriptive"]["provenance"]
    assert prov.get("candidate_search_profile") == "selfplay"
