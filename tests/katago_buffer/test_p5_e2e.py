"""Phase 5 — train_passes serial consumer rewrite + determinism (PLAN §3.4/§4.4/§4.5/§6).

CPU-ONLY, no GPU, no live-run interaction. Reads SYNTHESIZED shards under
``_scratch/p5/samples`` — the session-scoped autouse fixture in conftest.py
generates several epoch dirs of games there if the tree is empty (replacing the
retired setup script copy of the private development-run live
tree). Every write lands under ``_scratch/``.

What this verifies (each block prints its own line; ``main`` prints PASS):

  1. SHIM PARITY — ``_row_view_to_sample(load_packed_shard(p).row_view(i))``
     expands field-identical (support nodes, feats, policy, opp, stvalue,
     moves_left, value) to ``read_compact_shard(p)[i]`` for every row of a real
     shard, across several fixed D6 symmetries. This pins the PackedRowView ->
     ShrimpSampleData shim that feeds the unchanged ``expand_sample`` (PLAN §6).

  2. DETERMINISM — the pre-drawn per-row D6 vector and the survivor permutation
     are pure functions of (run_seed, epoch): identical across two draws AND the
     full select->train pipeline run twice (same seed) yields a bit-identical
     trained-row order, loss, and grad-norm trace (PLAN §4.4/§4.5). A DIFFERENT
     seed yields a different order (the augmentation actually varies).

  3. END-TO-END — ShrimpNet on CPU + AdamW + ShrimpTrainer:
     ``select_training_samples`` then ``train_passes`` for 2 epochs against the
     copied main_2 samples. Asserts: status completed, steps>0, finite loss,
     finite grad norm, the training diagnostics json is written and carries the
     new Phase-5 fields (reuse_ratio, window_rows, train_bucket_level,
     rows_skipped_off_legal), and the survivor truncation honors effective_rows.

  4. EMPTY-WINDOW — an empty PackedWindow short-circuits to status=skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from shrimp.config import ShrimpConfig, TrainingSection
from shrimp.model import ShrimpNet
from shrimp.samples import expand_sample
from shrimp.shards import read_compact_shard
from shrimp.trainer import (
    D6_SIZE,
    ShrimpTrainer,
    _aug_seed,
    _perm_seed,
    _row_view_to_sample,
)
from shrimp.window import PackedWindow, load_packed_shard

SCRATCH = Path(__file__).resolve().parent / "_scratch" / "p5"
SAMPLES = SCRATCH / "samples"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build_trainer(seed_unused: int = 0, **training) -> ShrimpTrainer:
    base = dict(
        # SMALL window + cap so the CPU run is quick while still exercising the
        # full path (taper + keep_prob subsample + governor + truncation + a
        # multi-nominal-batch micro-bucket loop). The task spec calls for a small
        # keep_target / train_samples_per_epoch; these are sized for a fast CPU
        # gate (~300 trained rows / epoch over batch_rows=16 => ~19 steps).
        shuffle_min_rows=200,
        shuffle_taper_window_scale=200.0,
        shuffle_keep_target_rows=1500,
        train_samples_per_epoch=80,
        max_train_bucket_size=500_000.0,
        max_train_bucket_per_new_data=8.0,
        batch_rows=16,
    )
    base.update(training)
    cfg = ShrimpConfig(device="cpu", training=TrainingSection(**base))
    model = ShrimpNet()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "bias_table" not in name and name != "tokens":
            decay.append(p)
        else:
            no_decay.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.training.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.training.learning_rate,
    )
    return ShrimpTrainer(model=model, config=cfg, optimizer=opt)


def _ctx(samples_dir: Path, diag_dir: Path, seed: int) -> SimpleNamespace:
    diag_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        config=SimpleNamespace(run=SimpleNamespace(seed=seed)),
        samples_dir=samples_dir,
        diagnostics_dir=diag_dir,
    )


def _components() -> SimpleNamespace:
    # NB: the production SharedComponents is slotted; tests use a SimpleNamespace
    # so attributes can be set freely (matches the Phase-4 verifier pattern). The
    # consumer reads only ``shared.sample_window`` / ``shared.sample_symmetries``.
    return SimpleNamespace(shared=SimpleNamespace(sample_window=None, sample_symmetries=None))


# ---------------------------------------------------------------------------
# 1. shim parity: PackedRowView -> ShrimpSampleData == read_compact_shard
# ---------------------------------------------------------------------------
def test_shim_parity() -> None:
    npzs = sorted(SAMPLES.glob("epoch_*/game_*.npz"))[:4]
    assert npzs, "no scratch shards"
    checked_rows = 0
    for npz in npzs:
        oracle_rows = read_compact_shard(npz)  # list[ShrimpSampleData]
        packed = load_packed_shard(npz)
        assert packed.n == len(oracle_rows), (npz.name, packed.n, len(oracle_rows))
        for i in range(packed.n):
            shim = _row_view_to_sample(packed.row_view(i))
            oracle = oracle_rows[i]
            # The shim must reproduce every field expand_sample reads.
            assert shim.turn_index == oracle.turn_index
            assert shim.current_player == oracle.current_player
            assert shim.phase == oracle.phase, (npz.name, i, shim.phase, oracle.phase)
            assert shim.records == oracle.records, (npz.name, i)
            assert shim.first_stone == oracle.first_stone
            assert shim.own_hot == oracle.own_hot
            assert shim.opp_hot == oracle.opp_hot
            assert shim.own_win == oracle.own_win
            assert shim.opp_win == oracle.opp_win
            assert shim.policy == oracle.policy
            assert shim.opp_policy == oracle.opp_policy
            assert float(shim.value) == float(oracle.value)
            assert shim.short_term_value == oracle.short_term_value
            assert float(shim.moves_left) == float(oracle.moves_left)
            # And the EXPANSION must be element-identical across several D6 syms.
            for sym in (0, 1, 5, 7, 11):
                es = expand_sample(shim, symmetry=sym)
                eo = expand_sample(oracle, symmetry=sym)
                assert es.support.num_nodes == eo.support.num_nodes, (npz.name, i, sym)
                assert es.support.legal_count == eo.support.legal_count
                assert np.array_equal(es.support.nbr, eo.support.nbr)
                assert np.array_equal(es.support.coords, eo.support.coords)
                assert np.allclose(es.feats, eo.feats), (npz.name, i, sym)
                assert np.allclose(es.policy, eo.policy)
                assert np.allclose(es.opp_policy, eo.opp_policy)
                assert abs(es.opp_coverage - eo.opp_coverage) < 1e-9
                assert np.allclose(es.stvalue, eo.stvalue)
                assert np.allclose(es.stvalue_mask, eo.stvalue_mask)
                assert abs(es.moves_left - eo.moves_left) < 1e-9
                assert abs(es.moves_left_mask - eo.moves_left_mask) < 1e-9
                assert abs(es.value - eo.value) < 1e-9
            checked_rows += 1
    print(f"  1. SHIM PARITY: {checked_rows} rows x 5 syms expand field-identical to "
          "read_compact_shard (PackedRowView->ShrimpSampleData shim is faithful)")


# ---------------------------------------------------------------------------
# 2a. the pre-drawn vectors are pure functions of (seed, epoch)
# ---------------------------------------------------------------------------
def test_predrawn_vectors_deterministic() -> None:
    seed, epoch, n = 13, 4, 137
    d6_a = np.random.default_rng(_aug_seed(seed, epoch)).integers(0, D6_SIZE, size=n, dtype=np.int64)
    d6_b = np.random.default_rng(_aug_seed(seed, epoch)).integers(0, D6_SIZE, size=n, dtype=np.int64)
    assert np.array_equal(d6_a, d6_b), "D6 vector not reproducible for same (seed, epoch)"
    assert d6_a.min() >= 0 and d6_a.max() < D6_SIZE
    # different epoch -> different draw (re-randomized each epoch)
    d6_c = np.random.default_rng(_aug_seed(seed, epoch + 1)).integers(0, D6_SIZE, size=n, dtype=np.int64)
    assert not np.array_equal(d6_a, d6_c), "D6 vector did not re-randomize across epochs"

    perm_a = np.random.default_rng(_perm_seed(seed, epoch)).permutation(n)
    perm_b = np.random.default_rng(_perm_seed(seed, epoch)).permutation(n)
    assert np.array_equal(perm_a, perm_b), "survivor permutation not reproducible"
    # the D6 and permutation streams are independent (distinct seed folds)
    assert _aug_seed(seed, epoch) != _perm_seed(seed, epoch)
    print("  2a. pre-drawn D6 vector + survivor permutation are reproducible pure functions "
          "of (seed, epoch), independent streams, re-randomized per epoch")


# ---------------------------------------------------------------------------
# 2b. full pipeline determinism: same seed -> identical survivor order/loss/grads
# ---------------------------------------------------------------------------
def _run_one_epoch(samples_dir: Path, diag_dir: Path, seed: int, *, model_seed: int, epoch: int = 1):
    torch.manual_seed(model_seed)  # identical initial weights across runs
    tr = _build_trainer()
    ctx = _ctx(samples_dir, diag_dir, seed)
    comp = _components()
    sel = tr.select_training_samples(ctx=ctx, components=comp, epoch=epoch)
    assert sel["status"] == "completed", sel
    out = tr.train_passes(
        passes=1,
        sample_window=comp.shared.sample_window,
        sample_symmetries=comp.shared.sample_symmetries,
        ctx=ctx,
        components=comp,
        epoch=epoch,
    )
    return sel, out


def test_pipeline_determinism() -> None:
    d1 = SCRATCH / "diag_det1"
    d2 = SCRATCH / "diag_det2"
    d3 = SCRATCH / "diag_det3"
    sel1, out1 = _run_one_epoch(SAMPLES, d1, seed=99, model_seed=7)
    sel2, out2 = _run_one_epoch(SAMPLES, d2, seed=99, model_seed=7)
    sel3, out3 = _run_one_epoch(SAMPLES, d3, seed=100, model_seed=7)

    # Same seed -> identical window selection + training trace.
    assert sel1["effective_rows"] == sel2["effective_rows"], (sel1["effective_rows"], sel2["effective_rows"])
    assert out1["trained_rows"] == out2["trained_rows"], (out1["trained_rows"], out2["trained_rows"])
    assert out1["steps"] == out2["steps"]
    # Loss + grad-norm bit-identical (deterministic D6 + permutation + truncation
    # + identical init weights => identical forward/backward).
    for k in out1:
        if k.startswith("loss_"):
            assert out1[k] == out2[k], f"loss {k} differs across same-seed runs: {out1[k]} vs {out2[k]}"
    assert out1["grad_norm_mean"] == out2["grad_norm_mean"], (out1["grad_norm_mean"], out2["grad_norm_mean"])
    assert out1["grad_norm_p95"] == out2["grad_norm_p95"]

    # A DIFFERENT run seed changes the D6 vector + survivor permutation, so the
    # training trace must differ (the augmentation genuinely varies). The window
    # math is seed-driven too, so at least the loss/grad trace should move.
    differs = (
        out3.get("grad_norm_mean") != out1.get("grad_norm_mean")
        or any(out3.get(k) != out1.get(k) for k in out1 if k.startswith("loss_"))
    )
    assert differs, "a different seed produced an identical trace (RNG not actually wired)"
    print(f"  2b. same seed -> identical survivor trace (trained_rows={out1['trained_rows']}, "
          f"steps={out1['steps']}, grad_mean={out1['grad_norm_mean']:.6g}); different seed -> different trace")


# ---------------------------------------------------------------------------
# 3. end-to-end: 2 epochs, finite loss, diagnostics + new fields
# ---------------------------------------------------------------------------
def test_e2e_two_epochs() -> None:
    diag = SCRATCH / "diag_e2e"
    torch.manual_seed(123)
    tr = _build_trainer()
    ctx = _ctx(SAMPLES, diag, seed=42)

    for epoch in (1, 2):
        comp = _components()
        sel = tr.select_training_samples(ctx=ctx, components=comp, epoch=epoch)
        assert sel["status"] == "completed", sel
        assert isinstance(comp.shared.sample_window, PackedWindow)
        win_n = comp.shared.sample_window.n
        assert win_n > 0, "window is empty"
        eff = sel["effective_rows"]

        out = tr.train_passes(
            passes=3,  # generic request; KataGo is a single pass
            sample_window=comp.shared.sample_window,
            sample_symmetries=comp.shared.sample_symmetries,
            ctx=ctx,
            components=comp,
            epoch=epoch,
        )
        assert out["status"] == "completed", out
        assert out["passes"] == 1, "KataGo is single-pass"
        assert out["generic_passes_requested"] == 3
        assert out["steps"] > 0, out
        assert out["window_rows"] == win_n
        # Truncation contract (PLAN §3.4/M3 — the load-bearing fidelity point):
        # exactly one pass capped at effective_rows. At radius 8 nothing is skipped
        # off-legal, so survivors == window_rows and trained == min(eff, win_n).
        assert out["rows_skipped_off_legal"] == 0, "radius-8 default should skip nothing"
        assert out["trained_rows"] == min(int(eff), int(win_n)), (
            "truncation did not cap at effective_rows: "
            f"trained={out['trained_rows']} eff={eff} win={win_n}"
        )
        # New Phase-5 diagnostic fields present.
        for field in ("reuse_ratio", "window_rows", "train_bucket_level", "rows_skipped_off_legal"):
            assert field in out, f"missing diagnostic field {field}: {sorted(out)}"
        # Every loss component finite.
        loss_keys = [k for k in out if k.startswith("loss_")]
        assert loss_keys, "no loss components reported"
        for k in loss_keys:
            assert np.isfinite(out[k]), f"non-finite {k} = {out[k]}"
        assert np.isfinite(out["grad_norm_mean"]) and out["grad_norm_mean"] >= 0.0
        assert np.isfinite(out["grad_norm_p95"])

        # Diagnostics json written with the new fields.
        diag_path = diag / f"shrimp.training.epoch_{epoch:06d}.json"
        assert diag_path.exists(), f"training diag not written: {diag_path}"
        disk = json.loads(diag_path.read_text())
        for field in ("reuse_ratio", "window_rows", "train_bucket_level",
                      "rows_skipped_off_legal", "trained_rows", "steps"):
            assert field in disk, f"diag json missing {field}"
        assert disk["status"] == "completed"

        print(f"     epoch {epoch}: window_rows={win_n} effective={eff} trained={out['trained_rows']} "
              f"steps={out['steps']} loss_total~{out.get('loss_total', out.get('loss_policy', 'n/a'))} "
              f"grad_mean={out['grad_norm_mean']:.4g} reuse={out['reuse_ratio']:.3g}")

    print("  3. E2E: 2 epochs trained on CPU, finite loss/grads, single-pass, "
          "diagnostics json carries the new Phase-5 fields, truncation honors effective_rows")


# ---------------------------------------------------------------------------
# 4. empty window short-circuits to skipped
# ---------------------------------------------------------------------------
def test_empty_window_skipped() -> None:
    diag = SCRATCH / "diag_empty"
    tr = _build_trainer()
    ctx = _ctx(SAMPLES, diag, seed=1)
    out = tr.train_passes(
        passes=1,
        sample_window=PackedWindow.empty(),
        sample_symmetries=None,
        ctx=ctx,
        components=_components(),
        epoch=1,
    )
    assert out["status"] == "skipped", out
    assert out["reason"] == "empty sample window"
    # non-PackedWindow (e.g. None) also skips, not crashes.
    out2 = tr.train_passes(
        passes=1, sample_window=None, sample_symmetries=None,
        ctx=ctx, components=_components(), epoch=1,
    )
    assert out2["status"] == "skipped", out2
    print("  4. empty/None sample_window -> status=skipped (no crash, no optimizer step)")


def main() -> int:
    if not SAMPLES.exists() or not any(SAMPLES.glob("epoch_*/game_*.npz")):
        # When run under pytest the conftest autouse fixture synthesizes these;
        # for a direct ``python`` invocation, generate them here.
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _shard_gen import generate_samples_tree

        generate_samples_tree(SAMPLES, epochs=3, games_per_epoch=8, max_plies=24, base_seed=5000)
    torch.use_deterministic_algorithms(True, warn_only=True)
    test_shim_parity()
    test_predrawn_vectors_deterministic()
    test_pipeline_determinism()
    test_e2e_two_epochs()
    test_empty_window_skipped()
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
