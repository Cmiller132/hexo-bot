"""Telemetry upgrade for the shrimp train/select path (diagnostics only).

These tests pin the *instrumentation* added around window building, selection,
and training — none of them assert on training math, only on the diagnostic
fields that were added on top of it:

1. ``window.build_window_split`` fills the optional ``diag`` out-param
   (shards_selected / shards_skipped / skipped_paths / rows_loaded /
   rows_post_thin), including the deliberately-torn-shard case, and leaves the
   returned window byte-for-byte what it was before (skip behaviour unchanged).
2. The select diagnostic (``ShrimpTrainer.select_training_samples`` result +
   the ``shrimp.select.epoch_*.json`` it writes atomically) carries the new
   keys (shards_skipped, skipped_paths, rows_loaded, rows_post_thin,
   window_epoch_span, shards_from_latest_epoch, select_seconds) alongside the
   pre-existing keep_prob / select_request / selected_rows / window_rows.
3. The per-row surprise-weight aggregation the trainer performs
   (mean / max / clamped-count over ``policy_surprise_weights``) is arithmetically
   correct on synthetic weights.

CPU-only, pure IO on tmp dirs — no GPU, no model step, no live run touched. The
fixture style mirrors tests/test_shrimp_durability.py (``_write_shard`` +
torn-npz) and tests/katago_buffer/test_p4_windowmath.py (fake ctx/components +
``_make_trainer`` around the pure select path).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from shrimp.batching import policy_surprise_weights
from shrimp.buffer_manifest import ShardEntry, scan_or_update_manifest
from shrimp.config import ShrimpConfig, TrainingSection
from shrimp.samples import ShrimpSampleData
from shrimp.shards import write_compact_shard
from shrimp.trainer import ShrimpTrainer
from shrimp.window import build_window_split, load_packed_shard


# --- fixtures (mirrors tests/test_shrimp_durability.py) ---------------------


def _sample(turn: int) -> ShrimpSampleData:
    return ShrimpSampleData(
        game_id="",
        turn_index=turn,
        current_player=turn % 2,
        phase="Opening",
        records=((0, 0, 0, 0), (1, -1, 1, 1)),
        first_stone=(1, -1),
        own_hot=((0, 0),),
        opp_hot=((1, -1),),
        own_win=(),
        opp_win=(),
        policy=((5, 0.7), (6, 0.3)),
        opp_policy=((7, 1.0),),
        q_policy=((5, 0.1), (6, -0.2)),
        gumbel_policy=((5, 0.6), (6, 0.4)),
        prior_logit=((5, 1.2), (6, -0.5)),
        value=0.5 if turn % 2 == 0 else -0.5,
        short_term_value=((2, 0.4),),
        moves_left=float(10 - turn),
        policy_surprise=0.05 * turn,
        metadata={"pcr_full": True},
    )


def _write_shard(samples_dir: Path, epoch: int, idx: int, samples) -> ShardEntry:
    game_key = epoch * 1_000_000 + idx
    rel = f"epoch_{epoch:06d}/game_{game_key}.npz"
    path = samples_dir / rel
    write_compact_shard(path, samples, sidecar={"epoch": epoch})
    return ShardEntry(rel_path=rel, rows=len(samples), generation=epoch, game_key=game_key)


# --- (a) build_window_split fills the diag dict -------------------------------


def test_build_window_diag_all_good(tmp_path) -> None:
    """With no torn shards, the diag reports every survivor loaded, zero skipped,
    and rows_loaded == rows_post_thin == window.n at keep_prob=1.0."""
    samples_dir = tmp_path / "samples"
    a = _write_shard(samples_dir, 1, 0, [_sample(0), _sample(1)])
    b = _write_shard(samples_dir, 1, 1, [_sample(0)])
    c = _write_shard(samples_dir, 2, 0, [_sample(0), _sample(1), _sample(2)])
    total = 2 + 1 + 3

    diag: dict = {}
    win = build_window_split(
        [a, b, c], keep_prob=1.0, rng=np.random.default_rng(0),
        samples_dir=samples_dir, diag=diag,
    )
    assert win.n == total
    assert diag["shards_selected"] == 3
    assert diag["shards_skipped"] == 0
    assert diag["skipped_paths"] == []
    assert diag["rows_loaded"] == total
    assert diag["rows_post_thin"] == total == win.n


def test_build_window_diag_reports_torn_shard(tmp_path) -> None:
    """A torn npz (garbage bytes, intact sidecar) is skipped with a RuntimeWarning
    AND surfaced in the diag: shards_skipped==1, its path in skipped_paths,
    rows_loaded counts only survivors, rows_post_thin == window.n."""
    samples_dir = tmp_path / "samples"
    good1 = _write_shard(samples_dir, 1, 0, [_sample(0), _sample(1)])
    bad = _write_shard(samples_dir, 1, 1, [_sample(0), _sample(1), _sample(2)])
    good2 = _write_shard(samples_dir, 1, 2, [_sample(0)])

    # Corrupt the middle shard's npz, leaving its sidecar intact.
    (samples_dir / bad.rel_path).write_bytes(b"not a valid npz -- torn by a power cut")
    with pytest.raises(Exception):
        load_packed_shard(samples_dir / bad.rel_path)

    diag: dict = {}
    with pytest.warns(RuntimeWarning, match="unreadable shard"):
        win = build_window_split(
            [good1, bad, good2], keep_prob=1.0, rng=np.random.default_rng(0),
            samples_dir=samples_dir, diag=diag,
        )

    # 2 + 1 survivor rows; the torn shard's 3 rows are dropped (skip unchanged).
    assert win.n == 3
    assert diag["shards_selected"] == 2
    assert diag["shards_skipped"] == 1
    assert diag["skipped_paths"] == [str(samples_dir / bad.rel_path)]
    assert diag["rows_loaded"] == 3  # only the two survivors' rows
    assert diag["rows_post_thin"] == 3 == win.n


def test_build_window_diag_skipped_paths_capped_at_20(tmp_path) -> None:
    """skipped_paths caps at 20 entries while shards_skipped keeps the true count."""
    samples_dir = tmp_path / "samples"
    entries = []
    for i in range(25):
        e = _write_shard(samples_dir, 1, i, [_sample(0)])
        (samples_dir / e.rel_path).write_bytes(b"torn")  # corrupt every shard
        entries.append(e)

    diag: dict = {}
    with pytest.warns(RuntimeWarning):
        win = build_window_split(
            entries, keep_prob=1.0, rng=np.random.default_rng(0),
            samples_dir=samples_dir, diag=diag,
        )
    assert win.n == 0
    assert diag["shards_skipped"] == 25          # true count uncapped
    assert len(diag["skipped_paths"]) == 20      # path list capped
    assert diag["shards_selected"] == 0


def test_build_window_diag_optional_backward_compatible(tmp_path) -> None:
    """Omitting ``diag`` must behave exactly as before (no crash, same window)."""
    samples_dir = tmp_path / "samples"
    a = _write_shard(samples_dir, 1, 0, [_sample(0), _sample(1)])
    win = build_window_split(
        [a], keep_prob=1.0, rng=np.random.default_rng(0), samples_dir=samples_dir
    )
    assert win.n == 2


# --- (b) select diag carries the new keys -------------------------------------


class _TinyModel(torch.nn.Module):
    """Minimal nn.Module so ShrimpTrainer.__init__ can partition params via
    named_parameters(); the select path never runs a forward/backward pass."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)


def _make_trainer(**training_overrides) -> ShrimpTrainer:
    """CPU trainer whose select path we exercise; the tiny model/optimizer exist
    only to satisfy __init__ — no train_passes/GPU is invoked here."""
    base = dict(
        max_train_bucket_size=500_000.0,
        train_samples_per_epoch=200,
        max_train_bucket_per_new_data=8.0,
        shuffle_min_rows=1,
        shuffle_taper_window_scale=10.0,
        shuffle_keep_target_rows=10_000,
    )
    base.update(training_overrides)
    cfg = ShrimpConfig(device="cpu", training=TrainingSection(**base))
    model = _TinyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return ShrimpTrainer(model=model, config=cfg, optimizer=opt)


def _fake_ctx(samples_dir: Path, diag_dir: Path, seed: int = 7) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(run=SimpleNamespace(seed=seed)),
        samples_dir=samples_dir,
        diagnostics_dir=diag_dir,
    )


def _fake_components() -> SimpleNamespace:
    return SimpleNamespace(shared=SimpleNamespace(sample_window=None))


def test_select_diag_carries_new_keys(tmp_path) -> None:
    """select_training_samples' return dict AND the atomically-written
    shrimp.select.epoch_*.json both carry the new telemetry keys, without
    dropping/renaming any pre-existing key."""
    samples_dir = tmp_path / "samples"
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    # Two producing epochs so window_epoch_span is non-degenerate.
    for i in range(6):
        _write_shard(samples_dir, 1, i, [_sample(0), _sample(1)])
    for i in range(6):
        _write_shard(samples_dir, 2, i, [_sample(0), _sample(1)])
    scan_or_update_manifest(samples_dir)  # writes .buffer_manifest.json

    # min_rows == the whole buffer (24 rows) makes the taper collapse to the floor
    # and the recent-window cut accumulate EVERY shard, so the window spans both
    # epochs (span 1..2, not just the newest). train_samples_per_epoch large enough
    # that file selection also keeps all shards.
    tr = _make_trainer(shuffle_min_rows=24, train_samples_per_epoch=1000)
    ctx = _fake_ctx(samples_dir, diag_dir)
    comp = _fake_components()
    out = tr.select_training_samples(ctx=ctx, components=comp, epoch=3)
    assert out["status"] == "completed", out.get("reason")

    # Pre-existing keys still present (no rename/retype).
    for key in ("keep_prob", "select_request", "selected_rows", "window_rows",
                "effective_rows", "reuse_ratio", "train_bucket_level"):
        assert key in out, f"pre-existing select key {key} vanished"

    # New telemetry keys present and well-typed.
    for key in ("shards_skipped", "skipped_paths", "rows_loaded", "rows_post_thin",
                "window_epoch_span", "shards_from_latest_epoch", "select_seconds"):
        assert key in out, f"new select key {key} missing"
    assert out["shards_skipped"] == 0
    assert out["skipped_paths"] == []
    assert out["rows_loaded"] >= out["window_rows"]  # pre-thin >= post-thin
    assert out["rows_post_thin"] == out["window_rows"]
    span = out["window_epoch_span"]
    assert span["min"] == 1 and span["max"] == 2 and span["epochs"] == 2
    assert 0.0 <= out["shards_from_latest_epoch"] <= 1.0
    assert out["select_seconds"] >= 0.0

    # The diag json was written (atomically -> no leftover tmp) with the same keys.
    diag_path = diag_dir / "shrimp.select.epoch_000003.json"
    assert diag_path.exists()
    assert not (diag_dir / "shrimp.select.epoch_000003.json.tmp").exists()
    disk = json.loads(diag_path.read_text(encoding="utf-8"))
    for key in ("shards_skipped", "skipped_paths", "rows_loaded", "rows_post_thin",
                "window_epoch_span", "shards_from_latest_epoch", "select_seconds",
                "keep_prob", "select_request", "selected_rows", "window_rows"):
        assert key in disk, f"select diag json missing {key}"


def test_select_diag_reports_skipped_shard(tmp_path) -> None:
    """A torn shard inside the selected window is surfaced in the select diag's
    shards_skipped / skipped_paths (previously only a RuntimeWarning)."""
    samples_dir = tmp_path / "samples"
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    written = [_write_shard(samples_dir, 1, i, [_sample(0), _sample(1)]) for i in range(8)]
    scan_or_update_manifest(samples_dir)
    # Corrupt one shard's npz in place (sidecar intact) so the manifest still lists
    # it but build_window_split skips it at load.
    torn = samples_dir / written[3].rel_path
    torn.write_bytes(b"torn by a power cut")

    # min_rows == the whole buffer (8 shards * 2 = 16 rows) makes the recent-window
    # cut accumulate EVERY shard, and the large per-epoch request keeps all of them,
    # guaranteeing the torn shard is selected and then skipped at load.
    tr = _make_trainer(shuffle_min_rows=16, train_samples_per_epoch=10_000)
    ctx = _fake_ctx(samples_dir, diag_dir)
    comp = _fake_components()
    with pytest.warns(RuntimeWarning):
        out = tr.select_training_samples(ctx=ctx, components=comp, epoch=1)
    assert out["status"] == "completed", out.get("reason")
    assert out["shards_skipped"] >= 1
    assert any("game_1000003" in p for p in out["skipped_paths"])
    # window_rows reflects only survivors (< rows_loaded_if_all_present is implied).
    assert out["window_rows"] == comp.shared.sample_window.n


# --- (c) surprise-weight aggregation correctness ------------------------------


def _aggregate_surprise(batches: list[list[float]], uniform_fraction: float,
                        max_weight: float) -> dict:
    """Replicate the trainer's per-batch surprise-weight reduce (trainer.py
    train_passes) over a list of per-batch surprise vectors, so the aggregation
    formula can be pinned without a GPU/model step. Each element is the list of
    policy_surprise values for one nominal batch's policy-valid rows."""
    wsum = 0.0
    wcount = 0
    wmax = 0.0
    clamped = 0
    thresh = float(max_weight)
    for surprises in batches:
        weights, _ = policy_surprise_weights(surprises, uniform_fraction, max_weight)
        if weights:
            wsum += float(sum(weights))
            wcount += len(weights)
            wmax = max(wmax, max(weights))
            clamped += sum(1 for w in weights if w >= thresh - 1e-9)
    return {
        "surprise_weight_mean": wsum / wcount if wcount else 0.0,
        "surprise_weight_max": float(wmax),
        "surprise_weight_clamped_count": int(clamped),
    }


def test_surprise_aggregation_all_zero_surprise() -> None:
    """All-zero surprise -> every weight is 1.0 (policy_surprise_weights contract):
    mean == 1.0, max == 1.0, no clamp fires."""
    agg = _aggregate_surprise([[0.0, 0.0, 0.0], [0.0, 0.0]], uniform_fraction=0.5,
                              max_weight=8.0)
    assert agg["surprise_weight_mean"] == pytest.approx(1.0)
    assert agg["surprise_weight_max"] == pytest.approx(1.0)
    assert agg["surprise_weight_clamped_count"] == 0


def test_surprise_aggregation_mean_is_one_without_clamp() -> None:
    """When no clamp fires, policy_surprise_weights guarantees a per-batch mean of
    1.0; the cross-batch mean over equal-size batches is therefore 1.0 too."""
    b1 = [0.1, 0.4, 0.2, 0.3]
    b2 = [1.0, 0.0, 2.0, 1.0]
    agg = _aggregate_surprise([b1, b2], uniform_fraction=0.5, max_weight=100.0)  # high cap: no clamp
    assert agg["surprise_weight_mean"] == pytest.approx(1.0)
    assert agg["surprise_weight_clamped_count"] == 0
    # Sanity: the max weight equals the largest single weight across both batches.
    w1, _ = policy_surprise_weights(b1, 0.5, 100.0)
    w2, _ = policy_surprise_weights(b2, 0.5, 100.0)
    assert agg["surprise_weight_max"] == pytest.approx(max(max(w1), max(w2)))


def test_surprise_aggregation_counts_clamped() -> None:
    """A low max_weight forces clamps; clamped_count matches the number of weights
    at the cap and max equals the cap."""
    # One highly-peaked row dominates the batch -> its weight wants to exceed 2.0.
    surprises = [10.0, 0.0, 0.0, 0.0]
    max_weight = 2.0
    w, _ = policy_surprise_weights(surprises, 0.5, max_weight)
    expected_clamped = sum(1 for x in w if x >= max_weight - 1e-9)
    assert expected_clamped >= 1, "fixture should force at least one clamp"

    agg = _aggregate_surprise([surprises], uniform_fraction=0.5, max_weight=max_weight)
    assert agg["surprise_weight_clamped_count"] == expected_clamped
    assert agg["surprise_weight_max"] == pytest.approx(max_weight)


def test_surprise_aggregation_empty_batches() -> None:
    """No policy-valid rows anywhere -> zeroed stats, no divide-by-zero."""
    agg = _aggregate_surprise([[], []], uniform_fraction=0.5, max_weight=8.0)
    assert agg["surprise_weight_mean"] == 0.0
    assert agg["surprise_weight_max"] == 0.0
    assert agg["surprise_weight_clamped_count"] == 0


# --- (d) end-to-end: train_passes writes the enriched training diag -----------


# Real replay shards make expand_rows legal at the default radius (our synthetic
# _sample rows carry off-legal policy actions by design and hard-error in
# expansion). Synthesize the same fully-expandable current-schema shards the p5
# e2e gate uses, via the shared katago-buffer shard generator, into a tmp dir.
def _real_shard_source(dest: Path) -> Path:
    """Populate ``dest`` with a handful of synthesized, fully-expandable
    shrimp_compact_v1 shards (the p5/p7 generator's random self-play rows) and
    return it. No live tree or private data required."""

    import sys

    kb = Path(__file__).resolve().parent / "katago_buffer"
    if str(kb) not in sys.path:
        sys.path.insert(0, str(kb))
    from _shard_gen import generate_samples_tree

    generate_samples_tree(dest, epochs=2, games_per_epoch=4, base_seed=4242)
    return dest


def test_train_passes_writes_enriched_training_diag(tmp_path, capsys) -> None:
    """A real (tiny) ShrimpNet CPU train pass over synthesized replay shards:
    the training diag json is written atomically and carries the new keys
    (trained_rows, surprise_weight_*, select_seconds, train_seconds) alongside the
    pre-existing ones, and the one-line summary is printed to stdout. Exercises the
    same select->train path as tests/katago_buffer/test_p5_e2e.py on fully
    expandable, current-schema fixtures."""
    from shrimp.model import ShrimpNet

    samples_dir = _real_shard_source(tmp_path / "samples")
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    scan_or_update_manifest(samples_dir)

    cfg = ShrimpConfig(
        device="cpu",
        training=TrainingSection(
            shuffle_min_rows=200,
            shuffle_taper_window_scale=200.0,
            shuffle_keep_target_rows=10_000,
            train_samples_per_epoch=80,
            max_train_bucket_size=500_000.0,
            max_train_bucket_per_new_data=8.0,
            batch_rows=16,
        ),
    )
    model = ShrimpNet()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    tr = ShrimpTrainer(model=model, config=cfg, optimizer=opt)
    ctx = _fake_ctx(samples_dir, diag_dir, seed=42)
    comp = SimpleNamespace(
        shared=SimpleNamespace(sample_window=None, sample_symmetries=None)
    )

    sel = tr.select_training_samples(ctx=ctx, components=comp, epoch=1)
    assert sel["status"] == "completed", sel
    win = comp.shared.sample_window
    assert win is not None and win.n > 0, "synthesized shards must build a non-empty window"

    out = tr.train_passes(
        passes=3, sample_window=win, sample_symmetries=None,
        ctx=ctx, components=comp, epoch=1,
    )
    assert out["status"] == "completed", out.get("reason")

    # Pre-existing training-diag keys preserved (no rename/retype).
    for key in ("window_rows", "trained_rows", "steps", "seconds", "reuse_ratio",
                "train_bucket_level", "rows_skipped_off_legal", "grad_norm_mean"):
        assert key in out, f"pre-existing training key {key} vanished"

    # New training-diag keys present + well-typed.
    for key in ("surprise_weight_mean", "surprise_weight_max",
                "surprise_weight_clamped_count", "select_seconds", "train_seconds"):
        assert key in out, f"new training key {key} missing"
    # trained_rows == the permutation-truncation length (rows actually consumed):
    # min(effective_rows, survivors), survivors == window minus off-legal skips.
    assert out["trained_rows"] == min(
        int(sel["effective_rows"]), int(win.n) - int(out["rows_skipped_off_legal"])
    )
    # surprise-weight mean over policy-valid rows is ~1.0 (no clamp on real rows at
    # the default max_weight=8.0); max >= mean; clamped-count nonnegative.
    assert 0.5 <= out["surprise_weight_mean"] <= 1.5
    assert out["surprise_weight_max"] >= out["surprise_weight_mean"]
    assert out["surprise_weight_clamped_count"] >= 0
    assert out["select_seconds"] >= 0.0 and out["train_seconds"] >= 0.0

    # Atomic write: diag json present, no leftover tmp, keys on disk.
    diag_path = diag_dir / "shrimp.training.epoch_000001.json"
    assert diag_path.exists()
    assert not (diag_dir / "shrimp.training.epoch_000001.json.tmp").exists()
    disk = json.loads(diag_path.read_text(encoding="utf-8"))
    for key in ("trained_rows", "surprise_weight_mean", "surprise_weight_max",
                "surprise_weight_clamped_count", "select_seconds", "train_seconds"):
        assert key in disk, f"training diag json missing {key}"

    # The one-line human summary was printed to stdout.
    printed = capsys.readouterr().out
    assert f"train epoch 1:" in printed
    assert "surprise mean" in printed and "window" in printed and "skipped" in printed
