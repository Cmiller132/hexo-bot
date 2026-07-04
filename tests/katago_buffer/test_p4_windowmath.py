"""Phase 4 self-test — KataGo taper window + train-bucket governor +
select_training_samples rewrite (PLAN §3.1-3.8, §5, §6).

CPU-only, no GPU, no model, no live-run interaction. The retired
reference oracle and the private development-run live tree are
unavailable publicly, so the window-math gates check hexfield against in-test
FIRST-PRINCIPLES references (reproducing the documented formulas) and the
data-driven gates run on SYNTHESIZED shards (``_shard_gen.generate_samples_tree``).
Every write (``scan_or_update_manifest`` -> ``.buffer_manifest.json``, the
per-epoch select diag, the keep_prob-subsampled window) lands ONLY under
``_scratch/`` — never under ``runs/*``.

Gates:
  1. WINDOW-MATH: ``compute_katago_window_rows`` reproduces the documented
     power-law taper formula (in-test reference), ``keep_prob`` == the
     ``min(target, used)/used`` formula, and ``_md5_path_fraction`` reproduces the
     md5-first-13-hex-digits / 2**52 fraction — across the radius-independent knob
     grid.
  2. RECENT-WINDOW: ``select_recent_window`` picks the NEWEST shards covering
     ``desired_rows`` (overshoot < one shard), re-sorted ascending.
  3. SPLIT + SELECTION: ``_split_by_md5`` partitions on the path md5 cut; the
     ``_select_files_for_rows`` overshoot-skip lands near ``requested_rows`` and
     is deterministic under a fixed rng.
  4. GOVERNOR: ``_update_train_bucket`` accrual / cap / monotone-reload branches
     against the documented governor formula on a scripted row stream.
  5. build_window_split: keep_prob=1.0 keeps every row (decode-parity preserved);
     keep_prob<1.0 yields a deterministic subset whose survivors still decode
     field-identically (the _subset_packed CSR rebuild is correct).
  6. DRY-RUN select_training_samples against synthesized shards with a minimal
     fake ctx/components: a plausible return dict (window_rows>0, effective_rows>0,
     monotone cumulative) AND ``components.shared.sample_window`` is a PackedWindow;
     a second call advances the governor deterministically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Make the shard generator importable regardless of pytest invocation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shard_gen import generate_samples_tree  # noqa: E402

from hexfield import shards as hex_shards
from hexfield.buffer_manifest import ShardEntry, scan_or_update_manifest
from hexfield.trainer import HexfieldTrainer
from hexfield.config import HexfieldConfig, TrainingSection
from hexfield.window import (
    PackedWindow,
    _md5_path_fraction,
    _select_files_for_rows,
    _split_by_md5,
    build_window_split,
    compute_katago_window_rows,
    keep_prob,
    load_packed_shard,
    select_recent_window,
)


# ----------------------------------------------------------------------
# first-principles references (replacing the retired dense_cnn oracle)
# ----------------------------------------------------------------------


def _ref_compute_katago_window_rows(
    usable_rows, *, min_rows, expand_window_per_row, taper_window_exponent, taper_window_scale
) -> int:
    """First-principles port of the documented power-law taper (window.py):

        offset      = taper_window_scale if not None else min_rows
        power_law_x = usable_rows - min_rows + offset
        unscaled    = power_law_x**exp - offset**exp
        scaled      = unscaled / (exp * offset**(exp - 1))
        result      = int(scaled * expand_window_per_row + min_rows)

    Shares no code with the production helper (plain python arithmetic)."""
    offset = float(taper_window_scale if taper_window_scale is not None else min_rows)
    power_law_x = float(usable_rows) - float(min_rows) + offset
    unscaled = power_law_x ** taper_window_exponent - offset ** taper_window_exponent
    scaled = unscaled / (taper_window_exponent * (offset ** (taper_window_exponent - 1.0)))
    return int(scaled * expand_window_per_row + float(min_rows))


def _ref_md5_path_fraction(value: str) -> float:
    """First-principles port: first 13 hex digits of md5(value) / 2**52."""
    import hashlib

    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:13]
    return int("0x" + digest, 16) / float(2**52)


class _FakeEntry:
    """Stand-in carrying just the attrs select_recent_window / _split_by_md5 /
    _select_files_for_rows touch (rows, generation, game_key, rel_path)."""

    def __init__(self, rows, generation, game_key, rel_path):
        self.rows = rows
        self.generation = generation
        self.game_key = game_key
        self.rel_path = rel_path


# ----------------------------------------------------------------------
# 1. window-math vs in-test first-principles references
# ----------------------------------------------------------------------


def test_window_math_parity() -> None:
    # compute_katago_window_rows across a knob grid. The call-site invariant is
    # usable_rows >= min_rows (the hexfield caller clamps max(window, min_rows)),
    # so the grid spans that real domain. The degenerate out-of-domain point
    # (negative base ** fractional exponent -> complex) is checked separately
    # below: hexfield and the reference BOTH raise the identical TypeError there,
    # proof the formula is faithful (not a silent divergence).
    exps = [0.5, 0.65, 0.8, 1.0]
    expands = [0.2, 0.4, 0.7]
    scales = [None, 10_000.0, 20_000.0, 50_000.0]
    min_rows_grid = [1, 20_000, 100_000]
    n = 0
    for mr in min_rows_grid:
        for e in exps:
            for ex in expands:
                for sc in scales:
                    # usable_rows >= min_rows (the real domain) + the floor itself.
                    for ur in [mr, mr + 1, mr + 5_000, mr + 280_000, mr + 980_000, mr + 5_000_000]:
                        kw = dict(
                            min_rows=mr,
                            expand_window_per_row=ex,
                            taper_window_exponent=e,
                            taper_window_scale=sc,
                        )
                        ours = compute_katago_window_rows(ur, **kw)
                        theirs = _ref_compute_katago_window_rows(ur, **kw)
                        assert ours == theirs, (
                            f"compute_katago_window_rows mismatch ur={ur} mr={mr} e={e} "
                            f"ex={ex} sc={sc}: {ours} != {theirs}"
                        )
                        assert isinstance(ours, int)
                        # Monotone non-decreasing in usable_rows at the floor step.
                        assert compute_katago_window_rows(ur, **kw) >= compute_katago_window_rows(
                            mr, **kw
                        )
                        n += 1
    print(f"  compute_katago_window_rows: {n} grid points == first-principles taper reference "
          "(usable>=min domain), monotone non-decreasing")

    # As usable_rows -> min_rows the window collapses to min_rows (clamp floor).
    for mr in (20_000, 100_000):
        assert compute_katago_window_rows(
            mr, min_rows=mr, expand_window_per_row=0.4, taper_window_exponent=0.65,
            taper_window_scale=20_000.0,
        ) == mr, "window must equal min_rows at the floor"

    # Degenerate (out-of-call-domain) point: hexfield and the reference raise the
    # SAME error (negative base ** fractional exponent -> complex -> TypeError).
    deg = dict(min_rows=100_000, expand_window_per_row=0.4, taper_window_exponent=0.65,
               taper_window_scale=10_000.0)
    ours_raised = ref_raised = None
    try:
        compute_katago_window_rows(0, **deg)
    except TypeError as ex:
        ours_raised = type(ex).__name__
    try:
        _ref_compute_katago_window_rows(0, **deg)
    except TypeError as ex:
        ref_raised = type(ex).__name__
    assert ours_raised == ref_raised == "TypeError", (
        f"degenerate-point behavior diverges: ours={ours_raised} ref={ref_raised}"
    )
    print("  compute_katago_window_rows: floor collapses to min_rows; "
          "degenerate point raises identically to the reference")

    # keep_prob == min(target, used)/used formula on a used/target grid.
    kpn = 0
    for used in [1, 100, 50_000, 299_999, 300_000, 300_001, 1_000_000]:
        for target in [1, 50_000, 300_000, 600_000]:
            ours = keep_prob(used, target)
            theirs = min(float(target), float(used)) / float(used)  # documented formula
            assert ours == theirs, f"keep_prob({used},{target}) {ours} != {theirs}"
            assert 0.0 < ours <= 1.0
            kpn += 1
    # used<=0 guard: hexfield defends a zero-divide, returning 1.0.
    assert keep_prob(0, 100) == 1.0
    print(f"  keep_prob: {kpn} (used,target) points == min(target,used)/used formula")

    # _md5_path_fraction == the md5-first-13-hex / 2**52 reference over many paths.
    paths = [f"epoch_{e:06d}/game_{e*1_000_000 + i}.npz" for e in range(1, 30) for i in range(40)]
    paths += ["", "x", "a/b/c.npz", "epoch_000001/game_1.npz"]
    for p in paths:
        ours = _md5_path_fraction(p)
        theirs = _ref_md5_path_fraction(p)
        assert ours == theirs, f"_md5_path_fraction({p!r}) {ours!r} != {theirs!r}"
        assert 0.0 <= ours < 1.0
    print(f"  _md5_path_fraction: {len(paths)} paths == md5-first-13-hex/2**52 reference")


# ----------------------------------------------------------------------
# 2. recent-window cut
# ----------------------------------------------------------------------


def test_select_recent_window() -> None:
    # Ascending-(generation,game_key) entries; each shard 100 rows. Newest first.
    entries = [_FakeEntry(100, g, g * 1_000_000 + i, f"e{g}/g{i}") for g in range(1, 6) for i in range(4)]
    # 20 shards * 100 = 2000 rows total. Ask for 350 -> 4 newest shards (400 rows).
    selected, used = select_recent_window(entries, 350)
    assert used == 400, f"used {used} != 400"
    assert len(selected) == 4, f"selected {len(selected)} != 4"
    # Newest 4 shards are the LAST 4 of the ascending list.
    assert [s.rel_path for s in selected] == [e.rel_path for e in entries[-4:]], "wrong shards"
    # Re-sorted ascending on return.
    keys = [(s.generation, s.game_key) for s in selected]
    assert keys == sorted(keys), "selected not ascending"

    # desired larger than total -> take everything.
    sel_all, used_all = select_recent_window(entries, 999_999)
    assert used_all == 2000 and len(sel_all) == 20
    # empty entries -> empty.
    assert select_recent_window([], 100) == ([], 0)
    print("  select_recent_window: newest-covering cut (used=400 for 4 shards)")


# ----------------------------------------------------------------------
# 3. md5 split + overshoot-skip selection
# ----------------------------------------------------------------------


def test_split_and_selection() -> None:
    entries = [_FakeEntry(100, g, g * 1_000_000 + i, f"epoch_{g:06d}/game_{g*1_000_000+i}.npz")
               for g in range(1, 6) for i in range(6)]

    # validation_fraction=0 -> all-train, empty val (default).
    tr, va = _split_by_md5(entries, validation_fraction=0.0)
    assert len(tr) == len(entries) and va == [], "vf=0 must be all-train"

    # vf>0 -> partition on the path md5 cut (val iff fraction >= 1 - vf), keyed on
    # str(rel_path). Reproduce the cut with the first-principles md5 reference.
    vf = 0.1
    tr2, va2 = _split_by_md5(entries, validation_fraction=vf)
    train_upper = 1.0 - vf
    exp_train = [e.rel_path for e in entries if _ref_md5_path_fraction(str(e.rel_path)) < train_upper]
    exp_val = [e.rel_path for e in entries if _ref_md5_path_fraction(str(e.rel_path)) >= train_upper]
    assert [e.rel_path for e in tr2] == exp_train, "md5 split train partition"
    assert [e.rel_path for e in va2] == exp_val, "md5 split val partition"
    assert len(tr2) + len(va2) == len(entries)
    assert va2, "vf=0.1 should route at least one shard to val (else the cut is untested)"
    print(f"  _split_by_md5: vf=0 all-train; vf=0.1 partition == md5-cut reference "
          f"(train={len(tr2)} val={len(va2)})")

    # overshoot-skip: requested 350, shards of 100 each -> lands near 350 (>=).
    rng = np.random.default_rng(12345)
    sel, rows = _select_files_for_rows(entries, 350, rng)
    assert rows >= 350, f"selection {rows} short of requested 350"
    assert rows <= 350 + 100, f"selection {rows} overshoots by > one shard"
    # deterministic under a fixed seed.
    sel_a, rows_a = _select_files_for_rows(entries, 350, np.random.default_rng(7))
    sel_b, rows_b = _select_files_for_rows(entries, 350, np.random.default_rng(7))
    assert rows_a == rows_b and [s.rel_path for s in sel_a] == [s.rel_path for s in sel_b], \
        "selection not deterministic for fixed seed"
    # requested larger than available -> take all.
    sel_all, rows_all = _select_files_for_rows(entries, 10_000, np.random.default_rng(1))
    assert rows_all == 100 * len(entries) and len(sel_all) == len(entries)
    print(f"  _select_files_for_rows: lands at {rows} for req=350 (<= +1 shard), deterministic")


# ----------------------------------------------------------------------
# 3b. keep_prob double-subsample accounting (bug fix)
# ----------------------------------------------------------------------


def test_keep_prob_selection_accounting() -> None:
    """When keep_prob<1.0, select_training_samples inflates the file-selection
    request by 1/kp so build_window_split's per-row Bernoulli(kp) thinning leaves
    ~requested_rows survivors, and debits/accounts effective_rows = the post-thin
    expectation (min(requested, round(selected*kp))). This mirrors the exact
    arithmetic of the fixed select_training_samples without touching a GPU/model."""
    import math

    # Big window of 100-row shards so the inflated (1/kp) request has headroom.
    entries = [_FakeEntry(100, g, g * 1_000_000 + i, f"epoch_{g:06d}/game_{g*1_000_000+i}.npz")
               for g in range(1, 40) for i in range(40)]  # 1560 shards * 100 = 156_000 rows
    total = sum(e.rows for e in entries)
    requested = 6_000

    # --- kp == 1.0: request and accounting are BIT-IDENTICAL to the old path. ----
    kp1 = keep_prob(used_rows=requested, keep_target_rows=10_000_000)  # >= target -> 1.0
    assert kp1 == 1.0
    select_request1 = requested if kp1 >= 1.0 else int(math.ceil(requested / kp1))
    assert select_request1 == requested  # no inflation
    _sel1, selected1 = _select_files_for_rows(entries, select_request1, np.random.default_rng(1))
    eff1_new = min(requested, int(round(selected1 * kp1)))
    eff1_old = min(requested, selected1)  # pre-fix formula
    assert eff1_new == eff1_old, "kp=1.0 accounting must be bit-identical to the old path"

    # --- kp < 1.0: inflate the request, account the post-thin survivors. --------
    used = 60_000  # window larger than the keep target -> kp = target/used < 1
    kp = keep_prob(used_rows=used, keep_target_rows=30_000)
    assert kp == pytest.approx(0.5)
    select_request = requested if kp >= 1.0 else int(math.ceil(requested / kp))
    assert select_request == int(math.ceil(requested / kp)) == 12_000  # inflated by 1/kp

    sel, selected = _select_files_for_rows(entries, select_request, np.random.default_rng(2025))
    assert selected >= select_request, "inflated selection must cover the 1/kp-scaled request"

    effective_rows = min(requested, int(round(selected * kp)))
    # The expected trained rows (post-thin) should be ~requested, NOT requested*kp.
    assert effective_rows == requested, (
        f"effective_rows {effective_rows} != requested {requested}; the 1/kp inflation "
        f"should make post-thin survivors reach the target"
    )

    # Simulate build_window_split's per-shard Bernoulli(kp) thinning and confirm the
    # actually-trained survivor count lands near effective_rows (the debited value) —
    # i.e. accounting matches what will be trained in expectation, closing the bug.
    thin_rng = np.random.default_rng(2025)
    ordered = sorted(sel, key=lambda e: (int(e.generation), int(e.game_key)))
    survivors = int(sum((thin_rng.random(int(e.rows)) < kp).sum() for e in ordered))
    # perm[:effective_rows] truncation in train_passes caps the trained rows at
    # effective_rows; survivors >= effective_rows means the epoch trains exactly
    # effective_rows rows. Assert survivors is close to (and at least ~) the target.
    assert survivors == pytest.approx(requested, rel=0.1), (
        f"post-thin survivors {survivors} not within 10% of requested {requested}"
    )
    assert survivors >= effective_rows * 0.9, (
        f"survivors {survivors} far below debited effective_rows {effective_rows}"
    )

    # --- window smaller than requested: exact behavior (train what exists). -----
    small = [_FakeEntry(100, 1, i, f"epoch_000001/game_{i}.npz") for i in range(20)]  # 2000 rows
    kp_s = keep_prob(used_rows=2_000, keep_target_rows=1_000)  # 0.5
    assert kp_s == pytest.approx(0.5)
    req_s = 5_000  # more than the (thinned) window can supply
    select_request_s = int(math.ceil(req_s / kp_s))  # 10_000 > 2_000 available
    sel_s, selected_s = _select_files_for_rows(small, select_request_s, np.random.default_rng(9))
    assert selected_s == 2_000, "short window returns all rows"
    effective_s = min(req_s, int(round(selected_s * kp_s)))
    assert effective_s == 1_000, (
        f"short-window effective_rows {effective_s} != round(2000*0.5)=1000"
    )
    assert effective_s < req_s, "when the window is smaller than requested, train what exists"
    print("  keep_prob accounting: kp=1.0 bit-identical; kp=0.5 inflates request 2x, "
          f"effective_rows={requested} post-thin survivors~{survivors}; "
          "short window trains round(selected*kp)")


# ----------------------------------------------------------------------
# 4. train-bucket governor
# ----------------------------------------------------------------------


def _make_trainer(**training_overrides) -> HexfieldTrainer:
    """A trainer with a CPU config and a tiny real model/optimizer (we only call
    the pure window/governor/selection methods, never train_passes).

    HexfieldTrainer.__init__ partitions params via ``model.named_parameters()``
    for per-group grad-norm logging, so the model must be a real ``nn.Module``
    (a bare SimpleNamespace has no ``named_parameters``); the linear layer is never
    forwarded/stepped by the paths these tests exercise.
    """
    import torch

    base = dict(
        max_train_bucket_size=500_000.0,
        train_samples_per_epoch=100_000,
        max_train_bucket_per_new_data=8.0,
    )
    base.update(training_overrides)
    cfg = HexfieldConfig(device="cpu", training=TrainingSection(**base))
    model = torch.nn.Linear(4, 3)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return HexfieldTrainer(model=model, config=cfg, optimizer=opt)


def test_update_train_bucket() -> None:
    tr = _make_trainer()
    # Check the governor against the documented formula inline (accrual = new_rows
    # * per_new_data, capped at max(bucket_size, samples_per_epoch); a decreasing
    # total rebases the watermark and zeroes the reload counter).
    cap = max(500_000.0, 100_000.0)

    # Step 1: 1000 new rows -> +8000 level, watermark -> 1000.
    tr._update_train_bucket(1000, window_start=0)
    assert tr.train_state.train_bucket_level == min(cap, 0.0 + 1000 * 8.0) == 8000.0
    assert tr.train_state.train_bucket_level_at_row == 1000
    assert tr.train_state.total_num_data_rows == 1000

    # Step 2: +500 rows -> +4000 -> 12000.
    tr._update_train_bucket(1500, window_start=10)
    assert tr.train_state.train_bucket_level == 12000.0
    assert tr.train_state.train_bucket_level_at_row == 1500
    assert tr.train_state.window_start_data_row_idx == 10

    # Step 3: same total -> no change (neither branch).
    before = tr.train_state.train_bucket_level
    tr._update_train_bucket(1500, window_start=10)
    assert tr.train_state.train_bucket_level == before

    # Step 4: total DECREASES (window regenerated) -> rebase watermark, zero reload.
    tr.train_state.train_steps_since_last_reload = 5
    tr._update_train_bucket(1200, window_start=3)
    assert tr.train_state.train_bucket_level_at_row == 1200
    assert tr.train_state.train_steps_since_last_reload == 0
    assert tr.train_state.train_bucket_level == min(before, cap) == before  # level clamped, not zeroed

    # Cap: a huge accrual saturates at cap.
    tr2 = _make_trainer(max_train_bucket_size=1000.0, train_samples_per_epoch=100)
    tr2._update_train_bucket(10_000_000, window_start=0)
    assert tr2.train_state.train_bucket_level == max(1000.0, 100.0) == 1000.0
    print("  _update_train_bucket: accrual / cap / no-op / monotone-reload branches match the "
          "documented governor formula")


# ----------------------------------------------------------------------
# 5. build_window_split keep_prob behaviour
# ----------------------------------------------------------------------


def test_build_window_split(samples_dir: Path, entries: list[ShardEntry]) -> None:
    # keep_prob=1.0 -> every row kept; decode-parity preserved per row.
    win_full = build_window_split(entries, keep_prob=1.0, rng=np.random.default_rng(0),
                                  samples_dir=samples_dir)
    total = sum(e.rows for e in entries)
    assert win_full.n == total, f"keep_prob=1.0 kept {win_full.n} != {total}"
    assert isinstance(win_full, PackedWindow)

    # Row-for-row decode parity of the full window vs the per-shard oracle (rows
    # are concatenated in (generation, game_key) order).
    ordered = sorted(entries, key=lambda e: (e.generation, e.game_key))
    base = 0
    checked = 0
    for e in ordered:
        oracle = hex_shards.read_compact_shard(samples_dir / e.rel_path)
        for k in range(len(oracle)):
            v = win_full.row_view(base + k)
            assert v.records() == oracle[k].records, f"records mismatch row {base+k}"
            assert v.value == oracle[k].value, f"value mismatch row {base+k}"
            assert v.policy() == oracle[k].policy, f"policy mismatch row {base+k}"
            assert v.short_term_value() == oracle[k].short_term_value, f"stv mismatch row {base+k}"
            checked += 1
        base += len(oracle)
    assert base == win_full.n
    print(f"  build_window_split keep_prob=1.0: n={win_full.n}, {checked} rows decode-parity vs oracle")

    # keep_prob<1.0 -> deterministic subset; survivors must still decode-parity.
    kp = 0.5
    win_a = build_window_split(entries, keep_prob=kp, rng=np.random.default_rng(99),
                               samples_dir=samples_dir)
    win_b = build_window_split(entries, keep_prob=kp, rng=np.random.default_rng(99),
                               samples_dir=samples_dir)
    assert win_a.n == win_b.n, f"keep_prob subsample not deterministic: {win_a.n} != {win_b.n}"
    assert 0 < win_a.n < total, f"subsample {win_a.n} not strictly between 0 and {total}"

    # Verify the SURVIVORS decode field-identically: reconstruct the exact keep
    # mask the function used (single shared rng, per-shard, (gen,game_key) order)
    # and compare survivor rows against the oracle.
    rng = np.random.default_rng(99)
    base = 0
    sv_checked = 0
    for e in ordered:
        oracle = hex_shards.read_compact_shard(samples_dir / e.rel_path)
        shard = load_packed_shard(samples_dir / e.rel_path)
        mask = rng.random(shard.n) < kp
        for k in range(len(oracle)):
            if not mask[k]:
                continue
            v = win_a.row_view(base)
            assert v.records() == oracle[k].records, f"survivor records mismatch at out-row {base}"
            assert v.value == oracle[k].value, f"survivor value mismatch at out-row {base}"
            assert v.policy() == oracle[k].policy, f"survivor policy mismatch at out-row {base}"
            assert v.short_term_value() == oracle[k].short_term_value, f"survivor stv mismatch"
            base += 1
            sv_checked += 1
    assert base == win_a.n, f"reconstructed survivor count {base} != window n {win_a.n}"
    print(f"  build_window_split keep_prob=0.5: deterministic n={win_a.n}; "
          f"{sv_checked} survivors decode-parity (CSR rebuild correct)")


# ----------------------------------------------------------------------
# 6. dry-run select_training_samples against copied real shards
# ----------------------------------------------------------------------


def _fake_ctx(samples_dir: Path, diag_dir: Path, seed: int = 7) -> SimpleNamespace:
    """Minimal RunContext stand-in: only the attrs select_training_samples reads
    (ctx.config.run.seed, ctx.samples_dir, ctx.diagnostics_dir)."""
    return SimpleNamespace(
        config=SimpleNamespace(run=SimpleNamespace(seed=seed)),
        samples_dir=samples_dir,
        diagnostics_dir=diag_dir,
    )


def _fake_components() -> SimpleNamespace:
    return SimpleNamespace(shared=SimpleNamespace(sample_window=None))


def test_select_training_samples_dryrun(samples_dir: Path, diag_dir: Path) -> None:
    # Tune the taper low so the modest copied window (a few hundred rows) clears
    # min_rows and produces a real window. requested small so effective_rows>0.
    tr = _make_trainer(
        shuffle_min_rows=1,
        shuffle_taper_window_scale=10.0,
        shuffle_keep_target_rows=10_000,
        train_samples_per_epoch=200,
        max_train_bucket_size=500_000.0,
        max_train_bucket_per_new_data=8.0,
    )
    ctx = _fake_ctx(samples_dir, diag_dir)
    comp = _fake_components()

    out = tr.select_training_samples(ctx=ctx, components=comp, epoch=1)
    assert out["status"] == "completed", f"epoch1 status {out['status']}: {out.get('reason')}"
    # plausible dict (PLAN §6 reference return shape).
    for key in ("total_rows", "live_total_rows", "desired_rows", "used_rows", "keep_prob",
                "effective_rows", "window_rows", "window_start", "train_bucket_level",
                "reuse_ratio"):
        assert key in out, f"return dict missing {key}"
    assert out["window_rows"] > 0, f"window_rows {out['window_rows']} not > 0"
    assert out["effective_rows"] > 0, f"effective_rows {out['effective_rows']} not > 0"
    assert out["total_rows"] >= out["live_total_rows"], "cumulative < live (monotone broken)"
    assert out["desired_rows"] >= 1
    assert 0.0 < out["keep_prob"] <= 1.0
    # the window handle is a PackedWindow on components.shared.
    assert isinstance(comp.shared.sample_window, PackedWindow), (
        f"sample_window is {type(comp.shared.sample_window).__name__}, not PackedWindow"
    )
    assert comp.shared.sample_window.n == out["window_rows"]
    # governor was credited (cumulative>0) and debited by effective_rows.
    cap = max(500_000.0, 200.0)
    expected_after = min(cap, out["total_rows"] * 8.0) - out["effective_rows"]
    assert abs(tr.train_state.train_bucket_level - expected_after) < 1e-6, (
        f"bucket level {tr.train_state.train_bucket_level} != expected {expected_after}"
    )
    assert tr.train_state.train_steps_since_last_reload == 1
    # the select diag was written under the (scratch) diagnostics dir.
    assert (diag_dir / "hexfield.select.epoch_000001.json").exists(), "select diag not written"
    print(f"  dry-run epoch 1: status=completed window_rows={out['window_rows']} "
          f"effective_rows={out['effective_rows']} total_rows={out['total_rows']} "
          f"keep_prob={out['keep_prob']:.3f} bucket={tr.train_state.train_bucket_level:.0f}")

    # A second epoch: same total (no new shards) -> no accrual; bucket debited
    # again by effective_rows; steps_since_last_reload advances to 2.
    bucket_before = tr.train_state.train_bucket_level
    comp2 = _fake_components()
    out2 = tr.select_training_samples(ctx=ctx, components=comp2, epoch=2)
    assert out2["status"] == "completed", f"epoch2 status {out2['status']}: {out2.get('reason')}"
    assert out2["total_rows"] == out["total_rows"], "cumulative changed with no new shards"
    assert tr.train_state.train_bucket_level == bucket_before - out2["effective_rows"], \
        "second-epoch debit wrong"
    assert tr.train_state.train_steps_since_last_reload == 2
    assert isinstance(comp2.shared.sample_window, PackedWindow)
    print(f"  dry-run epoch 2: no new rows -> no accrual; bucket {bucket_before:.0f} -> "
          f"{tr.train_state.train_bucket_level:.0f}; steps_since_reload=2")

    # Bucket-limited branch: a fresh trainer whose bucket can't cover effective_rows.
    tr_lim = _make_trainer(
        shuffle_min_rows=1,
        shuffle_taper_window_scale=10.0,
        shuffle_keep_target_rows=10_000,
        train_samples_per_epoch=200,
        max_train_bucket_size=500_000.0,
        max_train_bucket_per_new_data=0.0,  # never credits -> always limited
    )
    comp3 = _fake_components()
    out3 = tr_lim.select_training_samples(ctx=ctx, components=comp3, epoch=1)
    assert out3["status"] == "train_bucket_limited", f"expected limited, got {out3['status']}"
    assert isinstance(comp3.shared.sample_window, PackedWindow) and comp3.shared.sample_window.n == 0
    print(f"  dry-run bucket-limited: status=train_bucket_limited, empty PackedWindow set")


# ----------------------------------------------------------------------
# fixtures for the data-driven gates (5, 6): synthesized shards + manifest
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def _p4_root(tmp_path_factory) -> Path:
    """A synthesized samples tree + its manifest, built once for the module.

    Writes several epoch dirs of synthesized (expandable) shards, then scans the
    manifest (persisting ``.buffer_manifest.json`` into the samples dir — a tmp
    dir, never under ``runs/*``)."""
    root = tmp_path_factory.mktemp("p4_select")
    samples_dir = root / "samples"
    generate_samples_tree(
        samples_dir, epochs=3, games_per_epoch=4, max_plies=24, base_seed=4200
    )
    manifest = scan_or_update_manifest(samples_dir)
    assert manifest.entries, "manifest empty after synthesis"
    assert manifest.total_rows > 0
    assert manifest.cumulative_rows_ever >= manifest.total_rows
    (root / "diagnostics").mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def samples_dir(_p4_root: Path) -> Path:
    return _p4_root / "samples"


@pytest.fixture
def diag_dir(_p4_root: Path) -> Path:
    return _p4_root / "diagnostics"


@pytest.fixture
def entries(_p4_root: Path) -> list[ShardEntry]:
    manifest = scan_or_update_manifest(_p4_root / "samples")
    return list(manifest.entries)
