"""PLAN Phase 3 self-test: packed columnar window decode parity (PLAN §5.1/§5.5/§6).

Runs against SYNTHESIZED ``hexfield_compact_v1`` shards (the ``paths`` fixture in
conftest.py generates a handful across a couple of epoch subdirs; the private
private development-run live tree is unavailable publicly). Two gates:

1. DECODE PARITY — for each synthesized shard, ``load_packed_shard(path)`` then
   ``row_view(i)`` must reconstruct field-identical values to the existing oracle
   ``shards.read_compact_shard(path)[i]`` for EVERY row, including every
   CSR/ragged column (records, policy, opp_policy, own_hot/opp_hot/own_win/
   opp_win) and every scalar (turn_index, current_player, phase, value,
   moves_left, first_stone, short_term_value).

2. CONCAT — ``concat_packed`` of two shards has ``n == sum`` and CSR offsets
   correctly rebased: a ``row_view`` of a row that lives in part 2 still
   parity-matches that row decoded from its SOURCE shard.

CPU only, no GPU, no model.
"""

from __future__ import annotations

import numpy as np

from hexfield import shards
from hexfield.shards import _PHASES
from hexfield.window import PackedRowView, concat_packed, load_packed_shard


def _assert_row_parity(view: PackedRowView, oracle, *, where: str) -> None:
    """Field-identical check of one PackedRowView vs one HexfieldSampleData."""
    # --- scalars -------------------------------------------------------------
    assert view.turn_index == oracle.turn_index, f"{where}: turn_index {view.turn_index} != {oracle.turn_index}"
    assert view.current_player == oracle.current_player, f"{where}: current_player"
    # oracle.phase is the decoded NAME; view.phase is the u8 enum index.
    assert _PHASES[view.phase] == oracle.phase, f"{where}: phase {_PHASES[view.phase]!r} != {oracle.phase!r}"
    # value / moves_left are float32 round-tripped on both sides -> exact equal.
    assert view.value == oracle.value, f"{where}: value {view.value!r} != {oracle.value!r}"
    assert view.moves_left == oracle.moves_left, f"{where}: moves_left {view.moves_left!r} != {oracle.moves_left!r}"
    # --- first_stone ---------------------------------------------------------
    assert view.first_stone() == oracle.first_stone, f"{where}: first_stone {view.first_stone()} != {oracle.first_stone}"
    # --- records (history CSR: q,r,owner,placement_index) --------------------
    assert view.records() == oracle.records, f"{where}: records mismatch"
    # --- policy / opp_policy CSR ---------------------------------------------
    assert view.policy() == oracle.policy, f"{where}: policy mismatch"
    assert view.opp_policy() == oracle.opp_policy, f"{where}: opp_policy mismatch"
    # --- standing-cell qr CSR ------------------------------------------------
    assert view.own_hot() == oracle.own_hot, f"{where}: own_hot mismatch"
    assert view.opp_hot() == oracle.opp_hot, f"{where}: opp_hot mismatch"
    assert view.own_win() == oracle.own_win, f"{where}: own_win mismatch"
    assert view.opp_win() == oracle.opp_win, f"{where}: opp_win mismatch"
    # --- short-term value (block + mask) -------------------------------------
    assert view.short_term_value() == oracle.short_term_value, (
        f"{where}: short_term_value {view.short_term_value()} != {oracle.short_term_value}"
    )


def test_decode_parity(paths: list[str]) -> None:
    total_rows = 0
    for p in paths:
        packed = load_packed_shard(p)
        oracle = shards.read_compact_shard(p)
        assert packed.n == len(oracle), f"{p}: n {packed.n} != oracle {len(oracle)}"
        # horizons preserved verbatim from the shard's stored `horizons` column
        with np.load(p) as raw:
            assert tuple(packed.horizons) == tuple(int(h) for h in raw["horizons"]), (
                f"{p}: horizons {packed.horizons} != stored {tuple(int(h) for h in raw['horizons'])}"
            )
        for i in range(packed.n):
            view = packed.row_view(i)
            _assert_row_parity(view, oracle[i], where=f"{p}#row{i}")
            total_rows += 1
    print(f"  decode parity: {len(paths)} shards, {total_rows} rows field-identical to read_compact_shard")


def test_view_views_are_zero_copy(paths: list[str]) -> None:
    """row_view slices must be VIEWS into the packed columns (no copy)."""
    path = paths[0]
    packed = load_packed_shard(path)
    if packed.n == 0:
        return
    v = packed.row_view(0)
    # hist_qr view should share base with the packed hist_qr column
    if v.hist_qr.size:
        assert v.hist_qr.base is not None, f"{path}: hist_qr row_view is not a view"
    if v.pol_w.size:
        assert v.pol_w.base is not None, f"{path}: pol_w row_view is not a view"
    assert v.stvalue.base is not None, f"{path}: stvalue row_view is not a view"


def test_concat(paths: list[str]) -> None:
    # Pick two non-empty shards from different epochs where possible.
    chosen: list[str] = []
    for p in paths:
        if load_packed_shard(p).n > 0:
            chosen.append(p)
        if len(chosen) == 2:
            break
    assert len(chosen) == 2, "need two non-empty shards for concat test"
    pa_path, pb_path = chosen

    pa = load_packed_shard(pa_path)
    pb = load_packed_shard(pb_path)
    na, nb = pa.n, pb.n
    # oracle decodes from the SOURCE shards (load_packed_shard above consumed its
    # own copies; re-load fresh parts for concat since concat frees its inputs).
    oracle_a = shards.read_compact_shard(pa_path)
    oracle_b = shards.read_compact_shard(pb_path)

    merged = concat_packed([load_packed_shard(pa_path), load_packed_shard(pb_path)])
    assert merged.n == na + nb, f"concat n {merged.n} != {na + nb}"

    # Every offsets array must be monotone non-decreasing and start at 0.
    from hexfield.window import OFF_COLS

    for off in OFF_COLS:
        arr = merged.cols[off]
        assert arr.shape[0] == merged.n + 1, f"{off}: bad length {arr.shape[0]}"
        assert int(arr[0]) == 0, f"{off}: does not start at 0"
        assert np.all(np.diff(arr) >= 0), f"{off}: offsets not monotone after rebase"
        assert arr.dtype == np.int64, f"{off}: offsets not int64 after rebase"

    # Part-1 rows parity-match oracle_a; part-2 rows (after rebased offsets)
    # parity-match oracle_b at the SAME source index.
    for i in range(na):
        _assert_row_parity(merged.row_view(i), oracle_a[i], where=f"concat part1 row{i}")
    for j in range(nb):
        _assert_row_parity(merged.row_view(na + j), oracle_b[j], where=f"concat part2 row{j}")

    # Spot-check the rebase arithmetic directly for the second part's first row:
    # its global hist offset must equal the running total of part-1 history.
    if nb > 0:
        hist_off = merged.cols["hist_off"]
        # part-1 total history length == hist_off[na]
        part1_hist = int(hist_off[na])
        # part-2 row 0 raw length from oracle_b[0]:
        part2_row0_len = len(oracle_b[0].records)
        assert int(hist_off[na + 1]) == part1_hist + part2_row0_len, "hist_off rebase arithmetic wrong"

    # Ensure the qr-CSR rebase path was genuinely exercised: at least one
    # non-empty own_hot/opp_hot segment must exist somewhere in part 2, and the
    # global qr offset for part 2 must be non-zero (so 2*off slicing is tested
    # past the rebase boundary).
    opp_hot_off = merged.cols["opp_hot_off"]
    assert int(opp_hot_off[na]) > 0 or int(merged.cols["own_hot_off"][na]) > 0, (
        "test data has no standing-hot cells in part 1; qr rebase not exercised"
    )

    print(
        f"  concat: parts n={na}+{nb} -> {merged.n}; offsets rebased to int64; "
        f"qr-CSR rebase exercised (opp_hot base@part2={int(opp_hot_off[na])}); "
        f"part-2 rows parity-match their source shard"
    )

    # 3-shard concat: accumulate across two boundaries; every row still parity.
    if len(paths) >= 3:
        triple_paths = chosen + [next(q for q in paths if q not in chosen and load_packed_shard(q).n > 0)]
        oracles = [shards.read_compact_shard(q) for q in triple_paths]
        triple = concat_packed([load_packed_shard(q) for q in triple_paths])
        assert triple.n == sum(len(o) for o in oracles)
        base = 0
        for oc in oracles:
            for k in range(len(oc)):
                _assert_row_parity(triple.row_view(base + k), oc[k], where=f"triple row{base + k}")
            base += len(oc)
        print(f"  concat(3 shards): n={triple.n}; all rows parity across two rebase boundaries")


def test_empty_and_concat_with_empty(paths: list[str]) -> None:
    from hexfield.window import PackedWindow

    e = PackedWindow.empty()
    assert e.n == 0
    # concat([empty, real, empty]) == real
    real = load_packed_shard(paths[0])
    n_real = real.n
    oracle = shards.read_compact_shard(paths[0])
    merged = concat_packed([PackedWindow.empty(), load_packed_shard(paths[0]), PackedWindow.empty()])
    assert merged.n == n_real, f"concat-with-empty n {merged.n} != {n_real}"
    for i in range(merged.n):
        _assert_row_parity(merged.row_view(i), oracle[i], where=f"empty-concat row{i}")
    # concat of all-empty -> empty
    assert concat_packed([PackedWindow.empty(), PackedWindow.empty()]).n == 0
    print(f"  empty: PackedWindow.empty() ok; concat([empty,real,empty]) == real (n={n_real})")
