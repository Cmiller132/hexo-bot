"""Quick smoke: rust backend == serial element-wise on a small real window, with
a FIXED d6 (not RNG). Run BEFORE the full parity test to fail fast on kernel bugs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from shrimp.expand_backends import expand_rows
from shrimp.window import concat_packed, load_packed_shard

SAMPLES = Path(__file__).resolve().parent / "_scratch" / "p5" / "samples"


def load_window(limit=20):
    npzs = sorted(SAMPLES.glob("epoch_*/game_*.npz"))[:limit]
    assert npzs, f"no scratch shards under {SAMPLES}"
    return concat_packed([load_packed_shard(p) for p in npzs])


def rows_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return (
        a.support.num_nodes == b.support.num_nodes
        and a.support.legal_count == b.support.legal_count
        and a.support.stone_count == b.support.stone_count
        and a.support.halo_count == b.support.halo_count
        and np.array_equal(a.support.coords, b.support.coords)
        and np.array_equal(a.support.nbr, b.support.nbr)
        and np.array_equal(a.support.dist, b.support.dist)
        and np.array_equal(a.feats, b.feats)
        and np.array_equal(a.policy, b.policy)
        and np.array_equal(a.opp_policy, b.opp_policy)
        and abs(float(a.opp_coverage) - float(b.opp_coverage)) <= 1e-12
        and float(a.value) == float(b.value)
        and np.array_equal(a.stvalue, b.stvalue)
        and np.array_equal(a.stvalue_mask, b.stvalue_mask)
        and float(a.moves_left) == float(b.moves_left)
        and float(a.moves_left_mask) == float(b.moves_left_mask)
    )


def main():
    window = load_window()
    print(f"window.n={window.n}")
    # Fixed symmetry sweep: each row gets sym = (row index % 12).
    d6 = (np.arange(window.n) % 12).astype(np.int64)
    rows_s, valid_s = expand_rows(window, None, d6, backend="serial")
    rows_r, valid_r = expand_rows(window, None, d6, backend="rust")
    assert len(rows_s) == window.n == len(rows_r), (len(rows_s), len(rows_r))
    assert np.array_equal(valid_s, valid_r), "valid mask differs"
    bad = []
    for i, (a, b) in enumerate(zip(rows_s, rows_r)):
        if not rows_equal(a, b):
            bad.append(i)
    if bad:
        i = bad[0]
        a, b = rows_s[i], rows_r[i]
        print(f"FIRST MISMATCH row {i} (d6={int(d6[i])}):")
        print("  serial N/L/S/H:", a.support.num_nodes, a.support.legal_count, a.support.stone_count, a.support.halo_count)
        print("  rust   N/L/S/H:", b.support.num_nodes, b.support.legal_count, b.support.stone_count, b.support.halo_count)
        print("  coords_equal:", np.array_equal(a.support.coords, b.support.coords))
        print("  dist_equal:", np.array_equal(a.support.dist, b.support.dist))
        print("  nbr_equal:", np.array_equal(a.support.nbr, b.support.nbr))
        print("  feats_equal:", np.array_equal(a.feats, b.feats))
        if not np.array_equal(a.feats, b.feats):
            d = np.abs(a.feats.astype(np.float64) - b.feats.astype(np.float64))
            idx = np.unravel_index(np.argmax(d), d.shape)
            print("    max feats diff:", float(d.max()), "at", idx, "serial=", a.feats[idx], "rust=", b.feats[idx])
        print("  policy_equal:", np.array_equal(a.policy, b.policy))
        print("  opp_equal:", np.array_equal(a.opp_policy, b.opp_policy))
        print("  cov serial/rust:", a.opp_coverage, b.opp_coverage)
    assert not bad, f"{len(bad)} of {window.n} rows differ (first {bad[:10]})"
    print(f"SMOKE PASS: {window.n} rows rust==serial element-wise ({int(valid_s.sum())} valid)")


if __name__ == "__main__":
    main()
