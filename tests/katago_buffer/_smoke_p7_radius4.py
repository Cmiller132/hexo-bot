"""Radius-4 subprocess smoke: rust == serial under HEXFIELD_SUPPORT_RADIUS=4 on
REAL radius-8 main_2 shards (the support legitimately shrinks). No injection — this
checks the closed-form legality + halo shrink agree between the Rust twin and the
Python oracle when both read radius 4 at import. Run via env HEXFIELD_SUPPORT_RADIUS=4.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from hexfield.expand_backends import _resolve_support_radius, expand_rows
from hexfield.support import _SUPPORT_RADIUS
from hexfield.window import concat_packed, load_packed_shard

SAMPLES = Path(__file__).resolve().parent / "_scratch" / "p5" / "samples"


def rows_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    return (
        np.array_equal(a.support.coords, b.support.coords)
        and np.array_equal(a.support.nbr, b.support.nbr)
        and np.array_equal(a.support.dist, b.support.dist)
        and a.support.legal_count == b.support.legal_count
        and a.support.stone_count == b.support.stone_count
        and a.support.halo_count == b.support.halo_count
        and np.array_equal(a.feats, b.feats)
        and np.array_equal(a.policy, b.policy)
        and np.array_equal(a.opp_policy, b.opp_policy)
        and abs(float(a.opp_coverage) - float(b.opp_coverage)) <= 1e-12
        and np.array_equal(a.cell_q, b.cell_q)
        and np.array_equal(a.cell_q_mask, b.cell_q_mask)
        and float(a.policy_surprise) == float(b.policy_surprise)
    )


def main():
    print("env radius:", os.environ.get("HEXFIELD_SUPPORT_RADIUS"))
    print("support._SUPPORT_RADIUS:", _SUPPORT_RADIUS)
    print("rust resolve_support_radius:", _resolve_support_radius())
    assert _SUPPORT_RADIUS == 4 and _resolve_support_radius() == 4, "radius not 4"

    npzs = sorted(SAMPLES.glob("epoch_*/game_*.npz"))[:30]
    window = concat_packed([load_packed_shard(p) for p in npzs])
    d6 = (np.arange(window.n) % 12).astype(np.int64)

    rows_s, valid_s = expand_rows(window, None, d6, backend="serial", tolerate_off_legal=True)
    rows_r, valid_r = expand_rows(window, None, d6, backend="rust", tolerate_off_legal=True)
    assert np.array_equal(valid_s, valid_r), "valid mask differs at radius 4"
    bad = [i for i, (a, b) in enumerate(zip(rows_s, rows_r)) if not rows_equal(a, b)]
    assert not bad, f"{len(bad)} rows differ at radius 4 (first {bad[:8]})"
    n_skip = int((~valid_s).sum())
    print(f"RADIUS4 SMOKE PASS: n={window.n} n_skip={n_skip} rust==serial element-wise")


if __name__ == "__main__":
    main()
