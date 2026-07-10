#!/usr/bin/env python3
"""Export hexfield_eq structural data for the learn/ site demos.

Writes two JSON files into apps/showcase/web/learn/data/:

- eq_group_tables.json — the D6 group action tables the symmetry demos render:
  per-element coordinate matrices, slot (regular-rep) permutations, tap/axis
  permutations, the (7, 12, 12) conv weight-tie classes (84 free blocks), the
  1x1 tie classes, and the (237, 3) joint bias-orbit classes (81 free values).
- eq_walkthrough.json — the explainer's §6 position (docs/explainer_assets/
  walkthrough_position.json) run through the production featurizer: support
  segments, coords, nbr, all 46 feature planes, and the 12 ray lengths.

Everything is read from the hexfield_eq package itself so the site can never
drift from the code. Run under the production arch env
(scripts/prefit_env/hexfield_eq_raytap_a5.env sourced) from the repo root:

    set -a; source scripts/prefit_env/hexfield_eq_raytap_a5.env; set +a
    python scripts/export_eq_learn_data.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "apps" / "showcase" / "web" / "learn" / "data"


def check_env() -> None:
    expected = {"HEXFIELD_EQ_FEATURE_VERSION": "2", "HEXFIELD_EQ_TRUNK": "CCACCACA"}
    for key, want in expected.items():
        got = os.environ.get(key)
        if got != want:
            sys.exit(
                f"{key}={got!r}, expected {want!r} — source "
                "scripts/prefit_env/hexfield_eq_raytap_a5.env first"
            )


def export_group_tables() -> dict:
    import hexfield_eq.constants as C
    import hexfield_eq.equivariant as E
    import hexfield_eq.geometry as G

    group = E.build_group()
    n = E.GROUP  # 12

    # Coordinate action of each element as a 2x2 integer matrix on (q, r):
    # columns are the images of the basis vectors (1, 0) and (0, 1).
    coord_mats = []
    for g in range(n):
        c0 = G.apply_d6(g, 1, 0)
        c1 = G.apply_d6(g, 0, 1)
        coord_mats.append([[c0[0], c1[0]], [c0[1], c1[1]]])

    # Axis permutation per element: where each win axis {Q, R, QR} lands.
    # An axis is a line through the origin; g maps axis a to the axis of the
    # transformed delta.
    axis_deltas = [(1, 0), (0, 1), (1, -1)]  # Q, R, QR

    def axis_of(dq: int, dr: int) -> int:
        for i, (aq, ar) in enumerate(axis_deltas):
            if dq * ar - dr * aq == 0:
                return i
        raise AssertionError((dq, dr))

    axis_perms = []
    for g in range(n):
        axis_perms.append(
            [axis_of(*G.apply_d6(g, aq, ar)) for aq, ar in axis_deltas]
        )

    # Bias row map: every exact offset within the disk, plus ring/far/token
    # row descriptions.
    disk = []
    for dq, dr in G.disk_offsets(G.BIAS_DISK_RADIUS):
        disk.append([dq, dr, G.rel_bias_index(dq, dr)])

    lut, n_free = E.joint_bias_lut()
    conv = E.conv_gather_index()
    lin = E.linear_gather_index()

    return {
        "group_order": n,
        "coord_mats": coord_mats,
        "mult": group["mult"],
        "inv": group["inv"],
        "tap_perms": group["tapp"],
        "slot_perms": group["regp"],
        "axis_perms": axis_perms,
        "cosets": group["cosets"],
        "coset_of_slot": group["cos_of"],
        "bias": {
            "rows": G.BIAS_ROWS,
            "exact_rows": G.BIAS_EXACT_ROWS,
            "disk_radius": G.BIAS_DISK_RADIUS,
            "ring_min": G.BIAS_RING_MIN,
            "ring_max": G.BIAS_RING_MAX,
            "on_axis_base": G.BIAS_ON_AXIS_BASE,
            "off_axis_base": G.BIAS_OFF_AXIS_BASE,
            "far_row": G.BIAS_FAR_ROW,
            "disk": disk,
            "joint_classes": lut.tolist(),
            "free_values": int(n_free),
        },
        "conv_tie_classes": conv.tolist(),
        "conv_free_blocks": int(conv.max()) + 1,
        "linear_tie_classes": lin.tolist(),
        "linear_free_blocks": int(lin.max()) + 1,
        "constants": {
            "channels": C.CHANNELS,
            "c_orbit": C.C_ORBIT,
            "attention_heads": C.ATTENTION_HEADS,
            "num_features": C.NUM_FEATURES,
            "raylen_slots": C.RAYLEN_SLOTS,
            "ray_reach": C.RAY_REACH,
            "trunk_layout": C.TRUNK_LAYOUT,
            "num_tokens": C.NUM_TOKENS,
            "value_bins": C.VALUE_BINS,
            "moves_left_cap": C.MOVES_LEFT_CAP,
            "support_radius": 4,
            "window_len": 6,
        },
    }


def export_walkthrough() -> dict:
    from hexfield_eq.features import (
        PositionFacts,
        build_position,
        build_ray_lengths,
    )
    from hexfield_eq.model import HexfieldNet

    src = json.load(open(ROOT / "docs" / "explainer_assets" / "walkthrough_position.json"))
    facts = PositionFacts(
        records=tuple(tuple(r) for r in src["records"]),
        current_player=src["current_player"],
        phase=src["phase"],
        first_stone=tuple(src["first_stone"]),
    )
    support, feats = build_position(facts)
    raylen = build_ray_lengths(facts, support)
    n = feats.shape[0]

    net = HexfieldNet()
    params = sum(p.numel() for p in net.parameters())

    return {
        "source": src["source"],
        "records": src["records"],
        "current_player": src["current_player"],
        "phase": src["phase"],
        "first_stone": src["first_stone"],
        "n_nodes": n,
        "legal_count": int(support.legal_count),
        "stone_count": int(support.stone_count),
        "halo_count": int(support.halo_count),
        "coords": np.asarray(support.coords).astype(int).tolist(),
        "nbr": np.asarray(support.nbr).astype(int).tolist(),
        "dist": np.asarray(support.dist).astype(int).tolist(),
        "feats": np.round(feats.astype(float), 4).tolist(),
        "raylen": raylen.astype(int).tolist(),
        "model_params": int(params),
    }


def main() -> None:
    check_env()
    OUT.mkdir(parents=True, exist_ok=True)

    tables = export_group_tables()
    with open(OUT / "eq_group_tables.json", "w") as f:
        json.dump(tables, f, separators=(",", ":"))
    print(f"wrote eq_group_tables.json ({(OUT / 'eq_group_tables.json').stat().st_size} bytes)")

    walk = export_walkthrough()
    with open(OUT / "eq_walkthrough.json", "w") as f:
        json.dump(walk, f, separators=(",", ":"))
    print(f"wrote eq_walkthrough.json ({(OUT / 'eq_walkthrough.json').stat().st_size} bytes)")
    print(
        f"walkthrough: N={walk['n_nodes']} legal={walk['legal_count']} "
        f"stones={walk['stone_count']} halo={walk['halo_count']} params={walk['model_params']}"
    )


if __name__ == "__main__":
    main()
