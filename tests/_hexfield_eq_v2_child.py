"""Child-process featurizer checks for the HEXFIELD_EQ_FEATURE_VERSION gate.

Driven by tests/test_hexfield_eq_feature_v2.py (SPEC_RAYTAP_CONV.md §6.1 T1/T2,
Phase F). The feature version is an import-time env knob, so every check that
needs a specific version runs in a fresh interpreter with a controlled env (the
subprocess pattern of test_hexfield_eq_checkpoint_meta.py). Each subcommand
prints one JSON verdict line as the LAST stdout line; assertion failures
surface as a nonzero exit with the traceback on stderr.

Subcommands:
  corpus-hash        sha256 over the deterministic corpus' featurizer output
                     (python oracle AND rust featurize_states, feats + raylen)
                     plus the active plane-map constants — the T1 regression
                     probe. Runs against pre- and post-change code (constants
                     that predate the gate are read via getattr defaults).
  dump --out PATH    save the python-oracle corpus features (npz) for the
                     cross-version consistency check.
  v2-semantics --v1 PATH
                     v2-only semantic checks: shared planes + fork re-index vs
                     the v1 dump, liveK monotonicity and corpus coverage, the
                     empty-board scalar values, and an independent recompute of
                     the three new scalar planes (spec §1.3-1.4).
  typing             plane-map / typing-set assertions for the active version
                     (T2, spec §1.2).
  stem-lift          gen_stem_weight structure per derivation §8 against the
                     active map (T2; torch, fp32).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random

import numpy as np

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield_eq import constants as C
from hexfield_eq.engine_facts import facts_from_engine
from hexfield_eq.features import build_position, build_ray_lengths
from hexfield_eq.geometry import unpack_action_id

# --- deterministic corpus -------------------------------------------------------
# The ply-0 empty board (the spec §1.4 empty-board case), seeded random
# playouts, and the crafted 5-in-line game of test_hexfield_eq_rust_parity.py
# (drives the strong end of the line/liveK planes; a purely random corpus
# rarely forms 4+-in-window).

_LINE_GAME = [
    (0, 0),  # P0
    (0, 3),  # P1
    (0, 4),  # P1
    (1, 0),  # P0
    (2, 0),  # P0
    (1, 3),  # P1
    (1, 4),  # P1
    (3, 0),  # P0
    (4, 0),  # P0
    (2, 3),  # P1
]


def _random_state(seed: int, plies: int):
    state = api.new_game()
    rng = random.Random(seed)
    for _ in range(plies):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        q, r = unpack_action_id(rng.choice(ids))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            break
    return state


def corpus_states() -> list:
    states = [api.new_game()]  # ply 0: the empty-board case
    for seed in range(4):
        for plies in (1, 3, 7, 14, 24, 36):
            st = _random_state(seed * 1000 + plies, plies)
            if api.terminal(st) is None:
                states.append(st)
    st = api.new_game()
    for q, r in _LINE_GAME:
        if api.terminal(st) is not None:
            break
        api.apply_action(st, PlacementAction(AxialCoord(q=q, r=r)))
    if api.terminal(st) is None:
        states.append(st)
    return states


def _oracle_rows(states) -> list[tuple[object, np.ndarray, np.ndarray]]:
    """(support, feats, raylen) per corpus state, via the python oracle."""

    rows = []
    for st in states:
        facts = facts_from_engine(api.to_python_state(st))
        sup, feats = build_position(facts)
        rows.append((sup, feats, build_ray_lengths(facts, sup)))
    return rows


def _map_constants() -> dict:
    """The active plane-map constants; getattr defaults keep this runnable
    against the pre-gate code (which has no FEATURE_VERSION)."""

    return {
        "feature_version": int(getattr(C, "FEATURE_VERSION", 1)),
        "num_features": int(C.NUM_FEATURES),
        "f_own_fork": int(C.F_OWN_FORK),
        "f_opp_fork": int(C.F_OPP_FORK),
        "n_axis_quantities": int(getattr(C, "N_AXIS_QUANTITIES", 4)),
    }


# --- subcommands ----------------------------------------------------------------


def cmd_corpus_hash(_args) -> dict:
    states = corpus_states()
    h = hashlib.sha256()
    for sup, feats, raylen in _oracle_rows(states):
        h.update(np.int64(sup.num_nodes).tobytes())
        h.update(feats.tobytes())
        h.update(raylen.tobytes())
    out = _map_constants()
    out["num_states"] = len(states)
    out["python_sha256"] = h.hexdigest()
    try:
        from hexfield_eq import _rust
    except ImportError:
        _rust = None
    if _rust is None:
        out["rust_sha256"] = None
    else:
        hr = hashlib.sha256()
        for payload in _rust.featurize_states(states):
            hr.update(np.int64(payload["num_nodes"]).tobytes())
            hr.update(bytes(payload["feats"]))
            hr.update(bytes(payload["raylen"]))
        out["rust_sha256"] = hr.hexdigest()
    return out


def cmd_dump(args) -> dict:
    rows = _oracle_rows(corpus_states())
    np.savez(
        args.out,
        num_features=np.int64(C.NUM_FEATURES),
        node_counts=np.array([sup.num_nodes for sup, _, _ in rows], dtype=np.int64),
        coords=np.concatenate([sup.coords for sup, _, _ in rows], axis=0),
        feats=np.concatenate([feats for _, feats, _ in rows], axis=0),
    )
    return {**_map_constants(), "num_states": len(rows)}


def cmd_v2_semantics(args) -> dict:
    assert C.FEATURE_VERSION == 2, "run with HEXFIELD_EQ_FEATURE_VERSION=2"
    states = corpus_states()
    rows = _oracle_rows(states)

    # --- cross-version consistency vs the v1 dump (spec §1.2: planes 0-22
    # unchanged, fork planes re-indexed 23/24 -> 41/42 with unchanged values).
    v1 = np.load(args.v1)
    assert int(v1["num_features"]) == 25
    counts = [sup.num_nodes for sup, _, _ in rows]
    assert np.array_equal(v1["node_counts"], np.array(counts, dtype=np.int64)), (
        "support drift between the v1 and v2 corpus builds"
    )
    coords = np.concatenate([sup.coords for sup, _, _ in rows], axis=0)
    assert np.array_equal(v1["coords"], coords)
    feats = np.concatenate([f for _, f, _ in rows], axis=0)
    v1_feats = v1["feats"]
    assert np.array_equal(v1_feats[:, :23], feats[:, :23]), (
        "shared planes 0-22 differ between version 1 and version 2"
    )
    assert np.array_equal(v1_feats[:, 23], feats[:, C.F_OWN_FORK])
    assert np.array_equal(v1_feats[:, 24], feats[:, C.F_OPP_FORK])

    # --- liveK monotonicity: for both sides and every axis, the clean-window
    # count is monotone in the stone threshold (live >= live3 >= live4 >= live5,
    # all on the same /LIVE_NORM scale).
    for base_live, base3, base4, base5 in (
        (C.F_OWN_LIVE_Q, C.F_OWN_LIVE3_Q, C.F_OWN_LIVE4_Q, C.F_OWN_LIVE5_Q),
        (C.F_OPP_LIVE_Q, C.F_OPP_LIVE3_Q, C.F_OPP_LIVE4_Q, C.F_OPP_LIVE5_Q),
    ):
        for a in range(3):
            live = feats[:, base_live + a]
            l3, l4, l5 = feats[:, base3 + a], feats[:, base4 + a], feats[:, base5 + a]
            assert (l3 <= live + 1e-9).all() and (l4 <= l3 + 1e-9).all() and (l5 <= l4 + 1e-9).all()

    # --- coverage: the crafted line game must light up the strong thresholds
    # (otherwise the parity above is vacuous at the liveK end).
    live4_max = float(feats[:, C.F_OWN_LIVE4_Q : C.F_OPP_LIVE4_QR + 1].max())
    live5_max = float(feats[:, C.F_OWN_LIVE5_Q : C.F_OPP_LIVE5_QR + 1].max())
    assert live4_max > 0.0, "corpus never exercised live4"
    assert live5_max > 0.0, "corpus never exercised live5"

    # --- empty-board case (spec §1.4): ply 0 state is rows[0].
    sup0, feats0, _ = rows[0]
    assert sup0.num_nodes == 7
    assert np.array_equal(feats0[:, C.F_PLY], np.zeros(7, dtype=np.float32))
    assert np.array_equal(feats0[:, C.F_DIST_CENTROID], np.zeros(7, dtype=np.float32))
    assert np.array_equal(
        feats0[:, C.F_SPREAD], np.full(7, np.float32(1.0 / 16.0))
    )

    # --- independent recompute of the three scalars (spec §1.4 formulas) on
    # every non-empty state, to 1e-6 (the graded-float tolerance class).
    checked = 0
    for st, (sup, f, _) in zip(states, rows):
        facts = facts_from_engine(api.to_python_state(st))
        stones = facts.stones()
        if not stones:
            continue
        ply = min(facts.placements_made, 96) / 96.0
        assert np.abs(f[:, C.F_PLY] - np.float32(ply)).max() == 0.0
        sq = np.array([q for q, _ in stones], dtype=np.float64)
        sr = np.array([r for _, r in stones], dtype=np.float64)
        cq, cr = sq.mean(), sr.mean()

        def hexd(dq, dr):
            return (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) / 2.0

        spread = max(1.0, float(hexd(sq - cq, sr - cr).max()))
        assert np.abs(f[:, C.F_SPREAD] - min(spread, 16.0) / 16.0).max() <= 1e-6
        nq = sup.coords[:, 0].astype(np.float64)
        nr = sup.coords[:, 1].astype(np.float64)
        want = np.minimum(hexd(nq - cq, nr - cr) / (2.0 * spread), 1.0)
        assert np.abs(f[:, C.F_DIST_CENTROID] - want).max() <= 1e-6
        checked += 1
    assert checked > 0

    return {**_map_constants(), "live4_max": live4_max, "live5_max": live5_max}


def cmd_typing(_args) -> dict:
    from hexfield_eq import equivariant as eq

    version = C.FEATURE_VERSION
    axis_planes = set(eq._AXIS_PLANES)
    scalar_planes = set(eq._SCALAR_PLANES)
    assert axis_planes | scalar_planes == set(range(C.NUM_FEATURES))
    assert not (axis_planes & scalar_planes)
    assert eq.AXIS_PLANE_BASE == C.AXIS_PLANE_BASE == 11
    assert eq.N_AXIS_QUANTITIES == C.N_AXIS_QUANTITIES

    if version == 1:
        # The pre-gate 25-plane map, bit for bit (T1 regression half).
        assert C.NUM_FEATURES == 25
        assert C.N_AXIS_QUANTITIES == 4
        assert (C.F_OWN_FORK, C.F_OPP_FORK) == (23, 24)
        assert axis_planes == set(range(11, 23))
        assert scalar_planes == set(range(11)) | {23, 24}
    else:
        # The spec §1.2 46-plane map, including the fork re-index.
        assert C.NUM_FEATURES == 46
        assert C.N_AXIS_QUANTITIES == 10
        assert (C.F_OWN_FORK, C.F_OPP_FORK) == (41, 42)
        assert (C.F_PLY, C.F_DIST_CENTROID, C.F_SPREAD) == (43, 44, 45)
        assert axis_planes == set(range(11, 41))
        assert scalar_planes == set(range(11)) | set(range(41, 46))
        # plane = AXIS_PLANE_BASE + q*N_AXES + a for the liveK quantities q=4..9.
        assert C.F_OWN_LIVE3_Q == 11 + 4 * 3
        assert C.F_OPP_LIVE3_Q == 11 + 5 * 3
        assert C.F_OWN_LIVE4_Q == 11 + 6 * 3
        assert C.F_OPP_LIVE4_Q == 11 + 7 * 3
        assert C.F_OWN_LIVE5_Q == 11 + 8 * 3
        assert C.F_OPP_LIVE5_Q == 11 + 9 * 3
    return _map_constants()


def cmd_stem_lift(_args) -> dict:
    """gen_stem_weight structure per derivation §8 against the ACTIVE map:
    scalar-plane columns are slot-constant; each axis quantity's 3 columns form
    an equivariant triple M_reg(g) w_a == w_{cosp[g](a)}. Catches a stale
    typing set (the mis-placed-fork trap of spec §1.2) that the plane-map
    asserts alone would miss if equivariant.py hardcoded its own sets."""

    import torch

    from hexfield_eq import equivariant as eq

    torch.manual_seed(0)
    w0 = torch.randn(7, C.CHANNELS, C.NUM_FEATURES, dtype=torch.float64)
    ws = eq.gen_stem_weight(w0)  # (7, NF, C)
    G = eq.build_group()
    regp, cosp = G["regp"], G["cosp"]
    tol = 1e-12  # float64 round-off through the Reynolds sum

    # The per-column relations below hold at the CENTER tap only (tapp[g][0] == 0
    # for every g; direction taps satisfy the constraint jointly ACROSS taps),
    # matching the derivation test's tap-0 structure check.
    for p in eq._SCALAR_PLANES:
        col = ws[0, p].reshape(12, C.C_ORBIT)
        assert (col - col.mean(0, keepdim=True)).abs().max() < tol, (
            f"scalar plane {p} not slot-constant"
        )
    for q in range(eq.N_AXIS_QUANTITIES):
        cols = [
            ws[0, eq.AXIS_PLANE_BASE + q * 3 + a].reshape(12, C.C_ORBIT)
            for a in range(3)
        ]
        for g in range(12):
            for a in range(3):
                lhs = cols[a][regp[g]]
                rhs = cols[cosp[g][a]]
                assert (lhs - rhs).abs().max() < tol, (
                    f"axis quantity {q} not an equivariant triple (g={g}, a={a})"
                )
    return {**_map_constants(), "c_orbit": int(C.C_ORBIT)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("corpus-hash")
    p_dump = sub.add_parser("dump")
    p_dump.add_argument("--out", required=True)
    p_sem = sub.add_parser("v2-semantics")
    p_sem.add_argument("--v1", required=True)
    sub.add_parser("typing")
    sub.add_parser("stem-lift")
    args = parser.parse_args()
    handler = {
        "corpus-hash": cmd_corpus_hash,
        "dump": cmd_dump,
        "v2-semantics": cmd_v2_semantics,
        "typing": cmd_typing,
        "stem-lift": cmd_stem_lift,
    }[args.cmd]
    print(json.dumps({"ok": True, **handler(args)}))


if __name__ == "__main__":
    main()
