"""hexfield_eq ray-length parity + D6 covariance (Phase L0 gate).

docs/PLAN_REGISTER_LANE_RAY_ATTENTION.md Phase L0 /
docs/SPEC_REGISTER_LANE_RAY_ATTENTION.md §2. The side-relative ray lengths
(u8[RAYLEN_SLOTS] per cell: side own/opp x axis Q/R/QR x dir +/-, values
0..RAY_REACH, terminal blocker included) are computed three ways — the serve
Rust walk (``features.rs::build_ray_lengths`` via ``featurize_states``), the
train Rust walk (``replay_expand.rs`` over the reconstructed board), and the
Python oracle (``features.ray_lengths_for_cell``) — and pinned against each
other elementwise-EXACTLY (u8, no tolerance):

  1. SERVE parity — ``_rust.featurize_states``'s ``raylen`` vs the oracle on
     sampled engine states + the crafted line game.
  2. TRAIN parity across ALL 12 D6 — ``_rust.expand_shard_train``'s ``raylen``
     buffer vs the oracle on the D6-transformed facts (serve == train follows
     by transitivity through the oracle).
  3. D6 COVARIANCE — ``raylen[g.x, s, sigma_g(a), dir_g] == raylen[x, s, a, dir]``
     for all 12 g, with the reflection direction-swap derived by transforming
     the ray direction vector (no hand table).
  4. WALK SEMANTICS — a crafted board pins the blocker/pass-through/off-support
     rules of the L1 walk.
  5. WIRE ROUND-TRIP — the ``raylen`` buffer through ``_rust.build_serve_groups``
     (pad rows 0) and through ``batching.collate_rows`` reproduces the per-row
     Rust output exactly.

Runs in the hexgt-build venv via PYTHONPATH=packages/hexfield_eq/python (plus
the shared packages). CPU-only.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield_eq import constants as C
from hexfield_eq.engine_facts import facts_from_engine
from hexfield_eq.batching import collate_rows
from hexfield_eq.expand_backends import (
    _resolve_support_radius,
    _window_columns_as_bytes,
)
from hexfield_eq.features import (
    PositionFacts,
    build_position,
    build_ray_lengths,
    ray_lengths_for_cell,
    transform_facts,
)
from hexfield_eq.geometry import apply_d6, disk_offsets, pack_action_id, unpack_action_id
from hexfield_eq.samples import STV_HORIZONS, HexfieldSampleData
from hexfield_eq.shards import write_compact_shard
from hexfield_eq.window import concat_packed, load_packed_shard

try:
    from hexfield_eq import _rust
except ImportError:  # pragma: no cover
    _rust = None

needs_rust = pytest.mark.skipif(
    _rust is None, reason="hexfield_eq._rust not built (see the Phase-1 build gate)"
)

RL = C.RAYLEN_SLOTS

# Crafted game driving a length-5 P0 wall on the Q axis (the rust_parity
# corpus's line game): guarantees blocked, pass-through, AND full-reach rays.
_LINE_GAME = [
    (0, 0),
    (0, 3),
    (0, 4),
    (1, 0),
    (2, 0),
    (1, 3),
    (1, 4),
    (3, 0),
    (4, 0),
    (2, 3),
]

_AXIS_VECS = ((1, 0), (0, 1), (1, -1))  # Q, R, QR


# --- state / row generation (rust_parity idioms) --------------------------------


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


def _decision_states():
    states = []
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


def _rows_from_moves(moves, game_id: str) -> list[HexfieldSampleData]:
    state = api.new_game()
    rows: list[HexfieldSampleData] = []
    for q, r in moves:
        if api.terminal(state) is not None:
            break
        action_id = pack_action_id(q, r)
        if action_id not in set(api.legal_action_ids(state)):
            raise AssertionError(f"crafted move {(q, r)} is not legal")
        facts = facts_from_engine(api.to_python_state(state))
        rows.append(
            HexfieldSampleData(
                game_id=game_id,
                turn_index=len(facts.records),
                current_player=facts.current_player,
                phase=facts.phase,
                records=facts.records,
                first_stone=facts.first_stone,
                policy=((int(action_id), 1.0),),
                value=0.0,
                moves_left=-1.0,
            )
        )
        api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
    return rows


def _random_moves(seed: int, plies: int) -> list[tuple[int, int]]:
    state = api.new_game()
    rng = random.Random(seed)
    moves: list[tuple[int, int]] = []
    for _ in range(plies):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        aid = rng.choice(ids)
        q, r = unpack_action_id(aid)
        moves.append((q, r))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            break
    return moves


def _build_window(tmpdir: Path):
    samples: list[HexfieldSampleData] = []
    samples += _rows_from_moves(_LINE_GAME, "line")
    for seed in range(3):
        moves = _random_moves(3000 + seed, 30)
        if moves:
            samples += _rows_from_moves(moves, f"rand{seed}")
    assert samples, "no rows generated"
    shard = tmpdir / "epoch_000000" / "game_0.npz"
    write_compact_shard(shard, samples, short_term_value_horizons=STV_HORIZONS)
    return concat_packed([load_packed_shard(shard)])


def _facts_of_sample(sample: HexfieldSampleData) -> PositionFacts:
    return PositionFacts(
        records=tuple(sample.records),
        current_player=int(sample.current_player),
        phase=sample.phase,
        first_stone=sample.first_stone,
    )


# --- axis / direction transport under g -----------------------------------------


def _slot_perm(g: int) -> list[int]:
    """perm[slot] = the slot the value at ``slot`` lands on after applying g:
    the ray direction ``dir_sign * axis_vec`` maps to ``g . dir``, matched
    against +/- the axis vectors. The side index is g-invariant."""

    perm = [0] * RL
    for ai, (dq, dr) in enumerate(_AXIS_VECS):
        for di, sign in ((0, 1), (1, -1)):
            tq, tr = apply_d6(g, sign * dq, sign * dr)
            for aj, (aq, ar) in enumerate(_AXIS_VECS):
                if (tq, tr) == (aq, ar):
                    tgt = aj * 2 + 0
                    break
                if (tq, tr) == (-aq, -ar):
                    tgt = aj * 2 + 1
                    break
            else:  # pragma: no cover - every axis dir maps to an axis dir
                raise AssertionError(f"g={g} maps axis dir to off-axis {(tq, tr)}")
            for side in range(2):
                perm[side * 6 + ai * 2 + di] = side * 6 + tgt
    return perm


# --- 1. serve parity --------------------------------------------------------------


@needs_rust
def test_serve_raylen_parity() -> None:
    states = _decision_states()
    assert len(states) >= 15
    payloads = _rust.featurize_states(states)

    saw_blocked = False
    saw_full = False
    for state, payload in zip(states, payloads):
        facts = facts_from_engine(api.to_python_state(state))
        sup, _feats = build_position(facts)
        n = sup.num_nodes
        assert payload["num_nodes"] == n
        rust = np.frombuffer(payload["raylen"], dtype=np.uint8).reshape(n, RL)
        oracle = build_ray_lengths(facts, sup)
        assert np.array_equal(rust, oracle), "serve raylen != oracle"
        saw_blocked = saw_blocked or bool(((rust >= 1) & (rust <= 4)).any())
        saw_full = saw_full or bool((rust == C.RAY_REACH).any())
    # The corpus really exercises truncated AND full-reach rays.
    assert saw_blocked and saw_full


# --- 2. train parity, all 12 D6 -----------------------------------------------------


@needs_rust
def test_train_raylen_parity_all_12_d6() -> None:
    with tempfile.TemporaryDirectory() as td:
        window = _build_window(Path(td))
        n = window.n
        assert n > 0
        columns = _window_columns_as_bytes(window)
        radius = int(_resolve_support_radius())

        from hexfield_eq.expand_backends import _row_view_to_sample

        samples = [_row_view_to_sample(window.row_view(k)) for k in range(n)]

        for sym in range(12):
            result = _rust.expand_shard_train(
                columns,
                int(window.n),
                list(range(n)),
                [sym] * n,
                len(STV_HORIZONS),
                radius,
                False,
            )
            r = int(result["num_rows"])
            assert r == n
            node_off = np.frombuffer(
                bytes(result["node_off"]), dtype=np.int64, count=r + 1
            )
            total_nodes = int(node_off[r])
            raylen = np.frombuffer(
                bytes(result["raylen"]), dtype=np.uint8, count=total_nodes * RL
            ).reshape(-1, RL)

            for k in range(n):
                facts = transform_facts(_facts_of_sample(samples[k]), sym)
                sup, _feats = build_position(facts)
                a, b = int(node_off[k]), int(node_off[k + 1])
                assert b - a == sup.num_nodes, f"N mismatch row {k} sym {sym}"
                oracle = build_ray_lengths(facts, sup)
                assert np.array_equal(raylen[a:b], oracle), (
                    f"train raylen != oracle at row {k} sym {sym}"
                )


# --- 3. D6 covariance ----------------------------------------------------------------


def test_raylen_d6_covariance() -> None:
    """raylen[g.x, s, sigma_g(a), dir_g] == raylen[x, s, a, dir]: the data is
    recomputed from the transformed board, so it must transport exactly like a
    (side-invariant) axis-direction field."""

    states = _decision_states()[:8]
    checks = 0
    for state in states:
        facts = facts_from_engine(api.to_python_state(state))
        sup, _ = build_position(facts)
        base = build_ray_lengths(facts, sup)
        base_of = {
            (int(sup.coords[i][0]), int(sup.coords[i][1])): base[i]
            for i in range(sup.num_nodes)
        }
        for g in range(12):
            perm = _slot_perm(g)
            tfacts = transform_facts(facts, g)
            tsup, _ = build_position(tfacts)
            trl = build_ray_lengths(tfacts, tsup)
            for i in range(sup.num_nodes):
                x = (int(sup.coords[i][0]), int(sup.coords[i][1]))
                gx = apply_d6(g, x[0], x[1])
                j = tsup.index[gx]
                for slot in range(RL):
                    assert trl[j][perm[slot]] == base_of[x][slot], (state, g, x, slot)
                checks += 1
    assert checks > 0


# --- 4. walk semantics (crafted board) -------------------------------------------------


def test_walk_blocker_pass_through_and_off_support() -> None:
    support = set(disk_offsets(5))
    # me=0. Own stone at (2,0) passes through; opp stone at (3,0) is a terminal
    # blocker (included).
    owner_at = {(2, 0): 0, (3, 0): 1}

    rays = ray_lengths_for_cell(owner_at, support, 0, 0, 0)
    assert rays[0 * 6 + 0 * 2 + 0] == 3  # own, Q, +: empty, own, OPP-block
    assert rays[1 * 6 + 0 * 2 + 0] == 2  # opp, Q, +: empty, OWN-block
    # Unobstructed directions reach the full 5.
    assert rays[0 * 6 + 1 * 2 + 0] == C.RAY_REACH  # own, R, +
    assert rays[0 * 6 + 0 * 2 + 1] == C.RAY_REACH  # own, Q, -

    # Off-support truncation: from (3,0) the Q+ walk leaves the radius-5 disk
    # after (5,0) — geometric stop, nothing included beyond the support.
    rays_edge = ray_lengths_for_cell({}, support, 3, 0, 0)
    assert rays_edge[0 * 6 + 0 * 2 + 0] == 2  # (4,0), (5,0), then off-support
    assert rays_edge[0 * 6 + 0 * 2 + 1] == C.RAY_REACH  # Q- back through center

    # The blocker itself gets rays too (occupancy of x is never consulted), and
    # a blocker adjacent in-walk truncates to exactly its distance.
    rays_stone = ray_lengths_for_cell(owner_at, support, 2, 0, 0)
    assert rays_stone[0 * 6 + 0 * 2 + 0] == 1  # own, Q, +: (3,0) opp included


# --- 5. wire round-trip -----------------------------------------------------------------


@needs_rust
def test_wire_round_trip_serve_pack_and_collate() -> None:
    states = _decision_states()[:6]
    payloads = _rust.featurize_states(states)
    # Wire order is size-DESCENDING (payload.rs contract).
    order = sorted(
        range(len(states)), key=lambda i: -int(payloads[i]["num_nodes"])
    )

    feats_parts, qr_parts, nbr_parts, rl_parts, sizes = [], [], [], [], []
    per_row_rl = []
    for i in order:
        p = payloads[i]
        n = int(p["num_nodes"])
        sizes.append(n)
        feats32 = np.frombuffer(p["feats"], dtype=np.float32).reshape(n, C.NUM_FEATURES)
        feats_parts.append(feats32.astype(np.float16).tobytes())
        qr_parts.append(p["coords"])
        nbr32 = np.frombuffer(p["nbr"], dtype=np.int32).reshape(n, 6)
        nbr_parts.append(
            np.where(nbr32 < 0, 0xFFFF, nbr32).astype(np.uint16).tobytes()
        )
        rl = np.frombuffer(p["raylen"], dtype=np.uint8).reshape(n, RL)
        rl_parts.append(rl.tobytes())
        per_row_rl.append(rl)

    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    groups = _rust.build_serve_groups(
        b"".join(feats_parts),
        b"".join(qr_parts),
        b"".join(nbr_parts),
        b"".join(rl_parts),
        offsets.tolist(),
    )

    covered = 0
    for grp in groups:
        g, pad_to = int(grp["g"]), int(grp["pad_to"])
        start = int(grp["start"])
        rl_buf = np.frombuffer(bytes(grp["raylen"]), dtype=np.uint8).reshape(
            g, pad_to, RL
        )
        for k in range(g):
            n = sizes[start + k]
            assert np.array_equal(rl_buf[k, :n], per_row_rl[start + k])
            assert not rl_buf[k, n:].any(), "pad rows must be raylen 0"
            covered += 1
    assert covered == len(states)

    # Train-side collate: the same per-row arrays through batching.collate_rows.
    rows = []
    raylen_rows = []
    for i in order[:3]:
        facts = facts_from_engine(api.to_python_state(states[i]))
        sup, feats = build_position(facts)
        rows.append((sup, feats))
        raylen_rows.append(build_ray_lengths(facts, sup))
    pad_to = max(sup.num_nodes for sup, _ in rows) + 7
    batch = collate_rows(rows, pad_to=pad_to, raylen=raylen_rows)
    assert batch["raylen"].shape == (3, pad_to, RL)
    assert batch["raylen"].dtype == __import__("torch").uint8
    for gi, (sup, _f) in enumerate(rows):
        n = sup.num_nodes
        assert np.array_equal(batch["raylen"][gi, :n].numpy(), raylen_rows[gi])
        assert not batch["raylen"][gi, n:].any()
    # Without raylen the batch keeps the pre-L key set.
    assert "raylen" not in collate_rows(rows, pad_to=pad_to)


# --- 6. train-batch path (ExpandedRow -> collate_training) --------------------------


@needs_rust
def test_train_batch_carries_raylen_end_to_end(monkeypatch) -> None:
    """The D-S15 threading: both expand backends populate ExpandedRow.raylen
    identically (serial = the oracle in expand_sample; rust = the kernel
    buffer), and collate_training packs it into batch['raylen'] with pad rows
    0 — so a train batch carries the raylen key end-to-end. The serial oracle
    is gated off under C/A layouts (spec D-S29), so the test forces it on."""

    import torch

    from hexfield_eq import samples as samples_mod
    from hexfield_eq.batching import collate_training
    from hexfield_eq.expand_backends import expand_rows

    monkeypatch.setattr(samples_mod, "_EXPAND_RAYLEN", True)

    with tempfile.TemporaryDirectory() as td:
        window = _build_window(Path(td))
        n = window.n
        d6 = (np.arange(n, dtype=np.int64) % 12).astype(np.int64)
        rows_s, valid_s = expand_rows(window, None, d6, backend="serial")
        rows_r, valid_r = expand_rows(window, None, d6, backend="rust")
        assert np.array_equal(valid_s, valid_r)
        for k, (a, b) in enumerate(zip(rows_s, rows_r)):
            assert a is not None and b is not None
            assert a.raylen.shape == (a.support.num_nodes, RL)
            assert a.raylen.dtype == np.uint8
            assert np.array_equal(a.raylen, b.raylen), f"serial != rust raylen row {k}"

        subset = rows_s[:4]
        pad_to = max(r.support.num_nodes for r in subset) + 5
        batch = collate_training(subset, pad_to=pad_to)
        assert batch["raylen"].shape == (len(subset), pad_to, RL)
        assert batch["raylen"].dtype == torch.uint8
        for g, row in enumerate(subset):
            n_g = row.support.num_nodes
            assert np.array_equal(batch["raylen"][g, :n_g].numpy(), row.raylen)
            assert not batch["raylen"][g, n_g:].any()
