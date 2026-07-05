"""Tests for shrimp losses, targets, shards, and batching.

Several tests pin shrimp outputs against independent, in-test first-principles
references (the retired dense_cnn_restnet oracle has no public counterpart):
- 65-bin binned-value helpers vs a numpy two-hot / softmax-expectation reference
- segment legal-prefix CE vs a masked dense soft-CE reference on embedded rows
- masked binned losses including zero-denominator rows
- short-term-value targets and future-opponent-policy vs first-principles EMA /
  next-opponent-decision references
- legacy restnet shard cross-read via an in-test writer that mirrors the reader's
  on-disk contract
- finalize invariants (hard z, moves-left countdown, truncated, fast-mask)
- expansion: policy slot mapping, D6 symmetry, off-legal error
- shrimp_compact_v1 writer round-trip plus sidecar
- legacy restnet shard cross-read with derived legality and win-now
- micro-bucket accumulation vs monolithic loss/grads in fp64
- pair-budget bucket rule
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from shrimp_testkit import api, random_playout

from shrimp import constants as C
from shrimp.batching import (
    collate_training,
    pair_budget_microbuckets,
    split_stvalue_columns,
    step_global_denominators,
)
from shrimp.engine_facts import facts_from_engine, player_int
from shrimp.geometry import apply_d6, pack_action_id, unpack_action_id
from shrimp.losses import (
    binned_value_loss,
    decode_binned_value,
    decode_moves_left,
    shrimp_loss,
    scalar_to_binned_target,
    segment_policy_ce,
)
from shrimp.model import ShrimpNet
from shrimp.samples import (
    STV_HORIZONS,
    ShrimpSampleData,
    _future_opponent_policy,
    _short_term_value_targets,
    expand_sample,
    finalize_game_samples,
)
from shrimp.shards import (
    LEGACY_RESTNET_SCHEMA_VERSION,
    _pack_qr,
    read_compact_shard,
    read_legacy_restnet_shard,
    write_compact_shard,
)
from hexo_engine.types import AxialCoord, PlacementAction


def _sample_from_state(state, rng: random.Random, turn_index: int) -> ShrimpSampleData:
    facts = facts_from_engine(api.to_python_state(state))
    legal = sorted(api.legal_action_ids(state))
    chosen = rng.sample(legal, k=min(3, len(legal)))
    weights = [rng.random() + 0.1 for _ in chosen]
    total = sum(weights)
    policy = tuple((aid, w / total) for aid, w in zip(chosen, weights))
    return ShrimpSampleData(
        game_id="test",
        turn_index=turn_index,
        current_player=facts.current_player,
        phase=facts.phase,
        records=facts.records,
        first_stone=facts.first_stone,
        own_hot=facts.own_hot,
        opp_hot=facts.opp_hot,
        own_win=facts.own_win,
        opp_win=facts.opp_win,
        policy=policy,
        metadata={"pcr_full": True},
    )


def _make_game(seed: int, max_plies: int = 24):
    """Pending (player, sample, root_value) decisions from one random game."""

    rng = random.Random(seed)
    state = api.new_game()
    pending = []
    winner = None
    for ply in range(max_plies):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        sample = _sample_from_state(state, rng, ply)
        pending.append((sample.current_player, sample, rng.uniform(-0.8, 0.8)))
        q, r = unpack_action_id(rng.choice(ids))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            winner = player_int(api.terminal(state).winner)
            break
    return pending, winner


def _ref_scalar_to_binned_target(values: torch.Tensor) -> torch.Tensor:
    """First-principles two-hot reference: interpolate each scalar in [-1, 1]
    onto the VALUE_BINS support ``linspace(-1, 1, VALUE_BINS)`` and split its unit
    mass across the two adjacent bins by linear weight. Computed in fp64 with
    plain numpy so it shares no code with the production helper."""

    v = values.double().reshape(-1).numpy()
    out = np.zeros((v.size, C.VALUE_BINS), dtype=np.float64)
    pos = (v + 1.0) * ((C.VALUE_BINS - 1) / 2.0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    up = pos - lo
    rows = np.arange(v.size)
    np.add.at(out, (rows, lo), 1.0 - up)
    np.add.at(out, (rows, hi), up)
    return torch.from_numpy(out).reshape(*values.shape, C.VALUE_BINS)


def test_bin_helpers_match_reference() -> None:
    """scalar_to_binned_target / decode_binned_value against an independent
    two-hot-interpolation reference (replaces the retired dense_cnn oracle)."""

    torch.manual_seed(0)
    values = torch.rand(64) * 2.0 - 1.0
    mine = scalar_to_binned_target(values).double()
    theirs = _ref_scalar_to_binned_target(values)
    assert torch.allclose(mine, theirs, atol=1e-12)
    # Each row is a valid two-hot distribution.
    assert torch.allclose(mine.sum(dim=-1), torch.ones(64, dtype=torch.float64))

    logits = torch.randn(16, C.VALUE_BINS)
    # decode = softmax expectation over the linspace(-1, 1) support, clamped.
    bins = torch.linspace(-1.0, 1.0, C.VALUE_BINS, dtype=torch.float64)
    ref = (torch.softmax(logits.double(), dim=-1) * bins).sum(dim=-1).clamp(-1.0, 1.0)
    assert torch.allclose(decode_binned_value(logits).double(), ref, atol=1e-7)


def _ref_masked_soft_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Independent masked soft-cross-entropy reference: per row, softmax over the
    masked slots only, then -sum(target * log p), averaged over rows. Slots
    outside the mask are excluded from the normalizer (their logits set to -inf).
    Mirrors what the retired dense_cnn ``soft_cross_entropy`` oracle computed."""

    lg = logits.double().clone()
    lg[~mask] = float("-inf")
    log_p = lg - torch.logsumexp(lg, dim=-1, keepdim=True)
    # target is 0 off the mask; log_p is -inf there, so zero those terms
    # explicitly to avoid 0 * -inf = nan.
    contrib = torch.where(mask, target.double() * log_p, torch.zeros_like(log_p))
    per_row = -contrib.sum(dim=-1)
    return per_row.mean()


def test_segment_ce_matches_dense_masked_ce() -> None:
    """Segment-packed policy CE equals a dense masked soft-CE over the same
    logits/targets scattered onto arbitrary grid cells. Pins that packing the
    legal prefix is equivalent to masking a dense area (replaces the dense_cnn
    oracle with an in-test masked-softmax reference)."""

    torch.manual_seed(1)
    rng = np.random.RandomState(2)
    b, npad, area = 3, 24, 41 * 41
    legal_counts = torch.tensor([10, 17, 5])
    logits = torch.randn(b, npad)
    target = torch.zeros(b, npad)
    for g in range(b):
        l = int(legal_counts[g])
        t = torch.rand(l)
        target[g, :l] = t / t.sum()

    dense_logits = torch.randn(b, area)  # arbitrary values off the embedded cells
    dense_target = torch.zeros(b, area)
    dense_mask = torch.zeros(b, area, dtype=torch.bool)
    for g in range(b):
        l = int(legal_counts[g])
        cells = rng.choice(area, size=l, replace=False)
        dense_logits[g, cells] = logits[g, :l]
        dense_target[g, cells] = target[g, :l]
        dense_mask[g, cells] = True

    mine = segment_policy_ce(logits, legal_counts, target)
    theirs = _ref_masked_soft_ce(dense_logits, dense_target, dense_mask)
    assert torch.allclose(mine.double(), theirs, atol=1e-6)


def _ref_binned_value_loss(logits, target, mask=None):
    """First-principles binned-value CE reference: two-hot the scalar targets,
    then -sum(p_target * log_softmax(logits)) per row, averaged over the (masked)
    rows. fp64 throughout; shares no code with the production loss."""

    p = _ref_scalar_to_binned_target(torch.as_tensor(target, dtype=torch.float64))
    log_p = torch.log_softmax(logits.double(), dim=-1)
    per_row = -(p * log_p).sum(dim=-1)
    if mask is None:
        return per_row.mean()
    m = torch.as_tensor(mask, dtype=torch.float64)
    denom = float(m.sum().item())
    if denom <= 0.0:
        return torch.zeros((), dtype=torch.float64)
    return (per_row * m).sum() / denom


def test_binned_value_loss_matches_reference() -> None:
    """binned_value_loss (scalar targets, masked and unmasked) against an
    independent two-hot + masked-CE reference (replaces the dense_cnn oracle)."""

    torch.manual_seed(3)
    logits = torch.randn(8, C.VALUE_BINS)
    target = torch.rand(8) * 2.0 - 1.0
    assert torch.allclose(
        binned_value_loss(logits, target).double(),
        _ref_binned_value_loss(logits, target),
        atol=1e-6,
    )
    mask = torch.tensor([1.0, 0, 1, 0, 1, 1, 0, 0])
    assert torch.allclose(
        binned_value_loss(logits, target, mask=mask).double(),
        _ref_binned_value_loss(logits, target, mask=mask),
        atol=1e-6,
    )
    zero = binned_value_loss(logits, target, mask=torch.zeros(8))
    assert float(zero) == 0.0  # all-zero mask yields exactly 0.0


def test_binned_value_loss_bins_scalar_targets_in_fp32() -> None:
    """Under train autocast ``logits`` are fp16, so binned_value_loss casts the
    scalar target to fp16 (logits.dtype) at entry. The two-hot interpolation
    ``position = (v+1)*(VALUE_BINS-1)/2`` must then be computed in fp32, not fp16:
    near position ~32 (v~0) the fp16 ulp is ~1/64, mis-splitting the two-hot by that
    fraction of a bin. The fix casts the (already-fp16) scalar to fp32 before
    binning so the interpolation is fp32-exact.

    Adversarial value: for this fp16 input the fp16-arithmetic position rounds to
    32.125 while the exact (fp32/fp64) position is 32.140625 — a half-fp16-ulp
    mis-split of the two-hot mass across bins 32/33."""

    # The exact fp16 value the production cast (target -> logits.dtype) lands on.
    v_fp16 = torch.tensor(0.004395599999999944, dtype=torch.float16)
    v_exact = float(v_fp16)  # 0.00439453125, promoted to double
    # Single-row fp16 logits so a scalar target bins to the matching (VALUE_BINS,).
    fp16_logits = torch.randn(C.VALUE_BINS, dtype=torch.float16)

    # fp64 reference two-hot computed from the exact fp16 input value.
    pos = (v_exact + 1.0) * ((C.VALUE_BINS - 1) / 2.0)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    ref = torch.zeros(C.VALUE_BINS, dtype=torch.float64)
    up_w = pos - lo
    ref[lo] += 1.0 - up_w
    ref[hi] += up_w

    # NEW path (fixed): cast the fp16 scalar to fp32, then bin -> fp32 arithmetic.
    fp32_binned = scalar_to_binned_target(v_fp16.to(torch.float32))
    assert fp32_binned.dtype == torch.float32
    assert torch.allclose(fp32_binned.double(), ref, atol=1e-7), (
        f"fp32-binned two-hot {fp32_binned.tolist()} != fp64 ref {ref.tolist()}"
    )

    # OLD path (buggy): bin the fp16 scalar directly -> fp16 arithmetic; coarser.
    fp16_binned = scalar_to_binned_target(v_fp16)
    fp16_err = (fp16_binned.double() - ref).abs().max().item()
    fp32_err = (fp32_binned.double() - ref).abs().max().item()
    assert fp16_err > 1e-3, f"fp16 binning unexpectedly exact (err={fp16_err})"
    assert fp32_err < 1e-6 < fp16_err, (
        f"fix not load-bearing: fp32 err {fp32_err} vs fp16 err {fp16_err}"
    )

    # End-to-end: binned_value_loss with fp16 logits + a scalar target is finite and
    # builds an fp32 distribution internally (the CE lifts logits via _at_least_fp32).
    loss = binned_value_loss(fp16_logits, v_exact)
    assert torch.isfinite(loss).all()


def _ref_short_term_value_targets(decisions, index, player, horizons):
    """First-principles STV reference: from the current player's perspective
    (root values negated on opponent decisions), take every stepped full-turn
    future value (offsets 1, 3, 5, ...) and form a geometric EMA with decay
    (m-1)/(m+1) per horizon m. Independent of the production helper."""

    future = decisions[index + 1 :]
    perspective = [
        rv if fp == player else -rv for fp, _s, rv in future
    ]
    stepped = perspective[1::2]
    if not stepped:
        return ()
    out = []
    for m in horizons:
        decay = (m - 1.0) / (m + 1.0)
        w = 1.0
        num = 0.0
        den = 0.0
        for v in stepped:
            num += w * v
            den += w
            w *= decay
        out.append((int(m), num / den))
    return tuple(out)


def _ref_future_opponent_policy(decisions, index, player, *, mask_from_fast=False):
    """First-principles next-opponent-policy reference: scan forward for the first
    decision by the other side; prefer its gumbel_policy, else its visit policy;
    mask when mask_from_fast and that row is a fast (pcr_full=False) row."""

    for fp, fs, _rv in decisions[index + 1 :]:
        if fp != player:
            if mask_from_fast and not fs.metadata.get("pcr_full", True):
                return (), "fast_unrecorded_masked"
            if getattr(fs, "gumbel_policy", None):
                return tuple(fs.gumbel_policy), "future_opponent_gumbel"
            return tuple(fs.policy), "future_opponent_mcts"
    return (), "none"


def test_stv_and_opp_helpers_match_reference() -> None:
    """STV EMA targets and next-opponent-policy selection against independent
    first-principles references (replaces the dense_cnn_restnet oracle)."""

    rng = random.Random(7)
    # Synthetic decision sequence: player 0 first, then alternating in pairs.
    players_int = [0] + [1 if ((k - 1) // 2) % 2 == 0 else 0 for k in range(1, 23)]
    decisions = []
    for k, player in enumerate(players_int):
        policy = ((pack_action_id(k, -k), 0.7), (pack_action_id(k + 1, -k), 0.3))
        meta = {"pcr_full": rng.random() > 0.4}
        root_value = rng.uniform(-1.0, 1.0)
        decisions.append(
            (
                player,
                ShrimpSampleData(
                    game_id="", turn_index=k, current_player=player, phase="FirstStone",
                    records=(), first_stone=None, own_hot=(), opp_hot=(), own_win=(),
                    opp_win=(), policy=policy, metadata=meta,
                ),
                root_value,
            )
        )

    for index in range(len(players_int)):
        player = players_int[index]
        mine = _short_term_value_targets(decisions, index, player, STV_HORIZONS)
        theirs = _ref_short_term_value_targets(decisions, index, player, STV_HORIZONS)
        assert mine == theirs
        for mask_fast in (False, True):
            mine_opp = _future_opponent_policy(
                decisions, index, player, mask_from_fast=mask_fast
            )
            theirs_opp = _ref_future_opponent_policy(
                decisions, index, player, mask_from_fast=mask_fast
            )
            assert mine_opp == theirs_opp


def test_finalize_invariants() -> None:
    pending, winner = _make_game(11)
    assert len(pending) >= 6
    finalized = finalize_game_samples(pending, winner)
    n = len(finalized)
    for index, row in enumerate(finalized):
        z = 0.0 if winner is None else (1.0 if winner == row.current_player else -1.0)
        assert row.value == z
        assert row.moves_left == float(n - index - 1)
    truncated = finalize_game_samples(pending, None, truncated=True)
    assert all(row.moves_left == -1.0 for row in truncated)
    assert all(row.metadata["truncated"] for row in truncated)


def test_expand_policy_mapping_and_d6() -> None:
    pending, winner = _make_game(13)
    finalized = finalize_game_samples(pending, winner)
    sample = finalized[min(6, len(finalized) - 1)]
    base = expand_sample(sample, symmetry=0)
    assert np.isclose(base.policy.sum(), sum(w for _a, w in sample.policy))
    assert 0.0 <= base.opp_coverage <= 1.0

    for sym in range(12):
        rot = expand_sample(sample, symmetry=sym)
        assert rot.policy.shape == base.policy.shape
        # Each stored action's mass maps to the slot of its transformed cell.
        for action_id, weight in sample.policy:
            q, r = unpack_action_id(action_id)
            cell = apply_d6(sym, q, r)
            slot = rot.support.index[cell]
            assert slot < rot.support.legal_count
            assert rot.policy[slot] >= np.float32(weight) - 1e-7
        assert np.isclose(rot.policy.sum(), base.policy.sum())
        assert np.isclose(rot.opp_policy.sum(), base.opp_policy.sum())

    bad = ShrimpSampleData(
        game_id="", turn_index=0, current_player=sample.current_player, phase=sample.phase,
        records=sample.records, first_stone=sample.first_stone, own_hot=sample.own_hot,
        opp_hot=sample.opp_hot, own_win=sample.own_win, opp_win=sample.opp_win,
        policy=((pack_action_id(2000, 2000), 1.0),),  # cell outside the support
    )
    try:
        expand_sample(bad)
        raise AssertionError("off-legal policy target must raise")
    except ValueError:
        pass


def test_writer_roundtrip(tmp_path) -> None:
    pending, winner = _make_game(17)
    finalized = finalize_game_samples(pending, winner)
    path = tmp_path / "game.npz"
    rows = write_compact_shard(path, finalized)
    assert rows == len(finalized)
    sidecar = (tmp_path / "game.json").read_text(encoding="utf-8")
    assert '"lineage": "shrimp"' in sidecar
    assert '"shrimp_compact_v1"' in sidecar

    restored = read_compact_shard(path)
    assert len(restored) == len(finalized)
    for a, b in zip(finalized, restored):
        assert b.records == a.records
        assert b.current_player == a.current_player
        assert b.phase == a.phase
        assert b.first_stone == a.first_stone
        assert b.own_hot == a.own_hot and b.opp_hot == a.opp_hot
        assert b.own_win == a.own_win and b.opp_win == a.opp_win
        assert [aid for aid, _ in b.policy] == [aid for aid, _ in a.policy]
        assert np.allclose(
            [w for _aid, w in b.policy], np.asarray([w for _aid, w in a.policy], dtype=np.float32)
        )
        assert b.value == np.float32(a.value)
        assert b.moves_left == np.float32(a.moves_left)
        assert [h for h, _ in b.short_term_value] == [h for h, _ in a.short_term_value]


def _write_legacy_restnet_shard(
    path,
    *,
    facts,
    current_player,
    phase,
    turn_index,
    first_stone,
    policy,
    opp_policy,
    value,
    short_term_value,
    moves_left,
    horizons=STV_HORIZONS,
    schema_version=LEGACY_RESTNET_SCHEMA_VERSION,
) -> None:
    """Write a one-row restnet compact-v1 ``.npz`` in exactly the raw-array
    layout ``read_legacy_restnet_shard`` consumes.

    This is the public substitute for the retired ``dense_cnn_restnet.compact_io``
    writer: it pins the on-disk legacy schema the adapter promises to read, using
    only the reader's contract (records/stones counts must match; hot lists stored
    as packed engine coords; per-row CSR offsets for history/policy/opp; a
    per-row stvalue table gated by stvalue_mask). ``schema_version=None`` drops
    the column to exercise the "no version" accept path.
    """

    records = facts.records
    n_hist = len(records)
    horizons = [int(h) for h in horizons]
    h = len(horizons)
    stv = dict(short_term_value)

    arrays = {
        "num_rows": np.asarray(1, dtype=np.int64),
        "horizons": np.asarray(horizons, dtype=np.int64),
        "turn_index": np.asarray([turn_index], dtype=np.int32),
        "current_player": np.asarray([current_player], dtype=np.int64),
        "phase": np.asarray([phase], dtype=object),
        "value": np.asarray([value], dtype=np.float32),
        "moves_left": np.asarray([moves_left], dtype=np.float32),
        # History CSR: (q, r) interleaved, plus owner / placement-index columns.
        "hist_off": np.asarray([0, n_hist], dtype=np.int64),
        "hist_qr": np.asarray(
            [c for rec in records for c in (int(rec[0]), int(rec[1]))], dtype=np.int64
        ),
        "hist_owner": np.asarray([int(rec[2]) for rec in records], dtype=np.int64),
        "hist_idx": np.asarray([int(rec[3]) for rec in records], dtype=np.int64),
        # Stones count must equal history length (unified-records assumption); the
        # reader ignores stone contents, so a matching-length placeholder suffices.
        "stones_off": np.asarray([0, n_hist], dtype=np.int64),
        # Visit-policy CSR.
        "pol_off": np.asarray([0, len(policy)], dtype=np.int64),
        "pol_act": np.asarray([int(a) for a, _ in policy], dtype=np.int64),
        "pol_w": np.asarray([float(w) for _, w in policy], dtype=np.float32),
        # Opponent-policy CSR.
        "opp_off": np.asarray([0, len(opp_policy)], dtype=np.int64),
        "opp_act": np.asarray([int(a) for a, _ in opp_policy], dtype=np.int64),
        "opp_w": np.asarray([float(w) for _, w in opp_policy], dtype=np.float32),
        # Short-term value table, one column per horizon, masked by presence.
        "stvalue": np.asarray(
            [[float(stv.get(hz, 0.0)) for hz in horizons]], dtype=np.float32
        ),
        "stvalue_mask": np.asarray(
            [[1.0 if hz in stv else 0.0 for hz in horizons]], dtype=np.float32
        ),
        "first_q": np.asarray([first_stone[0] if first_stone else 0], dtype=np.int16),
        "first_r": np.asarray([first_stone[1] if first_stone else 0], dtype=np.int16),
        "first_present": np.asarray([1 if first_stone else 0], dtype=np.uint8),
        # Hot lists as packed engine coords with per-row CSR offsets.
        "own_hot_qr": _pack_qr(facts.own_hot),
        "own_hot_off": np.asarray([0, len(facts.own_hot)], dtype=np.int64),
        "opp_hot_qr": _pack_qr(facts.opp_hot),
        "opp_hot_off": np.asarray([0, len(facts.opp_hot)], dtype=np.int64),
    }
    if schema_version is not None:
        arrays["schema_version"] = np.asarray(schema_version, dtype=np.int32)
    np.savez(path, **arrays)


def test_legacy_crossread(tmp_path) -> None:
    rng = random.Random(19)
    state = random_playout(101, 15)
    if api.terminal(state) is not None:
        state = random_playout(103, 11)
    assert api.terminal(state) is None
    mirror = api.to_python_state(state)
    facts = facts_from_engine(mirror)
    legal = sorted(api.legal_action_ids(state))
    policy = ((legal[0], 0.6), (legal[-1], 0.4))

    path = tmp_path / "legacy.npz"
    _write_legacy_restnet_shard(
        path,
        facts=facts,
        current_player=facts.current_player,
        phase=str(mirror.phase.value),
        turn_index=5,
        first_stone=(
            (mirror.first_stone.q, mirror.first_stone.r) if facts.first_stone else None
        ),
        policy=policy,
        opp_policy=((legal[1], 1.0),),
        value=0.5,
        short_term_value=((2, 0.25), (16, -0.5)),
        moves_left=42.0,
    )

    rows = read_legacy_restnet_shard(path)
    assert len(rows) == 1
    row = rows[0]
    assert row.metadata["source"] == "legacy_shard"
    assert row.records == facts.records
    assert row.current_player == facts.current_player
    assert row.phase == facts.phase
    assert row.own_hot == facts.own_hot and row.opp_hot == facts.opp_hot
    # Win-now cells are derived from stones and match the engine facts.
    assert row.own_win == facts.own_win and row.opp_win == facts.opp_win
    assert row.moves_left == 42.0
    assert dict(row.short_term_value) == {2: np.float32(0.25), 16: np.float32(-0.5)}

    # Derived legality equals the engine's full legal set, not the stored column.
    expanded = expand_sample(row)
    sup_ids = [pack_action_id(q, r) for q, r in expanded.support.legal_coords().tolist()]
    assert sup_ids == legal


def _write_minimal_legacy_shard(path, *, schema_version) -> None:
    """Write a one-row restnet compact-v1 shard with the ``schema_version`` column
    set to ``schema_version`` (or dropped when ``None``), via the in-test legacy
    writer that mirrors the reader's on-disk contract."""

    state = random_playout(101, 15)
    if api.terminal(state) is not None:
        state = random_playout(103, 11)
    assert api.terminal(state) is None
    mirror = api.to_python_state(state)
    facts = facts_from_engine(mirror)
    legal = sorted(api.legal_action_ids(state))
    _write_legacy_restnet_shard(
        path,
        facts=facts,
        current_player=facts.current_player,
        phase=str(mirror.phase.value),
        turn_index=5,
        first_stone=(
            (mirror.first_stone.q, mirror.first_stone.r) if facts.first_stone else None
        ),
        policy=((legal[0], 0.6), (legal[-1], 0.4)),
        opp_policy=((legal[1], 1.0),),
        value=0.5,
        short_term_value=((2, 0.25), (16, -0.5)),
        moves_left=42.0,
        schema_version=schema_version,
    )


def test_legacy_shard_schema_version_guard(tmp_path) -> None:
    # A wrong stored legacy schema_version raises.
    bad = tmp_path / "bad.npz"
    _write_minimal_legacy_shard(bad, schema_version=99)
    try:
        read_legacy_restnet_shard(bad)
        raise AssertionError("wrong legacy schema_version must raise")
    except ValueError as exc:
        assert "schema" in str(exc) and "99" in str(exc)

    # The expected version (1) reads cleanly.
    ok = tmp_path / "ok.npz"
    _write_minimal_legacy_shard(ok, schema_version=1)
    rows = read_legacy_restnet_shard(ok)
    assert len(rows) == 1 and rows[0].metadata["source"] == "legacy_shard"

    # A missing schema_version column reads without error.
    absent = tmp_path / "absent.npz"
    _write_minimal_legacy_shard(absent, schema_version=None)
    rows = read_legacy_restnet_shard(absent)
    assert len(rows) == 1


def test_microbucket_loss_equals_monolithic_fp64() -> None:
    pending, winner = _make_game(23, max_plies=14)
    finalized = finalize_game_samples(pending, winner)
    rows = [expand_sample(s, symmetry=i % 12) for i, s in enumerate(finalized)]
    assert len(rows) >= 6
    horizons = STV_HORIZONS
    denoms = step_global_denominators(rows, horizons)

    def loss_for(model, row_subset, pad_to=None):
        batch = collate_training(row_subset, pad_to=pad_to)
        batch = split_stvalue_columns(batch, horizons)
        batch = {k: (v.double() if v.dtype == torch.float32 else v) for k, v in batch.items()}
        out = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
        total, _ = shrimp_loss(out, batch, denominators=denoms)
        return total

    torch.manual_seed(31)
    model = ShrimpNet().double()
    loss_mono = loss_for(model, rows)
    loss_mono.backward()
    mono = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    mono_total = loss_mono.detach().item()

    model2 = ShrimpNet().double()
    model2.load_state_dict({k: v for k, v in model.state_dict().items()})
    buckets = pair_budget_microbuckets(rows, budget=2.0e5)  # small budget forces multiple buckets
    assert len(buckets) >= 2
    assert sorted(id(r) for b in buckets for r in b) == sorted(id(r) for r in rows)
    total = 0.0
    for bucket in buckets:
        loss = loss_for(model2, bucket)
        loss.backward()
        total += loss.detach().item()
    assert abs(total - mono_total) <= 1e-10 * (1.0 + abs(mono_total))
    for name, p in model2.named_parameters():
        scale = 1.0 + mono[name].abs().max().item()
        assert (p.grad - mono[name]).abs().max().item() <= 1e-10 * scale, name


def test_pair_budget_bucket_rule() -> None:
    pending, winner = _make_game(29, max_plies=20)
    rows = [expand_sample(s) for s in finalize_game_samples(pending, winner)]
    budget = 5.0e5
    buckets = pair_budget_microbuckets(rows, budget=budget)
    for bucket in buckets:
        s_pad = max(r.support.num_nodes for r in bucket) + C.NUM_TOKENS
        if len(bucket) > 1:
            assert len(bucket) * s_pad**2 <= budget


def _fast_game_with_engine_facts(seed: int, max_plies: int = 24):
    """Play one random game recording, per decision, (a) an EMPTY-fact fast row
    exactly as selfplay's fast branch stores it and (b) the serve-time
    PositionFacts for the same decision state. Returns (pending, ref_facts,
    winner) where pending is the (player, empty-fast-sample, root_value) list."""

    rng = random.Random(seed)
    state = api.new_game()
    pending = []
    ref_facts = []
    winner = None
    for ply in range(max_plies):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        facts = facts_from_engine(api.to_python_state(state))
        ref_facts.append(facts)
        # Mirror selfplay's fast branch: empty first_stone / hot / win facts,
        # pcr_full False, records/current_player/phase captured pre-decision.
        sample = ShrimpSampleData(
            game_id="test",
            turn_index=ply,
            current_player=facts.current_player,
            phase=facts.phase,
            records=facts.records,
            first_stone=None,
            own_hot=(),
            opp_hot=(),
            own_win=(),
            opp_win=(),
            policy=(),
            metadata={"pcr_full": False},
        )
        pending.append((facts.current_player, sample, rng.uniform(-0.8, 0.8)))
        q, r = unpack_action_id(rng.choice(ids))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            winner = player_int(api.terminal(state).winner)
            break
    return pending, ref_facts, winner


def test_fast_row_facts_recomputed_at_write() -> None:
    """A written fast (value-only) row's first-stone / hot / win facts must equal
    a reference window_scan recompute from its own records -- and the serve-time
    engine facts -- closing the train/serve feature skew. Fast rows store empty
    facts on the search hot path; _populate_fast_facts fills them at write time."""

    from shrimp.features import window_scan
    from shrimp.selfplay import _populate_fast_facts

    pending, ref_facts, winner = _fast_game_with_engine_facts(41)
    finalized = finalize_game_samples(pending, winner, mask_opp_from_fast=True)
    # Every row is a fast row here; before populate their facts are empty.
    assert all(not s.metadata.get("pcr_full", False) for s in finalized)
    assert all(s.own_hot == () and s.first_stone is None for s in finalized)

    populated = [_populate_fast_facts(s) for s in finalized]
    saw_second_stone_first = False
    for row, facts in zip(populated, ref_facts):
        # (1) Matches a direct window_scan recompute from the row's records.
        wh_own, wh_opp, ww_own, ww_opp = window_scan(
            row.records, row.current_player, len(row.records)
        )
        assert row.own_hot == wh_own
        assert row.opp_hot == wh_opp
        assert row.own_win == ww_own
        assert row.opp_win == ww_opp
        # (2) Matches the serve-time engine facts (train/serve parity).
        assert row.own_hot == facts.own_hot
        assert row.opp_hot == facts.opp_hot
        assert row.own_win == facts.own_win
        assert row.opp_win == facts.opp_win
        assert row.first_stone == facts.first_stone
        # (3) first_stone is set for SecondStone-phase fast rows.
        if row.phase == "SecondStone":
            assert row.first_stone == (row.records[-1][0], row.records[-1][1])
            assert row.first_stone is not None
            saw_second_stone_first = True
        else:
            assert row.first_stone is None
    assert saw_second_stone_first, "game had no SecondStone decision to exercise first_stone"


def _hot_fast_sample(current_player: int) -> ShrimpSampleData:
    """An EMPTY-fact fast row (as selfplay stores it) over a hand-built position
    with four collinear player-0 stones on the Q axis -- a length-6 window of
    count 4 -> a hot window (HOT_MIN_COUNT=4) with >= HOT_MIN_PLACEMENTS=7
    placements. window_scan yields non-empty hot cells whose own/opp assignment
    depends on ``current_player``, so this exercises the non-empty recompute the
    random games never hit."""

    records = (
        (0, 0, 0, 1), (5, 5, 1, 2), (1, 0, 0, 3), (6, 5, 1, 4),
        (2, 0, 0, 5), (7, 5, 1, 6), (3, 0, 0, 7),
    )
    return ShrimpSampleData(
        game_id="test", turn_index=len(records), current_player=current_player,
        phase="FirstStone", records=records,
        first_stone=None, own_hot=(), opp_hot=(), own_win=(), opp_win=(),
        policy=(), metadata={"pcr_full": False},
    )


def test_fast_row_facts_nonempty_and_side_relative() -> None:
    """On a constructed hot position the recompute yields non-empty hot cells,
    and the own/opp split flips with current_player -- proving the fix is
    load-bearing (planes were force-zeroed before) and side-relative."""

    from shrimp.features import window_scan
    from shrimp.selfplay import _populate_fast_facts

    p0 = _populate_fast_facts(_hot_fast_sample(current_player=0))
    p1 = _populate_fast_facts(_hot_fast_sample(current_player=1))
    ref0 = window_scan(p0.records, 0, len(p0.records))
    assert (p0.own_hot, p0.opp_hot, p0.own_win, p0.opp_win) == ref0
    assert p0.own_hot and not p0.opp_hot          # player-0's window -> own for cp=0
    assert p1.opp_hot and not p1.own_hot          # same window -> opp for cp=1
    assert p0.own_hot == p1.opp_hot               # cells identical, side flipped


def test_fast_row_facts_survive_shard_roundtrip(tmp_path) -> None:
    """The recomputed fast-row facts persist through the compact-shard writer
    (the same path _writer_loop takes), so the stored planes are non-empty and
    match the reference -- not force-zeroed as before the fix."""

    from shrimp.selfplay import _populate_fast_facts

    populated = [_populate_fast_facts(_hot_fast_sample(current_player=1))]
    assert any(s.own_hot or s.opp_hot or s.own_win or s.opp_win for s in populated)

    path = tmp_path / "game.npz"
    write_compact_shard(path, populated)
    restored = read_compact_shard(path)
    for a, b in zip(populated, restored):
        assert b.own_hot == a.own_hot and b.opp_hot == a.opp_hot
        assert b.own_win == a.own_win and b.opp_win == a.opp_win
        assert b.first_stone == a.first_stone
        assert b.metadata["pcr_full"] is False  # still a value-only row


def test_resume_counts_only_committed_shards(tmp_path) -> None:
    """Resume accounting counts a game done only when its npz AND .json sidecar
    both exist. A sidecar-less (power-cut) npz is excluded from the done count
    but still feeds the next-key max so keys are never reused."""

    epoch_dir = tmp_path / "epoch_000000"
    epoch_dir.mkdir()
    # Two committed games (npz + sidecar) and one uncommitted (npz only).
    for key in (5, 6):
        (epoch_dir / f"game_{key}.npz").write_bytes(b"x")
        (epoch_dir / f"game_{key}.json").write_text("{}", encoding="utf-8")
    (epoch_dir / "game_9.npz").write_bytes(b"x")  # sidecar-less: not committed

    existing = sorted(epoch_dir.glob("game_*.npz"))
    already_done = sum(1 for p in existing if p.with_suffix(".json").exists())
    assert already_done == 2  # the sidecar-less npz does not count

    # Next key considers ALL npz (including the sidecar-less one) so 9 is never
    # reused: next_key == max(5, 6, 9) + 1.
    existing_keys = [int(p.stem.split("_", 1)[1]) for p in existing]
    assert max(existing_keys) + 1 == 10


def test_decode_moves_left_median() -> None:
    # decode_moves_left takes the softmax expectation over the 65-bin scalar
    # support [-1, 1] and maps it onto decisions [0, MOVES_LEFT_CAP]:
    #   decisions = (scalar + 1) / 2 * MOVES_LEFT_CAP.
    # Near-one-hot logits collapse the expectation onto the peak bin's scalar.
    cap = float(C.MOVES_LEFT_CAP)
    logits = torch.full((3, C.VALUE_BINS), -40.0)
    logits[0, 0] = 40.0   # bin 0  -> scalar -1 -> 0 decisions
    logits[1, 32] = 40.0  # bin 32 -> scalar  0 -> cap/2 decisions
    logits[2, 64] = 40.0  # bin 64 -> scalar +1 -> cap decisions
    decoded = decode_moves_left(logits)
    assert torch.allclose(decoded, torch.tensor([0.0, 0.5 * cap, cap]))


def test_rust_expand_rejects_horizon_mismatch() -> None:
    """The rust kernel copies the stvalue block POSITIONALLY, so a window whose
    stored horizons differ from the requested set (even at the same length) must be
    rejected — otherwise the STV heads train on the wrong horizon's target. The
    serial python path remaps by horizon VALUE and is unaffected."""
    import dataclasses

    pytest.importorskip(
        "shrimp._rust", reason="rust backend .so required for _expand_rows_rust"
    )
    from shrimp.expand_backends import _expand_rows_rust
    from shrimp.window import PackedWindow

    # A same-length but re-tuned horizon set: (2, 6, 16) stored vs (2, 6, 99) asked.
    win = dataclasses.replace(PackedWindow.empty(), horizons=(2, 6, 16))
    with pytest.raises(ValueError, match="horizons"):
        _expand_rows_rust(win, [], np.empty(0, dtype=np.int64), (2, 6, 99), False)

    # Reordered same values must also be rejected (positional copy is order-sensitive).
    with pytest.raises(ValueError, match="horizons"):
        _expand_rows_rust(win, [], np.empty(0, dtype=np.int64), (6, 2, 16), False)
