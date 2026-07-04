"""End-to-end test for truncated-game training with per-head masking.

Builds a completed game and a truncated game, runs them through
finalize_game_samples -> write_compact_shard -> read/expand ->
collate_training -> hexfield_loss, and asserts:

  (a) truncated rows are written and present (not dropped),
  (b) value / stvalue / cell_q / moves_left loss contributions are 0 for
      truncated rows (their per-row masks are 0),
  (c) policy loss is nonzero for truncated rows,
  (d) completed-game targets and per-head losses are identical whether
      finalized/expanded alone or mixed with truncated rows,
  (e) no exception is raised.

Does not import dense_cnn_restnet, so it runs without the oracle backend.
"""

from __future__ import annotations

import random

import numpy as np
import torch

import hexfield_testkit  # noqa: F401  (sys.path shim: exposes hexfield + engine)
from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield.batching import (
    collate_training,
    split_stvalue_columns,
    step_global_denominators,
)
from hexfield.engine_facts import facts_from_engine, player_int
from hexfield.geometry import unpack_action_id
from hexfield.losses import hexfield_loss
from hexfield.model import HexfieldNet
from hexfield.samples import (
    STV_HORIZONS,
    HexfieldSampleData,
    expand_sample,
    finalize_game_samples,
)
from hexfield.expand_backends import expand_rows
from hexfield.shards import read_compact_shard, write_compact_shard
from hexfield.window import load_packed_shard


def _sample_from_state(state, rng: random.Random, turn_index: int) -> HexfieldSampleData:
    """One full-search decision row with a small random visit policy + child Q."""
    facts = facts_from_engine(api.to_python_state(state))
    legal = sorted(api.legal_action_ids(state))
    chosen = rng.sample(legal, k=min(3, len(legal)))
    weights = [rng.random() + 0.1 for _ in chosen]
    total = sum(weights)
    policy = tuple((aid, w / total) for aid, w in zip(chosen, weights))
    # Child Q parallel to policy (cell_q head target), values in [-0.9, 0.9].
    q_policy = tuple((aid, rng.uniform(-0.9, 0.9)) for aid in chosen)
    return HexfieldSampleData(
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
        q_policy=q_policy,
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


def _full_rows(finalized):
    return [s for s in finalized if s.metadata.get("pcr_full", False)]


def _expand_all(samples):
    # Symmetry 0: identity expansion, row-for-row comparable.
    return [expand_sample(s, symmetry=0) for s in samples]


def test_truncated_rows_masked_completed_unchanged(tmp_path) -> None:
    # A completed game is finalized with a concrete winner (0/1) and
    # truncated=False. A truncated game is the same shape finalized with
    # winner=None and truncated=True. The winner label feeds only the hard-z
    # value target, which is masked off for truncated rows.
    comp_pending, _ = _make_game(101, max_plies=24)
    trunc_pending, _ = _make_game(202, max_plies=18)
    comp_winner = 0  # concrete winner => completed game (truncated=False)

    comp_final = finalize_game_samples(comp_pending, comp_winner, mask_opp_from_fast=True)
    trunc_final = finalize_game_samples(
        trunc_pending, None, truncated=True, mask_opp_from_fast=True
    )

    # (a) truncated rows carry the truncated flag + moves_left sentinel.
    assert all(s.metadata["truncated"] for s in trunc_final)
    assert all(s.moves_left == -1.0 for s in trunc_final)
    assert all(not s.metadata.get("truncated", False) for s in comp_final)

    comp_rows_src = _full_rows(comp_final)
    trunc_rows_src = _full_rows(trunc_final)
    assert comp_rows_src and trunc_rows_src

    # --- shard round-trip: truncated rows are written and the flag survives ---
    comp_path = tmp_path / "completed.npz"
    trunc_path = tmp_path / "truncated.npz"
    n_comp = write_compact_shard(comp_path, comp_rows_src)
    n_trunc = write_compact_shard(trunc_path, trunc_rows_src)
    assert n_comp == len(comp_rows_src)
    assert n_trunc == len(trunc_rows_src)  # (a) truncated rows present on disk

    # read_compact_shard reads the columnar shard: the truncated flag round-trips
    # through the outcome_valid column (a). This reader does not reconstruct
    # q_policy, so cell_q is checked via the packed train path below.
    comp_read = read_compact_shard(comp_path)
    trunc_read = read_compact_shard(trunc_path)
    assert all(r.metadata.get("truncated", False) for r in trunc_read)
    assert all(not r.metadata.get("truncated", False) for r in comp_read)

    # --- expand via the train path (PackedWindow + serial expand_rows) ----------
    # The serial path restores q_policy from the q_pol_q column and threads
    # outcome_valid -> metadata['truncated'] -> masks.
    def expand_via_window(path, n):
        win = load_packed_shard(path)
        rows, valid = expand_rows(win, None, np.zeros(win.n, dtype=np.int32), backend="serial")
        assert bool(valid.all()) and len(rows) == n
        return rows

    comp_exp = expand_via_window(comp_path, len(comp_rows_src))
    trunc_exp = expand_via_window(trunc_path, len(trunc_rows_src))

    for row in comp_exp:
        assert row.value_mask == 1.0
        assert float(row.moves_left_mask) == 1.0
    # completed games have nonzero cell_q mask on at least one row.
    assert any(float(row.cell_q_mask.sum()) > 0.0 for row in comp_exp)
    for row in trunc_exp:
        assert row.value_mask == 0.0
        assert float(row.moves_left_mask) == 0.0
        assert float(row.stvalue_mask.sum()) == 0.0
        assert float(row.cell_q_mask.sum()) == 0.0
        # policy / opp_policy are NOT masked: positive policy mass is preserved.
        assert float(row.policy.sum()) > 0.0

    # --- (d) completed-game expansion: in-memory vs packed path -----------------
    # In-memory finalized rows expand to identical targets and masks as the
    # packed-window rows.
    comp_exp_direct = _expand_all(comp_rows_src)
    for a, b in zip(comp_exp_direct, comp_exp):
        assert a.value == b.value and a.value_mask == b.value_mask == 1.0
        assert np.array_equal(a.stvalue, b.stvalue)
        assert np.array_equal(a.stvalue_mask, b.stvalue_mask)
        assert np.array_equal(a.cell_q, b.cell_q)
        assert np.array_equal(a.cell_q_mask, b.cell_q_mask)
        assert a.moves_left == b.moves_left
        assert a.moves_left_mask == b.moves_left_mask
        assert np.array_equal(a.policy, b.policy)

    # --- loss: completed-only vs mixed batch -----------------------------------
    horizons = STV_HORIZONS
    torch.manual_seed(7)
    model = HexfieldNet().double().eval()

    def run_loss(rows):
        denoms = step_global_denominators(rows, horizons)
        batch = collate_training(rows)
        batch = split_stvalue_columns(batch, horizons)
        batch = {k: (v.double() if v.dtype == torch.float32 else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
            total, comps = hexfield_loss(out, batch, denominators=denoms)
        return total, comps, denoms

    # (b)/(c): a truncated-only batch. The value/stvalue/cell_q/moves_left masks
    # are all zero, so those components are 0; policy is nonzero.
    t_total, t_comps, t_denoms = run_loss(trunc_exp)
    assert t_denoms["value"] == 0.0
    assert t_denoms["moves_left"] == 0.0
    assert t_denoms["cell_q"] == 0.0
    for h in horizons:
        assert t_denoms[f"stvalue_{h}"] == 0.0
    assert float(t_comps["value"]) == 0.0
    assert float(t_comps["moves_left"]) == 0.0
    assert float(t_comps["cell_q"]) == 0.0
    for h in horizons:
        assert float(t_comps[f"stvalue_{h}"]) == 0.0
    # (c) policy trains on truncated rows -> strictly positive CE.
    assert float(t_comps["policy"]) > 0.0
    # opp_policy may be zero (no future opponent on a tail row) but must be finite.
    assert np.isfinite(float(t_comps["opp_policy"]))

    # (d) the completed component losses are identical whether the batch is
    # completed-only or mixed with truncated rows. Truncated rows contribute zero
    # to every masked head's numerator and denominator, and the value/stvalue/
    # cell_q/moves_left step-global denominators count only completed rows.
    c_total, c_comps, c_denoms = run_loss(comp_exp)
    mixed_total, mixed_comps, mixed_denoms = run_loss(comp_exp + trunc_exp)

    # Masked-head denominators are identical (truncated rows excluded).
    assert mixed_denoms["value"] == c_denoms["value"] == float(len(comp_exp))
    assert mixed_denoms["moves_left"] == c_denoms["moves_left"]
    assert mixed_denoms["cell_q"] == c_denoms["cell_q"]
    for h in horizons:
        assert mixed_denoms[f"stvalue_{h}"] == c_denoms[f"stvalue_{h}"]

    # Masked heads: identical between completed-only and mixed.
    for key in ["value", "moves_left", "cell_q"] + [f"stvalue_{h}" for h in horizons]:
        assert float(mixed_comps[key]) == float(c_comps[key]), key


def test_finalize_truncated_does_not_mutate_completed() -> None:
    # Finalizing a completed game (winner set, truncated defaulted) produces
    # value = +/-1, moves_left counting down to 0, and no truncated flag.
    pending, _ = _make_game(303, max_plies=40)
    winner = 1  # concrete winner => completed game
    finalized = finalize_game_samples(pending, winner, mask_opp_from_fast=True)
    n = len(finalized)
    for index, row in enumerate(finalized):
        z = 1.0 if winner == row.current_player else -1.0
        assert row.value == z
        assert row.moves_left == float(n - index - 1)
        assert not row.metadata.get("truncated", False)
