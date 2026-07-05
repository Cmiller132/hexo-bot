"""CPU unit tests for the soft policy target, covering three components:

1. The soft-target transform produced by ``batching.collate_training``:
   target_soft = p^(1/2) on the visited support (policy > 0) within each row's
   legal prefix; unvisited-legal and off-prefix slots stay exactly 0. Renorm is
   deferred to ``segment_policy_ce``. See batching.py.
2. ``segment_policy_ce`` CE on the soft target (renorm inside, prefix-only).
3. ``shrimp_loss`` wiring: the ``soft_policy`` component is present iff both
   ``outputs['soft_policy']`` and ``batch['soft_policy']`` exist, weighted by
   SOFT_POLICY_WEIGHT.

Rows are built from synthetic supports and numpy targets, so the file runs
under plain CPU torch without the native MCTS module or a GPU.
"""

from __future__ import annotations

import numpy as np
import torch

from shrimp import constants as C
from shrimp.batching import collate_rows, collate_training
from shrimp.checkpoints import warm_start_into
from shrimp.losses import SOFT_POLICY_WEIGHT, shrimp_loss, segment_policy_ce
from shrimp.model import ShrimpNet
from shrimp.samples import STV_HORIZONS, ExpandedRow
from shrimp.support import build_support


def _row(policy: np.ndarray, gumbel_policy: np.ndarray | None = None) -> ExpandedRow:
    """An ExpandedRow over a one-stone support whose legal prefix carries ``policy``.

    Only the fields collate_training reads for the soft target (support, policy,
    and the optional gumbel target) are meaningful; the rest are valid
    placeholders. A single stone at the origin yields a legal prefix covering all
    empty cells within radius, so ``policy`` occupies the leading slots and the
    remaining legal slots stay at zero visits. ``legal_count`` equals
    support.legal_count. ``policy`` must carry positive total mass, provided by
    its leading nonzero entries. ``gumbel_policy`` (when given) is placed on the
    leading slots likewise and flags the row gumbel_policy_valid.
    """

    sup = build_support([(0, 0)])
    legal_count = sup.legal_count
    assert policy.shape[0] <= legal_count
    pol = np.zeros(legal_count, dtype=np.float32)
    pol[: policy.shape[0]] = policy
    gp = np.zeros(0, dtype=np.float32)
    gp_valid = 0.0
    if gumbel_policy is not None:
        assert gumbel_policy.shape[0] <= legal_count
        gp = np.zeros(legal_count, dtype=np.float32)
        gp[: gumbel_policy.shape[0]] = gumbel_policy
        gp_valid = 1.0
    h = len(STV_HORIZONS)
    return ExpandedRow(
        support=sup,
        feats=np.zeros((sup.num_nodes, 1), dtype=np.float32),
        policy=pol,
        opp_policy=np.zeros(legal_count, dtype=np.float32),
        opp_coverage=1.0,
        value=0.0,
        value_mask=1.0,
        policy_valid=1.0,
        stvalue=np.zeros(h, dtype=np.float32),
        stvalue_mask=np.zeros(h, dtype=np.float32),
        moves_left=0.0,
        moves_left_mask=0.0,
        cell_q=np.zeros(legal_count, dtype=np.float32),
        cell_q_mask=np.zeros(legal_count, dtype=np.float32),
        policy_surprise=0.0,
        gumbel_policy=gp,
        gumbel_policy_valid=gp_valid,
    )


def _katago_soft_reference(policy_row: np.ndarray, legal_count: int) -> np.ndarray:
    """Reference soft target on one row (pre-renorm, as collate emits it).

    target_soft = p^0.5 on the visited support (policy > 0) within the legal
    prefix; unvisited-legal cells and off-prefix slots stay exactly 0.
    p = visit_policy / sum(visit_policy). Renorm is not applied here; it happens
    inside segment_policy_ce.
    """

    out = np.zeros_like(policy_row)
    prefix = policy_row[:legal_count]
    p = prefix / max(prefix.sum(), 1e-12)
    support = p > 0
    out[:legal_count][support] = np.power(p[support], 0.5)
    return out


def test_collate_soft_target_matches_katago_transform() -> None:
    rng = np.random.default_rng(7)
    rows = [
        _row(np.array([0.5, 0.3, 0.2], dtype=np.float32)),
        _row(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),  # near-deterministic
        _row(rng.random(10).astype(np.float32) + 1e-3),
    ]
    batch = collate_training(rows)
    assert "soft_policy" in batch
    soft = batch["soft_policy"]
    policy = batch["policy"]
    assert soft.shape == policy.shape
    legal_counts = batch["legal_counts"]

    for g in range(len(rows)):
        lc = int(legal_counts[g].item())
        expect = _katago_soft_reference(policy[g].numpy(), lc)
        assert np.allclose(soft[g].numpy(), expect, atol=1e-6), g
        # Off the legal prefix the soft target is exactly zero.
        assert float(soft[g, lc:].abs().sum().item()) == 0.0, g
        # Positive exactly on the visited support (policy > 0); unvisited-legal
        # cells stay 0.
        support = policy[g, :lc] > 0
        assert bool((soft[g, :lc][support] > 0).all().item()), g
        assert float(soft[g, :lc][~support].abs().sum().item()) == 0.0, g


def test_soft_target_softens_relative_to_visit_policy() -> None:
    # ^0.5 softening flattens the distribution: the renormalized soft target
    # has lower max prob and higher entropy than the original visit policy.
    rows = [_row(np.array([0.7, 0.2, 0.1], dtype=np.float32))]
    batch = collate_training(rows)
    lc = int(batch["legal_counts"][0].item())
    p = batch["policy"][0, :lc]
    p = p / p.sum()
    soft = batch["soft_policy"][0, :lc]
    soft = soft / soft.sum()  # renorm as applied inside segment_policy_ce
    assert float(soft.max().item()) < float(p.max().item())
    ent = lambda d: float(-(d * d.clamp_min(1e-20).log()).sum().item())
    assert ent(soft) > ent(p)


def test_soft_target_is_pure_function_of_visit_policy() -> None:
    # Identical visit policies produce identical soft targets; collate is the
    # single derivation site.
    pol = np.array([0.4, 0.35, 0.25], dtype=np.float32)
    a = collate_training([_row(pol)])["soft_policy"]
    b = collate_training([_row(pol.copy())])["soft_policy"]
    assert torch.equal(a, b)


def test_soft_target_derives_from_gumbel_policy_when_valid() -> None:
    # A row carrying a gumbel improved-policy target π' derives its soft target
    # from π' (T=2 softened over π''s support), NOT from the visit policy: under
    # Sequential Halving the visit histogram is a schedule artifact. A row
    # without a gumbel target falls back to the visit softening, so mixed
    # batches select per row.
    visits = np.array([0.5, 0.3, 0.2, 0.0], dtype=np.float32)
    gumbel = np.array([0.1, 0.1, 0.1, 0.7], dtype=np.float32)  # disjoint shape
    batch = collate_training([_row(visits, gumbel), _row(visits)])
    lc = int(batch["legal_counts"][0].item())

    # Row 0: soft == π'^0.5 on π''s support (slot 3 positive, visit-only shape
    # ignored).
    expect0 = _katago_soft_reference(batch["gumbel_policy"][0].numpy(), lc)
    assert np.allclose(batch["soft_policy"][0].numpy(), expect0, atol=1e-6)
    assert float(batch["soft_policy"][0, 3].item()) > 0.0  # π' support wins

    # Row 1 (no gumbel target): unchanged visit-based softening.
    expect1 = _katago_soft_reference(batch["policy"][1].numpy(), lc)
    assert np.allclose(batch["soft_policy"][1].numpy(), expect1, atol=1e-6)
    assert float(batch["soft_policy"][1, 3].item()) == 0.0  # unvisited stays 0


def test_segment_ce_on_soft_target_renorms_and_is_finite() -> None:
    rows = [_row(np.array([0.6, 0.3, 0.1], dtype=np.float32))]
    batch = collate_training(rows)
    logits = torch.zeros_like(batch["soft_policy"], requires_grad=True)
    loss = segment_policy_ce(
        logits, batch["legal_counts"], batch["soft_policy"], denominator=1.0
    )
    assert torch.isfinite(loss)
    # Uniform logits -> CE == -sum(p_norm * log(1/L)) == log(L) over the legal
    # prefix; the soft target is renormalized inside segment_policy_ce.
    lc = int(batch["legal_counts"][0].item())
    assert abs(float(loss.item()) - np.log(lc)) < 1e-5
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_shrimp_loss_includes_weighted_soft_component() -> None:
    rows = [
        _row(np.array([0.5, 0.3, 0.2], dtype=np.float32)),
        _row(np.array([0.8, 0.1, 0.1], dtype=np.float32)),
    ]
    batch = collate_training(rows)
    b, npad = batch["policy"].shape
    torch.manual_seed(0)
    # Minimal outputs: only policy + value + soft_policy heads present.
    outputs = {
        "policy": torch.randn(b, npad),
        "value": torch.randn(b, 65),
        "soft_policy": torch.randn(b, npad),
    }
    batch_value = dict(batch)
    batch_value["value"] = torch.zeros(b, dtype=torch.float32)
    total, comps = shrimp_loss(outputs, batch_value)
    assert "soft_policy" in comps
    # The soft component enters total at weight SOFT_POLICY_WEIGHT; setting the
    # weight to 0 drops total by soft_policy_weight * component.
    total0, _ = shrimp_loss(outputs, batch_value, soft_policy_weight=0.0)
    delta = float((total - total0).item())
    expect = SOFT_POLICY_WEIGHT * float(comps["soft_policy"].item())
    assert abs(delta - expect) < 1e-5
    assert abs(SOFT_POLICY_WEIGHT - 4.0) < 1e-12


def test_shrimp_loss_omits_soft_when_head_absent() -> None:
    # A model without the soft head (no outputs['soft_policy']) does not raise
    # and does not add a soft component.
    rows = [_row(np.array([0.5, 0.5], dtype=np.float32))]
    batch = collate_training(rows)
    b, npad = batch["policy"].shape
    outputs = {"policy": torch.randn(b, npad), "value": torch.randn(b, 65)}
    bv = dict(batch)
    bv["value"] = torch.zeros(b, dtype=torch.float32)
    _, comps = shrimp_loss(outputs, bv)
    assert "soft_policy" not in comps


def _forward_batch():
    sup = build_support([(0, 0)])
    feats = np.zeros((sup.num_nodes, C.NUM_FEATURES), dtype=np.float32)
    return collate_rows([(sup, feats), (sup, feats)])


def test_model_emits_soft_policy_in_forward_only() -> None:
    batch = _forward_batch()
    model = ShrimpNet().eval()
    with torch.no_grad():
        out = model(batch["feats"], batch["nbr"], batch["mask"], batch["coords"])
        serve = model.forward_policy_value(
            batch["feats"], batch["nbr"], batch["mask"], batch["coords"]
        )
    assert "soft_policy" in out
    # Same shape as the main policy head (B, Npad).
    assert out["soft_policy"].shape == out["policy"].shape
    # The serve path does not emit soft_policy.
    assert "soft_policy" not in serve


def test_warm_start_zero_inits_missing_soft_head() -> None:
    # A checkpoint without soft_policy_* keys warm-starts into a model that has
    # them: matching trunk/heads load, the soft head keeps its fresh
    # _init_weights values, and warm_start_into does not raise.
    donor = ShrimpNet()
    fresh = ShrimpNet()
    # State dict without the soft head keys, built by dropping them from a donor.
    legacy_state = {
        k: v for k, v in donor.state_dict().items() if not k.startswith("soft_policy")
    }
    assert any(k.startswith("soft_policy") for k in fresh.state_dict())
    soft_before = {
        k: v.clone() for k, v in fresh.state_dict().items() if k.startswith("soft_policy")
    }
    summary = warm_start_into(fresh, legacy_state)
    assert summary["shape_mismatch"] == []
    assert summary["unexpected"] == []
    # Every missing key is a soft_policy_* param (nothing else dropped).
    assert summary["missing"] and all(
        k.startswith("soft_policy") for k in summary["missing"]
    )
    after = fresh.state_dict()
    # Matched (non-soft) params equal the donor; the soft head is unchanged.
    for k, v in legacy_state.items():
        assert torch.equal(after[k], v), k
    for k, v in soft_before.items():
        assert torch.equal(after[k], v), k
