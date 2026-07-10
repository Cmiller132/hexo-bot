"""Default sample target builders for common policy/value models.

This module handles only the shared case: a policy logit for each legal action
plus an optional scalar value. Model-specific heads, pair targets, search
traces, and auxiliary labels remain model-owned extensions.

Status (2026-06): wired but functionally unused. The two helper dataclasses
are instantiated into DefaultTrainingComponents by
`packages/hexo_train/python/hexo_train/defaults.py` on every pipeline run,
but no model package ever reads those handles back and no caller now exercises
the target builder. Every real model builds its own targets (e.g.
`packages/hexfield/python/hexfield/samples.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from hexo_utils.encoding.symmetry import (
    IDENTITY_D6,
    ActionSymmetryMapper,
    D6Symmetry,
    transform_action_ids,
)

from .records import TrainingSampleRecord


@dataclass(frozen=True, slots=True)
class ScalarValueTargetHelper:
    """Reusable scalar value target for simple win/loss/draw models."""

    win_value: float = 1.0
    loss_value: float = -1.0
    draw_value: float = 0.0

    def from_terminal_result(
        self,
        *,
        winner: Any | None,
        perspective: Any,
        is_draw: bool = False,
    ) -> float:
        """Return the value target from the sample player's perspective."""

        if is_draw or winner is None:
            return self.draw_value
        if winner == perspective:
            return self.win_value
        return self.loss_value


@dataclass(frozen=True, slots=True)
class LegalPolicyTargetHelper:
    """Reusable helper for weights over engine-provided legal actions."""

    def normalize(self, weights: Mapping[Any, float]) -> Mapping[Any, float]:
        """Normalize action weights into a probability distribution."""

        total = sum(max(0.0, float(weight)) for weight in weights.values())
        if total <= 0.0:
            return {action: 0.0 for action in weights}
        return {
            action: max(0.0, float(weight)) / total
            for action, weight in weights.items()
        }


# DEPRECATED(2026-06-12): no remaining caller; superseded by model-owned target
# construction (each model package builds dense tensors straight from its own
# sample rows).
@dataclass(frozen=True, slots=True)
class LegalPolicyValueTarget:
    """Default target for models trained on legal-action policy logits/value."""

    legal_action_ids: tuple[int, ...]
    policy_logits: object | None = None
    policy_logits_ref: object | None = None
    selected_action_id: int | None = None
    value: float | None = None
    symmetry: D6Symmetry = IDENTITY_D6
    metadata: Mapping[str, Any] = field(default_factory=dict)


# DEPRECATED(2026-06-12): no remaining caller (model packages build their own
# policy/value targets).
def build_legal_policy_value_target(
    record: TrainingSampleRecord,
    *,
    symmetry: D6Symmetry = IDENTITY_D6,
    action_mapper: ActionSymmetryMapper | None = None,
) -> LegalPolicyValueTarget:
    """Build the shared policy/value target for one training sample.

    The policy vector stays paired with legal-action order. Under symmetry, the
    action ids are transformed but their associated logits remain in the same
    sequence positions. Dense tensors, masks, and model-specific targets are
    still built by the model package after it receives this target.
    """

    if record.policy is None:
        raise ValueError("TrainingSampleRecord has no common policy record")

    if symmetry != IDENTITY_D6 and action_mapper is None:
        raise ValueError("non-identity symmetry requires an action_mapper")

    legal_action_ids = tuple(record.legal_action_ids)
    selected_action_id = record.policy.selected_action_id

    if action_mapper is not None:
        legal_action_ids = transform_action_ids(legal_action_ids, symmetry, action_mapper)
        if selected_action_id is not None:
            selected_action_id = action_mapper.transform_action_id(selected_action_id, symmetry)

    _validate_logits_shape(legal_action_ids, record.policy.logits)

    return LegalPolicyValueTarget(
        legal_action_ids=legal_action_ids,
        policy_logits=record.policy.logits,
        policy_logits_ref=record.policy.logits_ref,
        selected_action_id=selected_action_id,
        value=record.policy.value,
        symmetry=symmetry,
        metadata=record.policy.metadata,
    )


def _validate_logits_shape(action_ids: Sequence[int], logits: object | None) -> None:
    """Catch obvious mismatches without requiring a tensor dependency."""

    if logits is None or not hasattr(logits, "__len__"):
        return
    if len(logits) != len(action_ids):
        raise ValueError("policy logits length must match legal_action_ids length")
