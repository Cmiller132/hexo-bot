"""Default training components provided by `hexo_train`.

Defaults are intentionally small. They are useful for common policy/value
models, directory layout, and diagnostics, but they do not define what a
model's tensors mean. A plugin may accept a default or replace it.

This file should stay free of lifecycle decisions and file-writing side
effects. It only builds reusable handles that other pipeline files consume.
"""

from __future__ import annotations

from typing import Any, Mapping

from hexo_utils.samples import LegalPolicyTargetHelper, ScalarValueTargetHelper

from .artifacts import CheckpointStore
from .components import DefaultTrainingComponents, SharedComponents
from .context import RunContext
from .symmetry import D6SymmetrySelector


def build_shared_components(ctx: RunContext) -> SharedComponents:
    """Build model-neutral handles for one training run.

    The returned `SharedComponents` starts mostly empty. Later pipeline steps
    fill in mutable handles such as sample store, sample window, selected
    symmetries, and checkpoint state.
    """

    checkpoint_store = CheckpointStore(ctx.checkpoint_dir)
    defaults = DefaultTrainingComponents(
        # UNUSED(2026-06-12): the two target helpers are merged into
        # ModelComponents by components.build_model_components, but repo-wide
        # grep finds no plugin or trainer that ever reads
        # components.model.scalar_value_target / legal_policy_target back —
        # every model package builds its own targets. Wired but functionally
        # inert; kept so the default container stays complete.
        scalar_value_target=ScalarValueTargetHelper(),
        legal_policy_target=LegalPolicyTargetHelper(),
        symmetry_selector=D6SymmetrySelector(),
        checkpoint_store=checkpoint_store,
        diagnostics=ctx.diagnostics,
    )
    shared = SharedComponents(
        defaults=defaults,
        game_spec=_build_game_spec(ctx.section("shared")),
    )
    return shared


def _build_game_spec(shared_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Describe engine/game dimensions needed by model construction.

    The shape is still intentionally loose because the engine/game spec has not
    become a strongly typed training contract yet.
    """

    return dict(shared_config.get("game", {}))
