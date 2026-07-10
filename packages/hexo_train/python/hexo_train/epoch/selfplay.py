"""Self-play generation for one training epoch.

This file is the handoff from training orchestration to model-owned execution.
All four registered plugins implement `generate_selfplay()` and run their own
game loops (e.g. packages/dense_cnn_restnet/python/dense_cnn_restnet/selfplay.py
for the active lineage, which drives the shared Rust MCTS and writes NPZ shards
under the run dir), so the first dispatch branch below is the only one that
executes in any configured run. The `build_selfplay_request` and placeholder
branches are retained scaffolding from before plugins owned self-play.

Called once per epoch by hexo_train/epoch/loop.py (`run_epoch`).
"""

from __future__ import annotations

from typing import Any

from hexo_train.components import TrainingComponents
from hexo_train.context import RunContext


def generate_selfplay(
    ctx: RunContext,
    components: TrainingComponents,
    *,
    epoch: int,
) -> dict[str, Any]:
    """Generate or plan self-play samples for one epoch.

    Resolution order:

    1. Prefer a plugin's `generate_selfplay()` hook when it exists.
    2. Otherwise let a plugin build a structured request for future runner use.
    3. Otherwise return a clear placeholder payload.

    The result is stored on `components.shared.selfplay_result` so the sample
    finalizer can see what self-play produced or planned.
    """

    games_per_epoch = ctx.config.selfplay.games_per_epoch
    plugin = components.model.plugin

    if hasattr(plugin, "generate_selfplay"):
        # Full implementation path: the plugin binds model-owned players and
        # sample writers, then calls the runner or equivalent execution layer.
        result = plugin.generate_selfplay(
            ctx=ctx,
            components=components,
            epoch=epoch,
            games_per_epoch=games_per_epoch,
        )
    # UNUSED(2026-06-12): no plugin implements build_selfplay_request — repo-wide
    # grep (packages/, tests/, scripts/ excl. archive) finds only this dispatch.
    # All four registered plugins implement generate_selfplay, so the branch
    # above always wins; this and the placeholder branch below are unreachable
    # for every configured model.
    elif hasattr(plugin, "build_selfplay_request"):
        # Transitional path: useful while runner wiring is not complete but a
        # plugin can already describe the self-play work it needs.
        result = {
            "status": "planned",
            "request": plugin.build_selfplay_request(
                ctx=ctx,
                components=components,
                epoch=epoch,
                games_per_epoch=games_per_epoch,
            ),
            "note": "Future implementation should call hexo_runner self-play.",
        }
    else:
        # Last-resort placeholder keeps the pipeline shape executable while
        # making the missing runner/model integration explicit in diagnostics.
        result = {
            "status": "planned",
            "epoch": epoch,
            "games_per_epoch": games_per_epoch,
            "checkpoint_state": components.shared.checkpoint_state,
            "note": (
                "Runner self-play wiring is not implemented yet. Model plugins "
                "should bind sample stores into model-owned players or writers "
                "before calling the runner."
            ),
        }

    components.shared.selfplay_result = result
    return result
