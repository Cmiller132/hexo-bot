"""hexfield training plugin: hexo_train component composition boundary."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from hexo_train.components import ComponentOverrides

from .checkpoints import HexfieldCheckpointLoader, HexfieldCheckpointSaver
from .config import parse_hexfield_config
from .evaluation import evaluate_epoch as _evaluate_epoch
from .model import HexfieldNet
from .selfplay import generate_selfplay_epoch
from .trainer import HexfieldTrainer


class HexfieldPlugin:
    name = "hexfield_eq"

    def build_model(self, game_spec: Mapping[str, Any], config: Mapping[str, Any]) -> torch.nn.Module:
        _ = (game_spec, config)
        return HexfieldNet()

    def training_component_overrides(self, *, defaults, config, shared, model) -> ComponentOverrides:
        _ = (defaults, shared)
        if model is None:
            raise ValueError("HexfieldPlugin requires build_model() to run first")
        parsed = parse_hexfield_config(config)
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Weight decay applies to weights with ndim >= 2. Excluded (no
            # decay): biases and other 1-D params, the "tokens" param, and the
            # relative-position bias free params — the Phase-2 orbit-tied
            # "bias_free_table" tables AND the Phase-3b joint (row, head)-tied
            # "bias_theta" params (the latter are 1-D so also excluded on ndim,
            # but named-matched here so a future 2-D reshaping stays no-decay).
            # The register lane's "gate_bias" thresholds (Phase R0) are named the
            # same way for the same reason, as are the ray-tap ".alpha" reach
            # profiles (2-D but structural — decay would pull alpha[0] away
            # from its identity init, SPEC_RAYTAP_CONV.md §2.2). The tied trunk
            # base params (w_base / w0 / EquivLinear.wb) are >= 2-D weights and
            # DO decay, matching the passthrough conv weights; the lane's
            # q/k/v/out projections decay with them, its norm affines and
            # gate_bias do not.
            no_decay_named = (
                ("bias_free_table" in name)
                or ("bias_theta" in name)
                or ("gate_bias" in name)
                or name.endswith(".alpha")
            )
            if param.ndim >= 2 and not no_decay_named and name != "tokens":
                decay.append(param)
            else:
                no_decay.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": parsed.training.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=parsed.training.learning_rate,
        )
        trainer = HexfieldTrainer(model=model, config=parsed, optimizer=optimizer)
        return ComponentOverrides(
            trainer=trainer,
            optimizer=optimizer,
            checkpoint_loader=HexfieldCheckpointLoader(),
            checkpoint_saver=HexfieldCheckpointSaver(),
            uses_shared_sample_store=False,
            extra={
                "model_family": "hexfield_eq",
                "selfplay_mode": "continuous",
                "search_visits": parsed.selfplay.search_visits,
            },
        )

    def generate_selfplay(self, *, ctx, components, epoch: int, games_per_epoch: int) -> dict[str, Any]:
        return generate_selfplay_epoch(
            ctx=ctx, components=components, epoch=epoch, games_per_epoch=games_per_epoch
        )

    def evaluate_epoch(self, *, ctx, components, epoch: int) -> dict[str, Any]:
        return _evaluate_epoch(ctx=ctx, components=components, epoch=epoch)


plugin = HexfieldPlugin()


def get_plugin() -> HexfieldPlugin:
    return plugin
