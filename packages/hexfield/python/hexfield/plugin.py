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
    name = "hexfield"

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
            # decay): biases and other 1-D params, the "tokens" param, and
            # params whose name contains "bias_table".
            if param.ndim >= 2 and "bias_table" not in name and name != "tokens":
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
                "model_family": "hexfield",
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
