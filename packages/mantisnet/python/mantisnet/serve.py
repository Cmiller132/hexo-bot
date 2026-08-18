"""Checkpoint loading and the one batched read the serve paths share.

The research repo loads checkpoints inside its training driver; serving needs
exactly two things from a checkpoint file — a strict-loaded model and the
KLENT parameters π′ depends on — and one batched forward that every consumer
(bot search, analysis, lab) reads through.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ._rust import ACTION_ORDER_VERSION, MODEL_REPR_VERSION, RULES_VERSION
from .builder import collate_positions
from .improve import ImprovedPolicy, improved_policy
from .model import MantisConfig, MantisNet, strip_legacy_knobs

# The versions that gate whether a checkpoint's tensors mean what this build
# thinks they mean. The recorded torch build string is deliberately NOT
# compared: checkpoint tensors are portable across torch builds, and pinning
# `2.11.0+cu128` would refuse every CPU/XPU wheel for a CUDA-trained model.
_SEMANTIC_VERSIONS = {
    "MODEL_REPR_VERSION": MODEL_REPR_VERSION,
    "RULES_VERSION": RULES_VERSION,
    "ACTION_ORDER_VERSION": ACTION_ORDER_VERSION,
}


@dataclass(frozen=True)
class KlentParams:
    """The three parameters π′ depends on, as trained."""

    tau: float
    lam: float
    mass_floor: float


@dataclass
class LoadedCheckpoint:
    """A strict-loaded MantisNet plus the opinion parameters it trained with."""

    model: MantisNet
    config: MantisConfig
    klent: KlentParams
    provenance: dict


def load_checkpoint(path: Path | str, device: str = "cpu") -> LoadedCheckpoint:
    """Load an inference export, refusing anything semantically incompatible.

    The file must carry ``model``, ``model_config``, ``versions`` (matching
    this build's semantic versions), and ``klent`` (``tau``/``lam``/
    ``mass_floor``) — π′ is meaningless under parameters the model did not
    train with, so a missing block is an error, never a default.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    recorded = raw.get("versions")
    if not isinstance(recorded, dict):
        raise ValueError(f"{path} carries no versions block")
    for key, expected in _SEMANTIC_VERSIONS.items():
        if recorded.get(key) != expected:
            raise ValueError(
                f"{path} records {key}={recorded.get(key)!r} but this build is "
                f"{key}={expected}"
            )
    klent = raw.get("klent")
    if not isinstance(klent, dict) or not {"tau", "lam", "mass_floor"} <= set(klent):
        raise ValueError(
            f"{path} carries no klent block; tau, lam, and mass_floor are the "
            "parameters π′ depends on and must ride the export"
        )
    config = MantisConfig(**strip_legacy_knobs(raw["model_config"]))
    model = MantisNet(config)
    model.load_state_dict(raw["model"])
    model.to(device).eval()
    return LoadedCheckpoint(
        model=model,
        config=config,
        klent=KlentParams(
            tau=float(klent["tau"]),
            lam=float(klent["lam"]),
            mass_floor=float(klent["mass_floor"]),
        ),
        provenance=dict(raw.get("provenance", {})),
    )


@dataclass
class PositionRead:
    """One batched forward, decoded to fp32 CPU tensors.

    Per-cell tensors are flat over every position's legal cells in engine
    order; ``offsets`` cuts them per position. ``value``/``value_dist`` are
    per position.
    """

    logits: Tensor  # (N,) raw policy logits
    q_score: Tensor  # (N,) the acting score π′ ranks by
    q_values: Tensor  # (N,) action values in [-1, 1]
    value: Tensor  # (P,) the value head's scalar decode, in [-1, 1]
    value_dist: Tensor  # (P, K) softmax over the value bins
    improved: ImprovedPolicy  # π′ / v̂ / KL / normalized entropy
    offsets: Tensor  # (P + 1,) int64 legal offsets

    def row(self, index: int) -> slice:
        """The flat slice of position ``index``'s legal cells."""
        lo, hi = int(self.offsets[index]), int(self.offsets[index + 1])
        return slice(lo, hi)


def read_positions(
    model: MantisNet,
    positions,
    klent: KlentParams,
    device: str = "cpu",
) -> PositionRead:
    """Forward ``positions`` once and decode every serve-relevant head."""
    batch = collate_positions(list(positions)).to(device)
    with torch.no_grad():
        out = model.forward(batch, klent.mass_floor)
    logits = out.policy_logits.float().cpu()
    q_score = out.q_score.float().cpu()
    q_values = out.q_values.float().cpu()
    offsets = batch.legal_offsets.cpu()
    improved = improved_policy(logits, q_score, q_values, offsets, klent.tau, klent.lam)
    return PositionRead(
        logits=logits,
        q_score=q_score,
        q_values=q_values,
        value=out.value.float().cpu(),
        value_dist=out.value_dist.float().cpu(),
        improved=improved,
        offsets=offsets,
    )
