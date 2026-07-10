"""Checkpoint IO for the hexo_train pipeline.

Provides a strict load path that enforces bidirectional state-dict key equality
and a tolerant weights-only warm-start path."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

from .constants import FEATURE_VERSION
from .model import HexfieldNet
from .support import _SUPPORT_RADIUS
from .train_state import HexfieldTrainState


def save_checkpoint(path: Path, *, model: HexfieldNet, optimizer, epoch: int, extra: dict | None = None) -> Path:
    # Persist the load-bearing arch self-description (group_order / c_orbit /
    # feature_width / channels / heads / trunk_layout / bias reduction) so foreign
    # loaders rebuild the tie from meta first; the geometric LUTs also ride the
    # state dict as persistent buffers. `extra` wins on key collisions.
    arch = model.arch_meta() if hasattr(model, "arch_meta") else {}
    payload = {
        "meta": {"lineage": "hexfield_eq", "epoch": int(epoch), **arch, **(extra or {})},
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic save: the host takes real power cuts, and the supervisor always
    # resumes from the highest-numbered epoch_*.pt with no fallback — a torn
    # newest checkpoint crash-loops the run. Write to a temp file IN THE SAME
    # directory, fsync, then os.replace onto the target so the target is only
    # ever the fully-flushed file or the previous one. The tmp suffix keeps the
    # name off the supervisor's `epoch_*.pt` glob (it must end at `.pt`).
    tmp = path.with_name(path.name + ".tmp")
    # Clean up any stale tmp left by a previous crash (guarded for portability).
    try:
        tmp.unlink()
    except OSError:
        pass
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    # Best-effort directory fsync so the rename itself is durable. Guarded:
    # os.open(dir) / fsync on a directory is not portable to Windows test runs.
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass
    return path


def load_into(model: HexfieldNet, payload: dict, *, optimizer=None) -> dict:
    state = payload["model"]
    # The featurizer support radius is part of the input contract (spec
    # D-S26): a checkpoint trained at a different radius would silently shift
    # the input distribution, so a recorded mismatch fails loudly.
    meta_radius = (payload.get("meta") or {}).get("support_radius")
    if meta_radius is not None and int(meta_radius) != _SUPPORT_RADIUS:
        raise ValueError(
            f"checkpoint support_radius={int(meta_radius)} != this build's "
            f"HEXFIELD_EQ_SUPPORT_RADIUS={_SUPPORT_RADIUS}; refusing the load"
        )
    # The featurizer plane-map version is likewise part of the input contract
    # (SPEC_RAYTAP_CONV.md §1.1): a checkpoint trained under the other map
    # would silently read permuted/missing planes, so a recorded mismatch
    # fails loudly (same class as the support_radius check above).
    meta_fv = (payload.get("meta") or {}).get("feature_version")
    if meta_fv is not None and int(meta_fv) != FEATURE_VERSION:
        raise ValueError(
            f"checkpoint feature_version={int(meta_fv)} != this build's "
            f"HEXFIELD_EQ_FEATURE_VERSION={FEATURE_VERSION}; refusing the load"
        )
    expected = set(model.state_dict().keys())
    got = set(state.keys())
    if expected != got:
        missing = sorted(expected - got)[:5]
        unexpected = sorted(got - expected)[:5]
        raise ValueError(
            f"hexfield checkpoint key mismatch: missing={missing} unexpected={unexpected}"
        )
    model.load_state_dict(state, strict=True)
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
        # Move optimizer state tensors to the model's device (they load on CPU).
        dev = next(model.parameters()).device
        for st in optimizer.state.values():
            for key, val in st.items():
                if isinstance(val, torch.Tensor):
                    st[key] = val.to(dev)
    return payload.get("meta", {})


def warm_start_into(model: HexfieldNet, state: dict) -> dict:
    """Tolerant weights-only warm start.

    Loads every checkpoint key that is present in the model's state dict and has
    a matching shape. Model params missing from the checkpoint keep their
    freshly-constructed value. Shape-mismatched keys and checkpoint-only
    (unexpected) keys are dropped rather than loaded.

    Returns a summary dict {loaded, missing, unexpected, shape_mismatch}.
    """

    model_sd = model.state_dict()
    to_load: dict[str, torch.Tensor] = {}
    shape_mismatch: list[str] = []
    for key, val in state.items():
        if key not in model_sd:
            continue  # checkpoint-only key: dropped
        if model_sd[key].shape != val.shape:
            shape_mismatch.append(key)
            continue
        to_load[key] = val
    missing = sorted(set(model_sd) - set(to_load))
    unexpected = sorted(set(state) - set(model_sd))
    # strict=False: missing keys keep their initialized value; the matched
    # subset is copied into the live params (dtype/device unchanged).
    model.load_state_dict(to_load, strict=False)
    return {
        "loaded": len(to_load),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": shape_mismatch,
    }


class HexfieldCheckpointLoader:
    """hexo_train contract: load(ref, ctx, components) -> state dict.

    resume_from -> {"status": "loaded", "epoch": N};
    initialize_from -> weights-only warm start;
    None -> fresh random init.
    """

    def load(self, checkpoint_ref, *, ctx, components) -> dict[str, Any]:
        model = components.model.model
        optimizer = components.model.optimizer
        if checkpoint_ref is None:
            return {"status": "initialized", "note": "fresh init"}
        path = Path(checkpoint_ref)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        # Prefit checkpoints store the raw prefit dict; pipeline epochs
        # store the {meta, model, optimizer} shape.
        if "meta" in payload:
            resume = ctx.config.checkpoint.resume_from is not None
            # optimizer.load_state_dict restores the checkpoint's saved lr into
            # the param groups. Capture the current (config-built) lr before the
            # load and re-apply it after, so a config lr edit takes effect on
            # resume while the optimizer moment buffers are kept.
            config_lr = (
                [g["lr"] for g in optimizer.param_groups]
                if (resume and optimizer is not None)
                else None
            )
            if not resume:
                # initialize_from warm start (meta-shape checkpoint): tolerant
                # load of matching weights; optimizer state is not restored.
                summary = warm_start_into(model, payload["model"])
                return {
                    "status": "initialized_from",
                    "path": str(path),
                    "warm_start": summary,
                }
            # resume path (the warm-start branch returned above).
            meta = load_into(model, payload, optimizer=optimizer)
            if config_lr is not None:
                for group, lr in zip(optimizer.param_groups, config_lr):
                    group["lr"] = lr
            # Restore the train-bucket governor state from meta. A missing key
            # yields from_dict(None) -> fresh state.
            trainer = getattr(components.model, "trainer", None)
            if trainer is not None:
                trainer.train_state = HexfieldTrainState.from_dict(meta.get("train_state"))
            return {"status": "loaded", "epoch": int(meta.get("epoch", 0)), "path": str(path)}
        # prefit shape (raw prefit dict): tolerant warm start of matching weights.
        summary = warm_start_into(model, payload["model"])
        return {
            "status": "initialized_from",
            "path": str(path),
            "source": "bc_prefit",
            "warm_start": summary,
        }


class HexfieldCheckpointSaver:
    """hexo_train contract: save(name, ctx, components) -> path."""

    def save(self, *, name: str, ctx, components) -> Path:
        epoch = 0
        match = re.search(r"epoch_(\d+)", name)
        if match:
            epoch = int(match.group(1))
        # Persist the train-bucket governor state inside the checkpoint meta
        # when the trainer exposes a train_state.
        trainer = getattr(components.model, "trainer", None)
        extra = {
            "run": ctx.config.run.name,
            **(
                {"train_state": trainer.train_state.to_dict()}
                if getattr(trainer, "train_state", None) is not None
                else {}
            ),
        }
        return save_checkpoint(
            ctx.checkpoint_dir / f"{name}.pt",
            model=components.model.model,
            optimizer=components.model.optimizer,
            epoch=epoch,
            extra=extra,
        )
