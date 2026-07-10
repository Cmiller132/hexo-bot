"""BC prefit trainer and probe harness.

Optimizer: AdamW lr 1e-3, wd 1e-4 on matrix weights only (no decay on biases,
LayerNorm params, token inits, bias table), AMP + GradScaler, grad-clip 3.0
(the x12-tied trunk's global grad-norm runs sqrt(12)-12x hotter than a dense
trunk's; a 1.0 clip would silently rescale the LR non-uniformly), 500-step
linear LR warmup then a cosine decay LR -> LR/10 over the run
(trainer.scheduled_lr), an EMA twin (decay 0.9995) updated every step and
evaluated alongside the raw net. One optimizer step per nominal 32-row batch
via pair-budget micro-buckets with step-global denominators.

Per-epoch artifacts written to the run dir: checkpoint (raw + EMA weights),
one diagnostics.jsonl line (train/val CE + gap, per-group grad norms incl.
trunk_reg, token-stream max magnitude, ply-bucketed value MAE, threat-dense
row-subset metrics), and a probe npz over up to PROBE_ROWS frozen validation
rows (policy KL vs the previous epoch, entropy, value ECE vs realized
outcomes). The D6-consistency probe KL is not emitted: the equivariant trunk
(HEXFIELD_EQ_GROUP_ORDER > 1) is D6-invariant by construction, so it is
identically zero (BUGS_FOUND.md; plan §4).
"""

from __future__ import annotations

import os

# HEXFIELD_TRAIN_FLEX is not defaulted on for the prefit. The pair-budget
# micro-buckets pack a variable number of rows per step (batch/seq dim varies),
# which the inner FlexAttention compile (torch.compile(dynamic=False) in
# model.py) recompiles per shape. The prefit uses the eager materialized-bias
# path. Setting HEXFIELD_TRAIN_FLEX=1 in the env forces the flex path.

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .batching import (
    PAD_QUANTUM,
    collate_training,
    pair_budget_microbuckets,
    split_stvalue_columns,
    step_global_denominators,
)
from .constants import F_OPP_FORK, F_OWN_FORK, GROUP_ORDER
from .losses import SOFT_POLICY_WEIGHT, decode_binned_value, hexfield_loss
from .model import HexfieldNet
from .samples import STV_HORIZONS, expand_sample
from .shards import read_compact_shard
from .trainer import _check_raylen_once, scheduled_lr

NOMINAL_BATCH_ROWS = 32
WARMUP_STEPS = 500
LR = 1e-3
FINAL_LR = LR / 10.0  # cosine target over the run (C1)
WEIGHT_DECAY = 1e-4
# 3.0, not 1.0 (C2): the x12-tied trunk's pre-clip global grad-norm runs
# ~sqrt(12)-12x hotter than a dense trunk's; persistent clipping at 1.0 would
# silently rescale the LR non-uniformly across param groups.
GRAD_CLIP = 3.0
# EMA twin decay (C3); evaluated alongside the raw net each epoch.
EMA_DECAY = 0.9995
PROBE_ROWS = 1024
# Threat-dense row subset: rows whose max fork-plane input value reaches this
# (raw per-axis line count >= 2 of 3 => 0.667) — the positions the ray/register
# mechanisms exist for.
THREAT_PLANE_MIN = 0.6

# Ply buckets for the value-MAE breakdown (stones-on-board at the decision).
PLY_BUCKETS = ((0, 10), (11, 20), (21, 40), (41, 10_000))


def make_optimizer(model: HexfieldNet, lr: float = LR) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        # Orbit-tied bias free tables ("bias_free_table*"), the joint-tied
        # "bias_theta" params, the register lane's "gate_bias" thresholds, and
        # the ray-tap ".alpha" reach profiles (2-D but structural — decay
        # would pull alpha[0] away from its identity init, SPEC_RAYTAP_CONV.md
        # §2.2) stay no-decay even if >= 2-D; mirrors HexfieldPlugin's
        # predicate. The lane's q/k/v/out projections decay with the other
        # matrix weights; its norm affines are 1-D and land in no_decay on ndim.
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
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


class BootstrapShards(IterableDataset):
    """Yields ready nominal-step payloads: (list of micro-bucket batches, denoms)."""

    def __init__(
        self, shard_paths: list[Path], seed: int, epoch: int, train: bool,
        batch_rows: int = NOMINAL_BATCH_ROWS,
    ):
        self.shard_paths = list(shard_paths)
        self.seed = seed
        self.epoch = epoch
        self.train = train
        self.batch_rows = int(batch_rows)

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        workers = info.num_workers if info else 1
        rng = random.Random(f"{self.seed}/{self.epoch}/{wid}")
        order = list(self.shard_paths)
        rng.shuffle(order)
        my_shards = order[wid::workers]
        buffer = []
        for path in my_shards:
            buffer.extend(read_compact_shard(path))
            if len(buffer) < 4096 and path is not my_shards[-1]:
                continue
            rng.shuffle(buffer)
            yield from self._emit(buffer, rng)
            buffer = []
        if buffer:
            rng.shuffle(buffer)
            yield from self._emit(buffer, rng)

    def _emit(self, rows, rng):
        # Equivariant build (GROUP_ORDER > 1): the trunk is D6-equivariant by
        # construction, so BC-prefit augmentation is redundant — expand every row
        # under the identity symmetry (BUGS_FOUND.md: zero the caller-side draw;
        # the Rust sym kernel / Python transform path stay intact for the Phase-1
        # parity tests). The GROUP_ORDER == 1 passthrough ablation keeps the
        # random per-row draw.
        augment = self.train and GROUP_ORDER == 1
        for start in range(0, len(rows), self.batch_rows):
            chunk = rows[start : start + self.batch_rows]
            expanded = [
                expand_sample(s, symmetry=rng.randrange(12) if augment else 0)
                for s in chunk
            ]
            denoms = step_global_denominators(expanded, STV_HORIZONS)
            buckets = []
            for bucket in pair_budget_microbuckets(expanded, quantize=PAD_QUANTUM):
                # Round the padded node count up to a multiple of PAD_QUANTUM.
                # pair_budget_microbuckets above used the same quantum for its
                # budget.
                pad_to = -(-max(r.support.num_nodes for r in bucket) // PAD_QUANTUM) * PAD_QUANTUM
                batch = collate_training(bucket, pad_to=pad_to)
                buckets.append(split_stvalue_columns(batch, STV_HORIZONS))
            yield buckets, denoms


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _grad_norm_groups(model) -> dict[str, float]:
    """Per-group pre-clip L2 grad norms (trainer._build_grad_norm_groups's
    predicates, incl. trunk_reg — C4). Called after scaler.unscale_."""

    sq = {"trunk_conv": 0.0, "trunk_attn": 0.0, "trunk_reg": 0.0, "heads": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g2 = float(p.grad.detach().norm(2)) ** 2
        if name.startswith(("stem", "conv_blocks")):
            key = "trunk_conv"
        elif name == "tokens" or name.startswith(
            ("attn_blocks", "bias_free_tables", "bias_theta", "ray_blocks",
             "ray_bias_free_tables")
        ):
            key = "trunk_attn"
        elif name.startswith(("registers", "tok_reads")):
            key = "trunk_reg"
        else:
            key = "heads"
        sq[key] += g2
    return {k: v ** 0.5 for k, v in sq.items()}


def _ema_update(ema_model, model, decay: float = EMA_DECAY) -> None:
    """One EMA step (C3): params lerp toward the live net; buffers copy (they
    are constant LUTs). Runs on the eager module, never a compiled wrapper."""

    with torch.no_grad():
        ema_params = dict(ema_model.named_parameters())
        for name, p in model.named_parameters():
            ema_params[name].mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        ema_bufs = dict(ema_model.named_buffers())
        for name, b in model.named_buffers():
            ema_bufs[name].copy_(b)


@torch.no_grad()
def _token_stream_max(model, samples, device) -> float:
    """Max |pre-ln token stream| over a few probe rows (C4): the register
    lane's count-magnitude watchdog (plan §6 risk 2)."""

    model.eval()
    peak = 0.0
    for sample in samples:
        ident = expand_sample(sample, symmetry=0)
        batch = _to_device(collate_training([ident]), device)
        _cells, _tokens, pre_tokens, _gidx = model.trunk(
            batch["feats"], batch["nbr"], batch["mask"], batch["coords"],
            batch.get("raylen"),
        )
        peak = max(peak, float(pre_tokens.abs().max()))
    model.train()
    return peak


def run_step(
    model, buckets, denoms, device, scaler, optimizer, grad_stats,
    policy_target: str = "visit",
    soft_policy_weight: float = SOFT_POLICY_WEIGHT,
    group_stats: dict[str, list[float]] | None = None,
) -> dict:
    optimizer.zero_grad(set_to_none=True)
    components_sum: dict[str, float] = {}
    any_backward = False
    for batch in buckets:
        batch = _to_device(batch, device)
        _check_raylen_once(getattr(model, "_orig_mod", model), batch)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(
                batch["feats"], batch["nbr"], batch["mask"], batch["coords"],
                raylen=batch.get("raylen"),
            )
        loss, components = hexfield_loss(
            out, batch, denominators=denoms, policy_target=policy_target,
            soft_policy_weight=soft_policy_weight,
        )
        if not torch.isfinite(loss):
            # Skip buckets whose loss is non-finite (e.g. a non-finite cell_q)
            # rather than backpropagating them, and log the cell_q value.
            print(f"  [skip] non-finite bucket (cell_q={float(components.get('cell_q', float('nan'))):.3g})", flush=True)
            continue
        scaler.scale(loss).backward()
        any_backward = True
        for key, val in components.items():
            components_sum[key] = components_sum.get(key, 0.0) + float(val.detach())
    if not any_backward:
        # Every bucket was skipped, so no grads were scaled; skip the optimizer
        # update to avoid GradScaler's "No inf checks were recorded" assert.
        return components_sum
    scaler.unscale_(optimizer)
    if group_stats is not None:
        for key, val in _grad_norm_groups(getattr(model, "_orig_mod", model)).items():
            group_stats.setdefault(key, []).append(val)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    grad_stats.append(float(grad_norm))
    scaler.step(optimizer)
    scaler.update()
    return components_sum


@torch.no_grad()
def evaluate(
    model, shard_paths, device, max_rows: int | None = None,
    policy_target: str = "visit",
    soft_policy_weight: float = SOFT_POLICY_WEIGHT,
) -> dict:
    if max_rows is None:
        # Env-tunable for deadline ladders: the full 43k-row val sweep costs
        # ~12 min/epoch in batch-32 chunks; 12k rows keeps the top-1 SE ~0.4%.
        max_rows = int(os.environ.get("HEXFIELD_EQ_PREFIT_EVAL_ROWS", "50000"))
    model.eval()
    top1_hits = 0
    rows_seen = 0
    policy_ce_sum = 0.0
    value_ce_sum = 0.0
    evs: list[float] = []
    zs: list[float] = []
    # Per-row diagnostics aligned with evs/zs (C4): stones-on-board (the ply
    # bucket key), the threat-dense flag (max fork-plane input), top-1 hit.
    plies: list[int] = []
    threat: list[bool] = []
    hits: list[bool] = []
    for path in shard_paths:
        samples = read_compact_shard(path)
        for start in range(0, len(samples), NOMINAL_BATCH_ROWS):
            chunk = samples[start : start + NOMINAL_BATCH_ROWS]
            expanded = [expand_sample(s, symmetry=0) for s in chunk]
            denoms = step_global_denominators(expanded, STV_HORIZONS)
            chunk_policy = 0.0
            chunk_value = 0.0
            chunk_rows = len(expanded)
            # eval collate pads to the raw max N (pad_to=None), so the bucketing
            # uses quantize=1 to match.
            for bucket in pair_budget_microbuckets(expanded, quantize=1):
                batch = _to_device(
                    split_stvalue_columns(collate_training(bucket), STV_HORIZONS), device
                )
                out = model(
                    batch["feats"], batch["nbr"], batch["mask"], batch["coords"],
                    raylen=batch.get("raylen"),
                )
                _, comps = hexfield_loss(
                    out, batch, denominators=denoms, policy_target=policy_target,
                    soft_policy_weight=soft_policy_weight,
                )
                # components use the chunk-global denominator, so summing the
                # buckets gives the chunk-mean CE
                chunk_policy += float(comps["policy"])
                chunk_value += float(comps["value"])
                b = batch["value"].shape[0]
                ev = decode_binned_value(out["value"])
                evs.extend(ev.tolist())
                zs.extend(batch["value"].tolist())
                plies.extend(int(r.support.stone_count) for r in bucket)
                fork_max = (
                    batch["feats"][:, :, (F_OWN_FORK, F_OPP_FORK)].amax(dim=(1, 2))
                )
                threat.extend((fork_max >= THREAT_PLANE_MIN).tolist())
                npad = out["policy"].shape[1]
                prefix = torch.arange(npad, device=device).unsqueeze(0) < batch[
                    "legal_counts"
                ].unsqueeze(1)
                masked_logits = out["policy"].masked_fill(~prefix, float("-inf"))
                pred = masked_logits.argmax(dim=1)
                # Top-1 target mirrors the hexfield_loss per-row blend: rows with
                # a valid gumbel target are scored against it, others against the
                # visit histogram.
                target_dist = batch["policy"]
                if (
                    policy_target == "gumbel"
                    and "gumbel_policy" in batch
                    and "gumbel_policy_valid" in batch
                ):
                    use_g = (batch["gumbel_policy_valid"] > 0.0).to(target_dist.dtype)
                    use_g = use_g.unsqueeze(1)
                    target_dist = (
                        use_g * batch["gumbel_policy"] + (1.0 - use_g) * batch["policy"]
                    )
                tgt = target_dist.argmax(dim=1)
                hits.extend((pred == tgt).tolist())
                top1_hits += int((pred == tgt).sum())
                rows_seen += b
            policy_ce_sum += chunk_policy * chunk_rows
            value_ce_sum += chunk_value * chunk_rows
        if rows_seen >= max_rows:
            break
    evs_arr = np.asarray(evs)
    zs_arr = np.asarray(zs)
    deciles = np.quantile(evs_arr, np.linspace(0, 1, 11))
    ece = 0.0
    for k in range(10):
        lo, hi = deciles[k], deciles[k + 1]
        sel = (evs_arr >= lo) & (evs_arr <= hi if k == 9 else evs_arr < hi)
        if sel.sum() == 0:
            continue
        ece += sel.mean() * abs(evs_arr[sel].mean() - zs_arr[sel].mean())
    model.train()
    metrics = {
        "val_rows": rows_seen,
        "top1": top1_hits / max(rows_seen, 1),
        "policy_ce": policy_ce_sum / max(rows_seen, 1),
        "value_ce": value_ce_sum / max(rows_seen, 1),
        "value_ece": float(ece),
        "value_optimism": float(evs_arr.mean() - zs_arr.mean()),
    }
    # Ply-bucketed value MAE (C4, the head_audit bucket idea): |EV - z| per
    # stones-on-board band.
    mae = np.abs(evs_arr - zs_arr)
    ply_arr = np.asarray(plies)
    for lo, hi in PLY_BUCKETS:
        sel = (ply_arr >= lo) & (ply_arr <= hi)
        metrics[f"value_mae_ply_{lo}_{hi if hi < 10_000 else 'up'}"] = (
            float(mae[sel].mean()) if sel.any() else None
        )
    # Threat-dense subset (C4): the rows the ray/register mechanisms exist for.
    threat_arr = np.asarray(threat, dtype=bool)
    hits_arr = np.asarray(hits, dtype=bool)
    metrics["threat_rows"] = int(threat_arr.sum())
    metrics["threat_top1"] = (
        float(hits_arr[threat_arr].mean()) if threat_arr.any() else None
    )
    metrics["threat_value_mae"] = (
        float(mae[threat_arr].mean()) if threat_arr.any() else None
    )
    return metrics


@torch.no_grad()
def run_probe(model, probe_samples, device, prev_probs: list[np.ndarray] | None):
    """Frozen-probe forward: policy KL vs prev_probs, entropy, E[v].

    The D6-consistency KL (``probe_d6_kl``) is intentionally omitted: the
    equivariant trunk (HEXFIELD_EQ_GROUP_ORDER > 1) is D6-invariant by
    construction, so that metric is identically zero and its per-row rotated
    second forward is pure overhead (BUGS_FOUND.md; plan §4). The GROUP_ORDER == 1
    passthrough ablation is non-equivariant, but the probe is a prefit
    diagnostic, not a gate, so it is dropped uniformly."""

    model.eval()
    probs_out: list[np.ndarray] = []
    entropy: list[float] = []
    evs: list[float] = []
    kl_prev: list[float] = []
    for i, sample in enumerate(probe_samples):
        ident = expand_sample(sample, symmetry=0)
        batch = _to_device(collate_training([ident]), device)
        out = model(
            batch["feats"], batch["nbr"], batch["mask"], batch["coords"],
            raylen=batch.get("raylen"),
        )
        l = ident.support.legal_count
        p = torch.softmax(out["policy"][0, :l].float(), dim=0)
        probs_out.append(p.cpu().numpy())
        entropy.append(float(-(p * (p + 1e-12).log()).sum()))
        evs.append(float(decode_binned_value(out["value"])[0]))
        p_np = probs_out[-1]
        if prev_probs is not None:
            prev = prev_probs[i]
            kl_prev.append(float(np.sum(p_np * (np.log(p_np + 1e-12) - np.log(prev + 1e-12)))))
    model.train()
    metrics = {
        "probe_entropy": float(np.mean(entropy)),
        "probe_ev_mean": float(np.mean(evs)),
        "probe_policy_kl_prev": float(np.mean(kl_prev)) if kl_prev else None,
    }
    return probs_out, metrics


def save_checkpoint(
    path: Path, model, optimizer, scaler, epoch: int, global_step: int,
    ema_model=None,
):
    # meta mirrors the trainer path (checkpoints.py): foreign loaders (eval
    # arena, dashboard) rebuild the arch meta-first, so prefit checkpoints must
    # be self-describing too. `model` is the eager module, not a compiled wrapper.
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "meta": {
            "lineage": "hexfield_eq",
            "epoch": int(epoch),
            **model.arch_meta(),
        },
    }
    if ema_model is not None:
        payload["ema_model"] = ema_model.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: Path, model, optimizer=None, scaler=None, ema_model=None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # D1 (spec D-S32): a resumed checkpoint must match the BUILT model's arch
    # semantics — a ray_blockers flip in particular is a silent mask-semantics
    # change on an arm-4c resume, and trunk_layout / lane toggles would
    # otherwise only fail via oblique key mismatches (ray_blockers not at all).
    meta = payload.get("meta") or {}
    from .support import _SUPPORT_RADIUS

    for key, want in (
        ("trunk_layout", model._trunk_layout),
        ("reg_lane", model._reg_lane),
        ("reg_tok_read", model._reg_tok_read),
        ("ray_blockers", getattr(model, "_ray_blockers", None)),
        ("support_radius", _SUPPORT_RADIUS),
    ):
        got = meta.get(key)
        if got is not None and want is not None and got != want:
            raise ValueError(
                f"checkpoint meta {key}={got!r} != built model's {want!r}; "
                "refusing the resume (silent semantics flip)"
            )
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    if ema_model is not None:
        # Resume the EMA twin when present; otherwise re-seed it from the raw
        # weights (an EMA restarted mid-run converges back within ~1/decay steps).
        ema_model.load_state_dict(payload.get("ema_model", payload["model"]), strict=True)
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="dir with train/ and val/ shards")
    parser.add_argument("--out", required=True, help="run dir")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit-steps", type=int, default=0, help="smoke: stop each epoch early")
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--policy-target", choices=("visit", "gumbel"), default="visit",
        help="main-policy CE target; 'gumbel' uses the per-row completedQ blend "
        "(rows with gumbel_policy_valid > 0) — required for Gumbel-regime corpora",
    )
    parser.add_argument(
        "--soft-policy-weight", type=float, default=SOFT_POLICY_WEIGHT,
        help="soft-policy loss weight passed to hexfield_loss (default "
        f"{SOFT_POLICY_WEIGHT}); the ladder probes the soft-policy dominance "
        "with it (C4)",
    )
    # Throughput regime knobs (deadline ladder): a larger nominal batch feeds
    # the GPU (raise HEXFIELD_EQ_PAIR_BUDGET alongside so the micro-buckets
    # actually grow); scale --lr with it (~sqrt rule) and shorten warmup so it
    # stays a small fraction of the shorter run. All arms must share one regime.
    parser.add_argument("--batch-rows", type=int, default=NOMINAL_BATCH_ROWS,
                        help=f"rows per optimizer step (default {NOMINAL_BATCH_ROWS})")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"peak LR (default {LR}); cosine-decays to lr/10")
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS,
                        help=f"linear LR warmup steps (default {WARMUP_STEPS})")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    # TF32 for fp32 matmuls (group-norm/eval paths); autocast fp16 covers the
    # rest. Set before any model/compile work so every ladder arm shares
    # numerics.
    torch.set_float32_matmul_precision("high")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_shards = sorted(Path(args.data, "train").glob("shard_*.npz"))
    val_shards = sorted(Path(args.data, "val").glob("shard_*.npz"))
    if not train_shards or not val_shards:
        raise SystemExit(f"no shards under {args.data}")

    torch.manual_seed(args.seed)
    model = HexfieldNet().to(device)
    optimizer = make_optimizer(model, lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    # EMA twin (C3): updated every optimizer step, evaluated alongside the raw
    # net, saved in the checkpoint. Grad-free eager module.
    ema_model = copy.deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    global_step = 0
    start_epoch = 0
    if args.resume:
        payload = load_checkpoint(Path(args.resume), model, optimizer, scaler, ema_model)
        global_step = int(payload["global_step"])
        start_epoch = int(payload["epoch"]) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}", flush=True)

    # torch.compile the forward with dynamic=True, since the pair-budget
    # micro-buckets vary the batch/seq dim. The eager `model` is retained for
    # make_optimizer and save_checkpoint: the compiled wrapper prefixes
    # state_dict keys with "_orig_mod.", which the checkpoint loader does not
    # expect. Set HEXFIELD_PREFIT_COMPILE=0 to use plain eager.
    fwd = model
    if os.environ.get("HEXFIELD_PREFIT_COMPILE", "1") == "1":
        try:
            fwd = torch.compile(model, dynamic=True)
        except Exception as exc:  # pragma: no cover - older torch / compile unavailable
            print(f"torch.compile unavailable ({exc}); using eager forward", flush=True)

    # Frozen probe rows: first PROBE_ROWS rows of the (fixed-order) val shards.
    probe_samples = []
    for path in val_shards:
        probe_samples.extend(read_compact_shard(path))
        if len(probe_samples) >= PROBE_ROWS:
            break
    probe_samples = probe_samples[:PROBE_ROWS]
    prev_probs = None

    diag_path = out_dir / "diagnostics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        dataset = BootstrapShards(
            train_shards, args.seed, epoch, train=True, batch_rows=args.batch_rows
        )
        loader = DataLoader(
            dataset, batch_size=None, num_workers=args.workers,
            persistent_workers=False, prefetch_factor=4 if args.workers else None,
        )
        t0 = time.time()
        comp_totals: dict[str, float] = {}
        grad_stats: list[float] = []
        group_stats: dict[str, list[float]] = {}
        steps = 0
        lr = LR
        for buckets, denoms in loader:
            # Linear warmup then cosine decay LR -> LR/10 over the run (C1;
            # trainer.scheduled_lr — warmup in optimizer steps, decay in epochs).
            lr = scheduled_lr(
                schedule="cosine", base_lr=args.lr, final_lr=args.lr / 10.0,
                warmup_steps=args.warmup_steps, decay_epochs=args.epochs,
                global_step=global_step, epoch=epoch,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            comps = run_step(
                fwd, buckets, denoms, device, scaler, optimizer, grad_stats,
                policy_target=args.policy_target,
                soft_policy_weight=args.soft_policy_weight,
                group_stats=group_stats,
            )
            _ema_update(ema_model, model)
            for key, val in comps.items():
                comp_totals[key] = comp_totals.get(key, 0.0) + val
            steps += 1
            global_step += 1
            if steps % 25 == 0:
                print(f"epoch {epoch} step {steps} total {comp_totals['total']/steps:.4f} "
                      f"({(time.time()-t0)/steps:.3f}s/step)", flush=True)
            if args.limit_steps and steps >= args.limit_steps:
                break

        grads = np.asarray(grad_stats)
        val_metrics = evaluate(
            fwd, val_shards, device, policy_target=args.policy_target,
            soft_policy_weight=args.soft_policy_weight,
        )
        ema_metrics = evaluate(
            ema_model, val_shards, device, policy_target=args.policy_target,
            soft_policy_weight=args.soft_policy_weight,
        )
        prev_probs, probe_metrics = run_probe(fwd, probe_samples, device, prev_probs)
        train_policy = comp_totals.get("policy", 0.0) / max(steps, 1)
        train_value = comp_totals.get("value", 0.0) / max(steps, 1)
        record = {
            "epoch": epoch,
            "steps": steps,
            "global_step": global_step,
            "seconds": round(time.time() - t0, 1),
            "lr": lr,
            **{f"train_{k}": v / max(steps, 1) for k, v in comp_totals.items()},
            "grad_norm_mean": float(grads.mean()) if len(grads) else None,
            "grad_norm_p95": float(np.percentile(grads, 95)) if len(grads) else None,
            "clip_fraction": float((grads > GRAD_CLIP).mean()) if len(grads) else None,
            # Per-group pre-clip grad norms incl. the register lane (C4/C5).
            **{
                f"grad_norm_{g}": float(np.mean(vals)) if vals else None
                for g, vals in sorted(group_stats.items())
            },
            "amp_scale": float(scaler.get_scale()),
            # The register lane's count-magnitude watchdog (C4; plan §6 risk 2).
            "token_stream_max": _token_stream_max(model, probe_samples[:64], device),
            **val_metrics,
            **{f"ema_{k}": v for k, v in ema_metrics.items()},
            # Train-vs-val CE gaps (C4): the underfit/overfit needle the gate
            # reads directly.
            "train_val_policy_ce_gap": train_policy - val_metrics["policy_ce"],
            "train_val_value_ce_gap": train_value - val_metrics["value_ce"],
            **probe_metrics,
        }
        with open(diag_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        np.savez_compressed(
            out_dir / f"probe_epoch{epoch}.npz",
            probs=np.asarray(prev_probs, dtype=object),
            entropy=probe_metrics["probe_entropy"],
        )
        save_checkpoint(
            out_dir / f"checkpoint_epoch{epoch}.pt", model, optimizer, scaler,
            epoch, global_step, ema_model=ema_model,
        )


if __name__ == "__main__":
    main()
