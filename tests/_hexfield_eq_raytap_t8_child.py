"""T8 child (SPEC_RAYTAP_CONV.md §6.1) — one full-shape training step under
the import-time env the parent sets (HEXFIELD_EQ_CHANNELS=192,
HEXFIELD_EQ_TRUNK=CCLACCLACLA, HEXFIELD_EQ_REG_LANE=1,
HEXFIELD_EQ_SUPPORT_RADIUS=4, HEXFIELD_EQ_RAYTAP={0|both}), AMP as in
production (autocast fp16 + GradScaler + the prefit AdamW), B=48, S=648, on
CUDA. Prints one JSON line: {"peak_bytes": ..., "raytap": ...}.

Run by tests/test_hexfield_eq_raytap_t8.py (opt-in: HEXFIELD_RAYTAP_T8=1 and
an IDLE 12 GB-class GPU — never against the live soak).
"""

from __future__ import annotations

import json
import sys

import torch

from hexfield_eq import constants as C
from hexfield_eq.constants import DIRECTIONS
from hexfield_eq.geometry import disk_offsets
from hexfield_eq.model import HexfieldNet
from hexfield_eq.prefit import make_optimizer

B, S = 48, 648


def full_shape_batch(device: str):
    """A synthetic full-shape batch on a G-closed disk board: real neighbour
    graph + coords (the ray index build needs distinct live coords), random
    feats, mid-range random raylen."""

    cells = disk_offsets(15)[:S]  # 721-cell disk sliced to S rows
    n = len(cells)
    cidx = {c: i for i, c in enumerate(cells)}
    nbr1 = torch.full((n, 6), n, dtype=torch.long)
    for i, cell in enumerate(cells):
        for d, off in enumerate(DIRECTIONS):
            nb = (cell[0] + off[0], cell[1] + off[1])
            if nb in cidx:
                nbr1[i, d] = cidx[nb]
    coords1 = torch.tensor([list(c) for c in cells], dtype=torch.long)
    torch.manual_seed(0)
    batch = {
        "feats": torch.randn(B, n, C.NUM_FEATURES),
        "nbr": nbr1.unsqueeze(0).expand(B, -1, -1).contiguous(),
        "mask": torch.ones(B, n, dtype=torch.bool),
        "coords": coords1.unsqueeze(0).expand(B, -1, -1).contiguous(),
        "raylen": torch.randint(0, C.RAY_REACH + 1, (B, n, C.RAYLEN_SLOTS)).to(
            torch.uint8
        ),
    }
    return {k: v.to(device) for k, v in batch.items()}


def main() -> int:
    assert torch.cuda.is_available(), "T8 needs CUDA"
    device = "cuda"
    assert C.CHANNELS == 192 and C.TRUNK_LAYOUT == "CCLACCLACLA" and C.REG_LANE, (
        "T8 env not applied before import"
    )
    model = HexfieldNet().to(device).train()
    opt = make_optimizer(model)
    scaler = torch.amp.GradScaler("cuda")
    batch = full_shape_batch(device)

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(
            batch["feats"], batch["nbr"], batch["mask"], batch["coords"],
            raylen=batch["raylen"],
        )
        loss = sum(v.float().pow(2).mean() for v in out.values())
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    print(json.dumps({
        "raytap": C.RAYTAP,
        "peak_bytes": int(peak),
        "peak_gb": round(peak / 2**30, 3),
        "loss_finite": bool(torch.isfinite(loss)),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
