"""Export a lean ``{meta, model}`` hexfield_eq inference checkpoint.

Example from the repository root::

    python scripts/export_hexfield_eq_infer.py \
      /mnt/e/Hexo-BotTrainer/runs/hexfield_eq_main_2/checkpoints/epoch_000070.pt \
      --out apps/showcase/deploy/models/hexfield_eq_main2_ep70_infer.pt --verify
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PACKAGE_SRC = REPO / "packages" / "hexfield_eq" / "python"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

ARCH_ENV = {
    "channels": "HEXFIELD_EQ_CHANNELS",
    "group_order": "HEXFIELD_EQ_GROUP_ORDER",
    "c_orbit": "HEXFIELD_EQ_C_ORBIT",
    "attention_heads": "HEXFIELD_EQ_ATTENTION_HEADS",
    "support_radius": "HEXFIELD_EQ_SUPPORT_RADIUS",
    "trunk_layout": "HEXFIELD_EQ_TRUNK",
    "reg_lane": "HEXFIELD_EQ_REG_LANE",
    "reg_tok_read": "HEXFIELD_EQ_REG_TOK_READ",
    "cell_q": "HEXFIELD_EQ_CELL_Q",
    "feature_version": "HEXFIELD_EQ_FEATURE_VERSION",
    "raytap": "HEXFIELD_EQ_RAYTAP",
    "ray_blockers": "HEXFIELD_EQ_RAY_BLOCKERS",
}


def _string(value: object) -> str:
    return "1" if value is True else "0" if value is False else str(value)


def export(source: Path, out: Path, *, verify: bool) -> None:
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise SystemExit(f"error: checkpoint has no model state dict: {source}")
    if not isinstance(payload.get("meta"), dict):
        raise SystemExit(f"error: checkpoint has no architecture meta: {source}")
    meta = dict(payload["meta"])
    missing = [
        key
        for key in ARCH_ENV
        if key not in ("ray_blockers", "cell_q") and key not in meta
    ]
    if missing:
        raise SystemExit(f"error: checkpoint meta lacks architecture keys: {missing}")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "model": payload["model"]}, out)
    print(f"wrote {out} (dropped optimizer and training state)")

    if verify:
        # Architecture must be seeded before the first hexfield_eq import.
        for key, env_name in ARCH_ENV.items():
            os.environ[env_name] = _string(meta.get(key, True))
        from hexfield_eq.eval_arena import _load_hexfield_net

        model = _load_hexfield_net(out)
        print(f"verify OK: strict-loaded {sum(p.numel() for p in model.parameters()):,} params")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--out", type=Path,
        default=Path("apps/showcase/deploy/models/hexfield_eq_main2_ep70_infer.pt"),
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    export(args.checkpoint, args.out, verify=args.verify)


if __name__ == "__main__":
    main()
