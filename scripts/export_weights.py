"""Export an inference-only weights file from a shrimp training checkpoint.

A training checkpoint (runs/<run>/checkpoints/epoch_XXXXXX.pt) carries the model
state dict plus training-only bookkeeping (optimizer state, run/train_state
metadata). This script strips everything but the model weights, embeds a small
arch-metadata dict (channels / heads / trunk / epoch / run / export date), and
writes a compact weights-only .pt suitable for shipping and for the dashboard
Debug + Match tools.

Checkpoint shape (produced by shrimp.checkpoints.save_checkpoint):
    {"meta": {"lineage": "shrimp", "epoch": int, "run": str, ...},
     "model": <state_dict>,
     "optimizer": <state_dict or None>}
(A behavior-cloning prefit checkpoint instead has flat "epoch"/"global_step"/
"scaler" keys and no "meta"; both keep the weights under "model".)

The export keeps the shrimp-lineage shape so the exported file loads through
the same debug/eval loaders:
    {"meta": {..., "arch": {...}, "export": {...}}, "model": <state_dict>}

Usage:
    python scripts/export_weights.py CHECKPOINT.pt --out models/shrimp_main7_infer.pt \\
        --run shrimp_main_7 [--channels 192 --heads 3 --trunk CCACCACCACCACCA]
    python scripts/export_weights.py CHECKPOINT.pt --out OUT.pt --verify

--channels/--heads/--trunk default to the process env
(SHRIMP_CHANNELS/SHRIMP_ATTENTION_HEADS/SHRIMP_TRUNK); if unset there, the
arch is inferred from the state dict itself.

--verify reloads the exported file and instantiates shrimp's model class to
prove the weights load (strict). This needs torch installed and the shrimp
package importable (run scripts/build_native.sh first, or add
packages/shrimp/python to PYTHONPATH).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# shrimp is imported from the source tree (never pip-installed; see README).
_HF_SRC = REPO / "packages" / "shrimp" / "python"
if _HF_SRC.is_dir() and str(_HF_SRC) not in sys.path:
    sys.path.insert(0, str(_HF_SRC))


def _load_payload(path: Path):
    import torch

    if not path.is_file():
        raise SystemExit(f"error: checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _extract_state_dict(payload) -> dict:
    """Pull the model weights out of a training or prefit checkpoint."""

    if not isinstance(payload, dict):
        raise SystemExit("error: checkpoint payload is not a dict")
    state = payload.get("model")
    if not isinstance(state, dict):
        raise SystemExit(
            "error: checkpoint has no 'model' state dict "
            f"(top-level keys: {sorted(payload.keys())})"
        )
    return state


def _epoch_of(payload) -> int | None:
    meta = payload.get("meta")
    if isinstance(meta, dict) and meta.get("epoch") is not None:
        return int(meta["epoch"])
    if payload.get("epoch") is not None:  # prefit shape
        return int(payload["epoch"])
    return None


def _run_of(payload) -> str | None:
    meta = payload.get("meta")
    if isinstance(meta, dict) and meta.get("run"):
        return str(meta["run"])
    return None


def _resolve_arch(state_dict: dict, args) -> dict:
    """Arch from explicit args, else env, else inferred from the state dict."""

    channels = args.channels or _env_int("SHRIMP_CHANNELS")
    heads = args.heads or _env_int("SHRIMP_ATTENTION_HEADS")
    trunk = args.trunk or os.environ.get("SHRIMP_TRUNK")

    if channels is None or heads is None or trunk is None:
        try:
            from shrimp.model import infer_net_kwargs_from_state_dict

            inferred = infer_net_kwargs_from_state_dict(state_dict)
        except Exception as exc:  # noqa: BLE001 - report cleanly
            raise SystemExit(
                "error: arch not fully specified (need channels/heads/trunk) and "
                f"could not infer from the state dict: {exc}\n"
                "Pass --channels/--heads/--trunk or set the SHRIMP_* env vars."
            )
        channels = channels or int(inferred["channels"])
        heads = heads or int(inferred["attention_heads"])
        trunk = trunk or str(inferred["trunk_layout"])

    return {"channels": int(channels), "attention_heads": int(heads), "trunk_layout": str(trunk)}


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else None


def export(args) -> Path:
    src = Path(args.checkpoint)
    payload = _load_payload(src)

    print(f"loaded {src}")
    print(f"  top-level keys: {sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__}")

    state_dict = _extract_state_dict(payload)
    epoch = _epoch_of(payload)
    run = args.run or _run_of(payload)
    arch = _resolve_arch(state_dict, args)

    print(f"  weights tensors: {len(state_dict)}")
    print(f"  epoch: {epoch}  run: {run}")
    print(f"  arch: {arch}")
    dropped = [k for k in ("optimizer", "scaler", "global_step") if k in payload]
    if dropped:
        print(f"  dropping training-only keys: {dropped}")

    out_meta = {
        "lineage": "shrimp",
        "epoch": epoch,
        "run": run,
        "arch": arch,
        "export": {
            "source_checkpoint": src.name,
            "export_date": args.export_date,
            "tool": "scripts/export_weights.py",
        },
    }
    out_payload = {"meta": out_meta, "model": state_dict}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(out_payload, out)
    print(f"wrote {out}")

    if args.verify:
        _verify(out, arch)
    return out


def _verify(path: Path, arch: dict) -> None:
    """Reload the export and instantiate ShrimpNet to prove the weights load."""

    import torch

    print(f"verifying {path} ...")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"]
    try:
        from shrimp.model import ShrimpNet
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "error: --verify needs the shrimp package importable "
            f"(and torch): {exc}\n"
            "Build it (scripts/build_native.sh) or add packages/shrimp/python "
            "to PYTHONPATH."
        )
    model = ShrimpNet(
        channels=arch["channels"],
        attention_heads=arch["attention_heads"],
        trunk_layout=arch["trunk_layout"],
    )
    model.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"verify OK: ShrimpNet loaded strict, {n_params:,} params")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", help="training or prefit checkpoint .pt to export")
    parser.add_argument("--out", required=True, help="output weights-only .pt path")
    parser.add_argument("--run", default=None, help="run name for metadata (default: from checkpoint meta)")
    parser.add_argument("--channels", type=int, default=None, help="override SHRIMP_CHANNELS")
    parser.add_argument("--heads", type=int, default=None, help="override SHRIMP_ATTENTION_HEADS")
    parser.add_argument("--trunk", default=None, help="override SHRIMP_TRUNK layout string")
    parser.add_argument(
        "--export-date",
        default=_dt.date.today().isoformat(),
        help="export date recorded in metadata (default: today)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="reload the export and instantiate ShrimpNet to prove it loads",
    )
    args = parser.parse_args(argv)
    export(args)


if __name__ == "__main__":
    main()
