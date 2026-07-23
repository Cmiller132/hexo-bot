#!/usr/bin/env python3
"""Container benchmark for the hexfield_eq main_5 ep35 showcase path.

Only stdlib plus the application's existing torch/numpy/hexfield_eq stack are
used. Run this in a fresh process because architecture and FlexAttention gates
are frozen when hexfield_eq.model is first imported.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace


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
IMPORT_GATES = (
    "HEXFIELD_SERVE_FLEX",
    "HEXFIELD_FLEX_PAIR",
    "HEXFIELD_TRITON_CONV",
    "HEXFIELD_TRITON_ATTN",
    "HEXFIELD_TRITON_CONV_LN",
)


def csv_ints(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values or any(x < 1 for x in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def csv_names(raw: str) -> list[str]:
    allowed = {"live", "tss-off", "park-off", "leaves-off"}
    values = [x.strip() for x in raw.split(",") if x.strip()]
    bad = set(values) - allowed
    if not values or bad:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated {sorted(allowed)}; bad={sorted(bad)}"
        )
    return values


def env_value(value) -> str:
    if value is True:
        return "1"
    if value is False:
        return "0"
    return str(value)


def prime_checkpoint_env(torch, checkpoint: Path, device: str, xpu_flex: bool) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = payload.get("meta") if isinstance(payload, dict) else None
    if not isinstance(meta, dict):
        raise RuntimeError(f"checkpoint has no architecture metadata: {checkpoint}")
    missing = [
        key
        for key in ARCH_ENV
        if key not in ("cell_q", "ray_blockers") and key not in meta
    ]
    if missing:
        raise RuntimeError(f"checkpoint metadata missing {missing}")
    for key, name in ARCH_ENV.items():
        os.environ[name] = env_value(meta.get(key, True))

    # Establish an unambiguous import-time baseline in this fresh process.
    for name in IMPORT_GATES:
        os.environ[name] = "0"
    if device.split(":", 1)[0] == "xpu" and xpu_flex:
        os.environ["HEXFIELD_XPU_FLEX"] = "1"
        os.environ["HEXFIELD_SERVE_FLEX"] = "1"
        os.environ["HEXFIELD_FLEX_PAIR"] = "1"
    else:
        os.environ["HEXFIELD_XPU_FLEX"] = "0"
    return meta


def configure_serve_path(path: str) -> None:
    enabled = "1" if path == "optimized" else "0"
    os.environ["HEXFIELD_RUST_PACK"] = enabled
    os.environ["HEXFIELD_DEFER_DECODE"] = enabled
    os.environ["HEXFIELD_HOST_LEGAL_GATHER"] = enabled
    os.environ["HEXFIELD_DECODE_CACHE"] = enabled


def make_position(api, PlacementAction, AxialCoord, unpack_action_id, kind: str, plies: int):
    """Build a deterministic nonterminal legal position.

    Compact play samples the nearest legal quartile; wide play samples the
    farthest legal quartile. Retrying with another deterministic RNG stream
    avoids returning a terminal six-in-a-row position.
    """

    def distance(action_id: int) -> int:
        q, r = unpack_action_id(int(action_id))
        return max(abs(q), abs(r), abs(q + r))

    for attempt in range(64):
        state = api.new_game()
        actions: list[int] = []
        rng = random.Random(0xA310 + attempt * 1009 + (0 if kind == "compact" else 1))
        for _ in range(plies):
            legal = list(api.legal_action_ids(state))
            if not legal:
                break
            ordered = sorted(legal, key=lambda aid: (distance(aid), int(aid)))
            width = max(1, len(ordered) // 4)
            pool = ordered[:width] if kind == "compact" else ordered[-width:]
            aid = int(rng.choice(pool))
            q, r = unpack_action_id(aid)
            result = api.apply_action(
                state, PlacementAction(AxialCoord(q=int(q), r=int(r)))
            )
            actions.append(aid)
            if getattr(result, "terminal", None):
                break
        if len(actions) == plies and api.terminal(state) is None:
            return state, actions
    raise RuntimeError(f"could not construct nonterminal {kind} position at {plies} plies")


def case_settings(name: str, base: dict, tss_enabled: bool) -> tuple[dict, bool]:
    overrides = dict(base)
    enabled = bool(tss_enabled)
    if name == "tss-off":
        enabled = False
        overrides.update(
            tss_solver_mode=0,
            tss_solver_async=False,
            tss_solver_park=False,
            tss_solver_all_leaves=False,
            tss_interior_guard=False,
        )
    elif name == "park-off":
        overrides["tss_solver_park"] = False
    elif name == "leaves-off":
        overrides["tss_solver_all_leaves"] = False
    return overrides, enabled


def sync_device(torch, device) -> None:
    module = getattr(torch, device.type, None)
    sync = getattr(module, "synchronize", None)
    if callable(sync):
        sync()


def run_one(
    torch,
    rust,
    evaluator,
    state,
    *,
    visits: int,
    batch_size: int,
    case: str,
    base_overrides: dict,
    selfplay,
    seed: int,
    game_key: int,
) -> dict:
    from hexfield_eq.config import build_eval_search_kwargs

    overrides, tss_enabled = case_settings(
        case, base_overrides, selfplay.tss_enabled
    )
    kwargs = build_eval_search_kwargs(
        selfplay,
        visits=visits,
        virtual_batch_size=batch_size,
        active_root_limit=selfplay.active_root_limit,
    )
    kwargs["tss_enabled"] = tss_enabled
    session = rust.HexfieldMctsSession(max_states=65_536)
    sync_device(torch, evaluator.device)
    started = time.perf_counter()
    result = session.search(
        [game_key],
        (state,),
        seed=seed,
        evaluator=evaluator,
        move_temperatures=[0.0],
        divergence_overrides=overrides,
        **kwargs,
    )[0]
    sync_device(torch, evaluator.device)
    wall_s = time.perf_counter() - started
    session.discard(game_key)

    diag = result.get("diagnostics", {})
    ev = diag.get("evaluation", {})
    tss = diag.get("tss", {})
    eval_s = float(ev.get("evaluator_seconds", 0.0))
    encode_s = float(ev.get("encoding_seconds", 0.0))
    parse_s = float(ev.get("parse_seconds", 0.0))
    return {
        "wall_ms": wall_s * 1000.0,
        "eval_ms": eval_s * 1000.0,
        "encode_ms": encode_s * 1000.0,
        "parse_ms": parse_s * 1000.0,
        # For tss-off this is the measured MCTS/tree/control remainder. With
        # TSS on it also includes critical-path TSS coordination; park wait
        # sums below are overlapping leaf-time and must not be subtracted.
        "other_ms": max(0.0, (wall_s - eval_s - encode_s - parse_s) * 1000.0),
        "park_sum_ms": int(tss.get("park_wait_ms_sum", 0)),
        "park_max_ms": int(tss.get("park_wait_ms_max", 0)),
        "deep_calls": int(tss.get("deep_calls", 0)),
        "deep_nodes": int(tss.get("deep_nodes", 0)),
        "unique": int(ev.get("unique_states", 0)),
        "chunks": int(ev.get("evaluator_chunks", 0)),
        "action": int(result["action_id"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/models/hexfield_eq_main5_ep35_infer.pt"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("/app/configs/hexfield_eq_main_5.toml")
    )
    parser.add_argument("--device", default=os.environ.get("SHOWCASE_DEVICE", "xpu"))
    parser.add_argument("--serve-path", choices=("baseline", "optimized"), default="optimized")
    parser.add_argument(
        "--xpu-flex",
        choices=("off", "on"),
        default="off",
        help="experimental import-time FlexAttention probe; Triton stays off",
    )
    parser.add_argument("--visits", type=csv_ints, default=csv_ints("64,128,256,512"))
    parser.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("32"))
    parser.add_argument("--cases", type=csv_names, default=csv_names("live"))
    parser.add_argument("--compact-plies", type=int, default=18)
    parser.add_argument("--wide-plies", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=int(os.environ.get("SHOWCASE_TORCH_THREADS", "7") or "7"),
    )
    parser.add_argument("--rayon-threads", type=int, default=0)
    args = parser.parse_args()

    if args.rayon_threads:
        os.environ["RAYON_NUM_THREADS"] = str(args.rayon_threads)
    try:
        import torch
    except ImportError as exc:
        parser.error(f"torch is required in the showcase container: {exc}")
    torch.set_num_threads(args.torch_threads)
    meta = prime_checkpoint_env(
        torch, args.checkpoint, args.device, args.xpu_flex == "on"
    )
    configure_serve_path(args.serve_path)

    # Every hexfield_eq import is deliberately below checkpoint env priming.
    import hexo_engine
    from hexo_engine import api
    from hexo_engine.types import AxialCoord, PlacementAction
    from hexfield_eq import _rust
    from hexfield_eq.config import (
        build_divergence_overrides,
        parse_hexfield_config,
    )
    from hexfield_eq.eval_arena import _load_hexfield_net
    from hexfield_eq.geometry import unpack_action_id
    from hexfield_eq.inference import build_serve_evaluator

    del hexo_engine  # imported only to give a clearer missing-package failure
    with args.config.open("rb") as fh:
        raw = tomllib.load(fh)["model"]["config"]
    cfg = parse_hexfield_config(
        {
            "device": args.device,
            "selfplay": raw.get("selfplay", {}),
            "multi_stage_eval": raw.get("multi_stage_eval", {}),
        }
    )
    model = _load_hexfield_net(args.checkpoint)
    evaluator = build_serve_evaluator(model, cfg, role="eval")
    device = evaluator.device

    if device.type == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("SHOWCASE_DEVICE=xpu but torch.xpu is unavailable")
    compact = make_position(
        api, PlacementAction, AxialCoord, unpack_action_id, "compact", args.compact_plies
    )
    wide = make_position(
        api, PlacementAction, AxialCoord, unpack_action_id, "wide", args.wide_plies
    )
    positions = {"compact": compact, "wide": wide}
    supports = {
        name: int(_rust.featurize_states([state])[0]["num_nodes"])
        for name, (state, _) in positions.items()
    }

    print(
        f"torch={torch.__version__} device={device} threads={torch.get_num_threads()} "
        f"serve_path={args.serve_path} xpu_flex={args.xpu_flex}"
    )
    print(
        "arch="
        f"channels={meta.get('channels')} trunk={meta.get('trunk_layout')} "
        f"feature_v={meta.get('feature_version')} raytap={meta.get('raytap')} "
        f"rust_pack={evaluator._rust_pack} defer={evaluator._defer_decode} "
        f"host_gather={evaluator._host_legal_gather}"
    )
    for name, (_, actions) in positions.items():
        coords = [unpack_action_id(aid) for aid in actions]
        extent = (
            min(q for q, _ in coords),
            max(q for q, _ in coords),
            min(r for _, r in coords),
            max(r for _, r in coords),
        )
        print(
            f"{name}: plies={len(actions)} support={supports[name]} "
            f"extent=q[{extent[0]},{extent[1]}] r[{extent[2]},{extent[3]}] "
            f"actions={coords}"
        )

    # Warm the model/backend and the Rust call path without deep TSS.
    run_one(
        torch,
        _rust,
        evaluator,
        compact[0],
        visits=16,
        batch_size=args.batch_sizes[0],
        case="tss-off",
        base_overrides=build_divergence_overrides(cfg.selfplay),
        selfplay=cfg.selfplay,
        seed=args.seed - 1,
        game_key=8_000_000,
    )

    header = (
        "board    S    case       sims vb  wall_ms  eval_ms enc_ms parse "
        "other_ms park_sum/max deep(calls/nodes) avgB action"
    )
    print("\n" + header)
    print("-" * len(header))
    base_overrides = build_divergence_overrides(cfg.selfplay)
    key = 8_100_000
    failures = 0
    for board, (state, _) in positions.items():
        for case in args.cases:
            for batch_size in args.batch_sizes:
                for visits in args.visits:
                    key += 1
                    try:
                        row = run_one(
                            torch,
                            _rust,
                            evaluator,
                            state,
                            visits=visits,
                            batch_size=batch_size,
                            case=case,
                            base_overrides=base_overrides,
                            selfplay=cfg.selfplay,
                            seed=args.seed,
                            game_key=key,
                        )
                    except Exception as exc:
                        failures += 1
                        print(
                            f"{board:<8} {supports[board]:>4} {case:<10} "
                            f"{visits:>4} {batch_size:>2} ERROR "
                            f"{type(exc).__name__}: {exc}"
                        )
                        continue
                    avg_batch = row["unique"] / max(1, row["chunks"])
                    print(
                        f"{board:<8} {supports[board]:>4} {case:<10} "
                        f"{visits:>4} {batch_size:>2} "
                        f"{row['wall_ms']:>8.1f} {row['eval_ms']:>8.1f} "
                        f"{row['encode_ms']:>6.1f} {row['parse_ms']:>5.1f} "
                        f"{row['other_ms']:>8.1f} "
                        f"{row['park_sum_ms']:>6}/{row['park_max_ms']:<4} "
                        f"{row['deep_calls']:>4}/{row['deep_nodes']:<7} "
                        f"{avg_batch:>4.1f} {row['action']}"
                    )

    print(
        "\nNotes: eval_ms is the measured evaluator critical path (pack + XPU "
        "forward + decode). other_ms is wall - eval - encode - parse; with "
        "tss-off it is MCTS/tree/control overhead. park_sum is overlapping "
        "per-leaf wait and can exceed wall; use live-vs-park-off wall deltas "
        "for causal park cost. Toggle cases change search behavior and are "
        "diagnostic only."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
