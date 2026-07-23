#!/usr/bin/env python3
"""Parity gate for the behavior-preserving hexfield_eq XPU serve changes.

The hard gate compares baseline vs optimized evaluator reply bytes and a
fixed-seed, TSS-off MCTS search (the deterministic search anchor). An optional
live-TSS check reports, but does not hide, the async solver's documented
wall-clock scheduling variance.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tomllib
from pathlib import Path

import bench_hexfield_eq_main5_serve as bench


NBR_SENTINEL = 0xFFFF


def build_payload(np, rust, states: list, *, request_logits: bool = True) -> dict:
    rows = rust.featurize_states(states)
    order = sorted(range(len(states)), key=lambda i: (-int(rows[i]["num_nodes"]), i))
    feats, qr, nbr, raylen = [], [], [], []
    offsets, legal = [0], []
    for i in order:
        row = rows[i]
        n = int(row["num_nodes"])
        feats.append(np.frombuffer(row["feats"], dtype=np.float32).astype(np.float16))
        qr.append(np.frombuffer(row["coords"], dtype=np.int16))
        nrow = np.frombuffer(row["nbr"], dtype=np.int32)
        nbr.append(np.where(nrow < 0, NBR_SENTINEL, nrow).astype(np.uint16))
        raylen.append(np.frombuffer(row["raylen"], dtype=np.uint8))
        offsets.append(offsets[-1] + n)
        legal.append(int(row["legal_count"]))
    return {
        "abi": 1,
        "shape": (len(order), offsets[-1]),
        "node_feats": np.concatenate(feats).tobytes(),
        "node_qr": np.concatenate(qr).tobytes(),
        "nbr": np.concatenate(nbr).tobytes(),
        "raylen": np.concatenate(raylen).tobytes(),
        "node_row_offsets": offsets,
        "legal_counts": np.asarray(legal, dtype=np.int32).tobytes(),
        "request_moves_left": True,
        "request_logits": request_logits,
    }


def max_reply_delta(np, left: dict, right: dict) -> dict[str, float]:
    result = {}
    for key in sorted(left.keys() | right.keys()):
        if key not in left or key not in right:
            result[key] = float("inf")
            continue
        a = np.frombuffer(left[key], dtype=np.float32)
        b = np.frombuffer(right[key], dtype=np.float32)
        result[key] = (
            float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else
            (0.0 if a.shape == b.shape else float("inf"))
        )
    return result


def deterministic_overrides(base: dict) -> dict:
    out = dict(base)
    out.update(
        tss_solver_mode=0,
        tss_solver_async=False,
        tss_solver_park=False,
        tss_solver_all_leaves=False,
        tss_interior_guard=False,
    )
    return out


def search_once(
    rust,
    evaluator,
    state,
    *,
    selfplay,
    overrides: dict,
    visits: int,
    seed: int,
    key: int,
    tss_enabled: bool,
    virtual_batch_size: int,
):
    from hexfield_eq.config import build_eval_search_kwargs

    session = rust.HexfieldMctsSession(max_states=65_536)
    kwargs = build_eval_search_kwargs(
        selfplay,
        visits=visits,
        virtual_batch_size=virtual_batch_size,
        active_root_limit=selfplay.active_root_limit,
    )
    kwargs["tss_enabled"] = tss_enabled
    result = session.search(
        [key],
        (state,),
        seed=seed,
        evaluator=evaluator,
        move_temperatures=[0.0],
        divergence_overrides=overrides,
        **kwargs,
    )[0]
    session.discard(key)
    return result


def policy_summary(np, result: dict) -> dict:
    ids = np.frombuffer(result["visit_policy_action_ids_bytes"], dtype=np.uint32)
    weights = np.frombuffer(result["visit_policy_weights_bytes"], dtype=np.float32)
    qs = np.frombuffer(result["visit_policy_q_bytes"], dtype=np.float32)
    return {
        "action": int(result["action_id"]),
        "ids": ids,
        "weights": weights,
        "q": qs,
        "root_value": float(result["root_value"]),
    }


def compare_search(np, left: dict, right: dict) -> tuple[bool, str]:
    a, b = policy_summary(np, left), policy_summary(np, right)
    ids_equal = np.array_equal(a["ids"], b["ids"])
    weights_equal = np.array_equal(a["weights"], b["weights"])
    q_delta = (
        float(np.max(np.abs(a["q"] - b["q"])))
        if ids_equal and a["q"].shape == b["q"].shape and a["q"].size
        else float("inf")
    )
    root_delta = abs(a["root_value"] - b["root_value"])
    ok = (
        a["action"] == b["action"]
        and ids_equal
        and weights_equal
        and q_delta == 0.0
        and root_delta == 0.0
    )
    return ok, (
        f"action={a['action']}/{b['action']} ids_equal={ids_equal} "
        f"visits_equal={weights_equal} max_q_delta={q_delta:.3g} "
        f"root_delta={root_delta:.3g}"
    )


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
    parser.add_argument("--visits", type=int, default=128)
    parser.add_argument("--virtual-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--torch-threads", type=int, default=7)
    parser.add_argument(
        "--live-tss-check",
        action="store_true",
        help="also report live async-TSS repeat/optimized stability (advisory)",
    )
    args = parser.parse_args()

    import numpy as np
    import torch

    torch.set_num_threads(args.torch_threads)
    bench.prime_checkpoint_env(torch, args.checkpoint, args.device, False)

    # Imports below happen only after the checkpoint architecture and materialized
    # attention baseline are frozen.
    from hexo_engine import api
    from hexo_engine.types import AxialCoord, PlacementAction
    from hexfield_eq import _rust
    from hexfield_eq.config import build_divergence_overrides, parse_hexfield_config
    from hexfield_eq.eval_arena import _load_hexfield_net
    from hexfield_eq.geometry import unpack_action_id
    from hexfield_eq.inference import HexfieldEvaluator

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
    compact = bench.make_position(
        api, PlacementAction, AxialCoord, unpack_action_id, "compact", 18
    )[0]
    wide = bench.make_position(
        api, PlacementAction, AxialCoord, unpack_action_id, "wide", 18
    )[0]

    # Construct both evaluator modes over the same fp32 model. XPU does not
    # deepcopy/cast the model, so weights and forward arithmetic are identical.
    bench.configure_serve_path("baseline")
    baseline = HexfieldEvaluator(model, device=args.device)
    bench.configure_serve_path("optimized")
    optimized = HexfieldEvaluator(model, device=args.device)
    print(
        f"torch={torch.__version__} device={optimized.device} visits={args.visits} "
        f"baseline(rust={baseline._rust_pack},defer={baseline._defer_decode},"
        f"host={baseline._host_legal_gather}) "
        f"optimized(rust={optimized._rust_pack},defer={optimized._defer_decode},"
        f"host={optimized._host_legal_gather})"
    )

    failed = False
    payload = build_payload(np, _rust, [compact, wide], request_logits=True)
    base_reply = baseline.evaluate_payload(dict(payload))
    opt_reply = optimized.evaluate_payload(dict(payload))
    exact_reply = (
        base_reply.keys() == opt_reply.keys()
        and all(base_reply[key] == opt_reply[key] for key in base_reply)
    )
    deltas = max_reply_delta(np, base_reply, opt_reply)
    print(
        f"{'PASS' if exact_reply else 'FAIL'} evaluator reply bytes: "
        + ", ".join(f"{key}={value:.3g}" for key, value in deltas.items())
    )
    failed |= not exact_reply

    base_overrides = build_divergence_overrides(cfg.selfplay)
    det_overrides = deterministic_overrides(base_overrides)
    for index, (name, state) in enumerate((("compact", compact), ("wide", wide))):
        common = dict(
            selfplay=cfg.selfplay,
            overrides=det_overrides,
            visits=args.visits,
            seed=args.seed,
            tss_enabled=False,
            virtual_batch_size=args.virtual_batch_size,
        )
        first = search_once(
            _rust, baseline, state, key=9_000_000 + index * 10, **common
        )
        second = search_once(
            _rust, optimized, state, key=9_000_001 + index * 10, **common
        )
        ok, detail = compare_search(np, first, second)
        print(f"{'PASS' if ok else 'FAIL'} {name} deterministic search: {detail}")
        failed |= not ok

    if args.live_tss_check:
        print(
            "\nLive async-TSS advisory: tss_async.rs documents fixed-seed "
            "wall-clock dependence. A baseline-repeat mismatch is pre-existing "
            "scheduler variance, not an evaluator numeric regression."
        )
        for index, (name, state) in enumerate((("compact", compact), ("wide", wide))):
            common = dict(
                selfplay=cfg.selfplay,
                overrides=base_overrides,
                visits=args.visits,
                seed=args.seed,
                tss_enabled=cfg.selfplay.tss_enabled,
                virtual_batch_size=args.virtual_batch_size,
            )
            started = time.perf_counter()
            b1 = search_once(
                _rust, baseline, state, key=9_100_000 + index * 10, **common
            )
            b2 = search_once(
                _rust, baseline, state, key=9_100_001 + index * 10, **common
            )
            opt = search_once(
                _rust, optimized, state, key=9_100_002 + index * 10, **common
            )
            repeat_ok, repeat_detail = compare_search(np, b1, b2)
            opt_ok, opt_detail = compare_search(np, b1, opt)
            print(
                f"{name}: baseline_repeat={'stable' if repeat_ok else 'VARIED'} "
                f"({repeat_detail}); baseline_vs_opt="
                f"{'same' if opt_ok else 'VARIED'} ({opt_detail}); "
                f"elapsed={time.perf_counter() - started:.2f}s"
            )

    print(
        "\nHARD PARITY GATE: "
        + ("PASS" if not failed else "FAIL")
        + " (exact evaluator bytes + exact deterministic action/visit policy)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
