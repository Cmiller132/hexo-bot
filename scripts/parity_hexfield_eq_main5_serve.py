#!/usr/bin/env python3
"""Parity gate for the behavior-preserving hexfield_eq XPU serve changes.

The hard gate compares baseline vs optimized evaluator reply bytes and a
fixed-seed, TSS-off MCTS search (the deterministic search anchor). An optional
live-TSS check reports, but does not hide, the async solver's documented
wall-clock scheduling variance.
"""

from __future__ import annotations

import argparse
import gc
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


def perturb_second_row_features(np, payload: dict, num_features: int) -> None:
    """Keep row geometry identical while proving feature work stays per-row."""
    offsets = payload["node_row_offsets"]
    feats = np.frombuffer(payload["node_feats"], dtype=np.float16).copy()
    index = int(offsets[1]) * int(num_features)
    feats[index] = np.float16(feats[index] + np.float16(0.5))
    payload["node_feats"] = feats.tobytes()


def reply_rows_differ(np, payload: dict, reply: dict) -> bool:
    """Return whether either decoded value or legal-prefix logits differ."""
    values = np.frombuffer(reply["values_bytes"], dtype=np.float32)
    if values.shape != (2,) or not np.isfinite(values).all():
        return False
    if values[0] != values[1]:
        return True
    counts = np.frombuffer(payload["legal_counts"], dtype=np.int32)
    if counts.shape != (2,) or counts[0] != counts[1]:
        return False
    count = int(counts[0])
    logits = np.frombuffer(reply["priors_logits_bytes"], dtype=np.float32)
    return (
        logits.shape == (2 * count,)
        and np.isfinite(logits).all()
        and not np.array_equal(logits[:count], logits[count:])
    )


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


def release_device_cache(torch, device) -> None:
    """Release prior-case allocator pressure on the 4 GB A310."""
    gc.collect()
    bench.sync_device(torch, device)
    module = getattr(torch, device.type, None)
    empty_cache = getattr(module, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()
    bench.sync_device(torch, device)


def check_pair_index_rewrite(torch, model, model_impl, device) -> bool:
    """Compare the in-place pair builder with its allocation-heavy formula."""
    coords = torch.tensor(
        [
            [[0, 0], [2, -3], [-7, 5], [11, 4]],
            [[-13, 9], [1, 1], [8, -12], [0, 0]],
        ],
        dtype=torch.long,
        device=device,
    )
    mask = torch.tensor(
        [[True, True, False, True], [True, False, True, True]],
        dtype=torch.bool,
        device=device,
    )
    b, n, _ = coords.shape
    dq = coords[:, None, :, 0] - coords[:, :, None, 0]
    dr = coords[:, None, :, 1] - coords[:, :, None, 1]
    m = model._cell_bias_M
    w = 2 * m + 1
    qi = dq.clamp(-m, m) + m
    ri = dr.clamp(-m, m) + m
    cell_idx = model._cell_bias_lut[(qi * w + ri).reshape(-1)].reshape(b, n, n)
    s = model_impl.NUM_TOKENS + n
    reference_pair = coords.new_full((b, s, s), model_impl.BIAS_TOKEN_TOKEN_ROW)
    reference_pair[:, : model_impl.NUM_TOKENS, model_impl.NUM_TOKENS :] = (
        model_impl.BIAS_TOKEN_CELL_ROW
    )
    reference_pair[:, model_impl.NUM_TOKENS :, : model_impl.NUM_TOKENS] = (
        model_impl.BIAS_CELL_TOKEN_ROW
    )
    reference_pair[:, model_impl.NUM_TOKENS :, model_impl.NUM_TOKENS :] = cell_idx
    reference_key_pad = torch.cat(
        [mask.new_ones(b, model_impl.NUM_TOKENS), mask], dim=1
    )
    original_mode = model_impl._XPU_LEAN_BIAS
    try:
        for lean_mode in (False, True):
            model_impl._XPU_LEAN_BIAS = lean_mode
            with torch.no_grad():
                actual_pair, actual_key_pad = model._build_pair(coords, mask)
            expected_dtype = (
                torch.int32
                if lean_mode and device.type == "xpu"
                else torch.long
            )
            if not (
                actual_pair.dtype == expected_dtype
                and torch.equal(actual_pair.to(torch.long), reference_pair)
                and torch.equal(actual_key_pad, reference_key_pad)
            ):
                return False
    finally:
        model_impl._XPU_LEAN_BIAS = original_mode
    return True


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
    bench.add_serve_opt_arguments(parser)
    parser.add_argument(
        "--optimized-pair-ceiling",
        type=float,
        default=None,
        help=(
            "optimized Python grouping ceiling; when above baseline, also "
            "checks a worst-size wide payload whose group boundaries differ"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--torch-threads", type=int, default=7)
    parser.add_argument(
        "--live-tss-check",
        action="store_true",
        help="also report live async-TSS repeat/optimized stability (advisory)",
    )
    args = parser.parse_args()
    if (
        args.optimized_pair_ceiling is not None
        and args.optimized_pair_ceiling <= 0
    ):
        parser.error("--optimized-pair-ceiling must be > 0")

    import numpy as np
    import torch

    torch.set_num_threads(args.torch_threads)
    bench.prime_checkpoint_env(torch, args.checkpoint, args.device, False)

    # Imports below happen only after the checkpoint architecture and materialized
    # attention baseline are frozen.
    from hexo_engine import api
    from hexo_engine.types import AxialCoord, PlacementAction
    from hexfield_eq import _rust
    from hexfield_eq import model as model_impl
    from hexfield_eq.config import build_divergence_overrides, parse_hexfield_config
    from hexfield_eq.eval_arena import _load_hexfield_net
    from hexfield_eq.geometry import unpack_action_id
    from hexfield_eq import inference as inference_impl
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
    bench.configure_serve_path("optimized", **bench.serve_opt_overrides(args))
    optimized = HexfieldEvaluator(model, device=args.device)
    optimized_lean_bias = os.environ.get("HEXFIELD_XPU_LEAN_BIAS") == "1"
    optimized_head_split = (
        os.environ.get("HEXFIELD_XPU_ATTN_HEAD_SPLIT") == "1"
    )
    optimized_ray_coeff_lut = (
        os.environ.get("HEXFIELD_XPU_RAY_COEFF_LUT") == "1"
    )
    baseline_pair_ceiling = float(inference_impl.PAIR_CEILING)
    optimized_pair_ceiling = (
        float(args.optimized_pair_ceiling)
        if args.optimized_pair_ceiling is not None
        else baseline_pair_ceiling
    )
    if optimized_pair_ceiling != baseline_pair_ceiling and optimized._rust_pack:
        parser.error(
            "--optimized-pair-ceiling comparison requires --rust-pack off; "
            "the Rust planner freezes its environment on first use"
        )
    print(
        f"torch={torch.__version__} device={optimized.device} visits={args.visits} "
        f"baseline(rust={baseline._rust_pack},defer={baseline._defer_decode},"
        f"host={baseline._host_legal_gather},cache={baseline._decode_cache}) "
        f"optimized(rust={optimized._rust_pack},defer={optimized._defer_decode},"
        f"host={optimized._host_legal_gather},cache={optimized._decode_cache}) "
        f"pair_ceiling={baseline_pair_ceiling:g}/{optimized_pair_ceiling:g} "
        f"bias_chunk="
        f"{model_impl._BIAS_GATHER_CHUNK}/"
        f"{model_impl._BIAS_GATHER_CHUNK_THRESHOLD} "
        f"bias_max_elems={model_impl._BIAS_GATHER_MAX_ELEMS} "
        f"lean={optimized_lean_bias} "
        f"attn_head_split={optimized_head_split} "
        f"ray_coeff_lut={optimized_ray_coeff_lut}"
    )

    failed = False
    pair_ok = check_pair_index_rewrite(
        torch, model, model_impl, optimized.device
    )
    print(
        f"{'PASS' if pair_ok else 'FAIL'} pair-index rewrite: "
        "in-place/reference tensors identical"
    )
    failed |= not pair_ok
    release_device_cache(torch, optimized.device)

    chunk_threshold = model_impl._BIAS_GATHER_CHUNK_THRESHOLD
    production_gather_max = model_impl._BIAS_GATHER_MAX_ELEMS
    unchunked_threshold = 1 << 60
    # Run two different-feature rows per board so the byte gate exercises a
    # real B=2 forward, not two interchangeable outputs. Wide baseline remains
    # safe through the legacy one-shot gather at B=2; optimized exercises the
    # lean per-head XPU gather above S=1024.
    for board_name, state in (("compact", compact), ("wide", wide)):
        payload = build_payload(np, _rust, [state, state], request_logits=True)
        perturb_second_row_features(np, payload, model_impl.NUM_FEATURES)
        model_impl._BIAS_GATHER_CHUNK_THRESHOLD = unchunked_threshold
        model_impl._XPU_LEAN_BIAS = False
        model_impl._XPU_ATTN_HEAD_SPLIT = False
        model_impl._XPU_RAY_COEFF_LUT = False
        base_reply = baseline.evaluate_payload(dict(payload))
        release_device_cache(torch, optimized.device)
        model_impl._BIAS_GATHER_CHUNK_THRESHOLD = chunk_threshold
        model_impl._XPU_LEAN_BIAS = optimized_lean_bias
        model_impl._XPU_ATTN_HEAD_SPLIT = optimized_head_split
        model_impl._XPU_RAY_COEFF_LUT = optimized_ray_coeff_lut
        # Force multiple nonzero-offset index_select(out=...) slices on the
        # wide B=2 fixture without allocating a risky production-sized parity
        # batch. The same loop uses the larger production launch cap in serve.
        parity_gather_max = (
            min(production_gather_max, 1_000_000)
            if board_name == "wide" and optimized_lean_bias
            else production_gather_max
        )
        model_impl._BIAS_GATHER_MAX_ELEMS = parity_gather_max
        try:
            opt_reply = optimized.evaluate_payload(dict(payload))
        finally:
            model_impl._BIAS_GATHER_MAX_ELEMS = production_gather_max
        exact_reply = (
            base_reply.keys() == opt_reply.keys()
            and all(base_reply[key] == opt_reply[key] for key in base_reply)
        )
        distinct_feature_rows = reply_rows_differ(np, payload, base_reply)
        deltas = max_reply_delta(np, base_reply, opt_reply)
        print(
            f"{'PASS' if exact_reply and distinct_feature_rows else 'FAIL'} "
            f"{board_name} evaluator reply bytes: "
            + ", ".join(f"{key}={value:.3g}" for key, value in deltas.items())
            + f"; feature_rows_distinct={distinct_feature_rows}"
            + f"; gather_max={parity_gather_max}"
        )
        failed |= not (exact_reply and distinct_feature_rows)
        del payload, base_reply, opt_reply
        release_device_cache(torch, optimized.device)

    if optimized_pair_ceiling > baseline_pair_ceiling:
        wide_nodes = int(_rust.featurize_states([wide])[0]["num_nodes"])
        wide_s = wide_nodes + model_impl.NUM_TOKENS
        optimized_rows = min(
            inference_impl.MAX_GROUP_ROWS,
            int(optimized_pair_ceiling // (wide_s * wide_s)),
        )
        baseline_rows = int(baseline_pair_ceiling // (wide_s * wide_s))
        if optimized_rows <= baseline_rows:
            print(
                "FAIL wide regroup evaluator bytes: ceilings do not produce "
                f"different groups at S={wide_s}"
            )
            failed = True
        else:
            regroup_payload = build_payload(
                np, _rust, [wide] * optimized_rows, request_logits=True
            )
            perturb_second_row_features(
                np, regroup_payload, model_impl.NUM_FEATURES
            )
            sizes = [wide_nodes] * optimized_rows
            inference_impl.PAIR_CEILING = baseline_pair_ceiling
            baseline_plan = inference_impl.plan_groups(sizes)
            model_impl._BIAS_GATHER_CHUNK_THRESHOLD = chunk_threshold
            model_impl._XPU_LEAN_BIAS = False
            model_impl._XPU_ATTN_HEAD_SPLIT = False
            model_impl._XPU_RAY_COEFF_LUT = False
            regroup_base = baseline.evaluate_payload(dict(regroup_payload))
            release_device_cache(torch, optimized.device)
            inference_impl.PAIR_CEILING = optimized_pair_ceiling
            optimized_plan = inference_impl.plan_groups(sizes)
            model_impl._XPU_LEAN_BIAS = optimized_lean_bias
            model_impl._XPU_ATTN_HEAD_SPLIT = optimized_head_split
            model_impl._XPU_RAY_COEFF_LUT = optimized_ray_coeff_lut
            regroup_opt = optimized.evaluate_payload(dict(regroup_payload))
            regroup_exact = (
                baseline_plan != optimized_plan
                and regroup_base.keys() == regroup_opt.keys()
                and all(
                    regroup_base[key] == regroup_opt[key]
                    for key in regroup_base
                )
            )
            regroup_deltas = max_reply_delta(np, regroup_base, regroup_opt)
            print(
                f"{'PASS' if regroup_exact else 'FAIL'} "
                f"wide regroup evaluator bytes B={optimized_rows},S={wide_s}: "
                + ", ".join(
                    f"{key}={value:.3g}"
                    for key, value in regroup_deltas.items()
                )
                + f"; groups={baseline_plan}->{optimized_plan}"
            )
            failed |= not regroup_exact
            del regroup_payload, regroup_base, regroup_opt
            release_device_cache(torch, optimized.device)

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
        # game_key participates in the Gumbel-root seed. Each call owns a fresh
        # session, so using the same key is both safe and required for parity.
        game_key = 9_000_000 + index * 10
        inference_impl.PAIR_CEILING = baseline_pair_ceiling
        model_impl._BIAS_GATHER_CHUNK_THRESHOLD = chunk_threshold
        model_impl._XPU_LEAN_BIAS = False
        model_impl._XPU_ATTN_HEAD_SPLIT = False
        model_impl._XPU_RAY_COEFF_LUT = False
        first = search_once(_rust, baseline, state, key=game_key, **common)
        release_device_cache(torch, optimized.device)
        inference_impl.PAIR_CEILING = optimized_pair_ceiling
        model_impl._BIAS_GATHER_CHUNK_THRESHOLD = chunk_threshold
        model_impl._XPU_LEAN_BIAS = optimized_lean_bias
        model_impl._XPU_ATTN_HEAD_SPLIT = optimized_head_split
        model_impl._XPU_RAY_COEFF_LUT = optimized_ray_coeff_lut
        second = search_once(_rust, optimized, state, key=game_key, **common)
        ok, detail = compare_search(np, first, second)
        print(f"{'PASS' if ok else 'FAIL'} {name} deterministic search: {detail}")
        failed |= not ok
        del first, second
        release_device_cache(torch, optimized.device)

    if args.live_tss_check:
        model_impl._BIAS_GATHER_CHUNK_THRESHOLD = chunk_threshold
        print(
            "\nLive async-TSS advisory (chunked XPU safety gather on all calls): "
            "tss_async.rs documents fixed-seed wall-clock dependence. A "
            "baseline-repeat mismatch is pre-existing scheduler variance, not "
            "an evaluator numeric regression."
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
            game_key = 9_100_000 + index * 10
            inference_impl.PAIR_CEILING = baseline_pair_ceiling
            model_impl._XPU_LEAN_BIAS = False
            model_impl._XPU_ATTN_HEAD_SPLIT = False
            model_impl._XPU_RAY_COEFF_LUT = False
            b1 = search_once(_rust, baseline, state, key=game_key, **common)
            release_device_cache(torch, optimized.device)
            inference_impl.PAIR_CEILING = baseline_pair_ceiling
            model_impl._XPU_LEAN_BIAS = False
            model_impl._XPU_ATTN_HEAD_SPLIT = False
            model_impl._XPU_RAY_COEFF_LUT = False
            b2 = search_once(_rust, baseline, state, key=game_key, **common)
            release_device_cache(torch, optimized.device)
            inference_impl.PAIR_CEILING = optimized_pair_ceiling
            model_impl._XPU_LEAN_BIAS = optimized_lean_bias
            model_impl._XPU_ATTN_HEAD_SPLIT = optimized_head_split
            model_impl._XPU_RAY_COEFF_LUT = optimized_ray_coeff_lut
            opt = search_once(_rust, optimized, state, key=game_key, **common)
            repeat_ok, repeat_detail = compare_search(np, b1, b2)
            opt_ok, opt_detail = compare_search(np, b1, opt)
            print(
                f"{name}: baseline_repeat={'stable' if repeat_ok else 'VARIED'} "
                f"({repeat_detail}); baseline_vs_opt="
                f"{'same' if opt_ok else 'VARIED'} ({opt_detail}); "
                f"elapsed={time.perf_counter() - started:.2f}s"
            )
            del b1, b2, opt
            release_device_cache(torch, optimized.device)

    print(
        "\nHARD PARITY GATE: "
        + ("PASS" if not failed else "FAIL")
        + " (exact evaluator bytes + exact deterministic action/visit policy)"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
