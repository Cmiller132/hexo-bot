#!/usr/bin/env python3
"""Audit how often the SHOWCASE's played move differs from what the live
overlay draws, and attribute each divergence to a cause.

The overlay heats `visit_policy_weights_bytes` (the exported visit-share
distribution). Selection at temperature 0 does NOT rank on that array: it
applies `tactical_guard_weights`, then LCB-of-Q over cumulative root stats,
then a TSS root certificate override. This script measures the resulting
mismatch rate on self-play games driven exactly the way the showcase drives
its bot, and A/B's the LCB arm to attribute it.

Turn structure matters. A Hexo turn is 1-2 stones, and `bot_turn` calls
`search_one` once per stone with the SAME game_key, so the retained tree is
reused on a turn's second stone (root advances one ply, next search is one ply
later) but NOT across turns (the opponent's reply moves the root out from under
it). Self-play here uses one game_key PER SIDE to reproduce that exactly; a
single shared key would reuse on every move and overstate warmth.

Read-only: no server file is touched and no live game is involved. Run in a
fresh process — arch and FlexAttention gates freeze at first hexfield_eq
import.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tomllib
from collections import Counter
from pathlib import Path

# Reuse the bench script's checkpoint/env priming. It sits next to this file in
# the repo; the showcase container's rootfs is read-only, so this script is
# normally piped in over stdin, where `__file__` does not exist.
_here = str(Path(__file__).resolve().parent) if "__file__" in globals() else None
for _candidate in (_here, "/app/scripts"):
    if _candidate and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from bench_hexfield_eq_main5_serve import (  # noqa: E402
    configure_serve_path,
    prime_checkpoint_env,
)


def summarize(rows: list[dict], label: str) -> dict:
    """Reduce per-move records to the rates we actually want to report."""
    total = len(rows)
    if not total:
        return {"label": label, "moves": 0}
    snaps = [r for r in rows if not r["played_is_visual_max"]]
    faint = [r for r in snaps if r["played_weight"] < 0.05]
    causes = Counter(r["cause"] for r in snaps)
    ratios = [r["played_weight"] / r["max_weight"] for r in snaps if r["max_weight"] > 0]
    return {
        "label": label,
        "moves": total,
        "snap_moves": len(snaps),
        "snap_rate": len(snaps) / total,
        # A snap onto a cell carrying <5% of the visit mass is the one a viewer
        # reads as "it played somewhere that was not even highlighted".
        "faint_snap_moves": len(faint),
        "faint_snap_rate": len(faint) / total,
        "cause_breakdown": dict(causes),
        "median_played_over_max": (statistics.median(ratios) if ratios else None),
        "selection_labels": dict(Counter(r["action_selection"] for r in rows)),
        "lcb_override_rate": sum(r["lcb_override"] for r in rows) / total,
        "reused_tree_moves": sum(r["tree_reused"] for r in rows),
        "snap_rate_on_reused": (
            sum(1 for r in rows if r["tree_reused"] and not r["played_is_visual_max"])
            / max(1, sum(r["tree_reused"] for r in rows))
        ),
        "snap_rate_on_fresh": (
            sum(1 for r in rows if not r["tree_reused"] and not r["played_is_visual_max"])
            / max(1, sum(not r["tree_reused"] for r in rows))
        ),
    }


def play_games(
    *,
    engine,
    api,
    rust,
    evaluator,
    selfplay,
    overrides: dict,
    visits: int,
    games: int,
    max_plies: int,
    opening_plies: int,
    opening_temperature: float,
    seed: int,
    numpy,
) -> list[dict]:
    from hexo_engine.types import PlacementAction, unpack_coord_id
    from hexfield_eq.config import build_eval_search_kwargs

    kwargs = build_eval_search_kwargs(
        selfplay,
        visits=int(visits),
        virtual_batch_size=32,
        active_root_limit=selfplay.active_root_limit,
    )
    session = rust.HexfieldMctsSession(max_states=65_536)
    rows: list[dict] = []

    for game in range(games):
        state = api.new_game()
        # One key per side, exactly as the showcase does for its single bot:
        # a side's tree survives only across the stones of its own turn.
        # A Hexo turn is 1-2 stones, so the side tag must come from the engine,
        # never from ply parity.
        keys: dict[object, int] = {}
        last_search_ply: dict[object, int | None] = {}
        ply = 0
        while ply < max_plies and engine.terminal(state) is None:
            side = engine.current_player(state)
            if side not in keys:
                keys[side] = 9_400_000 + game * 4 + len(keys)
                last_search_ply[side] = None
            temperature = (
                opening_temperature if ply < opening_plies and opening_temperature > 0 else 0.0
            )
            result = session.search(
                [keys[side]],
                (state,),
                seed=(seed * 5003 + ply * 7919 + game * 104729) & ((1 << 63) - 1),
                evaluator=evaluator,
                move_temperatures=[float(temperature)],
                divergence_overrides=overrides,
                **kwargs,
            )[0]

            ids = numpy.frombuffer(
                result["visit_policy_action_ids_bytes"], dtype=numpy.uint32
            )
            weights = numpy.frombuffer(
                result["visit_policy_weights_bytes"], dtype=numpy.float32
            )
            action_id = int(result["action_id"])
            selection = str(result.get("action_selection", ""))
            lcb_override = bool(result.get("lcb_override", False))

            if len(weights):
                max_idx = int(numpy.argmax(weights))
                max_weight = float(weights[max_idx])
                visual_max_id = int(ids[max_idx])
            else:
                max_weight, visual_max_id = 0.0, -1
            match = numpy.nonzero(ids == numpy.uint32(action_id))[0]
            played_weight = float(weights[match[0]]) if len(match) else 0.0
            played_present = bool(len(match))

            if selection == "tss_deep_root_win":
                cause = "tss_certificate"
            elif temperature > 0.0:
                cause = "opening_temperature_sample"
            elif lcb_override:
                cause = "lcb_of_q_override"
            else:
                cause = "tactical_guard_or_other"

            reused = last_search_ply[side] == ply - 1
            rows.append(
                {
                    "game": game,
                    "ply": ply,
                    "tree_reused": reused,
                    "temperature": temperature,
                    "action_selection": selection,
                    "lcb_override": lcb_override,
                    "play_pruned": bool(result.get("play_pruned", False)),
                    "play_winner": bool(result.get("play_winner", False)),
                    "policy_entries": int(len(ids)),
                    "played_weight": played_weight,
                    "played_present_in_overlay": played_present,
                    "max_weight": max_weight,
                    "played_is_visual_max": (action_id == visual_max_id),
                    "cause": cause,
                }
            )
            last_search_ply[side] = ply

            engine.apply_action(state, PlacementAction(unpack_coord_id(action_id)))
            ply += 1
        for key in keys.values():
            session.discard(key)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("/models/hexfield_eq_main5_ep75_infer.pt")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("/app/configs/hexfield_eq_main_5.toml")
    )
    parser.add_argument("--device", default=os.environ.get("SHOWCASE_DEVICE", "xpu"))
    parser.add_argument("--visits", type=int, default=512)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=60)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--opening-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--torch-threads", type=int, default=7)
    parser.add_argument(
        "--arms",
        default="live,lcb-off,tss-off",
        help="comma-separated: live, lcb-off, tss-off, guard-off",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.torch_threads)
    prime_checkpoint_env(torch, args.checkpoint, args.device, False)
    configure_serve_path("optimized")

    import numpy
    import hexo_engine as engine
    from hexo_engine import api
    from hexfield_eq import _rust
    from hexfield_eq.config import build_divergence_overrides, parse_hexfield_config
    from hexfield_eq.eval_arena import _load_hexfield_net
    from hexfield_eq.inference import build_serve_evaluator

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
    base = build_divergence_overrides(cfg.selfplay)
    # The showcase's own serving overrides (see hexfield_eq_family.build_profile).
    base["ml_final_pick"] = False

    print(
        f"device={evaluator.device} visits={args.visits} games={args.games} "
        f"opening_plies={args.opening_plies}@T={args.opening_temperature} "
        f"lcb_z={base.get('lcb_z', 1.6)} ckpt={args.checkpoint.name}"
    )

    summaries = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        overrides = dict(base)
        if arm == "lcb-off":
            overrides["lcb_move_selection"] = False
        elif arm == "tss-off":
            overrides["tss_solver_root_guard"] = False
        rows = play_games(
            engine=engine,
            api=api,
            rust=_rust,
            evaluator=evaluator,
            selfplay=cfg.selfplay,
            overrides=overrides,
            visits=args.visits,
            games=args.games,
            max_plies=args.max_plies,
            opening_plies=args.opening_plies,
            opening_temperature=args.opening_temperature,
            seed=args.seed,
            numpy=numpy,
        )
        summary = summarize(rows, arm)
        summaries.append({"summary": summary, "rows": rows})
        print("\n=== arm:", arm, "===")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    if args.json_out:
        args.json_out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
