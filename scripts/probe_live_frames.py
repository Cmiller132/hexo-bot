#!/usr/bin/env python3
"""Capture real live-search telemetry frames and report what the overlay would
actually paint on each one.

The claim originally under test: on the final frame the browser filtered the
displayed cells down to `survivors` (plus the played move), and by then
sequential halving has reduced `survivors` to one or two actions -- so the frame
a viewer reads as "search complete" showed ~2 cells rather than the root's visit
distribution. Measured 2026-07-24: 16 exported, 2 painted.

This replays BOTH the old and the new client rules (apps/showcase/web/app.js,
livePolicyRows) against genuine frames instead of reasoning about them, so the
fix is verified against the engine rather than against the diff. It also reports
the selection verdict, which is the other half of the problem: the played move
is not always the cell the drawn distribution favours.

Read-only; pipe over stdin because the showcase container's rootfs is read-only.
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path

_here = str(Path(__file__).resolve().parent) if "__file__" in globals() else None
for _candidate in (_here, "/app/scripts"):
    if _candidate and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from bench_hexfield_eq_main5_serve import (  # noqa: E402
    configure_serve_path,
    prime_checkpoint_env,
)


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
    parser.add_argument("--plies", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(7)
    prime_checkpoint_env(torch, args.checkpoint, args.device, False)
    configure_serve_path("optimized")

    import numpy
    import hexo_engine as engine
    from hexo_engine import api
    from hexo_engine.types import PlacementAction, unpack_coord_id
    from hexfield_eq import _rust
    from hexfield_eq.config import (
        build_divergence_overrides,
        build_eval_search_kwargs,
        parse_hexfield_config,
    )
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
    overrides = build_divergence_overrides(cfg.selfplay)
    overrides["ml_final_pick"] = False
    kwargs = build_eval_search_kwargs(
        cfg.selfplay, visits=args.visits, virtual_batch_size=32,
        active_root_limit=cfg.selfplay.active_root_limit,
    )

    session = _rust.HexfieldMctsSession(max_states=65_536)
    state = api.new_game()
    print(f"device={evaluator.device} visits={args.visits}")
    print("\nply phase        round  policy  surv  OLD_painted  NEW_painted  note")
    print("-" * 92)
    regressions = 0

    for ply in range(args.plies):
        if engine.terminal(state) is not None:
            break
        frames: list[dict] = []

        def capture(event: dict) -> None:
            ids = numpy.frombuffer(
                event.get("policy_action_ids_bytes", b""), dtype=numpy.uint32
            )
            weights = numpy.frombuffer(
                event.get("policy_weights_bytes", b""), dtype=numpy.float32
            )
            surv = numpy.frombuffer(
                event.get("survivor_action_ids_bytes", b""), dtype=numpy.uint32
            )
            frames.append(
                {
                    "phase": str(event.get("phase", "?")),
                    "round": int(event.get("round", 0)),
                    "policy": [int(x) for x in ids],
                    "weights": [float(w) for w in weights],
                    "survivors": set(int(x) for x in surv),
                    "action_id": event.get("action_id"),
                    "count": event.get("policy_count"),
                }
            )

        kwargs["telemetry_callback"] = capture
        result = session.search(
            [7_700_001],
            (state,),
            seed=(args.seed + ply * 7919),
            evaluator=evaluator,
            move_temperatures=[0.0],
            divergence_overrides=overrides,
            **kwargs,
        )[0]
        played = int(result["action_id"])

        for frame in frames:
            cells = frame["policy"]
            weights = frame["weights"]
            keep = frame["survivors"]
            act = frame["action_id"]
            note = ""

            # OLD rule: every frame after the raw start was narrowed to
            # `survivors` plus the played move.
            if frame["phase"] == "start":
                old = set(cells)
            else:
                old = {c for c in cells if c in keep or c == act}

            # NEW rule: nothing is discarded. Round frames dim the candidates
            # the halving just cut; the complete frame is drawn whole. Only a
            # zero-weight cell goes unpainted (a certificate move is appended to
            # the export with weight 0.0 and is marked by the ring instead).
            new = {c for c, w in zip(cells, weights) if w > 0.0}

            if frame["phase"] == "complete":
                if len(new) < len(cells) - 1:
                    note = "ZERO-WEIGHT CELLS?"
                if played not in cells:
                    note = "PLAYED MISSING FROM EXPORT"
                    regressions += 1
                else:
                    top = cells[int(numpy.argmax(weights))] if len(weights) else -1
                    if top != played:
                        note = (
                            f"played != visit leader "
                            f"(lcb_override={bool(result.get('lcb_override'))}, "
                            f"{result.get('action_selection')})"
                        )
                if len(new) <= 2 < len(cells):
                    regressions += 1
            if frame["count"] is not None and int(frame["count"]) != len(cells):
                note = "COUNT MISMATCH — decoder would drop this frame"
                regressions += 1

            print(
                f"{ply:>3} {frame['phase']:<12} {frame['round']:>5} "
                f"{len(cells):>7} {len(keep):>5} {len(old):>12} {len(new):>12}  {note}"
            )

        engine.apply_action(state, PlacementAction(unpack_coord_id(played)))
    session.discard(7_700_001)
    print(f"\nregressions: {regressions}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
