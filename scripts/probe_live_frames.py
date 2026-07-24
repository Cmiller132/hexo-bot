#!/usr/bin/env python3
"""Capture real live-search telemetry frames and report what the overlay would
actually paint on each one.

Two claims have been under test here, both replayed against genuine frames
rather than reasoned about, so each fix is verified against the engine and not
against its own diff.

1. Coverage. The browser used to filter every frame after the raw start down to
   `survivors` plus the played move, and by then sequential halving has reduced
   `survivors` to one or two actions -- so the frame a viewer reads as "search
   complete" showed ~2 cells rather than the root's visit distribution.
   Measured 2026-07-24: 16 exported, 2 painted. Fixed by not discarding.

2. Continuity. Halving publishes a SHRINKING list, then republishes the whole
   candidate set on the completion frame, so painting each frame as the whole
   truth made cells blink out mid-search and blink back at the end -- reported
   as flicker. The old column here is the FIXED-COVERAGE client, i.e. the one
   that flickered: it paints exactly what its frame carries. The new column
   holds eliminated candidates in a per-stone model instead of dropping them,
   so it must score zero blink-out/blink-back cells.

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
    old_revivals = 0
    new_revivals = 0

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

        # One engine `start` phase expands into TWO client frames -- the
        # board-wide prior map and the candidate set -- and the client's model
        # is scoped to that split, so the replay has to expand it the same way
        # or the narrowing would be scored as cells disappearing.
        client: list[dict] = []
        for frame in frames:
            if frame["phase"] == "start":
                client.append({**frame, "scope": "board"})
                keep = frame["survivors"]
                idx = [i for i, c in enumerate(frame["policy"]) if c in keep]
                client.append({
                    **frame, "scope": "candidates",
                    "policy": [frame["policy"][i] for i in idx],
                    "weights": [frame["weights"][i] for i in idx],
                })
            else:
                client.append({**frame, "scope": "candidates"})

        old_seen: set[int] = set()
        old_live: set[int] = set()
        new_seen: set[int] = set()
        new_live: set[int] = set()
        scope = ""
        model: dict[int, float] = {}

        for frame in client:
            cells = frame["policy"]
            weights = frame["weights"]
            keep = frame["survivors"]
            act = frame["action_id"]
            note = ""

            # OLD rule: whatever the frame itself carried was the whole board
            # and nothing was held, so a candidate the halving stopped ranking
            # was deleted and then rebuilt when the completion frame
            # republished it. This is the version that flickered.
            old = {c for c, w in zip(cells, weights) if w > 0.0}

            # NEW rule: the layer is a model, not a repaint. A frame's own cells
            # refresh; cells the model already holds and the frame omits are
            # held (dimmed) rather than deleted. Only a zero-weight cell goes
            # unpainted -- a certificate move is appended to the export with
            # weight 0.0 and is marked by the ring instead. A scope change is
            # the one place cells are dropped.
            if frame["scope"] != scope:
                scope, model = frame["scope"], {}
            model.update(zip(cells, weights))
            new = {c for c, w in model.items() if w > 0.0}

            # A revival is a cell painted now that was painted earlier in this
            # stone but NOT on the frame just before -- it blinked out and back.
            if frame["scope"] == "board":   # first frame of a stone: fresh slate
                old_seen, old_live = set(), set()
                new_seen, new_live = set(), set()
            old_revivals += len(old & (old_seen - old_live))
            new_revivals += len(new & (new_seen - new_live))
            old_seen |= old
            new_seen |= new
            old_live, new_live = old, new

            if frame["phase"] == "complete":
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
            # Only meaningful on frames as the ENGINE emitted them. The
            # candidate frame is synthesized here (and, in production, by
            # expand_worker_event) by slicing the start frame's policy, so its
            # length legitimately disagrees with the count Rust sent -- the
            # real decoder validates before that split, not after.
            if (frame["phase"] != "start" and frame["count"] is not None
                    and int(frame["count"]) != len(cells)):
                note = "COUNT MISMATCH — decoder would drop this frame"
                regressions += 1

            label = (
                frame["phase"] if frame["phase"] != "start"
                else "start/board" if frame["scope"] == "board"
                else "start/cands"
            )
            print(
                f"{ply:>3} {label:<12} {frame['round']:>5} "
                f"{len(cells):>7} {len(keep):>5} {len(old):>12} {len(new):>12}  {note}"
            )

        engine.apply_action(state, PlacementAction(unpack_coord_id(played)))
    session.discard(7_700_001)
    # The old client blinked a cell out and back on nearly every halving; a
    # model-based layer must never do it. A nonzero new count IS the flicker.
    print(
        f"\nblink-out/blink-back cells: old={old_revivals} new={new_revivals}"
    )
    regressions += new_revivals
    print(f"regressions: {regressions}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
