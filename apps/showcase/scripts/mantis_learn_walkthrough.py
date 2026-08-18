"""Generate learn/data/mantis_walkthrough.json inside the showcase container.

Replays the featured position (ply 11 of the captured showcase game) through
the exact vendored builder + served checkpoint and dumps every entity and
model output the learn figures draw. Run: docker exec -i <app> python - < me
"""

import json
import sys

import torch

from mantisnet import builder
from mantisnet._rust import Position
from mantisnet.serve import load_checkpoint, read_positions

CKPT = "/models/mantis_cellnodes1_it402_infer.pt"

# The captured showcase game (T=0, reproducible); featured position = first 11
# moves, Mantis (P0) to place the first stone of its third full turn.
GAME = [(0, 0), (1, 0), (-2, 1), (-1, 3), (3, -4), (1, -3), (-3, 4), (-3, 6),
        (1, -1), (3, -6), (-1, 1)]
FOCUS = (-2, 6)  # the cell the search ends up choosing

AXES = [(1, 0), (0, 1), (1, -1)]


def hexdist(aq, ar, bq, br):
    dq, dr = aq - bq, ar - br
    return max(abs(dq), abs(dr), abs(dq + dr))


def main():
    loaded = load_checkpoint(CKPT, "cpu")
    model, klent = loaded.model, loaded.klent
    params = sum(p.numel() for p in model.parameters())

    pos = Position.replay(GAME)
    moves_remaining = pos.moves_remaining
    to_move = pos.current_player

    read = read_positions(model, [pos], klent, "cpu")
    row = read.row(0)
    prior = torch.softmax(read.logits[row], dim=0).tolist()
    q_values = read.q_values[row].tolist()
    q_score = read.q_score[row].tolist()
    pi_prime = read.improved.probs[row].tolist()
    v_hat = float(read.improved.v_hat[0])
    state_head = float(read.value[0])

    legal = [(int(q), int(r)) for q, r in pos.legal_moves()]

    # Stone owners in play order: the player to move before move i owns it.
    owners = [int(Position.replay(GAME[:i]).current_player)
              for i in range(len(GAME))]
    stones = [{"q": q, "r": r, "owner": int(owners[i])}
              for i, (q, r) in enumerate(GAME)]
    stone_owner = {(q, r): int(owners[i]) for i, (q, r) in enumerate(GAME)}

    # Live windows: every 6-cell axis segment holding >= 1 stone. Pattern
    # digits are mover-relative (0 empty, 1 own=to_move, 2 opponent), digit at
    # 3^k = slot k; canonical rank folds reversal (builder._TERN_RANK).
    occupied = set(stone_owner)
    windows = {}
    for q, r in occupied:
        for ai, (aq, ar) in enumerate(AXES):
            for back in range(6):
                sq, sr = q - back * aq, r - back * ar
                key = (ai, sq, sr)
                if key in windows:
                    continue
                cells = [(sq + k * aq, sr + k * ar) for k in range(6)]
                digits = []
                for cq, cr in cells:
                    if (cq, cr) not in occupied:
                        digits.append(0)
                    elif stone_owner[(cq, cr)] == to_move:
                        digits.append(1)
                    else:
                        digits.append(2)
                if all(d == 0 for d in digits):
                    continue
                code = sum(d * 3 ** k for k, d in enumerate(digits))
                windows[key] = {
                    "axis": ai,
                    "start": [sq, sr],
                    "cells": [[cq, cr] for cq, cr in cells],
                    "digits": digits,
                    "pattern": int(builder._TERN_RANK[code]),
                }
    win_list = sorted(windows.values(),
                      key=lambda w: (w["axis"], w["start"][0], w["start"][1]))

    # Coverage + nearest-stone distance per legal cell.
    covered = set()
    for w in win_list:
        for cq, cr in w["cells"]:
            covered.add((cq, cr))
    cells_out = []
    for i, (q, r) in enumerate(legal):
        near = min(hexdist(q, r, sq, sr) for sq, sr in occupied)
        cells_out.append({
            "q": q, "r": r,
            "covered": (q, r) in covered,
            "nearest": near,
            "prior": prior[i],
            "pi_prime": pi_prime[i],
            "q_value": q_values[i],
            "q_score": q_score[i],
        })

    # Radius edges into the focus cell: stones within 8, typed by
    # orbit-48 x own/opp x shared-axis (192 classes).
    fq, fr = FOCUS
    edges = []
    for (sq, sr), owner in sorted(stone_owner.items()):
        d = hexdist(fq, fr, sq, sr)
        if d > 8:
            continue
        dq, dr = sq - fq, sr - fr
        shared = dq == 0 or dr == 0 or dq + dr == 0
        edges.append({
            "stone": [sq, sr],
            "orbit": builder.orbit48_id(dq, dr),
            "own": owner == to_move,
            "shared_axis": bool(shared),
            "distance": d,
        })

    # Action rows for the focus cell: its 18 post-placement windows.
    rows = []
    for ai, (aq, ar) in enumerate(AXES):
        for back in range(6):
            sq, sr = fq - back * aq, fr - back * ar
            cells = [(sq + k * aq, sr + k * ar) for k in range(6)]
            digits = []
            for cq, cr in cells:
                if (cq, cr) == (fq, fr):
                    digits.append(1)
                elif (cq, cr) not in occupied:
                    digits.append(0)
                elif stone_owner[(cq, cr)] == to_move:
                    digits.append(1)
                else:
                    digits.append(2)
            post = sum(d * 3 ** k for k, d in enumerate(digits))
            slot = back
            was_empty = all(d == 0 for k, d in enumerate(digits) if k != slot)
            rows.append({
                "axis": ai,
                "start": [sq, sr],
                "slot": slot,
                "digits": digits,
                "cls": int(builder._TERN_POST1_CLASS[post, slot]),
                "empty": was_empty,
            })

    out = {
        "source": "generated in-container from the served checkpoint",
        "checkpoint": CKPT,
        "params": int(params),
        "klent": {"tau": klent.tau, "lam": klent.lam,
                  "mass_floor": klent.mass_floor},
        "config": {k: getattr(loaded.config, k) for k in
                   ("h", "blocks", "heads", "ffn_factor", "d_max",
                    "value_queries", "value_bins", "window_attention",
                    "cell_latents", "cell_nodes", "cell_node_scope")},
        "moves": [[q, r] for q, r in GAME],
        "to_move": int(to_move),
        "moves_remaining": int(moves_remaining),
        "stones": stones,
        "windows": win_list,
        "legal_count": len(legal),
        "covered_count": sum(1 for c in cells_out if c["covered"]),
        "cells": cells_out,
        "v_hat": v_hat,
        "state_head_untrained": state_head,
        "focus": {"q": fq, "r": fr, "radius_edges": edges,
                  "action_rows": rows},
    }
    json.dump(out, sys.stdout)


main()
