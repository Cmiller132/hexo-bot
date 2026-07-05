"""Phase-A BC bootstrap writer (spec §7): HF corpus -> shrimp_compact_v1 shards.

Replays every corpus game through hexo_engine (legality/terminal validated,
games failing validation are skipped and counted), builds one decision row per
placement with a ONE-HOT policy on the played move, hard z from the engine
winner, STV masked (no search values; horizons stored but empty), moves_left
real. Game-level md5 train/val split. Multi-game shards (~64 games each).

Imports shrimp via PYTHONPATH (shrimp is never pip-installed — see README).
Usually invoked through scripts/prefit_launch.sh, but can be run directly:
    python scripts/bootstrap_from_corpus.py --corpus <corpus.jsonl> --out data/prefit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "packages" / "shrimp" / "python"))

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from shrimp.engine_facts import player_int
from shrimp.features import record_phase, record_player, window_scan
from shrimp.geometry import pack_action_id
from shrimp.samples import STV_HORIZONS, ShrimpSampleData, finalize_game_samples
from shrimp.shards import write_compact_shard
from shrimp.support import build_support

VAL_PERCENT = 5
GAMES_PER_SHARD = 64


def replay_game(moves: list[list[int]], winner: int, game_hash: str):
    """Replay one corpus game; return finalized rows or None on validation failure.

    Facts are built INCREMENTALLY from the tested derivations (record phase /
    player from ordinal position; hot / standing-win via `window_scan`, both
    property-tested ≡ the engine in tests/test_shrimp_features.py) — the
    engine still rules on every placement's legality and on the terminal
    winner, but the heavyweight `to_python_state` mirror never runs.
    """

    state = api.new_game()
    pending = []
    records: list[tuple[int, int, int, int]] = []
    for ply, (q, r) in enumerate(moves):
        phase = record_phase(ply)
        current = record_player(ply)
        first_stone = (records[-1][0], records[-1][1]) if phase == "SecondStone" else None
        # The one-hot policy target below is projected onto the featurizer's
        # radius-limited legal prefix at train time (expand_sample -> _legal_slot);
        # a target outside that prefix is a hard ValueError. build_support here reads
        # the SAME SHRIMP_SUPPORT_RADIUS the featurizer uses, so this is the exact
        # membership check expand_sample will apply. Human corpus games occasionally
        # play far from the cluster; at small radii those cells fall outside the halo.
        # Reject the whole game (per-game bookkeeping in finalize_game_samples couples
        # rows via moves_left / opp_policy / short_term_value chains, so a single row
        # cannot be dropped without corrupting the survivors).
        support = build_support([(sq, sr) for sq, sr, _o, _i in records])
        if (int(q), int(r)) not in support.index or support.index[(int(q), int(r))] >= support.legal_count:
            return None, "off_support_target"
        own_hot, opp_hot, own_win, opp_win = window_scan(
            tuple(records), current, len(records)
        )
        sample = ShrimpSampleData(
            game_id=game_hash,
            turn_index=ply,
            current_player=current,
            phase=phase,
            records=tuple(records),
            first_stone=first_stone,
            own_hot=own_hot,
            opp_hot=opp_hot,
            own_win=own_win,
            opp_win=opp_win,
            policy=((pack_action_id(int(q), int(r)), 1.0),),
            metadata={"source": "hf_bootstrap"},
        )
        pending.append((current, sample, 0.0))
        try:
            result = api.apply_action(state, PlacementAction(AxialCoord(q=int(q), r=int(r))))
        except Exception:
            return None, f"illegal_move_at_{ply}"
        records.append((int(q), int(r), current, ply + 1))  # placement_index is 1-based
        if result.terminal and ply != len(moves) - 1:
            return None, f"premature_terminal_at_{ply}"
    terminal = api.terminal(state)
    if terminal is None:
        return None, "not_terminal_at_end"
    engine_winner = player_int(terminal.winner)
    corpus_winner = 0 if int(winner) == 1 else 1
    if engine_winner != corpus_winner:
        return None, "winner_mismatch"
    # STV masked for BC (no search values): empty horizons at finalize.
    return finalize_game_samples(pending, engine_winner, horizons=()), None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(REPO / "data" / "hexo-bootstrap-corpus" / "hexo_human_corpus.jsonl"))
    parser.add_argument("--out", default=str(REPO / "data" / "shrimp_bootstrap"))
    parser.add_argument("--limit", type=int, default=0, help="games (0 = all)")
    args = parser.parse_args()

    out = Path(args.out)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    buffers = {"train": [], "val": []}
    shard_counts = {"train": 0, "val": 0}
    row_counts = {"train": 0, "val": 0}
    games = {"train": 0, "val": 0}
    skipped: dict[str, int] = {}

    def flush(split: str, force: bool = False) -> None:
        buf = buffers[split]
        if not buf or (not force and games[split] % GAMES_PER_SHARD != 0):
            return
        path = out / split / f"shard_{shard_counts[split]:04d}.npz"
        rows = write_compact_shard(
            path, buf, short_term_value_horizons=STV_HORIZONS,
            sidecar={"split": split, "source": "hf_bootstrap"},
        )
        row_counts[split] += rows
        shard_counts[split] += 1
        buf.clear()

    with open(args.corpus, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if args.limit and line_no >= args.limit:
                break
            game = json.loads(line)
            rows, reason = replay_game(game["moves"], game["winner"], game["game_hash"])
            if rows is None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            digest = hashlib.md5(game["game_hash"].encode("utf-8")).digest()
            split = "val" if digest[0] % 100 < VAL_PERCENT else "train"
            buffers[split].extend(rows)
            games[split] += 1
            flush(split)
            if line_no % 500 == 0:
                print(f"[{line_no}] train={row_counts['train']} val={row_counts['val']} skipped={sum(skipped.values())}", flush=True)

    flush("train", force=True)
    flush("val", force=True)
    summary = {
        "games": games,
        "rows": row_counts,
        "shards": shard_counts,
        "skipped": skipped,
        "val_percent": VAL_PERCENT,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
