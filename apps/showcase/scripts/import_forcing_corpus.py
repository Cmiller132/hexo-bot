"""Convert the community forcing-solver corpus into the lab preset contract.

Input:  forcing_solver_corpus1.jsonl (beside this script) — one puzzle per
        line: a stone list with an attacker to move, usually a recorded solve
        line, and solver metadata this importer deliberately drops (the lab
        shows positions, not third-party verdicts).
Output: apps/showcase/web/learn/data/forcing_corpus.json with
        doc.positions[] entries in the lab's extended preset shape:

- sequence entries {id, title, moves, line?}: the stone list in placement
  order (the corpus files carry the 1-2-2 record order; one known entry is
  sorted instead and uses the engine-verified reordering pinned below). The
  optional ``line`` is the submitter's recorded solve continuation — the lab
  pre-loads it as a second, deletable line.
- free entries {id, title, free: {p0, p1, to_move}}: positions no legal game
  reaches (the attacker moves out of turn parity), loadable only as free-edit.

Every emitted move list is replayed through hexo_engine here; any illegal
placement, early terminal, parity mismatch, or rules-config divergence aborts
the run. Run from the repo root:

    python apps/showcase/scripts/import_forcing_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path

import hexo_engine as engine
from hexo_engine.types import AxialCoord, PlacementAction

SRC = Path(__file__).with_name("forcing_solver_corpus1.jsonl")
OUT = Path(__file__).resolve().parents[1] / "web" / "learn" / "data" / "forcing_corpus.json"

# The one corpus entry whose stone list is coordinate-sorted rather than
# placement-ordered. This ordering was found by a within-8 DFS over the same
# stones and is engine-verified below like every other entry.
REORDERED = {
    "0l4291i_live": [
        [0, 0], [-1, 0], [0, 1], [-1, 2], [1, 1], [-2, 3], [-2, 4], [-3, 3],
        [-1, 4], [-1, 3], [0, 2], [2, 0], [0, 5], [-1, 6], [0, 3], [1, 3],
        [2, 3], [0, 4], [1, 2], [3, 2], [4, 2], [3, 3], [4, 1], [3, 5],
        [6, 1], [4, 0], [4, 4], [7, 1], [8, 1], [5, 2], [6, 0], [9, 0],
        [10, -1], [6, -1], [7, -2], [6, -2], [5, -2], [6, -3], [7, -4],
        [7, -3], [8, -4], [7, -1], [7, 2], [9, -4], [10, -4], [8, -3],
        [8, -2], [11, 0], [11, 1], [8, -1], [8, 0], [12, 3], [14, -8],
        [10, 1], [11, -6], [15, -7], [17, -8], [18, -8], [8, 4], [18, -5],
        [16, -12], [8, 5], [13, -4],
    ],
}


def record_owner(i: int) -> int:
    """Owner (0/1) of placement-record index i under the 1-2-2 turn structure."""
    if i == 0:
        return 0
    return 1 - (((i - 1) // 2) % 2)


def replay(cells: list[list[int]], label: str) -> bool:
    """Replay through the engine; returns True when the FINAL state is terminal.

    Raises on an illegal placement or a terminal state before the end.
    """
    state = engine.new_game()
    for i, (q, r) in enumerate(cells):
        action = PlacementAction(AxialCoord(q=int(q), r=int(r)))
        if not engine.is_legal_action(state, action):
            raise SystemExit(f"{label}: placement {i} at ({q},{r}) is illegal")
        result = engine.apply_action(state, action)
        if result.terminal and i < len(cells) - 1:
            raise SystemExit(f"{label}: terminal at placement {i} with trailing moves")
    return engine.terminal(state) is not None


def main() -> None:
    entries = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    titles_seen: dict[str, int] = {}
    positions = []
    for e in entries:
        eid = e["id"]
        pos = e["position"]
        cfg = pos.get("config")
        if cfg is not None and (cfg.get("win_length") != 6 or cfg.get("placement_radius") != 8):
            raise SystemExit(f"{eid}: rules config diverges from the engine: {cfg}")
        title = e.get("name") or eid
        titles_seen[title] = titles_seen.get(title, 0) + 1
        if titles_seen[title] > 1:
            title = f"{title} ({eid})"

        stones = pos["stones"]
        attacker = 0 if pos["attacker"] == "P1" else 1
        moves = REORDERED.get(eid) or [[s[0], s[1]] for s in stones]
        owners_ok = eid in REORDERED or all(
            record_owner(i) == (0 if s[2] == "P1" else 1) for i, s in enumerate(stones)
        )
        n = len(moves)
        parity_ok = owners_ok and record_owner(n) == attacker and (
            pos["placements_remaining"] == (1 if (n - 1) % 2 == 1 else 2)
        )
        if parity_ok:
            if replay(moves, eid):
                raise SystemExit(f"{eid}: puzzle position is already terminal")
            entry: dict = {"id": eid, "title": title, "moves": moves}
            line = e.get("line") or []
            if line:
                full = moves + [[c[0], c[1]] for c in line]
                if not replay(full, f"{eid}+line"):
                    raise SystemExit(f"{eid}: recorded line does not end the game")
                if record_owner(len(full) - 1) != attacker:
                    raise SystemExit(f"{eid}: recorded line's winner is not the attacker")
                entry["line"] = [[c[0], c[1]] for c in line]
            positions.append(entry)
        else:
            # Out-of-parity puzzle: representable only as a free-edit position.
            p0 = [[s[0], s[1]] for s in stones if s[2] == "P1"]
            p1 = [[s[0], s[1]] for s in stones if s[2] == "P2"]
            if abs(len(p0) - len(p1)) > 2:
                raise SystemExit(f"{eid}: stone counts too far from any turn parity")
            if e.get("line"):
                # A recorded line cannot replay under engine turn order here.
                pass
            positions.append({
                "id": eid, "title": title,
                "free": {"p0": p0, "p1": p1, "to_move": attacker},
            })

    n_seq = sum(1 for p in positions if "moves" in p)
    n_line = sum(1 for p in positions if "line" in p)
    n_free = sum(1 for p in positions if "free" in p)
    OUT.write_text(json.dumps({"positions": positions}, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(positions)} positions "
          f"({n_seq} sequences, {n_line} with recorded lines, {n_free} free-edit)")


if __name__ == "__main__":
    main()
