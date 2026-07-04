# Introduction to Hexo (the game)

Audience: a developer landing in this repo cold. Everything rule-related here is
derived directly from the rules engine at `packages/hexo_engine/rust/src/`
(`state.rs`, `rules.rs`, `legal.rs`, `tactics.rs`, `coord.rs`), which is the
single source of truth for game rules. Strategy/game-theory sections are general
Connect6-family theory adapted to the hex grid. This document is about the game
only; for the training system built around it, see `docs/ARCHITECTURE.md`.

## 1. What Hexo is

Hexo is a two-player placement game -- essentially **Connect6 played on an
unbounded hexagonal grid**. Players alternately place stones (Player 0 first),
and the first player to own **six stones in a contiguous straight line** wins
immediately. There are no captures, no territory, and **no draws**
(`state.rs:64`: "Hexo has no normal draw under the current rules").

Key properties at a glance:

| Property | Value | Source |
|---|---|---|
| Board | unbounded sparse hex grid | `coord.rs:3`, `board.rs` (AHashMap storage) |
| Coordinates | axial `(q, r)`, each `i16`; third cube axis `s = -q - r` | `coord.rs` |
| Opening | Player 0 must place exactly one stone at the origin `(0, 0)` | `rules.rs:17-23`, `state.rs:49` |
| Normal turn | two single-stone placements by the same player | `state.rs:46-56` (TurnPhase) |
| Legality | any empty cell within hex-distance **8** of any existing stone | `legal.rs:11` (`LEGAL_RADIUS = 8`) |
| Win | a fully-owned 6-cell line window; checked after **every single placement** | `tactics.rs:14,206-208`, `state.rs:304-310` |
| Draws | none in the engine | `state.rs:64` |

## 2. Board geometry and coordinates

- Cells are addressed by axial coordinates `HexCoord { q: i16, r: i16 }`
  (`coord.rs:11`). Distance is the standard cube-coordinate hex distance
  (`coord.rs:77-82`).
- Straight lines exist along exactly **three axes** (`tactics.rs:23-30`):
  `Q = (1, 0)`, `R = (0, 1)`, `QR = (1, -1)`. A hex grid has 6 directions but
  only 3 unique line axes (vs 4 on the square grid of Gomoku/Connect6).
- Every cell coordinate has a stable packed **action ID**:
  `(q + 32768) << 16 | (r + 32768)` (`legal.rs:24-28`). Integer ordering of IDs
  equals deterministic `(q, r)` ordering. The Python mirror lives in
  `packages/hexo_engine/python/hexo_engine/types.py` (`pack_coord_id`).

## 3. Turn structure (verified in `state.rs` / `rules.rs`)

Turns are represented **autoregressively**: the engine only ever applies one
stone at a time, and a phase machine tracks where the current player is inside
its turn (`TurnPhase`, `state.rs:46-56`):

| Phase | Who | Legal placements | Then |
|---|---|---|---|
| `Opening` | Player 0 | only `(0, 0)` (`rules.rs:17-23`) | Player 1 enters `FirstStone` |
| `FirstStone` | current player | any empty cell within distance 8 of any stone | same player enters `SecondStone` |
| `SecondStone { first }` | same player | as above, but **not** the cell just played (`rules.rs:25-29`, `MoveError::ReusedFirstStone`) | control passes; opponent enters `FirstStone` |

So the placement sequence of a game is:

```
ply 0: P0 forced at (0,0)          (one-stone opening turn)
ply 1: P1 FirstStone   } P1's turn
ply 2: P1 SecondStone  }
ply 3: P0 FirstStone   } P0's turn
ply 4: P0 SecondStone  }
...
```

This is exactly the Connect6 "1 then 2-2-2-..." scheme. The phase transition
logic is `state.rs:312-330`; the explicit comment at `state.rs:313-316` confirms
the opening is "a special one-stone turn by Player 0".

The 1-then-2 scheme is Connect6's balance mechanism, and it carries over intact:
after every completed turn, the player who just moved has exactly **one more
stone on the board** than the opponent. Compare Gomoku, where the first player's
permanent one-stone initiative is strong enough to need forbidden-move rules
(Renju) to patch; here the initiative alternates with each turn.

Legality detail: the radius-8 neighborhood is taken around **any stone of
either color** (union of disks; see the test oracle
`recompute_non_opening_legal_ids`, `state.rs:559-569`). After the opening, the
legal set is the radius-8 disk around the origin: 216 cells (217-cell disk
minus the occupied origin).

## 4. Win condition

- **Win = six in a line.** The engine tracks every 6-cell straight-line window
  (`WINDOW_LEN = 6`, `tactics.rs:14`). A placement touches exactly 18 windows
  (3 axes x 6 offsets, `tactics.rs:17`). A window is a win for a player when
  all six of its cells are that player's stones (`is_win_for`,
  `tactics.rs:206-208`). Six **or more** in a row wins -- there is no overline
  rule, because any 7-in-a-row contains a fully-owned 6-window.
- **Resolution is immediate and per-placement.** A win is checked after every
  single stone (`state.rs:304-310`). The moment a placement completes a
  6-window, the game is terminal: the winner is recorded, no further legal
  moves exist (`legal_move_count` returns 0, `state.rs:204-213`), and -- per
  the header comment at `state.rs:9-10` -- **"If the first stone of a two-stone
  turn wins, the second stone is never played."**
- **Simultaneous wins are impossible by construction.** Stones go down strictly
  one at a time and the win check runs after each, so "both players complete a
  line at once" cannot happen. When both players have unstoppable threats, the
  game is a pure race decided by move order: whoever physically completes six
  first wins. There is no tiebreak or priority rule because none is needed.

## 5. Threats and the defensive arithmetic

The engine maintains incremental window masks (`WindowStore`, `tactics.rs`)
with a small tactics vocabulary:

- **Active window**: a 6-window containing stones of exactly one player
  (`is_active`, `tactics.rs:184-186`). A window with both colors is dead for
  winning purposes.
- **Threat**: an active window with **>= 4** stones of one color
  (`threat_player`, `tactics.rs:189-192`). With two placements per turn, a
  4-of-6 single-color window can be completed within one turn -- hence the
  threshold.

From these two definitions falls out the core game theory, which is Connect6's
threat arithmetic on three axes instead of four:

- **The defensive budget is 2 stones per turn, fixed.** Every threat the
  opponent holds at the start of your turn must be neutralized (or pre-empted by
  your own faster win) before your turn ends.
- **One stone can answer several threats at once** if their windows intersect:
  the real question each turn is whether a **hitting set of size <= 2** exists
  for the opponent's threat windows. If the minimum hitting set exceeds your
  budget, you have lost -- the opponent completes a line on their next turn no
  matter what you do.
- **Offense is therefore about building intersecting-proof threats**: two
  threats whose windows share no common cell already saturate the defender's
  entire turn; a third independent threat (or a threat that needs two distinct
  blocking cells) is checkmate. Because each placement touches 18 windows,
  strong shapes generate multiple overlapping half-threats quickly, and a
  single quiet-looking stone can convert several of them to real threats at
  once.

## 6. "Sudden death"

"Sudden death" is a term you will hear around Connect6, and it applies fully to
Hexo. It is **not a rule or an engine mechanic** -- it describes the game's
character: throughout a game there are many positions where a single blunder
loses outright within the next few moves.

Why the game is like this:

- The defensive budget (2 stones) is small and constant, while the offensive
  tempo (2 stones) builds threats fast -- one inattentive turn can let the
  opponent assemble a threat set with no 2-stone hitting set, which is an
  immediate, unrecoverable loss (section 5).
- Wins are checked after every single placement and end the game on the spot;
  there is no draw to escape into and no material to grind back.
- Compare Chess or Go: there, a mistake usually costs material, territory, or
  initiative, and the loss plays out -- often resistibly -- over tens of moves.
  In Hexo, as in Connect6, the evaluation of a position can go from "balanced"
  to "forced loss in 2-3 turns" on one move. Games end abruptly, not by
  accumulation.

Practical consequence for anyone (human or program) playing the game: threat
detection and the hitting-set check are not optional tactical garnish -- they
are the floor. Any player that ever leaves a completable threat unanswered
loses immediately, so defensive vigilance has to be perfect on every single
turn, even deep into an otherwise strategic middlegame.

## 7. Draws and game length

The engine has no draw of any kind: `GameOutcome` (`state.rs:66-71`) always has
a winner, and the unbounded board can never fill up. A game that has not yet
produced six-in-a-row simply continues -- in principle indefinitely. In
practice, the sudden-death character of section 6 means decisive games are the
norm; harnesses that drive the engine impose their own external move caps, but
that is a property of the harness, not of the game.

## 8. Comparisons to related games

| Game | What transfers to Hexo | What does not |
|---|---|---|
| **Connect6** | Almost everything: the 1-then-2-2-2 placement scheme (designed to fix Gomoku's first-mover advantage), 6-in-a-row goal, threat counting and hitting-set defense (a turn answers at most 2 independent threats), the sudden-death character, no practical draw concern | Square grid has **4** line axes; Hexo's hex grid has **3** (`tactics.rs:23`), so each stone projects threats in fewer directions. Connect6 is played on a bounded 19x19; Hexo is unbounded with a radius-8 placement locality rule and a forced-origin opening |
| **Gomoku / Renju** | Line-completion intuition; threat-sequence thinking (fours/threes generalize to Hexo's >= 4-of-6 windows) | One stone per turn changes the entire defensive arithmetic; 5-in-a-row; overline and forbidden-move rules (Renju) have no Hexo equivalent |
| **Hex** | Only the grid. Despite the name, Hexo is NOT the connection game Hex | Hex's goal is connecting opposite board edges on a bounded rhombus; no line-of-N condition; Hex provably has no draws by topology, Hexo simply never fills its infinite board |
| **Go** | Essentially nothing rule-wise or tactically; both reward long-range judgment, but Go's slow accumulation of advantage is exactly what Hexo's sudden-death character lacks | Captures, territory, ko, komi, passing -- none exist in Hexo; losses in Go play out gradually, Hexo games end abruptly |

## 9. Glossary (game terms)

| Term | Meaning here |
|---|---|
| **ply / placement** | One single stone placed. The engine is fully autoregressive: a "move" in engine terms is always one stone (`Placement`, `state.rs:59-62`). |
| **turn** | One logical turn: 1 placement for the opening, 2 placements otherwise (`MoveRecord`, `state.rs:86-93`). |
| **window** | A specific 6-cell straight-line segment on one of the 3 axes; the unit of win/threat detection (`WindowKey`, `tactics.rs:56`). |
| **active window** | A window containing stones of exactly one player; the only windows that can still become wins. |
| **threat** | An active window with >= 4 stones (`tactics.rs:189-192`) -- completable within one turn. |
| **hitting set** | A set of cells that intersects every one of the opponent's threat windows; the defender needs one of size <= 2 every turn. |
| **standing win** | A win-in-1: an active window with 5 stones, completable by a single placement. |
| **sudden death** | The game's character (not a rule): blunders convert to forced losses within a few moves; see section 6. |
| **action ID** | Packed `u32` cell coordinate, `(q+2^15)<<16 \| (r+2^15)` (`legal.rs:24-28`); the engine's stable move encoding. |
| **D6** | The order-12 symmetry group of the hex grid about the origin (6 rotations x reflection): any position has up to 12 strategically identical images. |
