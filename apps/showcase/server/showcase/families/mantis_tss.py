"""TSS-Gumbel: Threat-Space Search inside the MantisNet decision session.

The vendored Gumbel session (``packages/mantisnet/rust/vendor/search``) is a
pure question-and-answer machine: it hands the driver leaf positions and takes
back priors and a value. That is the whole seam TSS needs. This module answers
those questions with proofs where proofs exist:

* **Leaf values.** A leaf whose λ¹ analysis is decided answers ±1 instead of the
  net's v̂ — the same hard value the hexfield_eq tree backs up. An undecided
  leaf that passes the deep gate gets a verified deep solve, submitted before
  the wave's model forward runs so the two overlap; a verified win/loss answers
  ±1, anything else answers the net.
* **Leaf priors.** Always the net's priors, then the λ¹ move guard. The session
  extends each line through ``prior_argmax``, so zeroing refuted replies is
  exactly what makes the line follow the forced continuation.
* **The root.** The same guard on the root priors, plus one verified deep solve
  running concurrently with the whole search. A proven win overrides the
  decision with the certificate's move.

Every verdict, classification, and solve comes from ``hexfield_eq._rust``'s
``TssProbe`` — the marshalling layer over the same Rust functions the
hexfield_eq tree calls. Nothing in this file decides what a threat is.

A solve names its position by the PLACEMENT PATH from the search root, never by
a stone set: the solver reads the turn phase (which stone of the turn is
pending, and which cell was the first), and only a true placement history
carries it.

Budgets. The root and the leaves are budgeted separately: a leaf solve is one
of hundreds per move and only sharpens a line's value (``node_cap``,
``wall_budget_ms``), while the root solve runs once and can replace the played
move (``root_node_cap``, ``root_wall_budget_ms``). Node caps bound each solve
deterministically; the wall clocks bound the move — past a clock nothing new is
submitted and nothing unfinished is waited on. A partial solve never produces a
value, so a wall clock can cost strength and can never cost soundness.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutureTimeout
from dataclasses import dataclass, replace
from typing import Any

# Deep solve at every undecided leaf ("all"), or only where the engine's
# board-level >=4-window index says something tactical is live ("threats").
_LEAF_GATES = ("threats", "all")


@dataclass(frozen=True)
class TssConfig:
    """TSS knobs for one MantisNet search.

    The root and the leaves get SEPARATE budgets, because they are different
    jobs. A leaf solve runs hundreds of times per move and only sharpens one
    line's value, so it is capped tight. The root solve runs ONCE per move and
    can replace the played move outright, so it is worth two orders of
    magnitude more nodes — post-mortem of game 34e4cb07: the forced wins the
    bot missed needed 1577, 1952 and 12880 solver nodes at the root, all of
    them invisible at the leaf cap of 500 and all of them under 600 ms.

    `apps/showcase/README.md` documents these for operators.

    enabled              run TSS at all. Off is the bare Gumbel search, byte
                         for byte.
    node_cap             solver nodes per LEAF solve.
    root_node_cap        solver nodes for the one root solve.
    leaf_gate            "threats" (default) solves only leaves with a live
                         >=4 window; "all" solves every leaf with an
                         undecided λ¹.
    workers              threads for leaf solves. The root solve gets its own
                         thread and never competes with them.
    wall_budget_ms       per-move ceiling on waiting for LEAF solves.
    root_wall_budget_ms  per-move ceiling on waiting for the ROOT solve. Its
                         own clock: the root solve is worth waiting for after
                         the search has decided, and must never stall a move
                         past this.
    """

    enabled: bool = True
    node_cap: int = 500
    root_node_cap: int = 20_000
    leaf_gate: str = "threats"
    workers: int = 3
    wall_budget_ms: int = 1500
    root_wall_budget_ms: int = 3000

    def __post_init__(self) -> None:
        if self.node_cap < 1:
            raise ValueError(f"tss node_cap must be >= 1, got {self.node_cap}")
        if self.root_node_cap < 1:
            raise ValueError(
                f"tss root_node_cap must be >= 1, got {self.root_node_cap}"
            )
        if self.leaf_gate not in _LEAF_GATES:
            raise ValueError(
                f"tss leaf_gate must be one of {list(_LEAF_GATES)}, got {self.leaf_gate!r}"
            )
        if self.workers < 1:
            raise ValueError(f"tss workers must be >= 1, got {self.workers}")
        if self.wall_budget_ms < 1:
            raise ValueError(
                f"tss wall_budget_ms must be >= 1, got {self.wall_budget_ms}"
            )
        if self.root_wall_budget_ms < 1:
            raise ValueError(
                f"tss root_wall_budget_ms must be >= 1, got {self.root_wall_budget_ms}"
            )

    @classmethod
    def from_profile(cls, raw: dict[str, Any] | None) -> "TssConfig":
        """Build from a search profile's ``[tss]`` table. Unknown keys are an
        error: a typo'd knob must not read as the default."""
        if not raw:
            return cls()
        known = {field for field in cls.__dataclass_fields__}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(
                f"unknown [tss] profile keys {unknown}; expected {sorted(known)}"
            )
        defaults = cls()
        return cls(
            enabled=bool(raw.get("enabled", defaults.enabled)),
            node_cap=int(raw.get("node_cap", defaults.node_cap)),
            root_node_cap=int(raw.get("root_node_cap", defaults.root_node_cap)),
            leaf_gate=str(raw.get("leaf_gate", defaults.leaf_gate)),
            workers=int(raw.get("workers", defaults.workers)),
            wall_budget_ms=int(raw.get("wall_budget_ms", defaults.wall_budget_ms)),
            root_wall_budget_ms=int(
                raw.get("root_wall_budget_ms", defaults.root_wall_budget_ms)
            ),
        )

    def with_enabled(self, enabled: bool) -> "TssConfig":
        """The per-game UI toggle: it decides `enabled` and nothing else."""
        if bool(enabled) == self.enabled:
            return self
        return replace(self, enabled=bool(enabled))


class LeafPlan:
    """One leaf's TSS answer, prepared before the wave's model forward runs."""

    __slots__ = ("move_classes", "hard_value", "future")

    def __init__(
        self,
        move_classes: list[int] | None,
        hard_value: float | None,
        future: "Future | None",
    ) -> None:
        self.move_classes = move_classes
        self.hard_value = hard_value
        self.future = future


def guard_weights(
    weights: list[float], move_classes: list[int] | None
) -> list[float]:
    """The λ¹ move guard, `search.rs::tactical_guard_weights` in Python.

    With a win-now move on the board every other move is zeroed; otherwise
    λ¹-refuted moves are zeroed. Zeroing everything is no guard at all, so the
    raw weights come back. `move_classes` is None when the probe reported the
    guard inert at this state.
    """
    if move_classes is None:
        return weights
    if len(move_classes) != len(weights):
        raise RuntimeError(
            f"TSS guard: {len(move_classes)} move classes for {len(weights)} weights"
        )
    guarded = list(weights)
    if any(cls == 1 for cls in move_classes):
        for index, cls in enumerate(move_classes):
            if cls != 1:
                guarded[index] = 0.0
    elif any(cls != -1 for cls in move_classes):
        for index, cls in enumerate(move_classes):
            if cls == -1:
                guarded[index] = 0.0
    if all(weight <= 0.0 for weight in guarded):
        return weights
    return guarded


def _engine_state(moves: list[tuple[int, int]]) -> Any:
    """The showcase engine state for a placement sequence.

    The probe's root must be a showcase `hexo_engine` state (that is the capsule
    hexfield_eq reads), and building it from the same move list the MantisNet
    root was replayed from is what keeps the two roots the same position.
    """
    import hexo_engine as engine
    from hexo_engine.types import AxialCoord, PlacementAction

    state = engine.new_game()
    for q, r in moves:
        engine.apply_action(state, PlacementAction(AxialCoord(int(q), int(r))))
    return state


class TssRunner:
    """TSS for one MantisNet search: one probe, one budget, one set of counters.

    Constructed per move and closed at the end of the search, so no solver state
    outlives the position it was built for.
    """

    def __init__(self, moves: list[tuple[int, int]], config: TssConfig) -> None:
        from hexfield_eq import _rust

        self._config = config
        self._probe = _rust.TssProbe(_engine_state(moves))
        self._started = time.monotonic()
        self._deadline = self._started + config.wall_budget_ms / 1000.0
        # The root gets its own clock. Leaf solves are worth nothing once the
        # wave they belong to has been answered, but the root solve stays worth
        # waiting for right up to the decision.
        self._root_deadline = self._started + config.root_wall_budget_ms / 1000.0
        self._leaf_pool = ThreadPoolExecutor(
            max_workers=config.workers, thread_name_prefix="tss-leaf"
        )
        # The root solve can override the played move, so it never queues behind
        # leaf solves for a pool thread.
        self._root_pool: ThreadPoolExecutor | None = None
        self._root_future: "Future | None" = None
        self._root_status = "skipped"
        self._root_ms = 0.0
        self._root_resolved = False
        self._proven_move: tuple[int, int] | None = None
        self._counters = {
            "lambda1_leaf_hits": 0,
            "lambda1_root_guard": 0,
            "leaf_guards": 0,
            "deep_attempted": 0,
            "deep_win": 0,
            "deep_loss": 0,
            "deep_unknown": 0,
            "deep_timeouts": 0,
            "root_timeouts": 0,
            "deep_nodes": 0,
            "verify_failed": 0,
        }

    # -- root ----------------------------------------------------------------

    def begin_root(
        self, legal_moves: list[tuple[int, int]], priors: list[float]
    ) -> list[float]:
        """Guard the root priors and start the root deep solve.

        Returns the priors the session should sample its candidates from. The
        deep solve runs only when λ¹ leaves the root undecided — a decided root
        is already answered, and the guard has already narrowed the candidates.
        """
        read = self._probe.lambda1([], legal_moves)
        classes = read["move_classes"]
        if classes is not None:
            self._counters["lambda1_root_guard"] += 1
        verdict = read["verdict"]
        if verdict is None:
            self._root_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="tss-root"
            )
            self._root_future = self._root_pool.submit(
                self._timed_solve, [], self._config.root_node_cap
            )
            self._counters["deep_attempted"] += 1
            self._root_status = "running"
        else:
            self._root_status = "lambda1_win" if verdict > 0.0 else "lambda1_loss"
        return guard_weights(priors, classes)

    def decision_override(self) -> tuple[int, int] | None:
        """The proven root move, or None.

        Waits out whatever is left of the ROOT budget: if the search decided
        first, a proven win is worth that wait; if the budget is already gone,
        the solve is dropped and the search's own move stands.
        """
        if self._root_resolved:
            return self._proven_move
        self._root_resolved = True
        if self._root_future is None:
            return None
        try:
            out, elapsed_ms = self._root_future.result(
                timeout=max(0.0, self._root_deadline - time.monotonic())
            )
        except _FutureTimeout:
            self._root_future.cancel()
            self._root_status = "timeout"
            self._counters["root_timeouts"] += 1
            return None
        self._absorb(out)
        self._root_status = str(out["status"])
        self._root_ms = float(elapsed_ms)
        if out["status"] == "win" and out["move"] is not None:
            q, r = out["move"]
            self._proven_move = (int(q), int(r))
        return self._proven_move

    # -- leaves --------------------------------------------------------------

    def plan_wave(
        self,
        paths: list[list[tuple[int, int]]],
        legal_lists: list[list[tuple[int, int]]],
    ) -> list[LeafPlan]:
        """λ¹ every leaf of one wave and submit the deep solves it earns.

        Called before the wave's model forward, so every submitted solve runs
        against it rather than after it.
        """
        plans: list[LeafPlan] = []
        for path, legal in zip(paths, legal_lists):
            read = self._probe.lambda1(path, legal)
            classes = read["move_classes"]
            if classes is not None:
                self._counters["leaf_guards"] += 1
            verdict = read["verdict"]
            future = None
            if (
                verdict is None
                and (self._config.leaf_gate == "all" or read["has_threats"])
                and self._remaining() > 0.0
            ):
                future = self._leaf_pool.submit(
                    self._probe.deep_solve, list(path), self._config.node_cap
                )
                self._counters["deep_attempted"] += 1
            plans.append(LeafPlan(classes, verdict, future))
        return plans

    def leaf_priors(self, plan: LeafPlan, priors: list[float]) -> list[float]:
        return guard_weights(priors, plan.move_classes)

    def leaf_value(self, plan: LeafPlan, net_value: float) -> float:
        """The value to answer this leaf with: λ¹, then a verified deep solve,
        then the net."""
        if plan.hard_value is not None:
            self._counters["lambda1_leaf_hits"] += 1
            return float(plan.hard_value)
        if plan.future is None:
            return net_value
        try:
            out = plan.future.result(timeout=self._remaining())
        except _FutureTimeout:
            plan.future.cancel()
            self._counters["deep_timeouts"] += 1
            return net_value
        self._absorb(out)
        value = out["value"]
        return net_value if value is None else float(value)

    # -- telemetry / lifecycle ----------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Per-move TSS counters, for the result dict and the live frame."""
        return {
            **self._counters,
            "root_status": self._root_status,
            "root_ms": round(self._root_ms, 3),
            "total_ms": round((time.monotonic() - self._started) * 1000.0, 3),
            "node_cap": self._config.node_cap,
            "root_node_cap": self._config.root_node_cap,
            "leaf_gate": self._config.leaf_gate,
        }

    def close(self) -> None:
        """Drop the pools. Queued solves are cancelled; one already running is
        abandoned — its result was never going to be waited on."""
        self._leaf_pool.shutdown(wait=False, cancel_futures=True)
        if self._root_pool is not None:
            self._root_pool.shutdown(wait=False, cancel_futures=True)

    # -- internals -----------------------------------------------------------

    def _timed_solve(
        self, path: list[tuple[int, int]], node_cap: int
    ) -> tuple[dict[str, Any], float]:
        started = time.monotonic()
        out = self._probe.deep_solve(list(path), node_cap)
        return out, (time.monotonic() - started) * 1000.0

    def _remaining(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    def _absorb(self, out: dict[str, Any]) -> None:
        status = str(out["status"])
        self._counters[f"deep_{status}"] += 1
        self._counters["deep_nodes"] += int(out["nodes"])
        self._counters["verify_failed"] += int(out["verify_failed"])


# -- the solver endpoint --------------------------------------------------


# How long a cancelled deep solve gets to drain through the solver's
# budget-exhaustion exits. Cancel checks sit on every node expansion, so the
# drain is near-immediate; the margin covers a heavily loaded host.
_CANCEL_GRACE_S = 15.0


def _timed_deep_solve(
    probe: Any, path: list[tuple[int, int]], node_cap: int, remaining_s: float
) -> dict[str, Any] | None:
    """One verified deep solve on its own thread, cancelled past the clock.

    ``deep_solve`` is a blocking Rust call. Past ``remaining_s`` the probe's
    cooperative cancel drains it through the solver's budget-exhaustion exits:
    the harvested result is ``"unknown"`` by construction, carries the real
    node count, and is marked ``timed_out``. Merely abandoning the future
    (the pre-cancel behavior) left the call churning to its node cap on the
    worker thread — hours of CPU and an unbounded PN tree at lab-scale caps,
    and the interpreter cannot even exit while that thread runs."""
    if remaining_s <= 0.0:
        return None
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tss-solve")
    try:
        future = pool.submit(probe.deep_solve, list(path), node_cap)
        try:
            return future.result(timeout=remaining_s)
        except _FutureTimeout:
            probe.cancel_deep_solve()
            try:
                out = future.result(timeout=_CANCEL_GRACE_S)
            except _FutureTimeout:  # pragma: no cover - cancel not honored
                future.cancel()
                return None
            out["timed_out"] = True
            return out
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _win_now_move(
    legal: list[tuple[int, int]], classes: list[int] | None,
    replay: Any = None, path: list[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    """Pick a class-1 (win-now) cell of a λ¹ read, or None.

    With two placements per turn, λ¹ can class EVERY cell win-now once the
    win is unstoppable (waste a stone, complete with the next) — and the
    first cell in legal order is then an arbitrary far corner. Given a
    ``replay`` callable the pick prefers a completion that ends the game on
    this very stone; otherwise it falls back to the first win-now cell,
    which is a genuine threat when the set is selective.
    """
    if classes is None:
        return None
    wins = [(int(q), int(r)) for (q, r), cls in zip(legal, classes) if cls == 1]
    if not wins:
        return None
    if replay is not None and path is not None:
        for cell in wins:
            if replay(path + [cell]).is_terminal:
                return cell
    return wins[0]


def _hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    dq, dr = a[0] - b[0], a[1] - b[1]
    return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def _walk_line(
    probe: Any, replay: Any, winner: int, first: tuple[int, int],
    config: TssConfig, deadline: float, line_cap: int,
) -> tuple[list[tuple[int, int]], int, int]:
    """Walk a proven root move down to the end of the game.

    Hexo turns place two stones, so the walk keys every ply on whose placement
    it is, not on alternation. On the winner's plies the next move must be
    certified (λ¹ win-now, else a verified deep solve) — an uncertified ply,
    the clock, or the depth cap ends the line early. On the defender's plies
    the walk plays the first reply that survives the λ¹ guard (the first
    legal reply when the guard is inert or refutes everything) and keeps
    going until the winning six lands.

    Returns ``(line, forced_through, nodes)``. ``forced_through`` counts the
    line's leading plies while every defender reply was uniquely forced —
    beyond it the defense had choices and the line is a certified
    demonstration against one of them, not the only proof path.

    The walk extends only while the winner's threats are LIVE: a defender ply
    whose guard refutes nothing means the previous winner move made no
    threat, and continuing from there just waltzes (the win survives any pair
    of quiet moves, so neither the certified solver move nor a picked defense
    makes progress — observed as stones marching to infinity). Quiet
    positions end the line; so do an uncertifiable winner ply, a λ¹ win-now
    FOR THE DEFENDER (it would contradict the certified win above it), the
    clock, and the depth cap.
    """
    nodes = 0
    line = [first]
    path = [first]
    forced_through: int | None = None  # None while the defense stays forced
    while len(line) < line_cap and time.monotonic() < deadline:
        position = replay(path)
        if position.is_terminal:
            break
        legal = position.legal_moves()
        read = probe.lambda1(path, legal)
        classes = read["move_classes"]
        if int(position.current_player) == winner:
            nxt = None
            if read["verdict"] is not None and float(read["verdict"]) > 0.0:
                nxt = _win_now_move(legal, classes, replay, path)
            if nxt is None:
                out = _timed_deep_solve(
                    probe, path, config.root_node_cap,
                    deadline - time.monotonic(),
                )
                if out is not None:
                    nodes += int(out["nodes"])
                    if str(out["status"]) == "win" and out["move"] is not None:
                        q, r = out["move"]
                        nxt = (int(q), int(r))
            if nxt is None:
                break
            line.append(nxt)
            path.append(nxt)
        else:
            if classes is None:
                break
            if any(cls == 1 for cls in classes):
                break
            keep = [
                (int(q), int(r))
                for (q, r), cls in zip(legal, classes)
                if cls != -1
            ]
            if len(keep) == len(legal):
                break  # nothing refuted: the threats went quiet
            if len(keep) != 1 and forced_through is None:
                forced_through = len(line)
            if keep:
                nxt = keep[0]
            else:
                # Every reply is refuted: any reply demonstrates, so the
                # defense at least fights near the action rather than
                # conceding from a far corner.
                cells = [(int(q), int(r)) for q, r in legal]
                nxt = min(cells, key=lambda c: _hex_distance(c, line[-1]))
            line.append(nxt)
            path.append(nxt)
    return line, (len(line) if forced_through is None else forced_through), nodes


def solve_position(
    moves: list[tuple[int, int]], config: TssConfig, *, line_cap: int = 24
) -> dict[str, Any]:
    """One root verdict for the solver view: λ¹, then the verified deep solve.

    Returns the verdict for the side to move, the proven move where one
    exists, the λ¹ guard classes over every legal move, and a line walked
    from the proven move to the end of the game (see ``_walk_line``;
    ``forced_through`` counts its uniquely-forced prefix). The whole call
    shares one wall clock (``root_wall_budget_ms``): the root solve takes
    what it needs and the line walk gets the remainder.
    """
    from hexfield_eq import _rust
    from mantisnet import _rust as mantis

    def replay(path: list[tuple[int, int]]) -> Any:
        return mantis.Position.replay(list(moves) + path)

    root = replay([])
    if root.is_terminal:
        raise ValueError("the position is terminal: nothing to solve")
    probe = _rust.TssProbe(_engine_state(moves))
    started = time.monotonic()
    deadline = started + config.root_wall_budget_ms / 1000.0
    legal = root.legal_moves()
    read = probe.lambda1([], legal)
    classes = read["move_classes"]
    nodes = 0
    proven: tuple[int, int] | None = None

    if read["verdict"] is not None:
        status = "win" if float(read["verdict"]) > 0.0 else "loss"
        source = "lambda1"
        if status == "win":
            proven = _win_now_move(legal, classes, replay, [])
    else:
        out = _timed_deep_solve(
            probe, [], config.root_node_cap, deadline - time.monotonic()
        )
        source = "deep"
        if out is None or out.get("timed_out"):
            # The clock, not the node cap, ended the solve; a harvested
            # cancel still reports the work it did.
            status = "timeout"
            if out is not None:
                nodes += int(out["nodes"])
        else:
            status = str(out["status"])
            nodes += int(out["nodes"])
            if status == "win" and out["move"] is not None:
                q, r = out["move"]
                proven = (int(q), int(r))

    line: list[tuple[int, int]] = []
    forced_through = 0
    if proven is not None:
        line, forced_through, walk_nodes = _walk_line(
            probe, replay, int(root.current_player), proven,
            config, deadline, line_cap,
        )
        nodes += walk_nodes

    return {
        "status": status,
        "source": source,
        "proven": {"q": proven[0], "r": proven[1]} if proven else None,
        "line": [{"q": q, "r": r} for q, r in line],
        "forced_through": forced_through,
        "guard": (
            None
            if classes is None
            else [
                {"q": int(q), "r": int(r), "cls": int(cls)}
                for (q, r), cls in zip(legal, classes)
            ]
        ),
        "nodes": nodes,
        "ms": round((time.monotonic() - started) * 1000.0, 3),
    }
