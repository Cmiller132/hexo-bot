"""MantisNet family: the vendored research lineage served as a showcase bot.

The model, encoder, and Gumbel decision session live in ``packages/mantisnet``
(MODEL_REPR_VERSION 7). Unlike shrimp/hexfield_eq, the architecture rides the
checkpoint's ``model_config`` — nothing is frozen at import time — so the
process-preparation hooks are no-ops and any mix of MantisNet checkpoints can
share a worker.

Serve search semantics reproduce the research repo's evaluation search
(``mantisnet.klent.search.gumbel_choose``) exactly: the ROOT evaluation
answers with the bare policy (softmax of raw logits — the Gumbel MuZero
sampling convention), every LEAF answers with the improved policy π′ and its
expected value v̂. All comparisons the session makes are within one root, so
the log-softmax constant drops out and decisions match the as-evaled search.

Live telemetry is pull-based: the driver reads `GumbelSearch.snapshot()`
between evaluation waves and packs frames from tensors it already computed.
The session is never called from telemetry, no extra forward runs, and the
evaluator call schedule is identical with the overlay on or off — parity-safe
by construction.

Threat-Space Search rides the same seam (`mantis_tss.py`): the driver mirrors
each Gumbel line's position so it knows every leaf's placement path from the
root, and answers the session with proofs where proofs exist. TSS off is the
bare search above, byte for byte.
"""

from __future__ import annotations

import math
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .mantis_tss import TssConfig, TssRunner

_SEED_MASK = (1 << 64) - 1

# Sequential-halving shape of the as-evaled search: candidates m is
# min(16, sims // 2, legal) in the research eval driver, with the 16 fixed.
_DEFAULT_CANDIDATES = 16
_DEFAULT_TEMPERATURE = 1.0

_ATTN_QUERY_REASON = (
    "MantisNet's attention rows run over a global token plus the stones — "
    "pick a stone as the query"
)
_ATTN_FLOOR = 1e-3


def _pack(q: int, r: int) -> int:
    from hexo_engine.types import AxialCoord, pack_coord_id

    return int(pack_coord_id(AxialCoord(int(q), int(r))))


def _moves_from_state(state: Any) -> list[tuple[int, int]]:
    """The placement sequence of a showcase engine state, in play order.

    The vendored engine builds positions by replay only (a board-shaped
    constructor would bypass its rules), so the family carries the history
    across. The featurization itself is order-free: it reads the resulting
    position, never the sequence.
    """
    import hexo_engine as engine

    mirror = engine.to_python_state(state)
    return [(int(rec.coord.q), int(rec.coord.r)) for rec in mirror.placement_history]


def _replay(moves: list[tuple[int, int]]) -> Any:
    from mantisnet import _rust

    return _rust.Position.replay(moves)


def _u32_bytes(values) -> bytes:
    import numpy as np

    return np.asarray(list(values), dtype=np.uint32).tobytes()


def _f32_bytes(values) -> bytes:
    import numpy as np

    return np.asarray(list(values), dtype=np.float32).tobytes()


def _served_value(read: Any, row: int) -> float:
    """The model's acting-time value v̂ = E_{π′}[Q], side-to-move POV.

    Every human-facing "value" goes through here. The served checkpoints come
    from KLENT self-play, which trains the action-value pathway and never the
    state-value head — ``read.value`` is an UNTRAINED readout of the trained
    trunk on those models (smooth, structured, and meaningless; charted, it
    drew a sawtooth). v̂ is the evaluation the model actually acts on, in the
    same side-to-move frame the summary/analysis contract expects.
    """
    return float(read.improved.v_hat[row])


class MantisnetEvaluator:
    """Model + device + as-trained KLENT parameters, read through one seam."""

    def __init__(self, model: Any, device: str) -> None:
        self.model = model
        self.device = device
        self.klent = model.mantis_klent

    def read(self, positions) -> Any:
        from mantisnet import read_positions

        return read_positions(self.model, positions, self.klent, self.device)


class _MantisnetSession:
    """Per-checkpoint session handle.

    The Gumbel search rebuilds its candidate lines every move (the as-evaled
    convention), so there is no cross-move tree state to keep or discard;
    ``discard`` exists because the pool reclaims per-game state through it.
    """

    def discard(self, game_key: int) -> None:
        return None


class MantisnetSearchProfile:
    def __init__(
        self,
        profile_path: Path | None,
        *,
        opening_plies: int,
        opening_temperature: float | None,
    ) -> None:
        candidates, temperature = _DEFAULT_CANDIDATES, _DEFAULT_TEMPERATURE
        tss = TssConfig()
        if profile_path is not None:
            with open(profile_path, "rb") as fh:
                raw = tomllib.load(fh)
            search = raw.get("search", {})
            candidates = int(search.get("candidates", candidates))
            temperature = float(search.get("temperature", temperature))
            tss = TssConfig.from_profile(raw.get("tss"))
        if candidates < 1:
            raise ValueError(f"profile candidates must be >= 1, got {candidates}")
        if not (math.isfinite(temperature) and temperature >= 0.0):
            raise ValueError(f"profile temperature must be finite >= 0, got {temperature}")
        self.candidates = candidates
        self.temperature = temperature
        self.tss = tss
        self.opening_plies = int(opening_plies)
        self.opening_temperature = (
            float(opening_temperature)
            if opening_temperature is not None
            else _DEFAULT_TEMPERATURE
        )

    def move_temperature(self, ply: int) -> float:
        if ply < self.opening_plies and self.opening_temperature > 0.0:
            return self.opening_temperature * self.temperature
        return 0.0

    def search_one(
        self, session: Any, evaluator: Any, state: Any, *,
        game_key: int, visits: int, seed: int, temperature: float,
        tss_enabled: bool,
        telemetry_callback: Any | None = None,
    ) -> dict:
        return _run_search(
            evaluator,
            _moves_from_state(state),
            candidates=self.candidates,
            visits=int(visits),
            seed=int(seed) & _SEED_MASK,
            temperature=float(temperature),
            game_key=int(game_key),
            # The per-game toggle can turn TSS OFF; it never turns on what this
            # checkpoint's search profile disabled.
            tss=self.tss.with_enabled(self.tss.enabled and bool(tss_enabled)),
            telemetry_callback=telemetry_callback,
        )


class _Line:
    """The driver's mirror of one Gumbel candidate line.

    The session owns the real line positions and never shows them, so the
    driver keeps its own copy in lockstep — the position, and the placement
    path from the search root that produced it. The path is what a TSS solve
    names its position by.
    """

    __slots__ = ("position", "path", "policy_rank", "evaluated", "terminal")

    def __init__(self, position: Any, path: list[tuple[int, int]]) -> None:
        self.position = position
        self.path = path
        self.policy_rank: int | None = None
        self.evaluated = False
        self.terminal = bool(position.is_terminal)


def _prior_argmax(priors: list[float]) -> int:
    """First index of the maximum — the vendored session's `prior_argmax` tie
    rule, which decides how a line is extended."""
    best = 0
    for index in range(1, len(priors)):
        if priors[index] > priors[best]:
            best = index
    return best


def _init_lines(root: Any, snap: dict) -> list[_Line]:
    """Mirror the candidate set the root evaluation produced.

    `snapshot()["actions"]` is the candidates in Gumbel-top order, which is the
    session's own line order; each line's position is the root advanced by its
    candidate.
    """
    lines: list[_Line] = []
    for q, r in snap["actions"]:
        position = root.copy()
        position.advance(int(q), int(r))
        lines.append(_Line(position, [(int(q), int(r))]))
    return lines


def _advance_lines(lines: list[_Line], snap: dict) -> list[int]:
    """Replay one wave's line extensions and return the lines it will emit.

    The session extends every already-evaluated surviving line through the
    prior argmax it was last given, drops any line that goes terminal, and
    emits the rest — in survivor order. This mirrors that exactly.
    """
    expected: list[int] = []
    for index in snap["survivors"]:
        line = lines[int(index)]
        if line.terminal:
            continue
        if line.evaluated:
            if line.policy_rank is None:
                raise RuntimeError(
                    "TSS line mirror: an evaluated line has no recorded prior argmax"
                )
            q, r = line.position.nth_legal(line.policy_rank)
            line.position.advance(int(q), int(r))
            line.path.append((int(q), int(r)))
            if line.position.is_terminal:
                line.terminal = True
                continue
        expected.append(int(index))
    return expected


def _match_leaves(
    lines: list[_Line], expected: list[int], leaves: list
) -> list[int]:
    """Map each pumped leaf to its mirrored line, by Zobrist hash.

    Two lines can transpose to the same position, so a hash bucket may hold
    several candidates; the wave-order candidate is preferred, which is the
    session's own emission order. A leaf that matches no mirrored line, or a
    mirrored line that was never emitted, means the mirror has drifted from the
    session — a bug, not a position to guess at.
    """
    buckets: dict[int, list[int]] = {}
    for slot, index in enumerate(expected):
        buckets.setdefault(int(lines[index].position.zobrist), []).append(slot)
    order: list[int] = []
    for leaf_index, (_key, leaf) in enumerate(leaves):
        bucket = buckets.get(int(leaf.zobrist))
        if not bucket:
            raise RuntimeError(
                f"TSS line mirror: pumped leaf {leaf_index} (zobrist "
                f"{int(leaf.zobrist):#x}, {leaf.stone_count} stones) matches no "
                f"mirrored line of the {len(expected)} this wave expected"
            )
        slot = leaf_index if leaf_index in bucket else bucket[0]
        bucket.remove(slot)
        if not bucket:
            del buckets[int(leaf.zobrist)]
        order.append(expected[slot])
    if buckets:
        raise RuntimeError(
            f"TSS line mirror: {sum(len(b) for b in buckets.values())} mirrored "
            f"lines were not emitted by the wave"
        )
    return order


def _run_search(
    evaluator: MantisnetEvaluator,
    moves: list[tuple[int, int]],
    *,
    candidates: int,
    visits: int,
    seed: int,
    temperature: float,
    game_key: int,
    tss: TssConfig,
    telemetry_callback: Any | None,
) -> dict:
    """Drive one Gumbel search and assemble the showcase result dict."""
    import torch

    from mantisnet import _rust

    def emit(frame: dict) -> None:
        if telemetry_callback is None:
            return
        try:
            telemetry_callback(frame)
        except Exception:
            # Presentation is best-effort and must never fail the search.
            return

    root = _replay(moves)
    legal = root.legal_moves()
    legal_ids = [_pack(q, r) for q, r in legal]

    search = _rust.GumbelSearch(max(1, visits), candidates, temperature, seed)
    search.begin(root)

    runner = TssRunner(moves, tss) if tss.enabled else None
    try:
        return _drive_search(
            search, root, legal, legal_ids, evaluator, runner, emit,
            visits=visits, game_key=game_key, torch=torch,
        )
    finally:
        if runner is not None:
            runner.close()


def _drive_search(
    search: Any, root: Any, legal: list[tuple[int, int]], legal_ids: list[int],
    evaluator: MantisnetEvaluator, runner: TssRunner | None,
    emit: Any, *, visits: int, game_key: int, torch: Any,
) -> dict:
    root_read = None
    root_priors: list[float] = []
    emitted_start = False
    last_progress = (0, 0)
    lines: list[_Line] | None = None
    while True:
        decided, leaves = search.pump()
        snap = search.snapshot()

        if runner is not None and lines is None and snap is not None:
            lines = _init_lines(root, snap)
        if root_read is not None and not emitted_start and snap is not None:
            emitted_start = True
            emit(_start_frame(
                game_key, legal_ids, root_priors, _served_value(root_read, 0),
                snap,
            ))
        # One frame per completed wave, not just per halving: the deepening
        # rounds are many waves long at higher budgets, and without the
        # per-wave ticks the overlay froze between cuts. The (round, visits)
        # pair identifies the wave; the viewer coalesces same-round frames it
        # has not yet drawn.
        if snap is not None and emitted_start and not decided:
            progress = (int(snap["round"]), int(snap["completed_visits"]))
            if progress != last_progress and progress[1] > 0:
                last_progress = progress
                emit(_round_frame(game_key, snap))

        if decided:
            break
        keys = [key for key, _leaf in leaves]
        # TSS runs BEFORE the forward: λ¹ decides which leaves earn a deep
        # solve, and those solves are already in flight while the net runs.
        order: list[int] = []
        plans = None
        if runner is not None and root_read is not None:
            order = _match_leaves(lines, _advance_lines(lines, snap), leaves)
            plans = runner.plan_wave(
                [lines[index].path for index in order],
                [leaf.legal_moves() for _key, leaf in leaves],
            )
        read = evaluator.read([leaf for _key, leaf in leaves])
        if root_read is None:
            # The root wave: answer with the bare policy — the Gumbel MuZero
            # root-sampling convention of the research eval search.
            root_read = read
            for j, key in enumerate(keys):
                row = read.row(j)
                priors = torch.softmax(read.logits[row], dim=0).tolist()
                if j == 0:
                    # The start frame paints the model's policy; the search
                    # samples from the guarded copy below.
                    root_priors = priors
                    if runner is not None:
                        priors = runner.begin_root(legal, priors)
                value = max(-1.0, min(1.0, float(read.value[j])))
                search.resume(key, priors, value)
        else:
            # Leaf waves: answer with pi-prime / v-hat, as the eval search does.
            for j, key in enumerate(keys):
                row = read.row(j)
                priors = read.improved.probs[row].tolist()
                value = max(-1.0, min(1.0, float(read.improved.v_hat[j])))
                if plans is not None:
                    priors = runner.leaf_priors(plans[j], priors)
                    value = runner.leaf_value(plans[j], value)
                search.resume(key, priors, value)
                if plans is not None:
                    line = lines[order[j]]
                    line.policy_rank = _prior_argmax(priors)
                    line.evaluated = True

    move = search.decision()
    final = search.snapshot()
    if move is None:
        raise RuntimeError("the Gumbel session finished without a decision")

    proven: tuple[int, int] | None = None
    action_selection: str | None = None
    if runner is not None:
        proven = runner.decision_override()
        # The badge marks a proof that CHANGED the move, the same rule the
        # hexfield_eq root guard uses for `tss_deep_root_win`. A proof that
        # agrees with the search still carries its +1 into the export.
        changed = proven is not None and proven != (int(move[0]), int(move[1]))
        if changed:
            move = proven
        action_selection = "tss_deep_root_win" if changed else "gumbel_sh_score"
    action_id = _pack(*move)

    result: dict[str, Any] = {
        "action_id": action_id,
        "visits": int(final["completed_visits"]) if final else int(visits),
        "root_value": 0.0,
    }
    if action_selection is not None:
        result["action_selection"] = action_selection
    if runner is not None:
        result["tss"] = runner.stats()
    if final is not None:
        ids = [_pack(q, r) for q, r in final["actions"]]
        line_visits = [int(v) for v in final["visits"]]
        values = [float(v) for v in final["values"]]
        if proven is not None:
            # A proven win is worth +1 and the export must carry the cell the
            # bot actually played, even though no line ever searched it.
            if action_id in ids:
                values[ids.index(action_id)] = 1.0
            else:
                ids.append(action_id)
                line_visits.append(0)
                values.append(1.0)
        try:
            chosen = ids.index(action_id)
            result["root_value"] = values[chosen]
        except ValueError:
            # Zero-candidate budgets decide by prior argmax; the chosen move
            # never became a line. The root net value is the honest report.
            if root_read is not None:
                result["root_value"] = _served_value(root_read, 0)
        result["visit_policy_action_ids_bytes"] = _u32_bytes(ids)
        result["visit_policy_weights_bytes"] = _f32_bytes(float(v) for v in line_visits)
        result["visit_policy_q_bytes"] = _f32_bytes(values)
        result["visit_policy_count"] = len(ids)
        emit(_complete_frame(
            game_key, action_id, result, final,
            proven=proven, tss_stats=result.get("tss"),
        ))
    elif root_read is not None:
        result["root_value"] = _served_value(root_read, 0)
        result["visit_policy_action_ids_bytes"] = _u32_bytes(legal_ids)
        result["visit_policy_weights_bytes"] = _f32_bytes(root_priors)
        result["visit_policy_count"] = len(legal_ids)
    return result


def _start_frame(
    game_key: int, legal_ids: list[int], root_priors: list[float],
    root_value: float, snap: dict,
) -> dict:
    survivor_ids = [_pack(q, r) for q, r in snap["actions"]]
    return {
        "phase": "start",
        "game_key": game_key,
        "round": 0,
        "rounds": int(snap["rounds"]),
        "visits": 0,
        "target_visits": int(snap["target_visits"]),
        "root_value": root_value,
        "policy_action_ids_bytes": _u32_bytes(legal_ids),
        "policy_weights_bytes": _f32_bytes(root_priors),
        "policy_visits_bytes": _u32_bytes([0] * len(legal_ids)),
        "policy_count": len(legal_ids),
        "survivor_action_ids_bytes": _u32_bytes(survivor_ids),
        "survivor_count": len(survivor_ids),
    }


def _softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def _round_frame(game_key: int, snap: dict) -> dict:
    ids = [_pack(q, r) for q, r in snap["actions"]]
    survivors = set(int(i) for i in snap["survivors"])
    survivor_ids = [ids[i] for i in sorted(survivors)]
    weights = _softmax([float(s) for s in snap["scores"]])
    return {
        "phase": "round",
        "game_key": game_key,
        "round": int(snap["round"]),
        "rounds": int(snap["rounds"]),
        "visits": int(snap["completed_visits"]),
        "target_visits": int(snap["target_visits"]),
        "policy_action_ids_bytes": _u32_bytes(ids),
        "policy_weights_bytes": _f32_bytes(weights),
        "policy_visits_bytes": _u32_bytes(int(v) for v in snap["visits"]),
        # Per-line running Q rides every wave, not just the completion frame:
        # the viewer's value ticks show what the search is learning as it
        # learns it. Lines with zero visits carry a placeholder 0.0 the viewer
        # must gate on the visit count.
        "policy_q_bytes": _f32_bytes(float(v) for v in snap["values"]),
        "policy_count": len(ids),
        "survivor_action_ids_bytes": _u32_bytes(survivor_ids),
        "survivor_count": len(survivor_ids),
    }


def _complete_frame(
    game_key: int, action_id: int, result: dict, snap: dict,
    *, proven: tuple[int, int] | None = None,
    tss_stats: dict | None = None,
) -> dict:
    """The answer frame: the final SH ranking, not the raw visit counts.

    Sequential halving spreads visits almost evenly at the served budgets
    (at 16 sims every line has exactly one), so a visit-share paint lights
    the whole candidate set — the search looked like it never narrowed.
    The softmax of the final scores is the ranking the decision was actually
    taken from, and the final survivor set rides along so the viewer can dim
    the lines the last cut eliminated. Per-line visits and values stay as
    hover readouts.

    A TSS-proven root move joins the frame at weight 0 with value +1: the
    board must be able to draw the cell the bot played even when no line
    searched it.
    """
    ids = [_pack(q, r) for q, r in snap["actions"]]
    survivors = set(int(i) for i in snap["survivors"])
    survivor_ids = [ids[i] for i in sorted(survivors)]
    weights = _softmax([float(s) for s in snap["scores"]])
    line_visits = [int(v) for v in snap["visits"]]
    values = [float(v) for v in snap["values"]]
    if proven is not None:
        proven_id = _pack(*proven)
        if proven_id in ids:
            values[ids.index(proven_id)] = 1.0
        else:
            ids.append(proven_id)
            weights.append(0.0)
            line_visits.append(0)
            values.append(1.0)
        if proven_id not in survivor_ids:
            survivor_ids.append(proven_id)
    frame = {
        "phase": "complete",
        "game_key": game_key,
        "round": int(snap["round"]),
        "rounds": int(snap["rounds"]),
        "visits": int(snap["completed_visits"]),
        "target_visits": int(snap["target_visits"]),
        "root_value": float(result["root_value"]),
        "action_id": action_id,
        "policy_kind": "score",
        "policy_action_ids_bytes": _u32_bytes(ids),
        "policy_weights_bytes": _f32_bytes(weights),
        "policy_visits_bytes": _u32_bytes(line_visits),
        "policy_q_bytes": _f32_bytes(values),
        "policy_count": len(ids),
        "survivor_action_ids_bytes": _u32_bytes(survivor_ids),
        "survivor_count": len(survivor_ids),
    }
    if tss_stats is not None:
        frame["tss_stats"] = dict(tss_stats)
    return frame


def _policy_rows(read: Any, position: Any, *, floor: float) -> list[dict]:
    import torch

    row = read.row(0)
    probs = torch.softmax(read.logits[row], dim=0)
    legal = position.legal_moves()
    rows = [
        {"q": int(q), "r": int(r), "p": float(p)}
        for (q, r), p in zip(legal, probs.tolist())
        if p >= floor
    ]
    rows.sort(key=lambda item: -item["p"])
    return rows


@contextmanager
def _critic_capture(model: Any):
    """Capture the critic head's categorical logits from the next forward.

    ``read_positions`` composes the (N, 3) categorical head into Q before
    returning, but the analysis views also want the masses themselves
    (win / loss / long-game). The final linear of the critic decoder runs
    exactly once per forward, so a hook on it recovers the logits without
    diverging the vendored model.
    """
    captured: list[Any] = []
    hook = model.mlp_q.out.register_forward_hook(
        lambda _m, _i, out: captured.append(out.detach())
    )
    try:
        yield captured
    finally:
        hook.remove()


def _trinomial(captured: list, expected: int) -> Any:
    """The captured critic logits as (N, 3) fp32 CPU masses (p⁺, p⁻, p°)."""
    if len(captured) != 1:
        raise RuntimeError(
            f"critic capture saw {len(captured)} forwards, expected exactly 1"
        )
    masses = captured[0].float().softmax(dim=-1).cpu()
    if masses.shape[0] != expected:
        raise RuntimeError(
            f"critic capture rows {tuple(masses.shape)} do not cover the "
            f"batch's {expected} legal cells"
        )
    return masses


def _klent_cells(read: Any, masses: Any, position: Any, row: int = 0) -> dict:
    """Per-legal-cell KLENT readout, columnar over engine legal order.

    Full coverage, no floor: the board overlays paint every legal cell, and a
    low-prior cell's Q is exactly what the disagreement view looks for. The
    masses decode as win / loss / long-game from the side to move (§ appendix
    B: p⁺, p⁻, and the remaining zero-return mass of the same simplex).
    """
    import torch

    sl = read.row(row)
    prior = torch.softmax(read.logits[sl], dim=0)
    tri = masses[sl]

    def _r4(values: Any) -> list[float]:
        return [round(float(v), 4) for v in values.tolist()]

    return {
        "coords": [[int(q), int(r)] for q, r in position.legal_moves()],
        "prior": _r4(prior),
        "improved": _r4(read.improved.probs[sl]),
        "q": _r4(read.q_values[sl]),
        "win": _r4(tri[:, 0]),
        "loss": _r4(tri[:, 1]),
        "long": _r4(tri[:, 2]),
        "kl": round(float(read.improved.kl[row]), 4),
        "norm_entropy": round(float(read.improved.norm_entropy[row]), 4),
    }


class MantisnetFamily:
    name = "mantisnet"
    supports_live_telemetry = True

    # -- process preparation ------------------------------------------------
    def prepare_process(self, specs) -> None:
        # Architecture rides each checkpoint's model_config; nothing to freeze.
        return None

    def prepare_serve_process(self, device: str) -> None:
        # The port's custom ops gate on CUDA at call time; CPU and XPU take
        # the eager reference paths with no import-time switches.
        return None

    # -- loading ------------------------------------------------------------
    def load_net(self, spec) -> Any:
        from mantisnet import load_checkpoint

        loaded = load_checkpoint(spec.checkpoint)
        model = loaded.model
        model.mantis_klent = loaded.klent
        model.mantis_provenance = loaded.provenance
        return model

    def build_evaluator(self, model: Any, device: str) -> MantisnetEvaluator:
        return MantisnetEvaluator(model, device)

    def build_session(self) -> _MantisnetSession:
        return _MantisnetSession()

    def build_profile(self, profile_path: Path | None, settings: Any) -> MantisnetSearchProfile:
        return MantisnetSearchProfile(
            profile_path,
            opening_plies=settings.opening_plies,
            opening_temperature=settings.opening_temperature,
        )

    # -- engine glue --------------------------------------------------------
    def decode_action(self, action_id: int) -> tuple[int, int]:
        from hexo_engine.types import unpack_coord_id

        coord = unpack_coord_id(int(action_id))
        return int(coord.q), int(coord.r)

    # -- readouts -----------------------------------------------------------
    def net_eval(self, model: Any, state: Any, *, policy_floor: float) -> dict:
        position = _replay(_moves_from_state(state))
        if position.is_terminal:
            # The builder refuses terminal positions (MantisNet only ever
            # evaluates decision states), and the game is decided anyway.
            # Null is the frontend's "no data" contract.
            return {
                "value": None, "stv": None, "moves_left": None,
                "legal_count": 0, "policy": [], "top_k": [],
            }
        evaluator = MantisnetEvaluator(model, _model_device(model))
        with _critic_capture(model) as captured:
            read = evaluator.read([position])
        masses = _trinomial(captured, int(read.offsets[-1]))
        rows = _policy_rows(read, position, floor=policy_floor)
        return {
            "value": _served_value(read, 0),
            # MantisNet has no short-term-value or moves-left heads; null is
            # the frontend's "no data", never a substituted zero.
            "stv": None,
            "moves_left": None,
            "legal_count": position.legal_count,
            "policy": rows,
            "top_k": rows[:5],
            "klent": _klent_cells(read, masses, position),
        }

    def searched_eval(
        self, session: Any, evaluator: Any, profile: MantisnetSearchProfile,
        state: Any, *, game_key: int, visits: int, seed: int,
    ) -> dict:
        return self._search_payload(
            evaluator, profile, _moves_from_state(state),
            game_key=game_key, visits=visits, seed=seed, want_wire=False,
        )

    def _search_payload(
        self, evaluator: Any, profile: MantisnetSearchProfile,
        moves: list[tuple[int, int]], *, game_key: int, visits: int, seed: int,
        want_wire: bool, want_frames: bool = False,
    ) -> dict:
        import numpy as np

        # `want_frames` collects the same telemetry frames the live-search SSE
        # streams, so the caller can replay the search in the live viewer.
        frames: list[dict] = []
        # No per-game toggle reaches analysis or the lab: those readouts follow
        # the profile, so a served analysis matches how the bot plays.
        result = _run_search(
            evaluator,
            moves,
            candidates=profile.candidates,
            visits=int(visits),
            seed=int(seed) & _SEED_MASK,
            temperature=0.0,
            game_key=int(game_key),
            tss=profile.tss,
            telemetry_callback=frames.append if want_frames else None,
        )
        best_q, best_r = self.decode_action(int(result["action_id"]))
        ids = np.frombuffer(result["visit_policy_action_ids_bytes"], dtype=np.uint32)
        weights = np.frombuffer(result["visit_policy_weights_bytes"], dtype=np.float32)
        total = float(weights.sum()) or 1.0
        rows = []
        for aid, w in zip(ids.tolist(), weights.tolist()):
            q, r = self.decode_action(int(aid))
            entry = {"q": q, "r": r, "p": float(w) / total}
            if want_wire:
                entry["w"] = float(w)
            rows.append(entry)
        rows.sort(key=lambda item: -item["p"])
        payload = {
            "visits": int(result["visits"]),
            "root_value": float(result["root_value"]),
            "best": {"q": best_q, "r": best_r},
            "visit_policy": rows,
        }
        if want_frames:
            payload["frames_raw"] = frames
        return payload

    # -- summary ------------------------------------------------------------
    def summary_row(self, state: Any) -> list[tuple[int, int]]:
        return _moves_from_state(state)

    def summary_eval(self, model: Any, rows: list[Any]) -> dict:
        evaluator = MantisnetEvaluator(model, _model_device(model))
        positions = [_replay(moves) for moves in rows]
        values: list[float | None] = [None] * len(positions)
        # Q of the move the game actually played from row i, and the best
        # legal Q — both side-to-move POV at row i, both from the critic head
        # the same forward already computed. The gap is the blunder signal
        # the analysis chart marks.
        played_q: list[float | None] = [None] * len(positions)
        best_q: list[float | None] = [None] * len(positions)
        # The final row of a finished game is terminal; the builder refuses
        # terminal positions (MantisNet only ever evaluates decision states),
        # so that row stays null — the frontend's "no data".
        live = [(i, pos) for i, pos in enumerate(positions) if not pos.is_terminal]
        # Chunks are sized by total legal cells, the term MantisNet's padded
        # graph actually grows with: shallow openings batch wide, deep boards
        # forward nearly alone, and peak memory stays bounded on the 4 GB
        # card. At least one row always forms a chunk.
        budget = 32_000
        start = 0
        while start < len(live):
            end, total = start, 0
            while end < len(live):
                cost = max(1, live[end][1].legal_count)
                if end > start and total + cost > budget:
                    break
                total += cost
                end += 1
            read = evaluator.read([pos for _i, pos in live[start:end]])
            for j, (i, pos) in enumerate(live[start:end]):
                values[i] = _served_value(read, j)
                q_row = read.q_values[read.row(j)]
                best_q[i] = round(float(q_row.max()), 4)
                if i + 1 < len(rows):
                    q, r = (int(v) for v in rows[i + 1][-1])
                    try:
                        slot = pos.legal_moves().index((q, r))
                    except ValueError as exc:
                        raise RuntimeError(
                            f"summary row {i}: played move ({q}, {r}) is not "
                            "among the position's legal cells"
                        ) from exc
                    played_q[i] = round(float(q_row[slot]), 4)
            start = end
        return {
            "value": values,
            "stv": [None] * len(values),
            "moves_left": [None] * len(values),
            "played_q": played_q,
            "best_q": best_q,
        }

    # -- lab ----------------------------------------------------------------
    def lab_eval_payload(
        self, model: Any, *, actions, stones, to_move, policy_floor,
        attention_cell, want_activations, want_features,
    ) -> dict:
        if stones is not None:
            # MantisNet's representation is a function of the stone set, the
            # mover, and moves_remaining — no history anywhere in it — so a
            # free-edit position evaluates exactly (see mantis_free).
            from .mantis_free import FreeLabPosition

            p0, p1 = stones
            position = FreeLabPosition(p0, p1, 0 if to_move is None else int(to_move))
            moves = None
            mode = "free"
        else:
            moves = [(int(q), int(r)) for q, r in actions]
            position = _replay(moves)
            mode = "sequence"
        if position.is_terminal:
            raise ValueError("the position is terminal: nothing to evaluate")
        evaluator = MantisnetEvaluator(model, _model_device(model))
        with _critic_capture(model) as captured:
            read = evaluator.read([position])
        masses = _trinomial(captured, int(read.offsets[-1]))
        rows = _policy_rows(read, position, floor=policy_floor)
        stones_list = position.stones()
        # Support order is the lab's shared convention: legal cells first,
        # then stones (the client slices coords[:legal_count] as the legal
        # set). Node indices in the internals payload follow it.
        support_coords = [
            [int(q), int(r)] for q, r in position.legal_moves()
        ] + [[int(q), int(r)] for q, r, _p in stones_list]
        payload = {
            "mode": mode,
            "to_move": position.current_player,
            "phase": (
                "Opening" if position.stone_count == 0
                else ("FirstStone" if position.moves_remaining == 2 else "SecondStone")
            ),
            "ply": len(moves) if moves is not None else position.stone_count,
            "legal_count": position.legal_count,
            "support": {
                "coords": support_coords,
                "legal_count": position.legal_count,
                "stone_count": position.stone_count,
                "halo_count": 0,
            },
            # v̂ is the one value served (see _served_value); the untrained
            # state head's scalar and bin distribution are deliberately absent
            # — the lab must not present noise as an internal worth reading.
            "value": _served_value(read, 0),
            "stv": None,
            "moves_left": None,
            "policy": rows,
            "top_k": rows[:5],
            "klent": _klent_cells(read, masses, position),
        }
        if mode == "free":
            # Nothing in MantisNet's input derives from history, so a
            # free-edit position zeroes no features (unlike shrimp's).
            payload["zeroed_features"] = []
        payload.update(
            _lab_internals(
                model, position, support_coords,
                moves=moves,
                attention_cell=attention_cell,
                want_activations=want_activations,
                want_features=want_features,
            )
        )
        return payload

    def lab_search_payload(
        self, session: Any, evaluator: Any, profile: MantisnetSearchProfile, *,
        actions, game_key: int, visits: int, seed: int,
        want_frames: bool = False,
    ) -> dict:
        moves = [(int(q), int(r)) for q, r in actions]
        return self._search_payload(
            evaluator, profile, moves,
            game_key=game_key, visits=visits, seed=seed, want_wire=True,
            want_frames=want_frames,
        )

    # -- device -------------------------------------------------------------
    def selfcheck_forward(self, model: Any, state: Any, device: str) -> dict:
        evaluator = MantisnetEvaluator(model, device)
        read = evaluator.read([_replay(_moves_from_state(state))])
        return {"value": read.value, "policy": read.logits}

    def selfcheck_autocast(self, device: str) -> bool:
        # The port serves fp32 eager on every device; no autocast anywhere.
        return False

    def warmup(self, model: Any, device: str) -> None:
        # The parity self-check already ran a serve forward on the device, and
        # MantisNet batches are ragged — there is no fixed shape to pre-JIT.
        return None


def _model_device(model: Any) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def _lab_internals(
    model: Any, position: Any, support_coords: list[list[int]], *,
    moves: list[tuple[int, int]] | None,
    attention_cell: tuple[int, int] | None,
    want_activations: bool,
    want_features: bool,
) -> dict:
    """Attention rows, per-block activation norms, and feature planes.

    One hooked trunk forward, mirroring the research deck's attention
    inspector: q/k are captured from each block's own projections and the
    bias logits are rebuilt with the model's own bucket rule, so the
    displayed rows are the attention the forward actually ran. Support-node
    indexing: legal cells in engine order, then stones in canonical order
    (the lab's shared support convention).
    """
    import math as _math

    import torch

    from mantisnet import collate_positions

    device = _model_device(model)
    batch = collate_positions([position]).to(device)
    cfg = model.cfg

    captures: list[dict[str, Any]] = [{} for _ in range(cfg.blocks)]
    block_outputs: list[tuple[Any, Any]] = []
    hooks = []
    for index, block in enumerate(model.blocks):
        hooks.append(block.wq.register_forward_hook(
            lambda _m, _i, out, index=index: captures[index].__setitem__("q", out.detach())
        ))
        hooks.append(block.wk.register_forward_hook(
            lambda _m, _i, out, index=index: captures[index].__setitem__("k", out.detach())
        ))
        hooks.append(block.register_forward_hook(
            lambda _m, _i, out: block_outputs.append((out[0].detach(), out[3]))
        ))
    try:
        with torch.no_grad():
            _s, _w, _g, cells = model.trunk(batch)
    finally:
        for hook in hooks:
            hook.remove()

    support_index = {(c[0], c[1]): i for i, c in enumerate(support_coords)}
    # The attention sequence is [four state-latent rows; stones] (model.py
    # §5.3 hardcodes the same four).
    global_rows = 4
    length = int(batch.attn_valid[0].sum())
    seq_coords = batch.coords[0, :length].cpu()

    out: dict[str, Any] = {}

    # -- attention rows ------------------------------------------------------
    if attention_cell is None:
        out["attention"] = {"available": True, "blocks": cfg.blocks, "heads": cfg.heads}
    else:
        query_seq = None
        for i in range(global_rows, length):
            if (int(seq_coords[i, 0]), int(seq_coords[i, 1])) == tuple(attention_cell):
                query_seq = i
                break
        if query_seq is None:
            out["attention"] = {"available": False, "reason": _ATTN_QUERY_REASON}
        else:
            # The model's own bucket rule and bias table (SELF diagonal,
            # TOKEN on every latent pair, axis rows replacing the distance
            # row for aligned pairs), so the displayed softmax is exactly
            # the one the forward ran.
            from mantisnet.attention import _bias_table, _bucket_index

            dim = cfg.h // cfg.heads
            buckets, _valid = _bucket_index(
                seq_coords[None, :, :],
                torch.tensor([length]),
                length,
                cfg.d_max,
                global_rows,
            )
            rows: list[list[dict]] = []
            for index, block in enumerate(model.blocks):
                q = captures[index]["q"][0, :length].float().cpu()
                k = captures[index]["k"][0, :length].float().cpu()
                q = q.view(length, cfg.heads, dim).transpose(0, 1)
                k = k.view(length, cfg.heads, dim).transpose(0, 1)
                logits = q @ k.transpose(-1, -2) / _math.sqrt(dim)
                table = _bias_table(
                    q,
                    block.dist_bias.detach().float().cpu(),
                    block.axis_bias.detach().float().cpu(),
                )
                logits += table[:, buckets[0].long()]
                weights = torch.softmax(logits[:, query_seq, :], dim=-1)
                head_rows = []
                for head in range(cfg.heads):
                    row = weights[head]
                    cells_map = {}
                    for i in range(global_rows, length):
                        w_i = float(row[i])
                        if w_i < _ATTN_FLOOR:
                            continue
                        node = support_index.get(
                            (int(seq_coords[i, 0]), int(seq_coords[i, 1]))
                        )
                        if node is not None:
                            cells_map[str(node)] = w_i
                    head_rows.append(
                        {
                            "cells": cells_map,
                            "tokens": [float(row[t]) for t in range(global_rows)],
                        }
                    )
                rows.append(head_rows)
            out["attention"] = {
                "available": True,
                "blocks": cfg.blocks,
                "heads": cfg.heads,
                "floor": _ATTN_FLOOR,
                "rows": rows,
            }

    # -- per-stage activation norms -----------------------------------------
    if want_activations:
        # Flat stone order -> support node, via each stone's sequence slot.
        stone_slots = batch.stone_slot.cpu()
        max_t = batch.coords.shape[1]
        stone_nodes = []
        for i in range(stone_slots.shape[0]):
            t = int(stone_slots[i]) % max_t
            coord = (int(batch.coords[0, t, 0]), int(batch.coords[0, t, 1]))
            stone_nodes.append(support_index.get(coord))

        def stage_norms(s_stream, cell_stream) -> list[float]:
            norms = [0.0] * len(support_coords)
            if s_stream is not None:
                s_norm = s_stream.float().norm(dim=-1).cpu()
                for i, node in enumerate(stone_nodes):
                    if node is not None:
                        norms[node] = float(s_norm[i])
            if cell_stream is not None:
                c_norm = cell_stream.float().norm(dim=-1).cpu()
                for rank in range(c_norm.shape[0]):
                    norms[rank] = float(c_norm[rank])
            return norms

        with torch.no_grad():
            # The stem is the embedding tables the trunk starts from;
            # recomputing the lookups reproduces it exactly.
            stem_stones = model.stone_table(batch.stone_own)
            stem_cells = None
            if getattr(model, "cell_occupancy_table", None) is not None:
                stem_cells = (
                    model.cell_occupancy_table(batch.cell_occupancy)
                    + model.cell_legal_table(batch.cell_is_legal)
                    + model.cell_nearest_table(batch.cell_nearest)
                )
        stages = [
            {
                "label": "stem",
                "kind": "stem",
                "norms": stage_norms(stem_stones, stem_cells),
            }
        ]
        for index, (s_stream, cell_stream) in enumerate(block_outputs):
            stages.append(
                {
                    "label": f"block {index + 1}",
                    "kind": "attn",
                    "norms": stage_norms(s_stream, cell_stream),
                }
            )
        stages.append(
            {"label": "output (shared LN)", "kind": "attn",
             "norms": stage_norms(_s, cells)}
        )
        out["activations"] = {"available": True, "blocks": stages}
    else:
        out["activations"] = {"available": True}

    # -- feature planes ------------------------------------------------------
    if want_features:
        n = len(support_coords)
        planes: dict[str, list[float]] = {
            "is_stone": [0.0] * n,
            "stone_own": [0.0] * n,
            # placement_frac is presentation only — the model never sees play
            # order (the batch's stone order is canonical, not chronological).
            # It comes from the request's placement list; a free-edit position
            # has no chronology, so the plane is absent there.
            **({"placement_frac": [0.0] * n} if moves is not None else {}),
            "cell_occupancy": [0.0] * n,
            "cell_nearest": [0.0] * n,
        }
        stones = position.stones()
        mover = position.current_player
        total = max(1, len(stones))
        order = {}
        if moves is not None:
            for i, (mq, mr) in enumerate(moves):
                order[(int(mq), int(mr))] = i + 1
        for q, r, player in stones:
            node = support_index[(int(q), int(r))]
            planes["is_stone"][node] = 1.0
            planes["stone_own"][node] = 1.0 if int(player) == mover else -1.0
            seq = order.get((int(q), int(r)))
            if seq is not None:
                planes["placement_frac"][node] = seq / total
        occupancy = batch.cell_occupancy.cpu()
        nearest = batch.cell_nearest.cpu()
        for rank in range(occupancy.shape[0]):
            planes["cell_occupancy"][rank] = float(occupancy[rank])
            planes["cell_nearest"][rank] = float(nearest[rank])
        out["features"] = {
            "available": True,
            "names": list(planes),
            "planes": [planes[name] for name in planes],
        }
    return out
