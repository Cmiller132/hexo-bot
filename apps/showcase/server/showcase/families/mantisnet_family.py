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
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

_SEED_MASK = (1 << 64) - 1

# Sequential-halving shape of the as-evaled search: candidates m is
# min(16, sims // 2, legal) in the research eval driver, with the 16 fixed.
_DEFAULT_CANDIDATES = 16
_DEFAULT_TEMPERATURE = 1.0

_INTROSPECTION_REASON = (
    "MantisNet lab internals (cell-attention rows, per-block norms, feature "
    "planes) are not wired yet"
)


def _pack(q: int, r: int) -> int:
    from hexo_engine.types import AxialCoord, pack_coord_id

    return int(pack_coord_id(AxialCoord(int(q), int(r))))


def _moves_from_state(state: Any) -> list[tuple[int, int]]:
    """The placement sequence of a showcase engine state, in play order.

    MantisNet's featurization consumes the move sequence (its attention rows
    are placement-ordered), so the family replays the history into its own
    vendored engine rather than reading the stone set.
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
        if profile_path is not None:
            with open(profile_path, "rb") as fh:
                raw = tomllib.load(fh)
            search = raw.get("search", {})
            candidates = int(search.get("candidates", candidates))
            temperature = float(search.get("temperature", temperature))
        if candidates < 1:
            raise ValueError(f"profile candidates must be >= 1, got {candidates}")
        if not (math.isfinite(temperature) and temperature >= 0.0):
            raise ValueError(f"profile temperature must be finite >= 0, got {temperature}")
        self.candidates = candidates
        self.temperature = temperature
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
        telemetry_callback: Any | None = None,
    ) -> dict:
        moves = _moves_from_state(state)
        return _run_search(
            evaluator,
            _replay(moves),
            candidates=self.candidates,
            visits=int(visits),
            seed=int(seed) & _SEED_MASK,
            temperature=float(temperature),
            game_key=int(game_key),
            telemetry_callback=telemetry_callback,
        )


def _run_search(
    evaluator: MantisnetEvaluator,
    root: Any,
    *,
    candidates: int,
    visits: int,
    seed: int,
    temperature: float,
    game_key: int,
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

    legal = root.legal_moves()
    legal_ids = [_pack(q, r) for q, r in legal]

    search = _rust.GumbelSearch(max(1, visits), candidates, temperature, seed)
    search.begin(root)

    root_read = None
    root_priors: list[float] = []
    emitted_start = False
    last_round = 0
    while True:
        decided, leaves = search.pump()
        snap = search.snapshot()

        if root_read is not None and not emitted_start and snap is not None:
            emitted_start = True
            emit(_start_frame(
                game_key, legal_ids, root_priors, float(root_read.value[0]), snap,
            ))
        if snap is not None and emitted_start and snap["round"] > last_round and not decided:
            last_round = snap["round"]
            emit(_round_frame(game_key, snap))

        if decided:
            break
        keys = [key for key, _leaf in leaves]
        read = evaluator.read([leaf for _key, leaf in leaves])
        if root_read is None:
            # The root wave: answer with the bare policy — the Gumbel MuZero
            # root-sampling convention of the research eval search.
            root_read = read
            for j, key in enumerate(keys):
                row = read.row(j)
                priors = torch.softmax(read.logits[row], dim=0).tolist()
                if j == 0:
                    root_priors = priors
                value = max(-1.0, min(1.0, float(read.value[j])))
                search.resume(key, priors, value)
        else:
            # Leaf waves: answer with pi-prime / v-hat, as the eval search does.
            for j, key in enumerate(keys):
                row = read.row(j)
                priors = read.improved.probs[row].tolist()
                value = max(-1.0, min(1.0, float(read.improved.v_hat[j])))
                search.resume(key, priors, value)

    move = search.decision()
    final = search.snapshot()
    if move is None:
        raise RuntimeError("the Gumbel session finished without a decision")
    action_id = _pack(*move)

    result: dict[str, Any] = {
        "action_id": action_id,
        "visits": int(final["completed_visits"]) if final else int(visits),
        "root_value": 0.0,
    }
    if final is not None:
        ids = [_pack(q, r) for q, r in final["actions"]]
        line_visits = [int(v) for v in final["visits"]]
        values = [float(v) for v in final["values"]]
        try:
            chosen = ids.index(action_id)
            result["root_value"] = values[chosen]
        except ValueError:
            # Zero-candidate budgets decide by prior argmax; the chosen move
            # never became a line. The root net value is the honest report.
            if root_read is not None:
                result["root_value"] = float(root_read.value[0])
        result["visit_policy_action_ids_bytes"] = _u32_bytes(ids)
        result["visit_policy_weights_bytes"] = _f32_bytes(float(v) for v in line_visits)
        result["visit_policy_q_bytes"] = _f32_bytes(values)
        result["visit_policy_count"] = len(ids)
        emit(_complete_frame(game_key, action_id, result, final))
    elif root_read is not None:
        result["root_value"] = float(root_read.value[0])
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
        "policy_count": len(ids),
        "survivor_action_ids_bytes": _u32_bytes(survivor_ids),
        "survivor_count": len(survivor_ids),
    }


def _complete_frame(
    game_key: int, action_id: int, result: dict, snap: dict,
) -> dict:
    return {
        "phase": "complete",
        "game_key": game_key,
        "round": int(snap["round"]),
        "rounds": int(snap["rounds"]),
        "visits": int(snap["completed_visits"]),
        "target_visits": int(snap["target_visits"]),
        "root_value": float(result["root_value"]),
        "action_id": action_id,
        "visit_policy_action_ids_bytes": result["visit_policy_action_ids_bytes"],
        "visit_policy_weights_bytes": result["visit_policy_weights_bytes"],
        "visit_policy_q_bytes": result["visit_policy_q_bytes"],
        "visit_policy_count": int(result["visit_policy_count"]),
        "policy_action_ids_bytes": result["visit_policy_action_ids_bytes"],
        "policy_weights_bytes": result["visit_policy_weights_bytes"],
        "policy_count": int(result["visit_policy_count"]),
    }


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
        evaluator = MantisnetEvaluator(model, _model_device(model))
        position = _replay(_moves_from_state(state))
        read = evaluator.read([position])
        rows = _policy_rows(read, position, floor=policy_floor)
        return {
            "value": float(read.value[0]),
            # MantisNet has no short-term-value or moves-left heads; null is
            # the frontend's "no data", never a substituted zero.
            "stv": None,
            "moves_left": None,
            "legal_count": position.legal_count,
            "policy": rows,
            "top_k": rows[:5],
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
        want_wire: bool,
    ) -> dict:
        import numpy as np

        result = _run_search(
            evaluator,
            _replay(moves),
            candidates=profile.candidates,
            visits=int(visits),
            seed=int(seed) & _SEED_MASK,
            temperature=0.0,
            game_key=int(game_key),
            telemetry_callback=None,
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
        return {
            "visits": int(result["visits"]),
            "root_value": float(result["root_value"]),
            "best": {"q": best_q, "r": best_r},
            "visit_policy": rows,
        }

    # -- summary ------------------------------------------------------------
    def summary_row(self, state: Any) -> list[tuple[int, int]]:
        return _moves_from_state(state)

    def summary_eval(self, model: Any, rows: list[Any]) -> dict:
        evaluator = MantisnetEvaluator(model, _model_device(model))
        values: list[float | None] = []
        # Fixed-size chunks bound peak memory: MantisNet's graph grows with
        # the board, and a whole game in one batch multiplies that by plies.
        chunk = 8
        for start in range(0, len(rows), chunk):
            block = rows[start : start + chunk]
            positions = [_replay(moves) for moves in block]
            read = evaluator.read(positions)
            values.extend(float(v) for v in read.value.tolist())
        return {
            "value": values,
            "stv": [None] * len(values),
            "moves_left": [None] * len(values),
        }

    # -- lab ----------------------------------------------------------------
    def lab_eval_payload(
        self, model: Any, *, actions, stones, to_move, policy_floor,
        attention_cell, want_activations, want_features,
    ) -> dict:
        if stones is not None:
            raise ValueError(
                "mantisnet cannot evaluate a free-edit position: its "
                "featurization consumes the placement sequence, which an "
                "unordered stone set does not determine"
            )
        moves = [(int(q), int(r)) for q, r in (actions or [])]
        position = _replay(moves)
        if position.is_terminal:
            raise ValueError("the position is terminal: nothing to evaluate")
        evaluator = MantisnetEvaluator(model, _model_device(model))
        read = evaluator.read([position])
        rows = _policy_rows(read, position, floor=policy_floor)
        improved_rows = [
            {"q": int(q), "r": int(r), "p": float(p)}
            for (q, r), p in zip(
                position.legal_moves(), read.improved.probs[read.row(0)].tolist()
            )
            if p >= policy_floor
        ]
        improved_rows.sort(key=lambda item: -item["p"])
        stones_list = position.stones()
        payload = {
            "mode": "sequence",
            "to_move": position.current_player,
            "phase": (
                "Opening" if position.stone_count == 0
                else ("FirstStone" if position.moves_remaining == 2 else "SecondStone")
            ),
            "ply": len(moves),
            "legal_count": position.legal_count,
            "support": {
                "coords": [
                    [int(q), int(r)] for q, r, _p in stones_list
                ] + [[int(q), int(r)] for q, r in position.legal_moves()],
                "legal_count": position.legal_count,
                "stone_count": position.stone_count,
                "halo_count": 0,
            },
            "value": float(read.value[0]),
            "value_dist": read.value_dist[0].tolist(),
            "stv": None,
            "moves_left": None,
            "v_hat": float(read.improved.v_hat[0]),
            "policy": rows,
            "improved_policy": improved_rows,
            "top_k": rows[:5],
            "attention": {"available": False, "reason": _INTROSPECTION_REASON},
            "activations": {"available": False, "reason": _INTROSPECTION_REASON},
        }
        if want_features:
            payload["features"] = {"available": False, "reason": _INTROSPECTION_REASON}
        return payload

    def lab_search_payload(
        self, session: Any, evaluator: Any, profile: MantisnetSearchProfile, *,
        actions, game_key: int, visits: int, seed: int,
    ) -> dict:
        moves = [(int(q), int(r)) for q, r in actions]
        return self._search_payload(
            evaluator, profile, moves,
            game_key=game_key, visits=visits, seed=seed, want_wire=True,
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
