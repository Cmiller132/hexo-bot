"""hexfield_eq checkpoint, search, and analysis adapter."""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import Any, Sequence

_SEED_MASK = (1 << 63) - 1
log = logging.getLogger(__name__)
_META_ENV = {
    "channels": "HEXFIELD_EQ_CHANNELS",
    "group_order": "HEXFIELD_EQ_GROUP_ORDER",
    "c_orbit": "HEXFIELD_EQ_C_ORBIT",
    "attention_heads": "HEXFIELD_EQ_ATTENTION_HEADS",
    "support_radius": "HEXFIELD_EQ_SUPPORT_RADIUS",
    "trunk_layout": "HEXFIELD_EQ_TRUNK",
    "reg_lane": "HEXFIELD_EQ_REG_LANE",
    "reg_tok_read": "HEXFIELD_EQ_REG_TOK_READ",
    "cell_q": "HEXFIELD_EQ_CELL_Q",
    "feature_version": "HEXFIELD_EQ_FEATURE_VERSION",
    "raytap": "HEXFIELD_EQ_RAYTAP",
    "ray_blockers": "HEXFIELD_EQ_RAY_BLOCKERS",
}


def _env_value(value: Any) -> str:
    return "1" if value is True else "0" if value is False else str(value)


def _arch_from_meta(meta: Any, checkpoint: Path) -> dict[str, str]:
    if not isinstance(meta, dict):
        raise RuntimeError(f"hexfield_eq checkpoint has no architecture meta: {checkpoint}")
    missing = [
        key
        for key in _META_ENV
        if key not in ("ray_blockers", "cell_q") and key not in meta
    ]
    if missing:
        raise RuntimeError(
            f"hexfield_eq checkpoint meta is missing architecture keys {missing}: {checkpoint}"
        )
    return {
        env_name: _env_value(meta.get(key, True))
        for key, env_name in _META_ENV.items()
    }


class HexfieldEqSearchProfile:
    def __init__(self, config_path: Path | str) -> None:
        from hexfield_eq.config import (
            build_divergence_overrides,
            parse_hexfield_config,
        )

        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        model_cfg = raw.get("model", {}).get("config", {})
        self.cfg = parse_hexfield_config(
            {
                "device": "cpu",
                "selfplay": model_cfg.get("selfplay", {}),
                "multi_stage_eval": model_cfg.get("multi_stage_eval", {}),
            }
        )
        self.selfplay = self.cfg.selfplay
        self.overrides = build_divergence_overrides(self.selfplay)
        mse = self.cfg.multi_stage_eval
        self.virtual_batch_size = int(mse.eval_virtual_batch_size or 32)
        self.opening_plies = int(mse.opening_plies)
        self.opening_temperature = float(mse.opening_temperature)

    def move_temperature(self, ply: int) -> float:
        if ply < self.opening_plies and self.opening_temperature > 0.0:
            return self.opening_temperature
        return 0.0

    def search_one(
        self, session: Any, evaluator: Any, state: Any, *,
        game_key: int, visits: int, seed: int, temperature: float,
    ) -> dict:
        from hexfield_eq.config import build_eval_search_kwargs

        kwargs = build_eval_search_kwargs(
            self.selfplay,
            visits=int(visits),
            virtual_batch_size=self.virtual_batch_size,
            active_root_limit=self.selfplay.active_root_limit,
        )
        return session.search(
            [int(game_key)],
            (state,),
            seed=int(seed) & _SEED_MASK,
            evaluator=evaluator,
            move_temperatures=[float(temperature)],
            divergence_overrides=self.overrides,
            **kwargs,
        )[0]


class HexfieldEqFamily:
    name = "hexfield_eq"

    _FREE_ZEROED = ("own_recency", "opp_recency", "opp_last_turn")
    _INTROSPECTION_REASON = (
        "hexfield_eq uses equivariant coordinate/pair and ray attention plus "
        "ray-tap trunk blocks; a faithful Lab visualization is not yet available"
    )

    def prepare_process(self, specs: Sequence[Any]) -> None:
        """Seed import-frozen HEXFIELD_EQ_* constants from checkpoint metadata."""
        if not specs:
            return
        import torch

        arches: list[tuple[Path, dict[str, str]]] = []
        for spec in specs:
            payload = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise RuntimeError(f"hexfield_eq checkpoint payload is not a dict: {spec.checkpoint}")
            arches.append((spec.checkpoint, _arch_from_meta(payload.get("meta"), spec.checkpoint)))
        expected = arches[0][1]
        conflicts = [str(path) for path, arch in arches[1:] if arch != expected]
        if conflicts:
            raise RuntimeError(
                "a worker process can host only one hexfield_eq arch; split the catalogue "
                f"(conflicting checkpoint(s): {', '.join(conflicts)})"
            )
        os.environ.update(expected)

    def prepare_serve_process(self, device: str) -> None:
        """Prime backend-appropriate import-time gates before model import.

        ``prepare_process`` has already installed checkpoint architecture env
        by the time this hook runs, so importing the lightweight serve-env
        module is safe. CUDA gets the established full serve profile. XPU keeps
        CUDA-only Triton gates off and enables FlexAttention only under the
        explicit parity/perf probe flag.
        """
        from hexfield_eq.serve_env import prime_serve_env_for_device

        applied = prime_serve_env_for_device(device)
        if applied:
            log.info(
                "hexfield_eq import-time serve gates for %s: %s",
                device,
                ", ".join(sorted(applied)),
            )
        elif str(device).split(":", 1)[0].lower() == "xpu":
            log.info(
                "hexfield_eq XPU uses materialized fp32 attention; CUDA-only "
                "Triton gates remain off (set HEXFIELD_XPU_FLEX=1 only for "
                "the A310 parity/perf probe)"
            )

    def load_net(self, spec: Any) -> Any:
        from hexfield_eq.eval_arena import _load_hexfield_net

        return _load_hexfield_net(spec.checkpoint)

    def build_evaluator(self, model: Any, device: str) -> Any:
        from hexfield_eq.config import parse_hexfield_config
        from hexfield_eq.inference import build_serve_evaluator

        cfg = parse_hexfield_config({"device": device, "selfplay": {}})
        evaluator = build_serve_evaluator(model, cfg, role="eval")
        log.info(
            "hexfield_eq evaluator: device=%s rust_pack=%s defer_decode=%s "
            "host_legal_gather=%s decode_cache=%s",
            device,
            evaluator._rust_pack,
            evaluator._defer_decode,
            evaluator._host_legal_gather,
            evaluator._decode_cache,
        )
        return evaluator

    def build_session(self) -> Any:
        from hexfield_eq import _rust

        return _rust.HexfieldMctsSession(max_states=65_536)

    def build_profile(self, profile_path: Path | None, settings: Any) -> HexfieldEqSearchProfile:
        return HexfieldEqSearchProfile(profile_path or Path(settings.search_config))

    def decode_action(self, action_id: int) -> tuple[int, int]:
        from hexfield_eq.geometry import unpack_action_id

        return unpack_action_id(action_id)

    @staticmethod
    def _row(state: Any) -> tuple[Any, Any, Any]:
        import numpy as np
        from hexfield_eq.engine_facts import facts_from_state
        from hexfield_eq.features import build_position, build_ray_lengths

        facts = facts_from_state(state)
        support, features = build_position(facts)
        features = features.astype(np.float16).astype(np.float32)
        return support, features, build_ray_lengths(facts, support)

    @staticmethod
    def _forward_rows(
        model: Any,
        rows: list[tuple[Any, Any, Any]],
        device: str | None = None,
        *,
        serve_path: bool = False,
        full_heads: bool = False,
    ) -> dict:
        import torch
        from hexfield_eq.batching import collate_rows

        target = torch.device(device) if device is not None else next(model.parameters()).device
        batch = collate_rows(
            [(support, features) for support, features, _ in rows],
            raylen=[raylen for _, _, raylen in rows],
        )
        feats = batch["feats"]
        # HexfieldEvaluator keeps wire-rounded features in fp16 and autocasts
        # the CUDA forward. CPU/XPU serve remains fp32.
        cuda_serve = serve_path and target.type == "cuda"
        if cuda_serve:
            feats = feats.half()
        with torch.no_grad(), torch.autocast(
            device_type=target.type, dtype=torch.float16, enabled=cuda_serve
        ):
            args = (
                feats.to(target), batch["nbr"].to(target),
                batch["mask"].to(target), batch["coords"].to(target),
            )
            if full_heads:
                # The current serve forward intentionally omits STV and the
                # training-only policy heads. Analysis/Lab are readout paths,
                # so use the model's full forward to obtain the real heads.
                out = model.forward(*args, raylen=batch["raylen"].to(target))
            else:
                out = model.forward_policy_value(
                    *args, request_moves_left=True,
                    raylen=batch["raylen"].to(target),
                )
        return {key: value.detach() for key, value in out.items()}

    @staticmethod
    def _stv_values(out: dict, row: int = 0) -> dict[str, float]:
        from hexfield_eq.losses import decode_binned_value
        from hexfield_eq.model import STV_HORIZONS

        return {
            str(h): round(float(decode_binned_value(
                out[f"stvalue_{h}"][row].reshape(1, -1).float()
            ).item()), 6)
            for h in STV_HORIZONS
        }

    def net_eval(self, model: Any, state: Any, *, policy_floor: float) -> dict:
        import torch
        from hexfield_eq.losses import decode_binned_value, decode_moves_left
        from ..jsonsafe import sanitize_json

        row = self._row(state)
        support = row[0]
        out = self._forward_rows(model, [row], full_heads=True)
        value = float(decode_binned_value(out["value"].float())[0].item())
        moves_left = float(decode_moves_left(out["moves_left"].float())[0].item())
        stv = self._stv_values(out)
        policy: list[dict[str, Any]] = []
        if support.legal_count:
            priors = torch.softmax(out["policy"][0, : support.legal_count].float(), dim=0)
            policy = [
                {"q": int(q), "r": int(r), "p": round(float(p), 6)}
                for (q, r), p in zip(support.legal_coords().tolist(), priors.cpu().tolist())
            ]
            policy.sort(key=lambda item: item["p"], reverse=True)
        return sanitize_json({
            "value": value,
            # Analysis has historically charted shrimp's shortest horizon as
            # a scalar. Keep that wire contract and expose all trained
            # hexfield horizons alongside it.
            "stv": stv["2"],
            "stv_horizons": stv,
            "moves_left": round(moves_left, 3),
            "legal_count": int(support.legal_count),
            "policy": [item for item in policy if item["p"] >= policy_floor],
            "top_k": policy[:5],
        })

    def summary_row(self, state: Any) -> Any:
        return self._row(state)

    def searched_eval(
        self, session: Any, evaluator: Any, profile: HexfieldEqSearchProfile,
        state: Any, *, game_key: int, visits: int, seed: int,
    ) -> dict:
        payload = self._search_payload(
            session, evaluator, profile, state,
            game_key=game_key, visits=visits, seed=seed,
        )
        for row in payload["visit_policy"]:
            row.pop("w", None)
        return payload

    def summary_eval(self, model: Any, rows: list[Any]) -> dict:
        from hexfield_eq.losses import decode_binned_value, decode_moves_left
        from ..jsonsafe import sanitize_json

        # Keep whole-game analysis memory-bounded. Hexfield support grows with
        # the game; padding every ply into one attention batch can explode
        # B*S^2 even though each individual position fits comfortably.
        values: list[float] = []
        stvs: list[float] = []
        stv_horizons: list[dict[str, float]] = []
        moves: list[float] = []
        for row in rows:
            out = self._forward_rows(model, [row], full_heads=True)
            values.append(round(float(decode_binned_value(out["value"].float())[0].item()), 6))
            decoded_stv = self._stv_values(out)
            stvs.append(decoded_stv["2"])
            stv_horizons.append(decoded_stv)
            moves.append(round(float(decode_moves_left(out["moves_left"].float())[0].item()), 3))
        return sanitize_json({
            "value": values, "stv": stvs,
            "stv_horizons": stv_horizons, "moves_left": moves,
        })

    @staticmethod
    def _replay_state(cells: list[tuple[int, int]]) -> Any:
        import hexo_engine as engine
        from hexo_engine.types import AxialCoord, PlacementAction

        state = engine.new_game()
        for q, r in cells:
            engine.apply_action(state, PlacementAction(AxialCoord(q=int(q), r=int(r))))
        return state

    @classmethod
    def _lab_row(
        cls, actions: list[tuple[int, int]] | None,
        stones: tuple[list[tuple[int, int]], list[tuple[int, int]]] | None,
        to_move: int | None,
    ) -> tuple[Any, Any, Any, Any, str]:
        import numpy as np
        from hexfield_eq.constants import F_OPP_LAST_TURN, F_OPP_RECENCY, F_OWN_RECENCY
        from hexfield_eq.engine_facts import facts_from_state
        from hexfield_eq.features import (
            PHASE_FIRST, PositionFacts, build_position, build_ray_lengths,
        )

        if actions is not None:
            facts = facts_from_state(cls._replay_state(actions))
            mode = "sequence"
        else:
            p0, p1 = stones or ([], [])
            ordered = sorted(
                [(q, r, 0) for q, r in p0] + [(q, r, 1) for q, r in p1]
            )
            facts = PositionFacts(
                records=tuple(
                    (int(q), int(r), int(owner), idx)
                    for idx, (q, r, owner) in enumerate(ordered)
                ),
                current_player=int(to_move if to_move is not None else 0),
                phase=PHASE_FIRST,
                first_stone=None,
            )
            mode = "free"
        support, feats = build_position(facts)
        if mode == "free":
            feats[:, [F_OWN_RECENCY, F_OPP_RECENCY, F_OPP_LAST_TURN]] = 0.0
        # Match the evaluator wire: fp16-round feature values before forward.
        feats = feats.astype(np.float16).astype(np.float32)
        return facts, support, feats, build_ray_lengths(facts, support), mode

    @staticmethod
    def _feature_names() -> list[str]:
        from hexfield_eq.constants import FEATURE_VERSION, NUM_FEATURES

        names = [
            "own_stone", "opp_stone", "empty", "legal", "phase_second",
            "first_stone", "player_colour", "own_recency", "opp_recency",
            "dist_to_stone", "opp_last_turn",
            "own_line_q", "own_line_r", "own_line_qr",
            "opp_line_q", "opp_line_r", "opp_line_qr",
            "own_live_q", "own_live_r", "own_live_qr",
            "opp_live_q", "opp_live_r", "opp_live_qr",
        ]
        if FEATURE_VERSION == 2:
            for threshold in (3, 4, 5):
                for side in ("own", "opp"):
                    names.extend(
                        f"{side}_live{threshold}_{axis}" for axis in ("q", "r", "qr")
                    )
        names.extend(["own_fork", "opp_fork"])
        if FEATURE_VERSION == 2:
            names.extend(["ply", "dist_centroid", "spread"])
        if len(names) != NUM_FEATURES:
            raise RuntimeError(
                f"hexfield feature-name map has {len(names)} entries, expected {NUM_FEATURES}"
            )
        return names

    @staticmethod
    def _sparse_policy(logits: Any, support: Any, floor: float) -> list[dict[str, Any]]:
        import torch

        if support.legal_count <= 0:
            return []
        priors = torch.softmax(logits[: support.legal_count].float(), dim=0)
        rows = [
            {"q": int(q), "r": int(r), "p": round(float(p), 6)}
            for (q, r), p in zip(
                support.legal_coords().tolist(), priors.cpu().tolist()
            )
        ]
        rows.sort(key=lambda item: (-item["p"], item["q"], item["r"]))
        return [item for item in rows if item["p"] >= floor]

    def lab_eval_payload(
        self, model: Any, *, actions: list[tuple[int, int]] | None,
        stones: tuple[list[tuple[int, int]], list[tuple[int, int]]] | None,
        to_move: int | None, policy_floor: float,
        attention_cell: tuple[int, int] | None, want_activations: bool,
        want_features: bool,
    ) -> dict:
        import torch
        from hexfield_eq.losses import decode_binned_value, decode_moves_left
        from ..jsonsafe import sanitize_json

        facts, support, feats, raylen, mode = self._lab_row(
            actions, stones, to_move
        )
        out = self._forward_rows(
            model, [(support, feats, raylen)], full_heads=True
        )
        value_logits = out["value"][0].reshape(1, -1).float()
        policy = self._sparse_policy(out["policy"][0], support, policy_floor)
        payload: dict[str, Any] = {
            "mode": mode,
            "to_move": int(facts.current_player),
            "phase": str(facts.phase),
            "ply": int(facts.placements_made),
            "legal_count": int(support.legal_count),
            "support": {
                "coords": support.coords.tolist(),
                "legal_count": int(support.legal_count),
                "stone_count": int(support.stone_count),
                "halo_count": int(support.halo_count),
            },
            "value": round(float(decode_binned_value(value_logits).item()), 6),
            "value_dist": [
                round(float(p), 5)
                for p in torch.softmax(value_logits[0], dim=0).cpu().tolist()
            ],
            "stv": self._stv_values(out),
            "moves_left": round(float(decode_moves_left(
                out["moves_left"][0].reshape(1, -1).float()
            ).item()), 3),
            "policy": policy,
            "opp_policy": self._sparse_policy(
                out["opp_policy"][0], support, policy_floor
            ),
            "soft_policy": self._sparse_policy(
                out["soft_policy"][0], support, policy_floor
            ),
            "top_k": policy[:5],
            "attention": {
                "available": False, "reason": self._INTROSPECTION_REASON,
            },
            "activations": {
                "available": False, "reason": self._INTROSPECTION_REASON,
            },
        }
        if want_features:
            payload["features"] = {
                "names": self._feature_names(),
                "planes": [
                    [round(float(v), 6) for v in feats[:, f].tolist()]
                    for f in range(feats.shape[1])
                ],
            }
        if mode == "free":
            payload["synthesized_history"] = True
            payload["zeroed_features"] = list(self._FREE_ZEROED)
        return sanitize_json(payload)

    def lab_search_payload(
        self, session: Any, evaluator: Any, profile: HexfieldEqSearchProfile, *,
        actions: list[tuple[int, int]], game_key: int, visits: int, seed: int,
    ) -> dict:
        return self._search_payload(
            session, evaluator, profile, self._replay_state(actions),
            game_key=game_key, visits=visits, seed=seed,
        )

    def _search_payload(
        self, session: Any, evaluator: Any, profile: HexfieldEqSearchProfile,
        state: Any, *, game_key: int, visits: int, seed: int,
    ) -> dict:
        import numpy as np
        from ..jsonsafe import sanitize_json

        try:
            result = profile.search_one(
                session, evaluator, state,
                game_key=game_key, visits=visits, seed=seed, temperature=0.0,
            )
        finally:
            session.discard(game_key)
        ids = np.frombuffer(result["visit_policy_action_ids_bytes"], dtype=np.uint32)
        weights = np.frombuffer(result["visit_policy_weights_bytes"], dtype=np.float32)
        total = float(weights.sum()) or 1.0
        rows = [
            {
                "q": q, "r": r, "p": round(float(w) / total, 6),
                "w": round(float(w), 4),
            }
            for (q, r), w in (
                (self.decode_action(int(action_id)), weight)
                for action_id, weight in zip(ids.tolist(), weights.tolist())
            )
        ]
        rows.sort(key=lambda item: (-item["p"], item["q"], item["r"]))
        best_q, best_r = self.decode_action(int(result["action_id"]))
        return sanitize_json({
            "visits": int(result["visits"]),
            "root_value": round(float(result["root_value"]), 6),
            "best": {"q": best_q, "r": best_r},
            "visit_policy": rows,
        })

    def selfcheck_forward(self, model: Any, state: Any, device: str) -> dict:
        out = self._forward_rows(
            model, [self._row(state)], device, serve_path=True
        )
        return {key: value.float().cpu() for key, value in out.items()}

    def selfcheck_autocast(self, device: str) -> bool:
        return device == "cuda"

    def warmup(self, model: Any, device: str) -> None:
        # The parity forward above initializes the family-specific serve path.
        return None
