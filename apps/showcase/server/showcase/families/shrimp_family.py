"""The existing shrimp serve path expressed as a model-family adapter."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Sequence

_SEED_MASK = (1 << 63) - 1


class SearchProfile:
    """As-trained shrimp eval-arena search invocation."""

    def __init__(self, config_path: Path | str) -> None:
        from shrimp.config import build_divergence_overrides, parse_shrimp_config

        with open(config_path, "rb") as fh:
            raw = tomllib.load(fh)
        model_cfg = raw.get("model", {}).get("config", {})
        cfg = parse_shrimp_config(
            {
                "device": "cpu",
                "selfplay": model_cfg.get("selfplay", {}),
                "multi_stage_eval": model_cfg.get("multi_stage_eval", {}),
            }
        )
        self.selfplay = cfg.selfplay
        self.overrides = build_divergence_overrides(cfg.selfplay)
        self.virtual_batch_size = int(cfg.multi_stage_eval.eval_virtual_batch_size or 32)
        self.opening_plies = int(cfg.multi_stage_eval.opening_plies)
        self.opening_temperature = float(cfg.multi_stage_eval.opening_temperature)

    def move_temperature(self, ply: int) -> float:
        if ply < self.opening_plies and self.opening_temperature > 0.0:
            return self.opening_temperature
        return 0.0

    def search_one(
        self, session: Any, evaluator: Any, state: Any, *,
        game_key: int, visits: int, seed: int, temperature: float,
    ) -> dict:
        sp = self.selfplay
        return session.search(
            [int(game_key)],
            (state,),
            visits=int(visits),
            c_puct=sp.c_puct,
            temperature=0.0,
            seed=int(seed) & _SEED_MASK,
            evaluator=evaluator,
            move_temperatures=[float(temperature)],
            divergence_overrides=self.overrides,
            virtual_batch_size=self.virtual_batch_size,
            active_root_limit=sp.active_root_limit,
            widening_policy_mass=sp.widening_policy_mass,
            widening_max_children=sp.widening_max_children,
            widening_min_children=sp.widening_min_children,
            fpu_reduction=sp.fpu_reduction,
            tss_enabled=sp.tss_enabled,
            search_parity_mode=sp.search_parity_mode,
        )[0]


def _load_checkpoint(path: Path) -> Any:
    """Strict-load a shrimp checkpoint exactly as the pre-adapter worker did."""
    import torch

    from shrimp.model import ShrimpNet, infer_net_kwargs_from_state_dict

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise RuntimeError(f"checkpoint payload has no 'model' state dict: {path}")
    state_dict = payload["model"]
    model = ShrimpNet(**infer_net_kwargs_from_state_dict(state_dict))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


class ShrimpFamily:
    name = "shrimp"

    def prepare_process(self, specs: Sequence[Any]) -> None:
        return None

    def load_net(self, spec: Any) -> Any:
        return _load_checkpoint(spec.checkpoint)

    def build_evaluator(self, model: Any, device: str) -> Any:
        from shrimp.inference import ShrimpEvaluator

        return ShrimpEvaluator(model, device=device)

    def build_session(self) -> Any:
        from shrimp import _rust

        return _rust.ShrimpMctsSession(max_states=65_536)

    def build_profile(self, profile_path: Path | None, settings: Any) -> SearchProfile:
        return SearchProfile(profile_path or Path(settings.search_config))

    def decode_action(self, action_id: int) -> tuple[int, int]:
        from shrimp.geometry import unpack_action_id

        return unpack_action_id(action_id)

    def net_eval(self, model: Any, state: Any, *, policy_floor: float) -> dict:
        from .. import analysis

        return analysis.net_eval(model, state, policy_floor=policy_floor)

    def searched_eval(
        self, session: Any, evaluator: Any, profile: SearchProfile, state: Any,
        *, game_key: int, visits: int, seed: int,
    ) -> dict:
        from .. import analysis

        return analysis.searched_eval(
            session, evaluator, profile, state,
            game_key=game_key, visits=visits, seed=seed,
            decode_action=self.decode_action,
        )

    def summary_row(self, state: Any) -> Any:
        from .. import analysis

        return analysis.featurize(state)

    def summary_eval(self, model: Any, rows: list[Any]) -> dict:
        from .. import analysis

        return analysis.summary_eval(model, rows)

    def lab_eval_payload(
        self, model: Any, *, actions: list[tuple[int, int]] | None,
        stones: tuple[list[tuple[int, int]], list[tuple[int, int]]] | None,
        to_move: int | None, policy_floor: float,
        attention_cell: tuple[int, int] | None, want_activations: bool,
        want_features: bool,
    ) -> dict:
        from .. import lab

        if actions is not None:
            facts, support, feats = lab.build_sequence_position(actions)
            mode = "sequence"
        else:
            p0, p1 = stones or ([], [])
            facts, support, feats = lab.build_free_position(
                p0, p1, to_move if to_move is not None else 0
            )
            mode = "free"
        payload = lab.eval_payload(
            model, facts, support, feats,
            policy_floor=policy_floor,
            attention_cell=attention_cell,
            want_activations=want_activations,
            want_features=want_features,
        )
        payload["mode"] = mode
        if mode == "free":
            payload["synthesized_history"] = True
            payload["zeroed_features"] = list(lab.FREE_ZEROED)
        return payload

    def lab_search_payload(
        self, session: Any, evaluator: Any, profile: SearchProfile, *,
        actions: list[tuple[int, int]], game_key: int, visits: int, seed: int,
    ) -> dict:
        from .. import lab

        return lab.search_payload(
            session, evaluator, profile, lab.replay_state(actions),
            game_key=game_key, visits=visits, seed=seed,
            decode_action=self.decode_action,
        )

    def selfcheck_forward(self, model: Any, state: Any, device: str) -> dict:
        from ..device import _forward
        from shrimp.batching import collate_rows
        from ..analysis import featurize

        import torch
        from shrimp.inference import serve_autocast

        batch = collate_rows([featurize(state)])
        return _forward(
            model, batch, torch.device(device), autocast_on=serve_autocast(device)
        )

    def selfcheck_autocast(self, device: str) -> bool:
        from shrimp.inference import serve_autocast

        return bool(serve_autocast(device))

    def warmup(self, model: Any, device: str) -> None:
        from ..device import warmup

        warmup(model, device)
