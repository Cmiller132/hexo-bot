"""Model-family boundary for showcase worker inference.

Implementations keep heavyweight model imports inside methods: importing the
catalogue in the web process must remain torch-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence


class SearchProfile(Protocol):
    def move_temperature(self, ply: int) -> float: ...

    def search_one(
        self,
        session: Any,
        evaluator: Any,
        state: Any,
        *,
        game_key: int,
        visits: int,
        seed: int,
        temperature: float,
    ) -> dict: ...


class ModelFamily(Protocol):
    """Extension point for one checkpoint/model/search implementation."""

    name: str

    def prepare_process(self, specs: Sequence[Any]) -> None: ...
    def load_net(self, spec: Any) -> Any: ...
    def build_evaluator(self, model: Any, device: str) -> Any: ...
    def build_session(self) -> Any: ...
    def build_profile(
        self, profile_path: Path | None, settings: Any
    ) -> SearchProfile: ...
    def decode_action(self, action_id: int) -> tuple[int, int]: ...
    def net_eval(self, model: Any, state: Any, *, policy_floor: float) -> dict: ...
    def searched_eval(
        self, session: Any, evaluator: Any, profile: SearchProfile, state: Any,
        *, game_key: int, visits: int, seed: int,
    ) -> dict: ...
    def summary_row(self, state: Any) -> Any: ...
    def summary_eval(self, model: Any, rows: list[Any]) -> dict: ...
    def lab_eval_payload(
        self,
        model: Any,
        *,
        actions: list[tuple[int, int]] | None,
        stones: tuple[list[tuple[int, int]], list[tuple[int, int]]] | None,
        to_move: int | None,
        policy_floor: float,
        attention_cell: tuple[int, int] | None,
        want_activations: bool,
        want_features: bool,
    ) -> dict: ...
    def lab_search_payload(
        self,
        session: Any,
        evaluator: Any,
        profile: SearchProfile,
        *,
        actions: list[tuple[int, int]],
        game_key: int,
        visits: int,
        seed: int,
    ) -> dict: ...
    def selfcheck_forward(self, model: Any, state: Any, device: str) -> dict: ...
    def selfcheck_autocast(self, device: str) -> bool: ...
    def warmup(self, model: Any, device: str) -> None: ...
