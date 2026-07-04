"""Shared helpers for the hexfield test suite (not collected by pytest).

Adds packages/hexfield/python to sys.path so tests can import hexfield without
it being installed into a venv.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

_PACKAGES = Path(__file__).resolve().parent.parent / "packages"
sys.path.insert(0, str(_PACKAGES / "hexfield" / "python"))

from hexo_engine import api
from hexo_engine.types import AxialCoord, PlacementAction

from hexfield.geometry import unpack_action_id


def random_playout(seed: int, plies: int) -> "api.HexoState":
    """Play `plies` uniform-random legal placements (stops early on terminal)."""

    state = api.new_game()
    rng = random.Random(seed)
    for _ in range(plies):
        ids = api.legal_action_ids(state)
        if not ids:
            break
        q, r = unpack_action_id(rng.choice(ids))
        result = api.apply_action(state, PlacementAction(AxialCoord(q=q, r=r)))
        if result.terminal:
            break
    return state


def sample_decision_states(
    seeds: range, plies_choices: tuple[int, ...]
) -> list["api.HexoState"]:
    """Non-terminal states from seeded random playouts (decision rows only)."""

    states = []
    for seed in seeds:
        for plies in plies_choices:
            state = random_playout(seed * 1000 + plies, plies)
            if api.terminal(state) is None:
                states.append(state)
    return states
