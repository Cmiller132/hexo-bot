"""Registry of showcase model-family adapters."""

from __future__ import annotations

from .base import ModelFamily
from .hexfield_eq_family import HexfieldEqFamily
from .shrimp_family import ShrimpFamily

_FAMILIES: dict[str, ModelFamily] = {
    "shrimp": ShrimpFamily(),
    "hexfield_eq": HexfieldEqFamily(),
}


def get_family(name: str) -> ModelFamily:
    normalized = str(name).strip().lower()
    try:
        return _FAMILIES[normalized]
    except KeyError as exc:
        raise ValueError(
            f"unknown model family {name!r}; expected one of {sorted(_FAMILIES)}"
        ) from exc


__all__ = ["get_family", "ModelFamily", "ShrimpFamily", "HexfieldEqFamily"]
