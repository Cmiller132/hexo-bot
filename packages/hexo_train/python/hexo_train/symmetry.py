"""Training-owned D6 symmetry selection.

`hexo_utils.encoding` defines what a D6 symmetry is. `hexo_train` owns when a
sample receives one during the training lifecycle, so models see a consistent
selection but remain free to decide how to apply it to their tensors.

The default selector is deterministic, not random-state based. Given the same
seed, epoch, sample index, game id, and turn index, it always chooses the same
symmetry. This makes training windows reproducible while still varying choices
across epochs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any, Mapping, Sequence

from hexo_utils.encoding import D6_SIZE, D6Symmetry


@dataclass(frozen=True, slots=True)
class SampleSymmetrySelection:
    """D6 choices attached to the current sample window.

    `symmetries[i]` corresponds to sample index `i` in the selected window.
    Models receive this object and decide how to apply each transform.
    """

    symmetries: Sequence[D6Symmetry] = field(default_factory=tuple)
    seed: int = 0
    epoch: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class D6SymmetrySelector:
    """Default deterministic selector for training-time D6 augmentation."""

    def choose(
        self,
        *,
        seed: int,
        epoch: int,
        sample_index: int,
        game_id: str = "",
        turn_index: int = 0,
    ) -> D6Symmetry:
        """Choose one deterministic pseudo-random D6 symmetry.

        The hash input includes epoch and sample identity so repeated epochs can
        see different augmentations without relying on mutable RNG state.
        """

        material = f"{seed}:{epoch}:{sample_index}:{game_id}:{turn_index}".encode("utf-8")
        digest = blake2b(material, digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="little", signed=False)
        return D6Symmetry(value % D6_SIZE)

    def select_for_window(
        self,
        sample_window: object,
        *,
        seed: int | None,
        epoch: int = 0,
    ) -> SampleSymmetrySelection:
        """Select symmetries for every visible sample in a window description."""

        count = _sample_count(sample_window)
        resolved_seed = int(seed or 0)
        symmetries = tuple(
            self.choose(seed=resolved_seed, epoch=epoch, sample_index=index)
            for index in range(count)
        )
        return SampleSymmetrySelection(
            symmetries=symmetries,
            seed=resolved_seed,
            epoch=epoch,
            metadata={
                "sample_count": count,
                "note": "Default deterministic D6 selection; models own tensor transforms.",
            },
        )


def _sample_count(sample_window: object) -> int:
    """Infer how many samples are visible in a loosely typed sample window."""

    if sample_window is None:
        return 0
    window_size = getattr(sample_window, "window_size", None)
    index = getattr(sample_window, "index", None)
    sample_count = getattr(index, "sample_count", 0)
    if window_size is None:
        return int(sample_count or 0)
    return min(int(window_size), int(sample_count or 0))
