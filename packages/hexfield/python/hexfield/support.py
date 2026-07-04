"""Support-set construction.

Definitions:

1. ``stones`` = occupied cells; ``legal`` = empty cells with hex-dist
   <= LEGAL_RADIUS of any stone (empty stone list => {(0, 0)}).
2. ``core = stones ∪ legal``; ``halo`` = cells hex-adjacent to core and not in
   core.
3. ``support = core ∪ halo``.

A single multi-source BFS of depth LEGAL_RADIUS+1 from the stones produces the
support, the halo, and the dist_to_stone values in one pass. core is the union
of radius-LEGAL_RADIUS disks; halo is the distance-(LEGAL_RADIUS+1) shell.

Node order: segments ``[ legal | stones | halo ]``, each ascending by packed
action id (== ascending signed (q, r)). The legal nodes of a row occupy slots
[0, legal_count).
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .constants import DIRECTIONS, HALO_DIST, LEGAL_RADIUS

# Model-side support radius. HEXFIELD_SUPPORT_RADIUS restricts the support to
# legal cells within hex-dist <= R of a stone (default LEGAL_RADIUS). Smaller R
# yields a smaller support. The bias table (BIAS_DISK_RADIUS) and DIST_SCALE
# remain at LEGAL_RADIUS. The Rust featurizer reads the same env var.
_SUPPORT_RADIUS = int(os.environ.get("HEXFIELD_SUPPORT_RADIUS", "").strip() or LEGAL_RADIUS)
_SUPPORT_HALO = _SUPPORT_RADIUS + 1


class SupportContractError(ValueError):
    """Raised when the closed-form legal set disagrees with the engine's
    ``expected_legal`` set passed to :func:`build_support`.

    The closed-form legality ``empty ∧ dist <= LEGAL_RADIUS`` equals the
    engine's legal set on decision states. On a terminal state the engine
    returns an empty legal set while the closed form yields a non-empty legal
    prefix.
    """


@dataclass(frozen=True)
class Support:
    """One position's support set in node order.

    coords: (N, 2) int32 axial (q, r) per node, [legal | stones | halo].
    dist:   (N,)  int32 hex distance to the nearest stone (0 when stone list
            is empty).
    nbr:    (N, 6) int32 row-local neighbour index per DIRECTIONS, -1 if absent.
    index:  coord -> row lookup for the support.
    """

    coords: np.ndarray
    legal_count: int
    stone_count: int
    halo_count: int
    dist: np.ndarray
    nbr: np.ndarray
    index: dict[tuple[int, int], int]

    @property
    def num_nodes(self) -> int:
        return int(self.coords.shape[0])

    def legal_coords(self) -> np.ndarray:
        return self.coords[: self.legal_count]

    def segments(self) -> tuple[range, range, range]:
        """(legal, stones, halo) row ranges."""

        a = self.legal_count
        b = a + self.stone_count
        return range(0, a), range(a, b), range(b, self.num_nodes)


def build_support(
    stones: list[tuple[int, int]],
    *,
    expected_legal: Iterable[tuple[int, int]] | None = None,
) -> Support:
    """Build the support set from the stone list (empty list => origin state).

    Legality is derived in closed form (``empty ∧ dist <= LEGAL_RADIUS``),
    which equals the engine's legal set on decision states. On a terminal state
    the engine's legal set is empty while the closed form produces a non-empty
    legal prefix.

    When ``expected_legal`` is ``None`` no validation runs. Passing an iterable
    of ``(q, r)`` coords requires the closed-form legal set to equal that set
    exactly, raising :class:`SupportContractError` otherwise. See
    :func:`assert_decision_support` for a standalone check.
    """

    support = _build_support(stones)
    if expected_legal is not None:
        _validate_legal(support, expected_legal)
    return support


def assert_decision_support(
    stones: list[tuple[int, int]],
    expected_legal: Iterable[tuple[int, int]],
) -> Support:
    """Build the support and validate the closed-form legal set.

    Wrapper over ``build_support(stones, expected_legal=...)`` that checks the
    closed-form legal set against ``expected_legal`` (e.g. engine legal action
    ids unpacked to coords). Raises :class:`SupportContractError` on any
    divergence. Returns the validated support.
    """

    return build_support(stones, expected_legal=expected_legal)


def _validate_legal(
    support: Support, expected_legal: Iterable[tuple[int, int]]
) -> None:
    expected = {(int(q), int(r)) for q, r in expected_legal}
    derived = {
        (int(q), int(r)) for q, r in support.coords[: support.legal_count].tolist()
    }
    if derived != expected:
        missing = sorted(expected - derived)
        extra = sorted(derived - expected)
        raise SupportContractError(
            "build_support legal set diverges from the engine "
            f"(closed-form {len(derived)} vs engine {len(expected)}); "
            f"in_engine_not_closed_form={missing[:8]} "
            f"in_closed_form_not_engine={extra[:8]} — "
            "build_support is decision-states-only; a non-empty closed-form "
            "legal set against an empty engine set indicates a TERMINAL state, "
            "which is never evaluated."
        )


def _build_support(stones: list[tuple[int, int]]) -> Support:
    if not stones:
        # Empty stone list: support = origin + its 6 halo neighbours (7 nodes,
        # 1 legal); dist is 0 for all nodes.
        ordered = [(0, 0)] + sorted(
            (dq, dr) for dq, dr in DIRECTIONS
        )
        coords = np.asarray(ordered, dtype=np.int32)
        dist = np.zeros(len(ordered), dtype=np.int32)
        index = {tuple(c): i for i, c in enumerate(ordered)}
        return Support(
            coords=coords,
            legal_count=1,
            stone_count=0,
            halo_count=6,
            dist=dist,
            nbr=_neighbor_table(ordered, index),
            index=index,
        )

    stone_set = set(stones)
    dist: dict[tuple[int, int], int] = {coord: 0 for coord in stone_set}
    frontier: deque[tuple[int, int]] = deque(stone_set)
    while frontier:
        cell = frontier.popleft()
        d = dist[cell]
        if d == _SUPPORT_HALO:
            continue
        q, r = cell
        for dq, dr in DIRECTIONS:
            nxt = (q + dq, r + dr)
            if nxt not in dist:
                dist[nxt] = d + 1
                frontier.append(nxt)

    legal = sorted(c for c, d in dist.items() if d <= _SUPPORT_RADIUS and c not in stone_set)
    stones_sorted = sorted(stone_set)
    halo = sorted(c for c, d in dist.items() if d == _SUPPORT_HALO)

    ordered = legal + stones_sorted + halo
    index = {coord: i for i, coord in enumerate(ordered)}
    return Support(
        coords=np.asarray(ordered, dtype=np.int32),
        legal_count=len(legal),
        stone_count=len(stones_sorted),
        halo_count=len(halo),
        dist=np.asarray([dist[c] for c in ordered], dtype=np.int32),
        nbr=_neighbor_table(ordered, index),
        index=index,
    )


def _neighbor_table(
    ordered: list[tuple[int, int]], index: dict[tuple[int, int], int]
) -> np.ndarray:
    nbr = np.full((len(ordered), 6), -1, dtype=np.int32)
    for row, (q, r) in enumerate(ordered):
        for k, (dq, dr) in enumerate(DIRECTIONS):
            j = index.get((q + dq, r + dr))
            if j is not None:
                nbr[row, k] = j
    return nbr
