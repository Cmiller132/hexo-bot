"""The MantisNet input builder: positions to graphs, graphs to batches.

This module implements the representation of ``docs/MODEL_SPEC.md`` §3–§4
and the batching contract of §9 under ``MODEL_REPR_VERSION``. It derives live
windows from the stone list without calling the engine's window walk, which
remains the independent oracle for §12.1.

Index conventions this module fixes (each is part of the representation):

- Window feature: the reversal-canonical rank of the mover-relative ternary
  slot pattern (empty, own, opponent), one of ``TERN_PATTERNS``.
- Decoder and incidence classes: reversal-orbit ranks of the joint ternary
  ``(pattern, slot)`` pairs, one of ``TERN_DEC_CLASSES`` or
  ``TERN_OCC_CLASSES`` respectively.
- Attention distance bucket: hex distance ``d >= 1`` maps to ``d - 1`` clamped
  to ``D_MAX - 1``; ``SELF`` is ``D_MAX``; ``TOKEN`` is ``D_MAX + 1`` and wins
  over ``SELF`` on the token–token pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ._rust import MODEL_REPR_VERSION

from .relay import relay_tables

WINDOW_LEN = 6
# Unit steps of the engine's axes, in canonical order Q, R, QR.
AXES = np.array([[1, 0], [0, 1], [1, -1]], dtype=np.int64)
ORBIT48_CLASSES = 48
CELL_NEAREST_UNREACHED = 9


def _rotate(qr: tuple[int, int]) -> tuple[int, int]:
    q, r = qr
    return -r, q + r


def _reflect(qr: tuple[int, int]) -> tuple[int, int]:
    q, r = qr
    return r, q


def _canonical_displacement(qr: tuple[int, int]) -> tuple[int, int]:
    images = []
    for seed in (qr, _reflect(qr)):
        image = seed
        for _ in range(6):
            images.append(image)
            image = _rotate(image)
    return min(images)


def _hex_distance(qr: np.ndarray) -> np.ndarray:
    return np.maximum.reduce(
        [np.abs(qr[..., 0]), np.abs(qr[..., 1]), np.abs(qr[..., 0] + qr[..., 1])]
    )


_ORBIT_REPRESENTATIVES = sorted(
    {
        (max(abs(q), abs(r), abs(q + r)), *_canonical_displacement((q, r)))
        for q in range(-12, 13)
        for r in range(-12, 13)
        if 1 <= max(abs(q), abs(r), abs(q + r)) <= 12
    }
)
assert len(_ORBIT_REPRESENTATIVES) == ORBIT48_CLASSES
_ORBIT_RANK = {
    (q, r): index for index, (_distance, q, r) in enumerate(_ORBIT_REPRESENTATIVES)
}


def orbit48_id(dq: int, dr: int, radius: int = 12) -> int:
    """Frozen D6 displacement class; radii outside ``1..12`` are refused."""
    if not 1 <= radius <= 12:
        raise ValueError(f"orbit radius must lie in 1..12, got {radius}")
    distance = max(abs(dq), abs(dr), abs(dq + dr))
    if not 1 <= distance <= radius:
        raise ValueError(
            f"displacement ({dq}, {dr}) has distance {distance}, outside 1..={radius}"
        )
    return _ORBIT_RANK[_canonical_displacement((dq, dr))]


# --- Ternary tables for the all-nonempty window scope.
#
# Every nonempty candidate window is a node, so a slot
# is empty, own, or opponent: a window is a base-3 pattern over its six slots
# (digit at 3^k is slot k, own = 1, opp = 2, mover-relative). A reflection
# reverses the digit string; canonical form is the numeric minimum of the
# pair. Own-only / opponent-only / mixed status is a pure function of the
# canonical pattern (reversal permutes slots, never digits), so no separate
# status feature exists. The all-own and all-opp patterns are terminal-only
# and unreachable from a live position; they keep their vocabulary rows so
# the class counts stay the asserted laws.

_POW3 = 3 ** np.arange(WINDOW_LEN, dtype=np.int64)
_TERN_DIGITS = (np.arange(729)[:, None] // _POW3[None, :]) % 3  # (729, 6)
_TERN_REV = (_TERN_DIGITS[:, ::-1] * _POW3[None, :]).sum(axis=1)
_TERN_CANON = np.minimum(np.arange(729, dtype=np.int64), _TERN_REV)

# 378 orbits of 729 patterns under reversal (27 palindromes); 377 nonempty.
_TERN_RANK = np.full(729, -1, dtype=np.int64)
_TERN_RANK[np.unique(_TERN_CANON[1:])] = np.arange(377)
_TERN_RANK = _TERN_RANK[_TERN_CANON]
TERN_PATTERNS = 377
assert len(np.unique(_TERN_CANON)) == 378 and int(_TERN_RANK.max()) + 1 == TERN_PATTERNS


def _tern_joint_classes() -> tuple[np.ndarray, np.ndarray]:
    """The ternary joint ``(pattern, slot)`` orbit tables, decoder and incidence.

    One enumeration of the involution ``(p, s) -> (reverse3(p), 5 - s)`` over
    all 729 x 6 pairs in ascending ``(p, s)`` order — 2187 orbits, asserted —
    then re-ranked restrictions: empty slots of nonempty patterns give the
    decoder table, occupied slots the incidence table. Their 726 + 1458
    orbits are the asserted 2184 nonempty-pattern classes.
    """
    joint = np.full((729, WINDOW_LEN), -1, dtype=np.int64)
    nxt = 0
    for p in range(729):
        rev = int(_TERN_REV[p])
        for s in range(WINDOW_LEN):
            if (p, s) <= (rev, WINDOW_LEN - 1 - s):
                joint[p, s] = nxt
                nxt += 1
            else:
                joint[p, s] = joint[rev, WINDOW_LEN - 1 - s]
    assert nxt == 2187

    def rerank(mask: np.ndarray) -> np.ndarray:
        ids = np.unique(joint[mask])
        table = np.full((729, WINDOW_LEN), -1, dtype=np.int64)
        table[mask] = np.searchsorted(ids, joint[mask])
        return table

    empty_slot = _TERN_DIGITS == 0
    dec = rerank(empty_slot & (np.arange(729) != 0)[:, None])
    occ = rerank(~empty_slot)
    return dec, occ


_TERN_DEC_CLASS, _TERN_OCC_CLASS = _tern_joint_classes()
TERN_DEC_CLASSES = int(_TERN_DEC_CLASS.max()) + 1
TERN_OCC_CLASSES = int(_TERN_OCC_CLASS.max()) + 1
assert TERN_DEC_CLASSES == 726 and TERN_OCC_CLASSES == 1458
assert TERN_DEC_CLASSES + TERN_OCC_CLASSES == 2184


# --- Step 4 action-row tables (MANTIS_GRAFT_SPEC §4, Step 4).
#
# Every legal action has 18 hypothetical post-placement windows (3 axes x 6
# candidate slots with an own stone inserted at the action cell). The
# post-placement class is joint in the post pattern and the inserted slot.
#
# Orbits of ``(post, slot)`` pairs whose slot
# digit is own, under the joint reversal — 1458 pairs, 729 orbits.

def _tern_post1_classes() -> np.ndarray:
    """(729, 6) orbit table of ``(post, slot)`` own-digit pairs; -1 elsewhere."""
    table = np.full((729, WINDOW_LEN), -1, dtype=np.int64)
    nxt = 0
    for post in range(729):
        rev = int(_TERN_REV[post])
        for s in range(WINDOW_LEN):
            if (post // 3**s) % 3 != 1:
                continue
            if (post, s) <= (rev, WINDOW_LEN - 1 - s):
                table[post, s] = nxt
                nxt += 1
            else:
                table[post, s] = table[rev, WINDOW_LEN - 1 - s]
    assert nxt == 729
    return table


_TERN_POST1_CLASS = _tern_post1_classes()
TERN_POST1_CLASSES = 729

# The classes of an own stone inserted into an empty candidate window: post
# pattern ``3**k`` at slot ``k``. Reversal folds slot ``k`` onto ``5 - k``, so
# the six inserts occupy three orbits, indexed by ``min(k, 5 - k)``. A cell's
# EMPTY rows are collated as counts over these orbits — the shared empty base
# makes every EMPTY row of one orbit the same hidden row.
ACTION_EMPTY_ORBITS = 3
ACTION_EMPTY_CLASSES = tuple(int(_TERN_POST1_CLASS[3**k, k]) for k in range(3))
assert len(set(ACTION_EMPTY_CLASSES)) == ACTION_EMPTY_ORBITS
assert all(
    _TERN_POST1_CLASS[3**k, k] == _TERN_POST1_CLASS[3 ** (5 - k), 5 - k]
    for k in range(WINDOW_LEN)
)

# Pre-window statuses of an action row.
ACTION_OWN, ACTION_OPP, ACTION_EMPTY, ACTION_MIXED = 0, 1, 2, 3

# Coordinate packing: q, r fit i16, so 21 bits of headroom per component is
# collision-free. Window identity packs the axis into the low two bits.
_QSHIFT = 1 << 21


def _pack(qr: np.ndarray) -> np.ndarray:
    """(n, 2) coordinates to collision-free int64 keys."""
    return qr[:, 0] * _QSHIFT + qr[:, 1]


@dataclass(frozen=True)
class PositionGraph:
    """One position's entities and index tables, in numpy (§9)."""

    # Stones, in the order given (engine canonical when built from a position).
    stone_own: np.ndarray  # (n_s,) int64: 0 = side to move, 1 = opponent
    stone_qr: np.ndarray  # (n_s, 2) int64, for the distance buckets only
    # All nonempty windows.
    window_feat: np.ndarray  # (n_w,) int64: ternary pattern rank
    window_id: np.ndarray  # (n_w, 3) int64: (axis, start_q, start_r), consumed
    # only through reversal-invariant pair classes (§5.1c) and by tests.
    # Stone <-> window incidence with joint occupied-slot classes.
    inc_stone: np.ndarray  # (e,) int64
    inc_window: np.ndarray  # (e,) int64
    inc_class: np.ndarray  # (e,) int64, < TERN_OCC_CLASSES
    # Policy decoder table over legal cells, in engine legal order.
    n_legal: int
    cell_qr: np.ndarray  # (n_legal, 2), builder/debug only
    cell_occupancy: np.ndarray  # EMPTY/OWN/OPP relative to mover
    cell_is_legal: np.ndarray  # (n_legal,) bool-as-int
    cell_nearest: np.ndarray  # 1..8, or 9 at the stone-free opening
    radius_src: np.ndarray  # stone source index
    radius_dst: np.ndarray  # legal-cell destination index
    radius_orbit: np.ndarray  # exact D6 orbit in 0..47
    radius_own: np.ndarray  # source occupancy: own=0, opp=1
    radius_on_axis: np.ndarray  # invariant boolean
    adjacency_src: np.ndarray  # directed legal-cell source
    adjacency_dst: np.ndarray  # directed legal-cell destination
    adjacency_axis: np.ndarray  # structural axis route in 0..2
    dec_cell: np.ndarray  # (e_d,) int64: legal-cell index
    dec_window: np.ndarray  # (e_d,) int64: represented window through it
    dec_class: np.ndarray  # (e_d,) int64: the (pattern, slot) class, < TERN_DEC_CLASSES
    moves_remaining: int  # 1 or 2
    # Dense (n_legal, 3, 6) action-row tables: kept-window index or -1;
    # post-placement class in the ternary vocabulary; pre-window status.
    action_window_index: np.ndarray
    action_post1_class: np.ndarray
    action_pre_status: np.ndarray

    @property
    def n_stones(self) -> int:
        return len(self.stone_own)

    @property
    def n_windows(self) -> int:
        return len(self.window_feat)


def _action_tables(
    legal_qr: np.ndarray,
    sorted_key: np.ndarray,
    order: np.ndarray,
    stone_own: np.ndarray,
    live_key: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Step 4 row tables: 18 hypothetical post-placement windows per action.

    Each legal cell's 11-cell line per axis is read once; candidate slot ``k``
    is the window starting ``k`` steps before the cell. The emitted window
    index refers to the all-nonempty kept-window list. Agreement between status
    and index is asserted, mirroring the donor's walk-consistency check.
    """
    n_legal = len(legal_qr)
    offs = np.arange(-(WINDOW_LEN - 1), WINDOW_LEN, dtype=np.int64)  # (11,)
    cells = (
        legal_qr[:, None, None, :]
        + AXES[None, :, None, :] * offs[None, None, :, None]
    )  # (n_legal, 3, 11, 2)
    key = _pack(cells.reshape(-1, 2))
    pos = np.searchsorted(sorted_key, key)
    pos_clip = np.minimum(pos, max(len(sorted_key) - 1, 0))
    hit = (
        (sorted_key[pos_clip] == key)
        if len(sorted_key)
        else np.zeros(len(key), dtype=bool)
    )
    occupant = np.where(hit, order[pos_clip] if len(order) else 0, -1)
    digit = np.zeros(len(key), dtype=np.int64)
    filled = occupant >= 0
    digit[filled] = np.where(stone_own[occupant[filled]] == 0, 1, 2)
    line = digit.reshape(n_legal, 3, 2 * WINDOW_LEN - 1)
    if line[:, :, WINDOW_LEN - 1].any():
        raise ValueError("a legal action cell is occupied")

    # windows[a, x, k, j] = line[a, x, j + 5 - k]: slot j of the candidate
    # window that starts k steps before the action cell.
    j_idx = (
        np.arange(WINDOW_LEN)[None, :]
        + (WINDOW_LEN - 1)
        - np.arange(WINDOW_LEN)[:, None]
    )  # (k, j)
    win_digits = line[:, :, j_idx]  # (n_legal, 3, 6, 6)
    pre = (win_digits * _POW3[None, None, None, :]).sum(axis=-1)  # (n_legal, 3, 6)
    own_mask = ((win_digits == 1) << np.arange(WINDOW_LEN)[None, None, None, :]).sum(
        axis=-1
    )
    opp_mask = ((win_digits == 2) << np.arange(WINDOW_LEN)[None, None, None, :]).sum(
        axis=-1
    )
    has_own, has_opp = own_mask > 0, opp_mask > 0
    status = np.where(
        has_own & ~has_opp,
        ACTION_OWN,
        np.where(
            has_opp & ~has_own,
            ACTION_OPP,
            np.where(has_own & has_opp, ACTION_MIXED, ACTION_EMPTY),
        ),
    )

    k_arr = np.arange(WINDOW_LEN)[None, None, :]
    post = pre + _POW3[None, None, :]
    post1 = _TERN_POST1_CLASS[post, k_arr]
    if post1.min() < 0:
        raise ValueError("a post-placement row lost its own stone")

    starts = legal_qr[:, None, None, :] - AXES[None, :, None, :] * np.arange(
        WINDOW_LEN, dtype=np.int64
    )[None, None, :, None]
    axis_idx = np.broadcast_to(
        np.arange(3, dtype=np.int64)[None, :, None], starts.shape[:3]
    )
    wkey = _pack(starts.reshape(-1, 2)) * 4 + axis_idx.reshape(-1)
    wpos = np.searchsorted(live_key, wkey)
    wpos_clip = np.minimum(wpos, max(len(live_key) - 1, 0))
    whit = (
        (live_key[wpos_clip] == wkey)
        if len(live_key)
        else np.zeros(len(wkey), dtype=bool)
    )
    window_index = np.where(whit, wpos_clip, -1).reshape(n_legal, 3, WINDOW_LEN)

    kept = status != ACTION_EMPTY
    if ((window_index >= 0) != kept).any():
        raise ValueError("the kept-window set disagrees with the action-row walk")
    return window_index, post1, status


def _cell_node_fields(
    stone_qr: np.ndarray, stone_own: np.ndarray, legal_qr: np.ndarray
) -> dict[str, np.ndarray]:
    n_legal = len(legal_qr)
    occupancy = np.zeros(n_legal, dtype=np.int64)
    is_legal = np.ones(n_legal, dtype=np.int64)
    if len(stone_qr):
        displacement = legal_qr[:, None, :] - stone_qr[None, :, :]
        distance = _hex_distance(displacement)
        nearest = distance.min(axis=1).astype(np.int64)
        if np.any(nearest > 8) or np.any(nearest == 0):
            raise ValueError("legal cells must be empty and within radius 8 of a stone")
        dst, src = np.nonzero((distance >= 1) & (distance <= 8))
        delta = displacement[dst, src]
        orbit = np.fromiter(
            (orbit48_id(int(q), int(r)) for q, r in delta),
            dtype=np.int64,
            count=len(delta),
        )
        on_axis = (
            (delta[:, 0] == 0)
            | (delta[:, 1] == 0)
            | (delta[:, 0] + delta[:, 1] == 0)
        ).astype(np.int64)
        radius_src = src.astype(np.int64)
        radius_dst = dst.astype(np.int64)
        radius_own = stone_own[src]
    else:
        nearest = np.full(n_legal, CELL_NEAREST_UNREACHED, dtype=np.int64)
        radius_src = radius_dst = orbit = radius_own = on_axis = np.empty(
            0, dtype=np.int64
        )

    index = {tuple(map(int, qr)): row for row, qr in enumerate(legal_qr)}
    adjacency = []
    for source, qr in enumerate(legal_qr):
        for axis, step in enumerate(AXES):
            for sign in (-1, 1):
                destination = index.get(tuple(map(int, qr + sign * step)))
                if destination is not None:
                    adjacency.append((destination, source, axis))
    adjacency.sort()
    if adjacency:
        adj = np.asarray(adjacency, dtype=np.int64)
        adjacency_dst, adjacency_src, adjacency_axis = adj.T
    else:
        adjacency_src = adjacency_dst = adjacency_axis = np.empty(0, dtype=np.int64)
    return {
        "cell_qr": legal_qr.copy(),
        "cell_occupancy": occupancy,
        "cell_is_legal": is_legal,
        "cell_nearest": nearest,
        "radius_src": radius_src,
        "radius_dst": radius_dst,
        "radius_orbit": orbit,
        "radius_own": radius_own,
        "radius_on_axis": on_axis,
        "adjacency_src": adjacency_src,
        "adjacency_dst": adjacency_dst,
        "adjacency_axis": adjacency_axis,
    }


def build(
    stone_qr: np.ndarray,
    stone_owner: np.ndarray,
    mover: int,
    legal_qr: np.ndarray,
    moves_remaining: int,
) -> PositionGraph:
    """Build one position's graph from the §11 input list.

    ``stone_qr`` is (n_s, 2) int, ``stone_owner`` (n_s,) int in {0, 1},
    ``mover`` the side to move, ``legal_qr`` (n_legal, 2) int in engine legal
    order. Raises ``ValueError`` for a terminal position (no legal moves):
    terminal positions are a builder error, not a silent default.

    Every nonempty candidate is kept under the ternary tables, and every legal
    action carries its 18 post-placement rows.
    """
    stone_qr = np.asarray(stone_qr, dtype=np.int64).reshape(-1, 2)
    stone_owner = np.asarray(stone_owner, dtype=np.int64).reshape(-1)
    legal_qr = np.asarray(legal_qr, dtype=np.int64).reshape(-1, 2)
    if len(legal_qr) == 0:
        raise ValueError("terminal position: the builder refuses it")
    if moves_remaining not in (1, 2):
        raise ValueError(f"moves_remaining must be 1 or 2, got {moves_remaining}")

    n_s = len(stone_qr)
    stone_own = (stone_owner != mover).astype(np.int64)
    cell_nodes = _cell_node_fields(stone_qr, stone_own, legal_qr)

    if n_s == 0:
        # Ply 0: no stones or windows. Every action row is an EMPTY insert.
        empty_key = np.empty(0, dtype=np.int64)
        actions = _action_tables(legal_qr, empty_key, empty_key, stone_own, empty_key)
        return PositionGraph(
            stone_own=stone_own,
            stone_qr=stone_qr,
            window_feat=np.empty(0, dtype=np.int64),
            window_id=np.empty((0, 3), dtype=np.int64),
            inc_stone=np.empty(0, dtype=np.int64),
            inc_window=np.empty(0, dtype=np.int64),
            inc_class=np.empty(0, dtype=np.int64),
            n_legal=len(legal_qr),
            **cell_nodes,
            dec_cell=np.empty(0, dtype=np.int64),
            dec_window=np.empty(0, dtype=np.int64),
            dec_class=np.empty(0, dtype=np.int64),
            moves_remaining=moves_remaining,
            action_window_index=actions[0],
            action_post1_class=actions[1],
            action_pre_status=actions[2],
        )

    stone_key = _pack(stone_qr)
    order = np.argsort(stone_key)
    sorted_key = stone_key[order]
    if np.any(sorted_key[1:] == sorted_key[:-1]):
        raise ValueError("duplicate stone coordinates")

    # Candidate windows: every (axis, start) through some stone — 18 per stone,
    # start = stone - k * axis for k in 0..5 (§3.2's builder walk).
    ks = np.arange(WINDOW_LEN, dtype=np.int64)
    # (n_s, 3, 6, 2): stone i, axis a, offset k.
    starts = stone_qr[:, None, None, :] - AXES[None, :, None, :] * ks[None, None, :, None]
    axis_idx = np.broadcast_to(np.arange(3, dtype=np.int64)[None, :, None], starts.shape[:3])
    wkey = _pack(starts.reshape(-1, 2)) * 4 + axis_idx.reshape(-1)
    uniq_key = np.unique(wkey)

    # Occupancy of each candidate: 6 cells, each looked up in the stone set.
    u_axis = uniq_key & 3
    u_start_packed = uniq_key >> 2  # arithmetic shift keeps the sign
    # Invert _pack: floor divmod puts a negative r into the high half of the
    # remainder range, since |r| stays far below _QSHIFT / 2.
    q, rem = np.divmod(u_start_packed, _QSHIFT)
    r = rem.copy()
    high = rem >= _QSHIFT // 2
    r[high] -= _QSHIFT
    q[high] += 1
    u_start = np.stack([q, r], axis=1)

    cells = u_start[:, None, :] + AXES[u_axis][:, None, :] * ks[None, :, None]  # (n_c, 6, 2)
    cell_key = _pack(cells.reshape(-1, 2))
    pos = np.searchsorted(sorted_key, cell_key)
    pos_clip = np.minimum(pos, n_s - 1)
    hit = sorted_key[pos_clip] == cell_key
    occupant = np.where(hit, order[pos_clip], -1).reshape(-1, WINDOW_LEN)  # stone index or -1
    occ_own = (occupant >= 0) & (stone_own[np.maximum(occupant, 0)] == 0)
    occ_opp = (occupant >= 0) & (stone_own[np.maximum(occupant, 0)] == 1)
    # Every candidate is nonempty (it came through a stone), so all are kept.
    # The ternary pattern carries the colours; a full six-own or six-opp digit
    # string is terminal-only, and terminal positions were refused above.
    keep = np.ones(len(uniq_key), dtype=bool)
    pattern = ((occ_own + 2 * occ_opp) * _POW3[None, :]).sum(axis=1)
    window_feat = _TERN_RANK[pattern]
    live_key = uniq_key[keep]
    window_id = np.column_stack([u_axis[keep], u_start[keep, 0], u_start[keep, 1]])

    # Incidence: one entry per occupied slot of each kept window. The class is
    # joint in the window's occupancy and the stone's own slot (§4.3), off the
    # raw pattern in slot order — `pattern`, like the decoder classes below.
    l_occupant = occupant[keep]  # (n_w, 6)
    w_idx, slot = np.nonzero(l_occupant >= 0)
    inc_stone = l_occupant[w_idx, slot]
    inc_window = w_idx.astype(np.int64)
    inc_class = _TERN_OCC_CLASS[pattern[w_idx], slot]
    if inc_class.size and inc_class.min() < 0:
        bad = int(np.argmin(inc_class))
        raise ValueError(
            f"incidence entry {bad} pairs window pattern "
            f"{int(pattern[w_idx[bad]])} with slot {int(slot[bad])}, "
            f"which that window does not occupy"
        )

    # Decoder table: each legal cell's represented windows, by the same 18-candidate
    # walk matched against the live set.
    n_legal = len(legal_qr)
    c_starts = legal_qr[:, None, None, :] - AXES[None, :, None, :] * ks[None, None, :, None]
    c_axis = np.broadcast_to(np.arange(3, dtype=np.int64)[None, :, None], c_starts.shape[:3])
    c_key = _pack(c_starts.reshape(-1, 2)) * 4 + c_axis.reshape(-1)
    wpos = np.searchsorted(live_key, c_key)
    wpos_clip = np.minimum(wpos, max(len(live_key) - 1, 0))
    c_hit = (live_key[wpos_clip] == c_key) if len(live_key) else np.zeros(len(c_key), bool)
    flat = np.nonzero(c_hit)[0]
    dec_cell = flat // (3 * WINDOW_LEN)
    dec_window = wpos_clip[flat]
    # The class is joint in the window's occupancy and the candidate's own slot
    # (§4.3), so it needs the window's raw pattern in slot order — `pattern`,
    # not the canonicalized rank the window embedding carries.
    dec_class = _TERN_DEC_CLASS[pattern[dec_window], flat % WINDOW_LEN]
    if dec_class.size and dec_class.min() < 0:
        bad = int(np.argmin(dec_class))
        raise ValueError(
            f"decoder entry {bad} pairs window pattern "
            f"{int(pattern[dec_window[bad]])} with slot "
            f"{int(flat[bad] % WINDOW_LEN)}, which that window already occupies"
        )

    actions = _action_tables(legal_qr, sorted_key, order, stone_own, live_key)
    return PositionGraph(
        stone_own=stone_own,
        stone_qr=stone_qr,
        window_feat=window_feat,
        window_id=window_id,
        inc_stone=inc_stone,
        inc_window=inc_window,
        inc_class=inc_class,
        n_legal=n_legal,
        **cell_nodes,
        dec_cell=dec_cell,
        dec_window=dec_window,
        dec_class=dec_class,
        moves_remaining=moves_remaining,
        action_window_index=actions[0],
        action_post1_class=actions[1],
        action_pre_status=actions[2],
    )


def from_position(pos) -> PositionGraph:
    """Build from a ``hexo_py.Position``. Terminal positions raise."""
    if pos.is_terminal:
        raise ValueError("terminal position: the builder refuses it")
    stones = pos.stones()
    if stones:
        arr = np.asarray(stones, dtype=np.int64)
        stone_qr, stone_owner = arr[:, :2], arr[:, 2]
    else:
        stone_qr = np.empty((0, 2), dtype=np.int64)
        stone_owner = np.empty(0, dtype=np.int64)
    legal = np.asarray(pos.legal_moves(), dtype=np.int64).reshape(-1, 2)
    return build(
        stone_qr,
        stone_owner,
        pos.current_player,
        legal,
        pos.moves_remaining,
    )


@dataclass
class Batch:
    """A collated batch: concatenated entities plus padded attention tables.

    Every index tensor is precomputed here; the forward performs no
    data-dependent index discovery (§9). Attention and the value readout use
    per-position padded layouts with four global rows in the leading slots.
    """

    n_pos: int
    # Concatenated entity features.
    stone_own: torch.Tensor  # (N_s,) long
    window_feat: torch.Tensor  # (N_w,) long
    # Window identities (axis, start_q, start_r), each in its position's own
    # frame. The model consumes them only through reversal-invariant pair
    # classes (§5.1c), never as raw coordinates.
    window_id: torch.Tensor  # (N_w, 3) long
    moves_idx: torch.Tensor  # (P,) long: moves_remaining - 1
    # Incidence, with window/stone indices globally offset.
    inc_stone: torch.Tensor  # (E,) long
    inc_window: torch.Tensor  # (E,) long
    inc_class: torch.Tensor  # (E,) long
    # Stone-attention padding: rows [global context; stones] per position.
    max_t: int
    stone_slot: torch.Tensor  # (N_s,) long, flat index into (P * max_t)
    coords: torch.Tensor  # (P, max_t, 2) int32; global rows and padding are zero
    attn_valid: torch.Tensor  # (P, max_t) bool
    # Value-readout padding: rows [pooled global; windows] per position.
    max_w: int
    window_slot: torch.Tensor  # (N_w,) long, flat index into (P * max_w)
    value_valid: torch.Tensor  # (P, max_w) bool
    # Policy decoder, cells concatenated in engine order per position.
    n_cells: int
    legal_offsets: torch.Tensor  # (P + 1,) long
    cell_pos: torch.Tensor  # (N_c,) long: position of each cell
    cell_occupancy: torch.Tensor  # (N_c,) EMPTY/OWN/OPP relative to mover
    cell_is_legal: torch.Tensor  # (N_c,) bool-as-long
    cell_nearest: torch.Tensor  # (N_c,) 1..8, or opening sentinel 9
    radius_src: torch.Tensor  # (E_r,) global stone source
    radius_dst: torch.Tensor  # (E_r,) global legal-cell destination
    radius_orbit: torch.Tensor  # (E_r,) exact D6 class
    radius_own: torch.Tensor  # (E_r,) source own/opp
    radius_on_axis: torch.Tensor  # (E_r,) invariant bool
    adjacency_src: torch.Tensor  # (E_a,) global legal-cell source
    adjacency_dst: torch.Tensor  # (E_a,) global legal-cell destination
    adjacency_axis: torch.Tensor  # (E_a,) structural axis route
    dec_cell: torch.Tensor  # (E_d,) long, global cell index
    dec_window: torch.Tensor  # (E_d,) long, global window index
    dec_class: torch.Tensor  # (E_d,) long, < TERN_DEC_CLASSES
    # Cell-pass relay (§5.1b): the decoder incidence sorted once at collation
    # into CSR views — by covered cell (relabelled compactly), by window, and
    # by class — so the pass runs as contiguous segment reductions.
    relay_cell_ptr: torch.Tensor  # (covered cells + 1,) long
    relay_window: torch.Tensor  # (E_d,) long: edge windows, cell order
    relay_class: torch.Tensor  # (E_d,) long: edge classes, cell order
    relay_win_ptr: torch.Tensor  # (N_w + 1,) long
    relay_wcell: torch.Tensor  # (E_d,) long: compact edge cells, window order
    relay_cls_ptr: torch.Tensor  # (TERN_DEC_CLASSES + 1,) long
    relay_ccell: torch.Tensor  # (E_d,) long: compact edge cells, class order
    # The §5.1c window-pair views are not collated: a window_attention model
    # derives them on its own device from window_id — the edge views cost
    # several times more to ship than to derive beside the model.
    #
    # Action-row tables. The kept rows (a nonempty candidate window through
    # the action cell) are in bijection with the decoder incidence — the same
    # 18-candidate walk in the same order — so they ride the ``dec_*`` views
    # and only add their post-placement class; EMPTY rows collapse to
    # per-orbit counts against the shared empty base.
    act_class: torch.Tensor  # (E_d,) long, < TERN_POST1_CLASSES
    act_rev: torch.Tensor  # (E_d,) long: window-major edge order
    act_empty: torch.Tensor  # (N_c, 3) long: EMPTY rows per orbit

    def to(self, device) -> "Batch":
        """The same batch with every tensor on ``device``."""
        moved = {
            name: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for name, v in vars(self).items()
        }
        return Batch(**moved)

    def pin_memory(self) -> "Batch":
        """The same batch in pinned host memory, so ``to`` is a true async DMA.

        ``non_blocking`` silently degrades to a synchronous staged copy from
        pageable memory; a prefetch worker pins ahead of the transfer instead.
        """
        pinned = {
            name: (v.pin_memory() if isinstance(v, torch.Tensor) else v)
            for name, v in vars(self).items()
        }
        return Batch(**pinned)


_RELAY_FIELDS = (
    "relay_cell_ptr",
    "relay_window",
    "relay_class",
    "relay_win_ptr",
    "relay_wcell",
    "relay_cls_ptr",
    "relay_ccell",
)


def _relay_fields(
    dec_cell: torch.Tensor,
    dec_window: torch.Tensor,
    dec_class: torch.Tensor,
    n_windows: int,
) -> dict:
    tables = relay_tables(
        dec_cell, dec_window, dec_class, n_windows, TERN_DEC_CLASSES
    )
    return dict(zip(_RELAY_FIELDS, tables))


def batch_from_arrays(**fields) -> Batch:
    """A ``Batch`` from per-tensor arrays, with the derived tables built here.

    Both external construction paths land on this one function — the
    ``hexo_py.build_batch*`` array dicts unpacked as kwargs, and the embedded
    Rust forward calling with torch tensors — so the collation-time relay
    derivation lives in exactly one place. The scalar fields are derived from
    tensor shapes; callers may also pass them, and a disagreement is refused.
    """
    scalars = {
        name: int(fields.pop(name))
        for name in ("n_pos", "max_t", "max_w", "n_cells")
        if name in fields
    }
    t = {name: torch.as_tensor(value) for name, value in fields.items()}
    derived = {
        "n_pos": int(t["attn_valid"].shape[0]),
        "max_t": int(t["attn_valid"].shape[1]),
        "max_w": int(t["value_valid"].shape[1]),
        "n_cells": int(t["cell_pos"].shape[0]),
    }
    for name, value in scalars.items():
        if value != derived[name]:
            raise ValueError(f"{name}={value} disagrees with the derived {derived[name]}")
    return Batch(
        **derived,
        **t,
        **_relay_fields(
            t["dec_cell"],
            t["dec_window"],
            t["dec_class"],
            int(t["window_feat"].shape[0]),
        ),
    )


def collate_positions(positions) -> Batch:
    """Build and collate positions with the Rust builder.

    ``hexo_py.build_batch`` runs in parallel with the GIL released and returns
    the same fields as ``collate([from_position(p) ...])`` under
    ``MODEL_REPR_VERSION``.
    """
    from . import _rust as hexo_py

    return batch_from_arrays(**hexo_py.build_batch(list(positions)))


def collate_prefixes(games, ts) -> Batch:
    """Move prefixes to one collated batch: replay + build, in parallel.

    Stored fitting positions are move prefixes
    (``docs/KLENT_FOR_HEXO.md`` §4.3).
    """
    from . import _rust as hexo_py

    return batch_from_arrays(**hexo_py.build_batch_prefixes(list(games), list(ts)))


_ACTION_SLOT_ORBIT = np.minimum(np.arange(WINDOW_LEN), WINDOW_LEN - 1 - np.arange(WINDOW_LEN))


def _action_fields(graphs: list[PositionGraph], dec_window: torch.Tensor) -> dict:
    """The Step 4 collated views, from the per-position dense row tables.

    The kept-row/decoder bijection is asserted per position — the two tables
    come from the same walk, and a silent divergence here would misclass every
    row downstream (the symmetric-bug hazard the audit tier exists for).
    """
    classes, counts = [], []
    for g in graphs:
        status = g.action_pre_status.reshape(g.n_legal, -1)
        kept = (status != ACTION_EMPTY).ravel()
        flat = np.nonzero(kept)[0]
        if not np.array_equal(flat // (3 * WINDOW_LEN), g.dec_cell):
            raise ValueError("action-row cells disagree with the decoder walk")
        if not np.array_equal(g.action_window_index.ravel()[kept], g.dec_window):
            raise ValueError("action-row windows disagree with the decoder walk")
        classes.append(g.action_post1_class.ravel()[kept])
        empty = (status == ACTION_EMPTY).reshape(g.n_legal, 3, WINDOW_LEN)
        counts.append(
            np.stack(
                [empty[:, :, _ACTION_SLOT_ORBIT == o].sum(axis=(1, 2)) for o in range(ACTION_EMPTY_ORBITS)],
                axis=1,
            )
        )
    act_class = np.concatenate(classes) if classes else np.empty(0, dtype=np.int64)
    return {
        "act_class": torch.from_numpy(act_class.astype(np.int64)),
        "act_rev": torch.from_numpy(
            np.argsort(dec_window.numpy(), kind="stable").astype(np.int64)
        ),
        "act_empty": torch.from_numpy(
            np.concatenate(counts).astype(np.int64).reshape(-1, ACTION_EMPTY_ORBITS)
        ),
    }


def collate(graphs: list[PositionGraph]) -> Batch:
    """Concatenate position graphs into one batch (§9)."""
    if not graphs:
        raise ValueError("empty batch")
    p = len(graphs)
    ns = np.array([g.n_stones for g in graphs])
    nw = np.array([g.n_windows for g in graphs])
    nl = np.array([g.n_legal for g in graphs])
    stone_off = np.concatenate([[0], np.cumsum(ns)])
    win_off = np.concatenate([[0], np.cumsum(nw)])
    cell_off = np.concatenate([[0], np.cumsum(nl)])

    global_rows = 4
    max_t = int(ns.max()) + global_rows
    max_w = int(nw.max()) + 1

    coords = np.zeros((p, max_t, 2), dtype=np.int32)
    attn_valid = np.zeros((p, max_t), dtype=bool)
    attn_valid[:, :global_rows] = True
    value_valid = np.zeros((p, max_w), dtype=bool)
    value_valid[:, 0] = True
    for i, g in enumerate(graphs):
        coords[i, global_rows : global_rows + g.n_stones] = g.stone_qr
        attn_valid[i, global_rows : global_rows + g.n_stones] = True
        value_valid[i, 1 : 1 + g.n_windows] = True

    def cat(parts, dtype=np.int64):
        return torch.from_numpy(np.concatenate(parts).astype(dtype)) if parts else torch.empty(0, dtype=torch.long)

    stone_slot = cat(
        [i * max_t + global_rows + np.arange(g.n_stones) for i, g in enumerate(graphs)]
    )
    window_slot = cat([i * max_w + 1 + np.arange(g.n_windows) for i, g in enumerate(graphs)])
    window_id = cat([g.window_id for g in graphs]).view(-1, 3)
    dec_cell = cat([g.dec_cell + cell_off[i] for i, g in enumerate(graphs)])
    dec_window = cat([g.dec_window + win_off[i] for i, g in enumerate(graphs)])
    dec_class = cat([g.dec_class for g in graphs])

    return Batch(
        n_pos=p,
        stone_own=cat([g.stone_own for g in graphs]),
        window_feat=cat([g.window_feat for g in graphs]),
        window_id=window_id,
        moves_idx=torch.tensor([g.moves_remaining - 1 for g in graphs], dtype=torch.long),
        inc_stone=cat([g.inc_stone + stone_off[i] for i, g in enumerate(graphs)]),
        inc_window=cat([g.inc_window + win_off[i] for i, g in enumerate(graphs)]),
        inc_class=cat([g.inc_class for g in graphs]),
        max_t=max_t,
        stone_slot=stone_slot,
        coords=torch.from_numpy(coords),
        attn_valid=torch.from_numpy(attn_valid),
        max_w=max_w,
        window_slot=window_slot,
        value_valid=torch.from_numpy(value_valid),
        n_cells=int(cell_off[-1]),
        legal_offsets=torch.from_numpy(cell_off.astype(np.int64)),
        cell_pos=cat([np.full(g.n_legal, i) for i, g in enumerate(graphs)]),
        cell_occupancy=cat([g.cell_occupancy for g in graphs]),
        cell_is_legal=cat([g.cell_is_legal for g in graphs]),
        cell_nearest=cat([g.cell_nearest for g in graphs]),
        radius_src=cat([g.radius_src + stone_off[i] for i, g in enumerate(graphs)]),
        radius_dst=cat([g.radius_dst + cell_off[i] for i, g in enumerate(graphs)]),
        radius_orbit=cat([g.radius_orbit for g in graphs]),
        radius_own=cat([g.radius_own for g in graphs]),
        radius_on_axis=cat([g.radius_on_axis for g in graphs]),
        adjacency_src=cat(
            [g.adjacency_src + cell_off[i] for i, g in enumerate(graphs)]
        ),
        adjacency_dst=cat(
            [g.adjacency_dst + cell_off[i] for i, g in enumerate(graphs)]
        ),
        adjacency_axis=cat([g.adjacency_axis for g in graphs]),
        dec_cell=dec_cell,
        dec_window=dec_window,
        dec_class=dec_class,
        **_relay_fields(dec_cell, dec_window, dec_class, int(win_off[-1])),
        **_action_fields(graphs, dec_window),
    )
