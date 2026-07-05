"""shrimp geometry tests: packing, distance, D6 algebra, bias rows.

Pure-Python. The packing cross-check uses the pure-Python hexo_engine.types
implementation. shrimp is imported by inserting
packages/shrimp/python onto sys.path (see below).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "shrimp" / "python"))

from hexo_engine.types import AxialCoord, pack_coord_id

from shrimp import constants, geometry


def test_pack_matches_engine_mirror_and_roundtrips() -> None:
    rng = random.Random(7)
    for _ in range(2000):
        q = rng.randint(-2000, 2000)
        r = rng.randint(-2000, 2000)
        packed = geometry.pack_action_id(q, r)
        assert packed == pack_coord_id(AxialCoord(q=q, r=r))
        assert geometry.unpack_action_id(packed) == (q, r)


def test_pack_order_equals_signed_qr_order() -> None:
    rng = random.Random(11)
    coords = [(rng.randint(-300, 300), rng.randint(-300, 300)) for _ in range(500)]
    by_packed = sorted(coords, key=lambda c: geometry.pack_action_id(*c))
    assert by_packed == sorted(coords)


def test_hex_dist_max_form_equals_half_sum_form() -> None:
    rng = random.Random(13)
    for _ in range(2000):
        dq = rng.randint(-30, 30)
        dr = rng.randint(-30, 30)
        assert geometry.hex_dist(dq, dr) == (abs(dq) + abs(dr) + abs(dq + dr)) // 2


def test_direction_algebra() -> None:
    d = constants.DIRECTIONS
    for i in range(6):
        assert geometry.rot60(*d[i]) == d[(i + 1) % 6]
        assert geometry.reflect(*d[i]) == d[5 - i]


def test_d6_group_properties() -> None:
    rng = random.Random(17)
    cells = [(rng.randint(-50, 50), rng.randint(-50, 50)) for _ in range(100)]
    for sym in range(12):
        inv = geometry.d6_inverse(sym)
        for q, r in cells:
            tq, tr = geometry.apply_d6(sym, q, r)
            # inverse composed with the transform gives the identity
            assert geometry.apply_d6(inv, tq, tr) == (q, r)
            # transform preserves distance to the origin
            assert geometry.hex_dist(tq, tr) == geometry.hex_dist(q, r)
            # transform preserves the on_win_axis predicate
            assert geometry.on_win_axis(tq, tr) == geometry.on_win_axis(q, r)
    # identity is index 0
    assert geometry.apply_d6(0, 5, -3) == (5, -3)


def test_bias_exact_rows_are_a_bijection_over_the_disk() -> None:
    offsets = geometry.disk_offsets(constants.BIAS_DISK_RADIUS)
    assert len(offsets) == constants.BIAS_EXACT_ROWS == 217
    rows = [geometry.rel_bias_index(dq, dr) for dq, dr in offsets]
    assert sorted(rows) == list(range(217))


def test_bias_ring_and_far_rows() -> None:
    for d in range(constants.BIAS_RING_MIN, constants.BIAS_RING_MAX + 1):
        # pure-axis offset (on a win axis) at distance d
        assert geometry.rel_bias_index(d, 0) == constants.BIAS_ON_AXIS_BASE + (d - 9)
        assert geometry.rel_bias_index(0, d) == constants.BIAS_ON_AXIS_BASE + (d - 9)
        assert geometry.rel_bias_index(d, -d) == constants.BIAS_ON_AXIS_BASE + (d - 9)
        # off-axis offset at distance d (d, -1) has dq!=0, dr!=0, dq+dr!=0 for d>=2
        assert geometry.rel_bias_index(d, -1) == constants.BIAS_OFF_AXIS_BASE + (d - 9)
    assert geometry.rel_bias_index(17, 0) == constants.BIAS_FAR_ROW
    assert geometry.rel_bias_index(40, -11) == constants.BIAS_FAR_ROW


def test_bias_rows_respect_d6_classes() -> None:
    """A D6 transform preserves an offset's distance and axis class: ring and
    far rows map to themselves, exact rows map to other exact rows."""

    rng = random.Random(23)
    for _ in range(500):
        dq = rng.randint(-20, 20)
        dr = rng.randint(-20, 20)
        row = geometry.rel_bias_index(dq, dr)
        for sym in range(12):
            tq, tr = geometry.apply_d6(sym, dq, dr)
            trow = geometry.rel_bias_index(tq, tr)
            if row >= constants.BIAS_ON_AXIS_BASE:
                assert trow == row  # ring and far rows map to the same row
            else:
                assert trow < constants.BIAS_ON_AXIS_BASE  # exact rows stay exact rows
