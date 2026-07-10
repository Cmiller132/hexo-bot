"""hexfield_eq serve planner parity: Rust ``debug_plan_groups`` (the boundary
planner inside ``build_serve_groups``, serve_pack.rs) == Python
``inference.plan_groups``, element-for-element (start, end, pad_to).

Modeled on ``scripts/_hexfield_plan_groups_parity.py`` (which covers only the
hexfield lineage). Beyond that battery this also pins the two hexfield_eq
planner behaviors the Rust side re-synced to:

  1. MAX_GROUP_ROWS — 300 equal-size rows fit the pair ceiling at pad 128, so
     the 260-row "stay graph-capturable" cap is the binding constraint and the
     plan must split 260 + 40 on BOTH sides.
  2. HEXFIELD_PAIR_CEILING / HEXFIELD_WASTE_FRACTION env overrides — both
     planners read the env ONCE (Python at module import, Rust at first
     planner use), so the override cases run in a fresh subprocess with the
     monkeypatched env, where both sides see the override from a cold start.

Runs in the hexgt-build venv via PYTHONPATH=packages/hexfield_eq/python. The
plans are integer tuples, so every comparison is exact equality.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest

try:
    from hexfield_eq import _rust
except ImportError:  # pragma: no cover
    _rust = None

needs_rust = pytest.mark.skipif(
    _rust is None, reason="hexfield_eq._rust not built (see the Phase-1 build gate)"
)


def _battery() -> list[tuple[str, list[int]]]:
    """The size battery from scripts/_hexfield_plan_groups_parity.py plus the
    hexfield_eq row-cap case (deterministic: fixed seed)."""
    cases: list[tuple[str, list[int]]] = [("empty", [])]
    for n in (1, 64, 65, 127, 128, 129, 65535):
        cases.append((f"single[{n}]", [n]))
    # Row cap: 300 equal rows, pair ceiling non-binding at pad 128.
    cases.append(("row-cap 100 x300", [100] * 300))
    cases.append(("waste-floor 1024/840/839", [1024] + [840] * 5 + [839] * 5))
    cases.append(("waste straddle desc", [1024, 1023, 840, 839, 838]))
    cases.append(("ceiling 256 x600", [256] * 600))
    cases.append(("ceiling 500 x200 (pad512)", [500] * 200))
    cases.append(("ceiling 1000 x80 (pad1024)", [1000] * 80))
    rng = random.Random(7)
    for t in range(40):
        sizes = sorted(
            (rng.randint(1, 2560) for _ in range(rng.randint(1, 300))), reverse=True
        )
        cases.append((f"rand-desc#{t}", sizes))
    for t in range(10):
        cases.append(
            (f"rand-unsorted#{t}", [rng.randint(1, 2560) for _ in range(rng.randint(1, 200))])
        )
    return cases


def _assert_parity(plan_groups, label: str, sizes: list[int]) -> None:
    rs = [tuple(int(y) for y in x) for x in _rust.debug_plan_groups([int(s) for s in sizes])]
    py = [tuple(int(y) for y in x) for x in plan_groups(np.asarray(sizes, dtype=np.int64))]
    assert rs == py, f"{label}: rust {rs[:8]} != python {py[:8]} ({len(rs)} vs {len(py)} groups)"


@needs_rust
def test_plan_groups_parity_default_env() -> None:
    from hexfield_eq.inference import plan_groups

    for label, sizes in _battery():
        _assert_parity(plan_groups, label, sizes)


@needs_rust
def test_plan_groups_row_cap_binds_at_260() -> None:
    """The [100]*300 case really exercises MAX_GROUP_ROWS: the first group is
    exactly 260 rows on both sides (not the whole 300)."""
    from hexfield_eq.inference import MAX_GROUP_ROWS, plan_groups

    assert MAX_GROUP_ROWS == 260
    sizes = [100] * 300
    py = [tuple(int(y) for y in x) for x in plan_groups(np.asarray(sizes, dtype=np.int64))]
    rs = [tuple(int(y) for y in x) for x in _rust.debug_plan_groups(sizes)]
    assert py == [(0, 260, 128), (260, 300, 128)]
    assert rs == py


# Child process body for the env-override cases: import BOTH planners fresh
# (so the module-import env read and the Rust once-read see the override),
# verify Python honored the env, then assert parity over a compact battery
# that the override actually reshapes.
_ENV_CHILD = textwrap.dedent(
    """
    import random
    import sys

    import numpy as np

    from hexfield_eq import _rust
    from hexfield_eq import inference

    expect_ceiling = float(sys.argv[1])
    expect_waste = float(sys.argv[2])
    assert inference.PAIR_CEILING == expect_ceiling, (
        inference.PAIR_CEILING, expect_ceiling)
    assert inference.WASTE_FRACTION == expect_waste, (
        inference.WASTE_FRACTION, expect_waste)

    cases = [
        [100] * 300,
        [500] * 200,
        [1024] + [840] * 5 + [839] * 5,
        [1024, 1023, 840, 839, 838],
        [256] * 600,
    ]
    rng = random.Random(11)
    for _ in range(20):
        cases.append(sorted(
            (rng.randint(1, 2560) for _ in range(rng.randint(1, 300))),
            reverse=True,
        ))
    for sizes in cases:
        rs = [tuple(int(y) for y in x) for x in _rust.debug_plan_groups(sizes)]
        py = [tuple(int(y) for y in x)
              for x in inference.plan_groups(np.asarray(sizes, dtype=np.int64))]
        assert rs == py, (sizes[:4], len(sizes), rs[:6], py[:6])
    print("PASS")
    """
)


def _run_env_child(monkeypatch, env_updates: dict[str, str], expect_ceiling: float, expect_waste: float) -> None:
    for key, value in env_updates.items():
        monkeypatch.setenv(key, value)
    result = subprocess.run(
        [sys.executable, "-c", _ENV_CHILD, repr(expect_ceiling), repr(expect_waste)],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"env-override child failed under {env_updates}:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PASS" in result.stdout


@needs_rust
def test_plan_groups_parity_pair_ceiling_env_override(monkeypatch) -> None:
    # A tight ceiling forces many small groups; both sides must agree.
    _run_env_child(
        monkeypatch, {"HEXFIELD_PAIR_CEILING": "1e6"}, expect_ceiling=1e6, expect_waste=0.18
    )


@needs_rust
def test_plan_groups_parity_waste_fraction_env_override(monkeypatch) -> None:
    # A looser waste bound merges more rows per group; both sides must agree.
    _run_env_child(
        monkeypatch,
        {"HEXFIELD_WASTE_FRACTION": "0.5"},
        expect_ceiling=3.8e7,
        expect_waste=0.5,
    )


@needs_rust
def test_plan_groups_parity_both_env_overrides(monkeypatch) -> None:
    # Combined override, including a waste fraction > 1.0 whose Python floor
    # goes NEGATIVE (never splits on waste) — the i64 floor path in Rust.
    _run_env_child(
        monkeypatch,
        {"HEXFIELD_PAIR_CEILING": "5e6", "HEXFIELD_WASTE_FRACTION": "1.5"},
        expect_ceiling=5e6,
        expect_waste=1.5,
    )
