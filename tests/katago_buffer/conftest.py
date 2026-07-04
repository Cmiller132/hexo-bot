"""Shared fixtures for the katago-buffer tests.

These tests once read real shards from a private development-run live tree
(copied in by ``_p5_setup_scratch.sh``). That tree does not exist publicly, so we
SYNTHESIZE equivalent ``hexfield_compact_v1`` shards here (see ``_shard_gen``) and
expose them through fixtures:

* ``paths`` — a handful of freshly synthesized shard paths (>= 3 non-empty shards
  spread across a couple of epoch subdirs, several carrying standing-hot cells)
  in a per-module tmp dir. Consumed by ``test_p3_window.py``.

* an autouse, session-scoped fixture that populates ``_scratch/p5/samples`` with
  several epoch dirs of synthesized shards if it is empty. ``test_p5_e2e.py`` and
  ``test_p7_rust_parity.py`` read that fixed path directly (module-level
  ``SAMPLES``), so the fixture just has to guarantee it is non-empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _shard_gen import generate_samples_tree  # noqa: E402


# ---------------------------------------------------------------------------
# p3: a small set of synthesized shard paths across a couple of epochs.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def paths(tmp_path_factory) -> list[str]:
    """Synthesize >= 3 non-empty compact shards across two epoch subdirs and
    return their ``.npz`` paths. Several rows carry standing-hot cells so the
    concat test's qr-CSR rebase assertion is exercised."""
    root = tmp_path_factory.mktemp("p3_samples")
    generate_samples_tree(
        root, epochs=2, games_per_epoch=3, max_plies=24, base_seed=4100, hot_first=True
    )
    shard_paths = [str(p) for p in sorted(root.glob("epoch_*/game_*.npz"))]
    assert len(shard_paths) >= 3, f"expected >= 3 synthesized shards, got {len(shard_paths)}"
    return shard_paths


# ---------------------------------------------------------------------------
# p5 / p7: ensure _scratch/p5/samples is populated with synthesized shards.
# ---------------------------------------------------------------------------
_P5_SAMPLES = _HERE / "_scratch" / "p5" / "samples"


@pytest.fixture(scope="session", autouse=True)
def _ensure_p5_scratch_samples() -> None:
    """If ``_scratch/p5/samples`` has no shards, synthesize several epoch dirs of
    games there (replacing the retired ``_p5_setup_scratch.sh`` copy of private
    live data). Leaves an already-populated tree untouched."""
    if _P5_SAMPLES.exists() and any(_P5_SAMPLES.glob("epoch_*/game_*.npz")):
        return
    _P5_SAMPLES.mkdir(parents=True, exist_ok=True)
    # A few epochs x several games each: enough rows for the taper/window math to
    # clear its (test-lowered) min_rows floor and for the rust/serial parity sweep
    # to have a few hundred rows, while staying fast on CPU.
    generate_samples_tree(
        _P5_SAMPLES, epochs=3, games_per_epoch=8, max_plies=24, base_seed=5000
    )
