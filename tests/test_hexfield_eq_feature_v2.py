"""Featurizer version gate — SPEC_RAYTAP_CONV.md §1, tests T1/T2 (Phase F).

T1 — featurizer parity + regression:

  * v1 REGRESSION (the load-bearing half, spec §9.1): under the default env
    (HEXFIELD_EQ_FEATURE_VERSION unset or "1") the featurizer output is
    byte-identical to the pre-change output, pinned by a sha256 golden captured
    from the pre-change tree (branch base 4b5ffc8a) on a fixed 26-state corpus
    — python oracle AND rust ``featurize_states``, feats + raylen bytes.
  * v2 PARITY: the full rust-parity battery (serve featurizer, all-12 D6
    expand, axis permutation) re-run in a child interpreter under
    HEXFIELD_EQ_FEATURE_VERSION=2 — test_hexfield_eq_rust_parity.py derives its
    plane sets from the constants, so the same assertions cover planes 23-45,
    including the empty-board cases of spec §1.4 (ply-1 states featurize the
    post-opening supports; the corpus-hash child covers the ply-0 board).
  * v2 SEMANTICS: empty-board scalar values, fork re-index + shared-plane
    consistency against a v1 dump, liveK monotonicity and corpus coverage, and
    an independent recompute of the three new scalar planes.

T2 — typing regeneration:

  * v2 typing sets (_SCALAR_PLANES = {0..10, 41..45}, _AXIS_PLANES = {11..40});
  * stem-lift structure per derivation §8 against the active map (both
    versions — the fork re-index trap);
  * the full-net D6 equivariance gate re-run under the 46-plane input rep
    (child pytest of test_hexfield_eq_equivariance.py).

Import-time env discipline: the feature version is read once at import
(constants.py), so every versioned check runs in a fresh interpreter with all
HEXFIELD* env stripped and only the version set (the subprocess pattern of
test_hexfield_eq_checkpoint_meta.py). Child logic lives in
tests/_hexfield_eq_v2_child.py. CPU-only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hexfield_eq import constants as C

try:
    from hexfield_eq import _rust
except ImportError:  # pragma: no cover
    _rust = None

needs_rust = pytest.mark.skipif(
    _rust is None, reason="hexfield_eq._rust not built (see the Phase-1 build gate)"
)

_REPO = Path(__file__).resolve().parents[1]
_CHILD = Path(__file__).with_name("_hexfield_eq_v2_child.py")

# T1 regression golden: sha256 over the child module's 26-state corpus
# (num_nodes + feats + raylen bytes per state), captured from the PRE-change
# tree with a freshly built pre-change .so. The rust featurizer output is
# byte-identical to the python oracle on this corpus (the graded /5 /6 /3
# divisions land on the same f32 values either way), so both paths share one
# golden. A failure here means the DEFAULT-env featurizer output drifted —
# that perturbs the live run (spec §9.1); do not update the golden without
# understanding exactly why it moved.
_GOLDEN_V1_SHA256 = "6eac8d6785312ed9a8833d8a11b16ec7326d79d1f74a60376c4fd0cbb03dc8f7"


def _child_env(version: str | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HEXFIELD")}
    if version is not None:
        env["HEXFIELD_EQ_FEATURE_VERSION"] = version
    return env


def _run_child(args: list[str], version: str | None, timeout: int = 900) -> dict:
    proc = subprocess.run(
        [sys.executable, str(_CHILD), *args],
        env=_child_env(version),
        capture_output=True,
        text=True,
        cwd=_REPO,
        timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"child {args} (version={version!r}) failed:\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _run_child_pytest(node_ids: list[str], version: str, timeout: int = 1800) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *node_ids, "-q"],
        env=_child_env(version),
        capture_output=True,
        text=True,
        cwd=_REPO,
        timeout=timeout,
    )
    assert proc.returncode == 0, (
        f"child pytest {node_ids} (version={version}) failed:\n"
        f"stdout={proc.stdout[-6000:]}\nstderr={proc.stderr[-3000:]}"
    )


# --- T1: version-1 regression ---------------------------------------------------


def test_v1_default_env_output_byte_identical() -> None:
    got = _run_child(["corpus-hash"], version=None)
    assert got["feature_version"] == 1
    assert got["num_features"] == 25
    assert (got["f_own_fork"], got["f_opp_fork"]) == (23, 24)
    assert got["n_axis_quantities"] == 4
    assert got["python_sha256"] == _GOLDEN_V1_SHA256
    if _rust is not None:
        assert got["rust_sha256"] == _GOLDEN_V1_SHA256


def test_v1_explicit_env_matches_default() -> None:
    got = _run_child(["corpus-hash"], version="1")
    assert got["python_sha256"] == _GOLDEN_V1_SHA256


def test_invalid_version_rejected_at_import() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import hexfield_eq.constants"],
        env=_child_env("3"),
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    assert proc.returncode != 0
    assert "HEXFIELD_EQ_FEATURE_VERSION" in proc.stderr


# --- T1: version-2 parity + semantics --------------------------------------------


@needs_rust
def test_v2_rust_parity_suite() -> None:
    """Serve parity, all-12 D6 expand parity, and the axis-permutation check of
    test_hexfield_eq_rust_parity.py, under the 46-plane map."""

    _run_child_pytest(["tests/test_hexfield_eq_rust_parity.py"], version="2")


def test_v2_semantics(tmp_path: Path) -> None:
    v1_npz = tmp_path / "v1_corpus.npz"
    dumped = _run_child(["dump", "--out", str(v1_npz)], version=None)
    assert dumped["num_features"] == 25
    got = _run_child(["v2-semantics", "--v1", str(v1_npz)], version="2")
    assert got["feature_version"] == 2
    assert got["live4_max"] > 0.0 and got["live5_max"] > 0.0


# --- T2: typing regeneration ------------------------------------------------------


def test_typing_sets_both_versions() -> None:
    got1 = _run_child(["typing"], version=None)
    assert (got1["num_features"], got1["n_axis_quantities"]) == (25, 4)
    got2 = _run_child(["typing"], version="2")
    assert (got2["num_features"], got2["n_axis_quantities"]) == (46, 10)
    assert (got2["f_own_fork"], got2["f_opp_fork"]) == (41, 42)


def test_stem_lift_structure_both_versions() -> None:
    _run_child(["stem-lift"], version=None)
    _run_child(["stem-lift"], version="2")


def test_v2_full_net_equivariance() -> None:
    """The end-to-end T2 guard: the full-net D6 equivariance gate (typed stem
    lift included) re-run under the 46-plane input rep. A mis-typed plane —
    e.g. the fork planes left in the axis set — trains fine and fails here."""

    _run_child_pytest(
        [
            "tests/test_hexfield_eq_equivariance.py::test_equivariance_from_scratch_init",
            "tests/test_hexfield_eq_equivariance.py::test_equivariance_with_randomized_params",
        ],
        version="2",
    )
