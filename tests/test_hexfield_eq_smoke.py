"""Phase-0 scaffold smoke test for the hexfield_eq package.

Gate for docs/PLAN_D6_EQUIVARIANT_REWRITE.md Phase 0. Asserts the new,
isolated package is a clean/importable/buildable copy of hexfield:

  * the package + its own cdylib (hexfield_eq._rust) import,
  * the copied dense trunk builds HexfieldNet() on CPU and runs one forward on
    a tiny synthetic support with finite outputs,
  * the new arch env knobs (HEXFIELD_EQ_GROUP_ORDER / HEXFIELD_EQ_C_ORBIT)
    parsed at import with the passthrough default (GROUP_ORDER=1, NUM_FEATURES=25),
  * the plugin is discoverable through the hexo_train `hexo_train.models`
    entry-point group.

Runs in the hexgt-build venv via PYTHONPATH=packages/hexfield_eq/python (plus
the shared testkit / opponent packages). CPU-only; the Triton serve kernels are
env-gated off by default so no CUDA is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# --- 1. package + submodules import -------------------------------------------

import hexfield_eq
from hexfield_eq import constants as C
from hexfield_eq.model import HexfieldNet
from hexfield_eq.plugin import get_plugin

# Phase-0 scaffold gate: asserts the passthrough (non-equivariant) copy of
# hexfield. Phase 3b makes the equivariant order-12 tie the DEFAULT build, so run
# this suite under HEXFIELD_EQ_GROUP_ORDER=1; it self-skips under the equivariant
# build (where GROUP_ORDER=12, C_ORBIT<CHANNELS, and the trunk is tied).
pytestmark = pytest.mark.skipif(
    C.GROUP_ORDER != 1,
    reason="Phase-0 passthrough scaffold gate; run with HEXFIELD_EQ_GROUP_ORDER=1",
)


def test_package_and_submodules_import() -> None:
    # __init__ re-exports the torch-free submodules.
    for name in ("constants", "geometry", "support", "features"):
        assert hasattr(hexfield_eq, name), name


def test_scaffold_constants_defaults() -> None:
    # Phase 1 widens the feature width to 25 (11 kept scalars + 12 graded
    # per-axis window planes + 2 fork scalars); the new knobs default to the
    # non-equivariant passthrough so the copied dense trunk still builds.
    assert C.NUM_FEATURES == 25
    assert C.GROUP_ORDER == 1
    assert C.C_ORBIT == C.CHANNELS
    # HEAD_DIM stays the plain CHANNELS/heads split (no equivariant constraint at
    # GROUP_ORDER=1).
    assert C.HEAD_DIM == C.CHANNELS // C.ATTENTION_HEADS


def test_rust_extension_is_hexfield_eq() -> None:
    # The mirrored cdylib imports and self-identifies as the hexfield_eq lineage
    # (not hexfield) — proves the rename + independent build landed.
    from hexfield_eq import _rust

    caps = _rust.capabilities()
    assert caps["model_family"] == "hexfield_eq"
    assert caps["num_features"] == C.NUM_FEATURES == 25


# --- 2. build + forward on a tiny synthetic support ---------------------------


def test_build_and_forward_cpu() -> None:
    torch.manual_seed(0)
    model = HexfieldNet().eval()

    b, n = 1, 7  # a tiny synthetic support (7 nodes, 1 row)
    feats = torch.randn(b, n, C.NUM_FEATURES)
    nbr = torch.randint(0, n, (b, n, 6), dtype=torch.long)
    mask = torch.ones(b, n, dtype=torch.bool)
    coords = torch.randint(-8, 9, (b, n, 2), dtype=torch.long)

    with torch.no_grad():
        out = model(feats, nbr, mask, coords)

    # Core heads present and finite.
    assert "policy" in out and "value" in out
    for key, value in out.items():
        assert torch.isfinite(value).all(), f"non-finite forward output at {key}"
    # Per-cell policy logit lands on every support node.
    assert out["policy"].shape[-1] == n


# --- 3. plugin discovery via the hexo_train entry point -----------------------


def test_plugin_discoverable_via_entry_point() -> None:
    from hexo_train.config import ModelConfig
    from hexo_train.registry import load_model_plugin

    # (a) module-path resolution — the registry path an explicit
    #     [model].module = "hexfield_eq.plugin" config uses; works under
    #     PYTHONPATH with no installed metadata.
    by_module = load_model_plugin(
        ModelConfig(name="hexfield_eq", module="hexfield_eq.plugin")
    )
    assert by_module.name == "hexfield_eq"

    # (b) entry-point / name resolution through the hexo_train.models group. This
    #     needs the package metadata (entry_points.txt) visible to this
    #     interpreter. Assert it resolves when installed; otherwise fall back to
    #     verifying the entry point is DECLARED in the package's own pyproject so
    #     the test still positively checks the wiring rather than silently
    #     skipping.
    from importlib.metadata import entry_points

    eps = {ep.name for ep in entry_points(group="hexo_train.models")}
    if "hexfield_eq" in eps:
        by_name = load_model_plugin(ModelConfig(name="hexfield_eq"))
        assert by_name.name == "hexfield_eq"
    else:
        _assert_entry_point_declared_in_pyproject()


def _assert_entry_point_declared_in_pyproject() -> None:
    import tomllib

    pyproject = (
        Path(__file__).resolve().parents[1]
        / "packages"
        / "hexfield_eq"
        / "pyproject.toml"
    )
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    group = data["project"]["entry-points"]["hexo_train.models"]
    assert group.get("hexfield_eq") == "hexfield_eq.plugin:get_plugin", group


def test_plugin_builds_model() -> None:
    plugin = get_plugin()
    assert plugin.name == "hexfield_eq"
    model = plugin.build_model({}, {})
    assert isinstance(model, HexfieldNet)
    assert sum(p.numel() for p in model.parameters()) > 0
