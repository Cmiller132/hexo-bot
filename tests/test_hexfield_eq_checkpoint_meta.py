"""Load-bearing checkpoint meta + env-prefix isolation for hexfield_eq.

Closes the "BUGS_FOUND" gaps that the config / env-prefix / checkpoint-meta task
targets:

  * checkpoints.py WRITES the arch self-description (group_order, c_orbit,
    feature_width / in_channels, channels, attention_heads, trunk_layout) into
    ``meta``, and ``infer_net_kwargs_from_state_dict`` READS meta FIRST so the
    exact arch is rebuilt from the checkpoint alone (no env), then strict-loads
    bit-for-bit;
  * the D4 eval config parses with EMPTY permanent anchors (the 15-plane hexfield
    BC stem is gone — a 25-plane net strict-load-fails on it);
  * the eq package reads the EQ-namespaced arch env (HEXFIELD_EQ_*) and does NOT
    read the live hexfield HEXFIELD_* names, so a mixed process cannot
    cross-configure the two trunks.

Build-agnostic: runs under both the equivariant default build (GROUP_ORDER=12)
and the passthrough build (HEXFIELD_EQ_GROUP_ORDER=1). The env-isolation checks
run child interpreters with a controlled env, independent of the parent's build.

Runs in the hexgt-build venv via PYTHONPATH=packages/hexfield_eq/python (plus the
shared testkit / opponent packages). CPU-only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from hexfield_eq import constants as C
from hexfield_eq.checkpoints import load_into, save_checkpoint
from hexfield_eq.config import (
    MultiStageEvalOpponents,
    MultiStageEvalSection,
    parse_hexfield_config,
)
from hexfield_eq.model import HexfieldNet, infer_net_kwargs_from_state_dict


# --- 1. checkpoint meta is load-bearing: rebuild the arch PURELY from meta ------


def test_arch_meta_carries_every_arch_field() -> None:
    """arch_meta records the full self-description a foreign loader needs — the
    import-time constants (group_order / c_orbit / feature_width) that are NOT
    constructor kwargs are here so a mismatch is detectable, plus the
    constructor-shaping fields."""

    model = HexfieldNet().eval()
    meta = model.arch_meta()

    # Import-time constants (a foreign process rebuilds these from meta, not env).
    assert meta["group_order"] == C.GROUP_ORDER
    assert meta["c_orbit"] == C.C_ORBIT
    assert meta["feature_width"] == C.NUM_FEATURES == 25
    assert meta["in_channels"] == C.NUM_FEATURES
    # Constructor-shaping fields.
    assert meta["channels"] == model.stem.out_channels
    assert meta["attention_heads"] == C.ATTENTION_HEADS
    assert meta["trunk_layout"] == C.TRUNK_LAYOUT
    assert meta["equivariant"] == (C.GROUP_ORDER == 12)


def test_rebuild_arch_from_meta_and_strict_load_bitwise(tmp_path: Path) -> None:
    """Build -> save -> rebuild the arch PURELY from the persisted meta (via
    infer_net_kwargs_from_state_dict) -> strict-load succeeds and every tensor is
    bit-identical."""

    torch.manual_seed(0)
    model = HexfieldNet().eval()
    orig_sd = model.state_dict()

    ckpt = tmp_path / "epoch_000007.pt"
    save_checkpoint(ckpt, model=model, optimizer=None, epoch=7, extra={"run": "meta_test"})

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = payload["meta"]
    sd = payload["model"]
    assert meta["epoch"] == 7
    assert meta["lineage"] == "hexfield_eq"

    # Rebuild the constructor kwargs from meta ONLY (no env consulted for the
    # shaping fields — channels/heads/trunk/in_channels all come from meta).
    kwargs = infer_net_kwargs_from_state_dict(sd, meta)
    assert kwargs["channels"] == meta["channels"]
    assert kwargs["attention_heads"] == meta["attention_heads"]
    assert kwargs["trunk_layout"] == meta["trunk_layout"]
    assert kwargs["in_channels"] == meta["in_channels"] == C.NUM_FEATURES

    rebuilt = HexfieldNet(**kwargs)
    # The arch matches BEFORE loading any weights: identical key set (this is what
    # makes the strict load succeed).
    assert set(rebuilt.state_dict().keys()) == set(orig_sd.keys())

    # Strict load (load_into enforces bidirectional key equality then strict=True).
    load_into(rebuilt, payload, optimizer=None)

    rebuilt_sd = rebuilt.state_dict()
    for key, val in orig_sd.items():
        assert torch.equal(rebuilt_sd[key], val), f"tensor mismatch after reload: {key}"


def test_infer_kwargs_from_state_dict_only_no_meta() -> None:
    """Even without meta, shape inference recovers channels + in_channels from the
    stem params (the meta path is preferred, this is the fallback)."""

    model = HexfieldNet().eval()
    sd = model.state_dict()
    kwargs = infer_net_kwargs_from_state_dict(sd)  # meta=None
    assert kwargs["channels"] == model.stem.out_channels
    assert kwargs["in_channels"] == C.NUM_FEATURES


def test_constructor_rejects_foreign_in_channels() -> None:
    """A checkpoint built at a different feature width fails LOUDLY at construction
    (the stem lift is import-time bound to NUM_FEATURES) instead of a cryptic
    stem shape mismatch at load."""

    with pytest.raises(ValueError, match="in_channels"):
        HexfieldNet(in_channels=C.NUM_FEATURES + 1)
    # The matching width builds fine.
    HexfieldNet(in_channels=C.NUM_FEATURES)


# --- 2. D4 config: empty anchors ----------------------------------------------


def test_eval_config_parses_with_empty_anchors() -> None:
    """Decision D4: permanent_anchors defaults to () and bc_prefit is gone from
    radius8_opponents. The 15-plane hexfield BC stem strict-load-fails on this
    25-plane net, so the stale anchors MUST be absent from the defaults."""

    opp = MultiStageEvalOpponents()
    assert opp.permanent_anchors == ()
    assert opp.radius8_opponents == ()

    section = MultiStageEvalSection()
    assert section.opponents.permanent_anchors == ()

    # A run toml that omits the anchors parses to the empty defaults.
    cfg = parse_hexfield_config({"multi_stage_eval": {}})
    assert cfg.multi_stage_eval.opponents.permanent_anchors == ()
    assert cfg.multi_stage_eval.opponents.radius8_opponents == ()

    # An explicit empty opponents table round-trips.
    cfg2 = parse_hexfield_config(
        {"multi_stage_eval": {"opponents": {"permanent_anchors": (), "radius8_opponents": ()}}}
    )
    assert cfg2.multi_stage_eval.opponents.permanent_anchors == ()


# --- 3. env-prefix isolation: HEXFIELD_EQ_* honored, HEXFIELD_* ignored --------


def _child_constants(env_overrides: dict[str, str]) -> dict:
    """Import hexfield_eq.constants in a fresh interpreter under a controlled env
    (every HEXFIELD* key stripped first, then the overrides applied) and return
    the parsed arch constants. Inherits PYTHONPATH so the package resolves."""

    child_env = {k: v for k, v in os.environ.items() if not k.startswith("HEXFIELD")}
    child_env.update(env_overrides)
    script = (
        "import json;"
        "from hexfield_eq import constants as C;"
        "print(json.dumps({"
        "'GROUP_ORDER': C.GROUP_ORDER,"
        "'CHANNELS': C.CHANNELS,"
        "'C_ORBIT': C.C_ORBIT,"
        "'ATTENTION_HEADS': C.ATTENTION_HEADS,"
        "'HEAD_DIM': C.HEAD_DIM,"
        "}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=child_env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"child import failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_hexfield_legacy_env_names_are_ignored() -> None:
    """Setting the LIVE hexfield names (HEXFIELD_GROUP_ORDER / HEXFIELD_CHANNELS /
    HEXFIELD_ATTENTION_HEADS) must NOT reconfigure the eq trunk — the defaults
    (12 / 96 / 3) survive. If the legacy names were read, GROUP_ORDER=1 +
    CHANNELS=64 + HEADS=7 would either take effect (wrong) or raise at import
    (64 % 7 != 0); a clean default import proves they are ignored."""

    got = _child_constants(
        {
            "HEXFIELD_GROUP_ORDER": "1",
            "HEXFIELD_CHANNELS": "64",
            "HEXFIELD_ATTENTION_HEADS": "7",
            "HEXFIELD_C_ORBIT": "64",
            "HEXFIELD_TRUNK": "A",
            "HEXFIELD_SUPPORT_RADIUS": "3",
        }
    )
    assert got["GROUP_ORDER"] == 12
    assert got["CHANNELS"] == 96
    assert got["ATTENTION_HEADS"] == 3


def test_hexfield_eq_env_names_are_honored() -> None:
    """The EQ-namespaced names DO reconfigure the eq trunk."""

    # Equivariant override: c=192, C_orbit=16, heads=3 (head_dim=64=4*16).
    got = _child_constants(
        {
            "HEXFIELD_EQ_GROUP_ORDER": "12",
            "HEXFIELD_EQ_CHANNELS": "192",
            "HEXFIELD_EQ_ATTENTION_HEADS": "3",
            "HEXFIELD_EQ_C_ORBIT": "16",
        }
    )
    assert got["GROUP_ORDER"] == 12
    assert got["CHANNELS"] == 192
    assert got["C_ORBIT"] == 16
    assert got["HEAD_DIM"] == 64

    # Passthrough override: GROUP_ORDER=1, arbitrary width + head count (no
    # equivariant divisibility constraints), C_ORBIT == CHANNELS.
    got2 = _child_constants(
        {
            "HEXFIELD_EQ_GROUP_ORDER": "1",
            "HEXFIELD_EQ_CHANNELS": "128",
            "HEXFIELD_EQ_ATTENTION_HEADS": "4",
        }
    )
    assert got2["GROUP_ORDER"] == 1
    assert got2["CHANNELS"] == 128
    assert got2["C_ORBIT"] == 128
    assert got2["ATTENTION_HEADS"] == 4


def test_eq_support_radius_env_is_namespaced() -> None:
    """The eq featurizer support radius reads HEXFIELD_EQ_SUPPORT_RADIUS, NOT the
    live HEXFIELD_SUPPORT_RADIUS."""

    # Legacy name ignored -> default LEGAL_RADIUS.
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("HEXFIELD")}
    child_env["HEXFIELD_SUPPORT_RADIUS"] = "4"
    script = (
        "from hexfield_eq import support as S;"
        "print(S._SUPPORT_RADIUS)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], env=child_env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert int(proc.stdout.strip().splitlines()[-1]) == C.LEGAL_RADIUS

    # EQ name honored.
    child_env2 = {k: v for k, v in os.environ.items() if not k.startswith("HEXFIELD")}
    child_env2["HEXFIELD_EQ_SUPPORT_RADIUS"] = "4"
    proc2 = subprocess.run(
        [sys.executable, "-c", script], env=child_env2, capture_output=True, text=True
    )
    assert proc2.returncode == 0, proc2.stderr
    assert int(proc2.stdout.strip().splitlines()[-1]) == 4
