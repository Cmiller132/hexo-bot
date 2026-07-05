"""Unit tests for SHOWCASE_DEVICE resolution and the startup parity
self-check. The CI/dev machines are CPU-only-torch (no `torch.xpu` attribute,
no CUDA device), which is exactly the environment the fallback paths must
survive: an accelerator request degrades to cpu with a note, never a crash.
The XPU-positive branches can only run on the A310 host (deploy-time gate)."""

from __future__ import annotations

import pytest
import torch

from showcase.config import Settings
from showcase.device import (
    SELFCHECK_TOL,
    resolve_device,
    selfcheck_wanted,
    verify_device,
    xpu_available,
)

_HAS_ACCEL = bool(torch.cuda.is_available() or xpu_available())


@pytest.mark.skipif(_HAS_ACCEL, reason="host has an accelerator; auto won't pick cpu")
def test_auto_resolves_cpu_without_accelerator():
    resolved = resolve_device("auto")
    assert resolved.device == "cpu"
    assert resolved.requested == "auto"
    assert resolved.notes == ()  # auto->cpu is not a fallback, no warning


def test_explicit_cpu_is_cpu_everywhere():
    resolved = resolve_device("cpu")
    assert resolved.device == "cpu"
    assert resolved.notes == ()


@pytest.mark.skipif(xpu_available(), reason="host actually has an XPU")
def test_xpu_request_without_xpu_falls_back_with_warning():
    resolved = resolve_device("xpu")
    assert resolved.device == "cpu"  # degraded, not crashed
    assert any("falling back to cpu" in note for note in resolved.notes)


@pytest.mark.skipif(torch.cuda.is_available(), reason="host actually has CUDA")
def test_cuda_request_without_cuda_falls_back_with_warning():
    resolved = resolve_device("cuda")
    assert resolved.device == "cpu"
    assert any("falling back to cpu" in note for note in resolved.notes)


def test_request_is_normalized_and_validated():
    assert resolve_device("  CPU ").device == "cpu"
    assert resolve_device("").requested == "auto"  # empty -> default
    with pytest.raises(ValueError, match="SHOWCASE_DEVICE"):
        resolve_device("tpu")


def test_xpu_available_is_false_on_cpu_torch():
    # CPU-only torch has no torch.xpu attribute; the getattr guard (and the
    # absent-ipex fallback import) must return False, not raise.
    if not hasattr(torch, "xpu"):
        assert xpu_available() is False


def test_selfcheck_default_on_for_accelerators_only():
    assert selfcheck_wanted(None, "xpu") is True
    assert selfcheck_wanted(None, "cuda") is True
    assert selfcheck_wanted(None, "cpu") is False
    assert selfcheck_wanted(False, "xpu") is False  # explicit off wins
    assert selfcheck_wanted(True, "cpu") is True    # explicit on wins


def test_verify_device_cpu_reference_path():
    """The full compare machinery, exercised with device==cpu (the reference
    compared against itself): builds the synthetic position, runs both
    forwards, and must agree within tolerance with finite outputs."""
    from hexfield.model import HexfieldNet

    model = HexfieldNet().eval()
    result = verify_device(model, "cpu")
    assert result.ok
    assert result.error is None
    assert result.value_diff < SELFCHECK_TOL
    assert result.policy_diff < SELFCHECK_TOL
    # verify_device leaves the model on the target device.
    assert next(model.parameters()).device.type == "cpu"


def test_settings_device_env_parsing(monkeypatch):
    monkeypatch.setenv("SHOWCASE_DEVICE", " XPU ")
    monkeypatch.setenv("SHOWCASE_DEVICE_SELFCHECK", "0")
    settings = Settings.from_env()
    assert settings.device == "xpu"
    assert settings.device_selfcheck is False

    monkeypatch.delenv("SHOWCASE_DEVICE")
    monkeypatch.setenv("SHOWCASE_DEVICE_SELFCHECK", "true")
    settings = Settings.from_env()
    assert settings.device == "auto"          # default
    assert settings.device_selfcheck is True

    monkeypatch.delenv("SHOWCASE_DEVICE_SELFCHECK")
    assert Settings.from_env().device_selfcheck is None  # default: on if not cpu

    monkeypatch.setenv("SHOWCASE_DEVICE_SELFCHECK", "maybe")
    with pytest.raises(ValueError, match="SHOWCASE_DEVICE_SELFCHECK"):
        Settings.from_env()
