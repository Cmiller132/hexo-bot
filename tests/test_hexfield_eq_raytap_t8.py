"""T8 — training-memory gate, full-shape half (SPEC_RAYTAP_CONV.md §6.1, K2
acceptance): one full-shape training step (B=48, S=648, C=192, layout
CCLACCLACLA, RAYTAP=both, REG_LANE=1, AMP as in production) on a 12 GB-class
device: no OOM, no batch-size reduction, and the peak CUDA memory delta vs the
RAYTAP=0 baseline recorded and <= 1.0 GB. (The small-shape gradient oracle
lives in test_hexfield_eq_raytap.py.)

OPT-IN: requires HEXFIELD_RAYTAP_T8=1 *and* CUDA. The gate allocates ~the full
card and MUST NOT run against the live soak's GPU — run it only when the GPU
is idle (spec §9.1 live-run isolation). The import env is set per child
subprocess (import-time env discipline).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[1]
_CHILD = Path(__file__).with_name("_hexfield_eq_raytap_t8_child.py")

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("HEXFIELD_RAYTAP_T8") != "1",
        reason="opt-in full-shape memory gate (HEXFIELD_RAYTAP_T8=1 on an IDLE GPU)",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="T8 needs CUDA"),
]

T8_ENV = {
    "HEXFIELD_EQ_CHANNELS": "192",
    "HEXFIELD_EQ_TRUNK": "CCLACCLACLA",
    "HEXFIELD_EQ_REG_LANE": "1",
    "HEXFIELD_EQ_REG_TOK_READ": "0",
    "HEXFIELD_EQ_SUPPORT_RADIUS": "4",
}


def _run_child(raytap: str) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HEXFIELD")}
    env.update(T8_ENV)
    env["HEXFIELD_EQ_RAYTAP"] = raytap
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(_CHILD)],
        env=env, capture_output=True, text=True, cwd=_REPO, timeout=1800,
    )
    assert proc.returncode == 0, (
        f"T8 child (raytap={raytap}) failed:\n"
        f"stdout={proc.stdout[-4000:]}\nstderr={proc.stderr[-4000:]}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_t8_full_shape_memory_gate() -> None:
    base = _run_child("0")
    rt = _run_child("both")
    assert base["loss_finite"] and rt["loss_finite"]
    delta_gb = (rt["peak_bytes"] - base["peak_bytes"]) / 2**30
    print(
        f"T8 peak: baseline {base['peak_gb']} GB, raytap-both {rt['peak_gb']} GB, "
        f"delta {delta_gb:+.3f} GB (gate <= 1.0)"
    )
    assert delta_gb <= 1.0, f"K2 memory gate exceeded: {delta_gb:+.3f} GB"
