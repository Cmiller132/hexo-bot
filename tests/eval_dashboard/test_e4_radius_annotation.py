"""E4 — radius-confound annotation + anchor exclusion.

The support radius is a process-global read once per process, so EVERY opponent
is featurized at the live HEXFIELD_SUPPORT_RADIUS. A radius-8-era anchor
(bc_prefit) forced to radius-4 plays OOD -> weaker -> inflates the candidate's
relative Elo. We cannot vary the radius per net (OnceLock + metadata-less frozen
checkpoints), so we:
  * TAG each radius-8-era edge ``featurized_ood`` + ``featurize_radius`` (flows
    into eval_pool.json rows via provenance),
  * EXCLUDE an OOD anchor from the pinned BT zero-point (kept descriptive),
  * surface ``ratings["fit"]["ood_opponents"]`` + a verdict note.

Because the radius is import-time (a process-global OnceLock), the radius-4 and
radius-8 scenarios each run in their OWN SUBPROCESS with the target
HEXFIELD_SUPPORT_RADIUS (it cannot be mutated in-process). The whole suite runs
at the default radius, so both radius scenarios are exercised via subprocesses
that this module launches from the repo root.

Run:
  python -m pytest tests/eval_dashboard/test_e4_radius_annotation.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from hexfield.config import parse_hexfield_config
from hexfield.multistage_eval import Opponent, Roster, _choose_anchor
import hexfield.eval_stats as eval_stats

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_radius_subprocess(probe: str, radius: int) -> subprocess.CompletedProcess:
    """Run a probe under HEXFIELD_SUPPORT_RADIUS=<radius> from the repo root, so
    it exercises THIS checkout's hexfield at the requested import-time radius."""

    env = dict(os.environ)
    env["HEXFIELD_SUPPORT_RADIUS"] = str(radius)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT / "packages" / "hexfield" / "python"), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def _roster():
    return Roster(
        candidate_label="cand_ep35",
        candidate_epoch=35,
        sealbot=None,
        champion=Opponent(label="ep30", role="champion", ckpt=Path("x"), epoch=30),
        opponents=(
            Opponent(label="bc_prefit", role="anchor", ckpt=Path("bc"), epoch=2),
            Opponent(label="ep30", role="champion", ckpt=Path("x"), epoch=30),
        ),
    )


_SUBPROC_R4_TAGGING = """
from pathlib import Path
from hexfield import support
from hexfield.config import parse_hexfield_config
from hexfield.multistage_eval import Opponent, Roster, _build_checkpoint_edge_from_match

assert support._SUPPORT_RADIUS == 4, support._SUPPORT_RADIUS
cfg = parse_hexfield_config({}).multi_stage_eval
roster = Roster(
    candidate_label="cand_ep35", candidate_epoch=35, sealbot=None,
    champion=Opponent("ep30", "champion", Path("x"), 30),
    opponents=(Opponent("bc_prefit", "anchor", Path("bc"), 2),
               Opponent("ep30", "champion", Path("x"), 30)),
)
bc, ep30 = roster.opponents[0], roster.opponents[1]
def _m(a, b):
    return {"score": {"decided": a + b, "a_wins": a}, "pentanomial": {}, "meta": {}}
bc_edge = _build_checkpoint_edge_from_match(roster, bc, _m(8, 0), cfg=cfg)
ep30_edge = _build_checkpoint_edge_from_match(roster, ep30, _m(5, 5), cfg=cfg)
assert bc_edge["descriptive"]["featurized_ood"] is True, bc_edge["descriptive"]
assert bc_edge["descriptive"]["featurize_radius"] == 4
assert bc_edge["descriptive"]["provenance"].get("featurized_ood") is True
assert ep30_edge["descriptive"]["featurized_ood"] is False, ep30_edge["descriptive"]
print("[E4] PASS tagging@r4: bc_prefit OOD, ep30 not-OOD, radius=4")
"""


def test_edge_tagging_radius4():
    # The radius is an import-time OnceLock; the suite's default radius is not 4,
    # so this radius-4 scenario runs in its own subprocess.
    proc = _run_radius_subprocess(_SUBPROC_R4_TAGGING, radius=4)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    assert proc.returncode == 0, f"radius-4 tagging subprocess failed:\n{proc.stderr}"
    assert "PASS tagging@r4" in proc.stdout


def test_anchor_excludes_ood():
    # bc_prefit (OOD) must NOT be chosen as anchor when a non-OOD ckpt edge exists.
    edges = [
        eval_stats.BTEdge(a="cand_ep35", b="bc_prefit", wins_a=8, wins_b=0),
        eval_stats.BTEdge(a="cand_ep35", b="ep30", wins_a=5, wins_b=5),
    ]
    roster = _roster()
    chosen = _choose_anchor(edges, roster, ood_labels={"bc_prefit"})
    assert chosen != "bc_prefit", f"OOD bc_prefit was picked as anchor: {chosen}"
    assert chosen == "ep30", f"expected non-OOD ep30 anchor, got {chosen}"

    # Without OOD exclusion, bc_prefit (anchor role) would win tier 2.
    chosen_no_excl = _choose_anchor(edges, roster, ood_labels=set())
    assert chosen_no_excl == "bc_prefit", chosen_no_excl
    print(f"[E4] PASS anchor-exclude: OOD->{chosen} (non-OOD->{chosen_no_excl})")


_SUBPROC_R4_STAGE_D = """
import tempfile
from pathlib import Path
from hexfield import support
from hexfield.config import parse_hexfield_config
from hexfield.multistage_eval import Opponent, Roster, _stage_d_pool

assert support._SUPPORT_RADIUS == 4, support._SUPPORT_RADIUS
cfg = parse_hexfield_config({}).multi_stage_eval
roster = Roster(
    candidate_label="cand_ep35", candidate_epoch=35, sealbot=None,
    champion=Opponent("ep30", "champion", Path("x"), 30),
    opponents=(Opponent("bc_prefit", "anchor", Path("bc"), 2),
               Opponent("ep30", "champion", Path("x"), 30)),
)
pool = {"edges": [
    {"epoch": 35, "a": "cand_ep35", "b": "bc_prefit", "wins_a": 8.0,
     "wins_b": 0.0, "weight": 1.0, "kind": "checkpoint", "raw": {}},
    {"epoch": 35, "a": "cand_ep35", "b": "ep30", "wins_a": 5.0,
     "wins_b": 5.0, "weight": 1.0, "kind": "checkpoint", "raw": {}},
]}
tmp = Path(tempfile.mkdtemp())
stage_d, ratings, verdict, _ = _stage_d_pool(cfg, roster, [], tmp, pool_doc=pool, append=False)
assert ratings["fit"].get("ood_opponents") == ["bc_prefit"], ratings["fit"]
assert ratings["fit"].get("anchor") != "bc_prefit", ratings["fit"]
assert verdict.get("ood_opponents") == ["bc_prefit"], verdict
assert "ood_note" in verdict, verdict
print("[E4] PASS stage-d@r4: ood_opponents=%s anchor=%s" % (
    ratings["fit"]["ood_opponents"], ratings["fit"]["anchor"]))
"""


def test_stage_d_surfaces_ood():
    # Radius-4 scenario in its own subprocess (import-time OnceLock radius).
    proc = _run_radius_subprocess(_SUBPROC_R4_STAGE_D, radius=4)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    assert proc.returncode == 0, f"radius-4 stage-d subprocess failed:\n{proc.stderr}"
    assert "PASS stage-d@r4" in proc.stdout


_SUBPROC_R8 = """
import sys
from pathlib import Path
from hexfield import support
from hexfield.config import parse_hexfield_config
from hexfield.multistage_eval import Opponent, Roster, _build_checkpoint_edge_from_match, _choose_anchor
import hexfield.eval_stats as eval_stats

assert support._SUPPORT_RADIUS == 8, support._SUPPORT_RADIUS
cfg = parse_hexfield_config({}).multi_stage_eval
roster = Roster(
    candidate_label="cand_ep35", candidate_epoch=35, sealbot=None,
    champion=Opponent("ep30", "champion", Path("x"), 30),
    opponents=(Opponent("bc_prefit", "anchor", Path("bc"), 2),
               Opponent("ep30", "champion", Path("x"), 30)),
)
bc = roster.opponents[0]
match = {"score": {"decided": 8, "a_wins": 8}, "pentanomial": {}, "meta": {}}
edge = _build_checkpoint_edge_from_match(roster, bc, match, cfg=cfg)
assert edge["descriptive"]["featurized_ood"] is False, edge["descriptive"]
assert edge["descriptive"]["featurize_radius"] == 8
# bc_prefit at native radius 8 is NOT OOD -> remains anchor-eligible.
edges = [eval_stats.BTEdge("cand_ep35", "bc_prefit", 8, 0),
         eval_stats.BTEdge("cand_ep35", "ep30", 5, 5)]
chosen = _choose_anchor(edges, roster, ood_labels=set())
assert chosen == "bc_prefit", chosen
print("[E4] PASS@r8 SUBPROCESS: bc_prefit NOT OOD, anchor-eligible, radius=8")
"""


def test_radius8_control_subprocess():
    proc = _run_radius_subprocess(_SUBPROC_R8, radius=8)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    assert proc.returncode == 0, f"radius-8 subprocess failed:\n{proc.stderr}"
    assert "PASS@r8" in proc.stdout


if __name__ == "__main__":
    test_edge_tagging_radius4()
    test_anchor_excludes_ood()
    test_stage_d_surfaces_ood()
    test_radius8_control_subprocess()
    print("E4 ALL GREEN")
