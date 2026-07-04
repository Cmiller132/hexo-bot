"""D1/D2/D3 dashboard-fix assertions for the hexfield eval data layer.

These pin the ``hexo_frontend.web`` data-layer behavior that surfaces the REAL
multi-stage eval (pinned Bradley-Terry ladder, pool ledger, roster-driven
headline) instead of the dead "no eval" path — WITHOUT reading any private live
run. Every test drives ``web`` against a SYNTHESIZED hexfield run dir built in a
``tmp_path`` fixture (see ``run_dir`` below), so the assertions pin controlled,
self-consistent values rather than frozen snapshots of an evolving live run.

Run from the repo root (WSL, with the public venv active) as:
  PYTHONPATH=packages/hexfield/python \
    python -m pytest tests/eval_dashboard/test_eval_dashboard_fixes.py -q
"""
import json
from collections import defaultdict
from pathlib import Path

import pytest

from hexo_frontend import web  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic hexfield run dir.
#
# All web.py readers are lineage-gated on ``_diag_prefix(run_dir) == "hexfield"``
# (which reads manifest.json -> model.name), so the run dir MUST carry a hexfield
# manifest or every reader returns [] / None. The multi-stage reports below are a
# small, self-consistent set (epochs 5, 30, 35) whose CONTROLLED values make the
# behavioral assertions pass:
#   * ep5  verdict REGRESS (the known startup artifact), bc_prefit in roster.
#   * ep30 verdict INCONCLUSIVE, bc_prefit still in roster.
#   * ep35 verdict PROMOTE (newest); candidate = cand_ep35 with a known Elo; the
#     ratings ladder carries BOTH cand_ep30 and bare ep30 split nodes; bc_prefit
#     is a permanent anchor but DROPPED from the ep35 roster (SEV-2 detection);
#     the primary edge (cand_ep35 vs ep30) has a 5-5 physical record.
# The eval_pool.json carries exactly EXPECTED_POOL_EDGES edges anchored on
# "sealbot", including a cand_ep35-vs-ep30 pairing whose raw physical record is
# 5-5 (so the W-L matrix cross-check against the report edge holds).
# --------------------------------------------------------------------------- #

EXPECTED_MS_REPORTS = 3          # epochs 5, 30, 35
EXPECTED_POOL_EDGES = 4          # controlled edge count in eval_pool.json
CAND_EP35_ELO = 142.5            # known candidate Elo pinned by D1
LATEST_EPOCH = 35


def _permanent_anchors_cfg() -> dict:
    # meta.config.opponents.permanent_anchors: list of [label, ...] entries; the
    # reader takes entry[0] as the anchor label.
    return {
        "opponents": {
            "permanent_anchors": [["bc_prefit", 0], ["ep5", 5]],
        },
        "full_search_visits": 400,
    }


def _sealbot_edge(winrate: float) -> dict:
    return {
        "opponent": "sealbot",
        "role": "sealbot",
        "kind": "anchor",
        "primary": False,
        "decided": False,
        "winrate": winrate,
        "winrate_ci95": [max(0.0, winrate - 0.1), min(1.0, winrate + 0.1)],
        "elo_point": 0.0,
        "provenance": {
            "physical_wins_a": 6,
            "physical_wins_b": 6,
            "eval_visits": 400,
            "n_pairs": 12,
            "opponent_search_profile": "puct",
        },
    }


def _report_ep5() -> dict:
    return {
        "meta": {
            "candidate_epoch": 5,
            "candidate_label": "cand_ep5",
            "anchor": "sealbot",
            "config": _permanent_anchors_cfg(),
            "pure_eval": True,
            "elapsed_seconds": 120.0,
        },
        "ratings": {
            "anchor": "sealbot",
            "players": [
                {"label": "cand_ep5", "elo": -30.0, "role": "candidate"},
                {"label": "sealbot", "elo": 0.0, "role": "anchor", "is_anchor": True},
                {"label": "bc_prefit", "elo": -80.0, "role": "anchor", "is_anchor": True},
            ],
            "fit": {"method": "bradley_terry"},
        },
        "verdict": {
            "label": "REGRESS",
            "primary": {"candidate": "cand_ep5"},
        },
        "edges": [
            _sealbot_edge(0.42),
            {
                "opponent": "bc_prefit",
                "role": "anchor",
                "kind": "anchor",
                "primary": False,
                "decided": True,
                "winrate": 0.55,
                "winrate_ci95": [0.4, 0.7],
                "elo_point": 30.0,
                "provenance": {
                    "physical_wins_a": 4,
                    "physical_wins_b": 3,
                    "eval_visits": 400,
                    "n_pairs": 7,
                    "opponent_search_profile": "gumbel",
                },
            },
        ],
        "roster": {
            "candidate": {"label": "cand_ep5", "epoch": 5},
            "champion": {},
            "sealbot": {"label": "sealbot"},
            "opponents": [
                {"label": "bc_prefit", "role": "anchor", "epoch": 0},
                {"label": "sealbot", "role": "anchor", "epoch": None},
            ],
        },
        "stages": [{"stage": "C_deep", "status": "ok", "opponents_played": 2}],
        "sealbot_winrate_ci95": [0.32, 0.52],
    }


def _report_ep30() -> dict:
    return {
        "meta": {
            "candidate_epoch": 30,
            "candidate_label": "cand_ep30",
            "anchor": "sealbot",
            "config": _permanent_anchors_cfg(),
            "pure_eval": True,
            "elapsed_seconds": 130.0,
        },
        "ratings": {
            "anchor": "sealbot",
            "players": [
                {"label": "cand_ep30", "elo": 90.0, "role": "candidate"},
                {"label": "sealbot", "elo": 0.0, "role": "anchor", "is_anchor": True},
                {"label": "bc_prefit", "elo": -80.0, "role": "anchor", "is_anchor": True},
            ],
            "fit": {"method": "bradley_terry"},
        },
        "verdict": {
            "label": "INCONCLUSIVE",
            "primary": {"candidate": "cand_ep30"},
        },
        "edges": [
            _sealbot_edge(0.48),
            {
                "opponent": "bc_prefit",
                "role": "anchor",
                "kind": "anchor",
                "primary": False,
                "decided": True,
                "winrate": 0.6,
                "winrate_ci95": [0.45, 0.75],
                "elo_point": 60.0,
                "provenance": {
                    "physical_wins_a": 5,
                    "physical_wins_b": 2,
                    "eval_visits": 400,
                    "n_pairs": 7,
                    "opponent_search_profile": "gumbel",
                },
            },
        ],
        "roster": {
            "candidate": {"label": "cand_ep30", "epoch": 30},
            "champion": {},
            "sealbot": {"label": "sealbot"},
            "opponents": [
                {"label": "bc_prefit", "role": "anchor", "epoch": 0},
                {"label": "sealbot", "role": "anchor", "epoch": None},
            ],
        },
        "stages": [{"stage": "C_deep", "status": "ok", "opponents_played": 2}],
        "sealbot_winrate_ci95": [0.38, 0.58],
    }


def _report_ep35() -> dict:
    return {
        "meta": {
            "candidate_epoch": 35,
            "candidate_label": "cand_ep35",
            "anchor": "sealbot",
            "config": _permanent_anchors_cfg(),
            "pure_eval": True,
            "elapsed_seconds": 140.0,
        },
        "ratings": {
            "anchor": "sealbot",
            # Split-node artifact: cand_ep30 and bare ep30 are the SAME file read
            # twice (candidate node vs prior-champion node) so the unify view can
            # show the delta rather than counting them as two bots.
            "players": [
                {"label": "cand_ep35", "elo": CAND_EP35_ELO, "role": "candidate"},
                {"label": "cand_ep30", "elo": 90.0, "role": "checkpoint"},
                {"label": "ep30", "elo": 88.0, "role": "champion"},
                {"label": "sealbot", "elo": 0.0, "role": "anchor", "is_anchor": True},
            ],
            "fit": {"method": "bradley_terry"},
        },
        "verdict": {
            "label": "PROMOTE",
            "primary": {"candidate": "cand_ep35"},
        },
        "edges": [
            # Primary edge: candidate vs prior champion (cand_ep35 vs ep30), 5-5
            # physical record — the value the pool W-L matrix must reproduce.
            {
                "opponent": "ep30",
                "role": "champion",
                "kind": "checkpoint",
                "primary": True,
                "decided": False,
                "winrate": 0.5,
                "winrate_ci95": [0.3, 0.7],
                "elo_point": 55.0,
                "provenance": {
                    "physical_wins_a": 5,
                    "physical_wins_b": 5,
                    "eval_visits": 400,
                    "n_pairs": 10,
                    "opponent_search_profile": "selfplay",
                },
            },
            _sealbot_edge(0.58),
        ],
        "roster": {
            "candidate": {"label": "cand_ep35", "epoch": 35},
            "champion": {"label": "ep30", "epoch": 30},
            "sealbot": {"label": "sealbot"},
            # bc_prefit is a configured permanent anchor but is ABSENT here — it
            # was DROPPED at ep35 (SEV-2 dropped-anchor detection).
            "opponents": [
                {"label": "ep30", "role": "champion", "epoch": 30},
                {"label": "sealbot", "role": "anchor", "epoch": None},
            ],
        },
        "stages": [{"stage": "C_deep", "status": "ok", "opponents_played": 2}],
        "sealbot_winrate_ci95": [0.48, 0.68],
    }


def _eval_pool() -> dict:
    # Controlled SealBot-pinned pool: EXPECTED_POOL_EDGES edges anchored on
    # "sealbot". Includes the cand_ep35-vs-ep30 pairing whose raw physical record
    # is 5-5 (top-level wins_a/wins_b are n_eff-weighted / fractional; the raw
    # block carries the true head-to-head).
    return {
        "format": "eval_pool",
        "version": 2,
        "anchor": "sealbot",
        "edges": [
            {
                "epoch": 35,
                "a": "cand_ep35",
                "b": "ep30",
                "wins_a": 4.7,
                "wins_b": 4.7,
                "weight": 1.0,
                "kind": "checkpoint",
                "raw": {"physical_wins_a": 5, "physical_wins_b": 5},
            },
            {
                "epoch": 35,
                "a": "cand_ep35",
                "b": "sealbot",
                "wins_a": 5.8,
                "wins_b": 4.2,
                "weight": 1.0,
                "kind": "anchor",
                "raw": {"physical_wins_a": 6, "physical_wins_b": 4},
            },
            {
                "epoch": 30,
                "a": "cand_ep30",
                "b": "sealbot",
                "wins_a": 4.8,
                "wins_b": 5.2,
                "weight": 1.0,
                "kind": "anchor",
                "raw": {"physical_wins_a": 5, "physical_wins_b": 5},
            },
            {
                "epoch": 5,
                "a": "cand_ep5",
                "b": "sealbot",
                "wins_a": 4.2,
                "wins_b": 5.8,
                "weight": 1.0,
                "kind": "anchor",
                "raw": {"physical_wins_a": 4, "physical_wins_b": 6},
            },
        ],
    }


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    """Build a synthetic hexfield run dir under ``tmp_path`` with a hexfield
    manifest (so the lineage-gated readers engage), three multi-stage eval
    reports (epochs 5, 30, 35), and a controlled ``eval_pool.json``."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"model": {"name": "hexfield"}}), encoding="utf-8"
    )
    diag = tmp_path / "diagnostics"
    diag.mkdir()
    for epoch, report in (
        (5, _report_ep5()),
        (30, _report_ep30()),
        (35, _report_ep35()),
    ):
        name = f"hexfield.multistage_eval.epoch_{epoch:06d}.json"
        (diag / name).write_text(json.dumps(report), encoding="utf-8")
    (diag / "eval_pool.json").write_text(json.dumps(_eval_pool()), encoding="utf-8")
    return tmp_path


def _load(run_dir: Path):
    eval_hist = web._evaluation_history(run_dir)
    ms_hist = web._multistage_eval_history(run_dir)
    pool = web._eval_pool_summary(run_dir)
    live = web._training_live_status(run_dir)
    epoch_hist = web._epoch_history(run_dir)
    return eval_hist, ms_hist, pool, live, epoch_hist


def test_d1_health_uses_real_multistage(run_dir: Path):
    eval_hist, ms_hist, pool, live, epoch_hist = _load(run_dir)
    assert len(ms_hist) >= EXPECTED_MS_REPORTS, (
        f"expected >={EXPECTED_MS_REPORTS} multistage reports, got {len(ms_hist)}"
    )

    health = web._learning_health(epoch_hist, eval_hist, live, ms_hist)
    msgs = health.get("messages") or []

    # (a) NO false "no eval" sentinel.
    assert not any("No SealBot evaluation result yet" in m for m in msgs), (
        "D1: false 'no eval' message still emitted: " + repr(msgs)
    )
    # (b) NO false hexfield "D6 missing" noise.
    assert not any("D6 augmentation preview is missing" in m for m in msgs), (
        "D1: false 'D6 missing' message still emitted: " + repr(msgs)
    )
    # (c) the new health fields are populated and match the newest report.
    latest_ms = ms_hist[-1]
    want_verdict = str(
        latest_ms.get("verdict_label") or latest_ms["verdict"]["label"]
    ).upper()
    assert health.get("latest_verdict") == want_verdict, (
        f"D1: latest_verdict {health.get('latest_verdict')!r} != {want_verdict!r}"
    )
    assert health.get("latest_cand_elo") is not None, "D1: candidate Elo not populated"
    # candidate elo must match the named candidate node in the latest report.
    cand_label = latest_ms["verdict"]["primary"]["candidate"]
    cand_node = next(
        p for p in latest_ms["ratings"]["players"] if p.get("label") == cand_label
    )
    assert abs(health["latest_cand_elo"] - cand_node["elo"]) < 1e-6, (
        f"D1: cand elo {health['latest_cand_elo']} != node {cand_node['elo']}"
    )
    assert health.get("latest_sealbot_winrate") is not None, "D1: sealbot winrate missing"
    assert health.get("latest_eval_epoch") == LATEST_EPOCH, (
        f"D1: latest_eval_epoch {health.get('latest_eval_epoch')} != {LATEST_EPOCH}"
    )
    # (d) status is on the known ladder and NOT the dead-path 'collecting'.
    assert health["status"] in {"ok", "improving", "watch", "intervene"}, (
        f"D1: unexpected status {health['status']!r}"
    )
    assert health["status"] != "collecting", "D1: still forced to 'collecting'"


def test_d1_dense_cnn_path_unchanged():
    # Feed a synthetic dense_cnn-shaped evaluation_history (mean_turns present) and
    # EMPTY multistage history: the legacy turns-based messages must still drive.
    epoch_hist = [
        {"epoch": 1, "status": "completed", "training": {"loss": 2.0}},
        {"epoch": 6, "status": "completed", "training": {"loss": 1.0}},
    ]
    eval_hist = [
        {"epoch": 1, "mean_turns": 20.0, "wins": 1, "games": 8},
        {"epoch": 6, "mean_turns": 28.0, "wins": 3, "games": 8},
    ]
    health = web._learning_health(epoch_hist, eval_hist, {}, [])
    msgs = health.get("messages") or []
    assert any("SealBot eval has" in m or "SealBot survival" in m for m in msgs), (
        "legacy turns-based message missing on dense_cnn path: " + repr(msgs)
    )
    assert health.get("latest_eval_mean_turns") == 28.0
    assert health.get("latest_verdict") is None, "dense_cnn path must not set latest_verdict"


def test_d2_pool_ledger_builds(run_dir: Path):
    _, ms_hist, pool, _, _ = _load(run_dir)
    assert pool is not None, "D2: eval_pool summary should be present"
    edges = pool["edges"]
    assert pool["edges_total"] == len(edges) == EXPECTED_POOL_EDGES, (
        f"D2: expected {EXPECTED_POOL_EDGES} edges, got {len(edges)}"
    )
    assert pool["anchor"] == "sealbot"

    # every edge carries the per-opponent W-L contract the renderer needs.
    for e in edges:
        assert {"a", "b", "wins_a", "wins_b", "epoch", "kind"} <= set(e), (
            "D2: pool edge missing a required key: " + repr(e)
        )

    # Aggregating wins_a/wins_b by (a,b) must reproduce the per-opponent record in
    # the newest report's edges. The pool stores n_eff-weighted wins for checkpoint
    # pairs; the raw physical record (5-5 for cand_ep35 vs ep30) lives in the
    # report edge's provenance, so cross-check the *physical* count there — against
    # the SYNTHESIZED newest report path, not any private live latest_path.
    latest_path = (
        run_dir / "diagnostics" / f"hexfield.multistage_eval.epoch_{LATEST_EPOCH:06d}.json"
    )
    rep = json.loads(latest_path.read_text(encoding="utf-8"))
    champ_edge = next(e for e in rep["edges"] if e.get("primary"))
    prov = champ_edge["provenance"]
    assert prov["physical_wins_a"] == 5 and prov["physical_wins_b"] == 5, (
        "D2: expected cand_ep35 vs ep30 physical record 5-5, got "
        f"{prov['physical_wins_a']}-{prov['physical_wins_b']}"
    )

    # the per-opponent matrix the renderer builds: group pool edges by (a,b).
    matrix = defaultdict(lambda: [0.0, 0.0, 0])
    for e in edges:
        key = (e["a"], e["b"])
        matrix[key][0] += float(e.get("wins_a") or 0)
        matrix[key][1] += float(e.get("wins_b") or 0)
        matrix[key][2] += 1
    assert ("cand_ep35", "ep30") in matrix, "D2: cand_ep35 vs ep30 pairing missing from pool"


def test_d2_d3_unify_has_both_readings(run_dir: Path):
    # The split-node artifact: the newest report's players include BOTH cand_ep30
    # and bare ep30 (same file, two Elo readings) so the unify step can show the
    # delta rather than reading them as two bots.
    _, ms_hist, _, _, _ = _load(run_dir)
    latest = ms_hist[-1]
    labels = {p.get("label") for p in latest["ratings"]["players"]}
    assert "cand_ep30" in labels and "ep30" in labels, (
        "D3: expected both cand_ep30 and ep30 split nodes in the ladder"
    )


def test_d3_roster_driven_headline_and_dropped_anchor(run_dir: Path):
    _, ms_hist, _, _, _ = _load(run_dir)
    latest = ms_hist[-1]
    roster = latest.get("roster") or {}
    assert roster, "D3: roster not shipped on multistage rows"
    perms = roster.get("permanent_anchors") or []
    assert "bc_prefit" in perms and "ep5" in perms, (
        f"D3: permanent_anchors not surfaced from config: {perms}"
    )
    present = {o.get("label") for o in (roster.get("opponents") or [])}
    # bc_prefit is a configured permanent anchor but DROPPED at ep35 (SEV-2) — the
    # dropped-anchor builder must be able to detect it (in perms, NOT in present).
    dropped = [a for a in perms if a not in present]
    assert "bc_prefit" in dropped, (
        f"D3: bc_prefit should be detected as a dropped anchor at ep35; present={present}"
    )
    # ...and at an EARLIER epoch (ep30) bc_prefit IS in the roster -> not dropped.
    ep30 = next((r for r in ms_hist if r.get("epoch") == 30), None)
    assert ep30 is not None
    present30 = {o.get("label") for o in (ep30.get("roster", {}).get("opponents") or [])}
    assert "bc_prefit" in present30, (
        f"D3: bc_prefit should be in ep30 roster; present={present30}"
    )


def test_d3_verdict_history_strip_data(run_dir: Path):
    _, ms_hist, _, _, _ = _load(run_dir)
    strip = [
        (r.get("epoch"), str((r.get("verdict_label") or (r.get("verdict") or {}).get("label") or "")).upper())
        for r in ms_hist
    ]
    epochs = [e for e, _ in strip]
    assert epochs == sorted(epochs), "D3: verdict strip not ascending by epoch"
    labels = [v for _, v in strip]
    assert labels[0] == "REGRESS", f"D3: ep5 should be REGRESS, got {labels[0]}"
    assert all(v for v in labels), "D3: every report must carry a verdict label"
