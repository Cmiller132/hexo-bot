"""The mantisnet family end to end: catalogue, worker runtime, telemetry
parity, device self-check, and the HTTP surface with a per-checkpoint sims
ladder.

Follows test_family_dispatch's pattern: the catalogue path stays torch-free,
_WorkerRuntime is imported only after load_bots_toml, and the runtime is
driven in-process with device_override="cpu".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from showcase.bots import load_bots_toml
from showcase.config import Settings

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _catalogue(path: Path, checkpoint: Path, *, sims_line: str = "") -> Path:
    toml = path / "bots.toml"
    toml.write_text(
        f"""sims = [8, 16]

[[checkpoint]]
id = "tiny-mantis"
family = "mantisnet"
checkpoint = "{checkpoint.as_posix()}"
label = "Tiny MantisNet"
run = "showcase_tiny_mantis"
epoch = 0
{sims_line}
""",
        encoding="utf-8",
    )
    return toml


def _settings(bots_toml: Path, tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "showcase.db",
        bots_toml=bots_toml,
        search_config=_REPO_ROOT / "configs" / "shrimp_main_7.toml",
        static_dir=_REPO_ROOT / "apps" / "showcase" / "web",
        workers=1,
        max_active_games=16,
        max_games_per_ip=8,
        moves_per_minute=100_000,
        analysis_per_minute=100_000,
        games_per_hour=100_000,
        idle_timeout_s=3600.0,
        bot_timeout_s=60.0,
        move_timeout_s=30.0,
        max_recycles_per_window=3,
        recycle_window_s=300.0,
        finished_ttl_s=3600.0,
        sweep_interval_s=3600.0,
        analysis_search_visit_cap=8,
        policy_floor=1e-4,
        torch_threads=2,
        ip_salt="test-salt",
    )


def test_catalogue_parses_mantisnet_and_per_checkpoint_sims(
    tiny_mantis_checkpoint, tmp_path
):
    toml = _catalogue(tmp_path, tiny_mantis_checkpoint, sims_line="sims = [4, 8]")
    catalogue = load_bots_toml(toml)
    spec = catalogue.checkpoints[0]
    assert spec.family == "mantisnet"
    assert spec.sims == (4, 8)
    assert "sims" not in spec.meta
    assert catalogue.sims == (8, 16)


def test_catalogue_rejects_bad_per_checkpoint_sims(tiny_mantis_checkpoint, tmp_path):
    toml = _catalogue(tmp_path, tiny_mantis_checkpoint, sims_line="sims = []")
    with pytest.raises(ValueError, match="non-empty array"):
        load_bots_toml(toml)


def test_mantisnet_worker_serves_every_seam(tiny_mantis_checkpoint, tmp_path):
    toml = _catalogue(tmp_path, tiny_mantis_checkpoint)
    settings = _settings(toml, tmp_path)
    catalogue = load_bots_toml(toml)
    # Import order matters: the catalogue path itself must stay torch-free.
    from showcase.bots import _WorkerRuntime

    runtime = _WorkerRuntime(list(catalogue.checkpoints), settings, device_override="cpu")

    import hexo_engine as engine
    from hexo_engine import api
    from hexo_engine.types import (
        AxialCoord, PlacementAction, pack_coord_id, unpack_coord_id,
    )

    # The origin opening, as GameSession.create places it.
    state = engine.new_game()
    origin = next(
        aid for aid in api.legal_action_ids(state)
        if unpack_coord_id(int(aid)).q == 0 and unpack_coord_id(int(aid)).r == 0
    )
    engine.apply_action(state, PlacementAction(unpack_coord_id(int(origin))))
    actions = [int(origin)]

    # -- bot_turn plays a full two-stone turn of engine-legal moves ---------
    out = runtime.bot_turn(
        bot_slug="tiny-mantis", game_key=5, actions=list(actions), seed=3, visits=8,
        tss_enabled=True,
    )
    assert len(out["actions"]) == 2
    for move in out["actions"]:
        aid = int(move["action_id"])
        legal = set(int(x) for x in api.legal_action_ids(state))
        assert aid in legal
        engine.apply_action(state, PlacementAction(unpack_coord_id(aid)))
        actions.append(aid)
        assert -1.0 <= move["root_value"] <= 1.0
        assert move["visits"] == 8

    # -- live telemetry: native sequence, byte-sane frames, and parity ------
    frames: list[dict] = []
    with_telemetry = runtime.bot_turn(
        bot_slug="tiny-mantis", game_key=5, actions=list(actions), seed=4, visits=8,
        tss_enabled=True, progress_callback=frames.append,
    )
    phases = [f.get("phase") for f in frames if f.get("kind") == "search_telemetry"]
    assert phases[0] == "start" and "complete" in phases
    # The driver ticks once per completed wave, so even a small budget yields
    # round frames between the start and the answer.
    assert phases.count("round") >= 2
    # Every round frame carries the per-line running Q alongside the visit
    # counts: the viewer's value ticks update as the search learns, not only
    # on the answer frame.
    for frame in frames:
        if frame.get("phase") != "round":
            continue
        assert len(frame["policy_q_bytes"]) == frame["policy_count"] * 4
        assert len(frame["policy_visits_bytes"]) == frame["policy_count"] * 4
    assert not any(f.get("_post_search") for f in frames), (
        "a native-telemetry family must never take the post-search fallback"
    )
    start = next(f for f in frames if f.get("phase") == "start")
    assert len(start["policy_action_ids_bytes"]) == start["policy_count"] * 4
    assert len(start["policy_weights_bytes"]) == start["policy_count"] * 4
    assert start["visits"] == 0
    # The answer frame carries the final SH ranking (scores), per-line visit
    # and value readouts, and the final survivor set for the last-cut dim.
    complete = next(f for f in frames if f.get("phase") == "complete")
    assert complete["policy_kind"] == "score"
    assert len(complete["policy_weights_bytes"]) == complete["policy_count"] * 4
    assert len(complete["policy_visits_bytes"]) == complete["policy_count"] * 4
    assert len(complete["policy_q_bytes"]) == complete["policy_count"] * 4
    assert 1 <= complete["survivor_count"] <= complete["policy_count"]
    assert len(complete["survivor_action_ids_bytes"]) == complete["survivor_count"] * 4
    without_telemetry = runtime.bot_turn(
        bot_slug="tiny-mantis", game_key=5, actions=list(actions), seed=4, visits=8,
        tss_enabled=True,
    )
    assert [m["action_id"] for m in with_telemetry["actions"]] == [
        m["action_id"] for m in without_telemetry["actions"]
    ]
    assert [m["root_value"] for m in with_telemetry["actions"]] == [
        m["root_value"] for m in without_telemetry["actions"]
    ]

    # -- analyze: value present, absent heads are null, search attaches -----
    analysis = runtime.analyze(
        bot_slug="tiny-mantis", actions=list(actions), want_search=True,
        search_visits=8, seed=9,
    )
    assert -1.0 <= analysis["value"] <= 1.0
    assert analysis["stv"] is None and analysis["moves_left"] is None
    assert analysis["legal_count"] > 0 and analysis["policy"]
    assert analysis["search"]["visits"] > 0
    assert set(analysis["search"]["best"]) == {"q", "r"}

    # -- summary: one entry per position, index i = after ply i -------------
    summary = runtime.summary(bot_slug="tiny-mantis", actions=list(actions))
    assert len(summary["value"]) == len(actions) + 1
    assert all(v is None for v in summary["stv"])
    assert all(v is None for v in summary["moves_left"])

    # -- a finished game: the terminal row is null, never an error ----------
    # P1 completes six in a row along r=3 on ply 11; MantisNet's builder
    # refuses terminal positions, so the final row (and a terminal analyze)
    # must come back as the frontend's "no data" rather than a 5xx.
    win = [
        (0, 0), (0, 3), (1, 3), (1, 0), (2, 0), (2, 3),
        (3, 3), (3, 0), (4, 0), (4, 3), (5, 3),
    ]
    win_ids = [int(pack_coord_id(AxialCoord(q, r))) for q, r in win]
    finished = runtime.summary(bot_slug="tiny-mantis", actions=list(win_ids))
    assert len(finished["value"]) == len(win_ids) + 1
    assert finished["value"][-1] is None
    assert all(v is not None for v in finished["value"][:-1])
    terminal_analysis = runtime.analyze(
        bot_slug="tiny-mantis", actions=list(win_ids), want_search=True,
        search_visits=8, seed=11,
    )
    assert terminal_analysis["value"] is None
    assert terminal_analysis["legal_count"] == 0
    assert terminal_analysis["policy"] == []
    assert "search" not in terminal_analysis
    assert terminal_analysis["to_move"] is None

    # -- lab eval: sequence mode works, free-edit is refused honestly -------
    coords = [(unpack_coord_id(a).q, unpack_coord_id(a).r) for a in actions]
    lab = runtime.lab_eval(
        bot_slug="tiny-mantis", actions=coords, stones=None, to_move=None,
        attention_cell=coords[0], want_activations=True, want_features=True,
    )
    assert lab["mode"] == "sequence" and lab["ply"] == len(actions)
    # The one value served is v̂; the untrained state head's scalar and bin
    # distribution are deliberately absent from the payload.
    assert -1.0 <= lab["value"] <= 1.0
    assert "value_dist" not in lab and "v_hat" not in lab
    # Attention rows: one per (block, head), weights over support nodes plus
    # one global-token summary weight; a stone query resolves.
    attn = lab["attention"]
    assert attn["available"] is True
    assert len(attn["rows"]) == 1 and len(attn["rows"][0]) == 2
    row = attn["rows"][0][0]
    assert len(row["tokens"]) == 1 and 0.0 <= row["tokens"][0] <= 1.0
    assert all(w >= attn["floor"] for w in row["cells"].values())
    # A non-stone query is refused with the family's reason, not a crash.
    off_stone = runtime.lab_eval(
        bot_slug="tiny-mantis", actions=coords, stones=None, to_move=None,
        attention_cell=(8, 8), want_activations=False, want_features=False,
    )
    assert off_stone["attention"]["available"] is False
    assert "stone" in off_stone["attention"]["reason"]
    # Activation stages: stem + one block + output, norms over every support
    # node, and the feature planes are name-driven.
    stages = lab["activations"]["blocks"]
    assert [s["label"] for s in stages] == ["stem", "block 1", "output (shared LN)"]
    n_support = len(lab["support"]["coords"])
    assert all(len(s["norms"]) == n_support for s in stages)
    assert lab["features"]["names"][0] == "is_stone"
    assert len(lab["features"]["planes"]) == len(lab["features"]["names"])
    assert all(len(p) == n_support for p in lab["features"]["planes"])
    rejected = runtime.lab_eval(
        bot_slug="tiny-mantis", actions=None, stones=([(0, 0)], [(1, 1)]),
        to_move=0, attention_cell=None, want_activations=False, want_features=False,
    )
    assert "reject" in rejected and "placement sequence" in rejected["reject"]

    # -- lab search ----------------------------------------------------------
    searched = runtime.lab_search(
        bot_slug="tiny-mantis", actions=coords, visits=8, seed=2,
    )
    assert searched["visits"] > 0 and searched["visit_policy"]
    assert "w" in searched["visit_policy"][0]

    # -- the web-side decoder accepts the family's frames --------------------
    from showcase.live_search import expand_worker_event

    public: list[dict] = []
    for frame in frames:
        if frame.get("kind") == "search_telemetry":
            public.extend(expand_worker_event(frame))
    kinds = [event["kind"] for event in public]
    assert "bare_policy" in kinds and "candidate_set" in kinds
    assert "search_round" in kinds and "search_complete" in kinds
    bare = next(e for e in public if e["kind"] == "bare_policy")
    # The decoder drops a policy whose buffers disagree; surviving rows carry
    # real cells with normalized-ish weights.
    assert bare["policy"] and all(
        set(row) >= {"q", "r", "p"} for row in bare["policy"]
    )
    complete = next(e for e in public if e["kind"] == "search_complete")
    assert complete["policy"] and complete["policy_kind"] == "score"
    assert complete["survivors"], "the final cut must reach the viewer"
    assert any("visits" in row for row in complete["policy"])
    assert any("value" in row for row in complete["policy"])


def test_verify_device_family_branch_on_cpu(tiny_mantis_checkpoint):
    from types import SimpleNamespace

    from showcase.device import verify_device
    from showcase.families import get_family

    family = get_family("mantisnet")
    model = family.load_net(SimpleNamespace(checkpoint=tiny_mantis_checkpoint))
    result = verify_device(model, "cpu", family=family)
    assert result.ok, result
    assert result.value_diff == 0.0 and result.policy_diff == 0.0


def test_http_surface_with_per_checkpoint_ladder(tiny_mantis_checkpoint, tmp_path):
    from fastapi.testclient import TestClient
    from test_showcase_api import create_game, fresh_ip, poll_until, resign

    from showcase.app import create_app

    toml = _catalogue(tmp_path, tiny_mantis_checkpoint, sims_line="sims = [4, 8]")
    settings = _settings(toml, tmp_path)
    with TestClient(create_app(settings)) as client:
        bots = client.get("/api/bots").json()
        entry = bots["checkpoints"][0]
        assert entry["family"] == "mantisnet"
        assert entry["sims"] == [4, 8]
        assert bots["sims"] == [8, 16]

        headers = fresh_ip()
        # The global ladder's 16 is outside this checkpoint's own ladder.
        refused = client.post(
            "/api/game",
            json={"checkpoint_id": "tiny-mantis", "sims": 16, "human_color": 0},
            headers=headers,
        )
        assert refused.status_code == 422
        assert "[4, 8]" in refused.json()["detail"]

        game = create_game(client, headers, checkpoint_id="tiny-mantis", sims=8)
        # Play a few plies against the bot over HTTP.
        for _ in range(3):
            snap = poll_until(client, game["id"])
            if snap["status"] == "finished":
                break
            cell = snap["legal"][len(snap["stones"]) % len(snap["legal"])]
            moved = client.post(
                f"/api/game/{game['id']}/move",
                json={"q": cell["q"], "r": cell["r"]},
                headers=headers,
            )
            assert moved.status_code == 200, moved.text
        else:
            snap = poll_until(client, game["id"])
        if snap["status"] != "finished":
            resign(client, game["id"], headers)


def test_mantisnet_serves_the_critic_read(tiny_mantis_checkpoint, tmp_path):
    """The KLENT critic rides every analysis surface: the per-cell klent
    block on analyze/lab_eval, played/best Q on the summary, replayable
    frames on a lab search that asks for them, and the solver seam."""
    toml = _catalogue(tmp_path, tiny_mantis_checkpoint)
    settings = _settings(toml, tmp_path)
    catalogue = load_bots_toml(toml)
    from showcase.bots import _WorkerRuntime

    runtime = _WorkerRuntime(
        list(catalogue.checkpoints), settings, device_override="cpu"
    )

    from hexo_engine.types import AxialCoord, pack_coord_id

    moves = [(0, 0), (0, 3), (1, 3), (1, 0), (2, 0)]
    ids = [int(pack_coord_id(AxialCoord(q, r))) for q, r in moves]

    # -- analyze: the klent block covers every legal cell -------------------
    analysis = runtime.analyze(
        bot_slug="tiny-mantis", actions=list(ids), want_search=False,
        search_visits=8, seed=1,
    )
    k = analysis["klent"]
    n = analysis["legal_count"]
    assert len(k["coords"]) == n
    for name in ("prior", "improved", "q", "win", "loss", "long"):
        assert len(k[name]) == n
    # One categorical simplex per cell composes Q (up to 4-decimal rounding).
    for i in range(n):
        assert k["win"][i] + k["loss"][i] + k["long"][i] == pytest.approx(
            1.0, abs=2e-3
        )
        assert k["q"][i] == pytest.approx(k["win"][i] - k["loss"][i], abs=2e-3)
    assert sum(k["prior"]) == pytest.approx(1.0, abs=1e-3 + n * 1e-4)
    assert sum(k["improved"]) == pytest.approx(1.0, abs=1e-3 + n * 1e-4)
    assert k["kl"] >= 0.0
    assert 0.0 <= k["norm_entropy"] <= 1.0

    # -- summary: the critic series; best can never trail played ------------
    summary = runtime.summary(bot_slug="tiny-mantis", actions=list(ids))
    played, best = summary["played_q"], summary["best_q"]
    assert len(played) == len(ids) + 1 and len(best) == len(ids) + 1
    assert played[-1] is None, "no move follows the final row"
    assert best[-1] is not None, "the final row is a live position"
    for i in range(len(ids)):
        assert played[i] is not None and best[i] is not None
        # 4-decimal payload rounding can nudge an equal pair by one ulp
        assert best[i] >= played[i] - 2e-4

    # A finished game's terminal row stays null in both series.
    win = [
        (0, 0), (0, 3), (1, 3), (1, 0), (2, 0), (2, 3),
        (3, 3), (3, 0), (4, 0), (4, 3), (5, 3),
    ]
    win_ids = [int(pack_coord_id(AxialCoord(q, r))) for q, r in win]
    finished = runtime.summary(bot_slug="tiny-mantis", actions=list(win_ids))
    assert finished["played_q"][-1] is None and finished["best_q"][-1] is None
    assert finished["played_q"][-2] is not None, (
        "the position before the winning move reads the winning move's Q"
    )

    # -- lab_eval: same block; the old improved_policy field is gone --------
    lab = runtime.lab_eval(
        bot_slug="tiny-mantis", actions=list(moves), stones=None, to_move=None,
        attention_cell=None, want_activations=False, want_features=False,
    )
    assert "klent" in lab
    assert "improved_policy" not in lab

    # -- lab_search: raw frames only when asked, expandable web-side --------
    plain = runtime.lab_search(
        bot_slug="tiny-mantis", actions=list(moves), visits=8, seed=2,
    )
    assert "frames_raw" not in plain
    framed = runtime.lab_search(
        bot_slug="tiny-mantis", actions=list(moves), visits=8, seed=2,
        want_frames=True,
    )
    raw = framed["frames_raw"]
    assert raw and raw[0]["phase"] == "start" and raw[-1]["phase"] == "complete"
    from showcase.live_search import expand_worker_event

    events = []
    for frame in raw:
        events.extend(expand_worker_event({"kind": "search_telemetry", **frame}))
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "bare_policy"
    assert "search_round" in kinds
    assert kinds[-1] == "search_complete"
    assert all("policy" in e for e in events if e["kind"] == "bare_policy")

    # -- solve: the worker seam answers, and rejects a terminal position ----
    solved = runtime.solve(bot_slug="tiny-mantis", actions=list(moves))
    assert solved["status"] in ("win", "loss", "unknown", "timeout")
    rejected = runtime.solve(bot_slug="tiny-mantis", actions=list(win))
    assert "reject" in rejected
