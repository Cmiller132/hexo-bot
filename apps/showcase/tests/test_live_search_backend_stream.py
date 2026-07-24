"""CPU/fake coverage for the live-search worker side channel and replay hub."""

from __future__ import annotations

import asyncio
import itertools
import struct
from dataclasses import replace
from types import SimpleNamespace

import hexo_engine as engine
import pytest
from hexo_engine import api
from hexo_engine.types import AxialCoord, pack_coord_id, unpack_coord_id

from showcase import bots
from showcase.app import create_app
from showcase.bots import BotPool, _WorkerRuntime
from showcase.families.hexfield_eq_family import _wire_telemetry_event
from showcase.live_search import (
    LiveSearchHub,
    LiveSearchSubscriberLimit,
    encode_sse_event,
    expand_worker_event,
    subscription_closed,
)


class _FakeProfile:
    def __init__(self) -> None:
        self.callback_flags: list[bool] = []

    def move_temperature(self, ply: int) -> float:
        return 0.0

    def search_one(self, session, evaluator, state, **kwargs):
        callback = kwargs.pop("telemetry_callback", None)
        self.callback_flags.append(callback is not None)
        legal = [int(action_id) for action_id in api.legal_action_ids(state)]
        action_id = legal[0]
        candidate_ids = legal[:2]

        ids_wire = struct.pack(f"={len(candidate_ids)}I", *candidate_ids)
        survivors_wire = struct.pack("=I", action_id)
        if callback is not None:
            callback(
                {
                    "phase": "start",
                    "round": 0,
                    "rounds": 1,
                    "visits": 0,
                    "target_visits": kwargs["visits"],
                    "policy_action_ids_bytes": ids_wire,
                    "policy_weights_bytes": struct.pack("=2f", 0.75, 0.25),
                    "survivor_action_ids_bytes": survivors_wire,
                }
            )
            callback(
                {
                    "phase": "round",
                    "round": 1,
                    "rounds": 1,
                    "visits": kwargs["visits"],
                    "target_visits": kwargs["visits"],
                    "policy_action_ids_bytes": struct.pack("=I", action_id),
                    "policy_weights_bytes": struct.pack("=f", 1.0),
                    "survivor_action_ids_bytes": survivors_wire,
                }
            )
            callback(
                {
                    "phase": "complete",
                    "visits": kwargs["visits"],
                    "target_visits": kwargs["visits"],
                    "root_value": 0.25,
                    "policy_action_ids_bytes": struct.pack("=I", action_id),
                    "policy_weights_bytes": struct.pack("=f", 1.0),
                    "action_id": action_id,
                }
            )
        weights_wire = struct.pack(
            f"={len(candidate_ids)}f", *([1.0] + [0.0] * (len(candidate_ids) - 1))
        )
        priors_wire = struct.pack(
            f"={len(candidate_ids)}f", *([0.75] + [0.25] * (len(candidate_ids) - 1))
        )
        return {
            "action_id": action_id,
            "root_value": 0.25,
            "visits": int(kwargs["visits"]),
            "visit_policy_action_ids_bytes": ids_wire,
            "visit_policy_weights_bytes": weights_wire,
            "root_prior_policy_action_ids_bytes": ids_wire,
            "root_prior_policy_weights_bytes": priors_wire,
        }


class _FakeFamily:
    @staticmethod
    def decode_action(action_id: int) -> tuple[int, int]:
        coord = unpack_coord_id(action_id)
        return coord.q, coord.r

    @staticmethod
    def net_eval(*args, **kwargs):
        raise AssertionError("live telemetry must not add a model forward")


def _fake_runtime(
    *, family_name: str = "hexfield_eq",
) -> tuple[_WorkerRuntime, _FakeProfile]:
    profile = _FakeProfile()
    runtime = _WorkerRuntime.__new__(_WorkerRuntime)
    runtime.bots = {
        "fake": SimpleNamespace(
            spec=SimpleNamespace(family=family_name),
            family=_FakeFamily(),
            model=object(),
            evaluator=object(),
            session=object(),
            profile=profile,
        )
    }
    return runtime, profile


def test_runtime_watch_maps_two_stones_and_preserves_return_bytes():
    actions = [pack_coord_id(AxialCoord(q=0, r=0))]
    runtime, profile = _fake_runtime()
    plain = runtime.bot_turn(
        bot_slug="fake", game_key=7, actions=actions, seed=11, visits=8
    )
    assert profile.callback_flags == [False, False]

    events: list[dict] = []

    def loop_side_callback(event: dict) -> None:
        events.extend(expand_worker_event(event))

    watched = runtime.bot_turn(
        bot_slug="fake", game_key=7, actions=actions, seed=11, visits=8,
        progress_callback=loop_side_callback,
    )
    assert watched == plain
    assert profile.callback_flags[-2:] == [True, True]
    assert [event["kind"] for event in events] == [
        "bare_policy", "candidate_set", "search_round", "search_complete", "stone",
        "bare_policy", "candidate_set", "search_round", "search_complete", "stone",
    ]
    assert [event["stone"] for event in events] == [1] * 5 + [2] * 5
    assert [event["ply"] for event in events] == [1] * 5 + [2] * 5
    assert len(events[0]["policy"]) == 2
    assert len(events[1]["policy"]) == 1
    assert events[4]["action"] == events[3]["action"]


def test_non_hex_watch_uses_post_search_result_fallback_only():
    runtime, profile = _fake_runtime(family_name="shrimp")
    events: list[dict] = []

    def loop_side_callback(event: dict) -> None:
        events.extend(expand_worker_event(event))

    watched = runtime.bot_turn(
        bot_slug="fake", game_key=7,
        actions=[pack_coord_id(AxialCoord(q=0, r=0))],
        seed=11, visits=8, progress_callback=loop_side_callback,
    )
    assert watched["actions"]
    assert profile.callback_flags == [False, False]
    assert [event["kind"] for event in events] == [
        "turn_start",
        "bare_policy", "candidate_set", "search_complete", "stone",
        "bare_policy", "candidate_set", "search_complete", "stone",
    ]
    assert events[0]["post_search"] is True
    assert "visits" not in events[1]
    assert events[1]["post_search"] is True
    assert events[2]["target_visits"] == 8
    assert events[3]["post_search"] is True
    assert "weight" in events[3]["policy"][0]


def test_hexfield_raw_telemetry_stays_wire_until_loop_side_expansion():
    from hexfield_eq.geometry import pack_action_id

    ids = [pack_action_id(1, -1), pack_action_id(2, -1)]
    policy_ids_wire = struct.pack("=2I", *ids)
    wire = _wire_telemetry_event(
        {
            "phase": "start",
            "round": 0,
            "rounds": 2,
            "visits": 4,
            "target_visits": 16,
            "root_value": 0.125,
            "policy_action_ids_bytes": policy_ids_wire,
            "policy_weights_bytes": struct.pack("=2f", 3.0, 1.0),
            "policy_visits_bytes": struct.pack("=2I", 3, 1),
            "survivor_action_ids_bytes": struct.pack("=I", ids[0]),
            "action_id": ids[0],
        }
    )
    assert wire["phase"] == "start"
    assert isinstance(wire["policy_action_ids_bytes"], bytes)
    assert wire["policy_action_ids_bytes"] is policy_ids_wire
    assert "policy" not in wire

    bare, candidates = expand_worker_event(
        {
            "kind": "search_telemetry",
            "stone": 1,
            "ply": 4,
            **wire,
        }
    )
    assert [row["p"] for row in bare["policy"]] == [0.75, 0.25]
    assert [row["visits"] for row in bare["policy"]] == [3, 1]
    assert "survivors" not in bare
    assert candidates["survivors"] == [{"q": 1, "r": -1}]
    assert candidates["policy"] == [bare["policy"][0]]
    assert candidates["action"] == {"q": 1, "r": -1}


def test_replay_hub_bounds_replay_and_slow_subscriber_queue():
    hub = LiveSearchHub(replay_size=3, subscriber_queue_size=2)
    for value in range(1, 6):
        hub.publish("g", {"kind": "round", "value": value})
    subscription = hub.subscribe("g", after_seq=3)
    assert [event["seq"] for event in subscription.replay] == [4, 5]

    for value in range(6, 9):
        hub.publish("g", {"kind": "round", "value": value})
    queued = [subscription.queue.get_nowait(), subscription.queue.get_nowait()]
    assert [event["seq"] for event in queued] == [7, 8]

    hub.begin_run("g")
    current = hub.publish("g", {"kind": "turn_start"})
    assert current["seq"] == 9
    replayed = hub.subscribe("g")
    assert [event["seq"] for event in replayed.replay] == [9]
    replayed.close()

    hub.drop("g")
    assert subscription.queue.get_nowait()["seq"] == 9
    assert subscription_closed(subscription.queue.get_nowait())
    subscription.close()


def test_default_subscriber_buffer_cap_and_sse_framing():
    hub = LiveSearchHub(max_subscribers=1)
    subscription = hub.subscribe("g")
    assert subscription.queue.maxsize >= 64
    with pytest.raises(LiveSearchSubscriberLimit):
        hub.subscribe("g")
    subscription.close()

    assert encode_sse_event(
        {"seq": 7, "kind": "turn_start", "run_id": 3}
    ) == (
        'id: 7\nevent: search\ndata: '
        '{"seq":7,"kind":"turn_start","run_id":3}\n\n'
    )


class _FakeQueue:
    def __init__(self, items=None) -> None:
        self.items = list(items or [])

    def put(self, item) -> None:
        self.items.append(item)

    def get(self):
        return self.items.pop(0)


class _FakeProc:
    def is_alive(self) -> bool:
        return True


def _pool_for_submit() -> BotPool:
    pool = BotPool.__new__(BotPool)
    pool._settings = SimpleNamespace(move_timeout_s=0.5)
    pool._loop = asyncio.get_running_loop()
    pool._job_queues = [_FakeQueue()]
    pool._procs = [_FakeProc()]
    pool._futures = {}
    pool._progress_callbacks = {}
    pool._started = set()
    pool._job_worker = {}
    pool._job_ids = itertools.count(1)
    pool._poisoned = [False]
    pool._recycle_locks = [asyncio.Lock()]
    return pool


def test_pool_progress_precedes_final_and_callback_failure_is_isolated():
    async def scenario():
        pool = _pool_for_submit()
        seen: list[dict] = []

        def callback(payload: dict) -> None:
            seen.append(payload)
            if payload["kind"] == "boom":
                raise RuntimeError("viewer callback failed")

        async def worker():
            while not pool._job_queues[0].items:
                await asyncio.sleep(0)
            job_id = pool._job_queues[0].items[-1][0]
            pool._mark_started(job_id)
            pool._deliver_progress(job_id, {"kind": "round"})
            pool._deliver_progress(job_id, {"kind": "boom"})
            pool._resolve(job_id, {"ok": {"actions": [1, 2]}})
            pool._deliver_progress(job_id, {"kind": "late"})
            return job_id

        worker_task = asyncio.create_task(worker())
        result = await pool._submit(
            0, "move", {}, 0.5, progress_callback=callback
        )
        job_id = await worker_task
        return result, seen, job_id in pool._progress_callbacks

    result, seen, registered = asyncio.run(scenario())
    assert result == {"actions": [1, 2]}
    assert [event["kind"] for event in seen] == ["round", "boom"]
    assert registered is False


def test_worker_main_injects_progress_without_pickling_callback(monkeypatch):
    calls: list[dict] = []

    class FakeRuntime:
        def __init__(self, specs, settings, *, device_override=None):
            pass

        def bot_turn(self, *, progress_callback=None, **kwargs):
            calls.append(kwargs)
            assert progress_callback is not None
            progress_callback({"kind": "round"})
            return {"actions": []}

        def release_gpu_cache_if_large(self):
            pass

    monkeypatch.setattr(bots, "_WorkerRuntime", FakeRuntime)
    jobs = _FakeQueue(
        [
            (
                9,
                "move",
                {
                    "bot_slug": "fake",
                    "game_key": 1,
                    "actions": [],
                    "seed": 2,
                    "visits": 3,
                    "watch_search": True,
                },
            ),
            None,
        ]
    )
    results = _FakeQueue()
    bots._worker_main(0, [], SimpleNamespace(), jobs, results, "cpu")
    assert [frame[0] for frame in results.items] == [
        bots._READY, bots._STARTED, bots._PROGRESS, 9,
    ]
    assert "watch_search" not in calls[0]


def test_api_watch_query_is_opt_in_and_stream_is_owner_only(settings, tmp_path):
    from fastapi.testclient import TestClient
    from starlette.requests import Request

    cpu_settings = replace(
        settings,
        db_path=tmp_path / "showcase-live.db",
        device="cpu",
        gpu_reprobe_s=0.0,
    )
    with TestClient(create_app(cpu_settings)) as client:
        headers = {"CF-Connecting-IP": "10.99.0.1"}
        watched = client.post(
            "/api/game?watch_search=1",
            json={"checkpoint_id": "tiny", "sims": 8, "human_color": 1},
            headers=headers,
        )
        assert watched.status_code == 200
        game_id = watched.json()["id"]
        assert client.app.state.sessions[game_id].watch_search is True

        denied = client.get(
            f"/api/game/{game_id}/search-stream",
            headers={**headers, "Cookie": "showcase_token=wrong"},
        )
        assert denied.status_code == 403

        # Exercise the authorized route's real StreamingResponse iterator
        # without leaving an open TestClient stream waiting for a heartbeat.
        hub = client.app.state.live_search_hub
        hub.begin_run(game_id)
        replay_event = hub.publish(
            game_id,
            {
                "kind": "turn_start",
                "run_id": 41,
                "attempt": 1,
                "base_ply": 1,
            },
        )
        route = next(
            item
            for item in client.app.routes
            if getattr(item, "path", None)
            == "/api/game/{game_id}/search-stream"
        )
        token = client.cookies.get("showcase_token")
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": f"/api/game/{game_id}/search-stream",
            "raw_path": f"/api/game/{game_id}/search-stream".encode(),
            "query_string": b"",
            "headers": [
                (b"cookie", f"showcase_token={token}".encode()),
                (b"last-event-id", str(replay_event["seq"] - 1).encode()),
            ],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "root_path": "",
            "app": client.app,
        }

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        request = Request(scope, receive)
        response = client.portal.call(route.endpoint, game_id, request)

        async def first_chunk():
            iterator = response.body_iterator.__aiter__()
            try:
                return await iterator.__anext__()
            finally:
                await iterator.aclose()

        chunk = client.portal.call(first_chunk)
        assert response.media_type == "text/event-stream"
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
        assert chunk == encode_sse_event(replay_event)

        client.post(f"/api/game/{game_id}/resign", headers=headers)
        plain = client.post(
            "/api/game",
            json={"checkpoint_id": "tiny", "sims": 8, "human_color": 1},
            headers={"CF-Connecting-IP": "10.99.0.2"},
        )
        assert plain.status_code == 200
        plain_id = plain.json()["id"]
        assert client.app.state.sessions[plain_id].watch_search is False
        client.post(f"/api/game/{plain_id}/resign")
