"""Unit tests for BotPool._submit fault handling: fast worker-death detection,
GPU->CPU failover on device fault, and transparent auto-retry.

These drive _submit with fake processes/queues (no torch, no real subprocess) so
the recycle/retry control flow is exercised deterministically — the class of bug
(a mp-pickling/identity mismatch) that a coarser harness missed before. Written
as plain sync tests over asyncio.run so no async pytest plugin is required.
"""
from __future__ import annotations

import asyncio
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from showcase import bots  # noqa: E402
from showcase.bots import BotPool, BotPoolError, BotPoolTimeout  # noqa: E402

bots._LIVENESS_POLL_S = 0.02  # fast polling for the tests


class FakeProc:
    def __init__(self) -> None:
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.alive = False

    def kill(self) -> None:
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        return None


class FakeQueue:
    def __init__(self) -> None:
        self.items: list = []

    def put(self, item) -> None:
        self.items.append(item)


class FakeCtx:
    def Queue(self) -> FakeQueue:  # noqa: N802 (mp API shape)
        return FakeQueue()


class FakeSettings:
    workers = 1
    move_timeout_s = 0.5
    bot_timeout_s = 0.5
    max_recycles_per_window = 3
    recycle_window_s = 300.0
    gpu_fault_threshold = 1  # this deployment fails over on the first fault
    gpu_reprobe_s = 0.0
    gpu_reprobe_healthy_streak = 2
    device = "xpu"


def _make_pool() -> BotPool:
    """A BotPool wired for single-shard testing without start()/real procs.
    Must be called inside a running loop (uses get_running_loop)."""
    pool = BotPool.__new__(BotPool)
    pool._specs = []
    pool._settings = FakeSettings()
    pool._ctx = FakeCtx()
    pool._job_queues = [FakeQueue()]
    pool._procs = [FakeProc()]
    pool._futures = {}
    pool._job_worker = {}
    pool._job_ids = itertools.count(1)
    pool._loop = asyncio.get_running_loop()
    pool._recycle_locks = [asyncio.Lock()]
    pool._recycle_times = [[]]
    pool._poisoned = [False]
    pool._worker_devices = [None]
    pool._gpu_fault_times = [[]]
    pool._reprobe_streak = 0
    pool._reprobe_task = None
    pool._accel_device = "xpu"
    pool._ready_waiters = {}
    pool._ready_pending = {}
    pool._stopping = False

    def fake_spawn(index, queue):
        proc = FakeProc()
        pool._procs[index] = proc
        return proc

    async def fake_await_ready(index):
        return None

    pool._spawn_proc = fake_spawn  # type: ignore[assignment]
    pool._await_ready = fake_await_ready  # type: ignore[assignment]
    return pool


def _last_job_id(queue: FakeQueue) -> int:
    return queue.items[-1][0]


def test_happy_path_returns_ok():
    async def scenario():
        pool = _make_pool()

        async def worker():
            await asyncio.sleep(0.01)
            pool._resolve(_last_job_id(pool._job_queues[0]), {"ok": {"move": 42}})

        task = asyncio.ensure_future(worker())
        out = await pool._submit(0, "move", {}, 0.5, recycle_on_hang=True, retries=1)
        await task
        return out

    assert asyncio.run(scenario()) == {"move": 42}


def test_worker_death_triggers_failover_and_retry_succeeds():
    async def scenario():
        pool = _make_pool()
        calls = {"n": 0}

        async def worker():
            while not pool._job_queues[0].items:  # first job arrives
                await asyncio.sleep(0.005)
            calls["n"] += 1
            pool._procs[0].alive = False  # simulate SIGSEGV death, no reply
            await asyncio.sleep(0.06)  # let _await_reply detect + recycle
            q = pool._job_queues[0]  # fresh queue after recycle
            while not q.items:
                await asyncio.sleep(0.005)
            calls["n"] += 1
            pool._resolve(_last_job_id(q), {"ok": {"move": 7}})

        task = asyncio.ensure_future(worker())
        out = await pool._submit(0, "move", {}, 0.5, recycle_on_hang=True, retries=1)
        await task
        return out, pool._worker_devices[0], calls["n"]

    out, dev, n = asyncio.run(scenario())
    assert out == {"move": 7}
    assert dev == "cpu"  # threshold=1 -> failover on first fault
    assert n == 2


def test_device_fault_error_payload_retries():
    async def scenario():
        pool = _make_pool()
        attempts = {"n": 0}

        async def worker():
            while not pool._job_queues[0].items:
                await asyncio.sleep(0.005)
            attempts["n"] += 1
            pool._resolve(_last_job_id(pool._job_queues[0]),
                          {"error": "torch.OutOfMemoryError: XPU out of memory"})
            await asyncio.sleep(0.06)
            q = pool._job_queues[0]
            while not q.items:
                await asyncio.sleep(0.005)
            attempts["n"] += 1
            pool._resolve(_last_job_id(q), {"ok": {"ok": True}})

        task = asyncio.ensure_future(worker())
        out = await pool._submit(0, "analyze", {}, 0.5, recycle_on_hang=True, retries=1)
        await task
        return out, pool._worker_devices[0], attempts["n"]

    out, dev, n = asyncio.run(scenario())
    assert out == {"ok": True}
    assert dev == "cpu"
    assert n == 2


def test_retries_exhausted_raises():
    async def scenario():
        pool = _make_pool()

        async def worker():
            for _ in range(5):
                while not pool._job_queues[0].items:
                    await asyncio.sleep(0.005)
                jid = _last_job_id(pool._job_queues[0])
                pool._job_queues[0].items.clear()
                pool._resolve(jid, {"error": "xpu device-side assert"})
                await asyncio.sleep(0.03)

        task = asyncio.ensure_future(worker())
        raised = False
        try:
            await pool._submit(0, "move", {}, 0.5, recycle_on_hang=True, retries=1)
        except BotPoolError:
            raised = True
        task.cancel()
        return raised

    assert asyncio.run(scenario()) is True


def test_true_hang_times_out():
    async def scenario():
        pool = _make_pool()  # worker never replies, stays alive -> hang
        raised = False
        try:
            await pool._submit(0, "move", {}, 0.2, recycle_on_hang=True, retries=0)
        except BotPoolTimeout:
            raised = True
        return raised

    assert asyncio.run(scenario()) is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
