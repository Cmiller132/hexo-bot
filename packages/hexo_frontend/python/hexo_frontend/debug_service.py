"""Server-side manager for the Debug-tab CPU inference worker.

The dashboard HTTP server (``web.py``) stays torch-free; this module owns the
child worker process (``debug_worker``) and brokers requests to it:

* spawns the worker lazily on first use, launched with ``CUDA_VISIBLE_DEVICES=""``
  so it can never contend for the training GPU (default: under the WSL venv that
  matches the run; configurable);
* **serializes** requests behind a lock (the worker is single-threaded, so this
  is the natural request queue) with a per-request timeout and auto-restart if
  the worker wedges or dies;
* **caches** results LRU by (op, checkpoint, position, params) so re-opening the
  same view is instant and never re-hits the worker.

None of this blocks the dashboard's live-status long-poll: torch lives in the
child, and only Debug-tab endpoints ever call in here.
"""

from __future__ import annotations

import json
import os
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Default WSL interpreter for the out-of-process debug worker, used ONLY on the
# Windows -> WSL bridge path (see _worker_command below). Point it at a WSL venv
# that has the native extensions built. Override with HEXO_DEBUG_WSL_PYTHON, or
# replace the whole command with HEXO_DEBUG_WORKER_CMD. On Linux/WSL (the
# documented dashboard flow) the worker runs natively and this is never used.
DEFAULT_WSL_PYTHON = "/root/.venvs/hexo-bot/bin/python"

# Generous per-request ceiling: a cold checkpoint load + a 512-visit CPU search
# can take a few seconds; analyze is sub-second once warm.
DEFAULT_TIMEOUT = 120.0
PING_TIMEOUT = 90.0  # first ping also pays the torch import cost
CACHE_MAX = 256


def _to_wsl(path: Path | str) -> str:
    p = str(path)
    if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
        return f"/mnt/{p[0].lower()}{p[2:].replace(chr(92), '/')}"
    return str(path).replace("\\", "/")


class DebugWorkerError(RuntimeError):
    """Transport/process-level failure: the worker died, wedged, or never started."""


class DebugWorkerTimeout(DebugWorkerError):
    """The worker blew its request deadline (likely mid-compute; must be killed)."""


class DebugRequestError(RuntimeError):
    """Per-request failure reported by a HEALTHY worker (an ``ok:false`` reply).

    The child catches its own request exceptions and keeps serving
    (``debug_worker.main``), so this must never restart the process: restarting
    would drop the worker's checkpoint LRU and re-pay the torch import only to
    hit the same deterministic error again."""


class DebugWorker:
    """Owns one inference child process and serializes requests to it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._lines: "queue.Queue[str | None]" = queue.Queue()
        self._next_id = 0
        self._err_path = Path(tempfile.gettempdir()) / "hexo_debug_worker.err"
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        # The SHRIMP_SUPPORT_RADIUS the live worker process was spawned at
        # (None = not keyed on radius, e.g. dense/hexgt or a cold/legacy spawn).
        # support._SUPPORT_RADIUS is read-once at import, so a radius change can
        # only take effect by respawning the worker with a new spawn env.
        self._radius: int | None = None

    # -- process lifecycle -----------------------------------------------------

    def _argv_and_env(self, radius: int | None) -> tuple[list[str], dict[str, str]]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""  # belt-and-suspenders; worker also forces CPU
        if radius is not None:
            # Shrimp support radius is read-once at import in support.py / the
            # Rust featurizer, so it must be in the spawn env. Only set when a
            # radius applies; a None radius leaves the env untouched.
            env["SHRIMP_SUPPORT_RADIUS"] = str(radius)
        pkg_python_dir = Path(__file__).resolve().parent.parent  # .../hexo_frontend/python

        override = os.environ.get("HEXO_DEBUG_WORKER_CMD")
        if override:
            return shlex.split(override), env

        use_wsl = sys.platform == "win32" and os.environ.get("HEXO_DEBUG_USE_WSL", "1") != "0"
        if use_wsl:
            venv = os.environ.get("HEXO_DEBUG_WSL_PYTHON", DEFAULT_WSL_PYTHON)
            worktree = _to_wsl(Path.cwd())
            pp = _to_wsl(pkg_python_dir)
            # The env dict does NOT cross `wsl.exe -e bash -lc`, so SHRIMP_SUPPORT_RADIUS
            # must be inlined into the command like CUDA_VISIBLE_DEVICES=/PYTHONPATH=.
            rad_inline = f"SHRIMP_SUPPORT_RADIUS={radius} " if radius is not None else ""
            inner = (
                f"cd {shlex.quote(worktree)} && CUDA_VISIBLE_DEVICES= {rad_inline}"
                f"PYTHONPATH={shlex.quote(pp)} {shlex.quote(venv)} -u -m hexo_frontend.debug_worker"
            )
            return ["wsl.exe", "-e", "bash", "-lc", inner], env

        # Native fallback (worker on the same interpreter, e.g. tests under WSL).
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(pkg_python_dir) + (os.pathsep + existing if existing else "")
        return [sys.executable, "-u", "-m", "hexo_frontend.debug_worker"], env

    def _spawn(self, radius: int | None) -> None:
        argv, env = self._argv_and_env(radius)
        err = open(self._err_path, "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            env=env,
            text=True,
            bufsize=1,
        )
        self._radius = radius  # record the radius the live process was spawned at
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, args=(self._proc,), daemon=True)
        self._reader.start()

    def _read_loop(self, proc: subprocess.Popen) -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._lines.put(line)
        finally:
            self._lines.put(None)  # sentinel: stdout closed

    def _alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _ensure_started(self, radius: int | None) -> None:
        if self._alive():
            return
        self._spawn(radius)
        self._ping()

    def _err_tail(self) -> str:
        try:
            text = self._err_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return text[-1200:]

    def _ping(self) -> None:
        resp = self._exchange({"op": "ping"}, timeout=PING_TIMEOUT)
        if not (isinstance(resp, dict) and resp.get("pong")):
            raise DebugWorkerError(f"worker ping failed; stderr tail:\n{self._err_tail()}")

    # -- request plumbing ------------------------------------------------------

    def _exchange(self, payload: dict[str, Any], *, timeout: float) -> Any:
        """Send one request, return its result (assumes lock held + proc alive)."""

        assert self._proc is not None and self._proc.stdin is not None
        self._next_id += 1
        req_id = self._next_id
        msg = dict(payload)
        msg["id"] = req_id
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise DebugWorkerError(f"worker stdin broken: {exc}") from exc

        while True:
            try:
                line = self._lines.get(timeout=timeout)
            except queue.Empty as exc:
                raise DebugWorkerTimeout(f"worker timed out after {timeout:.0f}s") from exc
            if line is None:
                raise DebugWorkerError(f"worker exited; stderr tail:\n{self._err_tail()}")
            line = line.strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue  # ignore any non-protocol noise that reached stdout
            if resp.get("id") != req_id:
                continue  # stale/mismatched; keep reading
            if not resp.get("ok"):
                # The worker replied, so the process is healthy: this is an
                # application-level error (bad request), not a transport one.
                raise DebugRequestError(str(resp.get("error", "unknown worker error")))
            return resp.get("result")

    def request(self, op: str, *, timeout: float = DEFAULT_TIMEOUT,
                radius: int | None = None, **fields: Any) -> Any:
        """Serialized request to the worker.

        ``radius`` is a worker-lifecycle param (the target SHRIMP_SUPPORT_RADIUS),
        NOT a request field: it is never forwarded into ``_exchange``. ``None``
        means "don't care — reuse whatever is running, spawn at None if cold", so
        dense/hexgt/record_row requests never trigger a respawn. A shrimp request
        passes a concrete int and respawns the worker iff it differs from the live
        process radius (support._SUPPORT_RADIUS is read-once at import).

        Failure handling, by class:
        * ``DebugRequestError`` (worker answered ``ok:false``) propagates without
          touching the process — the worker is healthy and a retry of the same
          request would deterministically fail again.
        * ``DebugWorkerTimeout`` kills the (likely mid-compute) worker so the
          next request gets a fresh process, but is NOT retried: resending the
          identical request would just burn the full budget a second time.
        * Any other ``DebugWorkerError`` (dead process, broken pipe) restarts
          the worker once and resends — that path self-heals."""

        with self._lock:
            if radius is not None and self._alive() and self._radius != radius:
                # Live worker is at the wrong radius for this request; respawn it.
                self.shutdown_locked()
            try:
                self._ensure_started(radius)
                return self._exchange({"op": op, **fields}, timeout=timeout)
            except DebugWorkerTimeout:
                self.shutdown_locked()
                raise
            except DebugWorkerError:
                self.shutdown_locked()
                # One restart attempt: a wedged/crashed worker should self-heal.
                self._ensure_started(radius)
                return self._exchange({"op": op, **fields}, timeout=timeout)

    # -- cached requests -------------------------------------------------------

    def cached(self, signature: str, op: str, *, timeout: float = DEFAULT_TIMEOUT,
               radius: int | None = None, **fields: Any) -> Any:
        """request() behind the LRU result cache.

        ``signature`` is the caller-built cache key (web.py's _debug_signature:
        a JSON of [prefix+params, checkpoint path, action ids, n, radius]) and
        must capture every input that changes the result — a stale hit is served
        verbatim with no worker round-trip. Errors are never cached. ``radius``
        is forwarded to request() as a lifecycle param; it enters the cache via
        the signature (the caller folds it in), not the lookup here."""

        with self._lock:
            hit = self._cache.get(signature)
            if hit is not None:
                self._cache.move_to_end(signature)
                return hit
        result = self.request(op, timeout=timeout, radius=radius, **fields)
        with self._lock:
            self._cache[signature] = result
            self._cache.move_to_end(signature)
            while len(self._cache) > CACHE_MAX:
                self._cache.popitem(last=False)
        return result

    def status(self) -> dict[str, Any]:
        return {
            "alive": self._alive(),
            "radius": self._radius,
            "cached_results": len(self._cache),
            "mode": "wsl" if (sys.platform == "win32" and os.environ.get("HEXO_DEBUG_USE_WSL", "1") != "0"
                              and not os.environ.get("HEXO_DEBUG_WORKER_CMD")) else "native",
        }

    # -- shutdown --------------------------------------------------------------

    def shutdown_locked(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def shutdown(self) -> None:
        with self._lock:
            self.shutdown_locked()


# Module-level singleton: one worker per dashboard process.
_WORKER: DebugWorker | None = None
_WORKER_LOCK = threading.Lock()


def get_worker() -> DebugWorker:
    """Lazily-created module singleton — the one worker every web.py debug
    endpoint (and the Match Arena checkpoint bots) shares."""

    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = DebugWorker()
        return _WORKER
