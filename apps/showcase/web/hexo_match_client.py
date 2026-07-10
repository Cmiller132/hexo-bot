"""Hexo showcase match client: connect YOUR bot to a showcase server.

Zero dependencies (stdlib urllib). Copy this one file into your project, or
run it directly for a demo match with the built-in random bot:

    python hexo_match_client.py --server https://<host> --agent my-bot \
        --checkpoint <id> --sims 128 --random-demo

Writing an adapter is one method:

    from hexo_match_client import BotAdapter, MatchServer

    class MyBot(BotAdapter):
        def select_stone(self, state: dict) -> tuple[int, int]:
            # state["history"]: chronological [{q, r, color}] for the whole game
            # state["legal"]:   [{q, r}] cells you may play RIGHT NOW
            # state["you"]:     your color (0 moves first)
            cell = my_engine.think(state)
            return cell.q, cell.r

    result = MyBot().play_match(
        MatchServer("https://<host>"), agent="my-bot",
        checkpoint_id="main7-ep90", sims=128, agent_color="random",
    )

`select_stone` is called once per stone. Hexo turns place TWO stones each
(after the one-stone opening), so expect two calls per turn; the fresh state
(with your first stone applied) rides each call.

The game rules in one paragraph: stones go on an unbounded hex grid; the first
player to make six in a row along any of the three hex axes wins. Player 0
opens with a single forced stone at the origin (the server auto-plays it —
games start at ply 1); every turn after that places two stones. `state["legal"]`
is authoritative: play only cells from it.

Wire protocol (full spec: the downloadable bot-api.md on the server's API tab):
    POST /api/match                       -> {match_id, token, state}
    GET  /api/match/{id}?wait=25          long-poll; bearer auth
    POST /api/match/{id}/move {q, r}      one stone per call
    POST /api/match/{id}/resign
    POST /api/match/{id}/retry            after status == "bot_failed"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

DEFAULT_TIMEOUT_S = 40.0   # per-request cap; must exceed the server's max ?wait=25
LONG_POLL_S = 25.0         # server-side cap on ?wait=


class MatchError(RuntimeError):
    """Server rejected a request (4xx) or the match is gone."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class MatchServer:
    """One showcase server. `request` is the single HTTP seam — tests (and
    exotic transports) can subclass and override it."""

    def __init__(self, base_url: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def request(
        self, method: str, path: str, *, body: dict | None = None,
        token: str | None = None,
    ) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + path,
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {token}"} if token else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode()).get("detail", "")
            except Exception:
                detail = exc.reason
            raise MatchError(exc.code, str(detail)) from None

    # -- convenience ---------------------------------------------------------

    def bots(self) -> dict[str, Any]:
        """The server's catalogue: playable checkpoints + allowed sims values."""
        return self.request("GET", "/api/bots")

    def create_match(
        self, *, agent: str, checkpoint_id: str, sims: int,
        agent_color: int | str = 0,
    ) -> "Match":
        payload = self.request(
            "POST", "/api/match",
            body={
                "agent": agent, "checkpoint_id": checkpoint_id,
                "sims": sims, "agent_color": agent_color,
            },
        )
        return Match(self, payload["match_id"], payload["token"], payload["state"])


class Match:
    """Handle for one live match. Keep `token` secret — it IS the auth."""

    def __init__(self, server: MatchServer, match_id: str, token: str, state: dict) -> None:
        self.server = server
        self.match_id = match_id
        self.token = token
        self.state = state

    def _call(self, method: str, path: str, body: dict | None = None) -> dict[str, Any]:
        payload = self.server.request(
            method, f"/api/match/{self.match_id}{path}", body=body, token=self.token
        )
        self.state = payload
        return payload

    def refresh(self, wait: float = 0.0) -> dict[str, Any]:
        """Fetch state; `wait` long-polls server-side until you can act."""
        return self._call("GET", f"?wait={min(max(wait, 0.0), LONG_POLL_S):g}")

    def move(self, q: int, r: int) -> dict[str, Any]:
        """Place one stone. Two-stone turns = two calls (state tells you:
        `stones_left_this_turn`)."""
        return self._call("POST", "/move", {"q": int(q), "r": int(r)})

    def resign(self) -> dict[str, Any]:
        return self._call("POST", "/resign")

    def retry(self) -> dict[str, Any]:
        """Re-run the server bot's turn after `status == "bot_failed"`."""
        return self._call("POST", "/retry")


class BotAdapter(ABC):
    """Subclass, implement `select_stone`, call `play_match`.

    The loop handles everything else: long-polling for your turn, the
    two-stones-per-turn shape, transparent retries when the server bot
    hiccups, and transient-network backoff.
    """

    #: how many times a transient failure (network error, bot_failed) is
    #: retried before resigning the match. Overridable per subclass.
    max_retries = 5

    @abstractmethod
    def select_stone(self, state: dict[str, Any]) -> tuple[int, int]:
        """Return (q, r) of the stone to place now. Must be a cell from
        state["legal"]. Called once per stone (twice per post-opening turn)."""

    def on_state(self, state: dict[str, Any]) -> None:
        """Optional observation hook: called on every fresh state (including
        opponent moves), useful for pondering or logging. Default: no-op."""

    def play_match(
        self, server: MatchServer, *, agent: str, checkpoint_id: str,
        sims: int, agent_color: int | str = 0, verbose: bool = False,
    ) -> dict[str, Any]:
        """Create a match and drive it to completion. Returns the final state
        (state["result"] holds winner/termination)."""
        match = server.create_match(
            agent=agent, checkpoint_id=checkpoint_id, sims=sims,
            agent_color=agent_color,
        )
        if verbose:
            print(f"match {match.match_id} vs {checkpoint_id}@{sims} — you are color {match.state['you']}")
        failures = 0
        while True:
            state = match.state
            self.on_state(state)
            status = state["status"]
            if status == "finished":
                if verbose:
                    print(f"finished: {json.dumps(state['result'])}")
                return state
            if status == "your_turn":
                q, r = self.select_stone(state)
                try:
                    match.move(q, r)
                    failures = 0
                except MatchError as exc:
                    if exc.status == 422:  # our bug: illegal move
                        raise
                    failures += 1
                    if failures > self.max_retries:
                        match.resign()
                        raise
                    time.sleep(min(2.0 * failures, 10.0))
                    match.refresh()
                continue
            if status == "bot_failed":
                failures += 1
                if failures > self.max_retries:
                    return match.resign()
                if verbose:
                    print(f"server bot hiccuped; retrying ({failures}/{self.max_retries})")
                time.sleep(min(2.0 * failures, 10.0))
                match.retry()
                continue
            # bot_thinking: long-poll until something changes.
            match.refresh(wait=LONG_POLL_S)


class RandomBot(BotAdapter):
    """Reference adapter: uniform-random legal stone. Useful as a smoke test
    for your server and as the minimal example of the adapter contract."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_stone(self, state: dict[str, Any]) -> tuple[int, int]:
        cell = self._rng.choice(state["legal"])
        return cell["q"], cell["r"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hexo showcase match client")
    parser.add_argument("--server", required=True, help="server base URL")
    parser.add_argument("--agent", required=True, help="your bot's public name")
    parser.add_argument("--checkpoint", help="opponent checkpoint id (see --list)")
    parser.add_argument("--sims", type=int, help="opponent search budget (see --list)")
    parser.add_argument("--color", default="random", help="0 | 1 | random (default)")
    parser.add_argument("--list", action="store_true", help="list playable bots and exit")
    parser.add_argument(
        "--random-demo", action="store_true",
        help="play one match with the built-in random bot (expect to lose)",
    )
    args = parser.parse_args(argv)

    server = MatchServer(args.server)
    if args.list:
        print(json.dumps(server.bots(), indent=2))
        return 0
    if not args.random_demo:
        parser.error("implement your own BotAdapter, or pass --random-demo")
    if not args.checkpoint or args.sims is None:
        parser.error("--checkpoint and --sims are required (see --list)")
    color: int | str = args.color if args.color == "random" else int(args.color)
    final = RandomBot().play_match(
        server, agent=args.agent, checkpoint_id=args.checkpoint,
        sims=args.sims, agent_color=color, verbose=True,
    )
    return 0 if final["result"] else 1


if __name__ == "__main__":
    sys.exit(main())
