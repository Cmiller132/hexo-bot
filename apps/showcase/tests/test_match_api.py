"""External bot match API (/api/match/*) + SDK adapter loop, end to end.

Uses the session-scoped app/client from conftest (tiny real checkpoint, one
worker). The SDK is exercised against the FastAPI TestClient through a
transport shim, so the full BotAdapter loop — long-poll, two-stones-per-turn,
finish detection — runs a real match against the real worker pool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk"))

from hexo_match_client import BotAdapter, MatchError, MatchServer, RandomBot  # noqa: E402


def test_web_sdk_copy_in_sync():
    """The downloadable SDK served from web/ must be byte-identical to the
    canonical sdk/ file — the API tab hands it to bot developers."""
    base = Path(__file__).resolve().parents[1]
    canonical = (base / "sdk" / "hexo_match_client.py").read_bytes()
    served = (base / "web" / "hexo_match_client.py").read_bytes()
    assert canonical == served, "run: cp sdk/hexo_match_client.py web/"


def _create(client, **overrides):
    body = {"agent": "pytest-bot", "checkpoint_id": "tiny", "sims": 8, "agent_color": 0}
    body.update(overrides)
    return client.post("/api/match", json=body)


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# -- endpoint contract ----------------------------------------------------------


def test_create_match_shape(client):
    # agent_color=1: the forced color-0 opening stone is auto-played at
    # creation and color 1 (the agent) moves next, so the match starts
    # immediately in your_turn with no bot search pending.
    resp = _create(client, agent_color=1)
    assert resp.status_code == 200
    payload = resp.json()
    assert set(payload) == {"match_id", "token", "idle_timeout_s", "state"}
    state = payload["state"]
    assert state["you"] == 1
    assert state["status"] == "your_turn"
    assert state["agent"] == "pytest-bot"
    assert state["ply"] == 1
    assert state["history"][0] == {"q": 0, "r": 0, "color": 0}
    # Clean up so later tests don't hit the active-game cap.
    client.post(f"/api/match/{payload['match_id']}/resign", headers=_auth(payload["token"]))


def test_match_auth_required(client):
    payload = _create(client).json()
    match_id = payload["match_id"]
    assert client.get(f"/api/match/{match_id}").status_code == 401
    assert (
        client.get(f"/api/match/{match_id}", headers=_auth("wrong")).status_code == 403
    )
    # A human-game read of an ACTIVE match is denied too (no cookie).
    assert client.get(f"/api/game/{match_id}").status_code in (403, 404)
    client.post(f"/api/match/{match_id}/resign", headers=_auth(payload["token"]))


def test_match_validation(client):
    assert _create(client, checkpoint_id="nope").status_code == 404
    assert _create(client, sims=7).status_code == 422
    assert _create(client, agent="!!!").status_code == 422


def test_human_endpoints_reject_match_kind(client):
    """A match session is not reachable through the human move endpoint with
    the bearer token (cookie auth is a different namespace)."""
    payload = _create(client).json()
    match_id = payload["match_id"]
    resp = client.post(
        f"/api/game/{match_id}/move", json={"q": 5, "r": 5},
        headers=_auth(payload["token"]),
    )
    assert resp.status_code == 403
    client.post(f"/api/match/{match_id}/resign", headers=_auth(payload["token"]))


def test_move_and_wait_flow(client):
    payload = _create(client, agent_color=0).json()  # server bot plays turn 2
    match_id, token = payload["match_id"], payload["token"]
    assert payload["state"]["status"] == "bot_thinking"
    # Long-poll until the bot's opening turn is done.
    state = client.get(f"/api/match/{match_id}?wait=25", headers=_auth(token)).json()
    assert state["status"] == "your_turn"
    assert state["stones_left_this_turn"] == 2
    assert state["legal"], "legal moves must ride your_turn states"
    # Moving out of turn / illegal cell handling.
    occupied = state["history"][0]
    resp = client.post(
        f"/api/match/{match_id}/move",
        json={"q": occupied["q"], "r": occupied["r"]},
        headers=_auth(token),
    )
    assert resp.status_code == 422
    # Two legal stones complete the turn and hand over to the bot.
    for stones_left in (2, 1):
        assert state["stones_left_this_turn"] == stones_left
        cell = state["legal"][0]
        state = client.post(
            f"/api/match/{match_id}/move", json=cell, headers=_auth(token)
        ).json()
    assert state["status"] in ("bot_thinking", "finished")
    final = client.post(f"/api/match/{match_id}/resign", headers=_auth(token)).json()
    assert final["status"] == "finished"
    assert final["result"]["termination"] == "resign"
    assert final["result"]["human_result"] == -1  # resigning loses


def test_resigned_match_counts_in_feed_and_stats(client):
    payload = _create(client, agent="feed-check-bot").json()
    match_id, token = payload["match_id"], payload["token"]
    client.post(f"/api/match/{match_id}/resign", headers=_auth(token))
    feed = client.get("/api/games").json()["games"]
    mine = [g for g in feed if g["id"] == match_id]
    assert mine and mine[0]["nickname"] == "feed-check-bot"
    # Finished matches are publicly readable through the human game endpoint,
    # exactly like finished human games (shareable URLs, analysis).
    assert client.get(f"/api/game/{match_id}").status_code == 200


# -- SDK adapter, full match ------------------------------------------------------


class _TestClientServer(MatchServer):
    """SDK transport shim: route through the in-process FastAPI TestClient."""

    def __init__(self, client) -> None:
        super().__init__("http://testserver")
        self._client = client

    def request(self, method, path, *, body=None, token=None):
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        resp = self._client.request(method, path, json=body, headers=headers)
        if resp.status_code >= 400:
            raise MatchError(resp.status_code, resp.json().get("detail", ""))
        return resp.json()


class _FirstLegalBot(BotAdapter):
    """Deterministic minimal adapter; also counts its stone calls."""

    def __init__(self) -> None:
        self.calls = 0

    def select_stone(self, state):
        self.calls += 1
        cell = state["legal"][0]
        return cell["q"], cell["r"]


@pytest.mark.parametrize("adapter_cls", [_FirstLegalBot, lambda: RandomBot(seed=7)])
def test_sdk_plays_full_match(client, adapter_cls):
    bot = adapter_cls()
    final = bot.play_match(
        _TestClientServer(client), agent="sdk-e2e-bot",
        checkpoint_id="tiny", sims=8, agent_color=0,
    )
    assert final["status"] == "finished"
    assert final["result"]["termination"] in ("six_in_line", "resign")
    # The record is queryable afterwards like any finished game.
    assert client.get(f"/api/game/{final['match_id']}").status_code == 200
