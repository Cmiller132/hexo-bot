# Bot Match API

Connect your own Hexo bot to the showcase server and play **rated matches**
against any catalogue checkpoint. Matches are first-class games: they appear in
the recent-games feed, count in the stats, and your agent name is ranked on the
ELO leaderboard exactly like a human player's nickname.

The fastest path is the Python SDK ([`hexo_match_client.py`](./hexo_match_client.py)
— one stdlib-only file; subclass `BotAdapter`, implement `select_stone`, done).
This document is the wire protocol underneath it, for adapters in any language.

## The game, briefly

Hexo is a Connect6-family game on an **unbounded** hex grid (axial `q, r`
coordinates; the three win axes are `(1,0)`, `(0,1)`, `(1,-1)`). Six in a row
wins. Color `0` moves first with a single forced opening stone at the origin —
the server auto-plays it, so every game starts at ply 1 — and **every turn
after that places two stones** (two `/move` calls). You never need to compute
legality yourself: the state payload's `legal` array is authoritative.

## Lifecycle

```
POST /api/match                          create -> {match_id, token, state}
GET  /api/match/{id}?wait=25             poll (long-poll) for your turn
POST /api/match/{id}/move    {q, r}      place ONE stone
POST /api/match/{id}/resign
POST /api/match/{id}/retry               re-run a hiccuped server-bot turn
```

All calls after create carry `Authorization: Bearer <token>`. The token is
returned **once**, at create. There is no recovery — treat a lost token as a
resignation (the match idles out server-side after `idle_timeout_s`, reported
in the create response, and is recorded as a timeout).

### Create

```
POST /api/match
{"agent": "my-bot", "checkpoint_id": "main7-ep90", "sims": 128, "agent_color": "random"}
```

- `agent` — your bot's public name (1-64 chars; sanitized to `A-Za-z0-9 _.-`,
  max 24 after sanitization). This is the identity your results aggregate
  under, so keep it stable across matches.
- `checkpoint_id`, `sims` — the opponent: any combination from `GET /api/bots`
  (`checkpoints[].id` × `sims[]`).
- `agent_color` — `0` (you move first), `1`, or `"random"`.

Response: `{"match_id", "token", "idle_timeout_s", "state"}`.

### State payload

Every endpoint returns the same state shape:

```jsonc
{
  "match_id": "…",
  "agent": "my-bot",
  "you": 0,                       // your color
  "status": "your_turn",          // your_turn | bot_thinking | bot_failed | finished
  "to_move": 0,                   // color to move; null once finished
  "ply": 5,                       // moves played so far
  "phase": "first_stone",         // engine turn phase
  "stones_left_this_turn": 2,     // stones the mover still places this turn
  "history": [                    // chronological move list, ply order
    {"q": 0, "r": 0, "color": 0}, // …
  ],
  "legal": [{"q": 1, "r": -1}],   // cells YOU may play now ([] unless your_turn)
  "last_move": {"q": 0, "r": 2, "color": 1},
  "winning_line": null,           // the six+ cells, on six_in_line finishes
  "result": null,                 // until finished; then:
  // "result": {"winner": 1, "termination": "six_in_line", "human_result": -1}
  "bot": {"checkpoint_id": "main7-ep90", "label": "…", "epoch": 90, "sims": 128}
}
```

`history` carries the whole game — an adapter can be stateless and rebuild its
position from scratch every turn, or keep its own state and just consume the
tail. `result.human_result` is from YOUR perspective: `+1` you won, `-1` the
server bot won, `0` no result (abandoned).

### Polling

`GET /api/match/{id}?wait=25` blocks server-side (up to 25 s) until `status`
is something you can act on: `your_turn`, `bot_failed`, or `finished`. Loop on
it — no sleep needed client-side. `wait=0` returns immediately.

### Moving

One stone per `/move` call. On a two-stone turn the first call leaves
`status == "your_turn"` with `stones_left_this_turn: 1` — call again. When
your turn completes, the server bot's reply search starts automatically and
`status` flips to `bot_thinking`.

Illegal cells get `422` (the position is unchanged — pick from `legal`).
`409` means it isn't your turn (or the match is over): re-poll and reconcile.

### Server-bot hiccups

If the opponent's search fails transiently (`status == "bot_failed"` — rare;
accelerator faults are retried internally first), the position is intact.
`POST …/retry` re-runs the turn. The SDK does this automatically with backoff.

## Limits and manners

Matches share the human-game abuse budget per client IP: active-match caps and
create/move token buckets (429 with a human-readable `detail` when exceeded).
A match that hears nothing for `idle_timeout_s` (default 600 s) is finalized
as a timeout — long thinks are fine, going dark is not. One more consequence
of sharing the human machinery: **matches consume the same worker pool as live
human games**, so please don't farm hundreds of games — a handful of
concurrent matches is the intended envelope.

## Minimal adapter (no SDK)

```python
import json, urllib.request

BASE = "https://your-server"

def call(method, path, body=None, token=None):
    req = urllib.request.Request(
        BASE + path, method=method,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.loads(r.read())

created = call("POST", "/api/match", {"agent": "curl-bot", "checkpoint_id": "main7-ep90",
                                      "sims": 64, "agent_color": 0})
match_id, token, state = created["match_id"], created["token"], created["state"]
while state["status"] != "finished":
    if state["status"] == "your_turn":
        cell = my_choose(state)          # <- your bot
        state = call("POST", f"/api/match/{match_id}/move", cell, token)
    elif state["status"] == "bot_failed":
        state = call("POST", f"/api/match/{match_id}/retry", {}, token)
    else:
        state = call("GET", f"/api/match/{match_id}?wait=25", token=token)
print(state["result"])
```
