"""Chronological ELO ratings for the Game Stats tab — stdlib only.

Two independent rating pools share the same board: PLAYERS keyed by nickname
string (NULL/empty collapse to the `""` bucket, displayed "(anonymous)") and
BOTS keyed by their bots-table id. Every finished game is one head-to-head
between a player and a bot; both sides update from their own perspective with a
provisional K until they clear 10 games.

`compute_ratings` is a pure function over already-fetched rows — no DB handle,
no torch, no numpy — so the app can memoize it on the finished-game count and
the tests can exercise the math with plain `python`. Games MUST arrive in
chronological order (finished_at, id ascending); the caller's query guarantees
that, and the ordering is load-bearing because each update reads the running
rating.
"""

from __future__ import annotations

from typing import Any

_START = 1000.0
_K_PROVISIONAL = 40.0
_K_SETTLED = 20.0
_PROVISIONAL_GAMES = 10
_ANON_DISPLAY = "(anonymous)"


def _k_for(games: int) -> float:
    """Higher swing while an entity is provisional (its own game count < 10)."""
    return _K_PROVISIONAL if games < _PROVISIONAL_GAMES else _K_SETTLED


def _expected(rating_self: float, rating_other: float) -> float:
    """Logistic expected score of `self` against `other` (400-point scale)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_other - rating_self) / 400.0))


def _player_key(nickname: Any) -> str:
    """NULL / empty / whitespace nickname -> the shared anonymous bucket."""
    if nickname is None:
        return ""
    key = str(nickname).strip()
    return key


def compute_ratings(
    games_rows: list[Any], bots_index: dict[int, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Fold finished games into player and bot ratings.

    `games_rows` is an ordered iterable of dict/sqlite.Row with keys id,
    bot_id, result, nickname, finished_at (result in {-1, 0, 1}; +1 = human
    won, -1 = bot won, 0 = draw). `bots_index` maps bot_id ->
    {checkpoint_id, label, run, epoch, sims}. Returns {"players": [...],
    "bots": [...]}, each sorted by rating descending, ratings rounded to int in
    the output only.
    """
    players: dict[str, dict[str, Any]] = {}
    bots: dict[int, dict[str, Any]] = {}

    def _player(key: str) -> dict[str, Any]:
        entry = players.get(key)
        if entry is None:
            entry = {
                "key": key,
                "rating": _START,
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
            }
            players[key] = entry
        return entry

    def _bot(bot_id: int) -> dict[str, Any]:
        entry = bots.get(bot_id)
        if entry is None:
            entry = {
                "bot_id": bot_id,
                "rating": _START,
                "games": 0,
                "bot_wins": 0,
                "human_wins": 0,
                "draws": 0,
            }
            bots[bot_id] = entry
        return entry

    for row in games_rows:
        result = row["result"]
        if result not in (-1, 0, 1):
            continue
        bot_id = row["bot_id"]
        player = _player(_player_key(row["nickname"]))
        bot = _bot(bot_id)

        # Player-perspective actual score; bot's is the complement.
        if result == 1:
            s_player = 1.0
        elif result == -1:
            s_player = 0.0
        else:
            s_player = 0.5
        e_player = _expected(player["rating"], bot["rating"])
        e_bot = 1.0 - e_player

        k_player = _k_for(player["games"])
        k_bot = _k_for(bot["games"])
        player["rating"] += k_player * (s_player - e_player)
        bot["rating"] += k_bot * ((1.0 - s_player) - e_bot)

        # Per-side record, then bump both game counts (K reads the pre-game
        # count, so the increment lands after the rating update).
        if result == 1:
            player["wins"] += 1
            bot["human_wins"] += 1
        elif result == -1:
            player["losses"] += 1
            bot["bot_wins"] += 1
        else:
            player["draws"] += 1
            bot["draws"] += 1
        player["games"] += 1
        bot["games"] += 1

    return {
        "players": _players_out(players),
        "bots": _bots_out(bots, bots_index),
    }


def _winrate(wins: int, games: int) -> float:
    return (wins / games) if games else 0.0


def _players_out(players: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for entry in players.values():
        games = entry["games"]
        key = entry["key"]
        out.append(
            {
                "nickname": key,
                "display": _ANON_DISPLAY if key == "" else key,
                "rating": round(entry["rating"]),
                "games": games,
                "wins": entry["wins"],
                "losses": entry["losses"],
                "draws": entry["draws"],
                "winrate": _winrate(entry["wins"], games),
                "provisional": games < _PROVISIONAL_GAMES,
            }
        )
    out.sort(key=lambda e: e["rating"], reverse=True)
    return out


def _bots_out(
    bots: dict[int, dict[str, Any]], bots_index: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    out = []
    for bot_id, entry in bots.items():
        meta = bots_index.get(bot_id, {})
        games = entry["games"]
        out.append(
            {
                "bot_id": bot_id,
                "checkpoint_id": meta.get("checkpoint_id", ""),
                "label": meta.get("label", ""),
                "run": meta.get("run", ""),
                "epoch": meta.get("epoch", 0),
                "sims": meta.get("sims", 0),
                "rating": round(entry["rating"]),
                "games": games,
                "bot_wins": entry["bot_wins"],
                "human_wins": entry["human_wins"],
                "draws": entry["draws"],
                "human_winrate": _winrate(entry["human_wins"], games),
                "bot_winrate": _winrate(entry["bot_wins"], games),
                "provisional": games < _PROVISIONAL_GAMES,
            }
        )
    out.sort(key=lambda e: e["rating"], reverse=True)
    return out
