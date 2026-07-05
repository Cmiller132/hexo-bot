"""SQLite persistence for the showcase (schema per docs: bots / games /
analysis_cache + three stats views).

Single file, WAL mode; the app process is the only writer and its write volume
is one row per game plus occasional analysis-cache puts, so plain `sqlite3`
with a process-wide lock is sufficient — no ORM, no connection pool. Worker
processes never touch the DB.

Analysis payloads are stored as gzip-compressed JSON (`encode_payload` /
`decode_payload`) so the cache stays stdlib-only end to end.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bots (
  id          INTEGER PRIMARY KEY,
  slug        TEXT NOT NULL,
  label       TEXT NOT NULL,
  run         TEXT NOT NULL,
  epoch       INTEGER NOT NULL,
  visits      INTEGER NOT NULL,
  weights_sha TEXT NOT NULL,
  active_from TEXT NOT NULL
);
-- A "bot" is a (checkpoint, sims) combination: slug is the catalogue
-- checkpoint id, visits the per-game search budget. Rows are created lazily
-- on the first game with that combination, so stats stay per-strength.
CREATE UNIQUE INDEX IF NOT EXISTS bots_identity
  ON bots (slug, weights_sha, visits);

CREATE TABLE IF NOT EXISTS games (
  id           TEXT PRIMARY KEY,
  bot_id       INTEGER NOT NULL REFERENCES bots(id),
  human_color  INTEGER NOT NULL,
  started_at   TEXT NOT NULL,
  finished_at  TEXT,
  status       TEXT NOT NULL,
  result       INTEGER,
  termination  TEXT,
  ply_count    INTEGER,
  duration_s   REAL,
  nickname     TEXT,
  client_hash  TEXT,
  record       BLOB
);
CREATE INDEX IF NOT EXISTS games_bot_time ON games (bot_id, finished_at);
CREATE INDEX IF NOT EXISTS games_status   ON games (status);
CREATE INDEX IF NOT EXISTS games_feed     ON games (status, finished_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS analysis_cache (
  game_id  TEXT NOT NULL REFERENCES games(id),
  ply      INTEGER NOT NULL,
  bot_id   INTEGER NOT NULL REFERENCES bots(id),
  payload  BLOB NOT NULL,
  PRIMARY KEY (game_id, ply, bot_id)
);

CREATE VIEW IF NOT EXISTS v_bot_stats AS
SELECT b.slug, b.label, b.epoch, b.visits,
       COUNT(*)                                   AS games,
       AVG(g.result = -1)                         AS bot_winrate,
       AVG(g.ply_count)                           AS avg_plies,
       AVG(g.duration_s)                          AS avg_duration_s
FROM games g JOIN bots b ON b.id = g.bot_id
WHERE g.status = 'finished'
GROUP BY g.bot_id;

CREATE VIEW IF NOT EXISTS v_daily AS
SELECT date(started_at) AS day, COUNT(*) AS games,
       SUM(status = 'finished') AS finished
FROM games GROUP BY day;

CREATE VIEW IF NOT EXISTS v_hall_of_fame AS
SELECT g.nickname, b.label, g.ply_count, g.finished_at
FROM games g JOIN bots b ON b.id = g.bot_id
WHERE g.result = +1 AND g.nickname IS NOT NULL
ORDER BY b.visits DESC, g.ply_count ASC;
"""


def encode_payload(payload: dict[str, Any]) -> bytes:
    """Analysis payload dict -> gzip(JSON) blob for `analysis_cache.payload`."""
    return gzip.compress(json.dumps(payload, separators=(",", ":")).encode())


def decode_payload(blob: bytes) -> dict[str, Any]:
    """Inverse of `encode_payload`."""
    return json.loads(gzip.decompress(blob).decode())


class ShowcaseDB:
    """Thread-safe facade over the showcase SQLite file.

    Every method takes the internal lock; calls are short (single statements),
    so holding it across the event loop is fine at showcase volume.
    """

    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        if path.parent and str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- bots ---------------------------------------------------------------

    def upsert_bot(
        self, *, slug: str, label: str, run: str, epoch: int, visits: int,
        weights_sha: str, active_from: str,
    ) -> int:
        """Insert-or-refresh a bot row keyed by (slug, weights_sha, visits) —
        one row per played (checkpoint, sims) combination.

        A checkpoint refresh under the same slug changes the sha and therefore
        creates NEW rows — old games keep their true bot identity. A label
        tweak for the same weights updates in place. Returns the row id.
        """
        with self._lock:
            self._conn.execute(
                "INSERT INTO bots (slug, label, run, epoch, visits, weights_sha, active_from)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (slug, weights_sha, visits) DO UPDATE SET"
                "   label = excluded.label",
                (slug, label, run, epoch, visits, weights_sha, active_from),
            )
            row = self._conn.execute(
                "SELECT id FROM bots WHERE slug = ? AND weights_sha = ? AND visits = ?",
                (slug, weights_sha, visits),
            ).fetchone()
            self._conn.commit()
            return int(row["id"])

    def get_bot(self, bot_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM bots WHERE id = ?", (bot_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    # -- games ----------------------------------------------------------------

    def create_game(
        self, *, game_id: str, bot_id: int, human_color: int, started_at: str,
        client_hash: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO games (id, bot_id, human_color, started_at, status, client_hash)"
                " VALUES (?, ?, ?, ?, 'active', ?)",
                (game_id, bot_id, human_color, started_at, client_hash),
            )
            self._conn.commit()

    def finalize_game(
        self, *, game_id: str, finished_at: str, status: str, result: int,
        termination: str | None, ply_count: int, duration_s: float, record: bytes,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE games SET finished_at = ?, status = ?, result = ?,"
                " termination = ?, ply_count = ?, duration_s = ?, record = ?"
                " WHERE id = ?",
                (finished_at, status, result, termination, ply_count, duration_s,
                 record, game_id),
            )
            self._conn.commit()

    def set_nickname(self, game_id: str, nickname: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE games SET nickname = ? WHERE id = ?", (nickname, game_id)
            )
            self._conn.commit()

    def get_game(self, game_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM games WHERE id = ?", (game_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def list_finished(
        self, *, limit: int, before_finished_at: str | None = None,
        before_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recent-games feed page: finished games only, newest first, keyset
        cursor on (finished_at, id) so pages are stable under same-second
        finishes. Returns games joined with their bot row."""
        query = (
            "SELECT g.id, g.human_color, g.result, g.termination, g.ply_count,"
            " g.finished_at, g.nickname,"
            " b.slug AS bot_slug, b.label AS bot_label, b.epoch AS bot_epoch,"
            " b.visits AS bot_visits"
            " FROM games g JOIN bots b ON b.id = g.bot_id"
            " WHERE g.status = 'finished'"
        )
        params: list[Any] = []
        if before_finished_at is not None:
            query += " AND (g.finished_at < ? OR (g.finished_at = ? AND g.id < ?))"
            params += [before_finished_at, before_finished_at, before_id or ""]
        query += " ORDER BY g.finished_at DESC, g.id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def abandon_stale_active(self, finished_at: str) -> int:
        """Mark leftover 'active' rows abandoned (startup hygiene: sessions are
        in-memory only, so an active row after a restart is unrecoverable).
        Returns the number of rows swept."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE games SET status = 'abandoned', finished_at = ?, result = 0"
                " WHERE status = 'active'",
                (finished_at,),
            )
            self._conn.commit()
            return cur.rowcount

    # -- analysis cache -------------------------------------------------------

    def analysis_get(self, game_id: str, ply: int, bot_id: int) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM analysis_cache WHERE game_id = ? AND ply = ? AND bot_id = ?",
                (game_id, ply, bot_id),
            ).fetchone()
        return bytes(row["payload"]) if row is not None else None

    def analysis_put(self, game_id: str, ply: int, bot_id: int, payload: bytes) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO analysis_cache (game_id, ply, bot_id, payload)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT (game_id, ply, bot_id) DO UPDATE SET payload = excluded.payload",
                (game_id, ply, bot_id, payload),
            )
            self._conn.commit()

    # -- stats ----------------------------------------------------------------

    def bot_stats(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM v_bot_stats ORDER BY visits, epoch"
            ).fetchall()
        return [dict(r) for r in rows]

    def daily(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM v_daily ORDER BY day DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def hall_of_fame(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM v_hall_of_fame LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
