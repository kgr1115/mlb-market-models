"""
SQLite cache for odds snapshots.

Every successful scrape writes one row per (book, event) into
`odds_snapshots`. We never dedupe on polled_at — a row per poll is the
whole point, since that's how we reconstruct line-movement history and
compute opener / current / steam signals later.

Schema is intentionally tiny. If this grows, move to a real ORM.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .odds_models import OddsSnapshot, OddsBook


DEFAULT_CACHE_PATH = Path(
    os.environ.get("BBP_CACHE_PATH", "bbp_cache.sqlite")
).resolve()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    book            TEXT NOT NULL,
    event_id        TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    game_time_utc   TEXT NOT NULL,
    home_ml         INTEGER,
    away_ml         INTEGER,
    home_rl_line    REAL,
    home_rl_odds    INTEGER,
    away_rl_odds    INTEGER,
    total_line      REAL,
    over_odds       INTEGER,
    under_odds      INTEGER,
    polled_at_utc   TEXT NOT NULL,
    native_event_id TEXT,
    PRIMARY KEY (book, event_id, polled_at_utc)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_event
    ON odds_snapshots(event_id, polled_at_utc);

CREATE INDEX IF NOT EXISTS idx_snapshots_book_event
    ON odds_snapshots(book, event_id, polled_at_utc);

CREATE INDEX IF NOT EXISTS idx_snapshots_game_time
    ON odds_snapshots(game_time_utc);
"""


class OddsCache:
    """Thin wrapper over sqlite3 for persisting and querying snapshots."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_CACHE_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ----- writes ---------------------------------------------------------

    def insert(self, snapshot: OddsSnapshot) -> None:
        self.insert_many([snapshot])

    def insert_many(self, snapshots: Iterable[OddsSnapshot]) -> int:
        rows = [s.to_row() for s in snapshots]
        if not rows:
            return 0
        with self._conn() as c:
            # INSERT OR IGNORE: if someone double-polls and the (book,
            # event_id, polled_at_utc) key collides, the later row wins
            # only if its polled_at_utc differs (which it should, since
            # polled_at_utc is set at scrape time with microsecond precision).
            c.executemany(
                """
                INSERT OR REPLACE INTO odds_snapshots
                (book, event_id, home_team, away_team, game_time_utc,
                 home_ml, away_ml, home_rl_line, home_rl_odds, away_rl_odds,
                 total_line, over_odds, under_odds, polled_at_utc, native_event_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    # ----- reads ----------------------------------------------------------

    def latest(self, book: OddsBook, event_id: str) -> Optional[OddsSnapshot]:
        """Most recent snapshot for one book+event. None if we've never polled it."""
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT book, event_id, home_team, away_team, game_time_utc,
                       home_ml, away_ml, home_rl_line, home_rl_odds, away_rl_odds,
                       total_line, over_odds, under_odds, polled_at_utc, native_event_id
                FROM odds_snapshots
                WHERE book = ? AND event_id = ?
                ORDER BY polled_at_utc DESC
                LIMIT 1
                """,
                (book.value, event_id),
            )
            row = cur.fetchone()
        return OddsSnapshot.from_row(row) if row else None

    def opener(self, book: OddsBook, event_id: str) -> Optional[OddsSnapshot]:
        """Earliest snapshot for one book+event — our 'opener' proxy."""
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT book, event_id, home_team, away_team, game_time_utc,
                       home_ml, away_ml, home_rl_line, home_rl_odds, away_rl_odds,
                       total_line, over_odds, under_odds, polled_at_utc, native_event_id
                FROM odds_snapshots
                WHERE book = ? AND event_id = ?
                ORDER BY polled_at_utc ASC
                LIMIT 1
                """,
                (book.value, event_id),
            )
            row = cur.fetchone()
        return OddsSnapshot.from_row(row) if row else None

    def history(self, book: OddsBook, event_id: str) -> list[OddsSnapshot]:
        """All snapshots for one book+event, oldest first."""
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT book, event_id, home_team, away_team, game_time_utc,
                       home_ml, away_ml, home_rl_line, home_rl_odds, away_rl_odds,
                       total_line, over_odds, under_odds, polled_at_utc, native_event_id
                FROM odds_snapshots
                WHERE book = ? AND event_id = ?
                ORDER BY polled_at_utc ASC
                """,
                (book.value, event_id),
            )
            return [OddsSnapshot.from_row(r) for r in cur.fetchall()]

    def events_on_date(self, date_utc_prefix: str) -> list[str]:
        """All distinct event_ids whose game_time_utc starts with a given date.

        date_utc_prefix: "YYYY-MM-DD"
        """
        like = f"{date_utc_prefix}%"
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT DISTINCT event_id
                FROM odds_snapshots
                WHERE game_time_utc LIKE ?
                ORDER BY game_time_utc ASC
                """,
                (like,),
            )
            return [r[0] for r in cur.fetchall()]

    # ----- diagnostics ----------------------------------------------------

    def stats(self) -> dict:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]
            by_book = dict(
                c.execute(
                    "SELECT book, COUNT(*) FROM odds_snapshots GROUP BY book"
                ).fetchall()
            )
            latest_poll = c.execute(
                "SELECT MAX(polled_at_utc) FROM odds_snapshots"
            ).fetchone()[0]
        return {
            "path": str(self.path),
            "total_snapshots": total,
            "by_book": by_book,
            "latest_poll_utc": latest_poll,
        }
