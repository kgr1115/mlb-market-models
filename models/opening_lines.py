"""
Opening-line collector.

We need (opener, closer) pairs to train the closing-line model. Books
post openers at wildly varying times: Pinnacle typically posts 24-48h
pre-game, DraftKings often posts 8-10h pre-game, Caesars posts at
random times. The first time we see a line for a given (book, event)
becomes our "opener" snapshot.

This module just layers a convenience table on top of the existing
odds cache — it doesn't poll. The polling loop in `fetch_all_books`
(data/odds_client.py) already persists every snapshot; the first
snapshot per (book, event) naturally serves as the opener, and
`OddsCache.opener()` retrieves it.

What this module adds is:
    (a) a dedicated `opening_lines` table that pins the *earliest*
        snapshot we've ever seen for each (book, event), so even if the
        odds_snapshots table is later pruned, we preserve the opener;
    (b) helpers for the training pipeline to enumerate (opener, closer)
        pairs.

Schema
------
CREATE TABLE opening_lines (
    event_id TEXT NOT NULL,
    book TEXT NOT NULL,
    home_team TEXT,
    away_team TEXT,
    game_time_utc TEXT,
    home_ml INTEGER, away_ml INTEGER,
    home_rl_line REAL, home_rl_odds INTEGER, away_rl_odds INTEGER,
    total_line REAL, over_odds INTEGER, under_odds INTEGER,
    observed_at_utc TEXT NOT NULL,
    PRIMARY KEY (event_id, book)
);
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from data.odds_cache import OddsCache, DEFAULT_CACHE_PATH
from data.odds_models import OddsBook, OddsSnapshot


_SCHEMA = """
CREATE TABLE IF NOT EXISTS opening_lines (
    event_id        TEXT NOT NULL,
    book            TEXT NOT NULL,
    home_team       TEXT,
    away_team       TEXT,
    game_time_utc   TEXT,
    home_ml         INTEGER,
    away_ml         INTEGER,
    home_rl_line    REAL,
    home_rl_odds    INTEGER,
    away_rl_odds    INTEGER,
    total_line      REAL,
    over_odds       INTEGER,
    under_odds      INTEGER,
    observed_at_utc TEXT NOT NULL,
    PRIMARY KEY (event_id, book)
);

CREATE INDEX IF NOT EXISTS idx_openers_event
    ON opening_lines(event_id);
"""


class OpeningLineStore:
    """Durable store for the earliest snapshot per (book, event).

    Safe to call `record_opener()` every poll — it's a no-op if an
    opener already exists for that (book, event).
    """

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

    def record_opener(self, snap: OddsSnapshot) -> bool:
        """Insert the opener for (book, event) if one isn't already recorded.
        Returns True on a new opener insertion, False if it already existed.
        """
        with self._conn() as c:
            cur = c.execute(
                "SELECT 1 FROM opening_lines WHERE event_id = ? AND book = ?",
                (snap.event_id, snap.book.value),
            )
            if cur.fetchone() is not None:
                return False
            c.execute(
                """
                INSERT INTO opening_lines (
                    event_id, book, home_team, away_team, game_time_utc,
                    home_ml, away_ml, home_rl_line, home_rl_odds, away_rl_odds,
                    total_line, over_odds, under_odds, observed_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snap.event_id, snap.book.value, snap.home_team,
                    snap.away_team, snap.game_time_utc.isoformat(),
                    snap.home_ml, snap.away_ml, snap.home_rl_line,
                    snap.home_rl_odds, snap.away_rl_odds,
                    snap.total_line, snap.over_odds, snap.under_odds,
                    snap.polled_at_utc.isoformat(),
                ),
            )
        return True

    def get_opener(self, event_id: str,
                   book: OddsBook = OddsBook.PINNACLE) -> Optional[dict]:
        """Fetch the stored opener (as a dict) for one (book, event)."""
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT event_id, book, home_team, away_team, game_time_utc,
                       home_ml, away_ml, home_rl_line, home_rl_odds, away_rl_odds,
                       total_line, over_odds, under_odds, observed_at_utc
                FROM opening_lines
                WHERE event_id = ? AND book = ?
                """,
                (event_id, book.value),
            )
            row = cur.fetchone()
        if row is None:
            return None
        keys = ("event_id", "book", "home_team", "away_team", "game_time_utc",
                "home_ml", "away_ml", "home_rl_line", "home_rl_odds", "away_rl_odds",
                "total_line", "over_odds", "under_odds", "observed_at_utc")
        return dict(zip(keys, row))

    def get_all_openers_today(self, date_utc_prefix: Optional[str] = None
                               ) -> list[dict]:
        if date_utc_prefix is None:
            date_utc_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        like = f"{date_utc_prefix}%"
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT event_id, book, home_team, away_team, game_time_utc,
                       home_ml, away_ml, home_rl_line, home_rl_odds, away_rl_odds,
                       total_line, over_odds, under_odds, observed_at_utc
                FROM opening_lines
                WHERE game_time_utc LIKE ?
                ORDER BY game_time_utc, book
                """,
                (like,),
            )
            rows = cur.fetchall()
        keys = ("event_id", "book", "home_team", "away_team", "game_time_utc",
                "home_ml", "away_ml", "home_rl_line", "home_rl_odds", "away_rl_odds",
                "total_line", "over_odds", "under_odds", "observed_at_utc")
        return [dict(zip(keys, r)) for r in rows]

    def stats(self) -> dict:
        with self._conn() as c:
            total = c.execute("SELECT COUNT(*) FROM opening_lines").fetchone()[0]
            by_book = dict(
                c.execute(
                    "SELECT book, COUNT(*) FROM opening_lines GROUP BY book"
                ).fetchall()
            )
        return {
            "path": str(self.path),
            "total_openers": total,
            "by_book": by_book,
        }


# =============================================================================
# Module-level conveniences
# =============================================================================

_default_store: Optional[OpeningLineStore] = None


def _store() -> OpeningLineStore:
    global _default_store
    if _default_store is None:
        _default_store = OpeningLineStore()
    return _default_store


def record_opener(snap: OddsSnapshot) -> bool:
    """Idempotently pin the first-seen snapshot for (book, event) as
    the opener. Called from data.odds_client.fetch_all_books after
    each poll."""
    return _store().record_opener(snap)


def get_opener(event_id: str, book: OddsBook = OddsBook.PINNACLE) -> Optional[dict]:
    return _store().get_opener(event_id, book)


def get_all_openers_today(date_utc_prefix: Optional[str] = None) -> list[dict]:
    return _store().get_all_openers_today(date_utc_prefix)
