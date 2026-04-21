"""
Shared dataclasses for the odds data layer.

An OddsSnapshot is one point-in-time price set from one book for one game.
The scrapers normalize their book-specific JSON into this shape, and the
cache persists it. Downstream code (MarketData builder) reconstructs
line-movement history by querying multiple snapshots per (book, event).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class OddsBook(str, Enum):
    PINNACLE = "pinnacle"
    DRAFTKINGS = "draftkings"
    FANDUEL = "fanduel"


class Market(str, Enum):
    MONEYLINE = "moneyline"
    RUN_LINE = "run_line"
    TOTALS = "totals"


@dataclass
class OddsSnapshot:
    """One snapshot of all three markets for one event from one book."""
    book: OddsBook
    # event_id is a stable key we compute ourselves: "YYYY-MM-DD|AWAY|HOME"
    # so Pinnacle and DraftKings agree on it even though their native IDs differ
    event_id: str
    home_team: str          # normalized (see team_names.py)
    away_team: str          # normalized
    game_time_utc: datetime

    # Moneyline (american odds). None if book hasn't posted.
    home_ml: Optional[int] = None
    away_ml: Optional[int] = None

    # Run line (standard -1.5 / +1.5). home_rl_line is the line laid by home;
    # positive when home is the underdog, negative when home is favorite.
    home_rl_line: Optional[float] = None   # typically -1.5 or +1.5
    home_rl_odds: Optional[int] = None
    away_rl_odds: Optional[int] = None

    # Totals
    total_line: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None

    polled_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional: book-native event id, useful for debugging / refetch
    native_event_id: Optional[str] = None

    def to_row(self) -> tuple:
        """Flat tuple in column order for the SQLite odds_snapshots table."""
        return (
            self.book.value,
            self.event_id,
            self.home_team,
            self.away_team,
            self.game_time_utc.isoformat(),
            self.home_ml,
            self.away_ml,
            self.home_rl_line,
            self.home_rl_odds,
            self.away_rl_odds,
            self.total_line,
            self.over_odds,
            self.under_odds,
            self.polled_at_utc.isoformat(),
            self.native_event_id,
        )

    @classmethod
    def from_row(cls, row: tuple) -> "OddsSnapshot":
        (book, event_id, home, away, gt, hml, aml, rl_line, hrl, arl,
         tl, oo, uo, polled, native) = row
        return cls(
            book=OddsBook(book),
            event_id=event_id,
            home_team=home,
            away_team=away,
            game_time_utc=datetime.fromisoformat(gt),
            home_ml=hml,
            away_ml=aml,
            home_rl_line=rl_line,
            home_rl_odds=hrl,
            away_rl_odds=arl,
            total_line=tl,
            over_odds=oo,
            under_odds=uo,
            polled_at_utc=datetime.fromisoformat(polled),
            native_event_id=native,
        )


def make_event_id(game_time_utc: datetime, away_team: str, home_team: str) -> str:
    """Deterministic cross-book event key.

    We use the local (US/Eastern would be more correct, but UTC date is
    stable enough for the short lookup windows we care about) date plus
    normalized team names. All scrapers must call this so Pinnacle and
    DraftKings snapshots for the same game collide on the same event_id.
    """
    date_key = game_time_utc.strftime("%Y-%m-%d")
    return f"{date_key}|{away_team}|{home_team}"
