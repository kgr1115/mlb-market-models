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


def pick_main_run_line(
    home_candidates: list[tuple[float, int]],
    away_candidates: list[tuple[float, int]],
    home_ml: Optional[int],
    away_ml: Optional[int],
) -> tuple[Optional[float], Optional[int], Optional[int]]:
    """Select the MAIN baseball Run Line from a book's candidate selections.

    Some books publish both the main RL (favorite -1.5 / dog +1.5) and a
    "reverse" RL (favorite +1.5 / dog -1.5) under the same market type.
    Simple "last one wins" parsing can latch onto the reverse line and
    invert the UI — e.g. showing the ML favorite as a +1.5 -189 price when
    they should be at -1.5 +180.

    Each candidate is (handicap, american_price). Handicap is +1.5 or -1.5.
    This helper pairs them up (handicaps must sum to ~0 across home/away)
    and picks the pair that:
      (a) matches the ML favorite direction — ML fav's RL line should be
          -1.5 with *positive* odds, and
      (b) if ML is unavailable / pick'em, falls back to juice-sign: in the
          MAIN run line the side at -1.5 has POSITIVE odds (harder bet
          → better payout) and the side at +1.5 has NEGATIVE odds.

    Returns (home_rl_line, home_rl_odds, away_rl_odds) for the chosen pair,
    or (None, None, None) if no clean pair can be found.
    """
    if not home_candidates or not away_candidates:
        return None, None, None

    # Build valid pairs: home (h_h, h_p) + away (a_h, a_p) where handicaps
    # mirror each other within a small epsilon.
    pairs: list[tuple[float, int, int]] = []  # (home_h, home_p, away_p)
    for h_h, h_p in home_candidates:
        for a_h, a_p in away_candidates:
            if abs(h_h + a_h) < 0.01:
                pairs.append((h_h, h_p, a_p))

    if not pairs:
        return None, None, None

    # Determine ML-favorite direction.
    ml_fav: Optional[str] = None  # "home" | "away" | None
    if home_ml is not None and away_ml is not None:
        if home_ml < away_ml:
            ml_fav = "home"
        elif away_ml < home_ml:
            ml_fav = "away"

    # Preferred pair: ML favorite is the -1.5 side AND gets positive odds.
    if ml_fav == "home":
        for h_h, h_p, a_p in pairs:
            if h_h < 0 and h_p > 0:
                return h_h, h_p, a_p
    elif ml_fav == "away":
        for h_h, h_p, a_p in pairs:
            # Away is the -1.5 side when home_h > 0.
            if h_h > 0 and a_p > 0:
                return h_h, h_p, a_p

    # Fallback (no ML or no juice-aligned candidate): pick the pair whose
    # juice matches the MAIN-line pattern: -1.5 side has positive odds
    # and +1.5 side has negative odds.
    for h_h, h_p, a_p in pairs:
        if h_h < 0 and h_p > 0 and a_p < 0:
            return h_h, h_p, a_p
        if h_h > 0 and h_p < 0 and a_p > 0:
            return h_h, h_p, a_p

    # Last resort: return the first pair as-is.
    h_h, h_p, a_p = pairs[0]
    return h_h, h_p, a_p


def make_event_id(game_time_utc: datetime, away_team: str, home_team: str) -> str:
    """Deterministic cross-book event key.

    We use the local (US/Eastern would be more correct, but UTC date is
    stable enough for the short lookup windows we care about) date plus
    normalized team names. All scrapers must call this so Pinnacle and
    DraftKings snapshots for the same game collide on the same event_id.
    """
    date_key = game_time_utc.strftime("%Y-%m-%d")
    return f"{date_key}|{away_team}|{home_team}"
