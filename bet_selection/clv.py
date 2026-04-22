"""
Closing Line Value (CLV) tracking.

CLV is the canonical profitability proxy used by professional sports
bettors: on a long timeline, CLV in basis points ≈ ROI (±vig). A bettor
who consistently beats the close is +EV regardless of short-run variance
in outcomes.

Two flavors:

1. **Price-based CLV** — compare the American odds you got vs the closing
   odds on the same side. Simple and intuitive but conflates vig changes
   with real line movement.

2. **Fair-probability CLV** — de-vig both the bet-time market and the
   closing market (using the full two-way pair), then take the delta in
   fair probability of the side you bet. This is what sharps track
   because it isolates the true probability shift.

We implement both; fair-probability is the default.

Persistence is a lightweight SQLite table in the same DB the odds-cache
uses (bbp_cache.sqlite), keyed by (event_id, market, side, placed_ts).
The tracker is designed so a live worker can call `record_bet(...)` at
placement time and `finalize(...)` once the market closes (or at first
pitch when no post-close snapshot is available).
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# -----------------------------------------------------------------------------
# Core math
# -----------------------------------------------------------------------------

def _american_to_prob(american: int) -> float:
    if american < 0:
        return (-american) / (-american + 100.0)
    return 100.0 / (american + 100.0)


def _devig_two_way(a: int, b: int) -> tuple[float, float]:
    pa = _american_to_prob(a)
    pb = _american_to_prob(b)
    s = pa + pb
    if s <= 0:
        return 0.5, 0.5
    return pa / s, pb / s


def price_clv_cents(bet_odds: int, close_odds: int) -> float:
    """CLV expressed in cents of American odds improvement.

    Positive = you got better odds than the close (beat the market).
    E.g. bet -105, closed at -120 ⇒ +15 cents CLV.
    """
    return float(close_odds - bet_odds)


def fair_prob_clv_bps(
    bet_side_odds: int,
    bet_paired_odds: int,
    close_side_odds: int,
    close_paired_odds: int,
) -> float:
    """CLV in basis points of fair-probability improvement on your side.

    Positive = the market's fair probability on your side was higher at
    close than when you bet ⇒ you beat the market. Over a large sample
    the mean CLV (in bps) ≈ ROI in bps, minus vig.

    Arguments are the two-way pair at bet time and the two-way pair at
    close. "paired" is the other side of the same market (e.g. bet on
    HOME ML: bet_side = home_ml, bet_paired = away_ml).
    """
    p_bet, _ = _devig_two_way(bet_side_odds, bet_paired_odds)
    p_close, _ = _devig_two_way(close_side_odds, close_paired_odds)
    return (p_close - p_bet) * 10000.0


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS clv_bets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT NOT NULL,
    market      TEXT NOT NULL,          -- moneyline | run_line | totals
    side        TEXT NOT NULL,          -- home | away | over | under
    placed_ts   TEXT NOT NULL,          -- ISO 8601 UTC
    bet_odds    INTEGER NOT NULL,
    paired_odds INTEGER NOT NULL,
    book        TEXT,                   -- draftkings | fanduel | ...
    model_prob  REAL,
    confidence  REAL,
    close_ts    TEXT,
    close_odds  INTEGER,
    close_paired INTEGER,
    clv_cents   REAL,
    clv_bps     REAL,
    won         INTEGER                 -- NULL until graded; 1/0/-1 (push)
);
CREATE INDEX IF NOT EXISTS idx_clv_event ON clv_bets(event_id);
CREATE INDEX IF NOT EXISTS idx_clv_open  ON clv_bets(close_ts)
    WHERE close_ts IS NULL;
"""


@dataclass
class CLVRow:
    event_id: str
    market: str
    side: str
    bet_odds: int
    paired_odds: int
    placed_ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    book: Optional[str] = None
    model_prob: Optional[float] = None
    confidence: Optional[float] = None
    close_odds: Optional[int] = None
    close_paired: Optional[int] = None
    close_ts: Optional[str] = None
    clv_cents: Optional[float] = None
    clv_bps: Optional[float] = None
    won: Optional[int] = None


class CLVStore:
    """Persistent CLV ledger — same SQLite file as the odds cache.

    Not thread-safe; call from a single worker (the live poll loop).
    """
    def __init__(self, db_path: Optional[str | Path] = None):
        if db_path is None:
            db_path = Path(__file__).resolve().parent.parent / "bbp_cache.sqlite"
        self.db_path = str(db_path)
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def record_bet(self, row: CLVRow) -> int:
        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO clv_bets
                  (event_id, market, side, placed_ts, bet_odds, paired_odds,
                   book, model_prob, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (row.event_id, row.market, row.side, row.placed_ts,
                 row.bet_odds, row.paired_odds, row.book,
                 row.model_prob, row.confidence),
            )
            return int(cur.lastrowid)

    def finalize(self, bet_id: int,
                 close_odds: int, close_paired: int,
                 won: Optional[int] = None) -> None:
        """Record closing odds + (optional) final grade for a placed bet."""
        with self._conn() as c:
            row = c.execute(
                "SELECT bet_odds, paired_odds FROM clv_bets WHERE id = ?",
                (bet_id,),
            ).fetchone()
            if row is None:
                return
            clv_cents = price_clv_cents(int(row["bet_odds"]), int(close_odds))
            clv_bps = fair_prob_clv_bps(
                int(row["bet_odds"]), int(row["paired_odds"]),
                int(close_odds), int(close_paired),
            )
            c.execute(
                """
                UPDATE clv_bets
                   SET close_odds=?, close_paired=?, close_ts=?,
                       clv_cents=?, clv_bps=?, won=COALESCE(?, won)
                 WHERE id=?
                """,
                (close_odds, close_paired,
                 datetime.now(timezone.utc).isoformat(),
                 clv_cents, clv_bps, won, bet_id),
            )

    def pending(self) -> list[sqlite3.Row]:
        """Bets that still need a closing snapshot."""
        with self._conn() as c:
            return c.execute(
                "SELECT * FROM clv_bets WHERE close_ts IS NULL"
            ).fetchall()

    def summary(self) -> dict:
        """Aggregate CLV stats over every finalized bet."""
        with self._conn() as c:
            r = c.execute(
                """
                SELECT COUNT(*)           AS n,
                       AVG(clv_cents)     AS avg_cents,
                       AVG(clv_bps)       AS avg_bps,
                       AVG(CASE WHEN clv_bps > 0 THEN 1.0 ELSE 0.0 END) AS beat_pct
                  FROM clv_bets
                 WHERE clv_bps IS NOT NULL
                """
            ).fetchone()
            return dict(r) if r else {}


# -----------------------------------------------------------------------------
# Convenience constructors — normalize callers from the predictor side
# -----------------------------------------------------------------------------

def row_from_prediction(event_id: str, market: str, pick: str,
                         model_prob: float, confidence: float,
                         market_data, book: Optional[str] = None) -> Optional[CLVRow]:
    """Build a CLVRow from a PredictionResult + MarketData.

    Returns None if the pick is NO BET or the odds pair is missing.
    """
    if not pick or pick.startswith("NO BET"):
        return None
    m = market
    side = None
    odds = paired = None
    if m == "moneyline":
        if pick.startswith("HOME"):
            side, odds, paired = "home", market_data.home_ml_odds, market_data.away_ml_odds
        elif pick.startswith("AWAY"):
            side, odds, paired = "away", market_data.away_ml_odds, market_data.home_ml_odds
    elif m == "run_line":
        if pick.startswith("HOME"):
            side, odds, paired = "home", market_data.home_rl_odds, market_data.away_rl_odds
        elif pick.startswith("AWAY"):
            side, odds, paired = "away", market_data.away_rl_odds, market_data.home_rl_odds
    elif m == "totals":
        if pick.startswith("OVER"):
            side, odds, paired = "over", market_data.over_odds, market_data.under_odds
        elif pick.startswith("UNDER"):
            side, odds, paired = "under", market_data.under_odds, market_data.over_odds
    if side is None or odds is None or paired is None:
        return None
    return CLVRow(
        event_id=event_id,
        market=m,
        side=side,
        bet_odds=int(odds),
        paired_odds=int(paired),
        book=book,
        model_prob=float(model_prob),
        confidence=float(confidence),
    )
