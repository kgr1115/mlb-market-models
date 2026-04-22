"""
Line shopping across all books in the odds cache.

Rather than always preferring Pinnacle (sharp) or DraftKings (default),
this module scans the latest snapshot from every book for an event and
returns the *best available* price on each side. The EV on any given bet
is only as good as the price you actually take — so if FanDuel is +105
on Away ML while DK is -105, the real shop price is FanDuel.

Outputs
-------
- BestLine  : one book's winning quote on one side, with `odds` and
              (for spread/total) `line`
- ShoppedMarket : all six sides (home/away ML, home/away RL, over/under)
                  with best-line references, plus a devig and
                  two-book arbitrage test.

Why this matters
----------------
1) Raw EV. A 10-cent shop improvement on a 50% bet is +5% EV per unit.
2) Implied-prob floor. Using the *worst* implied prob from all books
   (the true "fair" anchor the shop can offer) is a more honest
   de-vigged probability than using a single book's vig-loaded quote.
3) Arbitrage / middling. When two books diverge enough, a middle or
   arb presents itself. We surface those as shop-level opportunities.

The module is read-only with respect to the cache; it does not poll.
Call `fetch_all_books(cache)` elsewhere, then hand the cache in here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from .odds_cache import OddsCache
from .odds_models import OddsBook, OddsSnapshot


# =============================================================================
# Best-line container
# =============================================================================

@dataclass
class BestLine:
    """The book with the best available price on one side.

    For moneyline: higher American odds = better payout to bettor.
    For totals/run-line: the bettor picks the combination of LINE and
    ODDS that maximizes expected return. We treat the *odds* as the tie
    breaker when lines match, and prefer the friendlier line for the
    side direction otherwise (over wants lower total, under wants higher).
    """
    side: str           # "home_ml", "away_ml", "home_rl", "away_rl", "over", "under"
    book: OddsBook
    odds: int
    line: Optional[float] = None     # None for moneyline
    polled_at_utc: Optional[str] = None

    def american(self) -> int:
        return self.odds


# =============================================================================
# Shopped market
# =============================================================================

@dataclass
class ShoppedMarket:
    """All six sides' best-available prices for one event."""
    event_id: str
    home_team: str
    away_team: str

    home_ml: Optional[BestLine] = None
    away_ml: Optional[BestLine] = None

    home_rl: Optional[BestLine] = None
    away_rl: Optional[BestLine] = None

    over: Optional[BestLine] = None
    under: Optional[BestLine] = None

    # Snapshots used, keyed by book.value, for provenance
    snapshots_by_book: dict[str, OddsSnapshot] = field(default_factory=dict)

    # ----- arbitrage helpers --------------------------------------------------

    def ml_shop_vig(self) -> Optional[float]:
        """Sum of implied probs across the shopped sides. <1.0 = arb."""
        if not self.home_ml or not self.away_ml:
            return None
        return _imp(self.home_ml.odds) + _imp(self.away_ml.odds)

    def total_shop_vig(self) -> Optional[float]:
        if not self.over or not self.under:
            return None
        # Only meaningful if books agree on the LINE
        if self.over.line != self.under.line:
            return None
        return _imp(self.over.odds) + _imp(self.under.odds)

    def rl_shop_vig(self) -> Optional[float]:
        if not self.home_rl or not self.away_rl:
            return None
        if self.home_rl.line is not None and self.away_rl.line is not None:
            # For RL: home_rl_line and away_rl_line should sum to 0 (e.g. -1.5/+1.5)
            # When lines differ the "shop" is a middle, not an arb on the same line.
            if self.home_rl.line + self.away_rl.line != 0:
                return None
        return _imp(self.home_rl.odds) + _imp(self.away_rl.odds)

    def has_arbitrage(self) -> list[str]:
        """Markets where sum of best-side implied prob < 1 (risk-free arb)."""
        out = []
        for name, v in (
            ("moneyline", self.ml_shop_vig()),
            ("totals", self.total_shop_vig()),
            ("run_line", self.rl_shop_vig()),
        ):
            if v is not None and v < 1.0:
                out.append(name)
        return out

    def has_middle(self) -> list[str]:
        """Totals/run-line middles: two books' lines differ by >= 0.5."""
        out = []
        if self.over and self.under and self.over.line is not None and self.under.line is not None:
            if self.over.line < self.under.line:
                # Can take OVER at low line and UNDER at high line → middle window
                out.append(f"totals ({self.over.line} / {self.under.line})")
        return out

    def to_dict(self) -> dict:
        def _bl(b: Optional[BestLine]):
            if b is None:
                return None
            return {
                "side": b.side, "book": b.book.value,
                "odds": b.odds, "line": b.line,
            }
        return {
            "event_id": self.event_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_ml": _bl(self.home_ml),
            "away_ml": _bl(self.away_ml),
            "home_rl": _bl(self.home_rl),
            "away_rl": _bl(self.away_rl),
            "over": _bl(self.over),
            "under": _bl(self.under),
            "arb_markets": self.has_arbitrage(),
            "middle_markets": self.has_middle(),
        }


# =============================================================================
# Public API
# =============================================================================

ALL_BOOKS: tuple[OddsBook, ...] = (
    OddsBook.DRAFTKINGS, OddsBook.FANDUEL,
)


def shop_event(cache: OddsCache, event_id: str,
               books: Iterable[OddsBook] = ALL_BOOKS) -> Optional[ShoppedMarket]:
    """Return the best price on every side for one event."""
    snaps: dict[str, OddsSnapshot] = {}
    for b in books:
        s = cache.latest(b, event_id)
        if s is not None:
            snaps[b.value] = s
    if not snaps:
        return None

    first = next(iter(snaps.values()))
    sm = ShoppedMarket(
        event_id=event_id,
        home_team=first.home_team,
        away_team=first.away_team,
        snapshots_by_book=snaps,
    )

    for s in snaps.values():
        _update_best_ml(sm, s)
        _update_best_rl(sm, s)
        _update_best_total(sm, s)

    return sm


def shop_all(cache: OddsCache, date_utc_prefix: str,
             books: Iterable[OddsBook] = ALL_BOOKS) -> list[ShoppedMarket]:
    """Shop every event on the given UTC date."""
    out: list[ShoppedMarket] = []
    for ev in cache.events_on_date(date_utc_prefix):
        sm = shop_event(cache, ev, books)
        if sm is not None:
            out.append(sm)
    return out


# =============================================================================
# Selecting the best side from one snapshot
# =============================================================================

def _update_best_ml(sm: ShoppedMarket, s: OddsSnapshot) -> None:
    if s.home_ml is not None:
        new = BestLine("home_ml", OddsBook(s.book), s.home_ml,
                       polled_at_utc=s.polled_at_utc.isoformat())
        sm.home_ml = _better(sm.home_ml, new)
    if s.away_ml is not None:
        new = BestLine("away_ml", OddsBook(s.book), s.away_ml,
                       polled_at_utc=s.polled_at_utc.isoformat())
        sm.away_ml = _better(sm.away_ml, new)


def _update_best_rl(sm: ShoppedMarket, s: OddsSnapshot) -> None:
    if s.home_rl_odds is None or s.away_rl_odds is None:
        return
    home = BestLine("home_rl", OddsBook(s.book), s.home_rl_odds,
                    line=s.home_rl_line,
                    polled_at_utc=s.polled_at_utc.isoformat())
    away_line = -s.home_rl_line if s.home_rl_line is not None else None
    away = BestLine("away_rl", OddsBook(s.book), s.away_rl_odds,
                    line=away_line,
                    polled_at_utc=s.polled_at_utc.isoformat())
    # For run line we want (a) the friendlier LINE for the side, then (b) best odds.
    # Friendlier line for home_rl: a HIGHER line (favorite would rather be -1.0
    # than -1.5; underdog would rather be +1.5 than +1.0).
    sm.home_rl = _better_rl(sm.home_rl, home, side_is_home=True)
    sm.away_rl = _better_rl(sm.away_rl, away, side_is_home=False)


def _update_best_total(sm: ShoppedMarket, s: OddsSnapshot) -> None:
    if s.total_line is None:
        return
    if s.over_odds is not None:
        new = BestLine("over", OddsBook(s.book), s.over_odds, line=s.total_line,
                       polled_at_utc=s.polled_at_utc.isoformat())
        # For OVER the friendlier line is LOWER (easier to go over 7.5 than 8.5)
        sm.over = _better_total(sm.over, new, prefer_lower=True)
    if s.under_odds is not None:
        new = BestLine("under", OddsBook(s.book), s.under_odds, line=s.total_line,
                       polled_at_utc=s.polled_at_utc.isoformat())
        # For UNDER the friendlier line is HIGHER
        sm.under = _better_total(sm.under, new, prefer_lower=False)


def _better(old: Optional[BestLine], new: BestLine) -> BestLine:
    """Pick the best American-odds price (higher = better for bettor)."""
    if old is None:
        return new
    return new if new.odds > old.odds else old


def _better_rl(old: Optional[BestLine], new: BestLine, side_is_home: bool) -> BestLine:
    """For the run line we evaluate expected value on the LINE first."""
    if old is None:
        return new
    # If lines differ pick the friendlier line for this side.
    # Home's line is negative when favorite. Higher (closer to 0 or positive)
    # is friendlier for home regardless of fav/dog.
    if old.line is None or new.line is None or old.line == new.line:
        return new if new.odds > old.odds else old
    # Compare by effective cover probability. We approximate: a higher home-RL
    # line means home covers more easily.
    better_line = new if new.line > old.line else old
    worse_line = old if better_line is new else new
    # Only swap to better_line if its odds are within 25 cents of worse_line's.
    # A 10-cent worse price on a friendlier line is still a shop win.
    if _cents_between(better_line.odds, worse_line.odds) <= 25:
        return better_line
    return worse_line


def _better_total(old: Optional[BestLine], new: BestLine,
                  prefer_lower: bool) -> BestLine:
    if old is None:
        return new
    if old.line is None or new.line is None or old.line == new.line:
        return new if new.odds > old.odds else old
    better_line = (new if (prefer_lower and new.line < old.line) or
                          (not prefer_lower and new.line > old.line) else old)
    worse_line = old if better_line is new else new
    if _cents_between(better_line.odds, worse_line.odds) <= 20:
        return better_line
    return worse_line


def _cents_between(a: int, b: int) -> int:
    """Absolute difference in American-odds 'cents' (-110 to +110 = 220)."""
    return abs(a - b)


def _imp(american: int) -> float:
    if american < 0:
        return (-american) / ((-american) + 100.0)
    return 100.0 / (american + 100.0)
