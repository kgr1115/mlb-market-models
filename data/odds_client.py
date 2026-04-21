"""
Unified odds client.

This is the entry point the rest of the app uses:

    from data import fetch_all_books, build_market_data

    fetch_all_books(cache)                       # polls + persists
    md = build_market_data(cache, "2026-04-19|...|...")  # -> predictors.MarketData

Pinnacle is preferred for current pricing (it's the sharp line); the
opener fields are filled from the earliest snapshot we have on file for
that event — meaning the longer the cache runs, the better the opener
approximation becomes. On first-ever poll, opener == current.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from predictors import MarketData

from .odds_cache import OddsCache
from .odds_draftkings import fetch_draftkings_snapshots
from .odds_fanduel import fetch_fanduel_snapshots
from .odds_models import OddsBook, OddsSnapshot
from .odds_pinnacle import fetch_pinnacle_snapshots
from .odds_theoddsapi import fetch_dk_via_theoddsapi_cached
from .line_shop import ShoppedMarket, shop_event

log = logging.getLogger(__name__)


def _fetch_draftkings() -> list[OddsSnapshot]:
    """DK source with the-odds-api preferred, Playwright as fallback.

    Tries the-odds-api first (with its internal 5-windows-per-day cache,
    so most calls don't hit the network). Falls back to the Playwright
    scraper only if the API returns nothing — keeps DK lines flowing
    when the API key is missing, the monthly credit budget is exhausted,
    or the the-odds-api service itself is down.
    """
    snaps = fetch_dk_via_theoddsapi_cached()
    if snaps:
        return snaps
    log.info("DK: the-odds-api returned 0 snaps; falling back to Playwright scraper")
    return fetch_draftkings_snapshots()


def fetch_all_books(cache: Optional[OddsCache] = None) -> dict[str, int]:
    """Poll every configured book once and persist to cache.

    Returns a dict of {book_name: snapshots_written}. Never raises — a
    book that 500s simply contributes 0 rows.
    """
    if cache is None:
        cache = OddsCache()

    counts: dict[str, int] = {}
    for name, fetcher in (
        ("pinnacle", fetch_pinnacle_snapshots),
        ("draftkings", _fetch_draftkings),
        ("fanduel", fetch_fanduel_snapshots),
    ):
        try:
            snaps = fetcher()
        except Exception as e:
            log.exception("Scraper %s crashed: %s", name, e)
            snaps = []
        n = cache.insert_many(snaps)
        counts[name] = n
        log.info("Wrote %d %s snapshots", n, name)
        try:
            from models.opening_lines import record_opener
            for s in snaps:
                record_opener(s)
        except Exception as e:
            log.debug("Opener record skipped: %s", e)
    return counts


def get_todays_events(cache: Optional[OddsCache] = None,
                      date_utc_prefix: Optional[str] = None) -> list[str]:
    """Return all event_ids for a given UTC date (default: today)."""
    if cache is None:
        cache = OddsCache()
    if date_utc_prefix is None:
        date_utc_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return cache.events_on_date(date_utc_prefix)


def build_market_data(cache: OddsCache, event_id: str,
                      shop: bool = True) -> Optional[MarketData]:
    """Assemble a predictors.MarketData object for one event.

    Strategy:
      - When shop=True (default): prices come from whichever book has
        the best price per side (see data.line_shop.shop_event).
      - When shop=False: current prices come from Pinnacle (sharp
        anchor) if available, else DK.
      - Opener prices: earliest Pinnacle snapshot for this event, else
        earliest DK snapshot.

    Returns None if neither book has any snapshot for this event yet.
    """
    pin_latest = cache.latest(OddsBook.PINNACLE, event_id)
    dk_latest = cache.latest(OddsBook.DRAFTKINGS, event_id)
    latest = pin_latest or dk_latest
    if latest is None:
        return None

    pin_opener = cache.opener(OddsBook.PINNACLE, event_id)
    dk_opener = cache.opener(OddsBook.DRAFTKINGS, event_id)
    opener = pin_opener or dk_opener

    md = MarketData()

    shopped: Optional[ShoppedMarket] = shop_event(cache, event_id) if shop else None

    def _best_or(side: str, fallback: Optional[int]) -> Optional[int]:
        if shopped is None:
            return fallback
        bl = getattr(shopped, side)
        return bl.odds if bl is not None else fallback

    hml = _best_or("home_ml", latest.home_ml)
    aml = _best_or("away_ml", latest.away_ml)
    if hml is not None:
        md.home_ml_odds = hml
    if aml is not None:
        md.away_ml_odds = aml
    if opener and opener.home_ml is not None:
        md.opener_home_ml_odds = opener.home_ml

    hrl = _best_or("home_rl", latest.home_rl_odds)
    arl = _best_or("away_rl", latest.away_rl_odds)
    if hrl is not None:
        md.home_rl_odds = hrl
    if arl is not None:
        md.away_rl_odds = arl
    if latest.home_rl_line is not None:
        md.home_is_rl_favorite = latest.home_rl_line < 0

    over = _best_or("over", latest.over_odds)
    under = _best_or("under", latest.under_odds)
    if latest.total_line is not None:
        md.total_line = latest.total_line
    if over is not None:
        md.over_odds = over
    if under is not None:
        md.under_odds = under
    if opener and opener.total_line is not None:
        md.opener_total = opener.total_line

    if shopped is not None:
        md.shopped = shopped  # type: ignore[attr-defined]

    # Steam detection
    if pin_latest and dk_latest:
        if (pin_opener and dk_opener and
            pin_latest.total_line is not None and pin_opener.total_line is not None and
            dk_latest.total_line is not None and dk_opener.total_line is not None):
            both_up = (pin_latest.total_line > pin_opener.total_line and
                       dk_latest.total_line > dk_opener.total_line)
            both_down = (pin_latest.total_line < pin_opener.total_line and
                         dk_latest.total_line < dk_opener.total_line)
            if both_up or both_down:
                md.steam_flag_over = both_up

        if (pin_opener and dk_opener and
            pin_latest.home_ml is not None and pin_opener.home_ml is not None and
            dk_latest.home_ml is not None and dk_opener.home_ml is not None):
            pin_shorter = pin_latest.home_ml < pin_opener.home_ml
            dk_shorter = dk_latest.home_ml < dk_opener.home_ml
            md.steam_flag_home = pin_shorter and dk_shorter

    return md


# =============================================================================
# Per-book builders — for the side-by-side UI and per-book recommendations.
# =============================================================================

_BOOK_LABELS: dict[OddsBook, str] = {
    OddsBook.DRAFTKINGS: "draftkings",
    OddsBook.FANDUEL:    "fanduel",
    OddsBook.PINNACLE:   "pinnacle",
}


def _snapshot_to_market_data(s: OddsSnapshot) -> MarketData:
    """Lift a single-book OddsSnapshot into a MarketData dataclass.

    Unlike build_market_data, this does NOT price-shop. Every field comes
    from the same book, so the UI can present an honest per-book view.
    """
    md = MarketData()
    if s.home_ml is not None:
        md.home_ml_odds = s.home_ml
    if s.away_ml is not None:
        md.away_ml_odds = s.away_ml
    if s.home_rl_odds is not None:
        md.home_rl_odds = s.home_rl_odds
    if s.away_rl_odds is not None:
        md.away_rl_odds = s.away_rl_odds
    if s.home_rl_line is not None:
        md.home_is_rl_favorite = s.home_rl_line < 0
    if s.total_line is not None:
        md.total_line = s.total_line
    if s.over_odds is not None:
        md.over_odds = s.over_odds
    if s.under_odds is not None:
        md.under_odds = s.under_odds
    return md


def build_per_book_markets(
    cache: OddsCache,
    event_id: str,
    books: Optional[tuple[OddsBook, ...]] = None,
) -> dict[str, MarketData]:
    """Return {book_label: MarketData} for every book that has a snapshot.

    Pinnacle is included so downstream consumers can use it as a
    fair-line anchor; the frontend decides whether to display it.
    """
    if books is None:
        books = (OddsBook.DRAFTKINGS, OddsBook.FANDUEL, OddsBook.PINNACLE)
    out: dict[str, MarketData] = {}
    for b in books:
        s = cache.latest(b, event_id)
        if s is not None:
            out[_BOOK_LABELS[b]] = _snapshot_to_market_data(s)
    return out


def build_per_book_opening_markets(
    cache: OddsCache,
    event_id: str,
    books: Optional[tuple[OddsBook, ...]] = None,
) -> dict[str, MarketData]:
    """Return {book_label: MarketData} of the EARLIEST (opening) snapshot
    we have per book.

    Mirrors build_per_book_markets but uses cache.opener() rather than
    cache.latest(). Also consults the durable OpeningLineStore when the
    snapshots table has been pruned — so very old finals can still
    surface their opening prices. If neither source has an opener for
    a given book, that book is simply omitted from the result.
    """
    if books is None:
        books = (OddsBook.DRAFTKINGS, OddsBook.FANDUEL, OddsBook.PINNACLE)

    # Durable opener table — optional, may not exist if the collector
    # never ran. Load lazily to avoid a hard dependency.
    durable: dict[str, dict] = {}
    try:
        from models.opening_lines import _store  # type: ignore
        for b in books:
            rec = _store().get_opener(event_id, b)
            if rec is not None:
                durable[_BOOK_LABELS[b]] = rec
    except Exception:
        durable = {}

    out: dict[str, MarketData] = {}
    for b in books:
        label = _BOOK_LABELS[b]
        s = cache.opener(b, event_id)
        if s is not None:
            out[label] = _snapshot_to_market_data(s)
            continue
        rec = durable.get(label)
        if rec is not None:
            md = MarketData()
            if rec.get("home_ml") is not None:
                md.home_ml_odds = rec["home_ml"]
            if rec.get("away_ml") is not None:
                md.away_ml_odds = rec["away_ml"]
            if rec.get("home_rl_odds") is not None:
                md.home_rl_odds = rec["home_rl_odds"]
            if rec.get("away_rl_odds") is not None:
                md.away_rl_odds = rec["away_rl_odds"]
            hr_line = rec.get("home_rl_line")
            if hr_line is not None:
                md.home_is_rl_favorite = hr_line < 0
            if rec.get("total_line") is not None:
                md.total_line = rec["total_line"]
            if rec.get("over_odds") is not None:
                md.over_odds = rec["over_odds"]
            if rec.get("under_odds") is not None:
                md.under_odds = rec["under_odds"]
            out[label] = md
    return out
