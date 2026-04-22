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
from predictors.shared import fair_prob_consensus, rlm_intensity

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
      - When shop=False: current prices come from DK if available,
        else FD.
      - Opener prices: earliest DK snapshot for this event, else
        earliest FD snapshot.

    Pinnacle data is intentionally NOT used for consensus or predictions
    — it's not a routable book for the user, and mixing its pricing into
    the consensus produced picks that looked good but couldn't be bet.

    Returns None if neither book has any snapshot for this event yet.
    """
    dk_latest = cache.latest(OddsBook.DRAFTKINGS, event_id)
    fd_latest = cache.latest(OddsBook.FANDUEL, event_id)
    latest = dk_latest or fd_latest
    if latest is None:
        return None

    dk_opener = cache.opener(OddsBook.DRAFTKINGS, event_id)
    fd_opener = cache.opener(OddsBook.FANDUEL, event_id)
    opener = dk_opener or fd_opener

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

    # No-vig fair-prob consensus across DK + FD. Pinnacle is excluded
    # (sharp reference, but not routable for the user). If only one of
    # DK/FD has a given market, the consensus = that single de-vigged
    # book. Predictors pick these up via fair_prob_for_side().
    def _snap_pair(snap, side: str) -> tuple[Optional[int], Optional[int]]:
        if snap is None:
            return (None, None)
        if side == "ml":
            return (snap.home_ml, snap.away_ml)
        if side == "rl":
            return (snap.home_rl_odds, snap.away_rl_odds)
        if side == "total":
            return (snap.over_odds, snap.under_odds)
        return (None, None)

    ml_pairs = [_snap_pair(dk_latest, "ml"), _snap_pair(fd_latest, "ml")]
    rl_pairs = [_snap_pair(dk_latest, "rl"), _snap_pair(fd_latest, "rl")]
    tot_pairs = [_snap_pair(dk_latest, "total"), _snap_pair(fd_latest, "total")]
    fh, fa = fair_prob_consensus(ml_pairs)
    if fh is not None:
        md.fair_prob_home_ml = fh
        md.fair_prob_away_ml = fa
    fh, fa = fair_prob_consensus(rl_pairs)
    if fh is not None:
        md.fair_prob_home_rl = fh
        md.fair_prob_away_rl = fa
    fo, fu = fair_prob_consensus(tot_pairs)
    if fo is not None:
        md.fair_prob_over = fo
        md.fair_prob_under = fu

    # RLM intensity scores — computed from opener→current movement
    # against public splits. +1 = sharp action on home/over.
    md.rlm_score_home = rlm_intensity(
        opener_fav_odds=md.opener_home_ml_odds,
        current_fav_odds=md.home_ml_odds,
        public_ticket_pct_fav=md.public_ticket_pct_home,
        steam_flag_fav=md.steam_flag_home,
    )
    if md.opener_total is not None:
        # For totals, "fav" = over. Total moving UP = over taking money.
        # rlm_intensity treats shortening (current < opener) as fav
        # money, so encode total move with a SIGN FLIP: a +0.5-run move
        # UP maps to -20 cents, which looks like "fav shortened".
        synth_opener = 0
        synth_current = -int(round((md.total_line - md.opener_total) * 40))
        md.rlm_score_over = rlm_intensity(
            opener_fav_odds=synth_opener,
            current_fav_odds=synth_current,
            public_ticket_pct_fav=md.public_ticket_pct_over,
            steam_flag_fav=md.steam_flag_over,
        )

    # Opener odds — populate all sides where available, for RLM scoring
    if opener:
        if opener.away_ml is not None:
            md.opener_away_ml_odds = opener.away_ml
        if opener.home_rl_odds is not None:
            md.opener_home_rl_odds = opener.home_rl_odds
        if opener.away_rl_odds is not None:
            md.opener_away_rl_odds = opener.away_rl_odds
        if opener.over_odds is not None:
            md.opener_over_odds = opener.over_odds
        if opener.under_odds is not None:
            md.opener_under_odds = opener.under_odds

    # Steam detection — both DK and FD moved the same direction
    if dk_latest and fd_latest:
        if (dk_opener and fd_opener and
            dk_latest.total_line is not None and dk_opener.total_line is not None and
            fd_latest.total_line is not None and fd_opener.total_line is not None):
            both_up = (dk_latest.total_line > dk_opener.total_line and
                       fd_latest.total_line > fd_opener.total_line)
            both_down = (dk_latest.total_line < dk_opener.total_line and
                         fd_latest.total_line < fd_opener.total_line)
            if both_up or both_down:
                md.steam_flag_over = both_up

        if (dk_opener and fd_opener and
            dk_latest.home_ml is not None and dk_opener.home_ml is not None and
            fd_latest.home_ml is not None and fd_opener.home_ml is not None):
            dk_shorter = dk_latest.home_ml < dk_opener.home_ml
            fd_shorter = fd_latest.home_ml < fd_opener.home_ml
            md.steam_flag_home = dk_shorter and fd_shorter

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

    Only DK + FD are returned. Pinnacle is a sharp anchor but the user
    can't route bets to it, so it's intentionally excluded from the UI.
    """
    if books is None:
        books = (OddsBook.DRAFTKINGS, OddsBook.FANDUEL)
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
        books = (OddsBook.DRAFTKINGS, OddsBook.FANDUEL)

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
