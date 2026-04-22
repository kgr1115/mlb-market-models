"""
Live-cache odds loader for the backtest.

Reads completed games from our live OddsCache (``bbp_cache.sqlite``) and
emits ``HistoricalOdds`` rows in the same shape the SBR / community
loaders produce. This lets the backtest corpus extend past the community
dataset's 2025-08-16 cutoff — every game we've scraped since then can
be graded as soon as its result is in.

Sourcing strategy
-----------------
For each event in the cache whose ``game_time_utc`` falls within the
requested season (or window):

  - **Closing line** = latest snapshot polled at or before first pitch.
    Mid-game polls are excluded — a 3rd-inning price is not a market
    closing line. If the latest poll actually ran after first pitch
    (can happen if a game starts before the 5-min poll tick), we take
    the most-recent pre-game poll we can find; failing that we fall
    back to the latest overall.
  - **Opening line**  = earliest snapshot on file.
  - **Book**          = DraftKings preferred, FanDuel fallback. Pinnacle
    snapshots are intentionally ignored so the historical price matches
    what a DK-or-FD bettor would actually have gotten.

If neither DK nor FD has any snapshot for an event, the event is
skipped (same treatment the community loader gives events with no
preferred book).

Run-line representation
-----------------------
The cache stores a single ``home_rl_line`` (handicap laid by home) and
paired ``home_rl_odds`` / ``away_rl_odds``. ``HistoricalOdds`` wants
``home_rl_line`` + ``home_rl_price`` and ``away_rl_line`` + ``away_rl_price``
— so we mirror: ``away_rl_line = -home_rl_line``.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from data.odds_cache import DEFAULT_CACHE_PATH
from data.odds_models import OddsBook, OddsSnapshot

from .historical_odds import HistoricalOdds

log = logging.getLogger(__name__)


# Same ordering as data/odds_client.py after the Pinnacle removal:
# DK first because it's the book we route bets to most; FD fills gaps.
_BOOK_PREFERENCE: tuple[OddsBook, ...] = (OddsBook.DRAFTKINGS, OddsBook.FANDUEL)


# =============================================================================
# Public API
# =============================================================================

def load_livecache_season_odds(
    season: int,
    cache_path: Optional[Path] = None,
    through_date: Optional[str] = None,
    since_date: Optional[str] = None,
) -> dict[str, HistoricalOdds]:
    """Return ``{event_id: HistoricalOdds}`` for every event in the cache
    that falls in ``season`` (and, optionally, before ``through_date``
    or after ``since_date`` — both inclusive, YYYY-MM-DD).

    The filter uses the event_id's leading date token (UTC) to match the
    community loader's ``game_date`` convention.
    """
    path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    if not path.exists():
        log.warning("livecache: no odds cache at %s — returning 0 rows", path)
        return {}

    out: dict[str, HistoricalOdds] = {}
    skipped_no_book = 0
    skipped_out_of_range = 0
    skipped_no_markets = 0

    # One connection, one query per event. We pull event_ids + game_time
    # up front so we don't reopen the connection in a tight loop.
    conn = sqlite3.connect(path, timeout=10.0)
    try:
        events = _list_events_for_season(
            conn, season, since_date=since_date, through_date=through_date,
        )
        for event_id, game_time_utc in events:
            game_date = event_id.split("|", 1)[0]

            snap_close, snap_open = _pick_book_snapshots(conn, event_id, game_time_utc)
            if snap_close is None and snap_open is None:
                skipped_no_book += 1
                continue

            ref = snap_close or snap_open  # team names / date parsing
            assert ref is not None

            # Markets coverage check — bail if the chosen book has nothing
            # we can grade. An event with ML only is still useful (ML bets
            # get graded); totals-only and RL-only are also fine.
            has_any_market = (
                ref.home_ml is not None
                or ref.home_rl_odds is not None
                or ref.total_line is not None
            )
            if not has_any_market:
                skipped_no_markets += 1
                continue

            ho = _to_historical_odds(event_id, game_date, snap_close, snap_open)
            out[event_id] = ho
    finally:
        conn.close()

    log.info(
        "livecache odds %d: %d events loaded "
        "(skipped %d no_book, %d no_markets, %d out_of_range) "
        "[through=%s, since=%s]",
        season, len(out), skipped_no_book, skipped_no_markets,
        skipped_out_of_range, through_date, since_date,
    )
    return out


def load_livecache_window_odds(
    start_date: str,
    end_date: str,
    cache_path: Optional[Path] = None,
) -> dict[str, HistoricalOdds]:
    """Slim variant that doesn't care about season, just a date window.

    Useful when the caller wants to grade "yesterday" or the last week
    without thinking about what season that falls in.
    """
    # Fan the work out to load_livecache_season_odds per season to reuse
    # its filtering/logging. Typically start_date and end_date are in the
    # same season.
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    merged: dict[str, HistoricalOdds] = {}
    for season in range(start_year, end_year + 1):
        merged.update(load_livecache_season_odds(
            season, cache_path=cache_path,
            since_date=start_date, through_date=end_date,
        ))
    return merged


# =============================================================================
# Implementation helpers
# =============================================================================

def _list_events_for_season(
    conn: sqlite3.Connection,
    season: int,
    since_date: Optional[str],
    through_date: Optional[str],
) -> list[tuple[str, datetime]]:
    """Distinct (event_id, game_time_utc) pairs in the requested window."""
    year_prefix = f"{season}-"
    where = ["game_time_utc LIKE ?"]
    params: list[object] = [f"{year_prefix}%"]

    if since_date:
        where.append("game_time_utc >= ?")
        params.append(since_date)
    if through_date:
        # make "through" inclusive of the whole calendar day
        where.append("game_time_utc < ?")
        params.append(_day_after(through_date))

    sql = f"""
        SELECT event_id, MIN(game_time_utc) AS gt
        FROM odds_snapshots
        WHERE {" AND ".join(where)}
        GROUP BY event_id
        ORDER BY gt ASC
    """
    rows = conn.execute(sql, params).fetchall()
    out: list[tuple[str, datetime]] = []
    for event_id, gt in rows:
        try:
            out.append((event_id, datetime.fromisoformat(gt)))
        except Exception:
            continue
    return out


def _day_after(ymd: str) -> str:
    """Return the YYYY-MM-DD that comes right after ymd (exclusive upper bound)."""
    from datetime import timedelta
    d = datetime.fromisoformat(ymd).date()
    return (d + timedelta(days=1)).isoformat()


def _pick_book_snapshots(
    conn: sqlite3.Connection,
    event_id: str,
    game_time_utc: datetime,
) -> tuple[Optional[OddsSnapshot], Optional[OddsSnapshot]]:
    """Return (closing_snapshot, opening_snapshot) from the preferred book.

    Walks ``_BOOK_PREFERENCE`` — returns as soon as one book has *any*
    snapshot for this event. The chosen book is used for both closing
    and opening, so per-book directionality (RL favorite side) is
    consistent across the pair.
    """
    gt_iso = game_time_utc.isoformat()
    for book in _BOOK_PREFERENCE:
        close = _fetch_closing(conn, book, event_id, gt_iso)
        opener = _fetch_opening(conn, book, event_id)
        if close is not None or opener is not None:
            return close, opener
    return None, None


def _fetch_closing(
    conn: sqlite3.Connection,
    book: OddsBook,
    event_id: str,
    game_time_iso: str,
) -> Optional[OddsSnapshot]:
    """Most recent pre-game snapshot. Falls back to overall latest if no
    pre-game poll exists (e.g. we only started polling a game after the
    first pitch — rare but happens)."""
    row = conn.execute(
        """
        SELECT book, event_id, home_team, away_team, game_time_utc,
               home_ml, away_ml, home_rl_line, home_rl_odds, away_rl_odds,
               total_line, over_odds, under_odds, polled_at_utc, native_event_id
        FROM odds_snapshots
        WHERE book = ? AND event_id = ? AND polled_at_utc <= ?
        ORDER BY polled_at_utc DESC
        LIMIT 1
        """,
        (book.value, event_id, game_time_iso),
    ).fetchone()
    if row is None:
        # Fall back to absolute-latest (post-game polls, if any).
        row = conn.execute(
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
        ).fetchone()
    return OddsSnapshot.from_row(row) if row else None


def _fetch_opening(
    conn: sqlite3.Connection,
    book: OddsBook,
    event_id: str,
) -> Optional[OddsSnapshot]:
    row = conn.execute(
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
    ).fetchone()
    return OddsSnapshot.from_row(row) if row else None


def _to_historical_odds(
    event_id: str,
    game_date: str,
    close: Optional[OddsSnapshot],
    opener: Optional[OddsSnapshot],
) -> HistoricalOdds:
    """Assemble a HistoricalOdds row from close/open snapshots.

    Either snapshot may be None; markets fall through to whichever side
    we have. When both are present, close dominates for graded prices
    and opener supplies the ``_open`` fields.
    """
    ref = close or opener
    assert ref is not None

    # --- ML ---------------------------------------------------------------
    away_ml_close = close.away_ml if close else None
    home_ml_close = close.home_ml if close else None
    away_ml_open  = opener.away_ml if opener else None
    home_ml_open  = opener.home_ml if opener else None

    # --- RL ---------------------------------------------------------------
    # ``home_rl_line`` in OddsSnapshot is the handicap laid BY HOME (neg
    # when home is fav). HistoricalOdds stores both sides' lines, so
    # mirror home -> away.
    home_rl_line: Optional[float] = None
    away_rl_line: Optional[float] = None
    home_rl_price: Optional[int] = None
    away_rl_price: Optional[int] = None
    if close and close.home_rl_line is not None:
        home_rl_line = float(close.home_rl_line)
        away_rl_line = -home_rl_line
        home_rl_price = close.home_rl_odds
        away_rl_price = close.away_rl_odds

    # --- Totals -----------------------------------------------------------
    total_close: Optional[float] = None
    total_over_close: Optional[int] = None
    total_under_close: Optional[int] = None
    if close and close.total_line is not None:
        total_close = float(close.total_line)
        total_over_close = close.over_odds
        total_under_close = close.under_odds

    total_open: Optional[float] = None
    total_over_open: Optional[int] = None
    total_under_open: Optional[int] = None
    if opener and opener.total_line is not None:
        total_open = float(opener.total_line)
        total_over_open = opener.over_odds
        total_under_open = opener.under_odds

    return HistoricalOdds(
        event_id=event_id,
        game_date=game_date,
        away_team=ref.away_team,
        home_team=ref.home_team,
        away_ml_close=away_ml_close,
        home_ml_close=home_ml_close,
        away_ml_open=away_ml_open,
        home_ml_open=home_ml_open,
        away_rl_line=away_rl_line,
        away_rl_price=away_rl_price,
        home_rl_line=home_rl_line,
        home_rl_price=home_rl_price,
        total_close=total_close,
        total_over_close=total_over_close,
        total_under_close=total_under_close,
        total_open=total_open,
        total_over_open=total_over_open,
        total_under_open=total_under_open,
    )
