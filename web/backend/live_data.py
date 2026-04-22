"""
Live-data orchestrator for the web API.

Every synthetic `_synth_*` function in `api.py` has a live counterpart
here. The API layer asks this module for real MarketData / TeamStats /
GameContext and falls back to its synthetic version whenever this
module returns None.

Design
------
- One `OddsCache`, one `ProjectionsCache`, one short-lived in-process
  schedule cache shared across every request.
- `warm_slate(date)` is the per-request hot-path: it polls DK/Pinnacle
  (if stale), reads the FanGraphs projection cache (refreshing once/day
  if empty or >12h old), and pulls today's MLB schedule+lineups+ump.
  Every subsequent call for the same date reuses the cached results
  until TTL expires.
- The builders (`build_market_for_event`, `build_team_stats_live`,
  `build_context_live`) are pure transforms over the cached data — they
  never hit the network themselves.

No imports from `web.backend.api` — this module is importable standalone
so tests and the background poller can use it without spinning up FastAPI.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from data.odds_cache import OddsCache
from data.odds_client import (
    fetch_all_books,
    build_market_data,
    build_per_book_markets,
    build_per_book_opening_markets,
)
from data.projections_cache import ProjectionsCache
from data.projections_fangraphs import fetch_all as fetch_all_projections
from data.projections_models import ProjectionSource
from data.projections_rollup import build_team_stats
from data.lineups_client import get_todays_games
from data.lineups_models import GameSchedule, LINEUP_CONFIRMED
from data.team_names import try_normalize_team
from data.weather import get_weather, GameWeather

from predictors import (
    GameContext, MarketData, TeamStats,
    PitcherStats, BullpenStats, OffenseStats,
)

log = logging.getLogger(__name__)


# =============================================================================
# Cache instances (module-level so a single poller + multiple request handlers
# share them)
# =============================================================================

# Both caches write to the same SQLite file by default (BBP_CACHE_PATH).
# We keep a single path so "bbp_cache.sqlite" is the canonical on-disk store.
_odds_cache: Optional[OddsCache] = None
_projections_cache: Optional[ProjectionsCache] = None


def odds_cache() -> OddsCache:
    global _odds_cache
    if _odds_cache is None:
        _odds_cache = OddsCache()
    return _odds_cache


def projections_cache() -> ProjectionsCache:
    global _projections_cache
    if _projections_cache is None:
        _projections_cache = ProjectionsCache()
    return _projections_cache


# =============================================================================
# Per-date warm cache
# =============================================================================

ODDS_TTL_SEC = 5 * 60               # poll DK/Pinnacle at most every 5 min
PROJECTIONS_TTL_SEC = 12 * 60 * 60  # refresh FanGraphs twice a day
SLATE_TTL_SEC = 60                  # MLB schedule+lineups — small TTL for score refresh

_slate_lock = threading.Lock()


class _SlateCache:
    """Everything we know about a single calendar date."""
    def __init__(self, d: date):
        self.date = d
        self.odds_polled_at: float = 0.0
        self.projections_fetched_at: float = 0.0
        # event_id -> GameSchedule (from MLB Stats API)
        self.schedule: dict[str, GameSchedule] = {}
        self.schedule_fetched_at: float = 0.0

    def needs_odds_refresh(self) -> bool:
        return (time.time() - self.odds_polled_at) > ODDS_TTL_SEC

    def needs_projections_refresh(self) -> bool:
        return (time.time() - self.projections_fetched_at) > PROJECTIONS_TTL_SEC

    def needs_schedule_refresh(self) -> bool:
        return (time.time() - self.schedule_fetched_at) > SLATE_TTL_SEC


_slate_by_date: dict[str, _SlateCache] = {}


def _ensure_projections_fresh(slate: _SlateCache) -> None:
    """Pull ROS projections from FanGraphs once per TTL.

    ZiPS + Steamer — ATC-style average is implicit via having both sources
    in the cache (predictors_rollup.build_* picks the configured source).
    """
    if not slate.needs_projections_refresh():
        return
    # If projections cache already has rows for today, skip the network.
    try:
        existing = projections_cache().stats()
        if existing.get("hitter_rows", 0) > 0 and existing.get("pitcher_rows", 0) > 0:
            # Still mark as fresh so we don't re-pull every request; a
            # scheduled task (or explicit refresh endpoint) can force it.
            slate.projections_fetched_at = time.time() - (PROJECTIONS_TTL_SEC - 60 * 60)
            return
    except Exception:
        pass

    try:
        pulled = fetch_all_projections((ProjectionSource.ZIPS, ProjectionSource.STEAMER))
    except Exception as e:
        log.exception("FanGraphs pull failed: %s", e)
        slate.projections_fetched_at = time.time()  # don't hammer on repeated failure
        return

    pc = projections_cache()
    total_h = total_p = 0
    for _, (hitters, pitchers) in pulled.items():
        total_h += pc.upsert_hitters(hitters)
        total_p += pc.upsert_pitchers(pitchers)
    log.info("Projections refreshed: %d hitter rows, %d pitcher rows", total_h, total_p)
    slate.projections_fetched_at = time.time()


def _ensure_odds_fresh(slate: _SlateCache) -> None:
    """Poll DK + FD + Pinnacle and persist to OddsCache (if past TTL).

    Pinnacle is still scraped (useful as a sharp reference in diagnostics)
    but is excluded from consensus and the per-book UI.
    """
    if not slate.needs_odds_refresh():
        return
    try:
        counts = fetch_all_books(odds_cache())
        log.info("Odds poll %s: %s", slate.date.isoformat(), counts)
    except Exception as e:
        log.exception("Odds poll failed: %s", e)
    # Mark polled even on failure so we don't hammer a broken endpoint.
    slate.odds_polled_at = time.time()


def _ensure_schedule_fresh(slate: _SlateCache) -> None:
    """Refresh MLB schedule + lineups + home plate ump from MLB Stats API."""
    if not slate.needs_schedule_refresh() and slate.schedule:
        return
    try:
        games = get_todays_games(slate.date.isoformat(), use_projected_fallback=True)
        slate.schedule = {g.event_id: g for g in games}
    except Exception as e:
        log.warning("MLB schedule refresh failed: %s", e)
        # Keep old schedule — better to show stale than nothing.
    slate.schedule_fetched_at = time.time()


def warm_slate(game_date: date, *, force_odds: bool = False,
               force_projections: bool = False) -> _SlateCache:
    """Ensure our in-memory slate for `game_date` is fresh, return it.

    Thread-safe. Called at the top of each request handler; cheap on the
    hot path because most requests in a 5-minute window hit the cached
    results.
    """
    key = game_date.isoformat()
    with _slate_lock:
        slate = _slate_by_date.get(key)
        if slate is None:
            slate = _SlateCache(game_date)
            _slate_by_date[key] = slate
        if force_odds:
            slate.odds_polled_at = 0.0
        if force_projections:
            slate.projections_fetched_at = 0.0
        _ensure_projections_fresh(slate)
        _ensure_odds_fresh(slate)
        _ensure_schedule_fresh(slate)
        return slate


# =============================================================================
# Event-id translation
# =============================================================================
# Our MLB schedule feed (web.backend.mlb_data) produces ids like
# "YYYY-MM-DD-AWAY@HOME" with 3-letter abbrs. The odds/lineups layer uses
# "YYYY-MM-DD|Full Team Away|Full Team Home". The translator below maps
# one to the other.

def odds_event_id_from_api_game(g: dict) -> Optional[str]:
    """Build an event_id matching the odds cache from an API game dict.

    g has raw["away"]["abbr"], raw["home"]["abbr"], and date.
    """
    raw = g["raw"]
    # Get canonical names from abbr
    away_abbr = raw["away"]["abbr"]
    home_abbr = raw["home"]["abbr"]
    from web.backend.mlb_data import TEAM_PARK_FACTORS as _ignored  # ensure module import
    # data.team_names has the abbr → canonical mapping for FG abbrs which
    # is a superset of what we need here.
    from data.team_names import FG_ABBR_TO_CANONICAL
    away_canon = FG_ABBR_TO_CANONICAL.get(away_abbr.upper())
    home_canon = FG_ABBR_TO_CANONICAL.get(home_abbr.upper())
    if not away_canon or not home_canon:
        return None
    d = g["date"]
    # Date is already YYYY-MM-DD (local ET). This may differ from the UTC
    # date used by the odds scraper for late west-coast games. We accept
    # this tradeoff — the on-disk event_id uses the gameDate UTC strftime.
    return f"{d}|{away_canon}|{home_canon}"


# =============================================================================
# Live builders
# =============================================================================

def build_market_live(game: dict, slate: _SlateCache) -> Optional[MarketData]:
    """Return a MarketData assembled from live DK/Pinnacle, or None if the
    cache has no snapshot for this event.
    """
    ev = odds_event_id_from_api_game(game)
    if not ev:
        return None
    # The odds cache uses the game's UTC date as the key. Also try
    # adjacent dates in case of timezone drift.
    candidates = [ev]
    # Append yesterday/tomorrow variants (late ET games slip into next UTC day)
    try:
        d_parts = ev.split("|", 1)
        from datetime import datetime as _dt, timedelta as _td
        base = _dt.strptime(d_parts[0], "%Y-%m-%d").date()
        for delta in (-1, +1):
            shifted = (base + _td(days=delta)).isoformat()
            candidates.append(f"{shifted}|{d_parts[1]}")
    except Exception:
        pass

    for candidate in candidates:
        md = build_market_data(odds_cache(), candidate, shop=True)
        if md is not None:
            return md
    return None


def build_per_book_markets_live(game: dict, slate: _SlateCache
                                ) -> dict[str, MarketData]:
    """Return {book_label: MarketData} for one game, trying ±1 UTC day.

    Empty dict if no books have snapshots. Keys are "draftkings" and
    "fanduel" (whichever are present in the cache). Pinnacle data is
    intentionally excluded — see data.odds_client.build_market_data.
    """
    ev = odds_event_id_from_api_game(game)
    if not ev:
        return {}
    candidates = [ev]
    try:
        d_parts = ev.split("|", 1)
        from datetime import datetime as _dt, timedelta as _td
        base = _dt.strptime(d_parts[0], "%Y-%m-%d").date()
        for delta in (-1, +1):
            shifted = (base + _td(days=delta)).isoformat()
            candidates.append(f"{shifted}|{d_parts[1]}")
    except Exception:
        pass

    for candidate in candidates:
        pb = build_per_book_markets(odds_cache(), candidate)
        if pb:
            return pb
    return {}


def build_per_book_opening_markets_live(game: dict, slate: _SlateCache
                                        ) -> dict[str, MarketData]:
    """Return {book_label: MarketData} of OPENING snapshots for one game,
    trying ±1 UTC day on the event id. Empty dict if nothing found.

    These are the prices the first time we ever saw each book quote the
    game — useful for showing the 'starting line' on final games.
    """
    ev = odds_event_id_from_api_game(game)
    if not ev:
        return {}
    candidates = [ev]
    try:
        d_parts = ev.split("|", 1)
        from datetime import datetime as _dt, timedelta as _td
        base = _dt.strptime(d_parts[0], "%Y-%m-%d").date()
        for delta in (-1, +1):
            shifted = (base + _td(days=delta)).isoformat()
            candidates.append(f"{shifted}|{d_parts[1]}")
    except Exception:
        pass

    for candidate in candidates:
        pb = build_per_book_opening_markets(odds_cache(), candidate)
        if pb:
            return pb
    return {}


def _pick_side_key(market_name: str, pick: str) -> Optional[str]:
    """Map a predictor's 'pick' string to the side key we price-shop against.

    Returns one of: "home_ml", "away_ml", "home_rl", "away_rl",
                    "over", "under", or None for NO BET / unknown.
    """
    if not pick or pick.upper().startswith("NO BET"):
        return None
    up = pick.upper()
    if market_name == "moneyline":
        if up.startswith("HOME"):
            return "home_ml"
        if up.startswith("AWAY"):
            return "away_ml"
    elif market_name == "run_line":
        if up.startswith("HOME"):
            return "home_rl"
        if up.startswith("AWAY"):
            return "away_rl"
    elif market_name == "totals":
        if "OVER" in up:
            return "over"
        if "UNDER" in up:
            return "under"
    return None


def _odds_for_side(md: MarketData, side_key: str) -> Optional[int]:
    """Pluck the American-odds price for a given side from a MarketData."""
    return {
        "home_ml": md.home_ml_odds,
        "away_ml": md.away_ml_odds,
        "home_rl": md.home_rl_odds,
        "away_rl": md.away_rl_odds,
        "over":    md.over_odds,
        "under":   md.under_odds,
    }.get(side_key)


def _line_for_side(md: MarketData, side_key: str) -> Optional[float]:
    """The LINE (point spread / total) associated with a side, if any."""
    if side_key in ("over", "under"):
        return md.total_line
    if side_key in ("home_rl", "away_rl"):
        # Run line is ±1.5; sign depends on who's favored
        if side_key == "home_rl":
            return -1.5 if md.home_is_rl_favorite else +1.5
        return +1.5 if md.home_is_rl_favorite else -1.5
    return None


def _format_pick_for_book(
    market_name: str,
    side_key: str,
    md: MarketData,
    original_pick: str,
) -> str:
    """Rebuild the pick label using the chosen book's own line/orientation.

    Different books can disagree on who is the RL favorite, or post
    different totals. The predictor's original pick string is computed
    from the consensus MarketData, so routing a rec to a book whose line
    differs from the consensus would leave a stale label (e.g. showing
    "AWAY +1.5 +164" when the book actually has that team at -1.5).
    This rebuild keeps the semantic team/side the same but writes the
    ±1.5 / total using the book we're recommending.
    """
    if market_name == "moneyline":
        if side_key == "home_ml":
            return "HOME"
        if side_key == "away_ml":
            return "AWAY"
        return original_pick
    if market_name == "run_line":
        if side_key == "home_rl":
            line = "-1.5" if md.home_is_rl_favorite else "+1.5"
            return f"HOME {line}"
        if side_key == "away_rl":
            line = "+1.5" if md.home_is_rl_favorite else "-1.5"
            return f"AWAY {line}"
        return original_pick
    if market_name == "totals":
        total = md.total_line if md.total_line is not None else ""
        if side_key == "over":
            return f"OVER {total}".strip()
        if side_key == "under":
            return f"UNDER {total}".strip()
        return original_pick
    return original_pick


def _ev_per_unit(american_odds: int, true_prob: float) -> float:
    """EV per $1 risked: p*win - (1-p)."""
    if american_odds >= 100:
        win = american_odds / 100.0
    else:
        win = 100.0 / abs(american_odds)
    return true_prob * win - (1.0 - true_prob)


def recommendations_from_predictions(
    predictions: dict,
    per_book: dict[str, MarketData],
    *,
    books: tuple[str, ...] = ("draftkings", "fanduel"),
) -> list[dict]:
    """Build per-market "bet here for the best price" recommendations.

    For each of moneyline / run_line / totals, this:
      1. Reads the model's pick + true probability (from the shopped
         prediction).
      2. Looks up the price that side gets at each book in `books`.
      3. Returns a record with the best-priced book, its odds, EV at
         that book, and a side-by-side comparison of DK vs FD prices.

    Skipped when:
      - model picked NO BET
      - neither DK nor FD has a snapshot for this event (the recommendation
        would have no book to route to)
    """
    recs: list[dict] = []
    for market_name in ("moneyline", "run_line", "totals"):
        pred = predictions.get(market_name)
        if pred is None:
            continue
        side = _pick_side_key(market_name, getattr(pred, "pick", "") or "")
        if side is None:
            continue

        true_prob = float(getattr(pred, "model_prob", 0.0))
        per_book_prices: dict[str, dict] = {}
        best_book: Optional[str] = None
        best_odds: Optional[int] = None
        for b in books:
            md = per_book.get(b)
            if md is None:
                per_book_prices[b] = {"available": False}
                continue
            price = _odds_for_side(md, side)
            line = _line_for_side(md, side)
            if price is None:
                per_book_prices[b] = {"available": False}
                continue
            ev = _ev_per_unit(price, true_prob) if true_prob > 0 else None
            per_book_prices[b] = {
                "available": True,
                "odds": price,
                "line": line,
                "ev_per_unit": round(ev, 4) if ev is not None else None,
            }
            # Higher American odds always = better payout for the bettor.
            if best_odds is None or price > best_odds:
                best_odds = price
                best_book = b
        if best_book is None:
            continue
        # Rebuild the pick label against the book we're routing to — the
        # predictor's original label is written from the consensus line,
        # which can disagree with the best book's orientation (e.g. DK
        # has STL -1.5 while Pinnacle/consensus have MIA -1.5). Keep the
        # same semantic side (HOME/AWAY/OVER/UNDER) but re-derive the
        # ±1.5 / total from the best book's MarketData, then append the
        # best-book odds so the label matches p.pick's format
        # ("AWAY -1.5 +164") and renders identically in the Daily Picks
        # and game-detail views.
        best_md = per_book.get(best_book)
        if best_md is not None:
            side_label = _format_pick_for_book(
                market_name, side, best_md, getattr(pred, "pick", "")
            )
        else:
            side_label = getattr(pred, "pick", "")
        if best_odds is not None:
            odds_str = f"+{best_odds}" if best_odds > 0 else f"{best_odds}"
            displayed_pick = f"{side_label} {odds_str}".strip()
        else:
            displayed_pick = side_label
        recs.append({
            "market": market_name,
            "pick": displayed_pick,
            "model_prob": round(true_prob, 4),
            "side": side,
            "best_book": best_book,
            "best_odds": best_odds,
            "best_line": per_book_prices[best_book].get("line"),
            "best_ev_per_unit": per_book_prices[best_book].get("ev_per_unit"),
            "books": per_book_prices,
        })
    return recs


def _find_schedule(game: dict, slate: _SlateCache) -> Optional[GameSchedule]:
    ev = odds_event_id_from_api_game(game)
    if ev and ev in slate.schedule:
        return slate.schedule[ev]
    # Fallback: date-agnostic match by team pair
    raw = game["raw"]
    from data.team_names import FG_ABBR_TO_CANONICAL
    home_canon = FG_ABBR_TO_CANONICAL.get(raw["home"]["abbr"].upper())
    away_canon = FG_ABBR_TO_CANONICAL.get(raw["away"]["abbr"].upper())
    if not home_canon or not away_canon:
        return None
    for g in slate.schedule.values():
        if g.home_team == home_canon and g.away_team == away_canon:
            return g
    return None


def build_team_stats_live(game: dict, slate: _SlateCache,
                          *, is_home: bool) -> Optional[TeamStats]:
    """Build TeamStats from FanGraphs ROS projections + real lineup.

    Returns None when:
      - no projection rows for the team (pre-season / stale cache)
      - no probable starter is listed for the team
    """
    sched = _find_schedule(game, slate)
    raw_side = game["raw"]["home"] if is_home else game["raw"]["away"]

    # Team canonical
    from data.team_names import FG_ABBR_TO_CANONICAL
    team_canon = FG_ABBR_TO_CANONICAL.get(raw_side["abbr"].upper())
    if not team_canon:
        return None

    # Starter name — prefer the lineups feed, fall back to the schedule feed
    starter_name: Optional[str] = None
    starter_throws = raw_side.get("pitcher_hand") or "R"
    if sched is not None:
        ps = sched.home_starter if is_home else sched.away_starter
        if ps and ps.name and ps.name != "TBD":
            starter_name = ps.name
            starter_throws = ps.throws or starter_throws
    if not starter_name:
        starter_name = raw_side.get("pitcher_name") or "TBD"
    if starter_name == "TBD":
        # No starter to rollup against — bail, synth fallback will fill it.
        return None

    pc = projections_cache()
    hitters = pc.latest_hitters(ProjectionSource.ZIPS, team=team_canon)
    pitchers = pc.latest_pitchers(ProjectionSource.ZIPS, team=team_canon)
    if not hitters or not pitchers:
        # Try Steamer as a secondary source
        hitters = pc.latest_hitters(ProjectionSource.STEAMER, team=team_canon)
        pitchers = pc.latest_pitchers(ProjectionSource.STEAMER, team=team_canon)
    if not hitters or not pitchers:
        return None

    lineup_ids: Optional[list[str]] = None
    lineup_is_confirmed = False
    if sched is not None:
        lu = sched.home_lineup if is_home else sched.away_lineup
        if lu and lu.slots:
            lineup_ids = lu.player_ids
            lineup_is_confirmed = lu.source == LINEUP_CONFIRMED

    ts = build_team_stats(
        hitters, pitchers, team_canon, starter_name,
        is_home=is_home, starter_throws=starter_throws,
        lineup_player_ids=lineup_ids,
    )
    # Stamp the real confirmation status
    ts.lineup_confirmed = lineup_is_confirmed
    ts.starter_confirmed = starter_name != "TBD"
    # Travel/rest aren't in the projection data — leave at library defaults
    # which are 0 miles for home, 1 day rest. The bet-selection layer's
    # sensitivity to these is low so we can fill them later if needed.
    if not is_home:
        # Best-effort travel marker — we don't know previous-game opponent
        # without a multi-day schedule join, so use 0 here as "unknown".
        ts.travel_miles_72h = 0
    return ts


def build_context_live(game: dict, slate: _SlateCache,
                       home_abbr: str, park_run_factor: float,
                       park_hr_factor: float, altitude_ft: int,
                       is_domed_home: bool) -> Optional[GameContext]:
    """Build GameContext from real weather + real umpire, using the park
    factors / altitude already known by the caller.

    Returns None when no weather fetch succeeds and no ump is known — the
    caller should then fall back to the synthetic context.
    """
    from data.team_names import FG_ABBR_TO_CANONICAL
    team_canon = FG_ABBR_TO_CANONICAL.get(home_abbr.upper())

    # First pitch
    iso = game["raw"].get("game_date_utc") or ""
    when: Optional[datetime] = None
    try:
        when = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        when = None

    # Day/night heuristic: before 19:00 UTC ~= daytime ET/CT.
    day_game = bool(when and when.hour < 19)

    # Umpire
    sched = _find_schedule(game, slate)
    ump_name: Optional[str] = None
    if sched and sched.home_plate_ump:
        ump_name = sched.home_plate_ump.name or None

    # Weather — skip for domed venues (roof-closed defaults)
    weather: Optional[GameWeather] = None
    if team_canon and not is_domed_home:
        try:
            weather = get_weather(team_canon, when)
        except Exception as e:
            log.debug("weather fetch failed for %s: %s", team_canon, e)

    # If we have neither weather nor ump AND venue is outdoor, return None
    # so caller can use the synthetic defaults.
    if weather is None and not ump_name and not is_domed_home:
        return None

    # Ump R/G — the MLB Stats API returns the plate ump's NAME but not
    # their personal runs-per-game average. Until Umpire Scorecards is
    # wired in as a real source (see data_sources.md), we fall back to
    # the league average from predictors.shared.LEAGUE, which makes
    # umpire_delta(ctx) == 0 for every game (no signal either way).
    # FIXME(umpire-data): replace with a real per-umpire R/G lookup
    # keyed on ``ump_name``. This is the single biggest silenced family
    # in the totals predictor — see project_family_signal_finding.md.
    from predictors.shared import LEAGUE as _LEAGUE
    ump_rpg = _LEAGUE["ump_runs_per_game"]
    ump_csr = 0.50

    roof = "closed" if is_domed_home else "none"
    wind_speed = 0.0 if is_domed_home else (weather.wind_speed_mph if weather else 0.0)
    wind_dir = "none" if is_domed_home else (weather.wind_relative_to_cf if weather else "none")
    temp_f = weather.temperature_f if weather else 72.0
    hum = weather.humidity_pct if weather else 55.0
    prec = 0.0 if is_domed_home else (weather.precipitation_pct if weather else 0.0)

    return GameContext(
        park_run_factor=park_run_factor,
        park_hr_factor=park_hr_factor,
        altitude_ft=altitude_ft,
        roof_status=roof,
        wind_speed_mph=wind_speed,
        wind_direction=wind_dir,
        temperature_f=temp_f,
        humidity_pct=hum,
        precipitation_pct=prec,
        ump_runs_per_game=ump_rpg,
        ump_called_strike_rate=ump_csr,
        day_game=day_game,
        doubleheader=False,
        extra_innings_prev_game=False,
    )


# =============================================================================
# Background poller
# =============================================================================

_poller_thread: Optional[threading.Thread] = None
_poller_stop = threading.Event()
POLLER_INTERVAL_SEC = 10 * 60


def _poll_loop() -> None:
    """Background worker: refreshes odds for the current ET date every
    POLLER_INTERVAL_SEC so openers + line-movement history stay dense.
    """
    import web.backend.mlb_data as mlb_data  # noqa: WPS433 local import to avoid cycles
    while not _poller_stop.is_set():
        try:
            today = mlb_data.today_et()
            warm_slate(today, force_odds=True)
        except Exception as e:
            log.exception("poller iteration failed: %s", e)
        _poller_stop.wait(POLLER_INTERVAL_SEC)


def start_background_poller() -> None:
    global _poller_thread
    if _poller_thread and _poller_thread.is_alive():
        return
    _poller_stop.clear()
    t = threading.Thread(target=_poll_loop, name="bbp-odds-poller", daemon=True)
    t.start()
    _poller_thread = t
    log.info("Background odds poller started (every %ds)", POLLER_INTERVAL_SEC)


def stop_background_poller() -> None:
    _poller_stop.set()
