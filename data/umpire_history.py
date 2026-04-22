"""
Historical plate-umpire lookup — MLB Stats API ``hydrate=officials``.

Why this module exists
----------------------
The totals predictor has a ``umpire`` family (5% weight) driven by
``ctx.ump_runs_per_game`` and ``ctx.ump_called_strike_rate``. In the
backtest path these two fields default to the league mean, which
silences the family to zero for every game. Per
``project_post_weather_family_signals.md`` this is the single largest
remaining dormant family after the weather unlock.

This module populates the R/G signal by:

  1. Fetching plate-umpire names for every historical game via the
     MLB Stats API schedule endpoint with ``hydrate=officials``. One
     HTTP call per season covers ~2,400 games and is safe to re-run
     (results cached on disk).
  2. Self-computing each ump's career-to-date R/G from the actual
     home+away runs of the games they worked. No external umpire
     dataset needed — we already have final scores for the whole
     corpus in ``historical_games.py``.

Called-strike rate is NOT populated here; that requires Statcast
pitch data. The totals predictor already handles the CSR default
(0.50) as a no-op branch, so leaving CSR at default simply means the
umpire family contributes only the R/G signal. This is the biggest
chunk of the signal per Umpire Scorecards methodology.

Usage
-----
    from data.umpire_history import prewarm_season, get_plate_ump

    prewarm_season(2024)                        # one HTTP call
    name = get_plate_ump(game_pk=745678)        # "Doug Eddings" or None

Then in the engine:
    from data.umpire_history import build_ump_rpg_lookup
    ump_rpg = build_ump_rpg_lookup(games)
    # ump_rpg[event_id] -> career R/G before this game (or None)

Endpoint
--------
    https://statsapi.mlb.com/api/v1/schedule
        ?sportId=1
        &season=YYYY
        &gameType=R
        &hydrate=officials

Response shape (abbreviated):
    dates: [ { games: [ { gamePk, officials: [ { official: {fullName},
                                                  officialType: "Home Plate" } ] } ] } ]
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


_SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&season={season}&gameType=R"
    "&hydrate=officials"
)
_HEADERS = {
    "User-Agent": "BBP-UmpireHistory/0.1 (+local)",
    "Accept": "application/json",
}
_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_CACHE_FILE = _CACHE_DIR / "umpire_history.json"
_REQUEST_TIMEOUT = 30.0
_MIN_SECONDS_BETWEEN_CALLS = 0.25
_last_call_ts = 0.0


# In-memory mirror. Keys are game_pk stringified (JSON keys are strings).
_mem_cache: Optional[dict[str, str]] = None


def _ensure_loaded() -> dict[str, str]:
    global _mem_cache
    if _mem_cache is not None:
        return _mem_cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _CACHE_FILE.exists():
        try:
            with _CACHE_FILE.open("r", encoding="utf-8") as f:
                _mem_cache = json.load(f)
        except Exception as e:
            log.warning("umpire_history: failed to load cache (%s); starting fresh", e)
            _mem_cache = {}
    else:
        _mem_cache = {}
    return _mem_cache


def save_cache() -> None:
    """Flush the in-memory cache to disk."""
    if _mem_cache is None:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_mem_cache, f, indent=0, sort_keys=True)
    os.replace(tmp, _CACHE_FILE)


def _throttle() -> None:
    global _last_call_ts
    now = time.monotonic()
    delta = now - _last_call_ts
    if delta < _MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(_MIN_SECONDS_BETWEEN_CALLS - delta)
    _last_call_ts = time.monotonic()


def _fetch_json(url: str) -> Optional[dict]:
    _throttle()
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        log.warning("umpire_history: HTTP error for %s: %s", url, e)
        return None


def _extract_plate_ump(officials_list) -> Optional[str]:
    """Return the Home Plate umpire's fullName from a schedule officials block.

    The MLB Stats API returns entries like::

        {"official": {"id": 123, "fullName": "Doug Eddings"},
         "officialType": "Home Plate"}

    Some games have missing officials (postponed, etc.) — return None
    in that case.
    """
    if not officials_list:
        return None
    for entry in officials_list:
        t = entry.get("officialType") or ""
        if t.lower().startswith("home plate") or t.lower() == "plate":
            off = entry.get("official") or {}
            name = off.get("fullName") or ""
            return name or None
    return None


def prewarm_season(season: int) -> int:
    """Fetch and cache plate-umpire names for every game in the season.

    Returns the number of (game_pk -> ump_name) entries newly added.
    Safe to call repeatedly; already-cached entries are skipped by the
    caller via ``get_plate_ump``.
    """
    cache = _ensure_loaded()
    url = _SCHEDULE_URL.format(season=season)
    data = _fetch_json(url)
    if not data:
        log.warning("umpire_history: no schedule data for season=%d", season)
        return 0

    added = 0
    missing = 0
    for day in data.get("dates", []):
        for g in day.get("games", []):
            if g.get("gameType") != "R":
                continue
            status = (g.get("status", {}) or {}).get("detailedState", "")
            if status != "Final":
                continue
            gpk = g.get("gamePk")
            if gpk is None:
                continue
            key = str(gpk)
            if key in cache:
                continue
            name = _extract_plate_ump(g.get("officials"))
            if name:
                cache[key] = name
                added += 1
            else:
                missing += 1
    log.info("umpire_history: season=%d added=%d missing_ump=%d total_cached=%d",
             season, added, missing, len(cache))
    return added


def get_plate_ump(game_pk: int) -> Optional[str]:
    cache = _ensure_loaded()
    return cache.get(str(game_pk))


def build_ump_rpg_lookup(games, min_games: int = 10) -> dict[str, Optional[float]]:
    """For each game, compute the plate ump's career-to-date runs/game.

    Iterates games in chronological order (by game_date, game_pk).
    For each game:
      * Look up the ump's running (games_worked, total_runs) BEFORE this game.
      * If games_worked >= min_games, record career R/G; else None (so
        the caller can fall back to league mean).
      * Then update the accumulator with this game's runs.

    Returns: {event_id: career_rpg_or_None}

    Rationale for min_games: with <10 games the running mean is noisy
    enough that the predictor is better off using the league mean
    (which yields umpire_delta == 0, i.e. no distortion).
    """
    out: dict[str, Optional[float]] = {}
    totals: dict[str, list] = {}   # ump_name -> [games_worked, total_runs]

    def _sort_key(g):
        gd = g.game_date
        if hasattr(gd, "toordinal"):
            return (gd.toordinal(), g.game_pk)
        return (str(gd), g.game_pk)

    for g in sorted(games, key=_sort_key):
        name = get_plate_ump(g.game_pk)
        if not name:
            out[g.event_id] = None
            continue
        state = totals.get(name)
        if state is None or state[0] < min_games:
            out[g.event_id] = None
        else:
            out[g.event_id] = state[1] / state[0]
        # Update accumulator with this game's actual runs.
        runs = (g.home_runs or 0) + (g.away_runs or 0)
        if state is None:
            totals[name] = [1, float(runs)]
        else:
            state[0] += 1
            state[1] += float(runs)
    return out
