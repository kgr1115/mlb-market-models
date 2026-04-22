"""
Historical team-level defensive stats — Baseball Savant OAA aggregate.

Why this module exists
----------------------
Per ``project_umpire_unlock.md`` the next-highest-leverage dormant family
after weather and umpire is ML defense. ``DefenseStats`` fields (oaa,
drs, catcher_framing_runs, bsr) are all at league mean (0) in the
backtest baseline, which makes ``moneyline.defense_score`` == 0 for
every team and silences the 7%-weight defense family entirely.

This module populates **OAA only** (40% of defense_score weight). OAA
is the most defensible and widely-used Statcast defensive metric and
the Savant player-level Fielder endpoint is freely accessible and
stable. DRS / framing / BsR are left at defaults — OAA on its own
breaks the silencing and is sufficient to test the unlock.

Endpoint
--------
    https://baseballsavant.mlb.com/leaderboard/outs_above_average
        ?type=Fielder&year=YYYY&min=0&csv=true

Returns player-level OAA rows with a ``display_team_name`` column. We
aggregate by team (sum of OAAs) to get team-season OAA. Note: a player
traded mid-season is attributed to their end-of-season team in Savant's
display_team_name, which is a known limitation of team-season
aggregation.

Usage
-----
    from data.fielding_history import get_team_oaa

    oaa = get_team_oaa("Houston Astros", season=2023)  # e.g. +15 or -8
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import time
import urllib.request
from pathlib import Path
from typing import Optional

from data.team_names import try_normalize_team

log = logging.getLogger(__name__)


_URL = ("https://baseballsavant.mlb.com/leaderboard/"
        "outs_above_average?type=Fielder&year={year}&min=0&csv=true")
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0 BBP-FieldingHistory/0.1",
    "Accept": "text/csv, text/plain, */*",
}
_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_CACHE_FILE = _CACHE_DIR / "fielding_history.json"
_REQUEST_TIMEOUT = 30.0
_MIN_SECONDS_BETWEEN_CALLS = 0.5
_last_call_ts = 0.0


# In-memory cache: {str(year): {canonical_team_name: oaa_float}}
_mem_cache: Optional[dict] = None


def _ensure_loaded() -> dict:
    global _mem_cache
    if _mem_cache is not None:
        return _mem_cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _CACHE_FILE.exists():
        try:
            with _CACHE_FILE.open("r", encoding="utf-8") as f:
                _mem_cache = json.load(f)
        except Exception as e:
            log.warning("fielding_history: failed to load cache (%s); starting fresh", e)
            _mem_cache = {}
    else:
        _mem_cache = {}
    return _mem_cache


def save_cache() -> None:
    if _mem_cache is None:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_mem_cache, f, indent=1, sort_keys=True)
    os.replace(tmp, _CACHE_FILE)


def _throttle() -> None:
    global _last_call_ts
    now = time.monotonic()
    delta = now - _last_call_ts
    if delta < _MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(_MIN_SECONDS_BETWEEN_CALLS - delta)
    _last_call_ts = time.monotonic()


def _fetch_csv(url: str) -> Optional[str]:
    _throttle()
    req = urllib.request.Request(url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            body = resp.read()
    except Exception as e:
        log.warning("fielding_history: fetch failed for %s: %s", url, e)
        return None
    if body[:3] == b'\xef\xbb\xbf':
        body = body[3:]
    return body.decode("utf-8", errors="replace")


def prewarm_season(season: int) -> int:
    """Fetch + cache team-season OAA totals for ``season``.

    Returns the number of teams populated (30 on success).
    No-op if the season is already cached.
    """
    cache = _ensure_loaded()
    key = str(season)
    if key in cache and cache[key]:
        return len(cache[key])

    body = _fetch_csv(_URL.format(year=season))
    if body is None:
        log.warning("fielding_history: season=%d fetch failed", season)
        return 0

    reader = csv.DictReader(io.StringIO(body))
    team_oaa: dict[str, float] = {}
    unmapped: set[str] = set()
    for row in reader:
        raw_team = row.get("display_team_name", "").strip()
        if not raw_team:
            continue
        team = try_normalize_team(raw_team)
        if not team:
            unmapped.add(raw_team)
            continue
        try:
            oaa_s = row.get("outs_above_average", "0") or "0"
            oaa = float(oaa_s)
        except ValueError:
            continue
        team_oaa[team] = team_oaa.get(team, 0.0) + oaa

    cache[key] = team_oaa
    if unmapped:
        log.warning("fielding_history: season=%d unmapped teams: %s",
                    season, sorted(unmapped))
    log.info("fielding_history: season=%d teams=%d  oaa_range=[%.0f,%.0f]",
             season, len(team_oaa),
             min(team_oaa.values()) if team_oaa else 0.0,
             max(team_oaa.values()) if team_oaa else 0.0)
    return len(team_oaa)


def get_team_oaa(team: str, season: int) -> float:
    """Return team-season OAA (sum across all fielders). 0.0 if unknown."""
    cache = _ensure_loaded()
    s = cache.get(str(season)) or {}
    return float(s.get(team, 0.0))
