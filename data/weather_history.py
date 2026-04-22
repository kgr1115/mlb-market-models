"""
Historical weather client — Open-Meteo archive API (free, no key).

Back-fills the same five signals `data/weather.py` produces for live games
(temperature_f, humidity_pct, wind_speed_mph, wind_direction_deg,
wind_relative_to_cf), keyed by (home_team canonical, first-pitch UTC hour).

Why this module exists
----------------------
The backtest harness (backtest/engine.py) used to leave GameContext
weather fields at their dataclass defaults (zeros), which meant four of
eight totals families (weather, umpire, pace, market) were silenced
during training. Per project_family_signal_finding.md that silencing is
what makes totals.pitcher / totals.bullpen look anti-predictive in the
per-family regression: they're absorbing signal that should belong to
weather. Populating historical weather is step 1 of the remediation
plan recorded in that memory file.

Endpoint
--------
    https://archive-api.open-meteo.com/v1/archive
        ?latitude=X&longitude=Y
        &start_date=YYYY-MM-DD&end_date=YYYY-MM-DD
        &hourly=temperature_2m,relative_humidity_2m,
                wind_speed_10m,wind_direction_10m
        &temperature_unit=fahrenheit
        &wind_speed_unit=mph
        &timezone=UTC

The archive API is rate-limited more aggressively than the forecast
endpoint (docs suggest ~10k calls/day with no account), so we cache
aggressively to a local JSON file keyed by (team, YYYY-MM-DD). One cache
row covers a whole game day for that venue regardless of first-pitch
time — we index into the hourly array ourselves.

Usage
-----
    from data.weather_history import get_historical_weather

    w = get_historical_weather("Chicago Cubs",
                               datetime(2023, 7, 15, 20, 5, tzinfo=UTC))
    if w is not None:
        ctx.wind_direction = w.wind_relative_to_cf
        ctx.wind_speed_mph = w.wind_speed_mph
        ctx.temperature_f  = w.temperature_f
        ctx.humidity_pct   = w.humidity_pct
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Reuse venue coords + classify_wind from the live module so there's only
# one source of truth for ballpark orientation.
from .weather import VENUES, GameWeather, _classify_wind

log = logging.getLogger(__name__)


_URL = "https://archive-api.open-meteo.com/v1/archive"
_HEADERS = {
    "User-Agent": "BBP-WeatherHistory/0.1 (+local)",
    "Accept": "application/json",
}
_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
_CACHE_FILE = _CACHE_DIR / "weather_history.json"
_MIN_SECONDS_BETWEEN_CALLS = 0.12   # be polite — ~500 req/min max
_last_call_ts = 0.0


# In-memory mirror of the on-disk cache. Lazy-loaded.
_mem_cache: Optional[dict[str, dict]] = None
_dirty = False


def _load_cache() -> dict[str, dict]:
    global _mem_cache
    if _mem_cache is not None:
        return _mem_cache
    if _CACHE_FILE.exists():
        try:
            _mem_cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("weather_history: cache unreadable (%s); starting fresh", e)
            _mem_cache = {}
    else:
        _mem_cache = {}
    return _mem_cache


def save_cache() -> None:
    """Flush the in-memory cache to disk. Safe to call many times."""
    global _dirty
    if not _dirty:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_mem_cache, separators=(",", ":")),
                   encoding="utf-8")
    os.replace(tmp, _CACHE_FILE)
    _dirty = False


def _http_get_json(url: str, timeout: float = 15.0) -> dict:
    global _last_call_ts
    delta = time.time() - _last_call_ts
    if delta < _MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(_MIN_SECONDS_BETWEEN_CALLS - delta)
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        _last_call_ts = time.time()
        return json.loads(resp.read().decode("utf-8"))


def _fetch_day(home_team: str, date_str: str) -> Optional[dict]:
    """Fetch one whole game day of hourly weather for a venue.

    Returns the raw Open-Meteo hourly dict (arrays keyed by field name).
    Cached on disk. Returns None on fetch error or unknown venue.
    """
    venue = VENUES.get(home_team)
    if not venue:
        return None
    lat, lon, _ = venue

    cache = _load_cache()
    cache_key = f"{home_team}|{date_str}"
    if cache_key in cache:
        entry = cache[cache_key]
        if entry.get("ok"):
            return entry.get("data")
        # Negative cache: don't keep hammering a broken date
        return None

    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ("temperature_2m,relative_humidity_2m,"
                   "wind_speed_10m,wind_direction_10m"),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
    }
    url = f"{_URL}?{urllib.parse.urlencode(params)}"

    global _dirty
    try:
        data = _http_get_json(url)
    except Exception as e:
        log.debug("weather_history: archive fetch failed %s %s: %s",
                  home_team, date_str, e)
        # Negative-cache so we don't retry within the same session
        cache[cache_key] = {"ok": False, "reason": str(e)[:120]}
        _dirty = True
        return None

    hourly = data.get("hourly") or {}
    if not hourly.get("time"):
        cache[cache_key] = {"ok": False, "reason": "empty hourly"}
        _dirty = True
        return None

    cache[cache_key] = {"ok": True, "data": hourly}
    _dirty = True
    return hourly


def get_historical_weather(home_team: str,
                           first_pitch_utc: Optional[datetime]
                           ) -> Optional[GameWeather]:
    """Fetch the closest-to-first-pitch weather bucket for a historical game.

    Returns None on unknown venue, missing first_pitch_utc, or archive
    API failure. Callers should treat None as "leave GameContext weather
    at its dataclass default" rather than crashing.
    """
    if first_pitch_utc is None:
        return None
    if first_pitch_utc.tzinfo is None:
        first_pitch_utc = first_pitch_utc.replace(tzinfo=timezone.utc)
    else:
        first_pitch_utc = first_pitch_utc.astimezone(timezone.utc)

    venue = VENUES.get(home_team)
    if not venue:
        return None
    _, _, cf_bearing = venue

    date_str = first_pitch_utc.strftime("%Y-%m-%d")
    hourly = _fetch_day(home_team, date_str)
    if not hourly:
        return None

    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    hums = hourly.get("relative_humidity_2m") or []
    winds = hourly.get("wind_speed_10m") or []
    wdirs = hourly.get("wind_direction_10m") or []
    if not times:
        return None

    target = first_pitch_utc.replace(minute=0, second=0, microsecond=0)
    best_idx = 0
    best_delta = math.inf
    for i, ts in enumerate(times):
        try:
            t = datetime.fromisoformat(ts)
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        delta = abs((t - target).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best_idx = i

    def _pick(arr, i, default):
        if 0 <= i < len(arr) and arr[i] is not None:
            try:
                return float(arr[i])
            except (TypeError, ValueError):
                return default
        return default

    temp = _pick(temps, best_idx, 72.0)
    hum = _pick(hums, best_idx, 55.0)
    wind = _pick(winds, best_idx, 5.0)
    wdir = _pick(wdirs, best_idx, 0.0)

    return GameWeather(
        temperature_f=temp,
        humidity_pct=hum,
        precipitation_pct=0.0,
        wind_speed_mph=wind,
        wind_direction_deg=wdir,
        wind_relative_to_cf=_classify_wind(wdir, cf_bearing, wind),
    )


def _split_hourly_by_date(hourly: dict) -> dict[str, dict]:
    """Split a multi-day Open-Meteo 'hourly' response into per-day dicts
    using the same shape _fetch_day returns. Keyed by YYYY-MM-DD."""
    times  = hourly.get("time") or []
    temps  = hourly.get("temperature_2m") or []
    hums   = hourly.get("relative_humidity_2m") or []
    winds  = hourly.get("wind_speed_10m") or []
    wdirs  = hourly.get("wind_direction_10m") or []
    by_day: dict[str, dict] = {}
    for i, ts in enumerate(times):
        date_str = ts[:10]  # ISO-8601 "YYYY-MM-DDTHH:MM"
        d = by_day.setdefault(date_str, {
            "time": [], "temperature_2m": [], "relative_humidity_2m": [],
            "wind_speed_10m": [], "wind_direction_10m": [],
        })
        d["time"].append(ts)
        d["temperature_2m"].append(temps[i] if i < len(temps) else None)
        d["relative_humidity_2m"].append(hums[i] if i < len(hums) else None)
        d["wind_speed_10m"].append(winds[i] if i < len(winds) else None)
        d["wind_direction_10m"].append(wdirs[i] if i < len(wdirs) else None)
    return by_day


def _fetch_range(home_team: str, start_date: str, end_date: str,
                 dates_needed: set[str]) -> int:
    """Fetch a whole date range for one venue in a single Open-Meteo call,
    then populate the per-day cache from the response. `dates_needed`
    marks which individual days should count as 'successful fetches'.
    Returns the count of newly-populated cache keys that match dates_needed.
    """
    venue = VENUES.get(home_team)
    if not venue:
        return 0
    lat, lon, _ = venue

    params = {
        "latitude":  f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start_date,
        "end_date":   end_date,
        "hourly": ("temperature_2m,relative_humidity_2m,"
                   "wind_speed_10m,wind_direction_10m"),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit":  "mph",
        "timezone": "UTC",
    }
    url = f"{_URL}?{urllib.parse.urlencode(params)}"

    cache = _load_cache()
    global _dirty
    try:
        data = _http_get_json(url, timeout=45.0)
    except Exception as e:
        log.warning("weather_history: range fetch failed %s %s..%s: %s",
                    home_team, start_date, end_date, e)
        return 0

    hourly = data.get("hourly") or {}
    if not hourly.get("time"):
        return 0

    populated = 0
    for date_str, day_hourly in _split_hourly_by_date(hourly).items():
        cache_key = f"{home_team}|{date_str}"
        if cache_key not in cache:
            cache[cache_key] = {"ok": True, "data": day_hourly}
            _dirty = True
            if date_str in dates_needed:
                populated += 1
    return populated


def prewarm_range(home_teams_by_date: dict[str, list[str]],
                  save_every: int = 5) -> int:
    """Bulk-prefetch weather for many (date, team) pairs.

    Pass a {"YYYY-MM-DD": [home_team, ...]} mapping — useful from the
    backtest harness to front-load the API hits before the main loop.
    Returns the count of newly-populated cache keys.

    Implementation: groups dates by team and issues ONE range query per
    team covering [min_date .. max_date]. That collapses ~2000 daily
    calls per season down to ~30 (one per home venue). Any cache keys
    already present are skipped — re-running is a no-op.

    `save_every` flushes the on-disk cache every N teams so an interrupt
    doesn't lose all in-memory progress.
    """
    # Invert: team -> set of dates it appears on
    dates_by_team: dict[str, set[str]] = {}
    for date_str, teams in home_teams_by_date.items():
        for team in teams:
            dates_by_team.setdefault(team, set()).add(date_str)

    cache = _load_cache()
    ok = 0
    processed = 0
    for team in sorted(dates_by_team):
        dates_needed = dates_by_team[team]
        # Skip dates already cached for this team
        missing = {d for d in dates_needed
                   if f"{team}|{d}" not in cache}
        if not missing:
            continue
        start = min(missing)
        end   = max(missing)
        populated = _fetch_range(team, start, end, missing)
        ok += populated
        processed += 1
        if processed % save_every == 0:
            save_cache()
    save_cache()
    return ok
