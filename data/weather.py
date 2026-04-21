"""
Weather client — Open-Meteo (free, no API key).

We fetch hourly forecast for each MLB venue on game day and pick the
bucket closest to first pitch (UTC). Everything is cached in-process
for 30 minutes since weather forecasts don't change materially in that
window.

Domed / retractable-roof-closed games still get a fetch (so tempearture
inside the dome can be approximated from outside ambient) but the caller
is responsible for deciding whether wind/precip matter.

Endpoint:
    https://api.open-meteo.com/v1/forecast
        ?latitude=X&longitude=Y
        &hourly=temperature_2m,relative_humidity_2m,precipitation_probability,
                wind_speed_10m,wind_direction_10m
        &temperature_unit=fahrenheit
        &wind_speed_unit=mph
        &timezone=UTC
        &forecast_days=2
"""
from __future__ import annotations

import json
import logging
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


# =============================================================================
# MLB venue coordinates (home team canonical name → lat/lon + stadium compass
# "out to center field" bearing in degrees). The bearing lets us translate a
# wind_direction_10m (compass degrees of ORIGIN) into "out to CF" vs "in from CF"
# vs "cross" relative to the ballpark orientation.
# =============================================================================

# (lat, lon, bearing_to_CF_degrees)
# Bearing = compass direction from home plate to dead center field.
# Source: park orientations cross-checked with Baseball Savant diagrams.
VENUES: dict[str, tuple[float, float, float]] = {
    "Arizona Diamondbacks":  (33.4453, -112.0667,  23),  # Chase Field
    "Atlanta Braves":        (33.8906,  -84.4677, 150),  # Truist Park
    "Baltimore Orioles":     (39.2838,  -76.6215,  36),  # Camden Yards
    "Boston Red Sox":        (42.3467,  -71.0972,  45),  # Fenway
    "Chicago Cubs":          (41.9484,  -87.6553,  36),  # Wrigley
    "Chicago White Sox":     (41.8299,  -87.6338,  35),  # Rate Field
    "Cincinnati Reds":       (39.0975,  -84.5066,  40),  # GABP
    "Cleveland Guardians":   (41.4962,  -81.6852,   0),  # Progressive
    "Colorado Rockies":      (39.7559, -104.9942,   0),  # Coors
    "Detroit Tigers":        (42.3390,  -83.0485,  30),  # Comerica
    "Houston Astros":        (29.7570,  -95.3554, 346),  # Minute Maid (roofed)
    "Kansas City Royals":    (39.0517,  -94.4803,  45),  # Kauffman
    "Los Angeles Angels":    (33.8003, -117.8827,  53),  # Angel Stadium
    "Los Angeles Dodgers":   (34.0739, -118.2400,  25),  # Dodger Stadium
    "Miami Marlins":         (25.7781,  -80.2197,  40),  # loanDepot (roofed)
    "Milwaukee Brewers":     (43.0280,  -87.9712, 132),  # American Family (roofed)
    "Minnesota Twins":       (44.9818,  -93.2776,  90),  # Target Field
    "New York Mets":         (40.7571,  -73.8458,   0),  # Citi Field
    "New York Yankees":      (40.8296,  -73.9262,  12),  # Yankee Stadium
    "Oakland Athletics":     (38.5806, -121.5008,  60),  # Sutter Health (Sacramento)
    "Philadelphia Phillies": (39.9061,  -75.1665,  20),  # Citizens Bank
    "Pittsburgh Pirates":    (40.4469,  -80.0057, 118),  # PNC Park
    "San Diego Padres":      (32.7073, -117.1566,   0),  # Petco
    "San Francisco Giants":  (37.7786, -122.3893,  96),  # Oracle
    "Seattle Mariners":      (47.5914, -122.3325,   9),  # T-Mobile (roofed)
    "St. Louis Cardinals":   (38.6226,  -90.1928,  58),  # Busch
    "Tampa Bay Rays":        (27.7682,  -82.6534,  45),  # Tropicana (dome)
    "Texas Rangers":         (32.7473,  -97.0847, 170),  # Globe Life (roofed)
    "Toronto Blue Jays":     (43.6414,  -79.3894,   0),  # Rogers Centre (roofed)
    "Washington Nationals":  (38.8730,  -77.0074,   0),  # Nationals Park
}


@dataclass
class GameWeather:
    temperature_f: float = 72.0
    humidity_pct: float = 55.0
    precipitation_pct: float = 0.0
    wind_speed_mph: float = 5.0
    wind_direction_deg: float = 0.0          # compass ORIGIN (0 = from N)
    wind_relative_to_cf: str = "none"        # "out", "in", "cross", "none"


_URL = "https://api.open-meteo.com/v1/forecast"
_HEADERS = {
    "User-Agent": "BBP-Weather/0.1 (+local)",
    "Accept": "application/json",
}
_CACHE_TTL_SEC = 30 * 60
_cache: dict[str, tuple[float, GameWeather]] = {}


def _http_get_json(url: str, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _classify_wind(wind_deg: float, cf_bearing: float,
                   wind_speed_mph: float) -> str:
    """Convert compass-origin wind degrees to "out/in/cross" relative to CF.

    wind_deg is the compass direction the wind is coming FROM (meteorological
    convention). Wind blowing *out* to CF originates from behind home plate,
    i.e. from compass direction ~= (cf_bearing + 180) mod 360.
    """
    if wind_speed_mph < 4.0:
        return "none"
    in_bearing = (cf_bearing + 180.0) % 360.0
    # Angular distance from wind origin to "from behind HP" (= blowing out)
    def adist(a: float, b: float) -> float:
        d = abs((a - b) % 360.0)
        return min(d, 360.0 - d)
    d_out = adist(wind_deg, in_bearing)   # wind from home → out to CF
    d_in = adist(wind_deg, cf_bearing)    # wind from CF → in to HP
    if d_out < 45.0:
        return "out"
    if d_in < 45.0:
        return "in"
    return "cross"


def get_weather(home_team: str, first_pitch_utc: Optional[datetime]
                ) -> Optional[GameWeather]:
    """Fetch hourly forecast for one game. None on error/unknown venue.

    Picks the hourly bucket closest to first_pitch_utc. If first_pitch_utc
    is None, uses now+3h as a rough proxy (typical pre-game prep window).
    """
    venue = VENUES.get(home_team)
    if not venue:
        log.debug("weather: no venue coords for %s", home_team)
        return None
    lat, lon, cf_bearing = venue

    when = first_pitch_utc or (datetime.now(timezone.utc))
    cache_key = f"{home_team}|{when.strftime('%Y-%m-%dT%H')}"
    now = time.time()
    cached = _cache.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL_SEC:
        return cached[1]

    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": ("temperature_2m,relative_humidity_2m,"
                   "precipitation_probability,"
                   "wind_speed_10m,wind_direction_10m"),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "UTC",
        "forecast_days": "2",
    }
    url = f"{_URL}?{urllib.parse.urlencode(params)}"

    try:
        data = _http_get_json(url)
    except Exception as e:
        log.warning("weather: Open-Meteo fetch failed for %s: %s", home_team, e)
        return None

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    hums = hourly.get("relative_humidity_2m") or []
    precs = hourly.get("precipitation_probability") or []
    winds = hourly.get("wind_speed_10m") or []
    wdirs = hourly.get("wind_direction_10m") or []

    if not times:
        return None

    # Pick bucket nearest to first pitch
    target = when.replace(minute=0, second=0, microsecond=0)
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
    prec = _pick(precs, best_idx, 0.0)
    wind = _pick(winds, best_idx, 5.0)
    wdir = _pick(wdirs, best_idx, 0.0)

    w = GameWeather(
        temperature_f=temp,
        humidity_pct=hum,
        precipitation_pct=prec,
        wind_speed_mph=wind,
        wind_direction_deg=wdir,
        wind_relative_to_cf=_classify_wind(wdir, cf_bearing, wind),
    )
    _cache[cache_key] = (now, w)
    return w
