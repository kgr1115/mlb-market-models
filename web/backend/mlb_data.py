"""
Real MLB schedule + score fetch via the free MLB Stats API.

No API key required. Docs: https://statsapi.mlb.com/docs/

We deliberately use stdlib (urllib) so the only install required to run the
app is fastapi + uvicorn. A 60-second in-process cache avoids hammering the
API during rapid reloads.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from datetime import date, datetime
from typing import Any, Optional

try:
    from zoneinfo import ZoneInfo  # py3.9+
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

log = logging.getLogger("mlb_data")

MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"
_SCHEDULE_CACHE: dict[str, tuple[float, list[dict]]] = {}
_CACHE_TTL_SEC = 60


def today_et() -> date:
    """MLB's business day rolls over in the East. Use ET so a 10pm PT game
    on Sunday still counts as 'today' for a user on the West Coast who hits
    the app after midnight UTC."""
    if ZoneInfo is None:
        return date.today()
    return datetime.now(ZoneInfo("America/New_York")).date()


def fetch_schedule(game_date: date, timeout: float = 6.0) -> list[dict]:
    """Return a list of normalized game dicts for ``game_date``.

    Raises URLError/TimeoutError on network failure — callers should catch
    and fall back to the mock generator.
    """
    key = game_date.isoformat()
    now = time.time()
    cached = _SCHEDULE_CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL_SEC:
        return cached[1]

    params = {
        "sportId": 1,
        "date": key,
        "hydrate": "probablePitcher,linescore,venue,team",
    }
    url = f"{MLB_STATS_BASE}/schedule?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MLBBettingPredictor/0.1 (+local)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    games: list[dict] = []
    for d in raw.get("dates", []) or []:
        for g in d.get("games", []) or []:
            parsed = _parse_game(g)
            if parsed is not None:
                games.append(parsed)

    _SCHEDULE_CACHE[key] = (now, games)
    return games


def _parse_game(g: dict) -> Optional[dict]:
    teams = g.get("teams") or {}
    home_t = teams.get("home") or {}
    away_t = teams.get("away") or {}
    home_team = home_t.get("team") or {}
    away_team = away_t.get("team") or {}
    if not home_team.get("abbreviation") or not away_team.get("abbreviation"):
        return None

    home_sp = home_t.get("probablePitcher") or {}
    away_sp = away_t.get("probablePitcher") or {}

    status = g.get("status") or {}
    abstract = status.get("abstractGameState", "Preview")
    detailed = status.get("detailedState", abstract)

    # Bucket into our three UI states
    if abstract == "Final" or detailed in ("Completed Early", "Game Over"):
        bucket = "final"
    elif abstract == "Live" or detailed in ("In Progress", "Manager challenge", "Delayed"):
        bucket = "live"
    else:
        bucket = "scheduled"

    linescore = g.get("linescore") or {}

    return {
        "game_pk": g.get("gamePk"),
        "game_date_utc": g.get("gameDate"),  # ISO8601 UTC
        "status": bucket,
        "status_detail": detailed,
        "current_inning": linescore.get("currentInning"),
        "inning_half": linescore.get("inningHalf"),  # Top/Bottom
        "venue": (g.get("venue") or {}).get("name", ""),
        "home": _parse_side(home_t, home_team, home_sp),
        "away": _parse_side(away_t, away_team, away_sp),
    }


def _parse_side(side: dict, team: dict, sp: dict) -> dict:
    hand = (sp.get("pitchHand") or {}).get("code", "R")
    return {
        "abbr": team.get("abbreviation", "") or "",
        "name": team.get("name", "") or "",
        "team_id": team.get("id"),
        "score": side.get("score"),
        "pitcher_name": sp.get("fullName") or "TBD",
        "pitcher_id": sp.get("id"),
        "pitcher_hand": hand if hand in ("L", "R") else "R",
    }


# Mapping from the 3-letter team abbreviation (or our historical abbr) to the
# team's home park factors. Used when seeding GameContext from real schedule.
TEAM_PARK_FACTORS: dict[str, tuple[float, float]] = {
    # (run_factor, hr_factor) — rough 2023-2025 estimates
    "NYY": (1.02, 1.15), "LAD": (0.97, 1.02), "BOS": (1.06, 1.05),
    "HOU": (1.01, 1.04), "ATL": (1.00, 1.04), "PHI": (1.03, 1.08),
    "SD":  (0.95, 0.92), "SDP": (0.95, 0.92),
    "NYM": (0.97, 0.94),
    "TOR": (1.02, 1.04), "SEA": (0.94, 0.92), "TEX": (1.02, 1.05),
    "CHC": (0.99, 1.01), "STL": (0.97, 0.96), "MIL": (1.00, 1.03),
    "ARI": (1.02, 1.06), "AZ":  (1.02, 1.06),
    "BAL": (1.02, 1.08), "TB":  (0.96, 0.92), "TBR": (0.96, 0.92),
    "MIN": (1.00, 1.02), "CLE": (0.99, 0.96), "DET": (0.96, 0.90),
    "COL": (1.18, 1.25),
    "SF":  (0.93, 0.88), "SFG": (0.93, 0.88),
    "KC":  (0.99, 0.92), "KCR": (0.99, 0.92),
    "CIN": (1.04, 1.13), "CWS": (1.01, 1.02), "CHW": (1.01, 1.02),
    "OAK": (0.96, 0.92), "ATH": (0.96, 0.92),
    "PIT": (0.98, 0.92), "MIA": (0.96, 0.88), "WSH": (1.00, 1.02), "WAS": (1.00, 1.02),
    "LAA": (0.99, 1.00), "ANA": (0.99, 1.00),
}


# Teams playing in a fixed roof/dome (weather mostly irrelevant)
DOME_TEAMS = {"TOR", "HOU", "MIL", "ARI", "AZ", "TB", "TBR", "MIA"}


def get_park_factors(home_abbr: str) -> tuple[float, float]:
    return TEAM_PARK_FACTORS.get(home_abbr.upper(), (1.00, 1.00))


def is_domed(home_abbr: str) -> bool:
    return home_abbr.upper() in DOME_TEAMS


def game_pk_to_id(game_date: date, away_abbr: str, home_abbr: str) -> str:
    """Stable game id for our URL routing."""
    return f"{game_date.isoformat()}-{away_abbr}@{home_abbr}"
