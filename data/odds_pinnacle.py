"""
Pinnacle odds scraper.

Uses Pinnacle's public "guest" Arcadia API — the same endpoint that
pinnacle.com itself calls from the browser. No auth beyond a guest
API key that ships with every page load. MLB is league 246.

Endpoints:
    GET /0.1/leagues/246/matchups       -> list of games (teams, start time)
    GET /0.1/leagues/246/markets/straight -> prices (ML, RL, totals)

The two are joined on `matchupId`. Matchups is the event index; markets
has one JSON object per (matchup, market type, period). We want period=0
(full game) only for our three markets, and only the top-level game
matchups (excluding series/round-robin specials).

Pinnacle returns prices in AMERICAN odds already (e.g. -112, +102) —
no decimal conversion is applied.

This is a scraper, not an official integration — expect occasional
breakage if Pinnacle changes response shapes. All fetch failures are
swallowed and return [] so the upstream predictor can fall back to DK.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from urllib import error, request

from .odds_models import OddsBook, OddsSnapshot, make_event_id
from .team_names import try_normalize_team

log = logging.getLogger(__name__)

# Public guest key Pinnacle embeds in their own frontend. If this stops
# working, open pinnacle.com in a browser, copy the X-API-Key header from
# any XHR call to guest.api.arcadia.pinnacle.com, and paste it here.
_GUEST_API_KEY = "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R"

_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
_MLB_LEAGUE_ID = 246

_HEADERS = {
    "X-API-Key": _GUEST_API_KEY,
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.pinnacle.com/",
    "Origin": "https://www.pinnacle.com",
}


def _http_get_json(url: str, timeout: float = 10.0) -> Any:
    req = request.Request(url, headers=_HEADERS)
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _parse_iso_utc(s: str) -> datetime:
    # Pinnacle returns e.g. "2026-04-19T23:05:00Z" or with explicit offset.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _as_int_price(v: Any) -> Optional[int]:
    """Pinnacle prices come in as int or numeric string; coerce safely."""
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(round(float(v)))
        except (TypeError, ValueError):
            return None


@dataclass
class _PinnacleMatchup:
    matchup_id: int
    away_team_raw: str
    home_team_raw: str
    start_utc: datetime


def _fetch_matchups() -> list[_PinnacleMatchup]:
    url = f"{_BASE}/leagues/{_MLB_LEAGUE_ID}/matchups"
    try:
        data = _http_get_json(url)
    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        log.warning("Pinnacle matchups fetch failed: %s", e)
        return []

    out: list[_PinnacleMatchup] = []
    for m in data or []:
        # Only top-level game matchups: parentId is None, type == "matchup",
        # exactly two participants (one home, one away).
        if m.get("parentId") is not None:
            continue
        if m.get("type") != "matchup":
            continue
        participants = m.get("participants") or []
        if len(participants) != 2:
            continue
        away_p = next((p for p in participants if p.get("alignment") == "away"), None)
        home_p = next((p for p in participants if p.get("alignment") == "home"), None)
        if not away_p or not home_p:
            continue
        try:
            start = _parse_iso_utc(m["startTime"])
        except (KeyError, ValueError):
            continue
        out.append(_PinnacleMatchup(
            matchup_id=m["id"],
            away_team_raw=away_p.get("name", ""),
            home_team_raw=home_p.get("name", ""),
            start_utc=start,
        ))
    return out


def _fetch_markets() -> list[dict]:
    url = f"{_BASE}/leagues/{_MLB_LEAGUE_ID}/markets/straight"
    try:
        return _http_get_json(url) or []
    except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        log.warning("Pinnacle markets fetch failed: %s", e)
        return []


def _price_for_designation(prices: list[dict], designation: str) -> Optional[int]:
    """Extract a price by designation ('home'/'away'/'over'/'under')."""
    for p in prices or []:
        if p.get("designation") == designation:
            return _as_int_price(p.get("price"))
    return None


def fetch_pinnacle_snapshots() -> list[OddsSnapshot]:
    """Fetch today's MLB slate from Pinnacle and normalize to OddsSnapshot list.

    Returns [] on any network or parse failure — callers should treat an
    empty result as "Pinnacle unavailable right now" and fall back to the
    other book.
    """
    matchups = _fetch_matchups()
    if not matchups:
        return []

    markets = _fetch_markets()

    # Only keep markets for actual game matchups (not series, round-robins,
    # or other special futures markets — those have the same league but
    # different participant counts and would pollute results). Also drop
    # alternate lines — we want the standard posted line per market.
    game_ids = {mu.matchup_id for mu in matchups}
    by_matchup: dict[int, list[dict]] = {}
    for mk in markets:
        if mk.get("period") != 0:
            continue
        m_id = mk.get("matchupId")
        if m_id is None or m_id not in game_ids:
            continue
        if mk.get("isAlternate"):
            continue
        by_matchup.setdefault(m_id, []).append(mk)

    polled_at = datetime.now(timezone.utc)
    snapshots: list[OddsSnapshot] = []

    for mu in matchups:
        home = try_normalize_team(mu.home_team_raw)
        away = try_normalize_team(mu.away_team_raw)
        if not home or not away:
            log.debug("Pinnacle: unknown teams %r / %r", mu.away_team_raw, mu.home_team_raw)
            continue

        event_id = make_event_id(mu.start_utc, away, home)
        mks = by_matchup.get(mu.matchup_id, [])

        ml = next((m for m in mks if m.get("type") == "moneyline"), None)
        spread = next((m for m in mks if m.get("type") == "spread"), None)
        total = next((m for m in mks if m.get("type") == "total"), None)

        # --- Moneyline ---
        home_ml = _price_for_designation(ml.get("prices", []), "home") if ml else None
        away_ml = _price_for_designation(ml.get("prices", []), "away") if ml else None

        # --- Run line (standard -1.5 / +1.5) ---
        home_rl_line: Optional[float] = None
        home_rl_odds: Optional[int] = None
        away_rl_odds: Optional[int] = None
        if spread:
            for p in spread.get("prices", []):
                pts = p.get("points")
                if pts not in (-1.5, 1.5):
                    continue
                price_int = _as_int_price(p.get("price"))
                if price_int is None:
                    continue
                if p.get("designation") == "home":
                    home_rl_line = float(pts)
                    home_rl_odds = price_int
                elif p.get("designation") == "away":
                    away_rl_odds = price_int

        # --- Total (main posted line) ---
        total_line: Optional[float] = None
        over_odds: Optional[int] = None
        under_odds: Optional[int] = None
        if total:
            by_points: dict[float, dict[str, int]] = {}
            for p in total.get("prices", []):
                pts = p.get("points")
                if pts is None:
                    continue
                des = p.get("designation")
                if des not in ("over", "under"):
                    continue
                pi = _as_int_price(p.get("price"))
                if pi is None:
                    continue
                by_points.setdefault(float(pts), {})[des] = pi

            # Pick the line whose over/under odds are closest to -110 —
            # that's almost always the main posted line even if Pinnacle
            # includes a few non-alternate variants.
            best_pts: Optional[float] = None
            best_score = 1e9
            for pts_key, sides in by_points.items():
                if "over" not in sides or "under" not in sides:
                    continue
                score = abs(sides["over"] + 110) + abs(sides["under"] + 110)
                if score < best_score:
                    best_score = score
                    best_pts = pts_key
            if best_pts is not None:
                total_line = best_pts
                over_odds = by_points[best_pts]["over"]
                under_odds = by_points[best_pts]["under"]

        snapshots.append(OddsSnapshot(
            book=OddsBook.PINNACLE,
            event_id=event_id,
            home_team=home,
            away_team=away,
            game_time_utc=mu.start_utc,
            home_ml=home_ml,
            away_ml=away_ml,
            home_rl_line=home_rl_line,
            home_rl_odds=home_rl_odds,
            away_rl_odds=away_rl_odds,
            total_line=total_line,
            over_odds=over_odds,
            under_odds=under_odds,
            polled_at_utc=polled_at,
            native_event_id=str(mu.matchup_id),
        ))

    return snapshots
