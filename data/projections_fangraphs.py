"""
FanGraphs rest-of-season projections loader.

FanGraphs publishes ZiPS, Steamer, ATC, and THE BAT projections as JSON
through a public internal API that backs their projections UI. No auth.

Endpoint:
    GET https://www.fangraphs.com/api/projections
        ?stats=bat|pit     # hitters vs pitchers
        &type=<system>     # rzips, steamerr, ratc, rthebat, rthebatx
        &team=0            # all teams
        &lg=all
        &players=0         # all players

Returns a JSON list where each row is one player with columns like
PA, wOBA, wRC+, etc. Field names differ between hitters and pitchers.

Rest-of-season flavors use the 'r'-prefixed type codes:
    rzips, steamerr, ratc, rthebat, rthebatx

Pre-season / full-season flavors (if you want them later): zips, steamer,
atc, thebat, thebatx.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from urllib import error, request

from .projections_models import (
    HitterProjection,
    PitcherProjection,
    ProjectionSource,
)
from .team_names import normalize_fg_abbr

log = logging.getLogger(__name__)


_BASE = "https://www.fangraphs.com/api/projections"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.fangraphs.com/projections",
}


# Map ProjectionSource -> FanGraphs ROS type codes
_ROS_TYPE: dict[ProjectionSource, str] = {
    ProjectionSource.ZIPS:     "rzips",
    ProjectionSource.STEAMER:  "steamerr",
    ProjectionSource.ATC:      "ratc",
    ProjectionSource.THE_BAT:  "rthebat",
    ProjectionSource.THE_BATX: "rthebatx",
}


def _http_get_json(url: str, timeout: float = 20.0) -> Any:
    req = request.Request(url, headers=_HEADERS)
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _float(row: dict, key: str, default: Optional[float] = None) -> Optional[float]:
    v = row.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _str_or_none(row: dict, key: str) -> Optional[str]:
    v = row.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def fetch_hitter_projections(source: ProjectionSource) -> list[HitterProjection]:
    """Pull ROS hitter projections from FanGraphs. Returns [] on any failure."""
    typ = _ROS_TYPE.get(source)
    if typ is None:
        raise ValueError(f"Unsupported projection source for hitters: {source}")

    url = f"{_BASE}?stats=bat&type={typ}&team=0&lg=all&players=0"
    try:
        data = _http_get_json(url)
    except (error.URLError, error.HTTPError, TimeoutError,
            json.JSONDecodeError) as e:
        log.warning("FanGraphs %s hitters fetch failed: %s", typ, e)
        return []
    if not isinstance(data, list):
        log.warning("FanGraphs %s hitters: unexpected payload type %s", typ, type(data))
        return []

    now = datetime.now(timezone.utc)
    out: list[HitterProjection] = []
    for row in data:
        pid = _str_or_none(row, "playerid") or _str_or_none(row, "playerids")
        name = _str_or_none(row, "PlayerName")
        if not pid or not name:
            continue
        team_abbr = _str_or_none(row, "Team")
        team = normalize_fg_abbr(team_abbr) if team_abbr else None
        out.append(HitterProjection(
            source=source,
            player_id=pid,
            name=name,
            team=team,
            pa=_float(row, "PA", 0.0) or 0.0,
            ab=_float(row, "AB", 0.0) or 0.0,
            avg=_float(row, "AVG", 0.0) or 0.0,
            obp=_float(row, "OBP", 0.0) or 0.0,
            slg=_float(row, "SLG", 0.0) or 0.0,
            iso=_float(row, "ISO", 0.0) or 0.0,
            woba=_float(row, "wOBA", 0.320) or 0.320,
            xwoba=None,   # projections don't include xwOBA
            bb_pct=_float(row, "BB%", 0.085) or 0.085,
            k_pct=_float(row, "K%", 0.225) or 0.225,
            barrel_pct=None,
            wrc_plus=_float(row, "wRC+", 100.0) or 100.0,
            bsr=_float(row, "BaseRunning", 0.0) or 0.0,
            bats="R",   # not in this payload; defaulted
            war=_float(row, "WAR"),
            fetched_at_utc=now,
        ))
    return out


def fetch_pitcher_projections(source: ProjectionSource) -> list[PitcherProjection]:
    """Pull ROS pitcher projections from FanGraphs. Returns [] on any failure."""
    typ = _ROS_TYPE.get(source)
    if typ is None:
        raise ValueError(f"Unsupported projection source for pitchers: {source}")

    url = f"{_BASE}?stats=pit&type={typ}&team=0&lg=all&players=0"
    try:
        data = _http_get_json(url)
    except (error.URLError, error.HTTPError, TimeoutError,
            json.JSONDecodeError) as e:
        log.warning("FanGraphs %s pitchers fetch failed: %s", typ, e)
        return []
    if not isinstance(data, list):
        return []

    now = datetime.now(timezone.utc)
    out: list[PitcherProjection] = []
    for row in data:
        pid = _str_or_none(row, "playerid") or _str_or_none(row, "playerids")
        name = _str_or_none(row, "PlayerName")
        if not pid or not name:
            continue
        team_abbr = _str_or_none(row, "Team")
        team = normalize_fg_abbr(team_abbr) if team_abbr else None
        out.append(PitcherProjection(
            source=source,
            player_id=pid,
            name=name,
            team=team,
            gs=_float(row, "GS", 0.0) or 0.0,
            g=_float(row, "G", 0.0) or 0.0,
            ip=_float(row, "IP", 0.0) or 0.0,
            era=_float(row, "ERA", 4.20) or 4.20,
            fip=_float(row, "FIP", 4.10) or 4.10,
            xfip=_float(row, "FIP", 4.10) or 4.10,   # xFIP not exposed; use FIP
            siera=_float(row, "FIP", 4.05) or 4.05,  # SIERA not exposed; use FIP
            k_pct=_float(row, "K%", 0.225) or 0.225,
            bb_pct=_float(row, "BB%", 0.085) or 0.085,
            k_bb_pct=_float(row, "K-BB%", 0.135) or 0.135,
            hr9=_float(row, "HR/9", 1.15) or 1.15,
            whip=_float(row, "WHIP", 1.28) or 1.28,
            xwoba_against=None,   # projection systems don't publish
            csw_pct=None,
            throws="R",           # not in payload; defaulted
            war=_float(row, "WAR"),
            fetched_at_utc=now,
        ))
    return out


def fetch_all(sources: Iterable[ProjectionSource] = (ProjectionSource.ZIPS,
                                                     ProjectionSource.STEAMER)
              ) -> dict[str, tuple[list[HitterProjection], list[PitcherProjection]]]:
    """Pull hitter + pitcher projections for each requested source.

    Default: ZiPS and Steamer (two independently-maintained projection
    systems — combining them via simple average is how ATC/FanGraphs Depth
    Charts already work).
    """
    out: dict[str, tuple[list[HitterProjection], list[PitcherProjection]]] = {}
    for src in sources:
        hitters = fetch_hitter_projections(src)
        pitchers = fetch_pitcher_projections(src)
        out[src.value] = (hitters, pitchers)
        log.info("FanGraphs %s: %d hitters, %d pitchers", src.value,
                 len(hitters), len(pitchers))
    return out
