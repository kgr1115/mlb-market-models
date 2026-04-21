"""
Injury / IL feed from MLB Stats API.

Endpoint: /api/v1/teams/{teamId}/roster?rosterType=fullRoster&hydrate=person
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from .lineups_models import InjuryEntry
from .team_names import try_normalize_team

log = logging.getLogger(__name__)

_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
_ROSTER_URL = ("https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
               "?rosterType=fullRoster&hydrate=person")
_REQUEST_TIMEOUT = 12.0
_USER_AGENT = "BBP-InjuriesClient/0.1"

_TEAM_ID_CACHE = {}


def _fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        log.warning("MLB Stats API: %s on %s", e, url)
        return {}


def _load_team_ids():
    global _TEAM_ID_CACHE
    if _TEAM_ID_CACHE:
        return _TEAM_ID_CACHE
    data = _fetch_json(_TEAMS_URL)
    for t in data.get("teams", []):
        name = try_normalize_team(t.get("name", ""))
        tid = t.get("id")
        if name and isinstance(tid, int):
            _TEAM_ID_CACHE[name] = tid
    log.info("MLB team-id cache populated: %d teams", len(_TEAM_ID_CACHE))
    return _TEAM_ID_CACHE


# Status codes that mean the player is on the IL or medically unavailable.
# Kept tight on purpose - excludes RM (reassigned to minors), SU (suspension),
# and generic roster moves because those arent injuries.
_UNAVAILABLE_CODES = {
    "D7", "D10", "D15", "D60",   # 7/10/15/60-Day IL
    "DTD",                         # Day-to-day
    "BRV",                         # Bereavement
    "PL",                          # Paternity leave
}


def fetch_team_injuries(team):
    ids = _load_team_ids()
    tid = ids.get(team)
    if tid is None:
        log.warning("fetch_team_injuries: unknown team %r", team)
        return []
    data = _fetch_json(_ROSTER_URL.format(team_id=tid))
    out = []
    for r in data.get("roster", []):
        status = (r.get("status", {}) or {})
        code = status.get("code") or ""
        if code == "A":
            continue
        if code not in _UNAVAILABLE_CODES:
            continue
        person = r.get("person", {}) or {}
        out.append(InjuryEntry(
            team=team,
            player_id=str(person.get("id", "")),
            name=person.get("fullName", ""),
            status=status.get("description") or code,
            note="",
        ))
    return out


def fetch_all_injuries():
    out = {}
    for team in _load_team_ids().keys():
        out[team] = fetch_team_injuries(team)
    total = sum(len(v) for v in out.values())
    log.info("Injuries feed: %d entries across %d teams", total, len(out))
    return out
