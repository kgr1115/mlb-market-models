"""
MLB Stats API client — schedule + probable pitchers + confirmed lineups
+ home plate umpire.

No API key required. Endpoint:
    https://statsapi.mlb.com/api/v1/schedule
        ?sportId=1
        &date=YYYY-MM-DD
        &hydrate=probablePitcher(note),lineups,officials,venue

We hit one endpoint per day, cache the raw JSON for a short TTL so
repeat calls during a session are free, and return a list of
GameSchedule objects whose event_ids line up with our odds scraper.

Confirmed lineups usually post ~2 hours before first pitch for day
games and ~2-3 hours before for nights. Before that window, the
`lineups` hydration returns empty and we leave TeamLineup unset —
the caller can then decide whether to fall back to the projected
scraper or let the rollup use implied top-9-by-PA.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional

from .lineups_models import (
    GameSchedule, LineupSlot, LINEUP_CONFIRMED, Official,
    ProbableStarter, TeamLineup,
)
from .odds_models import make_event_id
from .team_names import normalize_team, try_normalize_team

log = logging.getLogger(__name__)

_SCHEDULE_URL = (
    "https://statsapi.mlb.com/api/v1/schedule"
    "?sportId=1&date={date}"
    "&hydrate=probablePitcher(note),lineups,officials,venue"
)
_REQUEST_TIMEOUT = 12.0
_USER_AGENT = "BBP-LineupsClient/0.1"


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        log.warning("MLB Stats API HTTP %s on %s", e.code, url)
        return {}
    except urllib.error.URLError as e:
        log.warning("MLB Stats API network error on %s: %s", url, e.reason)
        return {}


def _parse_iso(ts: str) -> datetime:
    # MLB returns e.g. "2026-04-20T19:10:00Z"
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _parse_pitcher(team: str, node: Optional[dict],
                   confirmed: bool) -> Optional[ProbableStarter]:
    if not node:
        return None
    pid = str(node.get("id", ""))
    if not pid:
        return None
    throws = (node.get("pitchHand", {}) or {}).get("code", "R")
    return ProbableStarter(
        team=team,
        player_id=pid,
        name=node.get("fullName") or node.get("nameFirstLast") or "",
        throws=throws,
        confirmed=confirmed,
    )


def _parse_lineup(team: str, players: Optional[list]) -> Optional[TeamLineup]:
    if not players:
        return None
    slots: list[LineupSlot] = []
    for idx, p in enumerate(players, start=1):
        pid = str(p.get("id", ""))
        if not pid:
            continue
        slots.append(LineupSlot(
            order=idx,
            player_id=pid,
            name=p.get("fullName") or "",
            position=(p.get("primaryPosition", {}) or {}).get("abbreviation", ""),
            bats=(p.get("batSide", {}) or {}).get("code", "R"),
        ))
    if not slots:
        return None
    return TeamLineup(team=team, source=LINEUP_CONFIRMED, slots=slots)


def _parse_home_plate_ump(officials: Optional[list]) -> Optional[Official]:
    if not officials:
        return None
    for o in officials:
        if (o.get("officialType") or "").lower() == "home plate":
            off = o.get("official", {}) or {}
            return Official(name=off.get("fullName", ""), position="HP")
    return None


def fetch_schedule(date_yyyy_mm_dd: Optional[str] = None
                   ) -> list[GameSchedule]:
    """Return all MLB games scheduled on `date` (defaults to today, UTC).

    Games with unknown teams are skipped with a warning — never silently
    mis-mapped — so scraper bugs surface instead of producing bad event_ids.
    """
    date = date_yyyy_mm_dd or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = _SCHEDULE_URL.format(date=date)
    data = _fetch_json(url)

    out: list[GameSchedule] = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            try:
                home_raw = (g.get("teams", {}).get("home", {})
                              .get("team", {}).get("name", ""))
                away_raw = (g.get("teams", {}).get("away", {})
                              .get("team", {}).get("name", ""))
                home = try_normalize_team(home_raw)
                away = try_normalize_team(away_raw)
                if not home or not away:
                    log.warning("MLB schedule: unknown team pair %r / %r, skipping",
                                home_raw, away_raw)
                    continue

                game_time = _parse_iso(g["gameDate"])
                game_pk = int(g["gamePk"])

                home_starter = _parse_pitcher(
                    home,
                    g["teams"]["home"].get("probablePitcher"),
                    confirmed=bool(g["teams"]["home"].get("probablePitcher")),
                )
                away_starter = _parse_pitcher(
                    away,
                    g["teams"]["away"].get("probablePitcher"),
                    confirmed=bool(g["teams"]["away"].get("probablePitcher")),
                )

                lineups_node = g.get("lineups", {}) or {}
                home_lineup = _parse_lineup(home, lineups_node.get("homePlayers"))
                away_lineup = _parse_lineup(away, lineups_node.get("awayPlayers"))

                hp_ump = _parse_home_plate_ump(g.get("officials"))

                out.append(GameSchedule(
                    event_id=make_event_id(game_time, away, home),
                    game_pk=game_pk,
                    game_time_utc=game_time,
                    home_team=home,
                    away_team=away,
                    venue=(g.get("venue", {}) or {}).get("name", ""),
                    status=(g.get("status", {}) or {}).get("detailedState",
                                                            "Scheduled"),
                    home_starter=home_starter,
                    away_starter=away_starter,
                    home_lineup=home_lineup,
                    away_lineup=away_lineup,
                    home_plate_ump=hp_ump,
                ))
            except Exception as e:
                log.exception("MLB schedule: failed to parse game, continuing: %s", e)
                continue

    log.info("MLB schedule %s: %d games loaded", date, len(out))
    return out


def fetch_game(event_id: str) -> Optional[GameSchedule]:
    """Convenience: pull a single game by our canonical event_id.

    Pulls that day's full schedule (cheap, one HTTP call) and filters.
    """
    # event_id format: "YYYY-MM-DD|AWAY|HOME"
    try:
        date, _, _ = event_id.split("|")
    except ValueError:
        return None
    for g in fetch_schedule(date):
        if g.event_id == event_id:
            return g
    return None
