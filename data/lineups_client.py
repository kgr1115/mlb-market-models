"""
Unified lineups client — MLB Stats API (confirmed) + RotoWire (projected).

Source priority per game:
    1. MLB Stats API confirmed lineup     source=LINEUP_CONFIRMED
    2. RotoWire projected lineup          source=LINEUP_PROJECTED
    3. None                                → rollup falls back to implied
                                             top-9-by-PA (source=LINEUP_IMPLIED)

Probable starters come from MLB Stats API only — RotoWire's starter info
lags the MLB feed slightly, so there's no advantage to blending.

Entry points:
    get_todays_games(date=None) -> list[GameSchedule]
        - Pulls MLB schedule, confirmed lineups, probable starters, ump
        - Optionally backfills projected lineups when confirmed not yet posted
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .lineups_mlb import fetch_schedule
from .lineups_models import GameSchedule
from .lineups_projected import get_projected_lineups

log = logging.getLogger(__name__)


def get_todays_games(
    date_yyyy_mm_dd: Optional[str] = None,
    *,
    use_projected_fallback: bool = True,
) -> list[GameSchedule]:
    """Get the day's games with the best-available lineup for each team.

    Parameters
    ----------
    date_yyyy_mm_dd : "YYYY-MM-DD" in UTC, defaults to today UTC.
    use_projected_fallback : if True (default), any team missing a
        confirmed lineup is backfilled from RotoWire's projected lineup
        when the matchup + date match.
    """
    date = date_yyyy_mm_dd or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    games = fetch_schedule(date)
    if not games:
        return games

    if not use_projected_fallback:
        return games

    # Only hit RotoWire if at least one team needs it
    need_projected = any(
        g.home_lineup is None or g.away_lineup is None for g in games
    )
    if not need_projected:
        return games

    projected = get_projected_lineups(date)
    if not projected:
        return games

    backfilled = 0
    for g in games:
        key = f"{date}|{g.away_team}|{g.home_team}"
        proj = projected.get(key)
        if not proj:
            continue
        if g.home_lineup is None and proj.get("home") is not None:
            g.home_lineup = proj["home"]
            backfilled += 1
        if g.away_lineup is None and proj.get("away") is not None:
            g.away_lineup = proj["away"]
            backfilled += 1

    if backfilled:
        log.info("Backfilled %d lineup(s) from RotoWire projected", backfilled)
    return games
