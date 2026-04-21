"""
Dataclasses for lineups / probable pitchers / officials / injuries.

These sit between the MLB Stats API (and projected-lineup scrapers) and
the predictors' TeamStats consumer. Everything is normalized to our
canonical team name (via team_names.normalize_team) so downstream code
can cross-join the lineups feed with odds/projections on the same
event_id and team keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# How the lineup was obtained — used by predict_all() to set a
# lineup_certainty factor in the confidence calculation.
#   "confirmed"  — posted by the team / MLB Stats API
#   "projected"  — scraped from RotoWire / Lineups.com pre-official
#   "implied"    — no lineup available yet; rollup used top-9 by PA
LINEUP_CONFIRMED = "confirmed"
LINEUP_PROJECTED = "projected"
LINEUP_IMPLIED = "implied"


@dataclass
class LineupSlot:
    """One batting-order slot."""
    order: int              # 1..9
    player_id: str          # MLB Stats API person id (as string)
    name: str
    position: str = ""      # e.g., "CF", "DH"
    bats: str = "R"         # "L", "R", "S"


@dataclass
class TeamLineup:
    """A team's batting order for one game."""
    team: str                               # canonical MLB name
    source: str = LINEUP_IMPLIED            # see constants above
    slots: list[LineupSlot] = field(default_factory=list)

    @property
    def is_confirmed(self) -> bool:
        return self.source == LINEUP_CONFIRMED

    @property
    def player_ids(self) -> list[str]:
        return [s.player_id for s in self.slots]


@dataclass
class ProbableStarter:
    """One team's probable starting pitcher."""
    team: str
    player_id: str
    name: str
    throws: str = "R"                       # "L", "R"
    confirmed: bool = False                 # True when MLB publishes the official


@dataclass
class InjuryEntry:
    """One player on the IL or day-to-day."""
    team: str
    player_id: str
    name: str
    status: str = ""                        # "10-Day IL", "60-Day IL", "Day-to-Day", etc.
    note: str = ""                          # free-text detail when available


@dataclass
class Official:
    """Game official (home plate umpire is the one we care about for totals)."""
    name: str
    position: str = "HP"                    # "HP", "1B", "2B", "3B"


@dataclass
class GameSchedule:
    """Everything we know about one scheduled game."""
    event_id: str                           # matches our odds event_id
    game_pk: int                            # MLB Stats API gamePk
    game_time_utc: datetime
    home_team: str                          # canonical
    away_team: str
    venue: str = ""
    status: str = "Scheduled"               # "Scheduled", "In Progress", "Final", "Postponed"

    home_starter: Optional[ProbableStarter] = None
    away_starter: Optional[ProbableStarter] = None
    home_lineup: Optional[TeamLineup] = None
    away_lineup: Optional[TeamLineup] = None
    home_plate_ump: Optional[Official] = None

    def lineup_for(self, team: str) -> Optional[TeamLineup]:
        if team == self.home_team:
            return self.home_lineup
        if team == self.away_team:
            return self.away_lineup
        return None

    def starter_for(self, team: str) -> Optional[ProbableStarter]:
        if team == self.home_team:
            return self.home_starter
        if team == self.away_team:
            return self.away_starter
        return None
