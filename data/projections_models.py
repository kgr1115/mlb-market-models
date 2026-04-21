"""
Shared dataclasses for the projections layer.

A HitterProjection / PitcherProjection is one player's rest-of-season
forecast from one projection system (ZiPS, Steamer, ATC, THE BAT).
The rollup layer aggregates these into TeamStats that the predictors
consume.

All rates are stored as decimals (e.g. 0.280, not 280 or '28.0%').
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class ProjectionSource(str, Enum):
    ZIPS = "zips"
    STEAMER = "steamer"
    ATC = "atc"          # weighted average across major systems
    THE_BAT = "thebat"   # Derek Carty's THE BAT X
    THE_BATX = "thebatx"


@dataclass
class HitterProjection:
    source: ProjectionSource
    player_id: str              # FanGraphs player id (string; FG uses ints but we keep str)
    name: str
    team: Optional[str] = None   # canonical team name if mappable; may be None pre-season

    # Volume
    pa: float = 0.0
    ab: float = 0.0

    # Rate stats (decimals)
    avg: float = 0.000
    obp: float = 0.000
    slg: float = 0.000
    iso: float = 0.000
    woba: float = 0.320
    xwoba: Optional[float] = None
    bb_pct: float = 0.085
    k_pct: float = 0.225
    barrel_pct: Optional[float] = None

    # Park-adjusted rate
    wrc_plus: float = 100.0

    # Baserunning
    bsr: float = 0.0

    # Handedness of batter ('L'/'R'/'S') — drives platoon math
    bats: str = "R"

    # ROS WAR (nice-to-have, not used by predictors today)
    war: Optional[float] = None

    fetched_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PitcherProjection:
    source: ProjectionSource
    player_id: str
    name: str
    team: Optional[str] = None

    # Volume
    gs: float = 0.0
    g: float = 0.0
    ip: float = 0.0

    # Rate stats (decimals)
    era: float = 4.20
    fip: float = 4.10
    xfip: float = 4.10
    siera: float = 4.05
    k_pct: float = 0.225
    bb_pct: float = 0.085
    k_bb_pct: float = 0.135
    hr9: float = 1.15
    whip: float = 1.28
    xwoba_against: Optional[float] = None
    csw_pct: Optional[float] = None

    # Handedness of pitcher ('L'/'R')
    throws: str = "R"

    # ROS WAR (nice-to-have)
    war: Optional[float] = None

    fetched_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PitcherDraPlus:
    """Custom DRA+-style pitcher evaluation.

    100 = league average; >100 = better than average; index is scaled
    so 15 points = ~1 standard deviation. Components are kept on the
    object for explainability (UI tooltips, confidence-score inputs).
    """
    player_id: str
    name: str
    dra_plus: float = 100.0

    # Components (all in runs-per-game units before normalization)
    raw_xwoba_against: float = 0.320
    framing_adjustment: float = 0.0   # runs per 100 PA; catcher-aware
    park_adjustment: float = 0.0
    opponent_adjustment: float = 0.0
    sample_pa: int = 0

    computed_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TeamWinProjection:
    """Output of the PECOTA-alt Monte Carlo season simulator."""
    team: str
    games_remaining: int
    mean_wins_ros: float         # expected wins in remaining games
    projected_full_season_wins: float   # current_wins + mean_wins_ros
    win_dist_pctiles: dict = field(default_factory=dict)   # {10: x, 25: y, 50: z, 75: ..., 90: ...}
    playoffs_prob: Optional[float] = None     # if a threshold was supplied
    # True single-game win probability used as the per-game seed
    per_game_win_prob: float = 0.500
    iterations: int = 10000
    computed_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
