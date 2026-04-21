"""
Live data layer for the MLB betting predictor.
"""
from .odds_models import OddsSnapshot, OddsBook, Market
from .odds_cache import OddsCache
from .odds_client import fetch_all_books, build_market_data, get_todays_events
from .line_shop import BestLine, ShoppedMarket, shop_event, shop_all, ALL_BOOKS
from .team_names import normalize_team, normalize_fg_abbr

from .projections_models import (
    HitterProjection,
    PitcherProjection,
    PitcherDraPlus,
    ProjectionSource,
    TeamWinProjection,
)
from .projections_cache import ProjectionsCache
from .projections_fangraphs import (
    fetch_hitter_projections,
    fetch_pitcher_projections,
    fetch_all as fetch_all_projections,
)
from .projections_rollup import (
    build_offense_stats,
    build_pitcher_stats,
    build_bullpen_stats,
    build_team_stats,
    PROJECTION_WEIGHT,
    ROLLING_WEIGHT,
)
from .projections_dra import compute_pitcher_dra_plus, normalize_league_relative
from .projections_monte_carlo import (
    simulate_season,
    pythag_win_prob,
    flat_schedule_probs,
)
from .rolling_stats import (
    HitterRolling, PitcherRolling, BullpenRolling, StatPick,
    preferred_hitter, preferred_pitcher, preferred_bullpen,
    build_offense_from_rolling, build_pitcher_from_rolling,
    build_bullpen_from_rolling, source_mix, WINDOW_MIN,
)

from .lineups_models import (
    GameSchedule, TeamLineup, LineupSlot, ProbableStarter,
    InjuryEntry, Official,
    LINEUP_CONFIRMED, LINEUP_PROJECTED, LINEUP_IMPLIED,
)
from .lineups_mlb import fetch_schedule, fetch_game
from .lineups_injuries import fetch_team_injuries, fetch_all_injuries
from .lineups_projected import get_projected_lineups
from .lineups_client import get_todays_games

__all__ = [
    "OddsSnapshot", "OddsBook", "Market", "OddsCache",
    "fetch_all_books", "build_market_data", "get_todays_events",
    "BestLine", "ShoppedMarket", "shop_event", "shop_all", "ALL_BOOKS",
    "normalize_team", "normalize_fg_abbr",
    "HitterProjection", "PitcherProjection", "PitcherDraPlus",
    "ProjectionSource", "TeamWinProjection",
    "ProjectionsCache",
    "fetch_hitter_projections", "fetch_pitcher_projections",
    "fetch_all_projections",
    "build_offense_stats", "build_pitcher_stats", "build_bullpen_stats",
    "build_team_stats", "PROJECTION_WEIGHT", "ROLLING_WEIGHT",
    "compute_pitcher_dra_plus", "normalize_league_relative",
    "simulate_season", "pythag_win_prob", "flat_schedule_probs",
    "HitterRolling", "PitcherRolling", "BullpenRolling", "StatPick",
    "preferred_hitter", "preferred_pitcher", "preferred_bullpen",
    "build_offense_from_rolling", "build_pitcher_from_rolling",
    "build_bullpen_from_rolling", "source_mix", "WINDOW_MIN",
    "GameSchedule", "TeamLineup", "LineupSlot", "ProbableStarter",
    "InjuryEntry", "Official",
    "LINEUP_CONFIRMED", "LINEUP_PROJECTED", "LINEUP_IMPLIED",
    "fetch_schedule", "fetch_game",
    "fetch_team_injuries", "fetch_all_injuries",
    "get_projected_lineups", "get_todays_games",
]
