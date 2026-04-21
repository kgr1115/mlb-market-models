"""
MLB betting predictors package.
"""
from .shared import (
    PitcherStats, BullpenStats, OffenseStats, DefenseStats,
    TeamStats, GameContext, MarketData, PredictionResult,
    LEAGUE,
)
from .moneyline import predict_moneyline, MONEYLINE_WEIGHTS
from .run_line import predict_run_line, RUN_LINE_WEIGHTS
from .totals import predict_totals, TOTALS_WEIGHTS

from .f5 import predict_f5, F5_WEIGHTS
from .team_totals import predict_team_total, TEAM_TOTALS_WEIGHTS
from .nrfi import predict_nrfi

from .narrow_gate import (
    gate_pick, narrow_picks, surviving_only, gate_summary,
    GATES, MarketGate,
)

from .slow_features import (
    slow_features_score,
    catcher_framing_x_ump,
    travel_timezone_shock,
    bullpen_leverage_fatigue,
    lineup_spot_shakeup,
    humidity_park_carry,
    platoon_misalignment,
    callup_hot_stretch,
    SLOW_FEATURE_WEIGHTS,
)


def predict_all(home, away, ctx, market, *, include_soft=True):
    out = {
        "moneyline": predict_moneyline(home, away, ctx, market),
        "run_line":  predict_run_line(home, away, ctx, market),
        "totals":    predict_totals(home, away, ctx, market),
    }
    if include_soft:
        out["f5"] = predict_f5(home, away, ctx, market)
        out["nrfi"] = predict_nrfi(home, away, ctx)
        out["team_total_home"] = predict_team_total("home", home, away, ctx, market, team_total_line=4.5)
        out["team_total_away"] = predict_team_total("away", home, away, ctx, market, team_total_line=4.5)
    return out


__all__ = [
    "PitcherStats", "BullpenStats", "OffenseStats", "DefenseStats",
    "TeamStats", "GameContext", "MarketData", "PredictionResult",
    "LEAGUE",
    "predict_moneyline", "predict_run_line", "predict_totals", "predict_all",
    "predict_f5", "predict_team_total", "predict_nrfi",
    "MONEYLINE_WEIGHTS", "RUN_LINE_WEIGHTS", "TOTALS_WEIGHTS",
    "F5_WEIGHTS", "TEAM_TOTALS_WEIGHTS",
    "slow_features_score", "catcher_framing_x_ump", "travel_timezone_shock",
    "bullpen_leverage_fatigue", "lineup_spot_shakeup", "humidity_park_carry",
    "platoon_misalignment", "callup_hot_stretch",
    "SLOW_FEATURE_WEIGHTS",
    "gate_pick", "narrow_picks", "surviving_only", "gate_summary",
    "GATES", "MarketGate",
]
