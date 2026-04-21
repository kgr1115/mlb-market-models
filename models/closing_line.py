"""
Closing-line predictor — reframe the betting target.

Rationale
---------
The memory file `project_train_test_finding.md` is the smoking gun:
predicting GAME OUTCOMES from season-length features produces no
out-of-sample edge. The DraftKings closing line already prices
everything we're computing; our model is just a noisier version of
what the market already knows.

A different target
------------------
Predict where the line will CLOSE, not who will win.

Why this works:
    1. If we can predict closer lines better than the current line, we
       profit via Closing Line Value (CLV). Over a long sample CLV
       translates 1:1 into ROI (±vig), so this is a *direct edge*.
    2. The "truth" we're training against is observable a few hours
       after market open (the close) — not weeks later (the game
       outcome). Label noise is massively reduced.
    3. We keep all our feature engineering. The mapping between
       features and the closing line is tighter than features → outcome
       because the closing line is itself a function of features.

Pipeline
--------
Opener lines are snapshotted as soon as each book posts (see
`models/opening_lines.py`). Between open and close, the line moves in
response to:
    * sharp money reacting to pending lineup / weather news
    * square volume skewing books' liability
    * cross-market corrections (totals reshuffling when starters change)

We fit (offline) a regression: delta_close = f(features_at_open). At
prediction time we output:
    * predicted close_ml, close_total, close_rl
    * predicted closing probabilities (de-vigged)
    * direction of expected movement

Consumers (bet_selection/slip.py) use these predictions to:
    * BET NOW when current line is BETTER than predicted close (snipe)
    * WAIT when current line is WORSE than predicted close (will improve)
    * SKIP bets where predicted close == current (no CLV edge)

For the initial version of this module we use a zero-coefficient stub:
the model returns predicted_close == current line. This is the right
bootstrapping behavior while we collect opener/close pairs. Downstream
code checks .coefficients_fitted — when True, the CLV signal kicks in
and flows into slip.build_slip's CLV flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from predictors import TeamStats, GameContext, MarketData
from predictors.shared import american_to_prob, remove_vig_two_way


# =============================================================================
# Prediction container
# =============================================================================

@dataclass
class ClosingLinePrediction:
    market: str
    predicted_close_home_prob: Optional[float] = None
    predicted_close_total: Optional[float] = None
    predicted_close_home_ml_odds: Optional[int] = None
    predicted_close_over_odds: Optional[int] = None
    current_implied_prob: Optional[float] = None
    current_line: Optional[float] = None
    clv_edge: float = 0.0         # predicted_close_prob - current_implied_prob
    direction: str = "neutral"     # "shorten_home" | "lengthen_home" | "up" | "down" | "neutral"
    coefficients_fitted: bool = False
    detail: dict = field(default_factory=dict)


# =============================================================================
# Fitted coefficients (start with zeros — trained offline once corpus exists)
# =============================================================================

# ---- Moneyline line-movement coefficients ----
# Prediction: delta_home_prob_close = sum(coef_i * feature_i)
# Features are keyed by name; missing features default to 0.
MOVEMENT_COEFS_ML: dict[str, float] = {
    # All zero until the offline fit produces coefficients. This maintains
    # backward compatibility; ClosingLinePrediction.clv_edge == 0 means no
    # CLV-based gating happens.
    "home_sp_siera_edge": 0.0,
    "slow_features_score": 0.0,
    "away_travel_penalty": 0.0,
    "home_new_leadoff": 0.0,
    "away_new_leadoff": 0.0,
    "lineup_confirmed_gap": 0.0,
    "ump_csw_bias": 0.0,
    # Opener-to-current movement itself is a feature — sharps have acted
    "movement_so_far": 0.0,
}

MOVEMENT_COEFS_TOTAL: dict[str, float] = {
    "humidity_excess_at_coors": 0.0,
    "wind_score": 0.0,
    "home_sp_siera_edge": 0.0,
    "away_sp_siera_edge": 0.0,
    "bullpen_leverage_depleted": 0.0,
    "temp_delta_from_70": 0.0,
    "movement_so_far": 0.0,
}

MOVEMENT_COEFS_RL: dict[str, float] = {
    "home_sp_siera_edge": 0.0,
    "away_sp_siera_edge": 0.0,
    "ml_implied_gap": 0.0,     # -1.5 is correlated to ML implied probability gap
    "movement_so_far": 0.0,
}


def coefficients_loaded() -> bool:
    """True once the offline fit has populated any coefficient in any
    of the three movement-coefficient tables."""
    for t in (MOVEMENT_COEFS_ML, MOVEMENT_COEFS_TOTAL, MOVEMENT_COEFS_RL):
        if any(abs(v) > 1e-9 for v in t.values()):
            return True
    return False


# =============================================================================
# Predictors
# =============================================================================

def _ml_features(home: TeamStats, away: TeamStats, ctx: GameContext,
                  market: MarketData) -> dict[str, float]:
    feats: dict[str, float] = {
        "home_sp_siera_edge": away.pitcher.siera - home.pitcher.siera,
        "movement_so_far": 0.0,
    }
    if market.opener_home_ml_odds is not None:
        feats["movement_so_far"] = _cents_move(
            market.opener_home_ml_odds, market.home_ml_odds)
    return feats


def _total_features(home: TeamStats, away: TeamStats, ctx: GameContext,
                     market: MarketData) -> dict[str, float]:
    feats: dict[str, float] = {
        "humidity_excess_at_coors": 0.0,
        "wind_score": 0.0,
        "home_sp_siera_edge": away.pitcher.siera - home.pitcher.siera,
        "away_sp_siera_edge": home.pitcher.siera - away.pitcher.siera,
        "bullpen_leverage_depleted": 0.0,
        "temp_delta_from_70": ctx.temperature_f - 70.0,
        "movement_so_far": 0.0,
    }
    if ctx.park_hr_factor > 1.05 and ctx.roof_status != "closed":
        feats["humidity_excess_at_coors"] = (ctx.humidity_pct - 50.0) / 50.0
    if ctx.wind_direction == "out":
        feats["wind_score"] = ctx.wind_speed_mph / 10.0
    elif ctx.wind_direction == "in":
        feats["wind_score"] = -ctx.wind_speed_mph / 10.0
    if market.opener_total is not None:
        feats["movement_so_far"] = market.total_line - market.opener_total
    return feats


def _rl_features(home: TeamStats, away: TeamStats, ctx: GameContext,
                  market: MarketData) -> dict[str, float]:
    feats: dict[str, float] = {
        "home_sp_siera_edge": away.pitcher.siera - home.pitcher.siera,
        "away_sp_siera_edge": home.pitcher.siera - away.pitcher.siera,
        "ml_implied_gap": 0.0,
        "movement_so_far": 0.0,
    }
    try:
        ih, ia = remove_vig_two_way(market.home_ml_odds, market.away_ml_odds)
        feats["ml_implied_gap"] = ih - ia
    except Exception:
        pass
    return feats


def _dot(features: dict[str, float], coefs: dict[str, float]) -> float:
    return sum(features.get(k, 0.0) * c for k, c in coefs.items())


def predict_closing_line_ml(home: TeamStats, away: TeamStats, ctx: GameContext,
                             market: MarketData) -> ClosingLinePrediction:
    feats = _ml_features(home, away, ctx, market)
    delta = _dot(feats, MOVEMENT_COEFS_ML)
    current_home_prob, _ = remove_vig_two_way(
        market.home_ml_odds, market.away_ml_odds)
    predicted_home_prob = max(0.01, min(0.99, current_home_prob + delta))
    direction = ("shorten_home" if delta > 0.005 else
                 "lengthen_home" if delta < -0.005 else "neutral")
    return ClosingLinePrediction(
        market="moneyline",
        predicted_close_home_prob=predicted_home_prob,
        current_implied_prob=current_home_prob,
        clv_edge=predicted_home_prob - current_home_prob,
        direction=direction,
        coefficients_fitted=any(abs(v) > 1e-9 for v in MOVEMENT_COEFS_ML.values()),
        detail={"features": feats, "delta_home_prob": delta},
    )


def predict_closing_line_total(home: TeamStats, away: TeamStats, ctx: GameContext,
                                market: MarketData) -> ClosingLinePrediction:
    feats = _total_features(home, away, ctx, market)
    delta_line = _dot(feats, MOVEMENT_COEFS_TOTAL)
    predicted_total = market.total_line + delta_line
    direction = ("up" if delta_line > 0.05 else
                 "down" if delta_line < -0.05 else "neutral")
    return ClosingLinePrediction(
        market="totals",
        predicted_close_total=predicted_total,
        current_line=market.total_line,
        clv_edge=delta_line,
        direction=direction,
        coefficients_fitted=any(abs(v) > 1e-9 for v in MOVEMENT_COEFS_TOTAL.values()),
        detail={"features": feats, "delta_total": delta_line},
    )


def predict_closing_line_rl(home: TeamStats, away: TeamStats, ctx: GameContext,
                             market: MarketData) -> ClosingLinePrediction:
    feats = _rl_features(home, away, ctx, market)
    delta = _dot(feats, MOVEMENT_COEFS_RL)
    # Represent RL CLV as a home-side odds delta in cents
    return ClosingLinePrediction(
        market="run_line",
        clv_edge=delta,
        direction=("shorten_home" if delta > 0.005 else
                   "lengthen_home" if delta < -0.005 else "neutral"),
        coefficients_fitted=any(abs(v) > 1e-9 for v in MOVEMENT_COEFS_RL.values()),
        detail={"features": feats},
    )


def predicted_closing_by_event(predictions_per_event: dict[str, dict]
                                ) -> dict[str, dict]:
    """Shape: {event_id: {"moneyline": predicted_home_prob, "totals": predicted_total}}

    Convenience reshape for slip.build_slip's `predicted_closing_by_event`
    argument, which expects a nested dict keyed first by event then by
    market with the model's predicted de-vigged probability.
    """
    out: dict[str, dict] = {}
    for event_id, by_market in predictions_per_event.items():
        per_event: dict[str, float] = {}
        for market_name, pred in by_market.items():
            if market_name == "moneyline" and pred.predicted_close_home_prob is not None:
                per_event["moneyline"] = pred.predicted_close_home_prob
            elif market_name == "totals" and pred.predicted_close_total is not None:
                # Convert predicted total into a de-vigged over/under prob
                # via a naive lookup table; a serious fit replaces this.
                per_event["totals"] = 0.50
        out[event_id] = per_event
    return out


def _cents_move(opener: int, current: int) -> float:
    """Normalize American-odds movement to a compact scalar."""
    return float(current - opener) / 100.0
