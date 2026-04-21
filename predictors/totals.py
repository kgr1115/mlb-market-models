"""
Totals (Over/Under) predictor — 8-family framework.

Feature-family weights from Totals_Indicators_Research.docx:
    1. Starting Pitchers (combined)  25%
    2. Bullpens (combined)            15%
    3. Offense (combined)             20%
    4. Park                           15%
    5. Weather                        12%
    6. Umpire                          5%
    7. Pace & Schedule                 3%
    8. Market                          5%
"""
from __future__ import annotations

import math
from typing import Optional

from .shared import (
    LEAGUE, TeamStats, GameContext, MarketData, PredictionResult,
    american_to_prob, american_to_decimal, remove_vig_two_way,
    logistic, clamp, z, ev_per_unit,
    confidence_score, family_agreement, market_sharpness,
)


TOTALS_WEIGHTS = {
    "pitcher":    0.25,
    "bullpen":    0.15,
    "offense":    0.20,
    "park":       0.15,
    "weather":    0.12,
    "umpire":     0.05,
    "pace":       0.03,
    "market":     0.05,
}

TOTAL_STD_DEV = 3.3
UNDER_PUBLIC_NUDGE = 0.10
COORS_PARK_FACTOR_THRESHOLD = 1.15
COORS_OVER_BUMP = 0.40
MIN_EDGE = 0.025
# 75.0 matches the HIGH label boundary in shared.confidence_score(): once
# a totals pick reaches HIGH confidence we let it become a bet rather than
# suppressing it. The narrow_gate layer still applies a secondary
# edge threshold (see predictors/narrow_gate.py::GATES["totals"]).
# Historical caveat: backtest 2021-2025 pooled ROI on totals was -2.81%;
# size these bets smaller than run-line until CLV validates live.
TOTALS_MIN_CONFIDENCE_SCORE = 75.0


def pitcher_delta(home: TeamStats, away: TeamStats) -> float:
    def sp_run_prevention(p):
        return (
            0.30 * z(p.siera, "sp_siera", invert=True) +
            0.20 * z(p.xfip,  "sp_xfip",  invert=True) +
            0.20 * z(p.xwoba_against, "sp_xwoba_against", invert=True) +
            0.15 * z(p.k_bb_pct, "sp_k_bb_pct") +
            0.15 * z(p.rolling_30d_era, "sp_rolling_era", invert=True)
        )
    combined = (sp_run_prevention(home.pitcher) + sp_run_prevention(away.pitcher)) / 2
    return -combined * 0.6


def bullpen_delta(home: TeamStats, away: TeamStats) -> float:
    def bp_prev(bp):
        score = (
            0.40 * z(bp.fip, "bp_fip", invert=True) +
            0.25 * z(bp.hi_lev_k_pct, "bp_hi_lev_k_pct") +
            0.20 * z(bp.shutdown_pct, "bp_shutdown_pct")
        )
        if bp.middle_relief_fip is not None:
            score += 0.15 * z(bp.middle_relief_fip, "bp_fip", invert=True)
        else:
            score *= (1.0 / 0.85)
        return score
    combined = (bp_prev(home.bullpen) + bp_prev(away.bullpen)) / 2
    fatigue = 0.0
    for bp in (home.bullpen, away.bullpen):
        if bp.closer_pitches_last3d >= 30:
            fatigue += 0.10
        if bp.setup_pitches_last3d >= 35:
            fatigue += 0.05
    return -combined * 0.4 + fatigue


def offense_delta(home: TeamStats, away: TeamStats) -> float:
    def off_prod(o):
        return (
            0.40 * z(o.wrc_plus, "off_wrc_plus") +
            0.20 * z(o.xwOBA, "off_xwOBA") +
            0.15 * z(o.obp, "off_obp") +
            0.15 * z(o.iso, "off_iso") +
            0.05 * z(o.barrel_pct, "off_barrel_pct") +
            0.05 * z(o.k_pct, "off_k_pct", invert=True)
        )
    combined = (off_prod(home.offense) + off_prod(away.offense)) / 2
    return combined * 0.55


def park_delta(ctx: GameContext, baseline: float) -> float:
    return baseline * (ctx.park_run_factor - 1.0)


def weather_delta(ctx: GameContext) -> float:
    if ctx.roof_status == "closed":
        return 0.0
    delta = 0.0
    if ctx.wind_direction == "out":
        if ctx.wind_speed_mph >= 15:
            delta += 1.25
        elif ctx.wind_speed_mph >= 10:
            delta += 0.65
        elif ctx.wind_speed_mph >= 5:
            delta += 0.25
    elif ctx.wind_direction == "in":
        if ctx.wind_speed_mph >= 15:
            delta -= 1.10
        elif ctx.wind_speed_mph >= 10:
            delta -= 0.55
        elif ctx.wind_speed_mph >= 5:
            delta -= 0.20
    elif ctx.wind_direction == "cross" and ctx.wind_speed_mph >= 15:
        delta -= 0.10
    temp_delta = (ctx.temperature_f - 70.0) * 0.04
    temp_delta = clamp(temp_delta, -0.80, 0.80)
    delta += temp_delta
    if ctx.humidity_pct < 30:
        delta += 0.05
    elif ctx.humidity_pct > 80:
        delta -= 0.05
    return delta


def umpire_delta(ctx: GameContext) -> float:
    mean = LEAGUE["ump_runs_per_game"]
    return clamp(ctx.ump_runs_per_game - mean, -1.0, 1.0)


def pace_delta(home: TeamStats, away: TeamStats, ctx: GameContext) -> float:
    d = 0.0
    if ctx.day_game and ctx.temperature_f >= 85:
        d += 0.10
    if ctx.doubleheader:
        d -= 0.15
    if ctx.extra_innings_prev_game:
        d -= 0.15
    return d


def market_delta(market: MarketData) -> float:
    if market.opener_total is None:
        return 0.0
    diff = market.total_line - market.opener_total
    if market.public_ticket_pct_over is not None:
        pub = market.public_ticket_pct_over
        if diff > 0 and pub < 0.45:
            return +0.15
        if diff < 0 and pub > 0.55:
            return -0.15
    return clamp(diff * 0.3, -0.2, 0.2)


def predict_totals(
    home: TeamStats,
    away: TeamStats,
    ctx: GameContext,
    market: MarketData,
    min_edge: float = MIN_EDGE,
    under_public_nudge: float = UNDER_PUBLIC_NUDGE,
) -> PredictionResult:
    assert home.is_home is True and away.is_home is False

    baseline = 2.0 * LEAGUE["league_runs_per_game_per_team"]

    d_pitcher = pitcher_delta(home, away)  * TOTALS_WEIGHTS["pitcher"] / 0.25
    d_bullpen = bullpen_delta(home, away)  * TOTALS_WEIGHTS["bullpen"] / 0.15
    d_offense = offense_delta(home, away)  * TOTALS_WEIGHTS["offense"] / 0.20
    d_park    = park_delta(ctx, baseline)  * TOTALS_WEIGHTS["park"]    / 0.15
    d_weather = weather_delta(ctx)         * TOTALS_WEIGHTS["weather"] / 0.12
    d_umpire  = umpire_delta(ctx)          * TOTALS_WEIGHTS["umpire"]  / 0.05
    d_pace    = pace_delta(home, away, ctx) * TOTALS_WEIGHTS["pace"]   / 0.03
    d_market  = market_delta(market)       * TOTALS_WEIGHTS["market"]  / 0.05

    projected_total = (
        baseline + d_pitcher + d_bullpen + d_offense + d_park +
        d_weather + d_umpire + d_pace + d_market + ctx.league_run_drift
    )

    coors_flag = ctx.park_run_factor >= COORS_PARK_FACTOR_THRESHOLD
    std_dev = TOTAL_STD_DEV + (0.6 if coors_flag else 0.0)
    if coors_flag:
        projected_total += COORS_OVER_BUMP

    projected_total_for_pick = projected_total - under_public_nudge

    p_over = 1.0 - _norm_cdf((market.total_line - projected_total_for_pick) / std_dev)
    p_under = 1.0 - p_over

    implied_over, implied_under = remove_vig_two_way(market.over_odds, market.under_odds)
    edge_over = p_over - implied_over
    edge_under = p_under - implied_under

    # Market signal lives in market_delta (5% weight family) — no
    # additional market_sharpness bump here to avoid double-counting.

    if edge_over >= edge_under and edge_over >= min_edge:
        side, odds, prob, implied, edge = f"OVER {market.total_line}", market.over_odds, p_over, implied_over, edge_over
        direction = +1
    elif edge_under > edge_over and edge_under >= min_edge:
        side, odds, prob, implied, edge = f"UNDER {market.total_line}", market.under_odds, p_under, implied_under, edge_under
        direction = -1
    else:
        side = "NO BET"
        odds = None
        if edge_over >= edge_under:
            prob, implied, edge, direction = p_over, implied_over, edge_over, +1
        else:
            prob, implied, edge, direction = p_under, implied_under, edge_under, -1

    certainty = 1.0
    if not (home.starter_confirmed and away.starter_confirmed):
        certainty -= 0.35
    if not (home.lineup_confirmed and away.lineup_confirmed):
        certainty -= 0.10
    if ctx.wind_direction in ("out", "in") and ctx.wind_speed_mph >= 15:
        certainty -= 0.10
    certainty = clamp(certainty, 0.4, 1.0)

    # Every d_* is the signed CONTRIBUTION to projected_total — positive
    # means the family pushes runs up (supports OVER, direction +1).
    # d_pitcher and d_bullpen are already negative when pitching/pen is
    # strong, so they map DIRECTLY to OVER/UNDER direction without a sign
    # flip. (Earlier versions negated them here, which double-flipped and
    # made family_agreement score the wrong direction for those two.)
    family_dirs = {
        "pitcher": d_pitcher,
        "bullpen": d_bullpen,
        "offense": d_offense,
        "park":    d_park,
        "weather": d_weather,
        "umpire":  d_umpire,
        "pace":    d_pace,
        "market":  d_market,
    }
    agree = family_agreement(family_dirs, direction)

    variance_pen = 0.0
    if coors_flag:
        variance_pen += 0.30
    if ctx.wind_direction in ("out", "in") and ctx.wind_speed_mph >= 15:
        variance_pen += 0.15

    extra = 0.0
    if abs(projected_total - market.total_line) < 0.3:
        extra += 0.20
    if direction == -1:
        if home.bullpen.closer_pitches_last3d >= 35 or away.bullpen.closer_pitches_last3d >= 35:
            extra += 0.25

    conf, label = confidence_score(
        edge=edge,
        family_agreement=agree,
        input_certainty=certainty,
        variance_penalty=variance_pen,
        extra_penalty=extra,
    )

    if not globals().get("_BYPASS_GATES", False):
        if side != "NO BET" and conf < TOTALS_MIN_CONFIDENCE_SCORE:
            side = "NO BET"
            odds = None

    ev = ev_per_unit(prob, odds) if odds is not None else 0.0
    pick_str = f"{side} {_fmt_odds(odds)}" if odds is not None else "NO BET"

    return PredictionResult(
        market="totals",
        pick=pick_str,
        odds=odds,
        model_prob=prob,
        implied_prob=implied,
        edge=edge,
        confidence=conf,
        confidence_label=label,
        expected_value_per_unit=ev,
        detail={
            "diff_by_family": {
                "pitcher": d_pitcher, "bullpen": d_bullpen,
                "offense": d_offense, "park": d_park,
                "weather": d_weather, "umpire": d_umpire,
                "pace": d_pace, "market": d_market,
            },
            "baseline_total": baseline,
            "projected_total": projected_total,
            "projected_total_for_pick": projected_total_for_pick,
            "std_dev": std_dev,
            "coors_overlay": coors_flag,
            "p_over": p_over,
            "p_under": p_under,
            "implied_over_prob": implied_over,
            "implied_under_prob": implied_under,
            "variance_penalty": variance_pen,
            "family_agreement": agree,
            "input_certainty": certainty,
        },
    )


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _fmt_odds(odds):
    if odds is None:
        return ""
    return f"+{odds}" if odds > 0 else str(odds)
