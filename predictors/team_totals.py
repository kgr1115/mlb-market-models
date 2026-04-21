"""
Team Totals predictor (O/U how many runs ONE team scores).

Why team totals are softer than game totals
-------------------------------------------
1. Lower handle → books hang lines with less precision. FanDuel and
   DraftKings frequently disagree by 0.5 runs on the same team total.
2. The line is a *product* of offense vs. pitcher. Most books derive it
   from two dependent pieces and don't recompute it when one input
   updates late (e.g. opponent's starter scratch → SP is now worse, but
   the total sits).
3. Correlation-baked vig is worse on team totals than on game totals.
   You can sometimes find -105/-105 two-way markets instead of -115/-115.

Scoring
-------
Pure offense × opponent-pitcher × park × weather × (ump runs per game),
with no other-team offense information. Produces an *expected runs for
this team* value μ, then:

    P(OVER line) = 1 - Poisson(line - 0.5 ; μ)   if line is a half
    P(OVER line) = 1 - sum_{k<=line-1} Poisson(k ; μ)  otherwise

We model the per-team runs as Poisson(μ) truncated at a small upper
tail; runs are overdispersed vs. pure Poisson so we widen via an
empirical dispersion factor derived from season-level runs variance.

Output bet is the side whose model_prob beats implied_prob by MIN_EDGE.
"""
from __future__ import annotations

import math
from typing import Literal, Optional

from .shared import (
    LEAGUE, TeamStats, GameContext, MarketData, PredictionResult,
    american_to_prob, remove_vig_two_way, clamp, z, ev_per_unit,
    confidence_score, family_agreement,
)


# Offense (60%), opp SP (25%), context (park+weather+ump) (15%)
TEAM_TOTALS_WEIGHTS = {
    "offense":  0.60,
    "opp_sp":   0.25,
    "context":  0.15,
}

# Runs-per-game neutralization: scale 1.0 unit of edge_score → +0.40 runs
EDGE_TO_RUNS = 0.40

# Variance-inflation factor for per-team runs (Poisson-mix widening)
DISPERSION = 1.35

MIN_EDGE = 0.030


def offense_expected_runs_rating(team: TeamStats) -> float:
    """A unit-less offense rating centered at 0 (league avg)."""
    o = team.offense
    s = 0.0
    s += 0.40 * z(o.wrc_plus, "off_wrc_plus")
    s += 0.25 * z(o.xwOBA, "off_xwOBA")
    s += 0.15 * z(o.iso, "off_iso")
    s += 0.10 * z(o.barrel_pct, "off_barrel_pct")
    s += 0.05 * z(o.obp, "off_obp")
    s += 0.05 * z(o.k_pct, "off_k_pct", invert=True)
    return s


def opp_sp_runs_suppression(opp: TeamStats) -> float:
    """Positive = opposing SP suppresses runs (good for UNDER)."""
    p = opp.pitcher
    s = 0.0
    s += 0.40 * z(p.siera, "sp_siera", invert=True)
    s += 0.25 * z(p.xwoba_against, "sp_xwoba_against", invert=True)
    s += 0.20 * z(p.k_bb_pct, "sp_k_bb_pct")
    s += 0.15 * z(p.rolling_30d_era, "sp_rolling_era", invert=True)
    return s


def team_total_context_score(ctx: GameContext) -> float:
    """Park, weather (temperature/wind), ump-specific runs lift."""
    s = 0.0
    s += 1.5 * (ctx.park_run_factor - 1.00)
    if ctx.wind_direction == "out":
        s += 0.02 * ctx.wind_speed_mph
    elif ctx.wind_direction == "in":
        s -= 0.02 * ctx.wind_speed_mph
    # Temp: +5 per 10°F above 70 (ball carries further)
    s += (ctx.temperature_f - 70.0) / 200.0
    # Ump: known hitter-friendly = +0.10 per RPG above league
    s += (ctx.ump_runs_per_game - LEAGUE["ump_runs_per_game"]) * 0.30
    return s


def expected_team_runs(team: TeamStats, opp: TeamStats, ctx: GameContext) -> float:
    """Blend the three family scores into an expected-runs point estimate."""
    off = offense_expected_runs_rating(team)
    sp_suppress = opp_sp_runs_suppression(opp)
    ctx_add = team_total_context_score(ctx)

    # Edge score: positive favors OVER
    edge = (TEAM_TOTALS_WEIGHTS["offense"] * off
            - TEAM_TOTALS_WEIGHTS["opp_sp"] * sp_suppress
            + TEAM_TOTALS_WEIGHTS["context"] * ctx_add)

    baseline = LEAGUE["league_runs_per_game_per_team"]
    mu = max(0.5, baseline + edge * EDGE_TO_RUNS)
    return mu


def _poisson_pmf(k: int, mu: float) -> float:
    """mu^k e^-mu / k!"""
    if k < 0:
        return 0.0
    try:
        return math.exp(k * math.log(mu) - mu - math.lgamma(k + 1))
    except ValueError:
        return 0.0


def _neg_binom_cdf_le(x: int, mu: float, disp: float = DISPERSION) -> float:
    """P(runs <= x) using negative-binomial approximation of
    overdispersed team runs. We fix the variance as disp * mu."""
    if disp <= 1.0:
        # Fall back to Poisson
        cum = 0.0
        for k in range(x + 1):
            cum += _poisson_pmf(k, mu)
        return clamp(cum, 0.0, 1.0)
    # NegBinom parameters with mean mu and variance disp*mu
    variance = disp * mu
    p = mu / variance
    r = mu * p / (1 - p)
    # CDF via sum of pmfs up to x
    cum = 0.0
    prob_at_k = (1 - p) ** r       # P(X=0) when k=0 in the neg-binom(r, p) form
    cum += prob_at_k
    for k in range(1, x + 1):
        # ratio P(X=k) / P(X=k-1) = ((r + k - 1) / k) * (1 - p)
        prob_at_k *= ((r + k - 1) / k) * (1 - p)
        cum += prob_at_k
    return clamp(cum, 0.0, 1.0)


def _prob_over(line: float, mu: float) -> float:
    """P(runs > line) with half-point-aware rounding."""
    if line == int(line):
        # Integer line: OVER = runs >= line + 1
        return 1.0 - _neg_binom_cdf_le(int(line), mu)
    # Half line (e.g. 3.5): OVER = runs >= ceil(line)
    return 1.0 - _neg_binom_cdf_le(int(math.floor(line)), mu)


def predict_team_total(
    team_side: Literal["home", "away"],
    home: TeamStats,
    away: TeamStats,
    ctx: GameContext,
    market: MarketData,
    team_total_line: float,
    over_odds: int = -110,
    under_odds: int = -110,
    min_edge: float = MIN_EDGE,
) -> PredictionResult:
    """Predict OVER/UNDER the team total for `team_side`."""
    assert home.is_home is True and away.is_home is False
    if team_side == "home":
        team, opp = home, away
    else:
        team, opp = away, home

    mu = expected_team_runs(team, opp, ctx)
    p_over = _prob_over(team_total_line, mu)
    p_under = 1.0 - p_over

    implied_over, implied_under = remove_vig_two_way(over_odds, under_odds)
    edge_over = p_over - implied_over
    edge_under = p_under - implied_under

    if edge_over >= edge_under and edge_over >= min_edge:
        side, odds, prob, implied, edge = (f"{team_side.upper()} O{team_total_line}",
                                           over_odds, p_over, implied_over, edge_over)
        direction = +1
    elif edge_under > edge_over and edge_under >= min_edge:
        side, odds, prob, implied, edge = (f"{team_side.upper()} U{team_total_line}",
                                           under_odds, p_under, implied_under, edge_under)
        direction = -1
    else:
        side = "NO BET"
        odds = None
        if edge_over >= edge_under:
            prob, implied, edge, direction = p_over, implied_over, edge_over, +1
        else:
            prob, implied, edge, direction = p_under, implied_under, edge_under, -1

    # Input certainty: heavy penalty if opposing starter isn't confirmed
    certainty = 1.0
    if not opp.starter_confirmed:
        certainty -= 0.50
    if not team.lineup_confirmed:
        certainty -= 0.15
    certainty = clamp(certainty, 0.3, 1.0)

    # Single-family agreement proxy: directionality of the three inputs
    family_directional = {
        "offense": offense_expected_runs_rating(team),
        "opp_sp": -opp_sp_runs_suppression(opp),
        "context": team_total_context_score(ctx),
    }
    agree = family_agreement(family_directional, direction)

    conf, label = confidence_score(
        edge=edge, family_agreement=agree, input_certainty=certainty,
    )

    if side != "NO BET" and label in ("LOW", "LEAN"):
        side = "NO BET"
        odds = None

    ev = ev_per_unit(prob, odds) if odds is not None else 0.0
    pick_str = f"{side} {_fmt_odds(odds)}" if odds is not None else "NO BET"

    return PredictionResult(
        market=f"team_total_{team_side}",
        pick=pick_str,
        odds=odds,
        model_prob=prob,
        implied_prob=implied,
        edge=edge,
        confidence=conf,
        confidence_label=label,
        expected_value_per_unit=ev,
        detail={
            "team_side": team_side,
            "line": team_total_line,
            "expected_runs": mu,
            "p_over": p_over,
            "p_under": p_under,
            "family_scores": family_directional,
            "input_certainty": certainty,
        },
    )


def _fmt_odds(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)
