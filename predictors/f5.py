"""
First-5-Innings (F5) predictor.

Why F5 is a softer market
-------------------------
1. Bullpens are removed from the equation. F5 resolves before the pen
   is used in 80%+ of games (starters go 5+ in ~70% of GS). This collapses
   a noisy, hard-to-value input into near-zero influence.
2. The books build F5 from the full-game ML by subtracting a flat
   bullpen/context adjustment. That adjustment is a simplification; any
   team-by-team deviation is unpriced edge.
3. Volume is an order of magnitude smaller, so sharp money moves lines
   less aggressively than on the full-game ML. Soft books post F5 late
   and rarely adjust intraday until first pitch.

Scoring
-------
We re-use moneyline.pitcher_score with *heavier* weight on pitcher
(this is the F5 market), add ONLY the top-of-order slice of offense
(the pitcher will only face ~18 batters in 5 innings — heart-of-order
matters more than tail), and skip bullpen entirely. Park and weather
contribute the same as on the full game.

Output is a home F5 win probability. We bet either home-F5 or away-F5
when model_prob - implied_prob >= MIN_EDGE.

Ties (5-5 after 5) are common; most US books grade F5 ML as a tie/push
if they include the "3-way" variant, or as a refund on "2-way". We
output probabilities assuming the 2-way-with-tie-push version (the
standard DraftKings / FanDuel grading rule), and if the book has 3-way
posted, the caller should use .detail["p_tie"] directly.
"""
from __future__ import annotations

import math
from typing import Optional

from .shared import (
    TeamStats, GameContext, MarketData, PredictionResult,
    american_to_prob, remove_vig_two_way,
    logistic, clamp, z, ev_per_unit,
    confidence_score, family_agreement,
    HOME_FIELD_WIN_PROB_LIFT,
)
from . import moneyline as ml


F5_WEIGHTS = {
    "pitcher":       0.60,   # 60% — F5 is fundamentally a pitcher market
    "top_of_order":  0.20,   # 20% — only the top slots see the pitcher twice
    "environment":   0.10,   # park, wind
    "defense":       0.05,   # framing and OAA in 15 outs
    "market":        0.05,
}

# Home-field lift is smaller on F5 (no last-bat half-inning yet)
F5_HOME_LIFT = 0.020

# Per-run scale for F5: a 1.0 weighted edge ~ 0.30 runs in first 5 innings
F5_FAMILY_DIFF_TO_RUNS = 0.30
F5_MARGIN_STD = 2.6   # std deviation of F5 margin is ~2.6 runs

MIN_EDGE = 0.030    # F5 is a bit more volatile; require a slightly bigger edge


def top_of_order_score(team: TeamStats) -> float:
    """Weighted score for top-of-order hitters only."""
    o = team.offense
    s = 0.0
    s += 0.55 * z(o.top_of_order_obp, "off_top_obp")
    s += 0.25 * z(o.iso, "off_iso")
    s += 0.20 * z(o.wrc_plus, "off_wrc_plus")
    return s


def environment_score(ctx: GameContext) -> float:
    """Park + wind influence on first-5 scoring."""
    s = 0.0
    # Park: extra hitter park inflates both sides' F5 totals but does not
    # change win prob much. We downweight park for F5.
    park_boost = (ctx.park_run_factor - 1.0) * 1.5
    s += park_boost
    # Wind: blowing out in the first 5 helps totals but is direction-neutral
    # between teams unless we know whose lineup is more of a fly-ball team.
    # Default to neutral.
    return s


def defense_score(team: TeamStats, opp: TeamStats) -> float:
    """Framing + OAA difference contributes a small amount in 15 outs."""
    s = 0.0
    s += 0.5 * (z(team.defense.catcher_framing_runs, "def_framing") -
                z(opp.defense.catcher_framing_runs, "def_framing"))
    s += 0.5 * (z(team.defense.oaa, "def_oaa") -
                z(opp.defense.oaa, "def_oaa"))
    return s / 2


def predict_f5(
    home: TeamStats,
    away: TeamStats,
    ctx: GameContext,
    market: MarketData,
    home_f5_ml: Optional[int] = None,
    away_f5_ml: Optional[int] = None,
    min_edge: float = MIN_EDGE,
) -> PredictionResult:
    """Predict the F5 moneyline.

    Because MarketData doesn't carry F5 prices in the current schema,
    they are passed explicitly. If not provided we fall back to the
    full-game ML shifted toward -110 (a flat approximation books use).
    """
    assert home.is_home is True and away.is_home is False
    if home_f5_ml is None:
        home_f5_ml = _approximate_f5_from_ml(market.home_ml_odds)
    if away_f5_ml is None:
        away_f5_ml = _approximate_f5_from_ml(market.away_ml_odds)

    h_fam = {
        "pitcher":      ml.pitcher_score(home, away),
        "top_of_order": top_of_order_score(home),
        "environment":  environment_score(ctx),
        "defense":      defense_score(home, away),
        "market":       0.0,
    }
    a_fam = {
        "pitcher":      ml.pitcher_score(away, home),
        "top_of_order": top_of_order_score(away),
        "environment":  environment_score(ctx),
        "defense":      defense_score(away, home),
        "market":       0.0,
    }
    diff = {k: h_fam[k] - a_fam[k] for k in h_fam}
    edge_score = sum(F5_WEIGHTS[k] * diff[k] for k in diff)

    # Expected F5 margin and outcome probs
    expected_f5_margin = edge_score * F5_FAMILY_DIFF_TO_RUNS
    expected_f5_margin += 0.08   # tiny home-field in first 5

    # Under 2-way-with-push grading, P(home win F5) = P(margin > 0),
    # P(away win F5) = P(margin < 0), P(tie) = small mass at 0.
    p_home_win = 1.0 - _norm_cdf(0.5, expected_f5_margin, F5_MARGIN_STD)
    p_away_win = _norm_cdf(-0.5, expected_f5_margin, F5_MARGIN_STD)
    p_tie = 1.0 - p_home_win - p_away_win

    # Re-normalize to 2-way (push means refund — EV calculated on 2-way pool)
    denom = p_home_win + p_away_win
    if denom > 0:
        p_home_2way = p_home_win / denom
        p_away_2way = p_away_win / denom
    else:
        p_home_2way = p_away_2way = 0.5

    implied_home, implied_away = remove_vig_two_way(home_f5_ml, away_f5_ml)
    edge_home = p_home_2way - implied_home
    edge_away = p_away_2way - implied_away

    if edge_home >= edge_away and edge_home >= min_edge:
        side, odds, prob, implied, edge = ("HOME F5", home_f5_ml,
                                           p_home_2way, implied_home, edge_home)
        direction = +1
    elif edge_away > edge_home and edge_away >= min_edge:
        side, odds, prob, implied, edge = ("AWAY F5", away_f5_ml,
                                           p_away_2way, implied_away, edge_away)
        direction = -1
    else:
        side = "NO BET"
        odds = None
        if edge_home >= edge_away:
            prob, implied, edge, direction = p_home_2way, implied_home, edge_home, +1
        else:
            prob, implied, edge, direction = p_away_2way, implied_away, edge_away, -1

    agree = family_agreement(diff, direction)
    certainty = 1.0
    if not (home.starter_confirmed and away.starter_confirmed):
        certainty -= 0.4     # F5 is all about starters — hefty penalty if unconfirmed
    certainty = clamp(certainty, 0.3, 1.0)
    conf, label = confidence_score(
        edge=edge, family_agreement=agree, input_certainty=certainty,
        variance_penalty=0.0, extra_penalty=0.0,
    )

    # Higher MIN bar on F5 — only fire HIGH/MEDIUM picks
    if side != "NO BET" and label in ("LOW", "LEAN"):
        side = "NO BET"
        odds = None

    ev = ev_per_unit(prob, odds) if odds is not None else 0.0
    pick_str = f"{side} {_fmt_odds(odds)}" if odds is not None else "NO BET"

    return PredictionResult(
        market="f5",   # extends the Literal in shared.py; grade as market_extended
        pick=pick_str,
        odds=odds,
        model_prob=prob,
        implied_prob=implied,
        edge=edge,
        confidence=conf,
        confidence_label=label,
        expected_value_per_unit=ev,
        detail={
            "home_family_scores": h_fam,
            "away_family_scores": a_fam,
            "weighted_diff": edge_score,
            "expected_f5_margin": expected_f5_margin,
            "p_tie": p_tie,
            "p_home_f5": p_home_win,
            "p_away_f5": p_away_win,
            "home_f5_ml_odds": home_f5_ml,
            "away_f5_ml_odds": away_f5_ml,
        },
    )


# -----------------------------------------------------------------------------

def _approximate_f5_from_ml(full_game_ml: int) -> int:
    """Rough approximation: F5 ML typically sits ~10-20 cents closer to
    pick'em than the full-game ML, because the bullpen swing is stripped.
    """
    if full_game_ml < 0:
        # favorite: bring price back toward -110
        return max(full_game_ml + 20, -200)
    return min(full_game_ml - 20, 180)


def _norm_cdf(threshold: float, mean: float, std: float) -> float:
    if std <= 0:
        return 1.0 if mean <= threshold else 0.0
    return 0.5 * (1.0 + math.erf((threshold - mean) / (std * math.sqrt(2.0))))


def _fmt_odds(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)
