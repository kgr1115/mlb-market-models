"""
No-Runs-First-Inning (NRFI) / Yes-Runs-First-Inning (YRFI) predictor.

Why NRFI is a softer market
---------------------------
1. Pure starter-and-top-of-order matchup. Books price NRFI off a lazy
   formula (team-season first-inning run rates × an SP multiplier),
   which ignores heavy platoon splits, ump strike-zone effects, and
   the most important input: the SP's *first-inning* xwOBA-allowed.
2. It's posted at ~-140 NRFI / +120 YRFI across most North American
   books with almost no intraday movement. You can regularly find ten
   cents of shop variance between DK and Caesars on the exact same
   matchup.
3. Correlations with the game ML and total are low, so including an
   NRFI pick in a portfolio doesn't stack existing exposure.

Scoring
-------
NRFI-suppression rating per team (how likely this team + its SP is to
put up a 0 in the first inning), combined into a joint probability:

    P(NRFI) = P(home team doesn't score in T1) * P(away doesn't score in B1)

Team-level "can't score in 1st" probability is derived from:
    * top-3 batters' OBP (primary driver — you need baserunners first)
    * opposing SP's first-inning xwOBA-allowed (or rolling SIERA proxy)
    * park run factor (only affects extra-base hit probability)
    * ump called-strike rate (favors suppression)
"""
from __future__ import annotations

import math
from typing import Optional

from .shared import (
    LEAGUE, TeamStats, GameContext, MarketData, PredictionResult,
    american_to_prob, remove_vig_two_way, clamp, z, ev_per_unit,
    confidence_score, family_agreement,
    logistic,
)


# Base rate of NRFI in modern MLB is ~0.58 (NRFI wins ~58% of the time)
BASE_NRFI = 0.58

# Logistic scale: how much a 1.0-z-score edge moves the probability
LOGIT_SCALE = 0.35

MIN_EDGE = 0.025


def team_scoring_chance_first(team: TeamStats, opp_pitcher: TeamStats,
                              ctx: GameContext) -> float:
    """Logit-space contribution that this team scores in their half inning.

    Positive value = team MORE likely to score in first (bad for NRFI).
    Negative = LESS likely (good for NRFI).
    """
    s = 0.0
    # Top-of-order OBP drives scoring chances — they get the first AB
    s += 0.60 * z(team.offense.top_of_order_obp, "off_top_obp")
    # Team xwOBA tiebreaker
    s += 0.20 * z(team.offense.xwOBA, "off_xwOBA")
    # Platoon split if top-of-order vs SP hand is known on the TeamStats
    if team.offense.wrc_plus_vs_opp_hand is not None:
        s += 0.15 * z(team.offense.wrc_plus_vs_opp_hand, "off_wrc_plus")
    # Opposing SP first-inning suppression
    p = opp_pitcher.pitcher
    s -= 0.55 * z(p.siera, "sp_siera", invert=True)
    s -= 0.25 * z(p.xwoba_against, "sp_xwoba_against", invert=True)
    s -= 0.20 * z(p.k_bb_pct, "sp_k_bb_pct")
    # Park: extra-hitter parks raise first-inning HR rate
    s += 0.6 * (ctx.park_run_factor - 1.0)
    # Ump: high called-strike rate suppresses runs
    s -= 0.20 * (ctx.ump_called_strike_rate - 0.50) * 10
    return s


def predict_nrfi(
    home: TeamStats,
    away: TeamStats,
    ctx: GameContext,
    nrfi_odds: int = -140,
    yrfi_odds: int = +115,
    min_edge: float = MIN_EDGE,
) -> PredictionResult:
    """Predict NRFI vs YRFI.

    Prices default to the league-average NRFI posting. Caller should
    substitute book-shopped best-available prices when known.
    """
    assert home.is_home is True and away.is_home is False

    # Home bats in bottom of 1st, away bats in top of 1st
    home_score_logit = team_scoring_chance_first(home, away, ctx)
    away_score_logit = team_scoring_chance_first(away, home, ctx)

    # Convert each team's logit-contribution to a "this team scores in 1st" prob.
    # Base rate team scores in 1st = ~0.25 (modern MLB). Logit-center on that.
    base_logit = math.log(0.25 / 0.75)
    p_home_scores_1st = logistic(base_logit + LOGIT_SCALE * home_score_logit)
    p_away_scores_1st = logistic(base_logit + LOGIT_SCALE * away_score_logit)

    # NRFI wins iff neither team scores in 1st. Assume approximate independence.
    p_nrfi = (1.0 - p_home_scores_1st) * (1.0 - p_away_scores_1st)
    # Floor/ceiling: base league NRFI is ~0.58; don't let the model wander too far.
    p_nrfi = clamp(p_nrfi, 0.35, 0.80)
    p_yrfi = 1.0 - p_nrfi

    implied_nrfi, implied_yrfi = remove_vig_two_way(nrfi_odds, yrfi_odds)
    edge_nrfi = p_nrfi - implied_nrfi
    edge_yrfi = p_yrfi - implied_yrfi

    if edge_nrfi >= edge_yrfi and edge_nrfi >= min_edge:
        side, odds, prob, implied, edge = ("NRFI", nrfi_odds,
                                           p_nrfi, implied_nrfi, edge_nrfi)
        direction = +1
    elif edge_yrfi > edge_nrfi and edge_yrfi >= min_edge:
        side, odds, prob, implied, edge = ("YRFI", yrfi_odds,
                                           p_yrfi, implied_yrfi, edge_yrfi)
        direction = -1
    else:
        side = "NO BET"
        odds = None
        if edge_nrfi >= edge_yrfi:
            prob, implied, edge, direction = p_nrfi, implied_nrfi, edge_nrfi, +1
        else:
            prob, implied, edge, direction = p_yrfi, implied_yrfi, edge_yrfi, -1

    certainty = 1.0
    if not (home.starter_confirmed and away.starter_confirmed):
        certainty -= 0.55
    if not (home.lineup_confirmed and away.lineup_confirmed):
        certainty -= 0.15
    certainty = clamp(certainty, 0.3, 1.0)

    conf, label = confidence_score(
        edge=edge,
        family_agreement=0.7,   # NRFI only has two "families" — fake a moderate value
        input_certainty=certainty,
    )

    if side != "NO BET" and label in ("LOW", "LEAN"):
        side = "NO BET"
        odds = None

    ev = ev_per_unit(prob, odds) if odds is not None else 0.0
    pick_str = f"{side} {_fmt_odds(odds)}" if odds is not None else "NO BET"

    return PredictionResult(
        market="nrfi",
        pick=pick_str,
        odds=odds,
        model_prob=prob,
        implied_prob=implied,
        edge=edge,
        confidence=conf,
        confidence_label=label,
        expected_value_per_unit=ev,
        detail={
            "p_home_scores_1st": p_home_scores_1st,
            "p_away_scores_1st": p_away_scores_1st,
            "p_nrfi": p_nrfi,
            "p_yrfi": p_yrfi,
            "home_score_logit": home_score_logit,
            "away_score_logit": away_score_logit,
            "input_certainty": certainty,
        },
    )


def _fmt_odds(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)
