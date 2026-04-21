"""
Run Line predictor — 5-family framework.

Feature-family weights from RunLine_Indicators_Research.docx:
    1. Starting Pitchers  30%
    2. Bullpen            20%
    3. Offense            25%
    4. Context            15%
    5. Market             10%

The run line is (almost always) -1.5/+1.5. Unlike the moneyline we can't
treat this as "who is stronger"; the favorite needs to win by 2+ and the
dog is alive if they lose by exactly 1. That means EXPECTED MARGIN and
MARGIN VARIANCE both matter.

Pipeline:
  (a) score each family (z-scores)
  (b) convert weighted diff to "expected margin" in runs
     (calibrated: 1.0 family-diff ≈ 0.55 run margin)
  (c) add home-field adjustment to expected margin (+0.18 runs on avg)
  (d) apply home-favorite-with-1.5 discount  (no bottom-9 at-bats)
  (e) use a normal model P(margin >= 2) / P(margin <= 1) for cover prob
  (f) compare to de-vigged market RL prob for both sides
  (g) pick the side with positive edge above threshold
  (h) score confidence (one-run-game frequency => variance penalty)
"""
from __future__ import annotations

import math
from typing import Optional

from .shared import (
    LEAGUE, HOME_FIELD_WIN_PROB_LIFT,
    TeamStats, GameContext, MarketData, PredictionResult,
    american_to_prob, american_to_decimal, remove_vig_two_way,
    logistic, clamp, z, ev_per_unit,
    confidence_score, family_agreement, market_sharpness,
)
from . import moneyline as ml


RUN_LINE_WEIGHTS = {
    "pitcher":     0.30,
    "bullpen":     0.20,
    "offense":     0.25,
    "context":     0.15,
    "market":      0.10,
}

# Calibration: 1.0 unit of weighted family-diff corresponds to ~0.55 runs
# of expected margin. Tune empirically.
FAMILY_DIFF_TO_RUNS = 0.55

# Home-field ~0.18-run margin boost (matches ~0.04 win-prob lift)
HOME_MARGIN_LIFT = 0.18

# ~28% of MLB games end by exactly 1 run, so the standard deviation of
# game margin is roughly 4.0 runs.
MARGIN_STD_DEV = 4.0

MIN_EDGE = 0.025


def context_score(team: TeamStats, opponent: TeamStats, ctx: GameContext) -> float:
    """Park, weather, ump, lineup confirmation, defense mini-contribution."""
    s = 0.0
    # Park helps neither ML nor RL directly, but helps sluggers cover -1.5:
    # extreme hitter park + better slugging = bigger margins, good for favorite.
    if ctx.park_run_factor > 1.05:
        iso_edge = team.offense.iso - opponent.offense.iso
        s += 0.25 * (iso_edge / 0.020)
    # Wind blowing out helps the better offense cover big
    if ctx.wind_direction == "out" and ctx.wind_speed_mph >= 10:
        iso_edge = team.offense.iso - opponent.offense.iso
        s += 0.20 * (iso_edge / 0.020) * (ctx.wind_speed_mph / 10.0)
    # Ump favorable to offenses: small help to bigger offense for blowouts
    if ctx.ump_runs_per_game > 9.2:
        off_edge = team.offense.wrc_plus - opponent.offense.wrc_plus
        s += 0.15 * (off_edge / 20.0)
    # Defense edge — can suppress 1-run leakage (esp. for favorite covering)
    s += 0.40 * (z(team.defense.oaa, "def_oaa") - z(opponent.defense.oaa, "def_oaa")) / 2
    # Lineups unconfirmed: flat damping
    if not (team.lineup_confirmed and opponent.lineup_confirmed):
        s *= 0.8
    return s


def market_score_rl(team_is_home: bool, market: MarketData) -> float:
    """Small z-like bump from RL-specific line move or public splits.
    Positive score favors the chosen team covering.
    """
    if market.opener_home_ml_odds is None:
        return 0.0
    # moved_toward_home: home odds got more negative (shorter) closing vs opener.
    # That's where money came in on late-market action.
    moved_toward_home = market.home_ml_odds < market.opener_home_ml_odds
    want_home = team_is_home
    s = 0.0
    if market.public_ticket_pct_home is not None:
        pub_on_home = market.public_ticket_pct_home
        # Classic RLM: line moves AWAY from public action
        if want_home and moved_toward_home and pub_on_home < 0.45:
            s += 0.5
        if (not want_home) and (not moved_toward_home) and pub_on_home > 0.55:
            s += 0.5
    else:
        # Fallback when no public-ticket data: late money direction alone,
        # magnitude-scaled. Caps at ~0.2 so it can't overpower per-family gates.
        delta = abs(market.home_ml_odds - market.opener_home_ml_odds)
        mag = min(delta / 200.0, 1.0) * 0.2
        if want_home:
            s += mag if moved_toward_home else -mag
        else:
            s += mag if (not moved_toward_home) else -mag
    if market.steam_flag_home:
        s += 0.25 if want_home else -0.25
    return s


# -----------------------------------------------------------------------------
# Main predictor
# -----------------------------------------------------------------------------

def predict_run_line(
    home: TeamStats,
    away: TeamStats,
    ctx: GameContext,
    market: MarketData,
    min_edge: float = MIN_EDGE,
) -> PredictionResult:
    assert home.is_home is True and away.is_home is False

    # Re-use moneyline family functions for pitcher/bullpen/offense
    h_fam = {
        "pitcher": ml.pitcher_score(home, away),
        "bullpen": ml.bullpen_score(home),
        "offense": ml.offense_score(home, away.pitcher.throws),
        "context": context_score(home, away, ctx),
        "market":  market_score_rl(True, market),
    }
    a_fam = {
        "pitcher": ml.pitcher_score(away, home),
        "bullpen": ml.bullpen_score(away),
        "offense": ml.offense_score(away, home.pitcher.throws),
        "context": context_score(away, home, ctx),
        "market":  market_score_rl(False, market),
    }
    diff = {k: h_fam[k] - a_fam[k] for k in h_fam}
    weighted_diff = sum(RUN_LINE_WEIGHTS[k] * diff[k] for k in diff)

    # Expected home margin in runs
    expected_margin_home = weighted_diff * FAMILY_DIFF_TO_RUNS + HOME_MARGIN_LIFT

    # HOME favorite with -1.5 discount: the home team never bats in bottom 9
    # when leading, so when they're the favorite their realized margins are
    # systematically smaller. Research shows ~ -0.15 runs of effective margin.
    if market.home_is_rl_favorite:
        expected_margin_home -= 0.15

    # Cover probabilities via Normal(mean=expected_margin_home, sd=MARGIN_STD_DEV)
    #   Favorite (laying -1.5) covers iff margin >= 2
    #   Dog (+1.5) covers iff margin <= 1  (from fav's POV)
    # If home is favorite, fav-covers = P(home_margin >= 2)
    #                     dog-covers = P(home_margin <= 1) = 1 - P(home_margin >= 2)
    p_home_margin_ge2 = _prob_normal_ge(expected_margin_home, 2.0, MARGIN_STD_DEV)
    p_home_margin_le_minus2 = _prob_normal_le(expected_margin_home, -2.0, MARGIN_STD_DEV)

    if market.home_is_rl_favorite:
        p_home_rl_cover = p_home_margin_ge2
        p_away_rl_cover = 1 - p_home_margin_ge2
    else:
        # away is favorite, laying -1.5; they cover iff away margin >= 2,
        # i.e. home margin <= -2
        p_away_rl_cover = p_home_margin_le_minus2
        p_home_rl_cover = 1 - p_home_margin_le_minus2

    # Market
    implied_home, implied_away = remove_vig_two_way(market.home_rl_odds, market.away_rl_odds)
    edge_home = p_home_rl_cover - implied_home
    edge_away = p_away_rl_cover - implied_away

    # Market signal lives in the "market" family (10% weight) — no
    # additional market_sharpness bump here to avoid double-counting.

    # Pick side
    if edge_home >= edge_away and edge_home >= min_edge:
        side, odds, prob, implied, edge = "HOME -1.5" if market.home_is_rl_favorite else "HOME +1.5", \
            market.home_rl_odds, p_home_rl_cover, implied_home, edge_home
        direction = +1
    elif edge_away > edge_home and edge_away >= min_edge:
        side, odds, prob, implied, edge = "AWAY -1.5" if not market.home_is_rl_favorite else "AWAY +1.5", \
            market.away_rl_odds, p_away_rl_cover, implied_away, edge_away
        direction = -1
    else:
        side = "NO BET"
        odds = None
        if edge_home >= edge_away:
            prob, implied, edge, direction = p_home_rl_cover, implied_home, edge_home, +1
        else:
            prob, implied, edge, direction = p_away_rl_cover, implied_away, edge_away, -1

    # Confidence — down-weight low-margin (close) games with high variance
    certainty = 1.0
    if not (home.starter_confirmed and away.starter_confirmed):
        certainty -= 0.35
    if not (home.lineup_confirmed and away.lineup_confirmed):
        certainty -= 0.15
    certainty = clamp(certainty, 0.4, 1.0)

    agree = family_agreement(diff, direction)

    # Variance penalty: games with |expected margin| near 1-1.5 are the
    # highest-variance RL spots (cover flips on a single run).
    variance_pen = 0.0
    if abs(expected_margin_home) < 1.5:
        variance_pen = 0.40
    elif abs(expected_margin_home) < 2.0:
        variance_pen = 0.20

    # Extra: tired-closer penalty for favorite
    extra = 0.0
    fav_bp = home.bullpen if market.home_is_rl_favorite else away.bullpen
    if fav_bp.closer_pitches_last3d >= 35:
        extra += 0.30

    conf, label = confidence_score(
        edge=edge,
        family_agreement=agree,
        input_certainty=certainty,
        variance_penalty=variance_pen,
        extra_penalty=extra,
    )

    # Confidence gate: LEAN/LOW-band picks were net-negative across the
    # multi-season backtest. Require at least MEDIUM before firing.
    # Bypassable via module-level _BYPASS_GATES flag (train/test tuner).
    if not globals().get("_BYPASS_GATES", False):
        if side != "NO BET" and label in ("LOW", "LEAN"):
            side = "NO BET"
            odds = None

    ev = ev_per_unit(prob, odds) if odds is not None else 0.0
    pick_str = f"{side} {_fmt_odds(odds)}" if odds is not None else "NO BET"

    return PredictionResult(
        market="run_line",
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
            "weighted_diff": weighted_diff,
            "expected_margin_home": expected_margin_home,
            "p_home_rl_cover": p_home_rl_cover,
            "p_away_rl_cover": p_away_rl_cover,
            "implied_home_prob": implied_home,
            "implied_away_prob": implied_away,
            "variance_penalty": variance_pen,
            "family_agreement": agree,
            "input_certainty": certainty,
        },
    )


# -----------------------------------------------------------------------------
# Normal-CDF helpers (no SciPy dependency)
# -----------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _prob_normal_ge(mean: float, threshold: float, std: float) -> float:
    """P(X >= threshold) for X ~ N(mean, std)."""
    if std <= 0:
        return 1.0 if mean >= threshold else 0.0
    return 1.0 - _norm_cdf((threshold - mean) / std)

def _prob_normal_le(mean: float, threshold: float, std: float) -> float:
    """P(X <= threshold) for X ~ N(mean, std)."""
    if std <= 0:
        return 1.0 if mean <= threshold else 0.0
    return _norm_cdf((threshold - mean) / std)


def _fmt_odds(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)
