"""
Moneyline predictor — 7-family framework.

Feature-family weights come from the research in
Moneyline_Indicators_Research.docx:
    1. Starting Pitchers    35%
    2. Bullpen              20%
    3. Offense              20%
    4. Defense/Baserunning   7%
    5. Situational           8%
    6. Environment           5%
    7. Market (adjustment)   5%

Pipeline:
  (a) compute each family's z-score for home and away
  (b) take (home - away) differential per family, weighted sum => "edge_score"
  (c) add home-field win-probability lift
  (d) sigmoid the edge_score => model_win_prob(home)
  (e) de-vig market odds => implied_prob
  (f) model_prob - implied_prob => edge per side
  (g) apply heavy-favorite penalty and market sharpness bump
  (h) pick the side with the largest positive edge (respecting min-edge threshold)
  (i) score confidence

All weights are tunable — they live in MONEYLINE_WEIGHTS.
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


MONEYLINE_WEIGHTS = {
    "pitcher":     0.35,
    "bullpen":     0.20,
    "offense":     0.20,
    "defense":     0.07,
    "situational": 0.08,
    "environment": 0.05,
    "market":      0.05,
}

LOGISTIC_SCALE_K = 1.0
MIN_EDGE = 0.025


def pitcher_score(team: TeamStats, opponent: TeamStats) -> float:
    p = team.pitcher
    s = 0.0
    s += 0.28 * z(p.siera, "sp_siera", invert=True)
    s += 0.18 * z(p.xfip,  "sp_xfip",  invert=True)
    s += 0.18 * z(p.k_bb_pct, "sp_k_bb_pct")
    s += 0.12 * z(p.csw_pct, "sp_csw_pct")
    s += 0.12 * z(p.xwoba_against, "sp_xwoba_against", invert=True)
    s += 0.05 * z(p.ip_per_gs, "sp_ip_per_gs")
    s += 0.07 * z(p.rolling_30d_era, "sp_rolling_era", invert=True)
    if p.xwoba_vs_opp_hand is not None:
        platoon_z = z(p.xwoba_vs_opp_hand, "sp_xwoba_against", invert=True)
        s = 0.85 * s + 0.15 * platoon_z
    return s


def bullpen_score(team: TeamStats) -> float:
    bp = team.bullpen
    s = 0.0
    s += 0.35 * z(bp.fip, "bp_fip", invert=True)
    s += 0.25 * z(bp.hi_lev_k_pct, "bp_hi_lev_k_pct")
    s += 0.15 * z(bp.shutdown_pct, "bp_shutdown_pct")
    s += 0.15 * z(bp.meltdown_pct, "bp_meltdown_pct", invert=True)
    rest_pen = 0.0
    if bp.closer_pitches_last3d >= 30:
        rest_pen += 0.3
    if bp.setup_pitches_last3d >= 35:
        rest_pen += 0.2
    s += 0.10 * (-rest_pen)
    return s


def offense_score(team: TeamStats, opp_pitcher_throws: str) -> float:
    o = team.offense
    s = 0.0
    s += 0.35 * z(o.wrc_plus, "off_wrc_plus")
    s += 0.15 * z(o.xwOBA, "off_xwOBA")
    s += 0.10 * z(o.wOBA, "off_wOBA")
    s += 0.10 * z(o.obp, "off_obp")
    s += 0.08 * z(o.iso, "off_iso")
    s += 0.08 * z(o.barrel_pct, "off_barrel_pct")
    s += 0.07 * z(o.k_pct, "off_k_pct", invert=True)
    s += 0.07 * z(o.top_of_order_obp, "off_top_obp")
    if o.wrc_plus_vs_opp_hand is not None:
        platoon_z = z(o.wrc_plus_vs_opp_hand, "off_wrc_plus")
        s = 0.85 * s + 0.15 * platoon_z
    return s


def defense_score(team: TeamStats) -> float:
    d = team.defense
    s = 0.0
    s += 0.40 * z(d.oaa, "def_oaa")
    s += 0.30 * z(d.drs, "def_drs")
    s += 0.20 * z(d.catcher_framing_runs, "def_framing")
    s += 0.10 * z(d.bsr, "def_bsr")
    return s


def situational_score(team: TeamStats, opponent: TeamStats) -> float:
    s = 0.0
    s += 0.40 * ((team.form_last10_win_pct - 0.500) / 0.150)
    s += 0.30 * ((team.form_last20_win_pct - 0.500) / 0.120)
    rest_edge = clamp(team.rest_days - opponent.rest_days, -3, 3) / 3.0
    s += 0.20 * rest_edge
    travel_z = -clamp((team.travel_miles_72h - 500) / 1500.0, 0, 1)
    s += 0.05 * travel_z
    if not team.meaningful_game:
        s -= 0.10
    return s


def market_score_ml(team_is_home: bool, market: MarketData) -> float:
    """Market family: opener->close line movement and public splits.
    Positive = this team's side is favored by late market action.
    Mirrors market_score_rl for framework consistency.
    """
    if market.opener_home_ml_odds is None:
        return 0.0
    moved_toward_home = market.home_ml_odds < market.opener_home_ml_odds
    want_home = team_is_home
    s = 0.0
    if market.public_ticket_pct_home is not None:
        pub_on_home = market.public_ticket_pct_home
        if want_home and moved_toward_home and pub_on_home < 0.45:
            s += 0.5
        if (not want_home) and (not moved_toward_home) and pub_on_home > 0.55:
            s += 0.5
    else:
        delta = abs(market.home_ml_odds - market.opener_home_ml_odds)
        mag = min(delta / 200.0, 1.0) * 0.2
        if want_home:
            s += mag if moved_toward_home else -mag
        else:
            s += mag if (not moved_toward_home) else -mag
    if market.steam_flag_home:
        s += 0.25 if want_home else -0.25
    return s


def environment_score(team: TeamStats, opponent: TeamStats, ctx: GameContext) -> float:
    s = 0.0
    if ctx.park_run_factor > 1.05:
        off_edge = team.offense.wrc_plus - opponent.offense.wrc_plus
        s += 0.5 * (off_edge / 20.0)
    if ctx.park_run_factor < 0.95:
        pitch_edge = -(team.pitcher.siera - opponent.pitcher.siera)
        s += 0.5 * (pitch_edge / 0.5)
    if ctx.ump_runs_per_game > 9.2:
        off_edge = team.offense.wrc_plus - opponent.offense.wrc_plus
        s += 0.3 * (off_edge / 20.0)
    return s


def predict_moneyline(
    home: TeamStats,
    away: TeamStats,
    ctx: GameContext,
    market: MarketData,
    min_edge: float = MIN_EDGE,
) -> PredictionResult:
    assert home.is_home is True, "home team must have is_home=True"
    assert away.is_home is False, "away team must have is_home=False"

    h_fam = {
        "pitcher":     pitcher_score(home, away),
        "bullpen":     bullpen_score(home),
        "offense":     offense_score(home, away.pitcher.throws),
        "defense":     defense_score(home),
        "situational": situational_score(home, away),
        "environment": environment_score(home, away, ctx),
        "market":      market_score_ml(True, market),
    }
    a_fam = {
        "pitcher":     pitcher_score(away, home),
        "bullpen":     bullpen_score(away),
        "offense":     offense_score(away, home.pitcher.throws),
        "defense":     defense_score(away),
        "situational": situational_score(away, home),
        "environment": environment_score(away, home, ctx),
        "market":      market_score_ml(False, market),
    }

    diff_by_family = {k: h_fam[k] - a_fam[k] for k in h_fam}
    edge_score = sum(
        MONEYLINE_WEIGHTS[k] * diff_by_family[k] for k in diff_by_family
    )

    HFA_LOGIT = math.log((0.5 + HOME_FIELD_WIN_PROB_LIFT) /
                         (0.5 - HOME_FIELD_WIN_PROB_LIFT))
    logit_home = LOGISTIC_SCALE_K * edge_score + HFA_LOGIT
    model_home_prob = logistic(logit_home)
    model_away_prob = 1.0 - model_home_prob

    implied_home, implied_away = remove_vig_two_way(
        market.home_ml_odds, market.away_ml_odds
    )
    edge_home = model_home_prob - implied_home
    edge_away = model_away_prob - implied_away

    # Market signal lives in the "market" family (5% weight) — no
    # additional market_sharpness bump here to avoid double-counting.

    hf_penalty_home = 0.04 if market.home_ml_odds <= -175 else 0.0
    hf_penalty_away = 0.04 if market.away_ml_odds <= -175 else 0.0
    edge_home -= hf_penalty_home
    edge_away -= hf_penalty_away

    if edge_home >= edge_away and edge_home >= min_edge:
        side, odds, prob, implied, edge = "HOME", market.home_ml_odds, \
            model_home_prob, implied_home, edge_home
        direction = +1
        hf_penalty_applied = hf_penalty_home
    elif edge_away > edge_home and edge_away >= min_edge:
        side, odds, prob, implied, edge = "AWAY", market.away_ml_odds, \
            model_away_prob, implied_away, edge_away
        direction = -1
        hf_penalty_applied = hf_penalty_away
    else:
        side = "NO BET"
        odds = None
        if edge_home >= edge_away:
            prob, implied, edge, direction = model_home_prob, implied_home, edge_home, +1
        else:
            prob, implied, edge, direction = model_away_prob, implied_away, edge_away, -1
        hf_penalty_applied = 0.0

    certainty = 1.0
    if not (home.starter_confirmed and away.starter_confirmed):
        certainty -= 0.35
    if not (home.lineup_confirmed and away.lineup_confirmed):
        certainty -= 0.15
    certainty = clamp(certainty, 0.4, 1.0)

    agree = family_agreement(diff_by_family, direction)

    extra = 0.0
    if side == "HOME" and home.bullpen.closer_pitches_last3d >= 35:
        extra += 0.30
    if side == "AWAY" and away.bullpen.closer_pitches_last3d >= 35:
        extra += 0.30
    if hf_penalty_applied > 0:
        extra += 0.35

    conf, label = confidence_score(
        edge=edge,
        family_agreement=agree,
        input_certainty=certainty,
        extra_penalty=extra,
    )

    if not globals().get("_BYPASS_GATES", False):
        if side != "NO BET" and label in ("LOW", "LEAN"):
            side = "NO BET"
            odds = None

    ev = ev_per_unit(prob, odds) if odds is not None else 0.0
    pick_str = f"{side} {_fmt_odds(odds)}" if odds is not None else "NO BET"

    return PredictionResult(
        market="moneyline",
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
            "diff_by_family": diff_by_family,
            "edge_score_weighted": edge_score,
            "model_home_prob": model_home_prob,
            "model_away_prob": model_away_prob,
            "implied_home_prob": implied_home,
            "implied_away_prob": implied_away,
            "heavy_favorite_penalty": hf_penalty_applied,
            "family_agreement": agree,
            "input_certainty": certainty,
        },
    )


def _fmt_odds(odds):
    if odds is None:
        return ""
    return f"+{odds}" if odds > 0 else str(odds)
