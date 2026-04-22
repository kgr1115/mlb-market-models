"""
Shared data structures and helpers for MLB betting predictors.

Every predictor module (moneyline, run_line, totals) consumes the same
TeamStats / GameContext / MarketData objects defined here. This means you
ingest once (pybaseball, MLB Stats API, Odds API, Open-Meteo) and feed all
three models from one normalized object graph.

All numeric league averages below are set to reasonable MLB values for
~2023-2025 run environment. You should override them as the season
progresses (or replace them with live FanGraphs leaderboard pulls).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Literal


# =============================================================================
# League-average constants (used as z-score baselines)
# =============================================================================

LEAGUE = {
    # starting pitcher (ERA-era neutralized)
    "sp_siera":        {"mean": 4.05, "std": 0.55},   # lower is better
    "sp_xfip":         {"mean": 4.10, "std": 0.55},   # lower is better
    "sp_k_bb_pct":     {"mean": 0.135, "std": 0.055}, # higher is better
    "sp_csw_pct":      {"mean": 0.285, "std": 0.020}, # higher is better
    "sp_xwoba_against":{"mean": 0.320, "std": 0.025}, # lower is better
    "sp_ip_per_gs":    {"mean": 5.30,  "std": 0.70},  # higher is better
    "sp_rolling_era":  {"mean": 4.05,  "std": 1.20},  # lower is better

    # bullpen
    "bp_fip":          {"mean": 4.00, "std": 0.45},   # lower is better
    "bp_hi_lev_k_pct": {"mean": 0.235, "std": 0.045}, # higher is better
    "bp_meltdown_pct": {"mean": 0.14,  "std": 0.04},  # lower is better
    "bp_shutdown_pct": {"mean": 0.30,  "std": 0.05},  # higher is better

    # offense (team-level)
    "off_wrc_plus":    {"mean": 100.0, "std": 10.0},  # higher is better
    "off_wOBA":        {"mean": 0.320, "std": 0.015},
    "off_xwOBA":       {"mean": 0.320, "std": 0.015},
    "off_obp":         {"mean": 0.320, "std": 0.015},
    "off_iso":         {"mean": 0.165, "std": 0.020},
    "off_barrel_pct":  {"mean": 0.080, "std": 0.015},
    "off_k_pct":       {"mean": 0.225, "std": 0.025}, # lower is better
    "off_top_obp":     {"mean": 0.340, "std": 0.020}, # top-of-order OBP, higher better

    # defense
    "def_oaa":         {"mean": 0.0,   "std": 10.0},  # higher is better (season-to-date)
    "def_drs":         {"mean": 0.0,   "std": 15.0},
    "def_framing":     {"mean": 0.0,   "std": 6.0},
    "def_bsr":         {"mean": 0.0,   "std": 5.0},

    # run environment
    "league_runs_per_game_per_team": 4.50,  # recent-season average
    "ump_runs_per_game":             8.80,  # plate ump mean
    "ump_rpg_std":                   0.35,
}

# Home-field win-probability lift, empirical ~0.04 in MLB
HOME_FIELD_WIN_PROB_LIFT = 0.040


# =============================================================================
# Dataclasses: every field has a league-average default so partial inputs work
# =============================================================================

@dataclass
class PitcherStats:
    name: str = "Unknown"
    throws: Literal["L", "R"] = "R"
    siera: float = LEAGUE["sp_siera"]["mean"]
    xfip: float = LEAGUE["sp_xfip"]["mean"]
    k_bb_pct: float = LEAGUE["sp_k_bb_pct"]["mean"]    # as decimal, e.g. 0.18
    csw_pct: float = LEAGUE["sp_csw_pct"]["mean"]
    xwoba_against: float = LEAGUE["sp_xwoba_against"]["mean"]
    ip_per_gs: float = LEAGUE["sp_ip_per_gs"]["mean"]
    rolling_30d_era: float = LEAGUE["sp_rolling_era"]["mean"]
    # platoon split: xwOBA allowed vs the opposing lineup's dominant hand
    xwoba_vs_opp_hand: Optional[float] = None
    # times-through-the-order penalty (true if opponent bats deep into order)
    third_tto_xwoba: Optional[float] = None


@dataclass
class BullpenStats:
    fip: float = LEAGUE["bp_fip"]["mean"]
    hi_lev_k_pct: float = LEAGUE["bp_hi_lev_k_pct"]["mean"]
    meltdown_pct: float = LEAGUE["bp_meltdown_pct"]["mean"]
    shutdown_pct: float = LEAGUE["bp_shutdown_pct"]["mean"]
    closer_pitches_last3d: int = 0       # >30-40 = tired
    setup_pitches_last3d: int = 0
    days_since_closer_used: int = 2
    # middle-relief (inn 5-7) ERA/FIP — critical for totals
    middle_relief_fip: Optional[float] = None


@dataclass
class OffenseStats:
    wrc_plus: float = LEAGUE["off_wrc_plus"]["mean"]
    wOBA: float = LEAGUE["off_wOBA"]["mean"]
    xwOBA: float = LEAGUE["off_xwOBA"]["mean"]
    obp: float = LEAGUE["off_obp"]["mean"]
    iso: float = LEAGUE["off_iso"]["mean"]
    barrel_pct: float = LEAGUE["off_barrel_pct"]["mean"]
    k_pct: float = LEAGUE["off_k_pct"]["mean"]
    top_of_order_obp: float = LEAGUE["off_top_obp"]["mean"]
    # split-specific wRC+ vs the hand of the opposing starter
    wrc_plus_vs_opp_hand: Optional[float] = None


@dataclass
class DefenseStats:
    oaa: float = LEAGUE["def_oaa"]["mean"]
    drs: float = LEAGUE["def_drs"]["mean"]
    catcher_framing_runs: float = LEAGUE["def_framing"]["mean"]
    bsr: float = LEAGUE["def_bsr"]["mean"]


@dataclass
class TeamStats:
    name: str = "Team"
    is_home: bool = True
    pitcher: PitcherStats = field(default_factory=PitcherStats)
    bullpen: BullpenStats = field(default_factory=BullpenStats)
    offense: OffenseStats = field(default_factory=OffenseStats)
    defense: DefenseStats = field(default_factory=DefenseStats)
    # situational
    form_last10_win_pct: float = 0.500
    form_last20_win_pct: float = 0.500
    rest_days: int = 1
    travel_miles_72h: float = 0.0
    meaningful_game: bool = True   # eliminated teams play differently in Sep
    lineup_confirmed: bool = False
    # luck-adjusted team strength priors — optional.
    # pythagorean_win_pct comes from run-differential (Pyth exp 1.83)
    # third_order_win_pct is BaseRuns-based (FanGraphs "3rd-order W%").
    # Both provide a less-noisy baseline than raw record.
    pythagorean_win_pct: Optional[float] = None
    third_order_win_pct: Optional[float] = None
    starter_confirmed: bool = True


@dataclass
class GameContext:
    # Season-level run environment correction: baseline team stats come from
    # prior year, but league scoring drifts year-to-year. Adding this drift
    # to projected_total aligns our projection with the current run era.
    league_run_drift: float = 0.0
    park_run_factor: float = 1.00      # 1.00 neutral, Coors ~1.18, Oracle ~0.93
    park_hr_factor: float = 1.00
    altitude_ft: float = 500
    roof_status: Literal["open", "closed", "none"] = "none"
    # weather (ignored if roof_status == "closed")
    wind_speed_mph: float = 0.0
    wind_direction: Literal["out", "in", "cross", "none"] = "none"
    temperature_f: float = 70.0
    humidity_pct: float = 50.0
    precipitation_pct: float = 0.0
    # umpire
    ump_runs_per_game: float = LEAGUE["ump_runs_per_game"]
    ump_called_strike_rate: float = 0.50   # league mean ≈ 0.50
    # schedule
    day_game: bool = False
    doubleheader: bool = False
    extra_innings_prev_game: bool = False


@dataclass
class MarketData:
    # Moneyline
    home_ml_odds: int = -110
    away_ml_odds: int = -110
    opener_home_ml_odds: Optional[int] = None
    opener_away_ml_odds: Optional[int] = None
    # Run line (assume -1.5 / +1.5 standard)
    home_rl_odds: int = +100   # favorite with -1.5 gets positive odds typically
    away_rl_odds: int = -120
    home_is_rl_favorite: bool = True   # true if home laying -1.5
    opener_home_rl_odds: Optional[int] = None
    opener_away_rl_odds: Optional[int] = None
    # Totals
    total_line: float = 8.5
    over_odds: int = -110
    under_odds: int = -110
    opener_total: Optional[float] = None
    opener_over_odds: Optional[int] = None
    opener_under_odds: Optional[int] = None
    # No-vig fair probabilities (consensus across books when >1 available).
    # Pros treat these as the "market truth"; edge = model_prob - fair_prob.
    # Populated by odds_client.build_market_data (live) and engine._build_market_data
    # (backtest). When None, predictors fall back to per-book remove_vig_two_way.
    fair_prob_home_ml: Optional[float] = None
    fair_prob_away_ml: Optional[float] = None
    fair_prob_home_rl: Optional[float] = None
    fair_prob_away_rl: Optional[float] = None
    fair_prob_over: Optional[float] = None
    fair_prob_under: Optional[float] = None
    # Reverse-line-movement intensity (-1..+1): positive on home/over side means
    # the line moved toward that side despite public betting against it — classic
    # sharp tell. Zero when public-split data is unavailable.
    rlm_score_home: float = 0.0
    rlm_score_over: float = 0.0
    # Public splits (0-1) — optional sharp/public signal
    public_ticket_pct_home: Optional[float] = None
    public_money_pct_home: Optional[float] = None
    public_ticket_pct_over: Optional[float] = None
    public_money_pct_over: Optional[float] = None
    steam_flag_home: bool = False
    steam_flag_over: bool = False


@dataclass
class PredictionResult:
    market: Literal["moneyline", "run_line", "totals"]
    pick: str                     # e.g. "HOME -110", "AWAY +1.5 -120", "OVER 8.5 -105", "NO BET"
    odds: Optional[int]           # american odds for the pick; None for NO BET
    model_prob: float             # model's true probability the pick wins
    implied_prob: float           # vig-stripped market implied probability
    edge: float                   # model_prob - implied_prob
    confidence: float             # 0..100, product-level confidence score
    confidence_label: str         # "LOW" | "MEDIUM" | "HIGH" | "LEAN"
    expected_value_per_unit: float  # EV per $1 risked (positive => +EV)
    detail: dict = field(default_factory=dict)  # all sub-scores for explainability


# =============================================================================
# Math helpers
# =============================================================================

def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability (still contains vig)."""
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)


def prob_to_american(p: float) -> int:
    """Convert a probability to the fair American odds."""
    p = clamp(p, 1e-6, 1 - 1e-6)
    if p >= 0.5:
        return int(round(-p / (1 - p) * 100))
    return int(round((1 - p) / p * 100))


def american_to_decimal(odds: int) -> float:
    """American odds → decimal payout (e.g. +150 -> 2.50)."""
    if odds > 0:
        return 1 + odds / 100.0
    return 1 + 100.0 / (-odds)


def remove_vig_two_way(odds_a: int, odds_b: int) -> tuple[float, float]:
    """Strip vig from a 2-way market via proportional scaling."""
    pa = american_to_prob(odds_a)
    pb = american_to_prob(odds_b)
    s = pa + pb
    return pa / s, pb / s


def fair_prob_consensus(
    books_two_way: list[tuple[Optional[int], Optional[int]]],
) -> tuple[Optional[float], Optional[float]]:
    """No-vig consensus fair probability from multiple books.

    Input: list of (odds_a, odds_b) tuples, one per book. Books with None
    on either side are skipped. Each remaining book is de-vigged
    independently; results are averaged to produce the consensus.

    The average-of-no-vig approach is standard practice among pro bettors
    — more robust than arithmetic-average-of-raw-odds because it handles
    asymmetric juice across books, and converges on the true market
    probability as more sharp books are added.

    Returns (consensus_prob_a, consensus_prob_b); both None if no valid
    book was provided.
    """
    probs_a: list[float] = []
    probs_b: list[float] = []
    for oa, ob in books_two_way:
        if oa is None or ob is None:
            continue
        try:
            pa, pb = remove_vig_two_way(int(oa), int(ob))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        probs_a.append(pa)
        probs_b.append(pb)
    if not probs_a:
        return None, None
    mean_a = sum(probs_a) / len(probs_a)
    mean_b = sum(probs_b) / len(probs_b)
    return mean_a, mean_b


def fair_prob_for_side(
    market: "MarketData",
    side: Literal["home_ml", "away_ml", "home_rl", "away_rl", "over", "under"],
) -> float:
    """Return the market's best estimate of true probability for one side.

    Prefers MarketData.fair_prob_* (the consensus no-vig across books, set
    by odds_client / backtest engine). Falls back to remove_vig_two_way on
    the side's own two-way odds if no consensus was precomputed.
    """
    attr = f"fair_prob_{side}"
    v = getattr(market, attr, None)
    if v is not None:
        return float(v)
    # Fallback per-book de-vig.
    if side in ("home_ml", "away_ml"):
        h, a = remove_vig_two_way(market.home_ml_odds, market.away_ml_odds)
        return h if side == "home_ml" else a
    if side in ("home_rl", "away_rl"):
        h, a = remove_vig_two_way(market.home_rl_odds, market.away_rl_odds)
        return h if side == "home_rl" else a
    if side in ("over", "under"):
        o, u = remove_vig_two_way(market.over_odds, market.under_odds)
        return o if side == "over" else u
    return 0.5


def logistic(x: float) -> float:
    """Sigmoid."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def z(value: float, key: str, invert: bool = False) -> float:
    """Standardize a stat using the LEAGUE mean/std table.

    invert=True: lower values are better (ERA, xFIP, SIERA, xwOBA-against)
    invert=False: higher values are better (K-BB%, wRC+, Barrel%)
    """
    mean = LEAGUE[key]["mean"]
    std = LEAGUE[key]["std"]
    if std == 0:
        return 0.0
    raw = (value - mean) / std
    return -raw if invert else raw


def ev_per_unit(model_prob: float, american_odds: int) -> float:
    """Expected value of a $1 stake at these odds, given model probability."""
    dec = american_to_decimal(american_odds)
    return model_prob * (dec - 1) - (1 - model_prob)


# =============================================================================
# Confidence scoring
# =============================================================================

def confidence_score(
    edge: float,
    family_agreement: float,          # 0..1 share of family scores agreeing w/ pick
    input_certainty: float,           # 0..1 (lineups, weather, starter confirmed)
    variance_penalty: float = 0.0,    # 0..1 subtract more for volatile spots
    extra_penalty: float = 0.0,       # heavy-favorite, short-rest pen, coors, etc.
) -> tuple[float, str]:
    """Return (0-100 confidence, label).

    Edge contributes the most; agreement/certainty are multipliers.
    Tuned so:
      edge 0.05, agreement 0.8, certainty 1.0 => ~72 (HIGH LEAN)
      edge 0.02, agreement 0.6, certainty 0.7 => ~35 (LOW)
    """
    edge_term = clamp(edge, -0.10, 0.15) * 400   # +0.05 edge ≈ +20 points
    base = 50.0 + edge_term
    base *= clamp(0.5 + family_agreement * 0.5, 0.3, 1.0)
    base *= clamp(0.6 + input_certainty * 0.4, 0.5, 1.0)
    base -= variance_penalty * 20.0
    base -= extra_penalty * 15.0
    score = clamp(base, 0.0, 100.0)

    if score >= 75:
        label = "HIGH"
    elif score >= 55:
        label = "MEDIUM"
    elif score >= 35:
        label = "LEAN"
    else:
        label = "LOW"
    return score, label


def family_agreement(family_scores: dict, direction: int) -> float:
    """Share of family z-scores that agree with the pick direction.

    direction: +1 if we like HOME/OVER, -1 if we like AWAY/UNDER.
    Families with near-zero scores (|z| < 0.1) are treated as neutral
    and not counted toward agreement or disagreement.
    """
    agree = 0
    counted = 0
    for _, v in family_scores.items():
        if abs(v) < 0.1:
            continue
        counted += 1
        if (v > 0 and direction > 0) or (v < 0 and direction < 0):
            agree += 1
    if counted == 0:
        return 0.5
    return agree / counted


def rlm_intensity(
    opener_fav_odds: Optional[int],
    current_fav_odds: Optional[int],
    public_ticket_pct_fav: Optional[float],
    steam_flag_fav: bool = False,
) -> float:
    """Reverse-line-movement intensity score in [-1, +1].

    Positive = the favored side here is the SHARP side (public bet the
    other way, line moved toward this side anyway). Negative = public is
    on this side and the line moved accordingly (no edge).

    Intensity scales with magnitude of movement, hardens if we also see
    a steam flag, and dampens when public split data is missing.
    """
    if opener_fav_odds is None or current_fav_odds is None:
        return 0.0
    moved_toward_fav = current_fav_odds < opener_fav_odds
    delta = abs(current_fav_odds - opener_fav_odds)
    mag = min(delta / 30.0, 1.0)   # 30 cents = full magnitude
    # Direction: +1 if fav shortened (late money on fav), else -1
    direction = 1.0 if moved_toward_fav else -1.0
    if public_ticket_pct_fav is not None:
        pub = public_ticket_pct_fav
        # Classic RLM: movement opposes public action
        if moved_toward_fav and pub < 0.40:
            score = mag * 1.0
        elif moved_toward_fav and pub < 0.50:
            score = mag * 0.5
        elif (not moved_toward_fav) and pub > 0.60:
            score = -mag * 1.0
        elif (not moved_toward_fav) and pub > 0.50:
            score = -mag * 0.5
        else:
            # Movement aligned with public — no sharp signal
            score = mag * direction * 0.2
    else:
        # No public-split data: movement alone, damped
        score = mag * direction * 0.4
    if steam_flag_fav:
        score += 0.25 if direction > 0 else -0.25
    return max(-1.0, min(1.0, score))


def market_sharpness(market: MarketData, side: Literal["home", "away", "over", "under"]) -> float:
    """Small signal 0..0.015 based on RLM-style sharp indicators.

    RLM = line moves against public money: public hammering one side while
    line drifts toward the other side → sharps are on the other side.

    When public-ticket data is unavailable (most historical rows), fall back
    to a weaker line-movement-only signal: side the line moved TOWARD gets
    a small positive bump, on the premise that late market action is mildly
    more informative than the opener.
    """
    signal = 0.0
    if side in ("home", "away") and market.opener_home_ml_odds is not None:
        # moved_toward_home: home line got shorter (-150 -> -170) = money on home
        moved_toward_home = market.home_ml_odds < market.opener_home_ml_odds
        if market.public_ticket_pct_home is not None:
            pub = market.public_ticket_pct_home
            # Classic RLM: line moves AWAY from public action
            if side == "home" and moved_toward_home and pub < 0.45:
                signal += 0.01
            if side == "away" and (not moved_toward_home) and pub > 0.55:
                signal += 0.01
        else:
            # Fallback: line-move-only signal, magnitude-scaled
            delta = abs(market.home_ml_odds - market.opener_home_ml_odds)
            mag = clamp(delta / 200.0, 0.0, 1.0) * 0.004  # up to 0.4 pp
            if side == "home":
                signal += mag if moved_toward_home else -mag
            else:
                signal += mag if (not moved_toward_home) else -mag
        if market.steam_flag_home:
            signal += 0.005 if side == "home" else -0.005

    if side in ("over", "under") and market.opener_total is not None:
        total_moved_up = market.total_line > market.opener_total
        if market.public_ticket_pct_over is not None:
            pub = market.public_ticket_pct_over
            if side == "over" and (not total_moved_up) and pub < 0.45:
                signal += 0.01
            if side == "under" and total_moved_up and pub > 0.55:
                signal += 0.01
        else:
            # Fallback: line-move-only. If total moved UP, sharps took over.
            diff = market.total_line - market.opener_total
            mag = clamp(abs(diff) / 1.0, 0.0, 1.0) * 0.004
            if side == "over":
                signal += mag if total_moved_up else -mag
            else:
                signal += mag if (not total_moved_up) else -mag
        if market.steam_flag_over:
            signal += 0.005 if side == "over" else -0.005

    return clamp(signal, -0.015, 0.015)
