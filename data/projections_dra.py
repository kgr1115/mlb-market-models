"""
Custom DRA+ approximator.

Baseball Prospectus' DRA (Deserved Run Average) is their best pitcher
evaluation metric — it isolates pitcher skill from defense, park,
catcher framing, and opponent quality. DRA+ is the league-relative
version where 100 = average and >100 = better than average.

We can't replicate DRA exactly (it uses proprietary PITCHf/x-era mixed
models), but we can approximate the same spirit using free Statcast
data:

    raw = xwOBA-against        # "what the batter earned" — removes BABIP luck
    - framing_credit           # catcher's framing runs per PA, opponent-faced
    + park_adjustment          # Coors inflates ERA; neutral-park adjust
    + opponent_adjustment      # faced weak lineups? adjust up. Strong? down.

Then normalize to a league-relative index where 100 = mean.

Inputs are supplied by the caller (typically pulled from pybaseball
+ Savant); this module is pure math so it has no network or pandas
dependency.
"""
from __future__ import annotations

from dataclasses import replace
from statistics import mean, pstdev
from typing import Iterable, Optional

from .projections_models import PitcherDraPlus


# League baselines (2023–2025 era). The predictor's LEAGUE dict already
# has these; duplicated here so the DRA+ math is self-contained.
_LEAGUE_XWOBA_AGAINST = 0.320
_LEAGUE_RUNS_PER_PA = 0.118          # approx — 4.5 R/G / 38 PA/G

# Framing: a catcher with +10 framing runs over 6,000 PA is worth
# ~0.00167 runs per PA. We'll apply this proportionally.
_FRAMING_RUN_PER_PA_AT_PLUS_ONE = 0.000167


def _runs_above_avg_from_xwoba(xwoba: float, pa: int) -> float:
    """Convert xwOBA-against to runs above average.

    A +0.010 delta in wOBA ≈ +0.00125 runs per PA (wOBA coefficients).
    So (xwoba - league) * 0.125 * PA ≈ extra runs allowed.
    """
    return (xwoba - _LEAGUE_XWOBA_AGAINST) * 0.125 * pa


def compute_pitcher_dra_plus(
    *,
    player_id: str,
    name: str,
    xwoba_against: float,
    sample_pa: int,
    catcher_framing_runs_season: float = 0.0,
    catcher_pa_faced: int = 0,
    park_run_factor: float = 1.00,
    opponent_wrc_plus_faced: float = 100.0,
) -> PitcherDraPlus:
    """Compute one pitcher's DRA+.

    All inputs are per-pitcher season-to-date aggregates the caller
    assembles from Statcast + Savant + schedule data:

    - xwoba_against : season xwOBA allowed (Statcast `xwoba` on the
                      pitcher leaderboard).
    - sample_pa     : PA faced this season. Pitchers with very few PA
                      get heavy regression to mean via the `stability`
                      adjustment below.
    - catcher_framing_runs_season : aggregated runs saved/cost by the
                      catchers who caught this pitcher this season.
    - catcher_pa_faced : PA those catchers were behind the plate for.
    - park_run_factor : multiplicative park factor for runs (1.00 = neutral).
    - opponent_wrc_plus_faced : weighted-average wRC+ of lineups faced
                      this season. Higher → we adjust the pitcher's
                      value *up* (they allowed fewer runs *against
                      tougher opposition* than the raw number suggests).

    Returns a PitcherDraPlus with the scaled 100-centered index and
    the individual components preserved for explainability.
    """
    # --- 1) Start from xwOBA-against → runs above average ---
    runs_above_raw = _runs_above_avg_from_xwoba(xwoba_against, sample_pa)

    # --- 2) Framing credit: remove the catcher's contribution ---
    # If catchers saved runs, the raw xwOBA looks better than the
    # pitcher actually was → subtract their credit from the pitcher.
    framing_per_pa = (
        catcher_framing_runs_season / max(catcher_pa_faced, 1)
        if catcher_pa_faced > 0 else 0.0
    )
    framing_adjustment = framing_per_pa * sample_pa

    # --- 3) Park adjustment: neutralize to league park ---
    # A park_run_factor of 1.10 means the park inflated runs by 10%.
    # We DEFLATE the pitcher's raw runs-allowed-above-avg to neutralize.
    park_runs_inflation = (park_run_factor - 1.0) * _LEAGUE_RUNS_PER_PA * sample_pa
    park_adjustment = -park_runs_inflation  # subtract the park's contribution

    # --- 4) Opponent-quality adjustment ---
    # If they faced lineups averaging 110 wRC+ (10% above avg), their raw
    # xwOBA-against would be expected to be ~0.010 higher than avg just
    # from opponent strength. Credit them for that.
    opp_delta = (opponent_wrc_plus_faced - 100.0) / 100.0     # e.g. 0.10
    opponent_runs_expected = opp_delta * _LEAGUE_RUNS_PER_PA * sample_pa
    opponent_adjustment = -opponent_runs_expected    # tougher opp → reduce their "penalty"

    # --- 5) Stability / regression to mean ---
    # Small-sample pitchers shouldn't get extreme DRA+ values. Regress
    # toward 0 using an empirical shrinkage factor: at 400 PA we trust
    # 60% of the signal; at 100 PA, 20%.
    stability = sample_pa / (sample_pa + 300.0)  # 0..~0.7 in typical range

    net_runs_above = stability * (
        runs_above_raw - framing_adjustment + park_adjustment + opponent_adjustment
    )

    # --- 6) Convert to DRA+ scale ---
    # DRA+ = 100 + (runs_saved_per_9_ip / scale) * points_per_scale
    # We use 15 points per 1 standard deviation of runs_above across MLB
    # (roughly 12 runs over a half-season for qualified starters).
    # Runs above mean saved if net < 0, so invert sign for "good=higher".
    runs_per_9 = -net_runs_above / max(sample_pa, 1) * 38.0  # 38 PA per 9 IP
    # Calibrate: a typical ace saves ~1.0 R/9 vs avg → DRA+ ≈ 125
    dra_plus = 100.0 + runs_per_9 * 25.0

    return PitcherDraPlus(
        player_id=player_id,
        name=name,
        dra_plus=round(dra_plus, 1),
        raw_xwoba_against=xwoba_against,
        framing_adjustment=round(framing_adjustment, 2),
        park_adjustment=round(park_adjustment, 2),
        opponent_adjustment=round(opponent_adjustment, 2),
        sample_pa=sample_pa,
    )


def normalize_league_relative(results: Iterable[PitcherDraPlus]
                              ) -> list[PitcherDraPlus]:
    """Post-hoc: rescale a batch so the actual mean is 100 and SD ≈ 15.

    Useful if you're computing DRA+ for a whole league at once and
    want the output to match the 100/15 convention exactly rather than
    relying on the calibration constants above.
    """
    results = list(results)
    if len(results) < 5:
        return results
    vals = [r.dra_plus for r in results]
    m = mean(vals)
    s = pstdev(vals) or 1.0
    return [
        replace(r, dra_plus=round(100.0 + 15.0 * (r.dra_plus - m) / s, 1))
        for r in results
    ]
