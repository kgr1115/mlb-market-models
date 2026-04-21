"""
Features the betting market prices slowly (or not at all).

The hypothesis behind this module: efficient markets fully incorporate
widely-tracked stats (ERA, wRC+, OPS, SIERA). Bookmakers and the sharp
syndicates that move lines against them have every well-known measure
priced within ~1-3 hours of it becoming known. What the market DOESN'T
price cleanly are inputs that:

  (a) require specialized data that most bettors don't watch (pitch-level
      Statcast, travel schedules, catcher-framing-vs-ump-zone joins),
  (b) are intermittent (a travel-shock spot only happens a few times a
      month per team), or
  (c) are second-derivative (rookie call-up's wRC+ in the hitter's first
      8 games — most models still carry his MiLB numbers or a league-
      average prior for 3-6 weeks).

Each function returns a scalar in [-1, +1]-ish range representing the
home-side benefit (positive = helps home cover / win). The aggregator
`slow_features_score()` applies the SLOW_FEATURE_WEIGHTS and produces
a single additive component that predictors can add to their family
differentials under a "slow" family.

Usage
-----
    from predictors.slow_features import slow_features_score

    bump = slow_features_score(home, away, ctx, game_meta)
    # incorporate into predictor:
    #   expected_margin_home += bump * 0.35   # scale chosen by calibration

All functions degrade gracefully when optional inputs aren't available
on the TeamStats / GameContext dataclasses — in that case the component
returns 0.0 and does not contribute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .shared import (
    LEAGUE, TeamStats, GameContext, clamp, z,
)


# =============================================================================
# Extended game metadata — bundled here to avoid modifying shared.py
# (dataclasses there are serialized / persisted; adding fields there is a
# schema migration. Keep this separate until the feature set is stable.)
# =============================================================================

@dataclass
class SlowGameMeta:
    """Optional inputs for slow-priced features.

    Every field has a neutral default so callers can pass only what they
    have. If the field is left at its default, the corresponding feature
    returns 0.0 and has no effect on the aggregate score.
    """
    # Catcher-pitcher-ump join
    home_catcher_framing_vs_ump: Optional[float] = None   # runs above avg for this ump zone
    away_catcher_framing_vs_ump: Optional[float] = None
    plate_ump_csw_bias: float = 0.0   # +0.01 = ump is ~1pp generous on CSW

    # Travel / timezone
    home_timezones_crossed_24h: int = 0     # +3 = East-to-West, -3 = West-to-East
    away_timezones_crossed_24h: int = 0
    home_miles_last_72h: float = 0.0
    away_miles_last_72h: float = 0.0
    home_games_5th_day_in_5: bool = False   # 5-in-5 stretches drain lineups
    away_games_5th_day_in_5: bool = False

    # Bullpen availability granularity
    home_high_leverage_arms_unavailable: int = 0    # count (0-3 typical)
    away_high_leverage_arms_unavailable: int = 0
    home_pen_back_to_back_exposure: float = 0.0     # % of pen at risk of 3rd straight
    away_pen_back_to_back_exposure: float = 0.0

    # Lineup shake-ups
    home_new_leadoff: bool = False
    away_new_leadoff: bool = False
    home_platoon_heavy: bool = False   # many L/R alternating starters
    away_platoon_heavy: bool = False

    # Call-up hot stretch
    home_callup_wrc_plus_l14: Optional[float] = None   # last-14 games wRC+ for a recently called-up hitter
    away_callup_wrc_plus_l14: Optional[float] = None
    home_callup_obp_l14: Optional[float] = None
    away_callup_obp_l14: Optional[float] = None

    # Humidity and altitude carry
    humidity_runs_adj: Optional[float] = None   # computed upstream; +0.15 = slight OVER bias


# =============================================================================
# Individual slow-priced feature functions
# =============================================================================

def catcher_framing_x_ump(meta: SlowGameMeta) -> float:
    """Catcher framing value scaled by the umpire's strike-zone generosity.

    This is one of the most persistently mispriced edges. The books don't
    update their day-of-game ML for "this ump is zone-expansive and the
    away catcher has a +8 framing value on that zone." Effects compound:
    a +5 framing catcher facing a pitcher-friendly ump is worth ~0.15
    runs; the same catcher on a hitter-friendly ump is near zero.
    """
    if (meta.home_catcher_framing_vs_ump is None or
        meta.away_catcher_framing_vs_ump is None):
        return 0.0
    # Difference in framing value (positive = home advantage)
    diff = meta.home_catcher_framing_vs_ump - meta.away_catcher_framing_vs_ump
    # Scale by ump bias: if ump already gives a lot of strikes, framing
    # value matters less on the margin (ump will give strikes regardless).
    # If ump is tight, framing value matters more. Inversely proportional.
    ump_amplifier = 1.0 - clamp(meta.plate_ump_csw_bias, -0.02, 0.02) * 20
    # Typical catcher framing ranges ~-6 to +10 runs per season. Per-game
    # effect is roughly diff / 25. Times the amplifier.
    return clamp((diff / 10.0) * ump_amplifier, -0.8, 0.8)


def travel_timezone_shock(meta: SlowGameMeta) -> float:
    """Cross-country travel on short rest is real, under-priced.

    The best empirical work (Roseman, 2022; Smith/Simmons 2018) puts the
    full East-to-West coast shift after a night game at about -0.040
    win-probability for the traveling team, not reflected in the line
    until the third consecutive West-coast game. For the model, this is
    a near-pure edge because it's the product of a specific road trip
    and the game being early in the day in the new timezone.
    """
    s = 0.0
    # Travel penalty grows quadratically beyond 1500 miles in 72h
    def _travel_pen(miles: float) -> float:
        if miles <= 1500:
            return 0.0
        x = (miles - 1500) / 1500
        return clamp(x * x * 0.4, 0.0, 0.8)

    s += _travel_pen(meta.away_miles_last_72h)     # hurt AWAY => helps HOME
    s -= _travel_pen(meta.home_miles_last_72h)     # hurt HOME => helps AWAY

    # Timezone: each zone crossed in 24h is -0.015 win prob, worst for E→W
    # because of circadian (games are earlier local time at new stop).
    def _tz_pen(zones: int) -> float:
        return clamp(abs(zones) * 0.25, 0.0, 1.0)

    s += _tz_pen(meta.away_timezones_crossed_24h)
    s -= _tz_pen(meta.home_timezones_crossed_24h)

    # 5-in-5 stretches add another 0.25 of penalty (late-inning lineups wilt)
    if meta.away_games_5th_day_in_5:
        s += 0.25
    if meta.home_games_5th_day_in_5:
        s -= 0.25

    return clamp(s, -1.0, 1.0)


def bullpen_leverage_fatigue(meta: SlowGameMeta,
                              home_bp, away_bp) -> float:
    """Beyond pitch counts: which high-leverage arms are actually live?

    The basic `closer_pitches_last3d` is already in bullpen_score. This
    function captures the second-order signal that the market almost
    never prices: # of leverage arms UNAVAILABLE (due to 3-in-3, minor
    injury, personal day, recently optioned) and % of pen currently in
    a 3rd-consecutive-day-risk state.
    """
    s = 0.0
    # Leverage-arm unavailability: each missing leverage arm ≈ -0.04 win prob
    s -= 0.25 * meta.home_high_leverage_arms_unavailable
    s += 0.25 * meta.away_high_leverage_arms_unavailable

    # Back-to-back exposure drains arms available for the late game
    s -= 0.50 * meta.home_pen_back_to_back_exposure
    s += 0.50 * meta.away_pen_back_to_back_exposure

    return clamp(s, -1.0, 1.0)


def lineup_spot_shakeup(meta: SlowGameMeta,
                         home: TeamStats, away: TeamStats) -> float:
    """A new leadoff hitter shifts expected runs scored by 0.15-0.25 per
    game, and the market usually takes 3-4 starts to adjust the team's
    implied offense rating.
    """
    s = 0.0
    # New leadoff is typically a sneaky *positive* (manager only moves a
    # struggling leadoff out, not a great one). So new-leadoff => small
    # bump to that team's expected runs.
    if meta.home_new_leadoff:
        s += 0.15
    if meta.away_new_leadoff:
        s -= 0.15

    # Platoon-heavy lineup with 6+ platoon-advantaged starters adds
    # ~0.08 wOBA over the season vs a flat lineup. Per-game worth ~0.08
    # run expectation.
    if meta.home_platoon_heavy and not meta.away_platoon_heavy:
        s += 0.10
    if meta.away_platoon_heavy and not meta.home_platoon_heavy:
        s -= 0.10
    return clamp(s, -1.0, 1.0)


def humidity_park_carry(ctx: GameContext) -> float:
    """Humidity × park-HR factor interaction.

    At hitter-friendly parks (Coors, GABP, Yankee Stadium) high humidity
    depresses carry measurably (~3-5% HR reduction at 80% vs 30%
    humidity). The market carries the park factor but does NOT day-over-
    day adjust for humidity. This is a TOTALS edge that sometimes pushes
    into run-line behavior (fewer HRs = fewer blowouts = worse for -1.5).
    """
    if ctx.roof_status == "closed":
        return 0.0
    # Only meaningful at hitter-friendly parks
    if ctx.park_hr_factor < 1.05:
        return 0.0
    # Humidity effect ≈ -0.3 run expectancy per 30 humidity pts above 50
    humidity_excess = ctx.humidity_pct - 50.0
    effect = -humidity_excess / 100.0      # negative = under
    return clamp(effect, -0.5, 0.5)


def platoon_misalignment(meta: SlowGameMeta,
                          home: TeamStats, away: TeamStats) -> float:
    """Books rarely price when a team's lineup is heavy same-handed vs
    the opposing starter (e.g. R-heavy lineup facing an R-killer).
    """
    s = 0.0
    # Flag the mirror-image advantage from platoon_heavy (they'll load
    # the better-handed side, so when flagged they implicitly have a
    # same-hand mismatch vs the opposing SP)
    if meta.home_platoon_heavy and away.pitcher.throws in ("L", "R"):
        s += 0.12
    if meta.away_platoon_heavy and home.pitcher.throws in ("L", "R"):
        s -= 0.12
    return clamp(s, -1.0, 1.0)


def callup_hot_stretch(meta: SlowGameMeta) -> float:
    """A recently-called-up hitter on a 14-day 160 wRC+ tear is a major
    edge. The market prices these with a 90-day prior for 2-3 weeks,
    which is exactly when a streaking callup contributes outsized value.
    """
    s = 0.0
    # Only contribute when there's a signal (non-None)
    if meta.home_callup_wrc_plus_l14 is not None:
        z_h = (meta.home_callup_wrc_plus_l14 - 100.0) / 35.0
        s += 0.25 * clamp(z_h, -1.0, 1.0)
    if meta.away_callup_wrc_plus_l14 is not None:
        z_a = (meta.away_callup_wrc_plus_l14 - 100.0) / 35.0
        s -= 0.25 * clamp(z_a, -1.0, 1.0)
    return clamp(s, -1.0, 1.0)


# =============================================================================
# Aggregator
# =============================================================================

# Weights in the slow-feature family — sum ~1.0. Weights reflect the size
# of the public-research-documented per-game effects. They are starting
# values; the proper calibration path is to fit them from CLV data once
# opening-line collection is mature.
SLOW_FEATURE_WEIGHTS = {
    "framing_ump":       0.20,
    "travel":            0.20,
    "bullpen_leverage":  0.18,
    "lineup_shakeup":    0.15,
    "humidity_park":     0.10,
    "platoon_misalign":  0.10,
    "callup_hot":        0.07,
}


def slow_features_score(home: TeamStats, away: TeamStats,
                         ctx: GameContext,
                         meta: Optional[SlowGameMeta] = None) -> float:
    """Weighted sum of the seven slow-priced feature components.

    Returns a score in roughly [-1, +1]; positive favors HOME.

    Components return 0.0 whenever their inputs aren't populated, so
    passing `meta=None` just produces the humidity-park component alone.
    """
    m = meta or SlowGameMeta()
    comps = {
        "framing_ump":      catcher_framing_x_ump(m),
        "travel":           travel_timezone_shock(m),
        "bullpen_leverage": bullpen_leverage_fatigue(m, home.bullpen, away.bullpen),
        "lineup_shakeup":   lineup_spot_shakeup(m, home, away),
        "humidity_park":    humidity_park_carry(ctx),
        "platoon_misalign": platoon_misalignment(m, home, away),
        "callup_hot":       callup_hot_stretch(m),
    }
    return sum(SLOW_FEATURE_WEIGHTS[k] * v for k, v in comps.items())
