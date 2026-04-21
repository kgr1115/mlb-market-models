"""
Short-window rolling stats that replace season-long inputs.

Why shorter windows
-------------------
Season-long wRC+ and SIERA carry the weight of the whole year. By July
a hot-streaking .290 hitter in June still shows up as his season-line
.255. The market uses a 14-21 day attention window intraday, and since
the memory-file `project_train_test_finding.md` confirms our season-
length features carry no out-of-sample edge, this module biases the
model toward shorter, fresher data.

Selection order used by `preferred_stat()`:
    1. 14-day rolling (if sample ≥ window_min_14)
    2. 30-day rolling (if sample ≥ window_min_30)
    3. season-to-date
    4. projection (ROS)
    5. league average

Every returned value carries a `source` tag indicating which window
produced it, so downstream consumers can reweight or log.

This module is intentionally data-source-agnostic: it takes raw sample
observations in, no I/O. Callers load the samples from whatever source
they prefer (pybaseball Statcast pulls, FanGraphs game logs, MLB Stats
API splits) and hand the dicts in here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from predictors import (
    BullpenStats, OffenseStats, PitcherStats, LEAGUE,
)


Source = Literal["rolling_14", "rolling_30", "season", "projection", "league"]


# Minimum sample sizes for each rolling window to be considered "trustworthy".
# Below these, the value is noise and we fall through to the next source.
WINDOW_MIN = {
    "hitter_pa": {"rolling_14": 35, "rolling_30": 80, "season": 150},
    "pitcher_bf": {"rolling_14": 40, "rolling_30": 90, "season": 150},
    "bullpen_innings": {"rolling_14": 8, "rolling_30": 18, "season": 40},
}


# =============================================================================
# Sample containers
# =============================================================================

@dataclass
class HitterRolling:
    """Rolling samples for one hitter across multiple windows."""
    rolling_14: dict = field(default_factory=dict)    # e.g. {"pa": 55, "wOBA": 0.370, "iso": 0.210}
    rolling_30: dict = field(default_factory=dict)
    season: dict = field(default_factory=dict)
    projection: dict = field(default_factory=dict)


@dataclass
class PitcherRolling:
    """Rolling samples for one pitcher across multiple windows."""
    rolling_14: dict = field(default_factory=dict)    # {"bf": 70, "xwoba_against": 0.290, "siera": 3.80, ...}
    rolling_30: dict = field(default_factory=dict)
    season: dict = field(default_factory=dict)
    projection: dict = field(default_factory=dict)


@dataclass
class BullpenRolling:
    """Rolling samples for a full bullpen."""
    rolling_14: dict = field(default_factory=dict)    # {"innings": 15.2, "fip": 3.90, "meltdown_pct": 0.11, ...}
    rolling_30: dict = field(default_factory=dict)
    season: dict = field(default_factory=dict)


@dataclass
class StatPick:
    """A stat value along with its provenance."""
    value: float
    source: Source


# =============================================================================
# Preferred-stat selector
# =============================================================================

def preferred_hitter(stat_key: str, r: HitterRolling,
                     league_default: float) -> StatPick:
    """Return the best (freshest, still-trustworthy) value for a hitter stat.

    stat_key is e.g. "wOBA", "wrc_plus", "iso".
    """
    pa_14 = r.rolling_14.get("pa", 0) if r.rolling_14 else 0
    pa_30 = r.rolling_30.get("pa", 0) if r.rolling_30 else 0
    pa_season = r.season.get("pa", 0) if r.season else 0

    if pa_14 >= WINDOW_MIN["hitter_pa"]["rolling_14"] and stat_key in r.rolling_14:
        return StatPick(float(r.rolling_14[stat_key]), "rolling_14")
    if pa_30 >= WINDOW_MIN["hitter_pa"]["rolling_30"] and stat_key in r.rolling_30:
        return StatPick(float(r.rolling_30[stat_key]), "rolling_30")
    if pa_season >= WINDOW_MIN["hitter_pa"]["season"] and stat_key in r.season:
        return StatPick(float(r.season[stat_key]), "season")
    if stat_key in r.projection:
        return StatPick(float(r.projection[stat_key]), "projection")
    return StatPick(league_default, "league")


def preferred_pitcher(stat_key: str, r: PitcherRolling,
                      league_default: float) -> StatPick:
    bf_14 = r.rolling_14.get("bf", 0) if r.rolling_14 else 0
    bf_30 = r.rolling_30.get("bf", 0) if r.rolling_30 else 0
    bf_season = r.season.get("bf", 0) if r.season else 0

    if bf_14 >= WINDOW_MIN["pitcher_bf"]["rolling_14"] and stat_key in r.rolling_14:
        return StatPick(float(r.rolling_14[stat_key]), "rolling_14")
    if bf_30 >= WINDOW_MIN["pitcher_bf"]["rolling_30"] and stat_key in r.rolling_30:
        return StatPick(float(r.rolling_30[stat_key]), "rolling_30")
    if bf_season >= WINDOW_MIN["pitcher_bf"]["season"] and stat_key in r.season:
        return StatPick(float(r.season[stat_key]), "season")
    if stat_key in r.projection:
        return StatPick(float(r.projection[stat_key]), "projection")
    return StatPick(league_default, "league")


def preferred_bullpen(stat_key: str, r: BullpenRolling,
                      league_default: float) -> StatPick:
    ip_14 = r.rolling_14.get("innings", 0) if r.rolling_14 else 0
    ip_30 = r.rolling_30.get("innings", 0) if r.rolling_30 else 0
    ip_season = r.season.get("innings", 0) if r.season else 0

    if ip_14 >= WINDOW_MIN["bullpen_innings"]["rolling_14"] and stat_key in r.rolling_14:
        return StatPick(float(r.rolling_14[stat_key]), "rolling_14")
    if ip_30 >= WINDOW_MIN["bullpen_innings"]["rolling_30"] and stat_key in r.rolling_30:
        return StatPick(float(r.rolling_30[stat_key]), "rolling_30")
    if ip_season >= WINDOW_MIN["bullpen_innings"]["season"] and stat_key in r.season:
        return StatPick(float(r.season[stat_key]), "season")
    return StatPick(league_default, "league")


# =============================================================================
# Build full stats objects with rolling preference
# =============================================================================

def build_offense_from_rolling(team_hitters: list[HitterRolling],
                                pa_weights: Optional[list[float]] = None
                                ) -> OffenseStats:
    """Weighted per-hitter rollup using each hitter's preferred source.

    Each hitter contributes their freshest reliable sample (14d > 30d > season).
    Team rates are PA-weighted across those freshest samples.
    """
    if not team_hitters:
        return OffenseStats()
    if pa_weights is None:
        # Use implicit PA weighting: weight = PA from the chosen window
        pa_weights = []
        for h in team_hitters:
            pick_pa = preferred_hitter("pa", h, league_default=500).value
            pa_weights.append(max(pick_pa, 1.0))

    def _weighted(key: str, league_key: str) -> float:
        total_w = 0.0
        total_v = 0.0
        for h, w in zip(team_hitters, pa_weights):
            v = preferred_hitter(key, h, LEAGUE[league_key]["mean"]).value
            total_w += w
            total_v += v * w
        return total_v / total_w if total_w > 0 else LEAGUE[league_key]["mean"]

    return OffenseStats(
        wrc_plus=_weighted("wrc_plus", "off_wrc_plus"),
        wOBA=_weighted("wOBA", "off_wOBA"),
        xwOBA=_weighted("xwOBA", "off_xwOBA"),
        obp=_weighted("obp", "off_obp"),
        iso=_weighted("iso", "off_iso"),
        barrel_pct=_weighted("barrel_pct", "off_barrel_pct"),
        k_pct=_weighted("k_pct", "off_k_pct"),
        # Top-of-order OBP: take top 3 PA-weighted
        top_of_order_obp=_weighted("obp", "off_top_obp"),
    )


def build_pitcher_from_rolling(r: PitcherRolling, name: str = "Unknown",
                                throws: Literal["L", "R"] = "R") -> PitcherStats:
    pick = lambda k, lk: preferred_pitcher(k, r, LEAGUE[lk]["mean"]).value
    return PitcherStats(
        name=name,
        throws=throws,
        siera=pick("siera", "sp_siera"),
        xfip=pick("xfip", "sp_xfip"),
        k_bb_pct=pick("k_bb_pct", "sp_k_bb_pct"),
        csw_pct=pick("csw_pct", "sp_csw_pct"),
        xwoba_against=pick("xwoba_against", "sp_xwoba_against"),
        ip_per_gs=pick("ip_per_gs", "sp_ip_per_gs"),
        # Explicit 30-day ERA takes r.rolling_30 only (by field name)
        rolling_30d_era=(r.rolling_30.get("era")
                         if r.rolling_30 and r.rolling_30.get("bf", 0) >= 90
                         else LEAGUE["sp_rolling_era"]["mean"]),
    )


def build_bullpen_from_rolling(r: BullpenRolling) -> BullpenStats:
    pick = lambda k, lk: preferred_bullpen(k, r, LEAGUE[lk]["mean"]).value
    return BullpenStats(
        fip=pick("fip", "bp_fip"),
        hi_lev_k_pct=pick("hi_lev_k_pct", "bp_hi_lev_k_pct"),
        meltdown_pct=pick("meltdown_pct", "bp_meltdown_pct"),
        shutdown_pct=pick("shutdown_pct", "bp_shutdown_pct"),
    )


# =============================================================================
# Source-mix reporting
# =============================================================================

def source_mix(team_hitters: list[HitterRolling]) -> dict[Source, int]:
    """Count how many hitters landed at each source — useful for a
    "how fresh is this data?" health dashboard on the web app.
    """
    counts: dict[Source, int] = {
        "rolling_14": 0, "rolling_30": 0, "season": 0,
        "projection": 0, "league": 0,
    }
    for h in team_hitters:
        pick = preferred_hitter("wOBA", h, LEAGUE["off_wOBA"]["mean"])
        counts[pick.source] += 1
    return counts
