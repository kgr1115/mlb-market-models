"""
Team rollup + projection-to-form blender.

Takes a pool of HitterProjection / PitcherProjection objects (typically
one system's ROS) and produces:

    - lineup-weighted OffenseStats for a team (pass an explicit 9-man
      lineup when confirmed lineups are available; otherwise the top-9
      by PA are used as a proxy)
    - PitcherStats for a named starter (matches by player_id or name)
    - BullpenStats aggregated across all non-starter pitchers on the team

Blending (60% projection / 40% rolling form) is applied when current
"rolling" stats are supplied via the `rolling_*` kwargs. If you don't
have rolling data, pass None and the projection stands alone.

The blend coefficients live at the top of this module so you can tune
them without touching callers.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

from predictors import (
    BullpenStats,
    OffenseStats,
    PitcherStats,
    TeamStats,
)

from .projections_models import HitterProjection, PitcherProjection

log = logging.getLogger(__name__)


# --- blend knobs (tune here) -------------------------------------------------

#: How much weight to give the ROS projection when rolling-form data is
#: also supplied. 0.6 means 60% projection / 40% rolling 30-day form.
PROJECTION_WEIGHT = 0.60
ROLLING_WEIGHT = 1.0 - PROJECTION_WEIGHT

#: Minimum projected PA (for hitters) or IP (for pitchers) before we
#: trust the projection value over the league-average default.
MIN_PA_FOR_HITTER = 50.0
MIN_IP_FOR_PITCHER = 20.0


def _blend(proj: Optional[float], rolling: Optional[float],
           default: float) -> float:
    """60/40 blend. If one side is None, the other stands alone."""
    if proj is None and rolling is None:
        return default
    if rolling is None:
        return float(proj)
    if proj is None:
        return float(rolling)
    return PROJECTION_WEIGHT * float(proj) + ROLLING_WEIGHT * float(rolling)


# =============================================================================
# Offense rollup
# =============================================================================

def build_offense_stats(
    hitters: Iterable[HitterProjection],
    team: str,
    lineup_player_ids: Optional[Iterable[str]] = None,
    *,
    rolling_wrc_plus: Optional[float] = None,
    rolling_wOBA: Optional[float] = None,
    rolling_obp: Optional[float] = None,
    rolling_iso: Optional[float] = None,
    rolling_k_pct: Optional[float] = None,
) -> OffenseStats:
    """Aggregate per-hitter projections into a team OffenseStats.

    Weighting: each hitter's contribution to team rate stats is weighted
    by their projected PA. This naturally upweights regulars over bench
    bats, which is the right behavior when a confirmed lineup isn't yet
    available. When `lineup_player_ids` IS provided, only those nine are
    included (still PA-weighted within the 9).

    `rolling_*` kwargs are optional 30-day team rolling numbers; when
    supplied, they're blended 40% with the 60% projection.
    """
    team_hitters = [h for h in hitters if h.team == team]
    if lineup_player_ids is not None:
        wanted = set(map(str, lineup_player_ids))
        team_hitters = [h for h in team_hitters if h.player_id in wanted]

    # Fall back to top-9-by-PA if no lineup was given
    if lineup_player_ids is None:
        team_hitters = sorted(team_hitters, key=lambda h: h.pa, reverse=True)[:9]

    total_pa = sum(h.pa for h in team_hitters if h.pa > 0)
    if total_pa < MIN_PA_FOR_HITTER * 3:   # too thin to trust — use defaults
        log.debug("Offense rollup: thin PA sample for %s (%.0f PA across %d); "
                  "using defaults blended with rolling if provided",
                  team, total_pa, len(team_hitters))
        proj_wrc = proj_woba = proj_obp = proj_iso = proj_kpct = None
        top_obp = None
    else:
        def wavg(key: str) -> float:
            num = sum(getattr(h, key) * h.pa for h in team_hitters if h.pa > 0)
            return num / total_pa

        proj_wrc = wavg("wrc_plus")
        proj_woba = wavg("woba")
        proj_obp = wavg("obp")
        proj_iso = wavg("iso")
        proj_kpct = wavg("k_pct")
        # Top-of-order OBP: take mean of top 3 by projected OBP
        # (proxy for the 1-2-3 hitters when the actual lineup is unknown)
        top3 = sorted(team_hitters, key=lambda h: h.obp, reverse=True)[:3]
        top_obp = sum(h.obp for h in top3) / 3 if top3 else None

    return OffenseStats(
        wrc_plus=_blend(proj_wrc, rolling_wrc_plus, 100.0),
        wOBA=_blend(proj_woba, rolling_wOBA, 0.320),
        xwOBA=_blend(proj_woba, rolling_wOBA, 0.320),   # proj has no xwOBA, reuse wOBA
        obp=_blend(proj_obp, rolling_obp, 0.320),
        iso=_blend(proj_iso, rolling_iso, 0.165),
        k_pct=_blend(proj_kpct, rolling_k_pct, 0.225),
        top_of_order_obp=(top_obp if top_obp is not None else 0.340),
        # These come from Statcast — not projections — so they're left at
        # the league default; the DRA+/Statcast layer fills them when run.
    )


# =============================================================================
# Pitcher rollup
# =============================================================================

def build_pitcher_stats(
    pitchers: Iterable[PitcherProjection],
    team: str,
    starter_name_or_id: str,
    *,
    throws: str = "R",
    rolling_siera: Optional[float] = None,
    rolling_k_bb_pct: Optional[float] = None,
    rolling_ip_per_gs: Optional[float] = None,
    rolling_30d_era: Optional[float] = None,
) -> PitcherStats:
    """Build a PitcherStats for tonight's starter.

    Lookup order: exact player_id match → case-insensitive name substring.
    If no match, returns a PitcherStats with league-average defaults.
    """
    match: Optional[PitcherProjection] = None
    key = starter_name_or_id.lower().strip()
    for p in pitchers:
        if p.team != team:
            continue
        if p.player_id == starter_name_or_id or p.name.lower() == key:
            match = p
            break
    if match is None:
        for p in pitchers:
            if p.team == team and key in p.name.lower():
                match = p
                break

    if match is None:
        log.warning("build_pitcher_stats: no match for %r on %s", starter_name_or_id, team)
        return PitcherStats(name=starter_name_or_id, throws=throws)

    # Projection has FIP/ERA/K-BB%/IP but not SIERA/CSW%/xwOBA-against.
    # We use FIP as a SIERA proxy — not perfect but directionally correct.
    ip_per_gs = match.ip / match.gs if match.gs > 0 else 5.3

    return PitcherStats(
        name=match.name,
        throws=throws,
        siera=_blend(match.fip, rolling_siera, 4.05),
        xfip=_blend(match.fip, rolling_siera, 4.10),
        k_bb_pct=_blend(match.k_bb_pct, rolling_k_bb_pct, 0.135),
        csw_pct=0.285,   # projection doesn't expose it; set from LEAGUE mean
        xwoba_against=0.320,
        ip_per_gs=_blend(ip_per_gs, rolling_ip_per_gs, 5.30),
        rolling_30d_era=_blend(match.era, rolling_30d_era, 4.05),
    )


# =============================================================================
# Bullpen rollup
# =============================================================================

def build_bullpen_stats(
    pitchers: Iterable[PitcherProjection],
    team: str,
    starter_name_or_id: Optional[str] = None,
) -> BullpenStats:
    """Aggregate non-starters on a team into a BullpenStats.

    "Non-starter" heuristic: pitchers with GS/G < 0.5 and more than 5 G
    projected. Excludes the named starter if provided. Falls back to
    league-average BullpenStats if the pool is too thin.
    """
    team_pit = [p for p in pitchers if p.team == team]
    if starter_name_or_id:
        key = starter_name_or_id.lower().strip()
        team_pit = [
            p for p in team_pit
            if p.player_id != starter_name_or_id and key not in p.name.lower()
        ]

    relievers = [
        p for p in team_pit
        if p.g > 5 and (p.gs / p.g if p.g > 0 else 1.0) < 0.5
    ]
    total_ip = sum(r.ip for r in relievers if r.ip > 0)
    if total_ip < MIN_IP_FOR_PITCHER or not relievers:
        log.debug("Bullpen rollup: thin IP for %s (%.0f across %d relievers); "
                  "using defaults", team, total_ip, len(relievers))
        return BullpenStats()

    def wavg(key: str) -> float:
        num = sum(getattr(r, key) * r.ip for r in relievers if r.ip > 0)
        return num / total_ip

    return BullpenStats(
        fip=wavg("fip"),
        hi_lev_k_pct=wavg("k_pct"),   # projection doesn't split by leverage
        meltdown_pct=0.14,            # proj doesn't have SD/MD; use league mean
        shutdown_pct=0.30,
        # Day-of rest/pitch-count fields are filled by the MLB Stats API loader,
        # not by projections.
    )


# =============================================================================
# Full team builder (convenience)
# =============================================================================

def build_team_stats(
    hitters: Iterable[HitterProjection],
    pitchers: Iterable[PitcherProjection],
    team: str,
    starter_name_or_id: str,
    *,
    is_home: bool = True,
    starter_throws: str = "R",
    lineup_player_ids: Optional[Iterable[str]] = None,
    rolling_wrc_plus: Optional[float] = None,
    rolling_siera: Optional[float] = None,
) -> TeamStats:
    """One-shot helper: full TeamStats from projection pools.

    This is the function most callers want. Pass the ROS projection
    pools, tonight's probable starter, and (optionally) the confirmed
    lineup. Returns a fully populated TeamStats ready for predict_all().
    """
    offense = build_offense_stats(
        hitters, team,
        lineup_player_ids=lineup_player_ids,
        rolling_wrc_plus=rolling_wrc_plus,
    )
    pitcher = build_pitcher_stats(
        pitchers, team, starter_name_or_id,
        throws=starter_throws,
        rolling_siera=rolling_siera,
    )
    bullpen = build_bullpen_stats(
        pitchers, team, starter_name_or_id=starter_name_or_id,
    )
    return TeamStats(
        name=team,
        is_home=is_home,
        pitcher=pitcher,
        bullpen=bullpen,
        offense=offense,
        # defense is Statcast-driven — leave at defaults until the OAA
        # loader lands; the DRA+ approximator will also fill framing.
        lineup_confirmed=lineup_player_ids is not None,
        starter_confirmed=True,
    )
