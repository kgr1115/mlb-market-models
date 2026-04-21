"""
Prior-year baseline stats provider.

For a 2021 backtest we use 2019 full-season MLB Stats API data as each
team's "pregame baseline" (2020 was the COVID 60-game season — too noisy
to use as a baseline). This makes the backtest leak-free: every game is
evaluated using stats that were unambiguously known before the season
started.

Phase 2 (walk-forward): replace this module with a YTD-through-the-day-
before provider — same public API, different implementation.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from data.team_names import try_normalize_team
from predictors import (
    BullpenStats, DefenseStats, OffenseStats, PitcherStats, TeamStats,
)

log = logging.getLogger(__name__)

_USER_AGENT = "BBP-Backtest/0.1"
_REQ_TIMEOUT = 20.0


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_REQ_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --- Metric approximations ---------------------------------------------------
# MLB Stats API exposes traditional stats (AVG, OBP, SLG, K%, BB%, ERA, FIP
# inputs) but NOT FanGraphs-derived ones (wRC+, wOBA, SIERA, CSW%, K-BB%).
# We approximate the FanGraphs ones below so the predictor gets inputs in
# the shape it expects.

# Linear-weights wOBA from traditional AVG/OBP/SLG: rough but good for a baseline.
def _woba_from_obp_slg(obp: float, slg: float) -> float:
    return 0.33 * obp + 0.27 * slg


# wRC+ proxy: scale team OPS relative to league mean, mean=100, SD~=15.
def _wrc_plus_from_ops(team_ops: float, league_ops: float) -> float:
    if league_ops <= 0:
        return 100.0
    return 100.0 * (team_ops / league_ops)


# FIP = (13*HR + 3*(BB+HBP) - 2*K)/IP + FIP_constant
_FIP_CONSTANT = 3.10


def _fip(hr: int, bb: int, k: int, ip: float, hbp: int = 0) -> float:
    if ip <= 0:
        return 4.50
    return (13 * hr + 3 * (bb + hbp) - 2 * k) / ip + _FIP_CONSTANT


def _kbb_pct(k: int, bb: int, bf: int) -> float:
    if bf <= 0:
        return 0.135
    return (k - bb) / bf


@dataclass
class BaselineStats:
    """Holds the prior-year league stats for quick per-team lookup."""
    season: int
    # team canonical name -> OffenseStats / BullpenStats / DefenseStats
    offense: dict = field(default_factory=dict)
    bullpen: dict = field(default_factory=dict)
    defense: dict = field(default_factory=dict)
    # player fullName.lower().strip() -> PitcherStats
    pitchers: dict = field(default_factory=dict)
    # league averages used by the predictor's confidence layer
    league_ops: float = 0.750

    def team_stats(self, team: str, starter_name: str, *,
                   is_home: bool = True) -> TeamStats:
        off = self.offense.get(team) or OffenseStats()
        bp = self.bullpen.get(team) or BullpenStats()
        defn = self.defense.get(team) or DefenseStats()
        sp = self.pitchers.get((starter_name or "").lower().strip()) \
             or PitcherStats()
        return TeamStats(
            name=team, is_home=is_home,
            pitcher=sp, bullpen=bp, offense=off, defense=defn,
            form_last10_win_pct=0.500,
            form_last20_win_pct=0.500,
            rest_days=1, travel_miles_72h=0.0,
            meaningful_game=True,
            lineup_confirmed=False,
            starter_confirmed=bool(starter_name),
        )


def load_baseline(season: int) -> BaselineStats:
    """Load one season's worth of team + pitcher baselines."""
    log.info("baseline: loading %d team hitting / pitching / individual pitching",
             season)

    # --- Team hitting ---
    url_h = (f"https://statsapi.mlb.com/api/v1/teams/stats?stats=season"
             f"&group=hitting&season={season}&sportIds=1")
    data_h = _get_json(url_h)
    offense: dict[str, OffenseStats] = {}

    league_ops_num = league_ops_den = 0
    for sp in data_h["stats"][0]["splits"]:
        team = try_normalize_team(sp["team"]["name"])
        if not team:
            continue
        st = sp["stat"]

        def _f(k, default):
            v = st.get(k)
            if v is None or v == "":
                return default
            try:
                return float(v)
            except ValueError:
                return default

        obp = _f("obp", 0.320)
        slg = _f("slg", 0.410)
        ops = _f("ops", 0.730)
        ab = _f("atBats", 1.0)
        k = st.get("strikeOuts", 0) or 0
        pa = st.get("plateAppearances", 0) or ab * 1.14
        hr = st.get("homeRuns", 0) or 0
        hits = st.get("hits", 0) or 0
        doubles = st.get("doubles", 0) or 0
        triples = st.get("triples", 0) or 0
        # ISO = SLG - AVG
        avg = _f("avg", 0.250)
        iso = max(0.0, slg - avg)
        woba = _woba_from_obp_slg(obp, slg)
        kpct = (k / pa) if pa > 0 else 0.225

        offense[team] = OffenseStats(
            wOBA=woba, xwOBA=woba,                   # no xStats in this API
            iso=iso, obp=obp, k_pct=kpct,
            top_of_order_obp=min(0.400, obp + 0.020),
            wrc_plus=100.0,                           # filled below once league OPS known
        )
        league_ops_num += ops * (ab or 1)
        league_ops_den += (ab or 1)

    league_ops = (league_ops_num / league_ops_den) if league_ops_den > 0 \
                  else 0.730

    # Fill in wRC+ now that we know league OPS
    for team, off in offense.items():
        # Rebuild OPS from OBP+SLG proxies: slg = (iso + avg_proxy), where
        # avg_proxy ~= obp - league_bb_rate_adj; simpler: use approximate
        # team OPS = obp + (iso + 0.250)
        team_ops = off.obp + (off.iso + 0.250)
        off.wrc_plus = round(_wrc_plus_from_ops(team_ops, league_ops), 1)

    # --- Team pitching (for bullpen aggregate only, as starter varies per game) ---
    url_p = (f"https://statsapi.mlb.com/api/v1/teams/stats?stats=season"
             f"&group=pitching&season={season}&sportIds=1")
    data_p = _get_json(url_p)
    bullpen: dict[str, BullpenStats] = {}
    for sp in data_p["stats"][0]["splits"]:
        team = try_normalize_team(sp["team"]["name"])
        if not team:
            continue
        st = sp["stat"]
        ip_str = str(st.get("inningsPitched", "0.0"))
        # MLB format IP: "161.1" = 161 + 1/3
        try:
            whole, thirds = ip_str.split(".")
            ip = int(whole) + int(thirds) / 3.0
        except ValueError:
            ip = 0.0
        k = int(st.get("strikeOuts", 0) or 0)
        bb = int(st.get("baseOnBalls", 0) or 0)
        hr = int(st.get("homeRuns", 0) or 0)
        bf = int(st.get("battersFaced", 0) or 0) or int(ip * 4.3)
        bullpen[team] = BullpenStats(
            fip=_fip(hr, bb, k, ip),
            hi_lev_k_pct=(k / bf) if bf > 0 else 0.230,
            meltdown_pct=0.14, shutdown_pct=0.30,
        )

    defense = {t: DefenseStats() for t in offense}  # traditional API has no OAA

    # --- Individual pitcher stats (for SP lookup) ---
    pitchers: dict[str, PitcherStats] = {}
    # Paginate: endpoint caps at ~200 per page; pull aggressively
    offset = 0
    batch = 200
    while True:
        url_i = (f"https://statsapi.mlb.com/api/v1/stats?stats=season"
                 f"&group=pitching&season={season}&sportIds=1"
                 f"&limit={batch}&offset={offset}&playerPool=all")
        try:
            data_i = _get_json(url_i)
        except Exception as e:
            log.warning("pitcher stats page %d failed: %s", offset, e)
            break
        splits = data_i["stats"][0].get("splits", [])
        if not splits:
            break
        for sp in splits:
            p = sp.get("player", {})
            name = (p.get("fullName") or "").lower().strip()
            if not name:
                continue
            st = sp["stat"]
            ip_str = str(st.get("inningsPitched", "0.0"))
            try:
                whole, thirds = ip_str.split(".")
                ip = int(whole) + int(thirds) / 3.0
            except ValueError:
                ip = 0.0
            k = int(st.get("strikeOuts", 0) or 0)
            bb = int(st.get("baseOnBalls", 0) or 0)
            hr = int(st.get("homeRuns", 0) or 0)
            bf = int(st.get("battersFaced", 0) or 0) or int(ip * 4.3)
            gs = int(st.get("gamesStarted", 0) or 0)
            era = st.get("era")
            try:
                era = float(era) if era else 4.50
            except ValueError:
                era = 4.50
            fip = _fip(hr, bb, k, ip)
            kbb = _kbb_pct(k, bb, bf)
            ip_per_gs = (ip / gs) if gs > 0 else 5.3

            pitchers[name] = PitcherStats(
                name=p.get("fullName") or "",
                throws=(p.get("pitchHand", {}) or {}).get("code", "R"),
                siera=fip, xfip=fip,
                k_bb_pct=kbb,
                csw_pct=0.285,
                xwoba_against=0.320,
                ip_per_gs=ip_per_gs,
                rolling_30d_era=era,
            )
        if len(splits) < batch:
            break
        offset += batch

    log.info("baseline: offense=%d  bullpen=%d  pitchers=%d  league_ops=%.3f",
             len(offense), len(bullpen), len(pitchers), league_ops)
    return BaselineStats(
        season=season, offense=offense, bullpen=bullpen,
        defense=defense, pitchers=pitchers, league_ops=league_ops,
    )
