"""
Backtest engine.

Iterates historical games, builds (TeamStats, MarketData, GameContext)
for each using:
  - Prior-year baseline stats for both teams (load_baseline)
  - Closing lines from SBR for MarketData
  - A simple GameContext (park factor + weather defaults)

Calls predict_all(), grades each non-NO-BET pick against the actual
outcome, and aggregates:
  - Overall ROI in units (1u per bet)
  - Per-market (ML/RL/Totals) record + ROI
  - Confidence-bucket (LOW/LEAN/MEDIUM/HIGH) record + ROI
  - Equity curve (running bankroll starting at 1000)

A "bet" is any prediction whose `pick` is not "NO BET". Unit size: 1.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

from predictors import GameContext, MarketData, predict_all

from .historical_games import HistoricalGame
from .historical_odds import HistoricalOdds
from .historical_stats import BaselineStats

log = logging.getLogger(__name__)


# Park factors for run-scoring. 2021 baseline numbers from Baseball-Savant.
# (Simplified; we use the same static map across seasons for now.)
_PARK_RUN_FACTOR = {
    "Colorado Rockies": 1.18,
    "Boston Red Sox": 1.07,
    "Cincinnati Reds": 1.05,
    "Baltimore Orioles": 1.05,
    "Texas Rangers": 1.03,
    "Philadelphia Phillies": 1.02,
    "Chicago White Sox": 1.02,
    "Toronto Blue Jays": 1.02,
    "Arizona Diamondbacks": 1.01,
    # ...rest default to 1.00
    "San Francisco Giants": 0.92,
    "Oakland Athletics": 0.93,
    "Miami Marlins": 0.93,
    "Seattle Mariners": 0.96,
    "San Diego Padres": 0.96,
    "New York Mets": 0.97,
    "St. Louis Cardinals": 0.98,
}

# HR-specific park factors. Many parks play differently for homers than
# for total runs — short porches, jet streams, and elevation spike HRs
# without commensurate singles/doubles gains.
_PARK_HR_FACTOR = {
    "Colorado Rockies": 1.14,       # thin air helps fly balls more than base hits
    "Cincinnati Reds": 1.17,        # Great American Small Park
    "Baltimore Orioles": 1.25,      # pre-2022 wall; still HR-friendly after
    "New York Yankees": 1.12,       # short right-field porch
    "Philadelphia Phillies": 1.06,
    "Chicago White Sox": 1.08,
    "Chicago Cubs": 1.05,           # with the wind
    "Milwaukee Brewers": 1.06,
    "Boston Red Sox": 1.00,         # Green Monster hurts HR despite doubles spike
    "Arizona Diamondbacks": 1.04,
    "Detroit Tigers": 0.94,
    "Oakland Athletics": 0.88,      # HR-killer coliseum
    "San Francisco Giants": 0.88,   # Oracle marine layer
    "Miami Marlins": 0.92,
    "Seattle Mariners": 0.93,
    "San Diego Padres": 0.95,
    "Kansas City Royals": 0.94,
    "Pittsburgh Pirates": 0.95,
    "St. Louis Cardinals": 0.96,
}

# League runs-per-team-per-game by year (Baseball-Reference).
# Used to compute season-specific drift correction for the totals model:
# team stats come from baseline year, but league run environment shifts.
LEAGUE_RPG = {
    2015: 4.25,
    2016: 4.48,
    2017: 4.65,
    2018: 4.45,
    2019: 4.83,
    2020: 4.65,
    2021: 4.53,
    2022: 4.28,
    2023: 4.62,
    2024: 4.39,
    2025: 4.39,
}


def _league_run_drift(season: int, baseline_season: int) -> float:
    """Total-runs drift between current and baseline season.

    We project totals using team stats from baseline_season, but the
    market line reflects season's current run environment. Shift our
    projection by 2 * (current_rpg - baseline_rpg) to bridge the gap.
    """
    cur = LEAGUE_RPG.get(season)
    base = LEAGUE_RPG.get(baseline_season)
    if cur is None or base is None:
        return 0.0
    return 2.0 * (cur - base)


def _build_rolling_form(games, league_rpg: float, window: int = 30,
                         min_games: int = 15):
    """For each (team, event_id), return (rs_delta, ra_delta) vs league_rpg.

    Walks games in date order. For each team, keeps a deque of the last
    `window` runs_scored and runs_allowed from *previous* in-season games.
    Any game within the first `min_games` of a team's season uses zeros
    (fall back to prior-year baseline only).

    Returns dict keyed by (team, event_id) -> (rs_delta, ra_delta) where
    delta = rolling_mean - league_rpg. Positive rs_delta => team scoring
    more than league. Positive ra_delta => team allowing more.
    """
    from collections import deque, defaultdict
    rs = defaultdict(lambda: deque(maxlen=window))
    ra = defaultdict(lambda: deque(maxlen=window))
    out = {}
    sorted_games = sorted(games, key=lambda g: (g.game_date, g.game_pk))
    for g in sorted_games:
        h_ok = len(rs[g.home_team]) >= min_games
        a_ok = len(rs[g.away_team]) >= min_games
        if h_ok:
            h_rs = sum(rs[g.home_team]) / len(rs[g.home_team]) - league_rpg
            h_ra = sum(ra[g.home_team]) / len(ra[g.home_team]) - league_rpg
        else:
            h_rs = h_ra = 0.0
        if a_ok:
            a_rs = sum(rs[g.away_team]) / len(rs[g.away_team]) - league_rpg
            a_ra = sum(ra[g.away_team]) / len(ra[g.away_team]) - league_rpg
        else:
            a_rs = a_ra = 0.0
        out[(g.home_team, g.event_id)] = (h_rs, h_ra)
        out[(g.away_team, g.event_id)] = (a_rs, a_ra)
        # Record this game's outcome AFTER using the pre-game rolling state
        rs[g.home_team].append(g.home_runs)
        ra[g.home_team].append(g.away_runs)
        rs[g.away_team].append(g.away_runs)
        ra[g.away_team].append(g.home_runs)
    return out


def _build_rolling_winpct(games, short: int = 10, long: int = 20,
                          min_games: int = 5):
    """For each (team, event_id), return (winpct_last_N_short, winpct_last_N_long).

    Uses the same pre-game perspective as `_build_rolling_form`: values for
    game G reflect the team's record in its N most recent games BEFORE G.
    Any game where the team has fewer than `min_games` prior in-season
    games returns (0.500, 0.500) (league-neutral default).

    This drives `form_last10_win_pct` and `form_last20_win_pct` which
    situational_score() in predictors/moneyline.py reads. Without this,
    those fields were stuck at 0.500 league-mean and the situational
    family was effectively dormant.
    """
    from collections import deque, defaultdict
    recent_short = defaultdict(lambda: deque(maxlen=short))
    recent_long  = defaultdict(lambda: deque(maxlen=long))
    out = {}
    sorted_games = sorted(games, key=lambda g: (g.game_date, g.game_pk))
    for g in sorted_games:
        def _wp(dq, team):
            if len(dq[team]) < min_games:
                return 0.500
            return sum(dq[team]) / len(dq[team])
        hs = _wp(recent_short, g.home_team)
        hl = _wp(recent_long,  g.home_team)
        as_ = _wp(recent_short, g.away_team)
        al  = _wp(recent_long,  g.away_team)
        out[(g.home_team, g.event_id)] = (hs, hl)
        out[(g.away_team, g.event_id)] = (as_, al)
        # Record after-the-fact so future games see this result
        h_won = 1.0 if g.home_runs > g.away_runs else 0.0
        a_won = 1.0 - h_won
        recent_short[g.home_team].append(h_won)
        recent_long[g.home_team].append(h_won)
        recent_short[g.away_team].append(a_won)
        recent_long[g.away_team].append(a_won)
    return out


def _build_rest_days(games):
    """For each (team, event_id), return days since that team's last game.

    Pre-game perspective: the value for game G is the gap between G and
    the team's prior game. First game of a team's season returns 1 (a
    neutral default — no rest/fatigue signal on opening day).
    """
    from collections import defaultdict
    last_date = {}
    out = {}
    sorted_games = sorted(games, key=lambda g: (g.game_date, g.game_pk))
    for g in sorted_games:
        for team in (g.home_team, g.away_team):
            prev = last_date.get(team)
            if prev is None:
                rest = 1
            else:
                try:
                    rest = (g.game_date - prev).days
                except Exception:
                    rest = 1
                if rest < 0:
                    rest = 1
                if rest > 7:
                    rest = 7   # clamp — ASB/IL gaps aren't really "rested"
            out[(team, g.event_id)] = rest
            last_date[team] = g.game_date
    return out


def _apply_rolling_adjust(team_stats, rs_delta: float, ra_delta: float):
    """Nudge team offense and pitching stats based on rolling form.

    Rule of thumb: each run/game above league average ~ +10 wRC+.
    Each run/game allowed above league average ~ +0.25 xFIP.
    Blend 50/50 with prior-year baseline to avoid overfitting to small samples.
    """
    # Offense: blend baseline wRC+ with rolling-implied wRC+
    baseline_wrc = team_stats.offense.wrc_plus
    rolling_wrc = 100.0 + rs_delta * 10.0
    team_stats.offense.wrc_plus = 0.5 * baseline_wrc + 0.5 * rolling_wrc
    # Pitching: only the starting pitcher is game-specific; the team's runs
    # allowed rate captures staff quality including bullpen. Bump xFIP + SIERA
    # proportionally, blended 50/50.
    baseline_xfip = team_stats.pitcher.xfip
    rolling_xfip = 4.10 + ra_delta * 0.25
    team_stats.pitcher.xfip = 0.5 * baseline_xfip + 0.5 * rolling_xfip
    baseline_siera = team_stats.pitcher.siera
    rolling_siera = 4.05 + ra_delta * 0.25
    team_stats.pitcher.siera = 0.5 * baseline_siera + 0.5 * rolling_siera


def _american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal multiplier."""
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def _profit_per_unit(pick_won: bool, odds: int) -> float:
    """Net profit per 1 unit risked. Lose => -1.0, win => (decimal-1)."""
    if not pick_won:
        return -1.0
    return _american_to_decimal(odds) - 1.0


def _build_market_data(od: HistoricalOdds) -> MarketData:
    """Translate an HistoricalOdds row into a MarketData the predictor consumes.

    Populates fair_prob_* fields via per-book de-vig (single-book in
    historical data). Predictors consume these via fair_prob_for_side()
    which falls back to live per-book de-vig if missing.
    """
    from predictors.shared import remove_vig_two_way
    md = MarketData()
    if od.away_ml_close is not None and od.home_ml_close is not None:
        md.away_ml_odds = int(od.away_ml_close)
        md.home_ml_odds = int(od.home_ml_close)
        try:
            h, a = remove_vig_two_way(md.home_ml_odds, md.away_ml_odds)
            md.fair_prob_home_ml = h
            md.fair_prob_away_ml = a
        except Exception:
            pass
    if od.home_ml_open is not None:
        md.opener_home_ml_odds = int(od.home_ml_open)
    if od.away_ml_open is not None:
        md.opener_away_ml_odds = int(od.away_ml_open)
    # Run line: SBR publishes a per-team line; we normalize to -1.5/+1.5 pricing.
    if od.home_rl_line is not None and od.home_rl_price is not None:
        md.home_is_rl_favorite = od.home_rl_line < 0
        md.home_rl_odds = int(od.home_rl_price)
        md.away_rl_odds = int(od.away_rl_price) if od.away_rl_price else -120
        try:
            h, a = remove_vig_two_way(md.home_rl_odds, md.away_rl_odds)
            md.fair_prob_home_rl = h
            md.fair_prob_away_rl = a
        except Exception:
            pass
    if od.total_close is not None:
        md.total_line = float(od.total_close)
        if od.total_over_close:
            md.over_odds = int(od.total_over_close)
        if od.total_under_close:
            md.under_odds = int(od.total_under_close)
        try:
            o, u = remove_vig_two_way(md.over_odds, md.under_odds)
            md.fair_prob_over = o
            md.fair_prob_under = u
        except Exception:
            pass
    # Opening total — lights up totals market family (previously all zeros).
    # Missing for SBR 2018-19; populated by community loader 2021+.
    if getattr(od, "total_open", None) is not None:
        md.opener_total = float(od.total_open)
    if getattr(od, "total_over_open", None) is not None:
        md.opener_over_odds = int(od.total_over_open)
    if getattr(od, "total_under_open", None) is not None:
        md.opener_under_odds = int(od.total_under_open)
    return md


_PARK_UTC_OFFSET_HOURS = {
    # MLB park -> typical UTC offset during season (DST, April-Oct).
    # Used only to classify day vs night from game_time_utc — approximate.
    "Arizona Diamondbacks": -7,    # AZ does not observe DST
    "Boston Red Sox": -4,
    "Baltimore Orioles": -4,
    "New York Yankees": -4,
    "New York Mets": -4,
    "Philadelphia Phillies": -4,
    "Pittsburgh Pirates": -4,
    "Washington Nationals": -4,
    "Cleveland Guardians": -4,
    "Cleveland Indians": -4,
    "Cincinnati Reds": -4,
    "Detroit Tigers": -4,
    "Miami Marlins": -4,
    "Tampa Bay Rays": -4,
    "Atlanta Braves": -4,
    "Toronto Blue Jays": -4,
    "Chicago Cubs": -5,
    "Chicago White Sox": -5,
    "Houston Astros": -5,
    "Kansas City Royals": -5,
    "Milwaukee Brewers": -5,
    "Minnesota Twins": -5,
    "St. Louis Cardinals": -5,
    "Texas Rangers": -5,
    "Colorado Rockies": -6,
    "Los Angeles Angels": -7,
    "Los Angeles Dodgers": -7,
    "Oakland Athletics": -7,
    "San Diego Padres": -7,
    "San Francisco Giants": -7,
    "Seattle Mariners": -7,
}


def _is_day_game(game: HistoricalGame) -> bool:
    """Return True if the game's local first pitch is before 17:00."""
    t = game.game_time_utc
    if t is None:
        return False
    off = _PARK_UTC_OFFSET_HOURS.get(game.home_team, -5)
    local_hour = (t.hour + off) % 24
    return local_hour < 17


def _build_doubleheader_set(games) -> set:
    """Return the set of event_ids whose (date, home, away) has >1 entry.
    The community dataset event-id scheme collides on doubleheaders, so in
    practice only the second/nightcap game survives the dict write — but
    any surviving doubleheader row should still be flagged."""
    counts = {}
    for g in games:
        key = (g.game_date, g.home_team, g.away_team)
        counts[key] = counts.get(key, 0) + 1
    flagged = set()
    for g in games:
        key = (g.game_date, g.home_team, g.away_team)
        if counts.get(key, 0) >= 2:
            flagged.add(g.event_id)
    return flagged


def _build_game_context(game: HistoricalGame,
                        league_run_drift: float = 0.0,
                        doubleheader_set: Optional[set] = None,
                        with_weather: bool = False,
                        ump_rpg_lookup: Optional[dict] = None) -> GameContext:
    """GameContext from known home-team park + derived day/doubleheader flags.

    If ``with_weather`` is True, back-fills temperature_f, humidity_pct,
    wind_speed_mph and wind_direction from the Open-Meteo archive API
    (via data.weather_history). This is opt-in because historical weather
    fetches hit the network and the caller should prewarm the cache.

    If ``ump_rpg_lookup`` is provided, each event_id it contains with a
    non-None value populates ``ctx.ump_runs_per_game`` so the totals
    umpire family fires instead of silently returning zero. See
    ``data/umpire_history.py`` and ``project_post_weather_family_signals.md``.
    """
    ctx = GameContext()
    ctx.park_run_factor = _PARK_RUN_FACTOR.get(game.home_team, 1.00)
    # Keep park_hr_factor mirroring park_run_factor in backtest. We have a
    # richer _PARK_HR_FACTOR map below, but in backtest the historical
    # totals/moneyline models react to HR/run divergence and that produced
    # a regression (see project_pro_edge_integration notes). Live code can
    # populate ctx.park_hr_factor from real batted-ball data instead.
    ctx.park_hr_factor = ctx.park_run_factor
    ctx.league_run_drift = league_run_drift
    ctx.day_game = _is_day_game(game)
    if doubleheader_set is not None and game.event_id in doubleheader_set:
        ctx.doubleheader = True

    if with_weather:
        # Import inside the branch so plain backtests with no weather
        # don't pay the import cost.
        from data.weather_history import get_historical_weather
        w = get_historical_weather(game.home_team, game.game_time_utc)
        if w is not None:
            ctx.temperature_f = w.temperature_f
            ctx.humidity_pct = w.humidity_pct
            ctx.wind_speed_mph = w.wind_speed_mph
            ctx.wind_direction = w.wind_relative_to_cf
            # Domed parks: weather_delta already returns 0 when
            # roof_status is "closed", so preserve that semantic.
            # Infer closed-roof from venue heuristically by park name
            # the same way the live path does; safer to leave
            # roof_status at "open" (the default) for outdoor parks.

    if ump_rpg_lookup is not None:
        rpg = ump_rpg_lookup.get(game.event_id)
        if rpg is not None:
            ctx.ump_runs_per_game = float(rpg)
    return ctx


# --- Grading ----------------------------------------------------------------

def _grade_moneyline(pick: str, game: HistoricalGame) -> Optional[bool]:
    """Return True if pick won, False if lost, None if NO BET / unparseable."""
    if not pick or pick.startswith("NO BET"):
        return None
    if pick.startswith("HOME"):
        return game.home_runs > game.away_runs
    if pick.startswith("AWAY"):
        return game.away_runs > game.home_runs
    return None


def _grade_run_line(pick: str, md: MarketData,
                    game: HistoricalGame) -> Optional[bool]:
    if not pick or pick.startswith("NO BET"):
        return None
    # Pick strings look like "HOME -1.5 +110" or "AWAY +1.5 -130"
    diff = game.home_runs - game.away_runs
    if pick.startswith("HOME"):
        # Home laying the line (-1.5) wins iff they win by 2+
        if md.home_is_rl_favorite:
            return diff >= 2
        # Home getting +1.5 wins iff they win OR lose by exactly 1
        return diff >= 0 or diff == -1
    if pick.startswith("AWAY"):
        if md.home_is_rl_favorite:
            # Away getting +1.5 wins iff they win OR lose by exactly 1
            return -diff >= 0 or -diff == -1
        # Away laying -1.5 wins iff they win by 2+
        return -diff >= 2
    return None


def _grade_totals(pick: str, md: MarketData,
                  game: HistoricalGame) -> Optional[bool]:
    if not pick or pick.startswith("NO BET"):
        return None
    total = game.home_runs + game.away_runs
    line = md.total_line
    if total == line:
        return None  # push — treated as void
    if pick.startswith("OVER"):
        return total > line
    if pick.startswith("UNDER"):
        return total < line
    return None


def _odds_for_pick(pick: str, market: str, md: MarketData) -> Optional[int]:
    """What odds were offered on this pick? Used for profit computation."""
    if not pick or pick.startswith("NO BET"):
        return None
    if market == "moneyline":
        return md.home_ml_odds if pick.startswith("HOME") else md.away_ml_odds
    if market == "run_line":
        return md.home_rl_odds if pick.startswith("HOME") else md.away_rl_odds
    if market == "totals":
        return md.over_odds if pick.startswith("OVER") else md.under_odds
    return None


@dataclass
class MarketPerformance:
    bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    units_won: float = 0.0
    units_wagered: float = 0.0   # flat: ==bets; kelly: variable per bet
    clv_sum: float = 0.0         # sum of CLV (closing line value) bps
    clv_count: int = 0
    edge_sum: float = 0.0        # sum of pred.edge over every counted bet
    edge_count: int = 0          # number of bets contributing to edge_sum

    @property
    def win_pct(self) -> float:
        played = self.wins + self.losses
        return (self.wins / played) if played else 0.0

    @property
    def roi_pct(self) -> float:
        # Prefer stake-weighted ROI when we have it; fall back to bet-count
        # ROI for legacy flat-stake runs that never populated units_wagered.
        denom = self.units_wagered if self.units_wagered > 0 else self.bets
        return (self.units_won / denom * 100.0) if denom else 0.0

    @property
    def avg_clv_bps(self) -> float:
        return (self.clv_sum / self.clv_count) if self.clv_count else 0.0

    @property
    def avg_edge_pct(self) -> float:
        """Mean model edge (model_prob - implied_prob) as a percent."""
        return (self.edge_sum / self.edge_count * 100.0) if self.edge_count else 0.0


@dataclass
class BacktestResults:
    season: int
    totals: MarketPerformance = field(default_factory=MarketPerformance)
    by_market: dict = field(default_factory=lambda: {
        "moneyline": MarketPerformance(),
        "run_line": MarketPerformance(),
        "totals": MarketPerformance(),
    })
    by_confidence: dict = field(default_factory=lambda: {
        "LOW": MarketPerformance(),
        "LEAN": MarketPerformance(),
        "MEDIUM": MarketPerformance(),
        "HIGH": MarketPerformance(),
    })
    equity_curve: list = field(default_factory=list)   # [{"date": "...", "equity": 1000}, ...]
    games_evaluated: int = 0
    games_missing_odds: int = 0


def run_backtest(
    games: Iterable[HistoricalGame],
    odds: dict[str, HistoricalOdds],
    baseline: BaselineStats,
    *,
    starting_bankroll: float = 1000.0,
    min_confidence: float = 0.0,
    with_weather: bool = False,
    with_ump: bool = False,
    ump_rpg_lookup: Optional[dict] = None,
    kelly_sizing: bool = False,
    kelly_fraction: float = 0.25,
    kelly_max_stake_pct: float = 0.03,
    unit_size_pct: float = 0.01,
) -> BacktestResults:
    """Run the backtest over the provided games.

    min_confidence : bets with confidence below this are ignored.
                     Default 0 (take every non-NO-BET prediction).
    with_weather   : if True, back-fill GameContext weather from the
                     Open-Meteo archive API (see data/weather_history).
                     Adds ~1 HTTP request per unique (venue, date) the
                     first time it's seen; subsequent runs read from the
                     on-disk cache. Needed for the totals-family
                     diagnostic to see weather as a real signal instead
                     of a silenced zero.
    """
    # The season being backtested is the games' season, not the baseline.
    # Infer from the first game's date; fall back to baseline.season.
    inferred_season = baseline.season
    if games:
        gd = games[0].game_date
        if hasattr(gd, "year"):
            inferred_season = gd.year
        elif isinstance(gd, str) and len(gd) >= 4:
            try:
                inferred_season = int(gd[:4])
            except ValueError:
                pass
    res = BacktestResults(season=inferred_season)
    # Compute per-season totals drift (baseline year stats shift vs current era)
    drift = _league_run_drift(inferred_season, baseline.season)
    if drift != 0.0:
        log.info("backtest season=%d baseline=%d league_run_drift=%+.2f runs/game",
                 inferred_season, baseline.season, drift)
    # Rolling in-season form adjustment (last-30-game runs vs league avg)
    season_rpg = LEAGUE_RPG.get(inferred_season, 4.50)
    rolling = _build_rolling_form(games, league_rpg=season_rpg)
    # Rolling win% (last 10 / last 20) — feeds situational_score form fields
    rolling_wp = _build_rolling_winpct(games)
    # Rest days since last game — feeds situational_score rest_edge
    rest_days = _build_rest_days(games)
    dh_set = _build_doubleheader_set(games)

    if with_weather:
        # Prewarm the (venue, date) cache in bulk BEFORE the main loop.
        # This serializes the network hits so we don't stall the prediction
        # loop one row at a time — and lets us flush the cache once at the
        # end rather than after each game.
        from data.weather_history import prewarm_range, save_cache as _save_wx
        by_date: dict[str, list[str]] = {}
        for g in games:
            if g.game_time_utc is None:
                continue
            ds = g.game_time_utc.strftime("%Y-%m-%d")
            by_date.setdefault(ds, []).append(g.home_team)
        # Dedupe (team, date) pairs so we don't double-fetch doubleheaders
        for d in by_date:
            by_date[d] = sorted(set(by_date[d]))
        if by_date:
            log.info("backtest weather: prewarming %d (venue,date) pairs",
                     sum(len(v) for v in by_date.values()))
            ok = prewarm_range(by_date)
            log.info("backtest weather: %d/%d fetches succeeded", ok,
                     sum(len(v) for v in by_date.values()))
            _save_wx()

    # Build / accept an umpire R/G lookup. If the caller supplied one
    # (typical for multi-season runners that want cross-season career
    # priors) we use it verbatim; otherwise build from current-season
    # games only — which gives zero career priors for April games but
    # still populates the family for the rest of the year.
    if with_ump and ump_rpg_lookup is None:
        from data.umpire_history import (
            prewarm_season as _prewarm_ump,
            save_cache as _save_ump,
            build_ump_rpg_lookup,
        )
        _prewarm_ump(inferred_season)
        _save_ump()
        ump_rpg_lookup = build_ump_rpg_lookup(games)
        n_pop = sum(1 for v in ump_rpg_lookup.values() if v is not None)
        log.info("backtest ump: %d/%d games with career R/G",
                 n_pop, len(ump_rpg_lookup))

    bankroll = starting_bankroll
    daily_equity: dict[str, float] = {}   # date -> running bankroll EOD

    for game in games:
        od = odds.get(game.event_id)
        if od is None:
            res.games_missing_odds += 1
            continue
        res.games_evaluated += 1

        home = baseline.team_stats(game.home_team, game.home_starter,
                                   is_home=True)
        away = baseline.team_stats(game.away_team, game.away_starter,
                                   is_home=False)
        # Apply rolling in-season adjustment to both teams
        h_rs, h_ra = rolling.get((game.home_team, game.event_id), (0.0, 0.0))
        a_rs, a_ra = rolling.get((game.away_team, game.event_id), (0.0, 0.0))
        _apply_rolling_adjust(home, h_rs, h_ra)
        _apply_rolling_adjust(away, a_rs, a_ra)
        # Situational form + rest populate: TESTED 2026-04-21 and REVERTED.
        # Train ML situational delta jumped 0% → +7.56% SIGNAL (real train edge),
        # but TEST delta was only +0.52% (below 1% noise floor). Pooled ROI
        # regressed -1.78% → -1.88% because the new picks shifted ML bullpen
        # disagree-ROI from +0.89% to +5.36% anti. Classic train/test instability
        # (see project_family_train_test_instability.md). Helpers kept for
        # future attempts gated on defense/OAA co-population.
        md = _build_market_data(od)
        ctx = _build_game_context(game, league_run_drift=drift,
                                  doubleheader_set=dh_set,
                                  with_weather=with_weather,
                                  ump_rpg_lookup=ump_rpg_lookup)

        preds = predict_all(home, away, ctx, md, include_soft=False)
        for market_name, pred in preds.items():
            if market_name not in res.by_market:
                continue
            if pred.confidence < min_confidence:
                continue
            pick = pred.pick

            # Grade
            if market_name == "moneyline":
                won = _grade_moneyline(pick, game)
            elif market_name == "run_line":
                won = _grade_run_line(pick, md, game)
            else:  # totals
                won = _grade_totals(pick, md, game)

            if won is None:
                if pick and not pick.startswith("NO BET"):
                    # push (totals landed exactly on the line, etc.)
                    perf = res.by_market[market_name]
                    perf.bets += 1
                    perf.pushes += 1
                    res.totals.bets += 1
                    res.totals.pushes += 1
                continue

            odds_int = _odds_for_pick(pick, market_name, md)
            if odds_int is None:
                continue
            profit_per_unit = _profit_per_unit(won, odds_int)

            pred_edge = getattr(pred, "edge", None)

            # Stake sizing: flat 1u by default; fractional-Kelly when enabled
            if kelly_sizing:
                from bet_selection.kelly import kelly_stake
                unit_dollars = max(1.0, starting_bankroll * unit_size_pct)
                stake_dollars = kelly_stake(
                    model_prob=pred.model_prob,
                    american_odds=odds_int,
                    bankroll=bankroll,
                    confidence=pred.confidence,
                    kelly_fraction=kelly_fraction,
                    max_stake_pct=kelly_max_stake_pct,
                )
                stake_units = stake_dollars / unit_dollars if unit_dollars > 0 else 0.0
                if stake_units <= 0:
                    # Kelly says don't bet (edge disappeared under de-vig) — skip
                    continue
            else:
                stake_units = 1.0
            profit = profit_per_unit * stake_units

            perf = res.by_market[market_name]
            perf.bets += 1
            perf.wins += 1 if won else 0
            perf.losses += 0 if won else 1
            perf.units_won += profit
            perf.units_wagered += stake_units
            if pred_edge is not None:
                perf.edge_sum += pred_edge
                perf.edge_count += 1

            res.totals.bets += 1
            res.totals.wins += 1 if won else 0
            res.totals.losses += 0 if won else 1
            res.totals.units_won += profit
            res.totals.units_wagered += stake_units
            if pred_edge is not None:
                res.totals.edge_sum += pred_edge
                res.totals.edge_count += 1

            b = res.by_confidence.get(pred.confidence_label)
            if b is not None:
                b.bets += 1
                b.wins += 1 if won else 0
                b.losses += 0 if won else 1
                b.units_won += profit
                b.units_wagered += stake_units
                if pred_edge is not None:
                    b.edge_sum += pred_edge
                    b.edge_count += 1

            if kelly_sizing:
                # Bankroll moves in dollars; equity curve is dollars.
                bankroll += profit * unit_dollars
            else:
                # Legacy: equity curve denominated in cumulative units.
                bankroll += profit
            daily_equity[game.game_date] = round(bankroll, 2)

    # Build sorted equity curve
    for d in sorted(daily_equity.keys()):
        res.equity_curve.append({"date": d, "equity": daily_equity[d]})

    return res
