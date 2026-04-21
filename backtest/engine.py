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
    """Translate an HistoricalOdds row into a MarketData the predictor consumes."""
    md = MarketData()
    if od.away_ml_close is not None and od.home_ml_close is not None:
        md.away_ml_odds = int(od.away_ml_close)
        md.home_ml_odds = int(od.home_ml_close)
    if od.away_ml_open is not None:
        md.opener_home_ml_odds = int(od.home_ml_open) if od.home_ml_open else None
    # Run line: SBR publishes a per-team line; we normalize to -1.5/+1.5 pricing.
    if od.home_rl_line is not None and od.home_rl_price is not None:
        md.home_is_rl_favorite = od.home_rl_line < 0
        md.home_rl_odds = int(od.home_rl_price)
        md.away_rl_odds = int(od.away_rl_price) if od.away_rl_price else -120
    if od.total_close is not None:
        md.total_line = float(od.total_close)
        if od.total_over_close:
            md.over_odds = int(od.total_over_close)
        if od.total_under_close:
            md.under_odds = int(od.total_under_close)
    # Opening total — lights up totals market family (previously all zeros).
    # Missing for SBR 2018-19; populated by community loader 2021+.
    if getattr(od, "total_open", None) is not None:
        md.opener_total = float(od.total_open)
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
                        with_weather: bool = False) -> GameContext:
    """GameContext from known home-team park + derived day/doubleheader flags.

    If ``with_weather`` is True, back-fills temperature_f, humidity_pct,
    wind_speed_mph and wind_direction from the Open-Meteo archive API
    (via data.weather_history). This is opt-in because historical weather
    fetches hit the network and the caller should prewarm the cache.
    """
    ctx = GameContext()
    ctx.park_run_factor = _PARK_RUN_FACTOR.get(game.home_team, 1.00)
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

    @property
    def win_pct(self) -> float:
        played = self.wins + self.losses
        return (self.wins / played) if played else 0.0

    @property
    def roi_pct(self) -> float:
        return (self.units_won / self.bets * 100.0) if self.bets else 0.0


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
        md = _build_market_data(od)
        ctx = _build_game_context(game, league_run_drift=drift,
                                  doubleheader_set=dh_set,
                                  with_weather=with_weather)

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
            profit = _profit_per_unit(won, odds_int)

            perf = res.by_market[market_name]
            perf.bets += 1
            perf.wins += 1 if won else 0
            perf.losses += 0 if won else 1
            perf.units_won += profit

            res.totals.bets += 1
            res.totals.wins += 1 if won else 0
            res.totals.losses += 0 if won else 1
            res.totals.units_won += profit

            b = res.by_confidence.get(pred.confidence_label)
            if b is not None:
                b.bets += 1
                b.wins += 1 if won else 0
                b.losses += 0 if won else 1
                b.units_won += profit

            bankroll += profit
            daily_equity[game.game_date] = round(bankroll, 2)

    # Build sorted equity curve
    for d in sorted(daily_equity.keys()):
        res.equity_curve.append({"date": d, "equity": daily_equity[d]})

    return res
