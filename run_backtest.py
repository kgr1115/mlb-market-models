"""
CLI driver for the historical backtest.

Usage:
    python run_backtest.py                   # default: season=2021, baseline=2019
    python run_backtest.py 2022              # season=2022, baseline=2021
    python run_backtest.py 2022 2019         # season=2022, baseline=2019
    python run_backtest.py 2026 2025         # current season (livecache only)
    python run_backtest.py 2026 2025 --through 2026-04-21
                                             # only grade games up to that date
    python run_backtest.py 2022 2019 --weather   # include historical weather
    python run_backtest.py 2026 2025 --refresh-games
                                             # force MLB Stats API re-fetch
                                             # (bypass historical_games cache)
    python run_backtest.py 2025 2024 --no-livecache
                                             # community dataset only (revert
                                             # behavior before this change)

Writes JSON to web/backend/backtest_results.json (stable path the API loads).

`--weather` back-fills GameContext weather from the Open-Meteo archive API
(see data/weather_history.py). First run prewarms the cache — expect a
couple thousand HTTP requests for a full season; subsequent runs read
from data/cache/weather_history.json.

Season dispatch
---------------
  - 2018-2019: Sportsbook Reviews Online XLSX
  - 2021-2024: community dataset
  - 2025:      community (through 2025-08-16) + livecache (after)
  - 2026+:     live OddsCache replay
See ``run_multi_backtest.load_odds_for_season`` for the authoritative
dispatcher — this driver just delegates.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from backtest import (
    load_season_games, load_baseline,
    run_backtest, write_results_json,
)
from run_multi_backtest import load_odds_for_season

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("run_backtest")


def main(argv: list[str]) -> int:
    with_weather = False
    use_livecache = True
    refresh_games = False
    through_date: "str | None" = None
    positional: list[str] = []

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--weather":
            with_weather = True
            i += 1
            continue
        if arg == "--no-livecache":
            use_livecache = False
            i += 1
            continue
        if arg == "--refresh-games":
            refresh_games = True
            i += 1
            continue
        if arg == "--through":
            through_date = argv[i + 1]
            i += 2
            continue
        positional.append(arg)
        i += 1

    season = int(positional[0]) if len(positional) >= 1 else 2021
    # 2020 was COVID-shortened (60 games, wildly non-representative), so when
    # backtesting 2021 we jump to 2019 for the baseline.
    if len(positional) >= 2:
        baseline_season = int(positional[1])
    elif season == 2021:
        baseline_season = 2019
    else:
        baseline_season = season - 1

    log.info("=" * 60)
    log.info("Backtest: season=%d   baseline=%d   through=%s   livecache=%s",
             season, baseline_season, through_date, use_livecache)
    log.info("=" * 60)

    log.info("[1/4] Loading %d schedule + final scores ...", season)
    games = load_season_games(season, use_cache=not refresh_games)
    log.info("      loaded %d historical games", len(games))

    log.info("[2/4] Loading %d closing odds (auto-dispatch by season) ...",
             season)
    odds = load_odds_for_season(season, use_livecache=use_livecache)
    log.info("      loaded %d odds rows", len(odds))

    if through_date is not None:
        games = [g for g in games if g.game_date <= through_date]
        odds = {ev: o for ev, o in odds.items() if o.game_date <= through_date}
        log.info("      --through %s: %d games, %d odds rows",
                 through_date, len(games), len(odds))

    log.info("[3/4] Loading %d baseline team + pitcher stats ...",
             baseline_season)
    baseline = load_baseline(baseline_season)
    log.info("      baseline ready: %d teams (offense), %d pitchers",
             len(baseline.offense), len(baseline.pitchers))

    log.info("[4/4] Running predictions + grading bets ... (weather=%s)",
             with_weather)
    res = run_backtest(games, odds, baseline, with_weather=with_weather)

    out_path = Path(__file__).resolve().parent / "web" / "backend" / "backtest_results.json"
    write_results_json(res, str(out_path), starting_bankroll=1000.0,
                       baseline_season=baseline_season)
    log.info("Wrote results to %s", out_path)

    # Console summary
    t = res.totals
    print()
    print("=" * 60)
    print(f"BACKTEST SUMMARY  season={season}  baseline={baseline_season}")
    print("=" * 60)
    print(f"Games evaluated   : {res.games_evaluated}")
    print(f"Games missing odds: {res.games_missing_odds}")
    print(f"Total bets        : {t.bets}")
    print(f"Win %             : {t.win_pct*100:.1f}%")
    print(f"Units won         : {t.units_won:+.2f}")
    print(f"ROI               : {t.roi_pct:+.2f}%")
    print(f"Ending bankroll   : ${res.equity_curve[-1]['equity']:.2f}")
    print()
    print("By market:")
    for m, p in res.by_market.items():
        print(f"  {m:10s}  n={p.bets:4d}  win%={p.win_pct*100:5.1f}  "
              f"roi={p.roi_pct:+.2f}%")
    print()
    print("By confidence:")
    for lbl in ("LOW", "LEAN", "MEDIUM", "HIGH"):
        if lbl in res.by_confidence:
            p = res.by_confidence[lbl]
            print(f"  {lbl:6s}  n={p.bets:4d}  win%={p.win_pct*100:5.1f}  "
                  f"roi={p.roi_pct:+.2f}%")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
