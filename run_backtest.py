"""
CLI driver for the historical backtest.

Usage:
    python run_backtest.py                   # default: season=2021, baseline=2019
    python run_backtest.py 2022              # season=2022, baseline=2021
    python run_backtest.py 2022 2019         # season=2022, baseline=2019
    python run_backtest.py 2022 2019 --weather   # include historical weather

Writes JSON to web/backend/backtest_results.json (stable path the API loads).

`--weather` back-fills GameContext weather from the Open-Meteo archive API
(see data/weather_history.py). First run prewarms the cache — expect a
couple thousand HTTP requests for a full season; subsequent runs read
from data/cache/weather_history.json.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from backtest import (
    load_season_games, load_sbr_season_odds, load_baseline,
    run_backtest, write_results_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("run_backtest")


def main(argv: list[str]) -> int:
    with_weather = False
    positional: list[str] = []
    for arg in argv[1:]:
        if arg == "--weather":
            with_weather = True
        else:
            positional.append(arg)

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
    log.info("Backtest: season=%d   baseline=%d", season, baseline_season)
    log.info("=" * 60)

    log.info("[1/4] Loading %d schedule + final scores ...", season)
    games = load_season_games(season)
    log.info("      loaded %d historical games", len(games))

    log.info("[2/4] Loading %d closing odds from Sportsbook Reviews Online ...",
             season)
    odds = load_sbr_season_odds(season)
    log.info("      loaded %d odds rows", len(odds))

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
