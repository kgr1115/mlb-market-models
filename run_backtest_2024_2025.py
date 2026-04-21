"""
One-off driver: backtest 2024 and 2025 only (pooled + per-season).

Reuses helpers from run_multi_backtest so the methodology / output format
is identical, just with a restricted season list.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from backtest import (
    load_season_games, load_baseline,
    run_backtest, write_results_json, BacktestResults,
)
from run_multi_backtest import load_odds_for_season, pool_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("run_backtest_2024_2025")

SEASON_PAIRS = [
    (2024, 2023),
    (2025, 2024),
]

OUT_DIR = Path(__file__).resolve().parent / "web" / "backend"


def main() -> int:
    per_season: list[BacktestResults] = []

    for season, baseline_season in SEASON_PAIRS:
        log.info("=" * 60)
        log.info("Backtest: season=%d   baseline=%d", season, baseline_season)
        log.info("=" * 60)
        games = load_season_games(season)
        odds = load_odds_for_season(season)
        baseline = load_baseline(baseline_season)
        res = run_backtest(games, odds, baseline)
        per_season.append(res)

        out_path = OUT_DIR / f"backtest_2024_2025_{season}.json"
        write_results_json(res, str(out_path),
                           starting_bankroll=1000.0,
                           baseline_season=baseline_season)
        log.info("Wrote %s (n=%d bets, ROI=%+.2f%%)",
                 out_path.name, res.totals.bets, res.totals.roi_pct)

    pooled = pool_results(per_season)
    pooled_path = OUT_DIR / "backtest_2024_2025_pooled.json"
    write_results_json(pooled, str(pooled_path),
                       starting_bankroll=1000.0, baseline_season=None)
    log.info("Wrote pooled %s", pooled_path)

    # --- Console summary ---------------------------------------------------
    print()
    print("=" * 72)
    print("BACKTEST  (2024 + 2025)")
    print("=" * 72)
    print(f"{'season':>7}  {'n bets':>7}  {'win%':>5}  "
          f"{'units':>8}  {'ROI':>7}  {'end $':>9}")
    print("  " + "-" * 50)
    for (season, _), r in zip(SEASON_PAIRS, per_season):
        t = r.totals
        end_bk = r.equity_curve[-1]["equity"] if r.equity_curve else 1000.0
        print(f"{season:>7}  {t.bets:>7d}  {t.win_pct*100:>5.1f}  "
              f"{t.units_won:>+8.2f}  {t.roi_pct:>+7.2f}%  {end_bk:>9.2f}")
    t = pooled.totals
    end_bk = pooled.equity_curve[-1]["equity"] if pooled.equity_curve else 1000.0
    print("  " + "-" * 50)
    print(f"{'POOLED':>7}  {t.bets:>7d}  {t.win_pct*100:>5.1f}  "
          f"{t.units_won:>+8.2f}  {t.roi_pct:>+7.2f}%  {end_bk:>9.2f}")

    print()
    print("Pooled by market:")
    for mkt in ("moneyline", "run_line", "totals"):
        p = pooled.by_market[mkt]
        print(f"  {mkt:10s}  n={p.bets:5d}  win%={p.win_pct*100:5.1f}  "
              f"roi={p.roi_pct:+6.2f}%")

    print()
    print("Pooled by confidence:")
    for lbl in ("HIGH", "MEDIUM", "LOW"):
        p = pooled.by_confidence.get(lbl)
        if not p:
            continue
        print(f"  {lbl:6s}  n={p.bets:5d}  win%={p.win_pct*100:5.1f}  "
              f"roi={p.roi_pct:+6.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
