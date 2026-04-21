"""
Multi-season backtest driver.

Runs the backtest for each (season, baseline_season) pair, pools the per-bet
rows into one big corpus, and writes a combined report + per-season reports.

Output files:
    web/backend/backtest_results.json          -> pooled (what API serves)
    web/backend/backtest_results_{YYYY}.json   -> per-season breakdowns
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from backtest import (
    load_season_games, load_sbr_season_odds, load_community_season_odds,
    load_baseline,
    run_backtest, write_results_json, BacktestResults,
)
from backtest.engine import MarketPerformance


def load_odds_for_season(season: int) -> dict:
    """Dispatch: SBR for 2018-2019, community dataset for 2021+.

    SBR's free XLSX archive stopped after 2021, so we use the ArnavSaraogi
    community dataset (scraped from the same SportsbookReview source but
    with per-book detail) for 2021-2025. Using community for 2021 too
    keeps methodology consistent across the post-COVID corpus.
    """
    if season <= 2019:
        return load_sbr_season_odds(season)
    return load_community_season_odds(season)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("run_multi_backtest")

# Seasons to backtest, paired with the baseline season to use.
# 2020 skipped - COVID-shortened (60 games, non-representative).
# 2022+ added via community dataset (SBR free archive ended after 2021).
# 2023 was the first pitch-clock + shift-ban season, so pre/post split is
# interesting for era-sensitivity analysis.
SEASON_PAIRS = [
    (2018, 2017),   # juiced-ball era  (SBR)
    (2019, 2018),   # juiced-ball era  (SBR)
    (2021, 2019),   # deadened ball    (community; 2-yr gap because of COVID)
    (2022, 2021),   # pre pitch-clock  (community)
    (2023, 2022),   # pitch clock + shift ban start (community)
    (2024, 2023),   # post pitch-clock (community)
    (2025, 2024),   # current season, partial through Aug 16 (community)
]

OUT_DIR = Path(__file__).resolve().parent / "web" / "backend"


def _merge_perf(dst: MarketPerformance, src: MarketPerformance) -> None:
    dst.bets += src.bets
    dst.wins += src.wins
    dst.losses += src.losses
    dst.pushes += src.pushes
    dst.units_won += src.units_won


def pool_results(per_season: list[BacktestResults]) -> BacktestResults:
    """Sum bet counts + units across seasons; stitch equity curves end-to-end."""
    if not per_season:
        raise ValueError("no results to pool")
    pooled = BacktestResults(season=per_season[0].season)
    for r in per_season:
        pooled.games_evaluated += r.games_evaluated
        pooled.games_missing_odds += r.games_missing_odds
        _merge_perf(pooled.totals, r.totals)
        for mkt in pooled.by_market:
            _merge_perf(pooled.by_market[mkt], r.by_market.get(mkt, MarketPerformance()))
        for lbl in pooled.by_confidence:
            _merge_perf(pooled.by_confidence[lbl],
                        r.by_confidence.get(lbl, MarketPerformance()))

    # Stitch equity curves: each season restarts at $1000, but we want a
    # continuous "if you ran the model over these seasons in sequence" view.
    running = 1000.0
    stitched: list[dict] = []
    for r in per_season:
        if not r.equity_curve:
            continue
        for pt in r.equity_curve:
            # delta from that season's $1000 start
            delta = pt["equity"] - 1000.0
            stitched.append({"date": pt["date"],
                             "equity": round(running + delta, 2)})
        running += (r.equity_curve[-1]["equity"] - 1000.0)
    pooled.equity_curve = stitched
    return pooled


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

        out_path = OUT_DIR / f"backtest_results_{season}.json"
        write_results_json(res, str(out_path),
                           starting_bankroll=1000.0,
                           baseline_season=baseline_season)
        log.info("Wrote %s (n=%d bets, ROI=%+.2f%%)",
                 out_path.name, res.totals.bets, res.totals.roi_pct)

    pooled = pool_results(per_season)
    pooled_path = OUT_DIR / "backtest_results.json"
    write_results_json(pooled, str(pooled_path),
                       starting_bankroll=1000.0, baseline_season=None)

    # Patch pooled JSON meta + note so the API communicates the real story.
    import json as _json
    with pooled_path.open("r") as f:
        payload = _json.load(f)
    positive = pooled.totals.roi_pct > 0
    payload["meta"] = {
        "seasons": [s for s, _ in SEASON_PAIRS],
        "baseline_seasons": {str(s): b for s, b in SEASON_PAIRS},
        "games_evaluated": pooled.games_evaluated,
        "games_missing_odds": pooled.games_missing_odds,
        "model": "prior-year-baseline + rolling in-season form",
        "features": [
            "prior-year team/pitcher stats (FanGraphs)",
            "league-drift correction (LEAGUE_RPG by season)",
            "rolling 30-game in-season form adjustment",
            "confidence gating: MEDIUM+ on ML/RL, score>=99 on totals",
        ],
        "source": "real",
        "note_detail": (
            "Pooled multi-season backtest (2018, 2019, 2021-2025; "
            "2020 COVID skipped). SBR closing lines 2018-19, "
            "community-scraped DraftKings closing lines 2021-25. "
            f"Headline ROI {pooled.totals.roi_pct:+.2f}% on "
            f"{pooled.totals.bets} bets over {pooled.games_evaluated} games."
        ),
    }
    season_list = "+".join(str(s) for s, _ in SEASON_PAIRS)
    if positive:
        payload["note"] = (
            f"Real multi-season backtest ({season_list}). "
            f"{pooled.totals.bets} bets, {pooled.totals.roi_pct:+.2f}% ROI "
            f"(${pooled.equity_curve[-1]['equity']:.0f} from $1000). "
            "Model shows edge vs closing lines; treat results as "
            "backtest only, not a guarantee of live performance."
        )
    else:
        payload["note"] = (
            f"Real multi-season backtest ({season_list}). "
            f"{pooled.totals.bets} bets, {pooled.totals.roi_pct:+.2f}% ROI. "
            "Model does not currently have edge vs. closing lines - "
            "do not use live until recalibration."
        )
    with pooled_path.open("w") as f:
        _json.dump(payload, f, indent=2, default=str)
    log.info("Wrote pooled %s", pooled_path)

    # --- Console summary ---------------------------------------------------
    print()
    print("=" * 72)
    print("MULTI-SEASON BACKTEST")
    print("=" * 72)
    print(f"{'season':>7}  {'n bets':>7}  {'win%':>5}  "
          f"{'units':>8}  {'ROI':>7}  {'end $':>9}")
    print("  " + "-" * 50)
    for (season, baseline_season), r in zip(SEASON_PAIRS, per_season):
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
    for lbl in ("LOW", "LEAN", "MEDIUM", "HIGH"):
        p = pooled.by_confidence[lbl]
        print(f"  {lbl:6s}  n={p.bets:5d}  win%={p.win_pct*100:5.1f}  "
              f"roi={p.roi_pct:+6.2f}%")

    print()
    print("Per-season confidence breakdown:")
    print(f"  {'season':>7}  {'LOW roi':>9}  {'LEAN roi':>9}  "
          f"{'MED roi':>9}  {'HIGH roi':>9}")
    for (season, _), r in zip(SEASON_PAIRS, per_season):
        row = [f"{season:>7}"]
        for lbl in ("LOW", "LEAN", "MEDIUM", "HIGH"):
            p = r.by_confidence[lbl]
            row.append(f"{p.roi_pct:>+8.2f}%")
        print("  " + "  ".join(row))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
