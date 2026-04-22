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
    load_livecache_season_odds, load_baseline,
    run_backtest, write_results_json, BacktestResults,
)
from backtest.engine import MarketPerformance


# The community dataset's last scrape covers 2025-08-16. Any game with
# event_id >= this date needs to come from our live OddsCache instead.
# If the community dataset gets re-scraped later we just shift this date
# forward (or drop the livecache merge for those seasons).
_COMMUNITY_CUTOFF_DATE = "2025-08-16"


def load_odds_for_season(season: int, use_livecache: bool = True) -> dict:
    """Dispatch loader by season, merging live-cache data past the
    community dataset's 2025-08-16 cutoff when ``use_livecache`` is set.

      - 2018-2019: SBR XLSX
      - 2021-2024: community dataset only
      - 2025:      community (through 2025-08-16) + livecache (after)
      - 2026+:     livecache only

    SBR's free XLSX archive stopped after 2021. The community dataset
    (ArnavSaraogi/mlb-odds-scraper) extends through 2025-08-16. Beyond
    that, the live OddsCache owns the corpus.
    """
    if season <= 2019:
        return load_sbr_season_odds(season)

    if season < 2025:
        return load_community_season_odds(season)

    if season == 2025:
        odds = load_community_season_odds(season)
        if use_livecache:
            # Livecache fills in 2025-08-17 onward. Community data wins
            # on overlap, so we only pull livecache for dates strictly
            # after the cutoff.
            from datetime import datetime, timedelta
            cutoff = datetime.fromisoformat(_COMMUNITY_CUTOFF_DATE).date()
            since = (cutoff + timedelta(days=1)).isoformat()
            tail = load_livecache_season_odds(season, since_date=since)
            # Prefer existing community rows on event_id collision (shouldn't
            # happen given the date cut, but be defensive).
            for ev, row in tail.items():
                odds.setdefault(ev, row)
        return odds

    # 2026 and beyond — livecache only.
    if use_livecache:
        return load_livecache_season_odds(season)
    return {}


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
    (2025, 2024),   # full season (community through 8/16, livecache after)
    (2026, 2025),   # current season (livecache only)
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
    # ----- CLI flags ---------------------------------------------------------
    # --through YYYY-MM-DD    Only grade events up to this date (for
    #                         incremental eval on a rolling basis).
    # --seasons 2025,2026     Restrict to these seasons (comma-separated).
    # --no-livecache          Skip livecache data (revert to community-only).
    # --refresh-games         Force-refetch MLB Stats API schedules
    #                         (bypasses the historical_games SQLite cache).
    # --weather               Back-fill GameContext weather from Open-Meteo
    #                         archive (populates the dormant totals weather
    #                         family; requires network on first run, then
    #                         reads from data/cache/weather_history.json).
    # --ump                   Back-fill plate-umpire career R/G from the
    #                         MLB Stats API + self-computed running mean
    #                         (populates the dormant totals umpire family;
    #                         cache: data/cache/umpire_history.json).
    through_date: "str | None" = None
    season_filter: "set[int] | None" = None
    use_livecache = True
    refresh_games = False
    with_weather = False
    with_ump = False

    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--through":
            through_date = argv[i + 1]
            i += 2
            continue
        if arg == "--seasons":
            season_filter = {int(s) for s in argv[i + 1].split(",")}
            i += 2
            continue
        if arg == "--no-livecache":
            use_livecache = False
            i += 1
            continue
        if arg == "--refresh-games":
            refresh_games = True
            i += 1
            continue
        if arg == "--weather":
            with_weather = True
            i += 1
            continue
        if arg == "--ump":
            with_ump = True
            i += 1
            continue
        log.warning("Unknown arg: %s (ignoring)", arg)
        i += 1

    season_pairs = SEASON_PAIRS
    if season_filter is not None:
        season_pairs = [p for p in SEASON_PAIRS if p[0] in season_filter]
        if not season_pairs:
            log.error("No matching seasons in %s", season_filter)
            return 1

    per_season: list[BacktestResults] = []

    # Build cross-season ump career-R/G lookup ONCE so early-season games
    # have career priors from prior seasons (engine would otherwise only
    # see current-season history and give April umps zero prior). The
    # prior-year 2017 seeds the accumulator for 2018-opener games.
    ump_rpg_lookup = None
    if with_ump:
        from data.umpire_history import (
            prewarm_season as _prewarm_ump,
            save_cache as _save_ump,
            build_ump_rpg_lookup,
        )
        prior_year = min(s for s, _ in season_pairs) - 1
        all_games: list = []
        for s in [prior_year] + sorted({s for s, _ in season_pairs}):
            _prewarm_ump(s)
            all_games.extend(load_season_games(s, use_cache=not refresh_games))
        _save_ump()
        ump_rpg_lookup = build_ump_rpg_lookup(all_games)
        n_pop = sum(1 for v in ump_rpg_lookup.values() if v is not None)
        log.info("ump: cross-season career-R/G lookup: %d/%d games populated",
                 n_pop, len(ump_rpg_lookup))

    for season, baseline_season in season_pairs:
        log.info("=" * 60)
        log.info("Backtest: season=%d   baseline=%d", season, baseline_season)
        log.info("=" * 60)
        games = load_season_games(season, use_cache=not refresh_games)
        odds = load_odds_for_season(season, use_livecache=use_livecache)

        # Apply --through filter uniformly across SBR / community / livecache.
        if through_date is not None:
            cut = through_date
            games = [g for g in games if g.game_date <= cut]
            odds = {ev: o for ev, o in odds.items() if o.game_date <= cut}
            log.info("--through %s applied: %d games, %d odds rows remain",
                     cut, len(games), len(odds))

        baseline = load_baseline(baseline_season)
        res = run_backtest(games, odds, baseline,
                           with_weather=with_weather,
                           with_ump=with_ump,
                           ump_rpg_lookup=ump_rpg_lookup)
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
        "seasons": [s for s, _ in season_pairs],
        "baseline_seasons": {str(s): b for s, b in season_pairs},
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
        "through_date": through_date,
        "livecache": use_livecache,
        "note_detail": (
            "Pooled multi-season backtest. "
            "SBR closing lines 2018-19, community-scraped DK lines "
            "2021 through 2025-08-16, live OddsCache replay after. "
            f"Headline ROI {pooled.totals.roi_pct:+.2f}% on "
            f"{pooled.totals.bets} bets over {pooled.games_evaluated} games."
        ),
    }
    season_list = "+".join(str(s) for s, _ in season_pairs)
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
    for (season, baseline_season), r in zip(season_pairs, per_season):
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
    for (season, _), r in zip(season_pairs, per_season):
        row = [f"{season:>7}"]
        for lbl in ("LOW", "LEAN", "MEDIUM", "HIGH"):
            p = r.by_confidence[lbl]
            row.append(f"{p.roi_pct:>+8.2f}%")
        print("  " + "  ".join(row))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
