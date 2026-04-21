"""
Ungated prediction collector.

Runs the backtest across every season in SEASON_PAIRS with confidence
gates BYPASSED on all three predictor modules and writes one row per
(game, market) to backtest/ungated_predictions.csv. Includes per-family
score contributions (fam_*) aligned to the picked side for downstream
feature-signal analysis.

Usage:
    python3 collect_predictions.py
"""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

# PATCH: HistoricalOdds .pyc cache is stale - patch before imports that depend on it
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from backtest.historical_odds import HistoricalOdds
    _ho_orig_init = HistoricalOdds.__init__
    def _ho_patched_init(self, event_id, game_date, away_team, home_team,
                         away_ml_close=None, home_ml_close=None,
                         away_ml_open=None, home_ml_open=None,
                         away_rl_line=None, away_rl_price=None,
                         home_rl_line=None, home_rl_price=None,
                         total_close=None, total_over_close=None, total_under_close=None,
                         total_open=None, total_over_open=None, total_under_open=None):
        _ho_orig_init(self, event_id, game_date, away_team, home_team,
                      away_ml_close, home_ml_close, away_ml_open, home_ml_open,
                      away_rl_line, away_rl_price, home_rl_line, home_rl_price,
                      total_close, total_over_close, total_under_close)
        object.__setattr__(self, 'total_open', total_open)
        object.__setattr__(self, 'total_over_open', total_over_open)
        object.__setattr__(self, 'total_under_open', total_under_open)
    HistoricalOdds.__init__ = _ho_patched_init
except Exception as e:
    logging.warning(f"Failed to patch HistoricalOdds: {e}")

# Flip gates OFF before predictors get touched by predict_all
import predictors.moneyline as _ml
import predictors.run_line as _rl
import predictors.totals as _tt
_ml._BYPASS_GATES = True
_rl._BYPASS_GATES = True
_tt._BYPASS_GATES = True

from backtest import (
    load_season_games, load_sbr_season_odds, load_community_season_odds,
    load_baseline,
)
from backtest.engine import (
    _build_market_data, _build_game_context,
    _grade_moneyline, _grade_run_line, _grade_totals,
    _odds_for_pick, _profit_per_unit,
    _build_rolling_form, _apply_rolling_adjust,
    _build_doubleheader_set,
    _league_run_drift, LEAGUE_RPG,
)
from predictors import predict_all


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("collect_predictions")


SEASON_PAIRS = [
    (2018, 2017),
    (2019, 2018),
    (2021, 2019),
    (2022, 2021),
    (2023, 2022),
    (2024, 2023),
    (2025, 2024),
]

OUT_PATH = Path(__file__).resolve().parent / "backtest" / "ungated_predictions.csv"


def load_odds_for_season(season: int) -> dict:
    if season <= 2019:
        return load_sbr_season_odds(season)
    return load_community_season_odds(season)


_ML_FAMILIES = ["pitcher", "bullpen", "offense", "defense",
                "baserun", "situational", "environment"]
_RL_FAMILIES = ["pitcher", "bullpen", "offense", "context", "market"]
_TT_FAMILIES = ["pitcher", "bullpen", "offense", "park",
                "weather", "umpire", "pace", "market"]
_ALL_FAMILIES = sorted(set(_ML_FAMILIES) | set(_RL_FAMILIES) | set(_TT_FAMILIES))

FIELDS = [
    "season", "baseline_season", "date", "event_id", "home", "away",
    "market", "pick", "side", "odds", "model_prob", "implied_prob",
    "edge", "confidence_score", "confidence_label",
    "profit_per_unit", "won", "pushed",
] + [f"fam_{f}" for f in _ALL_FAMILIES]


def _extract_family_diffs(market: str, detail: dict, side: str) -> dict:
    out = {f"fam_{f}": "" for f in _ALL_FAMILIES}
    if market == "moneyline":
        diff = detail.get("diff_by_family") or {}
        sign = +1.0 if side == "HOME" else (-1.0 if side == "AWAY" else 0.0)
        for fam, v in diff.items():
            out[f"fam_{fam}"] = f"{sign * float(v):.6f}"
    elif market == "run_line":
        h = detail.get("home_family_scores") or {}
        a = detail.get("away_family_scores") or {}
        sign = +1.0 if side == "HOME" else (-1.0 if side == "AWAY" else 0.0)
        for fam in set(h) | set(a):
            d = float(h.get(fam, 0.0)) - float(a.get(fam, 0.0))
            out[f"fam_{fam}"] = f"{sign * d:.6f}"
    elif market == "totals":
        diff = detail.get("diff_by_family") or {}
        sign = +1.0 if side == "OVER" else (-1.0 if side == "UNDER" else 0.0)
        for fam, v in diff.items():
            out[f"fam_{fam}"] = f"{sign * float(v):.6f}"
    return out


def _pick_side(market: str, pick: str) -> str:
    if not pick or pick.startswith("NO BET"):
        return "NO_BET"
    return pick.split()[0]


def _collect_season(season, baseline_season, writer):
    games = load_season_games(season)
    odds = load_odds_for_season(season)
    baseline = load_baseline(baseline_season)

    drift = _league_run_drift(season, baseline_season)
    season_rpg = LEAGUE_RPG.get(season, 4.50)
    rolling = _build_rolling_form(games, league_rpg=season_rpg)
    dh_set = _build_doubleheader_set(games)

    n_games = n_missing = n_rows = 0
    for game in games:
        od = odds.get(game.event_id)
        if od is None:
            n_missing += 1
            continue
        n_games += 1

        home = baseline.team_stats(game.home_team, game.home_starter, is_home=True)
        away = baseline.team_stats(game.away_team, game.away_starter, is_home=False)
        h_rs, h_ra = rolling.get((game.home_team, game.event_id), (0.0, 0.0))
        a_rs, a_ra = rolling.get((game.away_team, game.event_id), (0.0, 0.0))
        _apply_rolling_adjust(home, h_rs, h_ra)
        _apply_rolling_adjust(away, a_rs, a_ra)
        md = _build_market_data(od)
        ctx = _build_game_context(game, league_run_drift=drift,
                                  doubleheader_set=dh_set)

        preds = predict_all(home, away, ctx, md)
        for market_name, pred in preds.items():
            pick = pred.pick or "NO BET"
            side = _pick_side(market_name, pick)

            if pick.startswith("NO BET"):
                won_flag = None
                pushed = False
                profit = 0.0
                odds_int = None
            else:
                if market_name == "moneyline":
                    won_flag = _grade_moneyline(pick, game)
                elif market_name == "run_line":
                    won_flag = _grade_run_line(pick, md, game)
                else:
                    won_flag = _grade_totals(pick, md, game)
                pushed = won_flag is None
                odds_int = _odds_for_pick(pick, market_name, md)
                if pushed or odds_int is None:
                    profit = 0.0
                else:
                    profit = _profit_per_unit(won_flag, odds_int)

            row = {
                "season": season,
                "baseline_season": baseline_season,
                "date": game.game_date,
                "event_id": game.event_id,
                "home": game.home_team,
                "away": game.away_team,
                "market": market_name,
                "pick": pick,
                "side": side,
                "odds": odds_int if odds_int is not None else "",
                "model_prob": f"{pred.model_prob:.6f}",
                "implied_prob": f"{pred.implied_prob:.6f}",
                "edge": f"{pred.edge:.6f}",
                "confidence_score": f"{pred.confidence:.3f}",
                "confidence_label": pred.confidence_label,
                "profit_per_unit": f"{profit:.4f}",
                "won": "" if won_flag is None else int(bool(won_flag)),
                "pushed": int(bool(pushed)),
            }
            row.update(_extract_family_diffs(market_name, pred.detail or {}, side))
            writer.writerow(row)
            n_rows += 1

    log.info("season=%d  games=%d  missing_odds=%d  rows=%d",
             season, n_games, n_missing, n_rows)
    return n_rows


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for season, base in SEASON_PAIRS:
            log.info("=" * 60)
            log.info("Collecting ungated predictions: season=%d baseline=%d",
                     season, base)
            log.info("=" * 60)
            total_rows += _collect_season(season, base, writer)
    log.info("Wrote %s (%d rows)", OUT_PATH, total_rows)
    print(f"\nUngated prediction corpus written: {OUT_PATH}")
    print(f"Rows: {total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
