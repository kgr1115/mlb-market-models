"""
Quick parse-and-smoke check for the 2026-04-21 pro-edge integration.

Run: python verify_pro_edges.py

Checks:
  1. All modified modules import cleanly.
  2. MarketData has the new fair_prob_* / opener_* / rlm_score_* fields.
  3. fair_prob_for_side() returns a prob in (0,1) for each side.
  4. rlm_intensity() produces expected signs.
  5. kelly_stake() is wired and returns a non-negative dollar amount.
  6. CLVStore can record and finalize a fake bet, and summary() runs.
  7. A predict_all() call on synthetic inputs succeeds end-to-end.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path


def _ok(label: str) -> None:
    print(f"[OK]   {label}")


def _fail(label: str, err: Exception) -> None:
    print(f"[FAIL] {label}: {err!r}")
    traceback.print_exc()


def main() -> int:
    failures = 0

    # --- 1. imports
    try:
        from predictors import TeamStats, GameContext, MarketData, predict_all
        from predictors.shared import (
            fair_prob_consensus, fair_prob_for_side, rlm_intensity,
        )
        from bet_selection import kelly_stake, CLVStore, CLVRow
        from backtest.engine import _build_market_data
        from data.odds_client import build_market_data
        _ok("imports")
    except Exception as e:
        _fail("imports", e)
        return 1

    # --- 2. MarketData fields
    try:
        md = MarketData()
        for attr in ("fair_prob_home_ml", "fair_prob_over",
                      "opener_home_rl_odds", "rlm_score_home",
                      "rlm_score_over"):
            assert hasattr(md, attr), attr
        _ok("MarketData has new fields")
    except Exception as e:
        _fail("MarketData fields", e); failures += 1

    # --- 3. fair_prob_for_side fallback
    try:
        md = MarketData()
        md.home_ml_odds = -150
        md.away_ml_odds = +130
        p = fair_prob_for_side(md, "home_ml")
        assert 0.0 < p < 1.0, p
        # Now with consensus set explicitly
        md.fair_prob_home_ml = 0.6
        md.fair_prob_away_ml = 0.4
        assert fair_prob_for_side(md, "home_ml") == 0.6
        _ok("fair_prob_for_side")
    except Exception as e:
        _fail("fair_prob_for_side", e); failures += 1

    # --- 4. rlm_intensity
    try:
        # Fav shortened from -110 to -140, public on dog: classic RLM
        v = rlm_intensity(opener_fav_odds=-110, current_fav_odds=-140,
                          public_ticket_pct_fav=0.35)
        assert v > 0, v
        # Movement but aligned with public
        v2 = rlm_intensity(opener_fav_odds=-110, current_fav_odds=-140,
                           public_ticket_pct_fav=0.70)
        assert v2 < v, (v, v2)
        _ok("rlm_intensity")
    except Exception as e:
        _fail("rlm_intensity", e); failures += 1

    # --- 5. kelly_stake
    try:
        s = kelly_stake(model_prob=0.55, american_odds=-110,
                        bankroll=1000.0, confidence=80.0)
        assert s >= 0, s
        assert s <= 1000.0 * 0.03 + 0.01, s  # capped at 3%
        _ok("kelly_stake")
    except Exception as e:
        _fail("kelly_stake", e); failures += 1

    # --- 6. CLVStore
    try:
        with tempfile.TemporaryDirectory() as d:
            store = CLVStore(Path(d) / "clv.sqlite")
            row = CLVRow(event_id="E1", market="moneyline", side="home",
                          bet_odds=-110, paired_odds=-110,
                          model_prob=0.55, confidence=80.0)
            bet_id = store.record_bet(row)
            store.finalize(bet_id, close_odds=-130, close_paired=+110, won=1)
            summary = store.summary()
            assert summary["n"] == 1, summary
            assert summary["avg_bps"] is not None, summary
        _ok("CLVStore round-trip")
    except Exception as e:
        _fail("CLVStore", e); failures += 1

    # --- 7. predict_all end-to-end
    try:
        home = TeamStats(name="H", is_home=True)
        away = TeamStats(name="A", is_home=False)
        ctx = GameContext(park_run_factor=1.02, park_hr_factor=1.08)
        market = MarketData()
        market.home_ml_odds = -140
        market.away_ml_odds = +120
        market.home_rl_odds = +110
        market.away_rl_odds = -130
        market.home_is_rl_favorite = True
        market.total_line = 8.5
        market.over_odds = -105
        market.under_odds = -115
        market.opener_home_ml_odds = -110
        market.rlm_score_home = 0.4
        preds = predict_all(home, away, ctx, market, include_soft=False)
        for m in ("moneyline", "run_line", "totals"):
            assert m in preds, m
            p = preds[m]
            assert hasattr(p, "pick"), p
            assert hasattr(p, "confidence"), p
        _ok("predict_all end-to-end")
    except Exception as e:
        _fail("predict_all", e); failures += 1

    print()
    if failures:
        print(f"FAILED: {failures} check(s)")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
