"""
Convert BacktestResults into the JSON shape /api/backtest/summary expects.
"""
from __future__ import annotations

import json
import os
from typing import Any

from .engine import BacktestResults


def to_api_shape(res: BacktestResults, starting_bankroll: float = 1000.0
                 ) -> dict[str, Any]:
    ending = (res.equity_curve[-1]["equity"]
              if res.equity_curve else starting_bankroll)

    period_start = res.equity_curve[0]["date"] if res.equity_curve else None
    period_end = res.equity_curve[-1]["date"] if res.equity_curve else None

    # Days in period
    days = 0
    if period_start and period_end:
        from datetime import date as _d
        try:
            d1 = _d.fromisoformat(period_start)
            d2 = _d.fromisoformat(period_end)
            days = (d2 - d1).days
        except ValueError:
            days = len(res.equity_curve)

    return {
        "period": {
            "start": period_start, "end": period_end, "days": days,
        },
        "totals": {
            "bets": res.totals.bets,
            "wins": res.totals.wins,
            "losses": res.totals.losses,
            "pushes": res.totals.pushes,
            "win_pct": round(res.totals.win_pct, 3),
            "units_won": round(res.totals.units_won, 2),
            "roi_pct": round(res.totals.roi_pct, 2),
            "starting_bankroll": starting_bankroll,
            "ending_bankroll": ending,
        },
        "by_market": {
            m: {
                "bets": p.bets, "wins": p.wins, "losses": p.losses,
                "pushes": p.pushes,
                "win_pct": round(p.win_pct, 3),
                "units_won": round(p.units_won, 2),
                "roi_pct": round(p.roi_pct, 2),
            }
            for m, p in res.by_market.items()
        },
        "by_confidence": {
            label: {
                "bets": p.bets, "wins": p.wins, "losses": p.losses,
                "win_pct": round(p.win_pct, 3),
                "units_won": round(p.units_won, 2),
                "roi_pct": round(p.roi_pct, 2),
            }
            for label, p in res.by_confidence.items()
        },
        "equity_curve": res.equity_curve,
        "meta": {
            "season": res.season,
            "games_evaluated": res.games_evaluated,
            "games_missing_odds": res.games_missing_odds,
            "model": "prior-year-baseline",
            "baseline_season": res.season,   # (set externally if different)
            "source": "real",
        },
        "note": None,
    }


def write_results_json(res: BacktestResults, path: str,
                       starting_bankroll: float = 1000.0,
                       baseline_season: int = None) -> None:
    payload = to_api_shape(res, starting_bankroll)
    if baseline_season is not None:
        payload["meta"]["baseline_season"] = baseline_season
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
