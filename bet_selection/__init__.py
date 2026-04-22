"""
Bet-selection package.

`build_slip(picks, bankroll)` is the one-stop entry point used by the
web/desktop UI: it takes every gated pick for the day and returns a
ranked, Kelly-sized, exposure-capped slip.

Modules
-------
- kelly.py       : fractional Kelly sizing
- ranker.py      : rank by EV × confidence, dedupe correlated legs
- bankroll.py    : daily-cap / total-exposure guardrails
- slip.py        : orchestrates kelly + ranker + bankroll
- clv.py         : closing-line-value ledger + math helpers
"""
from .kelly import kelly_stake, kelly_fraction_for_pick, KELLY_FRACTION
from .ranker import rank_picks, dedupe_correlated, portfolio_correlation
from .bankroll import BankrollPolicy, apply_exposure_caps
from .slip import build_slip, BetSlipLeg, BetSlip
from .clv import (
    CLVStore, CLVRow, row_from_prediction,
    price_clv_cents, fair_prob_clv_bps,
)

__all__ = [
    "kelly_stake", "kelly_fraction_for_pick", "KELLY_FRACTION",
    "rank_picks", "dedupe_correlated", "portfolio_correlation",
    "BankrollPolicy", "apply_exposure_caps",
    "build_slip", "BetSlipLeg", "BetSlip",
    "CLVStore", "CLVRow", "row_from_prediction",
    "price_clv_cents", "fair_prob_clv_bps",
]
