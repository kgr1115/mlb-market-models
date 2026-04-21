"""
Fractional-Kelly stake sizing.

Kelly formula for a binary outcome at American odds:

    f* = (b * p - (1 - p)) / b

where
    p = model probability the bet wins
    b = decimal odds - 1 (the net payout per $1 risked on a win)

We cap real stakes to a FRACTION of full Kelly (default 0.25) because
(a) the model probability is a point estimate with variance, and full
Kelly on a noisy estimate drastically overstakes, and (b) correlated
legs compound unexpected drawdown.

Further guardrails:
  * Never stake more than MAX_STAKE_PCT of bankroll on any single bet
    regardless of what Kelly says.
  * Drop a bet entirely if f* < 0 (i.e. negative-EV — shouldn't happen
    after gating but we re-check).
  * Confidence multiplier: scale the Kelly fraction by (confidence/100)
    so a MEDIUM pick bets half of what a HIGH pick does at the same EV.
"""
from __future__ import annotations

import math
from typing import Optional


# Defaults — override per-bettor via BankrollPolicy.
KELLY_FRACTION = 0.25          # 25% Kelly (a.k.a. quarter Kelly)
MAX_STAKE_PCT = 0.03           # never more than 3% of bankroll on a single bet


def _american_to_decimal(american: int) -> float:
    if american < 0:
        return 1 + 100.0 / (-american)
    return 1 + american / 100.0


def kelly_fraction_for_pick(model_prob: float, american_odds: int) -> float:
    """Full Kelly fraction of bankroll for a single-outcome bet.

    Returns a value in [0, 1]; negative Kelly is clamped to 0 (don't bet).
    """
    if not (0.0 < model_prob < 1.0):
        return 0.0
    b = _american_to_decimal(american_odds) - 1
    if b <= 0:
        return 0.0
    q = 1 - model_prob
    f = (b * model_prob - q) / b
    return max(0.0, f)


def kelly_stake(model_prob: float,
                american_odds: int,
                bankroll: float,
                *,
                confidence: float = 100.0,
                kelly_fraction: float = KELLY_FRACTION,
                max_stake_pct: float = MAX_STAKE_PCT) -> float:
    """Return the $ stake for one bet.

    stake = bankroll * min(max_stake_pct, kelly_fraction * (conf/100) * f_full_kelly)

    * `confidence` is the 0-100 score produced by the predictor
    * `kelly_fraction` is the global fractional-Kelly multiplier
    """
    f = kelly_fraction_for_pick(model_prob, american_odds)
    if f <= 0:
        return 0.0
    conf_mult = max(0.0, min(1.0, confidence / 100.0))
    scaled = f * kelly_fraction * conf_mult
    capped = min(scaled, max_stake_pct)
    return round(bankroll * capped, 2)


def stake_to_units(stake: float, unit_size: float) -> float:
    """Express a stake in units (e.g. 1u = $100 → stake $237 → 2.37u)."""
    if unit_size <= 0:
        return 0.0
    return round(stake / unit_size, 2)
