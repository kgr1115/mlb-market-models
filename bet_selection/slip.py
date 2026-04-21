"""
Bet-slip builder — the single entry point for the UI.

Pipeline:

    raw_picks  -- from predictors (already narrowed via narrow_gate)
         │
         ├─  rank_picks()          : EV × confidence ordering, alt-leg tagging
         │
         ├─  dedupe_correlated()   : 1 leg per event by default
         │
         ├─  kelly_stake()         : fractional Kelly stake per leg
         │
         └─  apply_exposure_caps() : per-event, per-market, daily caps

Output is a BetSlip containing per-leg stakes (in dollars AND units),
the book to route each leg to (from the shopped-market metadata), a
total exposure summary, and a `clv_warning_flags` list for legs where
the current line is *worse* than the predicted closing line (we should
wait), and `clv_positive_flags` where we have CLV-positive conditions.

The frontend renders this as:

    TODAY'S SLIP
    ────────────
    1. [RL] COL +1.5 (-115) @ FanDuel  — Stake: $75 (0.75u)  Conf 88  EV +4.2%
    2. [F5] ATL F5 ML (-125) @ DK      — Stake: $50 (0.50u)  Conf 72  EV +3.8%
    ...
    Total exposure: $325  (3.3% of bankroll, 4 legs)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from predictors import PredictionResult
from data.line_shop import ShoppedMarket

from .kelly import kelly_stake, stake_to_units, KELLY_FRACTION
from .bankroll import BankrollPolicy, apply_exposure_caps, summarize_exposure
from .ranker import rank_picks, dedupe_correlated, RankedPick


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class BetSlipLeg:
    """One leg of the slip, ready to display / route."""
    rank: int
    event_id: str
    market: str
    pick: str
    odds: int
    book: Optional[str]
    stake_dollars: float
    stake_units: float
    confidence: float
    edge: float
    ev_per_unit: float
    is_primary: bool
    correlation_with_primary: float = 0.0
    clv_flag: str = ""    # "positive" | "warning" | ""


@dataclass
class BetSlip:
    legs: list[BetSlipLeg] = field(default_factory=list)
    total_stake: float = 0.0
    total_stake_units: float = 0.0
    policy: Optional[BankrollPolicy] = None
    exposure: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "legs": [leg.__dict__ for leg in self.legs],
            "total_stake": self.total_stake,
            "total_stake_units": self.total_stake_units,
            "exposure": self.exposure,
        }


# =============================================================================
# Builder
# =============================================================================

def build_slip(picks: Iterable[PredictionResult],
               event_id_of: dict[int, str],
               policy: BankrollPolicy,
               *,
               shopped_markets: Optional[dict[str, ShoppedMarket]] = None,
               predicted_closing_by_event: Optional[dict[str, dict]] = None,
               max_legs_per_event: int = 1) -> BetSlip:
    """Produce a complete bet slip from gated picks.

    Parameters
    ----------
    picks : iterable of PredictionResult, already passed through narrow_gate
    event_id_of : mapping id(result) -> event_id, used for dedupe + caps
    policy : BankrollPolicy
    shopped_markets : optional map event_id -> ShoppedMarket, used to route
        each leg to the book offering the best price on the chosen side
    predicted_closing_by_event : optional map event_id -> {market: predicted_prob}
        used to flag CLV-positive or CLV-warning situations
    """
    picks = list(picks)
    ranked = rank_picks(picks, event_id_of)
    kept = dedupe_correlated(ranked, max_legs_per_event=max_legs_per_event)

    # Kelly-size every leg
    pre_stakes: list[tuple[str, str, float]] = []
    for rp in kept:
        r = rp.result
        stake = kelly_stake(
            model_prob=r.model_prob,
            american_odds=r.odds if r.odds else -110,
            bankroll=policy.bankroll,
            confidence=r.confidence,
            kelly_fraction=policy.kelly_fraction,
            max_stake_pct=policy.max_stake_pct,
        )
        # Alt legs (non-primary) get a 40% stake haircut due to correlation
        if not rp.is_primary:
            stake *= 0.4
        pre_stakes.append((rp.event_id, r.market, stake))

    capped = apply_exposure_caps(pre_stakes, policy)

    # Build slip legs
    slip = BetSlip(policy=policy)
    for (rp, (_, _, stake)) in zip(kept, capped):
        r = rp.result
        book = _route_book(rp.event_id, r.market, r.pick, shopped_markets)
        clv_flag = _clv_flag(rp.event_id, r, predicted_closing_by_event)
        leg = BetSlipLeg(
            rank=rp.rank,
            event_id=rp.event_id,
            market=r.market,
            pick=r.pick,
            odds=r.odds if r.odds else 0,
            book=book,
            stake_dollars=stake,
            stake_units=(stake_to_units(stake, policy.unit_size)
                         if policy.unit_size else 0.0),
            confidence=r.confidence,
            edge=r.edge,
            ev_per_unit=r.expected_value_per_unit,
            is_primary=rp.is_primary,
            correlation_with_primary=rp.correlation_with_primary,
            clv_flag=clv_flag,
        )
        if leg.stake_dollars > 0:
            slip.legs.append(leg)

    slip.total_stake = sum(l.stake_dollars for l in slip.legs)
    slip.total_stake_units = sum(l.stake_units for l in slip.legs)
    slip.exposure = summarize_exposure(
        [(l.event_id, l.market, l.stake_dollars) for l in slip.legs],
        policy,
    )
    return slip


# =============================================================================
# Helpers
# =============================================================================

def _route_book(event_id: str, market: str, pick: str,
                 shopped_markets: Optional[dict[str, ShoppedMarket]]) -> Optional[str]:
    """Return the book that offers the best price on the chosen side.

    pick strings look like "HOME -1.5 +110", "OVER 8.5 -105", etc.
    We infer the side key into ShoppedMarket and return .book.
    """
    if not shopped_markets or event_id not in shopped_markets:
        return None
    sm = shopped_markets[event_id]
    # Crude mapping pick → side
    pick_upper = pick.upper()
    if market == "moneyline":
        if pick_upper.startswith("HOME"):
            return sm.home_ml.book.value if sm.home_ml else None
        if pick_upper.startswith("AWAY"):
            return sm.away_ml.book.value if sm.away_ml else None
    if market == "run_line":
        if pick_upper.startswith("HOME"):
            return sm.home_rl.book.value if sm.home_rl else None
        if pick_upper.startswith("AWAY"):
            return sm.away_rl.book.value if sm.away_rl else None
    if market == "totals":
        if "OVER" in pick_upper:
            return sm.over.book.value if sm.over else None
        if "UNDER" in pick_upper:
            return sm.under.book.value if sm.under else None
    # soft markets: no shop record → None
    return None


def _clv_flag(event_id: str, r: PredictionResult,
              predicted_closing_by_event: Optional[dict[str, dict]]) -> str:
    """Flag the leg as "positive" (current line beats predicted close)
    or "warning" (line expected to move in our favor — wait for better).
    """
    if not predicted_closing_by_event or event_id not in predicted_closing_by_event:
        return ""
    pred = predicted_closing_by_event[event_id].get(r.market)
    if pred is None:
        return ""
    # Convention: pred is the predicted *de-vigged* probability for the
    # side the model is picking. If pred < implied_prob → market is
    # expected to shorten (bet NOW), CLV-positive. If pred > implied →
    # wait, line moving our way, CLV-warning.
    if pred < r.implied_prob - 0.005:
        return "positive"    # line is expected to shorten; snipe it
    if pred > r.implied_prob + 0.005:
        return "warning"     # line expected to drift our way; wait
    return ""
