"""
Rank and dedupe candidate bets.

Why ranking matters
-------------------
On any given day we'll have 15-30 gated-through picks but only 5-10 of
them are worth staking (bankroll pooling + correlation erosion). We
order by:

    score = expected_value_per_unit * (confidence / 100) * size_penalty

Where `size_penalty` downweights bets that rely on tiny implied
probabilities (e.g. a 68%-model on a -400 fav has lower upside).

Correlation dedupe
------------------
Many picks on the same event move together (if the model likes home ML
it will usually also like home -1.5 and the over). To avoid stacking
the same event, we limit each event to at most one "primary" bet from
this shortlist:

    "primary leg" = highest-EV leg for the event
    Other picks on the same event are demoted to "alt" and can still
    be surfaced in the UI but their stake is downscaled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from predictors import PredictionResult


# Correlation coefficients between markets for the SAME event. Empirical
# enough to act as a penalty for portfolio overlap.
MARKET_CORR = {
    ("moneyline", "run_line"): 0.62,
    ("moneyline", "totals"):   0.18,
    ("run_line",  "totals"):   0.15,
    ("moneyline", "f5"):       0.55,
    ("run_line",  "f5"):       0.50,
    ("totals",    "f5"):       0.30,
    ("nrfi",      "totals"):   0.35,
    ("team_total_home", "totals"): 0.70,
    ("team_total_away", "totals"): 0.70,
    ("team_total_home", "moneyline"): 0.30,
    ("team_total_away", "moneyline"): 0.30,
}


@dataclass
class RankedPick:
    event_id: str
    result: PredictionResult
    score: float
    rank: int = 0
    is_primary: bool = True       # False for demoted alt legs
    correlation_with_primary: float = 0.0


def _score(r: PredictionResult) -> float:
    """Composite ranking score."""
    if r.odds is None or r.pick.startswith("NO BET"):
        return -1e9
    ev = max(0.0, r.expected_value_per_unit)
    conf = max(0.0, min(1.0, r.confidence / 100.0))
    # size_penalty: bets with implied prob > 0.75 get modest discount
    # because the payoff is small and variance of edge estimate is wide.
    size_penalty = 1.0 if r.implied_prob <= 0.75 else 0.8
    return ev * conf * size_penalty


def _market_pair_corr(m1: str, m2: str) -> float:
    if m1 == m2:
        return 1.0
    return MARKET_CORR.get((m1, m2), MARKET_CORR.get((m2, m1), 0.0))


def portfolio_correlation(existing: Iterable[PredictionResult],
                          candidate: PredictionResult,
                          event_ids: dict) -> float:
    """Correlation of `candidate` vs the max-correlated existing leg.

    `event_ids` is a dict {id(result): event_id} mapping each leg to
    its source game, which lets us restrict correlation to same-event legs.
    """
    my_event = event_ids.get(id(candidate))
    if my_event is None:
        return 0.0
    max_corr = 0.0
    for e in existing:
        if event_ids.get(id(e)) != my_event:
            continue
        c = _market_pair_corr(candidate.market, e.market)
        if c > max_corr:
            max_corr = c
    return max_corr


def rank_picks(results: Iterable[PredictionResult],
               event_id_of: dict[int, str]) -> list[RankedPick]:
    """Produce a ranked list of betable picks, tagging alt legs."""
    ranked: list[RankedPick] = []
    for r in results:
        if r.pick.startswith("NO BET"):
            continue
        ranked.append(RankedPick(
            event_id=event_id_of.get(id(r), ""),
            result=r, score=_score(r),
        ))
    ranked.sort(key=lambda x: x.score, reverse=True)
    # Dedupe: first pick per event is primary, rest are alts
    seen: dict[str, RankedPick] = {}
    for i, rp in enumerate(ranked):
        rp.rank = i + 1
        if rp.event_id in seen:
            rp.is_primary = False
            primary = seen[rp.event_id]
            rp.correlation_with_primary = _market_pair_corr(
                rp.result.market, primary.result.market)
        else:
            seen[rp.event_id] = rp
    return ranked


def dedupe_correlated(ranked: list[RankedPick],
                      max_legs_per_event: int = 1) -> list[RankedPick]:
    """Keep at most `max_legs_per_event` highest-ranked legs per event.

    The default of 1 is the conservative "one bet per game" policy. Raising
    to 2 permits one primary + one alt (e.g. ML + F5) at the cost of
    increased portfolio correlation.
    """
    kept: list[RankedPick] = []
    seen: dict[str, int] = {}
    for rp in ranked:
        n = seen.get(rp.event_id, 0)
        if n >= max_legs_per_event:
            continue
        kept.append(rp)
        seen[rp.event_id] = n + 1
    return kept
