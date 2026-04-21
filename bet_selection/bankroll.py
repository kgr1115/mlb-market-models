"""
Bankroll management and exposure caps.

Caps stack:
    1) Per-bet cap:  MAX_STAKE_PCT of bankroll on any single leg (enforced in kelly.py)
    2) Per-event cap: across all legs on the same game, cap at PER_EVENT_CAP_PCT
    3) Daily cap:    sum of stakes today <= DAILY_CAP_PCT
    4) Market cap:   per-market exposure (e.g. never more than X% of bankroll
                     on totals across all games) — catches correlated days

When a cap is hit, lower-ranked legs are rescaled or dropped to fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BankrollPolicy:
    bankroll: float
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.03
    per_event_cap_pct: float = 0.04
    daily_cap_pct: float = 0.15
    market_cap_pct: dict[str, float] = field(default_factory=lambda: {
        "moneyline":        0.05,
        "run_line":         0.06,
        "totals":           0.05,
        "f5":               0.04,
        "nrfi":             0.03,
        "team_total_home":  0.04,
        "team_total_away":  0.04,
    })
    unit_size: Optional[float] = None   # if set, output "1.5u" instead of "$150"


def apply_exposure_caps(stakes: list[tuple[str, str, float]],
                         policy: BankrollPolicy
                         ) -> list[tuple[str, str, float]]:
    """
    stakes: list of (event_id, market, stake_dollars) — ranked high→low
    returns: the same list with stakes possibly reduced or zeroed out
    """
    per_event: dict[str, float] = {}
    per_market: dict[str, float] = {}
    day_total = 0.0

    per_event_cap = policy.bankroll * policy.per_event_cap_pct
    daily_cap = policy.bankroll * policy.daily_cap_pct

    out: list[tuple[str, str, float]] = []
    for ev, mkt, s in stakes:
        if s <= 0:
            out.append((ev, mkt, 0.0))
            continue
        # Daily cap
        room_day = max(0.0, daily_cap - day_total)
        # Per-event cap
        room_event = max(0.0, per_event_cap - per_event.get(ev, 0.0))
        # Per-market cap
        mkt_cap = policy.bankroll * policy.market_cap_pct.get(mkt, 0.05)
        room_market = max(0.0, mkt_cap - per_market.get(mkt, 0.0))

        final = min(s, room_day, room_event, room_market)
        final = round(final, 2)

        per_event[ev] = per_event.get(ev, 0.0) + final
        per_market[mkt] = per_market.get(mkt, 0.0) + final
        day_total += final
        out.append((ev, mkt, final))

    return out


def summarize_exposure(stakes: list[tuple[str, str, float]],
                       policy: BankrollPolicy) -> dict:
    totals = {
        "total": sum(s for _, _, s in stakes),
        "by_market": {},
        "by_event": {},
        "bankroll": policy.bankroll,
        "daily_cap_dollars": policy.bankroll * policy.daily_cap_pct,
    }
    for ev, mkt, s in stakes:
        totals["by_market"][mkt] = totals["by_market"].get(mkt, 0.0) + s
        totals["by_event"][ev] = totals["by_event"].get(ev, 0.0) + s
    totals["pct_of_bankroll"] = (totals["total"] / policy.bankroll
                                  if policy.bankroll > 0 else 0.0)
    return totals
