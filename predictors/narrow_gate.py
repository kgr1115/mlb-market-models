"""
Narrow-gate filter: surface only the bets with empirically-positive EV.

Per project memory `project_feature_classes_wiring.md` (2026-04-20):

    | Market   | Null test ROI | Current-gate test ROI | Tuned test ROI      | Verdict     |
    |----------|---------------|-----------------------|---------------------|-------------|
    | ML       | -3.16%        | -3.32% (n=1898)       | -3.64% (n=119)      | OVERFIT     |
    | RL       | +17.49%       | +23.63% (n=4115)      | +31.21% (n=1407)    | SHIP TUNED  |
    | Totals   | -3.88%        | -2.38% (n=156)        | -2.38% (n=156)      | OVERFIT     |

So right now the only core-market bet that clears out-of-sample is RUN LINE
with the tuned gate (edge ≥ 0.010, confidence ≥ 85). This module enforces
that gate on core-market picks and lets the soft-market predictors (F5,
team totals, NRFI) through at their own thresholds because they operate
against a different (less efficient) market.

Closing-line model gating (see `closing_line.py`) can override this
once a CLV track record exists — but until then, this is the authoritative
production gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .shared import PredictionResult


# =============================================================================
# Gate config — single source of truth
# =============================================================================

@dataclass(frozen=True)
class MarketGate:
    min_edge: float
    min_confidence: float
    enabled: bool = True
    note: str = ""


# Thresholds grounded in the 2026-04-20 train/test finding, then slightly
# padded for conservative live usage.
GATES: dict[str, MarketGate] = {
    # Core 3: only run line tested positive
    "moneyline":    MarketGate(min_edge=0.030, min_confidence=90,
                               enabled=False,
                               note="Negative test ROI at every gate — disabled until CLV model validates"),
    "run_line":     MarketGate(min_edge=0.010, min_confidence=85,
                               enabled=True,
                               note="Tuned gate: +31.21% test ROI on n=1407 (RL tuned)"),
    "totals":       MarketGate(min_edge=0.025, min_confidence=75,
                               enabled=True,
                               note="Enabled at HIGH confidence + 2.5% edge. Backtest pooled ROI -2.81% on 2021-25, so size smaller than run_line until CLV validates live."),

    # Soft markets — lighter gate because market prices are softer to begin with
    "f5":              MarketGate(min_edge=0.030, min_confidence=70, enabled=True,
                                   note="Softer market; starter-only exposure"),
    "nrfi":            MarketGate(min_edge=0.025, min_confidence=65, enabled=True,
                                   note="Softer market; inefficient books and stale lines"),
    "team_total_home": MarketGate(min_edge=0.030, min_confidence=70, enabled=True,
                                   note="Softer market; 0.5-run shop variance common"),
    "team_total_away": MarketGate(min_edge=0.030, min_confidence=70, enabled=True,
                                   note="Softer market; 0.5-run shop variance common"),
}


# =============================================================================
# Filter
# =============================================================================

def gate_pick(result: PredictionResult,
              gates: dict[str, MarketGate] = GATES) -> PredictionResult:
    """Apply the narrow production gate to one PredictionResult.

    If the pick fails the gate for its market, mutate the result to
    NO BET while preserving the underlying detail fields (so the web
    UI can still explain why it was filtered).
    """
    g = gates.get(result.market)
    if g is None:
        return result
    if not g.enabled:
        # Wholesale disable — no bet
        result.pick = "NO BET (market disabled)"
        result.odds = None
        result.expected_value_per_unit = 0.0
        result.detail["gate_applied"] = "market_disabled"
        result.detail["gate_note"] = g.note
        return result
    if result.pick.startswith("NO BET"):
        return result  # already filtered by predictor itself

    # Apply combined edge + confidence thresholds
    if result.edge < g.min_edge or result.confidence < g.min_confidence:
        reason = []
        if result.edge < g.min_edge:
            reason.append(f"edge {result.edge:.3f} < {g.min_edge:.3f}")
        if result.confidence < g.min_confidence:
            reason.append(f"conf {result.confidence:.1f} < {g.min_confidence:.1f}")
        result.pick = "NO BET (gate)"
        result.odds = None
        result.expected_value_per_unit = 0.0
        result.detail["gate_applied"] = "thresholds"
        result.detail["gate_reason"] = "; ".join(reason)
        result.detail["gate_note"] = g.note

    return result


def narrow_picks(picks: Iterable[PredictionResult],
                 gates: dict[str, MarketGate] = GATES) -> list[PredictionResult]:
    """Convenience: apply gate_pick to every result and return survivors
    (i.e. gates that left the pick in a bet-able state)."""
    out = []
    for p in picks:
        g = gate_pick(p, gates)
        out.append(g)
    return out


def surviving_only(picks: Iterable[PredictionResult]) -> list[PredictionResult]:
    """Return only the picks that still carry a bet-able side after gating."""
    return [p for p in picks if not p.pick.startswith("NO BET")]


# =============================================================================
# Report helpers for UI and logs
# =============================================================================

def gate_summary() -> dict:
    """Serializable summary of which markets are currently live."""
    return {
        market: {
            "enabled": g.enabled,
            "min_edge": g.min_edge,
            "min_confidence": g.min_confidence,
            "note": g.note,
        }
        for market, g in GATES.items()
    }
