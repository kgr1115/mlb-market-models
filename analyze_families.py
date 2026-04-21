"""
Per-family signal diagnostic (train set only, 2018-22).

For each (market, family), measures two things on TRAIN:

1. Point-biserial correlation between the family's side-aligned score and
   the bet outcome (1=won, 0=lost). A positive correlation means the
   family's magnitude is telling you *correctly* which side to bet.
   Zero = noise. Negative = anti-predictive.

2. "Family-agree" ROI: restrict to bets where this family's side-aligned
   score is > 0 (i.e. the family agrees with the overall pick direction)
   vs. when it's <= 0. A family that's doing useful work should show
   positive delta: agree-ROI > disagree-ROI.

Also prints the family's contribution to the weighted sum (mean abs score ×
weight) so you can sort by how much each family is *moving* the prediction.

Output: per-market ranked table. Flags candidates to prune (noise) or
invert (anti-signal).
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "backtest" / "ungated_predictions.csv"
TRAIN_SEASONS = {2018, 2019, 2021, 2022}

# Weights used by each predictor (for "influence" display)
ML_WEIGHTS = {
    "pitcher": 0.30, "bullpen": 0.12, "offense": 0.28, "defense": 0.08,
    "baserun": 0.04, "situational": 0.08, "environment": 0.10,
}
RL_WEIGHTS = {
    "pitcher": 0.30, "bullpen": 0.20, "offense": 0.25, "context": 0.15,
    "market": 0.10,
}
TT_WEIGHTS = {
    "pitcher": 0.25, "bullpen": 0.15, "offense": 0.20, "park": 0.15,
    "weather": 0.12, "umpire": 0.05, "pace": 0.03, "market": 0.05,
}
WEIGHTS = {"moneyline": ML_WEIGHTS, "run_line": RL_WEIGHTS, "totals": TT_WEIGHTS}


def load_train():
    rows = []
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            if int(r["season"]) not in TRAIN_SEASONS:
                continue
            if r["pick"].startswith("NO BET"):
                continue
            if r["won"] == "":   # unresolved / push
                continue
            rows.append(r)
    return rows


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def analyze_market(rows, mkt):
    mrows = [r for r in rows if r["market"] == mkt]
    weights = WEIGHTS[mkt]
    print(f"\n{'=' * 88}\n{mkt.upper()}   n_train_bets={len(mrows)}\n{'=' * 88}")
    print(f"{'family':14s} {'weight':>7s} {'mean|score|':>11s} "
          f"{'influence':>10s} {'corr(w.score,won)':>18s} "
          f"{'agree_roi':>10s} {'dis_roi':>10s} {'roi_delta':>10s}"
          f" {'n_agree':>7s}")
    print("-" * 102)

    diagnostics = []
    for fam in weights:
        col = f"fam_{fam}"
        values = []
        outcomes = []
        profits_agree = []
        profits_disagree = []
        for r in mrows:
            v = r.get(col, "")
            if v == "":
                continue
            try:
                score = float(v)
            except ValueError:
                continue
            won = int(r["won"])
            profit = float(r["profit_per_unit"])
            values.append(score)
            outcomes.append(won)
            if score > 0:
                profits_agree.append(profit)
            elif score < 0:
                profits_disagree.append(profit)
        if not values:
            continue
        mean_abs = sum(abs(v) for v in values) / len(values)
        influence = weights[fam] * mean_abs
        corr = pearson(values, outcomes)
        agree_roi  = (sum(profits_agree) / len(profits_agree) * 100) if profits_agree else 0.0
        dis_roi    = (sum(profits_disagree) / len(profits_disagree) * 100) if profits_disagree else 0.0
        roi_delta  = agree_roi - dis_roi
        n_agree    = len(profits_agree)
        diagnostics.append((fam, weights[fam], mean_abs, influence, corr,
                            agree_roi, dis_roi, roi_delta, n_agree))

    # Sort by influence (largest contribution to the weighted sum, first)
    diagnostics.sort(key=lambda x: -x[3])
    for fam, w, mabs, infl, corr, ag, dis, delta, na in diagnostics:
        # Flag interpretation
        flag = ""
        if abs(corr) < 0.02 and abs(delta) < 0.5:
            flag = "  <- NOISE"
        elif corr < -0.02 or delta < -1.0:
            flag = "  <- ANTI-SIGNAL (prune/invert)"
        elif corr > 0.02 and delta > 1.0:
            flag = "  <- SIGNAL"
        print(f"{fam:14s} {w:>7.2f} {mabs:>11.3f} {infl:>10.4f} "
              f"{corr:>+18.4f} {ag:>+9.2f}% {dis:>+9.2f}% {delta:>+9.2f}%"
              f" {na:>7d}{flag}")


def main():
    rows = load_train()
    print(f"Loaded {len(rows)} graded TRAIN bets (2018-22).")
    for mkt in ("moneyline", "run_line", "totals"):
        analyze_market(rows, mkt)


if __name__ == "__main__":
    main()
