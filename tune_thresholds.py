"""
Train/test grid search over (min_edge, min_confidence_score) per market.

Reads backtest/ungated_predictions.csv (from collect_predictions.py),
splits 2018-2022 = TRAIN, 2023-2025 = TEST, and for each market
grid-searches thresholds on TRAIN only.

The train-optimal thresholds are then applied to TEST for an honest
out-of-sample number. For each market we also show:
  - Current live thresholds applied to TEST (what we'd ship today)
  - "Bet everything non-NO-BET" null baseline on TEST

Floor: require N >= 200 bets for a threshold to be eligible; otherwise
grid optimizers will pick tiny over-fit cells.
"""
from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("tune_thresholds")


CSV_PATH = Path(__file__).resolve().parent / "backtest" / "ungated_predictions.csv"

TRAIN_SEASONS = {2018, 2019, 2021, 2022}
TEST_SEASONS  = {2023, 2024, 2025}

# Current live thresholds (for comparison against tuned ones)
# - ML/RL: LEAN+LOW rewritten to NO BET (conf < 55)
# - Totals: conf < 99 -> NO BET
# - All three share MIN_EDGE = 0.025
CURRENT = {
    "moneyline": {"min_edge": 0.025, "min_conf": 55.0},
    "run_line":  {"min_edge": 0.025, "min_conf": 55.0},
    "totals":    {"min_edge": 0.025, "min_conf": 99.0},
}

# Grid
EDGE_GRID = [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.060, 0.080, 0.100]
CONF_GRID = [0.0, 30.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 99.0]
MIN_N = 200


def load_rows():
    rows = []
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            if r["pick"].startswith("NO BET"):
                continue
            if r["won"] == "":   # push or odds missing
                # Count pushes as "bets" with 0 profit only if pushed == 1;
                # otherwise skip (odds missing means we shouldn't grade).
                if r["pushed"] != "1":
                    continue
            rows.append({
                "season": int(r["season"]),
                "market": r["market"],
                "edge": float(r["edge"]),
                "conf": float(r["confidence_score"]),
                "profit": float(r["profit_per_unit"]),
                "pushed": r["pushed"] == "1",
            })
    return rows


def roi(bets):
    n = len(bets)
    if n == 0:
        return 0.0, 0
    return sum(b["profit"] for b in bets) / n * 100.0, n


def apply_threshold(rows, mkt, min_edge, min_conf):
    """Filter rows for a market to those that clear BOTH thresholds."""
    return [b for b in rows
            if b["market"] == mkt
            and b["edge"] >= min_edge
            and b["conf"] >= min_conf]


def grid_search(train_rows, mkt):
    """Return (best_edge, best_conf, best_train_roi, best_n) for one market.
    Tie-break: among cells with ROI > 0 and N >= MIN_N, pick the one with
    the largest N (more data = more reliable)."""
    best = None
    for me in EDGE_GRID:
        for mc in CONF_GRID:
            bets = apply_threshold(train_rows, mkt, me, mc)
            r, n = roi(bets)
            if n < MIN_N:
                continue
            key = (r, n)  # higher ROI wins, break ties by n
            if best is None or key > (best[2], best[3]):
                best = (me, mc, r, n)
    return best


def main():
    all_rows = load_rows()
    train = [r for r in all_rows if r["season"] in TRAIN_SEASONS]
    test  = [r for r in all_rows if r["season"] in TEST_SEASONS]

    print(f"Total ungated graded rows : {len(all_rows):,}")
    print(f"  TRAIN (2018-22)         : {len(train):,}")
    print(f"  TEST  (2023-25)         : {len(test):,}")
    print()

    print(f"{'market':10s}  {'regime':24s}  {'min_edge':>9s} "
          f"{'min_conf':>9s} {'n_test':>7s} {'roi_test':>9s}")
    print("-" * 80)

    pooled_test_picks = []
    recommendations = {}

    for mkt in ("moneyline", "run_line", "totals"):
        # (1) current live thresholds on TEST
        cur = CURRENT[mkt]
        cur_test = apply_threshold(test, mkt, cur["min_edge"], cur["min_conf"])
        cur_r, cur_n = roi(cur_test)

        # (2) null / "bet everything" on TEST
        null_test = apply_threshold(test, mkt, 0.0, 0.0)
        null_r, null_n = roi(null_test)

        # (3) grid-search TRAIN, then evaluate on TEST
        best = grid_search(train, mkt)
        if best is None:
            print(f"{mkt:10s}  (no train cell passed N>={MIN_N})")
            continue
        me, mc, train_roi, train_n = best
        tuned_test = apply_threshold(test, mkt, me, mc)
        tuned_r, tuned_n = roi(tuned_test)

        # Also report train performance of tuned cell for sanity
        print(f"{mkt:10s}  {'NULL (everything)':24s}  "
              f"{0.0:>9.3f} {0.0:>9.1f} {null_n:>7d} {null_r:>+8.2f}%")
        print(f"{mkt:10s}  {'CURRENT (live gates)':24s}  "
              f"{cur['min_edge']:>9.3f} {cur['min_conf']:>9.1f} "
              f"{cur_n:>7d} {cur_r:>+8.2f}%")
        print(f"{mkt:10s}  {'TUNED (train optimum)':24s}  "
              f"{me:>9.3f} {mc:>9.1f} {tuned_n:>7d} {tuned_r:>+8.2f}%  "
              f"(train: n={train_n} roi={train_roi:+.2f}%)")
        print()

        recommendations[mkt] = {
            "current": {"min_edge": cur["min_edge"],
                        "min_conf": cur["min_conf"],
                        "test_n": cur_n, "test_roi": cur_r},
            "tuned":   {"min_edge": me, "min_conf": mc,
                        "train_n": train_n, "train_roi": train_roi,
                        "test_n": tuned_n, "test_roi": tuned_r},
            "null":    {"test_n": null_n, "test_roi": null_r},
        }
        pooled_test_picks.extend(tuned_test)

    # Pooled TEST performance of tuned thresholds
    if pooled_test_picks:
        pr, pn = roi(pooled_test_picks)
        print("-" * 80)
        print(f"POOLED test (all tuned thresholds applied): "
              f"n={pn} roi={pr:+.2f}%")

    # Judgment call helper
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    for mkt, rec in recommendations.items():
        cur_r = rec["current"]["test_roi"]
        tun_r = rec["tuned"]["test_roi"]
        tun_train = rec["tuned"]["train_roi"]
        delta = tun_r - cur_r
        overfit = tun_train - tun_r
        verdict = ""
        if tun_r > 0 and tun_train > 0:
            verdict = "SHIP TUNED"
        elif tun_r > cur_r and tun_r > -1.0:
            verdict = "TUNED IMPROVES — cautious ship"
        elif tun_train > 0 and tun_r < 0:
            verdict = "OVERFIT — tuned wins train, loses test"
        else:
            verdict = "NO EDGE — ship defensive / don't bet"
        print(f"  {mkt:10s}  train ROI {tun_train:+.2f}%  test ROI {tun_r:+.2f}%  "
              f"vs current test {cur_r:+.2f}%  (delta {delta:+.2f}%)  "
              f"[overfit gap {overfit:+.2f}%]  => {verdict}")


if __name__ == "__main__":
    main()
