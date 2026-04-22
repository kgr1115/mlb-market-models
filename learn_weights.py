"""
Learned-weight diagnostic.

For each market, fits logistic regression of `won` on the side-aligned
fam_* columns in backtest/ungated_predictions.csv. The fam_* values are
already signed by pick direction: positive = "our side has this family's
advantage". A positive coefficient means the family is predictive; a
negative coefficient means it's anti-signal (should be flipped or zeroed).

Train: 2018-22. Test: 2023-25.

Also fits OLS on profit_per_unit (dollar-weighted) alongside, since bet
payoffs vary with odds and logistic-on-won under-weights +EV longshots.

Output: per-market table comparing
  - current hand-tuned weight
  - learned sign (logistic)
  - learned relative magnitude (normalized so |sum| = |current sum|)
  - learned sign from profit_per_unit regression (cross-check)
  - test-set log-loss with current vs learned

Does NOT modify any predictor file; it's a diagnostic only.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

CSV_PATH = Path(__file__).resolve().parent / "backtest" / "ungated_predictions.csv"

TRAIN_SEASONS = {2018, 2019, 2021, 2022}
TEST_SEASONS = {2023, 2024, 2025, 2026}

ML_FAMS = ["pitcher", "bullpen", "offense", "defense", "baserun", "situational", "environment"]
RL_FAMS = ["pitcher", "bullpen", "offense", "context", "market"]
TT_FAMS = ["pitcher", "bullpen", "offense", "park", "weather", "umpire", "pace", "market"]

ML_WEIGHTS = {"pitcher": 0.30, "bullpen": 0.12, "offense": 0.28, "defense": 0.08,
              "baserun": 0.04, "situational": 0.08, "environment": 0.10}
RL_WEIGHTS = {"pitcher": 0.30, "bullpen": 0.20, "offense": 0.25, "context": 0.15,
              "market": 0.10}
TT_WEIGHTS = {"pitcher": 0.25, "bullpen": 0.15, "offense": 0.20, "park": 0.15,
              "weather": 0.12, "umpire": 0.05, "pace": 0.03, "market": 0.05}

MARKETS = {
    "moneyline": (ML_FAMS, ML_WEIGHTS),
    "run_line":  (RL_FAMS, RL_WEIGHTS),
    "totals":    (TT_FAMS, TT_WEIGHTS),
}


def load_rows(markets):
    rows_by_mkt = defaultdict(list)
    with CSV_PATH.open() as f:
        r = csv.DictReader(f)
        for row in r:
            if row["pick"].startswith("NO BET"):
                continue
            if row["won"] == "":
                continue
            if row["market"] not in markets:
                continue
            rows_by_mkt[row["market"]].append(row)
    return rows_by_mkt


def build_xy(rows, fams):
    X = np.zeros((len(rows), len(fams)), dtype=float)
    y = np.zeros(len(rows), dtype=int)
    prof = np.zeros(len(rows), dtype=float)
    for i, row in enumerate(rows):
        for j, fam in enumerate(fams):
            v = row.get(f"fam_{fam}", "")
            X[i, j] = float(v) if v != "" else 0.0
        y[i] = int(row["won"])
        prof[i] = float(row["profit_per_unit"])
    return X, y, prof


def split_by_season(rows, seasons):
    return [r for r in rows if int(r["season"]) in seasons]


def log_loss(y_true, y_prob, eps=1e-9):
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -float(np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def fit_market(mkt, fams, current_weights, rows):
    train = split_by_season(rows, TRAIN_SEASONS)
    test  = split_by_season(rows, TEST_SEASONS)
    Xtr, ytr, ptr = build_xy(train, fams)
    Xte, yte, pte = build_xy(test, fams)

    # Zero out columns with no variance (dormant families) to avoid spurious coefs
    keep = [j for j in range(len(fams)) if np.std(Xtr[:, j]) > 1e-9]
    fams_kept = [fams[j] for j in keep]
    Xtr_k = Xtr[:, keep]
    Xte_k = Xte[:, keep]

    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
    clf.fit(Xtr_k, ytr)
    coefs = clf.coef_[0]
    intercept = clf.intercept_[0]
    ptr_hat = clf.predict_proba(Xtr_k)[:, 1]
    pte_hat = clf.predict_proba(Xte_k)[:, 1]

    # OLS on profit_per_unit (dollar-weighted sign + magnitude)
    ols = LinearRegression()
    ols.fit(Xtr_k, ptr)
    ols_coefs = ols.coef_

    # Baseline log-loss: predict base_rate everywhere
    base_rate = ytr.mean()
    baseline_ll_tr = log_loss(ytr, np.full_like(ytr, base_rate, dtype=float))
    baseline_ll_te = log_loss(yte, np.full_like(yte, base_rate, dtype=float))

    print(f"\n{'=' * 92}")
    print(f"{mkt.upper()}  n_train={len(train)}  n_test={len(test)}  base_rate_tr={base_rate:.3f}")
    print(f"{'=' * 92}")
    print(f"{'family':14s} {'cur_wt':>7s} {'logistic_coef':>14s} {'sign_match':>11s} "
          f"{'ols_coef':>10s} {'ols_sign':>9s}  {'flag':<30s}")
    print("-" * 92)

    # Map kept-only coefs back to full family list
    full_coefs = {}
    full_ols = {}
    for i, fam in enumerate(fams_kept):
        full_coefs[fam] = coefs[i]
        full_ols[fam] = ols_coefs[i]

    for fam in fams:
        cur = current_weights[fam]
        lc = full_coefs.get(fam, 0.0)
        oc = full_ols.get(fam, 0.0)
        sign_match = "—" if fam not in full_coefs else ("OK" if (lc >= 0 and cur >= 0) or (lc < 0 and cur < 0) else "FLIP")
        ols_sign = "—" if fam not in full_ols else ("+" if oc > 0 else "-")
        flag = ""
        if fam not in full_coefs:
            flag = "dormant (no variance)"
        elif sign_match == "FLIP":
            flag = "LEARNED SIGN DISAGREES"
        elif abs(lc) < 0.03:
            flag = "weak signal"
        print(f"{fam:14s} {cur:>+7.2f} {lc:>+14.4f} {sign_match:>11s} "
              f"{oc:>+10.5f} {ols_sign:>9s}  {flag:<30s}")

    # Test-set log-loss: learned vs baseline
    ll_learned = log_loss(yte, pte_hat)
    print(f"\n  train log-loss (learned):   {log_loss(ytr, ptr_hat):.5f}")
    print(f"  train log-loss (baseline):  {baseline_ll_tr:.5f}")
    print(f"  test  log-loss (learned):   {ll_learned:.5f}")
    print(f"  test  log-loss (baseline):  {baseline_ll_te:.5f}")
    print(f"  test improvement over base: {(baseline_ll_te - ll_learned):+.5f}")

    # Normalize learned coefs to match sum-of-|current_weights| scale for drop-in weights
    cur_abs_sum = sum(abs(current_weights[f]) for f in fams)
    learned_abs_sum = sum(abs(full_coefs.get(f, 0.0)) for f in fams)
    if learned_abs_sum > 0:
        scale = cur_abs_sum / learned_abs_sum
    else:
        scale = 0.0
    print("\n  Suggested drop-in weights (logistic coef × scale, same scale as current):")
    for fam in fams:
        new_w = full_coefs.get(fam, 0.0) * scale
        if fam not in full_coefs:
            new_w = 0.0
        print(f"    {fam:14s}  cur={current_weights[fam]:+7.2f}  suggested={new_w:+7.3f}  "
              f"{'SIGN FLIP' if (new_w * current_weights[fam]) < 0 and new_w != 0 else ''}")


def main():
    rows_by = load_rows(set(MARKETS.keys()))
    for mkt, (fams, wts) in MARKETS.items():
        rows = rows_by.get(mkt, [])
        if not rows:
            print(f"{mkt}: no rows")
            continue
        fit_market(mkt, fams, wts, rows)


if __name__ == "__main__":
    main()
