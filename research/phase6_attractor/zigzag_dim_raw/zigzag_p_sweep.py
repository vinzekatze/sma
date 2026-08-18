#!/usr/bin/env python3
"""
Свип по p (размерность вложения) для объединённого пула 1d(4%)+10m(0.4%).

КОНТРАКТ КАУЗАЛЬНОСТИ:
  Алгоритм не видит ничего после origin (step).
  Пул: j ∈ [p-1, step-1]. Aux: searchsorted(dates_aux, cur_date, side="left").

Фиксировано: T_prim=4%, T_10m=0.4%, H=1, MIN_HISTORY=50.
K = 3·(p+1) для каждого p.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR    = Path(__file__).parent
DATA        = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT         = BASE_DIR / "results"

T_PRIM      = 0.04
T_10M       = 0.004
H           = 1
MIN_HISTORY = 50
P_GRID      = [3, 4, 5, 6, 7, 8, 9]


def load_tf(tf):
    with open(DATA / f"{tf}.json") as f:
        data = json.load(f)
    h     = np.array([d["high"]  for d in data], dtype=np.float64)
    l     = np.array([d["low"]   for d in data], dtype=np.float64)
    dates = np.array([d["begin"] for d in data])
    return h, l, dates


def find_pivots_hl(highs, lows, dates, thr):
    pivots = []
    direction = 0
    ext_val = (highs[0] + lows[0]) / 2.0
    ext_idx = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1;  ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val, ext_idx = lows[i], i
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                pivots.append((ext_idx, ext_val, dates[ext_idx]))
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                pivots.append((ext_idx, ext_val, dates[ext_idx]))
                direction = 1;  ext_val, ext_idx = highs[i], i
    if not pivots:
        return np.array([]), np.array([])
    prices = np.array([pv for _, pv, _ in pivots])
    pdates = np.array([d  for _, _,  d in pivots])
    return prices, pdates


def build_X(prices, p):
    """X[i] = [price[i], log(p[i]/p[i-1]), ..., log(p[i-p+2]/p[i-p+1])]"""
    n  = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def lwr_pred(X_pool, y_pool, x_q, k):
    if len(X_pool) < k:
        return np.nan
    mu    = X_pool.mean(0)
    sigma = X_pool.std(0)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)
    Xn    = (X_pool - mu) / sigma
    xn    = (x_q    - mu) / sigma
    dists = np.linalg.norm(Xn - xn, axis=1)
    order = np.argsort(dists)
    knn   = order[:k]
    xi    = dists[order[k - 1]]
    if xi < 1e-12:
        return float(y_pool[knn].mean())
    w   = np.exp(-0.5 * (dists[knn] / xi) ** 2)
    ws  = np.sqrt(w)
    A   = np.column_stack([np.ones(k), Xn[knn]]) * ws[:, None]
    b   = y_pool[knn] * ws
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0] + coef[1:] @ xn)


def walk_forward(prices_prim, dates_prim, X_prim,
                 aux_list, p, k):
    n = len(prices_prim)
    preds, actuals = [], []

    for step in range(max(MIN_HISTORY, p), n - H):
        if np.any(np.isnan(X_prim[step])):
            continue
        target   = prices_prim[step + H]
        cur_date = dates_prim[step]

        j_arr = np.arange(p - 1, step)
        valid  = ~np.any(np.isnan(X_prim[j_arr]), axis=1) & (j_arr + H < n)
        X_rows = list(X_prim[j_arr[valid]])
        y_rows = list(prices_prim[j_arr[valid] + H])

        for prices_a, dates_a, X_a in aux_list:
            ce = int(np.searchsorted(dates_a, cur_date, side="left"))
            if ce < p:
                continue
            na  = len(prices_a)
            j_a = np.arange(p - 1, min(ce, na - H))
            if len(j_a) == 0:
                continue
            valid_a = ~np.any(np.isnan(X_a[j_a]), axis=1) & (j_a + H < na)
            X_rows.extend(X_a[j_a[valid_a]])
            y_rows.extend(prices_a[j_a[valid_a] + H])

        if len(X_rows) < k:
            continue

        pred = lwr_pred(np.array(X_rows), np.array(y_rows), X_prim[step], k)
        if np.isnan(pred):
            continue
        preds.append(pred)
        actuals.append(target)

    return np.array(preds), np.array(actuals)


def rmae(preds, actuals):
    if len(preds) < 5:
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    if dz < 1e-12:
        return np.nan
    return float(np.mean(np.abs(preds - actuals)) / dz)


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")

    prices_prim, dates_prim = find_pivots_hl(h1d,  l1d,  d1d,  T_PRIM)
    prices_10m,  dates_10m  = find_pivots_hl(h10m, l10m, d10m, T_10M)

    print(f"SBER  1d  T={T_PRIM*100:.0f}%   → {len(prices_prim)} пивотов")
    print(f"SBER  10m T={T_10M*100:.1f}% → {len(prices_10m)} пивотов")
    print()
    print(f"{'p':>3}  {'K':>4}  {'rMAE_1d':>9}  {'rMAE_comb':>10}  {'vs_1d':>7}  {'vs_p3':>7}  n_test")
    print("─" * 62)

    records    = []
    r_p3_comb  = None

    for p in P_GRID:
        k      = 3 * (p + 1)
        X_prim = build_X(prices_prim, p)
        X_10m  = build_X(prices_10m,  p)

        pr0, ac0 = walk_forward(prices_prim, dates_prim, X_prim, [], p, k)
        r0 = rmae(pr0, ac0)

        aux      = [(prices_10m, dates_10m, X_10m)]
        pr1, ac1 = walk_forward(prices_prim, dates_prim, X_prim, aux, p, k)
        r1 = rmae(pr1, ac1)

        if p == 3:
            r_p3_comb = r1

        vs_1d = (r1 / r0 - 1) * 100 if not np.isnan(r0) else np.nan
        vs_p3 = (r1 / r_p3_comb - 1) * 100 if r_p3_comb else np.nan

        mark = " ←" if (not np.isnan(vs_p3) and vs_p3 < -0.3) else ""
        print(f"  p={p}  K={k:3d}  {r0:.4f}     {r1:.4f}     {vs_1d:+6.1f}%  {vs_p3:+6.1f}%  {len(pr1)}{mark}")

        records.append({"p": p, "K": k,
                        "rMAE_1d": r0, "rMAE_comb": r1,
                        "vs_1d_pct": vs_1d, "vs_p3_pct": vs_p3,
                        "n_test": len(pr1)})

    df = pd.DataFrame(records)
    df.to_csv(OUT / "p_sweep.csv", index=False)

    # ── График ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["p"], df["rMAE_1d"],   "o--", color="gray",      lw=1.5, ms=6,
            label="только 1d")
    ax.plot(df["p"], df["rMAE_comb"], "o-",  color="steelblue", lw=2,   ms=8,
            label="1d + 10m (0.4%)")

    best = df.loc[df["rMAE_comb"].idxmin()]
    ax.axvline(best["p"], color="steelblue", lw=1, ls=":", alpha=0.6)
    ax.annotate(f"  p={int(best['p'])}\n  rMAE={best['rMAE_comb']:.4f}",
                xy=(best["p"], best["rMAE_comb"]),
                xytext=(best["p"] + 0.15, best["rMAE_comb"] + 0.003),
                fontsize=8.5, color="steelblue")

    r_p3 = df[df["p"] == 3]["rMAE_comb"].values[0]
    ax.axhline(r_p3, color="steelblue", lw=0.8, ls="--", alpha=0.35,
               label=f"p=3 rMAE={r_p3:.4f}")

    ax.set_xlabel("p (размерность вложения)")
    ax.set_ylabel("rMAE")
    ax.set_title("Свип p: LWR 1d(4%) + 10m(0.4%) — SBER\n"
                 "K = 3·(p+1), H=1, walk-forward причинный")
    ax.set_xticks(P_GRID)
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT / "p_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nЛучший p по rMAE_comb: p={int(best['p'])}, rMAE={best['rMAE_comb']:.4f}")
    print(f"CSV + график: {OUT}")


if __name__ == "__main__":
    run()
