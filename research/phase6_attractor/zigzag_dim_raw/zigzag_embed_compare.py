#!/usr/bin/env python3
"""
Сравнение вложений: цена+разности vs чистые разности.
SBER 1d, T=2%, LWR K=3(p+1), H=1.

Вариант A — текущий:
  X[i] = [price[i], log(p[i]/p[i-1]), log(p[i-1]/p[i-2])]
  p=3, K=12

Вариант B — чистые разности (norm_study: log_swing):
  X[i] = [log(p[i]/p[i-1]), log(p[i-1]/p[i-2])]
  p=2, K=9

Вариант C — ratio (logtrend-норм.) вместо сырой цены:
  X[i] = [ratio[i], log(r[i]/r[i-1]), log(r[i-1]/r[i-2])]
  p=3, K=12

Метрика: rMAE в пространстве цены (не ratio).
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT         = BASE_DIR / "results"

T_ZZ        = 0.02
H           = 1
MIN_HISTORY = 50


def load_1d():
    with open(DATA_DIR / "1d.json") as f:
        data = json.load(f)
    close = np.array([d["close"] for d in data], dtype=np.float64)
    high  = np.array([d["high"]  for d in data], dtype=np.float64)
    low   = np.array([d["low"]   for d in data], dtype=np.float64)
    t     = np.arange(len(close), dtype=np.float64)
    return high, low, close, t


def logtrend(close, t):
    """Causal logtrend OLS — инкрементальные суммы, без разгрева."""
    n = len(close)
    lc = np.log(close)
    S0 = np.arange(1, n + 1, dtype=np.float64)
    S1 = np.cumsum(t)
    S2 = np.cumsum(t ** 2)
    Sy = np.cumsum(lc)
    St = np.cumsum(t * lc)
    denom = S0 * S2 - S1 ** 2
    b = np.where(np.abs(denom) > 1e-12, (S0 * St - S1 * Sy) / denom, 0.0)
    a = np.where(np.abs(denom) > 1e-12, (Sy - b * S1) / S0, lc)
    return np.exp(a + b * t)


def find_pivots(highs, lows, thr):
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
                pivots.append((ext_idx, ext_val))
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                pivots.append((ext_idx, ext_val))
                direction = 1;  ext_val, ext_idx = highs[i], i
    idx    = np.array([i for i, _ in pivots])
    prices = np.array([p for _, p in pivots])
    return idx, prices


def build_X_price(prices, p):
    """X[i] = [price[i], log(p[i]/p[i-1]), ..., log(p[i-p+2]/p[i-p+1])]"""
    n  = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def build_X_diff(prices, p):
    """X[i] = [log(p[i]/p[i-1]), ..., log(p[i-p+1]/p[i-p])]  — только разности."""
    n  = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p), np.nan)
    for i in range(p, n):
        for lag in range(p):
            X[i, lag] = lp[i - lag] - lp[i - lag - 1]
    return X


def lwr(X_pool, y_pool, x_q, k):
    if len(X_pool) < k:
        return np.nan
    mu    = X_pool.mean(0)
    sigma = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
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


def walk_forward(prices, X, k, relative=False):
    """
    relative=False: цель = prices[step+H]  (абсолютная цена)
    relative=True:  цель = log(prices[step+H]/prices[step]),
                    реконструкция: pred_price = prices[step] * exp(pred_log_ret)
    """
    p = X.shape[1]
    n = len(prices)
    preds, actuals = [], []
    lp = np.log(prices)
    for step in range(max(MIN_HISTORY, p), n - H):
        if np.any(np.isnan(X[step])):
            continue
        j_arr = np.arange(p - 1, step)
        valid  = ~np.any(np.isnan(X[j_arr]), axis=1) & (j_arr + H < n)
        Xp = X[j_arr[valid]]
        if relative:
            yp = lp[j_arr[valid] + H] - lp[j_arr[valid]]   # log-return
        else:
            yp = prices[j_arr[valid] + H]
        if len(Xp) < k:
            continue
        pred_raw = lwr(Xp, yp, X[step], k)
        if np.isnan(pred_raw):
            continue
        pred = prices[step] * np.exp(pred_raw) if relative else pred_raw
        preds.append(pred)
        actuals.append(prices[step + H])
    return np.array(preds), np.array(actuals)


def rmae(preds, actuals):
    if len(preds) < 5:
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return np.nan if dz < 1e-12 else float(np.mean(np.abs(preds - actuals)) / dz)


def run():
    highs, lows, close, t = load_1d()
    trend  = logtrend(close, t)
    ratio  = close / trend

    piv_idx, prices = find_pivots(highs, lows, T_ZZ)
    rat_piv = ratio[piv_idx]   # ratio в точках пивотов

    print(f"SBER 1d  T={T_ZZ*100:.0f}%  n_пивотов={len(prices)}")
    print()

    # (name, X, k, relative_target)
    variants = [
        ("A: price+diff p=3 K=12",  build_X_price(prices,  3), 12, False),
        ("B: diff_only  p=2 K=9",   build_X_diff (prices,  2),  9, True),
        ("B2: diff_only p=3 K=12",  build_X_diff (prices,  3), 12, True),
        ("C: ratio+diff p=3 K=12",  build_X_price(rat_piv, 3), 12, True),
        ("M0: random walk",         None, None, False),
    ]

    records = []
    print(f"  {'Вариант':<26}  {'цель':>10}  {'rMAE':>7}  {'vs A':>7}  n")
    print("  " + "─" * 60)

    r_A = None
    for name, X, k, rel in variants:
        if name.startswith("M0"):
            pr = prices[:-H]
            ac = prices[H:]
            r  = rmae(pr, ac)
            vs = f"{(r / r_A - 1)*100:+.1f}%" if r_A else "—"
            print(f"  {name:<26}  {'price':>10}  {r:.4f}  {vs:>7}  {len(pr)}")
            records.append({"variant": name, "target": "price", "rMAE": r, "n": len(pr)})
            continue
        pr, ac = walk_forward(prices, X, k, relative=rel)
        r  = rmae(pr, ac)
        tgt = "log-ret→price" if rel else "price"
        if r_A is None:
            r_A = r
            vs = "—"
        else:
            vs = f"{(r / r_A - 1)*100:+.1f}%"
        print(f"  {name:<26}  {tgt:>10}  {r:.4f}  {vs:>7}  {len(pr)}")
        records.append({"variant": name, "target": tgt, "rMAE": r, "n": len(pr)})

    df = pd.DataFrame(records)
    df.to_csv(OUT / "embed_compare.csv", index=False)

    # ── График ────────────────────────────────────────────────────────────────
    plot_vars = [r for r in records if not r["variant"].startswith("M0")]
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]
    for i, row in enumerate(plot_vars):
        ax.bar(i, row["rMAE"], color=colors[i % len(colors)], alpha=0.8,
               label=row["variant"])
    r_m0 = next(r["rMAE"] for r in records if r["variant"].startswith("M0"))
    ax.axhline(r_m0, color="gray", lw=1.2, ls="--", label=f"M0 rMAE={r_m0:.4f}")
    ax.set_xticks(range(len(plot_vars)))
    ax.set_xticklabels([r["variant"].split("(")[0].strip() for r in plot_vars],
                       fontsize=8.5)
    ax.set_ylabel("rMAE")
    ax.set_title(f"Сравнение вложений: LWR на зигзаге  |  SBER 1d T=2%, H=1")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT / "embed_compare.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + график: {OUT}")


if __name__ == "__main__":
    run()
