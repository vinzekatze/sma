#!/usr/bin/env python3
"""
Свип T_1h при фиксированном пуле 1d(4%) + 10m(0.4%).
Ищем: добавляет ли 1h что-то поверх уже хорошей базы 1d+10m.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT      = BASE_DIR / "results"

P          = 3
K          = 3 * (P + 1)   # 12
H          = 1
MIN_HISTORY = 50
T_PRIM     = 0.04
T_10M      = 0.004

T_1H_GRID  = [0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.015,
              0.018, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050]


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
    return (np.array([p for _, p, _ in pivots]),
            np.array([d for _, _, d in pivots]))


def build_X(prices, p):
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


def walk_forward(prices_prim, dates_prim, X_prim, aux_list, p, k, h, min_history):
    n = len(prices_prim)
    preds, actuals = [], []
    for step in range(max(min_history, p), n - h):
        if np.any(np.isnan(X_prim[step])):
            continue
        cur_date = dates_prim[step]
        j_arr = np.arange(p - 1, step)
        valid = ~np.any(np.isnan(X_prim[j_arr]), axis=1) & (j_arr + h < n)
        X_rows = list(X_prim[j_arr[valid]])
        y_rows = list(prices_prim[j_arr[valid] + h])
        for prices_a, dates_a, X_a in aux_list:
            ce = int(np.searchsorted(dates_a, cur_date, side="left"))
            if ce < p:
                continue
            na = len(prices_a)
            j_a = np.arange(p - 1, min(ce, na - h))
            if len(j_a) == 0:
                continue
            valid_a = ~np.any(np.isnan(X_a[j_a]), axis=1) & (j_a + h < na)
            X_rows.extend(X_a[j_a[valid_a]])
            y_rows.extend(prices_a[j_a[valid_a] + h])
        if len(X_rows) < k:
            continue
        pred = lwr_pred(np.array(X_rows), np.array(y_rows), X_prim[step], k)
        if not np.isnan(pred):
            preds.append(pred)
            actuals.append(prices_prim[step + h])
    return np.array(preds), np.array(actuals)


def rmae(preds, actuals):
    if len(preds) < 5:
        return np.nan
    mean_dz = np.mean(np.abs(np.diff(actuals)))
    return float(np.mean(np.abs(preds - actuals)) / mean_dz) if mean_dz > 1e-12 else np.nan


def run():
    h1d, l1d, d1d     = load_tf("1d")
    h1h, l1h, d1h     = load_tf("1h")
    h10m, l10m, d10m  = load_tf("10m")

    prices_prim, dates_prim = find_pivots_hl(h1d, l1d, d1d, T_PRIM)
    X_prim = build_X(prices_prim, P)

    prices_10m, dates_10m = find_pivots_hl(h10m, l10m, d10m, T_10M)
    X_10m = build_X(prices_10m, P)
    aux_base = [(prices_10m, dates_10m, X_10m)]

    # Базовые линии
    preds0, acts0 = walk_forward(prices_prim, dates_prim, X_prim,
                                 [], P, K, H, MIN_HISTORY)
    r_1d_only = rmae(preds0, acts0)

    preds1, acts1 = walk_forward(prices_prim, dates_prim, X_prim,
                                 aux_base, P, K, H, MIN_HISTORY)
    r_1d_10m = rmae(preds1, acts1)

    print(f"SBER 1d T={T_PRIM*100:.0f}%: {len(prices_prim)} пивотов")
    print(f"10m  T={T_10M*100:.1f}%: {len(prices_10m)} пивотов")
    print(f"\n  только 1d:      rMAE={r_1d_only:.4f}")
    print(f"  1d + 10m:       rMAE={r_1d_10m:.4f}  ({(r_1d_10m/r_1d_only-1)*100:+.1f}% vs 1d)")
    print(f"\n  Свип T_1h поверх 1d+10m:")
    print(f"  {'T_1h':>6}  {'n_1h':>7}  {'rMAE':>7}  {'vs 1d+10m':>10}  {'vs 1d':>8}")
    print("  " + "─" * 48)

    records = []
    best_T, best_r = None, r_1d_10m

    for T_1h in T_1H_GRID:
        prices_1h, dates_1h = find_pivots_hl(h1h, l1h, d1h, T_1h)
        if len(prices_1h) < P + H + 1:
            continue
        X_1h = build_X(prices_1h, P)
        aux = aux_base + [(prices_1h, dates_1h, X_1h)]
        preds, acts = walk_forward(prices_prim, dates_prim, X_prim,
                                   aux, P, K, H, MIN_HISTORY)
        r = rmae(preds, acts)
        vs_10m = (r / r_1d_10m - 1) * 100
        vs_1d  = (r / r_1d_only - 1) * 100
        marker = " ←" if r < best_r else ""
        print(f"  T_1h={T_1h*100:4.1f}%  {len(prices_1h):7d}  {r:.4f}  {vs_10m:+9.1f}%  {vs_1d:+7.1f}%{marker}")
        if r < best_r:
            best_r, best_T = r, T_1h
        records.append({"T_1h": T_1h, "n_1h": len(prices_1h),
                         "rMAE": r, "vs_1d_10m": vs_10m, "vs_1d": vs_1d})

    df = pd.DataFrame(records)
    df.to_csv(OUT / "1h_on_top_sweep.csv", index=False)

    if best_T:
        print(f"\n  Лучший T_1h={best_T*100:.1f}%  rMAE={best_r:.4f}"
              f"  vs 1d+10m={( best_r/r_1d_10m-1)*100:+.1f}%"
              f"  vs 1d={( best_r/r_1d_only-1)*100:+.1f}%")
    else:
        print(f"\n  1h не улучшает 1d+10m")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["T_1h"] * 100, df["rMAE"], "o-", color="steelblue", lw=2, ms=7, label="1d+10m+1h")
    ax.axhline(r_1d_10m, color="darkorange", lw=1.5, ls="--", label=f"1d+10m ({r_1d_10m:.4f})")
    ax.axhline(r_1d_only, color="gray",      lw=1.2, ls=":",  label=f"только 1d ({r_1d_only:.4f})")
    ax.set_xlabel("T_1h, %")
    ax.set_ylabel("rMAE")
    ax.set_title(f"Добавление 1h поверх 1d(T={T_PRIM*100:.0f}%)+10m(T={T_10M*100:.1f}%)\n"
                 f"SBER, LWR p={P} k={K}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "1h_on_top_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  График: {OUT / '1h_on_top_sweep.png'}")


if __name__ == "__main__":
    run()
