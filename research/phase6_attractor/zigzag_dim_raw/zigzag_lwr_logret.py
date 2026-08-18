#!/usr/bin/env python3
"""
LWR-abs vs LWR-log: сравнение целевого пространства.

Признаки: X = [price, lr1, lr2]  (p=3, идентичны)
LWR-abs:  y_j = p[j+H]                  → ŷ
LWR-log:  y_j = log(p[j+H] / p[j])      → ŷ = p_cur · exp(lr_hat)

Свип K = [6, 10, 20, 35, 50, 75, 100, 150, 200, 350, 500]
SBER 1d(4%) + 10m(0.4%), p=3, H=1.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT      = BASE_DIR / "results"

T_1D        = 0.04
T_10M       = 0.004
P_PRED      = 3
H           = 1
MIN_HISTORY = 50

K_GRID = [6, 10, 20, 35, 50, 75, 100, 150, 200, 350, 500]


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    vals, dts = [], []
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
                vals.append(ext_val); dts.append(dates[ext_idx])
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                direction = 1; ext_val, ext_idx = highs[i], i
    if not vals:
        return np.array([]), np.array([])
    return np.array(vals), np.array(dts)


def build_X(prices, p):
    """X[i] = [prices[i], lr1, lr2, ..., lr(p-1)]"""
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def rmae(preds, actuals):
    preds = np.asarray(preds, dtype=float)
    if len(preds) < 5 or np.any(np.isnan(preds)):
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return float(np.mean(np.abs(preds - actuals)) / dz) if dz > 1e-12 else np.nan


def wls_predict(X_pool_n, y, x_q_n, knn_idx, w_gauss):
    """WLS через нормализованные признаки. Возвращает скалярный прогноз."""
    ws = np.sqrt(w_gauss)
    A  = np.column_stack([np.ones(len(knn_idx)), X_pool_n[knn_idx]]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, y[knn_idx] * ws, rcond=None)
    return float(c[0] + c[1:] @ x_q_n)


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m = find_pivots(h10m, l10m, d10m, T_10M)

    X1d  = build_X(p1d,  P_PRED)
    X10m = build_X(p10m, P_PRED)

    n1d = len(p1d)
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")

    print(f"SBER 1d {n1d} пив  10m {len(p10m)} пив")
    print(f"P={P_PRED}  H={H}  K_GRID={K_GRID}\n")

    # Буферы: dict K → list прогнозов
    buf_abs = {K: [] for K in K_GRID}
    buf_log = {K: [] for K in K_GRID}
    actuals = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X1d[step])):
            continue

        # ── Собрать пул ────────────────────────────────────────────────────────
        j1d = np.arange(P_PRED - 1, step)
        v1d = ~np.any(np.isnan(X1d[j1d]), axis=1) & (j1d + H < n1d)
        Xp  = list(X1d[j1d[v1d]])
        yp  = list(p1d[j1d[v1d] + H])           # абс. цена
        yp_src = list(p1d[j1d[v1d]])             # текущая цена пула (для лог-цели)

        ce = int(ce10m_all[step]); na = len(p10m)
        j10 = np.arange(P_PRED - 1, min(ce, na - H))
        if len(j10):
            v10 = ~np.any(np.isnan(X10m[j10]), axis=1)
            idx = j10[v10]
            Xp.extend(X10m[idx])
            yp.extend(p10m[idx + H])
            yp_src.extend(p10m[idx])

        if len(Xp) < P_PRED + 2:
            continue

        X_pool  = np.array(Xp)
        y_abs   = np.array(yp, dtype=float)
        y_src   = np.array(yp_src, dtype=float)
        y_lr    = np.log(y_abs / y_src)          # log(p_next / p_cur) для пула

        x_q = X1d[step]
        p_cur = float(p1d[step])

        # Нормализация признаков (одна, общая для обоих методов)
        mu  = X_pool.mean(0)
        sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn  = (X_pool - mu) / sig
        xn  = (x_q    - mu) / sig

        dists = np.linalg.norm(Xn - xn, axis=1)
        order = np.argsort(dists)

        N = len(y_abs)

        for K in K_GRID:
            k_eff = min(K, N)
            if k_eff < P_PRED + 2:
                buf_abs[K].append(np.nan)
                buf_log[K].append(np.nan)
                continue

            knn = order[:k_eff]
            xi  = dists[order[k_eff - 1]]

            if xi < 1e-12:
                buf_abs[K].append(float(y_abs[knn].mean()))
                buf_log[K].append(float(p_cur * np.exp(y_lr[knn].mean())))
                continue

            w = np.exp(-0.5 * (dists[knn] / xi) ** 2)

            # LWR-abs: y = абс. цена
            y_hat_abs = wls_predict(Xn, y_abs, xn, knn, w)
            buf_abs[K].append(y_hat_abs)

            # LWR-log: y = лог-доходность, реконструкция через p_cur
            lr_hat = wls_predict(Xn, y_lr, xn, knn, w)
            buf_log[K].append(p_cur * np.exp(lr_hat))

        actuals.append(float(p1d[step + H]))

    acts = np.array(actuals)
    n    = len(acts)
    print(f"Шагов: {n}\n")

    # ── Таблица rMAE ──────────────────────────────────────────────────────────
    print(f"{'K':>5}  {'LWR-abs':>8}  {'LWR-log':>8}  {'Δ':>7}  {'winner':>8}")
    print("-" * 46)
    r_abs_list = []
    r_log_list = []
    for K in K_GRID:
        ra = rmae(buf_abs[K], acts)
        rl = rmae(buf_log[K], acts)
        r_abs_list.append(ra)
        r_log_list.append(rl)
        delta = rl - ra
        w = "log" if rl < ra else ("abs" if ra < rl else "tie")
        sign = f"{delta:+.4f}"
        print(f"{K:>5}  {ra:>8.4f}  {rl:>8.4f}  {sign:>7}  {w:>8}")

    best_abs = min(r_abs_list)
    best_log = min(r_log_list)
    best_K_abs = K_GRID[np.argmin(r_abs_list)]
    best_K_log = K_GRID[np.argmin(r_log_list)]

    print(f"\nЛучший LWR-abs: K={best_K_abs}  rMAE={best_abs:.4f}")
    print(f"Лучший LWR-log: K={best_K_log}  rMAE={best_log:.4f}")
    print(f"Разница: {(best_log/best_abs - 1)*100:+.2f}%")

    # ── График ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(K_GRID, r_abs_list, "o-", label="LWR-abs (текущий)",
            color="steelblue", linewidth=1.8)
    ax.plot(K_GRID, r_log_list, "s--", label="LWR-log (лог-доходность)",
            color="tomato",    linewidth=1.8)
    ax.axhline(best_abs, color="steelblue", linestyle=":", alpha=0.5,
               label=f"best abs={best_abs:.4f} K={best_K_abs}")
    ax.axhline(best_log, color="tomato",    linestyle=":", alpha=0.5,
               label=f"best log={best_log:.4f} K={best_K_log}")
    ax.set_xscale("log")
    ax.set_xlabel("K (соседи, log-шкала)")
    ax.set_ylabel("rMAE")
    ax.set_title(
        f"LWR-abs vs LWR-log  |  SBER 1d+10m  p={P_PRED}  H={H}  n={n}\n"
        f"Признаки: [price, lr1, lr2]  |  Цель: abs vs log(p_next/p_cur)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "lwr_logret.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {OUT}/lwr_logret.png")


if __name__ == "__main__":
    run()
