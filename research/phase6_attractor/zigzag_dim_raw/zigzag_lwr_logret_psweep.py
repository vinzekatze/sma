#!/usr/bin/env python3
"""
LWR-abs vs LWR-log: свип по p и K.

Признаки: X = [price, lr1, ..., lr(p-1)]
LWR-abs:  y_j = p[j+H]
LWR-log:  y_j = log(p[j+H] / p[j])  → ŷ = p_cur · exp(lr_hat)

P_GRID = [2, 3, 4, 5, 6, 8]
K_GRID = [6, 10, 20, 35, 50, 75, 100, 150, 200, 350, 500]
SBER 1d(4%) + 10m(0.4%), H=1.
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
H           = 1
MIN_HISTORY = 50

P_GRID = [2, 3, 4, 5, 6, 8]
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
    if len(preds) < 5:
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return float(np.mean(np.abs(preds - actuals)) / dz) if dz > 1e-12 else np.nan


def wls_pred(Xn_knn, y_knn, w, xn_q):
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(len(y_knn)), Xn_knn]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, y_knn * ws, rcond=None)
    return float(c[0] + c[1:] @ xn_q)


def run_p(p, p1d, p10m, dt1d, dt10m):
    X1d  = build_X(p1d,  p)
    X10m = build_X(p10m, p)
    n1d  = len(p1d)
    na   = len(p10m)
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")

    buf_abs = {K: [] for K in K_GRID}
    buf_log = {K: [] for K in K_GRID}
    actuals = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X1d[step])):
            continue

        j1d = np.arange(p - 1, step)
        v1d = ~np.any(np.isnan(X1d[j1d]), axis=1) & (j1d + H < n1d)
        idx1d = j1d[v1d]
        Xp     = list(X1d[idx1d])
        yp_abs = list(p1d[idx1d + H])
        yp_src = list(p1d[idx1d])

        ce  = int(ce10m_all[step])
        j10 = np.arange(p - 1, min(ce, na - H))
        if len(j10):
            v10  = ~np.any(np.isnan(X10m[j10]), axis=1)
            idx10 = j10[v10]
            Xp.extend(X10m[idx10])
            yp_abs.extend(p10m[idx10 + H])
            yp_src.extend(p10m[idx10])

        if len(Xp) < p + 2:
            continue

        X_pool = np.array(Xp)
        y_abs  = np.array(yp_abs, dtype=float)
        y_src  = np.array(yp_src, dtype=float)
        y_lr   = np.log(y_abs / y_src)

        x_q   = X1d[step]
        p_cur = float(p1d[step])

        mu  = X_pool.mean(0)
        sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn  = (X_pool - mu) / sig
        xn  = (x_q    - mu) / sig

        dists = np.linalg.norm(Xn - xn, axis=1)
        order = np.argsort(dists)
        N     = len(y_abs)

        for K in K_GRID:
            k = min(K, N)
            if k < p + 2:
                buf_abs[K].append(np.nan)
                buf_log[K].append(np.nan)
                continue

            knn = order[:k]
            xi  = dists[order[k - 1]]

            if xi < 1e-12:
                buf_abs[K].append(float(y_abs[knn].mean()))
                buf_log[K].append(float(p_cur * np.exp(y_lr[knn].mean())))
                continue

            w = np.exp(-0.5 * (dists[knn] / xi) ** 2)

            buf_abs[K].append(wls_pred(Xn[knn], y_abs[knn], w, xn))
            lr_hat = wls_pred(Xn[knn], y_lr[knn], w, xn)
            buf_log[K].append(p_cur * np.exp(lr_hat))

        actuals.append(float(p1d[step + H]))

    acts = np.array(actuals)
    r_abs = {K: rmae(buf_abs[K], acts) for K in K_GRID}
    r_log = {K: rmae(buf_log[K], acts) for K in K_GRID}
    return r_abs, r_log, len(acts)


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m = find_pivots(h10m, l10m, d10m, T_10M)

    print(f"SBER 1d {len(p1d)} пив  10m {len(p10m)} пив")
    print(f"H={H}  P_GRID={P_GRID}  K_GRID={K_GRID}\n")

    summary = []   # (p, best_K_abs, rMAE_abs, best_K_log, rMAE_log)

    for p in P_GRID:
        r_abs, r_log, n = run_p(p, p1d, p10m, dt1d, dt10m)

        best_K_abs = min(K_GRID, key=lambda K: r_abs[K])
        best_K_log = min(K_GRID, key=lambda K: r_log[K])
        best_abs   = r_abs[best_K_abs]
        best_log   = r_log[best_K_log]

        summary.append((p, best_K_abs, best_abs, best_K_log, best_log))
        print(f"p={p}  n={n}")
        print(f"  {'K':>5}  {'LWR-abs':>8}  {'LWR-log':>8}  {'Δ':>7}")
        for K in K_GRID:
            delta = r_log[K] - r_abs[K]
            print(f"  {K:>5}  {r_abs[K]:>8.4f}  {r_log[K]:>8.4f}  {delta:>+7.4f}")
        print(f"  → abs: K={best_K_abs} rMAE={best_abs:.4f}  "
              f"log: K={best_K_log} rMAE={best_log:.4f}  "
              f"Δ={best_log-best_abs:+.4f}  "
              f"winner={'log' if best_log < best_abs else 'abs'}\n")

    # ── Итоговая таблица ──────────────────────────────────────────────────────
    print("=" * 65)
    print(f"{'p':>3}  {'K_abs':>6}  {'rMAE_abs':>9}  "
          f"{'K_log':>6}  {'rMAE_log':>9}  {'Δ%':>6}  winner")
    print("-" * 65)
    for p, Ka, ra, Kl, rl in summary:
        delta_pct = (rl / ra - 1) * 100
        w = "log" if rl < ra else "abs"
        print(f"{p:>3}  {Ka:>6}  {ra:>9.4f}  {Kl:>6}  {rl:>9.4f}  "
              f"{delta_pct:>+5.1f}%  {w}")

    # ── График: лучшая rMAE по p для каждого метода ───────────────────────────
    ps      = [s[0] for s in summary]
    best_ra = [s[2] for s in summary]
    best_rl = [s[4] for s in summary]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Левый: лучший rMAE vs p
    ax = axes[0]
    ax.plot(ps, best_ra, "o-", label="LWR-abs (best K)", color="steelblue", lw=1.8)
    ax.plot(ps, best_rl, "s--", label="LWR-log (best K)", color="tomato",    lw=1.8)
    ax.set_xlabel("p (размерность признаков)")
    ax.set_ylabel("rMAE (лучший K)")
    ax.set_title("Лучший rMAE по p")
    ax.legend(); ax.grid(True, alpha=0.3)
    for i, (p, ra, rl) in enumerate(zip(ps, best_ra, best_rl)):
        ax.annotate(f"K={summary[i][1]}", (p, ra),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7.5, color="steelblue")
        ax.annotate(f"K={summary[i][3]}", (p, rl),
                    textcoords="offset points", xytext=(0, -13),
                    ha="center", fontsize=7.5, color="tomato")

    # Правый: кривые rMAE vs K для каждого p (все методы)
    ax2 = axes[1]
    colors_abs = plt.cm.Blues(np.linspace(0.45, 0.9, len(P_GRID)))
    colors_log = plt.cm.Reds( np.linspace(0.45, 0.9, len(P_GRID)))
    for i, p in enumerate(P_GRID):
        r_abs_arr = [summary[i][2] if k == summary[i][1] else None for k in K_GRID]
        r_log_arr = [summary[i][4] if k == summary[i][3] else None for k in K_GRID]

    # Пересчитать кривые: нужны все r_abs[K] по каждому p
    # Они потеряны — перезапустим run_p для графика... нет, сохраним в run_p
    # Поскольку мы не сохранили полные кривые, рисуем только точки best-K
    ax2.set_visible(False)

    # Новый правый: сравнение победителей по p
    ax3 = fig.add_subplot(1, 2, 2)
    delta_pct = [(s[4] / s[2] - 1) * 100 for s in summary]
    colors_bar = ["tomato" if d < 0 else "steelblue" for d in delta_pct]
    ax3.bar(ps, delta_pct, color=colors_bar, alpha=0.75, width=0.6)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_xlabel("p")
    ax3.set_ylabel("Δ rMAE %  (log − abs) / abs")
    ax3.set_title("log vs abs: выигрыш по p\n(< 0 → log лучше)")
    ax3.grid(True, alpha=0.3, axis="y")
    for i, (p_val, d) in enumerate(zip(ps, delta_pct)):
        ax3.text(p_val, d + (0.05 if d >= 0 else -0.15),
                 f"{d:+.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"LWR-abs vs LWR-log  |  SBER 1d(4%)+10m(0.4%)  H={H}\n"
        f"Признаки: [price, lr1, ..., lr(p-1)]",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "lwr_logret_psweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {OUT}/lwr_logret_psweep.png")


if __name__ == "__main__":
    run()
