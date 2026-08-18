#!/usr/bin/env python3
"""
RBF-log (Nadaraya-Watson в log-return пространстве): свип K и p.

Поиск соседей: X=[price, lr1, ..., lr(p-1)], z-score L2  (то же, что LWR)
Цель:         y_j = log(p[j+H] / p[j])                  (log-return)
Веса:         w_j = exp(-0.5·(d_j/ξ)²),  ξ = dist до K-го соседа  (Гауссово ядро)
Прогноз:      lr_hat = Σ w_j·y_j / Σ w_j               (zeroth-order)
Реконструкция: ŷ = p_cur · exp(lr_hat)

Отличие от Simplex-log: K — свободный параметр (не p+1); Гауссово ядро.
Отличие от LWR-log:     degree-0 (нет линейной модели).
Отличие от S-map:        degree-0 (vs degree-1 у S-map).

P_GRID = [3, 8]   (LWR-оптимум и Simplex-оптимум)
K_GRID = [4, 6, 9, 12, 20, 35, 50, 75, 100, 150, 200, 350, 500]

Эталоны:
  LWR-abs       p=3 K=50          → 0.4194
  Simplex-log   p=8 k=9           → 0.4238
  Ens(LWR+Sx)   p_sx=8 α=0.55     → 0.4055
  3-way best    θ=1 α=(0.30,0.50,0.20) → 0.4004

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

P_GRID = [2, 3, 4, 5, 6, 8, 10, 12]
K_GRID = [3, 4, 6, 9, 12, 20, 35, 50, 75, 100, 150, 200, 350, 500]

REF_LWR    = 0.4194
REF_SX_LOG = 0.4238   # Simplex-log p=8
REF_ENS_SX = 0.4055
REF_3WAY   = 0.4004


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
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def rmae(preds, actuals):
    preds = np.asarray(preds, dtype=float)
    valid = ~np.isnan(preds)
    if valid.sum() < 5:
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return float(np.mean(np.abs(preds[valid] - actuals[valid])) / dz) if dz > 1e-12 else np.nan


def run_p(p, p1d, p10m, dt1d, dt10m):
    X1d  = build_X(p1d,  p)
    X10m = build_X(p10m, p)
    n1d  = len(p1d); na = len(p10m)
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")

    buf = {K: [] for K in K_GRID}
    actuals = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X1d[step])):
            continue

        ce = int(ce10m_all[step])
        p_cur = float(p1d[step])

        j1d = np.arange(p - 1, step)
        v1d = ~np.any(np.isnan(X1d[j1d]), axis=1) & (j1d + H < n1d)
        idx1 = j1d[v1d]
        Xp     = list(X1d[idx1])
        yp_abs = list(p1d[idx1 + H])
        yp_src = list(p1d[idx1])

        ce = int(ce10m_all[step])
        j10 = np.arange(p - 1, min(ce, na - H))
        if len(j10):
            v10  = ~np.any(np.isnan(X10m[j10]), axis=1)
            idx10 = j10[v10]
            Xp.extend(X10m[idx10])
            yp_abs.extend(p10m[idx10 + H])
            yp_src.extend(p10m[idx10])

        if len(Xp) < 2:
            for K in K_GRID:
                buf[K].append(np.nan)
            actuals.append(float(p1d[step + H]))
            continue

        X_pool = np.array(Xp)
        y_abs  = np.array(yp_abs, dtype=float)
        y_src  = np.array(yp_src, dtype=float)
        y_lr   = np.log(y_abs / y_src)

        x_q = X1d[step]
        mu  = X_pool.mean(0)
        sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn  = (X_pool - mu) / sig
        xn  = (x_q    - mu) / sig

        dists = np.linalg.norm(Xn - xn, axis=1)
        order = np.argsort(dists)
        N     = len(y_lr)

        for K in K_GRID:
            k = min(K, N)
            knn = order[:k]
            xi  = dists[order[k - 1]]
            if xi < 1e-12:
                lr_hat = float(y_lr[knn].mean())
            else:
                w      = np.exp(-0.5 * (dists[knn] / xi) ** 2)
                lr_hat = float((w @ y_lr[knn]) / w.sum())
            buf[K].append(p_cur * np.exp(lr_hat))

        actuals.append(float(p1d[step + H]))

    acts = np.array(actuals)
    return {K: rmae(buf[K], acts) for K in K_GRID}, len(acts)


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m = find_pivots(h10m, l10m, d10m, T_10M)

    print(f"SBER 1d {len(p1d)} пив  10m {len(p10m)} пив")
    print(f"H={H}  P_GRID={P_GRID}  K_GRID={K_GRID}")
    print(f"Эталоны: LWR={REF_LWR:.4f}  Sx-log(p=8)={REF_SX_LOG:.4f}  "
          f"Ens(+Sx)={REF_ENS_SX:.4f}  3-way={REF_3WAY:.4f}\n")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=False)

    summary = []
    for ax, p in zip(axes.flat, P_GRID):
        r_map, n = run_p(p, p1d, p10m, dt1d, dt10m)
        rs = [r_map[K] for K in K_GRID]

        best_K = K_GRID[int(np.nanargmin(rs))]
        best_r = r_map[best_K]
        summary.append((p, best_K, best_r))

        print(f"p={p}  n={n}")
        print(f"  {'K':>5}  {'RBF-log':>8}  {'vs_LWR':>7}  {'vs_SxLog':>9}")
        for K, r in zip(K_GRID, rs):
            mark = " ←" if K == best_K else ""
            vs_lwr = f"{(r/REF_LWR-1)*100:+.1f}%" if not np.isnan(r) else "  nan"
            vs_sx  = f"{(r/REF_SX_LOG-1)*100:+.1f}%" if not np.isnan(r) else "  nan"
            print(f"  {K:>5}  {r:>8.4f}  {vs_lwr:>7}  {vs_sx:>9}{mark}")
        print(f"  → лучший K={best_K}  rMAE={best_r:.4f}\n")

        ax.plot(K_GRID, rs, "o-", color="darkorange", lw=1.8, label=f"RBF-log p={p}")
        ax.axhline(REF_LWR,    color="steelblue", ls=":",  lw=1.2,
                   label=f"LWR-abs p=3={REF_LWR:.4f}")
        ax.axhline(REF_SX_LOG, color="tomato",    ls="--", lw=1.2,
                   label=f"Sx-log p=8={REF_SX_LOG:.4f}")
        ax.axhline(REF_ENS_SX, color="green",     ls="--", lw=1.2,
                   label=f"Ens(LWR+Sx)={REF_ENS_SX:.4f}")
        ax.axhline(REF_3WAY,   color="purple",    ls="-",  lw=1.0, alpha=0.6,
                   label=f"3-way={REF_3WAY:.4f}")
        ax.scatter([best_K], [best_r], color="darkorange", zorder=5, s=70)
        ax.annotate(f"K={best_K}\n{best_r:.4f}",
                    (best_K, best_r), textcoords="offset points",
                    xytext=(8, 6), fontsize=8.5, color="darkorange")
        # Отметить k=p+1 (Simplex-эквивалент)
        k_sx = p + 1
        if k_sx in K_GRID:
            r_sx_eq = r_map[k_sx]
            ax.scatter([k_sx], [r_sx_eq], color="tomato", zorder=4, s=50, marker="^",
                       label=f"k=p+1={k_sx} (Simplex-eq): {r_sx_eq:.4f}")
        ax.set_xscale("log")
        ax.set_xlabel("K (log-шкала)")
        ax.set_ylabel("rMAE")
        ax.set_title(f"RBF-log p={p}  (признаки: [price, lr1..lr{p-1}])")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.3)

    print("=" * 45)
    print(f"{'p':>3}  {'best K':>7}  {'rMAE':>7}  {'vs LWR':>8}  {'vs SxLog':>9}")
    print("-" * 45)
    for p, bk, br in summary:
        print(f"{p:>3}  {bk:>7}  {br:>7.4f}  "
              f"{(br/REF_LWR-1)*100:>+7.1f}%  {(br/REF_SX_LOG-1)*100:>+8.1f}%")

    fig.suptitle(
        f"RBF-log (NW zeroth-order, log-return цель, Гауссово ядро)\n"
        f"SBER 1d(4%)+10m(0.4%)  H={H}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "rbf_log.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {OUT}/rbf_log.png")


if __name__ == "__main__":
    run()
