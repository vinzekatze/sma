#!/usr/bin/env python3
"""
LWR: расширенный свип K (6..5000) vs S-map θ=8.
SBER 1d(4%) + 10m(0.4%), p=3, H=1.
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

T_1D        = 0.04
T_10M       = 0.004
P           = 3
H           = 1
MIN_HISTORY = 50
THETA_SMAP  = 8.0

K_GRID = [6, 9, 12, 15, 18, 24, 30, 50, 75, 100,
          150, 200, 300, 500, 1000, 2000, 5000]


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    pivots, direction = [], 0
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
                pivots.append((ext_val, dates[ext_idx]))
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                pivots.append((ext_val, dates[ext_idx]))
                direction = 1;  ext_val, ext_idx = highs[i], i
    if not pivots:
        return np.array([]), np.array([])
    return (np.array([v for v, _ in pivots]),
            np.array([d for _, d in pivots]))


def build_X(prices, p):
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def rmae(preds, actuals):
    dz = np.mean(np.abs(np.diff(actuals)))
    return np.nan if dz < 1e-12 else float(np.mean(np.abs(preds - actuals)) / dz)


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dates_1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dates_10m = find_pivots(h10m, l10m, d10m, T_10M)
    X_1d  = build_X(p1d,  P)
    X_10m = build_X(p10m, P)
    n1d   = len(p1d)
    ce10m_all = np.searchsorted(dates_10m, dates_1d, side="left")

    print(f"1d:{len(p1d)}пив  10m:{len(p10m)}пив  p={P} H={H}")
    print(f"K_GRID: {K_GRID}")
    print(f"S-map θ={THETA_SMAP} (эталон: rMAE=0.4322)\n")

    K_MAX = max(K_GRID)
    # Накапливаем предсказания: для каждого K + S-map
    preds_lwr  = {k: [] for k in K_GRID}
    preds_smap = []
    actuals_all = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X_1d[step])):
            continue

        # Собрать пул
        j1d = np.arange(P - 1, step)
        v1d = ~np.any(np.isnan(X_1d[j1d]), axis=1) & (j1d + H < n1d)
        Xp  = list(X_1d[j1d[v1d]]);  yp = list(p1d[j1d[v1d] + H])

        ce = int(ce10m_all[step]); na = len(p10m)
        j10 = np.arange(P - 1, min(ce, na - H))
        if len(j10):
            v10 = ~np.any(np.isnan(X_10m[j10]), axis=1)
            Xp.extend(X_10m[j10[v10]]);  yp.extend(p10m[j10[v10] + H])

        N = len(Xp)
        if N < P + 2:
            continue

        X_pool = np.array(Xp);  y_pool = np.array(yp)
        x_q    = X_1d[step]

        # Z-score пула
        mu    = X_pool.mean(0)
        sigma = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn    = (X_pool - mu) / sigma
        xn    = (x_q    - mu) / sigma

        # Все расстояния (нужны для K_MAX и S-map)
        dists = np.linalg.norm(Xn - xn, axis=1)
        order = np.argsort(dists)

        # ── LWR по всем K ────────────────────────────────────────────────────
        for k in K_GRID:
            k_eff = min(k, N)
            knn   = order[:k_eff]
            xi    = dists[order[k_eff - 1]]
            if xi < 1e-12:
                pred = float(y_pool[knn].mean())
            else:
                w   = np.exp(-0.5 * (dists[knn] / xi) ** 2)
                ws  = np.sqrt(w)
                A   = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
                b   = y_pool[knn] * ws
                c, *_ = np.linalg.lstsq(A, b, rcond=None)
                pred = float(c[0] + c[1:] @ xn)
            preds_lwr[k].append(pred)

        # ── S-map θ=8 ────────────────────────────────────────────────────────
        mean_d = dists.mean()
        w_sm   = np.exp(-THETA_SMAP * dists / (mean_d + 1e-12))
        ws_sm  = np.sqrt(w_sm)
        A_sm   = np.column_stack([np.ones(N), Xn]) * ws_sm[:, None]
        b_sm   = y_pool * ws_sm
        c_sm, *_ = np.linalg.lstsq(A_sm, b_sm, rcond=None)
        preds_smap.append(float(c_sm[0] + c_sm[1:] @ xn))

        actuals_all.append(float(p1d[step + H]))

    actuals   = np.array(actuals_all)
    r_smap    = rmae(np.array(preds_smap), actuals)
    records   = []

    print(f"{'K':>6}  {'rMAE':>8}  {'vs S-map':>10}  {'vs K=12':>9}")
    print("─" * 44)

    r_k12 = rmae(np.array(preds_lwr[12]), actuals)
    r_best = np.inf;  k_best = None

    for k in K_GRID:
        r = rmae(np.array(preds_lwr[k]), actuals)
        vs_sm = f"{(r/r_smap-1)*100:+.1f}%"
        vs_12 = f"{(r/r_k12 -1)*100:+.1f}%"
        mk = ""
        if r < r_best:
            r_best = r; k_best = k; mk = " ←"
        print(f"  K={k:4d}  {r:.4f}   {vs_sm:>10}  {vs_12:>9}{mk}")
        records.append({"K": k, "rMAE_lwr": r,
                        "vs_smap_pct": (r/r_smap-1)*100,
                        "vs_k12_pct" : (r/r_k12 -1)*100})

    print(f"\n  S-map θ={THETA_SMAP}  {r_smap:.4f}   (эталон)")
    print(f"  Лучший K={k_best}  rMAE={r_best:.4f}  "
          f"vs S-map: {(r_best/r_smap-1)*100:+.1f}%")

    df = pd.DataFrame(records)
    df.to_csv(OUT / "lwr_k_extended.csv", index=False)

    # ── График ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))
    ks = df["K"].values
    ax.plot(ks, df["rMAE_lwr"], "o-", color="steelblue", lw=2.5, ms=8, label="LWR")
    ax.axhline(r_smap, color="crimson", lw=2, ls="--",
               label=f"S-map θ={THETA_SMAP}  rMAE={r_smap:.4f}")
    ax.axvline(k_best, color="steelblue", lw=1, ls=":", alpha=0.5)
    ax.annotate(f"K={k_best}\n{r_best:.4f}",
                xy=(k_best, r_best), xytext=(k_best*1.3, r_best+0.02),
                fontsize=9, color="steelblue",
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=1))
    ax.set_xscale("log")
    ax.set_xlabel("K (число соседей, log-шкала)")
    ax.set_ylabel("rMAE")
    ax.set_title(
        f"LWR: расширенный свип K  vs  S-map θ={THETA_SMAP}\n"
        f"SBER 1d({T_1D*100:.0f}%) + 10m({T_10M*100:.1f}%)  p={P}  H={H}"
    )
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=7.5, rotation=45)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT / "lwr_k_extended.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"CSV + график → {OUT}")


if __name__ == "__main__":
    run()
