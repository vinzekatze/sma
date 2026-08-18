#!/usr/bin/env python3
"""
S-map на зигзаге SBER: свип по θ.

S-map (Sugihara 1994):
  w_i = exp(−θ · dᵢ / mean_d)
  где dᵢ = ‖x_i − x_q‖  (z-score пула),  mean_d = среднее по ВСЕМУ пулу.
  θ=0 → глобальный OLS (все точки с равными весами).
  θ→∞ → только ближайший сосед.
  Регрессия: WLS  ŷ = c₀ + c₁ᵀ·x_q (линейная, как LWR).

Пул: SBER 1d (T=4%) + 10m (T=0.4%), p=3, H=1.
Эффективная ширина полосы S-map: query-adaptive (пересчитывается для каждого шага).

Сравнение: LWR K=12 (оптимум, rMAE=0.4548) + LWR K=30 (расширенная полоса).

КАУЗАЛЬНОСТЬ: пул строго j < step; 10m — до даты step.
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

THETA_GRID  = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
K_LWR       = 12    # LWR reference


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    h     = np.array([d["high"]  for d in raw], dtype=np.float64)
    l_    = np.array([d["low"]   for d in raw], dtype=np.float64)
    dates = np.array([d["begin"] for d in raw])
    return h, l_, dates


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
    return (np.array([pv for pv, _ in pivots]),
            np.array([d  for _, d  in pivots]))


def build_X(prices, p):
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def rmae(preds, actuals):
    if len(preds) < 5:
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return np.nan if dz < 1e-12 else float(np.mean(np.abs(preds - actuals)) / dz)


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")

    p1d,  dates_1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dates_10m = find_pivots(h10m, l10m, d10m, T_10M)
    X_1d  = build_X(p1d,  P)
    X_10m = build_X(p10m, P)

    n1d  = len(p1d)
    ce10m_all = np.searchsorted(dates_10m, dates_1d, side="left")

    print(f"SBER  1d  T={T_1D *100:.0f}%  → {n1d} пивотов")
    print(f"SBER  10m T={T_10M*100:.1f}% → {len(p10m)} пивотов")
    print(f"p={P}  H={H}  θ_grid={THETA_GRID}")
    print()

    # ─── Walk-forward: один проход, все θ за шаг ─────────────────────────────
    # Накапливаем (pred, actual) для каждого θ + LWR K=12
    n_theta    = len(THETA_GRID)
    preds_th   = [[] for _ in range(n_theta)]   # S-map по θ
    preds_lwr  = []                               # LWR K=12
    actuals_all = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X_1d[step])):
            continue
        actual   = float(p1d[step + H])
        x_q      = X_1d[step]
        cur_date = dates_1d[step]

        # Собрать объединённый пул
        j1d = np.arange(P - 1, step)
        v1d = ~np.any(np.isnan(X_1d[j1d]), axis=1) & (j1d + H < n1d)
        Xp  = list(X_1d[j1d[v1d]])
        yp  = list(p1d [j1d[v1d] + H])

        ce  = int(ce10m_all[step])
        na  = len(p10m)
        j10 = np.arange(P - 1, min(ce, na - H))
        if len(j10):
            v10 = ~np.any(np.isnan(X_10m[j10]), axis=1)
            Xp.extend(X_10m[j10[v10]])
            yp.extend(p10m [j10[v10] + H])

        if len(Xp) < P + 2:  # нужно хотя бы p+2 точек для регрессии
            continue

        X_pool = np.array(Xp);  y_pool = np.array(yp)
        N      = len(X_pool)

        # Z-score (по пулу)
        mu    = X_pool.mean(0)
        sigma = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn    = (X_pool - mu) / sigma
        xn    = (x_q    - mu) / sigma

        # Расстояния от запроса до всего пула
        dists  = np.linalg.norm(Xn - xn, axis=1)
        mean_d = dists.mean()

        # ── S-map по всем θ ──────────────────────────────────────────────────
        skip_step = False
        for ti, theta in enumerate(THETA_GRID):
            if mean_d < 1e-12:
                w = np.ones(N)
            else:
                w = np.exp(-theta * dists / mean_d)

            ws  = np.sqrt(w)
            A   = np.column_stack([np.ones(N), Xn]) * ws[:, None]
            b   = y_pool * ws
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            pred = float(coef[0] + coef[1:] @ xn)
            preds_th[ti].append(pred)

        # ── LWR K=12 ─────────────────────────────────────────────────────────
        order = np.argsort(dists)
        knn   = order[:K_LWR]
        xi    = dists[order[K_LWR - 1]]
        if xi < 1e-12:
            pred_lwr = float(y_pool[knn].mean())
        else:
            wl  = np.exp(-0.5 * (dists[knn] / xi) ** 2)
            wls = np.sqrt(wl)
            Al  = np.column_stack([np.ones(K_LWR), Xn[knn]]) * wls[:, None]
            bl  = y_pool[knn] * wls
            cl, *_ = np.linalg.lstsq(Al, bl, rcond=None)
            pred_lwr = float(cl[0] + cl[1:] @ xn)
        preds_lwr.append(pred_lwr)

        actuals_all.append(actual)

    actuals = np.array(actuals_all)
    print(f"Шагов: {len(actuals)}\n")

    r_lwr  = rmae(np.array(preds_lwr), actuals)
    records = []

    print(f"{'θ':>6}  {'rMAE':>8}  {'vs θ=0':>8}  {'vs LWR':>8}")
    print("─" * 42)

    r_th0 = None
    for ti, theta in enumerate(THETA_GRID):
        r = rmae(np.array(preds_th[ti]), actuals)
        if r_th0 is None:
            r_th0 = r
            vs0 = "—"
        else:
            vs0 = f"{(r/r_th0-1)*100:+.1f}%"
        vs_lwr = f"{(r/r_lwr-1)*100:+.1f}%"
        mk = " ← min" if (ti > 0 and not np.isnan(r)
                          and r < min(rmae(np.array(preds_th[j]), actuals)
                                      for j in range(ti) if preds_th[j])) else ""
        print(f"  θ={theta:5.2f}  {r:.4f}   {vs0:>8}  {vs_lwr:>8}{mk}")
        records.append({"theta": theta, "rMAE": r, "vs_th0_pct": 0.0 if r_th0 is None else (r/r_th0-1)*100,
                        "vs_lwr_pct": (r/r_lwr-1)*100})

    print(f"\n  LWR K={K_LWR}       {r_lwr:.4f}   (эталон)")

    df = pd.DataFrame(records)
    df.to_csv(OUT / "smap_theta_sweep.csv", index=False)

    # ── График ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    thetas = df["theta"].values
    rmae_v = df["rMAE"].values

    ax = axes[0]
    ax.plot(thetas, rmae_v, "o-", color="steelblue", lw=2.5, ms=9, label="S-map")
    ax.axhline(r_lwr, color="crimson", lw=1.5, ls="--",
               label=f"LWR K={K_LWR}  rMAE={r_lwr:.4f}")
    ax.axhline(r_th0, color="gray", lw=1, ls=":",
               label=f"θ=0 (OLS)  rMAE={r_th0:.4f}")
    for t_, r_ in zip(thetas, rmae_v):
        ax.annotate(f"{r_:.4f}", (t_, r_), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7.5)
    ax.set_xlabel("θ")
    ax.set_ylabel("rMAE")
    ax.set_title("S-map: rMAE vs θ")
    ax.set_xscale("symlog", linthresh=0.3)
    ax.set_xticks(thetas)
    ax.set_xticklabels([str(t) for t in thetas], fontsize=8)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.2)

    ax = axes[1]
    vs_lwr = df["vs_lwr_pct"].values
    clrs = ["seagreen" if v <= 0 else "salmon" for v in vs_lwr]
    ax.bar(range(len(thetas)), vs_lwr, color=clrs, alpha=0.85)
    ax.axhline(0, color="crimson", lw=1.5, ls="--", label=f"LWR K={K_LWR}")
    for i, (t_, v) in enumerate(zip(thetas, vs_lwr)):
        ax.annotate(f"{v:+.1f}%", (i, v),
                    textcoords="offset points",
                    xytext=(0, 5 if v >= 0 else -12),
                    ha="center", fontsize=8)
    ax.set_xticks(range(len(thetas)))
    ax.set_xticklabels([str(t) for t in thetas], fontsize=8)
    ax.set_xlabel("θ")
    ax.set_ylabel("S-map vs LWR, % rMAE")
    ax.set_title("S-map vs LWR (отриц. = S-map лучше)")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"S-map (w=exp(−θ·d/mean_d)) vs LWR  |  SBER 1d({T_1D*100:.0f}%) + 10m({T_10M*100:.1f}%)\n"
        f"X=[price, lr₁, lr₂]  p={P}  H={H}  пул = ВСЕ исторические точки",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(OUT / "smap_theta_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + график → {OUT}")


if __name__ == "__main__":
    run()
