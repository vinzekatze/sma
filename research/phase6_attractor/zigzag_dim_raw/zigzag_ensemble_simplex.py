#!/usr/bin/env python3
"""
Ансамбль LWR-abs + Simplex-log: свип α и p_simplex.

LWR-abs:    p=3, K=50  (зафиксированы как лучшие)
Simplex-log: p ∈ P_SX_GRID, k=p+1, цель log(p_next/p_cur)

Ансамбль: ŷ = α·LWR + (1−α)·Simplex-log,  α ∈ [0..1] шаг 0.05

Эталоны:
  LWR-abs  p=3 K=50         → 0.4194
  Simplex-log p=8 k=9       → 0.4238
  Ens(LWR+S-map) α=0.65     → 0.4143  ← текущий лучший

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

P_LWR   = 3
K_LWR   = 50
P_SX_GRID = [3, 4, 6, 8, 10, 12]
A_GRID  = np.round(np.arange(0.0, 1.05, 0.05), 2)

REF_LWR  = 0.4194
REF_SX   = 0.4238
REF_ENS  = 0.4143   # LWR+S-map


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


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m = find_pivots(h10m, l10m, d10m, T_10M)

    n1d = len(p1d); na = len(p10m)
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")

    # Матрицы признаков для всех нужных p
    p_max = max(P_LWR, max(P_SX_GRID))
    X1d_all  = {p: build_X(p1d,  p) for p in set([P_LWR] + P_SX_GRID)}
    X10m_all = {p: build_X(p10m, p) for p in set([P_LWR] + P_SX_GRID)}

    print(f"SBER 1d {n1d} пив  10m {na} пив")
    print(f"LWR: p={P_LWR} K={K_LWR}   Simplex-log: p={P_SX_GRID}")
    print(f"Эталоны: LWR={REF_LWR:.4f}  Sx-log={REF_SX:.4f}  "
          f"Ens(LWR+Smap)={REF_ENS:.4f}\n")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    pred_lwr = []                           # LWR-abs прогнозы
    pred_sx  = {p: [] for p in P_SX_GRID}  # Simplex-log прогнозы
    actuals  = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X1d_all[P_LWR][step])):
            continue

        ce = int(ce10m_all[step])

        # ── LWR-abs пул (p=3) ────────────────────────────────────────────────
        j1d = np.arange(P_LWR - 1, step)
        v1d = ~np.any(np.isnan(X1d_all[P_LWR][j1d]), axis=1) & (j1d + H < n1d)
        idx1 = j1d[v1d]
        Xp3  = list(X1d_all[P_LWR][idx1])
        yp3  = list(p1d[idx1 + H])

        j10 = np.arange(P_LWR - 1, min(ce, na - H))
        if len(j10):
            v10 = ~np.any(np.isnan(X10m_all[P_LWR][j10]), axis=1)
            i10 = j10[v10]
            Xp3.extend(X10m_all[P_LWR][i10])
            yp3.extend(p10m[i10 + H])

        p_cur = float(p1d[step])
        y_lwr = np.nan
        if len(Xp3) >= P_LWR + 2:
            X_pool = np.array(Xp3); y_abs = np.array(yp3)
            x_q    = X1d_all[P_LWR][step]
            mu  = X_pool.mean(0)
            sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
            Xn  = (X_pool - mu) / sig
            xn  = (x_q    - mu) / sig
            d3  = np.linalg.norm(Xn - xn, axis=1)
            ord3 = np.argsort(d3)
            k_eff = min(K_LWR, len(y_abs))
            knn = ord3[:k_eff]; xi = d3[ord3[k_eff - 1]]
            if xi < 1e-12:
                y_lwr = float(y_abs[knn].mean())
            else:
                w  = np.exp(-0.5 * (d3[knn] / xi) ** 2); ws = np.sqrt(w)
                A  = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
                c, *_ = np.linalg.lstsq(A, y_abs[knn] * ws, rcond=None)
                y_lwr = float(c[0] + c[1:] @ xn)
        pred_lwr.append(y_lwr)

        # ── Simplex-log пул (для каждого p_sx) ───────────────────────────────
        for p_sx in P_SX_GRID:
            if np.any(np.isnan(X1d_all[p_sx][step])):
                pred_sx[p_sx].append(np.nan)
                continue

            k_sx = p_sx + 1
            j1s  = np.arange(p_sx - 1, step)
            v1s  = ~np.any(np.isnan(X1d_all[p_sx][j1s]), axis=1) & (j1s + H < n1d)
            idx1s = j1s[v1s]
            Xps   = list(X1d_all[p_sx][idx1s])
            yps_abs = list(p1d[idx1s + H])
            yps_src = list(p1d[idx1s])

            j10s = np.arange(p_sx - 1, min(ce, na - H))
            if len(j10s):
                v10s  = ~np.any(np.isnan(X10m_all[p_sx][j10s]), axis=1)
                i10s  = j10s[v10s]
                Xps.extend(X10m_all[p_sx][i10s])
                yps_abs.extend(p10m[i10s + H])
                yps_src.extend(p10m[i10s])

            y_sx = np.nan
            if len(Xps) >= k_sx:
                X_ps  = np.array(Xps)
                ya_sx = np.array(yps_abs, dtype=float)
                ys_sx = np.array(yps_src, dtype=float)
                ylr_sx = np.log(ya_sx / ys_sx)

                x_qs = X1d_all[p_sx][step]
                mu_s  = X_ps.mean(0)
                sig_s = np.where(X_ps.std(0) < 1e-10, 1.0, X_ps.std(0))
                Xns   = (X_ps - mu_s) / sig_s
                xns   = (x_qs - mu_s) / sig_s
                ds    = np.linalg.norm(Xns - xns, axis=1)
                ords  = np.argsort(ds)
                knn_s = ords[:k_sx]
                d1    = ds[ords[0]]
                if d1 < 1e-12:
                    y_sx = float(p_cur * np.exp(ylr_sx[ords[0]]))
                else:
                    w_s  = np.exp(-ds[knn_s] / d1)
                    ws_s = w_s / w_s.sum()
                    y_sx = float(p_cur * np.exp(ws_s @ ylr_sx[knn_s]))
            pred_sx[p_sx].append(y_sx)

        actuals.append(float(p1d[step + H]))

    acts = np.array(actuals)
    n    = len(acts)
    arr_lwr = np.array(pred_lwr, dtype=float)
    print(f"Шагов: {n}")
    print(f"LWR-abs:     rMAE={rmae(arr_lwr, acts):.4f}\n")

    # ── α-свип для каждого p_sx ───────────────────────────────────────────────
    best_overall = (np.inf, None, None, None)   # (rMAE, p_sx, α, label)

    fig, axes = plt.subplots(1, len(P_SX_GRID), figsize=(20, 5), sharey=True)

    print(f"{'p_sx':>5}  {'k':>3}  {'α_best':>7}  "
          f"{'rMAE_ens':>9}  {'vs_LWR%':>8}  {'vs_Ens%':>8}")
    print("-" * 52)

    for ax, p_sx in zip(axes, P_SX_GRID):
        arr_sx = np.array(pred_sx[p_sx], dtype=float)
        r_sx   = rmae(arr_sx, acts)

        r_alpha = []
        for α in A_GRID:
            blend = α * arr_lwr + (1 - α) * arr_sx
            r_alpha.append(rmae(blend, acts))

        r_alpha = np.array(r_alpha)
        best_idx = np.nanargmin(r_alpha)
        best_a   = A_GRID[best_idx]
        best_r   = r_alpha[best_idx]

        vs_lwr = (best_r / REF_LWR - 1) * 100
        vs_ens = (best_r / REF_ENS - 1) * 100
        print(f"{p_sx:>5}  {p_sx+1:>3}  {best_a:>7.2f}  "
              f"{best_r:>9.4f}  {vs_lwr:>+7.1f}%  {vs_ens:>+7.1f}%")

        if best_r < best_overall[0]:
            best_overall = (best_r, p_sx, best_a, f"p_sx={p_sx}")

        # График кривой α
        ax.plot(A_GRID, r_alpha, "o-", lw=1.5, color="purple", ms=4)
        ax.axhline(REF_LWR, color="steelblue", ls=":",  lw=1.2, label=f"LWR={REF_LWR:.4f}")
        ax.axhline(REF_ENS, color="green",     ls="--", lw=1.2, label=f"Ens(+Smap)={REF_ENS:.4f}")
        ax.axhline(r_sx,    color="tomato",    ls=":",  lw=1.0, label=f"Sx-log={r_sx:.4f}")
        ax.axvline(best_a,  color="gray",      ls="-",  lw=0.8, alpha=0.5)
        ax.scatter([best_a], [best_r], color="red", zorder=5, s=50)
        ax.annotate(f"α={best_a:.2f}\n{best_r:.4f}",
                    (best_a, best_r), textcoords="offset points",
                    xytext=(8, -18), fontsize=8, color="red")
        ax.set_title(f"Simplex-log p={p_sx} k={p_sx+1}\nSx-log={r_sx:.4f}")
        ax.set_xlabel("α  (1→100% LWR)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    axes[0].set_ylabel("rMAE")
    fig.suptitle(
        f"Ансамбль LWR-abs(p={P_LWR},K={K_LWR}) + Simplex-log  |  SBER 1d+10m  H={H}\n"
        f"ŷ = α·LWR + (1−α)·Simplex-log",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ensemble_simplex.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'='*52}")
    r_best, p_best, a_best, lbl = best_overall
    print(f"Лучший ансамбль: {lbl}  α={a_best:.2f}  rMAE={r_best:.4f}")
    print(f"  vs LWR-abs:       {(r_best/REF_LWR-1)*100:+.1f}%")
    print(f"  vs Ens(LWR+Smap): {(r_best/REF_ENS-1)*100:+.1f}%")
    print(f"\nГрафик → {OUT}/ensemble_simplex.png")


if __name__ == "__main__":
    run()
