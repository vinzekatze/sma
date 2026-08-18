#!/usr/bin/env python3
"""
LP-коррекция прогнозов зигзага — пересмотренная версия.

Проблема v1: d=1,2 слишком малы (нет смысла проецировать из p=3 в 1-2D),
k=15..50 слишком велики для чистого зигзага.

Исправление:
  P_LP = 7   — окружающее пространство LP (выше FNN_max=6)
  d ∈ {3, 6} — по FNN: 3 (1d-alone), 6 (1d+10m combined)
  k_lp ∈ {5, 7, 10, 12, 15} — точечная локальность

Схема LP-коррекции:
  1. LWR K=50 p=3 → ŷ
  2. Строим предсказанный вектор в P_LP=7 пространстве:
       X_pred = [ŷ, log(ŷ/p_cur), lr1_cur, lr2_cur, lr3_cur, lr4_cur, lr5_cur]
     где lr1..lr5_cur известны из истории зигзага.
  3. LP находит k_lp ближайших в 7D пуле (j < step, каузально),
     строит SVD, проецирует X_pred → d-мерное подпространство.
  4. Корректированная цена = X_corr[0].

Источники: LWR K=50 p=3, S-map θ=10 p=3, ансамбль α=0.65.
SBER 1d(4%) + 10m(0.4%), H=1.
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

T_1D        = 0.04
T_10M       = 0.004
P_PRED      = 3      # LWR/S-map embedding
P_LP        = 7      # LP ambient space
H           = 1
MIN_HISTORY = 50
K_LWR       = 50
THETA_SMAP  = 10.0
ALPHA_ENS   = 0.65

D_LP_GRID = [3, 6]
K_LP_GRID = [5, 7, 10, 12, 15]

R_LWR  = 0.4181
R_SMAP = 0.4251
R_ENS  = 0.4143


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    vals, dts, direction = [], [], 0
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
                vals.append((ext_val, dates[ext_idx]))
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append((ext_val, dates[ext_idx]))
                direction = 1; ext_val, ext_idx = highs[i], i
    if not vals:
        return np.array([]), np.array([])
    return (np.array([v for v, _ in vals]),
            np.array([d for _, d in vals]))


def build_X(prices, p):
    """X[i] = [price[i], lr1, lr2, ..., lr(p-1)]"""
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def rmae(preds, actuals):
    arr = np.asarray(preds, dtype=float)
    if len(arr) < 5 or np.any(np.isnan(arr)):
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return np.nan if dz < 1e-12 else float(np.mean(np.abs(arr - actuals)) / dz)


def make_x_pred_plp(y_hat, x_cur_plp):
    """
    Предсказанный вектор в P_LP-мерном пространстве.
    X[step+1] = [ŷ, log(ŷ/p_cur), lr1_cur, lr2_cur, ..., lr(P_LP-2)_cur]

    x_cur_plp = X_LP[step] = [p_cur, lr1, lr2, ..., lr(P_LP-1)]
    Сдвигаем лаговые признаки: lr_k of next = lr_(k-1) of current.
    """
    p_lp = len(x_cur_plp)
    x_pred = np.empty(p_lp)
    x_pred[0] = y_hat
    p_cur = x_cur_plp[0]
    x_pred[1] = float(np.log(y_hat / p_cur)) if p_cur > 1e-12 else 0.0
    for i in range(2, p_lp):
        x_pred[i] = x_cur_plp[i - 1]   # lr_(i-1) of step → lr_i of step+1
    return x_pred


def lp_correct(X_pool_lp, x_pred, k_lp, d_lp):
    """
    LP-коррекция: проецирует x_pred (P_LP-мерный) на d_lp-мерное
    подпространство, найденное SVD k_lp ближайших соседей в X_pool_lp.
    Все операции в z-score пространстве пула.
    Возвращает скорректированную цену (компонента 0).
    """
    N = len(X_pool_lp)
    k = min(k_lp, N)
    if k <= d_lp:
        return float(x_pred[0])

    mu    = X_pool_lp.mean(0)
    sigma = np.where(X_pool_lp.std(0) < 1e-10, 1.0, X_pool_lp.std(0))
    Xn      = (X_pool_lp - mu) / sigma
    xn_pred = (x_pred     - mu) / sigma

    dists = np.linalg.norm(Xn - xn_pred, axis=1)
    knn   = np.argsort(dists)[:k]

    Z      = Xn[knn]
    center = Z.mean(0)
    Zc     = Z - center

    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    V = Vt[:d_lp].T   # P_LP × d_lp

    delta   = xn_pred - center
    xc_norm = center + V @ (V.T @ delta)
    x_corr  = xc_norm * sigma + mu

    return float(x_corr[0])


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dates_1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dates_10m = find_pivots(h10m, l10m, d10m, T_10M)

    # Признаковые матрицы: p=3 для LWR/S-map, p=7 для LP
    X3_1d  = build_X(p1d,  P_PRED)
    X3_10m = build_X(p10m, P_PRED)
    X7_1d  = build_X(p1d,  P_LP)
    X7_10m = build_X(p10m, P_LP)

    n1d = len(p1d)
    ce10m_all = np.searchsorted(dates_10m, dates_1d, side="left")

    print(f"SBER 1d {n1d} пив  10m {len(p10m)} пив")
    print(f"P_PRED={P_PRED}  P_LP={P_LP}  H={H}")
    print(f"k_lp={K_LP_GRID}  d_lp={D_LP_GRID}")
    print(f"Базовые: LWR={R_LWR:.4f}  S-map={R_SMAP:.4f}  Ens={R_ENS:.4f}\n")

    raw_lwr   = []
    raw_smap  = []
    lp_buf    = {(src, k, d): []
                 for src in ("lwr", "smap", "ens")
                 for k in K_LP_GRID for d in D_LP_GRID}
    actuals_all = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X3_1d[step])) or np.any(np.isnan(X7_1d[step])):
            continue

        # ── Пул для LWR/S-map (p=3) ──────────────────────────────────────────
        j1d = np.arange(P_PRED - 1, step)
        v1d = ~np.any(np.isnan(X3_1d[j1d]), axis=1) & (j1d + H < n1d)
        Xp3 = list(X3_1d[j1d[v1d]]); yp3 = list(p1d[j1d[v1d] + H])

        ce = int(ce10m_all[step]); na = len(p10m)
        j10 = np.arange(P_PRED - 1, min(ce, na - H))
        if len(j10):
            v10 = ~np.any(np.isnan(X3_10m[j10]), axis=1)
            Xp3.extend(X3_10m[j10[v10]]); yp3.extend(p10m[j10[v10] + H])

        if len(Xp3) < P_PRED + 2:
            continue

        X_pool3 = np.array(Xp3); y_pool = np.array(yp3)
        x_q3    = X3_1d[step]

        mu3    = X_pool3.mean(0)
        sig3   = np.where(X_pool3.std(0) < 1e-10, 1.0, X_pool3.std(0))
        Xn3    = (X_pool3 - mu3) / sig3
        xn3    = (x_q3    - mu3) / sig3

        dists3 = np.linalg.norm(Xn3 - xn3, axis=1)
        order3 = np.argsort(dists3)

        # LWR K=50
        k_eff = min(K_LWR, len(y_pool))
        knn   = order3[:k_eff]
        xi    = dists3[order3[k_eff - 1]]
        if xi < 1e-12:
            y_lwr = float(y_pool[knn].mean())
        else:
            w  = np.exp(-0.5 * (dists3[knn] / xi) ** 2)
            ws = np.sqrt(w)
            A  = np.column_stack([np.ones(k_eff), Xn3[knn]]) * ws[:, None]
            b  = y_pool[knn] * ws
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
            y_lwr = float(c[0] + c[1:] @ xn3)

        # S-map θ=10
        N3     = len(y_pool)
        mean_d = dists3.mean()
        w_sm   = np.exp(-THETA_SMAP * dists3 / (mean_d + 1e-12))
        ws_sm  = np.sqrt(w_sm)
        A_sm   = np.column_stack([np.ones(N3), Xn3]) * ws_sm[:, None]
        b_sm   = y_pool * ws_sm
        c_sm, *_ = np.linalg.lstsq(A_sm, b_sm, rcond=None)
        y_smap = float(c_sm[0] + c_sm[1:] @ xn3)

        y_ens = ALPHA_ENS * y_lwr + (1 - ALPHA_ENS) * y_smap

        raw_lwr.append(y_lwr)
        raw_smap.append(y_smap)

        # ── Пул для LP (p=7): только состояния, не цели ─────────────────────
        j1d7 = np.arange(P_LP - 1, step)
        v1d7 = ~np.any(np.isnan(X7_1d[j1d7]), axis=1)
        Xp7  = list(X7_1d[j1d7[v1d7]])

        j10_7 = np.arange(P_LP - 1, min(ce, na))
        if len(j10_7):
            v10_7 = ~np.any(np.isnan(X7_10m[j10_7]), axis=1)
            Xp7.extend(X7_10m[j10_7[v10_7]])

        if len(Xp7) < P_LP + 1:
            for src in ("lwr", "smap", "ens"):
                for k_lp in K_LP_GRID:
                    for d_lp in D_LP_GRID:
                        lp_buf[(src, k_lp, d_lp)].append(
                            y_lwr if src == "lwr" else
                            y_smap if src == "smap" else y_ens)
            actuals_all.append(float(p1d[step + H]))
            continue

        X_pool7 = np.array(Xp7)
        x_cur7  = X7_1d[step]

        # ── LP-коррекция ──────────────────────────────────────────────────────
        for src, y_hat in [("lwr", y_lwr), ("smap", y_smap), ("ens", y_ens)]:
            x_pred7 = make_x_pred_plp(y_hat, x_cur7)
            for k_lp in K_LP_GRID:
                for d_lp in D_LP_GRID:
                    y_corr = lp_correct(X_pool7, x_pred7, k_lp, d_lp)
                    lp_buf[(src, k_lp, d_lp)].append(y_corr)

        actuals_all.append(float(p1d[step + H]))

    actuals = np.array(actuals_all)
    n = len(actuals)
    print(f"Шагов: {n}\n")

    r_lwr_raw  = rmae(raw_lwr,  actuals)
    r_smap_raw = rmae(raw_smap, actuals)
    arr_ens    = np.array(raw_lwr) * ALPHA_ENS + np.array(raw_smap) * (1 - ALPHA_ENS)
    r_ens_raw  = rmae(arr_ens, actuals)
    print(f"Сырые: LWR={r_lwr_raw:.4f}  S-map={r_smap_raw:.4f}  "
          f"Ens α={ALPHA_ENS}={r_ens_raw:.4f}\n")

    # ── Таблицы результатов ───────────────────────────────────────────────────
    records = []
    for src, r_raw, label in [
        ("lwr",  r_lwr_raw,  "LWR K=50"),
        ("smap", r_smap_raw, "S-map θ=10"),
        ("ens",  r_ens_raw,  f"Ens α={ALPHA_ENS}"),
    ]:
        print(f"=== LP-коррекция → {label} (raw={r_raw:.4f}) ===")
        print(f"{'k_lp':>5}  {'d_lp':>4}  {'rMAE':>8}  {'vs raw':>8}  {'vs ENS-base':>12}")
        print("─" * 50)
        best_r = np.inf
        for k_lp in K_LP_GRID:
            for d_lp in D_LP_GRID:
                arr = np.array(lp_buf[(src, k_lp, d_lp)], dtype=float)
                r = rmae(arr, actuals)
                mk = " ←" if r < best_r else ""
                if r < best_r:
                    best_r = r
                print(f"  {k_lp:3d}    {d_lp:1d}    {r:.4f}"
                      f"   {(r/r_raw-1)*100:+.1f}%   {(r/R_ENS-1)*100:+.1f}%{mk}")
                records.append({"source": label, "k_lp": k_lp, "d_lp": d_lp,
                                 "rMAE": r,
                                 "vs_raw_pct":      (r / r_raw - 1) * 100,
                                 "vs_ens_base_pct": (r / R_ENS  - 1) * 100})
        print()

    df = pd.DataFrame(records)
    best5 = df.nsmallest(5, "rMAE")
    print("=== Топ-5 LP-вариантов ===")
    print(best5[["source", "k_lp", "d_lp", "rMAE",
                 "vs_raw_pct", "vs_ens_base_pct"]].to_string(index=False))
    df.to_csv(OUT / "lp_correction2.csv", index=False)

    # ── Матрица rMAE (d_lp строки, k_lp столбцы) для каждого источника ───────
    print("\n=== Матрицы rMAE ===")
    for src, label in [("lwr", "LWR"), ("smap", "S-map"), ("ens", "Ens")]:
        print(f"\n{label}  (строки=d_lp, столбцы=k_lp)")
        header = "       " + "  ".join(f"k={k:3d}" for k in K_LP_GRID)
        print(header)
        for d_lp in D_LP_GRID:
            row = f"  d={d_lp}  "
            for k_lp in K_LP_GRID:
                r = rmae(np.array(lp_buf[(src, k_lp, d_lp)]), actuals)
                row += f"  {r:.4f}"
            print(row)

    # ── График ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)

    src_cfg = [
        ("lwr",  r_lwr_raw,  "LWR K=50",    "steelblue"),
        ("smap", r_smap_raw, "S-map θ=10",  "darkorange"),
        ("ens",  r_ens_raw,  f"Ens α={ALPHA_ENS}", "purple"),
    ]

    for col, (src, r_raw, label, color) in enumerate(src_cfg):
        for row_idx, d_lp in enumerate(D_LP_GRID):
            ax = axes[row_idx][col]
            rmae_vals = [rmae(np.array(lp_buf[(src, k, d_lp)]), actuals)
                         for k in K_LP_GRID]
            clrs = ["seagreen" if r < R_ENS
                    else ("steelblue" if r < r_raw else "salmon")
                    for r in rmae_vals]
            ax.bar(range(len(K_LP_GRID)), rmae_vals, color=clrs, alpha=0.85)
            ax.axhline(r_raw, color=color,    lw=1.5, ls="--",
                       label=f"raw {r_raw:.4f}")
            ax.axhline(R_ENS, color="crimson", lw=1.2, ls=":",
                       label=f"Ens base {R_ENS:.4f}")
            for xi, r in enumerate(rmae_vals):
                ax.text(xi, r + 0.002, f"{r:.4f}", ha="center", fontsize=7.5)
            ax.set_xticks(range(len(K_LP_GRID)))
            ax.set_xticklabels([str(k) for k in K_LP_GRID], fontsize=8.5)
            ax.set_xlabel("k_lp")
            ax.set_ylabel("rMAE")
            ax.set_title(f"LP({label})  d={d_lp}")
            ax.legend(fontsize=7.5)
            ax.grid(axis="y", alpha=0.2)
            y_lo = min(rmae_vals + [R_ENS]) - 0.01
            y_hi = max(rmae_vals + [r_raw]) + 0.025
            ax.set_ylim(y_lo, y_hi)

    fig.suptitle(
        f"LP-коррекция v2  |  P_LP={P_LP}  d∈{D_LP_GRID}  k∈{K_LP_GRID}\n"
        f"SBER 1d({T_1D*100:.0f}%)+10m({T_10M*100:.1f}%)  p_pred={P_PRED}  H={H}  n={n}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(OUT / "lp_correction2.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + график → {OUT}")


if __name__ == "__main__":
    run()
