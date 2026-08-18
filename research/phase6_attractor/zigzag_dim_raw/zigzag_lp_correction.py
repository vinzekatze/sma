#!/usr/bin/env python3
"""
LP-коррекция прогнозов зигзага.

После того как LWR/S-map даёт сырую цену ŷ, строим предсказанный
следующий вектор состояния:
  X_pred = [ŷ, log(ŷ/p_cur), log(p_cur/p_prev)]

LP-коррекция находит k_lp ближайших исторических состояний к X_pred
(каузально: j < step), строит локальное d_lp-мерное подпространство
через SVD и проецирует X_pred на него. Корректированная цена — первая
компонента проекции.

Это аналог LP-коррекции траектории из app3, применённой к одной
предсказанной точке.

Свип: k_lp ∈ {15,30,50}, d_lp ∈ {1,2}.
Источники: LWR K=50, S-map θ=10, ансамбль α=0.65.

SBER 1d(4%) + 10m(0.4%), p=3, H=1.
Базовые: LWR=0.4181, S-map=0.4251, Ens=0.4143.
КАУЗАЛЬНОСТЬ: пул строго j < step.
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
P           = 3
H           = 1
MIN_HISTORY = 50
K_LWR       = 50
THETA_SMAP  = 10.0
ALPHA_ENS   = 0.65

K_LP_GRID = [15, 30, 50]
D_LP_GRID = [1, 2]

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


def lp_correct(X_pool, x_pred, k_lp, d_lp):
    """
    Проецирует x_pred на d_lp-мерное локальное подпространство,
    найденное по k_lp ближайшим соседям в X_pool (z-score пространство).
    Возвращает корректированную цену (первая компонента).
    """
    N = len(X_pool)
    k = min(k_lp, N)
    if k < d_lp + 1:
        return float(x_pred[0])

    mu    = X_pool.mean(0)
    sigma = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
    Xn      = (X_pool - mu) / sigma
    xn_pred = (x_pred  - mu) / sigma

    dists  = np.linalg.norm(Xn - xn_pred, axis=1)
    knn    = np.argsort(dists)[:k]

    Z      = Xn[knn]
    center = Z.mean(0)
    Zc     = Z - center

    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    V = Vt[:d_lp].T  # p × d_lp

    delta   = xn_pred - center
    xc_norm = center + V @ (V.T @ delta)
    x_corr  = xc_norm * sigma + mu

    return float(x_corr[0])


def make_x_pred(y_hat, x_cur):
    """
    Предсказанный следующий вектор состояния для p=3:
      X[step+1] = [ŷ, log(ŷ/p_cur), log(p_cur/p_prev)]
    x_cur = [p_cur, lr1_cur, lr2_cur]
    """
    p_cur  = x_cur[0]
    lr1_pred = np.log(y_hat / p_cur) if p_cur > 1e-12 else 0.0
    lr2_pred = x_cur[1]   # log(p_cur/p_prev) — уже известен
    return np.array([y_hat, lr1_pred, lr2_pred])


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dates_1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dates_10m = find_pivots(h10m, l10m, d10m, T_10M)

    X_1d  = build_X(p1d,  P)
    X_10m = build_X(p10m, P)

    n1d = len(p1d)
    ce10m_all = np.searchsorted(dates_10m, dates_1d, side="left")

    print(f"SBER 1d {n1d} пив  10m {len(p10m)} пив")
    print(f"p={P}  H={H}  K_LWR={K_LWR}  θ={THETA_SMAP}  α_ens={ALPHA_ENS}")
    print(f"k_lp={K_LP_GRID}  d_lp={D_LP_GRID}")
    print(f"Базовые: LWR={R_LWR:.4f}  S-map={R_SMAP:.4f}  Ens={R_ENS:.4f}\n")

    # Буферы: сырые предсказания + LP-скорректированные
    raw_lwr  = []
    raw_smap = []
    # LP: ключ (source, k_lp, d_lp)
    lp_preds = {(src, k, d): []
                for src in ("lwr", "smap", "ens")
                for k in K_LP_GRID
                for d in D_LP_GRID}
    actuals_all = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X_1d[step])):
            continue

        # Построить пул (1d + 10m)
        j1d = np.arange(P - 1, step)
        v1d = ~np.any(np.isnan(X_1d[j1d]), axis=1) & (j1d + H < n1d)
        Xp  = list(X_1d[j1d[v1d]])
        yp  = list(p1d[j1d[v1d] + H])

        ce = int(ce10m_all[step]); na = len(p10m)
        j10 = np.arange(P - 1, min(ce, na - H))
        if len(j10):
            v10 = ~np.any(np.isnan(X_10m[j10]), axis=1)
            Xp.extend(X_10m[j10[v10]])
            yp.extend(p10m[j10[v10] + H])

        N = len(Xp)
        if N < P + 2:
            continue

        X_pool = np.array(Xp); y_pool = np.array(yp)
        x_q    = X_1d[step]

        # Z-score пула (для LWR и S-map)
        mu    = X_pool.mean(0)
        sigma = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn    = (X_pool - mu) / sigma
        xn    = (x_q    - mu) / sigma

        dists = np.linalg.norm(Xn - xn, axis=1)
        order = np.argsort(dists)

        # ── LWR K=50 ─────────────────────────────────────────────────────────
        k_eff = min(K_LWR, N)
        knn   = order[:k_eff]
        xi    = dists[order[k_eff - 1]]
        if xi < 1e-12:
            y_lwr = float(y_pool[knn].mean())
        else:
            w  = np.exp(-0.5 * (dists[knn] / xi) ** 2)
            ws = np.sqrt(w)
            A  = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
            b  = y_pool[knn] * ws
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
            y_lwr = float(c[0] + c[1:] @ xn)

        # ── S-map θ=10 ───────────────────────────────────────────────────────
        mean_d = dists.mean()
        w_sm   = np.exp(-THETA_SMAP * dists / (mean_d + 1e-12))
        ws_sm  = np.sqrt(w_sm)
        A_sm   = np.column_stack([np.ones(N), Xn]) * ws_sm[:, None]
        b_sm   = y_pool * ws_sm
        c_sm, *_ = np.linalg.lstsq(A_sm, b_sm, rcond=None)
        y_smap = float(c_sm[0] + c_sm[1:] @ xn)

        y_ens = ALPHA_ENS * y_lwr + (1 - ALPHA_ENS) * y_smap

        raw_lwr.append(y_lwr)
        raw_smap.append(y_smap)

        # ── LP-коррекция ──────────────────────────────────────────────────────
        # LP-пул: те же X_pool (состояния пивотов), но без y (LP работает
        # с состояниями X, не с целями y)
        # Для каждого источника строим x_pred и корректируем
        X_pool_states = X_pool  # исторические состояния [price, lr1, lr2]

        for src, y_hat in [("lwr", y_lwr), ("smap", y_smap), ("ens", y_ens)]:
            x_pred = make_x_pred(y_hat, x_q)
            for k_lp in K_LP_GRID:
                for d_lp in D_LP_GRID:
                    y_corr = lp_correct(X_pool_states, x_pred, k_lp, d_lp)
                    lp_preds[(src, k_lp, d_lp)].append(y_corr)

        actuals_all.append(float(p1d[step + H]))

    actuals = np.array(actuals_all)
    n = len(actuals)
    print(f"Шагов: {n}\n")

    r_lwr  = rmae(raw_lwr,  actuals)
    r_smap = rmae(raw_smap, actuals)
    r_ens  = rmae(np.array(raw_lwr) * ALPHA_ENS +
                  np.array(raw_smap) * (1 - ALPHA_ENS), actuals)

    print(f"Сырые (пересчёт):  LWR={r_lwr:.4f}  S-map={r_smap:.4f}  "
          f"Ens α={ALPHA_ENS}={r_ens:.4f}\n")

    # ── Таблица результатов ───────────────────────────────────────────────────
    records = []
    for src, r_base, tag in [("lwr", r_lwr, "LWR"), ("smap", r_smap, "S-map"),
                              ("ens", r_ens, "Ens")]:
        print(f"=== LP-коррекция → {tag} (base={r_base:.4f}) ===")
        print(f"{'k_lp':>6}  {'d_lp':>5}  {'rMAE':>8}  {'vs raw':>8}  "
              f"{'vs ENS-base':>12}")
        print("─" * 50)
        best_r = np.inf
        for k_lp in K_LP_GRID:
            for d_lp in D_LP_GRID:
                r = rmae(lp_preds[(src, k_lp, d_lp)], actuals)
                vs_raw = f"{(r/r_base-1)*100:+.1f}%"
                vs_ens = f"{(r/R_ENS-1)*100:+.1f}%"
                mk = " ←" if r < best_r else ""
                if r < best_r:
                    best_r = r
                print(f"  {k_lp:4d}     {d_lp:1d}     {r:.4f}"
                      f"   {vs_raw:>8}   {vs_ens:>12}{mk}")
                records.append({"source": tag, "k_lp": k_lp, "d_lp": d_lp,
                                 "rMAE": r, "vs_raw_pct": (r/r_base-1)*100,
                                 "vs_ens_base_pct": (r/R_ENS-1)*100})
        print()

    # Итог: лучшие LP-варианты
    df = pd.DataFrame(records)
    best = df.nsmallest(5, "rMAE")
    print("=== Топ-5 LP-вариантов ===")
    print(best[["source", "k_lp", "d_lp", "rMAE",
                "vs_raw_pct", "vs_ens_base_pct"]].to_string(index=False))

    df.to_csv(OUT / "lp_correction.csv", index=False)

    # ── График ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    src_cfg = [
        ("lwr",  r_lwr,  "LWR K=50",    "steelblue"),
        ("smap", r_smap, "S-map θ=10",  "darkorange"),
        ("ens",  r_ens,  f"Ens α={ALPHA_ENS}", "purple"),
    ]

    for ax, (src, r_base, label, color) in zip(axes, src_cfg):
        sub = df[df["source"] == label.split()[0].replace("Ens", "Ens")]
        # Filter correctly
        sub = df[df["source"] == ("LWR" if src == "lwr" else
                                   "S-map" if src == "smap" else "Ens")]

        xs = np.arange(len(K_LP_GRID) * len(D_LP_GRID))
        labels_x = [f"k={k}\nd={d}" for k in K_LP_GRID for d in D_LP_GRID]
        rmae_vals = [rmae(lp_preds[(src, k, d)], actuals)
                     for k in K_LP_GRID for d in D_LP_GRID]
        clrs = ["seagreen" if r < R_ENS else
                ("steelblue" if r < r_base else "salmon")
                for r in rmae_vals]
        bars = ax.bar(xs, rmae_vals, color=clrs, alpha=0.85)
        ax.axhline(r_base, color=color, lw=1.5, ls="--",
                   label=f"{label} raw {r_base:.4f}")
        ax.axhline(R_ENS, color="crimson", lw=1.2, ls=":",
                   label=f"Ens base {R_ENS:.4f}")
        for bar, r in zip(bars, rmae_vals):
            ax.text(bar.get_x() + bar.get_width()/2, r + 0.001,
                    f"{r:.4f}", ha="center", fontsize=7.5)
        ax.set_xticks(xs); ax.set_xticklabels(labels_x, fontsize=8)
        ax.set_ylabel("rMAE"); ax.set_title(f"LP-коррекция → {label}")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(min(rmae_vals + [R_ENS]) - 0.01,
                    max(rmae_vals + [r_base]) + 0.02)

    fig.suptitle(
        f"LP-коррекция прогноза зигзага  |  SBER 1d({T_1D*100:.0f}%)+10m({T_10M*100:.1f}%)"
        f"  p={P}  H={H}  n={n}\n"
        f"Свип k_lp∈{K_LP_GRID}  d_lp∈{D_LP_GRID}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(OUT / "lp_correction.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + график → {OUT}")


if __name__ == "__main__":
    run()
