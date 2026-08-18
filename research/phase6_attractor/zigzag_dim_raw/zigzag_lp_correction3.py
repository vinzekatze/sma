#!/usr/bin/env python3
"""
LP-коррекция прогнозов зигзага — v3: log-return пространство.

Проблема v1-v2: LP-пул содержал абсолютную цену → SVD выделял ценовой тренд
→ проекция смещала ŷ к историческим уровням цен.

Решение: убрать абсолютную цену из LP-пространства.

LP-пространство: X_LP[i] = [lr1[i], lr2[i], ..., lr6[i]]  (6D, только лог-доходности)
Предсказанный вектор:
  x_pred = [log(ŷ/p_cur), lr1_cur, lr2_cur, lr3_cur, lr4_cur, lr5_cur]
           = [lr1_pred,    X_LP[step][0], ..., X_LP[step][4]]

LP корректирует lr1_pred → lr1_corr → ŷ_corr = p_cur · exp(lr1_corr)

Log-return пространство масштаб-инвариантно:
  - нет эффекта «историческая цена 90₽ vs 300₽»
  - SVD выделяет паттерны ДИНАМИКИ, а не ценовые уровни
  - соседи подбираются по похожести моментума (z-score lr1 разделяет 1d и 10m)

P_LP = 6   (6 лог-доходностей)
d ∈ {1, 2, 3, 5}  (FNN: 3 для 1d-only, 5-6 для 1d+10m)
k ∈ {5, 7, 10, 12, 15, 20, 30}

SBER 1d(4%) + 10m(0.4%), p_pred=3, H=1.
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
P_PRED      = 3
P_LP        = 6     # число лог-доходностей в LP-пространстве
H           = 1
MIN_HISTORY = 50
K_LWR       = 50
THETA_SMAP  = 10.0
ALPHA_ENS   = 0.65

D_LP_GRID = [1, 2, 3, 5]
K_LP_GRID = [5, 7, 10, 12, 15, 20, 30]

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


def build_X_pred(prices, p):
    """Признаки для LWR/S-map: [price, lr1, ..., lr(p-1)]"""
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def build_X_lr(prices, p_lr):
    """LP-пространство: только лог-доходности [lr1, lr2, ..., lrp]"""
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p_lr), np.nan)
    for i in range(p_lr, n):
        for lag in range(p_lr):
            X[i, lag] = lp[i - lag] - lp[i - lag - 1]
    return X


def rmae(preds, actuals):
    arr = np.asarray(preds, dtype=float)
    if len(arr) < 5 or np.any(np.isnan(arr)):
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return np.nan if dz < 1e-12 else float(np.mean(np.abs(arr - actuals)) / dz)


def make_x_pred_lr(y_hat, p_cur, x_cur_lr):
    """
    Предсказанный следующий вектор в log-return пространстве.
    X_lr[step+1] = [log(ŷ/p_cur), lr1_cur, lr2_cur, ..., lr(P_LP-1)_cur]
    x_cur_lr = X_lr[step] = [lr1, lr2, ..., lrP_LP]
    """
    p_lr = len(x_cur_lr)
    x_pred = np.empty(p_lr)
    x_pred[0] = float(np.log(y_hat / p_cur)) if p_cur > 1e-12 else 0.0
    for i in range(1, p_lr):
        x_pred[i] = x_cur_lr[i - 1]
    return x_pred


def lp_correct_lr(X_pool_lr, x_pred_lr, p_cur, k_lp, d_lp):
    """
    LP-коррекция в log-return пространстве.
    Возвращает скорректированную цену: p_cur · exp(lr1_corr).
    """
    N = len(X_pool_lr)
    k = min(k_lp, N)
    if k <= d_lp:
        return float(p_cur * np.exp(x_pred_lr[0]))

    mu    = X_pool_lr.mean(0)
    sigma = np.where(X_pool_lr.std(0) < 1e-10, 1.0, X_pool_lr.std(0))
    Xn      = (X_pool_lr - mu) / sigma
    xn_pred = (x_pred_lr - mu) / sigma

    dists = np.linalg.norm(Xn - xn_pred, axis=1)
    knn   = np.argsort(dists)[:k]

    Z      = Xn[knn]
    center = Z.mean(0)
    Zc     = Z - center

    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    V = Vt[:d_lp].T   # P_LP × d_lp

    delta   = xn_pred - center
    xc_norm = center + V @ (V.T @ delta)
    x_corr  = xc_norm * sigma + mu   # de-normalize

    lr1_corr = x_corr[0]
    return float(p_cur * np.exp(lr1_corr))


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dates_1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dates_10m = find_pivots(h10m, l10m, d10m, T_10M)

    X3_1d  = build_X_pred(p1d,  P_PRED)
    X3_10m = build_X_pred(p10m, P_PRED)
    Xlr_1d  = build_X_lr(p1d,  P_LP)
    Xlr_10m = build_X_lr(p10m, P_LP)

    n1d = len(p1d)
    ce10m_all = np.searchsorted(dates_10m, dates_1d, side="left")

    print(f"SBER 1d {n1d} пив  10m {len(p10m)} пив")
    print(f"P_PRED={P_PRED}  P_LP={P_LP} (log-return)  H={H}")
    print(f"d_lp={D_LP_GRID}  k_lp={K_LP_GRID}")
    print(f"Базовые: LWR={R_LWR:.4f}  S-map={R_SMAP:.4f}  Ens={R_ENS:.4f}\n")

    raw_lwr  = []
    raw_smap = []
    lp_buf   = {(src, k, d): []
                for src in ("lwr", "smap", "ens")
                for k in K_LP_GRID for d in D_LP_GRID}
    actuals_all = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X3_1d[step])) or np.any(np.isnan(Xlr_1d[step])):
            continue

        # ── Пул для LWR/S-map (p=3, с абс. ценой) ───────────────────────────
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

        mu3  = X_pool3.mean(0)
        sig3 = np.where(X_pool3.std(0) < 1e-10, 1.0, X_pool3.std(0))
        Xn3  = (X_pool3 - mu3) / sig3
        xn3  = (x_q3    - mu3) / sig3
        d3   = np.linalg.norm(Xn3 - xn3, axis=1)
        ord3 = np.argsort(d3)

        # LWR K=50
        k_eff = min(K_LWR, len(y_pool))
        knn   = ord3[:k_eff]; xi = d3[ord3[k_eff - 1]]
        if xi < 1e-12:
            y_lwr = float(y_pool[knn].mean())
        else:
            w  = np.exp(-0.5 * (d3[knn] / xi) ** 2); ws = np.sqrt(w)
            A  = np.column_stack([np.ones(k_eff), Xn3[knn]]) * ws[:, None]
            c, *_ = np.linalg.lstsq(A, y_pool[knn] * ws, rcond=None)
            y_lwr = float(c[0] + c[1:] @ xn3)

        # S-map θ=10
        N3 = len(y_pool); mean_d = d3.mean()
        w_sm = np.exp(-THETA_SMAP * d3 / (mean_d + 1e-12)); ws_sm = np.sqrt(w_sm)
        A_sm = np.column_stack([np.ones(N3), Xn3]) * ws_sm[:, None]
        c_sm, *_ = np.linalg.lstsq(A_sm, y_pool * ws_sm, rcond=None)
        y_smap = float(c_sm[0] + c_sm[1:] @ xn3)

        y_ens = ALPHA_ENS * y_lwr + (1 - ALPHA_ENS) * y_smap

        raw_lwr.append(y_lwr); raw_smap.append(y_smap)

        # ── Пул для LP (log-return пространство) ─────────────────────────────
        j1d_lr = np.arange(P_LP, step)          # нужно P_LP лагов → start с P_LP
        v1d_lr = ~np.any(np.isnan(Xlr_1d[j1d_lr]), axis=1)
        Xp_lr  = list(Xlr_1d[j1d_lr[v1d_lr]])

        j10_lr = np.arange(P_LP, min(ce, na))
        if len(j10_lr):
            v10_lr = ~np.any(np.isnan(Xlr_10m[j10_lr]), axis=1)
            Xp_lr.extend(Xlr_10m[j10_lr[v10_lr]])

        if len(Xp_lr) < P_LP + 1:
            for src in ("lwr", "smap", "ens"):
                y_fb = y_lwr if src == "lwr" else y_smap if src == "smap" else y_ens
                for k in K_LP_GRID:
                    for d in D_LP_GRID:
                        lp_buf[(src, k, d)].append(y_fb)
            actuals_all.append(float(p1d[step + H])); continue

        X_pool_lr = np.array(Xp_lr)
        x_cur_lr  = Xlr_1d[step]
        p_cur     = float(p1d[step])

        for src, y_hat in [("lwr", y_lwr), ("smap", y_smap), ("ens", y_ens)]:
            x_pred_lr = make_x_pred_lr(y_hat, p_cur, x_cur_lr)
            for k_lp in K_LP_GRID:
                for d_lp in D_LP_GRID:
                    y_corr = lp_correct_lr(X_pool_lr, x_pred_lr, p_cur, k_lp, d_lp)
                    lp_buf[(src, k_lp, d_lp)].append(y_corr)

        actuals_all.append(float(p1d[step + H]))

    actuals = np.array(actuals_all)
    n = len(actuals)
    print(f"Шагов: {n}\n")

    r_lwr_raw  = rmae(raw_lwr,  actuals)
    r_smap_raw = rmae(raw_smap, actuals)
    arr_ens    = np.array(raw_lwr)*ALPHA_ENS + np.array(raw_smap)*(1-ALPHA_ENS)
    r_ens_raw  = rmae(arr_ens, actuals)
    print(f"Сырые: LWR={r_lwr_raw:.4f}  S-map={r_smap_raw:.4f}  "
          f"Ens={r_ens_raw:.4f}\n")

    # ── Матрицы rMAE (d строки, k столбцы) ───────────────────────────────────
    records = []
    for src, r_raw, label in [
        ("lwr",  r_lwr_raw,  "LWR"),
        ("smap", r_smap_raw, "S-map"),
        ("ens",  r_ens_raw,  "Ens"),
    ]:
        print(f"=== {label} (raw={r_raw:.4f}) — матрица rMAE ===")
        header = f"{'d\\k':>5}  " + "  ".join(f"k={k:2d}" for k in K_LP_GRID)
        print(header)
        best_r = np.inf; best_kd = None
        for d_lp in D_LP_GRID:
            row = f"  d={d_lp}  "
            for k_lp in K_LP_GRID:
                arr = np.array(lp_buf[(src, k_lp, d_lp)], dtype=float)
                r = rmae(arr, actuals)
                row += f"  {r:.4f}"
                mk = ""
                if r < best_r:
                    best_r = r; best_kd = (k_lp, d_lp); mk = "*"
                records.append({"source": label, "k_lp": k_lp, "d_lp": d_lp,
                                 "rMAE": r,
                                 "vs_raw_pct": (r/r_raw-1)*100,
                                 "vs_ens_base_pct": (r/R_ENS-1)*100})
            print(row)
        print(f"  → лучшие: k={best_kd[0]}  d={best_kd[1]}  "
              f"rMAE={best_r:.4f}  vs raw: {(best_r/r_raw-1)*100:+.1f}%  "
              f"vs Ens-base: {(best_r/R_ENS-1)*100:+.1f}%\n")

    df = pd.DataFrame(records)
    df.to_csv(OUT / "lp_correction3.csv", index=False)

    # Топ-10
    best10 = df.nsmallest(10, "rMAE")
    print("=== Топ-10 LP-lr вариантов ===")
    print(best10[["source","k_lp","d_lp","rMAE","vs_raw_pct",
                  "vs_ens_base_pct"]].to_string(index=False))

    # ── График: heatmap rMAE для каждого источника ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    src_cfg = [
        ("lwr",  r_lwr_raw,  "LWR K=50"),
        ("smap", r_smap_raw, "S-map θ=10"),
        ("ens",  r_ens_raw,  f"Ens α={ALPHA_ENS}"),
    ]

    for ax, (src, r_raw, label) in zip(axes, src_cfg):
        mat = np.array([[rmae(np.array(lp_buf[(src, k, d)]), actuals)
                         for k in K_LP_GRID]
                        for d in D_LP_GRID])
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                       vmin=min(mat.min(), R_ENS)*0.98,
                       vmax=max(mat.max(), r_raw)*1.01)
        for i, d_lp in enumerate(D_LP_GRID):
            for j, k_lp in enumerate(K_LP_GRID):
                r = mat[i, j]
                mark = "✓" if r < R_ENS else ("~" if r < r_raw else "✗")
                ax.text(j, i, f"{r:.4f}\n{mark}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if r > (mat.max()*0.7 + mat.min()*0.3) else "black")
        ax.set_xticks(range(len(K_LP_GRID)))
        ax.set_xticklabels([str(k) for k in K_LP_GRID])
        ax.set_yticks(range(len(D_LP_GRID)))
        ax.set_yticklabels([f"d={d}" for d in D_LP_GRID])
        ax.set_xlabel("k_lp"); ax.set_ylabel("d_lp")
        ax.set_title(f"LP-lr → {label}  (raw={r_raw:.4f})")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"LP-коррекция v3: log-return пространство  P_LP={P_LP}D\n"
        f"SBER 1d({T_1D*100:.0f}%)+10m({T_10M*100:.1f}%)  "
        f"p_pred={P_PRED}  H={H}  n={n}  "
        f"Ens-base={R_ENS:.4f}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(OUT / "lp_correction3.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + heatmap → {OUT}")


if __name__ == "__main__":
    run()
