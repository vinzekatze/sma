#!/usr/bin/env python3
"""
LP-коррекция поверх 4-way ансамбля.

4-way: α_lwr=0.05·LWR + α_sx=0.35·Sx-log + α_sm=0.20·Smap(θ=1) + α_rbf=0.40·RBF-log
       rMAE = 0.3963

LP-коррекция v3 (log-return пространство, §6.18):
  LP-пространство: X_LP[i] = [lr1..lr6]  (P_LP=6)
  x_pred = [log(ŷ/p_cur), lr1_cur, ..., lr5_cur]
  ŷ_corr = p_cur · exp(lr1_corr)

Свип d∈{1,2,3,5}, k∈{5,7,10,12,15,20,30} — те же, что §6.18.

Вопрос: даёт ли LP дополнительный выигрыш поверх уже сильного ансамбля?

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

# 4-way параметры
P_LWR  = 3;  K_LWR  = 50
P_SX   = 8;  K_SX   = P_SX + 1
THETA  = 1.0
P_RBF  = 3;  K_RBF  = 12
A_LWR, A_SX, A_SM, A_RBF = 0.05, 0.35, 0.20, 0.40

# LP параметры
P_LP    = 6
D_GRID  = [1, 2, 3, 5]
K_GRID  = [5, 7, 10, 12, 15, 20, 30]

REF_4WAY = 0.3963
REF_3WAY = 0.4004
REF_ENS2 = 0.4055


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


def build_X_lr(prices, p_lr):
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p_lr), np.nan)
    for i in range(p_lr, n):
        for lag in range(p_lr):
            X[i, lag] = lp[i - lag] - lp[i - lag - 1]
    return X


def rmae(preds, actuals):
    preds = np.asarray(preds, dtype=float)
    valid = ~np.isnan(preds)
    if valid.sum() < 5:
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return float(np.mean(np.abs(preds[valid] - actuals[valid])) / dz) if dz > 1e-12 else np.nan


def lp_correct_lr(X_pool_lr, x_pred_lr, p_cur, k_lp, d_lp):
    N = len(X_pool_lr); k = min(k_lp, N)
    if k <= d_lp:
        return float(p_cur * np.exp(x_pred_lr[0]))
    mu    = X_pool_lr.mean(0)
    sigma = np.where(X_pool_lr.std(0) < 1e-10, 1.0, X_pool_lr.std(0))
    Xn      = (X_pool_lr - mu) / sigma
    xn_pred = (x_pred_lr - mu) / sigma
    dists = np.linalg.norm(Xn - xn_pred, axis=1)
    knn   = np.argsort(dists)[:k]
    Z = Xn[knn]; center = Z.mean(0); Zc = Z - center
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    V = Vt[:d_lp].T
    delta   = xn_pred - center
    xc_norm = center + V @ (V.T @ delta)
    x_corr  = xc_norm * sigma + mu
    return float(p_cur * np.exp(x_corr[0]))


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m = find_pivots(h10m, l10m, d10m, T_10M)

    n1d = len(p1d); na = len(p10m)
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")

    Xs = {p: (build_X(p1d, p), build_X(p10m, p))
          for p in set([P_LWR, P_SX, P_RBF])}
    Xlr_1d  = build_X_lr(p1d,  P_LP)
    Xlr_10m = build_X_lr(p10m, P_LP)

    print(f"SBER 1d {n1d} пив  10m {na} пив")
    print(f"4-way: LWR({A_LWR}) + Sx({A_SX}) + Smap({A_SM}) + RBF({A_RBF})")
    print(f"LP: P_LP={P_LP}  d={D_GRID}  k={K_GRID}")
    print(f"Эталоны: 4-way={REF_4WAY:.4f}  3-way={REF_3WAY:.4f}  "
          f"Ens2={REF_ENS2:.4f}\n")

    raw_ens  = []
    lp_buf   = {(k, d): [] for k in K_GRID for d in D_GRID}
    actuals  = []

    for step in range(MIN_HISTORY, n1d - H):
        if any(np.any(np.isnan(Xs[p][0][step])) for p in [P_LWR, P_SX, P_RBF]):
            continue
        if np.any(np.isnan(Xlr_1d[step])):
            continue

        ce    = int(ce10m_all[step])
        p_cur = float(p1d[step])

        def make_pool(p):
            X1d_, X10m_ = Xs[p]
            j1d = np.arange(p - 1, step)
            v1d = ~np.any(np.isnan(X1d_[j1d]), axis=1) & (j1d + H < n1d)
            idx1 = j1d[v1d]
            Xp = list(X1d_[idx1]); ya = list(p1d[idx1 + H]); ys = list(p1d[idx1])
            j10 = np.arange(p - 1, min(ce, na - H))
            if len(j10):
                v10 = ~np.any(np.isnan(X10m_[j10]), axis=1); idx10 = j10[v10]
                Xp.extend(X10m_[idx10]); ya.extend(p10m[idx10 + H]); ys.extend(p10m[idx10])
            if len(Xp) < p + 2:
                return None
            X_pool = np.array(Xp); y_abs = np.array(ya); y_lr = np.log(y_abs / np.array(ys))
            x_q = X1d_[step]
            mu = X_pool.mean(0); sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
            Xn = (X_pool - mu) / sig; xn = (x_q - mu) / sig
            d  = np.linalg.norm(Xn - xn, axis=1)
            return Xn, xn, d, y_abs, y_lr, len(y_abs)

        # LWR + S-map (p=3)
        res3 = make_pool(P_LWR)
        y_lwr = y_smap = np.nan
        if res3:
            Xn, xn, d, y_abs, y_lr, N = res3
            ord_ = np.argsort(d)
            k_eff = min(K_LWR, N); knn = ord_[:k_eff]; xi = d[ord_[k_eff-1]]
            if xi < 1e-12:
                y_lwr = float(y_abs[knn].mean())
            else:
                w = np.exp(-0.5*(d[knn]/xi)**2); ws = np.sqrt(w)
                A = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:,None]
                c, *_ = np.linalg.lstsq(A, y_abs[knn]*ws, rcond=None)
                y_lwr = float(c[0] + c[1:] @ xn)
            mean_d = d.mean() + 1e-12
            w_sm = np.exp(-THETA * d / mean_d); ws_sm = np.sqrt(w_sm)
            A_sm = np.column_stack([np.ones(N), Xn]) * ws_sm[:,None]
            c_sm, *_ = np.linalg.lstsq(A_sm, y_abs*ws_sm, rcond=None)
            y_smap = float(c_sm[0] + c_sm[1:] @ xn)

        # Simplex-log (p=8)
        res8 = make_pool(P_SX)
        y_sx = np.nan
        if res8 and res8[5] >= K_SX:
            Xn, xn, d, y_abs, y_lr, N = res8
            ords = np.argsort(d); knn = ords[:K_SX]; d1 = d[ords[0]]
            if d1 < 1e-12:
                y_sx = float(p_cur * np.exp(y_lr[ords[0]]))
            else:
                w = np.exp(-d[knn]/d1); w /= w.sum()
                y_sx = float(p_cur * np.exp(w @ y_lr[knn]))

        # RBF-log (p=3, K=12)
        y_rbf = np.nan
        if res3 and res3[5] >= K_RBF:
            Xn, xn, d, y_abs, y_lr, N = res3
            ord_ = np.argsort(d); k = min(K_RBF, N); knn = ord_[:k]; xi = d[ord_[k-1]]
            if xi < 1e-12:
                y_rbf = float(p_cur * np.exp(y_lr[knn].mean()))
            else:
                w = np.exp(-0.5*(d[knn]/xi)**2)
                y_rbf = float(p_cur * np.exp((w @ y_lr[knn]) / w.sum()))

        # 4-way ансамбль
        if any(np.isnan(v) for v in [y_lwr, y_sx, y_smap, y_rbf]):
            raw_ens.append(np.nan)
            for k in K_GRID:
                for dv in D_GRID:
                    lp_buf[(k, dv)].append(np.nan)
            actuals.append(float(p1d[step + H]))
            continue

        y_ens = A_LWR*y_lwr + A_SX*y_sx + A_SM*y_smap + A_RBF*y_rbf
        raw_ens.append(y_ens)

        # ── LP-коррекция ──────────────────────────────────────────────────────
        # LP-пул (log-return пространство, P_LP=6)
        j1d_lr = np.arange(P_LP, step)
        v1d_lr = ~np.any(np.isnan(Xlr_1d[j1d_lr]), axis=1)
        Xp_lr  = list(Xlr_1d[j1d_lr[v1d_lr]])

        j10_lr = np.arange(P_LP, min(ce, na))
        if len(j10_lr):
            v10_lr = ~np.any(np.isnan(Xlr_10m[j10_lr]), axis=1)
            Xp_lr.extend(Xlr_10m[j10_lr[v10_lr]])

        x_cur_lr = Xlr_1d[step]

        if len(Xp_lr) < P_LP + 1:
            for k in K_GRID:
                for dv in D_GRID:
                    lp_buf[(k, dv)].append(y_ens)
        else:
            X_pool_lr = np.array(Xp_lr)
            # x_pred в log-return пространстве
            x_pred_lr = np.empty(P_LP)
            x_pred_lr[0] = np.log(y_ens / p_cur) if p_cur > 1e-12 else 0.0
            x_pred_lr[1:] = x_cur_lr[:P_LP - 1]

            for k_lp in K_GRID:
                for d_lp in D_GRID:
                    y_corr = lp_correct_lr(X_pool_lr, x_pred_lr, p_cur, k_lp, d_lp)
                    lp_buf[(k_lp, d_lp)].append(y_corr)

        actuals.append(float(p1d[step + H]))

    acts = np.array(actuals); n = len(acts)
    r_ens = rmae(raw_ens, acts)
    print(f"Шагов: {n}")
    print(f"4-way ансамбль (сырой): {r_ens:.4f}\n")

    # ── Матрица rMAE (d × k) ──────────────────────────────────────────────────
    records = []
    print(f"{'d\\k':>5}  " + "  ".join(f"k={k:>2}" for k in K_GRID))
    print("-" * (7 + 8*len(K_GRID)))
    best_r = np.inf; best_kd = None
    for d_lp in D_GRID:
        row = f"  d={d_lp}  "
        for k_lp in K_GRID:
            arr = np.array(lp_buf[(k_lp, d_lp)], dtype=float)
            r = rmae(arr, acts)
            row += f"  {r:.4f}"
            records.append({"d": d_lp, "k": k_lp, "rMAE": r,
                            "vs_4way": (r/r_ens - 1)*100,
                            "vs_3way": (r/REF_3WAY - 1)*100})
            if r < best_r:
                best_r = r; best_kd = (k_lp, d_lp)
        print(row)

    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(OUT / "lp_on_ensemble4.csv", index=False)

    print(f"\nЛучший: k={best_kd[0]}  d={best_kd[1]}  rMAE={best_r:.4f}")
    print(f"  vs 4-way: {(best_r/r_ens-1)*100:+.2f}%")
    print(f"  vs 3-way: {(best_r/REF_3WAY-1)*100:+.2f}%")
    print(f"  vs Ens2:  {(best_r/REF_ENS2-1)*100:+.2f}%")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    mat = np.array([[rmae(np.array(lp_buf[(k, d)]), acts)
                     for k in K_GRID] for d in D_GRID])
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                   vmin=min(mat.min(), r_ens)*0.995,
                   vmax=max(mat.max(), r_ens)*1.005)
    for i, d_lp in enumerate(D_GRID):
        for j, k_lp in enumerate(K_GRID):
            r = mat[i, j]
            mark = "✓" if r < r_ens else "✗"
            ax.text(j, i, f"{r:.4f}\n{mark}", ha="center", va="center",
                    fontsize=8,
                    color="white" if r > (mat.max()*0.6 + mat.min()*0.4) else "black")
    ax.set_xticks(range(len(K_GRID))); ax.set_xticklabels([str(k) for k in K_GRID])
    ax.set_yticks(range(len(D_GRID))); ax.set_yticklabels([f"d={d}" for d in D_GRID])
    ax.set_xlabel("k_lp"); ax.set_ylabel("d_lp")
    ax.set_title(
        f"LP-коррекция поверх 4-way ансамбля  (сырой={r_ens:.4f})\n"
        f"P_LP={P_LP}  SBER 1d+10m  H={H}  n={n}\n"
        f"✓ = лучше 4-way,  ✗ = хуже"
    )
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "lp_on_ensemble4.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + heatmap → {OUT}")


if __name__ == "__main__":
    run()
