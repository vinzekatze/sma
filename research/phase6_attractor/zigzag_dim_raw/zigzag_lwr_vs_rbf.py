#!/usr/bin/env python3
"""
Сравнение LWR и RBF на зигзаге SBER.

LWR (baseline): k ближайших по z-score L2 → WLS с Гауссовыми весами →
  ŷ = c₀ + c₁ᵀ·x_q  (линейная локальная модель).

RBF (app3): те же k соседей → интерполяция радиально-базисными функциями:
  Φ[i,j] = exp(−‖xᵢ−xⱼ‖² / 2σ²),  σ = dist до k-го соседа
  w = (Φ + λI)⁻¹ · y
  ŷ = Σ wᵢ · exp(−‖xᵢ−x_q‖² / 2σ²)

Оба метода работают в z-score нормированном пространстве пула.

Параметры (оптимум из предыдущих экспериментов):
  T_1d=4%, T_10m=0.4%, p=3, K=12, H=1.

Дополнительно: свип K, чтобы проверить, какое число соседей оптимально
для каждого метода.

КАУЗАЛЬНОСТЬ: пул строго j < step, aux — до даты step.
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
K_GRID      = [6, 9, 12, 15, 18, 24, 30]
RBF_LAM     = 1e-6   # коэффициент регуляризации


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


def _zscore(X_pool, x_q):
    mu    = X_pool.mean(0)
    sigma = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
    return (X_pool - mu) / sigma, (x_q - mu) / sigma


def _knn(Xn, xn, k):
    dists = np.linalg.norm(Xn - xn, axis=1)
    order = np.argsort(dists)
    return order[:k], dists[order[k - 1]]


def predict_lwr(X_pool, y_pool, x_q, k):
    if len(X_pool) < k:
        return np.nan
    Xn, xn = _zscore(X_pool, x_q)
    knn, xi = _knn(Xn, xn, k)
    if xi < 1e-12:
        return float(y_pool[knn].mean())
    w   = np.exp(-0.5 * (np.linalg.norm(Xn[knn] - xn, axis=1) / xi) ** 2)
    ws  = np.sqrt(w)
    A   = np.column_stack([np.ones(k), Xn[knn]]) * ws[:, None]
    b   = y_pool[knn] * ws
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0] + coef[1:] @ xn)


def predict_nw(X_pool, y_pool, x_q, k):
    """Nadaraya-Watson: взвешенное среднее (LWR степень 0)."""
    if len(X_pool) < k:
        return np.nan
    Xn, xn = _zscore(X_pool, x_q)
    knn, xi = _knn(Xn, xn, k)
    if xi < 1e-12:
        return float(y_pool[knn].mean())
    w = np.exp(-0.5 * (np.linalg.norm(Xn[knn] - xn, axis=1) / xi) ** 2)
    return float((w * y_pool[knn]).sum() / (w.sum() + 1e-12))


def predict_rbf(X_pool, y_pool, x_q, k):
    """RBF интерполяция в z-score пространстве (per app3._rbf_approx)."""
    if len(X_pool) < k:
        return np.nan
    Xn, xn = _zscore(X_pool, x_q)
    knn, xi = _knn(Xn, xn, k)
    if xi < 1e-12:
        return float(y_pool[knn].mean())
    Xk = Xn[knn]
    yk = y_pool[knn]
    # Φ[i,j] = exp(−‖xᵢ−xⱼ‖² / 2σ²)
    diff = Xk[:, None, :] - Xk[None, :, :]       # (k, k, p)
    Phi  = np.exp(-0.5 * np.sum(diff ** 2, axis=2) / xi ** 2)
    lam  = RBF_LAM * (np.trace(Phi) / k + 1e-12)
    w, *_ = np.linalg.lstsq(Phi + lam * np.eye(k), yk, rcond=None)
    phi_q = np.exp(-0.5 * np.sum((Xk - xn) ** 2, axis=1) / xi ** 2)
    return float(phi_q @ w)


def walk_forward(prices_prim, dates_prim, X_prim,
                 prices_aux, dates_aux, X_aux, k, method="lwr"):
    fn = {"lwr": predict_lwr, "nw": predict_nw, "rbf": predict_rbf}[method]
    n  = len(prices_prim)
    preds, actuals = [], []

    for step in range(MIN_HISTORY, n - H):
        if np.any(np.isnan(X_prim[step])):
            continue
        target = prices_prim[step + H]

        j_arr  = np.arange(P - 1, step)
        valid  = ~np.any(np.isnan(X_prim[j_arr]), axis=1) & (j_arr + H < n)
        X_rows = list(X_prim[j_arr[valid]])
        y_rows = list(prices_prim[j_arr[valid] + H])

        if prices_aux is not None:
            ce = int(np.searchsorted(dates_aux, dates_prim[step], side="left"))
            na = len(prices_aux)
            j_a = np.arange(P - 1, min(ce, na - H))
            if len(j_a):
                vv = ~np.any(np.isnan(X_aux[j_a]), axis=1)
                X_rows.extend(X_aux[j_a[vv]])
                y_rows.extend(prices_aux[j_a[vv] + H])

        if len(X_rows) < k:
            continue

        pred = fn(np.array(X_rows), np.array(y_rows), X_prim[step], k)
        if np.isnan(pred):
            continue
        preds.append(pred)
        actuals.append(target)

    return np.array(preds), np.array(actuals)


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

    print(f"SBER  1d  T={T_1D *100:.0f}%  → {len(p1d )} пивотов")
    print(f"SBER  10m T={T_10M*100:.1f}% → {len(p10m)} пивотов")
    print(f"p={P}  H={H}  RBF_λ={RBF_LAM}")
    print()

    records = []

    # ── Свип K, три метода ────────────────────────────────────────────────────
    print(f"{'K':>4}  {'LWR':>8}  {'NW':>8}  {'RBF':>8}  "
          f"{'NW-LWR':>8}  {'RBF-LWR':>9}")
    print("─" * 58)

    for k in K_GRID:
        pr_l, ac_l = walk_forward(p1d, dates_1d, X_1d,
                                   p10m, dates_10m, X_10m, k, "lwr")
        pr_n, ac_n = walk_forward(p1d, dates_1d, X_1d,
                                   p10m, dates_10m, X_10m, k, "nw")
        pr_r, ac_r = walk_forward(p1d, dates_1d, X_1d,
                                   p10m, dates_10m, X_10m, k, "rbf")
        rl = rmae(pr_l, ac_l)
        rn = rmae(pr_n, ac_n)
        rr = rmae(pr_r, ac_r)
        vs_n = f"{(rn/rl-1)*100:+.1f}%" if not np.isnan(rl) else "n/a"
        vs_r = f"{(rr/rl-1)*100:+.1f}%" if not np.isnan(rl) else "n/a"
        print(f"  K={k:2d}  {rl:.4f}   {rn:.4f}   {rr:.4f}   "
              f"{vs_n:>8}  {vs_r:>9}")
        records.append({"K": k, "rMAE_lwr": rl, "rMAE_nw": rn, "rMAE_rbf": rr,
                        "n": len(pr_l)})

    df = pd.DataFrame(records)
    df.to_csv(OUT / "lwr_vs_rbf.csv", index=False)

    # ── График ────────────────────────────────────────────────────────────────
    ks = df["K"].values
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(ks, df["rMAE_lwr"], "o-",  color="steelblue",  lw=2, ms=8, label="LWR (линейный)")
    ax.plot(ks, df["rMAE_nw"],  "s--", color="darkorange",  lw=2, ms=8, label="NW  (взв. среднее)")
    ax.plot(ks, df["rMAE_rbf"], "^:",  color="crimson",     lw=2, ms=8, label="RBF (интерполяция)")
    ax.set_xlabel("K (число соседей)")
    ax.set_ylabel("rMAE")
    ax.set_title("rMAE vs K")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1]
    vs_nw  = [(rn / rl - 1) * 100 for rl, rn in zip(df["rMAE_lwr"], df["rMAE_nw"])]
    vs_rbf = [(rr / rl - 1) * 100 for rl, rr in zip(df["rMAE_lwr"], df["rMAE_rbf"])]
    w = 1.8
    ax.bar([k - w/2 for k in ks], vs_nw,  width=w, color="darkorange", alpha=0.8, label="NW vs LWR")
    ax.bar([k + w/2 for k in ks], vs_rbf, width=w, color="crimson",    alpha=0.8, label="RBF vs LWR")
    ax.axhline(0, color="gray", lw=1)
    ax.set_xlabel("K (число соседей)")
    ax.set_ylabel("% изменение rMAE vs LWR")
    ax.set_title("NW и RBF vs LWR (отриц. = лучше LWR)")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"LWR vs RBF на зигзаге  |  SBER 1d({T_1D*100:.0f}%) + 10m({T_10M*100:.1f}%)\n"
        f"X=[price, lr₁, lr₂]  p={P}  H={H}  λ={RBF_LAM}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(OUT / "lwr_vs_rbf.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + график → {OUT}")


if __name__ == "__main__":
    run()
