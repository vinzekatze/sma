#!/usr/bin/env python3
"""
LWR с метрикой amp_cos: свип по alpha. SBER 1d, T=2%, p=3, K=12, H=1.

МЕТРИКА (поиск соседей):
  d(x,q) = alpha·|log(‖x‖/‖q‖)|_norm + (1−alpha)·(1−cos(x,q))_norm
WLS-регрессия — по z-score нормированным признакам пула (как в стандартном LWR).
Baseline: стандартный LWR (L2 после z-score нормализации).

КОНТРАКТ КАУЗАЛЬНОСТИ: X_pool строго [0..step-1], нет look-ahead.
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

T_ZZ        = 0.02
P           = 3
K           = 3 * (P + 1)   # 12
H           = 1
MIN_HISTORY = 50
A_GRID      = np.round(np.linspace(0.0, 1.0, 11), 2)


def load_1d():
    with open(DATA / "1d.json") as f:
        data = json.load(f)
    return (np.array([d["high"] for d in data], dtype=np.float64),
            np.array([d["low"]  for d in data], dtype=np.float64))


def find_pivots(highs, lows, thr):
    pivots    = []
    direction = 0
    ext_val   = (highs[0] + lows[0]) / 2.0
    ext_idx   = 0
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
                pivots.append(ext_val)
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                pivots.append(ext_val)
                direction = 1;  ext_val, ext_idx = highs[i], i
    return np.array(pivots)


def build_X(prices, p):
    n  = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def _cosine_dist(X_pool, x_q):
    nX = np.linalg.norm(X_pool, axis=1)
    nq = float(np.linalg.norm(x_q))
    if nq < 1e-10:
        return np.ones(len(X_pool))
    with np.errstate(invalid='ignore', divide='ignore'):
        sim = np.where(nX > 1e-10, (X_pool @ x_q) / (nX * nq), 0.0)
    return 1.0 - sim.clip(-1.0, 1.0)


def amp_cos_dists(X_pool, x_q, alpha):
    """app6: d_amp = |log(‖x‖/‖q‖)|, d_shape = cosine."""
    nX = np.linalg.norm(X_pool, axis=1)
    nq = float(np.linalg.norm(x_q))
    with np.errstate(divide='ignore', invalid='ignore'):
        d_amp = np.where(
            (nX > 1e-10) & (nq > 1e-10),
            np.abs(np.log(nX / nq)),
            np.abs(nX - nq),
        )
    d_shape = _cosine_dist(X_pool, x_q)
    max_a = max(float(d_amp.max()), 1e-10)
    max_s = max(float(d_shape.max()), 1e-10)
    return float(alpha) * (d_amp / max_a) + (1.0 - float(alpha)) * (d_shape / max_s)


def blend_dists(X_pool, x_q, alpha):
    """app5: d_amp = L2 (сырой), d_shape = cosine."""
    d_amp   = np.linalg.norm(X_pool - x_q, axis=1)
    d_shape = _cosine_dist(X_pool, x_q)
    max_a = max(float(d_amp.max()), 1e-10)
    max_s = max(float(d_shape.max()), 1e-10)
    return float(alpha) * (d_amp / max_a) + (1.0 - float(alpha)) * (d_shape / max_s)


def split_dists(X_pool, x_q, alpha):
    """d_amp = |log(price_i/price_j)| по X[:,0]; d_shape = cos только по X[:,1:] (лог-доходности)."""
    p0_pool = X_pool[:, 0]
    p0_q    = float(x_q[0])
    with np.errstate(divide='ignore', invalid='ignore'):
        d_amp = np.where(
            (p0_pool > 1e-10) & (p0_q > 1e-10),
            np.abs(np.log(p0_pool / p0_q)),
            np.abs(p0_pool - p0_q),
        )
    if X_pool.shape[1] < 2:
        d_shape = np.zeros(len(X_pool))
    else:
        d_shape = _cosine_dist(X_pool[:, 1:], x_q[1:])
    max_a = max(float(d_amp.max()), 1e-10)
    max_s = max(float(d_shape.max()), 1e-10)
    return float(alpha) * (d_amp / max_a) + (1.0 - float(alpha)) * (d_shape / max_s)


def split_norm_dists(X_pool, x_q, alpha):
    """Z-score нормировка пула, затем split: d_amp по норм. цене, d_shape по косинусу норм. лог-доходностей."""
    mu    = X_pool.mean(0)
    sigma = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
    Xn    = (X_pool - mu) / sigma
    xn    = (x_q    - mu) / sigma
    d_amp = np.abs(Xn[:, 0] - xn[0])
    if Xn.shape[1] < 2:
        d_shape = np.zeros(len(Xn))
    else:
        d_shape = _cosine_dist(Xn[:, 1:], xn[1:])
    max_a = max(float(d_amp.max()), 1e-10)
    max_s = max(float(d_shape.max()), 1e-10)
    return float(alpha) * (d_amp / max_a) + (1.0 - float(alpha)) * (d_shape / max_s)


def lwr_l2(X_pool, y_pool, x_q, k):
    if len(X_pool) < k:
        return np.nan
    mu    = X_pool.mean(0)
    sigma = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
    Xn    = (X_pool - mu) / sigma
    xn    = (x_q    - mu) / sigma
    dists = np.linalg.norm(Xn - xn, axis=1)
    order = np.argsort(dists)
    knn   = order[:k]
    xi    = dists[order[k - 1]]
    if xi < 1e-12:
        return float(y_pool[knn].mean())
    w   = np.exp(-0.5 * (dists[knn] / xi) ** 2)
    ws  = np.sqrt(w)
    A   = np.column_stack([np.ones(k), Xn[knn]]) * ws[:, None]
    b   = y_pool[knn] * ws
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0] + coef[1:] @ xn)


def lwr_custom(X_pool, y_pool, x_q, k, alpha, metric="amp_cos"):
    """NN ищется по custom metric; WLS по z-score нормализованному пулу."""
    if len(X_pool) < k:
        return np.nan
    if metric == "amp_cos":
        dists = amp_cos_dists(X_pool, x_q, alpha)
    elif metric == "blend":
        dists = blend_dists(X_pool, x_q, alpha)
    elif metric == "split":
        dists = split_dists(X_pool, x_q, alpha)
    else:
        dists = split_norm_dists(X_pool, x_q, alpha)
    order = np.argsort(dists)
    knn   = order[:k]
    xi    = dists[order[k - 1]]
    if xi < 1e-12:
        return float(y_pool[knn].mean())
    w   = np.exp(-0.5 * (dists[knn] / xi) ** 2)
    ws  = np.sqrt(w)
    mu    = X_pool.mean(0)
    sigma = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
    Xn    = (X_pool - mu) / sigma
    xn    = (x_q    - mu) / sigma
    A   = np.column_stack([np.ones(k), Xn[knn]]) * ws[:, None]
    b   = y_pool[knn] * ws
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0] + coef[1:] @ xn)


def walk_forward(prices, X, alpha=None, metric="amp_cos"):
    """alpha=None → L2 baseline."""
    n = len(prices)
    preds, actuals = [], []
    for step in range(max(MIN_HISTORY, P), n - H):
        if np.any(np.isnan(X[step])):
            continue
        j_arr = np.arange(P - 1, step)
        valid  = ~np.any(np.isnan(X[j_arr]), axis=1) & (j_arr + H < n)
        Xp = X[j_arr[valid]]
        yp = prices[j_arr[valid] + H]
        if len(Xp) < K:
            continue
        pred = (lwr_l2(Xp, yp, X[step], K) if alpha is None
                else lwr_custom(Xp, yp, X[step], K, alpha, metric))
        if np.isnan(pred):
            continue
        preds.append(pred)
        actuals.append(prices[step + H])
    return np.array(preds), np.array(actuals)


def rmae(preds, actuals):
    if len(preds) < 5:
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return np.nan if dz < 1e-12 else float(np.mean(np.abs(preds - actuals)) / dz)


def run():
    highs, lows = load_1d()
    prices = find_pivots(highs, lows, T_ZZ)
    X = build_X(prices, P)

    print(f"SBER 1d  T={T_ZZ*100:.0f}%  n_пивотов={len(prices)}")
    print(f"p={P}  K={K}  H={H}")
    print()

    pr0, ac0 = walk_forward(prices, X, alpha=None)
    r_l2 = rmae(pr0, ac0)
    print(f"L2 baseline:  rMAE={r_l2:.4f}  n={len(pr0)}")
    print()
    print(f"  {'alpha':>5}  {'rMAE':>8}  {'vs_L2':>8}  n")
    print("  " + "─" * 32)

    records = [{"metric": "L2", "alpha": None, "rMAE": r_l2, "vs_L2_pct": 0.0, "n_test": len(pr0)}]
    results = {}  # metric → {alpha → rMAE}

    for metric_name, color in [("amp_cos", "steelblue"), ("blend", "darkorange"), ("split", "seagreen"), ("split_norm", "crimson")]:
        print(f"\n  {'alpha':>5}  {'rMAE':>8}  {'vs_L2':>8}   ({metric_name})")
        print("  " + "─" * 38)
        best_r, best_a = r_l2, None
        res = {}
        for alpha in A_GRID:
            pr, ac = walk_forward(prices, X, alpha=alpha, metric=metric_name)
            r = rmae(pr, ac)
            vs = (r / r_l2 - 1) * 100 if not np.isnan(r_l2) else np.nan
            mark = " ←" if r < best_r else ""
            print(f"  α={alpha:.1f}  {r:.4f}   {vs:+6.1f}%{mark}")
            if r < best_r:
                best_r, best_a = r, alpha
            res[float(alpha)] = r
            records.append({"metric": metric_name, "alpha": alpha,
                            "rMAE": r, "vs_L2_pct": vs, "n_test": len(pr)})
        results[metric_name] = res
        if best_a is not None:
            print(f"  → лучший α={best_a:.1f}  rMAE={best_r:.4f}  vs L2: {(best_r/r_l2-1)*100:+.1f}%")
        else:
            print(f"  → L2 лучше всех α")

    df = pd.DataFrame(records)
    df.to_csv(OUT / "lwr_ampcos_alpha_sweep.csv", index=False)

    # ── График ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    alphas = list(A_GRID)
    for metric_name, color, label in [
        ("amp_cos",    "steelblue",  "amp_cos    (d_amp=log‖x‖/‖q‖)"),
        ("blend",      "darkorange", "blend      (d_amp=L2)"),
        ("split",      "seagreen",   "split      (d_amp=log price, d_shape=cos log-ret)"),
        ("split_norm", "crimson",    "split_norm (z-score, d_amp=|Δprice_norm|, d_shape=cos log-ret_norm)"),
    ]:
        ys = [results[metric_name][a] for a in alphas]
        ax.plot(alphas, ys, "o-", color=color, lw=2, ms=7, label=label)

    ax.axhline(r_l2, color="gray", lw=1.5, ls="--",
               label=f"L2 baseline  rMAE={r_l2:.4f}")
    ax.set_xlabel("α  (0 = косинус, 1 = амплитуда)")
    ax.set_ylabel("rMAE")
    ax.set_title(f"LWR custom metric: свип по α  |  SBER 1d T=2%,  p={P}, K={K}, H=1")
    ax.set_xticks(A_GRID)
    ax.legend()
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(OUT / "lwr_ampcos_alpha_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + график: {OUT}")


if __name__ == "__main__":
    run()
