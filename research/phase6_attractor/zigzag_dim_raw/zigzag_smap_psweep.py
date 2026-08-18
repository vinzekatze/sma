#!/usr/bin/env python3
"""
S-map θ=1: свип p. Проверяем оптимальную размерность вложения.

S-map: все точки пула, веса exp(-θ·d/mean_d), degree-1 WLS.
θ=1 зафиксирован (лучший в 3-way и 4-way ансамбле).
P_GRID = [2, 3, 4, 5, 6, 8, 10, 12]

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
THETA       = 1.0

P_GRID = [2, 3, 4, 5, 6, 8, 10, 12]

REF_LWR  = 0.4194
REF_SMAP = 0.5442   # S-map p=3 θ=1 standalone


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

    preds   = []
    actuals = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X1d[step])):
            continue

        ce  = int(ce10m_all[step])
        j1d = np.arange(p - 1, step)
        v1d = ~np.any(np.isnan(X1d[j1d]), axis=1) & (j1d + H < n1d)
        idx1 = j1d[v1d]
        Xp   = list(X1d[idx1]);  yp = list(p1d[idx1 + H])

        j10 = np.arange(p - 1, min(ce, na - H))
        if len(j10):
            v10  = ~np.any(np.isnan(X10m[j10]), axis=1)
            idx10 = j10[v10]
            Xp.extend(X10m[idx10]);  yp.extend(p10m[idx10 + H])

        y_sm = np.nan
        if len(Xp) >= p + 2:
            X_pool = np.array(Xp); y_abs = np.array(yp)
            x_q    = X1d[step]
            mu  = X_pool.mean(0)
            sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
            Xn  = (X_pool - mu) / sig
            xn  = (x_q    - mu) / sig
            d   = np.linalg.norm(Xn - xn, axis=1)
            N   = len(y_abs)
            mean_d = d.mean() + 1e-12
            w_sm   = np.exp(-THETA * d / mean_d); ws_sm = np.sqrt(w_sm)
            A_sm   = np.column_stack([np.ones(N), Xn]) * ws_sm[:, None]
            c_sm, *_ = np.linalg.lstsq(A_sm, y_abs * ws_sm, rcond=None)
            y_sm = float(c_sm[0] + c_sm[1:] @ xn)

        preds.append(y_sm)
        actuals.append(float(p1d[step + H]))

    return rmae(preds, np.array(actuals)), len(actuals)


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m = find_pivots(h10m, l10m, d10m, T_10M)

    print(f"SBER 1d {len(p1d)} пив  10m {len(p10m)} пив")
    print(f"S-map θ={THETA}  P_GRID={P_GRID}")
    print(f"Эталоны: LWR(p=3,K=50)={REF_LWR:.4f}  "
          f"S-map(p=3,θ=1) standalone={REF_SMAP:.4f}\n")

    print(f"{'p':>3}  {'rMAE':>7}  {'vs p=3':>8}  {'vs LWR':>8}")
    print("-" * 34)

    results = []
    r_p3 = None
    for p in P_GRID:
        r, n = run_p(p, p1d, p10m, dt1d, dt10m)
        results.append((p, r))
        if p == 3:
            r_p3 = r
        vs_p3 = f"{(r/r_p3-1)*100:+.1f}%" if r_p3 else "  —"
        print(f"{p:>3}  {r:>7.4f}  {vs_p3:>8}  {(r/REF_LWR-1)*100:>+7.1f}%")

    best_p, best_r = min(results, key=lambda x: x[1])
    print(f"\nЛучший: p={best_p}  rMAE={best_r:.4f}")

    # График
    ps = [r[0] for r in results]
    rs = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ps, rs, "o-", color="orange", lw=1.8, label=f"S-map θ={THETA}")
    ax.axhline(REF_LWR,  color="steelblue", ls=":", lw=1.2,
               label=f"LWR-abs p=3={REF_LWR:.4f}")
    ax.scatter([best_p], [best_r], color="red", zorder=5, s=70)
    ax.annotate(f"p={best_p}\n{best_r:.4f}", (best_p, best_r),
                textcoords="offset points", xytext=(8, 4),
                fontsize=9, color="red")
    ax.set_xlabel("p"); ax.set_ylabel("rMAE (standalone)")
    ax.set_title(f"S-map θ={THETA}: свип p  |  SBER 1d+10m  H={H}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "smap_psweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"График → {OUT}/smap_psweep.png")


if __name__ == "__main__":
    run()
