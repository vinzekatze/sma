#!/usr/bin/env python3
"""
Simplex-abs vs Simplex-log: свип по p.

Simplex (Sugihara 1990):
  k = p+1 ближайших соседей
  w_i = exp(-d_i / d_1)   (d_1 = расстояние до 1-го соседа)
  Simplex-abs: ŷ = Σ w_i·p[j+H]           / Σ w_i
  Simplex-log: ŷ = p_cur·exp(Σ w_i·log(p[j+H]/p[j]) / Σ w_i)

Гипотеза: Simplex-abs катастрофичен на смешанном 1d+10m пуле (усредняет
абс. цены разных масштабов). Simplex-log масштаб-инвариантен и должен
работать аналогично zeroth-order LWR в log-пространстве.

P_GRID = [2, 3, 4, 5, 6, 8]
Эталон: LWR-abs p=3 K=50 → rMAE ≈ 0.4194
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

P_GRID    = [2, 3, 4, 5, 6, 8]
LWR_REF   = 0.4194   # LWR-abs p=3 K=50 (из zigzag_lwr_logret.py)


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
    """X[i] = [prices[i], lr1, lr2, ..., lr(p-1)]"""
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
    n1d  = len(p1d)
    na   = len(p10m)
    k    = p + 1   # Simplex: ровно p+1 соседей
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")

    preds_abs = []
    preds_log = []
    actuals   = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X1d[step])):
            continue

        j1d = np.arange(p - 1, step)
        v1d = ~np.any(np.isnan(X1d[j1d]), axis=1) & (j1d + H < n1d)
        idx1d  = j1d[v1d]
        Xp     = list(X1d[idx1d])
        yp_abs = list(p1d[idx1d + H])
        yp_src = list(p1d[idx1d])

        ce  = int(ce10m_all[step])
        j10 = np.arange(p - 1, min(ce, na - H))
        if len(j10):
            v10   = ~np.any(np.isnan(X10m[j10]), axis=1)
            idx10 = j10[v10]
            Xp.extend(X10m[idx10])
            yp_abs.extend(p10m[idx10 + H])
            yp_src.extend(p10m[idx10])

        if len(Xp) < k:
            preds_abs.append(np.nan)
            preds_log.append(np.nan)
            actuals.append(float(p1d[step + H]))
            continue

        X_pool = np.array(Xp)
        y_abs  = np.array(yp_abs, dtype=float)
        y_src  = np.array(yp_src, dtype=float)
        y_lr   = np.log(y_abs / y_src)   # log-доходность для каждой точки пула

        x_q   = X1d[step]
        p_cur = float(p1d[step])

        # Z-score нормализация для расстояний
        mu  = X_pool.mean(0)
        sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn  = (X_pool - mu) / sig
        xn  = (x_q    - mu) / sig

        dists = np.linalg.norm(Xn - xn, axis=1)
        order = np.argsort(dists)
        knn   = order[:k]

        d1 = dists[order[0]]
        if d1 < 1e-12:
            # Точное совпадение: ближайший сосед — прямой ответ
            preds_abs.append(float(y_abs[order[0]]))
            preds_log.append(float(p_cur * np.exp(y_lr[order[0]])))
        else:
            w  = np.exp(-dists[knn] / d1)
            ws = w / w.sum()

            preds_abs.append(float(ws @ y_abs[knn]))
            lr_hat = float(ws @ y_lr[knn])
            preds_log.append(float(p_cur * np.exp(lr_hat)))

        actuals.append(float(p1d[step + H]))

    acts = np.array(actuals)
    return rmae(preds_abs, acts), rmae(preds_log, acts), len(acts)


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m = find_pivots(h10m, l10m, d10m, T_10M)

    print(f"SBER 1d {len(p1d)} пив  10m {len(p10m)} пив")
    print(f"H={H}  Simplex k=p+1  P_GRID={P_GRID}")
    print(f"Эталон: LWR-abs p=3 K=50 → {LWR_REF:.4f}\n")

    print(f"{'p':>3}  {'k':>3}  {'Simplex-abs':>12}  "
          f"{'vs_LWR%':>8}  {'Simplex-log':>12}  {'vs_LWR%':>8}  "
          f"{'Δlog-abs':>9}  winner")
    print("-" * 78)

    results = []
    for p in P_GRID:
        r_abs, r_log, n = run_p(p, p1d, p10m, dt1d, dt10m)
        k = p + 1
        vs_abs = (r_abs / LWR_REF - 1) * 100 if not np.isnan(r_abs) else np.nan
        vs_log = (r_log / LWR_REF - 1) * 100 if not np.isnan(r_log) else np.nan
        delta  = r_log - r_abs if not (np.isnan(r_abs) or np.isnan(r_log)) else np.nan
        winner = ("log" if r_log < r_abs else "abs") if not np.isnan(delta) else "?"

        def fmt(v): return f"{v:.4f}" if not np.isnan(v) else "  nan "
        def fmtp(v): return f"{v:+.1f}%" if not np.isnan(v) else "  nan"
        def fmtd(v): return f"{v:+.4f}" if not np.isnan(v) else "  nan"

        print(f"{p:>3}  {k:>3}  {fmt(r_abs):>12}  {fmtp(vs_abs):>8}  "
              f"{fmt(r_log):>12}  {fmtp(vs_log):>8}  {fmtd(delta):>9}  {winner}")
        results.append((p, k, r_abs, r_log))

    # ── График ────────────────────────────────────────────────────────────────
    ps      = [r[0] for r in results]
    r_abs_l = [r[2] for r in results]
    r_log_l = [r[3] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ps, r_abs_l, "o-",  label="Simplex-abs", color="steelblue", lw=1.8)
    ax.plot(ps, r_log_l, "s--", label="Simplex-log", color="tomato",    lw=1.8)
    ax.axhline(LWR_REF, color="green", linestyle=":", lw=1.5,
               label=f"LWR-abs p=3 K=50 = {LWR_REF:.4f}")
    ax.set_xlabel("p (размерность признаков)")
    ax.set_ylabel("rMAE")
    ax.set_title(
        f"Simplex-abs vs Simplex-log  |  k=p+1  |  SBER 1d+10m  H={H}\n"
        f"Признаки: [price, lr1, ..., lr(p-1)]"
    )
    ax.legend(); ax.grid(True, alpha=0.3)
    # Аннотации k
    for p_val, ra, rl in zip(ps, r_abs_l, r_log_l):
        for r, col, dy in [(ra, "steelblue", 8), (rl, "tomato", -14)]:
            if not np.isnan(r):
                ax.annotate(f"k={p_val+1}", (p_val, r),
                            textcoords="offset points", xytext=(0, dy),
                            ha="center", fontsize=7.5, color=col)
    fig.tight_layout()
    fig.savefig(OUT / "simplex_logret.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {OUT}/simplex_logret.png")


if __name__ == "__main__":
    run()
