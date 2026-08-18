#!/usr/bin/env python3
"""
01_frac_only_sweep.py

LWR на SBER 10m: пул — ТОЛЬКО фрактальные события T_frac, свип T_frac.
Версия 2: работаем в log-price пространстве (логарифмическая шкала графика).

Цель       : T=4% пивоты (запрос из T_big ряда)
Пул        : только T_frac события, без T_big в пуле
Метка y    : следующий T_frac log-пивот → exp() → цена
Оценка     : pred vs p_big[step+1] → rMAE (в исходных ценах)

Зигзаг на log(price) с аддитивным порогом T:
  log(high) - log(low) >= T  ≡  high/low >= exp(T) ≈ 1 + T (для малых T)

T_BIG задаётся через argv: python script.py 0.04  (default 0.04)
"""
import json
import sys
import numpy as np
from pathlib import Path

HERE  = Path(__file__).parent
DATA  = HERE.parent.parent.parent / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

H           = 1
MIN_HISTORY = 50
P           = 3
K           = 50

T_BIG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04


def _make_grid(t):
    base = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
            0.010, 0.012, 0.015, 0.020, 0.025, 0.030, 0.040]
    grid = sorted(set([round(v, 4) for v in base if v <= t * 1.001] + [t]))
    extras = [round(t * f, 4) for f in (0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85)]
    grid = sorted(set(grid + [e for e in extras if 0.001 <= e <= t]))
    return grid

T_FRAC_GRID = _make_grid(T_BIG)


# ── данные ───────────────────────────────────────────────────────────────────
def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


# ── зигзаг в log-price пространстве ──────────────────────────────────────────
def find_pivots_log(highs, lows, dates, thr):
    """Зигзаг на log(price): аддитивный порог (≈ % в лог-пространстве).
    Возвращает log-цены пивотов и их даты."""
    lh = np.log(highs)
    ll = np.log(lows)
    vals_log, dts = [], []
    direction, ext_val, ext_idx = 0, (lh[0] + ll[0]) / 2.0, 0
    for i in range(len(highs)):
        if direction == 0:
            if lh[i] - ext_val >= thr:
                direction, ext_val, ext_idx = 1, lh[i], i
            elif ext_val - ll[i] >= thr:
                direction, ext_val, ext_idx = -1, ll[i], i
        elif direction == 1:
            if lh[i] > ext_val:
                ext_val, ext_idx = lh[i], i
            elif ext_val - ll[i] >= thr:
                vals_log.append(ext_val); dts.append(dates[ext_idx])
                direction, ext_val, ext_idx = -1, ll[i], i
        else:
            if ll[i] < ext_val:
                ext_val, ext_idx = ll[i], i
            elif lh[i] - ext_val >= thr:
                vals_log.append(ext_val); dts.append(dates[ext_idx])
                direction, ext_val, ext_idx = 1, lh[i], i
    return np.array(vals_log), np.array(dts)


# ── эмбеддинг в log-price пространстве ───────────────────────────────────────
def build_X_log(log_prices, p):
    """col 0 = log(price), col 1+ = простые разности лог-цен (= лог-доходности)."""
    n = len(log_prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = log_prices[i]
        for lag in range(1, p):
            X[i, lag] = log_prices[i - lag + 1] - log_prices[i - lag]
    return X


# ── LWR ──────────────────────────────────────────────────────────────────────
def lwr(xn, Xn, y_log):
    """LWR в нормализованном пространстве. y_log — метки в log-price."""
    d     = np.linalg.norm(Xn - xn, axis=1)
    k_eff = min(K, len(d))
    ord_  = np.argsort(d)
    knn   = ord_[:k_eff]
    xi    = d[ord_[k_eff - 1]]
    if xi < 1e-12:
        return float(y_log[knn].mean()), d.min()
    w  = np.exp(-0.5 * (d[knn] / xi) ** 2)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, y_log[knn] * ws, rcond=None)
    return float(c[0] + c[1:] @ xn), float(d[knn].mean())


def rmae(errs, acts):
    acts = np.asarray(acts, dtype=float)
    pers = np.abs(acts[2:] - acts[:-2]) if len(acts) >= 3 else np.abs(np.diff(acts))
    dz   = float(np.mean(pers)) if len(pers) > 0 else 1.0
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


# ── прогоны ──────────────────────────────────────────────────────────────────
def run_t_only(lp_big, X_big, p_big, n_big):
    """Baseline: пул = T_big события, label = lp_big[j+1] → exp → rMAE."""
    errs, acts, pool_szs = [], [], []
    for step in range(MIN_HISTORY, n_big - H):
        if np.any(np.isnan(X_big[step])):
            continue
        j   = np.arange(P - 1, step)
        v   = ~np.any(np.isnan(X_big[j]), axis=1) & (j + H < n_big)
        idx = j[v]
        if len(idx) < P + 2:
            continue
        X_pool = X_big[idx]
        y_log  = lp_big[idx + H]
        x_q    = X_big[step]
        mu  = X_pool.mean(0)
        sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn  = (X_pool - mu) / sig
        xn  = (x_q    - mu) / sig
        pred_log, _ = lwr(xn, Xn, y_log)
        if np.isnan(pred_log):
            continue
        pred = np.exp(pred_log)
        errs.append(pred - float(p_big[step + H]))
        acts.append(float(p_big[step + H]))
        pool_szs.append(len(idx))
    return np.array(errs), np.array(acts), np.array(pool_szs)


def run_frac_only(lp_big, dt_big, X_big,
                  lp_small, dt_small, X_small,
                  p_big, n_big, n_small):
    """Пул = T_frac события, label = lp_small[j+1] → exp → rMAE."""
    errs, acts, pool_szs, mean_dists = [], [], [], []
    for step in range(MIN_HISTORY, n_big - H):
        if np.any(np.isnan(X_big[step])):
            continue
        ce  = int(np.searchsorted(dt_small, dt_big[step], side="left"))
        j   = np.arange(P - 1, min(ce, n_small - H))
        if len(j) == 0:
            continue
        v   = ~np.any(np.isnan(X_small[j]), axis=1)
        idx = j[v]
        if len(idx) < P + 2:
            continue
        X_pool = X_small[idx]
        y_log  = lp_small[idx + H]
        x_q    = X_big[step]
        mu  = X_pool.mean(0)
        sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn  = (X_pool - mu) / sig
        xn  = (x_q    - mu) / sig
        pred_log, md = lwr(xn, Xn, y_log)
        if np.isnan(pred_log):
            continue
        pred = np.exp(pred_log)
        errs.append(pred - float(p_big[step + H]))
        acts.append(float(p_big[step + H]))
        pool_szs.append(len(idx))
        mean_dists.append(md)
    return (np.array(errs), np.array(acts),
            np.array(pool_szs), np.array(mean_dists))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    h, l, d = load_tf("10m")

    lp_big, dt_big = find_pivots_log(h, l, d, T_BIG)
    p_big = np.exp(lp_big)  # оригинальные цены для rMAE
    n_big = len(lp_big)
    X_big = build_X_log(lp_big, P)

    print(f"SBER 10m  T_big={T_BIG*100:.1f}%  n_big={n_big}  (log-price embedding)")
    print(f"Параметры LWR: p={P}  K={K}  H={H}\n")

    # baseline
    e0, a0, ps0 = run_t_only(lp_big, X_big, p_big, n_big)
    r0 = rmae(e0, a0)
    print(f"{'─'*72}")
    print(f"  baseline  lwr_t_only    rMAE={r0:.4f}   пул avg={ps0.mean():.0f}   n={len(e0)}")
    print(f"{'─'*72}")
    print(f"  {'T_frac':>7}  {'n_small':>7}  {'ratio':>6}  {'rMAE':>7}  {'Δ vs base':>10}  {'pool_avg':>8}  {'mean_d':>8}")
    print(f"  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*10}  {'─'*8}  {'─'*8}")

    results = []
    for T_frac in T_FRAC_GRID:
        lp_sm, dt_sm = find_pivots_log(h, l, d, T_frac)
        n_sm  = len(lp_sm)
        ratio = n_sm / n_big
        X_sm  = build_X_log(lp_sm, P)

        e1, a1, ps1, md1 = run_frac_only(
            lp_big, dt_big, X_big,
            lp_sm, dt_sm, X_sm,
            p_big, n_big, n_sm)

        r1    = rmae(e1, a1) if len(e1) > 0 else np.nan
        delta = (r1 - r0) / r0 * 100 if not np.isnan(r1) else np.nan
        tag   = " ← sanity" if abs(T_frac - T_BIG) < 1e-6 else ""

        print(f"  {T_frac*100:>6.2f}%  {n_sm:>7d}  {ratio:>6.1f}×  "
              f"{r1:>7.4f}  {delta:>+9.1f}%  {ps1.mean():>8.0f}  {md1.mean():>8.3f}{tag}")
        results.append(dict(T_frac=T_frac, n_small=n_sm, ratio=ratio,
                            rMAE=r1, delta=delta,
                            pool_avg=float(ps1.mean()), mean_d=float(md1.mean()),
                            n_steps=len(e1)))

    best = min((r for r in results if not np.isnan(r["rMAE"])),
               key=lambda x: x["rMAE"])
    print(f"\n{'═'*72}")
    print(f"  M0                         rMAE = 1.0000")
    print(f"  baseline lwr_t_only        rMAE = {r0:.4f}   ({(1-r0)*100:.1f}% лучше M0)")
    print(f"  best frac_only  T={best['T_frac']*100:.2f}%   rMAE = {best['rMAE']:.4f}"
          f"   ({(1-best['rMAE'])*100:.1f}% лучше M0)")
    print(f"{'═'*72}")

    # сохранить CSV
    tag = f"t{int(T_BIG*100)}"
    csv_path = RESULTS / f"sweep_{tag}_log.csv"
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nResults → {csv_path}")


if __name__ == "__main__":
    main()
