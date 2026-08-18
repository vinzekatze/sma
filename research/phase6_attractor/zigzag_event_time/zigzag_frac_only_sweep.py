#!/usr/bin/env python3
"""
zigzag_frac_only_sweep.py

LWR на SBER 10m: пул — ТОЛЬКО фрактальные события T_frac, свип T_frac.

Цель       : T=4% пивоты (запрос из T_big ряда)
Пул        : только T_frac события, без T_big в пуле
Метка y    : следующий T_frac пивот (p_small[j+1], H=1 в фрактальном времени)
Оценка     : pred vs p_big[step+1] → rMAE

Базовая линия: LWR только T_big в пуле (label = p_big[j+1]).

Ожидаемый эффект масштаба:
  query col 1,2 ≈ ±4%, pool col 1,2 ≈ ±T_frac%
  После joint z-score запрос — outlier по log-return измерениям.
  При малых T_frac соседи выбираются почти только по цене (col 0).
  При T_frac=0.04 (sanity check) результат должен совпасть с baseline.
"""
import json
import numpy as np
from pathlib import Path

HERE  = Path(__file__).parent
DATA  = HERE.parent.parent.parent / "data" / "candles" / "SBER"

H           = 1
MIN_HISTORY = 50
P           = 3
K           = 50

# T_BIG задаётся через argv: python script.py 0.04  (default 0.04)
import sys as _sys
T_BIG = float(_sys.argv[1]) if len(_sys.argv) > 1 else 0.04

# сетка: от T_BIG/20 до T_BIG (sanity), ~14 точек
def _make_grid(t):
    base = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
            0.010, 0.012, 0.015, 0.020, 0.025, 0.030, 0.040]
    grid = sorted(set([round(v, 4) for v in base if v <= t * 1.001] + [t]))
    # добавляем дробные шаги ниже t
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


def find_pivots(highs, lows, dates, thr):
    vals, dts = [], []
    direction = 0
    ext_val = (highs[0] + lows[0]) / 2.0
    ext_idx = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1; ext_val, ext_idx = highs[i], i
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
    return np.array(vals), np.array(dts)


def build_X(prices, p):
    n  = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


# ── LWR ──────────────────────────────────────────────────────────────────────
def lwr(xn, Xn, y_abs):
    d     = np.linalg.norm(Xn - xn, axis=1)
    k_eff = min(K, len(d))
    ord_  = np.argsort(d)
    knn   = ord_[:k_eff]
    xi    = d[ord_[k_eff - 1]]
    if xi < 1e-12:
        return float(y_abs[knn].mean()), d.min()
    w  = np.exp(-0.5 * (d[knn] / xi) ** 2); ws = np.sqrt(w)
    A  = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, y_abs[knn] * ws, rcond=None)
    return float(c[0] + c[1:] @ xn), float(d[knn].mean())


def rmae(errs, acts):
    dz = float(np.mean(np.abs(np.diff(acts))))
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


# ── прогоны ──────────────────────────────────────────────────────────────────
def run_t4_only(p_big, X_big, n_big):
    """Baseline: пул = T=4% события, label = следующий T=4% пивот."""
    errs, acts, pool_szs = [], [], []
    for step in range(MIN_HISTORY, n_big - H):
        if np.any(np.isnan(X_big[step])):
            continue
        j    = np.arange(P - 1, step)
        v    = ~np.any(np.isnan(X_big[j]), axis=1) & (j + H < n_big)
        idx  = j[v]
        if len(idx) < P + 2:
            continue
        X_pool = X_big[idx]
        y_abs  = p_big[idx + H]
        x_q    = X_big[step]
        mu  = X_pool.mean(0)
        sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn  = (X_pool - mu) / sig
        xn  = (x_q    - mu) / sig
        pred, _ = lwr(xn, Xn, y_abs)
        if np.isnan(pred):
            continue
        errs.append(pred - float(p_big[step + H]))
        acts.append(float(p_big[step + H]))
        pool_szs.append(len(idx))
    return np.array(errs), np.array(acts), np.array(pool_szs)


def run_frac_only(p_big, dt_big, X_big, p_small, dt_small, X_small, n_big, n_small):
    """Пул = только T_frac события, label = следующий T_frac пивот."""
    errs, acts, pool_szs, mean_dists = [], [], [], []
    for step in range(MIN_HISTORY, n_big - H):
        if np.any(np.isnan(X_big[step])):
            continue
        ce   = int(np.searchsorted(dt_small, dt_big[step], side="left"))
        j    = np.arange(P - 1, min(ce, n_small - H))
        if len(j) == 0:
            continue
        v    = ~np.any(np.isnan(X_small[j]), axis=1)
        idx  = j[v]
        if len(idx) < P + 2:
            continue
        X_pool = X_small[idx]
        y_abs  = p_small[idx + H]
        x_q    = X_big[step]
        mu  = X_pool.mean(0)
        sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn  = (X_pool - mu) / sig
        xn  = (x_q    - mu) / sig
        pred, md = lwr(xn, Xn, y_abs)
        if np.isnan(pred):
            continue
        errs.append(pred - float(p_big[step + H]))
        acts.append(float(p_big[step + H]))
        pool_szs.append(len(idx))
        mean_dists.append(md)
    return (np.array(errs), np.array(acts),
            np.array(pool_szs), np.array(mean_dists))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    h, l, d = load_tf("10m")
    p_big, dt_big = find_pivots(h, l, d, T_BIG)
    n_big = len(p_big)
    X_big = build_X(p_big, P)

    print(f"SBER 10m  T_big={T_BIG*100:.1f}%  n_big={n_big}")
    print(f"Параметры LWR: p={P}  K={K}  H={H}\n")

    # baseline
    e0, a0, ps0 = run_t4_only(p_big, X_big, n_big)
    r0 = rmae(e0, a0)
    print(f"{'─'*72}")
    print(f"  baseline  lwr_t4_only    rMAE={r0:.4f}   пул avg={ps0.mean():.0f}   n={len(e0)}")
    print(f"{'─'*72}")
    print(f"  {'T_frac':>7}  {'n_small':>7}  {'ratio':>6}  {'rMAE':>7}  {'Δ vs base':>10}  {'pool_avg':>8}  {'mean_d':>8}")
    print(f"  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*10}  {'─'*8}  {'─'*8}")

    results = []
    for T_frac in T_FRAC_GRID:
        p_sm, dt_sm = find_pivots(h, l, d, T_frac)
        n_sm  = len(p_sm)
        ratio = n_sm / n_big
        X_sm  = build_X(p_sm, P)

        e1, a1, ps1, md1 = run_frac_only(
            p_big, dt_big, X_big, p_sm, dt_sm, X_sm, n_big, n_sm)

        r1   = rmae(e1, a1) if len(e1) > 0 else np.nan
        delta = (r1 - r0) / r0 * 100 if not np.isnan(r1) else np.nan
        tag  = " ← sanity" if abs(T_frac - T_BIG) < 1e-6 else ""

        print(f"  {T_frac*100:>6.2f}%  {n_sm:>7d}  {ratio:>6.1f}×  "
              f"{r1:>7.4f}  {delta:>+9.1f}%  {ps1.mean():>8.0f}  {md1.mean():>8.3f}{tag}")
        results.append(dict(T_frac=T_frac, n_small=n_sm, ratio=ratio,
                            rMAE=r1, delta=delta,
                            pool_avg=ps1.mean(), mean_d=md1.mean(), n_steps=len(e1)))

    best = min((r for r in results if not np.isnan(r["rMAE"])),
               key=lambda x: x["rMAE"])
    print(f"\n{'═'*72}")
    print(f"  M0                         rMAE = 1.0000")
    print(f"  baseline lwr_t4_only       rMAE = {r0:.4f}   ({(1-r0)*100:.1f}% лучше M0)")
    print(f"  best frac_only  T={best['T_frac']*100:.2f}%   rMAE = {best['rMAE']:.4f}"
          f"   ({(1-best['rMAE'])*100:.1f}% лучше M0)")
    print(f"{'═'*72}")


if __name__ == "__main__":
    main()
