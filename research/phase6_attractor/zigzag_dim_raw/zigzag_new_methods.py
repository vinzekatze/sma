#!/usr/bin/env python3
"""
Новые методы локальной аппроксимации: Simplex, LWR multi-p, LOWESS.

Методы (один проход, все одновременно):
  1. LWR K=50 и K=100, p∈{2,3,4,5}          → 8 вариантов
  2. Simplex (k=p+1, weights=exp(-d/d₁))      → 4 варианта
  3. LOWESS K=50 2-iter bisquare, p=3          → 1 вариант
  4. S-map θ=10, p=3                           → 1 (эталон)

Simplex (Sugihara 1990):
  Zeroth-order: ŷ = Σ w_i·y_i / Σ w_i
  k = E+1 = p+1  (минимум для окружения запроса в p-мерном пространстве)
  w_i = exp(−d_i / d_1)  (d_1 = расстояние до ближайшего соседа)
  Локальнее NW: bandwidth привязан к 1-му соседу, а не к k-му.

LOWESS (§7 обзора):
  LWR + итеративное bisquare перевзвешивание:
  u = residual / (6·MAD);  w_bisq = (1−u²)² если |u|<1, иначе 0
  Снижает влияние «выбросов»-соседей (экстремальные периоды рынка).

Ансамбль:
  Пары (топ-8): оптим. α свипом 0..1 шаг 0.05
  Тройки (топ-6): равный вес 1/3
  Четвёрки (топ-8): равный вес 1/4

Базовые: LWR p=3 K=50 = 0.4181 | Ens prev best = 0.4143
SBER 1d(4%) + 10m(0.4%), H=1.
КАУЗАЛЬНОСТЬ: пул строго j < step.
"""

import json
import itertools
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
P_LIST      = [2, 3, 4, 5]
H           = 1
MIN_HISTORY = 50
K50         = 50
K100        = 100
THETA_SMAP  = 10.0
N_LOWESS    = 2      # bisquare iterations

R_LWR_MIX  = 0.4181
R_ENS_MIX  = 0.4143


# ── Загрузка данных ──────────────────────────────────────────────────────────

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


# ── Предикторы ───────────────────────────────────────────────────────────────

def lwr_pred(Xn, y_pool, xn, dists, order, k):
    k_eff = min(k, len(y_pool))
    knn   = order[:k_eff]
    xi    = dists[order[k_eff - 1]]
    if xi < 1e-12:
        return float(y_pool[knn].mean())
    w  = np.exp(-0.5 * (dists[knn] / xi) ** 2)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
    b  = y_pool[knn] * ws
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(c[0] + c[1:] @ xn)


def simplex_pred(y_pool, dists, order, p):
    """Sugihara Simplex: k=p+1 соседей, bandwidth=d_1 (1-й сосед)."""
    k = p + 1
    if len(y_pool) < k:
        return None
    knn = order[:k]
    d1  = dists[order[0]]
    if d1 < 1e-12:
        return float(y_pool[knn].mean())
    w = np.exp(-dists[knn] / d1)
    return float((w * y_pool[knn]).sum() / (w.sum() + 1e-12))


def lowess_pred(Xn, y_pool, xn, dists, order, k, n_iter):
    """LWR + итеративное bisquare перевзвешивание."""
    k_eff = min(k, len(y_pool))
    knn   = order[:k_eff]
    xi    = dists[order[k_eff - 1]]
    if xi < 1e-12:
        return float(y_pool[knn].mean())
    w_gauss = np.exp(-0.5 * (dists[knn] / xi) ** 2)
    w = w_gauss.copy()
    for _ in range(n_iter):
        if w.sum() < 1e-12:
            w = w_gauss; break
        ws = np.sqrt(w)
        A  = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
        b  = y_pool[knn] * ws
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = y_pool[knn] - (c[0] + Xn[knn] @ c[1:])
        mad   = np.median(np.abs(resid))
        if mad < 1e-12:
            break
        u    = resid / (6.0 * mad)
        bisq = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        w    = w_gauss * bisq
    if w.sum() < 1e-12:
        w = w_gauss
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
    b  = y_pool[knn] * ws
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(c[0] + c[1:] @ xn)


def smap_pred(Xn, y_pool, xn, dists, theta):
    N      = len(y_pool)
    mean_d = dists.mean()
    w      = np.exp(-theta * dists / (mean_d + 1e-12))
    ws     = np.sqrt(w)
    A      = np.column_stack([np.ones(N), Xn]) * ws[:, None]
    b      = y_pool * ws
    c, *_  = np.linalg.lstsq(A, b, rcond=None)
    return float(c[0] + c[1:] @ xn)


def _valid(v):
    """True если v — число и не NaN."""
    if v is None:
        return False
    try:
        return not np.isnan(float(v))
    except Exception:
        return False


# ── Метрика ──────────────────────────────────────────────────────────────────

def rmae(preds, actuals):
    arr = np.asarray(preds, dtype=float)
    if len(arr) < 5 or np.any(np.isnan(arr)):
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return np.nan if dz < 1e-12 else float(np.mean(np.abs(arr - actuals)) / dz)


# ── Основной цикл ────────────────────────────────────────────────────────────

def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dates_1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dates_10m = find_pivots(h10m, l10m, d10m, T_10M)

    Xs_1d  = {p: build_X(p1d,  p) for p in P_LIST}
    Xs_10m = {p: build_X(p10m, p) for p in P_LIST}

    n1d = len(p1d)
    ce10m_all = np.searchsorted(dates_10m, dates_1d, side="left")

    print(f"SBER 1d {n1d} пив  10m {len(p10m)} пив")
    print(f"P_LIST={P_LIST}  K50={K50}  K100={K100}  θ={THETA_SMAP}  H={H}\n")
    print(f"Базовые: LWR p=3 K=50={R_LWR_MIX:.4f}  |  Ens prev={R_ENS_MIX:.4f}\n")

    # имена коллекций
    method_names = (
        [f"lwr50_p{p}"   for p in P_LIST] +
        [f"lwr100_p{p}"  for p in P_LIST] +
        [f"simplex_p{p}" for p in P_LIST] +
        ["lowess_p3", "smap_p3"]
    )
    preds_buf = {nm: [] for nm in method_names}
    actuals_all = []

    for step in range(MIN_HISTORY, n1d - H):
        # Проверить: X[step] валиден для всех p
        if any(np.any(np.isnan(Xs_1d[p][step])) for p in P_LIST):
            continue

        step_preds = {nm: None for nm in method_names}
        skip = False

        for p in P_LIST:
            # Построить пул для этого p
            j1d = np.arange(p - 1, step)
            v1d = ~np.any(np.isnan(Xs_1d[p][j1d]), axis=1) & (j1d + H < n1d)
            Xp  = list(Xs_1d[p][j1d[v1d]])
            yp  = list(p1d[j1d[v1d] + H])

            ce = int(ce10m_all[step]); na = len(p10m)
            j10 = np.arange(p - 1, min(ce, na - H))
            if len(j10):
                v10 = ~np.any(np.isnan(Xs_10m[p][j10]), axis=1)
                Xp.extend(Xs_10m[p][j10[v10]])
                yp.extend(p10m[j10[v10] + H])

            if len(Xp) < p + 2:
                skip = True; break

            X_pool = np.array(Xp); y_pool = np.array(yp)
            x_q    = Xs_1d[p][step]

            mu     = X_pool.mean(0)
            sigma  = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
            Xn     = (X_pool - mu) / sigma
            xn     = (x_q    - mu) / sigma

            dists  = np.linalg.norm(Xn - xn, axis=1)
            order  = np.argsort(dists)

            step_preds[f"lwr50_p{p}"]   = lwr_pred(Xn, y_pool, xn, dists, order, K50)
            step_preds[f"lwr100_p{p}"]  = lwr_pred(Xn, y_pool, xn, dists, order, K100)
            step_preds[f"simplex_p{p}"] = simplex_pred(y_pool, dists, order, p)

            if p == 3:
                step_preds["lowess_p3"] = lowess_pred(Xn, y_pool, xn, dists, order,
                                                       K50, N_LOWESS)
                step_preds["smap_p3"]   = smap_pred(Xn, y_pool, xn, dists, THETA_SMAP)

        if skip or not all(_valid(v) for v in step_preds.values()):
            continue

        for nm in method_names:
            preds_buf[nm].append(step_preds[nm])
        actuals_all.append(float(p1d[step + H]))

    actuals = np.array(actuals_all)
    n = len(actuals)
    print(f"Шагов: {n}\n")

    # ── Индивидуальные метрики ────────────────────────────────────────────────
    results = {}
    for nm in method_names:
        r = rmae(preds_buf[nm], actuals)
        results[nm] = (r, np.array(preds_buf[nm], dtype=float))

    ranked = sorted(results.items(), key=lambda x: x[1][0])

    print("=== Индивидуальные методы ===")
    print(f"{'Метод':<18}  {'rMAE':>8}  {'vs LWR-mix':>11}  {'vs Ens-mix':>11}")
    print("─" * 57)
    for nm, (r, _) in ranked:
        print(f"  {nm:<16}  {r:.4f}"
              f"   {(r/R_LWR_MIX-1)*100:+.1f}%{'':<5}"
              f"  {(r/R_ENS_MIX-1)*100:+.1f}%")
    print(f"\n  {'LWR K=50 p=3':16}  {R_LWR_MIX:.4f}   (baseline)")
    print(f"  {'Ens prev best':16}  {R_ENS_MIX:.4f}   (baseline)")

    # ── Ансамбль: пары (оптим. α) ────────────────────────────────────────────
    TOP_PAIR = min(8, len(ranked))
    top_pair_names = [nm for nm, _ in ranked[:TOP_PAIR]]

    pair_records = []
    for nm1, nm2 in itertools.combinations(top_pair_names, 2):
        arr1 = results[nm1][1]; arr2 = results[nm2][1]
        best_r = np.inf; best_a = 0.5
        for a in np.arange(0.0, 1.05, 0.05):
            r_a = rmae(a * arr1 + (1 - a) * arr2, actuals)
            if not np.isnan(r_a) and r_a < best_r:
                best_r = r_a; best_a = round(a, 2)
        pair_records.append((nm1, nm2, best_a, best_r))
    pair_records.sort(key=lambda x: x[3])

    print(f"\n=== Пары (топ {TOP_PAIR} методов, оптим. α) ===")
    print(f"{'M1':<18}  {'M2':<18}  {'α':>5}  {'rMAE':>8}  {'vs Ens-mix':>11}")
    print("─" * 72)
    for nm1, nm2, a, r in pair_records[:12]:
        print(f"  {nm1:<16}  {nm2:<16}  {a:.2f}   {r:.4f}"
              f"   {(r/R_ENS_MIX-1)*100:+.1f}%")

    # ── Ансамбль: тройки (равный вес) ────────────────────────────────────────
    TOP_TRIPLE = min(6, len(ranked))
    top_triple_names = [nm for nm, _ in ranked[:TOP_TRIPLE]]

    triple_records = []
    for nm1, nm2, nm3 in itertools.combinations(top_triple_names, 3):
        blend = (results[nm1][1] + results[nm2][1] + results[nm3][1]) / 3
        r = rmae(blend, actuals)
        triple_records.append((nm1, nm2, nm3, r))
    triple_records.sort(key=lambda x: x[3])

    print(f"\n=== Тройки (топ {TOP_TRIPLE} методов, равный вес 1/3) ===")
    print(f"{'M1':<18}  {'M2':<18}  {'M3':<18}  {'rMAE':>8}  {'vs Ens-mix':>11}")
    print("─" * 90)
    for nm1, nm2, nm3, r in triple_records[:8]:
        print(f"  {nm1:<16}  {nm2:<16}  {nm3:<16}  {r:.4f}"
              f"   {(r/R_ENS_MIX-1)*100:+.1f}%")

    # ── Ансамбль: четвёрки (равный вес) ──────────────────────────────────────
    TOP_QUAD = min(8, len(ranked))
    top_quad_names = [nm for nm, _ in ranked[:TOP_QUAD]]

    quad_records = []
    for combo in itertools.combinations(top_quad_names, 4):
        blend = np.mean([results[nm][1] for nm in combo], axis=0)
        r = rmae(blend, actuals)
        quad_records.append((combo, r))
    quad_records.sort(key=lambda x: x[1])

    print(f"\n=== Четвёрки (топ {TOP_QUAD}, равный вес 1/4, топ-5) ===")
    for combo, r in quad_records[:5]:
        print(f"  {' + '.join(c[:12] for c in combo)}"
              f"  →  {r:.4f}  vs Ens-mix: {(r/R_ENS_MIX-1)*100:+.1f}%")

    # ── ИТОГ ─────────────────────────────────────────────────────────────────
    best_ind    = ranked[0]
    best_pair   = pair_records[0]
    best_triple = triple_records[0]
    best_quad   = quad_records[0]
    print(f"\n=== ИТОГ ===")
    print(f"  Одиночный:  {best_ind[0]:<18} rMAE={best_ind[1][0]:.4f}"
          f"  vs Ens-mix: {(best_ind[1][0]/R_ENS_MIX-1)*100:+.1f}%")
    print(f"  Пара:       {best_pair[0]}+{best_pair[1]}"
          f"  α={best_pair[2]:.2f}  rMAE={best_pair[3]:.4f}"
          f"  vs Ens-mix: {(best_pair[3]/R_ENS_MIX-1)*100:+.1f}%")
    print(f"  Тройка:     {'+'.join(best_triple[:3])}"
          f"  rMAE={best_triple[3]:.4f}"
          f"  vs Ens-mix: {(best_triple[3]/R_ENS_MIX-1)*100:+.1f}%")
    print(f"  Четвёрка:   {'+'.join(c[:10] for c in best_quad[0])}"
          f"  rMAE={best_quad[1]:.4f}"
          f"  vs Ens-mix: {(best_quad[1]/R_ENS_MIX-1)*100:+.1f}%")

    # ── Сохранить CSV ─────────────────────────────────────────────────────────
    pd.DataFrame([
        {"method": nm, "rMAE": r,
         "vs_lwr_pct": (r/R_LWR_MIX-1)*100,
         "vs_ens_pct": (r/R_ENS_MIX-1)*100}
        for nm, (r, _) in ranked
    ]).to_csv(OUT / "new_methods_individual.csv", index=False)

    pd.DataFrame([
        {"m1": nm1, "m2": nm2, "alpha": a, "rMAE": r}
        for nm1, nm2, a, r in pair_records
    ]).to_csv(OUT / "new_methods_pairs.csv", index=False)

    pd.DataFrame([
        {"m1": nm1, "m2": nm2, "m3": nm3, "rMAE": r}
        for nm1, nm2, nm3, r in triple_records
    ]).to_csv(OUT / "new_methods_triples.csv", index=False)

    # ── График ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    names_plot = [nm for nm, _ in ranked]
    rmae_plot  = [r for _, (r, _) in ranked]
    clrs = ["seagreen" if r < R_ENS_MIX
            else ("steelblue" if r < R_LWR_MIX else "salmon")
            for r in rmae_plot]
    bars = ax.barh(names_plot[::-1], rmae_plot[::-1], color=clrs[::-1], alpha=0.85)
    ax.axvline(R_LWR_MIX, color="gray",   lw=1.5, ls="--",
               label=f"LWR K=50 p=3  {R_LWR_MIX:.4f}")
    ax.axvline(R_ENS_MIX, color="crimson", lw=1.5, ls=":",
               label=f"Ens prev  {R_ENS_MIX:.4f}")
    for bar, r in zip(bars[::-1], rmae_plot):
        ax.text(r + 0.001, bar.get_y() + bar.get_height()/2,
                f"{r:.4f}", va="center", fontsize=7.5)
    ax.set_xlabel("rMAE"); ax.set_title("Индивидуальные методы")
    ax.legend(fontsize=8.5); ax.grid(axis="x", alpha=0.2)

    ax = axes[1]
    show_n = min(14, len(pair_records))
    pair_labels = [f"{nm1[:10]}\n+{nm2[:10]}" for nm1, nm2, _, _ in pair_records[:show_n]]
    pair_rmae   = [r for _, _, _, r in pair_records[:show_n]]
    clrs2 = ["seagreen" if r < R_ENS_MIX else "salmon" for r in pair_rmae]
    bars2 = ax.barh(pair_labels[::-1], pair_rmae[::-1], color=clrs2[::-1], alpha=0.85)
    ax.axvline(R_ENS_MIX, color="crimson", lw=1.5, ls=":",
               label=f"Ens prev  {R_ENS_MIX:.4f}")
    for bar, r in zip(bars2[::-1], pair_rmae):
        ax.text(r + 0.001, bar.get_y() + bar.get_height()/2,
                f"{r:.4f}", va="center", fontsize=7.5)
    ax.set_xlabel("rMAE"); ax.set_title(f"Лучшие пары (оптим. α, топ {show_n})")
    ax.legend(fontsize=8.5); ax.grid(axis="x", alpha=0.2)

    fig.suptitle(
        f"Simplex · LWR multi-p · LOWESS  |  SBER 1d({T_1D*100:.0f}%)+10m({T_10M*100:.1f}%)"
        f"  H={H}  n={n}",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(OUT / "new_methods.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + график → {OUT}")


if __name__ == "__main__":
    run()
