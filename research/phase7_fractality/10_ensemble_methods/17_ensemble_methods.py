#!/usr/bin/env python3
"""
17_ensemble_methods.py — 4-way ансамбль: LWR, Simplex, S-map, RBF-log

Фаза 1: Независимый параметрический свип каждого метода
  LWR:     p=3, K=75 — зафиксирован (текущий baseline=0.3714)
  Simplex: p ∈ {3,4,6,8,10}, K=p+1  (EDM Sugihara & May 1990)
  S-map:   p=3, θ ∈ {0.5,1,2,4,8}   (Sugihara 1994, весь пул)
  RBF-log: p=3, K ∈ {5,10,15,20,30} (взвешенное среднее без экстраполяции)

Фаза 2: 4-way ансамбль с лучшими параметрами
  - Равномерные веса (0.25 × 4)
  - Oracle (per-step лучший метод, потолок)
  - Walk-forward стекинг M ∈ {10,20,50,100}: OLS-веса на последних M шагах

Контракт каузальности:
  Для шага step = текущий T_big пивот:
    pool: события пула строго до confirm_big[step] (searchsorted left)
    X_big: использует только lp_big[0..step]
  Никаких данных после step.

Общий пул: frac-only T=3.6%, dir-filter (наш текущий best).
Все методы предсказывают y_rel = log(p_{t+1}/p_t), реконструкция:
  price_hat = exp(lp_big[step] + y_rel_hat)
"""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
from scipy.optimize import nnls

HERE    = Path(__file__).parent
DATA    = HERE.parent.parent.parent / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

T_BIG    = 0.04
INTERVAL = "10m"
T_FIXED  = 0.036   # оптимальный T_frac (frac-only)

H           = 1
MIN_HISTORY = 50
REF_RMAE    = 0.3714  # LWR p=3 K=75 baseline

# ── параметры свипа ────────────────────────────────────────────────────────────
P_LWR_FIXED = 3
K_LWR_FIXED = 75

P_SX_GRID  = [3, 4, 6, 8, 10]       # Simplex: K=p+1
THETA_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]  # S-map: все точки пула
K_RBF_GRID = [5, 10, 15, 20, 30]    # RBF-log: K соседей
P_SMAP     = 3
P_RBF      = 3

STACKING_M = [10, 20, 50, 100]

# все p, для которых нужно предвычислить X
P_ALL = sorted(set([P_LWR_FIXED, P_SMAP, P_RBF] + P_SX_GRID))


# ── загрузка ──────────────────────────────────────────────────────────────────
def load_candles(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots_log(highs, lows, dates, thr):
    lh, ll = np.log(highs), np.log(lows)
    vals_log, confirm_dates, dirs, confirm_bars = [], [], [], []
    direction, ext_val = 0, (lh[0] + ll[0]) / 2.0
    for i in range(len(highs)):
        if direction == 0:
            if lh[i] - ext_val >= thr:
                direction, ext_val = 1, lh[i]
            elif ext_val - ll[i] >= thr:
                direction, ext_val = -1, ll[i]
        elif direction == 1:
            if lh[i] > ext_val:
                ext_val = lh[i]
            elif ext_val - ll[i] >= thr:
                vals_log.append(ext_val)
                confirm_dates.append(dates[i]); dirs.append(+1); confirm_bars.append(i)
                direction, ext_val = -1, ll[i]
        else:
            if ll[i] < ext_val:
                ext_val = ll[i]
            elif lh[i] - ext_val >= thr:
                vals_log.append(ext_val)
                confirm_dates.append(dates[i]); dirs.append(-1); confirm_bars.append(i)
                direction, ext_val = 1, lh[i]
    return (np.array(vals_log), np.array(confirm_dates),
            np.array(dirs), np.array(confirm_bars))


def build_X(log_prices, p):
    """Hybrid embedding: X[i] = [log_price, diff_{i→i-1}, diff_{i-1→i-2}, ...]"""
    n = len(log_prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = log_prices[i]
        for lag in range(1, p):
            X[i, lag] = log_prices[i - lag + 1] - log_prices[i - lag]
    return X


# ── каузальный пул ────────────────────────────────────────────────────────────
def causal_pool(confirm_date, dir_query, conf_f, Xf, dir_f, y_rel_f, p):
    """
    Возвращает (X_pool_diffs, y_pool) или (None, None).
    X_pool_diffs = Xf[:, 1:] — только diff-колонки (без log-price).
    """
    ce  = int(np.searchsorted(conf_f, confirm_date, side='left'))
    rng = np.arange(p - 1, min(ce, len(conf_f) - H))
    if len(rng) == 0:
        return None, None
    valid = ~np.any(np.isnan(Xf[rng]), axis=1) & ~np.isnan(y_rel_f[rng])
    idx   = rng[valid]
    if len(idx) < p + 2:
        return None, None
    idx_d = idx[dir_f[idx] == dir_query]
    if len(idx_d) < p + 2:
        idx_d = idx
    return Xf[idx_d][:, 1:], y_rel_f[idx_d]


# ── методы прогноза ────────────────────────────────────────────────────────────
def predict_lwr(xq, X_pool, y_pool, k):
    """LWR: Гауссовы веса по K соседям, взвешенный OLS."""
    dists = np.linalg.norm(X_pool - xq, axis=1)
    keff  = min(k, len(dists))
    if keff < 2:
        return np.nan
    knn = np.argsort(dists)[:keff]
    d   = dists[knn]
    xi  = d.max()
    if xi < 1e-12:
        return float(y_pool[knn].mean())
    w  = np.exp(-0.5 * (d / xi) ** 2)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(keff), X_pool[knn]]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, y_pool[knn] * ws, rcond=None)
    return float(c[0] + c[1:] @ xq)


def predict_simplex(xq, X_pool, y_pool, k):
    """
    Simplex projection (Sugihara & May 1990).
    w_i = exp(-d_i / d_1), d_1 = расстояние до 1-го соседа.
    Взвешенное среднее y_rel — без линейной экстраполяции.
    """
    dists = np.linalg.norm(X_pool - xq, axis=1)
    keff  = min(k, len(dists))
    if keff < 2:
        return np.nan
    knn = np.argsort(dists)[:keff]
    d   = dists[knn]
    d1  = d[0]
    if d1 < 1e-12:
        return float(y_pool[knn[0]])
    w     = np.exp(-d / d1)
    w_sum = w.sum()
    if w_sum < 1e-12:
        return np.nan
    return float(w @ y_pool[knn] / w_sum)


def predict_smap(xq, X_pool, y_pool, theta):
    """
    S-map (Sugihara 1994): использует ВЕСЬ пул.
    w_i = exp(-θ × d_i / d̄), d̄ = mean(||xi - xq||).
    Взвешенный OLS по всем точкам.
    """
    n = len(y_pool)
    if n < 3:
        return np.nan
    dists  = np.linalg.norm(X_pool - xq, axis=1)
    d_mean = dists.mean()
    if d_mean < 1e-12:
        return float(y_pool.mean())
    w  = np.exp(-theta * dists / d_mean)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(n), X_pool]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, y_pool * ws, rcond=None)
    result = float(c[0] + c[1:] @ xq)
    if not np.isfinite(result) or abs(result) > 2.0:
        return np.nan
    return result


def predict_rbf(xq, X_pool, y_pool, k):
    """
    RBF-log: Гауссовы веса по K соседям, взвешенное СРЕДНЕЕ y_rel.
    Отличие от LWR: нет линейной экстраполяции — чисто мультипликативный ансамбль.
    """
    dists = np.linalg.norm(X_pool - xq, axis=1)
    keff  = min(k, len(dists))
    if keff < 2:
        return np.nan
    knn = np.argsort(dists)[:keff]
    d   = dists[knn]
    xi  = d.max()
    if xi < 1e-12:
        return float(y_pool[knn].mean())
    w     = np.exp(-0.5 * (d / xi) ** 2)
    w_sum = w.sum()
    if w_sum < 1e-12:
        return np.nan
    return float(w @ y_pool[knn] / w_sum)


# ── метрика ───────────────────────────────────────────────────────────────────
def rmae(errs, acts):
    e = np.asarray(errs, float); a = np.asarray(acts, float)
    m = np.isfinite(e) & np.isfinite(a)
    if m.sum() < 2:
        return np.nan
    dz = float(np.mean(np.abs(np.diff(a[m]))))
    return float(np.mean(np.abs(e[m])) / dz) if dz > 1e-12 else np.nan


# ── walk-forward одиночного метода ────────────────────────────────────────────
def wf_single(predict_fn, param,
              lp_big, confirm_big, dir_big,
              conf_f, Xf, dir_f, y_rel_f, p):
    X_big = build_X(lp_big, p)
    errs, acts = [], []
    for step in range(MIN_HISTORY, len(lp_big) - H):
        if np.any(np.isnan(X_big[step])):
            continue
        xq     = X_big[step][1:]
        actual = float(np.exp(lp_big[step + H]))
        X_pool, y_pool = causal_pool(
            confirm_big[step], int(dir_big[step]),
            conf_f, Xf, dir_f, y_rel_f, p
        )
        if X_pool is None:
            continue
        y_rel_hat = predict_fn(xq, X_pool, y_pool, param)
        if not np.isfinite(y_rel_hat):
            continue
        errs.append(float(np.exp(lp_big[step] + y_rel_hat)) - actual)
        acts.append(actual)
    return np.array(errs), np.array(acts)


# ── walk-forward 4-way ансамбля ───────────────────────────────────────────────
def wf_ensemble_4way(lp_big, confirm_big, dir_big,
                     pool_by_p,
                     best_lwr, best_sx, best_sm, best_rbf,
                     stacking_M_list):
    """
    pool_by_p: {p: (conf_f, Xf, dir_f, y_rel_f)}
    best_*: (p, param) — лучшие параметры из Фазы 1.
    Возвращает results dict (стратегия → (errs, acts)),
    P_mat (N,4) абсолютные прогнозы, A_arr (N,) факт.
    """
    method_cfg = [
        ("lwr",     predict_lwr,     best_lwr[0], best_lwr[1]),
        ("simplex", predict_simplex, best_sx[0],  best_sx[1]),
        ("smap",    predict_smap,    best_sm[0],  best_sm[1]),
        ("rbf",     predict_rbf,     best_rbf[0], best_rbf[1]),
    ]
    X_big_by_p = {p: build_X(lp_big, p)
                  for p in set(cfg[2] for cfg in method_cfg)}

    preds_matrix = []
    acts_list    = []

    for step in range(MIN_HISTORY, len(lp_big) - H):
        actual = float(np.exp(lp_big[step + H]))
        row, ok = [], True
        for _, fn, p, param in method_cfg:
            if np.any(np.isnan(X_big_by_p[p][step])):
                ok = False; break
            xq = X_big_by_p[p][step][1:]
            conf_f, Xf, dir_f, y_rel_f = pool_by_p[p]
            X_pool, y_pool = causal_pool(
                confirm_big[step], int(dir_big[step]),
                conf_f, Xf, dir_f, y_rel_f, p
            )
            if X_pool is None:
                ok = False; break
            y_rel_hat = fn(xq, X_pool, y_pool, param)
            if not np.isfinite(y_rel_hat):
                ok = False; break
            row.append(float(np.exp(lp_big[step] + y_rel_hat)))
        if not ok or len(row) < 4:
            continue
        preds_matrix.append(row)
        acts_list.append(actual)

    P_mat = np.array(preds_matrix)   # (N, 4)
    A_arr = np.array(acts_list)      # (N,)
    N     = len(A_arr)
    names = [cfg[0] for cfg in method_cfg]

    results = {}

    # Индивидуальные методы
    for i, name in enumerate(names):
        results[f"single_{name}"] = (P_mat[:, i] - A_arr, A_arr)

    # Равномерные веса
    results["uniform"] = (P_mat.mean(axis=1) - A_arr, A_arr)

    # Oracle (per-step ближайший к факту)
    best_idx    = np.argmin(np.abs(P_mat - A_arr[:, None]), axis=1)
    pred_oracle = P_mat[np.arange(N), best_idx]
    results["oracle"] = (pred_oracle - A_arr, A_arr)

    # Walk-forward стекинг (nnls + нормализация)
    for M in stacking_M_list:
        pred_stack = np.full(N, np.nan)
        for i in range(N):
            if i < M:
                pred_stack[i] = P_mat[i].mean()
                continue
            P_hist = P_mat[i - M:i]
            y_hist = A_arr[i - M:i]
            w, _   = nnls(P_hist, y_hist)
            if w.sum() > 1e-10:
                w = w / w.sum()
            else:
                w = np.ones(4) / 4.0
            pred_stack[i] = float(w @ P_mat[i])
        results[f"stack_M{M}"] = (pred_stack - A_arr, A_arr)

    return results, P_mat, A_arr, names


# ── визуализация ──────────────────────────────────────────────────────────────
def plot_sweep(sweep_results, title, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (method_name, labels, rmae_list) in zip(axes.flat, sweep_results):
        colors = ["steelblue" if r < REF_RMAE else "lightcoral"
                  for r in rmae_list]
        bars = ax.bar(range(len(labels)), rmae_list, color=colors, alpha=0.85)
        ax.axhline(REF_RMAE, color="red", lw=1.2, ls="--",
                   label=f"LWR baseline {REF_RMAE}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("rMAE")
        ax.set_title(method_name)
        for bar, r in zip(bars, rmae_list):
            if np.isfinite(r):
                d = (r - REF_RMAE) / REF_RMAE * 100
                ax.text(bar.get_x() + bar.get_width() / 2, r + 0.001,
                        f"{r:.4f}\n({d:+.1f}%)",
                        ha="center", va="bottom", fontsize=7)
        ax.legend(fontsize=7)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


def plot_rolling(confirm_big, results, title, out_path):
    parse  = lambda s: datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    all_dt = np.array([parse(s) for s in confirm_big])
    W      = 50

    fig, axes = plt.subplots(2, 1, figsize=(15, 9),
                             gridspec_kw={"height_ratios": [1, 2]})

    # bar chart
    ax = axes[0]
    names = list(results.keys())
    vals  = [rmae(e, a) for e, a in results.values()]
    colors = ["steelblue" if v < REF_RMAE else "lightcoral" for v in vals]
    ax.barh(names, vals, color=colors, alpha=0.85)
    ax.axvline(REF_RMAE, color="red", lw=1.2, ls="--",
               label=f"baseline {REF_RMAE}")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            d = (v - REF_RMAE) / REF_RMAE * 100
            ax.text(v + 0.0005, i, f"{v:.4f} ({d:+.1f}%)",
                    va="center", fontsize=8)
    ax.set_xlabel("rMAE")
    ax.set_title(title)
    ax.legend(fontsize=8)

    # rolling rMAE
    ax = axes[1]
    colors_line = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, (errs, acts)), col in zip(results.items(), colors_line):
        if len(errs) < 2:
            continue
        dt_slice = all_dt[MIN_HISTORY:MIN_HISTORY + len(errs)]
        roll = [rmae(errs[max(0, i - W + 1):i + 1],
                     acts[max(0, i - W + 1):i + 1])
                for i in range(len(errs))]
        r = rmae(errs, acts)
        ax.plot(dt_slice, roll, lw=0.9, color=col,
                label=f"{name} ({r:.4f})")
    ax.axhline(REF_RMAE, color="red", lw=1.2, ls="--",
               label=f"baseline ({REF_RMAE})", zorder=5)
    ax.set_ylabel("rMAE (rolling W=50)")
    ax.legend(fontsize=7, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"T_BIG={T_BIG*100:.1f}%  INTERVAL={INTERVAL}  T_FIXED={T_FIXED*100:.1f}%\n")

    h, l, d = load_candles(INTERVAL)
    lp_big, confirm_big, dir_big, _ = find_pivots_log(h, l, d, T_BIG)
    print(f"T_BIG: {len(lp_big)} пивотов")

    lp_f, conf_f, dir_f, _ = find_pivots_log(h, l, d, T_FIXED)
    y_rel_f = np.array([lp_f[j + H] - lp_f[j] if j + H < len(lp_f) else np.nan
                        for j in range(len(lp_f))])
    print(f"T_frac={T_FIXED*100:.1f}%: {len(lp_f)} пивотов в пуле")

    print(f"Предвычисляем X для p={P_ALL} ...", flush=True)
    pool_by_p = {}
    for p in P_ALL:
        Xf = build_X(lp_f, p)
        pool_by_p[p] = (conf_f, Xf, dir_f, y_rel_f)

    # ═══════════════════════════════════════════════════════════════════════════
    # ФАЗА 1: Параметрический свип
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("ФАЗА 1: Параметрический свип каждого метода")
    print("═" * 62)

    sweep_results = []  # (method_label, param_labels, rmae_list)

    # ── LWR (зафиксирован) ────────────────────────────────────────────────────
    print(f"\n--- LWR (зафиксирован: p={P_LWR_FIXED}, K={K_LWR_FIXED}) ---")
    conf_f0, Xf0, dir_f0, y_rel_f0 = pool_by_p[P_LWR_FIXED]
    e_lwr, a_lwr = wf_single(predict_lwr, K_LWR_FIXED,
                              lp_big, confirm_big, dir_big,
                              conf_f0, Xf0, dir_f0, y_rel_f0, P_LWR_FIXED)
    r_lwr = rmae(e_lwr, a_lwr)
    print(f"  LWR p=3 K=75: rMAE={r_lwr:.4f}")
    sweep_results.append(("LWR (baseline)", ["p=3,K=75"], [r_lwr]))

    # ── Simplex ───────────────────────────────────────────────────────────────
    print(f"\n--- Simplex p ∈ {P_SX_GRID}, K=p+1 ---")
    sx_rmae, sx_labels = [], []
    best_sx_rmae, best_sx_p = np.inf, P_SX_GRID[0]
    for p_sx in P_SX_GRID:
        k_sx = p_sx + 1
        cf, Xf_p, df, yr = pool_by_p[p_sx]
        e, a = wf_single(predict_simplex, k_sx,
                         lp_big, confirm_big, dir_big,
                         cf, Xf_p, df, yr, p_sx)
        r = rmae(e, a)
        mark = " ◄" if r < best_sx_rmae else ""
        if r < best_sx_rmae:
            best_sx_rmae, best_sx_p = r, p_sx
        print(f"  Simplex p={p_sx} K={k_sx}: rMAE={r:.4f}"
              f"  Δ={(r - r_lwr) / r_lwr * 100:+.1f}%{mark}")
        sx_rmae.append(r); sx_labels.append(f"p={p_sx},K={k_sx}")
    sweep_results.append(("Simplex (K=p+1)", sx_labels, sx_rmae))

    # ── S-map ─────────────────────────────────────────────────────────────────
    print(f"\n--- S-map p={P_SMAP}, θ ∈ {THETA_GRID} ---")
    sm_rmae, sm_labels = [], []
    best_sm_rmae, best_sm_theta = np.inf, THETA_GRID[0]
    cf_sm, Xf_sm, df_sm, yr_sm = pool_by_p[P_SMAP]
    for theta in THETA_GRID:
        e, a = wf_single(predict_smap, theta,
                         lp_big, confirm_big, dir_big,
                         cf_sm, Xf_sm, df_sm, yr_sm, P_SMAP)
        r = rmae(e, a)
        mark = " ◄" if r < best_sm_rmae else ""
        if r < best_sm_rmae:
            best_sm_rmae, best_sm_theta = r, theta
        print(f"  S-map θ={theta}: rMAE={r:.4f}"
              f"  Δ={(r - r_lwr) / r_lwr * 100:+.1f}%{mark}")
        sm_rmae.append(r); sm_labels.append(f"θ={theta}")
    sweep_results.append(("S-map (p=3, весь пул)", sm_labels, sm_rmae))

    # ── RBF-log ───────────────────────────────────────────────────────────────
    print(f"\n--- RBF-log p={P_RBF}, K ∈ {K_RBF_GRID} ---")
    rbf_rmae, rbf_labels = [], []
    best_rbf_rmae, best_rbf_k = np.inf, K_RBF_GRID[0]
    cf_rbf, Xf_rbf, df_rbf, yr_rbf = pool_by_p[P_RBF]
    for k_rbf in K_RBF_GRID:
        e, a = wf_single(predict_rbf, k_rbf,
                         lp_big, confirm_big, dir_big,
                         cf_rbf, Xf_rbf, df_rbf, yr_rbf, P_RBF)
        r = rmae(e, a)
        mark = " ◄" if r < best_rbf_rmae else ""
        if r < best_rbf_rmae:
            best_rbf_rmae, best_rbf_k = r, k_rbf
        print(f"  RBF-log K={k_rbf}: rMAE={r:.4f}"
              f"  Δ={(r - r_lwr) / r_lwr * 100:+.1f}%{mark}")
        rbf_rmae.append(r); rbf_labels.append(f"K={k_rbf}")
    sweep_results.append(("RBF-log (p=3)", rbf_labels, rbf_rmae))

    print(f"\n{'─' * 62}")
    print("Лучшие параметры (Фаза 1):")
    print(f"  LWR:     p=3, K=75                    rMAE={r_lwr:.4f} (baseline)")
    print(f"  Simplex: p={best_sx_p}, K={best_sx_p + 1}  "
          f"                   rMAE={best_sx_rmae:.4f}"
          f"  Δ={(best_sx_rmae - r_lwr) / r_lwr * 100:+.1f}%")
    print(f"  S-map:   θ={best_sm_theta}, p={P_SMAP}  "
          f"                   rMAE={best_sm_rmae:.4f}"
          f"  Δ={(best_sm_rmae - r_lwr) / r_lwr * 100:+.1f}%")
    print(f"  RBF-log: K={best_rbf_k}, p={P_RBF}  "
          f"                   rMAE={best_rbf_rmae:.4f}"
          f"  Δ={(best_rbf_rmae - r_lwr) / r_lwr * 100:+.1f}%")

    # CSV Phase 1
    out1 = RESULTS / f"phase1_sweep_{INTERVAL}.csv"
    with open(out1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "param", "rmae", "delta_pct_vs_lwr"])
        for m_name, labels, rmae_list in sweep_results:
            for label, r in zip(labels, rmae_list):
                d = (r - r_lwr) / r_lwr * 100
                w.writerow([m_name, label, f"{r:.4f}", f"{d:.2f}"])
    print(f"\nCSV → {out1}")

    plot_sweep(sweep_results, "Фаза 1: Параметрический свип методов",
               RESULTS / f"phase1_sweep_{INTERVAL}.png")

    # ═══════════════════════════════════════════════════════════════════════════
    # ФАЗА 2: 4-way ансамбль
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 62)
    print("ФАЗА 2: 4-way ансамбль")
    print("═" * 62)
    print(f"  LWR:     p={P_LWR_FIXED}, K={K_LWR_FIXED}")
    print(f"  Simplex: p={best_sx_p}, K={best_sx_p + 1}")
    print(f"  S-map:   θ={best_sm_theta}, p={P_SMAP}")
    print(f"  RBF-log: K={best_rbf_k}, p={P_RBF}")
    print()

    best_lwr = (P_LWR_FIXED, K_LWR_FIXED)
    best_sx  = (best_sx_p, best_sx_p + 1)
    best_sm  = (P_SMAP, best_sm_theta)
    best_rbf = (P_RBF, best_rbf_k)

    results_ens, P_mat, A_arr, method_names = wf_ensemble_4way(
        lp_big, confirm_big, dir_big,
        pool_by_p,
        best_lwr, best_sx, best_sm, best_rbf,
        STACKING_M
    )

    print(f"\n{'─' * 62}")
    print(f"{'Стратегия':<32}  {'rMAE':>8}  {'Δ vs LWR':>9}")
    print(f"{'─' * 62}")
    for name, (errs, acts) in results_ens.items():
        r = rmae(errs, acts)
        if np.isfinite(r):
            d    = (r - r_lwr) / r_lwr * 100
            mark = " ◄" if r < r_lwr else ""
            print(f"  {name:<30}  {r:>8.4f}  {d:>+8.1f}%{mark}")
    print(f"{'─' * 62}")
    print(f"  LWR baseline              {r_lwr:>8.4f}")

    # CSV Phase 2
    out2 = RESULTS / f"phase2_ensemble_{INTERVAL}.csv"
    with open(out2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "rmae", "delta_pct", "n"])
        for name, (errs, acts) in results_ens.items():
            r = rmae(errs, acts)
            d = (r - r_lwr) / r_lwr * 100
            w.writerow([name, f"{r:.4f}", f"{d:.2f}", len(errs)])
    print(f"\nCSV → {out2}")

    # Графики Phase 2
    ens_keys   = {k: v for k, v in results_ens.items()
                  if not k.startswith("single_")}
    single_keys = {k: v for k, v in results_ens.items()
                   if k.startswith("single_") or k == "uniform"}

    plot_rolling(confirm_big, ens_keys,
                 "Фаза 2: Стратегии ансамблирования",
                 RESULTS / f"phase2_ensemble_{INTERVAL}.png")
    plot_rolling(confirm_big, single_keys,
                 "Фаза 2: Индивидуальные методы",
                 RESULTS / f"phase2_singles_{INTERVAL}.png")

    # ── Финальная сводка ──────────────────────────────────────────────────────
    print(f"\n{'═' * 62}")
    print("ИТОГ")
    print(f"{'═' * 62}")
    all_res = list(results_ens.items())
    best_name = min(all_res, key=lambda x: rmae(*x[1]) if np.isfinite(rmae(*x[1])) else np.inf)
    r_best = rmae(*best_name[1])
    print(f"  Baseline LWR:  rMAE={r_lwr:.4f}")
    print(f"  Лучшая стратегия: [{best_name[0]}]  rMAE={r_best:.4f}"
          f"  Δ={(r_best - r_lwr) / r_lwr * 100:+.1f}%")
    print(f"  Oracle потолок: rMAE={rmae(*results_ens['oracle']):.4f}"
          f"  Δ={(rmae(*results_ens['oracle']) - r_lwr) / r_lwr * 100:+.1f}%")


if __name__ == "__main__":
    main()
