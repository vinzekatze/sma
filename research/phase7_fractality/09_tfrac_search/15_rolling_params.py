#!/usr/bin/env python3
"""
15_rolling_params.py — Rolling mean/median оптимальных T_frac и K

Гипотеза: скользящее среднее tf_oracle по последним M событиям даёт
лучший каузальный T_frac, чем одиночный lag-1 или фиксированный 3.6%.

Тест:
  tf_roll(M) = mean(tf_oracle[t-M .. t-1])  → использовать как T_frac шага t
  K_roll(M)  = round(mean(K_oracle[t-M .. t-1]))  → K шага t

Диапазон M: [3, 5, 10, 20, 50, 100]
Дополнительно: rolling median (более робастный к выбросам)

ACF tf_oracle до лага 50 — ищем долгосрочную память.
"""
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

HERE    = Path(__file__).parent
DATA    = HERE.parent.parent.parent / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
ORACLE  = HERE.parent / "08_adaptive_fd" / "results" / "kappa_oracle_10m.csv"
STAB    = RESULTS / "stability_10m.csv"
REF_CSV = HERE.parent.parent / "05_lwr_improvements" / "results" / "best_10m_T4_Tf36.csv"

T_BIG    = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
INTERVAL = sys.argv[2]        if len(sys.argv) > 2 else "10m"

H           = 1
P           = 3
MIN_HISTORY = 50

T_FRAC_GRID = np.round(np.arange(0.005, T_BIG, 0.001), 4)
K_GRID      = np.array([10, 20, 35, 50, 75, 100, 150, 200])
T_FIXED     = 0.036
K_FIXED     = 75
M_GRID      = [3, 5, 10, 20, 50, 100]
REF_RMAE    = 0.3714


# ── загрузка ─────────────────────────────────────────────────────────────────
def load_candles(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots_log(highs, lows, dates, thr):
    lh, ll = np.log(highs), np.log(lows)
    vals_log, confirm_dates, dirs, confirm_bars = [], [], [], []
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
                vals_log.append(ext_val)
                confirm_dates.append(dates[i]); dirs.append(+1); confirm_bars.append(i)
                direction, ext_val, ext_idx = -1, ll[i], i
        else:
            if ll[i] < ext_val:
                ext_val, ext_idx = ll[i], i
            elif lh[i] - ext_val >= thr:
                vals_log.append(ext_val)
                confirm_dates.append(dates[i]); dirs.append(-1); confirm_bars.append(i)
                direction, ext_val, ext_idx = 1, lh[i], i
    return (np.array(vals_log), np.array(confirm_dates),
            np.array(dirs), np.array(confirm_bars))


def build_X(log_prices, p):
    n = len(log_prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = log_prices[i]
        for lag in range(1, p):
            X[i, lag] = log_prices[i - lag + 1] - log_prices[i - lag]
    return X


def load_oracle_csv(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            s = int(row["step"])
            rows[s] = {k: float(v) for k, v in row.items() if k != "step"}
    return rows


def load_stab_csv(path):
    """Загружает step, tf_oracle, K_oracle из stability CSV."""
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            s = int(row["step"])
            rows[s] = {"tf_oracle": float(row["tf_oracle"]),
                       "K_oracle":  float(row["K_oracle"])}
    return rows


def causal_pool(confirm_date, dir_query, dt_f1, Xf1, dir_f1, y_rel):
    ce    = int(np.searchsorted(dt_f1, confirm_date, side='left'))
    rng   = np.arange(P - 1, min(ce, len(dt_f1) - H))
    if len(rng) == 0:
        return None, None
    valid = ~np.any(np.isnan(Xf1[rng]), axis=1) & ~np.isnan(y_rel[rng])
    idx   = rng[valid]
    if len(idx) < P + 2:
        return None, None
    idx_d = idx[dir_f1[idx] == dir_query]
    if len(idx_d) < P + 2:
        idx_d = idx
    return Xf1[idx_d][:, 1:], y_rel[idx_d]


def lwr_predict(xq, X_pool, y_pool, k):
    dists = np.linalg.norm(X_pool - xq, axis=1)
    keff  = min(k, len(dists))
    if keff < 2:
        return np.nan
    knn   = np.argsort(dists)[:keff]
    X_sel, y_sel, d = X_pool[knn], y_pool[knn], dists[knn]
    xi = d.max()
    if xi < 1e-12:
        return float(y_sel.mean())
    w  = np.exp(-0.5 * (d / xi) ** 2)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(keff), X_sel]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, y_sel * ws, rcond=None)
    return float(c[0] + c[1:] @ xq)


def rmae(errs, acts):
    e = np.asarray(errs, float); a = np.asarray(acts, float)
    m = np.isfinite(e) & np.isfinite(a)
    if m.sum() < 2:
        return np.nan
    dz = float(np.mean(np.abs(np.diff(a[m]))))
    return float(np.mean(np.abs(e[m])) / dz) if dz > 1e-12 else np.nan


def compute_acf(x, max_lag=50):
    x  = np.asarray(x, float)
    x  = x[np.isfinite(x)]
    xc = x - x.mean()
    c0 = float(xc @ xc)
    if c0 < 1e-12:
        return np.arange(1, max_lag + 1), np.zeros(max_lag)
    lags, vals = [], []
    for lag in range(1, min(max_lag + 1, len(xc))):
        vals.append(float(xc[lag:] @ xc[:-lag]) / c0)
        lags.append(lag)
    return np.array(lags), np.array(vals)


# ── rolling walk-forward ──────────────────────────────────────────────────────
def wf_rolling(stab_map, lp_big, confirm_big, dir_big, pool_cache,
               M, param="tf", aggfunc="mean"):
    """
    param="tf"  : rolling T_frac, K=K_FIXED
    param="K"   : rolling K, T_frac=T_FIXED
    aggfunc     : "mean" | "median"
    """
    X_big  = build_X(lp_big, P)
    grid_tf = np.array(sorted(pool_cache.keys()))
    steps  = sorted(stab_map)

    errs, acts = [], []

    for i, step in enumerate(steps):
        if step < MIN_HISTORY or step + H >= len(lp_big):
            continue
        if np.any(np.isnan(X_big[step])):
            continue
        if i < M:
            continue

        # история из M предыдущих шагов (строго каузально)
        prev_steps = steps[i - M: i]
        history_tf = np.array([stab_map[s]["tf_oracle"] for s in prev_steps])
        history_K  = np.array([stab_map[s]["K_oracle"]  for s in prev_steps])

        if aggfunc == "mean":
            tf_val = float(np.mean(history_tf))
            k_val  = float(np.mean(history_K))
        else:
            tf_val = float(np.median(history_tf))
            k_val  = float(np.median(history_K))

        # выбор параметров для этого шага
        if param == "tf":
            tf_key = float(grid_tf[np.argmin(np.abs(grid_tf - tf_val))])
            k_use  = K_FIXED
        elif param == "K":
            tf_key = float(grid_tf[np.argmin(np.abs(grid_tf - T_FIXED))])
            k_use  = int(K_GRID[np.argmin(np.abs(K_GRID - k_val))])
        else:  # both
            tf_key = float(grid_tf[np.argmin(np.abs(grid_tf - tf_val))])
            k_use  = int(K_GRID[np.argmin(np.abs(K_GRID - k_val))])

        lp_f1, conf_f1, dir_f1, Xf1, y_rel = pool_cache[tf_key]
        X_pool, y_pool = causal_pool(
            confirm_big[step], int(dir_big[step]),
            conf_f1, Xf1, dir_f1, y_rel
        )
        if X_pool is None:
            continue

        p_rel  = lwr_predict(X_big[step][1:], X_pool, y_pool, k_use)
        actual = float(np.exp(lp_big[step + H]))
        if np.isfinite(p_rel):
            errs.append(float(np.exp(lp_big[step] + p_rel)) - actual)
            acts.append(actual)

    return np.array(errs), np.array(acts)


def wf_fixed(stab_map, lp_big, confirm_big, dir_big, pool_cache):
    X_big   = build_X(lp_big, P)
    grid_tf = np.array(sorted(pool_cache.keys()))
    tf_key  = float(grid_tf[np.argmin(np.abs(grid_tf - T_FIXED))])
    errs, acts = [], []
    for step in sorted(stab_map):
        if step < MIN_HISTORY or step + H >= len(lp_big): continue
        if np.any(np.isnan(X_big[step])): continue
        lp_f1, conf_f1, dir_f1, Xf1, y_rel = pool_cache[tf_key]
        X_pool, y_pool = causal_pool(confirm_big[step], int(dir_big[step]),
                                     conf_f1, Xf1, dir_f1, y_rel)
        if X_pool is None: continue
        p_rel  = lwr_predict(X_big[step][1:], X_pool, y_pool, K_FIXED)
        actual = float(np.exp(lp_big[step + H]))
        if np.isfinite(p_rel):
            errs.append(float(np.exp(lp_big[step] + p_rel)) - actual)
            acts.append(actual)
    return np.array(errs), np.array(acts)


# ── визуализация ──────────────────────────────────────────────────────────────
def plot_acf(tf_seq, k_seq, out_path):
    lags_tf, acf_tf = compute_acf(tf_seq, 50)
    lags_k,  acf_k  = compute_acf(k_seq,  50)
    ci = 1.96 / np.sqrt(len(tf_seq))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    for ax, lags, acf_v, color, title in [
        (axes[0], lags_tf, acf_tf, "steelblue",  "ACF tf_oracle (до лага 50)"),
        (axes[1], lags_k,  acf_k,  "darkorange",  "ACF K_oracle  (до лага 50)"),
    ]:
        ax.bar(lags, acf_v, color=color, alpha=0.7, width=0.6)
        ax.axhline( ci, color="gray", lw=0.8, ls="--", label=f"95% CI ±{ci:.3f}")
        ax.axhline(-ci, color="gray", lw=0.8, ls="--")
        ax.axhline(0,   color="black", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("Лаг (T_big события)")
        ax.set_ylabel("ACF")
        ax.legend(fontsize=8)

    plt.suptitle(f"SBER {INTERVAL} | T_big={T_BIG*100:.1f}% | n={len(tf_seq)}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


def plot_sweep(m_grid, tf_rmae_mean, tf_rmae_med, k_rmae_mean, k_rmae_med,
               r_fix, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, mean_r, med_r, color, title in [
        (axes[0], tf_rmae_mean, tf_rmae_med, "steelblue",  "Rolling T_frac (K=75 фикс)"),
        (axes[1], k_rmae_mean,  k_rmae_med,  "darkorange",  "Rolling K (T_frac=3.6% фикс)"),
    ]:
        ax.plot(m_grid, mean_r, marker="o", ms=6, color=color,
                lw=1.2, label="rolling mean")
        ax.plot(m_grid, med_r,  marker="s", ms=6, color=color,
                lw=1.2, ls="--", alpha=0.7, label="rolling median")
        ax.axhline(r_fix, color="black", lw=1.0, ls=":", label=f"fixed ({r_fix:.4f})")
        ax.axhline(REF_RMAE, color="red", lw=0.8, ls="--", label=f"05_best ({REF_RMAE})")
        ax.set_xlabel("M (окно событий)")
        ax.set_ylabel("rMAE")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xticks(m_grid)

    plt.suptitle(f"Sweep M: rolling mean/median оптимальных параметров\n"
                 f"SBER {INTERVAL} | T_big={T_BIG*100:.1f}%", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


def plot_rolling_tf(confirm_big, stab_map, M_best, out_path):
    """Покажем что rolling mean делает с T_frac по времени."""
    parse  = lambda s: datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    steps  = sorted(stab_map)
    tf_seq = np.array([stab_map[s]["tf_oracle"] for s in steps])
    dt     = np.array([parse(confirm_big[s]) for s in steps])

    roll_mean = np.array([
        np.mean(tf_seq[max(0, i-M_best):i]) if i >= M_best else np.nan
        for i in range(len(tf_seq))
    ])
    roll_med = np.array([
        np.median(tf_seq[max(0, i-M_best):i]) if i >= M_best else np.nan
        for i in range(len(tf_seq))
    ])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dt, tf_seq * 100, color="lightgray", lw=0.6, alpha=0.8, label="tf_oracle (raw)")
    ax.plot(dt, roll_mean * 100, color="steelblue", lw=1.2, label=f"rolling mean M={M_best}")
    ax.plot(dt, roll_med  * 100, color="tomato", lw=1.0, ls="--",
            label=f"rolling median M={M_best}")
    ax.axhline(T_FIXED * 100, color="black", lw=0.8, ls=":", label="fixed 3.6%")
    ax.set_ylabel("T_frac (%)")
    ax.set_title(f"Rolling T_frac vs raw tf_oracle  (M={M_best})")
    ax.legend(fontsize=8, ncol=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"T_BIG={T_BIG*100:.1f}%  INTERVAL={INTERVAL}")
    print(f"M_GRID={M_GRID}")
    print()

    h, l, d = load_candles(INTERVAL)
    lp_big, confirm_big, dir_big, _ = find_pivots_log(h, l, d, T_BIG)
    print(f"T_BIG: {len(lp_big)} пивотов")

    stab_map = load_stab_csv(STAB)
    print(f"Стабильность-записей: {len(stab_map)}")

    print(f"Предвычисляем {len(T_FRAC_GRID)} зигзагов ...", flush=True)
    pool_cache = {}
    for tf in T_FRAC_GRID:
        lp_f, conf_f, dir_f, _ = find_pivots_log(h, l, d, tf)
        Xf1   = build_X(lp_f, P)
        y_rel = np.array([lp_f[j+H] - lp_f[j] if j+H < len(lp_f) else np.nan
                          for j in range(len(lp_f))])
        pool_cache[tf] = (lp_f, conf_f, dir_f, Xf1, y_rel)
    print()

    # ── ACF до лага 50 ───────────────────────────────────────────────────────
    steps   = sorted(stab_map)
    tf_seq  = np.array([stab_map[s]["tf_oracle"] for s in steps])
    k_seq   = np.array([stab_map[s]["K_oracle"]  for s in steps])

    _, acf_tf = compute_acf(tf_seq, 50)
    _, acf_k  = compute_acf(k_seq,  50)
    ci = 1.96 / np.sqrt(len(tf_seq))

    print("=== ACF tf_oracle (лаги 1-10) ===")
    for lag, v in enumerate(acf_tf[:10], 1):
        bar = "◄" if abs(v) > ci else " "
        print(f"  lag={lag:2d}  {v:+.3f}  {bar}")

    print("\n=== ACF K_oracle (лаги 1-10) ===")
    for lag, v in enumerate(acf_k[:10], 1):
        bar = "◄" if abs(v) > ci else " "
        print(f"  lag={lag:2d}  {v:+.3f}  {bar}")
    print()

    # ── базовый ──────────────────────────────────────────────────────────────
    e_fix, a_fix = wf_fixed(stab_map, lp_big, confirm_big, dir_big, pool_cache)
    r_fix = rmae(e_fix, a_fix)
    print(f"Baseline fixed T=3.6% K=75:  rMAE={r_fix:.4f}")
    print()

    # ── sweep M ──────────────────────────────────────────────────────────────
    tf_rmae_mean, tf_rmae_med = [], []
    k_rmae_mean,  k_rmae_med  = [], []

    print(f"{'':5}  {'tf mean':>9}  {'tf med':>9}  {'K mean':>9}  {'K med':>9}")
    for M in M_GRID:
        e, a = wf_rolling(stab_map, lp_big, confirm_big, dir_big,
                          pool_cache, M, param="tf", aggfunc="mean")
        rm_tf_m = rmae(e, a)

        e, a = wf_rolling(stab_map, lp_big, confirm_big, dir_big,
                          pool_cache, M, param="tf", aggfunc="median")
        rm_tf_d = rmae(e, a)

        e, a = wf_rolling(stab_map, lp_big, confirm_big, dir_big,
                          pool_cache, M, param="K", aggfunc="mean")
        rm_k_m = rmae(e, a)

        e, a = wf_rolling(stab_map, lp_big, confirm_big, dir_big,
                          pool_cache, M, param="K", aggfunc="median")
        rm_k_d = rmae(e, a)

        tf_rmae_mean.append(rm_tf_m); tf_rmae_med.append(rm_tf_d)
        k_rmae_mean.append(rm_k_m);   k_rmae_med.append(rm_k_d)

        def fmt(r):
            if not np.isfinite(r): return "     —"
            s = f"{r:.4f}"
            return f"*{s}" if r < r_fix else f" {s}"

        print(f"M={M:<3}  {fmt(rm_tf_m):>9}  {fmt(rm_tf_d):>9}"
              f"  {fmt(rm_k_m):>9}  {fmt(rm_k_d):>9}")

    print()
    print(f"Baseline: {r_fix:.4f}  |  05_best: {REF_RMAE:.4f}")
    print("* = лучше baseline")

    # лучший M для T_frac
    best_M_tf = M_GRID[int(np.nanargmin(
        [min(m, d) for m, d in zip(tf_rmae_mean, tf_rmae_med)]))]

    plot_acf(tf_seq, k_seq, RESULTS / f"acf50_{INTERVAL}.png")
    plot_sweep(M_GRID, tf_rmae_mean, tf_rmae_med, k_rmae_mean, k_rmae_med,
               r_fix, RESULTS / f"rolling_sweep_{INTERVAL}.png")
    plot_rolling_tf(confirm_big, stab_map, best_M_tf,
                    RESULTS / f"rolling_tf_vis_{INTERVAL}.png")


if __name__ == "__main__":
    main()
