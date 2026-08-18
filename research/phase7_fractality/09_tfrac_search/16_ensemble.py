#!/usr/bin/env python3
"""
16_ensemble.py — Ансамблирование по T_frac и K (раздельно)

Идея: вместо выбора одного T_frac или K усредняем прогнозы по нескольким
значениям. Не нужно предсказывать оптимум — диверсификация снижает дисперсию.

Эксперимент A — T_frac ансамбль (K=75 фиксирован):
  Тестируем наборы T_frac разной ширины, центрированные вокруг 3.6%:
    narrow  : {3.0, 3.3, 3.6, 3.9}%           — 4 значения
    medium  : {2.0, 2.6, 3.2, 3.6, 4.0}%      — 5 значений
    wide    : {1.0, 1.6, 2.2, 2.8, 3.4, 4.0}% — 6 значений
    all     : все 35 значений сетки (0.5..3.9%)
  Веса: равномерные vs inv-distance от 3.6%.

Эксперимент B — K ансамбль (T_frac=3.6% фиксирован):
  Наборы K:
    small   : {20, 35, 50}
    medium  : {35, 50, 75, 100}
    large   : {75, 100, 150, 200}
    all     : {10, 20, 35, 50, 75, 100, 150, 200}
  Веса: равномерные.

Эксперимент C — лучший T_frac ансамбль + лучший K ансамбль вместе.
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
REF_CSV = HERE.parent.parent / "05_lwr_improvements" / "results" / "best_10m_T4_Tf36.csv"

T_BIG    = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
INTERVAL = sys.argv[2]        if len(sys.argv) > 2 else "10m"

H           = 1
P           = 3
MIN_HISTORY = 50

T_FRAC_GRID = np.round(np.arange(0.005, T_BIG, 0.001), 4)
K_FIXED     = 75
T_FIXED     = 0.036
REF_RMAE    = 0.3714

# ── наборы для ансамбля ───────────────────────────────────────────────────────
def snap(values, grid):
    """Привязываем набор значений к ближайшим точкам сетки."""
    return sorted({float(grid[np.argmin(np.abs(grid - v))]) for v in values})


TF_SETS_DEF = {
    "narrow  (4 pt)": [0.030, 0.033, 0.036, 0.039],
    "medium  (5 pt)": [0.020, 0.026, 0.032, 0.036, 0.040],
    "wide    (6 pt)": [0.010, 0.016, 0.022, 0.028, 0.034, 0.040],
    "full   (35 pt)": list(T_FRAC_GRID),
}

K_SETS = {
    "small  {20,35,50}":        [20, 35, 50],
    "medium {35,50,75,100}":    [35, 50, 75, 100],
    "large  {75,100,150,200}":  [75, 100, 150, 200],
    "all    {10..200}":         [10, 20, 35, 50, 75, 100, 150, 200],
}


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


# ── walk-forward с ансамблем ──────────────────────────────────────────────────
def wf_ensemble_tf(lp_big, confirm_big, dir_big, pool_cache,
                   tf_set, weights=None, k=K_FIXED):
    """Ансамбль по tf_set при фиксированном K."""
    X_big = build_X(lp_big, P)
    if weights is None:
        weights = np.ones(len(tf_set)) / len(tf_set)
    else:
        weights = np.asarray(weights) / np.sum(weights)

    errs, acts = [], []
    for step in range(MIN_HISTORY, len(lp_big) - H):
        if np.any(np.isnan(X_big[step])):
            continue
        xq     = X_big[step][1:]
        actual = float(np.exp(lp_big[step + H]))

        preds_w = []
        for tf, w in zip(tf_set, weights):
            lp_f1, conf_f1, dir_f1, Xf1, y_rel = pool_cache[tf]
            X_pool, y_pool = causal_pool(
                confirm_big[step], int(dir_big[step]),
                conf_f1, Xf1, dir_f1, y_rel
            )
            if X_pool is None:
                continue
            p_rel = lwr_predict(xq, X_pool, y_pool, k)
            if np.isfinite(p_rel):
                preds_w.append((float(np.exp(lp_big[step] + p_rel)), w))

        if not preds_w:
            continue
        ps, ws = zip(*preds_w)
        ws_arr = np.array(ws); ws_arr /= ws_arr.sum()
        pred = float(np.dot(ps, ws_arr))
        errs.append(pred - actual)
        acts.append(actual)

    return np.array(errs), np.array(acts)


def wf_ensemble_K(lp_big, confirm_big, dir_big, pool_fixed,
                  k_set, tf_key=None, pool_cache=None):
    """Ансамбль по k_set при фиксированном T_frac."""
    X_big = build_X(lp_big, P)
    if pool_fixed is not None:
        lp_f1, conf_f1, dir_f1, Xf1, y_rel = pool_fixed
    else:
        lp_f1, conf_f1, dir_f1, Xf1, y_rel = pool_cache[tf_key]

    errs, acts = [], []
    for step in range(MIN_HISTORY, len(lp_big) - H):
        if np.any(np.isnan(X_big[step])):
            continue
        xq     = X_big[step][1:]
        actual = float(np.exp(lp_big[step + H]))

        X_pool, y_pool = causal_pool(
            confirm_big[step], int(dir_big[step]),
            conf_f1, Xf1, dir_f1, y_rel
        )
        if X_pool is None:
            continue

        preds = []
        for k in k_set:
            p_rel = lwr_predict(xq, X_pool, y_pool, k)
            if np.isfinite(p_rel):
                preds.append(float(np.exp(lp_big[step] + p_rel)))

        if not preds:
            continue
        errs.append(float(np.mean(preds)) - actual)
        acts.append(actual)

    return np.array(errs), np.array(acts)


def wf_ensemble_both(lp_big, confirm_big, dir_big, pool_cache,
                     tf_set, k_set):
    """Ансамбль по tf_set × k_set, равномерные веса."""
    X_big = build_X(lp_big, P)
    n_tf  = len(tf_set)

    errs, acts = [], []
    for step in range(MIN_HISTORY, len(lp_big) - H):
        if np.any(np.isnan(X_big[step])):
            continue
        xq     = X_big[step][1:]
        actual = float(np.exp(lp_big[step + H]))

        preds = []
        for tf in tf_set:
            lp_f1, conf_f1, dir_f1, Xf1, y_rel = pool_cache[tf]
            X_pool, y_pool = causal_pool(
                confirm_big[step], int(dir_big[step]),
                conf_f1, Xf1, dir_f1, y_rel
            )
            if X_pool is None:
                continue
            for k in k_set:
                p_rel = lwr_predict(xq, X_pool, y_pool, k)
                if np.isfinite(p_rel):
                    preds.append(float(np.exp(lp_big[step] + p_rel)))

        if not preds:
            continue
        errs.append(float(np.mean(preds)) - actual)
        acts.append(actual)

    return np.array(errs), np.array(acts)


# ── визуализация ──────────────────────────────────────────────────────────────
def plot_results(confirm_big, lp_big, results, out_path, title):
    parse  = lambda s: datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    all_dt = np.array([parse(s) for s in confirm_big])
    W      = 50

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    fig, axes = plt.subplots(2, 1, figsize=(15, 9),
                             gridspec_kw={"height_ratios": [1.2, 2]})

    # ── bar chart rMAE ────────────────────────────────────────────────────────
    ax = axes[0]
    names = list(results.keys())
    vals  = [rmae(e, a) for e, a in results.values()]
    colors_bar = ["steelblue" if v < REF_RMAE else "lightcoral" for v in vals]
    ax.barh(names, vals, color=colors_bar, alpha=0.85)
    ax.axvline(REF_RMAE, color="red", lw=1.2, ls="--", label=f"baseline {REF_RMAE}")
    for i, v in enumerate(vals):
        delta = (v - REF_RMAE) / REF_RMAE * 100
        ax.text(v + 0.0005, i, f"{v:.4f} ({delta:+.1f}%)", va="center", fontsize=8)
    ax.set_xlabel("rMAE")
    ax.set_title(title)
    ax.legend(fontsize=8)

    # ── rolling rMAE ─────────────────────────────────────────────────────────
    ax = axes[1]
    for (name, (errs, acts)), col in zip(results.items(), colors):
        if len(errs) < 2:
            continue
        # выровнять по времени: первый шаг = MIN_HISTORY
        step_start = MIN_HISTORY
        dt_slice   = all_dt[step_start:step_start + len(errs)]
        roll = [rmae(errs[max(0,i-W+1):i+1], acts[max(0,i-W+1):i+1])
                for i in range(len(errs))]
        r    = rmae(errs, acts)
        ax.plot(dt_slice, roll, lw=0.9, color=col,
                label=f"{name.strip()} ({r:.4f})")

    if REF_CSV.exists():
        ref    = np.loadtxt(REF_CSV, delimiter=",", skiprows=1)
        re, ra = ref[:, 1], ref[:, 2]
        rroll  = [rmae(re[max(0,i-W+1):i+1], ra[max(0,i-W+1):i+1])
                  for i in range(len(re))]
        dt_ref = all_dt[MIN_HISTORY:MIN_HISTORY + len(rroll)]
        ax.plot(dt_ref, rroll, color="red", lw=1.2, ls="--",
                label=f"baseline ({REF_RMAE})", zorder=5)

    ax.set_ylabel("rMAE (rolling W=50)")
    ax.legend(fontsize=7, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"T_BIG={T_BIG*100:.1f}%  INTERVAL={INTERVAL}")
    print()

    h, l, d = load_candles(INTERVAL)
    lp_big, confirm_big, dir_big, _ = find_pivots_log(h, l, d, T_BIG)
    print(f"T_BIG: {len(lp_big)} пивотов")

    print(f"Предвычисляем {len(T_FRAC_GRID)} зигзагов ...", flush=True)
    pool_cache = {}
    for tf in T_FRAC_GRID:
        lp_f, conf_f, dir_f, _ = find_pivots_log(h, l, d, tf)
        Xf1   = build_X(lp_f, P)
        y_rel = np.array([lp_f[j+H] - lp_f[j] if j+H < len(lp_f) else np.nan
                          for j in range(len(lp_f))])
        pool_cache[tf] = (lp_f, conf_f, dir_f, Xf1, y_rel)

    grid    = np.array(sorted(pool_cache.keys()))
    tf_fixed_key = float(grid[np.argmin(np.abs(grid - T_FIXED))])
    pool_fixed   = pool_cache[tf_fixed_key]

    # ── baseline ─────────────────────────────────────────────────────────────
    print("Baseline ...", flush=True)
    e_fix, a_fix = wf_ensemble_tf(lp_big, confirm_big, dir_big,
                                  pool_cache, [tf_fixed_key], k=K_FIXED)
    r_fix = rmae(e_fix, a_fix)
    print(f"  fixed T=3.6% K=75: rMAE={r_fix:.4f}\n")

    # ── A. T_frac ансамбль ───────────────────────────────────────────────────
    print("=== A. T_frac ансамбль (K=75) ===", flush=True)
    results_tf = {"fixed T=3.6%": (e_fix, a_fix)}

    for name, tf_vals_raw in TF_SETS_DEF.items():
        tf_set = snap(tf_vals_raw, grid)
        print(f"  {name}: {[f'{v*100:.1f}' for v in tf_set]}%", flush=True)

        # равномерные веса
        e, a = wf_ensemble_tf(lp_big, confirm_big, dir_big,
                              pool_cache, tf_set, k=K_FIXED)
        r = rmae(e, a)
        results_tf[f"{name} uniform"] = (e, a)
        print(f"    uniform: {r:.4f}  Δ={( r-r_fix)/r_fix*100:+.1f}%")

        # inv-distance веса от 3.6%
        dists_from_center = np.abs(np.array(tf_set) - T_FIXED) + 1e-6
        inv_w = 1.0 / dists_from_center
        e, a = wf_ensemble_tf(lp_big, confirm_big, dir_big,
                              pool_cache, tf_set, weights=inv_w, k=K_FIXED)
        r = rmae(e, a)
        results_tf[f"{name} inv-dist"] = (e, a)
        print(f"    inv-dist: {r:.4f}  Δ={(r-r_fix)/r_fix*100:+.1f}%")

    # ── B. K ансамбль ────────────────────────────────────────────────────────
    print("\n=== B. K ансамбль (T_frac=3.6%) ===", flush=True)
    results_k = {"fixed K=75": (e_fix, a_fix)}

    for name, k_set in K_SETS.items():
        e, a = wf_ensemble_K(lp_big, confirm_big, dir_big, pool_fixed, k_set)
        r    = rmae(e, a)
        results_k[name] = (e, a)
        print(f"  {name}: rMAE={r:.4f}  Δ={(r-r_fix)/r_fix*100:+.1f}%")

    # ── C. Лучшая комбинация ─────────────────────────────────────────────────
    best_tf_name = min(
        [k for k in results_tf if k != "fixed T=3.6%"],
        key=lambda k: rmae(*results_tf[k])
    )
    best_k_name  = min(
        [k for k in results_k if k != "fixed K=75"],
        key=lambda k: rmae(*results_k[k])
    )

    # извлекаем tf_set и k_set для лучших
    best_tf_raw  = TF_SETS_DEF[best_tf_name.rsplit(" ", 1)[0].strip()]
    best_tf_set  = snap(best_tf_raw, grid)
    best_k_set   = K_SETS[best_k_name]

    print(f"\n=== C. Лучшая комбинация ===")
    print(f"  T_frac: {best_tf_name} → {[f'{v*100:.1f}' for v in best_tf_set]}%")
    print(f"  K:      {best_k_name}")

    # оба по отдельности + совместно
    results_c = {
        "fixed":            (e_fix, a_fix),
        f"best tf ensemble": results_tf[best_tf_name],
        f"best K ensemble":  results_k[best_k_name],
    }

    e, a = wf_ensemble_both(lp_big, confirm_big, dir_big, pool_cache,
                            best_tf_set, best_k_set)
    r    = rmae(e, a)
    results_c["tf+K ensemble"] = (e, a)
    print(f"  tf+K ensemble: rMAE={r:.4f}  Δ={(r-r_fix)/r_fix*100:+.1f}%")

    # итоговая таблица
    print(f"\n{'─'*55}")
    print(f"{'Стратегия':<32}  {'rMAE':>8}  {'Δ':>8}")
    all_res = list(results_tf.items()) + \
              [(k, v) for k, v in results_k.items() if k != "fixed K=75"] + \
              [("tf+K ensemble", results_c["tf+K ensemble"])]
    for name, (e, a) in all_res:
        r = rmae(e, a)
        d = (r - r_fix) / r_fix * 100
        mark = " ◄" if r < r_fix else ""
        print(f"  {name:<30}  {r:>8.4f}  {d:>+7.1f}%{mark}")
    print(f"\n  baseline fixed T=3.6% K=75     {r_fix:>8.4f}")

    # CSV итог
    out_csv = RESULTS / f"ensemble_results_{INTERVAL}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "rmae", "delta_pct", "n"])
        for name, (e, a) in all_res:
            r = rmae(e, a)
            d = (r - r_fix) / r_fix * 100
            w.writerow([name, f"{r:.4f}", f"{d:.2f}", len(e)])
    print(f"\nCSV → {out_csv}")

    plot_results(confirm_big, lp_big, results_tf,
                 RESULTS / f"ensemble_tf_{INTERVAL}.png",
                 f"A. T_frac ансамбль (K={K_FIXED})")
    plot_results(confirm_big, lp_big, results_k,
                 RESULTS / f"ensemble_K_{INTERVAL}.png",
                 "B. K ансамбль (T_frac=3.6%)")
    plot_results(confirm_big, lp_big, results_c,
                 RESULTS / f"ensemble_best_{INTERVAL}.png",
                 "C. Лучшая комбинация")


if __name__ == "__main__":
    main()
