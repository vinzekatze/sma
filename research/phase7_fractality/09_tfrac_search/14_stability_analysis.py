#!/usr/bin/env python3
"""
14_stability_analysis.py — Стабильность оптимальных K и T_frac (раздельно)

Два независимых оракула:
  tf_oracle(t) — лучший T_frac при K=75  (данные из скр.09)
  K_oracle(t)  — лучший K при T_frac=3.6%  (свип здесь)

Анализируем каждый по отдельности:
  1. Распределение и частота переключений
  2. ACF по лагам событий
  3. Lag-1 стратегия: использовать оптимум(t-1) для прогноза на шаге t+1
     (каузально: исход step t известен к моменту прогноза step t+1)

Почему раздельно: совместный свип 35×8=280 комбинаций на шаг — оракул
перефитирует шум, ACF становится бессмысленной.
"""
import csv
import json
import sys
import numpy as np
from scipy import stats
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
REF_CSV = HERE.parent.parent / "05_lwr_improvements" / "results" / "best_10m_T4_Tf36.csv"

T_BIG    = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
INTERVAL = sys.argv[2]        if len(sys.argv) > 2 else "10m"

H           = 1
P           = 3
MIN_HISTORY = 50

T_FRAC_GRID = np.round(np.arange(0.005, T_BIG, 0.001), 4)
K_GRID      = np.array([10, 20, 35, 50, 75, 100, 150, 200])
T_FIXED     = 0.036   # для K-oracle
K_FIXED     = 75      # для tf-oracle (= текущий дефолт)

REF_RMAE = 0.3714


# ── загрузка ─────────────────────────────────────────────────────────────────
def load_tf(name):
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


def load_oracle(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            s = int(row["step"])
            rows[s] = {k: float(v) for k, v in row.items() if k != "step"}
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


def lwr_predict_k(xq, X_pool, y_pool, k):
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
    errs = np.asarray(errs, float); acts = np.asarray(acts, float)
    m    = np.isfinite(errs) & np.isfinite(acts)
    if m.sum() < 2:
        return np.nan
    a = acts[m]
    pers = np.abs(a[2:] - a[:-2]) if len(a) >= 3 else np.abs(np.diff(a))
    dz = float(np.mean(pers)) if len(pers) > 0 else 1.0
    return float(np.mean(np.abs(errs[m])) / dz) if dz > 1e-12 else np.nan


def compute_acf(x, max_lag=20):
    x  = np.asarray(x, float)[np.isfinite(np.asarray(x, float))]
    xc = x - x.mean()
    c0 = float(xc @ xc)
    if c0 < 1e-12:
        return np.arange(1, max_lag + 1), np.zeros(max_lag)
    lags, vals = [], []
    for lag in range(1, min(max_lag + 1, len(xc))):
        vals.append(float(xc[lag:] @ xc[:-lag]) / c0)
        lags.append(lag)
    return np.array(lags), np.array(vals)


# ── K-oracle ──────────────────────────────────────────────────────────────────
def k_oracle_sweep(oracle_map, lp_big, confirm_big, dir_big, pool_fixed):
    X_big = build_X(lp_big, P)
    lp_f1, conf_f1, dir_f1, Xf1, y_rel = pool_fixed
    results = {}

    for step in sorted(oracle_map):
        if step < MIN_HISTORY or np.any(np.isnan(X_big[step])):
            continue
        if step + H >= len(lp_big):
            continue

        X_pool, y_pool = causal_pool(
            confirm_big[step], int(dir_big[step]),
            conf_f1, Xf1, dir_f1, y_rel
        )
        if X_pool is None:
            continue

        xq     = X_big[step][1:]
        actual = float(np.exp(lp_big[step + H]))

        best_k, best_err = None, np.inf
        for k in K_GRID:
            p_rel = lwr_predict_k(xq, X_pool, y_pool, k)
            if np.isfinite(p_rel):
                err = abs(float(np.exp(lp_big[step] + p_rel)) - actual)
                if err < best_err:
                    best_err, best_k = err, k

        if best_k is not None:
            results[step] = dict(K_oracle=best_k, actual=actual)

    return results


# ── lag walk-forward ──────────────────────────────────────────────────────────
def wf_fixed(oracle_map, lp_big, confirm_big, dir_big, pool_cache):
    """Baseline: T_frac=3.6%, K=75."""
    X_big  = build_X(lp_big, P)
    grid   = np.array(sorted(pool_cache.keys()))
    tf_key = float(grid[np.argmin(np.abs(grid - T_FIXED))])
    errs, acts = [], []
    for step in sorted(oracle_map):
        if step < MIN_HISTORY or step + H >= len(lp_big): continue
        if np.any(np.isnan(X_big[step])): continue
        lp_f1, conf_f1, dir_f1, Xf1, y_rel = pool_cache[tf_key]
        X_pool, y_pool = causal_pool(confirm_big[step], int(dir_big[step]),
                                     conf_f1, Xf1, dir_f1, y_rel)
        if X_pool is None: continue
        p_rel = lwr_predict_k(X_big[step][1:], X_pool, y_pool, K_FIXED)
        actual = float(np.exp(lp_big[step + H]))
        if np.isfinite(p_rel):
            errs.append(float(np.exp(lp_big[step] + p_rel)) - actual)
            acts.append(actual)
    return np.array(errs), np.array(acts)


def wf_lag_tf(oracle_map, lp_big, confirm_big, dir_big, pool_cache, lag=1):
    """Lag-N T_frac: берём tf_oracle(t-lag), K=75 фиксированный."""
    X_big  = build_X(lp_big, P)
    grid   = np.array(sorted(pool_cache.keys()))
    steps  = sorted(oracle_map)
    errs, acts = [], []
    for i, step in enumerate(steps):
        if i < lag or step < MIN_HISTORY or step + H >= len(lp_big): continue
        if np.any(np.isnan(X_big[step])): continue
        prev  = steps[i - lag]
        tf    = oracle_map[prev]["tf_oracle"]
        tf_key = float(grid[np.argmin(np.abs(grid - tf))])
        lp_f1, conf_f1, dir_f1, Xf1, y_rel = pool_cache[tf_key]
        X_pool, y_pool = causal_pool(confirm_big[step], int(dir_big[step]),
                                     conf_f1, Xf1, dir_f1, y_rel)
        if X_pool is None: continue
        p_rel = lwr_predict_k(X_big[step][1:], X_pool, y_pool, K_FIXED)
        actual = float(np.exp(lp_big[step + H]))
        if np.isfinite(p_rel):
            errs.append(float(np.exp(lp_big[step] + p_rel)) - actual)
            acts.append(actual)
    return np.array(errs), np.array(acts)


def wf_lag_K(k_oracle_map, oracle_map, lp_big, confirm_big, dir_big,
             pool_fixed, lag=1):
    """Lag-N K: берём K_oracle(t-lag), T_frac=3.6% фиксированный."""
    X_big  = build_X(lp_big, P)
    steps  = sorted(k_oracle_map)
    lp_f1, conf_f1, dir_f1, Xf1, y_rel = pool_fixed
    errs, acts = [], []
    for i, step in enumerate(steps):
        if i < lag or step < MIN_HISTORY or step + H >= len(lp_big): continue
        if np.any(np.isnan(X_big[step])): continue
        prev = steps[i - lag]
        k    = k_oracle_map[prev]["K_oracle"]
        X_pool, y_pool = causal_pool(confirm_big[step], int(dir_big[step]),
                                     conf_f1, Xf1, dir_f1, y_rel)
        if X_pool is None: continue
        p_rel = lwr_predict_k(X_big[step][1:], X_pool, y_pool, k)
        actual = float(np.exp(lp_big[step + H]))
        if np.isfinite(p_rel):
            errs.append(float(np.exp(lp_big[step] + p_rel)) - actual)
            acts.append(actual)
    return np.array(errs), np.array(acts)


# ── визуализация ──────────────────────────────────────────────────────────────
def plot_all(confirm_big, oracle_map, k_oracle_map, lag_results, out_path):
    parse  = lambda s: datetime.strptime(s[:16], "%Y-%m-%d %H:%M")

    tf_steps = sorted(oracle_map)
    k_steps  = sorted(k_oracle_map)
    tf_seq   = np.array([oracle_map[s]["tf_oracle"] for s in tf_steps])
    k_seq    = np.array([k_oracle_map[s]["K_oracle"] for s in k_steps])
    dt_tf    = np.array([parse(confirm_big[s]) for s in tf_steps])
    dt_k     = np.array([parse(confirm_big[s]) for s in k_steps])

    ci = 1.96 / np.sqrt(len(tf_seq))
    lags_tf, acf_tf = compute_acf(tf_seq, 15)
    lags_k,  acf_k  = compute_acf(k_seq,  15)

    W = 50
    fig = plt.figure(figsize=(16, 16))
    gs  = fig.add_gridspec(4, 2, hspace=0.40, wspace=0.30)

    # ── tf по времени ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(dt_tf, tf_seq * 100, color="steelblue", lw=0.7, alpha=0.7)
    ax.axhline(np.median(tf_seq) * 100, color="tomato", lw=1.0, ls="--",
               label=f"median {np.median(tf_seq)*100:.1f}%")
    ax.set_ylabel("tf_oracle (%)")
    ax.set_title("tf_oracle по времени  (K=75)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

    # ── K по времени ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.step(dt_k, k_seq, color="darkorange", lw=0.9, where="post", alpha=0.8)
    ax.axhline(np.median(k_seq), color="tomato", lw=1.0, ls="--",
               label=f"median {int(np.median(k_seq))}")
    ax.set_ylabel("K_oracle")
    ax.set_title("K_oracle по времени  (T_frac=3.6%)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))

    # ── ACF tf ───────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(lags_tf, acf_tf, color="steelblue", alpha=0.75, width=0.6)
    ax.axhline( ci, color="gray", lw=0.8, ls="--", label=f"95% CI ±{ci:.3f}")
    ax.axhline(-ci, color="gray", lw=0.8, ls="--")
    ax.axhline(0,   color="black", lw=0.6)
    ax.set_title(f"ACF tf_oracle  (lag-1={acf_tf[0]:+.3f})")
    ax.set_xlabel("Лаг (события)"); ax.set_ylabel("ACF")
    ax.legend(fontsize=8)

    # ── ACF K ────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(lags_k, acf_k, color="darkorange", alpha=0.75, width=0.6)
    ax.axhline( ci, color="gray", lw=0.8, ls="--", label=f"95% CI ±{ci:.3f}")
    ax.axhline(-ci, color="gray", lw=0.8, ls="--")
    ax.axhline(0,   color="black", lw=0.6)
    ax.set_title(f"ACF K_oracle  (lag-1={acf_k[0]:+.3f})")
    ax.set_xlabel("Лаг (события)"); ax.set_ylabel("ACF")
    ax.legend(fontsize=8)

    # ── гистограмма |Δtf| ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    dtf = np.abs(np.diff(tf_seq)) * 100
    ax.hist(dtf, bins=25, color="steelblue", alpha=0.75, edgecolor="white")
    ax.set_xlabel("|Δtf_oracle| (%)")
    ax.set_ylabel("событий")
    pct0 = (dtf < 0.15).mean() * 100
    ax.set_title(f"|Δtf|: median={np.median(dtf):.2f}%  "
                 f"без изм.(≤0.15%): {pct0:.1f}%")

    # ── гистограмма K ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    vals, cnts = np.unique(k_seq, return_counts=True)
    ax.bar([str(int(v)) for v in vals], cnts / len(k_seq) * 100,
           color="darkorange", alpha=0.75)
    ax.set_xlabel("K_oracle"); ax.set_ylabel("%")
    same = (np.diff(k_seq) == 0).mean() * 100
    ax.set_title(f"Распределение K_oracle  (без изм.: {same:.1f}%)")

    # ── rolling rMAE: стратегии ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, :])
    colors = {"fixed T=3.6% K=75": "black",
              "tf lag-1":          "steelblue",
              "tf lag-2":          "cornflowerblue",
              "K  lag-1":          "darkorange",
              "K  lag-2":          "peachpuff"}
    all_steps = sorted(oracle_map)
    all_dt    = np.array([parse(confirm_big[s]) for s in all_steps])

    for name, (e, a) in lag_results.items():
        if len(e) < 2: continue
        roll = [rmae(e[max(0,i-W+1):i+1], a[max(0,i-W+1):i+1])
                for i in range(len(e))]
        r    = rmae(e, a)
        n    = min(len(roll), len(all_dt))
        ax.plot(all_dt[:n], roll[:n], lw=0.9,
                color=colors.get(name, "gray"),
                label=f"{name}  ({r:.4f})")

    if REF_CSV.exists():
        ref    = np.loadtxt(REF_CSV, delimiter=",", skiprows=1)
        re, ra = ref[:, 1], ref[:, 2]
        rroll  = [rmae(re[max(0,i-W+1):i+1], ra[max(0,i-W+1):i+1])
                  for i in range(len(re))]
        n = min(len(rroll), len(all_dt))
        ax.plot(all_dt[:n], rroll[:n], color="red", lw=1.2, ls="--",
                label=f"05_best ({REF_RMAE})", zorder=5)

    ax.set_ylabel("rMAE (rolling W=50)")
    ax.set_title("Стратегии vs baseline (rolling rMAE)")
    ax.legend(fontsize=8, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.suptitle(f"Стабильность K и T_frac  |  SBER {INTERVAL}"
                 f"  T_big={T_BIG*100:.1f}%", fontsize=12, y=1.002)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"T_BIG={T_BIG*100:.1f}%  INTERVAL={INTERVAL}")
    print(f"tf_oracle: K={K_FIXED} фикс, T_frac свип {len(T_FRAC_GRID)} значений")
    print(f"K_oracle:  T_frac={T_FIXED*100:.1f}% фикс, K свип {K_GRID}")
    print()

    h, l, d = load_tf(INTERVAL)
    lp_big, confirm_big, dir_big, _ = find_pivots_log(h, l, d, T_BIG)
    print(f"T_BIG: {len(lp_big)} пивотов")

    oracle_map = load_oracle(ORACLE)
    print(f"tf_oracle записей: {len(oracle_map)}")

    print(f"Предвычисляем {len(T_FRAC_GRID)} зигзагов ...", flush=True)
    pool_cache = {}
    for tf in T_FRAC_GRID:
        lp_f, conf_f, dir_f, _ = find_pivots_log(h, l, d, tf)
        Xf1   = build_X(lp_f, P)
        y_rel = np.array([lp_f[j+H] - lp_f[j] if j+H < len(lp_f) else np.nan
                          for j in range(len(lp_f))])
        pool_cache[tf] = (lp_f, conf_f, dir_f, Xf1, y_rel)

    tf_fixed_key = float(T_FRAC_GRID[np.argmin(np.abs(T_FRAC_GRID - T_FIXED))])
    pool_fixed   = pool_cache[tf_fixed_key]

    print(f"K-oracle свип ({len(K_GRID)} K) ...", flush=True)
    k_oracle_map = k_oracle_sweep(oracle_map, lp_big, confirm_big, dir_big, pool_fixed)
    print(f"  записей: {len(k_oracle_map)}")
    print()

    # ── статистика ───────────────────────────────────────────────────────────
    tf_seq = np.array([oracle_map[s]["tf_oracle"] for s in sorted(oracle_map)])
    k_seq  = np.array([k_oracle_map[s]["K_oracle"] for s in sorted(k_oracle_map)])

    print("=== tf_oracle (K=75) ===")
    print(f"  median={np.median(tf_seq)*100:.2f}%  std={tf_seq.std()*100:.2f}%")
    dtf = np.abs(np.diff(tf_seq))
    print(f"  |Δtf| между событиями: median={np.median(dtf)*100:.2f}%"
          f"  p75={np.percentile(dtf,75)*100:.2f}%")
    _, a_tf = compute_acf(tf_seq, 5)
    print(f"  ACF: lag1={a_tf[0]:+.3f}  lag2={a_tf[1]:+.3f}  lag3={a_tf[2]:+.3f}")
    print()

    print("=== K_oracle (T_frac=3.6%) ===")
    vals, cnts = np.unique(k_seq, return_counts=True)
    for v, c in zip(vals, cnts):
        print(f"  K={int(v):3d}  {c:4d} ({c/len(k_seq)*100:.1f}%)")
    same = (np.diff(k_seq) == 0).mean() * 100
    print(f"  Без изменений: {same:.1f}%")
    _, a_k = compute_acf(k_seq, 5)
    print(f"  ACF: lag1={a_k[0]:+.3f}  lag2={a_k[1]:+.3f}  lag3={a_k[2]:+.3f}")
    print()

    # ── lag-стратегии ────────────────────────────────────────────────────────
    print("Lag-стратегии ...", flush=True)
    e_fix, a_fix = wf_fixed(oracle_map, lp_big, confirm_big, dir_big, pool_cache)

    lag_results = {"fixed T=3.6% K=75": (e_fix, a_fix)}
    for lag in (1, 2):
        e, a = wf_lag_tf(oracle_map, lp_big, confirm_big, dir_big, pool_cache, lag)
        lag_results[f"tf lag-{lag}"] = (e, a)
        e, a = wf_lag_K(k_oracle_map, oracle_map, lp_big, confirm_big,
                        dir_big, pool_fixed, lag)
        lag_results[f"K  lag-{lag}"] = (e, a)

    print()
    r_fix = rmae(e_fix, a_fix)
    print(f"{'─'*52}")
    print(f"{'Стратегия':<22}  {'rMAE':>8}  {'Δ vs fixed':>10}  {'n':>5}")
    for name, (e, a) in lag_results.items():
        r = rmae(e, a)
        d = (r - r_fix) / r_fix * 100 if np.isfinite(r_fix) else np.nan
        print(f"  {name:<20}  {r:>8.4f}  {d:>+9.1f}%  {len(e):>5}")
    print(f"\n  05_best ref             {REF_RMAE:>8.4f}")

    # CSV
    out_csv = RESULTS / f"stability_{INTERVAL}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "tf_oracle", "K_oracle"])
        for s in sorted(set(oracle_map) & set(k_oracle_map)):
            w.writerow([s, oracle_map[s]["tf_oracle"], k_oracle_map[s]["K_oracle"]])
    print(f"\nCSV → {out_csv}")

    plot_all(confirm_big, oracle_map, k_oracle_map, lag_results,
             RESULTS / f"stability_{INTERVAL}.png")


if __name__ == "__main__":
    main()
