#!/usr/bin/env python3
"""
08_adaptive_fd.py — Адаптивный T_FRAC через Higuchi FD + Richardson с κ-коррекцией

Формула:
    H_raw = 2 - D            (Higuchi на log(high/low))
    H_eff = kappa * H_raw    (κ — коэффициент ослабления памяти)
    T_frac = T_big * r^(-H_eff)

Мотивация κ: зигзаг — пороговый фильтр. Higuchi D считается на сыром ряде,
но зигзаг-события соответствуют более слабой памяти (мелкие флуктуации удалены).
κ < 1 корректирует эффективный Hurst под фильтрованный процесс.

Sweep: κ ∈ [0.05, 1.0], шаг 0.05. Два режима D:
  raw2 : Higuchi на log(high/low), окно = 2 крупных движения
  raw1 : Higuchi на log(high/low), окно = 1 крупное движение

Референс: 05_lwr_best (фиксированный T_frac=3.6%), rMAE=0.3714
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
RESULTS.mkdir(exist_ok=True)
REF_CSV = HERE.parent / "05_lwr_improvements" / "results" / "best_10m_T4_Tf36.csv"

T_BIG    = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
K_MAX    = int(sys.argv[2])   if len(sys.argv) > 2 else 10
R        = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
INTERVAL = sys.argv[4]        if len(sys.argv) > 4 else "10m"

H           = 1
P           = 3
K           = 75
MIN_HISTORY = 50
D_FALLBACK  = 1.43
T_FRAC_MIN  = 0.005
T_FRAC_MAX  = T_BIG * 0.99
T_FRAC_GRID = np.round(np.arange(0.005, T_BIG, 0.001), 4)

KAPPA_GRID = np.round(np.arange(0.05, 1.05, 0.05), 2)   # 20 значений

REF_RMAE = 0.3714


# ── данные ───────────────────────────────────────────────────────────────────
def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


# ── зигзаг ───────────────────────────────────────────────────────────────────
def find_pivots_log(highs, lows, dates, thr):
    lh, ll = np.log(highs), np.log(lows)
    vals_log, pivot_dates, confirm_dates, dirs, confirm_bars = [], [], [], [], []
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
                vals_log.append(ext_val); pivot_dates.append(dates[ext_idx])
                confirm_dates.append(dates[i]); dirs.append(+1); confirm_bars.append(i)
                direction, ext_val, ext_idx = -1, ll[i], i
        else:
            if ll[i] < ext_val:
                ext_val, ext_idx = ll[i], i
            elif lh[i] - ext_val >= thr:
                vals_log.append(ext_val); pivot_dates.append(dates[ext_idx])
                confirm_dates.append(dates[i]); dirs.append(-1); confirm_bars.append(i)
                direction, ext_val, ext_idx = 1, lh[i], i
    return (np.array(vals_log), np.array(pivot_dates),
            np.array(confirm_dates), np.array(dirs), np.array(confirm_bars))


# ── эмбеддинг ─────────────────────────────────────────────────────────────────
def build_X(log_prices, p):
    n = len(log_prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = log_prices[i]
        for lag in range(1, p):
            X[i, lag] = log_prices[i - lag + 1] - log_prices[i - lag]
    return X


# ── Higuchi FD ────────────────────────────────────────────────────────────────
def higuchi_fd(x, k_max):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < k_max * 3:
        return np.nan
    L = np.empty(k_max)
    for k in range(1, k_max + 1):
        Lk, n_loops = 0.0, 0
        for m in range(1, k + 1):
            sub = x[m - 1::k]
            n_m = len(sub) - 1
            if n_m < 1:
                continue
            Lmk = np.sum(np.abs(np.diff(sub))) * (N - 1) / (n_m * k)
            Lk += Lmk
            n_loops += 1
        L[k - 1] = Lk / (k * n_loops) if n_loops > 0 else np.nan
    ks = np.arange(1, k_max + 1, dtype=float)
    valid = np.isfinite(L) & (L > 0)
    if valid.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(np.log(ks[valid]), np.log(L[valid]), 1)
    return -slope


# ── два источника D ───────────────────────────────────────────────────────────
def fd_raw(lh, ll, confirm_bars, step, n_swings):
    if step < n_swings:
        return np.nan
    bar_from = int(confirm_bars[step - n_swings])
    bar_to   = int(confirm_bars[step])
    if bar_to - bar_from < K_MAX * 3:
        return np.nan
    dh = higuchi_fd(lh[bar_from:bar_to + 1], K_MAX)
    dl = higuchi_fd(ll[bar_from:bar_to + 1], K_MAX)
    vals = [v for v in [dh, dl] if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def richardson_tfrac(D, kappa):
    """
    H_raw = 2 - D                (Higuchi на сыром ряде)
    H_eff = kappa * H_raw        (эффективный Hurst для зигзаг-событий)
    T_frac = T_big * r^(-H_eff)

    κ=1: чистый Richardson (T_frac≈2.73% при D=1.45, r=2)
    κ=0: H_eff=0 → T_frac=T_big (фрактальных уровней нет)
    κ<1: ослабленная память (пороговая фильтрация зигзага уменьшает видимый Hurst)
    """
    if not np.isfinite(D) or D >= 2.0:
        D = D_FALLBACK
    H_raw = 2.0 - D
    H_eff = kappa * H_raw
    return float(np.clip(T_BIG * R ** (-H_eff), T_FRAC_MIN, T_FRAC_MAX))


# ── каузальная обрезка пула ───────────────────────────────────────────────────
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


# ── LWR ──────────────────────────────────────────────────────────────────────
def lwr_predict(xq, X_pool, y_pool):
    dists = np.linalg.norm(X_pool - xq, axis=1)
    keff  = min(K, len(dists))
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
    acts = np.asarray(acts, dtype=float)
    pers = np.abs(acts[2:] - acts[:-2]) if len(acts) >= 3 else np.abs(np.diff(acts))
    dz   = float(np.mean(pers)) if len(pers) > 0 else 1.0
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


# ── walk-forward ──────────────────────────────────────────────────────────────
def walk_forward(lp_big, confirm_big, dir_big, confirm_bars_big, n_big,
                 lh, ll, pool_cache, n_swings, kappa):
    grid  = np.array(sorted(pool_cache.keys()))
    X_big = build_X(lp_big, P)

    steps, errs, acts, tf_used, d_used = [], [], [], [], []

    for step in range(MIN_HISTORY, n_big - H):
        if np.any(np.isnan(X_big[step])):
            continue

        D = fd_raw(lh, ll, confirm_bars_big, step, n_swings)
        if not np.isfinite(D):
            D = D_FALLBACK

        t_frac   = richardson_tfrac(D, kappa)
        t_frac_q = float(grid[np.argmin(np.abs(grid - t_frac))])

        lp_f1, confirm_f1, dir_f1, Xf1, y_rel = pool_cache[t_frac_q]

        X_pool, y_pool = causal_pool(
            confirm_big[step], int(dir_big[step]),
            confirm_f1, Xf1, dir_f1, y_rel
        )
        if X_pool is None:
            continue

        xq       = X_big[step][1:]
        pred_rel = lwr_predict(xq, X_pool, y_pool)
        if not np.isfinite(pred_rel):
            continue

        pred   = float(np.exp(lp_big[step] + pred_rel))
        actual = float(np.exp(lp_big[step + H]))

        steps.append(step)
        errs.append(pred - actual)
        acts.append(actual)
        tf_used.append(t_frac_q)
        d_used.append(D)

    return (np.array(steps), np.array(errs), np.array(acts),
            np.array(tf_used), np.array(d_used))


# ── sweep κ ───────────────────────────────────────────────────────────────────
def sweep_kappa(lp_big, confirm_big, dir_big, confirm_bars_big, n_big,
                lh, ll, pool_cache, n_swings):
    """Возвращает массивы kappa_grid и rMAE для каждого κ."""
    rmaes = []
    for kappa in KAPPA_GRID:
        steps, errs, acts, _, _ = walk_forward(
            lp_big, confirm_big, dir_big, confirm_bars_big, n_big,
            lh, ll, pool_cache, n_swings, kappa
        )
        rmaes.append(rmae(errs, acts))
    return np.array(rmaes)


# ── визуализация ──────────────────────────────────────────────────────────────
def plot_results(confirm_dates, lp_big,
                 kappa_grid, rmae2, rmae1,
                 best_kappa2, best_kappa1,
                 res_best2, res_best1, out_path):
    parse  = lambda s: datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    all_dt = np.array([parse(s) for s in confirm_dates])
    W      = 50

    fig, axes = plt.subplots(4, 1, figsize=(16, 12),
                              gridspec_kw={"height_ratios": [1.5, 1, 1, 0.9]})

    # ── sweep κ vs rMAE ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(kappa_grid, rmae2, color="darkorange", marker="o", ms=5, lw=1.2,
            label=f"raw N=2  (opt κ={best_kappa2:.2f})")
    ax.plot(kappa_grid, rmae1, color="tomato",     marker="s", ms=5, lw=1.2,
            label=f"raw N=1  (opt κ={best_kappa1:.2f})")
    ax.axhline(REF_RMAE, color="steelblue", lw=1.0, ls="--", label=f"05_best {REF_RMAE}")
    ax.axvline(best_kappa2, color="darkorange", lw=0.7, ls=":")
    ax.axvline(best_kappa1, color="tomato",     lw=0.7, ls=":")

    best2 = rmae2[np.argmin(rmae2)]
    best1 = rmae1[np.argmin(rmae1)]
    ax.scatter([best_kappa2], [best2], color="darkorange", s=80, zorder=5)
    ax.scatter([best_kappa1], [best1], color="tomato",     s=80, zorder=5)

    ax.set_xlabel("κ (коэффициент ослабления памяти)")
    ax.set_ylabel("rMAE")
    ax.set_title(
        f"SBER {INTERVAL} | T_big={T_BIG*100:.1f}% | r={R} | K_MAX={K_MAX}\n"
        f"Sweep κ: raw2 opt={best2:.4f}  raw1 opt={best1:.4f}  05_best={REF_RMAE}"
    )
    ax.legend(fontsize=9)

    # ── rolling rMAE при оптимальном κ ───────────────────────────────────────
    ax = axes[1]
    for res, col, lbl in [(res_best2, "darkorange", f"raw N=2 κ={best_kappa2:.2f}"),
                           (res_best1, "tomato",     f"raw N=1 κ={best_kappa1:.2f}")]:
        e, a, s = res["errs"], res["acts"], res["steps"]
        roll = [rmae(e[max(0, i-W+1):i+1], a[max(0, i-W+1):i+1])
                for i in range(len(e))]
        ax.plot(all_dt[s], roll, color=col, lw=0.9, label=f"{lbl} (global {rmae(e,a):.4f})")

    if REF_CSV.exists():
        ref   = np.loadtxt(REF_CSV, delimiter=",", skiprows=1)
        re, ra = ref[:, 1], ref[:, 2]
        rroll  = [rmae(re[max(0, i-W+1):i+1], ra[max(0, i-W+1):i+1])
                  for i in range(len(re))]
        n = min(len(rroll), len(res_best2["steps"]))
        ax.plot(all_dt[res_best2["steps"][:n]], rroll[:n],
                color="steelblue", lw=1.2, label=f"05_best ({REF_RMAE})", zorder=5)

    ax.set_ylabel("rMAE (rolling W=50)")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── T_frac при оптимальном κ ─────────────────────────────────────────────
    ax = axes[2]
    for res, col, lbl in [(res_best2, "darkorange", f"raw N=2 κ={best_kappa2:.2f}"),
                           (res_best1, "tomato",     f"raw N=1 κ={best_kappa1:.2f}")]:
        ax.plot(all_dt[res["steps"]], res["tf_used"] * 100,
                color=col, lw=0.8, alpha=0.85, label=lbl)
    ax.axhline(3.6, color="steelblue", lw=0.8, ls="--", label="05_best 3.6%")
    ax.set_ylabel("T_frac (%)")
    ax.legend(fontsize=8, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── D (сырой) ─────────────────────────────────────────────────────────────
    ax = axes[3]
    for res, col, lbl in [(res_best2, "darkorange", "raw N=2  D"),
                           (res_best1, "tomato",     "raw N=1  D")]:
        ax.plot(all_dt[res["steps"]], res["d_used"],
                color=col, lw=0.7, alpha=0.75, label=lbl)
    ax.axhline(1.5,        color="gray", lw=0.6, ls="--", label="BM 1.5")
    ax.axhline(D_FALLBACK, color="gray", lw=0.6, ls=":",  label=f"fallback {D_FALLBACK}")
    ax.set_ylabel("D (Higuchi)")
    ax.legend(fontsize=7, ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"T_BIG={T_BIG*100:.1f}%  K_MAX={K_MAX}  R={R}  INTERVAL={INTERVAL}")
    print(f"H_eff = κ × (2-D)   T_frac = T_big × r^(-H_eff)")
    print(f"κ sweep: {KAPPA_GRID}")
    print()

    h, l, d = load_tf(INTERVAL)
    lh, ll  = np.log(h), np.log(l)

    lp_big, _, confirm_big, dir_big, confirm_bars_big = find_pivots_log(h, l, d, T_BIG)
    n_big = len(lp_big)
    print(f"T_BIG: {n_big} пивотов  |  баров: {len(h)}")

    print(f"Предвычисляем {len(T_FRAC_GRID)} зигзагов ...", flush=True)
    pool_cache = {}
    for tf in T_FRAC_GRID:
        lp_f, _, conf_f, dir_f, _ = find_pivots_log(h, l, d, tf)
        Xf1   = build_X(lp_f, P)
        y_rel = np.array([lp_f[j+H] - lp_f[j] if j+H < len(lp_f) else np.nan
                          for j in range(len(lp_f))])
        pool_cache[tf] = (lp_f, conf_f, dir_f, Xf1, y_rel)
    print()

    # ── sweep κ ──────────────────────────────────────────────────────────────
    for n_swings, mode_name in [(2, "raw2"), (1, "raw1")]:
        print(f"Sweep κ [{mode_name}] ...", flush=True)
        rmaes = sweep_kappa(lp_big, confirm_big, dir_big, confirm_bars_big, n_big,
                            lh, ll, pool_cache, n_swings)
        if mode_name == "raw2":
            rmae2 = rmaes
        else:
            rmae1 = rmaes
        for kappa, r in zip(KAPPA_GRID, rmaes):
            marker = " ◄ opt" if r == np.nanmin(rmaes) else ""
            print(f"  κ={kappa:.2f}  rMAE={r:.4f}{marker}")
        print()

    best_kappa2 = KAPPA_GRID[np.nanargmin(rmae2)]
    best_kappa1 = KAPPA_GRID[np.nanargmin(rmae1)]

    print(f"{'─'*50}")
    print(f"05_best фиксированный T_frac=3.6%:  rMAE={REF_RMAE}")
    print(f"raw2  opt κ={best_kappa2:.2f}  rMAE={rmae2.min():.4f}"
          f"  Δ={(rmae2.min()-REF_RMAE)/REF_RMAE*100:+.1f}%")
    print(f"raw1  opt κ={best_kappa1:.2f}  rMAE={rmae1.min():.4f}"
          f"  Δ={(rmae1.min()-REF_RMAE)/REF_RMAE*100:+.1f}%")
    print()

    # ── полный прогон при оптимальном κ ─────────────────────────────────────
    print("Прогон при оптимальном κ ...", flush=True)
    res_best2 = {}
    steps, errs, acts, tf, dv = walk_forward(
        lp_big, confirm_big, dir_big, confirm_bars_big, n_big,
        lh, ll, pool_cache, 2, best_kappa2
    )
    res_best2 = dict(steps=steps, errs=errs, acts=acts, tf_used=tf, d_used=dv)

    steps, errs, acts, tf, dv = walk_forward(
        lp_big, confirm_big, dir_big, confirm_bars_big, n_big,
        lh, ll, pool_cache, 1, best_kappa1
    )
    res_best1 = dict(steps=steps, errs=errs, acts=acts, tf_used=tf, d_used=dv)

    # T_frac статистика
    for name, res, kappa in [("raw2", res_best2, best_kappa2),
                               ("raw1", res_best1, best_kappa1)]:
        tf_arr = res["tf_used"]
        print(f"  {name} κ={kappa:.2f}: T_frac mean={tf_arr.mean()*100:.2f}%"
              f"  std={tf_arr.std()*100:.2f}%"
              f"  min={tf_arr.min()*100:.2f}%  max={tf_arr.max()*100:.2f}%")

    # CSV sweep
    out_csv = RESULTS / f"kappa_sweep_{INTERVAL}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kappa", "rmae_raw2", "rmae_raw1"])
        for kappa, r2, r1 in zip(KAPPA_GRID, rmae2, rmae1):
            w.writerow([kappa, r2, r1])
    print(f"\nSweep CSV → {out_csv}")

    fig_path = RESULTS / f"kappa_sweep_{INTERVAL}.png"
    plot_results(confirm_big, lp_big,
                 KAPPA_GRID, rmae2, rmae1,
                 best_kappa2, best_kappa1,
                 res_best2, res_best1, fig_path)


if __name__ == "__main__":
    main()
