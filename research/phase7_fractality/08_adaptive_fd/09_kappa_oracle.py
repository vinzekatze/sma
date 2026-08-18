#!/usr/bin/env python3
"""
09_kappa_oracle.py — Per-event оракул: какой κ был бы оптимален на каждом шаге?

Для каждого шага t:
  1. Прогнозируем со всеми T_frac из сетки (pool_cache уже построен).
  2. Находим T_frac_opt = argmin |err| — лучший T_frac для данного исхода.
  3. Вычисляем κ_oracle = -log(T_frac_opt / T_big) / (H_raw * log(R)).

Диагностика:
  - Если std(κ_oracle) мала → κ=0.30 уже оптимален, адаптация бесполезна.
  - Если велика и коррелирует с D → есть сигнал для rolling κ.

Выходы:
  - Таблица: step, T_frac_opt, κ_oracle, D, err_oracle, err_fixed (κ=0.30)
  - Потолок: oracle rMAE vs fixed κ=0.30 vs 05_best
  - График: 5 панелей — sweep-кривые, κ vs time, κ vs D, T_frac_opt vs time,
             rolling rMAE oracle vs fixed vs 05_best
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

KAPPA_FIXED = 0.30    # оптимум из скрипта 08
REF_RMAE    = 0.3714  # 05_best


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


def fd_raw(lh, ll, confirm_bars, step, n_swings=2):
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


def kappa_from_tfrac(t_frac_opt, H_raw):
    """κ = -log(T_frac_opt/T_big) / (H_raw * log(R))."""
    if H_raw < 1e-6 or t_frac_opt <= 0:
        return np.nan
    return -np.log(t_frac_opt / T_BIG) / (H_raw * np.log(R))


def richardson_tfrac(D, kappa):
    if not np.isfinite(D) or D >= 2.0:
        D = D_FALLBACK
    H_eff = kappa * (2.0 - D)
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


# ── oracle walk-forward ───────────────────────────────────────────────────────
def oracle_walk_forward(lp_big, confirm_big, dir_big, confirm_bars_big, n_big,
                        lh, ll, pool_cache):
    """
    Для каждого шага предсказывает всеми T_frac → находит лучший.
    Возвращает per-step записи + прогноз фиксированным κ=KAPPA_FIXED.
    """
    grid  = np.array(sorted(pool_cache.keys()))
    X_big = build_X(lp_big, P)

    records = []

    for step in range(MIN_HISTORY, n_big - H):
        if np.any(np.isnan(X_big[step])):
            continue

        D      = fd_raw(lh, ll, confirm_bars_big, step, n_swings=2)
        D_safe = D if np.isfinite(D) else D_FALLBACK
        H_raw  = 2.0 - D_safe
        xq     = X_big[step][1:]
        actual = float(np.exp(lp_big[step + H]))

        # ── прогноз по каждому T_frac ─────────────────────────────────────
        preds_by_tf = {}
        for tf in grid:
            lp_f1, confirm_f1, dir_f1, Xf1, y_rel = pool_cache[tf]
            X_pool, y_pool = causal_pool(
                confirm_big[step], int(dir_big[step]),
                confirm_f1, Xf1, dir_f1, y_rel
            )
            if X_pool is None:
                continue
            pred_rel = lwr_predict(xq, X_pool, y_pool)
            if np.isfinite(pred_rel):
                preds_by_tf[tf] = float(np.exp(lp_big[step] + pred_rel))

        if not preds_by_tf:
            continue

        # ── лучший T_frac (оракул) ────────────────────────────────────────
        best_tf  = min(preds_by_tf, key=lambda tf: abs(preds_by_tf[tf] - actual))
        best_err = preds_by_tf[best_tf] - actual
        kappa_oracle = kappa_from_tfrac(best_tf, H_raw)

        # ── фиксированный κ=0.30 ──────────────────────────────────────────
        tf_fixed = float(grid[np.argmin(np.abs(grid - richardson_tfrac(D_safe, KAPPA_FIXED)))])
        if tf_fixed in preds_by_tf:
            err_fixed = preds_by_tf[tf_fixed] - actual
        else:
            err_fixed = np.nan

        records.append(dict(
            step          = step,
            D             = D_safe,
            H_raw         = H_raw,
            tf_oracle     = best_tf,
            kappa_oracle  = kappa_oracle,
            err_oracle    = best_err,
            tf_fixed      = tf_fixed,
            err_fixed     = err_fixed,
            actual        = actual,
        ))

    return records


# ── визуализация ──────────────────────────────────────────────────────────────
def plot(confirm_dates, lp_big, records, out_path):
    parse  = lambda s: datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    all_dt = np.array([parse(s) for s in confirm_dates])
    W      = 50

    steps     = np.array([r["step"]         for r in records])
    kappas    = np.array([r["kappa_oracle"]  for r in records])
    ds        = np.array([r["D"]             for r in records])
    tf_oracle = np.array([r["tf_oracle"]     for r in records])
    tf_fixed  = np.array([r["tf_fixed"]      for r in records])
    err_or    = np.array([r["err_oracle"]    for r in records])
    err_fx    = np.array([r["err_fixed"]     for r in records])
    acts      = np.array([r["actual"]        for r in records])
    dt        = all_dt[steps]

    kappa_fin = kappas[np.isfinite(kappas)]

    fig, axes = plt.subplots(5, 1, figsize=(16, 16),
                              gridspec_kw={"height_ratios": [1.2, 1, 0.9, 0.9, 1.2]})

    # ── 1. гистограмма κ ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.hist(kappa_fin, bins=40, color="steelblue", alpha=0.75, edgecolor="white")
    ax.axvline(np.median(kappa_fin), color="darkorange", lw=1.5, ls="--",
               label=f"median κ={np.median(kappa_fin):.3f}")
    ax.axvline(KAPPA_FIXED, color="tomato", lw=1.5, ls=":",
               label=f"κ_fixed={KAPPA_FIXED}")
    ax.set_xlabel("κ_oracle")
    ax.set_ylabel("событий")
    ax.set_title(
        f"SBER {INTERVAL} | T_big={T_BIG*100:.1f}% | r={R}\n"
        f"κ_oracle: mean={kappa_fin.mean():.3f}  median={np.median(kappa_fin):.3f}"
        f"  std={kappa_fin.std():.3f}  p5={np.percentile(kappa_fin,5):.3f}"
        f"  p95={np.percentile(kappa_fin,95):.3f}"
    )
    ax.legend(fontsize=9)

    # ── 2. κ по времени ──────────────────────────────────────────────────────
    ax = axes[1]
    mask = np.isfinite(kappas)
    ax.plot(dt[mask], kappas[mask], color="steelblue", lw=0.7, alpha=0.7)
    ax.axhline(np.median(kappa_fin), color="darkorange", lw=1.0, ls="--",
               label=f"median {np.median(kappa_fin):.3f}")
    ax.axhline(KAPPA_FIXED, color="tomato", lw=1.0, ls=":",
               label=f"κ_fixed={KAPPA_FIXED}")
    ax.set_ylabel("κ_oracle")
    ax.set_ylim(-0.1, min(kappa_fin.max() * 1.1, 5.0))
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── 3. scatter κ vs D ────────────────────────────────────────────────────
    ax = axes[2]
    ax.scatter(ds[mask], kappas[mask], s=6, alpha=0.4, color="steelblue")
    corr = np.corrcoef(ds[mask], kappas[mask])[0, 1]
    ax.set_xlabel("D (Higuchi)")
    ax.set_ylabel("κ_oracle")
    ax.set_title(f"κ vs D   r={corr:.3f}")
    ax.set_ylim(-0.1, min(kappa_fin.max() * 1.1, 5.0))
    # линия тренда
    if mask.sum() > 5:
        z = np.polyfit(ds[mask], kappas[mask], 1)
        xr = np.linspace(ds[mask].min(), ds[mask].max(), 50)
        ax.plot(xr, np.polyval(z, xr), color="tomato", lw=1.2)

    # ── 4. T_frac_opt по времени ─────────────────────────────────────────────
    ax = axes[3]
    ax.plot(dt, tf_oracle * 100, color="steelblue", lw=0.7, alpha=0.7,
            label="T_frac oracle")
    ax.plot(dt, tf_fixed  * 100, color="darkorange", lw=0.8, alpha=0.6,
            label=f"T_frac κ={KAPPA_FIXED}")
    ax.axhline(3.6, color="gray", lw=0.8, ls="--", label="05_best 3.6%")
    ax.set_ylabel("T_frac (%)")
    ax.legend(fontsize=8, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── 5. rolling rMAE: оракул vs fixed vs 05_best ──────────────────────────
    ax = axes[4]
    for err, col, lbl in [(err_or, "steelblue",  f"oracle (rMAE={rmae(err_or,acts):.4f})"),
                           (err_fx, "darkorange", f"κ={KAPPA_FIXED} (rMAE={rmae(err_fx,acts):.4f})")]:
        valid_idx = np.isfinite(err)
        if valid_idx.sum() < 2:
            continue
        e_v, a_v = err[valid_idx], acts[valid_idx]
        roll = [rmae(e_v[max(0, i-W+1):i+1], a_v[max(0, i-W+1):i+1])
                for i in range(len(e_v))]
        ax.plot(dt[valid_idx], roll, color=col, lw=0.9, label=lbl)

    if REF_CSV.exists():
        ref   = np.loadtxt(REF_CSV, delimiter=",", skiprows=1)
        re, ra = ref[:, 1], ref[:, 2]
        rroll  = [rmae(re[max(0, i-W+1):i+1], ra[max(0, i-W+1):i+1])
                  for i in range(len(re))]
        n = min(len(rroll), len(steps))
        ax.plot(dt[:n], rroll[:n], color="black", lw=1.2, ls="--",
                label=f"05_best ({REF_RMAE})", zorder=5)

    ax.set_ylabel("rMAE (rolling W=50)")
    ax.legend(fontsize=8, ncol=2)
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
    print(f"Oracle: для каждого шага ищем T_frac_opt → κ_oracle = -log(T_frac_opt/T_big)/(H_raw*log(R))")
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

    print("Oracle walk-forward ...", flush=True)
    records = oracle_walk_forward(
        lp_big, confirm_big, dir_big, confirm_bars_big, n_big,
        lh, ll, pool_cache
    )
    print(f"  шагов: {len(records)}")
    print()

    # ── статистика κ ─────────────────────────────────────────────────────────
    kappas   = np.array([r["kappa_oracle"] for r in records])
    kappa_ok = kappas[np.isfinite(kappas)]
    err_or   = np.array([r["err_oracle"] for r in records])
    err_fx   = np.array([r["err_fixed"]  for r in records])
    acts     = np.array([r["actual"]     for r in records])

    print("κ_oracle статистика:")
    print(f"  mean={kappa_ok.mean():.3f}  median={np.median(kappa_ok):.3f}"
          f"  std={kappa_ok.std():.3f}")
    print(f"  p5={np.percentile(kappa_ok,5):.3f}  p25={np.percentile(kappa_ok,25):.3f}"
          f"  p75={np.percentile(kappa_ok,75):.3f}  p95={np.percentile(kappa_ok,95):.3f}")
    print()

    # корреляция κ с D
    ds = np.array([r["D"] for r in records])
    m  = np.isfinite(kappas) & np.isfinite(ds)
    corr = np.corrcoef(ds[m], kappas[m])[0, 1] if m.sum() > 5 else np.nan
    print(f"Корреляция κ_oracle ~ D: r={corr:.3f}")
    print()

    # ── итоговые rMAE ────────────────────────────────────────────────────────
    r_oracle = rmae(err_or, acts)
    r_fixed  = rmae(err_fx[np.isfinite(err_fx)], acts[np.isfinite(err_fx)])
    print(f"{'─'*50}")
    print(f"05_best фиксированный T_frac=3.6%:  rMAE={REF_RMAE}")
    print(f"κ={KAPPA_FIXED} фиксированный:        rMAE={r_fixed:.4f}"
          f"  Δ={(r_fixed-REF_RMAE)/REF_RMAE*100:+.1f}%")
    print(f"Oracle (потолок):               rMAE={r_oracle:.4f}"
          f"  Δ={(r_oracle-REF_RMAE)/REF_RMAE*100:+.1f}%")
    print()

    # ── CSV ──────────────────────────────────────────────────────────────────
    out_csv = RESULTS / f"kappa_oracle_{INTERVAL}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "D", "H_raw", "tf_oracle", "kappa_oracle",
                    "err_oracle", "tf_fixed", "err_fixed", "actual"])
        for r in records:
            w.writerow([r["step"], r["D"], r["H_raw"], r["tf_oracle"],
                        r["kappa_oracle"], r["err_oracle"],
                        r["tf_fixed"], r["err_fixed"], r["actual"]])
    print(f"CSV → {out_csv}")

    fig_path = RESULTS / f"kappa_oracle_{INTERVAL}.png"
    plot(confirm_big, lp_big, records, fig_path)


if __name__ == "__main__":
    main()
