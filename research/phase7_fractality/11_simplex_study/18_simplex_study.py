#!/usr/bin/env python3
"""
18_simplex_study.py — Глубокое исследование Simplex projection (чистая версия)

Откуда стартуем: скр.17 обнаружил Simplex rMAE=0.2727 (-26.6% vs LWR=0.3714).
Здесь: скр.17 использовал build_X, которая добавляла log_price якорной колонкой,
а causal_pool её отбрасывал через [:, 1:]. Мёртвый код. Убрано.

Вложение (честное):
  X[i, k] = log_prices[i-k] - log_prices[i-k-1],  k = 0..n_feats-1
  n_feats=1 → xq = [lp[i] - lp[i-1]]   (1D: амплитуда последнего плеча)
  n_feats=2 → xq = [lp[i]-lp[i-1], lp[i-1]-lp[i-2]]   (2D: два плеча)

Метод Simplex (Sugihara & May 1990):
  w_k = exp(-d_k / d_1),  d_1 = расстояние до ближайшего соседа
  pred_y_rel = Σ w_k·y_rel_k / Σ w_k
  price_hat  = exp(lp_curr + pred_y_rel)

Эксперименты:
  1. 2D-сетка n_feats × K  (T_frac=3.6%)
  2. T_frac свип при лучшей (n_feats, K)
  3. Rolling rMAE топ-5 конфигураций

Контракт каузальности:
  pool: searchsorted(conf_f, confirm_big[step], 'left') — строго до step
  X_big[step]: только lp_big[0..step]
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

HERE    = Path(__file__).parent
DATA    = HERE.parent.parent.parent / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

T_BIG    = 0.04
INTERVAL = "10m"

H           = 1
MIN_HISTORY = 50
REF_LWR     = 0.3714   # LWR p=3 K=75 из скр.16
REF_SX17    = 0.2727   # Simplex скр.17 (p=2 → n_feats=1, K=4)

N_FEATS_GRID = [1, 2, 3, 4, 5, 7, 9]
K_GRID       = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 75]

T_FIXED       = 0.036
T_FRAC_SWEEP  = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.036, 0.040, 0.050]

ROLLING_W = 50


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


def build_X(log_prices, n_feats):
    """
    Только diff-признаки.
    X[i, k] = log_prices[i-k] - log_prices[i-k-1],  k=0..n_feats-1
    Первые n_feats строк — NaN (нет предыстории).
    """
    n = len(log_prices)
    X = np.full((n, n_feats), np.nan)
    for i in range(n_feats, n):
        for k in range(n_feats):
            X[i, k] = log_prices[i - k] - log_prices[i - k - 1]
    return X


def causal_pool(confirm_date, dir_query, conf_f, Xf, dir_f, y_rel_f, n_feats):
    """
    Возвращает (X_pool, y_pool) — матрица уже содержит только diff-признаки,
    никакого strip не нужно.
    """
    ce  = int(np.searchsorted(conf_f, confirm_date, side='left'))
    rng = np.arange(n_feats, min(ce, len(conf_f) - H))
    if len(rng) == 0:
        return None, None
    valid = ~np.any(np.isnan(Xf[rng]), axis=1) & ~np.isnan(y_rel_f[rng])
    idx   = rng[valid]
    if len(idx) < n_feats + 2:
        return None, None
    idx_d = idx[dir_f[idx] == dir_query]
    if len(idx_d) < n_feats + 2:
        idx_d = idx
    return Xf[idx_d], y_rel_f[idx_d]


def predict_simplex(xq, X_pool, y_pool, k):
    """
    Simplex: K ближайших соседей, w_i = exp(-d_i / d_1).
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


def rmae(errs, acts):
    e = np.asarray(errs, float); a = np.asarray(acts, float)
    m = np.isfinite(e) & np.isfinite(a)
    if m.sum() < 2:
        return np.nan
    dz = float(np.mean(np.abs(np.diff(a[m]))))
    return float(np.mean(np.abs(e[m])) / dz) if dz > 1e-12 else np.nan


def wf_simplex(k, lp_big, confirm_big, dir_big,
               conf_f, Xf, dir_f, y_rel_f, n_feats):
    X_big = build_X(lp_big, n_feats)
    errs, acts, step_idx = [], [], []
    for step in range(MIN_HISTORY, len(lp_big) - H):
        if np.any(np.isnan(X_big[step])):
            continue
        xq     = X_big[step]                         # ← без strip, это уже дiffs
        actual = float(np.exp(lp_big[step + H]))
        X_pool, y_pool = causal_pool(
            confirm_big[step], int(dir_big[step]),
            conf_f, Xf, dir_f, y_rel_f, n_feats
        )
        if X_pool is None:
            continue
        y_rel_hat = predict_simplex(xq, X_pool, y_pool, k)
        if not np.isfinite(y_rel_hat):
            continue
        errs.append(float(np.exp(lp_big[step] + y_rel_hat)) - actual)
        acts.append(actual)
        step_idx.append(step)
    return np.array(errs), np.array(acts), np.array(step_idx)


# ── визуализация ──────────────────────────────────────────────────────────────
def plot_heatmap(rmae_grid, n_feats_grid, k_grid, best_nf, best_k, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    ax = axes[0]
    vmin = np.nanmin(rmae_grid)
    vmax = min(np.nanmax(rmae_grid), REF_LWR + 0.05)
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color='#e0e0e0')
    im = ax.imshow(rmae_grid, aspect='auto', cmap=cmap,
                   vmin=vmin, vmax=vmax, origin='upper')
    plt.colorbar(im, ax=ax, label="rMAE")
    ax.set_xticks(range(len(k_grid)))
    ax.set_xticklabels([str(k) for k in k_grid], fontsize=8)
    ax.set_yticks(range(len(n_feats_grid)))
    ax.set_yticklabels([str(nf) for nf in n_feats_grid])
    ax.set_xlabel("K (число соседей)")
    ax.set_ylabel("n_feats (число diff-признаков)")
    ax.set_title("rMAE(n_feats, K)\nn_feats=1: xq=[lp[i]−lp[i−1]]")
    for i in range(len(n_feats_grid)):
        for j in range(len(k_grid)):
            r = rmae_grid[i, j]
            if np.isfinite(r):
                star = "★" if (n_feats_grid[i] == best_nf and k_grid[j] == best_k) else ""
                fc   = "white" if r < vmin + (vmax - vmin) * 0.35 else "black"
                ax.text(j, i, f"{r:.3f}{star}", ha='center', va='center',
                        fontsize=6.5, color=fc,
                        fontweight='bold' if star else 'normal')

    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(n_feats_grid)))
    for i, (nf, col) in enumerate(zip(n_feats_grid, colors)):
        ks, rs = [], []
        for j, kv in enumerate(k_grid):
            r = rmae_grid[i, j]
            if np.isfinite(r):
                ks.append(kv); rs.append(r)
        if ks:
            ax.plot(ks, rs, 'o-', color=col, lw=1.5, ms=5,
                    label=f"n_feats={nf}")
    ax.axhline(REF_LWR,  color='red',   lw=1.2, ls='--',
               label=f"LWR baseline ({REF_LWR})")
    ax.axhline(REF_SX17, color='green', lw=1.0, ls=':',
               label=f"Sx скр.17 ({REF_SX17})")
    ax.set_xscale('log')
    ax.set_xlabel("K (log scale)")
    ax.set_ylabel("rMAE")
    ax.set_title("K-профили при разных n_feats")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Simplex: 2D-сетка n_feats × K   (T_frac=3.6%, SBER 10m)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


def plot_tfrac(tf_vals, rmae_vals, best_tf, out_path, best_nf, best_k):
    fig, ax = plt.subplots(figsize=(10, 5))
    r_min = min(rmae_vals)
    colors = ["steelblue" if r <= r_min * 1.005 else
              "limegreen"  if r < REF_SX17 else "lightcoral"
              for r in rmae_vals]
    bars = ax.bar([f"{t*100:.1f}%" for t in tf_vals], rmae_vals,
                  color=colors, alpha=0.85)
    ax.axhline(REF_LWR,  color='red',   lw=1.2, ls='--',
               label=f"LWR baseline ({REF_LWR})")
    ax.axhline(REF_SX17, color='green', lw=1.0, ls=':',
               label=f"Sx скр.17 ({REF_SX17})")
    for bar, r in zip(bars, rmae_vals):
        d = (r - r_min) / r_min * 100
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.002,
                f"{r:.4f}\n({d:+.1f}%)", ha='center', va='bottom', fontsize=8)
    ax.set_xlabel("T_frac")
    ax.set_ylabel("rMAE")
    ax.set_title(f"T_frac свип для Simplex n_feats={best_nf}, K={best_k}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


def plot_rolling(confirm_big, configs, out_path):
    parse  = lambda s: datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    all_dt = np.array([parse(s) for s in confirm_big])

    fig, axes = plt.subplots(2, 1, figsize=(15, 9),
                             gridspec_kw={"height_ratios": [1, 2]})

    ax = axes[0]
    names = [c[0] for c in configs]
    vals  = [rmae(c[1], c[2]) for c in configs]
    clr   = ["steelblue" if v < REF_LWR else "lightcoral" for v in vals]
    ax.barh(names, vals, color=clr, alpha=0.85)
    ax.axvline(REF_LWR,  color='red',   lw=1.2, ls='--', label=f"LWR ({REF_LWR})")
    ax.axvline(REF_SX17, color='green', lw=1.0, ls=':', label=f"Sx скр.17 ({REF_SX17})")
    for i, v in enumerate(vals):
        if np.isfinite(v):
            d = (v - REF_LWR) / REF_LWR * 100
            ax.text(v + 0.001, i, f"{v:.4f} ({d:+.1f}%)", va='center', fontsize=8)
    ax.set_xlabel("rMAE")
    ax.legend(fontsize=8)
    ax.set_title("Simplex: лучшие конфигурации")

    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    for (label, errs, acts, sidx), col in zip(configs, colors):
        if len(errs) < ROLLING_W:
            continue
        dt_sl = all_dt[sidx]
        roll  = [rmae(errs[max(0, i - ROLLING_W + 1):i + 1],
                      acts[max(0, i - ROLLING_W + 1):i + 1])
                 for i in range(len(errs))]
        r = rmae(errs, acts)
        ax.plot(dt_sl, roll, lw=1.0, color=col, label=f"{label} ({r:.4f})")

    ax.axhline(REF_LWR,  color='red',   lw=1.2, ls='--', alpha=0.7,
               label=f"LWR ({REF_LWR})", zorder=5)
    ax.axhline(REF_SX17, color='green', lw=1.0, ls=':', alpha=0.7,
               label=f"Sx скр.17 ({REF_SX17})")
    ax.set_ylabel(f"rMAE (rolling W={ROLLING_W})")
    ax.legend(fontsize=7, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"SBER {INTERVAL}  T_BIG={T_BIG*100:.1f}%\n")

    h, l, d = load_candles(INTERVAL)
    lp_big, confirm_big, dir_big, _ = find_pivots_log(h, l, d, T_BIG)
    print(f"T_BIG: {len(lp_big)} пивотов")

    lp_f, conf_f, dir_f, _ = find_pivots_log(h, l, d, T_FIXED)
    y_rel_f = np.array([lp_f[j + H] - lp_f[j] if j + H < len(lp_f) else np.nan
                        for j in range(len(lp_f))])
    print(f"T_frac={T_FIXED*100:.1f}%: {len(lp_f)} пивотов\n")

    print(f"Предвычисляем X для n_feats={N_FEATS_GRID} ...", flush=True)
    pool_by_nf = {}
    for nf in N_FEATS_GRID:
        pool_by_nf[nf] = (conf_f, build_X(lp_f, nf), dir_f, y_rel_f)

    # ═══════════════════════════════════════════════════════════════════════════
    # ЧАСТЬ 1: 2D-сетка n_feats × K
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print("ЧАСТЬ 1: 2D-сетка n_feats × K  (T_frac=3.6%)")
    print("=" * 65)
    header = "  n_feats\\K  " + "  ".join(f"{k:>6}" for k in K_GRID)
    print(header)

    rmae_grid  = np.full((len(N_FEATS_GRID), len(K_GRID)), np.nan)
    errs_store = {}

    for i, nf in enumerate(N_FEATS_GRID):
        cf, Xf, df, yr = pool_by_nf[nf]
        row = f"  n_feats={nf:2d}  "
        for j, k in enumerate(K_GRID):
            e, a, sidx = wf_simplex(k, lp_big, confirm_big, dir_big,
                                    cf, Xf, df, yr, nf)
            r = rmae(e, a)
            rmae_grid[i, j] = r
            errs_store[(nf, k)] = (e, a, sidx)
            row += f"  {r:.4f}"
        print(row, flush=True)

    flat = sorted(
        [(rmae_grid[i, j], N_FEATS_GRID[i], K_GRID[j])
         for i in range(len(N_FEATS_GRID))
         for j in range(len(K_GRID))
         if np.isfinite(rmae_grid[i, j])]
    )
    best_r, best_nf, best_k = flat[0]

    print(f"\nТоп-10:")
    print(f"  {'n_feats':>7}  {'K':>4}  {'rMAE':>8}  {'Δ vs LWR':>10}  {'Δ vs Sx17':>10}")
    for r, nf, k in flat[:10]:
        dl = (r - REF_LWR)  / REF_LWR  * 100
        ds = (r - REF_SX17) / REF_SX17 * 100
        mark = " ◄◄◄" if (nf == best_nf and k == best_k) else ""
        print(f"  {nf:>7}  {k:>4}  {r:>8.4f}  {dl:>+9.1f}%  {ds:>+9.1f}%{mark}")

    out1 = RESULTS / f"grid_nfeats_K_{INTERVAL}.csv"
    with open(out1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_feats", "K", "rmae", "delta_lwr_pct", "delta_sx17_pct"])
        for r, nf, k in flat:
            w.writerow([nf, k, f"{r:.4f}",
                        f"{(r-REF_LWR)/REF_LWR*100:.2f}",
                        f"{(r-REF_SX17)/REF_SX17*100:.2f}"])
    print(f"\nCSV → {out1}")

    plot_heatmap(rmae_grid, N_FEATS_GRID, K_GRID, best_nf, best_k,
                 RESULTS / f"heatmap_nfeats_K_{INTERVAL}.png")

    # ═══════════════════════════════════════════════════════════════════════════
    # ЧАСТЬ 2: T_frac свип
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"ЧАСТЬ 2: T_frac свип  (n_feats={best_nf}, K={best_k})")
    print(f"{'='*65}")

    tf_rmae_list  = []
    tf_errs_list  = []
    for tf in T_FRAC_SWEEP:
        lp_tf, cf_tf, df_tf, _ = find_pivots_log(h, l, d, tf)
        yr_tf  = np.array([lp_tf[j + H] - lp_tf[j] if j + H < len(lp_tf) else np.nan
                           for j in range(len(lp_tf))])
        Xf_tf  = build_X(lp_tf, best_nf)
        e, a, sidx = wf_simplex(best_k, lp_big, confirm_big, dir_big,
                                 cf_tf, Xf_tf, df_tf, yr_tf, best_nf)
        r = rmae(e, a)
        tf_rmae_list.append(r)
        tf_errs_list.append((f"T={tf*100:.1f}%", e, a, sidx))
        r_min_so_far = min(x for x in tf_rmae_list if np.isfinite(x))
        mark = " ◄" if r == r_min_so_far else ""
        print(f"  T={tf*100:.1f}%  pool={len(lp_tf):6d}"
              f"  rMAE={r:.4f}  Δ={(r-best_r)/best_r*100:+.1f}%{mark}")

    best_tf_i = int(np.nanargmin(tf_rmae_list))
    best_tf   = T_FRAC_SWEEP[best_tf_i]
    best_tf_r = tf_rmae_list[best_tf_i]
    print(f"\nЛучший T_frac={best_tf*100:.1f}%  rMAE={best_tf_r:.4f}")

    out2 = RESULTS / f"tfrac_sweep_{INTERVAL}.csv"
    with open(out2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_frac", "n_pool", "rmae", "delta_best_pct"])
        for tf, r in zip(T_FRAC_SWEEP, tf_rmae_list):
            lp_tf2, *_ = find_pivots_log(h, l, d, tf)
            w.writerow([f"{tf:.3f}", len(lp_tf2), f"{r:.4f}",
                        f"{(r-best_tf_r)/best_tf_r*100:.2f}"])
    print(f"CSV → {out2}")

    plot_tfrac(T_FRAC_SWEEP, tf_rmae_list, best_tf,
               RESULTS / f"tfrac_sweep_{INTERVAL}.png", best_nf, best_k)

    # ═══════════════════════════════════════════════════════════════════════════
    # ЧАСТЬ 3: Rolling rMAE
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("ЧАСТЬ 3: Rolling rMAE — топ-5")
    print(f"{'='*65}")

    rolling_configs = []
    for r, nf, k in flat[:5]:
        label = f"nf={nf} K={k}"
        e, a, sidx = errs_store[(nf, k)]
        rolling_configs.append((label, e, a, sidx))
        print(f"  {label}  rMAE={r:.4f}")

    if best_tf != T_FIXED:
        lbl = f"nf={best_nf} K={best_k} T={best_tf*100:.1f}%"
        _, e_tf, a_tf, sidx_tf = tf_errs_list[best_tf_i]
        rolling_configs.append((lbl, e_tf, a_tf, sidx_tf))
        print(f"  {lbl}  rMAE={best_tf_r:.4f}  (лучший T_frac)")

    plot_rolling(confirm_big, rolling_configs,
                 RESULTS / f"rolling_rmae_{INTERVAL}.png")

    # ── итог ──────────────────────────────────────────────────────────────────
    overall_best_r = min(best_r, best_tf_r)
    print(f"\n{'═'*65}")
    print("ИТОГ")
    print(f"{'═'*65}")
    print(f"  LWR baseline:     rMAE={REF_LWR:.4f}")
    print(f"  Simplex скр.17:   rMAE={REF_SX17:.4f}  (n_feats=1, K=4, T=3.6%)")
    print(f"  Simplex 2D best:  rMAE={best_r:.4f}"
          f"  (n_feats={best_nf}, K={best_k}, T=3.6%)"
          f"  Δ скр.17={(best_r-REF_SX17)/REF_SX17*100:+.2f}%")
    print(f"  Simplex T_f best: rMAE={best_tf_r:.4f}"
          f"  (n_feats={best_nf}, K={best_k}, T={best_tf*100:.1f}%)"
          f"  Δ скр.17={(best_tf_r-REF_SX17)/REF_SX17*100:+.2f}%")
    print(f"  Итого vs LWR:     Δ={(overall_best_r-REF_LWR)/REF_LWR*100:+.1f}%")


if __name__ == "__main__":
    main()
