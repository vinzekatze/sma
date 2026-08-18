#!/usr/bin/env python3
"""
11_distance_features.py — Расстояние до соседей как сигнал выбора T_frac

Для каждого шага t и каждого T_frac из сетки вычисляем:
  dist_mean  — среднее расстояние до K ближайших соседей
  dist_min   — расстояние до 1-го соседа
  dist_max   — максимум топ-K (= ξ, bandwidth LWR)
  dist_std   — разброс расстояний топ-K
  dist_cv    — CV = std/mean
  n_pool     — размер каузального пула

Гипотезы:
  H1: T_frac с минимальным dist_mean ≈ tf_oracle (лучший пул = лучший прогноз)
  H2: dist_std / dist_mean при фиксированном T_frac коррелирует с tf_oracle
  H3: n_pool коррелирует с tf_oracle

Referenс: oracle CSV из 09_kappa_oracle.py (tf_oracle, kappa_oracle, err_oracle)
"""
import csv
import json
import sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE    = Path(__file__).parent
DATA    = HERE.parent.parent.parent / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
ORACLE  = HERE.parent / "08_adaptive_fd" / "results" / "kappa_oracle_10m.csv"

T_BIG    = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
INTERVAL = sys.argv[2]        if len(sys.argv) > 2 else "10m"

H           = 1
P           = 3
K           = 75
MIN_HISTORY = 50
T_FRAC_MIN  = 0.005
T_FRAC_MAX  = T_BIG * 0.99
T_FRAC_GRID = np.round(np.arange(0.005, T_BIG, 0.001), 4)


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
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if k != "step" else int(v)
                         for k, v in row.items()})
    return rows


# ── пул ──────────────────────────────────────────────────────────────────────
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


def dist_stats(xq, X_pool):
    """Расстояния от xq до всех точек пула, топ-K."""
    dists = np.linalg.norm(X_pool - xq, axis=1)
    keff  = min(K, len(dists))
    top_k = np.sort(dists)[:keff]
    return dict(
        dist_mean = float(top_k.mean()),
        dist_min  = float(top_k[0]),
        dist_max  = float(top_k[-1]),
        dist_std  = float(top_k.std()),
        dist_cv   = float(top_k.std() / top_k.mean()) if top_k.mean() > 1e-12 else np.nan,
        n_pool    = len(dists),
        n_keff    = keff,
    )


# ── основной проход ───────────────────────────────────────────────────────────
def run(oracle_rows, lp_big, confirm_big, dir_big, confirm_bars_big,
        lh, ll, pool_cache):
    grid  = np.array(sorted(pool_cache.keys()))
    X_big = build_X(lp_big, P)

    # step → tf_oracle
    tf_map = {r["step"]: r["tf_oracle"] for r in oracle_rows}

    records = []
    n_done  = 0

    for rec in oracle_rows:
        step = rec["step"]
        if step < MIN_HISTORY or np.any(np.isnan(X_big[step])):
            continue

        xq = X_big[step][1:]

        # расстояния для каждого T_frac
        per_tf = {}
        for tf in grid:
            lp_f1, confirm_f1, dir_f1, Xf1, y_rel = pool_cache[tf]
            X_pool, _ = causal_pool(
                confirm_big[step], int(dir_big[step]),
                confirm_f1, Xf1, dir_f1, y_rel
            )
            if X_pool is None:
                continue
            per_tf[tf] = dist_stats(xq, X_pool)

        if not per_tf:
            continue

        tf_list  = np.array(sorted(per_tf))
        dm_arr   = np.array([per_tf[tf]["dist_mean"] for tf in tf_list])
        ds_arr   = np.array([per_tf[tf]["dist_std"]  for tf in tf_list])
        dc_arr   = np.array([per_tf[tf]["dist_cv"]   for tf in tf_list])
        dn_arr   = np.array([per_tf[tf]["dist_min"]  for tf in tf_list])
        np_arr   = np.array([per_tf[tf]["n_pool"]    for tf in tf_list])

        # T_frac с минимальным средним расстоянием
        tf_min_dm  = float(tf_list[np.argmin(dm_arr)])
        tf_min_ds  = float(tf_list[np.argmin(ds_arr)])
        tf_min_cv  = float(tf_list[np.argmin(dc_arr)])
        tf_max_np  = float(tf_list[np.argmax(np_arr)])

        # статистики при фиксированном T_frac=3.6% (≈ κ=0.30)
        tf_fixed_key = float(grid[np.argmin(np.abs(grid - 0.036))])
        if tf_fixed_key in per_tf:
            fs = per_tf[tf_fixed_key]
        else:
            fs = {k: np.nan for k in ["dist_mean","dist_min","dist_max",
                                       "dist_std","dist_cv","n_pool"]}

        # slope dist_mean vs log(T_frac) — крутизна изменения расстояний
        valid = np.isfinite(dm_arr)
        if valid.sum() > 5:
            slope_dm, _ = np.polyfit(np.log(tf_list[valid]), dm_arr[valid], 1)
        else:
            slope_dm = np.nan

        records.append(dict(
            step        = step,
            tf_oracle   = rec["tf_oracle"],
            kappa_ora   = rec["kappa_oracle"],
            err_oracle  = rec["err_oracle"],
            # T_frac отбора по расстоянию
            tf_min_dm   = tf_min_dm,
            tf_min_ds   = tf_min_ds,
            tf_min_cv   = tf_min_cv,
            tf_max_np   = tf_max_np,
            # при fixed T_frac=3.6%
            fix_dm      = fs["dist_mean"],
            fix_ds      = fs["dist_std"],
            fix_cv      = fs["dist_cv"],
            fix_np      = fs["n_pool"],
            fix_min     = fs["dist_min"],
            fix_max     = fs["dist_max"],
            # slope
            slope_dm    = slope_dm,
            # диапазон dist_mean по сетке
            dm_range    = float(dm_arr.max() - dm_arr.min()) if valid.sum() > 1 else np.nan,
            dm_at_oracle = per_tf.get(rec["tf_oracle"], {}).get("dist_mean", np.nan),
        ))

        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{len(oracle_rows)} ...", flush=True)

    return records


def rmae(errs, acts):
    acts = np.asarray(acts, dtype=float)
    pers = np.abs(acts[2:] - acts[:-2]) if len(acts) >= 3 else np.abs(np.diff(acts))
    dz   = float(np.mean(pers)) if len(pers) > 0 else 1.0
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


# ── корреляции ────────────────────────────────────────────────────────────────
FEAT_KEYS = [
    ("tf_min_dm",  "T_frac при min dist_mean"),
    ("tf_min_ds",  "T_frac при min dist_std"),
    ("tf_min_cv",  "T_frac при min dist_cv"),
    ("tf_max_np",  "T_frac при max n_pool"),
    ("fix_dm",     "dist_mean @ T=3.6%"),
    ("fix_ds",     "dist_std  @ T=3.6%"),
    ("fix_cv",     "dist_cv   @ T=3.6%"),
    ("fix_np",     "n_pool    @ T=3.6%"),
    ("fix_min",    "dist_min  @ T=3.6%"),
    ("fix_max",    "dist_max  @ T=3.6%"),
    ("slope_dm",   "slope dist_mean vs log(T_frac)"),
    ("dm_range",   "range dist_mean по сетке"),
    ("dm_at_oracle","dist_mean @ tf_oracle"),
]


def print_corr(records, target_key, target_label):
    target = np.array([r[target_key] for r in records])
    print(f"\n=== Корреляция с {target_label} ===")
    print(f"{'Признак':<32}  {'Pearson r':>10}  {'Spearman ρ':>10}  {'n':>6}")
    results = []
    for key, label in FEAT_KEYS:
        vals = np.array([r[key] for r in records])
        mask = np.isfinite(vals) & np.isfinite(target)
        if mask.sum() < 20:
            print(f"  {label:<30}  {'—':>10}  {'—':>10}  {mask.sum():>6}")
            results.append((label, np.nan, np.nan))
            continue
        rp, _ = stats.pearsonr(vals[mask], target[mask])
        rs, _ = stats.spearmanr(vals[mask], target[mask])
        flag  = " ◄" if abs(rs) > 0.15 else ""
        print(f"  {label:<30}  {rp:>+10.3f}  {rs:>+10.3f}  {mask.sum():>6}{flag}")
        results.append((label, rp, rs))
    return results


# ── визуализация ──────────────────────────────────────────────────────────────
def plot_h1(records, out_path):
    """H1: tf_min_dm vs tf_oracle."""
    tf_or = np.array([r["tf_oracle"] for r in records])
    tf_md = np.array([r["tf_min_dm"] for r in records])
    mask  = np.isfinite(tf_or) & np.isfinite(tf_md)
    rs, _ = stats.spearmanr(tf_md[mask], tf_or[mask])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # scatter
    ax = axes[0]
    ax.scatter(tf_md[mask] * 100, tf_or[mask] * 100, s=5, alpha=0.3,
               color="steelblue")
    ax.plot([T_FRAC_MIN*100, T_BIG*100], [T_FRAC_MIN*100, T_BIG*100],
            color="gray", lw=0.8, ls="--", label="ideal")
    ax.set_xlabel("T_frac при min dist_mean (%)")
    ax.set_ylabel("T_frac oracle (%)")
    ax.set_title(f"H1: T_frac(min_dist) vs T_frac(oracle)  ρ={rs:.3f}")
    ax.legend(fontsize=8)

    # распределение совпадений (в пределах 1 шага сетки = 0.1%)
    diff = np.abs(tf_md[mask] - tf_or[mask])
    pct_close = (diff < 0.002).mean() * 100
    ax2 = axes[1]
    ax2.hist(diff * 100, bins=30, color="steelblue", alpha=0.75, edgecolor="white")
    ax2.set_xlabel("|tf_min_dist − tf_oracle| (%)")
    ax2.set_ylabel("событий")
    ax2.set_title(f"Расхождение tf_min_dist vs tf_oracle\n"
                  f"в пределах 0.2%: {pct_close:.1f}%  |  median={np.median(diff)*100:.2f}%")

    plt.suptitle(f"SBER {INTERVAL} | T_big={T_BIG*100:.1f}% | n={mask.sum()}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


def plot_dist_profile(records, out_path, n_show=6):
    """Профиль dist_mean по T_frac для нескольких случайных шагов."""
    np.random.seed(42)
    idx = np.random.choice(len(records), size=min(n_show, len(records)), replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, si in enumerate(idx):
        r = records[si]
        ax = axes[i]
        # Нужно перестроить профиль из CSV — храним tf_min_dm и dm_range
        # В этом скрипте не храним полный профиль, но можем отметить ключевые точки
        ax.set_title(f"step={r['step']}\n"
                     f"tf_oracle={r['tf_oracle']*100:.1f}%  "
                     f"tf_min_dm={r['tf_min_dm']*100:.1f}%\n"
                     f"dm_range={r['dm_range']:.4f}  "
                     f"fix_dm={r['fix_dm']:.4f}", fontsize=8)
        # Показываем основные точки
        ax.bar(["oracle", "min_dm", "fixed\n3.6%"],
               [r.get("dm_at_oracle", 0) or 0,
                r["fix_dm"] - r.get("dm_range", 0) / 2,
                r["fix_dm"]],
               color=["tomato", "steelblue", "darkorange"], alpha=0.75)
        ax.set_ylabel("dist_mean")

    plt.suptitle("Профиль dist_mean: oracle vs min_dist vs fixed", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


def plot_fix_dm_vs_oracle(records, out_path):
    """dist_mean и dist_std при fixed T_frac=3.6% vs tf_oracle."""
    tf_or = np.array([r["tf_oracle"] for r in records])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, key, label in [
        (axes[0], "fix_dm",  "dist_mean @ 3.6%"),
        (axes[1], "fix_ds",  "dist_std  @ 3.6%"),
        (axes[2], "fix_cv",  "dist_cv   @ 3.6%"),
    ]:
        vals = np.array([r[key] for r in records])
        mask = np.isfinite(vals) & np.isfinite(tf_or)
        rs, _ = stats.spearmanr(vals[mask], tf_or[mask])
        ax.scatter(vals[mask], tf_or[mask] * 100, s=5, alpha=0.3, color="steelblue")
        if mask.sum() > 5:
            z  = np.polyfit(vals[mask], tf_or[mask] * 100, 1)
            xr = np.linspace(vals[mask].min(), vals[mask].max(), 50)
            ax.plot(xr, np.polyval(z, xr), color="tomato", lw=1.5)
        ax.set_xlabel(label)
        ax.set_ylabel("tf_oracle (%)")
        ax.set_title(f"ρ={rs:.3f}")

    plt.suptitle(f"Расстояния при fixed T_frac=3.6% vs tf_oracle\n"
                 f"SBER {INTERVAL} T_big={T_BIG*100:.1f}%", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"T_BIG={T_BIG*100:.1f}%  INTERVAL={INTERVAL}")
    print()

    h, l, d = load_tf(INTERVAL)
    lh, ll  = np.log(h), np.log(l)

    lp_big, confirm_big, dir_big, confirm_bars_big = find_pivots_log(h, l, d, T_BIG)
    print(f"T_BIG: {len(lp_big)} пивотов")

    oracle_rows = load_oracle(ORACLE)
    print(f"Oracle записей: {len(oracle_rows)}")
    print()

    print(f"Предвычисляем {len(T_FRAC_GRID)} зигзагов ...", flush=True)
    pool_cache = {}
    for tf in T_FRAC_GRID:
        lp_f, conf_f, dir_f, _ = find_pivots_log(h, l, d, tf)
        lp_f_full = lp_f
        Xf1   = build_X(lp_f, P)
        y_rel = np.array([lp_f[j+H] - lp_f[j] if j+H < len(lp_f) else np.nan
                          for j in range(len(lp_f))])
        pool_cache[tf] = (lp_f_full, conf_f, dir_f, Xf1, y_rel)

    print(f"Считаем расстояния (735 шагов × {len(T_FRAC_GRID)} T_frac) ...", flush=True)
    records = run(oracle_rows, lp_big, confirm_big, dir_big, confirm_bars_big,
                  lh, ll, pool_cache)
    print(f"Готово: {len(records)} записей")
    print()

    # ── совпадение H1 ─────────────────────────────────────────────────────────
    tf_or = np.array([r["tf_oracle"] for r in records])
    tf_md = np.array([r["tf_min_dm"] for r in records])
    mask  = np.isfinite(tf_or) & np.isfinite(tf_md)
    diff  = np.abs(tf_md[mask] - tf_or[mask])
    rs_h1, _ = stats.spearmanr(tf_md[mask], tf_or[mask])
    print(f"H1 (min dist_mean → tf_oracle):")
    print(f"  Spearman ρ = {rs_h1:.3f}")
    print(f"  Совпадение ±0.2%: {(diff<0.002).mean()*100:.1f}%")
    print(f"  Совпадение ±0.5%: {(diff<0.005).mean()*100:.1f}%")
    print(f"  Медиана расхождения: {np.median(diff)*100:.2f}%")
    print()

    # ── корреляции ────────────────────────────────────────────────────────────
    print_corr(records, "tf_oracle", "tf_oracle")
    print_corr(records, "kappa_ora", "kappa_oracle")

    # ── CSV ──────────────────────────────────────────────────────────────────
    out_csv = RESULTS / f"dist_features_{INTERVAL}.csv"
    keys_out = ["step","tf_oracle","kappa_ora","err_oracle",
                "tf_min_dm","tf_min_ds","tf_min_cv","tf_max_np",
                "fix_dm","fix_ds","fix_cv","fix_np","fix_min","fix_max",
                "slope_dm","dm_range","dm_at_oracle"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys_out)
        for r in records:
            w.writerow([r[k] for k in keys_out])
    print(f"\nCSV → {out_csv}")

    plot_h1(records, RESULTS / f"h1_min_dist_{INTERVAL}.png")
    plot_fix_dm_vs_oracle(records, RESULTS / f"fix_dist_vs_oracle_{INTERVAL}.png")


if __name__ == "__main__":
    main()
