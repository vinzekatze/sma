#!/usr/bin/env python3
"""
10_tfrac_features.py — Что предсказывает оптимальный T_frac?

Загружаем oracle CSV (09_kappa_oracle.py) → для каждого шага вычисляем
каузальные признаки из двух источников:

  A) T_big зигзаг:
       amp_curr   — амплитуда текущего события |lp[step] - lp[step-1]|
       amp_prev   — амплитуда предыдущего
       amp_ma5/10 — средние за 5/10 событий
       amp_z      — z-score амплитуды (относительно ma10)
       amp_ratio  — amp_curr / amp_ma5
       dur_curr   — длина события в барах
       dur_ma5    — средняя длина за 5 событий
       dur_ratio  — dur_curr / dur_ma5
       dir_curr   — направление (+1 вверх / -1 вниз)

  B) Сырой лог-ряд (log(high), log(low)):
       vol_10/30/100 — std log(high/low) за N баров
       vol_r10_30    — vol_10 / vol_30  (краткосрочная vs долгосрочная)
       atr_10/30     — mean log(high/low) за N баров (упрощённый ATR)
       rng_10/30     — max(lh)-min(ll) за N баров (диапазон)

  C) Автокорреляция tf_oracle:
       tf_lag1 — tf_oracle на шаге step-1 (в walk-forward: rolling best)

Метрики: Pearson r + Spearman ρ vs tf_oracle и kappa_oracle.
График: матрица scatter + correlation heatmap.
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
ORACLE  = (HERE.parent / "08_adaptive_fd" / "results"
           / "kappa_oracle_10m.csv")

T_BIG    = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
INTERVAL = sys.argv[2]        if len(sys.argv) > 2 else "10m"


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


def load_oracle(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if k != "step" else int(v)
                         for k, v in row.items()})
    return rows


# ── признаки ─────────────────────────────────────────────────────────────────
def rolling_std(arr, start, end):
    sl = arr[max(0, start):end + 1]
    return float(np.std(sl)) if len(sl) > 1 else np.nan


def rolling_mean(arr, start, end):
    sl = arr[max(0, start):end + 1]
    return float(np.mean(sl)) if len(sl) > 0 else np.nan


def rolling_range(lh, ll, start, end):
    s, e = max(0, start), end + 1
    return float(lh[s:e].max() - ll[s:e].min()) if e > s else np.nan


def build_features(oracle_rows, lp_big, dirs_big, confirm_bars_big,
                   lh, ll, tf_oracle_arr):
    """Вычисляем признаки для каждой oracle-записи."""
    step_to_idx = {r["step"]: i for i, r in enumerate(oracle_rows)}
    n_big = len(lp_big)

    # амплитуды и длины T_big событий
    amps  = np.abs(np.diff(lp_big))          # amps[i] = |lp[i+1]-lp[i]|, i=0..n-2
    durs  = np.diff(confirm_bars_big.astype(float))  # дл. в барах

    # log(high/low) как прокси волатильности одного бара
    bar_vols = lh - ll   # log(H/L) ≥ 0

    rows_out = []
    for rec in oracle_rows:
        step = rec["step"]
        if step < 10:
            continue
        bar = int(confirm_bars_big[step])

        # ── T_big признаки ────────────────────────────────────────────────
        # amps[step-1] = |lp[step]-lp[step-1]| (текущее событие)
        # amps[step-2] = предыдущее
        if step - 1 < 0 or step - 1 >= len(amps):
            continue

        amp_curr  = float(amps[step - 1])
        amp_prev  = float(amps[step - 2]) if step >= 2 else np.nan

        w5 = amps[max(0, step - 5):step]
        w10 = amps[max(0, step - 10):step]
        amp_ma5  = float(w5.mean())  if len(w5) > 0 else np.nan
        amp_ma10 = float(w10.mean()) if len(w10) > 0 else np.nan
        amp_std10 = float(w10.std()) if len(w10) > 1 else np.nan
        amp_z    = (amp_curr - amp_ma10) / amp_std10 if amp_std10 and amp_std10 > 0 else np.nan
        amp_ratio = amp_curr / amp_ma5 if amp_ma5 and amp_ma5 > 0 else np.nan

        if step - 1 < len(durs):
            dur_curr = float(durs[step - 1])
        else:
            dur_curr = np.nan
        d5 = durs[max(0, step - 5):step]
        dur_ma5  = float(d5.mean()) if len(d5) > 0 else np.nan
        dur_ratio = dur_curr / dur_ma5 if dur_ma5 and dur_ma5 > 0 else np.nan

        dir_curr = float(dirs_big[step])

        # ── сырой ряд признаки ────────────────────────────────────────────
        vol_10  = rolling_std(bar_vols, bar - 10,  bar)
        vol_30  = rolling_std(bar_vols, bar - 30,  bar)
        vol_100 = rolling_std(bar_vols, bar - 100, bar)
        vol_r   = vol_10 / vol_30 if vol_30 and vol_30 > 0 else np.nan
        atr_10  = rolling_mean(bar_vols, bar - 10, bar)
        atr_30  = rolling_mean(bar_vols, bar - 30, bar)
        rng_10  = rolling_range(lh, ll, bar - 10, bar)
        rng_30  = rolling_range(lh, ll, bar - 30, bar)

        # ── автокорреляция tf_oracle ─────────────────────────────────────
        prev_step = step - 1
        tf_lag1 = tf_oracle_arr.get(prev_step, np.nan)

        rows_out.append(dict(
            step      = step,
            tf_oracle = rec["tf_oracle"],
            kappa     = rec["kappa_oracle"],
            # T_big
            amp_curr  = amp_curr,
            amp_prev  = amp_prev,
            amp_ma5   = amp_ma5,
            amp_ma10  = amp_ma10,
            amp_z     = amp_z,
            amp_ratio = amp_ratio,
            dur_curr  = dur_curr,
            dur_ma5   = dur_ma5,
            dur_ratio = dur_ratio,
            dir_curr  = dir_curr,
            # raw
            vol_10    = vol_10,
            vol_30    = vol_30,
            vol_100   = vol_100,
            vol_r10_30 = vol_r,
            atr_10    = atr_10,
            atr_30    = atr_30,
            rng_10    = rng_10,
            rng_30    = rng_30,
            # lag
            tf_lag1   = tf_lag1,
        ))

    return rows_out


FEATURE_NAMES = [
    "amp_curr", "amp_prev", "amp_ma5", "amp_ma10",
    "amp_z", "amp_ratio",
    "dur_curr", "dur_ma5", "dur_ratio",
    "dir_curr",
    "vol_10", "vol_30", "vol_100", "vol_r10_30",
    "atr_10", "atr_30", "rng_10", "rng_30",
    "tf_lag1",
]


# ── корреляции ────────────────────────────────────────────────────────────────
def correlations(rows, target_key):
    target = np.array([r[target_key] for r in rows])
    results = []
    for feat in FEATURE_NAMES:
        vals = np.array([r[feat] for r in rows])
        mask = np.isfinite(vals) & np.isfinite(target)
        if mask.sum() < 20:
            results.append((feat, np.nan, np.nan, 0))
            continue
        r_p, _ = stats.pearsonr(vals[mask], target[mask])
        r_s, _ = stats.spearmanr(vals[mask], target[mask])
        results.append((feat, r_p, r_s, mask.sum()))
    results.sort(key=lambda x: abs(x[2]) if np.isfinite(x[2]) else 0, reverse=True)
    return results


# ── визуализация ──────────────────────────────────────────────────────────────
def plot_correlations(corr_tf, corr_kappa, rows, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    def bar_plot(ax, corrs, title):
        names = [c[0] for c in corrs]
        spear = [c[2] if np.isfinite(c[2]) else 0 for c in corrs]
        pears = [c[1] if np.isfinite(c[1]) else 0 for c in corrs]
        y     = np.arange(len(names))
        ax.barh(y - 0.2, spear, 0.35, label="Spearman ρ", color="steelblue", alpha=0.8)
        ax.barh(y + 0.2, pears, 0.35, label="Pearson r",  color="darkorange", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, color="black", lw=0.7)
        for v in [-0.1, 0.1]:
            ax.axvline(v, color="gray", lw=0.5, ls="--")
        ax.set_xlabel("Корреляция")
        ax.set_title(title)
        ax.legend(fontsize=8)

    bar_plot(axes[0], corr_tf,    "Признаки → tf_oracle")
    bar_plot(axes[1], corr_kappa, "Признаки → kappa_oracle")

    plt.suptitle(
        f"SBER {INTERVAL} | T_big={T_BIG*100:.1f}% | n={len(rows)}\n"
        "Корреляция признаков с оптимальным T_frac/κ",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


def plot_top_scatter(rows, corr_tf, out_path, top_n=6):
    """Scatter plot топ-N признаков vs tf_oracle."""
    top = [c for c in corr_tf if np.isfinite(c[2])][:top_n]
    tf  = np.array([r["tf_oracle"] for r in rows])

    cols = 3
    rows_n = (top_n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(15, rows_n * 4))
    axes = axes.flatten()

    for i, (feat, r_p, r_s, n) in enumerate(top):
        vals = np.array([r[feat] for r in rows])
        mask = np.isfinite(vals) & np.isfinite(tf)
        ax = axes[i]
        ax.scatter(vals[mask], tf[mask] * 100, s=5, alpha=0.3, color="steelblue")
        # тренд
        if mask.sum() > 5:
            z   = np.polyfit(vals[mask], tf[mask] * 100, 1)
            xr  = np.linspace(np.nanmin(vals[mask]), np.nanmax(vals[mask]), 50)
            ax.plot(xr, np.polyval(z, xr), color="tomato", lw=1.5)
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("T_frac opt (%)", fontsize=9)
        ax.set_title(f"{feat}  |  ρ={r_s:.3f}  r={r_p:.3f}  n={n}", fontsize=9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Топ признаков vs tf_oracle", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


def plot_autocorr(rows, out_path, max_lag=20):
    """Автокорреляция tf_oracle по лагам зигзаг-событий."""
    tf = np.array([r["tf_oracle"] for r in rows])
    tf = tf[np.isfinite(tf)]
    tf_c = tf - tf.mean()
    c0   = float(tf_c @ tf_c)
    lags, acfs = [], []
    for lag in range(1, min(max_lag + 1, len(tf_c))):
        acf = float(tf_c[lag:] @ tf_c[:-lag]) / c0
        lags.append(lag)
        acfs.append(acf)

    ci = 1.96 / np.sqrt(len(tf_c))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lags, acfs, color="steelblue", alpha=0.7, width=0.6)
    ax.axhline( ci, color="gray", lw=0.8, ls="--")
    ax.axhline(-ci, color="gray", lw=0.8, ls="--")
    ax.axhline(0,   color="black", lw=0.7)
    ax.set_xlabel("Лаг (T_big события)")
    ax.set_ylabel("ACF")
    ax.set_title(
        f"Автокорреляция tf_oracle по событиям T_big\n"
        f"n={len(tf_c)}  (серые линии = 95% CI = ±{ci:.3f})"
    )
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

    lp_big, confirm_dates, dirs_big, confirm_bars_big = find_pivots_log(h, l, d, T_BIG)
    print(f"T_BIG: {len(lp_big)} пивотов")

    oracle_rows = load_oracle(ORACLE)
    print(f"Oracle записей: {len(oracle_rows)}")
    print()

    # словарь step → tf_oracle (для лаговых признаков)
    tf_oracle_map = {r["step"]: r["tf_oracle"] for r in oracle_rows}

    print("Вычисляем признаки ...", flush=True)
    feat_rows = build_features(
        oracle_rows, lp_big, dirs_big, confirm_bars_big,
        lh, ll, tf_oracle_map
    )
    print(f"  записей с признаками: {len(feat_rows)}")
    print()

    # ── корреляции ────────────────────────────────────────────────────────────
    corr_tf    = correlations(feat_rows, "tf_oracle")
    corr_kappa = correlations(feat_rows, "kappa")

    print("=== Корреляция с tf_oracle (по |Spearman ρ|) ===")
    print(f"{'Признак':<16}  {'Pearson r':>10}  {'Spearman ρ':>10}  {'n':>6}")
    for feat, rp, rs, n in corr_tf:
        flag = " ◄" if abs(rs) > 0.10 else ""
        print(f"  {feat:<14}  {rp:>+10.3f}  {rs:>+10.3f}  {n:>6}{flag}")
    print()

    print("=== Корреляция с kappa_oracle (по |Spearman ρ|) ===")
    print(f"{'Признак':<16}  {'Pearson r':>10}  {'Spearman ρ':>10}  {'n':>6}")
    for feat, rp, rs, n in corr_kappa:
        flag = " ◄" if abs(rs) > 0.10 else ""
        print(f"  {feat:<14}  {rp:>+10.3f}  {rs:>+10.3f}  {n:>6}{flag}")
    print()

    # ── автокорреляция tf_oracle ──────────────────────────────────────────────
    tf_all = np.array([r["tf_oracle"] for r in oracle_rows if np.isfinite(r["tf_oracle"])])
    tf_c   = tf_all - tf_all.mean()
    c0     = float(tf_c @ tf_c)
    acf1   = float(tf_c[1:] @ tf_c[:-1]) / c0
    print(f"Автокорреляция tf_oracle lag-1: {acf1:.3f}")
    print()

    # ── CSV с признаками ─────────────────────────────────────────────────────
    out_csv = RESULTS / f"features_{INTERVAL}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "tf_oracle", "kappa"] + FEATURE_NAMES)
        for r in feat_rows:
            w.writerow([r["step"], r["tf_oracle"], r["kappa"]]
                       + [r[fn] for fn in FEATURE_NAMES])
    print(f"Features CSV → {out_csv}")

    plot_correlations(corr_tf, corr_kappa, feat_rows,
                      RESULTS / f"correlations_{INTERVAL}.png")
    plot_top_scatter(feat_rows, corr_tf,
                     RESULTS / f"scatter_top_{INTERVAL}.png", top_n=6)
    plot_autocorr(feat_rows,
                  RESULTS / f"autocorr_tf_{INTERVAL}.png", max_lag=20)


if __name__ == "__main__":
    main()
