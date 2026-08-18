#!/usr/bin/env python3
"""
20_stability_oracle.py — Стабильность oracle T_frac между соседними прогнозами.

Сетка T_frac: [0.015..0.040] шаг 0.001 (26 точек) — плотная зона оптимума.
Данные: SBER 10m, T_BIG=4%, M=2, K=50, H=1.

Графики:
  1. Rolling vis по датам (up / down раздельно) — как в скр.18
  2. Scatter lag-1: oracle_tf[i] vs oracle_tf[i+1] внутри каждого направления
  3. ACF oracle_tf: лаги 1-15, up vs down
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
from pathlib import Path

HERE    = Path(__file__).parent
DATA    = HERE.parent.parent.parent / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

T_BIG       = 0.04
M           = 2
K           = 50
H           = 1
MIN_HISTORY = 20

T_FRAC_GRID = [round(v, 3) for v in np.arange(0.015, 0.041, 0.001)]  # 26 точек


# ── данные / зигзаг / пул / LWR ──────────────────────────────────────────────

def load_candles(path):
    with open(path) as f:
        raw = json.load(f)
    lh = np.log(np.array([c["high"]  for c in raw], dtype=np.float64))
    ll = np.log(np.array([c["low"]   for c in raw], dtype=np.float64))
    dt = np.array([c["begin"] for c in raw])
    return lh, ll, dt


def build_zigzag(lh, ll, dt, thr):
    lp, cd, dirs = [], [], []
    cur_dir = 0
    ext     = (lh[0] + ll[0]) / 2.0
    for i in range(len(lh)):
        if cur_dir == 0:
            if lh[i] - ext >= thr:
                cur_dir, ext = 1, lh[i]
            elif ext - ll[i] >= thr:
                cur_dir, ext = -1, ll[i]
        elif cur_dir == 1:
            if lh[i] > ext:
                ext = lh[i]
            elif ext - ll[i] >= thr:
                lp.append(ext); cd.append(dt[i]); dirs.append(+1)
                cur_dir, ext = -1, ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= thr:
                lp.append(ext); cd.append(dt[i]); dirs.append(-1)
                cur_dir, ext = 1, lh[i]
    return np.array(lp), np.array(cd), np.array(dirs, dtype=np.int8)


def build_pool(lp, cd, dirs, m):
    rows, tgts, row_dirs, row_dates = [], [], [], []
    for j in range(m, len(lp) - 1):
        feat = np.array([lp[j - lag] - lp[j - lag - 1] for lag in range(m)])
        if not np.all(np.isfinite(feat)):
            continue
        tgt = lp[j + 1] - lp[j]
        if not np.isfinite(tgt):
            continue
        rows.append(feat); tgts.append(tgt)
        row_dirs.append(dirs[j]); row_dates.append(cd[j])
    if not rows:
        return None
    return (np.array(rows), np.array(tgts),
            np.array(row_dirs, dtype=np.int8), np.array(row_dates))


def lwr_predict(qv, fm, tgt, indices):
    feats = fm[indices]
    dists = np.linalg.norm(feats - qv, axis=1)
    d_max = dists.max()
    if d_max < 1e-12:
        return float(tgt[indices].mean())
    w  = np.exp(-0.5 * (dists / d_max) ** 2)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(len(indices)), feats]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, tgt[indices] * ws, rcond=None)
    return float(c[0] + c[1:] @ qv)


# ── oracle walk-forward ───────────────────────────────────────────────────────

def oracle_walkforward(lp_big, cd_big, dir_big, pools):
    records = []
    n_big   = len(lp_big)

    for step in range(max(MIN_HISTORY, M), n_big - H):
        qv = np.array([lp_big[step - lag] - lp_big[step - lag - 1]
                       for lag in range(M)])
        if not np.all(np.isfinite(qv)):
            continue

        qdir      = int(dir_big[step])
        actual_lp = lp_big[step + H]

        tf_errors = {}
        for tf, (fm, tgt, dirs, pool_dates) in pools.items():
            ce = int(np.searchsorted(pool_dates, cd_big[step], side="left"))
            if ce < K + 1:
                continue
            fm_s   = fm[:ce];  tgt_s = tgt[:ce];  dirs_s = dirs[:ce]
            mask     = dirs_s == qdir
            cand_idx = np.where(mask)[0]
            if len(cand_idx) < K:
                continue
            dists   = np.linalg.norm(fm_s[cand_idx] - qv, axis=1)
            top_k   = cand_idx[np.argpartition(dists, K - 1)[:K]]
            pred_lr = lwr_predict(qv, fm_s, tgt_s, top_k)
            err     = np.exp(lp_big[step] + pred_lr) - np.exp(actual_lp)
            tf_errors[tf] = float(err)

        if len(tf_errors) < 2:
            continue

        tfs      = np.array(list(tf_errors.keys()))
        abs_errs = np.abs(np.array(list(tf_errors.values())))
        best_idx = np.argmin(abs_errs)

        records.append({
            "step":       step,
            "direction":  qdir,
            "oracle_tf":  tfs[best_idx],
            "oracle_err": abs_errs[best_idx],
            "confirm_dt": cd_big[step],
        })

    return pd.DataFrame(records)


# ── автокорреляция ────────────────────────────────────────────────────────────

def acf(x, max_lag):
    x  = x - x.mean()
    c0 = np.dot(x, x)
    return np.array([np.dot(x[:len(x)-lag], x[lag:]) / c0
                     for lag in range(1, max_lag + 1)])


# ── графики ───────────────────────────────────────────────────────────────────

def plot_rolling_vis(up_df, down_df):
    parse = lambda s: datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    M_win = 50

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax, sub_df, title, color in [
        (axes[0], up_df,   "Восходящий прогноз  (LOW pivot → прогноз роста)",    "steelblue"),
        (axes[1], down_df, "Нисходящий прогноз  (HIGH pivot → прогноз падения)", "tomato"),
    ]:
        srt    = sub_df.sort_values("confirm_dt")
        dt     = np.array([parse(s) for s in srt["confirm_dt"]])
        tf_seq = srt["oracle_tf"].values * 100

        roll_mean = np.array([
            np.mean(tf_seq[max(0, i - M_win):i]) if i >= M_win else np.nan
            for i in range(len(tf_seq))
        ])
        roll_med = np.array([
            np.median(tf_seq[max(0, i - M_win):i]) if i >= M_win else np.nan
            for i in range(len(tf_seq))
        ])

        ax.plot(dt, tf_seq,      color="lightgray", lw=0.7, alpha=0.9, label="oracle T_frac (raw)")
        ax.plot(dt, roll_mean,   color=color,        lw=1.4, label=f"rolling mean  M={M_win}")
        ax.plot(dt, roll_med,    color=color,        lw=1.1, ls="--", alpha=0.8,
                label=f"rolling median  M={M_win}")
        ax.set_ylabel("Oracle T_frac (%)")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    fig.suptitle("Oracle T_frac по времени (сетка 0.1%)  —  SBER 10m, T_big=4%", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "stability_rolling_vis_10m.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → stability_rolling_vis_10m.png")


def plot_lag1_scatter(up_df, down_df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, sub_df, title, color in [
        (axes[0], up_df,   "Восходящий (up)",    "steelblue"),
        (axes[1], down_df, "Нисходящий (down)",  "tomato"),
    ]:
        srt = sub_df.sort_values("confirm_dt")
        tf  = srt["oracle_tf"].values * 100
        x, y = tf[:-1], tf[1:]

        # jitter для раскрытия наложенных точек
        jitter = np.random.default_rng(0).uniform(-0.03, 0.03, size=len(x))
        ax.scatter(x + jitter, y + jitter, alpha=0.3, s=10, color=color)

        # линия y=x
        lim = [min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1]
        ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)

        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{title}\nlag-1  r = {r:.3f}", fontsize=11)
        ax.set_xlabel("oracle_tf[i] (%)")
        ax.set_ylabel("oracle_tf[i+1] (%)")
        ax.grid(alpha=0.25)

    fig.suptitle("Scatter lag-1: oracle T_frac между соседними прогнозами  (SBER 10m, T_big=4%)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "stability_lag1_scatter_10m.png", dpi=150)
    plt.close(fig)
    print("  → stability_lag1_scatter_10m.png")


def plot_acf(up_df, down_df):
    MAX_LAG = 15
    fig, ax = plt.subplots(figsize=(9, 4))

    for sub_df, label, color in [
        (up_df,   "up",   "steelblue"),
        (down_df, "down", "tomato"),
    ]:
        srt    = sub_df.sort_values("confirm_dt")
        tf_seq = srt["oracle_tf"].values
        lags   = np.arange(1, MAX_LAG + 1)
        corrs  = acf(tf_seq, MAX_LAG)
        ax.plot(lags, corrs, marker="o", markersize=5, color=color,
                linewidth=1.4, label=label)

    # 95% CI для белого шума: ±1.96/√n
    n_min = min(len(up_df), len(down_df))
    ci    = 1.96 / np.sqrt(n_min)
    ax.axhline( ci, color="gray", lw=0.8, ls="--", alpha=0.7, label=f"95% CI ±{ci:.3f}")
    ax.axhline(-ci, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(0,   color="black", lw=0.5)

    ax.set_xlabel("Лаг (число событий одного направления)")
    ax.set_ylabel("Автокорреляция")
    ax.set_title("ACF oracle T_frac  (SBER 10m, T_big=4%)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xticks(np.arange(1, MAX_LAG + 1))
    fig.tight_layout()
    fig.savefig(RESULTS / "stability_acf_10m.png", dpi=150)
    plt.close(fig)
    print("  → stability_acf_10m.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    lh, ll, dt = load_candles(DATA / "10m.json")
    print(f"Свечей: {len(dt)}  ({dt[0][:10]} … {dt[-1][:10]})")

    lp_big, cd_big, dir_big = build_zigzag(lh, ll, dt, T_BIG)
    print(f"T_BIG={T_BIG*100:.0f}%  пивотов: {len(lp_big)}")
    print(f"Сетка T_frac: {T_FRAC_GRID[0]*100:.1f}%–{T_FRAC_GRID[-1]*100:.1f}%  "
          f"шаг 0.1%  ({len(T_FRAC_GRID)} точек)\n")

    print("Строим пулы...")
    pools = {}
    for tf in T_FRAC_GRID:
        lp_f, cd_f, dir_f = build_zigzag(lh, ll, dt, tf)
        result = build_pool(lp_f, cd_f, dir_f, M)
        if result is not None:
            pools[tf] = result
    print(f"  пулов построено: {len(pools)}\n")

    print("Oracle walk-forward...")
    df = oracle_walkforward(lp_big, cd_big, dir_big, pools)
    df.to_csv(RESULTS / "stability_oracle_10m.csv", index=False)

    up_df   = df[df["direction"] == -1].reset_index(drop=True)
    down_df = df[df["direction"] == +1].reset_index(drop=True)

    print(f"Оракулов: {len(df)}")
    print(f"  up   медиана={up_df['oracle_tf'].median()*100:.2f}%  "
          f"std={up_df['oracle_tf'].std()*100:.2f}%  n={len(up_df)}")
    print(f"  down медиана={down_df['oracle_tf'].median()*100:.2f}%  "
          f"std={down_df['oracle_tf'].std()*100:.2f}%  n={len(down_df)}")

    print("\nГрафики...")
    plot_rolling_vis(up_df, down_df)
    plot_lag1_scatter(up_df, down_df)
    plot_acf(up_df, down_df)
    print("Готово.")


if __name__ == "__main__":
    main()
