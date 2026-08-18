#!/usr/bin/env python3
"""
18_direction_oracle.py — Oracle T_frac с разбивкой по направлению прогноза.

Для каждого шага walk-forward (пивот T_BIG зигзага):
  - Перебираем все T_frac из сетки
  - Для каждого T_frac: LWR с фильтром по направлению (как в lwr_ref.py)
  - oracle_tf = argmin |err| для данной точки
  - Сохраняем direction (+1=HIGH→down-forecast, -1=LOW→up-forecast)

Графики:
  1. Гистограммы oracle T_frac: up vs down (раздельно)
  2. CDF: up vs down на одном графике
  3. Scatter oracle T_frac по времени: up и down

Данные: SBER 10m, T_BIG=4%, M=2, K=50, H=1
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

T_BIG  = 0.04
M      = 2       # embedding dim (число плечей в векторе)
K      = 50      # число соседей LWR
H      = 1       # горизонт (следующий пивот)
MIN_HISTORY = 20 # минимум пивотов T_BIG до начала walk-forward

T_FRAC_GRID = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
               0.010, 0.012, 0.015, 0.020, 0.025, 0.030, 0.040]


# ── данные ───────────────────────────────────────────────────────────────────

def load_candles(path):
    with open(path) as f:
        raw = json.load(f)
    lh = np.log(np.array([c["high"]  for c in raw], dtype=np.float64))
    ll = np.log(np.array([c["low"]   for c in raw], dtype=np.float64))
    dt = np.array([c["begin"] for c in raw])
    return lh, ll, dt


# ── зигзаг ───────────────────────────────────────────────────────────────────

def build_zigzag(lh, ll, dt, thr):
    """
    Каузальный зигзаг в log-price. Пивот фиксируется при подтверждении
    противоположной стороной (дата подтверждения, не экстремума).

    Возвращает (log_pivot_prices, confirm_dates, directions).
      directions: +1 = HIGH пивот (следующий прогноз вниз)
                  -1 = LOW  пивот (следующий прогноз вверх)
    """
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


# ── пул признаков ─────────────────────────────────────────────────────────────

def build_pool(lp, cd, dirs, m):
    """
    Для каждого пивота j (от m до n-2):
      feature[lag] = lp[j-lag] - lp[j-lag-1]  (лог-доходность плеча lag)
      target       = lp[j+1]   - lp[j]         (лог-доходность до след. пивота)
      date         = cd[j]     (дата подтверждения пивота j, для каузального среза)

    Возвращает (features, targets, directions, pivot_confirm_dates).
    """
    rows, tgts, row_dirs, row_dates = [], [], [], []
    for j in range(m, len(lp) - 1):
        feat = np.array([lp[j - lag] - lp[j - lag - 1] for lag in range(m)])
        if not np.all(np.isfinite(feat)):
            continue
        tgt = lp[j + 1] - lp[j]
        if not np.isfinite(tgt):
            continue
        rows.append(feat)
        tgts.append(tgt)
        row_dirs.append(dirs[j])
        row_dates.append(cd[j])
    if not rows:
        return None
    return (np.array(rows), np.array(tgts),
            np.array(row_dirs, dtype=np.int8), np.array(row_dates))


# ── LWR ──────────────────────────────────────────────────────────────────────

def lwr_predict(qv, fm, tgt, indices):
    """
    LWR: гауссово ядро, bandwidth = d_max среди K соседей.
    Возвращает прогнозную лог-доходность.
    """
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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    lh, ll, dt = load_candles(DATA / "10m.json")
    print(f"Свечей: {len(dt)}  ({dt[0][:10]} … {dt[-1][:10]})")

    # T_BIG зигзаг (весь датасет — каузальность через срез по дате в цикле)
    lp_big, cd_big, dir_big = build_zigzag(lh, ll, dt, T_BIG)
    n_big = len(lp_big)
    print(f"T_BIG={T_BIG*100:.0f}%  пивотов: {n_big}\n")

    # Предвычисляем пулы для каждого T_frac (весь датасет)
    pools = {}
    for tf in T_FRAC_GRID:
        lp_f, cd_f, dir_f = build_zigzag(lh, ll, dt, tf)
        result = build_pool(lp_f, cd_f, dir_f, M)
        if result is None:
            continue
        pools[tf] = result
        print(f"  T_frac={tf*100:.1f}%  строк пула: {len(result[0])}")

    # Walk-forward: oracle per-point
    records = []
    skipped = 0

    for step in range(max(MIN_HISTORY, M), n_big - H):
        # Вектор запроса из T_BIG зигзага (M лог-доходностей последних плечей)
        qv = np.array([lp_big[step - lag] - lp_big[step - lag - 1]
                       for lag in range(M)])
        if not np.all(np.isfinite(qv)):
            skipped += 1
            continue

        qdir      = int(dir_big[step])
        actual_lp = lp_big[step + H]

        # Прогноз при каждом T_frac
        tf_errors = {}
        for tf, (fm, tgt, dirs, pool_dates) in pools.items():
            # Каузальный срез: только пивоты пула с датой подтверждения < cd_big[step]
            ce = int(np.searchsorted(pool_dates, cd_big[step], side="left"))
            if ce < K + 1:
                continue

            fm_s   = fm[:ce]
            tgt_s  = tgt[:ce]
            dirs_s = dirs[:ce]

            # Только однонаправленные соседи (как в lwr_ref.py)
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
            skipped += 1
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

    df = pd.DataFrame(records)
    df.to_csv(RESULTS / "oracle_direction_10m.csv", index=False)
    print(f"\nОраклов: {len(df)}  (пропущено: {skipped})")

    up_df   = df[df["direction"] == -1]   # LOW пивот → прогноз роста
    down_df = df[df["direction"] == +1]   # HIGH пивот → прогноз падения
    print(f"  up   (LOW  pivot): {len(up_df)} точек,  медиана oracle_tf={up_df['oracle_tf'].median()*100:.1f}%")
    print(f"  down (HIGH pivot): {len(down_df)} точек,  медиана oracle_tf={down_df['oracle_tf'].median()*100:.1f}%")

    _plot_histograms(df, up_df, down_df)
    _plot_cdf(up_df, down_df)
    _plot_rolling_vis(up_df, down_df)
    print("\nГрафики сохранены в results/")


# ── графики ───────────────────────────────────────────────────────────────────

def _plot_histograms(df, up_df, down_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    xs     = np.array(T_FRAC_GRID) * 100
    labels = [f"{v:.1f}" for v in xs]

    for ax, sub_df, title, color in [
        (axes[0], up_df,   "Восходящий прогноз\n(LOW pivot → прогноз роста)",    "#2196F3"),
        (axes[1], down_df, "Нисходящий прогноз\n(HIGH pivot → прогноз падения)", "#F44336"),
    ]:
        counts = [(sub_df["oracle_tf"] == tf).sum() for tf in T_FRAC_GRID]
        ax.bar(xs, counts, width=(xs[1] - xs[0]) * 0.8, color=color, alpha=0.82)
        med = sub_df["oracle_tf"].median() * 100
        ax.axvline(med, color="black", linestyle="--", linewidth=1.5,
                   label=f"Медиана {med:.1f}%")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Oracle T_frac (%)")
        ax.set_ylabel("Число точек")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Oracle T_frac по направлению прогноза  (SBER 10m, T_big=4%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "hist_direction_10m.png", dpi=150)
    plt.close(fig)


def _plot_cdf(up_df, down_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    for sub_df, label, color in [
        (up_df,   "Восходящий (up)",    "#2196F3"),
        (down_df, "Нисходящий (down)",  "#F44336"),
    ]:
        vals = np.sort(sub_df["oracle_tf"].values) * 100
        cdf  = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, cdf, label=label, color=color, linewidth=2, where="post")
    ax.set_xlabel("Oracle T_frac (%)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF оптимального T_frac по направлению  (SBER 10m, T_big=4%)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "cdf_direction_10m.png", dpi=150)
    plt.close(fig)


def _plot_rolling_vis(up_df, down_df):
    parse = lambda s: datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    M     = 50  # окно rolling (~383 точек на направление, M=50 ≈ 13%)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    for ax, sub_df, title, color in [
        (axes[0], up_df,   "Восходящий прогноз  (LOW pivot → прогноз роста)",    "steelblue"),
        (axes[1], down_df, "Нисходящий прогноз  (HIGH pivot → прогноз падения)", "tomato"),
    ]:
        srt    = sub_df.sort_values("confirm_dt")
        dt     = np.array([parse(s) for s in srt["confirm_dt"]])
        tf_seq = srt["oracle_tf"].values

        roll_mean = np.array([
            np.mean(tf_seq[max(0, i - M):i]) if i >= M else np.nan
            for i in range(len(tf_seq))
        ])
        roll_med = np.array([
            np.median(tf_seq[max(0, i - M):i]) if i >= M else np.nan
            for i in range(len(tf_seq))
        ])

        ax.plot(dt, tf_seq * 100, color="lightgray", lw=0.7, alpha=0.9,
                label="oracle T_frac (raw)")
        ax.plot(dt, roll_mean * 100, color=color, lw=1.4,
                label=f"rolling mean  M={M}")
        ax.plot(dt, roll_med  * 100, color=color, lw=1.1, ls="--", alpha=0.8,
                label=f"rolling median  M={M}")
        ax.set_ylabel("Oracle T_frac (%)")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, ncol=3)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    fig.suptitle("Oracle T_frac по времени: up vs down  (SBER 10m, T_big=4%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "rolling_tf_vis_direction_10m.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
