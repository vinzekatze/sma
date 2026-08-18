"""
Визуализация прогнозов пивотов в пространстве реальных цен (рублях).

Для каждого тикера:
  1. Полная панорама: цена OHLC + зигзаг + прогнозы (последние 120 шагов)
  2. Зум: последние 40 шагов со стрелками прогнозов
  3. Scatter pred vs actual в ценовом пространстве
  4. Rolling-ошибки
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

BASE   = Path(__file__).parent
DATA   = BASE.parent.parent.parent / "data" / "candles"
OUT    = BASE / "results"

THRESH = 0.02
VARIANTS = ["ratio", "raw", "swing_pct", "log_swing", "ratio_zscore"]
COLORS = {
    "ratio":        "steelblue",
    "raw":          "#777777",
    "swing_pct":    "darkorange",
    "log_swing":    "#2ca02c",
    "ratio_zscore": "purple",
}
LABELS = {
    "ratio":        "ratio (BASE)",
    "raw":          "raw price",
    "swing_pct":    "swing_pct",
    "log_swing":    "log_swing",
    "ratio_zscore": "ratio_zscore",
}


# ── Утилиты ──────────────────────────────────────────────────────────────────

def logtrend_causal(close):
    n  = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    N  = np.arange(1, n + 1, dtype=np.float64)
    St, Sp   = np.cumsum(t),      np.cumsum(lc)
    St2, Stp = np.cumsum(t ** 2), np.cumsum(t * lc)
    den = N * St2 - St ** 2
    b   = np.where(den > 1e-12, (N * Stp - St * Sp) / den, 0.0)
    a   = (Sp - b * St) / N
    trd = np.exp(a + b * t)
    trd[:2] = close[:2]
    return trd


def find_pivots(ratio, thr):
    pivots    = [0]
    direction = 0
    ext_val, ext_idx = ratio[0], 0
    for i in range(1, len(ratio)):
        v = ratio[i]
        if direction == 0:
            if abs(v - ext_val) >= thr * ext_val:
                direction = 1 if v > ext_val else -1
                ext_val, ext_idx = v, i
        elif direction == 1:
            if v > ext_val:
                ext_val, ext_idx = v, i
            elif (ext_val - v) >= thr * ext_val:
                pivots.append(ext_idx)
                direction = -1
                ext_val, ext_idx = v, i
        else:
            if v < ext_val:
                ext_val, ext_idx = v, i
            elif (v - ext_val) >= thr * ext_val:
                pivots.append(ext_idx)
                direction = 1
                ext_val, ext_idx = v, i
    return np.array(pivots, dtype=int)


def load_series(ticker):
    path = DATA / ticker / "1d.json"
    with open(path) as f:
        data = json.load(f)
    close  = np.array([d["close"] for d in data], dtype=np.float64)
    open_  = np.array([d["open"]  for d in data], dtype=np.float64)
    high   = np.array([d["high"]  for d in data], dtype=np.float64)
    low    = np.array([d["low"]   for d in data], dtype=np.float64)
    dates  = np.array([d["begin"] for d in data])
    trend  = logtrend_causal(close)
    ratio  = close / trend
    pivots = find_pivots(ratio, THRESH)
    return close, open_, high, low, dates, trend, ratio, pivots


def pred_ratio_to_price(pred_ratio_vals, trend_at_step):
    """
    pred_ratio_vals — массив предсказанных ratio из CSV.
    trend_at_step   — trend в баре текущего пивота (step).
    Возвращает предсказанные ЦЕНЫ (в той же валюте что close).
    """
    return pred_ratio_vals * trend_at_step


# ── Основной график ──────────────────────────────────────────────────────────

def make_price_chart(ticker, window=120, zoom=40):
    df = pd.read_csv(OUT / f"{ticker}_norm_study.csv")
    close, open_, high, low, dates, trend, ratio, pivots = load_series(ticker)

    # Тестовые шаги (последние window)
    df_w = df.tail(window).copy().reset_index(drop=True)
    steps = df_w["step"].values          # индекс пивота в массиве pivots[]

    # Bar-индексы пивотов в calendar-time
    piv_bars       = pivots[steps]
    piv_bars_next  = pivots[(steps + 1).clip(max=len(pivots) - 1)]

    # Реальные ЦЕНЫ: текущий и следующий пивот
    price_cur      = close[piv_bars]
    price_next_act = close[piv_bars_next]   # целевая цена (actual следующего пивота)

    # M0 в ценах
    price_m0 = price_cur  # M0: предсказываем текущую цену

    # Trend в баре шага (для конвертации ratio → цена)
    trend_cur = trend[piv_bars]

    # Предсказанные ЦЕНЫ для каждого варианта
    pred_prices = {}
    for v in VARIANTS:
        pred_prices[v] = pred_ratio_to_price(df_w[f"pred_{v}"].values, trend_cur)

    # Ошибки (цена)
    price_mean_dz = np.nanmean(np.abs(np.diff(price_next_act)))

    # === Рисуем ===============================================================
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(
        f"{ticker}: прогнозы пивотов в ценовом пространстве\n"
        f"последние {window} шагов теста  (T=2%, θ=8, p=2)",
        fontsize=12)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Полный вид: цены пивотов + все прогнозы ───────────────────────────
    ax = fig.add_subplot(gs[0, :])

    # Реальный зигзаг (линия по текущим пивотам)
    ax.plot(steps, price_cur, color="black", lw=1.4, marker="o", ms=3,
            label="actual (текущий пивот)")
    # Следующий actual (target)
    ax.scatter(steps + 1, price_next_act, color="black", s=30, marker="x",
               zorder=5, label="actual_next (target)")
    # M0
    ax.scatter(steps + 1, price_m0, color="#cccccc", s=15, marker="^",
               zorder=3, label="M0 (random walk)")
    # Прогнозы вариантов (только основные 4, ratio_zscore пунктиром)
    for v in ["ratio", "swing_pct", "log_swing", "raw"]:
        ax.scatter(steps + 1, pred_prices[v],
                   color=COLORS[v], s=18, marker="D", alpha=0.75,
                   zorder=4, label=LABELS[v])
    ax.scatter(steps + 1, pred_prices["ratio_zscore"],
               color=COLORS["ratio_zscore"], s=12, marker="+", alpha=0.5,
               label=LABELS["ratio_zscore"])

    ax.set_title("Цены пивотов и прогнозы следующего пивота")
    ax.set_xlabel("step (индекс пивота)")
    ax.set_ylabel("Цена, руб.")
    ax.legend(fontsize=7.5, ncol=3)
    ax.grid(alpha=0.2)

    # ── 2. Зум: последние zoom шагов, ratio vs raw ───────────────────────────
    df_z  = df_w.tail(zoom).reset_index(drop=True)
    st_z  = df_z["step"].values
    pb_z  = pivots[st_z]
    pb_z1 = pivots[(st_z + 1).clip(max=len(pivots) - 1)]
    pc_z  = close[pb_z]
    pn_z  = close[pb_z1]
    tr_z  = trend[pb_z]

    for col_idx, (var_pair, title) in enumerate([
        (["ratio", "raw"],           "ratio (BASE) vs raw price"),
        (["swing_pct", "log_swing"], "swing_pct vs log_swing"),
    ]):
        ax = fig.add_subplot(gs[1, col_idx])

        # Цены текущих пивотов
        ax.plot(st_z, pc_z, color="black", lw=1.6, marker="o", ms=5,
                label="actual (текущий)", zorder=5)
        # Следующий actual
        ax.scatter(st_z + 1, pn_z, color="black", s=40, marker="x",
                   zorder=6, label="actual_next")

        # Стрелки + маркеры прогнозов
        for _, row in df_z.iterrows():
            s = int(row["step"])
            bar_s = pivots[s]
            pc_s  = close[bar_s]
            tr_s  = trend[bar_s]
            for v in var_pair:
                pred_p = row[f"pred_{v}"] * tr_s
                if not np.isnan(pred_p):
                    ax.annotate(
                        "", xy=(s + 1, pred_p), xytext=(s, pc_s),
                        arrowprops=dict(
                            arrowstyle="->", color=COLORS[v],
                            lw=1.0, alpha=0.55))
        for v in var_pair:
            pp = pred_ratio_to_price(df_z[f"pred_{v}"].values, tr_z)
            ax.scatter(st_z + 1, pp,
                       color=COLORS[v], s=35, marker="D", alpha=0.9,
                       zorder=5, label=LABELS[v])

        ax.set_title(f"Зум ({zoom} шагов): {title}")
        ax.set_xlabel("step")
        ax.set_ylabel("Цена, руб.")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.2)

    # ── 3. Scatter pred_price vs actual_price ─────────────────────────────────
    for col_idx, var_group in enumerate([
        ["ratio", "raw"],
        ["swing_pct", "log_swing"],
    ]):
        ax = fig.add_subplot(gs[2, col_idx])
        lo, hi = price_next_act.min(), price_next_act.max()
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, ls="--",
                label="ideal")
        for v in var_group:
            pp  = pred_prices[v]
            msk = ~np.isnan(pp)
            if msk.sum() == 0:
                continue
            ax.scatter(price_next_act[msk], pp[msk],
                       color=COLORS[v], alpha=0.25, s=12,
                       label=LABELS[v])
            zf = np.polyfit(price_next_act[msk], pp[msk], 1)
            xs = np.linspace(lo, hi, 50)
            ax.plot(xs, np.polyval(zf, xs),
                    color=COLORS[v], lw=2.0, alpha=0.9)

            mae_p = np.nanmean(np.abs(pp - price_next_act))
            bias_p = np.nanmean(pp - price_next_act)
            rmae_p = mae_p / price_mean_dz
            lbl_extra = f"  rMAE={rmae_p:.3f}  bias={bias_p:+.1f}р"
            ax.scatter([], [], color=COLORS[v], s=0,
                       label=LABELS[v] + lbl_extra)

        handles = [h for h in ax.get_legend_handles_labels()[0]
                   if not isinstance(h, matplotlib.collections.PathCollection)
                   or h.get_offsets().shape[0] > 0]
        ax.legend(fontsize=6.5, loc="upper left")
        ax.set_title("Pred vs Actual (цена, руб.)")
        ax.set_xlabel("actual цена следующего пивота, руб.")
        ax.set_ylabel("pred цена, руб.")
        ax.grid(alpha=0.2)

    out_path = OUT / f"{ticker}_price_forecast.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  {ticker}: {out_path}")


# ── График ошибок по времени (цены) ──────────────────────────────────────────

def make_error_chart(ticker, window=200):
    df   = pd.read_csv(OUT / f"{ticker}_norm_study.csv")
    close, open_, high, low, dates, trend, ratio, pivots = load_series(ticker)

    df_w  = df.tail(window).copy().reset_index(drop=True)
    steps = df_w["step"].values
    pb    = pivots[steps]
    pb1   = pivots[(steps + 1).clip(max=len(pivots) - 1)]

    price_next_act = close[pb1]
    trend_cur      = trend[pb]
    mean_dz_p      = np.nanmean(np.abs(np.diff(price_next_act)))

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(
        f"{ticker}: ошибки прогноза (в рублях)  последние {window} шагов",
        fontsize=11)

    # Верхний: rolling signed_err (цена)
    ax = axes[0]
    ax.axhline(0, color="black", lw=0.9)
    for v in ["ratio", "swing_pct", "log_swing", "raw"]:
        pp   = pred_ratio_to_price(df_w[f"pred_{v}"].values, trend_cur)
        serr = pd.Series(pp - price_next_act).rolling(15, min_periods=5).mean()
        ax.plot(steps, serr, color=COLORS[v], lw=1.8,
                label=LABELS[v], alpha=0.85)
    m0_err = pd.Series(close[pb] - price_next_act).rolling(15, min_periods=5).mean()
    ax.plot(steps, m0_err, color="#aaaaaa", lw=1.0, ls="--",
            label="M0", alpha=0.7)
    ax.set_ylabel("rolling mean (pred−actual), руб.  (w=15)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    # Нижний: rolling rMAE (цена)
    ax = axes[1]
    m0_roll = pd.Series(np.abs(close[pb] - price_next_act)).rolling(15, min_periods=5).mean() / mean_dz_p
    ax.plot(steps, m0_roll, color="#aaaaaa", lw=1.0, ls="--",
            label="M0", alpha=0.7)
    for v in ["ratio", "swing_pct", "log_swing", "raw"]:
        pp   = pred_ratio_to_price(df_w[f"pred_{v}"].values, trend_cur)
        roll = pd.Series(np.abs(pp - price_next_act)).rolling(15, min_periods=5).mean() / mean_dz_p
        ax.plot(steps, roll, color=COLORS[v], lw=1.8,
                label=LABELS[v], alpha=0.85)
    ax.set_ylabel("rolling rMAE (w=15)")
    ax.set_xlabel("step (пивот)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    out_path = OUT / f"{ticker}_price_errors.png"
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  {ticker}: {out_path}")


def main():
    for ticker in ["SBER", "LKOH"]:
        csv_path = OUT / f"{ticker}_norm_study.csv"
        if not csv_path.exists():
            print(f"  {ticker}: CSV не найден, пропускаю")
            continue
        print(f"\n{ticker}:")
        make_price_chart(ticker, window=120, zoom=40)
        make_error_chart(ticker, window=200)
    print("\nГотово.")


if __name__ == "__main__":
    main()
