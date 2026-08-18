#!/usr/bin/env python3
"""
zigzag_compare_T4.py

Сравнение пивотов T=4% на 1d и T=4% на 10m.
График: цена 1d + оба набора пивотов на одном полотне.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT      = BASE_DIR / "results"
OUT.mkdir(exist_ok=True)

T = 0.04   # одинаковый порог


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    dates  = [datetime.fromisoformat(d["begin"][:10]) for d in raw]
    highs  = np.array([d["high"]  for d in raw], dtype=np.float64)
    lows   = np.array([d["low"]   for d in raw], dtype=np.float64)
    closes = np.array([d["close"] for d in raw], dtype=np.float64)
    return dates, highs, lows, closes


def find_pivots(highs, lows, dates, thr):
    vals, dts, ptypes = [], [], []
    direction = 0
    ext_val = (highs[0] + lows[0]) / 2.0
    ext_idx = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1; ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val, ext_idx = lows[i], i
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx]); ptypes.append(1)
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx]); ptypes.append(-1)
                direction = 1; ext_val, ext_idx = highs[i], i
    return np.array(vals), dts, np.array(ptypes, dtype=int)


def main():
    d1d,  h1d,  l1d,  c1d  = load_tf("1d")
    d10m, h10m, l10m, c10m = load_tf("10m")

    # Пивоты T=4% на обоих TF
    p1d,  dt1d,  t1d  = find_pivots(h1d,  l1d,  d1d,  T)
    p10m, dt10m, t10m = find_pivots(h10m, l10m, d10m, T)

    print(f"1d  T=4%: {len(p1d)} пивотов")
    print(f"10m T=4%: {len(p10m)} пивотов")

    # Совпадения: для каждого 1d-пивота ищем ближайший 10m-пивот по дате
    dt1d_arr  = np.array([d.toordinal() for d in dt1d])
    dt10m_arr = np.array([d.toordinal() for d in dt10m])
    match_days = []
    price_diff = []
    for i, (d, v) in enumerate(zip(dt1d_arr, p1d)):
        j = int(np.argmin(np.abs(dt10m_arr - d)))
        day_diff   = abs(int(dt10m_arr[j]) - int(d))
        pct_diff   = abs(p10m[j] - v) / v * 100
        match_days.append(day_diff)
        price_diff.append(pct_diff)
    match_days  = np.array(match_days)
    price_diff  = np.array(price_diff)

    print(f"\nСоответствие 1d-пивот ↔ ближайший 10m-пивот (T=4%):")
    print(f"  Разница по дате: median={np.median(match_days):.0f} дн  "
          f"p75={np.percentile(match_days,75):.0f} дн  "
          f"max={match_days.max():.0f} дн")
    print(f"  Разница по цене: median={np.median(price_diff):.2f}%  "
          f"p75={np.percentile(price_diff,75):.2f}%  "
          f"max={price_diff.max():.2f}%")
    print(f"  Совпадают в ±1 день: {(match_days <= 1).mean()*100:.1f}%")
    print(f"  Совпадают в ±3 дня:  {(match_days <= 3).mean()*100:.1f}%")
    print(f"  Совпадают в ±7 дней: {(match_days <= 7).mean()*100:.1f}%")

    # ── Графики ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(18, 11),
                             gridspec_kw={"height_ratios": [2, 1]})

    # ── Верхний: цена 1d + пивоты обоих TF ───────────────────────────────────
    ax = axes[0]
    ax.plot(d1d, c1d, lw=0.6, color="#aaaaaa", zorder=1, label="SBER 1d close")

    # 1d пивоты
    for v, d, tp in zip(p1d, dt1d, t1d):
        marker = "^" if tp == -1 else "v"   # ^ для LOW (впадина), v для HIGH (вершина)
        color  = "green" if tp == -1 else "red"
        ax.scatter(d, v, marker=marker, s=60, color=color, zorder=3,
                   linewidths=0.5, edgecolors="black")

    # 10m пивоты T=4% — чуть прозрачнее, другой размер
    for v, d, tp in zip(p10m, dt10m, t10m):
        marker = "^" if tp == -1 else "v"
        color  = "#00cc44" if tp == -1 else "#ff6600"
        ax.scatter(d, v, marker=marker, s=20, color=color, zorder=2,
                   alpha=0.6, linewidths=0)

    # Легенда
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker="v", color="w", markerfacecolor="red",
               markersize=9, markeredgecolor="black", label="1d HIGH (вершина)"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="green",
               markersize=9, markeredgecolor="black", label="1d LOW (впадина)"),
        Line2D([0],[0], marker="v", color="w", markerfacecolor="#ff6600",
               markersize=7, label="10m HIGH T=4%"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#00cc44",
               markersize=7, label="10m LOW T=4%"),
    ]
    ax.legend(handles=legend_els, loc="upper left", fontsize=8)
    ax.set_title(f"SBER: пивоты T=4% на 1d (крупные) и 10m (мелкие)\n"
                 f"1d: {len(p1d)} пив   10m: {len(p10m)} пив", fontsize=11)
    ax.set_ylabel("Цена, руб.")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

    # ── Нижний: гистограмма сдвига по дате (ближайший 10m-сосед) ─────────────
    ax2 = axes[1]
    bins = range(0, min(int(match_days.max()) + 2, 31))
    ax2.hist(match_days, bins=bins, color="steelblue", alpha=0.8, edgecolor="white")
    ax2.axvline(np.median(match_days), color="red", lw=1.5, ls="--",
                label=f"median={np.median(match_days):.0f} дн")
    ax2.axvline(1, color="orange", lw=1.0, ls=":",
                label=f"±1 день: {(match_days<=1).mean()*100:.0f}%")
    ax2.set_xlabel("Разница в днях (1d-пивот ↔ ближайший 10m-пивот T=4%)")
    ax2.set_ylabel("count")
    ax2.set_title("Насколько 10m T=4% совпадает с 1d T=4% по дате")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUT / "zigzag_compare_T4.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {out_path}")


if __name__ == "__main__":
    main()
