#!/usr/bin/env python3
"""Визуализация зигзага T=4% на ценовом графике SBER 1d (последние N баров)."""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER" / "1d.json"
OUT      = BASE_DIR / "results"

T     = 0.04
N_BAR = 300   # последние N баров


def load_1d():
    with open(DATA) as f:
        data = json.load(f)
    o = np.array([d["open"]  for d in data], dtype=np.float64)
    h = np.array([d["high"]  for d in data], dtype=np.float64)
    l = np.array([d["low"]   for d in data], dtype=np.float64)
    c = np.array([d["close"] for d in data], dtype=np.float64)
    dates = [d["begin"][:10] for d in data]
    return o, h, l, c, dates


def find_pivots_hl(highs, lows, thr):
    pivots    = []
    direction = 0
    ext_val   = (highs[0] + lows[0]) / 2.0
    ext_idx   = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1;  ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val, ext_idx = lows[i], i
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                pivots.append((ext_idx, ext_val, 'H'))
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                pivots.append((ext_idx, ext_val, 'L'))
                direction = 1;  ext_val, ext_idx = highs[i], i
    return pivots


def draw_candles(ax, idx_arr, opens, highs, lows, closes):
    for i, gi in enumerate(idx_arr):
        o, h, l, c = opens[gi], highs[gi], lows[gi], closes[gi]
        color = "#2ca02c" if c >= o else "#d62728"
        ax.plot([i, i], [l, h], color=color, lw=0.8, zorder=2)
        body_lo, body_hi = min(o, c), max(o, c)
        body_h = max(body_hi - body_lo, h * 0.001)
        ax.add_patch(Rectangle((i - 0.35, body_lo), 0.7, body_h,
                                color=color, zorder=3))


def run():
    o, h, l, c, dates = load_1d()
    n_total  = len(c)
    start    = max(0, n_total - N_BAR)
    idx_full = np.arange(start, n_total)

    pivots_full = find_pivots_hl(h, l, T)

    # Пивоты в диапазоне отображения
    piv_in_view = [(bi, pr, dr) for bi, pr, dr in pivots_full if bi >= start]

    # Локальный x-индекс
    def to_local(bar_idx):
        return bar_idx - start

    fig, ax = plt.subplots(figsize=(16, 6))
    draw_candles(ax, idx_full, o, h, l, c)

    # Зигзаг
    if piv_in_view:
        xs = [to_local(bi) for bi, _, _ in piv_in_view]
        ys = [pr for _, pr, _ in piv_in_view]
        ax.plot(xs, ys, color="dodgerblue", lw=1.8, zorder=4, label=f"Зигзаг T={T*100:.0f}%")
        for xi, yi, dr in zip(xs, ys, [dr for _, _, dr in piv_in_view]):
            color = "red" if dr == 'H' else "green"
            ax.scatter(xi, yi, color=color, s=40, zorder=5)

    # Метки по оси X: каждые ~50 баров
    tick_step = 50
    tick_pos   = list(range(0, N_BAR, tick_step))
    tick_lbl   = [dates[start + i] if start + i < n_total else "" for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=7)

    ax.set_xlim(-1, N_BAR)
    ax.set_ylabel("Цена, руб.")
    ax.set_title(f"SBER 1d — зигзаг HIGH/LOW, T={T*100:.0f}%  (последние {N_BAR} баров)\n"
                 f"пивотов в окне: {len(piv_in_view)}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()

    out = OUT / f"zigzag_T{int(T*100)}pct_chart.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"График: {out}")


if __name__ == "__main__":
    run()
