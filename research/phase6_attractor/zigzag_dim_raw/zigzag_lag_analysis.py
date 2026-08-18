#!/usr/bin/env python3
"""
zigzag_lag_analysis.py

Анализ задержки зигзага и момента получения прогноза.

Ключевые понятия:
  ext_idx  — бар, на котором произошёл реальный экстремум (HIGH или LOW)
  conf_idx — бар, на котором экстремум ПОДТВЕРЖДЁН (цена откатила на T%)

  lag   = conf_idx[K] - ext_idx[K]
          Сколько баров прошло от экстремума до его подтверждения.
          Именно тогда мы УЗНАЁМ, что пивот K зафиксирован, и можем
          запустить прогноз следующего пивота K+1.

  gap   = ext_idx[K+1] - conf_idx[K]
          Сколько баров от подтверждения K до реального экстремума K+1.
          Потенциальный горизонт позиции.

  inter = conf_idx[K+1] - conf_idx[K]
          Полный интервал между сигналами (lag[K+1] + gap[K+1]).

Вопрос о перерисовке:
  Алгоритм НИКОГДА не удаляет и не сдвигает уже зафиксированный пивот.
  После записи в vals[] точка неизменна. Перерисовки нет.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT      = BASE_DIR / "results"
OUT.mkdir(exist_ok=True)

T_1D  = 0.04
T_10M = 0.004


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64))


def find_pivots_detailed(highs, lows, thr):
    """
    Возвращает:
      vals       — цена пивота
      ext_idxs   — бар экстремума
      conf_idxs  — бар подтверждения (когда цена откатила на thr)
      ptypes     — тип: 1=HIGH, -1=LOW
    """
    vals, ext_idxs, conf_idxs, ptypes = [], [], [], []
    direction = 0
    ext_val = (highs[0] + lows[0]) / 2.0
    ext_idx = 0
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
                vals.append(ext_val)
                ext_idxs.append(ext_idx)   # реальный экстремум (уже в прошлом)
                conf_idxs.append(i)        # момент подтверждения (текущий бар)
                ptypes.append(1)
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val)
                ext_idxs.append(ext_idx)
                conf_idxs.append(i)
                ptypes.append(-1)
                direction = 1; ext_val, ext_idx = highs[i], i
    return (np.array(vals),
            np.array(ext_idxs,  dtype=int),
            np.array(conf_idxs, dtype=int),
            np.array(ptypes,    dtype=int))


def stats(arr, name):
    arr = arr[arr >= 0]  # исключить отрицательные (быть не должно, но на всякий)
    print(f"  {name}:")
    print(f"    min={arr.min():.0f}  p25={np.percentile(arr,25):.0f}  "
          f"median={np.median(arr):.0f}  p75={np.percentile(arr,75):.0f}  "
          f"max={arr.max():.0f}  mean={arr.mean():.1f}")


def analyze(label, thr, highs, lows):
    vals, ext_idxs, conf_idxs, ptypes = find_pivots_detailed(highs, lows, thr)
    n = len(vals)

    lag   = conf_idxs - ext_idxs                  # баров от пика до подтверждения
    gap   = ext_idxs[1:] - conf_idxs[:-1]         # от подтверждения до следующего пика
    inter = conf_idxs[1:] - conf_idxs[:-1]        # от сигнала до следующего сигнала

    # gap может быть отрицательным: пик следующего пивота сформировался ДО подтверждения
    # текущего — значит, прогноз «опаздывает» (следующий пик уже позади к моменту сигнала)
    gap_neg_pct = (gap < 0).mean() * 100

    print(f"\n{'═'*55}")
    print(f"  {label}  T={thr*100:.1f}%   пивотов: {n}   баров: {len(highs)}")
    print(f"{'═'*55}")
    stats(lag,   "lag   (пик → подтверждение, баров)")
    stats(inter, "inter (сигнал → следующий сигнал, баров)")
    print(f"  gap   (подтверждение → следующий пик):")
    print(f"    min={gap.min():.0f}  p25={np.percentile(gap,25):.0f}  "
          f"median={np.median(gap):.0f}  p75={np.percentile(gap,75):.0f}  "
          f"max={gap.max():.0f}  mean={gap.mean():.1f}")
    print(f"    gap < 0 (пик уже позади к моменту сигнала): {gap_neg_pct:.1f}%")

    # Распределение lag по типу пивота
    for ttype, lbl in [(1, "HIGH"), (-1, "LOW")]:
        m = ptypes == ttype
        l = lag[m]
        print(f"  lag по типу  {lbl}: "
              f"median={np.median(l):.0f}  mean={l.mean():.1f}  max={l.max():.0f}")

    return lag, gap, inter, gap_neg_pct


def main():
    h1d,  l1d  = load_tf("1d")
    h10m, l10m = load_tf("10m")

    print("SBER — Анализ задержки зигзага")
    print("=" * 55)
    print("""
Алгоритм подтверждает пивот только когда цена откатывает на T%.
До этого момента «незакрытый» пик — лишь кандидат, не сигнал.
После записи пивот никогда не изменяется — ПЕРЕРИСОВКИ НЕТ.
    """)

    lag_1d, gap_1d, inter_1d, gneg_1d = analyze("1d  ", T_1D,  h1d,  l1d)
    lag_10m, gap_10m, inter_10m, gneg_10m = analyze("10m ", T_10M, h10m, l10m)

    print(f"\n{'─'*55}")
    print("ПРАКТИЧЕСКИЙ ВЫВОД:")
    print(f"  1d (T=4%): сигнал приходит через median={np.median(lag_1d):.0f} бар")
    print(f"    после реального пика/впадины.")
    print(f"    Следующий пик — ещё через median={np.median(gap_1d):.0f} баров.")
    print(f"    В {gneg_1d:.0f}% случаев следующий пик уже ПОЗАДИ к моменту сигнала")
    print(f"    (т.е. мы «опаздываем» на этот ход и прогнозируем уже следующий).")

    # ── Графики ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row, (label, lag, gap, inter, thr) in enumerate([
        ("1d T=4%",   lag_1d,  gap_1d,  inter_1d,  T_1D),
        ("10m T=0.4%", lag_10m, gap_10m, inter_10m, T_10M),
    ]):
        # Lag
        ax = axes[row, 0]
        bins = range(0, min(int(lag.max()) + 2, 60))
        ax.hist(lag, bins=bins, color="steelblue", alpha=0.8, edgecolor="white")
        ax.axvline(np.median(lag), color="red", lw=1.5, ls="--",
                   label=f"median={np.median(lag):.0f}")
        ax.set_xlabel("баров (пик → подтверждение)")
        ax.set_ylabel("count"); ax.set_title(f"{label}: lag")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Gap
        ax = axes[row, 1]
        g_clip = np.clip(gap, -20, 80)
        ax.hist(g_clip, bins=40, color="darkorange", alpha=0.8, edgecolor="white")
        ax.axvline(0, color="black", lw=1.0, ls=":")
        ax.axvline(np.median(gap), color="red", lw=1.5, ls="--",
                   label=f"median={np.median(gap):.0f}")
        ax.set_xlabel("баров (подтверждение → следующий пик)")
        ax.set_title(f"{label}: gap\n(отриц. = пик уже позади)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Inter
        ax = axes[row, 2]
        i_clip = np.clip(inter, 0, 80)
        ax.hist(i_clip, bins=40, color="mediumseagreen", alpha=0.8, edgecolor="white")
        ax.axvline(np.median(inter), color="red", lw=1.5, ls="--",
                   label=f"median={np.median(inter):.0f}")
        ax.set_xlabel("баров (сигнал → следующий сигнал)")
        ax.set_title(f"{label}: интервал между сигналами")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Задержка зигзага SBER: lag / gap / inter", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "zigzag_lag.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {OUT}/zigzag_lag.png")


if __name__ == "__main__":
    main()
