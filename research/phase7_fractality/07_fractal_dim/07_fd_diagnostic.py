#!/usr/bin/env python3
"""
07_fd_diagnostic.py — Higuchi FD на log(high) и log(low), расширяющееся окно

Вычисляем фрактальную размерность в каждой точке подтверждения T_BIG-пивота:
  - окно [0 .. confirm_bar] — строго каузально
  - отдельно на log(high) и log(low)
  - метод Higuchi 1988 (k_max=10 по умолчанию)

График: 3 панели — цена зигзага / FD(log_high) / FD(log_low) по времени.
"""
import json
import sys
import time
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

T_BIG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
K_MAX = int(sys.argv[2])   if len(sys.argv) > 2 else 10
MIN_BARS = K_MAX * 30      # минимум баров до первого вычисления FD


# ── данные ───────────────────────────────────────────────────────────────────
def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


# ── зигзаг ───────────────────────────────────────────────────────────────────
def find_pivots_log(highs, lows, dates, thr):
    """Возвращает (vals_log, pivot_dates, confirm_dates, dirs, confirm_bars)."""
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


# ── Higuchi FD ────────────────────────────────────────────────────────────────
def higuchi_fd(x, k_max):
    """
    Higuchi (1988) fractal dimension for 1D time series x.
    D ∈ [1.0, 2.0]: 1.0 = гладкий тренд, 1.5 = броуновское движение, 2.0 = максимальная шероховатость.

    Алгоритм:
      для каждого шага k и смещения m:
        L_m(k) = sum|x[m+ik] - x[m+(i-1)k]| * (N-1)/(n_m * k)
      L(k) = mean_m(L_m(k)) / k
      D = -slope( log(L(k)) vs log(k) )
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
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


# ── основной расчёт ───────────────────────────────────────────────────────────
def compute_fd_swing(series, confirm_bars, k_max, n_swings=2):
    """
    Скользящее окно длиной n_swings крупных движений.

    Для каждого пивота step (step >= n_swings):
        окно = series[ confirm_bars[step - n_swings] : confirm_bars[step] + 1 ]

    Длина окна в барах непостоянна — зависит от того, сколько баров заняли
    последние n_swings пивотов. При волатильном рынке окно короче,
    при спокойном — длиннее.
    """
    n = len(confirm_bars)
    fds = np.full(n, np.nan)
    win_sizes = np.full(n, np.nan)
    t0 = time.time()
    for step in range(n_swings, n):
        bar_from = int(confirm_bars[step - n_swings])
        bar_to   = int(confirm_bars[step])
        window   = series[bar_from : bar_to + 1]
        win_sizes[step] = len(window)
        if len(window) < k_max * 3:          # минимум данных для надёжной оценки
            continue
        fds[step] = higuchi_fd(window, k_max)
        if (step + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {step+1}/{n} пивотов за {elapsed:.1f}с  "
                  f"(win_size={len(window)} баров) ...", flush=True)
    return fds, win_sizes


# ── проверка на синтетических данных ─────────────────────────────────────────
def sanity_check(k_max):
    np.random.seed(42)
    n = 10_000
    bm   = np.cumsum(np.random.randn(n))   # BM → D ≈ 1.5
    trend = np.linspace(0, 1, n)            # тренд → D ≈ 1.0
    noise = np.random.randn(n)              # белый шум → D ≈ 2.0?
    d_bm    = higuchi_fd(bm,    k_max)
    d_trend = higuchi_fd(trend, k_max)
    d_noise = higuchi_fd(noise, k_max)
    print(f"  Sanity check (k_max={k_max}):")
    print(f"    Тренд    → D = {d_trend:.3f}  (ожидание ≈ 1.0)")
    print(f"    BM       → D = {d_bm:.3f}  (ожидание ≈ 1.5)")
    print(f"    Белый шум → D = {d_noise:.3f}  (ожидание ≈ 2.0)")


# ── визуализация ──────────────────────────────────────────────────────────────
def plot(confirm_dates, p_big, fd_high, fd_low, win_sizes, out_path, n_swings):
    parse = lambda s: datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    dates_dt = np.array([parse(s) for s in confirm_dates])

    fig, axes = plt.subplots(4, 1, figsize=(16, 12),
                              gridspec_kw={"height_ratios": [2, 1, 1, 0.8]},
                              sharex=False)

    ref_lines = [(1.0, "тренд", ":"), (1.5, "BM", "--"), (2.0, "шум", ":")]

    # ── панель 1: цена ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(dates_dt, p_big, color="steelblue", lw=0.8, label="T_BIG зигзаг")
    ax.set_ylabel("Цена SBER (руб.)")
    ax.set_title(
        f"SBER 10m | T_big={T_BIG*100:.1f}% | Higuchi FD (k_max={K_MAX})\n"
        f"Скользящее окно = последние {n_swings} крупных движения"
    )
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── панель 2: FD(log_high) ────────────────────────────────────────────────
    ax = axes[1]
    valid = np.isfinite(fd_high)
    ax.plot(dates_dt[valid], fd_high[valid], color="tomato", lw=0.9,
            label="Higuchi D  [log(high)]")
    for y, lbl, ls in ref_lines:
        ax.axhline(y, color="gray", lw=0.6, ls=ls)
        ax.text(dates_dt[valid][0], y + 0.03, lbl, fontsize=7, color="gray")
    ax.set_ylabel("D")
    ax.set_ylim(0.8, 2.2)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── панель 3: FD(log_low) ─────────────────────────────────────────────────
    ax = axes[2]
    valid = np.isfinite(fd_low)
    ax.plot(dates_dt[valid], fd_low[valid], color="darkorange", lw=0.9,
            label="Higuchi D  [log(low)]")
    for y, lbl, ls in ref_lines:
        ax.axhline(y, color="gray", lw=0.6, ls=ls)
        ax.text(dates_dt[valid][0], y + 0.03, lbl, fontsize=7, color="gray")
    ax.set_ylabel("D")
    ax.set_ylim(0.8, 2.2)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # ── панель 4: размер окна в барах ────────────────────────────────────────
    ax = axes[3]
    valid = np.isfinite(win_sizes)
    ax.plot(dates_dt[valid], win_sizes[valid], color="slategray", lw=0.7,
            label="окно (баров)")
    ax.set_ylabel("баров")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.tight_layout()
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"T_BIG={T_BIG*100:.1f}%  K_MAX={K_MAX}  MIN_BARS={MIN_BARS}")
    print()

    # синтетическая проверка
    sanity_check(K_MAX)
    print()

    h, l, d = load_tf("10m")
    lh, ll  = np.log(h), np.log(l)

    lp_big, _, confirm_dates, _, confirm_bars = find_pivots_log(h, l, d, T_BIG)
    p_big = np.exp(lp_big)
    n_big = len(lp_big)
    print(f"T_BIG пивотов: {n_big}  (баров всего: {len(h)})")
    print()

    N_SWINGS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    print(f"Скользящее окно = {N_SWINGS} крупных движения")
    print()

    print("Вычисляем FD(log_high) ...")
    t0 = time.time()
    fd_high, win_h = compute_fd_swing(lh, confirm_bars, K_MAX, N_SWINGS)
    print(f"  готово за {time.time()-t0:.1f}с")

    print("Вычисляем FD(log_low) ...")
    t0 = time.time()
    fd_low, win_l = compute_fd_swing(ll, confirm_bars, K_MAX, N_SWINGS)
    print(f"  готово за {time.time()-t0:.1f}с")

    # статистика
    for name, fd, ws in [("log(high)", fd_high, win_h), ("log(low)", fd_low, win_l)]:
        v  = fd[np.isfinite(fd)]
        wv = ws[np.isfinite(ws)]
        print(f"\n{name}:  D mean={v.mean():.3f}  std={v.std():.3f}  "
              f"min={v.min():.3f}  max={v.max():.3f}  median={np.median(v):.3f}")
        print(f"  окно (баров): mean={wv.mean():.0f}  min={wv.min():.0f}  "
              f"max={wv.max():.0f}  median={np.median(wv):.0f}")

    # сохранение CSV
    import csv
    csv_path = RESULTS / f"fd_swing{N_SWINGS}_T{int(T_BIG*100)}_k{K_MAX}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pivot_idx", "confirm_date", "confirm_bar",
                    "lp_big", "fd_high", "fd_low", "win_bars"])
        for i in range(n_big):
            w.writerow([i, confirm_dates[i], int(confirm_bars[i]),
                        lp_big[i], fd_high[i], fd_low[i], win_h[i]])
    print(f"\nResults → {csv_path}")

    # график
    fig_path = RESULTS / f"fd_swing{N_SWINGS}_T{int(T_BIG*100)}_k{K_MAX}.png"
    plot(confirm_dates, p_big, fd_high, fd_low, win_h, fig_path, N_SWINGS)


if __name__ == "__main__":
    main()
