#!/usr/bin/env python3
"""
simplex_ref.py — Референс: Simplex прогноз на зигзаге (событийное время)

Схема
─────
1.  log(high), log(low) свечей тикера

2.  T_big=4%  →  зигзаг  →  lp_big[], conf_big[], dir_big[]
    dir = +1 (вершина/HIGH), −1 (впадина/LOW)
    Пивот подтверждается в баре conf_big[i], когда противоположная сторона
    превысила T_big.

3.  T_frac=3.6%  →  зигзаг  →  lp_f[], conf_f[], dir_f[]
    Фрактальный пул: более частые события того же масштабного порядка.
    Оптимум T_frac ≈ 0.9·T_big (свип скр.18).

4.  Вектор запроса (1D):
      xq = lp_big[i] − lp_big[i−1]
    Амплитуда последнего T_big плеча (знак кодирует направление).

5.  Вектор пула (1D):
      xp[j] = lp_f[j] − lp_f[j−1]

6.  Каузальная обрезка — строго до подтверждения i-го события:
      j_max = searchsorted(conf_f, conf_big[i], side='left')
    Ни один пул-элемент не знает о будущем относительно conf_big[i].

7.  Фильтр направления:
      J = { j < j_max : dir_f[j] == dir_big[i], lp_f[j+1] известен }
      Если |J| < MIN_POOL → J снимается, берём всё до j_max.

8.  K ближайших соседей по |xq − xp[j]|:
      d_j  = |xq − xp[j]|
      w_j  = exp(−d_j / d_1),  d_1 = min(d_j)   (Simplex-вес)
      y_rel = Σ w_j · (lp_f[j+1] − lp_f[j]) / Σ w_j

9.  Реконструкция цены:
      price_hat = exp(lp_big[i] + y_rel)

Результаты (SBER 10m, walk-forward 2013-2025, скр.18)
──────────────────────────────────────────────────────
  M0  (random walk)      rMAE = 1.000
  LWR (n_feats=2 K=75)   rMAE = 0.371  (−62.9% vs M0)
  Simplex (этот скрипт)  rMAE = 0.261  (−73.9% vs M0,  −29.8% vs LWR)
"""
import csv
import json
import sys
import numpy as np
from pathlib import Path

# ── параметры ─────────────────────────────────────────────────────────────────
T_BIG   = 0.04     # порог крупного зигзага
T_FRAC  = 0.036    # порог фрактального пула (≈ 0.9 · T_big)
K       = 12       # число соседей Simplex
H       = 1        # горизонт прогноза (событий)
MIN_HISTORY = 50   # минимум T_big событий до начала прогноза
MIN_POOL    = K + 2  # минимальный размер пула для dir-фильтра

DATA = Path(__file__).parent.parent.parent / "data" / "candles"


# ── загрузка свечей ───────────────────────────────────────────────────────────
def load_candles(ticker: str, interval: str) -> tuple:
    path = DATA / ticker / f"{interval}.json"
    with open(path) as f:
        raw = json.load(f)
    return (
        np.array([c["high"]  for c in raw], dtype=np.float64),
        np.array([c["low"]   for c in raw], dtype=np.float64),
        np.array([c["begin"] for c in raw]),
    )


# ── зигзаг ────────────────────────────────────────────────────────────────────
def zigzag(highs: np.ndarray, lows: np.ndarray, dates: np.ndarray,
           threshold: float) -> tuple:
    """
    Каузальный зигзаг по логарифмическому порогу.

    Возвращает (lp, conf_dates, dirs):
      lp[]         — log-значение пивота в момент его образования
      conf_dates[] — дата подтверждения (бар, где зафиксирован разворот)
      dirs[]       — +1 вершина (HIGH), −1 впадина (LOW)
    """
    lh, ll = np.log(highs), np.log(lows)
    lp, conf_dates, dirs = [], [], []
    direction = 0
    ext_val   = (lh[0] + ll[0]) / 2.0

    for i in range(len(highs)):
        if direction == 0:
            if lh[i] - ext_val >= threshold:
                direction, ext_val = 1, lh[i]
            elif ext_val - ll[i] >= threshold:
                direction, ext_val = -1, ll[i]
        elif direction == 1:
            if lh[i] > ext_val:
                ext_val = lh[i]
            elif ext_val - ll[i] >= threshold:
                lp.append(ext_val)
                conf_dates.append(dates[i])
                dirs.append(+1)
                direction, ext_val = -1, ll[i]
        else:
            if ll[i] < ext_val:
                ext_val = ll[i]
            elif lh[i] - ext_val >= threshold:
                lp.append(ext_val)
                conf_dates.append(dates[i])
                dirs.append(-1)
                direction, ext_val = 1, lh[i]

    return np.array(lp), np.array(conf_dates), np.array(dirs)


# ── каузальная обрезка пула ───────────────────────────────────────────────────
def cutoff_pool(confirm_date, dir_query: int,
                lp_f: np.ndarray, conf_f: np.ndarray, dir_f: np.ndarray) -> tuple:
    """
    Возвращает (xp, y_rel) для шага с датой подтверждения confirm_date.

    xp[j]    = lp_f[j] − lp_f[j−1]           (1D признак пула)
    y_rel[j] = lp_f[j+1] − lp_f[j]           (целевая лог-доходность)

    Каузальность: j_max = searchsorted(conf_f, confirm_date, 'left')
    → пул включает только события, подтверждённые строго раньше confirm_date.
    Фильтр направления: dir_f[j] == dir_query; откат на всё, если мало точек.
    """
    j_max = int(np.searchsorted(conf_f, confirm_date, side='left'))
    # j+H тоже должен быть подтверждён строго до confirm_date:
    #   conf_f[j+H] < confirm_date  →  j+H <= j_max-1  →  j <= j_max-H-1
    valid_range = np.arange(1, min(j_max - H, len(lp_f) - H))
    if len(valid_range) == 0:
        return None, None

    xp_all    = lp_f[valid_range]     - lp_f[valid_range - 1]
    y_rel_all = lp_f[valid_range + H] - lp_f[valid_range]

    finite = np.isfinite(xp_all) & np.isfinite(y_rel_all)
    xp_all, y_rel_all = xp_all[finite], y_rel_all[finite]
    dir_all = dir_f[valid_range[finite]]

    if len(xp_all) == 0:
        return None, None

    # направленный срез
    mask_dir = dir_all == dir_query
    if mask_dir.sum() >= MIN_POOL:
        return xp_all[mask_dir], y_rel_all[mask_dir]
    return xp_all, y_rel_all


# ── Simplex прогноз ───────────────────────────────────────────────────────────
def simplex_predict(xq: float, xp: np.ndarray, y_rel: np.ndarray, k: int) -> float:
    """
    1D Simplex projection.

    d_j  = |xq − xp[j]|
    w_j  = exp(−d_j / d_1),  d_1 = min(d_j)
    y_hat = Σ w_j · y_rel[j] / Σ w_j
    """
    dists = np.abs(xp - xq)
    k_eff = min(k, len(dists))
    if k_eff < 2:
        return np.nan

    nn  = np.argpartition(dists, k_eff - 1)[:k_eff]
    d   = dists[nn]
    d1  = d.min()

    if d1 < 1e-14:
        return float(y_rel[nn[np.argmin(d)]])

    w = np.exp(-d / d1)
    return float(w @ y_rel[nn] / w.sum())


# ── метрики ───────────────────────────────────────────────────────────────────
def rmae(errors: np.ndarray, actuals: np.ndarray) -> float:
    m  = np.isfinite(errors) & np.isfinite(actuals)
    dz = float(np.mean(np.abs(np.diff(actuals[m]))))
    return float(np.mean(np.abs(errors[m])) / dz) if dz > 1e-12 else np.nan


# ── walk-forward ──────────────────────────────────────────────────────────────
def walk_forward(lp_big: np.ndarray, conf_big: np.ndarray, dir_big: np.ndarray,
                 lp_f: np.ndarray, conf_f: np.ndarray, dir_f: np.ndarray,
                 k: int = K) -> list:
    """
    Walk-forward по всем T_big пивотам начиная с MIN_HISTORY.

    Возвращает список словарей с полями:
      step, confirm_date, direction, actual, predicted, error, pool_size
    """
    results = []
    for i in range(MIN_HISTORY, len(lp_big) - H):
        actual = float(np.exp(lp_big[i + H]))
        xq     = float(lp_big[i] - lp_big[i - 1])

        xp, y_rel = cutoff_pool(conf_big[i], int(dir_big[i]),
                                 lp_f, conf_f, dir_f)
        if xp is None:
            continue

        y_hat = simplex_predict(xq, xp, y_rel, k)
        if not np.isfinite(y_hat):
            continue

        predicted = float(np.exp(lp_big[i] + y_hat))
        results.append({
            "step":         i,
            "confirm_date": conf_big[i],
            "direction":    int(dir_big[i]),
            "actual":       actual,
            "predicted":    predicted,
            "error":        predicted - actual,
            "pool_size":    len(xp),
        })
    return results


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ticker   = sys.argv[1] if len(sys.argv) > 1 else "SBER"
    interval = sys.argv[2] if len(sys.argv) > 2 else "10m"
    t_big    = float(sys.argv[3]) if len(sys.argv) > 3 else T_BIG
    t_frac   = float(sys.argv[4]) if len(sys.argv) > 4 else T_FRAC

    print(f"{ticker} {interval}  T_big={t_big*100:.1f}%  T_frac={t_frac*100:.1f}%  K={K}")

    highs, lows, dates = load_candles(ticker, interval)

    lp_big, conf_big, dir_big = zigzag(highs, lows, dates, t_big)
    lp_f,   conf_f,   dir_f   = zigzag(highs, lows, dates, t_frac)
    print(f"T_big: {len(lp_big)} пивотов   T_frac: {len(lp_f)} пивотов")

    rows = walk_forward(lp_big, conf_big, dir_big, lp_f, conf_f, dir_f)
    print(f"Прогнозов: {len(rows)}")

    errors  = np.array([r["error"]  for r in rows])
    actuals = np.array([r["actual"] for r in rows])
    r_mae   = rmae(errors, actuals)
    print(f"rMAE = {r_mae:.4f}")

    out = Path(__file__).parent / f"simplex_{ticker}_{interval}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"CSV  → {out}")


if __name__ == "__main__":
    main()
