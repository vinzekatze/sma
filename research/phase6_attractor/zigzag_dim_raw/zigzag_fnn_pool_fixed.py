#!/usr/bin/env python3
"""
FNN для объединённого пула 1d+10m — исправленная версия.

КОНТРАКТ КАУЗАЛЬНОСТИ:
  FNN — диагностика структуры аттрактора, не прогноз.
  Используем все данные (нет walk-forward), без look-ahead bias.

ИСПРАВЛЕНИЕ:
  Предыдущий код делал split по чётности индекса (z[0::2] / z[1::2]),
  что смешивало H и L пивоты случайным образом (первый пивот может быть
  как 'H', так и 'L' — зависит от данных).
  Исправление: разбиение строго по меткам 'H'/'L' из find_pivots_hl.

Объединение 1d и 10m: пивоты сортируются по дате — единый временной ряд.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT      = BASE_DIR / "results"

T_PRIM  = 0.04    # 1d, % порог зигзага
T_10M   = 0.004   # 10m
P_MAX   = 10
R_THR   = 15.0    # Kennel 1992: порог роста расстояния при увеличении p
A_THR   = 2.0     # порог абсолютного расстояния (шумовой пол)


def load_tf(tf):
    with open(DATA / f"{tf}.json") as f:
        data = json.load(f)
    h     = np.array([d["high"]  for d in data], dtype=np.float64)
    l     = np.array([d["low"]   for d in data], dtype=np.float64)
    dates = np.array([d["begin"] for d in data])
    return h, l, dates


def find_pivots_hl(highs, lows, dates, thr):
    """
    Зигзаг high-to-low: HIGH для пиков, LOW для впадин.
    Возвращает list of (price, date_str, 'H'/'L').
    """
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
                pivots.append((ext_val, dates[ext_idx], 'H'))
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                pivots.append((ext_val, dates[ext_idx], 'L'))
                direction = 1;  ext_val, ext_idx = highs[i], i
    return pivots


def get_dir_series(pivots_1d, pivots_10m, direction):
    """
    Объединить пивоты из обоих TF для заданного направления.
    Сортировать по дате → единый временной ряд.
    Возвращает массив цен.
    """
    pts  = [(date, price) for price, date, d in pivots_1d  if d == direction]
    pts += [(date, price) for price, date, d in pivots_10m if d == direction]
    pts.sort(key=lambda x: x[0])
    return np.array([p for _, p in pts])


def build_embedding(z, p):
    """
    X[i] = [z[i], log(z[i]/z[i-1]), log(z[i-1]/z[i-2]), ...]
    Размерность = p. Первые p-1 строк = NaN.
    """
    n  = len(z)
    lz = np.log(z)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = z[i]
        for lag in range(1, p):
            X[i, lag] = lz[i - lag + 1] - lz[i - lag]
    return X


def fnn_fraction(z, p, r_thr=R_THR, a_thr=A_THR):
    """Доля ложных ближайших соседей (Kennel 1992)."""
    n = len(z)
    if n < p + 2:
        return np.nan
    X_p  = build_embedding(z, p)
    X_p1 = build_embedding(z, p + 1)
    valid = np.where(~np.any(np.isnan(X_p1), axis=1))[0]
    if len(valid) < 5:
        return np.nan
    Xv   = X_p[valid]
    Xv1  = X_p1[valid]
    std_z = float(np.std(z))
    if std_z < 1e-12:
        return np.nan
    nn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(Xv)
    dists, idxs = nn.kneighbors(Xv)
    nn_idx = idxs[:, 1]
    d_p    = dists[:, 1]
    d_p1   = np.linalg.norm(Xv1 - Xv1[nn_idx], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d_p > 1e-12, d_p1 / d_p, np.inf)
    false_nn = (ratio > r_thr) | (d_p1 / std_z > a_thr)
    return float(false_nn.mean())


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")

    pivots_1d  = find_pivots_hl(h1d,  l1d,  d1d,  T_PRIM)
    pivots_10m = find_pivots_hl(h10m, l10m, d10m, T_10M)

    n_1d_up  = sum(1 for _, _, d in pivots_1d  if d == 'H')
    n_1d_dn  = sum(1 for _, _, d in pivots_1d  if d == 'L')
    n_10m_up = sum(1 for _, _, d in pivots_10m if d == 'H')
    n_10m_dn = sum(1 for _, _, d in pivots_10m if d == 'L')

    z_up = get_dir_series(pivots_1d, pivots_10m, 'H')
    z_dn = get_dir_series(pivots_1d, pivots_10m, 'L')

    print("=" * 55)
    print(f"SBER 1d  T={T_PRIM*100:.0f}%:  {len(pivots_1d):4d} пивотов  "
          f"(H={n_1d_up}, L={n_1d_dn})")
    print(f"SBER 10m T={T_10M*100:.1f}%: {len(pivots_10m):4d} пивотов  "
          f"(H={n_10m_up}, L={n_10m_dn})")
    print(f"Объединённый пул (sorted by date): H={len(z_up)}, L={len(z_dn)}")
    print("=" * 55)
    print()
    print(f"{'p':>4}  {'FNN_up (H)':>12}  {'FNN_dn (L)':>12}")
    print("─" * 35)

    fnn_up_list = []
    fnn_dn_list = []
    p_min_up = p_min_dn = None

    for p in range(1, P_MAX + 1):
        fu = fnn_fraction(z_up, p)
        fd = fnn_fraction(z_dn, p)
        fnn_up_list.append(fu)
        fnn_dn_list.append(fd)

        up_s = f"{fu:.4f}" if fu is not None and not np.isnan(fu) else "  nan"
        dn_s = f"{fd:.4f}" if fd is not None and not np.isnan(fd) else "  nan"
        mark = ""
        if fu == 0.0 and p_min_up is None:
            p_min_up = p; mark += " ← up=0"
        if fd == 0.0 and p_min_dn is None:
            p_min_dn = p; mark += " ← dn=0"
        print(f"  p={p:2d}  {up_s:>12}  {dn_s:>12}{mark}")

    print()
    print(f"p_min (FNN=0):  хаи = {p_min_up if p_min_up else f'>{P_MAX}'},"
          f"  лои = {p_min_dn if p_min_dn else f'>{P_MAX}'}")

    # ── График ────────────────────────────────────────────────────────────────
    ps = list(range(1, len(fnn_up_list) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(
        f"FNN — объединённый пул 1d(T=4%)+10m(T=0.4%) — SBER\n"
        f"Исправлено: split по меткам H/L  |  H={len(z_up)}, L={len(z_dn)} пивотов",
        fontsize=10
    )

    for ax, vals, label, color in [
        (axes[0], fnn_up_list, f"Хаи (H={len(z_up)})", "steelblue"),
        (axes[1], fnn_dn_list, f"Лои (L={len(z_dn)})", "darkorange"),
    ]:
        ys = [v if (v is not None and not np.isnan(v)) else np.nan for v in vals]
        ax.plot(ps, ys, "o-", color=color, lw=2, ms=7)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(label)
        ax.set_xlabel("p (размерность вложения)")
        ax.set_ylabel("Доля ложных соседей FNN")
        ax.set_xticks(ps)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    out_png = OUT / "fnn_pool_1d_10m_fixed.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"График: {out_png}")


if __name__ == "__main__":
    run()
