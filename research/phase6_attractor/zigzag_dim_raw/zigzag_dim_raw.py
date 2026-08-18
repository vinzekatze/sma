#!/usr/bin/env python3
"""
FNN-диагностика зигзага на чистых ценах.

Зигзаг: high-to-low (HIGH для пиков, LOW для впадин), только 1d.
Вектор задержек: [price[i], log(price[i]/price[i-1]), log(price[i-1]/price[i-2]), ...]
Свип по T. FNN считается отдельно для восходящих (хаи) и нисходящих (лои) пивотов.
Ищем p при котором FNN=0 для каждого направления и каждого T.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER" / "1d.json"
OUT      = BASE_DIR / "results"

TICKER = "SBER"

T_GRID = [0.010, 0.012, 0.015, 0.018, 0.020, 0.022, 0.025, 0.028,
          0.030, 0.035, 0.040, 0.045, 0.050, 0.060, 0.080, 0.100]

P_MAX  = 16
R_THR  = 15.0   # Kennel 1992: порог роста расстояния
A_THR  = 2.0    # порог на абсолютное расстояние (шумовой пол)


# ── Загрузка ──────────────────────────────────────────────────────────────────

def load_1d():
    with open(DATA) as f:
        data = json.load(f)
    highs = np.array([d["high"]  for d in data], dtype=np.float64)
    lows  = np.array([d["low"]   for d in data], dtype=np.float64)
    return highs, lows


# ── Зигзаг high-to-low ────────────────────────────────────────────────────────

def find_pivots_hl(highs, lows, thr):
    """
    Торговый зигзаг: HIGH для пиков, LOW для впадин.
    Возвращает список (bar_idx, price, direction) где direction = 'H' или 'L'.
    Первый пивот добавляется при первом подтверждённом развороте.
    """
    pivots    = []
    direction = 0       # 0=инициализация, 1=ищем новый хай, -1=ищем новый лой
    ext_val   = (highs[0] + lows[0]) / 2.0
    ext_idx   = 0

    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1
                ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1
                ext_val, ext_idx = lows[i], i

        elif direction == 1:
            if highs[i] > ext_val:
                ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                pivots.append((ext_idx, ext_val, 'H'))
                direction = -1
                ext_val, ext_idx = lows[i], i

        else:  # direction == -1
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                pivots.append((ext_idx, ext_val, 'L'))
                direction = 1
                ext_val, ext_idx = highs[i], i

    return pivots  # [(bar_idx, price, 'H'/'L'), ...]


# ── Матрица вложения ──────────────────────────────────────────────────────────

def build_embedding(z, p):
    """
    z: одномерная серия пивотов (только хаи или только лои).
    X[i] = [z[i], log(z[i]/z[i-1]), log(z[i-1]/z[i-2]), ..., log(z[i-p+2]/z[i-p+1])]
    Размерность = p. Первые p-1 строк = NaN.
    """
    n   = len(z)
    X   = np.full((n, p), np.nan)
    lz  = np.log(z)
    for i in range(p - 1, n):
        X[i, 0] = z[i]
        for lag in range(1, p):
            X[i, lag] = lz[i - lag + 1] - lz[i - lag]   # log(z[i-lag+1]/z[i-lag])
    return X


# ── FNN ───────────────────────────────────────────────────────────────────────

def fnn_fraction(z, p, r_thr=R_THR, a_thr=A_THR):
    """
    Доля ложных ближайших соседей при вложении размерности p.
    Сравниваем расстояния в R^p и R^{p+1}.
    """
    n = len(z)
    if n < p + 2:
        return np.nan

    X_p  = build_embedding(z, p)
    X_p1 = build_embedding(z, p + 1)

    # Убираем NaN-строки (нужны строки с валидным X_p И X_p1, т.е. i >= p)
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
    nn_idx = idxs[:, 1]       # ближайший сосед (не сам себя)
    d_p    = dists[:, 1]      # расстояние в p-мерном пространстве

    d_p1 = np.linalg.norm(Xv1 - Xv1[nn_idx], axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(d_p > 1e-12, d_p1 / d_p, np.inf)

    false_nn = (ratio > r_thr) | (d_p1 / std_z > a_thr)
    return float(false_nn.mean())


def p_min_zero(z, p_max=P_MAX):
    """
    Возвращает минимальный p при котором FNN=0 (и массив FNN по всем p).
    """
    fracs = []
    for p in range(1, p_max + 1):
        f = fnn_fraction(z, p)
        fracs.append(f)
        if f is not None and f == 0.0:
            break   # нашли, дальше обычно тоже 0
    # добить до p_max для графика
    while len(fracs) < p_max:
        fracs.append(fnn_fraction(z, len(fracs) + 1))
    return fracs


# ── Основной анализ ───────────────────────────────────────────────────────────

def run():
    highs, lows = load_1d()
    print(f"SBER 1d: {len(highs)} баров\n")
    print(f"{'T':>6}  {'n_piv':>6}  {'n_up':>5}  {'n_dn':>5}  {'p_min_up':>9}  {'p_min_dn':>9}")
    print("─" * 55)

    records   = []
    fnn_table = []   # для графика: per-(T, direction, p) значение FNN

    for T in T_GRID:
        pivots = find_pivots_hl(highs, lows, T)
        if len(pivots) < 20:
            print(f"  T={T*100:.1f}%: слишком мало пивотов ({len(pivots)}), пропускаю")
            continue

        prices = np.array([p for _, p, _ in pivots])
        dirs   = [d for _, _, d in pivots]

        up_z   = prices[[i for i, d in enumerate(dirs) if d == 'H']]
        down_z = prices[[i for i, d in enumerate(dirs) if d == 'L']]

        if len(up_z) < 10 or len(down_z) < 10:
            continue

        fnn_up   = p_min_zero(up_z,   P_MAX)
        fnn_down = p_min_zero(down_z, P_MAX)

        def first_zero(fracs):
            for i, f in enumerate(fracs):
                if f is not None and f == 0.0:
                    return i + 1
            return f">{P_MAX}"

        pm_up = first_zero(fnn_up)
        pm_dn = first_zero(fnn_down)

        print(f"  T={T*100:4.1f}%  {len(pivots):6d}  {len(up_z):5d}  {len(down_z):5d}  {str(pm_up):>9}  {str(pm_dn):>9}")

        records.append({
            "T": T, "n_piv": len(pivots), "n_up": len(up_z), "n_dn": len(down_z),
            "p_min_up": pm_up if isinstance(pm_up, int) else np.nan,
            "p_min_dn": pm_dn if isinstance(pm_dn, int) else np.nan,
        })

        for p_idx, (fu, fd) in enumerate(zip(fnn_up, fnn_down)):
            fnn_table.append({"T": T, "p": p_idx + 1, "fnn_up": fu, "fnn_dn": fd})

    df_sum = pd.DataFrame(records)
    df_fnn = pd.DataFrame(fnn_table)
    df_sum.to_csv(OUT / "summary.csv", index=False)
    df_fnn.to_csv(OUT / "fnn_table.csv", index=False)

    # ── График 1: FNN(p) для каждого T ───────────────────────────────────────
    T_plot = [T for T in T_GRID if T in df_sum["T"].values]
    n_T    = len(T_plot)
    cmap   = plt.cm.viridis(np.linspace(0, 1, n_T))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(f"FNN — SBER 1d, high/low зигзаг, вектор [price, log-returns]\n"
                 f"R_thr={R_THR}, A_thr={A_THR}", fontsize=11)

    for ax, col, title in zip(axes, ["fnn_up", "fnn_dn"], ["Хаи (up)", "Лои (down)"]):
        for j, T in enumerate(T_plot):
            sub = df_fnn[df_fnn["T"] == T].sort_values("p")
            ax.plot(sub["p"], sub[col], color=cmap[j], lw=1.5,
                    label=f"T={T*100:.1f}%")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(title)
        ax.set_xlabel("p (размерность вложения)")
        ax.set_ylabel("доля ложных соседей FNN")
        ax.set_xticks(range(1, P_MAX + 1))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=6.5, ncol=2, loc="upper right")

    plt.tight_layout()
    fig.savefig(OUT / "fnn_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── График 2: p_min(T) для хаёв и лоёв ──────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(df_sum["T"] * 100, df_sum["p_min_up"], "o-", color="steelblue",
             lw=2, ms=7, label="p_min хаи")
    ax2.plot(df_sum["T"] * 100, df_sum["p_min_dn"], "s--", color="darkorange",
             lw=2, ms=7, label="p_min лои")
    ax2.set_xlabel("T, %")
    ax2.set_ylabel("p_min (первый p с FNN=0)")
    ax2.set_title("Минимальное вложение p_min vs порог зигзага T\n"
                  "SBER 1d, high/low, вектор [price, log-returns]")
    ax2.legend()
    ax2.grid(alpha=0.2)
    fig2.savefig(OUT / "pmin_vs_T.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nCSV и графики: {OUT}")


if __name__ == "__main__":
    run()
