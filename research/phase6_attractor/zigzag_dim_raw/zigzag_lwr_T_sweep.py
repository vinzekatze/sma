#!/usr/bin/env python3
"""
LWR прогноз зигзага: свип по T.

Зигзаг: high-to-low (HIGH для пиков, LOW для впадин), только 1d SBER.
Вектор задержек: [price[i], log(price[i]/price[i-1]), log(price[i-1]/price[i-2])]
p=3 (FNN p_min=3), k=3*(p+1)=12, адаптивная полоса ξ = dist до k-го соседа.
Пул: все пивоты j < step (без TRAIN_WIN — LWR сам взвешивает по близости).
Цель: найти минимум rMAE по T.

Метрика: rMAE = mean|pred - actual| / mean|Δprice|
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER" / "1d.json"
OUT      = BASE_DIR / "results"

P   = 3                    # embedding dim (из FNN)
K   = 3 * (P + 1)         # = 12, минимальный пул для устойчивого LWR
H   = 1                    # горизонт: следующий пивот
MIN_HISTORY = 50           # минимум пивотов до начала теста

T_GRID = [0.010, 0.012, 0.015, 0.018, 0.020, 0.022, 0.025, 0.028,
          0.030, 0.035, 0.040, 0.045, 0.050, 0.060, 0.080, 0.100]


# ── Данные ────────────────────────────────────────────────────────────────────

def load_1d():
    with open(DATA) as f:
        data = json.load(f)
    highs = np.array([d["high"] for d in data], dtype=np.float64)
    lows  = np.array([d["low"]  for d in data], dtype=np.float64)
    return highs, lows


# ── Зигзаг high-to-low ────────────────────────────────────────────────────────

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


# ── Вектор задержек ───────────────────────────────────────────────────────────

def build_X(prices, p):
    """
    X[i] = [price[i], log(price[i]/price[i-1]), ..., log(price[i-p+2]/price[i-p+1])]
    Первые p-1 строк = NaN.
    """
    n  = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


# ── LWR ───────────────────────────────────────────────────────────────────────

def lwr_pred(X_pool, y_pool, x_q, k):
    """
    LWR: k ближайших соседей, адаптивная Гауссова полоса ξ = dist до k-го.
    Возвращает предсказание или np.nan.
    """
    if len(X_pool) < k:
        return np.nan

    # Нормировка пула для устойчивости расстояний
    mu    = X_pool.mean(0)
    sigma = X_pool.std(0)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)
    Xn    = (X_pool - mu) / sigma
    xn    = (x_q    - mu) / sigma

    dists  = np.linalg.norm(Xn - xn, axis=1)
    order  = np.argsort(dists)
    knn    = order[:k]
    xi     = dists[order[k - 1]]          # расстояние до k-го соседа
    if xi < 1e-12:
        return float(y_pool[knn].mean())

    w  = np.exp(-0.5 * (dists[knn] / xi) ** 2)
    ws = np.sqrt(w)
    Xk = Xn[knn]
    yk = y_pool[knn]
    A  = np.column_stack([np.ones(k), Xk]) * ws[:, None]
    b  = yk * ws
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0] + coef[1:] @ xn)


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(prices, p, k, h, min_history):
    """
    Walk-forward LWR прогноз.
    Возвращает (preds, actuals) массивы по тестовым шагам.
    """
    X      = build_X(prices, p)
    n      = len(prices)
    preds, actuals = [], []

    test_start = max(min_history, p)
    for step in range(test_start, n - h):
        if np.any(np.isnan(X[step])):
            continue
        target = prices[step + h]

        # Пул: все j < step с валидными признаками и доступной целевой
        pool_idx = np.arange(p - 1, step)              # строго j < step
        pool_idx = pool_idx[~np.any(np.isnan(X[pool_idx]), axis=1)]
        pool_idx = pool_idx[pool_idx + h < n]
        if len(pool_idx) < k:
            continue
        X_pool = X[pool_idx]
        y_pool = prices[pool_idx + h]

        pred = lwr_pred(X_pool, y_pool, X[step], k)
        if np.isnan(pred):
            continue
        preds.append(pred)
        actuals.append(target)

    return np.array(preds), np.array(actuals)


# ── Метрики ───────────────────────────────────────────────────────────────────

def rmae(preds, actuals):
    errs     = np.abs(preds - actuals)
    mean_dz  = np.mean(np.abs(np.diff(actuals)))
    if mean_dz < 1e-12:
        return np.nan
    return float(errs.mean() / mean_dz)

def rmae_m0(prices_full, actuals_start_idx, actuals):
    """M0: предсказываем текущую цену как следующую."""
    # actuals_start_idx — индекс первого actual в prices_full
    m0_preds = actuals[:-1]   # current → predict next
    m0_acts  = actuals[1:]
    mean_dz  = np.mean(np.abs(np.diff(actuals)))
    if mean_dz < 1e-12 or len(m0_preds) == 0:
        return np.nan
    return float(np.abs(m0_preds - m0_acts).mean() / mean_dz)


# ── Главная функция ───────────────────────────────────────────────────────────

def run():
    highs, lows = load_1d()
    print(f"SBER 1d: {len(highs)} баров")
    print(f"p={P}  k={K}  H={H}\n")
    print(f"{'T':>6}  {'n_piv':>6}  {'n_test':>7}  {'rMAE_M0':>8}  {'rMAE_LWR':>9}  {'vs_M0':>7}")
    print("─" * 56)

    records = []

    for T in T_GRID:
        pivots = find_pivots_hl(highs, lows, T)
        if len(pivots) < MIN_HISTORY + P + H + K:
            print(f"  T={T*100:.1f}%: мало пивотов ({len(pivots)}), пропуск")
            continue

        prices = np.array([pr for _, pr, _ in pivots])
        n      = len(prices)

        preds, actuals = walk_forward(prices, P, K, H, MIN_HISTORY)
        if len(preds) < 10:
            continue

        r_lwr  = rmae(preds, actuals)
        mean_dz = np.mean(np.abs(np.diff(actuals)))
        r_m0   = float(np.abs(prices[MIN_HISTORY: MIN_HISTORY + len(actuals)] - actuals).mean() / mean_dz) \
                 if mean_dz > 1e-12 else np.nan

        # Проще: M0 = predict current pivot = no-change
        # actual[i] = prices[step+1], m0[i] = prices[step]
        # Используем preds как эталон шагов, восстановим M0 иначе
        m0_arr = prices[MIN_HISTORY: MIN_HISTORY + len(actuals)]
        r_m0   = float(np.abs(m0_arr - actuals).mean() / mean_dz) if mean_dz > 1e-12 else np.nan

        vs_m0  = (r_lwr / r_m0 - 1) * 100 if r_m0 and r_m0 > 0 else np.nan

        print(f"  T={T*100:4.1f}%  {n:6d}  {len(preds):7d}  {r_m0:8.4f}  {r_lwr:9.4f}  {vs_m0:+6.1f}%")
        records.append({"T": T, "n_piv": n, "n_test": len(preds),
                         "rMAE_M0": r_m0, "rMAE_LWR": r_lwr, "vs_M0_pct": vs_m0})

    df = pd.DataFrame(records)
    df.to_csv(OUT / "lwr_T_sweep.csv", index=False)

    # ── График ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["T"] * 100, df["rMAE_LWR"], "o-", color="steelblue",
            lw=2, ms=7, label="LWR (p=3, k=12)")
    ax.plot(df["T"] * 100, df["rMAE_M0"],  "s--", color="gray",
            lw=1.5, ms=6, alpha=0.7, label="M0 (random walk)")
    if df["rMAE_LWR"].notna().any():
        best_T = df.loc[df["rMAE_LWR"].idxmin(), "T"]
        best_r = df["rMAE_LWR"].min()
        ax.axvline(best_T * 100, color="steelblue", lw=1, ls=":", alpha=0.6)
        ax.annotate(f"min T={best_T*100:.1f}%\nrMAE={best_r:.4f}",
                    xy=(best_T * 100, best_r),
                    xytext=(best_T * 100 + 0.5, best_r + 0.02),
                    fontsize=8, color="steelblue")
    ax.set_xlabel("T, %  (порог зигзага)")
    ax.set_ylabel("rMAE")
    ax.set_title("LWR прогноз зигзага vs T\n"
                 "SBER 1d, high/low, [price, log-ret, log-ret], p=3, k=12")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "lwr_T_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nCSV и график: {OUT}")
    if len(df):
        best = df.loc[df["rMAE_LWR"].idxmin()]
        print(f"Лучший T={best['T']*100:.1f}%  rMAE={best['rMAE_LWR']:.4f}  vs M0={best['vs_M0_pct']:+.1f}%")


if __name__ == "__main__":
    run()
