#!/usr/bin/env python3
"""
Свип по T вспомогательных рядов (1h, 10m) при фиксированном первичном 1d.

Структура:
  Часть 1 — свип T_1h,  пул = 1d + 1h
  Часть 2 — свип T_10m, пул = 1d + 10m
  Часть 3 — контроль: только 1d | +1h лучший | +10m лучший | 1d+1h+10m лучшие

Первичный ряд: SBER 1d, T_prim ∈ {2%, 4%}.
Вспомогательные ряды: те же признаки p=3, те же параметры LWR.
LWR: p=3, k=12, адаптивная полоса ξ = dist до k-го соседа.
Каузальность: aux-пул ограничен строго до даты текущего первичного пивота.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT      = BASE_DIR / "results"

P   = 3
K   = 3 * (P + 1)   # 12
H   = 1
MIN_HISTORY = 50

T_PRIM_LIST = [0.02, 0.04]

T_1H_GRID  = [0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.015,
              0.018, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050]

T_10M_GRID = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010,
              0.012, 0.015, 0.018, 0.020, 0.025, 0.030]


# ── Загрузка ──────────────────────────────────────────────────────────────────

def load_tf(tf):
    with open(DATA / f"{tf}.json") as f:
        data = json.load(f)
    h     = np.array([d["high"]  for d in data], dtype=np.float64)
    l     = np.array([d["low"]   for d in data], dtype=np.float64)
    dates = np.array([d["begin"] for d in data])
    return h, l, dates


# ── Зигзаг high-to-low ────────────────────────────────────────────────────────

def find_pivots_hl(highs, lows, dates, thr):
    pivots = []
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
                pivots.append((ext_idx, ext_val, dates[ext_idx]))
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                pivots.append((ext_idx, ext_val, dates[ext_idx]))
                direction = 1;  ext_val, ext_idx = highs[i], i
    if not pivots:
        return np.array([]), np.array([])
    prices = np.array([p for _, p, _ in pivots])
    pdates = np.array([d for _, _, d in pivots])
    return prices, pdates


# ── Признаки ─────────────────────────────────────────────────────────────────

def build_X(prices, p):
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
    if len(X_pool) < k:
        return np.nan
    mu    = X_pool.mean(0)
    sigma = X_pool.std(0)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)
    Xn    = (X_pool - mu) / sigma
    xn    = (x_q    - mu) / sigma
    dists = np.linalg.norm(Xn - xn, axis=1)
    order = np.argsort(dists)
    knn   = order[:k]
    xi    = dists[order[k - 1]]
    if xi < 1e-12:
        return float(y_pool[knn].mean())
    w   = np.exp(-0.5 * (dists[knn] / xi) ** 2)
    ws  = np.sqrt(w)
    A   = np.column_stack([np.ones(k), Xn[knn]]) * ws[:, None]
    b   = y_pool[knn] * ws
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0] + coef[1:] @ xn)


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(prices_prim, dates_prim, X_prim,
                 aux_list,   # [(prices_a, dates_a, X_a), ...]
                 p, k, h, min_history):
    """
    aux_list: список вспомогательных рядов уже с построенными признаками.
    Каузальность aux: строго до даты текущего шага (searchsorted side=left).
    """
    n = len(prices_prim)
    preds, actuals = [], []

    for step in range(max(min_history, p), n - h):
        if np.any(np.isnan(X_prim[step])):
            continue
        target   = prices_prim[step + h]
        cur_date = dates_prim[step]

        # Первичный пул: j ∈ [p-1, step-1]
        j_arr = np.arange(p - 1, step)
        valid = ~np.any(np.isnan(X_prim[j_arr]), axis=1) & (j_arr + h < n)
        X_rows = list(X_prim[j_arr[valid]])
        y_rows = list(prices_prim[j_arr[valid] + h])

        # Вспомогательные пулы
        for prices_a, dates_a, X_a in aux_list:
            ce = int(np.searchsorted(dates_a, cur_date, side="left"))
            if ce < p:
                continue
            na = len(prices_a)
            j_a = np.arange(p - 1, min(ce, na - h))
            if len(j_a) == 0:
                continue
            valid_a = ~np.any(np.isnan(X_a[j_a]), axis=1) & (j_a + h < na)
            X_rows.extend(X_a[j_a[valid_a]])
            y_rows.extend(prices_a[j_a[valid_a] + h])

        if len(X_rows) < k:
            continue

        pred = lwr_pred(np.array(X_rows), np.array(y_rows), X_prim[step], k)
        if np.isnan(pred):
            continue
        preds.append(pred)
        actuals.append(target)

    return np.array(preds), np.array(actuals)


def compute_rmae(preds, actuals):
    if len(preds) < 5:
        return np.nan
    mean_dz = np.mean(np.abs(np.diff(actuals)))
    if mean_dz < 1e-12:
        return np.nan
    return float(np.mean(np.abs(preds - actuals)) / mean_dz)


# ── Основной прогон ───────────────────────────────────────────────────────────

def run():
    h1d_raw, l1d_raw, dates1d_raw = load_tf("1d")
    h1h_raw, l1h_raw, dates1h_raw = load_tf("1h")
    h10m_raw, l10m_raw, dates10m_raw = load_tf("10m")

    print(f"SBER: 1d={len(h1d_raw)} 1h={len(h1h_raw)} 10m={len(h10m_raw)} баров\n")

    all_records = []

    for T_prim in T_PRIM_LIST:
        prices_prim, dates_prim = find_pivots_hl(h1d_raw, l1d_raw, dates1d_raw, T_prim)
        if len(prices_prim) < MIN_HISTORY + P + H:
            print(f"T_prim={T_prim*100:.0f}%: мало пивотов, пропуск")
            continue
        X_prim = build_X(prices_prim, P)
        print(f"=== T_prim={T_prim*100:.0f}%  ({len(prices_prim)} пивотов) ===")

        # Базовый (только 1d)
        preds0, acts0 = walk_forward(prices_prim, dates_prim, X_prim, [], P, K, H, MIN_HISTORY)
        r_base = compute_rmae(preds0, acts0)
        mean_dz = np.mean(np.abs(np.diff(acts0))) if len(acts0) > 1 else 1.0
        print(f"  Только 1d:  rMAE={r_base:.4f}  n={len(preds0)}")

        # ── Часть 1: свип T_1h ────────────────────────────────────────────────
        print(f"\n  Свип T_1h (пул 1d+1h):")
        print(f"  {'T_1h':>6}  {'n_piv_1h':>9}  {'n_test':>7}  {'rMAE':>7}  {'vs_base':>8}")
        print("  " + "─" * 46)
        best_1h_T, best_1h_r = None, r_base

        for T_1h in T_1H_GRID:
            prices_1h, dates_1h = find_pivots_hl(h1h_raw, l1h_raw, dates1h_raw, T_1h)
            if len(prices_1h) < P + H + 1:
                continue
            X_1h = build_X(prices_1h, P)
            aux = [(prices_1h, dates_1h, X_1h)]
            preds, acts = walk_forward(prices_prim, dates_prim, X_prim,
                                       aux, P, K, H, MIN_HISTORY)
            r = compute_rmae(preds, acts)
            vs = (r / r_base - 1) * 100 if r_base else np.nan
            marker = " ←" if (r < best_1h_r) else ""
            print(f"  T_1h={T_1h*100:4.1f}%  {len(prices_1h):9d}  {len(preds):7d}  {r:.4f}  {vs:+7.1f}%{marker}")
            if r < best_1h_r:
                best_1h_r, best_1h_T = r, T_1h
            all_records.append({"T_prim": T_prim, "aux": "1h", "T_aux": T_1h,
                                 "n_piv_aux": len(prices_1h), "n_test": len(preds),
                                 "rMAE": r, "vs_base_pct": vs})

        print(f"\n  Лучший T_1h={best_1h_T*100:.1f}%  rMAE={best_1h_r:.4f}")

        # ── Часть 2: свип T_10m ───────────────────────────────────────────────
        print(f"\n  Свип T_10m (пул 1d+10m):")
        print(f"  {'T_10m':>6}  {'n_piv_10m':>10}  {'n_test':>7}  {'rMAE':>7}  {'vs_base':>8}")
        print("  " + "─" * 48)
        best_10m_T, best_10m_r = None, r_base

        for T_10m in T_10M_GRID:
            prices_10m, dates_10m = find_pivots_hl(h10m_raw, l10m_raw, dates10m_raw, T_10m)
            if len(prices_10m) < P + H + 1:
                continue
            X_10m = build_X(prices_10m, P)
            aux = [(prices_10m, dates_10m, X_10m)]
            preds, acts = walk_forward(prices_prim, dates_prim, X_prim,
                                       aux, P, K, H, MIN_HISTORY)
            r = compute_rmae(preds, acts)
            vs = (r / r_base - 1) * 100 if r_base else np.nan
            marker = " ←" if (r < best_10m_r) else ""
            print(f"  T_10m={T_10m*100:4.1f}%  {len(prices_10m):10d}  {len(preds):7d}  {r:.4f}  {vs:+7.1f}%{marker}")
            if r < best_10m_r:
                best_10m_r, best_10m_T = r, T_10m
            all_records.append({"T_prim": T_prim, "aux": "10m", "T_aux": T_10m,
                                 "n_piv_aux": len(prices_10m), "n_test": len(preds),
                                 "rMAE": r, "vs_base_pct": vs})

        print(f"\n  Лучший T_10m={best_10m_T*100:.1f}%  rMAE={best_10m_r:.4f}")

        # ── Часть 3: контроль ─────────────────────────────────────────────────
        print(f"\n  Контроль (лучшие T_aux):")
        combos = []
        if best_1h_T:
            p1h, d1h = find_pivots_hl(h1h_raw, l1h_raw, dates1h_raw, best_1h_T)
            X1h = build_X(p1h, P)
            combos.append(("1h_best", [(p1h, d1h, X1h)]))
        if best_10m_T:
            p10m, d10m = find_pivots_hl(h10m_raw, l10m_raw, dates10m_raw, best_10m_T)
            X10m = build_X(p10m, P)
            combos.append(("10m_best", [(p10m, d10m, X10m)]))
        if best_1h_T and best_10m_T:
            combos.append(("1h+10m_best", [(p1h, d1h, X1h), (p10m, d10m, X10m)]))

        for label, aux in combos:
            preds, acts = walk_forward(prices_prim, dates_prim, X_prim,
                                       aux, P, K, H, MIN_HISTORY)
            r = compute_rmae(preds, acts)
            vs = (r / r_base - 1) * 100 if r_base else np.nan
            print(f"    {label:<18}  rMAE={r:.4f}  vs_base={vs:+.1f}%")
            all_records.append({"T_prim": T_prim, "aux": label, "T_aux": np.nan,
                                 "n_piv_aux": np.nan, "n_test": len(preds),
                                 "rMAE": r, "vs_base_pct": vs})

        print()

    # ── CSV ───────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    df.to_csv(OUT / "aux_T_sweep.csv", index=False)

    # ── Графики ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Свип T вспомогательных рядов: rMAE на 1d-прогнозе\n"
                 "SBER, LWR p=3 k=12, high/low зигзаг", fontsize=11)

    for row, T_prim in enumerate(T_PRIM_LIST):
        sub = df[df["T_prim"] == T_prim]
        r_base_val = sub[sub["aux"] == "1h"]["rMAE"].iloc[0] if False else None
        # base из записей
        base_rows = [r for r in all_records
                     if r["T_prim"] == T_prim and r["aux"] == "1h"]
        # базовый rMAE берём из отдельного расчёта — он не записан в df, пересчитаем через vs_base_pct
        # Проще: rMAE_base = rMAE / (1 + vs_base_pct/100)
        if len(sub[sub["aux"] == "1h"]):
            row0 = sub[sub["aux"] == "1h"].iloc[0]
            r_base_val = row0["rMAE"] / (1 + row0["vs_base_pct"] / 100)

        for col, aux_name in enumerate(["1h", "10m"]):
            ax = axes[row][col]
            sub_aux = sub[sub["aux"] == aux_name].sort_values("T_aux")
            if len(sub_aux) == 0:
                continue
            ax.plot(sub_aux["T_aux"] * 100, sub_aux["rMAE"],
                    "o-", color="steelblue", lw=2, ms=6)
            if r_base_val:
                ax.axhline(r_base_val, color="gray", lw=1.2, ls="--",
                           label=f"только 1d ({r_base_val:.4f})")
            if len(sub_aux):
                best = sub_aux.loc[sub_aux["rMAE"].idxmin()]
                ax.axvline(best["T_aux"] * 100, color="steelblue",
                           lw=1, ls=":", alpha=0.6)
                ax.annotate(f"min T={best['T_aux']*100:.1f}%",
                            xy=(best["T_aux"] * 100, best["rMAE"]),
                            xytext=(best["T_aux"] * 100 + 0.1, best["rMAE"] + 0.003),
                            fontsize=7.5, color="steelblue")
            ax.set_title(f"T_prim={T_prim*100:.0f}%, aux={aux_name}")
            ax.set_xlabel("T_aux, %")
            ax.set_ylabel("rMAE")
            ax.legend(fontsize=7.5)
            ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(OUT / "aux_T_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"CSV и график: {OUT}")


if __name__ == "__main__":
    run()
