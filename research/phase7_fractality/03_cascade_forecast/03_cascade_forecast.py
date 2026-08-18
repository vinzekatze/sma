#!/usr/bin/env python3
"""
03_cascade_forecast.py — Каскадный прогноз: 2 уровня фрактальности

Алгоритм:
  Ур.1 (как B1): X_big[step] → K1 соседей в T_frac1 пуле
                  метка p_f1[j+1] → предсказание p_big[step+1]

  Ур.2 (cascade sub-prediction для каждого соседа j):
        Вопрос: что произошло после T_frac1[j]?
        Query:  X_f2[c2_j]  — T_frac2-контекст на момент T_frac1[j]
        Пул:   T_frac1 события m < j, представленные T_frac2-контекстом X_f2[c2_m]
        Метка: p_f1[m+1]     — следующий T_frac1 после m (тот же масштаб, что и цель)
        → y_hat_j = LWR предсказание p_f1[j+1] через T_frac2-контекст

  Финал: LWR на (Xn1[knn1], y_hat) с весами уровня 1 → pred p_big[step+1]

Аналогия с B1:
  B1:       X_big[step]  → пул X_f1[m]      → метка p_f1[m+1] → pred p_big[step+1]
  Cascade:  X_f2[c2_j]  → пул X_f2[c2_m]   → метка p_f1[m+1] → pred p_f1[j+1]
  Разница: другой эмбеддинг пространства поиска (T_frac2 vs T_frac1),
           метка та же (T_frac1 цена).

Параметры: T_big=4%, T_frac1=3%, T_frac2=2.5%, K1=K2=20, P=3, H=1

Запуск:
  python 03_cascade_forecast.py           # полный прогон
  python 03_cascade_forecast.py --test 10 # только первые N шагов
"""
import json, csv, argparse
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent.parent.parent / "data" / "candles" / "SBER"

T_BIG   = 0.04
T_FRAC1 = 0.03
T_FRAC2 = 0.025

H           = 1
MIN_HISTORY = 50
P           = 3
K1          = 20
K2          = 20

B1_K50  = 0.4023   # frac-only эксп.01, K=50
T4_BASE = 0.4163   # baseline LWR T_big only, K=50


# ── данные ───────────────────────────────────────────────────────────────────
def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    vals, dts = [], []
    direction, ext_val, ext_idx = 0, (highs[0] + lows[0]) / 2.0, 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction, ext_val, ext_idx = 1, highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                direction, ext_val, ext_idx = -1, lows[i], i
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                direction, ext_val, ext_idx = -1, lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                direction, ext_val, ext_idx = 1, highs[i], i
    return np.array(vals), np.array(dts)


def build_X(prices, p):
    n  = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def rmae(errs, acts):
    acts = np.asarray(acts, dtype=float)
    pers = np.abs(acts[2:] - acts[:-2]) if len(acts) >= 3 else np.abs(np.diff(acts))
    dz   = float(np.mean(pers)) if len(pers) > 0 else 1.0
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


def lwr_fit(xn_q, Xn_pool, y_pool, K):
    d    = np.linalg.norm(Xn_pool - xn_q, axis=1)
    k    = min(K, len(d))
    ord_ = np.argsort(d)
    knn  = ord_[:k]
    xi   = d[ord_[k - 1]]
    if xi < 1e-12:
        return float(y_pool[knn].mean())
    w  = np.exp(-0.5 * (d[knn] / xi) ** 2)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(k), Xn_pool[knn]]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, y_pool[knn] * ws, rcond=None)
    return float(c[0] + c[1:] @ xn_q)


# ── Baseline: LWR T_big only ─────────────────────────────────────────────────
def run_baseline(p_big, X_big, n_big, test_n):
    errs, acts = [], []
    for step in range(MIN_HISTORY, n_big - H):
        if test_n and len(errs) >= test_n:
            break
        if np.any(np.isnan(X_big[step])):
            continue
        idx = np.where(
            ~np.any(np.isnan(X_big[:step]), axis=1)
            & (np.arange(step) >= P - 1)
        )[0]
        if len(idx) < P + 2:
            continue
        mu  = X_big[idx].mean(0)
        sig = np.where(X_big[idx].std(0) < 1e-10, 1.0, X_big[idx].std(0))
        Xn  = (X_big[idx]  - mu) / sig
        xn  = (X_big[step] - mu) / sig
        pred = lwr_fit(xn, Xn, p_big[idx + H], K1)
        if np.isnan(pred):
            continue
        errs.append(pred - float(p_big[step + H]))
        acts.append(float(p_big[step + H]))
    return np.array(errs), np.array(acts)


# ── B1: frac-only T_frac1 ────────────────────────────────────────────────────
def run_b1(p_big, dt_big, X_big, n_big,
           p_f1, dt_f1, X_f1, n_f1, test_n):
    errs, acts = [], []
    ce1_all = np.searchsorted(dt_f1, dt_big, side='left')
    for step in range(MIN_HISTORY, n_big - H):
        if test_n and len(errs) >= test_n:
            break
        if np.any(np.isnan(X_big[step])):
            continue
        ce1 = int(ce1_all[step])
        idx = np.where(
            ~np.any(np.isnan(X_f1[:ce1]), axis=1)
            & (np.arange(ce1) >= P - 1)
            & (np.arange(ce1) + H < n_f1)
        )[0]
        if len(idx) < P + 2:
            continue
        mu  = X_f1[idx].mean(0)
        sig = np.where(X_f1[idx].std(0) < 1e-10, 1.0, X_f1[idx].std(0))
        Xn  = (X_f1[idx]   - mu) / sig
        xn  = (X_big[step] - mu) / sig
        pred = lwr_fit(xn, Xn, p_f1[idx + H], K1)
        if np.isnan(pred):
            continue
        errs.append(pred - float(p_big[step + H]))
        acts.append(float(p_big[step + H]))
    return np.array(errs), np.array(acts)


# ── CASCADE: T_frac2-indexed T_frac1 sub-prediction ─────────────────────────
def run_cascade(p_big, dt_big, X_big, n_big,
                p_f1, dt_f1, X_f1, n_f1,
                p_f2, dt_f2, X_f2, n_f2,
                test_n, verbose):
    """
    Для каждого T_frac1 соседа j строим sub-prediction через T_frac2-контекст.

    Sub-pool = T_frac1 события m < j, представленные своим T_frac2-контекстом.
    Query    = T_frac2-контекст на момент j  (X_f2[c2_j]).
    Метка    = p_f1[m+1] (следующий T_frac1 после m).

    Это аналог B1 на уровне ниже:
      B1:      X_big → пул X_f1[m] → метка p_f1[m+1] → pred p_big[step+1]
      Cascade: X_f2[c2_j] → пул X_f2[c2_m] → метка p_f1[m+1] → pred p_f1[j+1]
    """
    ce1_all = np.searchsorted(dt_f1, dt_big, side='left')   # (n_big,)
    c2j_all = np.searchsorted(dt_f2, dt_f1,  side='left') - 1  # (n_f1,)

    valid_f1 = ~np.any(np.isnan(X_f1), axis=1)
    valid_f2 = ~np.any(np.isnan(X_f2), axis=1)

    # Предвычисляем маску валидности для sub-pool (T_frac1 событий)
    # m валиден, если: f1-эмбеддинг валиден, достаточно истории, метка в пределах,
    # T_frac2-контекст c2_m существует и валиден
    c2j_clip = np.clip(c2j_all, 0, n_f2 - 1)
    valid_c2  = (c2j_all >= 0) & valid_f2[c2j_clip]
    valid_sub = (valid_f1
                 & (np.arange(n_f1) >= P - 1)
                 & (np.arange(n_f1) + H < n_f1)
                 & valid_c2)
    valid_sub_idx = np.where(valid_sub)[0]  # отсортированные валидные T_frac1 индексы

    errs, acts = [], []
    fallback_cnt = 0
    total_inner  = 0

    for step in range(MIN_HISTORY, n_big - H):
        if test_n and len(errs) >= test_n:
            break
        if np.any(np.isnan(X_big[step])):
            continue

        # --- Уровень 1: T_frac1 пул (как B1) ---
        ce1 = int(ce1_all[step])
        rng1 = np.arange(ce1)
        mask1 = valid_f1[:ce1] & (rng1 >= P - 1) & (rng1 + H < n_f1)
        idx1 = np.where(mask1)[0]
        if len(idx1) < P + 2:
            continue

        X1_pool = X_f1[idx1]
        mu1  = X1_pool.mean(0)
        sig1 = np.where(X1_pool.std(0) < 1e-10, 1.0, X1_pool.std(0))
        Xn1  = (X1_pool     - mu1) / sig1
        xn1  = (X_big[step] - mu1) / sig1

        d1   = np.linalg.norm(Xn1 - xn1, axis=1)
        k1   = min(K1, len(d1))
        ord1 = np.argsort(d1)
        knn1 = ord1[:k1]
        xi1  = d1[ord1[k1 - 1]]

        # --- Уровень 2: cascade sub-prediction для каждого соседа j ---
        y_hat = np.empty(k1)

        for li in range(k1):
            j    = idx1[knn1[li]]
            total_inner += 1

            c2_j = int(c2j_all[j])
            if c2_j < 0 or np.any(np.isnan(X_f2[c2_j])):
                y_hat[li] = p_f1[j + H]
                fallback_cnt += 1
                continue

            # Sub-pool: T_frac1 события m < j с валидным T_frac2-контекстом
            # Используем предвычисленный отсортированный массив valid_sub_idx
            cut   = int(np.searchsorted(valid_sub_idx, j, side='left'))
            idx_m = valid_sub_idx[:cut]  # m < j, валидные

            if len(idx_m) < P + 2:
                y_hat[li] = p_f1[j + H]
                fallback_cnt += 1
                continue

            # T_frac2-контексты для sub-pool и query
            c2_m    = c2j_all[idx_m]      # (n_pool,) T_frac2 индексы для m
            X2_pool = X_f2[c2_m]          # (n_pool, P) T_frac2 эмбеддинги
            y_pool  = p_f1[idx_m + H]     # (n_pool,)  метки: p_f1[m+1]
            x_q2    = X_f2[c2_j]          # (P,) T_frac2 эмбеддинг для j

            mu2  = X2_pool.mean(0)
            sig2 = np.where(X2_pool.std(0) < 1e-10, 1.0, X2_pool.std(0))
            Xn2  = (X2_pool - mu2) / sig2
            xn2  = (x_q2   - mu2) / sig2

            pred_j = lwr_fit(xn2, Xn2, y_pool, K2)
            y_hat[li] = pred_j if not np.isnan(pred_j) else p_f1[j + H]

        # --- Финал: LWR на (Xn1[knn1], y_hat) ---
        valid_h = ~np.isnan(y_hat)
        if valid_h.sum() < 2:
            continue

        knn1_v  = knn1[valid_h]
        y_hat_v = y_hat[valid_h]

        if xi1 < 1e-12:
            pred_final = float(y_hat_v.mean())
        else:
            w1   = np.exp(-0.5 * (d1[knn1_v] / xi1) ** 2)
            ws1  = np.sqrt(w1)
            A1   = np.column_stack([np.ones(len(knn1_v)),
                                     Xn1[knn1_v]]) * ws1[:, None]
            c1, *_ = np.linalg.lstsq(A1, y_hat_v * ws1, rcond=None)
            pred_final = float(c1[0] + c1[1:] @ xn1)

        if np.isnan(pred_final):
            continue

        actual = float(p_big[step + H])
        errs.append(pred_final - actual)
        acts.append(actual)

        if verbose:
            print(f"  step={step:4d}  y_hat=[{y_hat_v.min():.2f},{y_hat_v.max():.2f}]"
                  f"  pred={pred_final:.2f}  actual={actual:.2f}"
                  f"  err={pred_final-actual:+.2f}"
                  f"  fb={fallback_cnt}/{total_inner}")

    return np.array(errs), np.array(acts), fallback_cnt, total_inner


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=int, default=0)
    args = ap.parse_args()
    test_n  = args.test
    verbose = bool(test_n)

    h, l, d = load_tf("10m")
    p_big, dt_big = find_pivots(h, l, d, T_BIG)
    p_f1,  dt_f1  = find_pivots(h, l, d, T_FRAC1)
    p_f2,  dt_f2  = find_pivots(h, l, d, T_FRAC2)
    n_big, n_f1, n_f2 = len(p_big), len(p_f1), len(p_f2)

    X_big = build_X(p_big, P)
    X_f1  = build_X(p_f1,  P)
    X_f2  = build_X(p_f2,  P)

    mode = f"ТЕСТ ({test_n} шагов)" if test_n else "ПОЛНЫЙ ПРОГОН"
    print(f"SBER 10m | {mode}")
    print(f"T_big={T_BIG*100:.0f}% n={n_big} | "
          f"T_f1={T_FRAC1*100:.0f}% n={n_f1} | "
          f"T_f2={T_FRAC2*100:.1f}% n={n_f2}")
    print(f"K1={K1}  K2={K2}  P={P}  H={H}\n")

    print("B0 (baseline LWR T_big only, K=K1)...")
    e0, a0 = run_baseline(p_big, X_big, n_big, test_n)
    r0 = rmae(e0, a0)
    print(f"  rMAE={r0:.4f}  n={len(e0)}\n")

    print("B1 (frac-only T_frac1=3%, K=K1)...")
    e1, a1 = run_b1(p_big, dt_big, X_big, n_big,
                    p_f1, dt_f1, X_f1, n_f1, test_n)
    r1 = rmae(e1, a1)
    print(f"  rMAE={r1:.4f}  n={len(e1)}\n")

    print("CASCADE (T_frac2-indexed T_frac1 sub-prediction, K1=K2=20)...")
    if verbose:
        print(f"  {'step':>4}  {'y_hat_range':>20}  {'pred':>8}  "
              f"{'actual':>8}  {'err':>7}  fb/tot")
    ec, ac, fb, tot = run_cascade(
        p_big, dt_big, X_big, n_big,
        p_f1, dt_f1, X_f1, n_f1,
        p_f2, dt_f2, X_f2, n_f2,
        test_n, verbose)
    rc = rmae(ec, ac)
    print(f"  rMAE={rc:.4f}  n={len(ec)}"
          f"  fallback={fb}/{tot} ({fb/tot*100:.1f}%)\n")

    print(f"{'═'*64}")
    if not test_n:
        print(f"  B1_K50 (эксп.01)            rMAE = {B1_K50:.4f}")
        print(f"  T4_baseline_K50             rMAE = {T4_BASE:.4f}")
    print(f"  B0 baseline   K={K1:2d}          rMAE = {r0:.4f}")
    print(f"  B1 frac-only  K={K1:2d}          rMAE = {r1:.4f}"
          f"  Δ B0={(r1-r0)/r0*100:+.1f}%")
    print(f"  CASCADE                      rMAE = {rc:.4f}"
          f"  Δ B0={(rc-r0)/r0*100:+.1f}%  "
          f"Δ B1={(rc-r1)/r1*100:+.1f}%")
    print(f"{'═'*64}")

    if not test_n:
        out = HERE / "results" / "cascade_result.csv"
        rows = [
            {"method": "B0_baseline_K20", "rMAE": r0, "n": len(e0)},
            {"method": "B1_frac_only_K20", "rMAE": r1, "n": len(e1)},
            {"method": "CASCADE_K20",      "rMAE": rc, "n": len(ec)},
            {"method": "B1_K50_ref",        "rMAE": B1_K50,  "n": ""},
            {"method": "T4_base_K50_ref",   "rMAE": T4_BASE, "n": ""},
        ]
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "rMAE", "n"])
            w.writeheader(); w.writerows(rows)
        print(f"  Результаты: {out}")


if __name__ == "__main__":
    main()
