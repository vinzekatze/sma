#!/usr/bin/env python3
"""
sanity_check_forecast.py — sanity-тест алгоритма прогноза на синтетических рядах.

Запускать после любых изменений в ядре прогноза (build_X, ensemble_predict).

ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ (проверено 2026-06-27):
  white noise (N+10, n=500)  →  rMAE ≈ 0.75   (хуже SBER, но лучше M0 — тривиально)
  random walk (σ=1%, n=500)  →  rMAE ≈ 1.03   (≈ M0 — мартингал, ожидаемо)
  SBER 1d T=4% (n=883)       →  rMAE ≈ 0.42   (намного лучше шума — реальная структура)

ИНТЕРПРЕТАЦИЯ:
  - Белый шум: алгоритм гравитирует к среднему через k-NN. M0 (random walk) —
    специфически плохой предиктор для i.i.d. ряда. Оптимум = μ = 1/√2 ≈ 0.707 rMAE.
    «Победа над M0 на шуме» — тривиальна, не является признаком реальной структуры.
  - Случайное блуждание: предсказание текущего значения (M0) — оптимум (мартингал).
    Алгоритм не может улучшить — rMAE ≈ 1.0. Если rMAE << 1.0 → тревожный знак.
  - SBER: rMAE ≈ 0.42 — разрыв от шума значительный. Структура реальная.

КРАСНЫЕ ФЛАГИ:
  - random walk rMAE < 0.7  → алгоритм переобучается или есть look-ahead
  - SBER rMAE > 0.6         → регрессия качества
  - SBER rMAE ≈ white noise → алгоритм потерял структуру
"""
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent / "data" / "candles" / "SBER"

# ── параметры ансамбля (те же что в основном эксперименте) ──────────────────────
P_LWR = 3;  K_LWR = 50
P_SX  = 8;  K_SX  = P_SX + 1
THETA = 1.0
P_RBF = 3;  K_RBF = 12
A_LWR = 0.05; A_SX = 0.35; A_SM = 0.20; A_RBF = 0.40
MIN_HISTORY = 50
H = 1


def build_X(prices, p):
    n = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def predict_one(step, prices, X_lwr, X_sx):
    x_q3 = X_lwr[step]; x_q8 = X_sx[step]
    if np.any(np.isnan(x_q3)) or np.any(np.isnan(x_q8)):
        return np.nan

    idx   = np.arange(P_LWR - 1, step)
    valid3 = idx[~np.any(np.isnan(X_lwr[idx]), axis=1) & (idx + H < len(prices))]
    if len(valid3) < P_LWR + 2:
        return np.nan

    y_abs3 = prices[valid3 + H]
    y_lr3  = np.log(y_abs3 / prices[valid3])
    mu3    = X_lwr[valid3].mean(0)
    sig3   = np.where(X_lwr[valid3].std(0) < 1e-10, 1.0, X_lwr[valid3].std(0))
    Xn3    = (X_lwr[valid3] - mu3) / sig3
    xn3    = (x_q3 - mu3) / sig3
    d3     = np.linalg.norm(Xn3 - xn3, axis=1)

    # LWR
    k   = min(K_LWR, len(valid3)); ord_ = np.argsort(d3)
    knn = ord_[:k]; xi = d3[ord_[k - 1]]
    if xi < 1e-12:
        y_lwr = float(y_abs3[knn].mean())
    else:
        w = np.exp(-0.5 * (d3[knn] / xi) ** 2); ws = np.sqrt(w)
        A = np.column_stack([np.ones(k), Xn3[knn]]) * ws[:, None]
        c, *_ = np.linalg.lstsq(A, y_abs3[knn] * ws, rcond=None)
        y_lwr = float(c[0] + c[1:] @ xn3)

    # S-map
    N3   = len(valid3); md = d3.mean() + 1e-12
    w_sm = np.exp(-THETA * d3 / md); ws_sm = np.sqrt(w_sm)
    A_sm = np.column_stack([np.ones(N3), Xn3]) * ws_sm[:, None]
    c_sm, *_ = np.linalg.lstsq(A_sm, y_abs3 * ws_sm, rcond=None)
    y_smap = float(c_sm[0] + c_sm[1:] @ xn3)

    # RBF
    y_rbf = np.nan
    if len(valid3) >= K_RBF:
        knn_r = np.argsort(d3)[:K_RBF]; xi_r = d3[knn_r[-1]]
        if xi_r < 1e-12:
            y_rbf = float(prices[step]) * float(np.exp(y_lr3[knn_r].mean()))
        else:
            w_r = np.exp(-0.5 * (d3[knn_r] / xi_r) ** 2)
            y_rbf = float(prices[step]) * float(np.exp((w_r @ y_lr3[knn_r]) / w_r.sum()))

    # Simplex (p=8)
    idx8   = np.arange(P_SX - 1, step)
    valid8 = idx8[~np.any(np.isnan(X_sx[idx8]), axis=1) & (idx8 + H < len(prices))]
    y_sx   = np.nan
    if len(valid8) >= K_SX:
        y_abs8 = prices[valid8 + H]; y_lr8 = np.log(y_abs8 / prices[valid8])
        mu8    = X_sx[valid8].mean(0)
        sig8   = np.where(X_sx[valid8].std(0) < 1e-10, 1.0, X_sx[valid8].std(0))
        Xn8    = (X_sx[valid8] - mu8) / sig8
        xn8    = (x_q8 - mu8) / sig8
        d8     = np.linalg.norm(Xn8 - xn8, axis=1)
        ords   = np.argsort(d8); knn8 = ords[:K_SX]; d1 = d8[ords[0]]
        if d1 < 1e-12:
            y_sx = float(prices[step]) * float(np.exp(y_lr8[ords[0]]))
        else:
            w_s = np.exp(-d8[knn8] / d1); w_s /= w_s.sum()
            y_sx = float(prices[step]) * float(np.exp(w_s @ y_lr8[knn8]))

    y_sx_eff  = y_sx  if not np.isnan(y_sx)  else y_lwr
    y_rbf_eff = y_rbf if not np.isnan(y_rbf) else y_lwr
    return A_LWR * y_lwr + A_SX * y_sx_eff + A_SM * y_smap + A_RBF * y_rbf_eff


def run_series(prices, name):
    X3 = build_X(prices, P_LWR)
    X8 = build_X(prices, P_SX)
    errs, acts = [], []
    for step in range(MIN_HISTORY, len(prices) - H):
        pred = predict_one(step, prices, X3, X8)
        if pred is None or np.isnan(pred):
            continue
        actual = float(prices[step + H])
        errs.append(pred - actual)
        acts.append(actual)
    errs = np.array(errs); acts = np.array(acts)
    dz   = float(np.mean(np.abs(np.diff(acts)))) + 1e-12
    rmae = float(np.mean(np.abs(errs))) / dz
    return rmae, len(errs)


def find_pivots(highs, lows, thr):
    vals = []; direction = 0; ext_val = (highs[0] + lows[0]) / 2
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1; ext_val = highs[i]
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val = lows[i]
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val = highs[i]
            elif ext_val - lows[i] >= thr * ext_val:
                vals.append(ext_val); direction = -1; ext_val = lows[i]
        else:
            if lows[i] < ext_val:
                ext_val = lows[i]
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); direction = 1; ext_val = highs[i]
    return np.array(vals)


def main():
    rng = np.random.default_rng(42)
    N   = 500

    wn = rng.standard_normal(N) + 10.0
    rw = np.exp(np.cumsum(rng.standard_normal(N) * 0.01)) * 100.0

    with open(DATA / "1d.json") as f:
        raw = json.load(f)
    h    = np.array([d["high"] for d in raw])
    l    = np.array([d["low"]  for d in raw])
    sber = find_pivots(h, l, 0.04)

    series = [
        (wn,   "white noise (N+10, n=500)"),
        (rw,   "random walk (σ=1%,  n=500)"),
        (sber, f"SBER 1d T=4%       (n={len(sber)})"),
    ]

    EXPECTED = {
        "white noise": (0.65, 0.85),
        "random walk": (0.85, 1.15),
        "SBER":        (0.30, 0.55),
    }

    print("=" * 65)
    print("  Sanity check: ensemble forecast на синтетике")
    print("=" * 65)
    print(f"  {'Ряд':<35}  {'rMAE':>6}  {'n_pred':>7}  Статус")
    print("─" * 65)

    all_ok = True
    for prices, name in series:
        rmae, n_pred = run_series(prices, name)
        key = "white noise" if "noise" in name else \
              "random walk" if "walk" in name else "SBER"
        lo, hi = EXPECTED[key]
        ok = lo <= rmae <= hi
        status = "OK" if ok else "!! FAIL !!"
        if not ok:
            all_ok = False
        print(f"  {name:<35}  {rmae:>6.4f}  {n_pred:>7d}  {status}")

    print("─" * 65)
    print(f"\n  M0 baseline = 1.000 (random walk prediction)")
    print(f"  Оптимум белого шума ≈ 0.707 (предсказание среднего)")
    print(f"\n  {'PASSED' if all_ok else 'FAILED'}")
    print("=" * 65)
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
