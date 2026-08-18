#!/usr/bin/env python3
"""
zigzag_no10m_check.py

Тот же 4-way ансамбль, но БЕЗ 10m-пула.
Цель: выяснить, что вызывает рост rMAE на поздних шагах —
режим рынка или смешение 10m-свингов в пуле.

Если временна́я таблица rMAE похожа на версию с 10m → виноват режим.
Если поздние шаги улучшаются → виноват 10m-пул.
"""

import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"

T_1D        = 0.04
H           = 1
MIN_HISTORY = 50

P_LWR  = 3;  K_LWR = 50
P_SX   = 8;  K_SX  = P_SX + 1
THETA  = 1.0
P_RBF  = 3;  K_RBF = 12

A_LWR = 0.05; A_SX = 0.35; A_SM = 0.20; A_RBF = 0.40

# rMAE из полного прогона (с 10m) — для сравнения
REF_10M_BINS = {
    "50..132":   0.2213,
    "133..215":  0.2103,
    "216..298":  0.2386,
    "299..381":  0.1983,
    "382..464":  0.2490,
    "465..547":  0.2298,
    "548..630":  0.5222,
    "631..713":  0.6214,
    "714..796":  0.7963,
    "797..881":  0.6744,
}


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    vals, dts, bidxs = [], [], []
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
                vals.append(ext_val); dts.append(dates[ext_idx]); bidxs.append(ext_idx)
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx]); bidxs.append(ext_idx)
                direction = 1; ext_val, ext_idx = highs[i], i
    if not vals:
        return np.array([]), np.array([]), np.array([], int)
    return np.array(vals), np.array(dts), np.array(bidxs, int)


def build_X(prices, p):
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def run():
    h1d, l1d, d1d = load_tf("1d")
    p1d, dt1d, _ = find_pivots(h1d, l1d, d1d, T_1D)
    n1d = len(p1d)

    Xs = {p: build_X(p1d, p) for p in set([P_LWR, P_SX, P_RBF])}

    print(f"SBER 1d {n1d} пивотов  (10m НЕ используется)")
    print(f"Веса: α_lwr={A_LWR} α_sx={A_SX} α_sm={A_SM} α_rbf={A_RBF}\n")

    raw_errs     = []
    raw_actuals  = []
    raw_step_seq = []
    raw_pool_sz  = []

    for step in range(MIN_HISTORY, n1d - H):
        skip = any(np.any(np.isnan(Xs[p][step])) for p in [P_LWR, P_SX, P_RBF])
        if skip:
            continue

        p_cur = float(p1d[step])

        def make_pool_1d(p):
            X1d_ = Xs[p]
            j1d  = np.arange(p - 1, step)
            v1d  = ~np.any(np.isnan(X1d_[j1d]), axis=1) & (j1d + H < n1d)
            idx1 = j1d[v1d]
            if len(idx1) < p + 2:
                return None
            X_pool = X1d_[idx1]
            y_abs  = p1d[idx1 + H]
            y_lr   = np.log(y_abs / p1d[idx1])
            x_q    = X1d_[step]
            mu  = X_pool.mean(0)
            sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
            Xn  = (X_pool - mu) / sig
            xn  = (x_q    - mu) / sig
            d   = np.linalg.norm(Xn - xn, axis=1)
            return Xn, xn, d, y_abs, y_lr, len(idx1)

        res3 = make_pool_1d(P_LWR)
        y_lwr = y_smap = np.nan
        if res3:
            Xn, xn, d, y_abs, y_lr, N = res3
            ord_ = np.argsort(d)
            k_eff = min(K_LWR, N); knn = ord_[:k_eff]; xi = d[ord_[k_eff-1]]
            if xi < 1e-12:
                y_lwr = float(y_abs[knn].mean())
            else:
                w = np.exp(-0.5*(d[knn]/xi)**2); ws = np.sqrt(w)
                A_ = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:,None]
                c, *_ = np.linalg.lstsq(A_, y_abs[knn]*ws, rcond=None)
                y_lwr = float(c[0] + c[1:] @ xn)
            mean_d = d.mean() + 1e-12
            w_sm   = np.exp(-THETA * d / mean_d); ws_sm = np.sqrt(w_sm)
            A_sm   = np.column_stack([np.ones(N), Xn]) * ws_sm[:,None]
            c_sm, *_ = np.linalg.lstsq(A_sm, y_abs*ws_sm, rcond=None)
            y_smap = float(c_sm[0] + c_sm[1:] @ xn)

        res8 = make_pool_1d(P_SX)
        y_sx = np.nan
        if res8 and res8[5] >= K_SX:
            Xn, xn, d, y_abs, y_lr, N = res8
            ords = np.argsort(d); knn = ords[:K_SX]; d1 = d[ords[0]]
            if d1 < 1e-12:
                y_sx = float(p_cur * np.exp(y_lr[ords[0]]))
            else:
                w = np.exp(-d[knn]/d1); w /= w.sum()
                y_sx = float(p_cur * np.exp(w @ y_lr[knn]))

        y_rbf = np.nan
        if res3 and res3[5] >= K_RBF:
            Xn, xn, d, y_abs, y_lr, N = res3
            ord_ = np.argsort(d); k = min(K_RBF, N); knn = ord_[:k]
            xi = d[ord_[k-1]]
            if xi < 1e-12:
                y_rbf = float(p_cur * np.exp(y_lr[knn].mean()))
            else:
                w = np.exp(-0.5*(d[knn]/xi)**2)
                y_rbf = float(p_cur * np.exp((w @ y_lr[knn]) / w.sum()))

        pred_ens = A_LWR*y_lwr + A_SX*y_sx + A_SM*y_smap + A_RBF*y_rbf
        if np.isnan(pred_ens):
            continue

        actual = float(p1d[step + H])
        raw_errs.append(pred_ens - actual)
        raw_actuals.append(actual)
        raw_step_seq.append(step)
        raw_pool_sz.append(res3[5] if res3 else 0)

    acts     = np.array(raw_actuals)
    errs     = np.array(raw_errs)
    step_seq = np.array(raw_step_seq)
    pool_szs = np.array(raw_pool_sz)
    n_steps  = len(acts)
    dz       = float(np.mean(np.abs(np.diff(acts))))

    global_rmae = float(np.mean(np.abs(errs)) / dz)
    print(f"Шагов: {n_steps}   global rMAE (без 10m): {global_rmae:.4f}")
    print(f"Справка: с 10m rMAE = 0.3963\n")

    # Хронологические бины (те же границы, что в версии с 10m)
    N_BINS = 10
    bin_edges = np.percentile(step_seq, np.linspace(0, 100, N_BINS + 1))
    bin_edges[-1] += 1

    ref_vals = list(REF_10M_BINS.values())

    print(f"{'Шаги':>12}  {'n':>4}  {'rMAE (no10m)':>13}  {'rMAE (10m)':>11}  "
          f"{'Δ':>7}  {'pool(no10m)':>12}")
    print("─" * 72)
    for i in range(N_BINS):
        m = (step_seq >= bin_edges[i]) & (step_seq < bin_edges[i+1])
        if m.sum() < 3:
            continue
        r_no10 = float(np.mean(np.abs(errs[m])) / dz)
        r_10m  = ref_vals[i]
        delta  = r_no10 - r_10m
        label  = f"{int(bin_edges[i])}..{int(bin_edges[i+1])-1}"
        print(f"{label:>12}  {m.sum():>4}  {r_no10:>13.4f}  {r_10m:>11.4f}  "
              f"{delta:>+7.4f}  {np.mean(pool_szs[m]):>12.1f}")

    print(f"\n{'─'*40}")
    print(f"rMAE при увеличении MIN_HISTORY (без 10m):")
    print(f"  {'Отброшено':>10}  {'Осталось':>9}  {'rMAE':>7}")
    for skip in [0, 50, 100, 200, 300, 400]:
        m = step_seq >= step_seq[0] + skip
        if m.sum() < 10:
            break
        r = float(np.mean(np.abs(errs[m])) / dz)
        print(f"  {skip:>10}  {m.sum():>9}  {r:>7.4f}")


if __name__ == "__main__":
    run()
