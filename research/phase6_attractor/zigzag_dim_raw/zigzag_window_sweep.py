#!/usr/bin/env python3
"""
zigzag_window_sweep.py

Свип размера скользящего окна пула W (в 1d-пивотах).

Для шага step: пул = пивоты из [step−W, step) по 1d,
и 10m-пивоты в тот же датовый диапазон [dt1d[step−W]..dt1d[step]).
Если step < W — берём всё доступное (окно не применяется).

Цель: найти W, при котором поздние шаги (режим 2020+) улучшаются,
а ранние не деградируют сильнее, чем нужно для сохранения пула.

W_GRID = [50, 100, 150, 200, 300, 500, "all"]
"""

import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"

T_1D        = 0.04
T_10M       = 0.004
H           = 1
MIN_HISTORY = 50

P_LWR  = 3;  K_LWR = 50
P_SX   = 8;  K_SX  = P_SX + 1
THETA  = 1.0
P_RBF  = 3;  K_RBF = 12

A_LWR = 0.05; A_SX = 0.35; A_SM = 0.20; A_RBF = 0.40

W_GRID = [50, 100, 150, 200, 300, 500, None]   # None = all history
N_BINS = 10


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    vals, dts = [], []
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
                vals.append(ext_val); dts.append(dates[ext_idx])
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                direction = 1; ext_val, ext_idx = highs[i], i
    if not vals:
        return np.array([]), np.array([])
    return np.array(vals), np.array(dts)


def build_X(prices, p):
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def run_window(W, p1d, dt1d, p10m, dt10m, Xs, n1d, na):
    """Прогон ансамбля с окном W. W=None → вся история."""
    raw_errs    = []
    raw_actuals = []
    raw_step    = []
    raw_pool_sz = []

    for step in range(MIN_HISTORY, n1d - H):
        skip = any(np.any(np.isnan(Xs[p][0][step])) for p in [P_LWR, P_SX, P_RBF])
        if skip:
            continue

        ce    = int(np.searchsorted(dt10m, dt1d[step], side="left"))
        p_cur = float(p1d[step])

        # Начало 1d-окна
        start_1d = (max(0, step - W) if W is not None else 0)
        # Начало 10m-окна по дате
        start_date = dt1d[start_1d]
        ce_start   = int(np.searchsorted(dt10m, start_date, side="left"))

        def make_pool(p):
            X1d_, X10m_ = Xs[p]
            j1d = np.arange(max(p - 1, start_1d), step)
            v1d = ~np.any(np.isnan(X1d_[j1d]), axis=1) & (j1d + H < n1d)
            idx1   = j1d[v1d]
            Xp     = list(X1d_[idx1])
            yp_abs = list(p1d[idx1 + H])
            yp_src = list(p1d[idx1])

            j10 = np.arange(max(p - 1, ce_start), min(ce, na - H))
            if len(j10):
                v10   = ~np.any(np.isnan(X10m_[j10]), axis=1)
                idx10 = j10[v10]
                Xp.extend(X10m_[idx10])
                yp_abs.extend(p10m[idx10 + H])
                yp_src.extend(p10m[idx10])

            if len(Xp) < p + 2:
                return None
            X_pool = np.array(Xp)
            y_abs  = np.array(yp_abs, dtype=float)
            y_lr   = np.log(y_abs / np.array(yp_src, dtype=float))
            x_q    = X1d_[step]
            mu  = X_pool.mean(0)
            sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
            Xn  = (X_pool - mu) / sig
            xn  = (x_q    - mu) / sig
            d   = np.linalg.norm(Xn - xn, axis=1)
            return Xn, xn, d, y_abs, y_lr, X_pool.shape[0]

        res3 = make_pool(P_LWR)
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

        res8 = make_pool(P_SX)
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

        raw_errs.append(pred_ens - float(p1d[step + H]))
        raw_actuals.append(float(p1d[step + H]))
        raw_step.append(step)
        raw_pool_sz.append(res3[5] if res3 else 0)

    return (np.array(raw_errs), np.array(raw_actuals),
            np.array(raw_step), np.array(raw_pool_sz))


def rmae_arr(errs, acts):
    dz = np.mean(np.abs(np.diff(acts)))
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


def main():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m = find_pivots(h10m, l10m, d10m, T_10M)

    n1d = len(p1d); na = len(p10m)
    Xs = {p: (build_X(p1d, p), build_X(p10m, p))
          for p in set([P_LWR, P_SX, P_RBF])}

    print(f"SBER 1d {n1d} пив  10m {na} пив\n")

    # Собираем результаты per W
    results = {}
    for W in W_GRID:
        label = str(W) if W else "all"
        print(f"W={label:>4} ...", end=" ", flush=True)
        errs, acts, steps, pools = run_window(W, p1d, dt1d, p10m, dt10m, Xs, n1d, na)
        results[label] = (errs, acts, steps, pools)
        g = rmae_arr(errs, acts)
        print(f"rMAE={g:.4f}  пул avg={pools.mean():.0f}")

    # ── Сводная таблица: global rMAE ─────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"{'W':>6}  {'global rMAE':>12}  {'avg pool':>9}  {'min pool':>9}")
    print(f"{'─'*55}")
    for W in W_GRID:
        label = str(W) if W else "all"
        errs, acts, steps, pools = results[label]
        g = rmae_arr(errs, acts)
        print(f"{label:>6}  {g:>12.4f}  {pools.mean():>9.0f}  {pools.min():>9.0f}")

    # ── Временна́я таблица rMAE по бинам ─────────────────────────────────────
    # Используем шаги из варианта "all" как общую сетку
    ref_steps = results["all"][2]
    bin_edges  = np.percentile(ref_steps, np.linspace(0, 100, N_BINS + 1))
    bin_edges[-1] += 1

    w_labels = [str(W) if W else "all" for W in W_GRID]
    col_w    = 6

    print(f"\nrMAE по хронологическим бинам:")
    header = f"{'Шаги':>12}" + "".join(f"{lbl:>{col_w+1}}" for lbl in w_labels)
    print(header)
    print("─" * (13 + (col_w + 1) * len(W_GRID)))

    for i in range(N_BINS):
        lo, hi = int(bin_edges[i]), int(bin_edges[i+1]) - 1
        row = f"{lo}..{hi:>3}"

        row_vals = []
        for W in W_GRID:
            label = str(W) if W else "all"
            errs, acts, steps, _ = results[label]
            m = (steps >= bin_edges[i]) & (steps < bin_edges[i+1])
            if m.sum() < 3:
                row_vals.append("  —  ")
            else:
                r = float(np.mean(np.abs(errs[m])) / np.mean(np.abs(np.diff(acts))))
                row_vals.append(f"{r:.4f}")

        print(f"{row:>12}" + "".join(f"{v:>{col_w+1}}" for v in row_vals))

    # ── Таблица среднего размера пула по бинам для W=200 ─────────────────────
    print(f"\nСредний размер пула по бинам (W=200 vs all):")
    print(f"{'Шаги':>12}  {'pool W=200':>11}  {'pool all':>10}")
    for i in range(N_BINS):
        lo, hi = int(bin_edges[i]), int(bin_edges[i+1]) - 1
        for lbl, W_ in [("200", 200), ("all", None)]:
            pass  # вычислим ниже
        errs_200, _, steps_200, pools_200 = results["200"]
        errs_all, _, steps_all, pools_all = results["all"]
        m200 = (steps_200 >= bin_edges[i]) & (steps_200 < bin_edges[i+1])
        mall  = (steps_all  >= bin_edges[i]) & (steps_all  < bin_edges[i+1])
        p200 = pools_200[m200].mean() if m200.sum() > 0 else float("nan")
        pall = pools_all[mall].mean()  if mall.sum()  > 0 else float("nan")
        print(f"{lo}..{hi:>3}  {p200:>11.0f}  {pall:>10.0f}")


if __name__ == "__main__":
    main()
