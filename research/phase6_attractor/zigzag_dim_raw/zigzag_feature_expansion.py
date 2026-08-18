#!/usr/bin/env python3
"""
zigzag_feature_expansion.py

Расширение признакового вектора X=[price, lr1, lr2] (p=3)
дополнительными признаками:

  log_inter  = log(bars от предыдущего пивота до текущего)
               Темп рынка: быстрые vs затяжные колебания.
               Вычисляется из bar_idx каждого пивота в исходном свечном ряду.

  pos_ma_W   = (price - MA_W[bar]) / MA_W[bar]
               Позиция пивота относительно каузальной скользящей средней.
               MA вычисляется по mid=(high+low)/2 исходного свечного ряда.

Варианты (8):
  base        — X=[price, lr1, lr2]            (воспроизводит baseline)
  +inter      — X=[price, lr1, lr2, log_inter]
  +pos_50/100/200
              — X=[price, lr1, lr2, pos_ma]
  +both_50/100/200
              — X=[price, lr1, lr2, log_inter, pos_ma]

Оба признака применяются к 1d и 10m пулу независимо.
log_inter для 10m — в барах 10m (масштаб отличается от 1d, но z-score нивелирует).

Причинность: log_inter и pos_ma вычисляются в момент подтверждения пивота,
используя только прошлые данные. Нет look-ahead.
"""
import json
import numpy as np
from pathlib import Path

BASE_DIR    = Path(__file__).parent
DATA        = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT         = BASE_DIR / "results"
OUT.mkdir(exist_ok=True)

T_1D  = 0.04
T_10M = 0.004
H     = 1
MIN_HISTORY = 50

P_LWR = 3;  K_LWR = 50
P_SX  = 8;  K_SX  = P_SX + 1
THETA = 1.0
P_RBF = 3;  K_RBF = 12

A_LWR = 0.05; A_SX = 0.35; A_SM = 0.20; A_RBF = 0.40

MA_WINDOWS = [50, 100, 200]


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    """Возвращает (vals, dts, bar_idxs) где bar_idxs — бар экстремума в свечном ряду."""
    vals, dts, bar_idxs = [], [], []
    direction = 0
    ext_val = (highs[0] + lows[0]) / 2.0
    ext_idx = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1; ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val, ext_idx = lows[i], i
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                bar_idxs.append(ext_idx)
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                bar_idxs.append(ext_idx)
                direction = 1; ext_val, ext_idx = highs[i], i
    return np.array(vals), np.array(dts), np.array(bar_idxs, dtype=int)


def causal_ma(mids, W):
    cs     = np.concatenate([[0.0], np.cumsum(mids)])
    counts = np.minimum(np.arange(1, len(mids) + 1), W)
    start  = np.maximum(0, np.arange(len(mids)) - W + 1)
    return (cs[1:] - cs[start]) / counts


def build_X(prices, p):
    n  = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def build_X_ext(prices, bar_idxs, mids_full, p,
                include_inter, include_posma, ma_w):
    """
    Строит расширенную матрицу признаков.
    prices:    цены пивотов (длина n)
    bar_idxs:  бары пивотов в исходном свечном ряду (длина n)
    mids_full: (high+low)/2 исходного свечного ряда (длина = кол-во свечей)
    """
    X    = build_X(prices, p)
    cols = [X]

    if include_inter:
        log_inter = np.full(len(prices), np.nan)
        for i in range(1, len(prices)):
            diff = int(bar_idxs[i]) - int(bar_idxs[i - 1])
            if diff > 0:
                log_inter[i] = np.log(float(diff))
        cols.append(log_inter.reshape(-1, 1))

    if include_posma:
        MA     = causal_ma(mids_full, ma_w)
        pos_ma = np.full(len(prices), np.nan)
        for i in range(len(prices)):
            mv = MA[bar_idxs[i]]
            if mv > 1e-10:
                pos_ma[i] = (prices[i] - mv) / mv
        cols.append(pos_ma.reshape(-1, 1))

    return np.column_stack(cols) if len(cols) > 1 else X


def get_pool(step, ce, X1d, X10m, p1d, p10m, n1d, na, min_lag):
    """
    Собирает пул для одного запроса.
    min_lag: минимальный lag (p-1 для данного p).
    Возвращает (Xn, xn, d, y_abs, y_lr, N) или None.
    """
    x_q = X1d[step]
    if np.any(np.isnan(x_q)):
        return None

    j1d  = np.arange(min_lag, step)
    v1d  = ~np.any(np.isnan(X1d[j1d]), axis=1) & (j1d + H < n1d)
    idx1 = j1d[v1d]

    Xp     = list(X1d[idx1])
    yp_abs = list(p1d[idx1 + H])
    yp_src = list(p1d[idx1])

    j10 = np.arange(min_lag, min(ce, na - H))
    if len(j10):
        v10   = ~np.any(np.isnan(X10m[j10]), axis=1)
        idx10 = j10[v10]
        Xp.extend(X10m[idx10])
        yp_abs.extend(p10m[idx10 + H])
        yp_src.extend(p10m[idx10])

    n_feat = x_q.shape[0]
    if len(Xp) < n_feat + 2:
        return None

    X_pool = np.array(Xp)
    y_abs  = np.array(yp_abs, dtype=float)
    y_lr   = np.log(y_abs / np.array(yp_src, dtype=float))
    mu     = X_pool.mean(0)
    sig    = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
    Xn     = (X_pool - mu) / sig
    xn     = (x_q    - mu) / sig
    d      = np.linalg.norm(Xn - xn, axis=1)
    return Xn, xn, d, y_abs, y_lr, X_pool.shape[0]


def ensemble_step(res_lwr, res_sx, p_cur):
    y_lwr = y_smap = y_sx = y_rbf = np.nan

    if res_lwr:
        Xn, xn, d, y_abs, y_lr, N = res_lwr
        # LWR
        k_eff = min(K_LWR, N); ord_ = np.argsort(d)
        knn = ord_[:k_eff]; xi = d[ord_[k_eff - 1]]
        if xi < 1e-12:
            y_lwr = float(y_abs[knn].mean())
        else:
            w = np.exp(-0.5*(d[knn]/xi)**2); ws = np.sqrt(w)
            A = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
            c, *_ = np.linalg.lstsq(A, y_abs[knn]*ws, rcond=None)
            y_lwr = float(c[0] + c[1:] @ xn)
        # S-map
        mean_d = d.mean() + 1e-12
        w_sm   = np.exp(-THETA*d/mean_d); ws_sm = np.sqrt(w_sm)
        A_sm   = np.column_stack([np.ones(N), Xn]) * ws_sm[:, None]
        c_sm, *_ = np.linalg.lstsq(A_sm, y_abs*ws_sm, rcond=None)
        y_smap = float(c_sm[0] + c_sm[1:] @ xn)
        # RBF-log
        if N >= K_RBF:
            k = min(K_RBF, N); knn_r = np.argsort(d)[:k]; xi_r = d[knn_r[-1]]
            if xi_r < 1e-12:
                y_rbf = p_cur * float(np.exp(y_lr[knn_r].mean()))
            else:
                w_r = np.exp(-0.5*(d[knn_r]/xi_r)**2)
                y_rbf = p_cur * float(np.exp((w_r @ y_lr[knn_r]) / w_r.sum()))

    if res_sx and res_sx[5] >= K_SX:
        Xn_s, xn_s, d_s, y_abs_s, y_lr_s, N_s = res_sx
        ords = np.argsort(d_s); knn_s = ords[:K_SX]; d1 = d_s[ords[0]]
        if d1 < 1e-12:
            y_sx = p_cur * float(np.exp(y_lr_s[ords[0]]))
        else:
            w_s = np.exp(-d_s[knn_s]/d1); w_s /= w_s.sum()
            y_sx = p_cur * float(np.exp(w_s @ y_lr_s[knn_s]))

    return A_LWR*y_lwr + A_SX*y_sx + A_SM*y_smap + A_RBF*y_rbf


def run(X_p3_1d, X_p3_10m, X_p8_1d, X_p8_10m,
        p1d, dt1d, p10m, dt10m, n1d, na):
    errs, acts, pool_szs = [], [], []
    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X_p3_1d[step])) or np.any(np.isnan(X_p8_1d[step])):
            continue
        ce    = int(np.searchsorted(dt10m, dt1d[step], side="left"))
        p_cur = float(p1d[step])
        res_lwr = get_pool(step, ce, X_p3_1d, X_p3_10m, p1d, p10m, n1d, na,
                           min_lag=P_LWR - 1)
        res_sx  = get_pool(step, ce, X_p8_1d, X_p8_10m, p1d, p10m, n1d, na,
                           min_lag=P_SX - 1)
        pred = ensemble_step(res_lwr, res_sx, p_cur)
        if np.isnan(pred):
            continue
        errs.append(pred - float(p1d[step + H]))
        acts.append(float(p1d[step + H]))
        pool_szs.append(res_lwr[5] if res_lwr else 0)
    return np.array(errs), np.array(acts), np.array(pool_szs)


def rmae(errs, acts):
    dz = float(np.mean(np.abs(np.diff(acts))))
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


def main():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    mid1d  = (h1d  + l1d)  / 2.0
    mid10m = (h10m + l10m) / 2.0

    p1d,  dt1d,  bidx1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m, bidx10m = find_pivots(h10m, l10m, d10m, T_10M)
    n1d = len(p1d); na = len(p10m)
    print(f"SBER: 1d {n1d} пив  10m {na} пив\n")

    def mx(p, inter, posma, w):
        return (build_X_ext(p1d,  bidx1d,  mid1d,  p, inter, posma, w),
                build_X_ext(p10m, bidx10m, mid10m, p, inter, posma, w))

    variants = {"base": (mx(P_LWR, False, False, None), mx(P_SX, False, False, None))}
    variants["+inter"] = (mx(P_LWR, True, False, None), mx(P_SX, True, False, None))
    for w in MA_WINDOWS:
        variants[f"+pos_{w}"]  = (mx(P_LWR, False, True, w), mx(P_SX, False, True, w))
        variants[f"+both_{w}"] = (mx(P_LWR, True,  True, w), mx(P_SX, True,  True, w))

    results = {}
    for name, (Xp3, Xp8) in variants.items():
        print(f"Запуск: {name:<14} ...", end=" ", flush=True)
        errs, acts, pool_szs = run(Xp3[0], Xp3[1], Xp8[0], Xp8[1],
                                   p1d, dt1d, p10m, dt10m, n1d, na)
        results[name] = (errs, acts, pool_szs)
        print(f"rMAE={rmae(errs, acts):.4f}")

    base_r = rmae(results["base"][0], results["base"][1])

    print(f"\n{'═'*52}")
    print(f"  {'Вариант':<15}  {'rMAE':>8}  {'Δ vs base':>10}  {'пул avg':>8}")
    print(f"{'─'*52}")
    for name, (errs, acts, pool_szs) in results.items():
        r     = rmae(errs, acts)
        delta = (r - base_r) / base_r * 100
        flag  = " ←" if r < base_r else ""
        print(f"  {name:<15}  {r:.4f}    {delta:+.2f}%       {pool_szs.mean():>7.0f}{flag}")


if __name__ == "__main__":
    main()
