#!/usr/bin/env python3
"""
zigzag_typed_models.py

Три варианта 4-way ансамбля:

  baseline     — смешанный пул без информации о типе (воспроизводит rMAE≈0.3963)
  typed_pool   — пул фильтруется по типу текущего пивота:
                   current=HIGH → пул = только HIGH-пивоты (предсказываем следующий LOW)
                   current=LOW  → пул = только LOW-пивоты  (предсказываем следующий HIGH)
  type_feature — смешанный пул + тип пивота ±0.5 добавлен в X как последний признак

Причинность:
  Зигзаг строго чередует HIGH и LOW → тип следующего пивота детерминирован.
  Тип текущего пивота известен в момент его подтверждения.
  Нет look-ahead ни в одном из вариантов.
"""
import json
import numpy as np
from pathlib import Path

BASE_DIR    = Path(__file__).parent
DATA        = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT         = BASE_DIR / "results"
OUT.mkdir(exist_ok=True)

T_1D        = 0.04
T_10M       = 0.004
H           = 1
MIN_HISTORY = 50

P_LWR = 3;  K_LWR = 50
P_SX  = 8;  K_SX  = P_SX + 1
THETA = 1.0
P_RBF = 3;  K_RBF = 12

A_LWR = 0.05; A_SX = 0.35; A_SM = 0.20; A_RBF = 0.40

VARIANTS = ["baseline", "typed_pool", "type_feature"]


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    vals, dts, ptypes = [], [], []
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
                vals.append(ext_val); dts.append(dates[ext_idx]); ptypes.append(1)
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx]); ptypes.append(-1)
                direction = 1; ext_val, ext_idx = highs[i], i
    return np.array(vals), np.array(dts), np.array(ptypes, dtype=int)


def build_X(prices, p):
    n  = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def build_X_tf(prices, ptypes, p):
    """X с добавленным признаком типа ±0.5 в последнем столбце."""
    X        = build_X(prices, p)
    type_col = np.full(len(prices), np.nan)
    type_col[p - 1:] = ptypes[p - 1:] * 0.5
    return np.column_stack([X, type_col])


def get_pool(p, step, ce, cur_type, variant,
             X1d, X10m, X1d_tf, X10m_tf,
             p1d, p10m, pt1d, pt10m, n1d, na):
    """
    Строит пул для одного запроса. Возвращает (Xn, xn, d, y_abs, y_lr, N)
    или None если пул слишком мал.
    """
    use_tf = (variant == "type_feature")
    Xa  = X1d_tf if use_tf else X1d
    Xb  = X10m_tf if use_tf else X10m
    x_q = Xa[step]

    j1d = np.arange(max(p - 1, 0), step)
    if variant == "typed_pool":
        j1d = j1d[pt1d[j1d] == cur_type]
    v1d  = ~np.any(np.isnan(Xa[j1d]), axis=1) & (j1d + H < n1d)
    idx1 = j1d[v1d]

    Xp     = list(Xa[idx1])
    yp_abs = list(p1d[idx1 + H])
    yp_src = list(p1d[idx1])

    j10 = np.arange(max(p - 1, 0), min(ce, na - H))
    if variant == "typed_pool":
        j10 = j10[pt10m[j10] == cur_type]
    if len(j10):
        v10   = ~np.any(np.isnan(Xb[j10]), axis=1)
        idx10 = j10[v10]
        Xp.extend(Xb[idx10])
        yp_abs.extend(p10m[idx10 + H])
        yp_src.extend(p10m[idx10])

    if len(Xp) < p + 2:
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


def ensemble_predict(res_lwr, res_sx, p_cur):
    """4-way ансамбль по готовым пулам."""
    y_lwr = y_smap = y_sx = y_rbf = np.nan

    if res_lwr:
        Xn, xn, d, y_abs, y_lr, N = res_lwr
        # LWR
        k_eff = min(K_LWR, N)
        ord_  = np.argsort(d); knn = ord_[:k_eff]; xi = d[ord_[k_eff - 1]]
        if xi < 1e-12:
            y_lwr = float(y_abs[knn].mean())
        else:
            w = np.exp(-0.5 * (d[knn] / xi) ** 2); ws = np.sqrt(w)
            A = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
            c, *_ = np.linalg.lstsq(A, y_abs[knn] * ws, rcond=None)
            y_lwr = float(c[0] + c[1:] @ xn)
        # S-map
        mean_d = d.mean() + 1e-12
        w_sm   = np.exp(-THETA * d / mean_d); ws_sm = np.sqrt(w_sm)
        A_sm   = np.column_stack([np.ones(N), Xn]) * ws_sm[:, None]
        c_sm, *_ = np.linalg.lstsq(A_sm, y_abs * ws_sm, rcond=None)
        y_smap = float(c_sm[0] + c_sm[1:] @ xn)
        # RBF-log
        if N >= K_RBF:
            k   = min(K_RBF, N)
            knn = np.argsort(d)[:k]; xi = d[knn[-1]]
            if xi < 1e-12:
                y_rbf = p_cur * float(np.exp(y_lr[knn].mean()))
            else:
                w = np.exp(-0.5 * (d[knn] / xi) ** 2)
                y_rbf = p_cur * float(np.exp((w @ y_lr[knn]) / w.sum()))

    if res_sx and res_sx[5] >= K_SX:
        Xn_sx, xn_sx, d_sx, y_abs_sx, y_lr_sx, N_sx = res_sx
        ords = np.argsort(d_sx); knn = ords[:K_SX]; d1 = d_sx[ords[0]]
        if d1 < 1e-12:
            y_sx = p_cur * float(np.exp(y_lr_sx[ords[0]]))
        else:
            w = np.exp(-d_sx[knn] / d1); w /= w.sum()
            y_sx = p_cur * float(np.exp(w @ y_lr_sx[knn]))

    return A_LWR * y_lwr + A_SX * y_sx + A_SM * y_smap + A_RBF * y_rbf


def run_variant(variant, p1d, dt1d, pt1d, p10m, dt10m, pt10m,
                X1d_p3, X10m_p3, X1d_p8, X10m_p8,
                X1d_p3_tf, X10m_p3_tf, X1d_p8_tf, X10m_p8_tf,
                n1d, na):
    errs, acts, types_out, pool_szs = [], [], [], []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X1d_p3[step])) or np.any(np.isnan(X1d_p8[step])):
            continue

        ce       = int(np.searchsorted(dt10m, dt1d[step], side="left"))
        p_cur    = float(p1d[step])
        cur_type = int(pt1d[step])

        kw = dict(step=step, ce=ce, cur_type=cur_type, variant=variant,
                  p1d=p1d, p10m=p10m, pt1d=pt1d, pt10m=pt10m, n1d=n1d, na=na)

        res_lwr = get_pool(P_LWR, X1d=X1d_p3, X10m=X10m_p3,
                           X1d_tf=X1d_p3_tf, X10m_tf=X10m_p3_tf, **kw)
        res_sx  = get_pool(P_SX,  X1d=X1d_p8, X10m=X10m_p8,
                           X1d_tf=X1d_p8_tf, X10m_tf=X10m_p8_tf, **kw)

        pred = ensemble_predict(res_lwr, res_sx, p_cur)
        if np.isnan(pred):
            continue

        actual = float(p1d[step + H])
        errs.append(pred - actual)
        acts.append(actual)
        types_out.append(cur_type)
        pool_szs.append(res_lwr[5] if res_lwr else 0)

    return (np.array(errs), np.array(acts),
            np.array(types_out, dtype=int), np.array(pool_szs))


def rmae_global(errs, global_dz):
    return float(np.mean(np.abs(errs)) / global_dz) if global_dz > 1e-12 else np.nan


def main():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")

    p1d,  dt1d,  pt1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m, pt10m = find_pivots(h10m, l10m, d10m, T_10M)

    n1d = len(p1d); na = len(p10m)
    print(f"SBER: 1d {n1d} пив  10m {na} пив")
    print(f"  1d  HIGH={( pt1d==1).sum()}  LOW={( pt1d==-1).sum()}")
    print(f"  10m HIGH={(pt10m==1).sum()}  LOW={(pt10m==-1).sum()}")

    # Стандартные матрицы признаков (без type feature)
    X1d_p3  = build_X(p1d,  P_LWR)
    X10m_p3 = build_X(p10m, P_LWR)
    X1d_p8  = build_X(p1d,  P_SX)
    X10m_p8 = build_X(p10m, P_SX)

    # Расширенные матрицы признаков (с type feature ±0.5)
    X1d_p3_tf  = build_X_tf(p1d,  pt1d,  P_LWR)
    X10m_p3_tf = build_X_tf(p10m, pt10m, P_LWR)
    X1d_p8_tf  = build_X_tf(p1d,  pt1d,  P_SX)
    X10m_p8_tf = build_X_tf(p10m, pt10m, P_SX)

    common = dict(p1d=p1d, dt1d=dt1d, pt1d=pt1d, p10m=p10m, dt10m=dt10m, pt10m=pt10m,
                  X1d_p3=X1d_p3, X10m_p3=X10m_p3, X1d_p8=X1d_p8, X10m_p8=X10m_p8,
                  X1d_p3_tf=X1d_p3_tf, X10m_p3_tf=X10m_p3_tf,
                  X1d_p8_tf=X1d_p8_tf, X10m_p8_tf=X10m_p8_tf,
                  n1d=n1d, na=na)

    results = {}
    for var in VARIANTS:
        print(f"Запуск: {var} ...", flush=True)
        results[var] = run_variant(var, **common)
        r_tmp = rmae_global(results[var][0],
                            float(np.mean(np.abs(np.diff(results[var][1])))))
        print(f"  → rMAE={r_tmp:.4f}")

    # Общий знаменатель — из baseline (одинаковые acts для сравнимости)
    base_acts = results["baseline"][1]
    global_dz = float(np.mean(np.abs(np.diff(base_acts))))
    base_r    = rmae_global(results["baseline"][0], global_dz)

    print(f"\n{'═'*58}")
    print(f"  {'Вариант':<15}  {'rMAE':>8}  {'Δ vs baseline':>13}  {'пул avg':>8}")
    print(f"{'─'*58}")
    for var in VARIANTS:
        errs, acts, types_out, pool_szs = results[var]
        r     = rmae_global(errs, global_dz)
        delta = (r - base_r) / base_r * 100
        print(f"  {var:<15}  {r:.4f}    {delta:+.2f}%           {pool_szs.mean():>7.0f}")

    print(f"\n{'─'*58}")
    print("  rMAE по типу перехода (знаменатель = global_dz):")
    print(f"  {'Вариант':<15}  {'HIGH→LOW':>10}  {'LOW→HIGH':>10}  {'n_HL':>5}  {'n_LH':>5}")
    print(f"{'─'*58}")
    for var in VARIANTS:
        errs, acts, types_out, pool_szs = results[var]
        m_hl  = types_out ==  1
        m_lh  = types_out == -1
        r_hl  = rmae_global(errs[m_hl], global_dz)
        r_lh  = rmae_global(errs[m_lh], global_dz)
        print(f"  {var:<15}  {r_hl:>10.4f}  {r_lh:>10.4f}  {m_hl.sum():>5}  {m_lh.sum():>5}")

    print(f"\n{'─'*58}")
    print("  Средний размер пула по вариантам:")
    for var in VARIANTS:
        errs, acts, types_out, pool_szs = results[var]
        print(f"  {var:<15}  avg={pool_szs.mean():.0f}  "
              f"min={pool_szs.min():.0f}  p10={np.percentile(pool_szs,10):.0f}")


if __name__ == "__main__":
    main()
