#!/usr/bin/env python3
"""
zigzag_forecast_ref.py — РЕФЕРЕНС-РЕАЛИЗАЦИЯ (актуально 2026-06-28)

Событийный аттрактор зигзага: 4-way ансамбль на SBER 10m.

Парадигма
─────────
  Целевой ряд     : пивоты T=4%  на 10m  (~776 пивотов)
  Фрактальный пул : пивоты T=0.6% на 10m (~22 555 пивотов, ratio ≈ 7×)
  Горизонт h      : 1 событие вперёд (следующий пивот)
  Вложение        : X = [price, lr₁, lr₂]  (p=3), hybrid (value + log-returns)

Ансамбль
────────
  LWR       p=3  K=50   α=0.05
  Simplex   p=8  k=9    α=0.35
  S-map     θ=1  p=3    α=0.20  (K = весь пул)
  RBF-log   p=3  K=12   α=0.40

Результаты (SBER, подтверждены прогоном 2026-06-28)
────────────────────────────────────────────────────
  M0 (random walk)          rMAE = 1.0000
  only_t4  (без фрактала)   rMAE = 0.3960   (−60.4% vs M0)
  plus_frac (+T=0.6% пул)   rMAE = 0.3911   (−60.9% vs M0)

Данные
──────
  data/candles/SBER/10m.json  (MOEX ISS, закрытые бары)
"""
import json
import numpy as np
from pathlib import Path

# ── пути ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DATA = HERE.parent.parent / "data" / "candles" / "SBER"

# ── параметры ─────────────────────────────────────────────────────────────────
T_BIG       = 0.04    # целевые события (крупные развороты, ~4%)
T_SMALL     = 0.006   # фрактальный пул (~0.6%, ratio ≈ 7×)
H           = 1       # горизонт прогноза (1 событие)
MIN_HISTORY = 50      # минимум событий в истории до старта walk-forward

P_LWR = 3;  K_LWR = 50         # LWR
P_SX  = 8;  K_SX  = P_SX + 1  # Simplex
THETA = 1.0                     # S-map bandwidth
P_RBF = 3;  K_RBF = 12         # RBF

A_LWR = 0.05; A_SX = 0.35; A_SM = 0.20; A_RBF = 0.40  # веса ансамбля


# ── загрузка данных ───────────────────────────────────────────────────────────
def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


# ── зигзаг ───────────────────────────────────────────────────────────────────
def find_pivots(highs, lows, dates, thr):
    """Каузальный зигзаг по порогу T (доля цены)."""
    vals, dts = [], []
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
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                direction = 1; ext_val, ext_idx = highs[i], i
    return np.array(vals), np.array(dts)


# ── вложение ─────────────────────────────────────────────────────────────────
def build_X(prices, p):
    """Hybrid embedding: X[i] = [price(i), lr(i→i-1), lr(i-1→i-2), ...]"""
    n  = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


# ── пул + нормализация ────────────────────────────────────────────────────────
def get_pool(step, ce, X_big, X_small, p_big, p_small,
             n_big, n_small, p_lag, use_fractal):
    """
    Собирает пул соседей для шага step.
    Каузальность: пул включает только события строго до step (по индексу и дате).
    Возвращает (Xn, xn, d, y_abs, y_lr, N) или None.
    """
    x_q = X_big[step]
    if np.any(np.isnan(x_q)):
        return None

    j_big = np.arange(max(p_lag - 1, 0), step)
    v_big = ~np.any(np.isnan(X_big[j_big]), axis=1) & (j_big + H < n_big)
    idx_b = j_big[v_big]
    Xp     = list(X_big[idx_b])
    yp_abs = list(p_big[idx_b + H])
    yp_src = list(p_big[idx_b])

    if use_fractal and ce > 0:
        j_sm = np.arange(max(p_lag - 1, 0), min(ce, n_small - H))
        if len(j_sm):
            v_sm  = ~np.any(np.isnan(X_small[j_sm]), axis=1)
            idx_s = j_sm[v_sm]
            Xp.extend(X_small[idx_s])
            yp_abs.extend(p_small[idx_s + H])
            yp_src.extend(p_small[idx_s])

    n_feat = x_q.shape[0]
    if len(Xp) < n_feat + 2:
        return None

    X_pool = np.array(Xp)
    y_abs  = np.array(yp_abs, dtype=float)
    y_lr   = np.log(y_abs / np.array(yp_src, dtype=float))

    # joint z-score по всему пулу + запросу (каузальный)
    mu  = X_pool.mean(0)
    sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
    Xn  = (X_pool - mu) / sig
    xn  = (x_q    - mu) / sig
    d   = np.linalg.norm(Xn - xn, axis=1)
    return Xn, xn, d, y_abs, y_lr, X_pool.shape[0]


# ── один шаг ансамбля ────────────────────────────────────────────────────────
def ensemble_step(res_lwr, res_sx, p_cur):
    y_lwr = y_smap = y_sx = y_rbf = np.nan

    if res_lwr:
        Xn, xn, d, y_abs, y_lr, N = res_lwr
        k_eff = min(K_LWR, N)
        ord_  = np.argsort(d)
        knn   = ord_[:k_eff]
        xi    = d[ord_[k_eff - 1]]
        if xi < 1e-12:
            y_lwr = float(y_abs[knn].mean())
        else:
            w  = np.exp(-0.5*(d[knn]/xi)**2); ws = np.sqrt(w)
            A  = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:,None]
            c, *_ = np.linalg.lstsq(A, y_abs[knn]*ws, rcond=None)
            y_lwr = float(c[0] + c[1:] @ xn)

        mean_d  = d.mean() + 1e-12
        w_sm    = np.exp(-THETA*d/mean_d); ws_sm = np.sqrt(w_sm)
        A_sm    = np.column_stack([np.ones(N), Xn]) * ws_sm[:,None]
        c_sm, *_ = np.linalg.lstsq(A_sm, y_abs*ws_sm, rcond=None)
        y_smap  = float(c_sm[0] + c_sm[1:] @ xn)

        if N >= K_RBF:
            knn_r = np.argsort(d)[:K_RBF]
            xi_r  = d[knn_r[-1]]
            if xi_r < 1e-12:
                y_rbf = p_cur * float(np.exp(y_lr[knn_r].mean()))
            else:
                w_r   = np.exp(-0.5*(d[knn_r]/xi_r)**2)
                y_rbf = p_cur * float(np.exp((w_r @ y_lr[knn_r]) / w_r.sum()))

    if res_sx and res_sx[5] >= K_SX:
        Xn_s, xn_s, d_s, y_abs_s, y_lr_s, N_s = res_sx
        ords  = np.argsort(d_s)
        knn_s = ords[:K_SX]
        d1    = d_s[ords[0]]
        if d1 < 1e-12:
            y_sx = p_cur * float(np.exp(y_lr_s[ords[0]]))
        else:
            w_s  = np.exp(-d_s[knn_s]/d1); w_s /= w_s.sum()
            y_sx = p_cur * float(np.exp(w_s @ y_lr_s[knn_s]))

    return A_LWR*y_lwr + A_SX*y_sx + A_SM*y_smap + A_RBF*y_rbf


# ── walk-forward ──────────────────────────────────────────────────────────────
def run(p_big, dt_big, X_big_p3, X_big_p8,
        p_small, dt_small, X_sm_p3, X_sm_p8,
        n_big, n_small, use_fractal):
    errs, acts, pool_szs = [], [], []
    for step in range(MIN_HISTORY, n_big - H):
        if np.any(np.isnan(X_big_p3[step])) or np.any(np.isnan(X_big_p8[step])):
            continue
        ce    = int(np.searchsorted(dt_small, dt_big[step], side="left"))
        p_cur = float(p_big[step])

        res_lwr = get_pool(step, ce, X_big_p3, X_sm_p3, p_big, p_small,
                           n_big, n_small, P_LWR, use_fractal)
        res_sx  = get_pool(step, ce, X_big_p8, X_sm_p8, p_big, p_small,
                           n_big, n_small, P_SX, use_fractal)

        pred = ensemble_step(res_lwr, res_sx, p_cur)
        if np.isnan(pred):
            continue

        errs.append(pred - float(p_big[step + H]))
        acts.append(float(p_big[step + H]))
        pool_szs.append(res_lwr[5] if res_lwr else 0)

    return np.array(errs), np.array(acts), np.array(pool_szs)


def rmae(errs, acts):
    dz = float(np.mean(np.abs(np.diff(acts))))
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    h10m, l10m, d10m = load_tf("10m")

    p_big,   dt_big   = find_pivots(h10m, l10m, d10m, T_BIG)
    p_small, dt_small = find_pivots(h10m, l10m, d10m, T_SMALL)
    n_big = len(p_big); n_small = len(p_small)

    print(f"SBER 10m  T_big={T_BIG*100:.1f}%  T_small={T_SMALL*100:.2f}%")
    print(f"  крупные: {n_big} пивотов   мелкие: {n_small} пивотов"
          f"   ratio={n_small/n_big:.1f}×")

    X_big_p3  = build_X(p_big,   P_LWR)
    X_big_p8  = build_X(p_big,   P_SX)
    X_small_p3 = build_X(p_small, P_LWR)
    X_small_p8 = build_X(p_small, P_SX)

    print("only_t4  ...", flush=True)
    e1, a1, ps1 = run(p_big, dt_big, X_big_p3, X_big_p8,
                      p_small, dt_small, X_small_p3, X_small_p8,
                      n_big, n_small, use_fractal=False)
    r1 = rmae(e1, a1)
    print(f"  rMAE={r1:.4f}   пул avg={ps1.mean():.0f}")

    print("plus_frac ...", flush=True)
    e2, a2, ps2 = run(p_big, dt_big, X_big_p3, X_big_p8,
                      p_small, dt_small, X_small_p3, X_small_p8,
                      n_big, n_small, use_fractal=True)
    r2 = rmae(e2, a2)
    print(f"  rMAE={r2:.4f}   пул avg={ps2.mean():.0f}")

    print(f"\n{'═'*55}")
    print(f"  M0                         rMAE = 1.0000")
    print(f"  only_t4  (без фрактала)    rMAE = {r1:.4f}   ({(1-r1)*100:.1f}% лучше M0)")
    print(f"  plus_frac (+T_small пул)   rMAE = {r2:.4f}   ({(1-r2)*100:.1f}% лучше M0)")
    print(f"  Прирост от фрактального пула: {(r1-r2)/r1*100:+.2f}%")
    print(f"{'═'*55}")


if __name__ == "__main__":
    main()
