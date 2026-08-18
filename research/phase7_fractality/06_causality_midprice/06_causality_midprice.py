#!/usr/bin/env python3
"""
06_causality_midprice.py — Две проверки алгоритма 05_lwr_best

ТЕСТ 1 — Жёсткий тест каузальности
─────────────────────────────────────────────────────────────────────
Контракт каузальности: алгоритм в точке step НЕ видит ничего после
бара подтверждения T_BIG-пивота.

Проблема в текущей реализации:
  - Дата пивота = dates[ext_idx] — день достижения экстремума
  - Подтверждение = бар i, где ext_val - ll[i] >= thr (для HIGH→LOW)
  - Между ext_idx и i может пройти несколько баров
  - За это время в пул попадают T_FRAC-пивоты с датой между ext_idx и i

Hard-cutoff: используем дату ПОДТВЕРЖДЕНИЯ (confirm_date) как cutoff
пула вместо даты пивота.

Метрика: Δ rMAE = rMAE(hard) - rMAE(soft)
  Δ > 0 — была утечка, hard-cutoff ухудшает
  Δ ≈ 0 — утечки нет, алгоритм каузален

ТЕСТ 2 — Midprice-зигзаг
─────────────────────────────────────────────────────────────────────
Midprice = (high + low) / 2 — единый ряд, без асимметрии high/low.
Зигзаг «точка-точка»: ищем экстремумы самого midprice.
Полезно для:
  - Переноса метода на нефинансовые ряды
  - Упрощения алгоритма (один ряд вместо двух)

Тест: сравниваем rMAE (midprice-зигзаг) vs rMAE (high/low-зигзаг).
Если близко — метод применим к generic 1D-рядам без потери качества.

Обе версии: midprice=(H+L)/2 и midprice=(O+C)/2 (если open доступен).

Параметры: T_BIG=0.04, T_FRAC=0.036, P=3, K=75, H=1 — как в 05_lwr_best.
"""
import json
import sys
import numpy as np
from pathlib import Path

HERE    = Path(__file__).parent
DATA    = HERE.parent.parent.parent / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

T_BIG   = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
T_FRAC  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.036
H       = 1
P       = 3
K       = 75
MIN_HISTORY = 50


# ── данные ───────────────────────────────────────────────────────────────────
def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    highs  = np.array([d["high"]  for d in raw], dtype=np.float64)
    lows   = np.array([d["low"]   for d in raw], dtype=np.float64)
    opens  = np.array([d["open"]  for d in raw], dtype=np.float64)
    closes = np.array([d["close"] for d in raw], dtype=np.float64)
    dates  = np.array([d["begin"] for d in raw])
    return highs, lows, opens, closes, dates


# ── зигзаг: high/low, с трекингом даты подтверждения ─────────────────────────
def find_pivots_log_hl(highs, lows, dates, thr):
    """
    Зигзаг на log(high)/log(low).
    Возвращает:
        vals_log     : log-цены пивотов
        pivot_dates  : даты пивотов (день экстремума)
        confirm_dates: даты подтверждения (день разворота)
        dirs         : +1=HIGH, -1=LOW
    """
    lh, ll = np.log(highs), np.log(lows)
    vals_log, pivot_dates, confirm_dates, dirs = [], [], [], []
    direction, ext_val, ext_idx = 0, (lh[0] + ll[0]) / 2.0, 0
    for i in range(len(highs)):
        if direction == 0:
            if lh[i] - ext_val >= thr:
                direction, ext_val, ext_idx = 1, lh[i], i
            elif ext_val - ll[i] >= thr:
                direction, ext_val, ext_idx = -1, ll[i], i
        elif direction == 1:
            if lh[i] > ext_val:
                ext_val, ext_idx = lh[i], i
            elif ext_val - ll[i] >= thr:
                vals_log.append(ext_val)
                pivot_dates.append(dates[ext_idx])
                confirm_dates.append(dates[i])          # ← дата подтверждения
                dirs.append(+1)
                direction, ext_val, ext_idx = -1, ll[i], i
        else:
            if ll[i] < ext_val:
                ext_val, ext_idx = ll[i], i
            elif lh[i] - ext_val >= thr:
                vals_log.append(ext_val)
                pivot_dates.append(dates[ext_idx])
                confirm_dates.append(dates[i])          # ← дата подтверждения
                dirs.append(-1)
                direction, ext_val, ext_idx = 1, lh[i], i
    return (np.array(vals_log), np.array(pivot_dates),
            np.array(confirm_dates), np.array(dirs))


# ── зигзаг: midprice=(H+L)/2 или (O+C)/2 ────────────────────────────────────
def find_pivots_log_mid(mid, dates, thr):
    """
    Зигзаг «точка-точка» на log(midprice).
    Пивот — локальный максимум/минимум самого midprice.
    Возвращает vals_log, pivot_dates, confirm_dates, dirs.
    """
    lm = np.log(mid)
    vals_log, pivot_dates, confirm_dates, dirs = [], [], [], []
    direction, ext_val, ext_idx = 0, lm[0], 0
    for i in range(len(mid)):
        if direction == 0:
            if lm[i] - ext_val >= thr:
                direction, ext_val, ext_idx = 1, lm[i], i
            elif ext_val - lm[i] >= thr:
                direction, ext_val, ext_idx = -1, lm[i], i
        elif direction == 1:
            if lm[i] > ext_val:
                ext_val, ext_idx = lm[i], i
            elif ext_val - lm[i] >= thr:
                vals_log.append(ext_val)
                pivot_dates.append(dates[ext_idx])
                confirm_dates.append(dates[i])
                dirs.append(+1)
                direction, ext_val, ext_idx = -1, lm[i], i
        else:
            if lm[i] < ext_val:
                ext_val, ext_idx = lm[i], i
            elif lm[i] - ext_val >= thr:
                vals_log.append(ext_val)
                pivot_dates.append(dates[ext_idx])
                confirm_dates.append(dates[i])
                dirs.append(-1)
                direction, ext_val, ext_idx = 1, lm[i], i
    return (np.array(vals_log), np.array(pivot_dates),
            np.array(confirm_dates), np.array(dirs))


# ── эмбеддинг и LWR ──────────────────────────────────────────────────────────
def build_X(log_prices, p):
    n = len(log_prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = log_prices[i]
        for lag in range(1, p):
            X[i, lag] = log_prices[i - lag + 1] - log_prices[i - lag]
    return X


def lwr_predict(xq, X_sel, y_sel):
    d  = np.linalg.norm(X_sel - xq, axis=1)
    xi = d.max()
    if xi < 1e-12:
        return float(y_sel.mean())
    w  = np.exp(-0.5 * (d / xi) ** 2)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(len(d)), X_sel]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, y_sel * ws, rcond=None)
    return float(c[0] + c[1:] @ xq)


def rmae(errs, acts):
    acts = np.asarray(acts, dtype=float)
    pers = np.abs(acts[2:] - acts[:-2]) if len(acts) >= 3 else np.abs(np.diff(acts))
    dz   = float(np.mean(pers)) if len(pers) > 0 else 1.0
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


# ── walk-forward (общий) ─────────────────────────────────────────────────────
def walk_forward(lp_big, dt_big, dir_big, n_big,
                 lp_f1,  dt_f1,  dir_f1,  n_f1,
                 cutoff_dates=None):
    """
    cutoff_dates: если None — использует dt_big (мягкий, как в exp.05).
                  Если передан массив len=n_big — строгий cutoff по confirm-датам.
    """
    X_big = build_X(lp_big, P)
    Xf1   = build_X(lp_f1,  P)
    y_rel = np.array([
        lp_f1[j + H] - lp_f1[j] if j + H < n_f1 else np.nan
        for j in range(n_f1)
    ])

    if cutoff_dates is None:
        cutoff = dt_big
    else:
        cutoff = cutoff_dates

    ce_all = np.searchsorted(dt_f1, cutoff, side='left')

    errs, acts, pool_sizes, confirm_lags = [], [], [], []

    for step in range(MIN_HISTORY, n_big - H):
        if np.any(np.isnan(X_big[step])):
            continue

        ce  = int(ce_all[step])
        rng = np.arange(P - 1, min(ce, n_f1 - H))
        if len(rng) == 0:
            continue
        v   = ~np.any(np.isnan(Xf1[rng]), axis=1)
        idx = rng[v]
        if len(idx) < P + 2:
            continue

        q_dir = int(dir_big[step])
        idx_d = idx[dir_f1[idx] == q_dir]
        if len(idx_d) < P + 2:
            idx_d = idx

        Xs  = Xf1[idx_d][:, 1:]
        xqs = X_big[step][1:]

        dists = np.linalg.norm(Xs - xqs, axis=1)
        keff  = min(K, len(dists))
        knn   = np.argsort(dists)[:keff]
        sel   = idx_d[knn]

        pred_rel = lwr_predict(xqs, Xs[knn], y_rel[sel])
        if not np.isfinite(pred_rel):
            continue

        pred   = float(np.exp(lp_big[step] + pred_rel))
        actual = float(np.exp(lp_big[step + H]))

        errs.append(pred - actual)
        acts.append(actual)
        pool_sizes.append(len(idx_d))

        # диагностика: сколько пивотов из пула попало бы при hard-cutoff
        if cutoff_dates is None:
            ce_hard = int(np.searchsorted(dt_f1, cutoff_dates[step]
                                          if cutoff_dates is not None else dt_big[step],
                                          side='left'))
        confirm_lags.append(ce - int(np.searchsorted(dt_f1, dt_big[step], side='left'))
                            if cutoff_dates is not None else 0)

    return np.array(errs), np.array(acts), np.array(pool_sizes)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    highs, lows, opens, closes, dates = load_tf("10m")

    print(f"SBER 10m | T_big={T_BIG*100:.1f}% ({(np.exp(T_BIG)-1)*100:.2f}% в ценах) "
          f"| T_frac={T_FRAC*100:.1f}% | K={K} P={P} H={H}")
    print("=" * 70)

    # ══ ТЕСТ 1: каузальность ════════════════════════════════════════════════
    print("\n[ТЕСТ 1] Жёсткий тест каузальности")
    print("-" * 70)

    lp_big, dt_big, dt_big_confirm, dir_big = find_pivots_log_hl(highs, lows, dates, T_BIG)
    lp_f1,  dt_f1,  dt_f1_confirm,  dir_f1  = find_pivots_log_hl(highs, lows, dates, T_FRAC)
    n_big, n_f1 = len(lp_big), len(lp_f1)

    print(f"T_BIG: {n_big} пивотов | T_FRAC: {n_f1} пивотов")

    # Лаг подтверждения пивотов T_BIG (баров между экстремумом и разворотом)
    big_lag = np.searchsorted(dates, dt_big_confirm) - np.searchsorted(dates, dt_big)
    print(f"T_BIG confirm lag: median={int(np.median(big_lag))} баров, "
          f"max={int(np.max(big_lag))}, "
          f"p95={int(np.percentile(big_lag, 95))}")

    # Soft (как в exp.05: cutoff = дата пивота)
    e_soft, a_soft, ps_soft = walk_forward(
        lp_big, dt_big, dir_big, n_big,
        lp_f1,  dt_f1,  dir_f1,  n_f1,
        cutoff_dates=None
    )
    r_soft = rmae(e_soft, a_soft)
    print(f"\nSoft cutoff (pivot date): rMAE = {r_soft:.4f}   n={len(e_soft)}")

    # Hard (cutoff = дата подтверждения разворота)
    e_hard, a_hard, ps_hard = walk_forward(
        lp_big, dt_big_confirm, dir_big, n_big,
        lp_f1,  dt_f1,          dir_f1,  n_f1,
        cutoff_dates=dt_big_confirm
    )
    r_hard = rmae(e_hard, a_hard)
    print(f"Hard cutoff (confirm date): rMAE = {r_hard:.4f}   n={len(e_hard)}")

    delta_caus = (r_hard - r_soft) / r_soft * 100
    if delta_caus > 1.0:
        verdict = "⚠ Есть утечка: soft использует будущие данные"
    elif delta_caus < -1.0:
        verdict = "✓ Нет утечки: soft был излишне консервативен (hard открывает легитимные данные)"
    else:
        verdict = "✓ Нет утечки: разница незначима"
    print(f"\nΔ rMAE (hard - soft) = {delta_caus:+.2f}%   {verdict}")
    print(f"Pool avg: soft={ps_soft.mean():.0f}  hard={ps_hard.mean():.0f}")

    # ══ ТЕСТ 2: midprice-зигзаг ════════════════════════════════════════════
    print("\n[ТЕСТ 2] Midprice-зигзаг")
    print("-" * 70)

    mid_hl = (highs + lows) / 2.0
    mid_oc = (opens + closes) / 2.0

    for label, mid in [("(H+L)/2", mid_hl), ("(O+C)/2", mid_oc)]:
        lp_big_m, dt_big_m, dt_big_cm, dir_big_m = find_pivots_log_mid(mid, dates, T_BIG)
        lp_f1_m,  dt_f1_m,  dt_f1_cm,  dir_f1_m  = find_pivots_log_mid(mid, dates, T_FRAC)
        n_big_m, n_f1_m = len(lp_big_m), len(lp_f1_m)

        print(f"\nMidprice {label}: n_big={n_big_m} пивотов | n_f1={n_f1_m} пивотов")

        # Soft cutoff (для сравнимости с тестом 1 soft)
        e_m, a_m, ps_m = walk_forward(
            lp_big_m, dt_big_m, dir_big_m, n_big_m,
            lp_f1_m,  dt_f1_m,  dir_f1_m,  n_f1_m,
            cutoff_dates=None
        )
        r_m = rmae(e_m, a_m)
        delta_m = (r_m - r_soft) / r_soft * 100
        print(f"  rMAE = {r_m:.4f}   n={len(e_m)}   vs H/L soft: {delta_m:+.1f}%")

        # Hard cutoff
        e_mh, a_mh, ps_mh = walk_forward(
            lp_big_m, dt_big_cm, dir_big_m, n_big_m,
            lp_f1_m,  dt_f1_m,   dir_f1_m,  n_f1_m,
            cutoff_dates=dt_big_cm
        )
        r_mh = rmae(e_mh, a_mh)
        delta_mh = (r_mh - r_hard) / r_hard * 100
        print(f"  rMAE (hard) = {r_mh:.4f}   n={len(e_mh)}   vs H/L hard: {delta_mh:+.1f}%")

    # ══ сводка ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("СВОДКА")
    print(f"  Референс 4-way ensemble:        rMAE = 0.3911")
    print(f"  05_lwr_best (H/L, soft):        rMAE = 0.3762  (baseline этого теста)")
    print(f"  Тест 1 soft (H/L, pivot date):  rMAE = {r_soft:.4f}")
    print(f"  Тест 1 hard (H/L, confirm):     rMAE = {r_hard:.4f}   Δ={delta_caus:+.2f}%")
    print("=" * 70)

    # ── сохранение ──────────────────────────────────────────────────────────
    import csv
    rows = [
        {"test": "T1_soft_HL", "rMAE": r_soft, "n": len(e_soft), "pool_avg": ps_soft.mean()},
        {"test": "T1_hard_HL", "rMAE": r_hard, "n": len(e_hard), "pool_avg": ps_hard.mean()},
    ]
    for label, mid in [("HL", mid_hl), ("OC", mid_oc)]:
        lp_big_m, dt_big_m, dt_big_cm, dir_big_m = find_pivots_log_mid(mid, dates, T_BIG)
        lp_f1_m,  dt_f1_m,  _,          dir_f1_m  = find_pivots_log_mid(mid, dates, T_FRAC)
        n_big_m, n_f1_m = len(lp_big_m), len(lp_f1_m)
        e_s, a_s, ps_s = walk_forward(lp_big_m, dt_big_m, dir_big_m, n_big_m,
                                       lp_f1_m, dt_f1_m, dir_f1_m, n_f1_m)
        e_h, a_h, ps_h = walk_forward(lp_big_m, dt_big_cm, dir_big_m, n_big_m,
                                       lp_f1_m, dt_f1_m, dir_f1_m, n_f1_m,
                                       cutoff_dates=dt_big_cm)
        rows.append({"test": f"T2_soft_{label}", "rMAE": rmae(e_s, a_s),
                     "n": len(e_s), "pool_avg": float(ps_s.mean())})
        rows.append({"test": f"T2_hard_{label}", "rMAE": rmae(e_h, a_h),
                     "n": len(e_h), "pool_avg": float(ps_h.mean())})

    out = RESULTS / f"causality_midprice_T{int(T_BIG*100)}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "rMAE", "n", "pool_avg"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults → {out}")


if __name__ == "__main__":
    main()
