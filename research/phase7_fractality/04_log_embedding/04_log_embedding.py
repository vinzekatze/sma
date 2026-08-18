#!/usr/bin/env python3
"""
04_log_embedding.py — Сравнение эмбеддингов: linear vs log-price

Идея: работаем полностью в log-цене (= логарифмическая шкала графика).
  - Зигзаг ищется на log(high/low) с аддитивным порогом T
  - col 0 = log(pivot_price)      ← позиция на лог-графике
  - col 1 = lp[j] - lp[j-1]      ← простая разность в лог-пространстве
  - col 2 = lp[j-1] - lp[j-2]

LWR предсказывает log-цену следующего пивота → exp() → оригинальная цена.
rMAE вычисляется в исходных ценах (сравнимо с эксп.01: B1 K=50 = 0.4023).

T_big=4%, T_frac1=3%, K=50, P=3, H=1, SBER 10m
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent.parent.parent / "data" / "candles" / "SBER"

T_BIG   = 0.04
T_FRAC1 = 0.03

H           = 1
MIN_HISTORY = 50
P           = 3
K           = 50

B1_K50_REF = 0.4023   # из эксп.01, linear embedding


# ── данные ───────────────────────────────────────────────────────────────────
def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


# ── зигзаг: два варианта ─────────────────────────────────────────────────────
def find_pivots_linear(highs, lows, dates, thr):
    """Стандартный зигзаг: мультипликативный порог (% от текущей цены)."""
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


def find_pivots_log(highs, lows, dates, thr):
    """Зигзаг на log(price): аддитивный порог (= % в лог-пространстве)."""
    lh = np.log(highs)
    ll = np.log(lows)
    vals_log, dts = [], []
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
                vals_log.append(ext_val); dts.append(dates[ext_idx])
                direction, ext_val, ext_idx = -1, ll[i], i
        else:
            if ll[i] < ext_val:
                ext_val, ext_idx = ll[i], i
            elif lh[i] - ext_val >= thr:
                vals_log.append(ext_val); dts.append(dates[ext_idx])
                direction, ext_val, ext_idx = 1, lh[i], i
    return np.array(vals_log), np.array(dts)  # возвращает log-цены


# ── эмбеддинги ───────────────────────────────────────────────────────────────
def build_X_linear(prices, p):
    """col 0 = price, col 1,2 = log-returns."""
    n  = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def build_X_log(log_prices, p):
    """col 0 = log(price), col 1,2 = простые разности лог-цен."""
    n = len(log_prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = log_prices[i]
        for lag in range(1, p):
            X[i, lag] = log_prices[i - lag + 1] - log_prices[i - lag]
    return X


# ── LWR + метрики ────────────────────────────────────────────────────────────
def rmae(errs, acts):
    acts = np.asarray(acts, dtype=float)
    pers = np.abs(acts[2:] - acts[:-2]) if len(acts) >= 3 else np.abs(np.diff(acts))
    dz   = float(np.mean(pers)) if len(pers) > 0 else 1.0
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


def lwr_predict(xn_q, Xn_pool, y_pool):
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


# ── walk-forward B1 ──────────────────────────────────────────────────────────
def run_b1(X_big, dt_big, n_big,
           X_f1, dt_f1, n_f1,
           labels_f1, actuals_big,
           pred_transform=None):
    """
    B1: query X_big[step] → пул X_f1 → LWR → pred.
    labels_f1:   метки для пула (log-цены или обычные)
    actuals_big: фактические значения для ошибки (всегда обычные цены)
    pred_transform: функция для перевода предсказания в пространство actuals
    """
    errs, acts = [], []
    ce1_all = np.searchsorted(dt_f1, dt_big, side='left')

    for step in range(MIN_HISTORY, n_big - H):
        if np.any(np.isnan(X_big[step])):
            continue
        ce1 = int(ce1_all[step])
        rng = np.arange(ce1)
        idx = np.where(
            ~np.any(np.isnan(X_f1[:ce1]), axis=1)
            & (rng >= P - 1)
            & (rng + H < n_f1)
        )[0]
        if len(idx) < P + 2:
            continue

        X_pool = X_f1[idx]
        mu  = X_pool.mean(0)
        sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn  = (X_pool      - mu) / sig
        xn  = (X_big[step] - mu) / sig

        pred = lwr_predict(xn, Xn, labels_f1[idx + H])
        if np.isnan(pred):
            continue

        if pred_transform is not None:
            pred = pred_transform(pred)

        errs.append(pred - float(actuals_big[step + H]))
        acts.append(float(actuals_big[step + H]))

    return np.array(errs), np.array(acts)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    h, l, d = load_tf("10m")

    print(f"SBER 10m | T_big={T_BIG*100:.0f}% | T_f1={T_FRAC1*100:.0f}% | K={K} | P={P} | H={H}")

    # ── Linear B1 ────────────────────────────────────────────────────────────
    p_big_lin, dt_big_lin = find_pivots_linear(h, l, d, T_BIG)
    p_f1_lin,  dt_f1_lin  = find_pivots_linear(h, l, d, T_FRAC1)
    n_big_lin, n_f1_lin   = len(p_big_lin), len(p_f1_lin)

    X_big_lin = build_X_linear(p_big_lin, P)
    X_f1_lin  = build_X_linear(p_f1_lin,  P)

    e_lin, a_lin = run_b1(
        X_big_lin, dt_big_lin, n_big_lin,
        X_f1_lin,  dt_f1_lin,  n_f1_lin,
        labels_f1=p_f1_lin, actuals_big=p_big_lin)
    r_lin = rmae(e_lin, a_lin)

    print(f"\nLinear embedding:")
    print(f"  n_big={n_big_lin}  n_f1={n_f1_lin}")
    print(f"  rMAE = {r_lin:.4f}   n_steps={len(e_lin)}")

    # ── Log-price B1 ─────────────────────────────────────────────────────────
    # find_pivots_log возвращает log-цены пивотов
    lp_big, dt_big_log = find_pivots_log(h, l, d, T_BIG)
    lp_f1,  dt_f1_log  = find_pivots_log(h, l, d, T_FRAC1)
    n_big_log, n_f1_log = len(lp_big), len(lp_f1)

    X_big_log = build_X_log(lp_big, P)
    X_f1_log  = build_X_log(lp_f1,  P)

    # actuals в оригинальных ценах для сравнимости rMAE
    p_big_from_log = np.exp(lp_big)

    e_log, a_log = run_b1(
        X_big_log, dt_big_log, n_big_log,
        X_f1_log,  dt_f1_log,  n_f1_log,
        labels_f1=lp_f1,             # метки в лог-пространстве
        actuals_big=p_big_from_log,  # ошибка в исходных ценах
        pred_transform=np.exp)       # log-предсказание → цена
    r_log = rmae(e_log, a_log)

    print(f"\nLog-price embedding (T in log-space ≈ {(np.exp(T_BIG)-1)*100:.1f}% в ценах):")
    print(f"  n_big={n_big_log}  n_f1={n_f1_log}")
    print(f"  rMAE = {r_log:.4f}   n_steps={len(e_log)}")

    # ── итог ─────────────────────────────────────────────────────────────────
    delta = (r_log - r_lin) / r_lin * 100
    print(f"\n{'═'*54}")
    print(f"  Справка эксп.01 (linear, K=50):  rMAE = {B1_K50_REF:.4f}")
    print(f"  Linear  (этот прогон, K={K}):      rMAE = {r_lin:.4f}")
    print(f"  Log     (этот прогон, K={K}):      rMAE = {r_log:.4f}"
          f"   Δ={delta:+.1f}%")
    verdict = "НЕ ХУЖЕ ✓" if r_log <= r_lin * 1.005 else f"хуже на {delta:+.1f}%"
    print(f"  Вердикт: {verdict}")
    print(f"{'═'*54}")


if __name__ == "__main__":
    main()
