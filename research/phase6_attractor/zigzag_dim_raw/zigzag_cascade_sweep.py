#!/usr/bin/env python3
"""
Строгий октавный каскад на зигзаге: свип по числу уровней N.

Алгоритм (cascade_algorithm.md):
  p_levels = [p_fit × 2^(N-1), ..., p_fit × 2, p_fit]
  Уровень k: K соседей в пуле при p_levels[k] → суб-векторы при p_levels[k+1].
  Строгая воронка: пул k+1 = только суб-векторы отобранных на уровне k.
  K одинаков на всех уровнях.

Пул: SBER 1d (T=2%) + 1h (T=0.5%) + 10m (T=0.2%).
  N=1 (baseline): обычный LWR без каскада.

Параметры (оптимум из предыдущих экспериментов):
  p_fit=3, K=12, H=1, метрика L2 + z-score.

КАУЗАЛЬНОСТЬ: query и пул строго до текущего origin (step).
  1d: j ∈ [p-1, step-1]
  1h/10m: j ∈ [p-1, searchsorted(dates_aux, dates_1d[step]) - 1]
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR    = Path(__file__).parent
DATA        = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT         = BASE_DIR / "results"
OUT.mkdir(exist_ok=True)

T_1D        = 0.02
T_1H        = 0.005
T_10M       = 0.002
P_FIT       = 3
K           = 3 * (P_FIT + 1)   # 12
H           = 1
MIN_HISTORY = 50
N_MAX       = 5


# ─── Загрузка / зигзаг ───────────────────────────────────────────────────────

def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    h     = np.array([d["high"]  for d in raw], dtype=np.float64)
    l_    = np.array([d["low"]   for d in raw], dtype=np.float64)
    dates = np.array([d["begin"] for d in raw])
    return h, l_, dates


def find_pivots(highs, lows, dates, thr):
    pivots, direction = [], 0
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
                pivots.append((ext_val, dates[ext_idx]))
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                pivots.append((ext_val, dates[ext_idx]))
                direction = 1;  ext_val, ext_idx = highs[i], i
    if not pivots:
        return np.array([]), np.array([])
    return (np.array([pv for pv, _ in pivots]),
            np.array([d  for _, d  in pivots]))


# ─── Вложение ────────────────────────────────────────────────────────────────

def build_X(prices, p):
    """X[i] = [price[i], log(p[i]/p[i-1]), ..., log(p[i-p+2]/p[i-p+1])]"""
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def build_Xs(prices, p_list):
    return {p: build_X(prices, p) for p in p_list}


# ─── LWR / k-NN ──────────────────────────────────────────────────────────────

def knn_zscore(X_pool, x_q, k):
    """K ближайших по L2 в z-score пространстве пула."""
    if len(X_pool) < k:
        return None
    mu    = X_pool.mean(0)
    sigma = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
    Xn    = (X_pool - mu) / sigma
    xn    = (x_q    - mu) / sigma
    dists = np.linalg.norm(Xn - xn, axis=1)
    return np.argsort(dists)[:k]


def lwr(X_pool, y_pool, x_q, k):
    if len(X_pool) < k:
        return np.nan
    mu    = X_pool.mean(0)
    sigma = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
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


# ─── Пул: (tf_arr, loc_arr) — numpy-массивы индексов ────────────────────────
# tf: 0=1d, 1=1h, 2=10m

def build_pool(step, p, ce1h, ce10m, n_by_tf):
    """
    Возвращает (tf_arr, loc_arr): numpy-массивы int32, dtype int8.
    Каузально: 1d j<step; 1h/10m j<ce*.
    Граница j+H<n гарантируется через n_by_tf[tf]-H.
    """
    j0 = np.arange(p - 1, min(step, n_by_tf[0] - H), dtype=np.int32)
    j1 = np.arange(p - 1, min(ce1h, n_by_tf[1] - H), dtype=np.int32)
    j2 = np.arange(p - 1, min(ce10m, n_by_tf[2] - H), dtype=np.int32)
    tf_arr  = np.concatenate([
        np.zeros(len(j0), dtype=np.int8),
        np.ones (len(j1), dtype=np.int8),
        np.full (len(j2), 2, dtype=np.int8),
    ])
    loc_arr = np.concatenate([j0, j1, j2])
    return tf_arr, loc_arr


def extract_features(tf_arr, loc_arr, p, Xs_by_tf):
    """
    Векторизованное извлечение признаков из матриц Xs.
    Возвращает (X_pool, tf_valid, loc_valid) после удаления NaN-строк.
    """
    n = len(tf_arr)
    if n == 0:
        return np.empty((0, p)), np.empty(0, np.int8), np.empty(0, np.int32)

    X_out = np.full((n, p), np.nan, dtype=np.float64)
    for tf_id in range(3):
        mask = tf_arr == tf_id
        if not mask.any():
            continue
        X = Xs_by_tf[tf_id].get(p)
        if X is None:
            continue
        locs   = loc_arr[mask]
        in_rng = locs < len(X)
        idx_in = np.where(mask)[0][in_rng]
        if idx_in.size:
            X_out[idx_in] = X[locs[in_rng]]

    valid = ~np.any(np.isnan(X_out), axis=1)
    return X_out[valid], tf_arr[valid], loc_arr[valid]


# ─── Каскад ──────────────────────────────────────────────────────────────────

def cascade_step(tf_arr, loc_arr, level_p, next_p, query_vec, k, Xs_by_tf):
    """
    Один переход уровня: level_p → next_p.
    Возвращает (tf_arr_new, loc_arr_new) для пула при next_p.
    """
    X_pool, tf_v, loc_v = extract_features(tf_arr, loc_arr, level_p, Xs_by_tf)
    if len(X_pool) < k:
        return np.empty(0, np.int8), np.empty(0, np.int32)

    idx_k = knn_zscore(X_pool, query_vec, k)
    if idx_k is None:
        return np.empty(0, np.int8), np.empty(0, np.int32)

    n_sub = level_p - next_p + 1  # суб-векторов с одного соседа

    # Строим суб-векторы: для каждого из K соседей порождаем n_sub кандидатов
    # Кандидат k_sub: (tf_id, loc - k_sub), loc - k_sub ≥ next_p - 1
    tf_cands  = []
    loc_cands = []
    for idx in idx_k:
        tf_id = int(tf_v[idx])
        loc   = int(loc_v[idx])
        for k_sub in range(n_sub):
            sub_loc = loc - k_sub
            if sub_loc >= next_p - 1:
                tf_cands.append(tf_id)
                loc_cands.append(sub_loc)

    if not tf_cands:
        return np.empty(0, np.int8), np.empty(0, np.int32)

    tf_cands_arr  = np.array(tf_cands,  dtype=np.int8)
    loc_cands_arr = np.array(loc_cands, dtype=np.int32)

    # Дедупликация по (tf_id, loc)
    keys  = tf_cands_arr.astype(np.int64) * (10 ** 9) + loc_cands_arr
    _, ui = np.unique(keys, return_index=True)
    return tf_cands_arr[ui], loc_cands_arr[ui]


# ─── Один прогноз ────────────────────────────────────────────────────────────

def predict_cascade(step, n_levels, query_vecs, Xs_by_tf, prices_by_tf,
                    ce1h, ce10m, n_by_tf, k, h):
    """
    query_vecs: dict p → np.array (вектор запроса из 1d на step).
    Возвращает pred (float или nan) и actual.
    """
    actual   = float(prices_by_tf[0][step + h])
    p_levels = [P_FIT * (2 ** (n_levels - 1 - lv)) for lv in range(n_levels)]
    p_max    = p_levels[0]

    # Проверка query
    for p in p_levels:
        if p not in query_vecs:
            return np.nan, actual

    # Глобальный пул при p_max
    tf_arr, loc_arr = build_pool(step, p_max, ce1h, ce10m, n_by_tf)

    if n_levels == 1:
        # Прямой LWR без каскада
        X_pool, tf_v, loc_v = extract_features(tf_arr, loc_arr, P_FIT, Xs_by_tf)
        y_arr = np.array([float(prices_by_tf[int(tf_v[i])][int(loc_v[i]) + h])
                          for i in range(len(tf_v))])
        return lwr(X_pool, y_arr, query_vecs[P_FIT], k), actual

    # Каскад: N-1 уровней переходов
    for lv in range(len(p_levels) - 1):
        level_p = p_levels[lv]
        next_p  = p_levels[lv + 1]
        tf_arr, loc_arr = cascade_step(
            tf_arr, loc_arr, level_p, next_p, query_vecs[level_p], k, Xs_by_tf
        )
        if len(tf_arr) == 0:
            return np.nan, actual

    # Финальный LWR при p_fit
    X_pool, tf_v, loc_v = extract_features(tf_arr, loc_arr, P_FIT, Xs_by_tf)
    if len(X_pool) < k:
        return np.nan, actual
    y_arr = np.array([float(prices_by_tf[int(tf_v[i])][int(loc_v[i]) + h])
                      for i in range(len(tf_v))])
    return lwr(X_pool, y_arr, query_vecs[P_FIT], k), actual


# ─── Метрика и основной цикл ─────────────────────────────────────────────────

def rmae(preds, actuals):
    if len(preds) < 5:
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return np.nan if dz < 1e-12 else float(np.mean(np.abs(preds - actuals)) / dz)


def run():
    # Загрузка
    h1d,  l1d,  d1d  = load_tf("1d")
    h1h,  l1h,  d1h  = load_tf("1h")
    h10m, l10m, d10m = load_tf("10m")

    p1d,  dates_1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p1h,  dates_1h  = find_pivots(h1h,  l1h,  d1h,  T_1H)
    p10m, dates_10m = find_pivots(h10m, l10m, d10m, T_10M)

    print(f"SBER")
    print(f"  1d  T={T_1D *100:.0f}%  → {len(p1d )} пивотов")
    print(f"  1h  T={T_1H *100:.1f}% → {len(p1h )} пивотов")
    print(f"  10m T={T_10M*100:.1f}% → {len(p10m)} пивотов")
    print(f"p_fit={P_FIT}  K={K}  H={H}  N_MAX={N_MAX}")
    print()

    # Все нужные p: p_fit × {1, 2, 4, 8, 16}
    all_p = sorted({P_FIT * (2 ** exp) for exp in range(N_MAX)})
    print(f"Строим X для p ∈ {all_p} (3 таймфрейма)...", flush=True)
    Xs_by_tf     = [build_Xs(p1d, all_p), build_Xs(p1h, all_p), build_Xs(p10m, all_p)]
    prices_by_tf = [p1d, p1h, p10m]
    n_by_tf      = [len(p1d), len(p1h), len(p10m)]
    print("Готово.\n")

    # Предвычислить ce1h, ce10m для каждого шага в 1d
    ce1h_all  = np.searchsorted(dates_1h,  dates_1d, side="left")
    ce10m_all = np.searchsorted(dates_10m, dates_1d, side="left")

    n1d        = len(p1d)
    step_start = MIN_HISTORY   # = 50; p_max ≤ 48 < 50 для всех N

    records = []
    print(f"{'N':>3}  {'p_levels':>22}  {'rMAE':>8}  {'vs N=1':>8}  {'n_test':>7}  {'n_nan':>7}")
    print("─" * 68)

    r_N1 = None

    for n_levels in range(1, N_MAX + 1):
        p_levels = [P_FIT * (2 ** (n_levels - 1 - lv)) for lv in range(n_levels)]
        p_max    = p_levels[0]

        preds, actuals, n_nan = [], [], 0

        for step in range(step_start, n1d - H):
            # Запросные векторы из 1d (все уровни)
            query_vecs = {}
            ok = True
            for p in p_levels:
                X1 = Xs_by_tf[0].get(p)
                if (X1 is None or step >= len(X1)
                        or np.any(np.isnan(X1[step]))):
                    ok = False; break
                query_vecs[p] = X1[step]
            if not ok:
                n_nan += 1; continue

            pred, actual = predict_cascade(
                step, n_levels, query_vecs, Xs_by_tf, prices_by_tf,
                int(ce1h_all[step]), int(ce10m_all[step]), n_by_tf, K, H
            )
            if np.isnan(pred):
                n_nan += 1; continue
            preds.append(pred)
            actuals.append(actual)

        preds   = np.array(preds)
        actuals = np.array(actuals)
        r       = rmae(preds, actuals)

        if r_N1 is None:
            r_N1 = r; vs = "—"
        else:
            vs = f"{(r / r_N1 - 1)*100:+.1f}%"

        lv_str = "→".join(str(p) for p in p_levels)
        print(f"  N={n_levels}  {lv_str:>22}  {r:.4f}   {vs:>8}  "
              f"{len(preds):>7}  {n_nan:>7}", flush=True)

        records.append({
            "n_levels" : n_levels,
            "p_levels" : str(p_levels),
            "p_max"    : p_max,
            "rMAE"     : r,
            "vs_N1_pct": 0.0 if (r_N1 is None or r_N1 == r)
                         else (r / r_N1 - 1) * 100,
            "n_test"   : len(preds),
            "n_nan"    : n_nan,
        })

    df = pd.DataFrame(records)
    df.to_csv(OUT / "cascade_levels_sweep.csv", index=False)

    # ── Графики ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ns = df["n_levels"].values

    ax = axes[0]
    ax.plot(ns, df["rMAE"], "o-", color="steelblue", lw=2.5, ms=9)
    ax.axhline(df["rMAE"].iloc[0], color="gray", lw=1.2, ls="--",
               label=f"N=1 (baseline)  rMAE={df['rMAE'].iloc[0]:.4f}")
    for n, r in zip(ns, df["rMAE"]):
        ax.annotate(f"{r:.4f}", (n, r), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xlabel("N уровней каскада")
    ax.set_ylabel("rMAE")
    ax.set_title("rMAE vs число уровней")
    ax.set_xticks(ns)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.2)

    ax = axes[1]
    vs_pct = [(r / df["rMAE"].iloc[0] - 1) * 100 for r in df["rMAE"]]
    clrs   = ["steelblue" if v <= 0 else "crimson" for v in vs_pct]
    ax.bar(ns, vs_pct, color=clrs, alpha=0.85)
    ax.axhline(0, color="gray", lw=1)
    for n, v in zip(ns, vs_pct):
        ax.annotate(f"{v:+.1f}%", (n, v),
                    textcoords="offset points",
                    xytext=(0, 5 if v >= 0 else -12),
                    ha="center", fontsize=9)
    ax.set_xlabel("N уровней каскада")
    ax.set_ylabel("% изменение rMAE vs N=1")
    ax.set_title("Изменение vs baseline")
    ax.set_xticks(ns)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Строгий октавный каскад (p_fit×{{1,2,4,8,16}})  |  SBER\n"
        f"1d({T_1D*100:.0f}%) + 1h({T_1H*100:.1f}%) + 10m({T_10M*100:.1f}%)  "
        f"p_fit={P_FIT}  K={K}  H={H}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(OUT / "cascade_levels_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + график → {OUT}")


if __name__ == "__main__":
    run()
