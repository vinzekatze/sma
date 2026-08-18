#!/usr/bin/env python3
"""
lwr_lp_correction.py — Walk-forward LWR + LP-коррекция (log-return пространство).

Базовый LWR (T_query=4%, T_pool=3.6%, m=2, K=75): rMAE=0.3760

LP-коррекция (по zigzag_lp_correction3.py v3):
  Пространство: P_LP лог-доходностей пивотов T_query
  Прогнозный вектор: [y_hat_lr, lr1_cur, lr2_cur, ..., lr(P_LP-1)_cur]
  SVD k_lp соседей → d_lp главных компонент → проекция → lr1_corr → цена

Sweep: P_LP × k_lp × d_lp → матрица rMAE.

Результаты (SBER 10m, walk-forward 2013-2025):
  Лучшая LP-коррекция: P_LP=3, k=10, d=1 → rMAE=0.3754 (−0.2% vs базового LWR)
  Вывод: LP даёт слабый сигнал, но слишком мало пивотов для устойчивого выигрыша.
          Усложнять алгоритм нецелесообразно. Базовый LWR rMAE=0.3760 — финальный результат."""

Каузальность:
  Зигзаги строятся по ВСЕМ данным (каузальная функция — пивот определяется
  только по прошлому). Для каждого шага i:
    C2: pool_confirm_dates[j]   < query_confirm_dates[i]  (событие пула известно)
    C3: pool_confirm_dates[j+1] < query_confirm_dates[i]  (целевое событие тоже известно)
    LP: pool_confirm_dates[j]   < query_confirm_dates[i]  (LP-пул — только C2)
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── параметры ─────────────────────────────────────────────────────────────────
T_QUERY     = 0.04
T_POOL      = 0.036
M_EMBED     = 2       # размерность вложения LWR
K_LWR       = 75      # число соседей LWR
P_LP        = 6       # размерность LP-пространства (число лог-доходностей)
H           = 1       # горизонт прогноза (событий T_query)
MIN_HISTORY = 50      # минимум шагов до начала прогнозирования
MIN_DIR     = M_EMBED + 2  # минимум однонаправленных событий для LWR

P_LP_GRID = [2, 3, 4]
K_LP_GRID = [5, 10, 15, 20, 30, 50]

TICKER   = "SBER"
INTERVAL = "10m"

DATA = Path(__file__).parent.parent.parent / "data" / "candles"


# ── загрузка свечей ───────────────────────────────────────────────────────────

def load_log_candles(ticker: str, interval: str) -> tuple:
    path = DATA / ticker / f"{interval}.json"
    with open(path) as file_handle:
        raw_candles = json.load(file_handle)
    log_highs = np.log(np.array([c["high"] for c in raw_candles], dtype=np.float64))
    log_lows  = np.log(np.array([c["low"]  for c in raw_candles], dtype=np.float64))
    dates     = np.array([c["begin"] for c in raw_candles])
    return log_highs, log_lows, dates


# ── зигзаг ────────────────────────────────────────────────────────────────────

def build_zigzag(log_highs: np.ndarray, log_lows: np.ndarray,
                 dates: np.ndarray, threshold: float) -> tuple:
    """
    Каузальный зигзаг. Пивот фиксируется в баре подтверждения.
    Возвращает (log_pivot_prices, confirm_dates, pivot_directions).
    """
    log_pivot_prices = []
    confirm_dates    = []
    pivot_directions = []
    current_direction = 0
    extreme_log_price = (log_highs[0] + log_lows[0]) / 2.0

    for bar_index in range(len(log_highs)):
        if current_direction == 0:
            if log_highs[bar_index] - extreme_log_price >= threshold:
                current_direction, extreme_log_price = 1, log_highs[bar_index]
            elif extreme_log_price - log_lows[bar_index] >= threshold:
                current_direction, extreme_log_price = -1, log_lows[bar_index]
        elif current_direction == 1:
            if log_highs[bar_index] > extreme_log_price:
                extreme_log_price = log_highs[bar_index]
            elif extreme_log_price - log_lows[bar_index] >= threshold:
                log_pivot_prices.append(extreme_log_price)
                confirm_dates.append(dates[bar_index])
                pivot_directions.append(+1)
                current_direction, extreme_log_price = -1, log_lows[bar_index]
        else:
            if log_lows[bar_index] < extreme_log_price:
                extreme_log_price = log_lows[bar_index]
            elif log_highs[bar_index] - extreme_log_price >= threshold:
                log_pivot_prices.append(extreme_log_price)
                confirm_dates.append(dates[bar_index])
                pivot_directions.append(-1)
                current_direction, extreme_log_price = 1, log_highs[bar_index]

    return (
        np.array(log_pivot_prices),
        np.array(confirm_dates),
        np.array(pivot_directions, dtype=np.int8),
    )


# ── один шаг LWR ──────────────────────────────────────────────────────────────

def lwr_predict_step(
    query_log_prices: np.ndarray,
    query_pivot_dirs: np.ndarray,
    step_i: int,
    pool_log_prices: np.ndarray,
    pool_confirm_dates: np.ndarray,
    pool_pivot_dirs: np.ndarray,
    query_confirm_date: str,
    m_embed: int,
    k_lwr: int,
    min_dir: int,
) -> tuple:
    """
    Один шаг LWR с каузальной обрезкой по дате.

    C2: pool_confirm_dates[j]   < query_confirm_date
    C3: pool_confirm_dates[j+1] < query_confirm_date
        → j+1 < j_max  →  j <= j_max-2

    Возвращает (predicted_log_return, num_neighbors) или (None, 0).
    """
    j_max = int(np.searchsorted(pool_confirm_dates, query_confirm_date, side='left'))
    valid_end = j_max - 1  # включая: j <= j_max-2 → range до j_max-1
    valid_indices = np.arange(m_embed, valid_end)
    if len(valid_indices) == 0:
        return None, 0

    feature_matrix   = np.zeros((len(valid_indices), m_embed))
    target_returns   = np.zeros(len(valid_indices))
    pool_dirs_local  = np.zeros(len(valid_indices), dtype=np.int8)

    for row, j in enumerate(valid_indices):
        for lag in range(m_embed):
            feature_matrix[row, lag] = (
                pool_log_prices[j - lag] - pool_log_prices[j - lag - 1]
            )
        target_returns[row]  = pool_log_prices[j + 1] - pool_log_prices[j]
        pool_dirs_local[row] = pool_pivot_dirs[j]

    finite_mask = (
        np.all(np.isfinite(feature_matrix), axis=1) & np.isfinite(target_returns)
    )
    feature_matrix  = feature_matrix[finite_mask]
    target_returns  = target_returns[finite_mask]
    pool_dirs_local = pool_dirs_local[finite_mask]

    query_direction = int(query_pivot_dirs[step_i])
    query_vector    = np.array([
        query_log_prices[step_i - lag] - query_log_prices[step_i - lag - 1]
        for lag in range(m_embed)
    ])

    dir_mask      = pool_dirs_local == query_direction
    dir_count     = int(dir_mask.sum())
    if dir_count < min_dir:
        return None, 0

    if dir_count < k_lwr:
        return None, 0  # строгий: недостаточно соседей — прогноз не делается

    candidate_features = feature_matrix[dir_mask]
    candidate_targets  = target_returns[dir_mask]

    distances = np.linalg.norm(candidate_features - query_vector, axis=1)
    nearest   = np.argpartition(distances, k_lwr - 1)[:k_lwr]

    neighbor_features = candidate_features[nearest]
    neighbor_targets  = candidate_targets[nearest]
    neighbor_dists    = distances[nearest]
    max_dist          = neighbor_dists.max()

    if max_dist < 1e-12:
        return float(neighbor_targets.mean()), k_lwr

    gaussian_weights = np.exp(-0.5 * (neighbor_dists / max_dist) ** 2)
    sqrt_weights     = np.sqrt(gaussian_weights)

    design   = np.column_stack([np.ones(k_lwr), neighbor_features])
    wdesign  = design * sqrt_weights[:, None]
    wtargets = neighbor_targets * sqrt_weights

    coefficients, *_ = np.linalg.lstsq(wdesign, wtargets, rcond=None)
    predicted_lr = float(coefficients[0] + coefficients[1:] @ query_vector)
    return predicted_lr, k_lwr


# ── LP пул ────────────────────────────────────────────────────────────────────

def build_lp_pool(
    pool_log_prices: np.ndarray,
    pool_confirm_dates: np.ndarray,
    query_confirm_date: str,
    p_lp: int,
) -> np.ndarray | None:
    """
    Строит LP-пул: матрицу (N, p_lp) лог-доходностей T_pool пивотов,
    подтверждённых строго до query_confirm_date (C2 только — цель не нужна).
    """
    j_max = int(np.searchsorted(pool_confirm_dates, query_confirm_date, side='left'))
    valid_indices = np.arange(p_lp, j_max)
    if len(valid_indices) == 0:
        return None

    rows = np.zeros((len(valid_indices), p_lp))
    for row, j in enumerate(valid_indices):
        for lag in range(p_lp):
            rows[row, lag] = pool_log_prices[j - lag] - pool_log_prices[j - lag - 1]

    finite_mask = np.all(np.isfinite(rows), axis=1)
    result = rows[finite_mask]
    return result if len(result) > 0 else None


def make_lp_query_vector(
    predicted_lr: float,
    query_log_prices: np.ndarray,
    step_i: int,
    p_lp: int,
) -> np.ndarray:
    """
    Прогнозный вектор для LP-коррекции.

    x_pred[0]   = predicted_lr  (прогноз lr1 от LWR)
    x_pred[k]   = query_log_prices[step_i-k+1] − query_log_prices[step_i-k]
                  (текущие лог-доходности сдвинутые на 1)
    """
    x_pred = np.empty(p_lp)
    x_pred[0] = predicted_lr
    for k in range(1, p_lp):
        pivot_j = step_i - k + 1
        if pivot_j > 0:
            x_pred[k] = query_log_prices[pivot_j] - query_log_prices[pivot_j - 1]
        else:
            x_pred[k] = 0.0
    return x_pred


def lp_correct_log_return(
    pool_lr_matrix: np.ndarray,
    lp_query_vector: np.ndarray,
    k_lp: int,
    d_lp: int,
) -> float:
    """
    LP-коррекция прогнозного вектора в log-return пространстве.

    1. z-score нормировка пула и запроса
    2. k_lp ближайших соседей в нормированном пространстве
    3. SVD → d_lp главных компонент локального многообразия
    4. Проекция нормированного запроса на это многообразие
    5. Де-нормировка → скорректированная lr1

    Возвращает скорректированную лог-доходность lr1_corr.
    Если k_lp <= d_lp (недостаточно для SVD) — возвращает исходный lr1 без изменений.
    """
    n_pool = len(pool_lr_matrix)
    k = min(k_lp, n_pool)
    if k <= d_lp:
        return float(lp_query_vector[0])  # нет коррекции

    mu    = pool_lr_matrix.mean(0)
    sigma = np.where(pool_lr_matrix.std(0) < 1e-10, 1.0, pool_lr_matrix.std(0))
    norm_pool  = (pool_lr_matrix - mu) / sigma
    norm_query = (lp_query_vector - mu) / sigma

    distances = np.linalg.norm(norm_pool - norm_query, axis=1)
    knn_idx   = np.argsort(distances)[:k]

    z_knn      = norm_pool[knn_idx]
    center     = z_knn.mean(0)
    z_centered = z_knn - center

    _, _, vt = np.linalg.svd(z_centered, full_matrices=False)
    components = vt[:d_lp].T  # (p_lp, d_lp)

    delta        = norm_query - center
    projected    = center + components @ (components.T @ delta)
    corrected    = projected * sigma + mu

    return float(corrected[0])


# ── метрики ───────────────────────────────────────────────────────────────────

def rmae(errors: np.ndarray, actuals: np.ndarray) -> float:
    mask = np.isfinite(errors) & np.isfinite(actuals)
    dz   = float(np.mean(np.abs(np.diff(actuals[mask]))))
    return float(np.mean(np.abs(errors[mask])) / dz) if dz > 1e-12 else np.nan


# ── walk-forward ──────────────────────────────────────────────────────────────

def walk_forward():
    log_highs, log_lows, dates = load_log_candles(TICKER, INTERVAL)
    print(f"Загружено: {len(dates)} свечей  ({dates[0][:10]} … {dates[-1][:10]})")

    query_log_prices, query_confirm_dates, query_dirs = build_zigzag(
        log_highs, log_lows, dates, T_QUERY
    )
    pool_log_prices, pool_confirm_dates, pool_dirs = build_zigzag(
        log_highs, log_lows, dates, T_POOL
    )
    print(f"T_query={T_QUERY*100:.1f}%: {len(query_log_prices)} пивотов")
    print(f"T_pool ={T_POOL*100:.1f}%:  {len(pool_log_prices)} пивотов\n")

    n_steps = len(query_log_prices)
    raw_lwr_predictions = []
    lp_predictions      = {(p, k, d): []
                           for p in P_LP_GRID
                           for k in K_LP_GRID
                           for d in range(1, p)}
    actual_prices       = []
    skipped             = 0

    for step_i in range(MIN_HISTORY, n_steps - H):
        actual_price = float(np.exp(query_log_prices[step_i + H]))
        confirm_date = query_confirm_dates[step_i]

        # LWR прогноз
        predicted_lr, num_nb = lwr_predict_step(
            query_log_prices, query_dirs, step_i,
            pool_log_prices, pool_confirm_dates, pool_dirs,
            confirm_date, M_EMBED, K_LWR, MIN_DIR,
        )
        if predicted_lr is None:
            skipped += 1
            continue

        predicted_price = float(np.exp(query_log_prices[step_i] + predicted_lr))
        raw_lwr_predictions.append(predicted_price)
        actual_prices.append(actual_price)

        # LP коррекция для каждого P_LP
        for p_lp in P_LP_GRID:
            lp_pool = build_lp_pool(
                query_log_prices, query_confirm_dates, confirm_date, p_lp
            )
            lp_query_vector = make_lp_query_vector(
                predicted_lr, query_log_prices, step_i, p_lp
            )
            for k_lp in K_LP_GRID:
                for d_lp in range(1, p_lp):
                    if lp_pool is None:
                        lp_predictions[(p_lp, k_lp, d_lp)].append(predicted_price)
                    else:
                        lr_corr = lp_correct_log_return(
                            lp_pool, lp_query_vector, k_lp, d_lp
                        )
                        corr_price = float(np.exp(query_log_prices[step_i] + lr_corr))
                        lp_predictions[(p_lp, k_lp, d_lp)].append(corr_price)

    actual_array = np.array(actual_prices)
    lwr_errors   = np.array(raw_lwr_predictions) - actual_array
    r_lwr        = rmae(lwr_errors, actual_array)

    print(f"Прогнозов: {len(actual_prices)}   пропущено (мало соседей): {skipped}")
    print(f"\nБазовый LWR (m={M_EMBED} K={K_LWR}): rMAE = {r_lwr:.4f}\n")

    records = []
    best_rmae, best_params = np.inf, None

    for p_lp in P_LP_GRID:
        print(f"=== P_LP={p_lp}  (строки=d_lp, столбцы=k_lp) ===")
        header = f"{'d\\k':>5}  " + "  ".join(f"k={k:2d}" for k in K_LP_GRID)
        print(header)
        for d_lp in range(1, p_lp):
            row_str = f"  d={d_lp}  "
            for k_lp in K_LP_GRID:
                preds  = np.array(lp_predictions[(p_lp, k_lp, d_lp)])
                errors = preds - actual_array
                r      = rmae(errors, actual_array)
                row_str += f"  {r:.4f}"
                if r < best_rmae:
                    best_rmae, best_params = r, (p_lp, k_lp, d_lp)
                records.append({
                    "p_lp": p_lp, "k_lp": k_lp, "d_lp": d_lp, "rMAE": r,
                    "vs_lwr_pct": (r / r_lwr - 1) * 100,
                })
            print(row_str)
        print()

    print(
        f"Лучшая LP-коррекция: P_LP={best_params[0]} k={best_params[1]} d={best_params[2]}"
        f"  rMAE={best_rmae:.4f}  vs LWR: {(best_rmae/r_lwr-1)*100:+.1f}%"
    )

    out_dir  = Path(__file__).parent
    csv_path = out_dir / "lwr_lp_correction_results.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print(f"CSV → {csv_path}")


if __name__ == "__main__":
    walk_forward()
