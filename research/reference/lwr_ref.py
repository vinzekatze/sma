#!/usr/bin/env python3
"""
lwr_ref.py — LWR прогноз цены на зигзаге (событийное время).

Схема работы
────────────
1. Загрузить свечи → log(high), log(low)
2. Обрезать по точке отсчёта (все данные после неё недоступны)
3. Построить зигзаги T_query и T_pool из обрезанных данных
4. Из T_query: вектор запроса (m последних лог-доходностей) + направление
   Из T_pool:  матрица признаков + целевые лог-доходности
5. Найти k однонаправленных соседей → LWR → прогнозная цена

Использование
─────────────
    python lwr_ref.py data/candles/SBER/10m.json
    python lwr_ref.py data/candles/SBER/10m.json --origin 50
    python lwr_ref.py data/candles/SBER/10m.json --t-query 0.04 --t-pool 0.036 --m 2 --k 75
"""
import argparse
import json
import numpy as np
from pathlib import Path


# ── 1. Загрузка свечей ────────────────────────────────────────────────────────

def load_log_candles(file_path: str) -> tuple:
    """
    Читает JSON-файл свечей, сразу логарифмирует high и low.

    Формат файла: список словарей с полями "high", "low", "begin".
    Возвращает (log_highs, log_lows, dates).
    """
    with open(file_path) as file_handle:
        raw_candles = json.load(file_handle)

    log_highs = np.log(np.array([candle["high"]  for candle in raw_candles], dtype=np.float64))
    log_lows  = np.log(np.array([candle["low"]   for candle in raw_candles], dtype=np.float64))
    dates     = np.array([candle["begin"] for candle in raw_candles])

    return log_highs, log_lows, dates


# ── 2. Обрезка по точке отсчёта ───────────────────────────────────────────────

def trim_to_origin(log_highs: np.ndarray, log_lows: np.ndarray,
                   dates: np.ndarray, origin_offset: int) -> tuple:
    """
    Оставляет только свечи до точки отсчёта включительно.

    origin_offset — количество свечей, отрезаемых с конца.
      0   → все свечи доступны (точка отсчёта = последняя свеча)
      100 → последние 100 свечей недоступны

    Работает напрямую с массивами, не сообщает об обрезке никаким функциям.
    """
    if origin_offset < 0:
        raise ValueError(f"origin_offset не может быть отрицательным: {origin_offset}")

    if origin_offset == 0:
        return log_highs, log_lows, dates

    cutoff_index = len(log_highs) - origin_offset
    if cutoff_index <= 0:
        raise ValueError(
            f"origin_offset={origin_offset} >= числа свечей ({len(log_highs)})"
        )

    return log_highs[:cutoff_index], log_lows[:cutoff_index], dates[:cutoff_index]


# ── 3. Построение зигзага ─────────────────────────────────────────────────────

def build_zigzag(log_highs: np.ndarray, log_lows: np.ndarray,
                 dates: np.ndarray, threshold: float) -> tuple:
    """
    Каузальный зигзаг по логарифмическому порогу.

    Пивот фиксируется когда противоположная сторона отклонилась
    от экстремума на threshold. Подтверждение — в том же баре.

    Возвращает (log_pivot_prices, confirm_dates, pivot_directions):
      log_pivot_prices — log-цена экстремума
      confirm_dates    — дата подтверждения (не экстремума)
      pivot_directions — +1 вершина (HIGH), −1 впадина (LOW)
    """
    log_pivot_prices = []
    confirm_dates    = []
    pivot_directions = []

    current_direction = 0
    extreme_log_price = (log_highs[0] + log_lows[0]) / 2.0

    for bar_index in range(len(log_highs)):
        if current_direction == 0:
            if log_highs[bar_index] - extreme_log_price >= threshold:
                current_direction = 1
                extreme_log_price = log_highs[bar_index]
            elif extreme_log_price - log_lows[bar_index] >= threshold:
                current_direction = -1
                extreme_log_price = log_lows[bar_index]

        elif current_direction == 1:
            if log_highs[bar_index] > extreme_log_price:
                extreme_log_price = log_highs[bar_index]
            elif extreme_log_price - log_lows[bar_index] >= threshold:
                log_pivot_prices.append(extreme_log_price)
                confirm_dates.append(dates[bar_index])
                pivot_directions.append(+1)
                current_direction = -1
                extreme_log_price = log_lows[bar_index]

        else:  # current_direction == -1
            if log_lows[bar_index] < extreme_log_price:
                extreme_log_price = log_lows[bar_index]
            elif log_highs[bar_index] - extreme_log_price >= threshold:
                log_pivot_prices.append(extreme_log_price)
                confirm_dates.append(dates[bar_index])
                pivot_directions.append(-1)
                current_direction = 1
                extreme_log_price = log_highs[bar_index]

    return (
        np.array(log_pivot_prices),
        np.array(confirm_dates),
        np.array(pivot_directions, dtype=np.int8),
    )


# ── 4. Построение векторов ────────────────────────────────────────────────────

def build_pool_vectors(log_pivot_prices: np.ndarray,
                       pivot_directions: np.ndarray,
                       embedding_dim: int) -> tuple:
    """
    Строит матрицу признаков и целевые лог-доходности для пула.

    Для каждого пивота j (от embedding_dim до n−2):
      feature_row[lag] = log_pivot_prices[j−lag] − log_pivot_prices[j−lag−1]
        lag = 0..embedding_dim−1  — лог-доходности последних m плечей
      target_log_return = log_pivot_prices[j+1] − log_pivot_prices[j]
        — лог-доходность до следующего пивота (цель прогноза)

    Возвращает (pool_feature_matrix, pool_target_log_returns, pool_pivot_directions).
    """
    num_pivots    = len(log_pivot_prices)
    valid_indices = np.arange(embedding_dim, num_pivots - 1)

    pool_feature_matrix      = np.zeros((len(valid_indices), embedding_dim))
    pool_target_log_returns  = np.zeros(len(valid_indices))
    pool_pivot_directions    = np.zeros(len(valid_indices), dtype=np.int8)

    for row_index, pivot_index in enumerate(valid_indices):
        for lag in range(embedding_dim):
            pool_feature_matrix[row_index, lag] = (
                log_pivot_prices[pivot_index - lag]
                - log_pivot_prices[pivot_index - lag - 1]
            )
        pool_target_log_returns[row_index] = (
            log_pivot_prices[pivot_index + 1] - log_pivot_prices[pivot_index]
        )
        pool_pivot_directions[row_index] = pivot_directions[pivot_index]

    finite_mask = (
        np.all(np.isfinite(pool_feature_matrix), axis=1)
        & np.isfinite(pool_target_log_returns)
    )
    return (
        pool_feature_matrix[finite_mask],
        pool_target_log_returns[finite_mask],
        pool_pivot_directions[finite_mask],
    )


def build_query_vector(log_pivot_prices: np.ndarray, embedding_dim: int) -> np.ndarray:
    """
    Строит вектор запроса из последних embedding_dim плечей T_query зигзага.

    query_vector[lag] = log_pivot_prices[−1−lag] − log_pivot_prices[−2−lag]
    """
    num_pivots = len(log_pivot_prices)
    if num_pivots < embedding_dim + 1:
        raise ValueError(
            f"Недостаточно пивотов в T_query: нужно ≥ {embedding_dim + 1}, есть {num_pivots}"
        )

    query_vector = np.array([
        log_pivot_prices[-1 - lag] - log_pivot_prices[-2 - lag]
        for lag in range(embedding_dim)
    ])
    return query_vector


# ── 5. Поиск соседей и LWR прогноз ───────────────────────────────────────────

def find_neighbors(query_vector: np.ndarray,
                   pool_feature_matrix: np.ndarray,
                   pool_pivot_directions: np.ndarray,
                   query_direction: int,
                   num_neighbors: int) -> np.ndarray | None:
    """
    Находит индексы num_neighbors ближайших соседей по евклидову расстоянию.

    Ищет только среди однонаправленных событий пула (pivot_direction == query_direction).
    Если однонаправленных меньше num_neighbors — возвращает None (прогноз не делается).

    Возвращает массив индексов в pool_feature_matrix или None.
    """
    direction_mask    = pool_pivot_directions == query_direction
    candidate_indices = np.where(direction_mask)[0]

    if len(candidate_indices) < num_neighbors:
        return None

    candidate_features = pool_feature_matrix[candidate_indices]
    distances          = np.linalg.norm(candidate_features - query_vector, axis=1)

    nearest_positions = np.argpartition(distances, num_neighbors - 1)[:num_neighbors]
    neighbor_indices  = candidate_indices[nearest_positions]

    return neighbor_indices


def lwr_predict(query_vector: np.ndarray,
                pool_feature_matrix: np.ndarray,
                pool_target_log_returns: np.ndarray,
                neighbor_indices: np.ndarray) -> float:
    """
    LWR (Locally Weighted Regression) прогноз лог-доходности до следующего пивота.

    Ядро: w_j = exp(−0.5 · (d_j / d_max)²)  — гауссово, bandwidth = d_max.
    Модель: y = c₀ + c₁·x₁ + … + cₘ·xₘ  (взвешенная OLS).
    Возвращает прогнозируемую лог-доходность.
    """
    neighbor_features      = pool_feature_matrix[neighbor_indices]
    neighbor_target_returns = pool_target_log_returns[neighbor_indices]

    distances    = np.linalg.norm(neighbor_features - query_vector, axis=1)
    max_distance = distances.max()

    if max_distance < 1e-12:
        return float(neighbor_target_returns.mean())

    gaussian_weights  = np.exp(-0.5 * (distances / max_distance) ** 2)
    sqrt_weights      = np.sqrt(gaussian_weights)

    design_matrix     = np.column_stack([np.ones(len(neighbor_indices)), neighbor_features])
    weighted_design   = design_matrix * sqrt_weights[:, None]
    weighted_targets  = neighbor_target_returns * sqrt_weights

    coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_targets, rcond=None)

    predicted_log_return = float(coefficients[0] + coefficients[1:] @ query_vector)
    return predicted_log_return


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LWR прогноз цены следующего зигзаг-пивота",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file",
        help="Путь к JSON-файлу свечей (поля: high, low, begin)",
    )
    parser.add_argument(
        "--origin", type=int, default=0, metavar="N",
        help="Точка отсчёта: отбросить N последних свечей (0 = все свечи)",
    )
    parser.add_argument(
        "--t-query", type=float, default=0.04, metavar="T",
        help="Порог зигзага запроса (доля, напр. 0.04 = 4%%)",
    )
    parser.add_argument(
        "--t-pool", type=float, default=0.036, metavar="T",
        help="Порог зигзага пула (доля, напр. 0.036 = 3.6%%)",
    )
    parser.add_argument(
        "--m", type=int, default=2, metavar="M",
        help="Размерность вложения (число плечей в векторе)",
    )
    parser.add_argument(
        "--k", type=int, default=75, metavar="K",
        help="Число соседей для LWR",
    )
    parser.add_argument(
        "--steps", type=int, default=1, metavar="N",
        help="Число итеративных шагов прогноза (1 = один пивот вперёд)",
    )
    args = parser.parse_args()

    # 1. Загрузка
    log_highs, log_lows, dates = load_log_candles(args.file)
    print(f"Свечей загружено: {len(dates)}  ({dates[0][:10]} … {dates[-1][:10]})")

    # 2. Обрезка — единственная точка управления каузальностью
    log_highs, log_lows, dates = trim_to_origin(log_highs, log_lows, dates, args.origin)
    print(f"После обрезки:    {len(dates)} свечей  (последняя: {dates[-1][:10]})")

    # 3. Зигзаги строятся только из обрезанных данных
    query_log_prices, _, query_directions = build_zigzag(
        log_highs, log_lows, dates, args.t_query
    )
    pool_log_prices, _, pool_directions_zz = build_zigzag(
        log_highs, log_lows, dates, args.t_pool
    )
    print(f"T_query = {args.t_query*100:.1f}%:  {len(query_log_prices)} пивотов")
    print(f"T_pool  = {args.t_pool*100:.1f}%:  {len(pool_log_prices)} пивотов")

    # 4. Векторы
    pool_feature_matrix, pool_target_log_returns, pool_pivot_directions = build_pool_vectors(
        pool_log_prices, pool_directions_zz, args.m
    )
    query_vector    = build_query_vector(query_log_prices, args.m)
    query_direction = int(query_directions[-1])

    direction_label = "HIGH (+1)" if query_direction == 1 else "LOW (−1)"
    print(f"Вектор запроса:   {np.round(query_vector, 5).tolist()}  [{direction_label}]")
    print(f"Пул:              {len(pool_feature_matrix)} событий")

    # 5. Прогноз
    neighbor_indices = find_neighbors(
        query_vector, pool_feature_matrix, pool_pivot_directions,
        query_direction, args.k,
    )
    if neighbor_indices is None:
        same_dir_count = int((pool_pivot_directions == query_direction).sum())
        print(f"\nПрогноз невозможен: однонаправленных событий {same_dir_count} < k={args.k}")
        return

    predicted_log_return = lwr_predict(
        query_vector, pool_feature_matrix, pool_target_log_returns, neighbor_indices,
    )
    predicted_price = float(np.exp(query_log_prices[-1] + predicted_log_return))

    direction_sym = "▲" if query_direction > 0 else "▼"
    print(f"\nШаг 1 {direction_sym} → {predicted_price:.4f}  (lr={predicted_log_return:+.5f})")

    # ── итеративные шаги 2..N ────────────────────────────────────────────────
    if args.steps > 1:
        step_query_vector = np.concatenate([[predicted_log_return], query_vector[:-1]])
        step_direction    = -query_direction
        cumulative_lr     = predicted_log_return

        for step in range(2, args.steps + 1):
            step_neighbors = find_neighbors(
                step_query_vector, pool_feature_matrix, pool_pivot_directions,
                step_direction, args.k,
            )
            if step_neighbors is None:
                same_dir = int((pool_pivot_directions == step_direction).sum())
                print(f"Шаг {step} → невозможен: {same_dir} однонаправленных < k={args.k}")
                break

            step_lr = lwr_predict(
                step_query_vector, pool_feature_matrix,
                pool_target_log_returns, step_neighbors,
            )
            cumulative_lr += step_lr
            step_price = float(np.exp(query_log_prices[-1] + cumulative_lr))
            step_sym   = "▲" if step_direction > 0 else "▼"
            print(f"Шаг {step} {step_sym} → {step_price:.4f}  (lr={step_lr:+.5f}  cumul={cumulative_lr:+.5f})")

            step_query_vector = np.concatenate([[step_lr], step_query_vector[:-1]])
            step_direction    = -step_direction


if __name__ == "__main__":
    main()
