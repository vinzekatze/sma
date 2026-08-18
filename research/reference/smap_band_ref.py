#!/usr/bin/env python3
"""
smap_band_ref.py — S-map прогноз ПОЛОСЫ неопределённости (не только точки)
на зигзаге (событийное время). Прямое расширение smap_ref.py.

Мотивация (2026-07-08): в app8 обнаружено, что взвешенные квантили пула
(то же распределение соседей, что даёт S-map-точку через регрессию) чаще
содержат факт, чем сама точка. Направление B (событийная фрактальность)
разворачивается на полосу как основной продукт — см. память
project-phase7-uncertainty-field-pivot. Этот файл — чистая, каузально
проверенная реализация полосы, по образцу smap_ref.py, БЕЗ сокращений.

Схема работы (не изменилась относительно smap_ref.py)
──────────────────────────────────────────────────────
1. Загрузить свечи → log(high), log(low)
2. Обрезать по точке отсчёта (все данные после неё недоступны) —
   ЕДИНСТВЕННАЯ точка контроля каузальности, в самом начале алгоритма.
3. Построить зигзаги T_query и T_pool из ОБРЕЗАННЫХ данных
4. Из T_query: вектор запроса (m последних лог-доходностей) + направление
   Из T_pool:  матрица признаков + целевые лог-доходности
5. Отобрать однонаправленные события пула → S-map веса → точка И полоса

Что нового относительно smap_ref.py
────────────────────────────────────
  smap_weights(...)   — веса вынесены в отдельную функцию (были внутри
                         smap_predict) — переиспользуются и для точки
                         (регрессия), и для полосы (квантили), БЕЗ
                         повторного вычисления расстояний.
  weighted_quantile()  — взвешенные квантили pool_target_log_returns
                         с теми же весами w_j = exp(-θ·d_j/mean(d)).
  smap_forecast_step() — считает точку И полосу ОДНИМ вызовом (общие веса).

Точка — регрессия (OLS), полоса — сырое распределение соседей с теми же
весами. Это РАЗНЫЕ операции: точка не обязана быть медианой полосы.

Использование
─────────────
    python smap_band_ref.py data/candles/SBER/10m.json
    python smap_band_ref.py data/candles/SBER/10m.json --origin 50 --steps 2
    python smap_band_ref.py data/candles/SBER/10m.json --t-query 0.04 --t-pool 0.036 --m 2 --theta 2
"""
import argparse
import json
import numpy as np
from pathlib import Path

QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


# ── 1. Загрузка свечей ────────────────────────────────────────────────────────

def load_log_candles(file_path: str) -> tuple:
    """
    Читает JSON-файл свечей, сразу логарифмирует high и low.

    Формат файла: список словарей с полями "high", "low", "begin".
    Возвращает (log_highs, log_lows, dates).
    """
    with open(file_path) as file_handle:
        raw_candles = json.load(file_handle)

    log_highs = np.log(np.array([candle["high"] for candle in raw_candles], dtype=np.float64))
    log_lows  = np.log(np.array([candle["low"]  for candle in raw_candles], dtype=np.float64))
    dates     = np.array([candle["begin"] for candle in raw_candles])

    return log_highs, log_lows, dates


# ── 2. Обрезка по точке отсчёта — ЕДИНСТВЕННАЯ точка контроля каузальности ────

def trim_to_origin(log_highs: np.ndarray, log_lows: np.ndarray,
                   dates: np.ndarray, origin_offset: int) -> tuple:
    """
    Оставляет только свечи до точки отсчёта включительно.

    origin_offset — количество свечей, отрезаемых с конца.
      0   → все свечи доступны (точка отсчёта = последняя свеча)
      100 → последние 100 свечей недоступны

    Работает напрямую с массивами, НИКАКИХ алгоритмических ограничителей
    ниже по пайплайну — вся каузальность обеспечивается ИСКЛЮЧИТЕЛЬНО этой
    обрезкой в самом начале. Всё, что вызывается после (зигзаг, пул,
    прогноз), работает с уже обрезанными массивами и не знает о существовании
    данных за пределами origin.
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
                 dates: np.ndarray, threshold: float, min_bars: int = 0) -> tuple:
    """
    Каузальный зигзаг по логарифмическому порогу.

    Пивот фиксируется когда противоположная сторона отклонилась
    от экстремума на threshold. Подтверждение — в том же баре.

    min_bars — минимальное число баров между ПОДТВЕРЖДЁННЫМИ пивотами
    (аналог параметра Depth в классических zigzag-индикаторах MT4/5).
    threshold сам по себе не защищает от «мельчания»: на мелком T пивоты
    могут подтверждаться каждые 1-2 бара, что на практике бесполезно —
    к моменту реакции трейдера цена уже съедает заметную часть такого
    мелкого плеча. min_bars=0 (по умолчанию) — поведение не меняется
    относительно исходной версии.

    Важная деталь причинности/точности: пока идёт «пауза» (порог по цене
    уже пройден, но min_bars ещё не набрано), экстремум-кандидат НЕ
    отслеживается отдельно — если цена продолжит уходить глубже во время
    паузы, а затем частично отыграет НАЗАД к моменту, когда min_bars
    наберётся, новое плечо стартует от цены НА БАРЕ ПОДТВЕРЖДЕНИЯ, не от
    самой глубокой точки паузы. Это стандартное упрощение (тот же
    компромисс, что в MT-подобных zigzag с Depth), не баг — но при
    QA/сравнении с другими реализациями это стоит держать в уме.

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
    last_pivot_bar_index = -min_bars - 1   # первый пивот никогда не блокируется

    for bar_index in range(len(log_highs)):
        enough_bars = (bar_index - last_pivot_bar_index) >= min_bars

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
            elif extreme_log_price - log_lows[bar_index] >= threshold and enough_bars:
                log_pivot_prices.append(extreme_log_price)
                confirm_dates.append(dates[bar_index])
                pivot_directions.append(+1)
                last_pivot_bar_index = bar_index
                current_direction = -1
                extreme_log_price = log_lows[bar_index]

        else:  # current_direction == -1
            if log_lows[bar_index] < extreme_log_price:
                extreme_log_price = log_lows[bar_index]
            elif log_highs[bar_index] - extreme_log_price >= threshold and enough_bars:
                log_pivot_prices.append(extreme_log_price)
                confirm_dates.append(dates[bar_index])
                pivot_directions.append(-1)
                last_pivot_bar_index = bar_index
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
                       embedding_dim: int,
                       horizon: int = 1) -> tuple:
    """
    Строит матрицу признаков и целевые лог-доходности для пула.

    Для каждого пивота j (от embedding_dim до n−horizon−1):
      feature_row[lag] = log_pivot_prices[j−lag] − log_pivot_prices[j−lag−1]
        lag = 0..embedding_dim−1  — лог-доходности последних m плечей
      target_log_return = log_pivot_prices[j+horizon] − log_pivot_prices[j]
        — КУМУЛЯТИВНАЯ лог-доходность через `horizon` пивотов вперёд
        (horizon=1 — до следующего пивота; horizon=2 — «туда и обратно»,
        прямой исторический readout, БЕЗ промежуточного прогноза шага 1 —
        см. app9.py: при θ=0 цепочка «точка шага 1 → вектор шага 2» не
        нужна, обе полосы читаются из пула независимо и одновременно).
      direction — направление ПИВОТА j (не пивота j+horizon) — фильтр по
        типу исходного события, как и для horizon=1.

    Возвращает (pool_feature_matrix, pool_target_log_returns, pool_pivot_directions).
    """
    num_pivots    = len(log_pivot_prices)
    valid_indices = np.arange(embedding_dim, num_pivots - horizon)

    pool_feature_matrix     = np.zeros((len(valid_indices), embedding_dim))
    pool_target_log_returns = np.zeros(len(valid_indices))
    pool_pivot_directions   = np.zeros(len(valid_indices), dtype=np.int8)

    for row_index, pivot_index in enumerate(valid_indices):
        for lag in range(embedding_dim):
            pool_feature_matrix[row_index, lag] = (
                log_pivot_prices[pivot_index - lag]
                - log_pivot_prices[pivot_index - lag - 1]
            )
        pool_target_log_returns[row_index] = (
            log_pivot_prices[pivot_index + horizon] - log_pivot_prices[pivot_index]
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


# ── 4b. Кросс-тикерный пул (эксп.17, D_allpeers) ──────────────────────────────
#
# Обязателен на крупных T (20-30%): у одного тикера на таком пороге всего
# 60-100 пивотов за всю историю — событий в направленно-отфильтрованном пуле
# (~половина от этого) слишком мало для устойчивых квантилей. Пулим ВСЕ
# тикеры "голубых фишек" MOEX (список из эксп.17, без ручного отбора —
# D_allpeers оказался не хуже отбора по корреляции).

DATA_DIR = Path(__file__).parents[2] / "data" / "candles"

UNIVERSE = [
    "SBER", "LKOH", "CHMF", "NVTK", "MGNT", "VTBR", "NLMK", "MRKP", "GAZP", "PLZL",
    "ROSN", "TATN", "MTSS", "ALRS", "MOEX", "SNGS", "IRAO", "RUAL", "MAGN", "PHOR",
    "AFLT", "HYDR", "SIBN", "TRNFP", "RTKM",
    "BANE", "BANEP", "MTLR", "MTLRP", "RASP", "VSMO", "KMAZ", "AKRN", "MSNG", "TGKA",
    "UPRO", "PIKK", "MVID", "LSRG", "CBOM", "GCHE", "SVAV", "FESH", "KZOS", "NKNC",
]


def load_ticker_candles(ticker: str, interval: str, data_dir: Path = DATA_DIR):
    """Загружает лог-свечи тикера из data/candles/{ticker}/{interval}.json.
    Возвращает (log_highs, log_lows, dates) или None, если файла нет/пуст —
    НЕ бросает исключение (кросс-тикерный пул должен молча пропускать
    недоступных пиров, как в build_causal_pool ниже)."""
    path = data_dir / ticker / f"{interval}.json"
    if not path.exists():
        return None
    try:
        return load_log_candles(str(path))
    except (json.JSONDecodeError, KeyError):
        return None


def build_causal_pool(cutoff_date: str, ticker_arrays: dict, t_pool: float,
                      embedding_dim: int, horizon: int = 1, min_bars: int = 0) -> tuple:
    """
    Кросс-тикерный причинно обрезанный пул: для КАЖДОГО тикера в
    ticker_arrays сырые бары ПЕРВЫМ ДЕЛОМ обрезаются по дате <= cutoff_date
    (жёсткая обрезка данных в самом начале — как trim_to_origin для
    целевого тикера, см. модульный docstring), ЗАТЕМ строится зигзаг T_pool
    и векторы пула. Никаких алгоритмических ограничителей ниже по цепочке —
    build_zigzag/build_pool_vectors не знают, что данные обрезаны, они
    просто получают более короткие массивы.

    ticker_arrays: {ticker: (log_highs, log_lows, dates)} — ПОЛНЫЕ
    (необрезанные) массивы каждого тикера, загруженные один раз вызывающим
    кодом. cutoff_date — дата подтверждения origin'а целевого тикера
    (строка, сравнима лексикографически с датами свечей). horizon —
    см. build_pool_vectors (1 = следующий пивот, 2 = «туда-обратно»).
    min_bars — см. build_zigzag (минимум баров между пивотами, применяется
    ОДИНАКОВО ко всем тикерам пула, не только к целевому).

    Возвращает (pool_feature_matrix, pool_target_log_returns, pool_pivot_directions)
    — объединённые по всем тикерам.
    """
    feats_list, tars_list, dirs_list = [], [], []
    for ticker, (log_highs, log_lows, dates) in ticker_arrays.items():
        mask = dates <= cutoff_date                       # ── обрезка в начале ──
        if mask.sum() < embedding_dim + horizon + 2:
            continue
        log_highs_c, log_lows_c, dates_c = log_highs[mask], log_lows[mask], dates[mask]

        pivot_prices, _, pivot_directions = build_zigzag(log_highs_c, log_lows_c, dates_c, t_pool, min_bars)
        feats, tars, dirs = build_pool_vectors(pivot_prices, pivot_directions, embedding_dim, horizon)
        if len(tars):
            feats_list.append(feats); tars_list.append(tars); dirs_list.append(dirs)

    if not tars_list:
        return np.zeros((0, embedding_dim)), np.zeros(0), np.zeros(0, dtype=np.int8)
    return (np.vstack(feats_list), np.concatenate(tars_list), np.concatenate(dirs_list))


# ── 5. Фильтр направления, S-map веса, точка и полоса ────────────────────────

def filter_direction_pool(pool_feature_matrix: np.ndarray,
                          pool_target_log_returns: np.ndarray,
                          pool_pivot_directions: np.ndarray,
                          query_direction: int,
                          min_pool_size: int) -> tuple:
    """
    Отбирает из пула только однонаправленные с запросом события.

    Если однонаправленных меньше min_pool_size — возвращает (None, None)
    и прогноз не делается.

    Возвращает (direction_features, direction_targets) или (None, None).
    """
    direction_mask = pool_pivot_directions == query_direction
    num_matching   = direction_mask.sum()

    if num_matching < min_pool_size:
        return None, None

    return (
        pool_feature_matrix[direction_mask],
        pool_target_log_returns[direction_mask],
    )


def smap_weights(query_vector: np.ndarray,
                 pool_feature_matrix: np.ndarray,
                 theta: float) -> np.ndarray:
    """
    S-map (Sugihara 1994) веса: w_j = exp(−θ · d_j / mean(d)).

      θ = 0  → все веса равны → глобальная OLS / безусловная полоса
      θ → ∞  → концентрация на ближайшем соседе

    Вырожденный случай (все точки пула совпадают с запросом,
    mean_distance≈0) — возвращает равные веса.
    """
    distances     = np.linalg.norm(pool_feature_matrix - query_vector, axis=1)
    mean_distance = distances.mean()

    if mean_distance < 1e-14:
        return np.ones(len(distances))
    if theta == 0:
        return np.ones(len(distances))
    return np.exp(-theta * distances / mean_distance)


def smap_predict_from_weights(weights: np.ndarray,
                              pool_feature_matrix: np.ndarray,
                              pool_target_log_returns: np.ndarray,
                              query_vector: np.ndarray) -> float:
    """Взвешенная OLS: y = c₀ + c₁·x₁ + … + cₘ·xₘ. Возвращает прогнозируемую
    лог-доходность (точка)."""
    sqrt_weights     = np.sqrt(weights)
    design_matrix    = np.column_stack([np.ones(len(pool_target_log_returns)),
                                        pool_feature_matrix])
    weighted_design  = design_matrix * sqrt_weights[:, None]
    weighted_targets = pool_target_log_returns * sqrt_weights

    coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_targets, rcond=None)
    return float(coefficients[0] + coefficients[1:] @ query_vector)


def smap_predict(query_vector: np.ndarray,
                 pool_feature_matrix: np.ndarray,
                 pool_target_log_returns: np.ndarray,
                 theta: float) -> float:
    """Точка (для обратной совместимости с smap_ref.py — при необходимости
    только точки, без полосы, без пересчёта весов снаружи)."""
    weights = smap_weights(query_vector, pool_feature_matrix, theta)
    return smap_predict_from_weights(weights, pool_feature_matrix,
                                     pool_target_log_returns, query_vector)


def weighted_quantile(values: np.ndarray, weights: np.ndarray,
                      quantiles: tuple) -> dict:
    """
    Взвешенные квантили values по весам weights (линейная интерполяция по
    взвешенной эмпирической CDF, midpoint-поправка). Возвращает
    {quantile_level: value}.
    """
    order = np.argsort(values)
    v, w  = values[order], weights[order]

    if w.sum() < 1e-14:
        median = float(np.median(values))
        return {q: median for q in quantiles}

    cumulative_weight  = np.cumsum(w) - 0.5 * w
    cumulative_weight /= w.sum()
    band_values = np.interp(quantiles, cumulative_weight, v)
    return {q: float(x) for q, x in zip(quantiles, band_values)}


def smap_forecast_step(query_vector: np.ndarray,
                       pool_feature_matrix: np.ndarray,
                       pool_target_log_returns: np.ndarray,
                       theta: float,
                       quantile_levels: tuple = QUANTILE_LEVELS) -> dict:
    """
    Один шаг прогноза: точка (регрессия) И полоса (взвешенные квантили),
    ОБЩИЕ веса (одно вычисление расстояний вместо двух).

    Возвращает {"log_return": float, "band": {quantile: log_return}}.
    """
    weights    = smap_weights(query_vector, pool_feature_matrix, theta)
    log_return = smap_predict_from_weights(weights, pool_feature_matrix,
                                           pool_target_log_returns, query_vector)
    band       = weighted_quantile(pool_target_log_returns, weights, quantile_levels)
    return {"log_return": log_return, "band": band}


# ── 6. Многошаговый прогноз (точка ведёт цепочку; полоса — на каждом шаге) ────

def run_band_forecast(query_vector: np.ndarray, query_direction: int,
                      last_log_price: float,
                      pool_feature_matrix: np.ndarray,
                      pool_target_log_returns: np.ndarray,
                      pool_pivot_directions: np.ndarray,
                      theta: float, min_pool_size: int, steps: int,
                      quantile_levels: tuple = QUANTILE_LEVELS) -> list[dict]:
    """
    Итеративный прогноз (направление чередуется по свойству зигзага). На
    КАЖДОМ шаге цепочка ведётся ТОЧКОЙ (как smap_ref.py) — модель не знает
    будущего факта, это честный live-прогноз. Полоса на шаге h — взвешенные
    квантили пула, ветвящиеся от точки ПРЕДЫДУЩЕГО шага (локальная оценка
    ЭТОГО шага, не композиция дисперсии по всей цепочке — то же ограничение,
    что в run_forecast из app8.py).

    query_vector   — уже построен вызывающим кодом (build_query_vector)
    last_log_price — log_pivot_prices[-1] запросного зигзага (цена origin'а)

    Возвращает список словарей:
      ok=True:  {"step","ok","direction","log_return","cumulative_lr",
                "price","band_prices" (dict quantile->price), "pool_size"}
      ok=False: {"step","ok","direction","pool_size","reason"}
    """
    direction     = query_direction
    cumulative_lr = 0.0
    results: list[dict] = []

    for step in range(1, steps + 1):
        direction_features, direction_targets = filter_direction_pool(
            pool_feature_matrix, pool_target_log_returns, pool_pivot_directions,
            direction, min_pool_size,
        )
        if direction_features is None:
            pool_size = int((pool_pivot_directions == direction).sum())
            results.append({
                "step": step, "ok": False, "direction": direction,
                "pool_size": pool_size,
                "reason": f"пул {pool_size} < min_pool={min_pool_size}",
            })
            break

        step_result = smap_forecast_step(query_vector, direction_features,
                                         direction_targets, theta, quantile_levels)
        log_return     = step_result["log_return"]
        cumulative_lr += log_return
        price          = float(np.exp(last_log_price + cumulative_lr))

        # полоса ветвится от точки ПРЕДЫДУЩЕГО шага (cumulative_lr до
        # добавления log_return этого шага), не от точки этого же шага
        cumulative_lr_prev = cumulative_lr - log_return
        band_prices = {
            q: float(np.exp(last_log_price + cumulative_lr_prev + qlr))
            for q, qlr in step_result["band"].items()
        }

        results.append({
            "step": step, "ok": True, "direction": direction,
            "log_return": log_return, "cumulative_lr": cumulative_lr,
            "price": price, "band_prices": band_prices,
            "pool_size": len(direction_targets),
        })

        query_vector = np.concatenate([[log_return], query_vector[:-1]])
        direction    = -direction

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="S-map прогноз точки И полосы неопределённости для зигзага",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file", help="Путь к JSON-файлу свечей целевого тикера (поля: high, low, begin)")
    parser.add_argument("--origin", type=int, default=0, metavar="N",
                        help="Точка отсчёта: отбросить N последних свечей (0 = все свечи)")
    parser.add_argument("--t-query", type=float, default=0.20, metavar="T",
                        help="Порог зигзага запроса (доля, напр. 0.20 = 20%%). "
                             "На малых T (≤5%%) у одного тикера почти нет пивотов "
                             "для устойчивого пула — нужен крупный порог.")
    parser.add_argument("--t-pool", type=float, default=0.20 * 0.8987, metavar="T",
                        help="Порог зигзага пула (доля). По умолчанию T_query×0.8987 (эксп.17f).")
    parser.add_argument("--pool", choices=["own", "cross"], default="cross",
                        help="own — только T_pool целевого тикера (мало событий на крупных T); "
                             "cross — кросс-тикерный пул UNIVERSE (44 тикера, эксп.17, реком.)")
    parser.add_argument("--interval", default="1d", metavar="IV",
                        help="Интервал для загрузки пиров кросс-тикерного пула (data/candles/*/IV.json)")
    parser.add_argument("--m", type=int, default=8, metavar="M",
                        help="Размерность вложения (число плечей в векторе)")
    parser.add_argument("--theta", type=float, default=2.0, metavar="θ",
                        help="Параметр локализации S-map (0 = глобальная OLS, >0 = локальнее)")
    parser.add_argument("--min-pool", type=int, default=None, metavar="N",
                        help="Минимум однонаправленных событий (по умолчанию m+2)")
    parser.add_argument("--steps", type=int, default=2, metavar="N",
                        help="Число итеративных шагов прогноза (2 = уход+возврат)")
    args = parser.parse_args()

    min_pool_size = args.min_pool if args.min_pool is not None else args.m + 2

    # 1. Загрузка целевого тикера
    log_highs, log_lows, dates = load_log_candles(args.file)
    print(f"Свечей загружено: {len(dates)}  ({dates[0][:10]} … {dates[-1][:10]})")

    # 2. Обрезка — единственная точка управления каузальностью (целевой тикер)
    log_highs, log_lows, dates = trim_to_origin(log_highs, log_lows, dates, args.origin)
    print(f"После обрезки:    {len(dates)} свечей  (последняя: {dates[-1][:10]})")

    # 3. T_query зигзаг — только из обрезанных данных целевого тикера
    query_log_prices, query_dates, query_directions = build_zigzag(
        log_highs, log_lows, dates, args.t_query)
    print(f"T_query = {args.t_query*100:.1f}%:  {len(query_log_prices)} пивотов")

    query_vector    = build_query_vector(query_log_prices, args.m)
    query_direction = int(query_directions[-1])
    last_log_price  = float(query_log_prices[-1])
    cutoff_date     = str(query_dates[-1])

    # 4. Пул — own (T_pool того же тикера) или cross (кросс-тикерный, реком.)
    if args.pool == "own":
        pool_log_prices, _, pool_directions_zz = build_zigzag(log_highs, log_lows, dates, args.t_pool)
        pool_feature_matrix, pool_target_log_returns, pool_pivot_directions = build_pool_vectors(
            pool_log_prices, pool_directions_zz, args.m)
        print(f"T_pool  = {args.t_pool*100:.1f}%:  {len(pool_log_prices)} пивотов (own)")
    else:
        # ── обрезка КАЖДОГО пира по дате origin'а — в build_causal_pool, в самом
        #    начале обработки этого пира, до построения его зигзага ──
        ticker_arrays = {}
        for ticker in UNIVERSE:
            loaded = load_ticker_candles(ticker, args.interval)
            if loaded is not None:
                ticker_arrays[ticker] = loaded
        pool_feature_matrix, pool_target_log_returns, pool_pivot_directions = build_causal_pool(
            cutoff_date, ticker_arrays, args.t_pool, args.m)
        print(f"T_pool  = {args.t_pool*100:.1f}%:  кросс-тикерный пул, "
              f"{len(ticker_arrays)}/{len(UNIVERSE)} тикеров доступно")

    direction_label = "HIGH (+1)" if query_direction == 1 else "LOW (−1)"
    print(f"Вектор запроса:   {np.round(query_vector, 5).tolist()}  [{direction_label}]")
    print(f"Пул (всего):      {len(pool_feature_matrix)} событий  θ={args.theta}")

    # 5-6. Прогноз (точка + полоса), steps шагов
    results = run_band_forecast(
        query_vector, query_direction, last_log_price,
        pool_feature_matrix, pool_target_log_returns, pool_pivot_directions,
        args.theta, min_pool_size, args.steps,
    )

    print()
    for r in results:
        direction_sym = "▲" if r["direction"] > 0 else "▼"
        if not r["ok"]:
            print(f"Шаг {r['step']} {direction_sym} → невозможен: {r['reason']}")
            break
        band = r["band_prices"]
        print(f"Шаг {r['step']} {direction_sym} → точка={r['price']:.4f}  "
              f"(lr={r['log_return']:+.5f}, пул={r['pool_size']})")
        print(f"         полоса: 10%={band[0.1]:.4f}  25%={band[0.25]:.4f}  "
              f"50%={band[0.5]:.4f}  75%={band[0.75]:.4f}  90%={band[0.9]:.4f}")


if __name__ == "__main__":
    main()
