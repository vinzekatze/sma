"""
range_forecast — прогноз ДИАПАЗОНА колебаний цены (min(low)..max(high)) за
h=1..H баров вперёд, БЕЗ прогноза точки close. Portировано из
prototype/forcaster/ui/app16-range-forecast.py (сессия 2026-09-06) — см.
docs/plans/app16_range_forecast_migration_plan.md.

Метод: K аналогичных исторических ситуаций (dratio delay-embedding, S-map),
якорь — ИЗВЕСТНЫЙ close[origin] (прогноз close не строится вообще). У
каждого соседа смотрим его СОБСТВЕННЫЙ реализованный H-барный диапазон
относительно ЕГО ЖЕ close, взвешенно (band_lambda-style weighted_quantile)
переносим на текущий close[origin]. Один поиск соседей переиспользуется для
всех шагов h=1..H и всех уровней покрытия (50/75/90%) — см.
rolling_cummin_cummax/predict_zones_multistep.

Экспонента S-map: exp(-(θ·d_shape/d_mean + Σ_f λ_f·(rank_pool_f -
rank_query_f)²)) — d_shape — L2-расстояние по форме Δratio, rank_* — готовые
каузальные bar-native ранги band_lambda (volume/trend/velocity/acceleration/
volatility, compute_bar_rank_dict) — переиспользованы БЕЗ изменений (см.
feedback_critical_libraries — reuse, not reinvent). la0/simplex (жёсткий
k-NN) НЕ перенесены в прод — smap валидирован как единственный рабочий
режим (README эксп.02, критика §5).

Калибратор (θ+λ+read-квантиль по уровням) — range_forecast_calibrator.py.
"""
from __future__ import annotations

import numpy as np

from .band_lambda import compute_bar_rank_dict, weighted_quantile
from .normalize import _logtrend_causal

RANK_FEATURES = ["volume", "trend", "velocity", "acceleration", "volatility"]  # band_lambda.BAR_FEATURES, no leg_age
MAX_CANDLES_DEFAULT = 5000  # обрезка входных данных — см. план §8 п.3, ограничивает стоимость независимо от масштаба


def truncate_candles(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray,
                      max_candles: int = MAX_CANDLES_DEFAULT):
    """Последние max_candles баров — общее ограничение стоимости калибровки/
    прогноза вместо специального случая под 1h (план §8 п.3). Параметр, НЕ
    молчаливая константа — пробрасывается из запроса/UI (по образцу
    simplex_ensemble.bars: "точек в библиотеке"/"все точки библиотеки").
    max_candles<=0 означает "без обрезки, вся история"."""
    if max_candles <= 0 or len(close) <= max_candles:
        return close, high, low, volume
    return close[-max_candles:], high[-max_candles:], low[-max_candles:], volume[-max_candles:]


def dratio_from_close(close: np.ndarray) -> np.ndarray:
    """Причинный logtrend (project standard, normalize._logtrend_causal) →
    ratio = close/trend → dratio = diff(ratio). Используется и kernel'ом, и
    калибратором — единая точка вычисления, чтобы числа не разъезжались."""
    trend = _logtrend_causal(close)
    ratio = close / np.maximum(trend, 1e-10)
    return np.diff(ratio)


def build_delay_matrix(series: np.ndarray, p: int):
    n_rows = len(series) - p
    if n_rows < 1:
        return np.zeros((0, p)), np.zeros(0)
    X = series[np.arange(n_rows)[:, None] + np.arange(p)]
    y = series[p:]
    return X, y


def rolling_cummin_cummax(low: np.ndarray, high: np.ndarray, H_max: int):
    """cml[t, h-1] = min(low[t+1..t+h]), cmh[t, h-1] = max(high[t+1..t+h]),
    for h=1..H_max in one pass — каждый шаг мультишагового прогноза читает
    один и тот же массив, без пересканирования на шаг."""
    n = len(low)
    cml = np.full((n, H_max), np.nan)
    cmh = np.full((n, H_max), np.nan)
    if n - H_max < 1:
        return cml, cmh
    wl = np.lib.stride_tricks.sliding_window_view(low[1:], H_max)
    wh = np.lib.stride_tricks.sliding_window_view(high[1:], H_max)
    m = len(wl)
    cml[:m] = np.minimum.accumulate(wl, axis=1)
    cmh[:m] = np.maximum.accumulate(wh, axis=1)
    return cml, cmh


def find_range_neighbors(dratio: np.ndarray, roll_low: np.ndarray, origin: int, p: int, theiler: int,
                          theta: float, lambdas: dict | None = None, rank_dict: dict | None = None):
    """S-map neighbor search (единственный режим в проде — la0/simplex не
    перенесены, см. модульный докстринг). lambdas/rank_dict — опциональные
    λ-взвешенные bar-native признаки band_lambda в экспоненте, наравне с θ.
    NaN-ранги (окно прогрева признака) дают NaN-вес, отфильтровываются
    существующей проверкой валидности ниже — специальной обработки не нужно."""
    pool_end = origin - theiler
    if pool_end < p + 5:
        return None
    X_all, _y_all = build_delay_matrix(dratio[:pool_end], p)
    if len(X_all) < 5:
        return None
    q = dratio[origin - p:origin]

    d = np.linalg.norm(X_all - q, axis=1)
    d_mean = max(float(d.mean()), 1e-10)
    nb_bar = np.arange(len(X_all)) + p
    sq = theta * d / d_mean
    if lambdas and rank_dict:
        for f in RANK_FEATURES:
            lam = lambdas.get(f, 0.0)
            if not lam:
                continue
            rv = rank_dict[f]
            rank_q = rv[origin] if origin < len(rv) else np.nan
            clipped = np.clip(nb_bar, 0, len(rv) - 1)
            pool_vals = np.where(nb_bar < len(rv), rv[clipped], np.nan)
            sq = sq + lam * (pool_vals - rank_q) ** 2
    w = np.exp(-sq)

    valid = ((nb_bar >= 0) & (nb_bar < len(roll_low))
             & np.isfinite(roll_low[np.clip(nb_bar, 0, len(roll_low) - 1)]) & np.isfinite(w))
    if valid.sum() < 5:
        return None
    wv = w[valid]
    return nb_bar[valid], wv / wv.sum()


def _level_read_quantiles(level, q_read_by_level: dict | None):
    """(q_lo, q_hi) для чтения квантиля пула для этого уровня — калиброванный
    per-level read-квантиль (range_forecast_calibrator.py), либо номинал."""
    q_lo, q_hi = (100 - level) / 200.0, (100 + level) / 200.0
    if q_read_by_level and str(level) in {str(k) for k in q_read_by_level}:
        qr = q_read_by_level.get(level, q_read_by_level.get(str(level)))
        return qr["low"], qr["high"]
    return q_lo, q_hi


def predict_zones_multistep(dratio: np.ndarray, close: np.ndarray, cml: np.ndarray, cmh: np.ndarray,
                             origin: int, p: int, theiler: int, theta: float, levels: tuple, H_max: int,
                             lambdas: dict | None = None, rank_dict: dict | None = None,
                             q_read_by_level: dict | None = None):
    """Один поиск соседей — валидность гейтится по САМОМУ ДАЛЬНЕМУ шагу H_max
    (самый строгий случай, поэтому валиден и для всех меньших h) — затем
    дешёвое чтение квантиля по шагу из cml/cmh. Возвращает {h: {level: (low,
    high)}} для h=1..H_max, + число соседей."""
    res = find_range_neighbors(dratio, cml[:, H_max - 1], origin, p, theiler, theta, lambdas, rank_dict)
    if res is None:
        return None
    nb_bar_v, wv = res
    nb_close = close[nb_bar_v]
    anchor = close[origin]
    zones_by_step = {}
    for h in range(1, H_max + 1):
        off_low = (cml[nb_bar_v, h - 1] - nb_close) / np.maximum(nb_close, 1e-10)
        off_high = (cmh[nb_bar_v, h - 1] - nb_close) / np.maximum(nb_close, 1e-10)
        zones = {}
        for level in levels:
            q_lo, q_hi = _level_read_quantiles(level, q_read_by_level)
            b_low = weighted_quantile(off_low, wv, (q_lo,))[q_lo]
            b_high = weighted_quantile(off_high, wv, (q_hi,))[q_hi]
            zones[level] = (anchor * (1 + b_low), anchor * (1 + b_high))
        zones_by_step[h] = zones
    return zones_by_step, len(nb_bar_v)
