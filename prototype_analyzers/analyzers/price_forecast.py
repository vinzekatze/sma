"""Восстановление цены из slope-прогноза (метод «ускорение» — победитель
slope_predictability_test.py/slope_predictability_sweep.py) + доверительный
интервал из resid_var.

Горизонт h=1 (следующий бар) — однократное интегрирование, без накопления
ошибки цепочкой (см. CLAUDE.md: «реконструкция цены — всегда однократный
cumsum»). Всё считается в log-пространстве, обратно в цену — через exp().
"""
from __future__ import annotations

import numpy as np

from .trend_variance import rolling_trend_variance


def rolling_window_mean(y: np.ndarray, window: int) -> np.ndarray:
    n = len(y)
    mean = np.full(n, np.nan)
    if n < window:
        return mean
    cs = np.concatenate(([0.0], np.cumsum(y)))
    s = cs[window:] - cs[:n - window + 1]
    mean[window - 1:] = s / window
    return mean


def forecast_next_bar(log_price: np.ndarray, window: int) -> dict[str, np.ndarray]:
    """Прогноз log_price[t+1] для каждого t (каузально, окно [t-window+1, t]).

    anchor[t]     — значение линии тренда окна в самой точке t (конец окна):
                     mean(y_window) + slope[t]*(window-1)/2
    pred_slope[t] — экстраполяция по ускорению: slope[t] + (slope[t]-slope[t-1])
    pred_log[t]   — anchor[t] + pred_slope[t]  ~ прогноз log_price[t+1]
    var[t]        — resid_var[t] (для полосы неопределённости, ±k·sqrt(var))
    """
    slope, var = rolling_trend_variance(log_price, window)
    mean_y = rolling_window_mean(log_price, window)
    anchor = mean_y + slope * (window - 1) / 2.0

    pred_slope = np.full(len(log_price), np.nan)
    pred_slope[1:] = slope[1:] + (slope[1:] - slope[:-1])

    pred_log = anchor + pred_slope
    return dict(slope=slope, var=var, anchor=anchor, pred_slope=pred_slope, pred_log=pred_log)
