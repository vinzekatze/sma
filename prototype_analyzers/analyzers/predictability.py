"""Обобщённый walk-forward тест предсказуемости произвольного каузального
1D-ряда: персистенция, экстраполяция по дельте (pred = x[t]+(x[t]-x[t-1])),
LWR без каскада, LWR с каскадом. Общий код для slope- и accel- тестов
предсказуемости (см. slope_predictability_*.py, accel_predictability_*.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lwr_cascade import predict_no_cascade, predict_cascade


def run_walk_forward(series: np.ndarray, valid_start: int, p_fit: int, n_levels: int,
                     xi_lwr: int, n_origins: int, step: int) -> pd.DataFrame:
    """series — полный ряд (может содержать NaN до valid_start); каждый origin
    оценивается на срезе series[valid_start:origin+1] (каузально, без NaN)."""
    p_max = p_fit * (2 ** (n_levels - 1))
    min_origin = valid_start + p_max + xi_lwr + 5

    last_origin = len(series) - 2
    origins = list(range(last_origin - (n_origins - 1) * step, last_origin + 1, step))
    origins = [o for o in origins if o >= min_origin]

    rows = []
    for origin in origins:
        s = series[valid_start:origin + 1]
        true_val = series[origin + 1]
        pred_persist = s[-1]
        pred_delta = s[-1] + (s[-1] - s[-2])
        pred_nocascade = predict_no_cascade(s, p_fit, xi_lwr)
        pred_cascade = predict_cascade(s, p_fit, n_levels, xi_lwr)
        rows.append(dict(origin=origin, true=true_val,
                          pred_persist=pred_persist, pred_delta=pred_delta,
                          pred_nocascade=pred_nocascade, pred_cascade=pred_cascade))
    return pd.DataFrame(rows)
