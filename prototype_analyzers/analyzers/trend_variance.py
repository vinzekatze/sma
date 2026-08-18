"""Скользящий линейный тренд (OLS) и дисперсия остатков.

Для каждого бара t считается каузальная OLS-регрессия
log(close) ~ i, i = 0..window-1, на окне [t-window+1, t].

Возвращает:
  - slope[t]     — наклон тренда окна, заканчивающегося в t (направление
                    и крутизна, единицы: log-цена за бар);
  - resid_var[t] — дисперсия остатков (log_price - trend_line) внутри
                    этого же окна (разброс вокруг тренда).

NaN, пока истории меньше window баров. Всё каузально: окно [t-window+1, t]
не использует ничего после t.
"""
from __future__ import annotations

import numpy as np


def rolling_trend_variance(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    slope = np.full(n, np.nan)
    resid_var = np.full(n, np.nan)
    if window < 3 or n < window:
        return slope, resid_var

    j = np.arange(n, dtype=np.float64)
    cs_y = np.concatenate(([0.0], np.cumsum(y)))
    cs_jy = np.concatenate(([0.0], np.cumsum(j * y)))
    cs_yy = np.concatenate(([0.0], np.cumsum(y * y)))

    w = float(window)
    Sx = w * (w - 1) / 2.0
    Sxx = (w - 1) * w * (2 * w - 1) / 6.0
    denom = w * Sxx - Sx ** 2

    # t пробегает window-1 .. n-1 (0-индексация); j0 = t-window+1 пробегает 0 .. n-window
    Sy = cs_y[window:n + 1] - cs_y[0:n - window + 1]
    Sjy = cs_jy[window:n + 1] - cs_jy[0:n - window + 1]
    Syy = cs_yy[window:n + 1] - cs_yy[0:n - window + 1]
    j0 = np.arange(0, n - window + 1, dtype=np.float64)
    Sxy = Sjy - j0 * Sy

    s = (w * Sxy - Sx * Sy) / denom
    b = (Sy - s * Sx) / w
    ssr = Syy - b * Sy - s * Sxy

    slope[window - 1:] = s
    resid_var[window - 1:] = np.maximum(ssr, 0.0) / (w - 2)

    return slope, resid_var


def single_window_trend(y: np.ndarray, origin: int, window: int) -> tuple[int, np.ndarray, float]:
    """Тренд + std остатков ровно ОДНОГО окна [origin-window+1, origin].

    Возвращает (j0, fitted, std) — j0 - индекс начала окна, fitted -
    значения линии тренда на [j0, origin], std - std остатков (ddof=2).
    """
    j0 = origin - window + 1
    if j0 < 0:
        raise ValueError(f"недостаточно истории: origin={origin}, window={window}")

    seg = y[j0:origin + 1]
    x = np.arange(window, dtype=np.float64)
    s, b = np.polyfit(x, seg, 1)
    fitted = b + s * x
    resid = seg - fitted
    std = resid.std(ddof=2)
    return j0, fitted, std
