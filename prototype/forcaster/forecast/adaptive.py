"""
Адаптивные параметры прогнозатора:

  find_optimal_ma   — подбирает окно MA, минимизирующее Hurst ratio-ряда
                      (чем меньше Hurst → тем более возвратный ряд → лучше LA)

  regime_mask_for_pool — строит булеву маску для пула задержек:
                         True = вектор задержек из «предсказуемой» зоны (H < thr)
"""

from __future__ import annotations

import numpy as np

from .normalize import normalize
from .hurst import rolling_hurst


# ── адаптивное окно MA ────────────────────────────────────────────────────────

def find_optimal_ma(
    candles: list[dict],
    ma_min: int = 200,
    ma_max: int = 5000,
    n_steps: int = 25,
    hurst_window: int = 200,
    hurst_step: int = 40,
) -> tuple[int, dict[int, float]]:
    """
    Перебирает MA-окна от ma_min до ma_max (n_steps шагов),
    для каждого вычисляет средний Hurst Δratio-ряда.

    Returns
    -------
    best_ma   : MA-окно с минимальным средним Hurst
    scores    : {ma_window: mean_hurst}
    """
    step = max(1, (ma_max - ma_min) // (n_steps - 1))
    windows = list(range(ma_min, ma_max + 1, step))

    scores: dict[int, float] = {}
    for ma_w in windows:
        norm   = normalize(candles, window=ma_w)
        valid  = norm.dropna(subset=["ma"]).reset_index(drop=True)
        dratio = np.diff(valid["ratio"].values)
        _, h_vals = rolling_hurst(dratio, window=hurst_window, step=hurst_step)
        scores[ma_w] = float(np.nanmean(h_vals))

    best_ma = min(scores, key=scores.get)
    return best_ma, scores


# ── адаптивный порог Hurst (по процентилю распределения) ─────────────────────

def hurst_percentile_threshold(
    h_vals: np.ndarray,
    percentile: float,
) -> float:
    """
    Возвращает значение H соответствующее N-му процентилю.
    Например, percentile=30 → порог, ниже которого 30% всех H-значений.
    """
    return float(np.nanpercentile(h_vals, percentile))


def regime_mask_for_pool(
    bar_hurst: np.ndarray,   # H для каждого бара valid-ряда (len = len(valid))
    pool_size: int,           # кол-во строк в delay-матрице
    p: int,
    threshold: float,
) -> np.ndarray:
    """
    Булева маска длиной pool_size.
    Строка k delay-матрицы охватывает бары [k, k+p-1]; центр ≈ k + p//2.
    Помечаем строку как «предсказуемую», если H в центре < threshold.
    """
    n = len(bar_hurst)
    centers = np.minimum(np.arange(pool_size) + p // 2, n - 1)
    return bar_hurst[centers] < threshold
