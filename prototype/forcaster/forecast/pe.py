"""
Permutation Entropy (PE) — локальная мера стохастичности временного ряда.

Низкая PE → ряд содержит повторяющиеся упорядоченные паттерны → детерминированная структура.
Высокая PE → паттерны случайны → стохастика.

Исследование 18/19 показало: структурные участки Δratio (PE < q25) дают
в 3-4× меньше ложных соседей по FNN при m=2–3 — эффект устойчив по всем 8 тикерам.
"""

from __future__ import annotations
from itertools import permutations
from math import factorial

import numpy as np


def _make_perm_index(order: int) -> dict[tuple[int, ...], int]:
    return {p: i for i, p in enumerate(permutations(range(order)))}


# кэш индексов для типичных порядков
_PERM_CACHE: dict[int, dict] = {}


def perm_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Нормализованная Permutation Entropy для вектора x.

    order: длина паттерна (рекомендуется 3–5)
    delay: задержка между элементами паттерна
    Returns: PE ∈ [0, 1]; 0 — полностью упорядоченный, 1 — максимально случайный.
    """
    run = order * delay
    if len(x) < run:
        return np.nan
    if order not in _PERM_CACHE:
        _PERM_CACHE[order] = _make_perm_index(order)
    idx_map = _PERM_CACHE[order]
    counts = np.zeros(factorial(order))
    for i in range(len(x) - run + delay):
        w = x[i: i + run: delay][:order]
        counts[idx_map[tuple(np.argsort(w))]] += 1
    p = counts[counts > 0]
    p /= p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(factorial(order)))


def rolling_pe(series: np.ndarray, win: int = 50,
               order: int = 3, delay: int = 1) -> np.ndarray:
    """
    Rolling Permutation Entropy вдоль series.
    Первые (win-1) значений = NaN.
    """
    result = np.full(len(series), np.nan)
    for i in range(win - 1, len(series)):
        result[i] = perm_entropy(series[i - win + 1: i + 1], order, delay)
    return result


def pe_adaptive_mask(
    bar_pe: np.ndarray,
    pool_size: int,
    p: int,
    base_quantile: float,
    n_neighbors: int,
    step: float = 0.05,
    max_quantile: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Адаптивная маска по PE.

    Начинает с base_quantile и поднимает порог шагами step, пока в маске
    не наберётся >= n_neighbors строк. Останавливается не позже max_quantile.
    Если и при max_quantile недостаточно — полный fallback (все строки True).

    Returns: (mask, used_quantile)
      used_quantile == 1.0 при достижении верхней границы без fallback.
      used_quantile  > 1.0 означает полный fallback.
    """
    valid_pe = bar_pe[~np.isnan(bar_pe)]
    if len(valid_pe) == 0 or pool_size == 0:
        return np.ones(pool_size, dtype=bool), max_quantile + step

    q = base_quantile
    while q <= max_quantile + 1e-9:
        thr = float(np.quantile(valid_pe, min(q, 1.0)))
        mask = pe_mask_for_pool(bar_pe, pool_size, p, thr)
        if mask.sum() >= n_neighbors:
            return mask, min(q, max_quantile)
        q = round(q + step, 10)

    # полный fallback
    return np.ones(pool_size, dtype=bool), max_quantile + step


def pe_proximity_mask(
    bar_pe: np.ndarray,
    pool_size: int,
    p: int,
    pe_origin: float,
    n_neighbors: int,
    base_delta: float = 0.005,
    max_delta: float = 0.05,
    step: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Маска по близости PE к точке отсчёта.

    Выбирает строки пула, у которых |PE_k − pe_origin| ≤ delta.
    Адаптивно расширяет delta шагами step пока не наберётся >= n_neighbors строк.
    Если при max_delta всё равно мало — полный fallback.

    Returns: (mask, used_delta)
      used_delta > max_delta означает полный fallback.
    """
    if step is None:
        step = base_delta

    if pool_size == 0 or np.isnan(pe_origin):
        return np.ones(pool_size, dtype=bool), max_delta + step

    centers  = np.minimum(np.arange(pool_size) + p // 2, len(bar_pe) - 1)
    pool_pe  = bar_pe[centers]
    valid    = ~np.isnan(pool_pe)
    diffs    = np.where(valid, np.abs(pool_pe - pe_origin), np.inf)

    delta = base_delta
    while delta <= max_delta + 1e-9:
        mask = diffs <= delta
        if mask.sum() >= n_neighbors:
            return mask, min(delta, max_delta)
        delta = round(delta + step, 10)

    return np.ones(pool_size, dtype=bool), max_delta + step


def pe_mask_for_pool(
    bar_pe: np.ndarray,
    pool_size: int,
    p: int,
    threshold: float,
) -> np.ndarray:
    """
    Булева маска длиной pool_size для пула задержек.

    Строка k delay-матрицы охватывает бары [k, k+p-1]; центр ≈ k + p//2.
    Помечаем строку как «структурную» если PE в центре < threshold.
    Аналог regime_mask_for_pool из adaptive.py.

    bar_pe : rolling PE для каждого бара dratio-ряда
    pool_size : кол-во строк в delay-матрице (≈ origin_k - p)
    threshold : PE < threshold → структурный участок
    """
    n = len(bar_pe)
    centers = np.minimum(np.arange(pool_size) + p // 2, n - 1)
    valid = ~np.isnan(bar_pe[centers])
    below = bar_pe[centers] < threshold
    return valid & below
