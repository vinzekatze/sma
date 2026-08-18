"""LWR (Locally Weighted Regression) + строгий каскадный поиск соседей.

Прямой перенос алгоритма из research/cascade_algorithm.md и его реализации
в research/phase6_attractor/113_mlp_sweep/113_mlp_sweep.py (levels_aligned,
lwr_predict, воронка каскада) — без изменений логики, только обобщение с
конкретно ряда att на произвольный 1D каузальный ряд (см. feedback
"КАСКАД — НИКАКОЙ САМОДЕЯТЕЛЬНОСТИ").

Вход везде — уже каузально обрезанный ряд series[0..n-1] (n-1 = origin);
функции предсказывают значение в точке origin+1 (не входит в series).
"""
from __future__ import annotations

import numpy as np


def levels_aligned(p_fit: int, p_max: int) -> list[int]:
    """Октавные уровни каскада p_lv_1 > ... > p_fit (по возрастанию удвоения)."""
    levs = [p_fit]
    p = p_fit
    while p * 2 <= p_max:
        p *= 2
        levs.append(p)
    return list(reversed(levs))


def lwr_predict(X_nn: np.ndarray, y_nn: np.ndarray, vec_f: np.ndarray, h_bw: float) -> float:
    w = np.exp(-0.5 * (np.linalg.norm(X_nn - vec_f, axis=1) / h_bw) ** 2)
    A = np.hstack([np.ones((len(X_nn), 1)), X_nn])
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vec_f @ c[1:])


def _build_library(series: np.ndarray, p_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Векторы длины p_dim, заканчивающиеся в t, и цель series[t+1]."""
    n = len(series)
    t_arr = np.arange(p_dim, n - 1)
    X = np.column_stack([series[t_arr - (p_dim - 1 - j)] for j in range(p_dim)])
    y = series[t_arr + 1]
    vec_f = series[-p_dim:].copy()
    return t_arr, X, y, vec_f


def _cascade_filter(X_full: np.ndarray, vec_full0: np.ndarray, levels: list[int], xi_lwr: int) -> np.ndarray:
    """Строгая воронка (без expansion) — см. cascade_algorithm.md."""
    cands = np.arange(len(X_full))
    for k_lvl, p_lvl in enumerate(levels):
        is_last = k_lvl == len(levels) - 1
        xi_clip = min(xi_lwr, len(cands))
        if len(cands) > xi_clip:
            dists = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full0[-p_lvl:], axis=1)
            cands = cands[np.argpartition(dists, xi_clip - 1)[:xi_clip]]
        if not is_last:
            p_next = levels[k_lvl + 1]
            radius = p_lvl - p_next
            offsets = np.arange(radius + 1)
            expanded = cands[:, None] - offsets[None, :]
            expanded = np.clip(expanded, 0, len(X_full) - 1)
            cands = np.unique(expanded)
    return cands


def predict_no_cascade(series: np.ndarray, p_fit: int, xi_lwr: int) -> float:
    """Одноуровневый LWR: прямой поиск xi_lwr соседей в p_fit-мерном пространстве."""
    n = len(series)
    if n - p_fit - 1 < 3:
        return float("nan")
    t_arr, X, y, vec_f = _build_library(series, p_fit)
    if len(X) < xi_lwr:
        return float("nan")
    xi_clip = min(xi_lwr, len(X))
    dists = np.linalg.norm(X - vec_f, axis=1)
    idx = np.argpartition(dists, xi_clip - 1)[:xi_clip]
    X_nn, y_nn = X[idx], y[idx]
    h_bw = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)
    return lwr_predict(X_nn, y_nn, vec_f, h_bw)


def predict_cascade(series: np.ndarray, p_fit: int, n_levels: int, xi_lwr: int) -> float:
    """LWR с каскадным сужением пула через октавные уровни p_lv."""
    p_max = p_fit * (2 ** (n_levels - 1))
    levels = levels_aligned(p_fit, p_max)
    p_top = levels[0]
    n = len(series)
    if n - p_top - 1 < 3:
        return float("nan")
    t_arr, X_full, y_base, vec_full0 = _build_library(series, p_top)
    if len(X_full) < xi_lwr:
        return float("nan")
    cands = _cascade_filter(X_full, vec_full0, levels, xi_lwr)
    if len(cands) < p_fit + 2:
        return float("nan")
    X_nn = X_full[cands, -p_fit:]
    y_nn = y_base[cands]
    vec_f = vec_full0[-p_fit:]
    h_bw = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)
    return lwr_predict(X_nn, y_nn, vec_f, h_bw)
