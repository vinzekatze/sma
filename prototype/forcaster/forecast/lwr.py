"""
LWR — Locally Weighted Regression (Gaussian kernel).

Отличие от LA1 только в шаге подгонки модели:
  LA1 : отобрать Ξ ближайших соседей, равные веса
  LWR : отобрать те же Ξ ближайших соседей, затем взвесить их гауссианой
        w_i = exp(−d_i² / 2h²),  h = d(Ξ-го соседа)   (адаптивная полоса)

Ключевое отличие от наивной реализации "веса для всех 44k точек":
только Ξ соседей попадают в WLS — иначе глобальное среднее подавляет
локальную структуру и прогноз вырождается в плоскую линию.
"""

from __future__ import annotations
import numpy as np
from .embedding import build_delay_matrix, last_vector
from .la import _huber_irls, _pca_transform, _apply_pca


def forecast_lwr(
    dratio: np.ndarray,
    origin_k: int,
    p: int,
    n_neighbors: int,
    horizon: int,
    tau: int = 1,
    regime_mask: np.ndarray | None = None,
    norm_vecs: bool = False,
    use_huber: bool = False,
    pca_k: int = 0,
) -> np.ndarray:
    """
    Iterative LWR forecast на Δratio.

    Шаги за один итерационный шаг:
    1. Найти Ξ ближайших соседей (жёсткий отбор, как в LA1)
    2. Взвесить их гауссианой с h = dist(Ξ-го соседа)
    3. WLS: lstsq(diag(√w)·[1|X_nn],  diag(√w)·y_nn)

    Args:
        dratio:      полный Δratio ряд
        origin_k:    индекс в ratio-ряде, откуда начинаем прогноз
        n_neighbors: Ξ — размер жёсткого соседства (рекомендуется ≥ 3(p+1))
        horizon:     шагов прогноза
        tau:         задержка (только 1)
    """
    if tau != 1:
        raise NotImplementedError("tau > 1 not supported")

    history = dratio[:origin_k]
    X, y    = build_delay_matrix(history, p, tau)
    current = last_vector(history, p, tau).copy()
    out     = np.empty(horizon)

    if regime_mask is not None and regime_mask.sum() >= n_neighbors:
        X, y = X[regime_mask], y[regime_mask]

    # ── precompute comparison space once ───────────────────────────────────────
    eps = 1e-8
    _use_pca = pca_k > 0 and pca_k < p
    if _use_pca:
        if norm_vecs:
            s = np.std(X, axis=1, keepdims=True)
            s[s < eps] = eps
            X_pre = X / s
        else:
            X_pre = X
        X_cmp, _pca_mean, _pca_V = _pca_transform(X_pre, pca_k)
    elif norm_vecs:
        s = np.std(X, axis=1, keepdims=True)
        s[s < eps] = eps
        X_cmp = X / s
        _pca_mean = _pca_V = None
    else:
        X_cmp = X
        _pca_mean = _pca_V = None

    for h in range(horizon):
        # project current query into comparison space
        if _use_pca:
            if norm_vecs:
                sv = max(float(np.std(current)), eps)
                q_cmp = _apply_pca(current / sv, _pca_mean, _pca_V)
            else:
                q_cmp = _apply_pca(current, _pca_mean, _pca_V)
        elif norm_vecs:
            sv = max(float(np.std(current)), eps)
            q_cmp = current / sv
        else:
            q_cmp = current

        dists = np.linalg.norm(X_cmp - q_cmp, axis=1)

        # жёсткий отбор Ξ ближайших (как в LA1)
        nn_idx   = np.argpartition(dists, n_neighbors)[:n_neighbors]
        nn_dists = dists[nn_idx]

        # гауссова полоса = расстояние до самого дальнего из Ξ соседей
        h_bw = nn_dists.max()
        if h_bw < 1e-12:
            h_bw = 1e-10

        weights = np.exp(-0.5 * (nn_dists / h_bw) ** 2)

        X_nn = X[nn_idx]
        y_nn = y[nn_idx]
        A_nn = np.hstack([np.ones((n_neighbors, 1)), X_nn])

        if use_huber:
            coeffs = _huber_irls(A_nn, y_nn, init_w=weights)
        else:
            w_sqrt = np.sqrt(weights)
            coeffs, _, _, _ = np.linalg.lstsq(
                w_sqrt[:, None] * A_nn,
                w_sqrt * y_nn,
                rcond=None,
            )

        val         = float(coeffs[0] + current @ coeffs[1:])
        out[h]      = val
        current     = np.roll(current, -1)
        current[-1] = val

    return out
