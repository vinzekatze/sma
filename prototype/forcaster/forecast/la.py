"""
LA1 — Local Approximation, first order.

Algorithm (Loskutov, ch. 12-13):
  1. Build delay matrix from Δratio history up to origin
  2. Find Ξ nearest-neighbour delay vectors (Euclidean, L2)
  3. Fit linear model  y ≈ a₀ + X·a  on the neighbours via SVD (numpy lstsq)
  4. Predict one Δratio step; shift delay vector; repeat for horizon

Reconstruction:
  ratio[t+h] = ratio[t] + Σ dratio_hat[0..h]
  price[t+h] = ratio[t+h] * MA[t]   (MA assumed constant over horizon)
"""

from __future__ import annotations
import numpy as np
from .embedding import build_delay_matrix, last_vector


# ── PCA projection ────────────────────────────────────────────────────────────

def _pca_transform(
    X: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project X (shape N×p) onto its top-k principal components.

    Uses the p×p covariance matrix (cheap when p << N).

    Returns
    -------
    X_proj : shape (N, k)  — projected pool
    mean   : shape (p,)    — column means of X
    V_k    : shape (k, p)  — top-k eigenvectors (row = PC direction)
    """
    mean  = X.mean(axis=0)
    Xc    = X - mean
    cov   = Xc.T @ Xc / max(len(Xc) - 1, 1)        # (p, p)
    vals, vecs = np.linalg.eigh(cov)                 # ascending eigenvalues
    V_k   = vecs[:, ::-1][:, :k].T                  # (k, p), descending variance
    return Xc @ V_k.T, mean, V_k


def _apply_pca(vec: np.ndarray, mean: np.ndarray, V_k: np.ndarray) -> np.ndarray:
    """Project a single vector into the PCA space computed by _pca_transform."""
    return (vec - mean) @ V_k.T


# ── Huber IRLS ────────────────────────────────────────────────────────────────

def _huber_irls(
    A: np.ndarray,
    y: np.ndarray,
    init_w: np.ndarray | None = None,
    delta: float = 1.35,
    max_iter: int = 30,
) -> np.ndarray:
    """
    Iteratively reweighted LS for Huber loss.

    A   : design matrix with bias column already included, shape (n, p+1)
    y   : targets, shape (n,)
    init_w : optional prior weights (e.g. Gaussian from LWR).  When given,
             the effective weight at each step is init_w * huber_w.
    delta  : Huber threshold in units of the robust scale MAD/0.6745
    """
    prior = init_w if init_w is not None else np.ones(len(y))
    sw = np.sqrt(prior)
    coeffs, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y, rcond=None)

    for _ in range(max_iter):
        r = y - A @ coeffs
        scale = max(float(np.median(np.abs(r))) / 0.6745, 1e-10)
        hub_w = np.where(
            np.abs(r) <= delta * scale,
            1.0,
            (delta * scale) / (np.abs(r) + 1e-12),
        )
        combined_w = prior * hub_w
        sw = np.sqrt(combined_w)
        c_new, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y, rcond=None)
        if np.max(np.abs(c_new - coeffs)) < 1e-7:
            return c_new
        coeffs = c_new
    return coeffs


# ── primitive operations ──────────────────────────────────────────────────────

def find_neighbors(
    X: np.ndarray,
    query: np.ndarray,
    n: int,
    norm_vecs: bool = False,
) -> np.ndarray:
    """
    Indices of the n nearest rows in X to query (L2).

    norm_vecs: if True, divide every vector (rows of X and query) by its own
    std before computing distances.  Quiet-period and volatile-period vectors
    with the *same shape* of movement then become neighbours regardless of
    amplitude.  Vectors whose std < 1e-8 are left unchanged.
    """
    if norm_vecs:
        eps = 1e-8
        s = np.std(X, axis=1, keepdims=True)
        s[s < eps] = eps
        X_cmp = X / s
        sq = float(np.std(query))
        q_cmp = query / max(sq, eps)
    else:
        X_cmp, q_cmp = X, query
    dists = np.linalg.norm(X_cmp - q_cmp, axis=1)
    return np.argsort(dists)[:n]


def fit_la1(X: np.ndarray, y: np.ndarray, use_huber: bool = False) -> np.ndarray:
    """
    Fit y ≈ a₀ + X·a.  Returns coefficients [a₀, a₁, …, aₚ], shape (p+1,).
    use_huber: replace OLS with Huber IRLS — reduces influence of outlier y values.
    """
    A = np.hstack([np.ones((len(X), 1)), X])
    if use_huber:
        return _huber_irls(A, y)
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs


def apply_la1(vec: np.ndarray, coeffs: np.ndarray) -> float:
    return float(coeffs[0] + vec @ coeffs[1:])


# ── main forecast ─────────────────────────────────────────────────────────────

def forecast_la1(
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
    Iterative LA1 forecast on a Δratio series.

    Args:
        dratio:       Full Δratio array  (dratio[i] = ratio[i+1] − ratio[i])
        origin_k:     Index in the *ratio* series at which to start forecasting.
        p:            Embedding dimension
        n_neighbors:  Nearest neighbours Ξ (lecture rule: ≥ 3(p+1))
        horizon:      Steps to predict
        tau:          Delay time (only tau=1 supported)
        regime_mask:  Optional bool array of length len(X). When provided, search
                      is restricted to True rows; falls back to full pool if
                      fewer than n_neighbors rows are available.
        pca_k:        If > 0, project pool to k principal components before
                      neighbour search (applied after norm_vecs if both enabled).
                      Regression always uses the original space.

    Returns:
        dratio_hat: shape (horizon,)
    """
    if tau != 1:
        raise NotImplementedError("tau > 1 not yet supported")

    history = dratio[:origin_k]
    X, y    = build_delay_matrix(history, p, tau)
    vec     = last_vector(history, p, tau).copy()
    out     = np.empty(horizon)

    # apply regime filter once (pool is static during forecast)
    if regime_mask is not None and regime_mask.sum() >= n_neighbors:
        Xs, ys = X[regime_mask], y[regime_mask]
    else:
        Xs, ys = X, y

    # ── precompute comparison space (once, before the iterative loop) ──────────
    _use_pca = pca_k > 0 and pca_k < p
    if _use_pca:
        # σ-normalise rows first when requested, then PCA
        eps = 1e-8
        if norm_vecs:
            s = np.std(Xs, axis=1, keepdims=True)
            s[s < eps] = eps
            Xs_pre = Xs / s
        else:
            Xs_pre = Xs
        Xs_cmp, _pca_mean, _pca_V = _pca_transform(Xs_pre, pca_k)
    elif norm_vecs:
        eps = 1e-8
        s = np.std(Xs, axis=1, keepdims=True)
        s[s < eps] = eps
        Xs_cmp = Xs / s
        _pca_mean = _pca_V = None
    else:
        Xs_cmp = Xs
        _pca_mean = _pca_V = None

    for h in range(horizon):
        # project current query vector into the comparison space
        if _use_pca:
            if norm_vecs:
                sv = max(float(np.std(vec)), 1e-8)
                vec_cmp = _apply_pca(vec / sv, _pca_mean, _pca_V)
            else:
                vec_cmp = _apply_pca(vec, _pca_mean, _pca_V)
        elif norm_vecs:
            sv = max(float(np.std(vec)), 1e-8)
            vec_cmp = vec / sv
        else:
            vec_cmp = vec

        idx    = find_neighbors(Xs_cmp, vec_cmp, n_neighbors, norm_vecs=False)
        coeffs = fit_la1(Xs[idx], ys[idx], use_huber=use_huber)
        val    = apply_la1(vec, coeffs)
        out[h] = val
        vec    = np.roll(vec, -1)
        vec[-1] = val

    return out


# ── reconstruction ────────────────────────────────────────────────────────────

def reconstruct_price(
    dratio_hat: np.ndarray,
    ratio_at_origin: float,
    ma_at_origin: float,
) -> np.ndarray:
    """
    Convert Δratio forecast → price forecast.

    ratio_hat[h] = ratio_at_origin + cumsum(dratio_hat)[h]
    price_hat[h] = ratio_hat[h] * ma_at_origin   (MA treated as constant)
    """
    ratio_hat = ratio_at_origin + np.cumsum(dratio_hat)
    return ratio_hat * ma_at_origin
