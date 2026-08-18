"""
Delay-space embedding for the LA method (τ = 1, series = Δratio).

Delay vector i:  [series[i], series[i+1], ..., series[i+p-1]]
Label i:         series[i+p]   (one step past the window)
"""

import numpy as np


def build_delay_matrix(
    series: np.ndarray, p: int, tau: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build delay matrix X (shape N×p) and next-step labels y (shape N).

    N = len(series) - (p-1)*tau - 1
    """
    span = (p - 1) * tau + 1
    n = len(series) - span
    if n < 1:
        raise ValueError(f"Series too short ({len(series)}) for p={p}, tau={tau}")
    offsets = np.arange(p) * tau
    X = np.stack([series[off: off + n] for off in offsets], axis=1)
    y = series[span:]
    return X, y


def last_vector(series: np.ndarray, p: int, tau: int = 1) -> np.ndarray:
    """
    Most recent delay vector from *series* — the query used to start forecasting.
    Shape: (p,), oldest component first, newest last.
    """
    needed = (p - 1) * tau + 1
    tail = series[-needed:]
    return tail[::tau][:p]
