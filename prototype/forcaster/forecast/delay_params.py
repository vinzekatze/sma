"""
Estimation of delay-space parameters for the LA method.

  τ — delay time: first zero crossing of the ACF of the normalised ratio series.
  p — embedding dimension: chosen empirically (start 3–7, see note below).

Lecture rules (Loskutov, ch. 10–13):
  - τ: near first zero of ACF. Small τ → vectors collinear (false neighbours);
    large τ → components decouple and carry no mutual information.
  - p: empirical. Stop increasing when forecast quality stops improving.
    Reliable only when N > 10^p  (for N=44k: p ≤ 4–5 solid, 6–7 borderline).
  - Neighbours for LA1: Ξ ≥ 3(p+1).
"""

import numpy as np


def compute_acf(series: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """
    Autocorrelation function for lags 0..max_lag, normalised so acf[0] = 1.

    Uses a direct dot-product loop — O(n * max_lag) but straightforward
    and accurate. For n=44k, max_lag=200 runs in well under one second.
    """
    x = series - series.mean()
    std = x.std()
    if std == 0:
        return np.ones(max_lag + 1)
    x = x / std
    n = len(x)
    return np.array([
        float(np.dot(x[: n - lag], x[lag:]) / n) if lag < n else 0.0
        for lag in range(max_lag + 1)
    ])


def first_acf_zero(acf: np.ndarray) -> int:
    """
    Index of the first lag at which ACF crosses or touches zero.
    Falls back to len(acf)-1 if no crossing is found.
    """
    for i in range(1, len(acf)):
        if acf[i] <= 0.0:
            return i
    return len(acf) - 1


def recommended_p(n_samples: int) -> tuple[int, int]:
    """
    Return (p_safe_max, p_border_max) based on data length.
    Lecture: reliable when N > 10^p  →  p < log10(N).
    """
    import math
    limit = math.log10(n_samples)
    return int(limit) - 1, int(limit)
