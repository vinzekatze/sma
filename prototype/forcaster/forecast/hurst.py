"""
Rolling Hurst exponent for market regime detection.

H < 0.5  — mean-reverting (LA more reliable)
H ≈ 0.5  — random walk
H > 0.5  — trending/persistent (LA unreliable, avoid forecasting)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def hurst_rs(series: np.ndarray) -> float:
    """
    R/S Hurst estimate with log-spaced lags.
    Returns NaN when series is too short or constant.
    """
    n = len(series)
    max_exp = int(np.log2(n // 2))
    if max_exp < 3:
        return np.nan

    # start from 8 — lags 2 and 4 give trivial R/S≈1 and bias H upward
    lags = [2 ** k for k in range(3, max_exp + 1)]
    rs_pts: list[tuple[int, float]] = []

    for lag in lags:
        chunks = [series[i: i + lag] for i in range(0, n - lag + 1, lag)]
        rs_vals = []
        for c in chunks:
            s = np.std(c, ddof=1)
            if s == 0:
                continue
            dev = np.cumsum(c - np.mean(c))
            rs_vals.append((dev.max() - dev.min()) / s)
        if rs_vals:
            rs_pts.append((lag, float(np.mean(rs_vals))))

    if len(rs_pts) < 3:
        return np.nan

    lx = np.log([p[0] for p in rs_pts])
    ly = np.log([p[1] for p in rs_pts])
    return float(np.polyfit(lx, ly, 1)[0])


def rolling_hurst(
    series: np.ndarray,
    window: int = 200,
    step: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Hurst in a rolling window.

    Returns
    -------
    indices : array of bar indices (relative to *series*) at window end
    values  : Hurst values; NaN where computation failed
    """
    n = len(series)
    centers = np.arange(window, n, step)
    values  = np.array([hurst_rs(series[c - window: c]) for c in centers])
    return centers, values


def unpredictable_zones(
    bar_times: np.ndarray,          # timestamps aligned with *series* (len = len(dratio)+1 or len(valid))
    hurst_indices: np.ndarray,      # output of rolling_hurst — indices into dratio
    hurst_values: np.ndarray,
    threshold: float = 0.5,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Merge consecutive windows with H >= threshold into time-range pairs.
    """
    bad = hurst_values >= threshold
    if not bad.any():
        return []

    zones: list[tuple] = []
    in_zone  = False
    t_start  = None

    for idx, is_bad in zip(hurst_indices, bad):
        t = pd.Timestamp(bar_times[min(int(idx), len(bar_times) - 1)])
        if is_bad and not in_zone:
            in_zone = True
            t_start = t
        elif not is_bad and in_zone:
            in_zone = False
            zones.append((t_start, t))
    if in_zone and t_start is not None:
        zones.append((t_start, pd.Timestamp(bar_times[-1])))

    return zones
