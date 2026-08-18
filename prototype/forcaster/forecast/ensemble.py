"""
Ensemble LA1 forecast: average LA1 predictions over a grid of
(ma_window, p) combinations.

Each combination uses its own normalisation and delay embedding.
All individual forecasts are anchored to the same origin close price
(ratio * MA = close regardless of window), so averaging is meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .normalize import normalize
from .la import forecast_la1, reconstruct_price


def range_vals(start: int, end: int, step: int) -> list[int]:
    """
    Inclusive integer range.
    step == 0 or start == end  →  single value [start].
    """
    if step == 0 or start == end:
        return [start]
    if step < 0 or end < start:
        return [start]
    return list(range(start, end + 1, step))


def forecast_ensemble(
    candles: list[dict],
    origin_ts: pd.Timestamp,
    ma_windows: list[int],
    p_values: list[int],
    horizon: int,
    tau: int = 1,
) -> tuple[np.ndarray, list[np.ndarray], list[tuple[int, int]]]:
    """
    Run LA1 for every (ma_window, p) pair; return mean and individuals.

    Returns
    -------
    mean_price  : shape (horizon,)   — averaged price forecast
    all_prices  : list[(horizon,)]   — one array per valid combination
    params_used : list[(ma_w, p)]    — params corresponding to all_prices
    """
    all_prices:  list[np.ndarray]      = []
    params_used: list[tuple[int, int]] = []

    for ma_w in ma_windows:
        norm   = normalize(candles, window=ma_w)
        valid  = norm.dropna(subset=["ma"]).reset_index(drop=True)
        dratio = np.diff(valid["ratio"].values)

        origin_k = int(np.searchsorted(valid["begin"].values, origin_ts))
        origin_k = min(origin_k, len(valid) - 2)

        ratio0 = float(valid["ratio"].iloc[origin_k])
        ma0    = float(valid["ma"].iloc[origin_k])

        for p in p_values:
            if origin_k < p + 10:
                continue
            xi = 3 * (p + 1)
            try:
                dhat      = forecast_la1(dratio, origin_k, p, xi, horizon, tau)
                price_hat = reconstruct_price(dhat, ratio0, ma0)
                all_prices.append(price_hat)
                params_used.append((ma_w, p))
            except Exception:
                pass

    if not all_prices:
        raise ValueError("No valid forecasts produced by ensemble")

    mean_price = np.mean(all_prices, axis=0)
    return mean_price, all_prices, params_used
