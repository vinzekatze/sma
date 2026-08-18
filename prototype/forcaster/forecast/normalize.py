"""
Normalization of price series via slow SMA.

    ratio(t) = close(t) / SMA(close, window)(t)

The ratio is more stationary than raw price: dimensionless, oscillates
around 1.0, captures relative deviation from the slow trend.

Reconstruction: price ≈ ratio * MA(t_last)
Valid when window >> forecast horizon. Default: 1000 >> 40, factor = 25.
"""

import pandas as pd

# ~7 months of 1h bars (1000 / ~7h/day ≈ 143 trading days).
# For 10m data consider 3000-5000 to get comparable calendar coverage.
MA_WINDOW = 1000


def normalize(candles: list[dict], window: int = MA_WINDOW) -> pd.DataFrame:
    """
    Compute slow SMA and ratio = close / SMA for a candle series.

    Args:
        candles: list of OHLCV dicts (from forcaster.data.moex)
        window:  SMA window in bars

    Returns:
        DataFrame with columns: begin, open, high, low, close, ma, ratio.
        First (window-1) rows have NaN in ma/ratio — insufficient history.
    """
    df = pd.DataFrame(candles)[["begin", "open", "high", "low", "close", "volume"]]
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.sort_values("begin").reset_index(drop=True)

    df["ma"]    = df["close"].rolling(window, min_periods=window).mean()
    df["ratio"] = df["close"] / df["ma"]

    return df
