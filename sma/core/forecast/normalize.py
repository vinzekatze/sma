"""
Normalization of price series via causal logtrend (project standard, see
CLAUDE.md "Ключевые методические решения" — logtrend causal OLS, −2.4% MAPE
vs SMA(1000) on 8 tickers 1d, скр.41).

    trend(t) = exp(a + b·t)         — causal log-linear OLS fit, incremental
                                       cumulative sums, O(N), no warmup/window
    ratio(t) = close(t) / trend(t)  — dimensionless, oscillates around 1.0

Ported from prototype/forcaster/ui/app.py:_logtrend_causal /
_normalize_logtrend (2026-06) — the same normalization the reference
spectrogram implementation used. Replaces the old SMA-window normalize()
this file used to hold (see CLAUDE.md repo-structure note, now resolved):
that version is orphaned once the Hurst/MA analysis-tab features it served
are removed (2026-08-08) — nothing else in sma/ calls normalize() any more,
and whatever does (the spectrogram analyzer) needs the causal, no-window
method anyway.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _logtrend_causal(close: np.ndarray) -> np.ndarray:
    """Causal log-linear OLS trend: trend[i] fit only from close[:i+1] (no
    look-ahead), via incremental cumulative sums — no warmup, no window."""
    n = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t)
    ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc)
    cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t)
    trend[:2] = close[:2]  # first two points: OLS is underdetermined/degenerate
    return trend


def normalize(candles: list[dict]) -> pd.DataFrame:
    """
    Compute the causal logtrend and ratio = close / trend for a candle series.

    Args:
        candles: list of OHLCV dicts (from sma.core.data.moex)

    Returns:
        DataFrame with columns: begin, open, high, low, close, volume,
        trend, ratio. No NaN warm-up rows — the causal fit is defined from
        the very first bar (unlike the old SMA-window version).
    """
    df = pd.DataFrame(candles)[["begin", "open", "high", "low", "close", "volume"]]
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.sort_values("begin").reset_index(drop=True)

    close = df["close"].to_numpy(dtype=np.float64)
    df["trend"] = _logtrend_causal(close)
    df["ratio"] = df["close"] / df["trend"]

    return df
