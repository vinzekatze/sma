"""Trend ruler: OLS trend over a fixed window ending at an origin bar, with
residual-std bands and an optional linear extension past the origin.

Port of the former client-side computeLivePreview in sma/ui/trend_ruler.js —
same math (localWindowTrend / extensionBands). The route resolves the origin
and fetches the window and the bars after it from the full DB history
(sma/api/routes/series.py); this module only does the arithmetic.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

# Mirrors sma/ui/state.js:INTERVAL_SECONDS (calendar fallback for future dates).
INTERVAL_SECONDS = {"1m": 60, "10m": 600, "1h": 3600, "1d": 86400, "1w": 604800, "1mo": 2592000}


def _ols_window(log_prices: list[float], window: int) -> dict:
    """OLS fit of the last `window` log prices. Index 0 of the fit is the
    oldest bar in the window; fitted[i] = b + s·i."""
    n = window
    sum_x = sum_y = sum_xy = sum_xx = 0.0
    for i in range(n):
        y = log_prices[i]
        sum_x += i
        sum_y += y
        sum_xy += i * y
        sum_xx += i * i
    denom = n * sum_xx - sum_x * sum_x
    s = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - s * sum_x) / n
    fitted = [b + s * i for i in range(n)]
    ssr = sum((log_prices[i] - fitted[i]) ** 2 for i in range(n))
    return {"fitted": fitted, "std": math.sqrt(ssr / (n - 2)), "slope": s}


def _future_time(
    seq_begins: list[str], origin_date: str, h: int, interval: str,
) -> str:
    """Same rule as sma/ui/chart.js:futureDateAt — the real bar's begin when
    it exists (`seq_begins[h]`, origin is index 0), else a calendar step from
    the origin date."""
    if h < len(seq_begins):
        return seq_begins[h]
    base = datetime.strptime(origin_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    step = timedelta(seconds=INTERVAL_SECONDS.get(interval, 86400))
    return (base + step * h).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def compute_trend_ruler(
    window_candles: list[dict],
    future_candles: list[dict],
    interval: str,
    window: int,
    bands: list[float],
    need_extension: bool,
    n_future: int,
) -> dict | None:
    """`window_candles`: exactly `window` bars ascending, the last one being the
    origin. `future_candles`: bars after the origin (ascending, may be shorter
    than n_future). Returns None when there isn't enough history."""
    if window < 3 or len(window_candles) != window:
        return None
    closes = [float(c["close"]) for c in window_candles]
    log_prices = [math.log(v) for v in closes]
    fit = _ols_window(log_prices, window)

    band_list = sorted({k for k in bands if k > 0}, reverse=True)
    origin = window_candles[-1]
    origin_date = origin["begin"]
    bands_out = [
        {
            "k": k,
            "hi": [math.exp(v + k * fit["std"]) for v in fit["fitted"]],
            "lo": [math.exp(v - k * fit["std"]) for v in fit["fitted"]],
        }
        for k in band_list
    ]

    extension = None
    if need_extension and band_list:
        seq_begins = [origin_date] + [c["begin"] for c in future_candles]
        last_log = fit["fitted"][-1]
        ext_times, trend_price = [], []
        ext_bands = [{"k": k, "hi": [], "lo": []} for k in band_list]
        for h in range(n_future + 1):
            logv = last_log + h * fit["slope"]
            ext_times.append(_future_time(seq_begins, origin_date, h, interval))
            trend_price.append(math.exp(logv))
            for idx, k in enumerate(band_list):
                ext_bands[idx]["hi"].append(math.exp(logv + k * fit["std"]))
                ext_bands[idx]["lo"].append(math.exp(logv - k * fit["std"]))
        extension = {
            "n_future": n_future,
            "times": ext_times,
            "trend_price": trend_price,
            "bands": ext_bands,
        }

    return {
        "origin_date": origin_date,
        "window": window,
        "trend": {
            "times": [c["begin"] for c in window_candles],
            "price": [math.exp(v) for v in fit["fitted"]],
        },
        "bands": bands_out,
        "extension": extension,
    }
