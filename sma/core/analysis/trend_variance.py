"""
Rolling OLS trend + residual-variance analyzer — second entry in the
Анализ-tab analyzer family (see sma/core/analysis/spectrogram.py's docstring
for the "one module per analyzer, no shared registry yet" convention this
follows).

Ported from prototype_analyzers/app.py + analyzers/trend_variance.py
(2026-08-17/18 prototype session, see docs/plans/
trend_variance_analyzer_migration_plan.md for the full design rationale).
Everything here is a MEASURED characteristic of the series (trend line,
±k·std residual bands, slope/variance oscillator, an "acceleration fan"),
NOT a price forecast — the prototype's predictability sweeps
(slope_predictability_*.py, accel_predictability_sweep.py, price_forecast_*.py,
not ported) found linear trend continuation loses to price persistence, so
nothing here claims predictive value.

Origin is always the LAST available bar (migration plan §3.2 "Вариант A") —
no historical-origin picker in this first cut.

Causality: `single_window_trend`/`rolling_trend_variance` only ever look at
[t-window+1, t] for any t, so the whole module is causal by construction —
same guarantee the prototype already relied on.
"""
from __future__ import annotations

import numpy as np


def rolling_trend_variance(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """slope[t]/resid_var[t] of the causal OLS fit on window [t-window+1, t].

    O(n) via cumulative sums (no per-t polyfit) — NaN before `window` points
    of history are available. Faithful port of prototype_analyzers/
    analyzers/trend_variance.py, unchanged.
    """
    n = len(y)
    slope = np.full(n, np.nan)
    resid_var = np.full(n, np.nan)
    if window < 3 or n < window:
        return slope, resid_var

    j = np.arange(n, dtype=np.float64)
    cs_y = np.concatenate(([0.0], np.cumsum(y)))
    cs_jy = np.concatenate(([0.0], np.cumsum(j * y)))
    cs_yy = np.concatenate(([0.0], np.cumsum(y * y)))

    w = float(window)
    Sx = w * (w - 1) / 2.0
    Sxx = (w - 1) * w * (2 * w - 1) / 6.0
    denom = w * Sxx - Sx ** 2

    Sy = cs_y[window:n + 1] - cs_y[0:n - window + 1]
    Sjy = cs_jy[window:n + 1] - cs_jy[0:n - window + 1]
    Syy = cs_yy[window:n + 1] - cs_yy[0:n - window + 1]
    j0 = np.arange(0, n - window + 1, dtype=np.float64)
    Sxy = Sjy - j0 * Sy

    s = (w * Sxy - Sx * Sy) / denom
    b = (Sy - s * Sx) / w
    ssr = Syy - b * Sy - s * Sxy

    slope[window - 1:] = s
    resid_var[window - 1:] = np.maximum(ssr, 0.0) / (w - 2)

    return slope, resid_var


def single_window_trend(y: np.ndarray, origin: int, window: int) -> tuple[int, np.ndarray, float]:
    """Trend + residual std of exactly ONE window [origin-window+1, origin].

    Returns (j0, fitted, std) — j0 is the window's start index, fitted is
    the trend line's values on [j0, origin], std is residual std (ddof=2).
    """
    j0 = origin - window + 1
    if j0 < 0:
        raise ValueError(f"недостаточно истории: origin={origin}, window={window}")

    seg = y[j0:origin + 1]
    x = np.arange(window, dtype=np.float64)
    s, b = np.polyfit(x, seg, 1)
    fitted = b + s * x
    resid = seg - fitted
    std = resid.std(ddof=2)
    return j0, fitted, std


def _nan_to_none(arr: np.ndarray) -> list[float | None]:
    return [None if np.isnan(v) else float(v) for v in arr]


def _slope_window_for_accel(log_price: np.ndarray, origin: int, window: int, m_accel: int) -> np.ndarray:
    """slope values at each of the last m_accel points ending at origin,
    computed on the MINIMAL sub-array that covers them (not the full
    history) — used only when show_oscillator is False, so the
    acceleration fan doesn't force the full O(n) rolling pass just to read
    m_accel trailing slope values (see migration plan §2, правка 2026-08-18)."""
    sub_start = max(0, origin - window - m_accel + 2)
    sub_log = log_price[sub_start:origin + 1]
    slope_sub, _ = rolling_trend_variance(sub_log, window)
    return slope_sub


def compute_trend_variance(
    candles: list[dict],
    window: int = 200,
    bands: list[float] | None = None,
    show_extension: bool = False,
    n_future: int = 50,
    show_oscillator: bool = False,
    show_accel_fan: bool = False,
    m_accel: int = 50,
    n_accel: int = 50,
) -> dict:
    """
    Returns a dict:
      {"origin_date", "window",
       "trend": {"times", "price"},
       "bands": [{"k", "hi", "lo"}, ...],           # widest k first
       "extension": null or {"n_future", "trend_price", "bands": [...]},
       "oscillator": null or {"times", "slope", "var"},
       "accel_fan": null or {"price", "direction", "d0"}}

    `oscillator` is computed (and the full O(n) rolling_trend_variance pass
    paid for) ONLY when show_oscillator is True — see module docstring and
    migration plan §2/§3.4. `accel_fan`, when requested WITHOUT the
    oscillator, uses a cheap sub-array pass instead (_slope_window_for_accel).

    Raises ValueError if there isn't enough history for the requested
    window (+ m_accel, if show_accel_fan).
    """
    if window < 3:
        raise ValueError("window должен быть >= 3")

    closes = np.array([float(c["close"]) for c in candles], dtype=np.float64)
    dates = [c["begin"] for c in candles]
    n = len(closes)

    origin = n - 1
    origin_min = window - 1
    if show_accel_fan:
        origin_min = max(origin_min, window + m_accel - 2)
    if origin < origin_min:
        raise ValueError(f"Недостаточно истории ({n} баров) для выбранных настроек.")

    log_price = np.log(closes)
    j0, fitted_log, std_log = single_window_trend(log_price, origin, window)
    fitted_price = np.exp(fitted_log)
    seg_times = dates[j0:origin + 1]
    # Same OLS slope rolling_trend_variance would give at t=origin (both are
    # the causal fit of the exact same window) — reading it off the fit we
    # already computed means extension/accel-fan never need the full-history
    # rolling pass just to learn the CURRENT slope.
    local_slope = (fitted_log[-1] - fitted_log[0]) / (window - 1)

    band_list = sorted({k for k in (bands or []) if k > 0}, reverse=True)
    bands_out = []
    for k in band_list:
        bands_out.append({
            "k": k,
            "hi": np.exp(fitted_log + k * std_log).tolist(),
            "lo": np.exp(fitted_log - k * std_log).tolist(),
        })

    result: dict = {
        "origin_date": dates[origin],
        "window": window,
        "trend": {"times": seg_times, "price": fitted_price.tolist()},
        "bands": bands_out,
        "extension": None,
        "oscillator": None,
        "accel_fan": None,
    }

    if show_extension and band_list:
        h = np.arange(0, n_future + 1, dtype=np.float64)
        ext_trend_log = fitted_log[-1] + h * local_slope
        ext_bands = [
            {
                "k": k,
                "hi": np.exp(ext_trend_log + k * std_log).tolist(),
                "lo": np.exp(ext_trend_log - k * std_log).tolist(),
            }
            for k in band_list
        ]
        result["extension"] = {
            "n_future": n_future,
            "trend_price": np.exp(ext_trend_log).tolist(),
            "bands": ext_bands,
        }

    slope_series = None
    if show_oscillator:
        slope_series, var_series = rolling_trend_variance(log_price, window)
        result["oscillator"] = {
            "times": dates,
            "slope": _nan_to_none(slope_series),
            "var": _nan_to_none(var_series),
        }

    if show_accel_fan:
        if slope_series is not None:
            slope_for_accel, origin_local = slope_series, origin
        else:
            slope_for_accel = _slope_window_for_accel(log_price, origin, window, m_accel)
            origin_local = len(slope_for_accel) - 1

        _, fitted_slope_of_slope, _ = single_window_trend(slope_for_accel, origin_local, m_accel)
        d0 = (fitted_slope_of_slope[-1] - fitted_slope_of_slope[0]) / (m_accel - 1)
        slope_hyp = local_slope + n_accel * d0

        x_win = np.arange(window, dtype=np.float64)
        fan_log = fitted_log[0] + slope_hyp * x_win
        result["accel_fan"] = {
            "price": np.exp(fan_log).tolist(),
            "direction": "up" if d0 >= 0 else "down",
            "d0": float(d0),
        }

    return result
