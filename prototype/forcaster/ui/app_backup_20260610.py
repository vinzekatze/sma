"""
Streamlit web interface — MOEX Forecaster.
Run from project root:  streamlit run forcaster/ui/app.py
"""

from __future__ import annotations

import sys
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# ensure project root is on sys.path regardless of how streamlit is launched
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forcaster.data.moex import download_candles, save_candles, INTERVALS
from forcaster.forecast.normalize import normalize

# ── per-interval settings ─────────────────────────────────────────────────────

_MA_DEFAULTS: dict[str, int] = {
    "1m": 500, "10m": 300, "1h": 200, "1d": 200, "1w": 52, "1mo": 24,
}
_MA_SLIDER: dict[str, tuple[int, int, int]] = {
    "1m":  (2, 2000, 1), "10m": (2, 2000, 1),
    "1h":  (2, 2000, 1), "1d":  (2, 2000, 1),
    "1w":  (2, 2000, 1), "1mo": (2, 2000, 1),
}
_BAR_DELTA: dict[str, pd.Timedelta] = {
    "1m":  pd.Timedelta(minutes=1),
    "10m": pd.Timedelta(minutes=10),
    "1h":  pd.Timedelta(hours=1),
    "1d":  pd.Timedelta(days=1),
    "1w":  pd.Timedelta(weeks=1),
    "1mo": pd.Timedelta(days=30),
}
_MA_AUTO_PARAMS: dict[str, dict] = {
    "1m":  dict(ma_min=1000, ma_max=10000, hurst_window=500, hurst_step=100),
    "10m": dict(ma_min=500,  ma_max=5000,  hurst_window=300, hurst_step=60),
    "1h":  dict(ma_min=200,  ma_max=5000,  hurst_window=200, hurst_step=40),
    "1d":  dict(ma_min=60,   ma_max=500,   hurst_window=50,  hurst_step=10),
    "1w":  dict(ma_min=20,   ma_max=200,   hurst_window=20,  hurst_step=4),
    "1mo": dict(ma_min=8,    ma_max=60,    hurst_window=10,  hurst_step=2),
}
from forcaster.forecast.delay_params import compute_acf, first_acf_zero, recommended_p
from forcaster.forecast.la import forecast_la1, reconstruct_price
from forcaster.forecast.lwr import forecast_lwr
from forcaster.forecast.ensemble import forecast_ensemble, range_vals
from forcaster.forecast.hurst import rolling_hurst, unpredictable_zones
from forcaster.forecast.adaptive import (find_optimal_ma,
                                          hurst_percentile_threshold,
                                          regime_mask_for_pool)
from forcaster.forecast.phase import pool_phase_labels, query_phase_label, PHASE_NAMES
from forcaster.forecast.pe import rolling_pe, pe_mask_for_pool, pe_adaptive_mask, pe_proximity_mask
from forcaster.forecast.filterbank import (forecast_fb, forecast_lwr_fb, make_filter_bank,
                                            forecast_fb_damped, FB_P, FB_XI)

_ROOT    = Path(__file__).parent.parent.parent
DATA_DIR = _ROOT / "data" / "candles"

# ── cached helpers ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _normalize(candles_key: str, ma_window: int, _v: int = 2) -> pd.DataFrame:
    candles = json.loads(candles_key)
    return normalize(candles, window=ma_window)


def _logtrend_causal(close: np.ndarray) -> np.ndarray:
    """Causal log-linear OLS trend: trend[t] = exp(a + b*t) fitted on close[:t+1]."""
    n   = len(close)
    t   = np.arange(n, dtype=np.float64)
    lc  = np.log(np.maximum(close, 1e-10))
    cn  = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t)
    ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc)
    cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a     = (cy - b * ct) / cn
    trend = np.exp(a + b * t)
    trend[:2] = close[:2]   # первые 2 точки: OLS не определён, берём close напрямую
    return trend


@st.cache_data(show_spinner=False)
def _normalize_logtrend(candles_key: str) -> pd.DataFrame:
    """Normalization by causal log-linear trend (no warmup, no window parameter)."""
    candles = json.loads(candles_key)
    df      = pd.DataFrame(candles)
    df["begin"] = pd.to_datetime(df["begin"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    close       = df["close"].values.astype(np.float64)
    trend       = _logtrend_causal(close)
    df["ma"]    = trend
    df["ratio"] = close / trend
    return df


@st.cache_data(show_spinner=False)
def _acf(ratio_bytes: bytes, max_lag: int) -> tuple[np.ndarray, int]:
    ratio = np.frombuffer(ratio_bytes, dtype=np.float64).copy()
    acf   = compute_acf(ratio, max_lag=max_lag)
    return acf, first_acf_zero(acf)


@st.cache_data(show_spinner=False)
def _forecast(dratio_bytes: bytes, origin_k: int,
              p: int, xi: int, horizon: int,
              regime_bytes: bytes | None = None,
              norm_vecs: bool = False,
              use_huber: bool = False,
              pca_k: int = 0) -> np.ndarray:
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    mask   = np.frombuffer(regime_bytes, dtype=bool).copy() if regime_bytes else None
    return forecast_la1(dratio, origin_k, p, xi, horizon, regime_mask=mask,
                        norm_vecs=norm_vecs, use_huber=use_huber, pca_k=pca_k)


@st.cache_data(show_spinner=False)
def _forecast_lwr(dratio_bytes: bytes, origin_k: int,
                  p: int, xi: int, horizon: int,
                  regime_bytes: bytes | None = None,
                  norm_vecs: bool = False,
                  use_huber: bool = False,
                  pca_k: int = 0) -> np.ndarray:
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    mask   = np.frombuffer(regime_bytes, dtype=bool).copy() if regime_bytes else None
    return forecast_lwr(dratio, origin_k, p, xi, horizon, regime_mask=mask,
                        norm_vecs=norm_vecs, use_huber=use_huber, pca_k=pca_k)


@st.cache_data(show_spinner=False)
def _forecast_fb(series_bytes: bytes, origin_k: int, horizon: int,
                 p: int = FB_P, xi: int = FB_XI,
                 use_lwr: bool = True,
                 component_idx: tuple[int, ...] = (3, 4, 5),
                 p_per_component: tuple[tuple[int, int], ...] = ()) -> np.ndarray:
    series  = np.frombuffer(series_bytes, dtype=np.float64).copy()
    ppc_dict = dict(p_per_component) if p_per_component else None
    return forecast_fb(series, origin_k, horizon, p=p, n_neighbors=xi,
                       use_lwr=use_lwr, component_idx=list(component_idx),
                       p_per_component=ppc_dict)


@st.cache_data(show_spinner=False)
def _forecast_fb_autop(
    dratio_bytes: bytes,
    origin_k: int,
    horizon: int,
    val_h: int,
    p_slow: int,
    p_fast_min: int,
    p_fast_max: int,
    p_fast_step: int,
    active_components: tuple[int, ...] = (1, 2, 3, 4, 5),
    top_n_noise: int = 1,
) -> tuple[np.ndarray, dict]:
    """
    Filter bank + фиксированный p для медленных C3-C5,
    авто-p (перебор + валидация на компоненте) для шумовых C0-C2.

    Для шумовых: берётся top_n_noise лучших p по val_mape, прогнозы усредняются.

    per_comp_info[ci] = {
        "slow": True/False,
        "top":  [{"p": int, "xi": int, "val_mape": float|None}, ...],  # отсортировано
    }
    """
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    components = make_filter_bank(dratio[:origin_k])  # (6, origin_k)

    per_comp: dict[int, dict] = {}
    hats: list[tuple[int, np.ndarray]] = []

    for ci in range(6):
        comp_hist = components[ci]

        if ci >= 3:
            # ── Медленные C3-C5: фиксированный p ─────────────────────────────
            p_ci  = p_slow
            xi_ci = max(63, 3 * (p_ci + 1))
            dhat  = (forecast_lwr(comp_hist, len(comp_hist), p_ci, xi_ci, horizon)
                     if len(comp_hist) >= xi_ci + p_ci + 2 else np.zeros(horizon))
            per_comp[ci] = {
                "slow": True,
                "top":  [{"p": p_ci, "xi": xi_ci, "val_mape": None}],
            }
        else:
            # ── Шумовые C0-C2: перебор p, сбор кандидатов ───────────────────
            val_origin      = origin_k - val_h
            comp_val_actual = comp_hist[val_origin:origin_k]

            candidates: list[tuple[float, int, int]] = []   # (mape, p, xi)

            for p_ci in range(p_fast_min, p_fast_max + 1, p_fast_step):
                xi_ci = max(3 * (p_ci + 1), p_ci + 2)
                if val_origin < xi_ci + p_ci + 2:
                    continue
                try:
                    dhat_val = forecast_lwr(
                        comp_hist[:val_origin], val_origin, p_ci, xi_ci, val_h,
                    )
                    mape = float(np.mean(
                        np.abs(dhat_val - comp_val_actual)
                        / (np.abs(comp_val_actual) + 1e-12)
                    ))
                    candidates.append((mape, p_ci, xi_ci))
                except Exception:
                    continue

            candidates.sort(key=lambda x: x[0])
            top_cands = candidates[:max(1, top_n_noise)]

            # Усредняем прогнозы top_n лучших p
            sub_dhats = []
            for _, p_ci, xi_ci in top_cands:
                if len(comp_hist) >= xi_ci + p_ci + 2:
                    try:
                        sub_dhats.append(
                            forecast_lwr(comp_hist, len(comp_hist), p_ci, xi_ci, horizon)
                        )
                    except Exception:
                        pass

            dhat = np.mean(sub_dhats, axis=0) if sub_dhats else np.zeros(horizon)
            per_comp[ci] = {
                "slow": False,
                "top":  [{"p": p_ci, "xi": xi_ci, "val_mape": mape}
                          for mape, p_ci, xi_ci in top_cands],
            }

        hats.append((ci, dhat))

    active_set = set(active_components)
    total_dhat = (
        np.sum([d for ci, d in hats if ci in active_set], axis=0)
        if any(ci in active_set for ci, _ in hats) else np.zeros(horizon)
    )
    return total_dhat, per_comp


@st.cache_data(show_spinner=False)
def _forecast_fb_merged_noise(
    dratio_bytes: bytes,
    origin_k: int,
    horizon: int,
    p: int,
    xi: int,
    fast_components: tuple[int, ...],   # из {0,1,2} — складываются в один сигнал
    slow_components: tuple[int, ...],   # из {3,4,5} — каждый отдельно
    p_per_slow: tuple[tuple[int, int], ...] = (),
) -> np.ndarray:
    """
    Filter bank: шумовые компоненты (fast_components ⊆ {0,1,2}) суммируются
    в один сигнал и прогнозируются единым LWR.
    Медленные (slow_components ⊆ {3,4,5}) прогнозируются по отдельности.
    Возвращает dratio_hat[horizon].
    """
    dratio     = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    components = make_filter_bank(dratio[:origin_k])  # (6, origin_k)
    hats: list[np.ndarray] = []

    # ── Объединённый шум ──────────────────────────────────────────────────────
    if fast_components:
        merged = np.sum([components[ci] for ci in fast_components], axis=0)
        xi_eff = max(xi, 3 * (p + 1))
        if len(merged) >= xi_eff + p + 2:
            hats.append(forecast_lwr(merged, len(merged), p, xi_eff, horizon))

    # ── Медленные по отдельности ──────────────────────────────────────────────
    ppc = dict(p_per_slow)
    for ci in slow_components:
        p_ci  = ppc.get(ci, p)
        xi_ci = max(3 * (p_ci + 1), xi) if ci not in ppc else 3 * (p_ci + 1)
        comp  = components[ci]
        if len(comp) >= xi_ci + p_ci + 2:
            hats.append(forecast_lwr(comp, len(comp), p_ci, xi_ci, horizon))

    return np.sum(hats, axis=0) if hats else np.zeros(horizon)


@st.cache_data(show_spinner=False)
def _forecast_fb_damped_cached(
    dratio_bytes: bytes,
    origin_k: int,
    horizon: int,
    p_slow: int = FB_P,
    xi_slow: int = FB_XI,
    gamma_c1: float = 0.5,
    gamma_c2: float = 0.8,
    slow_components: tuple[int, ...] = (3, 4, 5),
) -> np.ndarray:
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    return forecast_fb_damped(dratio, origin_k, horizon,
                              p_slow=p_slow, xi_slow=xi_slow,
                              gamma_c1=gamma_c1, gamma_c2=gamma_c2,
                              slow_components=list(slow_components))


@st.cache_data(show_spinner=False)
def _fb_slow_trend(
    dratio_bytes: bytes,
    ratio_bytes: bytes,
    ma_bytes: bytes,
    origin_k: int,
    n_display: int = 200,
    component_idx: tuple[int, ...] = (3, 4, 5),
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes slow-trend history for filter bank display.
    Returns (bar_indices, slow_prices, junction_price).
    Anchors the slow ratio to the actual ratio at the start of the display window,
    then integrates slow_dratio forward — so the curve tracks the actual price
    but with fast noise removed.
    """
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    ratio  = np.frombuffer(ratio_bytes,  dtype=np.float64).copy()
    ma     = np.frombuffer(ma_bytes,     dtype=np.float64).copy()

    comp = make_filter_bank(dratio[:origin_k])      # (6, origin_k), causal
    slow_dratio = np.zeros(origin_k)
    for ci in component_idx:
        if 0 <= ci < comp.shape[0]:
            slow_dratio += comp[ci]

    anchor_idx = max(0, origin_k - n_display)
    slow_dratio_seg = slow_dratio[anchor_idx:origin_k]
    slow_ratio_0    = float(ratio[anchor_idx])

    if len(slow_dratio_seg) > 0:
        slow_ratio_seg = slow_ratio_0 + np.concatenate([[0.0], np.cumsum(slow_dratio_seg)])
    else:
        slow_ratio_seg = np.array([slow_ratio_0])

    n = min(len(slow_ratio_seg), len(ma) - anchor_idx)
    slow_prices  = slow_ratio_seg[:n] * ma[anchor_idx: anchor_idx + n]
    bar_indices  = np.arange(anchor_idx, anchor_idx + n, dtype=int)
    junction_price = float(slow_prices[-1])

    return bar_indices, slow_prices, junction_price


def _ma_trend(ma_values: np.ndarray, origin_k: int, horizon: int, fit_bars: int) -> np.ndarray:
    """
    Fit linear trend to the last `fit_bars` MA values before origin_k,
    extrapolate `horizon` steps forward.
    Returns array of shape (horizon,) — MA estimates for bars origin_k+1 … origin_k+horizon.
    """
    n   = min(fit_bars, origin_k)
    seg = ma_values[origin_k - n : origin_k]
    x   = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, seg, 1)
    future_x = np.arange(n, n + horizon, dtype=float)
    extrap = intercept + slope * future_x
    # не уходим в отрицательные значения
    return np.maximum(extrap, seg[-1] * 0.01)


@st.cache_data(show_spinner=False)
def _find_optimal_ma(
    candles_key: str,
    ma_min: int = 200, ma_max: int = 5000,
    hurst_window: int = 200, hurst_step: int = 40,
) -> tuple[int, dict]:
    candles = json.loads(candles_key)
    return find_optimal_ma(candles, ma_min=ma_min, ma_max=ma_max,
                           hurst_window=hurst_window, hurst_step=hurst_step)


@st.cache_data(show_spinner=False)
def _hurst_zones(dratio_bytes: bytes,
                 window: int = 200, step: int = 20,
                 ) -> tuple[np.ndarray, np.ndarray]:
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    return rolling_hurst(dratio, window=window, step=step)


@st.cache_data(show_spinner=False)
def _rolling_pe(dratio_bytes: bytes, win: int = 50, order: int = 3) -> np.ndarray:
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    return rolling_pe(dratio, win=win, order=order)


@st.cache_data(show_spinner=False)
def _auto_p_forecast(
    dratio_bytes: bytes,
    ratio_bytes: bytes,
    origin_k: int,
    val_horizon: int,
    p_max: int,
    horizon: int,
    norm_vecs: bool,
    use_huber: bool,
    use_lwr: bool = False,
    top_n: int = 10,
    series_is_ratio: bool = False,
    use_pca: bool = True,
    bar_hurst_bytes: bytes = b"",
    hurst_thr: float = 1.0,
    use_hurst_filter: bool = False,
    bar_pe_bytes: bytes = b"",
    pe_thr: float = 1.0,
    use_pe_filter: bool = False,
    pe_mode: str = "threshold",
    pe_delta_base: float = 0.005,
    pe_delta_max: float = 0.05,
    use_filterbank: bool = False,
    fb_p: int = FB_P,
    fb_xi: int = FB_XI,
    fb_use_lwr: bool = True,
    fb_components: tuple[int, ...] = (3, 4, 5),
) -> tuple[list[dict], dict]:
    """
    Retrospective p-selection via walk-forward validation (LA1 or LWR).

    series_is_ratio=True: LA applied directly to ratio (no Δ).
      - history = ratio[:val_origin+1], forecast returns ratio_hat directly
      - val_mape = |ratio_hat - actual_ratio| / actual_ratio (no cumsum)

    For each (p, pca_k): forecast from (origin_k - val_horizon), compute MAPE.
    Returns top-N results (sorted by MAPE asc) and full scores dict.
    Each result dict: {p, pca_k, mape, dhat: list[float]}.
    """
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    ratio  = np.frombuffer(ratio_bytes,  dtype=np.float64).copy()

    val_origin   = origin_k - val_horizon
    ratio0_val   = float(ratio[val_origin])
    actual_ratio = ratio[val_origin + 1 : val_origin + 1 + val_horizon]
    total_h      = val_horizon + horizon

    # ratio mode: pass ratio series and shift origin by +1 so history includes ratio[val_origin]
    series     = ratio  if series_is_ratio else dratio
    origin_arg = val_origin + 1 if series_is_ratio else val_origin

    tasks = [
        (p, k)
        for p in range(2, p_max + 1)
        if val_origin >= p + 5
        for k in (range(2, p) if use_pca else [0])
    ]
    if not tasks:
        return [], {}

    _bar_hurst = np.frombuffer(bar_hurst_bytes, dtype=np.float64).copy() if bar_hurst_bytes else None
    _bar_pe    = np.frombuffer(bar_pe_bytes,    dtype=np.float64).copy() if bar_pe_bytes    else None

    # PE origin для proximity-режима: бар перед val_origin
    _pe_origin_val: float | None = None
    if use_pe_filter and _bar_pe is not None and pe_mode == "proximity":
        _idx = origin_arg - 1
        if 0 <= _idx < len(_bar_pe) and not np.isnan(_bar_pe[_idx]):
            _pe_origin_val = float(_bar_pe[_idx])

    def _make_mask(p: int, pool_size: int) -> np.ndarray | None:
        if pool_size <= 0:
            return None
        if not use_hurst_filter and not use_pe_filter:
            return None
        mask = np.ones(pool_size, dtype=bool)
        centers = np.minimum(np.arange(pool_size) + p // 2, pool_size - 1)
        if use_hurst_filter and _bar_hurst is not None:
            c = np.minimum(centers, len(_bar_hurst) - 1)
            mask &= ~np.isnan(_bar_hurst[c]) & (_bar_hurst[c] < hurst_thr)
        if use_pe_filter and _bar_pe is not None:
            if pe_mode == "proximity" and _pe_origin_val is not None:
                c = np.minimum(centers, len(_bar_pe) - 1)
                valid_c = ~np.isnan(_bar_pe[c])
                mask &= valid_c & (np.abs(_bar_pe[c] - _pe_origin_val) <= pe_delta_max)
            else:
                c = np.minimum(centers, len(_bar_pe) - 1)
                mask &= ~np.isnan(_bar_pe[c]) & (_bar_pe[c] < pe_thr)
        return mask

    fn = forecast_lwr if use_lwr else forecast_la1
    scores: dict[tuple[int, int], float] = {}
    all_results: list[tuple[float, int, int, np.ndarray]] = []

    for p, pca_k in tasks:
        xi = 3 * (p + 1)
        pool_size = max(0, origin_arg - p)
        pm = _make_mask(p, pool_size)
        try:
            dhat = fn(series, origin_arg, p, xi, total_h,
                      norm_vecs=norm_vecs, use_huber=use_huber, pca_k=pca_k,
                      regime_mask=pm)
        except Exception:
            continue
        if series_is_ratio:
            ratio_hat = dhat[:val_horizon]
        else:
            ratio_hat = ratio0_val + np.cumsum(dhat[:val_horizon])
        n = min(len(ratio_hat), len(actual_ratio))
        if n == 0:
            continue
        mape = float(np.mean(
            np.abs(ratio_hat[:n] - actual_ratio[:n]) / (np.abs(actual_ratio[:n]) + 1e-12)
        ))
        scores[(p, pca_k)] = mape
        all_results.append((mape, p, pca_k, dhat))

    # ── filter bank candidate ──────────────────────────────────────────────────
    if use_filterbank:
        try:
            # Use ratio series directly if series_is_ratio, otherwise dratio
            fb_series    = ratio  if series_is_ratio else dratio
            fb_origin    = val_origin + 1 if series_is_ratio else origin_arg
            fb_dhat = forecast_fb(fb_series, fb_origin, total_h,
                                  p=fb_p, n_neighbors=fb_xi, use_lwr=fb_use_lwr,
                                  component_idx=list(fb_components))
            if series_is_ratio:
                fb_r_hat = fb_dhat[:val_horizon]       # already ratio values
            else:
                fb_r_hat = ratio0_val + np.cumsum(fb_dhat[:val_horizon])
            n_fb = min(len(fb_r_hat), len(actual_ratio))
            if n_fb > 0:
                fb_mape = float(np.mean(
                    np.abs(fb_r_hat[:n_fb] - actual_ratio[:n_fb])
                    / (np.abs(actual_ratio[:n_fb]) + 1e-12)
                ))
                scores[(fb_p, -1)] = fb_mape
                all_results.append((fb_mape, fb_p, -1, fb_dhat))
        except Exception:
            pass

    if not all_results:
        return [], {}

    all_results.sort(key=lambda x: x[0])
    top_cands = [
        {"p": p, "pca_k": k, "mape": m, "dhat": d.tolist()}
        for m, p, k, d in all_results[:top_n]
    ]
    return top_cands, scores


@st.cache_data(show_spinner=False)
def _forecast_ensemble(
    candles_key: str,
    origin_ts_str: str,
    ma_windows: tuple[int, ...],
    p_values: tuple[int, ...],
    horizon: int,
) -> tuple[list[float], list[list[float]], list[tuple[int, int]]]:
    candles = json.loads(candles_key)
    mean_p, all_p, params = forecast_ensemble(
        candles, pd.Timestamp(origin_ts_str),
        list(ma_windows), list(p_values), horizon,
    )
    return mean_p.tolist(), [f.tolist() for f in all_p], params


# ── page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="MOEX Forecaster", layout="wide")
st.title("MOEX Forecaster")

# read current interval before sidebar renders (selectbox is in main area, but its
# key is in session_state from the previous run — default "1d" on first load)
_cur_interval = st.session_state.get("interval_val", "1d")
_cur_ticker   = st.session_state.get("ticker_val", "SBER").upper().strip()
_ma_key       = f"ma_window_{_cur_ticker}_{_cur_interval}"   # per-ticker + per-interval
_ma_min, _ma_max, _ma_step = _MA_SLIDER.get(_cur_interval, (100, 5000, 50))
if _ma_key not in st.session_state:
    st.session_state[_ma_key] = _MA_DEFAULTS.get(_cur_interval, 1000)

# ── sidebar: parameters ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("Параметры")

    # вычисляем до рендера слайдера, иначе Streamlit запрещает изменять session_state
    if st.session_state.get("_auto_ma_pending") and "candles_key_for_ma" in st.session_state:
        # проверяем, что сохранённые свечи принадлежат текущему тикеру
        if st.session_state.get("candles_ticker_for_ma", "") == _cur_ticker:
            _ap = _MA_AUTO_PARAMS.get(_cur_interval, _MA_AUTO_PARAMS["1h"])
            with st.spinner("Подбираю MA…"):
                best, scores = _find_optimal_ma(
                    st.session_state["candles_key_for_ma"],
                    ma_min=_ap["ma_min"], ma_max=_ap["ma_max"],
                    hurst_window=_ap["hurst_window"], hurst_step=_ap["hurst_step"],
                )
            st.session_state[_ma_key] = int(max(_ma_min, min(_ma_max, best)))
            st.session_state[f"_auto_ma_scores_{_cur_ticker}_{_cur_interval}"] = scores
        st.session_state.pop("_auto_ma_pending", None)

    norm_mode = st.radio(
        "Нормализация",
        ["SMA", "logtrend"],
        index=1,
        horizontal=True,
        key="norm_mode",
        help="SMA: ratio = close / SMA(window) — классика, теряет window−1 баров на разгрев.\n"
             "logtrend: ratio = close / exp(OLS(log(close))) — causal, без разгрева, "
             "использует всю историю. Даёт −2.4% MAPE на 8 тикерах (скр.41).",
    )
    _use_logtrend = (norm_mode == "logtrend")

    ma_window = st.slider(
        "Окно скользящей средней (MA)",
        min_value=_ma_min, max_value=_ma_max, step=_ma_step,
        key=_ma_key,
        disabled=_use_logtrend,
        help="Медленный тренд для нормализации. Чем меньше — тем быстрее MA "
             "реагирует на движение цены, но тем больше паттернов может смыть.",
    )

    if st.button("🔍 Подобрать MA автоматически", use_container_width=True,
                 disabled=_use_logtrend,
                 help="Перебирает ~25 значений MA и выбирает то, при котором "
                      "Δratio наиболее возвратен (минимальный средний Hurst)"):
        st.session_state["_auto_ma_pending"] = True
        st.rerun()   # следующий запуск вычислит MA до рендера слайдера

    _scores_key = f"_auto_ma_scores_{_cur_ticker}_{_cur_interval}"
    if _scores_key in st.session_state:
        best   = min(st.session_state[_scores_key], key=st.session_state[_scores_key].get)
        h_best = st.session_state[_scores_key][best]
        st.caption(f"Авто: MA={best}  (Hurst={h_best:.3f})")

    st.divider()
    st.subheader("LA-прогноз")

    p = st.slider("Размерность вложения p", 2, 100, 70,
                  help="Длина вектора задержек. Надёжно при N > 10^p.")

    xi_min = 3 * (p + 1)
    xi = st.slider("Соседей Ξ", xi_min, max(xi_min * 4, 300), xi_min,
                   help=f"Правило лекции: Ξ ≥ 3(p+1) = {xi_min}")

    horizon = st.slider("Горизонт прогноза (шагов)", 5, 100, 15)

    norm_vecs = st.checkbox(
        "Нормировать форму вектора (w/σ)",
        help="Делит каждый вектор задержек на своё σ перед поиском соседей. "
             "Паттерны с одинаковой формой, но разным масштабом (тихий/шумный "
             "рынок) становятся соседями — уменьшает число ложных соседей.",
    )
    use_huber = st.checkbox(
        "Huber-регрессия (робастная подгонка)",
        help="Заменяет OLS на IRLS с Huber-штрафом. Соседи с аномальным y "
             "(гэп, новость в один бар) получают сниженный вес — не влияют "
             "на подгонку так сильно, как при обычном МНК.",
    )

    _pca_available = p > 3   # нужно хотя бы 2 компоненты при max = p-1 ≥ 3
    use_pca = st.checkbox(
        "PCA-метрика",
        disabled=not _pca_available,
        help="Проецирует пул задержек в k главных компонент перед поиском соседей. "
             "L2 в PCA-пространстве = расстояние Маханалобиса в исходном — убирает "
             "коллинеарность строк, улучшает качество отбора."
             + ("" if _pca_available else " (недоступно при p ≤ 3)"),
    ) if _pca_available else False
    pca_k = st.slider(
        "k компонент PCA", 2, p - 1, max(2, p // 2),
        help="Количество главных компонент. Рекомендуется p//2. "
             "PCA вычисляется один раз до итеративного цикла.",
        disabled=not use_pca,
    ) if (use_pca and _pca_available) else 0

    st.divider()
    st.subheader("Реконструкция цены")
    use_ma_trend = st.checkbox(
        "Линейный тренд MA",
        value=True,
        help="Аппроксимирует последние N баров MA прямой и экстраполирует её "
             "вперёд. Убирает систематическую ошибку при малом окне MA, когда "
             "тренд заметно смещается за горизонт прогноза.",
    )
    ma_fit_bars = st.slider(
        "Баров для подгонки тренда", 5, 200, 10,
        help="Сколько последних баров MA используется для линейной регрессии.",
        disabled=not use_ma_trend,
    )

    use_ratio_series = st.checkbox(
        "Ряд: ratio (без Δ)",
        help="LA применяется напрямую к ratio = close/MA вместо Δratio. "
             "FNN-анализ показывает, что ratio имеет аттрактор при p≈4, "
             "тогда как Δratio не сходится. Реконструкция: price = ratio_hat × MA.",
    )

    st.caption(f"τ = 1 (фиксировано)  |  ряд: {'ratio' if use_ratio_series else 'Δratio'}")
    st.caption(f"Минимум соседей для LA1: {xi_min}")

    st.divider()
    st.subheader("Фильтр соседей")

    st.markdown("**Режим Хёрста**")
    show_hurst  = st.checkbox("Показать зоны на графике", value=True)
    use_regime  = st.checkbox("Фильтр по Hurst",
                              help="Искать соседей только в предсказуемых зонах (H < порога). "
                                   "Fallback на полный пул если предсказуемых < Ξ.")
    hurst_pct   = st.slider(
        "Процентиль H для порога", 10, 60, 20, 5,
        help="Порог = N-й процентиль rolling Hurst. P30 ≈ нижние 30% (наиболее возвратные).",
        disabled=not (show_hurst or use_regime),
    )

    st.markdown("**Фазовый фильтр**")
    use_phase = st.checkbox(
        "Фильтр по фазе ratio",
        help="Искать соседей только с той же позицией ratio (ниже/нейтрал/выше MA) "
             "и тем же направлением движения. 6 фаз. Fallback на полный пул если < Ξ.",
    )
    phase_margin_pct = st.slider(
        "Граница фаз (±% от MA)", 0.5, 5.0, 2.0, 0.5,
        help="dn = 1 − margin, up = 1 + margin. Нейтральная зона — между ними.",
        disabled=not use_phase,
    )
    phase_dn = 1.0 - phase_margin_pct / 100.0
    phase_up = 1.0 + phase_margin_pct / 100.0

    st.markdown("**PE-фильтр**")
    use_pe = st.checkbox(
        "Фильтр по Permutation Entropy",
        help="Фильтрует пул соседей по значению PE. Два режима: порог (структурные "
             "участки) или близость к PE точки отсчёта.",
    )
    pe_win = st.slider(
        "Окно PE (баров)", 20, 200, 50, 5,
        help="Размер скользящего окна для вычисления Permutation Entropy.",
        disabled=not use_pe,
    )
    pe_mode = st.radio(
        "Режим PE-фильтра",
        ["threshold", "proximity"],
        format_func=lambda x: "Порог (структурные)" if x == "threshold" else "Близость к origin",
        horizontal=True,
        disabled=not use_pe,
    ) if use_pe else "threshold"

    if pe_mode == "threshold":
        pe_pct = st.slider(
            "Начальный процентиль PE", 10, 40, 25, 5,
            help="Старт адаптации: N-й процентиль rolling PE. P25 ≈ нижние 25%.",
            disabled=not use_pe,
        )
        pe_max_pct = st.slider(
            "Макс. процентиль PE (до fallback)", pe_pct + 5, 95,
            min(75, max(pe_pct + 5, 75)), 5,
            help="Если соседей < Ξ при этом процентиле — полный fallback.",
            disabled=not use_pe,
        )
        pe_delta_base = pe_delta_max = 0.0  # не используются
    else:
        pe_delta_base = st.slider(
            "Начальная ΔPE", 0.001, 0.05, 0.005, 0.001,
            format="%.3f",
            help="Начальная ширина окна вокруг PE_origin. Адаптивно расширяется.",
            disabled=not use_pe,
        )
        pe_delta_max = st.slider(
            "Макс. ΔPE (до fallback)", pe_delta_base, 0.2,
            min(0.05, max(pe_delta_base, 0.05)), 0.005,
            format="%.3f",
            help="При превышении — полный fallback на весь пул.",
            disabled=not use_pe,
        )
        pe_pct = pe_max_pct = 0  # не используются

    st.caption("Все фильтры комбинируются AND. Fallback на полный пул при нехватке соседей.")

    st.divider()
    st.subheader("Filter bank")
    st.caption(
        "LWR на медленных компонентах Δratio (C3+C4+C5, периоды >52 баров). "
        "Используется при выборе модели «LWR + filter bank» или включении в авто-p."
    )
    _COMP_LABELS = {
        0: "C0 (~2-4 бара)",
        1: "C1 (~4-8 баров)",
        2: "C2 (~8-16 баров)",
        3: "C3 (~16-52 бара)",
        4: "C4 (~52-103 бара)",
        5: "C5 (~103+ баров)",
    }
    # p_min по правилу T_min/4 для каждой компоненты
    _COMP_P_DEFAULTS = {0: 3, 1: 5, 2: 8, 3: FB_P, 4: 26, 5: 30}

    fb_p = st.slider(
        "p по умолчанию (filter bank)", 2, 300, FB_P, 1,
        key="fb_p",
        help="Правило для медленных компонент: p ≥ T_min/4 ≈ 13 (C3 период ≈52 бара). "
             "Производственный оптимум: p=20. "
             "Используется для компонент без индивидуального p.",
    )
    fb_xi = 3 * (fb_p + 1)
    fb_components_raw = st.multiselect(
        "Компоненты filter bank",
        options=[0, 1, 2, 3, 4, 5],
        default=[3, 4, 5],
        format_func=lambda x: _COMP_LABELS[x],
        key="fb_components",
        help="C3+C4+C5 (slow) — исследованный оптимум (−22%, p<0.0001). "
             "C0+C1+C2 — быстрый шум (ожидаемо хуже). "
             "Крайности для экспериментов.",
    )
    fb_components = tuple(sorted(fb_components_raw)) if fb_components_raw else (3, 4, 5)

    # Индивидуальный p для каждой выбранной компоненты
    fb_p_per_comp: dict[int, int] = {}
    with st.expander("Индивидуальный p для компонент"):
        st.caption(
            "Переопределяет «p по умолчанию» для конкретных компонент. "
            "ξ для каждой = 3×(p+1) автоматически."
        )
        for _ci in fb_components:
            _p_default = _COMP_P_DEFAULTS.get(_ci, fb_p)
            fb_p_per_comp[_ci] = st.slider(
                _COMP_LABELS[_ci], 2, 300, _p_default, 1,
                key=f"fb_p_c{_ci}",
            )
    fb_ppc_tuple = tuple(sorted(fb_p_per_comp.items()))  # hashable для кэша

    use_d2ratio = st.checkbox(
        "d²ratio нормализация",
        value=False,
        key="fb_use_d2ratio",
        help=(
            "Подаёт второе разностное (d²ratio = diff(diff(ratio))) в filter bank "
            "вместо dratio. Реконструкция: ratio[origin] + cumsum(d²hat) — однократный cumsum. "
            "Результат исслед. 38: −14% поверх dratio+FB, −33% суммарно от raw dratio."
        ),
    )
    fb_merge_noise = st.checkbox(
        "Объединить шумовые C0–C2 в один сигнал",
        value=False,
        key="fb_merge_noise",
        help=(
            "Компоненты C0–C2 (из выбранных) суммируются в один широкополосный "
            "сигнал и прогнозируются одним LWR. Медленные C3–C5 остаются раздельными. "
            "Работает только в dratio-режиме."
        ),
    )

    fb_damped_ar = st.checkbox(
        "Damped AR для C1+C2",
        value=False,
        key="fb_damped_ar",
        help=(
            "C1 (8–16б) и C2 (16–32б) прогнозируются через AR(BIC) × γ^h. "
            "C3–C5 — стандартный LWR. C0 — ноль. "
            "Исслед. 51–52 (8 тикеров, 400 origins): γ_C1=0.5, γ_C2=0.8 → −8.01% MAPE."
        ),
    )
    if fb_damped_ar:
        _fdg1, _fdg2 = st.columns(2)
        fb_gamma_c1 = _fdg1.slider(
            "γ C1", 0.10, 1.0, 0.50, 0.05, key="fb_gamma_c1",
            help="Decay для C1 (~8–16б). 0.5 = оптимум (скр.52).",
        )
        fb_gamma_c2 = _fdg2.slider(
            "γ C2", 0.10, 1.0, 0.80, 0.05, key="fb_gamma_c2",
            help="Decay для C2 (~16–32б). 0.8 = оптимум (скр.52).",
        )
    else:
        fb_gamma_c1 = 0.5
        fb_gamma_c2 = 0.8

    # Информационная строка
    _xi_info = "  |  ".join(
        f"C{ci}: p={fb_p_per_comp[ci]}, ξ={3*(fb_p_per_comp[ci]+1)}"
        for ci in fb_components
    )
    _d2_label = "d²ratio" if use_d2ratio else "dratio"
    st.caption(f"sep-режим  |  {_xi_info}  |  вход: {_d2_label}")

    st.divider()
    st.subheader("Авто-p по компонентам")
    st.caption(
        "C3–C5 (медленные): один фиксированный p + LWR. "
        "C0–C2 (шум): перебор p, валидация на компоненте отдельно."
    )
    fbauto_p_slow = st.slider(
        "p для C3–C5", 2, 100, FB_P, 1, key="fbauto_p_slow",
        help="Фиксированный p для медленных компонент. ξ = max(63, 3·(p+1)).",
    )
    _fba1, _fba2, _fba3 = st.columns(3)
    fbauto_p_min  = int(_fba1.number_input("p мин",  1, 50,  2, 1, key="fbauto_p_min"))
    fbauto_p_max  = int(_fba2.number_input("p макс", 1, 100, 16, 1, key="fbauto_p_max"))
    fbauto_p_step = int(_fba3.number_input("шаг",    1, 10,  1,  1, key="fbauto_p_step"))
    fbauto_val_h  = st.slider(
        "Горизонт валидации шума (баров)", 2, 50, 10, 1, key="fbauto_val_h",
        help="Шумовые C0–C2 валидируются на последних N барах перед origin.",
    )
    fbauto_top_n = st.slider(
        "Top-N p для шума", 1, 20, 3, 1, key="fbauto_top_n",
        help="Берутся N лучших p по val_mape, прогнозы усредняются.",
    )
    _FBAUTO_COMP_LABELS = {
        0: "C0 ~2-4б (шум)",   1: "C1 ~4-8б (шум)",   2: "C2 ~8-16б (шум)",
        3: "C3 ~16-52б",       4: "C4 ~52-103б",       5: "C5 103+б",
    }
    fbauto_components = tuple(sorted(st.multiselect(
        "Компоненты в сумму",
        options=[0, 1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        format_func=lambda x: _FBAUTO_COMP_LABELS[x],
        key="fbauto_components",
        help="C0 — широкополосный шум, обычно вредит прогнозу. "
             "Снимите галку для исключения из итоговой суммы.",
    )))

# ── data controls ─────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
ticker     = c1.text_input("Тикер", value="SBER", key="ticker_val").upper().strip()

interval = c2.selectbox(
    "Интервал",
    options=list(INTERVALS),
    index=list(INTERVALS).index("1d"),
    key="interval_val",
)

cache_path = DATA_DIR / ticker / f"{interval}.json"
load_btn   = c3.button("Загрузить",    type="primary", use_container_width=True)
reset_btn  = c4.button("Сбросить кэш", use_container_width=True,
                        disabled=not cache_path.exists())

if cache_path.exists():
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    st.caption(f"Кэш: `{cache_path.relative_to(_ROOT)}` — {mtime:%d.%m.%Y %H:%M}")
else:
    st.caption("Кэш отсутствует")

st.divider()

# ── data loading ──────────────────────────────────────────────────────────────

if reset_btn:
    cache_path.unlink()
    st.rerun()

raw_candles: list[dict] | None = None

if load_btn:
    if cache_path.exists():
        raw_candles = json.loads(cache_path.read_text(encoding="utf-8"))
        st.info(f"Из кэша: {len(raw_candles):,} свечей")
    else:
        with st.spinner(f"Скачиваю {ticker} [{interval}] с MOEX…"):
            raw_candles = download_candles(ticker, interval, show_progress=False)
        if not raw_candles:
            st.error(f"Нет данных для **{ticker}**")
            st.stop()
        save_candles(raw_candles, cache_path)
        st.success(f"Скачано: {len(raw_candles):,} свечей")
elif cache_path.exists():
    raw_candles = json.loads(cache_path.read_text(encoding="utf-8"))

if not raw_candles:
    st.stop()

# ── normalise ─────────────────────────────────────────────────────────────────

# cache key: the JSON string (stable across re-runs as long as file doesn't change)
candles_key = cache_path.read_text(encoding="utf-8")
# сохраняем для кнопки «Авто MA» (sidebar не имеет доступа к candles_key)
st.session_state["candles_key_for_ma"]    = candles_key
st.session_state["candles_ticker_for_ma"] = ticker

norm  = _normalize(candles_key, ma_window)
valid = norm.dropna(subset=["ma"]).reset_index(drop=True)

if _use_logtrend:
    valid = _normalize_logtrend(candles_key).dropna(subset=["ratio"]).reset_index(drop=True)

PERIODS = {"1 мес": 30, "3 мес": 90, "6 мес": 180, "1 год": 365, "3 года": 1095, "Всё": None}

# ── period selector + view ────────────────────────────────────────────────────

period  = st.radio("Период", list(PERIODS), index=1, horizontal=True)
days    = PERIODS[period]
last_dt = valid["begin"].iloc[-1]
view    = (valid[valid["begin"] >= last_dt - pd.Timedelta(days=days)].copy()
           if days else valid.copy())

# rangebreaks
if interval in ("1d", "1w", "1mo"):
    rangebreaks = [dict(bounds=["sat", "mon"])]
else:
    _has_evening = (valid["begin"].dt.hour >= 19).any()
    _overnight   = [0, 10] if _has_evening else [19, 10]
    rangebreaks  = [dict(bounds=["sat", "mon"]),
                    dict(bounds=_overnight, pattern="hour")]

_grid   = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
_layout = dict(dragmode="pan", paper_bgcolor="rgba(0,0,0,0)",
               plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0))

# ── forecast section ─────────────────────────────────────────────────────────

st.subheader("Прогноз")

model = st.radio(
    "Модель",
    ["LA1 — одиночный", "LA1 — авто p", "LWR — авто p",
     "LWR (Гауссовы веса)", "LWR + filter bank", "LA1 + filter bank", "LA1 — ансамбль"],
    horizontal=True,
)
_is_auto_p = model in ("LA1 — авто p", "LWR — авто p")
_is_fb     = model in ("LWR + filter bank", "LA1 + filter bank")
_fb_use_lwr = (model == "LWR + filter bank")

# ── model-specific controls ───────────────────────────────────────────────────

ens_ma_windows: list[int] = []
ens_p_values:   list[int] = []
show_individuals           = False
auto_p_val_h               = 10
auto_p_max                 = 30
ap_n_extra                 = 4

ap_use_pca  = True
ap_use_fb   = False
ap_fb_use_lwr = (model == "LWR — авто p")   # FB метод совпадает с базовой моделью авто-p
if _is_auto_p:
    ap1, ap2, ap3 = st.columns(3)
    auto_p_val_h = int(ap1.number_input(
        "Окно валидации (баров назад)", 5, 50, 5, 1, key="ap_val_h",
        help="Сколько последних известных баров используется для отбора p",
    ))
    auto_p_max = int(ap2.number_input(
        "Максимум p", 5, 100, 70, 1, key="ap_max",
    ))
    ap_n_extra = int(ap3.number_input(
        "Доп. прогнозов", 0, 9, 4, 1, key="ap_n_extra",
        help="Сколько вариантов кроме лучшего показывать на графике",
    ))
    apc1, apc2 = st.columns(2)
    ap_use_pca = apc1.checkbox(
        "PCA-метрика в авто-p",
        value=True,
        key="ap_use_pca",
        help="Включить перебор числа PCA-компонент k вместе с p. "
             "Отключите для ускорения: перебирается только p без PCA (pca_k=0).",
    )
    ap_use_fb = apc2.checkbox(
        "Filter bank в авто-p",
        value=False,
        key="ap_use_fb",
        help="Добавить LWR+filter bank (C3+C4+C5, p из сайдбара) как кандидата "
             "наряду с обычным перебором p. Побеждает тот, у кого меньше val_mape.",
    )
    _pca_note = "× k∈[2,p-1]" if ap_use_pca else "без PCA"
    _fb_note  = f"  |  +filter bank p={fb_p}" if ap_use_fb else ""
    st.caption(
        f"Перебор p ∈ [2, {auto_p_max}] {_pca_note}{_fb_note}  |  "
        f"валидация: {auto_p_val_h} баров  |  прогноз вперёд: {horizon} баров  |  "
        f"показывать: {ap_n_extra + 1} прогноза"
    )
elif model == "LA1 — ансамбль":
    ens_ma_windows = [ma_window]
    st.caption("Перебор по размерности p. Окно MA — из сайдбара.")
    ep1, ep2, ep3 = st.columns(3)
    ep_s = ep1.number_input("p от",  2, 15, 3, 1, key="ep_s")
    ep_e = ep2.number_input("p до",  2, 15, 7, 1, key="ep_e")
    ep_t = ep3.number_input("p шаг", 0, 10, 1, 1, key="ep_t",
                             help="0 или от=до → одно значение")
    ens_p_values = range_vals(int(ep_s), int(ep_e), int(ep_t))
    show_individuals = st.checkbox("Показать отдельные прогнозы", value=False)
    st.caption(
        f"MA={ma_window}  |  p: {ens_p_values}  |  "
        f"Комбинаций: **{len(ens_p_values)}**  |  горизонт: {horizon}"
    )

# ── origin picker ─────────────────────────────────────────────────────────────

fo_col1, fo_col2 = st.columns([3, 1])
with fo_col1:
    _min_p_idx = (auto_p_max + auto_p_val_h + 10 if _is_auto_p else p + 10)
    min_date = valid["begin"].iloc[min(_min_p_idx, len(valid) - 2)].date()
    max_date = valid["begin"].iloc[-horizon - 1].date()

    if "origin_date_val" not in st.session_state:
        st.session_state["origin_date_val"] = max(
            min_date, min((last_dt - pd.Timedelta(days=5)).date(), max_date)
        )
    else:
        st.session_state["origin_date_val"] = max(
            min_date, min(st.session_state["origin_date_val"], max_date)
        )

    origin_date = st.date_input(
        "Начало прогноза",
        min_value=min_date, max_value=max_date,
        key="origin_date_val",
        help="Прогноз строится от этой даты. Реальные данные после точки "
             "остаются на графике — для сравнения с прогнозом.",
    )

run_btn = fo_col2.button("▶ Прогноз", type="primary", use_container_width=True)

origin_ts = pd.Timestamp(origin_date)
origin_k  = int(np.searchsorted(valid["begin"].values, origin_ts))
origin_k  = max(p + 10, min(origin_k, len(valid) - 2))
origin_bar = valid.iloc[origin_k]

if model == "LA1 — одиночный":
    st.caption(
        f"Старт: **{origin_bar['begin']:%d.%m.%Y}**  |  "
        f"p={p}, Ξ={xi}, горизонт={horizon}"
    )
elif _is_auto_p:
    val_origin_bar = valid.iloc[max(0, origin_k - auto_p_val_h)]
    st.caption(
        f"Старт: **{origin_bar['begin']:%d.%m.%Y}**  |  "
        f"валидация: {val_origin_bar['begin']:%d.%m.%Y} → {origin_bar['begin']:%d.%m.%Y}  |  "
        f"p ∈ [2, {auto_p_max}], горизонт вперёд: {horizon}"
    )
else:
    st.caption(f"Старт: **{origin_bar['begin']:%d.%m.%Y}**")

# ── run forecast ──────────────────────────────────────────────────────────────

dratio    = np.diff(valid["ratio"].values)
dratio_key = dratio.tobytes()
d2ratio    = np.diff(dratio)
d2ratio_key = d2ratio.tobytes()

_cur_phase_lbl = PHASE_NAMES[
    query_phase_label(valid["ratio"].values, dratio, origin_k, phase_dn, phase_up)
]
st.caption(f"Фаза: **{_cur_phase_lbl}**")

# ── rolling Hurst + адаптивный порог + режимная маска ────────────────────────

h_idx, h_vals = _hurst_zones(dratio_key)
valid_h       = h_vals[~np.isnan(h_vals)]

hurst_thr = hurst_percentile_threshold(valid_h, hurst_pct) if len(valid_h) else 0.60

# bar_hurst[i] — значение Hurst для бара i в valid
bar_hurst = np.full(len(valid), np.nan)
for ci, hv in zip(h_idx, h_vals):
    if not np.isnan(hv):
        bar_hurst[int(ci): int(ci) + 20] = hv

# rolling PE по всему dratio-ряду
bar_pe   = _rolling_pe(dratio_key, win=pe_win if use_pe else 50) if use_pe else None
valid_pe = bar_pe[~np.isnan(bar_pe)] if bar_pe is not None else np.array([])


# маски для пула соседей при заданном origin_k
def _make_regime_mask(ok: int) -> np.ndarray | None:
    if not use_regime:
        return None
    pool_size = max(0, ok - p)
    if pool_size < 1:
        return None
    return regime_mask_for_pool(bar_hurst, pool_size, p, hurst_thr)


def _make_phase_mask(ok: int) -> np.ndarray | None:
    if not use_phase:
        return None
    pool_size = max(0, ok - p)
    if pool_size < 1:
        return None
    ratio_arr  = valid["ratio"].values
    dratio_hist = dratio[:ok]
    labels  = pool_phase_labels(ratio_arr, dratio_hist, p, phase_dn, phase_up)
    q_label = query_phase_label(ratio_arr, dratio_hist, ok, phase_dn, phase_up)
    return (labels == q_label)


def _make_pe_mask(ok: int) -> tuple[np.ndarray | None, float | None]:
    """Returns (mask, used_value).
    threshold-режим: used_value — квантиль, > max_quantile → fallback.
    proximity-режим: used_value — delta, > max_delta → fallback.
    """
    if not use_pe or bar_pe is None:
        return None, None
    pool_size = max(0, ok - p)
    if pool_size < 1:
        return None, None
    if pe_mode == "proximity":
        pe_origin_idx = ok - 1
        if pe_origin_idx < 0 or pe_origin_idx >= len(bar_pe):
            return None, None
        pe_origin_val = float(bar_pe[pe_origin_idx])
        if np.isnan(pe_origin_val):
            return None, None
        mask, used = pe_proximity_mask(
            bar_pe, pool_size, p,
            pe_origin=pe_origin_val,
            n_neighbors=xi,
            base_delta=pe_delta_base,
            max_delta=pe_delta_max,
        )
    else:
        mask, used = pe_adaptive_mask(
            bar_pe, pool_size, p,
            base_quantile=pe_pct / 100.0,
            n_neighbors=xi,
            step=0.05,
            max_quantile=pe_max_pct / 100.0,
        )
    return mask, used


def _combined_mask(ok: int) -> np.ndarray | None:
    pe_m, _ = _make_pe_mask(ok)
    masks = [m for m in [
        _make_regime_mask(ok),
        _make_phase_mask(ok),
        pe_m,
    ] if m is not None]
    if not masks:
        return None
    result = masks[0]
    for m in masks[1:]:
        result = result & m
    return result

# показываем адаптивный порог Hurst
st.sidebar.caption(f"Порог H (P{hurst_pct}): **{hurst_thr:.3f}**  "
                   f"| предсказуемо: {(valid_h < hurst_thr).mean()*100:.0f}%")

# показываем PE-фильтр для текущего origin
if use_pe and len(valid_pe) and origin_k > p:
    _pe_pool, _pe_used = _make_pe_mask(origin_k)
    if _pe_pool is not None:
        _pe_n     = int(_pe_pool.sum())
        _pe_total = len(_pe_pool)
        if pe_mode == "proximity":
            _pe_origin_val = float(bar_pe[origin_k - 1]) if origin_k > 0 else float("nan")
            _is_fb   = (_pe_used is not None and _pe_used > pe_delta_max + 1e-9)
            _relaxed = (not _is_fb and _pe_used is not None
                        and _pe_used > pe_delta_base + 1e-9)
            _badge   = " ⬆расширен" if _relaxed else (" ⚠ fallback" if _is_fb else "")
            _val_lbl = f"{_pe_origin_val:.4f}" if not np.isnan(_pe_origin_val) else "n/a"
            st.sidebar.caption(
                f"PE_origin: **{_val_lbl}**  |  "
                f"PE пул: **{_pe_n:,}** / {_pe_total:,}  |  "
                f"eff. ΔPE: **{_pe_used:.3f}**{_badge}"
            )
        else:
            _is_fb   = (_pe_used is not None and _pe_used > pe_max_pct / 100.0)
            _relaxed = (not _is_fb and _pe_used is not None
                        and _pe_used > pe_pct / 100.0 + 1e-9)
            _badge   = " ⬆ослаблен" if _relaxed else (" ⚠ fallback" if _is_fb else "")
            _q_label = "fallback" if _is_fb else f"P{round(_pe_used * 100)}"
            st.sidebar.caption(
                f"PE пул: **{_pe_n:,}** / {_pe_total:,}  |  "
                f"eff. квантиль: **{_q_label}**{_badge}"
            )

# показываем долю пула в текущей фазе
if use_phase and origin_k > p:
    _pool_labels = pool_phase_labels(
        valid["ratio"].values, dratio[:origin_k], p, phase_dn, phase_up
    )
    _q_lbl = query_phase_label(valid["ratio"].values, dratio, origin_k, phase_dn, phase_up)
    _same  = (_pool_labels == _q_lbl).sum()
    st.sidebar.caption(
        f"Фаза: **{PHASE_NAMES[_q_lbl]}**  |  "
        f"в пуле: {_same:,} / {len(_pool_labels):,} "
        f"({100*_same/max(len(_pool_labels),1):.0f}%)"
    )

# ── forecast ──────────────────────────────────────────────────────────────────

forecast_price:   np.ndarray | None       = None
forecast_indivs:  list[np.ndarray] | None = None
forecast_begin:   pd.Series | None        = None

_cmask_pre    = _combined_mask(origin_k)
_regime_bytes = _cmask_pre.tobytes() if _cmask_pre is not None else b""
_ss_params_key = (model, ma_window, p, xi, horizon, ticker,
                  tuple(ens_ma_windows), tuple(ens_p_values),
                  use_regime, hurst_pct, norm_vecs, use_huber,
                  use_phase, phase_margin_pct, pca_k,
                  auto_p_val_h, auto_p_max, use_ratio_series,
                  use_pe, pe_win if use_pe else 0, pe_mode if use_pe else "",
                  pe_pct if use_pe else 0, pe_max_pct if use_pe else 0,
                  pe_delta_base if use_pe else 0, pe_delta_max if use_pe else 0,
                  ap_use_pca, _regime_bytes,
                  use_regime, hurst_thr, use_pe, pe_pct if use_pe else 0,
                  ap_use_fb, fb_p, fb_xi, ap_fb_use_lwr, fb_components,
                  fb_ppc_tuple, use_d2ratio, fb_merge_noise,
                  fb_damped_ar, fb_gamma_c1, fb_gamma_c2, norm_mode)

if run_btn:
    with st.spinner("Считаю прогноз…"):
        ratio0 = float(valid["ratio"].iloc[origin_k])
        ma0    = float(valid["ma"].iloc[origin_k])
        cmask  = _combined_mask(origin_k)
        rb     = cmask.tobytes() if cmask is not None else None

        _ma_vals = valid["ma"].values

        if use_ma_trend:
            _ma_fwd = _ma_trend(_ma_vals, origin_k, horizon, ma_fit_bars)
        else:
            _ma_fwd = ma0  # scalar — numpy broadcasts fine

        ratio_arr = valid["ratio"].values
        ratio_key = ratio_arr.tobytes()

        if model == "LA1 — одиночный":
            if use_ratio_series:
                dhat = _forecast(ratio_key, origin_k + 1, p, xi, horizon, rb,
                                 norm_vecs, use_huber, pca_k)
                forecast_price = dhat * _ma_fwd
            else:
                dhat = _forecast(dratio_key, origin_k, p, xi, horizon, rb,
                                 norm_vecs, use_huber, pca_k)
                forecast_price = reconstruct_price(dhat, ratio0, _ma_fwd)
            forecast_indivs = None
        elif model == "LWR (Гауссовы веса)":
            if use_ratio_series:
                dhat = _forecast_lwr(ratio_key, origin_k + 1, p, xi, horizon, rb,
                                     norm_vecs, use_huber, pca_k)
                forecast_price = dhat * _ma_fwd
            else:
                dhat = _forecast_lwr(dratio_key, origin_k, p, xi, horizon, rb,
                                     norm_vecs, use_huber, pca_k)
                forecast_price = reconstruct_price(dhat, ratio0, _ma_fwd)
            forecast_indivs = None
        elif _is_fb:
            _fb_lwr  = (model == "LWR + filter bank")
            _fb_cidx = tuple(fb_components)
            if fb_damped_ar and not use_ratio_series and not use_d2ratio:
                _slow_ci = tuple(ci for ci in _fb_cidx if ci >= 3)
                dhat = _forecast_fb_damped_cached(
                    dratio_key, origin_k, horizon,
                    p_slow=fb_p, xi_slow=fb_xi,
                    gamma_c1=fb_gamma_c1, gamma_c2=fb_gamma_c2,
                    slow_components=_slow_ci or (3, 4, 5),
                )
                _ma_vals_bytes = valid["ma"].values.tobytes()
                _sl_idx, _sl_prices, _junction = _fb_slow_trend(
                    dratio_key, ratio_key, _ma_vals_bytes,
                    origin_k, n_display=200,
                    component_idx=_slow_ci or _fb_cidx,
                )
                _slow_ratio_at_origin = _junction / ma0
                ratio_hat      = _slow_ratio_at_origin + np.cumsum(dhat)
                forecast_price = ratio_hat * _ma_fwd
                st.session_state["fc_fb_slow_x"]   = _sl_idx
                st.session_state["fc_fb_slow_y"]   = _sl_prices
                st.session_state["fc_fb_anchor_y"] = _junction
                forecast_indivs = None
            elif fb_merge_noise and not use_ratio_series and not use_d2ratio:
                _fast_ci = tuple(ci for ci in _fb_cidx if ci < 3)
                _slow_ci = tuple(ci for ci in _fb_cidx if ci >= 3)
                dhat = _forecast_fb_merged_noise(
                    dratio_key, origin_k, horizon,
                    p=fb_p, xi=fb_xi,
                    fast_components=_fast_ci,
                    slow_components=_slow_ci,
                    p_per_slow=fb_ppc_tuple,
                )
                _ma_vals_bytes = valid["ma"].values.tobytes()
                _sl_idx, _sl_prices, _junction = _fb_slow_trend(
                    dratio_key, ratio_key, _ma_vals_bytes,
                    origin_k, n_display=200, component_idx=_slow_ci or _fb_cidx,
                )
                _slow_ratio_at_origin = _junction / ma0
                ratio_hat      = _slow_ratio_at_origin + np.cumsum(dhat)
                forecast_price = ratio_hat * _ma_fwd
                st.session_state["fc_fb_slow_x"]   = _sl_idx
                st.session_state["fc_fb_slow_y"]   = _sl_prices
                st.session_state["fc_fb_anchor_y"] = _junction
                forecast_indivs = None
            elif use_ratio_series:
                dhat = _forecast_fb(ratio_key, origin_k + 1, horizon,
                                    p=fb_p, xi=fb_xi, use_lwr=_fb_lwr,
                                    component_idx=_fb_cidx,
                                    p_per_component=fb_ppc_tuple)
                forecast_price = dhat * _ma_fwd
                st.session_state["fc_fb_slow_x"]   = None
                st.session_state["fc_fb_slow_y"]   = None
                st.session_state["fc_fb_anchor_y"] = float(dhat[0] * ma0)
            elif use_d2ratio:
                # d²ratio + однократный cumsum (исслед. 38: −33% от raw dratio).
                # Прогноз — ускорение ratio; slow trend от dratio несовместим
                # (разные пространства) и создаёт угол на стыке → не показываем.
                dhat = _forecast_fb(d2ratio_key, origin_k - 1, horizon,
                                    p=fb_p, xi=fb_xi, use_lwr=_fb_lwr,
                                    component_idx=_fb_cidx,
                                    p_per_component=fb_ppc_tuple)
                ratio0_d2   = float(valid["ratio"].iloc[origin_k])
                ratio_hat   = ratio0_d2 + np.cumsum(dhat)
                forecast_price = ratio_hat * _ma_fwd
                st.session_state["fc_fb_slow_x"]   = None
                st.session_state["fc_fb_slow_y"]   = None
                st.session_state["fc_fb_anchor_y"] = float(valid["close"].iloc[origin_k])
            else:
                dhat = _forecast_fb(dratio_key, origin_k, horizon,
                                    p=fb_p, xi=fb_xi, use_lwr=_fb_lwr,
                                    component_idx=_fb_cidx,
                                    p_per_component=fb_ppc_tuple)
                # slow trend history + correct anchor (не actual close, а slow trend)
                _ma_vals_bytes = valid["ma"].values.tobytes()
                _sl_idx, _sl_prices, _junction = _fb_slow_trend(
                    dratio_key, ratio_key, _ma_vals_bytes,
                    origin_k, n_display=200, component_idx=_fb_cidx,
                )
                _slow_ratio_at_origin = _junction / ma0
                ratio_hat   = _slow_ratio_at_origin + np.cumsum(dhat)
                forecast_price = ratio_hat * _ma_fwd
                st.session_state["fc_fb_slow_x"]   = _sl_idx
                st.session_state["fc_fb_slow_y"]   = _sl_prices
                st.session_state["fc_fb_anchor_y"] = _junction
            forecast_indivs = None
        elif _is_auto_p:
            _ap_pe_thr = (float(np.quantile(valid_pe, pe_pct / 100.0))
                          if use_pe and len(valid_pe) else 1.0)
            top_cands, ap_scores = _auto_p_forecast(
                dratio_key, ratio_key, origin_k,
                auto_p_val_h, auto_p_max, horizon,
                norm_vecs, use_huber,
                use_lwr=(model == "LWR — авто p"),
                top_n=10,
                series_is_ratio=use_ratio_series,
                use_pca=ap_use_pca,
                bar_hurst_bytes=bar_hurst.tobytes() if use_regime else b"",
                hurst_thr=hurst_thr,
                use_hurst_filter=use_regime,
                bar_pe_bytes=bar_pe.tobytes() if (use_pe and bar_pe is not None) else b"",
                pe_thr=_ap_pe_thr,
                use_pe_filter=use_pe,
                pe_mode=pe_mode,
                pe_delta_base=pe_delta_base,
                pe_delta_max=pe_delta_max,
                use_filterbank=ap_use_fb,
                fb_p=fb_p,
                fb_xi=fb_xi,
                fb_use_lwr=ap_fb_use_lwr,
                fb_components=fb_components,
            )
            if not top_cands:
                st.warning("Авто-p: нет результатов — слишком мало данных или p_max мал")
                st.stop()
            val_origin_k  = origin_k - auto_p_val_h
            ratio0_val    = float(valid["ratio"].iloc[val_origin_k])
            val_ma_actual = _ma_vals[val_origin_k + 1 : val_origin_k + 1 + auto_p_val_h]

            # reconstruct prices for each top candidate
            top_cands_prices = []
            for cand in top_cands:
                c_dhat = np.array(cand["dhat"])
                if use_ratio_series:
                    c_val_ratio  = c_dhat[:auto_p_val_h]
                    c_main_ratio = c_dhat[auto_p_val_h : auto_p_val_h + horizon]
                else:
                    c_val_ratio  = ratio0_val + np.cumsum(c_dhat[:auto_p_val_h])
                    c_main_ratio = ratio0_val + np.cumsum(c_dhat)[auto_p_val_h : auto_p_val_h + horizon]
                c_val_price  = c_val_ratio[:len(val_ma_actual)] * val_ma_actual
                c_main_price = c_main_ratio * _ma_fwd
                c_junction_y = float(c_val_price[-1]) if len(c_val_price) else float(c_main_price[0])
                top_cands_prices.append({
                    "p": cand["p"], "pca_k": cand["pca_k"], "mape": cand["mape"],
                    "val_price":  c_val_price.tolist(),
                    "main_price": c_main_price.tolist(),
                    "junction_y": c_junction_y,
                })

            best = top_cands_prices[0]
            forecast_price  = np.array(best["main_price"])
            forecast_indivs = None
            st.session_state["fc_ap_model"]       = model
            st.session_state["fc_ap_best"]        = best["p"]
            st.session_state["fc_ap_best_pca_k"]  = best["pca_k"]
            st.session_state["fc_ap_val_mape"]    = best["mape"]
            st.session_state["fc_ap_val_price"]   = np.array(best["val_price"])
            st.session_state["fc_ap_val_orig_k"]  = val_origin_k
            st.session_state["fc_ap_junction_y"]  = best["junction_y"]
            st.session_state["fc_ap_scores"]      = ap_scores
            st.session_state["fc_ap_top3_prices"] = top_cands_prices
        else:  # ансамбль
            mean_l, all_l, _ = _forecast_ensemble(
                candles_key, str(origin_ts),
                tuple(ens_ma_windows), tuple(ens_p_values), horizon,
            )
            forecast_price  = np.array(mean_l)
            forecast_indivs = [np.array(f) for f in all_l]

    st.session_state["fc_price"]    = forecast_price
    st.session_state["fc_indivs"]   = forecast_indivs
    st.session_state["fc_origin_k"] = origin_k
    st.session_state["fc_params"]   = _ss_params_key
    st.session_state["fc_regime_b"] = _regime_bytes

# restore from session state if all params unchanged
if (
    "fc_price" in st.session_state
    and st.session_state.get("fc_params") == _ss_params_key
    and st.session_state.get("fc_origin_k") == origin_k
    and st.session_state.get("fc_regime_b") == _regime_bytes
):
    forecast_price   = st.session_state["fc_price"]
    forecast_indivs  = st.session_state["fc_indivs"]
    forecast_origin_k = st.session_state["fc_origin_k"]

    future_valid = valid.iloc[forecast_origin_k + 1: forecast_origin_k + 1 + horizon]
    if len(future_valid) < horizon:
        bar_delta  = _BAR_DELTA.get(interval, pd.Timedelta(hours=1))
        last_known = (future_valid["begin"].iloc[-1] if len(future_valid)
                      else valid["begin"].iloc[forecast_origin_k])
        extra = [last_known + bar_delta * i
                 for i in range(1, horizon - len(future_valid) + 1)]
        forecast_begin = pd.concat(
            [future_valid["begin"].reset_index(drop=True), pd.Series(extra)],
            ignore_index=True,
        )
    else:
        forecast_begin = future_valid["begin"].reset_index(drop=True)

# ── FB авто-p по компонентам ──────────────────────────────────────────────────

_fbauto_key = (origin_k, horizon, fbauto_val_h,
               fbauto_p_slow, fbauto_p_min, fbauto_p_max, fbauto_p_step,
               fbauto_components, fbauto_top_n)

_fbauto_col1, _fbauto_col2 = st.columns([1, 4])
_fbauto_run = _fbauto_col1.button("▶ FB авто-p", type="secondary")
if _fbauto_run:
    with st.spinner("FB авто-p: перебираю p для C0–C2…"):
        _fa_dhat, _fa_comp = _forecast_fb_autop(
            dratio_key, origin_k, horizon,
            fbauto_val_h, fbauto_p_slow,
            fbauto_p_min, fbauto_p_max, fbauto_p_step,
            fbauto_components, fbauto_top_n,
        )
    st.session_state["fbauto_result"] = {
        "dhat":     _fa_dhat.tolist(),
        "per_comp": _fa_comp,
        "key":      _fbauto_key,
    }

# ── результаты FB авто-p ──────────────────────────────────────────────────────

_fbauto_forecast_price: np.ndarray | None = None
_fbauto_forecast_begin: pd.Series | None  = None

if (
    "fbauto_result" in st.session_state
    and st.session_state["fbauto_result"]["key"] == _fbauto_key
):
    _fa  = st.session_state["fbauto_result"]
    _fa_dhat   = np.array(_fa["dhat"])
    _fa_comp   = _fa["per_comp"]

    _fa_ratio0 = float(valid["ratio"].iloc[origin_k])
    _fa_ma0    = float(valid["ma"].iloc[origin_k])
    _fa_ratio_hat   = _fa_ratio0 + np.cumsum(_fa_dhat)
    _fbauto_forecast_price = _fa_ratio_hat * _fa_ma0

    # временна́я ось (та же логика, что у основного прогноза)
    _fa_future = valid.iloc[origin_k + 1: origin_k + 1 + horizon]
    if len(_fa_future) < horizon:
        _fa_bar_dt  = _BAR_DELTA.get(interval, pd.Timedelta(hours=1))
        _fa_last    = (_fa_future["begin"].iloc[-1] if len(_fa_future)
                       else valid["begin"].iloc[origin_k])
        _fa_extra   = [_fa_last + _fa_bar_dt * i
                       for i in range(1, horizon - len(_fa_future) + 1)]
        _fbauto_forecast_begin = pd.concat(
            [_fa_future["begin"].reset_index(drop=True), pd.Series(_fa_extra)],
            ignore_index=True,
        )
    else:
        _fbauto_forecast_begin = _fa_future["begin"].reset_index(drop=True)

    # таблица per-component
    _COMP_LABELS_LOCAL = {
        0: "C0 ~2-4б",  1: "C1 ~4-8б",   2: "C2 ~8-16б",
        3: "C3 ~16-52б", 4: "C4 ~52-103б", 5: "C5 103+б",
    }
    _fa_active_set = set(fbauto_components)
    _fa_rows = []
    for _ci in range(6):
        _ci_info  = _fa_comp.get(_ci, _fa_comp.get(str(_ci), {}))
        _included = _ci in _fa_active_set
        _is_slow  = _ci_info.get("slow", _ci >= 3)
        _top      = _ci_info.get("top", [])

        if not _included:
            _fa_rows.append({
                "C": f"C{_ci}", "в сумме": "✗", "тип": "медл." if _is_slow else "шум",
                "top p": "—", "avg val MAPE": "—",
            })
        elif _is_slow:
            _p_val = _top[0]["p"] if _top else "—"
            _xi_val = _top[0]["xi"] if _top else "—"
            _fa_rows.append({
                "C": f"C{_ci}", "в сумме": "✓", "тип": "медл.",
                "top p": f"{_p_val}  ξ={_xi_val}", "avg val MAPE": "—",
            })
        else:
            _p_list = ", ".join(str(t["p"]) for t in _top)
            _mapes  = [t["val_mape"] for t in _top if t["val_mape"] is not None]
            _avg_m  = f"{np.mean(_mapes):.5f}" if _mapes else "—"
            _fa_rows.append({
                "C": f"C{_ci}", "в сумме": "✓", "тип": "шум",
                "top p": _p_list, "avg val MAPE": _avg_m,
            })
    with _fbauto_col2:
        st.dataframe(pd.DataFrame(_fa_rows), hide_index=True, use_container_width=True)

# ── chart 1: candlestick + MA + forecast ─────────────────────────────────────

fig1 = go.Figure()

fig1.add_trace(go.Candlestick(
    x=view["begin"], open=view["open"], high=view["high"],
    low=view["low"],  close=view["close"],
    name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
))
fig1.add_trace(go.Scatter(
    x=view["begin"], y=view["ma"],
    name=("logtrend" if _use_logtrend else f"SMA({ma_window})"),
    line=dict(color="#ff9800", width=1.5),
))

_fb_params_match = (
    _is_fb and
    st.session_state.get("fc_params") == _ss_params_key and
    st.session_state.get("fc_origin_k") == origin_k
)

# ── FB slow trend line (before forecast section) ──────────────────────────────
if _fb_params_match and st.session_state.get("fc_fb_slow_x") is not None:
    _sl_idx    = st.session_state["fc_fb_slow_x"]
    _sl_prices = st.session_state["fc_fb_slow_y"]
    _sl_ts     = valid["begin"].iloc[_sl_idx].values
    fig1.add_trace(go.Scatter(
        x=_sl_ts, y=_sl_prices,
        name="Медленный тренд (FB)",
        line=dict(color="rgba(0,210,170,0.80)", width=1.5, dash="dot"),
        mode="lines",
    ))

if forecast_price is not None and forecast_begin is not None:
    anchor_x = [valid["begin"].iloc[forecast_origin_k]]
    # авто-p: соединяем solid линию с концом валидационной (не с фактическим close)
    _ap_active = _is_auto_p and st.session_state.get("fc_ap_model") == model
    if _ap_active and "fc_ap_junction_y" in st.session_state:
        anchor_y = [st.session_state["fc_ap_junction_y"]]
    elif _fb_params_match and "fc_fb_anchor_y" in st.session_state:
        anchor_y = [st.session_state["fc_fb_anchor_y"]]
    else:
        anchor_y = [valid["close"].iloc[forecast_origin_k]]
    x_full   = pd.concat([pd.Series(anchor_x), forecast_begin], ignore_index=True)

    fig1.add_vline(
        x=valid["begin"].iloc[forecast_origin_k].timestamp() * 1000,
        line=dict(color="rgba(255,214,0,0.5)", width=1.5, dash="dot"),
    )

    # individual ensemble traces (thin, semi-transparent)
    if show_individuals and forecast_indivs:
        for indiv in forecast_indivs:
            fig1.add_trace(go.Scatter(
                x=x_full,
                y=np.concatenate([anchor_y, indiv]),
                line=dict(color="rgba(100,210,255,0.18)", width=1),
                mode="lines", showlegend=False, hoverinfo="skip",
            ))

    # auto-p: validation window + candidate traces + mean + std band
    if _ap_active and "fc_ap_val_price" in st.session_state:
        _ap_val_orig_k = st.session_state["fc_ap_val_orig_k"]

        _val_t0 = valid["begin"].iloc[_ap_val_orig_k]
        _val_t1 = valid["begin"].iloc[forecast_origin_k]
        fig1.add_vrect(
            x0=_val_t0, x1=_val_t1,
            fillcolor="rgba(255,200,0,0.07)", line_width=0, layer="below",
        )
        fig1.add_vline(
            x=_val_t0.timestamp() * 1000,
            line=dict(color="rgba(255,160,0,0.45)", width=1, dash="dot"),
        )
        _val_x       = valid.iloc[_ap_val_orig_k : forecast_origin_k + 1]["begin"].reset_index(drop=True)
        _val_start_y = float(valid["close"].iloc[_ap_val_orig_k])
        _top_shown   = st.session_state.get("fc_ap_top3_prices", [])[:ap_n_extra + 1]
        _MEAN_C      = "#e040fb"
        _BC          = "rgba(120, 190, 255, 0.6)"

        _all_val_ys = [
            np.concatenate([[_val_start_y], np.array(c["val_price"])])
            for c in _top_shown
        ]

        # std band in validation zone
        if len(_all_val_ys) >= 2:
            _vlen    = min(len(v) for v in _all_val_ys)
            _vx_list = list(_val_x[:_vlen])
            _vm_arr  = np.array([v[:_vlen] for v in _all_val_ys])
            _v_mean  = _vm_arr.mean(axis=0)
            _v_std   = _vm_arr.std(axis=0)
            fig1.add_trace(go.Scatter(
                x=_vx_list + _vx_list[::-1],
                y=np.concatenate([_v_mean + _v_std, (_v_mean - _v_std)[::-1]]),
                fill="toself", fillcolor="rgba(120, 190, 255, 0.20)",
                line=dict(width=0),
                mode="lines", showlegend=False, hoverinfo="skip",
            ))
            fig1.add_trace(go.Scatter(
                x=_vx_list, y=_v_mean + _v_std,
                line=dict(color=_BC, width=1, dash="dot"),
                mode="lines", showlegend=False, hoverinfo="skip",
            ))
            fig1.add_trace(go.Scatter(
                x=_vx_list, y=_v_mean - _v_std,
                line=dict(color=_BC, width=1, dash="dot"),
                mode="lines", showlegend=False, hoverinfo="skip",
            ))

        # best validation (dashed yellow)
        if _top_shown:
            _cv_y0 = _all_val_ys[0]
            _n0    = min(len(_val_x), len(_cv_y0))
            fig1.add_trace(go.Scatter(
                x=_val_x[:_n0], y=_cv_y0[:_n0],
                name=f"Вал. #1 p={_top_shown[0]['p']}, pca={_top_shown[0]['pca_k']}",
                line=dict(color="#ffd600", width=2, dash="dash"),
                mode="lines",
            ))

        # mean validation (solid fuchsia)
        if len(_all_val_ys) >= 2:
            fig1.add_trace(go.Scatter(
                x=_val_x[:_vlen], y=_v_mean,
                name="Вал. среднее",
                line=dict(color=_MEAN_C, width=2),
                mode="lines",
            ))

    # auto-p: main forecast mean + std band
    if _ap_active and "fc_ap_top3_prices" in st.session_state:
        _top_shown = st.session_state["fc_ap_top3_prices"][:ap_n_extra + 1]
        _MEAN_C    = "#e040fb"
        _BC        = "rgba(120, 190, 255, 0.6)"
        _c_x_full  = pd.concat([pd.Series(anchor_x), forecast_begin], ignore_index=True)

        if len(_top_shown) >= 2:
            _all_main = np.array([c["main_price"] for c in _top_shown])
            _m_mean   = _all_main.mean(axis=0)
            _m_std    = _all_main.std(axis=0)
            _jy_arr   = np.array([c["junction_y"] for c in _top_shown])
            _mean_jy  = float(_jy_arr.mean())
            _std_jy   = float(_jy_arr.std())
            _band_top = np.concatenate([[_mean_jy + _std_jy], _m_mean + _m_std])
            _band_bot = np.concatenate([[_mean_jy - _std_jy], _m_mean - _m_std])
            _cx_list  = list(_c_x_full)
            fig1.add_trace(go.Scatter(
                x=_cx_list + _cx_list[::-1],
                y=np.concatenate([_band_top, _band_bot[::-1]]),
                fill="toself", fillcolor="rgba(120, 190, 255, 0.20)",
                line=dict(width=0),
                mode="lines", showlegend=False, hoverinfo="skip",
            ))
            fig1.add_trace(go.Scatter(
                x=_cx_list, y=_band_top,
                line=dict(color=_BC, width=1, dash="dot"),
                mode="lines", showlegend=False, hoverinfo="skip",
            ))
            fig1.add_trace(go.Scatter(
                x=_cx_list, y=_band_bot,
                line=dict(color=_BC, width=1, dash="dot"),
                mode="lines", showlegend=False, hoverinfo="skip",
            ))

        # mean forecast (solid fuchsia)
        if len(_top_shown) >= 2:
            fig1.add_trace(go.Scatter(
                x=_c_x_full,
                y=np.concatenate([[_mean_jy], _m_mean]),
                name="Среднее",
                line=dict(color=_MEAN_C, width=2.5),
                mode="lines",
            ))

    # mean / single forecast (bold yellow)
    if _ap_active:
        _ap_mape = st.session_state.get("fc_ap_val_mape", float("nan"))
        label = (f"#1 p={st.session_state.get('fc_ap_best','?')} "
                 f"pca={st.session_state.get('fc_ap_best_pca_k','?')} "
                 f"(MAPE {_ap_mape*100:.2f}%)")
    elif model == "LA1 — ансамбль":
        label = "Ансамбль LA1"
    elif model == "LWR (Гауссовы веса)":
        label = "LWR"
    elif model == "LWR + filter bank":
        _fb_ser = "ratio" if use_ratio_series else ("d²ratio" if use_d2ratio else "Δratio")
        label = f"LWR+FB {_fb_ser}  p={fb_p} ξ={fb_xi}"
    elif model == "LA1 + filter bank":
        _fb_ser = "ratio" if use_ratio_series else ("d²ratio" if use_d2ratio else "Δratio")
        label = f"LA1+FB {_fb_ser}  p={fb_p} ξ={fb_xi}"
    else:
        label = "Прогноз LA1"
    fig1.add_trace(go.Scatter(
        x=x_full,
        y=np.concatenate([anchor_y, forecast_price]),
        name=label,
        line=dict(color="#ffd600", width=3),
        mode="lines",
    ))

# ── FB авто-p трейс ──────────────────────────────────────────────────────────
if _fbauto_forecast_price is not None and _fbauto_forecast_begin is not None:
    _fa_anchor_x = [valid["begin"].iloc[origin_k]]
    _fa_anchor_y = [float(valid["close"].iloc[origin_k])]
    _fa_x_full   = pd.concat([pd.Series(_fa_anchor_x), _fbauto_forecast_begin],
                              ignore_index=True)
    fig1.add_trace(go.Scatter(
        x=_fa_x_full,
        y=np.concatenate([_fa_anchor_y, _fbauto_forecast_price]),
        name=(
            f"FB авто-p  C={list(fbauto_components)}  "
            f"шум p∈[{fbauto_p_min},{fbauto_p_max}]  медл p={fbauto_p_slow}"
        ),
        line=dict(color="#00e5ff", width=2.5, dash="dot"),
        mode="lines",
    ))

# ── Hurst unpredictable zones ─────────────────────────────────────────────────

if show_hurst:
    valid_mask = ~np.isnan(h_vals)
    if valid_mask.any():
        zones = unpredictable_zones(
            valid["begin"].values,
            h_idx[valid_mask],
            h_vals[valid_mask],
            threshold=hurst_thr,
        )
        # filter to visible period for performance
        view_start = view["begin"].iloc[0]
        view_end   = view["begin"].iloc[-1]
        for z0, z1 in zones:
            if z1 < view_start or z0 > view_end:
                continue
            fig1.add_vrect(
                x0=max(z0, view_start), x1=min(z1, view_end),
                fillcolor="rgba(255, 60, 60, 0.10)",
                line_width=0,
                layer="below",
            )

fig1.update_layout(
    **_layout, yaxis_title="Цена, руб.",
    xaxis_rangeslider_visible=False, height=500,
    legend=dict(orientation="h", y=1.02, x=0),
)
fig1.update_xaxes(rangebreaks=rangebreaks, **_grid)
fig1.update_yaxes(**_grid)
st.plotly_chart(fig1, use_container_width=True, config={"scrollZoom": True})

# ── chart 2: rolling PE (только если фильтр включён) ─────────────────────────

if use_pe and bar_pe is not None and len(valid_pe):
    # bar_pe[i] → конец окна dratio[i] → бар valid.iloc[i+1]
    _pe_dates = valid["begin"].iloc[1:].reset_index(drop=True)
    _pe_vals  = pd.Series(bar_pe)

    _view_start = view["begin"].iloc[0]
    _view_end   = view["begin"].iloc[-1]
    _in_view = (_pe_dates >= _view_start) & (_pe_dates <= _view_end)
    _px = _pe_dates[_in_view].values
    _py = _pe_vals[_in_view].values

    _pe_min = float(np.nanmin(_py)) if len(_py) else 0.8
    _pe_max_v = float(np.nanmax(_py)) if len(_py) else 1.0

    # PE в точке origin (для proximity)
    _pe_origin_display = (float(bar_pe[origin_k - 1])
                          if origin_k > 0 and not np.isnan(bar_pe[origin_k - 1])
                          else None)

    fig_pe = go.Figure()

    if pe_mode == "proximity" and _pe_origin_display is not None:
        # зоны: полоса base_delta (зелёная) и max_delta (жёлтая) вокруг PE_origin
        _lo_base = _pe_origin_display - pe_delta_base
        _hi_base = _pe_origin_display + pe_delta_base
        _lo_max  = _pe_origin_display - pe_delta_max
        _hi_max  = _pe_origin_display + pe_delta_max
        fig_pe.add_hrect(y0=max(_lo_max,  _pe_min), y1=min(_hi_max,  _pe_max_v),
                         fillcolor="rgba(255,152,0,0.10)", line_width=0, layer="below")
        fig_pe.add_hrect(y0=max(_lo_base, _pe_min), y1=min(_hi_base, _pe_max_v),
                         fillcolor="rgba(38,166,154,0.20)", line_width=0, layer="below")
        fig_pe.add_hline(
            y=_pe_origin_display,
            line=dict(color="rgba(255,214,0,0.9)", width=1.5, dash="dash"),
            annotation_text=f"PE_origin = {_pe_origin_display:.4f}",
            annotation_font_color="rgba(255,214,0,1)", annotation_font_size=10,
        )
        fig_pe.add_hline(
            y=_lo_max, line=dict(color="rgba(255,152,0,0.6)", width=1, dash="dot"),
        )
        fig_pe.add_hline(
            y=_hi_max, line=dict(color="rgba(255,152,0,0.6)", width=1, dash="dot"),
            annotation_text=f"±{pe_delta_max:.3f}",
            annotation_font_color="rgba(255,152,0,1)", annotation_font_size=10,
        )
    else:
        _q_lo = float(np.quantile(valid_pe, pe_pct / 100.0))
        _q_hi = float(np.quantile(valid_pe, pe_max_pct / 100.0))
        fig_pe.add_hrect(y0=_pe_min,  y1=_q_lo,
                         fillcolor="rgba(38,166,154,0.15)", line_width=0, layer="below")
        fig_pe.add_hrect(y0=_q_lo, y1=_q_hi,
                         fillcolor="rgba(255,152,0,0.10)", line_width=0, layer="below")
        fig_pe.add_hrect(y0=_q_hi, y1=_pe_max_v,
                         fillcolor="rgba(239,83,80,0.10)", line_width=0, layer="below")
        fig_pe.add_hline(
            y=_q_lo,
            line=dict(color="rgba(38,166,154,0.8)", width=1.2, dash="dash"),
            annotation_text=f"P{pe_pct} = {_q_lo:.4f}",
            annotation_font_color="rgba(38,166,154,1)", annotation_font_size=10,
        )
        fig_pe.add_hline(
            y=_q_hi,
            line=dict(color="rgba(255,152,0,0.8)", width=1.2, dash="dash"),
            annotation_text=f"P{pe_max_pct} = {_q_hi:.4f}",
            annotation_font_color="rgba(255,152,0,1)", annotation_font_size=10,
        )

    fig_pe.add_trace(go.Scatter(
        x=_px, y=_py,
        mode="lines",
        line=dict(color="rgba(100,181,246,0.9)", width=1),
        showlegend=False,
    ))

    if "fc_origin_k" in st.session_state:
        _ok = st.session_state["fc_origin_k"]
        if 1 <= _ok < len(valid):
            fig_pe.add_vline(
                x=valid["begin"].iloc[_ok].timestamp() * 1000,
                line=dict(color="rgba(255,214,0,0.5)", width=1.5, dash="dot"),
            )

    fig_pe.update_layout(
        **_layout,
        yaxis_title="PE",
        height=160,
        showlegend=False,
        yaxis=dict(range=[_pe_min - 0.002, _pe_max_v + 0.002], **_grid),
    )
    fig_pe.update_xaxes(rangebreaks=rangebreaks, **_grid)
    st.plotly_chart(fig_pe, use_container_width=True, config={"scrollZoom": True})

# ── chart 3: volume ───────────────────────────────────────────────────────────

vol_colors = np.where(view["close"] >= view["open"], "#26a69a", "#ef5350")
fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(
    x=view["begin"], y=view["volume"],
    marker_color=vol_colors,
    showlegend=False,
))
fig_vol.update_layout(
    **_layout, yaxis_title="Объём", height=140,
    bargap=0.1, showlegend=False,
)
fig_vol.update_xaxes(rangebreaks=rangebreaks, **_grid)
fig_vol.update_yaxes(**_grid)
st.plotly_chart(fig_vol, use_container_width=True, config={"scrollZoom": True})

# ── chart 3: ratio ────────────────────────────────────────────────────────────

fig2 = go.Figure()
fig2.add_hline(y=1.0, line=dict(color="rgba(128,128,128,0.4)", width=1, dash="dot"))
fig2.add_trace(go.Scatter(
    x=view["begin"], y=view["ratio"],
    line=dict(color="#7c4dff", width=1), showlegend=False,
))
fig2.update_layout(
    **_layout, yaxis_title=("close / logtrend" if _use_logtrend else f"close / SMA({ma_window})"),
    xaxis_title="Дата", height=200, showlegend=False,
)
fig2.update_xaxes(rangebreaks=rangebreaks, **_grid)
fig2.update_yaxes(**_grid)
st.plotly_chart(fig2, use_container_width=True, config={"scrollZoom": True})

st.caption(
    f"{ticker} [{interval}] — {len(view):,} / {len(valid):,} свечей  |  "
    f"история: {valid['begin'].iloc[0]:%d.%m.%Y} — {last_dt:%d.%m.%Y}"
)

# ── авто p: результаты валидации ──────────────────────────────────────────────

if _is_auto_p and "fc_ap_best" in st.session_state and st.session_state.get("fc_ap_model") == model:
    _ap_best     = st.session_state["fc_ap_best"]
    _ap_best_pca = st.session_state.get("fc_ap_best_pca_k", 0)
    _ap_mape     = st.session_state["fc_ap_val_mape"]
    _ap_scores   = st.session_state["fc_ap_scores"]   # {(p, pca_k): mape}
    _is_fb_win   = (_ap_best_pca == -1)
    if _is_fb_win:
        _pca_label = "filter bank"
        _xi_label  = str(fb_xi)
    else:
        _pca_label = f"pca_k={_ap_best_pca}" if _ap_best_pca > 0 else "без PCA"
        _xi_label  = str(3 * (_ap_best + 1))
    st.caption(
        f"Авто p: победитель **p={_ap_best}, {_pca_label}**  |  "
        f"MAPE на валидации: **{_ap_mape * 100:.2f}%**  |  "
        f"Ξ = {_xi_label}"
    )
    if _ap_scores:
        # aggregate: best mape per p (min over pca_k), track winning pca_k
        _best_per_p:    dict[int, float] = {}
        _best_pca_per_p: dict[int, int]  = {}
        for (_p, _k), _m in _ap_scores.items():
            if _p not in _best_per_p or _m < _best_per_p[_p]:
                _best_per_p[_p]     = _m
                _best_pca_per_p[_p] = _k
        _ps    = sorted(_best_per_p)
        _mapes = [_best_per_p[_p] * 100 for _p in _ps]
        _clrs  = ["#ffd600" if _p == _ap_best else "#7c4dff" for _p in _ps]
        def _bar_label(p_val: int, pca_val: int) -> str:
            if pca_val == -1:
                return "fb"
            return f"k={pca_val}" if pca_val > 0 else "—"
        _texts = [_bar_label(_p, _best_pca_per_p[_p]) for _p in _ps]
        fig_ap = go.Figure(go.Bar(
            x=_ps, y=_mapes, marker_color=_clrs,
            text=_texts, textposition="outside", textfont=dict(size=9),
        ))
        fig_ap.update_layout(
            **_layout, height=200,
            xaxis_title="p", yaxis_title="MAPE валидации, %",
            showlegend=False, bargap=0.15,
        )
        fig_ap.update_xaxes(**_grid)
        fig_ap.update_yaxes(**_grid)
        st.plotly_chart(fig_ap, use_container_width=True)

# ── ACF expander ──────────────────────────────────────────────────────────────

with st.expander("Анализ параметров пространства задержек", expanded=False):
    ratio_arr  = valid["ratio"].values
    dratio_arr = np.diff(ratio_arr)
    n_valid    = len(ratio_arr)
    p_safe, p_border = recommended_p(n_valid)

    ml_ratio = min(500, n_valid // 4)
    ml_diff  = min(200, len(dratio_arr) // 4)

    acf_ratio, tau_ratio = _acf(ratio_arr.tobytes(), ml_ratio)
    acf_diff,  tau_diff  = _acf(dratio_arr.tobytes(), ml_diff)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("τ (ratio)", f"{tau_ratio} баров", help="Первый ноль АКФ ratio — I(1)")
    mc2.metric("τ (Δratio)", f"{tau_diff} баров",  help="Первый ноль АКФ нормализованной доходности")
    mc3.metric("p надёжный / предел", f"{p_safe} / {p_border}")
    mc4.metric("Соседей LA1, p=5", f"≥ {3*6}")

    st.info(
        f"**ratio** — I(1) процесс, первый ноль АКФ на лаге {tau_ratio}. "
        f"**Δratio** почти некоррелирован (τ={tau_diff}) — нелинейные зависимости "
        "могут присутствовать даже при нулевой линейной АКФ."
    )

    def _acf_fig(vals, tau, title, color):
        lags = np.arange(len(vals))
        fig  = go.Figure()
        fig.add_hline(y=0, line=dict(color="rgba(128,128,128,0.4)", width=1))
        fig.add_vline(x=min(tau, len(vals)-1),
                      line=dict(color="#ff9800", width=2, dash="dash"),
                      annotation_text=f" τ={tau}", annotation_font_color="#ff9800")
        fig.add_trace(go.Bar(x=lags, y=vals,
                             marker_color=[color if v >= 0 else "#ef5350" for v in vals]))
        fig.update_layout(**_layout, title=title, xaxis_title="Лаг",
                          yaxis_title="АКФ", height=240, showlegend=False, bargap=0.05)
        fig.update_yaxes(**_grid)
        return fig

    col_l, col_r = st.columns(2)
    col_l.plotly_chart(_acf_fig(acf_ratio[:201], min(tau_ratio, 200),
                                "АКФ(ratio) [первые 200]", "#7c4dff"),
                       use_container_width=True)
    col_r.plotly_chart(_acf_fig(acf_diff, tau_diff,
                                "АКФ(Δratio)", "#26a69a"),
                       use_container_width=True)
    st.caption(f"N = {n_valid:,} баров. Рекомендуемый диапазон p: 3–{p_border}.")

# ── Спектрограмма ─────────────────────────────────────────────────────────────

with st.expander("Спектрограмма Δratio", expanded=False):
    from scipy.signal import spectrogram as _scipy_spectrogram  # noqa: PLC0415

    _sp_dmin = valid["begin"].dt.date.min()
    _sp_dmax = valid["begin"].dt.date.max()
    _sp_def_start = max(
        _sp_dmin,
        (valid["begin"].iloc[-1] - pd.Timedelta(days=365)).date(),
    )

    _spc1, _spc2, _spc3 = st.columns([2, 2, 1])
    _sp_start = _spc1.date_input(
        "Начало", value=_sp_def_start,
        min_value=_sp_dmin, max_value=_sp_dmax, key="sp_start",
    )
    _sp_end = _spc2.date_input(
        "Конец", value=_sp_dmax,
        min_value=_sp_dmin, max_value=_sp_dmax, key="sp_end",
    )
    _sp_log_y = _spc3.checkbox("Лог. ось Y", value=False, key="sp_log_y")

    _sp_mask = (
        (valid["begin"].dt.date >= _sp_start) &
        (valid["begin"].dt.date <= _sp_end)
    )
    _sp_df = valid[_sp_mask].reset_index(drop=True)
    _sp_n  = len(_sp_df) - 1  # длина dratio

    if _sp_start >= _sp_end:
        st.warning("Начало периода должно быть раньше конца.")
    elif _sp_n < 32:
        st.warning(f"Слишком короткий период ({_sp_n} баров Δratio). Нужно минимум 32.")
    else:
        _sp_dratio = np.diff(_sp_df["ratio"].values)

        _spc4, _spc5 = st.columns(2)
        _sp_win_options = [w for w in [4, 8, 16, 32, 64, 128, 256, 512] if w <= _sp_n // 2]
        if not _sp_win_options:
            _sp_win_options = [4]
        if len(_sp_win_options) > 1:
            _sp_nperseg = _spc4.select_slider(
                "Окно STFT (баров)", options=_sp_win_options,
                value=min(64, _sp_win_options[-1]), key="sp_nperseg",
            )
        else:
            _sp_nperseg = _sp_win_options[0]
            _spc4.metric("Окно STFT", f"{_sp_nperseg} баров")
        _sp_overlap_pct = _spc5.slider(
            "Перекрытие (%)", 50, 95, 75, step=5, key="sp_overlap",
        )
        _sp_noverlap = int(_sp_nperseg * _sp_overlap_pct / 100)

        # Диапазон отображаемых частот (fmin, fmax)
        _SP_FMIN_OPTS = {
            "0 (все)":       0.0,
            "C4/C5 T≈128б":  0.0078125,
            "C3/C4 T≈64б":   0.015625,
            "C2/C3 T≈32б":   0.03125,
            "C1/C2 T≈16б":   0.0625,
            "C0/C1 T≈8б":    0.125,
        }
        _SP_FMAX_OPTS = {
            "0.5 Nyq (все)": 0.5,
            "C0/C1 T≈8б":    0.125,
            "C1/C2 T≈16б":   0.0625,
            "C2/C3 T≈32б":   0.03125,
            "C3/C4 T≈64б":   0.015625,
            "C4/C5 T≈128б":  0.0078125,
        }
        _spc6, _spc7 = st.columns(2)
        _sp_fmin = _SP_FMIN_OPTS[_spc6.selectbox(
            "Мин. частота (низ оси)", list(_SP_FMIN_OPTS.keys()), index=0, key="sp_fmin",
        )]
        _sp_fmax = _SP_FMAX_OPTS[_spc7.selectbox(
            "Макс. частота (верх оси)", list(_SP_FMAX_OPTS.keys()), index=0, key="sp_fmax",
        )]

        _f_spec, _t_spec, _Sxx = _scipy_spectrogram(
            _sp_dratio, fs=1.0,
            nperseg=_sp_nperseg, noverlap=_sp_noverlap,
            scaling="density", window="hann",
        )
        _Sxx_db = 10 * np.log10(_Sxx + 1e-20)

        # Привязка временны́х отсчётов STFT к реальным датам
        _sp_dates_arr = _sp_df["begin"].values[1:]  # даты для dratio
        _t_idx = np.round(_t_spec).astype(int).clip(0, _sp_n - 1)
        _t_labels = (
            pd.DatetimeIndex(_sp_dates_arr[_t_idx]).strftime("%Y-%m-%d").tolist()
        )

        # Реальные частоты среза (Гц, для fs=1.0):
        # butter(order, Wn) без fs → fc = Wn * (Nyquist=0.5) = Wn/2
        _CUTOFFS_HZ     = [wn / 2 for wn in [0.25, 0.125, 0.0625, 0.03125, 0.015625]]
        _CUTOFFS_LABELS = ["C0/C1", "C1/C2", "C2/C3", "C3/C4", "C4/C5"]
        _CUTOFFS_PERIODS = [round(1.0 / fc) for fc in _CUTOFFS_HZ]

        # Обрезаем по выбранному диапазону [fmin, fmax]
        if _sp_fmin >= _sp_fmax:
            st.warning("Мин. частота должна быть меньше макс. частоты.")
            _sp_fmin = 0.0
        _sp_fmask = (_f_spec >= _sp_fmin) & (_f_spec <= _sp_fmax)
        _f_plot   = _f_spec[_sp_fmask]
        _Sxx_plot = _Sxx_db[_sp_fmask, :]

        _zmin = float(np.percentile(_Sxx_plot, 5))
        _zmax = float(np.percentile(_Sxx_plot, 95))

        _fig_sp = go.Figure(go.Heatmap(
            z=_Sxx_plot,
            x=_t_labels,
            y=_f_plot.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="дБ/Гц", thickness=12, len=0.8),
            zmin=_zmin, zmax=_zmax,
            hoverongaps=False,
        ))

        for _fc_hz, _lbl, _per in zip(
            _CUTOFFS_HZ, _CUTOFFS_LABELS, _CUTOFFS_PERIODS
        ):
            if _sp_fmin <= _fc_hz <= _sp_fmax and len(_f_plot) > 0:
                _fig_sp.add_hline(
                    y=_fc_hz,
                    line=dict(color="rgba(255,80,80,0.75)", width=1.5, dash="dot"),
                    annotation_text=f" {_lbl} T≈{_per}б",
                    annotation_font_color="rgba(255,140,140,1.0)",
                    annotation_font_size=9,
                    annotation_position="right",
                )

        _sp_f_hi = float(_f_plot[-1]) if len(_f_plot) > 0 else _sp_fmax
        _sp_f_lo = float(_f_plot[0])  if len(_f_plot) > 0 else _sp_fmin
        _sp_yaxis: dict = dict(**_grid, title="Частота (цикл/бар)")
        if _sp_log_y:
            _sp_log_lo = max(_sp_f_lo, 1e-4)
            _sp_yaxis.update(type="log", range=[np.log10(_sp_log_lo), np.log10(_sp_f_hi)])
        else:
            _sp_yaxis.update(range=[_sp_f_lo, _sp_f_hi])

        _fig_sp.update_layout(
            **_layout, height=400,
            title=dict(
                text=f"Спектрограмма Δratio — {ticker} [{interval}]",
                font=dict(size=12),
            ),
            xaxis=dict(**_grid, title="Дата"),
            yaxis=_sp_yaxis,
        )
        st.plotly_chart(_fig_sp, use_container_width=True, config={"scrollZoom": True})

        _freq_res  = 1.0 / _sp_nperseg
        _time_step = _sp_nperseg - _sp_noverlap
        st.caption(
            f"N={_sp_n:,} баров · окно={_sp_nperseg} бар · "
            f"перекрытие={_sp_noverlap} · "
            f"Δf={_freq_res:.4f} цикл/бар · шаг по времени={_time_step} баров"
        )
