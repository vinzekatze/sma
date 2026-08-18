"""
Streamlit prototype: MOEX LWR + Filter Bank forecaster.

Модель: logtrend-нормализация + causal Butterworth filter bank + LWR на C3–C5.
Опция: Damped AR(BIC) для C1+C2 (скр.52: γ_C1=0.5, γ_C2=0.8 → −8% MAPE).

Run from prototype root:  streamlit run forcaster/ui/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forcaster.data.moex import download_candles, save_candles, INTERVALS
from forcaster.forecast.filterbank import (
    make_filter_bank, forecast_fb, forecast_fb_damped, forecast_fb_merged_slow,
    forecast_lp_attractor, forecast_lp_auto_p,
    forecast_smooth_lwr, forecast_smooth_auto_p,
    FB_P, FB_XI,
)

_ROOT    = Path(__file__).parent.parent.parent
DATA_DIR = _ROOT / "data" / "candles"

_BAR_DELTA: dict[str, pd.Timedelta] = {
    "1m":  pd.Timedelta(minutes=1),
    "10m": pd.Timedelta(minutes=10),
    "1h":  pd.Timedelta(hours=1),
    "1d":  pd.Timedelta(days=1),
    "1w":  pd.Timedelta(weeks=1),
    "1mo": pd.Timedelta(days=30),
}

# ── cached helpers ────────────────────────────────────────────────────────────

def _logtrend_causal(close: np.ndarray) -> np.ndarray:
    """Causal log-linear OLS trend (no warmup, no window)."""
    n   = len(close)
    lc  = np.log(np.maximum(close, 1e-10))
    t   = np.arange(n, dtype=np.float64)
    cn  = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t)
    trend[:2] = close[:2]
    return trend


@st.cache_data(show_spinner=False)
def _normalize_logtrend(candles_key: str) -> pd.DataFrame:
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
def _run_forecast_fb(
    dratio_bytes: bytes,
    origin_k: int,
    horizon: int,
    p: int,
    xi: int,
    component_idx: tuple[int, ...],
) -> np.ndarray:
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    return forecast_fb(dratio, origin_k, horizon,
                       p=p, n_neighbors=xi,
                       use_lwr=True,
                       component_idx=list(component_idx))


@st.cache_data(show_spinner=False)
def _run_forecast_fb_damped(
    dratio_bytes: bytes,
    origin_k: int,
    horizon: int,
    p: int,
    xi: int,
    gamma_c1: float,
    gamma_c2: float,
    slow_components: tuple[int, ...],
) -> np.ndarray:
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    return forecast_fb_damped(dratio, origin_k, horizon,
                               p_slow=p, xi_slow=xi,
                               gamma_c1=gamma_c1, gamma_c2=gamma_c2,
                               slow_components=list(slow_components))


@st.cache_data(show_spinner=False)
def _slow_trend_history(
    dratio_bytes: bytes,
    ratio_bytes: bytes,
    ma_bytes: bytes,
    origin_k: int,
    n_display: int,
    component_idx: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Медленный тренд через историю для визуализации фильтрбанка.
    Возвращает (bar_indices, slow_prices).
    Якорится на actual ratio в начале окна → линия следует за ценой,
    но без быстрого шума.
    """
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    ratio  = np.frombuffer(ratio_bytes,  dtype=np.float64).copy()
    ma     = np.frombuffer(ma_bytes,     dtype=np.float64).copy()

    comp = make_filter_bank(dratio[:origin_k])
    slow_dratio = np.zeros(origin_k)
    for ci in component_idx:
        if 0 <= ci < comp.shape[0]:
            slow_dratio += comp[ci]

    anchor_idx      = max(0, origin_k - n_display)
    slow_ratio_0    = float(ratio[anchor_idx])
    slow_dratio_seg = slow_dratio[anchor_idx:origin_k]
    slow_ratio_seg  = slow_ratio_0 + np.concatenate([[0.0], np.cumsum(slow_dratio_seg)])
    n               = min(len(slow_ratio_seg), len(ma) - anchor_idx)
    slow_prices     = slow_ratio_seg[:n] * ma[anchor_idx: anchor_idx + n]
    bar_indices     = np.arange(anchor_idx, anchor_idx + n, dtype=int)
    return bar_indices, slow_prices


@st.cache_data(show_spinner=False)
def _slow_trend_forward(
    dratio_full_bytes: bytes,
    junction_ratio: float,
    ma_seg_bytes: bytes,
    origin_k: int,
    n_future: int,
    component_idx: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Фактический медленный тренд вперёд (если есть реальные данные за origin).
    dratio_full_bytes: np.diff(_working_ratio[:origin_k + 1 + n_future])
    ma_seg_bytes:      _ma_vals[origin_k : origin_k + 1 + n_future]
    Возвращает (bar_indices, slow_prices) начиная с origin_k.
    """
    dratio_full = np.frombuffer(dratio_full_bytes, dtype=np.float64).copy()
    ma_seg      = np.frombuffer(ma_seg_bytes,      dtype=np.float64).copy()

    comp        = make_filter_bank(dratio_full)
    slow_dratio = np.zeros(len(dratio_full))
    for ci in component_idx:
        if 0 <= ci < comp.shape[0]:
            slow_dratio += comp[ci]

    fwd_incr       = slow_dratio[origin_k: origin_k + n_future]
    slow_ratio_seg = junction_ratio + np.concatenate([[0.0], np.cumsum(fwd_incr)])
    n_out          = min(len(slow_ratio_seg), len(ma_seg))
    slow_prices    = slow_ratio_seg[:n_out] * ma_seg[:n_out]
    bar_indices    = np.arange(origin_k, origin_k + n_out, dtype=int)
    return bar_indices, slow_prices


@st.cache_data(show_spinner=False)
def _compute_c0_sigma(dratio_bytes: bytes, sigma_window: int) -> float:
    """σ_C0 — std of the C0 component over the last sigma_window bars."""
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    comp   = make_filter_bank(dratio)
    c0     = comp[0]
    win_st = max(0, len(c0) - sigma_window)
    return float(np.std(c0[win_st:], ddof=1))


@st.cache_data(show_spinner=False)
def _run_forecast_merged_slow(
    dratio_bytes: bytes,
    origin_k: int,
    horizon: int,
    p: int,
    xi: int,
    damped_c1: bool,
    gamma_c1: float,
) -> np.ndarray:
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    return forecast_fb_merged_slow(
        dratio, origin_k, horizon,
        p=p, xi=xi,
        merged_components=[2, 3, 4, 5],
        damped_c1=damped_c1,
        gamma_c1=gamma_c1,
    )


@st.cache_data(show_spinner=False)
def _run_forecast_lp_auto_p(
    dratio_bytes: bytes,
    origin_k: int,
    horizon: int,
    wn: float,
    tau: int,
    p_candidates: tuple[int, ...],
    top_n: int,
) -> list[tuple[int, float, np.ndarray]]:
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    return forecast_lp_auto_p(dratio, origin_k, horizon, wn=wn, tau=tau,
                               p_candidates=list(p_candidates), top_n=top_n)


@st.cache_data(show_spinner=False)
def _run_ma_lwr(
    dsig_bytes: bytes,
    origin_k: int,
    horizon: int,
    p: int,
    xi: int,
) -> np.ndarray:
    sig = np.frombuffer(dsig_bytes, dtype=np.float64).copy()
    return forecast_smooth_lwr(sig, origin_k, horizon, p=p, xi=xi)


@st.cache_data(show_spinner=False)
def _run_ma_auto_p(
    dsig_bytes: bytes,
    origin_k: int,
    horizon: int,
    tau: int,
    p_candidates: tuple[int, ...],
    top_n: int,
) -> list[tuple[int, float, np.ndarray]]:
    sig = np.frombuffer(dsig_bytes, dtype=np.float64).copy()
    return forecast_smooth_auto_p(sig, origin_k, horizon, tau=tau,
                                   p_candidates=list(p_candidates), top_n=top_n)


@st.cache_data(show_spinner=False)
def _run_forecast_lp_att(
    dratio_bytes: bytes,
    origin_k: int,
    horizon: int,
    p: int,
    xi: int,
    wn: float,
) -> np.ndarray:
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    return forecast_lp_attractor(dratio, origin_k, horizon, p=p, xi=xi, wn=wn)


@st.cache_data(show_spinner=False)
def _run_conformal_calibration(
    ratio_bytes: bytes,
    n_cal: int,
    val_h: int,
    fb_p: int,
    fb_xi: int,
    fb_damped_ar: bool,
    gamma_c1: float,
    gamma_c2: float,
    slow_components: tuple[int, ...],
    smooth_ma_w: int = 1,
    merged_slow: bool = False,
    merged_damped_c1: bool = True,
    merged_gamma_c1: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward калибровка: n_cal непересекающихся окон по val_h баров назад от origin.
    ratio_bytes — сырой ratio (для сравнения с фактическими ценами).
    smooth_ma_w — окно быстрой MA для сглаживания перед diff (1 = без сглаживания).
    Возвращает:
        cal_errors   (n_cal, val_h) — |ratio_hat[h] - ratio_actual[h]| (в ratio)
        cal_valmapes (n_cal,)       — MAPE каждого окна (vs raw actual)
    """
    raw_ratio = np.frombuffer(ratio_bytes, dtype=np.float64).copy()
    # Сглаживание: модель обучается на smooth ratio, ошибки считаются vs raw actual
    ratio = (pd.Series(raw_ratio).rolling(smooth_ma_w, min_periods=1)
               .mean().values.astype(np.float64)
             if smooth_ma_w > 1 else raw_ratio)
    origin_k = len(ratio) - 1
    min_hist = max(200, fb_xi + fb_p + 5)

    cal_errors   = np.full((n_cal, val_h), np.nan)
    cal_valmapes = np.full(n_cal, np.nan)

    for i in range(1, n_cal + 1):
        cal_origin = origin_k - i * val_h
        if cal_origin < min_hist:
            break
        dratio_i = np.diff(ratio[: cal_origin + 1])
        if merged_slow:
            dhat = forecast_fb_merged_slow(
                dratio_i, cal_origin, val_h,
                p=fb_p, xi=fb_xi,
                merged_components=[2, 3, 4, 5],
                damped_c1=merged_damped_c1,
                gamma_c1=merged_gamma_c1,
            )
        elif fb_damped_ar:
            dhat = forecast_fb_damped(
                dratio_i, cal_origin, val_h,
                p_slow=fb_p, xi_slow=fb_xi,
                gamma_c1=gamma_c1, gamma_c2=gamma_c2,
                slow_components=list(slow_components),
            )
        else:
            dhat = forecast_fb(
                dratio_i, cal_origin, val_h,
                p=fb_p, n_neighbors=fb_xi,
                use_lwr=True,
                component_idx=list(slow_components),
            )
        ratio0    = float(ratio[cal_origin])
        ratio_hat = ratio0 + np.cumsum(dhat)
        actual_r  = raw_ratio[cal_origin + 1: cal_origin + 1 + val_h]
        n_cmp     = min(len(ratio_hat), len(actual_r))
        if n_cmp < 1:
            continue
        errs = np.abs(ratio_hat[:n_cmp] - actual_r[:n_cmp])
        cal_errors[i - 1, :n_cmp] = errs
        cal_valmapes[i - 1]       = float(
            np.mean(errs / (np.abs(actual_r[:n_cmp]) + 1e-10))
        )
    return cal_errors, cal_valmapes


def _compute_lwc_band(
    cal_errors: np.ndarray,      # (n_cal, val_h)
    cal_valmapes: np.ndarray,    # (n_cal,)
    vm_test: float,
    h_kernel: float,
    alpha: float,
    forecast_price: np.ndarray,  # (n_fc,)
    ma_fwd: np.ndarray,          # (≥ n_fc,)
    val_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Locally Weighted Conformal prediction band (в единицах цены)."""
    valid = ~np.isnan(cal_valmapes)
    if valid.sum() < 3:
        nan = np.full(len(forecast_price), np.nan)
        return nan, nan.copy()
    err_v = cal_errors[valid]
    vm_v  = cal_valmapes[valid]
    log_r = np.log(np.maximum(vm_v, 1e-10) / max(vm_test, 1e-10))
    w     = np.exp(-log_r ** 2 / (2 * h_kernel ** 2))

    n_fc    = len(forecast_price)
    upper_p = np.empty(n_fc)
    lower_p = np.empty(n_fc)

    for h_idx in range(n_fc):
        h_cal = min(h_idx, val_h - 1)      # глубина калибровки ограничена val_h
        s     = err_v[:, h_cal]
        good  = ~np.isnan(s)
        if good.sum() < 2:
            upper_p[h_idx] = lower_p[h_idx] = np.nan
            continue
        s_g, w_g = s[good], w[good]
        # Conformal augmentation (Vovk 2005): добавляем виртуальную точку score=∞
        w_aug = np.append(w_g, float(np.mean(w_g)))
        s_aug = np.append(s_g, np.inf)
        idx_s = np.argsort(s_aug)
        w_s   = w_aug[idx_s] / w_aug[idx_s].sum()
        j     = int(np.searchsorted(np.cumsum(w_s), 1.0 - alpha))
        q     = float(s_aug[idx_s[min(j, len(idx_s) - 1)]])
        if not np.isfinite(q):
            q = float(np.nanmax(s_g))
        ma_h           = float(ma_fwd[min(h_idx, len(ma_fwd) - 1)])
        upper_p[h_idx] = forecast_price[h_idx] + q * ma_h
        lower_p[h_idx] = forecast_price[h_idx] - q * ma_h

    return upper_p, lower_p


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="MOEX Forecaster", layout="wide",
                   initial_sidebar_state="expanded")

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Параметры")

    # --- инструмент ---
    ticker   = st.text_input("Тикер", value="SBER", key="ticker_val").upper().strip()
    interval = st.selectbox("Интервал", list(INTERVALS),
                             index=list(INTERVALS).index("1d"), key="interval_val")

    st.divider()
    st.subheader("Прогноз")

    horizon = st.slider("Горизонт (баров)", 1, 200, 30, 1, key="horizon")
    val_h   = horizon   # окно валидации = горизонт прогноза

    st.divider()
    st.subheader("Предобработка ряда")
    _use_sm = st.checkbox("Быстрая MA ratio перед Δ", value=False, key="use_smooth",
                           help="ratio → MA(w) → diff → filter bank. "
                                "Снижает дисперсию C0, компоненты чище.")
    smooth_ma_w = (st.slider("Окно MA", 2, 20, 5, 1, key="smooth_ma_w") if _use_sm else 1)

    st.divider()
    st.subheader("Filter Bank")

    fb_p  = st.slider("p (LWR)", 2, 300, FB_P, 1, key="fb_p",
                      help="Порядок вложения для LWR на медленных компонентах. "
                           "Правило: p ≥ T_min/4, T_min≈52 для C3 → p_min≈13. "
                           "Стандарт исследования: p=20.")
    fb_xi = 3 * (fb_p + 1)
    st.caption(f"ξ = 3·(p+1) = {fb_xi}")

    _COMP_LABELS = {3: "C3 ~16–52б", 4: "C4 ~52–103б", 5: "C5 103+б"}
    fb_components = tuple(sorted(st.multiselect(
        "Медленные компоненты",
        options=[3, 4, 5],
        default=[3, 4, 5],
        format_func=lambda x: _COMP_LABELS[x],
        key="fb_components",
    ))) or (3, 4, 5)

    merged_slow = st.checkbox(
        "Merged slow: C2+C3–C5 → единый LWR",
        value=False, key="merged_slow",
        help="Объединяет C2+C3+C4+C5 в один сигнал перед LWR, не разбивая аттрактор "
             "на компоненты. C1 — Damped AR ниже, C0 — полоса неопределённости. "
             "Мотивация: сохраняет межкомпонентную динамику, важную для движения цены.",
    )
    if merged_slow:
        st.caption(
            "Режим: **C2+C3+C4+C5 → merged LWR**  |  C1 → Damped AR  |  C0 → полоса"
        )

    st.divider()
    st.subheader("Damped AR" + (" для C1" if merged_slow else " для C1+C2"))
    if merged_slow:
        st.caption(
            "При merged slow C2 прогнозируется совместно с C3–C5 через LWR. "
            "C1 (4–8б) прогнозируется AR(BIC) × γ^h."
        )
        fb_damped_ar    = False   # не используется при merged_slow
        merged_damped_c1 = st.checkbox("Damped AR для C1", value=True,
                                        key="merged_damped_c1",
                                        help="AR(BIC) × γ_C1^h для C1 (~4–8б). "
                                             "Оптимум γ=0.5 (скр.52).")
        merged_gamma_c1 = st.slider("γ C1", 0.10, 1.0, 0.50, 0.05,
                                     key="merged_gamma_c1",
                                     disabled=not merged_damped_c1,
                                     help="Decay C1. Оптимум=0.5 (скр.52).")
        gamma_c1 = merged_gamma_c1
        gamma_c2 = 0.8   # не используется
    else:
        st.caption(
            "C1 (8–16б) и C2 (16–32б) прогнозируются AR(BIC) × γ^h. "
            "C0 — ноль. Исслед. 52: γ_C1=0.5, γ_C2=0.8 → **−8.01% MAPE**."
        )
        fb_damped_ar = st.checkbox("Включить Damped AR", value=False, key="fb_damped_ar")
        if fb_damped_ar:
            _g1, _g2 = st.columns(2)
            gamma_c1 = _g1.slider("γ C1", 0.10, 1.0, 0.50, 0.05, key="fb_gamma_c1",
                                   help="Decay C1 (~8–16б). Оптимум=0.5.")
            gamma_c2 = _g2.slider("γ C2", 0.10, 1.0, 0.80, 0.05, key="fb_gamma_c2",
                                   help="Decay C2 (~16–32б). Оптимум=0.8.")
        else:
            gamma_c1 = 0.5
            gamma_c2 = 0.8
        merged_damped_c1 = False
        merged_gamma_c1  = 0.5

    st.divider()
    st.subheader("Полоса неопределённости C0")
    st.caption(
        "C0 — непрогнозируемый шум; его вклад в цену после h шагов ~ σ_C0·√h. "
        "Скр.53 (4 тикера, 200 origins): k=0.90→68%, **k=1.25→80%**, k=1.65→90%."
    )
    show_band    = st.checkbox("Показать полосу ±k·σ_C0·√h", value=True, key="show_band")
    if show_band:
        _b1, _b2     = st.columns(2)
        k_band       = _b1.slider("k", 0.5, 3.0, 1.25, 0.05, key="k_band",
                                   help="Скр.53: k=1.25 → 80% покрытие.")
        sigma_win_c0 = _b2.slider("σ окно (баров)", 20, 300, 100, 10, key="sigma_win_c0",
                                   help="Число баров для оценки локальной σ_C0.")
    else:
        k_band       = 1.25
        sigma_win_c0 = 100

    st.divider()
    st.subheader("LWC Conformal Band")
    st.caption(
        "Walk-forward на N окнах по horizon баров: локальная калибровка на текущем тикере. "
        "Веса по сходству val_mape (Locally Weighted Conformal, фаза 1 скр.05). "
        "Покрывает модельную ошибку **+** C0 шум."
    )
    show_lwc = st.checkbox("Показать LWC-полосу", value=True, key="show_lwc")
    if show_lwc:
        _l1, _l2, _l3   = st.columns(3)
        n_cal            = _l1.slider("N окон", 5, 100, 15, 1, key="n_cal",
                                       help="Число walk-forward шагов назад.")
        _lwc_cov         = _l2.select_slider("Покрытие", options=[68, 80, 90], value=80,
                                              key="lwc_cov", format_func=lambda x: f"{x}%")
        lwc_h_kernel     = _l3.slider("h ядра", 0.3, 3.0, 1.0, 0.1, key="lwc_h",
                                       help="Ширина ядра в log-val_mape. h=1.0 — оптимум скр.05.")
        lwc_alpha        = (100 - _lwc_cov) / 100
    else:
        n_cal, lwc_alpha, lwc_h_kernel = 15, 0.20, 1.0

    st.divider()
    st.subheader("LP аттрактор + LWR")
    st.caption(
        "LP(Wn) на dratio → LWR. Прогноз от val_origin на val_h+horizon шагов. "
        "Якорь и валидация — по MA цен."
    )
    lp_att_mode = st.checkbox("Включить", value=False, key="lp_att_mode",
                               help="Damped AR и merged_slow игнорируются.")
    if lp_att_mode:
        _LP_WN_OPTS   = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5]
        _LP_WN_LABELS = {
            0.015625: "0.016  T≈128б  (C4/C5)",
            0.03125:  "0.031  T≈64б   (C3/C4)",
            0.0625:   "0.063  T≈32б   (C2/C3)",
            0.125:    "0.125  T≈16б   (C1/C2) ★",
            0.25:     "0.25   T≈8б    (C0/C1)",
            0.5:      "0.5    Nyquist",
        }
        lp_wn = st.select_slider(
            "Частота среза Wn", options=_LP_WN_OPTS, value=0.125,
            format_func=lambda w: _LP_WN_LABELS[w], key="lp_wn",
        )
        lp_ma_window = st.slider(
            "Окно MA (сравнение)", 2, 300, 50, 1, key="lp_ma_window",
            help="MA цен закрытия: якорь прогноза = MA[val_origin], "
                 "val_mape считается vs MA в окне валидации.",
        )
        tau_shift = st.slider(
            "Сдвиг по фазе τ (баров)", 0, 50, 0, 1, key="tau_shift",
            help="Визуальная компенсация задержки LP фильтра. "
                 "LP(0.125) order=4 ≈ 7 баров. Также окно τ-валидации авто-p.",
        )
        st.divider()
        lp_auto_p = st.checkbox("Авто-p (τ-валидация)", value=False, key="lp_auto_p",
                                  help="Перебирает p, выбирает лучший по MSE на τ барах LP-сигнала.")
        if lp_auto_p:
            lp_top_n = st.slider("Top-N кандидатов", 1, 8, 3, 1, key="lp_top_n")
            _p_range = st.slider("p диапазон", 2, 100, (4, 30), 1, key="lp_p_range")
            lp_p_cands = tuple(range(_p_range[0], _p_range[1] + 1))
        else:
            lp_top_n   = 1
            lp_p_cands = (fb_p,)
    else:
        lp_wn        = 0.125
        lp_ma_window = 50
        tau_shift    = 0
        lp_auto_p    = False
        lp_top_n     = 1
        lp_p_cands   = (fb_p,)

# ── data load ─────────────────────────────────────────────────────────────────

cache_path = DATA_DIR / ticker / f"{interval}.json"

c1, c2, c3 = st.columns([3, 1, 1])
load_btn  = c2.button("Загрузить",    type="primary", use_container_width=True)
reset_btn = c3.button("Сбросить кэш", use_container_width=True)

if reset_btn and cache_path.exists():
    cache_path.unlink()
    st.rerun()

raw_candles: list[dict] | None = None
if load_btn:
    if cache_path.exists():
        raw_candles = json.loads(cache_path.read_text(encoding="utf-8"))
        c1.info(f"Из кэша: {len(raw_candles):,} свечей")
    else:
        with st.spinner(f"Скачиваю {ticker} [{interval}] с MOEX…"):
            raw_candles = download_candles(ticker, interval, show_progress=False)
        if not raw_candles:
            st.error(f"Нет данных для **{ticker}**")
            st.stop()
        save_candles(raw_candles, cache_path)
        c1.success(f"Скачано: {len(raw_candles):,} свечей")
elif cache_path.exists():
    raw_candles = json.loads(cache_path.read_text(encoding="utf-8"))

if not raw_candles:
    st.info("Введите тикер и нажмите «Загрузить».")
    st.stop()

# ── normalise (logtrend) ──────────────────────────────────────────────────────

candles_key = cache_path.read_text(encoding="utf-8")
valid       = _normalize_logtrend(candles_key).dropna(subset=["ratio"]).reset_index(drop=True)

# ── period selector ───────────────────────────────────────────────────────────

PERIODS = {"1 мес": 30, "3 мес": 90, "6 мес": 180, "1 год": 365, "3 года": 1095, "Всё": None}
period  = st.radio("Период", list(PERIODS), index=2, horizontal=True)
days    = PERIODS[period]
last_dt = valid["begin"].iloc[-1]
view    = (valid[valid["begin"] >= last_dt - pd.Timedelta(days=days)].copy()
           if days else valid.copy())

# rangebreaks для пропусков (выходные / ночь)
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

# ── origin selector ───────────────────────────────────────────────────────────

_origin_max   = len(valid) - 1
_origin_min   = max(200, val_h + fb_xi + fb_p + 5)
_origin_default = _origin_max

# Clamp stored value to [_origin_min, _origin_max] before calling st.slider.
# Without this, when _origin_min increases (e.g. p grows), Streamlit resets
# origin_k to _origin_default (_origin_max) instead of clamping to min.
if "origin_k" in st.session_state:
    st.session_state["origin_k"] = int(
        np.clip(st.session_state["origin_k"], _origin_min, _origin_max)
    )

origin_k = st.slider(
    "Origin (последняя известная свеча)",
    min_value=_origin_min, max_value=_origin_max,
    value=_origin_default, step=1, key="origin_k",
)
origin_ts  = valid["begin"].iloc[origin_k]
ratio0     = float(valid["ratio"].iloc[origin_k])
ma0        = float(valid["ma"].iloc[origin_k])

# ── подготовка серий ──────────────────────────────────────────────────────────

ratio_arr      = valid["ratio"].values.astype(np.float64)
# Быстрое MA-сглаживание перед differencing (опционально)
_working_ratio = (pd.Series(ratio_arr).rolling(smooth_ma_w, min_periods=1)
                    .mean().values.astype(np.float64)
                  if smooth_ma_w > 1 else ratio_arr)

dratio     = np.diff(_working_ratio[:origin_k + 1])
dratio_key = dratio.tobytes()

ratio_bytes = ratio_arr[:origin_k + 1].tobytes()          # raw — для ошибок калибровки
_wk_bytes   = _working_ratio[:origin_k + 1].tobytes()     # smooth — для якоря тренда
ma_bytes    = valid["ma"].values[:origin_k + 1].tobytes()

# ── MA-сигнал для MA+LWR режима ───────────────────────────────────────────────

val_origin_k = max(0, origin_k - val_h)   # самый старый бар окна валидации

# MA цен для якоря и сравнения в LP att режиме
_close_vals  = valid["close"].values.astype(np.float64)
_ma_smooth   = (pd.Series(_close_vals)
                .rolling(lp_ma_window, min_periods=1).mean().values.astype(np.float64))
_ma_anchor   = float(_ma_smooth[val_origin_k])   # якорь прогноза = MA[val_origin_k]
_lp_total_h  = val_h + horizon                    # полная длина прогноза в LP att режиме

# форвардные MA-значения для реконструкции цены прогноза
_ma_vals = valid["ma"].values.astype(np.float64)
_ma_fwd  = _ma_vals[origin_k + 1: origin_k + 1 + horizon]
if len(_ma_fwd) < horizon:
    bar_delta = _BAR_DELTA.get(interval, pd.Timedelta(hours=1))
    _last_ma  = float(_ma_vals[min(origin_k + len(_ma_fwd), len(_ma_vals) - 1)])
    _ma_fwd   = np.concatenate([_ma_fwd, np.full(horizon - len(_ma_fwd), _last_ma)])

# ── медленный тренд (вычисляем ДО прогноза — нужен якорь) ───────────────────

# Собираем компоненты в историческую линию тренда:
#   merged_slow    → C2 входит в slow trend (прогнозируется LWR вместе с C3-C5)
#   damped AR      → C1+C2 (прогнозируются AR)
#   полоса C0      → C0 (история подтягивается к цене)
if merged_slow:
    _extra_comps = {2} | ({1} if merged_damped_c1 else set()) | ({0} if show_band else set())
else:
    _extra_comps = (({1, 2} if fb_damped_ar else set()) | ({0} if show_band else set()))
_trend_components = (tuple(sorted(_extra_comps | set(fb_components)))
                     if _extra_comps else fb_components)

_sl_idx, _sl_prices = _slow_trend_history(
    dratio_key, _wk_bytes, ma_bytes,
    origin_k, n_display=min(300, len(view)),
    component_idx=_trend_components,
)

# Якорь прогноза = конец тренда на origin (seamless соединение с линией тренда)
_junction_price = float(_sl_prices[-1]) if len(_sl_prices) > 0 else float(valid["close"].iloc[origin_k])
_junction_ratio = _junction_price / ma0

# Фактический тренд вперёд (C2-C5, если за origin есть реальные данные)
_FWDTREND_COMPS  = (2, 3, 4, 5)
_n_fwd_avail     = min(horizon, len(valid) - 1 - origin_k)
_fwd_trend_idx   = np.array([], dtype=int)
_fwd_trend_prices: np.ndarray = np.array([])
if _n_fwd_avail > 0:
    _end_k           = origin_k + _n_fwd_avail
    _dratio_fwd_full = np.diff(_working_ratio[:_end_k + 1])
    _ma_fwd_seg      = _ma_vals[origin_k: _end_k + 1]
    _fwd_trend_idx, _fwd_trend_prices = _slow_trend_forward(
        _dratio_fwd_full.tobytes(), _junction_ratio,
        _ma_fwd_seg.tobytes(), origin_k, _n_fwd_avail, _FWDTREND_COMPS,
    )

# ── кнопка прогноза ───────────────────────────────────────────────────────────

_ss_key = (ticker, interval, origin_k, horizon,
           fb_p, fb_xi, fb_components, fb_damped_ar, gamma_c1, gamma_c2,
           smooth_ma_w, show_lwc, n_cal,
           merged_slow, merged_damped_c1, merged_gamma_c1,
           lp_att_mode, lp_wn, lp_ma_window,
           tau_shift, lp_auto_p, lp_top_n, lp_p_cands)

_fc1, _fc2 = st.columns([1, 5])
run_btn = _fc1.button("▶ Прогноз", type="primary", use_container_width=True)

forecast_price:   np.ndarray | None = None
forecast_begin:   pd.Series  | None = None
val_price:        np.ndarray | None = None
val_begin:        pd.Series  | None = None
val_mape_val:     float | None      = None
cal_errors:       np.ndarray | None = None
cal_valmapes:     np.ndarray | None = None
lp_top_results:   list | None       = None

if run_btn:
    _spinner_msg = (f"Считаю прогноз + {n_cal} калибровочных окон LWC…"
                    if show_lwc else "Считаю прогноз…")
    with st.spinner(_spinner_msg):
        if lp_att_mode and lp_auto_p:
            # Запускаем от val_origin_k, горизонт = val_h + horizon
            _lp_dratio_val = np.diff(_working_ratio[:val_origin_k + 1])
            lp_top_results = _run_forecast_lp_auto_p(
                _lp_dratio_val.tobytes(), val_origin_k, _lp_total_h,
                wn=lp_wn, tau=max(1, tau_shift),
                p_candidates=lp_p_cands, top_n=lp_top_n,
            )
            dhat = lp_top_results[0][2] if lp_top_results else np.zeros(_lp_total_h)
        elif lp_att_mode:
            _lp_dratio_val = np.diff(_working_ratio[:val_origin_k + 1])
            dhat = _run_forecast_lp_att(
                _lp_dratio_val.tobytes(), val_origin_k, _lp_total_h,
                p=fb_p, xi=fb_xi, wn=lp_wn,
            )
        elif merged_slow:
            dhat = _run_forecast_merged_slow(
                dratio_key, origin_k, horizon,
                p=fb_p, xi=fb_xi,
                damped_c1=merged_damped_c1,
                gamma_c1=merged_gamma_c1,
            )
        elif fb_damped_ar:
            dhat = _run_forecast_fb_damped(
                dratio_key, origin_k, horizon,
                p=fb_p, xi=fb_xi,
                gamma_c1=gamma_c1, gamma_c2=gamma_c2,
                slow_components=fb_components,
            )
        else:
            dhat = _run_forecast_fb(
                dratio_key, origin_k, horizon,
                p=fb_p, xi=fb_xi,
                component_idx=fb_components,
            )

        if lp_att_mode:
            # Якорь = MA[val_origin_k]; logtrend для val_origin_k → origin_k+horizon
            _anchor_ratio  = _ma_anchor / float(_ma_vals[val_origin_k])
            _ma_fwd_lp     = _ma_vals[val_origin_k + 1: val_origin_k + 1 + _lp_total_h]
            if len(_ma_fwd_lp) < _lp_total_h:
                _last_ma_lp = float(_ma_vals[min(val_origin_k + len(_ma_fwd_lp), len(_ma_vals) - 1)])
                _ma_fwd_lp  = np.concatenate([_ma_fwd_lp,
                                               np.full(_lp_total_h - len(_ma_fwd_lp), _last_ma_lp)])
            ratio_hat      = _anchor_ratio + np.cumsum(dhat[:_lp_total_h])
            forecast_price = ratio_hat * _ma_fwd_lp
            # val_mape: первые val_h баров vs MA цен в окне валидации
            _ma_val_ref  = _ma_smooth[val_origin_k + 1: origin_k + 1]
            _n_cmp_lp    = min(val_h, len(_ma_val_ref), len(forecast_price))
            if _n_cmp_lp > 0:
                val_mape_val = float(np.mean(
                    np.abs(forecast_price[:_n_cmp_lp] - _ma_val_ref[:_n_cmp_lp])
                    / (np.abs(_ma_val_ref[:_n_cmp_lp]) + 1e-10)
                ))
        else:
            ratio_hat      = _junction_ratio + np.cumsum(dhat)
            forecast_price = ratio_hat * _ma_fwd

        # Валидация: прогноз от val_origin (только не LP att — там уже сделано выше)
        if val_origin_k >= _origin_min and not lp_att_mode:
            dratio_val = np.diff(_working_ratio[:val_origin_k + 1])
            if lp_att_mode and lp_auto_p:
                _val_res = _run_forecast_lp_auto_p(
                    dratio_val.tobytes(), val_origin_k, val_h,
                    wn=lp_wn, tau=max(1, tau_shift),
                    p_candidates=lp_p_cands, top_n=1,
                )
                dhat_val = _val_res[0][2] if _val_res else np.zeros(val_h)
            elif lp_att_mode:
                dhat_val = _run_forecast_lp_att(
                    dratio_val.tobytes(), val_origin_k, val_h, p=fb_p, xi=fb_xi, wn=lp_wn)
            elif merged_slow:
                dhat_val = _run_forecast_merged_slow(
                    dratio_val.tobytes(), val_origin_k, val_h,
                    p=fb_p, xi=fb_xi,
                    damped_c1=merged_damped_c1,
                    gamma_c1=merged_gamma_c1,
                )
            elif fb_damped_ar:
                dhat_val = _run_forecast_fb_damped(
                    dratio_val.tobytes(), val_origin_k, val_h,
                    p=fb_p, xi=fb_xi,
                    gamma_c1=gamma_c1, gamma_c2=gamma_c2,
                    slow_components=fb_components,
                )
            else:
                dhat_val = _run_forecast_fb(
                    dratio_val.tobytes(), val_origin_k, val_h,
                    p=fb_p, xi=fb_xi,
                    component_idx=fb_components,
                )
            ratio0_val       = float(_working_ratio[val_origin_k])
            ratio_hat_val    = ratio0_val + np.cumsum(dhat_val)
            _ma_val_fwd      = _ma_vals[val_origin_k + 1: val_origin_k + 1 + val_h]
            val_price        = ratio_hat_val[:len(_ma_val_fwd)] * _ma_val_fwd
            actual_ratio_val = ratio_arr[val_origin_k + 1: val_origin_k + 1 + val_h]
            n_cmp            = min(len(ratio_hat_val), len(actual_ratio_val))
            val_mape_val     = float(np.mean(
                np.abs(ratio_hat_val[:n_cmp] - actual_ratio_val[:n_cmp])
                / (np.abs(actual_ratio_val[:n_cmp]) + 1e-10)
            ))

        # LWC калибровка: n_cal непересекающихся окон по val_h баров назад от origin
        if show_lwc and not lp_att_mode:
            cal_errors, cal_valmapes = _run_conformal_calibration(
                ratio_bytes, n_cal, val_h,
                fb_p, fb_xi, fb_damped_ar, gamma_c1, gamma_c2, fb_components,
                smooth_ma_w,
                merged_slow=merged_slow,
                merged_damped_c1=merged_damped_c1,
                merged_gamma_c1=merged_gamma_c1,
            )

    st.session_state.update({
        "fc_price":       forecast_price,
        "val_price":      val_price,
        "val_mape":       val_mape_val,
        "fc_origin_k":    origin_k,
        "fc_ss_key":      _ss_key,
        "fc_junction":    _junction_price,
        "cal_errors":     cal_errors,
        "cal_valmapes":   cal_valmapes,
        "lp_top_results": lp_top_results,
    })

# восстановление из session_state
if (
    "fc_price" in st.session_state
    and st.session_state.get("fc_ss_key") == _ss_key
    and st.session_state.get("fc_origin_k") == origin_k
):
    forecast_price  = st.session_state["fc_price"]
    val_price       = st.session_state.get("val_price")
    val_mape_val    = st.session_state.get("val_mape")
    cal_errors      = st.session_state.get("cal_errors")
    cal_valmapes    = st.session_state.get("cal_valmapes")
    lp_top_results  = st.session_state.get("lp_top_results")

    if lp_att_mode:
        # LP att: forecast_begin от val_origin_k+1 на _lp_total_h баров
        _fc_start  = val_origin_k + 1
        _fc_len    = _lp_total_h
        _avail_fc  = valid.iloc[_fc_start: _fc_start + _fc_len]
        if len(_avail_fc) < _fc_len:
            bar_delta  = _BAR_DELTA.get(interval, pd.Timedelta(hours=1))
            _last_ts   = (_avail_fc["begin"].iloc[-1] if len(_avail_fc)
                          else valid["begin"].iloc[val_origin_k])
            _extra     = pd.Series([_last_ts + bar_delta * i
                                    for i in range(1, _fc_len - len(_avail_fc) + 1)])
            forecast_begin = pd.concat(
                [_avail_fc["begin"].reset_index(drop=True), _extra], ignore_index=True)
        else:
            forecast_begin = _avail_fc["begin"].reset_index(drop=True)
    else:
        future_valid = valid.iloc[origin_k + 1: origin_k + 1 + horizon]
        if len(future_valid) < horizon:
            bar_delta  = _BAR_DELTA.get(interval, pd.Timedelta(hours=1))
            last_known = (future_valid["begin"].iloc[-1] if len(future_valid)
                          else valid["begin"].iloc[origin_k])
            extra = [last_known + bar_delta * i
                     for i in range(1, horizon - len(future_valid) + 1)]
            forecast_begin = pd.concat(
                [future_valid["begin"].reset_index(drop=True), pd.Series(extra)],
                ignore_index=True,
            )
        else:
            forecast_begin = future_valid["begin"].reset_index(drop=True)

    if val_price is not None:
        val_future   = valid.iloc[val_origin_k + 1: val_origin_k + 1 + val_h]
        val_begin    = val_future["begin"].reset_index(drop=True)

    if val_mape_val is not None:
        if lp_att_mode and lp_auto_p and lp_top_results:
            _lbl_detail = f"val_mape · LP({lp_wn})+LWR · p={lp_top_results[0][0]}★"
        elif lp_att_mode:
            _lbl_detail = f"val_mape · LP({lp_wn})+LWR · p={fb_p}"
        elif fb_damped_ar:
            _lbl_detail = f"val_mape · LWR+FB+Damped AR · p={fb_p} · γC1={gamma_c1} γC2={gamma_c2}"
        else:
            _lbl_detail = f"val_mape · LWR+FB · p={fb_p}"
        _fc2.metric(_lbl_detail, f"{val_mape_val:.4f}",
                    help=f"Средний MAPE прогноза на последних {val_h} барах перед origin.")

# ── chart ─────────────────────────────────────────────────────────────────────

fig = go.Figure()

# свечи
fig.add_trace(go.Candlestick(
    x=view["begin"], open=view["open"], high=view["high"],
    low=view["low"],  close=view["close"],
    name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    whiskerwidth=0.5,
))

# медленный тренд
_sl_dates = valid["begin"].iloc[_sl_idx].values
_trend_lbl = f"C{''.join(str(c) for c in _trend_components)} тренд"
fig.add_trace(go.Scatter(
    x=_sl_dates, y=_sl_prices,
    mode="lines", name=_trend_lbl,
    line=dict(color="rgba(100,181,246,0.7)", width=1.8, dash="dot"),
))

# фактический медленный тренд C2-C5 вперёд (пунктир, только если есть данные)
if len(_fwd_trend_idx) > 0:
    _fwd_dates = valid["begin"].iloc[_fwd_trend_idx].values
    fig.add_trace(go.Scatter(
        x=_fwd_dates, y=_fwd_trend_prices,
        mode="lines", name="C2-C5 тренд факт",
        line=dict(color="rgba(100,181,246,0.45)", width=1.5, dash="dash"),
    ))

# MA-линия сравнения (LP att режим)
if lp_att_mode:
    _ma_view_mask = view.index
    _ma_hist_x    = view["begin"]
    _ma_hist_y    = pd.Series(_close_vals).rolling(lp_ma_window, min_periods=1).mean().values[_ma_view_mask]
    fig.add_trace(go.Scatter(
        x=_ma_hist_x, y=_ma_hist_y,
        mode="lines", name=f"MA({lp_ma_window})",
        line=dict(color="rgba(255,183,77,0.7)", width=1.5),
    ))

# вертикальная линия origin
fig.add_vline(x=str(origin_ts)[:10],
              line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"))

# прогноз — начинается с точки junction (конец тренда) для seamless соединения
if forecast_price is not None and forecast_begin is not None:
    n_fc = min(len(forecast_price), len(forecast_begin))

    # ── метка прогноза ────────────────────────────────────────────────────────
    if lp_att_mode and lp_auto_p and lp_top_results:
        fc_lbl = f"LP({lp_wn})+LWR p={lp_top_results[0][0]}★" + (f" τ={tau_shift}б" if tau_shift else "")
    elif lp_att_mode:
        fc_lbl = f"LP({lp_wn})+LWR p={fb_p}" + (f" τ={tau_shift}б" if tau_shift else "")
    elif fb_damped_ar:
        fc_lbl = f"LWR+FB+Damped(γC1={gamma_c1}/γC2={gamma_c2}) p={fb_p}"
    else:
        fc_lbl = f"LWR+FB p={fb_p}"

    # ── x/y прогнозной линии ──────────────────────────────────────────────────
    if lp_att_mode:
        # Якорь — MA[val_origin_k]; прогноз от val_origin_k на val_h+horizon
        _fc_x      = pd.concat([pd.Series([valid["begin"].iloc[val_origin_k]]),
                                 forecast_begin[:n_fc]], ignore_index=True)
        _fc_anchor = _ma_anchor
    else:
        # tau_shift: сдвигаем прогноз влево на τ баров (компенсация задержки фильтра)
        _junc_k  = max(0, origin_k - tau_shift)
        _junc_ts = valid["begin"].iloc[_junc_k]
        if tau_shift > 0:
            _fc_start_k = _junc_k + 1
            _avail      = max(0, min(n_fc, len(valid) - _fc_start_k))
            _real_dates = (valid["begin"].iloc[_fc_start_k: _fc_start_k + _avail]
                           .reset_index(drop=True) if _avail > 0
                           else pd.Series([], dtype="datetime64[ns]"))
            if _avail < n_fc:
                _bar_delta = _BAR_DELTA.get(interval, pd.Timedelta(hours=1))
                _last_ts   = _real_dates.iloc[-1] if len(_real_dates) > 0 else _junc_ts
                _synth     = pd.Series([_last_ts + _bar_delta * i
                                        for i in range(1, n_fc - _avail + 1)])
                _fc_dates  = pd.concat([_real_dates, _synth], ignore_index=True)
            else:
                _fc_dates = _real_dates
            _fc_x = pd.concat([pd.Series([_junc_ts]), _fc_dates[:n_fc]], ignore_index=True)
        else:
            _fc_x = pd.concat([pd.Series([valid["begin"].iloc[origin_k]]),
                                forecast_begin[:n_fc]], ignore_index=True)
        _fc_anchor = float(valid["close"].iloc[origin_k]) if lp_att_mode else _junction_price

    _fc_y = np.concatenate([[_fc_anchor], forecast_price[:n_fc]])

    # LWC Conformal Band — самый нижний слой (под C0 полосой)
    if (show_lwc and n_fc > 0
            and cal_errors is not None and cal_valmapes is not None
            and val_mape_val is not None):
        _lwc_up, _lwc_lo = _compute_lwc_band(
            cal_errors, cal_valmapes, val_mape_val,
            lwc_h_kernel, lwc_alpha,
            forecast_price[:n_fc], _ma_fwd, val_h,
        )
        _lwc_upper_y = np.concatenate([[_junction_price], _lwc_up])
        _lwc_lower_y = np.concatenate([[_junction_price], _lwc_lo])
        fig.add_trace(go.Scatter(
            x=_fc_x, y=_lwc_upper_y, mode="lines",
            line=dict(color="rgba(255,183,77,0.25)", width=0.8, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=_fc_x, y=_lwc_lower_y, mode="lines",
            line=dict(color="rgba(255,183,77,0.25)", width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(255,183,77,0.10)",
            name=f"LWC {int((1-lwc_alpha)*100)}% (N={n_cal})",
            hoverinfo="skip",
        ))

    # Полоса неопределённости C0 — добавляем ДО линии прогноза (чтобы линия была поверх)
    if show_band and n_fc > 0:
        _sigma_c0 = _compute_c0_sigma(dratio_key, sigma_win_c0)
        _h_arr    = np.arange(1, n_fc + 1, dtype=np.float64)
        _band_r   = k_band * _sigma_c0 * np.sqrt(_h_arr)            # в ratio
        _upper_y  = np.concatenate([[_junction_price],
                                     forecast_price[:n_fc] + _band_r * _ma_fwd[:n_fc]])
        _lower_y  = np.concatenate([[_junction_price],
                                     forecast_price[:n_fc] - _band_r * _ma_fwd[:n_fc]])
        # Верхняя граница (invisible fill anchor)
        fig.add_trace(go.Scatter(
            x=_fc_x, y=_upper_y, mode="lines",
            line=dict(color="rgba(0,229,255,0.20)", width=0.8, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
        # Нижняя граница + заливка до верхней
        fig.add_trace(go.Scatter(
            x=_fc_x, y=_lower_y, mode="lines",
            line=dict(color="rgba(0,229,255,0.20)", width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(0,229,255,0.10)",
            name=f"C0 ±{k_band:.1f}σ√h",
            hoverinfo="skip",
        ))

    if (lp_att_mode and lp_auto_p
            and lp_top_results is not None and len(lp_top_results) > 1):
        _rank_opacities = [0.50, 0.35, 0.22, 0.15, 0.10]
        for _rank, (_p_i, _sc_i, _dhat_i) in enumerate(lp_top_results[1:], 1):
            _rh_i  = _anchor_ratio + np.cumsum(_dhat_i[:_lp_total_h])
            _fp_i  = _rh_i * _ma_fwd_lp[:len(_rh_i)]
            _n_i   = min(len(_fp_i), n_fc)
            _fy_i  = np.concatenate([[_fc_anchor], _fp_i[:_n_i]])
            _op    = _rank_opacities[min(_rank - 1, len(_rank_opacities) - 1)]
            fig.add_trace(go.Scatter(
                x=_fc_x[:len(_fy_i)], y=_fy_i,
                mode="lines",
                name=f"p={_p_i} (#{_rank + 1}, mse={_sc_i:.2e})",
                line=dict(color=f"rgba(0,229,255,{_op})", width=1.2, dash="dot"),
            ))

    fig.add_trace(go.Scatter(
        x=_fc_x, y=_fc_y,
        mode="lines", name=fc_lbl,
        line=dict(color="#00e5ff", width=2.5),
    ))

# валидационный прогноз (серый пунктир)
if val_price is not None and val_begin is not None:
    n_val = min(len(val_price), len(val_begin))
    fig.add_trace(go.Scatter(
        x=val_begin[:n_val], y=val_price[:n_val],
        mode="lines", name=f"val ({val_h}б)",
        line=dict(color="rgba(180,180,180,0.6)", width=1.5, dash="dot"),
    ))

fig.update_layout(
    **_layout, height=520,
    xaxis=dict(**_grid, title="", rangebreaks=rangebreaks,
               rangeslider=dict(visible=False)),
    yaxis=dict(**_grid, title="Цена"),
    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

# ── статистика ────────────────────────────────────────────────────────────────

_info1, _info2, _info3 = st.columns(3)
_info1.metric("Баров в истории", f"{origin_k:,}")
_info2.metric("logtrend ratio", f"{ratio0:.4f}")
_info3.metric("Trend (MA)", f"{ma0:.2f}")

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
    _sp_n  = len(_sp_df) - 1

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

        _sp_dates_arr = _sp_df["begin"].values[1:]
        _t_idx    = np.round(_t_spec).astype(int).clip(0, _sp_n - 1)
        _t_labels = (
            pd.DatetimeIndex(_sp_dates_arr[_t_idx]).strftime("%Y-%m-%d").tolist()
        )

        # Реальные частоты среза фильтрбанка (Гц, fs=1):
        # butter(order, Wn) без fs → fc_real = Wn * Nyquist = Wn/2
        _CUTOFFS_HZ     = [wn / 2 for wn in [0.25, 0.125, 0.0625, 0.03125, 0.015625]]
        _CUTOFFS_LABELS = ["C0/C1", "C1/C2", "C2/C3", "C3/C4", "C4/C5"]
        _CUTOFFS_PERIODS = [round(1.0 / fc) for fc in _CUTOFFS_HZ]

        if _sp_fmin >= _sp_fmax:
            st.warning("Мин. частота должна быть меньше макс. частоты.")
            _sp_fmin = 0.0
        _sp_fmask = (_f_spec >= _sp_fmin) & (_f_spec <= _sp_fmax)
        _f_plot   = _f_spec[_sp_fmask]
        _Sxx_plot = _Sxx_db[_sp_fmask, :]

        _zmin = float(np.percentile(_Sxx_plot, 5))
        _zmax = float(np.percentile(_Sxx_plot, 95))

        _fig_sp = go.Figure(go.Heatmap(
            z=_Sxx_plot, x=_t_labels, y=_f_plot.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="дБ/Гц", thickness=12, len=0.8),
            zmin=_zmin, zmax=_zmax, hoverongaps=False,
        ))
        for _fc_hz, _lbl, _per in zip(_CUTOFFS_HZ, _CUTOFFS_LABELS, _CUTOFFS_PERIODS):
            if _sp_fmin <= _fc_hz <= _sp_fmax and len(_f_plot) > 0:
                _fig_sp.add_hline(
                    y=_fc_hz,
                    line=dict(color="rgba(255,80,80,0.75)", width=1.5, dash="dot"),
                    annotation_text=f" {_lbl} T≈{_per}б",
                    annotation_font_color="rgba(255,140,140,1.0)",
                    annotation_font_size=9,
                    annotation_position="right",
                )

        _sp_f_hi  = float(_f_plot[-1]) if len(_f_plot) > 0 else _sp_fmax
        _sp_f_lo  = float(_f_plot[0])  if len(_f_plot) > 0 else _sp_fmin
        _sp_yaxis: dict = dict(**_grid, title="Частота (цикл/бар)")
        if _sp_log_y:
            _sp_log_lo = max(_sp_f_lo, 1e-4)
            _sp_yaxis.update(type="log",
                             range=[np.log10(_sp_log_lo), np.log10(_sp_f_hi)])
        else:
            _sp_yaxis.update(range=[_sp_f_lo, _sp_f_hi])

        _fig_sp.update_layout(
            **_layout, height=400,
            title=dict(text=f"Спектрограмма Δratio — {ticker} [{interval}]",
                       font=dict(size=12)),
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
