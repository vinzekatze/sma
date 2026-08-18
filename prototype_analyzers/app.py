"""prototype_analyzers/app · статистические анализаторы (не прогноз).

Отдельный прототип для экспериментов со статистическими индикаторами —
не связан с прогнозным пайплайном prototype/forcaster.

Анализатор «скользящий тренд + дисперсия»:
  - OLS-регрессия на log(close); на график накладывается в ценовом
    пространстве (exp), поверх обычных свечей;
  - на основном графике — свечи + ОДНО окно (window баров, включая точку
    отсчёта origin), линия OLS-тренда + НЕСКОЛЬКО вложенных полос
    ±k·std(остатков) с разными k (вместо двух отдельных окон — тот же
    эффект «нескольких масштабов» проще получается вариацией k);
  - опционально — границы полос (только границы, не центр) продлеваются
    пунктиром на n шагов вперёд: линейная экстраполяция ТЕКУЩЕГО slope[origin]
    (без ускорения), ширина полосы (std) держится постоянной. Это НЕ
    прогноз — чисто визуальный ориентир (в духе regression channel),
    точность как цены не валидирована (см. price_forecast_multistep_test.py
    — линейное продолжение тренда стабильно хуже персистенции цены);
  - осциллятор снизу — ОПЦИОНАЛЕН (галочка "Показать осциллятор", по
    умолчанию выключен): полный проход rolling_trend_variance по всей
    истории считается только при включённой галочке (экономия расчётов +
    не занимает подпанель, которая может понадобиться другому
    осциллятору); при включении — скользящие каузальные значения без
    разностей между соседними окнами, переключатель наклон/дисперсия;
  - «веер ускорения» (по умолчанию ВЫКЛЮЧЕН — вспомогательная
    аналитика): второй («гипотетический») линейный тренд из ТОЙ ЖЕ
    стартовой точки j0, наклон скорректирован на измеренное (усреднённое
    по m точкам) ускорение на n_accel шагов. Обе линии строго внутри окна
    — не прогноз, см. accel_predictability_sweep.py (ускорение по одной
    точке — шум как предиктор).

Запуск (из prototype_analyzers/): streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_APP_DIR = Path(__file__).parent
_ROOT = _APP_DIR.parent  # .../sma (корень проекта)

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from sma.data.moex import download_candles, save_candles, INTERVALS
from analyzers.trend_variance import rolling_trend_variance, single_window_trend

DATA_DIR = _ROOT / "data" / "candles"

st.set_page_config(page_title="Аналитические индикаторы · прототип", layout="wide")

BAR_SECONDS = {"1m": 60, "10m": 600, "1h": 3600, "1d": 86400, "1w": 604800, "1mo": 2592000}


@st.cache_data(show_spinner=False)
def _load_candles_cached(ticker: str, interval: str):
    path = DATA_DIR / ticker / f"{interval}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data if data else None


def _fetch_and_save(ticker: str, interval: str):
    data = download_candles(ticker, interval)
    if data:
        save_candles(data, DATA_DIR / ticker / f"{interval}.json")
    return data


with st.sidebar:
    st.title("Статистические анализаторы")
    st.caption("Прототип — не прогноз. Скользящий тренд + дисперсия.")

    ticker = st.text_input("Тикер", value="SBER").strip().upper()
    interval = st.selectbox("Интервал", list(INTERVALS.keys()), index=list(INTERVALS.keys()).index("1d"))

    if st.button("Загрузить / обновить данные"):
        with st.spinner(f"Загрузка {ticker} {interval}..."):
            _fetch_and_save(ticker, interval)
        st.cache_data.clear()

    st.divider()
    window = st.slider("Окно (баров)", min_value=10, max_value=2000, value=200, step=1)

    st.divider()
    st.subheader("Полосы (k·std)")
    k1 = st.slider("Полоса 1", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    k2 = st.slider("Полоса 2", min_value=0.0, max_value=5.0, value=2.0, step=0.1)
    k3 = st.slider("Полоса 3", min_value=0.0, max_value=5.0, value=3.0, step=0.1)

    st.divider()
    show_oscillator = st.checkbox("Показать осциллятор", value=False)
    osc_mode = st.radio(
        "Осциллятор", ["Тренд (slope)", "Дисперсия (resid_var)"],
        disabled=not show_oscillator,
    )

    st.divider()
    st.subheader("Продление полос в будущее")
    show_band_extension = st.checkbox("Продлить границы полос пунктиром", value=True)
    n_future = st.slider("n (шагов вперёд)", min_value=1, max_value=200, value=50, step=1)

    st.divider()
    show_accel_fan = st.checkbox("Показать веер ускорения (вспомогательно)", value=False)
    m_accel = st.slider("m (точек для усреднения ускорения)", min_value=3, max_value=200, value=50, step=1)
    n_accel = st.slider("n (шагов ускорения)", min_value=1, max_value=200, value=50, step=1)


data = _load_candles_cached(ticker, interval)
if not data:
    st.warning("Нет данных для этого тикера/интервала. Нажмите «Загрузить / обновить данные».")
    st.stop()

times = pd.to_datetime([c["begin"] for c in data])
opens = np.array([float(c["open"]) for c in data])
highs = np.array([float(c["high"]) for c in data])
lows = np.array([float(c["low"]) for c in data])
closes = np.array([float(c["close"]) for c in data])
n = len(closes)

origin_min = window - 1
if show_accel_fan:
    origin_min = max(origin_min, window + m_accel - 2)

if n <= origin_min:
    st.warning(f"Недостаточно баров ({n}) для выбранных настроек.")
    st.stop()

log_price = np.log(closes)

origin = st.slider(
    "Точка отсчёта (индекс бара)",
    min_value=origin_min,
    max_value=n - 1,
    value=n - 1,
)
st.caption(f"origin: {times[origin]}")

j0, fitted_log, std_log = single_window_trend(log_price, origin, window)
fitted_price = np.exp(fitted_log)
local_slope = (fitted_log[-1] - fitted_log[0]) / (window - 1)

if show_oscillator:
    slope_series, var_series = rolling_trend_variance(log_price, window)

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    row_heights=[0.7, 0.3], vertical_spacing=0.03,
) if show_oscillator else make_subplots(rows=1, cols=1)

fig.add_trace(go.Candlestick(
    x=times, open=opens, high=highs, low=lows, close=closes,
    name=ticker, showlegend=False,
), row=1, col=1)

seg_times = times[j0:origin + 1]
k_list = sorted([k for k in (k1, k2, k3) if k > 0], reverse=True)  # большие сначала — рисуются под меньшими
for rank, k in enumerate(k_list):
    alpha = 0.10 + 0.08 * rank
    band_hi_price = np.exp(fitted_log + k * std_log)
    band_lo_price = np.exp(fitted_log - k * std_log)
    fig.add_trace(go.Scatter(
        x=seg_times, y=band_hi_price, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=seg_times, y=band_lo_price, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=f"rgba(31,119,180,{alpha:.3f})",
        name=f"±{k:.1f}·std", hoverinfo="skip",
    ), row=1, col=1)

fig.add_trace(go.Scatter(
    x=seg_times, y=fitted_price, mode="lines", name="тренд окна",
    line=dict(color="#1f77b4", width=2.5),
), row=1, col=1)

if show_band_extension and k_list:
    step_seconds = BAR_SECONDS.get(interval, 86400)
    h = np.arange(0, n_future + 1)
    ext_times = times[origin] + pd.to_timedelta(h * step_seconds, unit="s")
    ext_trend_log = fitted_log[-1] + h * local_slope
    for k in k_list:
        hi = np.exp(ext_trend_log + k * std_log)
        lo = np.exp(ext_trend_log - k * std_log)
        fig.add_trace(go.Scatter(
            x=ext_times, y=hi, mode="lines", showlegend=False,
            line=dict(color="#1f77b4", width=1, dash="dot"), hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=ext_times, y=lo, mode="lines", showlegend=False,
            line=dict(color="#1f77b4", width=1, dash="dot"), hoverinfo="skip",
        ), row=1, col=1)

if show_accel_fan:
    _, fitted_slope_of_slope, _ = single_window_trend(slope_series, origin, m_accel)
    d0 = (fitted_slope_of_slope[-1] - fitted_slope_of_slope[0]) / (m_accel - 1)
    slope_hyp = slope_series[origin] + n_accel * d0

    x_win = np.arange(window, dtype=np.float64)
    y0 = fitted_log[0]
    fan_log = y0 + slope_hyp * x_win
    fan_price = np.exp(fan_log)

    fill_color = "rgba(44,160,44,0.25)" if d0 >= 0 else "rgba(214,39,40,0.25)"

    fig.add_trace(go.Scatter(
        x=seg_times, y=fitted_price, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=seg_times, y=fan_price, mode="lines",
        name="тренд + n·ускорение",
        line=dict(color="#7f7f7f", width=1.5, dash="dot"),
        fill="tonexty", fillcolor=fill_color,
    ), row=1, col=1)

if show_oscillator:
    if osc_mode.startswith("Тренд"):
        colors = np.where(slope_series >= 0, "#2ca02c", "#d62728")
        fig.add_trace(go.Bar(
            x=times, y=slope_series, name="slope (тренд)",
            marker_color=colors,
        ), row=2, col=1)
        fig.update_yaxes(title_text="slope", row=2, col=1, zeroline=True)
    else:
        fig.add_trace(go.Scatter(
            x=times, y=var_series, mode="lines", name="resid_var (дисперсия)",
            line=dict(color="#9467bd", width=1),
            fill="tozeroy", fillcolor="rgba(148,103,189,0.2)",
        ), row=2, col=1)
        fig.update_yaxes(title_text="resid_var", row=2, col=1, rangemode="tozero")

fig.update_layout(
    height=800 if show_oscillator else 600,
    hovermode="x unified", legend=dict(orientation="h"), xaxis_rangeslider_visible=False,
)

st.plotly_chart(fig, use_container_width=True)

if show_band_extension:
    st.caption(
        f"Пунктир — границы полос продолжены на {n_future} шагов вперёд по текущему "
        "наклону тренда (без ускорения, ширина полосы постоянна). Визуальный ориентир, "
        "НЕ провалидированный прогноз цены (линейное продолжение тренда стабильно хуже "
        "персистенции — см. price_forecast_multistep_test.py)."
    )

if show_accel_fan:
    st.caption(
        f"Веер — куда сместился бы наклон тренда, если бы ускорение (усреднённое "
        f"по последним {m_accel} точкам slope) действовало ещё {n_accel} шагов "
        "(не прогноз, обе линии — внутри окна): "
        "зелёный = ускорение положительное (в сторону роста), "
        "красный = отрицательное (в сторону падения)."
    )
