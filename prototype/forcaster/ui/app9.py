"""
app9: S-map ГОРИЗОНТАЛЬНЫЕ ЗОНЫ неопределённости на 2 шага (фаза 7 —
событийная фрактальность, направление B). Прямой потомок app8, но не
прогноз-траектория, а прогноз-ПОЛЕ: от последнего подтверждённого события
(origin) — цветные горизонтальные зоны цены (красные = ниже, зелёные =
выше), куда исторически уходили аналоги (шаг 1 — «уход») и куда
возвращались (шаг 2 — «уход + возврат», единый показатель, БЕЗ цепочки
через промежуточную точку).

Итог сессии 2026-07-08, зафиксировавший эту конфигурацию:
  — θ=0 (локализация S-map не пережила честную OOS-проверку ни на одном
    из 7 протестированных тикеров — см. research/phase7_fractality/
    29_band_calibration/) — веса всегда равномерные.
  — m=6 — фиксировано (при θ=0 почти не влияет на полосу, эмпирически
    подтверждено на 7 тикерах, разброс pinball в 4-м знаке).
  — T_pool = T_query (единый порог — просто «T»); индивидуальная
    калибровка T_pool на тикер не пережила OOS на 2/7 тикеров и показала
    признаки дрейфа оптимума во времени (29c_tpool_curve_diagnostic.py) —
    фиксированный T_pool=T_query остаётся самым надёжным выбором.
  — Кросс-тикерный пул (UNIVERSE, эксп.17) ОБЯЗАТЕЛЕН — на T=20-30% у
    одного тикера слишком мало пивотов для устойчивых квантилей.
  — Ровно 2 шага (уход + возврат) — не длинная цепочка; читаются НАПРЯМУЮ
    из истории (horizon=1 и horizon=2 в build_pool_vectors), без
    промежуточного прогноза точки шага 1 — см. smap_band_ref.py.

Движок прогноза — research/reference/smap_band_ref.py (каузальность
проверена causality_check_smap_band.py, включая кросс-тикерный пул и
horizon=2). Этот файл — только UI-слой + загрузка данных.

Запуск (из prototype/): streamlit run forcaster/ui/app9.py
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

_UI_DIR = Path(__file__).parent
_ROOT = _UI_DIR.parents[2]                       # .../sma (корень проекта)
_REF_DIR = _ROOT / "research" / "reference"

for p in (str(_ROOT), str(_REF_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sma.data.moex import download_candles, save_candles, INTERVALS
from smap_band_ref import (
    load_ticker_candles, build_zigzag, build_causal_pool, weighted_quantile,
    trim_to_origin, UNIVERSE,
)
from band_lambda_calibrator_ref import (
    load_universe_ticker_data, mask_ticker_data, FEATURE_ORDER,
    MIN_BARS as CALIB_MIN_BARS,
)
from smap_band_weighted_ref import pool_values_and_weights

DATA_DIR = _ROOT / "data" / "candles"

THETA = 0.0     # фиксировано — см. модульный docstring
M = 6           # фиксировано — см. модульный docstring
MIN_BARS = CALIB_MIN_BARS   # =5, синхронизировано с band_lambda_calibrator_ref (2026-07-08):
                            # калиброванные веса подобраны на пивотах min_bars=5, origin,
                            # построенный с min_bars=0, был бы структурно другим событием —
                            # применяется теперь ко ВСЕМ прогонам app9, не только калиброванным

# λ, откалиброванные band_lambda_calibrator_ref.py (многопроходный координатный
# спуск, objective=pinball_norm_avg, 3 прохода, T=20%, 2026-07-08). Ключ —
# (тикер, T% округлённый до 0.1) → используется, только если пользователь
# держит именно этот T; иначе — равномерные веса (θ=0-подобный fallback),
# как раньше. Единственная запись пока — тестовый прогон на SBER.
CALIBRATED_LAMBDAS = {
    ("SBER", 20.0): {
        "volume": 0.0, "trend": 10.0, "leg_age": 8.0,
        "velocity": 0.0, "acceleration": 0.0, "volatility": 2.0,
    },
}

st.set_page_config(page_title="app9 · Зоны S-map", layout="wide")


# ═══════════════════════════════════════════════════════════════════════════════
# Данные
# ═══════════════════════════════════════════════════════════════════════════════

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


@st.cache_data(show_spinner=False)
def _load_universe_ticker_data_cached(interval: str):
    """Кросс-тикерный пул + bar-нативные признаки (объём/тренд/скорость/
    ускорение/волатильность), нужен только для калиброванного (взвешенного)
    пути — считается один раз на интервал, ~10-15с на 45 тикеров."""
    return load_universe_ticker_data(interval)


def _to_arrays(data):
    times   = np.array([c["begin"] for c in data])
    opens   = np.array([float(c["open"])  for c in data])
    highs   = np.array([float(c["high"])  for c in data])
    lows    = np.array([float(c["low"])   for c in data])
    closes  = np.array([float(c["close"]) for c in data])
    return times, opens, highs, lows, closes


# ═══════════════════════════════════════════════════════════════════════════════
# Сайдбар
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("app9 · Зоны")
    st.caption("S-map, θ=0, m=6, кросс-тикерный пул — 2 шага (уход + возврат)")

    ticker   = st.text_input("Тикер", "SBER", key="ticker").upper().strip()
    interval = st.selectbox("Интервал", list(INTERVALS.keys()),
                            index=list(INTERVALS.keys()).index("1d") if "1d" in INTERVALS else 0)

    if st.button("Обновить данные (целевой тикер)"):
        _fetch_and_save(ticker, interval)
        _load_candles_cached.clear()
        st.rerun()

    if st.button("Обновить данные пиров (кросс-тикерный пул)",
                 help="Докачивает недостающие данные для тикеров UNIVERSE на "
                      "выбранный интервал (пропускает уже скачанные). Может "
                      "занять несколько минут на 44 тикера."):
        jobs = [tk for tk in UNIVERSE if tk != ticker
               and not (DATA_DIR / tk / f"{interval}.json").exists()]
        progress = st.progress(0.0, text="Скачивание пиров…")
        for i, tk in enumerate(jobs, start=1):
            progress.progress(i / len(jobs) if jobs else 1.0, text=f"{tk} ({i}/{len(jobs)})")
            _fetch_and_save(tk, interval)
        progress.empty()
        _load_candles_cached.clear()
        st.rerun()

    st.divider()
    st.subheader("Параметры")
    t_pct = st.slider("T, %  (T_query = T_pool)", 5.0, 40.0, 20.0, 0.5,
                      help="Единственный основной параметр — порог зигзага (и запроса, "
                           "и пула, они равны). Информативный диапазон, по итогам "
                           "исследования фазы 7 — 20-30%. На малых T (≤10%) кросс-тикерный "
                           "пул всё ещё нужен, но сигнал слабее проверен.")
    st.markdown("**Зоны вероятности**")
    st.caption("Каждый чекбокс — центральный интервал вокруг медианы (напр. «50%» = "
              "средние 50% исходов, от 25-го до 75-го перцентиля). Включённые зоны одного "
              "шага просто накладываются друг на друга (более узкая — поверх более широкой).")
    LEVEL_OPTIONS = [25, 50, 60, 75, 90, 95]
    DEFAULT_ON = {50, 75, 90}
    level_cols = st.columns(3)
    enabled_levels = []
    for i, lvl in enumerate(LEVEL_OPTIONS):
        with level_cols[i % 3]:
            if st.checkbox(f"{lvl}%", value=(lvl in DEFAULT_ON), key=f"app9_level_{lvl}"):
                enabled_levels.append(lvl)
    if not enabled_levels:
        st.error("Включите хотя бы одну зону.")
        st.stop()

    with st.expander("Зафиксированные параметры (обычно не меняются)"):
        st.write(f"θ (локализация S-map) = **{THETA}** — калибровка не пережила "
                 f"честную OOS-проверку ни на одном из 7 тестовых тикеров (эксп.29).")
        st.write(f"m (размерность вложения) = **{M}** — при θ=0 почти не влияет на полосу.")
        st.caption("research/phase7_fractality/29_band_calibration/")

    st.divider()
    origin_offset = st.slider("Точка отсчёта (баров от конца)", 0, 3000, 0,
                              help="0 — последний доступный бар (живой прогноз). "
                                   ">0 — бэктест из более ранней точки.")
    show_bars = st.slider("Баров на графике", 100, 3000, 400, 50)

    run_btn = st.button("▶  Построить зоны", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Данные и зигзаг целевого тикера
# ═══════════════════════════════════════════════════════════════════════════════

data = _load_candles_cached(ticker, interval)
if not data:
    st.info("Нет локальных данных. Нажмите «Обновить данные (целевой тикер)» в сайдбаре.")
    st.stop()

times, opens, highs, lows, closes = _to_arrays(data)
log_highs_full = np.log(np.maximum(highs, 1e-10))
log_lows_full  = np.log(np.maximum(lows, 1e-10))

log_highs_c, log_lows_c, dates_c = trim_to_origin(log_highs_full, log_lows_full, times, origin_offset)
opens_c, highs_c, lows_c, closes_c = (arr[: len(dates_c)] for arr in (opens, highs, lows, closes))
cutoff_bar_date = str(dates_c[-1])

t_query = t_pct / 100.0
q_lp, q_dates, q_dirs = build_zigzag(log_highs_c, log_lows_c, dates_c, t_query, MIN_BARS)

st.subheader(f"{ticker} · {interval} · зоны S-map (θ=0, m={M}, T={t_pct:.1f}%)")

c1, c2, c3 = st.columns(3)
c1.metric("Пивотов T", len(q_lp))
c2.metric("Направление origin", "▲ HIGH" if (len(q_dirs) and q_dirs[-1] == 1) else ("▼ LOW" if len(q_dirs) else "—"))
c3.metric("Точка отсчёта", cutoff_bar_date[:16])

if len(q_lp) < 3:
    st.error(f"Недостаточно пивотов T={t_pct:.1f}% для построения зон: есть {len(q_lp)}, нужно ≥3. "
             f"Уменьшите T.")
    st.stop()

origin_log_price = float(q_lp[-1])
origin_price      = float(np.exp(origin_log_price))
origin_date       = str(q_dates[-1])
origin_direction  = int(q_dirs[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# Построение зон (кросс-тикерный пул, θ=0 → безусловные квантили)
# ═══════════════════════════════════════════════════════════════════════════════

ZONE_OPACITY = 0.22   # фиксированная прозрачность одного уровня; вложенные уровни
                      # одного цвета просто накладываются друг на друга — Plotly
                      # сам суммирует альфа-канал, узкие (более вероятные) зоны
                      # визуально гуще БЕЗ отдельной формулы плотности.


def build_zones(values: np.ndarray, weights: np.ndarray, levels: list[int], base_log_price: float,
                color: str) -> list[dict]:
    """Для каждого включённого уровня (напр. 50, 75, 90 — % вероятностной
    массы) строит ОДИН центральный интервал [(100-L)/2 .. (100+L)/2]
    перцентиль. weights — веса пула (равномерные ones, если калиброванных
    λ для этого тикера/T нет — см. CALIBRATED_LAMBDAS). Цвет ОДИН на весь
    шаг (см. вызывающий код — определяется чередованием направления
    зигзага, не знаком цены): шаг всегда либо целиком «падение», либо
    целиком «рост» — зигзаг по построению не может пойти в другую сторону
    раньше следующего подтверждённого пивота. Уровни рисуются от широкого
    к узкому — узкие (менее прозрачные по накоплению альфы) оказываются
    «гуще» в центре естественным образом."""
    zones = []
    for level in sorted(levels, reverse=True):   # широкий первым
        frac = level / 100.0
        q_lo, q_hi = (1.0 - frac) / 2.0, (1.0 + frac) / 2.0
        bounds_lr = weighted_quantile(values, weights, [q_lo, q_hi])
        lo = float(np.exp(base_log_price + bounds_lr[q_lo]))
        hi = float(np.exp(base_log_price + bounds_lr[q_hi]))
        zones.append({"lo": lo, "hi": hi, "color": color, "opacity": ZONE_OPACITY, "level": level})
    return zones


calibrated_lambdas = CALIBRATED_LAMBDAS.get((ticker, round(t_pct, 1)))
use_weighted = calibrated_lambdas is not None

if run_btn:
    # ── ДОРОГАЯ часть — загрузка кросс-тикерного пула, только по кнопке ──
    if use_weighted:
        with st.spinner("Загрузка кросс-тикерного пула + признаков (калиброванные веса)…"):
            universe_data = _load_universe_ticker_data_cached(interval)
            if ticker not in universe_data:
                st.error(f"Нет признаков для {ticker} — тикер отсутствует в UNIVERSE или не "
                         f"скачаны данные. Используйте «Обновить данные пиров», затем повторите.")
                st.stop()
            # обрезать ЦЕЛЕВОЙ тикер по origin_offset — та же причинная точка,
            # что и q_lp/q_dates/q_dirs выше (mask_ticker_data не пересчитывает
            # ранги, только обрезает уже посчитанные — они причинны по построению)
            ticker_data_trimmed = dict(universe_data)
            ticker_data_trimmed.update(mask_ticker_data({ticker: universe_data[ticker]}, cutoff_bar_date))

            raw = pool_values_and_weights(ticker, t_query, calibrated_lambdas, ticker_data_trimmed,
                                          origin_index=None)
            if raw is None or not raw["steps"][1]["ok"] or not raw["steps"][2]["ok"]:
                n1 = raw["steps"][1]["pool_size"] if raw else 0
                n2 = raw["steps"][2]["pool_size"] if raw else 0
                st.error(f"Пул слишком мал: шаг 1 = {n1}, шаг 2 = {n2} (нужно ≥{M+2}). "
                         f"Увеличьте T или проверьте, что данные пиров загружены.")
                st.stop()
            s1, s2 = raw["steps"][1], raw["steps"][2]
            n_pool_1, n_pool_2 = s1["pool_size"], s2["pool_size"]
            st.session_state.update({
                "app9_ptr1": s1["values"], "app9_w1": s1["weights"],
                "app9_ptr2": s2["values"], "app9_w2": s2["weights"],
                "app9_weighted": True,
            })
    else:
        with st.spinner("Загрузка кросс-тикерного пула…"):
            ticker_arrays = {}
            for tk in UNIVERSE:
                loaded = load_ticker_candles(tk, interval)
                if loaded is not None:
                    ticker_arrays[tk] = loaded
            if ticker not in ticker_arrays:
                ticker_arrays[ticker] = (log_highs_full, log_lows_full, times)

            t_pool = t_query  # T_pool = T_query, см. docstring
            pfm1, ptr1, pdir1 = build_causal_pool(cutoff_bar_date, ticker_arrays, t_pool, M,
                                                  horizon=1, min_bars=MIN_BARS)
            pfm2, ptr2, pdir2 = build_causal_pool(cutoff_bar_date, ticker_arrays, t_pool, M,
                                                  horizon=2, min_bars=MIN_BARS)

            mask1 = pdir1 == origin_direction
            mask2 = pdir2 == origin_direction

            n_pool_1, n_pool_2 = int(mask1.sum()), int(mask2.sum())
            min_pool = M + 2
            if n_pool_1 < min_pool or n_pool_2 < min_pool:
                st.error(f"Пул слишком мал: шаг 1 = {n_pool_1}, шаг 2 = {n_pool_2} "
                         f"(нужно ≥{min_pool}). Увеличьте T или проверьте, что данные пиров загружены.")
                st.stop()

            st.session_state.update({
                "app9_ptr1": ptr1[mask1], "app9_w1": np.ones(n_pool_1),
                "app9_ptr2": ptr2[mask2], "app9_w2": np.ones(n_pool_2),
                "app9_weighted": False,
            })

    # сохраняем СЫРЫЕ лог-доходности+веса пула (не готовые зоны) — построение
    # зон дешёвое (только квантили) и пересчитывается ниже РЕАКТИВНО, на
    # каждое изменение таблицы границ, без повторной загрузки пула
    st.session_state.update({
        "app9_n_pool_1": n_pool_1, "app9_n_pool_2": n_pool_2,
        "app9_origin_price": origin_price, "app9_origin_date": origin_date,
        "app9_origin_direction": origin_direction, "app9_origin_log_price": origin_log_price,
        "app9_cutoff_date": cutoff_bar_date, "app9_interval": interval, "app9_t_pct": t_pct,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Отображение — зоны пересчитываются здесь, РЕАКТИВНО на таблицу границ
# ═══════════════════════════════════════════════════════════════════════════════

if "app9_ptr1" in st.session_state and st.session_state.get("app9_interval") == interval:
    n_pool_1    = st.session_state["app9_n_pool_1"]
    n_pool_2    = st.session_state["app9_n_pool_2"]
    op          = st.session_state["app9_origin_price"]
    origin_dt   = st.session_state["app9_origin_date"]
    cutoff_dt   = st.session_state["app9_cutoff_date"]
    saved_direction = st.session_state["app9_origin_direction"]
    saved_log_price = st.session_state["app9_origin_log_price"]

    if cutoff_dt != cutoff_bar_date or st.session_state.get("app9_t_pct") != t_pct:
        st.warning("T или точка отсчёта изменились с последнего расчёта пула — зоны ниже "
                  "ещё старые. Нажмите «Построить зоны», чтобы обновить.")

    # цвет шага — по чередованию направления зигзага, НЕ по цене:
    # origin=HIGH(+1) → шаг1 всегда идёт ВНИЗ (красный) → шаг2 обратно ВВЕРХ (зелёный)
    # origin=LOW(−1)  → шаг1 всегда идёт ВВЕРХ (зелёный) → шаг2 обратно ВНИЗ (красный)
    step1_color = "red" if saved_direction == 1 else "green"
    step2_color = "green" if saved_direction == 1 else "red"

    zones_1 = build_zones(st.session_state["app9_ptr1"], st.session_state["app9_w1"],
                          enabled_levels, saved_log_price, step1_color)
    zones_2 = build_zones(st.session_state["app9_ptr2"], st.session_state["app9_w2"],
                          enabled_levels, saved_log_price, step2_color)

    if st.session_state.get("app9_weighted"):
        st.success(f"Калиброванные веса активны: {ticker}, T={t_pct:.1f}%  λ={calibrated_lambdas}")
    else:
        st.caption("Равномерные веса (θ=0) — калибровка для этого тикера/T не задана "
                  "(см. CALIBRATED_LAMBDAS в коде).")

    cc1, cc2 = st.columns(2)
    cc1.metric("Пул шаг 1 (уход)", n_pool_1)
    cc2.metric("Пул шаг 2 (уход+возврат)", n_pool_2)

    # ── таблица зон ──
    with st.expander("Зоны в числах"):
        for label, zones in [("Шаг 1 — уход", zones_1), ("Шаг 2 — уход+возврат", zones_2)]:
            st.markdown(f"**{label}**")
            rows = [{"Уровень, %": z["level"], "Цена от": round(z["lo"], 4),
                    "Цена до": round(z["hi"], 4), "Цвет": z["color"]} for z in zones]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── график ──
    show_from = max(0, len(dates_c) - show_bars)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dates_c[show_from:], open=opens_c[show_from:], high=highs_c[show_from:],
        low=lows_c[show_from:], close=closes_c[show_from:], name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    zz_mask = np.isin(q_dates, dates_c[show_from:])
    if zz_mask.any():
        fig.add_trace(go.Scatter(
            x=q_dates[zz_mask], y=np.exp(q_lp[zz_mask]), mode="lines+markers",
            name=f"зигзаг T={t_pct:.1f}%", line=dict(color="#42a5f5", width=1.5),
            marker=dict(size=5),
        ))

    def _bar_seconds(iv: str) -> float:
        table = {"1m": 60, "10m": 600, "1h": 3600, "1d": 86400, "1w": 604800, "1mo": 2592000}
        return table.get(iv, 86400)

    from datetime import datetime, timedelta
    try:
        origin_dt_obj = datetime.fromisoformat(origin_dt.replace("Z", ""))
    except ValueError:
        origin_dt_obj = datetime.fromisoformat(origin_dt[:19])
    span = timedelta(seconds=_bar_seconds(interval) * max(show_bars // 8, 5))
    x0_1, x1_1 = origin_dt_obj, origin_dt_obj + span     # сегмент 1 — шаг 1
    x0_2, x1_2 = x1_1, x1_1 + span                       # сегмент 2 — шаг 2

    for zones, xa, xb in [(zones_1, x0_1, x1_1), (zones_2, x0_2, x1_2)]:
        for z in zones:
            fillcolor = f"rgba(38,166,154,{z['opacity']:.3f})" if z["color"] == "green" \
                else f"rgba(239,83,80,{z['opacity']:.3f})"
            fig.add_shape(
                type="rect", x0=xa, x1=xb, y0=z["lo"], y1=z["hi"],
                fillcolor=fillcolor, line=dict(width=0), layer="above",
            )

    fig.add_vline(x=origin_dt_obj, line_width=1.5, line_dash="dash", line_color="#ffffff",
                  annotation_text="origin", annotation_position="top left")
    fig.add_hline(y=op, line_width=1, line_dash="dot", line_color="#ffd600",
                 annotation_text=f"origin price {op:.4f}", annotation_position="right")
    fig.add_vline(x=x1_1, line_width=1, line_dash="dot", line_color="#888888")

    fig.update_layout(
        height=650, xaxis_rangeslider_visible=False,
        title=f"{ticker} {interval} · T={t_pct:.1f}% · зоны: шаг 1 (уход) | шаг 2 (уход+возврат)",
        template="plotly_dark", showlegend=True, legend=dict(orientation="h", y=-0.12),
    )
    fig.update_xaxes(type="date")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Цвет шага определяется чередованием направления зигзага (не знаком цены): "
        f"origin — {'HIGH' if saved_direction == 1 else 'LOW'}, значит шаг 1 "
        f"({'падение' if step1_color == 'red' else 'рост'}) — "
        f"{'красный' if step1_color == 'red' else 'зелёный'}, шаг 2 "
        f"({'рост' if step2_color == 'green' else 'падение'}) — "
        f"{'зелёный' if step2_color == 'green' else 'красный'}. Левый сегмент — шаг 1, "
        "правый — шаг 2. Внутри шага более узкие включённые уровни (напр. 50%) накладываются "
        "на более широкие (напр. 90%) — плотнее выглядит центр. Оба шага читаются из истории "
        "напрямую и независимо, без промежуточного прогноза точки."
    )
else:
    st.info("Настройте параметры в сайдбаре и нажмите «Построить зоны».")
