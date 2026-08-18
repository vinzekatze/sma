"""
Streamlit prototype v2: LWR auto-p — прямой grid search.

Стек: logtrend-нормализация + LWR на dratio, прямой перебор (p, ξ [, pca_k])
на валидационном окне, топ-N кандидатов на графике + тепловая карта p×ξ.

Отличие от app.py: нет filter bank — исследуем, насколько простой LWR с
подобранными параметрами конкурентоспособен без частотного разложения.

Run (из prototype/):  streamlit run forcaster/ui/app2.py
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
from forcaster.forecast.lwr import forecast_lwr

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

# ── logtrend ──────────────────────────────────────────────────────────────────

def _logtrend_causal(close: np.ndarray) -> np.ndarray:
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


# ── grid search ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _grid_search_lwr(
    dratio_bytes: bytes,
    ratio_bytes:  bytes,
    origin_k:   int,
    val_h:      int,
    horizon:    int,
    p_min:      int,
    p_max:      int,
    p_step:     int,
    xi_mode:    str,                  # "rule" | "multipliers" | "grid"
    xi_mults:   tuple[int, ...],      # xi_mode="multipliers": ξ = m*(p+1)
    xi_min:     int,                  # xi_mode="grid"
    xi_max:     int,
    xi_step:    int,
    top_n:      int,
    norm_vecs:  bool,
    use_huber:  bool,
    pca_mode:   str,                  # "none" | "half_p" | "range"
    pca_k_min:  int,                  # pca_mode="range"
    pca_k_max:  int,
    pca_k_step: int,
) -> tuple[list[dict], dict]:
    """
    Прямой перебор (p, ξ [, pca_k]).
    Валидация: прогноз от val_origin = origin_k − val_h, MAPE на [val_origin+1…origin_k].
    Прогноз вперёд: от origin_k на horizon (единый прогон total_h = val_h + horizon).
    Возвращает (топ-N кандидатов, scores).
    scores: ключ "(p, xi, pca_k)" → val_mape.
    """
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    ratio  = np.frombuffer(ratio_bytes,  dtype=np.float64).copy()

    val_origin   = origin_k - val_h
    ratio0_val   = float(ratio[val_origin])
    actual_ratio = ratio[val_origin + 1 : val_origin + 1 + val_h]
    total_h      = val_h + horizon

    all_results: list[tuple[float, int, int, int, np.ndarray]] = []
    scores: dict[str, float] = {}

    for p in range(p_min, p_max + 1, p_step):
        if val_origin < p + 5:
            continue

        # сетка ξ
        if xi_mode == "rule":
            xi_list = [3 * (p + 1)]
        elif xi_mode == "multipliers":
            xi_list = sorted({m * (p + 1) for m in xi_mults if m * (p + 1) >= p + 2})
        else:  # "grid"
            xi_list = list(range(max(xi_min, p + 2), xi_max + 1, xi_step))
        if not xi_list:
            continue

        # сетка pca_k
        if pca_mode == "none":
            pca_list = [0]
        elif pca_mode == "half_p":
            pca_list = [max(2, p // 2)] if p >= 4 else [0]
        else:  # "range"
            pca_list = list(range(pca_k_min, min(pca_k_max + 1, p), pca_k_step))
            if not pca_list:
                pca_list = [0]

        for xi in xi_list:
            if val_origin < xi + p + 2:
                continue
            for pca_k in pca_list:
                try:
                    dhat = forecast_lwr(
                        dratio, val_origin, p, xi, total_h,
                        norm_vecs=norm_vecs, use_huber=use_huber, pca_k=pca_k,
                    )
                except Exception:
                    continue
                ratio_hat_val = ratio0_val + np.cumsum(dhat[:val_h])
                n = min(len(ratio_hat_val), len(actual_ratio))
                if n == 0:
                    continue
                mape = float(np.mean(
                    np.abs(ratio_hat_val[:n] - actual_ratio[:n])
                    / (np.abs(actual_ratio[:n]) + 1e-12)
                ))
                scores[f"({p},{xi},{pca_k})"] = mape
                all_results.append((mape, p, xi, pca_k, dhat))

    if not all_results:
        return [], {}

    all_results.sort(key=lambda x: x[0])
    top_cands = [
        {"p": p, "xi": xi, "pca_k": pca_k, "mape": m, "dhat": d.tolist()}
        for m, p, xi, pca_k, d in all_results[:top_n]
    ]
    return top_cands, scores


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="MOEX LWR auto-p v2", layout="wide",
                   initial_sidebar_state="expanded")

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("LWR auto-p v2")
    st.caption("Прямой grid search по (p, ξ) на валидационном окне.")

    ticker   = st.text_input("Тикер", value="SBER", key="ticker_val").upper().strip()
    interval = st.selectbox("Интервал", list(INTERVALS),
                             index=list(INTERVALS).index("1d"), key="interval_val")

    st.divider()
    st.subheader("Горизонт")
    horizon = st.slider("Горизонт прогноза (баров)", 1, 100, 20, 1, key="horizon")
    val_h   = st.slider("Окно валидации (баров назад)", 2, 100, 20, 1, key="val_h",
                        help="Прогноз строится от origin − val_h, MAPE считается "
                             "на следующих val_h барах (они уже известны).")

    st.divider()
    st.subheader("Сетка p")
    _p1, _p2, _p3 = st.columns(3)
    p_min  = int(_p1.number_input("min", 2, 200,  2, 1, key="p_min"))
    p_max  = int(_p2.number_input("max", 2, 500, 80, 1, key="p_max"))
    p_step = int(_p3.number_input("шаг", 1,  50,  2, 1, key="p_step"))
    _n_p   = max(1, (p_max - p_min) // p_step + 1)
    st.caption(f"~{_n_p} значений p")

    st.divider()
    st.subheader("Сетка ξ (соседей)")
    xi_mode = st.radio(
        "Режим",
        ["rule", "multipliers", "grid"],
        format_func=lambda x: {
            "rule":        "Правило 3·(p+1)",
            "multipliers": "Множители m·(p+1)",
            "grid":        "Явная сетка",
        }[x],
        key="xi_mode",
    )

    xi_mults = (2, 3, 4, 5)
    xi_min, xi_max, xi_step = 10, 200, 10

    if xi_mode == "multipliers":
        _ms = st.multiselect(
            "Множители m (ξ = m·(p+1))", [2, 3, 4, 5, 6, 8, 10],
            default=[2, 3, 4, 5], key="xi_mults",
        )
        xi_mults = tuple(sorted(_ms)) if _ms else (3,)
        st.caption(f"ξ будет: {', '.join(f'{m}·(p+1)' for m in xi_mults)}")
    elif xi_mode == "grid":
        _x1, _x2, _x3 = st.columns(3)
        xi_min  = int(_x1.number_input("ξ min", 2, 2000,  10, 1, key="xi_min"))
        xi_max  = int(_x2.number_input("ξ max", 2, 5000, 200, 1, key="xi_max"))
        xi_step = int(_x3.number_input("шаг",   1,  200,  10, 1, key="xi_step"))
        _n_xi   = max(1, (xi_max - xi_min) // xi_step + 1)
        st.caption(f"~{_n_xi} значений ξ | сетка ~{_n_p * _n_xi} точек")
    else:
        st.caption("ξ = 3·(p+1) — стандартное правило")

    st.divider()
    st.subheader("PCA-метрика")
    pca_mode = st.radio(
        "Режим PCA",
        ["none", "half_p", "range"],
        format_func=lambda x: {
            "none":   "Без PCA",
            "half_p": "k = p//2",
            "range":  "Сетка k",
        }[x],
        key="pca_mode",
        help="Проецирует матрицу задержек в k главных компонент перед поиском соседей.",
    )
    pca_k_min, pca_k_max, pca_k_step = 2, 20, 2
    if pca_mode == "range":
        _k1, _k2, _k3 = st.columns(3)
        pca_k_min  = int(_k1.number_input("k min", 2, 100,  2, 1, key="pca_k_min"))
        pca_k_max  = int(_k2.number_input("k max", 2, 200, 20, 1, key="pca_k_max"))
        pca_k_step = int(_k3.number_input("шаг",   1,  20,  2, 1, key="pca_k_step"))

    st.divider()
    st.subheader("Опции LWR")
    norm_vecs = st.checkbox("Нормировать форму вектора (w/σ)", key="norm_vecs",
                             help="Паттерны с одинаковой формой, но разным масштабом, "
                                  "становятся соседями.")
    use_huber = st.checkbox("Huber-регрессия (робастная)", key="use_huber",
                             help="IRLS с Huber-штрафом — снижает влияние гэпов.")

    st.divider()
    st.subheader("Отображение")
    top_n      = st.slider("Top-N прогнозов на графике", 1, 20, 5, 1, key="top_n")
    show_val   = st.checkbox("Показывать вал. окно лучшего кандидата", value=True, key="show_val")
    show_heatmap = st.checkbox("Тепловая карта p×ξ", value=True, key="show_heatmap",
                                disabled=(xi_mode == "rule"),
                                help="Доступна при режиме multipliers или grid.")

# ── data load ─────────────────────────────────────────────────────────────────

cache_path = DATA_DIR / ticker / f"{interval}.json"

c1, c2, c3 = st.columns([3, 1, 1])
c1.title("LWR auto-p — прямой перебор")
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

# ── нормализация (logtrend) ───────────────────────────────────────────────────

candles_key = cache_path.read_text(encoding="utf-8")
valid       = _normalize_logtrend(candles_key).dropna(subset=["ratio"]).reset_index(drop=True)

# ── period selector ───────────────────────────────────────────────────────────

PERIODS = {"1 мес": 30, "3 мес": 90, "6 мес": 180, "1 год": 365, "3 года": 1095, "Всё": None}
period  = st.radio("Период", list(PERIODS), index=2, horizontal=True)
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

_grid_style = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
_layout     = dict(dragmode="pan", paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0))

# ── origin selector ───────────────────────────────────────────────────────────

_origin_min     = max(p_max + val_h + 10, 100)
_origin_max     = len(valid) - 1
_origin_default = _origin_max

if _origin_min >= _origin_max:
    st.error(f"Недостаточно баров: нужно минимум {_origin_min + 1}, есть {len(valid)}.")
    st.stop()

origin_k  = st.slider(
    "Origin (последняя известная свеча)",
    min_value=_origin_min, max_value=_origin_max,
    value=_origin_default, step=1, key="origin_k",
)
origin_ts = valid["begin"].iloc[origin_k]
ratio0    = float(valid["ratio"].iloc[origin_k])
ma0       = float(valid["ma"].iloc[origin_k])

ratio_arr  = valid["ratio"].values.astype(np.float64)
dratio     = np.diff(ratio_arr[:origin_k + 1])
dratio_key = dratio.tobytes()
ratio_key  = ratio_arr[:origin_k + 1].tobytes()

# MA вперёд для реконструкции цены
_ma_vals = valid["ma"].values.astype(np.float64)
_ma_fwd  = _ma_vals[origin_k + 1: origin_k + 1 + horizon]
if len(_ma_fwd) < horizon:
    _last_ma = float(_ma_vals[min(origin_k + len(_ma_fwd), len(_ma_vals) - 1)])
    _ma_fwd  = np.concatenate([_ma_fwd, np.full(horizon - len(_ma_fwd), _last_ma)])

# MA для валидационного окна (val_origin+1 … origin_k)
val_origin_k = origin_k - val_h
_ma_val_fwd  = _ma_vals[val_origin_k + 1: val_origin_k + 1 + val_h]

# ── запуск поиска ─────────────────────────────────────────────────────────────

_ss_key = (ticker, interval, origin_k, horizon, val_h,
           p_min, p_max, p_step, xi_mode,
           xi_mults if xi_mode == "multipliers" else (),
           xi_min if xi_mode == "grid" else 0,
           xi_max if xi_mode == "grid" else 0,
           xi_step if xi_mode == "grid" else 0,
           top_n, norm_vecs, use_huber, pca_mode,
           pca_k_min if pca_mode == "range" else 0,
           pca_k_max if pca_mode == "range" else 0,
           pca_k_step if pca_mode == "range" else 0)

_btn_col, _info_col = st.columns([1, 5])
run_btn = _btn_col.button("▶ Поиск", type="primary", use_container_width=True)

top_cands_prices: list[dict] | None = None
scores: dict | None = None

if run_btn:
    with st.spinner("Grid search…"):
        top_cands, scores_raw = _grid_search_lwr(
            dratio_key, ratio_key,
            origin_k, val_h, horizon,
            p_min, p_max, p_step,
            xi_mode,
            xi_mults if xi_mode == "multipliers" else (),
            xi_min if xi_mode == "grid" else 0,
            xi_max if xi_mode == "grid" else 0,
            xi_step if xi_mode == "grid" else 1,
            top_n, norm_vecs, use_huber,
            pca_mode,
            pca_k_min if pca_mode == "range" else 2,
            pca_k_max if pca_mode == "range" else 20,
            pca_k_step if pca_mode == "range" else 2,
        )
    if not top_cands:
        st.warning("Нет результатов — попробуйте уменьшить p_min, p_max или val_h.")
        st.stop()

    # реконструкция цен для каждого кандидата
    _results = []
    for cand in top_cands:
        dhat    = np.array(cand["dhat"])
        # валидационный прогноз: от val_origin на val_h шагов
        ratio0_v    = float(ratio_arr[val_origin_k])
        val_rhat    = ratio0_v + np.cumsum(dhat[:val_h])
        val_prices  = val_rhat[:len(_ma_val_fwd)] * _ma_val_fwd
        # основной прогноз: от origin_k на horizon (через смещение val_h в dhat)
        # ratio0 в origin_k = ratio0_v + cumsum(dratio[val_origin_k:origin_k])
        # но dhat — непрерывный прогноз от val_origin, так что:
        # ratio_hat[val_h + h] = ratio0_v + sum(dhat[:val_h+h+1])
        ratio0_full = ratio0_v + float(np.sum(dhat[:val_h]))
        # однако в точке origin_k у нас известный ratio0 — лучше переякорить
        # чтобы прогноз стартовал ровно с actual ratio в origin
        delta = ratio0 - ratio0_full
        main_rhat   = ratio0 + np.cumsum(dhat[val_h: val_h + horizon])
        main_prices = main_rhat[:len(_ma_fwd)] * _ma_fwd
        _results.append({
            "p":          cand["p"],
            "xi":         cand["xi"],
            "pca_k":      cand["pca_k"],
            "mape":       cand["mape"],
            "val_prices": val_prices.tolist(),
            "fc_prices":  main_prices.tolist(),
        })

    st.session_state.update({
        "gs_results":    _results,
        "gs_scores":     scores_raw,
        "gs_origin_k":   origin_k,
        "gs_ss_key":     _ss_key,
        "gs_val_origin": val_origin_k,
    })

# восстановление из session state
if (
    "gs_results" in st.session_state
    and st.session_state.get("gs_ss_key") == _ss_key
    and st.session_state.get("gs_origin_k") == origin_k
):
    top_cands_prices = st.session_state["gs_results"]
    scores           = st.session_state["gs_scores"]
    val_origin_k     = st.session_state["gs_val_origin"]

# метрика лучшего кандидата
if top_cands_prices:
    best = top_cands_prices[0]
    _xi_label = ("3·(p+1)" if xi_mode == "rule"
                 else f"ξ={best['xi']}")
    _pca_label = ("" if best["pca_k"] == 0 else f" · pca_k={best['pca_k']}")
    _info_col.metric(
        f"val_mape (лучший) · p={best['p']} · {_xi_label}{_pca_label}",
        f"{best['mape']:.4f}",
        help=f"Средний MAPE прогноза на последних {val_h} барах перед origin.",
    )

# ── forecast_begin ────────────────────────────────────────────────────────────

forecast_begin: pd.Series | None = None
val_begin:      pd.Series | None = None

if top_cands_prices:
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

    val_future = valid.iloc[val_origin_k + 1: val_origin_k + 1 + val_h]
    val_begin  = val_future["begin"].reset_index(drop=True)

# ── график ────────────────────────────────────────────────────────────────────

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=view["begin"], open=view["open"], high=view["high"],
    low=view["low"],  close=view["close"],
    name=ticker,
    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    whiskerwidth=0.5,
))

# logtrend линия (последние 200 баров)
_trend_start = max(0, origin_k - 200)
fig.add_trace(go.Scatter(
    x=valid["begin"].iloc[_trend_start: origin_k + 1],
    y=valid["ma"].values[_trend_start: origin_k + 1],
    mode="lines", name="logtrend",
    line=dict(color="rgba(100,181,246,0.5)", width=1.5, dash="dot"),
))

fig.add_vline(
    x=str(origin_ts)[:10],
    line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"),
)

# топ-N прогнозов: лучший — яркий, остальные — бледнее
_COLORS = [
    "#00e5ff", "#80cbc4", "#b39ddb", "#ffcc80",
    "#ef9a9a", "#a5d6a7", "#90caf9", "#fff176",
    "#f48fb1", "#ce93d8", "#80deea", "#bcaaa4",
    "#ffe082", "#c5e1a5", "#b0bec5", "#ffab91",
    "#e6ee9c", "#80cbc4", "#b0bec5", "#ffcc80",
]

if top_cands_prices and forecast_begin is not None:
    for rank, cand in enumerate(top_cands_prices):
        fc_prices = np.array(cand["fc_prices"])
        n_fc      = min(len(fc_prices), len(forecast_begin))
        opacity   = 1.0 - rank * (0.6 / max(1, len(top_cands_prices) - 1))
        color_hex = _COLORS[rank % len(_COLORS)]
        width     = 2.5 if rank == 0 else 1.5

        # добавляем точку origin как anchor
        _fc_x = pd.concat(
            [pd.Series([valid["begin"].iloc[origin_k]]), forecast_begin[:n_fc]],
            ignore_index=True,
        )
        _fc_y = np.concatenate([[float(valid["close"].iloc[origin_k])],
                                  fc_prices[:n_fc]])

        _xi_lbl = f"ξ={cand['xi']}" if xi_mode != "rule" else "ξ=rule"
        _pca_lbl = f" k={cand['pca_k']}" if cand["pca_k"] > 0 else ""
        name = f"#{rank+1} p={cand['p']} {_xi_lbl}{_pca_lbl} mape={cand['mape']:.4f}"

        # RGB из hex для rgba
        _r, _g, _b = (int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        line_color = f"rgba({_r},{_g},{_b},{opacity:.2f})"

        fig.add_trace(go.Scatter(
            x=_fc_x, y=_fc_y,
            mode="lines", name=name,
            line=dict(color=line_color, width=width),
        ))

    # валидационная кривая лучшего (серый пунктир)
    if show_val and val_begin is not None:
        best = top_cands_prices[0]
        val_prices = np.array(best["val_prices"])
        n_val = min(len(val_prices), len(val_begin))
        fig.add_trace(go.Scatter(
            x=val_begin[:n_val], y=val_prices[:n_val],
            mode="lines", name=f"val #{1} ({val_h}б)",
            line=dict(color="rgba(180,180,180,0.6)", width=1.5, dash="dot"),
        ))

fig.update_layout(
    **_layout, height=520,
    xaxis=dict(**_grid_style, title="", rangebreaks=rangebreaks,
               rangeslider=dict(visible=False)),
    yaxis=dict(**_grid_style, title="Цена"),
    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

# ── статистика ────────────────────────────────────────────────────────────────

_s1, _s2, _s3 = st.columns(3)
_s1.metric("Баров в истории", f"{origin_k:,}")
_s2.metric("logtrend ratio", f"{ratio0:.4f}")
_s3.metric("Trend (MA)", f"{ma0:.2f}")

# ── таблица кандидатов ────────────────────────────────────────────────────────

if top_cands_prices:
    st.subheader("Топ кандидатов по val_mape")
    _df_data = []
    for rank, cand in enumerate(top_cands_prices, 1):
        _df_data.append({
            "Ранг":      rank,
            "p":         cand["p"],
            "ξ":         cand["xi"],
            "pca_k":     cand["pca_k"] if cand["pca_k"] > 0 else "—",
            "val_mape":  f"{cand['mape']:.5f}",
            "ξ/p+1":     f"{cand['xi'] / (cand['p'] + 1):.2f}",
        })
    st.dataframe(pd.DataFrame(_df_data), use_container_width=True, hide_index=True)

    n_total = len(scores) if scores else 0
    best_mape = top_cands_prices[0]["mape"] if top_cands_prices else float("nan")
    st.caption(
        f"Проверено комбинаций: **{n_total:,}**  |  "
        f"Лучший val_mape: **{best_mape:.5f}**  |  "
        f"Гейт (0.0043): {'✅ ниже' if best_mape < 0.0043 else '❌ выше'}"
    )

# ── тепловая карта p × ξ ─────────────────────────────────────────────────────

if (top_cands_prices and scores and show_heatmap and xi_mode != "rule"):
    with st.expander("Тепловая карта val_mape(p, ξ)", expanded=True):
        # парсим ключи "(p,xi,pca_k)"
        import ast as _ast
        _heat: dict[tuple[int, int], float] = {}
        for key, mape in scores.items():
            try:
                p_v, xi_v, _ = _ast.literal_eval(key)
                # берём минимум по pca_k для данной пары (p, xi)
                if (p_v, xi_v) not in _heat or mape < _heat[(p_v, xi_v)]:
                    _heat[(p_v, xi_v)] = mape
            except Exception:
                continue

        if _heat:
            _ps  = sorted({k[0] for k in _heat})
            _xis = sorted({k[1] for k in _heat})
            _Z   = np.full((len(_xis), len(_ps)), np.nan)
            for (p_v, xi_v), mape in _heat.items():
                if p_v in _ps and xi_v in _xis:
                    _Z[_xis.index(xi_v), _ps.index(p_v)] = mape

            _fig_h = go.Figure(go.Heatmap(
                z=_Z, x=[str(p) for p in _ps], y=[str(x) for x in _xis],
                colorscale="RdYlGn_r",
                colorbar=dict(title="val_mape", thickness=12, len=0.8),
                hovertemplate="p=%{x}<br>ξ=%{y}<br>mape=%{z:.5f}<extra></extra>",
            ))
            _fig_h.update_layout(
                **_layout, height=400,
                title=dict(text="val_mape(p, ξ) — минимум по pca_k",
                           font=dict(size=12)),
                xaxis=dict(**_grid_style, title="p"),
                yaxis=dict(**_grid_style, title="ξ"),
            )
            st.plotly_chart(_fig_h, use_container_width=True,
                            config={"scrollZoom": True})

            # подсказка: оптимальная строка
            _best_key = min(_heat, key=_heat.get)
            st.caption(
                f"Оптимум: p={_best_key[0]}, ξ={_best_key[1]}, "
                f"val_mape={_heat[_best_key]:.5f}  "
                f"(ξ/(p+1) = {_best_key[1] / (_best_key[0] + 1):.2f})"
            )
