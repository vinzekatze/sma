"""
app8: S-map/KNN прогноз зигзага в событийном времени (фаза 7 — событийная фрактальность).

Алгоритм — из research/reference/smap_ref.py:
  1. log(high), log(low) → каузальный зигзаг (порог T, доля лог-цены)
  2. Целевой зигзаг T_query задаёт вектор запроса (m последних лог-доходностей плеч) + направление
  3. Пул(ы) соседей T_pool задают матрицу признаков + целевые лог-доходности до следующего пивота
  4. Пул фильтруется по направлению (однонаправленные с запросом события)
  5. Предсказание (переключатель в сайдбаре):
       — S-map (Sugihara, 1994): взвешенная OLS по пулу, w_j = exp(−θ·d_j/mean(d))
       — KNN: среднее целевых лог-доходностей k ближайших соседей (k=1 — прямое
         использование значения ближайшего соседа, без усреднения)
  6. Итеративный многошаговый прогноз: предсказанная лог-доходность входит в контекст
     следующего шага, направление на каждом шаге чередуется (свойство зигзага)

Расширение относительно smap_ref.py:
  — Несколько пулов соседей одновременно (разные пороги T и/или интервалы),
    объединяются в один пул перед S-map (концепция фазы 7: крупные события
    строятся из мелких, T_frac ≈ 0.85×T_big).
  — Многошаговый прогноз в UI (не только CLI --steps).

Оформление — по образцу app6.py: сайдбар с секциями и help-текстами,
кэширование через st.cache_data, Plotly dark-theme график, session_state
для персистентности результатов между релогинами виджетов.

Каузальность: origin_offset обрезает целевой интервал по числу баров от конца;
все остальные источники пула (другие интервалы) обрезаются по дате (<= дата
последнего доступного бара целевого интервала) — единственная точка обрезки.

Run (из prototype/): streamlit run forcaster/ui/app8.py
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from forcaster.data.moex import download_candles, save_candles, INTERVALS

DATA_DIR = _root / "data" / "candles"

_PERIOD = {
    "1m": timedelta(minutes=1), "10m": timedelta(minutes=10),
    "1h": timedelta(hours=1),   "1w": timedelta(weeks=1),
    "1mo": timedelta(days=30),
}

# Кросс-тикерный пул (эксп.17 research/phase7_fractality/17_large_scale_pooled.py) —
# фиксированный универсум "голубых фишек" MOEX, без ручного отбора/корреляции
# (D_allpeers оказался как минимум не хуже отбора по корреляции — H3, эксп.17).
UNIVERSE = [
    "SBER", "LKOH", "CHMF", "NVTK", "MGNT", "VTBR", "NLMK", "MRKP", "GAZP", "PLZL",
    "ROSN", "TATN", "MTSS", "ALRS", "MOEX", "SNGS", "IRAO", "RUAL", "MAGN", "PHOR",
    "AFLT", "HYDR", "SIBN", "TRNFP", "RTKM",
    "BANE", "BANEP", "MTLR", "MTLRP", "RASP", "VSMO", "KMAZ", "AKRN", "MSNG", "TGKA",
    "UPRO", "PIKK", "MVID", "LSRG", "CBOM", "GCHE", "SVAV", "FESH", "KZOS", "NKNC",
]
VOLUME_PROFILE_BINS = 40

# ═══════════════════════════════════════════════════════════════════════════════
# Данные
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_candles(ticker: str, interval: str):
    path = DATA_DIR / ticker / f"{interval}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        if data:
            return data
    return None

def _fetch_and_save(ticker: str, interval: str):
    data = download_candles(ticker, interval)
    if data:
        save_candles(data, DATA_DIR / ticker / f"{interval}.json")
    return data

def _to_arrays(data):
    times   = np.array([c["begin"] for c in data])
    closes  = np.array([float(c["close"])  for c in data])
    opens   = np.array([float(c["open"])   for c in data])
    highs   = np.array([float(c["high"])   for c in data])
    lows    = np.array([float(c["low"])    for c in data])
    volumes = np.array([float(c.get("volume", 0.0)) for c in data])
    return times, opens, highs, lows, closes, volumes

# ═══════════════════════════════════════════════════════════════════════════════
# ZigZag (causal), из smap_ref.py — индексы вместо дат (для кэширования)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_zigzag_raw(log_highs: np.ndarray, log_lows: np.ndarray, threshold: float) -> tuple:
    """Каузальный зигзаг по логарифмическому порогу.

    Возвращает (log_pivot_prices, pivot_bar_idx, pivot_directions):
      pivot_bar_idx    — индекс бара подтверждения (не экстремума) в переданных массивах
      pivot_directions — +1 вершина (HIGH), −1 впадина (LOW)
    """
    prices: list[float] = []
    idxs:   list[int]   = []
    dirs:   list[int]   = []

    direction = 0
    ext_val = (log_highs[0] + log_lows[0]) / 2.0
    n = len(log_highs)

    for i in range(n):
        h, l = log_highs[i], log_lows[i]
        if direction == 0:
            if h - ext_val >= threshold:
                direction = 1; ext_val = h
            elif ext_val - l >= threshold:
                direction = -1; ext_val = l
        elif direction == 1:
            if h > ext_val:
                ext_val = h
            elif ext_val - l >= threshold:
                prices.append(ext_val); idxs.append(i); dirs.append(1)
                direction = -1; ext_val = l
        else:
            if l < ext_val:
                ext_val = l
            elif h - ext_val >= threshold:
                prices.append(ext_val); idxs.append(i); dirs.append(-1)
                direction = 1; ext_val = h

    return (np.array(prices), np.array(idxs, dtype=np.int64),
            np.array(dirs, dtype=np.int8))


@st.cache_data(show_spinner=False)
def _build_zigzag_cached(log_highs_bytes: bytes, log_lows_bytes: bytes,
                         threshold: float) -> tuple:
    log_highs = np.frombuffer(log_highs_bytes, dtype=np.float64)
    log_lows  = np.frombuffer(log_lows_bytes,  dtype=np.float64)
    return _build_zigzag_raw(log_highs, log_lows, threshold)


def build_zigzag(log_highs: np.ndarray, log_lows: np.ndarray, threshold: float) -> tuple:
    return _build_zigzag_cached(log_highs.tobytes(), log_lows.tobytes(), threshold)


# ── percentage-symmetric зигзаг (эксп.23) ────────────────────────────────────
#
# Стандартный порог (`ext−price>=T` вниз, `price−ext>=T` вверх, один и тот же
# T) симметричен в ЛОГ-единицах, но не в процентах: при T=0.20 разворот вниз
# от пика требует падения на 1−e^(−T)≈18.13%, разворот вверх от впадины —
# роста на e^T−1≈22.14%. Один номинальный порог требует разного реального
# движения цены в разные стороны. Percentage-symmetric порог выводит ДВА
# разных лог-порога (T_down, T_up) из ОДНОГО процента отката P_pct — тогда
# движение симметрично именно в процентах, не в логах.
#
# Эксп.23 (research/phase7_fractality/23_asymmetric_zigzag/): улучшение rMAE
# на 4 из 7 тикеров (SBER/LKOH/NVTK/MGNT), но НЕ универсальное (CHMF заметно
# хуже) — не заменяет стандарт, а добавлена здесь как альтернатива для
# ручного исследования.

def pct_to_log_thresholds(p_pct: float) -> tuple[float, float]:
    """P_pct (доля отката, одинаковая в обе стороны) -> (T_down, T_up) в лог-единицах."""
    t_down = -np.log(1 - p_pct)
    t_up = np.log(1 + p_pct)
    return t_down, t_up


def _build_zigzag_asym_raw(log_highs: np.ndarray, log_lows: np.ndarray,
                           t_down: float, t_up: float) -> tuple:
    """Тот же алгоритм, что _build_zigzag_raw, но t_down (подтверждение
    разворота ВНИЗ от пика) и t_up (подтверждение разворота ВВЕРХ от
    впадины) — разные пороги."""
    prices: list[float] = []
    idxs:   list[int]   = []
    dirs:   list[int]   = []

    direction = 0
    ext_val = (log_highs[0] + log_lows[0]) / 2.0
    n = len(log_highs)

    for i in range(n):
        h, l = log_highs[i], log_lows[i]
        if direction == 0:
            if h - ext_val >= t_up:
                direction = 1; ext_val = h
            elif ext_val - l >= t_down:
                direction = -1; ext_val = l
        elif direction == 1:
            if h > ext_val:
                ext_val = h
            elif ext_val - l >= t_down:
                prices.append(ext_val); idxs.append(i); dirs.append(1)
                direction = -1; ext_val = l
        else:
            if l < ext_val:
                ext_val = l
            elif h - ext_val >= t_up:
                prices.append(ext_val); idxs.append(i); dirs.append(-1)
                direction = 1; ext_val = h

    return (np.array(prices), np.array(idxs, dtype=np.int64),
            np.array(dirs, dtype=np.int8))


@st.cache_data(show_spinner=False)
def _build_zigzag_asym_cached(log_highs_bytes: bytes, log_lows_bytes: bytes,
                              t_down: float, t_up: float) -> tuple:
    log_highs = np.frombuffer(log_highs_bytes, dtype=np.float64)
    log_lows  = np.frombuffer(log_lows_bytes,  dtype=np.float64)
    return _build_zigzag_asym_raw(log_highs, log_lows, t_down, t_up)


def build_zigzag_asym(log_highs: np.ndarray, log_lows: np.ndarray,
                      t_down: float, t_up: float) -> tuple:
    return _build_zigzag_asym_cached(log_highs.tobytes(), log_lows.tobytes(), t_down, t_up)


def build_zigzag_unified(log_highs: np.ndarray, log_lows: np.ndarray,
                         threshold_frac: float, mode: str) -> tuple:
    """threshold_frac — доля (0.20 = 20%); интерпретация зависит от mode.
    mode="symmetric"  — стандартный лог-порог (threshold_frac = T напрямую).
    mode="asymmetric" — threshold_frac трактуется как P_pct, (T_down,T_up)
                        выводятся из него (pct_to_log_thresholds)."""
    if mode == "asymmetric":
        t_down, t_up = pct_to_log_thresholds(threshold_frac)
        return build_zigzag_asym(log_highs, log_lows, t_down, t_up)
    return build_zigzag(log_highs, log_lows, threshold_frac)


# ═══════════════════════════════════════════════════════════════════════════════
# Вложение и S-map, из smap_ref.py (вложение векторизовано)
# ═══════════════════════════════════════════════════════════════════════════════

def build_pool_vectors(log_pivot_prices: np.ndarray, pivot_directions: np.ndarray,
                       m: int) -> tuple:
    """
    Для каждого пивота j (m..n-2):
      feature[lag] = price[j-lag] - price[j-lag-1]   lag=0..m-1
      target       = price[j+1]  - price[j]
    """
    n = len(log_pivot_prices)
    valid = np.arange(m, n - 1)
    if len(valid) == 0:
        return (np.zeros((0, m)), np.zeros(0), np.zeros(0, dtype=np.int8))

    lag_idx = valid[:, None] - np.arange(m)[None, :]
    feat    = log_pivot_prices[lag_idx] - log_pivot_prices[lag_idx - 1]
    target  = log_pivot_prices[valid + 1] - log_pivot_prices[valid]
    dirs    = pivot_directions[valid]

    finite = np.all(np.isfinite(feat), axis=1) & np.isfinite(target)
    return feat[finite], target[finite], dirs[finite]


def build_query_vector(log_pivot_prices: np.ndarray, m: int) -> np.ndarray | None:
    n = len(log_pivot_prices)
    if n < m + 1:
        return None
    lags = np.arange(m)
    return log_pivot_prices[n - 1 - lags] - log_pivot_prices[n - 2 - lags]


def smap_weights(query_vector: np.ndarray, pool_feature_matrix: np.ndarray, theta: float) -> np.ndarray:
    """S-map (Sugihara 1994) веса: w_j = exp(-theta * d_j / mean(d))."""
    distances     = np.linalg.norm(pool_feature_matrix - query_vector, axis=1)
    mean_distance = distances.mean()
    if mean_distance < 1e-14:
        return np.ones(len(distances))
    return np.ones(len(distances)) if theta == 0 else np.exp(-theta * distances / mean_distance)


def smap_predict(query_vector: np.ndarray, pool_feature_matrix: np.ndarray,
                 pool_target_log_returns: np.ndarray, theta: float) -> float:
    """S-map (Sugihara 1994): w_j = exp(-theta * d_j / mean(d)); взвешенный МНК."""
    weights = smap_weights(query_vector, pool_feature_matrix, theta)
    sqrt_w  = np.sqrt(weights)
    design  = np.column_stack([np.ones(len(pool_target_log_returns)), pool_feature_matrix])
    coeffs, *_ = np.linalg.lstsq(design * sqrt_w[:, None],
                                  pool_target_log_returns * sqrt_w, rcond=None)
    return float(coeffs[0] + coeffs[1:] @ query_vector)


def knn_weights(query_vector: np.ndarray, pool_feature_matrix: np.ndarray, k: int) -> np.ndarray:
    """Бинарные веса: 1 для k ближайших по d_j = ‖x_q − x_pool[j]‖₂, иначе 0."""
    distances = np.linalg.norm(pool_feature_matrix - query_vector, axis=1)
    k_eff = max(1, min(k, len(distances)))
    nearest = np.argpartition(distances, k_eff - 1)[:k_eff]
    weights = np.zeros(len(distances))
    weights[nearest] = 1.0
    return weights


def knn_predict(query_vector: np.ndarray, pool_feature_matrix: np.ndarray,
                pool_target_log_returns: np.ndarray, k: int) -> float:
    """k ближайших соседей по d_j = ‖x_q − x_pool[j]‖₂; прогноз = среднее их
    целевых лог-доходностей. k=1 вырождается в прямое использование значения
    ближайшего соседа (среднее из одного элемента)."""
    weights = knn_weights(query_vector, pool_feature_matrix, k)
    return float(pool_target_log_returns[weights > 0].mean())


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> list[float]:
    """Взвешенные квантили values по весам weights (линейная интерполяция по
    взвешенной эмпирической CDF, midpoint-поправка)."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    if w.sum() < 1e-14:
        return [float(np.median(values))] * len(quantiles)
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return [float(x) for x in np.interp(quantiles, cw, v)]


# ═══════════════════════════════════════════════════════════════════════════════
# Мультипул: объединение нескольких зигзагов перед S-map
# ═══════════════════════════════════════════════════════════════════════════════

def combine_direction_pool(pool_sources: list[dict], direction: int,
                           min_pool: int) -> tuple:
    """
    Отбирает из каждого источника пула однонаправленные с запросом события
    и объединяет их в общую матрицу признаков / целевых лог-доходностей.

    pool_sources[i] = {"label", "features", "targets", "directions"}
    Возвращает (features, targets, breakdown) или (None, None, breakdown).
    breakdown — {label: n_matching} для отображения состава пула на шаге.
    """
    feats_list, tars_list, breakdown = [], [], {}
    for src in pool_sources:
        mask = src["directions"] == direction
        n_match = int(mask.sum())
        breakdown[src["label"]] = n_match
        if n_match:
            feats_list.append(src["features"][mask])
            tars_list.append(src["targets"][mask])

    total = sum(breakdown.values())
    if total < min_pool or not feats_list:
        return None, None, breakdown
    return np.vstack(feats_list), np.concatenate(tars_list), breakdown


QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


def run_forecast(query_log_prices: np.ndarray, query_direction: int,
                 pool_sources: list[dict], m: int, method: str, theta: float, k: int,
                 min_pool: int, steps: int,
                 quantile_levels: tuple[float, ...] = QUANTILE_LEVELS) -> list[dict]:
    """Итеративный многошаговый прогноз (направление чередуется по свойству зигзага).
    method="smap" — взвешенный МНК (Sugihara 1994); method="knn" — среднее
    целевых лог-доходностей k ближайших соседей (k=1 = прямое использование).

    На каждом шаге дополнительно считает взвешенные квантили целевых
    лог-доходностей ТОГО ЖЕ пула, что использовался для точки (S-map веса
    или бинарные k-NN веса) — «поле неопределённости»: разброс того, куда
    исторически вело плечо у аналогов текущего состояния, ветвящийся от уже
    посчитанной точки предыдущего шага (не полная композиция дисперсии по
    цепочке шагов — локальная оценка на каждом шаге отдельно)."""
    query_vector = build_query_vector(query_log_prices, m)
    if query_vector is None:
        return []

    direction      = query_direction
    cumulative_lr  = 0.0
    base_log_price = float(query_log_prices[-1])
    results: list[dict] = []

    for step in range(1, steps + 1):
        feats, tars, breakdown = combine_direction_pool(pool_sources, direction, min_pool)
        total_pool = sum(breakdown.values())
        if feats is None:
            results.append({
                "step": step, "ok": False, "direction": direction,
                "pool_total": total_pool, "breakdown": breakdown,
                "reason": f"пул {total_pool} < min_pool={min_pool}",
            })
            break

        weights = knn_weights(query_vector, feats, k) if method == "knn" \
            else smap_weights(query_vector, feats, theta)
        lr = knn_predict(query_vector, feats, tars, k) if method == "knn" \
            else smap_predict(query_vector, feats, tars, theta)
        q_lr = weighted_quantile(tars, weights, list(quantile_levels))

        cumulative_lr_prev = cumulative_lr
        cumulative_lr += lr
        price = float(np.exp(base_log_price + cumulative_lr))
        q_prices = {q: float(np.exp(base_log_price + cumulative_lr_prev + qlr))
                   for q, qlr in zip(quantile_levels, q_lr)}
        results.append({
            "step": step, "ok": True, "direction": direction,
            "log_return": lr, "cumulative_lr": cumulative_lr, "price": price,
            "pool_total": total_pool, "breakdown": breakdown,
            "quantiles": q_prices,
        })

        query_vector = np.concatenate([[lr], query_vector[:-1]])
        direction = -direction

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Каузальная обрезка
# ═══════════════════════════════════════════════════════════════════════════════

def trim_by_bars(arr_tuple: tuple, origin_offset: int) -> tuple:
    """Отбрасывает origin_offset последних баров (единственная точка обрезки для целевого интервала)."""
    if origin_offset <= 0:
        return arr_tuple
    cutoff = len(arr_tuple[0]) - origin_offset
    if cutoff <= 0:
        raise ValueError(f"origin_offset={origin_offset} >= числа баров ({len(arr_tuple[0])})")
    return tuple(a[:cutoff] for a in arr_tuple)


def trim_by_date(arr_tuple: tuple, dates: np.ndarray, cutoff_date: str) -> tuple:
    """Отбрасывает бары с датой > cutoff_date (для источников пула на других интервалах)."""
    mask = dates <= cutoff_date
    return tuple(a[mask] for a in arr_tuple)


def future_date(last_date: str, interval: str, bars_ahead: int) -> str:
    base = pd.Timestamp(last_date)
    if interval == "1d":
        return str(base + pd.offsets.BDay(bars_ahead))
    return str(base + _PERIOD.get(interval, timedelta(days=1)) * bars_ahead)


# ═══════════════════════════════════════════════════════════════════════════════
# Volume Profile — плотность сделок по цене (справа от графика)
# ═══════════════════════════════════════════════════════════════════════════════

def build_volume_profile(highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray,
                         n_bins: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Гистограмма объёма по цене: объём каждого бара размазывается РАВНОМЕРНО по
    [low, high] этого бара (приближение Volume Profile по дневным/интрадей
    свечам — тиковых данных о цене сделки нет). Работает в лог-цене для
    равномерных бинов, возвращает (price_bin_centers, volume_per_bin) в
    АБСОЛЮТНОЙ цене.

    Тот же приём, что research/phase7_fractality/20_price_levels/
    20b_volume_profile_check.py (эксп.20) — самостоятельная копия, чтобы
    прототип не зависел от research-скриптов.
    """
    log_h = np.log(np.maximum(highs, 1e-10))
    log_l = np.log(np.maximum(lows, 1e-10))
    lo, hi = np.nanmin(log_l), np.nanmax(log_h)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None

    edges = np.linspace(lo, hi, n_bins + 1)
    profile = np.zeros(n_bins)
    binw = (hi - lo) / n_bins
    for a, b, v in zip(log_l, log_h, volumes):
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(v)) or b <= a or v <= 0:
            continue
        b0 = int(np.floor((a - lo) / binw)); b1 = int(np.floor((b - lo) / binw))
        b0 = max(0, min(b0, n_bins - 1)); b1 = max(0, min(b1, n_bins - 1))
        if b0 == b1:
            profile[b0] += v
        else:
            span = b - a
            for k in range(b0, b1 + 1):
                bin_lo, bin_hi = lo + k * binw, lo + (k + 1) * binw
                overlap = max(0.0, min(b, bin_hi) - max(a, bin_lo))
                profile[k] += v * (overlap / span)

    bin_centers = np.exp(edges[:-1] + binw / 2.0)
    return bin_centers, profile


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="app8 · ZigZag S-map", layout="wide",
                   initial_sidebar_state="expanded")

with st.sidebar:
    st.title("app8 · ZigZag S-map")

    ticker   = st.text_input("Тикер", "SBER", key="ticker").upper().strip()
    interval = st.selectbox("Интервал (целевой)", list(INTERVALS.keys()),
                            index=list(INTERVALS.keys()).index("1d"),
                            help="1d — масштаб, на котором проверялся кросс-тикерный пул "
                                 "(эксп.17, полное покрытие данных по всем 45 тикерам). "
                                 "На других интервалах доступность данных у пиров ограничена.")

    st.divider()
    with st.expander("Алгоритм и обозначения"):
        st.markdown(
            "**Схема**\n\n"
            "```\n"
            "log(high), log(low) → каузальный зигзаг(T)\n"
            "\n"
            "T_query (целевой интервал) → вектор запроса x_q (m лог-доходностей плеч)\n"
            "                            → направление запроса (+1 верх / −1 низ)\n"
            "\n"
            "T_pool[1..k] (любые интервалы/пороги, ЦЕЛЕВОЙ тикер + кросс-тикерный\n"
            "  пул из 44 пиров MOEX, эксп.17) → объединённый пул {признаки, цель}\n"
            "  → фильтр по направлению (только однонаправленные с запросом)\n"
            "\n"
            "Метод предсказания — S-map ИЛИ KNN (переключатель в сайдбаре):\n"
            "\n"
            "S-map (Sugihara 1994):\n"
            "  d_j = ‖x_q − x_pool[j]‖₂,  w_j = exp(−θ·d_j / mean(d))\n"
            "  ŷ = взвешенная OLS(x_pool, target; w) → лог-доходность до след. пивота\n"
            "\n"
            "KNN:\n"
            "  d_j = ‖x_q − x_pool[j]‖₂\n"
            "  ŷ = среднее target по k ближайшим (k=1 → прямое значение соседа)\n"
            "\n"
            "Многошаговый прогноз (событийное время, не бары):\n"
            "  шаг h: x_q ← [ŷ_{h−1}, x_q[:-1]],  направление ← −направление\n"
            "```\n\n"
            "**Обозначения**\n\n"
            "| Символ | Описание |\n"
            "|:------:|----------|\n"
            "| T_query | порог зигзага целевого ряда (%, доля лог-цены) |\n"
            "| T_pool  | порог(и) зигзага пула соседей; можно несколько одновременно, "
            "в т.ч. на других интервалах — событийная фрактальность (фаза 7) |\n"
            "| m | размерность вложения — число последних плеч зигзага в векторе |\n"
            "| θ | (только S-map) параметр локализации: 0 = глобальная OLS, >0 = локальнее |\n"
            "| k | (только KNN) число ближайших соседей, усредняемых в прогноз |\n"
            "| min_pool | минимум однонаправленных событий пула для прогноза |\n"
            "| steps | число пивотов вперёд (каждый шаг — следующий пивот, направление чередуется) |\n"
            "| origin | точка отсчёта: обрезка целевого интервала на N баров от конца |\n\n"
            "**Каузальность**: origin_offset обрезает целевой интервал по барам; "
            "источники пула на других интервалах обрезаются по дате "
            "(≤ дата последнего доступного бара целевого интервала)."
        )

    st.divider()
    st.subheader("Тип порога зигзага")
    zigzag_mode = st.radio(
        "Порог", ["symmetric", "asymmetric"], horizontal=True,
        format_func=lambda m: "Симметричный (лог), стандарт" if m == "symmetric"
        else "Percentage-symmetric (эксп.23)",
        help="symmetric — один и тот же порог T в лог-единицах для разворота вверх и "
             "вниз (стандарт всей фазы 7). Он СИММЕТРИЧЕН В ЛОГАХ, но не в процентах: "
             "при T=0.20 разворот вниз от пика требует падения на 1−e^(−T)≈18.1%, "
             "разворот вверх от впадины — роста на e^T−1≈22.1%.  \n"
             "asymmetric (эксп.23) — вводит P_pct (один и тот же % отката в обе стороны) "
             "и выводит из него РАЗНЫЕ лог-пороги T_down/T_up. Улучшило rMAE на 4 из 7 "
             "проверенных тикеров (SBER/LKOH/NVTK/MGNT), но не универсально (CHMF — хуже) "
             "— не стандарт, а альтернатива для сравнения.  \n"
             "В обоих режимах ползунки ниже (T_query, T_pool) означают одно и то же число "
             "(0-30%) — при asymmetric оно трактуется как P_pct, а не как прямой T.")

    st.divider()
    st.subheader("Целевой зигзаг (T_query)")
    t_query_pct = st.slider("T_query, %", 0.5, 30.0, 20.0, 0.5,
                            help="Порог зигзага целевого ряда (при asymmetric — P_pct, "
                                 "процент отката в обе стороны). Задаёт вектор запроса "
                                 "и направление (+1 верх / −1 низ) следующего события. "
                                 "20% — «крупный масштаб» из эксп.17 (месячные события, 1d).")
    t_query = t_query_pct / 100.0
    if zigzag_mode == "asymmetric":
        _t_down, _t_up = pct_to_log_thresholds(t_query)
        st.caption(f"→ T_down={_t_down:.4f} (разворот вниз), T_up={_t_up:.4f} (разворот вверх) "
                   f"в лог-единицах")

    st.divider()
    st.subheader("Авто-пул по ratio (T_pool = ratio × T_query)")
    st.caption(
        "Фаза 7: T_pool ≈ 0.85-0.90 × T_query на том же интервале (для SBER откалибровано "
        "0.8987, эксп.17f). Живой авто-режим: пересчитывается на каждое изменение T_query "
        "(ratio индивидуален на тикер — см. память проекта, при смене тикера подбирайте заново)."
    )
    auto_ratio_on = st.checkbox("Авто-пул включён (тот же интервал, что и целевой)", value=True)
    ratio = st.number_input("ratio (T_pool / T_query)", min_value=0.1, max_value=1.0,
                            value=0.8987, step=0.0001, format="%.4f", disabled=not auto_ratio_on)
    auto_t_pool_pct = round(t_query_pct * ratio, 2)
    if auto_ratio_on:
        st.caption(f"→ авто T_pool = {auto_t_pool_pct:.2f}% на интервале {interval} "
                   f"(+ кросс-тикерный пул {len(UNIVERSE) - 1} пиров)")

    st.divider()
    st.subheader("Доп. пулы соседей (вручную)")
    st.caption(
        "Каждая строка — НЕЗАВИСИМЫЙ от авто-пула зигзаг-источник (интервал + порог), "
        "например другой интервал (событийная фрактальность, фаза 7) или ручной T_pool "
        "без привязки к ratio. Для КАЖДОЙ строки автоматически добавляется кросс-тикерный "
        f"пул (все {len(UNIVERSE) - 1} пиров MOEX того же интервала/порога, эксп.17)."
    )
    default_pools = pd.DataFrame(columns=["Интервал", "T (%)", "Вкл"])
    pool_df = st.data_editor(
        st.session_state.get("app8_pool_df", default_pools),
        num_rows="dynamic",
        use_container_width=True,
        key="app8_pool_editor",
        column_config={
            "Интервал": st.column_config.SelectboxColumn(options=list(INTERVALS.keys()), required=True),
            "T (%)":    st.column_config.NumberColumn(min_value=0.1, max_value=30.0, step=0.1, required=True),
            "Вкл":      st.column_config.CheckboxColumn(default=True),
        },
    )
    st.session_state["app8_pool_df"] = pool_df

    if st.button("Обновить данные (целевой тикер, все интервалы)"):
        used_intervals = {interval} | set(pool_df["Интервал"].dropna().tolist())
        for iv in used_intervals:
            _fetch_and_save(ticker, iv)
        _load_candles.clear()
        st.rerun()

    if st.button("Обновить данные пиров (кросс-тикерный пул, эксп.17)",
                 help="Докачивает данные для всех тикеров UNIVERSE, у которых ещё нет "
                      "локального кэша на нужный интервал (не трогает уже скачанное — "
                      "быстрее при повторных запусках). Может занять несколько минут "
                      "на 44 тикера."):
        used_intervals = {interval} | set(pool_df["Интервал"].dropna().tolist())
        jobs = [(peer, iv) for iv in used_intervals for peer in UNIVERSE
                if peer != ticker and not (DATA_DIR / peer / f"{iv}.json").exists()]
        progress = st.progress(0.0, text="Скачивание пиров…")
        for i, (peer, iv) in enumerate(jobs, start=1):
            progress.progress(i / len(jobs) if jobs else 1.0,
                              text=f"{peer} [{iv}] ({i}/{len(jobs)})")
            _fetch_and_save(peer, iv)
        progress.empty()
        _load_candles.clear()
        st.rerun()

    st.divider()
    st.subheader("Модель прогноза")
    m = st.slider("m (размерность вложения)", 1, 100, 3,
                 help="Число последних плеч зигзага в векторе запроса/признаков. "
                      "Больше m — длиннее память, но меньше валидных векторов в пуле. "
                      "Используется обоими методами (S-map и KNN) — задаёт вектор, "
                      "по которому считается расстояние до соседей.")

    method = st.radio(
        "Метод", ["smap", "knn"], horizontal=True,
        format_func=lambda x: "S-map (стандарт, фаза 7)" if x == "smap" else "KNN (k ближайших соседей)",
        help="S-map — взвешенный МНК по всему пулу (Sugihara 1994). "
             "KNN — прогноз = среднее целевых лог-доходностей k ближайших по "
             "d_j=‖x_q−x_pool[j]‖₂ соседей; при k=1 это прямое использование "
             "значения ближайшего соседа (среднее из одного элемента).")

    if method == "smap":
        theta = st.slider("θ (локализация S-map)", 0.0, 100.0, 25.7, 0.1,
                          help="w_j = exp(−θ·d_j/mean(d)). θ=0 — глобальная OLS (все веса равны), "
                               "θ→∞ — только ближайший сосед. По умолчанию — калибровка SBER "
                               "(эксп.17f: m=3 θ=25.697 T_ratio=0.8987) — при смене тикера подбирайте заново, "
                               "калибровка индивидуальна на тикер.")
        k = 1
    else:
        k = st.slider("k (число соседей)", 1, 200, 5,
                     help="Сколько ближайших по расстоянию событий пула усредняется. "
                          "k=1 — прямое использование значения ближайшего соседа, без усреднения.")
        theta = 0.0

    auto_min_pool = st.checkbox(f"min_pool = {'m + 2' if method == 'smap' else 'k'} (авто)", value=True)
    _auto_min_pool = (m + 2) if method == "smap" else k
    min_pool = _auto_min_pool if auto_min_pool else st.slider(
        "min_pool (вручную)", 1, 2000, _auto_min_pool,
        help="Минимум однонаправленных событий в объединённом пуле для прогноза шага.")

    st.divider()
    st.subheader("Прогноз")
    steps = st.slider("Шагов вперёд (пивотов)", 1, 500, 5,
                      help="Каждый шаг — следующий пивот зигзага (не фиксированное число баров). "
                           "Направление чередуется автоматически.")
    origin_offset = st.slider("Точка отсчёта (баров от конца, целевой интервал)", 0, 2000, 0,
                              help="0 — прогноз из последнего доступного бара. "
                                   ">0 — бэктест: на графике/в таблице показываются "
                                   "фактические последующие пивоты для сравнения.")
    show_bars = st.slider("Баров на графике", 100, 3000, 400, 50)
    show_uncertainty = st.checkbox("Поле неопределённости (квантили пула по шагам)", value=True,
                                   help="На каждом шаге — взвешенные квантили целевых лог-доходностей "
                                        "того же пула соседей, что дал точку (10-90% и 25-75%). "
                                        "Ветвится от точки прогноза предыдущего шага — разброс "
                                        "ОДНОГО шага, не накопленная неопределённость всей цепочки.")
    show_pool_zigzags = st.checkbox("Показать зигзаги пула на графике (тот же интервал)", value=True)
    show_volume_profile = st.checkbox("Индикатор плотности сделок справа (Volume Profile)", value=True,
                                      help="Гистограмма объёма по цене за отображаемое окно — "
                                           "объём каждого бара размазан по [low,high] этого бара.")

    run_btn = st.button("▶  Прогноз", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Ручная коррекция амплитуды")
    amp_multiplier = st.slider(
        "Множитель амплитуды прогноза", 0.0, 3.0, 1.0, 0.05,
        help="Один общий множитель на ВЕСЬ прогноз (все шаги сразу) — растягивает/сжимает "
             "предсказанную траекторию для визуального поиска закономерностей, не видных в цифрах "
             "(эксп.20: модель систематически недооценивает крупные движения). Не требует "
             "повторного расчёта S-map, применяется мгновенно к уже посчитанному прогнозу.")

# ── Загрузка целевых данных ────────────────────────────────────────────────────

data = _load_candles(ticker, interval)
if not data:
    st.info("Нет данных. Нажмите «Обновить данные (все интервалы)» в сайдбаре.")
    st.stop()

times, opens, highs, lows, closes, volumes = _to_arrays(data)
n_total = len(times)
if origin_offset >= n_total:
    st.error(f"origin_offset={origin_offset} >= числа баров ({n_total})")
    st.stop()

log_highs_full = np.log(np.maximum(highs, 1e-10))
log_lows_full  = np.log(np.maximum(lows,  1e-10))

# ── Каузальная обрезка целевого интервала (единственная точка обрезки) ────────
cutoff = n_total - origin_offset
times_c, opens_c, highs_c, lows_c, closes_c, volumes_c = trim_by_bars(
    (times, opens, highs, lows, closes, volumes), origin_offset)
log_highs_c = log_highs_full[:cutoff]
log_lows_c  = log_lows_full[:cutoff]
cutoff_date = str(times_c[-1])

# ── Целевой зигзаг T_query ─────────────────────────────────────────────────────
q_prices, q_idx, q_dirs = build_zigzag_unified(log_highs_c, log_lows_c, t_query, zigzag_mode)

st.subheader(f"{ticker} · {interval} · S-map на зигзаге (событийное время)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Пивотов T_query", len(q_prices))
c2.metric("Направление запроса", "▲ HIGH" if (len(q_dirs) and q_dirs[-1] == 1) else ("▼ LOW" if len(q_dirs) else "—"))
c3.metric("Точка отсчёта", cutoff_date[:16])
c4.metric("m / метод / min_pool", f"{m} / {method}·{theta if method == 'smap' else k} / {min_pool}")

if len(q_prices) < m + 1:
    st.error(f"Недостаточно пивотов T_query для m={m}: есть {len(q_prices)}, нужно ≥ {m+1}. "
             f"Уменьшите T_query или m.")
    st.stop()

# ── Построение пулов соседей ───────────────────────────────────────────────────

def _build_pool_source(src_ticker: str, p_interval: str, p_threshold: float, cutoff_date: str, m: int):
    """(feats, tars, dirs, n_pivots) для одного источника (тикер×интервал×порог),
    обрезанного по дате <= cutoff_date. None, если данных недостаточно."""
    p_data = _load_candles(src_ticker, p_interval)
    if not p_data:
        return None
    p_times, _, p_highs, p_lows, _, _ = _to_arrays(p_data)
    p_log_h = np.log(np.maximum(p_highs, 1e-10))
    p_log_l = np.log(np.maximum(p_lows, 1e-10))
    (p_times_c, p_log_h_c, p_log_l_c) = trim_by_date((p_times, p_log_h, p_log_l), p_times, cutoff_date)
    if len(p_times_c) < 3:
        return None
    pp_prices, pp_idx, pp_dirs = build_zigzag_unified(p_log_h_c, p_log_l_c, p_threshold, zigzag_mode)
    feats, tars, dirs = build_pool_vectors(pp_prices, pp_dirs, m)
    return feats, tars, dirs, len(pp_prices)


pool_sources: list[dict] = []
pool_table_rows: list[dict] = []


def _add_pool_entry(p_interval: str, p_t_pct: float, tag: str = "") -> None:
    """Добавляет один источник пула (целевой тикер + кросс-тикерный пул, эксп.17)
    в pool_sources/pool_table_rows для (p_interval, p_t_pct). tag — пометка в
    таблице «Источники пула» (например, «авто»)."""
    p_threshold = float(p_t_pct) / 100.0
    suffix = f" [{tag}]" if tag else ""

    # ── целевой тикер на этом (интервал, T) ──
    res = _build_pool_source(ticker, p_interval, p_threshold, cutoff_date, m)
    if res is None:
        pool_table_rows.append({"Тикер": ticker + suffix, "Интервал": p_interval, "T (%)": p_t_pct,
                                "Пивотов": 0, "Векторов": 0, "Статус": "нет данных/мало баров"})
    else:
        feats, tars, dirs, n_piv = res
        pool_sources.append({"label": f"{ticker} {p_interval}·T={p_t_pct:.2f}%{suffix}",
                             "features": feats, "targets": tars, "directions": dirs})
        pool_table_rows.append({"Тикер": ticker + suffix, "Интервал": p_interval, "T (%)": p_t_pct,
                                "Пивотов": n_piv, "Векторов": len(tars), "Статус": "ok"})

    # ── кросс-тикерный пул (эксп.17, D_allpeers) — все пиры универсума на том же (интервал, T),
    #    объединены в один источник (иначе разбивка по шагам захламляется 44 колонками) ──
    peer_feats, peer_tars, peer_dirs = [], [], []
    n_peers_ok = 0
    for peer in UNIVERSE:
        if peer == ticker:
            continue
        res = _build_pool_source(peer, p_interval, p_threshold, cutoff_date, m)
        if res is None:
            continue
        f, t, d, _ = res
        peer_feats.append(f); peer_tars.append(t); peer_dirs.append(d)
        n_peers_ok += 1
    if peer_tars:
        pool_sources.append({
            "label": f"кросс-тикер {p_interval}·T={p_t_pct:.2f}%{suffix}",
            "features": np.vstack(peer_feats), "targets": np.concatenate(peer_tars),
            "directions": np.concatenate(peer_dirs),
        })
    pool_table_rows.append({
        "Тикер": f"кросс-тикер ({n_peers_ok}/{len(UNIVERSE) - 1} пиров){suffix}",
        "Интервал": p_interval, "T (%)": p_t_pct, "Пивотов": "—",
        "Векторов": sum(len(t) for t in peer_tars), "Статус": "ok" if peer_tars else "нет данных у пиров",
    })


if auto_ratio_on:
    _add_pool_entry(interval, auto_t_pool_pct, tag=f"авто ratio={ratio:.4f}")

for _, row in pool_df.iterrows():
    if not bool(row.get("Вкл", True)):
        continue
    p_interval = row.get("Интервал")
    p_t_pct    = row.get("T (%)")
    if not p_interval or p_t_pct is None:
        continue
    _add_pool_entry(p_interval, float(p_t_pct))

st.markdown("**Источники пула** (целевой тикер + кросс-тикерный пул MOEX, эксп.17)")
st.dataframe(pd.DataFrame(pool_table_rows), use_container_width=True, hide_index=True)

if not pool_sources or sum(len(s["targets"]) for s in pool_sources) == 0:
    st.warning("Нет валидных источников пула. Включите «Авто-пул» или добавьте строку "
              "в таблицу «Доп. пулы соседей».")
    st.stop()

# ── Прогноз (сохраняется в session_state, персистентен между релогинами виджетов) ──

if run_btn:
    query_direction = int(q_dirs[-1])
    with st.spinner(f"{method.upper()} прогноз: {steps} шагов…"):
        fc_results = run_forecast(q_prices, query_direction, pool_sources,
                                  m, method, theta, k, min_pool, steps)
    st.session_state.update({
        "app8_fc": fc_results,
        "app8_origin": origin_offset,
        "app8_cutoff_date": cutoff_date,
        "app8_last_pivot_date": str(times_c[q_idx[-1]]),
        "app8_last_pivot_price": float(np.exp(q_prices[-1])),
        "app8_interval": interval,
        "app8_median_gap": int(np.median(np.diff(q_idx))) if len(q_idx) > 1 else 1,
    })

# ── Отображение ─────────────────────────────────────────────────────────────────

if "app8_fc" in st.session_state and st.session_state.get("app8_interval") == interval:
    fc_results        = st.session_state["app8_fc"]
    last_pivot_date   = st.session_state["app8_last_pivot_date"]
    last_pivot_price  = st.session_state["app8_last_pivot_price"]
    median_gap        = max(1, st.session_state["app8_median_gap"])

    ok_results = [r for r in fc_results if r["ok"]]
    if not ok_results:
        st.warning("Ни один шаг прогноза не выполнен — пул слишком мал на первом же шаге.")
    else:
        # ── фактические последующие пивоты (только для бэктеста, origin_offset>0) ──
        actual_rows: list[dict] = []
        if origin_offset > 0:
            fa_prices, fa_idx, fa_dirs = build_zigzag_unified(log_highs_full, log_lows_full, t_query, zigzag_mode)
            future_mask = fa_idx > (q_idx[-1] if len(q_idx) else -1)
            # выравниваем по количеству фактических пивотов, найденных на полном ряду после origin
            fut_prices = fa_prices[future_mask][:len(ok_results)]
            fut_idx    = fa_idx[future_mask][:len(ok_results)]
            for k, (fp, fi) in enumerate(zip(fut_prices, fut_idx), start=1):
                actual_rows.append({"step": k, "actual_price": float(np.exp(fp)),
                                    "actual_date": str(times[fi])})

        # ── таблица шагов ────────────────────────────────────────────────────────
        table_rows = []
        for r in fc_results:
            row = {
                "Шаг": r["step"],
                "Направление": "▲" if r["direction"] > 0 else "▼",
                "Пул (n)": r["pool_total"],
            }
            if r["ok"]:
                row["Лог-доходность"] = round(r["log_return"], 5)
                row["Цена (прогноз)"] = round(r["price"], 4)
                q = r.get("quantiles", {})
                if q:
                    row["10%"] = round(q.get(0.1), 4) if 0.1 in q else None
                    row["25%"] = round(q.get(0.25), 4) if 0.25 in q else None
                    row["75%"] = round(q.get(0.75), 4) if 0.75 in q else None
                    row["90%"] = round(q.get(0.9), 4) if 0.9 in q else None
            else:
                row["Лог-доходность"] = None
                row["Цена (прогноз)"] = None
                row["Причина остановки"] = r["reason"]
            match = next((a for a in actual_rows if a["step"] == r["step"]), None)
            if match:
                row["Цена (факт)"] = round(match["actual_price"], 4)
                if r["ok"]:
                    row["Ошибка, %"] = round(
                        (r["price"] / match["actual_price"] - 1) * 100, 3)
            table_rows.append(row)

        st.markdown("**Прогноз по шагам**")
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        with st.expander("Состав пула по шагам (breakdown по источникам)"):
            bd_rows = []
            for r in fc_results:
                bd = {"Шаг": r["step"], "Направление": "▲" if r["direction"] > 0 else "▼"}
                bd.update(r["breakdown"])
                bd_rows.append(bd)
            st.dataframe(pd.DataFrame(bd_rows), use_container_width=True, hide_index=True)

        # ── график ──────────────────────────────────────────────────────────────
        show_from = max(0, cutoff - show_bars)

        vp = None
        if show_volume_profile:
            vp = build_volume_profile(highs_c[show_from:], lows_c[show_from:],
                                      volumes_c[show_from:], VOLUME_PROFILE_BINS)

        if vp is not None:
            fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                                column_widths=[0.84, 0.16], horizontal_spacing=0.01)
            _main = dict(row=1, col=1)
        else:
            fig = go.Figure()
            _main = {}

        fig.add_trace(go.Candlestick(
            x=times_c[show_from:], open=opens_c[show_from:], high=highs_c[show_from:],
            low=lows_c[show_from:], close=closes_c[show_from:], name="OHLC",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), **_main)

        # зигзаг T_query (в пределах отображаемого окна)
        zz_mask = q_idx >= show_from
        if zz_mask.any():
            fig.add_trace(go.Scatter(
                x=times_c[q_idx[zz_mask]], y=np.exp(q_prices[zz_mask]),
                mode="lines+markers", name=f"зигзаг T_query={t_query_pct:.1f}%",
                line=dict(color="#42a5f5", width=1.5), marker=dict(size=5),
            ), **_main)

        # зигзаги пула того же интервала (тонкие, справочно)
        if show_pool_zigzags:
            same_interval_pcts = []
            if auto_ratio_on:
                same_interval_pcts.append(auto_t_pool_pct)
            for _, row in pool_df.iterrows():
                if not bool(row.get("Вкл", True)) or row.get("Интервал") != interval:
                    continue
                if row.get("T (%)") is not None:
                    same_interval_pcts.append(float(row.get("T (%)")))
            for pool_pct in same_interval_pcts:
                pt = pool_pct / 100.0
                if pt <= 0:
                    continue
                pp, pidx, _ = build_zigzag_unified(log_highs_c, log_lows_c, pt, zigzag_mode)
                pmask = pidx >= show_from
                if pmask.any():
                    fig.add_trace(go.Scatter(
                        x=times_c[pidx[pmask]], y=np.exp(pp[pmask]),
                        mode="lines", name=f"пул T={pool_pct:.2f}%",
                        line=dict(color="rgba(255,180,60,0.35)", width=1),
                    ), **_main)

        # прогноз: от последнего пивота через шаги (нужен номинальный шаг по датам —
        # событийное время не даёт точной календарной привязки, это только визуализация)
        fc_x = [last_pivot_date] + [
            future_date(last_pivot_date, interval, median_gap * r["step"]) for r in fc_results
        ]
        fc_y = [last_pivot_price] + [r["price"] if r["ok"] else np.nan for r in fc_results]
        method_label = "S-map" if method == "smap" else f"KNN(k={k})"

        # поле неопределённости: взвешенные квантили пула на каждом шаге,
        # две вложенные полосы (10-90% и 25-75%) вокруг точки прогноза
        if show_uncertainty:
            def _band(lo_q: float, hi_q: float) -> tuple[list, list]:
                lo = [last_pivot_price] + [
                    r["quantiles"].get(lo_q, np.nan) if r["ok"] else np.nan for r in fc_results]
                hi = [last_pivot_price] + [
                    r["quantiles"].get(hi_q, np.nan) if r["ok"] else np.nan for r in fc_results]
                return lo, hi

            for lo_q, hi_q, opacity, label in [(0.1, 0.9, 0.12, "10-90%"), (0.25, 0.75, 0.20, "25-75%")]:
                band_lo, band_hi = _band(lo_q, hi_q)
                fig.add_trace(go.Scatter(
                    x=fc_x, y=band_lo, mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ), **_main)
                fig.add_trace(go.Scatter(
                    x=fc_x, y=band_hi, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=f"rgba(255,214,0,{opacity})",
                    name=f"поле неопределённости {label}", hoverinfo="skip",
                ), **_main)

        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_y, mode="lines+markers", name=f"{method_label} прогноз ({steps} шагов)",
            line=dict(color="#ffd600", width=2.5, dash="dash"), marker=dict(size=6),
        ), **_main)

        # ручная коррекция амплитуды — один общий множитель на весь прогноз (не пересчитывает
        # S-map, растягивает/сжимает уже посчитанную траекторию для визуального анализа)
        if amp_multiplier != 1.0:
            adj_cum = 0.0
            adj_y = [last_pivot_price]
            for r in fc_results:
                if not r["ok"]:
                    adj_y.append(np.nan)
                    continue
                adj_cum += r["log_return"] * amp_multiplier
                adj_y.append(float(np.exp(np.log(last_pivot_price) + adj_cum)))
            fig.add_trace(go.Scatter(
                x=fc_x, y=adj_y, mode="lines+markers",
                name=f"ручная коррекция ×{amp_multiplier:.2f}",
                line=dict(color="#ff7043", width=2, dash="dashdot"), marker=dict(size=5),
            ), **_main)

        # факт (только бэктест)
        if actual_rows:
            act_x = [last_pivot_date] + [a["actual_date"] for a in actual_rows]
            act_y = [last_pivot_price] + [a["actual_price"] for a in actual_rows]
            fig.add_trace(go.Scatter(
                x=act_x, y=act_y, mode="lines+markers", name="факт (бэктест)",
                line=dict(color="#ffffff", width=1.5, dash="dot"), marker=dict(size=5),
            ), **_main)

        fig.add_vline(
            x=cutoff_date, line_width=1.5, line_dash="dash", line_color="#ffffff",
            annotation_text=f"origin −{origin_offset}б" if origin_offset > 0 else "origin",
            annotation_position="top left", **_main,
        )

        if vp is not None:
            bin_centers, vp_profile = vp
            fig.add_trace(go.Bar(
                x=vp_profile, y=bin_centers, orientation="h", name="плотность сделок",
                marker=dict(color="rgba(120,170,255,0.55)"), showlegend=False,
            ), row=1, col=2)
            fig.update_xaxes(title_text="объём", row=1, col=2, showgrid=False)
            fig.update_yaxes(showticklabels=False, row=1, col=2)

        fig.update_layout(
            height=600, xaxis_rangeslider_visible=False,
            title=f"{ticker} {interval} · {zigzag_mode} · T_query={t_query_pct:.1f}% · "
                  f"m={m} · {method_label} · {steps} шагов вперёд",
            template="plotly_dark", legend=dict(orientation="h", y=-0.18),
        )
        fig.update_xaxes(type="date", **_main)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Прогнозные точки на графике расставлены с номинальным шагом "
            f"= медианный интервал между пивотами T_query ({median_gap} бар.) — "
            "это визуализация, реальный момент следующего пивота заранее не определён "
            "(событийное, не календарное время). Индикатор плотности сделок справа — объём за "
            "отображаемое окно графика, размазанный по [low,high] каждого бара (не тиковые данные)."
        )
else:
    st.info("Настройте параметры и нажмите «▶ Прогноз» в сайдбаре.")
