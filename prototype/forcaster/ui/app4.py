"""
app4: W_max адаптивный p_fit — Theiler + Local Projective + ACF.

Алгоритм:
  Сигнал: dratio = diff(ratio),  ratio = close / logtrend_causal(close).
  1. P_search=64 (фикс.). Скан W от W_min до P_search (шаг W_step).
  2. При каждом W: Theiler-пул K соседей в 64D dratio → LP (n_lp итераций) →
     d_eff (SVD @ d_thresh) → ACF lag-1 критерий.
     ACF: CI95 = 1.96/√(K·P_ref),  P_ref = max(7, 2·d_after+1).
  3. W_max = наибольшее W, где критерий проходит. d_local = d_after при W_max.
  4. p_fit = 2·d_local + margin (настраиваемый margin).
  5. Прогноз: W_max пул → рефайн до ξ в p_fit-пространстве (dratio) → LWR + acc_ang.
     Реконструкция: ratio[origin] + cumsum(dratio_pred) → price.
     Опционально: LP-коррекция каждого шага.

Ключевое отличие от lp_wmax_visual.py: всё в пространстве dratio, не ratio.
Это устраняет проблему «ложных соседей по уровню» при LWR.

Run (из prototype/):  streamlit run forcaster/ui/app4.py
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
from scipy.spatial import KDTree

from forcaster.data.moex import download_candles, save_candles, INTERVALS

_ROOT    = Path(__file__).parent.parent.parent
DATA_DIR = _ROOT / "data" / "candles"

_BAR_DELTA: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1), "10m": pd.Timedelta(minutes=10),
    "1h": pd.Timedelta(hours=1),   "1d":  pd.Timedelta(days=1),
    "1w": pd.Timedelta(weeks=1),   "1mo": pd.Timedelta(days=30),
}


# ══════════════════════════════════════════════════════════════════════════════
# Алгоритмические функции
# ══════════════════════════════════════════════════════════════════════════════

def _logtrend_causal(close: np.ndarray) -> np.ndarray:
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t); trend[:2] = close[:2]
    return trend


def _svd_d_local(X: np.ndarray, thresh: float) -> tuple[int, np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0)
    _, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    vf = sv ** 2 / (sv ** 2).sum(); cv = np.cumsum(vf)
    d  = int(np.searchsorted(cv, thresh)) + 1
    return d, sv, cv


def _lp_step(X: np.ndarray, d: int, k_prime: int) -> np.ndarray:
    k_eff = min(k_prime, len(X) - 1)
    d_eff = min(d, k_eff - 1)
    _, ai  = KDTree(X).query(X, k=k_eff + 1)
    X_new  = np.empty_like(X)
    for i in range(len(X)):
        nn = ai[i, 1:]; Xnn = X[nn]; cent = Xnn.mean(0)
        _, _, Vt = np.linalg.svd(Xnn - cent, full_matrices=False)
        Vd = Vt[:d_eff]; xc = X[i] - cent
        X_new[i] = cent + Vd.T @ (Vd @ xc)
    return X_new


def _lp_clean(X_pool: np.ndarray, d: int, k_prime: int, n_iter: int) -> np.ndarray:
    X = X_pool.copy()
    for _ in range(n_iter):
        X = _lp_step(X, d, k_prime)
    return X


def _acf1_score(
    X_orig: np.ndarray, X_clean: np.ndarray, d_after: int,
) -> tuple[float, float, int, bool]:
    """Signed mean ACF lag-1 остатков. Возвращает (abs_acf1, CI95, P_ref, ok)."""
    P_ref = max(7, 2 * d_after + 1)
    resid = (X_orig - X_clean)[:, :P_ref]
    K     = len(X_orig)
    CI95  = 1.96 / np.sqrt(max(K * P_ref, 1))
    vals  = 0.0; cnt = 0
    for k in range(K):
        r = resid[k] - resid[k].mean()
        c0 = np.dot(r, r)
        if c0 < 1e-30 or P_ref < 2:
            continue
        vals += np.dot(r[:-1], r[1:]) / c0; cnt += 1
    acf1 = vals / max(cnt, 1)
    return abs(acf1), CI95, P_ref, abs(acf1) <= CI95


def _build_pool(
    signal: np.ndarray, t_orig: int, P: int, K: int, W: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """K ближайших соседей (конец окна) в P-embedding с Theiler |t_end - t_orig| ≥ W.

    signal = dratio (индексы: 0..N-2, где dratio[t] = ratio[t+1]-ratio[t]).
    t_orig = последний известный индекс в dratio (= origin_k - 1).
    """
    past_ends = np.arange(P - 1, t_orig, dtype=int)
    mask      = past_ends <= t_orig - W
    if mask.sum() < K:
        return None, None
    cand_ends = past_ends[mask]
    X_cands   = np.array([signal[t - P + 1: t + 1] for t in cand_ends])
    x_q       = signal[t_orig - P + 1: t_orig + 1]
    _, inds   = KDTree(X_cands).query(x_q.reshape(1, -1), k=K)
    return cand_ends[inds[0]], X_cands[inds[0]]


@st.cache_data(show_spinner=False)
def _search_wmax_cached(
    dratio_bytes: bytes, t_orig_d: int,
    P_search: int, K_search: int, K_prime: int, N_lp: int,
    d_thresh: float, W_min: int, W_step: int,
) -> dict:
    """Полный скан W → W_max. Кэшируется по (dratio, t_orig_d, параметрам).

    dratio_bytes: np.diff(ratio[:origin_k+1]).tobytes()
    t_orig_d:     origin_k - 1  (последний известный индекс dratio)
    """
    dratio = np.frombuffer(dratio_bytes, dtype=np.float64).copy()
    w_grid = list(range(W_min, P_search + 1, W_step))
    history: list[dict] = []
    W_max = None; d_local = None

    for W in w_grid:
        pool_times, X_pool = _build_pool(dratio, t_orig_d, P_search, K_search, W)
        if pool_times is None:
            history.append(dict(W=W, ok=False, n_cands=0,
                                d_before=np.nan, d_after=np.nan,
                                P_ref=np.nan, acf1=np.nan, CI95=np.nan,
                                pool_times=None, X_pool=None, X_clean=None))
            continue

        d_before, _, _ = _svd_d_local(X_pool, d_thresh)
        X_clean        = _lp_clean(X_pool, d_before, K_prime, N_lp)
        d_after, _, _  = _svd_d_local(X_clean, d_thresh)
        acf1, CI95, P_ref, ok = _acf1_score(X_pool, X_clean, d_after)

        history.append(dict(
            W=W, ok=ok, n_cands=len(pool_times),
            d_before=d_before, d_after=d_after,
            P_ref=P_ref, acf1=acf1, CI95=CI95,
            pool_times=pool_times.copy(),
            X_pool=X_pool.copy(),
            X_clean=X_clean.copy(),
        ))
        if ok:
            W_max = W; d_local = d_after

    return dict(history=history, W_max=W_max, d_local=d_local)


def _lwr_one_step(
    x_q: np.ndarray, X_c: np.ndarray, Y_c: np.ndarray,
    xi: int, lam: float = 0.01,
) -> float:
    d_pos = np.linalg.norm(X_c - x_q, axis=1)
    if len(x_q) >= 3 and lam > 0.0:
        acc_q  = x_q[-1] - 2 * x_q[-2] + x_q[-3]
        acc_c  = X_c[:, -1] - 2 * X_c[:, -2] + X_c[:, -3]
        d_comb = d_pos + lam * np.abs(acc_c - acc_q)
    else:
        d_comb = d_pos
    n_keep = min(xi, len(X_c))
    order  = np.argsort(d_comb)[:n_keep]
    Xs, Ys, ds = X_c[order], Y_c[order], d_comb[order]
    h = np.median(ds) + 1e-10
    w = np.exp(-0.5 * (ds / h) ** 2)
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    sw = np.sqrt(w)
    try:
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * Xd, sw * Ys, rcond=None)
        return float(c[0] + x_q @ c[1:])
    except Exception:
        return float(np.average(Ys, weights=w))


def _lp_corr_step(
    v: np.ndarray, X_lib_pfit: np.ndarray, d_local: int, k_corr: int,
) -> float:
    """Проецирует вектор v (длина p_fit) на d_local-мерную плоскость аттрактора."""
    k_eff = min(k_corr, len(X_lib_pfit) - 1)
    d_eff = min(d_local, k_eff - 1)
    if d_eff < 1:
        return float(v[-1])
    dists = np.linalg.norm(X_lib_pfit - v, axis=1)
    idx   = np.argpartition(dists, k_eff)[:k_eff]
    Xnn   = X_lib_pfit[idx]; cent = Xnn.mean(0)
    _, _, Vt = np.linalg.svd(Xnn - cent, full_matrices=False)
    Vd = Vt[:d_eff]; xc = v - cent
    return float((cent + Vd.T @ (Vd @ xc))[-1])


def _forecast_wmax(
    dratio: np.ndarray, t_orig_d: int, horizon: int,
    pool_times: np.ndarray, X_clean: np.ndarray,
    p_fit: int, d_local: int,
    acc_lambda: float = 0.01,
    use_lp_corr: bool = False,
    k_corr: int = 10,
) -> np.ndarray:
    """
    Итеративный прогноз dratio[t_orig_d+1 .. t_orig_d+horizon].

    dratio:     np.diff(ratio[:origin_k+1]), индексы 0..origin_k-1
    t_orig_d:   origin_k - 1  (последний известный dratio)
    pool_times: конечные индексы окон в dratio-пуле (из _search_wmax_cached)
    X_clean:    LP-очищенный пул (K × P_search) в dratio-пространстве

    Возвращает dratio_pred (horizon,).
    Реконструкция в UI: ratio[origin_k] + cumsum(dratio_pred) → price.
    """
    xi = 3 * (p_fit + 1) + 5

    # LWR обучающая матрица в p_fit-пространстве (dratio)
    # pool_times[i] = t → X[i] = dratio[t-p+1:t+1], Y[i] = dratio[t+1]
    X_raw = np.array([dratio[t - p_fit + 1: t + 1] for t in pool_times])
    Y_raw = np.array([dratio[t + 1]                for t in pool_times])

    # Рефайн: xi ближайших к query в p_fit-пространстве dratio
    x_q   = dratio[t_orig_d - p_fit + 1: t_orig_d + 1]
    dists = np.linalg.norm(X_raw - x_q, axis=1)
    n_keep = min(xi, len(X_raw))
    order  = np.argsort(dists)[:n_keep]
    X_c, Y_c = X_raw[order], Y_raw[order]

    # LP-коррекционная библиотека: последние p_fit столбцов очищенного пула (dratio)
    X_lib_pfit = X_clean[:, -p_fit:] if use_lp_corr else None

    x = x_q.copy()
    preds: list[float] = []
    for _ in range(horizon):
        y = _lwr_one_step(x, X_c, Y_c, xi, acc_lambda)
        x_new = np.r_[x[1:], y]
        if X_lib_pfit is not None:
            y     = _lp_corr_step(x_new, X_lib_pfit, d_local, k_corr)
            x_new = np.r_[x[1:], y]
        preds.append(y)
        x = x_new

    return np.array(preds)


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="W_max Forecaster (app4)", layout="wide",
                   initial_sidebar_state="expanded")

# ── ШАГ 0: nav/pfit мутации до виджетов ──────────────────────────────────────

if "_nav_delta" in st.session_state:
    _delta = st.session_state.pop("_nav_delta")
    st.session_state["origin_k"] = int(np.clip(
        st.session_state.get("origin_k", 0) + _delta,
        0, 9999))
    st.session_state["_auto_fc"] = True

# ── САЙДБАР ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("W_max app4")

    ticker   = st.text_input("Тикер", value="SBER", key="ticker").upper().strip()
    interval = st.selectbox("Интервал", list(INTERVALS),
                             index=list(INTERVALS).index("1d"), key="interval")

    st.divider()
    st.subheader("Горизонт")
    horizon = st.slider("Горизонт (баров)", 1, 200, 30, 1, key="horizon")

    st.divider()
    st.subheader("W_max поиск")

    P_search = st.select_slider(
        "P_search (embedding)", options=[32, 48, 64, 96, 128], value=64,
        key="P_search",
        help="Размерность фазового пространства при поиске W. "
             "Большое P → полнее захватывает аттрактор, но медленнее.",
    )
    K_search = st.slider("K (пул Theiler)", 20, 100, 50, 5, key="K_search",
                          help="Размер Theiler-пула. K=50 — компромисс скорость/качество.")
    K_prime  = st.slider("k' (LP соседей)", 5, 30, 10, 1, key="K_prime",
                          help="Число соседей внутри пула при LP-очистке.")
    N_lp     = st.slider(
        "n_LP (итераций очистки)", 3, 40, 20, 1, key="N_lp",
        help="Итерации LP при скане W. n=20 — исследовательский стандарт. "
             "n=5-10 — быстрее, немного грубее.",
    )
    d_thresh = st.slider(
        "d_thresh (SVD %)", 50, 99, 80, 1, key="d_thresh",
        help="Порог накопленной дисперсии SVD для оценки d_local. "
             "80% — стандарт скр.98.",
    ) / 100.0
    W_min  = st.slider("W_min (мин. Theiler)", 2, 20, 9, 1, key="W_min")
    W_step = st.slider("W_step (шаг сетки)", 1, 8, 4, 1, key="W_step")

    _w_grid_preview = list(range(W_min, P_search + 1, W_step))
    st.caption(f"W сетка ({len(_w_grid_preview)} точек): {_w_grid_preview[:6]}"
               f"{'...' if len(_w_grid_preview) > 6 else ''}")

    st.divider()
    st.subheader("p_fit = 2·d_local + margin")
    margin = st.slider(
        "margin", 1, 6, 2, 1, key="margin",
        help="Такенс: p ≥ 2d+1. margin=1 — минимум, margin=2 — стандарт скр.98.",
    )

    st.divider()
    st.subheader("LWR")
    use_acc_ang = st.checkbox(
        "acc_ang (угловое ускорение)", value=True, key="use_acc_ang",
        help="Финальный отбор по d_pos + λ·|acc_q - acc_i|. "
             "Скр.80: λ=0.01 → −10.8% rMAE на 8 тикерах 1d.",
    )
    acc_lambda = 0.0
    if use_acc_ang:
        acc_lambda = st.slider(
            "λ (вес ускорения)", 0.001, 0.1, 0.01, 0.001,
            format="%.3f", key="acc_lambda",
        )
        st.caption(f"ξ = 3·(p_fit+1)+5  |  acc_ang λ={acc_lambda:.3f}")
    else:
        st.caption("ξ = 3·(p_fit+1)+5  |  acc_ang выкл.")

    st.divider()
    use_lp_corr = st.checkbox(
        "LP-коррекция траектории", value=False, key="use_lp_corr",
        help="После каждого LWR-шага проецирует предсказанную точку на локальное "
             "d_local-мерное подпространство аттрактора. "
             "LP-соседей = k' (те же параметры, что при очистке пула).",
    )
    if use_lp_corr:
        st.caption(f"LP-коррекция: k={K_prime}, d=d_local (из W_max)")

    st.divider()
    n_display = st.slider("История на графике (баров)", 50, 500, 150, 10, key="n_display")

# ── ШАГ 1: загрузка данных ────────────────────────────────────────────────────

cache_path = DATA_DIR / ticker / f"{interval}.json"

_c1, _c2, _c3 = st.columns([3, 1, 1])
load_btn  = _c2.button("Загрузить",    type="primary", use_container_width=True)
reset_btn = _c3.button("Сбросить кэш", use_container_width=True)

if reset_btn and cache_path.exists():
    cache_path.unlink(); st.rerun()

raw_candles: list[dict] | None = None
if load_btn:
    if cache_path.exists():
        raw_candles = json.loads(cache_path.read_text(encoding="utf-8"))
        _c1.info(f"Из кэша: {len(raw_candles):,} свечей")
    else:
        with st.spinner(f"Скачиваю {ticker} [{interval}] с MOEX…"):
            raw_candles = download_candles(ticker, interval, show_progress=False)
        if not raw_candles:
            st.error(f"Нет данных для **{ticker}**"); st.stop()
        save_candles(raw_candles, cache_path)
        _c1.success(f"Скачано: {len(raw_candles):,} свечей")
elif cache_path.exists():
    raw_candles = json.loads(cache_path.read_text(encoding="utf-8"))

if not raw_candles:
    st.info("Введите тикер и нажмите «Загрузить»."); st.stop()

# ── ШАГ 2: нормализация ───────────────────────────────────────────────────────

df = pd.DataFrame(raw_candles)
df["begin"] = pd.to_datetime(df["begin"])
for col in ("open", "high", "low", "close"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["close"]).reset_index(drop=True)

close    = df["close"].values.astype(np.float64)
logtrend = _logtrend_causal(close)
ratio    = close / logtrend

# ── ШАГ 3: границы origin ─────────────────────────────────────────────────────

_origin_min = max(P_search + K_search + W_min + 10, 200)
_origin_max = len(df) - 1

# Коррекция nav-дельты в допустимые границы
if st.session_state.get("origin_k") is not None:
    st.session_state["origin_k"] = int(np.clip(
        st.session_state["origin_k"], _origin_min, _origin_max))

# ── ШАГ 4: origin-слайдер и навигация ────────────────────────────────────────

PERIODS = {"1 мес": 30, "3 мес": 90, "6 мес": 180, "1 год": 365, "3 года": 1095, "Всё": None}
period  = st.radio("Период", list(PERIODS), index=2, horizontal=True)
days    = PERIODS[period]
last_dt = df["begin"].iloc[-1]
view    = (df[df["begin"] >= last_dt - pd.Timedelta(days=days)].copy()
           if days else df.copy())

origin_k  = st.slider("Origin", _origin_min, _origin_max, _origin_max, 1, key="origin_k")
origin_ts = df["begin"].iloc[origin_k]

_n1, _n2, _n3, _n4, _n5 = st.columns([1, 1, 4, 1, 1])
if _n1.button("⟪ −5", use_container_width=True):
    st.session_state["_nav_delta"] = -5; st.rerun()
if _n2.button("⟨ −1", use_container_width=True):
    st.session_state["_nav_delta"] = -1; st.rerun()
_n3.caption(f"&nbsp;&nbsp;&nbsp;{origin_ts.strftime('%d.%m.%Y')}  |  бар {origin_k}")
if _n4.button("+1 ⟩", use_container_width=True):
    st.session_state["_nav_delta"] = +1; st.rerun()
if _n5.button("+5 ⟫", use_container_width=True):
    st.session_state["_nav_delta"] = +5; st.rerun()

# ── ШАГ 5: W_max поиск (кэшируется) ─────────────────────────────────────────

# dratio каузально: только до origin_k включительно (diff даёт origin_k значений)
# dratio[t] = ratio[t+1] - ratio[t],  t ∈ 0..origin_k-1
# t_orig_d = origin_k - 1 (последний известный dratio)
_dratio_causal = np.diff(ratio[:origin_k + 1])   # len = origin_k
_t_orig_d      = origin_k - 1
_dratio_bytes  = _dratio_causal.tobytes()

_wmax_key = (
    _dratio_bytes, _t_orig_d,
    P_search, K_search, K_prime, N_lp,
    int(d_thresh * 1000), W_min, W_step,
)

_scan_result: dict | None = st.session_state.get("wmax_scan")
_wmax_cache_key = st.session_state.get("wmax_cache_key")

if _wmax_cache_key != _wmax_key:
    with st.spinner(
        f"W_max поиск… (P={P_search}, K={K_search}, k'={K_prime}, "
        f"n_LP={N_lp}, {len(_w_grid_preview)} значений W)"
    ):
        _scan_result = _search_wmax_cached(
            _dratio_bytes, _t_orig_d,
            P_search, K_search, K_prime, N_lp,
            d_thresh, W_min, W_step,
        )
    st.session_state["wmax_scan"]      = _scan_result
    st.session_state["wmax_cache_key"] = _wmax_key

scan    = _scan_result
W_max   = scan["W_max"]
d_local = scan["d_local"]
p_fit   = (2 * d_local + margin) if d_local is not None else None

# ── ШАГ 6: статус W_max ───────────────────────────────────────────────────────

_s1, _s2, _s3, _s4 = st.columns(4)
_s1.metric("W_max", str(W_max) if W_max is not None else "—")
_s2.metric("d_local", str(d_local) if d_local is not None else "—")
_s3.metric("p_fit = 2d+margin", str(p_fit) if p_fit is not None else "—")
if p_fit is not None:
    _xi = 3 * (p_fit + 1) + 5
    _s4.metric("ξ = 3(p+1)+5", str(_xi))
else:
    _s4.metric("ξ", "—")

# ── ШАГ 7: диагностическая таблица W-скана ───────────────────────────────────

with st.expander("W-sweep диагностика", expanded=False):
    hist = scan["history"]
    rows = []
    for h in hist:
        if np.isnan(h["acf1"]):
            rows.append({"W": h["W"], "n_кандидатов": "мало", "d_before": "—",
                         "d_after": "—", "P_ref": "—", "CI95": "—",
                         "|ACF lag-1|": "—", "ok": "—"})
        else:
            rows.append({
                "W": h["W"],
                "n_кандидатов": h["n_cands"],
                "d_before": h["d_before"],
                "d_after":  h["d_after"],
                "P_ref":    h["P_ref"],
                "CI95":     f"{h['CI95']:.4f}",
                "|ACF lag-1|": f"{h['acf1']:.4f}",
                "ok": "✓" if h["ok"] else "✗",
            })
    tbl = pd.DataFrame(rows)
    st.dataframe(
        tbl.style.apply(
            lambda col: ["background-color: #1e3a1e" if v == "✓"
                         else ("background-color: #3a1e1e" if v == "✗" else "")
                         for v in col],
            subset=["ok"],
        ),
        use_container_width=True, hide_index=True,
    )

# ── ШАГ 8: прогноз ────────────────────────────────────────────────────────────

_fc1, _fc2 = st.columns([1, 5])
run_btn = _fc1.button("▶ Прогноз", type="primary", use_container_width=True)

if W_max is None or p_fit is None:
    st.warning("W_max не найден для этого origin — прогноз невозможен. "
               "Попробуйте уменьшить W_min, увеличить K или выбрать другой origin.")
    st.stop()

_fc_key = (
    ticker, interval, origin_k, _t_orig_d, horizon, margin,
    P_search, K_search, K_prime, N_lp, int(d_thresh * 1000), W_min, W_step,
    acc_lambda, use_lp_corr,
)

if run_btn or st.session_state.pop("_auto_fc", False):
    # Найти X_clean при W_max
    _wmax_entry = next((h for h in scan["history"] if h["W"] == W_max), None)
    if _wmax_entry is None or _wmax_entry["pool_times"] is None:
        st.error("Не удалось найти запись W_max в истории скана."); st.stop()

    pool_times = _wmax_entry["pool_times"]
    X_clean    = _wmax_entry["X_clean"]

    with st.spinner(f"Прогноз…  p_fit={p_fit}  ξ={3*(p_fit+1)+5}  H={horizon}"):
        dratio_pred = _forecast_wmax(
            _dratio_causal, _t_orig_d, horizon,
            pool_times, X_clean,
            p_fit=p_fit, d_local=d_local,
            acc_lambda=acc_lambda,
            use_lp_corr=use_lp_corr,
            k_corr=K_prime,
        )

    # Реконструкция: ratio[origin_k] + cumsum(dratio_pred) → ratio_pred → price
    ratio_pred  = ratio[origin_k] + np.cumsum(dratio_pred)
    _lt_fwd = logtrend[origin_k + 1: origin_k + 1 + horizon]
    if len(_lt_fwd) < horizon:
        _last_lt = float(logtrend[min(origin_k + len(_lt_fwd), len(logtrend) - 1)])
        _lt_fwd  = np.concatenate([_lt_fwd, np.full(horizon - len(_lt_fwd), _last_lt)])
    forecast_price = ratio_pred * _lt_fwd

    # Временные метки прогноза
    _fut = df.iloc[origin_k + 1: origin_k + 1 + horizon]
    if len(_fut) < horizon:
        _bar_d = _BAR_DELTA.get(interval, pd.Timedelta(hours=1))
        _last  = _fut["begin"].iloc[-1] if len(_fut) else origin_ts
        _extra = pd.Series([_last + _bar_d * i for i in range(1, horizon - len(_fut) + 1)])
        forecast_begin = pd.concat([_fut["begin"].reset_index(drop=True), _extra],
                                   ignore_index=True)
    else:
        forecast_begin = _fut["begin"].reset_index(drop=True)

    # Привязка к последней известной цене
    junction_p = float(close[origin_k])

    _acc_str  = f"acc_ang λ={acc_lambda:.3f}" if acc_lambda > 0.0 else "acc_ang выкл."
    _lp_str   = f"  +LP-corr(k={K_prime},d={d_local})" if use_lp_corr else ""
    _lbl = (f"W_max={W_max}  d={d_local}  p={p_fit}  ξ={3*(p_fit+1)+5}"
            f"  {_acc_str}{_lp_str}")

    st.session_state.update({
        "fc4_key":       _fc_key,
        "fc4_origin_k":  origin_k,
        "fc4_price":     forecast_price,
        "fc4_begin":     forecast_begin,
        "fc4_junction":  junction_p,
        "fc4_lbl":       _lbl,
    })

# Загружаем снапшот
_snap: dict | None = None
if (st.session_state.get("fc4_key") == _fc_key
        and st.session_state.get("fc4_origin_k") == origin_k
        and st.session_state.get("fc4_price") is not None):
    _snap = {
        "price":     st.session_state["fc4_price"],
        "begin":     st.session_state["fc4_begin"],
        "junction":  st.session_state["fc4_junction"],
    }
    _fc2.caption(st.session_state.get("fc4_lbl", ""))

# ── ШАГ 9: график ─────────────────────────────────────────────────────────────

if interval in ("1d", "1w", "1mo"):
    rangebreaks = [dict(bounds=["sat", "mon"])]
else:
    _has_ev = (df["begin"].dt.hour >= 19).any()
    rangebreaks = [dict(bounds=["sat", "mon"]),
                   dict(bounds=([0, 10] if _has_ev else [19, 10]), pattern="hour")]

_grid   = dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
_layout = dict(dragmode="pan", paper_bgcolor="rgba(0,0,0,0)",
               plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0))

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=view["begin"], open=view["open"], high=view["high"],
    low=view["low"], close=view["close"], name=ticker,
    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    whiskerwidth=0.5,
))

# Лог-тренд оверлей
_lt_show_start = max(0, origin_k - n_display)
fig.add_trace(go.Scatter(
    x=df["begin"].iloc[_lt_show_start: origin_k + 1],
    y=logtrend[_lt_show_start: origin_k + 1],
    mode="lines", name="logtrend",
    line=dict(color="rgba(200,200,100,0.5)", width=1.5, dash="dot"),
))

# Ratio-восстановленная цена (ratio × logtrend) для отображения att
fig.add_trace(go.Scatter(
    x=df["begin"].iloc[_lt_show_start: origin_k + 1],
    y=ratio[_lt_show_start: origin_k + 1] * logtrend[_lt_show_start: origin_k + 1],
    mode="lines", name="ratio × logtrend",
    line=dict(color="rgba(100,181,246,0.6)", width=1.5),
))

fig.add_vline(x=str(origin_ts)[:10],
              line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"))

if _snap is not None:
    n_fc   = min(len(_snap["price"]), len(_snap["begin"]))
    _fc_x  = pd.Series([origin_ts] + list(_snap["begin"][:n_fc]))
    _fc_y  = np.concatenate([[_snap["junction"]], _snap["price"][:n_fc]])
    fig.add_trace(go.Scatter(
        x=_fc_x, y=_fc_y, mode="lines+markers",
        name=f"Прогноз W_max={W_max}  p={p_fit}",
        line=dict(color="#00e5ff", width=2.5),
        marker=dict(size=4),
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

# ── ШАГ 10: метрики ───────────────────────────────────────────────────────────

_m1, _m2, _m3, _m4, _m5 = st.columns(5)
_m1.metric("Баров в истории", f"{origin_k:,}")
_m2.metric("W_max / d_local", f"{W_max} / {d_local}")
_m3.metric("p_fit (2d+margin)", f"{p_fit}  (m={margin})")
_m4.metric("ξ LWR", f"{3*(p_fit+1)+5}")
_m5.metric("Цена (origin)", f"{close[origin_k]:.2f}")
