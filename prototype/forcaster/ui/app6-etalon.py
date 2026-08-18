"""
app6: Двухэтапный автоматический пайплайн.

Этап 1: PCA sweep m∈[m_min,m_max] → d_min=⌊mean d(m,τ₁)⌋, d_max=⌈mean d(m,τ₂)⌉.
Этап 2: sweep d∈[d_min,d_max] → S-map с PCA-проекцией пула в d-мерное подпространство
        → простое среднее по d.

Стек: logtrend causal OLS → каузальный LP-фильтр(ratio, m=x·d+y) → каскадный S-map.
Обозначения: m — окно задержки, K = 3·(m+1)+ξ — число соседей, d — размерность аттрактора.
Метрика: amp_cos. Каскад: октавный ×2. Нет утечки из будущего.

Run (из prototype/): streamlit run forcaster/ui/app6.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from forcaster.data.moex import download_candles, save_candles, INTERVALS

DATA_DIR = _root / "data" / "candles"

# ═══════════════════════════════════════════════════════════════════════════════
# Метрика расстояния: amp_cos
# ═══════════════════════════════════════════════════════════════════════════════

def _dists(X: np.ndarray, q: np.ndarray, blend_alpha: float = 0.5) -> np.ndarray:
    """amp_cos: α·|log(‖x‖/‖y‖)|_norm + (1−α)·cosine_norm."""
    n  = len(X)
    nX = np.linalg.norm(X, axis=1)
    nq = np.linalg.norm(q)
    if nq < 1e-10:
        d_shape = np.ones(n)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            sim = np.where(nX > 1e-10, (X @ q) / (nX * nq), 0.0)
        d_shape = 1.0 - sim.clip(-1.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_amp = np.where(
            (nX > 1e-10) & (nq > 1e-10),
            np.abs(np.log(nX / nq)),
            np.abs(nX - nq),
        )
    max_a = max(float(d_amp.max()), 1e-10)
    max_s = max(float(d_shape.max()), 1e-10)
    return blend_alpha * (d_amp / max_a) + (1.0 - blend_alpha) * (d_shape / max_s)


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
    times  = [c["begin"] for c in data]
    closes = np.array([float(c["close"]) for c in data])
    opens  = np.array([float(c["open"])  for c in data])
    highs  = np.array([float(c["high"])  for c in data])
    lows   = np.array([float(c["low"])   for c in data])
    return times, closes, opens, highs, lows

# ═══════════════════════════════════════════════════════════════════════════════
# Logtrend (causal OLS)
# ═══════════════════════════════════════════════════════════════════════════════

def _logtrend_causal(close: np.ndarray):
    n  = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a     = (cy - b * ct) / cn
    trend = np.exp(a + b * t)
    trend[:2] = close[:2]
    return trend, a, b

# ═══════════════════════════════════════════════════════════════════════════════
# LP-фильтр (causal, ratio → att)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _lp_proj_ratio_cached(ratio_bytes: bytes, m: int, d: int, k: int, n_iter: int,
                           blend_alpha: float = 0.5) -> np.ndarray:
    ratio = np.frombuffer(ratio_bytes, dtype=np.float64).copy()
    return _lp_proj_causal(ratio, m, d, k, n_iter, blend_alpha)

def _lp_proj_causal(ratio: np.ndarray, m: int, d: int, k: int, n_iter: int,
                    blend_alpha: float = 0.5) -> np.ndarray:
    """
    Причинный LP-фильтр: att[t] вычисляется только по ratio[0..t].
    att[t] = последний элемент проекции текущего окна на локальное d-мерное подпространство.
    Соседи — все исторические окна, завершившиеся строго до t (глубина задаётся trim'ом ratio).
    """
    n     = len(ratio)
    d_eff = min(d, m - 1)
    att   = ratio.copy().astype(np.float64)
    if d_eff < 1 or n < m + 1:
        return att

    nw = n - m + 1

    for _ in range(n_iter):
        X_all   = att[np.arange(nw)[:, None] + np.arange(m)]
        new_att = att.copy()

        for q in range(1, nw):
            X_hist = X_all[:q]
            curr   = X_all[q]
            k_eff  = min(k, len(X_hist))
            if k_eff < d_eff + 1:
                continue
            dists  = _dists(X_hist, curr, blend_alpha)
            sel    = np.argpartition(dists, k_eff - 1)[:k_eff]
            nn     = X_hist[sel]
            center = nn.mean(0)
            _, _, Vt = np.linalg.svd(nn - center, full_matrices=False)
            proj   = center + Vt[:d_eff].T @ (Vt[:d_eff] @ (curr - center))
            new_att[q + m - 1] = proj[-1]

        att = new_att

    return att

# ═══════════════════════════════════════════════════════════════════════════════
# Локальная размерность PCA (Этап 1)
# ═══════════════════════════════════════════════════════════════════════════════

def _pca_dim_local(X_nn: np.ndarray, threshold: float = 0.90) -> float:
    if len(X_nn) < 2:
        return np.nan
    center = X_nn.mean(axis=0)
    _, s, _ = np.linalg.svd(X_nn - center, full_matrices=False)
    var     = s ** 2
    cum_var = np.cumsum(var) / max(float(var.sum()), 1e-14)
    d = int(np.searchsorted(cum_var, threshold)) + 1
    return float(min(d, X_nn.shape[1]))


def compute_local_dim(
    ratio: np.ndarray,
    p_search: int,
    k_dim: int,
    pca_thresholds: tuple = (0.95, 0.97),
    blend_alpha: float = 0.5,
) -> dict:
    """PCA-размерность по хвосту ratio (без LP, без каскада). ratio уже обрезан до origin+bars."""
    n_eff = len(ratio)
    if n_eff < p_search:
        return {"error": "недостаточно истории"}
    query = ratio[n_eff - p_search : n_eff].astype(np.float64)
    max_s = n_eff - 1 - p_search
    if max_s < 0:
        return {"error": "библиотека пуста"}
    n_lib = max_s + 1
    if n_lib < k_dim:
        return {"error": f"библиотека мала: {n_lib} < {k_dim}"}
    starts = np.arange(0, max_s + 1)
    X_lib  = ratio[starts[:, None] + np.arange(p_search)].astype(np.float64)
    dists  = _dists(X_lib, query, blend_alpha)
    k_eff  = min(k_dim, len(X_lib))
    X_nn   = X_lib[np.argpartition(dists, k_eff - 1)[:k_eff]]
    return {f"pca_{t}": _pca_dim_local(X_nn, t) for t in pca_thresholds}


def sweep_local_dim(
    ratio: np.ndarray,
    p_values: list[int],
    xi_add: int,
    pca_thresholds: tuple = (0.95, 0.97),
    blend_alpha: float = 0.5,
) -> list[dict]:
    rows = []
    for p in p_values:
        k_dim = 3 * (p + 1) + xi_add
        res   = compute_local_dim(ratio, p, k_dim,
                                   pca_thresholds, blend_alpha)
        row = {"p_search": p, "K": k_dim}
        if "error" in res:
            for t in pca_thresholds:
                row[f"PCA_{int(t*100)}"] = float("nan")
            row["_error"] = res["error"]
        else:
            for t in pca_thresholds:
                row[f"PCA_{int(t*100)}"] = res[f"pca_{t}"]
            row["_error"] = ""
        rows.append(row)
    return rows

# ═══════════════════════════════════════════════════════════════════════════════
# Каскад S-map (октавный ×2)
# ═══════════════════════════════════════════════════════════════════════════════

def _cascade_levels(p_fit: int, n_levels: int, p_cascade_max: int = 1500) -> list[int]:
    levels = [p_fit * (2 ** (n_levels - 1 - k)) for k in range(n_levels)]
    levels = [p for p in levels if p <= p_cascade_max]
    return levels if levels else [p_fit]


def cascade_search(
    att: np.ndarray,
    context: np.ndarray,
    p_fit: int,
    n_levels: int,
    K: int,
    t_predict: int,
    blend_alpha: float = 0.5,
    p_cascade_max: int = 1500,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Строгий октавный каскад ×2 (cascade_algorithm.md).
    K = 3·(m+1)+ξ — полное число соседей на каждом уровне.
    Последний уровень возвращает весь пул (S-map).
    Гарантия: s + p_lv < t_predict для всех точек пула.
    Уровни с p > p_cascade_max снимаются сверху.
    Глубина истории ограничена trim'ом att (= len(att)).
    """
    n      = len(att)
    levels = _cascade_levels(p_fit, n_levels, p_cascade_max)
    p_top  = levels[0]

    max_s = min(n - p_top, t_predict - p_top - 1)
    if max_s < 0:
        return None, None

    pool  = np.arange(0, max_s + 1)
    if len(pool) < K:
        return None, None

    for level_idx, p_lv in enumerate(levels):
        is_last = (level_idx == n_levels - 1)

        if len(context) < p_lv:
            return None, None
        q = context[-p_lv:].astype(np.float64)

        valid = (pool + p_lv - 1) < n
        pool  = pool[valid]
        if len(pool) == 0:
            return None, None

        X_pool = att[pool[:, None] + np.arange(p_lv)]
        dists  = _dists(X_pool, q, blend_alpha)

        if is_last:
            valid_y  = pool + p_fit < n
            pool_sm  = pool[valid_y]
            X_sm     = X_pool[valid_y]
            if len(pool_sm) < 2:
                return None, None
            return X_sm, att[pool_sm + p_fit]

        n_sel      = min(K, len(pool))
        top_i      = np.argpartition(dists, n_sel - 1)[:n_sel]
        sel_starts = pool[top_i]

        p_next  = levels[level_idx + 1]
        r       = p_lv - p_next
        offsets = np.arange(r + 1)
        flat    = (sel_starts[:, None] + offsets[None, :]).ravel()
        max_s_next = min(n - p_next, t_predict - p_next - 1)
        flat    = flat[(flat >= 0) & (flat <= max_s_next)]
        pool    = np.unique(flat)
        if len(pool) == 0:
            return None, None

    return None, None

# ═══════════════════════════════════════════════════════════════════════════════
# S-map + PiP (без wSVD)
# ═══════════════════════════════════════════════════════════════════════════════

def _pip_project(Xf: np.ndarray, vf: np.ndarray,
                 d_proj: int) -> tuple[np.ndarray, np.ndarray]:
    center = Xf.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xf - center, full_matrices=False)
    V = Vt[:d_proj].T
    return (Xf - center) @ V, (vf - center) @ V


def _smap_step(
    X_pool: np.ndarray, y_pool: np.ndarray, vec_fit: np.ndarray,
    theta: float, blend_alpha: float, d_proj: int | None,
) -> float:
    p_eff  = min(X_pool.shape[1], len(vec_fit))
    Xf     = X_pool[:, -p_eff:]
    vf     = vec_fit[-p_eff:]
    d_arr  = _dists(Xf, vf, blend_alpha)
    d_mean = max(float(d_arr.mean()), 1e-10)
    w      = np.exp(-theta * d_arr / d_mean)
    if d_proj is not None and d_proj < p_eff and len(Xf) > d_proj:
        Xf, vf = _pip_project(Xf, vf, d_proj)
    A  = np.hstack([np.ones((len(Xf), 1)), Xf])
    sw = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_pool, rcond=None)
    return float(c[0] + vf @ c[1:])

# ═══════════════════════════════════════════════════════════════════════════════
# LP-коррекция траектории
# ═══════════════════════════════════════════════════════════════════════════════

def _lp_corr(v_m: np.ndarray, X_lib: np.ndarray, k: int, d: int,
             blend_alpha: float) -> float:
    """Проецирует m-мерный вектор на локальное d-мерное подпространство, возвращает последний элемент."""
    k_eff = min(k, len(X_lib))
    if k_eff < d + 1:
        return float(v_m[-1])
    dists  = _dists(X_lib, v_m, blend_alpha)
    idx    = np.argpartition(dists, k_eff - 1)[:k_eff]
    X_nn   = X_lib[idx]
    center = X_nn.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_nn - center, full_matrices=False)
    d_eff  = min(d, Vt.shape[0])
    Vd     = Vt[:d_eff].T
    v_proj = center + Vd @ (Vd.T @ (v_m - center))
    return float(v_proj[-1])

# ═══════════════════════════════════════════════════════════════════════════════
# Прогноз: один d
# ═══════════════════════════════════════════════════════════════════════════════

def run_forecast(
    att: np.ndarray,
    p_fit: int, n_levels: int, K: int,
    horizon: int, origin: int,
    blend_alpha: float, smap_theta: float, d_proj: int,
    p_cascade_max: int = 1500,
    use_lp_corr: bool = False,
) -> np.ndarray | None:
    """Возвращает fc_preds[horizon] — сырые att-предсказания, или None при ошибке."""
    n_eff     = origin + 1
    _min_pool = d_proj + 2
    fc_buf    = list(att[:n_eff])

    X_lib_lp: np.ndarray | None = None
    if use_lp_corr:
        n_lib = n_eff - p_fit
        if n_lib > 0:
            X_lib_lp = att[np.arange(n_lib)[:, None] + np.arange(p_fit)]

    for h in range(horizon):
        t   = n_eff + h
        ctx = np.array(fc_buf)
        X_nn, y_nn = cascade_search(
            att, ctx, p_fit, n_levels, K, t, blend_alpha, p_cascade_max,
        )
        if X_nn is None or len(X_nn) < _min_pool:
            return None
        pred = _smap_step(X_nn, y_nn, ctx[-p_fit:], smap_theta, blend_alpha, d_proj)
        if use_lp_corr and X_lib_lp is not None:
            v_m  = np.array(fc_buf[-(p_fit - 1):] + [pred], dtype=np.float64)
            pred = _lp_corr(v_m, X_lib_lp, K, d_proj, blend_alpha)
        fc_buf.append(pred)

    return np.array(fc_buf[n_eff:])

# ═══════════════════════════════════════════════════════════════════════════════
# Этап 2: sweep по d
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage2(
    ratio: np.ndarray,
    d_values: list[int],
    xy_x: int, xy_y: int, xi_add: int,
    n_levels: int, horizon: int, origin: int,
    progress_cb=None,
    blend_alpha: float = 0.5,
    smap_theta: float = 18.0,
    p_cascade_max: int = 1500,
    use_lp_corr: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Возвращает (results, failed). results содержит fc_preds (сырые att)."""
    results = []
    failed  = []
    for step_i, d in enumerate(d_values):
        if progress_cb:
            progress_cb(step_i, len(d_values), d)

        p_fit  = xy_x * d + xy_y
        K      = 3 * (p_fit + 1) + xi_add
        d_proj = d

        if p_fit < 2:
            failed.append({"d": d, "reason": "p_fit < 2"})
            continue

        att = _lp_proj_ratio_cached(ratio.tobytes(), p_fit, d, K, 1, blend_alpha)

        eff_levels = _cascade_levels(p_fit, n_levels, p_cascade_max)
        min_len = eff_levels[0] + p_fit + 1
        if len(att) < min_len:
            failed.append({"d": d, "reason": f"ряд мал для каскада (нужно {min_len})"})
            continue

        fc_preds = run_forecast(
            att, p_fit, n_levels, K, horizon, origin,
            blend_alpha, smap_theta, d_proj, p_cascade_max,
            use_lp_corr,
        )
        if fc_preds is None:
            failed.append({"d": d, "reason": f"pool < d+2={d+2}"})
            continue

        results.append({
            "d":        d,
            "m":        p_fit,
            "K":        K,
            "fc_preds": fc_preds,
        })

    return results, failed

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="app6 · Auto Pipeline", layout="wide",
                   initial_sidebar_state="expanded")

with st.sidebar:
    st.title("app6 · Auto Pipeline")

    ticker   = st.text_input("Тикер", "SBER", key="ticker").upper().strip()
    interval = st.selectbox("Интервал", list(INTERVALS.keys()),
                            index=list(INTERVALS.keys()).index("1d"))
    if st.button("Обновить данные"):
        _load_candles.clear()
        _fetch_and_save(ticker, interval)
        st.rerun()

    st.divider()
    with st.expander("Обозначения и схема метода"):
        st.markdown(
            "**Методы**\n\n"
            "| Блок | Метод |\n"
            "|------|-------|\n"
            "| Нормализация | **Logtrend** (causal OLS): ratio = close / exp(a+b·t) |\n"
            "| Фильтрация | **LP-фильтр** (Local Projective Filter): каузальная SVD-проекция ratio на локальное d-мерное касательное многообразие, построенное по K ближайшим соседям |\n"
            "| Оценка d | **PCA sweep**: SVD облака K соседей → локальная размерность аттрактора |\n"
            "| Поиск соседей | **Каскадный поиск**: иерархический coarse-to-fine NN в пространстве задержек |\n"
            "| Прогноз | **S-map** (Sugihara, 1994): локально-взвешенная авторегрессия с PCA-проекцией пула в d-мерное подпространство перед МНК |\n\n"
            "---\n\n"
            "**Обозначения**\n\n"
            "| Символ | Описание |\n"
            "|:------:|----------|\n"
            "| ratio | close / exp(a+b·t) — нормализованная цена |\n"
            "| att | LP(ratio) — аттрактор (гладкая компонента ratio) |\n"
            "| d | локальная размерность аттрактора; единая для LP-проекции, PCA sweep и S-map-проекции |\n"
            "| m | размер окна задержки: m = x·d + y (единый для LP и S-map) |\n"
            "| K | число ближайших соседей: K = 3·(m+1) + ξ |\n"
            "| ξ | добавка к K |\n"
            "| ρ(x,y) | метрика amp_cos: α·\\|log(‖x‖/‖y‖)\\|ₙ + (1−α)·(1−cos∠(x,y))ₙ |\n"
            "| α | параметр ρ: 0 = чистая форма (косинус), 1 = чистая амплитуда |\n"
            "| θ | параметр нелинейности S-map |\n"
            "| n | число уровней каскада |\n"
            "| H | горизонт прогноза |\n"
            "| τ₁, τ₂ | пороги PCA (доля объяснённой дисперсии) |\n"
            "| a, b | коэффициенты logtrend (causal OLS) в момент origin |\n"
            "| origin | точка отсчёта: att[t≤origin] — история, att[t>origin] — прогноз; a,b берутся каузально от origin |\n\n"
            "---\n\n"
            "**Схема пайплайна**\n\n"
            "```\n"
            "close → ratio = close / exp(a+b·t)\n"
            "\n"
            "Этап 1 — оценка d (PCA sweep, m перебирается напрямую):\n"
            "  for m ∈ [m_min, m_max]:\n"
            "    K = 3·(m+1) + ξ соседей для origin в ℝᵐ\n"
            "    SVD → d(m, τ₁), d(m, τ₂)\n"
            "  d_min = ⌊mean d(m,τ₁)⌋,  d_max = ⌈mean d(m,τ₂)⌉\n"
            "\n"
            "Этап 2 — прогноз (m теперь вычисляется из найденного d):\n"
            "  for d ∈ [d_min, d_max]:\n"
            "    m = x·d + y,   K = 3·(m+1) + ξ\n"
            "    att = LP(ratio, m, d)           ← каузальный LP-фильтр\n"
            "    for h = 1…H:\n"
            "      cascade(att, m, K) → S-map(θ) → âtt[t+h]\n"
            "    price_d = âtt · exp(a+b·t)      ← реконструкция\n"
            "  mean(price_d) → финальный прогноз (простое среднее)\n"
            "```"
        )

    st.divider()
    st.subheader("Согласование тракта")
    st.caption(
        "Параметры x, y, ξ, α задают единое m = x·d + y и K = 3·(m+1) + ξ.  \n"
        "Оба значения используются во всех трёх блоках одновременно:  \n\n"
        "**LP-фильтр:**  \n"
        "  окно m = x·d + y,  K соседей для SVD-проекции ratio → att  \n"
        "  каузальный: att[t] вычисляется только по ratio[0..t−1] — нет lookahead  \n\n"
        "**S-map:**  \n"
        "  то же m, тот же K  — суть согласования  \n"
        "  взвешенный МНК в d-мерном PCA-подпространстве  \n\n"
        "**PCA sweep:**  \n"
        "  тот же K = 3·(m+1) + ξ, но m здесь перебирается напрямую как параметр окна,  \n"
        "  а не вычисляется через x·d+y (d ещё неизвестен на этом этапе)  \n\n"
        "**Метрика amp_cos** (единая для всех трёх блоков):  \n"
        "  ρ(x,y) = α·|log(‖x‖/‖y‖)|ₙ + (1−α)·(1−cos∠(x,y))ₙ  \n"
        "  (ₙ — нормировка делением на max по текущему пулу)"
    )
    xy_x = st.slider("x", 1, 10, 3, key="xy_x",
                      help="Масштабный коэффициент в m = x·d + y.\n"
                           "Окно m растёт пропорционально размерности вложения d.\n"
                           "Большой x → более гладкий LP-фильтр при больших d.")
    xy_y = st.slider("y", 0, 20, 0, key="xy_y",
                      help="Сдвиг в m = x·d + y.\n"
                           "Минимальное окно при d=1: m = x + y.\n"
                           "y ≥ 1 предотвращает вырождение при малых d.")
    st.caption(f"m = x·d + y  ·  d=5 → m={xy_x*5+xy_y}  ·  d=20 → m={xy_x*20+xy_y}")
    xi_add = st.slider("ξ (добавка к K)", 0, 20, 1,
                       help="Добавка к числу соседей K = 3·(m+1) + ξ.\n"
                            "Базовое правило 3·(m+1): нижняя граница для устойчивой "
                            "МНК-регрессии при m предикторах (запас ≈3×).\n"
                            "ξ увеличивает K сверх минимума.\n"
                            "Действует везде: LP-фильтр, PCA sweep, каскад.")
    blend_alpha = st.slider("α (форма ↔ амплитуда)", 0.0, 1.0, 0.5, 0.05,
                            help="Метрика amp_cos (amplitude-cosine distance):\n"
                                 "  ρ(x,y) = α·|log(‖x‖/‖y‖)|ₙ + (1−α)·(1−cos∠(x,y))ₙ\n"
                                 "  (индекс ₙ — нормировка на [0,1])\n"
                                 "α=0: чистая косинусная метрика — только форма/направление, "
                                 "инвариантна к масштабу\n"
                                 "α=1: только амплитуда — |log(‖x‖/‖y‖)|\n"
                                 "Единая метрика для LP-фильтра, PCA sweep и S-map.")
    st.caption(
        f"ρ(x,y) = {blend_alpha}·|log(‖x‖/‖y‖)|ₙ + {round(1-blend_alpha, 2)}·(1−cos∠(x,y))ₙ"
    )

    st.divider()
    st.subheader("Каскадный поиск")
    st.caption(
        "Иерархический (coarse-to-fine) поиск в пространстве задержек. "
        "n уровней с убывающим окном (октавный ×2):  \n\n"
        "  уровень 0:   m₀ = m · 2ⁿ⁻¹  — крупный масштаб  \n"
        "  уровень k:   mₖ = m · 2ⁿ⁻¹⁻ᵏ  \n"
        "  уровень n−1: m  — базовый масштаб  \n\n"
        "На каждом уровне (кроме последнего): отбирается K ближайших по ρ,  \n"
        "их стартовые позиции расширяются в пул следующего уровня.  \n"
        "Последний уровень передаёт весь пул в S-map без отсечки.  \n"
        "Уровни с mₖ > m(max) снимаются сверху."
    )
    n_levels = st.slider("Уровней каскада", 1, 6, 4,
                         help="Число уровней n.\n"
                              "Окна уровней (октавный ×2): m·2ⁿ⁻¹, m·2ⁿ⁻², …, m.\n"
                              "n=1: одноуровневый поиск (нет иерархии).\n"
                              "Уровни с mₖ > m(max) снимаются автоматически.")
    p_cascade_max = st.slider("Макс. окно каскада m(max)", 100, 5000, 2000, 100,
                              help="Порог m(max): уровни с mₖ > m(max) удаляются сверху.\n"
                                   "Предотвращает ситуацию mₖ > длины доступного ряда.")
    bars = st.slider("Точек в библиотеке", 100, 5000, 3000, 100,
                     help="Глубина поиска соседей: размер скользящего окна библиотеки.\n"
                          "Ограничивает число исторических кандидатов на каждом уровне.")
    _ex_p    = xy_x * 20 + xy_y
    _ex_lvls = _cascade_levels(_ex_p, n_levels, p_cascade_max)
    st.caption(f"Пример d=20: m={_ex_p},  уровни={_ex_lvls}  ({len(_ex_lvls)} ур.)")

    st.divider()
    st.subheader("Измерение размерности PCA")
    st.caption(
        "Оценка локальной размерности d аттрактора методом локального PCA  \n"
        "(SVD облака K ближайших соседей в пространстве задержек сырого ratio, без LP).  \n"
        "На этом этапе m — прямой параметр sweep; формула m=x·d+y применяется только в Этапе 2.  \n\n"
        "Для каждого m ∈ [m_min, m_max], K = 3·(m+1) + ξ:  \n"
        "  1. K ближайших соседей точки origin в ℝᵐ (метрика ρ)  \n"
        "  2. SVD центрированного облака → s₁ ≥ s₂ ≥ … ≥ sₘ  \n"
        "  3. d(m, τ) = min{ k : (Σᵢ₌₁ᵏ sᵢ²) / (Σᵢ₌₁ᵐ sᵢ²) ≥ τ }  \n\n"
        "  d_min = ⌊ mean_m d(m, τ₁) ⌋  — левый порог → нижняя оценка  \n"
        "  d_max = ⌈ mean_m d(m, τ₂) ⌉  — правый порог → верхняя оценка  \n\n"
        "Прогноз итерирует d ∈ [d_min, d_max] и усредняет результаты."
    )
    pca_p_range = st.slider("Диапазон m", 3, 300, (3, 100), key="pca_p_range",
                             help="Диапазон окна задержки для sweep по m.\n"
                                  "Широкий диапазон → надёжнее средняя оценка d, "
                                  "но дольше вычисляется.")
    pca_thr_range = st.slider("Диапазон порогов", 0.80, 0.99, (0.90, 0.99), 0.01,
                              key="pca_thr_range",
                              help="Пороги τ для доли объяснённой дисперсии.\n"
                                   "Левый τ₁ → d_min (нижняя оценка размерности).\n"
                                   "Правый τ₂ → d_max (верхняя оценка).\n"
                                   "Типичные пары: 0.90/0.95 (широкий диапазон d) "
                                   "или 0.95/0.99 (узкий).")
    pca_thr1, pca_thr2 = pca_thr_range[0], pca_thr_range[1]
    _n_pts = pca_p_range[1] - pca_p_range[0] + 1
    st.caption(
        f"{_n_pts} точек  ·  K = 3·(m+1)+{xi_add}  ·  "
        f"d_min = ⌊mean d(m,τ₁={pca_thr1})⌋  ·  d_max = ⌈mean d(m,τ₂={pca_thr2})⌉"
    )
    pca_btn = st.button("Измерить PCA", use_container_width=True)

    st.divider()
    st.subheader("Прогноз")
    st.caption(
        "Итеративный пошаговый прогноз (iterated one-step-ahead) аттрактора att.  \n"
        "Для каждого d ∈ [d_min, d_max], m = x·d + y:  \n\n"
        "  1. LP-фильтр: att = LP(ratio, m, d)  \n"
        "  2. Итерация h = 1 … H:  \n"
        "     a. Каскадный поиск соседей: x_t = att[t−m : t]  \n"
        "     b. PCA-проекция пула в d-мерное подпространство (то же d, что в LP):  \n"
        "        X̃ = (X − μ) · V_d,   V_d — d правых сингулярных векторов SVD  \n"
        "     c. S-map (Sugihara, 1994) в проецированном пространстве:  \n"
        "        wᵢ = exp(−θ · ρ(x̃ᵢ, x̃_t) / ρ̄),   взвешенный МНК → âtt[t+h]  \n"
        "     d. âtt[t+h] добавляется в контекст следующего шага  \n"
        "  3. Реконструкция: price[t] = âtt[t] · exp(a + b·t)  \n\n"
        "Результаты по d усредняются (простое среднее, без взвешивания) → финальный прогноз."
    )
    horizon = st.slider("Горизонт (баров)", 1, 200, 40,
                        help="Число шагов вперёд H.\n"
                             "Каждый шаг — итерация S-map, предыдущее предсказание "
                             "добавляется в контекст (iterated one-step-ahead).")
    origin_offset = st.slider("Точка отсчёта (баров от конца)", 0, 500, 0,
                              help="Сдвиг точки отсчёта назад от последнего бара.\n"
                                   "0 = прогноз из последней доступной точки.\n"
                                   ">0 = проверка на исторических данных "
                                   "(фактические цены видны на графике как 'actual').")
    smap_theta = st.slider("θ (S-map)", 0.0, 50.0, 20.0, 0.5,
                           help="Параметр нелинейности S-map (Sugihara, 1994).\n"
                                "Веса: wᵢ = exp(−θ · ρ(x̃ᵢ, x̃_t) / ρ̄)\n"
                                "θ = 0: равные веса → глобальная линейная авторегрессия (AR)\n"
                                "θ → ∞: только ближайший сосед → сильная локальная нелинейность\n"
                                "Оптимум для финансовых рядов: θ ≈ 5…20.")
    use_lp_corr = st.checkbox(
        "LP-коррекция траектории",
        value=True,
        help="После каждого шага S-map проецирует предсказанную точку\n"
             "на локальное d-мерное подпространство аттрактора:\n"
             "  v_m = [att[t-m+1:t], pred]\n"
             "  pred ← SVD-проекция v_m на ближайших K соседях из истории.\n"
             "Параметры (m, d, K) — те же, что у LP-фильтра и S-map.\n"
             "Удерживает траекторию на многообразии при длинных горизонтах.",
    )
    run_btn = st.button("▶  Прогноз", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Предобработка")
    use_midprice = st.checkbox(
        "Входная цена (open+close)/2",
        value=False,
        help="Входная цена = среднее open и close вместо close.\n"
             "Снижает внутрибарный шум перед logtrend и LP-фильтром.\n"
             "Прогноз реконструируется в той же шкале — привязка к mid, а не close.\n"
             "Свечи и факт на графике остаются по close (только для отображения).",
    )

    st.divider()
    st.subheader("Коррекция")
    delta_ratio = st.slider("Поправка тренда (Δratio/бар)", -0.005, 0.005, 0.0, 0.0001,
                            format="%.4f",
                            help="Линейная коррекция в пространстве ratio:\n"
                                 "  âtt[j] ← âtt[j] + j·δ\n"
                                 "перед восстановлением цены через logtrend.\n"
                                 "Компенсирует систематическое угловое отклонение "
                                 "прогноза при длинных трендах.\n"
                                 "δ < 0 — сдвиг вниз, δ > 0 — вверх.\n"
                                 "Не пересчитывает прогноз — применяется при отрисовке.")
    if delta_ratio != 0.0:
        st.caption(f"Суммарная поправка за горизонт: {delta_ratio * horizon * 100:+.2f}% ratio")

# ── Загрузка данных ───────────────────────────────────────────────────────────

data = _load_candles(ticker, interval)
if not data:
    st.info("Нет данных. Нажмите «Обновить данные» в сайдбаре.")
    st.stop()

times, close, open_, high, low = _to_arrays(data)
price_input = (open_ + close) / 2 if use_midprice else close
logtrend, _a_arr, _b_arr = _logtrend_causal(price_input)
ratio  = price_input / np.maximum(logtrend, 1e-10)
n      = len(close)
origin = max(0, min(n - 1, n - 1 - origin_offset))
a_lt   = float(_a_arr[origin])
b_lt   = float(_b_arr[origin])
origin_price = float(price_input[origin])
# ── Гарантия причинности ──────────────────────────────────────────────────────
# ratio обрезается с двух сторон: будущее (справа) и глубина истории (слева).
# close / logtrend / times — полные, только для визуализации.
ratio       = ratio[:origin + 1]                        # убираем будущее
if bars > 0:
    ratio   = ratio[max(0, len(ratio) - bars):]         # ограничиваем глубину истории
# Отсюда: len(ratio) = min(bars, origin+1), последний элемент ≡ origin.
# Все функции алгоритма получают только этот массив — никакого look-ahead.
origin_algo = len(ratio) - 1                            # origin внутри обрезанного ratio
# ─────────────────────────────────────────────────────────────────────────────

# ── Измерение PCA (только этап 1) ────────────────────────────────────────────

def _run_stage1(ratio, pca_p_range, xi_add, pca_thr1, pca_thr2, blend_alpha):
    pca_p_vals = list(range(pca_p_range[0], pca_p_range[1] + 1))
    with st.spinner(f"PCA sweep ({len(pca_p_vals)} точек)…"):
        s1_rows = sweep_local_dim(
            ratio,
            p_values=pca_p_vals,
            xi_add=xi_add,
            pca_thresholds=(pca_thr1, pca_thr2),
            blend_alpha=blend_alpha,
        )
    _col1 = f"PCA_{int(pca_thr1*100)}"
    _col2 = f"PCA_{int(pca_thr2*100)}"
    vals1 = [r[_col1] for r in s1_rows if np.isfinite(r[_col1])]
    vals2 = [r[_col2] for r in s1_rows if np.isfinite(r[_col2])]
    mean1 = float(np.mean(vals1)) if vals1 else None
    mean2 = float(np.mean(vals2)) if vals2 else None
    d_min = int(np.floor(mean1)) if mean1 is not None else None
    d_max = int(np.ceil(mean2))  if mean2 is not None else None
    if d_min is not None and d_max is not None:
        d_min = max(2, d_min)
        d_max = max(d_min, d_max)
    st.session_state.update({
        "s1_rows": s1_rows, "s1_col1": _col1, "s1_col2": _col2,
        "s1_mean1": mean1, "s1_mean2": mean2,
        "s1_d_min": d_min, "s1_d_max": d_max,
    })
    return d_min, d_max


if pca_btn:
    _run_stage1(ratio, pca_p_range, xi_add, pca_thr1, pca_thr2, blend_alpha)

# ── Прогон (оба этапа) ────────────────────────────────────────────────────────

if run_btn:
    # ── Этап 1: PCA sweep ────────────────────────────────────────────────────
    d_min, d_max = _run_stage1(
        ratio, pca_p_range, xi_add, pca_thr1, pca_thr2, blend_alpha,
    )

    if d_min is None or d_max is None:
        st.error("Этап 1: не удалось определить d_min/d_max (недостаточно данных).")
        st.session_state.pop("s2_results", None)
    else:
        # ── Этап 2: sweep по d ───────────────────────────────────────────────
        d_values  = list(range(d_min, d_max + 1))
        prog_bar  = st.progress(0.0)
        prog_text = st.empty()

        def _cb(step: int, total: int, d: int) -> None:
            prog_bar.progress(step / total if total > 0 else 0.0)
            prog_text.caption(f"Этап 2: {step+1}/{total} — d={d}, m={xy_x*d+xy_y}")

        with st.spinner(f"Этап 2: d∈[{d_min},{d_max}] ({len(d_values)} значений)…"):
            s2_results, s2_failed = run_stage2(
                ratio,
                d_values, xy_x, xy_y, xi_add,
                n_levels, horizon, origin_algo,
                _cb,
                blend_alpha,
                smap_theta,
                p_cascade_max,
                use_lp_corr,
            )

        prog_bar.empty(); prog_text.empty()

        st.session_state.update({
            "s2_results":  s2_results,
            "s2_failed":   s2_failed,
            "s2_d_values": d_values,
            "s2_origin":   origin,
            "s2_n":        n,
            "s2_times":    times,
            "s2_horizon":  horizon,
            "s2_a_lt":     a_lt,
            "s2_b_lt":     b_lt,
            "s2_n_eff":    origin + 1,
            "s2_origin_price": origin_price,
        })

# ── Отображение ───────────────────────────────────────────────────────────────

import pandas as pd

# — Этап 1 —
if "s1_rows" in st.session_state:
    _ss = st.session_state
    _c1, _c2 = _ss["s1_col1"], _ss["s1_col2"]
    _dm, _dM = _ss.get("s1_d_min"), _ss.get("s1_d_max")

    st.subheader("Этап 1 — PCA sweep (ratio, без LP)")
    st.caption(
        f"amp_cos α={blend_alpha}  ·  K = 3·(m+1)+{xi_add}  ·  "
        f"origin −{origin_offset}б  ·  bars={bars}"
    )

    _m1v = _ss["s1_mean1"]; _m2v = _ss["s1_mean2"]
    ca, cb, cc = st.columns(3)
    ca.metric(f"mean {_c1}", f"{_m1v:.2f}" if _m1v is not None else "—")
    cb.metric(f"mean {_c2}", f"{_m2v:.2f}" if _m2v is not None else "—")
    if _dm is not None and _dM is not None:
        cc.metric("d range", f"[{_dm}, {_dM}]")
    else:
        cc.metric("d range", "—")

    _df1 = pd.DataFrame([
        {"m": r["p_search"], "K": r["K"],
         _c1: round(r[_c1], 1) if np.isfinite(r[_c1]) else None,
         _c2: round(r[_c2], 1) if np.isfinite(r[_c2]) else None}
        for r in _ss["s1_rows"]
    ])
    st.dataframe(_df1, use_container_width=True, hide_index=True, height=220)

    _s1_dropped = [r for r in _ss["s1_rows"] if r.get("_error")]
    if _s1_dropped:
        _dp_ps     = [r["p_search"] for r in _s1_dropped]
        _dp_reason = _s1_dropped[0]["_error"]
        st.caption(
            f"⚠ Пропущено {len(_s1_dropped)} m из mean: "
            f"m={_dp_ps[0]}…{_dp_ps[-1]}  —  {_dp_reason}"
        )

# — Этап 2 —
if "s2_results" in st.session_state:
    _ss       = st.session_state
    _s2_res   = _ss["s2_results"]
    _s2_orig  = _ss["s2_origin"]
    _s2_n     = _ss["s2_n"]
    _s2_times = _ss["s2_times"]
    _s2_hor   = _ss["s2_horizon"]
    _s2_dvs   = _ss["s2_d_values"]
    _s2_origin_pr = _ss.get("s2_origin_price")
    _dm, _dM  = _ss.get("s1_d_min"), _ss.get("s1_d_max")

    n_valid = len(_s2_res)
    n_total = len(_s2_dvs)
    st.subheader(f"Этап 2 — Прогноз  (d∈[{_dm},{_dM}],  {n_valid}/{n_total} валидных)")

    # Реконструкция цены из сырых att-предсказаний (пересчитывается на каждом рендере)
    _a_lt  = _ss["s2_a_lt"]
    _b_lt  = _ss["s2_b_lt"]
    _n_eff = _ss["s2_n_eff"]

    def _recon(fc_preds: np.ndarray) -> np.ndarray:
        return np.array([
            (fc_preds[j] + j * delta_ratio) * np.exp(_a_lt + _b_lt * (_n_eff + j))
            for j in range(len(fc_preds))
        ])

    # Выпавшие d
    _s2_failed = _ss.get("s2_failed", [])
    if _s2_failed:
        from collections import defaultdict
        _by_reason: dict[str, list[int]] = defaultdict(list)
        for _f in _s2_failed:
            _by_reason[_f["reason"]].append(_f["d"])
        _fail_parts = [
            f"{len(ds)} d ({reason}): {ds[0]}…{ds[-1]}" if len(ds) > 1
            else f"1 d ({reason}): d={ds[0]}"
            for reason, ds in _by_reason.items()
        ]
        st.caption("⚠ Пропущено: " + "  |  ".join(_fail_parts))

    if not _s2_res:
        st.warning("Нет валидных прогнозов. Попробуйте изменить параметры (x, y, n_levels, bars).")
        st.stop()

    # Реконструкция и среднее
    _fc_prices = [_recon(r["fc_preds"]) for r in _s2_res]
    fc_stack   = np.array(_fc_prices)
    avg_fc     = fc_stack.mean(axis=0)
    fc_idx     = np.arange(_s2_orig + 1, _s2_orig + 1 + _s2_hor)

    # Временны́е метки будущих баров — по номинальному периоду интервала
    try:
        from datetime import timedelta as _td
        _base = pd.Timestamp(_s2_times[-1])
        _PERIOD = {
            "1m":  _td(minutes=1),  "10m": _td(minutes=10),
            "1h":  _td(hours=1),    "1w":  _td(weeks=1),
            "1mo": _td(days=30),
        }
        if interval == "1d":
            future_times = [str(_base + pd.offsets.BDay(h)) for h in range(1, _s2_hor + 1)]
        elif interval in _PERIOD:
            _p = _PERIOD[interval]
            future_times = [str(_base + _p * h) for h in range(1, _s2_hor + 1)]
        else:
            _dt = pd.Timestamp(_s2_times[-1]) - pd.Timestamp(_s2_times[-2])
            future_times = [str(_base + _dt * h) for h in range(1, _s2_hor + 1)]
    except Exception:
        future_times = [f"+{h}" for h in range(1, _s2_hor + 1)]

    def _time_at(idx: int) -> str:
        if idx < _s2_n:
            return _s2_times[idx]
        return future_times[min(idx - _s2_n, len(future_times) - 1)]

    show_from = max(0, _s2_orig - 299)

    # y-диапазон из OHLC + actual + forecast — logtrend исключён из авторасчёта
    _y_lo = float(np.min(low[show_from : _s2_orig + 1]))
    _y_hi = float(np.max(high[show_from : _s2_orig + 1]))
    if _s2_orig < _s2_n - 1:
        _ae   = min(_s2_n, _s2_orig + _s2_hor + 1)
        _y_lo = min(_y_lo, float(close[_s2_orig : _ae].min()))
        _y_hi = max(_y_hi, float(close[_s2_orig : _ae].max()))
    for _fp in _fc_prices:
        _y_lo = min(_y_lo, float(_fp.min()))
        _y_hi = max(_y_hi, float(_fp.max()))
    _y_pad = (_y_hi - _y_lo) * 0.05

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=_s2_times[show_from:],
        open=open_[show_from:], high=high[show_from:],
        low=low[show_from:],    close=close[show_from:],
        name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    fig.add_trace(go.Scatter(
        x=_s2_times[show_from : _s2_orig + 1],
        y=logtrend[show_from : _s2_orig + 1],
        mode="lines", name="logtrend",
        line=dict(color="rgba(255, 200, 50, 0.45)", width=1, dash="dot"),
        hovertemplate="%{y:.4f}<extra>logtrend</extra>",
    ))

    # Фактические цены после origin (для сравнения)
    if _s2_orig < _s2_n - 1:
        actual_end = min(_s2_n, _s2_orig + _s2_hor + 1)
        fig.add_trace(go.Scatter(
            x=_s2_times[_s2_orig:actual_end],
            y=close[_s2_orig:actual_end],
            mode="lines", name="actual",
            line=dict(color="#ffffff", width=1.5, dash="dot"),
        ))

    # Отдельные d (тонкие, полупрозрачные)
    for fc_price_i, r in zip(_fc_prices, _s2_res):
        fc_x = [_s2_times[_s2_orig]] + [_time_at(int(i)) for i in fc_idx]
        fc_y = np.concatenate([[_s2_origin_pr], fc_price_i])
        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_y,
            mode="lines", name=f"d={r['d']}",
            line=dict(width=1, color="rgba(100,180,255,0.25)"),
            showlegend=False,
        ))

    # Среднее
    avg_x = [_s2_times[_s2_orig]] + [_time_at(int(i)) for i in fc_idx]
    avg_y = np.concatenate([[_s2_origin_pr], avg_fc])
    fig.add_trace(go.Scatter(
        x=avg_x, y=avg_y,
        mode="lines+markers", name=f"среднее  ({n_valid} прогнозов)",
        line=dict(color="#ffd600", width=2.5),
        marker=dict(size=3),
    ))

    fig.add_vline(
        x=_s2_times[_s2_orig],
        line_width=1.5, line_dash="dash", line_color="#ffffff",
        annotation_text=f"origin −{origin_offset}б" if origin_offset > 0 else "origin",
        annotation_position="top left",
    )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        xaxis_type="date",
        yaxis_range=[_y_lo - _y_pad, _y_hi + _y_pad],
        title=(f"{ticker} {interval}  ·  d∈[{_dm},{_dM}]  ·  "
               f"среднее {n_valid} прогнозов  ·  horizon={_s2_hor}"),
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.18),
    )

    if fc_idx[-1] >= _s2_n:
        fig.update_layout(xaxis_range=[_s2_times[show_from], _time_at(int(fc_idx[-1]))])

    st.plotly_chart(fig, use_container_width=True)

    # Таблица результатов этапа 2
    st.subheader("Параметры этапа 2")
    _df2 = pd.DataFrame([
        {
            "d":      r["d"],
            "m":      r["m"],
            "K":      r["K"],
            "fc[0]":  round(float(fp[0]),  4),
            "fc[-1]": round(float(fp[-1]), 4),
        }
        for r, fp in zip(_s2_res, _fc_prices)
    ])
    st.dataframe(_df2, use_container_width=True, hide_index=True, height=260)
