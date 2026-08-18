"""
app6-multiorigin: ансамбль прогнозов app6-etalon по нескольким соседним origin.

Идея: уровневый (band) прогноз в sma/ уже показал себя лучше точечного —
этот прототип не пытается его заменить. Задача — оценить ПОТЕНЦИАЛ развития
событий и ОПРЕДЕЛЁННОСТЬ ситуации, а не построить более точную точку.

Метод: для origin, origin−1, …, origin−(W−1) (W = «окно») независимо
прогоняется ПОЛНЫЙ двухэтапный пайплайн app6-etalon (PCA-sweep этап 1 →
d_min/d_max → каскадный S-map этап 2, усреднение по d). Каждый origin_i
даёт свою траекторию в ОТНОСИТЕЛЬНЫХ величинах (rel[h] = price[h]/origin_price_i − 1)
на h = 1…H шагов вперёд от СВОЕГО origin_i (не выровнено по календарным датам —
это разброс формы «типичного прогноза на H баров вперёд», не выровненный
по абсолютному будущему).

Кросс-origin среднее mean_rel(h) и разброс std_rel(h)/[p25,p75](h) —
основной результат: std_rel(h) растёт с горизонтом и показывает, насколько
согласованы прогнозы из близких точек отсчёта («определённость»).

⚠ origin−1…origin−(W−1) используют почти ту же историю, что и origin — это
НЕ независимые сценарии, а локальная чувствительность прогноза к сдвигу
точки отсчёта на несколько баров. Не путать с калиброванным доверительным
интервалом.

Ядро алгоритма (метрика, logtrend, LP-фильтр, каскад, S-map, LP-коррекция)
скопировано из app6-etalon.py без изменений — расхождение только в
оркестрации (несколько origin вместо одного) и агрегации результата.

Run (из prototype/): streamlit run forcaster/ui/app6-multiorigin.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forcaster.data.moex import download_candles, save_candles, INTERVALS

DATA_DIR = _root / "data" / "candles"

# ═══════════════════════════════════════════════════════════════════════════════
# Метрика расстояния: amp_cos  (без изменений от app6-etalon)
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
# Logtrend (causal OLS) — без изменений от app6-etalon
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
# LP-фильтр (causal, ratio → att) — без изменений от app6-etalon
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _lp_proj_ratio_cached(ratio_bytes: bytes, m: int, d: int, k: int, n_iter: int,
                           blend_alpha: float = 0.5) -> np.ndarray:
    ratio = np.frombuffer(ratio_bytes, dtype=np.float64).copy()
    return _lp_proj_causal(ratio, m, d, k, n_iter, blend_alpha)

def _lp_proj_causal(ratio: np.ndarray, m: int, d: int, k: int, n_iter: int,
                    blend_alpha: float = 0.5) -> np.ndarray:
    """Причинный LP-фильтр: att[t] вычисляется только по ratio[0..t]."""
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
# Локальная размерность PCA (Этап 1) — без изменений от app6-etalon
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
# Каскад S-map (октавный ×2) — без изменений от app6-etalon
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
    """Строгий октавный каскад ×2 (cascade_algorithm.md). Без изменений от app6-etalon."""
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
# S-map + PiP (без wSVD) — без изменений от app6-etalon
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


def _simplex_step(
    X_pool: np.ndarray, y_pool: np.ndarray, vec_fit: np.ndarray,
    blend_alpha: float, n_neighbors: int,
) -> float:
    """
    Simplex projection (Sugihara & May, 1990) — альтернатива S-map: вместо
    локально-взвешенной регрессии берёт n_neighbors (канонически E+1, здесь
    d_proj+1) ближайших соседей из уже отобранного каскадом пула и предсказывает
    взвешенным средним их известных y (без подгонки коэффициентов, без PCA-проекции —
    Simplex работает прямо в пространстве задержек):
        w_i = exp(−d_i / d_min) / Σ exp(−d_j / d_min),  d_min = min_i d_i
    d_min в знаменателе экспоненты (а не θ·d_mean, как в S-map) — единственный
    параметр масштаба, шкалируется расстоянием до самого близкого соседа.
    """
    p_eff = min(X_pool.shape[1], len(vec_fit))
    Xf    = X_pool[:, -p_eff:]
    vf    = vec_fit[-p_eff:]
    d_arr = _dists(Xf, vf, blend_alpha)

    k_eff = min(n_neighbors, len(Xf))
    idx   = np.argpartition(d_arr, k_eff - 1)[:k_eff]
    d_sel = d_arr[idx]
    y_sel = y_pool[idx]

    d_min = max(float(d_sel.min()), 1e-10)
    w = np.exp(-d_sel / d_min)
    w /= w.sum()
    return float((w * y_sel).sum())

# ═══════════════════════════════════════════════════════════════════════════════
# LP-коррекция траектории — без изменений от app6-etalon
# ═══════════════════════════════════════════════════════════════════════════════

def _lp_corr(v_m: np.ndarray, X_lib: np.ndarray, k: int, d: int,
             blend_alpha: float) -> float:
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
# Прогноз: один d. Каскад (cascade_search) и всё до него — БЕЗ ИЗМЕНЕНИЙ от
# app6-etalon; изменился только финальный шаг аппроксимации пула (method).
# ═══════════════════════════════════════════════════════════════════════════════

def run_forecast(
    att: np.ndarray,
    p_fit: int, n_levels: int, K: int,
    horizon: int, origin: int,
    blend_alpha: float, smap_theta: float, d_proj: int,
    p_cascade_max: int = 1500,
    use_lp_corr: bool = False,
    method: str = "smap",
) -> np.ndarray | None:
    """Возвращает fc_preds[horizon] — сырые att-предсказания, или None при ошибке.
    method: "smap" (локально-взвешенная регрессия) или "simplex" (Simplex projection —
    взвешенное среднее d_proj+1 ближайших соседей, без подгонки коэффициентов)."""
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
        if method == "simplex":
            pred = _simplex_step(X_nn, y_nn, ctx[-p_fit:], blend_alpha, d_proj + 1)
        else:
            pred = _smap_step(X_nn, y_nn, ctx[-p_fit:], smap_theta, blend_alpha, d_proj)
        if use_lp_corr and X_lib_lp is not None:
            v_m  = np.array(fc_buf[-(p_fit - 1):] + [pred], dtype=np.float64)
            pred = _lp_corr(v_m, X_lib_lp, K, d_proj, blend_alpha)
        fc_buf.append(pred)

    return np.array(fc_buf[n_eff:])

# ═══════════════════════════════════════════════════════════════════════════════
# Этап 2: sweep по d. Только method прокинут дальше — остальное от app6-etalon.
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
    method: str = "smap",
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
            use_lp_corr, method,
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
# НОВОЕ: ансамбль по нескольким origin
# ═══════════════════════════════════════════════════════════════════════════════

def _run_stage1_for_origin(ratio_i, pca_p_range, xi_add, pca_thr1, pca_thr2, blend_alpha):
    """Этап 1 (PCA sweep) для одного origin_i. Возвращает (d_min, d_max, s1_rows) или (None, None, s1_rows)."""
    pca_p_vals = list(range(pca_p_range[0], pca_p_range[1] + 1))
    s1_rows = sweep_local_dim(
        ratio_i, p_values=pca_p_vals, xi_add=xi_add,
        pca_thresholds=(pca_thr1, pca_thr2), blend_alpha=blend_alpha,
    )
    col1 = f"PCA_{int(pca_thr1*100)}"
    col2 = f"PCA_{int(pca_thr2*100)}"
    vals1 = [r[col1] for r in s1_rows if np.isfinite(r[col1])]
    vals2 = [r[col2] for r in s1_rows if np.isfinite(r[col2])]
    if not vals1 or not vals2:
        return None, None, s1_rows
    d_min = max(2, int(np.floor(np.mean(vals1))))
    d_max = max(d_min, int(np.ceil(np.mean(vals2))))
    return d_min, d_max, s1_rows


def _compute_overlap_check(per_origin: list[dict], price_input: np.ndarray, horizon: int) -> dict | None:
    """
    Самопроверка на перекрытии: для origin_i сравнивает предсказанный rel_i(h) с
    ФАКТИЧЕСКИМ price_input[origin_i+h]/origin_price_i − 1, если этот бар уже
    наступил (origin_i + h ≤ n−1).

    Работает даже при origin_offset=0 (живой прогноз): для origin_i c i>0 часть их
    горизонта — это бары между origin_i и главным origin, которые для origin_i были
    «будущим», а для нас уже случившийся факт. Доступное перекрытие
    overlap_h_max_i = min(H, origin_offset + i) — растёт с i, поэтому на больших h
    точек всё меньше (n(h) в результате).
    """
    n = len(price_input)
    rows: list[tuple[int, int, int, float, float]] = []
    for r in per_origin:
        origin_i = r["origin_i"]
        overlap_h_max = min(horizon, n - 1 - origin_i)
        if overlap_h_max < 1:
            continue
        actual_rel = price_input[origin_i + 1 : origin_i + 1 + overlap_h_max] / r["origin_price"] - 1.0
        pred_rel = r["rel"][:overlap_h_max]
        for h in range(1, overlap_h_max + 1):
            rows.append((r["i"], origin_i, h, float(pred_rel[h - 1]), float(actual_rel[h - 1])))

    if not rows:
        return None

    max_h  = max(row[2] for row in rows)
    h_axis = np.arange(1, max_h + 1)
    n_arr   = np.zeros(max_h, dtype=int)
    mae_arr = np.full(max_h, np.nan)
    hit_arr = np.full(max_h, np.nan)
    for h in h_axis:
        errs = [abs(p - a) for (_, _, hh, p, a) in rows if hh == h]
        hits = [1.0 if np.sign(p) == np.sign(a) else 0.0
                for (_, _, hh, p, a) in rows if hh == h and p != 0 and a != 0]
        n_arr[h - 1] = len(errs)
        if errs:
            mae_arr[h - 1] = float(np.mean(errs))
        if hits:
            hit_arr[h - 1] = float(np.mean(hits))

    return {"h": h_axis, "n": n_arr, "mae": mae_arr, "hit_rate": hit_arr}


# ═══════════════════════════════════════════════════════════════════════════════
# Взвешивание по давности — считается на готовом per_origin, БЕЗ пересчёта
# Этапов 1-2. Крутится интерактивно в UI (слайдер τ), сразу обновляя combined
# траекторию — позволяет увидеть, как усреднённый прогноз меняется в зависимости
# от того, каким весом учитывать более старые origin_i.
# ═══════════════════════════════════════════════════════════════════════════════

def _recency_weights(per_origin: list[dict], tau: float | None) -> np.ndarray:
    """w_i = 0.5^(i/τ) — экспоненциальное затухание с полупериодом τ origin's
    (= баров назад, т.к. origin_i = origin − i). tau=None/<=0 → равные веса (как раньше)."""
    idx = np.array([r["i"] for r in per_origin], dtype=np.float64)
    if tau is None or tau <= 0:
        return np.ones_like(idx)
    return 0.5 ** (idx / tau)


def _fit_vector(rel_h: np.ndarray) -> tuple[float, float]:
    """
    Интерпретирует rel_i(h) НЕ как точный путь, а как направление: OLS-прямая
    через origin (rel=0 при h=0 — это определение, не предположение) даёт
    наклон b (скорость движения, доля цены за бар) и RMSE отклонений от этой
    прямой — «шум» траектории (насколько сильно фактический путь виляет вокруг
    своего же направления).
    """
    h = np.arange(1, len(rel_h) + 1, dtype=np.float64)
    b = float(np.sum(h * rel_h) / np.sum(h ** 2))
    resid = rel_h - b * h
    noise = float(np.sqrt(np.mean(resid ** 2)))
    return b, noise


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w  = values[order], weights[order]
    cw    = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cw, v))


def _weighted_aggregate(per_origin: list[dict], weights: np.ndarray) -> dict:
    """Взвешенные mean/std/p25/p75 по origin для каждого h, на основе уже
    посчитанных rel_i(h) в per_origin."""
    rel_stack = np.array([r["rel"] for r in per_origin])  # (n_origins, horizon)
    w = np.asarray(weights, dtype=np.float64)
    w_sum = max(float(w.sum()), 1e-12)

    mean_rel = (rel_stack * w[:, None]).sum(axis=0) / w_sum
    var      = ((rel_stack - mean_rel[None, :]) ** 2 * w[:, None]).sum(axis=0) / w_sum
    std_rel  = np.sqrt(np.maximum(var, 0.0))

    horizon = rel_stack.shape[1]
    p25 = np.array([_weighted_quantile(rel_stack[:, h], w, 0.25) for h in range(horizon)])
    p75 = np.array([_weighted_quantile(rel_stack[:, h], w, 0.75) for h in range(horizon)])

    return {"mean_rel": mean_rel, "std_rel": std_rel, "p25": p25, "p75": p75}


def run_multi_origin(
    ratio_full: np.ndarray,
    a_arr: np.ndarray, b_arr: np.ndarray, price_input: np.ndarray,
    origin: int, window: int, bars: int,
    xy_x: int, xy_y: int, xi_add: int, n_levels: int, p_cascade_max: int,
    pca_p_range: tuple, pca_thr1: float, pca_thr2: float, blend_alpha: float,
    horizon: int, smap_theta: float, use_lp_corr: bool,
    progress_cb=None,
    method: str = "smap",
) -> dict:
    """
    Прогоняет пайплайн app6-etalon для origin, origin−1, …, origin−(window−1).
    Каждый origin_i получает СВОЮ причинную обрезку ratio (ratio_full[:origin_i+1],
    затем bars) — никакой утечки между origin.

    Этап 1 (PCA-sweep → d_min/d_max) считается ОДИН РАЗ по главному origin и
    переиспользуется для всех origin_i в окне: соседние origin почти полностью
    делят одну и ту же историю, локальная размерность на масштабе window баров
    не должна ощутимо отличаться, а точного прогноза (для которого имело бы смысл
    подбирать d индивидуально) здесь не требуется — задача про потенциал/определённость.

    Возвращает dict с per_origin (список по origin) и агрегатами
    mean_rel/std_rel/p25/p75 по относительным траекториям rel[h] = price[h]/origin_price_i − 1.
    """
    ratio_main = ratio_full[: origin + 1]
    if bars > 0:
        ratio_main = ratio_main[max(0, len(ratio_main) - bars):]

    d_min, d_max, _s1_rows = _run_stage1_for_origin(
        ratio_main, pca_p_range, xi_add, pca_thr1, pca_thr2, blend_alpha,
    )
    if d_min is None:
        return {"error": "Этап 1 (по главному origin): не удалось определить d_min/d_max.", "skipped": []}
    d_values = list(range(d_min, d_max + 1))

    per_origin = []
    skipped: list[dict] = []

    for i in range(window):
        origin_i = origin - i
        if origin_i < 10:
            skipped.append({"origin_i": origin_i, "reason": "origin_i < 10 (мало истории)"})
            continue

        ratio_i = ratio_full[: origin_i + 1]
        if bars > 0:
            ratio_i = ratio_i[max(0, len(ratio_i) - bars):]
        origin_algo_i = len(ratio_i) - 1

        a_i = float(a_arr[origin_i])
        b_i = float(b_arr[origin_i])
        origin_price_i = float(price_input[origin_i])

        s2_results, s2_failed = run_stage2(
            ratio_i, d_values, xy_x, xy_y, xi_add,
            n_levels, horizon, origin_algo_i,
            None, blend_alpha, smap_theta, p_cascade_max, use_lp_corr, method,
        )
        if not s2_results:
            skipped.append({"origin_i": origin_i, "reason": f"Этап 2: нет валидных d (d∈[{d_min},{d_max}])"})
            if progress_cb:
                progress_cb(i + 1, window, origin_i, ok=False)
            continue

        fc_prices = []
        for r in s2_results:
            fp = r["fc_preds"]
            price_j = np.array([
                fp[j] * np.exp(a_i + b_i * (origin_i + 1 + j))
                for j in range(len(fp))
            ])
            fc_prices.append(price_j)
        avg_price_i = np.mean(fc_prices, axis=0)
        rel_i = avg_price_i / origin_price_i - 1.0

        per_origin.append({
            "i": i, "origin_i": origin_i,
            "n_valid_d": len(s2_results), "n_total_d": len(d_values),
            "s2_failed": s2_failed,
            "origin_price": origin_price_i,
            "avg_price": avg_price_i,
            "rel": rel_i,
        })
        if progress_cb:
            progress_cb(i + 1, window, origin_i, ok=True)

    if not per_origin:
        return {"error": "Ни один origin не дал валидного прогноза.", "skipped": skipped, "d_min": d_min, "d_max": d_max}

    # Агрегация (среднее/квантили) по origin здесь НЕ считается — она зависит от
    # веса по давности, который крутится интерактивно в UI поверх уже готового
    # per_origin (см. _recency_weights/_weighted_aggregate), без пересчёта Этапов 1-2.
    overlap = _compute_overlap_check(per_origin, price_input, horizon)

    return {
        "per_origin": per_origin,
        "skipped": skipped,
        "d_min": d_min, "d_max": d_max,
        "overlap": overlap,
        "horizon": horizon,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="app6-multiorigin · Потенциал и определённость", layout="wide",
                   initial_sidebar_state="expanded")

with st.sidebar:
    st.title("app6-multiorigin")
    st.caption(
        "Ансамбль app6-etalon по W соседним origin (origin, origin−1, …, origin−W+1). "
        "Цель — не точка, а оценка потенциала/определённости через разброс между origin."
    )

    ticker   = st.text_input("Тикер", "SBER", key="ticker").upper().strip()
    interval = st.selectbox("Интервал", list(INTERVALS.keys()),
                            index=list(INTERVALS.keys()).index("1d"))
    if st.button("Обновить данные"):
        _load_candles.clear()
        _fetch_and_save(ticker, interval)
        st.rerun()

    st.divider()
    with st.expander("Метод: ансамбль по origin"):
        st.markdown(
            "**Этап 1** (PCA sweep) считается ОДИН РАЗ по главному origin → единые "
            "d_min, d_max для всего окна. Соседние origin_i почти полностью делят "
            "одну и ту же историю, локальная размерность на масштабе W баров не "
            "должна ощутимо плавать, а точного прогноза (ради которого имело бы "
            "смысл подбирать d индивидуально под каждый origin_i) здесь не нужно — "
            "задача про потенциал/определённость, а не про точку.\n\n"
            "Для каждого origin_i = origin − i,  i = 0…W−1:\n\n"
            "1. **Этап 2** — каскадный S-map прогноз для d ∈ [d_min, d_max] "
            "(общий диапазон), усреднение по d → avg_price_i(h), h = 1…H "
            "(как в app6-etalon), на ratio, обрезанном каузально до origin_i.\n"
            "2. Перевод в относительные величины: "
            "rel_i(h) = avg_price_i(h) / origin_price_i − 1.\n\n"
            "**Агрегация по origin** (h — шаг от СВОЕГО origin_i, не календарная дата):\n"
            "  mean_rel(h), std_rel(h), [p25,p75](h) по W траекториям rel_i.\n\n"
            "std_rel(h), растущий с горизонтом, — мера **определённости**: "
            "чем он меньше, тем согласованнее прогнозы из близких точек отсчёта.\n\n"
            "⚠ origin_i почти полностью используют одну и ту же историю — это "
            "чувствительность к сдвигу origin, не независимые сценарии и не "
            "калиброванный доверительный интервал."
        )

    st.divider()
    st.subheader("Согласование тракта")
    xy_x = st.slider("x", 1, 10, 3, key="xy_x",
                      help="Масштабный коэффициент в m = x·d + y.")
    xy_y = st.slider("y", 0, 20, 0, key="xy_y",
                      help="Сдвиг в m = x·d + y.")
    xi_add = st.slider("ξ (добавка к K)", 0, 20, 1,
                       help="Добавка к числу соседей K = 3·(m+1) + ξ.")
    blend_alpha = st.slider("α (форма ↔ амплитуда)", 0.0, 1.0, 0.5, 0.05,
                            help="Метрика amp_cos: α·|log(‖x‖/‖y‖)|ₙ + (1−α)·(1−cos∠)ₙ.")

    st.divider()
    st.subheader("Каскадный поиск")
    n_levels = st.slider("Уровней каскада", 1, 6, 6,
                         help="Число уровней n (октавный ×2 каскад).")
    p_cascade_max = st.slider("Макс. окно каскада m(max)", 100, 5000, 5000, 100,
                              help="Уровни с mₖ > m(max) снимаются сверху.")
    use_all_bars = st.checkbox("Все точки библиотеки", value=True,
                               help="Без обрезки глубины истории (bars=0 в app6-etalon).")
    if use_all_bars:
        bars = 0
        st.caption("bars = 0 → вся доступная история (для каждого origin_i своя длина).")
    else:
        bars = st.slider("Точек в библиотеке", 100, 5000, 3000, 100)

    st.divider()
    st.subheader("Измерение размерности PCA (Этап 1, один раз по главному origin)")
    pca_p_range = st.slider("Диапазон m", 3, 300, (3, 150), key="pca_p_range",
                            help="Считается ОДИН РАЗ по главному origin (не на каждый "
                                 "origin_i в окне) — можно позволить более широкий "
                                 "диапазон для надёжной оценки d_min/d_max без "
                                 "кратного роста времени.")
    pca_thr_range = st.slider("Диапазон порогов", 0.80, 0.99, (0.80, 0.85), 0.01,
                              key="pca_thr_range")
    pca_thr1, pca_thr2 = pca_thr_range[0], pca_thr_range[1]
    st.caption(
        f"K = 3·(m+1)+{xi_add}  ·  d_min=⌊mean d(m,τ₁={pca_thr1})⌋  ·  "
        f"d_max=⌈mean d(m,τ₂={pca_thr2})⌉  ·  считается один раз, единый d-диапазон "
        f"для всех origin_i в окне"
    )

    st.divider()
    st.subheader("Ансамбль по origin")
    window = st.slider("Окно W (число origin)", 1, 30, 10,
                       help="origin, origin−1, …, origin−(W−1). Этап 1 — один раз "
                            "по главному origin; Этап 2 прогоняется W раз (по разу "
                            "на каждый origin_i, с общим d-диапазоном).")
    origin_offset = st.slider("Точка отсчёта origin (баров от конца)", 0, 500, 0,
                              help="0 = origin — последний доступный бар.")
    st.caption("⚠ Этап 2 (каскадный S-map) всё равно прогоняется W раз — при "
               "bars=все и n_levels=6 прогон может занять время, ждите прогресс-бар.")

    st.divider()
    st.subheader("Прогноз")
    horizon = st.slider("Горизонт H (баров от своего origin_i)", 1, 100, 20,
                        help="Число шагов вперёд от КАЖДОГО origin_i (не выровнено "
                             "по календарным датам между origin).")
    method_label = st.radio(
        "Аппроксиматор пула на каждом шаге каскада",
        ["S-map", "Simplex projection"], index=0, key="mo_method", horizontal=True,
        help="Каскадный поиск соседей (уровни, K, октавное сужение) не меняется — "
             "меняется только финальный шаг на отобранном пуле.\n\n"
             "S-map (Sugihara, 1994): локально-взвешенная МНК-регрессия "
             "(веса exp(−θ·ρ/ρ̄)), с проекцией пула в d-мерное подпространство перед МНК.\n\n"
             "Simplex projection (Sugihara & May, 1990): без подгонки коэффициентов — "
             "просто взвешенное среднее d+1 ближайших соседей "
             "(веса exp(−ρᵢ/ρ_min), без проекции). Меньше параметров, менее гибко "
             "к локальной кривизне, часто устойчивее на малых/шумных выборках.",
    )
    method = "simplex" if method_label == "Simplex projection" else "smap"
    smap_theta = st.slider("θ (S-map)", 0.0, 50.0, 20.0, 0.5, disabled=(method == "simplex"))
    use_lp_corr = st.checkbox("LP-коррекция траектории", value=True)

    run_btn = st.button("▶  Прогноз (ансамбль по origin)", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Предобработка")
    use_midprice = st.checkbox(
        "Входная цена (open+close)/2", value=False,
        help="Входная цена = среднее open и close вместо close.",
    )

# ── Загрузка данных ───────────────────────────────────────────────────────────

data = _load_candles(ticker, interval)
if not data:
    st.info("Нет данных. Нажмите «Обновить данные» в сайдбаре.")
    st.stop()

times, close, open_, high, low = _to_arrays(data)
price_input = (open_ + close) / 2 if use_midprice else close
logtrend, a_arr, b_arr = _logtrend_causal(price_input)
ratio_full = price_input / np.maximum(logtrend, 1e-10)
n      = len(close)
origin = max(0, min(n - 1, n - 1 - origin_offset))
# a_arr[t]/b_arr[t] — причинные коэффициенты logtrend в момент t (инкрементальные
# суммы по close[0..t]), поэтому для каждого origin_i можно просто индексировать
# a_arr[origin_i]/b_arr[origin_i] без пересчёта — сам logtrend уже каузален.

# ── Прогон ансамбля ──────────────────────────────────────────────────────────

if run_btn:
    prog_bar  = st.progress(0.0)
    prog_text = st.empty()

    def _cb(step: int, total: int, origin_i: int, ok: bool = True) -> None:
        prog_bar.progress(step / total if total > 0 else 0.0)
        status = "ок" if ok else "пропущен"
        prog_text.caption(f"origin {step}/{total} — origin_i={origin_i} ({status})")

    with st.spinner(f"Ансамбль по {window} origin (Этап 1 + Этап 2 на каждый)…"):
        mo_result = run_multi_origin(
            ratio_full, a_arr, b_arr, price_input,
            origin, window, bars,
            xy_x, xy_y, xi_add, n_levels, p_cascade_max,
            pca_p_range, pca_thr1, pca_thr2, blend_alpha,
            horizon, smap_theta, use_lp_corr,
            progress_cb=_cb, method=method,
        )

    prog_bar.empty(); prog_text.empty()

    st.session_state.update({
        "mo_result": mo_result,
        "mo_method_label": method_label,
        "mo_origin": origin,
        "mo_n": n,
        "mo_times": times,
        "mo_interval": interval,
        "mo_origin_price": float(price_input[origin]),
    })

# ── Отображение ───────────────────────────────────────────────────────────────

if "mo_result" in st.session_state:
    _ss  = st.session_state
    res  = _ss["mo_result"]
    _n   = _ss["mo_n"]
    _times = _ss["mo_times"]
    _orig  = _ss["mo_origin"]
    _orig_price = _ss["mo_origin_price"]
    _method_label = _ss.get("mo_method_label", "S-map")

    if "error" in res:
        st.error(res["error"])
        _skipped = res.get("skipped", [])
        if _skipped:
            st.caption("Пропущенные origin: " + "; ".join(
                f"origin_i={s['origin_i']} ({s['reason']})" for s in _skipped
            ))
        st.stop()

    per_origin = res["per_origin"]
    n_valid    = len(per_origin)
    _hor       = res["horizon"]

    st.subheader(f"Ансамбль по origin — {n_valid} валидных из {n_valid + len(res['skipped'])}")
    st.caption(f"d ∈ [{res['d_min']}, {res['d_max']}] — определено один раз по главному origin, общее для всех origin_i")

    _skipped = res.get("skipped", [])
    if _skipped:
        st.caption("⚠ Пропущено: " + "; ".join(
            f"origin_i={s['origin_i']} ({s['reason']})" for s in _skipped
        ))

    # ── Взвешивание по давности (интерактивно, без пересчёта Этапов 1-2) ───────
    st.subheader("Вес прогноза по удалённости origin в прошлое")
    st.caption(
        "Крутите ползунок — комбинированная траектория и полоса пересчитываются "
        "мгновенно (Этапы 1-2 не перезапускаются). w_i = 0.5^(i/τ): i — сколько "
        "баров origin_i отстоит от главного origin, τ — полупериод в барах. "
        "Малый τ → в среднем участвуют почти только ближайшие origin (прогноз "
        "быстро 'реагирует' на смещение отсчёта); большой τ / выкл. → равные веса, "
        "как раньше."
    )
    _wc1, _wc2 = st.columns([1, 2])
    with _wc1:
        use_recency = st.checkbox("Затухание по давности", value=False, key="mo_use_recency")
    with _wc2:
        tau = st.slider(
            "τ (полупериод, баров)", 1, max(2, window), max(1, window // 3),
            key="mo_tau", disabled=not use_recency,
        )
    weights = _recency_weights(per_origin, tau if use_recency else None)

    agg = _weighted_aggregate(per_origin, weights)
    band = {
        "mean_price": _orig_price * (1.0 + agg["mean_rel"]),
        "p25_price":  _orig_price * (1.0 + agg["p25"]),
        "p75_price":  _orig_price * (1.0 + agg["p75"]),
    }
    band_uniform = None
    if use_recency:
        agg_uniform = _weighted_aggregate(per_origin, np.ones(len(per_origin)))
        band_uniform = {"mean_price": _orig_price * (1.0 + agg_uniform["mean_rel"])}

    # ── Вектор направления (линейная аппроксимация, крайние веса) ──────────────
    st.subheader("Вектор направления: линейная аппроксимация")
    st.caption(
        "Каждая линия origin — не точный путь, а зашумлённая оценка направления. "
        "Здесь rel_i(h) заменяется прямой через origin (OLS без свободного члена: "
        "rel=0 при h=0 по определению) — наклон b_i и RMSE отклонений от неё "
        "(«шум» траектории). Ниже — два КРАЙНИХ комбинированных вектора: без "
        "затухания (все origin поровну) и с максимальным затуханием (τ=1 — почти "
        "только ближайшие origin). Расхождение между ними — «ускорение»: если "
        "вектор с максимальным затуханием круче в ту же сторону — недавние origin "
        "тянут сильнее среднего (тренд усиливается); если слабее или в другую "
        "сторону — сигнал ослабевает/разворачивается."
    )
    _vectors   = [_fit_vector(r["rel"]) for r in per_origin]
    _b_arr     = np.array([v[0] for v in _vectors])
    _noise_arr = np.array([v[1] for v in _vectors])

    _w_uniform  = np.ones(len(per_origin))
    _w_maxdecay = _recency_weights(per_origin, 1.0)
    b_uniform  = float((_b_arr * _w_uniform).sum()  / _w_uniform.sum())
    b_maxdecay = float((_b_arr * _w_maxdecay).sum() / _w_maxdecay.sum())
    noise_weighted_avg = float((_noise_arr * weights).sum() / max(float(weights.sum()), 1e-12))

    _vc1, _vc2, _vc3 = st.columns(3)
    _vc1.metric("Вектор: без затухания", f"{b_uniform * _hor * 100:+.2f}% за H")
    _vc2.metric("Вектор: макс. затухание (τ=1)", f"{b_maxdecay * _hor * 100:+.2f}% за H")
    _vc3.metric("Средний шум траектории", f"{noise_weighted_avg * 100:.2f}%",
               help="Взвешенное (текущим весом по давности выше) RMSE отклонений "
                    "rel_i(h) от своей же прямой — насколько виляет путь, не куда "
                    "он в среднем идёт.")

    hide_noisy_lines = st.checkbox(
        "Скрыть шумные линии origin на графике (оставить только векторы)",
        value=False, key="mo_hide_noisy",
    )

    # ── Будущие календарные метки (для отрисовки, только за пределами n) ──────
    try:
        _base = pd.Timestamp(_times[-1])
        _PERIOD = {
            "1m": timedelta(minutes=1), "10m": timedelta(minutes=10),
            "1h": timedelta(hours=1),   "1w": timedelta(weeks=1),
            "1mo": timedelta(days=30),
        }
        if interval == "1d":
            future_times = [str(_base + pd.offsets.BDay(h)) for h in range(1, _hor + 1)]
        elif interval in _PERIOD:
            _p = _PERIOD[interval]
            future_times = [str(_base + _p * h) for h in range(1, _hor + 1)]
        else:
            _dt = pd.Timestamp(_times[-1]) - pd.Timestamp(_times[-2])
            future_times = [str(_base + _dt * h) for h in range(1, _hor + 1)]
    except Exception:
        future_times = [f"+{h}" for h in range(1, _hor + 1)]

    def _time_at(idx: int) -> str:
        if idx < _n:
            return _times[idx]
        return future_times[min(idx - _n, len(future_times) - 1)]

    show_from = max(0, _orig - 299 - window)

    # ── Основной график: свечи + индивидуальные origin + комбинированный прогноз ──
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=_times[show_from:],
        open=open_[show_from:], high=high[show_from:],
        low=low[show_from:],    close=close[show_from:],
        name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    if _orig < _n - 1:
        actual_end = min(_n, _orig + _hor + 1)
        fig.add_trace(go.Scatter(
            x=_times[_orig:actual_end], y=close[_orig:actual_end],
            mode="lines", name="actual",
            line=dict(color="#ffffff", width=1.5, dash="dot"),
        ))

    # Индивидуальные origin — угасающая прозрачность по удалённости от главного origin
    if not hide_noisy_lines:
        for r in per_origin:
            i = r["i"]
            fade = 0.55 * (1.0 - i / max(1, window - 1)) + 0.08
            fc_idx_i = np.arange(r["origin_i"] + 1, r["origin_i"] + 1 + _hor)
            fc_x = [_time_at(r["origin_i"])] + [_time_at(int(idx)) for idx in fc_idx_i]
            fc_y = np.concatenate([[r["origin_price"]], r["avg_price"]])
            fig.add_trace(go.Scatter(
                x=fc_x, y=fc_y, mode="lines",
                name=f"origin−{i}" if i > 0 else "origin (главный)",
                line=dict(width=1.5 if i == 0 else 1, color=f"rgba(100,180,255,{fade:.2f})"),
                showlegend=(i == 0),
            ))

    # Два крайних вектора направления (прямая от origin, наклон b_uniform/b_maxdecay)
    _vec_x = [_time_at(_orig), _time_at(int(_orig + _hor))]
    fig.add_trace(go.Scatter(
        x=_vec_x, y=[_orig_price, _orig_price * (1.0 + b_uniform * _hor)],
        mode="lines", name=f"вектор: без затухания ({b_uniform*_hor*100:+.1f}%)",
        line=dict(color="#4fc3f7", width=2, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=_vec_x, y=[_orig_price, _orig_price * (1.0 + b_maxdecay * _hor)],
        mode="lines", name=f"вектор: макс. затухание ({b_maxdecay*_hor*100:+.1f}%)",
        line=dict(color="#ff4081", width=2, dash="dash"),
    ))

    # Комбинированная траектория и полоса — из band (режим выбирается выше:
    # относительный шаг h ИЛИ абсолютная дата/сдвиг), в абсолютной цене.
    fc_idx_main = np.arange(_orig + 1, _orig + 1 + _hor)
    main_x = [_time_at(_orig)] + [_time_at(int(idx)) for idx in fc_idx_main]
    mean_price = band["mean_price"]
    p25_price  = band["p25_price"]
    p75_price  = band["p75_price"]

    fig.add_trace(go.Scatter(
        x=main_x, y=np.concatenate([[_orig_price], p75_price]),
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        connectgaps=False,
    ))
    fig.add_trace(go.Scatter(
        x=main_x, y=np.concatenate([[_orig_price], p25_price]),
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(255,214,0,0.15)", name="p25–p75",
        hoverinfo="skip", connectgaps=False,
    ))
    if use_recency and band_uniform is not None:
        # Равновзвешенная линия для сравнения — видно, насколько τ сдвигает прогноз.
        mean_price_uniform = band_uniform["mean_price"]
        fig.add_trace(go.Scatter(
            x=main_x, y=np.concatenate([[_orig_price], mean_price_uniform]),
            mode="lines", name="равновзвешенное (для сравнения)",
            line=dict(color="rgba(255,214,0,0.4)", width=1.5, dash="dot"),
            connectgaps=False,
        ))
    fig.add_trace(go.Scatter(
        x=main_x, y=np.concatenate([[_orig_price], mean_price]),
        mode="lines+markers",
        name=f"взвешенное среднее (τ={tau})" if use_recency else f"среднее по {n_valid} origin",
        line=dict(color="#ffd600", width=2.5), marker=dict(size=3),
        connectgaps=False,
    ))

    fig.add_vline(
        x=_time_at(_orig), line_width=1.5, line_dash="dash", line_color="#ffffff",
        annotation_text="origin", annotation_position="top left",
    )

    _y_lo = float(np.min(low[show_from:_orig + 1]))
    _y_hi = float(np.max(high[show_from:_orig + 1]))
    for r in per_origin:
        _y_lo = min(_y_lo, float(r["avg_price"].min()))
        _y_hi = max(_y_hi, float(r["avg_price"].max()))
    _vec_ends = [_orig_price * (1.0 + b_uniform * _hor), _orig_price * (1.0 + b_maxdecay * _hor)]
    _y_lo = min(_y_lo, min(_vec_ends))
    _y_hi = max(_y_hi, max(_vec_ends))
    _y_pad = (_y_hi - _y_lo) * 0.05

    fig.update_layout(
        height=600, xaxis_rangeslider_visible=False, xaxis_type="date",
        yaxis_range=[_y_lo - _y_pad, _y_hi + _y_pad],
        title=f"{ticker} {interval}  ·  W={n_valid} origin  ·  H={_hor}  ·  {_method_label}",
        template="plotly_dark", legend=dict(orientation="h", y=-0.18),
    )
    if fc_idx_main[-1] >= _n:
        fig.update_layout(xaxis_range=[_times[show_from], _time_at(int(fc_idx_main[-1]))])

    st.plotly_chart(fig, use_container_width=True)

    # ── Самопроверка на перекрытии ─────────────────────────────────────────────
    st.subheader("Самопроверка на перекрытии")
    st.caption(
        "Для origin_i c i>0 часть их горизонта — бары между origin_i и главным origin, "
        "которые для origin_i были «будущим», а на самом деле уже случились. Здесь "
        "предсказанный rel_i(h) сравнивается с ФАКТИЧЕСКИМ движением цены на этих "
        "барах — недорогой walk-forward без отдельного бэктеста. Работает и при "
        "origin_offset=0. Число точек n(h) падает с ростом h (перекрытие есть только "
        "у origin с i ≥ h) — большие h в хвосте статистически ненадёжны, смотреть "
        "в первую очередь на малые h."
    )
    _overlap = res.get("overlap")
    if _overlap is None:
        st.info(
            "Нет перекрытия при текущих настройках — увеличьте окно W или "
            "точку отсчёта origin (баров от конца), чтобы у части origin_i "
            "появились уже наступившие бары для сравнения."
        )
    else:
        _df_ov = pd.DataFrame({
            "h": _overlap["h"],
            "n (число origin)": _overlap["n"],
            "MAE, % (|прогноз − факт|)": np.round(_overlap["mae"] * 100, 2),
            "hit-rate направления, %": np.round(_overlap["hit_rate"] * 100, 1),
        })
        st.dataframe(_df_ov, use_container_width=True, hide_index=True,
                    height=min(360, 40 + 35 * len(_df_ov)))

    # ── Таблица по origin ──────────────────────────────────────────────────────
    st.subheader("Параметры по origin")
    _w_norm = weights / max(float(weights.sum()), 1e-12)
    _df = pd.DataFrame([
        {
            "origin_i": r["origin_i"],
            "смещение": f"−{r['i']}" if r["i"] > 0 else "0 (главный)",
            "вес w_i": round(float(weights[k]), 3),
            "вклад, %": round(float(_w_norm[k]) * 100, 1),
            "d валидных": f"{r['n_valid_d']}/{r['n_total_d']}",
            "rel(H), %": round(float(r["rel"][-1]) * 100, 2),
        }
        for k, r in enumerate(per_origin)
    ])
    st.dataframe(_df, use_container_width=True, hide_index=True, height=min(400, 40 + 35 * len(_df)))
