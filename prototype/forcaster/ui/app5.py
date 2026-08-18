"""
app5: Каскадный прогноз по ratio с перебором d.

Стек: logtrend → LP(ratio) → smoothed_ratio → cascade LWR → прогноз ratio → цена.
Каскад: строгая воронка (cascade_algorithm.md), суб-векторы скользящим окном.
Гейт: d ≥ x·LB+y (настраивается), K_LB_MAX = xi_lwr (нет потолка размерности).
Нет acc_ang. Нет утечки из будущего: библиотека на шаге t ограничена s+p_fit < t.

Run (из prototype/): streamlit run forcaster/ui/app5.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from forcaster.data.moex import download_candles, save_candles, INTERVALS

DATA_DIR = _root / "data" / "candles"

# ═══════════════════════════════════════════════════════════════════════════════
# Метрики расстояния
# ═══════════════════════════════════════════════════════════════════════════════

def _geodesic_dists(X: np.ndarray, q: np.ndarray, k_geo: int) -> np.ndarray:
    """Геодезические расстояния от q до каждой строки X через k-NN граф (L2 рёбра)."""
    n = len(X)
    pts = np.vstack([q[None, :], X])       # (n+1, p), узел 0 = запрос
    n_pts = len(pts)
    k_eff = min(k_geo, n_pts - 1)

    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)

    ri, ci, di = [], [], []
    for i in range(n_pts):
        nn = np.argpartition(D[i], k_eff)[:k_eff]
        ri.extend([i] * k_eff); ci.extend(nn.tolist()); di.extend(D[i, nn].tolist())

    adj = csr_matrix((di, (ri, ci)), shape=(n_pts, n_pts))
    adj = adj.maximum(adj.T)
    geo = shortest_path(adj, method="D", directed=False, indices=0)

    result = geo[1:].copy()
    inf_mask = ~np.isfinite(result)
    if inf_mask.any():                     # fallback для недостижимых узлов
        result[inf_mask] = np.linalg.norm(X[inf_mask] - q, axis=1)
    return result


def _dists(
    X: np.ndarray,
    q: np.ndarray,
    metric: str = "euclidean",
    k_geo: int = 10,
    geo_pool_max: int | None = None,
    blend_alpha: float = 0.5,
) -> np.ndarray:
    """
    Расстояния от каждой строки X до вектора q.
    blend: α·d_eucl_norm + (1−α)·d_cosine_norm, оба компонента нормализованы к [0,1].
    geo_pool_max: если len(X) > значения и metric="geodesic" → fallback на euclidean.
    """
    n = len(X)

    if metric in ("blend", "amp_cos"):
        nX = np.linalg.norm(X, axis=1)
        nq = np.linalg.norm(q)
        # shape: косинусное расстояние
        if nq < 1e-10:
            d_shape = np.ones(n)
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                sim = np.where(nX > 1e-10, (X @ q) / (nX * nq), 0.0)
            d_shape = 1.0 - sim.clip(-1.0, 1.0)
        if metric == "blend":
            # amplitude: L2
            d_amp = np.linalg.norm(X - q, axis=1)
        else:
            # amplitude: |log(‖x‖/‖y‖)| — разница в масштабе
            with np.errstate(invalid="ignore", divide="ignore"):
                d_amp = np.where(
                    (nX > 1e-10) & (nq > 1e-10),
                    np.abs(np.log(nX / nq)),
                    np.abs(nX - nq),
                )
        max_a = max(float(d_amp.max()), 1e-10)
        max_s = max(float(d_shape.max()), 1e-10)
        return blend_alpha * (d_amp / max_a) + (1.0 - blend_alpha) * (d_shape / max_s)

    if metric == "geodesic":
        if geo_pool_max is not None and n > geo_pool_max:
            metric = "euclidean"
        elif n < k_geo + 2:
            metric = "euclidean"
        else:
            return _geodesic_dists(X, q, k_geo)

    if metric == "cosine":
        nX = np.linalg.norm(X, axis=1)
        nq = np.linalg.norm(q)
        if nq < 1e-10:
            return np.ones(n)
        with np.errstate(invalid="ignore", divide="ignore"):
            sim = np.where(nX > 1e-10, (X @ q) / (nX * nq), 0.0)
        return 1.0 - sim.clip(-1.0, 1.0)

    if metric == "mahalanobis":
        if n <= X.shape[1] + 1:
            return np.linalg.norm(X - q, axis=1)
        try:
            VI = np.linalg.pinv(np.cov(X.T))
            d  = X - q
            return np.sqrt(np.einsum("ij,jk,ik->i", d, VI, d).clip(0))
        except Exception:
            return np.linalg.norm(X - q, axis=1)

    return np.linalg.norm(X - q, axis=1)   # euclidean


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
# Logtrend
# ═══════════════════════════════════════════════════════════════════════════════

def _logtrend_causal(close: np.ndarray):
    """Causal OLS logtrend. Возвращает (trend, a_final, b_final)."""
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
    return trend, float(a[-1]), float(b[-1])

def _lintrend_causal(close: np.ndarray):
    """Causal OLS linear trend. Возвращает (trend, a_final, b_final)."""
    n  = len(close)
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(close.astype(np.float64))
    cty = np.cumsum(t * close.astype(np.float64))
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a     = (cy - b * ct) / cn
    trend = a + b * t
    trend[:2] = close[:2]
    return trend, float(a[-1]), float(b[-1])

# ═══════════════════════════════════════════════════════════════════════════════
# LP-фильтр (на ratio → smoothed ratio, без diff)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _lp_proj_ratio_cached(ratio_bytes: bytes, m: int, d: int, k: int, n_iter: int,
                          metric: str = "euclidean", blend_alpha: float = 0.5) -> np.ndarray:
    ratio = np.frombuffer(ratio_bytes, dtype=np.float64).copy()
    return _lp_proj_causal(ratio, m, d, k, n_iter, metric, blend_alpha)

def _lp_proj_causal(ratio: np.ndarray, m: int, d: int, k: int, n_iter: int,
                    metric: str = "euclidean", blend_alpha: float = 0.5) -> np.ndarray:
    """
    Причинный Local Projective фильтр.
    att[t] вычисляется только по ratio[0..t]:
      - текущее окно [t-m+1..t] проецируется на локальное подпространство
      - соседи = исторические окна, завершившиеся строго до t (start < t-m+1)
      - att[t] = proj[-1] (последний элемент спроецированного окна)
    Без overlap-усреднения: каждая позиция обновляется ровно один раз.
    """
    n     = len(ratio)
    d_eff = min(d, m - 1)
    att   = ratio.copy().astype(np.float64)
    if d_eff < 1 or n < m + 1:
        return att

    nw = n - m + 1  # число окон

    for _ in range(n_iter):
        X_all   = att[np.arange(nw)[:, None] + np.arange(m)]  # (nw, m)
        new_att = att.copy()

        for q in range(1, nw):
            X_hist = X_all[:q]           # исторические окна (start < q)
            curr   = X_all[q]
            k_eff  = min(k, len(X_hist))
            if k_eff < d_eff + 1:
                continue
            # LP-фильтр не поддерживает geodesic (слишком медленно в цикле по барам)
            lp_metric = metric if metric != "geodesic" else "euclidean"
            dists  = _dists(X_hist, curr, lp_metric, blend_alpha=blend_alpha)
            sel    = np.argpartition(dists, k_eff - 1)[:k_eff]
            nn     = X_hist[sel]
            center = nn.mean(0)
            _, _, Vt = np.linalg.svd(nn - center, full_matrices=False)
            proj   = center + Vt[:d_eff].T @ (Vt[:d_eff] @ (curr - center))
            new_att[q + m - 1] = proj[-1]  # причинное обновление только последней позиции

        att = new_att

    return att

# ═══════════════════════════════════════════════════════════════════════════════
# Levina-Bickel (K_LB_MAX = xi_lwr)
# ═══════════════════════════════════════════════════════════════════════════════

def _lb_query(X_all: np.ndarray, k_lb: int) -> float:
    """
    LB оценка размерности для X_all[0] (запрос).
    X_all: (1+xi_lwr, p_lv) — строка 0 = запрос, остальные = соседи.
    """
    n = len(X_all)
    k = max(3, min(k_lb, n - 2))
    diff = X_all[:, None, :] - X_all[None, :, :]
    D    = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(D, np.inf)
    D_s  = np.sort(D, axis=1)
    rk   = D_s[0, k - 1]
    rj   = D_s[0, :k - 1]
    if rk <= 1e-14 or np.any(rj <= 1e-14):
        return np.nan
    return (k - 2) / float(np.sum(np.log(rk / rj)))

def _lb_global(query: np.ndarray, signal: np.ndarray,
               k: int, bars: int, n_eff: int) -> float:
    """LB на глобальной библиотеке без каскадной фильтрации."""
    p = len(query)
    max_s = min(n_eff - p - 1, len(signal) - p)
    if max_s < 0:
        return np.nan
    min_s = max(0, max_s + 1 - bars)
    starts = np.arange(min_s, max_s + 1)
    if len(starts) < 3:
        return np.nan
    lib   = signal[starts[:, None] + np.arange(p)]
    k_eff = min(k, len(lib))
    if k_eff < 3:
        return np.nan
    sel   = np.argpartition(np.linalg.norm(lib - query, axis=1), k_eff - 1)[:k_eff]
    return _lb_query(np.vstack([query[None, :], lib[sel]]), k_eff)

# ═══════════════════════════════════════════════════════════════════════════════
# Локальная размерность (PCA)
# ═══════════════════════════════════════════════════════════════════════════════

def _pca_dim_local(X_nn: np.ndarray, threshold: float = 0.90) -> float:
    """Число главных компонент, объясняющих >= threshold дисперсии."""
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
    origin: int,
    p_search: int,
    k_dim: int,
    bars: int,
    min_lag: int = 0,
    pca_thresholds: tuple = (0.95, 0.97),
    metric: str = "euclidean",
    k_geo: int = 10,
    geo_pool_max: int = 300,
    blend_alpha: float = 0.5,
) -> dict:
    """Оценка PCA-размерности вокруг origin по raw ratio (без LP, плоский K-NN)."""
    n_eff = origin + 1

    if n_eff < p_search:
        return {"error": "недостаточно истории для запроса"}

    query = ratio[n_eff - p_search : n_eff].astype(np.float64)

    max_s = n_eff - 1 - min_lag - p_search
    if max_s < 0:
        return {"error": "библиотека пуста (min_lag слишком велик)"}

    min_s = max(0, max_s + 1 - bars)
    n_lib = max_s - min_s + 1
    if n_lib < k_dim:
        return {"error": f"библиотека мала: {n_lib} < k_dim={k_dim}"}

    starts = np.arange(min_s, max_s + 1)
    X_lib  = ratio[starts[:, None] + np.arange(p_search)].astype(np.float64)
    dists  = _dists(X_lib, query, metric, k_geo, geo_pool_max, blend_alpha)
    k_eff  = min(k_dim, len(X_lib))
    X_nn   = X_lib[np.argpartition(dists, k_eff - 1)[:k_eff]]

    return {f"pca_{t}": _pca_dim_local(X_nn, t) for t in pca_thresholds}


def sweep_local_dim(
    ratio: np.ndarray,
    origin: int,
    p_values: list[int],
    xi_add: int,
    bars: int,
    min_lag: int = 0,
    pca_thresholds: tuple = (0.95, 0.97),
    metric: str = "euclidean",
    k_geo: int = 10,
    geo_pool_max: int = 300,
    blend_alpha: float = 0.5,
) -> list[dict]:
    rows = []
    for p in p_values:
        k_dim = 3 * (p + 1) + xi_add
        res   = compute_local_dim(
            ratio, origin, p, k_dim, bars, min_lag, pca_thresholds,
            metric, k_geo, geo_pool_max, blend_alpha,
        )
        row = {"p_search": p, "K": k_dim}
        if "error" in res:
            for t in pca_thresholds:
                row[f"PCA_{int(t*100)}"] = float("nan")
        else:
            for t in pca_thresholds:
                row[f"PCA_{int(t*100)}"] = res[f"pca_{t}"]
        rows.append(row)
    return rows

# ═══════════════════════════════════════════════════════════════════════════════
# Каскад (cascade_algorithm.md)
# ═══════════════════════════════════════════════════════════════════════════════

def _cascade_levels(p_fit: int, n_levels: int, factor: int = 2) -> list[int]:
    return [p_fit * (factor ** (n_levels - 1 - k)) for k in range(n_levels)]

def cascade_search(
    att: np.ndarray,        # LP-сглаженный ratio (фиксированная библиотека)
    ratio: np.ndarray,      # raw ratio до LP
    context: np.ndarray,    # буфер прогноза (att + предсказания)
    p_fit: int,
    n_levels: int,
    xi_lwr: int,
    bars: int,
    t_predict: int,         # индекс предсказываемого шага; библиотека: s+p_lv < t_predict
    metric: str = "euclidean",
    k_geo: int = 10,
    geo_pool_max: int = 300,
    blend_alpha: float = 0.5,
    cascade_factor: int = 2,
    smap_mode: bool = False,  # если True — последний уровень возвращает весь пул (для S-map)
) -> tuple[np.ndarray | None, np.ndarray | None, list[float], list[float]]:
    """
    Строгий каскад из cascade_algorithm.md.

    Уровни: p_lv_k = p_fit × 2^(N-1-k).
    Каждый уровень:
      1. Ищет xi_lwr ближайших в пуле по p_lv_k-мерному расстоянию (att).
      2. Каждый найденный вектор нарезается на суб-векторы p_lv_{k+1} скользящим окном.
      3. Объединение суб-векторов = пул следующего уровня (дедупликация по start-индексу).
    Гарантия корректности: s+p_lv < t_predict для всех точек пула (нет самосовпадения).

    Возвращает: (X_nn, y_nn, lb_att_levels, lb_ratio_levels)
      lb_att_levels[k]   = LB на att-векторах соседей уровня k (поиск по att)
      lb_ratio_levels[k] = LB на ratio-векторах тех же соседей (запрос из ratio)
    """
    n      = len(att)
    levels = _cascade_levels(p_fit, n_levels, cascade_factor)
    p_top  = levels[0]

    # Пул уровня 1: s + p_top ≤ t_predict - 1 → запрос att[t-p_top:t] не совпадает с библиотечным вектором
    # (p_top-ограничение, а не p_fit, исключает самосовпадение на L1–L3)
    max_s_window = n - p_top                   # максимально допустимый s (включительно)
    max_s_target = t_predict - p_top - 1       # нет перекрытия вектора библиотеки с запросом
    max_s = min(max_s_window, max_s_target)
    if max_s < 0:
        return None, None, [], []

    min_s = max(0, max_s + 1 - bars)
    pool  = np.arange(min_s, max_s + 1)
    if len(pool) < xi_lwr:
        return None, None, [], []

    lb_att_levels:   list[float] = []
    lb_ratio_levels: list[float] = []

    for level_idx, p_lv in enumerate(levels):
        is_last = (level_idx == n_levels - 1)

        # Запрос по att: последние p_lv значений контекста
        if len(context) < p_lv:
            return None, None, [], []
        q = context[-p_lv:].astype(np.float64)

        idx   = pool[:, None] + np.arange(p_lv)
        valid = (pool + p_lv - 1) < n
        pool  = pool[valid]; idx = idx[valid]
        if len(pool) == 0:
            return None, None, [], []

        X_pool = att[idx]
        dists  = _dists(X_pool, q, metric, k_geo, geo_pool_max, blend_alpha)

        n_sel      = min(xi_lwr, len(pool))
        top_i      = np.argpartition(dists, n_sel - 1)[:n_sel]
        sel_starts = pool[top_i]
        X_sel      = X_pool[top_i]

        # LB(L) — на att-векторах соседей, выбранных каскадом
        lb_lv_att = _lb_query(np.vstack([q[None, :], X_sel]), max(3, n_sel - 1))
        lb_att_levels.append(lb_lv_att)

        # LB(R) — на ratio-векторах тех же соседей; запрос из ratio в том же окне
        q_r_start = t_predict - p_lv
        if (q_r_start >= 0 and q_r_start + p_lv <= len(ratio)
                and np.all(sel_starts + p_lv - 1 < len(ratio))):
            q_ratio   = ratio[q_r_start: q_r_start + p_lv].astype(np.float64)
            ratio_sel = ratio[sel_starts[:, None] + np.arange(p_lv)]
            lb_lv_ratio = _lb_query(np.vstack([q_ratio[None, :], ratio_sel]), max(3, n_sel - 1))
        else:
            lb_lv_ratio = np.nan
        lb_ratio_levels.append(lb_lv_ratio)

        if is_last:
            if smap_mode:
                # Возвращаем весь пул последнего уровня; LB считается на нём
                valid_y  = pool + p_fit < n
                pool_sm  = pool[valid_y]
                X_sm     = X_pool[valid_y]
                if len(pool_sm) < 2:
                    return None, None, lb_att_levels, lb_ratio_levels
                lb_sm = _lb_query(np.vstack([q[None, :], X_sm]), max(3, len(pool_sm) - 1))
                lb_att_levels[-1]   = lb_sm
                lb_ratio_levels[-1] = np.nan   # пересчёт ratio-LB не нужен для S-map
                return X_sm, att[pool_sm + p_fit], lb_att_levels, lb_ratio_levels
            # Фильтр: цель att[s + p_fit] должна быть в границах массива
            valid_y    = sel_starts + p_fit < n
            sel_starts = sel_starts[valid_y]
            X_sel      = X_sel[valid_y]
            if len(X_sel) < 2:
                return None, None, lb_att_levels, lb_ratio_levels
            y_nn = att[sel_starts + p_fit]
            return X_sel, y_nn, lb_att_levels, lb_ratio_levels

        # Расширение до следующего уровня скользящим окном
        p_next   = levels[level_idx + 1]
        r        = p_lv - p_next
        offsets  = np.arange(r + 1)
        expanded = sel_starts[:, None] + offsets[None, :]
        flat     = expanded.ravel()

        max_s_next = min(n - p_next, t_predict - p_next - 1)
        flat       = flat[(flat >= 0) & (flat <= max_s_next)]
        pool       = np.unique(flat)
        if len(pool) == 0:
            return None, None, [], []

    return None, None, [], []  # не достигается

# ═══════════════════════════════════════════════════════════════════════════════
# LWR
# ═══════════════════════════════════════════════════════════════════════════════

def _pip_project(Xf: np.ndarray, vf: np.ndarray,
                 d_proj: int,
                 w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """PCA-проекция соседей и запроса на d_proj главных компонент.
    w: если передан — взвешенный центр и взвешенный SVD (wSVD)."""
    if w is not None:
        w_sum  = max(float(w.sum()), 1e-10)
        center = (w[:, None] * Xf).sum(0) / w_sum
        sw     = np.sqrt(w / w_sum)
        _, _, Vt = np.linalg.svd(sw[:, None] * (Xf - center), full_matrices=False)
    else:
        center   = Xf.mean(axis=0)
        _, _, Vt = np.linalg.svd(Xf - center, full_matrices=False)
    V = Vt[:d_proj].T          # (p_eff, d_proj)
    return (Xf - center) @ V, (vf - center) @ V


def _lwr_step(X_nn: np.ndarray, y_nn: np.ndarray, vec_fit: np.ndarray,
              metric: str = "euclidean", k_geo: int = 10,
              blend_alpha: float = 0.5,
              d_proj: int | None = None,
              pip_weighted: bool = False) -> float:
    """LWR: локальная линейная регрессия с гауссовыми весами. h_bw = max dist."""
    p_eff = min(X_nn.shape[1], len(vec_fit))
    Xf    = X_nn[:, -p_eff:]
    vf    = vec_fit[-p_eff:]
    d_arr = _dists(Xf, vf, metric, k_geo, blend_alpha=blend_alpha)
    h_bw  = max(float(d_arr.max()), 1e-10)
    w     = np.exp(-0.5 * (d_arr / h_bw) ** 2)
    if d_proj is not None and d_proj < p_eff and len(Xf) > d_proj:
        Xf, vf = _pip_project(Xf, vf, d_proj, w if pip_weighted else None)
    A     = np.hstack([np.ones((len(Xf), 1)), Xf])
    sw    = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vf @ c[1:])

def _smap_step(X_pool: np.ndarray, y_pool: np.ndarray, vec_fit: np.ndarray,
               theta: float = 2.0,
               metric: str = "euclidean", k_geo: int = 10,
               blend_alpha: float = 0.5,
               d_proj: int | None = None,
               pip_weighted: bool = False) -> float:
    """S-map: регрессия по всему пулу, w = exp(−θ · d / d̄). θ=0 → глобальная регрессия."""
    p_eff = min(X_pool.shape[1], len(vec_fit))
    Xf    = X_pool[:, -p_eff:]
    vf    = vec_fit[-p_eff:]
    d_arr = _dists(Xf, vf, metric, k_geo, blend_alpha=blend_alpha)
    d_mean = max(float(d_arr.mean()), 1e-10)
    w     = np.exp(-theta * d_arr / d_mean)
    if d_proj is not None and d_proj < p_eff and len(Xf) > d_proj:
        Xf, vf = _pip_project(Xf, vf, d_proj, w if pip_weighted else None)
    A     = np.hstack([np.ones((len(Xf), 1)), Xf])
    sw    = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_pool, rcond=None)
    return float(c[0] + vf @ c[1:])

# ═══════════════════════════════════════════════════════════════════════════════
# LP-коррекция (опционально)
# ═══════════════════════════════════════════════════════════════════════════════

def _lp_corr(v_m: np.ndarray, X_lib_m: np.ndarray, k: int, d: int,
             metric: str = "euclidean", k_geo: int = 10,
             blend_alpha: float = 0.5) -> float:
    """Проецирует m-мерный вектор на локальное d-мерное подпространство. Возвращает [-1]."""
    v = v_m.astype(float)
    k_eff = min(k, len(X_lib_m) - 1)
    if k_eff < d + 1:
        return float(v[-1])
    dists  = _dists(X_lib_m, v, metric, k_geo, blend_alpha=blend_alpha)
    idx    = np.argpartition(dists, k_eff)[:k_eff]
    X_nn   = X_lib_m[idx]
    center = X_nn.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_nn - center, full_matrices=False)
    d_eff  = min(d, len(Vt))
    Vd     = Vt[:d_eff].T
    v_c    = v - center
    v_proj = center + Vd @ (Vd.T @ v_c)
    return float(v_proj[-1])

# ═══════════════════════════════════════════════════════════════════════════════
# Один прогноз (origin + val)
# ═══════════════════════════════════════════════════════════════════════════════

def run_forecast(
    att: np.ndarray,        # LP-smoothed ratio, length n
    ratio: np.ndarray,      # raw ratio (для LB до LP)
    logtrend: np.ndarray,   # тренд для реконструкции цен (length n)
    a_lt: float, b_lt: float,  # OLS-коэффициенты для экстраполяции за пределы n
    p_fit: int,
    n_levels: int,
    xi_lwr: int,
    horizon: int,
    bars: int,
    use_lp_corr: bool,
    lp_m: int, lp_d: int, lp_k: int,
    origin: int,            # индекс последнего известного бара
    metric: str = "euclidean",
    k_geo: int = 10,
    geo_pool_max: int = 300,
    blend_alpha: float = 0.5,
    cascade_factor: int = 2,
    fixed_manifold: bool = False,
    smap_mode: bool = False,
    smap_theta: float = 2.0,
    d_proj: int | None = None,
    trend_is_log: bool = True,
    pip_weighted: bool = False,
) -> dict:
    n     = len(att)
    n_eff = origin + 1      # «текущее время» — первый индекс за пределами истории

    # LP-коррекция: библиотека ограничена историей до origin
    X_lib_lp: np.ndarray | None = None
    if use_lp_corr and lp_m >= 2:
        n_lib = n_eff - lp_m
        if n_lib > 0:
            idx_lp  = np.arange(n_lib)[:, None] + np.arange(lp_m)
            X_lib_lp = att[idx_lp]

    _lwr_min = (d_proj + 2) if (d_proj is not None) else (p_fit + 2)

    def _predict_one(context: np.ndarray, t: int):
        X_nn, y_nn, lb_att, lb_ratio = cascade_search(
            att, ratio, context, p_fit, n_levels, xi_lwr, bars, t,
            metric, k_geo, geo_pool_max, blend_alpha, cascade_factor, smap_mode)
        if X_nn is None or len(X_nn) < _lwr_min:
            raise ValueError(f"pool too small at t={t}: {0 if X_nn is None else len(X_nn)} < {_lwr_min}")
        vec_fit = context[-p_fit:]
        if smap_mode:
            pred = _smap_step(X_nn, y_nn, vec_fit, smap_theta, metric, k_geo, blend_alpha, d_proj, pip_weighted)
        else:
            pred = _lwr_step(X_nn, y_nn, vec_fit, metric, k_geo, blend_alpha, d_proj, pip_weighted)
        if use_lp_corr and X_lib_lp is not None and lp_m >= 2 and len(context) >= lp_m - 1:
            v_m  = np.append(context[-(lp_m - 1):], pred)
            pred = _lp_corr(v_m, X_lib_lp, lp_k, lp_d, metric, k_geo, blend_alpha)
        return pred, lb_att, lb_ratio

    # ── Прогноз от origin ────────────────────────────────────────────────────
    fc_buf: list[float] = list(att[:n_eff])
    fc_preds: list[float] = []
    lb_cascade_att:   list[float] = []  # LB(L) по уровням (только h=0)
    lb_cascade_ratio: list[float] = []  # LB(R) по уровням (только h=0)

    if fixed_manifold:
        # Каскад запускается один раз; LWR итерирует в фиксированном многообразии
        X_nn_fix, y_nn_fix, lb_att_fix, lb_ratio_fix = cascade_search(
            att, ratio, np.array(fc_buf), p_fit, n_levels, xi_lwr, bars, n_eff,
            metric, k_geo, geo_pool_max, blend_alpha, cascade_factor)
        lb_cascade_att   = lb_att_fix
        lb_cascade_ratio = lb_ratio_fix
        if X_nn_fix is None or len(X_nn_fix) < _lwr_min:
            raise ValueError(f"fixed_manifold pool too small: {0 if X_nn_fix is None else len(X_nn_fix)} < {_lwr_min}")
        for h in range(horizon):
            ctx = np.array(fc_buf)
            vec_fit = ctx[-p_fit:]
            pred    = _lwr_step(X_nn_fix, y_nn_fix, vec_fit, metric, k_geo, blend_alpha, d_proj, pip_weighted)
            if use_lp_corr and X_lib_lp is not None and lp_m >= 2 and len(ctx) >= lp_m - 1:
                v_m  = np.append(ctx[-(lp_m - 1):], pred)
                pred = _lp_corr(v_m, X_lib_lp, lp_k, lp_d, metric, k_geo, blend_alpha)
            fc_buf.append(pred)
            fc_preds.append(pred)
    else:
        for h in range(horizon):
            t   = n_eff + h
            ctx = np.array(fc_buf)
            pred, lb_att, lb_rat = _predict_one(ctx, t)
            if h == 0:
                lb_cascade_att   = lb_att
                lb_cascade_ratio = lb_rat
            fc_buf.append(pred)
            fc_preds.append(pred)

    # ── Реконструкция цен ─────────────────────────────────────────────────────
    def trend_at(i: int) -> float:
        if i < n:
            return float(logtrend[i])
        return float(np.exp(a_lt + b_lt * i)) if trend_is_log else float(a_lt + b_lt * i)

    fc_idx   = np.arange(n_eff, n_eff + horizon)
    fc_price = np.array([fc_preds[j] * trend_at(n_eff + j) for j in range(horizon)])

    return {
        "lb_cascade_att":   lb_cascade_att,    # list[float]: LB(L) на att по уровням
        "lb_cascade_ratio": lb_cascade_ratio,  # list[float]: LB(R) на ratio по уровням
        "fc_price":         fc_price,
        "fc_idx":           fc_idx,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Перебор d
# ═══════════════════════════════════════════════════════════════════════════════

def run_sweep(
    ratio: np.ndarray,
    close: np.ndarray,
    logtrend: np.ndarray,
    a_lt: float, b_lt: float,
    d_values: list[int],
    n_iter_map: list[tuple],
    xy_x: int,
    xy_y: int,
    xi_add: int,
    n_levels: int,
    bars: int,
    horizon: int,
    use_lp_corr: bool,
    origin: int,
    progress_cb=None,
    metric: str = "euclidean",
    k_geo: int = 10,
    geo_pool_max: int = 300,
    blend_alpha: float = 0.5,
    cascade_factor: int = 2,
    fixed_manifold: bool = False,
    smap_mode: bool = False,
    smap_theta: float = 2.0,
    d_proj_offset: int | None = None,
    val_horizon: int = 0,
    trend_is_log: bool = True,
    pip_weighted: bool = False,
) -> list[dict]:
    results = []
    tasks   = []

    # Формируем список задач (d, n_iter)
    for d in d_values:
        for (n_iter, d_min, d_max) in n_iter_map:
            if d_min <= d <= d_max:
                tasks.append((d, n_iter))
    tasks = list(dict.fromkeys(tasks))  # дедупликация с сохранением порядка

    for step_i, (d, n_iter) in enumerate(tasks):
        if progress_cb:
            progress_cb(step_i, len(tasks), d, n_iter)

        p_fit = xy_x * d + xy_y
        m     = p_fit
        xi    = 3 * (p_fit + 1) + xi_add
        k     = xi

        if p_fit < 2 or m < d + 1 or k < d + 1:
            continue

        # LP-фильтр (кэш по байтам + параметрам + метрике)
        att = _lp_proj_ratio_cached(ratio.tobytes(), m, d, k, n_iter, metric, blend_alpha)

        if len(att) < p_fit * (cascade_factor ** (n_levels - 1)) + p_fit + 1:
            continue  # недостаточно данных для каскада

        _d_proj: int | None = None
        if d_proj_offset is not None:
            _dp = d + d_proj_offset
            if 2 <= _dp < p_fit:
                _d_proj = _dp

        try:
            res = run_forecast(
                att, ratio, logtrend, a_lt, b_lt,
                p_fit, n_levels, xi,
                horizon, bars,
                use_lp_corr, m, d, k,
                origin,
                metric, k_geo, geo_pool_max, blend_alpha, cascade_factor, fixed_manifold,
                smap_mode, smap_theta,
                _d_proj, trend_is_log, pip_weighted,
            )
        except Exception:
            continue

        # Val-прогон: origin сдвинут назад на val_horizon, сравниваем с фактическими ценами
        val_mape = float("nan")
        if val_horizon > 0:
            val_origin = origin - val_horizon
            min_hist = p_fit * (cascade_factor ** (n_levels - 1)) + p_fit
            if val_origin >= min_hist:
                try:
                    val_res = run_forecast(
                        att, ratio, logtrend, a_lt, b_lt,
                        p_fit, n_levels, xi,
                        val_horizon, bars,
                        use_lp_corr, m, d, k,
                        val_origin,
                        metric, k_geo, geo_pool_max, blend_alpha, cascade_factor, fixed_manifold,
                        smap_mode, smap_theta,
                        _d_proj, trend_is_log, pip_weighted,
                    )
                    fc_val  = val_res["fc_price"]
                    act_val = close[val_origin + 1 : val_origin + val_horizon + 1]
                    if len(fc_val) == len(act_val) and len(act_val) > 0:
                        val_mape = float(np.mean(
                            np.abs(fc_val - act_val) / np.maximum(np.abs(act_val), 1e-10)
                        ))
                except Exception:
                    pass

        def _fmt(v): return round(float(v), 3) if not (np.isnan(v) or np.isinf(v)) else float("nan")

        lb_att   = res["lb_cascade_att"]                 # list[float]: LB(L) по уровням
        lb_ratio = res["lb_cascade_ratio"]               # list[float]: LB(R) по уровням
        lb_last  = lb_att[-1] if lb_att else np.nan      # гейт — последний уровень att
        threshold   = abs(xy_x**2 * lb_last + xy_x * xy_y + xy_y)
        gate_passed = not np.isnan(lb_last) and d >= threshold
        gate_margin = d - threshold if not np.isnan(lb_last) else -np.inf

        # Δ = Σ |LB_Lk_R − LB_Lk_L|  по уровням с валидными парами
        diffs = [abs(float(r) - float(l))
                 for r, l in zip([_fmt(v) for v in lb_ratio], [_fmt(v) for v in lb_att])
                 if not (np.isnan(r) or np.isnan(l))]
        delta_lb = round(sum(diffs), 3) if diffs else float("nan")

        results.append({
            "d":                 d,
            "n_iter":            n_iter,
            "m":                 m,
            "val_mape":          val_mape,
            "k":                 k,
            "p_fit":             p_fit,
            "xi_lwr":            xi,
            "lb_cascade_att":    [_fmt(v) for v in lb_att],
            "lb_cascade_ratio":  [_fmt(v) for v in lb_ratio],
            "lb_last":           _fmt(lb_last),
            "delta_lb":          delta_lb,
            "gate_margin":       round(gate_margin, 3) if not np.isinf(gate_margin) else float("-inf"),
            "gate_passed":       gate_passed,
            "fc_price":          res["fc_price"],
            "fc_idx":            res["fc_idx"],
        })

    return results

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="app5 · Cascade Ratio Forecast", layout="wide",
                   initial_sidebar_state="expanded")

with st.sidebar:
    st.title("app5 · Cascade Ratio")

    ticker   = st.text_input("Тикер", "SBER", key="ticker").upper().strip()
    interval = st.selectbox("Интервал", list(INTERVALS.keys()),
                            index=list(INTERVALS.keys()).index("1d"))
    if st.button("Обновить данные"):
        _load_candles.clear()
        _fetch_and_save(ticker, interval)
        st.rerun()

    st.divider()
    st.subheader("Нормализация")
    trend_mode = st.radio(
        "Тренд",
        ["logtrend (log OLS)", "lintrend (linear OLS)"],
        horizontal=True, label_visibility="collapsed",
        help="logtrend: ratio = close / exp(a+b·t) — стандарт; "
             "lintrend: ratio = close / (a+b·t) — линейный тренд в ценах",
    )
    trend_is_log = (trend_mode == "logtrend (log OLS)")

    st.divider()
    st.subheader("Параметры  m = p = x·d + y")
    xy_x = st.slider("x", 1, 10, 2, key="xy_x")
    xy_y = st.slider("y", 0, 20, 3, key="xy_y")
    st.caption(f"d=3: m=p={xy_x*3+xy_y}  |  d=11: m=p={xy_x*11+xy_y}")

    st.markdown("**n_iter по диапазону d**")
    with st.expander("Конфигурация n_iter", expanded=True):
        n_iter_cfg = []
        for ni in [1, 2, 3]:
            cols = st.columns([1, 3])
            enabled = cols[0].checkbox(f"n={ni}", value=(ni == 1), key=f"ni_{ni}")
            if enabled:
                dr = cols[1].slider(f"d range n={ni}", 1, 30, (5, 25), key=f"dr_{ni}")
                n_iter_cfg.append((ni, dr[0], dr[1]))

    st.divider()
    st.subheader("Каскад")
    cascade_mode = st.radio("Режим", ["Октавный ×2", "Треугольный ×3"],
                            horizontal=True, label_visibility="collapsed")
    cascade_factor = 2 if cascade_mode == "Октавный ×2" else 3
    n_levels = st.slider("Уровней каскада", 1, 6, 4)
    xi_add   = st.slider("ξ добавка", 0, 20, 1,
                         help="ξ_lwr = 3·(p_fit+1) + добавка")
    bars     = st.slider("Баров в библиотеке", 100, 5000, 3000, 100)
    fixed_manifold = st.checkbox("Фиксированное многообразие",
                                 help="Каскад один раз; регрессия итерирует в найденном пуле")
    smap_mode  = st.checkbox("S-map (весь пул, exp(−θ·d/d̄))", value=True,
                             help="Последний уровень: весь пул вместо top-ξ; ξ не ограничивает регрессию")
    smap_theta = st.slider("θ (S-map)", 0.0, 20.0, 18.0, 0.5,
                           disabled=not smap_mode, key="smap_theta")

    pip_enabled = st.checkbox("PiP-проекция (PCA перед регрессией)", value=True,
                              help="Prediction in Projection: LWR/S-map строится в d_proj-мерном "
                                   "локальном подпространстве соседей. d_proj = d + offset, "
                                   "где d — текущее значение sweep.")
    d_proj_offset: int | None = None
    pip_weighted = False
    if pip_enabled:
        d_proj_offset = st.slider("d_proj offset", -4, 10, 0,
                                  help="d_proj = d + offset. При offset=0: d_proj=d (размерность LP-многообразия). "
                                       "Пропускается если d_proj < 2 или d_proj ≥ p_fit.")
        pip_weighted = st.checkbox("wSVD (взвешенный SVD)", value=False,
                                   help="Подпространство строится на взвешенной ковариации: "
                                        "центр = Σwᵢxᵢ/Σwᵢ, SVD(√wᵢ·(xᵢ−center)). "
                                        "Без галочки — обычный SVD по среднему всех соседей.")
        _ex_d, _ex_pfit = 3, xy_x * 3 + xy_y
        _ex_dp = _ex_d + d_proj_offset
        _pip_ok = 2 <= _ex_dp < _ex_pfit
        st.caption(f"d=3: p_fit={_ex_pfit}, d_proj={_ex_d}+({d_proj_offset:+d})={_ex_dp} "
                   f"→ {'PiP активен' if _pip_ok else 'skip (вне диапазона)'}")

    _ex_lvls = _cascade_levels(xy_x * 3 + xy_y, n_levels, cascade_factor)
    st.caption("d=3: " + " → ".join(str(l) for l in _ex_lvls))

    st.divider()
    st.subheader("Прогноз")
    horizon       = st.slider("Горизонт (баров)", 1, 200, 80)
    origin_offset = st.slider("Origin (баров от конца)", 0, 500, 0,
                              help="0 = последний бар; N = прогноз от N баров назад")
    top_n         = st.slider("Top N прогнозов", 1, 50, 10)
    top_criterion = st.selectbox("Критерий топа",
                                 ["gate margin", "delta LB", "val_mape"],
                                 help="gate margin — ближе к 0 среди прошедших гейт; "
                                      "delta LB — наименьшее Σ|LB_R−LB_L|; "
                                      "val_mape — наименьшая ошибка на val-прогоне (требует val горизонт > 0)")
    val_horizon   = st.slider("Val горизонт (LOO)", 0, 100, 0,
                              help="0 = отключено. N > 0: для каждого d прогноз от (origin − N) на N шагов вперёд, "
                                   "сравнивается с фактическими ценами → val_mape для взвешивания ансамбля.")
    show_avg      = top_n > 1 and st.checkbox("Показать среднее", value=True)
    ensemble_weight = "равные"
    if show_avg:
        ensemble_weight = st.selectbox("Взвешивание ансамбля",
                                       ["равные", "геометрическое", "margin-баланс", "1/val_mape", "softmax"],
                                       index=2,
                                       help="равные — арифметическое среднее; "
                                            "геометрическое — exp(mean(log(цены))); "
                                            "margin-баланс — веса групп ±margin так, чтобы Σw·margin=0; "
                                            "1/val_mape и softmax — требуют val горизонт > 0")

    st.divider()
    use_lp_corr = st.checkbox("LP-коррекция", value=False,
                              help="Проецировать предсказание обратно на аттрактор")

    st.divider()
    st.caption(f"Гейт: d ≥ x²·LB + (x·y+y) − 1 = **{xy_x**2}·LB + {xy_x*xy_y + xy_y - 1}**")

    st.divider()
    st.subheader("Метрика расстояния")
    dist_metric = st.selectbox(
        "Метрика",
        ["euclidean", "cosine", "blend", "amp_cos", "mahalanobis", "geodesic"],
        index=3,
        help=(
            "euclidean — L2 (стандарт)\n"
            "cosine — угол, игнорирует амплитуду\n"
            "blend — α·L2_norm + (1−α)·cosine_norm\n"
            "amp_cos — α·|log(‖x‖/‖y‖)|_norm + (1−α)·cosine_norm\n"
            "mahalanobis — L2 в декоррелированном пространстве\n"
            "geodesic — кратчайший путь по k-NN графу; LP-фильтр всегда euclidean"
        ),
    )
    k_geo = 10
    geo_pool_max = 300
    blend_alpha = 0.5
    if dist_metric == "blend":
        blend_alpha = st.slider("α (L2 ↔ cosine)", 0.0, 1.0, 0.5, 0.05,
                                help="0 = чистый cosine, 1 = чистый L2")
        st.caption(f"d = {blend_alpha}·L2_norm + {round(1-blend_alpha, 2)}·cosine_norm")
    elif dist_metric == "amp_cos":
        blend_alpha = st.slider("α (амплитуда ↔ форма)", 0.0, 1.0, 0.5, 0.05,
                                help="0 = только форма (cosine), 1 = только амплитуда (log-норм)")
        st.caption(f"d = {blend_alpha}·|log‖x‖/‖y‖|_norm + {round(1-blend_alpha, 2)}·cosine_norm")
    if dist_metric == "geodesic":
        k_geo        = st.slider("k графа", 5, 30, 10,
                                 help="Число рёбер k-NN графа на узел")
        geo_pool_max = st.slider("Макс. пул (geodesic)", 50, 500, 150, 10,
                                 help="Если пул > значения — fallback на euclidean для этого уровня")
        st.caption("LP-фильтр использует euclidean (geodesic слишком медленный в пошаговом цикле).")

    run_btn = st.button("▶  Запустить", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Локальная размерность")
    dim_p_range  = st.slider("p_search диапазон", 3, 300, (3, 100), key="dim_p_range")
    dim_p_step   = st.slider("шаг", 1, 30, 1, key="dim_p_step")
    _dim_p_vals  = list(range(dim_p_range[0], dim_p_range[1] + 1, dim_p_step))
    st.caption(f"{len(_dim_p_vals)} точек · K = 3·(p+1)+{xi_add}")
    dim_pca_thr1 = st.slider("PCA порог 1", 0.70, 0.99, 0.95, 0.01, key="dim_pca_thr1")
    dim_pca_thr2 = st.slider("PCA порог 2", 0.70, 0.99, 0.97, 0.01, key="dim_pca_thr2")
    dim_min_lag  = st.slider("min_lag (баров)", 0, 200, 0, key="dim_min_lag",
                             help="Дополнительный отступ от origin (0 = стандартное причинное условие)")
    dim_bars     = st.slider("Баров библиотеки", 100, 5000, 3000, 100, key="dim_bars")
    dim_btn = st.button("📐 Размерность", use_container_width=True)

# ── Загрузка данных ───────────────────────────────────────────────────────────

data = _load_candles(ticker, interval)
if not data:
    st.info("Нет данных. Нажмите «Обновить данные» в сайдбаре.")
    st.stop()

times, close, open_, high, low = _to_arrays(data)
if trend_is_log:
    logtrend, a_lt, b_lt = _logtrend_causal(close)
else:
    logtrend, a_lt, b_lt = _lintrend_causal(close)
ratio  = close / np.maximum(logtrend, 1e-10)
n      = len(close)
origin = max(0, min(n - 1, n - 1 - origin_offset))

# ── Прогон ────────────────────────────────────────────────────────────────────

_same_run = (
    "sweep_results" in st.session_state
    and st.session_state.get("last_ticker")  == ticker
    and st.session_state.get("sweep_origin") == origin
    and st.session_state.get("last_metric")      == dist_metric
    and st.session_state.get("last_blend_alpha") == blend_alpha
    and st.session_state.get("last_xy_x") == xy_x
    and st.session_state.get("last_xy_y") == xy_y
    and st.session_state.get("last_cascade_factor")  == cascade_factor
    and st.session_state.get("last_fixed_manifold")  == fixed_manifold
    and st.session_state.get("last_smap_mode")       == smap_mode
    and st.session_state.get("last_smap_theta")      == smap_theta
    and st.session_state.get("last_d_proj_offset")   == d_proj_offset
    and st.session_state.get("last_val_horizon")     == val_horizon
    and st.session_state.get("last_trend_mode")      == trend_mode
    and st.session_state.get("last_pip_weighted")    == pip_weighted
)

if run_btn:
    d_values = sorted({d for (_, d_min, d_max) in n_iter_cfg for d in range(d_min, d_max + 1)})
    prog_bar  = st.progress(0.0)
    prog_text = st.empty()

    def _progress(step, total, d, n_iter):
        prog_bar.progress(step / total if total > 0 else 0.0)
        prog_text.caption(f"Шаг {step+1}/{total} — d={d}, n_iter={n_iter}")

    with st.spinner("Вычисление..."):
        results = run_sweep(
            ratio, close, logtrend, a_lt, b_lt,
            d_values, n_iter_cfg,
            xy_x, xy_y,
            xi_add, n_levels, bars,
            horizon,
            use_lp_corr, origin, _progress,
            dist_metric, k_geo, geo_pool_max,
            blend_alpha, cascade_factor, fixed_manifold,
            smap_mode, smap_theta,
            d_proj_offset,
            val_horizon,
            trend_is_log,
            pip_weighted,
        )

    prog_bar.empty(); prog_text.empty()

    st.session_state["sweep_results"] = results
    st.session_state["last_ticker"]   = ticker
    st.session_state["last_metric"]      = dist_metric
    st.session_state["last_blend_alpha"] = blend_alpha
    st.session_state["last_xy_x"] = xy_x
    st.session_state["last_xy_y"] = xy_y
    st.session_state["last_cascade_factor"]  = cascade_factor
    st.session_state["last_fixed_manifold"]  = fixed_manifold
    st.session_state["last_smap_mode"]       = smap_mode
    st.session_state["last_smap_theta"]      = smap_theta
    st.session_state["last_d_proj_offset"]   = d_proj_offset
    st.session_state["last_val_horizon"]     = val_horizon
    st.session_state["last_trend_mode"]      = trend_mode
    st.session_state["last_pip_weighted"]    = pip_weighted

    st.session_state["sweep_n"]       = n
    st.session_state["sweep_times"]   = times
    st.session_state["sweep_origin"]  = origin

elif _same_run:
    results = st.session_state["sweep_results"]
else:
    results = []

# ── Сортировка по выбранному критерию ────────────────────────────────────────
if results:
    if top_criterion == "val_mape":
        results = sorted(results, key=lambda r: (
            float("inf") if np.isnan(r.get("val_mape", float("nan"))) else r["val_mape"]
        ))
    elif top_criterion == "delta LB":
        results = sorted(results, key=lambda r: (
            float("inf") if (isinstance(r["delta_lb"], float) and np.isnan(r["delta_lb"])) else r["delta_lb"]
        ))
    else:  # gate margin — мягкий: ближайшие к 0 по |margin|
        results = sorted(results, key=lambda r:
            float("inf") if r["gate_margin"] == float("-inf") else abs(r["gate_margin"])
        )

# ── Локальная размерность ─────────────────────────────────────────────────────

if dim_btn:
    with st.spinner(f"Sweep размерности ({len(_dim_p_vals)} точек)..."):
        _dim_rows = sweep_local_dim(
            ratio, origin,
            p_values=_dim_p_vals,
            xi_add=xi_add,
            bars=dim_bars,
            min_lag=dim_min_lag,
            pca_thresholds=(dim_pca_thr1, dim_pca_thr2),
            metric=dist_metric,
            k_geo=k_geo,
            geo_pool_max=geo_pool_max,
            blend_alpha=blend_alpha,
        )
    st.session_state["dim_result"] = _dim_rows

if "dim_result" in st.session_state:
    import pandas as pd
    _rows  = st.session_state["dim_result"]
    _col1  = f"PCA_{int(dim_pca_thr1*100)}"
    _col2  = f"PCA_{int(dim_pca_thr2*100)}"
    st.subheader("Локальная размерность  (ratio, без LP)")
    st.caption(f"metric={dist_metric}  ·  min_lag={dim_min_lag}  ·  K = 3·(p+1)+{xi_add}")

    _df_dim = pd.DataFrame([
        {"p_search": r["p_search"], "K": r["K"],
         _col1: round(r[_col1], 1) if np.isfinite(r[_col1]) else None,
         _col2: round(r[_col2], 1) if np.isfinite(r[_col2]) else None}
        for r in _rows
    ])
    st.dataframe(_df_dim, use_container_width=True, hide_index=True, height=220)

    def _fmt_d(v): return f"{v:.2f}" if (v is not None and np.isfinite(float(v))) else "—"
    _ca, _cb, _cc = st.columns(3)
    _ca.markdown("**mean по sweep**")
    for _col, _thr, _cx in ((_col1, dim_pca_thr1, _cb), (_col2, dim_pca_thr2, _cc)):
        _vals = [r[_col] for r in _rows if np.isfinite(r[_col])]
        _cx.metric(f"PCA ≥{int(_thr*100)}%", _fmt_d(sum(_vals) / len(_vals)) if _vals else "—")

if not results:
    # Просто показываем свечной график без прогноза
    fig = go.Figure(go.Candlestick(
        x=times[-200:], open=open_[-200:], high=high[-200:],
        low=low[-200:],  close=close[-200:], name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    fig.update_layout(height=550, xaxis_rangeslider_visible=False,
                      title=f"{ticker} {interval}")
    st.plotly_chart(fig, use_container_width=True)
    st.stop()

# ── Таблица результатов ───────────────────────────────────────────────────────

st.subheader("Результаты перебора d")

import pandas as pd
n_lv = max((len(r["lb_cascade_att"]) for r in results), default=0)
df_rows = []
for r in results:
    row = {
        "d":      r["d"],
        "n_iter": r["n_iter"],
        "m":      r["m"],
        "k":      r["k"],
        "p_fit":  r["p_fit"],
        "ξ":      r["xi_lwr"],
    }
    for lv in range(n_lv):
        lb_r = r["lb_cascade_ratio"][lv] if lv < len(r["lb_cascade_ratio"]) else float("nan")
        lb_l = r["lb_cascade_att"][lv]   if lv < len(r["lb_cascade_att"])   else float("nan")
        row[f"LB L{lv+1} R"] = lb_r
        row[f"LB L{lv+1} L"] = lb_l
    row["LB last"] = r.get("lb_last", float("nan"))
    row["Δ LB"] = r["delta_lb"]
    row["margin"] = r["gate_margin"]
    row["gate ✓"] = "✓" if r["gate_passed"] else "✗"
    vm = r.get("val_mape", float("nan"))
    row["val_mape"] = round(vm, 4) if not np.isnan(vm) else float("nan")
    df_rows.append(row)
df = pd.DataFrame(df_rows)

def _style_row(row):
    color = "#1a3a1a" if row["gate ✓"] == "✓" else "#3a1a1a"
    return [f"background-color: {color}"] * len(row)

st.dataframe(df.style.apply(_style_row, axis=1), use_container_width=True, height=260)

# ── График ─────────────────────────────────────────────────────────────────────

top = results[:top_n]

# Суммарный margin топ прогнозов до балансировки
_finite_margins = [r["gate_margin"] for r in top if np.isfinite(r["gate_margin"])]
if _finite_margins:
    _total_margin = sum(_finite_margins)
    _n_pass = sum(1 for r in top if r["gate_passed"])
    _bias = "перевес ✓" if _total_margin > 0.1 else ("перевес ✗" if _total_margin < -0.1 else "≈баланс")
    st.caption(
        f"Σ margin top-{len(top)} (до балансировки): **{_total_margin:+.3f}**  ·  "
        f"✓ {_n_pass} / ✗ {len(top) - _n_pass}  ·  {_bias}"
    )

COLORS_FC = ["#00bcd4", "#ff9800", "#9c27b0", "#4caf50", "#f44336",
             "#2196f3", "#ffeb3b", "#e91e63", "#00e676", "#ff5722"]

display_n      = st.session_state.get("sweep_n", n)
display_times  = st.session_state.get("sweep_times", times)
sweep_origin   = st.session_state.get("sweep_origin", display_n - 1)

# Окно отображения: 300 баров до origin + всё после origin до конца серии
show_from = max(0, sweep_origin - 299)
# Временны́е метки за пределами серии (для прогноза в будущее)
try:
    from datetime import timedelta as _td
    _base = pd.Timestamp(display_times[-1])
    _PERIOD = {
        "1m":  _td(minutes=1),  "10m": _td(minutes=10),
        "1h":  _td(hours=1),    "1w":  _td(weeks=1),
        "1mo": _td(days=30),
    }
    if interval == "1d":
        future_times = [str(_base + pd.offsets.BDay(h)) for h in range(1, horizon + 1)]
    elif interval in _PERIOD:
        _p = _PERIOD[interval]
        future_times = [str(_base + _p * h) for h in range(1, horizon + 1)]
    else:
        _dt = pd.Timestamp(display_times[-1]) - pd.Timestamp(display_times[-2])
        future_times = [str(_base + _dt * h) for h in range(1, horizon + 1)]
except Exception:
    future_times = [f"+{h}" for h in range(1, horizon + 1)]

def time_at(idx: int) -> str:
    """Временна́я метка для абсолютного индекса бара."""
    if idx < display_n:
        return display_times[idx]
    return future_times[min(idx - display_n, len(future_times) - 1)]

fig = go.Figure()

# Свечной график: от show_from до конца серии
fig.add_trace(go.Candlestick(
    x=display_times[show_from:],
    open=open_[show_from:], high=high[show_from:],
    low=low[show_from:],    close=close[show_from:],
    name="OHLC",
    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
))

# Вертикальная линия в точке origin
fig.add_vline(
    x=display_times[sweep_origin],
    line_width=1.5, line_dash="dash", line_color="#ffd600",
    annotation_text=f"origin −{origin_offset}б" if origin_offset > 0 else "origin",
    annotation_position="top left",
)

# Фактические цены после origin (для сравнения, если origin < конца серии)
if sweep_origin < display_n - 1:
    actual_end = min(display_n, sweep_origin + horizon + 1)
    fig.add_trace(go.Scatter(
        x=display_times[sweep_origin:actual_end],
        y=close[sweep_origin:actual_end],
        mode="lines", name="actual",
        line=dict(color="#ffffff", width=1.5, dash="dot"),
    ))

for rank, r in enumerate(top):
    if show_avg:
        continue
    color_fc = COLORS_FC[rank % len(COLORS_FC)]
    lb_last = r["lb_cascade_att"][-1] if r["lb_cascade_att"] else float("nan")
    lbl = f"d={r['d']} n={r['n_iter']} p={r['p_fit']} LB={lb_last:.2f}"

    # Прогноз: от close[origin] через fc_idx
    fc_x = [display_times[sweep_origin]] + [time_at(int(i)) for i in r["fc_idx"]]
    fc_y = np.concatenate([[close[sweep_origin]], r["fc_price"]])
    fig.add_trace(go.Scatter(
        x=fc_x, y=fc_y,
        mode="lines+markers", name=f"fc {lbl}",
        line=dict(color=color_fc, width=2),
        marker=dict(size=4),
    ))

# Среднее прогнозов
if show_avg and len(top) > 1:
    fc_stack = np.array([r["fc_price"] for r in top])
    vm_list  = [r.get("val_mape", float("nan")) for r in top]
    vm_valid = [v for v in vm_list if not np.isnan(v)]

    if ensemble_weight == "геометрическое":
        avg_fc = np.exp(np.mean(np.log(np.maximum(fc_stack, 1e-10)), axis=0))
        weight_label = "геом."
    elif ensemble_weight == "margin-баланс":
        margins = np.array([r.get("gate_margin", float("nan")) for r in top])
        if np.isfinite(margins).all():
            pos_mask = margins >= 0
            neg_mask = margins < 0
            if pos_mask.any() and neg_mask.any():
                mean_pos = margins[pos_mask].mean()
                mean_neg = abs(margins[neg_mask].mean())
                denom = mean_pos + mean_neg
                w = np.zeros(len(margins))
                w[pos_mask] = (mean_neg / denom) / pos_mask.sum()
                w[neg_mask] = (mean_pos / denom) / neg_mask.sum()
            else:
                w = np.ones(len(margins)) / len(margins)
            avg_fc = (fc_stack * w[:, None]).sum(axis=0)
            weight_label = "margin-баланс"
        else:
            avg_fc = fc_stack.mean(axis=0)
            weight_label = "равные (нет margin)"
    elif ensemble_weight in ("1/val_mape", "softmax") and len(vm_valid) == len(top):
        vm = np.array(vm_list)
        if ensemble_weight == "1/val_mape":
            w = 1.0 / np.maximum(vm, 1e-10)
        else:  # softmax
            vm_mean = max(float(vm.mean()), 1e-10)
            w = np.exp(-vm / vm_mean)
        w /= w.sum()
        avg_fc = (fc_stack * w[:, None]).sum(axis=0)
        weight_label = f"взвеш. {ensemble_weight}"
    else:
        avg_fc = fc_stack.mean(axis=0)
        weight_label = "равные"

    fc_x_avg = [display_times[sweep_origin]] + [time_at(int(i)) for i in top[0]["fc_idx"]]
    fig.add_trace(go.Scatter(
        x=fc_x_avg, y=np.concatenate([[close[sweep_origin]], avg_fc]),
        mode="lines", name=f"fc среднее ({weight_label})",
        line=dict(color="#bdbdbd", width=2.5),
    ))

fig.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    xaxis_type="date",
    title=f"{ticker} {interval}  ·  origin −{origin_offset}б  ·  top-{top_n} по |margin| (гейт d≥|{xy_x**2}·LB+{xy_x*xy_y+xy_y}|)",
    template="plotly_dark",
    legend=dict(orientation="h", y=-0.18),
)

# Расширяем x-ось если прогноз уходит за пределы серии
if top and int(top[0]["fc_idx"][-1]) >= display_n:
    fig.update_layout(xaxis_range=[display_times[show_from],
                                   time_at(int(top[0]["fc_idx"][-1]))])

st.plotly_chart(fig, use_container_width=True)

# ── Детали top прогнозов ──────────────────────────────────────────────────────
if top:
    st.subheader("Параметры отображённых прогнозов")
    for rank, r in enumerate(top):
        gate_str = "✓ PASS" if r["gate_passed"] else "✗ FAIL"
        lb_last = r["lb_cascade_att"][-1] if r["lb_cascade_att"] else float("nan")
        lb_l_str = "  |  " + "  ".join(
            f"L{i+1}R={r['lb_cascade_ratio'][i]:.2f}/L={r['lb_cascade_att'][i]:.2f}"
            for i in range(len(r["lb_cascade_att"]))
            if not (np.isnan(r["lb_cascade_att"][i]) and np.isnan(r["lb_cascade_ratio"][i]))
        )
        st.caption(
            f"#{rank+1}  d={r['d']}  n_iter={r['n_iter']}  "
            f"m={r['m']}  k={r['k']}  p={r['p_fit']}  ξ={r['xi_lwr']}  "
            f"LB(L)last={lb_last:.2f}  margin={r['gate_margin']}  "
            f"{gate_str}{lb_l_str}"
        )
