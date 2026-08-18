"""
app7-simplex-ensemble: чистый ансамбль Simplex projection по нескольким origin.

Урезанная версия app6-multiorigin.py. Оставлено только то, что нужно для
ансамбля траекторий по origin:

  - причинный logtrend → LP-фильтр (att) → каскадный поиск соседей →
    Simplex projection (Sugihara & May, 1990) на каждом шаге h.
  - усреднение по d (диапазон определяется PCA-sweep Этапа 1, один раз
    по главному origin).
  - ансамбль по W origin (origin, origin−1, …, origin−W+1), каждый со
    своей причинной обрезкой ratio — без утечки между origin.
  - агрегация (mean/p25/p75) по W траекторий — РАВНЫЕ веса, без
    затухания по давности.

Убрано относительно app6-multiorigin (по запросу пользователя):
  - S-map как метод аппроксимации пула (остался только Simplex projection).
  - Предобработка входной цены (open+close)/2 — всегда close.
  - Взвешивание по давности origin (τ-затухание) — ансамбль усредняется
    поровну.
  - Векторы направления (OLS-прямая, «ускорение» между крайними весами).
  - Самопроверка на перекрытии (overlap walk-forward внутри окна).

Синхронизация K с simplex (см. research/cascade_algorithm.md — ЧИТАТЬ
ПЕРЕД ЛЮБОЙ РАБОТОЙ С КАСКАДОМ, никакой самодеятельности):

  Каскад использует ОДНО число xi_lwr на всех промежуточных уровнях —
  доказательство в cascade_algorithm.md показывает, что пул следующего
  уровня всегда больше xi (pool_min = xi + p_next > xi), НЕЗАВИСИМО от
  конкретного значения xi. Раньше в этом прототипе xi_lwr = K =
  3·(p_fit+1)+ξ — эвристика для LWR/S-map регрессии (нужен запас точек
  для устойчивой подгонки коэффициентов). Simplex projection коэффициентов
  не подгоняет — на финальном шаге ему нужно ровно d_proj+1 соседей
  (канонически E+1, Sugihara & May, 1990). Искать в каскаде больше
  соседей, чем в итоге использует simplex, смысла нет — поэтому:

    K_cascade = d_proj + 1   (используется в cascade_search на всех
                               промежуточных уровнях xi_lwr)

  Это отдельная переменная от K_filter = 3·(p_fit+1)+ξ, который остаётся
  внутренним параметром LP-фильтра (локальный PCA/SVD для att) и
  LP-коррекции траектории — там нужен запас точек для устойчивой
  локальной подгонки подпространства, это не связано с выбором
  аппроксиматора на финальном шаге прогноза.

Ядро алгоритма (метрика amp_cos, logtrend, LP-фильтр, каскад,
Simplex projection, LP-коррекция) — без изменений от app6-etalon /
app6-multiorigin, кроме описанной синхронизации K_cascade.

Run (из prototype/): streamlit run forcaster/ui/app7-simplex-ensemble.py
"""
from __future__ import annotations

import json
import sys
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
# Каскад (октавный ×2) — без изменений от app6-etalon.
# K (xi_lwr) теперь передаётся снаружи как K_cascade = d_proj + 1 — см. докстринг
# модуля и research/cascade_algorithm.md.
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
    K_cascade: int,
    t_predict: int,
    blend_alpha: float = 0.5,
    p_cascade_max: int = 1500,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Строгий октавный каскад ×2 (cascade_algorithm.md). Без изменений от app6-etalon,
    кроме имени параметра K → K_cascade (см. докстринг модуля)."""
    n      = len(att)
    levels = _cascade_levels(p_fit, n_levels, p_cascade_max)
    p_top  = levels[0]

    max_s = min(n - p_top, t_predict - p_top - 1)
    if max_s < 0:
        return None, None

    pool  = np.arange(0, max_s + 1)
    if len(pool) < K_cascade:
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

        n_sel      = min(K_cascade, len(pool))
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
# Simplex projection (Sugihara & May, 1990) — без изменений от app6-multiorigin.
# ═══════════════════════════════════════════════════════════════════════════════

def _simplex_step(
    X_pool: np.ndarray, y_pool: np.ndarray, vec_fit: np.ndarray,
    blend_alpha: float, n_neighbors: int,
) -> float:
    """
    Simplex projection: n_neighbors (канонически E+1 = d_proj+1) ближайших
    соседей из отобранного каскадом пула, взвешенное среднее их известных y
    (без подгонки коэффициентов, без PCA-проекции — работает прямо в
    пространстве задержек):
        w_i = exp(−d_i / d_min) / Σ exp(−d_j / d_min),  d_min = min_i d_i
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
# LP-коррекция траектории — без изменений от app6-etalon. Использует K_filter
# (запас точек для устойчивого локального PCA), НЕ K_cascade.
# ═══════════════════════════════════════════════════════════════════════════════

def _lp_corr(v_m: np.ndarray, X_lib: np.ndarray, K_filter: int, d: int,
             blend_alpha: float) -> float:
    k_eff = min(K_filter, len(X_lib))
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
# Прогноз: один d, полный горизонт. Каскад строго воронкой с K_cascade = d+1,
# финальный шаг — только Simplex projection.
# ═══════════════════════════════════════════════════════════════════════════════

def run_forecast(
    att: np.ndarray,
    p_fit: int, n_levels: int,
    horizon: int, origin: int,
    blend_alpha: float, d_proj: int,
    p_cascade_max: int = 1500,
    use_lp_corr: bool = False,
    K_filter: int = 0,
) -> np.ndarray | None:
    """Возвращает fc_preds[horizon] — сырые att-предсказания, или None при ошибке.

    K_cascade = d_proj + 1 — синхронизировано с simplex (см. докстринг модуля):
    ровно столько соседей нужно на каждом промежуточном уровне каскада и на
    финальном шаге Simplex projection, искать больше незачем.
    """
    K_cascade = d_proj + 1
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
            att, ctx, p_fit, n_levels, K_cascade, t, blend_alpha, p_cascade_max,
        )
        if X_nn is None or len(X_nn) < _min_pool:
            return None
        pred = _simplex_step(X_nn, y_nn, ctx[-p_fit:], blend_alpha, K_cascade)
        if use_lp_corr and X_lib_lp is not None:
            v_m  = np.array(fc_buf[-(p_fit - 1):] + [pred], dtype=np.float64)
            pred = _lp_corr(v_m, X_lib_lp, K_filter, d_proj, blend_alpha)
        fc_buf.append(pred)

    return np.array(fc_buf[n_eff:])

# ═══════════════════════════════════════════════════════════════════════════════
# Этап 2: sweep по d.
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage2(
    ratio: np.ndarray,
    d_values: list[int],
    xy_x: int, xy_y: int, xi_add: int,
    n_levels: int, horizon: int, origin: int,
    progress_cb=None,
    blend_alpha: float = 0.5,
    p_cascade_max: int = 1500,
    use_lp_corr: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Возвращает (results, failed). results содержит fc_preds (сырые att)."""
    results = []
    failed  = []
    for step_i, d in enumerate(d_values):
        if progress_cb:
            progress_cb(step_i, len(d_values), d)

        p_fit    = xy_x * d + xy_y
        K_filter = 3 * (p_fit + 1) + xi_add   # только для LP-фильтра и LP-коррекции
        d_proj   = d

        if p_fit < 2:
            failed.append({"d": d, "reason": "p_fit < 2"})
            continue

        att = _lp_proj_ratio_cached(ratio.tobytes(), p_fit, d, K_filter, 1, blend_alpha)

        eff_levels = _cascade_levels(p_fit, n_levels, p_cascade_max)
        min_len = eff_levels[0] + p_fit + 1
        if len(att) < min_len:
            failed.append({"d": d, "reason": f"ряд мал для каскада (нужно {min_len})"})
            continue

        fc_preds = run_forecast(
            att, p_fit, n_levels, horizon, origin,
            blend_alpha, d_proj, p_cascade_max,
            use_lp_corr, K_filter,
        )
        if fc_preds is None:
            failed.append({"d": d, "reason": f"pool < d+2={d+2}"})
            continue

        results.append({
            "d":        d,
            "m":        p_fit,
            "K_filter": K_filter,
            "fc_preds": fc_preds,
        })

    return results, failed

# ═══════════════════════════════════════════════════════════════════════════════
# Ансамбль по нескольким origin
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


def _aggregate(per_origin: list[dict]) -> dict:
    """Равновзвешенные mean/std/p25/p75 по origin для каждого h, на основе
    уже посчитанных rel_i(h) в per_origin."""
    rel_stack = np.array([r["rel"] for r in per_origin])  # (n_origins, horizon)
    mean_rel = rel_stack.mean(axis=0)
    std_rel  = rel_stack.std(axis=0)
    p25 = np.quantile(rel_stack, 0.25, axis=0)
    p75 = np.quantile(rel_stack, 0.75, axis=0)
    return {"mean_rel": mean_rel, "std_rel": std_rel, "p25": p25, "p75": p75}


def run_multi_origin(
    ratio_full: np.ndarray,
    a_arr: np.ndarray, b_arr: np.ndarray, price_input: np.ndarray,
    origin: int, window: int, bars: int,
    xy_x: int, xy_y: int, xi_add: int, n_levels: int, p_cascade_max: int,
    pca_p_range: tuple, pca_thr1: float, pca_thr2: float, blend_alpha: float,
    horizon: int, use_lp_corr: bool,
    progress_cb=None,
) -> dict:
    """
    Прогоняет пайплайн (LP-фильтр → каскад → Simplex projection) для
    origin, origin−1, …, origin−(window−1). Каждый origin_i получает СВОЮ
    причинную обрезку ratio (ratio_full[:origin_i+1], затем bars) — никакой
    утечки между origin.

    Этап 1 (PCA-sweep → d_min/d_max) считается ОДИН РАЗ по главному origin и
    переиспользуется для всех origin_i в окне: соседние origin почти полностью
    делят одну и ту же историю, локальная размерность на масштабе window баров
    не должна ощутимо отличаться.

    Возвращает dict с per_origin (список по origin) и равновзвешенными
    агрегатами mean_rel/std_rel/p25/p75 по относительным траекториям
    rel[h] = price[h]/origin_price_i − 1.
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
            None, blend_alpha, p_cascade_max, use_lp_corr,
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

    return {
        "per_origin": per_origin,
        "skipped": skipped,
        "d_min": d_min, "d_max": d_max,
        "horizon": horizon,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="app7-simplex-ensemble · Ансамбль Simplex по origin", layout="wide",
                   initial_sidebar_state="expanded")

with st.sidebar:
    st.title("app7-simplex-ensemble")
    st.caption(
        "Ансамбль Simplex projection по W соседним origin (origin, origin−1, …, "
        "origin−W+1). Только simplex, равные веса по origin, без само-проверок "
        "и векторов направления — только траектории и усреднённый прогноз."
    )

    ticker   = st.text_input("Тикер", "SBER", key="ticker").upper().strip()
    interval = st.selectbox("Интервал", list(INTERVALS.keys()),
                            index=list(INTERVALS.keys()).index("1d"))
    if st.button("Обновить данные"):
        _load_candles.clear()
        _fetch_and_save(ticker, interval)
        st.rerun()

    st.divider()
    with st.expander("Метод"):
        st.markdown(
            "**Этап 1** (PCA sweep) считается ОДИН РАЗ по главному origin → единые "
            "d_min, d_max для всего окна.\n\n"
            "Для каждого origin_i = origin − i,  i = 0…W−1:\n\n"
            "1. **Этап 2** — каскадный поиск соседей (октавный ×2, "
            "`research/cascade_algorithm.md`) + **Simplex projection** "
            "(Sugihara & May, 1990: взвешенное среднее d+1 ближайших соседей, "
            "без подгонки коэффициентов) для d ∈ [d_min, d_max], усреднение "
            "по d → avg_price_i(h), h = 1…H, на ratio, обрезанном каузально "
            "до origin_i.\n"
            "2. Перевод в относительные величины: "
            "rel_i(h) = avg_price_i(h) / origin_price_i − 1.\n\n"
            "**Агрегация по origin** (h — шаг от СВОЕГО origin_i, не "
            "календарная дата): равновзвешенные mean_rel(h), [p25,p75](h) "
            "по W траекториям rel_i.\n\n"
            "⚠ origin_i почти полностью используют одну и ту же историю — это "
            "чувствительность к сдвигу origin, не независимые сценарии и не "
            "калиброванный доверительный интервал."
        )
        st.markdown(
            "**Синхронизация K с simplex:** каскад ищет K_cascade = d+1 "
            "соседей на всех промежуточных уровнях — ровно столько, сколько "
            "в итоге использует Simplex projection на финальном шаге. Больше "
            "искать незачем (доказательство в cascade_algorithm.md не зависит "
            "от величины xi_lwr). K_filter = 3·(p_fit+1)+ξ — отдельный "
            "параметр, нужен только LP-фильтру и LP-коррекции (запас точек "
            "для устойчивого локального PCA/SVD)."
        )

    st.divider()
    st.subheader("Согласование тракта")
    xy_x = st.slider("x", 1, 10, 3, key="xy_x",
                      help="Масштабный коэффициент в m = x·d + y.")
    xy_y = st.slider("y", 0, 20, 0, key="xy_y",
                      help="Сдвиг в m = x·d + y.")
    xi_add = st.slider("ξ (добавка к K_filter)", 0, 20, 1,
                       help="Добавка к K_filter = 3·(m+1) + ξ — запас соседей "
                            "для локального PCA/SVD внутри LP-фильтра и "
                            "LP-коррекции. Каскад/simplex этот параметр не "
                            "использует (K_cascade = d+1).")
    blend_alpha = st.slider("α (форма ↔ амплитуда)", 0.0, 1.0, 0.5, 0.05,
                            help="Метрика amp_cos: α·|log(‖x‖/‖y‖)|ₙ + (1−α)·(1−cos∠)ₙ.")

    st.divider()
    st.subheader("Каскадный поиск")
    n_levels = st.slider("Уровней каскада", 1, 6, 6,
                         help="Число уровней n (октавный ×2 каскад).")
    p_cascade_max = st.slider("Макс. окно каскада m(max)", 100, 5000, 5000, 100,
                              help="Уровни с mₖ > m(max) снимаются сверху.")
    st.caption("K_cascade = d + 1 на всех уровнях — синхронизировано с simplex, см. «Метод».")
    use_all_bars = st.checkbox("Все точки библиотеки", value=True,
                               help="Без обрезки глубины истории (bars=0).")
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
        f"K_dim = 3·(m+1)+{xi_add}  ·  d_min=⌊mean d(m,τ₁={pca_thr1})⌋  ·  "
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
    st.caption("⚠ Этап 2 (каскад + simplex) всё равно прогоняется W раз — при "
               "bars=все и n_levels=6 прогон может занять время, ждите прогресс-бар.")

    st.divider()
    st.subheader("Прогноз")
    horizon = st.slider("Горизонт H (баров от своего origin_i)", 1, 100, 20,
                        help="Число шагов вперёд от КАЖДОГО origin_i (не выровнено "
                             "по календарным датам между origin).")
    use_lp_corr = st.checkbox("LP-коррекция траектории", value=True)

    run_btn = st.button("▶  Прогноз (ансамбль по origin)", type="primary", use_container_width=True)

# ── Загрузка данных ───────────────────────────────────────────────────────────

data = _load_candles(ticker, interval)
if not data:
    st.info("Нет данных. Нажмите «Обновить данные» в сайдбаре.")
    st.stop()

times, close, open_, high, low = _to_arrays(data)
price_input = close
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
            horizon, use_lp_corr,
            progress_cb=_cb,
        )

    prog_bar.empty(); prog_text.empty()

    st.session_state.update({
        "mo_result": mo_result,
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

    agg = _aggregate(per_origin)
    band = {
        "mean_price": _orig_price * (1.0 + agg["mean_rel"]),
        "p25_price":  _orig_price * (1.0 + agg["p25"]),
        "p75_price":  _orig_price * (1.0 + agg["p75"]),
    }

    hide_lines = st.checkbox(
        "Скрыть линии отдельных origin на графике (оставить только среднее и полосу)",
        value=False, key="mo_hide_lines",
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
    if not hide_lines:
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

    # Комбинированная траектория и полоса (равновзвешенные mean/p25-p75 по origin)
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
    fig.add_trace(go.Scatter(
        x=main_x, y=np.concatenate([[_orig_price], mean_price]),
        mode="lines+markers",
        name=f"среднее по {n_valid} origin",
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
    _y_pad = (_y_hi - _y_lo) * 0.05

    fig.update_layout(
        height=600, xaxis_rangeslider_visible=False, xaxis_type="date",
        yaxis_range=[_y_lo - _y_pad, _y_hi + _y_pad],
        title=f"{ticker} {interval}  ·  W={n_valid} origin  ·  H={_hor}  ·  Simplex projection",
        template="plotly_dark", legend=dict(orientation="h", y=-0.18),
    )
    if fc_idx_main[-1] >= _n:
        fig.update_layout(xaxis_range=[_times[show_from], _time_at(int(fc_idx_main[-1]))])

    st.plotly_chart(fig, use_container_width=True)

    # ── Таблица по origin ──────────────────────────────────────────────────────
    st.subheader("Параметры по origin")
    _df = pd.DataFrame([
        {
            "origin_i": r["origin_i"],
            "смещение": f"−{r['i']}" if r["i"] > 0 else "0 (главный)",
            "d валидных": f"{r['n_valid_d']}/{r['n_total_d']}",
            "rel(H), %": round(float(r["rel"][-1]) * 100, 2),
        }
        for r in per_origin
    ])
    st.dataframe(_df, use_container_width=True, hide_index=True, height=min(400, 40 + 35 * len(_df)))
