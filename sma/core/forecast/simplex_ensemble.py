"""
simplex_ensemble — ансамбль прогнозов Simplex projection по нескольким origin.

Второй, независимый model_type в архитектуре прогнозов (см. band_lambda.py
для первого). Не заменяет band_lambda — вспомогательный инструмент оценки
ситуации: не полоса с λ-калибровкой, а W независимых траекторий (по origin,
origin−1, …, origin−W+1), каждая от каузального LP-фильтр→каскад→Simplex
projection пайплайна, агрегированных равновзвешенно.

Портировано из prototype/forcaster/ui/app7-simplex-ensemble.py (одобрено
пользователем 2026-08-15) практически без изменений сути — только убран
Streamlit (кэш/прогресс-бар заменены на обычный progress_cb, ввод — numpy-
массивы вместо session_state). Ключевые методические решения, перенесённые
без изменений:

  - logtrend causal OLS: ratio = close / exp(a + b·t) — стандарт проекта
    (CLAUDE.md «Ключевые методические решения»). Отдельная реализация
    (не sma.core.forecast.normalize.normalize), т.к. здесь дополнительно
    нужны причинные массивы a[t]/b[t] по каждому t — для реконструкции цены
    на разных origin_i без пересчёта тренда заново на каждом origin.
  - K_cascade = d_proj + 1, синхронизировано с Simplex projection (см.
    research/cascade_algorithm.md — xi_lwr достаточен на всех уровнях
    каскада при ЛЮБОМ значении xi; Simplex не требует запаса точек для
    регрессии, как S-map/LWR, поэтому K_cascade держится минимальным).
    K_filter = 3·(p_fit+1)+ξ — ОТДЕЛЬНЫЙ параметр только для LP-фильтра и
    LP-коррекции траектории (локальный PCA/SVD нуждается в запасе точек).

Causality contract (см. память feedback-causality-enforcement): все функции
здесь работают только с переданными им массивами, никакого алгоритмического
ограничителя нет. Причинность обеспечивается ЕДИНСТВЕННО тем, что вызывающий
код (sma/api/task_manager.py) грузит close/times уже обрезанными до origin
(get_candles(..., until=origin_ts)) — поэтому origin всегда последний
элемент переданных массивов, отдельного origin_index параметр не нужен.
"""

from __future__ import annotations

import numpy as np

# ── дефолты (см. index.html simplex-panel / прототип app7-simplex-ensemble.py) ──

DEFAULT_WINDOW = 20  # project feedback 2026-08-25: "в итоге нащупал что он удобнее" (was 10)
DEFAULT_HORIZON = 20
DEFAULT_XY_X = 3
DEFAULT_XY_Y = 0
DEFAULT_XI_ADD = 1
DEFAULT_BLEND_ALPHA = 0.5
DEFAULT_N_LEVELS = 6
DEFAULT_P_CASCADE_MAX = 5000
DEFAULT_BARS = 0
DEFAULT_PCA_P_RANGE = (3, 150)
DEFAULT_PCA_THR1 = 0.80
DEFAULT_PCA_THR2 = 0.85
DEFAULT_USE_LP_CORR = True

# ═══════════════════════════════════════════════════════════════════════════════
# Метрика расстояния: amp_cos (без изменений от прототипа)
# ═══════════════════════════════════════════════════════════════════════════════


def _dists(X: np.ndarray, q: np.ndarray, blend_alpha: float = 0.5) -> np.ndarray:
    """amp_cos: α·|log(‖x‖/‖y‖)|_norm + (1−α)·cosine_norm."""
    n = len(X)
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
# Logtrend (causal OLS) — с массивами a[t]/b[t] (нужны для реконструкции на
# разных origin_i без пересчёта; sma.core.forecast.normalize.normalize даёт
# только trend, без a/b — см. докстринг модуля).
# ═══════════════════════════════════════════════════════════════════════════════


def _logtrend_causal(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t)
    ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc)
    cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t)
    trend[:2] = close[:2]
    return trend, a, b


# ═══════════════════════════════════════════════════════════════════════════════
# LP-фильтр (causal, ratio → att)
# ═══════════════════════════════════════════════════════════════════════════════


def _lp_proj_causal(
    ratio: np.ndarray, m: int, d: int, k: int, n_iter: int, blend_alpha: float = 0.5
) -> np.ndarray:
    """Причинный LP-фильтр: att[t] вычисляется только по ratio[0..t]."""
    n = len(ratio)
    d_eff = min(d, m - 1)
    att = ratio.copy().astype(np.float64)
    if d_eff < 1 or n < m + 1:
        return att

    nw = n - m + 1

    for _ in range(n_iter):
        X_all = att[np.arange(nw)[:, None] + np.arange(m)]
        new_att = att.copy()

        for q in range(1, nw):
            X_hist = X_all[:q]
            curr = X_all[q]
            k_eff = min(k, len(X_hist))
            if k_eff < d_eff + 1:
                continue
            dists = _dists(X_hist, curr, blend_alpha)
            sel = np.argpartition(dists, k_eff - 1)[:k_eff]
            nn = X_hist[sel]
            center = nn.mean(0)
            _, _, Vt = np.linalg.svd(nn - center, full_matrices=False)
            proj = center + Vt[:d_eff].T @ (Vt[:d_eff] @ (curr - center))
            new_att[q + m - 1] = proj[-1]

        att = new_att

    return att


# ═══════════════════════════════════════════════════════════════════════════════
# Локальная размерность PCA (Этап 1)
# ═══════════════════════════════════════════════════════════════════════════════


def _pca_dim_local(X_nn: np.ndarray, threshold: float = 0.90) -> float:
    if len(X_nn) < 2:
        return float("nan")
    center = X_nn.mean(axis=0)
    _, s, _ = np.linalg.svd(X_nn - center, full_matrices=False)
    var = s ** 2
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
    """PCA-размерность по хвосту ratio (без LP, без каскада)."""
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
    X_lib = ratio[starts[:, None] + np.arange(p_search)].astype(np.float64)
    dists = _dists(X_lib, query, blend_alpha)
    k_eff = min(k_dim, len(X_lib))
    X_nn = X_lib[np.argpartition(dists, k_eff - 1)[:k_eff]]
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
        res = compute_local_dim(ratio, p, k_dim, pca_thresholds, blend_alpha)
        row = {"p_search": p, "K": k_dim}
        if "error" in res:
            for t in pca_thresholds:
                row[f"PCA_{int(t * 100)}"] = float("nan")
            row["_error"] = res["error"]
        else:
            for t in pca_thresholds:
                row[f"PCA_{int(t * 100)}"] = res[f"pca_{t}"]
            row["_error"] = ""
        rows.append(row)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Каскад (октавный ×2). K_cascade = d_proj + 1 передаётся снаружи —
# синхронизация с Simplex projection, см. докстринг модуля.
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
    """Строгий октавный каскад ×2 (research/cascade_algorithm.md)."""
    n = len(att)
    levels = _cascade_levels(p_fit, n_levels, p_cascade_max)
    p_top = levels[0]

    max_s = min(n - p_top, t_predict - p_top - 1)
    if max_s < 0:
        return None, None

    pool = np.arange(0, max_s + 1)
    if len(pool) < K_cascade:
        return None, None

    for level_idx, p_lv in enumerate(levels):
        is_last = level_idx == n_levels - 1

        if len(context) < p_lv:
            return None, None
        q = context[-p_lv:].astype(np.float64)

        valid = (pool + p_lv - 1) < n
        pool = pool[valid]
        if len(pool) == 0:
            return None, None

        X_pool = att[pool[:, None] + np.arange(p_lv)]
        dists = _dists(X_pool, q, blend_alpha)

        if is_last:
            valid_y = pool + p_fit < n
            pool_sm = pool[valid_y]
            X_sm = X_pool[valid_y]
            if len(pool_sm) < 2:
                return None, None
            return X_sm, att[pool_sm + p_fit]

        n_sel = min(K_cascade, len(pool))
        top_i = np.argpartition(dists, n_sel - 1)[:n_sel]
        sel_starts = pool[top_i]

        p_next = levels[level_idx + 1]
        r = p_lv - p_next
        offsets = np.arange(r + 1)
        flat = (sel_starts[:, None] + offsets[None, :]).ravel()
        max_s_next = min(n - p_next, t_predict - p_next - 1)
        flat = flat[(flat >= 0) & (flat <= max_s_next)]
        pool = np.unique(flat)
        if len(pool) == 0:
            return None, None

    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Simplex projection (Sugihara & May, 1990)
# ═══════════════════════════════════════════════════════════════════════════════


def _simplex_step(
    X_pool: np.ndarray, y_pool: np.ndarray, vec_fit: np.ndarray,
    blend_alpha: float, n_neighbors: int,
) -> float:
    """n_neighbors (= d_proj+1) ближайших соседей из пула, взвешенное среднее
    их известных y (без подгонки коэффициентов, без PCA-проекции):
        w_i = exp(−d_i / d_min) / Σ exp(−d_j / d_min),  d_min = min_i d_i
    """
    p_eff = min(X_pool.shape[1], len(vec_fit))
    Xf = X_pool[:, -p_eff:]
    vf = vec_fit[-p_eff:]
    d_arr = _dists(Xf, vf, blend_alpha)

    k_eff = min(n_neighbors, len(Xf))
    idx = np.argpartition(d_arr, k_eff - 1)[:k_eff]
    d_sel = d_arr[idx]
    y_sel = y_pool[idx]

    d_min = max(float(d_sel.min()), 1e-10)
    w = np.exp(-d_sel / d_min)
    w /= w.sum()
    return float((w * y_sel).sum())


# ═══════════════════════════════════════════════════════════════════════════════
# LP-коррекция траектории. Использует K_filter, НЕ K_cascade (см. докстринг).
# ═══════════════════════════════════════════════════════════════════════════════


def _lp_corr(
    v_m: np.ndarray, X_lib: np.ndarray, K_filter: int, d: int, blend_alpha: float
) -> float:
    k_eff = min(K_filter, len(X_lib))
    if k_eff < d + 1:
        return float(v_m[-1])
    dists = _dists(X_lib, v_m, blend_alpha)
    idx = np.argpartition(dists, k_eff - 1)[:k_eff]
    X_nn = X_lib[idx]
    center = X_nn.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_nn - center, full_matrices=False)
    d_eff = min(d, Vt.shape[0])
    Vd = Vt[:d_eff].T
    v_proj = center + Vd @ (Vd.T @ (v_m - center))
    return float(v_proj[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# Прогноз: один d, полный горизонт.
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
    K_cascade = d_proj + 1 на всех уровнях каскада и на финальном шаге simplex."""
    K_cascade = d_proj + 1
    n_eff = origin + 1
    _min_pool = d_proj + 2
    fc_buf = list(att[:n_eff])

    X_lib_lp: np.ndarray | None = None
    if use_lp_corr:
        n_lib = n_eff - p_fit
        if n_lib > 0:
            X_lib_lp = att[np.arange(n_lib)[:, None] + np.arange(p_fit)]

    for h in range(horizon):
        t = n_eff + h
        ctx = np.array(fc_buf)
        X_nn, y_nn = cascade_search(
            att, ctx, p_fit, n_levels, K_cascade, t, blend_alpha, p_cascade_max,
        )
        if X_nn is None or len(X_nn) < _min_pool:
            return None
        pred = _simplex_step(X_nn, y_nn, ctx[-p_fit:], blend_alpha, K_cascade)
        if use_lp_corr and X_lib_lp is not None:
            v_m = np.array(fc_buf[-(p_fit - 1):] + [pred], dtype=np.float64)
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
    blend_alpha: float = 0.5,
    p_cascade_max: int = 1500,
    use_lp_corr: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Возвращает (results, failed). results содержит fc_preds (сырые att)."""
    results = []
    failed = []
    for d in d_values:
        p_fit = xy_x * d + xy_y
        K_filter = 3 * (p_fit + 1) + xi_add  # только для LP-фильтра и LP-коррекции
        d_proj = d

        if p_fit < 2:
            failed.append({"d": d, "reason": "p_fit < 2"})
            continue

        att = _lp_proj_causal(ratio, p_fit, d, K_filter, 1, blend_alpha)

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
            failed.append({"d": d, "reason": f"pool < d+2={d + 2}"})
            continue

        results.append({"d": d, "m": p_fit, "K_filter": K_filter, "fc_preds": fc_preds})

    return results, failed


# ═══════════════════════════════════════════════════════════════════════════════
# Ансамбль по нескольким origin
# ═══════════════════════════════════════════════════════════════════════════════


def _run_stage1_for_origin(
    ratio_i: np.ndarray, pca_p_range: tuple, xi_add: int,
    pca_thr1: float, pca_thr2: float, blend_alpha: float,
) -> tuple[int | None, int | None, list[dict]]:
    """Этап 1 (PCA sweep) для одного origin. Возвращает (d_min, d_max, s1_rows)
    или (None, None, s1_rows)."""
    pca_p_vals = list(range(pca_p_range[0], pca_p_range[1] + 1))
    s1_rows = sweep_local_dim(
        ratio_i, p_values=pca_p_vals, xi_add=xi_add,
        pca_thresholds=(pca_thr1, pca_thr2), blend_alpha=blend_alpha,
    )
    col1 = f"PCA_{int(pca_thr1 * 100)}"
    col2 = f"PCA_{int(pca_thr2 * 100)}"
    vals1 = [r[col1] for r in s1_rows if np.isfinite(r[col1])]
    vals2 = [r[col2] for r in s1_rows if np.isfinite(r[col2])]
    if not vals1 or not vals2:
        return None, None, s1_rows
    d_min = max(2, int(np.floor(np.mean(vals1))))
    d_max = max(d_min, int(np.ceil(np.mean(vals2))))
    return d_min, d_max, s1_rows


def _aggregate(per_origin: list[dict]) -> dict:
    """Равновзвешенные mean/std/p25/p75 по origin для каждого h (доступный
    первичный агрегат; произвольная ширина полосы P пересчитывается реактивно
    на фронтенде из per_origin[*]['rel'], см. sma/ui/chart.js)."""
    rel_stack = np.array([r["rel"] for r in per_origin])  # (n_origins, horizon)
    mean_rel = rel_stack.mean(axis=0)
    std_rel = rel_stack.std(axis=0)
    p25 = np.quantile(rel_stack, 0.25, axis=0)
    p75 = np.quantile(rel_stack, 0.75, axis=0)
    return {"mean_rel": mean_rel, "std_rel": std_rel, "p25": p25, "p75": p75}


def _forecast_one_origin(
    ratio_full: np.ndarray, a_arr: np.ndarray, b_arr: np.ndarray, price_input: np.ndarray,
    origin: int, i: int, bars: int,
    xy_x: int, xy_y: int, xi_add: int, n_levels: int, p_cascade_max: int,
    d_values: list[int], blend_alpha: float, use_lp_corr: bool, horizon: int,
) -> dict:
    """
    One origin_i's worth of run_multi_origin's per-origin loop, split out as
    a top-level (picklable) function so it can run in a ProcessPoolExecutor
    worker — see run_multi_origin's docstring for why each origin_i is safe
    to compute independently. Returns either a "skipped" shape (`reason` key
    present) or a full per_origin entry; run_multi_origin tells them apart
    via `"reason" in result` when reassembling its two output lists.
    """
    origin_i = origin - i
    if origin_i < 10:
        return {"i": i, "origin_i": origin_i, "reason": "origin_i < 10 (мало истории)"}

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
        blend_alpha, p_cascade_max, use_lp_corr,
    )
    if not s2_results:
        return {
            "i": i, "origin_i": origin_i,
            "reason": f"Этап 2: нет валидных d (d∈[{d_values[0]},{d_values[-1]}])",
        }

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

    return {
        "i": i, "origin_i": origin_i,
        "n_valid_d": len(s2_results), "n_total_d": len(d_values),
        "origin_price": origin_price_i,
        "avg_price": avg_price_i,
        "rel": rel_i,
    }


def init_worker_env() -> None:
    """
    ProcessPoolExecutor initializer — BLAS oversubscription guard (see memory
    blas_oversubscription_multiprocessing): each worker process must cap its
    own thread pool BEFORE numpy touches BLAS, or N worker processes x N BLAS
    threads each thrash the CPU instead of speeding anything up. Same fix as
    range_forecast_calibrator.init_worker_env, duplicated (not imported) to
    keep the model modules independent.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def run_multi_origin(
    ratio_full: np.ndarray,
    a_arr: np.ndarray, b_arr: np.ndarray, price_input: np.ndarray,
    origin: int, window: int, bars: int,
    xy_x: int, xy_y: int, xi_add: int, n_levels: int, p_cascade_max: int,
    pca_p_range: tuple, pca_thr1: float, pca_thr2: float, blend_alpha: float,
    horizon: int, use_lp_corr: bool,
    progress_cb=None,
    max_workers: int = 1,
) -> dict:
    """
    Прогоняет пайплайн (LP-фильтр → каскад → Simplex projection) для
    origin, origin−1, …, origin−(window−1). Каждый origin_i получает СВОЮ
    причинную обрезку ratio (ratio_full[:origin_i+1], затем bars) — никакой
    утечки между origin.

    Этап 1 (PCA-sweep → d_min/d_max) считается ОДИН РАЗ по главному origin и
    переиспользуется для всех origin_i в окне.

    Каждый origin_i независим от остальных (свои ratio_i/origin_algo_i,
    читает только общие read-only d_values/ratio_full/a_arr/b_arr) — при
    max_workers>1 window origin'ов считаются в ProcessPoolExecutor (project
    feedback 2026-08-25: "можем ввести многопоточность для него, а то он
    долго считается"), initializer=init_worker_env (BLAS thread cap на
    процесс, тот же приём, что и калибровка band_lambda). max_workers<=1
    (умолчание) — старый последовательный путь без накладных расходов на
    процессы, важно для маленьких window и для вызова без multiprocessing
    (напр. будущие тесты). Порядок progress_cb-вызовов при max_workers>1 не
    гарантированно по возрастанию i (futures завершаются в порядке
    готовности) — это только счётчик done/window для UI, порядок финальных
    per_origin/skipped списков ниже всегда восстанавливается по i, как в
    последовательном пути.

    Возвращает dict с per_origin (список по origin, каждый с "rel" —
    относительной траекторией rel_i(h) = price_i(h)/origin_price_i − 1) либо
    {"error": "..."} при неустранимой ошибке (нет истории/не удалось
    определить d_min/d_max/ни один origin не дал прогноза).
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

    results_by_i: dict[int, dict] = {}
    if max_workers > 1 and window > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker_env) as executor:
            futures = {
                executor.submit(
                    _forecast_one_origin,
                    ratio_full, a_arr, b_arr, price_input, origin, i, bars,
                    xy_x, xy_y, xi_add, n_levels, p_cascade_max, d_values,
                    blend_alpha, use_lp_corr, horizon,
                ): i
                for i in range(window)
            }
            done = 0
            for fut in as_completed(futures):
                results_by_i[futures[fut]] = fut.result()
                done += 1
                if progress_cb:
                    progress_cb(done, window)
    else:
        for i in range(window):
            results_by_i[i] = _forecast_one_origin(
                ratio_full, a_arr, b_arr, price_input, origin, i, bars,
                xy_x, xy_y, xi_add, n_levels, p_cascade_max, d_values,
                blend_alpha, use_lp_corr, horizon,
            )
            if progress_cb:
                progress_cb(i + 1, window)

    per_origin = []
    skipped: list[dict] = []
    for i in range(window):
        r = results_by_i[i]
        if "reason" in r:
            skipped.append({"origin_i": r["origin_i"], "reason": r["reason"]})
        else:
            per_origin.append(r)

    if not per_origin:
        return {"error": "Ни один origin не дал валидного прогноза.", "skipped": skipped, "d_min": d_min, "d_max": d_max}

    return {
        "per_origin": per_origin,
        "skipped": skipped,
        "d_min": d_min, "d_max": d_max,
        "horizon": horizon,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Точка входа для task_manager: массивы close/times уже причинно обрезаны до
# origin (последний элемент = origin) вызывающим кодом — см. докстринг модуля.
# ═══════════════════════════════════════════════════════════════════════════════


def forecast_ensemble(
    times: np.ndarray, close: np.ndarray,
    window: int = DEFAULT_WINDOW, horizon: int = DEFAULT_HORIZON,
    xy_x: int = DEFAULT_XY_X, xy_y: int = DEFAULT_XY_Y, xi_add: int = DEFAULT_XI_ADD,
    blend_alpha: float = DEFAULT_BLEND_ALPHA,
    n_levels: int = DEFAULT_N_LEVELS, p_cascade_max: int = DEFAULT_P_CASCADE_MAX,
    bars: int = DEFAULT_BARS,
    pca_p_range: tuple = DEFAULT_PCA_P_RANGE,
    pca_thr1: float = DEFAULT_PCA_THR1, pca_thr2: float = DEFAULT_PCA_THR2,
    use_lp_corr: bool = DEFAULT_USE_LP_CORR,
    progress_cb=None,
    max_workers: int = 1,
) -> dict:
    """
    Полный пайплайн от close/times до result_json-контракта (см. докстринг
    модуля sma/api/task_manager.py::_run_forecast_simplex). origin = len(close)-1
    (последний бар переданных массивов — причинность обеспечена обрезкой на
    вызывающей стороне, не здесь).

    max_workers — прокидывается в run_multi_origin как есть (см. его
    докстринг); >1 распараллеливает per-origin цикл через ProcessPoolExecutor.

    Возвращает {"error": "..."} при неустранимой ошибке — вызывающий код
    (task_manager) должен проверить это и поднять ValueError с этим текстом.
    """
    n = len(close)
    origin = n - 1
    price_input = close
    logtrend, a_arr, b_arr = _logtrend_causal(price_input)
    ratio_full = price_input / np.maximum(logtrend, 1e-10)

    mo = run_multi_origin(
        ratio_full, a_arr, b_arr, price_input,
        origin, window, bars,
        xy_x, xy_y, xi_add, n_levels, p_cascade_max,
        pca_p_range, pca_thr1, pca_thr2, blend_alpha,
        horizon, use_lp_corr,
        progress_cb=progress_cb,
        max_workers=max_workers,
    )
    if "error" in mo:
        return mo

    agg = _aggregate(mo["per_origin"])
    origin_date = str(times[origin])
    origin_price = float(close[origin])

    return {
        "origin_date": origin_date,
        "origin_extreme_date": origin_date,  # нет пивота — origin сам себе "экстремум"
        "origin_price": origin_price,
        "origin_direction": 1 if float(agg["mean_rel"][-1]) >= 0 else -1,
        "horizon": horizon, "window": window,
        "d_min": mo["d_min"], "d_max": mo["d_max"],
        "per_origin": [
            {
                "origin_i": int(r["origin_i"]),
                "origin_date": str(times[r["origin_i"]]),
                "origin_price": r["origin_price"],
                "rel": r["rel"].tolist(),
            }
            for r in mo["per_origin"]
        ],
        "skipped": mo["skipped"],
    }
