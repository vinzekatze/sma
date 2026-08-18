"""
app7: State-Space DMD — локально взвешенный Koopman-прогноз.

Заменяет S-map из app6 на матрицу перехода A (d×d):
  z_{t+1} ≈ A @ z_t   (в d-мерном PCA-подпространстве)

Два режима:
  Итерация       — A перефитируется на каждом шаге h по соседям текущего z_h
  Матр. степень  — A вычисляется один раз при h=0, затем z_h = A^h @ z_0

Реконструкция att из z: x = V @ z + center, att_next = x[-1]
Стек: logtrend causal OLS → LP-фильтр → каскадный поиск → DMD (ridge + clip λ).
Диагностика: спектр λ(A), |λ_max|(h), cond(Z'WZ), число клиппингов.

Run (из prototype/): streamlit run forcaster/ui/app7.py
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
    Соседи — все исторические окна, завершившиеся строго до t.
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
    """PCA-размерность по хвосту ratio. ratio уже обрезан до origin+bars."""
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
# Каскадный поиск (базовый — без DMD-пар)
# ═══════════════════════════════════════════════════════════════════════════════

def _cascade_levels(p_fit: int, n_levels: int, p_cascade_max: int = 1500) -> list[int]:
    levels = [p_fit * (2 ** (n_levels - 1 - k)) for k in range(n_levels)]
    levels = [p for p in levels if p <= p_cascade_max]
    return levels if levels else [p_fit]


def cascade_search_dmd(
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
    Октавный каскад. На последнем уровне возвращает пары окон:
      X_sm[i]      = att[pool[i] : pool[i]+p_fit]       — текущее состояние
      X_sm_next[i] = att[pool[i]+1 : pool[i]+p_fit+1]   — следующее состояние

    Оба окна гарантированно каузальны: pool[i]+p_fit < n (= len(att) = origin+1).
    """
    n      = len(att)
    levels = _cascade_levels(p_fit, n_levels, p_cascade_max)
    p_top  = levels[0]

    max_s = min(n - p_top, t_predict - p_top - 1)
    if max_s < 0:
        return None, None

    pool = np.arange(0, max_s + 1)
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
            # pool[i] + p_fit < n  →  X_sm и X_sm_next (до pool[i]+p_fit) каузальны
            valid_y   = pool + p_fit < n
            pool_sm   = pool[valid_y]
            if len(pool_sm) < 2:
                return None, None
            X_sm      = att[pool_sm[:, None] + np.arange(p_lv)]
            X_sm_next = att[pool_sm[:, None] + 1 + np.arange(p_lv)]
            return X_sm, X_sm_next

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
# DMD: построение локальной матрицы перехода A
# ═══════════════════════════════════════════════════════════════════════════════

def _dmd_build(
    X_pool: np.ndarray,       # (K, p_fit)  — текущие состояния соседей
    X_pool_next: np.ndarray,  # (K, p_fit)  — следующие состояния соседей
    vec_fit: np.ndarray,      # (p_fit,)    — запросный вектор (текущее состояние)
    blend_alpha: float,
    theta: float,
    d_proj: int,
    ridge_alpha: float,
    clip_eigenvalues: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Локально взвешенный DMD: A = argmin_A Σ w_i ‖z_{i+1} − A z_i‖²  + α‖A‖_F².

    PCA-базис строится по текущим состояниям X_pool (те же K соседей).
    Один базис используется для проекции как текущих, так и следующих состояний.

    Возвращает: A (d×d), z_q (d,), V (p×d), center (p,), diag dict.
    Реконструкция att: x = V @ z + center,  att_next = x[-1].
    """
    p_eff = min(X_pool.shape[1], len(vec_fit))
    Xf      = X_pool     [:, -p_eff:].astype(np.float64)
    Xf_next = X_pool_next[:, -p_eff:].astype(np.float64)
    vf      = vec_fit[-p_eff:].astype(np.float64)

    # PCA-базис из текущих состояний
    center = Xf.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xf - center, full_matrices=False)
    d_eff = min(d_proj, Vt.shape[0], len(Xf) - 1)
    d_eff = max(d_eff, 1)
    V  = Vt[:d_eff].T          # (p_eff, d_eff)

    # Проекция на общий базис
    Zf      = (Xf      - center) @ V  # (K, d_eff)
    Zf_next = (Xf_next - center) @ V  # (K, d_eff)
    z_q     = (vf      - center) @ V  # (d_eff,)

    # Веса (как в S-map): exp(−θ · dist / mean_dist)
    d_arr  = _dists(Xf, vf, blend_alpha)
    d_mean = max(float(d_arr.mean()), 1e-10)
    w      = np.exp(-theta * d_arr / d_mean)

    # Взвешенный ridge: A^T (Z^T W Z + αI) = Z_next^T W Z
    #   ⟺  A = (Z_next^T W Z)(Z^T W Z + αI)^{-1}
    W     = w[:, None]
    ZtWZ  = (Zf * W).T @ Zf                      # (d, d)
    ZnWZ  = (Zf_next * W).T @ Zf                 # (d, d)
    reg   = ridge_alpha * np.eye(d_eff)
    try:
        A = ZnWZ @ np.linalg.solve(ZtWZ + reg, np.eye(d_eff))
    except np.linalg.LinAlgError:
        A = np.eye(d_eff)

    # Диагностика до стабилизации
    eigvals_raw  = np.linalg.eigvals(A)
    mags_raw     = np.abs(eigvals_raw)
    lambda_max_raw = float(mags_raw.max())
    cond_ZtWZ    = float(np.linalg.cond(ZtWZ + reg))
    n_clipped    = 0

    # Стабилизация: клиппинг собственных значений на единичный круг
    if clip_eigenvalues:
        try:
            eigvals_c, eigvecs = np.linalg.eig(A)
            mags_c = np.abs(eigvals_c)
            n_clipped = int(np.sum(mags_c > 1.0))
            if n_clipped > 0:
                scale = np.where(mags_c > 1.0, 1.0 / mags_c, np.ones_like(mags_c))
                eigvals_c_clip = eigvals_c * scale
                A = (eigvecs @ np.diag(eigvals_c_clip) @ np.linalg.inv(eigvecs)).real
        except np.linalg.LinAlgError:
            pass

    eigvals_final    = np.linalg.eigvals(A)
    lambda_max_final = float(np.abs(eigvals_final).max())

    diag = {
        "eigvals_raw":       eigvals_raw,
        "eigvals_final":     eigvals_final,
        "lambda_max_raw":    lambda_max_raw,
        "lambda_max_final":  lambda_max_final,
        "cond_ZtWZ":        cond_ZtWZ,
        "n_clipped":        n_clipped,
        "K":                len(X_pool),
        "d_eff":            d_eff,
    }
    return A, z_q, V, center, diag

# ═══════════════════════════════════════════════════════════════════════════════
# Прогноз: итерация (A перефитируется на каждом шаге)
# ═══════════════════════════════════════════════════════════════════════════════

def run_forecast_dmd_iter(
    att: np.ndarray,
    p_fit: int, n_levels: int, K: int,
    horizon: int, origin: int,
    blend_alpha: float, theta: float, d_proj: int,
    ridge_alpha: float, clip_eigenvalues: bool,
    p_cascade_max: int = 1500,
) -> tuple[np.ndarray | None, list[dict]]:
    """
    На каждом шаге h: находит соседей около текущего z, перефитирует A_h,
    делает один шаг z_{h+1} = A_h @ z_h, извлекает att_next = (V @ z_next + center)[-1].
    """
    n_eff     = origin + 1
    _min_pool = d_proj + 2
    fc_buf    = list(att[:n_eff])
    diags     = []

    for h in range(horizon):
        t   = n_eff + h
        ctx = np.array(fc_buf)
        X_nn, X_nn_next = cascade_search_dmd(
            att, ctx, p_fit, n_levels, K, t, blend_alpha, p_cascade_max,
        )
        if X_nn is None or len(X_nn) < _min_pool:
            return None, diags
        A, z_q, V, center, diag = _dmd_build(
            X_nn, X_nn_next, ctx[-p_fit:],
            blend_alpha, theta, d_proj, ridge_alpha, clip_eigenvalues,
        )
        z_next   = A @ z_q
        x_next   = V @ z_next + center   # (p_eff,)
        att_next = float(x_next[-1])     # последний элемент = att в следующий момент
        diags.append(diag)
        fc_buf.append(att_next)

    return np.array(fc_buf[n_eff:]), diags

# ═══════════════════════════════════════════════════════════════════════════════
# Прогноз: матричная степень (A вычисляется один раз, z_h = A^h @ z_0)
# ═══════════════════════════════════════════════════════════════════════════════

def run_forecast_dmd_matpow(
    att: np.ndarray,
    p_fit: int, n_levels: int, K: int,
    horizon: int, origin: int,
    blend_alpha: float, theta: float, d_proj: int,
    ridge_alpha: float, clip_eigenvalues: bool,
    p_cascade_max: int = 1500,
) -> tuple[np.ndarray | None, list[dict]]:
    """
    A вычисляется один раз по соседям в точке origin.
    Затем z_h = A^h @ z_0 (итеративное применение одной матрицы).
    Реконструкция att_h = (V @ z_h + center)[-1] с фиксированным базисом.
    """
    n_eff     = origin + 1
    _min_pool = d_proj + 2
    ctx = np.array(att[:n_eff])
    t   = n_eff

    X_nn, X_nn_next = cascade_search_dmd(
        att, ctx, p_fit, n_levels, K, t, blend_alpha, p_cascade_max,
    )
    if X_nn is None or len(X_nn) < _min_pool:
        return None, []

    A, z_0, V, center, diag_0 = _dmd_build(
        X_nn, X_nn_next, ctx[-p_fit:],
        blend_alpha, theta, d_proj, ridge_alpha, clip_eigenvalues,
    )

    preds = []
    z_h   = z_0.copy()
    for _ in range(horizon):
        z_h = A @ z_h
        x_h = V @ z_h + center
        preds.append(float(x_h[-1]))

    return np.array(preds), [diag_0]

# ═══════════════════════════════════════════════════════════════════════════════
# Этап 2: sweep по d с DMD
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage2_dmd(
    ratio: np.ndarray,
    d_values: list[int],
    xy_x: int, xy_y: int, xi_add: int,
    n_levels: int, horizon: int, origin: int,
    propagation_mode: str,
    progress_cb=None,
    blend_alpha: float = 0.5,
    theta: float = 18.0,
    ridge_alpha: float = 0.1,
    clip_eigenvalues: bool = True,
    p_cascade_max: int = 1500,
) -> tuple[list[dict], list[dict]]:
    """propagation_mode: 'iter' | 'matpow' | 'both'."""
    results = []
    failed  = []

    for step_i, d in enumerate(d_values):
        if progress_cb:
            progress_cb(step_i, len(d_values), d)

        p_fit  = xy_x * d + xy_y
        K      = 3 * (p_fit + 1) + xi_add

        if p_fit < 2:
            failed.append({"d": d, "reason": "p_fit < 2"})
            continue

        att = _lp_proj_ratio_cached(ratio.tobytes(), p_fit, d, K, 1, blend_alpha)

        eff_levels = _cascade_levels(p_fit, n_levels, p_cascade_max)
        min_len = eff_levels[0] + p_fit + 1
        if len(att) < min_len:
            failed.append({"d": d, "reason": f"ряд мал (нужно {min_len})"})
            continue

        kwargs = dict(
            att=att, p_fit=p_fit, n_levels=n_levels, K=K,
            horizon=horizon, origin=origin,
            blend_alpha=blend_alpha, theta=theta, d_proj=d,
            ridge_alpha=ridge_alpha, clip_eigenvalues=clip_eigenvalues,
            p_cascade_max=p_cascade_max,
        )

        fc_iter    = fc_matpow    = None
        diags_iter = diags_matpow = None

        if propagation_mode in ("iter", "both"):
            fc_i, di = run_forecast_dmd_iter(**kwargs)
            if fc_i is not None:
                fc_iter, diags_iter = fc_i, di
            else:
                failed.append({"d": d, "reason": "iter: пул мал"})

        if propagation_mode in ("matpow", "both"):
            fc_m, dm = run_forecast_dmd_matpow(**kwargs)
            if fc_m is not None:
                fc_matpow, diags_matpow = fc_m, dm
            else:
                failed.append({"d": d, "reason": "matpow: пул мал"})

        if fc_iter is None and fc_matpow is None:
            continue

        results.append({
            "d":            d,
            "m":            p_fit,
            "K":            K,
            "fc_iter":      fc_iter,
            "fc_matpow":    fc_matpow,
            "diags_iter":   diags_iter,
            "diags_matpow": diags_matpow,
        })

    return results, failed

# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="app7 · State-Space DMD", layout="wide",
                   initial_sidebar_state="expanded")

with st.sidebar:
    st.title("app7 · State-Space DMD")

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
            "| Фильтрация | **LP-фильтр** (Local Projective): каузальная SVD-проекция ratio → att |\n"
            "| Оценка d | **PCA sweep**: SVD облака K соседей → локальная размерность аттрактора |\n"
            "| Поиск соседей | **Каскадный поиск**: иерархический coarse-to-fine NN |\n"
            "| Прогноз | **DMD** (Locally Weighted DMD): локальная матрица перехода A (d×d) |\n\n"
            "---\n\n"
            "**DMD в двух режимах**\n\n"
            "| Символ | Описание |\n"
            "|:------:|----------|\n"
            "| z_t | PCA-проекция delay-вектора att в d-мерное подпространство |\n"
            "| A | матрица перехода: z_{t+1} ≈ A @ z_t |\n"
            "| α_ridge | коэффициент L2-регуляризации A |\n"
            "| λ_i(A) | собственные значения A; |λ_i|≤1 → устойчивость |\n"
            "| Итерация | A перефитируется на каждом шаге h по K соседям текущего z_h |\n"
            "| Матр. степень | A вычисляется один раз при h=0; z_h = A^h @ z_0 |\n\n"
            "**Реконструкция att из z:**\n\n"
            "```\n"
            "x = V @ z + center   (p-мерный delay-вектор)\n"
            "att_next = x[-1]     (последний элемент = att следующего шага)\n"
            "```\n\n"
            "---\n\n"
            "**Схема пайплайна**\n\n"
            "```\n"
            "close → ratio = close / exp(a+b·t)\n"
            "\n"
            "Этап 1 — оценка d (PCA sweep):\n"
            "  for m ∈ [m_min, m_max]:\n"
            "    K = 3·(m+1) + ξ соседей для origin в ℝᵐ\n"
            "    SVD → d(m, τ₁), d(m, τ₂)\n"
            "  d_min = ⌊mean d(m,τ₁)⌋,  d_max = ⌈mean d(m,τ₂)⌉\n"
            "\n"
            "Этап 2 — прогноз DMD:\n"
            "  for d ∈ [d_min, d_max]:\n"
            "    m = x·d + y,   K = 3·(m+1) + ξ\n"
            "    att = LP(ratio, m, d)          ← LP-фильтр\n"
            "    Итерация: for h=1…H:\n"
            "      cascade(att, m, K) → (X_pool, X_pool_next)\n"
            "      A = WLS_ridge(X_pool, X_pool_next, w)\n"
            "      clip λ(A) на |λ|≤1\n"
            "      z_{h+1} = A @ z_h\n"
            "      att_next = (V @ z_next + μ)[-1]\n"
            "    Матр. степень: A один раз, z_h = A^h @ z_0\n"
            "    price = att · exp(a+b·t)\n"
            "  mean(price_d) → финальный прогноз\n"
            "```"
        )

    st.divider()
    st.subheader("Согласование тракта")
    st.caption(
        "m = x·d + y,   K = 3·(m+1) + ξ  —  единые для LP и DMD."
    )
    xy_x = st.slider("x", 1, 10, 2, key="xy_x")
    xy_y = st.slider("y", 0, 20, 3, key="xy_y")
    st.caption(f"m = x·d + y  ·  d=5 → m={xy_x*5+xy_y}  ·  d=20 → m={xy_x*20+xy_y}")
    xi_add = st.slider("ξ (добавка к K)", 0, 20, 1)
    blend_alpha = st.slider("α (форма ↔ амплитуда)", 0.0, 1.0, 0.5, 0.05,
                            help="Метрика amp_cos: α·|log(‖x‖/‖y‖)|ₙ + (1−α)·(1−cos∠(x,y))ₙ")
    st.caption(
        f"ρ(x,y) = {blend_alpha}·|log(‖x‖/‖y‖)|ₙ + {round(1-blend_alpha, 2)}·(1−cos∠(x,y))ₙ"
    )

    st.divider()
    st.subheader("Каскадный поиск")
    n_levels = st.slider("Уровней каскада", 1, 6, 4)
    p_cascade_max = st.slider("Макс. окно каскада m(max)", 100, 5000, 2000, 100)
    bars = st.slider("Точек в библиотеке", 100, 5000, 3000, 100)
    _ex_p    = xy_x * 20 + xy_y
    _ex_lvls = _cascade_levels(_ex_p, n_levels, p_cascade_max)
    st.caption(f"Пример d=20: m={_ex_p},  уровни={_ex_lvls}  ({len(_ex_lvls)} ур.)")

    st.divider()
    st.subheader("Измерение размерности PCA")
    pca_p_range = st.slider("Диапазон m", 3, 300, (3, 100), key="pca_p_range")
    pca_thr_range = st.slider("Диапазон порогов", 0.80, 0.99, (0.95, 0.99), 0.01,
                              key="pca_thr_range")
    pca_thr1, pca_thr2 = pca_thr_range[0], pca_thr_range[1]
    _n_pts = pca_p_range[1] - pca_p_range[0] + 1
    st.caption(
        f"{_n_pts} точек  ·  K = 3·(m+1)+{xi_add}  ·  "
        f"d_min = ⌊mean d(m,τ₁={pca_thr1})⌋  ·  d_max = ⌈mean d(m,τ₂={pca_thr2})⌉"
    )
    pca_btn = st.button("Измерить PCA", use_container_width=True)

    st.divider()
    st.subheader("Прогноз DMD")
    horizon = st.slider("Горизонт (баров)", 1, 200, 80)
    origin_offset = st.slider("Точка отсчёта (баров от конца)", 0, 500, 0,
                              help="0 = прогноз из последней точки. >0 = ретроспективная проверка.")
    dmd_theta = st.slider("θ (веса соседей)", 0.0, 50.0, 18.0, 0.5,
                          help="Экспоненциальные веса при фитинге A:\n"
                               "  w_i = exp(−θ · ρ(x_i, x_q) / ρ̄)\n"
                               "θ=0: равные веса (глобальная линейная DMD)\n"
                               "θ→∞: только ближайший сосед")
    ridge_alpha = st.slider("α_ridge (регуляризация A)", 0.001, 5.0, 0.1, 0.001,
                            help="L2-регуляризация матрицы перехода A:\n"
                                 "  A = (Z_next'WZ)(Z'WZ + α_ridge·I)⁻¹\n"
                                 "Малый α_ridge → A слабо регуляризована (рискованно при K < d²)\n"
                                 "Большой α_ridge → A стягивается к нулю (сглаживание)")
    st.caption(f"K/d² пример d=20: {3*(xy_x*20+xy_y+1)+xi_add}/{20**2} = "
               f"{(3*(xy_x*20+xy_y+1)+xi_add)/400:.2f}")
    clip_eigenvalues = st.checkbox("Клиппинг λ(A) на |λ|≤1", value=True,
                                   help="Стабилизация: собственные значения |λ|>1 проецируются "
                                        "на единичный круг λ ← λ/|λ|.\n"
                                        "Гарантирует ограниченность z_h при больших h.")
    propagation_mode = st.radio(
        "Режим распространения",
        ["Оба", "Итерация", "Матр. степень"],
        help="Итерация: A перефитируется на каждом шаге по соседям текущего z_h\n"
             "Матр. степень: A вычисляется один раз, z_h = A^h @ z_0",
    )
    _mode_map = {"Оба": "both", "Итерация": "iter", "Матр. степень": "matpow"}
    prop_mode = _mode_map[propagation_mode]

    run_btn = st.button("▶  Прогноз", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Коррекция")
    delta_ratio = st.slider("Поправка тренда (Δratio/бар)", -0.005, 0.005, 0.0, 0.0001,
                            format="%.4f",
                            help="Линейная поправка: âtt[j] ← âtt[j] + j·δ")
    if delta_ratio != 0.0:
        st.caption(f"Суммарная поправка за горизонт: {delta_ratio * horizon * 100:+.2f}% ratio")

# ── Загрузка данных ───────────────────────────────────────────────────────────

data = _load_candles(ticker, interval)
if not data:
    st.info("Нет данных. Нажмите «Обновить данные» в сайдбаре.")
    st.stop()

times, close, open_, high, low = _to_arrays(data)
logtrend, _a_arr, _b_arr = _logtrend_causal(close)
ratio  = close / np.maximum(logtrend, 1e-10)
n      = len(close)
origin = max(0, min(n - 1, n - 1 - origin_offset))
a_lt   = float(_a_arr[origin])
b_lt   = float(_b_arr[origin])
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
    d_min, d_max = _run_stage1(
        ratio, pca_p_range, xi_add, pca_thr1, pca_thr2, blend_alpha,
    )

    if d_min is None or d_max is None:
        st.error("Этап 1: не удалось определить d_min/d_max.")
        st.session_state.pop("s2_results", None)
    else:
        d_values  = list(range(d_min, d_max + 1))
        prog_bar  = st.progress(0.0)
        prog_text = st.empty()

        def _cb(step: int, total: int, d: int) -> None:
            prog_bar.progress(step / total if total > 0 else 0.0)
            prog_text.caption(f"Этап 2 DMD: {step+1}/{total} — d={d}, m={xy_x*d+xy_y}")

        with st.spinner(f"Этап 2 DMD: d∈[{d_min},{d_max}] ({len(d_values)} значений)…"):
            s2_results, s2_failed = run_stage2_dmd(
                ratio,
                d_values, xy_x, xy_y, xi_add,
                n_levels, horizon, origin_algo,
                prop_mode,
                _cb,
                blend_alpha,
                dmd_theta,
                ridge_alpha,
                clip_eigenvalues,
                p_cascade_max,
            )

        prog_bar.empty(); prog_text.empty()

        st.session_state.update({
            "s2_results":   s2_results,
            "s2_failed":    s2_failed,
            "s2_d_values":  d_values,
            "s2_origin":    origin,
            "s2_n":         n,
            "s2_times":     times,
            "s2_horizon":   horizon,
            "s2_a_lt":      a_lt,
            "s2_b_lt":      b_lt,
            "s2_n_eff":     origin + 1,
            "s2_prop_mode": prop_mode,
        })

# ── Отображение ───────────────────────────────────────────────────────────────

import pandas as pd

# — Этап 1 —
if "s1_rows" in st.session_state:
    _ss = st.session_state
    _c1, _c2 = _ss["s1_col1"], _ss["s1_col2"]
    _dm, _dM = _ss.get("s1_d_min"), _ss.get("s1_d_max")

    st.subheader("Этап 1 — PCA sweep (ratio, без LP)")
    st.caption(f"amp_cos α={blend_alpha}  ·  K = 3·(m+1)+{xi_add}  ·  origin −{origin_offset}б  ·  bars={bars}")

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

# — Этап 2: DMD —
if "s2_results" in st.session_state:
    _ss        = st.session_state
    _s2_res    = _ss["s2_results"]
    _s2_orig   = _ss["s2_origin"]
    _s2_n      = _ss["s2_n"]
    _s2_times  = _ss["s2_times"]
    _s2_hor    = _ss["s2_horizon"]
    _s2_dvs    = _ss["s2_d_values"]
    _s2_pmode  = _ss["s2_prop_mode"]
    _dm, _dM   = _ss.get("s1_d_min"), _ss.get("s1_d_max")

    n_valid = len(_s2_res)
    n_total = len(_s2_dvs)
    st.subheader(f"Этап 2 — Прогноз DMD  (d∈[{_dm},{_dM}],  {n_valid}/{n_total} валидных)")

    _a_lt  = _ss["s2_a_lt"]
    _b_lt  = _ss["s2_b_lt"]
    _n_eff = _ss["s2_n_eff"]

    def _recon(fc_preds: np.ndarray) -> np.ndarray:
        return np.array([
            (fc_preds[j] + j * delta_ratio) * np.exp(_a_lt + _b_lt * (_n_eff + j))
            for j in range(len(fc_preds))
        ])

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
        st.warning("Нет валидных прогнозов.")
        st.stop()

    # Реконструкция цены
    _fc_iter_prices   = [_recon(r["fc_iter"])   for r in _s2_res if r["fc_iter"]   is not None]
    _fc_matpow_prices = [_recon(r["fc_matpow"]) for r in _s2_res if r["fc_matpow"] is not None]

    avg_iter   = np.array(_fc_iter_prices).mean(axis=0)   if _fc_iter_prices   else None
    avg_matpow = np.array(_fc_matpow_prices).mean(axis=0) if _fc_matpow_prices else None

    fc_idx = np.arange(_s2_orig + 1, _s2_orig + 1 + _s2_hor)

    # Временны́е метки
    try:
        from datetime import timedelta as _td
        _base = pd.Timestamp(_s2_times[-1])
        _PERIOD = {
            "1m": _td(minutes=1), "10m": _td(minutes=10),
            "1h": _td(hours=1),   "1w":  _td(weeks=1),
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

    # y-диапазон
    _y_lo = float(np.min(low[show_from : _s2_orig + 1]))
    _y_hi = float(np.max(high[show_from : _s2_orig + 1]))
    if _s2_orig < _s2_n - 1:
        _ae   = min(_s2_n, _s2_orig + _s2_hor + 1)
        _y_lo = min(_y_lo, float(close[_s2_orig : _ae].min()))
        _y_hi = max(_y_hi, float(close[_s2_orig : _ae].max()))
    for _fp in _fc_iter_prices + _fc_matpow_prices:
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
    ))

    if _s2_orig < _s2_n - 1:
        actual_end = min(_s2_n, _s2_orig + _s2_hor + 1)
        fig.add_trace(go.Scatter(
            x=_s2_times[_s2_orig:actual_end],
            y=close[_s2_orig:actual_end],
            mode="lines", name="actual",
            line=dict(color="#ffffff", width=1.5, dash="dot"),
        ))

    # Тонкие линии по d
    _fc_x_base = [_s2_times[_s2_orig]] + [_time_at(int(i)) for i in fc_idx]
    for r in _s2_res:
        if r["fc_iter"] is not None:
            fp = _recon(r["fc_iter"])
            fig.add_trace(go.Scatter(
                x=_fc_x_base, y=np.concatenate([[close[_s2_orig]], fp]),
                mode="lines", name=f"iter d={r['d']}",
                line=dict(width=0.8, color="rgba(100,180,255,0.20)"),
                showlegend=False,
            ))
        if r["fc_matpow"] is not None:
            fp = _recon(r["fc_matpow"])
            fig.add_trace(go.Scatter(
                x=_fc_x_base, y=np.concatenate([[close[_s2_orig]], fp]),
                mode="lines", name=f"matpow d={r['d']}",
                line=dict(width=0.8, color="rgba(255,160,80,0.20)"),
                showlegend=False,
            ))

    # Средние
    if avg_iter is not None:
        n_iter_valid = len(_fc_iter_prices)
        fig.add_trace(go.Scatter(
            x=_fc_x_base, y=np.concatenate([[close[_s2_orig]], avg_iter]),
            mode="lines+markers", name=f"Итерация (среднее, {n_iter_valid} d)",
            line=dict(color="#64b5f6", width=2.5),
            marker=dict(size=3),
        ))
    if avg_matpow is not None:
        n_mp_valid = len(_fc_matpow_prices)
        fig.add_trace(go.Scatter(
            x=_fc_x_base, y=np.concatenate([[close[_s2_orig]], avg_matpow]),
            mode="lines+markers", name=f"Матр. степень (среднее, {n_mp_valid} d)",
            line=dict(color="#ffb74d", width=2.5),
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
        title=(f"{ticker} {interval}  ·  DMD d∈[{_dm},{_dM}]  ·  "
               f"α_ridge={ridge_alpha}  ·  clip_λ={clip_eigenvalues}  ·  H={_s2_hor}"),
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.18),
    )

    if fc_idx[-1] >= _s2_n:
        fig.update_layout(xaxis_range=[_s2_times[show_from], _time_at(int(fc_idx[-1]))])

    st.plotly_chart(fig, use_container_width=True)

    # ── Таблица параметров ──────────────────────────────────────────────────
    st.subheader("Параметры этапа 2")
    _df2_rows = []
    for r in _s2_res:
        row = {"d": r["d"], "m": r["m"], "K": r["K"]}
        if r["fc_iter"] is not None:
            fp = _recon(r["fc_iter"])
            row["iter fc[0]"]  = round(float(fp[0]),  4)
            row["iter fc[-1]"] = round(float(fp[-1]), 4)
        else:
            row["iter fc[0]"]  = None
            row["iter fc[-1]"] = None
        if r["fc_matpow"] is not None:
            fp = _recon(r["fc_matpow"])
            row["mp fc[0]"]  = round(float(fp[0]),  4)
            row["mp fc[-1]"] = round(float(fp[-1]), 4)
        else:
            row["mp fc[0]"]  = None
            row["mp fc[-1]"] = None
        _df2_rows.append(row)
    st.dataframe(pd.DataFrame(_df2_rows), use_container_width=True, hide_index=True, height=260)

    # ── Диагностика DMD ────────────────────────────────────────────────────
    st.subheader("Диагностика DMD")

    # Собираем диагностику по d (h=0 для всех режимов)
    _diag_rows = []
    for r in _s2_res:
        row = {"d": r["d"], "m": r["m"]}
        if r["diags_iter"] and len(r["diags_iter"]) > 0:
            d0 = r["diags_iter"][0]
            row["iter λ_max_raw"]   = round(d0["lambda_max_raw"],   3)
            row["iter λ_max_final"] = round(d0["lambda_max_final"], 3)
            row["iter n_clip"]      = d0["n_clipped"]
            row["iter cond(Z'WZ)"]  = f"{d0['cond_ZtWZ']:.1e}"
            row["iter K_actual"]    = d0["K"]
            row["iter d_eff"]       = d0["d_eff"]
        if r["diags_matpow"] and len(r["diags_matpow"]) > 0:
            d0 = r["diags_matpow"][0]
            row["mp λ_max_raw"]   = round(d0["lambda_max_raw"],   3)
            row["mp λ_max_final"] = round(d0["lambda_max_final"], 3)
            row["mp n_clip"]      = d0["n_clipped"]
            row["mp cond(Z'WZ)"]  = f"{d0['cond_ZtWZ']:.1e}"
        _diag_rows.append(row)

    if _diag_rows:
        st.dataframe(pd.DataFrame(_diag_rows), use_container_width=True, hide_index=True, height=260)

    # Метрики (средние по d)
    _lmr_i = [r["diags_iter"][0]["lambda_max_raw"]   for r in _s2_res if r["diags_iter"]]
    _lmf_i = [r["diags_iter"][0]["lambda_max_final"] for r in _s2_res if r["diags_iter"]]
    _nc_i  = [r["diags_iter"][0]["n_clipped"]        for r in _s2_res if r["diags_iter"]]
    _lmr_m = [r["diags_matpow"][0]["lambda_max_raw"]   for r in _s2_res if r["diags_matpow"]]
    _lmf_m = [r["diags_matpow"][0]["lambda_max_final"] for r in _s2_res if r["diags_matpow"]]
    _nc_m  = [r["diags_matpow"][0]["n_clipped"]        for r in _s2_res if r["diags_matpow"]]

    _metric_cols = st.columns(4)
    if _lmr_i:
        _metric_cols[0].metric("mean λ_max_raw (iter)",   f"{np.mean(_lmr_i):.3f}")
        _metric_cols[1].metric("mean λ_max_final (iter)", f"{np.mean(_lmf_i):.3f}")
        _metric_cols[2].metric("mean n_clip (iter)",      f"{np.mean(_nc_i):.1f}")
    if _lmr_m:
        _metric_cols[3].metric("mean λ_max_raw (mp)",     f"{np.mean(_lmr_m):.3f}")

    # График |λ_max|(h) по шагам (итеративный режим, первый валидный d)
    _iter_with_diags = [r for r in _s2_res if r["diags_iter"] and len(r["diags_iter"]) > 1]
    if _iter_with_diags:
        r0 = _iter_with_diags[0]
        _lm_raw   = [d["lambda_max_raw"]   for d in r0["diags_iter"]]
        _lm_final = [d["lambda_max_final"] for d in r0["diags_iter"]]
        _nc_h     = [d["n_clipped"]        for d in r0["diags_iter"]]

        fig_lm = go.Figure()
        fig_lm.add_trace(go.Scatter(
            x=list(range(len(_lm_raw))), y=_lm_raw,
            mode="lines", name="|λ_max| до клиппинга",
            line=dict(color="#ef5350", width=1.5),
        ))
        fig_lm.add_trace(go.Scatter(
            x=list(range(len(_lm_final))), y=_lm_final,
            mode="lines", name="|λ_max| после клиппинга",
            line=dict(color="#66bb6a", width=1.5),
        ))
        fig_lm.add_hline(y=1.0, line_dash="dash", line_color="#ffffff",
                         annotation_text="единичный круг")
        fig_lm.update_layout(
            title=f"|λ_max(A_h)| по шагам  (итерация, d={r0['d']})",
            xaxis_title="шаг h", yaxis_title="|λ_max|",
            height=300, template="plotly_dark",
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig_lm, use_container_width=True)

        # Число клиппингов по шагам
        fig_nc = go.Figure()
        fig_nc.add_trace(go.Bar(
            x=list(range(len(_nc_h))), y=_nc_h,
            name="n_clipped(h)",
            marker_color="#ffd600",
        ))
        fig_nc.update_layout(
            title=f"Число клиппингов |λ|>1 по шагам  (итерация, d={r0['d']})",
            xaxis_title="шаг h", yaxis_title="n_clipped",
            height=200, template="plotly_dark",
        )
        st.plotly_chart(fig_nc, use_container_width=True)

    # Спектр λ(A) на комплексной плоскости (матричная степень, первый d, h=0)
    _mp_with_diags = [r for r in _s2_res if r["diags_matpow"] and len(r["diags_matpow"]) > 0]
    if _mp_with_diags:
        r0    = _mp_with_diags[0]
        d0    = r0["diags_matpow"][0]
        ev_r  = d0["eigvals_raw"]
        ev_f  = d0["eigvals_final"]

        fig_ev = go.Figure()

        # Единичный круг
        _th = np.linspace(0, 2 * np.pi, 200)
        fig_ev.add_trace(go.Scatter(
            x=np.cos(_th), y=np.sin(_th),
            mode="lines", name="|λ|=1",
            line=dict(color="rgba(255,255,255,0.3)", dash="dash"),
        ))

        fig_ev.add_trace(go.Scatter(
            x=ev_r.real, y=ev_r.imag,
            mode="markers", name="λ (до клиппинга)",
            marker=dict(color="#ef5350", size=8, symbol="circle-open"),
        ))
        if d0["n_clipped"] > 0:
            fig_ev.add_trace(go.Scatter(
                x=ev_f.real, y=ev_f.imag,
                mode="markers", name="λ (после клиппинга)",
                marker=dict(color="#66bb6a", size=6),
            ))

        fig_ev.update_layout(
            title=(f"Спектр λ(A)  (матр. степень, d={r0['d']}, m={r0['m']})  "
                   f"·  K={d0['K']}  ·  d_eff={d0['d_eff']}  "
                   f"·  cond={d0['cond_ZtWZ']:.1e}"),
            xaxis_title="Re(λ)", yaxis_title="Im(λ)",
            height=400, template="plotly_dark",
            xaxis=dict(scaleanchor="y", scaleratio=1),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_ev, use_container_width=True)
