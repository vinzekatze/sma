"""
app3_ratio: app3 в пространстве ratio (вместо dratio/att).

Отличие от app3: att-фильтр применяется к ratio напрямую и возвращает
LP-очищенный ratio (не diff). LWR предсказывает ratio[t+1] из [ratio[t-p+1]..ratio[t]].
Реконструкция цены: ratio_hat * logtrend (без cumsum).

Мотивация: dratio/att центрирован около нуля → соседи отбираются по динамике.
ratio центрирован около 1, что создаёт «ложных соседей по уровню».
Данный файл позволяет проверить этот эффект эмпирически.

Run (из prototype/):  streamlit run forcaster/ui/app3_ratio.py
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
from scipy.signal import butter, sosfilt, sosfiltfilt, lfilter

from forcaster.data.moex import download_candles, save_candles, INTERVALS

_ROOT    = Path(__file__).parent.parent.parent
DATA_DIR = _ROOT / "data" / "candles"

FILTER_ORDER = 4
_AR_ORDER    = 20
_AR_PAD      = 40
_BAR_DELTA: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1), "10m": pd.Timedelta(minutes=10),
    "1h": pd.Timedelta(hours=1),   "1d":  pd.Timedelta(days=1),
    "1w": pd.Timedelta(weeks=1),   "1mo": pd.Timedelta(days=30),
}

# ── алгоритмические функции ───────────────────────────────────────────────────

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


def _ar_extend_forward(x: np.ndarray, order: int, n_extend: int) -> np.ndarray:
    """Продлить ряд x на n_extend баров с помощью AR(order)."""
    if len(x) < order + 1:
        return np.concatenate([x, np.zeros(n_extend)])
    n = len(x); rows = min(n - order, 500); start = n - order - rows
    X = np.column_stack([x[start + i: start + i + rows] for i in range(order)])
    y = x[start + order: start + order + rows]
    a, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    buf = list(x[-order:])
    ext = []
    for _ in range(n_extend):
        nxt = float(np.dot(a, buf[-order:][::-1]))
        ext.append(nxt); buf.append(nxt)
    return np.concatenate([x, ext])


def _lp_att_signal(dratio: np.ndarray, wn: float) -> np.ndarray:
    sos = butter(FILTER_ORDER, wn, btype="low", output="sos")
    return sosfilt(sos, dratio)


def _lp_att_ar_filtfilt(dratio: np.ndarray, wn: float) -> np.ndarray:
    """
    Pipeline B: AR-extended filtfilt. Нет фазовой задержки (τ≈0).
    dratio продлевается на AR_PAD баров AR(20), затем sosfiltfilt;
    хвост отбрасывается — краевой артефакт filtfilt уходит в AR-паддинг.
    Скр.69/71: +4% price-MAPE vs causal LP.
    """
    sos = butter(FILTER_ORDER, wn, btype="low", output="sos")
    ext = _ar_extend_forward(dratio, _AR_ORDER, _AR_PAD)
    return sosfiltfilt(sos, ext)[:len(dratio)]


def _ssa_att_signal(ratio: np.ndarray, W: int, L: int, k: int) -> np.ndarray:
    """
    Rolling causal SSA на ratio → LP-очищенный ratio (без diff).
    На каждом t >= W-1: SSA окна ratio[t-W+1..t], берём последнее реконструированное значение.
    Нет фазовой задержки. W=128 L=32 k=2 — рекомендуемый старт.
    """
    N   = len(ratio)
    out = ratio.copy().astype(np.float64)
    K_m = W - L + 1
    if K_m < 2:
        return ratio.copy()
    rows = np.arange(K_m)[:, None] + np.arange(L)[None, :]
    for t in range(W - 1, N):
        w  = ratio[t - W + 1:t + 1]
        X  = w[rows]
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        nk = min(k, len(s))
        Xr = (U[:, :nk] * s[:nk]) @ Vt[:nk, :]
        recon = np.zeros(W); cnt = np.zeros(W, dtype=np.int32)
        for i in range(K_m):
            recon[i:i + L] += Xr[i]; cnt[i:i + L] += 1
        out[t] = (recon / np.maximum(cnt, 1))[-1]
    return out   # ratio-space: LP-очищенный ratio, без diff


@st.cache_data(show_spinner=False)
def _ssa_att_full_cached(close_bytes: bytes, W: int, L: int, k: int) -> np.ndarray:
    """Rolling SSA на ratio → diff — кэшируется, срезается по origin_k позже."""
    close_arr = np.frombuffer(close_bytes, dtype=np.float64).copy()
    lt  = _logtrend_causal(close_arr)
    rat = close_arr / np.maximum(lt, 1e-10)
    return _ssa_att_signal(rat, W, L, k)


def _lp_proj_signal(ratio: np.ndarray, m: int, d_proj: int,
                    k: int, n_iter: int) -> np.ndarray:
    """
    Local Projective noise reduction (Grassberger-Hegger) на ratio → ratio.
    Для каждой точки вложения: PCA на k соседях → проекция на d_proj-мерное
    касательное подпространство аттрактора. Итерируется n_iter раз.
    Нет фазовой задержки (офлайн). Скр.75: лучший конфиг m=9 d=3 k=30 n=3.
    Возвращает LP-очищенный ratio (без diff, ratio-space).
    """
    from scipy.spatial import KDTree as _KDTree
    s   = ratio.copy().astype(np.float64)
    N   = len(s)
    k_eff = min(k, N - m)
    d_eff = min(d_proj, m - 1)
    for _ in range(n_iter):
        n_pts = N - m + 1
        if n_pts < k_eff + 1:
            break
        rows = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
        X    = s[rows]
        tree = _KDTree(X)
        _, inds = tree.query(X, k=k_eff + 1)
        X_proj = np.empty_like(X)
        for i in range(n_pts):
            nn       = inds[i, 1:]
            X_nn     = X[nn]
            centroid = X_nn.mean(axis=0)
            _, _, Vt = np.linalg.svd(X_nn - centroid, full_matrices=False)
            V_d      = Vt[:d_eff].T
            xc       = X[i] - centroid
            X_proj[i] = centroid + V_d @ (V_d.T @ xc)
        result = np.zeros(N); count = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            result[i:i + m] += X_proj[i]; count[i:i + m] += 1
        s = result / np.maximum(count, 1)
    return s   # ratio-space: LP-очищенный ratio, без diff


@st.cache_data(show_spinner=False)
def _lp_proj_att_full_cached(close_bytes: bytes,
                              m: int, d_proj: int, k: int, n_iter: int) -> np.ndarray:
    """Local Projective на ratio → diff — кэшируется, срезается по origin_k позже."""
    close_arr = np.frombuffer(close_bytes, dtype=np.float64).copy()
    lt  = _logtrend_causal(close_arr)
    rat = close_arr / np.maximum(lt, 1e-10)
    return _lp_proj_signal(rat, m, d_proj, k, n_iter)


def _kf2d_att_signal(dratio: np.ndarray, q_factor: float) -> np.ndarray:
    """
    2D constant-velocity Kalman для извлечения медленной компоненты dratio.
    state = [level, slope]; нет фазовой задержки.
    q_factor: q_slope = q_factor × σ_dratio. Больше → отзывчивее, меньше задержка, но шумнее.
    """
    sigma = float(np.std(dratio))
    if sigma < 1e-12:
        return np.zeros(len(dratio))
    q_slope = q_factor * sigma
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Q_mat = np.array([[0.0, 0.0], [0.0, q_slope ** 2]])
    R_var = sigma ** 2
    try:
        from scipy.linalg import solve_discrete_are
        P_ss = solve_discrete_are(F.T, H.reshape(-1, 1), Q_mat, np.array([[R_var]]))
        S_ss = float(H @ P_ss @ H) + R_var
        K_ss = (P_ss @ H) / S_ss
    except Exception:
        P = np.eye(2) * R_var
        for _ in range(500):
            P_p = F @ P @ F.T + Q_mat
            S = float(H @ P_p @ H) + R_var
            K_ss = (P_p @ H) / S
            P = (np.eye(2) - np.outer(K_ss, H)) @ P_p
    A_ss = (np.eye(2) - np.outer(K_ss, H)) @ F
    n = len(dratio)
    x = np.zeros(2)
    att = np.zeros(n)
    for k in range(n):
        x = A_ss @ x + K_ss * dratio[k]
        att[k] = x[0]
    return att


@st.cache_data(show_spinner=False)
def _kf2d_att_full_cached(close_bytes: bytes, q_factor: float) -> np.ndarray:
    """KF2D на ratio напрямую — кэшируется, срезается по origin_k позже.
    state=[level,slope] отслеживает ratio (level≈ratio, slope≈dratio).
    """
    close_arr = np.frombuffer(close_bytes, dtype=np.float64).copy()
    lt  = _logtrend_causal(close_arr)
    rat = close_arr / np.maximum(lt, 1e-10)
    return _kf2d_att_signal(rat, q_factor)   # ratio-space, без diff


def _compute_lp_tau(wn: float) -> int:
    """Групповая задержка causal Butterworth LP (FILTER_ORDER) при ω=0, в барах."""
    from scipy.signal import sos2tf, group_delay as _gd
    sos = butter(FILTER_ORDER, wn, btype="low", output="sos")
    b, a = sos2tf(sos)
    _, gd = _gd((b, a), w=1, whole=False)
    return int(round(float(gd[0])))


def _reconstruct_lp_prices(
    att: np.ndarray, ratio: np.ndarray, logtrend: np.ndarray,
    anchor_k: int, end_k: int,
) -> np.ndarray:
    # ratio-space: att[t] ≈ LP(ratio[t]), цена = att[t] * logtrend[t]
    return att[anchor_k: end_k + 1] * logtrend[anchor_k: end_k + 1]


def _select_neighbors(
    X_full: np.ndarray, vec_search: np.ndarray, xi: int,
    p_mid: int | None, vec_search_full: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Возвращает (индексы финальных соседей, bandwidth)."""
    dists = np.linalg.norm(X_full - vec_search, axis=1)
    xi    = min(xi, len(X_full))
    nn    = np.argpartition(dists, xi - 1)[:xi]

    if p_mid is not None and p_mid > 0:
        p_mid_eff = min(p_mid, X_full.shape[1])
        xi_2      = max(3 * (p_mid_eff + 1), 3)
        xi_2      = min(xi_2, len(nn))
        vec_mid   = vec_search_full[-p_mid_eff:]
        dists2    = np.linalg.norm(X_full[nn, -p_mid_eff:] - vec_mid, axis=1)
        nn2       = np.argpartition(dists2, xi_2 - 1)[:xi_2]
        nn        = nn[nn2]

    h_bw = max(float(np.linalg.norm(X_full[nn] - vec_search, axis=1).max()), 1e-10)
    return nn, h_bw


def _lwr_approx(X_nn: np.ndarray, y_nn: np.ndarray,
                vec_fit: np.ndarray, h_bw: float) -> float:
    """Локальная линейная регрессия с гауссовыми весами."""
    p_eff = min(X_nn.shape[1], len(vec_fit))
    Xf    = X_nn[:, -p_eff:]
    vf    = vec_fit[-p_eff:]
    w     = np.exp(-0.5 * (np.linalg.norm(Xf - vf, axis=1) / h_bw) ** 2)
    A     = np.hstack([np.ones((len(Xf), 1)), Xf])
    sw    = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vf @ c[1:])


def _lp_corr_point(v_pred: np.ndarray, X_lib_m: np.ndarray,
                   k: int = 30, d: int = 3, n_iter: int = 1) -> float:
    """Проецирует m-мерный вектор v_pred на локальное d-мерное подпространство аттрактора.
    v_pred[-1] — самое свежее значение (att_hat). Возвращает скорректированное значение.
    n_iter итераций: на каждой — новый поиск соседей из обновлённой точки.
    """
    v = v_pred.copy().astype(float)
    for _ in range(n_iter):
        dists  = np.linalg.norm(X_lib_m - v, axis=1)
        k_eff  = min(k, len(X_lib_m) - 1)
        idx    = np.argpartition(dists, k_eff)[:k_eff]
        X_nn   = X_lib_m[idx]
        center = X_nn.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_nn - center, full_matrices=False)
        Vd     = Vt[:min(d, len(Vt))].T
        v_c    = v - center
        v      = center + Vd @ (Vd.T @ v_c)
    return float(v[-1])


def _rbf_approx(X_nn: np.ndarray, y_nn: np.ndarray,
                vec_fit: np.ndarray, h_bw: float) -> float:
    """RBF: ŷ = Σ wᵢ·exp(−||x−xᵢ||²/2σ²), σ=h_bw."""
    p_eff  = min(X_nn.shape[1], len(vec_fit))
    Xf     = X_nn[:, -p_eff:]
    vf     = vec_fit[-p_eff:]
    xi     = len(Xf)
    diff   = Xf[:, None, :] - Xf[None, :, :]
    Phi    = np.exp(-0.5 * np.sum(diff ** 2, axis=2) / h_bw ** 2)
    lam    = 1e-6 * (np.trace(Phi) / xi + 1e-12)
    w, _, _, _ = np.linalg.lstsq(Phi + lam * np.eye(xi), y_nn, rcond=None)
    phi_q  = np.exp(-0.5 * np.sum((Xf - vf) ** 2, axis=1) / h_bw ** 2)
    return float(phi_q @ w)


def _step(
    X_full: np.ndarray, y_full: np.ndarray,
    vec_search: np.ndarray, vec_fit: np.ndarray,
    xi: int, p_mid: int | None, use_rbf: bool,
) -> tuple[float, float]:
    nn, h_bw = _select_neighbors(X_full, vec_search, xi, p_mid, vec_search)
    X_nn = X_full[nn]; y_nn = y_full[nn]
    fn   = _rbf_approx if use_rbf else _lwr_approx
    return fn(X_nn, y_nn, vec_fit, h_bw), h_bw


def _forecast_direct(
    att: np.ndarray, p_search: int, p_fit: int, xi: int, horizon: int,
    p_mid: int | None, use_rbf: bool,
) -> np.ndarray:
    n          = len(att)
    p_s        = min(p_search, n - 2)
    vec_search = att[-p_s:].copy()
    vec_fit    = att[-p_fit:].copy() if p_fit <= n else att.copy()
    out        = np.empty(horizon)
    for h in range(1, horizon + 1):
        n_s = n - p_s - h
        if n_s < max(xi, 3):
            out[h - 1:] = 0.0; break
        X = np.array([att[i: i + p_s] for i in range(n_s)])
        y = att[p_s + h - 1: p_s + h - 1 + n_s]
        val, _ = _step(X, y, vec_search, vec_fit, xi, p_mid, use_rbf)
        out[h - 1] = val
    return out


def _forecast_iterative(
    att: np.ndarray, p_search: int, p_fit: int, xi: int, horizon: int,
    p_mid: int | None, use_rbf: bool,
) -> np.ndarray:
    n   = len(att)
    p_s = min(p_search, n - 2)
    X   = np.array([att[i: i + p_s] for i in range(n - p_s)])
    y   = att[p_s:]
    if len(X) < max(xi, 3):
        return np.zeros(horizon)

    vec_search = att[-p_s:].copy()
    vec_fit    = att[-p_fit:].copy() if p_fit <= n else att.copy()

    nn0, h_bw0 = _select_neighbors(X, vec_search, xi, p_mid, vec_search)
    fn = _rbf_approx if use_rbf else _lwr_approx
    if use_rbf:
        X_nn0 = X[nn0]; y_nn0 = y[nn0]
        p_eff  = min(X_nn0.shape[1], len(vec_fit))
        Xf0    = X_nn0[:, -p_eff:]
        xi0    = len(Xf0)
        diff0  = Xf0[:, None, :] - Xf0[None, :, :]
        Phi0   = np.exp(-0.5 * np.sum(diff0 ** 2, axis=2) / h_bw0 ** 2)
        lam0   = 1e-6 * (np.trace(Phi0) / xi0 + 1e-12)
        w0, _, _, _ = np.linalg.lstsq(Phi0 + lam0 * np.eye(xi0), y_nn0, rcond=None)
        X_fixed, w_fixed, hbw_fixed = Xf0, w0, h_bw0
        def _predict_fixed(vf):
            phi = np.exp(-0.5 * np.sum((X_fixed - vf[-p_eff:]) ** 2, axis=1) / hbw_fixed ** 2)
            return float(phi @ w_fixed)
    else:
        X_nn0 = X[nn0]; y_nn0 = y[nn0]
        p_eff  = min(X_nn0.shape[1], len(vec_fit))
        Xf0    = X_nn0[:, -p_eff:]
        vf0    = vec_fit[-p_eff:]
        w0     = np.exp(-0.5 * (np.linalg.norm(Xf0 - vf0, axis=1) / h_bw0) ** 2)
        A0     = np.hstack([np.ones((len(Xf0), 1)), Xf0])
        sw0    = np.sqrt(w0)
        c0, _, _, _ = np.linalg.lstsq(sw0[:, None] * A0, sw0 * y_nn0, rcond=None)
        def _predict_fixed(vf):
            return float(c0[0] + vf[-p_eff:] @ c0[1:])

    out = np.empty(horizon)
    for h in range(horizon):
        val    = _predict_fixed(vec_fit)
        out[h] = val
        vec_search = np.roll(vec_search, -1); vec_search[-1] = val
        vec_fit    = np.roll(vec_fit,    -1); vec_fit[-1]    = val
    return out


def _forecast_iterative_refit(
    att: np.ndarray, p_search: int, p_fit: int, xi: int, horizon: int,
    p_mid: int | None, use_rbf: bool,
) -> np.ndarray:
    n   = len(att)
    p_s = min(p_search, n - 2)
    X   = np.array([att[i: i + p_s] for i in range(n - p_s)])
    y   = att[p_s:]
    if len(X) < max(xi, 3):
        return np.zeros(horizon)

    vec_search = att[-p_s:].copy()
    vec_fit    = att[-p_fit:].copy() if p_fit <= n else att.copy()
    out        = np.empty(horizon)
    for h in range(horizon):
        val, _ = _step(X, y, vec_search, vec_fit, xi, p_mid, use_rbf)
        out[h] = val
        vec_search = np.roll(vec_search, -1); vec_search[-1] = val
        vec_fit    = np.roll(vec_fit,    -1); vec_fit[-1]    = val
    return out


# ── non-uniform задержки ──────────────────────────────────────────────────────

NONUNIF_LAG_SETS: dict[str, list[int]] = {
    "Октавные geom_k9 ★": [128, 64, 32, 16, 8, 4, 2, 1, 0],
    "C2-C5 периоды k13":  [130, 103, 78, 52, 40, 26, 16, 12, 8, 5, 2, 1, 0],
}

_GEOM_BASES = {"×√2  полуоктавы": 2.0 ** 0.5, "×2  октавы": 2.0}

_SEARCH_METHODS: dict[str, str] = {
    "Стандартный L2":        "l2",
    "Нормализованный L2 ★":  "norm_l2",
    "Borda count":           "borda",
    "Адаптивный порог":      "adaptive",
    "Иерархический каскад":  "cascade",
}


def _make_geom_lags(k: int, base: float, max_lag: int) -> list[int]:
    """Геометрические лаги: max_lag / base^i для i=0..k-2, плюс 0. Дедупликация."""
    lags: set[int] = {0}
    for i in range(k - 1):
        lag = int(round(max_lag / (base ** i)))
        if lag > 0:
            lags.add(lag)
    return sorted(lags, reverse=True)


def _make_mat(att: np.ndarray, lags_desc: list[int], p_fit: int):
    """
    Строит матрицы для поиска (non-uniform лаги) и аппроксимации (consecutive p_fit).
    lags_desc: убывающий список, 0 = текущий момент.
    Возвращает X_search (m×k), X_fit (m×p_fit), y (m,).
    """
    max_lag  = lags_desc[0]
    lags_arr = np.array(lags_desc, dtype=np.int32)
    n = len(att); m = n - max_lag - 1; k = len(lags_desc)
    if m <= 0:
        return np.zeros((0, k)), np.zeros((0, p_fit)), np.zeros(0)
    t_arr    = np.arange(max_lag, n - 1, dtype=np.int32)
    X_search = np.column_stack([att[t_arr - lag] for lag in lags_arr])
    X_fit    = np.column_stack([att[t_arr - (p_fit - 1 - k_)] for k_ in range(p_fit)])
    return X_search, X_fit, att[t_arr + 1]


def _hier_schedule(xi_start: int, xi_fin: int, k: int) -> list[int]:
    """Геометрическое расписание воронки: xi_start → xi_fin за k шагов."""
    if k <= 1:
        return [xi_fin]
    r = (xi_fin / xi_start) ** (1.0 / (k - 1))
    sched = [max(xi_fin, int(round(xi_start * r ** i))) for i in range(k)]
    sched[-1] = xi_fin
    return sched


def _forecast_nonunif(
    att: np.ndarray, lags_desc: list[int], p_fit: int, xi: int, horizon: int,
    xi_start: int | None = None, xi_fin: int | None = None,
    unified: bool = True,
    search_method: str = "l2",
    adaptive_c: float = 2.0,
) -> np.ndarray:
    """LWR итеративный с пересчётом, non-uniform задержки.

    search_method: "l2" | "norm_l2" | "borda" | "adaptive" | "cascade"
    unified=True (рекомендуется): X_fit = X_search — единое пространство,
      bandwidth и веса LWR корректны. p_fit = k.
    """
    use_hier  = search_method == "cascade" and xi_start is not None and xi_fin is not None
    k         = len(lags_desc)
    sched     = _hier_schedule(xi_start, xi_fin, k) if use_hier else None
    p_fit_eff = k if unified else p_fit

    n = len(att)
    X_search, X_fit, y = _make_mat(att, lags_desc, 1 if unified else p_fit)
    xi_eff = min(xi_fin if use_hier else xi, len(X_search))
    if xi_eff < p_fit_eff + 2:
        return np.zeros(horizon)

    if search_method == "norm_l2":
        _sigma = np.maximum(np.std(X_search, axis=0), 1e-12)
        _Xn    = X_search / _sigma

    lags_arr    = np.array(lags_desc, dtype=np.int32)
    _xi_min_adp = max(p_fit_eff + 2, 3)
    buf = np.empty(n + horizon); buf[:n] = att
    out = np.empty(horizon)

    for h in range(horizon):
        t     = n + h - 1
        vec_s = buf[t - lags_arr]
        vec_f = vec_s if unified else buf[t - p_fit + 1: t + 1]

        if use_hier:
            cands = np.arange(len(X_search))
            for i in range(k):
                xi_i = min(sched[i], len(cands))
                if len(cands) <= xi_i:
                    continue
                d1  = np.abs(X_search[cands, i] - vec_s[i])
                top = np.argpartition(d1, xi_i - 1)[:xi_i]
                cands = cands[top]
            nn = cands
        elif search_method == "norm_l2":
            qn    = vec_s / _sigma
            dists = np.linalg.norm(_Xn - qn, axis=1)
            nn    = np.argpartition(dists, xi_eff - 1)[:xi_eff]
        elif search_method == "borda":
            ranks = np.zeros(len(X_search))
            for l in range(k):
                d_l    = np.abs(X_search[:, l] - vec_s[l])
                ranks += np.argsort(np.argsort(d_l)).astype(float)
            nn = np.argpartition(ranks, xi_eff - 1)[:xi_eff]
        elif search_method == "adaptive":
            cands = np.arange(len(X_search))
            for l in range(k):
                if len(cands) <= _xi_min_adp:
                    break
                d_l = np.abs(X_search[cands, l] - vec_s[l])
                med = float(np.median(d_l))
                mad = float(np.median(np.abs(d_l - med))) + 1e-12
                keep = np.where(d_l <= med + adaptive_c * mad)[0]
                if keep.size >= _xi_min_adp:
                    cands = cands[keep]
            if len(cands) > xi_eff:
                d_fin = np.linalg.norm(X_search[cands] - vec_s, axis=1)
                cands = cands[np.argpartition(d_fin, xi_eff - 1)[:xi_eff]]
            nn = cands
        else:  # "l2"
            dists = np.linalg.norm(X_search - vec_s, axis=1)
            nn    = np.argpartition(dists, xi_eff - 1)[:xi_eff]

        X_nn_f = X_search[nn] if unified else X_fit[nn]
        y_nn   = y[nn]; n_nn = len(nn)
        h_bw   = max(float(np.linalg.norm(X_nn_f - vec_f, axis=1).max()), 1e-10)
        w      = np.exp(-0.5 * (np.linalg.norm(X_nn_f - vec_f, axis=1) / h_bw) ** 2)
        A      = np.hstack([np.ones((n_nn, 1)), X_nn_f]); sw = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
        out[h] = float(c[0] + vec_f @ c[1:]); buf[t + 1] = out[h]
    return out


def _uniform_octave_levels(p_fit: int, p_max: int,
                           step_base: float = 2.0) -> list[int]:
    """Убывающие уровни стандартного каскада (оставлен для совместимости)."""
    if step_base >= 9999.0:
        return [p_fit]
    if step_base >= 999.0:
        return [p_max] if p_max == p_fit else [p_max, p_fit]
    levels: list[int] = []
    p = p_max
    while True:
        levels.append(p)
        p_next = int(p / step_base)
        if p_next < max(p_fit, 2):
            break
        p = p_next
    if levels[-1] != p_fit:
        levels.append(p_fit)
    return levels


def _levels_aligned(p_fit: int, p_max: int) -> list[int]:
    """P-aligned каскад ×2: уровни = p_fit × {1,2,4,...} ≤ p_max (скр.93, −12.41% rMAE).

    Гарантирует нестинг: каждый уровень кратен p_fit, поэтому соседи на грубом
    уровне гарантированно лежат в том же подпространстве, что и на тонком.
    p=9 → [36,18,9]; p=8 → [64,32,16,8]; p=16 → [64,16] если p_max=64.
    """
    levs = [p_fit]; p = p_fit
    while p * 2 <= p_max:
        p *= 2; levs.append(p)
    return list(reversed(levs))


def _cosine_dist_vecs(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - cos_sim(строки A, вектор b) ∈ [0, 2]. При нулевых нормах → 1."""
    norm_A = np.linalg.norm(A, axis=1)
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-12:
        return np.ones(len(A))
    cos = np.where(norm_A > 1e-12, (A @ b) / (norm_A * norm_b), 0.0)
    return 1.0 - np.clip(cos, -1.0, 1.0)


def _forecast_uniform_octave(
    att: np.ndarray, p_fit: int, p_max: int, horizon: int,
    xi_lwr: int = 30,
    use_rbf: bool = False,
    lwr_mode: str = "Итеративный с пересчётом",
    global_blend_pct: float = 0.0,
    acc_ang_lambda: float = 0.0,
    use_lp_corr: bool = False,
    lp_corr_m: int = 9,
    lp_corr_k: int = 30,
    lp_corr_d: int = 3,
    lp_corr_n: int = 1,
) -> np.ndarray:
    """LWR/RBF с p-aligned каскадом для равномерных задержек (скр.93).

    global_blend_pct: % ξ_lwr из каскадного пула; остаток ищется по всей истории.
    acc_ang_lambda: финальный отбор соседей по d_pos + λ·d_ang(acc).
      Скр.80: оптимальный λ=0.01, −10.8% rMAE на 8 тикерах 1d.

    ── Два разных понятия ξ ──────────────────────────────────────────────────────
    xi_lwr   — размер финального пула соседей для LWR-регрессии.
               Правило стабильности: xi_lwr ≥ 3·(p_fit+1), иначе система
               переопределена. Управляется пользователем.

    xi_search — размер пула на КАЖДОМ уровне каскада при поиске кандидатов.
               Сейчас = xi_lwr (решение из удобства, не тестировалось отдельно).
               Теоретически: на верхних уровнях (p_lvl >> p_fit) можно брать
               больший пул; меньший пул грубее фильтрует, но быстрее.
               TODO: протестировать xi_search > xi_lwr на верхних уровнях.
    ─────────────────────────────────────────────────────────────────────────────
    """
    levels  = _levels_aligned(p_fit, p_max)
    p_top   = levels[0]

    # xi_search = xi_lwr: каскад использует тот же размер пула, что и LWR.
    # Это упрощение — см. docstring выше. Изменять только xi_search, не xi_lwr,
    # если захочется исследовать другой размер пула на промежуточных уровнях.
    xi_search = xi_lwr

    n       = len(att)
    m       = n - p_top - 1
    if m < 3:
        return np.zeros(horizon)

    t_arr  = np.arange(p_top, n - 1)
    X_full = np.column_stack([att[t_arr - (p_top - 1 - j)] for j in range(p_top)])
    y_base = att[t_arr + 1]

    if len(X_full) < xi_lwr:
        return np.zeros(horizon)

    # LP-коррекция: библиотека m-мерных вложений (последние lp_corr_m столбцов X_full)
    X_lib_lp: np.ndarray | None = None
    if use_lp_corr and lwr_mode == "Итеративный с пересчётом":
        _m = min(lp_corr_m, X_full.shape[1])
        X_lib_lp = X_full[:, -_m:].copy()

    # acc_ang: матрица ускорений библиотеки (нужна только если λ > 0)
    if acc_ang_lambda > 0.0:
        acc_hist = np.zeros(n)
        if n >= 3:
            acc_hist[2:] = att[2:] - 2 * att[1:-1] + att[:-2]
        X_acc        = np.column_stack([acc_hist[t_arr - (p_fit - 1 - j)] for j in range(p_fit)])
        vec_acc_base = acc_hist[-p_fit:]
    else:
        X_acc = None; vec_acc_base = None; acc_hist = None

    fn = _rbf_approx if use_rbf else _lwr_approx

    def _apply_global_blend(cands_cas: np.ndarray, query_pfit: np.ndarray) -> np.ndarray:
        """Часть финального пула xi_lwr берётся из каскада, остаток — из всей истории.
        Итоговый размер пула = xi_lwr (требование LWR)."""
        if global_blend_pct <= 0.0:
            return cands_cas
        n_cas  = max(p_fit + 2, round(xi_lwr * global_blend_pct / 100.0))
        n_cas  = min(n_cas, len(cands_cas))
        n_glob = xi_lwr - n_cas
        if n_glob <= 0:
            return cands_cas
        if len(cands_cas) > n_cas:
            dc      = np.linalg.norm(X_full[cands_cas, -p_fit:] - query_pfit, axis=1)
            top_cas = cands_cas[np.argpartition(dc, n_cas - 1)[:n_cas]]
        else:
            top_cas = cands_cas
        dg       = np.linalg.norm(X_full[:, -p_fit:] - query_pfit, axis=1)
        n_g_eff  = min(n_glob, len(X_full))
        if n_g_eff < 1:
            return top_cas
        top_glob = np.argpartition(dg, n_g_eff - 1)[:n_g_eff]
        return np.unique(np.concatenate([top_cas, top_glob]))

    def _cascade(vec_full: np.ndarray) -> np.ndarray:
        """P-aligned каскад: на каждом уровне отбирает xi_search кандидатов,
        затем расширяет окно сдвигами ±radius для перехода к следующему уровню.
        Финальный уровень отбирает xi_lwr соседей для LWR."""
        cands = np.arange(len(X_full))
        for k, p_lvl in enumerate(levels):
            is_last = (k == len(levels) - 1)
            # xi_search используется на промежуточных уровнях каскада
            xi_here = xi_lwr if is_last else xi_search
            xi_clip = min(xi_here, len(cands))
            if len(cands) > xi_clip:
                dists = np.linalg.norm(
                    X_full[cands, -p_lvl:] - vec_full[-p_lvl:], axis=1)
                cands = cands[np.argpartition(dists, xi_clip - 1)[:xi_clip]]
            if not is_last:
                p_next  = levels[k + 1]
                radius  = p_lvl - p_next
                offsets = np.arange(radius + 1)
                expanded = cands[:, None] - offsets[None, :]
                expanded = np.clip(expanded, 0, len(X_full) - 1)
                cands = np.unique(expanded)
        return cands

    def _cascade_open(vec_full: np.ndarray) -> np.ndarray:
        """Каскад без финальной фильтрации — возвращает расширенный пул для acc_ang.
        На промежуточных уровнях использует xi_search; последний уровень пропускает
        все накопленные кандидаты (acc_ang_select сделает отбор до xi_lwr)."""
        cands = np.arange(len(X_full))
        for k, p_lvl in enumerate(levels):
            is_last = (k == len(levels) - 1)
            if not is_last:
                xi_clip = min(xi_search, len(cands))
                if len(cands) > xi_clip:
                    dists = np.linalg.norm(
                        X_full[cands, -p_lvl:] - vec_full[-p_lvl:], axis=1)
                    cands = cands[np.argpartition(dists, xi_clip - 1)[:xi_clip]]
                p_next   = levels[k + 1]
                radius   = p_lvl - p_next
                offsets  = np.arange(radius + 1)
                expanded = cands[:, None] - offsets[None, :]
                cands    = np.unique(np.clip(expanded, 0, len(X_full) - 1))
            # последний уровень: пул передаётся в _acc_ang_select без обрезки
        return cands

    def _acc_ang_select(cands_pool: np.ndarray, vec_full: np.ndarray,
                        vec_acc: np.ndarray) -> np.ndarray:
        """Финальный отбор ровно xi_lwr соседей по d_pos + λ·d_ang(acc).
        Именно здесь размер пула приводится к xi_lwr — требованию LWR."""
        if len(cands_pool) <= xi_lwr:
            return cands_pool
        d_pos = np.linalg.norm(X_full[cands_pool, -p_fit:] - vec_full[-p_fit:], axis=1)
        d_acc = _cosine_dist_vecs(X_acc[cands_pool], vec_acc)
        d_comb = d_pos + acc_ang_lambda * d_acc
        sel = np.argpartition(d_comb, xi_lwr - 1)[:xi_lwr]
        return cands_pool[sel]

    def _get_cands(vec_full: np.ndarray, vec_acc: np.ndarray) -> np.ndarray:
        """Каскад + опциональный acc_ang финальный отбор."""
        if acc_ang_lambda > 0.0:
            pool  = _cascade_open(vec_full)
            cands = _acc_ang_select(pool, vec_full, vec_acc)
        else:
            cands = _cascade(vec_full)
        return _apply_global_blend(cands, vec_full[-p_fit:])

    out = np.zeros(horizon)

    if lwr_mode == "Прямой":
        vec_full0 = att[-p_top:].copy()
        _va_base  = vec_acc_base if vec_acc_base is not None else np.zeros(p_fit)
        cands0    = _get_cands(vec_full0, _va_base)
        vec_f     = vec_full0[-p_fit:]
        for h in range(horizon):
            valid = cands0[cands0 < (m - h)]
            if len(valid) < p_fit + 2:
                break
            X_h  = X_full[valid, -p_fit:]
            y_h  = att[t_arr[valid] + h + 1]
            h_bw = max(float(np.linalg.norm(X_h - vec_f, axis=1).max()), 1e-10)
            out[h] = fn(X_h, y_h, vec_f, h_bw)
        return out

    if lwr_mode == "Итеративный фиксированный":
        vec_full0 = att[-p_top:].copy()
        _va_base  = vec_acc_base if vec_acc_base is not None else np.zeros(p_fit)
        cands0    = _get_cands(vec_full0, _va_base)
        X_nn      = X_full[cands0, -p_fit:]
        y_nn      = y_base[cands0]
        vec_f     = vec_full0[-p_fit:].copy()
        if len(cands0) < p_fit + 2:
            return out
        h_bw = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)
        for h in range(horizon):
            val    = fn(X_nn, y_nn, vec_f, h_bw)
            out[h] = val
            vec_f  = np.roll(vec_f, -1); vec_f[-1] = val
        return out

    # «Итеративный с пересчётом»
    buf = np.empty(n + horizon); buf[:n] = att
    # acc_buf: для acc_ang в итеративном режиме нужно ускорение прогнозируемого сигнала
    if acc_ang_lambda > 0.0:
        acc_buf = np.zeros(n + horizon)
        if n >= 3:
            acc_buf[2:n] = att[2:] - 2 * att[1:-1] + att[:-2]
    for h in range(horizon):
        t = n + h - 1
        if t - p_top + 1 < 0:
            buf[t + 1] = 0.0; continue
        vec_full = buf[t - p_top + 1: t + 1]
        if acc_ang_lambda > 0.0:
            if t >= 2:
                acc_buf[t] = buf[t] - 2 * buf[t - 1] + buf[t - 2]
            t_start    = max(0, t - p_fit + 1)
            vec_acc_q  = acc_buf[t_start: t + 1]
            if len(vec_acc_q) < p_fit:
                vec_acc_q = np.pad(vec_acc_q, (p_fit - len(vec_acc_q), 0))
            cands = _get_cands(vec_full, vec_acc_q)
        else:
            cands = _get_cands(vec_full, np.zeros(p_fit))
        X_nn     = X_full[cands, -p_fit:]
        y_nn     = y_base[cands]
        if len(cands) < p_fit + 2:
            buf[t + 1] = 0.0; continue
        vec_f = vec_full[-p_fit:]
        h_bw  = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)
        val   = fn(X_nn, y_nn, vec_f, h_bw)
        # LP-коррекция: проецируем предсказанную точку обратно на аттрактор
        if X_lib_lp is not None:
            _m   = X_lib_lp.shape[1]
            _t0  = max(0, t - _m + 2)
            _ctx = buf[_t0: t + 1]
            if len(_ctx) < _m - 1:
                _ctx = np.pad(_ctx, (_m - 1 - len(_ctx), 0))
            v_lp = np.concatenate([_ctx[-(  _m - 1):], [val]])
            val  = _lp_corr_point(v_lp, X_lib_lp, k=lp_corr_k, d=lp_corr_d, n_iter=lp_corr_n)
        out[h] = val; buf[t + 1] = val
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="LP+LWR Forecaster [ratio]", layout="wide",
                   initial_sidebar_state="expanded")

# ── ШАГ 0: мутации параметров — ДО любых виджетов ────────────────────────────
# Кнопки ±p_fit пишут _pfit_delta в session_state и делают rerun.
# Здесь дельта применяется к ключу слайдера ДО его рендера — слайдер
# сразу показывает новое значение. _auto_fc выставляется здесь, а не в кнопке.

if "_pfit_delta" in st.session_state:
    st.session_state["p_fit"] = int(np.clip(
        st.session_state.get("p_fit", 16) + st.session_state.pop("_pfit_delta"),
        2, 200))

# ── ШАГ 1: сайдбар — собираем все параметры ──────────────────────────────────
# Кнопки ±p_fit пишут _pfit_delta → пересчёт прогноза.
# Кнопки nav пишут _nav_delta → сдвиг origin.
# Слайдеры читают из session_state, уже обновлённого на шаге 0.

with st.sidebar:
    st.title("LP + LWR / RBF [ratio]")

    ticker   = st.text_input("Тикер", value="SBER", key="ticker").upper().strip()
    interval = st.selectbox("Интервал", list(INTERVALS),
                             index=list(INTERVALS).index("1d"), key="interval")

    st.divider()
    st.subheader("Параметры прогноза")

    lag_type = st.radio(
        "Тип задержек",
        ["Равномерные", *NONUNIF_LAG_SETS.keys(), "Геом. (настр.)"],
        index=0, key="lag_type",
        help=(
            "**Октавные geom_k9 ★** — {0,1,2,4,8,16,32,64,128}, 9 задержек по октавам. "
            "Лучший результат скр.68 (MAPE=0.02343, 8 тикеров). Быстрее равномерных в ~4×.\n\n"
            "**C2-C5 периоды k13** — 13 задержек по ключевым периодам C2-C5. "
            "MAPE=0.02376, значимо лучше baseline (p=0.022).\n\n"
            "**Геом. (настр.)** — геометрические лаги с настраиваемой базой и количеством. "
            "Позволяет попробовать полуоктавы (×√2) или произвольный шаг.\n\n"
            "**Равномерные** — классический sliding window [0..p_search-1]. "
            "Поддерживает каскад и RBF."
        ),
    )
    is_nonunif = lag_type != "Равномерные"

    horizon = st.slider("Горизонт (баров)", 1, 200, 30, 1, key="horizon")

    # ── параметры поиска ──────────────────────────────────────────────────────
    use_cascade   = False
    p_mid: int | None = None
    use_hier      = False
    xi_start: int | None = None
    xi_fin:   int | None = None
    xi_max        = 300
    unified_space = True
    search_method = "l2"
    adaptive_c    = 2.0
    p_max         = 64

    if is_nonunif:
        if lag_type == "Геом. (настр.)":
            geom_base_lbl = st.selectbox("База", list(_GEOM_BASES.keys()), key="geom_base")
            geom_base = _GEOM_BASES[geom_base_lbl]
            geom_k    = st.slider("k лагов", 3, 20, 9, 1, key="geom_k",
                                  help="Количество задержек включая lag=0.")
            geom_max  = st.slider("max лаг (баров)", 16, 512, 128, 8, key="geom_max",
                                  help="Максимальная задержка. 128б ≈ период C5.")
            lags_desc = _make_geom_lags(geom_k, geom_base, geom_max)
        else:
            lags_desc = NONUNIF_LAG_SETS[lag_type]
        k_lags   = len(lags_desc)
        p_search = lags_desc[0]
        st.caption(f"k={k_lags} задержек: {{{', '.join(str(l) for l in sorted(lags_desc))}}}")

        _sm_lbl = st.selectbox(
            "Метод поиска", list(_SEARCH_METHODS.keys()), index=0,
            key="search_method_sel",
            help=(
                "**Стандартный L2** — евклидово расстояние по всем лагам.\n\n"
                "**Нормализованный L2 ★** — каждое измерение делится на σ, "
                "все шкалы уравнены (диагональный Махаланобис).\n\n"
                "**Borda count** — ранги по каждому лагу суммируются; "
                "нормировка не нужна.\n\n"
                "**Адаптивный порог** — на каждом лаге отсекаются аутлайеры "
                "по median + c·MAD; расписание не нужно.\n\n"
                "**Иерархический каскад** — грубо-к-тонкому по лагам slow→fast, "
                "с геометрическим расписанием ξ_start → ξ_fin."
            ),
        )
        search_method = _SEARCH_METHODS[_sm_lbl]
        use_hier      = search_method == "cascade"

        if use_hier:
            xi_start = st.slider("ξ старт (медленный)", 100, 5000, 2000, 50,
                                 key="xi_start",
                                 help="Число кандидатов на первом уровне воронки.")
            xi_fin   = st.slider("ξ финал (быстрый)", 10, 500, 50, 5,
                                 key="xi_fin",
                                 help="Число финальных соседей.")
            xi = xi_fin
            st.caption("Воронка: " + " → ".join(
                str(s) for s in _hier_schedule(xi_start, xi_fin, k_lags)))
        else:
            xi = st.slider("ξ соседей", 10, 2000, 200, 10, key="xi",
                           help="Число ближайших соседей для LWR.")
            if search_method == "adaptive":
                adaptive_c = st.slider(
                    "c (MAD множитель)", 0.5, 5.0, 2.0, 0.1, key="adaptive_c",
                    help="Порог отсечения: median + c·MAD. Меньше → жёстче фильтр.",
                )
                st.caption(f"min соседей = p_fit+2 = {k_lags+2}  |  ξ — верхний предел")
            elif search_method == "norm_l2":
                st.caption("σ вычисляется по всей обучающей выборке (один раз)")
            elif search_method == "borda":
                st.caption(f"Суммирование рангов по {k_lags} измерениям")

        unified_space = st.checkbox(
            "Единое пространство (поиск = LWR)", value=True, key="unified_space",
            help="X_fit = X_search: LWR и поиск в одном пространстве. "
                 "Рекомендуется — bandwidth корректен.",
        )
        if unified_space:
            st.caption(f"p_fit = k = {k_lags}  (слайдер «p прогноз» игнорируется)")
        else:
            st.caption("⚠ Разные пространства — bandwidth может быть некорректен")
    else:
        lags_desc = []
        _P_MAX_OPTS = [16, 32, 64, 128, 256, 512, 1024]
        p_max = st.select_slider(
            "p макс. окно", options=_P_MAX_OPTS, value=256, key="p_max",
            help=(
                "Верхний предел p-aligned каскада. Уровни строятся снизу вверх: "
                "p_fit × {1,2,4,...} ≤ p_max (скр.93, −12.41% rMAE).\n\n"
                "p=9, p_max=64 → [36,18,9]; p=8, p_max=64 → [64,32,16,8]."
            ),
        )
        p_search = p_max
        k_lags   = 0

    p_fit = st.slider(
        "p прогноз", 2, 200, 13 if is_nonunif else 9, 1, key="p_fit",
        help="Размерность LWR-регрессии. Определяет уровни aligned-каскада.",
    )
    _pf1, _pf2, _pf3, _pf4 = st.columns(4)
    for _pf_btn, _pf_d in ((_pf1, -5), (_pf2, -1), (_pf3, +1), (_pf4, +5)):
        _lbl = f"{'−' if _pf_d < 0 else '+'}{abs(_pf_d)}"
        _key = f"pfit_btn_{'m' if _pf_d < 0 else 'p'}{abs(_pf_d)}"
        if _pf_btn.button(_lbl, key=_key, use_container_width=True):
            st.session_state["_pfit_delta"] = _pf_d
            st.rerun()

    if not is_nonunif:
        _lvls = _levels_aligned(p_fit, p_max)
        st.caption("Каскад (p-aligned ×2): " + " → ".join(str(_p) for _p in _lvls)
                   + f"  ({len(_lvls)} ур.)")

        _xi_base = 3 * (p_fit + 1)
        xi_extra = st.slider(
            "ξ_lwr добавка", 0, 50, 5, 1, key="xi_extra",
            help=(
                "**ξ_lwr** = 3·(p_fit+1) + добавка.\n\n"
                "Базовое значение 3·(p_fit+1) — минимум стабильности LWR "
                "(система не переопределена). Добавка позволяет расширить "
                "пул без ручного пересчёта при смене p.\n\n"
                "**ξ_search** (пул на каждом уровне каскада) = ξ_lwr — "
                "уравнены для удобства, не тестировались раздельно. "
                "Смотри TODO в _forecast_uniform_octave."
            ),
        )
        xi = _xi_base + xi_extra
        xi_max = xi   # для _fc_key (совместимость)
        st.caption(f"ξ_lwr = 3·({p_fit}+1) + {xi_extra} = **{xi}**  "
                   f"|  ξ_search = {xi} (= ξ_lwr)")

        use_global_blend = st.checkbox(
            "Глобальные соседи",
            value=True, key="use_global_blend",
            help=(
                "Часть ξ берётся из каскадного пула, остаток — из всей доступной истории.\n\n"
                "Идея: каскад находит «знакомые» траектории аттрактора, "
                "глобальный поиск добавляет кандидатов из других частей фазового "
                "пространства — потенциальные точки перехода."
            ),
        )
        if use_global_blend:
            global_blend_pct = st.slider(
                "% из каскада", 10, 90, 50, 5, key="global_blend_pct",
                help="50%: равный вклад каскада и глобала. "
                     "10%: почти всё из глобала. 90%: почти всё из каскада.",
            )
            _n_cas  = round(xi * global_blend_pct / 100)
            _n_glob = xi - _n_cas
            st.caption(f"Каскад: {global_blend_pct}% = **{_n_cas}** сос.  |  "
                       f"Глобал: {100 - global_blend_pct}% = **{_n_glob}** сос.")
        else:
            global_blend_pct = 0.0

        st.divider()
        use_acc_ang = st.checkbox(
            "Угловое ускорение (acc_ang)",
            value=True, key="use_acc_ang",
            help=(
                "Финальный отбор соседей по d_pos + λ·d_ang(acc), где\n\n"
                "d_ang(acc) = 1 − cos_sim(acc_q, acc_i),  "
                "acc[t] = att[t] − 2·att[t-1] + att[t-2].\n\n"
                "Кривизна траектории на аттракторе точнее идентифицирует "
                "похожие участки, чем только позиция.\n\n"
                "Скр.80: λ=0.01 → −10.8% rMAE на 8 тикерах 1d.\n\n"
                "В каскадном режиме: _cascade_open передаёт расширенный пул, "
                "acc_ang делает финальный выбор.\n\n"
                "В режиме «Без каскада»: acc_ang выбирает из всей истории."
            ),
        )
        if use_acc_ang:
            acc_ang_lambda = st.slider(
                "λ (вес ускорения)", 0.001, 0.1, 0.01, 0.001,
                format="%.3f", key="acc_ang_lambda",
                help="Оптимум по скр.80: λ=0.01. "
                     "auto-λ≈0.016 (выравнивание медиан). "
                     "При λ>0.05 возможна деградация.",
            )
            st.caption(f"acc_ang: λ={acc_ang_lambda:.3f}  |  Финальный отбор из расширенного пула каскада")
        else:
            acc_ang_lambda = 0.0

        st.divider()
        use_lp_corr = st.checkbox(
            "LP-коррекция траектории",
            value=False, key="use_lp_corr",
            help=(
                "После каждого шага LWR предсказанная точка проецируется обратно "
                "на аттрактор через Local Projective (1 итерация):\n\n"
                "  v = [att[t−(m−2)..t], att_hat]  (m значений)\n\n"
                "  att_hat_корр = LP_project(v, k соседей, d)\n\n"
                "Предотвращает расходимость траектории при горизонте H>5. "
                "Скр.96b/96c: +18% rMAE (H=1), +69% (H=5) на SBER 1d.\n\n"
                "**Важно:** параметры m, k, d должны совпадать с att-фильтром. "
                "При filter_type=«Local Projective» синхронизируется автоматически. "
                "Работает только в режиме «Итеративный с пересчётом»."
            ),
        )
        if use_lp_corr:
            # Параметры всегда берутся из att-фильтра.
            # Local Projective: из слайдеров фильтра (session_state обновляется при рендере).
            # Другие фильтры: дефолты m=9 k=30 d=3 (совпадают с параметрами скр.96b/96c).
            if st.session_state.get("filter_type") == "Local Projective":
                lp_corr_m = int(st.session_state.get("lp_m", 9))
                lp_corr_k = int(st.session_state.get("lp_k", 30))
                lp_corr_d = int(st.session_state.get("lp_d", 3))
                st.caption(f"m={lp_corr_m}  k={lp_corr_k}  d={lp_corr_d}  ← LP-фильтр")
            else:
                lp_corr_m, lp_corr_k, lp_corr_d = 9, 30, 3
                st.caption(f"m={lp_corr_m}  k={lp_corr_k}  d={lp_corr_d}  "
                           f"(дефолт; Local Projective не выбран)")
            lp_corr_n = st.slider("n (итераций коррекции)", 1, 20, 1, 1, key="lp_corr_n",
                                   help="Число итераций LP-проекции на каждом шаге прогноза. "
                                        "n=1: одна проекция (быстро). n>1: повторный поиск соседей "
                                        "из уже спроецированной точки — точнее, но дольше. "
                                        "Независим от n LP-фильтра.")
        else:
            lp_corr_m, lp_corr_k, lp_corr_d, lp_corr_n = 9, 30, 3, 1

        st.subheader("Аппроксиматор")
        use_rbf = st.checkbox("RBF вместо LWR", value=False, key="use_rbf")
        lwr_mode = st.radio(
            "Режим прогноза",
            ["Итеративный с пересчётом", "Итеративный фиксированный", "Прямой"],
            index=0, key="lwr_mode",
            help=(
                "**Итеративный с пересчётом** — на каждом шаге переискивает соседей "
                "по обновлённому вектору.\n\n"
                "**Итеративный фиксированный** — соседи фиксируются по начальной точке, "
                "модель прокатывается с обновлением vec_f.\n\n"
                "**Прямой** — для каждого горизонта h отдельная модель y[t+h+1]=f(x[t])."
            ),
        )
    else:
        st.caption(f"ξ_fit = 3·({p_fit}+1) = {3*(p_fit+1)} (справочно)")
        xi_max           = 300
        use_rbf          = False
        lwr_mode         = "Итеративный с пересчётом"
        global_blend_pct = 0.0
        acc_ang_lambda   = 0.0
        use_lp_corr      = False
        lp_corr_m, lp_corr_k, lp_corr_d, lp_corr_n = 9, 30, 3, 1

    st.divider()
    st.subheader("Att-фильтр")
    filter_type = st.radio(
        "Тип",
        ["LP Butterworth", "LP + AR-filtfilt", "KF2D (Kalman)",
         "SSA rolling", "Local Projective"],
        index=4, key="filter_type",
        help=(
            "**LP Butterworth** — causal фильтр, τ≈7б задержка. Baseline.\n\n"
            "**LP + AR-filtfilt** — AR(20) продлевает dratio на 40 баров, "
            "затем sosfiltfilt. τ≈0, краевой эффект уходит в AR-хвост. "
            "Скр.69/71: +4% price-MAPE vs causal.\n\n"
            "**KF2D** — 2D Kalman (level + slope). Нет фазовой задержки. "
            "Скр.71: MAPE ≈ LP+AR-filtfilt.\n\n"
            "**SSA rolling** — rolling SSA на ratio → diff. "
            "Нет фазовой задержки. k=1 — только тренд, k=2-3 — с осцилляциями. "
            "Скр.73-74: аттрактор при L≥16, k=1-3.\n\n"
            "**Local Projective** — Grassberger-Hegger: проекция каждой точки "
            "фазового пространства на локальное d-мерное касательное подпространство "
            "аттрактора (PCA по соседям). Офлайн, без фазовой задержки. "
            "Скр.75: m=9 d=3 k=30 n=3, сохраняет реальную динамику цены."
        ),
    )
    use_kf       = filter_type == "KF2D (Kalman)"
    use_filtfilt = filter_type == "LP + AR-filtfilt"
    use_ssa      = filter_type == "SSA rolling"
    use_lp_proj  = filter_type == "Local Projective"

    _WN_OPTS = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5]
    _WN_LBLS = {
        0.015625: "0.016  T≈128б  (C4/C5)",
        0.03125:  "0.031  T≈64б   (C3/C4)",
        0.0625:   "0.063  T≈32б   (C2/C3)",
        0.125:    "0.125  T≈16б   (C1/C2) ★",
        0.25:     "0.25   T≈8б    (C0/C1)",
        0.5:      "0.5    Nyquist",
    }
    if not use_ssa and not use_lp_proj:
        wn = st.select_slider("Частота среза Wn", options=_WN_OPTS, value=0.125,
                               format_func=lambda w: _WN_LBLS[w], key="wn")
    else:
        wn = 0.125   # не используется для SSA / Local Projective

    # defaults для неиспользуемых параметров
    kf_q = 0.4
    ssa_W = 128; ssa_L = 32; ssa_k = 2
    lp_m = 9;   lp_d = 3;   lp_k = 30;  lp_n = 3

    if use_kf:
        kf_q = st.slider(
            "q (шум процесса)", 0.01, 1.0, 0.4, 0.01, key="kf_q",
            help="q = q_factor × σ_dratio. Меньше q → глаже сигнал, больше задержка. "
                 "Больше q → отзывчивее, шумнее. Скр.71: оптимум ≈ 0.3-0.4.",
        )
        st.caption(f"LP τ={_compute_lp_tau(wn)}б (для сравнения)  ↔  KF2D без фазовой задержки")
    elif use_ssa:
        ssa_W = st.select_slider("Окно W", options=[64, 128, 256, 512], value=128,
                                  key="ssa_W",
                                  help="Размер скользящего окна. Больше → стабильнее SVD, медленнее.")
        ssa_L = st.select_slider("Эмбеддинг L", options=[8, 16, 32, 64], value=32,
                                  key="ssa_L",
                                  help="Длина вложения. L < W/2 рекомендуется. Скр.73: L=32 оптимум.")
        ssa_k = st.slider("Компоненты k", 1, min(8, ssa_L - 1), 2, 1, key="ssa_k",
                           help="Сколько ведущих компонент SSA оставить. "
                                "k=1 — только тренд (p_opt=3). k=2-3 — с осцилляциями.")
        st.caption(f"SSA на ratio → diff  |  τ≈0б  |  W={ssa_W} L={ssa_L} k={ssa_k}")
    elif use_lp_proj:
        lp_m = st.slider("m (размерность вложения)", 3, 100, 9, 1,
                          key="lp_m",
                          help="Размерность фазового пространства для LP-проекции. "
                               "Скр.75: m=9 (выше 2·d+1=7 для 3D-аттрактора). "
                               "Также используется LP-коррекцией прогноза.")
        lp_d = st.slider("d (размерн. аттрактора)", 1, max(1, lp_m - 1), min(3, lp_m - 1), 1,
                          key="lp_d",
                          help="Число ведущих касательных направлений (PCA-компонент). "
                               "Скр.75: d=2 — слишком мало (0/8), d=3-4 — работает. "
                               "d < m обязательно.")
        lp_k = st.slider("k (соседей)", 5, 1000, 30, 1, key="lp_k",
                          help="Число ближайших соседей для локальной PCA. "
                               "Скр.75: k=30-50. Больше k → глаже, но медленнее.")
        lp_n = st.slider("n (итераций фильтра)", 1, 50, 3, 1, key="lp_n",
                          help="Число итераций LP-проекции для att-фильтра. "
                               "Скр.75: n=1 даёт 3/8 тикеров, n=3 — 8/8. "
                               "LP-коррекция прогноза всегда использует n=1.")
        st.caption(f"Local Projective  |  τ≈0б  |  m={lp_m} d={lp_d} k={lp_k} n={lp_n}  "
                   f"(первый запуск ~30-60с, потом кэш)")
    elif use_filtfilt:
        st.caption(f"AR({_AR_ORDER}) pad={_AR_PAD} → sosfiltfilt  |  τ≈0б  |  +4% MAPE vs causal")
    else:
        st.caption(f"Butterworth order={FILTER_ORDER}, τ={_compute_lp_tau(wn)}б задержка")

    st.divider()
    n_display = st.slider("История на графике (баров)", 50, 500, 150, 10, key="n_display")
    lp_win    = st.slider("LP оверлей (баров)", 10, 200, 50, 5, key="lp_win",
                          help="Сколько последних баров LP-кривой показывать (со сдвигом τ).")

# ── ШАГ 2: загрузка данных ────────────────────────────────────────────────────

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

# ── ШАГ 3: нормализация ───────────────────────────────────────────────────────

df = pd.DataFrame(raw_candles)
df["begin"] = pd.to_datetime(df["begin"])
for col in ("open", "high", "low", "close"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["close"]).reset_index(drop=True)

close    = df["close"].values.astype(np.float64)
logtrend = _logtrend_causal(close)
ratio    = close / logtrend

# ── ШАГ 4: границы origin — только от данных, не от параметров алгоритма ──────
# _origin_min зависит только от max_lag (нужно для построения матрицы обучения).
# xi, p_fit и прочие параметры алгоритма сюда не входят — их нехватку
# обрабатывают сами функции прогноза (возвращают нули + предупреждение).

_max_lag    = lags_desc[0] if is_nonunif else p_max - 1
_origin_min = max(_max_lag + 50, 100)   # 50 строк минимум в обучающей матрице
_origin_max = len(df) - 1

# ── ШАГ 5: применяем _nav_delta до рендера слайдера ──────────────────────────
# Единственное место, где origin_k мутируется программно.

if "_nav_delta" in st.session_state:
    _delta = st.session_state.pop("_nav_delta")
    st.session_state["origin_k"] = int(np.clip(
        st.session_state.get("origin_k", _origin_max) + _delta,
        _origin_min, _origin_max))
    st.session_state["_auto_fc"] = True
elif st.session_state.get("origin_k") is not None:
    # Удерживаем в допустимом диапазоне при смене тикера / интервала
    st.session_state["origin_k"] = int(np.clip(
        st.session_state["origin_k"], _origin_min, _origin_max))

# ── ШАГ 6: origin-слайдер и nav-кнопки ───────────────────────────────────────

PERIODS = {"1 мес": 30, "3 мес": 90, "6 мес": 180, "1 год": 365, "3 года": 1095, "Всё": None}
period  = st.radio("Период", list(PERIODS), index=2, horizontal=True)
days    = PERIODS[period]
last_dt = df["begin"].iloc[-1]
view    = (df[df["begin"] >= last_dt - pd.Timedelta(days=days)].copy()
           if days else df.copy())

origin_k  = st.slider("Origin", _origin_min, _origin_max, _origin_max, 1, key="origin_k")
origin_ts = df["begin"].iloc[origin_k]

_n1, _n2, _n3, _n4, _n5 = st.columns([1, 1, 4, 1, 1])
if _n1.button("⟪ −5", use_container_width=True, help="−5 баров"):
    st.session_state["_nav_delta"] = -5; st.rerun()
if _n2.button("⟨ −1", use_container_width=True, help="−1 бар"):
    st.session_state["_nav_delta"] = -1; st.rerun()
_n3.caption(f"&nbsp;&nbsp;&nbsp;{origin_ts.strftime('%d.%m.%Y')}  |  бар {origin_k}")
if _n4.button("+1 ⟩", use_container_width=True, help="+1 бар"):
    st.session_state["_nav_delta"] = +1; st.rerun()
if _n5.button("+5 ⟫", use_container_width=True, help="+5 баров"):
    st.session_state["_nav_delta"] = +5; st.rerun()

# Предупреждение: мало строк для текущего ξ (не блокирует, алгоритм справится)
if is_nonunif:
    _m_rows = origin_k - _max_lag - 1
    _xi_cur = xi_fin if use_hier else xi
    if _m_rows < _xi_cur:
        st.warning(f"Мало истории: {_m_rows} строк обучения < ξ={_xi_cur}. "
                   "Алгоритм использует все доступные строки.")

# ── ШАГ 7: att-аттрактор в ratio-пространстве ────────────────────────────────
# att[t] ≈ LP(ratio[t]); длина = origin_k+1 (включает ratio[origin_k]).

ratio_orig = ratio[:origin_k + 1]   # causal ratio до origin включительно

if use_kf:
    # KF2D применяется к ratio (не dratio): level отслеживает ratio, slope ≈ dratio
    _att_full = _kf2d_att_full_cached(close.tobytes(), kf_q)
    att_orig  = _att_full[:origin_k + 1]
elif use_ssa:
    with st.spinner("SSA (первый раз кэшируется…)"):
        _att_full = _ssa_att_full_cached(close.tobytes(), ssa_W, ssa_L, ssa_k)
    att_orig  = _att_full[:origin_k + 1]
elif use_lp_proj:
    with st.spinner("Local Projective (первый раз ~30-60с, потом кэш…)"):
        _att_full = _lp_proj_att_full_cached(close[:origin_k + 1].tobytes(), lp_m, lp_d, lp_k, lp_n)
    att_orig  = _att_full   # длина = origin_k+1
elif use_filtfilt:
    att_orig = _lp_att_ar_filtfilt(ratio_orig, wn)   # LP на ratio
else:
    att_orig = _lp_att_signal(ratio_orig, wn)          # LP на ratio

TAU = 0 if (use_kf or use_filtfilt or use_ssa or use_lp_proj) else _compute_lp_tau(wn)

win_start   = max(0, origin_k - lp_win + 1)
# ratio-space: att[t] ≈ LP(ratio[t]), overlay = att[t]*logtrend[t]
lp_prices_h = _reconstruct_lp_prices(att_orig, ratio, logtrend, win_start, origin_k)

lp_bar_arr = np.arange(win_start, origin_k + 1) - TAU
_vm        = (lp_bar_arr >= 0) & (lp_bar_arr < len(df))
lp_x       = df["begin"].iloc[lp_bar_arr[_vm]].values
lp_y_raw   = lp_prices_h[_vm]

_actual_at = close[lp_bar_arr[_vm]]
bias       = float(np.mean(_actual_at - lp_y_raw))
lp_y       = lp_y_raw + bias

# junction: LP-цена в origin_k (без bias-коррекции — прогноз в ratio-space независим)
junction_p = float(att_orig[-1] * logtrend[origin_k])

# ── ШАГ 8: прогноз ────────────────────────────────────────────────────────────

_fbc1, _fbc2, _fbc3 = st.columns([1, 4, 2])
run_btn = _fbc1.button("▶ Прогноз", type="primary", use_container_width=True)
junc_offset_pct = _fbc3.slider(
    "Сдвиг привязки %", min_value=-5.0, max_value=5.0, value=0.0, step=0.1,
    key="junc_offset_pct",
    on_change=lambda: st.session_state.update({"_auto_fc": True}),
)

_fc_key = (ticker, interval, origin_k, horizon, lag_type, tuple(lags_desc),
           p_search, p_max, p_mid, p_fit, xi, xi_max, lwr_mode, use_rbf,
           wn, filter_type, kf_q, ssa_W, ssa_L, ssa_k, lp_m, lp_d, lp_k, lp_n,
           xi_start, xi_fin, unified_space, search_method, adaptive_c,
           global_blend_pct, acc_ang_lambda,
           use_lp_corr, lp_corr_m, lp_corr_k, lp_corr_d, lp_corr_n,
           junc_offset_pct)

if run_btn or st.session_state.pop("_auto_fc", False):
    with st.spinner("Считаю…"):
        # Сдвиг последнего att[origin_k]: в ratio-space это умножение на (1 + offset%).
        # Меняет стартовую точку запроса в каскаде.
        _att_input = att_orig.copy()
        if junc_offset_pct != 0.0:
            _att_input[-1] = _att_input[-1] * (1.0 + junc_offset_pct / 100.0)

        if is_nonunif:
            dhat = _forecast_nonunif(_att_input, lags_desc, p_fit, xi, horizon,
                                     xi_start=xi_start, xi_fin=xi_fin,
                                     unified=unified_space,
                                     search_method=search_method,
                                     adaptive_c=adaptive_c)
        else:
            dhat = _forecast_uniform_octave(_att_input, p_fit, p_max, horizon,
                                            xi_lwr=xi,
                                            use_rbf=use_rbf, lwr_mode=lwr_mode,
                                            global_blend_pct=global_blend_pct,
                                            acc_ang_lambda=acc_ang_lambda,
                                            use_lp_corr=use_lp_corr,
                                            lp_corr_m=lp_corr_m,
                                            lp_corr_k=lp_corr_k,
                                            lp_corr_d=lp_corr_d,
                                            lp_corr_n=lp_corr_n)

    # ratio-space: dhat — уже предсказанные ratio[origin+1..origin+H].
    # Нет cumsum; реконструкция = ratio_hat * logtrend.
    ratio_hat      = dhat
    _lt_fwd        = logtrend[origin_k + 1: origin_k + 1 + horizon]
    if len(_lt_fwd) < horizon:
        _last_lt = float(logtrend[min(origin_k + len(_lt_fwd), len(logtrend) - 1)])
        _lt_fwd  = np.concatenate([_lt_fwd, np.full(horizon - len(_lt_fwd), _last_lt)])
    forecast_price = ratio_hat * _lt_fwd

    _fut = df.iloc[origin_k + 1: origin_k + 1 + horizon]
    if len(_fut) < horizon:
        _bar_d = _BAR_DELTA.get(interval, pd.Timedelta(hours=1))
        _last  = _fut["begin"].iloc[-1] if len(_fut) else origin_ts
        _extra = pd.Series([_last + _bar_d * i for i in range(1, horizon - len(_fut) + 1)])
        forecast_begin = pd.concat([_fut["begin"].reset_index(drop=True), _extra],
                                   ignore_index=True)
    else:
        forecast_begin = _fut["begin"].reset_index(drop=True)

    if is_nonunif:
        _pf_str = f"k={k_lags}" if unified_space else f"pf={p_fit}"
        _sm_str = {"l2": "L2", "norm_l2": "normL2", "borda": "Borda",
                   "adaptive": f"Adp(c={adaptive_c})",
                   "cascade": f"cas {xi_start}→{xi_fin}"}[search_method]
        _lbl = f"LWR [{_sm_str}]  {lag_type}  ξ={xi}  {_pf_str}"
        if unified_space:
            _lbl += "  [uni]"
    else:
        _lvls_lbl  = _levels_aligned(p_fit, p_max)
        _apx_str   = "RBF" if use_rbf else "LWR"
        _mode_abbr = {"Итеративный с пересчётом": "refit",
                      "Итеративный фиксированный": "fixed",
                      "Прямой": "direct"}[lwr_mode]
        _cascade_str = "→".join(str(_p) for _p in _lvls_lbl)
        _lp_corr_str = f"  +LP(m={lp_corr_m},k={lp_corr_k},d={lp_corr_d},n={lp_corr_n})" if use_lp_corr else ""
        _lbl = (f"{_apx_str} aligned/{_mode_abbr}  [{_cascade_str}]  pf={p_fit}"
                f"  ξ={xi}{_lp_corr_str}")

    # Сохраняем снапшот: все данные для рендера, включая junction_p и TAU на момент
    # вычисления — чтобы линия прогноза не зависела от текущего состояния UI.
    st.session_state.update({
        "fc_key":       _fc_key,
        "fc_origin_k":  origin_k,
        "fc_price":     forecast_price,
        "fc_begin":     forecast_begin,
        "fc_junction_p": junction_p * (1.0 + junc_offset_pct / 100.0),
        "fc_tau":       TAU,
        "fc_lbl":       _lbl,
    })

# Загружаем снапшот только если ключ точно совпадает
_snap: dict | None = None
if (st.session_state.get("fc_key") == _fc_key
        and st.session_state.get("fc_origin_k") == origin_k
        and st.session_state.get("fc_price") is not None):
    _snap = {
        "price":      st.session_state["fc_price"],
        "begin":      st.session_state["fc_begin"],
        "junction_p": st.session_state["fc_junction_p"],
        "tau":        st.session_state["fc_tau"],
    }
    _fbc2.caption(st.session_state.get("fc_lbl", ""))

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

if use_kf:
    _overlay_name = f"KF2D  q={kf_q}"
elif use_ssa:
    _overlay_name = f"SSA W={ssa_W} L={ssa_L} k={ssa_k}  τ≈0"
elif use_lp_proj:
    _overlay_name = f"LocalProj m={lp_m} d={lp_d} k={lp_k} n={lp_n}  τ≈0"
elif use_filtfilt:
    _overlay_name = f"LP+AR-filtfilt({wn})  τ≈0"
else:
    _overlay_name = f"LP({wn})  τ={TAU}б"
fig.add_trace(go.Scatter(
    x=lp_x, y=lp_y, mode="lines",
    name=_overlay_name,
    line=dict(color="rgba(100,181,246,0.85)", width=2.0),
))

_junc_bar = max(0, origin_k - TAU)
fig.add_trace(go.Scatter(
    x=[df["begin"].iloc[_junc_bar]], y=[junction_p],
    mode="markers", name="Привязка",
    marker=dict(color="#00e5ff", size=8, symbol="circle"), showlegend=False,
))
fig.add_vline(x=str(origin_ts)[:10],
              line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dash"))

if _snap is not None:
    # Строим ось X прогноза из снапшота: junction_p и tau зафиксированы на момент расчёта
    _s_junc_bar = max(0, origin_k - _snap["tau"])
    _s_junc_ts  = df["begin"].iloc[_s_junc_bar]
    _bar_d      = _BAR_DELTA.get(interval, pd.Timedelta(hours=1))
    _fc_x_list  = [_s_junc_ts]
    _last_ts    = _s_junc_ts
    n_fc        = min(len(_snap["price"]), len(_snap["begin"]))
    for _i in range(1, n_fc + 1):
        _bi = _s_junc_bar + _i
        if _bi < len(df):
            _ts = df["begin"].iloc[_bi]
        else:
            _ts = _last_ts + _bar_d
        _fc_x_list.append(_ts)
        _last_ts = _ts

    _fc_x = pd.Series(_fc_x_list)
    _fc_y = np.concatenate([[_snap["junction_p"]], _snap["price"][:n_fc]])
    if is_nonunif:
        _fc_name = f"Прогноз LWR {lag_type}  ξ={xi}  pf={p_fit}"
    else:
        _mid_str    = f" pm={p_mid}" if p_mid is not None else ""
        _approx_str = "RBF" if use_rbf else "LWR"
        _fc_name    = (f"Прогноз {_approx_str} {lwr_mode}"
                       f"  ps={p_search}{_mid_str}  ξ={xi}  pf={p_fit}")
    fig.add_trace(go.Scatter(
        x=_fc_x, y=_fc_y, mode="lines",
        name=_fc_name,
        line=dict(color="#00e5ff", width=2.5),
    ))

fig.update_layout(
    **_layout, height=520,
    xaxis=dict(**_grid, title="", rangebreaks=rangebreaks, rangeslider=dict(visible=False)),
    yaxis=dict(**_grid, title="Цена"),
    legend=dict(orientation="h", yanchor="bottom", y=1.01,
                xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

# ── ШАГ 10: метрики ───────────────────────────────────────────────────────────

_i1, _i2, _i3, _i4 = st.columns(4)
_i1.metric("Баров в истории", f"{origin_k:,}")
_i2.metric("LP привязка",     f"{junction_p:.2f}")
if use_kf:
    _i3.metric("KF q / bias",        f"{kf_q} / {bias:+.2f}")
elif use_ssa:
    _i3.metric("SSA / bias",         f"W={ssa_W} L={ssa_L} k={ssa_k} / {bias:+.2f}")
elif use_lp_proj:
    _i3.metric("LocalProj / bias",   f"m={lp_m} d={lp_d} k={lp_k} n={lp_n} / {bias:+.2f}")
elif use_filtfilt:
    _i3.metric("AR-filtfilt / bias", f"τ≈0 / {bias:+.2f}")
else:
    _i3.metric("LP τ / bias",        f"{TAU}б / {bias:+.2f}")
if is_nonunif:
    _mid_label  = f"ξ={xi} / — / {p_fit}"
    _mid_header = "ξ / pm / pf"
else:
    _mid_label  = f"{p_search} / {p_mid} / {p_fit}" if p_mid else f"{p_search} / — / {p_fit}"
    _mid_header = "ps / pm / pf"
_i4.metric(_mid_header, _mid_label)
