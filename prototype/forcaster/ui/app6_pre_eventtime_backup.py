"""
app6: Двухэтапный автоматический пайплайн.

Этап 1: PCA sweep m∈[m_min,m_max] → d_min=⌊mean d(m,τ₁)⌋, d_max=⌈mean d(m,τ₂)⌉.
Этап 2: sweep d∈[d_min,d_max] → S-map с PCA-проекцией пула в d-мерное подпространство
        → простое среднее по d.

Стек: log(price) → каузальный LP-фильтр(ratio, m=x·d+y) → каскадный S-map.
Обозначения: m — окно задержки, K = 3·(m+1)+ξ — число соседей, d — размерность аттрактора.
Метрика: amp_cos. Каскад: октавный ×2. Нет утечки из будущего.

Run (из prototype/): streamlit run forcaster/ui/app6.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
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


def _ohlc_cosine_dist(X_ohlc: np.ndarray, q_ohlc: np.ndarray) -> np.ndarray:
    """Косинусное расстояние для OHLC-окон. X_ohlc: (n, 2p), q_ohlc: (2p,)."""
    nX = np.linalg.norm(X_ohlc, axis=1)
    nq = float(np.linalg.norm(q_ohlc))
    if nq < 1e-10:
        return np.ones(len(X_ohlc))
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(nX > 1e-10, (X_ohlc @ q_ohlc) / (nX * nq), 0.0)
    return 1.0 - sim.clip(-1.0, 1.0)


def _ohlc_windows(starts: np.ndarray, p: int,
                  shadow_arr: np.ndarray, pressure_arr: np.ndarray) -> np.ndarray:
    """(n, 2p) — конкатенация shadow и pressure окон для стартовых позиций starts."""
    idx = starts[:, None] + np.arange(p)
    return np.concatenate([shadow_arr[idx], pressure_arr[idx]], axis=1)


def _ohlc_query(t_predict: int, p: int,
                shadow_arr: np.ndarray, pressure_arr: np.ndarray) -> np.ndarray:
    """(2p,) — OHLC-окно запроса; будущие бары заменяются последним известным."""
    n     = len(shadow_arr)
    idx   = np.clip(np.arange(t_predict - p, t_predict), 0, n - 1)
    return np.concatenate([shadow_arr[idx], pressure_arr[idx]])


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
# Синтетический хаотический ряд (система Лоренца)
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_lorenz_candles(n_total: int = 3500, dt: float = 0.02, seed: int = 42) -> list:
    """Интегрирует систему Лоренца (RK4), возвращает fake-свечи по Z-координате."""
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def _deriv(x, y, z):
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    xyz = np.empty((n_total, 3))
    xyz[0] = [1.0, 0.0, 0.0]
    for i in range(1, n_total):
        x, y, z = xyz[i - 1]
        k1 = np.array(_deriv(x, y, z))
        h  = xyz[i - 1] + dt / 2 * k1
        k2 = np.array(_deriv(*h))
        h  = xyz[i - 1] + dt / 2 * k2
        k3 = np.array(_deriv(*h))
        h  = xyz[i - 1] + dt * k3
        k4 = np.array(_deriv(*h))
        xyz[i] = xyz[i - 1] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    warmup = 500                       # срез переходного процесса
    z = xyz[warmup:, 2]                # Z ∈ (0, ~50) — всегда положительна
    price = z * 2.0 + 10.0            # масштаб ~10–110, похоже на цену

    rng = np.random.default_rng(seed)
    candles = []
    base = date(2020, 1, 1)
    for i, c in enumerate(price):
        o = float(price[i - 1]) if i > 0 else float(c)
        noise = abs(float(rng.normal(0, c * 0.005)))
        candles.append({
            "begin": str(base + timedelta(days=i)),
            "open":  o,
            "high":  max(o, float(c)) + noise,
            "low":   min(o, float(c)) - noise,
            "close": float(c),
        })
    return candles

# ═══════════════════════════════════════════════════════════════════════════════
# KF2D (constant-velocity Kalman, предобработка цены)
# ═══════════════════════════════════════════════════════════════════════════════

def _kf2d_smooth(price: np.ndarray, q_factor: float) -> np.ndarray:
    """2D constant-velocity Kalman на сырых ценах; state=[level, slope]."""
    dp = np.diff(price)
    sigma = float(np.std(dp)) if len(dp) > 1 else 1.0
    if sigma < 1e-12:
        return price.copy()
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
            S   = float(H @ P_p @ H) + R_var
            K_ss = (P_p @ H) / S
            P   = (np.eye(2) - np.outer(K_ss, H)) @ P_p
    A_ss = (np.eye(2) - np.outer(K_ss, H)) @ F
    x = np.array([price[0], 0.0])
    out = np.empty(len(price))
    for k, z in enumerate(price):
        x = A_ss @ x + K_ss * z
        out[k] = x[0]
    return out


# SSA (rolling causal, на ratio)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _ssa_smooth_ratio(ratio_bytes: bytes, W: int, L: int, k: int) -> np.ndarray:
    """
    Rolling causal SSA: на каждом t берём ratio[t-W+1..t], SVD траекторной матрицы,
    реконструируем k компонентами. Нужен только последний элемент сглаженного окна —
    это Xr[K_m-1, L-1], что избавляет от полного диагонального усреднения.
    Для t < W-1 значение сохраняется без изменений.
    """
    ratio = np.frombuffer(ratio_bytes, dtype=np.float64).copy()
    N   = len(ratio)
    out = ratio.copy()
    K_m = W - L + 1
    if K_m < 2 or N < W:
        return out
    idx = np.arange(K_m)[:, None] + np.arange(L)[None, :]
    for t in range(W - 1, N):
        w          = ratio[t - W + 1 : t + 1]
        X          = w[idx]
        U, s, Vt   = np.linalg.svd(X, full_matrices=False)
        nk         = min(k, len(s))
        out[t]     = float(((U[:, :nk] * s[:nk]) @ Vt[:nk, :])[K_m - 1, L - 1])
    return out


# ZigZag (causal, предобработка ratio)
# ═══════════════════════════════════════════════════════════════════════════════

def _zigzag_find_pivots(ratio: np.ndarray, threshold: float) -> list[int]:
    """Возвращает список индексов подтверждённых пивотов ZigZag.

    ratio = log(price): абсолютная разность в log-пространстве ≈ относительное
    изменение цены, поэтому порог применяется как абсолютный (без домножения
    на текущий уровень ratio, в отличие от прежней ratio≈1 схемы).
    """
    n = len(ratio)
    if n < 3 or threshold <= 0:
        return [0]
    pivots: list[int] = [0]
    direction = 0
    ext_val, ext_idx = ratio[0], 0
    for i in range(1, n):
        v = ratio[i]
        if direction == 0:
            if abs(v - ext_val) >= threshold:
                direction = 1 if v > ext_val else -1
                ext_val, ext_idx = v, i
        elif direction == 1:
            if v > ext_val:
                ext_val, ext_idx = v, i
            elif (ext_val - v) >= threshold:
                pivots.append(ext_idx)
                direction = -1
                ext_val, ext_idx = v, i
        else:
            if v < ext_val:
                ext_val, ext_idx = v, i
            elif (v - ext_val) >= threshold:
                pivots.append(ext_idx)
                direction = 1
                ext_val, ext_idx = v, i
    return pivots


def _zigzag_ratio(ratio: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """
    Каузальный ZigZag на ratio. Отрезки между подтверждёнными пивотами
    заменяются линейной интерполяцией. Последний (незакрытый) отрезок — без изменений.
    """
    if len(ratio) < 3 or threshold <= 0:
        return ratio.copy()
    pivots = _zigzag_find_pivots(ratio, threshold)
    out = ratio.copy()
    for j in range(len(pivots) - 1):
        i0, i1 = pivots[j], pivots[j + 1]
        out[i0 : i1 + 1] = np.linspace(ratio[i0], ratio[i1], i1 - i0 + 1)
    return out


# Логарифмирование цены (без вычисления тренда)
# ═══════════════════════════════════════════════════════════════════════════════

def _log_price(close: np.ndarray) -> np.ndarray:
    """log(price) — базовый ряд для всего пайплайна, без detrending."""
    return np.log(np.maximum(close, 1e-10))

# ═══════════════════════════════════════════════════════════════════════════════
# LP-фильтр (causal, ratio → att)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _lp_proj_ratio_cached(ratio_bytes: bytes, m: int, d: int, k: int, n_iter: int,
                           blend_alpha: float = 0.5, tau: int = 1) -> np.ndarray:
    ratio = np.frombuffer(ratio_bytes, dtype=np.float64).copy()
    return _lp_proj_causal(ratio, m, d, k, n_iter, blend_alpha, tau)

def _lp_proj_causal(ratio: np.ndarray, m: int, d: int, k: int, n_iter: int,
                    blend_alpha: float = 0.5, tau: int = 1) -> np.ndarray:
    """
    Причинный LP-фильтр: att[t] вычисляется только по ratio[0..t].
    att[t] = последний элемент проекции текущего окна на локальное d-мерное подпространство.
    tau > 1: окно {att[q], att[q+τ], ..., att[q+τ(m-1)]}, обновляет att[q+τ(m-1)].
    """
    n     = len(ratio)
    d_eff = min(d, m - 1)
    att   = ratio.copy().astype(np.float64)
    span  = tau * (m - 1)
    if d_eff < 1 or n < span + 2:
        return att

    nw = n - span  # окна q = 0..nw-1; последний бар окна q = q+span

    for _ in range(n_iter):
        X_all   = att[np.arange(nw)[:, None] + np.arange(m)[None, :] * tau]
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
            new_att[q + span] = proj[-1]

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
    tau: int = 1,
    pool_mask: np.ndarray | None = None,
) -> dict:
    """PCA-размерность по хвосту ratio (без LP, без каскада). ratio уже обрезан до origin+bars."""
    n_eff = len(ratio)
    span  = tau * (p_search - 1)
    if n_eff < span + 2:
        return {"error": "недостаточно истории"}
    query = ratio[n_eff - 1 - span : n_eff : tau][:p_search].astype(np.float64)
    if len(query) < p_search:
        return {"error": "недостаточно истории для запроса"}
    max_s = n_eff - span - 2
    if max_s < 0:
        return {"error": "библиотека пуста"}
    starts = np.arange(0, max_s + 1)
    if pool_mask is not None:
        valid  = starts[starts < len(pool_mask)]
        starts = valid[pool_mask[valid]]
    if len(starts) < k_dim:
        return {"error": f"библиотека мала: {len(starts)} < {k_dim}"}
    idx    = starts[:, None] + np.arange(p_search)[None, :] * tau
    X_lib  = ratio[idx].astype(np.float64)
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
    tau: int = 1,
    use_zone_level: bool = False,
    zone_addend: int = 3,
    zone_overlap_pct: int = 0,
    n_levels: int = 2,
) -> list[dict]:
    origin = len(ratio) - 1
    rows   = []
    for p in p_values:
        k_dim  = 3 * (p + 1) + xi_add
        k_zone = max(2, k_dim >> (n_levels - 1))
        pmask = sliding_zone_mask(ratio, origin, p + zone_addend, k_zone, zone_overlap_pct=zone_overlap_pct) if use_zone_level else None
        res   = compute_local_dim(ratio, p, k_dim,
                                   pca_thresholds, blend_alpha, tau, pmask)
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
# Зонный глобальный уровень каскада (PIP + 2D zone KNN)
# ═══════════════════════════════════════════════════════════════════════════════

# Цвета зон для Plotly (direction × vol_level): тихий/средний/активный
_PIP_ZONE_BG = {
    (+1, 0): "rgba(232,248,238,0.50)",
    (+1, 1): "rgba(160,230,185,0.50)",
    (+1, 2): "rgba( 90,200,130,0.50)",
    (-1, 0): "rgba(253,238,237,0.50)",
    (-1, 1): "rgba(248,185,180,0.50)",
    (-1, 2): "rgba(230,110,100,0.50)",
}
_PIP_ZONE_ANN = {
    (+1, 0): "↑",  (+1, 1): "↑↑",  (+1, 2): "↑↑↑",
    (-1, 0): "↓",  (-1, 1): "↓↓",  (-1, 2): "↓↓↓",
}


def _pip_min_size(sig: np.ndarray, n_pips: int, min_size: int) -> list[int]:
    """PIP с ограничением минимального размера сегмента.
    Кандидат i пропускается если создаёт сегмент < min_size баров.
    """
    n = len(sig)
    if n_pips >= n:
        return list(range(n))
    selected = [0, n - 1]
    for _ in range(n_pips - 2):
        best_dist = -1.0
        best_idx  = -1
        for j in range(len(selected) - 1):
            a, b = selected[j], selected[j + 1]
            if b - a < 2:
                continue
            xa, ya = float(a), sig[a]
            xb, yb = float(b), sig[b]
            dx = xb - xa; dy = yb - ya
            L  = np.hypot(dx, dy) + 1e-12
            for i in range(a + 1, b):
                if (i - a) < min_size or (b - i) < min_size:
                    continue
                d_perp = abs(dy * (float(i) - xa) - dx * (sig[i] - ya)) / L
                if d_perp > best_dist:
                    best_dist = d_perp; best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        selected.sort()
    return selected


def _zone_vectors_normalized(
    sig: np.ndarray,
    pivots: list[int],
    ohlc_ratios: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[list[dict], np.ndarray]:
    """
    5D вектор {slope, vol, vol_trend, curvature, autocorr} для каждой зоны.
    slope     = наклон PIP-прямой (единиц/бар).
    vol       = std остатков относительно PIP-прямой.
    vol_trend = OLS-наклон |остатков| по времени (>0 — растущая vol, <0 — убывающая).
    curvature = среднее остатков (>0 — выпуклость вверх, <0 — вниз, ≈0 — линейная зона).
    autocorr  = лаг-1 автокорреляция остатков (>0 — инерционный шум, <0 — возвратный).
                Если ohlc_ratios заданы — все признаки считаются по sig (close-equiv);
                vol и vol_trend дополнительно усредняются по 4 OHLC-компонентам.
    Возвращает (zones, vecs_norm) — z-score нормализация по историческим зонам.
    Каждая зона: {a, b, direction (+1/-1), vol, vol_level (0/1/2)}.
    """
    zones: list[dict] = []
    vecs_raw: list[list[float]] = []
    n_sig = len(sig)
    for i in range(len(pivots) - 1):
        a, b = pivots[i], pivots[i + 1]
        if b <= a:
            continue
        seg = sig[a : min(b + 1, n_sig)].astype(float)
        n_seg = len(seg)
        if n_seg >= 2:
            t     = np.arange(n_seg, dtype=float)
            slope = (seg[-1] - seg[0]) / (n_seg - 1)
            line  = seg[0] + slope * t
            res   = seg - line          # остатки только по sig (close-equiv)

            if ohlc_ratios is not None:
                residuals = [res]
                for arr in ohlc_ratios:
                    seg_c = arr[a : min(b + 1, len(arr))].astype(float)
                    seg_c = seg_c[:n_seg]
                    residuals.append(seg_c - line[:len(seg_c)])
                all_res = np.concatenate(residuals)
            else:
                all_res = res

            vol = float(np.std(all_res))

            # vol_trend: OLS-наклон |остатков| по времени
            abs_res = np.abs(all_res)
            t_rep   = np.tile(t, len(all_res) // n_seg) if len(all_res) > n_seg else t
            t_c     = t_rep - t_rep.mean()
            denom   = float(np.dot(t_c, t_c))
            vol_trend = float(np.dot(t_c, abs_res) / denom) if denom > 1e-12 else 0.0

            # curvature: среднее остатков по sig (PIP-прямая — не OLS, среднее ≠ 0)
            curvature = float(res.mean())

            # autocorr: лаг-1 по остаткам sig
            if n_seg >= 3:
                r0 = res - res.mean()
                autocorr = float(np.dot(r0[:-1], r0[1:]) / (np.dot(r0, r0) + 1e-12))
            else:
                autocorr = 0.0
        else:
            slope = 0.0; vol = 0.0; vol_trend = 0.0; curvature = 0.0; autocorr = 0.0

        direction = +1 if sig[min(b, n_sig - 1)] >= sig[a] else -1
        zones.append({'a': a, 'b': b, 'direction': direction, 'vol': vol})
        vecs_raw.append([slope, vol, vol_trend, curvature, autocorr])

    arr      = np.array(vecs_raw, dtype=np.float64)
    hist_arr = arr[:-1] if len(arr) > 1 else arr
    means    = hist_arr.mean(axis=0)
    stds     = hist_arr.std(axis=0)
    stds     = np.where(stds < 1e-12, 1.0, stds)
    vecs_norm = (arr - means) / stds

    # vol_level по второй колонке (vol)
    vol_arr = arr[:, 1]
    qs = np.percentile(vol_arr, [33.3, 66.7]) if len(vol_arr) > 3 else np.array([np.median(vol_arr)] * 2)
    for z in zones:
        z['vol_level'] = int(np.searchsorted(qs, z['vol']))

    return zones, vecs_norm


def _pip_zone_search(
    ratio: np.ndarray,
    origin: int,
    p_fit: int,
    n_levels: int,
    K_zones: int,
    ohlc_ratios: tuple | None = None,
) -> tuple[np.ndarray | None, list[dict]]:
    """
    Зонный глобальный уровень каскада.
    min_zone = p_fit × 2^(n_levels−1)  (размерность верхнего обычного уровня каскада).
    n_pips   = N_avail // min_zone + 1  (максимум зон при данном ограничении).

    Находит K ближайших исторических зон к зоне origin в {slope, vol}-пространстве.
    Возвращает (pool_mask, zones_display).
    zones_display — список зон; ключ 'selected' True для выбранных.
    """
    min_zone = p_fit * (2 ** (n_levels - 1))
    n_avail  = origin + 1
    n_pips   = max(3, n_avail // min_zone + 1)

    pivots = _pip_min_size(ratio[:n_avail], n_pips, min_zone)
    if len(pivots) < 3:
        return None, []

    ohlc_trimmed = tuple(arr[:n_avail] for arr in ohlc_ratios) if ohlc_ratios is not None else None
    zones, vecs_norm = _zone_vectors_normalized(ratio[:n_avail], pivots, ohlc_trimmed)
    if len(zones) < 2:
        return None, []

    origin_vec = vecs_norm[-1]   # зона origin = последняя
    hist_vecs  = vecs_norm[:-1]
    hist_zones = zones[:-1]

    K_eff = min(K_zones, len(hist_zones))
    if K_eff == 0:
        return None, []

    dists    = np.sum((hist_vecs - origin_vec) ** 2, axis=1)
    top_set  = set(np.argpartition(dists, K_eff - 1)[:K_eff].tolist())

    mask = np.zeros(len(ratio), dtype=bool)
    for idx in top_set:
        z = hist_zones[idx]
        mask[z['a'] : z['b'] + 1] = True

    display: list[dict] = []
    for i, z in enumerate(hist_zones):
        display.append({**z, 'selected': i in top_set})
    display.append({**zones[-1], 'selected': True})   # origin-зона

    return (mask if mask.any() else None), display


def zone_level_mask(
    ratio: np.ndarray,
    origin: int,
    p_fit: int,
    n_levels: int,
    K_zones: int,
    ohlc_ratios: tuple | None = None,
) -> np.ndarray | None:
    """Только маска — обёртка для использования внутри run_stage2/run_forecast_refined."""
    mask, _ = _pip_zone_search(ratio, origin, p_fit, n_levels, K_zones, ohlc_ratios)
    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# Sliding-window zone search (заменяет PIP-зонирование)
# ═══════════════════════════════════════════════════════════════════════════════

def _sliding_zone_search(
    ratio: np.ndarray,
    origin: int,
    W: int,
    K_zones: int,
    ohlc_ratios: tuple | None = None,
    zone_overlap_pct: int = 0,
) -> tuple[np.ndarray | None, list[dict]]:
    """
    Скользящее окно размером W, шаг=1.
    Theiler window = W: исключаем окна с t_end > origin-W (перекрытие с origin-окном).
    zone_overlap_pct: допустимое перекрытие между выбранными зонами, % от W.
    Признаки окна: [slope, vol, vol_trend, curvature, autocorr] — OLS-аппроксимация.
    KNN в нормализованном 5D-пространстве → pool_mask = объединение K ближайших окон.
    """
    n = origin + 1
    if n < 2 * W + 1 or W < 2:
        return None, []

    # Матрица всех допустимых окон: строка i = ratio[i..i+W-1], t_end = i+W-1
    # Условие: t_end <= origin-W  →  i+W-1 <= origin-W  →  i <= origin-2W+1
    n_hist = origin - 2 * W + 2      # число допустимых исторических окон
    if n_hist <= 0:
        return None, []

    # Строим матрицу X: (n_hist, W)
    idx = np.arange(n_hist)[:, None] + np.arange(W)[None, :]   # (n_hist, W)
    X   = ratio[idx].astype(np.float64)                         # (n_hist, W)
    t   = np.arange(W, dtype=np.float64)
    t_c = t - t.mean()
    denom = float(np.dot(t_c, t_c))

    # OLS slope и intercept
    slopes     = X @ t_c / denom                           # (n_hist,)
    intercepts = X.mean(axis=1) - slopes * t.mean()
    lines      = intercepts[:, None] + slopes[:, None] * t  # (n_hist, W)
    res        = X - lines                                  # (n_hist, W)

    # OHLC: добавляем остатки всех 4 компонент для vol/vol_trend
    if ohlc_ratios is not None:
        ohlc_res = [res]
        for arr in ohlc_ratios:
            X_c = arr[idx].astype(np.float64)
            ohlc_res.append(X_c - lines)
        all_res = np.concatenate(ohlc_res, axis=1)          # (n_hist, W*5)
    else:
        all_res = res                                        # (n_hist, W)

    vol       = all_res.std(axis=1)                         # (n_hist,)
    curvature = res.mean(axis=1)                            # (n_hist,) — только по sig

    # vol_trend: OLS-наклон |остатков| по тайлированному t_c
    n_comp    = all_res.shape[1] // W
    t_c_rep   = np.tile(t_c, n_comp)
    denom_rep = float(np.dot(t_c_rep, t_c_rep))
    vol_trend = np.abs(all_res) @ t_c_rep / denom_rep       # (n_hist,)

    # autocorr лаг-1 по res (только sig)
    r0       = res - res.mean(axis=1, keepdims=True)
    num_ac   = (r0[:, :-1] * r0[:, 1:]).sum(axis=1)
    den_ac   = (r0 * r0).sum(axis=1)
    autocorr = num_ac / (den_ac + 1e-12)                    # (n_hist,)

    hist_vecs = np.column_stack([slopes, vol, vol_trend, curvature, autocorr])  # (n_hist, 5)

    # Origin-окно
    ow       = ratio[origin - W + 1 : origin + 1].astype(np.float64)
    ow_slope = float(np.dot(t_c, ow) / denom)
    ow_line  = (float(ow.mean()) - ow_slope * t.mean()) + ow_slope * t
    ow_res   = ow - ow_line
    if ohlc_ratios is not None:
        ohlc_ow = [ow_res]
        for arr in ohlc_ratios:
            seg = arr[origin - W + 1 : origin + 1].astype(np.float64)
            ohlc_ow.append(seg - ow_line)
        ow_all_res = np.concatenate(ohlc_ow)
    else:
        ow_all_res = ow_res
    ow_vol       = float(ow_all_res.std())
    ow_vtop      = float(np.dot(np.tile(t_c, len(ow_all_res) // W), np.abs(ow_all_res)) / denom_rep)
    ow_curve     = float(ow_res.mean())
    r0_ow        = ow_res - ow_res.mean()
    ow_autocorr  = float(np.dot(r0_ow[:-1], r0_ow[1:]) / (np.dot(r0_ow, r0_ow) + 1e-12))
    origin_vec   = np.array([ow_slope, ow_vol, ow_vtop, ow_curve, ow_autocorr])

    # Z-score по историческим окнам
    means = hist_vecs.mean(axis=0)
    stds  = hist_vecs.std(axis=0)
    stds  = np.where(stds < 1e-12, 1.0, stds)
    hist_norm   = (hist_vecs - means) / stds
    origin_norm = (origin_vec - means) / stds

    # Greedy KNN с контролируемым перекрытием между выбранными окнами.
    # zone_overlap_pct=0  → окна не пересекаются (excl_radius=W).
    # zone_overlap_pct=50 → соседние окна могут делить до W/2 баров.
    # zone_overlap_pct=100→ исключается только то же самое окно (excl_radius=1).
    overlap_bars = int(round(zone_overlap_pct / 100 * W))
    excl_radius  = max(1, W - overlap_bars)
    K_eff      = min(K_zones, n_hist)
    dists      = np.sum((hist_norm - origin_norm) ** 2, axis=1)
    sorted_idx = np.argsort(dists)
    excluded   = np.zeros(n_hist, dtype=bool)
    selected   = []
    for i in sorted_idx.tolist():
        if excluded[i]:
            continue
        selected.append(i)
        if len(selected) >= K_eff:
            break
        lo = max(0, i - excl_radius + 1)
        hi = min(n_hist, i + excl_radius)
        excluded[lo:hi] = True
    top_idx = set(selected)

    # Pool mask
    mask = np.zeros(len(ratio), dtype=bool)
    for i in top_idx:
        t_end = i + W - 1
        mask[t_end - W + 1 : t_end + 1] = True

    # Display: только выбранные окна + origin-окно; vol_level по квантилям выбранных
    sel_vols = [float(hist_vecs[i, 1]) for i in top_idx]
    qs = np.percentile(sel_vols, [33.3, 66.7]) if len(sel_vols) > 2 else np.array([np.median(sel_vols)] * 2)

    display: list[dict] = []
    for i in top_idx:
        t_end = i + W - 1
        win   = ratio[t_end - W + 1 : t_end + 1]
        display.append({
            'a': t_end - W + 1, 'b': t_end,
            'direction': +1 if win[-1] >= win[0] else -1,
            'vol': float(hist_vecs[i, 1]),
            'vol_level': int(np.searchsorted(qs, hist_vecs[i, 1])),
            'selected': True,
        })
    display.append({
        'a': origin - W + 1, 'b': origin,
        'direction': +1 if ow[-1] >= ow[0] else -1,
        'vol': ow_vol, 'vol_level': 1, 'selected': True,
    })

    return (mask if mask.any() else None), display


def sliding_zone_mask(
    ratio: np.ndarray,
    origin: int,
    W: int,
    K_zones: int,
    ohlc_ratios: tuple | None = None,
    zone_overlap_pct: int = 0,
) -> np.ndarray | None:
    mask, _ = _sliding_zone_search(ratio, origin, W, K_zones, ohlc_ratios, zone_overlap_pct)
    return mask


def regime_cluster_mask(
    att: np.ndarray,
    origin: int,
    p_fit: int,
    n_clusters: int = 5,
    n_pca: int = 0,
) -> np.ndarray | None:
    """
    KMeans-кластеризация фазового пространства att (задержки длины p_fit).
    pool_mask[s] = True  ←→  вектор att[s:s+p_fit] в том же кластере, что origin-вектор.
    n_pca > 0: предварительная PCA-редукция до n_pca компонент.
    """
    from sklearn.cluster import KMeans

    n_avail = origin - p_fit + 1
    if n_avail < max(n_clusters * 3, 10):
        return None

    starts = np.arange(n_avail)
    X = att[starts[:, None] + np.arange(p_fit)].astype(np.float64)  # (n_avail, p_fit)

    mu  = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    X_n = (X - mu) / std

    if 0 < n_pca < p_fit:
        from sklearn.decomposition import PCA
        pca  = PCA(n_components=n_pca, random_state=0)
        X_r  = pca.fit_transform(X_n)
        q_r  = pca.transform(((att[origin - p_fit + 1 : origin + 1].astype(np.float64) - mu) / std).reshape(1, -1))[0]
    else:
        X_r = X_n
        q_r = (att[origin - p_fit + 1 : origin + 1].astype(np.float64) - mu) / std

    km     = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    labels = km.fit_predict(X_r)
    origin_label = int(km.predict(q_r.reshape(1, -1))[0])

    mask = np.zeros(len(att), dtype=bool)
    mask[starts[labels == origin_label]] = True
    return mask


def _expand_mask_right(mask: np.ndarray, steps: int) -> np.ndarray:
    """Расширяет правую границу каждой True-зоны на `steps` баров (O(N))."""
    if steps <= 0:
        return mask
    n      = len(mask)
    result = mask.copy()
    extra  = 0
    for i in range(n):
        if mask[i]:
            extra = steps
        elif extra > 0:
            result[i] = True
            extra -= 1
    return result


def _shift_mask_right(mask: np.ndarray, steps: int) -> np.ndarray:
    """Сдвигает маску вправо на `steps` баров (левая и правая границы двигаются вместе)."""
    if steps <= 0:
        return mask
    n = len(mask)
    result = np.zeros(n, dtype=bool)
    if steps < n:
        result[steps:] = mask[:n - steps]
    return result


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
    shadow_arr: np.ndarray | None = None,
    pressure_arr: np.ndarray | None = None,
    ohlc_filter_last: bool = False,
    final_pool_all: bool = True,
    tau_base: int = 1,
    pool_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Строгий октавный каскад ×2 (cascade_algorithm.md).
    tau_base > 1: все векторы τ-разреженные; охват уровня p_lv = tau_base*(p_lv-1) баров.
    OHLC отключается при tau_base > 1 (окна OHLC только последовательные).
    final_pool_all=True: весь финальный пул → S-map; False: K лучших → S-map.
    """
    n      = len(att)
    levels = _cascade_levels(p_fit, n_levels, p_cascade_max)
    p_top  = levels[0]
    n_lvls = len(levels)

    span_top = tau_base * (p_top - 1)
    max_s    = min(n - 1 - span_top, t_predict - 1 - span_top)
    if max_s < 0:
        return None, None

    pool = np.arange(0, max_s + 1)
    if pool_mask is not None:
        _valid = pool[pool < len(pool_mask)]
        pool = _valid[pool_mask[_valid]]
    if len(pool) < K:
        return None, None

    use_ohlc = (shadow_arr is not None and pressure_arr is not None) and tau_base == 1
    K2 = 2 * K

    for level_idx, p_lv in enumerate(levels):
        is_last  = (level_idx == n_lvls - 1)
        span_lv  = tau_base * (p_lv - 1)

        if len(context) < span_lv + 1:
            return None, None
        q = context[-(span_lv + 1)::tau_base][:p_lv].astype(np.float64)
        if len(q) < p_lv:
            return None, None

        max_s = min(n - 1 - span_lv, t_predict - 1 - span_lv)
        pool  = pool[pool <= max_s]
        if len(pool) == 0:
            return None, None

        X_pool = att[pool[:, None] + np.arange(p_lv)[None, :] * tau_base]
        dists  = _dists(X_pool, q, blend_alpha)

        if is_last:
            y_step   = span_lv + 1                      # следующий бар после последнего элемента
            valid_y  = pool + y_step < n
            pool_sm  = pool[valid_y]
            X_sm     = X_pool[valid_y]
            dists_sm = dists[valid_y]
            if len(pool_sm) < 2:
                return None, None

            if ohlc_filter_last and use_ohlc and len(pool_sm) > K:
                n_v    = min(K2, len(pool_sm))
                top_v  = np.argpartition(dists_sm, n_v - 1)[:n_v]
                pool_v = pool_sm[top_v]
                ohlc_lib = _ohlc_windows(pool_v, p_lv, shadow_arr, pressure_arr)
                ohlc_q   = _ohlc_query(t_predict, p_lv, shadow_arr, pressure_arr)
                ohlc_d   = _ohlc_cosine_dist(ohlc_lib, ohlc_q)
                n_o    = min(K, len(pool_v))
                top_o  = np.argpartition(ohlc_d, n_o - 1)[:n_o]
                pool_f = pool_v[top_o]
                return att[pool_f[:, None] + np.arange(p_lv)[None, :] * tau_base], att[pool_f + y_step]

            if not final_pool_all and len(pool_sm) > K:
                n_sel  = min(K, len(pool_sm))
                top_k  = np.argpartition(dists_sm, n_sel - 1)[:n_sel]
                pool_f = pool_sm[top_k]
                return att[pool_f[:, None] + np.arange(p_lv)[None, :] * tau_base], att[pool_f + y_step]

            return X_sm, att[pool_sm + y_step]

        # Промежуточный уровень: геометрическое уменьшение K сверху вниз.
        # Верхние уровни используют более длинные (дискриминативные) векторы →
        # достаточно меньшего K. Нижний уровень получает полный K для регрессии.
        K_l = max(4, K >> (n_lvls - 1 - level_idx))
        n_sel1    = min(2 * K_l if use_ohlc else K_l, len(pool))
        top_i1    = np.argpartition(dists, n_sel1 - 1)[:n_sel1]
        cand_pool = pool[top_i1]
        cand_d    = dists[top_i1]

        if use_ohlc and len(cand_pool) > K_l:
            ohlc_lib   = _ohlc_windows(cand_pool, p_lv, shadow_arr, pressure_arr)
            ohlc_q     = _ohlc_query(t_predict, p_lv, shadow_arr, pressure_arr)
            ohlc_d     = _ohlc_cosine_dist(ohlc_lib, ohlc_q)
            n_sel2     = min(K_l, len(cand_pool))
            top_i2     = np.argpartition(ohlc_d, n_sel2 - 1)[:n_sel2]
            sel_starts = cand_pool[top_i2]
        else:
            n_f        = min(K_l, len(cand_pool))
            top_f      = np.argpartition(cand_d, n_f - 1)[:n_f]
            sel_starts = cand_pool[top_f]

        p_next     = levels[level_idx + 1]
        r          = p_lv - p_next
        offsets    = np.arange(r + 1) * tau_base
        flat       = (sel_starts[:, None] + offsets[None, :]).ravel()
        span_next  = tau_base * (p_next - 1)
        max_s_next = min(n - 1 - span_next, t_predict - 1 - span_next)
        flat       = flat[(flat >= 0) & (flat <= max_s_next)]
        if pool_mask is not None:
            flat = flat[flat < len(pool_mask)]
            flat = flat[pool_mask[flat]]
        pool       = np.unique(flat)
        if len(pool) == 0:
            return None, None

    return None, None

# ═══════════════════════════════════════════════════════════════════════════════
# Каскад τ-октавный (экспериментальный)
# ═══════════════════════════════════════════════════════════════════════════════

def _tau_levels(n_levels: int) -> list[int]:
    """τ_k = 2^(n_levels-1-k). Пример N=4: [8, 4, 2, 1]."""
    return [2 ** (n_levels - 1 - k) for k in range(n_levels)]


def cascade_search_tau(
    att: np.ndarray,
    context: np.ndarray,
    p_fit: int,
    n_levels: int,
    K: int,
    t_predict: int,
    blend_alpha: float = 0.5,
    final_pool_all: bool = True,
    shadow_arr: np.ndarray | None = None,
    pressure_arr: np.ndarray | None = None,
    ohlc_filter_last: bool = False,
    tau_base: int = 1,
    pool_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    τ-октавный каскад: m = p_fit постоянна, шаг задержки τ_k = tau_base * 2^(N-1-k).
    tau_base=1 (по умолчанию): τ ∈ {2^(N-1), ..., 1}.
    tau_base=2: τ ∈ {2^N, ..., 2} — финальный уровень τ=tau_base.

    OHLC применяется только при tau_base=1 на финальном уровне (вектора последовательные).
    """
    n    = len(att)
    taus = [tau_base * t for t in _tau_levels(n_levels)]
    m    = p_fit

    tau_top  = taus[0]
    span_top = tau_top * (m - 1)

    max_start = min(n - 1 - span_top, t_predict - 1 - span_top)
    if max_start < 0 or max_start < K - 1:
        return None, None

    pool     = np.arange(0, max_start + 1)
    if pool_mask is not None:
        _valid = pool[pool < len(pool_mask)]
        pool = _valid[pool_mask[_valid]]
    use_ohlc = shadow_arr is not None and pressure_arr is not None
    K2       = 2 * K

    for level_idx, tau in enumerate(taus):
        span    = tau * (m - 1)
        is_last = (level_idx == len(taus) - 1)

        # Запрос: context[-(span+1)::tau][:m]
        if len(context) < span + 1:
            return None, None
        q = context[-(span + 1)::tau][:m].astype(np.float64)
        if len(q) < m:
            return None, None

        # Ограничить пул: no look-ahead + правая граница att
        max_s = min(n - 1 - span, t_predict - 1 - span)
        pool  = pool[pool <= max_s]
        if len(pool) == 0:
            return None, None

        # Матрица векторов пула: (len(pool), m)
        idx    = pool[:, None] + np.arange(m)[None, :] * tau
        X_pool = att[idx].astype(np.float64)
        dists  = _dists(X_pool, q, blend_alpha)

        if is_last:
            y_step   = span + 1                          # следующий бар после последнего элемента
            valid_y  = pool + y_step < n
            pool_sm  = pool[valid_y]
            X_sm     = X_pool[valid_y]
            dists_sm = dists[valid_y]
            if len(pool_sm) < 2:
                return None, None

            # OHLC только при tau_base=1 (финальный τ=1, вектора последовательные)
            if ohlc_filter_last and use_ohlc and tau_base == 1 and len(pool_sm) > K:
                n_v    = min(K2, len(pool_sm))
                top_v  = np.argpartition(dists_sm, n_v - 1)[:n_v]
                pool_v = pool_sm[top_v]
                ohlc_lib = _ohlc_windows(pool_v, p_fit, shadow_arr, pressure_arr)
                ohlc_q   = _ohlc_query(t_predict, p_fit, shadow_arr, pressure_arr)
                ohlc_d   = _ohlc_cosine_dist(ohlc_lib, ohlc_q)
                n_o    = min(K, len(pool_v))
                top_o  = np.argpartition(ohlc_d, n_o - 1)[:n_o]
                pool_f = pool_v[top_o]
                return att[pool_f[:, None] + np.arange(p_fit)[None, :] * tau_base], att[pool_f + y_step]

            if not final_pool_all and len(pool_sm) > K:
                n_sel  = min(K, len(pool_sm))
                top_k  = np.argpartition(dists_sm, n_sel - 1)[:n_sel]
                pool_f = pool_sm[top_k]
                return att[pool_f[:, None] + np.arange(p_fit)[None, :] * tau_base], att[pool_f + y_step]

            return X_sm, att[pool_sm + y_step]

        # Промежуточный уровень: K лучших по val (OHLC не применяем — разреженные τ)
        n_sel      = min(K, len(pool))
        top_i      = np.argpartition(dists, n_sel - 1)[:n_sel]
        sel_starts = pool[top_i]

        # Переход τ → τ/2: t_start ∈ [t0, t0 + span_next]
        tau_next  = taus[level_idx + 1]
        span_next = tau_next * (m - 1)
        offsets   = np.arange(span_next + 1)
        flat      = (sel_starts[:, None] + offsets[None, :]).ravel()

        max_s_next = min(n - 1 - span_next, t_predict - 1 - span_next)
        flat       = flat[(flat >= 0) & (flat <= max_s_next)]
        if pool_mask is not None:
            flat = flat[flat < len(pool_mask)]
            flat = flat[pool_mask[flat]]
        pool       = np.unique(flat)
        if len(pool) == 0:
            return None, None

    return None, None


def _cascade_dispatch(
    att: np.ndarray,
    context: np.ndarray,
    p_fit: int,
    n_levels: int,
    K: int,
    t_predict: int,
    blend_alpha: float,
    p_cascade_max: int,
    shadow_arr,
    pressure_arr,
    ohlc_filter_last: bool,
    cascade_mode: str,
    final_pool_all: bool,
    tau_base: int = 1,
    pool_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if cascade_mode == "tau":
        return cascade_search_tau(
            att, context, p_fit, n_levels, K, t_predict,
            blend_alpha, final_pool_all,
            shadow_arr, pressure_arr, ohlc_filter_last,
            tau_base, pool_mask,
        )
    return cascade_search(
        att, context, p_fit, n_levels, K, t_predict,
        blend_alpha, p_cascade_max,
        shadow_arr, pressure_arr, ohlc_filter_last,
        final_pool_all, tau_base, pool_mask,
    )


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


def _lwr_step(
    X_pool: np.ndarray, y_pool: np.ndarray, vec_fit: np.ndarray,
    blend_alpha: float, d_proj: int | None,
) -> float:
    """LWR: гауссов kernel, h_bw = max(d). Проекция на подпространство сохраняется."""
    p_eff = min(X_pool.shape[1], len(vec_fit))
    Xf    = X_pool[:, -p_eff:]
    vf    = vec_fit[-p_eff:]
    d_arr = _dists(Xf, vf, blend_alpha)
    h_bw  = max(float(d_arr.max()), 1e-10)
    w     = np.exp(-0.5 * (d_arr / h_bw) ** 2)
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
    fixed_manifold: bool = False,
    use_lp_corr: bool = False,
    shadow_arr: np.ndarray | None = None,
    pressure_arr: np.ndarray | None = None,
    ohlc_filter_last: bool = False,
    cascade_mode: str = "p",
    final_pool_all: bool = True,
    tau_base: int = 1,
    pool_mask: np.ndarray | None = None,
    zone_pool_mask: np.ndarray | None = None,
    zone_slide: bool = False,
    use_lwr: bool = False,
) -> np.ndarray | None:
    """Возвращает fc_preds[horizon] — сырые att-предсказания, или None при ошибке.

    zone_pool_mask — маска зонного уровня: на шаге h модифицируется одним из режимов:
      zone_slide=False (expand): правая граница каждой зоны расширяется на h+1 баров.
      zone_slide=True  (slide):  маска целиком сдвигается вправо на h+1 баров.
    pool_mask — статическая маска (не изменяется), применяется поверх zone_pool_mask.
    """
    n_eff     = origin + 1
    _min_pool = d_proj + 2
    fc_buf    = list(att[:n_eff])

    def _vf(buf: list, p: int) -> np.ndarray:
        """τ-разреженный вектор длины p из конца buf."""
        if tau_base == 1:
            return np.array(buf[-p:], dtype=np.float64)
        span = tau_base * (p - 1)
        return np.array(buf[-(span + 1)::tau_base][:p], dtype=np.float64)

    # LP-коррекция: библиотека m-мерных τ-окон из истории att до origin
    X_lib_lp: np.ndarray | None = None
    if use_lp_corr:
        n_lib = n_eff - tau_base * (p_fit - 1) - 1
        if n_lib > 0:
            X_lib_lp = att[np.arange(n_lib)[:, None] + np.arange(p_fit)[None, :] * tau_base]

    if fixed_manifold:
        X_nn_fix, y_nn_fix = _cascade_dispatch(
            att, att, p_fit, n_levels, K, n_eff,
            blend_alpha, p_cascade_max,
            shadow_arr, pressure_arr, ohlc_filter_last,
            cascade_mode, final_pool_all, tau_base, pool_mask,
        )
        if X_nn_fix is None or len(X_nn_fix) < _min_pool:
            return None

        p_eff  = min(X_nn_fix.shape[1], p_fit)
        Xf     = X_nn_fix[:, -p_eff:]
        nX     = np.linalg.norm(Xf, axis=1)
        use_pip = d_proj < p_eff and len(Xf) > d_proj
        if use_pip:
            center   = Xf.mean(0)
            _, _, Vt = np.linalg.svd(Xf - center, full_matrices=False)
            V        = Vt[:d_proj].T
            Xf_reg   = (Xf - center) @ V
        else:
            center = None; V = None
            Xf_reg = Xf
        A_fixed = np.hstack([np.ones((len(Xf_reg), 1)), Xf_reg])

        for _ in range(horizon):
            vf = _vf(fc_buf, p_eff)

            nq = float(np.linalg.norm(vf))
            if nq < 1e-10:
                d_shape = np.ones(len(Xf))
            else:
                with np.errstate(invalid="ignore", divide="ignore"):
                    sim = np.where(nX > 1e-10, (Xf @ vf) / (nX * nq), 0.0)
                d_shape = 1.0 - sim.clip(-1.0, 1.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                d_amp = np.where(
                    (nX > 1e-10) & (nq > 1e-10),
                    np.abs(np.log(nX / nq)),
                    np.abs(nX - nq),
                )
            max_a = max(float(d_amp.max()), 1e-10)
            max_s = max(float(d_shape.max()), 1e-10)
            d_arr  = blend_alpha * (d_amp / max_a) + (1.0 - blend_alpha) * (d_shape / max_s)

            if use_lwr:
                h_bw = max(float(d_arr.max()), 1e-10)
                sw   = np.sqrt(np.exp(-0.5 * (d_arr / h_bw) ** 2))
            else:
                d_mean = max(float(d_arr.mean()), 1e-10)
                sw     = np.sqrt(np.exp(-smap_theta * d_arr / d_mean))

            vf_reg = (vf - center) @ V if use_pip else vf
            c, _, _, _ = np.linalg.lstsq(sw[:, None] * A_fixed, sw * y_nn_fix, rcond=None)
            pred = float(c[0] + vf_reg @ c[1:])
            if use_lp_corr and X_lib_lp is not None:
                v_m  = np.concatenate([_vf(fc_buf, p_fit)[1:], [pred]])
                pred = _lp_corr(v_m, X_lib_lp, K, d_proj, blend_alpha)
            fc_buf.append(pred)
    else:
        for h in range(horizon):
            t   = n_eff + h
            ctx = np.array(fc_buf)
            if zone_pool_mask is not None:
                _zm_h = (_shift_mask_right(zone_pool_mask, h + 1) if zone_slide
                         else _expand_mask_right(zone_pool_mask, h + 1))
                _pm_h = _zm_h if pool_mask is None else (pool_mask & _zm_h)
            else:
                _pm_h = pool_mask
            X_nn, y_nn = _cascade_dispatch(
                att, ctx, p_fit, n_levels, K, t,
                blend_alpha, p_cascade_max,
                shadow_arr, pressure_arr, ohlc_filter_last,
                cascade_mode, final_pool_all, tau_base, _pm_h,
            )
            if X_nn is None or len(X_nn) < _min_pool:
                return None
            pred = (_lwr_step(X_nn, y_nn, _vf(fc_buf, p_fit), blend_alpha, d_proj)
                    if use_lwr else
                    _smap_step(X_nn, y_nn, _vf(fc_buf, p_fit), smap_theta, blend_alpha, d_proj))
            if use_lp_corr and X_lib_lp is not None:
                v_m  = np.concatenate([_vf(fc_buf, p_fit)[1:], [pred]])
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
    fixed_manifold: bool = False,
    use_lp_corr: bool = False,
    shadow_arr: np.ndarray | None = None,
    pressure_arr: np.ndarray | None = None,
    ohlc_filter_last: bool = False,
    cascade_mode: str = "p",
    final_pool_all: bool = True,
    tau_base: int = 1,
    pool_mask: np.ndarray | None = None,
    use_zone_level: bool = False,
    zone_addend: int = 3,
    zone_overlap_pct: int = 0,
    ohlc_zone_ratios: tuple | None = None,
    zone_slide: bool = False,
    use_lwr: bool = False,
    zone_pool_mask_global: np.ndarray | None = None,
    use_cluster_level: bool = False,
    n_clusters: int = 5,
    n_pca_cluster: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Возвращает (results, failed). results содержит fc_preds (сырые att).

    zone_pool_mask_global — предвычисленная глобальная зонная маска (обходит per-d пересчёт).
    При наличии передаётся в run_forecast как zone_pool_mask (slide/expand работают).
    """
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

        att = _lp_proj_ratio_cached(ratio.tobytes(), p_fit, d, K, 1, blend_alpha, tau_base)

        if cascade_mode == "tau":
            span_top = tau_base * _tau_levels(n_levels)[0] * (p_fit - 1)
            min_len  = span_top + p_fit + 1
        else:
            eff_levels = _cascade_levels(p_fit, n_levels, p_cascade_max)
            min_len    = tau_base * (eff_levels[0] - 1) + p_fit + 1
        if len(att) < min_len:
            failed.append({"d": d, "reason": f"ряд мал для каскада (нужно {min_len})"})
            continue

        # Зонная маска: глобальная (предвычислена снаружи) или per-d (пересчёт здесь)
        if zone_pool_mask_global is not None:
            _zmask = zone_pool_mask_global
        elif use_zone_level:
            _k_zone = max(2, K >> (n_levels - 1))
            _zmask = sliding_zone_mask(ratio, origin, p_fit + zone_addend, _k_zone, ohlc_zone_ratios, zone_overlap_pct)
        else:
            _zmask = None

        # Кластерная маска фазового пространства (per-d, работает на att)
        _cmask = regime_cluster_mask(att, origin, p_fit, n_clusters, n_pca_cluster) if use_cluster_level else None

        # Объединяем маски: pool_mask (статическая) & зонная & кластерная
        _pm = pool_mask
        if _cmask is not None:
            _pm = _cmask if _pm is None else (_pm & _cmask)

        fc_preds = run_forecast(
            att, p_fit, n_levels, K, horizon, origin,
            blend_alpha, smap_theta, d_proj, p_cascade_max, fixed_manifold,
            use_lp_corr, shadow_arr, pressure_arr, ohlc_filter_last,
            cascade_mode, final_pool_all, tau_base,
            pool_mask=_pm,
            zone_pool_mask=_zmask,
            zone_slide=zone_slide,
            use_lwr=use_lwr,
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
# Итеративный уточняющий прогноз
# ═══════════════════════════════════════════════════════════════════════════════

def _smap_fixed_horizon(
    X_nn: np.ndarray, y_nn: np.ndarray,
    att_context: np.ndarray,
    horizon: int, p_fit: int, d_proj: int,
    smap_theta: float, blend_alpha: float,
    use_lp_corr: bool = False, K: int = 0,
    tau_base: int = 1,
    use_lwr: bool = False,
) -> np.ndarray | None:
    """S-map / LWR по фиксированному пулу X_nn/y_nn на horizon шагов вперёд.
    att_context — реальная история att[:n_eff]; fc_buf инициализируется от неё."""
    n_eff = len(att_context)
    if X_nn is None or len(X_nn) < d_proj + 2:
        return None

    def _vf(buf: list, p: int) -> np.ndarray:
        if tau_base == 1:
            return np.array(buf[-p:], dtype=np.float64)
        span = tau_base * (p - 1)
        return np.array(buf[-(span + 1)::tau_base][:p], dtype=np.float64)

    X_lib_lp = None
    if use_lp_corr:
        n_lib = n_eff - tau_base * (p_fit - 1) - 1
        if n_lib > 0:
            X_lib_lp = att_context[np.arange(n_lib)[:, None] + np.arange(p_fit)[None, :] * tau_base]

    p_eff  = min(X_nn.shape[1], p_fit)
    Xf     = X_nn[:, -p_eff:]
    nX     = np.linalg.norm(Xf, axis=1)
    use_pip = d_proj < p_eff and len(Xf) > d_proj
    if use_pip:
        center   = Xf.mean(0)
        _, _, Vt = np.linalg.svd(Xf - center, full_matrices=False)
        V        = Vt[:d_proj].T
        Xf_reg   = (Xf - center) @ V
    else:
        center = None; V = None; Xf_reg = Xf
    A_fixed = np.hstack([np.ones((len(Xf_reg), 1)), Xf_reg])

    fc_buf = list(att_context)
    for _ in range(horizon):
        vf  = _vf(fc_buf, p_eff)
        nq  = float(np.linalg.norm(vf))
        if nq < 1e-10:
            d_shape = np.ones(len(Xf))
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                sim = np.where(nX > 1e-10, (Xf @ vf) / (nX * nq), 0.0)
            d_shape = 1.0 - sim.clip(-1.0, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            d_amp = np.where(
                (nX > 1e-10) & (nq > 1e-10),
                np.abs(np.log(nX / nq)), np.abs(nX - nq),
            )
        max_a = max(float(d_amp.max()), 1e-10)
        max_s = max(float(d_shape.max()), 1e-10)
        d_arr = blend_alpha*(d_amp/max_a) + (1-blend_alpha)*(d_shape/max_s)
        if use_lwr:
            h_bw = max(float(d_arr.max()), 1e-10)
            sw   = np.sqrt(np.exp(-0.5 * (d_arr / h_bw) ** 2))
        else:
            d_mean = max(float(d_arr.mean()), 1e-10)
            sw     = np.sqrt(np.exp(-smap_theta * d_arr / d_mean))
        vf_reg = (vf - center) @ V if use_pip else vf
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A_fixed, sw * y_nn, rcond=None)
        pred = float(c[0] + vf_reg @ c[1:])
        if use_lp_corr and X_lib_lp is not None:
            v_m  = np.concatenate([_vf(fc_buf, p_fit)[1:], [pred]])
            pred = _lp_corr(v_m, X_lib_lp, K, d_proj, blend_alpha)
        fc_buf.append(pred)

    return np.array(fc_buf[n_eff:])


def run_forecast_refined(
    ratio: np.ndarray,
    d_values: list[int],
    xy_x: int, xy_y: int, xi_add: int,
    n_levels: int, horizon: int, origin: int,
    n_refine: int = 3,
    blend_alpha: float = 0.5,
    smap_theta: float = 20.0,
    p_cascade_max: int = 1500,
    use_lp_corr: bool = False,
    shadow_arr=None, pressure_arr=None, ohlc_filter_last: bool = False,
    cascade_mode: str = "p",
    final_pool_all: bool = True,
    tau_base: int = 1,
    progress_cb=None,
    pool_mask: np.ndarray | None = None,
    use_zone_level: bool = False,
    zone_addend: int = 3,
    zone_overlap_pct: int = 0,
    ohlc_zone_ratios: tuple | None = None,
    use_lwr: bool = False,
    use_cluster_level: bool = False,
    n_clusters: int = 5,
    n_pca_cluster: int = 0,
) -> tuple[np.ndarray | None, list[dict]]:
    """Итеративный уточняющий прогноз.

    Раунд 0 : контекст каскада = att_d (реальная история).
    Раунд r>0: контекст = concat([att_d, avg_fc_{r-1}]),
               но библиотека кандидатов — только реальная история (t_predict = n_eff).
    Выход: avg_fc после n_refine раундов.
    """
    n_eff = origin + 1

    # Предвычисление att для каждого d
    atts: dict[int, np.ndarray] = {}
    for d in d_values:
        p_fit = xy_x * d + xy_y
        K     = 3 * (p_fit + 1) + xi_add
        if p_fit < 2:
            continue
        att = _lp_proj_ratio_cached(ratio.tobytes(), p_fit, d, K, 1, blend_alpha, tau_base)
        if cascade_mode == "tau":
            span_top = tau_base * _tau_levels(n_levels)[0] * (p_fit - 1)
            min_len  = span_top + p_fit + 1
        else:
            eff_lvls = _cascade_levels(p_fit, n_levels, p_cascade_max)
            min_len  = tau_base * (eff_lvls[0] - 1) + p_fit + 1
        if len(att) < min_len:
            continue
        atts[d] = att

    if not atts:
        return None, []

    # Зонные маски per-d (вычисляем один раз до раундов, кэшируем по d)
    zone_masks: dict[int, np.ndarray | None] = {}
    if use_zone_level:
        for d in atts:
            p_fit = xy_x * d + xy_y
            K      = 3 * (p_fit + 1) + xi_add
            _k_zone = max(2, K >> (n_levels - 1))
            _zmask = sliding_zone_mask(ratio, origin, p_fit + zone_addend, _k_zone, ohlc_zone_ratios, zone_overlap_pct)
            zone_masks[d] = _zmask if _zmask is not None else None

    # Кластерные маски per-d (фазовое пространство att; вычисляем один раз до раундов)
    cluster_masks: dict[int, np.ndarray | None] = {}
    if use_cluster_level:
        for d, att_d in atts.items():
            p_fit = xy_x * d + xy_y
            cluster_masks[d] = regime_cluster_mask(att_d, origin, p_fit, n_clusters, n_pca_cluster)

    def _combined_mask(d: int) -> np.ndarray | None:
        base = pool_mask
        if use_zone_level:
            zm = zone_masks.get(d)
            if zm is not None:
                base = zm if base is None else (base & zm)
        if use_cluster_level:
            cm = cluster_masks.get(d)
            if cm is not None:
                base = cm if base is None else (base & cm)
        return base

    avg_fc: np.ndarray | None = None
    rounds_meta: list[dict]   = []
    n_ds = len(atts)

    for r in range(n_refine):
        round_preds: list[np.ndarray] = []
        round_ds: list[int]           = []

        for d_idx, (d, att_d) in enumerate(atts.items()):
            if progress_cb:
                progress_cb(r * n_ds + d_idx, n_refine * n_ds, d)

            p_fit  = xy_x * d + xy_y
            K      = 3 * (p_fit + 1) + xi_add

            # Контекст запроса: реальная история + прогноз предыдущего раунда
            context = att_d if avg_fc is None else np.concatenate([att_d, avg_fc])

            X_nn, y_nn = _cascade_dispatch(
                att_d, context, p_fit, n_levels, K, n_eff,
                blend_alpha, p_cascade_max,
                shadow_arr, pressure_arr, ohlc_filter_last,
                cascade_mode, final_pool_all, tau_base, _combined_mask(d),
            )

            fc = _smap_fixed_horizon(
                X_nn, y_nn, att_d[:n_eff], horizon, p_fit, d, smap_theta,
                blend_alpha, use_lp_corr, K, tau_base,
                use_lwr=use_lwr,
            )
            if fc is not None:
                round_preds.append(fc)
                round_ds.append(d)

        if not round_preds:
            break

        avg_fc = np.mean(round_preds, axis=0)
        rounds_meta.append({"round": r + 1, "n_valid": len(round_preds), "ds": round_ds})

    return avg_fc, rounds_meta

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
            "| Нормализация | **log(price)**: ratio = log(close), без detrending |\n"
            "| Фильтрация | **LP-фильтр** (Local Projective Filter): каузальная SVD-проекция ratio на локальное d-мерное касательное многообразие, построенное по K ближайшим соседям |\n"
            "| Оценка d | **PCA sweep**: SVD облака K соседей → локальная размерность аттрактора |\n"
            "| Поиск соседей | **Каскадный поиск**: иерархический coarse-to-fine NN в пространстве задержек |\n"
            "| Прогноз | **S-map** (Sugihara, 1994): локально-взвешенная авторегрессия с PCA-проекцией пула в d-мерное подпространство перед МНК |\n\n"
            "---\n\n"
            "**Обозначения**\n\n"
            "| Символ | Описание |\n"
            "|:------:|----------|\n"
            "| ratio | log(close) — логарифм цены, без detrending |\n"
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
            "| origin | точка отсчёта: att[t≤origin] — история, att[t>origin] — прогноз |\n\n"
            "---\n\n"
            "**Схема пайплайна**\n\n"
            "```\n"
            "close → ratio = log(close)\n"
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
            "    price_d = exp(âtt)              ← реконструкция\n"
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
    tau_base = st.slider("τ (глобальный шаг задержки)", 1, 8, 1,
                         help="Глобальный шаг τ для всего тракта синхронно:\n"
                              "  LP-фильтр, каскадный поиск, S-map, LP-коррекция.\n"
                              "τ=1 (по умолчанию): стандартные последовательные задержки.\n"
                              "τ>1: разреженные векторы {att[t], att[t+τ], …, att[t+τ(m−1)]}.\n"
                              "Охват вектора = τ(m−1) баров.\n"
                              "Тестовый стент: управлять вручную.\n"
                              "OHLC-фильтр автоматически отключается при τ>1.")
    if tau_base > 1:
        st.caption(f"τ={tau_base}  ·  охват d=5: {tau_base*(xy_x*5+xy_y-1)}б  "
                   f"·  охват d=20: {tau_base*(xy_x*20+xy_y-1)}б  "
                   f"·  OHLC отключён")

    st.divider()
    st.subheader("Каскадный поиск")
    cascade_mode_label = st.radio(
        "Режим каскада",
        options=["p-октавный", "τ-октавный"],
        index=0,
        horizontal=True,
        help="**p-октавный** (текущий): размерность вектора растёт с уровнем — "
             "m·2ⁿ⁻¹, …, m (τ=1 везде). Суб-векторы — непрерывные срезы соседа.\n\n"
             "**τ-октавный** (эксп.): размерность m постоянна, шаг задержки убывает — "
             "τ=2ⁿ⁻¹, …, 1. Охват уровня τ: τ(m−1) баров. "
             "Векторы следующего уровня строятся из att напрямую.",
    )
    cascade_mode = "tau" if cascade_mode_label == "τ-октавный" else "p"

    final_pool_all = st.checkbox(
        "Весь финальный пул → S-map",
        value=False,
        help="Вкл (по умолчанию): последний уровень каскада передаёт весь найденный пул "
             "в S-map без дополнительной отсечки. S-map сам взвешивает соседей через θ.\n\n"
             "Выкл: из финального пула отбирается K ближайших по ρ, и только они "
             "передаются в S-map. Более агрессивный отбор — меньше шума в пуле, "
             "но выше риск потерять разнообразие.",
    )

    st.caption(
        "Иерархический (coarse-to-fine) поиск в пространстве задержек. "
        "n уровней (октавный ×2):  \n\n"
        "**p-режим**: m₀=m·2ⁿ⁻¹, …, mₙ=m; τ=1 везде  \n"
        "**τ-режим**: m=const; τ₀=2ⁿ⁻¹, …, τₙ=1; охват τ·(m−1) баров  \n\n"
        "На промежуточных уровнях отбирается K ближайших,  \n"
        "их позиции расширяются в пул следующего уровня.  \n"
        "Уровни p-режима с mₖ > m(max) снимаются сверху."
    )
    n_levels = st.slider("Уровней каскада", 1, 6, 2,
                         help="Число уровней n.\n"
                              "p-режим: окна m·2ⁿ⁻¹, …, m.\n"
                              "τ-режим: шаги задержки 2ⁿ⁻¹, …, 1.\n"
                              "n=1: одноуровневый поиск.")
    p_cascade_max = st.slider("Макс. окно каскада m(max)", 100, 10000, 2000, 100,
                              help="Только для p-режима: уровни с mₖ > m(max) удаляются.\n"
                                   "В τ-режиме не используется (охват = τ·(m−1)).",
                              disabled=(cascade_mode == "tau"))
    bars = st.slider("Точек в библиотеке", 100, 10000, 3000, 100,
                     help="Глубина поиска соседей: размер скользящего окна библиотеки.\n"
                          "Ограничивает число исторических кандидатов на каждом уровне.")
    _ex_p = xy_x * 20 + xy_y
    if cascade_mode == "tau":
        _ex_taus = _tau_levels(n_levels)
        _ex_spans = [f"τ={t} → охват={t*(_ex_p-1)}б" for t in _ex_taus]
        st.caption(f"Пример d=20: m={_ex_p},  τ-уровни={_ex_taus}  "
                   f"({len(_ex_taus)} ур.)  ·  {_ex_spans[0]}")
    else:
        _ex_lvls = _cascade_levels(_ex_p, n_levels, p_cascade_max)
        st.caption(f"Пример d=20: m={_ex_p},  уровни={_ex_lvls}  ({len(_ex_lvls)} ур.)")

    st.divider()
    st.subheader("Зонный глобальный уровень")
    st.caption(
        "PIP-сегментация ratio → зоны {наклон, волатильность}.  \n"
        "Каскад ищет соседей только среди K ближайших исторических зон origin-зоны.  \n"
        "min_zone = p×2^(N−1), n_зон = N_баров / min_zone + 1 (максимум при ограничении).  \n"
        "Маска пересчитывается per-d (p_fit меняется → другой min_zone и число зон)."
    )
    use_zone_level = st.checkbox("Включить зонный уровень", value=False, key="use_zone_level")
    zone_addend = st.slider(
        "Добавка W (zone)", 0, 30, 3, key="zone_addend",
        help="W = p_fit + добавка. Размер окна режима чуть больше вектора задержки.\n"
             "K зон = K каскада (те же 3(m+1)+ξ соседей, что и в S-map).",
        disabled=not use_zone_level,
    )
    zone_overlap_pct = st.slider(
        "Перекрытие зон (% от W)", 0, 100, 0, step=5, key="zone_overlap_pct",
        help="Допустимое перекрытие выбранных зон в процентах от размера окна W.\n"
             "0% = зоны не пересекаются.\n"
             "50% = зоны могут делить до W/2 баров.\n"
             "100% = ограничений нет (поведение как у обычного KNN).",
        disabled=not use_zone_level,
    )
    zone_slide = st.checkbox(
        "Скользящая маска (slide)", value=True, key="zone_slide",
        help="Expand (по умолчанию): правая граница зоны расширяется на h+1 баров.  \n"
             "Slide: маска целиком сдвигается вправо на h+1 (левая граница тоже движется).",
        disabled=not use_zone_level,
    )

    st.divider()
    st.subheader("Кластеризация фазового пространства")
    st.caption(
        "KMeans на матрице задержек att (размерность p_fit). "
        "Каскад ищет соседей только среди векторов из того же кластера, что и origin-вектор.  \n"
        "Маска per-d: кластеры пересчитываются для каждого p_fit.  \n"
        "PCA=0 — без редукции; PCA>0 — поиск в PCA-пространстве (ускоряет при большом p_fit)."
    )
    use_cluster_level = st.checkbox("Включить кластеризацию ФП", value=True, key="use_cluster_level")
    n_clusters = st.slider(
        "K кластеров", 2, 20, 8, key="n_clusters",
        help="Число кластеров KMeans. Рекомендуется 3–8.",
        disabled=not use_cluster_level,
    )
    n_pca_cluster = st.slider(
        "PCA компоненты (0 = нет)", 0, 20, 8, key="n_pca_cluster",
        help="Если > 0 — PCA-редукция перед KMeans. "
             "Полезно при больших p_fit (уменьшает шум, ускоряет кластеризацию).",
        disabled=not use_cluster_level,
    )

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
    pca_p_range = st.slider("Диапазон m", 3, 300, (3, 150), key="pca_p_range",
                             help="Диапазон окна задержки для sweep по m.\n"
                                  "Широкий диапазон → надёжнее средняя оценка d, "
                                  "но дольше вычисляется.")
    pca_tau = st.slider("τ (шаг задержки PCA)", 1, 8, 1,
                        help="Шаг задержки в векторах при измерении PCA.\n"
                             "τ=1 (по умолчанию): вектор {ratio[t], ratio[t+1], ..., ratio[t+m−1]} — стандартные последовательные задержки.\n"
                             "τ>1: вектор {ratio[t], ratio[t+τ], ..., ratio[t+τ(m−1)]} — разреженные задержки, охват = τ(m−1) баров.\n"
                             "Применяется только при нажатии «Измерить PCA»; на прогноз не влияет.")
    pca_thr_range = st.slider("Диапазон порогов", 0.80, 0.99, (0.98, 0.99), 0.01,
                              key="pca_thr_range",
                              help="Пороги τ для доли объяснённой дисперсии.\n"
                                   "Левый τ₁ → d_min (нижняя оценка размерности).\n"
                                   "Правый τ₂ → d_max (верхняя оценка).\n"
                                   "Типичные пары: 0.90/0.95 (широкий диапазон d) "
                                   "или 0.95/0.99 (узкий).")
    pca_thr1, pca_thr2 = pca_thr_range[0], pca_thr_range[1]
    _n_pts = pca_p_range[1] - pca_p_range[0] + 1
    _pca_span_ex = pca_tau * (pca_p_range[1] - 1)
    st.caption(
        f"{_n_pts} точек  ·  K = 3·(m+1)+{xi_add}  ·  τ={pca_tau}  ·  "
        f"охват при m_max: {_pca_span_ex}б  ·  "
        f"d_min = ⌊mean d(m,τ₁={pca_thr1})⌋  ·  d_max = ⌈mean d(m,τ₂={pca_thr2})⌉"
    )
    pca_btn = st.button("Измерить PCA", use_container_width=True)

    st.divider()
    st.subheader("Прогноз")
    st.caption(
        "Итеративный пошаговый прогноз (iterated one-step-ahead) аттрактора att.  \n"
        "Для каждого d ∈ [d_min, d_max + Δd], m = x·d + y:  \n\n"
        "  1. LP-фильтр: att = LP(ratio, m, d)  \n"
        "  2. Итерация h = 1 … H:  \n"
        "     a. Каскадный поиск соседей: x_t = att[t−m : t]  \n"
        "     b. PCA-проекция пула в d-мерное подпространство (то же d, что в LP):  \n"
        "        X̃ = (X − μ) · V_d,   V_d — d правых сингулярных векторов SVD  \n"
        "     c. S-map (Sugihara, 1994) в проецированном пространстве:  \n"
        "        wᵢ = exp(−θ · ρ(x̃ᵢ, x̃_t) / ρ̄),   взвешенный МНК → âtt[t+h]  \n"
        "     d. âtt[t+h] добавляется в контекст следующего шага  \n"
        "  3. Реконструкция: price[t] = exp(âtt[t])  \n\n"
        "Результаты по d усредняются (простое среднее, без взвешивания) → финальный прогноз."
    )
    d_add = st.slider("Δd (добавка к d_max)", 0, 20, 0,
                      help="Расширяет верхнюю границу диапазона размерностей:\n"
                           "  d_values = [d_min … d_max + Δd]\n"
                           "Позволяет включить в ансамбль бо́льшие d,\n"
                           "чем даёт автоматический PCA sweep.\n"
                           "Δd=0 — диапазон определяется только измерителем.")
    horizon = st.slider("Горизонт (баров)", 1, 200, 40,
                        help="Число шагов вперёд H.\n"
                             "Каждый шаг — итерация S-map, предыдущее предсказание "
                             "добавляется в контекст (iterated one-step-ahead).")
    origin_offset = st.slider("Точка отсчёта (баров от конца)", 0, 500, 0,
                              help="Сдвиг точки отсчёта назад от последнего бара.\n"
                                   "0 = прогноз из последней доступной точки.\n"
                                   ">0 = проверка на исторических данных "
                                   "(фактические цены видны на графике как 'actual').")
    _regressor = st.radio(
        "Регрессор", ["S-map", "LWR"], index=0, horizontal=True, key="regressor",
        help="S-map: exp(−θ·d/d̄), параметр θ.\n"
             "LWR: гауссов kernel exp(−0.5·(d/d_max)²), без параметров.",
    )
    use_lwr = (_regressor == "LWR")
    smap_theta = st.slider("θ (S-map)", 0.0, 50.0, 0.5, 0.5,
                           disabled=use_lwr,
                           help="Параметр нелинейности S-map (Sugihara, 1994).\n"
                                "Веса: wᵢ = exp(−θ · ρ(x̃ᵢ, x̃_t) / ρ̄)\n"
                                "θ = 0: равные веса → глобальная линейная авторегрессия (AR)\n"
                                "θ → ∞: только ближайший сосед → сильная локальная нелинейность\n"
                                "Оптимум для финансовых рядов: θ ≈ 5…20.")
    fixed_manifold = st.checkbox(
        "Фиксированное многообразие",
        value=False,
        help="Каскадный поиск выполняется один раз для t=origin+1.\n"
             "S-map итерирует все шаги горизонта в зафиксированном пуле соседей.\n"
             "Быстрее; пул не дрейфует при накоплении предсказаний.",
    )
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
    use_refine = st.checkbox(
        "Итеративное уточнение",
        value=False,
        help="Заменяет обычный прогноз итеративной схемой:\n"
             "  Раунд 0: cascade(att) → S-map → avg по d\n"
             "  Раунд r: cascade([att, avg_{r-1}]) → S-map → avg по d\n"
             "Библиотека кандидатов всегда только реальная история.\n"
             "Контекст запроса в раунде r включает усреднённый прогноз раунда r-1.\n"
             "Выход — единственный прогноз (avg после N раундов).",
    )
    n_refine = st.slider("Раундов", 1, 10, 3,
                         help="Число итераций уточнения.\n"
                              "Раунд 1 ≡ обычный fixed-manifold прогноз с усреднением по d.",
                         disabled=not use_refine)
    use_ohlc_cascade = st.checkbox(
        "OHLC-фильтр в каскаде",
        value=False,
        help="Двухступенчатый отбор соседей на каждом уровне каскада:\n"
             "  1. 2K ближайших по amp_cos(val-окно)\n"
             "  2. K ближайших среди них по cosine(OHLC-окно)\n"
             "OHLC-окно: [shadow, pressure] той же длины, что val-окно уровня.\n"
             "  shadow   = (high−low)/close  — ширина тени\n"
             "  pressure = (close−low)/(high−low)  — позиция закрытия в баре\n"
             "Прогноз S-map использует только val."
    )
    ohlc_filter_last = st.checkbox(
        "OHLC-фильтр на последнем уровне",
        value=False,
        help="Применять ли OHLC-отбор на последнем уровне каскада.\n"
             "Вкл: пул S-map ограничен K соседями по OHLC (агрессивнее).\n"
             "Выкл: последний уровень отдаёт весь val-пул в S-map (мягче).",
        disabled=not use_ohlc_cascade,
    )
    run_btn = st.button("▶  Прогноз", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Предобработка")
    use_midprice = st.checkbox(
        "Входная цена (open+close)/2",
        value=True,
        help="Входная цена = среднее open и close вместо close.\n"
             "Снижает внутрибарный шум перед логтрендом и LP-фильтром.\n"
             "Прогноз реконструируется в той же шкале — привязка к mid, а не close.",
    )
    smooth_alpha = st.slider(
        "Сглаживание α (вес текущего бара)",
        min_value=0.50, max_value=1.00, value=1.00, step=0.05,
        help="Каузальное взвешенное сглаживание на ratio (после SSA):\n"
             "  ratio[t] ← α·ratio[t] + (1−α)·mean(ratio[t−W+1..t−1])\n"
             "α=1.0 — без сглаживания; W=2 → ratio[t−1] (прежнее поведение).\n"
             "Применяется после SSA.",
    )
    smooth_win = st.slider(
        "Сглаживание W (окно)",
        min_value=2, max_value=20, value=2, step=1,
        disabled=smooth_alpha >= 1.0,
        help="Число баров для усреднения прошлого: mean(ratio[t−W+1..t−1]).\n"
             "W=2 → один предыдущий бар (исходное поведение).",
    )
    use_kf = st.checkbox(
        "Фильтр Калмана (KF2D)",
        value=False,
        help="2D constant-velocity Kalman поверх входной цены (после α-сглаживания).\n"
             "State = [level, slope]; каузальный; устойчивый (steady-state gain).\n"
             "q — шум процесса (относительно σ цены): меньше → плавнее, больше → отзывчивее.",
    )
    kf_q = st.slider(
        "KF q (шум процесса)",
        min_value=0.01, max_value=2.0, value=0.4, step=0.01,
        disabled=not use_kf,
        help="q_slope = q · σ(Δprice). Диапазон: 0.01 (очень гладко) … 2.0 (почти без фильтрации).",
    )
    use_zigzag = st.checkbox("ZigZag на ratio", value=False, key="use_zigzag",
        help="Каузальный ZigZag: заменяет отрезки между подтверждёнными пивотами "
             "линейной интерполяцией.\nПрименяется до SSA.")
    zz_threshold = st.slider("ZigZag порог (%)", 0.5, 10.0, 2.0, 0.5,
        key="zz_threshold", format="%.1f%%",
        disabled=not use_zigzag,
        help="Минимальный разворот (% от значения пивота) для подтверждения нового экстремума.")

    use_ssa = st.checkbox(
        "SSA на ratio",
        value=True,
        help="Rolling causal SSA на нормализованном ratio (после логтренда, перед LP-фильтром).\n"
             "Каузальный: на каждом t использует только ratio[t−W+1..t].\n"
             "W — длина скользящего окна, L — длина вложения, k — число компонент.\n"
             "Меньше k → агрессивнее сглаживание. Кэшируется.",
    )
    ssa_W = st.select_slider("SSA W (окно)", options=[64, 128, 256, 512], value=256,
                             disabled=not use_ssa,
                             help="Длина скользящего окна ratio для SSA.")
    ssa_L = st.select_slider("SSA L (вложение)", options=[8, 16, 32, 64], value=8,
                             disabled=not use_ssa,
                             help="Длина строки траекторной матрицы. L < W/2.")
    ssa_k = st.slider("SSA k (компоненты)", 1, 8, 2, 1,
                      disabled=not use_ssa,
                      help="Число ведущих SVD-компонент для реконструкции. k=1-2 — максимальное сглаживание.\n"
                           "В sweep-режиме — верхняя граница: запускается k=1,2,…,ssa_k, результаты усредняются.")
    use_ssa_sweep = st.checkbox(
        "SSA sweep (k=1..ssa_k)", value=False, key="use_ssa_sweep",
        disabled=not use_ssa,
        help="Запустить прогноз для каждого k ∈ [1, ssa_k] и усреднить результаты.\n"
             "Этап 1 (PCA sweep) и Этап 2 прогоняются независимо для каждого k.",
    )

    st.subheader("Коррекция")
    delta_ratio = st.slider("Поправка тренда (Δratio/бар)", -0.005, 0.005, 0.0, 0.0001,
                            format="%.4f",
                            help="Линейная коррекция в пространстве ratio (log-price):\n"
                                 "  âtt[j] ← âtt[j] + j·δ\n"
                                 "перед восстановлением цены price = exp(âtt).\n"
                                 "Компенсирует систематическое угловое отклонение "
                                 "прогноза при длинных трендах.\n"
                                 "δ < 0 — сдвиг вниз, δ > 0 — вверх.\n"
                                 "Не пересчитывает прогноз — применяется при отрисовке.")
    if delta_ratio != 0.0:
        st.caption(f"Суммарная поправка за горизонт: {delta_ratio * horizon * 100:+.2f}% ratio")

# ── Загрузка данных ───────────────────────────────────────────────────────────

data = _load_candles(ticker, interval)
if not data and ticker == "LORENZ":
    data = _generate_lorenz_candles()
if not data:
    st.info("Нет данных. Нажмите «Обновить данные» в сайдбаре.\n\nДля теста введите тикер **LORENZ** — синтетический хаотический ряд (система Лоренца).")
    st.stop()

times, close, open_, high, low = _to_arrays(data)
price_input = (open_ + close) / 2 if use_midprice else close
if use_kf:
    price_input = _kf2d_smooth(price_input, kf_q)
ratio  = _log_price(price_input)                        # log(price), без detrending
n      = len(close)
origin = max(0, min(n - 1, n - 1 - origin_offset))

# OHLC-признаки (полный массив, затем обрезаются аналогично ratio)
_eps           = 1e-10
_shadow_full   = (high - low) / np.maximum(close, _eps)
_hl_range      = np.maximum(high - low, _eps)
_pressure_full = (close - low) / _hl_range
# ── Гарантия причинности ──────────────────────────────────────────────────────
# ratio обрезается с двух сторон: будущее (справа) и глубина истории (слева).
# close / times — полные, только для визуализации.
ratio       = ratio[:origin + 1]                        # убираем будущее
if bars > 0:
    ratio   = ratio[max(0, len(ratio) - bars):]         # ограничиваем глубину истории
# Отсюда: len(ratio) = min(bars, origin+1), последний элемент ≡ origin.
# Все функции алгоритма получают только этот массив — никакого look-ahead.
origin_algo = len(ratio) - 1                            # origin внутри обрезанного ratio

# ── Зонный уровень (display + глобальная маска для p_fit_max = xy_x*d_max+xy_y) ─
_pip_zones_display: list[dict] | None = None
_pip_trim_start    = origin + 1 - len(ratio)

# OHLC-ratios для зонного уровня: все 4 компонента в log-пространстве.
# Вычисляем здесь (до SSA/alpha), т.к. OHLC не сглаживается; длина ratio не меняется.
_ohlc_zone_ratios: tuple | None = None
if use_zone_level:
    _ohlc_zone_ratios = (
        _log_price(open_ [_pip_trim_start : origin + 1]),
        _log_price(high  [_pip_trim_start : origin + 1]),
        _log_price(low   [_pip_trim_start : origin + 1]),
        _log_price(close [_pip_trim_start : origin + 1]),
    )

if use_zone_level:
    # Для отображения — reference p_fit: берём d_max из session_state (если есть)
    _d_ref = st.session_state.get("s1_d_max", None)
    if _d_ref is not None:
        _p_ref  = xy_x * _d_ref + xy_y
        _W_disp = _p_ref + zone_addend
        _K_disp  = max(2, (3 * (_p_ref + 1) + xi_add) >> (n_levels - 1))
        _n_windows = max(0, origin_algo - 2 * _W_disp + 2)
        _, _pip_zones_display = _sliding_zone_search(
            ratio, origin_algo, _W_disp, _K_disp,
            ohlc_ratios=_ohlc_zone_ratios,
            zone_overlap_pct=zone_overlap_pct,
        )
        _n_sel = sum(1 for z in _pip_zones_display if z['selected'] and z['b'] < origin_algo)
        st.sidebar.caption(
            f"Зонный уровень (d_max={_d_ref}, p_fit={_p_ref}):  \n"
            f"W={_W_disp}б · K={_K_disp} · Theiler={_W_disp}б · {_n_windows} окон · {_n_sel} выбрано"
        )
    else:
        st.sidebar.caption("Зонный уровень: запустите PCA для отображения зон.")

# ZigZag на ratio (до SSA)
if use_zigzag:
    ratio = _zigzag_ratio(ratio, zz_threshold / 100.0)

# Сохраняем ratio до SSA для sweep-режима (разные k используют один и тот же базовый ряд)
_ratio_pre_ssa = ratio.copy()

# SSA на ratio (каузально: ratio уже обрезан до origin+1, будущего нет)
if use_ssa:
    with st.spinner("SSA на ratio (кэшируется)…"):
        ratio = _ssa_smooth_ratio(ratio.tobytes(), ssa_W, ssa_L, ssa_k)

# α-сглаживание на ratio (после SSA — убирает остаточные зубья)
if smooth_alpha < 1.0:
    _r_orig = ratio.copy()
    for t in range(1, len(ratio)):
        past = _r_orig[max(0, t - smooth_win + 1) : t]
        ratio[t] = smooth_alpha * _r_orig[t] + (1 - smooth_alpha) * past.mean()

# OHLC-признаки обрезаются идентично ratio
_trim_start = origin + 1 - len(ratio)                  # сколько снято слева
shadow_feat   = _shadow_full  [_trim_start : origin + 1]
pressure_feat = _pressure_full[_trim_start : origin + 1]

# Итоговый предобработанный ряд в пространстве цен (для отображения).
# exp(ratio) отражает все шаги: mid, α, KF, SSA (ratio = log(price) в этом пространстве).
_any_preproc = use_midprice or smooth_alpha < 1.0 or use_kf or use_zigzag or use_ssa
if _any_preproc:
    _prep_price_full = np.full(n, np.nan)
    _prep_price_full[_trim_start : origin + 1] = np.exp(ratio)
# ─────────────────────────────────────────────────────────────────────────────

# ── Измерение PCA (только этап 1) ────────────────────────────────────────────

def _run_stage1(ratio, pca_p_range, xi_add, pca_thr1, pca_thr2, blend_alpha, pca_tau=1,
                use_zone_level=False, zone_addend=3, zone_overlap_pct=0, n_levels=2):
    pca_p_vals = list(range(pca_p_range[0], pca_p_range[1] + 1))
    _zone_note = " + зонная маска" if use_zone_level else ""
    with st.spinner(f"PCA sweep ({len(pca_p_vals)} точек{_zone_note})…"):
        s1_rows = sweep_local_dim(
            ratio,
            p_values=pca_p_vals,
            xi_add=xi_add,
            pca_thresholds=(pca_thr1, pca_thr2),
            blend_alpha=blend_alpha,
            tau=pca_tau,
            use_zone_level=use_zone_level,
            zone_addend=zone_addend,
            zone_overlap_pct=zone_overlap_pct,
            n_levels=n_levels,
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
    _run_stage1(ratio, pca_p_range, xi_add, pca_thr1, pca_thr2, blend_alpha, pca_tau,
                use_zone_level=use_zone_level, zone_addend=zone_addend,
                zone_overlap_pct=zone_overlap_pct, n_levels=n_levels)

# ── Прогон (оба этапа) ────────────────────────────────────────────────────────

if run_btn:
    # SSA sweep: k=1..ssa_k или однократный прогон (k=None = использовать уже готовый ratio)
    _k_sweep = list(range(1, ssa_k + 1)) if (use_ssa and use_ssa_sweep) else [None]

    _sweep_refined:   list[np.ndarray] = []   # avg_fc per k (use_refine=True)
    _sweep_s2_all:    list[dict]       = []   # объединённые fc-результаты по d и k
    _sweep_s2_failed: list[dict]       = []
    _last_d_values:   list[int]        = []

    for _sweep_k in _k_sweep:
        # Подготовить ratio для текущего k
        if _sweep_k is not None:
            _ratio_k = _ssa_smooth_ratio(_ratio_pre_ssa.tobytes(), ssa_W, ssa_L, _sweep_k)
            if smooth_alpha < 1.0:
                _rk_orig = _ratio_k.copy()
                for _t in range(1, len(_ratio_k)):
                    _past = _rk_orig[max(0, _t - smooth_win + 1) : _t]
                    _ratio_k[_t] = smooth_alpha * _rk_orig[_t] + (1 - smooth_alpha) * _past.mean()
        else:
            _ratio_k = ratio  # уже подготовлен на уровне модуля

        _klabel = f" [SSA k={_sweep_k}]" if _sweep_k is not None else ""

        # ── Этап 1: PCA sweep ────────────────────────────────────────────────
        d_min, d_max = _run_stage1(
            _ratio_k, pca_p_range, xi_add, pca_thr1, pca_thr2, blend_alpha,
            use_zone_level=use_zone_level, zone_addend=zone_addend,
            zone_overlap_pct=zone_overlap_pct, n_levels=n_levels,
        )

        if d_min is None or d_max is None:
            st.error(f"Этап 1{_klabel}: не удалось определить d_min/d_max.")
            continue

        d_values = list(range(d_min, d_max + d_add + 1))
        _last_d_values = d_values
        d_max_eff = d_max + d_add

        prog_bar  = st.progress(0.0)
        prog_text = st.empty()

        def _cb(step: int, total: int, d: int, _lbl: str = _klabel) -> None:
            prog_bar.progress(step / total if total > 0 else 0.0)
            prog_text.caption(f"Этап 2{_lbl}: {step+1}/{total} — d={d}, m={xy_x*d+xy_y}")

        _sh = shadow_feat   if use_ohlc_cascade else None
        _pr = pressure_feat if use_ohlc_cascade else None

        _zmask_global = None
        if use_zone_level:
            _p_fit_global = xy_x * d_max + xy_y
            _W_global     = _p_fit_global + zone_addend
            _K_global     = max(2, (3 * (_p_fit_global + 1) + xi_add) >> (n_levels - 1))
            _zmask_global = sliding_zone_mask(
                _ratio_k, origin_algo, _W_global, _K_global, _ohlc_zone_ratios, zone_overlap_pct
            )

        if use_refine:
            with st.spinner(f"Итеративный прогноз{_klabel}: {n_refine} раундов × {len(d_values)} d-значений…"):
                _fc_k, _meta_k = run_forecast_refined(
                    _ratio_k, d_values, xy_x, xy_y, xi_add,
                    n_levels, horizon, origin_algo,
                    n_refine, blend_alpha, smap_theta, p_cascade_max,
                    use_lp_corr, _sh, _pr, ohlc_filter_last,
                    cascade_mode, final_pool_all, tau_base,
                    progress_cb=_cb,
                    pool_mask=_zmask_global,
                    use_lwr=use_lwr,
                    use_cluster_level=use_cluster_level,
                    n_clusters=n_clusters,
                    n_pca_cluster=n_pca_cluster,
                )
            prog_bar.empty(); prog_text.empty()
            if _fc_k is not None:
                _sweep_refined.append(_fc_k)
        else:
            with st.spinner(f"Этап 2{_klabel}: d∈[{d_min},{d_max_eff}] ({len(d_values)} значений)…"):
                _s2_k, _fail_k = run_stage2(
                    _ratio_k, d_values, xy_x, xy_y, xi_add,
                    n_levels, horizon, origin_algo,
                    _cb, blend_alpha, smap_theta, p_cascade_max,
                    fixed_manifold, use_lp_corr,
                    _sh, _pr, ohlc_filter_last,
                    cascade_mode, final_pool_all, tau_base,
                    zone_pool_mask_global=_zmask_global,
                    zone_slide=zone_slide,
                    use_lwr=use_lwr,
                    use_cluster_level=use_cluster_level,
                    n_clusters=n_clusters,
                    n_pca_cluster=n_pca_cluster,
                )
            prog_bar.empty(); prog_text.empty()
            _sweep_s2_all.extend(_s2_k)
            _sweep_s2_failed.extend(_fail_k)

    # ── Сохранить результаты ─────────────────────────────────────────────────
    if use_refine:
        avg_fc     = np.nanmean(_sweep_refined, axis=0) if _sweep_refined else None
        rounds_meta: list[dict] = []
        if avg_fc is not None:
            st.session_state.update({
                "s2_refined":      avg_fc,
                "s2_refined_meta": rounds_meta,
                "s2_origin":       origin,
                "s2_n":            n,
                "s2_times":        times,
                "s2_horizon":      horizon,
                "s2_n_eff":        origin + 1,
                "s2_pip_zones":   _pip_zones_display,
                "s2_trim_start":  origin + 1 - len(ratio),
            })
        else:
            st.error("Все итерации завершились без результата.")
        st.session_state.pop("s2_results", None)
    else:
        if _sweep_s2_all:
            st.session_state.update({
                "s2_results":  _sweep_s2_all,
                "s2_failed":   _sweep_s2_failed,
                "s2_d_values": _last_d_values,
                "s2_origin":   origin,
                "s2_n":        n,
                "s2_times":    times,
                "s2_horizon":  horizon,
                "s2_n_eff":    origin + 1,
                "s2_pip_zones":   _pip_zones_display,
                "s2_trim_start":  origin + 1 - len(ratio),
            })
        else:
            st.error("Все итерации завершились без результата.")
        st.session_state.pop("s2_refined", None)

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
    _dm, _dM  = _ss.get("s1_d_min"), _ss.get("s1_d_max")

    if _s2_orig >= len(low) or _s2_orig < 0:
        st.warning("Результаты прогноза устарели (данные изменились). Запустите прогноз заново.")
        st.stop()

    n_valid = len(_s2_res)
    n_total = len(_s2_dvs)
    st.subheader(f"Этап 2 — Прогноз  (d∈[{_dm},{_dM}],  {n_valid}/{n_total} валидных)")

    # Реконструкция цены из сырых att-предсказаний (пересчитывается на каждом рендере)
    # ratio = log(price) → price = exp(ratio_hat)
    _n_eff = _ss["s2_n_eff"]

    def _recon(fc_preds: np.ndarray) -> np.ndarray:
        return np.array([
            np.exp(fc_preds[j] + j * delta_ratio)
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

    # y-диапазон из OHLC + actual + forecast
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

    # Зонный уровень (PIP zones)
    _pip_zones_s2 = _ss.get("s2_pip_zones")
    _trim_s2      = _ss.get("s2_trim_start", 0)
    if _pip_zones_s2:
        for z in _pip_zones_s2:
            if not z['selected']:
                continue
            abs_a = z['a'] + _trim_s2; abs_b = z['b'] + _trim_s2
            if abs_b < show_from or abs_a > _s2_orig: continue
            t_a = _s2_times[max(abs_a, show_from)]
            t_b = _s2_times[min(abs_b, _s2_orig)]
            _key = (z['direction'], z['vol_level'])
            fig.add_vrect(
                x0=t_a, x1=t_b,
                fillcolor=_PIP_ZONE_BG.get(_key, "rgba(200,200,200,0.3)"),
                layer="below", line_width=0,
                annotation_text=_PIP_ZONE_ANN.get(_key, ""),
                annotation_position="top left",
                annotation_font_size=9,
            )

    fig.add_trace(go.Candlestick(
        x=_s2_times[show_from:],
        open=open_[show_from:], high=high[show_from:],
        low=low[show_from:],    close=close[show_from:],
        name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    if _any_preproc:
        fig.add_trace(go.Scatter(
            x=_s2_times[show_from : _s2_orig + 1],
            y=_prep_price_full[show_from : _s2_orig + 1],
            mode="lines", name="price (preprocessed)",
            line=dict(color="rgba(180, 120, 255, 0.75)", width=1),
            hovertemplate="%{y:.4f}<extra>price_prep</extra>",
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
        fc_y = np.concatenate([[price_input[_s2_orig]], fc_price_i])
        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_y,
            mode="lines", name=f"d={r['d']}",
            line=dict(width=1, color="rgba(100,180,255,0.25)"),
            showlegend=False,
        ))

    # Среднее
    avg_x = [_s2_times[_s2_orig]] + [_time_at(int(i)) for i in fc_idx]
    avg_y = np.concatenate([[price_input[_s2_orig]], avg_fc])
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

# — Итеративный прогноз —
if "s2_refined" in st.session_state and st.session_state["s2_refined"] is not None:
    _ss   = st.session_state
    _rf   = _ss["s2_refined"]          # (horizon,) att-предсказания
    _meta = _ss.get("s2_refined_meta", [])
    _n_eff = _ss["s2_n_eff"]
    _orig  = _ss["s2_origin"]
    _hor   = _ss["s2_horizon"]
    _times = _ss["s2_times"]
    _sn    = _ss["s2_n"]
    _dm2, _dM2 = _ss.get("s1_d_min"), _ss.get("s1_d_max")

    if _orig >= len(low) or _orig < 0:
        st.warning("Результаты итеративного прогноза устарели. Запустите заново.")
    else:
        st.subheader(f"Итеративный прогноз ({len(_meta)} раундов)")
        for rm in _meta:
            st.caption(f"Раунд {rm['round']}: {rm['n_valid']} d-значений, "
                       f"d={rm['ds']}")

        # Реконструкция цены: ratio = log(price) → price = exp(ratio_hat)
        def _recon_rf(fc: np.ndarray) -> np.ndarray:
            return np.array([
                np.exp(fc[j] + j * delta_ratio)
                for j in range(len(fc))
            ])

        rf_price = _recon_rf(_rf)

        # Временны́е метки будущих баров
        try:
            _base = pd.Timestamp(_times[-1])
            _dt   = pd.Timestamp(_times[-1]) - pd.Timestamp(_times[-2])
            future_times_rf = [str(_base + _dt * h) for h in range(1, _hor + 1)]
        except Exception:
            future_times_rf = [f"+{h}" for h in range(1, _hor + 1)]

        def _time_rf(idx: int) -> str:
            if idx < _sn:
                return _times[idx]
            return future_times_rf[min(idx - _sn, len(future_times_rf) - 1)]

        show_from_rf = max(0, _orig - 299)
        fc_idx_rf    = np.arange(_orig + 1, _orig + 1 + _hor)

        _y_lo2 = float(np.min(low[show_from_rf : _orig + 1]))
        _y_hi2 = float(np.max(high[show_from_rf : _orig + 1]))
        if _orig < _sn - 1:
            _ae2  = min(_sn, _orig + _hor + 1)
            _y_lo2 = min(_y_lo2, float(close[_orig : _ae2].min()))
            _y_hi2 = max(_y_hi2, float(close[_orig : _ae2].max()))
        _y_lo2 = min(_y_lo2, float(rf_price.min()))
        _y_hi2 = max(_y_hi2, float(rf_price.max()))
        _pad2  = (_y_hi2 - _y_lo2) * 0.05

        fig2 = go.Figure()

        # Зонный уровень (PIP zones)
        _pip_zones_rf = _ss.get("s2_pip_zones")
        _trim_rf      = _ss.get("s2_trim_start", 0)
        if _pip_zones_rf:
            for z in _pip_zones_rf:
                if not z['selected']:
                    continue
                abs_a = z['a'] + _trim_rf; abs_b = z['b'] + _trim_rf
                if abs_b < show_from_rf or abs_a > _orig: continue
                t_a = _times[max(abs_a, show_from_rf)]
                t_b = _times[min(abs_b, _orig)]
                _key = (z['direction'], z['vol_level'])
                fig2.add_vrect(
                    x0=t_a, x1=t_b,
                    fillcolor=_PIP_ZONE_BG.get(_key, "rgba(200,200,200,0.3)"),
                    layer="below", line_width=0,
                    annotation_text=_PIP_ZONE_ANN.get(_key, ""),
                    annotation_position="top left",
                    annotation_font_size=9,
                )

        fig2.add_trace(go.Candlestick(
            x=_times[show_from_rf:],
            open=open_[show_from_rf:], high=high[show_from_rf:],
            low=low[show_from_rf:],   close=close[show_from_rf:],
            name="OHLC", increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ))
        if _any_preproc:
            fig2.add_trace(go.Scatter(
                x=_times[show_from_rf : _orig + 1],
                y=_prep_price_full[show_from_rf : _orig + 1],
                mode="lines", name="price (preprocessed)",
                line=dict(color="rgba(180, 120, 255, 0.75)", width=1),
                hovertemplate="%{y:.4f}<extra>price_prep</extra>",
            ))
        if _orig < _sn - 1:
            _ae2 = min(_sn, _orig + _hor + 1)
            fig2.add_trace(go.Scatter(
                x=_times[_orig:_ae2], y=close[_orig:_ae2],
                mode="lines", name="actual",
                line=dict(color="white", width=1, dash="dot"),
            ))
        rf_x = [_times[_orig]] + [_time_rf(int(i)) for i in fc_idx_rf]
        rf_y = np.concatenate([[price_input[_orig]], rf_price])
        fig2.add_trace(go.Scatter(
            x=rf_x, y=rf_y, mode="lines", name="refined forecast",
            line=dict(color="#ff9800", width=2),
        ))
        fig2.add_vline(x=_times[_orig], line_color="gray", line_dash="dash", line_width=1)
        fig2.update_layout(
            height=560, xaxis_rangeslider_visible=False, xaxis_type="date",
            yaxis_range=[_y_lo2 - _pad2, _y_hi2 + _pad2],
            title=f"{ticker} {interval}  ·  итеративный прогноз  ·  "
                  f"d∈[{_dm2},{_dM2}]  ·  horizon={_hor}",
            template="plotly_dark",
            legend=dict(orientation="h", y=-0.18),
        )
        if fc_idx_rf[-1] >= _sn:
            fig2.update_layout(xaxis_range=[_times[show_from_rf], _time_rf(int(fc_idx_rf[-1]))])
        st.plotly_chart(fig2, use_container_width=True)
