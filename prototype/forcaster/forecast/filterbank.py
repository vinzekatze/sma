"""
Filter bank decomposition + LA/LWR forecast for slow components (C3+C4+C5).

Core idea (research scripts 30-34):
  dratio contains ~62% energy in fast noise (C0-C2, periods ≤4 bars).
  That noise dominates L2 distance in neighbour search → LA picks random neighbours.
  After removing it, LA/LWR finds structurally similar neighbours of slow components.

Result: −22% MAPE aggregate across 8 MOEX tickers (1d, N=1600, p<0.0001).

Rule: p ≥ T_min / 4 where T_min ≈ 52 bars for C3 (1d interval) → p_min ≈ 13.
At p < T_min/4 the delay vector cannot distinguish phases of the slow wave.

Two input modes:
  dratio mode (default): forecast Δratio → caller uses cumsum to get ratio_hat
  ratio  mode:           forecast slow ratio components directly → no cumsum needed
                         (fast components C0-C2 are discarded; forecast is smooth)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt

from .embedding import build_delay_matrix, last_vector


FILTER_ORDER: int = 4
CUTOFFS: list[float] = [0.25, 0.125, 0.0625, 0.03125, 0.015625]   # octave-spaced
SLOW_IDX: list[int] = [3, 4, 5]   # C3+C4+C5, periods >52 bars (1d)
FB_P:  int = 20                    # p ≥ T_min/4 ≈ 13; 20 is the validated sweet-spot
FB_XI: int = 3 * (FB_P + 1)       # = 63; lecture rule Ξ ≥ 3(p+1)


def make_filter_bank(series: np.ndarray) -> np.ndarray:
    """
    Causal Butterworth filter bank. Returns array of shape (6, N) — components C0..C5.

    Uses sosfilt (single-pass, causal) — strictly time-correct, no look-ahead.
    Do NOT replace with sosfiltfilt: that is zero-phase (non-causal) and
    introduces look-ahead bias in walk-forward evaluation.

    Component periods (1d bars):
      C0: ~2-4    C1: ~4-8    C2: ~8-16
      C3: ~16-52  C4: ~52-103  C5: 103+ (residual slow trend)
    """
    components: list[np.ndarray] = []
    remaining = series.copy()
    for fc in CUTOFFS:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        components.append(remaining - low)
        remaining = low
    components.append(remaining)      # C5: residual
    return np.array(components)       # (6, N)


def forecast_fb(
    series: np.ndarray,
    origin_k: int,
    horizon: int,
    p: int = FB_P,
    n_neighbors: int = FB_XI,
    component_idx: list[int] | None = None,
    use_lwr: bool = True,
    p_per_component: dict[int, int] | None = None,
) -> np.ndarray:
    """
    Unified filter bank forecast for dratio OR ratio series.

    dratio mode: returns dratio_hat (increments).
        Reconstruct: ratio_hat = ratio[origin_k] + cumsum(dratio_hat)
                     price_hat = ratio_hat * MA[origin_k]

    ratio mode:  returns slow_ratio_hat (absolute values of slow components).
        Reconstruct: price_hat = slow_ratio_hat * MA[origin_k]  (no cumsum)
        Note: fast components C0-C2 are discarded — forecast is a smoothed ratio.

    Args:
        series:             dratio[:] or ratio[:] — only series[:origin_k] is used.
        origin_k:           Forecast starts at origin_k + 1 (dratio) or origin_k (ratio).
        horizon:            Steps to predict.
        p:                  Default embedding dimension (fallback for components not in p_per_component).
        n_neighbors:        Default Ξ nearest neighbours (fallback). Rule: ≥ 3*(p+1).
        component_idx:      Slow component indices. Default [3, 4, 5] (C3+C4+C5).
        use_lwr:            True → Gaussian-weighted LWR; False → plain OLS (LA1).
        p_per_component:    Per-component p override, e.g. {0: 3, 3: 20, 4: 26}.
                            xi for that component is auto-computed as 3*(p_ci+1).
                            Components absent from the dict use global p / n_neighbors.
    """
    if component_idx is None:
        component_idx = SLOW_IDX

    history = series[:origin_k]
    comp    = make_filter_bank(history)   # (6, origin_k) — causal decomposition

    hats: list[np.ndarray] = []
    for ci in component_idx:
        p_ci  = p_per_component.get(ci, p) if p_per_component else p
        xi_ci = 3 * (p_ci + 1) if (p_per_component and ci in p_per_component) else n_neighbors

        comp_hist = comp[ci]
        X, y = build_delay_matrix(comp_hist, p_ci)
        if len(X) < xi_ci:
            hats.append(np.zeros(horizon))
            continue
        vec = last_vector(comp_hist, p=p_ci).copy()
        step_fn = _lwr_step if use_lwr else _la1_step
        hats.append(step_fn(X, y, vec, xi_ci, horizon))

    return np.sum(hats, axis=0) if hats else np.zeros(horizon)    # (horizon,)


# ── AR helpers for damped forecast ───────────────────────────────────────────

_AR_P_MAX: int = 20


def _fit_ar_bic(series: np.ndarray) -> tuple[int, np.ndarray]:
    """AR(p) order by BIC. Returns (best_p, coeffs=[intercept, a1..ap])."""
    n    = len(series)
    eps  = 1e-10
    best_bic, best_p, best_c = np.inf, 1, np.zeros(2)
    for p in range(1, min(_AR_P_MAX + 1, (n - 1) // 4)):
        nef = n - p
        X   = np.zeros((nef, p))
        for lag in range(p):
            X[:, lag] = series[p - 1 - lag: n - 1 - lag]
        X    = np.hstack([np.ones((nef, 1)), X])
        c, _, _, _ = np.linalg.lstsq(X, series[p:], rcond=None)
        ssr  = np.sum((series[p:] - X @ c) ** 2)
        bic  = nef * np.log(ssr / nef + eps) + (p + 1) * np.log(nef)
        if bic < best_bic:
            best_bic, best_p, best_c = bic, p, c
    return best_p, best_c


def _ar_damped(series: np.ndarray, horizon: int, gamma: float) -> np.ndarray:
    """AR(BIC) iterative forecast × gamma^h decay envelope."""
    if len(series) <= _AR_P_MAX + 2:
        return np.zeros(horizon)
    best_p, coeffs = _fit_ar_bic(series)
    buf = list(series[-best_p:])
    raw = np.empty(horizon)
    for h in range(horizon):
        val    = coeffs[0] + sum(coeffs[1 + k] * buf[-(k + 1)] for k in range(best_p))
        raw[h] = val
        buf.append(val)
    return raw * (gamma ** np.arange(1, horizon + 1))


def forecast_fb_damped(
    dratio: np.ndarray,
    origin_k: int,
    horizon: int,
    p_slow: int = FB_P,
    xi_slow: int = FB_XI,
    gamma_c1: float = 0.5,
    gamma_c2: float = 0.8,
    slow_components: list[int] | None = None,
) -> np.ndarray:
    """
    Filter bank + damped AR(BIC) for noise components C1 and C2.

    C3–C5: LWR (same as standard forecast_fb).
    C1:    AR(BIC) × gamma_c1^h  (periods ~8–16 bars)
    C2:    AR(BIC) × gamma_c2^h  (periods ~16–32 bars)
    C0:    zero (confirmed unpredictable — research scripts 48, 51)

    Research result (scripts 51–52, 8 tickers, 400 origins, 1d, logtrend):
      gamma_c1=0.5, gamma_c2=0.8 → −8.01% MAPE vs slow-only LWR baseline.
      Physical reason: C2 AR is accurate up to h≈5 (19× better than zero),
      so gentle decay (0.8) preserves the signal. C1 AR oscillates by h≈3,
      so aggressive decay (0.5) suppresses the divergence.
    """
    if slow_components is None:
        slow_components = SLOW_IDX

    history = dratio[:origin_k]
    comp    = make_filter_bank(history)   # (6, origin_k)

    hats: list[np.ndarray] = []

    # Slow C3–C5: LWR
    for ci in slow_components:
        comp_hist = comp[ci]
        X, y = build_delay_matrix(comp_hist, p_slow)
        if len(X) < xi_slow:
            hats.append(np.zeros(horizon))
            continue
        vec = last_vector(comp_hist, p=p_slow).copy()
        hats.append(_lwr_step(X, y, vec, xi_slow, horizon))

    # C1 and C2: damped AR
    hats.append(_ar_damped(comp[1], horizon, gamma_c1))
    hats.append(_ar_damped(comp[2], horizon, gamma_c2))

    return np.sum(hats, axis=0) if hats else np.zeros(horizon)


def forecast_fb_merged_slow(
    dratio: np.ndarray,
    origin_k: int,
    horizon: int,
    p: int = FB_P,
    xi: int | None = None,
    merged_components: list[int] | None = None,
    damped_c1: bool = True,
    gamma_c1: float = 0.5,
) -> np.ndarray:
    """
    Merged-slow режим: единый LWR на объединённом медленном сигнале.

    sum(C_i for i in merged_components) → один LWR  (default: C2+C3+C4+C5)
    C1 → Damped AR × gamma_c1^h  (если damped_c1=True)
    C0 → ноль  (полоса неопределённости считается в app отдельно)

    Мотивация (скр.54–56): аттрактор суммарного медленного сигнала не ломается
    межкомпонентным разбиением. LWR на объединённом C2-C5 описывает динамику
    курса лучше, чем суммирование независимых per-component прогнозов, хотя
    AGG MAPE оказался хуже — потому что C2 добавляет быстрые осцилляции,
    которые важны для описания движения цены, но труднее попасть в цифру.
    C1 через Damped AR остаётся, т.к. агрессивный γ=0.5 гасит расхождение
    фазы за горизонт прогноза.
    """
    if merged_components is None:
        merged_components = [2, 3, 4, 5]
    if xi is None:
        xi = 3 * (p + 1)

    history = dratio[:origin_k]
    comp    = make_filter_bank(history)

    # единый LWR на объединённом сигнале
    merged = np.sum([comp[ci] for ci in merged_components], axis=0)
    X, y   = build_delay_matrix(merged, p)
    if len(X) >= xi:
        vec       = last_vector(merged, p=p).copy()
        dhat_slow = _lwr_step(X, y, vec, xi, horizon)
    else:
        dhat_slow = np.zeros(horizon)

    # C1: Damped AR
    dhat_c1 = _ar_damped(comp[1], horizon, gamma_c1) if damped_c1 else np.zeros(horizon)

    return dhat_slow + dhat_c1


# ── LP attractor (единый фильтр, скр.58 архитектура) ─────────────────────────

_SOS_LP_ATT = butter(FILTER_ORDER, 0.125, btype="low", output="sos")


def forecast_lp_attractor(
    dratio: np.ndarray,
    origin_k: int,
    horizon: int,
    p: int = 8,
    xi: int | None = None,
    wn: float = 0.125,
) -> np.ndarray:
    """
    Единый LP(Wn) аттрактор + LWR.

    Один каузальный LP-фильтр на dratio вместо каскада filter bank.
    wn=0.125 (default) → аттрактор C2-C5 (период ≥16б), τ≈7 баров.
    wn=0.0625          → аттрактор C3-C5 (период ≥32б), τ≈14 баров.

    Returns dratio_hat: (horizon,) инкременты ratio.
    """
    if xi is None:
        xi = 3 * (p + 1)
    sos  = (butter(FILTER_ORDER, wn, btype="low", output="sos")
            if wn != 0.125 else _SOS_LP_ATT)
    att  = sosfilt(sos, dratio[:origin_k])
    X, y = build_delay_matrix(att, p)
    if len(X) < xi:
        return np.zeros(horizon)
    vec = last_vector(att, p=p).copy()
    return _lwr_step(X, y, vec, xi, horizon)


def forecast_smooth_lwr(
    signal: np.ndarray,
    origin_k: int,
    horizon: int,
    p: int = 8,
    xi: int | None = None,
) -> np.ndarray:
    """LWR на любом предварительно сглаженном сигнале (MA, LP и т.п.)."""
    if xi is None:
        xi = 3 * (p + 1)
    X, y = build_delay_matrix(signal[:origin_k], p)
    if len(X) < xi:
        return np.zeros(horizon)
    vec = last_vector(signal[:origin_k], p=p).copy()
    return _lwr_step(X, y, vec, xi, horizon)


def forecast_smooth_auto_p(
    signal: np.ndarray,
    origin_k: int,
    horizon: int,
    tau: int = 7,
    p_candidates: list[int] | None = None,
    top_n: int = 3,
) -> list[tuple[int, float, np.ndarray]]:
    """
    Авто-p через τ-окно на любом сглаженном сигнале (без LP фильтра).

    Параметры аналогичны forecast_lp_auto_p, но signal подаётся снаружи.
    """
    if p_candidates is None:
        p_candidates = [4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
    top_n = max(1, min(top_n, len(p_candidates)))

    eff_origin = max(1, origin_k - tau)
    sig_val    = signal[eff_origin: eff_origin + tau]

    scored: list[tuple[float, int, np.ndarray]] = []

    for p in p_candidates:
        xi = 3 * (p + 1)

        X_v, y_v = build_delay_matrix(signal[:eff_origin], p)
        if len(X_v) < xi:
            continue
        vec_v  = last_vector(signal[:eff_origin], p=p).copy()
        dhat_v = _lwr_step(X_v, y_v, vec_v, xi, len(sig_val))
        score  = float(np.mean((dhat_v - sig_val[:len(dhat_v)]) ** 2))

        X_f, y_f = build_delay_matrix(signal, p)
        if len(X_f) < xi:
            continue
        vec_f  = last_vector(signal, p=p).copy()
        dhat_f = _lwr_step(X_f, y_f, vec_f, xi, horizon)

        scored.append((score, p, dhat_f))

    scored.sort(key=lambda x: x[0])
    return [(p, score, dhat) for score, p, dhat in scored[:top_n]]


def forecast_lp_auto_p(
    dratio: np.ndarray,
    origin_k: int,
    horizon: int,
    wn: float = 0.125,
    tau: int = 7,
    p_candidates: list[int] | None = None,
    top_n: int = 3,
) -> list[tuple[int, float, np.ndarray]]:
    """
    Авто-p через τ-окно задержки LP фильтра.

    Для каждого p-кандидата:
      1. Валидация: LWR из origin_k-τ → τ шагов, сравнить с att[origin_k-τ:origin_k]
      2. Полный прогноз: LWR из origin_k → horizon шагов

    Возвращает top_n лучших [(p, val_mse, dhat_horizon)] по возрастанию val_mse.
    """
    if p_candidates is None:
        p_candidates = [4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
    top_n = max(1, min(top_n, len(p_candidates)))

    sos      = (butter(FILTER_ORDER, wn, btype="low", output="sos") if wn != 0.125 else _SOS_LP_ATT)
    att_full = sosfilt(sos, dratio[:origin_k])

    eff_origin = max(1, origin_k - tau)
    att_val    = att_full[eff_origin: eff_origin + tau]

    scored: list[tuple[float, int, np.ndarray]] = []

    for p in p_candidates:
        xi = 3 * (p + 1)

        # Валидация: обучаем на att[:eff_origin]
        X_v, y_v = build_delay_matrix(att_full[:eff_origin], p)
        if len(X_v) < xi:
            continue
        vec_v  = last_vector(att_full[:eff_origin], p=p).copy()
        dhat_v = _lwr_step(X_v, y_v, vec_v, xi, len(att_val))
        score  = float(np.mean((dhat_v - att_val[:len(dhat_v)]) ** 2))

        # Полный прогноз из origin_k
        X_f, y_f = build_delay_matrix(att_full, p)
        if len(X_f) < xi:
            continue
        vec_f  = last_vector(att_full, p=p).copy()
        dhat_f = _lwr_step(X_f, y_f, vec_f, xi, horizon)

        scored.append((score, p, dhat_f))

    scored.sort(key=lambda x: x[0])
    return [(p, score, dhat) for score, p, dhat in scored[:top_n]]


# ── backward-compat alias ─────────────────────────────────────────────────────

def forecast_lwr_fb(
    dratio: np.ndarray,
    origin_k: int,
    horizon: int,
    p: int = FB_P,
    n_neighbors: int = FB_XI,
    component_idx: list[int] | None = None,
) -> np.ndarray:
    """LWR filter bank on dratio. Alias for forecast_fb(..., use_lwr=True)."""
    return forecast_fb(dratio, origin_k, horizon, p=p, n_neighbors=n_neighbors,
                       component_idx=component_idx, use_lwr=True)


# ── internal step functions ───────────────────────────────────────────────────

def _lwr_step(
    X: np.ndarray,
    y: np.ndarray,
    vec: np.ndarray,
    n_neighbors: int,
    horizon: int,
) -> np.ndarray:
    """Iterative LWR: Gaussian kernel, adaptive bandwidth h = dist(Ξ-th neighbour)."""
    current = vec.copy()
    out     = np.empty(horizon)
    for h in range(horizon):
        dists  = np.linalg.norm(X - current, axis=1)
        nn_idx = np.argpartition(dists, n_neighbors)[:n_neighbors]
        h_bw   = max(float(dists[nn_idx].max()), 1e-10)
        w      = np.exp(-0.5 * (dists[nn_idx] / h_bw) ** 2)
        A      = np.hstack([np.ones((n_neighbors, 1)), X[nn_idx]])
        sw     = np.sqrt(w)
        coeffs, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx], rcond=None)
        val            = float(coeffs[0] + current @ coeffs[1:])
        out[h]         = val
        current        = np.roll(current, -1)
        current[-1]    = val
    return out


def _la1_step(
    X: np.ndarray,
    y: np.ndarray,
    vec: np.ndarray,
    n_neighbors: int,
    horizon: int,
) -> np.ndarray:
    """Iterative LA1: plain OLS on Ξ nearest neighbours (uniform weights)."""
    current = vec.copy()
    out     = np.empty(horizon)
    for h in range(horizon):
        dists  = np.linalg.norm(X - current, axis=1)
        nn_idx = np.argpartition(dists, n_neighbors)[:n_neighbors]
        A      = np.hstack([np.ones((n_neighbors, 1)), X[nn_idx]])
        coeffs, _, _, _ = np.linalg.lstsq(A, y[nn_idx], rcond=None)
        val            = float(coeffs[0] + current @ coeffs[1:])
        out[h]         = val
        current        = np.roll(current, -1)
        current[-1]    = val
    return out
