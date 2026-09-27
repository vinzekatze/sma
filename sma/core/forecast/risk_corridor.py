"""
risk_corridor — риск-корридор High/Low (running max/min) + полоса Close на
h=1..H шагов вперёд. Портировано из прод-заготовочного прототипа
`prototype/forcaster/ui/app27-risk-corridor.py` (победивший метод фазы 19,
см. docs/plans/app27_risk_corridor_migration_plan.md и память
project_phase19_*).

Метод (см. докстринг прототипа за полное обоснование каждого выбора):
  - kNN-поиск в пространстве dratio(close) (причинный logtrend-детренд),
    расстояние — amp_cos (`_dists`, sma.core.forecast.simplex_ensemble).
  - Взвешивание пула — Student-t кернель, nu=1.0 ФИКСИРОВАН (Коши, эксп.21
    фазы 19 — толстые хвосты дают вес соседям-предвестникам обвала).
  - fixed_mc + shared_pool: per-step когерентный путь, сэмплированный на
    каждом шаге сосед отдаёт СВОЮ реальную синхронную тройку (Close,High,Low)
    на том же баре — не независимый ресэмплинг по каждому ряду.
  - offsets=arange(h_max), БЕЗ -1 (off-by-one баг исправлен 2026-09-22, см.
    память project_phase19_offbyone_horizon_bug — переносится ИСПРАВЛЕННАЯ
    версия, старые research-скрипты фазы 19 содержат баг).

Параметры (p_fit=20, blend_alpha=0.75, theta=20.0, nu=1.0 константа,
n_sim=15000, h=20) валидированы честным temporal walk-forward и устойчиво
обобщаются на новые тикеры без калибровки (эксп.06 фазы 19: "рецепт
обобщается на 6 НОВЫХ тикеров, 6/6") — поэтому у этого инструмента, в отличие
от band_lambda/range_forecast, НЕТ per-тикер калибровки вообще.
"""
from __future__ import annotations

import numpy as np

from .band_lambda import weighted_quantile
from .simplex_ensemble import _dists, _logtrend_causal

NU = 1.0  # Student-t хвост, эксп.21 фазы 19 — фиксирован, не настраивается
THEILER_MARGIN = 5  # theiler = p_fit + THEILER_MARGIN, см. прототип

DEFAULT_P_FIT = 20
DEFAULT_BLEND_ALPHA = 0.75
DEFAULT_THETA = 20.0
DEFAULT_N_SIM = 15000
DEFAULT_H = 20
DEFAULT_COVERAGE_PCT = 95
DEFAULT_SEED = 42
MAX_CANDLES_DEFAULT = 5000  # общее ограничение стоимости — тот же приём, что range_forecast.MAX_CANDLES_DEFAULT


def build_pool(S: np.ndarray, origin: int, p_fit: int, theiler: int, h_max: int,
                theta: float, blend_alpha: float):
    """Плоский причинный пул delay-векторов длины p_fit на ряде S (dratio-
    пространство), взвешенный Student-t кернелем (nu=NU). anchors — индекс
    ПОСЛЕДНЕГО бара окна соседа в dratio-space (ещё БЕЗ close-space
    коррекции — её делает вызывающий код через +1)."""
    starts = np.arange(0, origin - p_fit - h_max + 2)
    if len(starts) == 0:
        return None, None
    anchors = starts + p_fit - 1
    valid = np.abs(origin - anchors) > theiler
    starts, anchors = starts[valid], anchors[valid]
    if len(starts) < 15:
        return None, None
    X_pool = S[starts[:, None] + np.arange(p_fit)]
    query = S[origin - p_fit + 1: origin + 1].astype(np.float64)
    finite_rows = np.all(np.isfinite(X_pool), axis=1)
    starts, anchors, X_pool = starts[finite_rows], anchors[finite_rows], X_pool[finite_rows]
    if len(starts) < 15 or not np.all(np.isfinite(query)):
        return None, None
    dist = _dists(X_pool, query, blend_alpha)

    mean_d = np.mean(dist) if np.mean(dist) > 0 else 1.0
    bw = mean_d / theta
    w = (1.0 + (dist / bw) ** 2 / NU) ** (-(NU + 1.0) / 2.0)

    wsum = w.sum()
    if wsum <= 0 or not np.isfinite(wsum):
        return None, None
    return anchors, w / wsum


def simulate_shared_pool(dratio_close, dratio_high, dratio_low, ratio_c_o, ratio_h_o, ratio_l_o,
                          anchors, weights, h_max, n_sim, seed):
    """fixed_mc: per-step когерентный путь (цель = собственное состояние
    частицы на h-1, не origin). shared_pool: сэмплированный на каждом шаге
    сосед отдаёт СВОЮ реальную синхронную тройку (Close,High,Low) — не
    независимый ресэмплинг по каждому ряду. Возвращает (close,high,low)
    пути в ratio-единицах (n_sim, h_max), ДО умножения на trend."""
    rng = np.random.default_rng(seed)
    draws_idx = np.empty((n_sim, h_max), dtype=np.int64)
    for h in range(1, h_max + 1):
        draws_idx[:, h - 1] = rng.choice(len(anchors), size=n_sim, p=weights)
    anchor_draws = anchors[draws_idx]
    offsets = np.arange(h_max)[None, :]
    close_path = ratio_c_o + np.cumsum(dratio_close[anchor_draws + offsets], axis=1)
    high_path = ratio_h_o + np.cumsum(dratio_high[anchor_draws + offsets], axis=1)
    low_path = ratio_l_o + np.cumsum(dratio_low[anchor_draws + offsets], axis=1)
    return close_path, high_path, low_path


def compute_risk_corridor(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    h_max: int = DEFAULT_H,
    p_fit: int = DEFAULT_P_FIT,
    blend_alpha: float = DEFAULT_BLEND_ALPHA,
    theta: float = DEFAULT_THETA,
    n_sim: int = DEFAULT_N_SIM,
    coverage_pct: float = DEFAULT_COVERAGE_PCT,
    seed: int = DEFAULT_SEED,
) -> dict | None:
    """
    Точка входа: close/high/low уже причинно обрезаны до origin (последний
    элемент = origin, тот же контракт, что simplex_ensemble.forecast_ensemble
    — см. докстринг модуля simplex_ensemble.py). Возвращает per-h границы
    (close_low/close_high — полоса Close, risk_low/risk_high — риск-огибающая
    running min(Low)/max(High)) + n_neighbors, либо None, если пул соседей
    недостаточен (см. build_pool).
    """
    n_bars = len(close)
    origin = n_bars - 1

    _, a_arr, b_arr = _logtrend_causal(close)
    a_o, b_o = float(a_arr[-1]), float(b_arr[-1])
    trend_full = np.exp(a_o + b_o * np.arange(n_bars))
    ratio_close = close / trend_full
    ratio_high = high / trend_full
    ratio_low = low / trend_full
    dratio_close = np.diff(ratio_close)
    dratio_high = np.diff(ratio_high)
    dratio_low = np.diff(ratio_low)

    theiler = p_fit + THEILER_MARGIN

    anchors, weights = build_pool(dratio_close, origin - 1, p_fit, theiler, h_max, theta, blend_alpha)
    if anchors is None:
        return None
    anchors = anchors + 1  # коррекция: окно из ПРИРАЩЕНИЙ длины p_fit заканчивается на +1 позже, чем окно из УРОВНЕЙ (см. эксп.01 фазы 19)

    close_path, high_path, low_path = simulate_shared_pool(
        dratio_close, dratio_high, dratio_low,
        ratio_close[origin], ratio_high[origin], ratio_low[origin],
        anchors, weights, h_max, n_sim, seed,
    )

    future_idx = origin + 1 + np.arange(h_max, dtype=np.float64)
    trend_fwd = np.exp(a_o + b_o * future_idx)
    close_price = close_path * trend_fwd[None, :]
    high_price = high_path * trend_fwd[None, :]
    low_price = low_path * trend_fwd[None, :]
    run_high = np.maximum.accumulate(high_price, axis=1)
    run_low = np.minimum.accumulate(low_price, axis=1)

    n_particles = high_price.shape[0]
    w_uniform = np.full(n_particles, 1.0 / n_particles)
    half = (100 - coverage_pct) / 200.0

    steps = {}
    for h in range(1, h_max + 1):
        col = h - 1
        close_low = weighted_quantile(close_price[:, col], w_uniform, (half,))[half]
        close_high = weighted_quantile(close_price[:, col], w_uniform, (1 - half,))[1 - half]
        risk_low = weighted_quantile(run_low[:, col], w_uniform, (half,))[half]
        risk_high = weighted_quantile(run_high[:, col], w_uniform, (1 - half,))[1 - half]
        steps[h] = {
            "close_low": close_low, "close_high": close_high,
            "risk_low": risk_low, "risk_high": risk_high,
        }

    return {"steps": steps, "n_neighbors": int(len(anchors))}
