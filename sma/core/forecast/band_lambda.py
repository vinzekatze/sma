"""
band_lambda — zigzag event-time forecaster for the uncertainty BAND (not a
point), with a λ-weighted pool kernel. λ is always the uniform-pool zero
vector in prod — λ-calibration was removed 2026-09-12 (see memory
project_phase7_calibration_removed_final); the kernel machinery stays
general (still accepts arbitrary λ) purely because splitting it out again
would cost more than it'd save.

Ported from research/reference/smap_band_ref.py + band_lambda_calibrator_ref.py
+ smap_band_weighted_ref.py (session 2026-07-08, see
docs/plans/band_forecast_migration_plan.md sections 1-2 for the full
methodology writeup and the approved SBER T=20% reference numbers used to
verify this port). This module holds every *pure* (no I/O) building block —
zigzag, rank features, pool assembly, live band forecast. The one async
helper (load_raw_ticker_arrays) is DB I/O only, no numpy.

Causality contract (see memory feedback-causality-enforcement): every
function here operates only on the arrays it's given — there is no
algorithmic look-ahead guard anywhere in this module. The ONLY place
causality is enforced is the caller truncating `dates <= cutoff` BEFORE
calling in (mask_ticker_data / build_causal_pool_with_mixed_ranks' own
per-ticker cutoff mask). This mirrors the reference exactly and is what
research/reference/causality_check_band_lambda_ref.py verifies bit-for-bit.
"""

from __future__ import annotations

import numpy as np

from .pivot_time_band import default_step_widths_bars, median_leg_durations_by_direction

# ── constants (docs/plans/band_forecast_migration_plan.md section 1) ─────────

DEFAULT_M = 6
DEFAULT_MIN_BARS = 5
DEFAULT_THETA = 0.0
Q_LEVELS: tuple = (0.1, 0.25, 0.5, 0.75, 0.9)
COVERAGE_LEVELS: tuple = (0.50, 0.75, 0.90)
RANK_WINDOW = 252
N_BINS = 40
WINDOW_VOL = 20
K_LEG = 10
DEFAULT_STEP_BARS_FALLBACK = 5  # matches sma/ui/chart.js:DEFAULT_STEP_BARS

FEATURE_ORDER = ["volume", "trend", "leg_age", "velocity", "acceleration", "volatility"]
BAR_FEATURES = ["volume", "trend", "velocity", "acceleration", "volatility"]
PIVOT_FEATURES = ["leg_age"]


# ── zigzag (causal, event-time decomposition) ────────────────────────────────

def build_zigzag(
    log_highs: np.ndarray,
    log_lows: np.ndarray,
    dates: np.ndarray,
    threshold: float,
    min_bars: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Causal log-threshold zigzag. A pivot fires when price retraces
    `threshold` (log) from the running extreme; min_bars enforces a minimum
    bar count between CONFIRMED pivots (Depth-like, see build_zigzag_ref for
    the pause-window caveat — unchanged behaviour, ported verbatim).

    Returns (log_pivot_prices, extreme_dates, confirm_dates, pivot_directions):
      log_pivot_prices — log price at the extremum
      extreme_dates    — date of the bar where the extremum actually printed
                          (display only — where reversals actually happened)
      confirm_dates    — date of the bar where the reversal was CONFIRMED
                          (the only causally valid point — this is what every
                          downstream computation and the forecast origin
                          anchor to, per docs/plans/band_forecast_migration_
                          plan.md section 3: "origin прогноза — по дате
                          подтверждения")
      pivot_directions — +1 top (HIGH), -1 bottom (LOW)
    """
    log_pivot_prices: list[float] = []
    extreme_dates: list = []
    confirm_dates: list = []
    pivot_directions: list[int] = []

    current_direction = 0
    extreme_log_price = (log_highs[0] + log_lows[0]) / 2.0
    extreme_bar_index = 0
    last_pivot_bar_index = -min_bars - 1   # first pivot is never blocked

    for bar_index in range(len(log_highs)):
        enough_bars = (bar_index - last_pivot_bar_index) >= min_bars

        if current_direction == 0:
            if log_highs[bar_index] - extreme_log_price >= threshold:
                current_direction = 1
                extreme_log_price = log_highs[bar_index]
                extreme_bar_index = bar_index
            elif extreme_log_price - log_lows[bar_index] >= threshold:
                current_direction = -1
                extreme_log_price = log_lows[bar_index]
                extreme_bar_index = bar_index

        elif current_direction == 1:
            if log_highs[bar_index] > extreme_log_price:
                extreme_log_price = log_highs[bar_index]
                extreme_bar_index = bar_index
            elif extreme_log_price - log_lows[bar_index] >= threshold and enough_bars:
                log_pivot_prices.append(extreme_log_price)
                extreme_dates.append(dates[extreme_bar_index])
                confirm_dates.append(dates[bar_index])
                pivot_directions.append(+1)
                last_pivot_bar_index = bar_index
                current_direction = -1
                extreme_log_price = log_lows[bar_index]
                extreme_bar_index = bar_index

        else:  # current_direction == -1
            if log_lows[bar_index] < extreme_log_price:
                extreme_log_price = log_lows[bar_index]
                extreme_bar_index = bar_index
            elif log_highs[bar_index] - extreme_log_price >= threshold and enough_bars:
                log_pivot_prices.append(extreme_log_price)
                extreme_dates.append(dates[extreme_bar_index])
                confirm_dates.append(dates[bar_index])
                pivot_directions.append(-1)
                last_pivot_bar_index = bar_index
                current_direction = 1
                extreme_log_price = log_highs[bar_index]
                extreme_bar_index = bar_index

    return (
        np.array(log_pivot_prices),
        np.array(extreme_dates),
        np.array(confirm_dates),
        np.array(pivot_directions, dtype=np.int8),
    )


def build_query_vector(log_pivot_prices: np.ndarray, embedding_dim: int) -> np.ndarray:
    """Last embedding_dim leg log-returns of the target's own zigzag — the
    S-map localization vector (only used when theta != 0, see smap_weights)."""
    num_pivots = len(log_pivot_prices)
    if num_pivots < embedding_dim + 1:
        raise ValueError(
            f"Недостаточно пивотов в T_query: нужно >= {embedding_dim + 1}, есть {num_pivots}"
        )
    return np.array([
        log_pivot_prices[-1 - lag] - log_pivot_prices[-2 - lag]
        for lag in range(embedding_dim)
    ])


def smap_weights(query_vector: np.ndarray, pool_feature_matrix: np.ndarray, theta: float) -> np.ndarray:
    """
    Classical S-map localization by embedding-vector distance (Sugihara
    1994): w_j = exp(-theta * d_j / mean(d)). theta=0 (the calibrated
    default, see plan section 1 — OOS calibration of theta never survived
    on any of the 7 test tickers) returns all-ones, i.e. a strict no-op —
    forecast_live_band with theta=0 reproduces the λ-only kernel exactly.
    """
    if theta == 0 or len(pool_feature_matrix) == 0:
        return np.ones(len(pool_feature_matrix))
    distances = np.linalg.norm(pool_feature_matrix - query_vector, axis=1)
    mean_distance = distances.mean()
    if mean_distance < 1e-14:
        return np.ones(len(distances))
    return np.exp(-theta * distances / mean_distance)


# ── bar-native features (rank series computed once per ticker, full history) ─

def to_percentile_rank(series: np.ndarray, window: int = RANK_WINDOW) -> np.ndarray:
    """Causal percentile rank of series[i] against the PRECEDING `window`
    values (i itself excluded)."""
    n = len(series)
    ranks = np.full(n, np.nan)
    for i in range(window, n):
        hist = series[i - window:i]
        finite = hist[np.isfinite(hist)]
        if len(finite) < 10 or not np.isfinite(series[i]):
            continue
        ranks[i] = float((finite < series[i]).mean())
    return ranks


def compute_density_rank_series(
    log_highs: np.ndarray, log_lows: np.ndarray, volumes: np.ndarray,
    window: int = RANK_WINDOW, n_bins: int = N_BINS,
) -> np.ndarray:
    """Causal volume profile: percentile rank of the bar's mid-price within
    the trailing `window` bars' volume histogram."""
    n = len(log_highs)
    mid = (log_highs + log_lows) / 2.0
    ranks = np.full(n, np.nan)
    for i in range(window, n):
        lo_bound = log_lows[i - window:i].min()
        hi_bound = log_highs[i - window:i].max()
        if not (hi_bound > lo_bound):
            continue
        hist, _edges = np.histogram(
            mid[i - window:i], bins=n_bins,
            range=(lo_bound, hi_bound), weights=volumes[i - window:i],
        )
        total = hist.sum()
        if total <= 0:
            continue
        bin_idx = int((mid[i] - lo_bound) / (hi_bound - lo_bound) * n_bins)
        bin_idx = max(0, min(bin_idx, n_bins - 1))
        cum_below = hist[:bin_idx].sum()
        cum_at = hist[bin_idx]
        ranks[i] = (cum_below + cum_at / 2.0) / total
    return ranks


def compute_trend_raw(log_highs: np.ndarray, log_lows: np.ndarray, window_ma: int = 100) -> np.ndarray:
    mid = (log_highs + log_lows) / 2.0
    n = len(mid)
    trend = np.full(n, np.nan)
    csum = np.concatenate([[0.0], np.cumsum(mid)])
    for i in range(window_ma, n):
        ma = (csum[i] - csum[i - window_ma]) / window_ma
        trend[i] = mid[i] - ma
    return trend


def compute_velocity_raw(log_highs: np.ndarray, log_lows: np.ndarray, window_v: int = 10) -> np.ndarray:
    mid = (log_highs + log_lows) / 2.0
    n = len(mid)
    vel = np.full(n, np.nan)
    vel[window_v:] = mid[window_v:] - mid[:-window_v]
    return vel


def compute_acceleration_raw(log_highs: np.ndarray, log_lows: np.ndarray, window_v: int = 10) -> np.ndarray:
    vel = compute_velocity_raw(log_highs, log_lows, window_v)
    n = len(vel)
    acc = np.full(n, np.nan)
    acc[window_v:] = vel[window_v:] - vel[:-window_v]
    return acc


def compute_volatility_raw(log_highs: np.ndarray, log_lows: np.ndarray, window_vol: int = WINDOW_VOL) -> np.ndarray:
    import pandas as pd
    mid = (log_highs + log_lows) / 2.0
    ret = np.diff(mid, prepend=np.nan)
    return pd.Series(ret).rolling(window_vol, min_periods=window_vol).std().to_numpy()


def compute_bar_rank_dict(log_highs: np.ndarray, log_lows: np.ndarray, volumes: np.ndarray) -> dict[str, np.ndarray]:
    """All 5 bar-native rank series for one ticker, computed once over its
    full available history (each value only depends on data before it, so
    this is safe to compute up front and slice later — see mask_ticker_data)."""
    return {
        "volume":       compute_density_rank_series(log_highs, log_lows, volumes),
        "trend":        to_percentile_rank(compute_trend_raw(log_highs, log_lows)),
        "velocity":     to_percentile_rank(compute_velocity_raw(log_highs, log_lows)),
        "acceleration": to_percentile_rank(compute_acceleration_raw(log_highs, log_lows)),
        "volatility":   to_percentile_rank(compute_volatility_raw(log_highs, log_lows)),
    }


# ── pivot-native features (T-dependent — recomputed per zigzag build) ────────

def causal_pivot_percentile_rank(values: np.ndarray, k: int = K_LEG, min_hist: int = 3) -> np.ndarray:
    """Causal percentile rank of values[i] against the PRECEDING up-to-k
    values (i excluded). values[0] is always NaN."""
    n = len(values)
    ranks = np.full(n, np.nan)
    for i in range(1, n):
        lo = max(1, i - k)
        hist = values[lo:i]
        finite = hist[np.isfinite(hist)]
        if len(finite) < min_hist or not np.isfinite(values[i]):
            continue
        ranks[i] = float((finite < values[i]).mean())
    return ranks


def leg_age_raw(confirm_dates: np.ndarray, dates_full: np.ndarray) -> np.ndarray:
    """Bars spent forming each confirmed leg (age[0] is always NaN — no
    prior pivot to measure from)."""
    bar_idx = np.searchsorted(dates_full, confirm_dates)
    n = len(bar_idx)
    age = np.full(n, np.nan)
    if n > 1:
        age[1:] = (bar_idx[1:] - bar_idx[:-1]).astype(float)
    return age


# ── pool assembly (bar+pivot ranks, causally truncated per ticker) ───────────

def build_pool_vectors_with_ranks(
    pivot_prices: np.ndarray, pivot_dirs: np.ndarray, pivot_ranks: dict[str, np.ndarray],
    embedding_dim: int, horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    n = len(pivot_prices)
    valid = np.arange(embedding_dim, n - horizon)
    if len(valid) == 0:
        empty_ranks = {k: np.zeros(0) for k in pivot_ranks}
        return np.zeros((0, embedding_dim)), np.zeros(0), np.zeros(0, dtype=np.int8), empty_ranks
    lag_idx = valid[:, None] - np.arange(embedding_dim)[None, :]
    feats = pivot_prices[lag_idx] - pivot_prices[lag_idx - 1]
    tars = pivot_prices[valid + horizon] - pivot_prices[valid]
    dirs = pivot_dirs[valid]
    ranks = {k: v[valid] for k, v in pivot_ranks.items()}
    finite = np.all(np.isfinite(feats), axis=1) & np.isfinite(tars)
    for k in ranks:
        finite &= np.isfinite(ranks[k])
    out_ranks = {k: v[finite] for k, v in ranks.items()}
    return feats[finite], tars[finite], dirs[finite], out_ranks


def build_causal_pool_with_mixed_ranks(
    cutoff_date: str,
    ticker_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    t_pool: float, embedding_dim: int, horizon: int, min_bars: int,
    bar_feat_names: list[str] = BAR_FEATURES,
    pivot_feat_names: list[str] = PIVOT_FEATURES,
    k_leg: int = K_LEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    ticker_data: {ticker: (log_highs, log_lows, dates, bar_rank_dict)} — full
    (untruncated) history per ticker. Each ticker is masked to dates <=
    cutoff_date FIRST, before its zigzag/pool is built — this is the single
    causality checkpoint for the pool (see module docstring).
    """
    feats_l, tars_l, dirs_l = [], [], []
    all_feat_names = bar_feat_names + pivot_feat_names
    ranks_l: dict[str, list] = {f: [] for f in all_feat_names}
    for _ticker, (lh, ll, dates, bar_rank_dict) in ticker_data.items():
        mask = dates <= cutoff_date
        if mask.sum() < embedding_dim + horizon + 2:
            continue
        lh_c, ll_c, dt_c = lh[mask], ll[mask], dates[mask]
        bar_rank_c = {f: bar_rank_dict[f][mask] for f in bar_feat_names}
        pivot_prices, _extreme_dates, confirm_dates, pivot_dirs = build_zigzag(lh_c, ll_c, dt_c, t_pool, min_bars)
        if len(pivot_prices) < 3:
            continue
        bar_idx = np.searchsorted(dt_c, confirm_dates)
        pivot_ranks = {f: bar_rank_c[f][bar_idx] for f in bar_feat_names}
        for pf in pivot_feat_names:
            raw = leg_age_raw(confirm_dates, dt_c) if pf == "leg_age" else None
            pivot_ranks[pf] = causal_pivot_percentile_rank(raw, k_leg)
        feats, tars, dirs, ranks = build_pool_vectors_with_ranks(
            pivot_prices, pivot_dirs, pivot_ranks, embedding_dim, horizon)
        if len(tars):
            feats_l.append(feats); tars_l.append(tars); dirs_l.append(dirs)
            for f in all_feat_names:
                ranks_l[f].append(ranks[f])
    if not tars_l:
        empty_ranks = {f: np.zeros(0) for f in all_feat_names}
        return np.zeros((0, embedding_dim)), np.zeros(0), np.zeros(0, dtype=np.int8), empty_ranks
    out_ranks = {f: np.concatenate(v) for f, v in ranks_l.items()}
    return np.vstack(feats_l), np.concatenate(tars_l), np.concatenate(dirs_l), out_ranks


def mask_ticker_data(
    ticker_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    cutoff_date: str,
) -> dict:
    """
    Truncate EVERY ticker (including the target, if present) to dates <=
    cutoff_date — rank_dict is sliced by the same mask. Safe because every
    rank[i] only depends on data strictly before i (see to_percentile_rank /
    compute_density_rank_series), so truncating AFTER i never changes the
    value AT i. This is the mechanism that turns "forecast as of a past
    origin" into "forecast on data that literally stops there" — used both
    for live vs. historical origins (sma/api/task_manager.py) and verified
    bit-for-bit by causality_check_band_lambda_ref.py's approach.
    """
    out = {}
    for tk, (lh, ll, dates, rank_dict) in ticker_data.items():
        mask = dates <= cutoff_date
        if mask.sum() < 20:
            continue
        out[tk] = (lh[mask], ll[mask], dates[mask], {f: v[mask] for f, v in rank_dict.items()})
    return out


# ── quantiles ─────────────────────────────────────────────────────────────────

def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: tuple) -> dict:
    """Weighted quantiles of `values` (linear interpolation on the weighted
    empirical CDF, midpoint correction). Returns {quantile: value}."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    if w.sum() < 1e-14:
        median = float(np.median(values))
        return {q: median for q in quantiles}
    cumulative_weight = np.cumsum(w) - 0.5 * w
    cumulative_weight /= w.sum()
    band_values = np.interp(quantiles, cumulative_weight, v)
    return {q: float(x) for q, x in zip(quantiles, band_values)}


# ── live forecast ─────────────────────────────────────────────────────────────

def pool_values_and_weights(
    target: str, t: float, lambdas: dict[str, float],
    ticker_data: dict, origin_index: int | None = None,
    m: int = DEFAULT_M, min_bars: int = DEFAULT_MIN_BARS, theta: float = DEFAULT_THETA,
) -> dict | None:
    """
    Raw (values, weights) per horizon — the building block behind
    forecast_live_band, exposed separately so a UI can recompute arbitrary
    quantile levels reactively without re-running the (comparatively
    expensive) pool build. origin_index=None -> the LAST available pivot in
    ticker_data[target] (a "live, as of now" forecast — combined with
    mask_ticker_data by the caller, this doubles as "as of a past origin").

    Returns None or {"origin_date", "origin_extreme_date", "origin_price",
    "origin_log_price", "origin_direction", "default_step_widths_bars",
    "steps": {1: {"ok", "pool_size", "values", "weights"}, 2: {...}}}.

    default_step_widths_bars ({"step1", "step2"}) — NOT part of the price
    forecast: a plain causal median (no λ, no pool weighting) of how many
    bars the target ticker's own past legs of the relevant direction
    typically took, from pivot_time_band.py. Lets the UI pre-fill each
    zone's horizontal span with something ticker-specific instead of a
    fixed constant — still fully overridable by the width sliders.

    origin_date is the CONFIRMATION bar — the only causally valid point, and
    what every truncation/lookup in this module keys off. origin_price/
    origin_log_price are q_lp[origin], the EXTREME log price of that same
    pivot (build_zigzag stores the extreme price, not the confirm bar's
    price) — origin_extreme_date is the bar that price actually printed on.
    Both dates are returned so a UI can choose which one to anchor a visual
    marker to; showing origin_price at origin_date (confirm) on a candle
    chart is misleading (that price may not resemble anything trading near
    the confirm bar) — see project feedback 2026-08-07 ("22 апреля... а сам
    прогноз строится в точке 26 мая").
    """
    lh_t, ll_t, dates_t, rank_dict_t = ticker_data[target]
    q_lp, q_extreme, q_dates, q_dirs = build_zigzag(lh_t, ll_t, dates_t, t, min_bars)
    if len(q_lp) < m + 3:
        return None
    origin = origin_index if origin_index is not None else len(q_lp) - 1
    if origin < 0 or origin >= len(q_lp):
        return None

    q_rank_pivot_leg_age = causal_pivot_percentile_rank(leg_age_raw(q_dates, dates_t), K_LEG)
    origin_date = str(q_dates[origin])
    origin_bar = int(np.searchsorted(dates_t, origin_date))
    origin_bar = min(origin_bar, len(rank_dict_t[BAR_FEATURES[0]]) - 1)
    rank_query = {f: rank_dict_t[f][origin_bar] for f in BAR_FEATURES}
    rank_query["leg_age"] = q_rank_pivot_leg_age[origin]
    if any(not np.isfinite(v) for v in rank_query.values()):
        return None

    query_vector = build_query_vector(q_lp[: origin + 1], m) if theta != 0 else None
    origin_direction = int(q_dirs[origin])
    origin_log_price = float(q_lp[origin])

    median_by_dir = median_leg_durations_by_direction(dates_t, q_dates, q_dirs, origin)
    width1_bars, width2_bars = default_step_widths_bars(
        median_by_dir, origin_direction, DEFAULT_STEP_BARS_FALLBACK
    )

    result = {
        "origin_date": origin_date, "origin_extreme_date": str(q_extreme[origin]),
        "origin_price": float(np.exp(origin_log_price)),
        "origin_log_price": origin_log_price, "origin_direction": origin_direction,
        "default_step_widths_bars": {"step1": width1_bars, "step2": width2_bars},
        "steps": {},
    }
    for h in (1, 2):
        feats, ptr, pdir, pranks = build_causal_pool_with_mixed_ranks(
            origin_date, ticker_data, t, m, h, min_bars)
        mask = pdir == origin_direction
        if mask.sum() < m + 2:
            result["steps"][h] = {"ok": False, "pool_size": int(mask.sum())}
            continue
        sq = np.zeros(int(mask.sum()))
        for f in FEATURE_ORDER:
            sq += lambdas[f] * (pranks[f][mask] - rank_query[f]) ** 2
        w = np.exp(-sq)
        if theta != 0:
            w = w * smap_weights(query_vector, feats[mask], theta)
        result["steps"][h] = {"ok": True, "pool_size": int(mask.sum()), "values": ptr[mask], "weights": w}
    return result


def forecast_live_band(
    target: str, t: float, lambdas: dict[str, float], ticker_data: dict,
    origin_index: int | None = None, q_levels: tuple = Q_LEVELS,
    m: int = DEFAULT_M, min_bars: int = DEFAULT_MIN_BARS, theta: float = DEFAULT_THETA,
) -> dict | None:
    """
    The calibrated band for one origin: h=1 (leg departure) and h=2
    (departure + return), read directly from the causal pool's weighted
    quantile distribution — not a chained point-forecast (see plan section 1:
    "полоса читается из пула независимо и одновременно" at theta=0).

    Returns None if there aren't enough pivots/pool. Otherwise:
      {"origin_date", "origin_extreme_date", "origin_price", "origin_log_price",
       "origin_direction", "default_step_widths_bars", "steps": {1: {...}, 2: {...}}}
    where steps[h] is either {"ok": False, "pool_size": int} or
    {"ok": True, "pool_size": int, "band_log_return": {q: lr}, "band_price": {q: price},
     "pool_values": [lr, ...], "pool_weights": [w, ...]}.

    origin_date (confirmation bar) is the causal anchor — everything the
    model computes is truncated to it. origin_extreme_date is where
    origin_price actually printed (see pool_values_and_weights docstring) —
    a UI should anchor any visual marker/band placement to THAT date, not
    origin_date, or the price line won't line up with anything on the chart
    near it (project feedback 2026-08-07).

    pool_values/pool_weights (log-returns relative to origin, and their
    kernel weights) are included alongside the fixed-Q_LEVELS band so a UI
    can recompute arbitrary quantile levels reactively — weighted_quantile
    on the same (values, weights) pair with a different `quantiles` tuple —
    without re-running the expensive pool build. Mirrors
    prototype/forcaster/ui/app9.py's session_state-cached raw arrays +
    reactive build_zones (2026-07-08 design), ported into the persisted
    result instead of a UI-local cache.
    """
    raw = pool_values_and_weights(target, t, lambdas, ticker_data, origin_index, m, min_bars, theta)
    if raw is None:
        return None
    origin_log_price = raw["origin_log_price"]
    result = {
        "origin_date": raw["origin_date"], "origin_extreme_date": raw["origin_extreme_date"],
        "origin_price": raw["origin_price"],
        "origin_log_price": origin_log_price,
        "origin_direction": raw["origin_direction"],
        "default_step_widths_bars": raw["default_step_widths_bars"],
        "steps": {},
    }
    for h in (1, 2):
        s = raw["steps"][h]
        if not s["ok"]:
            result["steps"][h] = {"ok": False, "pool_size": s["pool_size"]}
            continue
        band_lr = weighted_quantile(s["values"], s["weights"], q_levels)
        band_price = {q: float(np.exp(origin_log_price + lr)) for q, lr in band_lr.items()}
        result["steps"][h] = {
            "ok": True, "pool_size": s["pool_size"],
            "band_log_return": band_lr, "band_price": band_price,
            "pool_values": s["values"].tolist(), "pool_weights": s["weights"].tolist(),
        }
    return result


# ── DB-facing loader (the only I/O in this module) ───────────────────────────

async def load_raw_ticker_arrays(
    db, instrument_rows: list[dict], interval: str,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    {ticker: (log_highs, log_lows, dates, volumes)} for each of
    instrument_rows (dicts with "id"/"ticker" keys — target + pool,
    already deduplicated by the caller). Tickers with too little history
    are silently skipped, same tolerance as the reference's file-missing
    case — pool building already tolerates missing peers.
    """
    from sma.core.db import get_candles

    out: dict[str, tuple] = {}
    for row in instrument_rows:
        candles = await get_candles(db, row["id"], interval)
        if len(candles) < 20:
            continue
        log_highs = np.log(np.array([c["high"] for c in candles], dtype=np.float64))
        log_lows = np.log(np.array([c["low"] for c in candles], dtype=np.float64))
        dates = np.array([c["begin"] for c in candles])
        volumes = np.array([float(c["volume"]) for c in candles], dtype=np.float64)
        out[row["ticker"]] = (log_highs, log_lows, dates, volumes)
    return out


def build_ticker_data(
    raw_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]]:
    """Adds bar-native rank_dict on top of load_raw_ticker_arrays' output —
    the CPU-heavy step, meant to run in an executor (see task_manager.py)."""
    out = {}
    for ticker, (lh, ll, dates, vol) in raw_arrays.items():
        out[ticker] = (lh, ll, dates, compute_bar_rank_dict(lh, ll, vol))
    return out
