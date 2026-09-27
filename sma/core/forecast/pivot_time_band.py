"""
Auto-default step widths for band_lambda's UI zone geometry.

NOT part of the price forecast — a separate, deliberately uncalibrated
estimate of how many bars the h=1/h=2 zones should visually span by
default, from the TARGET TICKER'S OWN causal zigzag leg-duration history
(no cross-ticker pool, no lambda weighting). Operationalizes the pilot in
research/phase7_fractality/31_pivot_time_band/ (raw per-direction median
gap between confirmed pivots) — that pilot found the asymmetry between
up-leg and down-leg durations statistically significant (p=0.0149 on
SBER), which is why the median is taken PER DIRECTION, not pooled.

The user explicitly judged this "not a significant enough forecast" to
warrant its own lambda calibration (2026-08-26 conversation) — this stays
a plain median of a causal pool, nothing more.
"""
import numpy as np

MIN_LEGS_PER_DIRECTION = 5


def median_leg_durations_by_direction(
    dates_full: np.ndarray, q_dates: np.ndarray, q_dirs: np.ndarray, origin: int,
) -> dict[int, float]:
    """
    Causal median leg duration (bars), keyed by pivot direction (+1 top /
    -1 bottom), using only pivots up to and including `origin` — everything
    after origin is invisible, matching every other origin cutoff in
    band_lambda. A direction is omitted if fewer than
    MIN_LEGS_PER_DIRECTION past legs of it exist yet (too little history
    for a median to mean anything).
    """
    bar_idx = np.searchsorted(dates_full, q_dates[: origin + 1])
    if len(bar_idx) < 2:
        return {}
    gaps = np.diff(bar_idx)
    leg_dirs = q_dirs[1 : origin + 1]
    out: dict[int, float] = {}
    for direction in (1, -1):
        sample = gaps[leg_dirs == direction]
        if len(sample) >= MIN_LEGS_PER_DIRECTION:
            out[direction] = float(np.median(sample))
    return out


def zigzag_direction_stats(
    log_pivot_prices: np.ndarray, dates_full: np.ndarray, q_dates: np.ndarray, q_dirs: np.ndarray,
) -> dict:
    """
    Descriptive (non-causal — uses the FULL history, for a "current stats
    summary" display, not a live per-forecast default) leg statistics per
    direction: count, duration (bars), amplitude (%). Powers the "T
    statistics" panel next to the T-selector (2026-08-26 conversation) —
    the same per-direction split as median_leg_durations_by_direction/
    default_step_widths_bars above, since that pilot already found the
    asymmetry real (research/phase7_fractality/31_pivot_time_band).

    Amplitude is reported as an UNSIGNED percentage (size of the move,
    exp(|Δlog_price|) - 1) — direction is already conveyed by the up/down
    split, a signed number would just repeat that.
    """
    n_pivots = len(q_dirs)
    out: dict = {"n_total": n_pivots, "up": None, "down": None}
    if n_pivots < 2:
        return out

    bar_idx = np.searchsorted(dates_full, q_dates)
    durations = np.diff(bar_idx)
    log_amplitudes = np.abs(np.diff(log_pivot_prices))
    leg_dirs = q_dirs[1:]

    for name, direction in (("up", 1), ("down", -1)):
        mask = leg_dirs == direction
        n = int(mask.sum())
        if n == 0:
            out[name] = {"n": 0}
            continue
        dur = durations[mask].astype(np.float64)
        amp_pct = (np.expm1(log_amplitudes[mask])) * 100.0
        out[name] = {
            "n": n,
            "duration_bars": {
                "median": float(np.median(dur)),
                "mean": float(np.mean(dur)),
                "std": float(np.std(dur, ddof=1)) if n > 1 else 0.0,
            },
            "amplitude_pct": {
                "median": float(np.median(amp_pct)),
                "mean": float(np.mean(amp_pct)),
                "std": float(np.std(amp_pct, ddof=1)) if n > 1 else 0.0,
            },
        }
    return out


def default_step_widths_bars(
    median_by_direction: dict[int, float], origin_direction: int, fallback: float,
) -> tuple[float, float]:
    """
    width1 (h=1, "уход") — typical duration of the departure leg, whose
    direction is the OPPOSITE of the origin pivot's own type (zigzag
    strictly alternates).
    width2 (h=2, "уход+возврат") — typical duration of the return leg,
    same direction as the origin pivot's own type.
    Falls back to a fixed constant per side when that direction doesn't
    have enough history yet.
    """
    width1 = median_by_direction.get(-origin_direction, fallback)
    width2 = median_by_direction.get(origin_direction, fallback)
    return width1, width2
