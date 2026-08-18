"""
Spectrogram analyzer — STFT of Δratio, for eyeballing how "noisy" an
instrument is and how its spectral content shifts over time. First of
what's meant to be a small family of Analysis-tab analyzers (see
sma/core/analysis/ — one module per analyzer, no shared base class/registry
yet: YAGNI while there's exactly one, same call made for TaskManager's
kind dispatch — add a second analyzer by adding a second module + route
branch, not by building the abstraction preemptively).

Ported from prototype/forcaster/ui/app.py's "Спектрограмма Δratio" expander
(2026-06, still present unmodified in later prototype revisions before the
UI itself moved on) — same scipy call, same causal logtrend signal (see
sma/core/forecast/normalize.py), same filter-bank cutoff reference lines.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram as _scipy_spectrogram

from ..forecast.normalize import normalize

MIN_DRATIO_POINTS = 32  # same floor the prototype used

# Filter-bank cutoff frequencies (see CLAUDE.md "Filter bank"): causal
# Butterworth Wn values for the C0..C5 band edges. fc = Wn * Nyquist = Wn/2
# at fs=1 (one sample per bar). Drawn as horizontal reference lines on the
# spectrogram so a period band the LP-attractor pipeline cares about is
# visible at a glance.
_CUTOFF_WN = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
_CUTOFF_LABELS = ["C0/C1", "C1/C2", "C2/C3", "C3/C4", "C4/C5"]


def filter_bank_cutoffs() -> list[dict]:
    """[{label, freq (cycles/bar), period_bars}, ...] for the C0..C5 band edges."""
    out = []
    for wn, label in zip(_CUTOFF_WN, _CUTOFF_LABELS):
        fc = wn / 2.0
        out.append({"label": label, "freq": fc, "period_bars": round(1.0 / fc)})
    return out


def compute_spectrogram(
    candles: list[dict],
    depth_bars: int = 0,
    nperseg: int = 64,
    overlap_pct: float = 75.0,
    fmin: float = 0.0,
    fmax: float = 0.5,
) -> dict:
    """
    STFT of causal Δratio (see sma.core.forecast.normalize) over the last
    `depth_bars` candles (0 or >= len(candles) -> full history).

    Returns {"times": [...ISO date str, one per STFT time-step],
    "freqs": [...cycles/bar], "sxx_db": [[...]] (freq-major, matches
    scipy.signal.spectrogram's own axis order), "cutoffs": filter_bank_cutoffs(),
    "meta": {"n_bars", "freq_resolution", "time_step_bars", "nperseg", "noverlap"}}.

    Raises ValueError if there isn't enough history (< MIN_DRATIO_POINTS
    Δratio points) or nperseg is too large for the available data.
    """
    df = normalize(candles)
    if depth_bars and depth_bars > 0:
        df = df.iloc[-depth_bars:].reset_index(drop=True)

    ratio = df["ratio"].to_numpy(dtype=np.float64)
    dratio = np.diff(ratio)
    n = len(dratio)
    if n < MIN_DRATIO_POINTS:
        raise ValueError(
            f"Слишком короткий период ({n} баров Δratio). Нужно минимум {MIN_DRATIO_POINTS}."
        )
    if nperseg > n:
        raise ValueError(f"Окно STFT ({nperseg}) больше доступной истории ({n} баров Δratio).")

    noverlap = int(nperseg * overlap_pct / 100)
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    f_spec, t_spec, sxx = _scipy_spectrogram(
        dratio, fs=1.0, nperseg=nperseg, noverlap=noverlap,
        scaling="density", window="hann",
    )
    sxx_db = 10 * np.log10(sxx + 1e-20)

    # dratio[i] = ratio[i+1] - ratio[i], dated by df["begin"][i+1] — the bar
    # the increment is realized on. t_spec (scipy's STFT frame centers, in
    # samples) is rounded to the nearest dratio index for a date label.
    dates = df["begin"].to_numpy()[1:]
    t_idx = np.round(t_spec).astype(int).clip(0, n - 1)
    times = [str(d)[:10] for d in dates[t_idx]]

    if fmin >= fmax:
        fmin, fmax = 0.0, 0.5
    fmask = (f_spec >= fmin) & (f_spec <= fmax)

    return {
        "times": times,
        "freqs": f_spec[fmask].tolist(),
        "sxx_db": sxx_db[fmask, :].tolist(),
        "cutoffs": filter_bank_cutoffs(),
        "meta": {
            "n_bars": n,
            "freq_resolution": 1.0 / nperseg,
            "time_step_bars": nperseg - noverlap,
            "nperseg": nperseg,
            "noverlap": noverlap,
        },
    }
