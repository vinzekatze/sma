#!/usr/bin/env python3
"""
53_c0_uncertainty_bands.py

Validates σ_C0·√h uncertainty model for the C0 component.
C0 is the fast noise (periods ~2–8 bars) left over after filtering slow components.
It is unpredictable, but its cumulative contribution to the reconstructed price
should grow as σ_C0·√h (random walk).

Walk-forward on 8 MOEX 1d tickers, 50 origins each = 400 total.
For each origin:
  - Estimate σ_C0 causally (from history only, last 100 bars)
  - Compute actual cumsum_C0[h] using ground-truth future C0 values
  - Compare to σ_C0·√h prediction
  - Calibrate coverage factor k: |cumsum_C0| ≤ k·σ_C0·√h

Outputs:
  figures/53_c0_uncertainty_bands.png
"""

from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "prototype"))

from forcaster.forecast.filterbank import make_filter_bank

DATA_DIR = ROOT / "data" / "candles"
FIG_DIR  = ROOT / "research" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TICKERS   = ["SBER", "GAZP", "LKOH", "MGNT", "NVTK", "ROSN", "SNGS", "TATN"]
INTERVAL  = "1d"
N_ORIGINS = 50
VAL_H     = 10
MIN_K     = 400
SIGMA_WIN = 100   # rolling window for local σ_C0 estimate


# ── helpers ────────────────────────────────────────────────────────────────────

def _logtrend_causal(close: np.ndarray) -> np.ndarray:
    n   = len(close)
    lc  = np.log(np.maximum(close, 1e-10))
    t   = np.arange(n, dtype=np.float64)
    cn  = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a     = (cy - b * ct) / cn
    trend = np.exp(a + b * t)
    trend[:2] = close[:2]
    return trend


def _load_dratio(ticker: str) -> np.ndarray | None:
    path = DATA_DIR / ticker / f"{INTERVAL}.json"
    if not path.exists():
        return None
    close = np.array([float(c["close"]) for c in json.loads(path.read_text())],
                     dtype=np.float64)
    ratio = close / _logtrend_causal(close)
    return np.diff(ratio)


# ── Walk-forward data collection ───────────────────────────────────────────────

# actual_by_h[h-1] = list of actual cumsum_C0[1..h] values across all origins
actual_by_h = [[] for _ in range(VAL_H)]
band_by_h   = [[] for _ in range(VAL_H)]

# c0_all: per-ticker full C0 series for autocorr analysis
c0_series_for_acf: list[tuple[str, np.ndarray]] = []

print("Walk-forward: collecting C0 data…")

for ticker in TICKERS:
    dratio = _load_dratio(ticker)
    if dratio is None or len(dratio) < MIN_K + VAL_H + 1:
        print(f"  {ticker}: нет данных — пропуск")
        continue

    # Full-data causal filter bank (C0 for future bars is ground truth
    # because sosfilt is strictly causal: C0[k] is identical whether computed
    # from dratio[:N] or dratio[:N+M] for any M ≥ 0).
    c0_full = make_filter_bank(dratio)[0]
    c0_series_for_acf.append((ticker, c0_full))

    n_total = len(dratio)
    origins = np.linspace(MIN_K, n_total - VAL_H - 1, N_ORIGINS, dtype=int)

    for origin_k in origins:
        # Causal σ_C0: filter bank only up to origin_k
        c0_hist  = make_filter_bank(dratio[:origin_k])[0]
        win_st   = max(0, origin_k - SIGMA_WIN)
        sigma_c0 = float(np.std(c0_hist[win_st:], ddof=1))
        if sigma_c0 < 1e-12:
            continue

        # Ground-truth future C0 (causal filter applied to actual future dratio)
        c0_future = c0_full[origin_k: origin_k + VAL_H]

        for h_idx in range(VAL_H):         # h = h_idx + 1
            cumsum_h = float(np.sum(c0_future[:h_idx + 1]))
            band_h   = sigma_c0 * np.sqrt(h_idx + 1)
            actual_by_h[h_idx].append(cumsum_h)
            band_by_h[h_idx].append(band_h)

    print(f"  {ticker}: {len(origins)} origins, σ_C0 (last) = {float(np.std(c0_full[-SIGMA_WIN:])):.6f}")

actual_by_h = [np.array(v) for v in actual_by_h]
band_by_h   = [np.array(v) for v in band_by_h]
h_vals      = np.arange(1, VAL_H + 1)
n_obs_total = sum(len(a) for a in actual_by_h)
print(f"\nИтого наблюдений: {n_obs_total}")


# ── 1. Empirical std(cumsum_C0, h) vs σ_C0·√h ─────────────────────────────────

emp_std  = np.array([np.std(a, ddof=1) for a in actual_by_h])
pred_std = np.array([np.mean(b)         for b in band_by_h])
ratio_v  = emp_std / (pred_std + 1e-20)

print("\nEmpirical std(cumsum_C0) vs σ_C0·√h:")
print(f"{'h':>3}  {'emp×10⁴':>10}  {'pred×10⁴':>10}  {'ratio':>7}")
for h, e, p, r in zip(h_vals, emp_std, pred_std, ratio_v):
    print(f"{h:3d}  {e*1e4:10.4f}  {p*1e4:10.4f}  {r:7.3f}")

print(f"\nМеан calibration ratio: {ratio_v.mean():.3f} ± {ratio_v.std():.3f}")


# ── 2. Coverage calibration ────────────────────────────────────────────────────

all_actual = np.concatenate(actual_by_h)
all_band   = np.concatenate(band_by_h)

k_vals   = np.arange(0.5, 3.51, 0.05)
coverage = np.array([
    float(np.mean(np.abs(all_actual) <= k * all_band)) for k in k_vals
])

targets = [0.68, 0.80, 0.90]
k_best  = {}
for tgt in targets:
    idx = int(np.searchsorted(coverage, tgt))
    k_best[tgt] = float(k_vals[min(idx, len(k_vals) - 1)])

print("\nКалибровка k (|cumsum_C0| ≤ k·σ_C0·√h):")
for tgt, k in k_best.items():
    print(f"  {tgt*100:.0f}% покрытие → k = {k:.2f}")


# ── 3. Autocorrelation (Durbin-Watson) ─────────────────────────────────────────

print("\nАвтокорреляция C0 (Durbin-Watson; 2 = нет, <2 = положит., >2 = отриц.):")
dw_all = []
for ticker, c0 in c0_series_for_acf:
    denom = float(np.sum(c0 ** 2))
    dw    = float(np.sum(np.diff(c0) ** 2) / (denom + 1e-20))
    dw_all.append(dw)
    acf1 = float(np.corrcoef(c0[:-1], c0[1:])[0, 1])
    print(f"  {ticker}: DW={dw:.3f}  ACF(1)={acf1:+.3f}")
print(f"  Mean DW: {np.mean(dw_all):.3f}")


# ── 4. Figures ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("Script 53 — C0 Uncertainty Model Validation", fontsize=11, fontweight="bold")

# (a) Empirical vs √h prediction
ax = axes[0]
ax.plot(h_vals, emp_std * 1e4,  "o-", color="steelblue",  lw=1.8, ms=5, label="Empirical std")
ax.plot(h_vals, pred_std * 1e4, "s--", color="tomato",    lw=1.5, ms=4, label="σ_C0·√h (predicted)")
ax.set_xlabel("Horizon h (bars)"); ax.set_ylabel("std(cumsum C0) × 10⁴")
ax.set_title("Empirical vs √h model")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (b) Coverage vs k
ax = axes[1]
ax.plot(k_vals, coverage * 100, "-", color="steelblue", lw=1.5)
colors_t = ["#f4c542", "#f4a442", "#e05252"]
for tgt, col in zip(targets, colors_t):
    k_t = k_best[tgt]
    ax.axhline(tgt * 100, color=col, lw=0.9, ls="--")
    ax.axvline(k_t,        color=col, lw=0.9, ls="--")
    ax.annotate(f"k={k_t:.1f}→{tgt*100:.0f}%",
                xy=(k_t, tgt * 100), xytext=(k_t + 0.05, tgt * 100 - 4.5),
                fontsize=8, color=col)
ax.set_xlabel("k"); ax.set_ylabel("Coverage (%)")
ax.set_title("Coverage vs k  (±k·σ_C0·√h)")
ax.grid(True, alpha=0.3)

# (c) Calibration ratio emp/pred by h
ax = axes[2]
bars = ax.bar(h_vals, ratio_v, color=["steelblue" if 0.7 < r < 1.3 else "tomato"
                                       for r in ratio_v], alpha=0.75, width=0.7)
ax.axhline(1.0, color="tomato",   lw=1.5, ls="--", label="ratio=1 (ideal √h model)")
ax.axhline(1.0 + ratio_v.std(), color="grey", lw=0.8, ls=":", alpha=0.6)
ax.axhline(1.0 - ratio_v.std(), color="grey", lw=0.8, ls=":", alpha=0.6,
           label=f"±1 std ({ratio_v.std():.3f})")
ax.set_xlabel("Horizon h (bars)"); ax.set_ylabel("Empirical / Predicted")
ax.set_title("Calibration ratio  (should ≈ 1)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
ax.set_ylim(0, max(2.0, ratio_v.max() * 1.15))

plt.tight_layout()
out_fig = FIG_DIR / "53_c0_uncertainty_bands.png"
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nГрафик сохранён: {out_fig}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"Итог скр.53:")
print(f"  Calibration ratio:  {ratio_v.mean():.3f} ± {ratio_v.std():.3f}")
print(f"  k для 68% покрытия: {k_best[0.68]:.2f}")
print(f"  k для 80% покрытия: {k_best[0.80]:.2f}  ← рекомендован для прототипа")
print(f"  k для 90% покрытия: {k_best[0.90]:.2f}")
print(f"  Mean DW C0: {np.mean(dw_all):.3f}  "
      f"({'≈ i.i.d.' if 1.7 < np.mean(dw_all) < 2.3 else 'есть автокорреляция'})")
