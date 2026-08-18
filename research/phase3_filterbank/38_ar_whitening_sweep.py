"""
38 — Sweep порядка AR при h=1 (AR-whitening перед filter bank).

Скрипт 37: h=1 → −14.2% (оптимум). При h=1 Hamilton filter = AR(n_lags) whitening:
    hcycle[t] = ratio[t] - OLS(ratio[t] on ratio[t-1..t-n_lags])
    dhcycle = diff(hcycle) → filter bank → LA/LWR

Цель: найти оптимальный AR порядок.
n_lags ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20}

Параметры те же: p=20, xi=63, val_h=10, N=200, 8 тикеров 1d.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.stats import wilcoxon as _wilcox

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from sma.core.forecast.normalize import normalize
from sma.core.forecast.embedding import build_delay_matrix, last_vector

# ── параметры ─────────────────────────────────────────────────────────────────
TICKERS  = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL = "1d"
MA_WIN   = 1000
P        = 20
XI       = 3 * (P + 1)   # 63
VAL_H    = 10
N_EVAL   = 200

FILTER_ORDER = 4
CUTOFFS      = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
SLOW_IDX     = [3, 4, 5]

H_FIX    = 1   # зафиксировано по результатам скр.37
AR_GRID  = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]


# ── AR whitening (Hamilton h=1, n_lags=p_ar) ──────────────────────────────────
def ar_whiten(series: np.ndarray, p_ar: int) -> tuple[np.ndarray, np.ndarray]:
    """
    AR(p_ar) whitening: global OLS of ratio[t] on ratio[t-1..t-p_ar].
    Returns (htrend, hcycle). First p_ar rows are NaN.
    """
    n      = len(series)
    t_min  = p_ar   # первый t с полными лагами при h=1
    t_range = np.arange(t_min, n)

    X = np.column_stack(
        [np.ones(len(t_range))]
        + [series[t_range - k] for k in range(1, p_ar + 1)]
    )
    y    = series[t_range]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    trend = np.full(n, np.nan)
    for idx, t in enumerate(t_range):
        trend[t] = float(X[idx] @ beta)

    cycle = series - trend
    return trend, cycle


# ── filter bank ───────────────────────────────────────────────────────────────
def make_fb(series: np.ndarray) -> np.ndarray:
    components = []; remaining = series.copy()
    for fc in CUTOFFS:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        components.append(remaining - low); remaining = low
    components.append(remaining)
    return np.array(components)


# ── LWR ──────────────────────────────────────────────────────────────────────
def _lwr(X, y, vec, xi):
    v = vec.copy(); hat = np.empty(VAL_H)
    for step in range(VAL_H):
        dists  = np.linalg.norm(X - v, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        h_bw   = max(float(dists[nn_idx].max()), 1e-10)
        w      = np.exp(-0.5 * (dists[nn_idx] / h_bw) ** 2)
        A      = np.hstack([np.ones((xi, 1)), X[nn_idx]])
        sw     = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx], rcond=None)
        hat[step] = float(c[0] + v @ c[1:])
        v = np.roll(v, -1); v[-1] = hat[step]
    return hat


def _fb_forecast(COMP, vo):
    hats = []
    for ci in SLOW_IDX:
        Xc, yc = build_delay_matrix(COMP[ci, :vo], P)
        if len(Xc) < XI:
            hats.append(np.zeros(VAL_H)); continue
        vecc = last_vector(COMP[ci, :vo], p=P).copy()
        hats.append(_lwr(Xc, yc, vecc, XI))
    return np.sum(hats, axis=0)


# ── MAPE ─────────────────────────────────────────────────────────────────────
def mape_dratio(hat, ratio, vo):
    r_hat  = float(ratio[vo + 1]) + np.cumsum(hat)
    actual = ratio[vo + 2: vo + 2 + VAL_H]
    n_ = min(len(r_hat), len(actual));
    if n_ == 0: return np.nan
    return float(np.mean(np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)))


def mape_drcycle(hat, ratio, cycle, htrend, vo):
    c_hat  = float(cycle[vo + 1]) + np.cumsum(hat)
    r_hat  = c_hat + float(htrend[vo + 1])
    actual = ratio[vo + 2: vo + 2 + VAL_H]
    n_ = min(len(r_hat), len(actual))
    if n_ == 0: return np.nan
    return float(np.mean(np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)))


# ── основной цикл ─────────────────────────────────────────────────────────────
RES = {t: {"base": []} | {p: [] for p in AR_GRID} for t in TICKERS}

t_total = time.time()
for ticker in TICKERS:
    with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
        candles = json.load(f)
    df     = normalize(candles, window=MA_WIN).dropna(subset=["ma"]).reset_index(drop=True)
    ratio  = df["ratio"].values
    dratio = np.diff(ratio)

    t1 = time.time()
    COMP_BASE = make_fb(dratio)

    # Предвычислить cycle+COMP для каждого p_ar
    ar = {}
    for p_ar in AR_GRID:
        htrend, hcycle = ar_whiten(ratio, p_ar)
        dcycle = np.diff(np.nan_to_num(hcycle))
        ar[p_ar] = {
            "trend":   htrend,
            "cycle":   hcycle,
            "COMP_dr": make_fb(dcycle),
        }

    max_ar_nan = max(AR_GRID)
    min_orig   = max(P + XI + 5, max_ar_nan + P + 5)
    max_orig   = len(dratio) - VAL_H - 1
    origins    = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)

    for vo in origins:
        hat_b = _fb_forecast(COMP_BASE, vo)
        RES[ticker]["base"].append(mape_dratio(hat_b, ratio, vo))

        for p_ar in AR_GRID:
            hat_ar = _fb_forecast(ar[p_ar]["COMP_dr"], vo)
            RES[ticker][p_ar].append(
                mape_drcycle(hat_ar, ratio, ar[p_ar]["cycle"], ar[p_ar]["trend"], vo)
            )

    bm      = float(np.nanmedian(RES[ticker]["base"]))
    best_p  = min(AR_GRID, key=lambda p: float(np.nanmedian(RES[ticker][p])))
    best_m  = float(np.nanmedian(RES[ticker][best_p]))
    print(f"{ticker:>5}  N={len(origins)}  base={bm:.5f}  "
          f"best_p={best_p:>2}  best={best_m:.5f}  "
          f"Δ={(best_m-bm)/bm*100:+.1f}%  "
          + "  ".join(f"ar{p}={np.nanmedian(RES[ticker][p]):.4f}" for p in AR_GRID)
          + f"  {time.time()-t1:.1f}s")

print(f"\nTotal: {time.time()-t_total:.1f}s")


# ── сводная таблица Δ% ────────────────────────────────────────────────────────
agg_base = float(np.nanmedian(np.concatenate([RES[t]["base"] for t in TICKERS])))

print(f"\n{'':>5}  {'base':>8}" + "".join(f"  ar{p:>2}" for p in AR_GRID))
print("─" * (14 + 7 * len(AR_GRID)))

for ticker in TICKERS:
    bm = float(np.nanmedian(RES[ticker]["base"]))
    row = f"{ticker:>5}  {bm:>8.5f}"
    for p in AR_GRID:
        d = (np.nanmedian(RES[ticker][p]) - bm) / bm * 100
        row += f"  {d:>+5.1f}%"
    print(row)

print("─" * (14 + 7 * len(AR_GRID)))
row = f"{'AGG':>5}  {agg_base:>8.5f}"
agg_deltas = []
for p in AR_GRID:
    all_p    = np.concatenate([RES[t][p] for t in TICKERS])
    all_base = np.concatenate([RES[t]["base"] for t in TICKERS])
    med_p    = float(np.nanmedian(all_p))
    delta    = (med_p - agg_base) / agg_base * 100
    agg_deltas.append(delta)
    row += f"  {delta:>+5.1f}%"
print(row)

# Wilcoxon для лучшего и AR=4 (референс)
all_base_agg = np.concatenate([RES[t]["base"] for t in TICKERS])
best_p_agg   = AR_GRID[int(np.argmin(agg_deltas))]
for p_test in sorted({best_p_agg, 4}):
    all_p = np.concatenate([RES[t][p_test] for t in TICKERS])
    ok    = ~(np.isnan(all_p) | np.isnan(all_base_agg))
    _, pv = _wilcox(all_p[ok] - all_base_agg[ok])
    delta = (float(np.nanmedian(all_p)) - agg_base) / agg_base * 100
    print(f"\n  AR({p_test:>2}): MAPE={float(np.nanmedian(all_p)):.5f}  "
          f"Δ={delta:+.2f}%  Wilcoxon p={pv:.4f}")


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"AR-whitening sweep (h=1)  +  filter bank  |  p={P}  val_h={VAL_H}",
    fontsize=12,
)

# 1. Δ%(p_ar) aggregate
ax = axes[0]
colors = ["seagreen" if d < 0 else "tomato" for d in agg_deltas]
ax.bar([str(p) for p in AR_GRID], agg_deltas, color=colors, alpha=0.85)
ax.axhline(0, color="black", lw=1)
ax.axvline(AR_GRID.index(4) - 0.5 + 1, color="navy", lw=1, ls="--", alpha=0.5,
           label="AR(4) — Hamilton из скр.37")
ax.set_xlabel("AR порядок (n_lags)"); ax.set_ylabel("Δ% vs baseline")
ax.set_title("Aggregate MAPE vs AR порядок")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
for i, (p, d) in enumerate(zip(AR_GRID, agg_deltas)):
    ax.text(i, d + (0.3 if d >= 0 else -0.8), f"{d:+.1f}%",
            ha="center", va="bottom", fontsize=7)

# 2. Тепловая карта Δ% по тикерам
ax = axes[1]
delta_matrix = np.array([
    [(np.nanmedian(RES[t][p]) - np.nanmedian(RES[t]["base"])) / np.nanmedian(RES[t]["base"]) * 100
     for p in AR_GRID]
    for t in TICKERS
])
im = ax.imshow(delta_matrix, aspect="auto", cmap="RdYlGn_r",
               vmin=-40, vmax=40)
ax.set_xticks(range(len(AR_GRID))); ax.set_xticklabels([str(p) for p in AR_GRID])
ax.set_yticks(range(len(TICKERS))); ax.set_yticklabels(TICKERS, fontsize=9)
ax.set_xlabel("AR порядок"); ax.set_title("Δ% по тикерам (зелёный = лучше)")
plt.colorbar(im, ax=ax, label="Δ%")
for i in range(len(TICKERS)):
    for j in range(len(AR_GRID)):
        ax.text(j, i, f"{delta_matrix[i,j]:+.0f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(delta_matrix[i,j]) > 25 else "black")

# 3. Кривые MAPE(p_ar) per-ticker
ax = axes[2]
palette = plt.cm.tab10(np.linspace(0, 1, len(TICKERS)))
for i, ticker in enumerate(TICKERS):
    meds = [float(np.nanmedian(RES[ticker][p])) for p in AR_GRID]
    base = float(np.nanmedian(RES[ticker]["base"]))
    ax.plot([str(p) for p in AR_GRID], meds, "-o", color=palette[i],
            lw=1.5, ms=5, label=ticker)
    ax.axhline(base, color=palette[i], lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("AR порядок"); ax.set_ylabel("val_mape (медиана)")
ax.set_title("MAPE(p_ar) per-ticker\n(пунктир = baseline)")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "38_ar_whitening_sweep.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
