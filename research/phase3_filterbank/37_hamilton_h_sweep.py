"""
37 — Sweep параметра h для Hamilton filter + filter bank.

Скрипт 36: h=5 → −9.4%, h=10 → нейтрально, h=20 → хуже.
Цель: найти точный оптимум, форму кривой MAPE(h), per-ticker оптимумы.

Проверяем только drcycle режим (rcycle заведомо хуже из скр.36).
h ∈ {1, 2, 3, 4, 5, 6, 7, 8, 10, 15}

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
N_LAGS       = 4

H_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15]


# ── Hamilton filter ───────────────────────────────────────────────────────────
def hamilton_cycle(series: np.ndarray, h: int, n_lags: int = N_LAGS) -> np.ndarray:
    """Возвращает только cycle = series - hamilton_trend."""
    n     = len(series)
    t_min = h + n_lags - 1
    t_range = np.arange(t_min, n)

    X = np.column_stack(
        [np.ones(len(t_range))]
        + [series[t_range - h - k] for k in range(n_lags)]
    )
    y    = series[t_range]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    trend = np.full(n, np.nan)
    for idx, t in enumerate(t_range):
        trend[t] = float(X[idx] @ beta)

    return series - trend   # cycle, NaN там где trend NaN


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
    n_ = min(len(r_hat), len(actual))
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
# RES[ticker][h] = array of MAPE; RES[ticker]["base"] = baseline array
RES = {t: {"base": []} | {h: [] for h in H_GRID} for t in TICKERS}

t_total = time.time()
for ticker in TICKERS:
    with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
        candles = json.load(f)
    df     = normalize(candles, window=MA_WIN).dropna(subset=["ma"]).reset_index(drop=True)
    ratio  = df["ratio"].values
    dratio = np.diff(ratio)

    t1 = time.time()

    COMP_BASE = make_fb(dratio)

    # Предвычислить cycle, trend, COMP_dr для каждого h
    ham = {}
    for h in H_GRID:
        cycle = hamilton_cycle(ratio, h)
        # trend = ratio - cycle (восстановить)
        htrend = ratio - cycle
        dcycle = np.diff(np.nan_to_num(cycle))
        ham[h] = {
            "cycle":  cycle,
            "trend":  htrend,
            "COMP_dr": make_fb(dcycle),
        }

    max_h_nan = max(h + N_LAGS for h in H_GRID)
    min_orig  = max(P + XI + 5, max_h_nan + P + 5)
    max_orig  = len(dratio) - VAL_H - 1
    origins   = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)

    for vo in origins:
        hat_b = _fb_forecast(COMP_BASE, vo)
        RES[ticker]["base"].append(mape_dratio(hat_b, ratio, vo))

        for h in H_GRID:
            hat_h = _fb_forecast(ham[h]["COMP_dr"], vo)
            RES[ticker][h].append(
                mape_drcycle(hat_h, ratio, ham[h]["cycle"], ham[h]["trend"], vo)
            )

    bm = float(np.nanmedian(RES[ticker]["base"]))
    best_h   = min(H_GRID, key=lambda h: float(np.nanmedian(RES[ticker][h])))
    best_med = float(np.nanmedian(RES[ticker][best_h]))
    print(f"{ticker:>5}  N={len(origins)}  base={bm:.5f}  "
          f"best_h={best_h}  best={best_med:.5f}  "
          f"Δ={(best_med-bm)/bm*100:+.1f}%  "
          + "  ".join(f"h{h}={np.nanmedian(RES[ticker][h]):.4f}" for h in H_GRID)
          + f"  {time.time()-t1:.1f}s")

print(f"\nTotal: {time.time()-t_total:.1f}s")


# ── сводная таблица Δ% по h ───────────────────────────────────────────────────
agg_base = float(np.nanmedian(np.concatenate([RES[t]["base"] for t in TICKERS])))
print(f"\n{'':>5}  {'base':>8}" + "".join(f"  h={h:>2}" for h in H_GRID))
print("─" * (14 + 7 * len(H_GRID)))

for ticker in TICKERS:
    bm = float(np.nanmedian(RES[ticker]["base"]))
    row = f"{ticker:>5}  {bm:>8.5f}"
    for h in H_GRID:
        d = (np.nanmedian(RES[ticker][h]) - bm) / bm * 100
        row += f"  {d:>+5.1f}%"
    print(row)

print("─" * (14 + 7 * len(H_GRID)))
row = f"{'AGG':>5}  {agg_base:>8.5f}"
agg_deltas = []
for h in H_GRID:
    all_h    = np.concatenate([RES[t][h] for t in TICKERS])
    all_base = np.concatenate([RES[t]["base"] for t in TICKERS])
    med_h    = float(np.nanmedian(all_h))
    delta    = (med_h - agg_base) / agg_base * 100
    agg_deltas.append(delta)
    row += f"  {delta:>+5.1f}%"
print(row)

# Wilcoxon для лучшего h
best_h_agg = H_GRID[int(np.argmin(agg_deltas))]
all_best   = np.concatenate([RES[t][best_h_agg] for t in TICKERS])
all_base_  = np.concatenate([RES[t]["base"] for t in TICKERS])
ok = ~(np.isnan(all_best) | np.isnan(all_base_))
_, pv = _wilcox(all_best[ok] - all_base_[ok])
print(f"\nОптимальный h_agg={best_h_agg}: "
      f"MAPE={float(np.nanmedian(all_best)):.5f}  "
      f"Δ={agg_deltas[H_GRID.index(best_h_agg)]:+.2f}%  "
      f"Wilcoxon p={pv:.4f}")


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"Hamilton h-sweep  |  drcycle + filter bank  |  p={P}  val_h={VAL_H}",
    fontsize=12,
)

# 1. Δ%(h) aggregate
ax = axes[0]
agg_arr = []
for h in H_GRID:
    all_h = np.concatenate([RES[t][h] for t in TICKERS])
    agg_arr.append(float(np.nanmedian(all_h)))
deltas_agg = [(m - agg_base) / agg_base * 100 for m in agg_arr]
colors = ["seagreen" if d < 0 else "tomato" for d in deltas_agg]
ax.bar([str(h) for h in H_GRID], deltas_agg, color=colors, alpha=0.85)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("h (Hamilton lag)"); ax.set_ylabel("Δ% vs baseline")
ax.set_title("Aggregate MAPE vs h")
ax.grid(alpha=0.3, axis="y")
for i, (h, d) in enumerate(zip(H_GRID, deltas_agg)):
    ax.text(i, d + (0.3 if d >= 0 else -0.8), f"{d:+.1f}%",
            ha="center", va="bottom", fontsize=7)

# 2. Δ%(h) per-ticker тепловая карта
ax = axes[1]
delta_matrix = np.array([
    [(np.nanmedian(RES[t][h]) - np.nanmedian(RES[t]["base"])) / np.nanmedian(RES[t]["base"]) * 100
     for h in H_GRID]
    for t in TICKERS
])
im = ax.imshow(delta_matrix, aspect="auto", cmap="RdYlGn_r",
               vmin=-40, vmax=40)
ax.set_xticks(range(len(H_GRID))); ax.set_xticklabels([str(h) for h in H_GRID])
ax.set_yticks(range(len(TICKERS))); ax.set_yticklabels(TICKERS, fontsize=9)
ax.set_xlabel("h"); ax.set_title("Δ% по тикерам (зелёный = лучше)")
plt.colorbar(im, ax=ax, label="Δ%")
for i in range(len(TICKERS)):
    for j in range(len(H_GRID)):
        ax.text(j, i, f"{delta_matrix[i,j]:+.0f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(delta_matrix[i,j]) > 20 else "black")

# 3. Кривые MAPE(h) per-ticker
ax = axes[2]
palette = plt.cm.tab10(np.linspace(0, 1, len(TICKERS)))
for i, ticker in enumerate(TICKERS):
    meds = [float(np.nanmedian(RES[ticker][h])) for h in H_GRID]
    base = float(np.nanmedian(RES[ticker]["base"]))
    ax.plot([str(h) for h in H_GRID], meds, "-o", color=palette[i],
            lw=1.5, ms=5, label=ticker)
    ax.axhline(base, color=palette[i], lw=0.8, ls="--", alpha=0.5)

ax.set_xlabel("h"); ax.set_ylabel("val_mape (медиана)")
ax.set_title("MAPE(h) per-ticker\n(пунктир = baseline)")
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "37_hamilton_h_sweep.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
