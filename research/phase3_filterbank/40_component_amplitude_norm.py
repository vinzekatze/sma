"""
40 — Per-component амплитудная нормировка ПОСЛЕ filter bank.

Проблема скр.39: нормировка ДО filter bank не помогает:
  - d²ratio: 0.13× slow std → мартингал
  - z-score: 47× slow std → шум маскирует паттерны

Новая гипотеза: нестационарность медленных компонент dratio — по АМПЛИТУДЕ.
В трендовые периоды |C3-C5| велики, в боковые — малы. LA путает похожие
по форме, но разные по амплитуде паттерны.

Подход: нормировать каждую медленную компоненту по своей rolling_std ПОСЛЕ filter bank.
  z_ci[t] = comp_ci[t] / rolling_std(comp_ci, win_ci)
  LA на z_ci → z_hat_ci
  comp_hat_ci = z_hat_ci * sigma_ci(origin_k)
  dratio_hat  = sum(comp_hat_ci)  →  ratio[origin] + cumsum(dratio_hat)

Sweep: win ∈ {20, 50, 100, 200}  ×  компоненты {3,4,5}
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

WIN_GRID = [20, 50, 100, 200]
EPS      = 1e-10


# ── helpers ───────────────────────────────────────────────────────────────────
def rolling_std_causal(series: np.ndarray, win: int) -> np.ndarray:
    n   = len(series)
    out = np.full(n, np.nan)
    for i in range(win - 1, n):
        out[i] = float(np.std(series[i - win + 1: i + 1], ddof=1))
    return out


def make_fb(series: np.ndarray) -> np.ndarray:
    components = []; remaining = series.copy()
    for fc in CUTOFFS:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        components.append(remaining - low); remaining = low
    components.append(remaining)
    return np.array(components)


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


# ── baseline: dratio, без per-comp нормировки ─────────────────────────────────
def _fb_forecast_baseline(COMP, vo):
    hats = []
    for ci in SLOW_IDX:
        Xc, yc = build_delay_matrix(COMP[ci, :vo], P)
        if len(Xc) < XI:
            hats.append(np.zeros(VAL_H)); continue
        vecc = last_vector(COMP[ci, :vo], p=P).copy()
        hats.append(_lwr(Xc, yc, vecc, XI))
    return np.sum(hats, axis=0)


# ── per-comp нормировка ───────────────────────────────────────────────────────
def _fb_forecast_norm(COMP, SIGMA, sigma_at_origin, vo):
    """
    LWR на нормированных компонентах z_ci = comp_ci / sigma_ci.
    Реконструкция: z_hat_ci * sigma_ci(origin) → суммируем → dratio_hat.
    """
    hats = []
    for ci in SLOW_IDX:
        z_ci = COMP[ci] / (SIGMA[ci] + EPS)    # нормированная компонента
        z_ci = np.nan_to_num(z_ci, nan=0.0)    # NaN в начале (sigma не накоплена)

        Xc, yc = build_delay_matrix(z_ci[:vo], P)
        if len(Xc) < XI:
            hats.append(np.zeros(VAL_H)); continue
        vecc = last_vector(z_ci[:vo], p=P).copy()
        z_hat = _lwr(Xc, yc, vecc, XI)

        # Масштабировать обратно: текущая амплитуда компоненты
        s_orig = float(sigma_at_origin[ci])
        hats.append(z_hat * s_orig)
    return np.sum(hats, axis=0)


# ── MAPE ─────────────────────────────────────────────────────────────────────
def mape_dratio(hat, ratio, vo):
    r_hat  = float(ratio[vo + 1]) + np.cumsum(hat)
    actual = ratio[vo + 2: vo + 2 + VAL_H]
    n_ = min(len(r_hat), len(actual))
    if n_ == 0: return np.nan
    return float(np.mean(np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + EPS)))


# ── основной цикл ─────────────────────────────────────────────────────────────
RES = {t: {"base": []} | {w: [] for w in WIN_GRID} for t in TICKERS}

t_total = time.time()
for ticker in TICKERS:
    with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
        candles = json.load(f)
    df     = normalize(candles, window=MA_WIN).dropna(subset=["ma"]).reset_index(drop=True)
    ratio  = df["ratio"].values
    dratio = np.diff(ratio)

    t1 = time.time()
    COMP_BASE = make_fb(dratio)

    # Предвычислить sigma для каждой компоненты и каждого win
    norms = {}
    for win in WIN_GRID:
        SIGMA = {}
        for ci in SLOW_IDX:
            SIGMA[ci] = rolling_std_causal(COMP_BASE[ci], win)
        norms[win] = SIGMA

    max_sigma_offset = max(WIN_GRID)
    min_orig = max(P + XI + 5, max_sigma_offset + P + 5)
    max_orig = len(dratio) - VAL_H - 1
    origins  = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)

    for vo in origins:
        hat_b = _fb_forecast_baseline(COMP_BASE, vo)
        RES[ticker]["base"].append(mape_dratio(hat_b, ratio, vo))

        for win in WIN_GRID:
            SIGMA = norms[win]
            # sigma при vo (скалярное значение для масштабирования назад)
            sigma_orig = {ci: float(SIGMA[ci][vo]) if not np.isnan(SIGMA[ci][vo]) else 1e-6
                          for ci in SLOW_IDX}
            hat_n = _fb_forecast_norm(COMP_BASE, SIGMA, sigma_orig, vo)
            RES[ticker][win].append(mape_dratio(hat_n, ratio, vo))

    bm     = float(np.nanmedian(RES[ticker]["base"]))
    best_w = min(WIN_GRID, key=lambda w: float(np.nanmedian(RES[ticker][w])))
    best_m = float(np.nanmedian(RES[ticker][best_w]))
    print(f"{ticker:>5}  N={len(origins)}  base={bm:.5f}  "
          f"best_win={best_w:>3}  best={best_m:.5f}  "
          f"Δ={(best_m - bm) / bm * 100:+.1f}%  "
          + "  ".join(f"w{w}={np.nanmedian(RES[ticker][w]):.4f}" for w in WIN_GRID)
          + f"  {time.time() - t1:.1f}s")

print(f"\nTotal: {time.time() - t_total:.1f}s")


# ── сводная таблица ───────────────────────────────────────────────────────────
agg_base = float(np.nanmedian(np.concatenate([RES[t]["base"] for t in TICKERS])))

print(f"\n{'':>5}  {'base':>8}" + "".join(f"  w{w:>3}" for w in WIN_GRID))
print("─" * (14 + 8 * len(WIN_GRID)))

for ticker in TICKERS:
    bm  = float(np.nanmedian(RES[ticker]["base"]))
    row = f"{ticker:>5}  {bm:>8.5f}"
    for w in WIN_GRID:
        d = (np.nanmedian(RES[ticker][w]) - bm) / bm * 100
        row += f"  {d:>+5.1f}%"
    print(row)

print("─" * (14 + 8 * len(WIN_GRID)))
row = f"{'AGG':>5}  {agg_base:>8.5f}"
agg_deltas = []
for w in WIN_GRID:
    all_w    = np.concatenate([RES[t][w] for t in TICKERS])
    all_base = np.concatenate([RES[t]["base"] for t in TICKERS])
    med_w    = float(np.nanmedian(all_w))
    delta    = (med_w - agg_base) / agg_base * 100
    agg_deltas.append(delta)
    row += f"  {delta:>+5.1f}%"
print(row)

# Wilcoxon для лучшего win
all_base_agg = np.concatenate([RES[t]["base"] for t in TICKERS])
best_w_agg   = WIN_GRID[int(np.argmin(agg_deltas))]
all_best     = np.concatenate([RES[t][best_w_agg] for t in TICKERS])
ok = ~(np.isnan(all_best) | np.isnan(all_base_agg))
_, pv = _wilcox(all_best[ok] - all_base_agg[ok])
delta = (float(np.nanmedian(all_best)) - agg_base) / agg_base * 100
print(f"\n  Лучший win={best_w_agg}: MAPE={float(np.nanmedian(all_best)):.5f}  "
      f"Δ={delta:+.2f}%  Wilcoxon p={pv:.4f}")


# ── диагностика: std z_ci vs comp_ci ─────────────────────────────────────────
print("\n── Диагностика: std нормированных компонент (SBER, win=100) ──")
ticker = "SBER"
with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
    candles = json.load(f)
df     = normalize(candles, window=MA_WIN).dropna(subset=["ma"]).reset_index(drop=True)
dratio_s = np.diff(df["ratio"].values)
COMP_s   = make_fb(dratio_s)

for ci in SLOW_IDX:
    raw_std = np.std(COMP_s[ci])
    sigma   = rolling_std_causal(COMP_s[ci], 100)
    z_ci    = COMP_s[ci] / (sigma + EPS)
    z_ci    = np.nan_to_num(z_ci, nan=0.0)
    z_std   = np.std(z_ci[100:])   # после прогрева
    print(f"  C{ci}: raw_std={raw_std:.6f}  z_std={z_std:.3f}"
          f"  (амплитуда norm: {z_std/raw_std:.1f}×)")


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"Per-component амплитудная нормировка (ПОСЛЕ filter bank)  |  p={P}  val_h={VAL_H}",
    fontsize=12,
)

# 1. Δ%(win) aggregate
ax = axes[0]
colors = ["seagreen" if d < 0 else "tomato" for d in agg_deltas]
ax.bar([str(w) for w in WIN_GRID], agg_deltas, color=colors, alpha=0.85)
ax.axhline(0, color="black", lw=1)
ax.set_xlabel("rolling_std window"); ax.set_ylabel("Δ% vs baseline")
ax.set_title("Aggregate MAPE vs window")
ax.grid(alpha=0.3, axis="y")
for i, (w, d) in enumerate(zip(WIN_GRID, agg_deltas)):
    ax.text(i, d + (0.2 if d >= 0 else -0.6), f"{d:+.1f}%",
            ha="center", va="bottom", fontsize=9)

# 2. Тепловая карта
ax = axes[1]
delta_matrix = np.array([
    [(np.nanmedian(RES[t][w]) - np.nanmedian(RES[t]["base"])) / np.nanmedian(RES[t]["base"]) * 100
     for w in WIN_GRID]
    for t in TICKERS
])
im = ax.imshow(delta_matrix, aspect="auto", cmap="RdYlGn_r", vmin=-30, vmax=30)
ax.set_xticks(range(len(WIN_GRID))); ax.set_xticklabels([str(w) for w in WIN_GRID])
ax.set_yticks(range(len(TICKERS))); ax.set_yticklabels(TICKERS, fontsize=9)
ax.set_xlabel("window"); ax.set_title("Δ% по тикерам")
plt.colorbar(im, ax=ax, label="Δ%")
for i in range(len(TICKERS)):
    for j in range(len(WIN_GRID)):
        ax.text(j, i, f"{delta_matrix[i,j]:+.0f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(delta_matrix[i,j]) > 18 else "black")

# 3. Пример: C3 нормированный vs raw (SBER)
ax = axes[2]
ci_show = 3
win_show = best_w_agg
ticker = "SBER"
with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
    candles = json.load(f)
df_s    = normalize(candles, window=MA_WIN).dropna(subset=["ma"]).reset_index(drop=True)
dratio_s = np.diff(df_s["ratio"].values)
COMP_s   = make_fb(dratio_s)
sigma_s  = rolling_std_causal(COMP_s[ci_show], win_show)
z_ci_s   = np.nan_to_num(COMP_s[ci_show] / (sigma_s + EPS), nan=0.0)

x_tail = np.arange(-400, 0)
ax.plot(x_tail, COMP_s[ci_show][-400:] / np.std(COMP_s[ci_show][-400:]),
        label=f"C{ci_show} raw (norm)", color="steelblue", lw=1.5, alpha=0.9)
ax.plot(x_tail, z_ci_s[-400:] / np.std(z_ci_s[-400:]),
        label=f"C{ci_show} z(win={win_show})", color="darkorange", lw=1.2, alpha=0.8)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("bar"); ax.set_ylabel("normalized value")
ax.set_title(f"SBER C{ci_show}: raw vs amplitude-norm (win={win_show})\n(оба нормированы к σ=1 для сравнения)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "40_component_amplitude_norm.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
