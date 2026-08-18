"""
39 — Rolling Z-score нормализация dratio перед filter bank.

Проблема скр.38: d²ratio overdifference → медленные компоненты C3-C5 ~в 5-10× меньше,
LA предсказывает ~0, MAPE улучшается за счёт мартингальной стратегии, а не паттернов.

Гипотеза: z[t] = dratio[t] / sigma[t], где sigma = rolling_std(dratio, win).
Нормирование по волатильности делает паттерны при разных σ сопоставимыми для LA.
Медленные компоненты z сохраняют направленную структуру dratio, но без амплитудного дрейфа.
Реконструкция: dratio_hat = z_hat * sigma(origin_k) → ratio[origin] + cumsum(dratio_hat).

Sweep: win ∈ {10, 20, 30, 50, 100}
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

WIN_GRID = [10, 20, 30, 50, 100]
EPS      = 1e-10   # защита от деления на ноль


# ── rolling std (causal) ──────────────────────────────────────────────────────
def rolling_std(series: np.ndarray, win: int) -> np.ndarray:
    """Causal rolling std. Первые win-1 точек = NaN."""
    n   = len(series)
    out = np.full(n, np.nan)
    for i in range(win - 1, n):
        out[i] = float(np.std(series[i - win + 1: i + 1], ddof=1))
    return out


def make_zscore(dratio: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Causal rolling Z-score нормализация.
    Возвращает (z, sigma).
    Первые win-1 точек z = NaN; fill NaN → 0 для filter bank.
    """
    sigma = rolling_std(dratio, win)
    z     = dratio / (sigma + EPS)
    return z, sigma


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
    """Baseline: dratio forecast → ratio[vo+1] + cumsum."""
    r_hat  = float(ratio[vo + 1]) + np.cumsum(hat)
    actual = ratio[vo + 2: vo + 2 + VAL_H]
    n_ = min(len(r_hat), len(actual))
    if n_ == 0: return np.nan
    return float(np.mean(np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + EPS)))


def mape_zscore(z_hat, sigma_origin, ratio, vo):
    """
    Z-score forecast → dratio_hat = z_hat * sigma_origin → ratio reconstruction.
    Однократный cumsum от ratio[vo+1].
    """
    dratio_hat = z_hat * sigma_origin
    r_hat      = float(ratio[vo + 1]) + np.cumsum(dratio_hat)
    actual     = ratio[vo + 2: vo + 2 + VAL_H]
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

    # Предвычислить z и COMP для каждого win
    zs = {}
    for win in WIN_GRID:
        z, sigma = make_zscore(dratio, win)
        z_clean  = np.nan_to_num(z, nan=0.0)   # NaN в начале → 0
        zs[win] = {
            "z":     z,
            "sigma": sigma,
            "COMP":  make_fb(z_clean),
        }

    max_nan_offset = max(WIN_GRID)
    min_orig   = max(P + XI + 5, max_nan_offset + P + 5)
    max_orig   = len(dratio) - VAL_H - 1
    origins    = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)

    for vo in origins:
        hat_b = _fb_forecast(COMP_BASE, vo)
        RES[ticker]["base"].append(mape_dratio(hat_b, ratio, vo))

        for win in WIN_GRID:
            sigma_origin = float(zs[win]["sigma"][vo]) if not np.isnan(zs[win]["sigma"][vo]) else 1e-4
            hat_z = _fb_forecast(zs[win]["COMP"], vo)
            RES[ticker][win].append(
                mape_zscore(hat_z, sigma_origin, ratio, vo)
            )

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


# ── диагностика: std медленных компонент ─────────────────────────────────────
print("\n── std медленных компонент (C3+C4+C5) для последнего тикера ──")
ticker = "SBER"
with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
    candles = json.load(f)
df     = normalize(candles, window=MA_WIN).dropna(subset=["ma"]).reset_index(drop=True)
ratio  = df["ratio"].values
dratio = np.diff(ratio)

COMP_D   = make_fb(dratio)
slow_d   = sum(COMP_D[ci] for ci in SLOW_IDX)
print(f"  dratio slow C3-C5 std: {np.std(slow_d):.6f}  (ratio = 1.0)")

d2ratio  = np.diff(dratio)
COMP_D2  = make_fb(d2ratio)
slow_d2  = sum(COMP_D2[ci] for ci in SLOW_IDX)
print(f"  d2ratio slow C3-C5 std: {np.std(slow_d2):.6f}  "
      f"(ratio vs dratio = {np.std(slow_d2)/np.std(slow_d):.3f}×)")

for win in WIN_GRID:
    z, _ = make_zscore(dratio, win)
    z_c  = np.nan_to_num(z, nan=0.0)
    COMP_Z = make_fb(z_c)
    slow_z = sum(COMP_Z[ci] for ci in SLOW_IDX)
    print(f"  z(win={win:>3}) slow C3-C5 std: {np.std(slow_z):.6f}  "
          f"(ratio vs dratio = {np.std(slow_z)/np.std(slow_d):.3f}×)")


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"Rolling Z-score нормализация dratio + filter bank  |  p={P}  val_h={VAL_H}",
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
            ha="center", va="bottom", fontsize=8)

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
ax.set_xlabel("window"); ax.set_title("Δ% по тикерам (зелёный = лучше)")
plt.colorbar(im, ax=ax, label="Δ%")
for i in range(len(TICKERS)):
    for j in range(len(WIN_GRID)):
        ax.text(j, i, f"{delta_matrix[i,j]:+.0f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(delta_matrix[i,j]) > 18 else "black")

# 3. Пример медленной компоненты: dratio vs z(win=best) для SBER
ax = axes[2]
with open(DATA_DIR / "SBER" / f"{INTERVAL}.json") as f:
    candles = json.load(f)
df     = normalize(candles, window=MA_WIN).dropna(subset=["ma"]).reset_index(drop=True)
dratio_sber = np.diff(df["ratio"].values)
COMP_D_s = make_fb(dratio_sber)
slow_d_s = sum(COMP_D_s[ci] for ci in SLOW_IDX)

z_best, _ = make_zscore(dratio_sber, best_w_agg)
COMP_Z_s  = make_fb(np.nan_to_num(z_best, nan=0.0))
slow_z_s  = sum(COMP_Z_s[ci] for ci in SLOW_IDX)

x = np.arange(len(slow_d_s))
x_tail = x[-400:]
ax.plot(x_tail, slow_d_s[-400:], label=f"dratio slow (std={np.std(slow_d_s):.4f})",
        color="steelblue", lw=1.5, alpha=0.9)
ax2 = ax.twinx()
ax2.plot(x_tail, slow_z_s[-400:], label=f"z(win={best_w_agg}) slow",
         color="darkorange", lw=1.2, alpha=0.8)
ax.set_xlabel("bar"); ax.set_ylabel("dratio slow", color="steelblue")
ax2.set_ylabel(f"z(win={best_w_agg}) slow", color="darkorange")
ax.set_title(f"SBER: медленные компоненты C3-C5\ndratio vs z(win={best_w_agg})")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "39_rolling_zscore.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
