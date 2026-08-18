"""
52 — Оптимизация γ для damped AR(BIC) на шумовых компонентах C1+C2.

Из скр.51: damped_07 (γ=0.70) = −4.58% — лучший результат на C1+C2.
Цель скрипта:
  1. Свип γ = 0.1..0.95 → найти оптимальное глобальное γ
  2. Пер-компонентная оптимизация: γ_C1 × γ_C2 (сетка)
  3. "ar1_only" — AR(BIC) на h=1, далее ноль (крайний случай γ→0)
  4. Лучшая комбинация vs ar_bic_c12 и damped_07_c12 из скр.51

Медленные C3–C5: LWR p=20 ξ=63.
Walk-forward: N_ORIG=50 × 8 тикеров, 1d, logtrend, val_h=10.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL     = "1d"
N_ORIG       = 50
VAL_H        = 10
P_SLOW       = 20
XI_SLOW      = 63
P_BIC_MAX    = 20
EPS          = 1e-10
STD_CUTOFFS  = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
FILTER_ORDER = 4

# γ-свип: глобальный и пер-компонентный
GAMMA_SWEEP = np.round(np.arange(0.10, 0.96, 0.05), 2).tolist()  # 0.10..0.95
GRID_G      = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]    # сетка для γ_C1 × γ_C2

# ── вспомогательные функции ───────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    lc = np.log(close + EPS)
    n  = len(lc)
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t**2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = a + b * t
    trend[:2] = lc[:2]
    return np.exp(trend)


def make_fb(series: np.ndarray) -> np.ndarray:
    comps = []; rem = series.copy()
    for fc in STD_CUTOFFS:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, rem)
        comps.append(rem - low); rem = low
    comps.append(rem)
    return np.array(comps)


def fit_ar(series: np.ndarray, p: int) -> np.ndarray:
    n = len(series)
    if n <= p + 1:
        return np.zeros(p + 1)
    X = np.zeros((n - p, p))
    for lag in range(p):
        X[:, lag] = series[p - 1 - lag: n - 1 - lag]
    X = np.hstack([np.ones((n - p, 1)), X])
    c, _, _, _ = np.linalg.lstsq(X, series[p:], rcond=None)
    return c


def forecast_ar(series: np.ndarray, p: int, horizon: int,
                coeffs: np.ndarray) -> np.ndarray:
    buf = list(series[-p:])
    out = np.empty(horizon)
    for h in range(horizon):
        val = coeffs[0] + sum(coeffs[1 + k] * buf[-(k + 1)] for k in range(p))
        out[h] = val
        buf.append(val)
    return out


def fit_ar_bic(series: np.ndarray, p_max: int = P_BIC_MAX) -> tuple[int, np.ndarray]:
    n = len(series)
    best_bic, best_p, best_c = np.inf, 1, np.zeros(2)
    for p in range(1, min(p_max + 1, (n - 1) // 4)):
        c   = fit_ar(series, p)
        nef = n - p
        X   = np.zeros((nef, p))
        for lag in range(p):
            X[:, lag] = series[p - 1 - lag: n - 1 - lag]
        X   = np.hstack([np.ones((nef, 1)), X])
        ssr = np.sum((series[p:] - X @ c) ** 2)
        bic = nef * np.log(ssr / nef + EPS) + (p + 1) * np.log(nef)
        if bic < best_bic:
            best_bic, best_p, best_c = bic, p, c
    return best_p, best_c


def forecast_lwr_comp(series: np.ndarray, p: int, xi: int,
                      horizon: int) -> np.ndarray:
    n  = len(series)
    X  = np.array([series[i: i + p] for i in range(n - p)])
    y  = series[p:]
    if len(X) < xi:
        return np.zeros(horizon)
    vec = series[-p:].copy()
    out = np.empty(horizon)
    for h in range(horizon):
        dists  = np.linalg.norm(X - vec, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        nn_d   = dists[nn_idx]
        h_bw   = max(nn_d.max(), 1e-12)
        w      = np.exp(-0.5 * (nn_d / h_bw) ** 2)
        A      = np.hstack([np.ones((xi, 1)), X[nn_idx]])
        ws     = np.sqrt(w)
        coef, _, _, _ = np.linalg.lstsq(ws[:, None] * A, ws * y[nn_idx], rcond=None)
        val    = float(coef[0] + vec @ coef[1:])
        out[h] = val
        vec    = np.roll(vec, -1); vec[-1] = val
    return out


# ── walk-forward — собираем raw AR-прогнозы + slow_hat ───────────────────────
# Храним по origins: slow_hat, ar_raw_C1, ar_raw_C2, ar1_only_C1, ar1_only_C2
# Это позволит применять любое γ постфактум без повторного запуска

print("Walk-forward по тикерам (сбор AR-прогнозов)...")

# хранилища
slow_hats   = []         # slow_hat per origin
ar_raw      = {1: [], 2: []}   # ar_raw[ci] = список ndarray shape (VAL_H,)
ratios_0    = []         # ratio[origin_k]
actual_ratios = []       # ratio[origin_k+1..origin_k+VAL_H]

for ticker in TICKERS:
    print(f"  {ticker}", end="", flush=True)
    with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
        candles = json.load(f)
    close  = np.array([x["close"] for x in candles], dtype=np.float64)
    ratio  = close / logtrend_causal(close)
    dratio = np.diff(ratio)
    N      = len(dratio)

    origin_min = max(500, N // 2)
    origins    = np.linspace(origin_min, N - VAL_H - 5, N_ORIG, dtype=int)

    for origin_k in origins:
        comps_hist = make_fb(dratio[:origin_k])

        # Медленные C3–C5
        sh = np.zeros(VAL_H)
        for ci in [3, 4, 5]:
            ch = comps_hist[ci]
            if len(ch) >= XI_SLOW + P_SLOW + 2:
                sh += forecast_lwr_comp(ch, P_SLOW, XI_SLOW, VAL_H)
        slow_hats.append(sh)

        # AR(BIC) на C1 и C2
        for ci in [1, 2]:
            ch = comps_hist[ci]
            if len(ch) > P_BIC_MAX + 2:
                bp, bc  = fit_ar_bic(ch)
                raw     = forecast_ar(ch, bp, VAL_H, bc)
            else:
                raw = np.zeros(VAL_H)
            ar_raw[ci].append(raw)

        ratios_0.append(ratio[origin_k])
        actual_ratios.append(ratio[origin_k + 1: origin_k + VAL_H + 1])

    print(" ✓")

slow_hats     = np.array(slow_hats)        # (N_total, VAL_H)
ar_raw[1]     = np.array(ar_raw[1])        # (N_total, VAL_H)
ar_raw[2]     = np.array(ar_raw[2])        # (N_total, VAL_H)
ratios_0      = np.array(ratios_0)         # (N_total,)
actual_ratios = np.array(actual_ratios)    # (N_total, VAL_H)

N_total = len(slow_hats)
print(f"\n  Итого origins: {N_total}")

# ── вычисляем MAPE по γ постфактум (быстро, нет повторного запуска) ──────────

decay_matrix = np.array([g ** np.arange(1, VAL_H + 1) for g in GAMMA_SWEEP])
# shape: (len(GAMMA_SWEEP), VAL_H)

def mape_from_dhat(dhat_batch: np.ndarray) -> float:
    """dhat_batch: (N_total, VAL_H) → средний MAPE."""
    rh  = ratios_0[:, None] + np.cumsum(dhat_batch, axis=1)   # (N, VAL_H)
    err = np.abs(rh - actual_ratios) / (np.abs(actual_ratios) + EPS)
    return float(np.mean(err))

# baseline: std (slow only)
mape_std = mape_from_dhat(slow_hats)
print(f"\n  baseline std:  {mape_std:.5f}")

# ar_bic_c12 (из скр.51, γ=1.0 без затухания)
mape_ar_bic = mape_from_dhat(slow_hats + ar_raw[1] + ar_raw[2])
print(f"  ar_bic_c12:    {mape_ar_bic:.5f}  ({(mape_ar_bic-mape_std)/mape_std*100:+.2f}%)")

# ── 1. глобальный γ-свип ──────────────────────────────────────────────────────

print("\n  Глобальный γ-свип (один γ для C1 и C2):")
print(f"  {'γ':>6}  {'MAPE':>10}  {'Δ%':>9}")
print(f"  {'─'*30}")

global_mapes = []
for i, g in enumerate(GAMMA_SWEEP):
    d  = decay_matrix[i]        # (VAL_H,)
    dh = slow_hats + ar_raw[1] * d + ar_raw[2] * d
    m  = mape_from_dhat(dh)
    global_mapes.append(m)
    delta = (m - mape_std) / mape_std * 100
    flag  = " ★" if delta < -4.0 else (" ★" if delta < -3.0 else "")
    print(f"  {g:>6.2f}  {m:>10.5f}  {delta:>+8.2f}%{flag}")

best_g_idx  = int(np.argmin(global_mapes))
best_g      = GAMMA_SWEEP[best_g_idx]
best_g_mape = global_mapes[best_g_idx]
print(f"\n  ► Лучший глобальный γ = {best_g}  MAPE={best_g_mape:.5f}"
      f"  ({(best_g_mape-mape_std)/mape_std*100:+.2f}%)")

# ── 2. ar1_only: AR(BIC) только для h=1, h>1 → 0 ─────────────────────────────

mask_h1      = np.zeros(VAL_H); mask_h1[0] = 1.0
dh_ar1_only  = slow_hats + ar_raw[1] * mask_h1 + ar_raw[2] * mask_h1
mape_ar1only = mape_from_dhat(dh_ar1_only)
print(f"\n  ar1_only (h=1 AR, h>1 ноль): {mape_ar1only:.5f}"
      f"  ({(mape_ar1only-mape_std)/mape_std*100:+.2f}%)")

# ── 3. сетка γ_C1 × γ_C2 ─────────────────────────────────────────────────────

print(f"\n  Сетка γ_C1 × γ_C2 (реконструированный MAPE × 100):")
header = "     γC1\\γC2 " + "".join(f"  {g:>5.1f}" for g in GRID_G)
print(header)
print("  " + "─" * (len(header) - 2))

grid_mapes = np.zeros((len(GRID_G), len(GRID_G)))
best_pair  = (0.0, 0.0, np.inf)

for i, g1 in enumerate(GRID_G):
    row = f"    γC1={g1:.1f}  "
    for j, g2 in enumerate(GRID_G):
        d1  = g1 ** np.arange(1, VAL_H + 1)
        d2  = g2 ** np.arange(1, VAL_H + 1)
        dh  = slow_hats + ar_raw[1] * d1 + ar_raw[2] * d2
        m   = mape_from_dhat(dh)
        grid_mapes[i, j] = m
        if m < best_pair[2]:
            best_pair = (g1, g2, m)
        row += f"  {m*100:>5.3f}"
    print(row)

g1_best, g2_best, m_best = best_pair
print(f"\n  ► Лучшая пара: γ_C1={g1_best}  γ_C2={g2_best}"
      f"  MAPE={m_best:.5f}  ({(m_best-mape_std)/mape_std*100:+.2f}%)")

# ── итоговая таблица ─────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Итоговое сравнение (N={N_total} origins)")
print(f"  {'Метод':<28} {'MAPE':>10} {'Δ%':>10}")
print(f"  {'─'*50}")

summary = [
    ("std (baseline)", mape_std),
    ("ar_bic_c12", mape_ar_bic),
    ("damped_07_c12 (скр.51)", mape_from_dhat(
        slow_hats + ar_raw[1] * (0.7 ** np.arange(1, VAL_H+1))
                  + ar_raw[2] * (0.7 ** np.arange(1, VAL_H+1)))),
    (f"damped_{best_g}_c12 (лучший global)", best_g_mape),
    ("ar1_only_c12", mape_ar1only),
    (f"per-comp γC1={g1_best} γC2={g2_best}", m_best),
]
for name, m in summary:
    delta = (m - mape_std) / mape_std * 100
    flag  = " ★" if delta < -4.0 else (" ★" if delta < -2.0 else "  ")
    print(f"  {name:<28} {m:>10.5f} {delta:>+9.2f}%{flag}")

# ── рис 1: γ-свип ─────────────────────────────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(11, 5))
fig1.suptitle(
    f"Глобальный γ-свип: damped AR(BIC) на C1+C2\n"
    f"8 тикеров × {N_ORIG} origins, 1d, logtrend, val_h={VAL_H}",
    fontsize=12, fontweight="bold",
)
ax1.plot(GAMMA_SWEEP, [m * 100 for m in global_mapes],
         color="#42a5f5", lw=2.5, marker="o", ms=6, zorder=3)
ax1.axhline(mape_std * 100, color="#9e9e9e", lw=1.2, ls="--", label="std (baseline)")
ax1.axhline(mape_ar_bic * 100, color="#00897b", lw=1.2, ls="-.", label="ar_bic_c12")
ax1.axvline(best_g, color="#ffd600", lw=1.5, ls=":", zorder=2,
            label=f"best γ={best_g}")
ax1.annotate(f"γ={best_g}\n{best_g_mape*100:.4f}%",
             xy=(best_g, best_g_mape * 100),
             xytext=(best_g + 0.04, best_g_mape * 100 + 0.001),
             fontsize=9, color="#ffd600",
             arrowprops=dict(arrowstyle="->", color="#ffd600", lw=1))
ax1.set_xlabel("γ (коэффициент затухания)")
ax1.set_ylabel("MAPE валидации, %")
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
plt.tight_layout()
out1 = OUT_DIR / "52_gamma_sweep.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig1)
print(f"\n  Рис 1 → {out1}")

# ── рис 2: тепловая карта сетки γ_C1 × γ_C2 ──────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(8, 7))
fig2.suptitle(
    f"Тепловая карта MAPE×100: γ_C1 × γ_C2\n"
    f"(меньше = лучше; ★ = оптимум)",
    fontsize=12, fontweight="bold",
)
im = ax2.imshow(grid_mapes * 100, aspect="auto",
                cmap="RdYlGn_r",
                vmin=grid_mapes.min() * 100 - 0.005,
                vmax=mape_std * 100)
ax2.set_xticks(range(len(GRID_G))); ax2.set_xticklabels([str(g) for g in GRID_G])
ax2.set_yticks(range(len(GRID_G))); ax2.set_yticklabels([str(g) for g in GRID_G])
ax2.set_xlabel("γ_C2"); ax2.set_ylabel("γ_C1")
for i in range(len(GRID_G)):
    for j in range(len(GRID_G)):
        val = grid_mapes[i, j] * 100
        best_mark = "★" if (GRID_G[i] == g1_best and GRID_G[j] == g2_best) else ""
        ax2.text(j, i, f"{val:.3f}{best_mark}", ha="center", va="center",
                 fontsize=8, color="white" if val > mape_std * 100 * 0.985 else "black",
                 fontweight="bold" if best_mark else "normal")
plt.colorbar(im, ax=ax2, label="MAPE × 100")
plt.tight_layout()
out2 = OUT_DIR / "52_gamma_grid.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig2)
print(f"  Рис 2 → {out2}")

# ── рис 3: decay-конверты для лучших γ (объяснение механизма) ────────────────

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
fig3.suptitle(
    "Decay-конверты AR(BIC): как γ обнуляет прогноз к горизонту\n"
    f"(пример: среднее AR-предсказание C1 и C2 по {N_total} origins)",
    fontsize=12, fontweight="bold",
)
h_axis = np.arange(1, VAL_H + 1)
gammas_show = [0.3, 0.5, best_g, 0.7, 0.9, 1.0]
colors_show = ["#b71c1c", "#e65100", "#ffd600", "#42a5f5", "#9e9e9e", "#607d8b"]

for ax, ci, ci_label in zip(axes3, [1, 2], ["C1  8–16б", "C2  16–32б"]):
    mean_raw = np.mean(np.abs(ar_raw[ci]), axis=0)  # среднее |AR| по origins
    for g, col in zip(gammas_show, colors_show):
        decay = g ** h_axis
        lw = 2.5 if g == best_g else 1.5
        ls = "-" if g == best_g else "--"
        lbl = f"γ={g}" + (" ★" if g == best_g else "")
        ax.plot(h_axis, mean_raw * decay, color=col, lw=lw, ls=ls, label=lbl)
    ax.plot(h_axis, np.mean(np.abs(
        actual_ratios[:, :VAL_H] - (ratios_0[:, None] + np.cumsum(slow_hats, axis=1))
    ), axis=0), color="#00e5ff", lw=2, ls=":", label="actual MAE (ratio)")
    ax.set_title(ci_label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Горизонт h"); ax.set_ylabel("|AR forecast|")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
out3 = OUT_DIR / "52_decay_envelopes.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight"); plt.close(fig3)
print(f"  Рис 3 → {out3}")

print("\nГотово.")
