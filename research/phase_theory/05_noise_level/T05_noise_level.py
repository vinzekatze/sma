"""
T05 — Оценка уровня шума σ_noise через корреляционный интеграл.

Сигнал: ratio = close / logtrend (causal OLS).
Корреляционный интеграл C(m,r) при m ∈ {3,6,9,12,15}.
σ_noise: r* где slope(r) при m=9 устойчиво < m−0.5 (3 подряд точки).
r_k:     медиана расстояния до k-го соседа (k∈{10,20,30,50}) при m=9, Theiler=9.
Сравнение r_k30 / σ_noise — ключевое соотношение.
"""

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

ROOT    = Path(__file__).resolve().parents[3]
DATADIR = ROOT / "data" / "candles"
OUTDIR  = Path(__file__).resolve().parent
RESDIR  = OUTDIR / "results"
FIGDIR  = OUTDIR / "figures"
RESDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

TICKERS  = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL = "1d"

M_LIST   = [3, 6, 9, 12, 15]
M_WORK   = 9           # рабочий m для LP
K_LIST   = [10, 20, 30, 50]
K_WORK   = 30          # рабочий k для LP

N_REF    = 500         # опорных точек для C(m,r)
N_SAMPLE = 200         # точек для оценки r_grid
N_BINS   = 60          # бинов r (log-равномерно)
SEED     = 42
SMOOTH_W = 7           # окно сглаживания slope
N_CONSEC = 3           # подряд точек ниже порога → σ_noise
THRESH_D = 0.5         # порог: m − THRESH_D


# ── Данные ────────────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n   = len(close)
    lc  = np.log(np.maximum(close, 1e-10))
    t   = np.arange(n, dtype=np.float64)
    cn  = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    b     = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a     = (cy - b * ct) / cn
    trend = np.exp(a + b * t)
    trend[:2] = close[:2]
    return trend


def load_ratio(ticker: str) -> np.ndarray:
    raw   = json.loads((DATADIR / ticker / f"{INTERVAL}.json").read_text())
    cands = raw["candles"] if isinstance(raw, dict) else raw
    close = np.array([c["close"] for c in cands], dtype=np.float64)
    return close / logtrend_causal(close)


# ── Вложение ──────────────────────────────────────────────────────────────────

def embed(x: np.ndarray, m: int) -> np.ndarray:
    """Матрица задержек τ=1. Форма: (N−m+1, m)."""
    n_pts = len(x) - m + 1
    rows  = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
    return x[rows]


# ── Адаптивный r_grid ─────────────────────────────────────────────────────────

def compute_r_grid(X: np.ndarray, n_sample: int = N_SAMPLE,
                   theiler: int = None, n_bins: int = N_BINS,
                   seed: int = SEED) -> np.ndarray | None:
    n_pts = len(X)
    if theiler is None:
        theiler = X.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_pts, size=min(n_sample, n_pts), replace=False)

    D    = cdist(X[idx], X[idx])                        # (S, S)
    ri   = idx[:, None]; ci = idx[None, :]
    mask = (np.abs(ri - ci) > theiler) & (D > 0)
    d_v  = D[mask]

    if len(d_v) < 10:
        return None
    r_min = np.percentile(d_v, 0.5)
    r_max = np.percentile(d_v, 99.5)
    if r_min <= 0 or r_min >= r_max:
        r_min = d_v.min() * 1.01
    return np.geomspace(max(r_min, 1e-12), r_max, n_bins)


# ── Корреляционный интеграл ───────────────────────────────────────────────────

def corr_integral(X: np.ndarray, r_grid: np.ndarray,
                  n_ref: int = N_REF, theiler: int = None,
                  seed: int = SEED) -> np.ndarray:
    """C(m, r) через N_REF случайных опорных точек с Theiler-окном."""
    n_pts = len(X)
    if theiler is None:
        theiler = X.shape[1]
    rng     = np.random.default_rng(seed)
    ref_idx = rng.choice(n_pts, size=min(n_ref, n_pts), replace=False)

    all_dists = cdist(X[ref_idx], X)                          # (n_ref, n_pts)
    theiler_ok = np.abs(ref_idx[:, None] - np.arange(n_pts)) > theiler
    flat  = all_dists[theiler_ok]
    flat  = flat[flat > 0]
    flat.sort()

    if len(flat) == 0:
        return np.full(len(r_grid), np.nan)

    return np.searchsorted(flat, r_grid, side="right") / len(flat)


# ── Локальный наклон и σ_noise ────────────────────────────────────────────────

def local_slope(log_r: np.ndarray, log_C: np.ndarray,
                window: int = SMOOTH_W) -> np.ndarray:
    slope  = np.gradient(log_C, log_r)
    kernel = np.ones(window) / window
    return np.convolve(slope, kernel, mode="same")


def estimate_sigma_noise(r_grid: np.ndarray, slope: np.ndarray,
                         m: int, thresh_d: float = THRESH_D,
                         n_consec: int = N_CONSEC) -> float:
    """
    Первый r, где slope устойчиво ниже порога (m − thresh_d).
    Поиск начинается после краевого артефакта сглаживания.
    """
    threshold = m - thresh_d
    below     = slope < threshold
    start     = SMOOTH_W // 2 + 1           # пропуск краевых артефактов
    for i in range(start, len(below) - n_consec + 1):
        if np.all(below[i: i + n_consec]):
            return float(r_grid[i])
    return np.nan


# ── Расстояние до k-го соседа ─────────────────────────────────────────────────

def compute_r_k(X: np.ndarray, k_list: list,
                theiler: int = None) -> dict[int, float]:
    """Медианное расстояние до k-го соседа с Theiler-фильтром."""
    n_pts = len(X)
    if theiler is None:
        theiler = X.shape[1]
    k_max   = max(k_list)
    k_query = min(k_max + theiler + 5, n_pts - 1)

    tree        = KDTree(X)
    dists, inds = tree.query(X, k=k_query)       # (n_pts, k_query)

    results: dict[int, float] = {}
    for k in k_list:
        r_k_vals = []
        for i in range(n_pts):
            valid = np.abs(inds[i] - i) > theiler
            d_v   = dists[i, valid]
            if len(d_v) >= k:
                r_k_vals.append(d_v[k - 1])
        results[k] = float(np.median(r_k_vals)) if r_k_vals else np.nan
    return results


# ── Основной цикл ─────────────────────────────────────────────────────────────

rows_csv    = []
ticker_data = {}     # ticker → {m: {r_grid, C, slope}, sigma, rk}

print("Тикеры:", " ".join(TICKERS))
t0 = time.time()

for ticker in TICKERS:
    print(f"\n── {ticker} ──────────────────────────────")
    ratio = load_ratio(ticker)
    print(f"  N={len(ratio)}", flush=True)
    td = {}

    for m in M_LIST:
        X       = embed(ratio, m)
        r_grid  = compute_r_grid(X, theiler=m)
        if r_grid is None:
            print(f"  m={m}: r_grid не построен (мало данных)")
            continue
        C = corr_integral(X, r_grid, theiler=m)
        C = np.maximum(C, 1e-12)

        log_r = np.log(r_grid)
        log_C = np.log(C)
        slope = local_slope(log_r, log_C)

        td[m] = dict(r_grid=r_grid, C=C, slope=slope)
        print(f"  m={m:2d}  C=[{C.min():.4f},{C.max():.4f}]  "
              f"slope=[{slope.min():.2f},{slope.max():.2f}]", flush=True)

    # σ_noise при рабочем m
    sigma_noise = np.nan
    if M_WORK in td:
        sigma_noise = estimate_sigma_noise(
            td[M_WORK]["r_grid"], td[M_WORK]["slope"], m=M_WORK)

    if np.isnan(sigma_noise):
        print(f"  σ_noise (m={M_WORK}): не определён (slope не пересекает порог)")
    else:
        print(f"  σ_noise (m={M_WORK}): {sigma_noise:.6f}")

    # r_k при рабочем m
    X_work = embed(ratio, M_WORK)
    r_k    = compute_r_k(X_work, K_LIST, theiler=M_WORK)
    for k, v in r_k.items():
        print(f"  r_k{k:2d}: {v:.6f}")

    ratio_rk30_sigma = np.nan
    if not np.isnan(sigma_noise) and not np.isnan(r_k.get(K_WORK, np.nan)):
        ratio_rk30_sigma = r_k[K_WORK] / sigma_noise
    print(f"  r_k{K_WORK}/σ_noise = "
          + (f"{ratio_rk30_sigma:.3f}" if not np.isnan(ratio_rk30_sigma) else "—"))

    td["sigma"] = sigma_noise
    td["rk"]    = r_k
    ticker_data[ticker] = td

    rows_csv.append(dict(
        ticker           = ticker,
        N                = len(ratio),
        sigma_noise      = sigma_noise,
        **{f"r_k{k}": r_k.get(k, np.nan) for k in K_LIST},
        ratio_rk30_sigma = ratio_rk30_sigma,
    ))
    print(f"  [{ticker}] {time.time()-t0:.1f}с", flush=True)

print(f"\nЦикл завершён за {time.time()-t0:.1f}с")

# ── CSV ───────────────────────────────────────────────────────────────────────

csv_path = RESDIR / "noise_level.csv"
if rows_csv:
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows_csv[0].keys())
        wr.writeheader()
        wr.writerows(rows_csv)
    print(f"\nCSV: {csv_path}")

# ── Сводная таблица ───────────────────────────────────────────────────────────

print(f"\n{'Тикер':<6} {'σ_noise':>10} {'r_k10':>8} {'r_k20':>8} "
      f"{'r_k30':>8} {'r_k50':>8} {'rk30/σ':>8}")
print("─" * 62)
for row in rows_csv:
    def f(v): return f"{v:.5f}" if not (isinstance(v, float) and np.isnan(v)) else "  —"
    print(f"  {row['ticker']:<5} {f(row['sigma_noise']):>10} {f(row['r_k10']):>8} "
          f"{f(row['r_k20']):>8} {f(row['r_k30']):>8} {f(row['r_k50']):>8} "
          f"{f(row['ratio_rk30_sigma']):>8}")


# ── График A: log C(m,r) — per тикер ─────────────────────────────────────────

cols_g = 4
rows_g = (len(TICKERS) + cols_g - 1) // cols_g
fig_a, axes_a = plt.subplots(rows_g, cols_g, figsize=(16, 4 * rows_g))
fig_a.suptitle("T05: Корреляционный интеграл log C(m,r) — ratio  [пунктир = slope×m]",
               fontsize=11)
axes_a = axes_a.flatten()

colors_m = {3: "#1f77b4", 6: "#ff7f0e", 9: "#2ca02c", 12: "#d62728", 15: "#9467bd"}

for ax_i, ticker in enumerate(TICKERS):
    ax = axes_a[ax_i]
    td = ticker_data.get(ticker, {})

    for m in M_LIST:
        if m not in td:
            continue
        r_grid = td[m]["r_grid"]
        C      = td[m]["C"]
        log_r  = np.log(r_grid)
        log_C  = np.log(C)
        col    = colors_m[m]
        ax.plot(log_r, log_C, color=col, lw=1.4, label=f"m={m}")

        # Теоретическая линия slope=m от первой четверти диапазона
        i0 = len(log_r) // 5
        ax.plot(log_r[:i0 * 2],
                log_C[i0] + m * (log_r[:i0 * 2] - log_r[i0]),
                color=col, lw=0.7, ls="--", alpha=0.45)

    sigma = td.get("sigma", np.nan)
    if not np.isnan(sigma):
        ax.axvline(np.log(sigma), color="red", lw=1.2, ls=":",
                   label=f"σ≈{sigma:.4f}")

    ax.set_title(ticker, fontsize=9)
    ax.set_xlabel("log r", fontsize=7)
    ax.set_ylabel("log C", fontsize=7)
    ax.tick_params(labelsize=7)
    if ax_i == 0:
        ax.legend(fontsize=6, ncol=2)

for ax_i in range(len(TICKERS), len(axes_a)):
    axes_a[ax_i].set_visible(False)

fig_a.tight_layout()
fig_a.savefig(FIGDIR / "T05_corr_integral.png", dpi=120)
plt.close(fig_a)
print("Рис. A: T05_corr_integral.png")


# ── График B: slope(r) при m=9 — все тикеры ──────────────────────────────────

fig_b, ax_b = plt.subplots(figsize=(12, 5))
ax_b.set_title(f"T05: Локальный наклон slope(r) при m={M_WORK} — все тикеры", fontsize=11)
threshold = M_WORK - THRESH_D
ax_b.axhline(threshold, color="red", lw=1.3, ls="--",
             label=f"порог m−{THRESH_D} = {threshold}")
ax_b.axhline(M_WORK, color="grey", lw=0.8, ls=":", alpha=0.5,
             label=f"m={M_WORK} (шум-режим)")

cmap = plt.cm.tab10
for t_i, ticker in enumerate(TICKERS):
    td  = ticker_data.get(ticker, {})
    if M_WORK not in td:
        continue
    col    = cmap(t_i / max(len(TICKERS) - 1, 1))
    log_r  = np.log(td[M_WORK]["r_grid"])
    slope  = td[M_WORK]["slope"]
    ax_b.plot(log_r, slope, color=col, lw=1.4, label=ticker)
    sigma  = td.get("sigma", np.nan)
    if not np.isnan(sigma):
        ax_b.axvline(np.log(sigma), color=col, lw=0.8, ls=":", alpha=0.6)

ax_b.set_xlabel("log r")
ax_b.set_ylabel("slope = d(log C)/d(log r)")
ax_b.set_ylim(-1, M_WORK + 3)
ax_b.legend(fontsize=8, ncol=3)
fig_b.tight_layout()
fig_b.savefig(FIGDIR / "T05_slope_m9.png", dpi=120)
plt.close(fig_b)
print("Рис. B: T05_slope_m9.png")


# ── График C: r_k vs σ_noise scatter ─────────────────────────────────────────

fig_c, ax_c = plt.subplots(figsize=(8, 7))
ax_c.set_title(f"T05: r_k vs σ_noise  (m={M_WORK}) — per тикер", fontsize=11)

sigma_vals = [ticker_data[t].get("sigma", np.nan) for t in TICKERS]
colors_k   = {10: "#1f77b4", 20: "#ff7f0e", 30: "#2ca02c", 50: "#d62728"}

for k in K_LIST:
    points = [(sigma_vals[i], ticker_data[t]["rk"].get(k, np.nan))
              for i, t in enumerate(TICKERS)
              if "rk" in ticker_data.get(t, {}) and not np.isnan(sigma_vals[i])]
    if not points:
        continue
    sx, ry = zip(*points)
    ax_c.scatter(sx, ry, color=colors_k[k], s=90, zorder=3, label=f"k={k}")

# Диагональ и зелёная зона
s_valid = [s for s in sigma_vals if not np.isnan(s)]
if s_valid:
    lo, hi = min(s_valid) * 0.7, max(s_valid) * 1.5
    diag   = np.array([lo, hi])
    ax_c.plot(diag, diag, "k--", lw=0.9, label="r_k = σ_noise")
    ax_c.fill_between(diag, diag, diag * 10, alpha=0.07, color="green")
    ax_c.text((lo + hi) * 0.5, (lo + hi) * 0.5 * 3,
              "r_k > σ_noise\n(LP видит аттрактор)", fontsize=7,
              color="green", alpha=0.7)

# Подписи тикеров при k=K_WORK
for i, ticker in enumerate(TICKERS):
    s = sigma_vals[i]
    if np.isnan(s):
        continue
    rk30 = ticker_data[ticker].get("rk", {}).get(K_WORK, np.nan)
    if not np.isnan(rk30):
        ax_c.annotate(ticker, (s, rk30), fontsize=7,
                      xytext=(4, 3), textcoords="offset points")

ax_c.set_xlabel("σ_noise")
ax_c.set_ylabel("r_k (медиана расстояния до k-го соседа)")
ax_c.legend(fontsize=8)
fig_c.tight_layout()
fig_c.savefig(FIGDIR / "T05_rk_vs_sigma.png", dpi=120)
plt.close(fig_c)
print("Рис. C: T05_rk_vs_sigma.png")

print("\nГотово.")
