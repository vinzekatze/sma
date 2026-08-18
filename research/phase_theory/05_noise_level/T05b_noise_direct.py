"""
T05b — Прямая оценка σ_noise через residual LP: std/MAD(ratio − att).

Два конфига LP-фильтра:
  OLD: d=3,  m=9,  k=30,  n_iter=3  (текущий стандарт)
  NEW: d=11, m=33, k=110, n_iter=2  (лучшая точка эксп.101)

Для каждого конфига:
  att  = local_projective(ratio, m, d, k, n_iter)
  noise = ratio − att
  σ_std = std(noise),  σ_mad = MAD(noise) × 1.4826
  r_k   = медиана расстояния до k-го соседа в m-мерном вложении, k∈{30,110}

Ключевое соотношение: r_k / σ_noise — при рабочем k каждого конфига.
Диагностика noise: гистограммы, ACF, корреляционный интеграл.
"""

import csv
import json
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

CONFIGS = {
    "OLD": dict(d=3,  m=9,  k=30,  n_iter=3, k_work=30),
    "NEW": dict(d=11, m=33, k=110, n_iter=2, k_work=110),
}
K_LIST   = [30, 110]
ACF_LAGS = 40
N_REF    = 500    # для корреляционного интеграла на noise
N_BINS   = 60
SEED     = 42
SMOOTH_W = 7
M_CI     = [3, 6, 9, 12, 15]   # m для C(m,r) на noise


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


# ── Local Projective фильтр ───────────────────────────────────────────────────

def local_projective(series: np.ndarray, m: int, d_proj: int,
                     k: int, n_iter: int = 1) -> np.ndarray:
    s     = series.copy().astype(np.float64)
    N     = len(s)
    k_eff = min(k, N - m)

    for _ in range(n_iter):
        n_pts = N - m + 1
        if n_pts < k_eff + 1:
            break
        rows  = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
        X     = s[rows]
        tree  = KDTree(X)
        _, inds = tree.query(X, k=k_eff + 1)

        X_proj = np.empty_like(X)
        for i in range(n_pts):
            nn       = inds[i, 1:]
            X_nn     = X[nn]
            centroid = X_nn.mean(axis=0)
            centered = X_nn - centroid
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            V_d      = Vt[:d_proj].T
            xc       = X[i] - centroid
            X_proj[i] = centroid + V_d @ (V_d.T @ xc)

        result = np.zeros(N); count = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            result[i:i + m] += X_proj[i]; count[i:i + m] += 1
        s = result / np.maximum(count, 1)

    return s


# ── Вспомогательные функции ───────────────────────────────────────────────────

def embed(x: np.ndarray, m: int) -> np.ndarray:
    n_pts = len(x) - m + 1
    rows  = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
    return x[rows]


def compute_r_k(X: np.ndarray, k_list: list,
                theiler: int = None) -> dict[int, float]:
    n_pts = len(X)
    if theiler is None:
        theiler = X.shape[1]
    k_max   = max(k_list)
    k_query = min(k_max + theiler + 5, n_pts - 1)
    tree        = KDTree(X)
    dists, inds = tree.query(X, k=k_query)
    results: dict[int, float] = {}
    for k in k_list:
        vals = []
        for i in range(n_pts):
            valid = np.abs(inds[i] - i) > theiler
            d_v   = dists[i, valid]
            if len(d_v) >= k:
                vals.append(d_v[k - 1])
        results[k] = float(np.median(vals)) if vals else np.nan
    return results


def compute_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x    = x - x.mean()
    var  = np.var(x)
    if var < 1e-14:
        return np.zeros(max_lag + 1)
    acf = np.array([
        np.mean(x[:len(x) - lag] * x[lag:]) / var
        for lag in range(max_lag + 1)
    ])
    return acf


def compute_r_grid(X: np.ndarray, n_sample: int = 200,
                   theiler: int = None, n_bins: int = N_BINS,
                   seed: int = SEED):
    n_pts = len(X)
    if theiler is None:
        theiler = X.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_pts, size=min(n_sample, n_pts), replace=False)
    D   = cdist(X[idx], X[idx])
    ri  = idx[:, None]; ci = idx[None, :]
    mask = (np.abs(ri - ci) > theiler) & (D > 0)
    d_v  = D[mask]
    if len(d_v) < 10:
        return None
    r_min = np.percentile(d_v, 0.5)
    r_max = np.percentile(d_v, 99.5)
    if r_min <= 0 or r_min >= r_max:
        r_min = d_v.min() * 1.01
    return np.geomspace(max(r_min, 1e-12), r_max, n_bins)


def corr_integral(X: np.ndarray, r_grid: np.ndarray,
                  n_ref: int = N_REF, theiler: int = None,
                  seed: int = SEED) -> np.ndarray:
    n_pts = len(X)
    if theiler is None:
        theiler = X.shape[1]
    rng     = np.random.default_rng(seed)
    ref_idx = rng.choice(n_pts, size=min(n_ref, n_pts), replace=False)
    all_d   = cdist(X[ref_idx], X)
    ok      = np.abs(ref_idx[:, None] - np.arange(n_pts)) > theiler
    flat    = all_d[ok]
    flat    = flat[flat > 0]
    flat.sort()
    if len(flat) == 0:
        return np.full(len(r_grid), np.nan)
    return np.searchsorted(flat, r_grid, side="right") / len(flat)


def local_slope(log_r, log_C, window=SMOOTH_W):
    slope  = np.gradient(log_C, log_r)
    kernel = np.ones(window) / window
    return np.convolve(slope, kernel, mode="same")


# ── Основной цикл ─────────────────────────────────────────────────────────────

rows_csv    = []
ticker_data = {}

print("Тикеры:", " ".join(TICKERS))
t0 = time.time()

for ticker in TICKERS:
    print(f"\n── {ticker} ──────────────────────────────")
    ratio = load_ratio(ticker)
    print(f"  N={len(ratio)}", flush=True)
    td = {"ratio": ratio}

    for cfg_name, cfg in CONFIGS.items():
        m, d, k, n_iter = cfg["m"], cfg["d"], cfg["k"], cfg["n_iter"]
        print(f"  LP {cfg_name} (m={m} d={d} k={k} n={n_iter})...", flush=True)

        t1 = time.time()
        att   = local_projective(ratio, m=m, d_proj=d, k=k, n_iter=n_iter)
        noise = ratio - att
        print(f"    LP: {time.time()-t1:.1f}с")

        sigma_std = float(np.std(noise))
        sigma_mad = float(np.median(np.abs(noise - np.median(noise))) * 1.4826)
        print(f"    σ_std={sigma_std:.6f}  σ_mad={sigma_mad:.6f}")

        # r_k при рабочем m конфига
        X_m = embed(ratio, m)
        r_k = compute_r_k(X_m, K_LIST, theiler=m)
        for k_val, r in r_k.items():
            ratio_val = r / sigma_std if sigma_std > 0 else np.nan
            print(f"    r_k{k_val:3d}={r:.6f}  r_k/σ_std={ratio_val:.3f}")

        # ACF noise
        acf = compute_acf(noise, ACF_LAGS)

        # Корреляционный интеграл на noise
        ci_data = {}
        for m_ci in M_CI:
            if m_ci >= len(noise) - 1:
                continue
            X_ci   = embed(noise, m_ci)
            r_grid = compute_r_grid(X_ci, theiler=m_ci)
            if r_grid is None:
                continue
            C     = np.maximum(corr_integral(X_ci, r_grid, theiler=m_ci), 1e-12)
            slope = local_slope(np.log(r_grid), np.log(C))
            ci_data[m_ci] = dict(r_grid=r_grid, C=C, slope=slope)

        td[cfg_name] = dict(
            att=att, noise=noise,
            sigma_std=sigma_std, sigma_mad=sigma_mad,
            r_k=r_k, acf=acf, ci=ci_data
        )

        rows_csv.append(dict(
            ticker=ticker, config=cfg_name, N=len(ratio),
            m=m, d=d, k=k, n_iter=n_iter,
            sigma_std=sigma_std, sigma_mad=sigma_mad,
            **{f"r_k{kv}": r_k.get(kv, np.nan) for kv in K_LIST},
            **{f"ratio_k{kv}_std": (r_k.get(kv, np.nan) / sigma_std
                                    if sigma_std > 0 else np.nan)
               for kv in K_LIST},
        ))

    ticker_data[ticker] = td
    print(f"  [{ticker}] {time.time()-t0:.1f}с total", flush=True)

print(f"\nЦикл: {time.time()-t0:.1f}с")

# ── CSV ───────────────────────────────────────────────────────────────────────

csv_path = RESDIR / "noise_direct.csv"
with open(csv_path, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=rows_csv[0].keys())
    wr.writeheader(); wr.writerows(rows_csv)
print(f"CSV: {csv_path}")

# ── Сводная таблица ───────────────────────────────────────────────────────────

def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "  —"

print(f"\n{'Тикер':<6} {'cfg':<4} {'σ_std':>8} {'σ_mad':>8} "
      f"{'r_k30':>8} {'r_k110':>8} {'k30/σ':>7} {'k110/σ':>7}")
print("─" * 64)
for row in rows_csv:
    print(f"  {row['ticker']:<5} {row['config']:<4} {fmt(row['sigma_std']):>8} "
          f"{fmt(row['sigma_mad']):>8} {fmt(row['r_k30']):>8} {fmt(row['r_k110']):>8} "
          f"{fmt(row['ratio_k30_std']):>7} {fmt(row['ratio_k110_std']):>7}")

# ── График A: noise distribution — гистограммы ───────────────────────────────

fig_a, axes_a = plt.subplots(2, len(TICKERS), figsize=(20, 6))
fig_a.suptitle("T05b: Распределение noise = ratio − att", fontsize=11)

for t_i, ticker in enumerate(TICKERS):
    td = ticker_data[ticker]
    for r_i, cfg_name in enumerate(["OLD", "NEW"]):
        ax    = axes_a[r_i, t_i]
        noise = td[cfg_name]["noise"]
        sig   = td[cfg_name]["sigma_std"]
        ax.hist(noise, bins=60, density=True, color="#4c72b0" if cfg_name == "OLD" else "#dd8452",
                alpha=0.7, edgecolor="none")
        x = np.linspace(noise.min(), noise.max(), 200)
        ax.plot(x, np.exp(-0.5 * (x / sig) ** 2) / (sig * np.sqrt(2 * np.pi)),
                "k--", lw=0.9, alpha=0.6)
        ax.set_title(f"{ticker}\n{cfg_name} σ={sig:.4f}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("noise", fontsize=6)

fig_a.tight_layout()
fig_a.savefig(FIGDIR / "T05b_noise_dist.png", dpi=110)
plt.close(fig_a)
print("Рис. A: T05b_noise_dist.png")

# ── График B: ACF noise — все тикеры, OLD vs NEW ──────────────────────────────

lags = np.arange(ACF_LAGS + 1)
fig_b, axes_b = plt.subplots(1, 2, figsize=(14, 5))
fig_b.suptitle("T05b: ACF(noise) — среднее по тикерам", fontsize=11)
conf95 = 1.96 / np.sqrt(5000)

for ax, cfg_name, col in zip(axes_b, ["OLD", "NEW"], ["#4c72b0", "#dd8452"]):
    cfg = CONFIGS[cfg_name]
    acfs = np.array([ticker_data[t][cfg_name]["acf"] for t in TICKERS])
    mean_acf = acfs.mean(axis=0)
    std_acf  = acfs.std(axis=0)
    ax.bar(lags, mean_acf, color=col, alpha=0.6, label="среднее ACF")
    ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf,
                    color=col, alpha=0.2)
    ax.axhline(conf95, color="red", ls="--", lw=0.8, label=f"±1.96/√N")
    ax.axhline(-conf95, color="red", ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_title(f"{cfg_name}: d={cfg['d']} m={cfg['m']} k={cfg['k']} n={cfg['n_iter']}",
                 fontsize=9)
    ax.set_xlabel("lag"); ax.set_ylabel("ACF")
    ax.legend(fontsize=7)
    ax.set_xlim(0, ACF_LAGS)
    ax.set_ylim(-0.25, 1.05)

fig_b.tight_layout()
fig_b.savefig(FIGDIR / "T05b_acf.png", dpi=110)
plt.close(fig_b)
print("Рис. B: T05b_acf.png")

# ── График C: r_k / σ_noise сравнение — OLD vs NEW, k=30 vs k=110 ────────────

fig_c, axes_c = plt.subplots(1, 2, figsize=(14, 6))
fig_c.suptitle("T05b: r_k / σ_std  по тикерам — OLD (d=3) vs NEW (d=11)", fontsize=11)

x_pos  = np.arange(len(TICKERS))
colors = {30: "#1f77b4", 110: "#d62728"}
width  = 0.35

for ax, cfg_name in zip(axes_c, ["OLD", "NEW"]):
    cfg = CONFIGS[cfg_name]
    for ki, k_val in enumerate(K_LIST):
        ratios = []
        for ticker in TICKERS:
            r_k   = ticker_data[ticker][cfg_name]["r_k"].get(k_val, np.nan)
            sigma = ticker_data[ticker][cfg_name]["sigma_std"]
            ratios.append(r_k / sigma if sigma > 0 else np.nan)
        offset = (ki - 0.5) * width
        bars = ax.bar(x_pos + offset, ratios, width=width,
                      color=colors[k_val], alpha=0.8, label=f"k={k_val}")

    ax.axhline(1.0, color="k", ls="--", lw=1.2, label="r_k = σ_noise")
    ax.fill_between([-0.5, len(TICKERS) - 0.5], [1, 1], [ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 5],
                    alpha=0.05, color="green")
    ax.set_xticks(x_pos); ax.set_xticklabels(TICKERS, fontsize=8)
    ax.set_title(f"{cfg_name}: d={cfg['d']} m={cfg['m']} k={cfg['k']} n={cfg['n_iter']}",
                 fontsize=9)
    ax.set_ylabel("r_k / σ_std")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", lw=0.3)

fig_c.tight_layout()
fig_c.savefig(FIGDIR / "T05b_rk_vs_sigma.png", dpi=110)
plt.close(fig_c)
print("Рис. C: T05b_rk_vs_sigma.png")

# ── График D: корреляционный интеграл на noise — OLD vs NEW ──────────────────

fig_d, axes_d = plt.subplots(2, 4, figsize=(18, 8))
fig_d.suptitle("T05b: C(m,r) на noise (slope≈m = шум-режим)", fontsize=11)
colors_m = {3: "#1f77b4", 6: "#ff7f0e", 9: "#2ca02c", 12: "#d62728", 15: "#9467bd"}

for t_i, ticker in enumerate(TICKERS):
    ax  = axes_d[t_i // 4, t_i % 4]
    td  = ticker_data[ticker]

    for cfg_name, ls, lw in [("OLD", "-", 1.6), ("NEW", "--", 1.2)]:
        ci = td[cfg_name]["ci"]
        for m_ci in M_CI:
            if m_ci not in ci:
                continue
            r_g   = ci[m_ci]["r_grid"]
            C     = ci[m_ci]["C"]
            slope = ci[m_ci]["slope"]
            col   = colors_m.get(m_ci, "grey")
            label = f"m={m_ci} {cfg_name}" if t_i == 0 else None
            ax.plot(np.log(r_g), np.log(C), color=col, ls=ls, lw=lw,
                    label=label, alpha=0.85)

    ax.set_title(ticker, fontsize=9)
    ax.set_xlabel("log r", fontsize=7); ax.set_ylabel("log C", fontsize=7)
    ax.tick_params(labelsize=6)

axes_d[0, 0].legend(fontsize=5, ncol=2)
# Пояснение: сплошная=OLD, пунктир=NEW
fig_d.text(0.5, 0.01, "Сплошная = OLD (d=3,m=9),  пунктир = NEW (d=11,m=33)",
           ha="center", fontsize=9)
fig_d.tight_layout(rect=[0, 0.03, 1, 1])
fig_d.savefig(FIGDIR / "T05b_corr_integral_noise.png", dpi=110)
plt.close(fig_d)
print("Рис. D: T05b_corr_integral_noise.png")

print("\nГотово.")
