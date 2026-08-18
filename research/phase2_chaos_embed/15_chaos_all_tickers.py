"""
Проверка стохастичности Δratio vs ratio на всех доступных тикерах (1d).

Метрики:
  FNN — первое m где FNN < 5% (или "∞" если не достигает)
  D₂  — наклон корреляционного интеграла при m=2..6 (растёт → стохастик)

MA_WINDOW=1000 для всех тикеров (унифицировано).
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KDTree

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sma.core.forecast.normalize import normalize
from sma.core.forecast.embedding import build_delay_matrix

CANDLES_DIR = ROOT / "data/candles"
TICKERS     = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
MA_WINDOW   = 1000
TAU         = 1
M_MAX_FNN   = 10
M_MAX_D2    = 6
N_SAMPLE    = 1500   # последние бары для D₂
R_TOL       = 10.0   # FNN критерий 1
A_TOL       = 2.0    # FNN критерий 2


# ── FNN ──────────────────────────────────────────────────────────────────────

def fnn_curve(series: np.ndarray, m_max: int = M_MAX_FNN,
              r_tol: float = R_TOL, a_tol: float = A_TOL) -> dict[int, float]:
    sigma = np.std(series)
    results = {}
    for m in range(1, m_max + 1):
        n_m1 = len(series) - m * TAU - 1
        if n_m1 < 10:
            break
        n_m = len(series) - (m - 1) * TAU - 1

        X_m = np.column_stack([series[k: k + n_m] for k in range(m)])
        tree = KDTree(X_m[:n_m1])
        dists, idxs = tree.query(X_m[:n_m1], k=2)

        R_m  = dists[:, 1]
        j    = idxs[:, 1]

        extra_i = series[m: m + n_m1]
        extra_j = series[m: m + n_m1][j]

        R_m1   = np.sqrt(R_m**2 + (extra_i - extra_j)**2)
        crit1  = np.abs(extra_i - extra_j) / (R_m + 1e-12) > r_tol
        crit2  = R_m1 / sigma > a_tol
        valid  = R_m > 1e-12

        frac   = np.sum((crit1 | crit2)[valid]) / np.sum(valid) if valid.any() else np.nan
        results[m] = frac
    return results


def first_below(fnn_dict: dict, threshold: float = 0.05) -> int | None:
    for m in sorted(fnn_dict):
        if fnn_dict[m] < threshold:
            return m
    return None


# ── D₂ ───────────────────────────────────────────────────────────────────────

def d2_at_m(series: np.ndarray, m: int, n_r: int = 25) -> float:
    try:
        X, _ = build_delay_matrix(series[-N_SAMPLE:], m)
    except ValueError:
        return np.nan

    N = len(X)
    if N < 30:
        return np.nan

    # попарные расстояния нижнего треугольника
    dists_list = []
    chunk = 150
    for i in range(0, N, chunk):
        Xi = X[i: i + chunk]
        d  = np.sqrt(((Xi[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
        for li in range(len(Xi)):
            gi = i + li
            dists_list.append(d[li, :gi])

    if not dists_list:
        return np.nan
    d_flat = np.concatenate(dists_list)
    d_flat = d_flat[d_flat > 0]
    if len(d_flat) < 10:
        return np.nan

    r_min  = np.percentile(d_flat, 2)
    r_max  = np.percentile(d_flat, 55)
    if r_min >= r_max:
        return np.nan

    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    C_vals = np.array([np.mean(d_flat < r) for r in r_vals])

    valid  = C_vals > 0
    if valid.sum() < 4:
        return np.nan
    log_r  = np.log10(r_vals[valid])
    log_C  = np.log10(C_vals[valid])
    lo     = np.percentile(log_r, 20)
    hi     = np.percentile(log_r, 70)
    mask   = (log_r >= lo) & (log_r <= hi)
    if mask.sum() < 3:
        return np.nan
    slope, _ = np.polyfit(log_r[mask], log_C[mask], 1)
    return float(slope)


# ── основной цикл ─────────────────────────────────────────────────────────────

results = {}

for ticker in TICKERS:
    path = CANDLES_DIR / ticker / "1d.json"
    if not path.exists():
        continue

    with open(path) as f:
        candles = json.load(f)

    df     = normalize(candles, window=MA_WINDOW)
    df     = df.dropna(subset=["ma"]).reset_index(drop=True)
    ratio  = df["ratio"].values
    dratio = np.diff(ratio)

    print(f"{ticker}: {len(df)} баров", end="  ")

    fnn_dr = fnn_curve(dratio)
    fnn_rt = fnn_curve(ratio)
    p_dr   = first_below(fnn_dr)
    p_rt   = first_below(fnn_rt)

    d2_dr = {m: d2_at_m(dratio, m) for m in range(2, M_MAX_D2 + 1)}
    d2_rt = {m: d2_at_m(ratio,  m) for m in range(2, M_MAX_D2 + 1)}

    results[ticker] = {
        "fnn_dr": fnn_dr, "fnn_rt": fnn_rt,
        "p_dr": p_dr,     "p_rt":   p_rt,
        "d2_dr": d2_dr,   "d2_rt":  d2_rt,
        "n": len(df),
    }
    print(f"FNN: dratio={'∞' if p_dr is None else p_dr}  ratio={p_rt}  "
          f"| D₂(m=4): dratio={d2_dr.get(4, np.nan):.2f}  ratio={d2_rt.get(4, np.nan):.2f}")


# ── сводная таблица ───────────────────────────────────────────────────────────
print(f"\n{'Тикер':<6}  {'FNN dratio':>10}  {'FNN ratio':>10}  "
      f"{'D₂(m=4) dr':>11}  {'D₂(m=6) dr':>11}  "
      f"{'D₂(m=4) rt':>11}  {'D₂(m=6) rt':>11}  {'Вывод':>18}")
print("─" * 100)

for ticker, r in results.items():
    p_dr = "∞" if r["p_dr"] is None else str(r["p_dr"])
    p_rt = "∞" if r["p_rt"] is None else str(r["p_rt"])
    d4dr = r["d2_dr"].get(4, np.nan)
    d6dr = r["d2_dr"].get(6, np.nan)
    d4rt = r["d2_rt"].get(4, np.nan)
    d6rt = r["d2_rt"].get(6, np.nan)

    dr_stoch = (r["p_dr"] is None) or (not np.isnan(d6dr) and d6dr > d4dr + 0.3)
    rt_det   = (r["p_rt"] is not None) and (not np.isnan(d6rt) and d6rt < d4rt + 0.3)
    verdict  = ("dr:стохаст " if dr_stoch else "dr:det? ") + \
               ("rt:det" if rt_det else "rt:стохаст?")

    print(f"{ticker:<6}  {p_dr:>10}  {p_rt:>10}  "
          f"{d4dr:>11.2f}  {d6dr:>11.2f}  "
          f"{d4rt:>11.2f}  {d6rt:>11.2f}  {verdict:>18}")


# ── графики ───────────────────────────────────────────────────────────────────
n_tickers = len(results)
fig, axes = plt.subplots(2, n_tickers, figsize=(n_tickers * 2.2, 8))
fig.suptitle(f"FNN (верх) и D₂ (низ) — все тикеры 1d  MA={MA_WINDOW}", fontsize=12)

for col, (ticker, r) in enumerate(results.items()):
    # FNN
    ax_fnn = axes[0][col]
    ms_dr  = sorted(r["fnn_dr"])
    ms_rt  = sorted(r["fnn_rt"])
    ax_fnn.plot(ms_dr, [r["fnn_dr"][m]*100 for m in ms_dr],
                "o-", color="steelblue", lw=1.5, ms=4, label="Δratio")
    ax_fnn.plot(ms_rt, [r["fnn_rt"][m]*100 for m in ms_rt],
                "s-", color="seagreen",  lw=1.5, ms=4, label="ratio")
    ax_fnn.axhline(5, color="red", ls="--", lw=1)
    ax_fnn.set_title(ticker, fontsize=10)
    ax_fnn.set_ylim(0, 105)
    ax_fnn.set_xlabel("m")
    if col == 0:
        ax_fnn.set_ylabel("FNN %")
        ax_fnn.legend(fontsize=7)
    ax_fnn.grid(alpha=0.2)

    # D₂
    ax_d2  = axes[1][col]
    ms     = list(range(2, M_MAX_D2 + 1))
    d2_dr  = [r["d2_dr"].get(m, np.nan) for m in ms]
    d2_rt  = [r["d2_rt"].get(m, np.nan) for m in ms]
    ax_d2.plot(ms, d2_dr, "o-", color="steelblue", lw=1.5, ms=4)
    ax_d2.plot(ms, d2_rt, "s-", color="seagreen",  lw=1.5, ms=4)
    ax_d2.plot(ms, ms,    "k--", lw=0.6, alpha=0.3)
    ax_d2.set_xlabel("m")
    if col == 0:
        ax_d2.set_ylabel("D₂")
    ax_d2.grid(alpha=0.2)
    ax_d2.set_ylim(bottom=0)

plt.tight_layout()
out = ROOT / "research/figures/15_chaos_all_tickers.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=130)
print(f"\nГрафик: {out}")
plt.show()
