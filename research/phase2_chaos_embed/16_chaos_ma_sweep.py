"""
Зависимость стохастичности от окна MA — SBER 1d.

Вопрос: p=4 для ratio и ∞ для dratio — это свойство MA=1000,
или универсально при любом окне нормализации?

Перебираем MA = [50, 100, 200, 500, 1000, 2000, 5000].
Для каждого: FNN-кривая ratio и dratio, D₂(m=2..6).
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sma.core.forecast.normalize import normalize
from sma.core.forecast.embedding import build_delay_matrix

DATA_FILE = ROOT / "data/candles/SBER/1d.json"

MA_WINDOWS = [50, 100, 200, 500, 1000, 2000, 5000]
M_MAX      = 10
N_SAMPLE   = 1500
R_TOL      = 10.0
A_TOL      = 2.0


# ── FNN ──────────────────────────────────────────────────────────────────────

def fnn_curve(series, m_max=M_MAX):
    sigma = np.std(series)
    out = {}
    for m in range(1, m_max + 1):
        n_m1 = len(series) - m - 1
        n_m  = len(series) - (m - 1) - 1
        if n_m1 < 10:
            break
        X_m  = np.column_stack([series[k: k + n_m] for k in range(m)])
        tree = KDTree(X_m[:n_m1])
        dists, idxs = tree.query(X_m[:n_m1], k=2)
        R_m     = dists[:, 1]
        j       = idxs[:, 1]
        extra_i = series[m: m + n_m1]
        extra_j = series[m: m + n_m1][j]
        R_m1    = np.sqrt(R_m**2 + (extra_i - extra_j)**2)
        crit1   = np.abs(extra_i - extra_j) / (R_m + 1e-12) > R_TOL
        crit2   = R_m1 / sigma > A_TOL
        valid   = R_m > 1e-12
        out[m]  = float(np.sum((crit1 | crit2)[valid]) / np.sum(valid)) if valid.any() else np.nan
    return out


def first_below(d, thr=0.05):
    for m in sorted(d):
        if d[m] < thr:
            return m
    return None


# ── D₂ ───────────────────────────────────────────────────────────────────────

def d2_at_m(series, m, n_r=20):
    try:
        X, _ = build_delay_matrix(series[-N_SAMPLE:], m)
    except ValueError:
        return np.nan
    N = len(X)
    if N < 30:
        return np.nan
    dists_list = []
    chunk = 150
    for i in range(0, N, chunk):
        Xi = X[i: i + chunk]
        d  = np.sqrt(((Xi[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
        for li in range(len(Xi)):
            gi = i + li
            dists_list.append(d[li, :gi])
    d_flat = np.concatenate(dists_list)
    d_flat = d_flat[d_flat > 0]
    if len(d_flat) < 10:
        return np.nan
    r_min = np.percentile(d_flat, 2)
    r_max = np.percentile(d_flat, 55)
    if r_min >= r_max:
        return np.nan
    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    C_vals = np.array([np.mean(d_flat < r) for r in r_vals])
    valid  = C_vals > 0
    if valid.sum() < 4:
        return np.nan
    log_r = np.log10(r_vals[valid])
    log_C = np.log10(C_vals[valid])
    lo = np.percentile(log_r, 20)
    hi = np.percentile(log_r, 70)
    mask = (log_r >= lo) & (log_r <= hi)
    if mask.sum() < 3:
        return np.nan
    return float(np.polyfit(log_r[mask], log_C[mask], 1)[0])


# ── данные и расчёт ──────────────────────────────────────────────────────────

with open(DATA_FILE) as f:
    candles = json.load(f)

print(f"SBER 1d  {len(candles)} свечей")
print(f"{'MA':>5}  {'bars':>5}  {'p_ratio':>8}  {'p_dratio':>9}  "
      f"{'D₂(4)_rt':>10}  {'D₂(6)_rt':>10}  {'D₂(4)_dr':>10}  {'D₂(6)_dr':>10}")
print("─" * 85)

res = {}

for ma in MA_WINDOWS:
    df     = normalize(candles, window=ma)
    df     = df.dropna(subset=["ma"]).reset_index(drop=True)
    ratio  = df["ratio"].values
    dratio = np.diff(ratio)

    fnn_rt = fnn_curve(ratio)
    fnn_dr = fnn_curve(dratio)
    p_rt   = first_below(fnn_rt)
    p_dr   = first_below(fnn_dr)

    d2_rt = {m: d2_at_m(ratio,  m) for m in [4, 6]}
    d2_dr = {m: d2_at_m(dratio, m) for m in [4, 6]}

    res[ma] = {"fnn_rt": fnn_rt, "fnn_dr": fnn_dr,
               "p_rt": p_rt, "p_dr": p_dr,
               "d2_rt": d2_rt, "d2_dr": d2_dr, "n": len(df)}

    p_rt_s = str(p_rt) if p_rt else "∞"
    p_dr_s = str(p_dr) if p_dr else "∞"
    print(f"{ma:>5}  {len(df):>5}  {p_rt_s:>8}  {p_dr_s:>9}  "
          f"{d2_rt[4]:>10.2f}  {d2_rt[6]:>10.2f}  "
          f"{d2_dr[4]:>10.2f}  {d2_dr[6]:>10.2f}")


# ── графики ───────────────────────────────────────────────────────────────────
n = len(MA_WINDOWS)
colors_rt = plt.cm.Greens(np.linspace(0.35, 0.9, n))
colors_dr = plt.cm.Blues(np.linspace(0.35, 0.9, n))

fig, (ax_fnn_rt, ax_fnn_dr, ax_d2) = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Стохастичность при разных MA — SBER 1d", fontsize=12)

for i, ma in enumerate(MA_WINDOWS):
    r = res[ma]
    lbl = f"MA={ma}"

    ms_rt = sorted(r["fnn_rt"])
    ms_dr = sorted(r["fnn_dr"])

    ax_fnn_rt.plot(ms_rt, [r["fnn_rt"][m]*100 for m in ms_rt],
                   "o-", color=colors_rt[i], lw=1.5, ms=4, label=lbl)
    ax_fnn_dr.plot(ms_dr, [r["fnn_dr"][m]*100 for m in ms_dr],
                   "o-", color=colors_dr[i], lw=1.5, ms=4, label=lbl)

    ms_d2 = [2, 4, 6]
    ax_d2.plot(ms_d2, [r["d2_rt"].get(m, np.nan) for m in ms_d2],
               "s--", color=colors_rt[i], lw=1.2, ms=4)
    ax_d2.plot(ms_d2, [r["d2_dr"].get(m, np.nan) for m in ms_d2],
               "o-",  color=colors_dr[i], lw=1.2, ms=4, label=lbl)

for ax, title in [(ax_fnn_rt, "FNN — ratio"),
                  (ax_fnn_dr, "FNN — Δratio")]:
    ax.axhline(5, color="red", ls="--", lw=1, label="5% порог")
    ax.set_xlabel("m"); ax.set_ylabel("FNN %")
    ax.set_title(title); ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.2)

ax_d2.plot([], [], "s--", color="gray", label="ratio (светлее=меньше MA)")
ax_d2.plot([], [], "o-",  color="gray", label="Δratio (синий)")
ax_d2.plot([2,4,6], [2,4,6], "k--", lw=0.6, alpha=0.3, label="D₂=m (стохаст.)")
ax_d2.set_xlabel("m"); ax_d2.set_ylabel("D₂")
ax_d2.set_title("D₂ — ratio (зелёный) vs Δratio (синий)")
ax_d2.legend(fontsize=7); ax_d2.grid(alpha=0.2); ax_d2.set_ylim(bottom=0)

plt.tight_layout()
out = ROOT / "research/figures/16_chaos_ma_sweep.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=140)
print(f"\nГрафик: {out}")
plt.show()
