"""
DET как rolling gate: коррелирует ли детерминизм с точностью прогноза?

Берём 175 прогнозов из исследования 01 (SBER 1d).
Для каждого origin вычисляем DET на окне W баров до origin.
Сравниваем с val_mape и mape_f5 (фактическая точность).

Эталон из исследования 01:
  - Hurst vs mape_f5:   r ≈ 0    (не работает как gate)
  - val_mape vs mape_f5: r ≈ 0.51 (работает)
"""

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sma.core.forecast.normalize import normalize
from sma.core.forecast.embedding import build_delay_matrix

DATA_FILE    = ROOT / "data/candles/SBER/1d.json"
RESULTS_FILE = ROOT / "research/results/01_results.jsonl"

# ── параметры ────────────────────────────────────────────────────────────────
MA_WINDOW = 438    # то же что в исследовании 01
WIN       = 200    # окно DET (баров до origin)
M_DRATIO  = 5      # embedding для dratio
M_RATIO   = 4      # embedding для ratio
TARGET_RR = 0.10
L_MIN     = 2


# ── DET ──────────────────────────────────────────────────────────────────────

def compute_det(series: np.ndarray, m: int, target_rr=TARGET_RR, l_min=L_MIN) -> float:
    try:
        X, _ = build_delay_matrix(series, m)
    except ValueError:
        return np.nan

    N = len(X)
    if N < l_min + 2:
        return np.nan

    diff = X[:, None, :] - X[None, :, :]
    D    = np.sqrt((diff ** 2).sum(axis=-1))
    mask = ~np.eye(N, dtype=bool)
    d_flat = D[mask]
    eps  = np.percentile(d_flat, target_rr * 100)

    R    = (D < eps) & mask
    total_rec = int(R.sum())
    if total_rec == 0:
        return 0.0

    det_points = 0
    Ri = R.astype(np.int8)
    for k in list(range(1, N)) + list(range(-N + 1, 0)):
        d = np.diag(Ri, k)
        if len(d) < l_min:
            continue
        padded = np.concatenate([[0], d, [0]])
        diffs  = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends   = np.where(diffs == -1)[0]
        lengths = ends - starts
        det_points += int(lengths[lengths >= l_min].sum())

    return det_points / total_rec


# ── данные ───────────────────────────────────────────────────────────────────
with open(DATA_FILE) as f:
    candles = json.load(f)

df     = normalize(candles, window=MA_WINDOW)
df     = df.dropna(subset=["ma"]).reset_index(drop=True)
ratio  = df["ratio"].values
dratio = np.diff(ratio)

records = [json.loads(l) for l in open(RESULTS_FILE)]
print(f"Прогнозов: {len(records)}")

# ── вычислить DET для каждого origin ─────────────────────────────────────────
rows = []
for i, rec in enumerate(records):
    ok = rec["origin_k"]

    # dratio: окно [ok-WIN .. ok-1]  (индексы в dratio)
    dr_start = max(0, ok - WIN - M_DRATIO)
    dr_seg   = dratio[dr_start: ok]
    det_dr   = compute_det(dr_seg[-WIN:], M_DRATIO) if len(dr_seg) >= WIN else np.nan

    # ratio: окно [ok-WIN .. ok]
    rt_start = max(0, ok - WIN)
    rt_seg   = ratio[rt_start: ok + 1]
    det_rt   = compute_det(rt_seg[-WIN:], M_RATIO) if len(rt_seg) >= WIN else np.nan

    rows.append({
        "origin_k":  ok,
        "val_mape":  rec["val_mape"],
        "mape_f5":   rec["mape_f5"],
        "mape_f15":  rec["mape_f15"],
        "h_last":    rec["h_last"],
        "det_dr":    det_dr,
        "det_rt":    det_rt,
    })

    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(records)} готово…")

df_r = pd.DataFrame(rows).dropna()
print(f"Строк после dropna: {len(df_r)}")


# ── корреляции ───────────────────────────────────────────────────────────────
pairs = [
    ("DET dratio",  "det_dr",   "mape_f5",  "steelblue"),
    ("DET ratio",   "det_rt",   "mape_f5",  "seagreen"),
    ("Hurst",       "h_last",   "mape_f5",  "darkorange"),
    ("val_mape",    "val_mape", "mape_f5",  "purple"),
]

print(f"\n{'Предиктор':<14}  {'r (mape_f5)':>12}  {'p-value':>10}  {'r (mape_f15)':>13}")
print("-" * 56)
for name, xcol, _, _ in pairs:
    x = df_r[xcol].values
    r5,  p5  = stats.pearsonr(x, df_r["mape_f5"].values)
    r15, p15 = stats.pearsonr(x, df_r["mape_f15"].values)
    print(f"{name:<14}  {r5:>12.3f}  {p5:>10.4f}  {r15:>13.3f}")


# ── scatter-графики ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("DET как gate: корреляция с mape_f5 — SBER 1d (175 прогнозов)", fontsize=12)

for ax, (name, xcol, ycol, color) in zip(axes.flat, pairs):
    x = df_r[xcol].values
    y = df_r[ycol].values

    # cap outliers for visibility
    x_cap = np.clip(x, *np.percentile(x, [1, 99]))
    y_cap = np.clip(y, *np.percentile(y, [1, 99]))

    r, p = stats.pearsonr(x, y)
    ax.scatter(x_cap, y_cap, alpha=0.45, s=20, color=color)

    # trend line
    coeffs = np.polyfit(x_cap, y_cap, 1)
    xs = np.linspace(x_cap.min(), x_cap.max(), 100)
    ax.plot(xs, np.polyval(coeffs, xs), color=color, lw=2)

    ax.set_xlabel(name)
    ax.set_ylabel("mape_f5 (фактическая)")
    ax.set_title(f"{name}  →  r={r:.3f}  p={p:.4f}")
    ax.grid(alpha=0.25)

plt.tight_layout()
out = ROOT / "research/figures/13_det_gate_scatter.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=140)
print(f"\nScatter: {out}")


# ── gate-анализ: разделить по медиане DET ────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(13, 5))
fig2.suptitle("Распределение mape_f5 выше/ниже порога (медиана индикатора)", fontsize=11)

gate_cols = [
    ("DET dratio", "det_dr",   "steelblue"),
    ("DET ratio",  "det_rt",   "seagreen"),
    ("val_mape",   "val_mape", "purple"),
]

print(f"\n{'Индикатор':<14}  {'ниже медианы':>13}  {'выше медианы':>13}  {'разница×':>10}  {'MW p':>8}")
print("-" * 65)

for ax, (name, col, color) in zip(axes2, gate_cols):
    median = df_r[col].median()
    lo  = df_r[df_r[col] <= median]["mape_f5"].values
    hi  = df_r[df_r[col] >  median]["mape_f5"].values

    mw  = stats.mannwhitneyu(lo, hi, alternative="two-sided")
    ratio_med = np.median(hi) / np.median(lo) if np.median(lo) > 0 else np.nan

    print(f"{name:<14}  {np.median(lo):>13.4%}  {np.median(hi):>13.4%}  "
          f"{ratio_med:>10.2f}×  {mw.pvalue:>8.4f}")

    cap = np.percentile(df_r["mape_f5"].values, 97)
    bins = np.linspace(0, cap, 30)
    ax.hist(lo, bins=bins, alpha=0.6, color=color,     label=f"≤ медиана (n={len(lo)})")
    ax.hist(hi, bins=bins, alpha=0.6, color="lightgray", label=f"> медиана (n={len(hi)})")
    ax.axvline(np.median(lo), color=color,     ls="--", lw=1.5)
    ax.axvline(np.median(hi), color="gray",    ls="--", lw=1.5)
    ax.set_title(f"{name}  (MW p={mw.pvalue:.3f})")
    ax.set_xlabel("mape_f5")
    ax.set_ylabel("кол-во прогнозов")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
out2 = ROOT / "research/figures/13_det_gate_hist.png"
plt.savefig(out2, dpi=140)
print(f"Гистограммы: {out2}")

plt.show()
