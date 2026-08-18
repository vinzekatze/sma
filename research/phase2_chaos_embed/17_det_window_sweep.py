"""
DET при разных размерах окна: W = 50, 75, 100, 150, 200.

Гипотеза: маленькое окно (W=50–100) ближе к горизонту прогноза
и может сильнее коррелировать с реальной точностью, чем W=200.

Также проверяем:
  - Pearson r (DET vs mape_f5)
  - Gate-анализ (медиана DET: хуже/лучше)
  - Partial correlation DET vs mape_f5, контролируя val_mape
    (добавляет ли DET информацию поверх val_mape?)
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

MA_WINDOW  = 438
M_DRATIO   = 5
TARGET_RR  = 0.10
L_MIN      = 2
WINDOWS    = [50, 75, 100, 150, 200]


# ── DET ──────────────────────────────────────────────────────────────────────

def compute_det(series: np.ndarray, m: int,
                target_rr: float = TARGET_RR, l_min: int = L_MIN) -> float:
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
    eps    = np.percentile(d_flat, target_rr * 100)

    R         = (D < eps) & mask
    total_rec = int(R.sum())
    if total_rec == 0:
        return 0.0

    det_points = 0
    Ri = R.astype(np.int8)
    for k in list(range(1, N)) + list(range(-N + 1, 0)):
        d = np.diag(Ri, k)
        if len(d) < l_min:
            continue
        padded  = np.concatenate([[0], d, [0]])
        diffs   = np.diff(padded)
        starts  = np.where(diffs == 1)[0]
        ends    = np.where(diffs == -1)[0]
        lengths = ends - starts
        det_points += int(lengths[lengths >= l_min].sum())

    return det_points / total_rec


# ── данные ───────────────────────────────────────────────────────────────────

with open(DATA_FILE) as f:
    candles = json.load(f)

df_price = normalize(candles, window=MA_WINDOW)
df_price = df_price.dropna(subset=["ma"]).reset_index(drop=True)
ratio    = df_price["ratio"].values
dratio   = np.diff(ratio)

records  = [json.loads(l) for l in open(RESULTS_FILE)]
print(f"Прогнозов: {len(records)}")
print(f"Окна DET : {WINDOWS}\n")

# ── вычислить DET для каждого origin × window ────────────────────────────────

base_rows = []
det_cols  = {w: [] for w in WINDOWS}

for i, rec in enumerate(records):
    ok = rec["origin_k"]
    base_rows.append({
        "origin_k": ok,
        "val_mape": rec["val_mape"],
        "mape_f5":  rec["mape_f5"],
        "mape_f15": rec["mape_f15"],
        "h_last":   rec["h_last"],
    })
    for w in WINDOWS:
        start  = max(0, ok - w - M_DRATIO)
        seg    = dratio[start: ok]
        if len(seg) >= w:
            det_val = compute_det(seg[-w:], M_DRATIO)
        else:
            det_val = np.nan
        det_cols[w].append(det_val)

    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(records)} готово…")

df_r = pd.DataFrame(base_rows)
for w in WINDOWS:
    df_r[f"det_{w}"] = det_cols[w]

df_r = df_r.dropna()
print(f"\nСтрок после dropna: {len(df_r)}\n")


# ── partial correlation ───────────────────────────────────────────────────────

def partial_corr(x, y, z):
    """Pearson r(x, y | z): корреляция x–y при фиксированном z."""
    def res(a, b):
        slope, intercept, *_ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    rx = res(x, z)
    ry = res(y, z)
    return stats.pearsonr(rx, ry)


# ── таблица корреляций ────────────────────────────────────────────────────────

print(f"{'W':>5}  {'N':>4}  {'r(det,f5)':>10}  {'p':>7}  "
      f"{'r(det,val)':>11}  {'partial r|val':>14}  {'partial p':>10}")
print("─" * 70)

for w in WINDOWS:
    col  = f"det_{w}"
    sub  = df_r[[col, "val_mape", "mape_f5"]].dropna()
    n    = len(sub)
    det  = sub[col].values
    f5   = sub["mape_f5"].values
    vm   = sub["val_mape"].values

    r_f5,  p_f5  = stats.pearsonr(det, f5)
    r_vm,  _     = stats.pearsonr(det, vm)
    pr,    pp    = partial_corr(det, f5, vm)

    print(f"{w:>5}  {n:>4}  {r_f5:>10.3f}  {p_f5:>7.4f}  "
          f"{r_vm:>11.3f}  {pr:>14.3f}  {pp:>10.4f}")

# эталон val_mape
r_vm_f5, p_vm_f5 = stats.pearsonr(df_r["val_mape"], df_r["mape_f5"])
print(f"\nЭталон val_mape: r={r_vm_f5:.3f}  p={p_vm_f5:.4f}")


# ── gate-анализ ───────────────────────────────────────────────────────────────

print(f"\n{'W':>5}  {'ниже медианы':>14}  {'выше медианы':>14}  {'разн×':>7}  {'MW p':>8}")
print("─" * 55)

for w in WINDOWS:
    col    = f"det_{w}"
    sub    = df_r[[col, "mape_f5"]].dropna()
    median = sub[col].median()
    lo     = sub[sub[col] <= median]["mape_f5"].values
    hi     = sub[sub[col] >  median]["mape_f5"].values
    mw     = stats.mannwhitneyu(lo, hi, alternative="two-sided")
    ratio_ = np.median(hi) / np.median(lo) if np.median(lo) > 0 else np.nan
    print(f"{w:>5}  {np.median(lo):>14.4%}  {np.median(hi):>14.4%}  "
          f"{ratio_:>7.2f}×  {mw.pvalue:>8.4f}")

# эталон val_mape
vm_med = df_r["val_mape"].median()
lo_vm  = df_r[df_r["val_mape"] <= vm_med]["mape_f5"].values
hi_vm  = df_r[df_r["val_mape"] >  vm_med]["mape_f5"].values
mw_vm  = stats.mannwhitneyu(lo_vm, hi_vm, alternative="two-sided")
print(f"{'val_mape':>5}  {np.median(lo_vm):>14.4%}  {np.median(hi_vm):>14.4%}  "
      f"{np.median(hi_vm)/np.median(lo_vm):>7.2f}×  {mw_vm.pvalue:>8.4f}")


# ── графики ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, len(WINDOWS), figsize=(4 * len(WINDOWS), 8))
fig.suptitle("DET dratio при разных окнах W — SBER 1d (175 прогнозов)", fontsize=12)

colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(WINDOWS)))

for col_i, w in enumerate(WINDOWS):
    col  = f"det_{w}"
    sub  = df_r[[col, "mape_f5"]].dropna()
    det  = sub[col].values
    f5   = sub["mape_f5"].values
    r, p = stats.pearsonr(det, f5)
    c    = colors[col_i]

    # scatter
    ax = axes[0][col_i]
    x_cap = np.clip(det, *np.percentile(det, [1, 99]))
    y_cap = np.clip(f5,  *np.percentile(f5,  [1, 99]))
    ax.scatter(x_cap, y_cap, alpha=0.4, s=18, color=c)
    xs = np.linspace(x_cap.min(), x_cap.max(), 100)
    ax.plot(xs, np.polyval(np.polyfit(x_cap, y_cap, 1), xs), color=c, lw=2)
    ax.set_title(f"W={w}  r={r:.3f}  p={p:.3f}")
    ax.set_xlabel("DET")
    ax.set_ylabel("mape_f5")
    ax.grid(alpha=0.2)

    # gate histogram
    ax2 = axes[1][col_i]
    median = np.median(det)
    lo_g   = f5[det <= median]
    hi_g   = f5[det >  median]
    cap    = np.percentile(f5, 97)
    bins   = np.linspace(0, cap, 25)
    ax2.hist(lo_g, bins=bins, alpha=0.6, color=c,          label=f"≤ med (n={len(lo_g)})")
    ax2.hist(hi_g, bins=bins, alpha=0.6, color="lightgray", label=f"> med (n={len(hi_g)})")
    ax2.axvline(np.median(lo_g), color=c,      ls="--", lw=1.5)
    ax2.axvline(np.median(hi_g), color="gray", ls="--", lw=1.5)
    ax2.set_title(f"Gate W={w}")
    ax2.set_xlabel("mape_f5")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.2)

plt.tight_layout()
out = ROOT / "research/figures/17_det_window_sweep.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=130)
print(f"\nГрафик: {out}")
plt.show()
