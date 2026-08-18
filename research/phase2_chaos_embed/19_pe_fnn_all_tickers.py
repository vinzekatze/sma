"""
PE → FNN по всем тикерам (1d).

Воспроизводит 18_pe_fnn_local.py для каждого тикера и сводит результаты
в одну таблицу + сравнительный график FNN-кривых.

Вопрос: универсален ли вывод — «структурные участки (PE < q25) имеют
меньше ложных соседей при малых m, но аттрактора нет нигде»?
"""

import json, sys
from math import factorial
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KDTree

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from sma.core.forecast.normalize import normalize

# ── параметры ────────────────────────────────────────────────────────────────
MA_WINDOW = 1000
PE_ORDER  = 3
PE_DELAY  = 1
PE_WIN    = 50
STRUCT_Q  = 0.25
SEG_MIN   = 40
M_MAX     = 10
TAU       = 1
R_TOL     = 10.0
A_TOL     = 2.0

TICKERS = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]


# ── PE ────────────────────────────────────────────────────────────────────────
def _perm_index(order):
    return {p: i for i, p in enumerate(permutations(range(order)))}

_PERM_IDX = _perm_index(PE_ORDER)

def perm_entropy(x, order=PE_ORDER, delay=PE_DELAY):
    n = len(x)
    run = order * delay
    if n < run:
        return np.nan
    counts = np.zeros(factorial(order))
    for i in range(n - run + delay):
        w = x[i: i + run: delay][:order]
        counts[_PERM_IDX[tuple(np.argsort(w))]] += 1
    p = counts[counts > 0]; p /= p.sum()
    return -np.sum(p * np.log(p)) / np.log(factorial(order))

def rolling_pe(series, win=PE_WIN):
    result = np.full(len(series), np.nan)
    for i in range(win - 1, len(series)):
        result[i] = perm_entropy(series[i - win + 1: i + 1])
    return result


# ── FNN ───────────────────────────────────────────────────────────────────────
def fnn(series, m_max=M_MAX, tau=TAU):
    sigma = np.std(series)
    res = {}
    for m in range(1, m_max + 1):
        n_m1 = len(series) - m * tau - 1
        n_m  = len(series) - (m - 1) * tau - 1
        if n_m1 < 20:
            break
        X_m = np.column_stack([series[k*tau: k*tau+n_m] for k in range(m)])
        dists, idxs = KDTree(X_m[:n_m1]).query(X_m[:n_m1], k=2)
        R_m = dists[:, 1]; j = idxs[:, 1]
        ei = series[m*tau: m*tau+n_m1]
        ej = series[m*tau: m*tau+n_m1][j]
        R_m1 = np.sqrt(R_m**2 + (ei - ej)**2)
        is_false = (np.abs(ei - ej) / (R_m + 1e-12) > R_TOL) | (R_m1 / sigma > A_TOL)
        valid = R_m > 1e-12
        res[m] = float(np.sum(is_false[valid]) / np.sum(valid)) if np.any(valid) else np.nan
    return res

def fnn_on_segments(series, mask, seg_min=SEG_MIN):
    segs = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]: j += 1
            if j - i >= seg_min:
                segs.append(series[i:j])
            i = j
        else:
            i += 1
    if not segs:
        return {}, 0
    agg, wgt = {}, {}
    for seg in segs:
        for m, v in fnn(seg).items():
            if not np.isnan(v):
                agg[m]  = agg.get(m, 0.0) + v * len(seg)
                wgt[m]  = wgt.get(m, 0.0) + len(seg)
    return ({m: agg[m]/wgt[m] for m in sorted(agg)}, len(segs))


# ── обработка одного тикера ───────────────────────────────────────────────────
def process(ticker):
    path = DATA_DIR / ticker / "1d.json"
    with open(path) as f:
        candles = json.load(f)
    df = normalize(candles, window=MA_WINDOW).dropna(subset=["ma"]).reset_index(drop=True)
    dratio = np.diff(df["ratio"].values)

    pe_series = rolling_pe(dratio)
    valid_pe  = pe_series[~np.isnan(pe_series)]
    q_lo = np.quantile(valid_pe, STRUCT_Q)

    mask_struct = pe_series < q_lo
    n_bars      = int(mask_struct.sum())

    fnn_all             = fnn(dratio)
    fnn_struct, n_segs  = fnn_on_segments(dratio, mask_struct)

    return {
        "ticker":      ticker,
        "n_bars":      len(dratio),
        "pe_q25":      round(q_lo, 4),
        "pe_median":   round(float(np.median(valid_pe)), 4),
        "struct_bars": n_bars,
        "n_segs":      n_segs,
        "fnn_all":     fnn_all,
        "fnn_struct":  fnn_struct,
    }


# ── запуск ────────────────────────────────────────────────────────────────────
print("Тикер       баров  PE_q25  сегм.  n_сегм  │ FNN_all m2   FNN_struct m2  │ FNN_all m3   FNN_struct m3")
print("─" * 105)

results = []
for ticker in TICKERS:
    r = process(ticker)
    results.append(r)
    fa2 = r["fnn_all"].get(2, float("nan"))
    fs2 = r["fnn_struct"].get(2, float("nan"))
    fa3 = r["fnn_all"].get(3, float("nan"))
    fs3 = r["fnn_struct"].get(3, float("nan"))
    print(f"{ticker:<8}  {r['n_bars']:>5}  {r['pe_q25']:.4f}  "
          f"{r['struct_bars']:>5}  {r['n_segs']:>6}  │ "
          f"{fa2:>10.1%}   {fs2:>12.1%}  │ "
          f"{fa3:>10.1%}   {fs3:>12.1%}")


# ── сводная таблица по всем m ─────────────────────────────────────────────────
print("\n\n── Подробная таблица: FNN_all vs FNN_struct ──")
print(f"{'':>6}", end="")
for r in results:
    print(f"  {r['ticker']:>18}", end="")
print()
print(f"{'m':>6}", end="")
for _ in results:
    print(f"  {'all':>8} {'struct':>8}", end="")
print()
print("─" * (6 + len(results) * 20))

all_m = sorted({m for r in results for m in r["fnn_all"]})
for m in all_m:
    print(f"{m:>6}", end="")
    for r in results:
        a = r["fnn_all"].get(m, float("nan"))
        s = r["fnn_struct"].get(m, float("nan"))
        marker = "*" if (not np.isnan(s) and not np.isnan(a) and s < a * 0.8) else " "
        print(f"  {a:>7.1%}{marker} {s:>7.1%} ", end="")
    print()
print("  (* — struct лучше all на 20%+)")


# ── график: FNN-кривые по каждому тикеру ─────────────────────────────────────
n = len(results)
cols = 4
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5),
                         sharey=True, sharex=True)
axes = axes.flatten()

for i, r in enumerate(results):
    ax = axes[i]
    ms_all    = sorted(r["fnn_all"])
    ms_struct = sorted(r["fnn_struct"])

    ax.plot(ms_all, [r["fnn_all"][m] * 100 for m in ms_all],
            "o-", color="steelblue", linewidth=1.5, markersize=4, label="весь ряд")
    if ms_struct:
        ax.plot(ms_struct, [r["fnn_struct"][m] * 100 for m in ms_struct],
                "s--", color="seagreen", linewidth=1.5, markersize=4,
                label=f"struct (PE<q25, {r['n_segs']} сег.)")
    ax.axhline(5, color="red", linestyle=":", linewidth=1)
    ax.set_title(r["ticker"], fontsize=11, fontweight="bold")
    ax.set_xlabel("m")
    ax.set_ylabel("FNN %")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(f"FNN: весь ряд vs структурные участки (PE < q25, W={PE_WIN})  |  Δratio 1d",
             fontsize=12)
plt.tight_layout()
out = OUT_DIR / "19_pe_fnn_all_tickers.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out}")


# ── итог ─────────────────────────────────────────────────────────────────────
print("\n── Итог ──")
improved = [r["ticker"] for r in results
            if r["fnn_struct"].get(2, 1.0) < r["fnn_all"].get(2, 1.0) * 0.8]
flat     = [r["ticker"] for r in results if r["ticker"] not in improved]
print(f"Структурные участки заметно лучше (FNN m=2 < 80% от baseline): {improved or '—'}")
print(f"Без значимого улучшения: {flat or '—'}")

no_segs = [r["ticker"] for r in results if r["n_segs"] == 0]
if no_segs:
    print(f"Нет валидных сегментов (< {SEG_MIN} баров): {no_segs}")

plt.show()
