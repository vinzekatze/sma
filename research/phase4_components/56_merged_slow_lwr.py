"""
56 — LWR на объединённом медленном сигнале: sum(C2..C5) или sum(C3..C5).

Идея: не разбивать медленные компоненты на отдельные ряды (как в текущем sep-подходе),
а прогнозировать их сумму единым LWR. Аттрактор суммарного сигнала не ломается
межкомпонентными взаимодействиями.

Отбрасываем только C0 (~2-4б, шум) и C1 (~4-8б, слабая структура).
Прогнозируем: merged = sum(dratio[ci] for ci in [2,3,4,5]) одним forecast_lwr.

Конфигурации:
  A. sep_C3-C5  p=20         ← текущий стандарт (sep = per-component)
  B. merged_C3-C5  sweep p   ← merged без C2, разные p
  C. merged_C2-C5  sweep p   ← merged с C2, разные p

P_SWEEP: [5, 8, 13, 20, 30, 50] — включая FNN-оптимум (скр.54) и стандартный p=20.

Walk-forward: N_ORIG=200, horizon=15, 8 тикеров 1d.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.stats import ttest_rel

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
FIG_DIR  = ROOT / "research/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from sma.core.forecast.embedding import build_delay_matrix, last_vector

# ── параметры ─────────────────────────────────────────────────────────────────

TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
N_ORIG       = 200
HORIZON      = 15
FILTER_ORDER = 4
CUTOFFS      = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
P_SWEEP      = [5, 8, 13, 20, 30, 50]

CONFIGS: list[dict] = [
    # baseline: sep C3-C5, p=20
    {"name": "sep  C3-C5  p=20",  "mode": "sep",    "comp": [3,4,5], "p": 20},
]
for _p in P_SWEEP:
    CONFIGS.append({"name": f"merged C3-C5  p={_p}", "mode": "merged", "comp": [3,4,5], "p": _p})
for _p in P_SWEEP:
    CONFIGS.append({"name": f"merged C2-C5  p={_p}", "mode": "merged", "comp": [2,3,4,5], "p": _p})

# ── helpers ───────────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n   = len(close)
    lc  = np.log(np.maximum(close, 1e-10))
    t   = np.arange(n, dtype=np.float64)
    cn  = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t)
    trend[:2] = close[:2]
    return trend


def make_filter_bank(series: np.ndarray) -> np.ndarray:
    components: list[np.ndarray] = []
    remaining = series.copy()
    for fc in CUTOFFS:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        components.append(remaining - low)
        remaining = low
    components.append(remaining)
    return np.array(components)   # (6, N)


def forecast_lwr_single(hist: np.ndarray, p: int, horizon: int) -> np.ndarray:
    """LWR на одном сигнале, ξ = 3*(p+1)."""
    xi = 3 * (p + 1)
    if len(hist) < xi + p + 2:
        return np.zeros(horizon)
    X, y    = build_delay_matrix(hist, p, tau=1)
    current = last_vector(hist, p, tau=1).copy()
    if len(X) < xi:
        return np.zeros(horizon)
    out = np.empty(horizon)
    for h in range(horizon):
        dists  = np.linalg.norm(X - current, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        nn_d   = dists[nn_idx]
        h_bw   = nn_d.max()
        if h_bw < 1e-12:
            h_bw = 1e-10
        w        = np.exp(-0.5 * (nn_d / h_bw) ** 2)
        X_nn     = X[nn_idx];  y_nn = y[nn_idx]
        A        = np.hstack([np.ones((xi, 1)), X_nn])
        ws       = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(ws[:, None] * A, ws * y_nn, rcond=None)
        val      = float(c[0] + current @ c[1:])
        out[h]   = val
        current  = np.roll(current, -1)
        current[-1] = val
    return out

# ── walk-forward ──────────────────────────────────────────────────────────────

mapes: list[dict[str, list[float]]] = [{} for _ in CONFIGS]
for ci in range(len(CONFIGS)):
    for t in TICKERS:
        mapes[ci][t] = []

t0     = time.time()
n_done = 0

for ticker in TICKERS:
    path = DATA_DIR / ticker / "1d.json"
    with open(path) as f:
        candles = json.load(f)
    close = np.array([float(c["close"]) for c in candles], dtype=np.float64)
    ratio = close / logtrend_causal(close)
    N     = len(ratio)

    max_ok = N - 1 - HORIZON
    min_ok = max(300, N - 700)
    origins = np.unique(np.linspace(min_ok, max_ok, N_ORIG, dtype=int))

    for origin_k in origins:
        dratio      = np.diff(ratio[:origin_k + 1])
        comp        = make_filter_bank(dratio)          # (6, origin_k)
        actual_ratio = ratio[origin_k + 1: origin_k + 1 + HORIZON]
        if len(actual_ratio) < HORIZON:
            continue
        ratio0 = float(ratio[origin_k])

        for ci, cfg in enumerate(CONFIGS):
            if cfg["mode"] == "sep":
                # per-component LWR, сумма результатов (baseline)
                dhat = np.zeros(HORIZON)
                ok   = True
                for c_idx in cfg["comp"]:
                    d = forecast_lwr_single(comp[c_idx], cfg["p"], HORIZON)
                    if np.any(np.isnan(d)):
                        ok = False; break
                    dhat += d
            else:
                # единый LWR на суммарном сигнале
                merged = np.sum([comp[c_idx] for c_idx in cfg["comp"]], axis=0)
                dhat   = forecast_lwr_single(merged, cfg["p"], HORIZON)
                ok     = not np.any(np.isnan(dhat))

            if not ok or np.any(np.abs(dhat) > 1e6):
                continue

            ratio_hat = ratio0 + np.cumsum(dhat)
            mape = float(np.mean(
                np.abs(ratio_hat - actual_ratio) / (np.abs(actual_ratio) + 1e-10)
            ))
            if np.isfinite(mape):
                mapes[ci][ticker].append(mape)

        n_done += 1
        if n_done % 200 == 0:
            print(f"  {n_done}/{len(TICKERS)*len(origins)}  ({time.time()-t0:.0f}s)")

print(f"\nГотово за {time.time()-t0:.1f} с")

# ── агрегация ──────────────────────────────────────────────────────────────────

baseline_flat = [v for t in TICKERS for v in mapes[0][t]]
baseline_agg  = float(np.mean(baseline_flat))

print(f"\n── AGG MAPE ──────────────────────────────────────────────────────────────")
print(f"{'#':<2}  {'Конфигурация':<28}  {'AGG MAPE':>9}  {'Δ%':>8}  {'p-val':>7}")
print("─" * 62)

agg_results: list[dict] = []
for ci, cfg in enumerate(CONFIGS):
    flat = [v for t in TICKERS for v in mapes[ci][t]]
    if not flat:
        continue
    agg = float(np.mean(flat))
    delta = (agg / baseline_agg - 1.0) * 100.0 if ci > 0 else 0.0
    if ci > 0 and len(flat) == len(baseline_flat):
        _, pval = ttest_rel(flat, baseline_flat)
    else:
        pval = float("nan")
    sign = "✅" if delta < -0.5 else ("❌" if delta > 0.5 else "~")
    pstr = f"{pval:.4f}" if np.isfinite(pval) else "—"
    print(f"{ci:<2}  {cfg['name']:<28}  {agg:.5f}   {delta:>+7.2f}%  {sign}  {pstr}")
    agg_results.append({"ci": ci, "cfg": cfg, "agg": agg, "delta": delta,
                        "pval": pval, "flat": flat})

# ── лучшие по группам ─────────────────────────────────────────────────────────

for group_name, group_filter in [("merged C3-C5", lambda r: "C3-C5" in r["cfg"]["name"] and r["ci"] > 0),
                                  ("merged C2-C5", lambda r: "C2-C5" in r["cfg"]["name"])]:
    group = [r for r in agg_results if group_filter(r)]
    if not group:
        continue
    best = min(group, key=lambda r: r["agg"])
    print(f"\nЛучший {group_name}: {best['cfg']['name']}  "
          f"AGG={best['agg']:.5f}  Δ={best['delta']:+.2f}%  p={best['pval']:.4f}")
    print(f"  Per-ticker vs baseline:")
    for ticker in TICKERS:
        b = np.mean(mapes[0][ticker]) if mapes[0][ticker] else float("nan")
        c = np.mean(mapes[best["ci"]][ticker]) if mapes[best["ci"]][ticker] else float("nan")
        d = (c/b - 1)*100 if np.isfinite(b) and np.isfinite(c) else float("nan")
        print(f"    {ticker}: {b:.5f} → {c:.5f}  ({d:+.2f}%)")

# ── графики ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax_idx, (group_tag, group_filter) in enumerate([
        ("C3-C5", lambda r: "C3-C5" in r["cfg"]["name"]),
        ("C2-C5", lambda r: "C2-C5" in r["cfg"]["name"]),
]):
    ax = axes[ax_idx]
    group = [r for r in agg_results if group_filter(r)]
    if not group:
        continue
    ps    = [r["cfg"]["p"] for r in group]
    aggs  = [r["agg"] for r in group]
    deltas = [r["delta"] for r in group]
    colors = ["#43a047" if d < -0.5 else ("#e53935" if d > 0.5 else "#ffa726")
              for d in deltas]

    ax.bar(range(len(ps)), aggs, color=colors, alpha=0.8, width=0.6)
    ax.axhline(baseline_agg, color="#546e7a", lw=1.8, ls="--",
               label=f"sep C3-C5 p=20 = {baseline_agg:.5f}")
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([str(p) for p in ps])
    ax.set_xlabel("p")
    ax.set_ylabel("AGG MAPE")
    ax.set_title(f"merged {group_tag} — AGG MAPE vs p")
    ax.legend(fontsize=9)
    for i, (agg, delta) in enumerate(zip(aggs, deltas)):
        ax.text(i, agg + 0.0001, f"{delta:+.1f}%", ha="center", fontsize=8)

fig.suptitle("LWR на merged-сигнале vs sep-baseline (8 тикеров 1d, N=200 origins)")
fig.tight_layout()
path_fig = FIG_DIR / "56_merged_slow_lwr.png"
fig.savefig(path_fig, dpi=150)
print(f"\nГрафик: {path_fig}")

# ── итог ──────────────────────────────────────────────────────────────────────

best_all = min((r for r in agg_results if r["ci"] > 0), key=lambda r: r["agg"])
print(f"\n── Итог ──────────────────────────────────────────────────────────────────")
print(f"Baseline sep C3-C5 p=20:  {baseline_agg:.5f}")
print(f"Лучший merged:            {best_all['agg']:.5f}  "
      f"{best_all['delta']:+.2f}%  [{best_all['cfg']['name']}]")
pv = best_all["pval"]
if np.isfinite(pv):
    print(f"t-test: p={pv:.4f}  "
          f"({'значимо' if pv < 0.05 else 'НЕ значимо'})")
print(f"\n[Справка] Скр.52 Damped AR C1+C2: −8.01%")
print(f"[Справка] Скр.55 LWR sep C2-C5 лучший: +15.7%")
