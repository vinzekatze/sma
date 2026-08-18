"""
55 — Добавление C2 (и C1) в LWR: sweep p_C2, сравнение с baseline C3-C5.

Скрипт 54 показал: при w=8 (удаление C0+C1) аттрактор найден у 8/8 тикеров,
p_opt=5. Проверяем, даёт ли включение C2 в LWR улучшение AGG MAPE.

Конфигурации (ξ_ci = 3*(p_ci+1)):
  0. C3-C5  p=20            ← baseline
  1. C2-C5  p_C2=3
  2. C2-C5  p_C2=5          ← p_opt по FNN (скр.54)
  3. C2-C5  p_C2=8
  4. C2-C5  p_C2=13
  5. C1-C5  p_C1=3 p_C2=5
  6. C1-C5  p_C1=5 p_C2=5

Walk-forward: N_ORIG=200, horizon=15, origins равномерно из последних 700 баров.
Метрика: AGG MAPE = mean по всем (ticker, origin).
Сравнение со скр.52: Damped AR C1+C2 → −8.01% vs C3-C5 baseline.
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

# конфигурации: comp_p = {ci: p_ci}, ξ = 3*(p+1) для каждой
CONFIGS = [
    {"name": "C3-C5  p=20",             "comp_p": {3: 20, 4: 20, 5: 20}},
    {"name": "C2-C5  p_C2=3",           "comp_p": {2:  3, 3: 20, 4: 20, 5: 20}},
    {"name": "C2-C5  p_C2=5  ★FNN",     "comp_p": {2:  5, 3: 20, 4: 20, 5: 20}},
    {"name": "C2-C5  p_C2=8",           "comp_p": {2:  8, 3: 20, 4: 20, 5: 20}},
    {"name": "C2-C5  p_C2=13",          "comp_p": {2: 13, 3: 20, 4: 20, 5: 20}},
    {"name": "C1-C5  p_C1=3 p_C2=5",   "comp_p": {1:  3, 2:  5, 3: 20, 4: 20, 5: 20}},
    {"name": "C1-C5  p_C1=5 p_C2=5",   "comp_p": {1:  5, 2:  5, 3: 20, 4: 20, 5: 20}},
]

# ── logtrend + filter bank ────────────────────────────────────────────────────

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
    """Causal Butterworth filter bank → (6, N) components C0..C5."""
    components: list[np.ndarray] = []
    remaining = series.copy()
    for fc in CUTOFFS:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        components.append(remaining - low)
        remaining = low
    components.append(remaining)
    return np.array(components)


# ── LWR (встроенная реализация, не зависит от prototype) ─────────────────────

def forecast_lwr_comp(
    comp_hist: np.ndarray,
    p: int,
    horizon: int,
) -> np.ndarray:
    """LWR на одной компоненте. ξ = 3*(p+1)."""
    xi = 3 * (p + 1)
    n  = len(comp_hist)
    if n < xi + p + 2:
        return np.zeros(horizon)

    X, y    = build_delay_matrix(comp_hist, p, tau=1)
    current = last_vector(comp_hist, p, tau=1).copy()

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
        w = np.exp(-0.5 * (nn_d / h_bw) ** 2)

        X_nn = X[nn_idx];  y_nn = y[nn_idx]
        A    = np.hstack([np.ones((xi, 1)), X_nn])
        ws   = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(ws[:, None] * A, ws * y_nn, rcond=None)
        val      = float(c[0] + current @ c[1:])
        out[h]   = val
        current  = np.roll(current, -1)
        current[-1] = val

    return out


# ── walk-forward ──────────────────────────────────────────────────────────────

# mapes[cfg_idx][ticker] = list of mape values
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
    trend = logtrend_causal(close)
    ratio = close / trend
    N     = len(ratio)

    # равномерно отобранные origins из последних 700 баров
    max_ok = N - 1 - HORIZON
    min_ok = max(300, N - 700)
    if max_ok <= min_ok:
        continue
    origins = np.linspace(min_ok, max_ok, N_ORIG, dtype=int)
    origins = np.unique(origins)

    for origin_k in origins:
        dratio      = np.diff(ratio[:origin_k + 1])
        comp        = make_filter_bank(dratio)          # (6, origin_k)
        actual_ratio = ratio[origin_k + 1: origin_k + 1 + HORIZON]
        if len(actual_ratio) < HORIZON:
            continue
        ratio0 = float(ratio[origin_k])

        for ci, cfg in enumerate(CONFIGS):
            dratio_hat = np.zeros(HORIZON)
            ok = True
            for comp_idx, p_ci in cfg["comp_p"].items():
                comp_hist = comp[comp_idx]
                dhat = forecast_lwr_comp(comp_hist, p_ci, HORIZON)
                if np.any(np.isnan(dhat)) or np.any(np.abs(dhat) > 1e6):
                    ok = False
                    break
                dratio_hat += dhat

            if not ok:
                continue

            ratio_hat = ratio0 + np.cumsum(dratio_hat)
            mape = float(np.mean(
                np.abs(ratio_hat - actual_ratio)
                / (np.abs(actual_ratio) + 1e-10)
            ))
            if np.isfinite(mape):
                mapes[ci][ticker].append(mape)

        n_done += 1
        if n_done % 100 == 0:
            elapsed = time.time() - t0
            total   = len(TICKERS) * len(origins)
            print(f"  {n_done}/{total}  ({elapsed:.0f}s)")

print(f"\nГотово за {time.time() - t0:.1f} с")

# ── агрегация результатов ─────────────────────────────────────────────────────

print("\n── AGG MAPE по конфигурациям ─────────────────────────────────────────────")
print(f"{'#':<2}  {'Конфигурация':<35}  {'AGG MAPE':>9}  {'Δ% vs C3-C5':>12}  {'p-value':>9}")
print("─" * 75)

baseline_flat = [v for ticker in TICKERS for v in mapes[0][ticker]]
agg_results: list[dict] = []

for ci, cfg in enumerate(CONFIGS):
    flat = [v for ticker in TICKERS for v in mapes[ci][ticker]]
    if not flat:
        continue
    agg = float(np.mean(flat))
    delta_pct = (agg / np.mean(baseline_flat) - 1.0) * 100.0 if ci > 0 else 0.0

    if ci > 0 and len(flat) == len(baseline_flat):
        _, pval = ttest_rel(flat, baseline_flat)
    else:
        pval = float("nan")

    sign = "✅" if delta_pct < -0.5 else ("❌" if delta_pct > 0.5 else "~")
    print(f"{ci:<2}  {cfg['name']:<35}  {agg:.5f}   "
          f"{delta_pct:>+8.2f}%  {sign}  "
          f"{'p=' + f'{pval:.4f}' if np.isfinite(pval) else '—':>9}")
    agg_results.append({"ci": ci, "name": cfg["name"], "agg": agg,
                        "delta_pct": delta_pct, "pval": pval, "flat": flat})

# ── per-ticker разбивка для лучшей конфигурации C2-C5 ────────────────────────

best_c2_idx = min(
    (r for r in agg_results if "C2-C5" in r["name"]),
    key=lambda r: r["agg"],
    default=None,
)
if best_c2_idx is not None:
    print(f"\n── Per-ticker: {best_c2_idx['name']} vs C3-C5 baseline ─────────────────")
    print(f"{'Ticker':<6}  {'baseline':>9}  {'best C2-C5':>10}  {'Δ%':>8}")
    print("─" * 42)
    bci = best_c2_idx["ci"]
    for ticker in TICKERS:
        b_m = np.mean(mapes[0][ticker]) if mapes[0][ticker] else float("nan")
        c_m = np.mean(mapes[bci][ticker]) if mapes[bci][ticker] else float("nan")
        d   = (c_m / b_m - 1.0) * 100.0 if np.isfinite(b_m) and np.isfinite(c_m) else float("nan")
        print(f"{ticker:<6}  {b_m:.5f}   {c_m:.5f}   {d:>+7.2f}%")

# ── графики ───────────────────────────────────────────────────────────────────

# Рис. A: AGG MAPE по конфигурациям (барчарт)
fig_a, ax_a = plt.subplots(figsize=(11, 4))
names  = [r["name"] for r in agg_results]
aggs   = [r["agg"] for r in agg_results]
colors = ["#78909c" if r["ci"] == 0
          else ("#43a047" if r["delta_pct"] < -0.5
                else ("#e53935" if r["delta_pct"] > 0.5
                      else "#ffa726"))
          for r in agg_results]
bars = ax_a.barh(names, aggs, color=colors, alpha=0.85, height=0.55)
ax_a.axvline(aggs[0], color="#546e7a", lw=1.5, ls="--", label=f"baseline {aggs[0]:.5f}")
for bar, r in zip(bars, agg_results):
    label = f"  {r['delta_pct']:+.2f}%" if r["ci"] > 0 else "  baseline"
    ax_a.text(bar.get_width() + 2e-5, bar.get_y() + bar.get_height() / 2,
              label, va="center", fontsize=9)
ax_a.set_xlabel("AGG MAPE")
ax_a.set_title("LWR на C2/C1: AGG MAPE по конфигурациям (8 тикеров 1d, N=200 origins)")
ax_a.legend(fontsize=9)
fig_a.tight_layout()
path_a = FIG_DIR / "55_c2_lwr_configs.png"
fig_a.savefig(path_a, dpi=150)
print(f"\nРис. A: {path_a}")

# Рис. B: per-ticker MAPE для топ-3 конфигураций + baseline
top3 = sorted(agg_results, key=lambda r: r["agg"])[:4]  # baseline + 3 лучших
fig_b, ax_b = plt.subplots(figsize=(11, 5))
x = np.arange(len(TICKERS))
w = 0.18
for i, r in enumerate(top3):
    ci = r["ci"]
    vals = [np.mean(mapes[ci][t]) if mapes[ci][t] else float("nan") for t in TICKERS]
    ax_b.bar(x + i * w, vals, w,
             label=r["name"],
             alpha=0.85,
             color=["#78909c", "#43a047", "#1976d2", "#ffa726"][i])
ax_b.set_xticks(x + w * 1.5)
ax_b.set_xticklabels(TICKERS)
ax_b.set_ylabel("MAPE (mean по origins)")
ax_b.set_title("Per-ticker MAPE: топ конфигурации vs baseline")
ax_b.legend(fontsize=8)
fig_b.tight_layout()
path_b = FIG_DIR / "55_c2_lwr_perticker.png"
fig_b.savefig(path_b, dpi=150)
print(f"Рис. B: {path_b}")

# Рис. C: MAPE vs horizon (h=1..15) для baseline и лучшей C2-C5
if best_c2_idx is not None:
    # для этого нужны per-horizon данные — у нас их нет в текущей структуре
    # делаем быстрый дополнительный прогон на SBER для визуализации
    ticker = "SBER"
    path   = DATA_DIR / ticker / "1d.json"
    with open(path) as f:
        c_ = json.load(f)
    close_ = np.array([float(c["close"]) for c in c_], dtype=np.float64)
    trend_ = logtrend_causal(close_)
    ratio_ = close_ / trend_
    N_     = len(ratio_)
    max_ok_ = N_ - 1 - HORIZON
    min_ok_ = max(300, N_ - 500)
    origs_ = np.linspace(min_ok_, max_ok_, 100, dtype=int)
    origs_ = np.unique(origs_)

    mape_h_base = np.zeros((len(origs_), HORIZON))
    mape_h_best = np.zeros((len(origs_), HORIZON))
    bci = best_c2_idx["ci"]

    for oi, ok in enumerate(origs_):
        dr_ = np.diff(ratio_[:ok + 1])
        comp_ = make_filter_bank(dr_)
        actual_ = ratio_[ok + 1: ok + 1 + HORIZON]
        if len(actual_) < HORIZON:
            continue
        r0_ = float(ratio_[ok])

        for cfg_i, cfg_i_obj in [(0, CONFIGS[0]), (bci, CONFIGS[bci])]:
            dhat_ = np.zeros(HORIZON)
            ok_ = True
            for ci_, p_ci_ in cfg_i_obj["comp_p"].items():
                d_ = forecast_lwr_comp(comp_[ci_], p_ci_, HORIZON)
                if np.any(np.isnan(d_)):
                    ok_ = False; break
                dhat_ += d_
            if not ok_:
                continue
            rh_ = r0_ + np.cumsum(dhat_)
            errs_ = np.abs(rh_ - actual_) / (np.abs(actual_) + 1e-10)
            if cfg_i == 0:
                mape_h_base[oi] = errs_
            else:
                mape_h_best[oi] = errs_

    mh_base_mean = np.nanmean(mape_h_base, axis=0)
    mh_best_mean = np.nanmean(mape_h_best, axis=0)

    fig_c, ax_c_ = plt.subplots(figsize=(9, 4))
    hs = np.arange(1, HORIZON + 1)
    ax_c_.plot(hs, mh_base_mean, "o-", color="#78909c", lw=2, ms=5,
               label=f"C3-C5 p=20 (baseline)")
    ax_c_.plot(hs, mh_best_mean, "s-", color="#43a047", lw=2, ms=5,
               label=best_c2_idx["name"])
    ax_c_.set_xlabel("h (шаг горизонта)")
    ax_c_.set_ylabel("MAPE")
    ax_c_.set_title(f"MAPE vs горизонт — SBER 1d (100 origins)")
    ax_c_.legend(fontsize=9)
    ax_c_.grid(alpha=0.3)
    fig_c.tight_layout()
    path_c = FIG_DIR / "55_c2_lwr_horizon.png"
    fig_c.savefig(path_c, dpi=150)
    print(f"Рис. C: {path_c}")

# ── итог ─────────────────────────────────────────────────────────────────────

print("\n── Итог ──────────────────────────────────────────────────────────────────")
best_overall = min(agg_results, key=lambda r: r["agg"])
print(f"Лучшая конфиг.: {best_overall['name']}")
print(f"AGG MAPE:        {best_overall['agg']:.5f}  ({best_overall['delta_pct']:+.2f}% vs baseline)")
pv = best_overall["pval"]
if np.isfinite(pv):
    sig = "значимо (p<0.05)" if pv < 0.05 else "НЕ значимо"
    print(f"t-test:          p={pv:.4f}  ({sig})")
print(f"\n[Справка] Скр.52 Damped AR C1+C2: −8.01% vs C3-C5 baseline")
