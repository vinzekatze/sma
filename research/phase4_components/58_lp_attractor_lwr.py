"""
58 — Единый аттрактор: LWR на lowpass C2-C5 vs per-component baseline.

Гипотеза (скр.54 + наблюдения в прототипе): C0+C1 — шум, C2-C5 — единый
детерминированный аттрактор. Разбивка на подкомпоненты (C3/C4/C5 sep) ломает
фазовое пространство аттрактора. LWR на объединённом lowpass(Wn=0.125) сохраняет
аттракторную структуру → должен находить правильных соседей.

Фильтрбанк (3 полосы):
  attractor = lowpass(Wn=0.125)          период ≥ 8б  (C2+C3+C4+C5)
  C1        = bandpass(0.125..0.25)      период 4-8б  → damped AR γ=0.5
  C0        = highpass(Wn=0.25)          период <4б   → ноль (expectation)

Конфигурации:
  0.    baseline: per-comp LWR C3-C5 p=20 + damped AR C2(γ=0.8) + C1(γ=0.5)
  1-6.  LP-causal  p∈{4,5,8,12,20,30}: LWR на causal attractor
  7-11. LP-filtfilt p∈{4,5,8,20,30}:   то же, нулевая фаза (верхняя граница)

Разрыв между LP-causal и LP-filtfilt = стоимость τ≈7-баровой задержки.
Разрыв между LP-causal-best и baseline = выигрыш от сохранения аттрактора.

Протокол: 8 тикеров 1d, N_ORIG=200, horizon=15, logtrend.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfiltfilt
from scipy.stats import ttest_rel

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
FIG_DIR  = ROOT / "research/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
N_ORIG       = 200
HORIZON      = 15
FILTER_ORDER = 4

# новый фильтрбанк (3 полосы)
ATTRACTOR_WN = 0.125   # граница C1/C2 — аттрактор ниже
NOISE_WN     = 0.25    # граница C0/C1

# старый фильтрбанк (baseline)
CUTOFFS_OLD  = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
P_SLOW       = 20
XI_SLOW      = 3 * (P_SLOW + 1)

# damped AR параметры
GAMMA_C1     = 0.5
GAMMA_C2     = 0.8     # только в baseline
P_AR_MAX     = 20

# p-свип для LP-конфигураций
P_CAUSAL   = [4, 5, 8, 12, 20, 30]
P_FILTFILT = [4, 5, 8, 20, 30]

CONFIGS = [{"name": "baseline", "mode": "baseline", "p": None}]
# вариант A: LWR на аттракторе + damped AR C1 (γ=0.5)
for p in P_CAUSAL:
    CONFIGS.append({"name": f"LP-causal+C1  p={p:2d}", "mode": "causal",      "p": p, "c1": True})
# вариант B: LWR на аттракторе только, C1+C0 = ноль (чистый аттракторный прогноз)
for p in P_CAUSAL:
    CONFIGS.append({"name": f"LP-causal-noC1 p={p:2d}", "mode": "causal_noc1", "p": p, "c1": False})
# zero-phase reference: с C1 и без
for p in P_FILTFILT:
    CONFIGS.append({"name": f"LP-filtfilt+C1 p={p:2d}", "mode": "filtfilt",    "p": p, "c1": True})
for p in [5, 20, 30]:
    CONFIGS.append({"name": f"LP-filtfilt-noC1 p={p:2d}", "mode": "filtfilt_noc1", "p": p, "c1": False})

# ── фильтры (создаём один раз) ─────────────────────────────────────────────────

SOS_ATT   = butter(FILTER_ORDER, ATTRACTOR_WN, btype="low", output="sos")
SOS_NOISE = butter(FILTER_ORDER, NOISE_WN,     btype="low", output="sos")

# старый фильтрбанк для baseline
SOS_OLD = [butter(FILTER_ORDER, fc, btype="low", output="sos") for fc in CUTOFFS_OLD]

# ── вспомогательные функции ────────────────────────────────────────────────────

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


def make_old_fb(series: np.ndarray) -> np.ndarray:
    """Старый каузальный фильтрбанк → (6, N) компонент C0..C5."""
    comps, rem = [], series.copy()
    for sos in SOS_OLD:
        low = sosfilt(sos, rem)
        comps.append(rem - low)
        rem = low
    comps.append(rem)
    return np.array(comps)


def split_3band(series: np.ndarray, zero_phase: bool = False):
    """
    3-полосное разложение:
      attractor = lowpass(Wn=0.125)   — causal или filtfilt
      c1        = lowpass(Wn=0.25) − attractor   — всегда той же фазности что attractor
      (c0       = series − lowpass(Wn=0.25)  — не нужен для прогноза)
    """
    fn = sosfiltfilt if zero_phase else sosfilt
    att      = fn(SOS_ATT,   series)
    lp_noise = fn(SOS_NOISE, series)
    c1       = lp_noise - att
    return att, c1


def fit_ar_bic(series: np.ndarray, p_max: int = P_AR_MAX):
    n = len(series)
    best_bic, best_p, best_c = np.inf, 1, np.zeros(2)
    for p in range(1, min(p_max + 1, (n - 1) // 4)):
        X = np.column_stack([series[p - 1 - k: n - 1 - k] for k in range(p)])
        X = np.hstack([np.ones((n - p, 1)), X])
        c, _, _, _ = np.linalg.lstsq(X, series[p:], rcond=None)
        ssr  = np.sum((series[p:] - X @ c) ** 2)
        bic  = (n - p) * np.log(ssr / (n - p) + 1e-12) + (p + 1) * np.log(n - p)
        if bic < best_bic:
            best_bic, best_p, best_c = bic, p, c
    return best_p, best_c


def forecast_damped_ar(series: np.ndarray, horizon: int, gamma: float) -> np.ndarray:
    if len(series) < 4:
        return np.zeros(horizon)
    p, c = fit_ar_bic(series)
    buf  = list(series[-p:])
    raw  = np.empty(horizon)
    for h in range(horizon):
        val     = c[0] + sum(c[1 + k] * buf[-(k + 1)] for k in range(p))
        raw[h]  = val
        buf.append(val)
    return raw * (gamma ** np.arange(1, horizon + 1))


def forecast_lwr(series: np.ndarray, p: int, horizon: int) -> np.ndarray:
    """Итерационный LWR. ξ = 3*(p+1)."""
    xi = 3 * (p + 1)
    n  = len(series)
    if n < xi + p + 2:
        return np.zeros(horizon)
    X   = np.array([series[i: i + p] for i in range(n - p)])
    y   = series[p:]
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
        vec    = np.roll(vec, -1)
        vec[-1] = val
    return out


# ── прогноз одного origin ──────────────────────────────────────────────────────

def forecast_one(dratio_hist: np.ndarray, cfg: dict) -> np.ndarray:
    mode = cfg["mode"]
    p    = cfg["p"]

    if mode == "baseline":
        comp = make_old_fb(dratio_hist)
        dhat = np.zeros(HORIZON)
        for ci in [3, 4, 5]:
            dhat += forecast_lwr(comp[ci], P_SLOW, HORIZON)
        dhat += forecast_damped_ar(comp[2], HORIZON, GAMMA_C2)
        dhat += forecast_damped_ar(comp[1], HORIZON, GAMMA_C1)
        return dhat

    zero_phase = mode in ("filtfilt", "filtfilt_noc1")
    att, c1   = split_3band(dratio_hist, zero_phase=zero_phase)
    dhat       = forecast_lwr(att, p, HORIZON)
    if cfg.get("c1", True):
        dhat += forecast_damped_ar(c1, HORIZON, GAMMA_C1)
    return dhat


# ── walk-forward ───────────────────────────────────────────────────────────────

N_CONFIGS = len(CONFIGS)
mapes: list[dict[str, list[float]]] = [{t: [] for t in TICKERS} for _ in CONFIGS]

# trajectory-примеры: первые 4 origin SBER
TRAJ_TICKER = "SBER"
N_TRAJ      = 4
traj_data: list[dict] = []

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

    max_ok = N - 1 - HORIZON
    min_ok = max(300, N - 700)
    if max_ok <= min_ok:
        continue
    origins = np.unique(np.linspace(min_ok, max_ok, N_ORIG, dtype=int))

    traj_count = 0

    for origin_k in origins:
        dratio_hist  = np.diff(ratio[:origin_k + 1])
        actual_ratio = ratio[origin_k + 1: origin_k + 1 + HORIZON]
        if len(actual_ratio) < HORIZON:
            continue
        ratio0 = float(ratio[origin_k])

        record = (ticker == TRAJ_TICKER and traj_count < N_TRAJ)
        if record:
            entry = {"actual": actual_ratio.copy(), "ratio0": ratio0,
                     "origin_k": origin_k, "forecasts": {}}

        for ci, cfg in enumerate(CONFIGS):
            dhat = forecast_one(dratio_hist, cfg)
            if np.any(np.isnan(dhat)) or np.any(np.abs(dhat) > 1e6):
                if record:
                    entry["forecasts"][cfg["name"]] = None
                continue
            ratio_hat = ratio0 + np.cumsum(dhat)
            mape = float(np.mean(
                np.abs(ratio_hat - actual_ratio) / (np.abs(actual_ratio) + 1e-10)
            ))
            if np.isfinite(mape):
                mapes[ci][ticker].append(mape)
            if record:
                entry["forecasts"][cfg["name"]] = ratio_hat.copy()

        if record:
            traj_data.append(entry)
            traj_count += 1

        n_done += 1
        if n_done % 200 == 0:
            elapsed = time.time() - t0
            total   = len(TICKERS) * len(origins)
            print(f"  {n_done}/{total}  ({elapsed:.0f}s)")

print(f"\nГотово за {time.time() - t0:.1f}с")

# ── агрегация ──────────────────────────────────────────────────────────────────

baseline_flat = [v for t in TICKERS for v in mapes[0][t]]
agg_base      = float(np.mean(baseline_flat))
results: list[dict] = []

print("\n── AGG MAPE ─────────────────────────────────────────────────────────────────")
print(f"{'#':<2}  {'Конфигурация':<26}  {'AGG MAPE':>9}  {'Δ% baseline':>12}  p-value")
print("─" * 72)

for ci, cfg in enumerate(CONFIGS):
    flat = [v for t in TICKERS for v in mapes[ci][t]]
    if not flat:
        continue
    agg  = float(np.mean(flat))
    d    = (agg / agg_base - 1.0) * 100.0 if ci > 0 else 0.0
    if ci > 0 and len(flat) == len(baseline_flat):
        _, pval = ttest_rel(flat, baseline_flat)
    else:
        pval = float("nan")
    sign = "✅" if d < -0.5 else ("❌" if d > 0.5 else "~")
    p_str = f"p={pval:.4f}" if np.isfinite(pval) else "—"
    print(f"{ci:<2}  {cfg['name']:<26}  {agg:.5f}   {d:>+8.2f}%  {sign}  {p_str}")
    results.append({"ci": ci, "cfg": cfg, "agg": agg, "d": d, "pval": pval, "flat": flat})

# лучшие конфигурации
best_causal   = min((r for r in results if r["cfg"]["mode"] == "causal"),
                    key=lambda r: r["agg"], default=None)
best_filtfilt = min((r for r in results if r["cfg"]["mode"] == "filtfilt"),
                    key=lambda r: r["agg"], default=None)

print(f"\nЛучшая LP-causal:    {best_causal['cfg']['name'] if best_causal else '—'}")
print(f"Лучшая LP-filtfilt:  {best_filtfilt['cfg']['name'] if best_filtfilt else '—'}")

if best_causal and best_filtfilt:
    gap = (best_causal["agg"] / best_filtfilt["agg"] - 1.0) * 100.0
    print(f"Цена задержки τ≈7б:  {gap:+.2f}% (causal vs filtfilt при лучшем p)")

# per-ticker для лучшей causal
if best_causal:
    bci = best_causal["ci"]
    print(f"\n── Per-ticker: {best_causal['cfg']['name'].strip()} vs baseline ─────────")
    print(f"{'Ticker':<6}  {'baseline':>9}  {'LP-causal':>9}  {'Δ%':>8}")
    print("─" * 38)
    for ticker in TICKERS:
        bm = np.mean(mapes[0][ticker])     if mapes[0][ticker]   else float("nan")
        cm = np.mean(mapes[bci][ticker])   if mapes[bci][ticker] else float("nan")
        d  = (cm / bm - 1.0) * 100.0      if np.isfinite(bm) and np.isfinite(cm) else float("nan")
        print(f"{ticker:<6}  {bm:.5f}   {cm:.5f}   {d:>+7.2f}%")

# ── Рис. A: barh AGG MAPE всех конфигураций ───────────────────────────────────

fig_a, ax_a = plt.subplots(figsize=(12, 7))
names  = [r["cfg"]["name"] for r in results]
aggs   = [r["agg"]         for r in results]
colors = []
for r in results:
    if r["ci"] == 0:
        colors.append("#78909c")
    elif r["cfg"]["mode"] == "filtfilt":
        colors.append("#1e88e5" if r["d"] < -0.5 else ("#e53935" if r["d"] > 0.5 else "#ffa726"))
    else:
        colors.append("#43a047" if r["d"] < -0.5 else ("#e53935" if r["d"] > 0.5 else "#ffa726"))

bars = ax_a.barh(names, aggs, color=colors, alpha=0.85, height=0.6)
ax_a.axvline(agg_base, color="#546e7a", lw=1.5, ls="--",
             label=f"baseline {agg_base:.5f}")
for bar, r in zip(bars, results):
    label = f"  {r['d']:+.2f}%" if r["ci"] > 0 else "  baseline"
    ax_a.text(bar.get_width() + 5e-5,
              bar.get_y() + bar.get_height() / 2,
              label, va="center", fontsize=8)
ax_a.set_xlabel("AGG MAPE")
ax_a.set_title("58: LP-аттрактор LWR(Wn=0.125) vs baseline\n"
               "зелёный = causal, синий = filtfilt (нулевая фаза); 8 тикеров 1d N=200")
ax_a.legend(fontsize=9)
fig_a.tight_layout()
fig_a.savefig(FIG_DIR / "58_mape_all.png", dpi=150)
print(f"\nРис. A: {FIG_DIR / '58_mape_all.png'}")

# ── Рис. B: p-кривые causal vs filtfilt ───────────────────────────────────────

causal_res   = [r for r in results if r["cfg"]["mode"] == "causal"]
filtfilt_res = [r for r in results if r["cfg"]["mode"] == "filtfilt"]
p_c   = [r["cfg"]["p"] for r in causal_res]
agg_c = [r["agg"]      for r in causal_res]
p_ff  = [r["cfg"]["p"] for r in filtfilt_res]
agg_ff = [r["agg"]     for r in filtfilt_res]

fig_b, ax_b = plt.subplots(figsize=(9, 4))
ax_b.axhline(agg_base, ls="--", color="#78909c", lw=1.5,
             label=f"baseline (per-comp C3-C5)  {agg_base:.5f}")
ax_b.plot(p_c,  agg_c,  "o-",  color="#43a047", lw=2, ms=7, label="LP-causal (τ≈7б)")
ax_b.plot(p_ff, agg_ff, "s--", color="#1e88e5", lw=2, ms=7, label="LP-filtfilt (τ=0, эталон)")
ax_b.set_xlabel("p (размерность вложения)")
ax_b.set_ylabel("AGG MAPE")
ax_b.set_title("58: AGG MAPE vs p  —  causal vs zero-phase\n"
               "разрыв = цена задержки фильтра")
ax_b.legend(fontsize=9)
ax_b.grid(alpha=0.3)
fig_b.tight_layout()
fig_b.savefig(FIG_DIR / "58_p_curve.png", dpi=150)
print(f"Рис. B: {FIG_DIR / '58_p_curve.png'}")

# ── Рис. C: траектории прогноза (SBER) ────────────────────────────────────────

if traj_data and best_causal:
    n_plots = min(N_TRAJ, len(traj_data))
    ncols   = 2
    nrows   = (n_plots + 1) // 2
    fig_c, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    show = {
        "baseline":                 ("#e53935", "--",  "baseline (per-comp)"),
        best_causal["cfg"]["name"]: ("#43a047", "-",   best_causal["cfg"]["name"].strip()),
    }
    if best_filtfilt:
        show[best_filtfilt["cfg"]["name"]] = ("#1e88e5", "-.", best_filtfilt["cfg"]["name"].strip())

    x_fc = np.arange(1, HORIZON + 1)

    for idx, entry in enumerate(traj_data[:n_plots]):
        ax = axes[idx]
        ratio0 = entry["ratio0"]
        actual = entry["actual"]

        ax.plot([0], [ratio0], "ko", ms=5, zorder=5, label="_")
        ax.plot(x_fc, actual, "k-", lw=2.5, label="actual", zorder=4)

        for cfg_name, (color, ls, label) in show.items():
            fc = entry["forecasts"].get(cfg_name)
            if fc is not None:
                ax.plot(x_fc, fc, ls, color=color, lw=1.8, alpha=0.9, label=label)

        ax.axvline(0.5, color="#aaa", lw=0.8, ls=":")
        ax.set_title(f"{TRAJ_TICKER}  origin #{idx + 1}  (k={entry['origin_k']})",
                     fontsize=10)
        ax.set_xlabel("горизонт (бары)")
        ax.set_ylabel("ratio")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    # скрыть лишние оси
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig_c.suptitle(f"58: Траектории прогноза — {TRAJ_TICKER} (ratio)", fontsize=12)
    fig_c.tight_layout()
    fig_c.savefig(FIG_DIR / "58_trajectories.png", dpi=150)
    print(f"Рис. C: {FIG_DIR / '58_trajectories.png'}")

print("\nСкрипт 58 завершён.")
