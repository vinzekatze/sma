"""
68 — Валидация: c2c5_k13 (xi=700) vs baseline.

Из скр.67: лучшая конфигурация — c2c5_k13 (k=13, xi=700, p_fit=13).
Проверяем на полном протоколе: 8 тикеров, 100 origins.

Методы:
  A  c2c5_k13  xi=700  p_fit=13  ← новый чемпион
  B  geom_k9   xi=700  p_fit=13  ← второй non-uniform
  C  uniform_p80  xi=243  p_fit=13  ← natural baseline (скр.65 phase 2)
  D  uniform_p80  xi=500  p_fit=13  ← оптимальный uniform (скр.67)

t-test paired A vs C и A vs D.
"""

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

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
N_ORIG  = 100
HORIZON = 20
WN      = 0.125
P_FIT   = 13
FILTER_ORDER = 4

METHODS = {
    "c2c5_k13 xi=700":    (
        [130, 103, 78, 52, 40, 26, 16, 12, 8, 5, 2, 1, 0], 700),
    "geom_k9 xi=700":     (
        [128, 64, 32, 16, 8, 4, 2, 1, 0], 700),
    "uniform_p80 xi=243": (
        list(range(79, -1, -1)), 243),
    "uniform_p80 xi=500": (
        list(range(79, -1, -1)), 500),
}
METHOD_NAMES = list(METHODS.keys())

_SOS_LP = butter(FILTER_ORDER, WN, btype="low", output="sos")

LP_WIN = 50  # окно для bias-коррекции LP (баров)

def _compute_tau():
    from scipy.signal import sos2tf, group_delay
    b, a = sos2tf(_SOS_LP)
    _, gd = group_delay((b, a), w=1, whole=False)
    return int(round(float(gd[0])))

TAU = _compute_tau()


# ── утилиты ────────────────────────────────────────────────────────────────────

def logtrend_causal(close):
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t); trend[:2] = close[:2]
    return trend


def lp_mape(att_pred, att_actual, ratio0):
    h = min(len(att_pred), len(att_actual))
    if h == 0 or not np.isfinite(att_pred[:h]).all():
        return np.nan
    r_pred = ratio0 + np.cumsum(att_pred[:h])
    r_act  = ratio0 + np.cumsum(att_actual[:h])
    return float(np.mean(np.abs(r_pred - r_act) / (np.abs(r_act) + 1e-10)))


def lp_bias_ratio_j(att_hist, ratio, logtrend, close, origin_k):
    """
    ratio_j с bias-коррекцией: LP price в origin_k выравнивается к реальным
    ценам через τ-сдвиг, затем переводится обратно в ratio.
    """
    ws    = max(0, origin_k - LP_WIN + 1)
    lp_r  = ratio[ws] + np.concatenate([[0.0], np.cumsum(att_hist[ws:origin_k])])
    lp_p  = lp_r * logtrend[ws: origin_k + 1]
    barr  = np.arange(ws, origin_k + 1) - TAU
    valid = (barr >= 0) & (barr < len(close))
    bias  = float(np.mean(close[barr[valid]] - lp_p[valid])) if valid.any() else 0.0
    return (lp_p[-1] + bias) / logtrend[origin_k]


OHLC_KEYS = ("close", "open", "high", "low", "hl_mid")

def price_mape_ohlc(att_pred, origin_k, ratio_j, logtrend, ohlc, horizon):
    """
    Price-MAPE прогноза против close / open / high / low / (H+L)/2.
    Все 5 значений — честный эталон в рублях без LP-фазовых артефактов.
    """
    h = min(len(att_pred), horizon)
    if h == 0 or not np.isfinite(att_pred[:h]).all():
        return {k: np.nan for k in OHLC_KEYS}
    lt  = logtrend[origin_k + 1: origin_k + 1 + h]
    n   = min(h, len(lt))
    pp  = (ratio_j + np.cumsum(att_pred[:n])) * lt
    out = {}
    for key in ("close", "open", "high", "low"):
        p = ohlc[key][origin_k + 1: origin_k + 1 + n]
        out[key] = float(np.mean(np.abs(pp - p) / (np.abs(p) + 1e-10)))
    hl = 0.5 * (ohlc["high"][origin_k + 1: origin_k + 1 + n]
              + ohlc["low"] [origin_k + 1: origin_k + 1 + n])
    out["hl_mid"] = float(np.mean(np.abs(pp - hl) / (np.abs(hl) + 1e-10)))
    return out


def load_att(ticker):
    import json
    data   = json.loads((DATA_DIR / ticker / "1d.json").read_text())
    close  = np.array([c["close"] for c in data], dtype=np.float64)
    open_  = np.array([c["open"]  for c in data], dtype=np.float64)
    high   = np.array([c["high"]  for c in data], dtype=np.float64)
    low    = np.array([c["low"]   for c in data], dtype=np.float64)
    trend  = logtrend_causal(close)
    ratio  = close / np.maximum(trend, 1e-10)
    dratio = np.diff(ratio, prepend=ratio[0])
    ohlc   = {"close": close, "open": open_, "high": high, "low": low}
    return sosfilt(_SOS_LP, dratio), ratio, trend, ohlc


def make_matrices(att, lags_desc, p_fit):
    max_lag = lags_desc[0]; lags_arr = np.array(lags_desc, dtype=np.int32)
    n = len(att); m = n - max_lag - 1; k = len(lags_desc)
    if m <= 0:
        return np.zeros((0, k)), np.zeros((0, p_fit)), np.zeros(0)
    t_arr    = np.arange(max_lag, n - 1, dtype=np.int32)
    X_search = np.column_stack([att[t_arr - lag] for lag in lags_arr])
    X_fit    = np.column_stack([att[t_arr - (p_fit - 1 - k_)] for k_ in range(p_fit)])
    return X_search, X_fit, att[t_arr + 1]


def forecast_lwr(att_hist, lags_desc, p_fit, horizon, xi):
    n = len(att_hist)
    X_search, X_fit, y = make_matrices(att_hist, lags_desc, p_fit)
    xi_eff = min(xi, len(X_search))
    if xi_eff < p_fit + 2:
        return np.full(horizon, np.nan)
    lags_arr = np.array(lags_desc, dtype=np.int32)
    buf = np.empty(n + horizon); buf[:n] = att_hist
    out = np.empty(horizon)
    for h in range(horizon):
        t      = n + h - 1
        vec_s  = buf[t - lags_arr]
        vec_f  = buf[t - p_fit + 1: t + 1]
        dists  = np.linalg.norm(X_search - vec_s, axis=1)
        nn     = np.argpartition(dists, xi_eff - 1)[:xi_eff]
        h_bw   = max(float(dists[nn].max()), 1e-10)
        X_nn   = X_fit[nn]; y_nn = y[nn]
        w      = np.exp(-0.5 * (np.linalg.norm(X_nn - vec_f, axis=1) / h_bw) ** 2)
        A      = np.hstack([np.ones((xi_eff, 1)), X_nn]); sw = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
        out[h] = float(c[0] + vec_f @ c[1:]); buf[t + 1] = out[h]
    return out


# ── загрузка данных ────────────────────────────────────────────────────────────

print(f"TAU={TAU} баров, LP_WIN={LP_WIN}")
print("Загрузка данных…", flush=True)
att_data, ratio_data, logtrend_data, ohlc_data, origins_data = {}, {}, {}, {}, {}

max_lag_global = max(lags[0] for lags, _ in METHODS.values())
xi_max = max(xi for _, xi in METHODS.values())

for ticker in TICKERS:
    att_full, ratio, trend, ohlc = load_att(ticker)
    att_data[ticker]      = att_full
    ratio_data[ticker]    = ratio
    logtrend_data[ticker] = trend
    ohlc_data[ticker]     = ohlc
    n_total   = len(att_full)
    min_start = max_lag_global + xi_max + HORIZON + 10
    end_k     = n_total - HORIZON - 1
    start_k   = max(min_start, end_k - N_ORIG * 5)
    cands     = list(range(start_k, end_k))
    step      = max(1, len(cands) // N_ORIG)
    origins_data[ticker] = cands[::step][:N_ORIG]
    print(f"  {ticker}: {len(origins_data[ticker])} origins, n={n_total}", flush=True)


# ── прогон ────────────────────────────────────────────────────────────────────

# mapes[method][ticker]  = list of LP-MAPE per origin
# pmapes[method][ticker] = list of price-MAPE dicts per origin
mapes  = {name: {t: [] for t in TICKERS} for name in METHOD_NAMES}
pmapes = {name: {t: [] for t in TICKERS} for name in METHOD_NAMES}

t0     = time.time()
n_done = 0
total  = len(TICKERS) * N_ORIG

print(f"\nПрогон: {len(METHOD_NAMES)} методов × {total} origins…", flush=True)

for ticker in TICKERS:
    att_full = att_data[ticker]; ratio = ratio_data[ticker]
    logtrend = logtrend_data[ticker]; ohlc = ohlc_data[ticker]
    origins  = origins_data[ticker]
    for origin_k in origins:
        att_hist   = att_full[:origin_k]
        att_actual = att_full[origin_k: origin_k + HORIZON]
        ratio0     = float(ratio[origin_k])
        rj         = lp_bias_ratio_j(att_hist, ratio, logtrend, ohlc["close"], origin_k)
        for name, (lags, xi) in METHODS.items():
            dhat = forecast_lwr(att_hist, lags, P_FIT, HORIZON, xi)
            m    = lp_mape(dhat, att_actual, ratio0)
            if np.isfinite(m):
                mapes[name][ticker].append(m)
            pmapes[name][ticker].append(
                price_mape_ohlc(dhat, origin_k, rj, logtrend, ohlc, HORIZON))
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{total}  ({time.time()-t0:.0f}s)", flush=True)

print(f"Завершено за {time.time()-t0:.1f}с", flush=True)


# ── агрегация ─────────────────────────────────────────────────────────────────

def agg(name):
    return np.nanmean([v for t in TICKERS for v in mapes[name][t]])

def flat(name):
    return [v for t in TICKERS for v in mapes[name][t]]

agg_vals = {name: agg(name) for name in METHOD_NAMES}
champion = "c2c5_k13 xi=700"
baseline = "uniform_p80 xi=243"

# t-test champion vs каждый другой
print("\n── AGG LP-MAPE (8 тикеров, 100 origins) ─────────────────────────────────")
print(f"{'Метод':<24}  {'AGG MAPE':>10}  {'Δ% vs c2c5':>11}  {'p-value':>9}")
print("─" * 62)
for name in METHOD_NAMES:
    delta = (agg_vals[champion] / agg_vals[name] - 1) * 100
    if name != champion:
        fa = flat(champion); fb = flat(name)
        n_min = min(len(fa), len(fb))
        _, pval = ttest_rel(fa[:n_min], fb[:n_min])
        pstr = f"{pval:.4f}"
    else:
        pstr = "—"
    marker = " ←" if name == champion else ""
    print(f"{name:<24}  {agg_vals[name]:>10.5f}  {delta:>+10.1f}%  {pstr:>9}{marker}")

# per-ticker
print(f"\n── Per-ticker ────────────────────────────────────────────────────────────")
print(f"{'Ticker':<6}", end="")
for name in METHOD_NAMES:
    short = name[:12]
    print(f"  {short:>12}", end="")
print(f"  {'Δ% A-C':>8}  {'Δ% A-D':>8}")
print("─" * (6 + 14 * len(METHOD_NAMES) + 20))
for t in TICKERS:
    print(f"{t:<6}", end="")
    vals = {}
    for name in METHOD_NAMES:
        v = np.nanmean(mapes[name][t]) if mapes[name][t] else np.nan
        vals[name] = v
        print(f"  {v:>12.5f}", end="")
    dc = (vals[baseline]                    / vals[champion] - 1) * 100
    dd = (vals["uniform_p80 xi=500"]        / vals[champion] - 1) * 100
    print(f"  {dc:>+7.1f}%  {dd:>+7.1f}%")

# сводка
print(f"\n── Итог ─────────────────────────────────────────────────────────────────")
print(f"Чемпион:   {champion}")
print(f"  AGG MAPE = {agg_vals[champion]:.5f}")
dac = (agg_vals[baseline]             / agg_vals[champion] - 1) * 100
dad = (agg_vals['uniform_p80 xi=500'] / agg_vals[champion] - 1) * 100
print(f"  vs uniform_p80 xi=243 (baseline): {dac:>+.1f}%")
print(f"  vs uniform_p80 xi=500 (opt unif): {dad:>+.1f}%")

fa = flat(champion); fb = flat(baseline)
n_min = min(len(fa), len(fb))
_, p_ac = ttest_rel(fa[:n_min], fb[:n_min])
fa = flat(champion); fb = flat("uniform_p80 xi=500")
n_min = min(len(fa), len(fb))
_, p_ad = ttest_rel(fa[:n_min], fb[:n_min])
print(f"  t-test vs baseline:    p={p_ac:.6f}")
print(f"  t-test vs opt uniform: p={p_ad:.6f}")


# ── price-MAPE (против реальных OHLC цен) ─────────────────────────────────────

def agg_pm(name, key):
    return np.nanmean([m[key] for t in TICKERS for m in pmapes[name][t]])

print(f"\n── Price-MAPE vs OHLC  (τ={TAU}б, LP_WIN={LP_WIN}, bias-corrected) ─────────────")
hdr = f"{'Метод':<24}"
for k in OHLC_KEYS:
    hdr += f"  {k:>8}"
print(hdr + "  (чемпион по close)")
print("─" * (24 + 11 * len(OHLC_KEYS)))

best_close = min(METHOD_NAMES, key=lambda n: agg_pm(n, "close"))
for name in METHOD_NAMES:
    row = f"{name:<24}"
    for k in OHLC_KEYS:
        row += f"  {agg_pm(name, k):>8.5f}"
    marker = "  ←" if name == best_close else ""
    print(row + marker)

# разброс OHLC: насколько min/max отличается от close
print(f"\n── Spread: min/max price-MAPE по OHLC (лучший чемпион: {best_close}) ──────────")
for name in METHOD_NAMES:
    vals = {k: agg_pm(name, k) for k in ("close", "open", "high", "low")}
    mn, mx = min(vals.values()), max(vals.values())
    print(f"  {name:<24}  min={mn:.5f}  max={mx:.5f}  spread={mx/mn-1:+.1%}")


# ── рисунок ───────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# рис. A: per-ticker bar chart
x = np.arange(len(TICKERS)); w = 0.2
colors = ["#E53935", "#43A047", "#1976D2", "#7B1FA2"]
for i, name in enumerate(METHOD_NAMES):
    vals = [np.nanmean(mapes[name][t]) for t in TICKERS]
    ax1.bar(x + i * w, vals, w, label=name[:18], color=colors[i], alpha=0.85)
ax1.set_xticks(x + w * 1.5); ax1.set_xticklabels(TICKERS)
ax1.set_ylabel("LP-MAPE"); ax1.set_title("Per-ticker LP-MAPE")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3, axis="y")

# рис. B: AGG comparison bar
names_short = [n[:16] for n in METHOD_NAMES]
vals_agg = [agg_vals[n] for n in METHOD_NAMES]
bars = ax2.bar(names_short, vals_agg, color=colors, alpha=0.85, edgecolor="white")
for bar, val, name in zip(bars, vals_agg, METHOD_NAMES):
    d = (agg_vals[champion] / val - 1) * 100
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
             f"{d:+.1f}%", ha="center", fontsize=8)
ax2.set_ylabel("AGG LP-MAPE")
ax2.set_title(f"AGG LP-MAPE (8 тикеров, {N_ORIG} origins)")
ax2.tick_params(axis="x", rotation=15); ax2.grid(True, alpha=0.3, axis="y")

fig.suptitle(f"Валидация c2c5_k13 vs uniform  (p_fit={P_FIT})", fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "68_validation.png", dpi=150)
plt.close(fig)

print(f"\nРис.: {FIG_DIR}/68_validation.png")
print(f"\nСкрипт 68 завершён за {time.time()-t0:.1f}с")
