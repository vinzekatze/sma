"""
67 — Кривая насыщения xi для non-uniform задержек.

Из скр.66: c2c5_k13 (k=13) и geom_k9 (k=9) при xi=453 дают MAPE ≤ uniform_p80.
Гипотеза: увеличение xi продолжает улучшать non-uniform быстрее, чем uniform
(т.к. поиск в меньшем k-пространстве — менее «прокляты» размерностью).

Sweep xi: 42, 100, 200, 300, 400, 500, 700, 1000.
Lag sets: c2c5_k13, geom_k9, uniform_p80 (reference).

Протокол: SBER, LKOH, CHMF, MRKP, N_ORIG=50, p_fit=13, horizon=20.
"""

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
FIG_DIR  = ROOT / "research/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS  = ["SBER", "LKOH", "CHMF", "MRKP"]
N_ORIG   = 50
HORIZON  = 20
WN       = 0.125
P_FIT    = 13
FILTER_ORDER = 4

XI_SWEEP = [42, 100, 200, 300, 400, 500, 700, 1000]

LAG_SETS = {
    "uniform_p80": list(range(79, -1, -1)),
    "c2c5_k13":    [130, 103, 78, 52, 40, 26, 16, 12, 8, 5, 2, 1, 0],
    "geom_k9":     [128, 64, 32, 16, 8, 4, 2, 1, 0],
}

_SOS_LP = butter(FILTER_ORDER, WN, btype="low", output="sos")


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


def load_att(ticker):
    import json
    data   = json.loads((DATA_DIR / ticker / "1d.json").read_text())
    close  = np.array([c["close"] for c in data], dtype=np.float64)
    trend  = logtrend_causal(close)
    ratio  = close / np.maximum(trend, 1e-10)
    dratio = np.diff(ratio, prepend=ratio[0])
    return sosfilt(_SOS_LP, dratio), ratio


def make_matrices(att, lags_desc, p_fit):
    max_lag  = lags_desc[0]
    lags_arr = np.array(lags_desc, dtype=np.int32)
    n = len(att); m = n - max_lag - 1; k = len(lags_desc)
    if m <= 0:
        return np.zeros((0, k)), np.zeros((0, p_fit)), np.zeros(0)
    t_arr    = np.arange(max_lag, n - 1, dtype=np.int32)
    X_search = np.column_stack([att[t_arr - lag] for lag in lags_arr])
    X_fit    = np.column_stack([att[t_arr - (p_fit - 1 - k_)] for k_ in range(p_fit)])
    return X_search, X_fit, att[t_arr + 1]


def forecast_lwr(att_hist, lags_desc, p_fit, horizon, xi):
    max_lag = lags_desc[0]; n = len(att_hist)
    X_search, X_fit, y = make_matrices(att_hist, lags_desc, p_fit)
    xi_eff = min(xi, len(X_search))
    if xi_eff < p_fit + 2:
        return np.full(horizon, np.nan)
    lags_arr = np.array(lags_desc, dtype=np.int32)
    buf = np.empty(n + horizon); buf[:n] = att_hist
    out = np.empty(horizon)
    for h in range(horizon):
        t     = n + h - 1
        vec_s = buf[t - lags_arr]
        vec_f = buf[t - p_fit + 1: t + 1]
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

print("Загрузка данных…", flush=True)
att_data, ratio_data, origins_data = {}, {}, {}
max_lag_global = max(ls[0] for ls in LAG_SETS.values())
xi_max = max(XI_SWEEP)

for ticker in TICKERS:
    att_full, ratio = load_att(ticker)
    att_data[ticker] = att_full; ratio_data[ticker] = ratio
    n_total   = len(att_full)
    min_start = max_lag_global + xi_max + HORIZON + 10
    end_k     = n_total - HORIZON - 1
    start_k   = max(min_start, end_k - N_ORIG * 5)
    cands     = list(range(start_k, end_k))
    step      = max(1, len(cands) // N_ORIG)
    origins_data[ticker] = cands[::step][:N_ORIG]
    print(f"  {ticker}: {len(origins_data[ticker])} origins, n={n_total}", flush=True)


# ── прогон ────────────────────────────────────────────────────────────────────

lag_names = list(LAG_SETS.keys())

# mapes[lag_name][xi][ticker]
mapes = {
    name: {xi: {t: [] for t in TICKERS} for xi in XI_SWEEP}
    for name in lag_names
}

t0     = time.time()
n_done = 0
total  = len(TICKERS) * N_ORIG

print(f"\nПрогон: {len(lag_names)} lag_sets × {len(XI_SWEEP)} xi, {total} origins…", flush=True)

for ticker in TICKERS:
    att_full = att_data[ticker]; ratio = ratio_data[ticker]
    origins  = origins_data[ticker]
    for origin_k in origins:
        att_hist   = att_full[:origin_k]
        att_actual = att_full[origin_k: origin_k + HORIZON]
        ratio0     = float(ratio[origin_k])
        for name, lags in LAG_SETS.items():
            for xi in XI_SWEEP:
                dhat = forecast_lwr(att_hist, lags, P_FIT, HORIZON, xi)
                m    = lp_mape(dhat, att_actual, ratio0)
                if np.isfinite(m):
                    mapes[name][xi][ticker].append(m)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{total}  ({time.time()-t0:.0f}s)", flush=True)

print(f"Завершено за {time.time()-t0:.1f}с", flush=True)


# ── вывод ──────────────────────────────────────────────────────────────────────

def agg(name, xi):
    return np.nanmean([v for t in TICKERS for v in mapes[name][xi][t]])


agg_curves = {name: [agg(name, xi) for xi in XI_SWEEP] for name in lag_names}

print("\n── AGG LP-MAPE vs xi ─────────────────────────────────────────────────────")
hdr = f"{'xi':>6}"
for name in lag_names:
    hdr += f"  {name:>14}"
print(hdr + "   (Δ% c2c5 vs unif @ same xi)")
print("─" * 72)

for i, xi in enumerate(XI_SWEEP):
    row = f"{xi:>6}"
    for name in lag_names:
        row += f"  {agg_curves[name][i]:>14.5f}"
    m_c = agg_curves["c2c5_k13"][i]
    m_u = agg_curves["uniform_p80"][i]
    delta = (m_u / m_c - 1) * 100
    # отметим натуральный xi для каждого набора
    markers = []
    if xi == max(3 * (len(LAG_SETS["uniform_p80"]) + 1), 3*(P_FIT+1)):
        markers.append("unif*")
    if xi == max(3 * (len(LAG_SETS["c2c5_k13"]) + 1), 3*(P_FIT+1)):
        markers.append("c2c5*")
    if xi == max(3 * (len(LAG_SETS["geom_k9"]) + 1), 3*(P_FIT+1)):
        markers.append("geom*")
    tag = " [" + ", ".join(markers) + "]" if markers else ""
    print(f"{row}  {delta:>+7.1f}%{tag}")

print("\n(* — натуральный xi для данного набора)")

# минимальные значения
print("\n── Минимум MAPE по каждому набору ───────────────────────────────────────")
for name in lag_names:
    arr = agg_curves[name]
    best_i = int(np.nanargmin(arr))
    print(f"  {name:<20}  best xi={XI_SWEEP[best_i]:>5}  MAPE={arr[best_i]:.5f}")

# скорость убывания: выигрыш c2c5 vs uniform при разных xi
print("\n── c2c5_k13 vs uniform_p80 по xi ─────────────────────────────────────────")
print(f"{'xi':>6}  {'unif_p80':>10}  {'c2c5_k13':>10}  {'geom_k9':>9}  {'Δ% c2c5':>9}  {'Δ% geom':>9}")
print("─" * 62)
for i, xi in enumerate(XI_SWEEP):
    m_u = agg_curves["uniform_p80"][i]
    m_c = agg_curves["c2c5_k13"][i]
    m_g = agg_curves["geom_k9"][i]
    dc  = (m_u / m_c - 1) * 100
    dg  = (m_u / m_g - 1) * 100
    print(f"{xi:>6}  {m_u:>10.5f}  {m_c:>10.5f}  {m_g:>9.5f}  {dc:>+8.1f}%  {dg:>+8.1f}%")


# ── рисунок ────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = {"uniform_p80": "#1976D2", "c2c5_k13": "#E53935", "geom_k9": "#43A047"}
styles = {"uniform_p80": "-s", "c2c5_k13": "-o", "geom_k9": "-^"}
labels = {
    "uniform_p80": "uniform_p80  (k=80)",
    "c2c5_k13":    "c2c5_k13     (k=13)",
    "geom_k9":     "geom_k9      (k=9)",
}

# рис. A: MAPE vs xi
for name in lag_names:
    ax1.plot(XI_SWEEP, agg_curves[name], styles[name],
             color=colors[name], lw=2, ms=7, label=labels[name])
# отметим натуральный xi
nat_xi = {
    "uniform_p80": max(3*(len(LAG_SETS["uniform_p80"])+1), 3*(P_FIT+1)),
    "c2c5_k13":    max(3*(len(LAG_SETS["c2c5_k13"])+1),   3*(P_FIT+1)),
    "geom_k9":     max(3*(len(LAG_SETS["geom_k9"])+1),     3*(P_FIT+1)),
}
for name, nx in nat_xi.items():
    if nx in XI_SWEEP:
        idx = XI_SWEEP.index(nx)
        ax1.axvline(nx, color=colors[name], lw=1, ls=":", alpha=0.5)

ax1.set_xlabel("xi (число соседей)")
ax1.set_ylabel("AGG LP-MAPE")
ax1.set_title(f"Кривая насыщения xi\n({len(TICKERS)} тикера, {N_ORIG} origins, p_fit={P_FIT})")
ax1.legend(); ax1.grid(True, alpha=0.3)

# рис. B: выигрыш Δ% non-uniform vs uniform при каждом xi
for name in ["c2c5_k13", "geom_k9"]:
    deltas = [(agg_curves["uniform_p80"][i] / agg_curves[name][i] - 1) * 100
              for i in range(len(XI_SWEEP))]
    ax2.plot(XI_SWEEP, deltas, styles[name], color=colors[name],
             lw=2, ms=7, label=labels[name])
ax2.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
ax2.set_xlabel("xi (число соседей)")
ax2.set_ylabel("Δ% vs uniform_p80 (+ = non-uniform лучше)")
ax2.set_title("Выигрыш non-uniform над uniform_p80")
ax2.legend(); ax2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "67_xi_saturation.png", dpi=150)
plt.close(fig)

print(f"\nРис.: {FIG_DIR}/67_xi_saturation.png")
print(f"\nСкрипт 67 завершён за {time.time()-t0:.1f}с")
