"""
66 — Неравномерные задержки при фиксированном xi.

Скрипт 65 показал, что non-uniform лаги хуже uniform_p80.
Гипотеза: это конфаундинг — у non-uniform наборов xi=42-57 соседей
против xi=243 у uniform_p80. Меньше соседей → менее стабильный LWR.

Протокол: те же lag sets, но xi фиксируется. Три режима xi:
  xi_natural  — скр.65 (max(3*(k+1), 3*(p_fit+1)))
  xi_243      — как uniform_p80 (3*(80+1))
  xi_453      — как uniform_p150 (3*(150+1))

Тикеры: SBER, LKOH, CHMF, MRKP. N_ORIG=50, p_fit=13, horizon=20.
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

TICKERS = ["SBER", "LKOH", "CHMF", "MRKP"]
N_ORIG  = 50
HORIZON = 20
WN      = 0.125
P_FIT   = 13
FILTER_ORDER = 4

XI_MODES = {
    "natural": None,   # max(3*(k+1), 3*(p_fit+1))
    "xi_243":  243,    # = 3*(80+1), как uniform_p80
    "xi_453":  453,    # = 3*(150+1), как uniform_p150
}

LAG_SETS = {
    "uniform_p80":    list(range(79, -1, -1)),
    "uniform_p150":   list(range(149, -1, -1)),
    "geom_k9":        [128, 64, 32, 16, 8, 4, 2, 1, 0],
    "c2c5_k13":       [130, 103, 78, 52, 40, 26, 16, 12, 8, 5, 2, 1, 0],
    "quarter_k12":    [120, 90, 60, 50, 38, 25, 19, 12, 8, 4, 2, 0],
    "dense_near_k18": [130, 103, 80, 65, 52, 40, 32, 26, 20, 16, 12, 10, 8, 6, 4, 2, 1, 0],
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


# ── ядро прогноза ──────────────────────────────────────────────────────────────

def make_matrices(att, lags_desc, p_fit):
    max_lag  = lags_desc[0]
    lags_arr = np.array(lags_desc, dtype=np.int32)
    n = len(att)
    m = n - max_lag - 1
    k = len(lags_desc)
    if m <= 0:
        return np.zeros((0, k)), np.zeros((0, p_fit)), np.zeros(0)
    t_arr    = np.arange(max_lag, n - 1, dtype=np.int32)
    X_search = np.column_stack([att[t_arr - lag] for lag in lags_arr])
    X_fit    = np.column_stack([att[t_arr - (p_fit - 1 - k_)] for k_ in range(p_fit)])
    y        = att[t_arr + 1]
    return X_search, X_fit, y


def forecast_lwr(att_hist, lags_desc, p_fit, horizon, xi_fixed=None):
    """
    xi_fixed: если None — natural (max(3*(k+1), 3*(p_fit+1))),
              иначе — конкретное число соседей.
    """
    max_lag = lags_desc[0]
    n = len(att_hist)
    k = len(lags_desc)

    xi_nat = max(3 * (k + 1), 3 * (p_fit + 1))
    xi = xi_nat if xi_fixed is None else xi_fixed

    X_search, X_fit, y = make_matrices(att_hist, lags_desc, p_fit)
    if len(X_search) < xi:
        # если соседей не хватает, берём сколько есть
        xi = len(X_search)
    if xi < p_fit + 2:
        return np.full(horizon, np.nan)

    lags_arr = np.array(lags_desc, dtype=np.int32)
    buf = np.empty(n + horizon)
    buf[:n] = att_hist

    out = np.empty(horizon)
    for h in range(horizon):
        t     = n + h - 1
        vec_s = buf[t - lags_arr]
        vec_f = buf[t - p_fit + 1: t + 1]

        dists  = np.linalg.norm(X_search - vec_s, axis=1)
        xi_eff = min(xi, len(X_search))
        nn     = np.argpartition(dists, xi_eff - 1)[:xi_eff]
        h_bw   = max(float(dists[nn].max()), 1e-10)

        X_nn = X_fit[nn]; y_nn = y[nn]
        w    = np.exp(-0.5 * (np.linalg.norm(X_nn - vec_f, axis=1) / h_bw) ** 2)
        A    = np.hstack([np.ones((xi_eff, 1)), X_nn]); sw = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
        val  = float(c[0] + vec_f @ c[1:])

        out[h] = val; buf[t + 1] = val

    return out


# ── загрузка данных ────────────────────────────────────────────────────────────

print("Загрузка данных…", flush=True)
att_data, ratio_data, origins_data = {}, {}, {}
max_lag_global = max(ls[0] for ls in LAG_SETS.values())
xi_max = max(XI_MODES.values(), key=lambda v: v or 0) or max(
    max(3*(len(ls)+1), 3*(P_FIT+1)) for ls in LAG_SETS.values()
)

for ticker in TICKERS:
    att_full, ratio = load_att(ticker)
    att_data[ticker] = att_full; ratio_data[ticker] = ratio
    n_total  = len(att_full)
    min_start = max_lag_global + xi_max + HORIZON + 10
    end_k    = n_total - HORIZON - 1
    start_k  = max(min_start, end_k - N_ORIG * 5)
    cands    = list(range(start_k, end_k))
    step     = max(1, len(cands) // N_ORIG)
    origins_data[ticker] = cands[::step][:N_ORIG]
    print(f"  {ticker}: {len(origins_data[ticker])} origins, n={n_total}", flush=True)


# ── основной прогон ────────────────────────────────────────────────────────────

lag_names = list(LAG_SETS.keys())
xi_names  = list(XI_MODES.keys())

# mapes[xi_mode][lag_name][ticker]
mapes = {
    xm: {name: {t: [] for t in TICKERS} for name in lag_names}
    for xm in xi_names
}

t0     = time.time()
n_done = 0
total  = len(TICKERS) * N_ORIG
n_combos = len(xi_names) * len(lag_names)

print(f"\nПрогон: {n_combos} комбинаций (xi × lag_set), {total} origins…", flush=True)

for ticker in TICKERS:
    att_full = att_data[ticker]; ratio = ratio_data[ticker]
    origins  = origins_data[ticker]
    for origin_k in origins:
        att_hist   = att_full[:origin_k]
        att_actual = att_full[origin_k: origin_k + HORIZON]
        ratio0     = float(ratio[origin_k])
        for name, lags in LAG_SETS.items():
            for xm, xi_val in XI_MODES.items():
                dhat = forecast_lwr(att_hist, lags, P_FIT, HORIZON, xi_fixed=xi_val)
                m    = lp_mape(dhat, att_actual, ratio0)
                if np.isfinite(m):
                    mapes[xm][name][ticker].append(m)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{total}  ({time.time()-t0:.0f}s)", flush=True)

print(f"Завершено за {time.time()-t0:.1f}с", flush=True)


# ── вывод ──────────────────────────────────────────────────────────────────────

def agg(mapes_d, xm, name):
    return np.nanmean([v for t in TICKERS for v in mapes_d[xm][name][t]])


# Сводная таблица: строки=lag_sets, столбцы=xi_modes
ref = agg(mapes, "natural", "uniform_p80")

print("\n── AGG LP-MAPE по lag_set × xi_mode ──────────────────────────────────────")
hdr = f"{'Метод':<20}  {'k':>4}  {'xi_nat':>7}"
for xm in xi_names:
    hdr += f"  {xm:>9}"
print(hdr + "   (Δ% vs uniform_p80 natural)")
print("─" * 95)

for name in lag_names:
    lags = LAG_SETS[name]; k = len(lags)
    xi_nat_val = max(3*(k+1), 3*(P_FIT+1))
    row = f"{name:<20}  {k:>4}  {xi_nat_val:>7}"
    for xm in xi_names:
        m = agg(mapes, xm, name)
        row += f"  {m:>9.5f}"
    m_ref_here = agg(mapes, "natural", name)
    delta = (ref / m_ref_here - 1) * 100
    print(row + f"   [{delta:+.1f}%]")


# Лучший non-uniform для каждого xi
print(f"\n── Лучший non-uniform по xi_mode ─────────────────────────────────────────")
print(f"{'xi_mode':<12}  {'best_nonunif':<20}  {'MAPE':>9}  {'vs unif_p80':>12}  {'vs unif_p150':>13}")
print("─" * 74)
for xm in xi_names:
    best_name = min(
        (n for n in lag_names if not n.startswith("uniform")),
        key=lambda n: agg(mapes, xm, n),
    )
    m_best  = agg(mapes, xm, best_name)
    m_p80   = agg(mapes, xm, "uniform_p80")
    m_p150  = agg(mapes, xm, "uniform_p150")
    d80  = (m_p80  / m_best - 1) * 100
    d150 = (m_p150 / m_best - 1) * 100
    print(f"{xm:<12}  {best_name:<20}  {m_best:>9.5f}  {d80:>+11.1f}%  {d150:>+12.1f}%")


# Детальный взгляд: uniform_p80 vs best_nonunif при каждом xi
print(f"\n── Per-ticker при xi=243: лучший non-uniform vs uniform_p80 ───────────────")
best_243 = min(
    (n for n in lag_names if not n.startswith("uniform")),
    key=lambda n: agg(mapes, "xi_243", n),
)
print(f"Лучший non-uniform при xi_243: {best_243}")
print(f"{'Ticker':<8}  {'nonunif':>9}  {'unif_p80':>10}  {'Δ%':>7}")
print("─" * 38)
for t in TICKERS:
    m_nu = np.nanmean(mapes["xi_243"][best_243][t]) if mapes["xi_243"][best_243][t] else np.nan
    m_u  = np.nanmean(mapes["xi_243"]["uniform_p80"][t]) if mapes["xi_243"]["uniform_p80"][t] else np.nan
    gain = (m_u / m_nu - 1) * 100 if np.isfinite(m_nu) and m_nu > 0 else np.nan
    print(f"{t:<8}  {m_nu:>9.5f}  {m_u:>10.5f}  {gain:>+6.1f}%")


# ── рисунок ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, xm in zip(axes, xi_names):
    vals  = [agg(mapes, xm, n) for n in lag_names]
    colors = ["#1976D2" if n.startswith("uniform") else "#E53935" for n in lag_names]
    bars = ax.barh(lag_names, vals, color=colors, edgecolor="white")
    ref_here = agg(mapes, xm, "uniform_p80")
    ax.axvline(ref_here, color="#1976D2", lw=1.5, ls="--", alpha=0.7)
    for bar, name in zip(bars, lag_names):
        d = (ref_here / agg(mapes, xm, name) - 1) * 100
        ax.text(bar.get_width() + 2e-4, bar.get_y() + bar.get_height()/2,
                f"{d:+.1f}%", va="center", fontsize=7)
    ax.set_title(f"xi={xm}", fontsize=10)
    ax.set_xlabel("AGG LP-MAPE")
    ax.grid(True, alpha=0.3, axis="x")

axes[0].set_ylabel("Lag set")
fig.suptitle(f"Non-uniform lags при разных xi  ({len(TICKERS)} тикера, {N_ORIG} origins, p_fit={P_FIT})\n"
             f"синий=uniform, красный=non-uniform",
             fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "66_fixed_xi.png", dpi=150)
plt.close(fig)

print(f"\nРис.: {FIG_DIR}/66_fixed_xi.png")
print(f"\nСкрипт 66 завершён за {time.time()-t0:.1f}с")
