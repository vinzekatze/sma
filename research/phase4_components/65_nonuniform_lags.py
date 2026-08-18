"""
65 — Неравномерные задержки для C2-C5 аттрактора.

Гипотеза: для медленных компонент C2-C5 (период 8-130 баров) неравномерные
задержки информативнее равномерных при меньшей размерности вложения.

Фаза 1: расширенный sweep p_search = 80..200 (шаг 20), uniform, p_fit=13.
         Находим плато (скр.64 показал монотонный рост до p=80).

Фаза 2: тест наборов задержек (uniform + nonuniform) при фиксированном p_fit=13.
         xi = max(3*(k+1), 3*(p_fit+1)) — обеспечиваем минимум соседей для fit.

Метрика: LP-MAPE на att (C2-C5 через LP Butterworth Wn=0.125).
Режим: итеративный без пересчёта (mode 2 — как скр.63-64).
Протокол: 4 тикера, 1d, N_ORIG=50, horizon=20.
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
OUT_DIR  = Path(__file__).parent / "output65"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS  = ["SBER", "LKOH", "CHMF", "MRKP"]
N_ORIG   = 50
HORIZON  = 20
WN       = 0.125
P_FIT    = 13       # лучший из скр.64 (фаза 2)
FILTER_ORDER = 4

# Фаза 1: sweep p_search uniform
P1_RANGE = list(range(80, 201, 20))   # 80,100,120,140,160,180,200

# Фаза 2: наборы задержек (убывающий порядок, включают 0 = текущий момент)
# C2-C5 периоды: ~8-16, 16-52, 52-103, 103+ баров
LAG_SETS = {
    # равномерные (baseline)
    "uniform_p80":    list(range(79, -1, -1)),         # 80 dims, max_lag=79
    "uniform_p150":   list(range(149, -1, -1)),        # 150 dims, max_lag=149
    # степени двойки — захватывают все октавы C2-C5
    "geom_k9":        [128, 64, 32, 16, 8, 4, 2, 1, 0],
    # по ключевым периодам C2-C5 и их четвертям
    "c2c5_k13":       [130, 103, 78, 52, 40, 26, 16, 12, 8, 5, 2, 1, 0],
    # квадратурные задержки по доминирующим периодам ~10, 25, 75, 120 баров
    "quarter_k12":    [120, 90, 60, 50, 38, 25, 19, 12, 8, 4, 2, 0],
    # гибрид: редко в дальней зоне, часто в ближней
    "dense_near_k18": [130, 103, 80, 65, 52, 40, 32, 26, 20, 16, 12, 10, 8, 6, 4, 2, 1, 0],
}

_SOS_LP = butter(FILTER_ORDER, WN, btype="low", output="sos")


# ── утилиты ────────────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
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


def lp_mape(att_pred: np.ndarray, att_actual: np.ndarray, ratio0: float) -> float:
    h = min(len(att_pred), len(att_actual))
    if h == 0 or not np.isfinite(att_pred[:h]).all():
        return np.nan
    r_pred = ratio0 + np.cumsum(att_pred[:h])
    r_act  = ratio0 + np.cumsum(att_actual[:h])
    return float(np.mean(np.abs(r_pred - r_act) / (np.abs(r_act) + 1e-10)))


def load_att(ticker: str):
    import json
    data   = json.loads((DATA_DIR / ticker / "1d.json").read_text())
    close  = np.array([c["close"] for c in data], dtype=np.float64)
    trend  = logtrend_causal(close)
    ratio  = close / np.maximum(trend, 1e-10)
    dratio = np.diff(ratio, prepend=ratio[0])
    return sosfilt(_SOS_LP, dratio), ratio


# ── ядро прогноза ──────────────────────────────────────────────────────────────

def make_matrices(att: np.ndarray, lags_desc: list, p_fit: int):
    """
    lags_desc: убывающий список задержек (0 = текущий момент).
    Строит:
      X_search (m×k) — поиск в k-мерном пространстве задержек
      X_fit    (m×p_fit) — аппроксимация по p_fit последовательным задержкам (oldest→newest)
      y        (m,)  — att[t+1]
    Текущее время t = max_lag + j для j-й строки.
    """
    max_lag  = lags_desc[0]
    lags_arr = np.array(lags_desc, dtype=np.int32)
    n = len(att)
    m = n - max_lag - 1
    k = len(lags_desc)
    if m <= 0:
        return np.zeros((0, k)), np.zeros((0, p_fit)), np.zeros(0)

    t_arr = np.arange(max_lag, n - 1, dtype=np.int32)

    # Search: att[t - lag] для каждого lag
    X_search = np.column_stack([att[t_arr - lag] for lag in lags_arr])

    # Fit: p_fit последовательных (att[t-p_fit+1], ..., att[t])
    X_fit = np.column_stack([att[t_arr - (p_fit - 1 - k_)] for k_ in range(p_fit)])

    y = att[t_arr + 1]
    return X_search, X_fit, y


def forecast_lwr(att_hist: np.ndarray, lags_desc: list, p_fit: int,
                 horizon: int) -> np.ndarray:
    """
    LWR итеративный (mode 2: query обновляется, матрица — нет).
    lags_desc: убывающий список задержек (включает 0).
    """
    max_lag = lags_desc[0]
    n = len(att_hist)
    k = len(lags_desc)
    xi = max(3 * (k + 1), 3 * (p_fit + 1))

    X_search, X_fit, y = make_matrices(att_hist, lags_desc, p_fit)
    if len(X_search) < xi:
        return np.full(horizon, np.nan)

    lags_arr = np.array(lags_desc, dtype=np.int32)

    # буфер для итеративного прогноза (не перевыделяем)
    buf = np.empty(n + horizon)
    buf[:n] = att_hist

    out = np.empty(horizon)
    for h in range(horizon):
        t = n + h - 1   # текущее время

        vec_s = buf[t - lags_arr]              # k-мерный вектор поиска
        vec_f = buf[t - p_fit + 1: t + 1]     # p_fit-мерный вектор аппроксимации

        dists  = np.linalg.norm(X_search - vec_s, axis=1)
        xi_eff = min(xi, len(X_search))
        nn     = np.argpartition(dists, xi_eff - 1)[:xi_eff]
        h_bw   = max(float(dists[nn].max()), 1e-10)

        X_nn = X_fit[nn]
        y_nn = y[nn]
        w    = np.exp(-0.5 * (np.linalg.norm(X_nn - vec_f, axis=1) / h_bw) ** 2)
        A    = np.hstack([np.ones((xi_eff, 1)), X_nn])
        sw   = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
        val  = float(c[0] + vec_f @ c[1:])

        out[h]    = val
        buf[t + 1] = val

    return out


def forecast_uniform(att_hist: np.ndarray, p_search: int, p_fit: int,
                     horizon: int) -> np.ndarray:
    """Обёртка для равномерных задержек — lags = [p_search-1, ..., 1, 0]."""
    lags = list(range(p_search - 1, -1, -1))
    return forecast_lwr(att_hist, lags, p_fit, horizon)


# ── загрузка данных ────────────────────────────────────────────────────────────

print("Загрузка данных…", flush=True)
att_data, ratio_data, origins_data = {}, {}, {}

max_lag_global = max(
    max(ls[0] for ls in LAG_SETS.values()),
    max(P1_RANGE) - 1,
)
xi_max = max(
    max(max(3 * (len(ls) + 1), 3 * (P_FIT + 1)) for ls in LAG_SETS.values()),
    3 * (max(P1_RANGE) + 1),
)

for ticker in TICKERS:
    att_full, ratio = load_att(ticker)
    att_data[ticker]   = att_full
    ratio_data[ticker] = ratio

    n_total   = len(att_full)
    min_start = max_lag_global + xi_max + HORIZON + 10
    end_k     = n_total - HORIZON - 1
    start_k   = max(min_start, end_k - N_ORIG * 5)
    cands     = list(range(start_k, end_k))
    step      = max(1, len(cands) // N_ORIG)
    origins_data[ticker] = cands[::step][:N_ORIG]
    print(f"  {ticker}: {len(origins_data[ticker])} origins, n={n_total}", flush=True)


# ── фаза 1: sweep p_search uniform ────────────────────────────────────────────

print(f"\nФаза 1: sweep p_search {P1_RANGE[0]}…{P1_RANGE[-1]} (uniform, p_fit={P_FIT})", flush=True)
mapes_p1 = {ps: {t: [] for t in TICKERS} for ps in P1_RANGE}

t0     = time.time()
n_done = 0
total1 = len(TICKERS) * N_ORIG

for ticker in TICKERS:
    att_full = att_data[ticker]
    ratio    = ratio_data[ticker]
    origins  = origins_data[ticker]
    for origin_k in origins:
        att_hist   = att_full[:origin_k]
        att_actual = att_full[origin_k: origin_k + HORIZON]
        ratio0     = float(ratio[origin_k])
        for ps in P1_RANGE:
            dhat = forecast_uniform(att_hist, ps, P_FIT, HORIZON)
            m    = lp_mape(dhat, att_actual, ratio0)
            if np.isfinite(m):
                mapes_p1[ps][ticker].append(m)
        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{total1}  ({time.time()-t0:.0f}s)", flush=True)

print(f"Фаза 1 завершена за {time.time()-t0:.1f}с", flush=True)


# ── фаза 2: lag sets ──────────────────────────────────────────────────────────

print(f"\nФаза 2: lag sets ({len(LAG_SETS)} конфигураций, p_fit={P_FIT})", flush=True)
lag_names = list(LAG_SETS.keys())
mapes_p2  = {name: {t: [] for t in TICKERS} for name in lag_names}

t1     = time.time()
n_done2 = 0
total2  = len(TICKERS) * N_ORIG

for ticker in TICKERS:
    att_full = att_data[ticker]
    ratio    = ratio_data[ticker]
    origins  = origins_data[ticker]
    for origin_k in origins:
        att_hist   = att_full[:origin_k]
        att_actual = att_full[origin_k: origin_k + HORIZON]
        ratio0     = float(ratio[origin_k])
        for name, lags in LAG_SETS.items():
            dhat = forecast_lwr(att_hist, lags, P_FIT, HORIZON)
            m    = lp_mape(dhat, att_actual, ratio0)
            if np.isfinite(m):
                mapes_p2[name][ticker].append(m)
        n_done2 += 1
        if n_done2 % 50 == 0:
            print(f"  {n_done2}/{total2}  ({time.time()-t1:.0f}s)", flush=True)

print(f"Фаза 2 завершена за {time.time()-t1:.1f}с", flush=True)


# ── агрегация и вывод ──────────────────────────────────────────────────────────

def agg(mapes_dict, key):
    return np.nanmean([v for t in TICKERS for v in mapes_dict[key][t]])


# Фаза 1
agg1 = [agg(mapes_p1, ps) for ps in P1_RANGE]
best1_idx = int(np.nanargmin(agg1))
best1_ps  = P1_RANGE[best1_idx]

print("\n── Фаза 1: uniform p_search sweep (p_fit=13) ──────────────────────")
print(f"{'p_search':>10}  {'AGG MAPE':>10}  {'Δ% vs p80':>10}")
print("─" * 38)
ref_p1 = agg1[P1_RANGE.index(80)]
for i, ps in enumerate(P1_RANGE):
    gain   = (ref_p1 / agg1[i] - 1) * 100
    marker = " ← лучший" if ps == best1_ps else ""
    print(f"{ps:>10}  {agg1[i]:>10.5f}  {gain:>+9.1f}%{marker}")

# Фаза 2
agg2  = {name: agg(mapes_p2, name) for name in lag_names}
ref_val = agg2["uniform_p80"]

print(f"\n── Фаза 2: lag sets (p_fit={P_FIT}) ───────────────────────────────────────────")
print(f"{'Метод':<20}  {'k':>4}  {'max_lag':>7}  {'xi':>6}  {'AGG MAPE':>10}  {'Δ% vs unif_p80':>15}")
print("─" * 72)
for name in sorted(agg2, key=lambda n: agg2[n]):
    lags  = LAG_SETS[name]
    k     = len(lags)
    ml    = lags[0]
    xi    = max(3 * (k + 1), 3 * (P_FIT + 1))
    delta = (ref_val / agg2[name] - 1) * 100
    tag   = " ←" if name == "uniform_p80" else ""
    print(f"{name:<20}  {k:>4}  {ml:>7}  {xi:>6}  {agg2[name]:>10.5f}  {delta:>+14.1f}%{tag}")

# лучший non-uniform
best_nu = min(
    (n for n in lag_names if not n.startswith("uniform")),
    key=lambda n: agg2[n],
)
print(f"\nЛучший non-uniform: {best_nu}  (MAPE={agg2[best_nu]:.5f})")
print(f"vs uniform_p80:  MAPE={agg2['uniform_p80']:.5f}  "
      f"выигрыш={(agg2['uniform_p80']/agg2[best_nu]-1)*100:+.1f}%")
print(f"vs uniform_p150: MAPE={agg2['uniform_p150']:.5f}  "
      f"выигрыш={(agg2['uniform_p150']/agg2[best_nu]-1)*100:+.1f}%")

print(f"\n── Per-ticker: {best_nu} vs uniform_p80 ──────────────────────────")
print(f"{'Ticker':<8}  {'nonunif':>9}  {'unif_p80':>10}  {'Δ%':>7}")
print("─" * 42)
for t in TICKERS:
    m_nu = np.nanmean(mapes_p2[best_nu][t]) if mapes_p2[best_nu][t] else np.nan
    m_u  = np.nanmean(mapes_p2["uniform_p80"][t]) if mapes_p2["uniform_p80"][t] else np.nan
    gain = (m_u / m_nu - 1) * 100 if np.isfinite(m_nu) and m_nu > 0 else np.nan
    print(f"{t:<8}  {m_nu:>9.5f}  {m_u:>10.5f}  {gain:>+6.1f}%")


# ── рисунки ────────────────────────────────────────────────────────────────────

# Рис. A — sweep p_search
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(P1_RANGE, agg1, "b-o", lw=2, ms=7)
ax.axvline(best1_ps, color="blue", lw=1, ls="--", alpha=0.7,
           label=f"best p_search={best1_ps}")
ax.axhline(agg2["uniform_p80"], color="gray", lw=1, ls=":", alpha=0.6,
           label=f"phase2 uniform_p80={agg2['uniform_p80']:.5f}")
ax.set_xlabel("p_search (uniform)"); ax.set_ylabel("AGG LP-MAPE")
ax.set_title(f"LP-MAPE vs p_search  ({len(TICKERS)} тикера, {N_ORIG} origins, p_fit={P_FIT})")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "65_phase1_psearch.png", dpi=150)
plt.close(fig)

# Рис. B — lag sets bar chart
names_sorted = sorted(agg2, key=lambda n: agg2[n])
colors = ["#1976D2" if n.startswith("uniform") else "#E53935" for n in names_sorted]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.barh(names_sorted, [agg2[n] for n in names_sorted], color=colors, edgecolor="white")
ax.axvline(ref_val, color="#1976D2", lw=1.5, ls="--", alpha=0.7,
           label=f"uniform_p80 = {ref_val:.5f}")
for bar, name in zip(bars, names_sorted):
    delta = (ref_val / agg2[name] - 1) * 100
    lags  = LAG_SETS[name]; k = len(lags); xi = max(3*(k+1), 3*(P_FIT+1))
    ax.text(bar.get_width() + 3e-5,
            bar.get_y() + bar.get_height() / 2,
            f"{delta:+.1f}%  k={k} ξ={xi}",
            va="center", fontsize=8)
ax.set_xlabel("AGG LP-MAPE (меньше = лучше)")
ax.set_title(f"Lag sets — {len(TICKERS)} тикера, {N_ORIG} origins, p_fit={P_FIT}\n"
             f"синий=uniform, красный=non-uniform")
ax.legend(); ax.grid(True, alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(FIG_DIR / "65_phase2_lagsets.png", dpi=150)
plt.close(fig)

print(f"\nРис. A: {FIG_DIR}/65_phase1_psearch.png")
print(f"Рис. B: {FIG_DIR}/65_phase2_lagsets.png")
print(f"\nСкрипт 65 завершён за {time.time()-t0:.1f}с")
