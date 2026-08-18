"""
64 — Полный прогон: sweep p_search (4…80) на 8 тикерах.

Два режима для каждого p_search:
  split    — p_fit=16 (лучший из скр.63), поиск соседей в p_search-мерном пространстве
  baseline — p_fit=p_search (стандартный LWR)

Дополнительно: sweep p_fit (4…p_search) при p_search=67 для уточнения оптимума.

Протокол: 8 тикеров, 1d, N_ORIG=100, horizon=20, wn=0.125.
Режим: итеративный с пересчётом (лучший по скр.62).
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
OUT_DIR  = Path(__file__).parent / "output64"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS    = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
N_ORIG     = 100
HORIZON    = 20
WN         = 0.125
P_FIT_FIXED = 16           # лучший p_fit из скр.63
P_SEARCH_RANGE = range(4, 81, 2)   # 4,6,8,...,80 — шаг 2 для скорости
P_SEARCH_DETAIL = 67       # при этом p_search — дополнительный sweep p_fit
FILTER_ORDER = 4

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


def forecast_split(att_hist: np.ndarray, p_search: int, p_fit: int,
                   horizon: int) -> np.ndarray:
    """LWR итеративный с пересчётом, p_search для соседей, p_fit для аппроксимации."""
    p_fit   = min(p_fit, p_search)
    xi      = 3 * (p_search + 1)
    n       = len(att_hist)
    X_full  = np.array([att_hist[i: i + p_search] for i in range(n - p_search)])
    y_full  = att_hist[p_search:]
    if len(X_full) < xi:
        return np.full(horizon, np.nan)

    vec_s = att_hist[-p_search:].copy()
    vec_f = att_hist[-p_fit:].copy()
    out   = np.empty(horizon)

    for h in range(horizon):
        dists  = np.linalg.norm(X_full - vec_s, axis=1)
        nn     = np.argpartition(dists, xi)[:xi]
        h_bw   = max(float(dists[nn].max()), 1e-10)
        X_nn   = X_full[nn, -p_fit:]
        y_nn   = y_full[nn]
        w      = np.exp(-0.5 * (np.linalg.norm(X_nn - vec_f, axis=1) / h_bw) ** 2)
        A      = np.hstack([np.ones((xi, 1)), X_nn])
        sw     = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
        val    = float(c[0] + vec_f @ c[1:])
        out[h] = val
        vec_s = np.roll(vec_s, -1); vec_s[-1] = val
        vec_f = np.roll(vec_f, -1); vec_f[-1] = val

    return out


def load_att(ticker: str):
    import json
    data   = json.loads((DATA_DIR / ticker / "1d.json").read_text())
    close  = np.array([c["close"] for c in data], dtype=np.float64)
    trend  = logtrend_causal(close)
    ratio  = close / np.maximum(trend, 1e-10)
    dratio = np.diff(ratio, prepend=ratio[0])
    return sosfilt(_SOS_LP, dratio), ratio


# ── загрузка данных ────────────────────────────────────────────────────────────

print("Загрузка данных…", flush=True)
att_data, ratio_data, origins_data = {}, {}, {}

p_max = max(P_SEARCH_RANGE)
for ticker in TICKERS:
    att_full, ratio = load_att(ticker)
    att_data[ticker]   = att_full
    ratio_data[ticker] = ratio

    xi_max    = 3 * (p_max + 1)
    n_total   = len(att_full)
    min_start = p_max + xi_max + HORIZON + 10
    end_k     = n_total - HORIZON - 1
    start_k   = max(min_start, end_k - N_ORIG * 5)
    cands     = list(range(start_k, end_k))
    step      = max(1, len(cands) // N_ORIG)
    origins_data[ticker] = cands[::step][:N_ORIG]
    print(f"  {ticker}: {len(origins_data[ticker])} origins, n={n_total}", flush=True)

# ── фаза 1: sweep p_search ─────────────────────────────────────────────────────
# mapes_split[p_search][ticker] и mapes_base[p_search][ticker]

p_search_list = list(P_SEARCH_RANGE)
mapes_split = {ps: {t: [] for t in TICKERS} for ps in p_search_list}
mapes_base  = {ps: {t: [] for t in TICKERS} for ps in p_search_list}

t0     = time.time()
n_done = 0
total  = len(TICKERS) * N_ORIG

print(f"\nФаза 1: sweep p_search {p_search_list[0]}…{p_search_list[-1]}", flush=True)

for ticker in TICKERS:
    att_full = att_data[ticker]
    ratio    = ratio_data[ticker]
    origins  = origins_data[ticker]

    for origin_k in origins:
        att_hist   = att_full[:origin_k]
        att_actual = att_full[origin_k: origin_k + HORIZON]
        ratio0     = float(ratio[origin_k])

        for ps in p_search_list:
            # split: p_fit=16
            dhat = forecast_split(att_hist, ps, P_FIT_FIXED, HORIZON)
            m    = lp_mape(dhat, att_actual, ratio0)
            if np.isfinite(m):
                mapes_split[ps][ticker].append(m)

            # baseline: p_fit=p_search
            if ps != P_FIT_FIXED:          # избегаем двойного счёта когда ps==16
                dhat_b = forecast_split(att_hist, ps, ps, HORIZON)
                m_b    = lp_mape(dhat_b, att_actual, ratio0)
            else:
                dhat_b, m_b = dhat, m     # при ps==16 split==baseline
            if np.isfinite(m_b):
                mapes_base[ps][ticker].append(m_b)

        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{total}  ({time.time()-t0:.0f}s)", flush=True)

print(f"\nФаза 1 завершена за {time.time()-t0:.1f}с", flush=True)

# ── фаза 2: sweep p_fit при p_search=P_SEARCH_DETAIL ─────────────────────────

print(f"\nФаза 2: sweep p_fit при p_search={P_SEARCH_DETAIL}…", flush=True)
p_fit_range = range(4, P_SEARCH_DETAIL + 1)
mapes_pfit  = {pf: {t: [] for t in TICKERS} for pf in p_fit_range}

t1 = time.time()
n2 = 0

for ticker in TICKERS:
    att_full = att_data[ticker]
    ratio    = ratio_data[ticker]
    origins  = origins_data[ticker]

    for origin_k in origins:
        att_hist   = att_full[:origin_k]
        att_actual = att_full[origin_k: origin_k + HORIZON]
        ratio0     = float(ratio[origin_k])

        for pf in p_fit_range:
            dhat = forecast_split(att_hist, P_SEARCH_DETAIL, pf, HORIZON)
            m    = lp_mape(dhat, att_actual, ratio0)
            if np.isfinite(m):
                mapes_pfit[pf][ticker].append(m)

        n2 += 1
        if n2 % 100 == 0:
            print(f"  {n2}/{total}  ({time.time()-t1:.0f}s)", flush=True)

print(f"Фаза 2 завершена за {time.time()-t1:.1f}с", flush=True)

# ── агрегация ──────────────────────────────────────────────────────────────────

def agg_all(mapes_dict, key_list):
    return np.array([
        np.nanmean([v for t in TICKERS for v in mapes_dict[k][t]])
        for k in key_list
    ])

agg_split = agg_all(mapes_split, p_search_list)
agg_base  = agg_all(mapes_base,  p_search_list)

best_split_idx = int(np.nanargmin(agg_split))
best_base_idx  = int(np.nanargmin(agg_base))
best_ps_split  = p_search_list[best_split_idx]
best_ps_base   = p_search_list[best_base_idx]

p_fit_list = list(p_fit_range)
agg_pfit   = agg_all(mapes_pfit, p_fit_list)
best_pf_idx = int(np.nanargmin(agg_pfit))
best_pf     = p_fit_list[best_pf_idx]

# t-test split vs baseline при лучших p_search
flat_split = [v for t in TICKERS for v in mapes_split[best_ps_split][t]]
flat_base  = [v for t in TICKERS for v in mapes_base[best_ps_split][t]]
n_min = min(len(flat_split), len(flat_base))
_, pval = ttest_rel(flat_split[:n_min], flat_base[:n_min])

print("\n── Фаза 1: AGG LP-MAPE vs p_search ─────────────────────────────────────────")
print(f"{'p_search':>9}  {'split p_fit=16':>15}  {'baseline':>10}  {'выигрыш':>8}")
print("─" * 50)
for i, ps in enumerate(p_search_list):
    gain = (agg_base[i] / agg_split[i] - 1) * 100 if agg_split[i] > 0 else 0
    marker = " ←" if ps == best_ps_split else ""
    print(f"{ps:>9}  {agg_split[i]:>15.5f}  {agg_base[i]:>10.5f}  {gain:>7.1f}%{marker}")

print(f"\nЛучший split:    p_search={best_ps_split}  MAPE={agg_split[best_split_idx]:.5f}")
print(f"Лучший baseline: p_search={best_ps_base}  MAPE={agg_base[best_base_idx]:.5f}")
print(f"Выигрыш split vs baseline @ best split: "
      f"{(agg_base[best_split_idx]/agg_split[best_split_idx]-1)*100:.1f}%  p={pval:.4f}")

print(f"\n── Фаза 2: AGG LP-MAPE vs p_fit (p_search={P_SEARCH_DETAIL}) ────────────────")
print(f"Лучший p_fit={best_pf}  MAPE={agg_pfit[best_pf_idx]:.5f}")
print(f"vs p_fit=16: MAPE={agg_pfit[p_fit_list.index(16)]:.5f}")

# ── рис. A — sweep p_search ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(p_search_list, agg_split, "b-",  lw=2.0, label=f"split  p_fit={P_FIT_FIXED}")
ax.plot(p_search_list, agg_base,  "k--", lw=1.5, label="baseline  p_fit=p_search", alpha=0.7)
ax.axvline(best_ps_split, color="blue",  lw=1, ls=":", alpha=0.8,
           label=f"best split p_search={best_ps_split}")
ax.axvline(best_ps_base,  color="black", lw=1, ls=":", alpha=0.5,
           label=f"best base  p_search={best_ps_base}")
ax.set_xlabel("p_search"); ax.set_ylabel("AGG LP-MAPE")
ax.set_title(f"LP-MAPE vs p_search  (8 тикеров, {N_ORIG} origins, LWR итер.+пересчёт)")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "64_mape_vs_psearch.png", dpi=150)
plt.close(fig)
print(f"\nРис. A: {FIG_DIR}/64_mape_vs_psearch.png")

# ── рис. B — sweep p_fit при p_search=67 ─────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(p_fit_list, agg_pfit, color="#d65f5f", lw=2.0)
ax.axvline(best_pf, color="red",   lw=1.5, ls="--", label=f"best p_fit={best_pf}")
ax.axvline(16,      color="gray",  lw=1.0, ls=":",  label="p_fit=16 (из скр.63)")
base_line = agg_pfit[p_fit_list.index(P_SEARCH_DETAIL)] if P_SEARCH_DETAIL in p_fit_list else np.nan
if np.isfinite(base_line):
    ax.axhline(base_line, color="black", lw=1, ls=":", alpha=0.5,
               label=f"baseline p_fit={P_SEARCH_DETAIL}: {base_line:.5f}")
ax.set_xlabel("p_fit"); ax.set_ylabel("AGG LP-MAPE")
ax.set_title(f"LP-MAPE vs p_fit  (p_search={P_SEARCH_DETAIL}, 8 тикеров, {N_ORIG} origins)")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "64_mape_vs_pfit.png", dpi=150)
plt.close(fig)
print(f"Рис. B: {FIG_DIR}/64_mape_vs_pfit.png")

# ── per-ticker при лучших параметрах ──────────────────────────────────────────

print(f"\n── Per-ticker: split(ps={best_ps_split}, pf={P_FIT_FIXED}) vs baseline(ps=pf={best_ps_split}) ──")
print(f"{'Ticker':<8}  {'split':>8}  {'baseline':>10}  {'Δ%':>7}")
print("─" * 40)
for t in TICKERS:
    ms = np.nanmean(mapes_split[best_ps_split][t]) if mapes_split[best_ps_split][t] else np.nan
    mb = np.nanmean(mapes_base[best_ps_split][t])  if mapes_base[best_ps_split][t]  else np.nan
    gain = (mb / ms - 1) * 100 if np.isfinite(ms) and np.isfinite(mb) and ms > 0 else np.nan
    print(f"{t:<8}  {ms:>8.5f}  {mb:>10.5f}  {gain:>6.1f}%")

print(f"\nСкрипт 64 завершён за {time.time()-t0:.1f}с")
