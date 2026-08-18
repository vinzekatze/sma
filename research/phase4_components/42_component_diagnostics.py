"""
42 — Диагностика компонент filter bank: характеристики, спектр, предсказуемость.

Сравниваются два фильтр-банка:
  6-band  (октавный):      cutoffs = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
  12-band (полуоктавный):  cutoffs = [0.5 / √2^k  для k = 1..11]

Диагностика каждой компоненты (агрегат по 8 тикерам, 1d):
  • доля дисперсии (энергии) в total dratio
  • стационарность: ADF-тест (p-value; малый p → стационарна)
  • долгосрочная память: Хёрст R/S (H > 0.5 → персистентна)
  • сложность/хаотичность: пермутационная энтропия PE ∈ [0,1]
  • автокорреляция: ρ[1] и ∑|ρ[1..80]|
  • доминантный период: пик Welch PSD

Предсказуемость (SBER, walk-forward, 40 origins):
  • zero-MAE (мартингал: ĉ = 0 → предсказываем «ноль»)
  • LWR MAE  (p=20, ξ=63, по горизонтам 1,3,5,10,20)
  • Gain = zero_MAE / lwr_MAE  (> 1 → LWR превосходит мартингала)

Нормализация: logtrend causal OLS (стандарт фазы 4).
Фильтры: causal Butterworth sosfilt (без look-ahead).
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.signal import welch as sp_welch
from statsmodels.tsa.stattools import adfuller

ROOT     = Path(__file__).resolve().parent.parent.parent   # .../sma (project root)
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from sma.core.forecast.embedding import build_delay_matrix, last_vector

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS  = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL = "1d"
EPS      = 1e-10
FILTER_ORDER = 4

_SQ2 = math.sqrt(2)
CUTOFFS_6  = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
N_BANDS_6  = len(CUTOFFS_6) + 1   # 6
CUTOFFS_12 = [0.5 / (_SQ2 ** k) for k in range(1, 12)]
N_BANDS_12 = len(CUTOFFS_12) + 1  # 12

# Читаемые метки для 6-band
LABELS_6 = [
    "C0 [2–4 bar]",
    "C1 [4–8 bar]",
    "C2 [8–16 bar]",
    "C3 [16–52 bar]",
    "C4 [52–103 bar]",
    "C5 [103+ bar]",
]

# Метки для 12-band: вычисляем из cutoffs
def _make_labels_12():
    lo = 2.0
    labels = []
    for fc in CUTOFFS_12:
        hi = round(1.0 / fc, 1)
        labels.append(f"[{lo:.1f}–{hi:.1f}]")
        lo = hi
    labels.append(f"[{lo:.1f}+]")
    return labels

LABELS_12 = _make_labels_12()

# LWR параметры
P_LWR  = 20
XI_LWR = 3 * (P_LWR + 1)   # 63
HORIZONS    = [1, 3, 5, 10, 20]
N_ORIGINS_WF = 40   # walk-forward origins (SBER)

# Диагностика
PE_ORDER = 5
ACF_LAGS = 80

# Цвета компонент: warm (быстрые) → cool (медленные)
CMAP_6  = [plt.cm.RdYlGn(x) for x in np.linspace(0.1, 0.9, N_BANDS_6)]
CMAP_12 = [plt.cm.RdYlGn(x) for x in np.linspace(0.1, 0.9, N_BANDS_12)]


# ── нормализация (logtrend causal OLS) ────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    log_c = np.log(close + EPS)
    n  = len(log_c)
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);      ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(log_c);  cty = np.cumsum(t * log_c)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = a + b * t
    trend[:2] = log_c[:2]
    return np.exp(trend)


def load_dratio(ticker: str) -> tuple[np.ndarray, np.ndarray]:
    with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
        candles = json.load(f)
    close = np.array([c["close"] for c in candles], dtype=np.float64)
    ratio = close / logtrend_causal(close)
    return ratio, np.diff(ratio)


# ── filter bank ────────────────────────────────────────────────────────────────

def make_fb(series: np.ndarray, cutoffs: list) -> np.ndarray:
    """Causal Butterworth filter bank. Returns (n_bands, N)."""
    comps = []; remaining = series.copy()
    for fc in cutoffs:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        comps.append(remaining - low)
        remaining = low
    comps.append(remaining)
    return np.array(comps)


# ── диагностические метрики ────────────────────────────────────────────────────

def permutation_entropy(s: np.ndarray, order: int = PE_ORDER) -> float:
    if len(s) < order + 2:
        return np.nan
    counts: dict = {}
    for i in range(len(s) - order + 1):
        key = tuple(np.argsort(s[i: i + order]))
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    H = -np.sum(probs * np.log2(probs + EPS))
    H_max = np.log2(math.factorial(order))
    return float(H / H_max) if H_max > 0 else np.nan


def hurst_rs(s: np.ndarray, min_n: int = 20) -> float:
    n = len(s)
    if n < min_n * 2:
        return np.nan
    lags = np.unique(np.logspace(np.log10(min_n), np.log10(n // 2),
                                  num=12).astype(int))
    rs_vals = []
    for lag in lags:
        sub_rs = []
        for start in range(0, n - lag, lag):
            sub = s[start: start + lag]
            mu  = sub.mean(); dev = np.cumsum(sub - mu)
            R   = dev.max() - dev.min(); S = sub.std()
            if S > 1e-12:
                sub_rs.append(R / S)
        if sub_rs:
            rs_vals.append(float(np.mean(sub_rs)))
    if len(rs_vals) < 3:
        return np.nan
    lx = np.log(lags[: len(rs_vals)].astype(float))
    ly = np.log(np.array(rs_vals))
    return float(np.polyfit(lx, ly, 1)[0])


def compute_acf(s: np.ndarray, max_lag: int = ACF_LAGS) -> np.ndarray:
    n = len(s); sc = s - s.mean()
    var = (sc ** 2).mean()
    if var < 1e-20:
        return np.zeros(max_lag + 1)
    result = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        result[k] = float((sc[: n - k] * sc[k:]).mean() / var)
    return result


def dominant_period(s: np.ndarray) -> float:
    nperseg = min(len(s) // 4, 1024)
    if nperseg < 8:
        return np.nan
    freqs, psd = sp_welch(s, fs=1.0, nperseg=nperseg)
    # убираем DC
    mask = freqs > 0
    freqs, psd = freqs[mask], psd[mask]
    if len(freqs) == 0:
        return np.nan
    peak_f = float(freqs[np.argmax(psd)])
    return float(1.0 / peak_f) if peak_f > 0 else np.nan


def diagnose(s: np.ndarray) -> dict:
    adf_p = float(adfuller(s, maxlag=5, autolag=None)[1]) if len(s) > 20 else np.nan
    acf   = compute_acf(s)
    return {
        "var":        float(np.var(s)),
        "hurst":      hurst_rs(s),
        "pe":         permutation_entropy(s),
        "adf_p":      adf_p,
        "acf1":       float(acf[1]),
        "acf_sum":    float(np.sum(np.abs(acf[1:]))),
        "dom_period": dominant_period(s),
        "acf_arr":    acf,
    }


# ── агрегация диагностики (8 тикеров) ─────────────────────────────────────────

def aggregate_diagnostics(cutoffs: list, n_bands: int, tag: str) -> list[dict]:
    """Медианные диагностики по всем тикерам для каждой компоненты."""
    buckets: list[dict[str, list]] = [
        {k: [] for k in ["energy_frac","hurst","pe","adf_p","acf1","acf_sum","dom_period","acf_arr"]}
        for _ in range(n_bands)
    ]
    for ticker in TICKERS:
        _, dratio = load_dratio(ticker)
        COMP     = make_fb(dratio, cutoffs)
        energies = np.array([float(np.var(c)) for c in COMP])
        total_e  = energies.sum() + EPS
        for ci in range(n_bands):
            d = diagnose(COMP[ci])
            buckets[ci]["energy_frac"].append(float(energies[ci] / total_e))
            for k in ["hurst","pe","adf_p","acf1","acf_sum","dom_period"]:
                buckets[ci][k].append(d[k])
            buckets[ci]["acf_arr"].append(d["acf_arr"])

    result: list[dict] = []
    for ci in range(n_bands):
        row: dict = {}
        for k in ["energy_frac","hurst","pe","adf_p","acf1","acf_sum","dom_period"]:
            vals = [v for v in buckets[ci][k] if not np.isnan(v)]
            row[k] = float(np.median(vals)) if vals else np.nan
        # среднее ACF по тикерам
        row["acf_arr"] = np.mean(np.stack(buckets[ci]["acf_arr"]), axis=0)
        result.append(row)

    print(f"  [{tag}] диагностика по {len(TICKERS)} тикерам: OK")
    return result


# ── LWR walk-forward предсказуемость ──────────────────────────────────────────

def _lwr_step_vec(X, y, v, xi):
    dists  = np.linalg.norm(X - v, axis=1)
    nn_idx = np.argpartition(dists, xi)[:xi]
    h_bw   = max(float(dists[nn_idx].max()), EPS)
    w      = np.exp(-0.5 * (dists[nn_idx] / h_bw) ** 2)
    A      = np.hstack([np.ones((xi, 1)), X[nn_idx]])
    sw     = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx], rcond=None)
    return float(c[0] + v @ c[1:])


def lwr_forecast_comp(series: np.ndarray, horizon: int) -> np.ndarray:
    X, y = build_delay_matrix(series, P_LWR)
    if len(X) < XI_LWR:
        return np.zeros(horizon)
    v = last_vector(series, p=P_LWR).copy()
    hat = np.empty(horizon)
    for h in range(horizon):
        hat[h] = _lwr_step_vec(X, y, v, XI_LWR)
        v = np.roll(v, -1); v[-1] = hat[h]
    return hat


def predictability_wf(dratio: np.ndarray, cutoffs: list,
                       n_origins: int = N_ORIGINS_WF,
                       tag: str = "") -> dict:
    """
    Каузальный walk-forward: LWR MAE vs zero MAE для каждой компоненты
    по горизонтам HORIZONS.
    """
    n = len(dratio)
    h_max = max(HORIZONS)
    COMP_full = make_fb(dratio, cutoffs)
    n_bands   = len(COMP_full)

    min_orig = P_LWR + XI_LWR + 10
    max_orig = n - h_max - 2
    origins  = np.arange(max(min_orig, max_orig - n_origins), max_orig)

    lwr_e  = {ci: {h: [] for h in HORIZONS} for ci in range(n_bands)}
    zero_e = {ci: {h: [] for h in HORIZONS} for ci in range(n_bands)}

    t0 = time.time()
    for vo in origins:
        COMP_h = make_fb(dratio[:vo], cutoffs)
        for ci in range(n_bands):
            actual = COMP_full[ci, vo: vo + h_max]
            if len(actual) < h_max:
                continue
            hat = lwr_forecast_comp(COMP_h[ci], h_max)
            for h in HORIZONS:
                lwr_e[ci][h].append(float(np.mean(np.abs(hat[:h] - actual[:h]))))
                zero_e[ci][h].append(float(np.mean(np.abs(actual[:h]))))

    med_lwr  = {ci: {h: float(np.nanmedian(lwr_e[ci][h]))  for h in HORIZONS} for ci in range(n_bands)}
    med_zero = {ci: {h: float(np.nanmedian(zero_e[ci][h])) for h in HORIZONS} for ci in range(n_bands)}
    dt = time.time() - t0
    print(f"  [{tag}] {n_bands} comp × {len(origins)} origins: {dt:.1f}s")
    return {"lwr": med_lwr, "zero": med_zero, "n_bands": n_bands}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("42 — Диагностика компонент filter bank")
print("=" * 72)

# 1. Диагностика
print("\n[1/4] Диагностика 6-band (8 тикеров)...")
diag_6 = aggregate_diagnostics(CUTOFFS_6, N_BANDS_6, "6-band")

print("\n[2/4] Диагностика 12-band (8 тикеров)...")
diag_12 = aggregate_diagnostics(CUTOFFS_12, N_BANDS_12, "12-band")

# 2. SBER для визуализации и предсказуемости
print("\n[3/4] LWR предсказуемость walk-forward (SBER)...")
_, dratio_sber = load_dratio("SBER")
COMP6_sber  = make_fb(dratio_sber, CUTOFFS_6)
COMP12_sber = make_fb(dratio_sber, CUTOFFS_12)

pred_6  = predictability_wf(dratio_sber, CUTOFFS_6,  N_ORIGINS_WF, "6-band")
pred_12 = predictability_wf(dratio_sber, CUTOFFS_12, N_ORIGINS_WF, "12-band")

print("\n[4/4] Генерация графиков...")


# ── Вывод сводных таблиц ──────────────────────────────────────────────────────

def _fmt(v, fmt=".4f"):
    return f"{v:{fmt}}" if not np.isnan(v) else "  nan "


print("\n╔══ Сводная таблица: 6-band (медиана по 8 тикерам) ════════════════════╗")
hdr = f"{'':>4} {'Energy%':>8} {'Hurst':>7} {'PE':>7} {'ADF_p':>8} {'ACF1':>7} {'ACF∑':>7} {'DomPer':>8}"
print(hdr); print("─" * 66)
for ci, d in enumerate(diag_6):
    print(f"C{ci}   {d['energy_frac']*100:>7.2f}%  "
          f"{_fmt(d['hurst'],'.3f'):>7}  {_fmt(d['pe'],'.3f'):>7}  "
          f"{_fmt(d['adf_p'],'.4f'):>8}  {_fmt(d['acf1'],'.4f'):>7}  "
          f"{_fmt(d['acf_sum'],'.2f'):>7}  {_fmt(d['dom_period'],'.1f'):>8}  "
          f"{LABELS_6[ci]}")

print("\n╔══ Сводная таблица: 12-band (медиана по 8 тикерам) ═══════════════════╗")
print(hdr + "   Полоса"); print("─" * 80)
for ci, d in enumerate(diag_12):
    print(f"C{ci:>2}  {d['energy_frac']*100:>7.2f}%  "
          f"{_fmt(d['hurst'],'.3f'):>7}  {_fmt(d['pe'],'.3f'):>7}  "
          f"{_fmt(d['adf_p'],'.4f'):>8}  {_fmt(d['acf1'],'.4f'):>7}  "
          f"{_fmt(d['acf_sum'],'.2f'):>7}  {_fmt(d['dom_period'],'.1f'):>8}  "
          f"{LABELS_12[ci]}")

print("\n╔══ Gain LWR/zero по горизонту (SBER) ═════════════════════════════════╗")
print(f"{'':>5}  " + "".join(f"  h={h:>2}" for h in HORIZONS))
print("─── 6-band ─────────────────────────────────")
for ci in range(N_BANDS_6):
    gains = [pred_6["zero"][ci][h] / (pred_6["lwr"][ci][h] + EPS) for h in HORIZONS]
    print(f"C{ci}     " + "".join(f"  {g:>5.2f}" for g in gains) + f"   {LABELS_6[ci]}")
print("─── 12-band ────────────────────────────────")
for ci in range(N_BANDS_12):
    gains = [pred_12["zero"][ci][h] / (pred_12["lwr"][ci][h] + EPS) for h in HORIZONS]
    print(f"C{ci:>2}    " + "".join(f"  {g:>5.2f}" for g in gains) + f"   {LABELS_12[ci]}")


# ══════════════════════════════════════════════════════════════════════════════
#  ФИГУРА 1: Компоненты во времени (SBER, 6-band)
# ══════════════════════════════════════════════════════════════════════════════

PLOT_BARS = 500   # последние N баров для визуализации
xrange = np.arange(PLOT_BARS)

fig1, axes = plt.subplots(N_BANDS_6, 1, figsize=(14, 10), sharex=True)
fig1.suptitle(f"Filter bank 6-band — компоненты dratio  |  SBER 1d  (последние {PLOT_BARS} баров)",
              fontsize=12, fontweight="bold")

for ci, ax in enumerate(axes):
    tail = COMP6_sber[ci, -PLOT_BARS:]
    e_frac = diag_6[ci]["energy_frac"] * 100
    ax.plot(xrange, tail, color=CMAP_6[ci], linewidth=0.7, alpha=0.9)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylabel(f"C{ci}", fontsize=9, rotation=0, labelpad=28)
    ax.yaxis.set_label_coords(-0.04, 0.5)
    ax.tick_params(axis="y", labelsize=7)
    info = (f"E={e_frac:.1f}%  H={diag_6[ci]['hurst']:.3f}  "
            f"PE={diag_6[ci]['pe']:.3f}  ADF_p={diag_6[ci]['adf_p']:.3f}  "
            f"DomPer={diag_6[ci]['dom_period']:.1f}  {LABELS_6[ci]}")
    ax.text(0.01, 0.92, info, transform=ax.transAxes, fontsize=7.5,
            va="top", color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.grid(alpha=0.25, axis="y")

axes[-1].set_xlabel("Баров от конца", fontsize=9)
plt.tight_layout()
out1 = OUT_DIR / "42_components_6band.png"
fig1.savefig(out1, dpi=130, bbox_inches="tight")
print(f"  Рис 1: {out1}")
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
#  ФИГУРА 2: Компоненты во времени (SBER, 12-band)
# ══════════════════════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(N_BANDS_12, 1, figsize=(14, 18), sharex=True)
fig2.suptitle(f"Filter bank 12-band — компоненты dratio  |  SBER 1d  (последние {PLOT_BARS} баров)",
              fontsize=12, fontweight="bold")

for ci, ax in enumerate(axes2):
    tail = COMP12_sber[ci, -PLOT_BARS:]
    e_frac = diag_12[ci]["energy_frac"] * 100
    ax.plot(xrange, tail, color=CMAP_12[ci], linewidth=0.6, alpha=0.9)
    ax.axhline(0, color="gray", lw=0.4, ls="--")
    ax.set_ylabel(f"C{ci}", fontsize=8, rotation=0, labelpad=28)
    ax.yaxis.set_label_coords(-0.04, 0.5)
    ax.tick_params(axis="y", labelsize=6)
    info = (f"E={e_frac:.1f}%  H={diag_12[ci]['hurst']:.3f}  "
            f"PE={diag_12[ci]['pe']:.3f}  {LABELS_12[ci]}")
    ax.text(0.01, 0.88, info, transform=ax.transAxes, fontsize=7,
            va="top", color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.grid(alpha=0.2, axis="y")

axes2[-1].set_xlabel("Баров от конца", fontsize=9)
plt.tight_layout()
out2 = OUT_DIR / "42_components_12band.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"  Рис 2: {out2}")
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
#  ФИГУРА 3: Диагностические метрики — сравнение 6-band vs 12-band
# ══════════════════════════════════════════════════════════════════════════════

fig3, axes3 = plt.subplots(2, 4, figsize=(16, 7))
fig3.suptitle("Диагностические метрики (медиана по 8 тикерам, 1d)",
              fontsize=12, fontweight="bold")

def _bar_row(ax, diag, n_bands, key, ylabel, cmap_colors, ylim=None, hline=None, logscale=False):
    vals = [d[key] for d in diag]
    x = np.arange(n_bands)
    colors = cmap_colors[:n_bands]
    bars = ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(n_bands)], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)
    if hline is not None:
        ax.axhline(hline, color="red", lw=1, ls="--", alpha=0.7)
    if logscale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

# ── строка 1: 6-band ──────────────────────────────────────────────────────────
axes3[0, 0].set_title("6-band: Доля энергии %", fontsize=9)
_bar_row(axes3[0, 0], diag_6, N_BANDS_6, "energy_frac", "Доля", CMAP_6)
axes3[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

axes3[0, 1].set_title("6-band: Хёрст R/S", fontsize=9)
_bar_row(axes3[0, 1], diag_6, N_BANDS_6, "hurst", "H", CMAP_6, ylim=(0, 1), hline=0.5)

axes3[0, 2].set_title("6-band: PE (порядок 5)", fontsize=9)
_bar_row(axes3[0, 2], diag_6, N_BANDS_6, "pe", "PE", CMAP_6, ylim=(0, 1.05), hline=1.0)

axes3[0, 3].set_title("6-band: ADF p-value", fontsize=9)
_bar_row(axes3[0, 3], diag_6, N_BANDS_6, "adf_p", "p", CMAP_6, hline=0.05)
axes3[0, 3].axhline(0.05, color="red", lw=1, ls="--", label="α=0.05")

# ── строка 2: 12-band ─────────────────────────────────────────────────────────
axes3[1, 0].set_title("12-band: Доля энергии %", fontsize=9)
_bar_row(axes3[1, 0], diag_12, N_BANDS_12, "energy_frac", "Доля", CMAP_12)
axes3[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))

axes3[1, 1].set_title("12-band: Хёрст R/S", fontsize=9)
_bar_row(axes3[1, 1], diag_12, N_BANDS_12, "hurst", "H", CMAP_12, ylim=(0, 1), hline=0.5)

axes3[1, 2].set_title("12-band: PE (порядок 5)", fontsize=9)
_bar_row(axes3[1, 2], diag_12, N_BANDS_12, "pe", "PE", CMAP_12, ylim=(0, 1.05))

axes3[1, 3].set_title("12-band: ADF p-value", fontsize=9)
_bar_row(axes3[1, 3], diag_12, N_BANDS_12, "adf_p", "p", CMAP_12, hline=0.05)

plt.tight_layout()
out3 = OUT_DIR / "42_diagnostics_metrics.png"
fig3.savefig(out3, dpi=130, bbox_inches="tight")
print(f"  Рис 3: {out3}")
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
#  ФИГУРА 4: ACF по компонентам (SBER)
# ══════════════════════════════════════════════════════════════════════════════

fig4, axes4 = plt.subplots(2, N_BANDS_6, figsize=(16, 6))
fig4.suptitle("ACF компонент  |  SBER 1d  (медиана по 8 тикерам)",
              fontsize=11, fontweight="bold")
lags_arr = np.arange(ACF_LAGS + 1)
conf95 = 1.96 / np.sqrt(len(dratio_sber))   # 95% доверительный интервал

for ci in range(N_BANDS_6):
    ax_top = axes4[0, ci]
    acf_arr = diag_6[ci]["acf_arr"]
    ax_top.bar(lags_arr[1:], acf_arr[1:], color=CMAP_6[ci], alpha=0.7, width=1.0)
    ax_top.axhline(conf95,  color="gray", lw=0.8, ls="--")
    ax_top.axhline(-conf95, color="gray", lw=0.8, ls="--")
    ax_top.axhline(0, color="black", lw=0.5)
    ax_top.set_title(f"C{ci}: {LABELS_6[ci]}", fontsize=8)
    ax_top.set_ylim(-0.5, 0.8)
    ax_top.set_xlabel("Lag", fontsize=7)
    ax_top.tick_params(labelsize=7)
    ax_top.text(0.98, 0.95, f"ρ[1]={diag_6[ci]['acf1']:.3f}", transform=ax_top.transAxes,
                ha="right", va="top", fontsize=7)

# нижняя строка: 12-band, каждые 2 компоненты в один (усредняем пары для 12→6)
for ci in range(N_BANDS_6):
    ax_bot = axes4[1, ci]
    # Берём пару компонент 12-band для сравнения
    ci12a = ci * 2
    ci12b = min(ci * 2 + 1, N_BANDS_12 - 1)
    acf_a = diag_12[ci12a]["acf_arr"]
    acf_b = diag_12[ci12b]["acf_arr"]
    ax_bot.bar(lags_arr[1:], acf_a[1:], color=CMAP_12[ci12a], alpha=0.5, width=1.0,
               label=f"C{ci12a}")
    ax_bot.bar(lags_arr[1:], acf_b[1:], color=CMAP_12[ci12b], alpha=0.5, width=1.0,
               label=f"C{ci12b}")
    ax_bot.axhline(conf95,  color="gray", lw=0.8, ls="--")
    ax_bot.axhline(-conf95, color="gray", lw=0.8, ls="--")
    ax_bot.axhline(0, color="black", lw=0.5)
    ax_bot.set_title(f"12b C{ci12a}/{ci12b}: {LABELS_12[ci12a]}/{LABELS_12[ci12b]}", fontsize=7)
    ax_bot.set_ylim(-0.5, 0.8)
    ax_bot.set_xlabel("Lag", fontsize=7)
    ax_bot.tick_params(labelsize=7)
    ax_bot.legend(fontsize=6, loc="upper right")

axes4[0, 0].set_ylabel("ACF (6-band)", fontsize=8)
axes4[1, 0].set_ylabel("ACF (12-band пары)", fontsize=8)
plt.tight_layout()
out4 = OUT_DIR / "42_acf.png"
fig4.savefig(out4, dpi=130, bbox_inches="tight")
print(f"  Рис 4: {out4}")
plt.close(fig4)


# ══════════════════════════════════════════════════════════════════════════════
#  ФИГУРА 5: Предсказуемость по горизонту (Gain = zero/lwr)
# ══════════════════════════════════════════════════════════════════════════════

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 6))
fig5.suptitle(f"Gain предсказуемости LWR vs zero-forecast  |  SBER 1d  "
              f"(p={P_LWR}, ξ={XI_LWR}, N={N_ORIGINS_WF} origins)",
              fontsize=11, fontweight="bold")

h_arr = np.array(HORIZONS)
LINESTYLES = ["-", "--", "-.", ":", (0,(3,1,1,1)), "-"]

for ci in range(N_BANDS_6):
    gains = [pred_6["zero"][ci][h] / (pred_6["lwr"][ci][h] + EPS) for h in HORIZONS]
    ax5a.plot(h_arr, gains, color=CMAP_6[ci], lw=2.0,
              ls=LINESTYLES[ci % len(LINESTYLES)],
              marker="o", markersize=4, label=f"C{ci}: {LABELS_6[ci]}")

ax5a.axhline(1.0, color="black", lw=1.5, ls="--", label="Gain=1 (ничья с нулём)")
ax5a.set_xlabel("Горизонт (баров)", fontsize=10)
ax5a.set_ylabel("Gain = zero_MAE / lwr_MAE", fontsize=10)
ax5a.set_title("6-band: Gain по компонентам и горизонтам", fontsize=10)
ax5a.legend(fontsize=8, loc="upper right")
ax5a.grid(alpha=0.35)
ax5a.set_xticks(HORIZONS)

for ci in range(N_BANDS_12):
    gains = [pred_12["zero"][ci][h] / (pred_12["lwr"][ci][h] + EPS) for h in HORIZONS]
    ax5b.plot(h_arr, gains, color=CMAP_12[ci], lw=1.6,
              ls=LINESTYLES[ci % len(LINESTYLES)],
              marker="s", markersize=3, label=f"C{ci}: {LABELS_12[ci]}")

ax5b.axhline(1.0, color="black", lw=1.5, ls="--")
ax5b.set_xlabel("Горизонт (баров)", fontsize=10)
ax5b.set_ylabel("Gain = zero_MAE / lwr_MAE", fontsize=10)
ax5b.set_title("12-band: Gain по компонентам и горизонтам", fontsize=10)
ax5b.legend(fontsize=7, loc="upper right", ncol=2)
ax5b.grid(alpha=0.35)
ax5b.set_xticks(HORIZONS)

plt.tight_layout()
out5 = OUT_DIR / "42_predictability_gain.png"
fig5.savefig(out5, dpi=130, bbox_inches="tight")
print(f"  Рис 5: {out5}")
plt.close(fig5)


# ══════════════════════════════════════════════════════════════════════════════
#  ФИГУРА 6: Welch PSD — спектральный анализ компонент (SBER, 6-band)
# ══════════════════════════════════════════════════════════════════════════════

fig6, axes6 = plt.subplots(2, N_BANDS_6, figsize=(16, 7))
fig6.suptitle("Power spectral density (Welch)  |  SBER 1d",
              fontsize=11, fontweight="bold")

for ci in range(N_BANDS_6):
    # 6-band
    ax_top = axes6[0, ci]
    s6 = COMP6_sber[ci]
    nperseg6 = min(len(s6) // 4, 512)
    freqs6, psd6 = sp_welch(s6, fs=1.0, nperseg=nperseg6)
    mask6 = freqs6 > 0
    periods6 = 1.0 / freqs6[mask6]
    ax_top.semilogy(periods6, psd6[mask6], color=CMAP_6[ci], lw=1.5)
    ax_top.set_title(f"C{ci} (6-band)\n{LABELS_6[ci]}", fontsize=8)
    ax_top.set_xlabel("Период (баров)", fontsize=7)
    ax_top.tick_params(labelsize=7)
    ax_top.grid(alpha=0.3)
    ax_top.set_xlim(2, 300)
    # отмечаем доминантный период
    dp = diag_6[ci]["dom_period"]
    if not np.isnan(dp) and 2 <= dp <= 300:
        ax_top.axvline(dp, color="red", lw=1, ls="--", alpha=0.8)
        ax_top.text(dp, ax_top.get_ylim()[1], f"{dp:.0f}", color="red", fontsize=7,
                    ha="center", va="bottom")

    # 12-band (пара компонент)
    ax_bot = axes6[1, ci]
    ci12a = ci * 2
    ci12b = min(ci * 2 + 1, N_BANDS_12 - 1)
    for ci12, ls12 in [(ci12a, "-"), (ci12b, "--")]:
        s12 = COMP12_sber[ci12]
        nperseg12 = min(len(s12) // 4, 512)
        freqs12, psd12 = sp_welch(s12, fs=1.0, nperseg=nperseg12)
        mask12 = freqs12 > 0
        periods12 = 1.0 / freqs12[mask12]
        ax_bot.semilogy(periods12, psd12[mask12], color=CMAP_12[ci12], lw=1.3,
                        ls=ls12, label=f"C{ci12}:{LABELS_12[ci12]}")
    ax_bot.set_title(f"C{ci12a}/{ci12b} (12-band)", fontsize=8)
    ax_bot.set_xlabel("Период (баров)", fontsize=7)
    ax_bot.tick_params(labelsize=7)
    ax_bot.legend(fontsize=6)
    ax_bot.grid(alpha=0.3)
    ax_bot.set_xlim(2, 300)

axes6[0, 0].set_ylabel("PSD (6-band)", fontsize=8)
axes6[1, 0].set_ylabel("PSD (12-band пары)", fontsize=8)
plt.tight_layout()
out6 = OUT_DIR / "42_power_spectrum.png"
fig6.savefig(out6, dpi=130, bbox_inches="tight")
print(f"  Рис 6: {out6}")
plt.close(fig6)

print("\n" + "=" * 72)
print("Готово. Файлы:")
for out in [out1, out2, out3, out4, out5, out6]:
    print(f"  {out.name}")
print("=" * 72)
