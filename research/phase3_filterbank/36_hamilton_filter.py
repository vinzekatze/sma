"""
36 — Hamilton Filter как улучшенная оценка тренда перед filter bank.

Текущая схема (baseline):
    ratio = close / SMA(1000) → dratio = diff(ratio) → filter bank → LA/LWR

Новая схема:
    ratio = close / SMA(1000)
    htrend(ratio, h) = OLS(ratio[t] on ratio[t-h..t-h-3])  ← медленный тренд
    hcycle(ratio, h) = ratio - htrend                       ← стационарный цикл

    Режим A (drcycle): diff(hcycle) → filter bank → cumsum → cycle_hat + htrend[origin]
    Режим B (rcycle):  hcycle → filter bank → cycle_hat + htrend[origin]

В режиме B дополнительное дифференцирование не нужно — цикл уже центрирован и
стационарен по построению (OLS остаток). Это сохраняет больше структуры для LA.

Параметры Hamilton: h ∈ {5, 10, 20} — недельный, двухнедельный, месячный лаг (1d).
Реализация: global OLS на всём ряду (лаги смотрят в прошлое → causal по форме).

Сравнение на 8 тикерах 1d, p=20, xi=63, val_h=10, N_eval=200.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.stats import wilcoxon as _wilcox

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from sma.core.forecast.normalize import normalize
from sma.core.forecast.embedding import build_delay_matrix, last_vector

# ── параметры ─────────────────────────────────────────────────────────────────
TICKERS  = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL = "1d"
MA_WIN   = 1000
P        = 20
XI       = 3 * (P + 1)   # 63
VAL_H    = 10
N_EVAL   = 200

FILTER_ORDER = 4
CUTOFFS      = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
SLOW_IDX     = [3, 4, 5]

H_VALUES = [5, 10, 20]    # лаги Hamilton filter
N_LAGS   = 4              # число регрессоров (как в оригинальной статье)


# ── Hamilton filter ───────────────────────────────────────────────────────────
def hamilton_filter(series: np.ndarray, h: int, n_lags: int = N_LAGS):
    """
    Extracts trend and cycle via Hamilton (2018) lag regression.

    trend[t] = alpha + sum(beta_k * series[t-h-k], k=0..n_lags-1)
    cycle[t] = series[t] - trend[t]

    Global OLS — beta fitted on the full series (lags look only backward).
    Returns: (trend, cycle), both NaN for first h+n_lags-1 bars.
    """
    n      = len(series)
    t_min  = h + n_lags - 1   # первый t с полными лагами

    # Матрица регрессоров для наблюдений t = t_min..n-1
    t_range = np.arange(t_min, n)
    X = np.column_stack(
        [np.ones(len(t_range))]
        + [series[t_range - h - k] for k in range(n_lags)]
    )
    y = series[t_range]

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    trend = np.full(n, np.nan)
    for idx, t in enumerate(t_range):
        trend[t] = float(X[idx] @ beta)

    cycle = series - trend
    return trend, cycle


# ── filter bank ───────────────────────────────────────────────────────────────
def make_fb(series: np.ndarray) -> np.ndarray:
    components = []; remaining = series.copy()
    for fc in CUTOFFS:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        components.append(remaining - low); remaining = low
    components.append(remaining)
    return np.array(components)


# ── LWR ──────────────────────────────────────────────────────────────────────
def _lwr(X, y, vec, xi):
    v = vec.copy(); hat = np.empty(VAL_H)
    for h in range(VAL_H):
        dists  = np.linalg.norm(X - v, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        h_bw   = max(float(dists[nn_idx].max()), 1e-10)
        w      = np.exp(-0.5 * (dists[nn_idx] / h_bw) ** 2)
        A      = np.hstack([np.ones((xi, 1)), X[nn_idx]])
        sw     = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx], rcond=None)
        hat[h] = float(c[0] + v @ c[1:])
        v = np.roll(v, -1); v[-1] = hat[h]
    return hat


# ── метрики ───────────────────────────────────────────────────────────────────
def _fb_forecast(COMP, vo, p, xi):
    """Суммарный LWR прогноз по медленным компонентам."""
    hats = []
    for ci in SLOW_IDX:
        Xc, yc = build_delay_matrix(COMP[ci, :vo], p)
        if len(Xc) < xi:
            hats.append(np.zeros(VAL_H)); continue
        vecc = last_vector(COMP[ci, :vo], p=p).copy()
        hats.append(_lwr(Xc, yc, vecc, xi))
    return np.sum(hats, axis=0)


def mape_dratio(hat, ratio, vo):
    """Baseline: dratio режим. hat = dratio прогноз, vo — индекс в dratio."""
    r0     = float(ratio[vo + 1])
    r_hat  = r0 + np.cumsum(hat)
    actual = ratio[vo + 2: vo + 2 + VAL_H]
    n_     = min(len(r_hat), len(actual))
    if n_ == 0: return np.nan
    return float(np.mean(np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)))


def mape_drcycle(hat, ratio, cycle, htrend, vo):
    """
    drcycle режим: hat = diff(cycle) прогноз.
    Восстановление: cycle_hat = cycle[vo+1] + cumsum(hat)
                    ratio_hat = cycle_hat + htrend[vo+1] (замороженный тренд)
    vo — индекс в dratio/drcycle-пространстве.
    """
    c0     = float(cycle[vo + 1])
    c_hat  = c0 + np.cumsum(hat)
    t0     = float(htrend[vo + 1])   # замороженный тренд origin
    r_hat  = c_hat + t0
    actual = ratio[vo + 2: vo + 2 + VAL_H]
    n_     = min(len(r_hat), len(actual))
    if n_ == 0: return np.nan
    return float(np.mean(np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)))


def mape_rcycle(hat, ratio, htrend, vo):
    """
    rcycle режим: hat = cycle прогноз (абсолютный).
    Восстановление: ratio_hat = hat + htrend[vo+1] (замороженный тренд)
    vo — индекс в dratio-пространстве (origin = ratio[vo+1]).
    """
    t0     = float(htrend[vo + 1])
    r_hat  = hat + t0
    actual = ratio[vo + 2: vo + 2 + VAL_H]
    n_     = min(len(r_hat), len(actual))
    if n_ == 0: return np.nan
    return float(np.mean(np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)))


# ── основной цикл ─────────────────────────────────────────────────────────────
# Варианты: baseline + (drcycle, rcycle) × H_VALUES = 7 вариантов
VARIANTS = ["base"] + [f"dr_{h}" for h in H_VALUES] + [f"rc_{h}" for h in H_VALUES]
RES = {t: {v: [] for v in VARIANTS} for t in TICKERS}

print("Запуск... (p=20, xi=63, val_h=10, N=200, 8 тикеров)\n")

for ticker in TICKERS:
    with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
        candles = json.load(f)
    df     = normalize(candles, window=MA_WIN).dropna(subset=["ma"]).reset_index(drop=True)
    ratio  = df["ratio"].values
    dratio = np.diff(ratio)

    t1 = time.time()

    # Baseline filter bank
    COMP_BASE = make_fb(dratio)

    # Hamilton для каждого h
    hamilton = {}
    for h in H_VALUES:
        htrend, hcycle = hamilton_filter(ratio, h)
        dhcycle = np.diff(np.nan_to_num(hcycle))
        hamilton[h] = {
            "trend":   htrend,
            "cycle":   hcycle,
            "dcycle":  dhcycle,
            "COMP_rc": make_fb(np.nan_to_num(hcycle)),
            "COMP_dr": make_fb(dhcycle),
        }

    # Origins (общий min_orig по всем вариантам)
    min_h_nan = max(h + N_LAGS for h in H_VALUES)   # max NaN start из-за Hamilton
    min_orig  = max(P + XI + 5, min_h_nan + P + 5)
    max_orig  = len(dratio) - VAL_H - 1
    origins   = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)

    for vo in origins:
        # ── baseline ─────────────────────────────────────────────────────────
        hat = _fb_forecast(COMP_BASE, vo, P, XI)
        RES[ticker]["base"].append(mape_dratio(hat, ratio, vo))

        # ── Hamilton варианты ─────────────────────────────────────────────────
        for h in H_VALUES:
            htrend = hamilton[h]["trend"]
            hcycle = hamilton[h]["cycle"]
            COMP_dr = hamilton[h]["COMP_dr"]
            COMP_rc = hamilton[h]["COMP_rc"]

            # drcycle: diff(cycle) в filter bank
            hat_dr = _fb_forecast(COMP_dr, vo, P, XI)
            RES[ticker][f"dr_{h}"].append(
                mape_drcycle(hat_dr, ratio, hcycle, htrend, vo)
            )

            # rcycle: cycle напрямую в filter bank (без diff)
            hat_rc = _fb_forecast(COMP_rc, vo + 1, P, XI)
            RES[ticker][f"rc_{h}"].append(
                mape_rcycle(hat_rc, ratio, htrend, vo)
            )

    print(f"{ticker:>5}  N={len(origins)}  "
          + "  ".join(
              f"{v}={np.nanmedian(RES[ticker][v]):.5f}"
              for v in VARIANTS
          )
          + f"  {time.time()-t1:.1f}s")


# ── сводная таблица ───────────────────────────────────────────────────────────
print("\n" + "=" * 95)
header = f"{'':>5}  {'base':>8}" + "".join(f"  {'dr'+str(h):>8}  {'Δ%':>7}  {'rc'+str(h):>8}  {'Δ%':>7}" for h in H_VALUES)
print(header)
print("─" * 95)

agg = {v: np.concatenate([RES[t][v] for t in TICKERS]) for v in VARIANTS}
agg_base = float(np.nanmedian(agg["base"]))

for ticker in TICKERS:
    bm = float(np.nanmedian(RES[ticker]["base"]))
    row = f"{ticker:>5}  {bm:>8.5f}"
    for h in H_VALUES:
        dr = float(np.nanmedian(RES[ticker][f"dr_{h}"]))
        rc = float(np.nanmedian(RES[ticker][f"rc_{h}"]))
        row += f"  {dr:>8.5f}  {(dr-bm)/bm*100:>+6.1f}%  {rc:>8.5f}  {(rc-bm)/bm*100:>+6.1f}%"
    print(row)

print("─" * 95)
row = f"{'AGG':>5}  {agg_base:>8.5f}"
for h in H_VALUES:
    dr_med = float(np.nanmedian(agg[f"dr_{h}"]))
    rc_med = float(np.nanmedian(agg[f"rc_{h}"]))
    ok_dr  = ~(np.isnan(agg["base"]) | np.isnan(agg[f"dr_{h}"]))
    ok_rc  = ~(np.isnan(agg["base"]) | np.isnan(agg[f"rc_{h}"]))
    _, pv_dr = _wilcox(agg[f"dr_{h}"][ok_dr] - agg["base"][ok_dr])
    _, pv_rc = _wilcox(agg[f"rc_{h}"][ok_rc] - agg["base"][ok_rc])
    row += (f"  {dr_med:>8.5f}  {(dr_med-agg_base)/agg_base*100:>+6.1f}%"
            f"  {rc_med:>8.5f}  {(rc_med-agg_base)/agg_base*100:>+6.1f}%")
    print(f"    → Wilcoxon dr_{h}: p={pv_dr:.4f}  rc_{h}: p={pv_rc:.4f}")
print(row)
print("=" * 95)


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    f"Hamilton Filter + filter bank  |  p={P}  val_h={VAL_H}  8 тикеров",
    fontsize=12,
)

for col, h in enumerate(H_VALUES):
    # Верхний ряд: Δ% по тикерам (drcycle vs rcycle)
    ax = axes[0, col]
    x  = np.arange(len(TICKERS))
    w  = 0.35
    deltas_dr = [(np.nanmedian(RES[t][f"dr_{h}"]) - np.nanmedian(RES[t]["base"]))
                 / np.nanmedian(RES[t]["base"]) * 100 for t in TICKERS]
    deltas_rc = [(np.nanmedian(RES[t][f"rc_{h}"]) - np.nanmedian(RES[t]["base"]))
                 / np.nanmedian(RES[t]["base"]) * 100 for t in TICKERS]
    c_dr = ["seagreen" if d < 0 else "tomato" for d in deltas_dr]
    c_rc = ["royalblue" if d < 0 else "orange" for d in deltas_rc]
    ax.bar(x - w/2, deltas_dr, w, color=c_dr, alpha=0.85, label=f"drcycle h={h}")
    ax.bar(x + w/2, deltas_rc, w, color=c_rc, alpha=0.6,  label=f"rcycle  h={h}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(TICKERS, fontsize=8)
    ax.set_ylabel("Δ% vs baseline"); ax.set_title(f"h={h}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Нижний ряд: CDF aggregate
    ax = axes[1, col]
    s_b  = np.sort(agg["base"][~np.isnan(agg["base"])])
    s_dr = np.sort(agg[f"dr_{h}"][~np.isnan(agg[f"dr_{h}"])])
    s_rc = np.sort(agg[f"rc_{h}"][~np.isnan(agg[f"rc_{h}"])])
    dr_m = float(np.nanmedian(agg[f"dr_{h}"]))
    rc_m = float(np.nanmedian(agg[f"rc_{h}"]))
    ax.plot(np.linspace(0, 100, len(s_b)),  s_b,  "steelblue", lw=2.5,
            label=f"baseline ({agg_base:.5f})")
    ax.plot(np.linspace(0, 100, len(s_dr)), s_dr, "seagreen",  lw=2,
            label=f"drcycle ({dr_m:.5f}  {(dr_m-agg_base)/agg_base*100:+.1f}%)")
    ax.plot(np.linspace(0, 100, len(s_rc)), s_rc, "darkorange", lw=1.5, ls="--",
            label=f"rcycle  ({rc_m:.5f}  {(rc_m-agg_base)/agg_base*100:+.1f}%)")
    ax.set_title(f"CDF aggregate  h={h}")
    ax.set_xlabel("Перцентиль"); ax.set_ylabel("val_mape")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0)

plt.tight_layout()
out_path = OUT_DIR / "36_hamilton_filter.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
