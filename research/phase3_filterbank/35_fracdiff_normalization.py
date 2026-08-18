"""
35 — Fractional Differencing как нормализация перед filter bank.

Текущая схема (скр.34):
    ratio = close / SMA(1000)
    dratio = np.diff(ratio)           ← d=1, полное дифференцирование
    filter bank(dratio) → LA/LWR

Новая схема (этот скрипт):
    ratio = close / SMA(1000)
    fd = fracdiff(ratio, d_min)       ← d<1, сохраняет долгосрочную память
    filter bank(fd) → LA/LWR
    → ratio_hat (рекуррентное обращение fracdiff)

d_min — минимальное d ∈ {0.1..1.0} при котором ADF(fd) отвергает H0 (p<0.05).
Если ratio уже стационарен при d=0, d_min≈0 → fd≈ratio, filter bank работает
прямо с ratio (без дифференцирования).

Три режима:
  dratio   — baseline скр.34 (d=1)
  fd_dmin  — fracdiff(ratio, d_min), d ищется через ADF
  fd_fixed — fracdiff(ratio, d=0.5), фиксированное d для сравнения

Параметры те же, что в скр.34: p=20, xi=63, val_h=10, N_eval=200, 8 тикеров.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
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
TICKERS   = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL  = "1d"
MA_WINDOW = 1000
P         = 20
XI        = 3 * (P + 1)   # 63
VAL_H     = 10
N_EVAL    = 200

FILTER_ORDER = 4
CUTOFFS      = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
SLOW_IDX     = [3, 4, 5]

D_GRID       = np.round(np.arange(0.1, 1.01, 0.1), 2)
D_FIXED      = 0.5
FD_THRESHOLD = 1e-4   # обрезать хвост весов


# ── fractional differencing ───────────────────────────────────────────────────
def _fd_weights(d: float, threshold: float = FD_THRESHOLD) -> np.ndarray:
    """Веса дробного дифференцирования по биному Ньютона."""
    w = [1.0]
    k = 1
    while True:
        w_next = -w[-1] * (d - k + 1) / k
        if abs(w_next) < threshold:
            break
        w.append(w_next)
        k += 1
    return np.array(w)


def fracdiff(series: np.ndarray, d: float, threshold: float = FD_THRESHOLD) -> np.ndarray:
    """
    Causal fractional differencing.
    Первые (L-1) элементов — NaN (недостаточно истории).
    """
    w = _fd_weights(d, threshold)
    L = len(w)
    n = len(series)
    result = np.full(n, np.nan)
    for t in range(L - 1, n):
        chunk = series[max(0, t - L + 1): t + 1][::-1]
        result[t] = float(np.dot(w[: len(chunk)], chunk))
    return result


def find_min_d(ratio: np.ndarray, d_grid: np.ndarray = D_GRID) -> float:
    """
    Минимальное d из d_grid при котором ADF(fracdiff(ratio, d)) отвергает H0 (p<0.05).
    Возвращает 1.0 если ни одно d не даёт стационарности.
    """
    for d in d_grid:
        fd = fracdiff(ratio, d)
        fd_clean = fd[~np.isnan(fd)]
        if len(fd_clean) < 30:
            continue
        _, pval, *_ = adfuller(fd_clean, maxlag=5, regression="ct", autolag=None)
        if pval < 0.05:
            return float(d)
    return 1.0


def _reconstruct_ratio(
    fd_hat: np.ndarray,
    ratio_hist: np.ndarray,
    d: float,
    threshold: float = FD_THRESHOLD,
) -> np.ndarray:
    """
    Рекуррентное восстановление ratio из прогнозированного fd.

    fd[t] = Σ_k w_k * ratio[t-k]  →  ratio[t] = fd[t] - Σ_{k≥1} w_k * ratio[t-k]

    ratio_hist — вся история ratio до origin включительно.
    fd_hat     — прогноз fd на horizon шагов вперёд.
    """
    w = _fd_weights(d, threshold)
    horizon = len(fd_hat)
    ratio_ext = list(ratio_hist.copy())
    ratio_hat = np.empty(horizon)
    for h in range(horizon):
        # Сколько исторических значений нам нужно (w[1], w[2], ...)
        n_needed = len(w) - 1
        tail = ratio_ext[-(n_needed):]  # последние n_needed значений
        correction = sum(
            w[k] * tail[len(tail) - k]
            for k in range(1, min(len(w), len(tail) + 1))
        )
        val = fd_hat[h] - correction
        ratio_hat[h] = val
        ratio_ext.append(val)
    return ratio_hat


# ── filter bank ───────────────────────────────────────────────────────────────
def make_filter_bank(series: np.ndarray) -> np.ndarray:
    """Causal Butterworth filter bank, возвращает (6, N)."""
    components = []
    remaining  = series.copy()
    for fc in CUTOFFS:
        sos  = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low  = sosfilt(sos, remaining)
        components.append(remaining - low)
        remaining = low
    components.append(remaining)
    return np.array(components)


# ── LWR step ──────────────────────────────────────────────────────────────────
def _lwr_forecast(X: np.ndarray, y: np.ndarray, vec: np.ndarray, xi: int) -> np.ndarray:
    v   = vec.copy()
    hat = np.empty(VAL_H)
    for h in range(VAL_H):
        dists  = np.linalg.norm(X - v, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        h_bw   = max(float(dists[nn_idx].max()), 1e-10)
        w      = np.exp(-0.5 * (dists[nn_idx] / h_bw) ** 2)
        A      = np.hstack([np.ones((xi, 1)), X[nn_idx]])
        sw     = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx], rcond=None)
        hat[h] = float(c[0] + v @ c[1:])
        v = np.roll(v, -1)
        v[-1] = hat[h]
    return hat


# ── метрики ───────────────────────────────────────────────────────────────────
def _mape_from_dratio(hat: np.ndarray, ratio: np.ndarray, vo: int) -> float:
    """MAPE для dratio-режима (hat = прогноз dratio, восстанавливаем через cumsum)."""
    r0     = float(ratio[vo + 1])   # в dratio-индексах vo соответствует ratio[vo+1]
    r_hat  = r0 + np.cumsum(hat)
    actual = ratio[vo + 2: vo + 2 + VAL_H]
    n_     = min(len(r_hat), len(actual))
    if n_ == 0:
        return np.nan
    return float(np.mean(np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)))


def _mape_from_fd(
    hat: np.ndarray,
    ratio: np.ndarray,
    vo: int,
    d: float,
) -> float:
    """MAPE для fd-режима (hat = прогноз fd, восстанавливаем через обратный fracdiff)."""
    ratio_hist = ratio[: vo + 1]   # история до vo включительно
    r_hat  = _reconstruct_ratio(hat, ratio_hist, d)
    actual = ratio[vo + 1: vo + 1 + VAL_H]
    n_     = min(len(r_hat), len(actual))
    if n_ == 0:
        return np.nan
    return float(np.mean(np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)))


# ── основной цикл ─────────────────────────────────────────────────────────────
RES = {}

print(f"{'Тикер':>5}  {'d_min':>6}  {'Δ%(d_min)':>11}  {'Δ%(d=0.5)':>11}  "
      f"{'base':>8}  {'fd_min':>8}  {'fd_0.5':>8}  t")
print("─" * 90)

t_total = time.time()

for ticker in TICKERS:
    with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
        candles = json.load(f)
    df     = normalize(candles, window=MA_WINDOW).dropna(subset=["ma"]).reset_index(drop=True)
    ratio  = df["ratio"].values          # (N,)
    dratio = np.diff(ratio)              # (N-1,) baseline режим

    # Найти d_min на первых 2/3 данных (train)
    train_end  = int(len(ratio) * 0.66)
    d_min      = find_min_d(ratio[:train_end])

    # Предвычислить fd для двух d
    fd_dmin  = fracdiff(ratio, d_min)    # (N,), первые L-1 NaN
    fd_fixed = fracdiff(ratio, D_FIXED)  # (N,)

    # Filter bank один раз для всего ряда (causal → нет leak)
    COMP_BASE  = make_filter_bank(dratio)
    COMP_FDMIN = make_filter_bank(np.nan_to_num(fd_dmin))    # NaN → 0 в начале
    COMP_FDF   = make_filter_bank(np.nan_to_num(fd_fixed))

    # Точки origin
    # dratio: индексы в dratio-пространстве (0..N-2)
    # fd: индексы в ratio-пространстве (0..N-1)
    fd_L   = len(_fd_weights(d_min))
    min_orig_base = P + XI + 5
    min_orig_fd   = max(P + XI + 5, fd_L + P + 5)
    max_orig      = len(dratio) - VAL_H - 1   # оставляем запас для оба режима

    origins = np.arange(max(min_orig_base, min_orig_fd, max_orig - N_EVAL), max_orig)

    raw_base, raw_fdmin, raw_fdf = [], [], []
    t1 = time.time()

    for vo in origins:
        # ── baseline: dratio + filter bank ───────────────────────────────────
        hats_b = []
        for ci in SLOW_IDX:
            Xc, yc = build_delay_matrix(COMP_BASE[ci, :vo], P)
            if len(Xc) < XI:
                hats_b.append(np.zeros(VAL_H)); continue
            vecc = last_vector(COMP_BASE[ci, :vo], p=P).copy()
            hats_b.append(_lwr_forecast(Xc, yc, vecc, XI))
        raw_base.append(_mape_from_dratio(np.sum(hats_b, axis=0), ratio, vo))

        # ── fd_dmin: fracdiff(ratio, d_min) + filter bank ────────────────────
        # vo здесь — индекс в ratio-пространстве
        vo_r = vo + 1   # ratio[vo_r] = origin в ratio-пространстве
        hats_m = []
        for ci in SLOW_IDX:
            Xc, yc = build_delay_matrix(COMP_FDMIN[ci, :vo_r], P)
            if len(Xc) < XI:
                hats_m.append(np.zeros(VAL_H)); continue
            vecc = last_vector(COMP_FDMIN[ci, :vo_r], p=P).copy()
            hats_m.append(_lwr_forecast(Xc, yc, vecc, XI))
        raw_fdmin.append(_mape_from_fd(np.sum(hats_m, axis=0), ratio, vo_r, d_min))

        # ── fd_fixed: fracdiff(ratio, 0.5) + filter bank ─────────────────────
        hats_f = []
        for ci in SLOW_IDX:
            Xc, yc = build_delay_matrix(COMP_FDF[ci, :vo_r], P)
            if len(Xc) < XI:
                hats_f.append(np.zeros(VAL_H)); continue
            vecc = last_vector(COMP_FDF[ci, :vo_r], p=P).copy()
            hats_f.append(_lwr_forecast(Xc, yc, vecc, XI))
        raw_fdf.append(_mape_from_fd(np.sum(hats_f, axis=0), ratio, vo_r, D_FIXED))

    base_arr  = np.array(raw_base)
    fdmin_arr = np.array(raw_fdmin)
    fdf_arr   = np.array(raw_fdf)

    bm  = float(np.nanmedian(base_arr))
    mm  = float(np.nanmedian(fdmin_arr))
    fm  = float(np.nanmedian(fdf_arr))
    d_m = (mm - bm) / bm * 100
    d_f = (fm - bm) / bm * 100

    RES[ticker] = {
        "base":     base_arr,
        "fd_dmin":  fdmin_arr,
        "fd_fixed": fdf_arr,
        "d_min":    d_min,
        "bm": bm, "mm": mm, "fm": fm,
    }
    print(f"{ticker:>5}  {d_min:>6.2f}  {d_m:>+10.1f}%  {d_f:>+10.1f}%  "
          f"{bm:>8.5f}  {mm:>8.5f}  {fm:>8.5f}  {time.time()-t1:.1f}s")

print(f"\nTotal: {time.time()-t_total:.1f}s")


# ── сводная статистика ────────────────────────────────────────────────────────
all_base  = np.concatenate([RES[t]["base"]     for t in TICKERS])
all_fdmin = np.concatenate([RES[t]["fd_dmin"]  for t in TICKERS])
all_fdf   = np.concatenate([RES[t]["fd_fixed"] for t in TICKERS])

ok_m = ~(np.isnan(all_base) | np.isnan(all_fdmin))
ok_f = ~(np.isnan(all_base) | np.isnan(all_fdf))

agg_b  = float(np.nanmedian(all_base))
agg_m  = float(np.nanmedian(all_fdmin))
agg_f  = float(np.nanmedian(all_fdf))

_, pv_m = _wilcox(all_fdmin[ok_m] - all_base[ok_m])
_, pv_f = _wilcox(all_fdf[ok_f]   - all_base[ok_f])

print("\n" + "=" * 70)
print(f"Aggregate (медиана по всем тикерам):")
print(f"  baseline dratio+fb : {agg_b:.5f}")
print(f"  fd(d_min)+fb       : {agg_m:.5f}  "
      f"Δ={(agg_m-agg_b)/agg_b*100:+.2f}%  Wilcoxon p={pv_m:.4f}")
print(f"  fd(d=0.5)+fb       : {agg_f:.5f}  "
      f"Δ={(agg_f-agg_b)/agg_b*100:+.2f}%  Wilcoxon p={pv_f:.4f}")
print("=" * 70)

# d_min по тикерам
print("\nd_min по тикерам:")
for t in TICKERS:
    print(f"  {t:>5}: d_min={RES[t]['d_min']:.2f}")


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(
    f"Fractional Differencing vs dratio + filter bank  |  p={P}  val_h={VAL_H}",
    fontsize=11,
)

# 1. Δ% по тикерам
ax = axes[0]
x  = np.arange(len(TICKERS))
w  = 0.28
deltas_m = [(RES[t]["mm"] - RES[t]["bm"]) / RES[t]["bm"] * 100 for t in TICKERS]
deltas_f = [(RES[t]["fm"] - RES[t]["bm"]) / RES[t]["bm"] * 100 for t in TICKERS]
cm = ["seagreen" if d < 0 else "tomato" for d in deltas_m]
cf = ["royalblue" if d < 0 else "orange"  for d in deltas_f]
ax.bar(x - w, deltas_m, w, color=cm, alpha=0.85, label="fd(d_min)+fb")
ax.bar(x,     deltas_f, w, color=cf, alpha=0.6,  label="fd(d=0.5)+fb")
ax.axhline(0,                                color="black", lw=1)
ax.axhline((agg_m - agg_b) / agg_b * 100,   color="seagreen", lw=1.5, ls="--",
           label=f"aggr d_min {(agg_m-agg_b)/agg_b*100:+.1f}%")
ax.axhline((agg_f - agg_b) / agg_b * 100,   color="royalblue", lw=1.5, ls="--",
           label=f"aggr d=0.5 {(agg_f-agg_b)/agg_b*100:+.1f}%")
ax.set_xticks(x); ax.set_xticklabels(TICKERS, fontsize=9)
ax.set_ylabel("Δ%  (отриц. = лучше)")
ax.set_title("Δ% vs baseline (dratio+fb)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

# 2. CDF aggregate
ax = axes[1]
s_b = np.sort(all_base[~np.isnan(all_base)])
s_m = np.sort(all_fdmin[~np.isnan(all_fdmin)])
s_f = np.sort(all_fdf[~np.isnan(all_fdf)])
pct = np.linspace(0, 100, len(s_b))
ax.plot(np.linspace(0, 100, len(s_b)), s_b, "steelblue", lw=2.5,
        label=f"dratio+fb  (med={agg_b:.5f})")
ax.plot(np.linspace(0, 100, len(s_m)), s_m, "seagreen",  lw=2,
        label=f"fd(d_min)+fb (med={agg_m:.5f}  Δ={(agg_m-agg_b)/agg_b*100:+.1f}%)")
ax.plot(np.linspace(0, 100, len(s_f)), s_f, "darkorange", lw=1.5, ls="--",
        label=f"fd(d=0.5)+fb (med={agg_f:.5f}  Δ={(agg_f-agg_b)/agg_b*100:+.1f}%)")
ax.set_title(f"CDF aggregate  N={len(s_b)}")
ax.set_xlabel("Перцентиль"); ax.set_ylabel("val_mape")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0)

# 3. d_min по тикерам
ax = axes[2]
d_vals = [RES[t]["d_min"] for t in TICKERS]
colors = ["seagreen" if (RES[t]["mm"] - RES[t]["bm"]) < 0 else "tomato" for t in TICKERS]
ax.bar(TICKERS, d_vals, color=colors, alpha=0.85)
ax.axhline(1.0, color="gray",   lw=1.2, ls="--", label="d=1 (полное diff)")
ax.axhline(0.5, color="orange", lw=1.2, ls=":",  label="d=0.5 (fixed)")
ax.set_ylabel("d_min (ADF)")
ax.set_title("Оптимальное d по тикерам\n(зелёный = fd лучше baseline)")
ax.set_ylim(0, 1.15)
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_path = OUT_DIR / "35_fracdiff_normalization.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
