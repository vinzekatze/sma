"""
41 — Линейная нормализация vs SMA(1000): влияние на filter bank LWR.

Проблема SMA: разгрев 999 баров → из 1600 доступно только ~600 для delay matrix.
Линейный тренд (causal OLS) требует 2 точки → доступна ВСЯ история.

Методы:
  A  (baseline):  ratio = close / SMA(1000)              → dratio → FB+LWR
  B  (lintrend):  ratio = close / lintrend_causal(close) → dratio → FB+LWR
  C  (logtrend):  ratio = close / exp(logtrend_causal)   → dratio → FB+LWR

Оценка MAPE в пространстве ratio (≈ price MAPE, т.к. ratio ≈ 1).
Сравнение на ОДИНАКОВЫХ origins (там, где работают ВСЕ методы — после SMA-разгрева),
чтобы изолировать качество нормализации от объёма данных.

Параметры: p=20, xi=63, val_h=10, N_EVAL=200, 8 тикеров 1d.
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
EPS          = 1e-10


# ── нормализации ──────────────────────────────────────────────────────────────

def sma_ratio(close: np.ndarray, window: int) -> np.ndarray:
    """ratio = close / SMA(window). NaN для первых window-1 баров."""
    ratio = np.full_like(close, np.nan)
    for i in range(window - 1, len(close)):
        ratio[i] = close[i] / np.mean(close[i - window + 1: i + 1])
    return ratio


def lintrend_causal(close: np.ndarray) -> np.ndarray:
    """
    Causal OLS: trend[t] = a + b*t, fitted на close[:t+1].
    Возвращает trend значения (той же длины, что close).
    Использует инкрементальные суммы → O(N).
    """
    n   = len(close)
    t   = np.arange(n, dtype=np.float64)
    cn  = np.arange(1, n + 1, dtype=np.float64)   # кол-во точек до момента i
    ct  = np.cumsum(t)                              # sum(0..i)
    ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(close)
    cty = np.cumsum(t * close)

    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b     = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a     = (cy - b * ct) / cn
    trend = a + b * t

    # Первые 2 точки (denom=0): подставить close напрямую
    trend[:2] = close[:2]
    return trend


def logtrend_causal(close: np.ndarray) -> np.ndarray:
    """
    Causal OLS в log-пространстве: log(close) = a + b*t.
    Возвращает exp(a + b*t) — экспоненциальный тренд.
    Более подходит для мультипликативной динамики цен.
    """
    log_c = np.log(close + EPS)
    log_t = lintrend_causal(log_c)
    return np.exp(log_t)


# ── filter bank ───────────────────────────────────────────────────────────────

def make_fb(series: np.ndarray) -> np.ndarray:
    components = []; remaining = series.copy()
    for fc in CUTOFFS:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        components.append(remaining - low); remaining = low
    components.append(remaining)
    return np.array(components)   # (6, N)


# ── LWR step ──────────────────────────────────────────────────────────────────

def _lwr(X, y, vec, xi):
    v = vec.copy(); hat = np.empty(VAL_H)
    for step in range(VAL_H):
        dists  = np.linalg.norm(X - v, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        h_bw   = max(float(dists[nn_idx].max()), 1e-10)
        w      = np.exp(-0.5 * (dists[nn_idx] / h_bw) ** 2)
        A      = np.hstack([np.ones((xi, 1)), X[nn_idx]])
        sw     = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx], rcond=None)
        hat[step] = float(c[0] + v @ c[1:])
        v = np.roll(v, -1); v[-1] = hat[step]
    return hat


# ── filter bank LWR forecast ──────────────────────────────────────────────────

def fb_forecast(dratio_hist: np.ndarray) -> np.ndarray:
    """LWR filter bank на dratio_hist. Возвращает dratio_hat длины VAL_H."""
    comp = make_fb(dratio_hist)   # (6, len(dratio_hist))
    hats = []
    for ci in SLOW_IDX:
        Xc, yc = build_delay_matrix(comp[ci], P)
        if len(Xc) < XI:
            hats.append(np.zeros(VAL_H)); continue
        vecc = last_vector(comp[ci], p=P).copy()
        hats.append(_lwr(Xc, yc, vecc, XI))
    return np.sum(hats, axis=0)


# ── MAPE в пространстве ratio ─────────────────────────────────────────────────

def mape_ratio(dratio_hat: np.ndarray, ratio: np.ndarray, origin_k: int) -> float:
    """MAPE между прогнозом ratio и фактическим ratio на горизонте VAL_H."""
    r_hat  = float(ratio[origin_k]) + np.cumsum(dratio_hat)
    actual = ratio[origin_k + 1: origin_k + 1 + VAL_H]
    n_     = min(len(r_hat), len(actual))
    if n_ == 0: return np.nan
    return float(np.mean(np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + EPS)))


# ── основной цикл ─────────────────────────────────────────────────────────────
METHODS = ["sma", "lin", "log"]
RES = {t: {m: [] for m in METHODS} for t in TICKERS}
POOL_SIZES = {t: {m: [] for m in METHODS} for t in TICKERS}

t_total = time.time()
for ticker in TICKERS:
    with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
        candles = json.load(f)
    close = np.array([c["close"] for c in candles], dtype=np.float64)
    N     = len(close)

    # Предвычислить все три ratio
    ratio_sma = sma_ratio(close, MA_WIN)
    trend_lin = lintrend_causal(close)
    ratio_lin = close / trend_lin
    trend_log = logtrend_causal(close)
    ratio_log = close / trend_log

    # dratio для каждого метода
    dratio_sma = np.diff(ratio_sma)   # NaN где ratio_sma = NaN
    dratio_lin = np.diff(ratio_lin)
    dratio_log = np.diff(ratio_log)

    # Origins: последние N_EVAL позиций, где ВСЕ методы работают
    # SMA требует разгрева MA_WIN-1 баров; dratio теряет ещё 1
    # → первый valid origin для sma: MA_WIN - 1 + P + XI + 5
    sma_start  = MA_WIN - 1          # первый valid индекс ratio_sma
    min_origin = sma_start + P + XI + 5
    max_origin = N - VAL_H - 2
    origins    = np.arange(max(min_origin, max_origin - N_EVAL), max_origin)

    t1 = time.time()
    for k in origins:
        # ── SMA ──
        # dratio_sma[i] = ratio_sma[i+1] - ratio_sma[i]
        # история dratio для origin k: индексы sma_start..k-1 (включительно)
        hist_sma = dratio_sma[sma_start: k]
        if len(hist_sma) >= P + XI:
            hat  = fb_forecast(hist_sma)
            mape = mape_ratio(hat, ratio_sma, k)
            RES[ticker]["sma"].append(mape)
            POOL_SIZES[ticker]["sma"].append(len(hist_sma) - P)

        # ── LinTrend ──
        hist_lin = dratio_lin[2: k]   # пропускаем первые 2 нестабильных бара
        if len(hist_lin) >= P + XI:
            hat  = fb_forecast(hist_lin)
            mape = mape_ratio(hat, ratio_lin, k)
            RES[ticker]["lin"].append(mape)
            POOL_SIZES[ticker]["lin"].append(len(hist_lin) - P)

        # ── LogTrend ──
        hist_log = dratio_log[2: k]
        if len(hist_log) >= P + XI:
            hat  = fb_forecast(hist_log)
            mape = mape_ratio(hat, ratio_log, k)
            RES[ticker]["log"].append(mape)
            POOL_SIZES[ticker]["log"].append(len(hist_log) - P)

    med  = {m: float(np.nanmedian(RES[ticker][m])) for m in METHODS}
    pool = {m: int(np.median(POOL_SIZES[ticker][m])) if POOL_SIZES[ticker][m] else 0
            for m in METHODS}
    d_lin = (med["lin"] - med["sma"]) / med["sma"] * 100
    d_log = (med["log"] - med["sma"]) / med["sma"] * 100
    print(f"{ticker:>5}  N={len(origins)}"
          f"  sma={med['sma']:.5f}(pool={pool['sma']})"
          f"  lin={med['lin']:.5f}({d_lin:+.1f}%,pool={pool['lin']})"
          f"  log={med['log']:.5f}({d_log:+.1f}%,pool={pool['log']})"
          f"  {time.time() - t1:.1f}s")

print(f"\nTotal: {time.time() - t_total:.1f}s")


# ── сводная таблица ───────────────────────────────────────────────────────────
print("\n─── Сводная таблица (медиана MAPE) ───")
hdr = f"{'':>5}  {'SMA(1000)':>10}  {'lintrend':>10}  {'logtrend':>10}"
print(hdr); print("─" * len(hdr))

for ticker in TICKERS:
    med = {m: float(np.nanmedian(RES[ticker][m])) for m in METHODS}
    d_l = (med["lin"] - med["sma"]) / med["sma"] * 100
    d_g = (med["log"] - med["sma"]) / med["sma"] * 100
    print(f"{ticker:>5}  {med['sma']:>10.5f}  {med['lin']:>+8.1f}%    {med['log']:>+8.1f}%")

print("─" * len(hdr))
agg = {m: float(np.nanmedian(np.concatenate([RES[t][m] for t in TICKERS])))
       for m in METHODS}
d_l = (agg["lin"] - agg["sma"]) / agg["sma"] * 100
d_g = (agg["log"] - agg["sma"]) / agg["sma"] * 100
print(f"{'AGG':>5}  {agg['sma']:>10.5f}  {agg['lin']:>+8.1f}%    {agg['log']:>+8.1f}%")

# Wilcoxon: lin vs sma
all_sma = np.concatenate([RES[t]["sma"] for t in TICKERS])
all_lin = np.concatenate([RES[t]["lin"] for t in TICKERS])
all_log = np.concatenate([RES[t]["log"] for t in TICKERS])
ok_l = ~(np.isnan(all_sma) | np.isnan(all_lin))
ok_g = ~(np.isnan(all_sma) | np.isnan(all_log))
_, pv_l = _wilcox(all_lin[ok_l] - all_sma[ok_l])
_, pv_g = _wilcox(all_log[ok_g] - all_sma[ok_g])
print(f"\n  lintrend: Δ={d_l:+.2f}%  Wilcoxon p={pv_l:.4f}")
print(f"  logtrend: Δ={d_g:+.2f}%  Wilcoxon p={pv_g:.4f}")


# ── диагностика: статистики ratio ─────────────────────────────────────────────
print("\n─── Диагностика ratio (SBER): mean, std, min, max ───")
ticker = "SBER"
with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
    candles = json.load(f)
close_s   = np.array([c["close"] for c in candles], dtype=np.float64)
rs        = sma_ratio(close_s, MA_WIN)
rs_valid  = rs[~np.isnan(rs)]
rl        = close_s / lintrend_causal(close_s)
rg        = close_s / logtrend_causal(close_s)

print(f"  SMA(1000): mean={rs_valid.mean():.4f}  std={rs_valid.std():.4f}"
      f"  min={rs_valid.min():.4f}  max={rs_valid.max():.4f}")
print(f"  lintrend:  mean={rl.mean():.4f}  std={rl.std():.4f}"
      f"  min={rl.min():.4f}  max={rl.max():.4f}")
print(f"  logtrend:  mean={rg.mean():.4f}  std={rg.std():.4f}"
      f"  min={rg.min():.4f}  max={rg.max():.4f}")

# dratio std по методам
ds   = np.diff(rs_valid)
dl   = np.diff(rl)
dg   = np.diff(rg)
print(f"\n  dratio std: SMA={ds.std():.6f}  lin={dl.std():.6f}  log={dg.std():.6f}")

# slow components std для каждого метода
comp_s = make_fb(ds)
comp_l = make_fb(dl)
comp_g = make_fb(dg)
print(f"\n  Slow C3-C5 std (dratio):")
for ci in SLOW_IDX:
    print(f"    C{ci}: SMA={comp_s[ci].std():.6f}"
          f"  lin={comp_l[ci].std():.6f}"
          f"  log={comp_g[ci].std():.6f}")


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Линейная нормализация vs SMA(1000)  |  p={P}  val_h={VAL_H}  N={N_EVAL}",
    fontsize=12,
)

# 1. Per-ticker MAPE сравнение
ax = axes[0]
x  = np.arange(len(TICKERS))
w  = 0.25
meds_sma = [float(np.nanmedian(RES[t]["sma"])) for t in TICKERS]
meds_lin = [float(np.nanmedian(RES[t]["lin"])) for t in TICKERS]
meds_log = [float(np.nanmedian(RES[t]["log"])) for t in TICKERS]
ax.bar(x - w, meds_sma, w, label="SMA(1000)", color="steelblue")
ax.bar(x,     meds_lin, w, label="lintrend",  color="darkorange")
ax.bar(x + w, meds_log, w, label="logtrend",  color="forestgreen")
ax.set_xticks(x); ax.set_xticklabels(TICKERS, rotation=45, ha="right")
ax.set_ylabel("Медиана MAPE"); ax.set_title("MAPE по тикерам")
ax.legend()

# 2. Δ% от SMA для lin и log
ax = axes[1]
d_lin_list = [(ml - ms) / ms * 100 for ms, ml in zip(meds_sma, meds_lin)]
d_log_list = [(mg - ms) / ms * 100 for ms, mg in zip(meds_sma, meds_log)]
ax.bar(x - w/2, d_lin_list, w, label="lintrend", color="darkorange")
ax.bar(x + w/2, d_log_list, w, label="logtrend", color="forestgreen")
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x); ax.set_xticklabels(TICKERS, rotation=45, ha="right")
ax.set_ylabel("Δ% от SMA(1000)"); ax.set_title("Отклонение от baseline")
ax.legend()

# 3. Ratio динамика SBER (последние 200 баров)
ax = axes[2]
ticker = "SBER"
with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
    candles = json.load(f)
close_s = np.array([c["close"] for c in candles], dtype=np.float64)
rs      = sma_ratio(close_s, MA_WIN);  rs_v = rs[~np.isnan(rs)]
rl      = close_s / lintrend_causal(close_s)
rg      = close_s / logtrend_causal(close_s)
ax.plot(rs_v[-200:],         label="SMA(1000)",  alpha=0.8)
ax.plot(rl[-200:],           label="lintrend",   alpha=0.8)
ax.plot(rg[-200:],           label="logtrend",   alpha=0.8)
ax.axhline(1.0, color="black", lw=0.5, ls="--")
ax.set_title(f"Ratio SBER (последние 200 баров)")
ax.set_ylabel("ratio = close / trend"); ax.legend()

plt.tight_layout()
out = OUT_DIR / "41_linear_trend_norm.png"
plt.savefig(out, dpi=120)
print(f"\nФигура сохранена: {out}")
