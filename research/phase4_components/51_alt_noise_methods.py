"""
51 — Альтернативные методы для шумовых компонент C0–C2:

  direct_p8   — прямой multi-step OLS: отдельная линейная модель на каждый горизонт h,
                предсказывает cumsum(c[t+1..t+h]) напрямую — нет накопления ошибок
  damped_07   — AR(BIC) × decay γ=0.70^h (агрессивное затухание к нулю)
  damped_09   — AR(BIC) × decay γ=0.90^h (мягкое затухание)
  fourier_2   — 2 доминантные FFT-частоты + DC, OLS-экстраполяция синусоидами
  fourier_4   — 4 доминантные FFT-частоты + DC

Базовые линии (из скр.50): zero, ar_bic (−1.16% реконструированный MAPE)
Медленные C3–C5: LWR p=20 ξ=63 (без изменений)

Walk-forward: N_ORIG=50 × 8 тикеров, 1d, logtrend, val_h=10
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL     = "1d"
N_ORIG       = 50
VAL_H        = 10
P_SLOW       = 20
XI_SLOW      = 63
P_BIC_MAX    = 20
P_DIRECT     = 8          # лагов для прямого OLS
EPS          = 1e-10
STD_CUTOFFS  = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
FILTER_ORDER = 4

# окно (баров) для Фурье-аппроксимации по компоненте
FOURIER_WINDOWS = {0: 32, 1: 64, 2: 128}

METHODS  = ["zero", "ar_bic", "direct_p8", "damped_07", "damped_09", "fourier_2", "fourier_4"]
CONFIGS  = [
    "std",            # slow LWR only (baseline)
    "ar_bic_c12",     # лучший из скр.50: −1.16%
    "direct_c12",     # direct OLS на C1+C2
    "damped_07_c12",  # decay γ=0.7 на C1+C2
    "damped_09_c12",  # decay γ=0.9 на C1+C2
    "fourier_2_c12",  # 2 частоты на C1+C2
    "fourier_4_c12",  # 4 частоты на C1+C2
]
COMP_NAMES = {0: "C0  2–8б", 1: "C1  8–16б", 2: "C2  16–32б"}

# ── вспомогательные функции ───────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    lc = np.log(close + EPS)
    n  = len(lc)
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t**2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = a + b * t
    trend[:2] = lc[:2]
    return np.exp(trend)


def make_fb(series: np.ndarray) -> np.ndarray:
    comps = []; rem = series.copy()
    for fc in STD_CUTOFFS:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, rem)
        comps.append(rem - low); rem = low
    comps.append(rem)
    return np.array(comps)   # (6, N)


# ── AR-модели (из скр.50) ─────────────────────────────────────────────────────

def fit_ar(series: np.ndarray, p: int) -> np.ndarray:
    n = len(series)
    if n <= p + 1:
        return np.zeros(p + 1)
    X = np.zeros((n - p, p))
    for lag in range(p):
        X[:, lag] = series[p - 1 - lag: n - 1 - lag]
    X = np.hstack([np.ones((n - p, 1)), X])
    coeffs, _, _, _ = np.linalg.lstsq(X, series[p:], rcond=None)
    return coeffs


def forecast_ar(series: np.ndarray, p: int, horizon: int,
                coeffs: np.ndarray) -> np.ndarray:
    buf = list(series[-p:])
    out = np.empty(horizon)
    for h in range(horizon):
        val = coeffs[0] + sum(coeffs[1 + k] * buf[-(k + 1)] for k in range(p))
        out[h] = val
        buf.append(val)
    return out


def fit_ar_bic(series: np.ndarray, p_max: int = P_BIC_MAX) -> tuple[int, np.ndarray]:
    n = len(series)
    best_bic, best_p, best_c = np.inf, 1, np.zeros(2)
    for p in range(1, min(p_max + 1, (n - 1) // 4)):
        c   = fit_ar(series, p)
        nef = n - p
        X   = np.zeros((nef, p))
        for lag in range(p):
            X[:, lag] = series[p - 1 - lag: n - 1 - lag]
        X   = np.hstack([np.ones((nef, 1)), X])
        ssr = np.sum((series[p:] - X @ c) ** 2)
        bic = nef * np.log(ssr / nef + EPS) + (p + 1) * np.log(nef)
        if bic < best_bic:
            best_bic, best_p, best_c = bic, p, c
    return best_p, best_c


# ── LWR для медленных компонент (из скр.50) ───────────────────────────────────

def forecast_lwr_comp(series: np.ndarray, p: int, xi: int,
                      horizon: int) -> np.ndarray:
    n  = len(series)
    X  = np.array([series[i: i + p] for i in range(n - p)])
    y  = series[p:]
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
        vec    = np.roll(vec, -1); vec[-1] = val
    return out


# ── Новые методы ──────────────────────────────────────────────────────────────

def forecast_direct(series: np.ndarray, p: int, horizon: int) -> np.ndarray:
    """
    Прямой multi-step OLS. Для каждого горизонта h обучаем отдельную модель:
      features: [c[t], c[t-1], ..., c[t-p+1]]
      target:   cumsum(c[t+1..t+h]) = вклад компоненты в ratio за h шагов

    Возвращает dhat = diff(pred_cumsum[0..H]), что соответствует поэлементному
    вкладу в прогноз ratio без итеративного накопления ошибок.
    """
    n = len(series)
    n_train = n - p - horizon
    if n_train < 10:
        return np.zeros(horizon)

    # Lag-матрица: X[i, lag+1] = series[p-1-lag+i], lag=0 — самое свежее
    X = np.zeros((n_train, p + 1))
    X[:, 0] = 1.0
    for lag in range(p):
        X[:, lag + 1] = series[p - 1 - lag: p - 1 - lag + n_train]

    # Query: features на последних p значениях
    query = np.ones(p + 1)
    for lag in range(p):
        query[lag + 1] = series[-(lag + 1)]

    # Prefix sums для быстрого вычисления cumsum-целей
    cs    = np.concatenate([[0.0], np.cumsum(series)])
    start = np.arange(n_train) + p  # start[i] = p+i

    pred_cumsum = np.zeros(horizon + 1)  # pred_cumsum[0] = 0
    for h in range(1, horizon + 1):
        y_h    = cs[start + h] - cs[start]  # sum(series[p+i : p+i+h])
        coeffs, _, _, _ = np.linalg.lstsq(X, y_h, rcond=None)
        pred_cumsum[h] = float(coeffs @ query)

    return np.diff(pred_cumsum)   # dhat[h] = pred_cumsum[h] − pred_cumsum[h−1]


def forecast_damped(series: np.ndarray, horizon: int, gamma: float) -> np.ndarray:
    """AR(BIC) прогноз умножается на экспоненциальное затухание gamma^h."""
    if len(series) <= P_BIC_MAX + 2:
        return np.zeros(horizon)
    bp, bc  = fit_ar_bic(series)
    raw     = forecast_ar(series, bp, horizon, bc)
    decay   = gamma ** np.arange(1, horizon + 1)
    return raw * decay


def forecast_fourier(series: np.ndarray, n_freqs: int, window: int,
                     horizon: int) -> np.ndarray:
    """
    Fit DC + n_freqs синусоид через OLS на последних `window` барах,
    затем экстраполируем на horizon шагов вперёд.
    """
    seg = series[-window:] if len(series) >= window else series
    n   = len(seg)
    if n < 8:
        return np.zeros(horizon)

    t        = np.arange(n, dtype=np.float64)
    fft_amp  = np.abs(np.fft.rfft(seg))
    freqs    = np.fft.rfftfreq(n, d=1.0)
    fft_amp[0] = 0.0                          # убираем DC из отбора пиков
    top_idx  = np.argsort(fft_amp)[-n_freqs:]
    top_f    = freqs[top_idx]
    top_f    = top_f[top_f > 0]              # только положительные частоты

    if len(top_f) == 0:
        return np.zeros(horizon)

    # Дизайн-матрица: [1, cos(ω1·t), sin(ω1·t), ..., cos(ωk·t), sin(ωk·t)]
    cols = [np.ones(n)]
    for f in top_f:
        cols.extend([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)])
    X = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(X, seg, rcond=None)

    t_fut    = np.arange(n, n + horizon, dtype=np.float64)
    cols_fut = [np.ones(horizon)]
    for f in top_f:
        cols_fut.extend([np.cos(2 * np.pi * f * t_fut),
                         np.sin(2 * np.pi * f * t_fut)])
    return np.column_stack(cols_fut) @ coeffs


# ── walk-forward ──────────────────────────────────────────────────────────────

comp_mae     = {m: {ci: [] for ci in range(3)} for m in METHODS}
config_mapes = {c: [] for c in CONFIGS}
examples     = {}   # {ticker_Cx: {...}} для графиков

print("Walk-forward по тикерам...")
for ticker in TICKERS:
    print(f"  {ticker}", end="", flush=True)
    with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
        candles = json.load(f)
    close  = np.array([x["close"] for x in candles], dtype=np.float64)
    ratio  = close / logtrend_causal(close)
    dratio = np.diff(ratio)
    N      = len(dratio)

    origin_min = max(500, N // 2)
    origins    = np.linspace(origin_min, N - VAL_H - 5, N_ORIG, dtype=int)

    for origin_k in origins:
        comps_hist = make_fb(dratio[:origin_k])          # (6, origin_k)
        comps_full = make_fb(dratio[:origin_k + VAL_H])  # каузально ОК

        # Медленные C3–C5: LWR (базовая линия)
        slow_hat = np.zeros(VAL_H)
        for ci in [3, 4, 5]:
            ch = comps_hist[ci]
            if len(ch) >= XI_SLOW + P_SLOW + 2:
                slow_hat += forecast_lwr_comp(ch, P_SLOW, XI_SLOW, VAL_H)

        fast_pred = {m: {} for m in METHODS}

        for ci in range(3):
            ch        = comps_hist[ci]
            actual_ci = comps_full[ci][origin_k: origin_k + VAL_H]
            win       = FOURIER_WINDOWS[ci]
            preds: dict[str, np.ndarray] = {}

            preds["zero"] = np.zeros(VAL_H)

            # AR(BIC) — вычисляем один раз, переиспользуем для damped-вариантов
            if len(ch) > P_BIC_MAX + 2:
                bp, bc        = fit_ar_bic(ch)
                ar_raw        = forecast_ar(ch, bp, VAL_H, bc)
                preds["ar_bic"]    = ar_raw
                preds["damped_07"] = ar_raw * (0.7 ** np.arange(1, VAL_H + 1))
                preds["damped_09"] = ar_raw * (0.9 ** np.arange(1, VAL_H + 1))
            else:
                preds["ar_bic"]    = np.zeros(VAL_H)
                preds["damped_07"] = np.zeros(VAL_H)
                preds["damped_09"] = np.zeros(VAL_H)

            if len(ch) > P_DIRECT + VAL_H + 10:
                preds["direct_p8"] = forecast_direct(ch, P_DIRECT, VAL_H)
            else:
                preds["direct_p8"] = np.zeros(VAL_H)

            if len(ch) >= win + 8:
                preds["fourier_2"] = forecast_fourier(ch, 2, win, VAL_H)
                preds["fourier_4"] = forecast_fourier(ch, 4, win, VAL_H)
            else:
                preds["fourier_2"] = np.zeros(VAL_H)
                preds["fourier_4"] = np.zeros(VAL_H)

            for m in METHODS:
                comp_mae[m][ci].append(np.abs(preds[m] - actual_ci))
                fast_pred[m][ci] = preds[m]

            # Сохраняем первый origin SBER/LKOH для примера
            for tkr_ex in ["SBER", "LKOH"]:
                key = f"{tkr_ex}_C{ci}"
                if key not in examples and ticker == tkr_ex:
                    examples[key] = {
                        "ticker": ticker, "ci": ci, "origin_k": origin_k,
                        "hist":   comps_hist[ci].copy(),
                        "actual": comps_full[ci][origin_k: origin_k + VAL_H].copy(),
                        "preds":  {m: preds[m].copy() for m in METHODS},
                    }

        # Реконструированный MAPE
        ratio0       = ratio[origin_k]
        actual_ratio = ratio[origin_k + 1: origin_k + VAL_H + 1]

        def _mape(dhat: np.ndarray) -> float:
            rh = ratio0 + np.cumsum(dhat)
            nn = min(len(rh), len(actual_ratio))
            return float(np.mean(
                np.abs(rh[:nn] - actual_ratio[:nn]) / (np.abs(actual_ratio[:nn]) + EPS)
            ))

        c12 = lambda m: fast_pred[m][1] + fast_pred[m][2]
        config_dhat = {
            "std":            slow_hat,
            "ar_bic_c12":     slow_hat + c12("ar_bic"),
            "direct_c12":     slow_hat + c12("direct_p8"),
            "damped_07_c12":  slow_hat + c12("damped_07"),
            "damped_09_c12":  slow_hat + c12("damped_09"),
            "fourier_2_c12":  slow_hat + c12("fourier_2"),
            "fourier_4_c12":  slow_hat + c12("fourier_4"),
        }
        for cfg, dhat in config_dhat.items():
            config_mapes[cfg].append(_mape(dhat))

    print(" ✓")

# ── числовые результаты ───────────────────────────────────────────────────────

print()
for ci in range(3):
    name = COMP_NAMES[ci]
    print(f"\n{'='*62}")
    print(f"  {name}  (MAE, среднее по {N_ORIG*len(TICKERS)} origins)")
    print(f"  {'Метод':<15} {'h=1':>10} {'h=5':>10} {'h=10':>10}")
    print(f"  {'-'*50}")
    zero_m1 = np.mean([v[0] for v in comp_mae["zero"][ci]])
    for m in METHODS:
        arr = np.array(comp_mae[m][ci])
        m1  = np.mean(arr[:, 0])
        m5  = np.mean(arr[:, 4])
        m10 = np.mean(arr[:, 9])
        g1  = m1 / (zero_m1 + EPS)
        marker = " ★" if g1 < 0.90 else ("  " if g1 < 1.05 else " ✗")
        print(f"  {m:<15} {m1:>10.5f} {m5:>10.5f} {m10:>10.5f}{marker}")

print(f"\n{'='*62}")
std_m = np.mean(config_mapes["std"])
print(f"  Реконструированный AGG MAPE  (N={len(config_mapes['std'])} origins)")
print(f"  {'Конфигурация':<18} {'MAPE':>10} {'Δ% vs std':>12}")
print(f"  {'-'*43}")
for cfg in CONFIGS:
    m     = np.mean(config_mapes[cfg])
    delta = (m - std_m) / std_m * 100
    flag  = " ★" if delta < -0.3 else (" ✗" if delta > 0.3 else "  ")
    print(f"  {cfg:<18} {m:>10.5f} {delta:>+11.2f}%{flag}")

# ── рис 1: MAE компонент ──────────────────────────────────────────────────────

COLORS   = ["#9e9e9e", "#00897b", "#42a5f5", "#e65100",
            "#ef9a9a", "#7b1fa2", "#311b92"]
H_IDXS   = [0, 4, 9]
H_LABELS = ["h=1", "h=5", "h=10"]

fig1, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig1.suptitle(
    f"MAE шумовых компонент: новые методы vs baseline\n"
    f"8 тикеров × {N_ORIG} origins, 1d, logtrend, val_h={VAL_H}",
    fontsize=12, fontweight="bold",
)
x = np.arange(len(METHODS)); w = 0.22
for ax, ci in zip(axes, range(3)):
    zero_h1 = np.mean([v[0] for v in comp_mae["zero"][ci]])
    for j, (hi, hl) in enumerate(zip(H_IDXS, H_LABELS)):
        vals = [np.mean([v[hi] for v in comp_mae[m][ci]]) for m in METHODS]
        ax.bar(x + (j - 1) * w, vals, w,
               color=[COLORS[i] for i in range(len(METHODS))],
               alpha=[1.0, 0.7, 0.45][j], edgecolor="white", linewidth=0.5)
    ax.axhline(zero_h1, color="#9e9e9e", lw=1.2, ls="--", alpha=0.7, label="zero h=1")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=35, ha="right", fontsize=8)
    ax.set_title(COMP_NAMES[ci], fontsize=11, fontweight="bold")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.3)
handles = [plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=a) for a in [1.0, 0.7, 0.45]]
fig1.legend(handles, H_LABELS, loc="upper right", fontsize=9)
plt.tight_layout()
out1 = OUT_DIR / "51_mae_components.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"\n  Рис 1 → {out1}")

# ── рис 2: реконструированный MAPE ───────────────────────────────────────────

cfg_colors = ["#9e9e9e", "#00897b", "#42a5f5", "#e65100",
              "#ef9a9a", "#7b1fa2", "#311b92"]
cfg_vals   = [np.mean(config_mapes[c]) * 100 for c in CONFIGS]

fig2, ax2 = plt.subplots(figsize=(12, 5))
fig2.suptitle(
    "Реконструированный MAPE: slow LWR + альтернативные методы для C1+C2\n"
    f"8 тикеров × {N_ORIG} origins, 1d, logtrend",
    fontsize=12, fontweight="bold",
)
bars2 = ax2.bar(CONFIGS, cfg_vals, color=cfg_colors, edgecolor="white", linewidth=0.8)
for bar, val in zip(bars2, cfg_vals):
    delta = (val - cfg_vals[0]) / cfg_vals[0] * 100
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.00005,
             f"{delta:+.2f}%", ha="center", va="bottom",
             fontsize=9, fontweight="bold",
             color="#00897b" if delta < 0 else "#b71c1c")
ax2.set_ylabel("MAPE, %")
ax2.set_ylim(bottom=min(cfg_vals) * 0.97)
ax2.set_xticklabels(CONFIGS, rotation=20, ha="right")
ax2.grid(axis="y", alpha=0.3)
ax2.axhline(cfg_vals[0], color="#9e9e9e", lw=1, ls="--")
plt.tight_layout()
out2 = OUT_DIR / "51_mape_configs.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Рис 2 → {out2}")

# ── рис 3 и 4: примеры прогноза C1 и C2 ─────────────────────────────────────

PRED_COLORS = {
    "zero":      "#9e9e9e",
    "ar_bic":    "#00897b",
    "direct_p8": "#42a5f5",
    "damped_07": "#e65100",
    "damped_09": "#ef9a9a",
    "fourier_2": "#7b1fa2",
    "fourier_4": "#311b92",
}
PRED_STYLES = {
    "zero":      "--",
    "ar_bic":    "-.",
    "direct_p8": "-",
    "damped_07": "--",
    "damped_09": ":",
    "fourier_2": "-",
    "fourier_4": "-.",
}

for ci_ex, period_str, out_name in [
    (1, "8–16", "51_example_C1.png"),
    (2, "16–32", "51_example_C2.png"),
]:
    key = f"SBER_C{ci_ex}"
    if key not in examples:
        key = next((k for k in examples if k.endswith(f"_C{ci_ex}")), None)
    if key is None:
        print(f"  (нет примера C{ci_ex}, пропускаем)")
        continue

    ex     = examples[key]
    hist   = ex["hist"]
    n_show = min(80, len(hist))
    fig3, ax3 = plt.subplots(figsize=(14, 5))

    ax3.plot(np.arange(-n_show, 0), hist[-n_show:],
             color="#455a64", lw=1.5, label=f"история C{ci_ex}")
    ax3.plot(np.arange(VAL_H), ex["actual"],
             color="#ffd600", lw=2.5, label="actual (будущее)")

    for m in METHODS:
        ax3.plot(np.arange(VAL_H), ex["preds"][m],
                 color=PRED_COLORS[m], ls=PRED_STYLES[m], lw=1.8, label=m)

    ax3.axvline(0, color="white", lw=1, ls=":", alpha=0.5)
    ax3.set_title(
        f"Пример: C{ci_ex} компонента — {ex['ticker']} origin={ex['origin_k']}"
        f"\n(период {period_str} баров)",
        fontsize=11, fontweight="bold",
    )
    ax3.set_xlabel("Баров от origin")
    ax3.set_ylabel(f"Δratio (C{ci_ex})")
    ax3.legend(fontsize=9, ncol=4)
    ax3.grid(alpha=0.25)
    plt.tight_layout()
    out_path = OUT_DIR / out_name
    fig3.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Рис → {out_path}")

print("\nГотово.")
