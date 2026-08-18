"""
50 — AR(p) для шумовых компонент C0–C2: сравнение с нулём.

Методы для C0–C2:
  zero        — 0 (оптимум для C0 из скр.46)
  persistence — последнее значение компоненты
  ar1         — AR(1) OLS, итеративный прогноз
  ar2         — AR(2) OLS
  ar_bic      — AR(p), p по BIC (до P_BIC_MAX)
  lwr_p8      — LWR p=8 ξ=27 (лучший из скр.47 для slow, проверяем на fast)

Медленные C3–C5: стандартный LWR p=20 ξ=63.

Walk-forward: N_ORIG=50 origins/ticker, 8 тикеров, 1d, logtrend, val_h=10.

Рисунки:
  50_mae_components.png    — MAE по компоненте h=1/5/10, все методы
  50_mape_configs.png      — реконструированный MAPE: slow-only vs +AR на C1-C2
  50_example_forecast.png  — пример прогноза C1 одного тикера одного origin
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
N_ORIG       = 50        # origins на тикер
VAL_H        = 10        # горизонт валидации
P_SLOW       = 20        # p для C3-C5
XI_SLOW      = 63        # ξ для C3-C5
P_BIC_MAX    = 20        # максимальный p при поиске по BIC
P_LWR        = 8         # p LWR для шумовых (из скр.47)
XI_LWR       = max(27, 3 * (P_LWR + 1))
EPS          = 1e-10
STD_CUTOFFS  = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
FILTER_ORDER = 4

METHODS = ["zero", "persistence", "ar1", "ar2", "ar_bic", "lwr_p8"]
CONFIGS = ["std", "ar1_c12", "ar2_c12", "ar_bic_c12", "ar1_all", "ar2_all"]

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


# ── AR-модели ─────────────────────────────────────────────────────────────────

def fit_ar(series: np.ndarray, p: int) -> np.ndarray:
    """OLS AR(p): возвращает [intercept, a1, a2, ..., ap].
    a1 — коэф. для lag-1, a2 — lag-2, и т.д."""
    n = len(series)
    if n <= p + 1:
        return np.zeros(p + 1)
    # lag-матрица: строка i = [series[i+p-1], series[i+p-2], ..., series[i]]
    X = np.zeros((n - p, p))
    for lag in range(p):
        X[:, lag] = series[p - 1 - lag: n - 1 - lag]
    X = np.hstack([np.ones((n - p, 1)), X])
    y = series[p:]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs


def forecast_ar(series: np.ndarray, p: int, horizon: int,
                coeffs: np.ndarray) -> np.ndarray:
    """Итеративный AR(p) прогноз. buf = [..., y[t-2], y[t-1]] (новые справа)."""
    buf = list(series[-p:])
    out = np.empty(horizon)
    for h in range(horizon):
        # coeffs[1]*buf[-1] + coeffs[2]*buf[-2] + ...
        val = coeffs[0] + sum(coeffs[1 + lag] * buf[-(lag + 1)] for lag in range(p))
        out[h] = val
        buf.append(val)
    return out


def fit_ar_bic(series: np.ndarray, p_max: int = P_BIC_MAX) -> tuple[int, np.ndarray]:
    """Выбор AR(p) по BIC. Возвращает (best_p, coeffs)."""
    n = len(series)
    best_bic, best_p, best_coeffs = np.inf, 1, np.zeros(2)
    for p in range(1, min(p_max + 1, (n - 1) // 4)):
        coeffs = fit_ar(series, p)
        n_eff  = n - p
        X = np.zeros((n_eff, p))
        for lag in range(p):
            X[:, lag] = series[p - 1 - lag: n - 1 - lag]
        X = np.hstack([np.ones((n_eff, 1)), X])
        ssr = np.sum((series[p:] - X @ coeffs) ** 2)
        bic = n_eff * np.log(ssr / n_eff + EPS) + (p + 1) * np.log(n_eff)
        if bic < best_bic:
            best_bic, best_p, best_coeffs = bic, p, coeffs
    return best_p, best_coeffs


# ── LWR для одной компоненты ──────────────────────────────────────────────────

def _build_dm(series: np.ndarray, p: int):
    n  = len(series)
    X  = np.array([series[i: i + p] for i in range(n - p)])
    y  = series[p:]
    return X, y


def forecast_lwr_comp(series: np.ndarray, p: int, xi: int,
                      horizon: int) -> np.ndarray:
    X, y = _build_dm(series, p)
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


# ── walk-forward ──────────────────────────────────────────────────────────────

# comp_mae[method][ci] = список numpy-векторов MAE длины VAL_H
comp_mae = {m: {ci: [] for ci in range(3)} for m in METHODS}
config_mapes = {c: [] for c in CONFIGS}

# для примера: сохраним один origin
example_saved = None

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
        # --- разложение до origin_k (каузальное) ---
        comps_hist = make_fb(dratio[:origin_k])   # (6, origin_k)

        # --- "будущие" компоненты для оценки: разложение до origin_k + VAL_H ---
        # sosfilt каузален: comps_hist[ci][t] = comps_full[ci][t] для t < origin_k
        comps_full = make_fb(dratio[:origin_k + VAL_H])  # (6, origin_k+VAL_H)

        # --- медленные C3-C5: LWR p=20 ---
        slow_hat = np.zeros(VAL_H)
        for ci in [3, 4, 5]:
            ch = comps_hist[ci]
            if len(ch) >= XI_SLOW + P_SLOW + 2:
                slow_hat += forecast_lwr_comp(ch, P_SLOW, XI_SLOW, VAL_H)

        # --- шумовые C0-C2: все методы ---
        fast_pred: dict[str, dict[int, np.ndarray]] = {m: {} for m in METHODS}

        for ci in range(3):
            ch        = comps_hist[ci]
            actual_ci = comps_full[ci][origin_k: origin_k + VAL_H]

            preds: dict[str, np.ndarray] = {}

            preds["zero"] = np.zeros(VAL_H)

            preds["persistence"] = np.full(VAL_H, ch[-1] if len(ch) else 0.0)

            if len(ch) > 3:
                c1  = fit_ar(ch, 1)
                c2  = fit_ar(ch, 2)
                preds["ar1"] = forecast_ar(ch, 1, VAL_H, c1)
                preds["ar2"] = forecast_ar(ch, 2, VAL_H, c2)
            else:
                preds["ar1"] = np.zeros(VAL_H)
                preds["ar2"] = np.zeros(VAL_H)

            if len(ch) > P_BIC_MAX + 2:
                bp, bc = fit_ar_bic(ch)
                preds["ar_bic"] = forecast_ar(ch, bp, VAL_H, bc)
            else:
                preds["ar_bic"] = np.zeros(VAL_H)

            if len(ch) >= XI_LWR + P_LWR + 2:
                preds["lwr_p8"] = forecast_lwr_comp(ch, P_LWR, XI_LWR, VAL_H)
            else:
                preds["lwr_p8"] = np.zeros(VAL_H)

            for m in METHODS:
                mae_vec = np.abs(preds[m] - actual_ci)
                comp_mae[m][ci].append(mae_vec)
                fast_pred[m][ci] = preds[m]

        # --- реконструированный MAPE ---
        ratio0       = ratio[origin_k]
        actual_ratio = ratio[origin_k + 1: origin_k + VAL_H + 1]

        def _mape(dhat):
            rh = ratio0 + np.cumsum(dhat)
            n  = min(len(rh), len(actual_ratio))
            return float(np.mean(np.abs(rh[:n] - actual_ratio[:n])
                                 / (np.abs(actual_ratio[:n]) + EPS)))

        config_dhat = {
            "std":        slow_hat,
            "ar1_c12":    slow_hat + fast_pred["ar1"][1] + fast_pred["ar1"][2],
            "ar2_c12":    slow_hat + fast_pred["ar2"][1] + fast_pred["ar2"][2],
            "ar_bic_c12": slow_hat + fast_pred["ar_bic"][1] + fast_pred["ar_bic"][2],
            "ar1_all":    slow_hat + sum(fast_pred["ar1"][ci] for ci in range(3)),
            "ar2_all":    slow_hat + sum(fast_pred["ar2"][ci] for ci in range(3)),
        }
        for cfg, dhat in config_dhat.items():
            config_mapes[cfg].append(_mape(dhat))

        # сохраняем пример (SBER, первый origin)
        if example_saved is None and ticker == "SBER":
            example_saved = {
                "ticker":    ticker,
                "origin_k":  origin_k,
                "actual_c1": comps_full[1][origin_k: origin_k + VAL_H],
                "preds_c1":  {m: fast_pred[m][1] for m in METHODS},
                "hist_c1":   comps_hist[1],
            }

    print(" ✓")

# ── числовые результаты ───────────────────────────────────────────────────────

print()
for ci in range(3):
    name = COMP_NAMES[ci]
    print(f"\n{'='*55}")
    print(f"  {name}   (MAE, среднее по {N_ORIG*len(TICKERS)} origins)")
    print(f"  {'Метод':<15} {'h=1':>10} {'h=5':>10} {'h=10':>10}")
    print(f"  {'-'*48}")
    zero_m1 = np.mean([v[0] for v in comp_mae["zero"][ci]])
    for m in METHODS:
        arr = np.array(comp_mae[m][ci])   # (n_orig, VAL_H)
        m1  = np.mean(arr[:, 0])
        m5  = np.mean(arr[:, 4])
        m10 = np.mean(arr[:, 9])
        g1  = m1 / zero_m1
        marker = " ★" if g1 < 0.97 else ("  " if g1 < 1.03 else " ✗")
        print(f"  {m:<15} {m1:>10.5f} {m5:>10.5f} {m10:>10.5f}{marker}")

print(f"\n{'='*55}")
std_m = np.mean(config_mapes["std"])
print(f"  Реконструированный AGG MAPE  (N={len(config_mapes['std'])} origins)")
print(f"  {'Конфигурация':<15} {'MAPE':>10} {'Δ% vs std':>12}")
print(f"  {'-'*40}")
for cfg in CONFIGS:
    m     = np.mean(config_mapes[cfg])
    delta = (m - std_m) / std_m * 100
    flag  = " ★" if delta < -0.3 else (" ✗" if delta > 0.3 else "  ")
    print(f"  {cfg:<15} {m:>10.5f} {delta:>+11.2f}%{flag}")

# ── рис 1: MAE по компонентам ─────────────────────────────────────────────────

fig1, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig1.suptitle(
    f"MAE шумовых компонент: методы прогноза\n"
    f"8 тикеров × {N_ORIG} origins, 1d, logtrend, val_h=10",
    fontsize=12, fontweight="bold",
)

COLORS  = ["#9e9e9e", "#607d8b", "#42a5f5", "#1565c0", "#00897b", "#e53935"]
H_IDXS  = [0, 4, 9]
H_LABELS = ["h=1", "h=5", "h=10"]
x       = np.arange(len(METHODS))
w       = 0.22

for ax, ci in zip(axes, range(3)):
    zero_h1 = np.mean([v[0] for v in comp_mae["zero"][ci]])
    for j, (hi, hl) in enumerate(zip(H_IDXS, H_LABELS)):
        vals = [np.mean([v[hi] for v in comp_mae[m][ci]]) for m in METHODS]
        bars = ax.bar(x + (j - 1) * w, vals, w, label=hl,
                      color=[COLORS[i] for i in range(len(METHODS))],
                      alpha=[1.0, 0.7, 0.45][j], edgecolor="white", linewidth=0.5)
    ax.axhline(zero_h1, color="#9e9e9e", lw=1.2, ls="--", alpha=0.6, label="zero h=1")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=9)
    ax.set_title(COMP_NAMES[ci], fontsize=11, fontweight="bold")
    ax.set_ylabel("MAE")
    ax.grid(axis="y", alpha=0.3)

handles = [plt.Rectangle((0,0),1,1, fc="gray", alpha=a) for a in [1.0, 0.7, 0.45]]
fig1.legend(handles, H_LABELS, loc="upper right", fontsize=9)
plt.tight_layout()
out1 = OUT_DIR / "50_mae_components.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\n  Рис 1 → {out1}")
plt.close(fig1)

# ── рис 2: реконструированный MAPE ───────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.suptitle(
    "Реконструированный MAPE: slow LWR + разные стратегии для C0–C2\n"
    f"8 тикеров × {N_ORIG} origins, 1d, logtrend",
    fontsize=12, fontweight="bold",
)

cfg_colors = ["#9e9e9e", "#42a5f5", "#1565c0", "#00897b", "#ef9a9a", "#b71c1c"]
cfg_vals   = [np.mean(config_mapes[c]) * 100 for c in CONFIGS]
bars2      = ax2.bar(CONFIGS, cfg_vals, color=cfg_colors, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars2, cfg_vals):
    delta = (val - cfg_vals[0]) / cfg_vals[0] * 100
    sign  = f"{delta:+.2f}%"
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.0001,
             sign, ha="center", va="bottom", fontsize=9, fontweight="bold",
             color="#00897b" if delta < 0 else "#b71c1c")

ax2.set_ylabel("MAPE валидации, %")
ax2.set_ylim(bottom=min(cfg_vals) * 0.97)
ax2.set_xticklabels(CONFIGS, rotation=15, ha="right")
ax2.grid(axis="y", alpha=0.3)
ax2.axhline(cfg_vals[0], color="#9e9e9e", lw=1, ls="--")
plt.tight_layout()
out2 = OUT_DIR / "50_mape_configs.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"  Рис 2 → {out2}")
plt.close(fig2)

# ── рис 3: пример прогноза C1 ────────────────────────────────────────────────

if example_saved:
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    ex      = example_saved
    hist    = ex["hist_c1"]
    n_show  = min(80, len(hist))
    t_hist  = np.arange(-n_show, 0)
    t_fut   = np.arange(VAL_H)
    actual  = ex["actual_c1"]

    ax3.plot(t_hist, hist[-n_show:], color="#455a64", lw=1.5, label="история C1")
    ax3.plot(t_fut,  actual,          color="#ffd600", lw=2.5, label="actual C1 (будущее)")

    PRED_COLORS = {"zero": "#9e9e9e", "persistence": "#607d8b",
                   "ar1": "#42a5f5", "ar2": "#1565c0",
                   "ar_bic": "#00897b", "lwr_p8": "#e53935"}
    PRED_STYLES = {"zero": "--", "persistence": ":", "ar1": "-", "ar2": "-.",
                   "ar_bic": "-", "lwr_p8": "--"}

    for m in METHODS:
        pred = ex["preds_c1"][m]
        ax3.plot(t_fut, pred, color=PRED_COLORS[m], ls=PRED_STYLES[m],
                 lw=1.8, label=m)

    ax3.axvline(0, color="white", lw=1, ls=":", alpha=0.5)
    ax3.set_title(f"Пример: C1 компонента — {ex['ticker']} origin_k={ex['origin_k']}",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("Баров от origin")
    ax3.set_ylabel("Δratio (C1)")
    ax3.legend(fontsize=9, ncol=4)
    ax3.grid(alpha=0.25)

    plt.tight_layout()
    out3 = OUT_DIR / "50_example_forecast.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"  Рис 3 → {out3}")
    plt.close(fig3)

print("\nГотово.")
