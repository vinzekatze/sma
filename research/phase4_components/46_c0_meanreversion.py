"""
46 — Mean-reversion прогноз для C0 (антиперсистентная компонента).

Из скр.42: C0 (2–4 bar, 63.4% энергии) имеет H=0.28, ACF[1]=0.135, Gain(LWR)<1.
LWR хуже нулевого прогноза на C0. Причина: C0 антиперсистентна, LWR «тянет вверх».

Вопрос: может ли явная mean-reversion модель дать Gain > 1 для C0?
Если да — какой прирост MAPE от включения C0-прогноза?

Методы для C0:
  zero        : ĉ = 0  (текущий стандарт)
  persistence : ĉ = c[t]
  ar1         : ĉ[t+h] = φ^h × c[t], φ = OLS-rolling
  arp         : AR(p) rolling, p ∈ {1..8}, выбор по min RMSE на окне
  ou_fixed    : ĉ[t+h] = c[t] × exp(−θ·h), θ = −ln(ACF[1]) ← теоретическая OU
  LWR         : стандарт (ожидается Gain < 1)

Дополнительно: проверяем C1 (19.7% энергии, ACF[1]=0.898) с AR(1)/persistence.

Финал: best_C0 + C3–C5 LWR → сравнение MAPE со стандартом (zero_C0 + C3–C5 LWR).
"""

import json, sys, time, math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from statsmodels.tsa.stattools import acf as compute_acf

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from sma.core.forecast.embedding import build_delay_matrix, last_vector

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL     = "1d"
EPS          = 1e-10
STD_CUTOFFS  = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
SLOW_IDX     = [3, 4, 5]
C0_IDX       = 0
C1_IDX       = 1
FILTER_ORDER = 4
P_LWR        = 20; XI_LWR = 63
VAL_H        = 10   # стандарт
N_ORIG       = 50
AR_WINDOW    = 200  # окно для rolling AR-fit
AR_P_MAX     = 8    # максимальный порядок AR


# ── нормализация + данные ─────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    log_c = np.log(close + EPS); n = len(log_c)
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n+1, dtype=np.float64)
    ct=np.cumsum(t); ct2=np.cumsum(t**2); cy=np.cumsum(log_c); cty=np.cumsum(t*log_c)
    denom = cn*ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom>0, (cn*cty - ct*cy)/denom, 0.0)
    a = (cy - b*ct)/cn; trend = a+b*t; trend[:2] = log_c[:2]
    return np.exp(trend)


_CACHE: dict = {}

def load_data(ticker: str) -> tuple[np.ndarray, np.ndarray]:
    if ticker not in _CACHE:
        with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
            c = json.load(f)
        close = np.array([x["close"] for x in c], dtype=np.float64)
        ratio = close / logtrend_causal(close)
        _CACHE[ticker] = (ratio, np.diff(ratio))
    return _CACHE[ticker]


# ── filter bank ────────────────────────────────────────────────────────────────

def make_fb(series: np.ndarray, cutoffs: list, order: int = 4) -> np.ndarray:
    comps=[]; rem=series.copy()
    for fc in cutoffs:
        sos=butter(order,fc,btype="low",output="sos"); low=sosfilt(sos,rem)
        comps.append(rem-low); rem=low
    comps.append(rem); return np.array(comps)


# ── методы прогнозирования C0 ─────────────────────────────────────────────────

def forecast_zero(h: int) -> np.ndarray:
    return np.zeros(h)


def forecast_persistence(series: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, series[-1])


def forecast_ar1(series: np.ndarray, h: int, window: int = AR_WINDOW) -> np.ndarray:
    """AR(1): φ = OLS(c[t-1], c[t]) на последнем window."""
    k = min(window, len(series) - 1)
    if k < 5:
        return np.zeros(h)
    y = series[-(k):]; x = series[-(k+1):-1]
    xm = x.mean(); ym = y.mean()
    phi = np.sum((x - xm) * (y - ym)) / (np.sum((x - xm)**2) + EPS)
    # итеративный: ĉ[t+h] = φ^h × c[t] (AR1 теоретически)
    c0 = series[-1]
    return np.array([c0 * (phi ** i) for i in range(1, h + 1)])


def forecast_arp(series: np.ndarray, h: int, p_max: int = AR_P_MAX,
                 window: int = AR_WINDOW) -> np.ndarray:
    """AR(p) rolling: выбор p по min RMSE на последних 20 точках."""
    k = min(window, len(series))
    if k < p_max + 10:
        return forecast_ar1(series, h)

    history = series[-k:]
    best_p, best_rmse = 1, np.inf

    for p in range(1, min(p_max + 1, k // 3)):
        # Строим матрицу задержек
        X_arr = np.stack([history[i:k-p+i] for i in range(p)], axis=1)  # (k-p, p)
        y_arr = history[p:]
        if len(y_arr) < 5:
            continue
        # OLS
        try:
            coeffs, _, _, _ = np.linalg.lstsq(
                np.hstack([np.ones((len(X_arr),1)), X_arr]), y_arr, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = np.hstack([np.ones((len(X_arr),1)), X_arr]) @ coeffs
        rmse = float(np.sqrt(np.mean((pred - y_arr)**2)))
        if rmse < best_rmse:
            best_rmse = rmse; best_p = p

    # Итеративный прогноз с выбранным p
    p = best_p
    X_arr = np.stack([history[i:k-p+i] for i in range(p)], axis=1)
    y_arr = history[p:]
    try:
        coeffs, _, _, _ = np.linalg.lstsq(
            np.hstack([np.ones((len(X_arr),1)), X_arr]), y_arr, rcond=None)
    except np.linalg.LinAlgError:
        return forecast_ar1(series, h)

    intercept = coeffs[0]; ar_coeffs = coeffs[1:]
    buf = list(series[-p:])
    hat = []
    for _ in range(h):
        v = float(intercept + np.dot(ar_coeffs, buf[-p:][::-1]))
        hat.append(v); buf.append(v)
    return np.array(hat)


def forecast_ou_fixed(series: np.ndarray, h: int, theta: float) -> np.ndarray:
    """Ornstein-Uhlenbeck: ĉ[t+h] = c[t] × exp(−θ·h)."""
    c0 = series[-1]
    return np.array([c0 * math.exp(-theta * i) for i in range(1, h + 1)])


def forecast_lwr(series: np.ndarray, horizon: int, p: int, xi: int) -> np.ndarray:
    X, y = build_delay_matrix(series, p)
    if len(X) < xi: return np.zeros(horizon)
    v = last_vector(series, p=p).copy(); hat = np.empty(horizon)
    for h in range(horizon):
        dists  = np.linalg.norm(X - v, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        h_bw   = max(float(dists[nn_idx].max()), EPS)
        w      = np.exp(-0.5 * (dists[nn_idx] / h_bw)**2)
        A      = np.hstack([np.ones((xi,1)), X[nn_idx]]); sw = np.sqrt(w)
        c,_,_,_ = np.linalg.lstsq(sw[:,None]*A, sw*y[nn_idx], rcond=None)
        hat[h] = float(c[0] + v @ c[1:]); v = np.roll(v,-1); v[-1] = hat[h]
    return hat


# ── расчёт θ для OU из данных ─────────────────────────────────────────────────

def estimate_theta(series: np.ndarray) -> float:
    """θ = −ln(ACF[1]) из полной истории. Clamp: min 0, max 5."""
    if len(series) < 20: return 1.0
    rho1 = float(compute_acf(series, nlags=1, fft=True)[1])
    if rho1 <= 0 or rho1 >= 1:
        return 2.0  # дефолт для анти-персистентного
    return max(0.0, min(5.0, -math.log(rho1)))


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("46 — Mean-reversion forecast для C0")
print("=" * 70)
print(f"  Тикеры: {len(TICKERS)}, origins: {N_ORIG}/ticker, val_h: {VAL_H}")

# ── Шаг 1: оцениваем θ (OU) по всем тикерам ──────────────────────────────────

print("\n╔══ Шаг 1: оценка ACF[1] и θ (OU) для C0 ══╗")
thetas = []
for ticker in TICKERS:
    ratio, dratio = load_data(ticker)
    COMP = make_fb(dratio, STD_CUTOFFS, FILTER_ORDER)
    c0 = COMP[C0_IDX]
    rho1 = float(compute_acf(c0, nlags=1, fft=True)[1])
    th = estimate_theta(c0)
    thetas.append(th)
    print(f"  {ticker}: ACF[1]={rho1:.4f}  θ={th:.4f}")

theta_med = float(np.median(thetas))
print(f"\n  Медиана θ: {theta_med:.4f}  ↔  exp(−θ) = {math.exp(-theta_med):.4f}")
print(f"  Интерпретация: ĉ[t+1] = c[t] × {math.exp(-theta_med):.3f}  "
      f"(≈ ACF[1] по определению)")


# ── Шаг 2: Per-component walk-forward для C0 ─────────────────────────────────

print("\n╔══ Шаг 2: Walk-forward MAE для C0 (методы) ══╗")

METHODS_C0 = ["zero", "persistence", "ar1", "arp", "ou_fixed", "LWR"]
N_METH = len(METHODS_C0)

# mae_c0_all[ticker_idx, method_idx, horizon_idx]
mae_c0_all = []

t0 = time.time()
for ticker in TICKERS:
    ratio, dratio = load_data(ticker)
    n = len(dratio)
    COMP_full = make_fb(dratio, STD_CUTOFFS, FILTER_ORDER)
    theta_t = estimate_theta(COMP_full[C0_IDX])

    min_o = P_LWR + XI_LWR + 10
    max_o = n - VAL_H - 2
    origins = np.arange(max(min_o, max_o - N_ORIG), max_o)

    mae_sum = np.zeros((N_METH, VAL_H))
    cnt     = np.zeros((N_METH, VAL_H), dtype=int)

    for vo in origins:
        COMP_h = make_fb(dratio[:vo], STD_CUTOFFS, FILTER_ORDER)
        hist   = COMP_h[C0_IDX]
        actual = COMP_full[C0_IDX][vo + 1: vo + 1 + VAL_H]
        h_act  = len(actual)
        if h_act == 0: continue

        hats = [
            forecast_zero(VAL_H),
            forecast_persistence(hist, VAL_H),
            forecast_ar1(hist, VAL_H),
            forecast_arp(hist, VAL_H),
            forecast_ou_fixed(hist, VAL_H, theta_t),
            forecast_lwr(hist, VAL_H, P_LWR, XI_LWR),
        ]

        for mi, hat in enumerate(hats):
            err = np.abs(hat[:h_act] - actual)
            mae_sum[mi, :h_act] += err
            cnt[mi, :h_act]     += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mae = np.where(cnt > 0, mae_sum / cnt, np.nan)
    mae_c0_all.append(mae)

print(f"  Walk-forward C0: {time.time()-t0:.1f}s")

mae_c0_all = np.array(mae_c0_all)   # (8, N_METH, VAL_H)
mae_c0_med = np.nanmedian(mae_c0_all, axis=0)  # (N_METH, VAL_H)

# Gain vs zero
mi_zero = 0
print("\n  MAE компоненты C0 (медиана по 8 тикерам):")
print(f"  {'Метод':14} | {'h=1':>8} | {'h=3':>8} | {'h=5':>8} | {'h=10':>8} | {'Gain@h=1':>9} | {'Gain@h=5':>9}")
print("  " + "-" * 75)
for mi, method in enumerate(METHODS_C0):
    g1 = mae_c0_med[mi_zero, 0] / (mae_c0_med[mi, 0] + EPS)
    g5 = mae_c0_med[mi_zero, 4] / (mae_c0_med[mi, 4] + EPS)
    mark = " ←" if method == "zero" else ""
    print(f"  {method:<14} | {mae_c0_med[mi,0]:.4e} | {mae_c0_med[mi,2]:.4e} | "
          f"{mae_c0_med[mi,4]:.4e} | {mae_c0_med[mi,9]:.4e} | {g1:>9.4f} | {g5:>9.4f}{mark}")


# ── Шаг 3: sweep θ для OU ────────────────────────────────────────────────────

print("\n╔══ Шаг 3: Sweep θ (OU) для C0 ══╗")
THETAS = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

mae_ou_theta = {}  # {θ: mae (VAL_H,)}

for theta_val in THETAS:
    mae_sum = np.zeros((len(TICKERS), VAL_H))
    cnt_arr = np.zeros((len(TICKERS), VAL_H), dtype=int)

    for ti, ticker in enumerate(TICKERS):
        ratio, dratio = load_data(ticker)
        n = len(dratio)
        COMP_full = make_fb(dratio, STD_CUTOFFS, FILTER_ORDER)
        min_o = P_LWR + XI_LWR + 10; max_o = n - VAL_H - 2
        origins = np.arange(max(min_o, max_o - N_ORIG), max_o)

        for vo in origins:
            COMP_h = make_fb(dratio[:vo], STD_CUTOFFS, FILTER_ORDER)
            hist   = COMP_h[C0_IDX]
            actual = COMP_full[C0_IDX][vo + 1: vo + 1 + VAL_H]
            h_act  = len(actual)
            if h_act == 0: continue
            hat = forecast_ou_fixed(hist, VAL_H, theta_val)
            err = np.abs(hat[:h_act] - actual)
            mae_sum[ti, :h_act] += err
            cnt_arr[ti, :h_act] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mae_t = np.where(cnt_arr > 0, mae_sum / cnt_arr, np.nan)
    mae_ou_theta[theta_val] = float(np.nanmedian(mae_t[:, 0]))   # h=1

zero_mae_c0_h1 = float(mae_c0_med[mi_zero, 0])
print(f"\n  Zero MAE C0 @ h=1: {zero_mae_c0_h1:.5e}")
print(f"  {'θ':>8} | {'OU MAE h=1':>12} | {'Gain vs zero':>13}")
print("  " + "-" * 40)
for theta_val, mae_val in mae_ou_theta.items():
    gain = zero_mae_c0_h1 / (mae_val + EPS)
    mark = " ← теор." if abs(theta_val - theta_med) < 0.01 else ""
    print(f"  {theta_val:>8.2f} | {mae_val:>12.5e} | {gain:>13.5f}{mark}")

best_theta = min(mae_ou_theta, key=mae_ou_theta.get)
print(f"\n  Оптимальный θ: {best_theta}  MAE={mae_ou_theta[best_theta]:.5e}")


# ── Шаг 4: C1 — проверяем AR(1)/persistence ──────────────────────────────────

print("\n╔══ Шаг 4: C1 (19.7% энергии, ACF[1]≈0.90) — AR(1)/persistence ══╗")

METHODS_C1 = ["zero", "persistence", "ar1"]
mae_c1_all = []

t0 = time.time()
for ticker in TICKERS:
    ratio, dratio = load_data(ticker)
    n = len(dratio)
    COMP_full = make_fb(dratio, STD_CUTOFFS, FILTER_ORDER)
    min_o = P_LWR + XI_LWR + 10; max_o = n - VAL_H - 2
    origins = np.arange(max(min_o, max_o - N_ORIG), max_o)

    mae_sum = np.zeros((3, VAL_H)); cnt = np.zeros((3, VAL_H), dtype=int)
    for vo in origins:
        COMP_h = make_fb(dratio[:vo], STD_CUTOFFS, FILTER_ORDER)
        hist   = COMP_h[C1_IDX]
        actual = COMP_full[C1_IDX][vo + 1: vo + 1 + VAL_H]
        h_act  = len(actual)
        if h_act == 0: continue
        hats = [forecast_zero(VAL_H), forecast_persistence(hist, VAL_H), forecast_ar1(hist, VAL_H)]
        for mi, hat in enumerate(hats):
            err = np.abs(hat[:h_act] - actual)
            mae_sum[mi, :h_act] += err; cnt[mi, :h_act] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = np.where(cnt > 0, mae_sum / cnt, np.nan)
    mae_c1_all.append(mae)

print(f"  Walk-forward C1: {time.time()-t0:.1f}s")

mae_c1_med = np.nanmedian(np.array(mae_c1_all), axis=0)
print(f"\n  MAE компоненты C1:")
print(f"  {'Метод':14} | {'h=1':>8} | {'h=3':>8} | {'h=5':>8} | {'Gain@h=1':>9}")
print("  " + "-" * 55)
for mi, method in enumerate(METHODS_C1):
    g1 = mae_c1_med[0, 0] / (mae_c1_med[mi, 0] + EPS)
    print(f"  {method:<14} | {mae_c1_med[mi,0]:.4e} | {mae_c1_med[mi,2]:.4e} | "
          f"{mae_c1_med[mi,4]:.4e} | {g1:>9.2f}")


# ── Шаг 5: финальный MAPE — комбинации ──────────────────────────────────────

print("\n╔══ Шаг 5: Реконструированный AGG MAPE — комбинации ══╗")

# Конфигурации:
# std:      C3-C5 LWR + C0 zero + C1 zero     ← стандарт
# c0_ar1:   C3-C5 LWR + C0 AR(1) + C1 zero
# c0_ou:    C3-C5 LWR + C0 OU(θ=best) + C1 zero
# c0c1_ar1: C3-C5 LWR + C0 AR(1) + C1 AR(1)
# c0c1_pers:C3-C5 LWR + C0 zero + C1 persistence

CONFIGS = {
    "std":        {"c0": "zero",    "c1": "zero",        "slow": "LWR"},
    "c0_ar1":     {"c0": "ar1",     "c1": "zero",        "slow": "LWR"},
    f"c0_ou{best_theta:.0f}": {"c0": f"ou{best_theta}", "c1": "zero", "slow": "LWR"},
    "c0c1_ar1":   {"c0": "ar1",     "c1": "ar1",         "slow": "LWR"},
    "c1_pers":    {"c0": "zero",    "c1": "persistence",  "slow": "LWR"},
    "c0ar1+c1p":  {"c0": "ar1",     "c1": "persistence",  "slow": "LWR"},
}

mape_configs: dict[str, list] = {cfg: [] for cfg in CONFIGS}

t0 = time.time()
for ticker in TICKERS:
    ratio, dratio = load_data(ticker)
    n = len(dratio)
    theta_t = estimate_theta(make_fb(dratio, STD_CUTOFFS, FILTER_ORDER)[C0_IDX])
    min_o = P_LWR + XI_LWR + 10; max_o = n - VAL_H - 2
    origins = np.arange(max(min_o, max_o - N_ORIG), max_o)

    for vo in origins:
        COMP_h = make_fb(dratio[:vo], STD_CUTOFFS, FILTER_ORDER)
        actual_ratio = ratio[vo + 1: vo + 1 + VAL_H]
        n_act = len(actual_ratio)
        if n_act == 0: continue
        r0 = float(ratio[vo])

        # Slow компоненты (одинаковые для всех конфигураций)
        slow_hats = [forecast_lwr(COMP_h[ci], VAL_H, P_LWR, XI_LWR) for ci in SLOW_IDX]
        slow_sum = np.sum(slow_hats, axis=0)

        # C0 / C1 hat по конфигурации
        h_c0 = {
            "zero":        forecast_zero(VAL_H),
            "ar1":         forecast_ar1(COMP_h[C0_IDX], VAL_H),
            f"ou{best_theta}": forecast_ou_fixed(COMP_h[C0_IDX], VAL_H, best_theta),
        }
        h_c1 = {
            "zero":        forecast_zero(VAL_H),
            "ar1":         forecast_ar1(COMP_h[C1_IDX], VAL_H),
            "persistence": forecast_persistence(COMP_h[C1_IDX], VAL_H),
        }

        for cfg_name, cfg in CONFIGS.items():
            c0_key = cfg["c0"] if cfg["c0"] in h_c0 else "zero"
            c1_key = cfg["c1"] if cfg["c1"] in h_c1 else "zero"
            dratio_hat = slow_sum + h_c0[c0_key] + h_c1[c1_key]
            r_hat = r0 + np.cumsum(dratio_hat)
            mape  = float(np.mean(
                np.abs(r_hat[:n_act] - actual_ratio) / (np.abs(actual_ratio) + EPS)
            ))
            mape_configs[cfg_name].append(mape)

print(f"  Combined MAPE: {time.time()-t0:.1f}s")

mape_agg = {cfg: float(np.nanmedian(mape_configs[cfg])) for cfg in CONFIGS}
mape_std = mape_agg["std"]

print(f"\n  {'Конфигурация':25} | {'AGG MAPE':>9} | {'Δ% vs std':>10}")
print("  " + "-" * 50)
for cfg_name, mape in mape_agg.items():
    d = (mape - mape_std) / mape_std * 100
    mark = " ← стандарт" if cfg_name == "std" else (" ★" if d < -0.5 else "")
    print(f"  {cfg_name:<25} | {mape:>9.5f} | {d:>+9.2f}%{mark}")


# ══════════════════════════════════════════════════════════════════════════════
#  ГРАФИКИ
# ══════════════════════════════════════════════════════════════════════════════

METHOD_STYLES = {
    "zero":        {"color": "gray",       "ls": ":",  "lw": 1.5},
    "persistence": {"color": "steelblue",  "ls": "--", "lw": 1.8},
    "ar1":         {"color": "green",      "ls": "-.", "lw": 1.8},
    "arp":         {"color": "darkorange", "ls": "-.", "lw": 2.0},
    "ou_fixed":    {"color": "purple",     "ls": "-",  "lw": 1.8},
    "LWR":         {"color": "crimson",    "ls": "-",  "lw": 2.5},
}
horizons = np.arange(1, VAL_H + 1)

# ── Рис 1: MAE vs горизонт для C0 ────────────────────────────────────────────

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle(
    f"C0 (2–4 bar, 63.4% энергии): методы прогноза  |  val_h={VAL_H}",
    fontsize=12, fontweight="bold"
)

ax_mae = axes1[0]
for mi, method in enumerate(METHODS_C0):
    st = METHOD_STYLES.get(method, {"color": "black", "ls": "-", "lw": 1.5})
    ax_mae.plot(horizons, mae_c0_med[mi], color=st["color"], ls=st["ls"],
                lw=st["lw"], label=method, marker=".", ms=4, markevery=2)
ax_mae.set_title("MAE компоненты C0 vs горизонт", fontsize=10)
ax_mae.set_xlabel("Горизонт h"); ax_mae.set_ylabel("MAE C0")
ax_mae.legend(fontsize=9); ax_mae.grid(alpha=0.35)

ax_gain = axes1[1]
for mi, method in enumerate(METHODS_C0):
    if method == "zero": continue
    gain = mae_c0_med[mi_zero] / (mae_c0_med[mi] + EPS)
    st = METHOD_STYLES.get(method, {"color": "black", "ls": "-", "lw": 1.5})
    ax_gain.plot(horizons, gain, color=st["color"], ls=st["ls"],
                 lw=st["lw"], label=method, marker=".", ms=4, markevery=2)
ax_gain.axhline(1.0, color="black", lw=1.5, ls="--", label="Gain=1 (паритет)")
ax_gain.set_title("Gain(method / zero) для C0", fontsize=10)
ax_gain.set_xlabel("Горизонт h"); ax_gain.set_ylabel("Gain")
ax_gain.legend(fontsize=9); ax_gain.grid(alpha=0.35)

plt.tight_layout()
out1 = OUT_DIR / "46_c0_methods.png"
fig1.savefig(out1, dpi=140, bbox_inches="tight")
print(f"\n  Рис 1: {out1}")
plt.close(fig1)


# ── Рис 2: OU theta sweep ─────────────────────────────────────────────────────

fig2, ax2 = plt.subplots(figsize=(8, 5))
thetas_arr = np.array(THETAS)
ou_maes    = np.array([mae_ou_theta[th] for th in THETAS])
gain_arr   = zero_mae_c0_h1 / (ou_maes + EPS)

ax2.bar(range(len(THETAS)), gain_arr, color=["seagreen" if g > 1 else "tomato" for g in gain_arr],
        alpha=0.8, edgecolor="k", lw=0.5)
ax2.axhline(1.0, color="black", lw=1.5, ls="--", label="Gain=1")
ax2.axvline(THETAS.index(best_theta), color="green", lw=2, ls=":", label=f"Best θ={best_theta}")
ax2.axvline(THETAS.index(min(THETAS, key=lambda t: abs(t - theta_med))),
            color="blue", lw=1.5, ls="--", alpha=0.7, label=f"θ=−ln(ACF1)={theta_med:.2f}")
ax2.set_xticks(range(len(THETAS))); ax2.set_xticklabels([str(t) for t in THETAS])
ax2.set_xlabel("θ (параметр OU)", fontsize=10)
ax2.set_ylabel("Gain(OU / zero) @ h=1", fontsize=10)
ax2.set_title(f"Sweep θ для OU: Gain(OU / zero) при h=1", fontsize=10)
ax2.legend(fontsize=9); ax2.grid(alpha=0.35, axis="y")

plt.tight_layout()
out2 = OUT_DIR / "46_ou_theta_sweep.png"
fig2.savefig(out2, dpi=140, bbox_inches="tight")
print(f"  Рис 2: {out2}")
plt.close(fig2)


# ── Рис 3: MAPE bar chart по конфигурациям ───────────────────────────────────

fig3, ax3 = plt.subplots(figsize=(10, 5))
cfg_names = list(CONFIGS.keys())
mape_vals = [mape_agg[cfg] for cfg in cfg_names]
deltas    = [(m - mape_std) / mape_std * 100 for m in mape_vals]
colors3   = ["gray"] + ["seagreen" if d < 0 else "tomato" for d in deltas[1:]]
bars = ax3.bar(cfg_names, mape_vals, color=colors3, alpha=0.8, edgecolor="k", lw=0.8)

for bar, val, d in zip(bars, mape_vals, deltas):
    label = f"{val:.4f}\n({d:+.2f}%)"
    ax3.text(bar.get_x() + bar.get_width()/2, val + max(mape_vals) * 0.008,
             label, ha="center", va="bottom", fontsize=9, fontweight="bold")

ax3.set_ylabel("Медиана AGG MAPE (цена)", fontsize=10)
ax3.set_title(
    f"MAPE по конфигурациям (C0/C1-прогноз)  |  8 тикеров 1d, val_h={VAL_H}",
    fontsize=10
)
ax3.grid(alpha=0.35, axis="y"); ax3.tick_params(axis="x", rotation=20)
ax3.set_ylim(0, max(mape_vals) * 1.18)

plt.tight_layout()
out3 = OUT_DIR / "46_mape_configs.png"
fig3.savefig(out3, dpi=140, bbox_inches="tight")
print(f"  Рис 3: {out3}")
plt.close(fig3)


# ── Итоги ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ИТОГИ")
print("=" * 70)

mi_best = np.argmin(mae_c0_med[:, 0])
print(f"\n1. Лучший метод для C0 @ h=1: {METHODS_C0[mi_best]}")
print(f"   MAE: {mae_c0_med[mi_best, 0]:.5e}  vs zero: {mae_c0_med[mi_zero, 0]:.5e}")
print(f"   Gain: {mae_c0_med[mi_zero, 0]/(mae_c0_med[mi_best, 0]+EPS):.4f}")

print(f"\n2. OU theta оптимальный: θ={best_theta}  Gain@h=1={mae_c0_med[mi_zero,0]/(mae_ou_theta[best_theta]+EPS):.4f}")
print(f"   Теоретический θ=−ln(ACF[1])={theta_med:.3f}")

best_cfg = min(mape_agg, key=mape_agg.get)
print(f"\n3. Лучшая конфигурация: {best_cfg}")
print(f"   AGG MAPE: {mape_agg[best_cfg]:.5f}  Δ={((mape_agg[best_cfg]-mape_std)/mape_std*100):+.2f}% vs std")

print(f"\n4. C1 Gain(pers/zero) @ h=1: {mae_c1_med[0,0]/(mae_c1_med[1,0]+EPS):.2f}×")
print(f"   C1 Gain(ar1/zero)  @ h=1: {mae_c1_med[0,0]/(mae_c1_med[2,0]+EPS):.2f}×")

print(f"\nФайлы:")
for out in [out1, out2, out3]:
    print(f"  {out.name}")
print("=" * 70)
