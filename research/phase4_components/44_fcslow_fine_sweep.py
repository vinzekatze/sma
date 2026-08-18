"""
44 — Точное определение оптимальной полосы сигнала для LWR.

Вопрос: при каком точном пороге (периоде) заканчивается «полезный» для LWR
сигнал и начинается шум, включение которого ухудшает прогноз?

Скрипт 43 показал: fc_slow=0.0625 (>16 bar) лучше, чем >8 или >32 bar.
Но тестировались только октавные точки. Здесь — тонкий sweep с шагом 1–2 бара.

Эксперимент A — Fine sweep fc_slow:
  T ∈ {6..80 баров}, шаг 1–2 бара в критической зоне 8–26 баров.
  Для каждого T: cutoffs с fc_slow=1/T + 3 slow sub-bands octave ниже.
  Метрика: медиана AGG MAPE, val_h=10, 8 тикеров, 30 origins.

Эксперимент B — Filter order:
  При оптимальном T*, sweep filter_order ∈ {2, 4, 6, 8}.
  Показывает: помогает ли более крутой спад АЧХ.

Дополнительно: вычисляется Хёрст и PE «пограничной» узкой полосы
  [fc_slow × 0.7, fc_slow × 1.4] для каждого T — объясняет форму кривой MAPE.
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

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from sma.core.forecast.embedding import build_delay_matrix, last_vector

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS  = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL = "1d"
EPS      = 1e-10
P_LWR    = 20
XI_LWR   = 3 * (P_LWR + 1)
VAL_H    = 10
N_ORIG   = 30

# Эксперимент A: тестируемые периоды (ось X = период-граница в барах)
TEST_PERIODS = [
     6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 22, 24, 26, 28, 32,
    40, 52, 64, 80,
]

# Эксперимент B: sweep filter order при оптимальном T
FILTER_ORDERS = [2, 4, 6, 8]
STANDARD_PERIOD = 16
STANDARD_ORDER  = 4

# Хёрст: данные 12-band из скрипта 42 (медиана по 8 тикерам)
# Центральный период полосы: геом. среднее границ
HURST_12BAND = {
    # (lower_period, upper_period): H
    (2.0,  2.83): 0.274,
    (2.83, 4.0):  0.271,
    (4.0,  5.66): 0.296,
    (5.66, 8.0):  0.316,
    (8.0,  11.3): 0.355,
    (11.3, 16.0): 0.394,
    (16.0, 22.6): 0.444,
    (22.6, 32.0): 0.504,
    (32.0, 45.3): 0.580,
    (45.3, 64.0): 0.657,
    (64.0, 90.5): 0.746,
    (90.5, 512.): 0.856,
}
# Переводим в (center_period, H) для интерполяции
_HURST_CURVE = sorted(
    ((math.sqrt(lo * hi), h) for (lo, hi), h in HURST_12BAND.items()),
    key=lambda x: x[0]
)

def hurst_at_period(p: float) -> float:
    """Линейная интерполяция Хёрста по данным 12-band."""
    xs = [x for x, _ in _HURST_CURVE]
    ys = [y for _, y in _HURST_CURVE]
    if p <= xs[0]:  return ys[0]
    if p >= xs[-1]: return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= p <= xs[i + 1]:
            t = (p - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return float("nan")


# ── нормализация + данные ─────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    log_c = np.log(close + EPS)
    n = len(log_c)
    t = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(log_c); cty = np.cumsum(t * log_c)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = a + b * t
    trend[:2] = log_c[:2]
    return np.exp(trend)


_CACHE: dict = {}

def load_dratio(ticker: str) -> tuple[np.ndarray, np.ndarray]:
    if ticker not in _CACHE:
        with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
            c = json.load(f)
        close = np.array([x["close"] for x in c], dtype=np.float64)
        ratio = close / logtrend_causal(close)
        _CACHE[ticker] = (ratio, np.diff(ratio))
    return _CACHE[ticker]


# ── filter bank ────────────────────────────────────────────────────────────────

def make_fb(series: np.ndarray, cutoffs: list, order: int = 4) -> np.ndarray:
    comps = []; remaining = series.copy()
    for fc in cutoffs:
        sos = butter(order, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        comps.append(remaining - low)
        remaining = low
    comps.append(remaining)
    return np.array(comps)


def cutoffs_for_period(T: float, n_slow: int = 3, order: int = 4) -> tuple[list, list]:
    """
    Строит cutoffs для порога T (период в барах).
    Возвращает (all_cutoffs, slow_indices).
    """
    fc_slow = 1.0 / T

    # HF: октавные ступени от 0.25 вниз, остановка перед fc_slow
    hf = []
    fc = 0.25
    while fc > fc_slow * 1.05:
        hf.append(round(fc, 8))
        fc /= 2.0
    hf.append(round(fc_slow, 8))

    # Slow: n_slow-1 октавных ступеней ниже fc_slow
    slow_cuts = [round(fc_slow / (2 ** k), 8)
                 for k in range(1, n_slow) if fc_slow / (2 ** k) > 0.001]

    all_cuts = hf + slow_cuts
    n_bands  = len(all_cuts) + 1

    # Индексы slow компонент: те, у которых верхняя граница ≤ fc_slow
    s_idx = []
    for i in range(n_bands):
        upper = 0.5 if i == 0 else all_cuts[i - 1]
        if upper <= fc_slow + EPS:
            s_idx.append(i)
    return all_cuts, s_idx


# ── LWR ───────────────────────────────────────────────────────────────────────

def lwr_forecast(series: np.ndarray, horizon: int, p: int, xi: int) -> np.ndarray:
    X, y = build_delay_matrix(series, p)
    if len(X) < xi:
        return np.zeros(horizon)
    v = last_vector(series, p=p).copy()
    hat = np.empty(horizon)
    for h in range(horizon):
        dists  = np.linalg.norm(X - v, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        h_bw   = max(float(dists[nn_idx].max()), EPS)
        w      = np.exp(-0.5 * (dists[nn_idx] / h_bw) ** 2)
        A      = np.hstack([np.ones((xi, 1)), X[nn_idx]])
        sw     = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx], rcond=None)
        hat[h] = float(c[0] + v @ c[1:])
        v = np.roll(v, -1); v[-1] = hat[h]
    return hat


def run_config(T: float, order: int = 4, n_orig: int = N_ORIG) -> dict:
    """
    Walk-forward для порога T и filter order.
    Возвращает {ticker: mapes_array}.
    """
    cuts, s_idx = cutoffs_for_period(T, order=order)
    if not s_idx:
        return {t: np.full(n_orig, np.nan) for t in TICKERS}

    res = {}
    for ticker in TICKERS:
        ratio, dratio = load_dratio(ticker)
        n = len(dratio)
        COMP_full = make_fb(dratio, cuts, order=order)

        min_o = P_LWR + XI_LWR + 10
        max_o = n - VAL_H - 2
        origins = np.arange(max(min_o, max_o - n_orig), max_o)

        mapes = []
        for vo in origins:
            COMP_h = make_fb(dratio[:vo], cuts, order=order)
            hats = [lwr_forecast(COMP_h[ci], VAL_H, P_LWR, XI_LWR) for ci in s_idx]
            dratio_hat = np.sum(hats, axis=0)
            r0     = float(ratio[vo])
            r_hat  = r0 + np.cumsum(dratio_hat)
            actual = ratio[vo + 1: vo + 1 + VAL_H]
            n_act  = min(len(r_hat), len(actual))
            if n_act:
                mapes.append(float(np.mean(
                    np.abs(r_hat[:n_act] - actual[:n_act]) /
                    (np.abs(actual[:n_act]) + EPS)
                )))
        res[ticker] = np.array(mapes) if mapes else np.full(n_orig, np.nan)
    return res


def agg_mape(res: dict) -> float:
    all_v = np.concatenate(list(res.values()))
    return float(np.nanmedian(all_v[~np.isnan(all_v)]))


# ── диагностика узкой полосы (Хёрст + PE для «пограничной» компоненты) ────────

def hurst_rs(s: np.ndarray, min_n: int = 20) -> float:
    n = len(s)
    if n < min_n * 2: return np.nan
    lags = np.unique(np.logspace(np.log10(min_n), np.log10(n // 2), 12).astype(int))
    rs_vals = []
    for lag in lags:
        sub_rs = []; sub_arr = s
        for start in range(0, n - lag, lag):
            sub = sub_arr[start: start + lag]
            dev = np.cumsum(sub - sub.mean())
            S = sub.std()
            if S > 1e-12: sub_rs.append((dev.max() - dev.min()) / S)
        if sub_rs: rs_vals.append(float(np.mean(sub_rs)))
    if len(rs_vals) < 3: return np.nan
    lx = np.log(lags[: len(rs_vals)].astype(float))
    return float(np.polyfit(lx, np.log(np.array(rs_vals)), 1)[0])


def border_hurst(T: float) -> float:
    """Хёрст узкой полосы, центрированной на fc_slow = 1/T (ширина октава)."""
    fc_slow = 1.0 / T
    fc_lo   = fc_slow / math.sqrt(2)
    fc_hi   = fc_slow * math.sqrt(2)
    fc_hi   = min(fc_hi, 0.49)
    # Берём SBER как представитель
    _, dratio = load_dratio("SBER")
    sos_hi = butter(4, fc_hi, btype="low", output="sos")
    sos_lo = butter(4, fc_lo, btype="low", output="sos")
    lp_hi  = sosfilt(sos_hi, dratio)
    lp_lo  = sosfilt(sos_lo, dratio)
    band   = lp_hi - lp_lo
    return hurst_rs(band)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("44 — Fine sweep fc_slow: определение оптимальной полосы")
print("=" * 70)

# ── Эксперимент A ─────────────────────────────────────────────────────────────
print(f"\n╔══ Эксперимент A: fine sweep T={TEST_PERIODS[0]}..{TEST_PERIODS[-1]} bar ══╗")

RES_A: dict[int, dict]   = {}
MAPE_A: dict[int, float] = {}
H_BORDER: dict[int, float] = {}

t0 = time.time()
for T in TEST_PERIODS:
    cuts, s_idx = cutoffs_for_period(T)
    t1 = time.time()
    res = run_config(T)
    RES_A[T] = res
    MAPE_A[T] = agg_mape(res)
    H_BORDER[T] = border_hurst(T)   # Хёрст пограничной полосы
    std_marker = " ← стандарт" if T == STANDARD_PERIOD else ""
    print(f"  T={T:>3} bar | fc={1/T:.5f} | slow_idx={s_idx} "
          f"| MAPE={MAPE_A[T]:.5f} | H_border={H_BORDER[T]:.3f}"
          f"{std_marker}  ({time.time()-t1:.1f}s)")

print(f"\n  Total A: {time.time()-t0:.1f}s")

# Находим оптимум
T_opt = min(MAPE_A, key=MAPE_A.get)
mape_opt = MAPE_A[T_opt]
mape_std = MAPE_A[STANDARD_PERIOD]
delta_opt = (mape_opt - mape_std) / mape_std * 100
print(f"\n  Оптимум: T={T_opt} bar  MAPE={mape_opt:.5f}  Δ={delta_opt:+.2f}% vs стандарт (T=16)")

# ── Эксперимент B: filter order sweep ─────────────────────────────────────────
print(f"\n╔══ Эксперимент B: filter order sweep @ T={T_opt} bar ══╗")

RES_B: dict[int, dict]   = {}
MAPE_B: dict[int, float] = {}

t0b = time.time()
for order in FILTER_ORDERS:
    t1 = time.time()
    res = run_config(T_opt, order=order)
    RES_B[order] = res
    MAPE_B[order] = agg_mape(res)
    d = (MAPE_B[order] - mape_opt) / mape_opt * 100
    std_mark = " ← стандарт" if order == STANDARD_ORDER else ""
    print(f"  order={order} | MAPE={MAPE_B[order]:.5f}  Δ={d:+.2f}%{std_mark}  ({time.time()-t1:.1f}s)")

order_opt = min(MAPE_B, key=MAPE_B.get)
print(f"\n  Total B: {time.time()-t0b:.1f}s")
print(f"  Оптимальный order: {order_opt}  MAPE={MAPE_B[order_opt]:.5f}")


# ── Сводная таблица ───────────────────────────────────────────────────────────

print("\n" + "═" * 70)
print("СВОДНАЯ ТАБЛИЦА — Эксперимент A")
print("═" * 70)
ticker_hdr = "  ".join(f"{t:>7}" for t in TICKERS)
print(f"{'T':>5} {'fc_slow':>8} | {ticker_hdr} | {'AGG':>8} | {'Δ%':>7} | {'H_bord':>7}")
print("─" * 110)
for T in TEST_PERIODS:
    per_t = "  ".join(f"{float(np.nanmedian(RES_A[T][t])):>7.5f}" for t in TICKERS)
    d = (MAPE_A[T] - mape_std) / mape_std * 100
    mark = " *" if T == T_opt else (" ←" if T == STANDARD_PERIOD else "")
    print(f"{T:>5} {1/T:>8.5f} | {per_t} | {MAPE_A[T]:>8.5f} | {d:>+6.2f}%{mark} | {H_BORDER[T]:>7.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  ГРАФИКИ
# ══════════════════════════════════════════════════════════════════════════════

periods_arr = np.array(TEST_PERIODS)
mapes_arr   = np.array([MAPE_A[T] for T in TEST_PERIODS])
h_arr       = np.array([H_BORDER[T] for T in TEST_PERIODS])
deltas_arr  = (mapes_arr - mape_std) / mape_std * 100

# Интерполируем кривую Хёрста из 12-band данных
h_interp = np.array([hurst_at_period(T) for T in TEST_PERIODS])

PALETTE = plt.cm.tab10(np.linspace(0, 1, len(TICKERS)))

# ── Рис 1: MAPE vs период (главный результат) ─────────────────────────────────

fig1, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
fig1.suptitle(
    f"Fine sweep fc_slow  |  p={P_LWR}, ξ={XI_LWR}, val_h={VAL_H}  |  8 тикеров 1d",
    fontsize=12, fontweight="bold"
)

ax1 = axes[0]
# Per-ticker (светлые линии)
for i, ticker in enumerate(TICKERS):
    t_mapes = [float(np.nanmedian(RES_A[T][ticker])) for T in TEST_PERIODS]
    ax1.plot(periods_arr, t_mapes, color=PALETTE[i], lw=1.2, alpha=0.55,
             marker=".", ms=4, label=ticker)

# AGG (жирная)
ax1.plot(periods_arr, mapes_arr, "k-o", lw=2.5, ms=6, zorder=5, label="Aggregate")

# Маркеры
ax1.axvline(STANDARD_PERIOD, color="red",   lw=1.8, ls="--",
            label=f"Стандарт T={STANDARD_PERIOD}")
ax1.axvline(T_opt,           color="green", lw=1.8, ls=":",
            label=f"Оптимум T={T_opt}  (Δ={delta_opt:+.2f}%)")

# Зона H > 0.5 (серая заливка)
h05_lo = next((T for T in TEST_PERIODS if hurst_at_period(T) >= 0.5), None)
if h05_lo:
    ax1.axvspan(h05_lo, TEST_PERIODS[-1], color="lightgreen", alpha=0.12,
                label=f"H>0.5 (персистентная зона, T≥{h05_lo})")

ax1.set_ylabel("Медиана AGG MAPE", fontsize=10)
ax1.set_title("MAPE vs период-граница T (компоненты с периодом > T прогнозируются)", fontsize=10)
ax1.legend(fontsize=8, ncol=3, loc="upper right")
ax1.grid(alpha=0.35)

# ── Нижний subplot: Δ% + Хёрст на второй оси ──────────────────────────────────

ax2 = axes[1]
bars_c = ["seagreen" if d <= 0 else "tomato" for d in deltas_arr]
ax2.bar(periods_arr, deltas_arr, color=bars_c, alpha=0.7, width=0.8, edgecolor="k", lw=0.3)
ax2.axhline(0, color="black", lw=1)
ax2.axvline(STANDARD_PERIOD, color="red",   lw=1.5, ls="--")
ax2.axvline(T_opt,           color="green", lw=1.5, ls=":")
ax2.set_ylabel("Δ% vs стандарт T=16", fontsize=9, color="black")
ax2.set_xlabel("Период-граница T (баров)", fontsize=10)
ax2.grid(alpha=0.3, axis="y")

# Хёрст на правой оси (из скр.42, 12-band)
ax2r = ax2.twinx()
ax2r.plot(periods_arr, h_interp, "b-s", lw=1.8, ms=4, alpha=0.7, label="H (12-band)")
ax2r.plot(periods_arr, h_arr,    "b--^", lw=1.2, ms=3, alpha=0.5, label="H border (SBER)")
ax2r.axhline(0.5, color="blue", lw=1, ls=":", alpha=0.7)
ax2r.set_ylabel("Хёрст R/S", fontsize=9, color="blue")
ax2r.tick_params(axis="y", labelcolor="blue")
ax2r.set_ylim(0.2, 1.0)
ax2r.legend(fontsize=8, loc="upper left")

ax2.set_xlim(TEST_PERIODS[0] - 1, TEST_PERIODS[-1] + 2)
ax2.set_xticks(TEST_PERIODS[::2])
ax2.tick_params(axis="x", rotation=30)

plt.tight_layout()
out1 = OUT_DIR / "44_fcslow_fine_sweep.png"
fig1.savefig(out1, dpi=140, bbox_inches="tight")
print(f"\n  Рис 1: {out1}")
plt.close(fig1)


# ── Рис 2: маргинальный вклад + filter order ─────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Маргинальный анализ + влияние порядка фильтра", fontsize=11, fontweight="bold")

# Маргинальный вклад: MAPE(T) - MAPE(T+1) = сколько даёт добавление полосы [T..T+1]
ax3 = axes2[0]
marginal = np.diff(mapes_arr[::-1])[::-1]   # dMAPE/dT (снижение при уменьшении T)
periods_mid = (periods_arr[:-1] + periods_arr[1:]) / 2
colors_m = ["seagreen" if m > 0 else "tomato" for m in marginal]
ax3.bar(periods_mid, -marginal * 1000, color=colors_m, alpha=0.8, width=1.5,
        edgecolor="k", lw=0.3)
ax3.axhline(0, color="black", lw=1)
ax3.axvline(STANDARD_PERIOD, color="red", lw=1.5, ls="--")
ax3.set_xlabel("Полоса «добавляемая» (бары)", fontsize=10)
ax3.set_ylabel("Снижение MAPE × 10³ при добавлении полосы", fontsize=9)
ax3.set_title("Маргинальный вклад каждой частотной полосы\n(зелёный = улучшает, красный = ухудшает)", fontsize=9)
ax3.grid(alpha=0.35, axis="y")
ax3.set_xticks(periods_arr[::2]); ax3.tick_params(axis="x", rotation=30)

# Второй subplot: filter order sweep
ax4 = axes2[1]
orders_arr = np.array(FILTER_ORDERS)
mapes_ord  = np.array([MAPE_B[o] for o in FILTER_ORDERS])
mape_ord_std = MAPE_B[STANDARD_ORDER]
deltas_ord = (mapes_ord - mape_ord_std) / mape_ord_std * 100

# Per-ticker
for i, ticker in enumerate(TICKERS):
    t_m = [float(np.nanmedian(RES_B[o][ticker])) for o in FILTER_ORDERS]
    ax4.plot(orders_arr, t_m, color=PALETTE[i], lw=1.1, alpha=0.5,
             marker=".", ms=4, label=ticker)
ax4.plot(orders_arr, mapes_ord, "k-D", lw=2.5, ms=7, zorder=5, label="Aggregate")
ax4.axvline(STANDARD_ORDER, color="red", lw=1.8, ls="--", label=f"Стандарт order={STANDARD_ORDER}")
ax4.axvline(order_opt,      color="green", lw=1.8, ls=":",
            label=f"Оптимум order={order_opt}  ({(MAPE_B[order_opt]-mape_opt)/mape_opt*100:+.2f}%)")

# Добавляем Δ% над точками
ax4r = ax4.twinx()
ax4r.bar(orders_arr, deltas_ord, alpha=0.2, color=["seagreen" if d <= 0 else "tomato" for d in deltas_ord], width=0.5)
ax4r.set_ylabel("Δ% vs order=4", fontsize=9, color="gray")
ax4r.tick_params(axis="y", labelcolor="gray")
ax4r.axhline(0, color="gray", lw=0.8, ls="--")

ax4.set_xlabel("Порядок фильтра Баттерворта", fontsize=10)
ax4.set_ylabel("Медиана MAPE", fontsize=10)
ax4.set_title(f"Влияние порядка фильтра @ T={T_opt} bar", fontsize=10)
ax4.legend(fontsize=8, loc="upper right")
ax4.grid(alpha=0.35)
ax4.set_xticks(FILTER_ORDERS)

plt.tight_layout()
out2 = OUT_DIR / "44_marginal_order.png"
fig2.savefig(out2, dpi=140, bbox_inches="tight")
print(f"  Рис 2: {out2}")
plt.close(fig2)


# ── Рис 3: per-ticker heatmap ─────────────────────────────────────────────────

fig3, ax5 = plt.subplots(figsize=(13, 5))
mat = np.array([[float(np.nanmedian(RES_A[T][t])) for t in TICKERS]
                for T in TEST_PERIODS])
std_row = TEST_PERIODS.index(STANDARD_PERIOD)
mat_delta = (mat - mat[std_row, :]) / (mat[std_row, :] + EPS) * 100

im = ax5.imshow(mat_delta.T, aspect="auto", cmap="RdYlGn_r", vmin=-10, vmax=20)
ax5.set_yticks(range(len(TICKERS))); ax5.set_yticklabels(TICKERS, fontsize=9)
ax5.set_xticks(range(len(TEST_PERIODS)))
ax5.set_xticklabels([str(T) for T in TEST_PERIODS], rotation=45, ha="right", fontsize=8)
ax5.set_xlabel("Период-граница T (баров)", fontsize=10)
ax5.set_title(f"Δ% MAPE vs стандарт T=16 (красный=хуже, зелёный=лучше)", fontsize=10)

for j, t in enumerate(TICKERS):
    for i, T in enumerate(TEST_PERIODS):
        d = mat_delta[i, j]
        ax5.text(i, j, f"{d:+.1f}", ha="center", va="center",
                 fontsize=7, color="black" if abs(d) < 12 else "white")

# Вертикальная линия: стандарт и оптимум
std_x = TEST_PERIODS.index(STANDARD_PERIOD)
opt_x = TEST_PERIODS.index(T_opt)
ax5.axvline(std_x, color="red",   lw=2.5, ls="--", alpha=0.7)
ax5.axvline(opt_x, color="green", lw=2.5, ls=":",  alpha=0.9)
plt.colorbar(im, ax=ax5, label="Δ% MAPE", shrink=0.7)
plt.tight_layout()
out3 = OUT_DIR / "44_heatmap.png"
fig3.savefig(out3, dpi=140, bbox_inches="tight")
print(f"  Рис 3: {out3}")
plt.close(fig3)


# ── Итоги ─────────────────────────────────────────────────────────────────────

print("\n" + "═" * 70)
print("ИТОГИ")
print("═" * 70)

print(f"\nЭксперимент A (fine sweep T):")
print(f"  Оптимум:   T={T_opt:>3} bar  fc_slow={1/T_opt:.5f}  MAPE={mape_opt:.5f}  Δ={delta_opt:+.2f}%")
print(f"  Стандарт:  T={STANDARD_PERIOD:>3} bar  fc_slow={1/STANDARD_PERIOD:.5f}  MAPE={mape_std:.5f}")
print(f"  H_border при T={T_opt}: {H_BORDER[T_opt]:.3f}  "
      f"({'персистент.' if H_BORDER[T_opt]>0.5 else 'анти-перс.'})")

# Найти диапазон "плато" (Δ < 1%)
plateau = [T for T in TEST_PERIODS if abs((MAPE_A[T] - mape_std) / mape_std * 100) < 1.0]
if plateau:
    print(f"  Плато (Δ<1%): T ∈ [{min(plateau)}, {max(plateau)}] баров  "
          f"(fc_slow ∈ [{1/max(plateau):.4f}, {1/min(plateau):.4f}])")

print(f"\nЭксперимент B (filter order @ T={T_opt}):")
for o in FILTER_ORDERS:
    d = (MAPE_B[o] - MAPE_B[STANDARD_ORDER]) / MAPE_B[STANDARD_ORDER] * 100
    mark = " ← стандарт" if o == STANDARD_ORDER else (" *оптимум" if o == order_opt else "")
    print(f"  order={o}: MAPE={MAPE_B[o]:.5f}  Δ={d:+.2f}%{mark}")

print(f"\nФайлы:")
for out in [out1, out2, out3]:
    print(f"  {out.name}")
print("=" * 70)
