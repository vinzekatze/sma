"""
45 — Persistence vs LWR для медленных компонент (C3–C5).

Вопрос: за счёт чего LWR показывает огромный Gain vs zero-baseline?
  — потому что zero-baseline ужасен для персистентных сигналов (C[-1] ≠ 0)?
  — или LWR действительно извлекает паттерн сверх простого продолжения уровня?

Методы сравнения для каждой медленной компоненты C3, C4, C5:
  zero        : ĉ[t+h] = 0  (мартингал для dratio)
  persistence : ĉ[t+h] = c[t]  (последнее значение компоненты)
  linext      : ĉ[t+h] = c[t] + h·slope  (линейная экстраполяция по последним k барам)
  LWR         : ĉ[t+h] = LWR(p=20, ξ=63)

Метрики:
  MAE по горизонту h=1..20 для каждого метода и компоненты
  Gain(method/zero)        = zero_MAE / method_MAE
  Gain(LWR/persistence)    = persistence_MAE / LWR_MAE  ← ключевой вопрос
  Gain(LWR/linext)         = linext_MAE / LWR_MAE

Финальный агрегат: реконструированный MAPE (как в скр.43-44) для всех 4 методов.
"""

import json
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

TICKERS   = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL  = "1d"
EPS       = 1e-10

# Стандартная конфигурация (зафиксирована скр.43-44)
STD_CUTOFFS = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
SLOW_IDX    = [3, 4, 5]
FILTER_ORDER = 4

P_LWR  = 20
XI_LWR = 63
VAL_H  = 20   # горизонты 1..20
N_ORIG = 50   # origins на тикер
LINEXT_K = 20  # точек для линейной экстраполяции


# ── нормализация + данные ─────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    log_c = np.log(close + EPS)
    n = len(log_c)
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(log_c); cty = np.cumsum(t * log_c)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a     = (cy - b * ct) / cn
    trend = a + b * t
    trend[:2] = log_c[:2]
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
    comps = []; remaining = series.copy()
    for fc in cutoffs:
        sos = butter(order, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        comps.append(remaining - low)
        remaining = low
    comps.append(remaining)
    return np.array(comps)


# ── методы прогнозирования ────────────────────────────────────────────────────

def forecast_zero(h: int) -> np.ndarray:
    return np.zeros(h)


def forecast_persistence(series: np.ndarray, h: int) -> np.ndarray:
    return np.full(h, series[-1])


def forecast_linext(series: np.ndarray, h: int, k: int = LINEXT_K) -> np.ndarray:
    """Линейная экстраполяция: OLS по последним k точкам, продолжение тренда."""
    k = min(k, len(series))
    if k < 2:
        return forecast_persistence(series, h)
    x = np.arange(k, dtype=np.float64)
    y = series[-k:]
    # OLS: y ≈ a + b·x
    xm = x.mean(); ym = y.mean()
    slope = np.sum((x - xm) * (y - ym)) / (np.sum((x - xm) ** 2) + EPS)
    last  = series[-1]
    return np.array([last + (i + 1) * slope for i in range(h)])


def forecast_lwr(series: np.ndarray, horizon: int, p: int, xi: int) -> np.ndarray:
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


# ══════════════════════════════════════════════════════════════════════════════
#  Walk-forward: per-component MAE по горизонтам
# ══════════════════════════════════════════════════════════════════════════════

METHODS = ["zero", "persistence", "linext", "LWR"]
N_COMP  = len(SLOW_IDX)   # 3 компоненты
N_METH  = len(METHODS)

def run_wf_per_component(ticker: str) -> np.ndarray:
    """
    Walk-forward для одного тикера.
    Возвращает mae[method, comp, horizon] shaped (N_METH, N_COMP, VAL_H).
    """
    ratio, dratio = load_data(ticker)
    n = len(dratio)
    COMP_full = make_fb(dratio, STD_CUTOFFS, FILTER_ORDER)

    min_o = P_LWR + XI_LWR + 10
    max_o = n - VAL_H - 2
    origins = np.arange(max(min_o, max_o - N_ORIG), max_o)

    # накопитель: sum of |error| per (method, comp, h)
    mae_sum = np.zeros((N_METH, N_COMP, VAL_H))
    cnt     = np.zeros((N_METH, N_COMP, VAL_H), dtype=int)

    for vo in origins:
        COMP_h = make_fb(dratio[:vo], STD_CUTOFFS, FILTER_ORDER)
        for ci_pos, ci in enumerate(SLOW_IDX):
            hist   = COMP_h[ci]
            actual = COMP_full[ci][vo + 1: vo + 1 + VAL_H]
            h_act  = len(actual)
            if h_act == 0:
                continue

            hats = {
                "zero":        forecast_zero(VAL_H),
                "persistence": forecast_persistence(hist, VAL_H),
                "linext":      forecast_linext(hist, VAL_H),
                "LWR":         forecast_lwr(hist, VAL_H, P_LWR, XI_LWR),
            }

            for mi, method in enumerate(METHODS):
                pred = hats[method][:h_act]
                err  = np.abs(pred - actual)
                mae_sum[mi, ci_pos, :h_act] += err
                cnt[mi, ci_pos, :h_act]     += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mae = np.where(cnt > 0, mae_sum / cnt, np.nan)
    return mae   # (N_METH, N_COMP, VAL_H)


# ──────────────────────────────────────────────────────────────────────────────
# Walk-forward: реконструированный MAPE (агрегат, как в скр.44)
# ──────────────────────────────────────────────────────────────────────────────

def run_wf_mape(ticker: str) -> dict[str, np.ndarray]:
    """
    Возвращает {method: array of val_mape per origin}.
    """
    ratio, dratio = load_data(ticker)
    n = len(dratio)

    min_o = P_LWR + XI_LWR + 10
    max_o = n - VAL_H - 2
    origins = np.arange(max(min_o, max_o - N_ORIG), max_o)

    result: dict[str, list] = {m: [] for m in METHODS}

    for vo in origins:
        COMP_h = make_fb(dratio[:vo], STD_CUTOFFS, FILTER_ORDER)

        hats_per_method: dict[str, list] = {m: [] for m in METHODS}
        for ci in SLOW_IDX:
            hist = COMP_h[ci]
            hats_per_method["zero"].append(forecast_zero(VAL_H))
            hats_per_method["persistence"].append(forecast_persistence(hist, VAL_H))
            hats_per_method["linext"].append(forecast_linext(hist, VAL_H))
            hats_per_method["LWR"].append(forecast_lwr(hist, VAL_H, P_LWR, XI_LWR))

        actual_ratio = ratio[vo + 1: vo + 1 + VAL_H]
        r0 = float(ratio[vo])
        n_act = len(actual_ratio)
        if n_act == 0:
            continue

        for method in METHODS:
            dratio_hat = np.sum(hats_per_method[method], axis=0)
            r_hat = r0 + np.cumsum(dratio_hat)
            mape  = float(np.mean(
                np.abs(r_hat[:n_act] - actual_ratio) / (np.abs(actual_ratio) + EPS)
            ))
            result[method].append(mape)

    return {m: np.array(v) for m, v in result.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("45 — Persistence vs LWR для медленных компонент C3–C5")
print("=" * 70)
print(f"  Config: 6-band std, slow_idx={SLOW_IDX}, p={P_LWR}, ξ={XI_LWR}")
print(f"  Тикеры: {len(TICKERS)}, origins: {N_ORIG}/ticker, val_h: {VAL_H}")
print()

# ── 1. Per-component MAE ──────────────────────────────────────────────────────

print("╔══ Per-component MAE walk-forward ══╗")
t0 = time.time()

# mae_all[ticker_idx, method, comp, horizon]
mae_all = []
for ticker in TICKERS:
    t1 = time.time()
    mae = run_wf_per_component(ticker)
    mae_all.append(mae)
    print(f"  {ticker}: {time.time()-t1:.1f}s")

mae_all = np.array(mae_all)  # (8, N_METH, N_COMP, VAL_H)

# Медиана по тикерам
mae_med = np.nanmedian(mae_all, axis=0)  # (N_METH, N_COMP, VAL_H)
print(f"  Total: {time.time()-t0:.1f}s\n")

# Таблица: AGG MAE при h=1, h=5, h=10, h=20 по методам и компонентам
COMP_NAMES = [f"C{SLOW_IDX[i]}" for i in range(N_COMP)]
HORIZONS_PRINT = [0, 4, 9, 19]   # h=1,5,10,20 (0-indexed)

print(f"{'':12}", end="")
for ci_pos in range(N_COMP):
    for h in HORIZONS_PRINT:
        print(f"  {COMP_NAMES[ci_pos]}@h={h+1:2d}", end="")
print()
print("-" * 80)

for mi, method in enumerate(METHODS):
    print(f"  {method:<12}", end="")
    for ci_pos in range(N_COMP):
        for h in HORIZONS_PRINT:
            print(f"  {mae_med[mi, ci_pos, h]:.2e}", end="")
    print()

print()

# Gain таблица: LWR / persistence и LWR / zero
mi_zero = METHODS.index("zero")
mi_pers = METHODS.index("persistence")
mi_lext = METHODS.index("linext")
mi_lwr  = METHODS.index("LWR")

print("  Gain(method / zero) — ratio zero_MAE / method_MAE:")
print(f"{'':14}", end="")
for ci_pos in range(N_COMP):
    for h in HORIZONS_PRINT:
        print(f"  {COMP_NAMES[ci_pos]}@h={h+1:2d}", end="")
print()
print("-" * 80)

for mi, method in enumerate(METHODS[1:], start=1):
    print(f"  {method:<14}", end="")
    for ci_pos in range(N_COMP):
        for h in HORIZONS_PRINT:
            gain = mae_med[mi_zero, ci_pos, h] / (mae_med[mi, ci_pos, h] + EPS)
            print(f"  {gain:>8.1f}", end="")
    print()

print()
print("  Gain(LWR / persistence) — насколько LWR превосходит простое продолжение:")
print(f"{'':14}", end="")
for ci_pos in range(N_COMP):
    for h in HORIZONS_PRINT:
        print(f"  {COMP_NAMES[ci_pos]}@h={h+1:2d}", end="")
print()
print("-" * 80)

print(f"  {'LWR/pers':14}", end="")
for ci_pos in range(N_COMP):
    for h in HORIZONS_PRINT:
        gain = mae_med[mi_pers, ci_pos, h] / (mae_med[mi_lwr, ci_pos, h] + EPS)
        print(f"  {gain:>8.3f}", end="")
print()

print(f"  {'LWR/linext':14}", end="")
for ci_pos in range(N_COMP):
    for h in HORIZONS_PRINT:
        gain = mae_med[mi_lext, ci_pos, h] / (mae_med[mi_lwr, ci_pos, h] + EPS)
        print(f"  {gain:>8.3f}", end="")
print()

print(f"  {'pers/zero':14}", end="")
for ci_pos in range(N_COMP):
    for h in HORIZONS_PRINT:
        gain = mae_med[mi_zero, ci_pos, h] / (mae_med[mi_pers, ci_pos, h] + EPS)
        print(f"  {gain:>8.1f}", end="")
print()


# ── 2. Реконструированный MAPE ─────────────────────────────────────────────────

print("\n╔══ Реконструированный AGG MAPE (цена) ══╗")
t0 = time.time()

mape_all: dict[str, list] = {m: [] for m in METHODS}
for ticker in TICKERS:
    t1 = time.time()
    res = run_wf_mape(ticker)
    for method in METHODS:
        mape_all[method].extend(res[method].tolist())
    print(f"  {ticker}: {time.time()-t1:.1f}s")

mape_agg = {m: float(np.nanmedian(mape_all[m])) for m in METHODS}
mape_lwr = mape_agg["LWR"]

print(f"\n  Метод        |   AGG MAPE  | Δ% vs LWR")
print("  " + "-" * 38)
for method in METHODS:
    d = (mape_agg[method] - mape_lwr) / mape_lwr * 100
    mark = " ← baseline" if method == "LWR" else ""
    print(f"  {method:<14}|  {mape_agg[method]:.5f}  | {d:>+7.2f}%{mark}")
print(f"\n  Total: {time.time()-t0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
#  ГРАФИКИ
# ══════════════════════════════════════════════════════════════════════════════

METHOD_STYLES = {
    "zero":        {"color": "gray",        "ls": ":",  "lw": 1.5, "marker": None},
    "persistence": {"color": "steelblue",   "ls": "--", "lw": 1.8, "marker": None},
    "linext":      {"color": "darkorange",  "ls": "-.", "lw": 1.8, "marker": None},
    "LWR":         {"color": "crimson",     "ls": "-",  "lw": 2.5, "marker": "o"},
}
horizons = np.arange(1, VAL_H + 1)

# ── Рис 1: MAE vs horizon по компонентам ─────────────────────────────────────

fig1, axes = plt.subplots(1, N_COMP, figsize=(15, 5), sharey=False)
fig1.suptitle(
    f"MAE vs горизонт для медленных компонент  |  p={P_LWR}, ξ={XI_LWR}  |  8 тикеров 1d",
    fontsize=12, fontweight="bold"
)

for ci_pos in range(N_COMP):
    ax = axes[ci_pos]
    for mi, method in enumerate(METHODS):
        st = METHOD_STYLES[method]
        ax.plot(horizons, mae_med[mi, ci_pos], color=st["color"], ls=st["ls"],
                lw=st["lw"], label=method,
                marker=st["marker"] if st["marker"] else None,
                ms=5 if st["marker"] else 0, markevery=2)
    ax.set_title(f"{COMP_NAMES[ci_pos]}  (медиана по 8 тикерам)", fontsize=10)
    ax.set_xlabel("Горизонт h (баров)", fontsize=9)
    ax.set_ylabel("MAE компоненты", fontsize=9)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.35)
    ax.set_xlim(1, VAL_H)

plt.tight_layout()
out1 = OUT_DIR / "45_component_mae.png"
fig1.savefig(out1, dpi=140, bbox_inches="tight")
print(f"\n  Рис 1: {out1}")
plt.close(fig1)


# ── Рис 2: Gain(LWR/persistence) и Gain(LWR/linext) vs horizon ──────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle(
    "Gain(LWR / persistence) и Gain(LWR / linext) по горизонтам",
    fontsize=12, fontweight="bold"
)

COMP_COLORS = ["tab:blue", "tab:green", "tab:orange"]

ax_lp = axes2[0]
ax_ll = axes2[1]

for ci_pos in range(N_COMP):
    c = COMP_COLORS[ci_pos]
    gain_lp = mae_med[mi_pers, ci_pos] / (mae_med[mi_lwr, ci_pos] + EPS)
    gain_ll = mae_med[mi_lext, ci_pos] / (mae_med[mi_lwr, ci_pos] + EPS)

    ax_lp.plot(horizons, gain_lp, color=c, lw=2, marker="o", ms=4, markevery=2,
               label=COMP_NAMES[ci_pos])
    ax_ll.plot(horizons, gain_ll, color=c, lw=2, marker="o", ms=4, markevery=2,
               label=COMP_NAMES[ci_pos])

for ax, title in [(ax_lp, "Gain = persistence_MAE / LWR_MAE"),
                  (ax_ll, "Gain = linext_MAE / LWR_MAE")]:
    ax.axhline(1.0, color="black", lw=1.5, ls="--", label="Gain=1 (паритет)")
    ax.set_xlabel("Горизонт h (баров)", fontsize=10)
    ax.set_ylabel("Gain (> 1 = LWR лучше)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)
    ax.set_xlim(1, VAL_H)

plt.tight_layout()
out2 = OUT_DIR / "45_gain_vs_persistence.png"
fig2.savefig(out2, dpi=140, bbox_inches="tight")
print(f"  Рис 2: {out2}")
plt.close(fig2)


# ── Рис 3: Gain(method/zero) по компонентам и горизонтам (сетка) ─────────────

fig3, axes3 = plt.subplots(N_COMP, 3, figsize=(16, 12))
fig3.suptitle(
    "Gain(method / zero): декомпозиция «откуда берётся выигрыш»\n"
    "Верхнее: pers/zero | Среднее: LWR/zero | Нижнее: LWR/pers",
    fontsize=11, fontweight="bold"
)

for ci_pos in range(N_COMP):
    gain_pz = mae_med[mi_zero, ci_pos] / (mae_med[mi_pers, ci_pos] + EPS)
    gain_lz = mae_med[mi_zero, ci_pos] / (mae_med[mi_lwr,  ci_pos] + EPS)
    gain_lp = mae_med[mi_pers, ci_pos] / (mae_med[mi_lwr,  ci_pos] + EPS)

    for col_idx, (gain, title, clr) in enumerate([
        (gain_pz, "Gain(persistence / zero)",  "steelblue"),
        (gain_lz, "Gain(LWR / zero)",           "crimson"),
        (gain_lp, "Gain(LWR / persistence)",    "darkorange"),
    ]):
        ax = axes3[ci_pos][col_idx]
        ax.fill_between(horizons, 1, gain, where=(gain >= 1),
                        alpha=0.25, color="green")
        ax.fill_between(horizons, gain, 1, where=(gain < 1),
                        alpha=0.25, color="red")
        ax.plot(horizons, gain, color=clr, lw=2.2)
        ax.axhline(1.0, color="black", lw=1, ls="--")
        ax.set_title(f"{COMP_NAMES[ci_pos]}: {title}", fontsize=8.5)
        ax.set_xlabel("h", fontsize=8)
        ax.set_ylabel("Gain", fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(1, VAL_H)

plt.tight_layout()
out3 = OUT_DIR / "45_gain_decomposition.png"
fig3.savefig(out3, dpi=140, bbox_inches="tight")
print(f"  Рис 3: {out3}")
plt.close(fig3)


# ── Рис 4: Реконструированный MAPE (бар-чарт) ─────────────────────────────────

fig4, ax4 = plt.subplots(figsize=(8, 5))
mape_vals = [mape_agg[m] for m in METHODS]
colors4   = ["gray", "steelblue", "darkorange", "crimson"]
bars = ax4.bar(METHODS, mape_vals, color=colors4, alpha=0.8, edgecolor="k", lw=0.8)

# Добавляем значения над барами
for bar, val in zip(bars, mape_vals):
    ax4.text(bar.get_x() + bar.get_width() / 2, val + max(mape_vals) * 0.005,
             f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax4.set_ylabel("Медиана AGG MAPE (цена)", fontsize=10)
ax4.set_title(
    f"Реконструированный MAPE по методу прогноза компонент\n"
    f"8 тикеров 1d, {N_ORIG} origins, val_h={VAL_H}",
    fontsize=10
)
ax4.grid(alpha=0.35, axis="y")
ax4.set_ylim(0, max(mape_vals) * 1.15)

plt.tight_layout()
out4 = OUT_DIR / "45_reconstructed_mape.png"
fig4.savefig(out4, dpi=140, bbox_inches="tight")
print(f"  Рис 4: {out4}")
plt.close(fig4)


# ── Итоги ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ИТОГИ")
print("=" * 70)

print("\n1. Gain(LWR / persistence) @ h=1:")
for ci_pos in range(N_COMP):
    g = float(mae_med[mi_pers, ci_pos, 0] / (mae_med[mi_lwr, ci_pos, 0] + EPS))
    print(f"   {COMP_NAMES[ci_pos]}: {g:.4f}  "
          f"({'LWR лучше' if g > 1.01 else 'паритет' if g > 0.99 else 'pers лучше'})")

print("\n2. Gain(persistence / zero) @ h=1:")
for ci_pos in range(N_COMP):
    g = float(mae_med[mi_zero, ci_pos, 0] / (mae_med[mi_pers, ci_pos, 0] + EPS))
    print(f"   {COMP_NAMES[ci_pos]}: {g:.1f}×")

print("\n3. Реконструированный MAPE:")
for method in METHODS:
    d = (mape_agg[method] - mape_agg["LWR"]) / mape_agg["LWR"] * 100
    print(f"   {method:<14}: {mape_agg[method]:.5f}  ({d:+.2f}% vs LWR)")

print("\nФайлы:")
for out in [out1, out2, out3, out4]:
    print(f"  {out.name}")
print("=" * 70)
