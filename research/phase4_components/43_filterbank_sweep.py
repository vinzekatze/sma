"""
43 — Sweep конфигураций filter bank: поиск оптимального числа полос.

Два эксперимента:

  Эксперимент A — варьируем число slow sub-bands (N_SLOW):
    HF-часть фиксирована: cutoffs_hf = [0.25, 0.125, fc_slow]
    Slow-часть: N_SLOW полос ниже fc_slow с полуоктавным шагом (step = √2).
    fc_slow = 0.0625 (период > 16 баров).
    N_SLOW ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12}

  Эксперимент B — варьируем fc_slow (slow boundary):
    N_SLOW = 3 (фиксировано), HF-часть адаптируется.
    fc_slow ∈ {0.25, 0.125, 0.0625, 0.03125, 0.015625}
    Показывает: включение более быстрых компонент помогает или вредит?

  Метрика: медиана MAPE (val_h=10) по 8 тикерам, walk-forward 30 origins.
  Нормализация: logtrend causal OLS (стандарт фазы 4).
  LWR: p=20, ξ=63 для всех конфигураций.
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
from scipy.stats import wilcoxon

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
FILTER_ORDER = 4

P_LWR    = 20
XI_LWR   = 3 * (P_LWR + 1)   # 63
VAL_H    = 10
N_ORIG   = 30     # origins per ticker (баланс скорость/точность)

_SQ2 = math.sqrt(2)

# Эксперимент A: sweep N_SLOW при фиксированном fc_slow
FC_SLOW_A  = 0.0625
N_SLOW_LIST = [1, 2, 3, 4, 5, 6, 8, 10, 12]
# N_SLOW = 3 соответствует текущему стандарту (C3+C4+C5 в 6-band)
STANDARD_N_SLOW = 3

# Эксперимент B: sweep fc_slow при фиксированном N_SLOW
N_SLOW_B     = 3
FC_SLOW_LIST = [0.25, 0.125, 0.0625, 0.03125, 0.015625]

# Минимальная допустимая частота для slow компонент
# (ниже не делаем полосы — мало данных для LWR)
FC_MIN = 0.002   # период ~500 баров


# ── генерация конфигурации полос ──────────────────────────────────────────────

def make_cutoffs(n_slow: int, fc_slow: float, hf_cutoffs: list = None) -> tuple[list, int]:
    """
    Строит список cutoffs для filter bank.
    HF-часть: octave cutoffs от 0.25 до fc_slow.
    Slow-часть: n_slow полос с полуоктавным шагом (√2) ниже fc_slow.

    Возвращает (cutoffs, n_slow_actual) где n_slow_actual <= n_slow
    (ограничено FC_MIN снизу).
    """
    # HF часть: ступени от 0.25 до fc_slow, октавный шаг
    if hf_cutoffs is None:
        hf_cuts = []
        fc = 0.25
        while fc > fc_slow * 1.01:   # чуть выше fc_slow
            hf_cuts.append(fc)
            fc /= 2.0
        # последний HF cutoff — сам fc_slow
        hf_cuts.append(fc_slow)
    else:
        hf_cuts = list(hf_cutoffs) + [fc_slow]

    if n_slow <= 1:
        return hf_cuts, 1

    # Slow часть: полуоктавный шаг от fc_slow вниз
    slow_cuts = []
    fc = fc_slow / _SQ2
    for _ in range(n_slow - 1):
        if fc < FC_MIN:
            break
        slow_cuts.append(fc)
        fc /= _SQ2

    actual_n = len(slow_cuts) + 1
    return hf_cuts + slow_cuts, actual_n


def period_str(fc_lo: float, fc_hi: float) -> str:
    lo_p = int(round(1.0 / fc_hi)) if fc_hi > 0 else 0
    hi_p = int(round(1.0 / fc_lo)) if fc_lo > FC_MIN / 2 else 9999
    return f"{lo_p}–{hi_p}" if hi_p < 9999 else f"{lo_p}+"


# ── нормализация ───────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    log_c = np.log(close + EPS)
    n = len(log_c)
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
    comps = []; remaining = series.copy()
    for fc in cutoffs:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        comps.append(remaining - low)
        remaining = low
    comps.append(remaining)
    return np.array(comps)


def slow_indices(cutoffs: list, fc_slow: float) -> list:
    """Индексы компонент, чья верхняя граница ≤ fc_slow."""
    # cutoffs[i] — нижняя граница компоненты i (её верхняя граница = cutoffs[i-1] или 0.5)
    # Компонента i отвечает полосе [cutoffs[i], cutoffs[i-1]] (или [cutoffs[i], 0.5] для i=0)
    # Slow: компоненты с верхней частотой ≤ fc_slow
    # Верхняя граница C_i:
    #   i=0: 0.5 (Nyquist) — never slow
    #   i=k: cutoffs[k-1]
    #   last: 0 (residual) — always slow if fc_slow > 0
    n_bands = len(cutoffs) + 1
    result = []
    for i in range(n_bands):
        if i == 0:
            upper = 0.5
        else:
            upper = cutoffs[i - 1]
        if upper <= fc_slow + EPS:
            result.append(i)
    return result


# ── LWR ───────────────────────────────────────────────────────────────────────

def lwr_forecast_series(series: np.ndarray, horizon: int) -> np.ndarray:
    X, y = build_delay_matrix(series, P_LWR)
    if len(X) < XI_LWR:
        return np.zeros(horizon)
    v = last_vector(series, p=P_LWR).copy()
    hat = np.empty(horizon)
    for h in range(horizon):
        dists  = np.linalg.norm(X - v, axis=1)
        nn_idx = np.argpartition(dists, XI_LWR)[:XI_LWR]
        h_bw   = max(float(dists[nn_idx].max()), EPS)
        w      = np.exp(-0.5 * (dists[nn_idx] / h_bw) ** 2)
        A      = np.hstack([np.ones((XI_LWR, 1)), X[nn_idx]])
        sw     = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx], rcond=None)
        hat[h] = float(c[0] + v @ c[1:])
        v = np.roll(v, -1); v[-1] = hat[h]
    return hat


# ── walk-forward для одной конфигурации ───────────────────────────────────────

def run_wf_config(cutoffs: list, fc_slow: float,
                  ticker: str, n_orig: int = N_ORIG) -> np.ndarray:
    """
    Walk-forward MAPE для заданной конфигурации на одном тикере.
    Возвращает массив MAPE (n_orig,).
    """
    ratio, dratio = load_dratio(ticker)
    n = len(dratio)
    slow_idx = slow_indices(cutoffs, fc_slow)
    if not slow_idx:
        return np.full(n_orig, np.nan)

    COMP_full = make_fb(dratio, cutoffs)

    min_orig = P_LWR + XI_LWR + 10
    max_orig = n - VAL_H - 2
    origins  = np.arange(max(min_orig, max_orig - n_orig), max_orig)

    mapes = []
    for vo in origins:
        COMP_h = make_fb(dratio[:vo], cutoffs)
        # Прогноз: сумма LWR по slow компонентам
        hats = []
        for ci in slow_idx:
            hats.append(lwr_forecast_series(COMP_h[ci], VAL_H))
        dratio_hat = np.sum(hats, axis=0)

        r0     = float(ratio[vo])
        r_hat  = r0 + np.cumsum(dratio_hat)
        actual = ratio[vo + 1: vo + 1 + VAL_H]
        n_act  = min(len(r_hat), len(actual))
        if n_act == 0:
            continue
        mape = float(np.mean(np.abs(r_hat[:n_act] - actual[:n_act]) /
                              (np.abs(actual[:n_act]) + EPS)))
        mapes.append(mape)

    return np.array(mapes) if mapes else np.full(n_orig, np.nan)


def run_wf_all_tickers(cutoffs: list, fc_slow: float,
                        n_orig: int = N_ORIG) -> dict:
    """Запуск по всем тикерам. Возвращает {ticker: mapes_array}."""
    res = {}
    for ticker in TICKERS:
        res[ticker] = run_wf_config(cutoffs, fc_slow, ticker, n_orig)
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  ЭКСПЕРИМЕНТ A: sweep N_SLOW
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("43 — Sweep конфигураций filter bank")
print("=" * 70)

print(f"\n╔══ Эксперимент A: sweep N_SLOW (fc_slow={FC_SLOW_A}) ══════════════════╗")
print(f"  N_SLOW ∈ {N_SLOW_LIST}")
print(f"  LWR: p={P_LWR}, ξ={XI_LWR}, val_h={VAL_H}")
print(f"  Тикеры: {len(TICKERS)}, origins/ticker: {N_ORIG}\n")

RES_A: dict[int, dict] = {}   # n_slow → {ticker: mapes}
META_A: dict[int, dict] = {}  # n_slow → конфигурация

t_a0 = time.time()
for n_slow in N_SLOW_LIST:
    cutoffs, actual_n = make_cutoffs(n_slow, FC_SLOW_A)
    s_idx = slow_indices(cutoffs, FC_SLOW_A)

    META_A[n_slow] = {
        "cutoffs": cutoffs,
        "actual_n_slow": actual_n,
        "n_bands_total": len(cutoffs) + 1,
        "slow_idx": s_idx,
        "fc_min": cutoffs[-1] if len(cutoffs) > len([c for c in cutoffs if c > FC_SLOW_A]) else FC_SLOW_A,
    }
    fc_min = min(c for c in cutoffs if c <= FC_SLOW_A) if any(c <= FC_SLOW_A for c in cutoffs) else FC_SLOW_A

    t1 = time.time()
    res = run_wf_all_tickers(cutoffs, FC_SLOW_A, N_ORIG)
    RES_A[n_slow] = res

    all_mapes = np.concatenate([v for v in res.values() if len(v) > 0])
    all_valid  = all_mapes[~np.isnan(all_mapes)]
    med_agg = float(np.nanmedian(all_valid)) if len(all_valid) > 0 else np.nan
    is_std = " ← стандарт" if n_slow == STANDARD_N_SLOW else ""
    print(f"  N_SLOW={n_slow:>2} | {actual_n:>2} act | {len(cutoffs)+1:>2} total bands "
          f"| slow_idx={s_idx} "
          f"| AGG MAPE={med_agg:.5f}{is_std}  ({time.time()-t1:.1f}s)")

print(f"\n  Итого Эксперимент A: {time.time()-t_a0:.1f}s")


# ══════════════════════════════════════════════════════════════════════════════
#  ЭКСПЕРИМЕНТ B: sweep fc_slow
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n╔══ Эксперимент B: sweep fc_slow (N_SLOW={N_SLOW_B}) ═════════════════════╗")
print(f"  fc_slow ∈ {FC_SLOW_LIST}\n")

RES_B: dict[float, dict] = {}
META_B: dict[float, dict] = {}

t_b0 = time.time()
for fc_slow in FC_SLOW_LIST:
    cutoffs, actual_n = make_cutoffs(N_SLOW_B, fc_slow)
    s_idx = slow_indices(cutoffs, fc_slow)

    META_B[fc_slow] = {
        "cutoffs": cutoffs,
        "actual_n_slow": actual_n,
        "n_bands_total": len(cutoffs) + 1,
        "slow_idx": s_idx,
    }
    t1 = time.time()
    res = run_wf_all_tickers(cutoffs, fc_slow, N_ORIG)
    RES_B[fc_slow] = res

    all_mapes = np.concatenate([v for v in res.values() if len(v) > 0])
    all_valid  = all_mapes[~np.isnan(all_mapes)]
    med_agg = float(np.nanmedian(all_valid)) if len(all_valid) > 0 else np.nan
    period_lo = int(round(1.0 / fc_slow))
    is_std = " ← стандарт" if fc_slow == FC_SLOW_A else ""
    print(f"  fc_slow={fc_slow:.5f} (>{period_lo:>4} bar) | slow_idx={s_idx} "
          f"| AGG MAPE={med_agg:.5f}{is_std}  ({time.time()-t1:.1f}s)")

print(f"\n  Итого Эксперимент B: {time.time()-t_b0:.1f}s")


# ── сводные таблицы ───────────────────────────────────────────────────────────

print("\n" + "═" * 70)
print("СВОДНАЯ ТАБЛИЦА — Эксперимент A: MAPE по тикерам (val_h=10)")
print("═" * 70)

def agg_mape(res_dict: dict) -> tuple[float, float]:
    all_v = np.concatenate([v for v in res_dict.values()])
    valid = all_v[~np.isnan(all_v)]
    return float(np.nanmedian(valid)), float(np.nanmean(valid))

# Заголовок
ticker_header = "  ".join(f"{t:>7}" for t in TICKERS)
print(f"{'N_SLOW':>7} | {ticker_header} | {'AGG med':>9} | Δ% vs std")
print("─" * (7 + 2 + 7*len(TICKERS) + 2*len(TICKERS) + 20))

std_med, _ = agg_mape(RES_A[STANDARD_N_SLOW])
for n_slow in N_SLOW_LIST:
    per_ticker = "  ".join(
        f"{float(np.nanmedian(RES_A[n_slow][t])):>7.5f}" for t in TICKERS
    )
    med, _ = agg_mape(RES_A[n_slow])
    delta = (med - std_med) / std_med * 100
    marker = " ←" if n_slow == STANDARD_N_SLOW else ("*" if delta < 0 else "")
    print(f"{n_slow:>7} | {per_ticker} | {med:>9.5f} | {delta:>+7.2f}%{marker}")

print("\n" + "═" * 70)
print("СВОДНАЯ ТАБЛИЦА — Эксперимент B: MAPE по тикерам")
print("═" * 70)

std_b_med, _ = agg_mape(RES_B[FC_SLOW_A])
print(f"{'fc_slow':>10} {'period':>7} | {ticker_header} | {'AGG med':>9} | Δ% vs std")
print("─" * (10 + 8 + 2 + 7*len(TICKERS) + 2*len(TICKERS) + 20))
for fc_slow in FC_SLOW_LIST:
    per_ticker = "  ".join(
        f"{float(np.nanmedian(RES_B[fc_slow][t])):>7.5f}" for t in TICKERS
    )
    med, _ = agg_mape(RES_B[fc_slow])
    delta  = (med - std_b_med) / std_b_med * 100
    marker = " ←" if fc_slow == FC_SLOW_A else ("*" if delta < 0 else "")
    period = int(round(1.0 / fc_slow))
    print(f"{fc_slow:>10.5f} {'>'+str(period)+'b':>7} | {per_ticker} | "
          f"{med:>9.5f} | {delta:>+7.2f}%{marker}")


# ══════════════════════════════════════════════════════════════════════════════
#  ГРАФИКИ
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Графики]")

# ── Рис 1: AGG MAPE vs N_SLOW (Эксперимент A) ─────────────────────────────────

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle(
    f"Sweep числа slow sub-bands  |  fc_slow={FC_SLOW_A}  |  p={P_LWR}, ξ={XI_LWR}  "
    f"|  val_h={VAL_H}  |  8 тикеров 1d",
    fontsize=11, fontweight="bold"
)

ns_arr = np.array(N_SLOW_LIST)
agg_meds_A = np.array([agg_mape(RES_A[n])[0] for n in N_SLOW_LIST])
n_bands_total = np.array([META_A[n]["n_bands_total"] for n in N_SLOW_LIST])

# Per-ticker lines
ax = axes1[0]
PALETTE = plt.cm.tab10(np.linspace(0, 1, len(TICKERS)))
for i, ticker in enumerate(TICKERS):
    meds = [float(np.nanmedian(RES_A[n][ticker])) for n in N_SLOW_LIST]
    ax.plot(ns_arr, meds, color=PALETTE[i], lw=1.4, alpha=0.7,
            marker="o", ms=3, label=ticker)

ax.plot(ns_arr, agg_meds_A, color="black", lw=2.5, marker="D", ms=6,
        label="Aggregate")
ax.axvline(STANDARD_N_SLOW, color="red", lw=1.5, ls="--", label=f"Текущий стандарт (N_SLOW={STANDARD_N_SLOW})")
ax.set_xlabel("N_SLOW (число slow sub-bands)", fontsize=10)
ax.set_ylabel("Медиана MAPE (val_h=10)", fontsize=10)
ax.set_title("Per-ticker и AGG MAPE vs N_SLOW", fontsize=10)
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.35)
ax.set_xticks(N_SLOW_LIST)

# Δ% от стандарта
ax2 = axes1[1]
deltas_A = (agg_meds_A - std_med) / std_med * 100
colors_bar = ["seagreen" if d < 0 else "tomato" for d in deltas_A]
bars = ax2.bar(ns_arr, deltas_A, color=colors_bar, edgecolor="k", lw=0.5, alpha=0.85)
ax2.axhline(0, color="black", lw=1)
ax2.axvline(STANDARD_N_SLOW, color="red", lw=1.5, ls="--",
            label=f"Стандарт N_SLOW={STANDARD_N_SLOW}")
for bar, d in zip(bars, deltas_A):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{d:+.2f}%", ha="center", va="bottom", fontsize=8)
ax2.set_xlabel("N_SLOW", fontsize=10)
ax2.set_ylabel("Δ% от стандарта (N_SLOW=3)", fontsize=10)
ax2.set_title("Изменение AGG MAPE относительно стандарта", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.35, axis="y")
ax2.set_xticks(N_SLOW_LIST)

plt.tight_layout()
out1 = OUT_DIR / "43_nslow_sweep.png"
fig1.savefig(out1, dpi=130, bbox_inches="tight")
print(f"  Рис 1: {out1}")
plt.close(fig1)


# ── Рис 2: fc_slow sweep (Эксперимент B) ──────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle(
    f"Sweep fc_slow (slow boundary)  |  N_SLOW={N_SLOW_B}  |  p={P_LWR}, ξ={XI_LWR}  "
    f"|  val_h={VAL_H}  |  8 тикеров 1d",
    fontsize=11, fontweight="bold"
)

fc_arr  = np.array(FC_SLOW_LIST)
periods = [int(round(1 / fc)) for fc in FC_SLOW_LIST]
agg_meds_B = np.array([agg_mape(RES_B[fc])[0] for fc in FC_SLOW_LIST])

ax3 = axes2[0]
for i, ticker in enumerate(TICKERS):
    meds = [float(np.nanmedian(RES_B[fc][ticker])) for fc in FC_SLOW_LIST]
    ax3.plot(periods, meds, color=PALETTE[i], lw=1.4, alpha=0.7,
             marker="o", ms=3, label=ticker)
ax3.plot(periods, agg_meds_B, color="black", lw=2.5, marker="D", ms=6,
         label="Aggregate")
ax3.axvline(int(round(1 / FC_SLOW_A)), color="red", lw=1.5, ls="--",
            label=f"Стандарт (fc_slow={FC_SLOW_A})")
ax3.set_xlabel("Slow boundary: период (bar), компоненты с большим периодом используются", fontsize=9)
ax3.set_ylabel("Медиана MAPE (val_h=10)", fontsize=10)
ax3.set_title("Per-ticker MAPE vs slow boundary", fontsize=10)
ax3.set_xscale("log")
ax3.legend(fontsize=8, ncol=2)
ax3.grid(alpha=0.35, which="both")
ax3.set_xticks(periods); ax3.set_xticklabels([f">{p}" for p in periods], fontsize=8)

ax4 = axes2[1]
deltas_B = (agg_meds_B - std_b_med) / std_b_med * 100
colors_b2 = ["seagreen" if d < 0 else "tomato" for d in deltas_B]
bars2 = ax4.bar(range(len(FC_SLOW_LIST)), deltas_B, color=colors_b2,
                edgecolor="k", lw=0.5, alpha=0.85)
ax4.axhline(0, color="black", lw=1)
for bar, d, p in zip(bars2, deltas_B, periods):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{d:+.2f}%", ha="center", va="bottom", fontsize=8)
ax4.set_xlabel("fc_slow (slow boundary)", fontsize=10)
ax4.set_ylabel("Δ% от стандарта", fontsize=10)
ax4.set_title("Изменение AGG MAPE относительно стандарта", fontsize=10)
ax4.set_xticks(range(len(FC_SLOW_LIST)))
ax4.set_xticklabels([f">{p}b\n(fc={fc:.4f})" for fc, p in zip(FC_SLOW_LIST, periods)], fontsize=8)
ax4.grid(alpha=0.35, axis="y")

plt.tight_layout()
out2 = OUT_DIR / "43_fcslow_sweep.png"
fig2.savefig(out2, dpi=130, bbox_inches="tight")
print(f"  Рис 2: {out2}")
plt.close(fig2)


# ── Рис 3: Per-ticker heatmap (Эксперимент A) ─────────────────────────────────

fig3, ax5 = plt.subplots(figsize=(12, 5))
mat = np.array([[float(np.nanmedian(RES_A[n][t])) for t in TICKERS]
                for n in N_SLOW_LIST])   # shape (N_SLOW, N_TICKERS)

# Нормируем по строке: Δ% от стандарта (N_SLOW=3)
std_row_idx = N_SLOW_LIST.index(STANDARD_N_SLOW)
mat_delta = (mat - mat[std_row_idx, :]) / (mat[std_row_idx, :] + EPS) * 100

im = ax5.imshow(mat_delta, aspect="auto", cmap="RdYlGn_r", vmin=-15, vmax=15)
ax5.set_yticks(range(len(N_SLOW_LIST))); ax5.set_yticklabels([f"N_SLOW={n}" for n in N_SLOW_LIST])
ax5.set_xticks(range(len(TICKERS))); ax5.set_xticklabels(TICKERS)
ax5.set_title(f"Δ% MAPE от стандарта N_SLOW=3 (красный = хуже, зелёный = лучше)",
              fontsize=11)
for i, n in enumerate(N_SLOW_LIST):
    for j, t in enumerate(TICKERS):
        d = mat_delta[i, j]
        ax5.text(j, i, f"{d:+.1f}", ha="center", va="center", fontsize=8,
                 color="black" if abs(d) < 8 else "white")
plt.colorbar(im, ax=ax5, label="Δ% MAPE", shrink=0.8)

# Отмечаем стандартную строку
ax5.axhline(std_row_idx, color="red", lw=2, ls="--")
ax5.text(len(TICKERS) - 0.5, std_row_idx, " стандарт", color="red",
         fontsize=9, va="center")

plt.tight_layout()
out3 = OUT_DIR / "43_heatmap_nslow.png"
fig3.savefig(out3, dpi=130, bbox_inches="tight")
print(f"  Рис 3: {out3}")
plt.close(fig3)


# ── Рис 4: Комбинированный сводный ───────────────────────────────────────────

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle("Итоговый анализ конфигураций filter bank  |  AGG по 8 тикерам",
              fontsize=11, fontweight="bold")

# Лучшая конфигурация
best_n = N_SLOW_LIST[int(np.argmin(agg_meds_A))]
best_fc = FC_SLOW_LIST[int(np.argmin(agg_meds_B))]

ax6 = axes4[0]
ax6.plot(ns_arr, agg_meds_A * 100, "o-", color="steelblue", lw=2, ms=6,
         label="AGG MAPE (%)")
ax6.axvline(STANDARD_N_SLOW, color="red", lw=1.5, ls="--",
            label=f"Стандарт N_SLOW={STANDARD_N_SLOW}")
ax6.axvline(best_n, color="green", lw=1.5, ls=":",
            label=f"Оптимум N_SLOW={best_n} ({(agg_meds_A[N_SLOW_LIST.index(best_n)]-std_med)/std_med*100:+.2f}%)")
ax6.fill_between(ns_arr, agg_meds_A * 100,
                 float(np.min(agg_meds_A)) * 100, alpha=0.1, color="steelblue")
ax6.set_xlabel("N_SLOW", fontsize=10)
ax6.set_ylabel("Медиана AGG MAPE (%)", fontsize=10)
ax6.set_title("Эксп. A: MAPE vs число slow полос", fontsize=10)
ax6.legend(fontsize=9); ax6.grid(alpha=0.35)
ax6.set_xticks(N_SLOW_LIST)

ax7 = axes4[1]
ax7.plot(periods, agg_meds_B * 100, "s-", color="darkorange", lw=2, ms=6,
         label="AGG MAPE (%)")
ax7.axvline(int(round(1 / FC_SLOW_A)), color="red", lw=1.5, ls="--",
            label=f"Стандарт (>{int(round(1/FC_SLOW_A))} bar)")
best_period = int(round(1 / best_fc))
ax7.axvline(best_period, color="green", lw=1.5, ls=":",
            label=f"Оптимум >{best_period} bar ({(agg_meds_B[FC_SLOW_LIST.index(best_fc)]-std_b_med)/std_b_med*100:+.2f}%)")
ax7.set_xlabel("Нижняя граница slow компонент (баров)", fontsize=10)
ax7.set_ylabel("Медиана AGG MAPE (%)", fontsize=10)
ax7.set_title("Эксп. B: MAPE vs slow boundary", fontsize=10)
ax7.set_xscale("log")
ax7.set_xticks(periods); ax7.set_xticklabels([f">{p}" for p in periods], fontsize=9)
ax7.legend(fontsize=9); ax7.grid(alpha=0.35, which="both")

plt.tight_layout()
out4 = OUT_DIR / "43_summary.png"
fig4.savefig(out4, dpi=130, bbox_inches="tight")
print(f"  Рис 4: {out4}")
plt.close(fig4)


# ── Итоговые выводы ────────────────────────────────────────────────────────────

print("\n" + "═" * 70)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("═" * 70)

best_n_med = agg_meds_A[N_SLOW_LIST.index(best_n)]
delta_best_A = (best_n_med - std_med) / std_med * 100
print(f"\nЭксперимент A (N_SLOW sweep):")
print(f"  Стандарт  N_SLOW={STANDARD_N_SLOW}: AGG MAPE={std_med:.5f}")
print(f"  Оптимум   N_SLOW={best_n}:  AGG MAPE={best_n_med:.5f}  Δ={delta_best_A:+.2f}%")
all_deltas_A = [(n, (agg_meds_A[i] - std_med) / std_med * 100) for i, n in enumerate(N_SLOW_LIST)]
better = [(n, d) for n, d in all_deltas_A if d < 0]
if better:
    print(f"  Конфигурации лучше стандарта: {[(n, f'{d:+.2f}%') for n, d in better]}")
else:
    print(f"  Стандарт N_SLOW=3 оптимален или на уровне лучших")

best_fc_med = agg_meds_B[FC_SLOW_LIST.index(best_fc)]
delta_best_B = (best_fc_med - std_b_med) / std_b_med * 100
print(f"\nЭксперимент B (fc_slow sweep):")
print(f"  Стандарт  fc_slow={FC_SLOW_A}  (>{int(round(1/FC_SLOW_A))} bar): AGG MAPE={std_b_med:.5f}")
print(f"  Оптимум   fc_slow={best_fc:.5f} (>{best_period} bar): AGG MAPE={best_fc_med:.5f}  Δ={delta_best_B:+.2f}%")

print(f"\nФайлы:")
for out in [out1, out2, out3, out4]:
    print(f"  {out.name}")
print("=" * 70)
