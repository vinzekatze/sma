"""
PE → FNN: двухэтапный тест гипотезы о локальных структурных участках.

Вопрос: если разбить Δratio на участки по Permutation Entropy (низкая PE = структура),
подтвердит ли FNN наличие аттрактора на структурных участках?

Глобальный FNN показал: Δratio стохастичен (FNN никогда < 5%).
Гипотеза: это усреднение. На участках с низкой PE — FNN насыщается.

Этапы:
  1. Rolling PE(W=50, order=3) по всему ряду Δratio
  2. Сегментация: PE < q25 → "структурные", PE > q75 → "стохастические"
  3. FNN на трёх наборах точек:
       A) весь ряд (baseline, воспроизводит результат исследования 08)
       B) структурные участки (непрерывные сегменты PE < q25)
       C) стохастические участки (непрерывные сегменты PE > q75)
  4. Вывод: если FNN(B) насыщается при малом m — гипотеза подтверждена.
"""

import json, sys
from math import factorial
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KDTree

ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data/candles/SBER/1d.json"
OUT_DIR = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from sma.core.forecast.normalize import normalize

# ── параметры ────────────────────────────────────────────────────────────────
MA_WINDOW = 1000

PE_ORDER = 3       # порядок PE (число символов в паттерне)
PE_DELAY = 1       # задержка внутри PE-паттерна
PE_WIN   = 50      # скользящее окно для rolling PE

STRUCT_Q  = 0.25   # квантиль PE: ниже → "структурный"
STOCH_Q   = 0.75   # квантиль PE: выше → "стохастический"
SEG_MIN   = 40     # минимальная длина непрерывного сегмента для FNN

M_MAX  = 12        # перебираем m = 1..M_MAX
TAU    = 1
R_TOL  = 10.0      # FNN: порог критерия 1
A_TOL  = 2.0       # FNN: порог критерия 2


# ── Permutation Entropy ───────────────────────────────────────────────────────
def _build_perm_index(order: int) -> dict:
    return {p: i for i, p in enumerate(permutations(range(order)))}


def perm_entropy(x: np.ndarray, order: int = 3, delay: int = 1,
                 normalize_: bool = True) -> float:
    """Permutation Entropy для вектора x."""
    n = len(x)
    run = order * delay
    if n < run:
        return np.nan

    perm_idx = _build_perm_index(order)
    counts = np.zeros(factorial(order))
    for i in range(n - run + delay):
        window = x[i: i + run: delay][:order]
        counts[perm_idx[tuple(np.argsort(window))]] += 1

    p = counts[counts > 0]
    p /= p.sum()
    h = -np.sum(p * np.log(p))
    if normalize_:
        h /= np.log(factorial(order))
    return h


def rolling_pe(series: np.ndarray, win: int, order: int = 3,
               delay: int = 1) -> np.ndarray:
    """Rolling Permutation Entropy. Первые (win-1) значений = NaN."""
    result = np.full(len(series), np.nan)
    for i in range(win - 1, len(series)):
        result[i] = perm_entropy(series[i - win + 1: i + 1], order, delay)
    return result


# ── FNN ───────────────────────────────────────────────────────────────────────
def fnn(series: np.ndarray, m_max: int = M_MAX, tau: int = TAU,
        r_tol: float = R_TOL, a_tol: float = A_TOL) -> dict:
    """FNN fraction для m = 1..m_max."""
    sigma = np.std(series)
    results = {}
    for m in range(1, m_max + 1):
        n_m  = len(series) - (m - 1) * tau - 1
        n_m1 = len(series) - m * tau - 1
        if n_m1 < 20:
            break

        X_m = np.column_stack([series[k * tau: k * tau + n_m] for k in range(m)])
        tree = KDTree(X_m[:n_m1])
        dists, idxs = tree.query(X_m[:n_m1], k=2)

        R_m = dists[:, 1]
        j   = idxs[:, 1]

        extra_i = series[m * tau: m * tau + n_m1]
        extra_j = series[m * tau: m * tau + n_m1][j]

        R_m1  = np.sqrt(R_m**2 + (extra_i - extra_j)**2)
        crit1 = np.abs(extra_i - extra_j) / (R_m + 1e-12) > r_tol
        crit2 = R_m1 / sigma > a_tol
        is_false = crit1 | crit2

        valid = R_m > 1e-12
        results[m] = (np.sum(is_false[valid]) / np.sum(valid)
                      if np.any(valid) else np.nan)
    return results


def fnn_on_segments(series: np.ndarray, mask: np.ndarray,
                    m_max: int = M_MAX, tau: int = TAU,
                    seg_min: int = SEG_MIN) -> dict:
    """
    FNN на непрерывных сегментах, где mask=True.
    Для каждого сегмента длиной ≥ seg_min запускает FNN.
    Возвращает взвешенное среднее по длине сегмента.
    """
    # выделить непрерывные сегменты
    segments = []
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            seg = series[i:j]
            if len(seg) >= seg_min:
                segments.append(seg)
            i = j
        else:
            i += 1

    if not segments:
        print(f"  [!] нет сегментов длиной ≥ {seg_min}")
        return {}

    print(f"  Сегментов: {len(segments)}, "
          f"длины: min={min(len(s) for s in segments)} "
          f"max={max(len(s) for s in segments)} "
          f"median={int(np.median([len(s) for s in segments]))}")

    # FNN на каждом сегменте → взвешенное среднее (вес = длина сегмента)
    agg = {}
    weights = {}
    for seg in segments:
        res = fnn(seg, m_max=m_max, tau=tau, r_tol=R_TOL, a_tol=A_TOL)
        for m, val in res.items():
            if not np.isnan(val):
                agg.setdefault(m, 0.0)
                weights.setdefault(m, 0.0)
                agg[m]     += val * len(seg)
                weights[m] += len(seg)

    return {m: agg[m] / weights[m] for m in sorted(agg) if weights[m] > 0}


# ── данные ────────────────────────────────────────────────────────────────────
with open(DATA_FILE) as f:
    candles = json.load(f)

df = normalize(candles, window=MA_WINDOW)
df = df.dropna(subset=["ma"]).reset_index(drop=True)

dratio = np.diff(df["ratio"].values)
print(f"Δratio: {len(dratio)} баров  "
      f"({str(df['begin'].iloc[1])[:10]} … {str(df['begin'].iloc[-1])[:10]})")


# ── rolling PE ────────────────────────────────────────────────────────────────
print(f"\nВычисляю rolling PE (W={PE_WIN}, order={PE_ORDER})…")
pe_series = rolling_pe(dratio, win=PE_WIN, order=PE_ORDER, delay=PE_DELAY)

valid_pe = pe_series[~np.isnan(pe_series)]
q_lo = np.quantile(valid_pe, STRUCT_Q)
q_hi = np.quantile(valid_pe, STOCH_Q)

print(f"PE:  min={valid_pe.min():.4f}  q25={q_lo:.4f}  "
      f"median={np.median(valid_pe):.4f}  q75={q_hi:.4f}  max={valid_pe.max():.4f}")

# маски (на том же индексе, что dratio)
mask_struct = pe_series < q_lo
mask_stoch  = pe_series > q_hi

n_struct = int(mask_struct.sum())
n_stoch  = int(mask_stoch.sum())
print(f"Структурные (PE < q25): {n_struct} баров  "
      f"({100*n_struct/len(dratio):.1f}%)")
print(f"Стохастические (PE > q75): {n_stoch} баров  "
      f"({100*n_stoch/len(dratio):.1f}%)")


# ── FNN ───────────────────────────────────────────────────────────────────────
print("\n── FNN: весь ряд (baseline) ──")
fnn_all = fnn(dratio)

print("\n── FNN: структурные участки (PE < q25) ──")
fnn_struct = fnn_on_segments(dratio, mask_struct)

print("\n── FNN: стохастические участки (PE > q75) ──")
fnn_stoch = fnn_on_segments(dratio, mask_stoch)


# ── таблица ───────────────────────────────────────────────────────────────────
def first_below(d: dict, thr: float = 0.05):
    for m in sorted(d):
        if d.get(m, 1.0) < thr:
            return m
    return "—"

all_m = sorted(set(fnn_all) | set(fnn_struct) | set(fnn_stoch))

print("\n" + "=" * 65)
print(f"{'m':>3}  {'весь ряд':>12}  {'структурные':>12}  {'стохастич.':>12}")
print("-" * 65)
for m in all_m:
    a = fnn_all.get(m, float("nan"))
    s = fnn_struct.get(m, float("nan"))
    r = fnn_stoch.get(m, float("nan"))
    marker = " ◄" if (not np.isnan(s) and s < 0.05) else ""
    print(f"{m:>3}  {a:>11.1%}  {s:>11.1%}  {r:>11.1%}{marker}")

print("=" * 65)
print(f"FNN < 5% при m:  "
      f"весь={first_below(fnn_all)}  "
      f"структурные={first_below(fnn_struct)}  "
      f"стохастич.={first_below(fnn_stoch)}")


# ── график ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# subplot 1: rolling PE
ax_pe = fig.add_subplot(gs[0, :])
t = np.arange(len(pe_series))
ax_pe.plot(t, pe_series, color="steelblue", linewidth=0.7, label="PE")
ax_pe.axhline(q_lo, color="green",  linestyle="--", linewidth=1,
              label=f"q25 = {q_lo:.3f} (структурные)")
ax_pe.axhline(q_hi, color="tomato", linestyle="--", linewidth=1,
              label=f"q75 = {q_hi:.3f} (стохастические)")
ax_pe.fill_between(t, 0, pe_series, where=(pe_series < q_lo),
                   color="green", alpha=0.18, label="структурные")
ax_pe.fill_between(t, 0, pe_series, where=(pe_series > q_hi),
                   color="tomato", alpha=0.18, label="стохастические")
ax_pe.set_title(f"Rolling Permutation Entropy  (W={PE_WIN}, order={PE_ORDER})  —  SBER 1d Δratio")
ax_pe.set_xlabel("бар")
ax_pe.set_ylabel("PE (норм.)")
ax_pe.legend(fontsize=8, ncol=3)
ax_pe.set_xlim(0, len(pe_series))

# subplot 2-4: FNN кривые
def _fnn_bars(ax, fnn_d, color, title):
    ms   = sorted(fnn_d)
    vals = [fnn_d[m] * 100 for m in ms]
    ax.bar(ms, vals, color=color, alpha=0.75, width=0.6)
    ax.axhline(5, color="red", linestyle="--", linewidth=1.2, label="5% порог")
    p = first_below(fnn_d)
    if isinstance(p, int):
        ax.axvline(p, color="red", linestyle=":", alpha=0.7)
        ax.text(p + 0.15, 88, f"m={p}", color="red", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("m")
    ax.set_ylabel("FNN, %")
    ax.set_xticks(ms)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

_fnn_bars(fig.add_subplot(gs[1, 0]), fnn_all,
          "steelblue", "FNN — весь ряд (baseline)")
_fnn_bars(fig.add_subplot(gs[1, 1]), fnn_struct if fnn_struct else {0: 1.0},
          "seagreen",  f"FNN — структурные (PE < q{int(STRUCT_Q*100)})")

out_fig = OUT_DIR / "18_pe_fnn_local.png"
plt.suptitle("PE → FNN: структурные vs стохастические участки Δratio  (SBER 1d)",
             fontsize=12, y=1.01)
plt.savefig(out_fig, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_fig}")

# итоговый вывод
print("\n── Интерпретация ──")
p_struct = first_below(fnn_struct)
p_all    = first_below(fnn_all)
if isinstance(p_struct, int):
    print(f"✓ На структурных участках FNN насыщается при m={p_struct}.")
    print("  Гипотеза ПОДТВЕРЖДЕНА: низкая PE выявляет участки с аттрактором.")
    print("  Следующий шаг: фильтр соседей в LA — брать только точки с PE < q25.")
else:
    print("✗ FNN не насыщается даже на структурных участках.")
    if isinstance(p_all, int):
        print("  Весь ряд имеет структуру, фильтр по PE — лишний.")
    else:
        print("  Ряд стохастичен локально так же, как глобально.")
        print("  Гипотеза НЕ подтверждена. Фильтр соседей по PE — нецелесообразен.")

plt.show()
