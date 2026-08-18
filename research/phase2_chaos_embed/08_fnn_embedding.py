"""
FNN (False Nearest Neighbors) — определение оптимальной размерности вложения p.

Kennel et al., 1992: точка является "ложным" соседом, если при увеличении
размерности m→m+1 расстояние до ближайшего соседа резко возрастает (сосед
оказался близким лишь из-за проекции, а не реального соседства на аттракторе).

Критерии (оба должны выполняться):
  1. |x[i + m] - x[j + m]| / R_m > R_tol   (расстояние "выстреливает")
  2. R_{m+1} / σ > A_tol                    (сосед далёк относительно аттрактора)

Исследуем два ряда:
  - raw:   котировки close (SBER 1d)
  - dratio: Δ(close / SMA(1000)) — ряд, на котором работает LA в приложении
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

# ---------------------------------------------------------------------------
# пути
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_FILE = ROOT / "data/candles/SBER/1d.json"

sys.path.insert(0, str(ROOT))
from sma.core.forecast.normalize import normalize

# ---------------------------------------------------------------------------
# параметры
# ---------------------------------------------------------------------------
MA_WINDOW = 1000   # то же, что в приложении
TAU       = 1      # задержка (как в LA)
M_MAX     = 12     # перебираем m = 1..M_MAX
R_TOL     = 10.0   # порог критерия 1 (стандарт Kennel)
A_TOL     = 2.0    # порог критерия 2 (относительно σ)


# ---------------------------------------------------------------------------
# FNN
# ---------------------------------------------------------------------------
def fnn(series: np.ndarray, m_max: int = M_MAX, tau: int = TAU,
        r_tol: float = R_TOL, a_tol: float = A_TOL) -> dict:
    """
    Compute FNN fraction for m = 1..m_max.

    Returns dict: {m: fnn_fraction}
    """
    sigma = np.std(series)
    results = {}

    for m in range(1, m_max + 1):
        # число точек при текущем m и m+1
        # при m+1 нужен ещё один отсчёт: последний индекс = i + m*tau
        n_m   = len(series) - (m - 1) * tau - 1        # достаточно для m
        n_m1  = len(series) - m * tau - 1               # достаточно для m+1

        if n_m1 < 10:
            break

        # матрица задержек размерности m (строки = точки)
        X_m = np.column_stack([series[k * tau: k * tau + n_m] for k in range(m)])

        # KDTree: ищем 2 соседа (1-й — сама точка, 2-й — ближайший чужой)
        tree   = KDTree(X_m[:n_m1])
        dists, idxs = tree.query(X_m[:n_m1], k=2)

        R_m = dists[:, 1]           # расстояние до ближайшего соседа в m-мерном пространстве
        j   = idxs[:, 1]            # индексы соседей

        # компонента (m+1)-го измерения
        extra_i = series[m * tau: m * tau + n_m1]
        extra_j = series[m * tau: m * tau + n_m1][j]

        R_m1 = np.sqrt(R_m**2 + (extra_i - extra_j)**2)

        # критерий 1: расстояние сильно возросло
        crit1 = np.abs(extra_i - extra_j) / (R_m + 1e-12) > r_tol

        # критерий 2: сосед в m+1 далёк от аттрактора
        crit2 = R_m1 / sigma > a_tol

        # точка — ложный сосед, если выполняется хотя бы одно условие
        is_false = crit1 | crit2

        # нулевые расстояния (совпадающие точки) исключаем
        valid = R_m > 1e-12
        frac  = np.sum(is_false[valid]) / np.sum(valid) if np.any(valid) else np.nan

        results[m] = frac

    return results


# ---------------------------------------------------------------------------
# данные
# ---------------------------------------------------------------------------
with open(DATA_FILE) as f:
    candles = json.load(f)

print(f"Loaded {len(candles)} candles  "
      f"({candles[0]['begin'][:10]} … {candles[-1]['begin'][:10]})")

df = normalize(candles, window=MA_WINDOW)
df = df.dropna(subset=["ma"]).reset_index(drop=True)

raw    = df["close"].values
ratio  = df["ratio"].values
dratio = np.diff(ratio)

print(f"After SMA warm-up: {len(df)} bars")
print(f"  raw    mean={raw.mean():.2f}  σ={raw.std():.2f}")
print(f"  ratio  mean={ratio.mean():.4f}  σ={ratio.std():.4f}")
print(f"  dratio mean={dratio.mean():.6f}  σ={dratio.std():.6f}")


# ---------------------------------------------------------------------------
# расчёт FNN
# ---------------------------------------------------------------------------
print("\nComputing FNN…")

fnn_raw    = fnn(raw,    m_max=M_MAX, tau=TAU)
fnn_ratio  = fnn(ratio,  m_max=M_MAX, tau=TAU)
fnn_dratio = fnn(dratio, m_max=M_MAX, tau=TAU)

# ---------------------------------------------------------------------------
# вывод таблицы
# ---------------------------------------------------------------------------
header = f"{'m':>3}  {'raw (close)':>12}  {'ratio':>12}  {'dratio':>12}"
print()
print(header)
print("-" * len(header))
for m in sorted(set(fnn_raw) | set(fnn_ratio) | set(fnn_dratio)):
    r  = fnn_raw.get(m, float("nan"))
    rt = fnn_ratio.get(m, float("nan"))
    d  = fnn_dratio.get(m, float("nan"))
    flag = ""
    if d < 0.05:
        flag = " ← оптимум"
    print(f"{m:>3}  {r:>11.1%}  {rt:>11.1%}  {d:>11.1%}{flag}")


# ---------------------------------------------------------------------------
# рекомендации
# ---------------------------------------------------------------------------
def first_below(fnn_dict: dict, threshold: float = 0.05) -> int | None:
    for m in sorted(fnn_dict):
        if fnn_dict[m] < threshold:
            return m
    return None

p_raw    = first_below(fnn_raw)
p_ratio  = first_below(fnn_ratio)
p_dratio = first_below(fnn_dratio)

print()
print(f"Рекомендуемое p (FNN < 5%):")
print(f"  raw close : p = {p_raw}")
print(f"  ratio     : p = {p_ratio}")
print(f"  dratio    : p = {p_dratio}  ← ряд LA")


# ---------------------------------------------------------------------------
# график
# ---------------------------------------------------------------------------
ms_raw    = sorted(fnn_raw)
ms_ratio  = sorted(fnn_ratio)
ms_dratio = sorted(fnn_dratio)

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
fig.suptitle("FNN — SBER 1d  (τ=1)", fontsize=13)

datasets = [
    (axes[0], fnn_raw,    ms_raw,    "raw close",   "steelblue"),
    (axes[1], fnn_ratio,  ms_ratio,  "ratio (c/MA)", "darkorange"),
    (axes[2], fnn_dratio, ms_dratio, "Δratio (LA ряд)", "seagreen"),
]

for ax, fnn_d, ms, title, color in datasets:
    vals = [fnn_d[m] * 100 for m in ms]
    ax.bar(ms, vals, color=color, alpha=0.75, width=0.6)
    ax.axhline(5, color="red", linestyle="--", linewidth=1, label="5% порог")
    ax.set_xlabel("m (размерность вложения)")
    ax.set_ylabel("FNN, %")
    ax.set_title(title)
    ax.set_xticks(ms)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)

    # отметить первое m < 5%
    p_opt = first_below(fnn_d)
    if p_opt:
        ax.axvline(p_opt, color="red", linestyle=":", alpha=0.6)
        ax.text(p_opt + 0.1, 92, f"p={p_opt}", color="red", fontsize=9)

plt.tight_layout()
out = ROOT / "research/figures/08_fnn_embedding.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=140)
print(f"\nГрафик сохранён: {out}")
plt.show()
