"""
D₂ — корреляционная размерность (Grassberger–Procaccia, 1983).

C(r) = (2 / N(N-1)) * #{пар (i,j): dist(x_i, x_j) < r}

D₂ = d log C(r) / d log r  в области масштабирования (scaling region)

Признаки:
  D₂ конечная и нецелая → детерминированный аттрактор → LA применим
  D₂ растёт с m         → скорее случайный процесс

Считаем для dratio, ratio и raw close при m=1..8.
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sma.core.forecast.normalize import normalize
from sma.core.forecast.embedding import build_delay_matrix

DATA_FILE = ROOT / "data/candles/SBER/1d.json"

MA_WINDOW = 1000
M_MAX     = 8      # перебираем m=1..M_MAX
N_R       = 30     # точек по оси r (логарифмически)
N_SAMPLE  = 1500   # берём последние N_SAMPLE баров (быстрее, C(r) = O(N²))


# ── вычисление C(r) ──────────────────────────────────────────────────────────

def correlation_integral(series: np.ndarray, m: int, tau: int = 1,
                         n_r: int = N_R) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычислить C(r) для ряда series, размерности вложения m.

    Возвращает (r_vals, C_vals) для логарифмически равномерной сетки r.
    """
    try:
        X, _ = build_delay_matrix(series, m, tau)
    except ValueError:
        return np.array([]), np.array([])

    N = len(X)
    if N < 20:
        return np.array([]), np.array([])

    # попарные расстояния (нижний треугольник)
    # для N=1500 это 1500*1499/2 ≈ 1.1M пар — занимает ~8MB при float32
    # используем chunked подход для экономии памяти
    dists = []
    chunk = 200
    for i in range(0, N, chunk):
        Xi = X[i: i + chunk]
        d  = np.sqrt(((Xi[:, None, :] - X[None, :, :]) ** 2).sum(axis=-1))
        # берём только нижний треугольник относительно глобального i
        for local_row in range(len(Xi)):
            global_row = i + local_row
            dists.append(d[local_row, :global_row])

    if not dists:
        return np.array([]), np.array([])

    d_flat = np.concatenate(dists)
    d_flat = d_flat[d_flat > 0]
    if len(d_flat) == 0:
        return np.array([]), np.array([])

    r_min = np.percentile(d_flat, 1)
    r_max = np.percentile(d_flat, 60)   # не берём слишком большие r
    if r_min >= r_max:
        return np.array([]), np.array([])

    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    C_vals = np.array([np.mean(d_flat < r) for r in r_vals])

    return r_vals, C_vals


def estimate_d2(r_vals: np.ndarray, C_vals: np.ndarray,
                pct_lo: float = 20, pct_hi: float = 70) -> float:
    """
    Оценить наклон log C(r) / log r в scaling region.

    Scaling region: [pct_lo, pct_hi] процентили по r.
    Использует линейную регрессию в лог-пространстве.
    """
    valid = (C_vals > 0) & (r_vals > 0)
    if valid.sum() < 4:
        return np.nan

    log_r = np.log10(r_vals[valid])
    log_C = np.log10(C_vals[valid])

    lo = np.percentile(log_r, pct_lo)
    hi = np.percentile(log_r, pct_hi)
    mask = (log_r >= lo) & (log_r <= hi)
    if mask.sum() < 3:
        return np.nan

    slope, _ = np.polyfit(log_r[mask], log_C[mask], 1)
    return float(slope)


# ── данные ───────────────────────────────────────────────────────────────────
with open(DATA_FILE) as f:
    candles = json.load(f)

df     = normalize(candles, window=MA_WINDOW)
df     = df.dropna(subset=["ma"]).reset_index(drop=True)
ratio  = df["ratio"].values
dratio = np.diff(ratio)
raw    = df["close"].values

# берём последние N_SAMPLE баров
raw_seg    = raw[-N_SAMPLE:]
ratio_seg  = ratio[-N_SAMPLE:]
dratio_seg = dratio[-N_SAMPLE:]

print(f"Используем последние {N_SAMPLE} баров")
print(f"Перебор m=1..{M_MAX}, n_r={N_R} точек r")

# ── расчёт D₂(m) ─────────────────────────────────────────────────────────────
series_set = [
    ("raw close",    raw_seg,    "steelblue"),
    ("ratio",        ratio_seg,  "seagreen"),
    ("dratio (LA)",  dratio_seg, "darkorange"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Корреляционная размерность D₂ — SBER 1d  (N={N_SAMPLE} баров)", fontsize=12)

d2_results: dict[str, list[float]] = {}

for ax, (name, series, color) in zip(axes, series_set):
    d2_by_m = []
    print(f"\n{name}:")

    for m in range(1, M_MAX + 1):
        r_vals, C_vals = correlation_integral(series, m)
        if len(r_vals) == 0:
            d2_by_m.append(np.nan)
            print(f"  m={m}: нет данных")
            continue
        d2 = estimate_d2(r_vals, C_vals)
        d2_by_m.append(d2)
        print(f"  m={m}: D₂ = {d2:.3f}" if not np.isnan(d2) else f"  m={m}: D₂ = nan")

    d2_results[name] = d2_by_m
    ms = list(range(1, M_MAX + 1))

    ax.plot(ms, d2_by_m, "o-", color=color, lw=2, ms=6, label=name)
    ax.plot(ms, ms, "k--", lw=0.8, alpha=0.4, label="D₂=m (стохаст.)")
    ax.set_xlabel("m (размерность вложения)")
    ax.set_ylabel("D₂")
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, M_MAX + 0.5)
    ax.set_ylim(bottom=0)

    # отметить насыщение если D₂ перестаёт расти
    valid_d2 = [(i, v) for i, v in enumerate(d2_by_m) if not np.isnan(v)]
    if len(valid_d2) >= 3:
        # ищем точку где прирост D₂ падает ниже 0.2 (насыщение)
        for i in range(1, len(valid_d2)):
            delta = valid_d2[i][1] - valid_d2[i-1][1]
            if delta < 0.2 and valid_d2[i][1] > 0.3:
                sat_m = valid_d2[i][0] + 1   # +1 потому что m начинается с 1
                ax.axvline(sat_m, color=color, ls=":", alpha=0.7)
                ax.text(sat_m + 0.1, ax.get_ylim()[1] * 0.85,
                        f"насыщ.\nm={sat_m}", color=color, fontsize=8)
                break

plt.tight_layout()
out = ROOT / "research/figures/12_corr_dim.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=140)
print(f"\nГрафик D₂: {out}")

# ── сводная таблица ───────────────────────────────────────────────────────────
print(f"\n{'m':>3}", end="")
for name in d2_results:
    print(f"  {name[:12]:>12}", end="")
print()
print("-" * (5 + 14 * len(d2_results)))
for i, m in enumerate(range(1, M_MAX + 1)):
    print(f"{m:>3}", end="")
    for name in d2_results:
        v = d2_results[name][i]
        print(f"  {v:>12.3f}" if not np.isnan(v) else f"  {'—':>12}", end="")
    print()

# ── дополнительный график: log C(r) vs log r для m=4 ─────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle("log C(r) vs log r  при m=4 — scaling region", fontsize=11)

m_show = 4
for ax, (name, series, color) in zip(axes2, series_set):
    r_vals, C_vals = correlation_integral(series, m_show, n_r=50)
    if len(r_vals) == 0:
        ax.set_title(f"{name}\n(нет данных)")
        continue

    valid = C_vals > 0
    log_r = np.log10(r_vals[valid])
    log_C = np.log10(C_vals[valid])

    ax.plot(log_r, log_C, "o-", color=color, ms=4, lw=1.2)

    # scaling region
    lo = np.percentile(log_r, 20)
    hi = np.percentile(log_r, 70)
    mask = (log_r >= lo) & (log_r <= hi)
    if mask.sum() >= 3:
        coeffs = np.polyfit(log_r[mask], log_C[mask], 1)
        fit_x  = log_r[mask]
        ax.plot(fit_x, np.polyval(coeffs, fit_x), "r--", lw=2,
                label=f"D₂={coeffs[0]:.2f}")
        ax.axvspan(lo, hi, alpha=0.10, color="red", label="scaling region")

    ax.set_xlabel("log₁₀ r")
    ax.set_ylabel("log₁₀ C(r)")
    ax.set_title(f"{name}  m={m_show}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
out2 = ROOT / "research/figures/12_corr_dim_loglog.png"
plt.savefig(out2, dpi=140)
print(f"График log-log: {out2}")
plt.show()
