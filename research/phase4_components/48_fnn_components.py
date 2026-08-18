"""
48 — FNN (False Nearest Neighbors) для компонент filter bank C0–C5.

Метод FNN (Kennel et al. 1992) определяет минимальную размерность вложения p,
при которой пространство реконструкции «достаточно развёрнуто» и соседи
перестают быть «ложными».

Алгоритм:
  Для каждого d от 1 до d_max:
    1. Строим матрицу задержек X(d) размером (N-d) × d с лагом tau=1
    2. Для каждой точки находим ближайшего соседа NN(t) в d-мерном пространстве
    3. Сосед «ложный», если:
         |x[t+d] − x_NN[t+d]| / ||X(t) − X_NN(t)||  >  R_tol  (= 10)
       ИЛИ  ||X(t) − X_NN(t)|| < A_tol * σ(component)  (= 2)
    4. FNN_rate[d] = доля ложных соседей

Интерпретация:
  FNN → 0  при d = d_min  ⟹  компонента имеет детерминированный аттрактор
             размерности ≈ d_min
  FNN не опускается  ⟹  стохастический сигнал, аттрактора нет

Параметры: tau=1, R_tol=10, A_tol=2, Theiler window W=5.
"""

import json, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.spatial import KDTree

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL     = "1d"
EPS          = 1e-10
STD_CUTOFFS  = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
FILTER_ORDER = 4

# FNN параметры
TAU    = 1      # шаг задержки
D_MAX  = 30     # максимальная размерность
R_TOL  = 10.0   # порог «ложности» по расстоянию
A_TOL  = 2.0    # порог малого расстояния (шум)
THEILER = 5     # окно исключения временных соседей

COMP_LABELS = {
    0: "C0  2–4 bar  63.4%",
    1: "C1  4–8 bar  19.7%",
    2: "C2  8–16 bar  9.7%",
    3: "C3  16–52 bar  4.8%",
    4: "C4  52–103 bar  2.1%",
    5: "C5  103+ bar  0.8%",
}
COMP_COLORS = ["tab:red", "tab:orange", "tab:olive",
               "tab:blue", "tab:green", "tab:purple"]


# ── нормализация + данные ─────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    log_c = np.log(close + EPS); n = len(log_c)
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n+1, dtype=np.float64)
    ct=np.cumsum(t); ct2=np.cumsum(t**2); cy=np.cumsum(log_c); cty=np.cumsum(t*log_c)
    denom = cn*ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom>0, (cn*cty-ct*cy)/denom, 0.0)
    a=(cy-b*ct)/cn; trend=a+b*t; trend[:2]=log_c[:2]
    return np.exp(trend)


_CACHE: dict = {}

def load_data(ticker: str) -> tuple[np.ndarray, np.ndarray]:
    if ticker not in _CACHE:
        with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f: c=json.load(f)
        close = np.array([x["close"] for x in c], dtype=np.float64)
        ratio = close / logtrend_causal(close)
        _CACHE[ticker] = (ratio, np.diff(ratio))
    return _CACHE[ticker]


def make_fb(series: np.ndarray, cutoffs: list, order: int = 4) -> np.ndarray:
    comps=[]; rem=series.copy()
    for fc in cutoffs:
        sos=butter(order,fc,btype="low",output="sos"); low=sosfilt(sos,rem)
        comps.append(rem-low); rem=low
    comps.append(rem); return np.array(comps)


# ── FNN ───────────────────────────────────────────────────────────────────────

def fnn_rate(series: np.ndarray, d: int, tau: int = TAU,
             r_tol: float = R_TOL, a_tol: float = A_TOL,
             theiler: int = THEILER) -> float:
    """
    Возвращает долю ложных соседей при размерности d.
    """
    n = len(series)
    # Длина вложения: span = (d-1)*tau + 1, нужен ещё +1 шаг для проверки
    span = (d - 1) * tau + 1
    if span + 1 >= n:
        return np.nan

    # Матрица d-мерного вложения
    offsets = np.arange(d) * tau
    N = n - span - 1            # число точек, у которых есть следующий шаг
    X = np.stack([series[off: off + N] for off in offsets], axis=1)   # (N, d)
    x_next = series[span: span + N]   # x[t + d*tau] для каждой точки

    sigma = float(np.std(series))

    # KD-tree с Theiler exclusion через ручной перебор соседей
    tree = KDTree(X)
    # Запрашиваем k+theiler*2+1 соседей, чтобы хватило после исключения
    k_query = min(THEILER * 2 + 2, N - 1)
    dists_all, idx_all = tree.query(X, k=k_query + 1, workers=-1)

    false_count = 0
    valid_count = 0

    for i in range(N):
        # Ищем ближайшего соседа, исключая Theiler-окно
        nn_idx = None
        nn_dist = np.inf
        for j_pos in range(1, len(idx_all[i])):
            j = idx_all[i][j_pos]
            if abs(int(j) - i) > theiler:
                nn_idx = j
                nn_dist = dists_all[i][j_pos]
                break
        if nn_idx is None:
            continue

        valid_count += 1
        if nn_dist < a_tol * sigma:
            false_count += 1
            continue

        # Критерий Кеннела
        delta_next = abs(x_next[i] - x_next[nn_idx])
        if nn_dist < EPS:
            continue
        ratio = delta_next / nn_dist
        if ratio > r_tol:
            false_count += 1

    return false_count / valid_count if valid_count > 0 else np.nan


def compute_fnn_curve(series: np.ndarray, d_max: int = D_MAX) -> np.ndarray:
    """FNN rate для d=1..d_max."""
    return np.array([fnn_rate(series, d) for d in range(1, d_max + 1)])


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("48 — FNN для компонент filter bank C0–C5")
print("=" * 70)
print(f"  τ={TAU}, R_tol={R_TOL}, A_tol={A_TOL}, Theiler={THEILER}, d_max={D_MAX}")
print(f"  Тикеры: {', '.join(TICKERS)}\n")

n_comp = len(STD_CUTOFFS) + 1  # 6

# fnn_results[ticker][comp_idx] = array (D_MAX,)
fnn_results: dict[str, list] = {}

t0_total = time.time()
for ticker in TICKERS:
    ratio, dratio = load_data(ticker)
    COMP = make_fb(dratio, STD_CUTOFFS, FILTER_ORDER)
    fnn_results[ticker] = []
    t0t = time.time()
    print(f"  {ticker}:", end="", flush=True)
    for ci in range(n_comp):
        curve = compute_fnn_curve(COMP[ci], D_MAX)
        fnn_results[ticker].append(curve)
        print(f" C{ci}✓", end="", flush=True)
    print(f"  ({time.time()-t0t:.0f}s)")

print(f"\n  Всего: {time.time()-t0_total:.0f}s")

# Медиана по тикерам
fnn_med = np.array([
    np.nanmedian([fnn_results[t][ci] for t in TICKERS], axis=0)
    for ci in range(n_comp)
])  # (6, D_MAX)

# Таблица: FNN при ключевых d
print("\n  FNN rate (медиана по 8 тикерам):")
print(f"  {'Comp':24} |", "  ".join(f"d={d:>2}" for d in [1,2,3,4,5,6,8,10,12,15,20]))
print("  " + "-" * 100)
for ci in range(n_comp):
    label = COMP_LABELS[ci]
    vals  = "  ".join(f"{fnn_med[ci][d-1]:>5.2f}" for d in [1,2,3,4,5,6,8,10,12,15,20])
    print(f"  {label:24} | {vals}")

# Порог 10%: первый d где FNN < 10%
print("\n  Первый d, где FNN < 10% (или < 5%):")
print(f"  {'Comp':24} | d(FNN<10%) | d(FNN<5%)")
print("  " + "-" * 48)
for ci in range(n_comp):
    curve = fnn_med[ci]
    d10 = next((d+1 for d, v in enumerate(curve) if not np.isnan(v) and v < 0.10), ">"+str(D_MAX))
    d5  = next((d+1 for d, v in enumerate(curve) if not np.isnan(v) and v < 0.05), ">"+str(D_MAX))
    print(f"  {COMP_LABELS[ci]:24} | {str(d10):>10} | {str(d5):>9}")


# ══════════════════════════════════════════════════════════════════════════════
#  ГРАФИКИ
# ══════════════════════════════════════════════════════════════════════════════

dims = np.arange(1, D_MAX + 1)

# ── Рис 1: все компоненты на одном графике ────────────────────────────────────

fig1, ax1 = plt.subplots(figsize=(13, 6))
fig1.suptitle(
    f"FNN (False Nearest Neighbors) — компоненты C0–C5\n"
    f"6-band filter bank, logtrend, 8 тикеров 1d  |  τ={TAU}, R_tol={R_TOL}",
    fontsize=11, fontweight="bold"
)

for ci in range(n_comp):
    # Тонкие линии по тикерам
    for ticker in TICKERS:
        ax1.plot(dims, fnn_results[ticker][ci], color=COMP_COLORS[ci],
                 lw=0.7, alpha=0.25)
    # Медиана — жирная
    ax1.plot(dims, fnn_med[ci], color=COMP_COLORS[ci], lw=2.5,
             label=COMP_LABELS[ci], zorder=5)

ax1.axhline(0.10, color="gray", lw=1.5, ls="--", alpha=0.7, label="FNN=10%")
ax1.axhline(0.05, color="gray", lw=1.0, ls=":",  alpha=0.5, label="FNN=5%")
ax1.axvline(20, color="red", lw=1.5, ls="--", alpha=0.6, label="p=20 (стандарт)")
ax1.axvline(8,  color="orange", lw=1.5, ls=":", alpha=0.7, label="p=8")

ax1.set_xlabel("Размерность вложения d", fontsize=11)
ax1.set_ylabel("Доля ложных соседей (FNN rate)", fontsize=11)
ax1.set_xlim(1, D_MAX); ax1.set_ylim(0, 1.02)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax1.legend(fontsize=9, loc="upper right", ncol=2)
ax1.grid(alpha=0.35)

plt.tight_layout()
out1 = OUT_DIR / "48_fnn_all_components.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\n  Рис 1: {out1}")
plt.close(fig1)


# ── Рис 2: отдельные панели на каждую компоненту ─────────────────────────────

fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
fig2.suptitle(
    f"FNN по компонентам — отдельные панели (τ={TAU}, R_tol={R_TOL}, Theiler={THEILER})",
    fontsize=12, fontweight="bold"
)

for ci, ax in zip(range(n_comp), axes2.flatten()):
    # Per-ticker (светлые)
    for ticker in TICKERS:
        ax.plot(dims, fnn_results[ticker][ci], color=COMP_COLORS[ci],
                lw=1.0, alpha=0.35)
    # Медиана (жирная)
    ax.plot(dims, fnn_med[ci], color=COMP_COLORS[ci], lw=2.8, zorder=5)

    # Горизонтали
    ax.axhline(0.10, color="gray", lw=1.5, ls="--", alpha=0.7)
    ax.axhline(0.05, color="gray", lw=1.0, ls=":",  alpha=0.5)
    # Вертикали: p=20 и p=8
    ax.axvline(20, color="red",    lw=1.5, ls="--", alpha=0.6, label="p=20")
    ax.axvline(8,  color="orange", lw=1.5, ls=":", alpha=0.7, label="p=8")

    # Отметить первый d < 10%
    d10 = next((d+1 for d, v in enumerate(fnn_med[ci]) if not np.isnan(v) and v < 0.10), None)
    if d10 is not None and d10 <= D_MAX:
        ax.axvline(d10, color="green", lw=2.0, ls="-", alpha=0.7)
        ax.text(d10 + 0.3, 0.92, f"d*={d10}", color="green", fontsize=9, va="top")

    ax.set_title(COMP_LABELS[ci], fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("d", fontsize=9)
    ax.set_ylabel("FNN rate", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.30)

plt.tight_layout()
out2 = OUT_DIR / "48_fnn_panels.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"  Рис 2: {out2}")
plt.close(fig2)


# ── Рис 3: быстрые vs медленные (2 группы) ───────────────────────────────────

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle(
    "FNN: быстрые компоненты (C0–C2) vs медленные (C3–C5)",
    fontsize=12, fontweight="bold"
)

groups = [(range(3), "Быстрые C0–C2 (шум)"), (range(3, 6), "Медленные C3–C5 (сигнал)")]
for ax, (idxs, title) in zip(axes3, groups):
    for ci in idxs:
        ax.plot(dims, fnn_med[ci], color=COMP_COLORS[ci], lw=2.5,
                label=COMP_LABELS[ci], zorder=5)
        for ticker in TICKERS:
            ax.plot(dims, fnn_results[ticker][ci], color=COMP_COLORS[ci],
                    lw=0.7, alpha=0.2)
    ax.axhline(0.10, color="gray", lw=1.5, ls="--", alpha=0.7, label="FNN=10%")
    ax.axvline(20, color="red",    lw=1.5, ls="--", alpha=0.6, label="p=20")
    ax.axvline(8,  color="orange", lw=1.5, ls=":",  alpha=0.7, label="p=8")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("d", fontsize=10); ax.set_ylabel("FNN rate", fontsize=10)
    ax.set_xlim(1, D_MAX); ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=9); ax.grid(alpha=0.35)

plt.tight_layout()
out3 = OUT_DIR / "48_fnn_groups.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"  Рис 3: {out3}")
plt.close(fig3)


# ── Итоги ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ИТОГИ")
print("=" * 70)

print(f"\n  {'Компонента':24} | {'FNN@d=1':>8} | {'FNN@d=4':>8} | {'FNN@d=8':>8} | {'FNN@d=20':>8} | d*(FNN<10%)")
print("  " + "-" * 75)
for ci in range(n_comp):
    d10 = next((d+1 for d, v in enumerate(fnn_med[ci]) if not np.isnan(v) and v < 0.10), ">"+str(D_MAX))
    v1  = fnn_med[ci][0]; v4 = fnn_med[ci][3]; v8 = fnn_med[ci][7]; v20 = fnn_med[ci][19]
    print(f"  {COMP_LABELS[ci]:24} | {v1:>7.1%} | {v4:>7.1%} | {v8:>7.1%} | {v20:>7.1%} | {str(d10):>11}")

print(f"\nФайлы:")
for out in [out1, out2, out3]:
    print(f"  {out.name}")
print("=" * 70)
