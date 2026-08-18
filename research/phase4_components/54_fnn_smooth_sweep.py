"""
54 — FNN на сглаженном ряду: close → MA(w) → Δratio.

Гипотеза: высокочастотный шум маскирует аттрактор медленной составляющей.
При увеличении окна сглаживания FNN должен убывать быстрее (или до меньших m),
что соответствует проявлению детерминированной структуры.

Протокол:
  ratio   = close / logtrend_causal(close)
  smooth  = MA(ratio, w)          causal rolling mean
  dratio  = diff(smooth[:origin]) — как вход в LWR
  FNN(dratio, m) для m=1..D_MAX

Окна w ∈ WINDOWS (числа Фибоначчи + 1 = baseline без сглаживания).
Тикеры: все 8 доступных 1d.

Вывод:
  - таблица p_opt(w) по тикерам
  - тепловая карта FNN(m, w) (усредн. по тикерам) — рис. A
  - линии FNN(m) для каждого w — рис. B (один тикер + среднее)
  - p_opt(w) по тикерам + среднее — рис. C
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
FIG_DIR  = ROOT / "research/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ─────────────────────────────────────────────────────────────────

TICKERS = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
WINDOWS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]   # w=1: без сглаживания (baseline)

TAU     = 1
D_MAX   = 20     # размерности 1..D_MAX
R_TOL   = 10.0
A_TOL   = 2.0
THEILER = 5      # исключаем временных соседей |t−j| ≤ W
FNN_THR = 0.05   # порог «аттрактор найден»

# ── logtrend ──────────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n   = len(close)
    lc  = np.log(np.maximum(close, 1e-10))
    t   = np.arange(n, dtype=np.float64)
    cn  = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t)
    trend[:2] = close[:2]
    return trend


def causal_ma(x: np.ndarray, w: int) -> np.ndarray:
    """Causal rolling mean (min_periods=1)."""
    if w <= 1:
        return x.copy()
    out = np.empty_like(x)
    cs  = np.cumsum(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        lo = max(0, i - w + 1)
        out[i] = (cs[i] - (cs[lo - 1] if lo > 0 else 0.0)) / (i - lo + 1)
    return out


# ── FNN с Theiler window ──────────────────────────────────────────────────────

def fnn(series: np.ndarray,
        d_max: int  = D_MAX,
        tau:   int  = TAU,
        r_tol: float = R_TOL,
        a_tol: float = A_TOL,
        theiler: int = THEILER) -> dict[int, float]:
    """
    Возвращает {d: fnn_fraction} для d=1..d_max.
    Theiler window: исключаем соседей с |t − j| ≤ theiler
    (временная корреляция не должна маскироваться как пространственная).
    """
    sigma   = np.std(series)
    results = {}

    for d in range(1, d_max + 1):
        n_d  = len(series) - (d - 1) * tau - 1   # точек при размерности d
        n_d1 = len(series) - d * tau - 1          # точек при d+1

        if n_d1 < 20:
            break

        # матрица задержек d-мерная
        X_d = np.column_stack([series[k * tau: k * tau + n_d] for k in range(d)])
        X_q = X_d[:n_d1]   # запросы (можно строить дерево + запросы на одних данных)

        tree   = KDTree(X_q)
        # k=theiler+2: берём с запасом, потом отфильтруем временных соседей
        k_query = min(theiler + 2, n_d1)
        dists, idxs = tree.query(X_q, k=k_query)

        count_total = 0
        count_false = 0

        for i in range(n_d1):
            # первый сосед вне Theiler window
            nn_j   = None
            nn_dist = None
            for rank in range(1, k_query):  # пропускаем rank=0 (сам себя)
                j = idxs[i, rank]
                if abs(i - j) > theiler:
                    nn_j    = j
                    nn_dist = dists[i, rank]
                    break
            if nn_j is None or nn_dist < 1e-12:
                continue

            # дополнительное измерение d+1
            extra_i = series[d * tau + i]
            extra_j = series[d * tau + nn_j]

            R_d1 = np.sqrt(nn_dist ** 2 + (extra_i - extra_j) ** 2)

            crit1 = abs(extra_i - extra_j) / (nn_dist + 1e-12) > r_tol
            crit2 = R_d1 / (sigma + 1e-12) > a_tol

            count_total += 1
            if crit1 or crit2:
                count_false += 1

        results[d] = count_false / count_total if count_total > 0 else float("nan")

    return results


def first_below(fnn_d: dict[int, float], thr: float = FNN_THR) -> int | None:
    for d in sorted(fnn_d):
        v = fnn_d[d]
        if not np.isnan(v) and v < thr:
            return d
    return None


# ── загрузка данных ───────────────────────────────────────────────────────────

def load_dratio_smoothed(ticker: str, w: int) -> np.ndarray | None:
    path = DATA_DIR / ticker / "1d.json"
    if not path.exists():
        return None
    with open(path) as f:
        candles = json.load(f)
    close  = np.array([float(c["close"]) for c in candles], dtype=np.float64)
    trend  = logtrend_causal(close)
    ratio  = close / trend
    smooth = causal_ma(ratio, w)
    dratio = np.diff(smooth)
    return dratio


# ── основной цикл ─────────────────────────────────────────────────────────────

# results[ticker][w] = {d: fnn_frac}
results: dict[str, dict[int, dict[int, float]]] = {}

t0 = time.time()
total = len(TICKERS) * len(WINDOWS)
done  = 0

for ticker in TICKERS:
    results[ticker] = {}
    for w in WINDOWS:
        dratio = load_dratio_smoothed(ticker, w)
        if dratio is None:
            results[ticker][w] = {}
            done += 1
            continue
        fnn_d = fnn(dratio)
        results[ticker][w] = fnn_d
        done += 1
        p_opt = first_below(fnn_d)
        print(f"[{done:3d}/{total}] {ticker} w={w:3d}  p_opt={p_opt}")

print(f"\nГотово за {time.time() - t0:.1f} с")

# ── таблица p_opt(w) ──────────────────────────────────────────────────────────

print("\n── p_opt (первое d с FNN < 5%) ──────────────────────────────")
header = f"{'w':>5}  " + "  ".join(f"{t:>5}" for t in TICKERS) + "  mean"
print(header)
print("─" * len(header))

p_opt_table: dict[int, list[int | None]] = {}
for w in WINDOWS:
    row = []
    for ticker in TICKERS:
        row.append(first_below(results[ticker].get(w, {})))
    p_opt_table[w] = row
    vals = [v for v in row if v is not None]
    mean_str = f"{np.mean(vals):.1f}" if vals else "—"
    cells = "  ".join(f"{(str(v) if v else '—'):>5}" for v in row)
    print(f"{w:>5}  {cells}  {mean_str}")

# ── рис. A: тепловая карта FNN(d, w) усреднённая по тикерам ──────────────────

# матрица (len(WINDOWS), D_MAX)
fnn_mean = np.full((len(WINDOWS), D_MAX), np.nan)
for wi, w in enumerate(WINDOWS):
    for di in range(D_MAX):
        d = di + 1
        vals = []
        for ticker in TICKERS:
            v = results[ticker].get(w, {}).get(d)
            if v is not None and not np.isnan(v):
                vals.append(v)
        if vals:
            fnn_mean[wi, di] = np.mean(vals)

fig_a, ax_a = plt.subplots(figsize=(12, 5))
im = ax_a.imshow(
    fnn_mean * 100,
    aspect="auto", origin="lower",
    extent=[0.5, D_MAX + 0.5, -0.5, len(WINDOWS) - 0.5],
    cmap="RdYlGn_r", vmin=0, vmax=80,
)
ax_a.set_yticks(range(len(WINDOWS)))
ax_a.set_yticklabels([str(w) for w in WINDOWS])
ax_a.set_xticks(range(1, D_MAX + 1))
ax_a.axhline(0.5, color="white", lw=0.5, ls=":")   # между w=1 и w=2

# маркеры p_opt для среднего
for wi, w in enumerate(WINDOWS):
    row = p_opt_table[w]
    vals = [v for v in row if v is not None]
    if vals:
        p_med = int(np.round(np.median(vals)))
        ax_a.plot(p_med, wi, "w^", ms=5, zorder=5)

plt.colorbar(im, ax=ax_a, label="FNN %")
ax_a.set_xlabel("d (размерность вложения)")
ax_a.set_ylabel("окно сглаживания w")
ax_a.set_title("FNN % — среднее по 8 тикерам | белый треугольник = медиана p_opt")
fig_a.tight_layout()
path_a = FIG_DIR / "54_fnn_smooth_heatmap.png"
fig_a.savefig(path_a, dpi=150)
print(f"\nРис. A: {path_a}")

# ── рис. B: линии FNN(d) для каждого w — SBER ────────────────────────────────

fig_b, axes_b = plt.subplots(1, 2, figsize=(14, 5))
cmap_b = plt.cm.plasma
norm_b = plt.Normalize(vmin=0, vmax=len(WINDOWS) - 1)

for panel_idx, ticker in enumerate(["SBER", "ALL"]):
    ax = axes_b[panel_idx]
    ds = list(range(1, D_MAX + 1))

    for wi, w in enumerate(WINDOWS):
        if ticker == "ALL":
            # среднее по тикерам
            yvals = [fnn_mean[wi, d - 1] * 100 if not np.isnan(fnn_mean[wi, d - 1]) else np.nan
                     for d in ds]
        else:
            fnn_d = results[ticker].get(w, {})
            yvals = [fnn_d.get(d, np.nan) * 100 for d in ds]

        color = cmap_b(norm_b(wi))
        lw    = 2.0 if w in (1, 5, 21, 89) else 0.9
        label = f"w={w}" if w in (1, 5, 21, 89) else None
        ax.plot(ds, yvals, color=color, lw=lw, label=label, alpha=0.85)

    ax.axhline(5, color="red", ls="--", lw=1.2, label="5% порог")
    ax.set_xlabel("d (размерность вложения)")
    ax.set_ylabel("FNN %")
    ax.set_title(f"FNN(d) при разных w — {'SBER' if ticker == 'SBER' else 'среднее'}")
    ax.set_xlim(1, D_MAX)
    ax.set_ylim(-2, 105)
    ax.legend(fontsize=8, loc="upper right")

# цветовая шкала w
sm = plt.cm.ScalarMappable(cmap=cmap_b, norm=norm_b)
sm.set_array([])
plt.colorbar(sm, ax=axes_b, label="индекс w (0=w1, 9=w89)", shrink=0.8)

fig_b.tight_layout()
path_b = FIG_DIR / "54_fnn_smooth_curves.png"
fig_b.savefig(path_b, dpi=150)
print(f"Рис. B: {path_b}")

# ── рис. C: p_opt(w) по тикерам + медиана ────────────────────────────────────

fig_c, ax_c = plt.subplots(figsize=(10, 5))
cmap_c = plt.cm.tab10
for ti, ticker in enumerate(TICKERS):
    pts_w, pts_p = [], []
    for wi, w in enumerate(WINDOWS):
        p_o = first_below(results[ticker].get(w, {}))
        if p_o is not None:
            pts_w.append(w)
            pts_p.append(p_o)
    ax_c.plot(pts_w, pts_p, "o-", color=cmap_c(ti / 10),
              lw=1.2, ms=5, label=ticker, alpha=0.75)

# медиана по тикерам
med_w, med_p = [], []
for wi, w in enumerate(WINDOWS):
    vals = [v for v in p_opt_table[w] if v is not None]
    if vals:
        med_w.append(w)
        med_p.append(float(np.median(vals)))
ax_c.plot(med_w, med_p, "k^-", lw=2.5, ms=8, label="медиана", zorder=10)

ax_c.set_xscale("log")
ax_c.set_xlabel("окно сглаживания w (log)")
ax_c.set_ylabel("p_opt (первое d с FNN < 5%)")
ax_c.set_title("p_opt vs w сглаживания — 8 тикеров 1d")
ax_c.legend(fontsize=8, ncol=2)
ax_c.grid(True, alpha=0.3)
ax_c.set_xticks(WINDOWS)
ax_c.set_xticklabels([str(w) for w in WINDOWS])

fig_c.tight_layout()
path_c = FIG_DIR / "54_fnn_smooth_popt.png"
fig_c.savefig(path_c, dpi=150)
print(f"Рис. C: {path_c}")

# ── итоговый вывод ────────────────────────────────────────────────────────────

print("\n── Итог ──────────────────────────────────────────────────────")
for wi, w in enumerate(WINDOWS):
    vals = [v for v in p_opt_table[w] if v is not None]
    found = sum(1 for v in vals if v is not None)
    if vals:
        print(f"w={w:3d}: p_opt_median={np.median(vals):.1f}  "
              f"min={min(vals)}  max={max(vals)}  "
              f"найдено у {found}/{len(TICKERS)} тикеров")
    else:
        print(f"w={w:3d}: аттрактор не обнаружен ни у одного тикера")
