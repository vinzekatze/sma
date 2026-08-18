"""
73 — SSA как шумоподавление: проявление аттрактора.

Гипотеза: SSA удалит шумовые компоненты dratio и проявит детерминированный
аттрактор — аналогично MA(w≥8) из скр.54.

Протокол (диагностика, без прогноза):
  ratio   = close / logtrend_causal(close)
  dratio  = diff(ratio)
  Методы:
    raw       — dratio без обработки
    MA(8)     — causal MA(8) на dratio (эталон из скр.54)
    LP        — Butterworth LP causal (Wn=0.125, ord=8) на dratio
    SSA(L, k) — offline SSA, окно L, оставляем k ведущих компонент

  FNN(d=1..15) для каждого метода, усреднение по 8 тикерам.

Графики:
  A — сигнал: raw vs LP vs SSA лучший — последние 200 баров SBER
  B — FNN-кривые: raw, MA(8), LP, SSA-конфиги
  C — тепловая карта p_opt(L, k) для SSA
  D — FNN@d=5 (фракция) как ф-ция (L, k)
"""

import json, sys, time
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.spatial import KDTree

ROOT    = Path(__file__).resolve().parent.parent.parent
FIGDIR  = ROOT / "research" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
DATADIR = ROOT / "data" / "candles"
sys.path.insert(0, str(ROOT))

# ── параметры ─────────────────────────────────────────────────────────────────
TICKERS  = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL = "1d"

# SSA sweep
L_LIST  = [8, 16, 32, 64]
K_LIST  = [1, 2, 3, 4, 5]   # n_keep

# FNN
D_MAX   = 15
TAU     = 1
R_TOL   = 10.0
A_TOL   = 2.0
THEILER = 5
FNN_THR = 0.05   # порог «аттрактор найден»

# LP-causal
SOS_LP  = butter(8, 0.125, btype="low", output="sos")

SBER_PREVIEW = 200   # баров для графика сигнала

# ── вспомогательные функции ────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n  = len(close); lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    b  = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a  = (cy - b * ct) / cn
    trend = np.exp(a + b * t); trend[:2] = close[:2]
    return trend


def causal_ma(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    cs  = np.cumsum(np.concatenate([[0.0], x]))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (cs[i + 1] - cs[lo]) / (i - lo + 1)
    return out


def ssa_denoise(series: np.ndarray, L: int, n_keep: int) -> np.ndarray:
    """Offline SSA: SVD траекторной матрицы, реконструкция по n_keep компонентам."""
    N = len(series)
    K = N - L + 1
    if K < 1 or L < 2:
        return series.copy()
    # Траекторная (Hankel) матрица (K × L)
    rows = np.arange(K)[:, None] + np.arange(L)[None, :]
    X    = series[rows]           # (K, L)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    nk   = min(n_keep, len(s))
    # Реконструкция
    X_rec = (U[:, :nk] * s[:nk]) @ Vt[:nk, :]
    # Антидиагональное усреднение (Hankelization)
    result = np.zeros(N)
    count  = np.zeros(N, dtype=np.int32)
    for i in range(K):
        result[i:i + L] += X_rec[i]
        count [i:i + L] += 1
    return result / np.maximum(count, 1)


def fnn_fractions(series: np.ndarray, d_max: int = D_MAX) -> np.ndarray:
    """
    Возвращает массив fnn_frac[d] для d=1..d_max (индекс 0 = d=1).
    Theiler window = THEILER.
    """
    N   = len(series)
    sig = np.std(series)
    fracs = np.full(d_max, np.nan)
    for m in range(1, d_max + 1):
        # Вложение размерности m: строки (i, i+1..i+(m-1)) для i=0..N-m*TAU
        n_pts = N - (m + 1) * TAU
        if n_pts < 10:
            break
        idx = np.arange(m)[None, :] * TAU + np.arange(n_pts)[:, None]
        Xm  = series[idx]                   # (n_pts, m)
        idx1 = np.arange(m + 1)[None, :] * TAU + np.arange(n_pts)[:, None]
        Xm1 = series[idx1[:, :m + 1]]       # (n_pts, m+1) — следующая размерность

        tree = KDTree(Xm)
        # Ищем 2 ближайших (сам + сосед)
        dists, inds = tree.query(Xm, k=2)
        d_m   = dists[:, 1]                 # расстояние до ближайшего соседа в m
        nn    = inds[:, 1]                  # индекс ближайшего соседа

        # Theiler: исключаем временных соседей
        mask = np.abs(np.arange(n_pts) - nn) > THEILER
        if mask.sum() < 5:
            break

        d_m1 = np.linalg.norm(Xm1 - Xm1[nn], axis=1)  # расстояние в m+1

        valid = mask & (d_m > 1e-10)
        ratio_dist = np.where(valid, d_m1 / np.maximum(d_m, 1e-10), 0.0)
        crit1 = ratio_dist > R_TOL
        crit2 = (d_m1 / max(sig, 1e-10)) > A_TOL
        fnn_flag = (crit1 | crit2) & valid

        fracs[m - 1] = fnn_flag[valid].mean() if valid.sum() > 0 else 1.0
    return fracs


def p_opt(fracs: np.ndarray) -> int | None:
    """Первое d, где FNN < FNN_THR."""
    for d, f in enumerate(fracs, start=1):
        if not np.isnan(f) and f < FNN_THR:
            return d
    return None

# ── загрузка данных ────────────────────────────────────────────────────────────

def load_dratio(ticker: str) -> np.ndarray:
    path = DATADIR / ticker / f"{INTERVAL}.json"
    raw  = json.loads(path.read_text())
    candles = raw["candles"] if isinstance(raw, dict) else raw
    close = np.array([c["close"] for c in candles], dtype=np.float64)
    trend = logtrend_causal(close)
    ratio = close / trend
    return np.diff(ratio)

# ── главный цикл ───────────────────────────────────────────────────────────────

print("Загрузка данных...")
data = {t: load_dratio(t) for t in TICKERS}

# Название метода → FNN-фракции (N_tickers × D_MAX)
method_fracs: dict[str, np.ndarray] = {}

# --- baseline методы ---
def run_method(name: str, transform):
    print(f"  {name} ...", flush=True)
    fracs = []
    for t in TICKERS:
        sig  = transform(data[t])
        sig  = (sig - sig.mean()) / (sig.std() + 1e-12)
        fracs.append(fnn_fractions(sig))
    method_fracs[name] = np.array(fracs)   # (8, D_MAX)

t0 = time.time()
run_method("raw",    lambda x: x)
run_method("MA(8)",  lambda x: causal_ma(x, 8))
run_method("LP",     lambda x: sosfilt(SOS_LP, x))

# --- SSA configs ---
for L, k in product(L_LIST, K_LIST):
    run_method(f"SSA L={L} k={k}", lambda x, _L=L, _k=k: ssa_denoise(x, _L, _k))

print(f"FNN готов за {time.time()-t0:.1f}с")

# ── усреднение по тикерам ──────────────────────────────────────────────────────
mean_fracs = {name: arr.mean(axis=0) for name, arr in method_fracs.items()}

dims = np.arange(1, D_MAX + 1)

# ── График A: сигнал SBER ──────────────────────────────────────────────────────
dr_sber = data["SBER"]
n_show  = min(SBER_PREVIEW, len(dr_sber))
x_sber  = dr_sber[-n_show:]
# Лучший SSA по mean p_opt (наименьший p_opt → самый первый)
best_ssa = None
best_popt = 999
for name, arr in method_fracs.items():
    if not name.startswith("SSA"):
        continue
    po = [p_opt(row) or D_MAX + 1 for row in arr]
    med_po = np.median(po)
    if med_po < best_popt:
        best_popt = med_po
        best_ssa  = name

print(f"Лучший SSA: {best_ssa}")
L_best = int(best_ssa.split("L=")[1].split()[0])
k_best = int(best_ssa.split("k=")[1])

fig_a, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig_a.suptitle(f"73: Качество сигнала — SBER 1d (последние {n_show} баров)", fontsize=12)
axes[0].plot(x_sber, color="steelblue", lw=0.8)
axes[0].set_title("raw dratio")
axes[0].axhline(0, color="k", lw=0.5, ls="--")

lp_sig = sosfilt(SOS_LP, dr_sber)[-n_show:]
axes[1].plot(lp_sig, color="#fb8c00", lw=0.9)
axes[1].set_title("LP-causal (Wn=0.125, ord=8)  — τ≈75 баров")
axes[1].axhline(0, color="k", lw=0.5, ls="--")

ssa_sig = ssa_denoise(dr_sber, L_best, k_best)[-n_show:]
axes[2].plot(ssa_sig, color="#2ca02c", lw=0.9)
axes[2].set_title(f"SSA offline {best_ssa}")
axes[2].axhline(0, color="k", lw=0.5, ls="--")
axes[2].set_xlabel("бар")

fig_a.tight_layout()
fig_a.savefig(FIGDIR / "73_signal.png", dpi=120)
plt.close(fig_a)
print("Рис. A сохранён")

# ── График B: FNN-кривые ───────────────────────────────────────────────────────
fig_b, ax = plt.subplots(figsize=(11, 6))
ax.set_title("73: FNN-кривые (среднее по 8 тикерам)", fontsize=12)

# baseline + ключевые SSA конфиги
styles = {
    "raw":   ("black",   "-",  2.0,  "raw dratio"),
    "MA(8)": ("#1f77b4", "--", 1.8,  "MA(8) causal (эталон скр.54)"),
    "LP":    ("#fb8c00", "-.", 1.8,  "LP-causal (Wn=0.125)"),
}
for name, (col, ls, lw, label) in styles.items():
    ax.plot(dims, mean_fracs[name], color=col, ls=ls, lw=lw, label=label)

# Несколько SSA конфигов с лучшим k
colors_ssa = plt.cm.Greens(np.linspace(0.4, 0.9, len(L_LIST)))
for i, L in enumerate(L_LIST):
    name = f"SSA L={L} k={k_best}"
    ax.plot(dims, mean_fracs[name], color=colors_ssa[i], lw=1.4,
            label=f"SSA L={L} k={k_best}")

ax.axhline(FNN_THR, color="red", ls=":", lw=1.2, label=f"порог {FNN_THR}")
ax.set_xlabel("размерность d"); ax.set_ylabel("FNN-фракция")
ax.legend(fontsize=8); ax.set_ylim(0, 1)
fig_b.tight_layout()
fig_b.savefig(FIGDIR / "73_fnn_curves.png", dpi=120)
plt.close(fig_b)
print("Рис. B сохранён")

# ── График C: тепловая карта p_opt(L, k) ──────────────────────────────────────
popt_grid = np.full((len(L_LIST), len(K_LIST)), np.nan)
for i, L in enumerate(L_LIST):
    for j, k in enumerate(K_LIST):
        name = f"SSA L={L} k={k}"
        po_list = [p_opt(row) for row in method_fracs[name]]
        found   = [p for p in po_list if p is not None]
        popt_grid[i, j] = np.median(found) if found else D_MAX + 1

fig_c, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(popt_grid, aspect="auto", cmap="RdYlGn_r",
               vmin=1, vmax=D_MAX + 1)
ax.set_xticks(range(len(K_LIST))); ax.set_xticklabels(K_LIST)
ax.set_yticks(range(len(L_LIST))); ax.set_yticklabels(L_LIST)
ax.set_xlabel("n_keep (k)"); ax.set_ylabel("окно L")
ax.set_title("73: медиана p_opt по 8 тикерам\n(зелёный = меньше = лучше; серый = не найден)")
plt.colorbar(im, ax=ax, label="p_opt")
for i in range(len(L_LIST)):
    for j in range(len(K_LIST)):
        v = popt_grid[i, j]
        txt = str(int(v)) if not np.isnan(v) else "—"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                color="white" if v > 8 else "black")
fig_c.tight_layout()
fig_c.savefig(FIGDIR / "73_popt_heatmap.png", dpi=120)
plt.close(fig_c)
print("Рис. C сохранён")

# ── График D: доля тикеров с аттрактором (FNN@d=5 < THR) ─────────────────────
found_grid = np.zeros((len(L_LIST), len(K_LIST)))
for i, L in enumerate(L_LIST):
    for j, k in enumerate(K_LIST):
        name = f"SSA L={L} k={k}"
        found_grid[i, j] = sum(
            1 for row in method_fracs[name]
            if any(not np.isnan(f) and f < FNN_THR for f in row)
        )

fig_d, ax = plt.subplots(figsize=(7, 5))
im2 = ax.imshow(found_grid, aspect="auto", cmap="YlGn",
                vmin=0, vmax=len(TICKERS))
ax.set_xticks(range(len(K_LIST))); ax.set_xticklabels(K_LIST)
ax.set_yticks(range(len(L_LIST))); ax.set_yticklabels(L_LIST)
ax.set_xlabel("n_keep (k)"); ax.set_ylabel("окно L")
ax.set_title(f"73: тикеров из {len(TICKERS)} с найденным аттрактором")
plt.colorbar(im2, ax=ax, label="N тикеров")
for i in range(len(L_LIST)):
    for j in range(len(K_LIST)):
        v = found_grid[i, j]
        ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=10,
                color="black")
fig_d.tight_layout()
fig_d.savefig(FIGDIR / "73_attractor_found.png", dpi=120)
plt.close(fig_d)
print("Рис. D сохранён")

# ── Сводная таблица в консоль ──────────────────────────────────────────────────
print("\n── p_opt (медиана по 8 тикерам) ──────────────────────────────")
print(f"{'Метод':<25}  p_opt  найден(N/8)")
print("─" * 45)
for name in ["raw", "MA(8)", "LP"] + [f"SSA L={L} k={k}" for L, k in product(L_LIST, K_LIST)]:
    po_list = [p_opt(row) for row in method_fracs[name]]
    found   = [p for p in po_list if p is not None]
    med     = f"{np.median(found):.0f}" if found else "—"
    print(f"  {name:<23}  {med:>5}  {len(found)}/{len(TICKERS)}")

print("\nГотово.")
print(f"  A: {FIGDIR/'73_signal.png'}")
print(f"  B: {FIGDIR/'73_fnn_curves.png'}")
print(f"  C: {FIGDIR/'73_popt_heatmap.png'}")
print(f"  D: {FIGDIR/'73_attractor_found.png'}")
