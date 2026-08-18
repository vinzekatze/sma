"""
75 — Local Projective noise reduction (Grassberger-Hegger 1993).

Метод: вложить ряд в фазовое пространство размерности m, для каждой точки
найти k ближайших соседей, построить локальное касательное подпространство
аттрактора через PCA, спроецировать точку на него. Итерируется n_iter раз.
Диагональное усреднение → реконструкция скалярного ряда.

Отличие от Schreiber (скр.59): Schreiber проецирует на локальное среднее
(локальный центроид), Local Projective — на касательное d-мерное подпространство.
Более строгая математика для детерминированных аттракторов.

Применяем к ratio (как SSA в скр.74) → diff → att (filtered dratio).

Протокол (офлайн, диагностика):
  Тикеры: 8 × 1d. Методы:
    raw       — diff(ratio) без обработки
    LP-causal — Butterworth causal на diff(ratio), Wn=0.125 (эталон)
    SSA       — offline SSA на ratio, L=32 k=2 (лучший из скр.73)
    LP_proj   — Local Projective на ratio, несколько конфигов (m, d, k, n_iter)

  FNN(d=1..15) для каждого метода — проверяем, выявляется ли аттрактор.

Графики:
  A — сигнал: raw / LP-causal / SSA / LP_proj-best — SBER, последние 200 баров
  B — FNN-кривые (среднее по 8 тикерам)
  C — реконструкция цены: LP-causal / SSA / LP_proj — SBER, последние 300 баров
  D — фазовый портрет diff(ratio): raw / LP-causal / LP_proj-best
"""

import json, sys, time
from pathlib import Path

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

TICKERS  = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL = "1d"
SOS_LP   = butter(8, 0.125, btype="low", output="sos")

# FNN
D_MAX   = 15
TAU     = 1
R_TOL   = 10.0
A_TOL   = 2.0
THEILER = 5
FNN_THR = 0.05

SBER_SHOW  = 200   # баров для панели сигналов
ZOOM_PRICE = 300   # баров для реконструкции цены

# ── Local Projective конфиги (m, d_proj, k, n_iter) ───────────────────────────
LP_CONFIGS = [
    dict(m=5,  d=2, k=30, n=1, label="LP m=5 d=2 k=30 n=1"),
    dict(m=5,  d=3, k=30, n=3, label="LP m=5 d=3 k=30 n=3"),
    dict(m=9,  d=3, k=30, n=1, label="LP m=9 d=3 k=30 n=1"),
    dict(m=9,  d=3, k=30, n=3, label="LP m=9 d=3 k=30 n=3"),
    dict(m=9,  d=4, k=50, n=3, label="LP m=9 d=4 k=50 n=3"),
]

# ── helpers ────────────────────────────────────────────────────────────────────

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


def ssa_offline(series: np.ndarray, L: int, n_keep: int) -> np.ndarray:
    """Offline SSA: реконструкция по n_keep ведущим компонентам."""
    N = len(series); K = N - L + 1
    if K < 1:
        return series.copy()
    rows = np.arange(K)[:, None] + np.arange(L)[None, :]
    X    = series[rows]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    nk   = min(n_keep, len(s))
    Xr   = (U[:, :nk] * s[:nk]) @ Vt[:nk, :]
    result = np.zeros(N); count = np.zeros(N, dtype=np.int32)
    for i in range(K):
        result[i:i + L] += Xr[i]; count[i:i + L] += 1
    return result / np.maximum(count, 1)


def local_projective(series: np.ndarray, m: int, d_proj: int,
                     k: int, n_iter: int = 1) -> np.ndarray:
    """
    Local Projective noise reduction (Grassberger-Hegger).

    Для каждой точки фазового пространства:
      1. Найти k ближайших соседей
      2. PCA на соседях → d_proj ведущих направлений (касательное подпространство)
      3. Спроецировать точку на локальный аффинный d_proj-мерный многообразный
    Диагональное усреднение → реконструкция скалярного ряда.
    Итерируется n_iter раз.
    """
    s = series.copy().astype(np.float64)
    N = len(s)
    k_eff = min(k, N - m)   # не может быть больше числа точек

    for _ in range(n_iter):
        n_pts = N - m + 1
        if n_pts < k_eff + 1:
            break
        rows = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
        X    = s[rows]                    # (n_pts, m)

        tree = KDTree(X)
        _, inds = tree.query(X, k=k_eff + 1)  # +1: сам включён

        X_proj = np.empty_like(X)
        for i in range(n_pts):
            nn       = inds[i, 1:]        # исключаем саму точку
            X_nn     = X[nn]              # (k_eff, m)
            centroid = X_nn.mean(axis=0)
            centered = X_nn - centroid
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            # d_proj главных направлений касательного подпространства
            V_d      = Vt[:d_proj].T      # (m, d_proj)
            xc       = X[i] - centroid
            X_proj[i] = centroid + V_d @ (V_d.T @ xc)

        # Диагональное усреднение
        result = np.zeros(N); count = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            result[i:i + m] += X_proj[i]; count[i:i + m] += 1
        s = result / np.maximum(count, 1)

    return s


def fnn_fractions(series: np.ndarray) -> np.ndarray:
    N   = len(series); sig = np.std(series)
    fracs = np.full(D_MAX, np.nan)
    for m_dim in range(1, D_MAX + 1):
        n_pts = N - (m_dim + 1) * TAU
        if n_pts < 10:
            break
        idx  = np.arange(m_dim)[None, :] * TAU + np.arange(n_pts)[:, None]
        Xm   = series[idx]
        idx1 = np.arange(m_dim + 1)[None, :] * TAU + np.arange(n_pts)[:, None]
        Xm1  = series[idx1]
        tree = KDTree(Xm)
        dists, inds = tree.query(Xm, k=2)
        d_m = dists[:, 1]; nn = inds[:, 1]
        mask = np.abs(np.arange(n_pts) - nn) > THEILER
        if mask.sum() < 5:
            break
        d_m1  = np.linalg.norm(Xm1 - Xm1[nn], axis=1)
        valid = mask & (d_m > 1e-10)
        ratio_d = np.where(valid, d_m1 / np.maximum(d_m, 1e-10), 0.0)
        fnn_flag = ((ratio_d > R_TOL) | ((d_m1 / max(sig, 1e-10)) > A_TOL)) & valid
        fracs[m_dim - 1] = fnn_flag[valid].mean() if valid.sum() > 0 else 1.0
    return fracs


def p_opt_fn(fracs: np.ndarray):
    for d, f in enumerate(fracs, 1):
        if not np.isnan(f) and f < FNN_THR:
            return d
    return None

# ── загрузка данных ────────────────────────────────────────────────────────────

def load_data(ticker: str):
    raw    = json.loads((DATADIR / ticker / f"{INTERVAL}.json").read_text())
    cands  = raw["candles"] if isinstance(raw, dict) else raw
    close  = np.array([c["close"] for c in cands], dtype=np.float64)
    trend  = logtrend_causal(close)
    ratio  = close / trend
    return close, trend, ratio

print("Загрузка данных...")
all_data = {t: load_data(t) for t in TICKERS}

# ── вычисление сигналов и FNN ─────────────────────────────────────────────────

method_fracs: dict[str, np.ndarray] = {}   # name → (8, D_MAX)

def run_fnn(name: str, get_sig):
    print(f"  FNN {name} ...", flush=True)
    fracs = []
    for t in TICKERS:
        sig = get_sig(*all_data[t])
        sig = (sig - sig.mean()) / (sig.std() + 1e-12)
        fracs.append(fnn_fractions(sig))
    method_fracs[name] = np.array(fracs)

t0 = time.time()

run_fnn("raw",    lambda c, tr, ra: np.diff(ra))
run_fnn("LP-causal", lambda c, tr, ra: sosfilt(SOS_LP, np.diff(ra)))
run_fnn("SSA L=32 k=2",
        lambda c, tr, ra: np.diff(ssa_offline(ra, 32, 2)))

for cfg in LP_CONFIGS:
    run_fnn(cfg["label"],
            lambda c, tr, ra, _cfg=cfg: np.diff(
                local_projective(ra, _cfg["m"], _cfg["d"], _cfg["k"], _cfg["n"])))

print(f"FNN готов за {time.time()-t0:.1f}с")

mean_fracs = {n: arr.mean(axis=0) for n, arr in method_fracs.items()}
dims = np.arange(1, D_MAX + 1)

# ── Таблица p_opt ──────────────────────────────────────────────────────────────
print("\n── p_opt (медиана по 8 тикерам) ──────────────────────────────")
print(f"{'Метод':<30}  p_opt  N/8")
print("─" * 48)
for name in method_fracs:
    po_list = [p_opt_fn(row) for row in method_fracs[name]]
    found   = [p for p in po_list if p is not None]
    med     = f"{np.median(found):.0f}" if found else "—"
    print(f"  {name:<28}  {med:>5}  {len(found)}/8")

# ── График A: сигнал SBER ──────────────────────────────────────────────────────
close_s, trend_s, ratio_s = all_data["SBER"]
dratio_s = np.diff(ratio_s)
n_show   = min(SBER_SHOW, len(dratio_s))
sl_s     = slice(-n_show, None)

# Лучший LP_proj по p_opt
best_lp = min(LP_CONFIGS, key=lambda c: (
    np.median([p_opt_fn(r) or D_MAX + 1 for r in method_fracs[c["label"]]]),
    c["n"]
))
print(f"\nЛучший LP_proj: {best_lp['label']}")

lp_best_sig  = np.diff(local_projective(ratio_s, best_lp["m"], best_lp["d"],
                                         best_lp["k"], best_lp["n"]))
ssa_sig      = np.diff(ssa_offline(ratio_s, 32, 2))
lp_causal_dr = sosfilt(SOS_LP, dratio_s)

panels = [
    ("raw dratio",           dratio_s,      "steelblue"),
    ("LP-causal (Wn=0.125)", lp_causal_dr,  "#fb8c00"),
    ("SSA offline L=32 k=2", ssa_sig,       "#2ca02c"),
    (best_lp["label"],       lp_best_sig,   "#9467bd"),
]

fig_a, axes = plt.subplots(len(panels), 1, figsize=(13, 2.4 * len(panels)), sharex=True)
fig_a.suptitle(f"75: Качество сигнала — SBER (последние {n_show} баров)", fontsize=11)
for ax, (title, sig, col) in zip(axes, panels):
    ax.plot(sig[sl_s], color=col, lw=0.9)
    ax.set_title(title); ax.axhline(0, color="k", lw=0.4, ls="--")
axes[-1].set_xlabel("бар")
fig_a.tight_layout()
fig_a.savefig(FIGDIR / "75_signal.png", dpi=120)
plt.close(fig_a)
print("Рис. A сохранён")

# ── График B: FNN-кривые ───────────────────────────────────────────────────────
fig_b, ax = plt.subplots(figsize=(11, 6))
ax.set_title("75: FNN-кривые (среднее по 8 тикерам)", fontsize=12)

ax.plot(dims, mean_fracs["raw"],         color="black",   lw=2.0, ls="-",  label="raw")
ax.plot(dims, mean_fracs["LP-causal"],   color="#fb8c00", lw=1.8, ls="-.", label="LP-causal")
ax.plot(dims, mean_fracs["SSA L=32 k=2"],color="#2ca02c", lw=1.6, ls="--", label="SSA L=32 k=2")

colors_lp = ["#7b2d8b", "#9467bd", "#c5b0d5", "#6a0572", "#a23b72"]
for cfg, col in zip(LP_CONFIGS, colors_lp):
    ax.plot(dims, mean_fracs[cfg["label"]], color=col, lw=1.4, label=cfg["label"])

ax.axhline(FNN_THR, color="red", ls=":", lw=1.2, label=f"порог {FNN_THR}")
ax.set_xlabel("размерность d"); ax.set_ylabel("FNN-фракция")
ax.legend(fontsize=7, ncol=2); ax.set_ylim(0, 1)
fig_b.tight_layout()
fig_b.savefig(FIGDIR / "75_fnn_curves.png", dpi=120)
plt.close(fig_b)
print("Рис. B сохранён")

# ── График C: реконструкция цены ──────────────────────────────────────────────
N_s = len(close_s)
sl_p = slice(N_s - ZOOM_PRICE, N_s)

def price_from_smooth_ratio(smooth_r, trend):
    return smooth_r * trend

ratio_lp_causal   = ratio_s[0] + np.concatenate([[0.0], np.cumsum(lp_causal_dr)])
ratio_ssa         = ssa_offline(ratio_s, 32, 2)
ratio_lp_best     = local_projective(ratio_s, best_lp["m"], best_lp["d"],
                                      best_lp["k"], best_lp["n"])
# Второй LP_proj для сравнения
cfg2 = LP_CONFIGS[3]  # m=9 d=3 k=30 n=3
ratio_lp2         = local_projective(ratio_s, cfg2["m"], cfg2["d"],
                                      cfg2["k"], cfg2["n"])

fig_c, ax = plt.subplots(figsize=(14, 5))
ax.set_title(f"75: Реконструкция цены — SBER (последние {ZOOM_PRICE} баров)", fontsize=11)
ax.plot(close_s[sl_p],                                  color="black",   lw=0.9, label="close (факт)")
ax.plot(trend_s[sl_p],                                  color="grey",    lw=1.0, ls="--", label="logtrend", alpha=0.6)
ax.plot(price_from_smooth_ratio(ratio_lp_causal, trend_s)[sl_p],
        color="#fb8c00", lw=1.2, label="LP-causal cumsum", alpha=0.85)
ax.plot(price_from_smooth_ratio(ratio_ssa, trend_s)[sl_p],
        color="#2ca02c", lw=1.2, ls="--", label="SSA L=32 k=2", alpha=0.85)
ax.plot(price_from_smooth_ratio(ratio_lp_best, trend_s)[sl_p],
        color="#9467bd", lw=1.4, label=best_lp["label"], alpha=0.9)
ax.plot(price_from_smooth_ratio(ratio_lp2, trend_s)[sl_p],
        color="#c5b0d5", lw=1.2, ls="-.", label=cfg2["label"], alpha=0.85)
ax.legend(fontsize=8); ax.set_xlabel("бар"); ax.set_ylabel("цена")
fig_c.tight_layout()
fig_c.savefig(FIGDIR / "75_price_zoom.png", dpi=120)
plt.close(fig_c)
print("Рис. C сохранён")

# ── График D: фазовый портрет ──────────────────────────────────────────────────
fig_d, axes = plt.subplots(1, 4, figsize=(16, 4))
fig_d.suptitle("75: Фазовый портрет diff(ratio)[t] vs diff(ratio)[t-1] — SBER", fontsize=10)

def phase_plot(ax, sig, title, col):
    ax.scatter(sig[:-1], sig[1:], s=1.5, alpha=0.3, color=col)
    ax.set_title(title, fontsize=8)
    ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel("x[t-1]"); ax.set_ylabel("x[t]")

phase_plot(axes[0], dratio_s,               "raw dratio",           "steelblue")
phase_plot(axes[1], lp_causal_dr,           "LP-causal",            "#fb8c00")
phase_plot(axes[2], ssa_sig,                "SSA L=32 k=2",         "#2ca02c")
phase_plot(axes[3], lp_best_sig,            best_lp["label"][:25],  "#9467bd")
fig_d.tight_layout()
fig_d.savefig(FIGDIR / "75_phase_portrait.png", dpi=120)
plt.close(fig_d)
print("Рис. D сохранён")

print("\nГотово.")
for ltr, fn in zip("ABCD", ["75_signal.png", "75_fnn_curves.png",
                             "75_price_zoom.png", "75_phase_portrait.png"]):
    print(f"  {ltr}: {FIGDIR/fn}")
