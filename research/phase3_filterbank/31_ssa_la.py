"""
31 — SSA + LA.

SSA (Singular Spectrum Analysis / «Гусеница») — альтернативный способ
разложить ряд на компоненты без look-ahead и без boundary effects EMD.

Схема:
  1. Окно L → траекторная матрица X_traj (K × L)
  2. SVD → U, s, Vt
  3. Группировка сингулярных троек → реконструированные компоненты RC_i
  4. LA на каждой RC_i отдельно, сумма прогнозов

Walk-forward: SVD пересчитывается на каждом origin по dratio[:vo].
Boundary effects у SSA минимальны — реконструкция усредняет по диагоналям.
Конечный фрагмент длиной L имеет некоторый эффект, но на порядок меньше EMD.

Конфиги:
  baseline    — LWR на dratio
  ssa_all     — LWR на каждой RC (все), сумма
  ssa_lo      — LWR только на медленных RC (по аналогии с fb_lo_only)

Параметры sweep: L ∈ {20, 50, 100}, n_groups ∈ {3, 5, 8}.
"""

import json, sys, time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from sma.core.forecast.normalize import normalize
from sma.core.forecast.embedding import build_delay_matrix, last_vector

try:
    from scipy.stats import wilcoxon as _wlcx
    _HAS_SCI = True
except ImportError:
    _HAS_SCI = False

# ── параметры ─────────────────────────────────────────────────────────────────
TICKER    = "SBER"
INTERVAL  = "1d"
MA_WINDOW = 1000
P         = 5
XI        = 3 * (P + 1)   # 18
VAL_H     = 10
N_EVAL    = 200

L_LIST       = [20, 50, 100]
N_GROUPS     = 8      # сколько компонент оставляем (топ по singular value)
LO_FRAC      = 0.5    # «медленные» = нижние 50% по индексу (наибольший период)


# ── SSA ───────────────────────────────────────────────────────────────────────
def ssa_reconstruct(series: np.ndarray, L: int,
                    n_groups: int) -> np.ndarray:
    """
    Возвращает (n_groups, len(series)) — реконструированные компоненты.
    Компоненты отсортированы по убыванию singular value (i=0 — главный тренд).
    Последняя строка — сумма оставшихся (остаток).
    """
    N = len(series)
    K = N - L + 1
    if K < 1:
        return series[np.newaxis, :]

    # Траекторная матрица (K × L)
    idx   = np.arange(L)[None, :] + np.arange(K)[:, None]
    X_traj = series[idx]   # (K, L)

    # SVD
    U, s, Vt = np.linalg.svd(X_traj, full_matrices=False)
    n_keep   = min(n_groups - 1, len(s))   # последний — остаток

    components = np.zeros((n_groups, N))
    residual   = X_traj.copy()

    for i in range(n_keep):
        Xi = s[i] * np.outer(U[:, i], Vt[i, :])   # (K, L)
        residual -= Xi
        # Ганкелизация (усреднение по антидиагоналям)
        rc = np.zeros(N)
        wt = np.zeros(N)
        for k in range(K):
            rc[k: k + L] += Xi[k]
            wt[k: k + L] += 1
        components[i] = rc / wt

    # Остаток
    rc = np.zeros(N); wt = np.zeros(N)
    for k in range(K):
        rc[k: k + L] += residual[k]
        wt[k: k + L] += 1
    components[-1] = rc / wt

    return components


# ── LWR ──────────────────────────────────────────────────────────────────────
def _lwr_forecast(X: np.ndarray, y: np.ndarray, vec: np.ndarray,
                  xi: int) -> np.ndarray:
    v   = vec.copy()
    hat = np.empty(VAL_H)
    for h in range(VAL_H):
        dists  = np.linalg.norm(X - v, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        h_bw   = max(float(dists[nn_idx].max()), 1e-10)
        w      = np.exp(-0.5 * (dists[nn_idx] / h_bw) ** 2)
        A      = np.hstack([np.ones((xi, 1)), X[nn_idx]])
        sw     = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx], rcond=None)
        hat[h] = float(c[0] + v @ c[1:])
        v = np.roll(v, -1); v[-1] = hat[h]
    return hat


def _mape(hat: np.ndarray, ratio: np.ndarray, vo: int) -> float:
    r0     = float(ratio[vo])
    r_hat  = r0 + np.cumsum(hat)
    actual = ratio[vo + 1: vo + 1 + VAL_H]
    n_     = min(len(r_hat), len(actual))
    if n_ == 0:
        return np.nan
    return float(np.mean(
        np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)
    ))


# ── данные ────────────────────────────────────────────────────────────────────
with open(DATA_DIR / TICKER / f"{INTERVAL}.json") as f:
    candles = json.load(f)
df     = normalize(candles, window=MA_WINDOW).dropna(subset=["ma"]).reset_index(drop=True)
ratio  = df["ratio"].values
dratio = np.diff(ratio)
print(f"{TICKER} {INTERVAL}: {len(dratio)} баров Δratio")

min_orig = P + XI + max(L_LIST) + 5
max_orig = len(dratio) - VAL_H
origins  = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)
print(f"p={P}  ξ={XI}  origins={len(origins)}\n")


# ── baseline (один раз) ───────────────────────────────────────────────────────
base_mape = []
for vo in origins:
    X, y  = build_delay_matrix(dratio[:vo], P)
    vec   = last_vector(dratio[:vo], P).copy()
    base_mape.append(_mape(_lwr_forecast(X, y, vec, XI), ratio, vo))
base_mape = np.array(base_mape)
bm_base   = float(np.nanmedian(base_mape))
print(f"Baseline med = {bm_base:.5f}\n")


# ── sweep по L ────────────────────────────────────────────────────────────────
# RES[L] = {"all": arr, "lo": arr}
RES: dict[int, dict[str, np.ndarray]] = {}

for L in L_LIST:
    lo_idx = list(range(int(N_GROUPS * LO_FRAC), N_GROUPS))

    raw_all = []; raw_lo = []
    t0 = time.time()

    for vi, vo in enumerate(origins):
        series_vo = dratio[:vo]

        # SSA на dratio[:vo]
        comps = ssa_reconstruct(series_vo, L, N_GROUPS)   # (N_GROUPS, vo)

        hats = []
        for comp in comps:
            Xc, yc = build_delay_matrix(comp, P)
            if len(Xc) < XI:
                hats.append(np.zeros(VAL_H))
            else:
                vecc = last_vector(comp, P).copy()
                hats.append(_lwr_forecast(Xc, yc, vecc, XI))

        raw_all.append(_mape(np.sum(hats, axis=0), ratio, vo))
        raw_lo.append(_mape(np.sum([hats[i] for i in lo_idx], axis=0), ratio, vo))

    RES[L] = {
        "all": np.array(raw_all),
        "lo":  np.array(raw_lo),
    }

    med_all = float(np.nanmedian(RES[L]["all"]))
    med_lo  = float(np.nanmedian(RES[L]["lo"]))
    print(f"L={L:3d}  t={time.time()-t0:.1f}s  "
          f"ssa_all={med_all:.5f} ({(med_all-bm_base)/bm_base*100:+.1f}%)  "
          f"ssa_lo={med_lo:.5f} ({(med_lo-bm_base)/bm_base*100:+.1f}%)")


# ── таблица ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"{'Конфиг':<18}  {'median':>9}  {'Δ%':>7}  {'W-p':>8}")
print("─" * 50)
print(f"{'baseline':<18}  {bm_base:>9.5f}  {'—':>7}  {'—':>8}")

for L in L_LIST:
    for mode, arr in RES[L].items():
        name = f"ssa_L{L}_{mode}"
        med  = float(np.nanmedian(arr))
        dp   = (med - bm_base) / bm_base * 100
        wp   = "—"
        if _HAS_SCI:
            ok   = ~(np.isnan(base_mape) | np.isnan(arr))
            diff = arr[ok] - base_mape[ok]
            if ok.sum() >= 10 and not np.all(diff == 0):
                _, pv = _wlcx(diff)
                wp = f"{pv:.4f}"
        print(f"{name:<18}  {med:>9.5f}  {dp:>+6.2f}%  {wp:>8}")
    print()

print("=" * 72)

# Напоминание: filter bank результат для сравнения
print(f"\n[Справка скр.30]  fb_lo_only (C3+C4+C5):  med=0.00937  Δ=−28.3%")


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"SSA + LA  |  {TICKER} {INTERVAL}  "
    f"p={P}, ξ={XI}, n_groups={N_GROUPS}, N={N_EVAL}",
    fontsize=11,
)

# CDF для лучшего L
ax = axes[0]
best_L = min(L_LIST, key=lambda L: float(np.nanmedian(RES[L]["lo"])))
s_b = np.sort(base_mape[~np.isnan(base_mape)])
ax.plot(np.linspace(0, 100, len(s_b)), s_b,
        color="steelblue", lw=2.5, label=f"baseline  (med={bm_base:.4f})")
for mode, color in [("all", "tomato"), ("lo", "seagreen")]:
    arr = RES[best_L][mode]
    s   = np.sort(arr[~np.isnan(arr)])
    med = float(np.nanmedian(arr))
    ax.plot(np.linspace(0, 100, len(s)), s, color=color, lw=1.8,
            label=f"ssa_L{best_L}_{mode}  (med={med:.4f})")
ax.set_title(f"CDF val_mape (лучший L={best_L})")
ax.set_xlabel("Перцентиль"); ax.set_ylabel("val_mape")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0)

# Δ% по L
ax = axes[1]
x = np.arange(len(L_LIST))
w = 0.35
meds_all = [float(np.nanmedian(RES[L]["all"])) for L in L_LIST]
meds_lo  = [float(np.nanmedian(RES[L]["lo"]))  for L in L_LIST]
dp_all   = [(m - bm_base) / bm_base * 100 for m in meds_all]
dp_lo    = [(m - bm_base) / bm_base * 100 for m in meds_lo]

bars1 = ax.bar(x - w/2, dp_all, w, label="ssa_all", color="tomato",   alpha=0.8)
bars2 = ax.bar(x + w/2, dp_lo,  w, label="ssa_lo",  color="seagreen", alpha=0.8)
ax.axhline(0, color="black", lw=1)
ax.axhline(-28.3, color="steelblue", lw=1.5, ls="--",
           label="fb_lo_only (скр.30) −28.3%")
ax.set_xticks(x); ax.set_xticklabels([f"L={L}" for L in L_LIST])
ax.set_ylabel("ΔMAPE %  (отриц. = лучше)")
ax.set_title("SSA: Δ% по L  vs  filter bank")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
for bar in bars1 + bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
            f"{h:+.1f}%", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
out_path = OUT_DIR / "31_ssa_la.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
