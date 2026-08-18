"""
23 — PE-delay мягкий вес при больших p.

Скрипты 21-22 показали: при p=5 PE-delay не улучшает MAPE значимо,
потому что PE-вектор длины 5 несёт слабый сигнал (высокая автокорреляция).

Гипотеза: при больших p (10, 20, 40) delay-вектор охватывает больший
временной диапазон → PE-вектор той же длины p имеет больше вариативности
→ мягкий PE-вес начинает работать.

Тест: для каждого p ∈ {10, 20, 40}:
    baseline LWR  (без PE-взвешивания)
    LWR + soft PE weight, m_pe=p, α=0.25
    LWR + soft PE weight, m_pe=p, α=0.50

Метрика: val_mape (VAL_H баров).  Тикер: SBER 1d.
"""

import json, sys, time
from math import factorial
from itertools import permutations
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

# ── параметры ────────────────────────────────────────────────────────────────
TICKER    = "SBER"
INTERVAL  = "1d"
MA_WINDOW = 1000

P_LIST   = [10, 20, 40]
ALPHA_LIST = [0.0, 0.25, 0.50]   # 0.0 = baseline
_VAL_H   = 10
_PE_WIN  = 50
_PE_ORD  = 3
_N_EVAL  = 200


# ── PE ────────────────────────────────────────────────────────────────────────
_PIDX = {perm: i for i, perm in enumerate(permutations(range(_PE_ORD)))}

def _pe(x: np.ndarray) -> float:
    n = len(x)
    if n < _PE_ORD:
        return np.nan
    counts = np.zeros(factorial(_PE_ORD))
    for i in range(n - _PE_ORD + 1):
        counts[_PIDX[tuple(np.argsort(x[i:i + _PE_ORD]))]] += 1
    p = counts[counts > 0]; p /= p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(factorial(_PE_ORD)))

def rolling_pe(series: np.ndarray, win: int = _PE_WIN) -> np.ndarray:
    out = np.full(len(series), np.nan)
    for i in range(win - 1, len(series)):
        out[i] = _pe(series[i - win + 1: i + 1])
    return out


# ── PE-delay расстояния ───────────────────────────────────────────────────────
def _pe_dists(bar_pe: np.ndarray, pool_size: int, val_origin: int,
              m_pe: int) -> np.ndarray:
    q_start = val_origin - m_pe
    if q_start < 0 or pool_size + m_pe - 1 >= len(bar_pe):
        return np.zeros(pool_size)
    P_q = bar_pe[q_start: q_start + m_pe]
    if np.any(np.isnan(P_q)):
        return np.zeros(pool_size)
    idx  = np.arange(pool_size)[:, None] + np.arange(m_pe)[None, :]
    P_pl = bar_pe[idx]
    valid = ~np.any(np.isnan(P_pl), axis=1)
    d = np.zeros(pool_size)
    d[valid] = np.linalg.norm(P_pl[valid] - P_q, axis=1)
    return d


# ── LWR + мягкий PE-вес ──────────────────────────────────────────────────────
def _eval(X: np.ndarray, y: np.ndarray, vec: np.ndarray,
          pe_dists: np.ndarray, ratio: np.ndarray,
          val_origin: int, xi: int, alpha: float) -> tuple[float, float]:
    v   = vec.copy()
    hat = np.empty(_VAL_H)
    for h in range(_VAL_H):
        dists  = np.linalg.norm(X - v, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        nn_dx  = dists[nn_idx]
        h_bw   = max(float(nn_dx.max()), 1e-10)
        w      = np.exp(-0.5 * (nn_dx / h_bw) ** 2)
        if alpha > 0:
            nn_dpe = pe_dists[nn_idx]
            h_pe   = max(float(nn_dpe.mean()), 1e-10)
            w     *= np.exp(-alpha * (nn_dpe / h_pe) ** 2)
        A  = np.hstack([np.ones((xi, 1)), X[nn_idx]])
        sw = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx], rcond=None)
        hat[h] = float(c[0] + v @ c[1:])
        v = np.roll(v, -1); v[-1] = hat[h]

    d_k = float(np.sort(np.linalg.norm(X - vec, axis=1))[xi - 1])
    r0     = float(ratio[val_origin])
    r_hat  = r0 + np.cumsum(hat)
    actual = ratio[val_origin + 1: val_origin + 1 + _VAL_H]
    n_ = min(len(r_hat), len(actual))
    if n_ == 0:
        return np.nan, d_k
    mape = float(np.mean(
        np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)
    ))
    return mape, d_k


# ── данные ────────────────────────────────────────────────────────────────────
with open(DATA_DIR / TICKER / f"{INTERVAL}.json") as f:
    candles = json.load(f)
df     = normalize(candles, window=MA_WINDOW).dropna(subset=["ma"]).reset_index(drop=True)
ratio  = df["ratio"].values
dratio = np.diff(ratio)

print(f"{TICKER} {INTERVAL}: {len(dratio)} баров Δratio")
t0 = time.time()
bar_pe = rolling_pe(dratio)
print(f"Rolling PE: {time.time()-t0:.1f}s\n")


# ── walk-forward по p ─────────────────────────────────────────────────────────
# RES[p][alpha] = {"mape": np.ndarray, "dk": np.ndarray}
RES: dict[int, dict[float, dict]] = {}

for p in P_LIST:
    xi = 3 * (p + 1)
    min_orig = p + xi + _PE_WIN + p
    max_orig = len(dratio) - _VAL_H
    origins  = np.arange(max(min_orig, max_orig - _N_EVAL), max_orig)

    RES[p] = {a: {"mape": [], "dk": []} for a in ALPHA_LIST}

    t1 = time.time()
    print(f"p={p}  ξ={xi}  origins=[{origins[0]}…{origins[-1]}] ({len(origins)})", end="  ")
    for vo in origins:
        X, y  = build_delay_matrix(dratio[:vo], p)
        vec   = last_vector(dratio[:vo], p).copy()
        ps    = len(X)
        pd    = _pe_dists(bar_pe, ps, vo, p)   # m_pe = p

        for a in ALPHA_LIST:
            mape, dk = _eval(X, y, vec, pd, ratio, vo, xi, a)
            RES[p][a]["mape"].append(mape)
            RES[p][a]["dk"].append(dk)

    for a in ALPHA_LIST:
        RES[p][a]["mape"] = np.array(RES[p][a]["mape"])
        RES[p][a]["dk"]   = np.array(RES[p][a]["dk"])
    print(f"{time.time()-t1:.1f}s")


# ── таблица ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"{'p':>4}  {'α':>5}  {'ξ':>4}  {'MAPE mean':>10}  {'MAPE med':>9}  "
      f"{'vs base':>9}  {'d_k med':>8}")
print("─" * 72)
for p in P_LIST:
    xi = 3 * (p + 1)
    bm = float(np.nanmedian(RES[p][0.0]["mape"]))
    for a in ALPHA_LIST:
        mm  = float(np.nanmean(RES[p][a]["mape"]))
        med = float(np.nanmedian(RES[p][a]["mape"]))
        dd  = float(np.nanmedian(RES[p][a]["dk"]))
        tag = "baseline" if a == 0 else f"{(med-bm)/bm*100:+.2f}%"
        print(f"{p:>4}  {a:>5.2f}  {xi:>4}  {mm:>10.6f}  {med:>9.6f}  "
              f"{tag:>9}  {dd:>8.5f}")
    print()
print("=" * 72)

# Wilcoxon для лучшего α per p
try:
    from scipy.stats import wilcoxon
    print("\n── Wilcoxon: α=0.25 vs baseline ──")
    for p in P_LIST:
        b = RES[p][0.0]["mape"]; t = RES[p][0.25]["mape"]
        ok = ~(np.isnan(b) | np.isnan(t))
        diff = t[ok] - b[ok]
        if ok.sum() < 10 or np.all(diff == 0):
            continue
        stat, pv = wilcoxon(diff)
        med_b = float(np.nanmedian(b)); med_t = float(np.nanmedian(t))
        print(f"  p={p:2d}  α=0.25:  ΔMAPE={( med_t-med_b)/med_b*100:+.2f}%  "
              f"W={stat:.0f}  p={pv:.4f}  "
              f"{'✓ значимо' if pv < 0.05 else '— незначимо'}")
except ImportError:
    pass


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(
    f"PE-delay soft weight при разных p  |  {TICKER} {INTERVAL}  "
    f"(m_pe=p, val_h={_VAL_H})",
    fontsize=11,
)
COLORS_A = {0.0: "steelblue", 0.25: "seagreen", 0.50: "darkorange"}

for ax_i, p in enumerate(P_LIST):
    ax = axes[ax_i]
    for a in ALPHA_LIST:
        arr = np.sort(RES[p][a]["mape"][~np.isnan(RES[p][a]["mape"])])
        xi  = 3 * (p + 1)
        med = float(np.nanmedian(RES[p][a]["mape"]))
        lbl = f"α={a:.2f}  (med={med:.4f})"
        lw  = 2.5 if a == 0 else 1.8
        ax.plot(np.linspace(0, 100, len(arr)), arr,
                color=COLORS_A[a], linewidth=lw, label=lbl)
    ax.set_xlabel("Перцентиль")
    ax.set_ylabel("val_mape")
    ax.set_title(f"p={p}  ξ={3*(p+1)}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0)

plt.tight_layout()
out_path = OUT_DIR / "23_pe_delay_large_p.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")


# ── итог ──────────────────────────────────────────────────────────────────────
print("\n── Итог ──")
print(f"{'p':>4}  {'baseline med':>13}  {'α=0.25 med':>12}  {'Δ%':>7}  {'α=0.50 med':>12}  {'Δ%':>7}")
print("─" * 58)
for p in P_LIST:
    bm   = float(np.nanmedian(RES[p][0.0]["mape"]))
    m25  = float(np.nanmedian(RES[p][0.25]["mape"]))
    m50  = float(np.nanmedian(RES[p][0.50]["mape"]))
    print(f"{p:>4}  {bm:>13.5f}  {m25:>12.5f}  {(m25-bm)/bm*100:>+6.2f}%  "
          f"{m50:>12.5f}  {(m50-bm)/bm*100:>+6.2f}%")

print()
# Есть ли монотонный тренд: улучшение растёт с p?
deltas = {p: float(np.nanmedian(RES[p][0.25]["mape"])) / float(np.nanmedian(RES[p][0.0]["mape"])) - 1
          for p in P_LIST}
if all(deltas[P_LIST[i]] > deltas[P_LIST[i+1]] for i in range(len(P_LIST)-1)):
    print("Тренд: улучшение PE-weight РАСТЁТ с p (подтверждает гипотезу).")
elif all(deltas[P_LIST[i]] < deltas[P_LIST[i+1]] for i in range(len(P_LIST)-1)):
    print("Тренд: ухудшение PE-weight РАСТЁТ с p (гипотеза не подтверждена).")
else:
    best_p = min(P_LIST, key=lambda p: deltas[p])
    print(f"Нет монотонного тренда. Оптимальный p={best_p} "
          f"(δ={deltas[best_p]*100:+.2f}%).")

plt.show()
