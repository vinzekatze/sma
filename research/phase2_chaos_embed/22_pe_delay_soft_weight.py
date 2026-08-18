"""
22 — PE-delay как мягкий вес соседей (soft kernel).

Скрипт 21 показал: жёсткий topk-фильтр по PE-delay при p=5 ухудшает MAPE,
потому что выбрасывает геометрически близких соседей.

Новый подход: не отбрасывать соседей, а ВЗВЕШИВАТЬ их по PE-близости.
Ξ соседей отбираются стандартно (по d_x). Их LWR-вес множится на PE-вес:

    w_combined[i] = w_lwr[i] * w_pe[i]
    w_lwr[i]  = exp(-0.5 * (d_x[i] / h_x)²)      h_x = max(d_x among Ξ)
    w_pe[i]   = exp(-α * (d_pe[i] / h_pe)²)       h_pe = mean(d_pe among Ξ)

Переменные:
    m_pe  ∈ {5, 20}          — длина PE-delay вектора
    α     ∈ {0.25, 0.5, 1.0, 2.0}

Итого 8 конфигов + baseline (α=0) = 9 конфигов.
Walk-forward: 200 origins, SBER 1d, p=5, ξ=18, val_h=10.
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

_P      = 5
_XI     = 3 * (_P + 1)   # 18
_VAL_H  = 10

_PE_WIN  = 50
_PE_ORD  = 3

_N_EVAL  = 200

M_PE_LIST = [5, 20]
ALPHA_LIST = [0.25, 0.5, 1.0, 2.0]


# ── Permutation Entropy ───────────────────────────────────────────────────────
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


# ── PE-delay расстояния для всего пула ───────────────────────────────────────

def _pe_dists_for_pool(bar_pe: np.ndarray, pool_size: int,
                       val_origin: int, m_pe: int) -> np.ndarray:
    """
    d_pe[k] = ||bar_pe[k:k+m_pe] − bar_pe[val_origin-m_pe:val_origin]||

    Строки с NaN в PE-векторе → d_pe = 0 (нет штрафа, эквивалент query).
    """
    q_start = val_origin - m_pe
    if q_start < 0:
        return np.zeros(pool_size)

    P_q = bar_pe[q_start: q_start + m_pe]
    if np.any(np.isnan(P_q)):
        return np.zeros(pool_size)

    # нужны индексы до pool_size-1 + m_pe - 1 включительно
    if pool_size + m_pe - 1 >= len(bar_pe):
        return np.zeros(pool_size)

    idx  = np.arange(pool_size)[:, None] + np.arange(m_pe)[None, :]  # (pool_size, m_pe)
    P_pl = bar_pe[idx]  # (pool_size, m_pe)
    valid = ~np.any(np.isnan(P_pl), axis=1)
    dists = np.zeros(pool_size)
    dists[valid] = np.linalg.norm(P_pl[valid] - P_q, axis=1)
    return dists


# ── LWR + PE-вес (одна origin-точка) ─────────────────────────────────────────

def _eval_lwr_pe(X: np.ndarray, y: np.ndarray, vec: np.ndarray,
                 pe_dists: np.ndarray,
                 ratio: np.ndarray, val_origin: int,
                 alpha: float) -> tuple[float, float]:
    """
    LWR с мультипликативным PE-весом.
    alpha=0 → стандартный LWR (baseline).
    Возвращает (val_mape, d_k_x).
    """
    v   = vec.copy()
    hat = np.empty(_VAL_H)

    for h in range(_VAL_H):
        dists  = np.linalg.norm(X - v, axis=1)
        nn_idx = np.argpartition(dists, _XI)[:_XI]
        nn_dx  = dists[nn_idx]

        h_bw = max(float(nn_dx.max()), 1e-10)
        w_lwr = np.exp(-0.5 * (nn_dx / h_bw) ** 2)

        if alpha > 0:
            nn_dpe = pe_dists[nn_idx]
            h_pe   = max(float(nn_dpe.mean()), 1e-10)
            w_pe   = np.exp(-alpha * (nn_dpe / h_pe) ** 2)
            w      = w_lwr * w_pe
        else:
            w = w_lwr

        A  = np.hstack([np.ones((_XI, 1)), X[nn_idx]])
        sw = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx], rcond=None)
        hat[h] = float(c[0] + v @ c[1:])
        v = np.roll(v, -1); v[-1] = hat[h]

    # d_k при первом шаге (для метрики качества соседей)
    d0  = np.linalg.norm(X - vec, axis=1)
    d_k = float(np.sort(d0)[_XI - 1])

    r0     = float(ratio[val_origin])
    r_hat  = r0 + np.cumsum(hat)
    actual = ratio[val_origin + 1: val_origin + 1 + _VAL_H]
    n_     = min(len(r_hat), len(actual))
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
print(f"Rolling PE (W={_PE_WIN})…", end=" ", flush=True)
bar_pe = rolling_pe(dratio)
print(f"готово ({time.time()-t0:.1f}s)\n")

min_orig = _P + _XI + _PE_WIN + max(M_PE_LIST)
max_orig = len(dratio) - _VAL_H
origins  = np.arange(max(min_orig, max_orig - _N_EVAL), max_orig)
print(f"Walk-forward: {len(origins)} origins  [{origins[0]}…{origins[-1]}]")

# ── конфигурации ─────────────────────────────────────────────────────────────
# (m_pe, alpha) — alpha=0 это baseline LWR
CONFIGS: list[tuple[int, float]] = [(0, 0.0)]   # baseline первым
for m in M_PE_LIST:
    for a in ALPHA_LIST:
        CONFIGS.append((m, a))

def cfg_label(m, a):
    if a == 0:
        return "baseline (α=0)"
    return f"m_pe={m:2d}  α={a:.2f}"


# ── walk-forward ──────────────────────────────────────────────────────────────
# Кэшируем PE-расстояния для каждой origin и каждого m_pe
# RES[cfg_idx] = {"mape": [], "dk": []}
RES = [{"mape": [], "dk": []} for _ in CONFIGS]

t1 = time.time()
for i, vo in enumerate(origins):
    X, y = build_delay_matrix(dratio[:vo], _P)
    vec  = last_vector(dratio[:vo], _P).copy()
    ps   = len(X)

    # PE-дистанции: вычислить один раз на origin × m_pe
    pe_d_cache: dict[int, np.ndarray] = {}
    for m in M_PE_LIST:
        pe_d_cache[m] = _pe_dists_for_pool(bar_pe, ps, vo, m)

    for ci, (m, a) in enumerate(CONFIGS):
        pd = pe_d_cache.get(m, np.zeros(ps))
        mape, dk = _eval_lwr_pe(X, y, vec, pd, ratio, vo, alpha=a)
        RES[ci]["mape"].append(mape)
        RES[ci]["dk"].append(dk)

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(origins)}  ({time.time()-t1:.0f}s)…")

for ci in range(len(CONFIGS)):
    RES[ci]["mape"] = np.array(RES[ci]["mape"])
    RES[ci]["dk"]   = np.array(RES[ci]["dk"])

print(f"Walk-forward: {time.time()-t1:.1f}s\n")


# ── таблица ───────────────────────────────────────────────────────────────────
bm = float(np.nanmedian(RES[0]["mape"]))
print("=" * 78)
print(f"{'Конфиг':<22}  {'MAPE mean':>10}  {'MAPE med':>9}  {'vs base':>9}  {'d_k med':>8}")
print("─" * 78)
for ci, (m, a) in enumerate(CONFIGS):
    mm  = float(np.nanmean(RES[ci]["mape"]))
    med = float(np.nanmedian(RES[ci]["mape"]))
    dd  = float(np.nanmedian(RES[ci]["dk"]))
    tag = "baseline" if a == 0 else f"{(med-bm)/bm*100:+.2f}%"
    print(f"{cfg_label(m,a):<22}  {mm:>10.6f}  {med:>9.6f}  {tag:>9}  {dd:>8.5f}")
print("=" * 78)

# лучший конфиг
best_ci = int(np.argmin([float(np.nanmedian(RES[ci]["mape"])) for ci in range(len(CONFIGS))]))
best_m  = float(np.nanmedian(RES[best_ci]["mape"]))
print(f"\nЛучший:  {cfg_label(*CONFIGS[best_ci])}  "
      f"MAPE med={best_m:.6f}  ({(best_m-bm)/bm*100:+.2f}% vs baseline)")

# Wilcoxon только для лучшего
try:
    from scipy.stats import wilcoxon
    b_arr = RES[0]["mape"]
    t_arr = RES[best_ci]["mape"]
    ok = ~(np.isnan(b_arr) | np.isnan(t_arr))
    diff = t_arr[ok] - b_arr[ok]
    if ok.sum() >= 10 and not np.all(diff == 0):
        stat, p = wilcoxon(diff)
        print(f"Wilcoxon (лучший vs baseline): W={stat:.0f}  p={p:.4f}  "
              f"{'✓ значимо' if p < 0.05 else '— незначимо'}")
except ImportError:
    pass


# ── графики ───────────────────────────────────────────────────────────────────
# Две тепловые карты: MAPE med и ΔPE_median для каждой (m_pe, α)
# (без baseline-строки)

def _grid(m_pe_list, alpha_list, metric="mape"):
    mat = np.full((len(m_pe_list), len(alpha_list)), np.nan)
    for ri, m in enumerate(m_pe_list):
        for ci_a, a in enumerate(alpha_list):
            ci = CONFIGS.index((m, a))
            v = RES[ci][metric]
            mat[ri, ci_a] = float(np.nanmedian(v))
    return mat

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle(
    f"PE-delay мягкий вес: {TICKER} {INTERVAL}  (p={_P}, ξ={_XI}, val_h={_VAL_H})",
    fontsize=12,
)

# A: heatmap MAPE med
ax = axes[0]
mat = _grid(M_PE_LIST, ALPHA_LIST, "mape")
im  = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r")
plt.colorbar(im, ax=ax, label="MAPE median")
ax.set_xticks(range(len(ALPHA_LIST))); ax.set_xticklabels(ALPHA_LIST)
ax.set_yticks(range(len(M_PE_LIST)));  ax.set_yticklabels(M_PE_LIST)
ax.set_xlabel("α")
ax.set_ylabel("m_pe")
ax.set_title("A. MAPE median")
# annotate cells
for ri in range(len(M_PE_LIST)):
    for ci_a in range(len(ALPHA_LIST)):
        ax.text(ci_a, ri, f"{mat[ri, ci_a]:.5f}", ha="center", va="center",
                fontsize=7, color="black")
ax.axhline(-0.5 + 0.5, color="red", linewidth=0)
# baseline line
ax.set_title(f"A. MAPE median  (base={bm:.5f})")

# B: heatmap ΔMAPE vs baseline
ax = axes[1]
mat_d = (mat - bm) / bm * 100
im2   = ax.imshow(mat_d, aspect="auto", cmap="RdYlGn_r", vmin=-5, vmax=5)
plt.colorbar(im2, ax=ax, label="Δ MAPE vs baseline, %")
ax.set_xticks(range(len(ALPHA_LIST))); ax.set_xticklabels(ALPHA_LIST)
ax.set_yticks(range(len(M_PE_LIST)));  ax.set_yticklabels(M_PE_LIST)
ax.set_xlabel("α")
ax.set_ylabel("m_pe")
ax.set_title("B. Δ MAPE vs baseline, %")
for ri in range(len(M_PE_LIST)):
    for ci_a in range(len(ALPHA_LIST)):
        v = mat_d[ri, ci_a]
        ax.text(ci_a, ri, f"{v:+.2f}%", ha="center", va="center",
                fontsize=7, color="black")

# C: MAPE распределение для 3 кривых (baseline, лучший, худший)
ax = axes[2]
worst_ci = int(np.argmax([float(np.nanmedian(RES[ci]["mape"])) for ci in range(len(CONFIGS))]))
for ci, color, lw in [(0, "steelblue", 2.5),
                      (best_ci, "seagreen", 2.5),
                      (worst_ci, "tomato", 1.5)]:
    arr = np.sort(RES[ci]["mape"][~np.isnan(RES[ci]["mape"])])
    ax.plot(np.linspace(0, 100, len(arr)), arr,
            color=color, linewidth=lw, label=cfg_label(*CONFIGS[ci]))
ax.set_xlabel("Перцентиль")
ax.set_ylabel("val_mape")
ax.set_title("C. ECDF val_mape (baseline / лучший / худший)")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "22_pe_delay_soft_weight.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")


# ── итог ──────────────────────────────────────────────────────────────────────
print("\n── Итог ──")
print(f"baseline MAPE med : {bm:.5f}")
print(f"Лучший конфиг     : {cfg_label(*CONFIGS[best_ci])}  "
      f"({(best_m-bm)/bm*100:+.2f}%)")
worst_m = float(np.nanmedian(RES[worst_ci]["mape"]))
print(f"Худший конфиг     : {cfg_label(*CONFIGS[worst_ci])}  "
      f"({(worst_m-bm)/bm*100:+.2f}%)")

imp = (bm - best_m) / bm * 100
if imp > 1.5:
    print("\n→ Мягкий PE-вес УЛУЧШАЕТ точность LA.")
    print(f"  Оптимальные параметры: m_pe={CONFIGS[best_ci][0]}, α={CONFIGS[best_ci][1]}")
    print("  Следующий шаг: тест при больших p (p=20, 40) и других тикерах.")
elif imp > 0:
    print(f"\n→ Небольшое улучшение ({imp:.2f}%). Нужен тест на большей выборке.")
else:
    print(f"\n→ Мягкий PE-вес не помогает при p={_P}.")
    print("  Возможные причины:")
    print("  1. p=5 слишком мал — PE-delay вектор автокоррелирован, нет сигнала")
    print("  2. PE при MA_WIN=1000 не различает подсистемы хорошо")
    print("  3. Нужен другой признак режима (wavelet energy, IMF amplitude)")
    print("  Следующий шаг: тест при больших p (p=40, 60) или другом признаке режима.")

plt.show()
