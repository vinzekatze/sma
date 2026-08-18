"""
24 — PE-delay мягкий вес: multi-ticker тест.

Скрипт 23 показал: при p=20, m_pe=20, α=0.5 → −10% MAPE на SBER 1d,
но N=200 недостаточно для статистической значимости (Wilcoxon p=0.55).

Цель: агрегировать результаты по 8 тикерам (200 origins × 8 = 1600 точек)
и получить достаточно мощности для итогового вывода о PE-delay гипотезе.

Параметры: p=20, ξ=63, m_pe=20, α ∈ {0.0, 0.25, 0.50}, val_h=10.
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
TICKERS   = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL  = "1d"
MA_WINDOW = 1000

P       = 20
XI      = 3 * (P + 1)   # 63
M_PE    = P              # 20
ALPHAS  = [0.0, 0.25, 0.50]
_VAL_H  = 10
_PE_WIN = 50
_PE_ORD = 3
_N_EVAL = 200


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
def _eval_origin(X: np.ndarray, y: np.ndarray, vec: np.ndarray,
                 pe_dists: np.ndarray, ratio: np.ndarray,
                 val_origin: int, xi: int, alpha: float) -> float:
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

    r0     = float(ratio[val_origin])
    r_hat  = r0 + np.cumsum(hat)
    actual = ratio[val_origin + 1: val_origin + 1 + _VAL_H]
    n_ = min(len(r_hat), len(actual))
    if n_ == 0:
        return np.nan
    return float(np.mean(
        np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)
    ))


# ── per-ticker прогон ─────────────────────────────────────────────────────────
# RES[ticker][alpha] = np.ndarray of mape
RES: dict[str, dict[float, np.ndarray]] = {}

t_total = time.time()
for ticker in TICKERS:
    path = DATA_DIR / ticker / f"{INTERVAL}.json"
    with open(path) as f:
        candles = json.load(f)
    df     = normalize(candles, window=MA_WINDOW).dropna(subset=["ma"]).reset_index(drop=True)
    ratio  = df["ratio"].values
    dratio = np.diff(ratio)

    t1 = time.time()
    bar_pe = rolling_pe(dratio)

    min_orig = P + XI + _PE_WIN + P
    max_orig = len(dratio) - _VAL_H
    origins  = np.arange(max(min_orig, max_orig - _N_EVAL), max_orig)

    res_t = {a: [] for a in ALPHAS}

    for vo in origins:
        X, y  = build_delay_matrix(dratio[:vo], P)
        vec   = last_vector(dratio[:vo], P).copy()
        ps    = len(X)
        pd    = _pe_dists(bar_pe, ps, vo, M_PE)

        for a in ALPHAS:
            mape = _eval_origin(X, y, vec, pd, ratio, vo, XI, a)
            res_t[a].append(mape)

    for a in ALPHAS:
        res_t[a] = np.array(res_t[a])
    RES[ticker] = res_t

    bm  = float(np.nanmedian(res_t[0.0]))
    m50 = float(np.nanmedian(res_t[0.50]))
    print(f"{ticker:5s}  N={len(origins):3d}  "
          f"base_med={bm:.5f}  α=0.50_med={m50:.5f}  "
          f"Δ={( m50-bm)/bm*100:+.2f}%  "
          f"t={time.time()-t1:.1f}s")

print(f"\nВсего: {time.time()-t_total:.1f}s")


# ── агрегация ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"{'Тикер':>6}  {'N':>4}  {'base med':>9}  "
      f"{'α=0.25':>8}  {'Δ%':>7}  {'α=0.50':>8}  {'Δ%':>7}")
print("─" * 72)

all_base, all_025, all_050 = [], [], []

for ticker in TICKERS:
    r   = RES[ticker]
    bm  = float(np.nanmedian(r[0.0]))
    m25 = float(np.nanmedian(r[0.25]))
    m50 = float(np.nanmedian(r[0.50]))
    n   = int(np.sum(~np.isnan(r[0.0])))
    print(f"{ticker:>6}  {n:>4}  {bm:>9.5f}  "
          f"{m25:>8.5f}  {(m25-bm)/bm*100:>+6.2f}%  "
          f"{m50:>8.5f}  {(m50-bm)/bm*100:>+6.2f}%")
    all_base.append(r[0.0])
    all_025.append(r[0.25])
    all_050.append(r[0.50])

all_base = np.concatenate(all_base)
all_025  = np.concatenate(all_025)
all_050  = np.concatenate(all_050)

bm_all  = float(np.nanmedian(all_base))
m25_all = float(np.nanmedian(all_025))
m50_all = float(np.nanmedian(all_050))
n_all   = int(np.sum(~np.isnan(all_base)))

print("─" * 72)
print(f"{'ИТОГО':>6}  {n_all:>4}  {bm_all:>9.5f}  "
      f"{m25_all:>8.5f}  {(m25_all-bm_all)/bm_all*100:>+6.2f}%  "
      f"{m50_all:>8.5f}  {(m50_all-bm_all)/bm_all*100:>+6.2f}%")
print("=" * 72)


# ── Wilcoxon (агрегированный) ─────────────────────────────────────────────────
try:
    from scipy.stats import wilcoxon
    print("\n── Wilcoxon (aggregated, N≈1600) ──")
    for alpha, arr in [(0.25, all_025), (0.50, all_050)]:
        ok   = ~(np.isnan(all_base) | np.isnan(arr))
        diff = arr[ok] - all_base[ok]
        if ok.sum() < 10 or np.all(diff == 0):
            continue
        stat, pv = wilcoxon(diff)
        med_b = float(np.nanmedian(all_base))
        med_t = float(np.nanmedian(arr))
        sig = "✓ значимо" if pv < 0.05 else "— незначимо"
        print(f"  α={alpha:.2f}:  ΔMAPE={( med_t-med_b)/med_b*100:+.2f}%  "
              f"W={stat:.0f}  p={pv:.4f}  {sig}")
except ImportError:
    print("scipy не установлен, Wilcoxon пропущен")


# ── per-ticker boxplot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(
    f"PE-delay soft weight: multi-ticker  |  p={P}, m_pe={M_PE}, ξ={XI}  |  {INTERVAL}",
    fontsize=11,
)
COLORS_A = {0.0: "steelblue", 0.25: "seagreen", 0.50: "darkorange"}

for ax_i, ticker in enumerate(TICKERS):
    ax  = axes[ax_i // 4][ax_i % 4]
    r   = RES[ticker]
    bm  = float(np.nanmedian(r[0.0]))
    for a in ALPHAS:
        arr = np.sort(r[a][~np.isnan(r[a])])
        med = float(np.nanmedian(r[a]))
        lbl = f"α={a:.2f}  (med={med:.4f})"
        lw  = 2.5 if a == 0 else 1.8
        ax.plot(np.linspace(0, 100, len(arr)), arr,
                color=COLORS_A[a], linewidth=lw, label=lbl)
    ax.set_xlabel("Перцентиль")
    ax.set_ylabel("val_mape")
    ax.set_title(f"{ticker}  (base med={bm:.4f})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_ylim(0)

plt.tight_layout()
out_path = OUT_DIR / "24_pe_soft_multiticker.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")


# ── сводная кривая распределения (aggregated) ─────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.set_title(
    f"PE-delay soft weight: агрегированный CDF  |  N≈{n_all}  |  p={P}, m_pe={M_PE}",
    fontsize=11,
)
for a, arr, lbl in [
    (0.0,  all_base, f"baseline  (med={bm_all:.5f})"),
    (0.25, all_025,  f"α=0.25  (med={m25_all:.5f}  Δ={( m25_all-bm_all)/bm_all*100:+.2f}%)"),
    (0.50, all_050,  f"α=0.50  (med={m50_all:.5f}  Δ={( m50_all-bm_all)/bm_all*100:+.2f}%)"),
]:
    s = np.sort(arr[~np.isnan(arr)])
    ax2.plot(np.linspace(0, 100, len(s)), s,
             color=COLORS_A[a], linewidth=2, label=lbl)
ax2.set_xlabel("Перцентиль")
ax2.set_ylabel("val_mape")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_ylim(0)
plt.tight_layout()
out2 = OUT_DIR / "24_pe_soft_multiticker_cdf.png"
plt.savefig(out2, dpi=140, bbox_inches="tight")
print(f"CDF-график: {out2}")

plt.show()
print("\n── Готово ──")
