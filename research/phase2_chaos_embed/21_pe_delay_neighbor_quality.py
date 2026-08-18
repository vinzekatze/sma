"""
21 — PE-delay embedding: двухуровневый отбор соседей.

Гипотеза: вектор из последних p значений rolling PE («PE-траектория»)
позволяет находить эпохи с той же «активной подсистемой» точнее, чем
скалярная близость одного PE-значения (текущий pe_proximity).

Идея: delay-вектор x[k] = [dratio[k], ..., dratio[k+p-1]] и
PE-delay-вектор P[k] = [bar_pe[k], ..., bar_pe[k+p-1]] покрывают один
и тот же временной диапазон. Близость в PE-пространстве → похожая
«история сложности» → потенциально та же активная подсистема.

Walk-forward по N_EVAL origin-точкам SBER 1d.
Четыре конфигурации:
  0  baseline   — полный пул (без фильтра)
  1  pe_prox    — |PE_i − PE_q| ≤ DELTA_MAX (скаляр, текущий подход)
  2  pe_delay_a — L2 в PE-пространстве задержек, k_pre=K_A (tight)
  3  pe_delay_b — то же, k_pre=K_B (loose)

Метрики на каждом origin:
  d_k      — расст. до ξ-го соседа в x-пространстве (в пределах маски)
  val_mape — MAPE за VAL_H баров валидации (dratio → ratio → MAPE)
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

_P      = 5              # embedding dimension
_XI     = 3 * (_P + 1)  # 18 neighbours
_VAL_H  = 10             # validation horizon (bars)

_PE_WIN  = 50            # rolling PE window
_PE_ORD  = 3             # PE pattern order

_DELTA   = 0.05          # pe_prox: max |ΔPE|
_K_A     = 10 * _XI      # 180 — tight pre-filter
_K_B     = 30 * _XI      # 540 — loose pre-filter

_N_EVAL  = 200           # walk-forward points

CFGS   = ["baseline", "pe_prox", "pe_delay_a", "pe_delay_b"]
LABELS = {
    "baseline":   "Без фильтра",
    "pe_prox":    f"PE-скаляр (±{_DELTA})",
    "pe_delay_a": f"PE-delay tight (k={_K_A})",
    "pe_delay_b": f"PE-delay loose (k={_K_B})",
}
COLORS = {
    "baseline":   "steelblue",
    "pe_prox":    "darkorange",
    "pe_delay_a": "seagreen",
    "pe_delay_b": "mediumpurple",
}


# ── Permutation Entropy ───────────────────────────────────────────────────────
_PIDX = {perm: i for i, perm in enumerate(permutations(range(_PE_ORD)))}


def _pe(x: np.ndarray) -> float:
    n = len(x)
    if n < _PE_ORD:
        return np.nan
    counts = np.zeros(factorial(_PE_ORD))
    for i in range(n - _PE_ORD + 1):
        counts[_PIDX[tuple(np.argsort(x[i:i + _PE_ORD]))]] += 1
    p = counts[counts > 0]
    p /= p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(factorial(_PE_ORD)))


def rolling_pe(series: np.ndarray, win: int = _PE_WIN) -> np.ndarray:
    out = np.full(len(series), np.nan)
    for i in range(win - 1, len(series)):
        out[i] = _pe(series[i - win + 1: i + 1])
    return out


# ── маски пула ────────────────────────────────────────────────────────────────

def _mask_prox(bar_pe: np.ndarray, pool_size: int, val_origin: int) -> np.ndarray | None:
    """Скалярная PE-близость к val_origin (аналог pe_proximity_mask)."""
    pe_q = bar_pe[val_origin - 1] if val_origin > 0 else np.nan
    if np.isnan(pe_q):
        return None
    centers = np.minimum(np.arange(pool_size) + _P // 2, len(bar_pe) - 1)
    pool_pe = bar_pe[centers]
    mask = ~np.isnan(pool_pe) & (np.abs(pool_pe - pe_q) <= _DELTA)
    return mask if mask.sum() >= _XI else None


def _mask_pe_delay(bar_pe: np.ndarray, pool_size: int, val_origin: int,
                   k_pre: int) -> np.ndarray | None:
    """
    Двухэтапный: сначала L2-близость в PE-пространстве задержек,
    затем стандартный L2 в x-пространстве внутри отобранных.

    P_pool[k] = bar_pe[k : k+_P]   — PE вдоль того же окна, что X[k]
    P_q       = bar_pe[val_origin-_P : val_origin]
    """
    q_start = val_origin - _P
    if q_start < 0:
        return None
    P_q = bar_pe[q_start: q_start + _P]
    if np.any(np.isnan(P_q)):
        return None
    if pool_size + _P - 1 >= len(bar_pe):
        return None

    # строим PE-матрицу пула: (pool_size, _P)
    idx   = np.arange(pool_size)[:, None] + np.arange(_P)[None, :]
    P_pl  = bar_pe[idx]  # (pool_size, _P)
    valid = ~np.any(np.isnan(P_pl), axis=1)
    if valid.sum() < _XI:
        return None

    pe_dists = np.full(pool_size, np.inf)
    pe_dists[valid] = np.linalg.norm(P_pl[valid] - P_q, axis=1)

    k_eff = min(k_pre, int(valid.sum()))
    thr   = np.sort(pe_dists[valid])[k_eff - 1]
    mask  = (pe_dists <= thr + 1e-12) & valid
    return mask if mask.sum() >= _XI else None


# ── одна LA1-оценка ──────────────────────────────────────────────────────────

def _eval(X: np.ndarray, y: np.ndarray, vec: np.ndarray,
          ratio: np.ndarray, val_origin: int,
          mask: np.ndarray | None) -> tuple[float, float]:
    """
    Прогноз _VAL_H шагов из val_origin по маскированному пулу.
    Возвращает (val_mape, d_k).
    d_k — расстояние до ξ-го соседа в пределах применённой маски.
    """
    n = len(X)
    if mask is not None and mask[:n].sum() >= _XI:
        m = mask[:n]
        Xs, ys = X[m], y[m]
    else:
        Xs, ys = X, y

    if len(Xs) < _XI:
        Xs, ys = X, y  # fallback

    d   = np.linalg.norm(Xs - vec, axis=1)
    d_k = float(np.sort(d)[_XI - 1])

    v   = vec.copy()
    hat = np.empty(_VAL_H)
    for h in range(_VAL_H):
        dd  = np.linalg.norm(Xs - v, axis=1)
        idx = np.argsort(dd)[:_XI]
        A   = np.hstack([np.ones((_XI, 1)), Xs[idx]])
        c, _, _, _ = np.linalg.lstsq(A, ys[idx], rcond=None)
        hat[h] = float(c[0] + v @ c[1:])
        v = np.roll(v, -1)
        v[-1] = hat[h]

    r0     = float(ratio[val_origin])
    r_hat  = r0 + np.cumsum(hat)
    actual = ratio[val_origin + 1: val_origin + 1 + _VAL_H]
    n_     = min(len(r_hat), len(actual))
    if n_ == 0:
        return np.nan, d_k
    mape = float(np.mean(np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)))
    return mape, d_k


# ── данные ────────────────────────────────────────────────────────────────────
with open(DATA_DIR / TICKER / f"{INTERVAL}.json") as f:
    candles = json.load(f)

df     = normalize(candles, window=MA_WINDOW).dropna(subset=["ma"]).reset_index(drop=True)
ratio  = df["ratio"].values
dratio = np.diff(ratio)

print(f"{TICKER} {INTERVAL}: {len(dratio)} баров Δratio")

t0 = time.time()
print(f"Rolling PE (W={_PE_WIN}, order={_PE_ORD})…", end=" ", flush=True)
bar_pe = rolling_pe(dratio)
print(f"готово ({time.time()-t0:.1f}s)")

valid_pe = bar_pe[~np.isnan(bar_pe)]
print(f"PE: min={valid_pe.min():.4f}  q25={np.quantile(valid_pe, .25):.4f}  "
      f"med={np.median(valid_pe):.4f}  q75={np.quantile(valid_pe, .75):.4f}  "
      f"max={valid_pe.max():.4f}")

# диапазон origins
min_orig = _P + _XI + _PE_WIN + _P
max_orig = len(dratio) - _VAL_H
origins  = np.arange(max(min_orig, max_orig - _N_EVAL), max_orig)
print(f"Walk-forward: {len(origins)} origins  [{origins[0]}…{origins[-1]}]\n")


# ── walk-forward ──────────────────────────────────────────────────────────────
RES = {c: {"mape": [], "dk": [], "fb": 0} for c in CFGS}
t1  = time.time()

for i, vo in enumerate(origins):
    X, y = build_delay_matrix(dratio[:vo], _P)
    vec  = last_vector(dratio[:vo], _P).copy()
    ps   = len(X)  # pool_size

    masks = {
        "baseline":   None,
        "pe_prox":    _mask_prox(bar_pe, ps, vo),
        "pe_delay_a": _mask_pe_delay(bar_pe, ps, vo, _K_A),
        "pe_delay_b": _mask_pe_delay(bar_pe, ps, vo, _K_B),
    }

    for cfg in CFGS:
        m = masks[cfg]
        if m is None and cfg != "baseline":
            RES[cfg]["fb"] += 1
        mape, dk = _eval(X, y, vec, ratio, vo, mask=m)
        RES[cfg]["mape"].append(mape)
        RES[cfg]["dk"].append(dk)

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(origins)}  ({time.time()-t1:.0f}s elapsed)…")

for cfg in CFGS:
    RES[cfg]["mape"] = np.array(RES[cfg]["mape"])
    RES[cfg]["dk"]   = np.array(RES[cfg]["dk"])

print(f"Walk-forward done: {time.time()-t1:.1f}s\n")


# ── таблица ───────────────────────────────────────────────────────────────────
print("=" * 88)
print(f"{'':26}  {'MAPE mean':>10}  {'MAPE med':>9}  {'d_k mean':>9}  {'d_k med':>8}  "
      f"{'fallback':>8}  {'vs base':>9}")
print("─" * 88)

bm = float(np.nanmedian(RES["baseline"]["mape"]))
for cfg in CFGS:
    mm  = float(np.nanmean(RES[cfg]["mape"]))
    med = float(np.nanmedian(RES[cfg]["mape"]))
    dm  = float(np.nanmean(RES[cfg]["dk"]))
    dd  = float(np.nanmedian(RES[cfg]["dk"]))
    fb  = RES[cfg]["fb"]
    tag = "baseline" if cfg == "baseline" else (
        f"{'↓' if med < bm else '↑'}{abs(med - bm) / bm * 100:.2f}%"
    )
    print(f"{LABELS[cfg]:<26}  {mm:>10.6f}  {med:>9.6f}  {dm:>9.5f}  "
          f"{dd:>8.5f}  {fb:>8}  {tag:>9}")

print("=" * 88)

# процент случаев, когда каждый фильтр лучше baseline
print("\n── % origins где фильтр лучше baseline ──")
bm_arr = RES["baseline"]["mape"]
for cfg in CFGS[1:]:
    arr   = RES[cfg]["mape"]
    ok    = ~(np.isnan(bm_arr) | np.isnan(arr))
    better = np.sum(arr[ok] < bm_arr[ok])
    print(f"  {LABELS[cfg]:<26}: {better}/{ok.sum()} = {100*better/max(ok.sum(),1):.1f}%")

# Wilcoxon
try:
    from scipy.stats import wilcoxon
    print("\n── Wilcoxon signed-rank (val_mape, vs baseline) ──")
    for cfg in CFGS[1:]:
        a = RES["baseline"]["mape"]
        b = RES[cfg]["mape"]
        ok = ~(np.isnan(a) | np.isnan(b))
        diff = b[ok] - a[ok]
        if ok.sum() < 10 or np.all(diff == 0):
            print(f"  {LABELS[cfg]}: мало данных")
            continue
        stat, p = wilcoxon(diff)
        print(f"  {LABELS[cfg]:<26}: W={stat:.0f}  p={p:.4f}  "
              f"{'✓ значимо p<0.05' if p < 0.05 else '— незначимо'}")
except ImportError:
    print("scipy не установлен — тест пропущен")


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"PE-delay embedding: качество соседей и точность LA1  |  {TICKER} {INTERVAL}  "
    f"(p={_P}, ξ={_XI}, val_h={_VAL_H})",
    fontsize=11,
)

# A: boxplot val_mape
ax = axes[0]
data_mape = [RES[c]["mape"][~np.isnan(RES[c]["mape"])] for c in CFGS]
bp = ax.boxplot(data_mape, patch_artist=True, medianprops=dict(color="white", linewidth=2))
for patch, cfg in zip(bp["boxes"], CFGS):
    patch.set_facecolor(COLORS[cfg])
    patch.set_alpha(0.75)
ax.set_xticklabels([LABELS[c].replace(" (", "\n(") for c in CFGS], fontsize=7)
ax.set_ylabel("val_mape")
ax.set_title("A. MAPE на валидации")
ax.set_ylim(0)
ax.grid(axis="y", alpha=0.3)

# B: boxplot d_k
ax = axes[1]
data_dk = [RES[c]["dk"][~np.isnan(RES[c]["dk"])] for c in CFGS]
bp2 = ax.boxplot(data_dk, patch_artist=True, medianprops=dict(color="white", linewidth=2))
for patch, cfg in zip(bp2["boxes"], CFGS):
    patch.set_facecolor(COLORS[cfg])
    patch.set_alpha(0.75)
ax.set_xticklabels([LABELS[c].replace(" (", "\n(") for c in CFGS], fontsize=7)
ax.set_ylabel("d_k (в x-пространстве)")
ax.set_title("B. Расстояние до ξ-го соседа")
ax.set_ylim(0)
ax.grid(axis="y", alpha=0.3)

# C: PCA(PE-delay) — структура пула (весь исторический пул)
ax = axes[2]
_ps_demo = len(dratio) - _VAL_H - _P
if _ps_demo > 0 and _ps_demo + _P <= len(bar_pe):
    _idx = np.arange(_ps_demo)[:, None] + np.arange(_P)[None, :]
    _Ppl = bar_pe[_idx]
    _valid = ~np.any(np.isnan(_Ppl), axis=1)
    _Pv   = _Ppl[_valid]
    _Pc   = _Pv - _Pv.mean(axis=0)
    _cov  = _Pc.T @ _Pc / max(len(_Pc) - 1, 1)
    _vals, _vecs = np.linalg.eigh(_cov)
    _V2   = _vecs[:, ::-1][:, :2]  # top-2 компоненты
    _P2d  = _Pc @ _V2
    _pe_sc = bar_pe[np.arange(_ps_demo)[_valid] + _P // 2]
    sc = ax.scatter(_P2d[:, 0], _P2d[:, 1],
                    c=_pe_sc, cmap="RdYlGn_r",
                    s=3, alpha=0.35, rasterized=True)
    plt.colorbar(sc, ax=ax, label="PE скаляр")
    _exp = _vals[::-1][:2] / _vals.sum() * 100
    ax.set_title(f"C. PCA(PE-delay) пул  ({_exp[0]:.0f}%+{_exp[1]:.0f}% дисп.)")
else:
    ax.text(0.5, 0.5, "нет данных", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("C. PCA(PE-delay)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.grid(alpha=0.2)

plt.tight_layout()
out_path = OUT_DIR / "21_pe_delay_neighbor_quality.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")


# ── итог ──────────────────────────────────────────────────────────────────────
print("\n── Итог ──")
bm = float(np.nanmedian(RES["baseline"]["mape"]))

best = min(
    CFGS[1:],
    key=lambda c: float(np.nanmedian(RES[c]["mape"][~np.isnan(RES[c]["mape"])])),
)
bv  = float(np.nanmedian(RES[best]["mape"][~np.isnan(RES[best]["mape"])]))
imp = (bm - bv) / bm * 100

print(f"baseline median MAPE : {bm:.5f}")
for cfg in CFGS[1:]:
    v = float(np.nanmedian(RES[cfg]["mape"][~np.isnan(RES[cfg]["mape"])]))
    print(f"{LABELS[cfg]:<26}: {v:.5f}  ({(bm-v)/bm*100:+.2f}%)")

print()
if imp > 1.5:
    print(f"Лучший: {LABELS[best]}  ({imp:+.1f}% к baseline)")
    print("→ Гипотеза ПОДТВЕРЖДЕНА: PE-delay embedding улучшает точность LA1.")
    print("  Следующий шаг: перебор k_pre, m_pe≠p, multi-ticker тест.")
elif imp > 0:
    print(f"Лучший: {LABELS[best]}  ({imp:+.1f}% к baseline)")
    print("→ Небольшое улучшение. Необходим более широкий тест (больше тикеров/горизонтов).")
else:
    print(f"PE-delay не улучшает MAPE при текущих параметрах.")
    print(f"Лучший из фильтров: {LABELS[best]}  ({imp:+.1f}%)")
    print("→ Возможные причины: p слишком мал (p=5), PE_WIN не оптимален,")
    print("  или нужны другие k_pre. Рекомендуется перебор параметров.")

plt.show()
