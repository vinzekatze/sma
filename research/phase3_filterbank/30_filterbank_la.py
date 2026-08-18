"""
30 — Filter bank + LA.

Идея та же, что в EMD: разложить dratio на частотные компоненты, применить LA
к каждой отдельно, сложить прогнозы. Но вместо EMD — каузальный фильтр-банк.

Ключевое преимущество перед EMD:
  sosfilt (IIR, каузальный) — строго time-correct: comp[k] зависит только от
  dratio[:k]. Нет boundary effects, нет look-ahead, нет пересчёта при добавлении
  новых точек. Фильтр применяется один раз к полному доступному ряду.

Полосы (octave-spaced, по мотивам IMF-анализа из скр.28):
  C0: freq > 0.250             (период < 4 бара)   — высокочастотный шум
  C1: 0.125 < f < 0.250        (период  4–8 баров)
  C2: 0.0625 < f < 0.125       (период  8–16 баров)
  C3: 0.03125 < f < 0.0625     (период 16–32 бара)
  C4: 0.015625 < f < 0.03125   (период 32–64 бара)
  C5: f < 0.015625             (период > 64 баров) — низкочастотный тренд

Конфиги:
  baseline     — LWR на исходном dratio
  fb_all       — LWR на каждой Ci, сумма прогнозов
  fb_skip_c0   — то же, без C0 (аналог emd_skip1)
  fb_lo_only   — только C3+C4+C5 (медленные компоненты)
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
    from scipy.signal import butter, sosfilt
    from scipy.stats import wilcoxon as _wlcx
    _HAS_SCI = True
except ImportError:
    print("pip install scipy"); sys.exit(1)

# ── параметры ─────────────────────────────────────────────────────────────────
TICKER    = "SBER"
INTERVAL  = "1d"
MA_WINDOW = 1000
P         = 5
XI        = 3 * (P + 1)   # 18
VAL_H     = 10
N_EVAL    = 200

FILTER_ORDER = 4
# Граничные частоты: каждый раз делим пополам (октавный банк)
CUTOFFS = [0.25, 0.125, 0.0625, 0.03125, 0.015625]


# ── filter bank ───────────────────────────────────────────────────────────────
def make_filter_bank(series: np.ndarray, cutoffs: list[float],
                     order: int = 4) -> np.ndarray:
    """
    Каузальное октавное разложение. Возвращает (n_bands, len(series)).
    Последняя полоса — низкочастотный остаток.
    sosfilt — строго каузальный (только прошлое).
    """
    components = []
    remaining  = series.copy()
    for fc in cutoffs:
        sos  = butter(order, fc, btype="low", output="sos")
        low  = sosfilt(sos, remaining)
        high = remaining - low
        components.append(high)
        remaining = low
    components.append(remaining)
    return np.array(components)   # (len(cutoffs)+1, len(series))


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

# Фильтр-банк применяется ОДИН РАЗ на полном ряду (time-correct: comp[k] = f(dratio[:k]))
t0   = time.time()
COMP = make_filter_bank(dratio, CUTOFFS, FILTER_ORDER)   # (6, N)
n_bands = COMP.shape[0]
print(f"Filter bank: {n_bands} полос  t={time.time()-t0:.3f}s")

# Информация по полосам
recon_err = float(np.max(np.abs(COMP.sum(axis=0) - dratio)))
print(f"Реконструкция: max_err={recon_err:.2e}\n")

band_labels = [f"C{i}" for i in range(n_bands)]
band_labels[-1] += " (residue)"
print(f"{'Полоса':<15}  {'std':>10}  {'energy%':>8}  {'период'}")
print("─" * 50)
total_e = float(np.sum(COMP ** 2))
for i, (comp, lbl) in enumerate(zip(COMP, band_labels)):
    en  = float(np.sum(comp ** 2)) / total_e * 100
    # оценка периода через zero-crossing
    zc  = np.sum(np.diff(np.sign(comp)) != 0)
    T   = f"~{int(2*len(comp)/(zc+1))} баров" if zc > 0 else "—"
    print(f"{lbl:<15}  {comp.std():>10.6f}  {en:>7.2f}%  {T}")
print()

# Группы для конфигов
idx_lo = [3, 4, 5]   # C3, C4, C5 — медленные


# ── walk-forward ──────────────────────────────────────────────────────────────
min_orig = P + XI + 5
max_orig = len(dratio) - VAL_H
origins  = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)
print(f"Walk-forward: origins={len(origins)}")

mape_res = {c: [] for c in ["baseline", "fb_all", "fb_skip_c0", "fb_lo_only"]}

t1 = time.time()
for vo in origins:
    # baseline
    X, y  = build_delay_matrix(dratio[:vo], P)
    vec   = last_vector(dratio[:vo], P).copy()
    mape_res["baseline"].append(_mape(_lwr_forecast(X, y, vec, XI), ratio, vo))

    # LWR на каждой полосе до vo
    hats = []
    for comp in COMP:
        Xc, yc = build_delay_matrix(comp[:vo], P)
        if len(Xc) < XI:
            hats.append(np.zeros(VAL_H))
        else:
            vecc = last_vector(comp[:vo], P).copy()
            hats.append(_lwr_forecast(Xc, yc, vecc, XI))

    mape_res["fb_all"].append(
        _mape(np.sum(hats, axis=0), ratio, vo))
    mape_res["fb_skip_c0"].append(
        _mape(np.sum(hats[1:], axis=0), ratio, vo))
    mape_res["fb_lo_only"].append(
        _mape(np.sum([hats[i] for i in idx_lo], axis=0), ratio, vo))

for c in mape_res:
    mape_res[c] = np.array(mape_res[c])

print(f"t={time.time()-t1:.1f}s\n")


# ── таблица ───────────────────────────────────────────────────────────────────
bm = float(np.nanmedian(mape_res["baseline"]))
print("=" * 62)
print(f"baseline_med = {bm:.5f}  (N={len(origins)})")
print(f"\n{'Конфиг':<14}  {'mean':>9}  {'median':>9}  {'Δ%':>7}  {'W-p':>8}")
print("─" * 55)
for c in ["baseline", "fb_all", "fb_skip_c0", "fb_lo_only"]:
    arr = mape_res[c]
    mn  = float(np.nanmean(arr))
    med = float(np.nanmedian(arr))
    dp  = (med - bm) / bm * 100
    wp  = "—"
    if c != "baseline":
        ok   = ~(np.isnan(mape_res["baseline"]) | np.isnan(arr))
        diff = arr[ok] - mape_res["baseline"][ok]
        if ok.sum() >= 10 and not np.all(diff == 0):
            _, pv = _wlcx(diff)
            wp = f"{pv:.4f}"
    tag = "baseline" if c == "baseline" else f"{dp:>+6.2f}%"
    print(f"{c:<14}  {mn:>9.5f}  {med:>9.5f}  {tag:>7}  {wp:>8}")
print("=" * 62)


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"Filter bank + LA  |  {TICKER} {INTERVAL}  "
    f"order={FILTER_ORDER}  p={P}, ξ={XI}, N={N_EVAL}",
    fontsize=11,
)

# CDF
ax = axes[0]
COLORS = {"baseline": "steelblue", "fb_all": "tomato",
          "fb_skip_c0": "seagreen", "fb_lo_only": "darkorange"}
for c, arr in mape_res.items():
    s   = np.sort(arr[~np.isnan(arr)])
    med = float(np.nanmedian(arr))
    lw  = 2.5 if c == "baseline" else 1.8
    ax.plot(np.linspace(0, 100, len(s)), s,
            color=COLORS[c], lw=lw, label=f"{c}  (med={med:.4f})")
ax.set_title("CDF val_mape"); ax.set_xlabel("Перцентиль"); ax.set_ylabel("val_mape")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0)

# Полосы: std
ax = axes[1]
stds = [COMP[i].std() for i in range(n_bands)]
cols = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]
ax.bar(range(n_bands), stds, color=cols[:n_bands], alpha=0.8)
ax.set_xticks(range(n_bands))
ax.set_xticklabels([f"C{i}" for i in range(n_bands)], fontsize=9)
ax.set_title("std по полосам (амплитуда)")
ax.set_xlabel("Полоса"); ax.set_ylabel("std"); ax.grid(alpha=0.3, axis="y")

# scatter baseline vs fb_skip_c0
ax = axes[2]
b  = mape_res["baseline"]; e = mape_res["fb_skip_c0"]
ok = ~(np.isnan(b) | np.isnan(e))
ax.scatter(b[ok], e[ok], alpha=0.3, s=12, color="seagreen")
lim = max(b[ok].max(), e[ok].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
ax.set_xlabel("baseline MAPE"); ax.set_ylabel("fb_skip_c0 MAPE")
ax.set_title("baseline vs fb_skip_c0")
below = int(np.sum(e[ok] < b[ok]))
above = int(np.sum(e[ok] > b[ok]))
ax.text(0.05, 0.92, f"fb лучше: {below}  хуже: {above}",
        transform=ax.transAxes, fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "30_filterbank_la.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
