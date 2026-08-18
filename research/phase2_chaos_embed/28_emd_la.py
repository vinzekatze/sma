"""
28 — EMD + LA: LA на каждой IMF-компоненте отдельно.

Прямая проверка гипотезы «sum of subsystems»:
  dratio = IMF₁ + IMF₂ + ... + IMFₖ + residue
  → запустить LWR на каждой IMF независимо
  → сложить прогнозы
  → сравнить с LWR на исходном dratio

Идея: каждая IMF — более «простой» аттрактор, чем их смесь.
LWR на однородной IMF находит истинных соседей, а не случайно близких
из смешанного пространства.

Методическое ограничение: EMD выполняется ОДИН РАЗ на полном dratio
(non-time-correct). Причина: walk-forward EMD был бы корректнее, но
boundary effects EMD у края series сильнее, чем эффект look-ahead.
Для первичного теста концепции приемлемо.

Конфиги:
  baseline    — LWR на исходном dratio
  emd_all     — LWR на каждой IMF, сумма прогнозов
  emd_skip1   — то же, но без IMF-0 (высокочастотный шум)
  emd_only_lo — только нижние 50% IMF по частоте (медленные компоненты)

Тикер: SBER 1d, N=200.
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
    from PyEMD import EMD
except ImportError:
    print("pip install EMD-signal"); sys.exit(1)

# ── параметры ─────────────────────────────────────────────────────────────────
TICKER    = "SBER"
INTERVAL  = "1d"
MA_WINDOW = 1000
P         = 5
XI        = 3 * (P + 1)   # 18
VAL_H     = 10
N_EVAL    = 200


# ── LWR прогноз на VAL_H шагов ───────────────────────────────────────────────
def _lwr_forecast(X: np.ndarray, y: np.ndarray, vec: np.ndarray,
                  xi: int) -> np.ndarray:
    v   = vec.copy()
    hat = np.empty(VAL_H)
    for h in range(VAL_H):
        dists  = np.linalg.norm(X - v, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        nn_dx  = dists[nn_idx]
        h_bw   = max(float(nn_dx.max()), 1e-10)
        w      = np.exp(-0.5 * (nn_dx / h_bw) ** 2)
        A      = np.hstack([np.ones((xi, 1)), X[nn_idx]])
        sw     = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn_idx], rcond=None)
        hat[h] = float(c[0] + v @ c[1:])
        v = np.roll(v, -1); v[-1] = hat[h]
    return hat


def _mape_from_hat(hat: np.ndarray, ratio: np.ndarray, vo: int) -> float:
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


# ── EMD один раз на полном ряду ───────────────────────────────────────────────
print("EMD decomposition...", end=" ", flush=True)
t0   = time.time()
emd  = EMD()
imfs = emd(dratio)       # shape: (n_imfs, len(dratio))
n_imfs = imfs.shape[0]
print(f"{n_imfs} IMFs  t={time.time()-t0:.2f}s")

# Проверка реконструкции
recon_err = float(np.max(np.abs(imfs.sum(axis=0) - dratio)))
print(f"Реконструкция: max_err={recon_err:.2e}")

# Информация по IMF
print(f"\n{'IMF':>4}  {'std':>10}  {'energy%':>8}  {'dom_freq':>9}")
print("─" * 40)
total_energy = float(np.sum(imfs ** 2))
for i, imf in enumerate(imfs):
    en_pct  = float(np.sum(imf ** 2)) / total_energy * 100
    # dominant frequency через zero-crossing rate
    zc = np.sum(np.diff(np.sign(imf)) != 0)
    dom_freq = zc / (2 * len(imf))
    lbl = " (residue)" if i == n_imfs - 1 else ""
    print(f"{i:>4}  {imf.std():>10.6f}  {en_pct:>7.2f}%  {dom_freq:>9.4f}{lbl}")
print()


# ── разбивка IMF на группы ───────────────────────────────────────────────────
# high-freq: верхняя половина по dominant freq (кроме residue)
# low-freq: нижняя половина + residue
n_signal = n_imfs - 1   # без residue
lo_imfs  = list(range(n_signal // 2, n_imfs))    # медленные + residue
hi_imfs  = list(range(n_signal // 2))             # быстрые

print(f"IMF группы:  high-freq={hi_imfs}  low-freq={lo_imfs}")


# ── walk-forward ──────────────────────────────────────────────────────────────
min_orig = P + XI + 5
max_orig = len(dratio) - VAL_H
origins  = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)

CONFIGS = ["baseline", "emd_all", "emd_skip1", "emd_only_lo"]
mape_res = {c: [] for c in CONFIGS}

print(f"Walk-forward: origins={len(origins)}  p={P}  ξ={XI}")
t1 = time.time()

for vo in origins:
    # ── baseline ─────────────────────────────────────────────────────────────
    X, y  = build_delay_matrix(dratio[:vo], P)
    vec   = last_vector(dratio[:vo], P).copy()
    hat_b = _lwr_forecast(X, y, vec, XI)
    mape_res["baseline"].append(_mape_from_hat(hat_b, ratio, vo))

    # ── EMD конфиги ──────────────────────────────────────────────────────────
    hats_imf = []
    for i, imf in enumerate(imfs):
        Xi, yi = build_delay_matrix(imf[:vo], P)
        veci   = last_vector(imf[:vo], P).copy()
        if len(Xi) < XI:
            hats_imf.append(np.zeros(VAL_H))
        else:
            hats_imf.append(_lwr_forecast(Xi, yi, veci, XI))

    hat_all = np.sum(hats_imf, axis=0)
    mape_res["emd_all"].append(_mape_from_hat(hat_all, ratio, vo))

    hat_skip1 = np.sum([hats_imf[i] for i in range(1, n_imfs)], axis=0)
    mape_res["emd_skip1"].append(_mape_from_hat(hat_skip1, ratio, vo))

    hat_lo = np.sum([hats_imf[i] for i in lo_imfs], axis=0)
    mape_res["emd_only_lo"].append(_mape_from_hat(hat_lo, ratio, vo))

for c in CONFIGS:
    mape_res[c] = np.array(mape_res[c])

print(f"t={time.time()-t1:.1f}s\n")


# ── таблица ───────────────────────────────────────────────────────────────────
bm = float(np.nanmedian(mape_res["baseline"]))
print("=" * 60)
print(f"baseline_med = {bm:.5f}  (N={len(origins)})")
print(f"\n{'Конфиг':<14}  {'mean':>9}  {'median':>9}  {'Δ%':>7}  {'W-p':>8}")
print("─" * 55)

try:
    from scipy.stats import wilcoxon as _wlcx
    _HAS_SCI = True
except ImportError:
    _HAS_SCI = False

for c in CONFIGS:
    arr = mape_res[c]
    mn  = float(np.nanmean(arr))
    med = float(np.nanmedian(arr))
    dp  = (med - bm) / bm * 100
    wp  = "—"
    if _HAS_SCI and c != "baseline":
        ok   = ~(np.isnan(mape_res["baseline"]) | np.isnan(arr))
        diff = arr[ok] - mape_res["baseline"][ok]
        if ok.sum() >= 10 and not np.all(diff == 0):
            _, pv = _wlcx(diff)
            wp = f"{pv:.4f}"
    tag = "baseline" if c == "baseline" else f"{dp:>+6.2f}%"
    print(f"{c:<14}  {mn:>9.5f}  {med:>9.5f}  {tag:>7}  {wp:>8}")

print("=" * 60)


# ── визуализация ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"EMD + LA  |  {TICKER} {INTERVAL}  p={P}, ξ={XI}, N={N_EVAL}",
    fontsize=11,
)

# CDF val_mape
ax = axes[0]
COLORS = {"baseline": "steelblue", "emd_all": "tomato",
          "emd_skip1": "seagreen", "emd_only_lo": "darkorange"}
for c in CONFIGS:
    arr = mape_res[c]
    s   = np.sort(arr[~np.isnan(arr)])
    med = float(np.nanmedian(arr))
    lw  = 2.5 if c == "baseline" else 1.8
    ax.plot(np.linspace(0, 100, len(s)), s,
            color=COLORS[c], lw=lw, label=f"{c}  (med={med:.4f})")
ax.set_title("CDF val_mape")
ax.set_xlabel("Перцентиль")
ax.set_ylabel("val_mape")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0)

# IMF стандартные отклонения
ax = axes[1]
stds = [imfs[i].std() for i in range(n_imfs)]
ax.bar(range(n_imfs), stds, color="steelblue", alpha=0.7)
ax.set_title("IMF std (амплитуды)")
ax.set_xlabel("IMF index")
ax.set_ylabel("std")
ax.set_xticks(range(n_imfs))
ax.grid(alpha=0.3, axis="y")

# per-origin scatter: baseline vs emd_all
ax = axes[2]
b   = mape_res["baseline"]
e   = mape_res["emd_all"]
ok  = ~(np.isnan(b) | np.isnan(e))
ax.scatter(b[ok], e[ok], alpha=0.3, s=12, color="tomato")
lim = max(b[ok].max(), e[ok].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
ax.set_xlabel("baseline MAPE")
ax.set_ylabel("emd_all MAPE")
ax.set_title("baseline vs emd_all (каждая точка — origin)")
above = int(np.sum(e[ok] > b[ok]))
below = int(np.sum(e[ok] < b[ok]))
ax.text(0.05, 0.92, f"EMD лучше: {below}  хуже: {above}",
        transform=ax.transAxes, fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "28_emd_la.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
