"""
29 — EMD + LA: walk-forward (time-correct).

Скрипт 28 показал −45% / −53% улучшение, но EMD считался один раз на полном ряду
→ потенциальный look-ahead bias.

Здесь EMD пересчитывается на каждом origin по dratio[:vo] — строго time-correct.
Никаких данных из будущего в разложении нет.

Конфиги:
  baseline   — LWR на исходном dratio[:vo]
  emd_all    — LWR на каждой IMF[:vo], сумма прогнозов
  emd_skip1  — то же, без IMF-0 (высокочастотный компонент)

Диагностика: кол-во IMF на каждом origin, реконструкционная ошибка.
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
    from PyEMD import EMD as _EMD
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

min_orig = P + XI + 5
max_orig = len(dratio) - VAL_H
origins  = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)
print(f"p={P}  ξ={XI}  val_h={VAL_H}  origins={len(origins)}\n")


# ── walk-forward с пересчётом EMD ─────────────────────────────────────────────
mape_base   = []
mape_all    = []
mape_skip1  = []
n_imfs_log  = []

emd = _EMD()
t0  = time.time()

for vi, vo in enumerate(origins):
    series = dratio[:vo]

    # baseline
    X, y  = build_delay_matrix(series, P)
    vec   = last_vector(series, P).copy()
    hat_b = _lwr_forecast(X, y, vec, XI)
    mape_base.append(_mape(hat_b, ratio, vo))

    # EMD на series[:vo]  ← строго time-correct
    try:
        imfs = emd(series)            # (n_imfs, vo)
    except Exception:
        # редкий сбой EMD → используем только baseline
        mape_all.append(np.nan)
        mape_skip1.append(np.nan)
        n_imfs_log.append(0)
        continue

    n_imfs = imfs.shape[0]
    n_imfs_log.append(n_imfs)

    # LWR на каждой IMF
    hats = []
    for imf in imfs:
        Xi, yi = build_delay_matrix(imf, P)
        if len(Xi) < XI:
            hats.append(np.zeros(VAL_H))
        else:
            veci = last_vector(imf, P).copy()
            hats.append(_lwr_forecast(Xi, yi, veci, XI))

    # emd_all
    hat_all = np.sum(hats, axis=0)
    mape_all.append(_mape(hat_all, ratio, vo))

    # emd_skip1 (без IMF-0)
    if n_imfs > 1:
        hat_skip1 = np.sum(hats[1:], axis=0)
    else:
        hat_skip1 = hats[0]
    mape_skip1.append(_mape(hat_skip1, ratio, vo))

    if (vi + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"  {vi+1}/{len(origins)}  t={elapsed:.1f}s  "
              f"n_imfs_last={n_imfs}  "
              f"base_med={float(np.nanmedian(mape_base)):.5f}  "
              f"emd_all_med={float(np.nanmedian(mape_all)):.5f}",
              flush=True)

mape_base  = np.array(mape_base)
mape_all   = np.array(mape_all)
mape_skip1 = np.array(mape_skip1)
n_imfs_log = np.array(n_imfs_log)

print(f"\nTotal: {time.time()-t0:.1f}s")
print(f"IMFs per origin: min={n_imfs_log[n_imfs_log>0].min()}  "
      f"max={n_imfs_log.max()}  "
      f"mean={n_imfs_log[n_imfs_log>0].mean():.1f}")


# ── таблица ───────────────────────────────────────────────────────────────────
bm = float(np.nanmedian(mape_base))

print("\n" + "=" * 60)
print(f"Walk-forward EMD (time-correct)  |  N={len(origins)}")
print(f"baseline_med = {bm:.5f}")
print(f"\n{'Конфиг':<14}  {'mean':>9}  {'median':>9}  {'Δ%':>7}  {'W-p':>8}")
print("─" * 55)

try:
    from scipy.stats import wilcoxon as _wlcx
    _HAS_SCI = True
except ImportError:
    _HAS_SCI = False

for name, arr in [("baseline", mape_base),
                  ("emd_all",   mape_all),
                  ("emd_skip1", mape_skip1)]:
    mn  = float(np.nanmean(arr))
    med = float(np.nanmedian(arr))
    dp  = (med - bm) / bm * 100
    wp  = "—"
    if _HAS_SCI and name != "baseline":
        ok   = ~(np.isnan(mape_base) | np.isnan(arr))
        diff = arr[ok] - mape_base[ok]
        if ok.sum() >= 10 and not np.all(diff == 0):
            _, pv = _wlcx(diff)
            wp = f"{pv:.4f}"
    tag = "baseline" if name == "baseline" else f"{dp:>+6.2f}%"
    print(f"{name:<14}  {mn:>9.5f}  {med:>9.5f}  {tag:>7}  {wp:>8}")

print("=" * 60)


# ── сравнение скр. 28 (global EMD) vs скр. 29 (walk-forward) ─────────────────
print("\n── Сравнение: global EMD (скр.28) vs walk-forward EMD (скр.29) ──")
print(f"  {'Конфиг':<14}  {'global med':>11}  {'WF med':>9}  {'разница':>9}")
print("  " + "─" * 46)
global_28 = {"baseline": 0.01306, "emd_all": 0.00723, "emd_skip1": 0.00615}
for name, arr in [("baseline", mape_base), ("emd_all", mape_all), ("emd_skip1", mape_skip1)]:
    med_wf = float(np.nanmedian(arr))
    med_g  = global_28.get(name, 0)
    diff   = med_wf - med_g
    print(f"  {name:<14}  {med_g:>11.5f}  {med_wf:>9.5f}  {diff:>+9.5f}")


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"Walk-forward EMD + LA  |  {TICKER} {INTERVAL}  "
    f"p={P}, ξ={XI}, N={N_EVAL}",
    fontsize=11,
)

# CDF
ax = axes[0]
COLORS = {"baseline": "steelblue", "emd_all": "tomato", "emd_skip1": "seagreen"}
for name, arr in [("baseline", mape_base), ("emd_all", mape_all), ("emd_skip1", mape_skip1)]:
    s   = np.sort(arr[~np.isnan(arr)])
    med = float(np.nanmedian(arr))
    lw  = 2.5 if name == "baseline" else 1.8
    ax.plot(np.linspace(0, 100, len(s)), s,
            color=COLORS[name], lw=lw,
            label=f"{name}  (med={med:.4f})")
ax.set_title("CDF val_mape (walk-forward EMD)")
ax.set_xlabel("Перцентиль"); ax.set_ylabel("val_mape")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0)

# scatter baseline vs emd_skip1
ax = axes[1]
ok = ~(np.isnan(mape_base) | np.isnan(mape_skip1))
ax.scatter(mape_base[ok], mape_skip1[ok], alpha=0.3, s=12, color="seagreen")
lim = max(mape_base[ok].max(), mape_skip1[ok].max()) * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
ax.set_xlabel("baseline MAPE"); ax.set_ylabel("emd_skip1 MAPE")
ax.set_title("baseline vs emd_skip1")
above = int(np.sum(mape_skip1[ok] > mape_base[ok]))
below = int(np.sum(mape_skip1[ok] < mape_base[ok]))
ax.text(0.05, 0.92, f"EMD лучше: {below}  хуже: {above}",
        transform=ax.transAxes, fontsize=9)
ax.grid(alpha=0.3)

# кол-во IMFs по origins
ax = axes[2]
ax.plot(origins, n_imfs_log, color="steelblue", lw=1.2)
ax.axhline(n_imfs_log[n_imfs_log > 0].mean(), color="tomato",
           lw=1.5, ls="--", label=f"mean={n_imfs_log[n_imfs_log>0].mean():.1f}")
ax.set_title("Кол-во IMF по origins")
ax.set_xlabel("origin"); ax.set_ylabel("n_imfs")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
out_path = OUT_DIR / "29_emd_walkforward.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
