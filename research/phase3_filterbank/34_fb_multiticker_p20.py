"""
34 — Filter bank fb_lo_C3plus, p=20: multi-ticker тест.

Скрипт 32 (p=5): aggregate −5.4%, p=0.008. LKOH +36% — исключение.
Скрипт 33 (SBER): p=20 даёт абс. MAPE 0.00917 (−37.6% vs baseline p=20).

Цель: проверить, улучшается ли aggregate при p=20 и изменяется ли поведение LKOH.
Дополнительно: сравниваем с результатами p=5 (скр.32) в одной таблице.
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
TICKERS   = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL  = "1d"
MA_WINDOW = 1000
P         = 20
XI        = 3 * (P + 1)   # 63
VAL_H     = 10
N_EVAL    = 200

FILTER_ORDER = 4
CUTOFFS      = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
C3_IDX       = [3, 4, 5]   # C3+C4+C5

# Результаты скр. 32 (p=5) для сравнения в финальной таблице
PREV_P5 = {
    "CHMF": (-24.0, 0.02480), "LKOH": (+36.0, 0.02271),
    "MGNT": (-21.6, 0.02558), "MRKP": (+1.3,  0.02320),
    "NLMK": (-5.7,  0.02291), "NVTK": (+1.3,  0.02667),
    "SBER": (-28.2, 0.01306), "VTBR": (+10.0, 0.02208),
}


# ── filter bank ───────────────────────────────────────────────────────────────
def make_filter_bank(series: np.ndarray) -> np.ndarray:
    components = []
    remaining  = series.copy()
    for fc in CUTOFFS:
        sos  = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low  = sosfilt(sos, remaining)
        components.append(remaining - low)
        remaining = low
    components.append(remaining)
    return np.array(components)


# ── LWR ──────────────────────────────────────────────────────────────────────
def _lwr_forecast(X, y, vec, xi):
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


def _mape(hat, ratio, vo):
    r0     = float(ratio[vo])
    r_hat  = r0 + np.cumsum(hat)
    actual = ratio[vo + 1: vo + 1 + VAL_H]
    n_     = min(len(r_hat), len(actual))
    if n_ == 0:
        return np.nan
    return float(np.mean(
        np.abs(r_hat[:n_] - actual[:n_]) / (np.abs(actual[:n_]) + 1e-12)
    ))


# ── основной цикл ─────────────────────────────────────────────────────────────
RES = {}   # RES[ticker] = {"base": arr, "fb": arr}

t_total = time.time()
for ticker in TICKERS:
    with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f:
        candles = json.load(f)
    df     = normalize(candles, window=MA_WINDOW).dropna(subset=["ma"]).reset_index(drop=True)
    ratio  = df["ratio"].values
    dratio = np.diff(ratio)

    COMP = make_filter_bank(dratio)

    min_orig = P + XI + 5
    max_orig = len(dratio) - VAL_H
    origins  = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)

    raw_base, raw_fb = [], []
    t1 = time.time()

    for vo in origins:
        # baseline p=20
        X, y  = build_delay_matrix(dratio[:vo], P)
        vec   = last_vector(dratio[:vo], p=P).copy()
        raw_base.append(_mape(_lwr_forecast(X, y, vec, XI), ratio, vo))

        # filter bank: LA на C3, C4, C5 отдельно
        hats = []
        for ci in C3_IDX:
            Xc, yc = build_delay_matrix(COMP[ci, :vo], P)
            if len(Xc) < XI:
                hats.append(np.zeros(VAL_H))
            else:
                vecc = last_vector(COMP[ci, :vo], p=P).copy()
                hats.append(_lwr_forecast(Xc, yc, vecc, XI))
        raw_fb.append(_mape(np.sum(hats, axis=0), ratio, vo))

    RES[ticker] = {
        "base": np.array(raw_base),
        "fb":   np.array(raw_fb),
    }

    bm = float(np.nanmedian(RES[ticker]["base"]))
    fm = float(np.nanmedian(RES[ticker]["fb"]))
    print(f"{ticker:5s}  N={len(origins):3d}  "
          f"base={bm:.5f}  fb={fm:.5f}  "
          f"Δ={(fm-bm)/bm*100:+.2f}%  t={time.time()-t1:.1f}s")

print(f"\nTotal: {time.time()-t_total:.1f}s")


# ── таблица со сравнением p=5 vs p=20 ─────────────────────────────────────────
all_base = np.concatenate([RES[t]["base"] for t in TICKERS])
all_fb   = np.concatenate([RES[t]["fb"]   for t in TICKERS])
agg_bm   = float(np.nanmedian(all_base))
agg_fm   = float(np.nanmedian(all_fb))
ok        = ~(np.isnan(all_base) | np.isnan(all_fb))
_, pv_agg = _wlcx(all_fb[ok] - all_base[ok])

print("\n" + "=" * 88)
print(f"{'Тикер':>6}  {'base p=20':>10}  {'fb p=20':>9}  {'Δ% p=20':>9}  "
      f"{'base p=5':>9}  {'Δ% p=5':>9}  {'изменение':>10}")
print("─" * 88)

deltas_p20, deltas_p5 = [], []
for ticker in TICKERS:
    bm20 = float(np.nanmedian(RES[ticker]["base"]))
    fm20 = float(np.nanmedian(RES[ticker]["fb"]))
    d20  = (fm20 - bm20) / bm20 * 100
    d5   = PREV_P5[ticker][0]
    bm5  = PREV_P5[ticker][1]
    diff = d20 - d5   # насколько p=20 лучше/хуже p=5
    deltas_p20.append(d20); deltas_p5.append(d5)
    print(f"{ticker:>6}  {bm20:>10.5f}  {fm20:>9.5f}  {d20:>+8.1f}%  "
          f"{bm5:>9.5f}  {d5:>+8.1f}%  {diff:>+9.1f}pp")

print("─" * 88)
print(f"{'ИТОГО':>6}  {agg_bm:>10.5f}  {agg_fm:>9.5f}  "
      f"{(agg_fm-agg_bm)/agg_bm*100:>+8.1f}%  "
      f"{'(p=5: -5.4%)':>9}  {'':>9}")
print("=" * 88)
print(f"\nWilcoxon aggregate (p=20, N={ok.sum()}): "
      f"ΔMAPE={(agg_fm-agg_bm)/agg_bm*100:+.2f}%  p={pv_agg:.4f}")


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"Filter bank fb_lo_C3plus  |  p=20 vs p=5  |  {INTERVAL}  val_h={VAL_H}",
    fontsize=11,
)

# Δ% comparison: p=5 vs p=20
ax = axes[0]
x  = np.arange(len(TICKERS))
w  = 0.35
c_p5  = ["seagreen" if d < 0 else "tomato" for d in deltas_p5]
c_p20 = ["seagreen" if d < 0 else "tomato" for d in deltas_p20]
ax.bar(x - w/2, deltas_p5,  w, color=c_p5,  alpha=0.6, label="p=5  (скр.32)")
ax.bar(x + w/2, deltas_p20, w, color=c_p20, alpha=0.9, label="p=20 (скр.34)")
ax.axhline(0, color="black", lw=1)
ax.axhline((agg_fm-agg_bm)/agg_bm*100, color="navy",
           lw=1.5, ls="--", label=f"p=20 aggr {(agg_fm-agg_bm)/agg_bm*100:+.1f}%")
ax.set_xticks(x); ax.set_xticklabels(TICKERS, fontsize=9)
ax.set_ylabel("Δ%  (отриц. = лучше)")
ax.set_title("Δ% fb_lo_C3plus: p=5 vs p=20")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

# CDF агрегированный
ax = axes[1]
s_b = np.sort(all_base[~np.isnan(all_base)])
s_f = np.sort(all_fb[~np.isnan(all_fb)])
ax.plot(np.linspace(0, 100, len(s_b)), s_b, "steelblue", lw=2.5,
        label=f"baseline p=20  (med={agg_bm:.5f})")
ax.plot(np.linspace(0, 100, len(s_f)), s_f, "seagreen",  lw=2,
        label=f"fb_sep p=20    (med={agg_fm:.5f}  Δ={(agg_fm-agg_bm)/agg_bm*100:+.1f}%)")
ax.set_title(f"CDF агрегированный  N={len(s_b)}")
ax.set_xlabel("Перцентиль"); ax.set_ylabel("val_mape")
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(0)

# Per-ticker: base p=20 vs fb p=20
ax = axes[2]
PALETTE = plt.cm.tab10(np.linspace(0, 1, len(TICKERS)))
sorted_t = sorted(TICKERS, key=lambda t: float(np.nanmedian(RES[t]["base"])))
for i, ticker in enumerate(sorted_t):
    bm = float(np.nanmedian(RES[ticker]["base"]))
    fm = float(np.nanmedian(RES[ticker]["fb"]))
    s_b2 = np.sort(RES[ticker]["base"][~np.isnan(RES[ticker]["base"])])
    s_f2 = np.sort(RES[ticker]["fb"][~np.isnan(RES[ticker]["fb"])])
    ax.plot(np.linspace(0, 100, len(s_b2)), s_b2, color=PALETTE[i], lw=1.2,
            ls="--", alpha=0.5)
    ax.plot(np.linspace(0, 100, len(s_f2)), s_f2, color=PALETTE[i], lw=1.5,
            label=f"{ticker} {(fm-bm)/bm*100:+.1f}%")
ax.set_title("Per-ticker: base (--) vs fb_sep (─)")
ax.set_xlabel("Перцентиль"); ax.set_ylabel("val_mape")
ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_ylim(0)

plt.tight_layout()
out_path = OUT_DIR / "34_fb_multiticker_p20.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
