"""
32 — Filter bank fb_lo: multi-ticker тест.

Скрипты 30–31 показали: медленные компоненты dratio дают −28% MAPE на SBER.
Проверяем на всех 8 тикерах.

Гипотеза пользователя: часть тикеров принципиально шумнее (NVTK, LKOH —
сильная зависимость от внешних факторов). Для них base_mape высокий,
и никакой метод не поможет. Анализируем корреляцию Δ% с уровнем base_mape.

Тест:
  baseline       — LWR на dratio
  fb_lo_C3plus   — LWR на C3+C4+C5 (периоды 52–206 баров)  ← лучший из скр.30
  fb_lo_C4plus   — LWR на C4+C5    (периоды 109–206 баров)  ← ещё медленнее
  fb_lo_C2plus   — LWR на C2+C3+C4+C5 (периоды 27–206 баров) ← шире

Cutoffs: [0.25, 0.125, 0.0625, 0.03125, 0.015625], order=4.
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
    from scipy.stats import wilcoxon as _wlcx, spearmanr
    _HAS_SCI = True
except ImportError:
    print("pip install scipy"); sys.exit(1)

# ── параметры ─────────────────────────────────────────────────────────────────
TICKERS   = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL  = "1d"
MA_WINDOW = 1000
P         = 5
XI        = 3 * (P + 1)   # 18
VAL_H     = 10
N_EVAL    = 200

FILTER_ORDER = 4
CUTOFFS      = [0.25, 0.125, 0.0625, 0.03125, 0.015625]

# Индексы полос (0=высокочастотный, 5=residue)
BANDS = {
    "fb_lo_C2plus": [2, 3, 4, 5],   # C2+C3+C4+C5  period > 27 bars
    "fb_lo_C3plus": [3, 4, 5],      # C3+C4+C5     period > 52 bars
    "fb_lo_C4plus": [4, 5],         # C4+C5        period > 109 bars
}
CONFIGS = ["baseline"] + list(BANDS.keys())


# ── filter bank ───────────────────────────────────────────────────────────────
def make_filter_bank(series: np.ndarray) -> np.ndarray:
    """Каузальный октавный банк. Возвращает (6, N)."""
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
RES = {}   # RES[ticker][config] = np.ndarray

t_total = time.time()
for ticker in TICKERS:
    path = DATA_DIR / ticker / f"{INTERVAL}.json"
    with open(path) as f:
        candles = json.load(f)
    df     = normalize(candles, window=MA_WINDOW).dropna(subset=["ma"]).reset_index(drop=True)
    ratio  = df["ratio"].values
    dratio = np.diff(ratio)

    # Фильтр-банк один раз на полном ряду
    COMP = make_filter_bank(dratio)   # (6, N)

    min_orig = P + XI + 5
    max_orig = len(dratio) - VAL_H
    origins  = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)

    raw = {c: [] for c in CONFIGS}
    t1  = time.time()

    for vo in origins:
        # baseline
        X, y  = build_delay_matrix(dratio[:vo], P)
        vec   = last_vector(dratio[:vo], P).copy()
        raw["baseline"].append(_mape(_lwr_forecast(X, y, vec, XI), ratio, vo))

        # LWR на каждой полосе
        hats = []
        for comp in COMP:
            Xc, yc = build_delay_matrix(comp[:vo], P)
            if len(Xc) < XI:
                hats.append(np.zeros(VAL_H))
            else:
                vecc = last_vector(comp[:vo], P).copy()
                hats.append(_lwr_forecast(Xc, yc, vecc, XI))

        for cname, idx in BANDS.items():
            raw[cname].append(_mape(np.sum([hats[i] for i in idx], axis=0), ratio, vo))

    RES[ticker] = {c: np.array(raw[c]) for c in CONFIGS}

    bm  = float(np.nanmedian(RES[ticker]["baseline"]))
    b3  = float(np.nanmedian(RES[ticker]["fb_lo_C3plus"]))
    print(f"{ticker:5s}  N={len(origins):3d}  base={bm:.5f}  "
          f"fb_lo_C3plus={b3:.5f}  Δ={(b3-bm)/bm*100:+.2f}%  "
          f"t={time.time()-t1:.1f}s")

print(f"\nTotal: {time.time()-t_total:.1f}s")


# ── агрегированная таблица ────────────────────────────────────────────────────
print("\n" + "=" * 90)
print(f"{'Тикер':>6}  {'base_med':>9}  "
      f"{'C2+':>9}  {'Δ%':>6}  "
      f"{'C3+':>9}  {'Δ%':>6}  "
      f"{'C4+':>9}  {'Δ%':>6}")
print("─" * 90)

base_meds, delta_c3 = [], []

for ticker in TICKERS:
    r   = RES[ticker]
    bm  = float(np.nanmedian(r["baseline"]))
    c2  = float(np.nanmedian(r["fb_lo_C2plus"]))
    c3  = float(np.nanmedian(r["fb_lo_C3plus"]))
    c4  = float(np.nanmedian(r["fb_lo_C4plus"]))
    base_meds.append(bm)
    delta_c3.append((c3 - bm) / bm * 100)
    print(f"{ticker:>6}  {bm:>9.5f}  "
          f"{c2:>9.5f}  {(c2-bm)/bm*100:>+5.1f}%  "
          f"{c3:>9.5f}  {(c3-bm)/bm*100:>+5.1f}%  "
          f"{c4:>9.5f}  {(c4-bm)/bm*100:>+5.1f}%")

# Итого
all_base = np.concatenate([RES[t]["baseline"]     for t in TICKERS])
all_c3   = np.concatenate([RES[t]["fb_lo_C3plus"] for t in TICKERS])
agg_bm   = float(np.nanmedian(all_base))
agg_c3   = float(np.nanmedian(all_c3))
print("─" * 90)
print(f"{'ИТОГО':>6}  {agg_bm:>9.5f}  {'':>9}  {'':>6}  "
      f"{agg_c3:>9.5f}  {(agg_c3-agg_bm)/agg_bm*100:>+5.1f}%")
print("=" * 90)

# Wilcoxon агрегированный
ok   = ~(np.isnan(all_base) | np.isnan(all_c3))
diff = all_c3[ok] - all_base[ok]
_, pv = _wlcx(diff)
print(f"\nWilcoxon (fb_lo_C3plus, N={ok.sum()}): "
      f"ΔMAPE={(agg_c3-agg_bm)/agg_bm*100:+.2f}%  p={pv:.4f}")

# Корреляция Δ% с base_mape (Spearman)
rho, p_rho = spearmanr(base_meds, delta_c3)
print(f"Spearman(base_mape, Δ%): ρ={rho:+.3f}  p={p_rho:.3f}")
print("  (ρ > 0 → шумные тикеры выигрывают меньше)")


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Filter bank fb_lo: multi-ticker  |  {INTERVAL}  p={P}, ξ={XI}, N={N_EVAL}",
    fontsize=11,
)

# Per-ticker Δ% bar chart
ax = axes[0][0]
colors_bar = ["seagreen" if d < 0 else "tomato" for d in delta_c3]
ax.bar(TICKERS, delta_c3, color=colors_bar, alpha=0.85)
ax.axhline(0, color="black", lw=1)
ax.axhline((agg_c3-agg_bm)/agg_bm*100, color="steelblue",
           lw=1.5, ls="--", label=f"агрегат {(agg_c3-agg_bm)/agg_bm*100:+.1f}%")
ax.set_title("Δ% MAPE (fb_lo_C3plus vs baseline)")
ax.set_ylabel("Δ%  (отриц. = лучше)")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
for i, (t, d) in enumerate(zip(TICKERS, delta_c3)):
    ax.text(i, d + (0.3 if d >= 0 else -0.8), f"{d:+.1f}%",
            ha="center", va="bottom", fontsize=8)

# Scatter: base_mape vs Δ%
ax = axes[0][1]
ax.scatter(base_meds, delta_c3, s=60, zorder=5)
for t, x, y in zip(TICKERS, base_meds, delta_c3):
    ax.annotate(t, (x, y), textcoords="offset points",
                xytext=(5, 3), fontsize=8)
ax.axhline(0, color="black", lw=1, ls="--")
# Линия тренда
z = np.polyfit(base_meds, delta_c3, 1)
xr = np.linspace(min(base_meds), max(base_meds), 50)
ax.plot(xr, np.polyval(z, xr), "r--", lw=1.5, alpha=0.6)
ax.set_xlabel("base_mape (уровень непредсказуемости)")
ax.set_ylabel("Δ% fb_lo_C3plus")
ax.set_title(f"Предсказуемость vs улучшение  (ρ={rho:+.3f}, p={p_rho:.2f})")
ax.grid(alpha=0.3)

# CDF: лучшие тикеры
ax = axes[1][0]
sorted_t = sorted(TICKERS, key=lambda t: float(np.nanmedian(RES[t]["baseline"])))
PALETTE = plt.cm.tab10(np.linspace(0, 1, len(TICKERS)))
for i, ticker in enumerate(sorted_t):
    s_b = np.sort(RES[ticker]["baseline"][~np.isnan(RES[ticker]["baseline"])])
    s_f = np.sort(RES[ticker]["fb_lo_C3plus"][~np.isnan(RES[ticker]["fb_lo_C3plus"])])
    bm  = float(np.nanmedian(RES[ticker]["baseline"]))
    fc3 = float(np.nanmedian(RES[ticker]["fb_lo_C3plus"]))
    ax.plot(np.linspace(0, 100, len(s_b)), s_b,
            color=PALETTE[i], lw=1.5, ls="--", alpha=0.6)
    ax.plot(np.linspace(0, 100, len(s_f)), s_f,
            color=PALETTE[i], lw=1.5,
            label=f"{ticker}  b={bm:.4f}→{fc3:.4f}")
ax.set_title("CDF: base (--) vs fb_lo_C3plus (─)")
ax.set_xlabel("Перцентиль"); ax.set_ylabel("val_mape")
ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_ylim(0)

# Band comparison per ticker (stacked bars)
ax = axes[1][1]
x   = np.arange(len(TICKERS))
w   = 0.25
dp_c2 = [(float(np.nanmedian(RES[t]["fb_lo_C2plus"])) -
          float(np.nanmedian(RES[t]["baseline"]))) /
          float(np.nanmedian(RES[t]["baseline"])) * 100
         for t in TICKERS]
dp_c3 = delta_c3
dp_c4 = [(float(np.nanmedian(RES[t]["fb_lo_C4plus"])) -
          float(np.nanmedian(RES[t]["baseline"]))) /
          float(np.nanmedian(RES[t]["baseline"])) * 100
         for t in TICKERS]
ax.bar(x - w,   dp_c2, w, label="C2+ (>27bar)", color="darkorange", alpha=0.8)
ax.bar(x,       dp_c3, w, label="C3+ (>52bar)", color="seagreen",   alpha=0.8)
ax.bar(x + w,   dp_c4, w, label="C4+ (>109bar)", color="steelblue", alpha=0.8)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x); ax.set_xticklabels(TICKERS, fontsize=9)
ax.set_ylabel("Δ%  (отриц. = лучше)")
ax.set_title("Δ% по границе отсечения полос")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
out_path = OUT_DIR / "32_fb_multiticker.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
