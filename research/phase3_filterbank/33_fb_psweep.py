"""
33 — Filter bank: sweep по p для медленных компонент.

Скрипт 32 показал: fb_lo_C3+ (C3+C4+C5, периоды >52 баров) → −28% на SBER.
Но p=5 — это 5-барный delay-вектор при периоде компоненты 52–206 баров.
Несоответствие масштабов: 5 баров охватывают <10% одного цикла.

Гипотеза: при p=20–40 LA получает больше контекста о медленной динамике
→ улучшение усиливается.

Дополнительно тестируем две реализации медленных компонент:
  «sep»  — LA на C3, C4, C5 по отдельности, суммируем прогнозы (как в скр.30–32)
  «lp»   — sum(C3+C4+C5) = одиночный LP-фильтр dratio при cutoff=0.0625,
           LA на получившемся одном сигнале (проще в production)

Sweep: p ∈ {5, 10, 20, 40}.
Тикер: SBER 1d.
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
VAL_H     = 10
N_EVAL    = 200

P_LIST       = [5, 10, 20, 40]
FILTER_ORDER = 4
CUTOFFS      = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
C3_PLUS_IDX  = [3, 4, 5]   # C3, C4, C5 в filter bank
LP_CUTOFF    = 0.0625       # single LP: sum(C3+C4+C5) = LP(0.0625)


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
    return np.array(components)   # (6, N)


def make_lp(series: np.ndarray, fc: float) -> np.ndarray:
    sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
    return sosfilt(sos, series)


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

# Предвычисляем разложения один раз на полном ряду
COMP    = make_filter_bank(dratio)              # (6, N) — все полосы
LP_SIG  = make_lp(dratio, LP_CUTOFF)           # (N,)   — одиночный LP

recon_c3 = COMP[C3_PLUS_IDX[0]:].sum(axis=0)
# LP_SIG = одиночный LP(0.0625); recon_c3 = каскадный LP(0.25)→LP(0.125)→LP(0.0625)
# Ожидаемо разные (более крутой спад у каскадного)
print(f"LP_single vs sum(C3+C4+C5) max_diff: "
      f"{np.max(np.abs(LP_SIG - recon_c3)):.4f}  "
      f"(разные: одиночный vs каскадный фильтр)\n")


# ── walk-forward sweep по p ───────────────────────────────────────────────────
# RES[p] = {"base": arr, "sep": arr, "lp": arr}
RES: dict[int, dict[str, np.ndarray]] = {}

for p in P_LIST:
    xi      = 3 * (p + 1)
    min_orig = p + xi + 5
    max_orig = len(dratio) - VAL_H
    origins  = np.arange(max(min_orig, max_orig - N_EVAL), max_orig)

    raw = {"base": [], "sep": [], "lp": []}
    t0  = time.time()

    for vo in origins:
        # baseline
        X, y  = build_delay_matrix(dratio[:vo], p)
        vec   = last_vector(dratio[:vo], p=p).copy()
        raw["base"].append(_mape(_lwr_forecast(X, y, vec, xi), ratio, vo))

        # sep: LA на C3, C4, C5 по отдельности
        hats = []
        for ci in C3_PLUS_IDX:
            Xc, yc = build_delay_matrix(COMP[ci, :vo], p)
            if len(Xc) < xi:
                hats.append(np.zeros(VAL_H))
            else:
                vecc = last_vector(COMP[ci, :vo], p=p).copy()
                hats.append(_lwr_forecast(Xc, yc, vecc, xi))
        raw["sep"].append(_mape(np.sum(hats, axis=0), ratio, vo))

        # lp: LA на одиночном LP-сигнале
        Xl, yl = build_delay_matrix(LP_SIG[:vo], p)
        if len(Xl) >= xi:
            vecl = last_vector(LP_SIG[:vo], p=p).copy()
            raw["lp"].append(_mape(_lwr_forecast(Xl, yl, vecl, xi), ratio, vo))
        else:
            raw["lp"].append(np.nan)

    RES[p] = {k: np.array(v) for k, v in raw.items()}

    bm   = float(np.nanmedian(RES[p]["base"]))
    msep = float(np.nanmedian(RES[p]["sep"]))
    mlp  = float(np.nanmedian(RES[p]["lp"]))
    print(f"p={p:2d}  ξ={xi:3d}  t={time.time()-t0:.1f}s  "
          f"base={bm:.5f}  "
          f"sep={msep:.5f}({(msep-bm)/bm*100:+.1f}%)  "
          f"lp={mlp:.5f}({(mlp-bm)/bm*100:+.1f}%)")


# ── таблица ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"{'p':>4}  {'ξ':>4}  {'base_med':>9}  "
      f"{'sep':>9}  {'Δ%':>7}  {'W-p':>7}  "
      f"{'lp':>9}  {'Δ%':>7}  {'W-p':>7}")
print("─" * 72)

for p in P_LIST:
    xi  = 3 * (p + 1)
    bm  = float(np.nanmedian(RES[p]["base"]))
    msep = float(np.nanmedian(RES[p]["sep"]))
    mlp  = float(np.nanmedian(RES[p]["lp"]))
    dsep = (msep - bm) / bm * 100
    dlp  = (mlp  - bm) / bm * 100

    def wp(arr):
        if not _HAS_SCI: return "—"
        ok   = ~(np.isnan(RES[p]["base"]) | np.isnan(arr))
        diff = arr[ok] - RES[p]["base"][ok]
        if ok.sum() < 10 or np.all(diff == 0): return "—"
        _, pv = _wlcx(diff)
        return f"{pv:.3f}"

    print(f"{p:>4}  {xi:>4}  {bm:>9.5f}  "
          f"{msep:>9.5f}  {dsep:>+6.1f}%  {wp(RES[p]['sep']):>7}  "
          f"{mlp:>9.5f}  {dlp:>+6.1f}%  {wp(RES[p]['lp']):>7}")

print("=" * 72)


# ── графики ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"Filter bank: p sweep  |  {TICKER} {INTERVAL}  "
    f"fb_lo_C3+ (sep vs lp)  val_h={VAL_H}",
    fontsize=11,
)

# CDF по p (лучший вариант: lp или sep)
ax = axes[0]
COLORS_P = {5: "steelblue", 10: "seagreen", 20: "darkorange", 40: "tomato"}
for p in P_LIST:
    bm  = float(np.nanmedian(RES[p]["base"]))
    best_arr = RES[p]["lp"] if float(np.nanmedian(RES[p]["lp"])) < float(np.nanmedian(RES[p]["sep"])) \
               else RES[p]["sep"]
    best_med = float(np.nanmedian(best_arr))
    tag = "lp" if float(np.nanmedian(RES[p]["lp"])) < float(np.nanmedian(RES[p]["sep"])) else "sep"
    s   = np.sort(best_arr[~np.isnan(best_arr)])
    ax.plot(np.linspace(0, 100, len(s)), s, color=COLORS_P[p], lw=2,
            label=f"p={p} {tag}  (med={best_med:.4f}  Δ={(best_med-bm)/bm*100:+.1f}%)")

# baseline p=5 для ориентира
s_b = np.sort(RES[5]["base"][~np.isnan(RES[5]["base"])])
ax.plot(np.linspace(0, 100, len(s_b)), s_b, "k--", lw=1.5,
        label=f"baseline p=5 (med={float(np.nanmedian(RES[5]['base'])):.4f})")
ax.set_title("CDF: лучший filter bank по p")
ax.set_xlabel("Перцентиль"); ax.set_ylabel("val_mape")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0)

# Δ% по p: sep vs lp
ax = axes[1]
x  = np.arange(len(P_LIST))
w  = 0.35
dp_sep = [(float(np.nanmedian(RES[p]["sep"])) -
           float(np.nanmedian(RES[p]["base"]))) /
           float(np.nanmedian(RES[p]["base"])) * 100
          for p in P_LIST]
dp_lp  = [(float(np.nanmedian(RES[p]["lp"])) -
           float(np.nanmedian(RES[p]["base"]))) /
           float(np.nanmedian(RES[p]["base"])) * 100
          for p in P_LIST]
ax.bar(x - w/2, dp_sep, w, label="sep (3 компоненты)", color="seagreen", alpha=0.8)
ax.bar(x + w/2, dp_lp,  w, label="lp  (1 LP сигнал)",  color="darkorange", alpha=0.8)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x); ax.set_xticklabels([f"p={p}" for p in P_LIST])
ax.set_ylabel("Δ%  (отриц. = лучше)")
ax.set_title("sep vs lp по p")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
for i, (ds, dl) in enumerate(zip(dp_sep, dp_lp)):
    ax.text(i - w/2, ds + (0.3 if ds >= 0 else -0.8),
            f"{ds:+.1f}%", ha="center", va="bottom", fontsize=7)
    ax.text(i + w/2, dl + (0.3 if dl >= 0 else -0.8),
            f"{dl:+.1f}%", ha="center", va="bottom", fontsize=7)

# baseline_p: растёт ли baseline с p?
ax = axes[2]
base_meds_p = [float(np.nanmedian(RES[p]["base"])) for p in P_LIST]
ax.plot(P_LIST, base_meds_p, "ko-", lw=2, label="baseline (raw dratio)")
ax.plot(P_LIST, [float(np.nanmedian(RES[p]["sep"])) for p in P_LIST],
        "gs--", lw=1.8, label="fb_sep")
ax.plot(P_LIST, [float(np.nanmedian(RES[p]["lp"])) for p in P_LIST],
        "D-", color="darkorange", lw=1.8, label="fb_lp")
ax.set_xlabel("p"); ax.set_ylabel("median MAPE")
ax.set_title("Median MAPE vs p")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_xticks(P_LIST)

plt.tight_layout()
out_path = OUT_DIR / "33_fb_psweep.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nГрафик: {out_path}")
plt.show()
