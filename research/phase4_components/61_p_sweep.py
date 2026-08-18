"""
61 — Sweep размерности вектора задержек p (4…70) для итеративного LWR на LP-сигнале.

Наблюдение: при правильно подобранном p рядом с точкой отсчёта есть «окно» 5-15 баров,
где прогноз держит верную динамику; при больших p (~67) форма верная, но дрейф по Y;
мелкие p бессистемно болтаются.

Протокол (концептуальный запуск):
  2 тикера (SBER, LKOH), 1d, N_ORIG=30, horizon=20, wn=0.125.
  xi = 3*(p+1) для каждого p.

Выходы:
  A — LP-MAPE vs p (mean ± std по origins и тикерам)
  B — heatmap MAPE(origin_idx × p) для SBER
  C — 6 траекторий одного origin при p = [8, 20, 35, 50, 60, 67]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
FIG_DIR  = ROOT / "research/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS      = ["SBER", "LKOH"]
N_ORIG       = 30
HORIZON      = 20
FILTER_ORDER = 4
WN           = 0.125
P_RANGE      = range(4, 71)          # 4…70 включительно
TRAJ_TICKER  = "SBER"
TRAJ_P_LIST  = [8, 20, 35, 50, 60, 67]

_SOS_LP = butter(FILTER_ORDER, WN, btype="low", output="sos")


# ── утилиты ────────────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t); trend[:2] = close[:2]
    return trend


def _lwr_step(X: np.ndarray, y: np.ndarray, vec: np.ndarray, xi: int) -> float:
    dists = np.linalg.norm(X - vec, axis=1)
    nn    = np.argpartition(dists, xi)[:xi]
    h_bw  = max(float(dists[nn].max()), 1e-10)
    w     = np.exp(-0.5 * (dists[nn] / h_bw) ** 2)
    A     = np.hstack([np.ones((xi, 1)), X[nn]])
    sw    = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn], rcond=None)
    return float(c[0] + vec @ c[1:])


def forecast_lwr(att_hist: np.ndarray, p: int, horizon: int) -> np.ndarray:
    xi = 3 * (p + 1)
    n  = len(att_hist)
    X  = np.array([att_hist[i: i + p] for i in range(n - p)])
    y  = att_hist[p:]
    if len(X) < xi:
        return np.full(horizon, np.nan)
    vec = att_hist[-p:].copy()
    out = np.empty(horizon)
    for h in range(horizon):
        val = _lwr_step(X, y, vec, xi)
        out[h] = val
        vec = np.roll(vec, -1); vec[-1] = val
    return out


def lp_mape(att_pred: np.ndarray, att_actual: np.ndarray, ratio0: float) -> float:
    h = min(len(att_pred), len(att_actual))
    if h == 0 or not np.isfinite(att_pred[:h]).all():
        return np.nan
    r_pred = ratio0 + np.cumsum(att_pred[:h])
    r_act  = ratio0 + np.cumsum(att_actual[:h])
    return float(np.mean(np.abs(r_pred - r_act) / (np.abs(r_act) + 1e-10)))


def load_att(ticker: str):
    path = DATA_DIR / ticker / "1d.json"
    import json
    data = json.loads(path.read_text())
    close = np.array([c["close"] for c in data], dtype=np.float64)
    trend = logtrend_causal(close)
    ratio = close / np.maximum(trend, 1e-10)
    dratio = np.diff(ratio, prepend=ratio[0])
    att = sosfilt(_SOS_LP, dratio)
    return att, ratio


# ── основной цикл ──────────────────────────────────────────────────────────────

# mapes[ticker][p_idx] = list of LP-MAPE per origin
mapes = {t: {p: [] for p in P_RANGE} for t in TICKERS}

# heatmap для SBER: shape (N_ORIG, len(P_RANGE))
heatmap_mapes = np.full((N_ORIG, len(P_RANGE)), np.nan)

# траектории: один origin из SBER для выбранных p
traj_origin_k   = None
traj_att_actual = None
traj_ratio0     = None
traj_forecasts  = {}   # p -> att_pred

att_data = {}
ratio_data = {}
for ticker in TICKERS:
    att_data[ticker], ratio_data[ticker] = load_att(ticker)

p_max_all = max(P_RANGE)

for ticker in TICKERS:
    att_full  = att_data[ticker]
    ratio     = ratio_data[ticker]
    n_total   = len(att_full)
    min_start = p_max_all + 3 * (p_max_all + 1) + HORIZON + 10

    end_k   = n_total - HORIZON - 1
    start_k = max(min_start, end_k - N_ORIG * 3)
    candidates = list(range(start_k, end_k))
    step    = max(1, len(candidates) // N_ORIG)
    origins = candidates[::step][:N_ORIG]

    print(f"{ticker}: {len(origins)} origins, n={n_total}")

    for oi, origin_k in enumerate(origins):
        att_hist   = att_full[:origin_k]
        att_actual = att_full[origin_k: origin_k + HORIZON]
        ratio0     = float(ratio[origin_k])

        if ticker == TRAJ_TICKER and traj_origin_k is None:
            traj_origin_k   = origin_k
            traj_att_actual = att_actual.copy()
            traj_ratio0     = ratio0

        for pi, p in enumerate(P_RANGE):
            dhat = forecast_lwr(att_hist, p, HORIZON)
            m    = lp_mape(dhat, att_actual, ratio0)
            if np.isfinite(m):
                mapes[ticker][p].append(m)
            if ticker == TRAJ_TICKER:
                heatmap_mapes[oi, pi] = m

            if (ticker == TRAJ_TICKER
                    and origin_k == traj_origin_k
                    and p in TRAJ_P_LIST):
                traj_forecasts[p] = dhat.copy()

    print(f"  done")

# ── агрегация ──────────────────────────────────────────────────────────────────

p_list = list(P_RANGE)
agg_mean = np.array([
    np.mean([v for t in TICKERS for v in mapes[t][p]])
    for p in p_list
])
agg_std = np.array([
    np.std([v for t in TICKERS for v in mapes[t][p]])
    for p in p_list
])

best_idx = int(np.nanargmin(agg_mean))
print(f"\nЛучший p = {p_list[best_idx]}  (AGG LP-MAPE={agg_mean[best_idx]:.5f})")
print(f"baseline p=8 : AGG LP-MAPE={agg_mean[p_list.index(8)]:.5f}")
print(f"p=20         : AGG LP-MAPE={agg_mean[p_list.index(20)]:.5f}")
print(f"p=35         : AGG LP-MAPE={agg_mean[p_list.index(35)]:.5f}")
print(f"p=67         : AGG LP-MAPE={agg_mean[p_list.index(67)]:.5f}")

# ── рис. A — LP-MAPE vs p ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(p_list, agg_mean, lw=1.5, color="steelblue", label="mean LP-MAPE")
ax.fill_between(p_list,
                agg_mean - agg_std,
                agg_mean + agg_std,
                alpha=0.2, color="steelblue", label="±1 std")
ax.axvline(p_list[best_idx], color="green",  lw=1, ls="--",
           label=f"best p={p_list[best_idx]}")
ax.axvline(8,                color="orange", lw=1, ls=":",  label="p=8 (baseline)")
ax.set_xlabel("p (размерность вектора задержек)")
ax.set_ylabel("AGG LP-MAPE")
ax.set_title("LP-MAPE vs p  (LWR итеративный с пересчётом, 2 тикера, 30 origins)")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "61_mape_vs_p.png", dpi=150)
plt.close(fig)
print(f"Рис. A: {FIG_DIR}/61_mape_vs_p.png")

# ── рис. B — heatmap MAPE(origin × p) ────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(heatmap_mapes, aspect="auto", origin="lower",
               extent=[p_list[0], p_list[-1], 0, N_ORIG],
               vmax=np.nanpercentile(heatmap_mapes, 90))
plt.colorbar(im, ax=ax, label="LP-MAPE")
ax.set_xlabel("p")
ax.set_ylabel("origin index (хронологически)")
ax.set_title(f"Heatmap LP-MAPE(origin × p)  [{TRAJ_TICKER}]")
fig.tight_layout()
fig.savefig(FIG_DIR / "61_heatmap.png", dpi=150)
plt.close(fig)
print(f"Рис. B: {FIG_DIR}/61_heatmap.png")

# ── рис. C — траектории при разных p ─────────────────────────────────────────

if traj_forecasts:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
    axes = axes.flatten()

    ratio0 = traj_ratio0
    r_act  = ratio0 + np.cumsum(traj_att_actual[:HORIZON])
    h_ax   = np.arange(1, HORIZON + 1)

    for idx, p_v in enumerate(TRAJ_P_LIST):
        ax = axes[idx]
        ax.plot(h_ax, r_act, "k-", lw=1.5, label="actual", alpha=0.7)
        if p_v in traj_forecasts:
            fc  = traj_forecasts[p_v]
            r_fc = ratio0 + np.cumsum(fc[:HORIZON])
            m    = lp_mape(fc, traj_att_actual, ratio0)
            ax.plot(h_ax, r_fc, "b-", lw=1.5,
                    label=f"p={p_v}  MAPE={m:.4f}")
        ax.set_title(f"p = {p_v}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Траектории LP-ratio  [{TRAJ_TICKER}, один origin]", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "61_trajectories.png", dpi=150)
    plt.close(fig)
    print(f"Рис. C: {FIG_DIR}/61_trajectories.png")

print("\nСкрипт 61 завершён.")
