"""
63 — Разделение размерностей: p_search (поиск соседей) и p_fit (аппроксимация).

Гипотеза: большой p_search правильно локализует соседей на аттракторе,
малый p_fit даёт устойчивую аппроксимацию без дрейфа.

Поиск соседей: vec_search = att[-p_search:]  →  xi = 3*(p_search+1) соседей
Аппроксимация: X_fit = X_search[nn, -p_fit:]  →  LWR на vec_fit = att[-p_fit:]

Режим: итеративный с пересчётом (лучший по скр.62).

Протокол: SBER + LKOH, 1d, N_ORIG=20, horizon=20, wn=0.125.
p_search ∈ {20, 42, 67}  ×  p_fit ∈ 4…p_search (с шагом 1).
Baseline: стандартный LWR (p_search=p_fit) при тех же p.
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

TICKERS       = ["SBER", "LKOH"]
N_ORIG        = 20
HORIZON       = 20
FILTER_ORDER  = 4
WN            = 0.125
P_SEARCH_LIST = [20, 42, 67]
P_FIT_MIN     = 4
TRAJ_TICKER   = "SBER"
TRAJ_P_SEARCH = 42
TRAJ_P_FIT_LIST = [4, 8, 15, 20, 30, 42]   # p_fit при фиксированном p_search=42

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


def lp_mape(att_pred: np.ndarray, att_actual: np.ndarray, ratio0: float) -> float:
    h = min(len(att_pred), len(att_actual))
    if h == 0 or not np.isfinite(att_pred[:h]).all():
        return np.nan
    r_pred = ratio0 + np.cumsum(att_pred[:h])
    r_act  = ratio0 + np.cumsum(att_actual[:h])
    return float(np.mean(np.abs(r_pred - r_act) / (np.abs(r_act) + 1e-10)))


def load_att(ticker: str):
    import json
    data   = json.loads((DATA_DIR / ticker / "1d.json").read_text())
    close  = np.array([c["close"] for c in data], dtype=np.float64)
    trend  = logtrend_causal(close)
    ratio  = close / np.maximum(trend, 1e-10)
    dratio = np.diff(ratio, prepend=ratio[0])
    return sosfilt(_SOS_LP, dratio), ratio


def _lwr_fit_predict(X_nn: np.ndarray, y_nn: np.ndarray,
                     vec: np.ndarray, h_bw: float) -> float:
    """Взвешенный lstsq → предсказание."""
    xi = len(X_nn)
    dists = np.linalg.norm(X_nn - vec, axis=1)
    w  = np.exp(-0.5 * (dists / h_bw) ** 2)
    A  = np.hstack([np.ones((xi, 1)), X_nn])
    sw = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return float(c[0] + vec @ c[1:])


def forecast(att_hist: np.ndarray, p_search: int, p_fit: int,
             horizon: int) -> np.ndarray:
    """
    Итеративный LWR с пересчётом.
    Соседи ищутся в p_search-мерном пространстве,
    аппроксимация строится в p_fit-мерном (хвост вектора задержек).
    """
    assert p_fit <= p_search
    xi  = 3 * (p_search + 1)
    n   = len(att_hist)

    # матрица задержек в полной размерности
    X_full = np.array([att_hist[i: i + p_search] for i in range(n - p_search)])
    y_full = att_hist[p_search:]

    if len(X_full) < xi:
        return np.full(horizon, np.nan)

    # начальные векторы
    vec_search = att_hist[-p_search:].copy()
    vec_fit    = att_hist[-p_fit:].copy()

    out = np.empty(horizon)
    for h in range(horizon):
        # поиск соседей в высокой размерности
        dists = np.linalg.norm(X_full - vec_search, axis=1)
        nn    = np.argpartition(dists, xi)[:xi]
        h_bw  = max(float(dists[nn].max()), 1e-10)

        # аппроксимация в малой размерности (хвост p_fit элементов)
        X_nn_fit = X_full[nn, -p_fit:]
        y_nn     = y_full[nn]

        val = _lwr_fit_predict(X_nn_fit, y_nn, vec_fit, h_bw)
        out[h] = val

        # обновляем оба вектора
        vec_search = np.roll(vec_search, -1); vec_search[-1] = val
        vec_fit    = np.roll(vec_fit,    -1); vec_fit[-1]    = val

    return out


# ── основной цикл ──────────────────────────────────────────────────────────────

att_data, ratio_data = {}, {}
for ticker in TICKERS:
    att_data[ticker], ratio_data[ticker] = load_att(ticker)

p_max_all = max(P_SEARCH_LIST)

# results[p_search][p_fit] = list of LP-MAPE
results   = {ps: {} for ps in P_SEARCH_LIST}
# baseline: стандартный LWR p_search=p_fit
baseline  = {}   # p -> list of LP-MAPE

# траектории
traj_origin_k   = None
traj_att_actual = None
traj_ratio0     = None
traj_fc = {pf: None for pf in TRAJ_P_FIT_LIST}   # p_fit → att_pred

for ticker in TICKERS:
    att_full = att_data[ticker]
    ratio    = ratio_data[ticker]
    n_total  = len(att_full)
    min_start = p_max_all + 3 * (p_max_all + 1) + HORIZON + 10

    end_k      = n_total - HORIZON - 1
    start_k    = max(min_start, end_k - N_ORIG * 5)
    candidates = list(range(start_k, end_k))
    step       = max(1, len(candidates) // N_ORIG)
    origins    = candidates[::step][:N_ORIG]

    print(f"{ticker}: {len(origins)} origins", flush=True)

    for oi, origin_k in enumerate(origins):
        att_hist   = att_full[:origin_k]
        att_actual = att_full[origin_k: origin_k + HORIZON]
        ratio0     = float(ratio[origin_k])

        if ticker == TRAJ_TICKER and traj_origin_k is None:
            traj_origin_k   = origin_k
            traj_att_actual = att_actual.copy()
            traj_ratio0     = ratio0

        for p_search in P_SEARCH_LIST:
            p_fit_range = range(P_FIT_MIN, p_search + 1)

            for p_fit in p_fit_range:
                dhat = forecast(att_hist, p_search, p_fit, HORIZON)
                m    = lp_mape(dhat, att_actual, ratio0)
                if p_fit not in results[p_search]:
                    results[p_search][p_fit] = []
                if np.isfinite(m):
                    results[p_search][p_fit].append(m)

                # baseline: p_search == p_fit (стандартный LWR)
                if p_search == p_fit:
                    if p_fit not in baseline:
                        baseline[p_fit] = []
                    if np.isfinite(m):
                        baseline[p_fit].append(m)

                # траектории
                if (ticker == TRAJ_TICKER
                        and origin_k == traj_origin_k
                        and p_search == TRAJ_P_SEARCH
                        and p_fit in TRAJ_P_FIT_LIST):
                    traj_fc[p_fit] = dhat.copy()

        if (oi + 1) % 5 == 0:
            print(f"  {oi+1}/{len(origins)}", flush=True)

    print(f"  {ticker} done", flush=True)

# ── агрегация и вывод ──────────────────────────────────────────────────────────

print("\n── AGG LP-MAPE: лучшие p_fit для каждого p_search ───────────────────────────")
print(f"{'p_search':>9}  {'best p_fit':>10}  {'best MAPE':>10}  "
      f"{'baseline(p=p_search)':>22}  {'выигрыш':>8}")
print("─" * 70)

gains = {}
for ps in P_SEARCH_LIST:
    p_fits = sorted(results[ps].keys())
    means  = np.array([np.nanmean(results[ps][pf]) for pf in p_fits])
    bi     = int(np.nanargmin(means))
    best_pf = p_fits[bi]
    best_m  = means[bi]

    # baseline: стандартный LWR при p = ps (p_search=p_fit=ps)
    if ps in baseline and baseline[ps]:
        base_m = np.nanmean(baseline[ps])
    else:
        base_m = np.nan
    gain = (base_m / best_m - 1.0) * 100 if np.isfinite(base_m) else np.nan
    gains[ps] = (best_pf, best_m, base_m, gain)
    print(f"{ps:>9}  {best_pf:>10}  {best_m:>10.5f}  {base_m:>22.5f}  {gain:>7.1f}%")

# ── рис. A — MAPE(p_fit) для каждого p_search ────────────────────────────────

colors = {20: "#4878cf", 42: "#d65f5f", 67: "#6acc65"}

fig, axes = plt.subplots(1, len(P_SEARCH_LIST), figsize=(15, 5), sharey=True)
for ax, ps in zip(axes, P_SEARCH_LIST):
    p_fits = sorted(results[ps].keys())
    means  = [np.nanmean(results[ps][pf]) for pf in p_fits]
    ax.plot(p_fits, means, color=colors[ps], lw=1.5, label=f"p_search={ps}")

    # baseline: diagonal p_fit=p_search (только если есть данные)
    bl_pts = [(pf, np.nanmean(baseline[pf]))
              for pf in p_fits if pf in baseline and baseline[pf]]
    if bl_pts:
        bx, by = zip(*bl_pts)
        ax.plot(bx, by, "k--", lw=1, alpha=0.5, label="baseline (p_fit=p_search)")

    best_pf, best_m, _, _ = gains[ps]
    ax.axvline(best_pf, color=colors[ps], lw=1, ls=":", alpha=0.7,
               label=f"best p_fit={best_pf}")
    ax.set_title(f"p_search = {ps}")
    ax.set_xlabel("p_fit")
    ax.set_ylabel("AGG LP-MAPE")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle("LP-MAPE vs p_fit при фиксированном p_search  "
             "(LWR итер.+пересчёт, 2 тикера, 20 origins)", fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "63_mape_vs_pfit.png", dpi=150)
plt.close(fig)
print(f"\nРис. A: {FIG_DIR}/63_mape_vs_pfit.png")

# ── рис. B — heatmap MAPE(p_search × p_fit) для p_search=42 ──────────────────

ps42 = 42
p_fits_42 = sorted(results[ps42].keys())
means_42  = np.array([np.nanmean(results[ps42][pf]) for pf in p_fits_42])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(p_fits_42, means_42, color=colors[42], lw=1.5)
ax.axvline(gains[42][0], color="red", lw=1, ls="--",
           label=f"best p_fit={gains[42][0]}")
ax.axhline(gains[42][2], color="gray", lw=1, ls=":",
           label=f"baseline p={ps42}: {gains[42][2]:.5f}")
ax.set_xlabel("p_fit"); ax.set_ylabel("AGG LP-MAPE")
ax.set_title(f"p_search=42: MAPE vs p_fit  (выигрыш {gains[42][3]:.1f}%)")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "63_pfit_at42.png", dpi=150)
plt.close(fig)
print(f"Рис. B: {FIG_DIR}/63_pfit_at42.png")

# ── рис. C — траектории одного origin, p_search=42, разные p_fit ─────────────

if traj_origin_k is not None:
    h_ax  = np.arange(1, HORIZON + 1)
    r_act = traj_ratio0 + np.cumsum(traj_att_actual[:HORIZON])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    axes = axes.flatten()
    cmap = plt.cm.viridis(np.linspace(0, 1, len(TRAJ_P_FIT_LIST)))

    for idx, (p_fit, color) in enumerate(zip(TRAJ_P_FIT_LIST, cmap)):
        ax  = axes[idx]
        fc  = traj_fc.get(p_fit)
        ax.plot(h_ax, r_act, "k-", lw=1.5, label="actual", alpha=0.8)
        if fc is not None and np.isfinite(fc).all():
            r_fc = traj_ratio0 + np.cumsum(fc[:HORIZON])
            m    = lp_mape(fc, traj_att_actual, traj_ratio0)
            ax.plot(h_ax, r_fc, color=color, lw=1.5,
                    label=f"p_fit={p_fit}  MAPE={m:.4f}")
        ax.set_title(f"p_search=42, p_fit={p_fit}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Траектории LP-ratio [{TRAJ_TICKER}]  "
                 f"p_search=42, разные p_fit", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "63_trajectories.png", dpi=150)
    plt.close(fig)
    print(f"Рис. C: {FIG_DIR}/63_trajectories.png")

print("\nСкрипт 63 завершён.")
