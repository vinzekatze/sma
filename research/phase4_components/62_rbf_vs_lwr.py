"""
62 — RBF vs LWR: сравнение аппроксиматоров в трёх режимах прогноза.

LWR (locally weighted linear regression):
  ŷ = c₀ + c·x   — линейная по x, Гауссовы веса

RBF (radial basis function):
  ŷ = Σ wᵢ·φ(||x - xᵢ||)   φ(r)=exp(-r²/2σ²)
  — нелинейна по x, следует кривизне многообразия аттрактора

Три режима (× 2 аппроксиматора = 6 методов):
  direct       — отдельная модель на каждый горизонт h, query vec фиксирован
  iter         — параметры один раз, на каждом шаге новый vec (без нового фита)
  iter_refit   — на каждом шаге новые соседи + новый фит

Протокол: SBER + LKOH, 1d, N_ORIG=20, horizon=20, p sweep 4…70, wn=0.125.
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
N_ORIG       = 20
HORIZON      = 20
FILTER_ORDER = 4
WN           = 0.125
P_RANGE      = range(4, 71)
RBF_REG      = 1e-6        # регуляризация для RBF системы
TRAJ_TICKER  = "SBER"
TRAJ_P_LIST  = [8, 20, 42, 67]

_SOS_LP = butter(FILTER_ORDER, WN, btype="low", output="sos")

METHODS = ["lwr_direct", "lwr_iter", "lwr_iter_refit",
           "rbf_direct", "rbf_iter", "rbf_iter_refit"]

METHOD_LABELS = {
    "lwr_direct":      "LWR прямой",
    "lwr_iter":        "LWR итер.",
    "lwr_iter_refit":  "LWR итер.+пересчёт",
    "rbf_direct":      "RBF прямой",
    "rbf_iter":        "RBF итер.",
    "rbf_iter_refit":  "RBF итер.+пересчёт",
}
METHOD_COLORS = {
    "lwr_direct":      "#4878cf",
    "lwr_iter":        "#6acc65",
    "lwr_iter_refit":  "#d65f5f",
    "rbf_direct":      "#b47cc7",
    "rbf_iter":        "#c4ad66",
    "rbf_iter_refit":  "#77bedb",
}


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


def _find_neighbors(X: np.ndarray, vec: np.ndarray, xi: int):
    dists = np.linalg.norm(X - vec, axis=1)
    nn    = np.argpartition(dists, xi)[:xi]
    h_bw  = max(float(dists[nn].max()), 1e-10)
    return nn, h_bw


# ── LWR ────────────────────────────────────────────────────────────────────────

def _lwr_fit(X_nn: np.ndarray, y_nn: np.ndarray, h_bw: float):
    """Взвешенный lstsq → линейные коэффициенты c."""
    xi = len(X_nn)
    dists = np.linalg.norm(X_nn - X_nn.mean(0), axis=1)  # веса от центроида
    # но правильнее — веса от query vec; передаём снаружи
    return None  # placeholder, используем _lwr_fit_vec


def _lwr_fit_vec(X_nn: np.ndarray, y_nn: np.ndarray, vec: np.ndarray, h_bw: float):
    xi = len(X_nn)
    dists = np.linalg.norm(X_nn - vec, axis=1)
    w  = np.exp(-0.5 * (dists / h_bw) ** 2)
    A  = np.hstack([np.ones((xi, 1)), X_nn])
    sw = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
    return c


def _lwr_predict_c(vec: np.ndarray, c: np.ndarray) -> float:
    return float(c[0] + vec @ c[1:])


# ── RBF ────────────────────────────────────────────────────────────────────────

def _rbf_fit(X_nn: np.ndarray, y_nn: np.ndarray, h_bw: float):
    """Решаем (Φ + λI)w = y, Φᵢⱼ = exp(-||xᵢ-xⱼ||²/2σ²), σ=h_bw."""
    xi = len(X_nn)
    diff  = X_nn[:, None, :] - X_nn[None, :, :]   # xi×xi×p
    dists2 = np.sum(diff ** 2, axis=2)
    Phi   = np.exp(-0.5 * dists2 / h_bw ** 2)
    lam   = RBF_REG * (np.trace(Phi) / xi + 1e-12)
    w, _, _, _ = np.linalg.lstsq(Phi + lam * np.eye(xi), y_nn, rcond=None)
    return w


def _rbf_predict(vec: np.ndarray, X_nn: np.ndarray, w: np.ndarray,
                 h_bw: float) -> float:
    dists2 = np.sum((X_nn - vec) ** 2, axis=1)
    phi    = np.exp(-0.5 * dists2 / h_bw ** 2)
    return float(phi @ w)


# ── шесть методов прогноза ────────────────────────────────────────────────────

def forecast_lwr_direct(att_hist: np.ndarray, p: int, horizon: int) -> np.ndarray:
    xi  = 3 * (p + 1)
    n   = len(att_hist)
    vec = att_hist[-p:].copy()
    out = np.empty(horizon)
    for h in range(1, horizon + 1):
        n_s = n - p - h
        if n_s < xi:
            out[h - 1:] = np.nan; break
        X = np.array([att_hist[i: i + p] for i in range(n_s)])
        y = att_hist[p + h - 1: p + h - 1 + n_s]
        nn, h_bw = _find_neighbors(X, vec, xi)
        c = _lwr_fit_vec(X[nn], y[nn], vec, h_bw)
        out[h - 1] = _lwr_predict_c(vec, c)
    return out


def forecast_lwr_iter(att_hist: np.ndarray, p: int, horizon: int) -> np.ndarray:
    """Параметры один раз; на каждом шаге новый vec, новые соседи, но старый c."""
    xi  = 3 * (p + 1)
    n   = len(att_hist)
    X   = np.array([att_hist[i: i + p] for i in range(n - p)])
    y   = att_hist[p:]
    if len(X) < xi:
        return np.full(horizon, np.nan)
    vec = att_hist[-p:].copy()
    nn0, h_bw0 = _find_neighbors(X, vec, xi)
    c   = _lwr_fit_vec(X[nn0], y[nn0], vec, h_bw0)
    out = np.empty(horizon)
    for h in range(horizon):
        out[h] = _lwr_predict_c(vec, c)
        vec = np.roll(vec, -1); vec[-1] = out[h]
    return out


def forecast_lwr_iter_refit(att_hist: np.ndarray, p: int, horizon: int) -> np.ndarray:
    xi  = 3 * (p + 1)
    n   = len(att_hist)
    X   = np.array([att_hist[i: i + p] for i in range(n - p)])
    y   = att_hist[p:]
    if len(X) < xi:
        return np.full(horizon, np.nan)
    vec = att_hist[-p:].copy()
    out = np.empty(horizon)
    for h in range(horizon):
        nn, h_bw = _find_neighbors(X, vec, xi)
        c = _lwr_fit_vec(X[nn], y[nn], vec, h_bw)
        out[h] = _lwr_predict_c(vec, c)
        vec = np.roll(vec, -1); vec[-1] = out[h]
    return out


def forecast_rbf_direct(att_hist: np.ndarray, p: int, horizon: int) -> np.ndarray:
    xi  = 3 * (p + 1)
    n   = len(att_hist)
    vec = att_hist[-p:].copy()
    out = np.empty(horizon)
    for h in range(1, horizon + 1):
        n_s = n - p - h
        if n_s < xi:
            out[h - 1:] = np.nan; break
        X = np.array([att_hist[i: i + p] for i in range(n_s)])
        y = att_hist[p + h - 1: p + h - 1 + n_s]
        nn, h_bw = _find_neighbors(X, vec, xi)
        w = _rbf_fit(X[nn], y[nn], h_bw)
        out[h - 1] = _rbf_predict(vec, X[nn], w, h_bw)
    return out


def forecast_rbf_iter(att_hist: np.ndarray, p: int, horizon: int) -> np.ndarray:
    """Центры и веса фиксированы от начального origin; vec меняется."""
    xi  = 3 * (p + 1)
    n   = len(att_hist)
    X   = np.array([att_hist[i: i + p] for i in range(n - p)])
    y   = att_hist[p:]
    if len(X) < xi:
        return np.full(horizon, np.nan)
    vec = att_hist[-p:].copy()
    nn0, h_bw0 = _find_neighbors(X, vec, xi)
    X_nn0 = X[nn0]; y_nn0 = y[nn0]
    w0    = _rbf_fit(X_nn0, y_nn0, h_bw0)
    out   = np.empty(horizon)
    for h in range(horizon):
        out[h] = _rbf_predict(vec, X_nn0, w0, h_bw0)
        vec = np.roll(vec, -1); vec[-1] = out[h]
    return out


def forecast_rbf_iter_refit(att_hist: np.ndarray, p: int, horizon: int) -> np.ndarray:
    xi  = 3 * (p + 1)
    n   = len(att_hist)
    X   = np.array([att_hist[i: i + p] for i in range(n - p)])
    y   = att_hist[p:]
    if len(X) < xi:
        return np.full(horizon, np.nan)
    vec = att_hist[-p:].copy()
    out = np.empty(horizon)
    for h in range(horizon):
        nn, h_bw = _find_neighbors(X, vec, xi)
        w = _rbf_fit(X[nn], y[nn], h_bw)
        out[h] = _rbf_predict(vec, X[nn], w, h_bw)
        vec = np.roll(vec, -1); vec[-1] = out[h]
    return out


FORECAST_FNS = {
    "lwr_direct":     forecast_lwr_direct,
    "lwr_iter":       forecast_lwr_iter,
    "lwr_iter_refit": forecast_lwr_iter_refit,
    "rbf_direct":     forecast_rbf_direct,
    "rbf_iter":       forecast_rbf_iter,
    "rbf_iter_refit": forecast_rbf_iter_refit,
}


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
    att    = sosfilt(_SOS_LP, dratio)
    return att, ratio


# ── основной цикл ──────────────────────────────────────────────────────────────

p_list = list(P_RANGE)

# mapes[method][p] = list of LP-MAPE (все тикеры вместе)
mapes = {m: {p: [] for p in p_list} for m in METHODS}

# траектории: один origin SBER, выбранные p
traj_origin_k   = None
traj_att_actual = None
traj_ratio0     = None
traj_fc = {p: {m: None for m in METHODS} for p in TRAJ_P_LIST}

att_data   = {}
ratio_data = {}
for ticker in TICKERS:
    att_data[ticker], ratio_data[ticker] = load_att(ticker)

p_max_all = max(p_list)

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

        for p in p_list:
            for method, fn in FORECAST_FNS.items():
                dhat = fn(att_hist, p, HORIZON)
                m    = lp_mape(dhat, att_actual, ratio0)
                if np.isfinite(m):
                    mapes[method][p].append(m)

                if (ticker == TRAJ_TICKER
                        and origin_k == traj_origin_k
                        and p in TRAJ_P_LIST):
                    traj_fc[p][method] = dhat.copy()

        if (oi + 1) % 5 == 0:
            print(f"  {oi+1}/{len(origins)}", flush=True)

    print(f"  {ticker} done", flush=True)

# ── агрегация ──────────────────────────────────────────────────────────────────

agg = {}
for method in METHODS:
    agg[method] = np.array([
        np.nanmean(mapes[method][p]) for p in p_list
    ])

print("\n── AGG LP-MAPE по методам ───────────────────────────────────────────────────")
print(f"{'Метод':<25}  {'best p':>7}  {'best MAPE':>10}  {'p=42 MAPE':>10}")
print("─" * 60)
for method in METHODS:
    a   = agg[method]
    bi  = int(np.nanargmin(a))
    bp  = p_list[bi]
    bm  = a[bi]
    m42 = agg[method][p_list.index(42)] if 42 in p_list else np.nan
    print(f"{METHOD_LABELS[method]:<25}  {bp:>7}  {bm:>10.5f}  {m42:>10.5f}")

# ── рис. A — LP-MAPE vs p для всех 6 методов ────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for method in ["lwr_direct", "lwr_iter", "lwr_iter_refit"]:
    axes[0].plot(p_list, agg[method], label=METHOD_LABELS[method],
                 color=METHOD_COLORS[method], lw=1.5)
axes[0].set_title("LWR: LP-MAPE vs p")
axes[0].set_xlabel("p"); axes[0].set_ylabel("AGG LP-MAPE")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

for method in ["rbf_direct", "rbf_iter", "rbf_iter_refit"]:
    axes[1].plot(p_list, agg[method], label=METHOD_LABELS[method],
                 color=METHOD_COLORS[method], lw=1.5)
axes[1].set_title("RBF: LP-MAPE vs p")
axes[1].set_xlabel("p"); axes[1].set_ylabel("AGG LP-MAPE")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

fig.suptitle("LP-MAPE vs p  (2 тикера, 20 origins)", fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "62_mape_vs_p.png", dpi=150)
plt.close(fig)
print(f"\nРис. A: {FIG_DIR}/62_mape_vs_p.png")

# ── рис. B — LWR vs RBF попарно по режимам ───────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
pairs = [("lwr_direct", "rbf_direct"),
         ("lwr_iter",   "rbf_iter"),
         ("lwr_iter_refit", "rbf_iter_refit")]
titles = ["Прямой", "Итеративный", "Итеративный + пересчёт"]

for ax, (lm, rm), title in zip(axes, pairs, titles):
    ax.plot(p_list, agg[lm], color=METHOD_COLORS[lm], lw=1.5,
            label=f"LWR ({agg[lm].min():.4f})")
    ax.plot(p_list, agg[rm], color=METHOD_COLORS[rm], lw=1.5, ls="--",
            label=f"RBF ({agg[rm].min():.4f})")
    ax.set_title(title); ax.set_xlabel("p")
    ax.set_ylabel("AGG LP-MAPE"); ax.legend(); ax.grid(True, alpha=0.3)

fig.suptitle("LWR vs RBF по режимам", fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / "62_lwr_vs_rbf.png", dpi=150)
plt.close(fig)
print(f"Рис. B: {FIG_DIR}/62_lwr_vs_rbf.png")

# ── рис. C — траектории одного origin, 4 значения p ─────────────────────────

if traj_origin_k is not None:
    fig, axes = plt.subplots(len(TRAJ_P_LIST), 1,
                              figsize=(12, 4 * len(TRAJ_P_LIST)), sharex=False)
    h_ax = np.arange(1, HORIZON + 1)
    r_act = traj_ratio0 + np.cumsum(traj_att_actual[:HORIZON])

    for ax, p_v in zip(axes, TRAJ_P_LIST):
        ax.plot(h_ax, r_act, "k-", lw=2, label="actual", alpha=0.8)
        for method in METHODS:
            fc = traj_fc[p_v][method]
            if fc is None or not np.isfinite(fc).all():
                continue
            r_fc = traj_ratio0 + np.cumsum(fc[:HORIZON])
            ls   = "--" if method.startswith("rbf") else "-"
            ax.plot(h_ax, r_fc, ls=ls, lw=1.2,
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method])
        ax.set_title(f"p = {p_v}")
        ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Траектории LP-ratio [{TRAJ_TICKER}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "62_trajectories.png", dpi=150)
    plt.close(fig)
    print(f"Рис. C: {FIG_DIR}/62_trajectories.png")

print("\nСкрипт 62 завершён.")
