"""
60 — Коррекция траектории в фазовом пространстве для итеративного LWR

Проверяем три гипотезы о том, почему простое усреднение bias не работает,
и соответствующие решения:

  H1a: окно слишком широкое (τ=50) → попробовать τ=20
  H1b: нужна весовая функция по времени (недавние t важнее)
  H2:  bias зависит от того, где именно находился vec_t в фазовом пространстве
       → взвешивать по близости vec_t к текущему vec_origin (LWR в пространстве ошибок)
  H3:  шум в bias[h] по h → сгладить полиномом

Конфигурации:
  0. baseline (без коррекции)
  1. mean bias, τ=50, полный вектор
  2. mean bias, τ=50, только скаляр [-1] (проверка: важна ли полная размерность?)
  3. mean bias, τ=20 (H1a)
  4. time-weighted bias, τ=50 (H1b)
  5. phase-weighted bias, τ=50 (H2) — взвес по ||vec_t - vec_origin||
  6. phase-weighted + poly smooth H=2, τ=50 (H2+H3)

Метрика: LP-MAPE = MAPE по ratio_lp = ratio[T] + cumsum(att_pred)
         Сравниваем с ratio_lp_actual = ratio[T] + cumsum(att_actual)
         Только LP сигнал, без шума C0-C2.

Протокол: 8 тикеров, 1d, N_ORIG=200, horizon=15, p=8, xi=27, wn=0.125.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.stats import ttest_rel, pearsonr

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
FIG_DIR  = ROOT / "research/figures"
OUT_DIR  = Path(__file__).parent / "output60"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
N_ORIG       = 200
HORIZON      = 15
FILTER_ORDER = 4
WN           = 0.125
P            = 8
XI           = 3 * (P + 1)    # 27
TAU_WIDE     = 50
TAU_NARROW   = 20
TIME_DECAY   = 0.08            # λ для exp(-λ*(T-t))
POLY_DEGREE  = 2               # степень полинома сглаживания bias по h

_SOS_LP = butter(FILTER_ORDER, WN, btype="low", output="sos")

CONFIGS = [
    {"name": "baseline",                      "mode": "none"},
    {"name": "mean  τ=50  vec",               "mode": "mean",   "tau": TAU_WIDE,   "scalar": False},
    {"name": "mean  τ=50  scalar",            "mode": "mean",   "tau": TAU_WIDE,   "scalar": True},
    {"name": "mean  τ=20  vec    (H1a)",      "mode": "mean",   "tau": TAU_NARROW, "scalar": False},
    {"name": "time  τ=50  vec    (H1b)",      "mode": "time",   "tau": TAU_WIDE,   "scalar": False},
    {"name": "phase τ=50  vec    (H2)",       "mode": "phase",  "tau": TAU_WIDE,   "scalar": False},
    {"name": "phase τ=50  smooth (H2+H3)",    "mode": "phase",  "tau": TAU_WIDE,   "scalar": False,
     "smooth_poly": POLY_DEGREE},
]

# ── вспомогательные функции ────────────────────────────────────────────────────

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
    dists  = np.linalg.norm(X - vec, axis=1)
    nn     = np.argpartition(dists, xi)[:xi]
    h_bw   = max(float(dists[nn].max()), 1e-10)
    w      = np.exp(-0.5 * (dists[nn] / h_bw) ** 2)
    A      = np.hstack([np.ones((xi, 1)), X[nn]])
    sw     = np.sqrt(w)
    c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y[nn], rcond=None)
    return float(c[0] + vec @ c[1:])


def _build_Xy(att: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(att)
    return np.array([att[i: i + p] for i in range(n - p)]), att[p:]


def lp_mape(att_pred: np.ndarray, att_actual: np.ndarray, ratio0: float) -> float:
    h = min(len(att_pred), len(att_actual))
    if h == 0:
        return float("nan")
    r_pred = ratio0 + np.cumsum(att_pred[:h])
    r_act  = ratio0 + np.cumsum(att_actual[:h])
    return float(np.mean(np.abs(r_pred - r_act) / (np.abs(r_act) + 1e-10)))


# ── сбор сырых данных о bias ──────────────────────────────────────────────────

def collect_bias_data(
    att: np.ndarray, X: np.ndarray, y: np.ndarray,
    p: int, xi: int, origin_k: int, tau: int, horizon: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """
    Для каждой точки t в локальном окне запускает h-шаговый LWR
    и собирает (t, vec_t, deltas) где deltas[h] = vec_pred_h - vec_true_h.

    Использует X из att[:origin_k] — быстро, но слегка оптимистично.
    """
    t_end   = origin_k - horizon - 1
    t_start = max(p + 1, origin_k - tau - horizon)
    if t_start > t_end:
        return []

    result: list[tuple[int, np.ndarray, np.ndarray]] = []
    for t in range(t_start, t_end + 1):
        if t < p:
            continue
        vec_t = att[t - p: t].copy()
        vec   = vec_t.copy()
        deltas: list[np.ndarray] = []
        ok = True
        for h in range(1, horizon + 1):
            val      = _lwr_step(X, y, vec, xi)
            vec_raw  = np.roll(vec, -1); vec_raw[-1] = val
            t_h      = t + h
            if t_h >= len(att):
                ok = False; break
            vec_true = att[t_h - p: t_h]
            if len(vec_true) < p:
                ok = False; break
            deltas.append(vec_raw - vec_true)
            vec = vec_raw   # предсказанная траектория (не истинная)
        if ok and len(deltas) == horizon:
            result.append((t, vec_t, np.array(deltas)))   # deltas: (H, p)
    return result


def compute_bias(
    data: list[tuple[int, np.ndarray, np.ndarray]],
    origin_k: int, p: int, horizon: int,
    vec_origin: np.ndarray,
    mode: str,
    smooth_poly: int = 0,
) -> np.ndarray:
    """
    Вычисляет взвешенный bias[h] (horizon, p) из сырых данных.

    mode="mean"  — равные веса
    mode="time"  — exp(-λ*(T-t))
    mode="phase" — exp(-||vec_t - vec_origin||² / (2·σ²))
                   σ = медиана расстояний

    smooth_poly > 0 — дополнительное сглаживание по h полиномом.
    """
    if not data:
        return np.zeros((horizon, p))

    ts      = np.array([d[0] for d in data])
    vec_ts  = np.array([d[1] for d in data])   # (n, p)
    deltas  = np.array([d[2] for d in data])   # (n, H, p)
    n       = len(data)

    if mode == "mean":
        weights = np.ones(n)
    elif mode == "time":
        weights = np.exp(-TIME_DECAY * (origin_k - ts))
    elif mode == "phase":
        dists = np.linalg.norm(vec_ts - vec_origin, axis=1)
        bw    = max(float(np.median(dists)), 1e-10)
        weights = np.exp(-0.5 * (dists / bw) ** 2)
    else:
        weights = np.ones(n)

    w_sum = weights.sum()
    if w_sum < 1e-10:
        return np.zeros((horizon, p))
    weights /= w_sum

    # weighted mean: (H, p)
    bias = np.einsum("t,thp->hp", weights, deltas)

    if smooth_poly > 0:
        h_vals = np.arange(1, horizon + 1, dtype=float)
        for j in range(p):
            coeffs   = np.polyfit(h_vals, bias[:, j], smooth_poly)
            bias[:, j] = np.polyval(coeffs, h_vals)

    return bias


# ── прогноз с коррекцией ──────────────────────────────────────────────────────

def forecast_one(
    att: np.ndarray, X: np.ndarray, y: np.ndarray,
    p: int, xi: int, horizon: int,
    bias: np.ndarray | None, scalar_only: bool,
) -> np.ndarray:
    """
    Итеративный LWR с опциональной коррекцией вектора задержек.
    bias=None → baseline.
    scalar_only=True → корректируем только выходное значение ([-1] компонент),
                        вектор для следующего шага не меняем.
    scalar_only=False → корректируем весь вектор, кормим скорректированный дальше.
    """
    if len(X) < xi:
        return np.zeros(horizon)
    vec = att[-p:].copy()
    out = np.empty(horizon)
    for h in range(horizon):
        val     = _lwr_step(X, y, vec, xi)
        vec_raw = np.roll(vec, -1); vec_raw[-1] = val

        if bias is None:
            vec    = vec_raw
            out[h] = vec[-1]
        elif scalar_only:
            out[h] = vec_raw[-1] - bias[h, -1]   # только скалярная поправка
            vec    = vec_raw                       # вектор не трогаем
        else:
            vec    = vec_raw - bias[h]             # полный вектор
            out[h] = vec[-1]
    return out


# ── walk-forward ───────────────────────────────────────────────────────────────

n_cfg = len(CONFIGS)
# mapes[cfg_idx][ticker] = list of LP-MAPE values
mapes: list[dict] = [{t: [] for t in TICKERS} for _ in range(n_cfg)]

# для диагностики: сохраняем SNR по h (только H2 phase-config idx=5)
snr_records: list[np.ndarray] = []          # (horizon,) per origin
phase_dist_records: list[float] = []        # средняя дистанция в фаз. пространстве
d_mape_phase: list[float] = []              # Δ MAPE для phase config vs baseline

TRAJ_TICKER = "SBER"
N_TRAJ = 4
traj_data: list[dict] = []
traj_count = 0

t0     = time.time()
n_done = 0

for ticker in TICKERS:
    with open(DATA_DIR / ticker / "1d.json") as f:
        candles = json.load(f)
    close = np.array([float(c["close"]) for c in candles], dtype=np.float64)
    trend = logtrend_causal(close)
    ratio = close / trend
    N     = len(ratio)

    dratio_full = np.diff(ratio)
    att_full    = sosfilt(_SOS_LP, dratio_full)

    min_ok = max(P + XI + TAU_WIDE + HORIZON + 5, N - 700)
    max_ok = N - 2 - HORIZON
    if max_ok <= min_ok:
        continue
    origins = np.unique(np.linspace(min_ok, max_ok, N_ORIG, dtype=int))

    traj_count_ticker = 0

    for origin_k in origins:
        att_hist = att_full[:origin_k]
        X_full, y_full = _build_Xy(att_hist, P)
        if len(X_full) < XI:
            continue

        att_actual = att_full[origin_k: origin_k + HORIZON]
        if len(att_actual) < HORIZON:
            continue
        ratio0     = float(ratio[origin_k])
        vec_origin = att_hist[-P:].copy()

        # ── сбор bias-данных один раз (макс. окно) ────────────────────────────
        raw_wide   = collect_bias_data(att_hist, X_full, y_full, P, XI,
                                        origin_k, TAU_WIDE, HORIZON)
        raw_narrow = collect_bias_data(att_hist, X_full, y_full, P, XI,
                                        origin_k, TAU_NARROW, HORIZON)

        # phase-space SNR для диагностики
        if raw_wide:
            vec_ts  = np.array([d[1] for d in raw_wide])
            dists   = np.linalg.norm(vec_ts - vec_origin, axis=1)
            bw      = max(float(np.median(dists)), 1e-10)
            weights = np.exp(-0.5 * (dists / bw) ** 2)
            weights /= weights.sum()
            deltas  = np.array([d[2] for d in raw_wide])   # (n, H, p)
            b_phase = np.einsum("t,thp->hp", weights, deltas)
            b_std   = np.sqrt(np.einsum("t,thp->hp",
                              weights, (deltas - b_phase[None]) ** 2) + 1e-20)
            snr_h   = (np.linalg.norm(b_phase, axis=1)
                       / (np.linalg.norm(b_std, axis=1) + 1e-12))
            snr_records.append(snr_h)
            phase_dist_records.append(float(dists.mean()))

        # ── прогноз каждого конфига ───────────────────────────────────────────
        results_fc: list[np.ndarray] = []

        for ci, cfg in enumerate(CONFIGS):
            if cfg["mode"] == "none":
                bias = None; scalar = False
            else:
                raw   = raw_narrow if cfg["tau"] == TAU_NARROW else raw_wide
                bias  = compute_bias(raw, origin_k, P, HORIZON, vec_origin,
                                      cfg["mode"],
                                      smooth_poly=cfg.get("smooth_poly", 0))
                scalar = cfg.get("scalar", False)

            dhat = forecast_one(att_hist, X_full, y_full, P, XI, HORIZON,
                                 bias, scalar)
            results_fc.append(dhat)
            m = lp_mape(dhat, att_actual, ratio0)
            if np.isfinite(m):
                mapes[ci][ticker].append(m)

        # Δ MAPE phase vs baseline
        if len(results_fc) >= 6:
            m_base  = lp_mape(results_fc[0], att_actual, ratio0)
            m_phase = lp_mape(results_fc[5], att_actual, ratio0)
            if np.isfinite(m_base) and np.isfinite(m_phase):
                d_mape_phase.append(m_phase - m_base)

        # траектории SBER
        if ticker == TRAJ_TICKER and traj_count_ticker < N_TRAJ:
            traj_data.append({
                "origin_k":  origin_k,
                "att_actual": att_actual.copy(),
                "ratio0":     ratio0,
                "forecasts":  {cfg["name"]: fc.copy()
                               for cfg, fc in zip(CONFIGS, results_fc)},
            })
            traj_count_ticker += 1

        n_done += 1
        if n_done % 200 == 0:
            elapsed = time.time() - t0
            total   = len(TICKERS) * len(origins)
            print(f"  {n_done}/{total}  ({elapsed:.0f}s)")

print(f"\nГотово за {time.time() - t0:.1f}с")

# ── агрегация ──────────────────────────────────────────────────────────────────

base_flat = [v for t in TICKERS for v in mapes[0][t]]
agg_base  = float(np.mean(base_flat))

print(f"\n── AGG LP-MAPE ──────────────────────────────────────────────────────────────")
print(f"{'#':<2}  {'Конфигурация':<36}  {'AGG MAPE':>9}  {'Δ% base':>9}  p-value")
print("─" * 72)

for ci, cfg in enumerate(CONFIGS):
    flat = [v for t in TICKERS for v in mapes[ci][t]]
    if not flat:
        continue
    agg  = float(np.mean(flat))
    d    = (agg / agg_base - 1.0) * 100.0 if ci > 0 else 0.0
    if ci > 0 and len(flat) == len(base_flat):
        _, pval = ttest_rel(flat, base_flat)
        p_str = f"p={pval:.4f}"
    else:
        p_str = "—"
    sign = "✅" if d < -0.3 else ("❌" if d > 0.3 else "~")
    print(f"{ci:<2}  {cfg['name']:<36}  {agg:.5f}   {d:>+7.2f}%  {sign}  {p_str}")

# лучший конфиг по AGG
best_ci = min(range(1, n_cfg),
              key=lambda i: np.mean([v for t in TICKERS for v in mapes[i][t]])
              if any(mapes[i].values()) else float("inf"))
print(f"\nЛучший non-baseline: #{best_ci} «{CONFIGS[best_ci]['name']}»")

# per-ticker для лучшего
print(f"\n── Per-ticker: #{best_ci} vs baseline ───────────────────────────────────────")
print(f"{'Ticker':<6}  {'baseline':>9}  {'best_corr':>10}  {'Δ%':>8}")
for ticker in TICKERS:
    bm = np.mean(mapes[0][ticker])     if mapes[0][ticker]      else float("nan")
    cm = np.mean(mapes[best_ci][ticker]) if mapes[best_ci][ticker] else float("nan")
    d  = (cm / bm - 1) * 100 if np.isfinite(bm) and np.isfinite(cm) else float("nan")
    print(f"{ticker:<6}  {bm:.5f}   {cm:.5f}    {d:>+7.2f}%")

# ── диагностика SNR phase ─────────────────────────────────────────────────────

if snr_records:
    snr_arr  = np.array(snr_records)    # (N_runs, horizon)
    snr_mean = snr_arr.mean(axis=0)
    snr_med  = np.median(snr_arr, axis=0)
    print(f"\n── Phase-weighted SNR по горизонтам ─────────────────────────────────────────")
    print(f"{'h':>3}  {'SNR_mean':>9}  {'SNR_med':>8}")
    for h in range(HORIZON):
        print(f"{h+1:>3}  {snr_mean[h]:9.3f}  {snr_med[h]:8.3f}")

# ── рисунки ───────────────────────────────────────────────────────────────────

h_ax = np.arange(1, HORIZON + 1)

# ── Рис. A: barh AGG MAPE по конфигурациям ────────────────────────────────────

aggs   = [float(np.mean([v for t in TICKERS for v in mapes[ci][t]]))
          for ci in range(n_cfg) if any(mapes[ci].values())]
names  = [CONFIGS[ci]["name"] for ci in range(n_cfg) if any(mapes[ci].values())]
colors = []
for ci in range(n_cfg):
    if not any(mapes[ci].values()):
        continue
    if ci == 0:
        colors.append("#78909c")
    else:
        agg_ = float(np.mean([v for t in TICKERS for v in mapes[ci][t]]))
        d_   = (agg_ / agg_base - 1) * 100
        colors.append("#43a047" if d_ < -0.3 else ("#e53935" if d_ > 0.3 else "#ffa726"))

fig_a, ax_a = plt.subplots(figsize=(13, 5))
bars = ax_a.barh(names, aggs, color=colors, alpha=0.85, height=0.65)
ax_a.axvline(agg_base, color="#546e7a", lw=1.5, ls="--", label=f"baseline {agg_base:.5f}")
for bar, agg_, ci in zip(bars, aggs, range(n_cfg)):
    d_ = (agg_ / agg_base - 1) * 100 if ci > 0 else 0.0
    lbl = f"  {d_:+.2f}%" if ci > 0 else "  baseline"
    ax_a.text(bar.get_width() + 3e-6, bar.get_y() + bar.get_height() / 2,
              lbl, va="center", fontsize=8)
ax_a.set_xlabel("AGG LP-MAPE")
ax_a.set_title("60: Коррекция траектории в фазовом пространстве\n"
               "8 тикеров 1d N=200  |  LP-MAPE  |  зелёный = улучшение")
ax_a.legend(fontsize=9)
fig_a.tight_layout()
fig_a.savefig(FIG_DIR / "60_mape_configs.png", dpi=150)
print(f"\nРис. A: {FIG_DIR / '60_mape_configs.png'}")

# ── Рис. B: SNR phase vs h + распределение дистанций ─────────────────────────

if snr_records:
    snr_arr = np.array(snr_records)
    fig_b, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(13, 4))

    ax_b1.fill_between(h_ax, np.percentile(snr_arr, 25, axis=0),
                        np.percentile(snr_arr, 75, axis=0), alpha=0.2, color="#ff9800")
    ax_b1.plot(h_ax, snr_arr.mean(axis=0), "o-", color="#ff9800", lw=2, ms=6,
               label="SNR mean")
    ax_b1.plot(h_ax, np.median(snr_arr, axis=0), "s--", color="#e65100", lw=1.3,
               ms=4, label="SNR median")
    ax_b1.axhline(1.0, color="#aaa", lw=1, ls=":", label="SNR=1")
    ax_b1.set_xlabel("Горизонт h"); ax_b1.set_ylabel("SNR = ‖bias_mean‖ / ‖bias_std‖")
    ax_b1.set_title("60: Устойчивость phase-weighted bias по h")
    ax_b1.legend(fontsize=9); ax_b1.grid(alpha=0.3)

    ax_b2.hist(phase_dist_records, bins=40, color="#42a5f5", alpha=0.8, edgecolor="none")
    ax_b2.set_xlabel("Средняя дистанция vec_t → vec_origin (фаз. пространство)")
    ax_b2.set_ylabel("Число origin")
    ax_b2.set_title("60: Разброс калибровочных точек в фазовом пространстве")
    ax_b2.grid(alpha=0.3)
    fig_b.tight_layout()
    fig_b.savefig(FIG_DIR / "60_snr_phase.png", dpi=150)
    print(f"Рис. B: {FIG_DIR / '60_snr_phase.png'}")

# ── Рис. C: scatter — дистанция в фаз. пространстве vs Δ MAPE (phase config) ──

if d_mape_phase and phase_dist_records:
    d_arr  = np.array(d_mape_phase)
    pd_arr = np.array(phase_dist_records[:len(d_arr)])
    r, p_r = pearsonr(pd_arr, d_arr)
    pct_imp = float((d_arr < 0).mean() * 100)

    fig_c, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(13, 4))

    sc = ax_c1.scatter(pd_arr, d_arr, c=d_arr, cmap="RdYlGn_r",
                       vmin=np.percentile(d_arr, 5), vmax=np.percentile(d_arr, 95),
                       alpha=0.3, s=8)
    ax_c1.axhline(0, color="#aaa", lw=1, ls="--")
    ax_c1.set_xlabel("Средняя дистанция в фазовом пространстве (τ-окно)")
    ax_c1.set_ylabel("Δ LP-MAPE  phase corr − baseline  (< 0 = улучшение)")
    ax_c1.set_title(f"H2: Помогает ли близость в фаз. пространстве?\nr={r:.3f}  p={p_r:.3f}")
    plt.colorbar(sc, ax=ax_c1, label="Δ MAPE")
    ax_c1.grid(alpha=0.3)

    ax_c2.hist(d_arr, bins=60, color="#42a5f5", alpha=0.8, edgecolor="none")
    ax_c2.axvline(0, color="#e53935", lw=1.5, ls="--")
    ax_c2.axvline(d_arr.mean(), color="#43a047", lw=2, label=f"mean={d_arr.mean():.5f}")
    q25, q75 = np.percentile(d_arr, [25, 75])
    ax_c2.axvspan(q25, q75, alpha=0.15, color="#43a047",
                  label=f"IQR [{q25:.4f},{q75:.4f}]")
    ax_c2.set_xlabel("Δ LP-MAPE (phase − baseline)")
    ax_c2.set_ylabel("Число origin")
    ax_c2.set_title(f"H2: {pct_imp:.1f}% origin улучшились (phase-weighted)")
    ax_c2.legend(fontsize=9); ax_c2.grid(alpha=0.3)
    fig_c.tight_layout()
    fig_c.savefig(FIG_DIR / "60_phase_scatter.png", dpi=150)
    print(f"Рис. C: {FIG_DIR / '60_phase_scatter.png'}")

# ── Рис. D: структура bias в фаз. пространстве (тепловая карта по конфигу) ────

# Собираем bias для одного SBER origin для иллюстрации
_sber_recs = [r for r in [
    {"origin_k": traj["origin_k"], **traj} for traj in traj_data
] if True]

if traj_data:
    # выбираем один origin — последний из SBER
    _td = traj_data[-1]
    _ok = _td["origin_k"]
    _att = att_full[:_ok]    # att_full здесь относится к последнему тикеру в цикле
    # пересчитываем для SBER
    with open(DATA_DIR / "SBER/1d.json") as f:
        _close_s = np.array([float(c["close"]) for c in json.load(f)], dtype=np.float64)
    _trend_s = logtrend_causal(_close_s)
    _ratio_s = _close_s / _trend_s
    _att_s   = sosfilt(_SOS_LP, np.diff(_ratio_s))
    _att_hist = _att_s[:_ok]
    _X_s, _y_s = _build_Xy(_att_hist, P)
    _vec_orig = _att_hist[-P:].copy()
    _raw_s = collect_bias_data(_att_hist, _X_s, _y_s, P, XI, _ok, TAU_WIDE, HORIZON)

    if _raw_s:
        _bias_mean_s  = compute_bias(_raw_s, _ok, P, HORIZON, _vec_orig, "mean")
        _bias_phase_s = compute_bias(_raw_s, _ok, P, HORIZON, _vec_orig, "phase")

        fig_d, axes_d = plt.subplots(1, 2, figsize=(14, 4))
        vmax = max(np.abs(_bias_mean_s).max(), np.abs(_bias_phase_s).max())
        for ax_d, bm, lbl in zip(
            axes_d,
            [_bias_mean_s, _bias_phase_s],
            ["mean bias", "phase-weighted bias"],
        ):
            im = ax_d.imshow(bm.T, aspect="auto", cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax_d.set_xlabel("Горизонт h")
            ax_d.set_ylabel("Компонента вектора задержек")
            ax_d.set_xticks(range(HORIZON)); ax_d.set_xticklabels(range(1, HORIZON + 1))
            ax_d.set_yticks(range(P))
            ax_d.set_yticklabels([f"att[t−{P-1-j}]" for j in range(P)], fontsize=8)
            ax_d.set_title(f"SBER origin={_ok}: {lbl}")
            plt.colorbar(im, ax=ax_d, label="bias")
        fig_d.suptitle("60: Структура bias в пространстве задержек (один origin)")
        fig_d.tight_layout()
        fig_d.savefig(FIG_DIR / "60_bias_heatmap.png", dpi=150)
        print(f"Рис. D: {FIG_DIR / '60_bias_heatmap.png'}")

# ── Рис. E: траектории SBER (baseline vs best) ────────────────────────────────

if traj_data:
    n_plots = min(N_TRAJ, len(traj_data))
    best_name = CONFIGS[best_ci]["name"]
    fig_e, axes_e = plt.subplots(2, 2, figsize=(14, 8))
    axes_e = axes_e.flatten()
    _att_s = sosfilt(_SOS_LP, np.diff(logtrend_causal(
        np.array([float(c["close"]) for c in json.load(open(DATA_DIR / "SBER/1d.json"))],
                  dtype=np.float64)).__rtruediv__(  # ratio
        np.array([float(c["close"]) for c in json.load(open(DATA_DIR / "SBER/1d.json"))],
                  dtype=np.float64))))   # это громоздко — лучше просто показать LP-ratio

    for idx, td in enumerate(traj_data[:n_plots]):
        ax = axes_e[idx]
        r0 = td["ratio0"]
        h_ax_ = np.arange(1, HORIZON + 1)
        act_lp = r0 + np.cumsum(td["att_actual"])
        ax.plot(h_ax_, act_lp, "k-", lw=2.5, label="LP actual", zorder=5)
        ax.plot(h_ax_, r0 + np.cumsum(td["forecasts"]["baseline"]),
                "r--", lw=1.5, alpha=0.8, label="baseline")
        if best_name in td["forecasts"]:
            ax.plot(h_ax_, r0 + np.cumsum(td["forecasts"][best_name]),
                    "b-", lw=1.8, alpha=0.9, label=f"#{best_ci} best")
        ax.axhline(r0, color="#aaa", lw=0.7, ls=":")
        ax.set_title(f"SBER origin={td['origin_k']}", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    for ax in axes_e[n_plots:]:
        ax.set_visible(False)
    fig_e.suptitle(f"60: Траектории LP-ratio  baseline vs #{best_ci} «{best_name}»",
                   fontsize=11)
    fig_e.tight_layout()
    fig_e.savefig(FIG_DIR / "60_trajectories.png", dpi=150)
    print(f"Рис. E: {FIG_DIR / '60_trajectories.png'}")

print("\nСкрипт 60 завершён.")
