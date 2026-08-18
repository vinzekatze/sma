"""
59 — Шумоподавление Шрайбера в фазовом пространстве + LWR

Вместо частотной фильтрации (Butterworth, τ≈7-53б) — проекция каждой точки
фазового пространства на локальное многообразие аттрактора (локальный PCA
по k ближайшим прошлым соседям). Causal-версия: только t' < t.

Преимущество: нет задержки фильтра по построению — точка корректируется
в своём собственном времени, не в прошлом.

Конфигурации:
  0. baseline: per-comp LWR C3-C5 p=20 + damped AR C2(γ=0.8) + C1(γ=0.5)
  1. LP-causal p=8 (лучший из скр.58)
  2+. Schreiber(p_s, k, rank) + LWR(p_lwr)

Параметры Шрайбера:
  p_s   — размерность вложения (из FNN: 4-5 для чистого сигнала)
  rank  — размерность аттрактора (число удерживаемых сингулярных векторов)
  k     — число соседей для локального PCA

Параметры LWR после очистки:
  p_lwr — размерность для матрицы задержек LWR (можно отличаться от p_s)
  ξ     = 3*(p_lwr+1)

Протокол: 8 тикеров 1d, N_ORIG=200, horizon=15, logtrend.
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
from scipy.stats import ttest_rel

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
FIG_DIR  = ROOT / "research/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
N_ORIG       = 200
HORIZON      = 15
FILTER_ORDER = 4

# baseline (старый фильтрбанк)
CUTOFFS_OLD  = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
P_SLOW       = 20
XI_SLOW      = 3 * (P_SLOW + 1)
GAMMA_C1     = 0.5
GAMMA_C2     = 0.8
P_AR_MAX     = 20

# LP-causal (скр.58)
SOS_LP = butter(FILTER_ORDER, 0.125, btype="low", output="sos")
LP_P   = 8

# конфигурации Шрайбера для свипа
# (p_s, rank, k, p_lwr)  — p_lwr отвязан от p_s
SCHREIBER_CFGS = [
    # ── sweep p_s при k=30 ──────────────────────────────────────────────
    (4,  3, 30,  5),   # p_s=4  rank=3  (оставляем 3 из 4)
    (4,  3, 30,  8),   # то же, p_lwr=8
    (5,  4, 30,  5),   # p_s=5  rank=4  (оставляем 4 из 5)
    (5,  4, 30,  8),
    (5,  3, 30,  5),   # агрессивнее: rank=3
    (6,  4, 30,  5),
    (6,  5, 30,  8),
    (8,  4, 30,  8),   # глубокое вложение, удаляем 4 шума
    (8,  5, 30,  8),
    # ── sweep k при лучшем p_s ──────────────────────────────────────────
    (5,  4, 15,  5),
    (5,  4, 50,  5),
    (5,  4, 80,  5),
]

CONFIGS = [
    {"name": "baseline",          "mode": "baseline"},
    {"name": "LP-causal  p=8",    "mode": "lp_causal"},
]
for p_s, rank, k, p_lwr in SCHREIBER_CFGS:
    CONFIGS.append({
        "name": f"Schr p={p_s} r={rank} k={k:2d} lwr={p_lwr}",
        "mode": "schreiber",
        "p_s": p_s, "rank": rank, "k": k, "p_lwr": p_lwr,
    })

# ── вспомогательные функции ────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t); trend[:2] = close[:2]
    return trend


SOS_OLD = [butter(FILTER_ORDER, fc, btype="low", output="sos") for fc in CUTOFFS_OLD]

def make_old_fb(series: np.ndarray) -> np.ndarray:
    comps, rem = [], series.copy()
    for sos in SOS_OLD:
        low = sosfilt(sos, rem)
        comps.append(rem - low); rem = low
    comps.append(rem)
    return np.array(comps)


def fit_ar_bic(series: np.ndarray) -> tuple[int, np.ndarray]:
    n = len(series)
    best_bic, best_p, best_c = np.inf, 1, np.zeros(2)
    for p in range(1, min(P_AR_MAX + 1, (n - 1) // 4)):
        X = np.column_stack([series[p - 1 - k: n - 1 - k] for k in range(p)])
        X = np.hstack([np.ones((n - p, 1)), X])
        c, _, _, _ = np.linalg.lstsq(X, series[p:], rcond=None)
        ssr = np.sum((series[p:] - X @ c) ** 2)
        bic = (n - p) * np.log(ssr / (n - p) + 1e-12) + (p + 1) * np.log(n - p)
        if bic < best_bic:
            best_bic, best_p, best_c = bic, p, c
    return best_p, best_c


def forecast_damped_ar(series: np.ndarray, horizon: int, gamma: float) -> np.ndarray:
    if len(series) < 4:
        return np.zeros(horizon)
    p, c = fit_ar_bic(series)
    buf = list(series[-p:])
    raw = np.empty(horizon)
    for h in range(horizon):
        val = c[0] + sum(c[1 + k] * buf[-(k + 1)] for k in range(p))
        raw[h] = val; buf.append(val)
    return raw * (gamma ** np.arange(1, horizon + 1))


def forecast_lwr(series: np.ndarray, p: int, horizon: int) -> np.ndarray:
    xi = 3 * (p + 1)
    n  = len(series)
    if n < xi + p + 2:
        return np.zeros(horizon)
    X = np.array([series[i: i + p] for i in range(n - p)])
    y = series[p:]
    if len(X) < xi:
        return np.zeros(horizon)
    vec = series[-p:].copy()
    out = np.empty(horizon)
    for h in range(horizon):
        dists  = np.linalg.norm(X - vec, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        nn_d   = dists[nn_idx]
        h_bw   = max(nn_d.max(), 1e-12)
        w      = np.exp(-0.5 * (nn_d / h_bw) ** 2)
        A      = np.hstack([np.ones((xi, 1)), X[nn_idx]])
        ws     = np.sqrt(w)
        coef, _, _, _ = np.linalg.lstsq(ws[:, None] * A, ws * y[nn_idx], rcond=None)
        val    = float(coef[0] + vec @ coef[1:])
        out[h] = val
        vec    = np.roll(vec, -1); vec[-1] = val
    return out


# ── Шрайбер causal ────────────────────────────────────────────────────────────

def schreiber_causal(x: np.ndarray, p: int, k: int, rank: int) -> np.ndarray:
    """
    Каузальное шумоподавление Шрайбера.
    Для каждого t: проецируем X[t]=[x[t-p+1..t]] на локальный аттрактор,
    используя только прошлых соседей (t' < t).

    x_clean[t] = последний элемент спроецированного вектора.
    """
    n       = len(x)
    x_clean = x.copy()
    rank    = min(rank, p - 1)   # нельзя убрать 0 шумовых компонент

    # Матрица всех прошлых окон (строится инкрементально)
    # windows[i] = [x[i], x[i+1], ..., x[i+p-1]] — окно, заканчивающееся в i+p-1
    # Используем список для инкрементального добавления
    wins = np.empty((n, p), dtype=np.float64)   # prealloc (заполняется по ходу)

    for t in range(p - 1, n):
        # текущий вектор
        vec = x[t - p + 1: t + 1]          # shape (p,)
        # прошлые окна: wins[0..t-p] → заканчивались в t-1, t-2, ... p-1
        n_past = t - (p - 1)                # = t - p + 1
        if n_past > 0:
            wins[n_past - 1] = x[t - p: t]  # добавляем окно, закончившееся в t-1

        if n_past < k:
            continue

        past = wins[:n_past]                # (n_past, p)

        # k ближайших соседей
        dists  = np.linalg.norm(past - vec, axis=1)
        ki     = min(k, n_past)
        nn_idx = np.argpartition(dists, ki - 1)[:ki]
        nbrs   = past[nn_idx]               # (k, p)

        # локальный PCA
        centroid = nbrs.mean(axis=0)
        centered = nbrs - centroid
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        # проекция на подпространство аттрактора (rank компонент)
        vec_c = vec - centroid
        proj  = Vt[:rank].T @ (Vt[:rank] @ vec_c)
        x_clean[t] = (centroid + proj)[-1]  # последний элемент = текущее значение

    return x_clean


# ── прогноз одного origin ──────────────────────────────────────────────────────

def forecast_one(dratio_hist: np.ndarray, cfg: dict) -> np.ndarray:
    mode = cfg["mode"]

    if mode == "baseline":
        comp = make_old_fb(dratio_hist)
        dhat = np.zeros(HORIZON)
        for ci in [3, 4, 5]:
            dhat += forecast_lwr(comp[ci], P_SLOW, HORIZON)
        dhat += forecast_damped_ar(comp[2], HORIZON, GAMMA_C2)
        dhat += forecast_damped_ar(comp[1], HORIZON, GAMMA_C1)
        return dhat

    if mode == "lp_causal":
        att  = sosfilt(SOS_LP, dratio_hist)
        c1   = sosfilt(butter(FILTER_ORDER, 0.25, btype="low", output="sos"),
                       dratio_hist) - att
        dhat = forecast_lwr(att, LP_P, HORIZON)
        dhat += forecast_damped_ar(c1, HORIZON, GAMMA_C1)
        return dhat

    # mode == "schreiber"
    p_s   = cfg["p_s"]
    k     = cfg["k"]
    rank  = cfg["rank"]
    p_lwr = cfg["p_lwr"]

    dratio_clean = schreiber_causal(dratio_hist, p_s, k, rank)
    return forecast_lwr(dratio_clean, p_lwr, HORIZON)


# ── smoke test ─────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    print("=== smoke test ===", flush=True)
    rng    = np.random.default_rng(42)
    close_ = np.cumprod(1 + rng.normal(0, 0.01, 600)) * 100.0
    trend_ = logtrend_causal(close_)
    dr_    = np.diff(close_ / trend_)           # shape (599,)

    # 1. schreiber_causal — ключевой kwarg-тест
    schr_ = schreiber_causal(dr_[:200], p=5, k=30, rank=4)
    assert schr_.shape == (200,), f"schreiber shape: {schr_.shape}"
    assert not np.any(np.isnan(schr_)), "NaN in schreiber output"
    print("  schreiber_causal  OK", flush=True)

    # 2. forecast_one — по одной конфигурации каждого типа
    hist_ = dr_[:400]
    test_cfgs = [
        CONFIGS[0],                             # baseline
        CONFIGS[1],                             # lp_causal
        next(c for c in CONFIGS if c["mode"] == "schreiber"),
    ]
    for cfg_ in test_cfgs:
        out_ = forecast_one(hist_, cfg_)
        assert out_.shape == (HORIZON,),  f"{cfg_['name']}: shape {out_.shape}"
        assert not np.any(np.isnan(out_)), f"{cfg_['name']}: NaN in forecast"
        print(f"  forecast_one [{cfg_['name'].strip():30s}]  OK", flush=True)

    print("=== smoke test PASSED ===\n", flush=True)

_smoke_test()


# ── walk-forward ───────────────────────────────────────────────────────────────

N_CONFIGS = len(CONFIGS)
mapes: list[dict[str, list[float]]] = [{t: [] for t in TICKERS} for _ in CONFIGS]

# траектории SBER
TRAJ_TICKER = "SBER"
N_TRAJ      = 4
traj_data: list[dict] = []

# для визуализации сигналов: запишем один origin из SBER
sig_saved   = False
sig_example = {}

t0     = time.time()
n_done = 0

for ticker in TICKERS:
    with open(DATA_DIR / ticker / "1d.json") as f:
        candles = json.load(f)
    close = np.array([float(c["close"]) for c in candles], dtype=np.float64)
    trend = logtrend_causal(close)
    ratio = close / trend
    N     = len(ratio)

    max_ok = N - 1 - HORIZON
    min_ok = max(300, N - 700)
    if max_ok <= min_ok:
        continue
    origins = np.unique(np.linspace(min_ok, max_ok, N_ORIG, dtype=int))

    traj_count = 0

    for origin_k in origins:
        dratio_hist  = np.diff(ratio[:origin_k + 1])
        actual_ratio = ratio[origin_k + 1: origin_k + 1 + HORIZON]
        if len(actual_ratio) < HORIZON:
            continue
        ratio0 = float(ratio[origin_k])

        # запись для траекторий (SBER, первые N_TRAJ)
        record = (ticker == TRAJ_TICKER and traj_count < N_TRAJ)
        if record:
            entry = {"actual": actual_ratio.copy(), "ratio0": ratio0,
                     "origin_k": origin_k, "forecasts": {}}

        # запись для визуализации сигнала (первый подходящий origin SBER)
        if ticker == TRAJ_TICKER and not sig_saved and len(dratio_hist) >= 150:
            # сохраняем последние 150 баров сигналов для рис. D
            seg = dratio_hist[-150:]
            sig_example["raw"]  = seg.copy()
            sig_example["lp"]   = sosfilt(SOS_LP, dratio_hist)[-150:]
            # Schreiber с лучшими параметрами (p=5, r=4, k=30)
            schr = schreiber_causal(dratio_hist, p=5, k=30, rank=4)
            sig_example["schr"] = schr[-150:]
            sig_saved = True

        for ci, cfg in enumerate(CONFIGS):
            dhat = forecast_one(dratio_hist, cfg)
            if np.any(np.isnan(dhat)) or np.any(np.abs(dhat) > 1e6):
                if record:
                    entry["forecasts"][cfg["name"]] = None
                continue
            ratio_hat = ratio0 + np.cumsum(dhat)
            mape = float(np.mean(
                np.abs(ratio_hat - actual_ratio) / (np.abs(actual_ratio) + 1e-10)
            ))
            if np.isfinite(mape):
                mapes[ci][ticker].append(mape)
            if record:
                entry["forecasts"][cfg["name"]] = ratio_hat.copy()

        if record:
            traj_data.append(entry)
            traj_count += 1

        n_done += 1
        if n_done % 200 == 0:
            elapsed = time.time() - t0
            total   = len(TICKERS) * len(origins)
            print(f"  {n_done}/{total}  ({elapsed:.0f}s)")

print(f"\nГотово за {time.time() - t0:.1f}с")

# ── агрегация ──────────────────────────────────────────────────────────────────

baseline_flat = [v for t in TICKERS for v in mapes[0][t]]
agg_base      = float(np.mean(baseline_flat))
results: list[dict] = []

print("\n── AGG MAPE ─────────────────────────────────────────────────────────────────")
print(f"{'#':<2}  {'Конфигурация':<34}  {'AGG MAPE':>9}  {'Δ% baseline':>12}  p-value")
print("─" * 78)

for ci, cfg in enumerate(CONFIGS):
    flat = [v for t in TICKERS for v in mapes[ci][t]]
    if not flat:
        continue
    agg  = float(np.mean(flat))
    d    = (agg / agg_base - 1.0) * 100.0 if ci > 0 else 0.0
    if ci > 0 and len(flat) == len(baseline_flat):
        _, pval = ttest_rel(flat, baseline_flat)
    else:
        pval = float("nan")
    sign  = "✅" if d < -0.5 else ("❌" if d > 0.5 else "~")
    p_str = f"p={pval:.4f}" if np.isfinite(pval) else "—"
    print(f"{ci:<2}  {cfg['name']:<34}  {agg:.5f}   {d:>+8.2f}%  {sign}  {p_str}")
    results.append({"ci": ci, "cfg": cfg, "agg": agg, "d": d, "pval": pval, "flat": flat})

best_schr = min((r for r in results if r["cfg"]["mode"] == "schreiber"),
                key=lambda r: r["agg"], default=None)
print(f"\nЛучший Schreiber: {best_schr['cfg']['name'] if best_schr else '—'}")

if best_schr:
    bci = best_schr["ci"]
    print(f"\n── Per-ticker: {best_schr['cfg']['name']} vs baseline ─────────────────")
    print(f"{'Ticker':<6}  {'baseline':>9}  {'Schreiber':>9}  {'Δ%':>8}")
    print("─" * 38)
    for ticker in TICKERS:
        bm = np.mean(mapes[0][ticker])    if mapes[0][ticker]    else float("nan")
        sm = np.mean(mapes[bci][ticker])  if mapes[bci][ticker]  else float("nan")
        d  = (sm / bm - 1.0) * 100.0 if np.isfinite(bm) and np.isfinite(sm) else float("nan")
        print(f"{ticker:<6}  {bm:.5f}   {sm:.5f}   {d:>+7.2f}%")

# ── Рис. A: barh AGG MAPE ─────────────────────────────────────────────────────

fig_a, ax_a = plt.subplots(figsize=(13, 8))
names  = [r["cfg"]["name"] for r in results]
aggs   = [r["agg"]         for r in results]
colors = []
for r in results:
    if r["ci"] == 0:   colors.append("#78909c")
    elif r["cfg"]["mode"] == "lp_causal": colors.append("#fb8c00")
    else: colors.append("#43a047" if r["d"] < -0.5 else ("#e53935" if r["d"] > 0.5 else "#ffa726"))

bars = ax_a.barh(names, aggs, color=colors, alpha=0.85, height=0.65)
ax_a.axvline(agg_base, color="#546e7a", lw=1.5, ls="--",
             label=f"baseline {agg_base:.5f}")
for bar, r in zip(bars, results):
    label = f"  {r['d']:+.2f}%" if r["ci"] > 0 else "  baseline"
    ax_a.text(bar.get_width() + 3e-5,
              bar.get_y() + bar.get_height() / 2,
              label, va="center", fontsize=8)
ax_a.set_xlabel("AGG MAPE")
ax_a.set_title("59: Schreiber phase-space noise reduction + LWR\n"
               "оранжевый = LP-causal (скр.58), зелёный = Schreiber; 8 тикеров 1d N=200")
ax_a.legend(fontsize=9)
fig_a.tight_layout()
fig_a.savefig(FIG_DIR / "59_mape_all.png", dpi=150)
print(f"\nРис. A: {FIG_DIR / '59_mape_all.png'}")

# ── Рис. B: p_s-кривая (Schreiber) при k=30 ──────────────────────────────────

schr_k30 = [r for r in results
            if r["cfg"]["mode"] == "schreiber" and r["cfg"]["k"] == 30
            and r["cfg"]["p_lwr"] == r["cfg"]["p_s"]]  # p_lwr == p_s для чистоты

if schr_k30:
    fig_b, ax_b = plt.subplots(figsize=(9, 4))
    ax_b.axhline(agg_base, ls="--", color="#78909c", lw=1.5,
                 label=f"baseline {agg_base:.5f}")
    lp_r = next((r for r in results if r["cfg"]["mode"] == "lp_causal"), None)
    if lp_r:
        ax_b.axhline(lp_r["agg"], ls=":", color="#fb8c00", lw=1.5,
                     label=f"LP-causal p=8  {lp_r['agg']:.5f}")

    # группируем по rank
    ranks = sorted(set(r["cfg"]["rank"] for r in schr_k30))
    for rank in ranks:
        sub = sorted([r for r in schr_k30 if r["cfg"]["rank"] == rank],
                     key=lambda r: r["cfg"]["p_s"])
        ps   = [r["cfg"]["p_s"]  for r in sub]
        aggs_ = [r["agg"]        for r in sub]
        ax_b.plot(ps, aggs_, "o-", lw=2, ms=7, label=f"rank={rank}")

    ax_b.set_xlabel("p_s (размерность вложения Шрайбера)")
    ax_b.set_ylabel("AGG MAPE")
    ax_b.set_title("59: AGG MAPE vs p_s  (k=30, p_lwr=p_s)")
    ax_b.legend(fontsize=9); ax_b.grid(alpha=0.3)
    fig_b.tight_layout()
    fig_b.savefig(FIG_DIR / "59_ps_curve.png", dpi=150)
    print(f"Рис. B: {FIG_DIR / '59_ps_curve.png'}")

# ── Рис. C: траектории прогноза (SBER) ────────────────────────────────────────

if traj_data and best_schr:
    n_plots = min(N_TRAJ, len(traj_data))
    fig_c, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    x_fc = np.arange(1, HORIZON + 1)

    show = {
        "baseline":                   ("#e53935", "--",  "baseline"),
        best_schr["cfg"]["name"]:     ("#43a047", "-",   "Schreiber (best)"),
    }
    lp_name = "LP-causal  p=8"
    if any(lp_name in e["forecasts"] for e in traj_data):
        show[lp_name] = ("#fb8c00", "-.", "LP-causal p=8")

    for idx, entry in enumerate(traj_data[:n_plots]):
        ax = axes[idx]
        ax.plot([0], [entry["ratio0"]], "ko", ms=5, zorder=5)
        ax.plot(x_fc, entry["actual"], "k-", lw=2.5, label="actual", zorder=4)
        for cfg_name, (color, ls, label) in show.items():
            fc = entry["forecasts"].get(cfg_name)
            if fc is not None:
                ax.plot(x_fc, fc, ls, color=color, lw=1.8, alpha=0.9, label=label)
        ax.axvline(0.5, color="#aaa", lw=0.8, ls=":")
        ax.set_title(f"{TRAJ_TICKER}  origin #{idx+1}  (k={entry['origin_k']})",
                     fontsize=10)
        ax.set_xlabel("горизонт (бары)"); ax.set_ylabel("ratio")
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)
    for ax in axes[n_plots:]:
        ax.set_visible(False)
    fig_c.suptitle(f"59: Траектории прогноза — {TRAJ_TICKER}", fontsize=12)
    fig_c.tight_layout()
    fig_c.savefig(FIG_DIR / "59_trajectories.png", dpi=150)
    print(f"Рис. C: {FIG_DIR / '59_trajectories.png'}")

# ── Рис. D: визуализация качества очистки сигнала ─────────────────────────────

if sig_example:
    raw  = sig_example["raw"]
    lp   = sig_example["lp"]
    schr = sig_example["schr"]
    t_ax = np.arange(len(raw))

    fig_d, axes_d = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes_d[0].plot(t_ax, raw,  color="#78909c", lw=0.8, label="raw dratio")
    axes_d[0].set_title("raw dratio (с шумом C0+C1)")
    axes_d[0].axhline(0, color="#aaa", lw=0.5); axes_d[0].legend(fontsize=9)

    axes_d[1].plot(t_ax, lp,   color="#fb8c00", lw=1.5, label=f"LP-causal (τ≈7б)")
    axes_d[1].plot(t_ax, raw,  color="#78909c", lw=0.5, alpha=0.4)
    axes_d[1].set_title("Butterworth LP(0.125) — задержка заметна на трендах")
    axes_d[1].axhline(0, color="#aaa", lw=0.5); axes_d[1].legend(fontsize=9)

    axes_d[2].plot(t_ax, schr, color="#43a047", lw=1.5, label="Schreiber p=5 r=4 k=30")
    axes_d[2].plot(t_ax, raw,  color="#78909c", lw=0.5, alpha=0.4)
    axes_d[2].set_title("Schreiber — очищен от шума, без задержки")
    axes_d[2].axhline(0, color="#aaa", lw=0.5); axes_d[2].legend(fontsize=9)

    for ax in axes_d:
        ax.set_ylabel("dratio"); ax.grid(alpha=0.2)
    axes_d[-1].set_xlabel("бар (последние 150)")
    fig_d.suptitle(f"59: Качество очистки сигнала — {TRAJ_TICKER}", fontsize=12)
    fig_d.tight_layout()
    fig_d.savefig(FIG_DIR / "59_signal_quality.png", dpi=150)
    print(f"Рис. D: {FIG_DIR / '59_signal_quality.png'}")

print("\nСкрипт 59 завершён.")
