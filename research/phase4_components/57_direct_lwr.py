"""
57 — Direct multi-step LWR с компенсацией фазовой задержки фильтра.

Проблема: causal Butterworth (sosfilt) вносит задержку τ.
  comp[ci][t] ≈ actual_slow[t − τ]
  Итерационный LWR предсказывает comp[ci][origin+h] ≈ actual_slow[origin+h−τ].
  При τ > h — это прошлое: C3 τ≈41, C4/C5 τ≈82 >> horizon=15.
  Итеративное накопление ошибки дополнительно ухудшает результат.

Решение — Direct h-step LWR:
  Обучаем: X[i] → comp_ci[i + p − 1 + h + τ]   (отдельный WLS для каждого h)
  X[i][−1] = comp_ci[i+p−1], таргет на h+τ шагов вперёд от последнего элемента.
  Инференс: одна WLS per h, нет итерационного накопления.
  comp_ci[origin+h+τ] ≈ actual_slow[origin+h] → правильная временная привязка.

Конфигурации (p=20, ξ=63, компоненты C3+C4+C5):
  iter                — итерационный LWR τ=0 (baseline, скр.47 стандарт)
  direct τ=K          — direct h-step, K ∈ {0, 5, 10, 20, 30, 41, 60, 82}
  direct τ=per-comp   — per-component теор. τ: C3=τ_C3, C4=τ_C4, C5=τ_C5

τ=0 в direct режиме = multi-step ahead regression без фазовой коррекции.
Если τ=0 уже лучше iter → проблема в итерационном накоплении, не только в фазе.
Если τ=TAU_THEORY лучше τ=0 → фазовая задержка тоже значима.

Протокол: 8 тикеров 1d, N=200 origins, horizon=15, logtrend.
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
from scipy.signal import group_delay as sig_group_delay
from scipy.stats import ttest_rel

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
FIG_DIR  = ROOT / "research/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from sma.core.forecast.embedding import build_delay_matrix, last_vector

# ── параметры ─────────────────────────────────────────────────────────────────

TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
N_ORIG       = 200
HORIZON      = 15
FILTER_ORDER = 4
CUTOFFS      = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
P            = 20
XI           = 3 * (P + 1)    # 63
SLOW_IDX     = [3, 4, 5]

# τ-свип: покрывает 0 (no correction), теор. C3 (~41), теор. C4/C5 (~82)
TAU_SWEEP    = [0, 5, 10, 20, 30, 41, 60, 82]

# ── теоретические группо-задержки ────────────────────────────────────────────

def _tau_dc(Wn: float, order: int = 4) -> int:
    """Группо-задержка при DC (ω→0) для butter(order, Wn) lowpass."""
    b, a = butter(order, Wn)
    _, gd = sig_group_delay((b, a), w=[1e-4])
    return int(round(float(gd[0])))

# Нижняя граница каждой компоненты определяет фазовую задержку:
#   C3 = LP(0.0625) − LP(0.03125)  →  нижний фильтр Wn=0.03125
#   C4 = LP(0.03125) − LP(0.015625) → нижний фильтр Wn=0.015625
#   C5 = residual после LP(0.015625) → тот же фильтр
_LOWER_CUTOFF = {3: CUTOFFS[3], 4: CUTOFFS[4], 5: CUTOFFS[4]}
TAU_THEORY    = {ci: _tau_dc(_LOWER_CUTOFF[ci]) for ci in SLOW_IDX}

print("Теоретические группо-задержки (DC, нижняя граница компоненты):")
for ci in SLOW_IDX:
    print(f"  C{ci}: τ ≈ {TAU_THEORY[ci]} баров  "
          f"(Wn={_LOWER_CUTOFF[ci]:.5f}, T_band≈{int(1/_LOWER_CUTOFF[ci])} баров)")

# ── helpers ───────────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n   = len(close)
    lc  = np.log(np.maximum(close, 1e-10))
    t   = np.arange(n, dtype=np.float64)
    cn  = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t)
    trend[:2] = close[:2]
    return trend


def make_filter_bank(series: np.ndarray) -> np.ndarray:
    components: list[np.ndarray] = []
    remaining = series.copy()
    for fc in CUTOFFS:
        sos = butter(FILTER_ORDER, fc, btype="low", output="sos")
        low = sosfilt(sos, remaining)
        components.append(remaining - low)
        remaining = low
    components.append(remaining)
    return np.array(components)   # (6, N)


def _wls_once(X: np.ndarray, y: np.ndarray, vec: np.ndarray, xi: int) -> float:
    """Gaussian LWR — одна предсказание из xi ближайших соседей."""
    dists  = np.linalg.norm(X - vec, axis=1)
    idx    = np.argpartition(dists, xi)[:xi]
    h_bw   = max(float(dists[idx].max()), 1e-10)
    w      = np.exp(-0.5 * (dists[idx] / h_bw) ** 2)
    A      = np.hstack([np.ones((xi, 1)), X[idx]])
    sw     = np.sqrt(w)
    c, *_  = np.linalg.lstsq(sw[:, None] * A, sw * y[idx], rcond=None)
    return float(c[0] + vec @ c[1:])


def forecast_iter(comp_ci: np.ndarray, p: int, xi: int, horizon: int) -> np.ndarray:
    """Итерационный LWR: предсказываем 1 шаг, откатываем вектор, повторяем."""
    X, y = build_delay_matrix(comp_ci, p)
    if len(X) < xi:
        return np.zeros(horizon)
    cur = last_vector(comp_ci, p=p).copy()
    out = np.empty(horizon)
    for h in range(horizon):
        dists  = np.linalg.norm(X - cur, axis=1)
        idx    = np.argpartition(dists, xi)[:xi]
        h_bw   = max(float(dists[idx].max()), 1e-10)
        w      = np.exp(-0.5 * (dists[idx] / h_bw) ** 2)
        A      = np.hstack([np.ones((xi, 1)), X[idx]])
        sw     = np.sqrt(w)
        c, *_  = np.linalg.lstsq(sw[:, None] * A, sw * y[idx], rcond=None)
        val    = float(c[0] + cur @ c[1:])
        out[h] = val
        cur    = np.roll(cur, -1); cur[-1] = val
    return out


def forecast_direct(
    comp_ci: np.ndarray, p: int, xi: int, horizon: int, tau: int
) -> np.ndarray:
    """
    Direct h-step LWR с фазовой компенсацией τ.

    Для каждого h в 1..horizon строим отдельную задачу регрессии:
      X[i] = [comp_ci[i], ..., comp_ci[i+p-1]]   (delay vector, old→new)
      y[i] = comp_ci[i + p - 1 + h + τ]          (target: h+τ шагов от последнего)

    При τ=0: multi-step ahead regression без фазовой коррекции.
    При τ=TAU_THEORY[ci]: целевое значение соответствует actual_slow[origin+h].
    """
    N         = len(comp_ci)
    X_full, _ = build_delay_matrix(comp_ci, p)
    vec       = last_vector(comp_ci, p=p).copy()
    out       = np.zeros(horizon)

    for h in range(1, horizon + 1):
        y_start = p - 1 + h + tau    # comp_ci[y_start] = target для строки i=0
        if y_start >= N:
            break
        y_h = comp_ci[y_start:]
        n   = min(len(X_full), len(y_h))
        if n < xi:
            break
        out[h - 1] = _wls_once(X_full[:n], y_h[:n], vec, xi)

    return out

# ── конфигурации ─────────────────────────────────────────────────────────────

CONFIGS: list[dict] = [
    {"name": "iter",            "mode": "iter",       "tau": 0},
]
for _tau in TAU_SWEEP:
    CONFIGS.append({"name": f"direct τ={_tau:>2}", "mode": "direct",    "tau": _tau})
CONFIGS.append(    {"name": "direct τ=per-comp", "mode": "direct_per", "tau": None})

# ── walk-forward ──────────────────────────────────────────────────────────────

# mapes[cfg_i][ticker] — список AGG MAPE по каждому origin
mapes:   list[dict[str, list[float]]] = [{} for _ in CONFIGS]
# mapes_h[cfg_i][ticker][h] — список MAPE конкретного шага h
mapes_h: list[dict[str, list[list[float]]]] = [{} for _ in CONFIGS]
for ci in range(len(CONFIGS)):
    for t in TICKERS:
        mapes[ci][t]   = []
        mapes_h[ci][t] = [[] for _ in range(HORIZON)]

t0, n_done = time.time(), 0
total = len(TICKERS) * N_ORIG

for ticker in TICKERS:
    path = DATA_DIR / ticker / "1d.json"
    with open(path) as f:
        candles = json.load(f)
    close = np.array([float(c["close"]) for c in candles], dtype=np.float64)
    ratio = close / logtrend_causal(close)
    N     = len(ratio)

    max_ok  = N - 1 - HORIZON
    min_ok  = max(300, N - 700)
    origins = np.unique(np.linspace(min_ok, max_ok, N_ORIG, dtype=int))

    for origin_k in origins:
        dratio       = np.diff(ratio[:origin_k + 1])
        comp         = make_filter_bank(dratio)               # (6, origin_k)
        actual_ratio = ratio[origin_k + 1: origin_k + 1 + HORIZON]
        if len(actual_ratio) < HORIZON:
            continue
        ratio0 = float(ratio[origin_k])

        for ci, cfg in enumerate(CONFIGS):
            dhat = np.zeros(HORIZON)
            ok   = True
            for c_idx in SLOW_IDX:
                if cfg["mode"] == "iter":
                    d = forecast_iter(comp[c_idx], P, XI, HORIZON)
                elif cfg["mode"] == "direct":
                    d = forecast_direct(comp[c_idx], P, XI, HORIZON, cfg["tau"])
                else:  # direct_per: per-component τ
                    d = forecast_direct(comp[c_idx], P, XI, HORIZON, TAU_THEORY[c_idx])
                if np.any(np.isnan(d)):
                    ok = False; break
                dhat += d
            if not ok or np.any(np.abs(dhat) > 1e6):
                continue

            ratio_hat  = ratio0 + np.cumsum(dhat)
            mape_h_arr = np.abs(ratio_hat - actual_ratio) / (np.abs(actual_ratio) + 1e-10)
            mape_agg   = float(np.mean(mape_h_arr))
            if np.isfinite(mape_agg):
                mapes[ci][ticker].append(mape_agg)
                for h_idx in range(HORIZON):
                    if np.isfinite(mape_h_arr[h_idx]):
                        mapes_h[ci][ticker][h_idx].append(float(mape_h_arr[h_idx]))

        n_done += 1
        if n_done % 300 == 0:
            el  = time.time() - t0
            eta = el / n_done * (total - n_done)
            print(f"  {n_done}/{total}  ({el:.0f}s, ещё ~{eta:.0f}s)")

print(f"\nГотово за {time.time()-t0:.1f} с")

# ── агрегация и вывод ─────────────────────────────────────────────────────────

baseline_flat = [v for t in TICKERS for v in mapes[0][t]]
baseline_agg  = float(np.mean(baseline_flat))

print(f"\n── AGG MAPE ──────────────────────────────────────────────────────────────")
print(f"{'#':<2}  {'Конфигурация':<22}  {'AGG MAPE':>9}  {'Δ%':>7}  {'':>2}  {'p-val':>7}")
print("─" * 60)

results: list[dict] = []
for ci, cfg in enumerate(CONFIGS):
    flat = [v for t in TICKERS for v in mapes[ci][t]]
    if not flat:
        continue
    agg   = float(np.mean(flat))
    delta = (agg / baseline_agg - 1.0) * 100.0 if ci > 0 else 0.0
    if ci > 0 and len(flat) == len(baseline_flat):
        _, pval = ttest_rel(flat, baseline_flat)
    else:
        pval = float("nan")
    sign = "✅" if delta < -0.5 else ("❌" if delta > 0.5 else "~")
    pstr = f"{pval:.4f}" if np.isfinite(pval) else "—"
    print(f"{ci:<2}  {cfg['name']:<22}  {agg:.5f}   {delta:>+6.2f}%  {sign}  {pstr}")
    results.append({"ci": ci, "cfg": cfg, "agg": agg, "delta": delta,
                    "pval": pval, "flat": flat})

# лучший из sweep
direct_res = [r for r in results if r["cfg"]["mode"] == "direct"]
if direct_res:
    best = min(direct_res, key=lambda r: r["agg"])
    print(f"\nЛучший direct τ={best['cfg']['tau']}:  "
          f"AGG={best['agg']:.5f}  Δ={best['delta']:+.2f}%  p={best['pval']:.4f}")
    print("  Per-ticker vs baseline:")
    for ticker in TICKERS:
        b = np.mean(mapes[0][ticker])   if mapes[0][ticker]            else float("nan")
        c = np.mean(mapes[best["ci"]][ticker]) if mapes[best["ci"]][ticker] else float("nan")
        d = (c/b - 1)*100 if np.isfinite(b) and np.isfinite(c) else float("nan")
        print(f"    {ticker}: {b:.5f} → {c:.5f}  ({d:+.2f}%)")

# per-comp config
pc_res = next((r for r in results if r["cfg"]["mode"] == "direct_per"), None)
if pc_res:
    print(f"\ndirect τ=per-comp: AGG={pc_res['agg']:.5f}  "
          f"Δ={pc_res['delta']:+.2f}%  p={pc_res['pval']:.4f}")

print(f"\nТеор. τ: C3={TAU_THEORY[3]}, C4={TAU_THEORY[4]}, C5={TAU_THEORY[5]}")

# ── фигуры ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Фиг 1: AGG MAPE vs τ ──────────────────────────────────────────────────────
ax = axes[0]
taus   = [r["cfg"]["tau"] for r in direct_res]
aggs   = [r["agg"]        for r in direct_res]
deltas = [r["delta"]      for r in direct_res]
colors = ["#43a047" if d < -0.5 else ("#e53935" if d > 0.5 else "#ffa726")
          for d in deltas]

ax.bar(range(len(taus)), aggs, color=colors, alpha=0.8, width=0.6)
ax.axhline(baseline_agg, color="#546e7a", lw=2, ls="--",
           label=f"iter = {baseline_agg:.5f}")

# отметим теоретические τ
tau_to_xi = {v: i for i, v in enumerate(taus)}
for ci_label, t_th in TAU_THEORY.items():
    xi_pos = tau_to_xi.get(t_th)
    if xi_pos is not None:
        ax.axvline(xi_pos, color="cyan", lw=1.2, ls=":", alpha=0.7,
                   label=f"τ_C{ci_label}={t_th}")

ax.set_xticks(range(len(taus)))
ax.set_xticklabels([str(t) for t in taus])
ax.set_xlabel("τ (фазовая коррекция, баров)")
ax.set_ylabel("AGG MAPE")
ax.set_title("Direct LWR: AGG MAPE vs τ")
ax.legend(fontsize=8)
for i, (agg, delta) in enumerate(zip(aggs, deltas)):
    ax.text(i, agg + 0.00003, f"{delta:+.1f}%", ha="center", fontsize=8)

# ── Фиг 2: MAPE по горизонту h ────────────────────────────────────────────────
ax2 = axes[1]

def mean_mape_h(ci: int) -> list[float]:
    return [float(np.mean([v for t in TICKERS for v in mapes_h[ci][t][h]]))
            for h in range(HORIZON)]

# iter baseline
ax2.plot(range(1, HORIZON + 1), mean_mape_h(0),
         "o-", color="#546e7a", lw=2, label="iter")

# τ=0 (multi-step без коррекции)
ci_tau0 = next(r["ci"] for r in results if r["cfg"].get("tau") == 0 and r["cfg"]["mode"] == "direct")
ax2.plot(range(1, HORIZON + 1), mean_mape_h(ci_tau0),
         "s-", color="#ffa726", lw=1.8, label="direct τ=0")

# лучший τ из sweep
if direct_res:
    ax2.plot(range(1, HORIZON + 1), mean_mape_h(best["ci"]),
             "^-", color="#43a047", lw=2, label=f"direct τ={best['cfg']['tau']} (лучший)")

# per-comp τ
if pc_res:
    ax2.plot(range(1, HORIZON + 1), mean_mape_h(pc_res["ci"]),
             "D--", color="#ab47bc", lw=1.5, label="direct τ=per-comp")

ax2.set_xlabel("Горизонт h (баров)")
ax2.set_ylabel("MAPE")
ax2.set_title("MAPE по горизонту: iter vs direct")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

fig.suptitle(f"57 — Direct multi-step LWR vs Iterative  "
             f"(8 тик., 1d, N={N_ORIG} origins, p={P})")
fig.tight_layout()
path_fig = FIG_DIR / "57_direct_lwr.png"
fig.savefig(path_fig, dpi=150)
print(f"\nГрафик: {path_fig}")

# ── итог ──────────────────────────────────────────────────────────────────────
print(f"\n── Итог ─────────────────────────────────────────────────────────────────")
print(f"iter baseline:     {baseline_agg:.5f}")
if direct_res:
    print(f"Лучший direct:     {best['agg']:.5f}  τ={best['cfg']['tau']}  Δ={best['delta']:+.2f}%")
if pc_res:
    print(f"direct per-comp:   {pc_res['agg']:.5f}  Δ={pc_res['delta']:+.2f}%")
print(f"\nГипотеза:")
print(f"  τ=0 лучше iter   → основная проблема — накопление ошибки (не фаза)")
print(f"  τ>0 лучше τ=0    → фазовая задержка дополнительно значима")
tau0_delta = next((r["delta"] for r in results if r["cfg"].get("tau") == 0
                   and r["cfg"]["mode"] == "direct"), float("nan"))
best_delta = best["delta"] if direct_res else float("nan")
if np.isfinite(tau0_delta) and np.isfinite(best_delta):
    phase_share = (tau0_delta - best_delta)
    iter_share  = -tau0_delta
    print(f"  Вклад накопления ошибки: {iter_share:+.2f}%  (iter→direct τ=0)")
    print(f"  Вклад фазовой задержки:  {phase_share:+.2f}%  (τ=0→τ={best['cfg']['tau']})")
