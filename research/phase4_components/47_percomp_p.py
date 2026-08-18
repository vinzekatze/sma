"""
47 — Per-component p оптимизация для C3–C5.

Текущий стандарт: p=20, ξ=63 (=3(p+1)) одинаково для всех компонент.

Правила из теории:
  p ≥ T_min / 4  (захватить минимальный период компоненты)
  ξ ≥ 3(p+1)    (достаточно соседей для устойчивой регрессии)

T_min по компонентам:
  C3 (16–52 bar): T_min=16 → p_min=4.   Текущий p=20 избыточен?
  C4 (52–103 bar): T_min=52 → p_min=13. Текущий p=20 — на нижней границе.
  C5 (103+ bar): T_min=103 → p_min=26.  Текущий p=20 НАРУШАЕТ правило!

Эксперимент A: sweep p отдельно по каждой компоненте.
  ξ = max(63, 3(p+1))  — адаптивный, по правилу.
  Метрика: MAE компоненты @ h=1..10.

Эксперимент B: итоговый MAPE с per-component optimal p vs uniform p=20.
"""

import json, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
OUT_DIR  = ROOT / "research/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from sma.core.forecast.embedding import build_delay_matrix, last_vector

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS      = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL     = "1d"
EPS          = 1e-10
STD_CUTOFFS  = [0.25, 0.125, 0.0625, 0.03125, 0.015625]
SLOW_IDX     = [3, 4, 5]
FILTER_ORDER = 4
VAL_H        = 10
N_ORIG       = 50
P_STD        = 20
XI_STD       = 63  # = 3*(P_STD+1)

# sweep p для каждой компоненты (ξ = max(63, 3(p+1)) адаптивно)
P_SWEEP = {
    3: [4, 6, 8, 10, 13, 16, 20, 25, 30],    # T_min=16, p_min=4
    4: [8, 10, 13, 16, 20, 25, 30, 40, 50],   # T_min=52, p_min=13
    5: [10, 16, 20, 25, 30, 40, 50, 63, 80],  # T_min=103, p_min=26
}

# имена компонент и нижние граничные периоды (для отображения)
COMP_NAMES  = {3: "C3 (16–52 bar)", 4: "C4 (52–103 bar)", 5: "C5 (103+ bar)"}
COMP_PMIN   = {3: 4, 4: 13, 5: 26}   # p_min по правилу T/4


def xi_for_p(p: int) -> int:
    return max(XI_STD, 3 * (p + 1))


# ── нормализация + данные ─────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    log_c = np.log(close + EPS); n = len(log_c)
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n+1, dtype=np.float64)
    ct=np.cumsum(t); ct2=np.cumsum(t**2); cy=np.cumsum(log_c); cty=np.cumsum(t*log_c)
    denom = cn*ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom>0, (cn*cty-ct*cy)/denom, 0.0)
    a=(cy-b*ct)/cn; trend=a+b*t; trend[:2]=log_c[:2]
    return np.exp(trend)


_CACHE: dict = {}

def load_data(ticker: str) -> tuple[np.ndarray, np.ndarray]:
    if ticker not in _CACHE:
        with open(DATA_DIR / ticker / f"{INTERVAL}.json") as f: c=json.load(f)
        close = np.array([x["close"] for x in c], dtype=np.float64)
        ratio = close / logtrend_causal(close)
        _CACHE[ticker] = (ratio, np.diff(ratio))
    return _CACHE[ticker]


def make_fb(series: np.ndarray, cutoffs: list, order: int = 4) -> np.ndarray:
    comps=[]; rem=series.copy()
    for fc in cutoffs:
        sos=butter(order,fc,btype="low",output="sos"); low=sosfilt(sos,rem)
        comps.append(rem-low); rem=low
    comps.append(rem); return np.array(comps)


# ── LWR ────────────────────────────────────────────────────────────────────────

def lwr_forecast(series: np.ndarray, horizon: int, p: int, xi: int) -> np.ndarray:
    X, y = build_delay_matrix(series, p)
    if len(X) < xi: return np.zeros(horizon)
    v = last_vector(series, p=p).copy(); hat = np.empty(horizon)
    for h in range(horizon):
        dists  = np.linalg.norm(X - v, axis=1)
        nn_idx = np.argpartition(dists, xi)[:xi]
        h_bw   = max(float(dists[nn_idx].max()), EPS)
        w      = np.exp(-0.5 * (dists[nn_idx] / h_bw)**2)
        A      = np.hstack([np.ones((xi,1)), X[nn_idx]]); sw = np.sqrt(w)
        c,_,_,_ = np.linalg.lstsq(sw[:,None]*A, sw*y[nn_idx], rcond=None)
        hat[h] = float(c[0] + v@c[1:]); v=np.roll(v,-1); v[-1]=hat[h]
    return hat


# ══════════════════════════════════════════════════════════════════════════════
#  Эксперимент A: per-component p sweep
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("47 — Per-component p оптимизация (C3–C5)")
print("=" * 70)
print(f"  Стандарт: p={P_STD}, ξ={XI_STD}  |  val_h={VAL_H}  |  N_ORIG={N_ORIG}/ticker")
print()

# mae_sweep[ci][p_val] = median MAE array (VAL_H,)
mae_sweep: dict[int, dict[int, np.ndarray]] = {ci: {} for ci in SLOW_IDX}

print("╔══ Эксперимент A: p sweep per component ══╗")

for ci in SLOW_IDX:
    p_list = P_SWEEP[ci]
    print(f"\n  {COMP_NAMES[ci]}  (p_min_rule={COMP_PMIN[ci]})")
    print(f"  {'p':>5} | {'ξ':>5} | {'MAE@h=1':>10} | {'MAE@h=5':>10} | {'MAE@h=10':>10} | {'Δ% vs p=20':>11}")
    print("  " + "-" * 62)

    for p_val in p_list:
        xi_val = xi_for_p(p_val)
        min_o  = p_val + xi_val + 10

        mae_ticker = []
        for ticker in TICKERS:
            ratio, dratio = load_data(ticker)
            n = len(dratio)
            COMP_full = make_fb(dratio, STD_CUTOFFS, FILTER_ORDER)
            max_o = n - VAL_H - 2
            origins = np.arange(max(min_o, max_o - N_ORIG), max_o)
            if len(origins) == 0:
                continue

            mae_sum = np.zeros(VAL_H); cnt = np.zeros(VAL_H, dtype=int)
            for vo in origins:
                COMP_h = make_fb(dratio[:vo], STD_CUTOFFS, FILTER_ORDER)
                hist   = COMP_h[ci]
                actual = COMP_full[ci][vo+1:vo+1+VAL_H]
                h_act  = len(actual)
                if h_act == 0: continue
                hat = lwr_forecast(hist, VAL_H, p_val, xi_val)
                err = np.abs(hat[:h_act] - actual)
                mae_sum[:h_act] += err; cnt[:h_act] += 1
            with np.errstate(invalid="ignore", divide="ignore"):
                mae = np.where(cnt > 0, mae_sum / cnt, np.nan)
            mae_ticker.append(mae)

        if not mae_ticker:
            continue
        mae_med = np.nanmedian(np.array(mae_ticker), axis=0)
        mae_sweep[ci][p_val] = mae_med

    # Печатаем таблицу (нужен p=20 для Δ%)
    ref_mae = mae_sweep[ci].get(P_STD, None)
    for p_val in p_list:
        if p_val not in mae_sweep[ci]: continue
        m = mae_sweep[ci][p_val]
        xi_val = xi_for_p(p_val)
        if ref_mae is not None:
            d1 = (m[0] - ref_mae[0]) / (ref_mae[0] + EPS) * 100
        else:
            d1 = float("nan")
        mark = " ← стандарт" if p_val == P_STD else (
               " ← p_min_rule" if p_val == COMP_PMIN[ci] else "")
        print(f"  {p_val:>5} | {xi_for_p(p_val):>5} | "
              f"{m[0]:>10.4e} | {m[4]:>10.4e} | {m[9]:>10.4e} | {d1:>+10.2f}%{mark}")


# ══════════════════════════════════════════════════════════════════════════════
#  Выбор оптимального p по компонентам (min MAE при h=1..5 среднее)
# ══════════════════════════════════════════════════════════════════════════════

print("\n╔══ Оптимальный p по компонентам ══╗")
opt_p: dict[int, int] = {}

for ci in SLOW_IDX:
    best_p, best_score = P_STD, np.inf
    for p_val, mae_arr in mae_sweep[ci].items():
        score = float(np.mean(mae_arr[:5]))   # среднее MAE h=1..5
        if score < best_score:
            best_score = score; best_p = p_val
    opt_p[ci] = best_p
    ref_score = float(np.mean(mae_sweep[ci].get(P_STD, mae_arr)[:5]))
    delta = (best_score - ref_score) / (ref_score + EPS) * 100
    print(f"  {COMP_NAMES[ci]}: opt_p={best_p} (ξ={xi_for_p(best_p)})  "
          f"score={best_score:.4e}  Δ={delta:+.2f}% vs p=20")


# ══════════════════════════════════════════════════════════════════════════════
#  Эксперимент B: реконструированный MAPE
# ══════════════════════════════════════════════════════════════════════════════

print("\n╔══ Эксперимент B: реконструированный AGG MAPE ══╗")

# Конфигурации:
# std:      p=20 для всех C3-C5
# opt_per:  opt_p[ci] для каждого ci отдельно
# also sweep: all same p but vary (to double-check global optimum)
GLOBAL_P_SWEEP = sorted(set(
    [p for ps in P_SWEEP.values() for p in ps]
))

mape_configs: dict[str, list] = {}

for cfg_name in ["std", "opt_per"] + [f"p{p}" for p in GLOBAL_P_SWEEP]:
    mape_configs[cfg_name] = []

t0 = time.time()
for ticker in TICKERS:
    ratio, dratio = load_data(ticker)
    n = len(dratio)
    # Нужен общий min_o: максимальный из всех конфигов
    max_min_o = max(
        max(p + xi_for_p(p) + 10 for p in P_SWEEP[ci]) for ci in SLOW_IDX
    )
    max_o = n - VAL_H - 2
    origins = np.arange(max(max_min_o, max_o - N_ORIG), max_o)
    if len(origins) == 0: continue

    for vo in origins:
        COMP_h = make_fb(dratio[:vo], STD_CUTOFFS, FILTER_ORDER)
        actual_ratio = ratio[vo+1:vo+1+VAL_H]; n_act = len(actual_ratio)
        if n_act == 0: continue
        r0 = float(ratio[vo])

        # Кешируем LWR для всех (p, ci) пар
        lwr_cache: dict[tuple, np.ndarray] = {}
        needed_configs = [
            ("std",    {ci: P_STD for ci in SLOW_IDX}),
            ("opt_per", opt_p),
        ] + [(f"p{p}", {ci: p for ci in SLOW_IDX}) for p in GLOBAL_P_SWEEP]

        for cfg_name, p_map in needed_configs:
            dratio_hat = np.zeros(VAL_H)
            for ci in SLOW_IDX:
                p_val = p_map[ci]
                xi_val = xi_for_p(p_val)
                key = (ci, p_val, xi_val)
                if key not in lwr_cache:
                    lwr_cache[key] = lwr_forecast(COMP_h[ci], VAL_H, p_val, xi_val)
                dratio_hat = dratio_hat + lwr_cache[key]
            r_hat = r0 + np.cumsum(dratio_hat)
            mape  = float(np.mean(
                np.abs(r_hat[:n_act] - actual_ratio) / (np.abs(actual_ratio) + EPS)
            ))
            mape_configs[cfg_name].append(mape)

print(f"  Walk-forward: {time.time()-t0:.1f}s")

mape_agg = {cfg: float(np.nanmedian(mape_configs[cfg])) for cfg in mape_configs}
mape_std  = mape_agg["std"]

print(f"\n  {'Конфигурация':16} | {'AGG MAPE':>9} | {'Δ% vs std':>10}")
print("  " + "-" * 43)
# std + opt_per
for cfg in ["std", "opt_per"]:
    d = (mape_agg[cfg] - mape_std) / mape_std * 100
    mark = " ← стандарт" if cfg == "std" else (" ★" if d < -0.3 else "")
    print(f"  {cfg:<16} | {mape_agg[cfg]:>9.5f} | {d:>+9.2f}%{mark}")

# global p sweep
print()
for p in GLOBAL_P_SWEEP:
    cfg = f"p{p}"
    d = (mape_agg[cfg] - mape_std) / mape_std * 100
    mark = " ← стандарт" if p == P_STD else (" ★" if d < -0.3 else "")
    print(f"  p={p:<13} | {mape_agg[cfg]:>9.5f} | {d:>+9.2f}%{mark}")


# ══════════════════════════════════════════════════════════════════════════════
#  ГРАФИКИ
# ══════════════════════════════════════════════════════════════════════════════

COMP_COLORS = {3: "tab:blue", 4: "tab:green", 5: "tab:orange"}
horizons = np.arange(1, VAL_H + 1)

# ── Рис 1: MAE vs p (h=1 и h=5) по компонентам ───────────────────────────────

fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig1.suptitle(
    f"MAE компоненты vs p  |  ξ=max(63, 3(p+1))  |  val_h={VAL_H}",
    fontsize=12, fontweight="bold"
)

for ax_idx, ci in enumerate(SLOW_IDX):
    ax = axes1[ax_idx]
    p_vals = sorted(mae_sweep[ci].keys())
    mae_h1 = [float(mae_sweep[ci][p][0]) for p in p_vals]
    mae_h5 = [float(mae_sweep[ci][p][4]) for p in p_vals]

    ax.plot(p_vals, mae_h1, "o-", color="steelblue", lw=2, ms=6, label="MAE @ h=1")
    ax.plot(p_vals, mae_h5, "s--", color="darkorange", lw=2, ms=6, label="MAE @ h=5")

    # стандарт и оптимум
    ax.axvline(P_STD, color="red", lw=1.5, ls="--", label=f"p_std={P_STD}")
    ax.axvline(opt_p[ci], color="green", lw=2, ls=":", label=f"p_opt={opt_p[ci]}")
    ax.axvline(COMP_PMIN[ci], color="gray", lw=1.2, ls=":", alpha=0.7,
               label=f"p_min_rule={COMP_PMIN[ci]}")

    ax.set_title(COMP_NAMES[ci], fontsize=10)
    ax.set_xlabel("p (embedding dim)", fontsize=9)
    ax.set_ylabel("MAE компоненты", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.35)

plt.tight_layout()
out1 = OUT_DIR / "47_p_sweep_mae.png"
fig1.savefig(out1, dpi=140, bbox_inches="tight")
print(f"\n  Рис 1: {out1}")
plt.close(fig1)


# ── Рис 2: MAE vs горизонт (std p=20 vs opt_p) ───────────────────────────────

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
fig2.suptitle(
    f"MAE vs горизонт: p=20 (std) vs per-component opt_p",
    fontsize=12, fontweight="bold"
)

for ax_idx, ci in enumerate(SLOW_IDX):
    ax = axes2[ax_idx]
    m_std = mae_sweep[ci].get(P_STD)
    m_opt = mae_sweep[ci].get(opt_p[ci])
    if m_std is not None:
        ax.plot(horizons, m_std, "r--", lw=2, label=f"p=20 (std)", marker="o", ms=4, markevery=2)
    if m_opt is not None and opt_p[ci] != P_STD:
        ax.plot(horizons, m_opt, "g-", lw=2.5,
                label=f"p={opt_p[ci]} (opt)", marker="s", ms=4, markevery=2)
    ax.set_title(COMP_NAMES[ci], fontsize=10)
    ax.set_xlabel("Горизонт h", fontsize=9)
    ax.set_ylabel("MAE компоненты", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)
    ax.set_xlim(1, VAL_H)

plt.tight_layout()
out2 = OUT_DIR / "47_mae_vs_horizon.png"
fig2.savefig(out2, dpi=140, bbox_inches="tight")
print(f"  Рис 2: {out2}")
plt.close(fig2)


# ── Рис 3: MAPE(p) — глобальная кривая + per-component opt ──────────────────

fig3, ax3 = plt.subplots(figsize=(11, 5))
p_global = [p for p in GLOBAL_P_SWEEP]
mape_global = [mape_agg[f"p{p}"] for p in p_global]
deltas_global = [(m - mape_std) / mape_std * 100 for m in mape_global]

ax3.plot(p_global, mape_global, "k-o", lw=2.5, ms=7, zorder=5, label="Uniform p (все C3-C5)")
ax3.axhline(mape_std, color="red", lw=1.5, ls="--", label=f"std p=20 (MAPE={mape_std:.5f})")
ax3.axhline(mape_agg["opt_per"], color="green", lw=2, ls=":",
            label=f"per-comp opt (MAPE={mape_agg['opt_per']:.5f})")

for ci in SLOW_IDX:
    ax3.axvline(opt_p[ci], color=COMP_COLORS[ci], lw=1.5, ls=":", alpha=0.6,
                label=f"opt_p C{ci}={opt_p[ci]}")

ax3.set_xlabel("p (единый для всех компонент)", fontsize=10)
ax3.set_ylabel("Медиана AGG MAPE", fontsize=10)
ax3.set_title(
    f"Реконструированный MAPE vs p  |  8 тикеров 1d, val_h={VAL_H}",
    fontsize=10
)
ax3.legend(fontsize=9, loc="upper right")
ax3.grid(alpha=0.35)

# Второй вид: Δ%
ax3r = ax3.twinx()
ax3r.bar(p_global, deltas_global,
         color=["seagreen" if d < 0 else "tomato" for d in deltas_global],
         alpha=0.2, width=1.5)
ax3r.axhline(0, color="gray", lw=0.8, ls="--")
ax3r.set_ylabel("Δ% vs стандарт", fontsize=9, color="gray")
ax3r.tick_params(axis="y", labelcolor="gray")

plt.tight_layout()
out3 = OUT_DIR / "47_mape_vs_p.png"
fig3.savefig(out3, dpi=140, bbox_inches="tight")
print(f"  Рис 3: {out3}")
plt.close(fig3)


# ── Итоги ─────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("ИТОГИ")
print("=" * 70)

print("\n1. Оптимальный p по компонентам:")
for ci in SLOW_IDX:
    ref  = mae_sweep[ci].get(P_STD)
    best = mae_sweep[ci].get(opt_p[ci])
    if ref is not None and best is not None:
        d = (float(np.mean(best[:5])) - float(np.mean(ref[:5]))) / (float(np.mean(ref[:5]))+EPS) * 100
        rule = "OK" if opt_p[ci] >= COMP_PMIN[ci] else f"< p_min_rule ({COMP_PMIN[ci]})"
        print(f"   {COMP_NAMES[ci]}: p={opt_p[ci]} (ξ={xi_for_p(opt_p[ci])})  "
              f"Δ MAE h=1..5: {d:+.2f}%  rule: {rule}")

print(f"\n2. Реконструированный MAPE:")
print(f"   std (p=20 uniform):      {mape_std:.5f}")
print(f"   opt_per (per-comp p):    {mape_agg['opt_per']:.5f}  "
      f"Δ={((mape_agg['opt_per']-mape_std)/mape_std*100):+.2f}%")

best_global = min((p for p in GLOBAL_P_SWEEP if f"p{p}" in mape_agg),
                  key=lambda p: mape_agg[f"p{p}"])
print(f"   best uniform p={best_global}: {mape_agg[f'p{best_global}']:.5f}  "
      f"Δ={((mape_agg[f'p{best_global}']-mape_std)/mape_std*100):+.2f}%")

print(f"\nФайлы:")
for out in [out1, out2, out3]:
    print(f"  {out.name}")
print("=" * 70)
