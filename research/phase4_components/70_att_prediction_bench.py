"""
70 — Benchmark: методы оценки att_oracle[origin] по каузальным данным.

LP групповая задержка τ≈7б: att_causal[t] ≈ att_oracle[t − τ].
att_causal[origin] отстаёт от «истинного» LP-состояния на τ шагов.
Задача: оценить att_oracle[origin] используя только данные до origin.
Лучший метод → Pipeline E в скрипте 69.

Целевая траектория: att_oracle[origin−τ+1 .. origin]  (длина τ)
  Позиция j предикта att_pred[j] ≈ att_oracle[origin−τ+j+1]
  Финальная точка att_pred[τ−1] ≈ att_oracle[origin]  ← ключевой таргет

Методы:
  flat              : att_causal[origin] (= Pipeline A, baseline)
  linear_W{10,20,50}: линейная экстраполяция по последним W точкам att
  ar_att_p{5,10,20} : AR(p) на att, τ итерационных шагов вперёд
  lwr_p{5,10,20}_xi{}: LWR на att с последовательными лагами, τ шагов
  ar_dratio         : Pipeline B (AR-filtfilt на dratio)
  tau_oracle        : Pipeline D (реальные τ баров dratio, верхняя граница)

Метрики:
  mae_final : |att_pred[τ−1] − att_oracle[origin]|              (ключевая)
  mae_traj  : mean(|att_pred[j] − att_oracle[origin−τ+j+1]|)    (траектория)
  mae_norm  : mae_final / std(att_oracle[origin−50:origin])       (б/р)
  da_final  : знак поправки (att_pred[τ−1] − att_causal[origin]) == знаку ошибки

Протокол: SBER, LKOH, CHMF, MRKP; N_ORIG=100.
"""

import sys, time, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfiltfilt, group_delay
from scipy.stats import ttest_rel

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
FIG_DIR  = ROOT / "research/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ─────────────────────────────────────────────────────────────────

TICKERS      = ["SBER", "LKOH", "CHMF", "MRKP"]
N_ORIG       = 100
WN           = 0.125
FILTER_ORDER = 4
AR_PAD       = 40
AR_ORDER     = 20
STD_WIN      = 50   # окно для нормировки MAE
MIN_HIST     = 800  # минимальная история att для LWR

_SOS_LP = butter(FILTER_ORDER, WN, btype="low", output="sos")


def compute_group_delay_dc(sos):
    from scipy.signal import sos2tf
    b, a = sos2tf(sos)
    _, gd = group_delay((b, a), w=1, whole=False)
    return float(gd[0])


TAU = int(round(compute_group_delay_dc(_SOS_LP)))
print(f"Групповая задержка LP: {compute_group_delay_dc(_SOS_LP):.2f} → TAU={TAU} баров")

# ── утилиты данных ─────────────────────────────────────────────────────────────

def logtrend_causal(close):
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64); cn = np.arange(1, n+1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t**2)
    cy = np.cumsum(lc); cty = np.cumsum(t*lc)
    denom = cn*ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn*cty - ct*cy)/denom, 0.0)
    a = (cy - b*ct)/cn
    trend = np.exp(a + b*t); trend[:2] = close[:2]
    return trend


def load_data(ticker):
    data   = json.loads((DATA_DIR / ticker / "1d.json").read_text())
    close  = np.array([c["close"] for c in data], dtype=np.float64)
    trend  = logtrend_causal(close)
    ratio  = close / np.maximum(trend, 1e-10)
    dratio = np.diff(ratio, prepend=ratio[0])
    att_c  = sosfilt(_SOS_LP, dratio)
    att_o  = sosfiltfilt(_SOS_LP, dratio)
    return dratio, att_c, att_o


# ── AR-утилиты ─────────────────────────────────────────────────────────────────

def ar_fit(x, order):
    n = len(x)
    rows = min(n - order, 500)
    start = n - order - rows
    X = np.column_stack([x[start+i:start+i+rows] for i in range(order)])
    y = x[start+order:start+order+rows]
    a, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return a


def ar_predict_steps(x, a, tau):
    p = len(a)
    buf = list(x[-p:])
    out = []
    for _ in range(tau):
        nxt = float(np.dot(a, buf[-p:][::-1]))
        out.append(nxt)
        buf.append(nxt)
    return np.array(out)


def ar_extend_forward(x, order, n_extend):
    if len(x) < order + 1:
        return np.concatenate([x, np.zeros(n_extend)])
    a = ar_fit(x, order)
    return np.concatenate([x, ar_predict_steps(x, a, n_extend)])


# ── предикторы att ──────────────────────────────────────────────────────────────

def pred_flat(att_hist):
    """Константа: att_causal[origin] на все τ шагов."""
    return np.full(TAU, att_hist[-1])


def pred_linear(att_hist, W):
    """Линейная экстраполяция по последним W точкам att."""
    x = att_hist[-W:]
    t = np.arange(len(x), dtype=float)
    p = np.polyfit(t, x, 1)
    return np.polyval(p, np.arange(len(x), len(x)+TAU, dtype=float))


def pred_ar_att(att_hist, p):
    """AR(p) на att, итеративный прогноз τ шагов."""
    if len(att_hist) < p + 2:
        return pred_flat(att_hist)
    a = ar_fit(att_hist, p)
    return ar_predict_steps(att_hist, a, TAU)


def pred_lwr_att(att_hist, p_embed, xi):
    """LWR на att (последовательные лаги), итеративный прогноз τ шагов."""
    n = len(att_hist)
    if n <= p_embed + 2:
        return pred_flat(att_hist)
    lags  = np.arange(p_embed-1, -1, -1, dtype=np.int32)
    t_arr = np.arange(p_embed-1, n-1, dtype=np.int32)
    X     = np.column_stack([att_hist[t_arr - lag] for lag in lags])
    y_tr  = att_hist[t_arr + 1]
    xi_eff = min(xi, len(X))
    if xi_eff < p_embed + 2:
        return pred_flat(att_hist)
    buf = np.empty(n + TAU); buf[:n] = att_hist
    out = np.empty(TAU)
    for h in range(TAU):
        t   = n + h - 1
        vec = buf[t - lags]
        dists = np.linalg.norm(X - vec, axis=1)
        nn    = np.argpartition(dists, xi_eff-1)[:xi_eff]
        h_bw  = max(float(dists[nn].max()), 1e-10)
        X_nn  = X[nn]; y_nn = y_tr[nn]
        w     = np.exp(-0.5*(np.linalg.norm(X_nn-vec, axis=1)/h_bw)**2)
        A     = np.hstack([np.ones((xi_eff,1)), X_nn])
        sw    = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:,None]*A, sw*y_nn, rcond=None)
        out[h] = float(c[0] + vec @ c[1:]); buf[t+1] = out[h]
    return out


def pred_ar_dratio(dratio_hist):
    """Pipeline B: AR-filtfilt на dratio → att_B[origin] как фиксированная оценка."""
    ext   = ar_extend_forward(dratio_hist, AR_ORDER, AR_PAD)
    att_b = sosfiltfilt(_SOS_LP, ext)
    # att_b[len(dratio_hist)-1] — оценка att при последнем баре истории
    return np.full(TAU, att_b[len(dratio_hist) - 1])


def pred_tau_oracle(dratio_full, origin_k):
    """Pipeline D: filtfilt с реальными τ барами после origin (верхняя граница)."""
    dr_seg = dratio_full[:origin_k + TAU]
    dr_pad = ar_extend_forward(dr_seg, AR_ORDER, AR_PAD)
    att_d  = sosfiltfilt(_SOS_LP, dr_pad)
    return np.full(TAU, att_d[origin_k - 1])


# ── диспетчер предикторов ────────────────────────────────────────────────────

def run_predictor(name, att_hist, dratio_hist, dratio_full, origin_k):
    if name == "flat":
        return pred_flat(att_hist)
    if name.startswith("linear_W"):
        W = int(name[len("linear_W"):])
        return pred_linear(att_hist, W)
    if name.startswith("ar_att_p"):
        p = int(name[len("ar_att_p"):])
        return pred_ar_att(att_hist, p)
    if name.startswith("lwr_"):
        # формат: lwr_p{P}_xi{XI}
        parts = name.split("_")          # ["lwr", "p5", "xi200"]
        p_embed = int(parts[1][1:])
        xi      = int(parts[2][2:])
        return pred_lwr_att(att_hist, p_embed, xi)
    if name == "ar_dratio":
        return pred_ar_dratio(dratio_hist)
    if name == "tau_oracle":
        return pred_tau_oracle(dratio_full, origin_k)
    raise ValueError(f"Unknown predictor: {name}")


# ── список методов ───────────────────────────────────────────────────────────

METHOD_NAMES = [
    "flat",
    "linear_W10", "linear_W20", "linear_W50",
    "ar_att_p5",  "ar_att_p10", "ar_att_p20",
    "lwr_p5_xi63", "lwr_p5_xi200",
    "lwr_p10_xi200", "lwr_p10_xi500",
    "lwr_p20_xi500",
    "ar_dratio",
    "tau_oracle",
]

# человекочитаемые метки
METHOD_LABELS = {
    "flat":          "flat (Pipeline A)",
    "linear_W10":    "linear W=10",
    "linear_W20":    "linear W=20",
    "linear_W50":    "linear W=50",
    "ar_att_p5":     "AR-att p=5",
    "ar_att_p10":    "AR-att p=10",
    "ar_att_p20":    "AR-att p=20",
    "lwr_p5_xi63":   "LWR p=5 ξ=63",
    "lwr_p5_xi200":  "LWR p=5 ξ=200",
    "lwr_p10_xi200": "LWR p=10 ξ=200",
    "lwr_p10_xi500": "LWR p=10 ξ=500",
    "lwr_p20_xi500": "LWR p=20 ξ=500",
    "ar_dratio":     "AR-dratio (Pipeline B)",
    "tau_oracle":    "τ-oracle (Pipeline D, утечка!)",
}


# ── оценка одного предиктора ─────────────────────────────────────────────────

def eval_predictor(att_pred, att_oracle_full, origin_k, att_causal_at_origin):
    """
    att_pred            : np.array shape (TAU,) — τ предсказанных шагов
    att_oracle_full     : полный ряд att_oracle (sosfiltfilt)
    origin_k            : индекс origin
    att_causal_at_origin: att_causal[origin_k-1] — последнее каузальное значение

    Целевая траектория att_oracle[origin_k-TAU .. origin_k-1] (длина TAU).
    Финальная точка: att_oracle[origin_k-1].
    """
    if origin_k < TAU:
        return dict(mae_final=np.nan, mae_traj=np.nan, mae_norm=np.nan, da_final=np.nan)

    target       = att_oracle_full[origin_k - TAU: origin_k]  # shape (TAU,)
    oracle_final = float(target[-1])
    causal_val   = float(att_causal_at_origin)

    ws      = max(0, origin_k - STD_WIN)
    att_std = float(np.std(att_oracle_full[ws:origin_k]) + 1e-12)

    mae_final = abs(float(att_pred[-1]) - oracle_final)
    mae_traj  = float(np.mean(np.abs(att_pred - target)))
    mae_norm  = mae_final / att_std

    # направление поправки: att_pred должен двигаться в сторону oracle_final
    delta_oracle = oracle_final - causal_val
    delta_pred   = float(att_pred[-1]) - causal_val
    da_final = 1.0 if np.sign(delta_oracle) == np.sign(delta_pred) else 0.0

    return dict(mae_final=mae_final, mae_traj=mae_traj, mae_norm=mae_norm, da_final=da_final)


# ── загрузка данных ───────────────────────────────────────────────────────────

print("\nЗагрузка данных…", flush=True)
all_dratio, all_att_c, all_att_o = {}, {}, {}
all_origins = {}

for ticker in TICKERS:
    dr, att_c, att_o = load_data(ticker)
    all_dratio[ticker] = dr
    all_att_c[ticker]  = att_c
    all_att_o[ticker]  = att_o
    n       = len(dr)
    end_k   = n - TAU - 1
    start_k = max(MIN_HIST + TAU, end_k - N_ORIG * 5)
    cands   = list(range(start_k, end_k))
    step    = max(1, len(cands) // N_ORIG)
    all_origins[ticker] = cands[::step][:N_ORIG]
    print(f"  {ticker}: {len(all_origins[ticker])} origins, n={n}", flush=True)


# ── прогон ────────────────────────────────────────────────────────────────────

# results[method][ticker] = list of metric dicts
results = {m: {t: [] for t in TICKERS} for m in METHOD_NAMES}

t0     = time.time()
n_done = 0
total  = sum(len(all_origins[t]) for t in TICKERS)
print(f"\nПрогон: {len(METHOD_NAMES)} методов × {total} origins…", flush=True)

for ticker in TICKERS:
    dratio_full = all_dratio[ticker]
    att_c_full  = all_att_c[ticker]
    att_o_full  = all_att_o[ticker]

    for origin_k in all_origins[ticker]:
        att_hist     = att_c_full[:origin_k]
        dratio_hist  = dratio_full[:origin_k]
        causal_at_ok = float(att_c_full[origin_k - 1])

        for mname in METHOD_NAMES:
            att_pred = run_predictor(mname, att_hist, dratio_hist, dratio_full, origin_k)
            m = eval_predictor(att_pred, att_o_full, origin_k, causal_at_ok)
            results[mname][ticker].append(m)

        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{total}  ({time.time()-t0:.0f}с)", flush=True)

print(f"Завершено за {time.time()-t0:.1f}с", flush=True)


# ── агрегация ─────────────────────────────────────────────────────────────────

def agg(mname, metric):
    vals = [m[metric] for t in TICKERS for m in results[mname][t] if np.isfinite(m[metric])]
    return float(np.mean(vals)) if vals else np.nan

def flat_vals(mname, metric):
    return [m[metric] for t in TICKERS for m in results[mname][t] if np.isfinite(m[metric])]


# ── вывод: сводная таблица ────────────────────────────────────────────────────

print(f"\n── Benchmark τ-шагового предсказания att (τ={TAU} баров) ─────────────────────────")
print(f"  {'Метод':<30}  {'mae_final':>10}  {'mae_norm':>9}  {'mae_traj':>9}  {'da_final':>8}")
print(f"  {'─'*30}  {'─'*10}  {'─'*9}  {'─'*9}  {'─'*8}")

flat_mae = agg("flat", "mae_final")
rows_sorted = sorted(METHOD_NAMES, key=lambda m: agg(m, "mae_final"))
for mname in rows_sorted:
    mf = agg(mname, "mae_final")
    mn = agg(mname, "mae_norm")
    mt = agg(mname, "mae_traj")
    da = agg(mname, "da_final")
    rel = (flat_mae / mf - 1) * 100 if mname != "flat" and mf > 0 else 0.0
    marker = " ←" if mname == "tau_oracle" else ("  ★" if rel == max((flat_mae/agg(m,"mae_final")-1)*100 for m in METHOD_NAMES if m not in ("flat","tau_oracle") and agg(m,"mae_final") > 0) else "")
    print(f"  {METHOD_LABELS[mname]:<30}  {mf:>10.6f}  {mn:>9.3f}  {mt:>9.6f}  {da:>7.1%}   Δ={rel:+.1f}%{marker}")

# ── t-test: лучший vs flat ────────────────────────────────────────────────────

best_non_oracle = min(
    (m for m in METHOD_NAMES if m not in ("flat", "tau_oracle")),
    key=lambda m: agg(m, "mae_final")
)
fa = flat_vals("flat",          "mae_final")
fb = flat_vals(best_non_oracle, "mae_final")
n  = min(len(fa), len(fb))
if n > 2:
    _, pval = ttest_rel(fa[:n], fb[:n])
    d = (np.mean(fa[:n]) / np.mean(fb[:n]) - 1) * 100
    print(f"\nt-test лучший ({best_non_oracle}) vs flat: Δ={d:+.2f}%  p={pval:.4f}  {'✓' if pval<0.05 else '✗'}")

# ── per-ticker ────────────────────────────────────────────────────────────────

print(f"\n── Per-ticker mae_norm ─────────────────────────────────────────────────────────")
top5 = rows_sorted[:5]
header = f"  {'Ticker':<6}"
for m in top5:
    header += f"  {METHOD_LABELS[m][:14]:>14}"
header += f"  {'tau_oracle':>14}"
print(header)
print("  " + "─"*6 + ("  " + "─"*14) * (len(top5)+1))
for ticker in TICKERS:
    row = f"  {ticker:<6}"
    for m in top5:
        v = float(np.mean([r["mae_norm"] for r in results[m][ticker] if np.isfinite(r["mae_norm"])]))
        row += f"  {v:>14.3f}"
    vo = float(np.mean([r["mae_norm"] for r in results["tau_oracle"][ticker] if np.isfinite(r["mae_norm"])]))
    row += f"  {vo:>14.3f}"
    print(row)

# ── распределение ошибок: квантили ────────────────────────────────────────────

print(f"\n── Квантили mae_norm (p10 / p50 / p90) ─────────────────────────────────────────")
for mname in rows_sorted[:8]:  # топ-8
    v = flat_vals(mname, "mae_norm")
    q = np.percentile(v, [10, 50, 90])
    print(f"  {METHOD_LABELS[mname]:<30}  p10={q[0]:.3f}  p50={q[1]:.3f}  p90={q[2]:.3f}")

# ── прирост от каузального ────────────────────────────────────────────────────

print(f"\n── Сравнение с flat: Δmae_final (%) ─────────────────────────────────────────────")
print(f"  flat mae_final  AGG = {flat_mae:.6f}")
for mname in rows_sorted:
    if mname == "flat":
        continue
    mf = agg(mname, "mae_final")
    rel = (flat_mae / mf - 1) * 100
    bar = "█" * max(0, int(rel / 2))
    print(f"  {METHOD_LABELS[mname]:<30}  {rel:+6.1f}%  {bar}")


# ── рисунок ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. mae_norm по методам (сортировано)
ax = axes[0]
names_plot = [m for m in rows_sorted if m != "tau_oracle"]
vals_mn = [agg(m, "mae_norm") for m in names_plot]
labels_plot = [METHOD_LABELS[m] for m in names_plot]
colors = ["#B71C1C" if m == "flat"
          else "#1565C0" if m == "ar_dratio"
          else "#2E7D32" if m.startswith("lwr")
          else "#E65100" if m.startswith("ar_att")
          else "#6A1B9A"
          for m in names_plot]
bars = ax.barh(range(len(names_plot)), vals_mn, color=colors, alpha=0.8, edgecolor="white")
oracle_mn = agg("tau_oracle", "mae_norm")
ax.axvline(oracle_mn, color="#43A047", lw=2, ls="--", label=f"τ-oracle: {oracle_mn:.3f}")
ax.set_yticks(range(len(names_plot)))
ax.set_yticklabels(labels_plot, fontsize=8)
ax.set_xlabel("mae_norm (меньше = лучше)")
ax.set_title(f"MAE финальной точки / att_std\n(τ={TAU} шагов вперёд)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="x")
ax.invert_yaxis()

# 2. da_final по методам
ax = axes[1]
vals_da = [agg(m, "da_final") for m in rows_sorted]
labels_da = [METHOD_LABELS[m] for m in rows_sorted]
c2 = ["#43A047" if m == "tau_oracle"
      else "#B71C1C" if m == "flat"
      else "#1565C0" if m == "ar_dratio"
      else "#2E7D32" if m.startswith("lwr")
      else "#E65100" if m.startswith("ar_att")
      else "#6A1B9A"
      for m in rows_sorted]
ax.barh(range(len(rows_sorted)), vals_da, color=c2, alpha=0.8, edgecolor="white")
ax.axvline(0.5, color="black", lw=1, ls="--", alpha=0.5, label="random")
ax.set_yticks(range(len(rows_sorted)))
ax.set_yticklabels(labels_da, fontsize=8)
ax.set_xlabel("Direction accuracy финальной точки")
ax.set_title("DA: верно указано направление\nпоправки att (vs flat baseline)")
ax.set_xlim(0, 1)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="x")
ax.invert_yaxis()

# 3. mae_norm: scatter flat vs best по origin
ax = axes[2]
best_m = rows_sorted[0] if rows_sorted[0] != "flat" else rows_sorted[1]
flat_per = flat_vals("flat", "mae_norm")
best_per = flat_vals(best_m, "mae_norm")
n_sc = min(len(flat_per), len(best_per), 400)
ax.scatter(flat_per[:n_sc], best_per[:n_sc], s=6, alpha=0.4, color="#1565C0")
lim_max = max(max(flat_per[:n_sc]), max(best_per[:n_sc]))
ax.plot([0, lim_max], [0, lim_max], "k--", lw=1, alpha=0.5)
ax.set_xlabel(f"flat  mae_norm")
ax.set_ylabel(f"{METHOD_LABELS[best_m]}  mae_norm")
ax.set_title(f"flat vs лучший метод\n(точки ниже диагонали = улучшение)")
ax.grid(True, alpha=0.3)

fig.suptitle(
    f"Benchmark: τ={TAU}-шаговое предсказание att_oracle[origin]  "
    f"({len(TICKERS)} тикера, {N_ORIG} origins каждый)",
    fontsize=11
)
fig.tight_layout()
fig_path = FIG_DIR / "70_att_prediction_bench.png"
fig.savefig(fig_path, dpi=150)
plt.close(fig)

print(f"\nРис.: {fig_path}")
print(f"\nСкрипт 70 завершён за {time.time()-t0:.1f}с")
