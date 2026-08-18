"""
69 — Краевая коррекция LP-фильтра: AR-filtfilt vs causal.

Проблема: causal Butterworth LP (Wn=0.125, order=4) вносит групповую задержку τ.
att_causal[t] ≠ истинный C2-C5[t] — отстаёт на τ баров.
Первые τ баров прогноза «догоняют» прошлое, а не предсказывают будущее.

Три пайплайна (одна и та же LWR-модель поверху):
  A  causal       sosfilt(dratio[0:t])                     ← текущий baseline
  B  ar-filtfilt  sosfiltfilt(ar_extend(dratio[0:t], pad)) ← каузальный, без утечки
  C  oracle       sosfiltfilt(dratio[0:N])                 ← утечка из будущего, верхняя граница

Методы LWR: uniform_p80 (xi=500) и c2c5_k13 (xi=700).

Метрики:
  mape_full      — MAPE по всему горизонту [0:H]
  mape_trim      — MAPE по [τ:H] (пропускаем зону краевого эффекта)
  dir_acc        — direction accuracy: знак Δratio[τ:H] угадан верно, %

Протокол: SBER, LKOH, CHMF, MRKP; N_ORIG=50; horizon=20.
"""

import sys
import time
import json
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

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS      = ["SBER", "LKOH", "CHMF", "MRKP"]
N_ORIG       = 50
HORIZON      = 20
WN           = 0.125
FILTER_ORDER = 4
AR_PAD       = 40      # баров AR-экстраполяции для filtfilt (должно быть >> τ)
AR_ORDER     = 20      # порядок AR-модели для экстраполяции края

# методы LWR
METHODS = {
    "uniform_p80 xi=500": (list(range(79, -1, -1)), 500, 13),
    "c2c5_k13 xi=700":    ([130, 103, 78, 52, 40, 26, 16, 12, 8, 5, 2, 1, 0], 700, 13),
}

_SOS_LP = butter(FILTER_ORDER, WN, btype="low", output="sos")


# ── групповая задержка ─────────────────────────────────────────────────────────

def compute_group_delay_dc(sos):
    """Групповая задержка фильтра при ω=0 (DC)."""
    from scipy.signal import sos2tf
    b, a = sos2tf(sos)
    w, gd = group_delay((b, a), w=1, whole=False)
    return float(gd[0])

TAU    = int(round(compute_group_delay_dc(_SOS_LP)))
LP_WIN = 50
print(f"Групповая задержка LP-фильтра: τ = {compute_group_delay_dc(_SOS_LP):.2f} → TAU={TAU} баров, LP_WIN={LP_WIN}")


# ── AR-экстраполяция края ──────────────────────────────────────────────────────

def ar_extend_forward(x, order, n_extend):
    """Продлить ряд x на n_extend баров с помощью AR(order)."""
    if len(x) < order + 1:
        return np.concatenate([x, np.zeros(n_extend)])
    # OLS: X * a = y
    n = len(x)
    rows = min(n - order, 500)   # используем до 500 последних точек
    start = n - order - rows
    X = np.column_stack([x[start + i: start + i + rows] for i in range(order)])
    y = x[start + order: start + order + rows]
    a, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    buf = list(x[-order:])
    ext = []
    for _ in range(n_extend):
        nxt = float(np.dot(a, buf[-order:][::-1]))
        ext.append(nxt)
        buf.append(nxt)
    return np.concatenate([x, ext])


# ── три варианта построения att_hist ──────────────────────────────────────────

def build_att_causal(dratio_hist):
    """Pipeline A: обычный causal sosfilt."""
    return sosfilt(_SOS_LP, dratio_hist)


def build_att_ar_filtfilt(dratio_hist):
    """
    Pipeline B: filtfilt без утечки из будущего.
    dratio[0:t] → AR-extend на AR_PAD баров → filtfilt → взять [0:t].
    Правый краевой артефакт filtfilt падает в AR-хвост и выбрасывается.
    """
    ext = ar_extend_forward(dratio_hist, AR_ORDER, AR_PAD)
    att_ext = sosfiltfilt(_SOS_LP, ext)
    return att_ext[:len(dratio_hist)]


def build_att_tau_oracle(dratio_full, origin_k):
    """
    Pipeline D: filtfilt с минимальным lookahead τ баров.
    Использует dratio[0:t+τ] (реальные данные) + AR_PAD для стабилизации правого края.
    В live trading недоступен — показывает потолок для Pipeline B.
    Разрыв B→D = потери от того, что AR не идеально предсказывает next-τ baров dratio.
    """
    dr_segment = dratio_full[:origin_k + TAU]
    dr_padded  = ar_extend_forward(dr_segment, AR_ORDER, AR_PAD)
    att_ext    = sosfiltfilt(_SOS_LP, dr_padded)
    return att_ext[:origin_k]


# ── price-MAPE helpers ────────────────────────────────────────────────────────

def lp_bias_ratio_j(att_hist, ratio, logtrend, close, origin_k):
    """ratio_j с bias-коррекцией LP для прогноза в ценах."""
    ws    = max(0, origin_k - LP_WIN + 1)
    lp_r  = ratio[ws] + np.concatenate([[0.0], np.cumsum(att_hist[ws:origin_k])])
    lp_p  = lp_r * logtrend[ws: origin_k + 1]
    barr  = np.arange(ws, origin_k + 1) - TAU
    valid = (barr >= 0) & (barr < len(close))
    bias  = float(np.mean(close[barr[valid]] - lp_p[valid])) if valid.any() else 0.0
    return (lp_p[-1] + bias) / logtrend[origin_k]


OHLC_KEYS = ("close", "open", "high", "low", "hl_mid")

def price_mape_ohlc(att_pred, origin_k, ratio_j, logtrend, ohlc, horizon):
    """Price-MAPE против close/open/high/low/(H+L)/2."""
    h = min(len(att_pred), horizon)
    if h == 0 or not np.isfinite(att_pred[:h]).all():
        return {k: np.nan for k in OHLC_KEYS}
    lt  = logtrend[origin_k + 1: origin_k + 1 + h]
    n   = min(h, len(lt))
    pp  = (ratio_j + np.cumsum(att_pred[:n])) * lt
    out = {}
    for key in ("close", "open", "high", "low"):
        p = ohlc[key][origin_k + 1: origin_k + 1 + n]
        out[key] = float(np.mean(np.abs(pp - p) / (np.abs(p) + 1e-10)))
    hl = 0.5 * (ohlc["high"][origin_k + 1: origin_k + 1 + n]
              + ohlc["low"] [origin_k + 1: origin_k + 1 + n])
    out["hl_mid"] = float(np.mean(np.abs(pp - hl) / (np.abs(hl) + 1e-10)))
    return out


# ── утилиты данных ────────────────────────────────────────────────────────────

def logtrend_causal(close):
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64); cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = np.exp(a + b * t); trend[:2] = close[:2]
    return trend


def load_data(ticker):
    data   = json.loads((DATA_DIR / ticker / "1d.json").read_text())
    close  = np.array([c["close"] for c in data], dtype=np.float64)
    open_  = np.array([c["open"]  for c in data], dtype=np.float64)
    high   = np.array([c["high"]  for c in data], dtype=np.float64)
    low    = np.array([c["low"]   for c in data], dtype=np.float64)
    trend  = logtrend_causal(close)
    ratio  = close / np.maximum(trend, 1e-10)
    dratio = np.diff(ratio, prepend=ratio[0])
    att_oracle      = sosfiltfilt(_SOS_LP, dratio)
    att_causal_full = sosfilt(_SOS_LP, dratio)
    ohlc = {"close": close, "open": open_, "high": high, "low": low}
    return dratio, ratio, trend, att_causal_full, att_oracle, ohlc


# ── LWR-ядро ──────────────────────────────────────────────────────────────────

def make_matrices(att, lags_desc, p_fit):
    max_lag = lags_desc[0]; lags_arr = np.array(lags_desc, dtype=np.int32)
    n = len(att); k = len(lags_desc)
    if n - max_lag - 1 <= 0:
        return np.zeros((0, k)), np.zeros((0, p_fit)), np.zeros(0)
    t_arr    = np.arange(max_lag, n - 1, dtype=np.int32)
    X_search = np.column_stack([att[t_arr - lag] for lag in lags_arr])
    X_fit    = np.column_stack([att[t_arr - (p_fit - 1 - k_)] for k_ in range(p_fit)])
    return X_search, X_fit, att[t_arr + 1]


def forecast_lwr(att_hist, lags_desc, p_fit, horizon, xi):
    n = len(att_hist)
    X_search, X_fit, y = make_matrices(att_hist, lags_desc, p_fit)
    xi_eff = min(xi, len(X_search))
    if xi_eff < p_fit + 2:
        return np.full(horizon, np.nan)
    lags_arr = np.array(lags_desc, dtype=np.int32)
    buf = np.empty(n + horizon); buf[:n] = att_hist
    out = np.empty(horizon)
    for h in range(horizon):
        t     = n + h - 1
        vec_s = buf[t - lags_arr]
        vec_f = buf[t - p_fit + 1: t + 1]
        dists = np.linalg.norm(X_search - vec_s, axis=1)
        nn    = np.argpartition(dists, xi_eff - 1)[:xi_eff]
        h_bw  = max(float(dists[nn].max()), 1e-10)
        X_nn  = X_fit[nn]; y_nn = y[nn]
        w     = np.exp(-0.5 * (np.linalg.norm(X_nn - vec_f, axis=1) / h_bw) ** 2)
        A     = np.hstack([np.ones((xi_eff, 1)), X_nn]); sw = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
        out[h] = float(c[0] + vec_f @ c[1:]); buf[t + 1] = out[h]
    return out


# ── метрики ────────────────────────────────────────────────────────────────────

def compute_metrics(att_pred, ratio_raw_future, ratio0, tau):
    """
    Все пайплайны сравниваются против одного эталона: ratio_raw_future —
    фактические значения ratio[t+1:t+H] из сырой цены (без фазовых артефактов).
    Это честное сравнение: не зависит от того, causal или filtfilt att используется.
    """
    h = min(len(att_pred), len(ratio_raw_future), HORIZON)
    if h == 0 or not np.isfinite(att_pred[:h]).all():
        return dict(mape_full=np.nan, mape_trim=np.nan, dir_acc=np.nan)

    r_pred = ratio0 + np.cumsum(att_pred[:h])
    r_act  = ratio_raw_future[:h]
    eps    = np.abs(r_act) + 1e-10

    mape_full = float(np.mean(np.abs(r_pred - r_act) / eps))

    # обрезанная метрика: пропускаем первые tau баров
    if tau < h:
        mape_trim = float(np.mean(np.abs(r_pred[tau:] - r_act[tau:]) / eps[tau:]))
    else:
        mape_trim = np.nan

    # direction accuracy на кумулятивном ratio[tau:]
    if tau < h - 1:
        dr_pred = np.diff(r_pred[tau:])
        dr_act  = np.diff(r_act[tau:])
        dir_acc = float(np.mean(np.sign(dr_pred) == np.sign(dr_act)))
    else:
        dir_acc = np.nan

    return dict(mape_full=mape_full, mape_trim=mape_trim, dir_acc=dir_acc)


# ── загрузка ──────────────────────────────────────────────────────────────────

print("Загрузка данных…", flush=True)
all_dratio, all_ratio, all_logtrend = {}, {}, {}
all_att_causal, all_att_oracle, all_ohlc = {}, {}, {}
all_origins = {}

max_lag_global = max(lags[0] for lags, _, _ in METHODS.values())
xi_max = max(xi for _, xi, _ in METHODS.values())

for ticker in TICKERS:
    dr, ratio, trend, att_c, att_o, ohlc = load_data(ticker)
    all_dratio[ticker]     = dr
    all_ratio[ticker]      = ratio
    all_logtrend[ticker]   = trend
    all_att_causal[ticker] = att_c
    all_att_oracle[ticker] = att_o
    all_ohlc[ticker]       = ohlc
    n_total   = len(dr)
    min_start = max_lag_global + xi_max + HORIZON + 10
    end_k     = n_total - HORIZON - 1
    start_k   = max(min_start, end_k - N_ORIG * 5)
    cands     = list(range(start_k, end_k))
    step      = max(1, len(cands) // N_ORIG)
    all_origins[ticker] = cands[::step][:N_ORIG]
    print(f"  {ticker}: {len(all_origins[ticker])} origins, n={n_total}", flush=True)


# ── прогон ────────────────────────────────────────────────────────────────────
# results[pipeline][method][ticker]  = list of old metrics dicts (mape_full/trim/dir_acc)
# presults[pipeline][method][ticker] = list of price-MAPE dicts (OHLC)

PIPELINES = ["A_causal", "B_ar_filtfilt", "D_tau_oracle", "C_oracle"]
METHOD_NAMES = list(METHODS.keys())

results  = {
    pipe: {meth: {t: [] for t in TICKERS} for meth in METHOD_NAMES}
    for pipe in PIPELINES
}
presults = {
    pipe: {meth: {t: [] for t in TICKERS} for meth in METHOD_NAMES}
    for pipe in PIPELINES
}

t0     = time.time()
n_done = 0
total  = len(TICKERS) * N_ORIG
print(f"\nПрогон: {len(PIPELINES)} pipeline × {len(METHOD_NAMES)} методов × {total} origins…", flush=True)

for ticker in TICKERS:
    dratio     = all_dratio[ticker]
    ratio      = all_ratio[ticker]
    logtrend   = all_logtrend[ticker]
    att_c_full = all_att_causal[ticker]
    att_o_full = all_att_oracle[ticker]
    ohlc       = all_ohlc[ticker]
    origins    = all_origins[ticker]

    for origin_k in origins:
        dr_hist          = dratio[:origin_k]
        ratio0           = float(ratio[origin_k])
        ratio_raw_future = ratio[origin_k + 1: origin_k + 1 + HORIZON]

        att_hists = {
            "A_causal":      build_att_causal(dr_hist),
            "B_ar_filtfilt": build_att_ar_filtfilt(dr_hist),
            "D_tau_oracle":  build_att_tau_oracle(dratio, origin_k),
            "C_oracle":      att_o_full[:origin_k],
        }

        for meth_name, (lags, xi, p_fit) in METHODS.items():
            for pipe in PIPELINES:
                att_hist = att_hists[pipe]
                pred     = forecast_lwr(att_hist, lags, p_fit, HORIZON, xi)
                m        = compute_metrics(pred, ratio_raw_future, ratio0, TAU)
                results[pipe][meth_name][ticker].append(m)
                rj = lp_bias_ratio_j(att_hist, ratio, logtrend,
                                     ohlc["close"], origin_k)
                presults[pipe][meth_name][ticker].append(
                    price_mape_ohlc(pred, origin_k, rj, logtrend, ohlc, HORIZON))

        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{total}  ({time.time()-t0:.0f}s)", flush=True)

print(f"Завершено за {time.time()-t0:.1f}с", flush=True)


# ── агрегация ─────────────────────────────────────────────────────────────────

def agg_metric(pipe, meth, metric):
    vals = [m[metric] for t in TICKERS for m in results[pipe][meth][t]]
    return float(np.nanmean(vals))

def flat_metric(pipe, meth, metric):
    return [m[metric] for t in TICKERS for m in results[pipe][meth][t] if np.isfinite(m[metric])]


# ── вывод ──────────────────────────────────────────────────────────────────────

PIPE_LABELS = {
    "A_causal":      "A causal        (baseline)",
    "B_ar_filtfilt": "B ar-filtfilt   (AR approx τ)",
    "D_tau_oracle":  "D τ-oracle      (real τ bars)",
    "C_oracle":      "C full-oracle   (утечка!)",
}

print(f"\n── Результаты (τ={TAU} баров) ────────────────────────────────────────────────")

for meth_name in METHOD_NAMES:
    print(f"\n  Метод: {meth_name}")
    print(f"  {'Pipeline':<32}  {'MAPE full':>10}  {'MAPE trim':>10}  {'Dir acc':>8}")
    print(f"  {'─'*32}  {'─'*10}  {'─'*10}  {'─'*8}")
    baseline_full = agg_metric("A_causal", meth_name, "mape_full")
    baseline_trim = agg_metric("A_causal", meth_name, "mape_trim")
    for pipe in PIPELINES:
        mf = agg_metric(pipe, meth_name, "mape_full")
        mt = agg_metric(pipe, meth_name, "mape_trim")
        da = agg_metric(pipe, meth_name, "dir_acc")
        df = (baseline_full / mf - 1) * 100 if pipe != "A_causal" else 0.0
        dt = (baseline_trim / mt - 1) * 100 if pipe != "A_causal" else 0.0
        label = PIPE_LABELS[pipe]
        print(f"  {label:<32}  {mf:>10.5f}  {mt:>10.5f}  {da:>7.1%}   Δfull={df:+.1f}%  Δtrim={dt:+.1f}%")

# сравнение B vs A: t-test
print(f"\n── t-test B vs A (ar-filtfilt vs causal) ────────────────────────────────────")
for meth_name in METHOD_NAMES:
    for metric in ("mape_full", "mape_trim"):
        fa = flat_metric("A_causal",      meth_name, metric)
        fb = flat_metric("B_ar_filtfilt", meth_name, metric)
        n  = min(len(fa), len(fb))
        if n > 2:
            _, pval = ttest_rel(fa[:n], fb[:n])
            d = (np.nanmean(fa[:n]) / np.nanmean(fb[:n]) - 1) * 100
            sig = "✓" if pval < 0.05 else "✗"
            print(f"  {meth_name:<24}  {metric:<12}  Δ={d:+.2f}%  p={pval:.4f}  {sig}")

# pipeline vs pipeline: сводная таблица по методам
print(f"\n── Сводная таблица MAPE_trim (без первых {TAU} баров) ──────────────────────────")
print(f"  {'Pipeline':<32}", end="")
for m in METHOD_NAMES:
    print(f"  {m[:20]:>20}", end="")
print()
print(f"  {'─'*32}", end="")
for _ in METHOD_NAMES:
    print(f"  {'─'*20}", end="")
print()
for pipe in PIPELINES:
    print(f"  {PIPE_LABELS[pipe]:<32}", end="")
    for m in METHOD_NAMES:
        v = agg_metric(pipe, m, "mape_trim")
        print(f"  {v:>20.5f}", end="")
    print()

print(f"\n── Direction accuracy (bars {TAU}-{HORIZON}) ──────────────────────────────────────")
print(f"  {'Pipeline':<32}", end="")
for m in METHOD_NAMES:
    print(f"  {m[:20]:>20}", end="")
print()
print(f"  {'─'*32}", end="")
for _ in METHOD_NAMES:
    print(f"  {'─'*20}", end="")
print()
for pipe in PIPELINES:
    print(f"  {PIPE_LABELS[pipe]:<32}", end="")
    for m in METHOD_NAMES:
        v = agg_metric(pipe, m, "dir_acc")
        print(f"  {v:>19.1%} ", end="")
    print()


# ── price-MAPE (OHLC) по пайплайнам ──────────────────────────────────────────

def agg_pm(pipe, meth, key):
    return np.nanmean([m[key] for t in TICKERS for m in presults[pipe][meth][t]])

print(f"\n── Price-MAPE vs OHLC (bias-corrected LP, τ={TAU}б) ─────────────────────────────")
for meth_name in METHOD_NAMES:
    print(f"\n  Метод: {meth_name}")
    hdr = f"  {'Pipeline':<32}"
    for k in OHLC_KEYS:
        hdr += f"  {k:>8}"
    print(hdr)
    print(f"  {'─'*32}" + "  ────────" * len(OHLC_KEYS))
    base_close = agg_pm("A_causal", meth_name, "close")
    for pipe in PIPELINES:
        row = f"  {PIPE_LABELS[pipe]:<32}"
        for k in OHLC_KEYS:
            v = agg_pm(pipe, meth_name, k)
            row += f"  {v:>8.5f}"
        d = (base_close / agg_pm(pipe, meth_name, "close") - 1) * 100
        suffix = "" if pipe == "A_causal" else f"  Δclose={d:+.1f}%"
        print(row + suffix)

print(f"\n── Spread OHLC (разброс min/max среди 4 компонент) ──────────────────────────")
for pipe in PIPELINES:
    print(f"\n  {PIPE_LABELS[pipe]}")
    for meth_name in METHOD_NAMES:
        vals = {k: agg_pm(pipe, meth_name, k) for k in ("close", "open", "high", "low")}
        mn, mx = min(vals.values()), max(vals.values())
        best_k = min(vals, key=vals.get)
        print(f"    {meth_name:<24}  min={mn:.5f}({best_k})  max={mx:.5f}  spread={mx/mn-1:+.1%}")


# ── рисунок ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics_plot = [
    ("mape_full",  f"MAPE full [0:{HORIZON}]"),
    ("mape_trim",  f"MAPE trim [{TAU}:{HORIZON}]"),
    ("dir_acc",    f"Direction accuracy [{TAU}:{HORIZON}]"),
]
pipe_colors = {"A_causal": "#1976D2", "B_ar_filtfilt": "#E53935",
               "D_tau_oracle": "#FF9800", "C_oracle": "#43A047"}
pipe_short  = {"A_causal": "A causal", "B_ar_filtfilt": "B ar-filt",
               "D_tau_oracle": "D τ-oracle", "C_oracle": "C oracle"}

x = np.arange(len(METHOD_NAMES)); w = 0.2
for ax, (metric, title) in zip(axes, metrics_plot):
    for i, pipe in enumerate(PIPELINES):
        vals = [agg_metric(pipe, m, metric) for m in METHOD_NAMES]
        ax.bar(x + i * w, vals, w, label=pipe_short[pipe],
               color=pipe_colors[pipe], alpha=0.85, edgecolor="white")
    ax.set_xticks(x + w); ax.set_xticklabels([m[:16] for m in METHOD_NAMES], rotation=10, fontsize=8)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
    if "dir_acc" in metric:
        ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.4, label="random")
        ax.set_ylim(0, 1)

fig.suptitle(f"Краевая коррекция LP-фильтра  (τ={TAU} баров, {len(TICKERS)} тикера, {N_ORIG} origins)",
             fontsize=11)
fig.tight_layout()
fig.savefig(FIG_DIR / "69_edge_correction.png", dpi=150)
plt.close(fig)

print(f"\nРис.: {FIG_DIR}/69_edge_correction.png")
print(f"\nСкрипт 69 завершён за {time.time()-t0:.1f}с")
