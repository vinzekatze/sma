"""
71 — Kalman LP vs Butterworth LP: сравнение att-экстракторов.

Гипотеза: Kalman-фильтр (1D/2D) извлекает att без фазовой задержки,
что должно устранить краевую проблему и улучшить LWR-прогноз
vs Butterworth + AR-filtfilt (Pipeline B).

1D Kalman (random walk):
  att_{k+1} = att_k + N(0, Q)
  dratio_k  = att_k + N(0, R)
  → при SS эквивалентен EMA; GD(DC) ≈ (1-K)/K баров.
  Чтобы GD < τ_LP=7: K > 0.125 → фильтр шумный.

2D Kalman (constant velocity):
  state = [level, slope]; slope предсказывает следующий шаг →
  эффективная задержка значительно ниже 1D.

Параметр q_factor ∈ {0.05, 0.1, 0.2, 0.4}: q = q_factor * σ_dratio.
Kalman вычисляется на полном ряду при загрузке, затем срезается по origin.

Пайплайны:
  A_lp     : LP causal Butterworth        (baseline)
  B_lp     : LP + AR-filtfilt             (краевая коррекция)
  D_lp     : LP + τ-oracle               (верхняя граница, утечка)
  KF1D_*   : 1D Kalman, q = qf·σ
  KF2D_*   : 2D Kalman, q_slope = qf·σ

LWR: c2c5_k13 xi=700 для всех пайплайнов (чемпион из скр.68).
Метрики: price-MAPE close (ключевая), ratio MAPE_full, MAPE_trim.
Протокол: SBER LKOH CHMF MRKP; N_ORIG=100.
"""

import sys, time, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfiltfilt, group_delay, lfilter
from scipy.stats import ttest_rel

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
FIG_DIR  = ROOT / "research/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS      = ["SBER", "LKOH", "CHMF", "MRKP"]
N_ORIG       = 100
HORIZON      = 20
WN           = 0.125
FILTER_ORDER = 4
AR_PAD       = 40
AR_ORDER     = 20
LP_WIN       = 50

# LWR: c2c5_k13 xi=700
LWR_LAGS  = [130, 103, 78, 52, 40, 26, 16, 12, 8, 5, 2, 1, 0]
LWR_XI    = 700
LWR_PFIT  = 13

# Kalman q_factor sweep
KF_Q_FACTORS = [0.05, 0.1, 0.2, 0.4]

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
    att_c  = sosfilt(_SOS_LP, dratio)
    att_o  = sosfiltfilt(_SOS_LP, dratio)
    ohlc   = {"close": close, "open": open_, "high": high, "low": low}
    return dratio, ratio, trend, att_c, att_o, ohlc


# ── AR-утилиты ─────────────────────────────────────────────────────────────────

def ar_extend_forward(x, order, n_extend):
    if len(x) < order + 1:
        return np.concatenate([x, np.zeros(n_extend)])
    n = len(x)
    rows = min(n - order, 500)
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


# ── LP-пайплайны att ───────────────────────────────────────────────────────────

def build_att_causal(dratio_hist):
    return sosfilt(_SOS_LP, dratio_hist)


def build_att_ar_filtfilt(dratio_hist):
    ext = ar_extend_forward(dratio_hist, AR_ORDER, AR_PAD)
    return sosfiltfilt(_SOS_LP, ext)[:len(dratio_hist)]


def build_att_tau_oracle(dratio_full, origin_k):
    dr_seg = dratio_full[:origin_k + TAU]
    dr_pad = ar_extend_forward(dr_seg, AR_ORDER, AR_PAD)
    return sosfiltfilt(_SOS_LP, dr_pad)[:origin_k]


# ── Kalman-фильтры (вычисляются на полном ряду) ───────────────────────────────

def kalman_1d_full(dratio, q_factor):
    """
    1D random-walk Kalman на полном ряду (steady-state = EMA).
    q = q_factor * σ_dratio.
    """
    sigma = float(np.std(dratio))
    if sigma < 1e-12:
        return np.zeros(len(dratio))
    Q  = (q_factor * sigma) ** 2
    R  = sigma ** 2
    # Solve scalar Riccati: P_ss^2 + Q*P_ss - Q*R = 0
    P_ss = (-Q + np.sqrt(Q ** 2 + 4.0 * Q * R)) / 2.0
    K_ss = (P_ss + Q) / (P_ss + Q + R)
    # Steady-state = EMA: y[n] = (1-K)*y[n-1] + K*x[n]
    gd_dc = (1.0 - K_ss) / K_ss if K_ss > 0 else np.inf
    return lfilter([K_ss], [1.0, -(1.0 - K_ss)], dratio), K_ss, gd_dc


def kalman_2d_full(dratio, q_factor):
    """
    2D constant-velocity Kalman на полном ряду.
    state = [level, slope]; q_slope = q_factor * σ_dratio.
    Steady-state gain из DARE.
    """
    sigma = float(np.std(dratio))
    if sigma < 1e-12:
        return np.zeros(len(dratio)), 0.0, np.inf

    q_slope = q_factor * sigma
    R_var   = sigma ** 2

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Q_mat = np.array([[0.0, 0.0], [0.0, q_slope ** 2]])

    # DARE: P = F P F^T + Q - F P H^T (H P H^T + R)^{-1} H P F^T
    try:
        from scipy.linalg import solve_discrete_are
        P_ss = solve_discrete_are(F.T, H.reshape(-1, 1), Q_mat, np.array([[R_var]]))
        S_ss = float(H @ P_ss @ H) + R_var
        K_ss = (P_ss @ H) / S_ss          # shape (2,)
    except Exception:
        # Итеративная сходимость DARE как fallback
        P = np.eye(2) * R_var
        for _ in range(500):
            P_pred = F @ P @ F.T + Q_mat
            S = float(H @ P_pred @ H) + R_var
            K = (P_pred @ H) / S
            P = (np.eye(2) - np.outer(K, H)) @ P_pred
        K_ss = K

    A_ss = (np.eye(2) - np.outer(K_ss, H)) @ F

    # Вычислить att на полном ряду
    n   = len(dratio)
    x   = np.zeros(2)
    att = np.zeros(n)
    for k in range(n):
        x      = A_ss @ x + K_ss * dratio[k]
        att[k] = x[0]

    # Эффективная задержка: оценим как нормированное отставание от oracle
    gd_approx = float(K_ss[0])  # для информации
    return att, float(K_ss[0]), float(K_ss[1])


# ── LWR-прогноз ────────────────────────────────────────────────────────────────

def forecast_lwr(att_hist, horizon):
    lags_arr = np.array(LWR_LAGS, dtype=np.int32)
    max_lag  = lags_arr[0]
    n = len(att_hist)
    if n - max_lag - 1 <= 0:
        return np.full(horizon, np.nan)
    t_arr    = np.arange(max_lag, n - 1, dtype=np.int32)
    X_search = np.column_stack([att_hist[t_arr - lag] for lag in lags_arr])
    X_fit    = np.column_stack([att_hist[t_arr - (LWR_PFIT - 1 - k_)] for k_ in range(LWR_PFIT)])
    y_train  = att_hist[t_arr + 1]
    xi_eff   = min(LWR_XI, len(X_search))
    if xi_eff < LWR_PFIT + 2:
        return np.full(horizon, np.nan)
    buf = np.empty(n + horizon); buf[:n] = att_hist
    out = np.empty(horizon)
    for h in range(horizon):
        t     = n + h - 1
        vec_s = buf[t - lags_arr]
        vec_f = buf[t - LWR_PFIT + 1: t + 1]
        dists = np.linalg.norm(X_search - vec_s, axis=1)
        nn    = np.argpartition(dists, xi_eff - 1)[:xi_eff]
        h_bw  = max(float(dists[nn].max()), 1e-10)
        X_nn  = X_fit[nn]; y_nn = y_train[nn]
        w     = np.exp(-0.5 * (np.linalg.norm(X_nn - vec_f, axis=1) / h_bw) ** 2)
        A     = np.hstack([np.ones((xi_eff, 1)), X_nn]); sw = np.sqrt(w)
        c, _, _, _ = np.linalg.lstsq(sw[:, None] * A, sw * y_nn, rcond=None)
        out[h] = float(c[0] + vec_f @ c[1:]); buf[t + 1] = out[h]
    return out


# ── bias-corrected ratio_j ─────────────────────────────────────────────────────

def compute_ratio_j(att_hist, ratio, logtrend, close, origin_k, tau_shift):
    """
    Bias-corrected ratio_j для прогноза в ценах.
    tau_shift = TAU для LP (фазовая задержка), 0 для Kalman.
    """
    ws   = max(0, origin_k - LP_WIN + 1)
    lp_r = ratio[ws] + np.concatenate([[0.0], np.cumsum(att_hist[ws:origin_k])])
    lp_p = lp_r * logtrend[ws: origin_k + 1]
    barr  = np.arange(ws, origin_k + 1) - tau_shift
    valid = (barr >= 0) & (barr < len(close))
    bias  = float(np.mean(close[barr[valid]] - lp_p[valid])) if valid.any() else 0.0
    return (lp_p[-1] + bias) / logtrend[origin_k]


# ── метрики ─────────────────────────────────────────────────────────────────────

def compute_ratio_metrics(att_pred, ratio_raw_future, ratio0):
    h = min(len(att_pred), len(ratio_raw_future), HORIZON)
    if h == 0 or not np.isfinite(att_pred[:h]).all():
        return dict(mape_full=np.nan, mape_trim=np.nan, dir_acc=np.nan)
    r_pred = ratio0 + np.cumsum(att_pred[:h])
    r_act  = ratio_raw_future[:h]
    eps    = np.abs(r_act) + 1e-10
    mape_full = float(np.mean(np.abs(r_pred - r_act) / eps))
    if TAU < h:
        mape_trim = float(np.mean(np.abs(r_pred[TAU:] - r_act[TAU:]) / eps[TAU:]))
    else:
        mape_trim = np.nan
    if TAU < h - 1:
        dir_acc = float(np.mean(np.sign(np.diff(r_pred[TAU:])) == np.sign(np.diff(r_act[TAU:]))))
    else:
        dir_acc = np.nan
    return dict(mape_full=mape_full, mape_trim=mape_trim, dir_acc=dir_acc)


def compute_price_mape(att_pred, origin_k, ratio_j, logtrend, close):
    h  = min(len(att_pred), HORIZON)
    lt = logtrend[origin_k + 1: origin_k + 1 + h]
    n  = min(h, len(lt))
    if n == 0 or not np.isfinite(att_pred[:n]).all():
        return np.nan
    pp = (ratio_j + np.cumsum(att_pred[:n])) * lt
    p  = close[origin_k + 1: origin_k + 1 + n]
    return float(np.mean(np.abs(pp - p) / (np.abs(p) + 1e-10)))


# ── загрузка данных + предвычисление Kalman ───────────────────────────────────

print("\nЗагрузка данных и предвычисление фильтров…", flush=True)

all_dratio, all_ratio, all_logtrend = {}, {}, {}
all_att_c, all_att_o, all_ohlc = {}, {}, {}
all_att_kf  = {}   # [ticker][pipe_name] = att (полный ряд)
all_origins = {}
kf_info     = {}   # [pipe_name] → строка с параметрами

KF_NAMES_1D = [f"KF1D_q{qf}" for qf in KF_Q_FACTORS]
KF_NAMES_2D = [f"KF2D_q{qf}" for qf in KF_Q_FACTORS]
LP_PIPES    = ["A_lp", "B_lp", "D_lp"]
ALL_PIPES   = LP_PIPES + KF_NAMES_1D + KF_NAMES_2D

MIN_HIST = max(LWR_LAGS[0] + LWR_XI + 10, 800)

for ticker in TICKERS:
    dr, ratio, trend, att_c, att_o, ohlc = load_data(ticker)
    all_dratio[ticker]   = dr
    all_ratio[ticker]    = ratio
    all_logtrend[ticker] = trend
    all_att_c[ticker]    = att_c
    all_att_o[ticker]    = att_o
    all_ohlc[ticker]     = ohlc

    kf_dict = {}
    for qf in KF_Q_FACTORS:
        att_1d, K_ss, gd = kalman_1d_full(dr, qf)
        pname = f"KF1D_q{qf}"
        kf_dict[pname] = att_1d
        if ticker == TICKERS[0]:
            kf_info[pname] = f"K_ss={K_ss:.3f}  GD≈{gd:.1f}б"

        att_2d, k0, k1 = kalman_2d_full(dr, qf)
        pname = f"KF2D_q{qf}"
        kf_dict[pname] = att_2d
        if ticker == TICKERS[0]:
            kf_info[pname] = f"K=[{k0:.3f},{k1:.3f}]"

    all_att_kf[ticker] = kf_dict

    n     = len(dr)
    end_k = n - HORIZON - 1
    start_k = max(MIN_HIST + TAU, end_k - N_ORIG * 5)
    cands = list(range(start_k, end_k))
    step  = max(1, len(cands) // N_ORIG)
    all_origins[ticker] = cands[::step][:N_ORIG]
    print(f"  {ticker}: {len(all_origins[ticker])} origins, n={n}", flush=True)

# вывод параметров Kalman
print(f"\n  LP τ = {TAU} баров")
for pname in KF_NAMES_1D + KF_NAMES_2D:
    print(f"  {pname:<14}  {kf_info[pname]}")


# ── прогон ────────────────────────────────────────────────────────────────────

# results[pipe][ticker] = list of metric dicts
ratio_res  = {p: {t: [] for t in TICKERS} for p in ALL_PIPES}
price_res  = {p: {t: [] for t in TICKERS} for p in ALL_PIPES}

# tau_shift: LP-пайплайны используют TAU, Kalman — 0
PIPE_TAU = {p: (0 if p.startswith("KF") else TAU) for p in ALL_PIPES}

t0     = time.time()
n_done = 0
total  = sum(len(all_origins[t]) for t in TICKERS)
print(f"\nПрогон: {len(ALL_PIPES)} пайплайнов × {total} origins…", flush=True)

for ticker in TICKERS:
    dratio   = all_dratio[ticker]
    ratio    = all_ratio[ticker]
    logtrend = all_logtrend[ticker]
    att_c_f  = all_att_c[ticker]
    att_o_f  = all_att_o[ticker]
    ohlc     = all_ohlc[ticker]
    close    = ohlc["close"]
    kf_dict  = all_att_kf[ticker]

    for origin_k in all_origins[ticker]:
        dr_hist = dratio[:origin_k]
        ratio0  = float(ratio[origin_k])
        ratio_raw_future = ratio[origin_k + 1: origin_k + 1 + HORIZON]

        # att истории по пайплайнам
        att_hists = {
            "A_lp": att_c_f[:origin_k],
            "B_lp": build_att_ar_filtfilt(dr_hist),
            "D_lp": build_att_tau_oracle(dratio, origin_k),
        }
        for kname in KF_NAMES_1D + KF_NAMES_2D:
            att_hists[kname] = kf_dict[kname][:origin_k]

        for pipe in ALL_PIPES:
            att_h = att_hists[pipe]
            pred  = forecast_lwr(att_h, HORIZON)
            rm    = compute_ratio_metrics(pred, ratio_raw_future, ratio0)
            rj    = compute_ratio_j(att_h, ratio, logtrend, close, origin_k,
                                    PIPE_TAU[pipe])
            pm    = compute_price_mape(pred, origin_k, rj, logtrend, close)
            ratio_res[pipe][ticker].append(rm)
            price_res[pipe][ticker].append(pm)

        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{total}  ({time.time()-t0:.0f}с)", flush=True)

print(f"Завершено за {time.time()-t0:.1f}с", flush=True)


# ── агрегация ──────────────────────────────────────────────────────────────────

def agg_r(pipe, metric):
    vals = [m[metric] for t in TICKERS for m in ratio_res[pipe][t] if np.isfinite(m[metric])]
    return float(np.nanmean(vals)) if vals else np.nan

def agg_p(pipe):
    vals = [v for t in TICKERS for v in price_res[pipe][t] if np.isfinite(v)]
    return float(np.nanmean(vals)) if vals else np.nan

def flat_p(pipe):
    return [v for t in TICKERS for v in price_res[pipe][t] if np.isfinite(v)]


# ── вывод ─────────────────────────────────────────────────────────────────────

PIPE_LABELS = {
    "A_lp": "A LP causal    (τ=7б)",
    "B_lp": "B LP ar-filt   (краевая коррекция)",
    "D_lp": "D LP τ-oracle  (утечка τ баров)",
}
for qf in KF_Q_FACTORS:
    PIPE_LABELS[f"KF1D_q{qf}"] = f"KF1D q={qf}  ({kf_info.get(f'KF1D_q{qf}','')})"
    PIPE_LABELS[f"KF2D_q{qf}"] = f"KF2D q={qf}  ({kf_info.get(f'KF2D_q{qf}','')})"

base_pm = agg_p("A_lp")

print(f"\n── Price-MAPE close (LWR c2c5_k13 xi=700) ───────────────────────────────────────")
print(f"  {'Пайплайн':<50}  {'price-MAPE':>10}  {'Δ vs A':>8}  {'p-val':>7}")
print(f"  {'─'*50}  {'─'*10}  {'─'*8}  {'─'*7}")
fa = flat_p("A_lp")
for pipe in ALL_PIPES:
    pm  = agg_p(pipe)
    d   = (base_pm / pm - 1) * 100 if pipe != "A_lp" else 0.0
    fb  = flat_p(pipe)
    n_  = min(len(fa), len(fb))
    if n_ > 2 and pipe != "A_lp":
        _, pv = ttest_rel(fa[:n_], fb[:n_])
        pv_s = f"{pv:.4f}"
    else:
        pv_s = "—"
    star = " ←" if pipe == "D_lp" else ""
    print(f"  {PIPE_LABELS[pipe]:<50}  {pm:>10.5f}  {d:>+7.1f}%  {pv_s:>7}{star}")

print(f"\n── MAPE ratio (trim: пропуск τ={TAU}б) ─────────────────────────────────────────────")
print(f"  {'Пайплайн':<50}  {'mape_full':>10}  {'mape_trim':>10}  {'dir_acc':>8}")
print(f"  {'─'*50}  {'─'*10}  {'─'*10}  {'─'*8}")
for pipe in ALL_PIPES:
    mf = agg_r(pipe, "mape_full")
    mt = agg_r(pipe, "mape_trim")
    da = agg_r(pipe, "dir_acc")
    print(f"  {PIPE_LABELS[pipe]:<50}  {mf:>10.5f}  {mt:>10.5f}  {da:>7.1%}")

print(f"\n── Per-ticker price-MAPE (лучший KF2D vs B_lp) ─────────────────────────────────")
best_kf2d = min(KF_NAMES_2D, key=agg_p)
best_kf1d = min(KF_NAMES_1D, key=agg_p)
print(f"  {'Ticker':<6}  {'A_lp':>8}  {'B_lp':>8}  {'D_lp':>8}  {best_kf1d:>12}  {best_kf2d:>12}")
print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*12}")
for ticker in TICKERS:
    a  = np.nanmean([v for v in price_res["A_lp"][ticker] if np.isfinite(v)])
    b  = np.nanmean([v for v in price_res["B_lp"][ticker] if np.isfinite(v)])
    d  = np.nanmean([v for v in price_res["D_lp"][ticker] if np.isfinite(v)])
    k1 = np.nanmean([v for v in price_res[best_kf1d][ticker] if np.isfinite(v)])
    k2 = np.nanmean([v for v in price_res[best_kf2d][ticker] if np.isfinite(v)])
    print(f"  {ticker:<6}  {a:>8.5f}  {b:>8.5f}  {d:>8.5f}  {k1:>12.5f}  {k2:>12.5f}")

# t-test лучший KF2D vs B_lp
fb_b  = flat_p("B_lp")
fb_k2 = flat_p(best_kf2d)
fb_k1 = flat_p(best_kf1d)
n_ = min(len(fb_b), len(fb_k2))
if n_ > 2:
    _, pv2 = ttest_rel(fb_b[:n_], fb_k2[:n_])
    d2 = (np.mean(fb_b[:n_]) / np.mean(fb_k2[:n_]) - 1) * 100
    print(f"\nt-test лучший KF2D ({best_kf2d}) vs B_lp:  Δ={d2:+.2f}%  p={pv2:.4f}  {'✓' if pv2<0.05 else '✗'}")
n_ = min(len(fb_b), len(fb_k1))
if n_ > 2:
    _, pv1 = ttest_rel(fb_b[:n_], fb_k1[:n_])
    d1 = (np.mean(fb_b[:n_]) / np.mean(fb_k1[:n_]) - 1) * 100
    print(f"t-test лучший KF1D ({best_kf1d}) vs B_lp:  Δ={d1:+.2f}%  p={pv1:.4f}  {'✓' if pv1<0.05 else '✗'}")


# ── рисунок ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Price-MAPE по всем пайплайнам
ax = axes[0]
pipes_plot = ALL_PIPES
pm_vals = [agg_p(p) for p in pipes_plot]
labels  = [PIPE_LABELS[p][:35] for p in pipes_plot]
colors  = []
for p in pipes_plot:
    if p == "A_lp":   colors.append("#B71C1C")
    elif p == "B_lp": colors.append("#1565C0")
    elif p == "D_lp": colors.append("#43A047")
    elif "1D" in p:   colors.append("#E65100")
    else:              colors.append("#6A1B9A")

bars = ax.barh(range(len(pipes_plot)), pm_vals, color=colors, alpha=0.82, edgecolor="white")
ax.set_yticks(range(len(pipes_plot))); ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel("price-MAPE close (меньше = лучше)")
ax.set_title("Price-MAPE: Kalman vs LP\n(LWR c2c5_k13 xi=700)")
ax.grid(True, alpha=0.3, axis="x"); ax.invert_yaxis()

# Вертикальные линии для A и B
ax.axvline(agg_p("A_lp"), color="#B71C1C", lw=1.5, ls="--", alpha=0.6, label="A_lp")
ax.axvline(agg_p("B_lp"), color="#1565C0", lw=1.5, ls="--", alpha=0.6, label="B_lp")
ax.legend(fontsize=7)

# 2. KF1D: q_factor sweep (price-MAPE)
ax = axes[1]
q_vals = KF_Q_FACTORS
pm_1d = [agg_p(f"KF1D_q{qf}") for qf in q_vals]
pm_2d = [agg_p(f"KF2D_q{qf}") for qf in q_vals]
x = np.arange(len(q_vals))
w = 0.35
ax.bar(x - w/2, pm_1d, w, label="KF 1D", color="#E65100", alpha=0.82, edgecolor="white")
ax.bar(x + w/2, pm_2d, w, label="KF 2D", color="#6A1B9A", alpha=0.82, edgecolor="white")
ax.axhline(agg_p("A_lp"), color="#B71C1C", lw=1.5, ls="--", alpha=0.7, label="A LP causal")
ax.axhline(agg_p("B_lp"), color="#1565C0", lw=1.5, ls="--", alpha=0.7, label="B LP ar-filt")
ax.axhline(agg_p("D_lp"), color="#43A047", lw=1.5, ls=":",  alpha=0.7, label="D LP oracle")
ax.set_xticks(x); ax.set_xticklabels([f"q={qf}" for qf in q_vals])
ax.set_ylabel("price-MAPE close"); ax.set_title("KF q_factor sweep")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")

# 3. mape_trim по пайплайнам
ax = axes[2]
mt_vals = [agg_r(p, "mape_trim") for p in pipes_plot]
ax.barh(range(len(pipes_plot)), mt_vals, color=colors, alpha=0.82, edgecolor="white")
ax.set_yticks(range(len(pipes_plot))); ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel(f"ratio MAPE_trim (пропуск {TAU}б)")
ax.set_title(f"MAPE_trim [{TAU}:{HORIZON}]")
ax.axvline(agg_r("A_lp", "mape_trim"), color="#B71C1C", lw=1.5, ls="--", alpha=0.6)
ax.axvline(agg_r("B_lp", "mape_trim"), color="#1565C0", lw=1.5, ls="--", alpha=0.6)
ax.grid(True, alpha=0.3, axis="x"); ax.invert_yaxis()

fig.suptitle(
    f"Kalman LP vs Butterworth LP  (τ={TAU}б, {len(TICKERS)} тикера, {N_ORIG} origins)",
    fontsize=11
)
fig.tight_layout()
fig.savefig(FIG_DIR / "71_kalman_vs_lp.png", dpi=150)
plt.close(fig)

print(f"\nРис.: {FIG_DIR}/71_kalman_vs_lp.png")
print(f"\nСкрипт 71 завершён за {time.time()-t0:.1f}с")
