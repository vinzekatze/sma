"""
72 — Визуализация: восстановленные ценовые ряды LP vs Kalman.

Для каждого тикера (4 шт.) отображаем последние WIN_BARS баров:
  Верхняя панель: OHLC close + LP overlay + KF overlay
  Нижняя панель:  att-сигналы (LP causal, LP B, KF1D, KF2D)

Overlays строятся скользящим окном LP_WIN = 50 баров с bias-коррекцией:
  LP: смещение τ баров назад по времени + вертикальный bias
  KF: без смещения + вертикальный bias
"""

import sys, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import butter, sosfilt, sosfiltfilt, group_delay, lfilter

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/candles"
FIG_DIR  = ROOT / "research/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

# ── параметры ──────────────────────────────────────────────────────────────────

TICKERS   = ["SBER", "LKOH", "CHMF", "MRKP"]
WIN_BARS  = 300   # сколько баров отображать
LP_WIN    = 50    # окно bias-коррекции
WN        = 0.125
FILTER_ORDER = 4
AR_PAD    = 40
AR_ORDER  = 20
KF_Q      = 0.4   # лучший по скр.71

_SOS_LP = butter(FILTER_ORDER, WN, btype="low", output="sos")


def compute_tau(sos):
    from scipy.signal import sos2tf
    b, a = sos2tf(sos)
    _, gd = group_delay((b, a), w=1, whole=False)
    return int(round(float(gd[0])))

TAU = compute_tau(_SOS_LP)
print(f"TAU = {TAU}")

# ── утилиты ────────────────────────────────────────────────────────────────────

def logtrend_causal(close):
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64); cn = np.arange(1, n+1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t**2)
    cy = np.cumsum(lc); cty = np.cumsum(t*lc)
    denom = cn*ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn*cty-ct*cy)/denom, 0.0)
    a = (cy - b*ct)/cn
    trend = np.exp(a + b*t); trend[:2] = close[:2]
    return trend


def ar_extend_forward(x, order, n_extend):
    if len(x) < order + 1:
        return np.concatenate([x, np.zeros(n_extend)])
    n = len(x); rows = min(n - order, 500); start = n - order - rows
    X = np.column_stack([x[start+i:start+i+rows] for i in range(order)])
    y = x[start+order:start+order+rows]
    a, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    buf = list(x[-order:])
    ext = []
    for _ in range(n_extend):
        nxt = float(np.dot(a, buf[-order:][::-1]))
        ext.append(nxt); buf.append(nxt)
    return np.concatenate([x, ext])


# ── фильтры ────────────────────────────────────────────────────────────────────

def att_lp_causal(dratio):
    return sosfilt(_SOS_LP, dratio)

def att_lp_b(dratio):
    ext = ar_extend_forward(dratio, AR_ORDER, AR_PAD)
    return sosfiltfilt(_SOS_LP, ext)[:len(dratio)]

def att_lp_oracle(dratio):
    return sosfiltfilt(_SOS_LP, dratio)

def att_kf1d(dratio, q_factor):
    sigma = float(np.std(dratio))
    if sigma < 1e-12: return np.zeros(len(dratio))
    Q = (q_factor*sigma)**2; R = sigma**2
    P_ss = (-Q + np.sqrt(Q**2 + 4*Q*R)) / 2
    K_ss = (P_ss + Q) / (P_ss + Q + R)
    return lfilter([K_ss], [1.0, -(1.0-K_ss)], dratio)

def att_kf2d(dratio, q_factor):
    sigma = float(np.std(dratio))
    if sigma < 1e-12: return np.zeros(len(dratio))
    q_slope = q_factor * sigma
    F = np.array([[1.,1.],[0.,1.]]); H = np.array([1.,0.])
    Q_mat = np.array([[0.,0.],[0.,q_slope**2]]); R_var = sigma**2
    try:
        from scipy.linalg import solve_discrete_are
        P_ss = solve_discrete_are(F.T, H.reshape(-1,1), Q_mat, np.array([[R_var]]))
        S_ss = float(H @ P_ss @ H) + R_var
        K_ss = (P_ss @ H) / S_ss
    except Exception:
        P = np.eye(2)*R_var
        for _ in range(500):
            P_p = F@P@F.T + Q_mat; S = float(H@P_p@H)+R_var; K_ss = (P_p@H)/S
            P = (np.eye(2)-np.outer(K_ss,H))@P_p
    A_ss = (np.eye(2)-np.outer(K_ss,H))@F
    n = len(dratio); x = np.zeros(2); att = np.zeros(n)
    for k in range(n):
        x = A_ss@x + K_ss*dratio[k]; att[k] = x[0]
    return att


# ── price overlay builder ─────────────────────────────────────────────────────

def build_price_overlay(att_full, ratio, logtrend, close, i_start, i_end, tau_shift):
    """
    Строим price overlay поверх att_full[i_start:i_end].
    tau_shift: TAU для LP (визуальный сдвиг влево по оси X), 0 для KF.

    Возвращает: (x_indices, prices) — выравненные на close.
    """
    ws = i_start
    # reconstruct LP price в окне [ws, i_end]
    lp_r = ratio[ws] + np.concatenate([[0.], np.cumsum(att_full[ws:i_end])])
    lp_p = lp_r * logtrend[ws:i_end+1]          # shape i_end-ws+1

    # x-индексы (bar numbers), сдвинутые влево на tau_shift
    x_bars = np.arange(ws, i_end+1) - tau_shift  # shape i_end-ws+1

    # вычислить bias: сравниваем lp_p с close только там, где x_bars в [0, len)
    valid = (x_bars >= 0) & (x_bars < len(close))
    if valid.any():
        bias = float(np.mean(close[x_bars[valid]] - lp_p[valid]))
    else:
        bias = 0.

    return x_bars, lp_p + bias


# ── загрузка данных ────────────────────────────────────────────────────────────

print("Загрузка…")
data_all = {}
for ticker in TICKERS:
    raw   = json.loads((DATA_DIR / ticker / "1d.json").read_text())
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    high  = np.array([c["high"]  for c in raw], dtype=np.float64)
    low   = np.array([c["low"]   for c in raw], dtype=np.float64)
    dates = [c["begin"][:10] for c in raw]
    trend = logtrend_causal(close)
    ratio = close / np.maximum(trend, 1e-10)
    dr    = np.diff(ratio, prepend=ratio[0])

    att_c = att_lp_causal(dr)
    att_b = att_lp_b(dr)
    att_o = att_lp_oracle(dr)
    att_1 = att_kf1d(dr, KF_Q)
    att_2 = att_kf2d(dr, KF_Q)

    data_all[ticker] = dict(
        close=close, high=high, low=low, dates=dates,
        ratio=ratio, logtrend=trend, dratio=dr,
        att_c=att_c, att_b=att_b, att_o=att_o,
        att_k1=att_1, att_k2=att_2,
        n=len(close)
    )
    print(f"  {ticker}: n={len(close)}")


# ── рисунок ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(
    nrows=len(TICKERS), ncols=2,
    figsize=(22, 5*len(TICKERS)),
    gridspec_kw={"width_ratios": [3, 2]}
)
fig.suptitle(
    f"LP vs Kalman: price overlay и att-сигналы  (последние {WIN_BARS} баров)\n"
    f"LP τ={TAU}б сдвиг влево + bias  |  KF q={KF_Q}  без сдвига + bias",
    fontsize=12
)

for row, ticker in enumerate(TICKERS):
    d     = data_all[ticker]
    n     = d["n"]
    i_end = n - 1
    i_st  = max(0, i_end - WIN_BARS + 1)

    close = d["close"]; high = d["high"]; low = d["low"]
    dates = d["dates"]
    ratio = d["ratio"]; lt = d["logtrend"]

    # X-tick labels: каждые ~50 баров
    x_all  = np.arange(i_st, i_end+1)
    tick_x = x_all[::50]
    tick_l = [dates[i][:7] for i in tick_x]  # YYYY-MM

    # ── левая панель: цены + overlays ─────────────────────────────────────────
    ax = axes[row, 0]

    # OHLC bars (simple: gray for down, white for up candles)
    for i in range(i_st, i_end+1):
        col = "#c0392b" if close[i] < d["close"][i-1] else "#27ae60"
        ax.plot([i, i], [low[i], high[i]], color=col, lw=0.5, alpha=0.5)
    ax.plot(x_all, close[i_st:i_end+1], color="#555555", lw=0.8, label="close", zorder=3)

    # LP causal overlay (τ-сдвиг влево, только видимые бары)
    x_lpc, p_lpc = build_price_overlay(
        d["att_c"], ratio, lt, close, i_st, i_end, TAU)
    m_lpc = (x_lpc >= i_st) & (x_lpc <= i_end)
    ax.plot(x_lpc[m_lpc], p_lpc[m_lpc],
            color="#E53935", lw=1.3, alpha=0.85, label=f"LP causal (τ={TAU}б сдвиг)")

    # LP B overlay (AR-filtfilt)
    x_lpb, p_lpb = build_price_overlay(
        d["att_b"], ratio, lt, close, i_st, i_end, TAU)
    m_lpb = (x_lpb >= i_st) & (x_lpb <= i_end)
    ax.plot(x_lpb[m_lpb], p_lpb[m_lpb],
            color="#1565C0", lw=1.3, alpha=0.85, ls="--", label="LP B (ar-filtfilt)")

    # KF2D overlay (нет сдвига)
    x_kf2, p_kf2 = build_price_overlay(
        d["att_k2"], ratio, lt, close, i_st, i_end, 0)
    m_kf2 = (x_kf2 >= i_st) & (x_kf2 <= i_end)
    ax.plot(x_kf2[m_kf2], p_kf2[m_kf2],
            color="#7B1FA2", lw=1.6, alpha=0.9, label=f"KF2D q={KF_Q}")

    # LP oracle для ориентира
    x_lpo, p_lpo = build_price_overlay(
        d["att_o"], ratio, lt, close, i_st, i_end, 0)
    m_lpo = (x_lpo >= i_st) & (x_lpo <= i_end)
    ax.plot(x_lpo[m_lpo], p_lpo[m_lpo],
            color="#43A047", lw=1.0, alpha=0.5, ls=":", label="LP oracle (filtfilt)")

    ax.set_xlim(i_st, i_end)
    ax.set_xticks(tick_x); ax.set_xticklabels(tick_l, rotation=30, ha="right", fontsize=7)
    ax.set_title(f"{ticker}  — цены + overlays", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)

    # ── правая панель: att-сигналы ────────────────────────────────────────────
    ax2 = axes[row, 1]
    x_a = np.arange(i_st, i_end+1)

    # LP causal и oracle на одном масштабе
    ax2.plot(x_a, d["att_c"][i_st:i_end+1],
             color="#E53935", lw=0.8, alpha=0.7, label="LP causal")
    ax2.plot(x_a, d["att_b"][i_st:i_end+1],
             color="#1565C0", lw=1.0, alpha=0.8, ls="--", label="LP B")
    ax2.plot(x_a, d["att_o"][i_st:i_end+1],
             color="#43A047", lw=0.9, alpha=0.6, ls=":", label="LP oracle")
    ax2.plot(x_a, d["att_k1"][i_st:i_end+1],
             color="#E65100", lw=0.9, alpha=0.7, label=f"KF1D q={KF_Q}")
    ax2.plot(x_a, d["att_k2"][i_st:i_end+1],
             color="#7B1FA2", lw=1.3, alpha=0.9, label=f"KF2D q={KF_Q}")

    ax2.axhline(0, color="black", lw=0.5, alpha=0.3)
    ax2.set_xlim(i_st, i_end)
    ax2.set_xticks(tick_x); ax2.set_xticklabels(tick_l, rotation=30, ha="right", fontsize=7)
    ax2.set_title(f"{ticker}  — att-сигналы (dratio slow component)", fontsize=10)
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.2)

fig.tight_layout()
out_path = FIG_DIR / "72_kalman_price_viz.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nРис.: {out_path}")
