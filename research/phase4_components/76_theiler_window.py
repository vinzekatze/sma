"""
76 — Theiler window: исключение временны́х соседей при поиске в фазовом пространстве.

Стандарт хаотического анализа (Theiler 1986): соседние по времени точки исключаются
из пула кандидатов — они близки к запросу из-за непрерывности ряда (автокорреляция),
а не из-за структуры аттрактора.

Att-фильтр: Local Projective (m=9, d=3, k=30, n=3) — новый стандарт.
Применяется офлайн к ratio (одно вычисление на тикер), затем diff → att.
Не даёт фазового сдвига, раскрывает аттрактор.

Реализация Theiler: перед каскадом применяем маску
  temporal_dist[i] = (n-1) - t_arr[i]  >= W_theiler
Expansion каскада движется только назад по времени → Theiler не нарушается
на следующих уровнях.

Протокол walk-forward:
  8 тикеров × 1d.  h=1.
  Тест: последние 200 баров att, шаг 5 → ~40 origins/тикер.
  p_fit=16, p_max=64, xi=3*(p_fit+1)=51, октавный каскад (×2).
  W ∈ {0, 1, 3, 5, 10, 20, 50}.

Метрики:
  rMAE    — MAE / std(att)  (нормализация по тикеру)
  SignAcc — точность знака, %

Графики:
  A — ACF att (SBER) — естественный диапазон W
  B — rMAE по W (все тикеры + среднее)
  C — SignAcc по W (все тикеры + среднее)
  D — Гистограмма temporal_dist соседей: W=0 vs W=10 (SBER, 1 origin)
  E — Фазовый портрет с подсветкой соседей: W=0 / W=10 / исключённые
"""

import json, sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, correlate
from scipy.spatial import KDTree

ROOT    = Path(__file__).resolve().parent.parent.parent
FIGDIR  = ROOT / "research" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
DATADIR = ROOT / "data" / "candles"
sys.path.insert(0, str(ROOT))

TICKERS  = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
INTERVAL = "1d"
SOS_LP   = butter(8, 0.125, btype="low", output="sos")   # для сравнения в ACF

# Local Projective — новый стандарт att-фильтра
LP_M = 9; LP_D = 3; LP_K = 30; LP_N = 3

# LWR / каскад
P_FIT     = 16
P_MAX     = 64
XI_LWR    = 3 * (P_FIT + 1)   # = 51
STEP_BASE = 2.0

# Walk-forward
N_TEST  = 200
STEP_WF = 5
W_VALUES = [0, 1, 3, 5, 10, 20, 50]

COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"]

# ── helpers ────────────────────────────────────────────────────────────────────

def logtrend_causal(close: np.ndarray) -> np.ndarray:
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64); cn = np.arange(1, n+1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t**2); cy = np.cumsum(lc); cty = np.cumsum(t*lc)
    denom = cn*ct2 - ct**2
    b = np.where(denom > 0, (cn*cty - ct*cy)/denom, 0.0)
    a = (cy - b*ct)/cn
    trend = np.exp(a + b*t); trend[:2] = close[:2]
    return trend


def local_projective(series: np.ndarray, m: int, d: int, k: int, n_iter: int = 1
                     ) -> np.ndarray:
    """Local Projective noise reduction (Grassberger-Hegger 1993).

    Применяется к ratio. Возвращает diff(очищенного_ratio) = att.
    """
    s = series.copy().astype(np.float64)
    N = len(s)
    k_eff = min(k, N - m)
    d_eff = min(d, m - 1)

    for _ in range(n_iter):
        n_pts = N - m + 1
        rows  = np.arange(n_pts)[:, None] + np.arange(m)[None, :]
        X     = s[rows]
        tree  = KDTree(X)
        _, inds = tree.query(X, k=k_eff + 1)
        X_proj = np.empty_like(X)
        for i in range(n_pts):
            nn       = inds[i, 1:]
            X_nn_    = X[nn]
            centroid = X_nn_.mean(axis=0)
            _, _, Vt = np.linalg.svd(X_nn_ - centroid, full_matrices=False)
            V_d      = Vt[:d_eff].T
            xc       = X[i] - centroid
            X_proj[i] = centroid + V_d @ (V_d.T @ xc)
        result = np.zeros(N); count = np.zeros(N, dtype=np.int32)
        for i in range(n_pts):
            result[i:i+m] += X_proj[i]; count[i:i+m] += 1
        s = result / np.maximum(count, 1)
    return np.diff(s)


def load_att_lp(ticker: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Загружает данные, применяет Local Projective к ratio → att.

    Возвращает (att_lp, att_butter, ratio).
    att_lp     — Local Projective (новый стандарт)
    att_butter — LP Butterworth (для сравнения ACF)
    """
    raw   = json.loads((DATADIR / ticker / f"{INTERVAL}.json").read_text())
    cands = raw["candles"] if isinstance(raw, dict) else raw
    close = np.array([c["close"] for c in cands], dtype=np.float64)
    trend = logtrend_causal(close)
    ratio = close / trend
    att_lp     = local_projective(ratio, LP_M, LP_D, LP_K, LP_N)
    att_butter = sosfilt(SOS_LP, np.diff(ratio))
    return att_lp, att_butter, ratio


def octave_levels(p_fit: int, p_max: int, step: float = 2.0) -> list[int]:
    levels: list[int] = []; p = p_max
    while True:
        levels.append(p)
        p_next = int(p / step)
        if p_next < max(p_fit, 2): break
        p = p_next
    if levels[-1] != p_fit: levels.append(p_fit)
    return levels


def lwr_step(att_hist: np.ndarray, W_theiler: int = 0
             ) -> tuple[float, np.ndarray, np.ndarray]:
    """h=1 LWR + каскад + Theiler window.

    Возвращает: (prediction, cands_idx, temporal_dists_of_cands).
    """
    n = len(att_hist)
    m = n - P_MAX - 1
    if m < XI_LWR + 5:
        return np.nan, np.array([], dtype=int), np.array([])

    t_arr  = np.arange(P_MAX, n - 1)
    X_full = np.column_stack([att_hist[t_arr - (P_MAX-1-j)] for j in range(P_MAX)])
    y_base = att_hist[t_arr + 1]

    # Temporal distance: насколько давно каждая строка библиотеки
    temporal_dist = (n - 1) - t_arr    # >= 1 всегда; для последней строки = 1

    if W_theiler > 0:
        cands = np.where(temporal_dist >= W_theiler)[0]
    else:
        cands = np.arange(m)

    if len(cands) < XI_LWR:
        return np.nan, cands, temporal_dist[cands]

    levels   = octave_levels(P_FIT, P_MAX, STEP_BASE)
    vec_full = att_hist[-P_MAX:]

    for k, p_lvl in enumerate(levels):
        xi_lvl = min(XI_LWR, len(cands))
        if len(cands) > xi_lvl:
            dists = np.linalg.norm(X_full[cands, -p_lvl:] - vec_full[-p_lvl:], axis=1)
            cands = cands[np.argpartition(dists, xi_lvl-1)[:xi_lvl]]
        if k < len(levels) - 1:
            p_next  = levels[k+1]
            radius  = p_lvl - p_next
            offsets = np.arange(radius + 1)
            exp     = cands[:, None] - offsets[None, :]
            exp     = np.clip(exp, 0, m-1)
            cands   = np.unique(exp)

    if len(cands) < P_FIT + 2:
        return np.nan, cands, temporal_dist[cands]

    X_nn  = X_full[cands, -P_FIT:]
    y_nn  = y_base[cands]
    vec_f = vec_full[-P_FIT:]
    h_bw  = max(float(np.linalg.norm(X_nn - vec_f, axis=1).max()), 1e-10)

    dists_f = np.linalg.norm(X_nn - vec_f, axis=1)
    w       = np.exp(-0.5 * (dists_f / h_bw) ** 2)
    if w.sum() < 1e-15:
        return float(y_nn.mean()), cands, temporal_dist[cands]

    try:
        A    = np.column_stack([np.ones(len(X_nn)), X_nn])
        AtW  = (A * w[:, None]).T
        coef = np.linalg.lstsq(AtW @ A, AtW @ y_nn, rcond=None)[0]
        val  = float(coef[0] + coef[1:] @ vec_f)
    except np.linalg.LinAlgError:
        val  = float((w @ y_nn) / w.sum())

    return val, cands, temporal_dist[cands]

# ── загрузка данных ────────────────────────────────────────────────────────────

print("Загрузка + Local Projective (это займёт ~1-3 мин)...")
atts_lp     = {}
atts_butter = {}
t_load = time.time()
for tk in TICKERS:
    print(f"  {tk}...", end=" ", flush=True)
    t1 = time.time()
    atts_lp[tk], atts_butter[tk], _ = load_att_lp(tk)
    print(f"{time.time()-t1:.1f}с  (att={len(atts_lp[tk])} баров)")
print(f"Загрузка итого: {time.time()-t_load:.1f}с")

# ── walk-forward ───────────────────────────────────────────────────────────────

results = {W: {tk: {"mae": [], "sign": []} for tk in TICKERS} for W in W_VALUES}

t0 = time.time()
for tk in TICKERS:
    att = atts_lp[tk]
    n_att = len(att)
    origins = list(range(n_att - N_TEST, n_att - 1, STEP_WF))
    print(f"\n{tk}: {len(origins)} origins...", flush=True)

    for t_orig in origins:
        if t_orig + 1 >= n_att:
            continue
        actual = att[t_orig + 1]
        for W in W_VALUES:
            pred, _, _ = lwr_step(att[:t_orig+1], W)
            if np.isnan(pred):
                continue
            results[W][tk]["mae"].append(abs(pred - actual))
            results[W][tk]["sign"].append(1 if np.sign(pred) == np.sign(actual) else 0)

print(f"\nWalk-forward завершён за {time.time()-t0:.1f}с")

# ── агрегация ──────────────────────────────────────────────────────────────────

rmae   = {W: {} for W in W_VALUES}
sigacc = {W: {} for W in W_VALUES}

for W in W_VALUES:
    for tk in TICKERS:
        std_att = float(np.std(atts_lp[tk]))
        m_vals  = results[W][tk]["mae"]
        s_vals  = results[W][tk]["sign"]
        rmae[W][tk]   = float(np.mean(m_vals)) / std_att if m_vals else np.nan
        sigacc[W][tk] = float(np.mean(s_vals)) * 100     if s_vals else np.nan

print("\nrMAE / SignAcc по W:")
base_rmae = np.nanmean([rmae[0][tk] for tk in TICKERS])
for W in W_VALUES:
    m_r   = np.nanmean([rmae[W][tk] for tk in TICKERS])
    m_s   = np.nanmean([sigacc[W][tk] for tk in TICKERS])
    delta = (m_r - base_rmae) / base_rmae * 100
    sign  = "+" if delta >= 0 else ""
    print(f"  W={W:3d}: rMAE={m_r:.4f} ({sign}{delta:.2f}%)  SignAcc={m_s:.1f}%")

# ── визуализация ───────────────────────────────────────────────────────────────

att_sber   = atts_lp["SBER"]
att_sber_b = atts_butter["SBER"]
n_sber     = len(att_sber)
t_demo     = n_sber - N_TEST // 2

_, cands_w0,   tdist_w0   = lwr_step(att_sber[:t_demo+1], W_theiler=0)
_, cands_wopt, tdist_wopt = lwr_step(att_sber[:t_demo+1], W_theiler=10)

# ── A: ACF ────────────────────────────────────────────────────────────────────

def autocorr(x, max_lag=100):
    x = x - x.mean()
    c = correlate(x, x, mode="full"); c = c[len(c)//2:]; c /= c[0]
    return c[:max_lag+1]

MAX_LAG = 100
lags_acf = np.arange(MAX_LAG + 1)

fig_a, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig_a.suptitle("76-A: ACF att (SBER 1d) — LP Butterworth vs Local Projective", fontsize=11)

axes[0].bar(lags_acf, autocorr(att_sber_b), color="steelblue", width=0.8, alpha=0.7)
axes[0].set_title("LP Butterworth (Wn=0.125)")
axes[0].axhline(0, color="k", lw=0.5)
axes[0].axhline(1.96/np.sqrt(n_sber), color="red", lw=0.8, ls="--")
axes[0].axhline(-1.96/np.sqrt(n_sber), color="red", lw=0.8, ls="--")

axes[1].bar(lags_acf, autocorr(att_sber), color="seagreen", width=0.8, alpha=0.7)
axes[1].set_title("Local Projective (m=9 d=3 k=30 n=3)")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].axhline(1.96/np.sqrt(n_sber), color="red", lw=0.8, ls="--", label="95% CI")
axes[1].axhline(-1.96/np.sqrt(n_sber), color="red", lw=0.8, ls="--")
for W_mark in [5, 10, 20]:
    axes[1].axvline(W_mark, color="orange", lw=0.9, ls=":", alpha=0.9)
    axes[1].text(W_mark+0.5, axes[1].get_ylim()[0]*0.85 if axes[1].get_ylim()[0] < 0 else 0.05,
                 f"W={W_mark}", fontsize=7, color="orange")
axes[1].legend(fontsize=8); axes[1].set_xlabel("лаг")
for ax in axes: ax.set_ylabel("ACF")
fig_a.tight_layout()
fig_a.savefig(FIGDIR / "76_A_acf.png", dpi=130); plt.close(fig_a)
print("Рис. A сохранён")

# ── B: rMAE по W ─────────────────────────────────────────────────────────────

fig_b, ax = plt.subplots(figsize=(10, 5))
ax.set_title("76-B: rMAE = MAE/std(att) по Theiler W  [Local Projective att]", fontsize=11)
for i, tk in enumerate(TICKERS):
    vals = [rmae[W][tk] for W in W_VALUES]
    ax.plot(W_VALUES, vals, color=COLORS[i%len(COLORS)], lw=1.0,
            marker="o", ms=4, alpha=0.6, label=tk)
mean_r = [np.nanmean([rmae[W][tk] for tk in TICKERS]) for W in W_VALUES]
ax.plot(W_VALUES, mean_r, color="black", lw=2.2, marker="D", ms=6, label="СРЕДНЕЕ")
for W, v in zip(W_VALUES, mean_r):
    ax.annotate(f"{v:.3f}", (W, v), textcoords="offset points",
                xytext=(0, 8), fontsize=7, ha="center")
ax.set_xlabel("W (Theiler window)"); ax.set_ylabel("rMAE")
ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
fig_b.tight_layout()
fig_b.savefig(FIGDIR / "76_B_rmae.png", dpi=130); plt.close(fig_b)
print("Рис. B сохранён")

# ── C: SignAcc по W ──────────────────────────────────────────────────────────

fig_c, ax = plt.subplots(figsize=(10, 5))
ax.set_title("76-C: Точность знака (SignAcc) по Theiler W  [Local Projective att]", fontsize=11)
for i, tk in enumerate(TICKERS):
    vals = [sigacc[W][tk] for W in W_VALUES]
    ax.plot(W_VALUES, vals, color=COLORS[i%len(COLORS)], lw=1.0,
            marker="o", ms=4, alpha=0.6, label=tk)
mean_s = [np.nanmean([sigacc[W][tk] for tk in TICKERS]) for W in W_VALUES]
ax.plot(W_VALUES, mean_s, color="black", lw=2.2, marker="D", ms=6, label="СРЕДНЕЕ")
ax.axhline(50, color="grey", ls="--", lw=0.8, label="случайный угадыватель")
for W, v in zip(W_VALUES, mean_s):
    ax.annotate(f"{v:.1f}%", (W, v), textcoords="offset points",
                xytext=(0, 8), fontsize=7, ha="center")
ax.set_xlabel("W (Theiler window)"); ax.set_ylabel("SignAcc, %")
ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3)
fig_c.tight_layout()
fig_c.savefig(FIGDIR / "76_C_signacc.png", dpi=130); plt.close(fig_c)
print("Рис. C сохранён")

# ── D: Гистограммы temporal_dist ─────────────────────────────────────────────

fig_d, axes = plt.subplots(1, 2, figsize=(12, 5))
fig_d.suptitle(f"76-D: Temporal distance соседей — SBER, origin t={t_demo}", fontsize=11)
bins = np.arange(0, min(t_demo, 400) + 15, 10)

for i, (tdist, W_lbl, color) in enumerate([
        (tdist_w0,   "W=0",  "steelblue"),
        (tdist_wopt, "W=10", "seagreen")]):
    ax = axes[i]
    ax.set_title(f"{W_lbl}  ({len(tdist)} соседей)")
    if len(tdist):
        ax.hist(tdist, bins=bins, color=color, alpha=0.8)
        ax.axvline(np.median(tdist), color="red", lw=1.5, ls="--",
                   label=f"медиана={int(np.median(tdist))} баров")
        ax.legend(fontsize=8)
    ax.set_xlabel("temporal distance (баров до запроса)")
    ax.set_ylabel("кол-во соседей")

fig_d.tight_layout()
fig_d.savefig(FIGDIR / "76_D_temporal_hist.png", dpi=130); plt.close(fig_d)
print("Рис. D сохранён")

# ── E: Фазовый портрет с подсветкой ──────────────────────────────────────────

fig_e, ax = plt.subplots(figsize=(8, 8))
ax.set_title("76-E: Фазовый портрет — SBER\n"
             "серый=фон, красный=исключён(W<10), синий=W=0, зелёный=W=10, ★=запрос",
             fontsize=10)

att_use = att_sber[:t_demo+1]
n_lib   = len(att_use) - P_MAX - 1
t_arr_d = np.arange(P_MAX, P_MAX + n_lib)
tdist_all = (t_demo) - t_arr_d

if n_lib > 1:
    lib_t  = att_use[P_MAX    : P_MAX + n_lib]
    lib_tm = att_use[P_MAX-1  : P_MAX + n_lib - 1]

    ax.scatter(lib_tm, lib_t, s=1.5, color="lightgray", alpha=0.35, zorder=1)

    excl = tdist_all[:n_lib] < 10
    if excl.any():
        ax.scatter(lib_tm[excl], lib_t[excl], s=20, color="salmon",
                   alpha=0.5, zorder=2, label=f"Исключено Theiler<10 ({excl.sum()})")

    if len(cands_w0):
        idx = cands_w0[cands_w0 < n_lib]
        ax.scatter(lib_tm[idx], lib_t[idx], s=45, color="steelblue",
                   alpha=0.85, zorder=3, label=f"Соседи W=0 ({len(idx)})")

    if len(cands_wopt):
        idx = cands_wopt[cands_wopt < n_lib]
        ax.scatter(lib_tm[idx], lib_t[idx], s=45, color="seagreen",
                   alpha=0.85, zorder=4, label=f"Соседи W=10 ({len(idx)})")

ax.scatter([att_use[-2]], [att_use[-1]], s=220, color="gold",
           marker="*", zorder=5, label="Запрос")
ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
ax.set_xlabel("att[t-1]"); ax.set_ylabel("att[t]")
ax.legend(fontsize=8, loc="upper left")
fig_e.tight_layout()
fig_e.savefig(FIGDIR / "76_E_phase_portrait.png", dpi=130); plt.close(fig_e)
print("Рис. E сохранён")

# ── итоговая таблица ──────────────────────────────────────────────────────────

print("\n══════════════════════════════════════════════════════")
print(f"{'W':>5} | {'rMAE':>8} | {'ΔrMAE%':>8} | {'SignAcc':>9}")
print("──────────────────────────────────────────────────────")
for W, mr, ms in zip(W_VALUES, mean_r, mean_s):
    delta = (mr - base_rmae) / base_rmae * 100
    pref  = "+" if delta >= 0 else ""
    print(f"{W:>5} | {mr:>8.4f} | {pref}{delta:>7.2f}% | {ms:>8.1f}%")
print("══════════════════════════════════════════════════════")
print("\nГотово. Графики:")
for ltr, fn in zip("ABCDE", ["76_A_acf.png","76_B_rmae.png","76_C_signacc.png",
                              "76_D_temporal_hist.png","76_E_phase_portrait.png"]):
    print(f"  {ltr}: {FIGDIR/fn}")
