"""
98_att_wmax_compare.py — глобальный LP как шумодав + W_max для адаптивного p.

Три метода на тех же 5 origins, что lp_wmax_visual.py:

  A (синий):  att → W_max(LP-clean пула + ACF) → d_local → p=2d+2 → LWR(acc_ang) + LP corr
  B (красный): att → W_max(SVD пула, без ACF, берём максимальный) → d_local → p=2d+2 → LWR(acc_ang)
  C (зелёный): att → фикс. p=16, всё прошлое → LWR(acc_ang)  [baseline]

Ключевое: att = LP(ratio) c глобальными параметрами (m=9, d=3, k=30, n=3),
          применяется ко всему ряду до прогноза (некаузальный — только для визуального теста).
"""

import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# ── Параметры ──────────────────────────────────────────────────────────────────
DATA_PATH    = "data/candles/SBER/1d.json"
N_ORIGINS    = 5
ORIGIN_STEP  = 100
H_FORECAST   = 10
HISTORY_SHOW = 50

# Глобальный LP-фильтр (att)
ATT_M = 9;  ATT_D = 3;  ATT_K = 30;  ATT_N = 3

# W_max search
P_SEARCH  = 64
K_SEARCH  = 50
K_PRIME   = 10
N_LP_ITER = 20
D_THRESH  = 0.80
W_MIN     = 9
W_STEP    = 4
MARGIN    = 2      # p_fit = 2·d_local + MARGIN

# LWR
ACC_LAMBDA = 0.01
K_CORR     = K_PRIME

# Baseline (метод C)
P_BASELINE = 16

OUT_PATH = "research/phase5_attractor/98_att_wmax_compare.png"

# ── Данные ─────────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = json.load(f)
close = np.array([c["close"] for c in raw], dtype=np.float64)
dates = [r["begin"][:10] for r in raw]
N     = len(close)
print(f"SBER 1d: {N} баров  ({dates[0]} … {dates[-1]})")


def logtrend_causal(c):
    n = len(c); lc = np.log(np.maximum(c, 1e-10))
    t   = np.arange(n, dtype=np.float64)
    cn  = np.arange(1, n + 1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t ** 2)
    cy  = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a  = (cy - b * ct) / cn
    tr = np.exp(a + b * t); tr[:2] = c[:2]
    return tr


logtrend = logtrend_causal(close)
ratio    = close / np.maximum(logtrend, 1e-10)

# ── Каузальный LP-фильтр (att) ────────────────────────────────────────────────
def lp_filter_causal(series, m, d, k, n_iter):
    """
    Каузальный LP-фильтр: для каждой точки t проецируем delay-вектор
    только на соседей из строго прошлого (delay-векторы, заканчивающиеся до t).
    Гарантирует отсутствие утечки будущего в att[t].
    """
    N = len(series)
    s = series.copy()
    for it in range(n_iter):
        X_all = np.array([s[t - m + 1: t + 1] for t in range(m - 1, N)])
        s_new = s.copy()
        for i in range(len(X_all)):
            if i < k:
                continue  # недостаточно прошлых точек — оставляем как есть
            v      = X_all[i]
            X_past = X_all[:i]              # строго прошлые delay-векторы
            dists  = np.linalg.norm(X_past - v, axis=1)
            k_eff  = min(k, i)
            idx    = np.argpartition(dists, k_eff - 1)[:k_eff]
            Xnn    = X_past[idx]; cent = Xnn.mean(0)
            _, _, Vt = np.linalg.svd(Xnn - cent, full_matrices=False)
            d_eff  = min(d, k_eff - 1)
            Vd     = Vt[:d_eff]; xc = v - cent
            s_new[m - 1 + i] = (cent + Vd.T @ (Vd @ xc))[-1]
        s = s_new
        print(f"  LP iter {it + 1}/{n_iter}", flush=True)
    return s


print("\nВычисляем att = LP(ratio) [каузально]...", flush=True)
t0_lp = time.time()
att = lp_filter_causal(ratio, ATT_M, ATT_D, ATT_K, ATT_N)
print(f"att готов за {time.time()-t0_lp:.1f}с\n", flush=True)

price_att   = att * logtrend
price_close = close

# ── LP-утилиты для пула ────────────────────────────────────────────────────────
def svd_d_local(X):
    Xc = X - X.mean(0)
    _, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    cv = np.cumsum(sv ** 2) / (sv ** 2).sum()
    return int(np.searchsorted(cv, D_THRESH)) + 1


def lp_step(X, d, k_prime):
    k_eff = min(k_prime, len(X) - 1)
    d_eff = min(d, k_eff - 1)
    _, ai = KDTree(X).query(X, k=k_eff + 1)
    X_new = np.empty_like(X)
    for i in range(len(X)):
        Xnn  = X[ai[i, 1:]]; cent = Xnn.mean(0)
        _, _, Vt = np.linalg.svd(Xnn - cent, full_matrices=False)
        Vd = Vt[:d_eff]; xc = X[i] - cent
        X_new[i] = cent + Vd.T @ (Vd @ xc)
    return X_new


def lp_clean(X_pool, d, k_prime, n_iter):
    X = X_pool.copy()
    for _ in range(n_iter):
        X = lp_step(X, d, k_prime)
    return X


def acf1_ok(X_orig, X_clean, d_after):
    P_ref = max(7, 2 * d_after + 1)
    resid = (X_orig - X_clean)[:, :P_ref]
    K     = len(X_orig)
    CI95  = 1.96 / np.sqrt(max(K * P_ref, 1))
    vals  = 0.0; cnt = 0
    for row in resid:
        r = row - row.mean(); c0 = np.dot(r, r)
        if c0 < 1e-30 or P_ref < 2: continue
        vals += np.dot(r[:-1], r[1:]) / c0; cnt += 1
    return abs(vals / max(cnt, 1)) <= CI95

# ── Пул из att c Theiler ───────────────────────────────────────────────────────
def build_pool_att(t_orig, P, K, W):
    """K ближайших в P-embedding из att с Theiler W."""
    past_ends = np.arange(P - 1, t_orig, dtype=int)
    mask      = past_ends <= t_orig - W
    if mask.sum() < K:
        return None, None
    cand_ends = past_ends[mask]
    X_cands   = np.array([att[t - P + 1: t + 1] for t in cand_ends])
    x_q       = att[t_orig - P + 1: t_orig + 1]
    _, inds   = KDTree(X_cands).query(x_q.reshape(1, -1), k=K)
    return cand_ends[inds[0]], X_cands[inds[0]]

# ── Поиск W_max: Метод A (LP-clean + ACF) ─────────────────────────────────────
def search_wmax_A(t_orig):
    w_grid = list(range(W_MIN, P_SEARCH + 1, W_STEP))
    W_max = None; d_local = None
    pool_best = None; X_clean_best = None
    for W in w_grid:
        pool_times, X_pool = build_pool_att(t_orig, P_SEARCH, K_SEARCH, W)
        if pool_times is None: continue
        d_b   = svd_d_local(X_pool)
        X_cln = lp_clean(X_pool, d_b, K_PRIME, N_LP_ITER)
        d_a   = svd_d_local(X_cln)
        if acf1_ok(X_pool, X_cln, d_a):
            W_max = W; d_local = d_a
            pool_best     = pool_times.copy()
            X_clean_best  = X_cln.copy()
    return W_max, d_local, pool_best, X_clean_best

# ── Поиск W_max: Метод B (SVD только, берём максимальный W) ───────────────────
def search_wmax_B(t_orig):
    w_grid = list(range(W_MIN, P_SEARCH + 1, W_STEP))
    W_max = None; d_local = None; pool_best = None
    for W in w_grid:
        pool_times, X_pool = build_pool_att(t_orig, P_SEARCH, K_SEARCH, W)
        if pool_times is None: continue
        W_max    = W
        d_local  = svd_d_local(X_pool)
        pool_best = pool_times.copy()
    return W_max, d_local, pool_best

# ── LWR + acc_ang ──────────────────────────────────────────────────────────────
def xi_for(p):
    return 3 * (p + 1) + 5


def lwr_one_step(x_q, X_c, Y_c, xi):
    d_pos = np.linalg.norm(X_c - x_q, axis=1)
    p = len(x_q)
    if p >= 3:
        acc_q = x_q[-1] - 2 * x_q[-2] + x_q[-3]
        acc_c = X_c[:, -1] - 2 * X_c[:, -2] + X_c[:, -3]
        d_comb = d_pos + ACC_LAMBDA * np.abs(acc_c - acc_q)
    else:
        d_comb = d_pos
    n_keep = min(xi, len(X_c))
    order  = np.argsort(d_comb)[:n_keep]
    Xs, Ys, ds = X_c[order], Y_c[order], d_comb[order]
    h = np.median(ds) + 1e-10
    w = np.exp(-0.5 * (ds / h) ** 2)
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    Wm = np.diag(w)
    try:
        coef = np.linalg.lstsq(Xd.T @ Wm @ Xd, Xd.T @ Wm @ Ys, rcond=None)[0]
        return float(np.dot(np.r_[1.0, x_q], coef))
    except Exception:
        return float(np.average(Ys, weights=w))

# ── LP-коррекция шага ──────────────────────────────────────────────────────────
def lp_corr_step(v, X_lib_pfit, d_local):
    k_eff = min(K_CORR, len(X_lib_pfit) - 1)
    d_eff = min(d_local, k_eff - 1)
    if d_eff < 1: return float(v[-1])
    dists = np.linalg.norm(X_lib_pfit - v, axis=1)
    idx   = np.argpartition(dists, k_eff)[:k_eff]
    Xnn   = X_lib_pfit[idx]; cent = Xnn.mean(0)
    _, _, Vt = np.linalg.svd(Xnn - cent, full_matrices=False)
    Vd = Vt[:d_eff]; xc = v - cent
    return float((cent + Vd.T @ (Vd @ xc))[-1])

# ── Рекурсивный прогноз att ────────────────────────────────────────────────────
def recursive_forecast(t_orig, p_fit, pool_times, X_lib_corr=None, d_local=None):
    """
    pool_times: конечные индексы окон att в пуле.
    X_lib_corr: (K × P_SEARCH) LP-очищенный пул для LP-коррекции (метод A).
    Возвращает (att_pred, price_pred).
    """
    xi = xi_for(p_fit)
    X_raw = np.array([att[t - p_fit + 1: t + 1] for t in pool_times])
    Y_raw = np.array([att[t + 1]                 for t in pool_times])
    x_q   = att[t_orig - p_fit + 1: t_orig + 1]

    # Рефайн: ξ ближайших в p_fit-пространстве
    dists  = np.linalg.norm(X_raw - x_q, axis=1)
    order  = np.argsort(dists)[:xi]
    X_c, Y_c = X_raw[order], Y_raw[order]

    # LP-коррекционная библиотека: последние p_fit координат LP-очищенного пула
    X_lib_pfit = X_lib_corr[:, -p_fit:] if X_lib_corr is not None else None

    x = x_q.copy(); preds = []
    for _ in range(H_FORECAST):
        y = lwr_one_step(x, X_c, Y_c, xi)
        x_new = np.r_[x[1:], y]
        if X_lib_pfit is not None and d_local is not None:
            y = lp_corr_step(x_new, X_lib_pfit, d_local)
            x_new = np.r_[x[1:], y]
        preds.append(y); x = x_new

    att_pred   = np.array(preds)
    price_pred = att_pred * np.array([
        logtrend[t_orig + 1 + h] if t_orig + 1 + h < N else logtrend[-1]
        for h in range(H_FORECAST)
    ])
    return att_pred, price_pred

# ── Метод C: baseline ──────────────────────────────────────────────────────────
def forecast_C(t_orig):
    p = P_BASELINE; xi = xi_for(p)
    pool_times = np.arange(p - 1, t_orig, dtype=int)
    if len(pool_times) < xi:
        return None, None
    return recursive_forecast(t_orig, p, pool_times)

# ── Origins (те же, что lp_wmax_visual.py) ────────────────────────────────────
t_origins = [N - 1 - H_FORECAST - i * ORIGIN_STEP for i in range(N_ORIGINS)]

# ── Основной цикл ─────────────────────────────────────────────────────────────
results = []
for t_orig in t_origins:
    print(f"{'═'*60}")
    print(f"Origin: {dates[t_orig]}  (t={t_orig})", flush=True)
    t0 = time.time()

    # Метод A
    print("  A: W_max + LP-clean + LP-corr...", flush=True)
    W_max_A, d_A, pool_A, X_clean_A = search_wmax_A(t_orig)
    if W_max_A is not None:
        p_fit_A = 2 * d_A + MARGIN
        _, price_A = recursive_forecast(t_orig, p_fit_A, pool_A,
                                        X_lib_corr=X_clean_A, d_local=d_A)
        print(f"  A: W_max={W_max_A}  d={d_A}  p_fit={p_fit_A}", flush=True)
    else:
        p_fit_A = None; price_A = None
        print("  A: W_max не найден — пропуск", flush=True)

    # Метод B
    print("  B: W_max(SVD only, макс. W)...", flush=True)
    W_max_B, d_B, pool_B = search_wmax_B(t_orig)
    if W_max_B is not None:
        p_fit_B = 2 * d_B + MARGIN
        _, price_B = recursive_forecast(t_orig, p_fit_B, pool_B)
        print(f"  B: W_max={W_max_B}  d={d_B}  p_fit={p_fit_B}", flush=True)
    else:
        p_fit_B = None; price_B = None
        print("  B: пул не построен — пропуск", flush=True)

    # Метод C
    _, price_C = forecast_C(t_orig)
    print(f"  C: p={P_BASELINE}  {'ok' if price_C is not None else 'FAIL'}", flush=True)

    print(f"  ({time.time()-t0:.1f}с)", flush=True)

    results.append(dict(
        t_orig=t_orig, date=dates[t_orig],
        W_max_A=W_max_A, d_A=d_A, p_fit_A=p_fit_A, price_A=price_A,
        W_max_B=W_max_B, d_B=d_B, p_fit_B=p_fit_B, price_B=price_B,
        price_C=price_C,
    ))

# ── Фигура ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, N_ORIGINS, figsize=(5 * N_ORIGINS, 5))
if N_ORIGINS == 1:
    axes = [axes]

for ax, res in zip(axes, results):
    t0      = res["t_orig"]
    t_start = max(0, t0 - HISTORY_SHOW)
    idx_h   = np.arange(t_start, t0 + 1)
    idx_f   = np.arange(t0 + 1, min(t0 + H_FORECAST + 1, N))
    fut_idx = np.arange(t0 + 1, t0 + 1 + H_FORECAST)

    # Реальные данные
    ax.plot(idx_h, price_close[idx_h], color="lightgray", lw=0.7, zorder=1, label="close")
    ax.plot(idx_h, price_att[idx_h],   "k-", lw=1.5, zorder=2, label="att")
    ax.plot(idx_f, price_att[idx_f],   "k--", lw=1,  zorder=2, alpha=0.45, label="att (fut)")
    ax.axvline(t0, color="gray", ls=":", lw=0.8, alpha=0.6)

    # A
    if res["price_A"] is not None:
        lbl = f"A: W={res['W_max_A']},d={res['d_A']},p={res['p_fit_A']}"
        ax.plot(fut_idx, res["price_A"], "b-o", ms=3.5, lw=1.8, zorder=4, label=lbl)

    # B
    if res["price_B"] is not None:
        lbl = f"B: d={res['d_B']},p={res['p_fit_B']}"
        ax.plot(fut_idx, res["price_B"], "r-s", ms=3.5, lw=1.5, zorder=3, label=lbl)

    # C
    if res["price_C"] is not None:
        ax.plot(fut_idx, res["price_C"], "g-^", ms=3, lw=1.2, zorder=3,
                label=f"C: p={P_BASELINE}")

    ax.set_title(res["date"], fontsize=9)
    ax.legend(fontsize=6, loc="upper left")
    ax.set_xlabel("bar", fontsize=8)

fig.suptitle(
    "98: att как шумодав → W_max адаптивный p\n"
    "A=LP-clean+LP-corr (синий) | B=SVD-only (красный) | C=фикс.p=16 (зелёный)",
    fontsize=10
)
plt.tight_layout()
fig.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
print(f"\nФигура: {OUT_PATH}", flush=True)
plt.close()
