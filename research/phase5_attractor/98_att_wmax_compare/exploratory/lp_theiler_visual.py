"""
Visual test: LP + Theiler нулевой уровень p-aligned каскада.

Архитектура:
  1. Theiler W=P_MAX → K=ξ_lwr кандидатов без «сдвиговых копий»
  2. Local LP (n=5) → d_local@{80%,90%}, ACF-критерий качества
  3. Если |ACF lag-1| > CI95 → p_fit отброшен
  4. Каскад P_MAX/2 → ... → p_fit (рефайнинг Евклид)
  5. LWR (acc_ang λ=0.01, Гауссовы веса)
  6. Ансамбль: среднее по прошедшим p_fit

Без LP-коррекции LWR. Без глобальных соседей.
"""

import json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# ── параметры ─────────────────────────────────────────────────────────────────
DATA_PATH    = "data/candles/SBER/1d.json"
P_FIT_GRID   = [5, 6, 7, 9, 10, 12, 14]
H_FORECAST   = 10        # шагов прогноза вперёд
H_VAL        = 10        # шагов валидации (назад от origin)
N_ORIGINS    = 3
ORIGIN_STEP  = 10        # баров между origins
D_THRESHOLDS = [0.80, 0.90]
ACC_LAMBDA   = 0.01
N_LP_ITER    = 5
HISTORY_SHOW = 60        # баров истории на графике

COLORS_P = {5: "#1f77b4", 6: "#ff7f0e", 7: "#2ca02c",
            9: "#d62728", 10: "#9467bd", 12: "#8c564b", 14: "#e377c2"}

# ── данные ────────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = json.load(f)
close  = np.array([c["close"] for c in raw], dtype=np.float64)
dates  = [r["begin"][:10] for r in raw]
N      = len(close)

def logtrend_causal(c):
    n = len(c); lc = np.log(np.maximum(c, 1e-10))
    t   = np.arange(n, dtype=np.float64)
    cn  = np.arange(1, n+1, dtype=np.float64)
    ct  = np.cumsum(t);  ct2 = np.cumsum(t**2)
    cy  = np.cumsum(lc); cty = np.cumsum(t * lc)
    denom = cn * ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a  = (cy - b * ct) / cn
    tr = np.exp(a + b * t); tr[:2] = c[:2]
    return tr

logtrend = logtrend_causal(close)
ratio    = close / np.maximum(logtrend, 1e-10)

# ── вспомогательные ──────────────────────────────────────────────────────────
def p_max_for(p):
    P_CAP = 64; m = 1
    while p * m * 2 <= P_CAP: m *= 2
    return p * m

def xi_for(p):     return 3 * (p + 1) + 5

def kp_for(xi):    return max(3, xi // 3)

def cascade_levels(p):
    P_MAX = p_max_for(p); levels = []
    lv = p
    while lv <= P_MAX: levels.append(lv); lv *= 2
    return list(reversed(levels))   # [P_MAX, ..., p]

# ── LP нулевой уровень ───────────────────────────────────────────────────────
def build_pool_theiler(t_orig, P_MAX, K, W):
    """
    K соседей (по сигнальному концу окна) в P_MAX-embedding
    с Theiler-исключением |t_end - t_orig| < W.
    Возвращает (signal_times, X_pool) или (None, None).
    """
    # Все прошлые embedding-концы: от P_MAX-1 до t_orig-1
    past_ends = np.arange(P_MAX - 1, t_orig, dtype=int)
    mask      = past_ends <= t_orig - W        # Theiler
    if mask.sum() < K:
        return None, None
    cand_ends = past_ends[mask]
    X_cands   = np.array([ratio[t - P_MAX + 1: t + 1] for t in cand_ends])
    x_q       = ratio[t_orig - P_MAX + 1: t_orig + 1]
    _, inds   = KDTree(X_cands).query(x_q.reshape(1, -1), k=K)
    return cand_ends[inds[0]], X_cands[inds[0]]

def svd_d_local(X, thresh):
    Xc = X - X.mean(axis=0)
    _, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    vf = sv**2 / (sv**2).sum(); cv = np.cumsum(vf)
    return int(np.searchsorted(cv, thresh)) + 1, sv, cv

def lp_pool_iter(X, d, k_prime):
    k_eff = min(k_prime, len(X) - 1)
    d_eff = min(d, k_eff - 1)
    _, ai  = KDTree(X).query(X, k=k_eff + 1)
    X_new  = np.empty_like(X)
    for i in range(len(X)):
        nn = ai[i, 1:]; Xnn = X[nn]; cent = Xnn.mean(0)
        _, _, Vt = np.linalg.svd(Xnn - cent, full_matrices=False)
        Vd = Vt[:d_eff]; xc = X[i] - cent
        X_new[i] = cent + Vd.T @ (Vd @ xc)
    return X_new

def lp_quality(X_pool, d_local, k_prime, n_iter, K, p_fit):
    """Возвращает (X_clean, acf1, determined)."""
    CI95 = 1.96 / np.sqrt(max(K * p_fit, 1))
    X    = X_pool.copy()
    for _ in range(n_iter):
        X = lp_pool_iter(X, d_local, k_prime)
    # ACF lag-1 P_FIT-срезов остатков
    resid = (X_pool - X)[:, :p_fit]
    vals  = 0.0; cnt = 0
    for k in range(len(X_pool)):
        r  = resid[k] - resid[k].mean()
        c0 = np.dot(r, r)
        if c0 < 1e-30 or p_fit < 2: continue
        vals += np.dot(r[:-1], r[1:]) / c0; cnt += 1
    acf1 = vals / max(cnt, 1)
    return X, acf1, abs(acf1) <= CI95

# ── каскадный рефайнинг ──────────────────────────────────────────────────────
def cascade_refine(pool_signal_times, t_orig, levels, xi):
    """
    Рефайним пул через уменьшение масштаба (от levels[1] до levels[-1]=p_fit).
    levels[0] = P_MAX (нулевой уровень, уже выполнен LP).
    """
    cands = pool_signal_times.copy()
    for L in levels[1:]:
        x_q_L   = ratio[t_orig - L + 1: t_orig + 1]
        X_c_L   = np.array([ratio[t - L + 1: t + 1] for t in cands])
        dists   = np.linalg.norm(X_c_L - x_q_L, axis=1)
        n_keep  = min(xi, len(cands))
        cands   = cands[np.argsort(dists)[:n_keep]]
    return cands

# ── LWR с acc_ang ─────────────────────────────────────────────────────────────
def lwr_one_step(x_q, X_cands, Y_cands, xi, lam=0.01):
    """Один шаг прогноза ratio (WLS с acc_ang + Гауссовы веса)."""
    d_pos   = np.linalg.norm(X_cands - x_q, axis=1)
    p = len(x_q)
    if p >= 3:
        acc_q = x_q[-1] - 2*x_q[-2] + x_q[-3]
        acc_c = X_cands[:, -1] - 2*X_cands[:, -2] + X_cands[:, -3]
        d_ang = np.abs(acc_c - acc_q)
    else:
        d_ang = np.zeros(len(X_cands))
    d_comb  = d_pos + lam * d_ang
    n_keep  = min(xi, len(X_cands))
    order   = np.argsort(d_comb)[:n_keep]
    X_sel   = X_cands[order]; Y_sel = Y_cands[order]; d_sel = d_comb[order]
    h       = np.median(d_sel) + 1e-10
    w       = np.exp(-0.5 * (d_sel / h)**2)
    Xd      = np.column_stack([np.ones(len(X_sel)), X_sel])
    W_mat   = np.diag(w)
    try:
        coef = np.linalg.lstsq(Xd.T @ W_mat @ Xd, Xd.T @ W_mat @ Y_sel,
                               rcond=None)[0]
        return float(np.dot(np.r_[1.0, x_q], coef))
    except Exception:
        return float(np.average(Y_sel, weights=w))

def forecast_ratio(pool_times, t_orig, p_fit, xi, H, lam=0.01):
    """H шагов рекурсивного прогноза ratio."""
    X_c = np.array([ratio[t - p_fit + 1: t + 1] for t in pool_times])
    Y_c = np.array([ratio[t + 1]              for t in pool_times])
    x   = ratio[t_orig - p_fit + 1: t_orig + 1].copy()
    preds = []
    for _ in range(H):
        y = lwr_one_step(x, X_c, Y_c, xi, lam)
        preds.append(y); x = np.r_[x[1:], y]
    return np.array(preds)

def to_price(ratio_pred, t_start):
    """ratio_pred → цены через logtrend (экстраполяция за пределами данных)."""
    prices = []
    for h, r in enumerate(ratio_pred):
        t = t_start + h
        lt = logtrend[t] if t < N else logtrend[-1]
        prices.append(r * lt)
    return np.array(prices)

# ── основной цикл ─────────────────────────────────────────────────────────────
t_origins   = [N - 1 - H_FORECAST - i * ORIGIN_STEP for i in range(N_ORIGINS)]
all_results = {}   # t_orig → d_thresh → p_fit → dict

for t_orig in t_origins:
    print(f"\n{'═'*65}")
    print(f"Origin: {dates[t_orig]}  (t={t_orig})")
    all_results[t_orig] = {}

    for d_thresh in D_THRESHOLDS:
        print(f"\n  d_thresh = {d_thresh*100:.0f}%")
        print(f"  {'p':>4} {'P_MAX':>5} {'W':>4} {'ξ':>4} "
              f"{'d_loc':>5} {'ACF1':>8} {'CI95':>7} {'ok':>3} {'rMAE_val':>10}")

        res_d = {}
        for p_fit in P_FIT_GRID:
            P_MAX   = p_max_for(p_fit)
            W       = P_MAX
            xi      = xi_for(p_fit)
            k_prime = kp_for(xi)
            levels  = cascade_levels(p_fit)

            # Пул с Theiler
            pool_times, X_pool = build_pool_theiler(t_orig, P_MAX, xi, W)
            if pool_times is None:
                print(f"  {p_fit:>4} {P_MAX:>5} {W:>4} {xi:>4} — мало данных"); continue

            # LP качество
            d_loc, _, _ = svd_d_local(X_pool, d_thresh)
            _, acf1, ok = lp_quality(X_pool, d_loc, k_prime, N_LP_ITER, xi, p_fit)
            CI95 = 1.96 / np.sqrt(xi * p_fit)

            if not ok:
                print(f"  {p_fit:>4} {P_MAX:>5} {W:>4} {xi:>4} "
                      f"{d_loc:>5} {acf1:>+8.4f} {CI95:>7.4f} {'✗':>3}")
                res_d[p_fit] = dict(ok=False, d_loc=d_loc, acf1=acf1, CI95=CI95)
                continue

            # Каскад + LWR → прогноз
            final_times   = cascade_refine(pool_times, t_orig, levels, xi)
            ratio_pred    = forecast_ratio(final_times, t_orig, p_fit, xi, H_FORECAST)
            price_pred    = to_price(ratio_pred, t_orig + 1)

            # Валидация: прогнозируем H_VAL шагов назад
            t_val = t_orig - H_VAL
            rmae  = np.nan
            if t_val >= P_MAX + W:
                pv, Xpv = build_pool_theiler(t_val, P_MAX, xi, W)
                if pv is not None:
                    _, _, ok_v = lp_quality(Xpv, d_loc, k_prime, N_LP_ITER, xi, p_fit)
                    if ok_v:
                        ft_v = cascade_refine(pv, t_val, levels, xi)
                        rp_v = forecast_ratio(ft_v, t_val, p_fit, xi, H_VAL)
                        rt_v = ratio[t_val + 1: t_val + 1 + H_VAL]
                        nv   = min(len(rp_v), len(rt_v))
                        std  = np.std(ratio[max(0, t_orig-60): t_orig]) + 1e-10
                        if nv > 0:
                            rmae = np.mean(np.abs(rp_v[:nv] - rt_v[:nv])) / std

            print(f"  {p_fit:>4} {P_MAX:>5} {W:>4} {xi:>4} "
                  f"{d_loc:>5} {acf1:>+8.4f} {CI95:>7.4f} {'✓':>3} "
                  f"{rmae:>10.4f}" if not np.isnan(rmae) else
                  f"  {p_fit:>4} {P_MAX:>5} {W:>4} {xi:>4} "
                  f"{d_loc:>5} {acf1:>+8.4f} {CI95:>7.4f} {'✓':>3} {'—':>10}")

            res_d[p_fit] = dict(ok=True, d_loc=d_loc, acf1=acf1, CI95=CI95,
                                ratio_pred=ratio_pred, price_pred=price_pred, rmae=rmae)
        all_results[t_orig][d_thresh] = res_d

# ── графики: один на d_thresh ─────────────────────────────────────────────────
for d_thresh in D_THRESHOLDS:
    n_orig = len(t_origins)
    fig, axes = plt.subplots(1, n_orig, figsize=(8 * n_orig, 7), sharey=False)
    if n_orig == 1: axes = [axes]
    fig.suptitle(
        f"LP + Theiler каскад  |  d_thresh={int(d_thresh*100)}%  |  SBER 1d  |  "
        f"acc_ang λ={ACC_LAMBDA}  H={H_FORECAST}",
        fontsize=13, fontweight="bold"
    )

    for ax, t_orig in zip(axes, t_origins):
        date_o = dates[t_orig]
        res    = all_results[t_orig].get(d_thresh, {})

        # История цены
        t_lo = max(0, t_orig - HISTORY_SHOW)
        t_hi = min(N - 1, t_orig + H_FORECAST)
        ax.plot(np.arange(t_lo, t_orig + 1), close[t_lo: t_orig + 1],
                color="black", lw=1.8, label="SBER (факт)", zorder=5)
        ax.axvline(t_orig, color="navy", ls="--", lw=1.2, alpha=0.6)

        # Будущие реальные цены (для ориентира)
        t_fut = np.arange(t_orig + 1, min(N, t_orig + H_FORECAST + 1))
        if len(t_fut):
            ax.plot(t_fut, close[t_fut], color="black", lw=1.8, ls=":",
                    alpha=0.45, label="факт (будущее)", zorder=5)

        # Прогнозы по p_fit
        valid_preds = []
        for p_fit in P_FIT_GRID:
            r = res.get(p_fit, {})
            if not r.get("ok", False): continue
            t_pred = np.arange(t_orig + 1, t_orig + 1 + H_FORECAST)
            pp     = r["price_pred"][:len(t_pred)]
            col    = COLORS_P.get(p_fit, "gray")
            rmae_s = f"{r['rmae']:.3f}" if not np.isnan(r.get("rmae", np.nan)) else "?"
            ax.plot(t_pred, pp, color=col, lw=1.3, alpha=0.7,
                    label=f"p={p_fit}  d={r['d_loc']}  rMAE={rmae_s}")
            valid_preds.append(r["ratio_pred"][:H_FORECAST])

        # Ансамбль
        if valid_preds:
            min_len  = min(len(v) for v in valid_preds)
            ens_ratio = np.mean([v[:min_len] for v in valid_preds], axis=0)
            ens_price = to_price(ens_ratio, t_orig + 1)
            t_pred    = np.arange(t_orig + 1, t_orig + 1 + len(ens_price))
            ax.plot(t_pred, ens_price, color="red", lw=2.8, ls="-",
                    zorder=6, label="Ансамбль (среднее)")

        # Отброшенные p_fit
        rejected = [p for p, r in res.items() if not r.get("ok", False)]
        n_ok     = sum(1 for r in res.values() if r.get("ok", False))

        if rejected:
            ax.text(0.02, 0.03, f"Отброшены: {rejected}",
                    transform=ax.transAxes, fontsize=8, color="darkred",
                    bbox=dict(fc="mistyrose", ec="darkred", alpha=0.9))

        ax.set_title(f"origin {date_o}  |  прошли LP: {n_ok}/{len(P_FIT_GRID)}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Индекс бара"); ax.set_ylabel("Цена")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.85)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"research/phase5_attractor/lp_theiler_visual_d{int(d_thresh*100)}.png"
    fig.savefig(out, dpi=130)
    print(f"\n  → {out}")

# ── сводная таблица: сколько p_fit прошли LP по origin и thresh ───────────────
print(f"\n{'─'*55}")
print(f"{'':>12}", end="")
for d_thresh in D_THRESHOLDS:
    print(f"  d={int(d_thresh*100)}%", end="")
print()
for t_orig in t_origins:
    print(f"{dates[t_orig]:>12}", end="")
    for d_thresh in D_THRESHOLDS:
        res = all_results[t_orig].get(d_thresh, {})
        ok_ps  = [p for p, r in res.items() if r.get("ok", False)]
        bad_ps = [p for p, r in res.items() if not r.get("ok", False)]
        print(f"  {len(ok_ps)}/7 ✓{ok_ps}  ✗{bad_ps}", end="")
    print()

plt.show()
print("\nГотово.")
