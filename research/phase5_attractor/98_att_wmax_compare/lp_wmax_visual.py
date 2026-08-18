"""
lp_wmax_visual.py — адаптивный поиск W_max + теоретически верный p_fit (Такенс).

Алгоритм:
  1. P_search=64 (фикс.). Скан W от W_min до P_search (шаг 4).
  2. При каждом W: Theiler-пул K=50 в 64D → LP (n=20) → d_eff (SVD@80%) → ACF-критерий.
     ACF: signed mean lag-1, P_ref = max(7, 2·d_after+1), CI95 = 1.96/√(K·P_ref).
  3. W_max = наибольшее W, где критерий проходит. Сложные origins отбрасываются.
  4. p_fit = 2·d_local + MARGIN, три варианта: MARGIN ∈ {1, 2, 3}.
  5. Прогноз БЕЗ каскада: W_max пул → рефайн до ξ в p_fit-пространстве → LWR + acc_ang.
     Опционально: LP-коррекция каждого шага через X_clean[:, -p_fit:] (локальное многообразие).
  БОНУС: GP корреляционная размерность D₂ на полном ряду ratio.
"""

import json, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

# ── параметры ─────────────────────────────────────────────────────────────────
DATA_PATH    = "data/candles/SBER/1d.json"
P_SEARCH     = 64      # embedding для фазы поиска W
K_SEARCH     = 50      # размер Theiler-пула
K_PRIME      = 10      # LP-соседей внутри пула
N_LP_ITER    = 20      # LP-итераций при поиске
D_THRESH     = 0.80    # порог дисперсии SVD → d_local
W_MIN        = 9       # мин. Theiler-окно
W_STEP       = 4       # шаг сетки W
H_FORECAST   = 10      # шагов прогноза
N_ORIGINS    = 5
ORIGIN_STEP  = 100     # баров между origins
ACC_LAMBDA   = 0.01    # acc_ang λ
HISTORY_SHOW = 80      # баров истории на графике
MARGINS      = [1, 2, 3]   # p_fit = 2·d_local + MARGIN (три варианта)
K_CORR       = K_PRIME     # LP-коррекция: те же параметры что при очистке

# GP
GP_M         = 10      # embedding dim для GP
GP_N_EPS     = 25      # точек шкалы ε
GP_MAX_PTS   = 800     # точек embedding для C(ε)

# ── данные ────────────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = json.load(f)
close   = np.array([c["close"] for c in raw], dtype=np.float64)
dates   = [r["begin"][:10] for r in raw]
N       = len(close)
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

# ── каскадные утилиты ─────────────────────────────────────────────────────────
def p_max_for(p):
    P_CAP = 64; m = 1
    while p * m * 2 <= P_CAP:
        m *= 2
    return p * m


def xi_for(p):
    return 3 * (p + 1) + 5


def cascade_levels(p):
    P_MAX = p_max_for(p); lvs = []
    lv = p
    while lv <= P_MAX:
        lvs.append(lv); lv *= 2
    return list(reversed(lvs))   # [P_MAX, ..., p]

# ── пул с Theiler ─────────────────────────────────────────────────────────────
def build_pool(t_orig, P, K, W):
    """K ближайших соседей (конец окна) в P-embedding с Theiler |t_end − t_orig| ≥ W."""
    past_ends = np.arange(P - 1, t_orig, dtype=int)
    mask      = past_ends <= t_orig - W
    if mask.sum() < K:
        return None, None
    cand_ends = past_ends[mask]
    X_cands   = np.array([ratio[t - P + 1: t + 1] for t in cand_ends])
    x_q       = ratio[t_orig - P + 1: t_orig + 1]
    _, inds   = KDTree(X_cands).query(x_q.reshape(1, -1), k=K)
    return cand_ends[inds[0]], X_cands[inds[0]]

# ── Local Projective ──────────────────────────────────────────────────────────
def svd_d_local(X, thresh):
    Xc = X - X.mean(axis=0)
    _, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    vf = sv ** 2 / (sv ** 2).sum(); cv = np.cumsum(vf)
    d  = int(np.searchsorted(cv, thresh)) + 1
    return d, sv, cv


def lp_step(X, d, k_prime):
    k_eff = min(k_prime, len(X) - 1)
    d_eff = min(d, k_eff - 1)
    _, ai  = KDTree(X).query(X, k=k_eff + 1)
    X_new  = np.empty_like(X)
    for i in range(len(X)):
        nn  = ai[i, 1:]; Xnn = X[nn]; cent = Xnn.mean(0)
        _, _, Vt = np.linalg.svd(Xnn - cent, full_matrices=False)
        Vd = Vt[:d_eff]; xc = X[i] - cent
        X_new[i] = cent + Vd.T @ (Vd @ xc)
    return X_new


def lp_clean(X_pool, d, k_prime, n_iter):
    X = X_pool.copy()
    for _ in range(n_iter):
        X = lp_step(X, d, k_prime)
    return X


def acf1_score(X_orig, X_clean, d_after):
    """
    Signed mean ACF lag-1 по первым P_ref = max(7, 2·d_after+1) координатам остатков.
    Адаптивный P_ref = Такенс нижняя граница embedding для d_after.
    CI95 = 1.96/√(K·P_ref).
    Возвращает (abs_acf1, CI95, P_ref, ok).
    """
    P_ref = max(7, 2 * d_after + 1)
    resid = (X_orig - X_clean)[:, :P_ref]
    K     = len(X_orig)
    CI95  = 1.96 / np.sqrt(max(K * P_ref, 1))
    vals  = 0.0; cnt = 0
    for k in range(K):
        r  = resid[k] - resid[k].mean()
        c0 = np.dot(r, r)
        if c0 < 1e-30 or P_ref < 2:
            continue
        vals += np.dot(r[:-1], r[1:]) / c0; cnt += 1
    acf1_signed = vals / max(cnt, 1)
    abs_acf1    = abs(acf1_signed)
    return abs_acf1, CI95, P_ref, abs_acf1 <= CI95

# ── поиск W_max ───────────────────────────────────────────────────────────────
def search_wmax(t_orig):
    """
    Скан W ∈ [W_MIN..P_SEARCH] шаг W_STEP.
    Для каждого W: Theiler-пул 64D → LP (d_before) → d_after → ACF.
    W_max = наибольшее W с ok=True.  d_local = d_after при W_max.
    """
    w_grid  = list(range(W_MIN, P_SEARCH + 1, W_STEP))
    history = []
    W_max   = None
    d_local = None

    for W in w_grid:
        pool_times, X_pool = build_pool(t_orig, P_SEARCH, K_SEARCH, W)
        if pool_times is None:
            history.append(dict(W=W, ok=False, n_cands=0,
                                d_before=np.nan, d_after=np.nan,
                                P_ref=np.nan, acf1=np.nan, CI95=np.nan,
                                sv_before=None, cv_before=None,
                                sv_after=None,  cv_after=None,
                                pool_times=None, X_pool=None, X_clean=None))
            continue

        d_before, sv_b, cv_b = svd_d_local(X_pool, D_THRESH)
        X_clean              = lp_clean(X_pool, d_before, K_PRIME, N_LP_ITER)
        d_after, sv_a, cv_a  = svd_d_local(X_clean, D_THRESH)
        acf1, CI95, P_ref, ok = acf1_score(X_pool, X_clean, d_after)

        history.append(dict(
            W=W, ok=ok, n_cands=len(pool_times),
            d_before=d_before, d_after=d_after,
            P_ref=P_ref, acf1=acf1, CI95=CI95,
            sv_before=sv_b, cv_before=cv_b,
            sv_after=sv_a,  cv_after=cv_a,
            pool_times=pool_times.copy(),
            X_pool=X_pool.copy(), X_clean=X_clean.copy()
        ))

        if ok:
            W_max   = W
            d_local = d_after   # истинная размерность аттрактора после очистки

    return dict(history=history, W_max=W_max, d_local=d_local)

# ── LWR + acc_ang ─────────────────────────────────────────────────────────────
def lwr_one_step(x_q, X_c, Y_c, xi, lam=0.01):
    d_pos = np.linalg.norm(X_c - x_q, axis=1)
    p = len(x_q)
    if p >= 3:
        acc_q = x_q[-1] - 2 * x_q[-2] + x_q[-3]
        acc_c = X_c[:, -1] - 2 * X_c[:, -2] + X_c[:, -3]
        d_ang = np.abs(acc_c - acc_q)
    else:
        d_ang = np.zeros(len(X_c))
    d_comb = d_pos + lam * d_ang
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


# ── LP-коррекция предсказания ─────────────────────────────────────────────────
def lp_corr_step(v, X_lib_pfit, d_local):
    """
    Проецирует вектор v (длина p_fit) на d_local-мерную локальную плоскость
    аттрактора, оцениваемую по X_lib_pfit (K × p_fit, LP-очищенный пул).
    Параметры (d, k) — те же, что при очистке пула (K_CORR = K_PRIME).
    Возвращает последнюю координату проекции (предсказание следующего ratio).
    """
    k_eff = min(K_CORR, len(X_lib_pfit) - 1)
    d_eff = min(d_local, k_eff - 1)
    dists = np.linalg.norm(X_lib_pfit - v, axis=1)
    idx   = np.argpartition(dists, k_eff)[:k_eff]
    Xnn   = X_lib_pfit[idx]; cent = Xnn.mean(0)
    _, _, Vt = np.linalg.svd(Xnn - cent, full_matrices=False)
    Vd = Vt[:d_eff]; xc = v - cent
    projected = cent + Vd.T @ (Vd @ xc)
    return float(projected[-1])


# ── прогноз без каскада ───────────────────────────────────────────────────────
def forecast_direct(pool_times, X_clean, t_orig, p_fit, d_local, use_lp_corr):
    """
    Прогноз напрямую из W_max пула (без каскада):
      1. Берём p_fit-срез из X_clean для библиотеки LP-коррекции.
      2. Рефайним pool_times → ξ ближайших в p_fit-пространстве (по raw ratio).
      3. LWR + acc_ang (рекурсивно H шагов).
      4. После каждого шага — опционально LP-коррекция через X_lib_pfit.
    """
    xi = xi_for(p_fit)

    # Библиотека для LP-коррекции: последние p_fit координат X_clean
    X_lib_pfit = X_clean[:, -p_fit:]   # K × p_fit

    # X/Y для LWR — сырой ratio в p_fit-пространстве
    X_raw = np.array([ratio[t - p_fit + 1: t + 1] for t in pool_times])
    Y_raw = np.array([ratio[t + 1]               for t in pool_times])

    # Рефайн: сортируем по расстоянию в p_fit-пространстве, берём ξ
    x_q = ratio[t_orig - p_fit + 1: t_orig + 1]
    dists_pfit = np.linalg.norm(X_raw - x_q, axis=1)
    order_pfit = np.argsort(dists_pfit)[:xi]
    X_c = X_raw[order_pfit]; Y_c = Y_raw[order_pfit]

    x = x_q.copy()
    preds = []
    for _ in range(H_FORECAST):
        y = lwr_one_step(x, X_c, Y_c, xi, ACC_LAMBDA)
        x_new = np.r_[x[1:], y]
        if use_lp_corr:
            y = lp_corr_step(x_new, X_lib_pfit, d_local)
            x_new = np.r_[x[1:], y]
        preds.append(y); x = x_new

    ratio_pred = np.array(preds)
    price_pred = np.array([
        ratio_pred[h] * (logtrend[t_orig + 1 + h] if t_orig + 1 + h < N else logtrend[-1])
        for h in range(H_FORECAST)
    ])
    return ratio_pred, price_pred

# ── GP корреляционная размерность ─────────────────────────────────────────────
def gp_corr_dim(series):
    """D₂ по Grassberger–Procaccia. Delay embedding m=GP_M, τ=1."""
    n_pts = len(series) - GP_M + 1
    X     = np.array([series[i: i + GP_M] for i in range(n_pts)])
    np.random.seed(42)
    if len(X) > GP_MAX_PTS:
        idx = np.random.choice(len(X), GP_MAX_PTS, replace=False)
        X   = X[idx]
    dists    = pdist(X)
    eps_min  = np.percentile(dists[dists > 0], 3)
    eps_max  = np.percentile(dists, 60)
    epsilons = np.logspace(np.log10(eps_min), np.log10(eps_max), GP_N_EPS)
    C_eps    = np.array([np.mean(dists < e) for e in epsilons])
    valid    = (C_eps > 0.02) & (C_eps < 0.95)
    if valid.sum() < 4:
        return np.nan, epsilons, C_eps
    D2 = np.polyfit(np.log(epsilons[valid]), np.log(C_eps[valid]), 1)[0]
    return D2, epsilons, C_eps

# ── основной цикл ─────────────────────────────────────────────────────────────
t_origins   = [N - 1 - H_FORECAST - i * ORIGIN_STEP for i in range(N_ORIGINS)]
all_results = []

w_grid = list(range(W_MIN, P_SEARCH + 1, W_STEP))

print(f"\nP_search={P_SEARCH}  K={K_SEARCH}  k'={K_PRIME}  LP={N_LP_ITER}iter  "
      f"d@{int(D_THRESH*100)}%")
print(f"W_grid: {w_grid}")
print(f"P_ref = max(7, 2·d_after+1) адаптивно;  CI95 = 1.96/√(K·P_ref)\n")

for t_orig in t_origins:
    print(f"{'═'*70}")
    print(f"Origin: {dates[t_orig]}  (t={t_orig})")

    t0   = time.time()
    scan = search_wmax(t_orig)
    W_max   = scan["W_max"]
    d_local = scan["d_local"]

    print(f"  {'W':>4} {'n_cnd':>5} {'d_b':>4} {'d_a':>4} {'P_ref':>6} "
          f"{'CI95':>7} {'ACF':>8} {'ok':>3}")
    for h in scan["history"]:
        if np.isnan(h["acf1"]):
            print(f"  {h['W']:>4} {'—':>5} {'—':>4} {'—':>4} {'мало данных':>20}")
            continue
        ok_s = "✓" if h["ok"] else "✗"
        print(f"  {h['W']:>4} {h['n_cands']:>5} {h['d_before']:>4} {h['d_after']:>4} "
              f"{h['P_ref']:>6} {h['CI95']:>7.4f} {h['acf1']:>+8.4f} {ok_s:>3}")

    print(f"\n  → W_max={W_max}  d_local={d_local}")

    # Находим X_clean при W_max для LP-коррекции
    wmax_entry = None
    if W_max is not None:
        for h in scan["history"]:
            if h["W"] == W_max:
                wmax_entry = h; break

    res = dict(t_orig=t_orig, date=dates[t_orig], scan=scan,
               p_fits=[], forecasts={})   # forecasts: (margin, lp_corr) → (rp, pp)

    if W_max is not None and d_local is not None and wmax_entry is not None:
        pool_times = wmax_entry["pool_times"]
        X_clean    = wmax_entry["X_clean"]
        p_fits     = [2 * d_local + m for m in MARGINS]
        res["p_fits"] = p_fits

        print(f"  p_fits = {p_fits}  (margins {MARGINS})  |  LP-corr k={K_CORR} d={d_local}")
        for p_fit in p_fits:
            for lp in [False, True]:
                rp, pp = forecast_direct(pool_times, X_clean, t_orig,
                                         p_fit, d_local, use_lp_corr=lp)
                res["forecasts"][(p_fit, lp)] = (rp, pp)
                tag = "LP✓" if lp else "LP✗"
                print(f"    p={p_fit:2d} {tag}: {'ok' if pp is not None else 'FAIL'}")
    else:
        print("  → W_max не найден, прогноз пропущен")

    res["elapsed"] = time.time() - t0
    all_results.append(res)
    print(f"  Время: {res['elapsed']:.1f}с")

# ── GP ────────────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("GP корреляционная размерность (ratio, полный ряд)...")
D2, gp_eps, gp_C = gp_corr_dim(ratio)
print(f"  D₂ ≈ {D2:.3f}")

# ── сводная таблица ───────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"{'Дата':>12} {'W_max':>6} {'d_local':>8}  p_fits")
for res in all_results:
    W_max   = res["scan"]["W_max"]
    d_local = res["scan"]["d_local"]
    pfs     = res["p_fits"]
    print(f"{res['date']:>12} {str(W_max):>6} {str(d_local):>8}  {pfs}")
print(f"\nGP D₂ ≈ {D2:.3f}  (m={GP_M}, {GP_MAX_PTS} точек)")

# ═══════════════════════════════════════════════════════════════════════════════
# ФИГУРА 1: W-sweep диагностика
# ═══════════════════════════════════════════════════════════════════════════════
n_orig = len(all_results)
fig1 = plt.figure(figsize=(5 * n_orig, 13))
gs1  = gridspec.GridSpec(4, n_orig, hspace=0.5, wspace=0.35)
fig1.suptitle(
    f"SBER 1d  |  Поиск W_max  |  P_search={P_SEARCH}  K={K_SEARCH}  "
    f"LP={N_LP_ITER}iter  d@{int(D_THRESH*100)}%  |  P_ref=2·d_after+1 (адапт.)",
    fontsize=12, fontweight="bold"
)

for col, res in enumerate(all_results):
    hist = res["scan"]["history"]
    # Только строки с данными
    valid_h = [h for h in hist if not np.isnan(h["acf1"])]
    ws      = [h["W"]        for h in valid_h]
    d_bef   = [h["d_before"] for h in valid_h]
    d_aft   = [h["d_after"]  for h in valid_h]
    acfs    = [h["acf1"]     for h in valid_h]
    oks     = [h["ok"]       for h in valid_h]
    W_max   = res["scan"]["W_max"]
    d_local = res["scan"]["d_local"]
    p_fits  = res["p_fits"]
    pA      = p_fits[0] if p_fits else None

    # ── Row 0: d_eff before / after LP vs W ──────────────────────────────────
    ax = fig1.add_subplot(gs1[0, col])
    ax.plot(ws, d_bef, "o--", color="#aaaaaa", lw=1.3, ms=5,
            label="d_eff (до LP)")
    ax.plot(ws, d_aft, "s-",  color="steelblue", lw=2.2, ms=6,
            label="d_eff (после LP)")
    if W_max:
        ax.axvline(W_max, color="green", ls="--", lw=1.8, label=f"W_max={W_max}")
        ax.text(0.97, 0.95, f"d_local={d_local}\np_fit={pA}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="green", fontweight="bold",
                bbox=dict(fc="honeydew", ec="green", alpha=0.85))
    ax.set_title(f"{res['date']}", fontsize=10, fontweight="bold")
    ax.set_ylabel("d_eff"); ax.set_xlabel("W")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3)

    # ── Row 1: |ACF lag-1| vs W + адаптивная CI95 ────────────────────────────
    ax = fig1.add_subplot(gs1[1, col])
    ci95s   = [h["CI95"] for h in valid_h]
    bar_colors = ["#2ca02c" if ok else "#d62728" for ok in oks]
    ax.bar(ws, acfs, color=bar_colors, alpha=0.75, width=W_STEP * 0.8,
           label="|ACF lag-1|  (зел=✓, красн=✗)")
    # Адаптивная CI95 как ломаная
    if ci95s:
        ax.step(ws, ci95s, where="mid", color="black", ls="--", lw=1.5,
                label="CI95 (адапт.)")
    if W_max:
        ax.axvline(W_max, color="green", ls="--", lw=1.8)
    ax.set_ylabel("|ACF lag-1|"); ax.set_xlabel("W (Theiler)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(bottom=0)

    # ── Row 2: распределение возраста пула ───────────────────────────────────
    ax = fig1.add_subplot(gs1[2, col])
    ok_h  = [h for h in valid_h if h["ok"]  and h["pool_times"] is not None]
    bad_h = [h for h in valid_h if not h["ok"] and h["pool_times"] is not None]
    if ok_h:
        best = max(ok_h, key=lambda h: h["W"])
        ages = t_origins[col] - best["pool_times"]
        ax.hist(ages, bins=20, color="steelblue", alpha=0.7, density=True,
                label=f"W={best['W']} ✓ (W_max)")
    if bad_h:
        worst = min(bad_h, key=lambda h: h["W"])
        ages2 = t_origins[col] - worst["pool_times"]
        ax.hist(ages2, bins=20, color="salmon", alpha=0.55, density=True,
                label=f"W={worst['W']} ✗ (первый сбой)")
    ax.set_xlabel("Возраст соседа (баров назад)"); ax.set_ylabel("Плотность")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── Row 3: Scree-plot при W_max (до / после LP) ──────────────────────────
    ax = fig1.add_subplot(gs1[3, col])
    best_h = None
    if W_max:
        bests = [h for h in valid_h if h["W"] == W_max]
        if bests:
            best_h = bests[0]
    if best_h and best_h["cv_before"] is not None:
        n_comp = min(20, len(best_h["cv_before"]))
        ax.plot(range(1, n_comp + 1), best_h["cv_before"][:n_comp],
                "o--", color="#aaaaaa", lw=1.3, ms=4, label="до LP")
        ax.plot(range(1, n_comp + 1), best_h["cv_after"][:n_comp],
                "s-", color="steelblue", lw=2, ms=5, label="после LP")
        ax.axvline(best_h["d_before"], color="#888888", ls=":", lw=1.2)
        ax.axvline(best_h["d_after"],  color="steelblue", ls=":", lw=1.2,
                   label=f"d={best_h['d_after']}")
        ax.axhline(D_THRESH, color="black", ls="--", lw=1, alpha=0.5,
                   label=f"{int(D_THRESH*100)}%")
    ax.set_xlabel("Компонент"); ax.set_ylabel("Cumvar")
    ax.set_title(f"Scree при W_max={W_max}", fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

fig1.savefig("research/phase5_attractor/lp_wmax_sweep.png", dpi=130)
print("\n  → lp_wmax_sweep.png")

# ═══════════════════════════════════════════════════════════════════════════════
# ФИГУРА 2: прогнозы цены (2 строки × N_ORIGINS столбцов)
#   Строка 0: без LP-коррекции    Строка 1: с LP-коррекцией
#   Цвета: синий=2d+1, оранжевый=2d+2, зелёный=2d+3
# ═══════════════════════════════════════════════════════════════════════════════
MARGIN_COLORS = ["steelblue", "darkorange", "green"]
MARGIN_STYLES = ["-", "--", ":"]

fig2 = plt.figure(figsize=(8 * n_orig, 13))
gs2  = gridspec.GridSpec(2, n_orig, hspace=0.4, wspace=0.3)
fig2.suptitle(
    f"SBER 1d  |  Прогноз: p_fit = 2·d_local + margin (без каскада)  |  "
    f"H={H_FORECAST}  acc_ang λ={ACC_LAMBDA}  K_corr={K_CORR}",
    fontsize=12, fontweight="bold"
)

for col, res in enumerate(all_results):
    t_orig  = res["t_orig"]
    W_max   = res["scan"]["W_max"]
    d_local = res["scan"]["d_local"]
    t_lo    = max(0, t_orig - HISTORY_SHOW)
    t_fut   = np.arange(t_orig + 1, min(N, t_orig + H_FORECAST + 1))

    for row, lp in enumerate([False, True]):
        ax = fig2.add_subplot(gs2[row, col])

        ax.plot(np.arange(t_lo, t_orig + 1), close[t_lo: t_orig + 1],
                color="black", lw=1.8, label="факт", zorder=5)
        ax.axvline(t_orig, color="navy", ls="--", lw=1.1, alpha=0.6)
        if len(t_fut):
            ax.plot(t_fut, close[t_fut], color="black", lw=1.8, ls=":",
                    alpha=0.4, label="факт (буд.)", zorder=5)

        if W_max is None:
            ax.text(0.5, 0.5, "W_max не найден\n(origin пропущен)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="darkred",
                    bbox=dict(fc="mistyrose", ec="darkred", alpha=0.9))
        else:
            for m, color, ls in zip(MARGINS, MARGIN_COLORS, MARGIN_STYLES):
                p_fit = 2 * d_local + m
                fc    = res["forecasts"].get((p_fit, lp))
                if fc is None or fc[1] is None:
                    continue
                tp = np.arange(t_orig + 1, t_orig + 1 + len(fc[1]))
                ax.plot(tp, fc[1], color=color, lw=2.0, ls=ls, zorder=6,
                        label=f"p={p_fit} (2d+{m})")

        lp_tag = "с LP-коррекцией" if lp else "без LP-коррекции"
        title = (f"{res['date']}  W={W_max} d={d_local}  {lp_tag}"
                 if W_max else f"{res['date']}  —  пропущен")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("бар"); ax.set_ylabel("цена, руб.")
        ax.legend(fontsize=7, loc="best", framealpha=0.85)
        ax.grid(True, alpha=0.3)

fig2.savefig("research/phase5_attractor/lp_wmax_forecast.png", dpi=130)
print("  → lp_wmax_forecast.png")

# ═══════════════════════════════════════════════════════════════════════════════
# ФИГУРА 3: GP корреляционная размерность
# ═══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
fig3.suptitle(
    f"GP корреляционная размерность  |  SBER 1d  |  m={GP_M}  |  D₂ ≈ {D2:.3f}",
    fontsize=12, fontweight="bold"
)

# Panel A: ln C(ε) vs ln ε
ax = axes3[0]
valid_gp = (gp_C > 0) & (gp_C < 1)
ax.plot(np.log(gp_eps), np.log(np.maximum(gp_C, 1e-10)),
        "o-", color="steelblue", lw=1.8, ms=5, label="ln C(ε)")
if valid_gp.sum() >= 4:
    loge = np.log(gp_eps[valid_gp]); logC = np.log(gp_C[valid_gp])
    slope, intercept = np.polyfit(loge, logC, 1)
    ax.plot(loge, slope * loge + intercept, "r--", lw=2.0,
            label=f"наклон D₂ = {slope:.3f}")
ax.set_xlabel("ln ε"); ax.set_ylabel("ln C(ε)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.set_title("Корреляционный интеграл (log-log)")

# Panel B: C(ε) vs ε (линейный)
ax = axes3[1]
ax.plot(gp_eps, gp_C, "o-", color="steelblue", lw=1.8, ms=5)
ax.set_xlabel("ε"); ax.set_ylabel("C(ε)")
ax.set_title(f"C(ε) — доля пар с дистанцией < ε\n(D₂≈{D2:.3f},  m={GP_M},  ratio)")
ax.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig("research/phase5_attractor/lp_wmax_gp_dim.png", dpi=130)
print("  → lp_wmax_gp_dim.png")

plt.show()
print("\nГотово.")
