#!/usr/bin/env python3
"""
zigzag_bias_trend.py

Связь смещения 4-way ансамбля с долгосрочным трендом.

Для каждого шага вычисляем:
  pos_rel  = (p_cur − MA[bar]) / MA[bar]   — положение пивота относительно MA
  slope_MA = (MA[bar] − MA[bar−S]) / MA[bar−S]  — наклон MA (S=10 баров)

MA вычисляется каузально на mid=(high+low)/2 сырых 1d баров,
W_MA ∈ {20, 50, 100, 200}.

Анализ:
  1. Корреляция pos_rel / slope_MA со знаковой ошибкой (per type и global)
  2. Квантильные корзины (5 шт.) по pos_rel → mean(err) per bucket
  3. Условный rMAE: uptrend vs downtrend, + 4 комбинации (тренд × тип цели)

4-way: α_lwr=0.05, α_sx=0.35, α_sm=0.20, α_rbf=0.40
SBER 1d(4%) + 10m(0.4%), H=1.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT      = BASE_DIR / "results"
OUT.mkdir(exist_ok=True)

T_1D        = 0.04
T_10M       = 0.004
H           = 1
MIN_HISTORY = 50

P_LWR  = 3;  K_LWR = 50
P_SX   = 8;  K_SX  = P_SX + 1
THETA  = 1.0
P_RBF  = 3;  K_RBF = 12

A_LWR = 0.05; A_SX = 0.35; A_SM = 0.20; A_RBF = 0.40

W_MA_GRID = [20, 50, 100, 200]
SLOPE_LAG  = 10   # баров для наклона MA


# ─────────────────────────────────────────────────────────────────────────────

def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    """Возвращает (vals, dates, types, bar_idxs).
    types: 1=HIGH, -1=LOW. bar_idxs: индекс бара в исходном массиве."""
    vals, dts, ptypes, bidxs = [], [], [], []
    direction = 0
    ext_val = (highs[0] + lows[0]) / 2.0
    ext_idx = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1;  ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val, ext_idx = lows[i], i
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                ptypes.append(1);     bidxs.append(ext_idx)
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                ptypes.append(-1);    bidxs.append(ext_idx)
                direction = 1; ext_val, ext_idx = highs[i], i
    if not vals:
        return np.array([]), np.array([]), np.array([], int), np.array([], int)
    return (np.array(vals), np.array(dts),
            np.array(ptypes, int), np.array(bidxs, int))


def build_X(prices, p):
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def causal_ma(series, W):
    """Каузальная MA: ma[i] = mean(series[max(0,i−W+1):i+1])."""
    cs = np.concatenate([[0.0], np.cumsum(series)])
    counts = np.minimum(np.arange(1, len(series) + 1), W)
    start  = np.maximum(0, np.arange(len(series)) - W + 1)
    return (cs[np.arange(1, len(series) + 1)] - cs[start]) / counts


def rmae_full(preds, actuals):
    dz = np.mean(np.abs(np.diff(actuals)))
    return float(np.mean(np.abs(preds - actuals)) / dz) if dz > 1e-12 else np.nan


def spearman_r(x, y):
    r, p = stats.spearmanr(x, y)
    return float(r), float(p)


# ─────────────────────────────────────────────────────────────────────────────

def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d,  types1d, bidxs1d = find_pivots(h1d, l1d, d1d, T_1D)
    p10m, dt10m, _,       _        = find_pivots(h10m, l10m, d10m, T_10M)

    n1d = len(p1d); na = len(p10m)
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")

    Xs = {p: (build_X(p1d, p), build_X(p10m, p))
          for p in set([P_LWR, P_SX, P_RBF])}

    # ── Предвычислить каузальные MA на сырых 1d барах ────────────────────────
    mid1d = (h1d + l1d) / 2.0
    ma_arrays  = {W: causal_ma(mid1d, W) for W in W_MA_GRID}
    slp_arrays = {}
    for W, ma in ma_arrays.items():
        slp = np.zeros(len(ma))
        for i in range(SLOPE_LAG, len(ma)):
            slp[i] = (ma[i] - ma[i - SLOPE_LAG]) / (ma[i - SLOPE_LAG] + 1e-12)
        slp_arrays[W] = slp

    print(f"SBER 1d {n1d} пив  10m {na} пив  |  raw bars: {len(mid1d)}")
    print(f"W_MA={W_MA_GRID}  slope_lag={SLOPE_LAG}\n")

    # ── Прогон ансамбля ───────────────────────────────────────────────────────
    raw_preds    = []
    raw_actuals  = []
    raw_origins  = []   # p_cur (цена origin-пивота) для каждого шага
    raw_types    = []
    raw_bar_idxs = []   # bar index в 1d массиве для каждого шага
    raw_pool_sz  = []   # размер пула (p=3) на каждом шаге
    raw_step_seq = []   # номер step в 1d-пивотном ряду

    for step in range(MIN_HISTORY, n1d - H):
        skip = any(np.any(np.isnan(Xs[p][0][step])) for p in [P_LWR, P_SX, P_RBF])
        if skip:
            continue

        ce    = int(ce10m_all[step])
        p_cur = float(p1d[step])
        ttype = int(types1d[step + H])

        def make_pool(p):
            X1d_, X10m_ = Xs[p]
            j1d = np.arange(p - 1, step)
            v1d = ~np.any(np.isnan(X1d_[j1d]), axis=1) & (j1d + H < n1d)
            idx1 = j1d[v1d]
            Xp     = list(X1d_[idx1])
            yp_abs = list(p1d[idx1 + H])
            yp_src = list(p1d[idx1])
            j10 = np.arange(p - 1, min(ce, na - H))
            if len(j10):
                v10   = ~np.any(np.isnan(X10m_[j10]), axis=1)
                idx10 = j10[v10]
                Xp.extend(X10m_[idx10])
                yp_abs.extend(p10m[idx10 + H])
                yp_src.extend(p10m[idx10])
            if len(Xp) < p + 2:
                return None
            X_pool = np.array(Xp)
            y_abs  = np.array(yp_abs, dtype=float)
            y_lr   = np.log(y_abs / np.array(yp_src, dtype=float))
            x_q    = Xs[p][0][step]
            mu  = X_pool.mean(0)
            sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
            Xn  = (X_pool - mu) / sig
            xn  = (x_q    - mu) / sig
            d   = np.linalg.norm(Xn - xn, axis=1)
            return Xn, xn, d, y_abs, y_lr, X_pool.shape[0]

        res3 = make_pool(P_LWR)
        y_lwr = y_smap = np.nan
        if res3:
            Xn, xn, d, y_abs, y_lr, N = res3
            ord_ = np.argsort(d)
            k_eff = min(K_LWR, N); knn = ord_[:k_eff]; xi = d[ord_[k_eff-1]]
            if xi < 1e-12:
                y_lwr = float(y_abs[knn].mean())
            else:
                w = np.exp(-0.5*(d[knn]/xi)**2); ws = np.sqrt(w)
                A_ = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:,None]
                c, *_ = np.linalg.lstsq(A_, y_abs[knn]*ws, rcond=None)
                y_lwr = float(c[0] + c[1:] @ xn)
            mean_d = d.mean() + 1e-12
            w_sm   = np.exp(-THETA * d / mean_d); ws_sm = np.sqrt(w_sm)
            A_sm   = np.column_stack([np.ones(N), Xn]) * ws_sm[:,None]
            c_sm, *_ = np.linalg.lstsq(A_sm, y_abs*ws_sm, rcond=None)
            y_smap = float(c_sm[0] + c_sm[1:] @ xn)

        res8 = make_pool(P_SX)
        y_sx = np.nan
        if res8 and res8[5] >= K_SX:
            Xn, xn, d, y_abs, y_lr, N = res8
            ords = np.argsort(d); knn = ords[:K_SX]; d1 = d[ords[0]]
            if d1 < 1e-12:
                y_sx = float(p_cur * np.exp(y_lr[ords[0]]))
            else:
                w = np.exp(-d[knn]/d1); w /= w.sum()
                y_sx = float(p_cur * np.exp(w @ y_lr[knn]))

        y_rbf = np.nan
        if res3 and res3[5] >= K_RBF:
            Xn, xn, d, y_abs, y_lr, N = res3
            ord_ = np.argsort(d); k = min(K_RBF, N); knn = ord_[:k]
            xi = d[ord_[k-1]]
            if xi < 1e-12:
                y_rbf = float(p_cur * np.exp(y_lr[knn].mean()))
            else:
                w = np.exp(-0.5*(d[knn]/xi)**2)
                y_rbf = float(p_cur * np.exp((w @ y_lr[knn]) / w.sum()))

        pred_ens = A_LWR*y_lwr + A_SX*y_sx + A_SM*y_smap + A_RBF*y_rbf
        if np.isnan(pred_ens):
            continue

        pool_sz = res3[5] if res3 else 0   # размер пула для p=3

        raw_preds.append(pred_ens)
        raw_actuals.append(float(p1d[step + H]))
        raw_origins.append(p_cur)
        raw_types.append(ttype)
        raw_bar_idxs.append(int(bidxs1d[step]))
        raw_pool_sz.append(pool_sz)
        raw_step_seq.append(step)

    acts     = np.array(raw_actuals)
    preds    = np.array(raw_preds)
    p_curs   = np.array(raw_origins)
    types    = np.array(raw_types)
    bar_idxs = np.array(raw_bar_idxs)
    pool_szs = np.array(raw_pool_sz)
    step_seq = np.array(raw_step_seq)
    errs     = preds - acts
    n_steps  = len(acts)
    dz       = float(np.mean(np.abs(np.diff(acts))))

    print(f"Шагов: {n_steps}")
    print(f"Глобальный rMAE (4-way): {rmae_full(preds, acts):.4f}\n")

    # ── АНАЛИЗ ────────────────────────────────────────────────────────────────
    print(f"{'═'*72}")
    print("СЕКЦИЯ 1: Корреляция ошибки с pos_rel и slope_MA")
    print(f"{'═'*72}")
    hdr = f"{'W_MA':>5}  {'Spearman r (pos_rel)':>22}  {'Spearman r (slope)':>20}"
    print(f"\n{'':5}  {'─'*22}  {'─'*20}")
    print(f"{'W_MA':>5}  {'all':>6}  {'HIGH':>6}  {'LOW':>6}"
          f"  {'all':>6}  {'HIGH':>6}  {'LOW':>6}")
    print(f"{'─'*55}")

    all_pos_rel  = {}
    all_slope    = {}

    for W in W_MA_GRID:
        ma  = ma_arrays[W]
        slp = slp_arrays[W]
        pos_rel = (p_curs - ma[bar_idxs]) / (ma[bar_idxs] + 1e-12)
        slope   = slp[bar_idxs]

        all_pos_rel[W] = pos_rel
        all_slope[W]   = slope

        r_pr_all, p_pr_all  = spearman_r(pos_rel, errs)
        r_sl_all, p_sl_all  = spearman_r(slope,   errs)

        mask_h = types == 1; mask_l = types == -1
        r_pr_h, _ = spearman_r(pos_rel[mask_h], errs[mask_h])
        r_pr_l, _ = spearman_r(pos_rel[mask_l], errs[mask_l])
        r_sl_h, _ = spearman_r(slope[mask_h],   errs[mask_h])
        r_sl_l, _ = spearman_r(slope[mask_l],   errs[mask_l])

        def fmt(r, p):
            star = "*" if p < 0.05 else " "
            return f"{r:+.3f}{star}"

        print(f"{W:>5}  {fmt(r_pr_all,p_pr_all):>7}  {fmt(r_pr_h,0):>6}  {fmt(r_pr_l,0):>6}"
              f"  {fmt(r_sl_all,p_sl_all):>7}  {fmt(r_sl_h,0):>6}  {fmt(r_sl_l,0):>6}")

    print("\n(* p<0.05 для глобальной корреляции; per-type p-значения не выводятся)")

    # ── СЕКЦИЯ 2: Квантильные корзины ────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("СЕКЦИЯ 2: Mean(error) по квинтилям pos_rel  (W_MA=100)")
    print(f"{'═'*72}")
    W_show = 100
    pos_rel_100 = all_pos_rel[W_show]
    quintiles = np.quantile(pos_rel_100, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    labels_q   = [f"[{quintiles[i]*100:.1f}%..{quintiles[i+1]*100:.1f}%)"
                  for i in range(5)]
    print(f"\n{'Корзина':>26}  {'n':>4}  {'mean err ALL':>13}  "
          f"{'mean err HIGH':>14}  {'mean err LOW':>13}")
    for qi in range(5):
        lo, hi = quintiles[qi], quintiles[qi+1]
        m_all  = (pos_rel_100 >= lo) & (pos_rel_100 <= hi)
        m_h    = m_all & (types == 1)
        m_l    = m_all & (types == -1)
        n_b = m_all.sum()
        e_all = np.mean(errs[m_all]) if n_b > 0 else np.nan
        e_h   = np.mean(errs[m_h])   if m_h.sum() > 0 else np.nan
        e_l   = np.mean(errs[m_l])   if m_l.sum() > 0 else np.nan
        print(f"{labels_q[qi]:>26}  {n_b:>4}  {e_all:>+12.2f}  "
              f"{e_h:>+13.2f}  {e_l:>+12.2f}")

    # ── СЕКЦИЯ 3: Условный rMAE ───────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("СЕКЦИЯ 3: Условный rMAE (uptrend vs downtrend) — W_MA=100")
    print(f"{'═'*72}")
    up_mask   = pos_rel_100 >= 0
    down_mask = pos_rel_100 <  0
    combos = [
        ("uptrend  (p_cur > MA)", up_mask),
        ("downtrend (p_cur < MA)", down_mask),
        ("uptrend  + HIGH-target", up_mask   & (types == 1)),
        ("uptrend  + LOW-target",  up_mask   & (types == -1)),
        ("downtrend + HIGH-target", down_mask & (types == 1)),
        ("downtrend + LOW-target",  down_mask & (types == -1)),
    ]
    print(f"\n{'Условие':<30}  {'n':>4}  {'rMAE':>7}  {'mean_err':>9}")
    for label, mask in combos:
        n_c = mask.sum()
        if n_c < 5:
            print(f"{label:<30}  {n_c:>4}  {'—':>7}")
            continue
        r = float(np.mean(np.abs(errs[mask])) / dz)
        me = float(np.mean(errs[mask]))
        print(f"{label:<30}  {n_c:>4}  {r:>7.4f}  {me:>+8.2f}")

    print(f"\nСправка: global rMAE = {rmae_full(preds, acts):.4f}")

    # ── СЕКЦИЯ 4: rMAE по времени и по размеру пула ───────────────────────────
    print(f"\n{'═'*72}")
    print("СЕКЦИЯ 4: Эффект малого пула — rMAE по шагам и по размеру пула")
    print(f"{'═'*72}")

    # 4a. Бины по step (10 равных по количеству бинов)
    N_BINS = 10
    bin_edges = np.percentile(step_seq, np.linspace(0, 100, N_BINS + 1))
    bin_edges[-1] += 1  # включить последний шаг
    print(f"\nrMAE по хронологическим бинам (каждый ~{n_steps//N_BINS} шагов):")
    print(f"  {'Шаги (пивоты)':>18}  {'n':>4}  {'rMAE':>7}  "
          f"{'mean|pool|':>10}  {'min pool':>8}")
    for i in range(N_BINS):
        m = (step_seq >= bin_edges[i]) & (step_seq < bin_edges[i+1])
        if m.sum() < 3:
            continue
        r = float(np.mean(np.abs(errs[m])) / dz)
        print(f"  {int(bin_edges[i]):>6}..{int(bin_edges[i+1])-1:<6}  "
              f"  {m.sum():>4}  {r:>7.4f}  "
              f"{np.mean(pool_szs[m]):>10.0f}  {np.min(pool_szs[m]):>8.0f}")

    # 4b. Бины по размеру пула
    pool_breaks = [0, 500, 1000, 2000, 5000, 10000, 100000]
    labels_pb   = ["<500", "500-1k", "1k-2k", "2k-5k", "5k-10k", "10k+"]
    print(f"\nrMAE по размеру пула (p=3):")
    print(f"  {'Пул':>8}  {'n':>4}  {'rMAE':>7}  {'mean step':>10}")
    for i, lbl in enumerate(labels_pb):
        m = (pool_szs >= pool_breaks[i]) & (pool_szs < pool_breaks[i+1])
        if m.sum() < 3:
            continue
        r = float(np.mean(np.abs(errs[m])) / dz)
        print(f"  {lbl:>8}  {m.sum():>4}  {r:>7.4f}  "
              f"{np.mean(step_seq[m]):>10.1f}")

    # 4c. Порог MIN_HISTORY: rMAE при исключении первых N шагов
    print(f"\nrMAE при увеличении MIN_HISTORY (исключение первых N шагов):")
    print(f"  {'Отброшено':>10}  {'Осталось':>9}  {'rMAE':>7}")
    for skip in [0, 50, 100, 200, 300, 400]:
        m = step_seq >= step_seq[0] + skip
        if m.sum() < 10:
            break
        r = float(np.mean(np.abs(errs[m])) / dz)
        print(f"  {skip:>10}  {m.sum():>9}  {r:>7.4f}")

    # ── ГРАФИКИ ───────────────────────────────────────────────────────────────

    # Рис. 1: scatter pos_rel vs error для каждого W_MA
    fig1, axes1 = plt.subplots(2, 2, figsize=(13, 10))
    for ax, W in zip(axes1.flat, W_MA_GRID):
        pr = all_pos_rel[W] * 100  # в %
        for ttype, color, marker, lbl in [(1, "tomato", "^", "HIGH"),
                                           (-1, "steelblue", "v", "LOW")]:
            m = types == ttype
            ax.scatter(pr[m], errs[m], alpha=0.2, s=8,
                       color=color, marker=marker, label=lbl)
            # regression line
            slope_lr, intercept_lr, *_ = stats.linregress(pr[m], errs[m])
            x_fit = np.linspace(pr.min(), pr.max(), 100)
            ax.plot(x_fit, slope_lr*x_fit + intercept_lr,
                    color=color, lw=1.5, alpha=0.8)

        r_all, p_all = spearman_r(pr, errs)
        ax.axhline(0, color="black", lw=0.7, ls=":")
        ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.5)
        ax.set_xlabel(f"pos_rel = (p_cur−MA{W})/MA{W} (%)")
        ax.set_ylabel("pred − actual (руб.)")
        ax.set_title(f"W_MA={W}  |  Spearman r={r_all:+.3f}  p={p_all:.3f}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig1.suptitle(
        f"Ошибка ансамбля vs положение относительно MA | SBER 1d+10m H={H}  n={n_steps}",
        fontsize=11,
    )
    fig1.tight_layout()
    fig1.savefig(OUT / "trend_scatter.png", dpi=130, bbox_inches="tight")
    plt.close(fig1)

    # Рис. 2: bucket analysis + условный rMAE (все W_MA)
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))
    for ax, W in zip(axes2.flat, W_MA_GRID):
        pr = all_pos_rel[W] * 100
        quants = np.quantile(pr, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        centers = (quants[:-1] + quants[1:]) / 2
        bucket_err_h = []
        bucket_err_l = []
        bucket_err_a = []
        for qi in range(5):
            lo_q, hi_q = quants[qi], quants[qi+1]
            m_b = (pr >= lo_q) & (pr <= hi_q)
            bucket_err_a.append(np.mean(errs[m_b])           if m_b.sum() > 0 else np.nan)
            bucket_err_h.append(np.mean(errs[m_b & (types==1)])  if (m_b & (types==1)).sum()>0 else np.nan)
            bucket_err_l.append(np.mean(errs[m_b & (types==-1)]) if (m_b & (types==-1)).sum()>0 else np.nan)

        width = (centers[1] - centers[0]) * 0.25
        ax.bar(centers - width, bucket_err_h, width=width*1.8,
               color="tomato", alpha=0.7, label="HIGH-target")
        ax.bar(centers + width, bucket_err_l, width=width*1.8,
               color="steelblue", alpha=0.7, label="LOW-target")
        ax.plot(centers, bucket_err_a, "ko-", ms=5, lw=1.2, label="all")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.axvline(0, color="black", lw=0.7, ls="--", alpha=0.4)
        ax.set_xlabel(f"pos_rel квинтиль, MA{W} (%)")
        ax.set_ylabel("mean(pred − actual)")
        ax.set_title(f"W_MA={W}: mean error по квинтилям pos_rel")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    fig2.suptitle(
        f"Bucket-анализ: зависимость смещения от тренда | SBER 1d+10m",
        fontsize=11,
    )
    fig2.tight_layout()
    fig2.savefig(OUT / "trend_buckets.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)

    # Рис. 3: rMAE по шагам + размер пула
    fig3, (ax_r, ax_p) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # Rolling rMAE (окно 50 шагов)
    W_roll = 50
    roll_rmae = []
    roll_x    = []
    for i in range(W_roll, n_steps):
        sl = slice(i - W_roll, i)
        r  = float(np.mean(np.abs(errs[sl])) / dz)
        roll_rmae.append(r)
        roll_x.append(step_seq[i])
    ax_r.plot(roll_x, roll_rmae, color="steelblue", lw=1.2,
              label=f"rolling rMAE (W={W_roll})")
    ax_r.axhline(rmae_full(preds, acts), color="red", ls="--", lw=1.0,
                 label=f"global rMAE={rmae_full(preds, acts):.4f}")
    ax_r.set_ylabel("rMAE"); ax_r.set_xlabel("step (пивот 1d)")
    ax_r.set_title("Rolling rMAE по времени")
    ax_r.legend(fontsize=8); ax_r.grid(True, alpha=0.3)

    # Размер пула vs шаг
    ax_p.scatter(step_seq, pool_szs, alpha=0.3, s=5, color="darkorange")
    ax_p.axhline(K_LWR, color="steelblue", ls=":", lw=1.0,
                 label=f"K_LWR={K_LWR}")
    ax_p.set_ylabel("размер пула (p=3)"); ax_p.set_xlabel("step (пивот 1d)")
    ax_p.set_title("Размер пула по времени")
    ax_p.legend(fontsize=8); ax_p.grid(True, alpha=0.3)
    ax_p.set_yscale("log")

    fig3.suptitle("Эффект малого пула | SBER 1d+10m", fontsize=11)
    fig3.tight_layout()
    fig3.savefig(OUT / "pool_size_effect.png", dpi=130, bbox_inches="tight")
    plt.close(fig3)

    print(f"\nГрафики → {OUT}/trend_scatter.png")
    print(f"          {OUT}/trend_buckets.png")
    print(f"          {OUT}/pool_size_effect.png")


if __name__ == "__main__":
    run()
