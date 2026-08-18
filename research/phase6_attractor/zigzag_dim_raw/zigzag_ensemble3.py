#!/usr/bin/env python3
"""
3-компонентный ансамбль: LWR-abs + Simplex-log + S-map.

Зафиксированы лучшие параметры:
  LWR-abs:     p=3, K=50
  Simplex-log: p=8, k=9

Свип θ для S-map + свип весов ансамбля.
  Для каждого θ: ŷ = α_lwr·LWR + α_sx·Simplex + α_sm·S-map(θ)
  α_lwr + α_sx + α_sm = 1,  шаг 0.05

Эталоны:
  LWR-abs p=3 K=50            → 0.4194
  Ens(LWR+Smap) α=0.65        → 0.4143
  Ens(LWR+Sx-log p=8) α=0.55  → 0.4055

SBER 1d(4%) + 10m(0.4%), H=1.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT      = BASE_DIR / "results"

T_1D        = 0.04
T_10M       = 0.004
H           = 1
MIN_HISTORY = 50

P_LWR      = 3
K_LWR      = 50
P_SX       = 8
K_SX       = P_SX + 1
THETA_GRID = [1, 3, 5, 8, 10, 15, 20, 30]
STEP       = 0.05

REF_LWR    = 0.4194
REF_ENS2   = 0.4143
REF_ENS_SX = 0.4055


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    vals, dts = [], []
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
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                direction = 1; ext_val, ext_idx = highs[i], i
    if not vals:
        return np.array([]), np.array([])
    return np.array(vals), np.array(dts)


def build_X(prices, p):
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def rmae(preds, actuals):
    preds = np.asarray(preds, dtype=float)
    valid = ~np.isnan(preds)
    if valid.sum() < 5:
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return float(np.mean(np.abs(preds[valid] - actuals[valid])) / dz) if dz > 1e-12 else np.nan


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m = find_pivots(h10m, l10m, d10m, T_10M)

    n1d = len(p1d); na = len(p10m)
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")

    X1d_lwr  = build_X(p1d,  P_LWR)
    X10m_lwr = build_X(p10m, P_LWR)
    X1d_sx   = build_X(p1d,  P_SX)
    X10m_sx  = build_X(p10m, P_SX)

    print(f"SBER 1d {n1d} пив  10m {na} пив")
    print(f"LWR p={P_LWR} K={K_LWR}  |  Sx-log p={P_SX} k={K_SX}  |  S-map θ={THETA_GRID}")
    print(f"Эталоны: LWR={REF_LWR:.4f}  Ens(+Smap)={REF_ENS2:.4f}  "
          f"Ens(+Sx)={REF_ENS_SX:.4f}\n")

    # ── Walk-forward: собираем предсказания для LWR, Sx-log и S-map(θ) ────────
    pred_lwr  = []
    pred_sx   = []
    pred_smap = {th: [] for th in THETA_GRID}
    actuals   = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X1d_lwr[step])) or np.any(np.isnan(X1d_sx[step])):
            continue

        ce    = int(ce10m_all[step])
        p_cur = float(p1d[step])

        # ── Пул LWR / S-map (p=3) ────────────────────────────────────────────
        j1d = np.arange(P_LWR - 1, step)
        v1d = ~np.any(np.isnan(X1d_lwr[j1d]), axis=1) & (j1d + H < n1d)
        idx1 = j1d[v1d]
        Xp3  = list(X1d_lwr[idx1]);  yp3 = list(p1d[idx1 + H])

        j10 = np.arange(P_LWR - 1, min(ce, na - H))
        if len(j10):
            v10 = ~np.any(np.isnan(X10m_lwr[j10]), axis=1)
            i10 = j10[v10]
            Xp3.extend(X10m_lwr[i10]);  yp3.extend(p10m[i10 + H])

        y_lwr = np.nan
        smap_vals = {th: np.nan for th in THETA_GRID}

        if len(Xp3) >= P_LWR + 2:
            X_pool = np.array(Xp3); y_abs = np.array(yp3)
            x_q    = X1d_lwr[step]
            mu  = X_pool.mean(0)
            sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
            Xn  = (X_pool - mu) / sig
            xn  = (x_q    - mu) / sig
            d   = np.linalg.norm(Xn - xn, axis=1)
            ord_ = np.argsort(d)
            N    = len(y_abs)

            # LWR K=50
            k_eff = min(K_LWR, N)
            knn = ord_[:k_eff]; xi = d[ord_[k_eff - 1]]
            if xi < 1e-12:
                y_lwr = float(y_abs[knn].mean())
            else:
                w  = np.exp(-0.5 * (d[knn] / xi) ** 2); ws = np.sqrt(w)
                A  = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
                c, *_ = np.linalg.lstsq(A, y_abs[knn] * ws, rcond=None)
                y_lwr = float(c[0] + c[1:] @ xn)

            # S-map для каждого θ
            mean_d = d.mean() + 1e-12
            for th in THETA_GRID:
                w_sm  = np.exp(-th * d / mean_d); ws_sm = np.sqrt(w_sm)
                A_sm  = np.column_stack([np.ones(N), Xn]) * ws_sm[:, None]
                c_sm, *_ = np.linalg.lstsq(A_sm, y_abs * ws_sm, rcond=None)
                smap_vals[th] = float(c_sm[0] + c_sm[1:] @ xn)

        pred_lwr.append(y_lwr)
        for th in THETA_GRID:
            pred_smap[th].append(smap_vals[th])

        # ── Simplex-log (p=8) ─────────────────────────────────────────────────
        j1s = np.arange(P_SX - 1, step)
        v1s = ~np.any(np.isnan(X1d_sx[j1s]), axis=1) & (j1s + H < n1d)
        idx1s = j1s[v1s]
        Xps   = list(X1d_sx[idx1s])
        yas   = list(p1d[idx1s + H]);  yss = list(p1d[idx1s])

        j10s = np.arange(P_SX - 1, min(ce, na - H))
        if len(j10s):
            v10s = ~np.any(np.isnan(X10m_sx[j10s]), axis=1)
            i10s = j10s[v10s]
            Xps.extend(X10m_sx[i10s])
            yas.extend(p10m[i10s + H]);  yss.extend(p10m[i10s])

        y_sx = np.nan
        if len(Xps) >= K_SX:
            X_ps  = np.array(Xps)
            ya_sx = np.array(yas, dtype=float)
            ylr   = np.log(ya_sx / np.array(yss, dtype=float))
            x_qs  = X1d_sx[step]
            mu_s  = X_ps.mean(0)
            sig_s = np.where(X_ps.std(0) < 1e-10, 1.0, X_ps.std(0))
            Xns   = (X_ps - mu_s) / sig_s
            xns   = (x_qs - mu_s) / sig_s
            ds    = np.linalg.norm(Xns - xns, axis=1)
            ords  = np.argsort(ds)
            knn_s = ords[:K_SX]; d1 = ds[ords[0]]
            if d1 < 1e-12:
                y_sx = float(p_cur * np.exp(ylr[ords[0]]))
            else:
                w_s  = np.exp(-ds[knn_s] / d1)
                ws_s = w_s / w_s.sum()
                y_sx = float(p_cur * np.exp(ws_s @ ylr[knn_s]))
        pred_sx.append(y_sx)

        actuals.append(float(p1d[step + H]))

    acts  = np.array(actuals)
    n     = len(acts)
    A_lwr = np.array(pred_lwr, dtype=float)
    A_sx  = np.array(pred_sx,  dtype=float)

    print(f"Шагов: {n}")
    print(f"Одиночные: LWR={rmae(A_lwr,acts):.4f}  Sx-log={rmae(A_sx,acts):.4f}")
    for th in THETA_GRID:
        print(f"  S-map θ={th:>2}: {rmae(np.array(pred_smap[th]),acts):.4f}")
    print()

    # ── Свип весов для каждого θ ──────────────────────────────────────────────
    alphas = np.round(np.arange(0.0, 1.0 + STEP / 2, STEP), 2)

    global_best = (np.inf, None, None)  # (rMAE, θ, (a_lwr, a_sx, a_sm))
    theta_bests = {}  # θ → (best_r, best_cfg)

    print(f"{'θ':>4}  {'α_lwr':>6}  {'α_sx':>5}  {'α_sm':>5}  "
          f"{'rMAE':>7}  {'vs_LWR%':>8}  {'vs_Sx%':>7}")
    print("-" * 56)

    for th in THETA_GRID:
        A_sm = np.array(pred_smap[th], dtype=float)
        best_r = np.inf; best_cfg = None

        for a_lwr in alphas:
            for a_sx in alphas:
                a_sm = round(1.0 - a_lwr - a_sx, 10)
                if a_sm < -1e-9:
                    continue
                a_sm = max(0.0, a_sm)
                blend = a_lwr * A_lwr + a_sx * A_sx + a_sm * A_sm
                r = rmae(blend, acts)
                if r < best_r:
                    best_r = r; best_cfg = (a_lwr, a_sx, round(a_sm, 2))

        theta_bests[th] = (best_r, best_cfg)
        a_lwr_b, a_sx_b, a_sm_b = best_cfg
        print(f"{th:>4}  {a_lwr_b:>6.2f}  {a_sx_b:>5.2f}  {a_sm_b:>5.2f}  "
              f"{best_r:>7.4f}  {(best_r/REF_LWR-1)*100:>+7.1f}%  "
              f"{(best_r/REF_ENS_SX-1)*100:>+6.1f}%")

        if best_r < global_best[0]:
            global_best = (best_r, th, best_cfg)

    best_r, best_th, best_cfg = global_best
    a_lwr_b, a_sx_b, a_sm_b = best_cfg
    print(f"\n{'='*56}")
    print(f"Глобальный оптимум: θ={best_th}  "
          f"α_lwr={a_lwr_b:.2f}  α_sx={a_sx_b:.2f}  α_sm={a_sm_b:.2f}")
    print(f"  rMAE={best_r:.4f}")
    print(f"  vs LWR-abs:          {(best_r/REF_LWR-1)*100:+.1f}%")
    print(f"  vs Ens(LWR+Smap):    {(best_r/REF_ENS2-1)*100:+.1f}%")
    print(f"  vs Ens(LWR+Sx-log):  {(best_r/REF_ENS_SX-1)*100:+.1f}%")

    # ── График: лучший rMAE по θ + кривая α_sx при лучшем θ ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Левый: лучшая rMAE по θ
    ax = axes[0]
    ths = THETA_GRID
    rs  = [theta_bests[th][0] for th in ths]
    ax.plot(ths, rs, "o-", color="purple", lw=1.8)
    ax.axhline(REF_LWR,    color="steelblue", ls=":",  lw=1.2,
               label=f"LWR={REF_LWR:.4f}")
    ax.axhline(REF_ENS2,   color="orange",    ls="--", lw=1.2,
               label=f"Ens(+Smap)={REF_ENS2:.4f}")
    ax.axhline(REF_ENS_SX, color="green",     ls="--", lw=1.2,
               label=f"Ens(+Sx)={REF_ENS_SX:.4f}")
    ax.axhline(best_r,     color="red",       ls="-",  lw=1.0,
               label=f"3-way best={best_r:.4f}")
    ax.scatter([best_th], [best_r], color="red", zorder=5, s=60)
    for th, r in zip(ths, rs):
        cfg = theta_bests[th][1]
        ax.annotate(f"({cfg[0]:.2f},{cfg[1]:.2f},{cfg[2]:.2f})",
                    (th, r), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color="purple")
    ax.set_xlabel("θ (S-map)")
    ax.set_ylabel("rMAE (лучший ансамбль)")
    ax.set_title("Лучший 3-way ансамбль vs θ\n(α_lwr, α_sx, α_sm) в аннотациях")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Правый: срез α_sx при лучшем θ, фиксируем a_lwr=best
    ax2 = axes[1]
    A_sm_b = np.array(pred_smap[best_th], dtype=float)
    a_lwr_fix = a_lwr_b
    sx_vals = alphas[alphas <= 1.0 - a_lwr_fix]
    r_slice = []
    for a_sx in sx_vals:
        a_sm = max(0.0, round(1.0 - a_lwr_fix - a_sx, 10))
        blend = a_lwr_fix * A_lwr + a_sx * A_sx + a_sm * A_sm_b
        r_slice.append(rmae(blend, acts))
    ax2.plot(sx_vals, r_slice, "o-", color="purple", lw=1.8)
    ax2.axhline(REF_LWR,    color="steelblue", ls=":",  lw=1.2,
                label=f"LWR={REF_LWR:.4f}")
    ax2.axhline(REF_ENS2,   color="orange",    ls="--", lw=1.2,
                label=f"Ens(+Smap)={REF_ENS2:.4f}")
    ax2.axhline(REF_ENS_SX, color="green",     ls="--", lw=1.2,
                label=f"Ens(+Sx)={REF_ENS_SX:.4f}")
    ax2.axhline(best_r,     color="red",       ls="-",  lw=1.0,
                label=f"3-way best={best_r:.4f}")
    ax2.scatter([a_sx_b], [best_r], color="red", zorder=5, s=60)
    ax2.set_xlabel(f"α_sx (Simplex-log)  |  α_lwr={a_lwr_fix:.2f} зафиксирован")
    ax2.set_ylabel("rMAE")
    ax2.set_title(f"Срез при α_lwr={a_lwr_fix:.2f}, θ={best_th}\n"
                  f"(α_sm = 1 − {a_lwr_fix:.2f} − α_sx)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"3-way ансамбль: LWR(p={P_LWR},K={K_LWR}) + Sx-log(p={P_SX}) + S-map(θ)\n"
        f"SBER 1d(4%)+10m(0.4%)  H={H}  n={n}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ensemble3.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {OUT}/ensemble3.png")


if __name__ == "__main__":
    run()
