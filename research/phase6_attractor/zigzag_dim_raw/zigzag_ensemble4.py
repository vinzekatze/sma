#!/usr/bin/env python3
"""
4-компонентный ансамбль: LWR-abs + Simplex-log + S-map + RBF-log.

Зафиксированы лучшие параметры:
  LWR-abs:     p=3, K=50
  Simplex-log: p=8, k=9
  S-map:       p=3, θ=1  (лучший в 3-way)
  RBF-log:     p=3, K=12

Веса: α_lwr + α_sx + α_sm + α_rbf = 1, αᵢ ≥ 0, шаг 0.05
Комбинаций: C(23,3) = 1771

Эталоны:
  LWR-abs                    → 0.4194
  Ens(LWR+Sx-log) α=0.55    → 0.4055
  3-way (LWR+Sx+Smap θ=1)   → 0.4004

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

P_LWR  = 3;  K_LWR = 50
P_SX   = 8;  K_SX  = P_SX + 1
THETA  = 1.0
P_RBF  = 3;  K_RBF = 12

REF_LWR   = 0.4194
REF_ENS2  = 0.4055
REF_3WAY  = 0.4004
STEP      = 0.05


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

    p_max = max(P_LWR, P_SX, P_RBF)
    Xs = {p: (build_X(p1d, p), build_X(p10m, p))
          for p in set([P_LWR, P_SX, P_RBF])}

    print(f"SBER 1d {n1d} пив  10m {na} пив")
    print(f"LWR(p={P_LWR},K={K_LWR})  Sx-log(p={P_SX},k={K_SX})  "
          f"S-map(θ={THETA})  RBF-log(p={P_RBF},K={K_RBF})")
    print(f"Эталоны: LWR={REF_LWR:.4f}  Ens2={REF_ENS2:.4f}  3-way={REF_3WAY:.4f}\n")

    pred_lwr  = []
    pred_sx   = []
    pred_smap = []
    pred_rbf  = []
    actuals   = []

    for step in range(MIN_HISTORY, n1d - H):
        skip = any(np.any(np.isnan(Xs[p][0][step])) for p in [P_LWR, P_SX, P_RBF])
        if skip:
            continue

        ce    = int(ce10m_all[step])
        p_cur = float(p1d[step])

        # ── Строим пулы для каждого p ─────────────────────────────────────────
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
                v10  = ~np.any(np.isnan(X10m_[j10]), axis=1)
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

        # ── LWR + S-map (общий пул p=3) ───────────────────────────────────────
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
                A = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:,None]
                c, *_ = np.linalg.lstsq(A, y_abs[knn]*ws, rcond=None)
                y_lwr = float(c[0] + c[1:] @ xn)
            mean_d = d.mean() + 1e-12
            w_sm   = np.exp(-THETA * d / mean_d); ws_sm = np.sqrt(w_sm)
            A_sm   = np.column_stack([np.ones(N), Xn]) * ws_sm[:,None]
            c_sm, *_ = np.linalg.lstsq(A_sm, y_abs*ws_sm, rcond=None)
            y_smap = float(c_sm[0] + c_sm[1:] @ xn)
        pred_lwr.append(y_lwr); pred_smap.append(y_smap)

        # ── Simplex-log (p=8) ─────────────────────────────────────────────────
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
        pred_sx.append(y_sx)

        # ── RBF-log (p=3, K=12) ───────────────────────────────────────────────
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
        pred_rbf.append(y_rbf)

        actuals.append(float(p1d[step + H]))

    acts  = np.array(actuals); n = len(acts)
    A_lwr = np.array(pred_lwr,  dtype=float)
    A_sx  = np.array(pred_sx,   dtype=float)
    A_sm  = np.array(pred_smap, dtype=float)
    A_rbf = np.array(pred_rbf,  dtype=float)

    print(f"Шагов: {n}")
    print(f"Одиночные: LWR={rmae(A_lwr,acts):.4f}  Sx={rmae(A_sx,acts):.4f}  "
          f"Smap={rmae(A_sm,acts):.4f}  RBF={rmae(A_rbf,acts):.4f}\n")

    # ── Полный перебор весов ──────────────────────────────────────────────────
    alphas = np.round(np.arange(0.0, 1.0 + STEP/2, STEP), 2)

    best_r   = np.inf
    best_cfg = None
    records  = []

    for a_lwr in alphas:
        for a_sx in alphas:
            rem2 = round(1.0 - a_lwr - a_sx, 10)
            if rem2 < -1e-9: continue
            for a_sm in alphas:
                a_rbf = round(rem2 - a_sm, 10)
                if a_rbf < -1e-9: continue
                a_rbf = max(0.0, a_rbf)
                blend = a_lwr*A_lwr + a_sx*A_sx + a_sm*A_sm + a_rbf*A_rbf
                r = rmae(blend, acts)
                records.append((r, a_lwr, a_sx, a_sm, a_rbf))
                if r < best_r:
                    best_r = r; best_cfg = (a_lwr, a_sx, a_sm, a_rbf)

    a_lwr_b, a_sx_b, a_sm_b, a_rbf_b = best_cfg
    print(f"Лучший 4-компонентный ансамбль:")
    print(f"  α_lwr={a_lwr_b:.2f}  α_sx={a_sx_b:.2f}  "
          f"α_sm={a_sm_b:.2f}  α_rbf={a_rbf_b:.2f}")
    print(f"  rMAE={best_r:.4f}")
    print(f"  vs LWR-abs:        {(best_r/REF_LWR-1)*100:+.1f}%")
    print(f"  vs Ens2(LWR+Sx):   {(best_r/REF_ENS2-1)*100:+.1f}%")
    print(f"  vs 3-way:          {(best_r/REF_3WAY-1)*100:+.1f}%")

    records.sort(key=lambda x: x[0])
    print(f"\nТоп-10:")
    print(f"  {'rMAE':>7}  {'α_lwr':>6}  {'α_sx':>5}  {'α_sm':>5}  {'α_rbf':>6}")
    for row in records[:10]:
        print(f"  {row[0]:>7.4f}  {row[1]:>6.2f}  {row[2]:>5.2f}  "
              f"{row[3]:>5.2f}  {row[4]:>6.2f}")

    # ── Вклад RBF: сравнение 3-way и 4-way на одном графике ─────────────────
    # 2D срез: α_lwr vs α_sx при лучших α_sm и α_rbf
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Левый: rMAE vs α_rbf при лучших остальных фиксированных
    ax = axes[0]
    r_rbf_curve = []
    fix_sum = a_lwr_b + a_sx_b + a_sm_b
    rbf_vals = alphas[alphas <= round(1.0 - fix_sum, 10) + 1e-9]
    # Свип α_rbf, остальные масштабируются пропорционально
    for a_rbf in np.round(np.arange(0.0, 1.0 + STEP/2, STEP), 2):
        rem = round(1.0 - a_rbf, 10)
        if rem < -1e-9: continue
        # Берём лучшие 3-way веса, масштабируем
        s3 = a_lwr_b + a_sx_b + a_sm_b
        if s3 < 1e-9:
            al, asx, asm = rem/3, rem/3, rem/3
        else:
            al  = a_lwr_b / s3 * rem
            asx = a_sx_b  / s3 * rem
            asm = a_sm_b  / s3 * rem
        blend = al*A_lwr + asx*A_sx + asm*A_sm + a_rbf*A_rbf
        r_rbf_curve.append((a_rbf, rmae(blend, acts)))

    rbf_xs = [x[0] for x in r_rbf_curve]
    rbf_rs = [x[1] for x in r_rbf_curve]
    ax.plot(rbf_xs, rbf_rs, "o-", color="darkorange", lw=1.8, label="RBF-log доля растёт")
    ax.axhline(REF_3WAY,  color="purple",    ls="--", lw=1.2,
               label=f"3-way best={REF_3WAY:.4f}")
    ax.axhline(best_r,    color="red",       ls="-",  lw=1.0,
               label=f"4-way best={best_r:.4f}")
    ax.axhline(REF_ENS2,  color="green",     ls=":",  lw=1.2,
               label=f"Ens(LWR+Sx)={REF_ENS2:.4f}")
    ax.scatter([a_rbf_b], [best_r], color="red", zorder=5, s=60)
    ax.set_xlabel("α_rbf  (доля RBF-log)")
    ax.set_ylabel("rMAE")
    ax.set_title("RBF-log вклад при пропорц. масштабировании\nостальных компонент")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Правый: топ-20 комбинаций — bar chart
    ax2 = axes[1]
    top20 = records[:20]
    labels = [f"({r[1]:.2f},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f})" for r in top20]
    vals   = [r[0] for r in top20]
    colors = ["red" if v == best_r else "steelblue" for v in vals]
    ax2.barh(range(len(vals)), vals, color=colors, alpha=0.75)
    ax2.axvline(REF_3WAY, color="purple", ls="--", lw=1.2,
                label=f"3-way={REF_3WAY:.4f}")
    ax2.set_yticks(range(len(top20)))
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("rMAE")
    ax2.set_title("Топ-20 комбинаций весов\n(α_lwr, α_sx, α_sm, α_rbf)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle(
        f"4-way ансамбль: LWR(p={P_LWR},K={K_LWR}) + Sx-log(p={P_SX}) + "
        f"S-map(θ={THETA}) + RBF-log(p={P_RBF},K={K_RBF})\n"
        f"SBER 1d(4%)+10m(0.4%)  H={H}  n={n}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(OUT / "ensemble4.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {OUT}/ensemble4.png")


if __name__ == "__main__":
    run()
