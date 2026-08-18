#!/usr/bin/env python3
"""
LP-коррекция поврх отдельных методов, затем ансамбль.

Гипотеза: LP лучше работает на индивидуальных предсказаниях (больше ошибок),
чем на уже усреднённом ансамбле.

Конфигурации:
  A) Per-method: LP(LWR), LP(Sx), LP(Smap), LP(RBF) → 4-way ансамбль
  B) Log-группа: LP(Sx+RBF), Abs-группа: LP(LWR+Smap) → 2-way ансамбль
  C) Пары: LP(Sx+RBF) alone, LP(LWR+Smap) alone, LP(Sx)+LP(RBF) → сравнение

Для групп B/C блендинг внутри группы по пропорциям из 4-way оптимума:
  log: α_sx/(α_sx+α_rbf) = 0.35/0.75 ≈ 0.467 Sx + 0.533 RBF
  abs: α_lwr/(α_lwr+α_sm) = 0.05/0.25 = 0.20 LWR + 0.80 Smap

Свип LP: d∈{1,2,3,5}, k∈{5,7,10,12,15,20,30}
Ансамбль A: веса из 4-way оптимума (0.05, 0.35, 0.20, 0.40)
Ансамбль B: веса по сумме 4-way (log=0.75, abs=0.25)

Эталоны: 4-way=0.3963  3-way=0.4004

SBER 1d(4%) + 10m(0.4%), H=1.
"""

import json
import numpy as np
import pandas as pd
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

P_LWR  = 3;  K_LWR  = 50
P_SX   = 8;  K_SX   = P_SX + 1
THETA  = 1.0
P_RBF  = 3;  K_RBF  = 12
A_LWR, A_SX, A_SM, A_RBF = 0.05, 0.35, 0.20, 0.40

# внутригрупповые веса
W_SX_IN_LOG  = A_SX  / (A_SX + A_RBF)   # ≈0.467
W_RBF_IN_LOG = A_RBF / (A_SX + A_RBF)   # ≈0.533
W_LWR_IN_ABS = A_LWR / (A_LWR + A_SM)   # =0.200
W_SM_IN_ABS  = A_SM  / (A_LWR + A_SM)   # =0.800
W_LOG        = A_SX + A_RBF             # =0.75
W_ABS        = A_LWR + A_SM             # =0.25

P_LP   = 6
D_GRID = [1, 2, 3, 5]
K_GRID = [5, 7, 10, 12, 15, 20, 30]

REF_4WAY = 0.3963
REF_3WAY = 0.4004


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    vals, dts = [], []
    direction = 0; ext_val = (highs[0]+lows[0])/2.0; ext_idx = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i]-ext_val >= thr*ext_val:   direction=1;  ext_val,ext_idx=highs[i],i
            elif ext_val-lows[i] >= thr*ext_val:  direction=-1; ext_val,ext_idx=lows[i],i
        elif direction == 1:
            if highs[i] > ext_val:                ext_val,ext_idx=highs[i],i
            elif ext_val-lows[i] >= thr*ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                direction=-1; ext_val,ext_idx=lows[i],i
        else:
            if lows[i] < ext_val:                 ext_val,ext_idx=lows[i],i
            elif highs[i]-ext_val >= thr*ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                direction=1; ext_val,ext_idx=highs[i],i
    if not vals: return np.array([]), np.array([])
    return np.array(vals), np.array(dts)


def build_X(prices, p):
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p-1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p): X[i, lag] = lp[i-lag+1] - lp[i-lag]
    return X


def build_X_lr(prices, p_lr):
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p_lr), np.nan)
    for i in range(p_lr, n):
        for lag in range(p_lr): X[i, lag] = lp[i-lag] - lp[i-lag-1]
    return X


def rmae(preds, actuals):
    preds = np.asarray(preds, dtype=float); valid = ~np.isnan(preds)
    if valid.sum() < 5: return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return float(np.mean(np.abs(preds[valid]-actuals[valid]))/dz) if dz>1e-12 else np.nan


def lp_correct(X_pool_lr, x_pred_lr, p_cur, k_lp, d_lp):
    N = len(X_pool_lr); k = min(k_lp, N)
    if k <= d_lp: return float(p_cur * np.exp(x_pred_lr[0]))
    mu = X_pool_lr.mean(0); sigma = np.where(X_pool_lr.std(0)<1e-10, 1.0, X_pool_lr.std(0))
    Xn = (X_pool_lr-mu)/sigma; xn = (x_pred_lr-mu)/sigma
    dists = np.linalg.norm(Xn-xn, axis=1); knn = np.argsort(dists)[:k]
    Z = Xn[knn]; center = Z.mean(0); Zc = Z-center
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False); V = Vt[:d_lp].T
    delta = xn-center; xc_norm = center + V@(V.T@delta); x_corr = xc_norm*sigma+mu
    return float(p_cur * np.exp(x_corr[0]))


def make_x_pred(y_hat, p_cur, x_cur_lr):
    x = np.empty(P_LP)
    x[0] = np.log(y_hat/p_cur) if p_cur > 1e-12 else 0.0
    x[1:] = x_cur_lr[:P_LP-1]
    return x


def run():
    h1d, l1d, d1d   = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m = find_pivots(h10m, l10m, d10m, T_10M)
    n1d = len(p1d); na = len(p10m)
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")

    Xs = {p: (build_X(p1d,p), build_X(p10m,p)) for p in set([P_LWR,P_SX,P_RBF])}
    Xlr_1d  = build_X_lr(p1d,  P_LP)
    Xlr_10m = build_X_lr(p10m, P_LP)

    print(f"SBER 1d {n1d} пив  10m {na} пив")
    print(f"LP: P_LP={P_LP}  d={D_GRID}  k={K_GRID}")
    print(f"Внутригрупповые веса: Sx={W_SX_IN_LOG:.3f} RBF={W_RBF_IN_LOG:.3f} | "
          f"LWR={W_LWR_IN_ABS:.3f} Smap={W_SM_IN_ABS:.3f}")
    print(f"Групповые веса: log={W_LOG:.2f} abs={W_ABS:.2f}\n")

    # Буферы: raw предсказания каждого метода
    raw = {m: [] for m in ["lwr","sx","smap","rbf"]}
    # LP-скорректированные для каждого метода и параметра LP
    lp  = {(m, k, d): [] for m in ["lwr","sx","smap","rbf"]
           for k in K_GRID for d in D_GRID}
    actuals = []

    for step in range(MIN_HISTORY, n1d-H):
        if any(np.any(np.isnan(Xs[p][0][step])) for p in [P_LWR,P_SX,P_RBF]): continue
        if np.any(np.isnan(Xlr_1d[step])): continue

        ce = int(ce10m_all[step]); p_cur = float(p1d[step])

        def make_pool(p):
            X1d_, X10m_ = Xs[p]
            j1d = np.arange(p-1, step); v1d = ~np.any(np.isnan(X1d_[j1d]),axis=1)&(j1d+H<n1d)
            idx1 = j1d[v1d]; Xp=list(X1d_[idx1]); ya=list(p1d[idx1+H]); ys=list(p1d[idx1])
            j10 = np.arange(p-1, min(ce, na-H))
            if len(j10):
                v10=~np.any(np.isnan(X10m_[j10]),axis=1); idx10=j10[v10]
                Xp.extend(X10m_[idx10]); ya.extend(p10m[idx10+H]); ys.extend(p10m[idx10])
            if len(Xp)<p+2: return None
            X_pool=np.array(Xp); y_abs=np.array(ya); y_lr=np.log(y_abs/np.array(ys))
            x_q=X1d_[step]; mu=X_pool.mean(0); sig=np.where(X_pool.std(0)<1e-10,1.0,X_pool.std(0))
            Xn=(X_pool-mu)/sig; xn=(x_q-mu)/sig; d=np.linalg.norm(Xn-xn,axis=1)
            return Xn, xn, d, y_abs, y_lr, len(y_abs)

        res3=make_pool(P_LWR); res8=make_pool(P_SX)
        y_lwr=y_smap=y_sx=y_rbf=np.nan

        if res3:
            Xn,xn,d,y_abs,y_lr,N=res3; ord_=np.argsort(d)
            k_eff=min(K_LWR,N); knn=ord_[:k_eff]; xi=d[ord_[k_eff-1]]
            if xi<1e-12: y_lwr=float(y_abs[knn].mean())
            else:
                w=np.exp(-0.5*(d[knn]/xi)**2); ws=np.sqrt(w)
                A=np.column_stack([np.ones(k_eff),Xn[knn]])*ws[:,None]
                c,*_=np.linalg.lstsq(A,y_abs[knn]*ws,rcond=None)
                y_lwr=float(c[0]+c[1:]@xn)
            mean_d=d.mean()+1e-12; w_sm=np.exp(-THETA*d/mean_d); ws_sm=np.sqrt(w_sm)
            A_sm=np.column_stack([np.ones(N),Xn])*ws_sm[:,None]
            c_sm,*_=np.linalg.lstsq(A_sm,y_abs*ws_sm,rcond=None)
            y_smap=float(c_sm[0]+c_sm[1:]@xn)
            if N>=K_RBF:
                knn_r=ord_[:K_RBF]; xi_r=d[ord_[K_RBF-1]]
                if xi_r<1e-12: y_rbf=float(p_cur*np.exp(y_lr[knn_r].mean()))
                else:
                    w_r=np.exp(-0.5*(d[knn_r]/xi_r)**2)
                    y_rbf=float(p_cur*np.exp((w_r@y_lr[knn_r])/w_r.sum()))

        if res8 and res8[5]>=K_SX:
            Xn,xn,d,y_abs,y_lr,N=res8; ords=np.argsort(d); knn=ords[:K_SX]; d1=d[ords[0]]
            if d1<1e-12: y_sx=float(p_cur*np.exp(y_lr[ords[0]]))
            else:
                w=np.exp(-d[knn]/d1); w/=w.sum(); y_sx=float(p_cur*np.exp(w@y_lr[knn]))

        for m, yv in [("lwr",y_lwr),("sx",y_sx),("smap",y_smap),("rbf",y_rbf)]:
            raw[m].append(yv)

        # LP-пул
        j1d_lr=np.arange(P_LP,step); v1d_lr=~np.any(np.isnan(Xlr_1d[j1d_lr]),axis=1)
        Xp_lr=list(Xlr_1d[j1d_lr[v1d_lr]])
        j10_lr=np.arange(P_LP,min(ce,na))
        if len(j10_lr):
            v10_lr=~np.any(np.isnan(Xlr_10m[j10_lr]),axis=1); Xp_lr.extend(Xlr_10m[j10_lr[v10_lr]])
        x_cur_lr=Xlr_1d[step]

        valid_lp = (len(Xp_lr)>=P_LP+1 and
                    not any(np.isnan(v) for v in [y_lwr,y_sx,y_smap,y_rbf]))

        for k_lp in K_GRID:
            for d_lp in D_GRID:
                if not valid_lp:
                    for m in ["lwr","sx","smap","rbf"]:
                        lp[(m,k_lp,d_lp)].append(raw[m][-1])
                else:
                    X_pool_lr=np.array(Xp_lr)
                    for m, yv in [("lwr",y_lwr),("sx",y_sx),("smap",y_smap),("rbf",y_rbf)]:
                        xp=make_x_pred(yv, p_cur, x_cur_lr)
                        lp[(m,k_lp,d_lp)].append(lp_correct(X_pool_lr,xp,p_cur,k_lp,d_lp))

        actuals.append(float(p1d[step+H]))

    acts=np.array(actuals); n=len(acts)
    A_raw={m: np.array(raw[m],dtype=float) for m in raw}

    print(f"Шагов: {n}")
    for m in ["lwr","sx","smap","rbf"]:
        print(f"  raw {m}: {rmae(A_raw[m],acts):.4f}")
    r_4way_raw = rmae(A_LWR*A_raw["lwr"]+A_SX*A_raw["sx"]+A_SM*A_raw["smap"]+A_RBF*A_raw["rbf"], acts)
    print(f"  4-way raw: {r_4way_raw:.4f}\n")

    # ── Строим результаты по (d,k) ────────────────────────────────────────────
    alphas = np.round(np.arange(0.0, 1.0 + 0.025, 0.05), 2)

    records=[]
    best_A=np.inf; best_B=np.inf; best_A_kd=None; best_B_kd=None

    for d_lp in D_GRID:
        for k_lp in K_GRID:
            Alp={m: np.array(lp[(m,k_lp,d_lp)],dtype=float) for m in ["lwr","sx","smap","rbf"]}

            # A: свип 4-way весов по скорректированным методам
            rA_best = np.inf
            for a_lwr in alphas:
                for a_sx in alphas:
                    rem2 = round(1.0 - a_lwr - a_sx, 10)
                    if rem2 < -1e-9: continue
                    for a_sm in alphas:
                        a_rbf = round(rem2 - a_sm, 10)
                        if a_rbf < -1e-9: continue
                        blend = (a_lwr*Alp["lwr"] + a_sx*Alp["sx"] +
                                 a_sm*Alp["smap"]  + max(0,a_rbf)*Alp["rbf"])
                        r = rmae(blend, acts)
                        if r < rA_best: rA_best = r

            # B: свип α между log-LP и abs-LP группами
            log_corr = W_SX_IN_LOG*Alp["sx"] + W_RBF_IN_LOG*Alp["rbf"]
            abs_corr = W_LWR_IN_ABS*Alp["lwr"] + W_SM_IN_ABS*Alp["smap"]
            rB_best = min(rmae(a*log_corr + (1-a)*abs_corr, acts) for a in alphas)

            records.append({"config":"A_per","d":d_lp,"k":k_lp,"rMAE":rA_best,
                            "vs_4way":(rA_best/REF_4WAY-1)*100})
            records.append({"config":"B_grp","d":d_lp,"k":k_lp,"rMAE":rB_best,
                            "vs_4way":(rB_best/REF_4WAY-1)*100})

            if rA_best < best_A: best_A=rA_best; best_A_kd=(k_lp,d_lp)
            if rB_best < best_B: best_B=rB_best; best_B_kd=(k_lp,d_lp)

    df=pd.DataFrame(records)
    for cfg, label in [("A_per","A: per-method LP + свип 4-way весов"),
                        ("B_grp","B: log-LP + abs-LP + свип α")]:
        sub=df[df.config==cfg].nsmallest(5,"rMAE")
        print(f"\n{label}:")
        print(sub[["d","k","rMAE","vs_4way"]].to_string(index=False))

    print(f"\n{'='*40}")
    print(f"Лучший A: k={best_A_kd[0]} d={best_A_kd[1]} → {best_A:.4f}  "
          f"vs 4way: {(best_A/REF_4WAY-1)*100:+.2f}%")
    print(f"Лучший B: k={best_B_kd[0]} d={best_B_kd[1]} → {best_B:.4f}  "
          f"vs 4way: {(best_B/REF_4WAY-1)*100:+.2f}%")
    print(f"Эталон 4-way raw: {REF_4WAY:.4f}")

    df.to_csv(OUT/"lp_per_method.csv", index=False)

    # ── График: heatmap A и B рядом ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, cfg, label, best_r, best_kd_ in [
        (axes[0], "A_per", "A: per-method → 4-way ens", best_A, best_A_kd),
        (axes[1], "B_grp", "B: log-LP + abs-LP → 2-way ens", best_B, best_B_kd),
    ]:
        sub=df[df.config==cfg]
        mat=np.array([[sub[(sub.d==d)&(sub.k==k)].rMAE.values[0]
                       for k in K_GRID] for d in D_GRID])
        vmin=min(mat.min(), REF_4WAY)*0.995; vmax=max(mat.max(), REF_4WAY)*1.005
        im=ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=vmin, vmax=vmax)
        for i,d_lp in enumerate(D_GRID):
            for j,k_lp in enumerate(K_GRID):
                r=mat[i,j]; mark="✓" if r<REF_4WAY else "✗"
                ax.text(j,i,f"{r:.4f}\n{mark}",ha="center",va="center",fontsize=7.5,
                        color="white" if r>(mat.max()*0.6+mat.min()*0.4) else "black")
        ax.set_xticks(range(len(K_GRID))); ax.set_xticklabels([str(k) for k in K_GRID])
        ax.set_yticks(range(len(D_GRID))); ax.set_yticklabels([f"d={d}" for d in D_GRID])
        ax.set_xlabel("k_lp"); ax.set_ylabel("d_lp")
        ax.set_title(f"{label}\n4-way base={REF_4WAY:.4f}  best={best_r:.4f} "
                     f"(k={best_kd_[0]},d={best_kd_[1]})")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"LP-коррекция на отдельных методах/группах  |  SBER 1d+10m  H={H}  n={n}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT/"lp_per_method.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + heatmap → {OUT}")


if __name__ == "__main__":
    run()
