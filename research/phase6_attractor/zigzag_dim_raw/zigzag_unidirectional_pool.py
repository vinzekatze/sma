#!/usr/bin/env python3
"""
Однонаправленный пул зигзага.

Идея: пивоты строго чередуются HIGH/LOW. При прогнозе из пика в пул
включаются только прошлые пики, из трога — только трогии.
lr1 уже кодирует направление, но жёсткая фильтрация убирает «примеси»
из противоположного направления, ценой уменьшения пула вдвое.

Протокол:
  - LWR K-sweep (унинаправленный пул)
  - S-map θ=10 (унинаправленный пул)
  - Свип α ансамбля на лучшем K

Базовые значения (mixed pool):
  LWR K=50  rMAE=0.4181
  S-map θ=10 rMAE=0.4251
  Ансамбль α=0.65 rMAE=0.4143

SBER 1d(4%) + 10m(0.4%), p=3, H=1.
КАУЗАЛЬНОСТЬ: пул строго j < step, 10m — до даты step.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT      = BASE_DIR / "results"

T_1D        = 0.04
T_10M       = 0.004
P           = 3
H           = 1
MIN_HISTORY = 50
THETA_SMAP  = 10.0
ALPHA_ENS   = 0.65

K_GRID = [6, 9, 12, 15, 18, 24, 30, 50, 75, 100, 150, 200, 300, 500]

# Baselines (mixed pool)
R_LWR_MIX  = 0.4181
R_SMAP_MIX = 0.4251
R_ENS_MIX  = 0.4143


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    """Возвращает (prices, dates, dirs) где dirs: +1=HIGH/пик, -1=LOW/трог."""
    vals, dts, dirs, direction = [], [], [], 0
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
                vals.append(ext_val); dts.append(dates[ext_idx]); dirs.append(1)
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx]); dirs.append(-1)
                direction = 1; ext_val, ext_idx = highs[i], i
    if not vals:
        return np.array([]), np.array([]), np.array([], dtype=np.int8)
    return np.array(vals), np.array(dts), np.array(dirs, dtype=np.int8)


def build_X(prices, p):
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def rmae(preds, actuals):
    if len(preds) < 5:
        return np.nan
    dz = np.mean(np.abs(np.diff(actuals)))
    return np.nan if dz < 1e-12 else float(np.mean(np.abs(preds - actuals)) / dz)


def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")

    p1d,  dates_1d,  dirs_1d  = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dates_10m, dirs_10m = find_pivots(h10m, l10m, d10m, T_10M)

    X_1d  = build_X(p1d,  P)
    X_10m = build_X(p10m, P)

    n1d = len(p1d)
    ce10m_all = np.searchsorted(dates_10m, dates_1d, side="left")

    n_peaks  = int((dirs_1d ==  1).sum())
    n_troughs = int((dirs_1d == -1).sum())
    print(f"1d: {n1d} пив  (пиков={n_peaks}, трогов={n_troughs})")
    print(f"10m: {len(p10m)} пив")
    print(f"p={P}  H={H}  K_GRID={K_GRID}")
    print(f"θ_smap={THETA_SMAP}  α_ens={ALPHA_ENS}")
    print(f"\nБазовые (mixed pool): LWR={R_LWR_MIX:.4f}  S-map={R_SMAP_MIX:.4f}  Ens={R_ENS_MIX:.4f}\n")

    preds_lwr  = {k: [] for k in K_GRID}
    preds_smap = []
    actuals_all = []
    pool_sizes  = []

    for step in range(MIN_HISTORY, n1d - H):
        if np.any(np.isnan(X_1d[step])):
            continue

        q_dir = dirs_1d[step]

        # Пул 1d — только то же направление
        j1d = np.arange(P - 1, step)
        mask_1d = (~np.any(np.isnan(X_1d[j1d]), axis=1)
                   & (j1d + H < n1d)
                   & (dirs_1d[j1d] == q_dir))
        Xp = list(X_1d[j1d[mask_1d]])
        yp = list(p1d[j1d[mask_1d] + H])

        # Пул 10m — только то же направление
        ce = int(ce10m_all[step]); na = len(p10m)
        j10 = np.arange(P - 1, min(ce, na - H))
        if len(j10):
            mask_10m = (~np.any(np.isnan(X_10m[j10]), axis=1)
                        & (dirs_10m[j10] == q_dir))
            Xp.extend(X_10m[j10[mask_10m]])
            yp.extend(p10m[j10[mask_10m] + H])

        N = len(Xp)
        if N < P + 2:
            continue

        X_pool = np.array(Xp); y_pool = np.array(yp)
        x_q    = X_1d[step]
        pool_sizes.append(N)

        # Z-score
        mu    = X_pool.mean(0)
        sigma = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn    = (X_pool - mu) / sigma
        xn    = (x_q    - mu) / sigma

        dists = np.linalg.norm(Xn - xn, axis=1)
        order = np.argsort(dists)

        # LWR sweep
        for k in K_GRID:
            k_eff = min(k, N)
            knn   = order[:k_eff]
            xi    = dists[order[k_eff - 1]]
            if xi < 1e-12:
                pred = float(y_pool[knn].mean())
            else:
                w  = np.exp(-0.5 * (dists[knn] / xi) ** 2)
                ws = np.sqrt(w)
                A  = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
                b  = y_pool[knn] * ws
                c, *_ = np.linalg.lstsq(A, b, rcond=None)
                pred = float(c[0] + c[1:] @ xn)
            preds_lwr[k].append(pred)

        # S-map θ=10
        mean_d = dists.mean()
        w_sm   = np.exp(-THETA_SMAP * dists / (mean_d + 1e-12))
        ws_sm  = np.sqrt(w_sm)
        A_sm   = np.column_stack([np.ones(N), Xn]) * ws_sm[:, None]
        b_sm   = y_pool * ws_sm
        c_sm, *_ = np.linalg.lstsq(A_sm, b_sm, rcond=None)
        preds_smap.append(float(c_sm[0] + c_sm[1:] @ xn))

        actuals_all.append(float(p1d[step + H]))

    actuals = np.array(actuals_all)
    n = len(actuals)
    ps_arr = np.array(pool_sizes)
    print(f"Шагов: {n}   Пул: {ps_arr.mean():.0f} ср. ({ps_arr.min()}..{ps_arr.max()})\n")

    r_smap = rmae(np.array(preds_smap), actuals)

    records = []
    r_best = np.inf; k_best = K_GRID[0]

    print(f"{'K':>5}  {'rMAE':>8}  {'vs mix-LWR':>11}  {'vs mix-ens':>11}")
    print("─" * 50)

    for k in K_GRID:
        r = rmae(np.array(preds_lwr[k]), actuals)
        mk = ""
        if r < r_best:
            r_best = r; k_best = k; mk = " ←"
        print(f"  K={k:4d}  {r:.4f}   {(r/R_LWR_MIX-1)*100:+.1f}%{'':<5}  {(r/R_ENS_MIX-1)*100:+.1f}%{mk}")
        records.append({"K": k, "rMAE_dir": r,
                        "vs_mix_lwr_pct": (r / R_LWR_MIX - 1) * 100,
                        "vs_mix_ens_pct": (r / R_ENS_MIX - 1) * 100})

    print(f"\n  S-map θ={THETA_SMAP} (dir)  {r_smap:.4f}"
          f"   vs mix-smap: {(r_smap/R_SMAP_MIX-1)*100:+.1f}%"
          f"   vs mix-ens:  {(r_smap/R_ENS_MIX -1)*100:+.1f}%")

    # Свип α ансамбля
    preds_lwr_best = np.array(preds_lwr[k_best])
    preds_sm       = np.array(preds_smap)

    print(f"\nСвип α (dir-pool, LWR K={k_best} + S-map θ={THETA_SMAP}):")
    r_ens_best = np.inf; alpha_best = ALPHA_ENS
    for a in np.arange(0.0, 1.05, 0.05):
        r_a = rmae(a * preds_lwr_best + (1 - a) * preds_sm, actuals)
        mk = ""
        if r_a < r_ens_best:
            r_ens_best = r_a; alpha_best = a; mk = " ←"
        print(f"  α={a:.2f}  {r_a:.4f}   vs mix-ens: {(r_a/R_ENS_MIX-1)*100:+.1f}%{mk}")

    print(f"\nИТОГ dir-pool:")
    print(f"  Лучший K={k_best}  rMAE={r_best:.4f}  vs mix-ens: {(r_best/R_ENS_MIX-1)*100:+.1f}%")
    print(f"  Ансамбль α={alpha_best:.2f}  rMAE={r_ens_best:.4f}  vs mix-ens: {(r_ens_best/R_ENS_MIX-1)*100:+.1f}%")

    df = pd.DataFrame(records)
    df.to_csv(OUT / "unidirectional_pool.csv", index=False)

    # ── График ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ks = df["K"].values
    ax.plot(ks, df["rMAE_dir"], "o-", color="steelblue", lw=2.5, ms=8,
            label="LWR (dir pool)")
    ax.axhline(r_smap, color="purple", lw=1.5, ls="--",
               label=f"S-map θ={THETA_SMAP} (dir)  {r_smap:.4f}")
    ax.axhline(R_LWR_MIX, color="gray", lw=1.5, ls=":",
               label=f"LWR K=50 (mixed)  {R_LWR_MIX:.4f}")
    ax.axhline(R_ENS_MIX, color="crimson", lw=1.5, ls=":",
               label=f"Ens α=0.65 (mixed)  {R_ENS_MIX:.4f}")
    ax.axvline(k_best, color="steelblue", lw=1, ls=":", alpha=0.5)
    ax.annotate(f"K={k_best}\n{r_best:.4f}",
                xy=(k_best, r_best),
                xytext=(k_best * 1.4, r_best + 0.02),
                fontsize=9, color="steelblue",
                arrowprops=dict(arrowstyle="->", color="steelblue", lw=1))
    ax.set_xscale("log")
    ax.set_xlabel("K (log-шкала)")
    ax.set_ylabel("rMAE")
    ax.set_title("LWR: однонаправленный пул — свип K")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks], fontsize=7.5, rotation=45)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.2)

    ax = axes[1]
    vs_mix = df["vs_mix_ens_pct"].values
    clrs = ["seagreen" if v <= 0 else "salmon" for v in vs_mix]
    ax.bar(range(len(ks)), vs_mix, color=clrs, alpha=0.85)
    ax.axhline(0, color="crimson", lw=1.5, ls="--",
               label=f"Ens mixed (baseline 0.4143)")
    ax.axhline((R_LWR_MIX / R_ENS_MIX - 1) * 100, color="gray", lw=1.2, ls=":",
               label=f"LWR K=50 mixed {(R_LWR_MIX/R_ENS_MIX-1)*100:+.1f}%")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks], fontsize=8, rotation=45)
    ax.set_xlabel("K")
    ax.set_ylabel("vs Ens mixed (0.4143), % rMAE")
    ax.set_title("Dir pool vs Ens mixed")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Однонаправленный пул  |  SBER 1d({T_1D*100:.0f}%) + 10m({T_10M*100:.1f}%)\n"
        f"X=[price, lr₁, lr₂]  p={P}  H={H}  "
        f"Пул: только пивоты того же направления (HIGH→HIGH или LOW→LOW)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(OUT / "unidirectional_pool.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + график → {OUT}")


if __name__ == "__main__":
    run()
