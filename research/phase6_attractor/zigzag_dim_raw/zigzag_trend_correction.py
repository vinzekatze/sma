#!/usr/bin/env python3
"""
zigzag_trend_correction.py

Тренд-кондиционированная каузальная коррекция 4-way ансамбля.

4 режима = (тренд × тип цели):
  uptrend   + HIGH-target  (p_cur > MA, next pivot = HIGH)
  uptrend   + LOW-target
  downtrend + HIGH-target
  downtrend + LOW-target

Коррекция: pred_corr = pred − mean(last W_corr raw errors of same regime).
Применяется только если в режиме накоплено ≥ MIN_N наблюдений.

Свип:
  W_MA   ∈ {50, 100, 200}  — окно MA для определения тренда (на сырых 1d барах)
  W_corr ∈ {10, 20, 50, None}  — окно коррекции (None = вся история режима)

Эталон: 4-way raw rMAE = 0.3963
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

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

W_MA_GRID   = [50, 100, 200]
W_CORR_GRID = [10, 20, 50, None]   # None = вся история режима
MIN_N       = 5                     # минимум наблюдений в режиме
N_BINS      = 10                    # бины для временно́й таблицы

REGIMES = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
REGIME_LABELS = {
    (1,  1): "up+HIGH",
    (1, -1): "up+LOW ",
    (-1, 1): "dn+HIGH",
    (-1,-1): "dn+LOW ",
}
REF_RMAE = 0.3963


# ─────────────────────────────────────────────────────────────────────────────

def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
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
    cs = np.concatenate([[0.0], np.cumsum(series)])
    counts = np.minimum(np.arange(1, len(series) + 1), W)
    start  = np.maximum(0, np.arange(len(series)) - W + 1)
    return (cs[np.arange(1, len(series) + 1)] - cs[start]) / counts


def rmae_arr(errs, acts):
    dz = np.mean(np.abs(np.diff(acts)))
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


def apply_correction(preds, acts, types, p_curs, bar_idxs, ma, W_corr):
    """Каузальная 4-режимная коррекция. Возвращает скорректированные pred."""
    n = len(preds)
    corrected = np.empty(n)
    past_errors = {r: [] for r in REGIMES}

    for i in range(n):
        trend  = 1 if p_curs[i] >= ma[bar_idxs[i]] else -1
        regime = (trend, int(types[i]))
        buf    = past_errors[regime]

        if len(buf) >= MIN_N:
            window     = buf[-W_corr:] if W_corr is not None else buf
            correction = float(np.mean(window))
        else:
            correction = 0.0

        corrected[i] = preds[i] - correction
        past_errors[regime].append(float(preds[i] - acts[i]))  # raw error

    return corrected


# ─────────────────────────────────────────────────────────────────────────────

def main():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d,  types1d, bidxs1d = find_pivots(h1d, l1d, d1d,  T_1D)
    p10m, dt10m, _,        _       = find_pivots(h10m, l10m, d10m, T_10M)

    n1d = len(p1d); na = len(p10m)
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")

    Xs = {p: (build_X(p1d, p), build_X(p10m, p))
          for p in set([P_LWR, P_SX, P_RBF])}

    mid1d     = (h1d + l1d) / 2.0
    ma_arrays = {W: causal_ma(mid1d, W) for W in W_MA_GRID}

    print(f"SBER 1d {n1d} пив  10m {na} пив")
    print(f"Эталон 4-way rMAE = {REF_RMAE}\n")
    print("Прогон ансамбля...", flush=True)

    # ── Прогон ансамбля ───────────────────────────────────────────────────────
    raw_preds   = []
    raw_actuals = []
    raw_origins = []
    raw_types   = []
    raw_bar_idx = []
    raw_step    = []

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
            x_q    = X1d_[step]
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

        raw_preds.append(pred_ens)
        raw_actuals.append(float(p1d[step + H]))
        raw_origins.append(p_cur)
        raw_types.append(ttype)
        raw_bar_idx.append(int(bidxs1d[step]))
        raw_step.append(step)

    preds    = np.array(raw_preds)
    acts     = np.array(raw_actuals)
    p_curs   = np.array(raw_origins)
    types    = np.array(raw_types)
    bar_idxs = np.array(raw_bar_idx)
    steps    = np.array(raw_step)
    errs_raw = preds - acts
    n_steps  = len(acts)
    dz       = float(np.mean(np.abs(np.diff(acts))))

    print(f"Шагов: {n_steps}  raw rMAE: {rmae_arr(errs_raw, acts):.4f}\n")

    # ── Свип (W_MA × W_corr) ─────────────────────────────────────────────────
    print(f"{'═'*62}")
    print("СЕКЦИЯ 1: rMAE по (W_MA × W_corr)")
    print(f"{'═'*62}")

    best_r    = np.inf
    best_cfg  = None
    all_results = {}

    header = f"{'W_MA':>6}" + "".join(f"{'W='+str(w) if w else 'W=all':>9}"
                                       for w in W_CORR_GRID)
    print(f"\n{header}")
    print("─" * (7 + 9 * len(W_CORR_GRID)))

    for W_MA in W_MA_GRID:
        ma = ma_arrays[W_MA]
        row = f"{W_MA:>6}"
        for W_corr in W_CORR_GRID:
            cp = apply_correction(preds, acts, types, p_curs, bar_idxs, ma, W_corr)
            r  = rmae_arr(cp - acts, acts)
            all_results[(W_MA, W_corr)] = cp
            if r < best_r:
                best_r = r; best_cfg = (W_MA, W_corr)
            row += f"  {r:.4f}"
        print(row)

    print(f"\nraw (без коррекции): {REF_RMAE:.4f}")
    print(f"Лучший: W_MA={best_cfg[0]}  W_corr={best_cfg[1]}  rMAE={best_r:.4f}  "
          f"({(best_r/REF_RMAE-1)*100:+.2f}% vs raw)")

    # ── Секция 2: per-режим статистика для лучшей конфигурации ───────────────
    print(f"\n{'═'*62}")
    print(f"СЕКЦИЯ 2: Per-режим  (W_MA={best_cfg[0]}, W_corr={best_cfg[1]})")
    print(f"{'═'*62}")
    W_MA_b, W_corr_b = best_cfg
    ma_b  = ma_arrays[W_MA_b]
    cp_b  = all_results[best_cfg]
    errs_b = cp_b - acts

    print(f"\n{'Режим':>10}  {'n':>4}  {'mean_err raw':>13}  {'mean_err corr':>14}  "
          f"{'rMAE raw':>9}  {'rMAE corr':>10}")
    print("─" * 66)
    for regime in REGIMES:
        trend, ttype = regime
        trend_dir = 1 if True else -1  # placeholder
        # Реконструируем маску режима
        trend_mask = np.array([
            (1 if p_curs[i] >= ma_b[bar_idxs[i]] else -1) == trend
            for i in range(n_steps)
        ])
        type_mask  = types == ttype
        m = trend_mask & type_mask
        if m.sum() < 3:
            continue
        me_raw  = np.mean(errs_raw[m])
        me_corr = np.mean(errs_b[m])
        r_raw   = float(np.mean(np.abs(errs_raw[m])) / dz)
        r_corr  = float(np.mean(np.abs(errs_b[m]))   / dz)
        lbl     = REGIME_LABELS[regime]
        print(f"{lbl:>10}  {m.sum():>4}  {me_raw:>+12.2f}  {me_corr:>+13.2f}  "
              f"{r_raw:>9.4f}  {r_corr:>10.4f}")

    # ── Секция 3: временна́я таблица для лучшей конфигурации ─────────────────
    print(f"\n{'═'*62}")
    print(f"СЕКЦИЯ 3: rMAE по бинам — raw vs лучшая коррекция")
    print(f"{'═'*62}")
    bin_edges = np.percentile(steps, np.linspace(0, 100, N_BINS + 1))
    bin_edges[-1] += 1

    print(f"\n{'Шаги':>12}  {'raw':>7}  {'corr':>7}  {'Δ':>7}")
    print("─" * 38)
    for i in range(N_BINS):
        m = (steps >= bin_edges[i]) & (steps < bin_edges[i+1])
        if m.sum() < 3:
            continue
        r_raw  = float(np.mean(np.abs(errs_raw[m])) / dz)
        r_corr = float(np.mean(np.abs(errs_b[m]))   / dz)
        delta  = r_corr - r_raw
        lo, hi = int(bin_edges[i]), int(bin_edges[i+1]) - 1
        print(f"{lo}..{hi:>3}  {r_raw:>7.4f}  {r_corr:>7.4f}  {delta:>+7.4f}")

    # ── График: heatmap W_MA × W_corr ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Heatmap
    ax = axes[0]
    mat = np.array([[rmae_arr(all_results[(wma, wc)] - acts, acts)
                     for wc in W_CORR_GRID]
                    for wma in W_MA_GRID])
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                   vmin=mat.min() - 0.002, vmax=REF_RMAE + 0.002)
    ax.set_xticks(range(len(W_CORR_GRID)))
    ax.set_xticklabels([str(w) if w else "all" for w in W_CORR_GRID])
    ax.set_yticks(range(len(W_MA_GRID)))
    ax.set_yticklabels(W_MA_GRID)
    ax.set_xlabel("W_corr")
    ax.set_ylabel("W_MA")
    ax.set_title(f"rMAE (зелёный=лучше)\nraw={REF_RMAE:.4f}")
    for i in range(len(W_MA_GRID)):
        for j in range(len(W_CORR_GRID)):
            ax.text(j, i, f"{mat[i,j]:.4f}", ha="center", va="center",
                    fontsize=9, color="black")
    plt.colorbar(im, ax=ax)

    # Per-режим bar chart: rMAE raw vs corr
    ax2 = axes[1]
    ma_b = ma_arrays[best_cfg[0]]
    regime_raw  = []
    regime_corr = []
    r_labels    = []
    for regime in REGIMES:
        trend, ttype = regime
        trend_mask = np.array([
            (1 if p_curs[i] >= ma_b[bar_idxs[i]] else -1) == trend
            for i in range(n_steps)
        ])
        m = trend_mask & (types == ttype)
        if m.sum() < 3:
            continue
        regime_raw.append(float(np.mean(np.abs(errs_raw[m])) / dz))
        regime_corr.append(float(np.mean(np.abs(errs_b[m]))  / dz))
        r_labels.append(REGIME_LABELS[regime])

    x = np.arange(len(r_labels))
    w = 0.35
    ax2.bar(x - w/2, regime_raw,  width=w, color="steelblue", alpha=0.8, label="raw")
    ax2.bar(x + w/2, regime_corr, width=w, color="tomato",    alpha=0.8, label="corrected")
    ax2.axhline(REF_RMAE, color="black", ls=":", lw=1.0, label=f"global raw={REF_RMAE:.4f}")
    ax2.set_xticks(x); ax2.set_xticklabels(r_labels, fontsize=9)
    ax2.set_ylabel("rMAE"); ax2.set_title(f"Per-режим: raw vs corr\n(W_MA={best_cfg[0]}, W_corr={best_cfg[1]})")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Тренд-кондиционированная коррекция | SBER 1d+10m H=1", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "trend_correction.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {OUT}/trend_correction.png")


if __name__ == "__main__":
    main()
