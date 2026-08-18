#!/usr/bin/env python3
"""
zigzag_ar_correction.py

Коррекция ошибок 4-way ансамбля через авторегрессию (AR) и
экспоненциальное сглаживание (ETS).

Идея: если ошибки имеют временну́ю структуру (автокорреляцию),
можно предсказать следующую ошибку и вычесть её из прогноза.

Методы:
  ETS (Exponential Smoothing):
      ê[t+1] = α·e[t] + (1-α)·ê[t]
      α ∈ {0.05, 0.1, 0.2, 0.3, 0.5}

  AR(p) — авторегрессия с интерсептом:
      e[t] = c₀ + φ₁e[t-1] + ... + φₚe[t-p]
      p ∈ {1, 2, 3, 5}; OLS на каждом шаге по всей истории

Оба метода в двух вариантах:
  global    — один буфер ошибок на все шаги
  per-regime — отдельный буфер для каждого из 4 режимов
              (тренд × тип цели), тренд по MA100 сырых 1d баров

Каузальный контракт: коррекция в шаге t использует только ошибки [0..t-1].
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
OUT.mkdir(exist_ok=True)

T_1D = 0.04; T_10M = 0.004; H = 1; MIN_HISTORY = 50

P_LWR = 3; K_LWR = 50; P_SX = 8; K_SX = P_SX + 1
THETA = 1.0; P_RBF = 3; K_RBF = 12

A_LWR = 0.05; A_SX = 0.35; A_SM = 0.20; A_RBF = 0.40

W_MA_TREND  = 100               # MA для определения тренда
ETS_ALPHAS  = [0.05, 0.1, 0.2, 0.3, 0.5]
AR_ORDERS   = [1, 2, 3, 5]
MIN_N_AR    = 20                # мин. история перед применением AR
REGIMES     = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
N_BINS      = 10
REF_RMAE    = 0.3963


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
    ext_val = (highs[0] + lows[0]) / 2.0; ext_idx = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1;  ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val, ext_idx = lows[i], i
        elif direction == 1:
            if highs[i] > ext_val: ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                ptypes.append(1); bidxs.append(ext_idx)
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val: ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                ptypes.append(-1); bidxs.append(ext_idx)
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


# ─── Функции коррекции ────────────────────────────────────────────────────────

def ar_predict(errors: list, p: int) -> float:
    """AR(p) с интерсептом: предсказать следующую ошибку."""
    n = len(errors)
    if n < max(p + 2, MIN_N_AR):
        return 0.0
    e = np.array(errors, dtype=float)
    # X[k] = [1, e[p+k-1], ..., e[k]]  для k=0..n-p-1
    n_rows = n - p
    cols = [np.ones(n_rows)] + [e[p - 1 - lag: n - 1 - lag] for lag in range(p)]
    X = np.column_stack(cols)          # (n_rows, p+1)
    Y = e[p:]                          # (n_rows,)
    c, *_ = np.linalg.lstsq(X, Y, rcond=None)
    x_next = np.array([1.0] + [e[n - 1 - lag] for lag in range(p)])
    return float(x_next @ c)


def apply_ets_global(preds, acts, alpha):
    n = len(preds); corrected = np.empty(n); ema = 0.0
    for i in range(n):
        corrected[i] = preds[i] - ema
        ema = alpha * (preds[i] - acts[i]) + (1 - alpha) * ema
    return corrected


def apply_ets_regime(preds, acts, types, trend_dirs, alpha):
    n = len(preds); corrected = np.empty(n)
    ema = {r: 0.0 for r in REGIMES}
    for i in range(n):
        regime = (int(trend_dirs[i]), int(types[i]))
        corrected[i] = preds[i] - ema[regime]
        ema[regime] = alpha * (preds[i] - acts[i]) + (1 - alpha) * ema[regime]
    return corrected


def apply_ar_global(preds, acts, p):
    n = len(preds); corrected = np.empty(n); errors = []
    for i in range(n):
        corrected[i] = preds[i] - ar_predict(errors, p)
        errors.append(float(preds[i] - acts[i]))
    return corrected


def apply_ar_regime(preds, acts, types, trend_dirs, p):
    n = len(preds); corrected = np.empty(n)
    errors = {r: [] for r in REGIMES}
    for i in range(n):
        regime = (int(trend_dirs[i]), int(types[i]))
        corrected[i] = preds[i] - ar_predict(errors[regime], p)
        errors[regime].append(float(preds[i] - acts[i]))
    return corrected


# ─────────────────────────────────────────────────────────────────────────────

def main():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d,  types1d, bidxs1d = find_pivots(h1d, l1d, d1d, T_1D)
    p10m, dt10m, _,        _       = find_pivots(h10m, l10m, d10m, T_10M)

    n1d = len(p1d); na = len(p10m)
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")
    Xs = {p: (build_X(p1d, p), build_X(p10m, p))
          for p in set([P_LWR, P_SX, P_RBF])}

    mid1d = (h1d + l1d) / 2.0
    ma_trend = causal_ma(mid1d, W_MA_TREND)

    print(f"SBER 1d {n1d} пив  10m {na} пив")
    print("Прогон ансамбля...", flush=True)

    # ── Ансамблевый прогон ───────────────────────────────────────────────────
    raw_preds   = []
    raw_actuals = []
    raw_types   = []
    raw_origins = []
    raw_bar_idx = []
    raw_steps   = []

    for step in range(MIN_HISTORY, n1d - H):
        skip = any(np.any(np.isnan(Xs[p][0][step])) for p in [P_LWR, P_SX, P_RBF])
        if skip:
            continue
        ce = int(ce10m_all[step]); p_cur = float(p1d[step])
        ttype = int(types1d[step + H])

        def make_pool(p):
            X1d_, X10m_ = Xs[p]
            j1d = np.arange(p - 1, step)
            v1d = ~np.any(np.isnan(X1d_[j1d]), axis=1) & (j1d + H < n1d)
            idx1 = j1d[v1d]
            Xp = list(X1d_[idx1]); yp_abs = list(p1d[idx1 + H])
            yp_src = list(p1d[idx1])
            j10 = np.arange(p - 1, min(ce, na - H))
            if len(j10):
                v10 = ~np.any(np.isnan(X10m_[j10]), axis=1)
                idx10 = j10[v10]
                Xp.extend(X10m_[idx10]); yp_abs.extend(p10m[idx10 + H])
                yp_src.extend(p10m[idx10])
            if len(Xp) < p + 2:
                return None
            X_pool = np.array(Xp); y_abs = np.array(yp_abs, dtype=float)
            y_lr = np.log(y_abs / np.array(yp_src, dtype=float))
            x_q  = X1d_[step]
            mu = X_pool.mean(0); sig = np.where(X_pool.std(0)<1e-10, 1.0, X_pool.std(0))
            Xn = (X_pool - mu)/sig; xn = (x_q - mu)/sig
            d  = np.linalg.norm(Xn - xn, axis=1)
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
                A_ = np.column_stack([np.ones(k_eff), Xn[knn]])*ws[:,None]
                c, *_ = np.linalg.lstsq(A_, y_abs[knn]*ws, rcond=None)
                y_lwr = float(c[0] + c[1:] @ xn)
            mean_d = d.mean()+1e-12
            w_sm = np.exp(-THETA*d/mean_d); ws_sm = np.sqrt(w_sm)
            A_sm = np.column_stack([np.ones(N), Xn])*ws_sm[:,None]
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

        pred = A_LWR*y_lwr + A_SX*y_sx + A_SM*y_smap + A_RBF*y_rbf
        if np.isnan(pred):
            continue

        raw_preds.append(pred); raw_actuals.append(float(p1d[step+H]))
        raw_origins.append(p_cur); raw_types.append(ttype)
        raw_bar_idx.append(int(bidxs1d[step])); raw_steps.append(step)

    preds    = np.array(raw_preds);   acts  = np.array(raw_actuals)
    p_curs   = np.array(raw_origins); types = np.array(raw_types)
    bar_idxs = np.array(raw_bar_idx); steps = np.array(raw_steps)
    errs_raw = preds - acts
    n_steps  = len(acts)
    dz       = float(np.mean(np.abs(np.diff(acts))))

    # Направление тренда для каждого шага
    trend_dirs = np.array([
        1 if p_curs[i] >= ma_trend[bar_idxs[i]] else -1
        for i in range(n_steps)
    ])

    print(f"Шагов: {n_steps}  raw rMAE: {rmae_arr(errs_raw, acts):.4f}\n")

    # ── Применяем все методы коррекции ───────────────────────────────────────
    methods = {}   # name → corrected preds array

    for alpha in ETS_ALPHAS:
        methods[f"ETS-global  α={alpha}"] = apply_ets_global(preds, acts, alpha)
        methods[f"ETS-regime  α={alpha}"] = apply_ets_regime(
            preds, acts, types, trend_dirs, alpha)

    for p in AR_ORDERS:
        methods[f"AR({p})-global "] = apply_ar_global(preds, acts, p)
        methods[f"AR({p})-regime"] = apply_ar_regime(
            preds, acts, types, trend_dirs, p)

    # ── Таблица результатов ───────────────────────────────────────────────────
    print(f"{'═'*55}")
    print(f"{'Метод':<22}  {'rMAE':>7}  {'Δ vs raw':>9}")
    print(f"{'─'*55}")
    print(f"{'raw (4-way)':<22}  {REF_RMAE:>7.4f}  {'0.00%':>9}")
    print(f"{'─'*55}")

    scores = []
    for name, cp in methods.items():
        r = rmae_arr(cp - acts, acts)
        delta = (r / REF_RMAE - 1) * 100
        scores.append((name, r, delta, cp))
        print(f"{name:<22}  {r:>7.4f}  {delta:>+8.2f}%")

    scores.sort(key=lambda x: x[1])
    best_name, best_r, best_d, best_cp = scores[0]
    print(f"\nЛучший: {best_name.strip()}  rMAE={best_r:.4f}  ({best_d:+.2f}%)")

    # ── Временна́я таблица для топ-3 ──────────────────────────────────────────
    top3 = scores[:3]
    print(f"\n{'─'*65}")
    print("Временна́я таблица rMAE (raw vs топ-3 методов):")
    bin_edges = np.percentile(steps, np.linspace(0, 100, N_BINS + 1))
    bin_edges[-1] += 1

    hdr = f"{'Шаги':>12}  {'raw':>7}" + "".join(
        f"  {n.strip()[:14]:>14}" for n, *_ in top3)
    print(hdr); print("─" * len(hdr))
    for i in range(N_BINS):
        m = (steps >= bin_edges[i]) & (steps < bin_edges[i+1])
        if m.sum() < 3:
            continue
        lo, hi = int(bin_edges[i]), int(bin_edges[i+1]) - 1
        r_raw = float(np.mean(np.abs(errs_raw[m])) / dz)
        row = f"{lo}..{hi:>3}  {r_raw:>7.4f}"
        for _, _, _, cp in top3:
            r = float(np.mean(np.abs((cp - acts)[m])) / dz)
            row += f"  {r:>14.4f}"
        print(row)

    # ── График ────────────────────────────────────────────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))

    # Левый: bar chart всех методов
    names_b = ["raw"] + [s[0].strip() for s in scores]
    vals_b  = [REF_RMAE] + [s[1] for s in scores]
    cols_b  = ["#888"] + ["#2ecc71" if v < REF_RMAE else "#e74c3c" for v in vals_b[1:]]
    y_pos   = range(len(names_b))
    ax_l.barh(y_pos, vals_b, color=cols_b, alpha=0.8)
    ax_l.axvline(REF_RMAE, color="black", ls=":", lw=1.0)
    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels(names_b, fontsize=7)
    ax_l.invert_yaxis()
    ax_l.set_xlabel("rMAE")
    ax_l.set_title("rMAE всех методов\n(зелёный = лучше raw)")
    ax_l.grid(True, alpha=0.3, axis="x")

    # Правый: временна́я таблица для raw + топ-3
    bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(N_BINS)]
    bin_labels  = [f"{int(bin_edges[i])}–{int(bin_edges[i+1])-1}" for i in range(N_BINS)]

    def bin_rmae(errs_arr):
        out = []
        for i in range(N_BINS):
            m = (steps >= bin_edges[i]) & (steps < bin_edges[i+1])
            out.append(float(np.mean(np.abs(errs_arr[m])) / dz) if m.sum()>=3 else np.nan)
        return out

    ax_r.plot(range(N_BINS), bin_rmae(errs_raw), "o--",
              color="#888", lw=1.5, label="raw")
    style = [("o-", "#2ecc71"), ("s-", "#3498db"), ("^-", "#e67e22")]
    for (n, r, d, cp), (sty, col) in zip(top3, style):
        ax_r.plot(range(N_BINS), bin_rmae(cp - acts), sty,
                  color=col, lw=1.5, label=n.strip()[:18], alpha=0.85)
    ax_r.set_xticks(range(N_BINS))
    ax_r.set_xticklabels(bin_labels, rotation=35, ha="right", fontsize=7)
    ax_r.set_ylabel("rMAE"); ax_r.set_title("rMAE по бинам: raw vs топ-3")
    ax_r.legend(fontsize=8); ax_r.grid(True, alpha=0.3)

    fig.suptitle(
        f"AR / ETS коррекция ошибок ансамбля | SBER 1d+10m H=1  n={n_steps}",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "ar_correction.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {OUT}/ar_correction.png")


if __name__ == "__main__":
    main()
