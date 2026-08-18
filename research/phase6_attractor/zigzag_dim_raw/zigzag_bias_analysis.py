#!/usr/bin/env python3
"""
zigzag_bias_analysis.py

Анализ систематического смещения 4-way ансамбля по типу цели (HIGH/LOW)
и тест каузальных коррекций.

Каузальный контракт: все коррекционные коэффициенты оцениваются ТОЛЬКО на
errors[:step]. Данные из [step+1..] алгоритм не видит.

4-way: LWR(p=3,K=50) + Simplex-log(p=8,k=9) + S-map(θ=1,p=3) + RBF-log(p=3,K=12)
Веса: α_lwr=0.05, α_sx=0.35, α_sm=0.20, α_rbf=0.40  → rMAE=0.3963 (эталон)

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

REF_4WAY = 0.3963

MIN_N_CORR = 5    # минимум per-type шагов для rolling/mult
MIN_N_OLS  = 20   # минимум per-type шагов для OLS

ROLL_WINDOWS = [5, 10, 20, 50, None]  # None = все прошлые


# ─────────────────────────────────────────────────────────────────────────────

def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    """Возвращает (vals, dates, types) где types[i]=1 HIGH, -1 LOW."""
    vals, dts, ptypes = [], [], []
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
                vals.append(ext_val); dts.append(dates[ext_idx]); ptypes.append(1)
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx]); ptypes.append(-1)
                direction = 1; ext_val, ext_idx = highs[i], i
    if not vals:
        return np.array([]), np.array([]), np.array([], dtype=int)
    return np.array(vals), np.array(dts), np.array(ptypes, dtype=int)


def build_X(prices, p):
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def rmae_full(preds, actuals):
    preds = np.asarray(preds, dtype=float)
    dz = np.mean(np.abs(np.diff(actuals)))
    return float(np.mean(np.abs(preds - actuals)) / dz) if dz > 1e-12 else np.nan


# ─── Каузальные коррекции ────────────────────────────────────────────────────

def roll_corr(pred, ttype, past_errors, W):
    errs = past_errors[ttype]
    if len(errs) < MIN_N_CORR:
        return pred
    window = errs[-W:] if W else errs
    return pred - float(np.mean(window))


def roll_corr_global(pred, past_errors_all, W):
    if len(past_errors_all) < MIN_N_CORR:
        return pred
    window = past_errors_all[-W:] if W else past_errors_all
    return pred - float(np.mean(window))


def ols_corr(pred, ttype, past_pairs):
    pairs = past_pairs[ttype]
    if len(pairs) < MIN_N_OLS:
        return pred
    preds_p = np.array([p for p, a in pairs])
    acts_p  = np.array([a for p, a in pairs])
    if preds_p.std() < 1e-6:
        return pred
    A = np.column_stack([np.ones(len(preds_p)), preds_p])
    c, *_ = np.linalg.lstsq(A, acts_p, rcond=None)
    return float(c[0] + c[1] * pred)


def logmult_corr(pred, ttype, past_logratios):
    lrs = past_logratios[ttype]
    if len(lrs) < MIN_N_CORR or pred < 1e-6:
        return pred
    return pred * float(np.exp(np.mean(lrs)))


# ─────────────────────────────────────────────────────────────────────────────

def run():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")
    p1d,  dt1d,  types1d = find_pivots(h1d,  l1d,  d1d,  T_1D)
    p10m, dt10m, _        = find_pivots(h10m, l10m, d10m, T_10M)

    n1d = len(p1d); na = len(p10m)
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")

    Xs = {p: (build_X(p1d, p), build_X(p10m, p))
          for p in set([P_LWR, P_SX, P_RBF])}

    print(f"SBER 1d {n1d} пив  10m {na} пив")
    print(f"Веса: α_lwr={A_LWR} α_sx={A_SX} α_sm={A_SM} α_rbf={A_RBF}")
    print(f"Эталон 4-way rMAE: {REF_4WAY}\n")

    # ── Буферы результатов ────────────────────────────────────────────────────
    raw_preds   = []
    raw_actuals = []
    raw_origins = []  # p_cur (origin pivot price)
    raw_types   = []  # тип ЦЕЛИ: types1d[step+H]

    corr_keys = (
        [f"roll_W{W if W else 'all'}_bytype" for W in ROLL_WINDOWS] +
        [f"roll_W{W if W else 'all'}_global" for W in ROLL_WINDOWS] +
        ["ols_bytype", "logmult_bytype"]
    )
    corr_preds = {k: [] for k in corr_keys}

    # Каузальная история (только прошлые шаги)
    past_errors    = {1: [], -1: []}
    past_pairs     = {1: [], -1: []}
    past_logratios = {1: [], -1: []}
    past_errors_all = []

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

        actual = float(p1d[step + H])
        err    = pred_ens - actual

        raw_preds.append(pred_ens)
        raw_actuals.append(actual)
        raw_origins.append(p_cur)
        raw_types.append(ttype)

        # ── Применяем коррекции (используя только прошлые данные) ────────────
        for W in ROLL_WINDOWS:
            k_bt = f"roll_W{W if W else 'all'}_bytype"
            k_gl = f"roll_W{W if W else 'all'}_global"
            corr_preds[k_bt].append(roll_corr(pred_ens, ttype, past_errors, W))
            corr_preds[k_gl].append(roll_corr_global(pred_ens, past_errors_all, W))
        corr_preds["ols_bytype"].append(ols_corr(pred_ens, ttype, past_pairs))
        corr_preds["logmult_bytype"].append(logmult_corr(pred_ens, ttype, past_logratios))

        # ── Обновляем каузальную историю ─────────────────────────────────────
        past_errors[ttype].append(err)
        past_pairs[ttype].append((pred_ens, actual))
        if pred_ens > 1e-6:
            past_logratios[ttype].append(np.log(actual / pred_ens))
        past_errors_all.append(err)

    # ── Конвертируем в массивы ────────────────────────────────────────────────
    acts  = np.array(raw_actuals)
    preds = np.array(raw_preds)
    origs = np.array(raw_origins)
    types = np.array(raw_types)
    errs  = preds - acts
    n_steps = len(acts)

    dz_global = float(np.mean(np.abs(np.diff(acts))))  # знаменатель rMAE

    # ─── СЕКЦИЯ 1: Диагностика смещения ──────────────────────────────────────
    print(f"Шагов: {n_steps}")
    print(f"\n{'═'*68}")
    print("СЕКЦИЯ 1: Смещение по типу цели")
    print(f"{'═'*68}")

    swing_pred   = preds - origs   # предсказанный ход от origin
    swing_actual = acts  - origs   # реальный ход от origin

    for ttype, label in [(1, "HIGH-target"), (-1, "LOW-target")]:
        mask = types == ttype
        e  = errs[mask]
        p  = preds[mask]; a = acts[mask]
        sp = swing_pred[mask]; sa = swing_actual[mask]
        count = int(mask.sum())

        t_stat, p_val = stats.ttest_1samp(e, 0)
        # Swing ratio: насколько предсказанный ход относится к реальному
        ratio = np.median(sp / np.where(np.abs(sa) > 1e-6, sa, np.nan))

        print(f"\n{label}  (n={count})")
        print(f"  mean(pred − actual):        {np.mean(e):>+8.2f} руб.")
        print(f"  std(pred − actual):         {np.std(e):>8.2f} руб.")
        print(f"  mean |pred − actual|:       {np.mean(np.abs(e)):>8.2f} руб.")
        print(f"  mean(pred/actual − 1):      {np.mean(p/a - 1)*100:>+7.3f}%")
        print(f"  median swing ratio (p/a):   {ratio:>8.3f}  "
              f"({'занижает' if ratio < 1 else 'завышает'} ход)")
        print(f"  t-test bias≠0:  t={t_stat:+.2f}  p={p_val:.4f}  "
              f"{'*** значимо' if p_val < 0.05 else '— незначимо'}")
        print(f"  rMAE (тип):                 {np.mean(np.abs(e))/dz_global:>8.4f}  "
              f"(vs global {rmae_full(preds, acts):.4f})")

    ks_stat, ks_p = stats.ks_2samp(errs[types == 1], errs[types == -1])
    print(f"\nKS-тест HIGH vs LOW: stat={ks_stat:.4f}  p={ks_p:.4f}  "
          f"{'— разные распределения' if ks_p < 0.05 else '— схожие распределения'}")

    print(f"\n{'─'*40}")
    print("Автокорреляция ошибок (лаги 1–5):")
    for lag in range(1, 6):
        corr = float(np.corrcoef(errs[:-lag], errs[lag:])[0, 1])
        print(f"  lag {lag}: r={corr:+.4f}")

    # ─── СЕКЦИЯ 2: rMAE коррекций ─────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print("СЕКЦИЯ 2: rMAE каузальных коррекций")
    print(f"{'═'*68}")
    raw_r = rmae_full(preds, acts)
    print(f"\n{'Метод':<38}  {'rMAE':>7}  {'Δ vs raw':>9}")
    print(f"{'─'*58}")
    print(f"{'raw (4-way ансамбль)':<38}  {raw_r:>7.4f}  {'0.00%':>9}")

    results_table = []
    for key in corr_keys:
        cp = np.array(corr_preds[key], dtype=float)
        r  = rmae_full(cp, acts)
        delta = (r / raw_r - 1) * 100
        label = key.replace("_", " ")
        print(f"{label:<38}  {r:>7.4f}  {delta:>+8.2f}%")
        results_table.append((key, r, delta))

    best_key, best_r, best_d = min(results_table, key=lambda x: x[1])
    print(f"\nЛучший: {best_key}  rMAE={best_r:.4f}  ({best_d:+.2f}% vs raw)")

    # ─── СЕКЦИЯ 3: Графики ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 3.1 Scatter pred vs actual
    ax = axes[0, 0]
    for ttype, color, marker, lbl in [(1, "tomato", "^", "HIGH-target"),
                                       (-1, "steelblue", "v", "LOW-target")]:
        mask = types == ttype
        ax.scatter(acts[mask], preds[mask], alpha=0.25, s=10,
                   color=color, marker=marker, label=lbl)
    lo = np.percentile(np.concatenate([acts, preds]), 1)
    hi = np.percentile(np.concatenate([acts, preds]), 99)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="ideal")
    for ttype, color in [(1, "tomato"), (-1, "steelblue")]:
        mask = types == ttype
        m, b = np.polyfit(acts[mask], preds[mask], 1)
        x_fit = np.linspace(lo, hi, 100)
        ax.plot(x_fit, m*x_fit + b, color=color, lw=1.2, alpha=0.7,
                ls=":", label=f"fit {'+HIGH' if ttype==1 else '-LOW'} b={b:.0f}")
    ax.set_xlabel("actual price"); ax.set_ylabel("predicted price")
    ax.set_title("Scatter: pred vs actual  (по типу цели)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 3.2 Распределение ошибок по типу
    ax = axes[0, 1]
    for ttype, color, lbl in [(1, "tomato", "HIGH-target"), (-1, "steelblue", "LOW-target")]:
        mask = types == ttype
        e = errs[mask]
        ax.hist(e, bins=45, alpha=0.5, color=color,
                label=f"{lbl}  μ={np.mean(e):+.1f}")
        ax.axvline(np.mean(e), color=color, lw=2.0, ls="--")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("pred − actual  (руб.)")
    ax.set_ylabel("count")
    ax.set_title("Распределение ошибок по типу цели")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3.3 Ошибки во времени
    ax = axes[1, 0]
    for ttype, color, lbl in [(1, "tomato", "HIGH"), (-1, "steelblue", "LOW")]:
        mask = types == ttype
        idx  = np.where(mask)[0]
        ax.scatter(idx, errs[mask], alpha=0.25, s=7, color=color, label=lbl)
    # rolling mean по всем
    W_smooth = 20
    roll = np.convolve(errs, np.ones(W_smooth)/W_smooth, mode="valid")
    ax.plot(np.arange(len(roll)) + W_smooth//2, roll,
            color="black", lw=1.5, label=f"MA({W_smooth})")
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.set_xlabel("шаг"); ax.set_ylabel("pred − actual")
    ax.set_title("Ошибки во времени  (+ скользящее среднее)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3.4 rMAE: raw vs лучшие коррекции
    ax = axes[1, 1]
    show = [("raw", raw_r)] + [(k, v) for k, v, _ in results_table]
    bar_labels = [s[0].replace("_", "\n") for s in show]
    bar_vals   = [s[1] for s in show]
    bar_colors = ["red" if v == min(bar_vals) else
                  ("orange" if v <= raw_r else "steelblue") for v in bar_vals]
    ax.barh(range(len(bar_labels)), bar_vals, color=bar_colors, alpha=0.75)
    ax.axvline(raw_r, color="black", ls=":", lw=1.0, label=f"raw={raw_r:.4f}")
    ax.set_yticks(range(len(bar_labels)))
    ax.set_yticklabels(bar_labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("rMAE")
    ax.set_title("rMAE: raw vs коррекции  (оранжевый = улучшение)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(
        f"Анализ смещения 4-way ансамбля | SBER 1d(4%)+10m(0.4%)  H={H}  n={n_steps}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = OUT / "bias_analysis.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {out_path}")


if __name__ == "__main__":
    run()
