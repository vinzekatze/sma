#!/usr/bin/env python3
"""
LP-фильтрация ряда пивотов для зигзаг-S-map.

14 условий: {base} × p∈{2,3}  +  {att, base_res, att_res} × LP∈{A,B} × p∈{2,3}

==============================================================================
CAUSALITY CONTRACT
──────────────────
step = индекс текущего (последнего известного) пивота в z_prim.

PRIMARY (1d):
  att_z_prim[i] ← causal LP: соседи строго j < i (исторические окна до i).
  Тренировочный пул: j ∈ [pool_start, step-1]  (строго j < step).
  Признаки при j: из z_prim[0..j] и att_z_prim[0..j] — оба каузальны.
  Целевая при j: z_prim[j + H].

AUX (1h, 10m):
  att_z_aux[i] ← causal LP: соседи строго j < i.
  Граница пула: ce = searchsorted(dates_aux, dates_prim[step], side="left").
  Записи пула: j ∈ [pool_start, ce - H - 1].

НИ ОДНА функция не получает данных за пределами этих границ.
==============================================================================
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
OUT.mkdir(parents=True, exist_ok=True)

THRESH    = 0.02
TRAIN_WIN = 200
THETA     = 8.0
H         = 1
P_LIST    = [2, 3]

# LP конфигурации (m, d, k, n_iter)
LP_CONFIGS = {
    "A": dict(m=5, d=1, k=25, n_iter=1),   # theory: FNN p_min=5, d_A≈2 → m=5, d=1
    "B": dict(m=7, d=2, k=40, n_iter=1),   # расширенная: m=2*d_A+3, d=d_A
}

TF_LIST = ["1d", "1h", "10m"]


# ── Утилиты: данные ───────────────────────────────────────────────────────────

def logtrend_causal(close):
    n  = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    N  = np.arange(1, n + 1, dtype=np.float64)
    St, Sp   = np.cumsum(t),      np.cumsum(lc)
    St2, Stp = np.cumsum(t**2),   np.cumsum(t * lc)
    den = N * St2 - St**2
    b   = np.where(den > 1e-12, (N * Stp - St * Sp) / den, 0.0)
    a   = (Sp - b * St) / N
    trd = np.exp(a + b * t);  trd[:2] = close[:2]
    return trd


def find_pivots(ratio, thr=THRESH):
    pivots = [0];  direction = 0;  ev, ei = ratio[0], 0
    for i in range(1, len(ratio)):
        v = ratio[i]
        if direction == 0:
            if abs(v - ev) >= thr * ev:
                direction = 1 if v > ev else -1;  ev, ei = v, i
        elif direction == 1:
            if v > ev:  ev, ei = v, i
            elif (ev - v) >= thr * ev:
                pivots.append(ei);  direction = -1;  ev, ei = v, i
        else:
            if v < ev:  ev, ei = v, i
            elif (v - ev) >= thr * ev:
                pivots.append(ei);  direction = 1;  ev, ei = v, i
    return np.array(pivots, dtype=int)


def load_series(tf):
    with open(DATA / f"{tf}.json") as f:
        data = json.load(f)
    close  = np.array([d["close"] for d in data], dtype=np.float64)
    dates  = np.array([d["begin"] for d in data])
    trend  = logtrend_causal(close)
    ratio  = close / trend
    piv    = find_pivots(ratio)
    z      = ratio[piv]
    return z, dates[piv]


# ── Causal LP фильтр ──────────────────────────────────────────────────────────

def causal_lp(z, m, d, k, n_iter=1):
    """
    Local Projective фильтр, каузальный.
    att[i] = проекция окна z[i-m+1..i] на d-мерное подпространство,
    найденное по k ближайшим историческим окнам (start < текущего).
    Первые m-1 позиций остаются равными z (нет истории для проекции).
    """
    n    = len(z)
    d    = min(d, m - 1)
    att  = z.astype(np.float64).copy()
    if d < 1 or n < m + 1:
        return att

    for _ in range(n_iter):
        # sliding_window_view: (nw, m), nw = n - m + 1
        X_all   = np.lib.stride_tricks.sliding_window_view(att, m)  # (nw, m)
        nw      = len(X_all)
        new_att = att.copy()

        for q in range(1, nw):
            X_hist = X_all[:q]                  # окна с start < q (строго каузальные)
            curr   = X_all[q]
            k_eff  = min(k, len(X_hist))
            if k_eff < d + 1:
                continue
            dists = np.linalg.norm(X_hist - curr, axis=1)
            sel   = np.argpartition(dists, k_eff - 1)[:k_eff]
            nn    = X_hist[sel]
            center = nn.mean(0)
            _, _, Vt = np.linalg.svd(nn - center, full_matrices=False)
            proj   = center + Vt[:d].T @ (Vt[:d] @ (curr - center))
            new_att[q + m - 1] = proj[-1]       # каузальное обновление: только последняя позиция

        att = new_att

    return att


# ── Признаки ──────────────────────────────────────────────────────────────────

def make_X(z, att_z, variant, p):
    """
    Матрица признаков (n, dim).

    variant:
      'base'     — [z[i], Δz, Δ²z, ...]
      'att'      — [att[i], Δatt, Δ²att, ...]
      'base_res' — [z[i], Δz, ..., res[i]]      res = z - att
      'att_res'  — [att[i], Δatt, ..., res[i]]
    """
    n   = len(z)
    res = z - att_z
    sig = att_z if variant.startswith("att") else z
    add_res = variant.endswith("res")
    dim = p + (1 if add_res else 0)

    X = np.full((n, dim), np.nan)
    for i in range(p, n):
        s = sig[i - p + 1: i + 1]
        if np.any(np.isnan(s)):
            continue
        X[i, 0] = sig[i]
        for lag in range(1, p):
            X[i, lag] = sig[i - lag + 1] - sig[i - lag]
        if add_res:
            if np.isnan(res[i]):
                continue
            X[i, p] = res[i]
    return X


# ── S-map (с внутренней нормализацией пула) ───────────────────────────────────

def smap_pred(X_pool, y_pool, x_q, theta=THETA):
    if len(X_pool) < 3:
        return np.nan
    mu    = X_pool.mean(0)
    sigma = X_pool.std(0)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)
    Xn    = (X_pool - mu) / sigma
    xn    = (x_q    - mu) / sigma
    dists = np.linalg.norm(Xn - xn, axis=1)
    d_bar = dists.mean()
    if d_bar < 1e-12:
        return float(np.nanmean(y_pool))
    w  = np.exp(-theta * dists / d_bar)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(len(y_pool)), Xn]) * ws[:, None]
    b  = y_pool * ws
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0] + coef[1:] @ xn)


# ── Walk-forward ───────────────────────────────────────────────────────────────

def run():
    print("Загрузка данных SBER ...")
    series = {}
    for tf in TF_LIST:
        z, dates = load_series(tf)
        series[tf] = (z, dates)
        print(f"  {tf}: {len(z)} пивотов")

    z1d,  dates1d  = series["1d"]
    z1h,  dates1h  = series["1h"]
    z10m, dates10m = series["10m"]

    # Causal LP для каждого TF и каждой конфигурации
    att = {}    # att[(tf, lp_name)] = att_z array
    for lp_name, cfg in LP_CONFIGS.items():
        print(f"\nLP-{lp_name} (m={cfg['m']} d={cfg['d']} k={cfg['k']}):")
        for tf in TF_LIST:
            z_tf = series[tf][0]
            att_z = causal_lp(z_tf, cfg["m"], cfg["d"], cfg["k"], cfg["n_iter"])
            att[(tf, lp_name)] = att_z
            nan_frac = np.isnan(att_z).mean()
            diff_max = np.nanmax(np.abs(att_z - z_tf))
            print(f"  {tf}: NaN={nan_frac:.1%}  max|att-z|={diff_max:.5f}")

    # Матрицы признаков: (tf, variant, lp_name, p)
    # variant 'base' → att_z = z (нет LP-сглаживания)
    Xmats = {}
    variants_with_lp = ["att", "base_res", "att_res"]
    for tf in TF_LIST:
        z_tf = series[tf][0]
        for p in P_LIST:
            # BASE: att_z_dummy = z (без LP)
            Xmats[(tf, "base", "none", p)] = make_X(z_tf, z_tf, "base", p)
            for lp_name in LP_CONFIGS:
                att_z = att[(tf, lp_name)]
                for var in variants_with_lp:
                    Xmats[(tf, var, lp_name, p)] = make_X(z_tf, att_z, var, p)

    # Условия: список (variant, lp_name, p)
    conditions = []
    for p in P_LIST:
        conditions.append(("base", "none", p))
    for lp_name in LP_CONFIGS:
        for var in variants_with_lp:
            for p in P_LIST:
                conditions.append((var, lp_name, p))

    n1d        = len(z1d)
    test_start = TRAIN_WIN + max(P_LIST)
    test_end   = n1d - H
    n_test     = test_end - test_start
    print(f"\n  тест: [{test_start}…{test_end-1}]  n={n_test}")
    print(f"  условий: {len(conditions)}\n")

    records = []

    for step in range(test_start, test_end):
        if step % 100 == 0:
            print(f"  step {step}/{test_end-1}")

        actual    = z1d[step + H]
        cur_date  = dates1d[step]
        pool_start = step - TRAIN_WIN      # каузальное окно TRAIN_WIN пивотов

        rec = {"step": step, "actual": actual, "m0": z1d[step]}

        for (var, lp_name, p) in conditions:
            cond_key = f"{var}_{lp_name}_p{p}"

            # Запрос (query)
            x_q = Xmats[("1d", var, lp_name, p)][step]
            if np.any(np.isnan(x_q)):
                rec[cond_key] = np.nan
                continue

            # Пул: primary (1d)
            X_rows, y_rows = [], []
            X1d = Xmats[("1d", var, lp_name, p)]
            for j in range(max(pool_start, p), step):     # строго j < step
                if np.any(np.isnan(X1d[j])):
                    continue
                if j + H >= n1d:
                    continue
                X_rows.append(X1d[j])
                y_rows.append(z1d[j + H])

            # Пул: aux (1h, 10m)
            for tf_aux, dates_aux in [("1h", dates1h), ("10m", dates10m)]:
                z_aux   = series[tf_aux][0]
                X_aux   = Xmats[(tf_aux, var, lp_name, p)]
                ce = int(np.searchsorted(dates_aux, cur_date, side="left"))
                for j in range(p, ce - H):
                    if j + H >= len(z_aux):
                        break
                    if np.any(np.isnan(X_aux[j])):
                        continue
                    X_rows.append(X_aux[j])
                    y_rows.append(z_aux[j + H])

            if len(X_rows) < 3:
                rec[cond_key] = np.nan
                continue

            pred = smap_pred(np.array(X_rows), np.array(y_rows), x_q)
            rec[cond_key] = pred

        records.append(rec)

    df = pd.DataFrame(records)

    # ── Метрики ───────────────────────────────────────────────────────────────
    actual_arr = df["actual"].values
    mean_dz    = float(np.nanmean(np.abs(np.diff(actual_arr))))

    m0_rmae = float(np.nanmean(np.abs(df["m0"].values - actual_arr))) / mean_dz

    print(f"\n{'='*72}")
    print(f"  SBER  n={n_test}  M0 rMAE={m0_rmae:.4f}  (mean_dz={mean_dz:.5f})")
    print(f"{'='*72}\n")
    print(f"  {'Условие':<30}  {'p':>2}  {'bias':>9}  {'rMAE':>7}  {'vs base':>8}")
    print(f"  {'─'*70}")

    # base reference per p
    base_rmae = {}
    for p in P_LIST:
        ckey = f"base_none_p{p}"
        errs = df[ckey].values - actual_arr
        base_rmae[p] = float(np.nanmean(np.abs(errs))) / mean_dz

    summary = []
    for (var, lp_name, p) in conditions:
        ckey  = f"{var}_{lp_name}_p{p}"
        errs  = df[ckey].values - actual_arr
        rmae  = float(np.nanmean(np.abs(errs))) / mean_dz
        bias  = float(np.nanmean(errs))
        vs_b  = (rmae - base_rmae[p]) / base_rmae[p] * 100
        label = f"{var}/{lp_name}"
        print(f"  {label:<30}  {p:>2}  {bias:>+9.5f}  {rmae:>7.4f}  {vs_b:>+7.1f}%  {'':>6}")
        summary.append({"variant": var, "lp": lp_name, "p": p,
                         "rMAE": rmae, "bias": bias, "vs_base_pct": vs_b})

    # ── CSV ───────────────────────────────────────────────────────────────────
    df.to_csv(OUT / "lp_filter_preds.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "lp_filter_summary.csv", index=False)
    print(f"\n  CSV: {OUT}")

    # ── График: rMAE по условиям ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("LP-фильтр ряда пивотов: rMAE по условиям (SBER 1d+1h+10m)", fontsize=11)

    colors = {"none": "black", "A": "steelblue", "B": "darkorange"}
    markers = {"base": "o", "att": "s", "base_res": "^", "att_res": "D"}
    labels  = {"base": "base", "att": "att", "base_res": "base+res", "att_res": "att+res"}

    for ax_idx, p in enumerate(P_LIST):
        ax = axes[ax_idx]
        ax.axhline(base_rmae[p], color="black", lw=1.2, ls="--", label=f"base p={p}")
        ax.axhline(m0_rmae, color="gray", lw=0.8, ls=":", label="M0")

        for s in summary:
            if s["p"] != p:
                continue
            c = colors[s["lp"]]
            m = markers[s["variant"]]
            ax.scatter(labels[s["variant"]], s["rMAE"],
                       color=c, marker=m, s=80, zorder=5,
                       label=f"{labels[s['variant']]}/{s['lp']}")

        # Убрать дублирующиеся легенды
        handles, lbls = ax.get_legend_handles_labels()
        seen = {}
        for h, l in zip(handles, lbls):
            if l not in seen:
                seen[l] = h
        ax.legend(seen.values(), seen.keys(), fontsize=7.5, loc="upper right")
        ax.set_title(f"p={p}")
        ax.set_ylabel("rMAE")
        ax.set_xlabel("вариант признаков")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fig_path = OUT / "lp_filter_rmae.png"
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  График: {fig_path}")

    # ── Rolling rMAE ──────────────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig2.suptitle("Rolling rMAE (w=30): LP vs base  (SBER)", fontsize=11)
    roll_colors = {
        "base_none": "black",
        "att_A": "steelblue", "att_B": "royalblue",
        "base_res_A": "darkorange", "base_res_B": "orange",
        "att_res_A": "#2ca02c", "att_res_B": "limegreen",
    }
    for ax_idx, p in enumerate(P_LIST):
        ax = axes2[ax_idx]
        steps = df["step"].values
        for (var, lp_name, _p) in conditions:
            if _p != p:
                continue
            ckey  = f"{var}_{lp_name}_p{p}"
            errs  = np.abs(df[ckey].values - actual_arr)
            roll  = pd.Series(errs).rolling(30, min_periods=10).mean() / mean_dz
            clr_key = f"{var}_{lp_name}"
            c = roll_colors.get(clr_key, "gray")
            lw = 2.0 if lp_name == "none" else 1.4
            ax.plot(steps, roll, color=c, lw=lw, alpha=0.85,
                    label=f"{var}/{lp_name}")
        ax.axhline(m0_rmae, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.set_ylabel(f"rolling rMAE  p={p}")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(alpha=0.2)
    axes2[-1].set_xlabel("step (пивот)")
    plt.tight_layout()
    fig2_path = OUT / "lp_filter_rolling.png"
    plt.savefig(fig2_path, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"  График: {fig2_path}")
    print("\nГотово.")


if __name__ == "__main__":
    run()
