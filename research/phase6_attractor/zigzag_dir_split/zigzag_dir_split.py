"""
Влияние фильтрации пула по направлению пивота на качество зигзаг-прогноза.

6 условий: {BASE, SAME_DIR, OPP_DIR} × {p=2, p=3}

  BASE     — пул содержит все пивоты (текущий стандарт)
  SAME_DIR — пул содержит только пивоты того же типа (хай→предсказываем лой:
             берём обучающие пары где training-пивот тоже хай; аналогично для лоёв)
  OPP_DIR  — пул содержит только пивоты противоположного типа (контроль)

Направление пивота i: direction[i] = "H" (хай) если z[i] > z[i-1], иначе "L" (лой).
Фильтрация применяется ко всем источникам пула: 1d-окно + 1h + 10m.

Метрики: rMAE общий, rMAE_up, rMAE_down, bias, pool_size.
Контроль H3: снижается ли асимметрия bias_up/bias_down при SAME_DIR?

Параметры: theta=8, T=2%, TRAIN_WIN=200, h=1. Тикер: SBER 1d+1h+10m.
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
CONDITIONS = ["base", "same_dir", "opp_dir"]

# ── Утилиты ──────────────────────────────────────────────────────────────────

def logtrend_causal(close):
    n  = len(close)
    lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    N  = np.arange(1, n + 1, dtype=np.float64)
    St, Sp   = np.cumsum(t),      np.cumsum(lc)
    St2, Stp = np.cumsum(t ** 2), np.cumsum(t * lc)
    den = N * St2 - St ** 2
    b   = np.where(den > 1e-12, (N * Stp - St * Sp) / den, 0.0)
    a   = (Sp - b * St) / N
    trd = np.exp(a + b * t); trd[:2] = close[:2]
    return trd


def find_pivots(ratio, thr=THRESH):
    pivots = [0]; direction = 0; ev, ei = ratio[0], 0
    for i in range(1, len(ratio)):
        v = ratio[i]
        if direction == 0:
            if abs(v - ev) >= thr * ev:
                direction = 1 if v > ev else -1; ev, ei = v, i
        elif direction == 1:
            if v > ev: ev, ei = v, i
            elif (ev - v) >= thr * ev:
                pivots.append(ei); direction = -1; ev, ei = v, i
        else:
            if v < ev: ev, ei = v, i
            elif (v - ev) >= thr * ev:
                pivots.append(ei); direction = 1; ev, ei = v, i
    return np.array(pivots, dtype=int)


def pivot_directions(z):
    """
    direction[i] = 'H' (хай) если z[i] > z[i-1], 'L' (лой) иначе.
    direction[0] = 'H' условно (нет предыдущего).
    """
    d = np.where(z[1:] > z[:-1], 'H', 'L')
    return np.concatenate([['H'], d])


def make_X(z, p):
    """X[i] = [z[i], z[i]-z[i-1], ..., z[i-p+2]-z[i-p+1]], dim=p."""
    n = len(z)
    X = np.full((n, p), np.nan)
    for i in range(p, n):
        X[i, 0] = z[i]
        for k in range(1, p):
            X[i, k] = z[i - k + 1] - z[i - k]
    return X


def load_series(tf):
    with open(DATA / f"{tf}.json") as f:
        data = json.load(f)
    close = np.array([d["close"] for d in data], dtype=np.float64)
    dates = np.array([d["begin"] for d in data])
    trend = logtrend_causal(close)
    ratio = close / trend
    piv   = find_pivots(ratio)
    z     = ratio[piv]
    dirs  = pivot_directions(z)
    return z, dirs, dates[piv]


# ── S-map ────────────────────────────────────────────────────────────────────

def smap_pred(X_pool, y_pool, x_q, theta=THETA):
    if len(X_pool) < 3:
        return np.nan
    mu    = X_pool.mean(0)
    sigma = X_pool.std(0)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)
    Xn    = (X_pool - mu) / sigma
    xn    = (x_q    - mu) / sigma
    dists = np.sqrt(((Xn - xn) ** 2).sum(1))
    d_bar = dists.mean()
    w = np.ones(len(y_pool)) if (theta == 0.0 or d_bar < 1e-12) \
        else np.exp(-theta * dists / d_bar)
    ws   = np.sqrt(w)
    A    = np.column_stack([np.ones(len(y_pool)), Xn]) * ws[:, None]
    b    = y_pool * ws
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(coef[0] + coef[1:] @ xn)


# ── Построение пулов ─────────────────────────────────────────────────────────

def collect_pool_entries(step, ts, z_prim, X_prim, dirs_prim, dates_prim,
                         aux_list, p):
    """
    Возвращает:
      X_rows — все обучающие векторы (N, p)
      y_rows — таргеты z[j+H]
      d_rows — направление direction[j] для каждой строки ('H'/'L')
    """
    X_rows, y_rows, d_rows = [], [], []

    # 1d-окно
    for j in range(ts + p, step):
        if np.any(np.isnan(X_prim[j])):
            continue
        if j + H >= len(z_prim) or np.isnan(z_prim[j + H]):
            continue
        X_rows.append(X_prim[j])
        y_rows.append(z_prim[j + H])
        d_rows.append(dirs_prim[j])

    # aux: 1h, 10m
    cur_date = dates_prim[step]
    for z_a, X_a, dirs_a, dates_a in aux_list:
        ce = int(np.searchsorted(dates_a, cur_date, side="left"))
        for k in range(p, ce - H):
            if np.any(np.isnan(X_a[k])):
                continue
            if k + H >= len(z_a) or np.isnan(z_a[k + H]):
                continue
            X_rows.append(X_a[k])
            y_rows.append(z_a[k + H])
            d_rows.append(dirs_a[k])

    if not X_rows:
        return None, None, None
    return np.array(X_rows), np.array(y_rows), np.array(d_rows)


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward():
    print("Загрузка данных SBER ...")
    z1d,  dirs1d,  d1d  = load_series("1d")
    z1h,  dirs1h,  d1h  = load_series("1h")
    z10m, dirs10m, d10m = load_series("10m")

    print(f"  1d:{len(z1d)} пивотов  1h:{len(z1h)}  10m:{len(z10m)}")

    # Матрицы признаков для всех p
    X_mats = {}
    for series_key, z_s in [("1d", z1d), ("1h", z1h), ("10m", z10m)]:
        for p in P_LIST:
            X_mats[(series_key, p)] = make_X(z_s, p)

    n1d        = len(z1d)
    test_start = TRAIN_WIN + max(P_LIST)
    test_end   = n1d - H
    print(f"  тест: [{test_start}…{test_end}]  n={test_end - test_start}")

    records = []

    for step in range(test_start, test_end):
        ts         = max(0, step - TRAIN_WIN)
        actual_rat = z1d[step + H]
        dir_step   = dirs1d[step]          # 'H' или 'L'

        rec = {
            "step":       step,
            "actual":     actual_rat,
            "m0":         z1d[step],
            "direction":  dir_step,
        }

        for p in P_LIST:
            x_q = X_mats[("1d", p)][step]
            if np.any(np.isnan(x_q)):
                for cond in CONDITIONS:
                    rec[f"{cond}_p{p}"]      = np.nan
                    rec[f"pool_{cond}_p{p}"] = 0
                continue

            aux_list_p = [
                (z1h,  X_mats[("1h",  p)], dirs1h,  d1h),
                (z10m, X_mats[("10m", p)], dirs10m, d10m),
            ]

            Xp, yp, dp = collect_pool_entries(
                step, ts,
                z1d, X_mats[("1d", p)], dirs1d, d1d,
                aux_list_p, p)

            if Xp is None:
                for cond in CONDITIONS:
                    rec[f"{cond}_p{p}"]      = np.nan
                    rec[f"pool_{cond}_p{p}"] = 0
                continue

            # BASE: весь пул
            rec[f"base_p{p}"]      = smap_pred(Xp, yp, x_q)
            rec[f"pool_base_p{p}"] = len(Xp)

            # SAME_DIR: фильтр по direction == dir_step
            same_mask = (dp == dir_step)
            if same_mask.sum() >= 3:
                rec[f"same_dir_p{p}"] = smap_pred(Xp[same_mask], yp[same_mask], x_q)
            else:
                rec[f"same_dir_p{p}"] = np.nan
            rec[f"pool_same_dir_p{p}"] = same_mask.sum()

            # OPP_DIR: фильтр по direction != dir_step
            opp_mask = ~same_mask
            if opp_mask.sum() >= 3:
                rec[f"opp_dir_p{p}"] = smap_pred(Xp[opp_mask], yp[opp_mask], x_q)
            else:
                rec[f"opp_dir_p{p}"] = np.nan
            rec[f"pool_opp_dir_p{p}"] = opp_mask.sum()

        records.append(rec)

        if step % 100 == 0:
            print(f"  step {step}/{test_end}", flush=True)

    return pd.DataFrame(records)


# ── Аналитика ─────────────────────────────────────────────────────────────────

def analyze(df):
    mean_dz = df["actual"].diff().abs().mean()
    if mean_dz < 1e-12:
        mean_dz = 1.0
    m0_rmae = (df["m0"] - df["actual"]).abs().mean() / mean_dz

    print(f"\n{'='*70}")
    print(f"  SBER  n={len(df)}  M0 rMAE={m0_rmae:.4f}  (mean_dz={mean_dz:.5f})")
    print(f"{'='*70}")

    # ── Общий rMAE и bias ─────────────────────────────────────────────────────
    print(f"\n  {'Условие':16}  {'p':>2}  {'bias':>10}  {'rMAE':>7}  "
          f"{'vs M0':>8}  {'pool_mean':>10}")
    rows_summary = []
    for p in P_LIST:
        print(f"  {'─'*66}")
        for cond in CONDITIONS:
            col = f"{cond}_p{p}"
            s   = df[col].dropna()
            a   = (s - df.loc[s.index, "actual"]).abs()
            if len(s) == 0:
                continue
            bias      = (s - df.loc[s.index, "actual"]).mean()
            rmae      = a.mean() / mean_dz
            pool_mean = df[f"pool_{cond}_p{p}"].mean()
            label     = f"{cond}"
            print(f"  {label:16}  {p:>2}  {bias:>+10.5f}  {rmae:>7.4f}  "
                  f"{(rmae/m0_rmae-1)*100:>+7.1f}%  {pool_mean:>10.0f}")
            rows_summary.append({
                "condition": cond, "p": p, "bias": bias,
                "rMAE": rmae, "vs_m0_pct": (rmae/m0_rmae-1)*100,
                "pool_mean": pool_mean, "n": len(s),
            })

    # ── H3: bias_up / bias_down ───────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  H3: bias по направлению шага")
    print(f"  {'Условие':16}  {'p':>2}  {'bias_H (→лой)':>14}  "
          f"{'bias_L (→хай)':>14}  {'разница':>10}")
    for p in P_LIST:
        print(f"  {'─'*66}")
        for cond in CONDITIONS:
            col     = f"{cond}_p{p}"
            df_v    = df[["direction", "actual", col]].dropna(subset=[col])
            serr    = df_v[col] - df_v["actual"]
            bias_H  = serr[df_v["direction"] == "H"].mean()  # от хая → предсказываем лой
            bias_L  = serr[df_v["direction"] == "L"].mean()  # от лоя → предсказываем хай
            print(f"  {cond:16}  {p:>2}  {bias_H:>+14.5f}  "
                  f"{bias_L:>+14.5f}  {bias_H-bias_L:>+10.5f}")

    # ── Pool size statistics ──────────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  Размер пула: mean / min / <10 (% шагов с пулом < 10 точек)")
    for p in P_LIST:
        print(f"  p={p}:")
        for cond in CONDITIONS:
            pc = df[f"pool_{cond}_p{p}"]
            pct_small = (pc < 10).mean() * 100
            print(f"    {cond:12}  mean={pc.mean():6.0f}  "
                  f"min={pc.min():4.0f}  <10: {pct_small:.1f}%")

    return pd.DataFrame(rows_summary), m0_rmae


# ── График ────────────────────────────────────────────────────────────────────

COND_COLORS = {
    "base":     "steelblue",
    "same_dir": "darkorange",
    "opp_dir":  "gray",
}
COND_LS = {"base": "-", "same_dir": "-", "opp_dir": "--"}


def plot_results(df, df_summary, m0_rmae):
    mean_dz = df["actual"].diff().abs().mean()

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(
        "SBER: фильтрация пула по направлению пивота  "
        f"(θ={THETA}, T={THRESH*100:.0f}%, W={TRAIN_WIN})",
        fontsize=11)

    # 1. rMAE bar (сравнение условий × p)
    ax = axes[0, 0]
    x   = np.arange(len(CONDITIONS))
    w   = 0.35
    p2  = df_summary[df_summary["p"] == 2].set_index("condition")
    p3  = df_summary[df_summary["p"] == 3].set_index("condition")
    b1  = ax.bar(x - w/2, [p2.loc[c, "rMAE"] for c in CONDITIONS],
                 w, label="p=2", color=[COND_COLORS[c] for c in CONDITIONS], alpha=0.9)
    b2  = ax.bar(x + w/2, [p3.loc[c, "rMAE"] for c in CONDITIONS],
                 w, label="p=3", color=[COND_COLORS[c] for c in CONDITIONS], alpha=0.5,
                 hatch="//")
    ax.axhline(m0_rmae, color="black", lw=0.8, ls="--", label=f"M0 {m0_rmae:.3f}")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(CONDITIONS, fontsize=8)
    ax.set_title("rMAE по условиям и p")
    ax.set_ylabel("rMAE"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.2)

    # 2. H3: bias_H vs bias_L
    ax = axes[0, 1]
    for p_idx, p in enumerate(P_LIST):
        x_pos = np.arange(len(CONDITIONS)) + p_idx * 0.3 - 0.15
        bias_H, bias_L = [], []
        for cond in CONDITIONS:
            col  = f"{cond}_p{p}"
            dv   = df[["direction", "actual", col]].dropna(subset=[col])
            serr = dv[col] - dv["actual"]
            bias_H.append(serr[dv["direction"] == "H"].mean())
            bias_L.append(serr[dv["direction"] == "L"].mean())
        marker = "o" if p == 2 else "D"
        ax.plot(CONDITIONS, bias_H, marker=marker, color="salmon",
                ls="-", lw=1.5, label=f"bias_H p={p}")
        ax.plot(CONDITIONS, bias_L, marker=marker, color="steelblue",
                ls="--", lw=1.5, label=f"bias_L p={p}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("H3: bias по направлению (H=хай→лой, L=лой→хай)")
    ax.set_ylabel("mean(pred − actual)")
    ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.2)

    # 3. Rolling rMAE по времени
    ax = axes[0, 2]
    m0_roll = (df["m0"] - df["actual"]).abs().rolling(20, min_periods=5).mean() / mean_dz
    ax.plot(df["step"], m0_roll, color="black", lw=0.9, ls="--",
            alpha=0.5, label="M0")
    for cond in ["base", "same_dir"]:
        for p in P_LIST:
            roll = (df[f"{cond}_p{p}"] - df["actual"]).abs().rolling(
                20, min_periods=5).mean() / mean_dz
            ls   = "-" if p == 2 else ":"
            lw   = 2.0 if p == 2 else 1.5
            ax.plot(df["step"], roll, color=COND_COLORS[cond], ls=ls, lw=lw,
                    label=f"{cond} p={p}", alpha=0.85)
    ax.set_title("Rolling rMAE (w=20)")
    ax.set_xlabel("step"); ax.set_ylabel("rMAE")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.2)

    # 4. Scatter pred vs actual: base p2 vs same_dir p2
    for col_idx, cond in enumerate(["base", "same_dir"]):
        ax = axes[1, col_idx]
        for p, color, marker in [(2, "steelblue", "o"), (3, "darkorange", "D")]:
            col = f"{cond}_p{p}"
            sub = df[[col, "actual"]].dropna()
            if len(sub) == 0:
                continue
            ax.scatter(sub["actual"], sub[col],
                       alpha=0.2, s=10, color=color, marker=marker)
            lo, hi = sub["actual"].min(), sub["actual"].max()
            zf = np.polyfit(sub["actual"], sub[col], 1)
            xs = np.linspace(lo, hi, 50)
            rmae = (sub[col] - sub["actual"]).abs().mean() / mean_dz
            ax.plot(xs, np.polyval(zf, xs), color=color, lw=2,
                    label=f"p={p}  rMAE={rmae:.3f}")
        lo_all = df["actual"].min(); hi_all = df["actual"].max()
        ax.plot([lo_all, hi_all], [lo_all, hi_all], "k--", lw=0.8, label="ideal")
        ax.set_title(f"Pred vs Actual: {cond}")
        ax.set_xlabel("actual ratio"); ax.set_ylabel("pred ratio")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # 5. Pool size distribution: base vs same_dir (p=2)
    ax = axes[1, 2]
    for cond, color in [("base", "steelblue"), ("same_dir", "darkorange"),
                        ("opp_dir", "gray")]:
        sizes = df[f"pool_{cond}_p2"].dropna()
        ax.hist(sizes, bins=40, alpha=0.5, color=color, density=True,
                label=f"{cond} (mean={sizes.mean():.0f})")
    ax.axvline(10, color="red", lw=0.8, ls="--", label="min=10")
    ax.set_title("Распределение размера пула (p=2)")
    ax.set_xlabel("размер пула"); ax.set_ylabel("плотность")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    plt.tight_layout()
    out = OUT / "dir_split_results.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  График: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    df          = walk_forward()
    df_sum, m0  = analyze(df)
    plot_results(df, df_sum, m0)

    df.to_csv(OUT / "dir_split_detail.csv", index=False)
    df_sum.to_csv(OUT / "dir_split_summary.csv", index=False)
    print(f"  CSV: {OUT}")


if __name__ == "__main__":
    main()
