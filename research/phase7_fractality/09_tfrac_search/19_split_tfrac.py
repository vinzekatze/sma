#!/usr/bin/env python3
"""
19_split_tfrac.py — Проверка split T_frac: up=2.5% vs down=3.0%.

Сравниваем два режима на walk-forward (SBER 10m, T_BIG=4%, M=2, K=50, H=1):
  baseline : единый T_frac=3.0% для всех точек
  split    : up (LOW pivot)  → T_frac=2.5%
             down (HIGH pivot) → T_frac=3.0%

Метрика: rMAE total / rMAE_up / rMAE_down.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

HERE    = Path(__file__).parent
DATA    = HERE.parent.parent.parent / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

T_BIG  = 0.04
M      = 2
K      = 50
H      = 1
MIN_HISTORY = 20

TF_BASELINE = 0.030
TF_UP       = 0.025   # oracle медиана для up
TF_DOWN     = 0.030   # oracle медиана для down


# ── данные / зигзаг / пул / LWR — те же что в скр.18 ─────────────────────────

def load_candles(path):
    with open(path) as f:
        raw = json.load(f)
    lh = np.log(np.array([c["high"]  for c in raw], dtype=np.float64))
    ll = np.log(np.array([c["low"]   for c in raw], dtype=np.float64))
    dt = np.array([c["begin"] for c in raw])
    return lh, ll, dt


def build_zigzag(lh, ll, dt, thr):
    lp, cd, dirs = [], [], []
    cur_dir = 0
    ext     = (lh[0] + ll[0]) / 2.0
    for i in range(len(lh)):
        if cur_dir == 0:
            if lh[i] - ext >= thr:
                cur_dir, ext = 1, lh[i]
            elif ext - ll[i] >= thr:
                cur_dir, ext = -1, ll[i]
        elif cur_dir == 1:
            if lh[i] > ext:
                ext = lh[i]
            elif ext - ll[i] >= thr:
                lp.append(ext); cd.append(dt[i]); dirs.append(+1)
                cur_dir, ext = -1, ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= thr:
                lp.append(ext); cd.append(dt[i]); dirs.append(-1)
                cur_dir, ext = 1, lh[i]
    return np.array(lp), np.array(cd), np.array(dirs, dtype=np.int8)


def build_pool(lp, cd, dirs, m):
    rows, tgts, row_dirs, row_dates = [], [], [], []
    for j in range(m, len(lp) - 1):
        feat = np.array([lp[j - lag] - lp[j - lag - 1] for lag in range(m)])
        if not np.all(np.isfinite(feat)):
            continue
        tgt = lp[j + 1] - lp[j]
        if not np.isfinite(tgt):
            continue
        rows.append(feat)
        tgts.append(tgt)
        row_dirs.append(dirs[j])
        row_dates.append(cd[j])
    if not rows:
        return None
    return (np.array(rows), np.array(tgts),
            np.array(row_dirs, dtype=np.int8), np.array(row_dates))


def lwr_predict(qv, fm, tgt, indices):
    feats = fm[indices]
    dists = np.linalg.norm(feats - qv, axis=1)
    d_max = dists.max()
    if d_max < 1e-12:
        return float(tgt[indices].mean())
    w  = np.exp(-0.5 * (dists / d_max) ** 2)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(len(indices)), feats]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, tgt[indices] * ws, rcond=None)
    return float(c[0] + c[1:] @ qv)


def rmae(errs, acts_full, acts_sub=None):
    """rMAE: знаменатель — persistence baseline по полному ряду acts_full."""
    a = np.asarray(acts_full, dtype=float)
    pers = np.abs(a[2:] - a[:-2]) if len(a) >= 3 else np.abs(np.diff(a))
    dz = float(np.mean(pers)) if len(pers) > 0 else 1.0
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


# ── walk-forward ──────────────────────────────────────────────────────────────

def run(lp_big, cd_big, dir_big, pools, tf_map):
    """
    tf_map: dict direction → T_frac
      например {-1: 0.025, +1: 0.030}
    Возвращает DataFrame с колонками: step, direction, err, actual
    """
    records = []
    n_big   = len(lp_big)

    for step in range(max(MIN_HISTORY, M), n_big - H):
        qv = np.array([lp_big[step - lag] - lp_big[step - lag - 1]
                       for lag in range(M)])
        if not np.all(np.isfinite(qv)):
            continue

        qdir      = int(dir_big[step])
        tf        = tf_map[qdir]
        actual_lp = lp_big[step + H]

        fm, tgt, dirs, pool_dates = pools[tf]
        ce = int(np.searchsorted(pool_dates, cd_big[step], side="left"))
        if ce < K + 1:
            continue

        fm_s   = fm[:ce]
        tgt_s  = tgt[:ce]
        dirs_s = dirs[:ce]

        mask     = dirs_s == qdir
        cand_idx = np.where(mask)[0]
        if len(cand_idx) < K:
            continue

        dists   = np.linalg.norm(fm_s[cand_idx] - qv, axis=1)
        top_k   = cand_idx[np.argpartition(dists, K - 1)[:K]]
        pred_lr = lwr_predict(qv, fm_s, tgt_s, top_k)
        err     = np.exp(lp_big[step] + pred_lr) - np.exp(actual_lp)

        records.append({
            "step":      step,
            "direction": qdir,
            "err":       err,
            "actual":    float(np.exp(actual_lp)),
        })

    return pd.DataFrame(records)


def report(label, df):
    up_df   = df[df["direction"] == -1]
    down_df = df[df["direction"] == +1]

    acts_all = df["actual"].values
    a = np.asarray(acts_all, dtype=float)
    pers = np.abs(a[2:] - a[:-2]) if len(a) >= 3 else np.abs(np.diff(a))
    dz = float(np.mean(pers)) if len(pers) > 0 else 1.0

    r_total = float(np.mean(np.abs(df["err"].values))       / dz)
    r_up    = float(np.mean(np.abs(up_df["err"].values))    / dz)
    r_down  = float(np.mean(np.abs(down_df["err"].values))  / dz)

    print(f"  {label:<12}  rMAE={r_total:.4f}  "
          f"up={r_up:.4f} (n={len(up_df)})  "
          f"down={r_down:.4f} (n={len(down_df)})")
    return r_total, r_up, r_down


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    lh, ll, dt = load_candles(DATA / "10m.json")

    lp_big, cd_big, dir_big = build_zigzag(lh, ll, dt, T_BIG)
    print(f"T_BIG={T_BIG*100:.0f}%  пивотов: {len(lp_big)}\n")

    # Строим пулы только для нужных T_frac
    needed = {TF_BASELINE, TF_UP, TF_DOWN}
    pools  = {}
    for tf in sorted(needed):
        lp_f, cd_f, dir_f = build_zigzag(lh, ll, dt, tf)
        pools[tf] = build_pool(lp_f, cd_f, dir_f, M)
        print(f"  T_frac={tf*100:.1f}%  строк пула: {len(pools[tf][0])}")

    print()

    # Baseline: единый T_frac=3.0%
    df_base  = run(lp_big, cd_big, dir_big, pools,
                   {-1: TF_BASELINE, +1: TF_BASELINE})

    # Split: up=2.5%, down=3.0%
    df_split = run(lp_big, cd_big, dir_big, pools,
                   {-1: TF_UP, +1: TF_DOWN})

    # Честное сравнение: только step-ы где оба режима дали прогноз
    common = set(df_base["step"]) & set(df_split["step"])
    df_base  = df_base[df_base["step"].isin(common)].sort_values("step").reset_index(drop=True)
    df_split = df_split[df_split["step"].isin(common)].sort_values("step").reset_index(drop=True)
    print(f"Общих точек: {len(common)}  "
          f"(baseline={len(df_base)}, split={len(df_split)})\n")

    print("Результаты:")
    r_base_tot,  r_base_up,  r_base_down  = report("baseline", df_base)
    r_split_tot, r_split_up, r_split_down = report("split",    df_split)

    def delta(a, b):
        return f"{(b - a) / a * 100:+.2f}%"

    print(f"\n  delta total : {delta(r_base_tot,  r_split_tot)}")
    print(f"  delta up    : {delta(r_base_up,   r_split_up)}")
    print(f"  delta down  : {delta(r_base_down, r_split_down)}")

    # Сохраняем сводку
    summary = pd.DataFrame([
        {"mode": "baseline", "tf_up": TF_BASELINE, "tf_down": TF_BASELINE,
         "rMAE_total": r_base_tot,  "rMAE_up": r_base_up,  "rMAE_down": r_base_down},
        {"mode": "split",    "tf_up": TF_UP,       "tf_down": TF_DOWN,
         "rMAE_total": r_split_tot, "rMAE_up": r_split_up, "rMAE_down": r_split_down},
    ])
    summary.to_csv(RESULTS / "split_tfrac_result.csv", index=False)

    # График
    labels   = ["baseline\n(3.0% / 3.0%)", "split\n(up=2.5% / down=3.0%)"]
    rmae_tot = [r_base_tot,  r_split_tot]
    rmae_up  = [r_base_up,   r_split_up]
    rmae_dn  = [r_base_down, r_split_down]

    x  = np.arange(2)
    w  = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, rmae_tot, w, label="total",    color="#555555")
    ax.bar(x,     rmae_up,  w, label="up only",  color="steelblue")
    ax.bar(x + w, rmae_dn,  w, label="down only", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("rMAE")
    ax.set_title("Baseline vs Split T_frac  (SBER 10m, T_big=4%)", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (tot, up, dn) in enumerate(zip(rmae_tot, rmae_up, rmae_dn)):
        ax.text(i - w, tot + 0.002, f"{tot:.4f}", ha="center", fontsize=8)
        ax.text(i,     up  + 0.002, f"{up:.4f}",  ha="center", fontsize=8)
        ax.text(i + w, dn  + 0.002, f"{dn:.4f}",  ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "split_tfrac_bar_10m.png", dpi=150)
    plt.close(fig)
    print("\nГрафик → results/split_tfrac_bar_10m.png")


if __name__ == "__main__":
    main()
