#!/usr/bin/env python3
"""
17c_neighbor_distance_diagnostic.py — проверка механизма перед K-свипом:
стали ли K ближайших соседей ФИЗИЧЕСКИ БЛИЖЕ к запросу теперь, когда пул
кросс-тикерный (тысячи событий) вместо own-only (десятки-сотни)?

Не делает прогнозов — только строит те же пулы, что и walk_forward() из
17_large_scale_pooled.py (тот же каузальный контракт, тот же дедуп), и
записывает расстояние до K-го соседа для K∈{4,5,10,20,30,50} на каждом шаге.

Если распределение distance@K для D_allpeers заметно левее (меньше), чем для
B_own_frac — гипотеза "можем позволить себе меньший K" подтверждена
механически, и есть смысл гонять K-sweep. Если распределения совпадают —
пул стал больше, но не факт что ближе (например, если новые точки просто
добавляют шум на тех же расстояниях) — тогда K-sweep менее оправдан.

Фиксировано: SBER, T_big=20%, ratio=0.85 (лучшая подтверждённая точка).
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("exp17", HERE / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp17)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TARGET = "SBER"
T_BIG = 0.20
RATIO = 0.85
ARMS = ["A_baseline", "B_own_frac", "C_top8", "D_allpeers", "E_random8"]
K_GRID = [4, 5, 10, 20, 30, 50]


def build_pool_for_step(target_data, peer_data, rankings, checkpoints, t_big, t_frac, arm, i, full_lp, full_conf):
    """Копия внутренней части walk_forward() до этапа предсказания — возвращает (qvec, feats_d) или None."""
    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    confirm_date = full_conf[i]
    cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))

    t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]
    own_big_lp, _, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_big)
    if len(own_big_lp) < exp17.M + 1 or own_big_lp[-1] != full_lp[i]:
        return None

    qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(exp17.M)])
    if not np.all(np.isfinite(qvec)):
        return None
    q_dir = int(own_big_dir[-1])

    own_big_feats, own_big_tgts, own_big_dirs = exp17.build_pool_rows(own_big_lp, own_big_dir, exp17.M)
    pool_feats = [own_big_feats]; pool_dirs = [own_big_dirs]

    if arm != "A_baseline":
        own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
        f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, exp17.M)
        pool_feats.append(f); pool_dirs.append(dd)

        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, arm):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < exp17.M + 2:
                continue
            p_lh = peer_data[peer]["lh"][:p_cutoff]
            p_ll = peer_data[peer]["ll"][:p_cutoff]
            p_dt = p_dates[:p_cutoff]
            p_lp, _, p_dir = exp17.build_zigzag(p_lh, p_ll, p_dt, t_frac)
            f, tg, dd = exp17.build_pool_rows(p_lp, p_dir, exp17.M)
            pool_feats.append(f); pool_dirs.append(dd)

    feats = np.concatenate(pool_feats) if pool_feats else np.empty((0, exp17.M))
    dirs  = np.concatenate(pool_dirs)  if pool_dirs  else np.empty(0, dtype=np.int8)
    mask = dirs == q_dir
    feats_d = feats[mask]

    if len(feats_d):
        d = np.linalg.norm(feats_d - qvec, axis=1)
        dup_mask = d < exp17.DUP_EPS
        if dup_mask.any():
            feats_d = feats_d[~dup_mask]
            d = d[~dup_mask]
    else:
        d = np.empty(0)

    return d


def main():
    print("=== 17c_neighbor_distance_diagnostic ===")
    print(f"Target={TARGET}  T_big={T_BIG}  ratio={RATIO}  arms={ARMS}  K_GRID={K_GRID}")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, full_dirs = exp17.build_zigzag(lh, ll, dates, T_BIG)
    n_big = len(full_lp)
    t_frac = RATIO * T_BIG

    rows = []
    for arm in ARMS:
        dist_at_k = {k: [] for k in K_GRID}
        pool_sizes = []
        for i in range(exp17.MIN_HIST, n_big - 1):
            d = build_pool_for_step(target_data, peer_data, rankings, checkpoints, T_BIG, t_frac, arm, i, full_lp, full_conf)
            if d is None:
                continue
            pool_sizes.append(len(d))
            d_sorted = np.sort(d)
            for k in K_GRID:
                if len(d_sorted) >= k:
                    dist_at_k[k].append(d_sorted[k - 1])

        row = {"arm": arm, "pool_avg": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
               "n_steps": len(pool_sizes)}
        for k in K_GRID:
            vals = dist_at_k[k]
            row[f"dist@{k}_mean"] = float(np.mean(vals)) if vals else np.nan
            row[f"dist@{k}_median"] = float(np.median(vals)) if vals else np.nan
            row[f"dist@{k}_n"] = len(vals)
        rows.append(row)
        print(f"  {arm:<11s} pool_avg={row['pool_avg']:7.1f}  "
              + "  ".join(f"d@{k}={row[f'dist@{k}_mean']:.4f}" if not np.isnan(row[f'dist@{k}_mean']) else f"d@{k}=nan" for k in K_GRID))
        sys.stdout.flush()

    df = pd.DataFrame(rows)
    out = RESULTS / "neighbor_distance_diagnostic.csv"
    df.to_csv(out, index=False, float_format="%.5f")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
