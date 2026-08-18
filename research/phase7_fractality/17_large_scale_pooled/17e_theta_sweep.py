#!/usr/bin/env python3
"""
17e_theta_sweep.py — свип θ для S-map на той же подтверждённой точке
(SBER, T_big=20%, ratio=0.85, arm=D_allpeers), что и K-свип в 17d.

Мотивация: θ — гладкий аналог K (вес exp(-θ·d/mean(d)) по всему пулу вместо
жёсткого K-cutoff), и он уже отчасти самоадаптивен к плотности пула (нормирован
на mean(d), не на абсолютное расстояние) — в отличие от K. Но при очень
большом θ эффективное число точек с заметным весом может схлопнуться, и тогда
ожидается тот же тип обвала, что убил LWR на K=4 в 17d (S-map — тоже взвешенная
OLS, m=3+intercept=4 параметра).

Переиспользует build_pool_for_step из 17d_k_sweep.py — тот же каузальный
контракт и дедуп, без дублирования.
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent

spec17 = importlib.util.spec_from_file_location("exp17", HERE / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

specd = importlib.util.spec_from_file_location("ksweep", HERE / "17d_k_sweep.py")
ksweep = importlib.util.module_from_spec(specd)
specd.loader.exec_module(ksweep)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TARGET = "SBER"
T_BIG = 0.20
RATIO = 0.85
ARM = "D_allpeers"
THETA_GRID = [0.25, 0.5, 1, 2, 4, 8, 16, 32]

# референс из 17d (тот же прогон, для сравнения в выводе)
REF_BEST = {"LA0_K50": 0.6432, "LWR_K30": 0.6389, "Simplex_K4": 0.6630}


def main():
    print("=== 17e_theta_sweep ===")
    print(f"Target={TARGET}  T_big={T_BIG}  ratio={RATIO}  arm={ARM}  THETA_GRID={THETA_GRID}")

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

    records = []
    for i in range(exp17.MIN_HIST, n_big - 1):
        res = ksweep.build_pool_for_step(target_data, peer_data, rankings, checkpoints, T_BIG, t_frac, ARM, i, full_lp, full_conf)
        if res is None:
            continue
        qvec, feats_d, tgts_d, d, cur_lp = res

        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)

        row = {"step": i, "n_pool": len(tgts_d), "pers_err": pers_err}
        for theta in THETA_GRID:
            lr = exp17._smap(qvec, feats_d, tgts_d, exp17.MIN_POOL_SMAP, theta)
            row[f"e_theta{theta}"] = abs(float(np.exp(cur_lp + lr)) - actual_price) if np.isfinite(lr) else np.nan
        records.append(row)

    df = pd.DataFrame(records)
    dz = float(df["pers_err"].mean())
    print(f"\nШагов: {len(df)}   persistence denom: {dz:.5f}   pool_avg: {df['n_pool'].mean():.1f}\n")

    print(f"{'theta':<8} rMAE")
    summary = {}
    for theta in THETA_GRID:
        v = df[f"e_theta{theta}"].dropna()
        r = float(v.mean() / dz) if len(v) > 5 else np.nan
        summary[f"theta{theta}"] = r
        print(f"{theta:<8} {r:.4f}" if not np.isnan(r) else f"{theta:<8} nan")

    print("\nРеференс из 17d (K-свип, тот же прогон/точка):")
    for name, val in REF_BEST.items():
        print(f"  {name:<12} {val:.4f}")

    out = RESULTS / "theta_sweep.csv"
    df.to_csv(out, index=False, float_format="%.5f")
    pd.DataFrame([summary]).to_csv(RESULTS / "theta_sweep_summary.csv", index=False, float_format="%.5f")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
