#!/usr/bin/env python3
"""
17j_rbf_test.py — RBF (kernel ridge) на кросс-тикерном пуле: взорвётся или нет?

RBF использовался в фазе 7 раньше (15_method_sweep.py, zigzag_forecast_ref.py)
на T=2-4%, single-ticker — там был ХУДШИМ из 4 методов (LA0<LWR<Simplex<RBF).
На нашем пуле (D_allpeers, тысячи событий) ещё не пробовали.

Гипотеза пользователя: пул стал плотнее — с одной стороны это может помочь
(меньше экстраполяции), с другой навредить (RBF — kernel ridge на K×K грам-
матрице Φ_ij=exp(-||x_i-x_j||²/2σ²) среди K соседей; если K ближайших очень
близки друг к другу — Φ вырождается, регуляризация λ=1e-3 может не спасти).

Считаем и rMAE, и число обусловленности Φ на каждом шаге — чтобы увидеть
сам механизм, а не только его последствие.

Точка: SBER, T_big=20%, ratio=0.85, arm=D_allpeers (как в 17d/17e).
"""
import importlib.util
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
K_GRID = [4, 5, 10, 20, 30, 50]


def _rbf_k(d, feats, tgts, K):
    if len(d) < K:
        return np.nan, np.nan
    knn = np.argpartition(d, K - 1)[:K]
    nn_feats = feats[knn]; nn_tgts = tgts[knn]; nn_dists = d[knn]
    d_max = nn_dists.max()
    if d_max < 1e-12:
        return float(nn_tgts.mean()), 1.0
    sigma2 = 2.0 * d_max ** 2
    diff = nn_feats[:, None, :] - nn_feats[None, :, :]
    Phi = np.exp(-np.sum(diff ** 2, axis=2) / sigma2)
    lam = 1e-3
    cond = float(np.linalg.cond(Phi))
    try:
        w, *_ = np.linalg.lstsq(Phi + lam * np.eye(K), nn_tgts, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan, cond
    k_q = np.exp(-nn_dists ** 2 / sigma2)
    val = float(k_q @ w)
    return (val if np.isfinite(val) else np.nan), cond


def main():
    print("=== 17j_rbf_test ===")
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
        for K in K_GRID:
            lr, cond = _rbf_k(d, feats_d, tgts_d, K)
            row[f"e_RBF_K{K}"] = abs(float(np.exp(cur_lp + lr)) - actual_price) if np.isfinite(lr) else np.nan
            row[f"cond_K{K}"] = cond
        records.append(row)

    df = pd.DataFrame(records)
    dz = float(df["pers_err"].mean())
    print(f"\nШагов: {len(df)}   persistence denom: {dz:.5f}   pool_avg: {df['n_pool'].mean():.1f}\n")

    print(f"{'K':<6} {'rMAE':>8} {'n_valid':>8} {'cond_mean':>12} {'cond_median':>12} {'cond_max':>12}")
    for K in K_GRID:
        v = df[f"e_RBF_K{K}"].dropna()
        r = float(v.mean() / dz) if len(v) > 5 else np.nan
        c = df[f"cond_K{K}"].dropna()
        print(f"{K:<6} {r:>8.4f} {len(v):>8d} {c.mean():>12.2e} {c.median():>12.2e} {c.max():>12.2e}")

    print("\nРеференс из 17d (тот же прогон/точка, другие методы):")
    print("  LA0_K50=0.6432  LWR_K30=0.6389  Simplex_K4=0.6630  Smap(θ=1)=0.6738")

    out = RESULTS / "rbf_test.csv"
    df.to_csv(out, index=False, float_format="%.5f")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
