#!/usr/bin/env python3
"""
20g_calibrate_with_fixed_lambda.py — пересчёт (m, θ, T_ratio) ИНДИВИДУАЛЬНО
на тикер, с λ=2.0 ЗАФИКСИРОВАННЫМ (не калибруется, константа — LOO-
проверена в 20e/20f: работает на всех 7 тикеров, включая LKOH, лучше
по-фолдово выбранного LOO-λ).

Мотивация: раз λ=2.0 меняет саму целевую функцию взвешенной регрессии,
оптимальные m/θ/T_ratio (безопасные для индивидуальной калибровки —
[[feedback-individual-ticker-calibration]]) могут немного сдвинуться,
чтобы лучше дополнять новую весовую схему. Проверяем: (а) как меняются
конфигурации относительно эксп.17f/17r (без λ), (б) даёт ли
переоткалиброванный (m,θ,T_ratio | λ=2.0) лучший результат, чем просто
"старые m/θ/T_ratio + λ=2.0 сверху" (0.6878 средний, эксп.20 §8/следующая
проверка).
"""
import importlib.util
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

spec20f = importlib.util.spec_from_file_location("exp20f", HERE / "20f_individual_calibrator.py")
exp20f = importlib.util.module_from_spec(spec20f)
spec20f.loader.exec_module(exp20f)   # переиспользуем build_pools/eval_pools/golden/CALIB_BASE

FIXED_LAMBDA = 2.0

REFERENCE_NO_LAMBDA = {   # эксп.17f/17r — для сверки
    "SBER": {"m": 3, "theta": 25.697, "T_ratio": 0.8987, "rMAE": 0.6049},
    "LKOH": {"m": 4, "theta": 24.817, "T_ratio": 0.8331, "rMAE": 0.6961},
    "CHMF": {"m": 3, "theta": 11.492, "T_ratio": 0.9238, "rMAE": 0.6658},
    "NVTK": {"m": 2, "theta": 15.158, "T_ratio": 0.8080, "rMAE": 0.8136},
    "MGNT": {"m": 2, "theta": 11.814, "T_ratio": 0.6110, "rMAE": 0.6479},
    "VTBR": {"m": 3, "theta": 0.731,  "T_ratio": 0.6169, "rMAE": 0.6320},
    "NLMK": {"m": 2, "theta": 2.315,  "T_ratio": 0.8641, "rMAE": 0.7324},
}


def calibrate_fixed_lambda(target, m0, theta0, t_ratio0, lam):
    m, theta, T = m0, theta0, t_ratio0
    trace, prev = [], None
    for outer in range(exp20f.MAX_OUTER):
        best_m, best_v = m, float("inf")
        for mc in exp20f.M_GRID:
            pools = exp20f.build_pools(target, mc, T)
            v, _ = exp20f.eval_pools(pools, theta, lam, mc + 2)
            if v < best_v:
                best_v, best_m = v, mc
        m = best_m

        pools = exp20f.build_pools(target, m, T)
        theta = exp20f.golden(lambda th: exp20f.eval_pools(pools, th, lam, m + 2)[0],
                               exp20f.THETA_LO, exp20f.THETA_HI, exp20f.THETA_TOL)

        T = exp20f.golden(lambda t: exp20f.eval_pools(exp20f.build_pools(target, m, t), theta, lam, m + 2)[0],
                           exp20f.T_LO, exp20f.T_HI, exp20f.T_TOL)

        pools = exp20f.build_pools(target, m, T)
        v, n = exp20f.eval_pools(pools, theta, lam, m + 2)
        trace.append({"iter": outer + 1, "m": m, "theta": round(theta, 3), "T_ratio": round(T, 4), "rMAE": round(v, 4)})
        print(f"    iter {outer+1}: m={m} θ={theta:.3f} T_ratio={T:.4f} (λ={lam} фикс.) → rMAE={v:.4f} (n={n})")
        cur = (m, round(theta, 2), round(T, 3))
        if cur == prev:
            break
        prev = cur
    return m, theta, T, v, n, trace


def main():
    t0 = time.time()
    print(f"=== 20g_calibrate_with_fixed_lambda — (m,θ,T_ratio) при λ={FIXED_LAMBDA} фикс. ===\n")

    results = []
    for target, ref in REFERENCE_NO_LAMBDA.items():
        print(f"--- {target} (референс без λ: m={ref['m']} θ={ref['theta']} T_ratio={ref['T_ratio']} rMAE={ref['rMAE']}) ---")
        t1 = time.time()
        m, theta, T, v, n, trace = calibrate_fixed_lambda(target, ref["m"], ref["theta"], ref["T_ratio"], FIXED_LAMBDA)
        elapsed = time.time() - t1
        print(f"  → m={m} θ={theta:.3f} T_ratio={T:.4f} rMAE={v:.4f} (n={n})  ({elapsed:.1f}s)\n")

        results.append({
            "ticker": target,
            "m_ref": ref["m"], "theta_ref": ref["theta"], "T_ratio_ref": ref["T_ratio"], "rMAE_ref_no_lambda": ref["rMAE"],
            "m_new": m, "theta_new": round(theta, 3), "T_ratio_new": round(T, 4),
            "rMAE_recalibrated_with_lambda": round(v, 4), "n": n, "elapsed_s": round(elapsed, 1),
        })
        pd.DataFrame(results).to_csv(RESULTS / "calibrate_with_fixed_lambda.csv", index=False, float_format="%.4f")

    df = pd.DataFrame(results)
    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}")
    print(f"Среднее: без λ (эксп.17f/17r)={df['rMAE_ref_no_lambda'].mean():.4f}  "
          f"переоткалиброван с λ={FIXED_LAMBDA} фикс.={df['rMAE_recalibrated_with_lambda'].mean():.4f}")
    print(f"Улучшение на: {(df['rMAE_recalibrated_with_lambda'] < df['rMAE_ref_no_lambda']).sum()}/7 тикерах")
    print(f"\nИзменение m: {list(zip(df['m_ref'], df['m_new']))}")

    print(f"\nСохранено: {RESULTS / 'calibrate_with_fixed_lambda.csv'}")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
