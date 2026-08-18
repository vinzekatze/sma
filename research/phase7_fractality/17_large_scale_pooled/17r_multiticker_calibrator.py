#!/usr/bin/env python3
"""
17r_multiticker_calibrator.py — калибровка S-map (m, θ, T_ratio) на
D_allpeers-пуле для нескольких целевых тикеров (не только SBER).

Переиспользует calibrate_smap/build_all_pools/eval_smap_pools из
17f_calibrator.py без изменений — меняется только TARGET (пул пиров каждый
раз = UNIVERSE \\ {target}, ранжировка корреляций пересчитывается заново
для целевого тикера).

Мотивация: эксп.17b (Stage 2) проверял воспроизводимость эффекта
кросс-тикерного пула на 7 тикерах, но БЕЗ индивидуальной калибровки (m,θ,T)
— использовались фиксированные дефолты. Здесь — полная калибровка S-map
(единственный метод, определённый как "лучшее, что у нас есть" —
кросс-тикерный пул + S-map) отдельно для каждого тикера, чтобы увидеть
разброс оптимальных параметров и итогового rMAE между тикерами.

Точка: T_big=20%, D_allpeers, m∈{2..5}, θ∈[0,32], T_ratio∈[0.50,0.95] —
та же сетка калибратора, что и для SBER (эксп.17f).
"""
import time
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

spec17 = importlib.util.spec_from_file_location("exp17", HERE / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

spec17f = importlib.util.spec_from_file_location("exp17f", HERE / "17f_calibrator.py")
exp17f = importlib.util.module_from_spec(spec17f)
spec17f.loader.exec_module(exp17f)

T_BIG = 0.20
ARM = "D_allpeers"
TARGET_LIST = ["LKOH", "CHMF", "NVTK", "MGNT", "VTBR", "NLMK"]   # SBER уже откалиброван в эксп.17f


def calibrate_one(target):
    print(f"\n{'='*70}\n=== {target} ===")
    t0 = time.time()

    target_data = exp17.load_ticker(target)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != target]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != target}
    peer_data[target] = target_data  # для единообразия сигнатур не требуется, но не мешает

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, _ = exp17.build_zigzag(lh, ll, dates, T_BIG)
    print(f"Пивотов T_big: {len(full_lp)}  ({dates[0][:10]}…{dates[-1][:10]})")

    def_pools = exp17f.build_all_pools(target_data, peer_data, rankings, checkpoints, T_BIG,
                                        exp17f.DEF_M, exp17f.DEF_T_RATIO * T_BIG, ARM, full_lp, full_conf)
    def_v, def_n = exp17f.eval_smap_pools(def_pools, exp17f.DEF_THETA, exp17f.DEF_M + 2)
    print(f"Дефолт (m={exp17f.DEF_M} θ={exp17f.DEF_THETA} T_ratio={exp17f.DEF_T_RATIO}): "
          f"rMAE={def_v:.4f} (n={def_n})")

    m, theta, T, v, n, trace = exp17f.calibrate_smap(target_data, peer_data, rankings, checkpoints,
                                                      T_BIG, ARM, full_lp, full_conf)
    elapsed = time.time() - t0
    print(f"→ m={m} θ={theta:.3f} T_ratio={T:.4f} rMAE={v:.4f} (n={n})  ({elapsed:.1f}s)")

    return {"target": target, "n_big_pivots": len(full_lp),
            "default_rMAE": round(def_v, 4), "default_n": def_n,
            "m": m, "theta": round(theta, 3), "T_ratio": round(T, 4),
            "rMAE": round(v, 4), "n": n, "elapsed_s": round(elapsed, 1), "trace": trace}


def main():
    t0 = time.time()
    print("=== 17r_multiticker_calibrator === T_BIG=0.20 ARM=D_allpeers, только S-map")
    print(f"Тикеры: {TARGET_LIST}  (+ SBER уже откалиброван в эксп.17f: m=3 θ=25.697 T_ratio=0.8987 rMAE=0.6049)")

    results = [{
        "target": "SBER", "n_big_pivots": 73, "default_rMAE": 0.6738, "default_n": 42,
        "m": 3, "theta": 25.697, "T_ratio": 0.8987, "rMAE": 0.6049, "n": 42,
        "elapsed_s": None, "trace": None,
    }]
    for target in TARGET_LIST:
        r = calibrate_one(target)
        results.append(r)
        pd.DataFrame([{k: v for k, v in row.items() if k != "trace"} for row in results]).to_csv(
            RESULTS / "multiticker_calibration.csv", index=False, float_format="%.4f")

    df = pd.DataFrame([{k: v for k, v in row.items() if k != "trace"} for row in results])
    print(f"\n{'='*90}")
    print(df.to_string(index=False))
    print(f"{'='*90}")
    print(f"Среднее rMAE (откалиброван): {df['rMAE'].mean():.4f}   "
          f"среднее rMAE (дефолт): {df['default_rMAE'].mean():.4f}")

    import json
    with open(RESULTS / "multiticker_calibration_full.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nСохранено: {RESULTS}/multiticker_calibration.csv, multiticker_calibration_full.json")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
