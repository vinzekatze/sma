#!/usr/bin/env python3
"""
17b_multi_ticker_stage2.py — Stage 2: проверка воспроизводимости эффекта
кросс-тикерного пула (Stage 1, 17_large_scale_pooled.py) на нескольких
целевых тикерах.

Сужена сетка до самой перспективной области Stage 1: T_big∈{15%,20%},
ratio∈{0.70,0.85}. Все 5 плеч сохранены — H3 (корреляция vs random)
проверяется повторно на каждом тикере, не только на SBER.

Переиспользует load_ticker / build_zigzag / build_pool_rows / walk_forward /
compute_peer_rankings из 17_large_scale_pooled.py без изменений — каузальный
контракт и методы прогноза те же (см. докстринг там). Меняется только
целевой тикер: для каждого target пул пиров = UNIVERSE \\ {target}.

Никакого дополнительного тюнинга параметров m/K/θ — если Stage 2 не
подтвердит эффект, тюнинг преждевременен.
"""
import importlib.util
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("exp17", HERE / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp17)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TARGET_LIST = ["SBER", "LKOH", "CHMF", "NVTK", "MGNT", "VTBR", "NLMK"]
T_BIG_GRID = [0.15, 0.20]
RATIO_GRID = [0.70, 0.85]
ARMS = exp17.ARMS  # все 5, как в Stage 1


def main():
    t0 = time.time()
    print("=== 17b_multi_ticker_stage2 ===")
    print(f"Targets: {TARGET_LIST}")
    print(f"T_BIG_GRID={T_BIG_GRID}  RATIO_GRID={RATIO_GRID}  ARMS={ARMS}")
    sys.stdout.flush()

    print("Загрузка данных universe (25 тикеров)...")
    all_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    out_path = RESULTS / "stage2_sweep.csv"
    rows = []
    combos = [(tgt, tb, r, arm)
              for tgt in TARGET_LIST for tb in T_BIG_GRID for r in RATIO_GRID for arm in ARMS]
    total = len(combos)

    cur_target = None
    rankings = None
    checkpoints = None

    for idx, (target, t_big, ratio, arm) in enumerate(combos, 1):
        if target != cur_target:
            cur_target = target
            exp17.PEERS = [t for t in exp17.UNIVERSE if t != target]
            target_data = all_data[target]
            years = sorted(set(int(d[:4]) for d in target_data["dates"]))
            checkpoints = np.array([f"{y}-01-01" for y in years])
            rankings = exp17.compute_peer_rankings(target_data, all_data, checkpoints)
            n_valid_last = len(rankings[checkpoints[-1]]["all"])
            print(f"\n--- target={target}  баров={len(target_data['dates'])}  "
                  f"пиров с историей={n_valid_last}/{len(exp17.PEERS)} ---")
            sys.stdout.flush()

        t_frac = ratio * t_big
        df = exp17.walk_forward(all_data[target], all_data, rankings, checkpoints, t_big, t_frac, arm)
        metrics = exp17.summarize(df)
        row = {"target": target, "T_big": t_big, "ratio": ratio, "T_frac": t_frac, "arm": arm, **metrics}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, index=False, float_format="%.5f")

        elapsed = time.time() - t0
        eta = elapsed / idx * (total - idx)
        la0  = metrics.get("rMAE_LA0", float("nan"))
        lwr  = metrics.get("rMAE_LWR", float("nan"))
        smap = metrics.get("rMAE_Smap", float("nan"))
        print(f"[{idx:3d}/{total}] {target:5s} T_big={t_big*100:.0f}% ratio={ratio:.2f} arm={arm:<11s} "
              f"n={metrics.get('n_steps', 0):3d} LA0={la0:.4f} LWR={lwr:.4f} Smap={smap:.4f}  ETA {eta:.0f}s")
        sys.stdout.flush()

    print(f"\nСохранено: {out_path}")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
