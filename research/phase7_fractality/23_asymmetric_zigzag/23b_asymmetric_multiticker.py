#!/usr/bin/env python3
"""
23b_asymmetric_multiticker.py — проверка percentage-symmetric зигзага
(эксп.23) на остальных 6 тикерах из уже откалиброванного набора (эксп.17r).
Тот же протокол: подбор P_pct под сопоставимое (с эталонным symmetric)
число пивотов, независимая калибровка (m,θ,T_ratio), сравнение rMAE
каждого условия относительно СВОЕГО persistence.
"""
import importlib.util
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
spec23 = importlib.util.spec_from_file_location("exp23", HERE / "23_asymmetric_zigzag_calibration.py")
exp23 = importlib.util.module_from_spec(spec23)
spec23.loader.exec_module(exp23)   # выполнит main() модуля? нет — guard __name__ защищает

exp17 = exp23.exp17
RESULTS = HERE / "results"

# эталонные (symmetric) калибровки — эксп.17f/17r
REFERENCE = {
    "SBER": {"T_big": 0.20, "m": 3, "theta": 25.697, "T_ratio": 0.8987, "rMAE": 0.6049},
    "LKOH": {"T_big": 0.20, "m": 4, "theta": 24.817, "T_ratio": 0.8331, "rMAE": 0.6961},
    "CHMF": {"T_big": 0.20, "m": 3, "theta": 11.492, "T_ratio": 0.9238, "rMAE": 0.6658},
    "NVTK": {"T_big": 0.20, "m": 2, "theta": 15.158, "T_ratio": 0.8080, "rMAE": 0.8136},
    "MGNT": {"T_big": 0.20, "m": 2, "theta": 11.814, "T_ratio": 0.6110, "rMAE": 0.6479},
    "VTBR": {"T_big": 0.20, "m": 3, "theta": 0.731,  "T_ratio": 0.6169, "rMAE": 0.6320},
    "NLMK": {"T_big": 0.20, "m": 2, "theta": 2.315,  "T_ratio": 0.8641, "rMAE": 0.7324},
}
TICKERS_TO_RUN = ["LKOH", "CHMF", "NVTK", "MGNT", "VTBR", "NLMK"]   # SBER уже сделан в 23


def run_for_ticker(target, ref):
    target_data = exp17.load_ticker(target)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != target]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != target}
    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    ref_lp, _, _ = exp17.build_zigzag(lh, ll, dates, ref["T_big"])
    n_ref = len(ref_lp)

    counts = {p: exp23.pivot_count_asym(lh, ll, dates, p) for p in exp23.PPCT_SEARCH_GRID}
    best_p = min(counts, key=lambda p: abs(counts[p] - n_ref))
    print(f"  symmetric: {n_ref} пивотов (T_big={ref['T_big']})  "
          f"asymmetric: P_pct={best_p} → {counts[best_p]} пивотов")

    # временно подменяем DEF_M/DEF_THETA/DEF_T_RATIO стартом от эталона этого тикера
    exp23.DEF_M, exp23.DEF_THETA, exp23.DEF_T_RATIO = ref["m"], ref["theta"], ref["T_ratio"]

    t1 = time.time()
    m, theta, T, v, n, trace = exp23.calibrate(target_data, peer_data, rankings, checkpoints, best_p)
    elapsed = time.time() - t1
    print(f"  → m={m} θ={theta:.3f} T_ratio={T:.4f} rMAE={v:.4f} (n={n})  ({elapsed:.1f}s)")

    return {"ticker": target, "rMAE_symmetric": ref["rMAE"], "n_pivots_symmetric": n_ref,
            "P_pct_asym": best_p, "n_pivots_asym": counts[best_p],
            "m_asym": m, "theta_asym": round(theta, 3), "T_ratio_asym": round(T, 4),
            "rMAE_asymmetric": round(v, 4), "n_eval": n, "delta": round(v - ref["rMAE"], 4),
            "elapsed_s": round(elapsed, 1)}


def main():
    t0 = time.time()
    print("=== 23b_asymmetric_multiticker — percentage-symmetric зигзаг на 6 тикерах ===\n")

    results = [{"ticker": "SBER", "rMAE_symmetric": 0.6049, "n_pivots_symmetric": 73,
                "P_pct_asym": 0.21, "n_pivots_asym": 75, "m_asym": 4, "theta_asym": 26.340,
                "T_ratio_asym": 0.8581, "rMAE_asymmetric": 0.5900, "n_eval": 44,
                "delta": round(0.5900 - 0.6049, 4), "elapsed_s": 39.7}]  # уже посчитан в 23

    for ticker in TICKERS_TO_RUN:
        print(f"--- {ticker} ---")
        r = run_for_ticker(ticker, REFERENCE[ticker])
        results.append(r)
        pd.DataFrame(results).to_csv(RESULTS / "asymmetric_multiticker.csv", index=False, float_format="%.4f")
        print()

    df = pd.DataFrame(results)
    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}")
    print(f"Среднее: symmetric={df['rMAE_symmetric'].mean():.4f}  asymmetric={df['rMAE_asymmetric'].mean():.4f}")
    print(f"Улучшение на: {(df['delta']<0).sum()}/{len(df)} тикерах")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
