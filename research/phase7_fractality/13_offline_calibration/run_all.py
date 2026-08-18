#!/usr/bin/env python3
"""
run_all.py — Калибровка LWR + S-map и eval на последних N событиях для набора тикеров.

Для каждой комбинации (ticker, T_BIG):
  1. LWR: покоординатный спуск → optimal (m, K, T_ratio)
  2. S-map: покоординатный спуск → optimal (m, θ, T_ratio)
  3. eval: rMAE на последних [10, 25, 50] событиях через строго-каузальный ref-стиль

Использование:
  python run_all.py
  python run_all.py --tickers SBER GAZP LKOH --t-bigs 0.04 0.03
  python run_all.py --k-hi 400 --no-smap

Результаты:
  results/run_all_summary.csv   — сводная таблица
  results/lwr_{ticker}_{interval}_T{pct}.json
  results/smap_{ticker}_{interval}_T{pct}.json
  results/eval_last_n_{ticker}_{interval}_T{pct}.json
"""
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "reference"))

from _core import (load_log_candles, build_zigzag, get_pool,
                   rmae, golden, clear_cache)
from calibrate_lwr  import eval_lwr,  calibrate as _cal_lwr
from calibrate_smap import eval_smap, calibrate as _cal_smap
from eval_last_n    import run_eval

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

INTERVAL    = "10m"
DEF_K       = 75
DEF_M       = 2
DEF_T_RATIO = 0.85
DEF_THETA   = 2.0
WINDOWS     = [10, 25, 50]


def calibrate_ticker(ticker, t_big, k_hi, run_smap):
    lh, ll, dates = load_log_candles(ticker, INTERVAL)
    qlp, qconf, qdirs = build_zigzag(lh, ll, dates, t_big)
    n_piv = len(qlp)
    t_tag = f"T{round(t_big * 100):03d}"

    print(f"\n{'─'*60}")
    print(f"  {ticker}  {INTERVAL}  T={t_big*100:.0f}%  ({n_piv} пивотов)")
    print(f"{'─'*60}")

    row = {"ticker": ticker, "interval": INTERVAL, "t_big": t_big,
           "n_pivots": n_piv}

    # ── LWR ──────────────────────────────────────────────────────────────────
    print("[LWR] дефолт...")
    lwr_def, lwr_def_n = eval_lwr(qlp, qconf, qdirs, lh, ll, dates, t_big,
                                   DEF_M, DEF_K, DEF_T_RATIO)
    print(f"      rMAE={lwr_def:.4f}  (n={lwr_def_n})")

    print("[LWR] калибровка...")
    t0 = time.time()
    lwr_m, lwr_K, lwr_T, lwr_opt, lwr_opt_n, lwr_trace = _cal_lwr(
        qlp, qconf, qdirs, lh, ll, dates, t_big, k_hi)
    lwr_elapsed = time.time() - t0
    lwr_delta   = (lwr_opt - lwr_def) / lwr_def * 100
    print(f"      optimal m={lwr_m} K={lwr_K} T={lwr_T:.4f} → rMAE={lwr_opt:.4f} "
          f"Δ={lwr_delta:+.1f}%  ({lwr_elapsed:.1f}s)")

    lwr_result = {
        "method": "LWR", "ticker": ticker, "interval": INTERVAL,
        "t_big": t_big, "n_pivots": n_piv,
        "default": {"m": DEF_M, "K": DEF_K, "T_ratio": DEF_T_RATIO,
                    "rMAE": round(lwr_def, 4), "n_steps": lwr_def_n},
        "optimal": {"m": lwr_m, "K": lwr_K, "T_ratio": round(lwr_T, 4),
                    "rMAE": round(lwr_opt, 4), "n_steps": lwr_opt_n,
                    "delta_pct": round(lwr_delta, 2)},
        "trace": lwr_trace, "elapsed_s": round(lwr_elapsed, 1),
    }
    with open(RESULTS / f"lwr_{ticker}_{INTERVAL}_{t_tag}.json", "w") as f:
        json.dump(lwr_result, f, indent=2)

    row.update({"lwr_def": round(lwr_def, 4), "lwr_opt": round(lwr_opt, 4),
                "lwr_delta": round(lwr_delta, 2),
                "lwr_m": lwr_m, "lwr_K": lwr_K,
                "lwr_T_ratio": round(lwr_T, 4)})

    # ── S-map ─────────────────────────────────────────────────────────────────
    sm_m, sm_theta, sm_T = DEF_M, DEF_THETA, DEF_T_RATIO
    sm_opt = sm_def = float("nan")
    if run_smap:
        print("[S-map] дефолт...")
        sm_def, sm_def_n = eval_smap(qlp, qconf, qdirs, lh, ll, dates, t_big,
                                      DEF_M, DEF_THETA, DEF_T_RATIO)
        print(f"        rMAE={sm_def:.4f}  (n={sm_def_n})")

        print("[S-map] калибровка...")
        t0 = time.time()
        sm_m, sm_theta, sm_T, sm_opt, sm_opt_n, sm_trace = _cal_smap(
            qlp, qconf, qdirs, lh, ll, dates, t_big)
        sm_elapsed = time.time() - t0
        sm_delta   = (sm_opt - sm_def) / sm_def * 100
        print(f"        optimal m={sm_m} θ={sm_theta:.3f} T={sm_T:.4f} → rMAE={sm_opt:.4f} "
              f"Δ={sm_delta:+.1f}%  ({sm_elapsed:.1f}s)")

        sm_result = {
            "method": "S-map", "ticker": ticker, "interval": INTERVAL,
            "t_big": t_big, "n_pivots": n_piv,
            "default": {"m": DEF_M, "theta": DEF_THETA, "T_ratio": DEF_T_RATIO,
                        "rMAE": round(sm_def, 4), "n_steps": sm_def_n},
            "optimal": {"m": sm_m, "theta": round(sm_theta, 3), "T_ratio": round(sm_T, 4),
                        "rMAE": round(sm_opt, 4), "n_steps": sm_opt_n,
                        "delta_pct": round(sm_delta, 2)},
            "trace": sm_trace, "elapsed_s": round(sm_elapsed, 1),
        }
        with open(RESULTS / f"smap_{ticker}_{INTERVAL}_{t_tag}.json", "w") as f:
            json.dump(sm_result, f, indent=2)

        row.update({"smap_def": round(sm_def, 4), "smap_opt": round(sm_opt, 4),
                    "smap_delta": round(sm_delta, 2),
                    "smap_m": sm_m, "smap_theta": round(sm_theta, 3),
                    "smap_T_ratio": round(sm_T, 4)})

    # ── eval последних N ──────────────────────────────────────────────────────
    print("[eval] последние N событий...")
    clear_cache()   # сбросить кэш перед eval — иначе пул берёт глобальный срез
    eval_res = run_eval(ticker, INTERVAL, t_big,
                        lwr_m, lwr_K, lwr_T,
                        sm_m, sm_theta, sm_T)

    eval_out = {"ticker": ticker, "interval": INTERVAL, "t_big": t_big,
                "lwr_params":  {"m": lwr_m, "K": lwr_K, "T_ratio": round(lwr_T, 4)},
                "smap_params": {"m": sm_m, "theta": round(sm_theta, 3),
                                "T_ratio": round(sm_T, 4)},
                "windows": eval_res}
    with open(RESULTS / f"eval_last_n_{ticker}_{INTERVAL}_{t_tag}.json", "w") as f:
        json.dump(eval_out, f, indent=2)

    for win in WINDOWS:
        row[f"lwr_last{win}"]  = eval_res[win]["lwr"]
        row[f"smap_last{win}"] = eval_res[win]["smap"]

    # Сбросить кэш перед следующим тикером
    clear_cache()
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Калибровка + eval для набора тикеров",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers",  nargs="+",
                        default=["SBER", "GAZP", "LKOH", "CHMF", "NVTK", "PLZL"],
                        help="Список тикеров")
    parser.add_argument("--t-bigs",   nargs="+", type=float,
                        default=[0.04, 0.03, 0.02],
                        help="Пороги T_BIG зигзага")
    parser.add_argument("--k-hi",     type=int, default=300,
                        help="Верхняя граница K для LWR")
    parser.add_argument("--no-smap",  action="store_true",
                        help="Пропустить калибровку S-map")
    args = parser.parse_args()

    print(f"Тикеры: {args.tickers}")
    print(f"T_BIG:  {[f'{t*100:.0f}%' for t in args.t_bigs]}")
    print(f"K_HI:   {args.k_hi}")

    t_start = time.time()
    rows = []

    for ticker in args.tickers:
        for t_big in args.t_bigs:
            try:
                row = calibrate_ticker(ticker, t_big, args.k_hi,
                                       run_smap=not args.no_smap)
                rows.append(row)
            except Exception as e:
                print(f"  ОШИБКА {ticker} T={t_big*100:.0f}%: {e}")
                clear_cache()

    # ── Сводная таблица ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_csv = RESULTS / "run_all_summary.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n{'═'*70}")
    print("СВОДКА — rMAE in-sample (optimal) и out-of-sample (последние N)")
    print(f"{'═'*70}")

    cols_show = (["ticker", "t_big", "n_pivots",
                  "lwr_opt", "smap_opt",
                  "lwr_last10", "smap_last10",
                  "lwr_last50", "smap_last50"])
    cols_show = [c for c in cols_show if c in df.columns]
    print(df[cols_show].to_string(index=False))
    print(f"\nСохранено: {out_csv}")
    print(f"Всего: {(time.time()-t_start)/60:.1f} мин")


if __name__ == "__main__":
    main()
