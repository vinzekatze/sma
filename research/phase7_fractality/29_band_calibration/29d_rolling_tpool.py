#!/usr/bin/env python3
"""
29d_rolling_tpool.py — скользящая (walk-forward) предпрогнозная калибровка
T_pool/T_query по K последним ЗАВЕРШЁННЫМ событиям перед каждым origin'ом,
вместо одной глобальной калибровки на всю историю.

Мотивация (2026-07-08, по итогам 29c_tpool_curve_diagnostic.py): на
SBER/LKOH/MGNT «плато» оптимального T_pool на calib-периоде и на
test-периоде НЕ пересекаются — похоже на дрейф оптимума во времени, не на
шум. Глобальная калибровка (один T_pool на всю историю) принципиально не
может отследить дрейф. Пользователь предложил: раз θ=0 (локализации нет),
скользящая калибровка ДОЛЖНА быть каузально безопасна — на каждом origin'е
T_pool подбирается ТОЛЬКО по K событиям СТРОГО ДО него (их исходы уже
известны на момент origin'а), без обращения к будущему.

Причинность: на каждом шаге walk-forward calibration_window = origins с
индексами [i-K, i) в списке origin'ов ЦЕЛЕВОГО тикера (не включая сам
origin i) — используются только их РЕАЛЬНО НАСТУПИВШИЕ исходы (h=1,h=2),
которые к моменту origin'а i уже наступили (i-й origin календарно позже
всех K). T_pool находится golden-section по pinball на этом окне, затем
СРАЗУ применяется к origin'у i (без дальнейшей подгонки).

Использование:
    python 29d_rolling_tpool.py SBER --interval 1d --k 20
"""
import argparse
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REF_DIR = HERE.parents[1] / "reference"
sys.path.insert(0, str(REF_DIR))
from smap_band_ref import load_ticker_candles, build_zigzag, UNIVERSE

spec = importlib.util.spec_from_file_location("calib29", HERE / "29_smap_band_calibrator.py")
calib29 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib29)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

RATIO_LO, RATIO_HI, RATIO_TOL = 0.15, 3.00, 0.05   # чуть грубее допуск — экономия времени


def rolling_pinball_for_origin(origin, calib_window_origins, q_lp, q_dates, q_dirs,
                               ticker_arrays, t_query):
    """Golden-section по T_pool на calib_window_origins (K событий СТРОГО
    до origin), затем применяет найденный T_pool к САМОМУ origin'у.
    Возвращает (ratio_star, rows_for_origin)."""
    def obj(r):
        t_pool = t_query * r
        cached = calib29.build_pools_for_origins(calib_window_origins, q_lp, q_dates, q_dirs,
                                                  ticker_arrays, t_pool)
        return calib29.mean_pinball(calib29.eval_pools(cached))

    ratio_star = calib29.golden_section(obj, RATIO_LO, RATIO_HI, RATIO_TOL)
    t_pool_star = t_query * ratio_star
    cached_origin = calib29.build_pools_for_origins([origin], q_lp, q_dates, q_dirs,
                                                     ticker_arrays, t_pool_star)
    rows = calib29.eval_pools(cached_origin)
    return ratio_star, rows


def main():
    parser = argparse.ArgumentParser(description="Скользящая калибровка T_pool по K предыдущим событиям")
    parser.add_argument("ticker", help="Целевой тикер")
    parser.add_argument("--interval", default="1d", metavar="IV")
    parser.add_argument("--t-query", type=float, default=0.20, metavar="T")
    parser.add_argument("--k", type=int, default=20, metavar="K", help="Размер скользящего окна калибровки")
    parser.add_argument("--max-origins", type=int, default=None, help="Ограничить число origin'ов (для смоук-теста)")
    args = parser.parse_args()

    t0 = time.time()
    min_hist = calib29.M + 3

    print(f"Загрузка кросс-тикерного пула ({len(UNIVERSE)} тикеров)…")
    ticker_arrays = {}
    for tk in UNIVERSE:
        loaded = load_ticker_candles(tk, args.interval)
        if loaded is not None:
            ticker_arrays[tk] = loaded
    print(f"загружено {len(ticker_arrays)}/{len(UNIVERSE)} тикеров")

    log_highs, log_lows, dates = ticker_arrays[args.ticker]
    q_lp, q_dates, q_dirs = build_zigzag(log_highs, log_lows, dates, args.t_query)
    origins = calib29.prepare_origins(q_lp, min_hist)
    print(f"T_query={args.t_query*100:.1f}%: {len(q_lp)} пивотов, {len(origins)} origin'ов всего")

    walk_origins = origins[args.k:]     # первые K — burn-in для окна калибровки
    if args.max_origins:
        walk_origins = walk_origins[:args.max_origins]
    print(f"K={args.k}  walk-forward origin'ов для оценки: {len(walk_origins)}\n")

    rolling_rows, fixed_rows, ratios_used = [], [], []
    for idx, origin in enumerate(walk_origins):
        pos = origins.index(origin)
        calib_window = origins[pos - args.k: pos]

        ratio_star, rows_roll = rolling_pinball_for_origin(
            origin, calib_window, q_lp, q_dates, q_dirs, ticker_arrays, args.t_query)
        rolling_rows.extend(rows_roll)
        ratios_used.append(ratio_star)

        cached_fixed = calib29.build_pools_for_origins([origin], q_lp, q_dates, q_dirs,
                                                        ticker_arrays, args.t_query)
        fixed_rows.extend(calib29.eval_pools(cached_fixed))

        if (idx + 1) % 10 == 0 or idx == len(walk_origins) - 1:
            print(f"  [{idx+1:3d}/{len(walk_origins)}] origin={origin}  ratio={ratio_star:.3f}  "
                 f"[{time.time()-t0:.1f}s]")

    pb_roll = calib29.mean_pinball(rolling_rows)
    pb_fixed = calib29.mean_pinball(fixed_rows)
    print(f"\n=== {args.ticker} K={args.k}: скользящая калибровка vs фикс. T_pool=T_query ===")
    print(f"  скользящая:  pinball={pb_roll:.5f}")
    print(f"  фикс. (1.0): pinball={pb_fixed:.5f}")
    print(f"  относительно: {pb_roll/pb_fixed:.4f}  ({'лучше' if pb_roll < pb_fixed else 'хуже'})")
    print(f"  ratio использованных: mean={np.mean(ratios_used):.3f}  std={np.std(ratios_used):.3f}  "
          f"min={np.min(ratios_used):.3f}  max={np.max(ratios_used):.3f}")

    print(f"\nВсего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
