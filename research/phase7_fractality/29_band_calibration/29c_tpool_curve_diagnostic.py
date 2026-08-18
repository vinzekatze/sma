#!/usr/bin/env python3
"""
29c_tpool_curve_diagnostic.py — форма кривой pinball(T_pool/T_query),
отдельно на calib и на test, по каждому целевому тикеру. Чистая
диагностика (мелкая сетка + честные подвыборки, ничего не выбирается по
данным теста) — объясняет, почему golden-section калибровка T_pool
(29_smap_band_calibrator.py) не пережила OOS на LKOH/MGNT (+3.6%/+4.4%),
но помогла на SBER (-2.6%).

Вопросы, на которые должна ответить эта диагностика:
  1. Острый минимум (переобучение вероятно) или плоский (calib-выбор
     должен быть надёжен)?
  2. Совпадает ли минимум calib-кривой с минимумом test-кривой, или они
     расходятся (прямой признак переобучения самого выбора T_pool)?

θ=0, m=6 — фиксированы (см. 29_smap_band_calibrator.py).

Использование:
    python 29c_tpool_curve_diagnostic.py
"""
import importlib.util
import sys
import time
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

TARGETS = ["SBER", "LKOH", "GAZP", "MGNT", "CHMF", "MTSS", "PLZL"]
T_QUERY = 0.20
RATIO_GRID = np.round(np.arange(0.50, 1.55, 0.05), 2)


def curve_for_split(origins, q_lp, q_dates, q_dirs, ticker_arrays, t_query, ratio_grid):
    vals = []
    for ratio in ratio_grid:
        t_pool = t_query * ratio
        cached = calib29.build_pools_for_origins(origins, q_lp, q_dates, q_dirs, ticker_arrays, t_pool)
        vals.append(calib29.mean_pinball(calib29.eval_pools(cached)))
    return np.array(vals)


def main():
    t0 = time.time()
    print(f"Загрузка кросс-тикерного пула ({len(UNIVERSE)} тикеров)…")
    ticker_arrays = {}
    for tk in UNIVERSE:
        loaded = load_ticker_candles(tk, "1d")
        if loaded is not None:
            ticker_arrays[tk] = loaded
    print(f"загружено {len(ticker_arrays)}/{len(UNIVERSE)} тикеров\n")

    all_rows = []
    for target in TARGETS:
        log_highs, log_lows, dates = ticker_arrays[target]
        q_lp, q_dates, q_dirs = build_zigzag(log_highs, log_lows, dates, T_QUERY)
        min_hist = calib29.M + 3
        origins = calib29.prepare_origins(q_lp, min_hist)
        split_i = int(len(origins) * calib29.SPLIT_FRAC)
        calib_origins, test_origins = origins[:split_i], origins[split_i:]

        print(f"--- {target} (calib={len(calib_origins)}, test={len(test_origins)}) ---")
        calib_curve = curve_for_split(calib_origins, q_lp, q_dates, q_dirs, ticker_arrays, T_QUERY, RATIO_GRID)
        test_curve  = curve_for_split(test_origins,  q_lp, q_dates, q_dirs, ticker_arrays, T_QUERY, RATIO_GRID)

        calib_best_i, test_best_i = int(np.argmin(calib_curve)), int(np.argmin(test_curve))
        print(f"  calib минимум: ratio={RATIO_GRID[calib_best_i]:.2f}  pinball={calib_curve[calib_best_i]:.5f}  "
              f"(диапазон кривой: {calib_curve.min():.5f}-{calib_curve.max():.5f}, "
              f"размах={100*(calib_curve.max()/calib_curve.min()-1):.1f}%)")
        print(f"  test  минимум: ratio={RATIO_GRID[test_best_i]:.2f}  pinball={test_curve[test_best_i]:.5f}  "
              f"(диапазон кривой: {test_curve.min():.5f}-{test_curve.max():.5f}, "
              f"размах={100*(test_curve.max()/test_curve.min()-1):.1f}%)")
        # штраф на test за использование calib-оптимума
        test_at_calib_opt = test_curve[calib_best_i]
        penalty = 100 * (test_at_calib_opt / test_curve[test_best_i] - 1)
        print(f"  test pinball В ТОЧКЕ calib-оптимума: {test_at_calib_opt:.5f}  "
              f"(+{penalty:.1f}% хуже, чем настоящий test-оптимум)\n")

        for ratio, cv, tv in zip(RATIO_GRID, calib_curve, test_curve):
            all_rows.append({"target": target, "ratio": ratio, "calib_pinball": cv, "test_pinball": tv})

    df = pd.DataFrame(all_rows)
    out_path = RESULTS / "29c_tpool_curves.csv"
    df.to_csv(out_path, index=False)
    print(f"Сохранено: {out_path}")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
