#!/usr/bin/env python3
"""
29b_lopo_diagnostic.py — leave-one-peer-out диагностика: какой вклад в
качество ПОЛОСЫ (pinball loss) вносит КАЖДЫЙ отдельный пир кросс-тикерного
пула. Чистое ИЗМЕРЕНИЕ, не подгонка — переобучиться тут нечем (никакой
параметр не выбирается по данным, просто 44 сравнения "с пиром / без пира").

Мотивация (2026-07-08): после того как θ и m зафиксированы константами
(θ=0, m=6 — калибровка не пережила честный OOS-тест), пользователь
предложил следующий рычаг — вес источника пула (по тикеру): часть пиров
может систематически помогать прогнозу целевого тикера, часть — мешать
(шуметь). Пользователь явно ограничил объём работы: сначала ТОЛЬКО
диагностика (этот скрипт), решение о переходе к весам — отдельным шагом.

T_pool = T_query (ratio=1.0) — робастный дефолт по итогам предыдущего шага
(29_smap_band_calibrator.py показал, что индивидуальная калибровка T_pool
на тикер не переживает OOS на 2/7 тикеров) — НЕ индивидуально
откалиброванный T_pool, чтобы не путать эффект LOPO с эффектом
переобученного T_pool.

Используются только CALIB origin'ы (первые 65% календарно) — TEST
сохраняется неприкосновенным на случай, если по итогам диагностики решим
переходить к весам и понадобится честная OOS-проверка.

Причинность: build_causal_pool (см. smap_band_ref.py) — без изменений.

Использование:
    python 29b_lopo_diagnostic.py SBER --interval 1d
"""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="LOPO-диагностика вклада пиров в качество полосы")
    parser.add_argument("ticker", help="Целевой тикер (напр. SBER)")
    parser.add_argument("--interval", default="1d", metavar="IV")
    parser.add_argument("--t-query", type=float, default=0.20, metavar="T")
    parser.add_argument("--t-pool-ratio", type=float, default=1.0, metavar="R",
                        help="T_pool = T_query × R (по умолчанию 1.0 — робастный дефолт)")
    parser.add_argument("--split-frac", type=float, default=calib29.SPLIT_FRAC)
    args = parser.parse_args()

    t0 = time.time()
    min_hist = calib29.M + 3

    print(f"Загрузка кросс-тикерного пула ({len(UNIVERSE)} тикеров, интервал={args.interval})…")
    ticker_arrays = {}
    for tk in UNIVERSE:
        loaded = load_ticker_candles(tk, args.interval)
        if loaded is not None:
            ticker_arrays[tk] = loaded
    print(f"загружено {len(ticker_arrays)}/{len(UNIVERSE)} тикеров")

    if args.ticker not in ticker_arrays:
        raise SystemExit(f"Нет данных для целевого тикера {args.ticker}")

    log_highs, log_lows, dates = ticker_arrays[args.ticker]
    print(f"Целевой тикер {args.ticker}: {len(dates)} свечей  ({str(dates[0])[:10]} … {str(dates[-1])[:10]})")

    q_lp, q_dates, q_dirs = build_zigzag(log_highs, log_lows, dates, args.t_query)
    origins = calib29.prepare_origins(q_lp, min_hist)
    split_i = int(len(origins) * args.split_frac)
    calib_origins = origins[:split_i]
    print(f"T_query={args.t_query*100:.1f}%: {len(q_lp)} пивотов, "
          f"{len(calib_origins)} calib origin'ов (TEST не используется — сохранён)")

    t_pool = args.t_query * args.t_pool_ratio
    print(f"T_pool={t_pool*100:.2f}% (×{args.t_pool_ratio})  θ={calib29.THETA}  m={calib29.M}\n")

    print("Baseline (все пиры)…")
    cached_full = calib29.build_pools_for_origins(calib_origins, q_lp, q_dates, q_dirs, ticker_arrays, t_pool)
    pb_full = calib29.mean_pinball(calib29.eval_pools(cached_full))
    print(f"  pinball_full = {pb_full:.5f}  [{time.time()-t0:.1f}s]\n")

    peers = [p for p in UNIVERSE if p in ticker_arrays and p != args.ticker]
    print(f"LOPO по {len(peers)} пирам…")
    rows = []
    for i, peer in enumerate(peers, 1):
        reduced = {k: v for k, v in ticker_arrays.items() if k != peer}
        cached_wo = calib29.build_pools_for_origins(calib_origins, q_lp, q_dates, q_dirs, reduced, t_pool)
        pb_wo = calib29.mean_pinball(calib29.eval_pools(cached_wo))
        contribution = pb_wo - pb_full   # >0: без пира ХУЖЕ → пир ПОМОГАЛ. <0: без пира ЛУЧШЕ → пир МЕШАЛ.
        rows.append({"peer": peer, "pinball_without": pb_wo, "contribution": contribution})
        print(f"  [{i:2d}/{len(peers)}] без {peer:6s}: pinball={pb_wo:.5f}  "
              f"вклад={contribution:+.6f}  [{time.time()-t0:.1f}s]")

    df = pd.DataFrame(rows).sort_values("contribution", ascending=False)
    out_path = RESULTS / f"29b_lopo_{args.ticker}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n=== Итог LOPO для {args.ticker} (pinball_full={pb_full:.5f}) ===")
    print(f"Вклад: mean={df.contribution.mean():+.6f}  std={df.contribution.std():.6f}  "
          f"min={df.contribution.min():+.6f}  max={df.contribution.max():+.6f}")
    n_help = int((df.contribution > 0).sum())
    n_hurt = int((df.contribution < 0).sum())
    print(f"Помогают (вклад>0): {n_help}/{len(df)}   Мешают (вклад<0): {n_hurt}/{len(df)}")

    print(f"\nТоп-5 ПОМОГАЮТ больше всего (без них хуже):")
    print(df.head(5).to_string(index=False))
    print(f"\nТоп-5 МЕШАЮТ больше всего (без них лучше):")
    print(df.tail(5).to_string(index=False))

    print(f"\nСохранено: {out_path}")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
