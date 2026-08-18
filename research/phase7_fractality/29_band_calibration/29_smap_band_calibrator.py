#!/usr/bin/env python3
"""
29_smap_band_calibrator.py — калибратор T_pool/T_query для S-map ПОЛОСЫ
неопределённости (smap_band_ref.py) на двух шагах (уход + возврат),
кросс-тикерный пул, честная OOS-проверка через календарный сплит origin'ов.

Задача (сформулирована пользователем 2026-07-08): выбрать конфигурацию,
при которой И покрытие выше (факт чаще попадает в полосу), И полоса
достаточно узкая, чтобы на ней можно было принимать решение. Pinball loss
— строго честная (proper) метрика, которая уже балансирует оба требования
одновременно (искусственное расширение полосы штрафуется неограниченно,
см. память feedback-uncertainty-field-width-control).

θ И m ЗАФИКСИРОВАНЫ константами по итогам полного 3-параметрического
координатного спуска на 7 тикерах (SBER LKOH GAZP MGNT CHMF MTSS PLZL,
T_query=20%, results/run_log_29_all7.txt):
  - θ=0: калиброванная θ на честном TEST НЕ бьёт θ=0 (тот же пул) ни на
    одном из 7 тикеров — на 2/7 неотличимо (θ*≈0 сама по себе), на 5/7
    калиброванная θ ХУЖЕ на +0.5%…+6% (переобучение θ на calib-срезе).
    Локализация систематически не помогает — см. [[feedback-band-
    calibrator-no-fractal-constraint]].
  - m=6: при θ=0 target-значения пула НЕ зависят от m вообще (build_
    pool_vectors: target = разница log-цен, m влияет только на то, с
    какого пивота начинается пул) — эмпирически подтверждено: разброс
    pinball по m∈{2,3,5,6,8,13,20} на всех 7 тикерах в 4-м знаке
    (<0.5% относительно). m=6 — нигде не худший вариант, компромисс между
    「не слишком урезает историю」 и 「не самый маленький, на случай что
    m всё-таки пригодится, если вернёмся к точке」.

Единственный калибруемый параметр — T_pool/T_query (golden-section,
БЕЗ верхнего потолка: полоса не обязана следовать фрактальному T_frac<T_big,
это правило специфично для прогноза ТОЧКИ крупного события из мелких).

Каузальность: build_causal_pool обрезает каждого пира по дате в самом
начале (проверено causality_check_smap_band.py, Тест 4). calib/test —
календарный сплит origin'ов целевого тикера (test строго позже calib).

Использование:
    python 29_smap_band_calibrator.py SBER --interval 1d
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REF_DIR = Path(__file__).parents[2] / "reference"
sys.path.insert(0, str(REF_DIR))
from smap_band_ref import (
    load_ticker_candles, build_zigzag, build_query_vector,
    smap_weights, weighted_quantile, build_causal_pool, UNIVERSE,
)

THETA = 0.0     # фиксирован — см. docstring
M = 6           # фиксирован — см. docstring
RATIO_LO, RATIO_HI, RATIO_TOL = 0.15, 3.00, 0.02    # БЕЗ потолка <1 (не фрактальная задача)
Q_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
SPLIT_FRAC = 0.65
MIN_POOL = M + 2


def pinball(actual: float, band: dict, q_levels: tuple) -> float:
    total = 0.0
    for q in q_levels:
        diff = actual - band[q]
        total += max(q * diff, (q - 1) * diff)
    return total / len(q_levels)


def coverage(actual: float, band: dict) -> tuple:
    return (int(band[0.1] <= actual <= band[0.9]),
           int(band[0.25] <= actual <= band[0.75]))


def prepare_origins(q_lp, min_hist):
    return list(range(min_hist, len(q_lp) - 2))


def build_pools_for_origins(origins, q_lp, q_dates, q_dirs, ticker_arrays, t_pool):
    cached = []
    for origin in origins:
        q_lp_prefix = q_lp[: origin + 1]
        qv0 = build_query_vector(q_lp_prefix, M)
        q_dir = int(q_dirs[origin])
        origin_date = str(q_dates[origin])

        actual_lr_1 = float(q_lp[origin + 1] - q_lp[origin])
        actual_lr_2 = float(q_lp[origin + 2] - q_lp[origin + 1])

        pfm, ptr, pdir = build_causal_pool(origin_date, ticker_arrays, t_pool, M)
        cached.append((qv0, q_dir, actual_lr_1, actual_lr_2, pfm, ptr, pdir))
    return cached


def eval_pools(cached, q_levels=Q_LEVELS):
    """θ=0 фиксирован — веса всегда равномерные, полоса = безусловные
    квантили направленно-отфильтрованного пула."""
    rows = []
    for qv0, q_dir, actual_lr_1, actual_lr_2, pfm, ptr, pdir in cached:
        mask1 = pdir == q_dir
        if mask1.sum() >= MIN_POOL:
            band1 = weighted_quantile(ptr[mask1], np.ones(mask1.sum()), q_levels)
            rows.append({"h": 1, "pinball": pinball(actual_lr_1, band1, q_levels),
                        "cov1090": coverage(actual_lr_1, band1)[0],
                        "cov2575": coverage(actual_lr_1, band1)[1]})

        cur_dir2 = -q_dir
        mask2 = pdir == cur_dir2
        if mask2.sum() >= MIN_POOL:
            band2 = weighted_quantile(ptr[mask2], np.ones(mask2.sum()), q_levels)
            rows.append({"h": 2, "pinball": pinball(actual_lr_2, band2, q_levels),
                        "cov1090": coverage(actual_lr_2, band2)[0],
                        "cov2575": coverage(actual_lr_2, band2)[1]})
    return rows


def mean_pinball(rows):
    return float(np.mean([r["pinball"] for r in rows])) if rows else float("inf")


def golden_section(f, lo, hi, tol):
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    while abs(b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = f(d)
    return (a + b) / 2


def summarize(rows, label):
    for h in [1, 2]:
        sub = [r for r in rows if r["h"] == h]
        if not sub:
            print(f"  {label} h={h}: нет данных"); continue
        pb = np.mean([r["pinball"] for r in sub])
        c1090 = np.mean([r["cov1090"] for r in sub])
        c2575 = np.mean([r["cov2575"] for r in sub])
        print(f"  {label} h={h}: pinball={pb:.5f}  cov1090={c1090:.2f}  cov2575={c2575:.2f}  n={len(sub)}")
    pb_all = np.mean([r["pinball"] for r in rows]) if rows else float("nan")
    print(f"  {label} combined: pinball={pb_all:.5f}  n={len(rows)}")
    return pb_all


def main():
    parser = argparse.ArgumentParser(
        description="Калибратор T_pool/T_query для S-map полосы (θ=0, m=6 фиксированы)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("ticker", help="Целевой тикер (напр. SBER)")
    parser.add_argument("--interval", default="1d", metavar="IV")
    parser.add_argument("--t-query", type=float, default=0.20, metavar="T")
    parser.add_argument("--split-frac", type=float, default=SPLIT_FRAC)
    args = parser.parse_args()

    t0 = time.time()
    min_hist = M + 3

    print(f"Загрузка кросс-тикерного пула ({len(UNIVERSE)} тикеров, интервал={args.interval})…")
    ticker_arrays = {}
    for tk in UNIVERSE:
        loaded = load_ticker_candles(tk, args.interval)
        if loaded is not None:
            ticker_arrays[tk] = loaded
    print(f"загружено {len(ticker_arrays)}/{len(UNIVERSE)} тикеров")

    if args.ticker not in ticker_arrays:
        loaded = load_ticker_candles(args.ticker, args.interval)
        if loaded is None:
            raise SystemExit(f"Нет данных для целевого тикера {args.ticker} ({args.interval})")
        ticker_arrays[args.ticker] = loaded

    log_highs, log_lows, dates = ticker_arrays[args.ticker]
    print(f"Целевой тикер {args.ticker}: {len(dates)} свечей  ({str(dates[0])[:10]} … {str(dates[-1])[:10]})")

    q_lp, q_dates, q_dirs = build_zigzag(log_highs, log_lows, dates, args.t_query)
    print(f"T_query={args.t_query*100:.1f}%: {len(q_lp)} пивотов  (θ={THETA}, m={M} фиксированы)")

    origins = prepare_origins(q_lp, min_hist)
    split_i = int(len(origins) * args.split_frac)
    calib_origins, test_origins = origins[:split_i], origins[split_i:]
    print(f"Origin'ов: {len(origins)}  (calib={len(calib_origins)}, test={len(test_origins)}, "
          f"календарный сплит {args.split_frac:.0%})")
    assert max(calib_origins) < min(test_origins), "утечка: calib и test пересекаются по времени"

    print(f"\n=== калибровка (golden-section по T_pool/T_query, диапазон [{RATIO_LO},{RATIO_HI}]) ===")

    def obj(r):
        t_pool = args.t_query * r
        cached = build_pools_for_origins(calib_origins, q_lp, q_dates, q_dirs, ticker_arrays, t_pool)
        return mean_pinball(eval_pools(cached))

    ratio_star = golden_section(obj, RATIO_LO, RATIO_HI, RATIO_TOL)
    t_pool_star = args.t_query * ratio_star
    print(f"Выбрано: T_pool={t_pool_star*100:.2f}% (×{ratio_star:.4f})  calib_pinball={obj(ratio_star):.5f}")

    print("\n=== честная оценка на TEST (out-of-sample) ===")
    cached_test = build_pools_for_origins(test_origins, q_lp, q_dates, q_dirs, ticker_arrays, t_pool_star)
    test_rows = eval_pools(cached_test)
    summarize(test_rows, f"калиброванный (T_pool={t_pool_star*100:.1f}%)")

    t_pool_default = args.t_query * 1.0   # T_pool=T_query — нейтральный дефолт без калибровки
    cached_default = build_pools_for_origins(test_origins, q_lp, q_dates, q_dirs, ticker_arrays, t_pool_default)
    default_rows = eval_pools(cached_default)
    print()
    summarize(default_rows, f"дефолт без калибровки (T_pool=T_query={t_pool_default*100:.1f}%)")

    print(f"\nВсего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
