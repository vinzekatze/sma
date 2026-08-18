#!/usr/bin/env python3
"""
29e_t_sweep_diagnostic.py — честная (pinball + покрытие) проверка T=T_query=
T_pool по сетке значений, на 7 целевых тикерах. Отвечает на вопрос
пользователя (2026-07-08): T=25% в app9 «выглядит полезным» по картинке —
это чистое измерение (НЕ выбор/argmin, переобучиться нечем), подтверждает
или корректирует субъективное впечатление количественно.

Конфигурация — как в app9.py: θ=0, m=6 фиксированы, T_pool=T_query=T
(единый порог), кросс-тикерный пул (UNIVERSE), 2 шага (horizon=1,2),
направление шага читается напрямую из истории (без промежуточного
прогноза точки).

Метрика: mean pinball loss (5 квантилей 0.1/0.25/0.5/0.75/0.9 — тот же
стандартный набор, что и в остальной 29-серии, для сравнимости чисел) +
покрытие на уровнях 50/75/90% (соответствуют чекбоксам app9 — напрямую
интерпретируемо: «если бы вы включили зону 50%, как часто факт туда
попадал бы на самом деле»).

Использование:
    python 29e_t_sweep_diagnostic.py
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
from smap_band_ref import load_ticker_candles, build_zigzag, build_causal_pool, UNIVERSE

spec = importlib.util.spec_from_file_location("calib29", HERE / "29_smap_band_calibrator.py")
calib29 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib29)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TARGETS = ["SBER", "LKOH", "GAZP", "MGNT", "CHMF", "MTSS", "PLZL"]
T_GRID = [0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
COVERAGE_LEVELS = [0.50, 0.75, 0.90]   # соответствуют чекбоксам app9


def coverage_at(actual: float, band: dict, level: float) -> int:
    frac = level
    q_lo, q_hi = (1 - frac) / 2, (1 + frac) / 2
    return int(band[q_lo] <= actual <= band[q_hi])


def evaluate_ticker_at_t(target: str, t: float, ticker_arrays: dict) -> dict:
    log_highs, log_lows, dates = ticker_arrays[target]
    q_lp, q_dates, q_dirs = build_zigzag(log_highs, log_lows, dates, t)
    min_hist = calib29.M + 3
    origins = calib29.prepare_origins(q_lp, min_hist)   # все доступные, БЕЗ сплита — чистое измерение

    q_levels_needed = sorted(set(calib29.Q_LEVELS) |
                             {(1 - lv) / 2 for lv in COVERAGE_LEVELS} |
                             {(1 + lv) / 2 for lv in COVERAGE_LEVELS})

    rows_h1, rows_h2 = [], []
    for origin in origins:
        q_lp_prefix = q_lp[: origin + 1]
        from smap_band_ref import build_query_vector, build_pool_vectors, weighted_quantile
        qv0 = build_query_vector(q_lp_prefix, calib29.M)
        q_dir = int(q_dirs[origin])
        origin_date = str(q_dates[origin])

        actual_lr_1 = float(q_lp[origin + 1] - q_lp[origin])
        actual_lr_2 = float(q_lp[origin + 2] - q_lp[origin])   # КУМУЛЯТИВНО от origin —
        # совпадает с определением horizon=2 в build_pool_vectors/app9.py (прямой
        # historical readout, НЕ инкремент origin+1→origin+2 как в старом
        # fact-conditioned калибраторе 29_smap_band_calibrator.py — там другая схема шага 2)

        pfm1, ptr1, pdir1 = build_causal_pool(origin_date, ticker_arrays, t, calib29.M, horizon=1)
        pfm2, ptr2, pdir2 = build_causal_pool(origin_date, ticker_arrays, t, calib29.M, horizon=2)
        mask1 = pdir1 == q_dir
        mask2 = pdir2 == q_dir
        if mask1.sum() < calib29.M + 2 or mask2.sum() < calib29.M + 2:
            continue

        ones1 = np.ones(mask1.sum())
        band1 = weighted_quantile(ptr1[mask1], ones1, q_levels_needed)
        ones2 = np.ones(mask2.sum())
        band2 = weighted_quantile(ptr2[mask2], ones2, q_levels_needed)

        pb1 = calib29.pinball(actual_lr_1, band1, calib29.Q_LEVELS)
        pb2 = calib29.pinball(actual_lr_2, band2, calib29.Q_LEVELS)
        cov1 = {lv: coverage_at(actual_lr_1, band1, lv) for lv in COVERAGE_LEVELS}
        cov2 = {lv: coverage_at(actual_lr_2, band2, lv) for lv in COVERAGE_LEVELS}

        rows_h1.append({"pinball": pb1, **{f"cov{int(lv*100)}": cov1[lv] for lv in COVERAGE_LEVELS}})
        rows_h2.append({"pinball": pb2, **{f"cov{int(lv*100)}": cov2[lv] for lv in COVERAGE_LEVELS}})

    if not rows_h1:
        return None
    df1, df2 = pd.DataFrame(rows_h1), pd.DataFrame(rows_h2)
    result = {"target": target, "t_pct": round(t * 100, 1), "n": len(df1)}
    for h, df in [(1, df1), (2, df2)]:
        result[f"pinball_h{h}"] = df.pinball.mean()
        # нормировка на T — при большем T плечи зигзага физически крупнее,
        # сырой pinball механически растёт вместе с T; сравнивать T между
        # собой честно можно только через покрытие (масштабно-независимо по
        # построению) или через pinball, делённый на масштаб (T)
        result[f"pinball_norm_h{h}"] = df.pinball.mean() / t
        for lv in COVERAGE_LEVELS:
            cov = df[f"cov{int(lv*100)}"].mean()
            result[f"cov{int(lv*100)}_h{h}"] = cov
            result[f"cov{int(lv*100)}_h{h}_err"] = abs(cov - lv)   # отклонение от номинала
    return result


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
        for t in T_GRID:
            r = evaluate_ticker_at_t(target, t, ticker_arrays)
            if r:
                all_rows.append(r)
                print(f"  {target} T={t*100:.0f}%: n={r['n']}  "
                      f"pinball(h1,h2)=({r['pinball_h1']:.4f},{r['pinball_h2']:.4f})  "
                      f"cov50=({r['cov50_h1']:.2f},{r['cov50_h2']:.2f})  "
                      f"cov75=({r['cov75_h1']:.2f},{r['cov75_h2']:.2f})  "
                      f"cov90=({r['cov90_h1']:.2f},{r['cov90_h2']:.2f})  "
                      f"[{time.time()-t0:.1f}s]")

    df = pd.DataFrame(all_rows)
    out_path = RESULTS / "29e_t_sweep.csv"
    df.to_csv(out_path, index=False)

    # сырой pinball между T НЕ сравниваем (масштаб растёт вместе с T) —
    # ориентир: среднее |отклонение покрытия от номинала| по 3 уровням × 2 шагам
    df["cov_err_avg"] = df[[f"cov{lv}_h{h}_err" for lv in (50, 75, 90) for h in (1, 2)]].mean(axis=1)
    df["pinball_norm_avg"] = (df.pinball_norm_h1 + df.pinball_norm_h2) / 2

    print(f"\n=== per-ticker: T с лучшей калибровкой покрытия (мин. |cov−номинал|) ===")
    for target in TARGETS:
        sub = df[df.target == target]
        if not len(sub):
            continue
        best = sub.loc[sub.cov_err_avg.idxmin()]
        print(f"  {target}: лучшее T={best['t_pct']:.0f}%  cov_err_avg={best['cov_err_avg']:.3f}  "
              f"pinball_norm_avg={best['pinball_norm_avg']:.3f}  "
              f"(cov_err по сетке: {sub.cov_err_avg.min():.3f}-{sub.cov_err_avg.max():.3f})")

    print(f"\nСохранено: {out_path}")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
