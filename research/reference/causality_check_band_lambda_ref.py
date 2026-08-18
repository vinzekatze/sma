#!/usr/bin/env python3
"""
causality_check_band_lambda_ref.py — проверка каузальности band_lambda_
calibrator_ref.py / smap_band_weighted_ref.py: forecast_live_band на ПОЛНЫХ
данных должен совпадать БИТ-В-БИТ с forecast_live_band на данных, обрезанных
сразу после origin'а (и целевого тикера, и ВСЕХ пиров пула) — иначе где-то
просочилась информация из будущего.

Использование:
    python causality_check_band_lambda_ref.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from band_lambda_calibrator_ref import load_universe_ticker_data, FEATURE_ORDER, mask_ticker_data
from smap_band_weighted_ref import forecast_live_band

TARGET = "GAZP"
T = 0.20
LAMBDAS = {"volume": 5.0, "trend": 8.0, "leg_age": 6.0, "velocity": 3.0, "acceleration": 0.0, "volatility": 4.0}


def main():
    print(f"Загрузка ({TARGET}, T={T*100:.0f}%)…")
    ticker_data_full = load_universe_ticker_data("1d")
    if TARGET not in ticker_data_full:
        raise SystemExit(f"нет данных для {TARGET}")

    ok_count, fail_count = 0, 0

    # ── Тест 1: несколько origin'ов из середины истории, полные данные vs
    #    обрезанные сразу после origin'а ──
    from smap_band_ref import build_zigzag
    lh_t, ll_t, dates_t, _ = ticker_data_full[TARGET]
    q_lp, q_dates, q_dirs = build_zigzag(lh_t, ll_t, dates_t, T, 5)
    print(f"{TARGET}: {len(q_lp)} пивотов T={T*100:.0f}%\n")

    test_origins = [len(q_lp) // 4, len(q_lp) // 2, len(q_lp) * 3 // 4, len(q_lp) - 1]
    for origin in test_origins:
        r_full = forecast_live_band(TARGET, T, LAMBDAS, ticker_data_full, origin_index=origin)
        if r_full is None:
            print(f"origin={origin}: r_full=None (пул/пивотов недостаточно), пропуск")
            continue

        cutoff_date = r_full["origin_date"]
        ticker_data_trunc = mask_ticker_data(ticker_data_full, cutoff_date)
        if TARGET not in ticker_data_trunc:
            print(f"origin={origin}: целевой тикер выпал при обрезке, пропуск")
            continue
        r_trunc = forecast_live_band(TARGET, T, LAMBDAS, ticker_data_trunc, origin_index=None)

        match = True
        if r_trunc is None:
            match = False
            reason = "r_trunc=None, а r_full не None"
        else:
            if r_trunc["origin_date"] != r_full["origin_date"]:
                match = False
                reason = f"origin_date разошёлся: {r_trunc['origin_date']} vs {r_full['origin_date']}"
            else:
                reason = None
                for h in (1, 2):
                    sf, st = r_full["steps"][h], r_trunc["steps"][h]
                    if sf["ok"] != st["ok"]:
                        match = False; reason = f"h={h}: ok разошёлся ({sf['ok']} vs {st['ok']})"; break
                    if sf["ok"]:
                        if sf["pool_size"] != st["pool_size"]:
                            match = False; reason = f"h={h}: pool_size {sf['pool_size']} vs {st['pool_size']}"; break
                        for q in sf["band_price"]:
                            if not np.isclose(sf["band_price"][q], st["band_price"][q], rtol=1e-9):
                                match = False
                                reason = f"h={h}: band[{q}] {sf['band_price'][q]} vs {st['band_price'][q]}"
                                break
                        if not match:
                            break

        status = "OK" if match else "FAIL"
        print(f"origin={origin} ({cutoff_date[:10]}): {status}" + (f"  — {reason}" if not match else ""))
        if match:
            ok_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*50}\nИтог: {ok_count} OK, {fail_count} FAIL")
    if fail_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
