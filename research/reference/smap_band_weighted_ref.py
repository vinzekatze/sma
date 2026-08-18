#!/usr/bin/env python3
"""
smap_band_weighted_ref.py — ЖИВОЙ прогноз полосы неопределённости S-map с
весами пула (λ, band_lambda_calibrator_ref.py), а не только θ=0-равномерным
кернелом (smap_band_ref.py). Тонкая обёртка над строительными блоками
калибратора — "дай сегодняшнюю полосу с уже откалиброванными λ", без
повторного прогона калибровки.

Схема причинности НЕ меняется относительно smap_band_ref.py/band_lambda_
calibrator_ref.py: build_causal_pool_with_mixed_ranks обрезает каждого
пира по cutoff_date=origin_date внутри себя; bar-нативные ранги (объём,
тренд, скорость, ускорение, волатильность) вычислены заранее на ПОЛНОЙ
истории тикера, но каждое значение ranks[i] зависит только от 252 баров
ДО i — форвардная обрезка данных ПОСЛЕ origin'а не меняет значение ранга
В origin'е, поэтому пере-вычислять их на обрезанном массиве не нужно (то
же рассуждение, что в evaluate_origins_mixed калибратора). Каузальность
проверена causality_check_band_lambda_ref.py.

Использование:
    python smap_band_weighted_ref.py SBER --t 0.20 \\
        --lambda volume=0 --lambda trend=10 --lambda leg_age=8 \\
        --lambda velocity=0 --lambda acceleration=0 --lambda volatility=2
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from smap_band_ref import build_zigzag, weighted_quantile, UNIVERSE
from band_lambda_calibrator_ref import (
    M, MIN_BARS, K_LEG, Q_LEVELS, FEATURE_ORDER, BAR_FEATURES, PIVOT_FEATURES,
    causal_pivot_percentile_rank, leg_age_raw,
    build_causal_pool_with_mixed_ranks, load_universe_ticker_data,
)


def pool_values_and_weights(target: str, t: float, lambdas: dict, ticker_data: dict,
                            origin_index: int | None = None) -> dict | None:
    """Низкоуровневая версия forecast_live_band — вместо готовой полосы на
    фиксированных Q_LEVELS отдаёт СЫРЫЕ (значения, веса) пула на h=1/h=2.
    Нужна для UI с произвольными реактивными уровнями (напр. app9: набор
    процентов задаётся чекбоксами, полоса пересчитывается на каждое
    изменение БЕЗ повторной загрузки пула — дорогая часть тут, дешёвая
    (weighted_quantile по сохранённым values/weights) в вызывающем коде).

    Возвращает None или {"origin_date","origin_price","origin_direction",
    "steps": {1: {"ok","pool_size","values","weights"}, 2: {...}}}."""
    lh_t, ll_t, dates_t, rank_dict_t = ticker_data[target]
    q_lp, q_dates, q_dirs = build_zigzag(lh_t, ll_t, dates_t, t, MIN_BARS)
    if len(q_lp) < M + 3:
        return None
    origin = origin_index if origin_index is not None else len(q_lp) - 1
    if origin < 0 or origin >= len(q_lp):
        return None

    q_rank_pivot_leg_age = causal_pivot_percentile_rank(leg_age_raw(q_dates, dates_t), K_LEG)
    origin_date = str(q_dates[origin])
    origin_bar = int(np.searchsorted(dates_t, origin_date))
    origin_bar = min(origin_bar, len(rank_dict_t[BAR_FEATURES[0]]) - 1)
    rank_query = {f: rank_dict_t[f][origin_bar] for f in BAR_FEATURES}
    rank_query["leg_age"] = q_rank_pivot_leg_age[origin]
    if any(not np.isfinite(v) for v in rank_query.values()):
        return None

    origin_direction = int(q_dirs[origin])
    origin_log_price = float(q_lp[origin])

    result = {
        "origin_date": origin_date, "origin_price": float(np.exp(origin_log_price)),
        "origin_log_price": origin_log_price, "origin_direction": origin_direction, "steps": {},
    }
    for h in (1, 2):
        _f, ptr, pdir, pranks = build_causal_pool_with_mixed_ranks(
            origin_date, ticker_data, t, M, h, MIN_BARS)
        mask = pdir == origin_direction
        if mask.sum() < M + 2:
            result["steps"][h] = {"ok": False, "pool_size": int(mask.sum())}
            continue
        sq = np.zeros(int(mask.sum()))
        for f in FEATURE_ORDER:
            sq += lambdas[f] * (pranks[f][mask] - rank_query[f]) ** 2
        w = np.exp(-sq)
        result["steps"][h] = {"ok": True, "pool_size": int(mask.sum()), "values": ptr[mask], "weights": w}
    return result


def forecast_live_band(target: str, t: float, lambdas: dict, ticker_data: dict,
                       origin_index: int | None = None, q_levels: tuple = Q_LEVELS) -> dict | None:
    """Считает откалиброванную полосу (h=1 уход, h=2 уход+возврат) для
    ОДНОГО origin'а.

    target        — тикер, должен быть ключом ticker_data (со своим
                     bar-нативным rank_dict)
    t              — T_query=T_pool
    lambdas        — {feature: λ}, все 6 ключей FEATURE_ORDER обязательны
    ticker_data    — {ticker: (log_highs, log_lows, dates, bar_rank_dict)},
                     см. band_lambda_calibrator_ref.load_universe_ticker_data
                     ПОЛНАЯ (необрезанная) история — обрезка по origin'у
                     происходит через cutoff_date внутри build_causal_pool_
                     with_mixed_ranks, не здесь.
    origin_index   — индекс пивота T_query зигзага. None → последний
                     доступный (живой прогноз "на сегодня").

    Возвращает None, если пивотов/пула недостаточно. Иначе:
      {"origin_date", "origin_price", "origin_direction",
       "steps": {1: {...}, 2: {...}}}
      где steps[h] либо {"ok": False, "pool_size": int}, либо
      {"ok": True, "pool_size": int, "band_log_return": {q: lr},
       "band_price": {q: price}}.
    """
    raw = pool_values_and_weights(target, t, lambdas, ticker_data, origin_index)
    if raw is None:
        return None
    origin_log_price = raw["origin_log_price"]
    result = {
        "origin_date": raw["origin_date"], "origin_price": raw["origin_price"],
        "origin_direction": raw["origin_direction"], "steps": {},
    }
    for h in (1, 2):
        s = raw["steps"][h]
        if not s["ok"]:
            result["steps"][h] = {"ok": False, "pool_size": s["pool_size"]}
            continue
        band_lr = weighted_quantile(s["values"], s["weights"], q_levels)
        band_price = {q: float(np.exp(origin_log_price + lr)) for q, lr in band_lr.items()}
        result["steps"][h] = {
            "ok": True, "pool_size": s["pool_size"],
            "band_log_return": band_lr, "band_price": band_price,
        }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Живой прогноз откалиброванной S-map полосы (λ-веса пула)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ticker", help="Целевой тикер (напр. SBER)")
    parser.add_argument("--interval", default="1d", metavar="IV")
    parser.add_argument("--t", type=float, default=0.20, metavar="T")
    parser.add_argument("--lambda", dest="lambdas", action="append", default=[], metavar="FEATURE=VALUE",
                        help="Повторяемый параметр, напр. --lambda trend=10. "
                             "Признаки без явного значения → λ=0.")
    args = parser.parse_args()

    lambdas = {f: 0.0 for f in FEATURE_ORDER}
    for kv in args.lambdas:
        k, v = kv.split("=")
        if k not in FEATURE_ORDER:
            raise SystemExit(f"неизвестный признак '{k}', допустимы: {FEATURE_ORDER}")
        lambdas[k] = float(v)

    t0 = time.time()
    print(f"Загрузка кросс-тикерного пула + признаков ({len(UNIVERSE)} тикеров)…")
    ticker_data = load_universe_ticker_data(args.interval)
    if args.ticker not in ticker_data:
        raise SystemExit(f"Нет данных для {args.ticker} ({args.interval})")
    print(f"загружено {len(ticker_data)}/{len(UNIVERSE)} тикеров  [{time.time()-t0:.1f}s]\n")

    print(f"λ: {lambdas}")
    r = forecast_live_band(args.ticker, args.t, lambdas, ticker_data)
    if r is None:
        raise SystemExit("Недостаточно пивотов/пула для прогноза")

    dir_label = "▲ HIGH" if r["origin_direction"] == 1 else "▼ LOW"
    print(f"\n{args.ticker} origin={r['origin_date'][:16]}  цена={r['origin_price']:.4f}  {dir_label}")
    for h, label in [(1, "Шаг 1 (уход)"), (2, "Шаг 2 (уход+возврат)")]:
        s = r["steps"][h]
        if not s["ok"]:
            print(f"  {label}: невозможен, пул={s['pool_size']}")
            continue
        b = s["band_price"]
        print(f"  {label}  пул={s['pool_size']}: "
              f"10%={b[0.1]:.4f}  25%={b[0.25]:.4f}  50%={b[0.5]:.4f}  75%={b[0.75]:.4f}  90%={b[0.9]:.4f}")
    print(f"\n[{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
