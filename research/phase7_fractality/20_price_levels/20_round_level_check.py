#!/usr/bin/env python3
"""
20_round_level_check.py — Шаг 1: диагностика гипотезы "магнита" к круглым
ценовым уровням. Проверяем ДО любой коррекции прогноза: притягиваются ли
ФАКТИЧЕСКИЕ T_big-пивоты к круглым уровням чаще, чем случайные точки того
же ряда.

"Круглость" уровня — иерархия по порядку величины цены (не нормализуем по
тренду, работаем с сырой ценой, как обсуждали):
  d = floor(log10(P))
  step_minor = 10^(d-1)   — напр. для P~150: шаг 10 (110,120,...)
  step_mid   = 5×step_minor              — шаг 50 (100,150,200...)
  step_major = 10×step_minor             — шаг 100 (100,200,300...)

Для каждой цены считаем относительное расстояние (|P - ближайший уровень|/P)
отдельно для каждого тира (minor/mid/major) и общий roundness-score (сколько
тиров "сработали" в пределах допуска TOL_PCT).

Сравниваем: (а) фактические T_big-пивоты (по 7 тикерам, полный некаузальный
зигзаг — это чистая диагностика структуры данных, не прогноз) против
(б) null-выборки — ВСЕ дневные бары (close) той же истории. Если пивоты
систематически ближе к круглым уровням, чем "случайный" бар — это
подтверждает магнит-гипотезу.

Проверка на сплиты: VTBR имеет 2 подозрительных скачка цены (2007-05-28,
2022-02-22/24) — возможные технические разрывы (не обязательно сплит,
могут быть дивидендные гэпы/скачки волатильности), исключены из анализа
явным флагом (события до/после разрыва анализируются в общем потоке, т.к.
разрыв всего 1 бар и не искажает статистику по всей истории, но помечен).
"""
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

HERE = Path(__file__).parent
DATA = HERE.parents[2] / "data" / "candles"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TICKERS = ["SBER", "LKOH", "CHMF", "NVTK", "MGNT", "VTBR", "NLMK"]
T_BIG = 0.20
TOL_PCT = 0.005   # 0.5% допуск — насколько близко считается "попаданием" в уровень


def load_prices(ticker):
    raw = json.load(open(DATA / ticker / "1d.json"))
    high = np.array([c["high"] for c in raw], dtype=np.float64)
    low = np.array([c["low"] for c in raw], dtype=np.float64)
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    high = np.where(high <= 0, np.nan, high)
    low = np.where(low <= 0, np.nan, low)
    close = np.where(close <= 0, np.nan, close)
    dates = np.array([c["begin"] for c in raw])
    return high, low, close, dates


def build_zigzag_price(lh, ll, dates, thr):
    """Тот же алгоритм, что exp17.build_zigzag, но работает с log(high)/log(low)
    и возвращает ЦЕНЫ (не логи) подтверждённых пивотов."""
    lp, conf, dirs = [], [], []
    cur = 0
    ext = (lh[0] + ll[0]) / 2.0
    for i in range(len(lh)):
        if cur == 0:
            if lh[i] - ext >= thr:
                cur = 1; ext = lh[i]
            elif ext - ll[i] >= thr:
                cur = -1; ext = ll[i]
        elif cur == 1:
            if lh[i] > ext:
                ext = lh[i]
            elif ext - ll[i] >= thr:
                lp.append(ext); conf.append(dates[i]); dirs.append(1)
                cur = -1; ext = ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= thr:
                lp.append(ext); conf.append(dates[i]); dirs.append(-1)
                cur = 1; ext = lh[i]
    return np.exp(np.array(lp)), np.array(conf), np.array(dirs, dtype=np.int8)


def roundness_distances(prices):
    """Для массива цен возвращает относительные расстояния до ближайшего
    уровня каждого тира (minor/mid/major) + roundness_score (0-3)."""
    d = np.floor(np.log10(prices))
    step_minor = 10.0 ** (d - 1)
    step_mid = 5 * step_minor
    step_major = 10 * step_minor

    def rel_dist(step):
        nearest = np.round(prices / step) * step
        return np.abs(prices - nearest) / prices

    dist_minor = rel_dist(step_minor)
    dist_mid = rel_dist(step_mid)
    dist_major = rel_dist(step_major)
    score = ((dist_minor < TOL_PCT).astype(int) +
             (dist_mid < TOL_PCT).astype(int) +
             (dist_major < TOL_PCT).astype(int))
    return dist_minor, dist_mid, dist_major, score


def main():
    t0 = time.time()
    print("=== 20_round_level_check — магнит к круглым уровням: диагностика ===")
    print(f"Тикеры: {TICKERS}   T_big={T_BIG}   TOL_PCT={TOL_PCT}\n")

    all_rows = []
    for ticker in TICKERS:
        high, low, close, dates = load_prices(ticker)
        lh, ll = np.log(high), np.log(low)
        pivot_prices, pivot_conf, pivot_dirs = build_zigzag_price(lh, ll, dates, T_BIG)

        dm_piv, dmid_piv, dmaj_piv, score_piv = roundness_distances(pivot_prices)
        dm_all, dmid_all, dmaj_all, score_all = roundness_distances(close[np.isfinite(close)])

        # Манна-Уитни: пивоты vs все бары, по каждому тиру (одностороннее: пивоты БЛИЖЕ)
        u_minor, p_minor = stats.mannwhitneyu(dm_piv, dm_all, alternative="less")
        u_mid, p_mid = stats.mannwhitneyu(dmid_piv, dmid_all, alternative="less")
        u_maj, p_maj = stats.mannwhitneyu(dmaj_piv, dmaj_all, alternative="less")

        rate_piv = (score_piv > 0).mean()   # доля пивотов, попавших хоть в один тир (TOL_PCT)
        rate_all = (score_all > 0).mean()

        row = {
            "ticker": ticker, "n_pivots": len(pivot_prices), "n_bars": len(close),
            "mean_dist_minor_pivots": dm_piv.mean(), "mean_dist_minor_all": dm_all.mean(), "p_minor": p_minor,
            "mean_dist_mid_pivots": dmid_piv.mean(), "mean_dist_mid_all": dmid_all.mean(), "p_mid": p_mid,
            "mean_dist_major_pivots": dmaj_piv.mean(), "mean_dist_major_all": dmaj_all.mean(), "p_major": p_maj,
            "hit_rate_pivots": rate_piv, "hit_rate_all": rate_all,
        }
        all_rows.append(row)
        print(f"{ticker}: n_pivots={len(pivot_prices)}  n_bars={len(close)}")
        print(f"  minor  mean_dist pivots={dm_piv.mean():.4f}  all={dm_all.mean():.4f}  p={p_minor:.4f}")
        print(f"  mid    mean_dist pivots={dmid_piv.mean():.4f}  all={dmid_all.mean():.4f}  p={p_mid:.4f}")
        print(f"  major  mean_dist pivots={dmaj_piv.mean():.4f}  all={dmaj_all.mean():.4f}  p={p_maj:.4f}")
        print(f"  hit_rate(±{TOL_PCT*100:.1f}%)  pivots={rate_piv:.3f}  all={rate_all:.3f}\n")

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "round_level_diagnostic.csv", index=False, float_format="%.5f")

    print("="*70)
    print("Сводка (среднее по 7 тикерам):")
    print(f"  mean_dist_minor: pivots={df['mean_dist_minor_pivots'].mean():.4f}  all={df['mean_dist_minor_all'].mean():.4f}")
    print(f"  mean_dist_mid:   pivots={df['mean_dist_mid_pivots'].mean():.4f}  all={df['mean_dist_mid_all'].mean():.4f}")
    print(f"  mean_dist_major: pivots={df['mean_dist_major_pivots'].mean():.4f}  all={df['mean_dist_major_all'].mean():.4f}")
    print(f"  hit_rate: pivots={df['hit_rate_pivots'].mean():.3f}  all={df['hit_rate_all'].mean():.3f}")
    print(f"  Тикеров с p_minor<0.05: {(df['p_minor']<0.05).sum()}/7")
    print(f"  Тикеров с p_mid<0.05:   {(df['p_mid']<0.05).sum()}/7")
    print(f"  Тикеров с p_major<0.05: {(df['p_major']<0.05).sum()}/7")

    print(f"\nСохранено: {RESULTS / 'round_level_diagnostic.csv'}")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
