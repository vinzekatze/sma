#!/usr/bin/env python3
"""
20b_volume_profile_check.py — Шаг 1 (объёмный вариант): диагностика
гипотезы "магнита" к уровням с высоким историческим объёмом (Volume
Profile / High Volume Nodes), вместо круглых чисел (20_round_level_check
дал слабый/непоследовательный результат).

Методология (по мотивам замечания о "правильном" построении уровней —
не по всей истории целиком, а по СВЕЖЕМУ окну, т.к. память рынка о старых
уровнях выветривается, особенно за 15-20 лет истории):

Для каждого подтверждённого T_big-пивота i (дата confirm_date_i):
  1. Берём СКОЛЬЗЯЩЕЕ окно — последние WINDOW=252 бара (≈1 год) ДО
     confirm_date_i (каузально).
  2. Строим объёмный профиль по этому окну: объём каждого дня размазывается
     РАВНОМЕРНО по [log(low), log(high)] этого дня — стандартный приём
     аппроксимации Volume Profile по дневным свечам (тиковых данных нет).
  3. Смотрим плотность объёма В ТОЧКЕ цены пивота (интерполяция по бинам).
  4. Сравниваем с плотностью в ТОЧКАХ close всех ОСТАЛЬНЫХ баров ТОГО ЖЕ
     окна — получаем percentile rank пивота внутри его локального
     контекста (0.5 = типичная точка, 1.0 = самая "объёмная" точка окна).

Если пивоты систематически лендятся в точках с percentile rank > 0.5 —
подтверждение гипотезы (движение притягивается к объёмным уровням).
Полностью каузально и нормировано локально — не зависит от глобального
дрейфа цены за десятилетия.
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
WINDOW = 252
N_BINS = 50
MIN_HIST_BARS = 60   # минимум баров в окне, чтобы профиль был осмысленным


def load_data(ticker):
    raw = json.load(open(DATA / ticker / "1d.json"))
    high = np.array([c["high"] for c in raw], dtype=np.float64)
    low = np.array([c["low"] for c in raw], dtype=np.float64)
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    volume = np.array([c["volume"] for c in raw], dtype=np.float64)
    high = np.where(high <= 0, np.nan, high)
    low = np.where(low <= 0, np.nan, low)
    close = np.where(close <= 0, np.nan, close)
    dates = np.array([c["begin"] for c in raw])
    return np.log(high), np.log(low), np.log(close), volume, dates


def build_zigzag_price(lh, ll, dates, thr):
    lp, conf_idx = [], []
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
                lp.append(ext); conf_idx.append(i)
                cur = -1; ext = ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= thr:
                lp.append(ext); conf_idx.append(i)
                cur = 1; ext = lh[i]
    return np.array(lp), np.array(conf_idx, dtype=int)


def volume_profile(lh_win, ll_win, vol_win, n_bins):
    lo = np.nanmin(ll_win); hi = np.nanmax(lh_win)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    profile = np.zeros(n_bins)
    binw = (hi - lo) / n_bins
    for j in range(len(lh_win)):
        a, b, v = ll_win[j], lh_win[j], vol_win[j]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(v)) or b <= a:
            continue
        b0 = int(np.floor((a - lo) / binw)); b1 = int(np.floor((b - lo) / binw))
        b0 = max(0, min(b0, n_bins - 1)); b1 = max(0, min(b1, n_bins - 1))
        if b0 == b1:
            profile[b0] += v
        else:
            span = b - a
            for k in range(b0, b1 + 1):
                bin_lo, bin_hi = lo + k * binw, lo + (k + 1) * binw
                overlap = max(0.0, min(b, bin_hi) - max(a, bin_lo))
                profile[k] += v * (overlap / span)
    return edges, profile


def lookup_density(edges, profile, q):
    n_bins = len(profile)
    idx = int(np.floor((q - edges[0]) / (edges[-1] - edges[0]) * n_bins))
    idx = max(0, min(idx, n_bins - 1))
    return profile[idx]


def main():
    t0 = time.time()
    print("=== 20b_volume_profile_check — магнит к объёмным уровням (Volume Profile) ===")
    print(f"WINDOW={WINDOW} баров  N_BINS={N_BINS}\n")

    all_ranks = []
    summary_rows = []
    for ticker in TICKERS:
        lh, ll, lc, volume, dates = load_data(ticker)
        pivot_lp, pivot_idx = build_zigzag_price(lh, ll, dates, T_BIG)

        ranks = []
        for pj, i in zip(pivot_lp, pivot_idx):
            start = max(0, i - WINDOW)
            if i - start < MIN_HIST_BARS:
                continue
            lh_win, ll_win, vol_win = lh[start:i+1], ll[start:i+1], volume[start:i+1]
            res = volume_profile(lh_win, ll_win, vol_win, N_BINS)
            if res is None:
                continue
            edges, profile = res
            if profile.sum() <= 0:
                continue

            dens_pivot = lookup_density(edges, profile, pj)
            lc_win = lc[start:i+1]
            dens_others = np.array([lookup_density(edges, profile, q) for q in lc_win if np.isfinite(q)])
            if len(dens_others) < MIN_HIST_BARS:
                continue
            rank = stats.percentileofscore(dens_others, dens_pivot, kind="mean") / 100.0
            ranks.append(rank)

        ranks = np.array(ranks)
        all_ranks.extend(ranks.tolist())
        mean_r = ranks.mean() if len(ranks) else np.nan
        # одновыборочный тест против нуля 0.5 (Wilcoxon signed-rank на (rank-0.5))
        if len(ranks) > 5:
            stat, p = stats.wilcoxon(ranks - 0.5)
        else:
            p = np.nan
        print(f"{ticker}: n_pivots_valid={len(ranks)}  mean_percentile_rank={mean_r:.4f}  "
              f"median={np.median(ranks):.4f}  p(Wilcoxon vs 0.5)={p:.4f}")
        summary_rows.append({"ticker": ticker, "n_pivots": len(ranks), "mean_rank": mean_r,
                              "median_rank": np.median(ranks) if len(ranks) else np.nan, "p_value": p})

    df = pd.DataFrame(summary_rows)
    df.to_csv(RESULTS / "volume_profile_diagnostic.csv", index=False, float_format="%.5f")

    all_ranks = np.array(all_ranks)
    stat, p_all = stats.wilcoxon(all_ranks - 0.5)
    print(f"\n{'='*70}")
    print(f"ВСЕ ТИКЕРЫ ОБЪЕДИНЁННО: n={len(all_ranks)}  mean_rank={all_ranks.mean():.4f}  "
          f"median={np.median(all_ranks):.4f}  p(Wilcoxon vs 0.5)={p_all:.5f}")
    print(f"Тикеров с mean_rank>0.5: {(df['mean_rank']>0.5).sum()}/7")
    print(f"Тикеров с p<0.05: {(df['p_value']<0.05).sum()}/7")

    print(f"\nСохранено: {RESULTS / 'volume_profile_diagnostic.csv'}")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
