#!/usr/bin/env python3
"""
20c_hvn_predictor.py — проверка: расстояние (в лог-цене) от текущего
T_big-пивота P до ближайшего ВПЕРЕДИ-ЛЕЖАЩЕГО (в направлении следующего
хода) узла высокого объёма (HVN) — как ex-ante предиктор ФАКТИЧЕСКОЙ
величины следующего плеча.

Мотивация (20b_volume_profile_check.py): пивоты систематически лендятся в
зонах НИЗКОЙ объёмной плотности (percentile rank~0.10-0.16 на всех 7
тикерах, p<0.0001) — цена "пролетает" зоны низкого объёма и, предположительно,
тормозит у зон высокого объёма. Гипотеза: расстояние до ближайшего HVN
впереди может предсказывать, докуда долетит следующее движение — признак,
которого не хватало (волатильность/ATR/ER/размер прошлого плеча — 19h/17o —
все не сработали).

Причинно: волюм-профиль строится по скользящему окну WINDOW=252 бара ДО
confirm_date текущего пивота (тот же приём, что в 20b). HVN — первый бин
по направлению следующего хода (противоположно направлению текущего
пивота) с плотностью выше порога (60-й перцентиль ненулевых бинов окна).

SBER, LKOH, CHMF, NVTK, MGNT, VTBR, NLMK, T_big=20%.
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
MIN_HIST_BARS = 60
HVN_PERCENTILE = 60
MIN_HIST_PIVOTS = 30


def load_data(ticker):
    raw = json.load(open(DATA / ticker / "1d.json"))
    high = np.array([c["high"] for c in raw], dtype=np.float64)
    low = np.array([c["low"] for c in raw], dtype=np.float64)
    volume = np.array([c["volume"] for c in raw], dtype=np.float64)
    high = np.where(high <= 0, np.nan, high)
    low = np.where(low <= 0, np.nan, low)
    dates = np.array([c["begin"] for c in raw])
    return np.log(high), np.log(low), volume, dates


def build_zigzag(lh, ll, dates, thr):
    lp, conf_idx, dirs = [], [], []
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
                lp.append(ext); conf_idx.append(i); dirs.append(1)
                cur = -1; ext = ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= thr:
                lp.append(ext); conf_idx.append(i); dirs.append(-1)
                cur = 1; ext = lh[i]
    return np.array(lp), np.array(conf_idx, dtype=int), np.array(dirs, dtype=np.int8)


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


def nearest_hvn_distance(edges, profile, p_log, direction):
    n_bins = len(profile)
    binw = (edges[-1] - edges[0]) / n_bins
    p_idx = int(np.floor((p_log - edges[0]) / binw))
    p_idx = max(0, min(p_idx, n_bins - 1))
    nz = profile[profile > 0]
    if len(nz) < 5:
        return np.nan
    threshold = np.percentile(nz, HVN_PERCENTILE)

    if direction > 0:
        rng = range(p_idx + 1, n_bins)
    else:
        rng = range(p_idx - 1, -1, -1)
    for k in rng:
        if profile[k] >= threshold:
            bin_center = edges[0] + (k + 0.5) * binw
            return abs(bin_center - p_log)
    return np.nan   # не нашли в пределах окна


def main():
    t0 = time.time()
    print("=== 20c_hvn_predictor — расстояние до HVN как предиктор величины хода ===")
    print(f"WINDOW={WINDOW}  N_BINS={N_BINS}  HVN_PERCENTILE={HVN_PERCENTILE}\n")

    all_rows = []
    for ticker in TICKERS:
        lh, ll, volume, dates = load_data(ticker)
        full_lp, full_idx, full_dirs = build_zigzag(lh, ll, dates, T_BIG)
        n_big = len(full_lp)

        for i in range(MIN_HIST_PIVOTS, n_big - 1):
            bar_idx = full_idx[i]
            start = max(0, bar_idx - WINDOW)
            if bar_idx - start < MIN_HIST_BARS:
                continue
            lh_win, ll_win, vol_win = lh[start:bar_idx+1], ll[start:bar_idx+1], volume[start:bar_idx+1]
            res = volume_profile(lh_win, ll_win, vol_win, N_BINS)
            if res is None:
                continue
            edges, profile = res
            if profile.sum() <= 0:
                continue

            P = full_lp[i]
            q_dir = full_dirs[i]
            direction_of_travel = -q_dir
            dist_hvn = nearest_hvn_distance(edges, profile, P, direction_of_travel)
            if not np.isfinite(dist_hvn):
                continue

            actual_leg_abs = abs(full_lp[i + 1] - full_lp[i])
            all_rows.append({"ticker": ticker, "step": i, "dist_hvn": dist_hvn,
                              "actual_leg_abs": actual_leg_abs})

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "hvn_predictor.csv", index=False, float_format="%.6f")
    print(f"n_valid={len(df)} (из них с найденным HVN в пределах окна)\n")

    print("Корреляция dist_hvn vs actual_leg_abs, по тикерам:")
    for ticker in TICKERS:
        g = df[df["ticker"] == ticker]
        if len(g) < 10:
            print(f"  {ticker}: n={len(g)} — недостаточно данных")
            continue
        r, p = stats.pearsonr(g["dist_hvn"], g["actual_leg_abs"])
        rho, ps = stats.spearmanr(g["dist_hvn"], g["actual_leg_abs"])
        ratio = g["actual_leg_abs"].mean() / g["dist_hvn"].mean()
        print(f"  {ticker}: n={len(g):>3d}  r={r:+.3f} (p={p:.4f})  rho={rho:+.3f} (p={ps:.4f})  "
              f"mean(actual)/mean(dist_hvn)={ratio:.3f}")

    r_all, p_all = stats.pearsonr(df["dist_hvn"], df["actual_leg_abs"])
    rho_all, ps_all = stats.spearmanr(df["dist_hvn"], df["actual_leg_abs"])
    ratio_all = df["actual_leg_abs"].mean() / df["dist_hvn"].mean()
    print(f"\nВСЕ ТИКЕРЫ ОБЪЕДИНЁННО: n={len(df)}  r={r_all:+.3f} (p={p_all:.5f})  "
          f"rho={rho_all:+.3f} (p={ps_all:.5f})  mean(actual)/mean(dist_hvn)={ratio_all:.3f}")

    print(f"\nСохранено: {RESULTS / 'hvn_predictor.csv'}")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
