#!/usr/bin/env python3
"""
20e_volume_weighting_comparison.py — две версии подмешивания объёма в
S-map, сравнение с baseline (без объёма):

A) "CRUDE" (негативный контроль, воспроизводит предполагаемую старую
   ошибку пользователя): log(volume) добавляется КАК ДОПОЛНИТЕЛЬНАЯ
   КООРДИНАТА вектора признаков — конкатенация разнородных единиц
   (лог-доходности ~0.01-0.5 vs log-volume ~14-18) без нормировки,
   попадает в тот же евклидов d, что используется для θ-локализации.
   Ожидание: должно ухудшить результат (структурная причина понятна —
   объёмная координата доминирует в расстоянии).

B) "REFINED" (аккуратный вариант): объём НЕ входит в вектор признаков и
   евклидово расстояние вообще. Вместо этого — отдельный, уже нормированный
   (0-1 перцентильный ранг) множитель веса:
     w_j = exp(-θ·d_j/d_mean) × exp(-λ·(rank_j − rank_query)²)
   rank_j/rank_query — density_rank_at_origin (из 20b/20d): перцентиль
   плотности объёма В ТОЧКЕ АНКЕРА (не исхода) относительно своего же
   скользящего окна на 252 бара. Гипотеза: сосед полезнее, если он сам
   находился в похожей по разрежённости точке, что и текущий запрос —
   единицы совпадают по построению (оба 0-1), конкатенации нет.

rank_j — свойство ТОЛЬКО исторического события (не запроса), поэтому
кэшируется по (тикер, номер бара) один раз, не пересчитывается для каждой
пары (запрос, сосед).

LOO по 7 тикерам (как в 19j/20d): λ калибруется на остальных 6, для CRUDE
негативный контроль — просто применяется поверх уже откалиброванных
m/θ/T_ratio (без переоткалибровки θ под новую размерность — воспроизводим
"грубое" использование, а не честный лучший случай для CRUDE).
"""
import importlib.util
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
EXP17_DIR = HERE.parents[0] / "17_large_scale_pooled"
DATA = HERE.parents[2] / "data" / "candles"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

spec17 = importlib.util.spec_from_file_location("exp17", EXP17_DIR / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

T_BIG = 0.20
ARM = "D_allpeers"
WINDOW = 252
N_BINS = 50
MIN_HIST_BARS = 60

CALIB = {
    "SBER": {"m": 3, "theta": 25.697, "T_ratio": 0.8987},
    "LKOH": {"m": 4, "theta": 24.817, "T_ratio": 0.8331},
    "CHMF": {"m": 3, "theta": 11.492, "T_ratio": 0.9238},
    "NVTK": {"m": 2, "theta": 15.158, "T_ratio": 0.8080},
    "MGNT": {"m": 2, "theta": 11.814, "T_ratio": 0.6110},
    "VTBR": {"m": 3, "theta": 0.731,  "T_ratio": 0.6169},
    "NLMK": {"m": 2, "theta": 2.315,  "T_ratio": 0.8641},
}
LAMBDA_GRID = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

_raw_cache = {}
_rank_cache = {}


def get_raw(ticker):
    if ticker not in _raw_cache:
        raw = json.load(open(DATA / ticker / "1d.json"))
        high = np.array([c["high"] for c in raw], dtype=np.float64)
        low = np.array([c["low"] for c in raw], dtype=np.float64)
        close = np.array([c["close"] for c in raw], dtype=np.float64)
        volume = np.array([c["volume"] for c in raw], dtype=np.float64)
        high = np.where(high <= 0, np.nan, high); low = np.where(low <= 0, np.nan, low)
        close = np.where(close <= 0, np.nan, close)
        dates = np.array([c["begin"] for c in raw])
        _raw_cache[ticker] = (np.log(high), np.log(low), np.log(close), volume, dates)
    return _raw_cache[ticker]


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


def density_rank(edges, profile, q, ref_log_prices):
    n_bins = len(profile)
    binw = (edges[-1] - edges[0]) / n_bins

    def lookup(x):
        idx = int(np.floor((x - edges[0]) / binw))
        return profile[max(0, min(idx, n_bins - 1))]

    dens_q = lookup(q)
    dens_ref = np.array([lookup(x) for x in ref_log_prices if np.isfinite(x)])
    if len(dens_ref) < MIN_HIST_BARS:
        return np.nan
    return float((dens_ref < dens_q).mean() + 0.5 * (dens_ref == dens_q).mean())


def get_rank(ticker, bar_idx, pivot_log_price):
    key = (ticker, bar_idx)
    if key in _rank_cache:
        return _rank_cache[key]
    lh, ll, lc, volume, dates = get_raw(ticker)
    start = max(0, bar_idx - WINDOW)
    if bar_idx - start < MIN_HIST_BARS:
        _rank_cache[key] = np.nan
        return np.nan
    res = volume_profile(lh[start:bar_idx+1], ll[start:bar_idx+1], volume[start:bar_idx+1], N_BINS)
    if res is None:
        _rank_cache[key] = np.nan
        return np.nan
    edges, profile = res
    if profile.sum() <= 0:
        _rank_cache[key] = np.nan
        return np.nan
    r = density_rank(edges, profile, pivot_log_price, lc[start:bar_idx+1])
    _rank_cache[key] = r
    return r


def get_logvol(ticker, bar_idx):
    _, _, _, volume, _ = get_raw(ticker)
    v = volume[bar_idx]
    return float(np.log(v)) if v > 0 else np.nan


def bar_idx_of(dates_full, confirm_date):
    return int(np.searchsorted(dates_full, confirm_date, side="right")) - 1


def build_pool_rows_with_extra(lp, dirs, extra2, m):
    """extra2: массив (n,2) — [rank, logvol] на каждый пивот. Возвращает
    feats, tgts, dirs_out, extra_out(k,2) с той же фильтрацией, что exp17.build_pool_rows."""
    n = len(lp)
    if n < m + 2:
        return (np.empty((0, m)), np.empty(0), np.empty(0, dtype=np.int8), np.empty((0, 2)))
    idx = np.arange(m, n - 1)
    feats = np.zeros((len(idx), m)); tgts = np.zeros(len(idx))
    ds = np.zeros(len(idx), dtype=np.int8); ex = np.zeros((len(idx), 2))
    for row, j in enumerate(idx):
        for lag in range(m):
            feats[row, lag] = lp[j - lag] - lp[j - lag - 1]
        tgts[row] = lp[j + 1] - lp[j]
        ds[row] = dirs[j]
        ex[row] = extra2[j]
    finite = np.all(np.isfinite(feats), axis=1) & np.isfinite(tgts) & np.all(np.isfinite(ex), axis=1)
    return feats[finite], tgts[finite], ds[finite], ex[finite]


def _smap_vol(qvec, feats, tgts, ranks_j, rank_query, min_pool, theta, lam):
    if len(tgts) < min_pool:
        return np.nan
    d = np.linalg.norm(feats - qvec, axis=1)
    d_mean = d.mean()
    if d_mean < 1e-14:
        return float(tgts.mean())
    w_price = np.exp(-theta * d / d_mean)
    w_vol = np.exp(-lam * (ranks_j - rank_query) ** 2) if lam > 0 else np.ones_like(d)
    w = w_price * w_vol
    sw = np.sqrt(w)
    A = np.column_stack([np.ones(len(tgts)), feats]) * sw[:, None]
    b = tgts * sw
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(c[0] + c[1:] @ qvec)


def run_walkforward(target, m, theta, t_ratio):
    target_data = exp17.load_ticker(target)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != target]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != target}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    _, _, lc_full, _, dates_full = get_raw(target)
    t_frac = t_ratio * T_BIG
    full_lp, full_conf, _ = exp17.build_zigzag(lh, ll, dates, T_BIG)
    n_big = len(full_lp)

    records = []
    for i in range(exp17.MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
        t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]

        own_big_lp, own_big_conf, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, T_BIG)
        if len(own_big_lp) < m + 1 or own_big_lp[-1] != full_lp[i]:
            continue
        qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue
        q_dir = int(own_big_dir[-1])
        P_log = float(own_big_lp[-1])

        own_bar_idx = cutoff_idx - 1
        rank_query = get_rank(target, own_bar_idx, P_log)
        logvol_query = get_logvol(target, own_bar_idx)
        if not np.isfinite(rank_query) or not np.isfinite(logvol_query):
            continue

        def extra_for_chain(ticker, dates_arr, lp_arr):
            dates_full_t = get_raw(ticker)[4]
            out = np.zeros((len(lp_arr), 2))
            for k in range(len(lp_arr)):
                bidx = bar_idx_of(dates_full_t, dates_arr[k])
                out[k, 0] = get_rank(ticker, bidx, lp_arr[k])
                out[k, 1] = get_logvol(ticker, bidx)
            return out

        pool_feats, pool_tgts, pool_dirs, pool_extra = [], [], [], []

        ex = extra_for_chain(target, own_big_conf, own_big_lp)
        f, tg, dd, ee = build_pool_rows_with_extra(own_big_lp, own_big_dir, ex, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd); pool_extra.append(ee)

        own_frac_lp, own_frac_conf, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
        ex = extra_for_chain(target, own_frac_conf, own_frac_lp)
        f, tg, dd, ee = build_pool_rows_with_extra(own_frac_lp, own_frac_dir, ex, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd); pool_extra.append(ee)

        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < m + 2:
                continue
            p_lp, p_conf, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                                      peer_data[peer]["ll"][:p_cutoff],
                                                      p_dates[:p_cutoff], t_frac)
            ex = extra_for_chain(peer, p_conf, p_lp)
            f, tg, dd, ee = build_pool_rows_with_extra(p_lp, p_dir, ex, m)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd); pool_extra.append(ee)

        feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts)
        dirs = np.concatenate(pool_dirs); extra = np.concatenate(pool_extra)
        mask = dirs == q_dir
        feats_d, tgts_d, extra_d = feats[mask], tgts[mask], extra[mask]
        if len(feats_d):
            d = np.linalg.norm(feats_d - qvec, axis=1)
            dup = d < exp17.DUP_EPS
            if dup.any():
                feats_d, tgts_d, extra_d = feats_d[~dup], tgts_d[~dup], extra_d[~dup]
        if len(tgts_d) < m + 2:
            continue

        lr_base = exp17._smap(qvec, feats_d, tgts_d, m + 2, theta)

        qvec_crude = np.concatenate([qvec, [logvol_query]])
        feats_crude = np.column_stack([feats_d, extra_d[:, 1]])
        lr_crude = exp17._smap(qvec_crude, feats_crude, tgts_d, m + 2, theta)

        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)

        row = {"ticker": target, "step": i, "P_log": P_log, "actual_price": actual_price,
               "pers_err": pers_err, "lr_base": lr_base, "lr_crude": lr_crude,
               "rank_query": rank_query}
        for lam in LAMBDA_GRID:
            row[f"lr_refined_{lam}"] = _smap_vol(qvec, feats_d, tgts_d, extra_d[:, 0], rank_query,
                                                  m + 2, theta, lam)
        records.append(row)
    return pd.DataFrame(records)


def rmae(pred_log, df):
    pred_price = np.exp(df["P_log"] + pred_log)
    abs_err = (pred_price - df["actual_price"]).abs()
    dz = df["pers_err"].mean()
    return float(abs_err.mean() / dz) if dz > 1e-12 else np.nan


def main():
    t0 = time.time()
    print("=== 20e_volume_weighting_comparison — CRUDE (негативный контроль) vs REFINED ===\n")

    all_df = {}
    for ticker, params in CALIB.items():
        print(f"Walk-forward {ticker}...")
        df = run_walkforward(ticker, params["m"], params["theta"], params["T_ratio"])
        all_df[ticker] = df
        r_base = rmae(df["lr_base"], df)
        r_crude = rmae(df["lr_crude"], df)
        print(f"  n={len(df)}  rMAE base={r_base:.4f}  rMAE CRUDE(+logvol в векторе)={r_crude:.4f}  "
              f"Δ={r_crude-r_base:+.4f}")

    combined = pd.concat(all_df.values(), ignore_index=True)
    combined.to_csv(RESULTS / "volume_weighting_raw.csv", index=False, float_format="%.6f")

    print(f"\n{'='*70}\n=== REFINED: LOO по λ (мультипликативный вес по rank-схожести) ===")
    loo_rows = []
    for held_out in CALIB:
        other = pd.concat([all_df[t] for t in CALIB if t != held_out], ignore_index=True)
        best_lam, best_v = None, float("inf")
        for lam in LAMBDA_GRID:
            v = rmae(other[f"lr_refined_{lam}"], other)
            if v < best_v:
                best_v, best_lam = v, lam
        target_df = all_df[held_out]
        r_base = rmae(target_df["lr_base"], target_df)
        r_crude = rmae(target_df["lr_crude"], target_df)
        r_ref = rmae(target_df[f"lr_refined_{best_lam}"], target_df)
        print(f"  {held_out:<6s}  chosen λ={best_lam} (rMAE_on_other6={best_v:.4f})  "
              f"base={r_base:.4f}  crude={r_crude:.4f}  refined={r_ref:.4f}  "
              f"Δcrude={r_crude-r_base:+.4f}  Δrefined={r_ref-r_base:+.4f}")
        loo_rows.append({"ticker": held_out, "rMAE_base": r_base, "rMAE_crude": r_crude,
                          "lambda_chosen": best_lam, "rMAE_refined": r_ref,
                          "delta_crude": r_crude - r_base, "delta_refined": r_ref - r_base})

    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(RESULTS / "volume_weighting_loo_results.csv", index=False, float_format="%.5f")
    print(f"\n{'='*70}")
    print(loo_df.to_string(index=False))
    print(f"\nСреднее: base={loo_df['rMAE_base'].mean():.4f}  crude={loo_df['rMAE_crude'].mean():.4f}  "
          f"refined={loo_df['rMAE_refined'].mean():.4f}")
    print(f"CRUDE лучше baseline на: {(loo_df['delta_crude']<0).sum()}/7 тикерах")
    print(f"REFINED лучше baseline на: {(loo_df['delta_refined']<0).sum()}/7 тикерах")

    print(f"\nСохранено: {RESULTS}/volume_weighting_raw.csv, volume_weighting_loo_results.csv")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
