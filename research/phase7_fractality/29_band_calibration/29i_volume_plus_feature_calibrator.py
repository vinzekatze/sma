#!/usr/bin/env python3
"""
29i_volume_plus_feature_calibrator.py — проверяет, добавляет ли ВТОРАЯ фича
(trend/velocity/acceleration) что-то СВЕРХ уже откалиброванного объёмного
веса, а не ищет "лучший единственный признак" (по решению пользователя
2026-07-08: цель — выгодный НАБОР признаков, не один победитель).

Схема — координатный спуск, шаг 2 (шаг 1 = 29g_lambda_volume_calibrator.py,
уже сделан и зафиксирован): λ_vol ЗАФИКСИРОВАН на уже найденном для тикера
значении (из 29g, ниже — LAMBDA_VOL_BY_TICKER), ищется только λ_feat поверх
него. Вес: w = exp(-λ_vol·d_vol² - λ_feat·d_feat²) — оба фактора внутри
ОДНОГО кернела (не последовательная фильтрация/коррекция).

Для SBER (λ_vol=0 — объём там эффекта не дал, добавлять поверх нечего) —
по решению пользователя проверяются trend/velocity/acceleration САМИ ПО
СЕБЕ (без объёма), тем же однопризнаковым механизмом, что в 29h.

Признаки (trend/velocity/acceleration) переиспользованы из 29h_trend_
velocity_accel_screen.py напрямую — не дублируются здесь.

Методология калибровки λ_feat — та же честная схема (сетка+сглаживание,
несколько dev-сплитов, отдельный holdout), что в 29f/29g.

Использование:
    python 29i_volume_plus_feature_calibrator.py
"""
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REF_DIR = HERE.parents[1] / "reference"
sys.path.insert(0, str(REF_DIR))
from smap_band_ref import build_zigzag, build_query_vector, weighted_quantile, UNIVERSE

spec29 = importlib.util.spec_from_file_location("calib29", HERE / "29_smap_band_calibrator.py")
calib29 = importlib.util.module_from_spec(spec29)
spec29.loader.exec_module(calib29)

spec29f = importlib.util.spec_from_file_location("calib29f", HERE / "29f_t_calibrator.py")
calib29f = importlib.util.module_from_spec(spec29f)
spec29f.loader.exec_module(calib29f)

spec29g = importlib.util.spec_from_file_location("calib29g", HERE / "29g_lambda_volume_calibrator.py")
calib29g = importlib.util.module_from_spec(spec29g)
spec29g.loader.exec_module(calib29g)

spec29h = importlib.util.spec_from_file_location("screen29h", HERE / "29h_trend_velocity_accel_screen.py")
screen29h = importlib.util.module_from_spec(spec29h)
spec29h.loader.exec_module(screen29h)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

T_FIXED = 0.20
M = calib29.M
MIN_BARS = calib29f.MIN_BARS
COVERAGE_LEVELS = calib29f.COVERAGE_LEVELS
MIN_CALIB_ORIGINS = calib29f.MIN_CALIB_ORIGINS
MIN_TEST_ORIGINS = calib29f.MIN_TEST_ORIGINS
DEV_SPLIT_FRACS = [0.55, 0.65, 0.75, 0.85]
SMOOTH_WINDOW = calib29f.SMOOTH_WINDOW

# λ_vol, зафиксированный по итогам 29g (results/run_log_29g_lambda_calibrator_final.txt,
# 2026-07-08) — источник истины, не пересчитывается здесь
LAMBDA_VOL_BY_TICKER = {
    "SBER": 0.0, "LKOH": 2.0, "GAZP": 11.0, "MGNT": 12.0,
    "CHMF": 1.0, "MTSS": 2.0, "PLZL": 9.0,
}
FEATURE_LAMBDA_LO, FEATURE_LAMBDA_HI, FEATURE_LAMBDA_STEP = 0.0, 16.0, 2.0   # разведочный шаг, как в 29h


def build_causal_pool_with_2ranks(cutoff_date, ticker_data, t_pool, embedding_dim, horizon, min_bars):
    """ticker_data: {ticker: (log_highs, log_lows, dates, rank_vol, rank_feat)}"""
    feats_l, tars_l, dirs_l, rv_l, rf_l = [], [], [], [], []
    for ticker, (lh, ll, dates, rank_vol, rank_feat) in ticker_data.items():
        mask = dates <= cutoff_date
        if mask.sum() < embedding_dim + horizon + 2:
            continue
        lh_c, ll_c, dt_c = lh[mask], ll[mask], dates[mask]
        rv_c, rf_c = rank_vol[mask], rank_feat[mask]
        pivot_prices, confirm_dates, pivot_dirs = build_zigzag(lh_c, ll_c, dt_c, t_pool, min_bars)
        if len(pivot_prices) == 0:
            continue
        bar_idx = np.searchsorted(dt_c, confirm_dates)
        piv_rv, piv_rf = rv_c[bar_idx], rf_c[bar_idx]

        n = len(pivot_prices)
        valid = np.arange(embedding_dim, n - horizon)
        if len(valid) == 0:
            continue
        lag_idx = valid[:, None] - np.arange(embedding_dim)[None, :]
        feats = pivot_prices[lag_idx] - pivot_prices[lag_idx - 1]
        tars = pivot_prices[valid + horizon] - pivot_prices[valid]
        dirs = pivot_dirs[valid]
        rv = piv_rv[valid]
        rf = piv_rf[valid]
        finite = (np.all(np.isfinite(feats), axis=1) & np.isfinite(tars)
                 & np.isfinite(rv) & np.isfinite(rf))
        if finite.sum():
            feats_l.append(feats[finite]); tars_l.append(tars[finite])
            dirs_l.append(dirs[finite]); rv_l.append(rv[finite]); rf_l.append(rf[finite])

    if not tars_l:
        return (np.zeros((0, embedding_dim)), np.zeros(0), np.zeros(0, dtype=np.int8),
               np.zeros(0), np.zeros(0))
    return (np.vstack(feats_l), np.concatenate(tars_l), np.concatenate(dirs_l),
           np.concatenate(rv_l), np.concatenate(rf_l))


def evaluate_origins_combined(target, lam_vol, lam_feat, origins, q_lp, q_dates, q_dirs,
                              rank_vol_t, rank_feat_t, ticker_data):
    q_levels_needed = sorted(set(calib29.Q_LEVELS) |
                             {(1 - lv) / 2 for lv in COVERAGE_LEVELS} |
                             {(1 + lv) / 2 for lv in COVERAGE_LEVELS})
    rows_h1, rows_h2 = [], []
    for origin in origins:
        q_dir = int(q_dirs[origin])
        origin_date = str(q_dates[origin])
        origin_bar = int(np.searchsorted(ticker_data[target][2], origin_date))
        rv_query = rank_vol_t[min(origin_bar, len(rank_vol_t) - 1)]
        rf_query = rank_feat_t[min(origin_bar, len(rank_feat_t) - 1)]
        if not (np.isfinite(rv_query) and np.isfinite(rf_query)):
            continue

        actual_lr_1 = float(q_lp[origin + 1] - q_lp[origin])
        actual_lr_2 = float(q_lp[origin + 2] - q_lp[origin])

        pfm1, ptr1, pdir1, rv1, rf1 = build_causal_pool_with_2ranks(
            origin_date, ticker_data, T_FIXED, M, 1, MIN_BARS)
        pfm2, ptr2, pdir2, rv2, rf2 = build_causal_pool_with_2ranks(
            origin_date, ticker_data, T_FIXED, M, 2, MIN_BARS)
        mask1 = pdir1 == q_dir
        mask2 = pdir2 == q_dir
        if mask1.sum() < M + 2 or mask2.sum() < M + 2:
            continue

        w1 = np.exp(-lam_vol * (rv1[mask1] - rv_query) ** 2 - lam_feat * (rf1[mask1] - rf_query) ** 2)
        w2 = np.exp(-lam_vol * (rv2[mask2] - rv_query) ** 2 - lam_feat * (rf2[mask2] - rf_query) ** 2)
        if w1.sum() < 1e-9 or w2.sum() < 1e-9:
            continue

        band1 = weighted_quantile(ptr1[mask1], w1, q_levels_needed)
        band2 = weighted_quantile(ptr2[mask2], w2, q_levels_needed)

        pb1 = calib29.pinball(actual_lr_1, band1, calib29.Q_LEVELS)
        pb2 = calib29.pinball(actual_lr_2, band2, calib29.Q_LEVELS)
        rows_h1.append({"pinball": pb1, **{f"cov{int(lv*100)}": calib29g.coverage_at(actual_lr_1, band1, lv)
                                           for lv in COVERAGE_LEVELS}})
        rows_h2.append({"pinball": pb2, **{f"cov{int(lv*100)}": calib29g.coverage_at(actual_lr_2, band2, lv)
                                           for lv in COVERAGE_LEVELS}})

    if len(rows_h1) < 1:
        return None
    df1, df2 = pd.DataFrame(rows_h1), pd.DataFrame(rows_h2)
    cov_errs = []
    for lv in COVERAGE_LEVELS:
        cov_errs.append(abs(df1[f"cov{int(lv*100)}"].mean() - lv))
        cov_errs.append(abs(df2[f"cov{int(lv*100)}"].mean() - lv))
    return {"n": len(df1), "cov_err_avg": float(np.mean(cov_errs))}


def calibrate_feat_on_top(target, lam_vol, feat_name, rank_vol_all, rank_feat_all, base_data, t0):
    lh_t, ll_t, dates_t = base_data[target]
    q_lp, q_dates, q_dirs = build_zigzag(lh_t, ll_t, dates_t, T_FIXED, MIN_BARS)

    ticker_data = {tk: (lh, ll, dates, rank_vol_all[tk], rank_feat_all[tk])
                   for tk, (lh, ll, dates) in base_data.items()}

    min_hist = M + 3
    origins = calib29.prepare_origins(q_lp, min_hist)
    holdout_target_n = max(MIN_TEST_ORIGINS, 20)
    if len(origins) <= holdout_target_n:
        return {"target": target, "feature": feat_name, "skipped": True}
    holdout_cutoff_date = str(q_dates[origins[-holdout_target_n]])
    dev = [o for o in origins if str(q_dates[o]) < holdout_cutoff_date]
    holdout = [o for o in origins if str(q_dates[o]) >= holdout_cutoff_date]

    rank_vol_t = rank_vol_all[target]
    rank_feat_t = rank_feat_all[target]

    lam_grid = np.round(np.arange(FEATURE_LAMBDA_LO, FEATURE_LAMBDA_HI + 1e-9, FEATURE_LAMBDA_STEP), 3)
    rows_by_split = {frac: [] for frac in DEV_SPLIT_FRACS}
    for lam_feat in lam_grid:
        for frac in DEV_SPLIT_FRACS:
            split_i = int(len(dev) * frac)
            calib_o = dev[:split_i]
            if len(calib_o) < MIN_CALIB_ORIGINS:
                continue
            r = evaluate_origins_combined(target, lam_vol, float(lam_feat), calib_o, q_lp, q_dates, q_dirs,
                                          rank_vol_t, rank_feat_t, ticker_data)
            if r is not None:
                rows_by_split[frac].append({"t": float(lam_feat), "cov_err": r["cov_err_avg"]})

    candidates = {}
    for frac, rows in rows_by_split.items():
        res = calib29f._grid_argmin(rows)
        if res is not None:
            candidates[frac] = res
    if not candidates:
        return {"target": target, "feature": feat_name, "skipped": True}

    unique_lams = sorted(set(round(c["t_star"], 3) for c in candidates.values()))
    spread = round(max(unique_lams) - min(unique_lams), 3)

    holdout_results = {}
    for lam_feat in sorted(set(unique_lams) | {0.0}):
        r = evaluate_origins_combined(target, lam_vol, lam_feat, holdout, q_lp, q_dates, q_dirs,
                                      rank_vol_t, rank_feat_t, ticker_data)
        holdout_results[lam_feat] = r
    valid = {l: r for l, r in holdout_results.items() if r is not None}
    if not valid:
        return {"target": target, "feature": feat_name, "skipped": True}

    winner = min(valid, key=lambda l: valid[l]["cov_err_avg"])
    baseline = valid.get(0.0)   # λ_feat=0 == объём (или ничего, если λ_vol тоже 0) БЕЗ добавки
    rel = valid[winner]["cov_err_avg"] / baseline["cov_err_avg"] if baseline else None
    print(f"  [{target}/{feat_name}] λ_vol={lam_vol}(фикс)  λ_feat*={winner:.2f}  "
          f"spread={spread}  cov_err={valid[winner]['cov_err_avg']:.4f}  "
          f"baseline(feat=0)={baseline['cov_err_avg']:.4f}  "
          f"rel={rel:.4f}  ({'лучше' if rel < 1 else 'не лучше'})  [{time.time()-t0:.1f}s]")

    return {
        "target": target, "feature": feat_name, "skipped": False,
        "lambda_vol_fixed": lam_vol, "lambda_feat_star": winner, "spread": spread,
        "n_candidates": len(unique_lams), "holdout_cov_err": valid[winner]["cov_err_avg"],
        "baseline_cov_err": baseline["cov_err_avg"] if baseline else None,
        "rel_vs_volume_only": rel, "n_holdout": valid[winner]["n"],
    }


def main():
    t0 = time.time()
    print(f"Загрузка кросс-тикерного пула ({len(UNIVERSE)} тикеров)…")
    base_data, volume_data = {}, {}
    for tk in UNIVERSE:
        loaded = calib29g.load_ticker_candles_with_volume(tk, "1d")
        if loaded is None:
            continue
        lh, ll, dates, vol = loaded
        base_data[tk] = (lh, ll, dates)
        volume_data[tk] = vol
    rank_vol_all = {tk: calib29g.compute_density_rank_series(lh, ll, volume_data[tk],
                                                             calib29g.RANK_WINDOW, calib29g.N_BINS)
                    for tk, (lh, ll, dates) in base_data.items()}
    print(f"загружено {len(base_data)}/{len(UNIVERSE)} тикеров  [{time.time()-t0:.1f}s]\n")

    all_rows = []
    for feat_name, feat_fn in screen29h.FEATURES.items():
        rank_feat_all = {tk: feat_fn(lh, ll) for tk, (lh, ll, dates) in base_data.items()}
        print(f"=== Признак: {feat_name} ===")
        for target, lam_vol in LAMBDA_VOL_BY_TICKER.items():
            r = calibrate_feat_on_top(target, lam_vol, feat_name, rank_vol_all, rank_feat_all, base_data, t0)
            all_rows.append(r)
        print()

    rows = [r for r in all_rows if not r.get("skipped")]
    if rows:
        df = pd.DataFrame(rows)
        out_path = RESULTS / "29i_combined_summary.csv"
        df.to_csv(out_path, index=False)
        print("=== Итог: объём + вторая фича ===")
        print(df.to_string(index=False))
        print(f"\nСохранено: {out_path}")

    print(f"\nВсего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
