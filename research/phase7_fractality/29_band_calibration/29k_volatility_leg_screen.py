#!/usr/bin/env python3
"""
29k_volatility_leg_screen.py — быстрый скрининг трёх новых кандидатов веса
пула на 2 тикерах (GAZP, CHMF), грубой сеткой λ, ПРЕЖДЕ чем комбинировать
их в 29j (по образцу 29h для trend/velocity/acceleration).

Признаки (согласованы с пользователем 2026-07-08):
  volatility    — bar-нативный (как trend/velocity/acceleration): std
                  лог-доходностей mid-цены на trailing window_vol=20 баров
                  (причинно), затем перцентильный ранг (window=252) —
                  тот же путь кода, что и остальные bar-признаки (просто
                  ещё один rank_series в ticker_data), λ ищется ЧЕРЕЗ
                  calib29g.calibrate_lambda БЕЗ ИЗМЕНЕНИЙ.
  leg_amplitude — pivot-нативный, T-ЗАВИСИМЫЙ: |Δlog_price| последнего
                  ПОДТВЕРЖДЁННОГО плеча зигзага относительно последних
                  K=10 плеч (причинно). НЕ существует как единая по всей
                  истории величина — плечо определено только относительно
                  конкретного T, поэтому считается заново внутри
                  build_causal_pool_with_pivot_rank (там же, где строится
                  pivot_prices), а не предзагружается в ticker_data.
  leg_age       — pivot-нативный, T-ЗАВИСИМЫЙ: число баров, занятое
                  формированием последнего плеча, относительно последних
                  K=10 плеч (причинно). Тот же механизм, что leg_amplitude.

На стороне ЗАПРОСА (целевой тикер) pivot-нативные ранги считаются ОДИН
раз по q_lp/q_dates (T_query зафиксирован на весь прогон калибровки) —
origin уже индексирует ПИВОТ (не бар), поэтому rank_query = q_rank[origin]
напрямую, без bar_idx-маппинга, которым пользуются bar-нативные признаки.

θ=0, m=6, min_bars=5, T_query=T_pool=0.20 — как в 29g/29h. K=10 — компромисс
между «хватает контекста для ранга» и «не съедает лишние origin'ы на
старте истории» (min_hist=3 внутри causal_pivot_percentile_rank — реально
теряется ~3-4 origin'а, не K).

Использование:
    python 29k_volatility_leg_screen.py
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
from smap_band_ref import build_zigzag, weighted_quantile, UNIVERSE

spec29 = importlib.util.spec_from_file_location("calib29", HERE / "29_smap_band_calibrator.py")
calib29 = importlib.util.module_from_spec(spec29)
spec29.loader.exec_module(calib29)

spec29f = importlib.util.spec_from_file_location("calib29f", HERE / "29f_t_calibrator.py")
calib29f = importlib.util.module_from_spec(spec29f)
spec29f.loader.exec_module(calib29f)

spec29g = importlib.util.spec_from_file_location("calib29g", HERE / "29g_lambda_volume_calibrator.py")
calib29g = importlib.util.module_from_spec(spec29g)
spec29g.loader.exec_module(calib29g)

spec29h = importlib.util.spec_from_file_location("calib29h", HERE / "29h_trend_velocity_accel_screen.py")
calib29h = importlib.util.module_from_spec(spec29h)
spec29h.loader.exec_module(calib29h)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

M = calib29.M
SCREEN_TICKERS = ["GAZP", "CHMF"]
T_SCREEN = 0.20
WINDOW_VOL = 20
K_LEG = 10


# ── volatility — bar-нативный, reuse calib29g.calibrate_lambda напрямую ──

def compute_volatility_raw(log_highs: np.ndarray, log_lows: np.ndarray, window_vol: int = WINDOW_VOL) -> np.ndarray:
    mid = (log_highs + log_lows) / 2.0
    ret = np.diff(mid, prepend=np.nan)
    return pd.Series(ret).rolling(window_vol, min_periods=window_vol).std().to_numpy()


# ── leg_amplitude / leg_age — pivot-нативные, T-зависимые ──

def causal_pivot_percentile_rank(values: np.ndarray, k: int = K_LEG, min_hist: int = 3) -> np.ndarray:
    """Причинный перцентильный ранг values[i] относительно ПРЕДЫДУЩИХ до k
    значений (i само не входит). values[0] всегда NaN (плечо не определено
    до первого пивота)."""
    n = len(values)
    ranks = np.full(n, np.nan)
    for i in range(1, n):
        lo = max(1, i - k)
        hist = values[lo:i]
        finite = hist[np.isfinite(hist)]
        if len(finite) < min_hist or not np.isfinite(values[i]):
            continue
        ranks[i] = float((finite < values[i]).mean())
    return ranks


def leg_amplitude_raw(pivot_prices: np.ndarray) -> np.ndarray:
    n = len(pivot_prices)
    amp = np.full(n, np.nan)
    if n > 1:
        amp[1:] = np.abs(pivot_prices[1:] - pivot_prices[:-1])
    return amp


def leg_age_raw(confirm_dates: np.ndarray, dates_full: np.ndarray) -> np.ndarray:
    bar_idx = np.searchsorted(dates_full, confirm_dates)
    n = len(bar_idx)
    age = np.full(n, np.nan)
    if n > 1:
        age[1:] = (bar_idx[1:] - bar_idx[:-1]).astype(float)
    return age


def build_causal_pool_with_pivot_rank(cutoff_date, ticker_arrays, t_pool, embedding_dim, horizon,
                                      min_bars, feat_name, k=K_LEG):
    """ticker_arrays: {ticker: (lh, ll, dates)} — как smap_band_ref, без rank.
    Ранг считается ЗАНОВО из pivot_prices/confirm_dates этого вызова —
    T-зависим, не переиспользуется между разными T_pool/cutoff."""
    feats_l, tars_l, dirs_l, ranks_l = [], [], [], []
    for ticker, (lh, ll, dates) in ticker_arrays.items():
        mask = dates <= cutoff_date
        if mask.sum() < embedding_dim + horizon + 2:
            continue
        lh_c, ll_c, dt_c = lh[mask], ll[mask], dates[mask]
        pivot_prices, confirm_dates, pivot_dirs = build_zigzag(lh_c, ll_c, dt_c, t_pool, min_bars)
        if len(pivot_prices) < 3:
            continue
        raw = leg_amplitude_raw(pivot_prices) if feat_name == "leg_amplitude" else leg_age_raw(confirm_dates, dt_c)
        pivot_ranks = causal_pivot_percentile_rank(raw, k)
        feats, tars, dirs, ranks = calib29g.build_pool_vectors_with_rank(
            pivot_prices, pivot_dirs, pivot_ranks, embedding_dim, horizon)
        if len(tars):
            feats_l.append(feats); tars_l.append(tars); dirs_l.append(dirs); ranks_l.append(ranks)
    if not tars_l:
        return (np.zeros((0, embedding_dim)), np.zeros(0), np.zeros(0, dtype=np.int8), np.zeros(0))
    return (np.vstack(feats_l), np.concatenate(tars_l), np.concatenate(dirs_l), np.concatenate(ranks_l))


def evaluate_origins_pivot(target, t, lam, feat_name, origins, q_lp, q_dates, q_dirs, q_rank, ticker_arrays, k=K_LEG):
    q_levels_needed = sorted(set(calib29.Q_LEVELS) |
                             {(1 - lv) / 2 for lv in calib29g.COVERAGE_LEVELS} |
                             {(1 + lv) / 2 for lv in calib29g.COVERAGE_LEVELS})
    rows_h1, rows_h2 = [], []
    for origin in origins:
        q_dir = int(q_dirs[origin])
        origin_date = str(q_dates[origin])
        rank_query = q_rank[origin]
        if not np.isfinite(rank_query):
            continue

        actual_lr_1 = float(q_lp[origin + 1] - q_lp[origin])
        actual_lr_2 = float(q_lp[origin + 2] - q_lp[origin])

        _pfm1, ptr1, pdir1, prank1 = build_causal_pool_with_pivot_rank(
            origin_date, ticker_arrays, t, M, 1, calib29g.MIN_BARS, feat_name, k)
        _pfm2, ptr2, pdir2, prank2 = build_causal_pool_with_pivot_rank(
            origin_date, ticker_arrays, t, M, 2, calib29g.MIN_BARS, feat_name, k)
        mask1 = pdir1 == q_dir
        mask2 = pdir2 == q_dir
        if mask1.sum() < M + 2 or mask2.sum() < M + 2:
            continue

        w1 = np.exp(-lam * (prank1[mask1] - rank_query) ** 2)
        w2 = np.exp(-lam * (prank2[mask2] - rank_query) ** 2)
        if w1.sum() < 1e-9 or w2.sum() < 1e-9:
            continue

        band1 = weighted_quantile(ptr1[mask1], w1, q_levels_needed)
        band2 = weighted_quantile(ptr2[mask2], w2, q_levels_needed)
        pb1 = calib29.pinball(actual_lr_1, band1, calib29.Q_LEVELS)
        pb2 = calib29.pinball(actual_lr_2, band2, calib29.Q_LEVELS)
        rows_h1.append({"pinball": pb1, "width": band1[0.9] - band1[0.1],
                        **{f"cov{int(lv*100)}": calib29g.coverage_at(actual_lr_1, band1, lv)
                           for lv in calib29g.COVERAGE_LEVELS}})
        rows_h2.append({"pinball": pb2, "width": band2[0.9] - band2[0.1],
                        **{f"cov{int(lv*100)}": calib29g.coverage_at(actual_lr_2, band2, lv)
                           for lv in calib29g.COVERAGE_LEVELS}})

    if len(rows_h1) < 1:
        return None
    df1, df2 = pd.DataFrame(rows_h1), pd.DataFrame(rows_h2)
    cov_errs = []
    for lv in calib29g.COVERAGE_LEVELS:
        cov_errs.append(abs(df1[f"cov{int(lv*100)}"].mean() - lv))
        cov_errs.append(abs(df2[f"cov{int(lv*100)}"].mean() - lv))
    return {
        "n": len(df1),
        "pinball_norm_avg": (df1.pinball.mean() + df2.pinball.mean()) / 2 / t,
        "cov_err_avg": float(np.mean(cov_errs)),
        "width_avg": (df1.width.mean() + df2.width.mean()) / 2,
    }


def calibrate_lambda_pivot(target, t, feat_name, ticker_arrays, t0,
                           lambda_lo, lambda_hi, lambda_step, k=K_LEG):
    lh_t, ll_t, dates_t = ticker_arrays[target]
    q_lp, q_dates, q_dirs = build_zigzag(lh_t, ll_t, dates_t, t, calib29g.MIN_BARS)
    print(f"{target} T={t*100:.0f}%: {len(q_lp)} пивотов")

    q_raw = leg_amplitude_raw(q_lp) if feat_name == "leg_amplitude" else leg_age_raw(q_dates, dates_t)
    q_rank = causal_pivot_percentile_rank(q_raw, k)

    min_hist = M + 3
    origins = calib29.prepare_origins(q_lp, min_hist)
    holdout_target_n = max(calib29g.MIN_TEST_ORIGINS, 20)
    if len(origins) <= holdout_target_n:
        print(f"  ПРОПУСК: недостаточно origin'ов ({len(origins)})")
        return {"target": target, "feature": feat_name, "skipped": True}
    holdout_cutoff_date = str(q_dates[origins[-holdout_target_n]])
    dev = [o for o in origins if str(q_dates[o]) < holdout_cutoff_date]
    holdout = [o for o in origins if str(q_dates[o]) >= holdout_cutoff_date]
    print(f"  origin'ов: {len(origins)}  (dev={len(dev)}, holdout={len(holdout)})")

    lam_grid = np.round(np.arange(lambda_lo, lambda_hi + 1e-9, lambda_step), 3)
    rows_by_split = {frac: [] for frac in calib29g.DEV_SPLIT_FRACS}
    for lam in lam_grid:
        for frac in calib29g.DEV_SPLIT_FRACS:
            split_i = int(len(dev) * frac)
            calib_o = dev[:split_i]
            if len(calib_o) < calib29g.MIN_CALIB_ORIGINS:
                continue
            r = evaluate_origins_pivot(target, t, float(lam), feat_name, calib_o,
                                       q_lp, q_dates, q_dirs, q_rank, ticker_arrays, k)
            if r is not None:
                rows_by_split[frac].append({"t": float(lam), "score": r["pinball_norm_avg"]})

    candidates = {}
    for frac, rows in rows_by_split.items():
        res = calib29f._grid_argmin(rows, key="score")
        if res is not None:
            candidates[frac] = res
    if not candidates:
        print(f"  ПРОПУСК: ни один сплит не дал кандидата  [{time.time()-t0:.1f}s]")
        return {"target": target, "feature": feat_name, "skipped": True}

    print("  Кандидаты λ по сплитам:")
    for frac, c in sorted(candidates.items()):
        print(f"    split={frac:.2f}: λ*={c['t_star']:.2f}  calib_score={c['calib_score']:.4f}")

    unique_lams = sorted(set(round(c["t_star"], 3) for c in candidates.values()))
    spread = round(max(unique_lams) - min(unique_lams), 3)
    holdout_results = {}
    for lam in sorted(set(unique_lams) | {0.0}):
        r = evaluate_origins_pivot(target, t, lam, feat_name, holdout, q_lp, q_dates, q_dirs, q_rank, ticker_arrays, k)
        holdout_results[lam] = r

    valid = {l: r for l, r in holdout_results.items() if r is not None}
    if not valid or 0.0 not in valid:
        print(f"  ПРОПУСК: holdout недоступен  [{time.time()-t0:.1f}s]")
        return {"target": target, "feature": feat_name, "skipped": True}

    winner = min(valid, key=lambda l: valid[l]["pinball_norm_avg"])
    baseline = valid[0.0]
    rel = valid[winner]["pinball_norm_avg"] / baseline["pinball_norm_avg"]
    w = valid[winner]
    print(f"  Holdout: λ=0 baseline pinball_norm={baseline['pinball_norm_avg']:.4f}  "
          f"width={baseline['width_avg']:.4f}  (n={baseline['n']})")
    print(f"  Победитель λ={winner:.2f}  pinball_norm={w['pinball_norm_avg']:.4f}  width={w['width_avg']:.4f}  "
          f"относительно baseline={rel:.4f}  ({'лучше' if rel < 1 else 'не лучше'})")
    print(f"  [{time.time()-t0:.1f}s]\n")

    return {
        "target": target, "feature": feat_name, "skipped": False, "lambda_star": winner,
        "spread": spread, "pinball_rel": rel,
        "pinball_norm": w["pinball_norm_avg"], "baseline_pinball_norm": baseline["pinball_norm_avg"],
        "width_avg": w["width_avg"], "baseline_width": baseline["width_avg"],
        "cov_err_avg": w["cov_err_avg"], "baseline_cov_err": baseline["cov_err_avg"],
        "n_holdout": w["n"],
    }


def main():
    t0 = time.time()
    print(f"Загрузка кросс-тикерного пула ({len(UNIVERSE)} тикеров)…")
    ticker_arrays = {}
    ticker_data_vol = {}
    for tk in UNIVERSE:
        loaded = calib29g.load_ticker_candles_with_volume(tk, "1d")
        if loaded is None:
            continue
        lh, ll, dates, _vol = loaded
        ticker_arrays[tk] = (lh, ll, dates)
        ticker_data_vol[tk] = (lh, ll, dates, calib29h.to_percentile_rank(compute_volatility_raw(lh, ll)))
    print(f"загружено {len(ticker_arrays)}/{len(UNIVERSE)} тикеров  [{time.time()-t0:.1f}s]\n")

    # скрининг — грубая сетка, 2 сплита (не 4), 2 тикера (как 29h)
    calib29g.DEV_SPLIT_FRACS = [0.65, 0.80]
    SCREEN_LO, SCREEN_HI, SCREEN_STEP = 0.0, 16.0, 2.0

    all_rows = []

    print("=== Признак: volatility (bar-нативный, reuse calib29g) ===")
    for target in SCREEN_TICKERS:
        print(f"--- {target} ---")
        r = calib29g.calibrate_lambda(target, T_SCREEN, ticker_data_vol, t0, SCREEN_LO, SCREEN_HI, SCREEN_STEP)
        if not r.get("skipped"):
            all_rows.append({"feature": "volatility", "target": target, "lambda_star": r["lambda_star"],
                             "spread": r["spread"], "pinball_rel": r["rel_vs_baseline"],
                             "pinball_norm": r["holdout_pinball_norm"], "baseline_pinball_norm": r["baseline_pinball_norm"],
                             "width_avg": r["holdout_width"], "baseline_width": r["baseline_width"],
                             "cov_err_avg": r["holdout_cov_err"], "baseline_cov_err": r["baseline_cov_err"],
                             "n_holdout": r["n_holdout"]})

    for feat_name in ["leg_amplitude", "leg_age"]:
        print(f"=== Признак: {feat_name} (pivot-нативный, T-зависимый) ===")
        for target in SCREEN_TICKERS:
            print(f"--- {target} ---")
            r = calibrate_lambda_pivot(target, T_SCREEN, feat_name, ticker_arrays, t0, SCREEN_LO, SCREEN_HI, SCREEN_STEP)
            if not r.get("skipped"):
                all_rows.append({"feature": feat_name, **{k: v for k, v in r.items() if k != "feature"}})

    if all_rows:
        df = pd.DataFrame(all_rows)
        cols = ["feature", "target", "lambda_star", "spread", "pinball_rel", "pinball_norm",
               "baseline_pinball_norm", "width_avg", "baseline_width", "cov_err_avg", "baseline_cov_err", "n_holdout"]
        df = df[cols]
        out_path = RESULTS / "29k_screen_summary.csv"
        df.to_csv(out_path, index=False)
        print("\n=== Итог скрининга ===")
        print(df.to_string(index=False))
        print(f"\nСохранено: {out_path}")

    print(f"\nВсего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
