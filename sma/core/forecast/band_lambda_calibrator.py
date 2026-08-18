"""
band_lambda_calibrator — λ calibration (multi-pass coordinate descent) for
band_lambda's live forecaster. Pure computation only; DB orchestration
(pool resolution, incremental forecast_settings writes, ProcessPoolExecutor
fan-out across T) lives in sma/api/task_manager.py._run_calibration —
matching the convention set in band_lambda.py (I/O in task_manager,
algorithm here).

Ported from research/reference/band_lambda_calibrator_ref.py (session
2026-07-08). See docs/plans/band_forecast_migration_plan.md sections 1-2 for
the full methodology writeup and the approved SBER T=20% reference numbers
used to verify this port.

⚠️ objective is ALWAYS pinball_norm_avg (pinball/T, proper scoring rule) —
NOT cov_err_avg, which produced pure band-widening with no real accuracy
gain in an earlier iteration (see memory project-phase7-band-calibrator-
objective-fix). cov_err_avg / width_avg remain in the output purely as
diagnostics, never as the selection criterion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import band_lambda as bl

MIN_CALIB_ORIGINS = 25
MIN_TEST_ORIGINS = 15
DEV_SPLIT_FRACS = [0.55, 0.65, 0.75, 0.85]
SMOOTH_WINDOW = 3

DEFAULT_LAMBDA_LO, DEFAULT_LAMBDA_HI, DEFAULT_LAMBDA_STEP = 0.0, 16.0, 2.0
DEFAULT_MAX_PASSES = 3
DEFAULT_CONVERGENCE_TOL = 0.5   # in λ units — smaller than a typical lambda_step


# ── ProcessPoolExecutor entry points (must be top-level/picklable) ───────────

def init_worker_env() -> None:
    """
    ProcessPoolExecutor initializer — BLAS oversubscription guard (see memory
    blas-oversubscription-multiprocessing): each worker process must cap its
    own thread pool BEFORE numpy/pandas touch BLAS, or N worker processes x
    N BLAS threads each thrash the CPU instead of speeding anything up.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def calibrate_one_target(target: str, t_query: float, min_bars: int, m: int, ticker_data: dict) -> dict:
    """
    Picklable top-level wrapper for one (target, T) calibration job — the
    unit of work sma/api/task_manager.py._run_calibration fans out across a
    ProcessPoolExecutor (one process per T being calibrated in the same
    request, see docs/plans/band_forecast_migration_plan.md section 5.2).
    """
    return calibrate_combined_multipass(target, t_query, ticker_data, min_bars=min_bars, m=m)


# ── scoring ───────────────────────────────────────────────────────────────────

def pinball(actual: float, band: dict, q_levels: tuple) -> float:
    total = 0.0
    for q in q_levels:
        diff = actual - band[q]
        total += max(q * diff, (q - 1) * diff)
    return total / len(q_levels)


def coverage_at(actual: float, band: dict, level: float) -> int:
    q_lo, q_hi = (1 - level) / 2, (1 + level) / 2
    return int(band[q_lo] <= actual <= band[q_hi])


def prepare_origins(q_lp: np.ndarray, min_hist: int) -> list[int]:
    return list(range(min_hist, len(q_lp) - 2))


def evaluate_origins_mixed(
    target: str, t: float, lambdas: dict, origins: list[int],
    q_lp: np.ndarray, q_dates: np.ndarray, q_dirs: np.ndarray,
    rank_series_target: dict, q_rank_pivot: dict, ticker_data: dict,
    min_bars: int, m: int,
) -> dict | None:
    q_levels_needed = sorted(
        set(bl.Q_LEVELS)
        | {(1 - lv) / 2 for lv in bl.COVERAGE_LEVELS}
        | {(1 + lv) / 2 for lv in bl.COVERAGE_LEVELS}
    )
    rows_h1, rows_h2 = [], []
    for origin in origins:
        q_dir = int(q_dirs[origin])
        origin_date = str(q_dates[origin])
        origin_bar = int(np.searchsorted(ticker_data[target][2], origin_date))
        origin_bar = min(origin_bar, len(rank_series_target[bl.BAR_FEATURES[0]]) - 1)
        rank_query = {f: rank_series_target[f][origin_bar] for f in bl.BAR_FEATURES}
        for pf in bl.PIVOT_FEATURES:
            rank_query[pf] = q_rank_pivot[pf][origin]
        if any(not np.isfinite(v) for v in rank_query.values()):
            continue

        actual_lr_1 = float(q_lp[origin + 1] - q_lp[origin])
        actual_lr_2 = float(q_lp[origin + 2] - q_lp[origin])

        _f1, ptr1, pdir1, pranks1 = bl.build_causal_pool_with_mixed_ranks(
            origin_date, ticker_data, t, m, 1, min_bars)
        _f2, ptr2, pdir2, pranks2 = bl.build_causal_pool_with_mixed_ranks(
            origin_date, ticker_data, t, m, 2, min_bars)
        mask1 = pdir1 == q_dir
        mask2 = pdir2 == q_dir
        if mask1.sum() < m + 2 or mask2.sum() < m + 2:
            continue

        sq1 = np.zeros(int(mask1.sum()))
        sq2 = np.zeros(int(mask2.sum()))
        for f in bl.FEATURE_ORDER:
            sq1 += lambdas[f] * (pranks1[f][mask1] - rank_query[f]) ** 2
            sq2 += lambdas[f] * (pranks2[f][mask2] - rank_query[f]) ** 2
        w1, w2 = np.exp(-sq1), np.exp(-sq2)
        if w1.sum() < 1e-9 or w2.sum() < 1e-9:
            continue

        band1 = bl.weighted_quantile(ptr1[mask1], w1, q_levels_needed)
        band2 = bl.weighted_quantile(ptr2[mask2], w2, q_levels_needed)

        pb1 = pinball(actual_lr_1, band1, bl.Q_LEVELS)
        pb2 = pinball(actual_lr_2, band2, bl.Q_LEVELS)
        rows_h1.append({"pinball": pb1, "width": band1[0.9] - band1[0.1],
                        **{f"cov{int(lv*100)}": coverage_at(actual_lr_1, band1, lv)
                           for lv in bl.COVERAGE_LEVELS}})
        rows_h2.append({"pinball": pb2, "width": band2[0.9] - band2[0.1],
                        **{f"cov{int(lv*100)}": coverage_at(actual_lr_2, band2, lv)
                           for lv in bl.COVERAGE_LEVELS}})

    if len(rows_h1) < 1:
        return None
    df1, df2 = pd.DataFrame(rows_h1), pd.DataFrame(rows_h2)
    cov_errs = []
    for lv in bl.COVERAGE_LEVELS:
        cov_errs.append(abs(df1[f"cov{int(lv*100)}"].mean() - lv))
        cov_errs.append(abs(df2[f"cov{int(lv*100)}"].mean() - lv))
    return {
        "n": len(df1),
        "pinball_norm_avg": (df1.pinball.mean() + df2.pinball.mean()) / 2 / t,
        "cov_err_avg": float(np.mean(cov_errs)),
        "width_avg": (df1.width.mean() + df2.width.mean()) / 2,
    }


def _grid_argmin(rows: list, key: str = "score") -> dict | None:
    if len(rows) < SMOOTH_WINDOW:
        return None
    gdf = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    gdf["smooth"] = gdf[key].rolling(SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW).mean()
    valid = gdf.dropna(subset=["smooth"])
    if valid.empty:
        return None
    best = valid.loc[valid["smooth"].idxmin()]
    return {"t_star": float(best["t"]), "calib_score": float(best["smooth"])}


def calibrate_feature(
    target: str, t: float, feat_name: str, lambdas_fixed: dict, ticker_data: dict,
    lambda_lo: float, lambda_hi: float, lambda_step: float,
    q_lp: np.ndarray, q_dates: np.ndarray, q_dirs: np.ndarray,
    dev: list[int], holdout: list[int],
    rank_series_t: dict, q_rank_pivot: dict, min_bars: int, m: int,
) -> dict:
    lam_grid = np.round(np.arange(lambda_lo, lambda_hi + 1e-9, lambda_step), 3)
    rows_by_split: dict[float, list] = {frac: [] for frac in DEV_SPLIT_FRACS}
    for lam in lam_grid:
        trial = {**lambdas_fixed, feat_name: float(lam)}
        for frac in DEV_SPLIT_FRACS:
            split_i = int(len(dev) * frac)
            calib_o = dev[:split_i]
            if len(calib_o) < MIN_CALIB_ORIGINS:
                continue
            r = evaluate_origins_mixed(target, t, trial, calib_o, q_lp, q_dates, q_dirs,
                                       rank_series_t, q_rank_pivot, ticker_data, min_bars, m)
            if r is not None:
                rows_by_split[frac].append({"t": float(lam), "score": r["pinball_norm_avg"]})

    candidates = {}
    for frac, rows in rows_by_split.items():
        res = _grid_argmin(rows, key="score")
        if res is not None:
            candidates[frac] = res
    if not candidates:
        return {"feature": feat_name, "skipped": True}

    unique_lams = sorted(set(round(c["t_star"], 3) for c in candidates.values()))

    holdout_results = {}
    for lam in sorted(set(unique_lams) | {0.0}):
        trial = {**lambdas_fixed, feat_name: lam}
        r = evaluate_origins_mixed(target, t, trial, holdout, q_lp, q_dates, q_dirs,
                                   rank_series_t, q_rank_pivot, ticker_data, min_bars, m)
        holdout_results[lam] = r

    valid = {l: r for l, r in holdout_results.items() if r is not None}
    if not valid or 0.0 not in valid:
        return {"feature": feat_name, "skipped": True}

    winner = min(valid, key=lambda l: valid[l]["pinball_norm_avg"])
    baseline = valid[0.0]
    rel = valid[winner]["pinball_norm_avg"] / baseline["pinball_norm_avg"]
    w = valid[winner]
    return {
        "feature": feat_name, "skipped": False, "lambda_star": winner,
        "pinball_rel": rel, "pinball_norm": w["pinball_norm_avg"],
        "baseline_pinball_norm": baseline["pinball_norm_avg"],
        "width_avg": w["width_avg"], "baseline_width": baseline["width_avg"],
        "cov_err_avg": w["cov_err_avg"], "baseline_cov_err": baseline["cov_err_avg"],
        "n_holdout": w["n"],
    }


def calibrate_combined_multipass(
    target: str, t: float, ticker_data: dict,
    min_bars: int = bl.DEFAULT_MIN_BARS, m: int = bl.DEFAULT_M,
    lambda_lo: float = DEFAULT_LAMBDA_LO, lambda_hi: float = DEFAULT_LAMBDA_HI,
    lambda_step: float = DEFAULT_LAMBDA_STEP,
    max_passes: int = DEFAULT_MAX_PASSES, tol: float = DEFAULT_CONVERGENCE_TOL,
    progress_cb=None,
) -> dict:
    """
    Multi-pass coordinate descent over bl.FEATURE_ORDER — repeats a full
    pass until max|Δλ| between passes <= tol or max_passes is exhausted.
    progress_cb(done, total), if given, is called once per (pass, feature)
    — 6 features x max_passes total steps (upper bound; passes may stop
    early on convergence).

    Returns {"target", "skipped"} or a full result dict with "lambdas",
    "n_passes", "all_passes", "final_*" (pinball/width/cov_err, combo vs.
    λ=0 baseline, both evaluated on the untouched holdout — the number
    actually worth trusting).
    """
    lh_t, ll_t, dates_t, rank_dict_t = ticker_data[target]
    q_lp, _q_extreme, q_dates, q_dirs = bl.build_zigzag(lh_t, ll_t, dates_t, t, min_bars)
    if len(q_lp) < m + 5:
        return {"target": target, "skipped": True, "reason": f"too few pivots ({len(q_lp)})"}

    q_rank_pivot = {"leg_age": bl.causal_pivot_percentile_rank(bl.leg_age_raw(q_dates, dates_t), bl.K_LEG)}

    min_hist = m + 3
    origins = prepare_origins(q_lp, min_hist)
    holdout_target_n = max(MIN_TEST_ORIGINS, 20)
    if len(origins) <= holdout_target_n:
        return {"target": target, "skipped": True, "reason": f"too few origins ({len(origins)}) for holdout split"}
    holdout_cutoff_date = str(q_dates[origins[-holdout_target_n]])
    dev = [o for o in origins if str(q_dates[o]) < holdout_cutoff_date]
    holdout = [o for o in origins if str(q_dates[o]) >= holdout_cutoff_date]

    lambdas = {f: 0.0 for f in bl.FEATURE_ORDER}
    all_passes = []
    total_steps = max_passes * len(bl.FEATURE_ORDER)
    done_steps = 0
    for pass_num in range(1, max_passes + 1):
        prev_lambdas = dict(lambdas)
        per_feature = []
        for feat_name in bl.FEATURE_ORDER:
            res = calibrate_feature(target, t, feat_name, lambdas, ticker_data,
                                    lambda_lo, lambda_hi, lambda_step, q_lp, q_dates, q_dirs,
                                    dev, holdout, rank_dict_t, q_rank_pivot, min_bars, m)
            per_feature.append(res)
            if not res.get("skipped"):
                lambdas[feat_name] = res["lambda_star"]
            done_steps += 1
            if progress_cb:
                progress_cb(done_steps, total_steps)
        all_passes.append({"pass": pass_num, "lambdas": dict(lambdas), "per_feature": per_feature})
        max_delta = max(abs(lambdas[f] - prev_lambdas[f]) for f in bl.FEATURE_ORDER)
        if max_delta <= tol:
            break

    baseline_zero = {f: 0.0 for f in bl.FEATURE_ORDER}
    final_combo = evaluate_origins_mixed(target, t, lambdas, holdout, q_lp, q_dates, q_dirs,
                                         rank_dict_t, q_rank_pivot, ticker_data, min_bars, m)
    final_baseline = evaluate_origins_mixed(target, t, baseline_zero, holdout, q_lp, q_dates, q_dirs,
                                            rank_dict_t, q_rank_pivot, ticker_data, min_bars, m)
    rel_final = (final_combo["pinball_norm_avg"] / final_baseline["pinball_norm_avg"]
                if final_combo and final_baseline else None)

    return {
        "target": target, "skipped": False, "lambdas": lambdas, "n_passes": len(all_passes),
        "all_passes": all_passes,
        "final_pinball_norm": final_combo["pinball_norm_avg"] if final_combo else None,
        "final_baseline_pinball_norm": final_baseline["pinball_norm_avg"] if final_baseline else None,
        "final_rel": rel_final,
        "final_width": final_combo["width_avg"] if final_combo else None,
        "final_baseline_width": final_baseline["width_avg"] if final_baseline else None,
        "final_cov_err": final_combo["cov_err_avg"] if final_combo else None,
        "final_baseline_cov_err": final_baseline["cov_err_avg"] if final_baseline else None,
        "n_holdout": final_combo["n"] if final_combo else None,
    }
