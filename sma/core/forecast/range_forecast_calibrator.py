"""
range_forecast_calibrator — многофолдовый координатный спуск (θ + 5 λ-
признаков band_lambda + per-level read-квантиль), портирован из
research/phase8_single_step_ohlc/02_range_forecast/calibrate_theta.py
::calibrate_combined_multipass (сессия 2026-09-06). См.
docs/plans/app16_range_forecast_migration_plan.md.

Отличия от band_lambda's старой λ-калибровки (удалена из прода 2026-09-12
вместе с band_lambda_calibrator.py — сравнение исторического характера,
см. память project_phase7_calibration_removed_final):
- НЕЗАВИСИМЫЕ последовательные dev-фолды (walk-forward по разным периодам),
  не растущий префикс band_lambda (frac=0.55..0.85) — range_forecast работает
  по каждому бару, не по редким zigzag-пивотам.
- У каждого признака СВОЯ сетка (GRIDS): θ — L2-расстояние по форме (0..40
  шаг 2), λ_f — ранговая band_lambda-шкала (0..16 шаг 2) — разные единицы
  измерения в одной экспоненте S-map.
- Дополнительно (сессия 2026-09-06): пул-кандидат для holdout
  (_pooled_argmin), диагностика сходимости (max_delta/converged), уточняющая
  мелкая сетка (_refine_grid), калибровка read-квантиля Q_LOW/Q_HIGH
  ОТДЕЛЬНО на каждом уровне покрытия (objective остаётся pinball на
  ФИКСИРОВАННОМ номинале — НЕ coverage error, см. память
  project_phase7_band_calibrator_objective_fix).

ProcessPoolExecutor entry point (calibrate_range_forecast_task) — picklable
top-level wrapper, тот же паттерн band_lambda's старая λ-калибровка
использовала; DB-запись (upsert range_forecast_settings) — в
sma/api/task_manager.py, не здесь (I/O в task_manager, алгоритм здесь).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .band_lambda import compute_bar_rank_dict, weighted_quantile
from .range_forecast import (
    RANK_FEATURES, MAX_CANDLES_DEFAULT, build_delay_matrix, dratio_from_close,
    rolling_cummin_cummax, truncate_candles,
)

FEATURE_ORDER = ["theta"] + RANK_FEATURES

DEFAULT_THETA_GRID = tuple(np.round(np.arange(0.0, 42.0, 2.0), 3))
DEFAULT_LAMBDA_GRID = tuple(np.round(np.arange(0.0, 18.0, 2.0), 3))
GRIDS = {"theta": DEFAULT_THETA_GRID, **{f: DEFAULT_LAMBDA_GRID for f in RANK_FEATURES}}

SMOOTH_WINDOW = 3
N_FOLDS = 3
FOLD_SIZE = 200
N_HOLDOUT = 300
MIN_FOLD_ORIGINS = 25
DEFAULT_MAX_PASSES = 5
DEFAULT_TOL = 0.5
REFINE_STEP_FRACTION = 0.125
DEFAULT_REFINE = True

Q_READ_SPAN = 0.04
Q_READ_STEP = 0.01


def init_worker_env() -> None:
    """ProcessPoolExecutor initializer — BLAS oversubscription guard (see
    memory blas_oversubscription_multiprocessing)."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def pinball_single(actual: float, pred: float, tau: float) -> float:
    diff = actual - pred
    return max(tau * diff, (tau - 1) * diff)


def unconditional_quantile_range(close, roll_low, roll_high, origin, theiler, q_low, q_high):
    """Честный naive: невзвешенные эмпирические квантили H-барного размаха
    по всей причинно допустимой истории (то же окно Тайлера, тот же
    weighted_quantile) — тестирует, даёт ли k-NN кондиционирование что-то
    сверх типичного для тикера распределения диапазона."""
    pool_end = origin - theiler
    if pool_end < 20:
        return None
    idx = np.arange(pool_end)
    valid = np.isfinite(roll_low[idx])
    idx = idx[valid]
    if len(idx) < 20:
        return None
    nb_close = close[idx]
    off_low = (roll_low[idx] - nb_close) / np.maximum(nb_close, 1e-10)
    off_high = (roll_high[idx] - nb_close) / np.maximum(nb_close, 1e-10)
    w = np.ones(len(idx)) / len(idx)
    b_low = weighted_quantile(off_low, w, (q_low,))[q_low]
    b_high = weighted_quantile(off_high, w, (q_high,))[q_high]
    anchor = close[origin]
    return anchor * (1 + b_low), anchor * (1 + b_high)


def _smoothed_argmin(rows: list) -> dict | None:
    if len(rows) < SMOOTH_WINDOW:
        return None
    gdf = pd.DataFrame(rows).sort_values("value").reset_index(drop=True)
    gdf["smooth"] = gdf["score"].rolling(SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW).mean()
    valid = gdf.dropna(subset=["smooth"])
    if valid.empty:
        return None
    best = valid.loc[valid["smooth"].idxmin()]
    return {"value_star": float(best["value"]), "calib_score": float(best["smooth"])}


def _origin_context(t, dratio, close, high, low, p, theiler, H, q_low, q_high, roll_low, roll_high, rank_dict):
    pool_end = t - theiler
    if pool_end < p + 5:
        return None
    X_all, _y_all = build_delay_matrix(dratio[:pool_end], p)
    if len(X_all) < 5:
        return None
    q = dratio[t - p:t]
    d = np.linalg.norm(X_all - q, axis=1)
    nb_bar = np.arange(len(d)) + p
    valid = (nb_bar >= 0) & (nb_bar < len(roll_low)) & np.isfinite(roll_low[np.clip(nb_bar, 0, len(roll_low) - 1)])

    rank_query = {}
    for f in RANK_FEATURES:
        rv = rank_dict[f]
        if t >= len(rv) or not np.isfinite(rv[t]):
            return None
        rank_query[f] = float(rv[t])
    for f in RANK_FEATURES:
        rv = rank_dict[f]
        clipped = np.clip(nb_bar, 0, len(rv) - 1)
        valid &= (nb_bar < len(rv)) & np.isfinite(rv[clipped])

    if valid.sum() < 5:
        return None
    nb_bar_v = nb_bar[valid]
    d_v = d[valid]
    d_mean = max(float(d_v.mean()), 1e-10)
    nb_close = close[nb_bar_v]
    off_low = (roll_low[nb_bar_v] - nb_close) / np.maximum(nb_close, 1e-10)
    off_high = (roll_high[nb_bar_v] - nb_close) / np.maximum(nb_close, 1e-10)
    order_low, order_high = np.argsort(off_low), np.argsort(off_high)

    sqdiff = {f: (rank_dict[f][nb_bar_v] - rank_query[f]) ** 2 for f in RANK_FEATURES}

    actual_low = float(low[t + 1:t + 1 + H].min())
    actual_high = float(high[t + 1:t + 1 + H].max())
    anchor = float(close[t])
    pbu = None
    unc = unconditional_quantile_range(close, roll_low, roll_high, t, theiler, q_low, q_high)
    if unc is not None:
        u_low, u_high = unc
        pbu = (pinball_single(actual_low, u_low, q_low) + pinball_single(actual_high, u_high, q_high)) / 2 / anchor

    return {
        "origin": t, "d": d_v, "d_mean": d_mean, "sqdiff": sqdiff,
        "off_low_sorted": off_low[order_low], "order_low": order_low,
        "off_high_sorted": off_high[order_high], "order_high": order_high,
        "actual_low": actual_low, "actual_high": actual_high, "anchor": anchor, "pbu": pbu,
    }


def _relevel_pbu(contexts, close, roll_low, roll_high, theiler, q_low, q_high):
    out = []
    for ctx in contexts:
        unc = unconditional_quantile_range(close, roll_low, roll_high, ctx["origin"], theiler, q_low, q_high)
        pbu = None
        if unc is not None:
            u_low, u_high = unc
            pbu = (pinball_single(ctx["actual_low"], u_low, q_low)
                   + pinball_single(ctx["actual_high"], u_high, q_high)) / 2 / ctx["anchor"]
        out.append({**ctx, "pbu": pbu})
    return out


def _quantile_presorted(v_sorted, w, order, q):
    w_sorted = w[order]
    total = w_sorted.sum()
    if total < 1e-14:
        return float(np.median(v_sorted))
    cw = (np.cumsum(w_sorted) - 0.5 * w_sorted) / total
    return float(np.interp(q, cw, v_sorted))


def _weight(ctx, params: dict) -> np.ndarray:
    sq = params.get("theta", 0.0) * ctx["d"] / ctx["d_mean"]
    for f in RANK_FEATURES:
        lam = params.get(f, 0.0)
        if lam:
            sq = sq + lam * ctx["sqdiff"][f]
    return np.exp(-sq)


def _eval_params(params: dict, contexts: list, q_low, q_high, q_low_read=None, q_high_read=None):
    if q_low_read is None:
        q_low_read = q_low
    if q_high_read is None:
        q_high_read = q_high
    pb, pbu, cov = [], [], []
    for ctx in contexts:
        w = _weight(ctx, params)
        if w.sum() < 1e-14:
            continue
        band_low = _quantile_presorted(ctx["off_low_sorted"], w, ctx["order_low"], q_low_read)
        band_high = _quantile_presorted(ctx["off_high_sorted"], w, ctx["order_high"], q_high_read)
        pred_low = ctx["anchor"] * (1 + band_low)
        pred_high = ctx["anchor"] * (1 + band_high)
        pl = pinball_single(ctx["actual_low"], pred_low, q_low)
        ph = pinball_single(ctx["actual_high"], pred_high, q_high)
        pb.append((pl + ph) / 2 / ctx["anchor"])
        cov.append(int(pred_low <= ctx["actual_low"] and ctx["actual_high"] <= pred_high))
        if ctx["pbu"] is not None:
            pbu.append(ctx["pbu"])
    if len(pb) < MIN_FOLD_ORIGINS or not pbu:
        return None
    return {"ratio": float(np.mean(pb) / np.mean(pbu)), "coverage": float(np.mean(cov)), "n": len(pb)}


def _pooled_argmin(per_fold: list) -> dict | None:
    agg: dict[float, list] = {}
    for f in per_fold:
        for r in f["grid"]:
            agg.setdefault(r["value"], []).append(r["score"])
    rows = [{"value": v, "score": float(np.mean(s))} for v, s in agg.items()]
    return _smoothed_argmin(rows)


def _calibrate_one_value(eval_fn, grid, dev_fold_ctxs, holdout_ctx, baseline_value=0.0):
    per_fold = []
    for fi, ctx in enumerate(dev_fold_ctxs):
        rows = []
        for val in grid:
            r = eval_fn(val, ctx)
            if r is not None:
                rows.append({"value": float(val), "score": r["ratio"], "coverage": r["coverage"], "n": r["n"]})
        best = _smoothed_argmin([{"value": r["value"], "score": r["score"]} for r in rows])
        per_fold.append({"fold": fi, "grid": rows, "best": best})

    pooled_best = _pooled_argmin(per_fold)
    candidates = sorted(
        {round(f["best"]["value_star"], 3) for f in per_fold if f["best"] is not None}
        | ({round(pooled_best["value_star"], 3)} if pooled_best is not None else set())
        | {round(float(baseline_value), 3)}
    )
    holdout_results = {val: eval_fn(val, holdout_ctx) for val in candidates}
    valid_holdout = {v: r for v, r in holdout_results.items() if r is not None}
    if not valid_holdout:
        return {"skipped": True, "per_fold": per_fold, "pooled_best": pooled_best}

    winner = min(valid_holdout, key=lambda v: valid_holdout[v]["ratio"])
    return {
        "skipped": False, "value": winner,
        "holdout_ratio": valid_holdout[winner]["ratio"], "holdout_coverage": valid_holdout[winner]["coverage"],
        "candidates": [{"value": v, "holdout_ratio": r["ratio"], "holdout_coverage": r["coverage"]}
                        for v, r in sorted(valid_holdout.items())],
        "per_fold": per_fold, "pooled_best": pooled_best,
    }


def _calibrate_one_feature(feat_name, fixed_params, grid, dev_fold_ctxs, holdout_ctx, q_low, q_high):
    def eval_fn(val, ctx):
        trial = {**fixed_params, feat_name: float(val)}
        return _eval_params(trial, ctx, q_low, q_high)

    res = _calibrate_one_value(eval_fn, grid, dev_fold_ctxs, holdout_ctx, baseline_value=0.0)
    return {"feature": feat_name, **res}


def _refine_grid(coarse_grid, center) -> tuple:
    coarse_step = float(coarse_grid[1] - coarse_grid[0])
    fine_step = coarse_step * REFINE_STEP_FRACTION
    lo, hi = float(min(coarse_grid)), float(max(coarse_grid))
    span_lo = max(lo, center - coarse_step)
    span_hi = min(hi, center + coarse_step)
    return tuple(np.round(np.arange(span_lo, span_hi + fine_step / 2, fine_step), 3))


def _calibrate_q_read(which, params, q_low, q_high, q_low_read, q_high_read, grid, dev_fold_ctxs, holdout_ctx):
    def eval_fn(val, ctx):
        qlr = val if which == "low" else q_low_read
        qhr = val if which == "high" else q_high_read
        return _eval_params(params, ctx, q_low, q_high, qlr, qhr)

    nominal = q_low if which == "low" else q_high
    res = _calibrate_one_value(eval_fn, grid, dev_fold_ctxs, holdout_ctx, baseline_value=nominal)
    return {"which": which, **res}


def calibrate_combined_multipass(close, high, low, H, p, theiler, volumes=None, levels=(90,),
                                  feature_order=None, n_folds=N_FOLDS, fold_size=FOLD_SIZE, n_holdout=N_HOLDOUT,
                                  gap=None, max_passes=DEFAULT_MAX_PASSES, tol=DEFAULT_TOL, refine=DEFAULT_REFINE,
                                  calibrate_q_read=True, max_candles=MAX_CANDLES_DEFAULT, progress_cb=None):
    """Многопроходный координатный спуск (θ + 5 λ-признаков) + per-level
    read-квантиль — см. модульный докстринг. Возвращает {"params",
    "n_passes", "converged", "all_passes", "levels", "widest_level",
    "q_read_by_level", "q_read_calibration_by_level", "final_*"} или
    {"skipped": True, "reason": ...}."""
    if volumes is None:
        volumes = np.zeros(len(close))
    close, high, low, volumes = truncate_candles(close, high, low, volumes, max_candles)

    if feature_order is None:
        feature_order = list(FEATURE_ORDER) if volumes.any() else ["theta"] + [f for f in RANK_FEATURES if f != "volume"]

    widest = max(levels)
    q_low, q_high = (100 - widest) / 200.0, (100 + widest) / 200.0

    n = len(close)
    dratio = dratio_from_close(close)
    roll_low_full, roll_high_full = rolling_cummin_cummax(low, high, H)
    roll_low, roll_high = roll_low_full[:, H - 1], roll_high_full[:, H - 1]
    log_highs, log_lows = np.log(np.maximum(high, 1e-10)), np.log(np.maximum(low, 1e-10))
    rank_dict = compute_bar_rank_dict(log_highs, log_lows, volumes)
    if gap is None:
        gap = max(H, 5)

    holdout_origins = list(range(n - 1 - H - n_holdout, n - 1 - H))
    dev_end = n - 1 - H - n_holdout - gap
    dev_start = dev_end - n_folds * fold_size
    if dev_start < p + theiler + 20:
        return {"skipped": True, "reason": "insufficient history for dev/holdout split"}
    dev_origins_all = list(range(dev_start, dev_end))
    fold_origin_lists = [dev_origins_all[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]

    def build_ctx(origins):
        ctxs = [_origin_context(t, dratio, close, high, low, p, theiler, H, q_low, q_high, roll_low, roll_high, rank_dict)
                for t in origins]
        return [c for c in ctxs if c is not None]

    dev_fold_ctxs = [build_ctx(o) for o in fold_origin_lists]
    holdout_ctx = build_ctx(holdout_origins)
    if not holdout_ctx or all(not c for c in dev_fold_ctxs):
        return {"skipped": True, "reason": "no valid origins after feature/pool masking"}

    params = {f: 0.0 for f in feature_order}
    all_passes = []
    total_steps = max_passes * len(feature_order)
    done_steps = 0
    converged = False
    for pass_num in range(1, max_passes + 1):
        prev_params = dict(params)
        per_feature = []
        for feat_name in feature_order:
            res = _calibrate_one_feature(feat_name, params, GRIDS[feat_name], dev_fold_ctxs, holdout_ctx, q_low, q_high)
            per_feature.append(res)
            if not res.get("skipped"):
                params[feat_name] = res["value"]
            done_steps += 1
            if progress_cb:
                progress_cb(done_steps, total_steps)
        max_delta = max(abs(params[f] - prev_params[f]) for f in feature_order)
        all_passes.append({"pass": pass_num, "params": dict(params), "per_feature": per_feature, "max_delta": max_delta})
        if max_delta <= tol:
            converged = True
            break

    if refine:
        per_feature = []
        prev_params = dict(params)
        for feat_name in feature_order:
            fine_grid = _refine_grid(GRIDS[feat_name], params[feat_name])
            res = _calibrate_one_feature(feat_name, params, fine_grid, dev_fold_ctxs, holdout_ctx, q_low, q_high)
            per_feature.append(res)
            if not res.get("skipped"):
                params[feat_name] = res["value"]
        max_delta = max(abs(params[f] - prev_params[f]) for f in feature_order)
        all_passes.append({"pass": "refine", "params": dict(params), "per_feature": per_feature, "max_delta": max_delta})

    q_read_by_level = {}
    q_read_calibration_by_level = {}
    if calibrate_q_read:
        for lv in levels:
            lv_q_low, lv_q_high = (100 - lv) / 200.0, (100 + lv) / 200.0
            if lv == widest:
                lv_dev_ctxs, lv_holdout_ctx = dev_fold_ctxs, holdout_ctx
            else:
                lv_dev_ctxs = [_relevel_pbu(c, close, roll_low, roll_high, theiler, lv_q_low, lv_q_high) for c in dev_fold_ctxs]
                lv_holdout_ctx = _relevel_pbu(holdout_ctx, close, roll_low, roll_high, theiler, lv_q_low, lv_q_high)

            q_read = {"low": lv_q_low, "high": lv_q_high}
            calib = []
            grid_low = tuple(np.round(np.arange(max(1e-3, lv_q_low - Q_READ_SPAN),
                                                 min(0.5 - 1e-3, lv_q_low + Q_READ_SPAN) + Q_READ_STEP / 2, Q_READ_STEP), 3))
            grid_high = tuple(np.round(np.arange(max(0.5 + 1e-3, lv_q_high - Q_READ_SPAN),
                                                  min(1 - 1e-3, lv_q_high + Q_READ_SPAN) + Q_READ_STEP / 2, Q_READ_STEP), 3))
            res_low = _calibrate_q_read("low", params, lv_q_low, lv_q_high, q_read["low"], q_read["high"],
                                         grid_low, lv_dev_ctxs, lv_holdout_ctx)
            calib.append(res_low)
            if not res_low.get("skipped"):
                q_read["low"] = res_low["value"]
            res_high = _calibrate_q_read("high", params, lv_q_low, lv_q_high, q_read["low"], q_read["high"],
                                          grid_high, lv_dev_ctxs, lv_holdout_ctx)
            calib.append(res_high)
            if not res_high.get("skipped"):
                q_read["high"] = res_high["value"]

            q_read_by_level[lv] = q_read
            q_read_calibration_by_level[lv] = calib
    else:
        q_read_by_level = {lv: {"low": (100 - lv) / 200.0, "high": (100 + lv) / 200.0} for lv in levels}

    baseline = {f: 0.0 for f in feature_order}
    widest_read = q_read_by_level[widest]
    final_combo = _eval_params(params, holdout_ctx, q_low, q_high, widest_read["low"], widest_read["high"])
    final_baseline = _eval_params(baseline, holdout_ctx, q_low, q_high)
    rel_final = (final_combo["ratio"] / final_baseline["ratio"]) if final_combo and final_baseline else None

    return {
        "skipped": False, "params": params, "n_passes": len(all_passes), "converged": converged,
        "all_passes": all_passes, "levels": tuple(levels), "widest_level": widest,
        "q_read_by_level": q_read_by_level, "q_read_calibration_by_level": q_read_calibration_by_level,
        "final_ratio": final_combo["ratio"] if final_combo else None,
        "final_baseline_ratio": final_baseline["ratio"] if final_baseline else None,
        "final_rel": rel_final,
        "final_coverage": final_combo["coverage"] if final_combo else None,
        "final_baseline_coverage": final_baseline["coverage"] if final_baseline else None,
        "n_holdout": final_combo["n"] if final_combo else None,
        "n_folds": n_folds, "fold_size": fold_size, "gap": gap,
    }


def calibrate_range_forecast_task(instrument_id: int, interval: str, h_steps: int, p: int, theiler: int,
                                   levels: tuple, close, high, low, volumes) -> dict:
    """Picklable top-level ProcessPoolExecutor entry point. DB-запись — в
    task_manager (I/O не здесь)."""
    result = calibrate_combined_multipass(close, high, low, H=h_steps, p=p, theiler=theiler,
                                           volumes=volumes, levels=levels)
    return {"instrument_id": instrument_id, "interval": interval, "h_steps": h_steps, "p": p,
            "theiler": theiler, **result}
