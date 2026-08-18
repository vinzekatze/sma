#!/usr/bin/env python3
"""
29j_combined_calibrator.py — совместный калибратор весов пула (объём + тренд +
скорость + ускорение) для S-map полосы: координатный спуск по всем 4
признакам сразу — «пускай определит выгодную развесовку для каждого
признака» (прямой запрос пользователя, 2026-07-08). Заменяет 29i (парный
объём+1 признак) как основной комбинационный артефакт.

⚠️ ИСПРАВЛЕНИЕ OBJECTIVE (2026-07-08): 29g/29h/29i отбирали λ по cov_err_avg
(среднее |покрытие−номинал| по 6 уровням h1/h2×{50,75,90}%). Прямая проверка
на GAZP показала: «улучшение» cov_err_avg при λ*=11 целиком объяснялось ОДНИМ
из 6 покрытий (cov50_h2: 0.40→0.45), а ПОЛОСА при этом стала ШИРЕ (+2…+4% по
всем 4 замерам ширины) — размазывание, а не уточнение (см. [[feedback-
uncertainty-field-width-control]]). Пересчёт того же GAZP с pinball_norm_avg
(proper scoring rule, структурно штрафует лишнюю ширину) даёт λ*=0 — эффекта
нет. objective здесь и во всех дочерних расчётах — pinball_norm_avg; cov_err_
avg и явная ширина полосы (width_avg = band[0.9]−band[0.1]) — диагностика.

Координатный спуск (НЕ полный совместный grid — 4 признака × ~9 точек сетки
= 6561 комбинаций на тикер при полном переборе, нереально дорого): по
очереди калибруется λ ОДНОГО признака при уже найденных (или ещё нулевых) λ
остальных, один проход по порядку volume → trend → velocity → acceleration
(по убыванию силы одиночного сигнала в 29g/29h-скринингах). Признаки
коррелированы (velocity↔trend +0.51, velocity↔acceleration +0.66 — см.
память research-mlp... нет, см. корреляционную проверку 29h), поэтому
порядок влияет на то, кому достанется общий эффект; trend↔acceleration
почти независимы (+0.06) — это два действительно разных источника сигнала,
velocity — наполовину дубликат обоих.

θ=0, m=6, min_bars=5, T задаётся пользователем (--t, НЕ калибруется —
концептуальная позиция зафиксирована в 29g). Сетка λ по умолчанию грубая
(шаг 2.0, [0,16]) — как в 29h-скрининге, соразмерно координатному спуску
(4 прохода на тикер); для точечной калибровки под конкретный тикер в
основном приложении сузить шаг через --lambda-step.

Использование:
    python 29j_combined_calibrator.py GAZP --t 0.20
    python 29j_combined_calibrator.py --all --t 0.20
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
MIN_BARS = calib29g.MIN_BARS
COVERAGE_LEVELS = calib29g.COVERAGE_LEVELS
MIN_CALIB_ORIGINS = calib29g.MIN_CALIB_ORIGINS
MIN_TEST_ORIGINS = calib29g.MIN_TEST_ORIGINS
DEV_SPLIT_FRACS = calib29g.DEV_SPLIT_FRACS
coverage_at = calib29g.coverage_at

FEATURE_ORDER = ["volume", "trend", "velocity", "acceleration"]
DEFAULT_LAMBDA_LO, DEFAULT_LAMBDA_HI, DEFAULT_LAMBDA_STEP = 0.0, 16.0, 2.0


# ── обобщение build_pool_vectors_with_rank / build_causal_pool_with_rank
#    (29g) на СЛОВАРЬ рангов (произвольный набор признаков), а не один ──

def build_pool_vectors_with_ranks(pivot_prices, pivot_dirs, pivot_ranks: dict, embedding_dim, horizon):
    n = len(pivot_prices)
    valid = np.arange(embedding_dim, n - horizon)
    if len(valid) == 0:
        empty_ranks = {k: np.zeros(0) for k in pivot_ranks}
        return np.zeros((0, embedding_dim)), np.zeros(0), np.zeros(0, dtype=np.int8), empty_ranks
    lag_idx = valid[:, None] - np.arange(embedding_dim)[None, :]
    feats = pivot_prices[lag_idx] - pivot_prices[lag_idx - 1]
    tars = pivot_prices[valid + horizon] - pivot_prices[valid]
    dirs = pivot_dirs[valid]
    ranks = {k: v[valid] for k, v in pivot_ranks.items()}
    finite = np.all(np.isfinite(feats), axis=1) & np.isfinite(tars)
    for k in ranks:
        finite &= np.isfinite(ranks[k])
    out_ranks = {k: v[finite] for k, v in ranks.items()}
    return feats[finite], tars[finite], dirs[finite], out_ranks


def build_causal_pool_with_ranks(cutoff_date, ticker_data, t_pool, embedding_dim, horizon, min_bars, feat_names):
    """ticker_data: {ticker: (lh, ll, dates, {feat: rank_series})}"""
    feats_l, tars_l, dirs_l = [], [], []
    ranks_l = {f: [] for f in feat_names}
    for ticker, (lh, ll, dates, rank_dict) in ticker_data.items():
        mask = dates <= cutoff_date
        if mask.sum() < embedding_dim + horizon + 2:
            continue
        lh_c, ll_c, dt_c = lh[mask], ll[mask], dates[mask]
        rank_c = {f: rank_dict[f][mask] for f in feat_names}
        pivot_prices, confirm_dates, pivot_dirs = build_zigzag(lh_c, ll_c, dt_c, t_pool, min_bars)
        if len(pivot_prices) == 0:
            continue
        bar_idx = np.searchsorted(dt_c, confirm_dates)
        pivot_ranks = {f: rank_c[f][bar_idx] for f in feat_names}
        feats, tars, dirs, ranks = build_pool_vectors_with_ranks(
            pivot_prices, pivot_dirs, pivot_ranks, embedding_dim, horizon)
        if len(tars):
            feats_l.append(feats); tars_l.append(tars); dirs_l.append(dirs)
            for f in feat_names:
                ranks_l[f].append(ranks[f])
    if not tars_l:
        empty_ranks = {f: np.zeros(0) for f in feat_names}
        return np.zeros((0, embedding_dim)), np.zeros(0), np.zeros(0, dtype=np.int8), empty_ranks
    out_ranks = {f: np.concatenate(v) for f, v in ranks_l.items()}
    return np.vstack(feats_l), np.concatenate(tars_l), np.concatenate(dirs_l), out_ranks


def evaluate_origins_multi(target, t, lambdas: dict, origins, q_lp, q_dates, q_dirs,
                           rank_series_target: dict, ticker_data, feat_names, min_bars):
    q_levels_needed = sorted(set(calib29.Q_LEVELS) |
                             {(1 - lv) / 2 for lv in COVERAGE_LEVELS} |
                             {(1 + lv) / 2 for lv in COVERAGE_LEVELS})
    rows_h1, rows_h2 = [], []
    for origin in origins:
        q_dir = int(q_dirs[origin])
        origin_date = str(q_dates[origin])
        origin_bar = int(np.searchsorted(ticker_data[target][2], origin_date))
        origin_bar = min(origin_bar, len(rank_series_target[feat_names[0]]) - 1)
        rank_query = {f: rank_series_target[f][origin_bar] for f in feat_names}
        if any(not np.isfinite(v) for v in rank_query.values()):
            continue

        actual_lr_1 = float(q_lp[origin + 1] - q_lp[origin])
        actual_lr_2 = float(q_lp[origin + 2] - q_lp[origin])

        _pfm1, ptr1, pdir1, pranks1 = build_causal_pool_with_ranks(
            origin_date, ticker_data, t, M, 1, min_bars, feat_names)
        _pfm2, ptr2, pdir2, pranks2 = build_causal_pool_with_ranks(
            origin_date, ticker_data, t, M, 2, min_bars, feat_names)
        mask1 = pdir1 == q_dir
        mask2 = pdir2 == q_dir
        if mask1.sum() < M + 2 or mask2.sum() < M + 2:
            continue

        sq1 = np.zeros(int(mask1.sum()))
        sq2 = np.zeros(int(mask2.sum()))
        for f in feat_names:
            sq1 += lambdas[f] * (pranks1[f][mask1] - rank_query[f]) ** 2
            sq2 += lambdas[f] * (pranks2[f][mask2] - rank_query[f]) ** 2
        w1, w2 = np.exp(-sq1), np.exp(-sq2)
        if w1.sum() < 1e-9 or w2.sum() < 1e-9:
            continue

        band1 = weighted_quantile(ptr1[mask1], w1, q_levels_needed)
        band2 = weighted_quantile(ptr2[mask2], w2, q_levels_needed)

        pb1 = calib29.pinball(actual_lr_1, band1, calib29.Q_LEVELS)
        pb2 = calib29.pinball(actual_lr_2, band2, calib29.Q_LEVELS)
        rows_h1.append({"pinball": pb1, "width": band1[0.9] - band1[0.1],
                        **{f"cov{int(lv*100)}": coverage_at(actual_lr_1, band1, lv)
                           for lv in COVERAGE_LEVELS}})
        rows_h2.append({"pinball": pb2, "width": band2[0.9] - band2[0.1],
                        **{f"cov{int(lv*100)}": coverage_at(actual_lr_2, band2, lv)
                           for lv in COVERAGE_LEVELS}})

    if len(rows_h1) < 1:
        return None
    df1, df2 = pd.DataFrame(rows_h1), pd.DataFrame(rows_h2)
    cov_errs = []
    for lv in COVERAGE_LEVELS:
        cov_errs.append(abs(df1[f"cov{int(lv*100)}"].mean() - lv))
        cov_errs.append(abs(df2[f"cov{int(lv*100)}"].mean() - lv))
    return {
        "n": len(df1),
        "pinball_norm_avg": (df1.pinball.mean() + df2.pinball.mean()) / 2 / t,
        "cov_err_avg": float(np.mean(cov_errs)),
        "width_avg": (df1.width.mean() + df2.width.mean()) / 2,
    }


def calibrate_feature(target, t, feat_name, lambdas_fixed: dict, feat_names, ticker_data, t0,
                      lambda_lo, lambda_hi, lambda_step, q_lp, q_dates, q_dirs, dev, holdout,
                      rank_series_t):
    """Калибрует λ ОДНОГО признака (feat_name) при фиксированных λ остальных
    (lambdas_fixed) — один шаг координатного спуска. Схема идентична 29g:
    сетка × DEV_SPLIT_FRACS → _grid_argmin (по pinball_norm) → honest holdout."""
    lam_grid = np.round(np.arange(lambda_lo, lambda_hi + 1e-9, lambda_step), 3)
    rows_by_split = {frac: [] for frac in DEV_SPLIT_FRACS}
    for lam in lam_grid:
        trial = {**lambdas_fixed, feat_name: float(lam)}
        for frac in DEV_SPLIT_FRACS:
            split_i = int(len(dev) * frac)
            calib_o = dev[:split_i]
            if len(calib_o) < MIN_CALIB_ORIGINS:
                continue
            r = evaluate_origins_multi(target, t, trial, calib_o, q_lp, q_dates, q_dirs,
                                       rank_series_t, ticker_data, feat_names, MIN_BARS)
            if r is not None:
                rows_by_split[frac].append({"t": float(lam), "score": r["pinball_norm_avg"]})

    candidates = {}
    for frac, rows in rows_by_split.items():
        res = calib29f._grid_argmin(rows, key="score")
        if res is not None:
            candidates[frac] = res
    if not candidates:
        print(f"    [{feat_name}] ПРОПУСК: ни один сплит не дал кандидата")
        return {"feature": feat_name, "skipped": True}

    unique_lams = sorted(set(round(c["t_star"], 3) for c in candidates.values()))
    spread = round(max(unique_lams) - min(unique_lams), 3)

    holdout_results = {}
    for lam in sorted(set(unique_lams) | {0.0}):
        trial = {**lambdas_fixed, feat_name: lam}
        r = evaluate_origins_multi(target, t, trial, holdout, q_lp, q_dates, q_dirs,
                                   rank_series_t, ticker_data, feat_names, MIN_BARS)
        holdout_results[lam] = r

    valid = {l: r for l, r in holdout_results.items() if r is not None}
    if not valid or 0.0 not in valid:
        print(f"    [{feat_name}] ПРОПУСК: holdout недоступен")
        return {"feature": feat_name, "skipped": True}

    winner = min(valid, key=lambda l: valid[l]["pinball_norm_avg"])
    baseline = valid[0.0]
    rel = valid[winner]["pinball_norm_avg"] / baseline["pinball_norm_avg"]
    w = valid[winner]
    print(f"    [{feat_name}] кандидаты по сплитам: " +
          ", ".join(f"{frac:.2f}→{c['t_star']:.2f}" for frac, c in sorted(candidates.items())))
    print(f"    [{feat_name}] λ*={winner:.2f}  pinball_norm={w['pinball_norm_avg']:.4f} "
          f"(baseline={baseline['pinball_norm_avg']:.4f}, rel={rel:.4f})  "
          f"width={w['width_avg']:.4f} (baseline={baseline['width_avg']:.4f})  "
          f"cov_err={w['cov_err_avg']:.4f} (baseline={baseline['cov_err_avg']:.4f})")

    return {
        "feature": feat_name, "skipped": False, "lambda_star": winner, "spread": spread,
        "pinball_rel": rel, "pinball_norm": w["pinball_norm_avg"],
        "baseline_pinball_norm": baseline["pinball_norm_avg"],
        "width_avg": w["width_avg"], "baseline_width": baseline["width_avg"],
        "cov_err_avg": w["cov_err_avg"], "baseline_cov_err": baseline["cov_err_avg"],
        "n_holdout": w["n"],
    }


def calibrate_combined(target, t, ticker_data, t0, lambda_lo, lambda_hi, lambda_step):
    lh_t, ll_t, dates_t, rank_dict_t = ticker_data[target]
    q_lp, q_dates, q_dirs = build_zigzag(lh_t, ll_t, dates_t, t, MIN_BARS)
    print(f"{target} T={t*100:.0f}%: {len(q_lp)} пивотов")

    min_hist = M + 3
    origins = calib29.prepare_origins(q_lp, min_hist)
    holdout_target_n = max(MIN_TEST_ORIGINS, 20)
    if len(origins) <= holdout_target_n:
        print(f"  ПРОПУСК: недостаточно origin'ов ({len(origins)}) для holdout-схемы")
        return {"target": target, "skipped": True}
    holdout_cutoff_date = str(q_dates[origins[-holdout_target_n]])
    dev = [o for o in origins if str(q_dates[o]) < holdout_cutoff_date]
    holdout = [o for o in origins if str(q_dates[o]) >= holdout_cutoff_date]
    print(f"  origin'ов: {len(origins)}  (dev={len(dev)}, holdout={len(holdout)})")

    lambdas = {f: 0.0 for f in FEATURE_ORDER}
    per_feature = []
    for feat_name in FEATURE_ORDER:
        print(f"  --- координата: {feat_name}  (фикс. остальные: {lambdas}) ---")
        res = calibrate_feature(target, t, feat_name, lambdas, FEATURE_ORDER, ticker_data, t0,
                                lambda_lo, lambda_hi, lambda_step, q_lp, q_dates, q_dirs,
                                dev, holdout, rank_dict_t)
        per_feature.append(res)
        if not res.get("skipped"):
            lambdas[feat_name] = res["lambda_star"]

    baseline_zero = {f: 0.0 for f in FEATURE_ORDER}
    final_combo = evaluate_origins_multi(target, t, lambdas, holdout, q_lp, q_dates, q_dirs,
                                         rank_dict_t, ticker_data, FEATURE_ORDER, MIN_BARS)
    final_baseline = evaluate_origins_multi(target, t, baseline_zero, holdout, q_lp, q_dates, q_dirs,
                                            rank_dict_t, ticker_data, FEATURE_ORDER, MIN_BARS)
    rel_final = (final_combo["pinball_norm_avg"] / final_baseline["pinball_norm_avg"]
                if final_combo and final_baseline else None)

    print(f"  ИТОГ комбинации {lambdas}:")
    if final_combo and final_baseline:
        print(f"    pinball_norm={final_combo['pinball_norm_avg']:.4f} "
              f"(baseline={final_baseline['pinball_norm_avg']:.4f}, rel={rel_final:.4f})  "
              f"width={final_combo['width_avg']:.4f} (baseline={final_baseline['width_avg']:.4f})  "
              f"cov_err={final_combo['cov_err_avg']:.4f} (baseline={final_baseline['cov_err_avg']:.4f})")
    print(f"  [{time.time()-t0:.1f}s]\n")

    return {
        "target": target, "skipped": False, "lambdas": lambdas, "per_feature": per_feature,
        "final_pinball_norm": final_combo["pinball_norm_avg"] if final_combo else None,
        "final_baseline_pinball_norm": final_baseline["pinball_norm_avg"] if final_baseline else None,
        "final_rel": rel_final,
        "final_width": final_combo["width_avg"] if final_combo else None,
        "final_baseline_width": final_baseline["width_avg"] if final_baseline else None,
        "final_cov_err": final_combo["cov_err_avg"] if final_combo else None,
        "final_baseline_cov_err": final_baseline["cov_err_avg"] if final_baseline else None,
        "n_holdout": final_combo["n"] if final_combo else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Совместный калибратор λ (объём+тренд+скорость+ускорение) для S-map полосы, "
                     "координатный спуск, objective=pinball_norm_avg")
    parser.add_argument("ticker", nargs="?", help="Целевой тикер (напр. GAZP)")
    parser.add_argument("--all", action="store_true", help="Прогнать все 7 целевых тикеров")
    parser.add_argument("--t", type=float, default=0.20, metavar="T",
                        help="T_query=T_pool, доля — выбор пользователя, не калибруется")
    parser.add_argument("--lambda-lo", type=float, default=DEFAULT_LAMBDA_LO, metavar="L")
    parser.add_argument("--lambda-hi", type=float, default=DEFAULT_LAMBDA_HI, metavar="L")
    parser.add_argument("--lambda-step", type=float, default=DEFAULT_LAMBDA_STEP, metavar="L")
    args = parser.parse_args()
    if not args.all and not args.ticker:
        parser.error("укажите тикер или --all")
    targets = calib29f.TARGETS if args.all else [args.ticker]
    t = args.t

    t0 = time.time()
    print(f"Загрузка кросс-тикерного пула + признаков ({len(UNIVERSE)} тикеров)…")
    ticker_data = {}
    for tk in UNIVERSE:
        loaded = calib29g.load_ticker_candles_with_volume(tk, "1d")
        if loaded is None:
            continue
        lh, ll, dates, vol = loaded
        rank_dict = {
            "volume": calib29g.compute_density_rank_series(lh, ll, vol, calib29g.RANK_WINDOW, calib29g.N_BINS),
            "trend": calib29h.to_percentile_rank(calib29h.compute_trend_raw(lh, ll)),
            "velocity": calib29h.to_percentile_rank(calib29h.compute_velocity_raw(lh, ll)),
            "acceleration": calib29h.to_percentile_rank(calib29h.compute_acceleration_raw(lh, ll)),
        }
        ticker_data[tk] = (lh, ll, dates, rank_dict)
    print(f"загружено {len(ticker_data)}/{len(UNIVERSE)} тикеров  [{time.time()-t0:.1f}s]\n")

    results = []
    for target in targets:
        print(f"--- {target} (T={t*100:.0f}%, λ∈[{args.lambda_lo},{args.lambda_hi}] шаг={args.lambda_step}) ---")
        results.append(calibrate_combined(target, t, ticker_data, t0,
                                          args.lambda_lo, args.lambda_hi, args.lambda_step))

    ok = [r for r in results if not r.get("skipped")]
    if ok:
        summary_rows, detail_rows = [], []
        for r in ok:
            summary_rows.append({
                "target": r["target"],
                **{f"lambda_{f}": r["lambdas"][f] for f in FEATURE_ORDER},
                "final_pinball_norm": r["final_pinball_norm"],
                "final_baseline_pinball_norm": r["final_baseline_pinball_norm"],
                "final_rel": r["final_rel"],
                "final_width": r["final_width"],
                "final_baseline_width": r["final_baseline_width"],
                "final_cov_err": r["final_cov_err"],
                "final_baseline_cov_err": r["final_baseline_cov_err"],
                "n_holdout": r["n_holdout"],
            })
            for pf in r["per_feature"]:
                if not pf.get("skipped"):
                    detail_rows.append({"target": r["target"], **pf})
        df = pd.DataFrame(summary_rows)
        out_path = RESULTS / "29j_combined_summary.csv"
        df.to_csv(out_path, index=False)
        print(df.to_string(index=False))
        print(f"\nСохранено: {out_path}")
        if detail_rows:
            ddf = pd.DataFrame(detail_rows)
            det_path = RESULTS / "29j_combined_detail.csv"
            ddf.to_csv(det_path, index=False)
            print(f"Детали по признакам: {det_path}")

    skipped = [r["target"] for r in results if r.get("skipped")]
    if skipped:
        print(f"Пропущены (недостаточно данных): {skipped}")

    print(f"\nВсего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
