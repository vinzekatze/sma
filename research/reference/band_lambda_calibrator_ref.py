#!/usr/bin/env python3
"""
band_lambda_calibrator_ref.py — калибратор весов пула (λ) для S-map полосы
неопределённости на 6 признаках, многопроходный координатный спуск.
Прямое продолжение smap_band_ref.py — та же каузальная схема (build_zigzag,
build_causal_pool), но кернел S-map получает ДОПОЛНИТЕЛЬНЫЙ мультипликативный
вес по контексту события, не только θ-локализацию по фазовому вектору.

Консолидировано из research/phase7_fractality/29_band_calibration/29g,29h,
29k,29l — по итогам сессии 2026-07-08, после того как набор из 6 признаков
подтвердил честный (pinball_norm_avg-объектив, не cov_err_avg — см. ниже)
эффект на нескольких из 7 тестовых тикеров. Это ЦЕЛЕВОЙ АРТЕФАКТ
(не разовая диагностика) — держать в актуальном состоянии.

⚠️ objective — ТОЛЬКО pinball_norm_avg (pinball/T, proper scoring rule).
НЕ cov_err_avg (среднее |покрытие−номинал|) — тот давал размазывание
полосы без реальной точности (GAZP «-60%», CHMF «-13%» из первых прогонов
29g были ПОЛНОСТЬЮ артефактом этой метрики: улучшение объяснялось ОДНИМ из
6 покрытий, полоса при этом становилась ШИРЕ). См. память project-phase7-
band-calibrator-objective-fix. cov_err_avg и явная ширина полосы
(width_avg = band[0.9]−band[0.1]) остаются в выводе ТОЛЬКО как диагностика.

Признаки (rank ∈ [0,1], причинный перцентильный ранг — единый масштаб,
делает λ разных признаков сравнимыми):
  volume       — bar-нативный. Перцентильный ранг цены бара в объёмном
                 профиле trailing 252 бара (крупные пивоты приземляются в
                 зонах низкой объёмной плотности — прецедент эксп.20b/20e
                 точечного пайплайна фазы 7).
  trend        — bar-нативный. Лог-расстояние mid-цены от SMA(100),
                 перцентильный ранг (252 бара).
  velocity     — bar-нативный. Лог-доходность за последние 10 баров,
                 перцентильный ранг (252 бара).
  acceleration — bar-нативный. Изменение velocity за 10 баров (вторая
                 производная), перцентильный ранг (252 бара).
  volatility   — bar-нативный. std лог-доходностей mid-цены, trailing
                 20 баров, перцентильный ранг (252 бара).
  leg_age      — PIVOT-нативный, T-ЗАВИСИМЫЙ. Число баров, занятое
                 формированием последнего ПОДТВЕРЖДЁННОГО плеча зигзага,
                 перцентильный ранг относительно последних K=10 плеч.
                 Плечо существует только относительно конкретного T —
                 считается ЗАНОВО на каждый вызов build_causal_pool_with_
                 mixed_ranks (сразу после build_zigzag, который и так там
                 строится), не предзагружается как bar-нативные признаки.

  leg_amplitude (размер последнего плеча) ИСКЛЮЧЁН — скрининг дал λ*=0 на
  2/2 тикерах (GAZP, CHMF), сигнала нет.

Вес пула: w_j = exp(−Σ_f λ_f · (rank_pool[f]_j − rank_query[f])²) —
мультипликативный кернел-фактор (НЕ фильтр, НЕ постфактум-коррекция —
оба паттерна систематически проваливались в фазе 7, см. память feedback-
kernel-weight-over-filter-correction), домножается на равномерный θ=0
S-map вес (θ калибровка не пережила OOS ни на одном из 7 тикеров).

Многопроходный координатный спуск (calibrate_combined_multipass): один
проход = один раз по FEATURE_ORDER (volume→trend→leg_age→velocity→
acceleration→volatility, по убыванию ожидаемой силы одиночного сигнала).
Проходы повторяются, пока λ КАЖДОГО признака не стабилизируется
(max|Δλ| ≤ tol между проходами) или не исчерпан max_passes. Признаки
коррелированы (velocity↔trend +0.51, velocity↔acceleration +0.66,
trend↔acceleration +0.06 — почти независимы) — порядок в ОДНОМ проходе
влияет на то, кому достанется общий эффект; повторные проходы дают
более ранним признакам шанс скорректироваться под уже откалиброванные
поздние (см. PLZL: однопроходная версия дала чуть лучший результат, чем
6-признаковая однопроходная — координатный шум, не деградация).

Методология калибровки одного признака (как T в 29f, λ в 29g/29j/29l):
сетка × DEV_SPLIT_FRACS calib-срезов dev-части истории → сглаженный
argmin на каждом срезе → кандидаты проверяются на ОТДЕЛЬНОМ честном
holdout (последние origin'ы, НИКОГДА не участвует в подборе) РОВНО ОДИН
РАЗ, побеждает лучший по pinball_norm_avg (включая λ=0 — законный исход,
если ни один кандидат не бьёт baseline).

θ=0, m=6, min_bars=5, T задаётся пользователем (--t) — НЕ калибруется
(T — выбор пользователя под свою торговую задачу, см. память project-
phase7-uncertainty-field-pivot и docstring исходного 29g).

Использование:
    python band_lambda_calibrator_ref.py SBER --t 0.20
    python band_lambda_calibrator_ref.py --all --t 0.20 --max-passes 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from smap_band_ref import (
    build_zigzag, build_query_vector, weighted_quantile, UNIVERSE, DATA_DIR,
)

TARGETS = ["SBER", "LKOH", "GAZP", "MGNT", "CHMF", "MTSS", "PLZL"]

M = 6
MIN_BARS = 5
Q_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
COVERAGE_LEVELS = [0.50, 0.75, 0.90]
RANK_WINDOW = 252
N_BINS = 40
WINDOW_VOL = 20
K_LEG = 10

MIN_CALIB_ORIGINS = 25
MIN_TEST_ORIGINS = 15
DEV_SPLIT_FRACS = [0.55, 0.65, 0.75, 0.85]
SMOOTH_WINDOW = 3

FEATURE_ORDER = ["volume", "trend", "leg_age", "velocity", "acceleration", "volatility"]
BAR_FEATURES = ["volume", "trend", "velocity", "acceleration", "volatility"]
PIVOT_FEATURES = ["leg_age"]

DEFAULT_LAMBDA_LO, DEFAULT_LAMBDA_HI, DEFAULT_LAMBDA_STEP = 0.0, 16.0, 2.0
DEFAULT_MAX_PASSES = 3
DEFAULT_CONVERGENCE_TOL = 0.5   # в единицах λ — меньше типичного lambda_step


# ── загрузка данных (лог-свечи + объём) ─────────────────────────────────

def load_ticker_candles_with_volume(ticker: str, interval: str, data_dir: Path = DATA_DIR):
    path = data_dir / ticker / f"{interval}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
        log_highs = np.log(np.array([c["high"] for c in raw], dtype=np.float64))
        log_lows  = np.log(np.array([c["low"]  for c in raw], dtype=np.float64))
        dates     = np.array([c["begin"] for c in raw])
        volumes   = np.array([float(c.get("volume", 0.0)) for c in raw], dtype=np.float64)
        return log_highs, log_lows, dates, volumes
    except (json.JSONDecodeError, KeyError):
        return None


# ── bar-нативные признаки (не зависят от T) ─────────────────────────────

def to_percentile_rank(series: np.ndarray, window: int = RANK_WINDOW) -> np.ndarray:
    """Причинный перцентильный ранг series[i] относительно ПРЕДЫДУЩИХ
    window значений (i само не входит)."""
    n = len(series)
    ranks = np.full(n, np.nan)
    for i in range(window, n):
        hist = series[i - window:i]
        finite = hist[np.isfinite(hist)]
        if len(finite) < 10 or not np.isfinite(series[i]):
            continue
        ranks[i] = float((finite < series[i]).mean())
    return ranks


def compute_density_rank_series(log_highs, log_lows, volumes, window=RANK_WINDOW, n_bins=N_BINS):
    """Причинный объёмный профиль: перцентильный ранг цены бара в
    объёмной гистограмме trailing window баров ДО него."""
    n = len(log_highs)
    mid = (log_highs + log_lows) / 2.0
    ranks = np.full(n, np.nan)
    for i in range(window, n):
        lo_bound = log_lows[i - window:i].min()
        hi_bound = log_highs[i - window:i].max()
        if not (hi_bound > lo_bound):
            continue
        hist, edges = np.histogram(mid[i - window:i], bins=n_bins,
                                   range=(lo_bound, hi_bound), weights=volumes[i - window:i])
        total = hist.sum()
        if total <= 0:
            continue
        bin_idx = int((mid[i] - lo_bound) / (hi_bound - lo_bound) * n_bins)
        bin_idx = max(0, min(bin_idx, n_bins - 1))
        cum_below = hist[:bin_idx].sum()
        cum_at = hist[bin_idx]
        ranks[i] = (cum_below + cum_at / 2.0) / total
    return ranks


def compute_trend_raw(log_highs, log_lows, window_ma=100):
    mid = (log_highs + log_lows) / 2.0
    n = len(mid)
    trend = np.full(n, np.nan)
    csum = np.concatenate([[0.0], np.cumsum(mid)])
    for i in range(window_ma, n):
        ma = (csum[i] - csum[i - window_ma]) / window_ma
        trend[i] = mid[i] - ma
    return trend


def compute_velocity_raw(log_highs, log_lows, window_v=10):
    mid = (log_highs + log_lows) / 2.0
    n = len(mid)
    vel = np.full(n, np.nan)
    vel[window_v:] = mid[window_v:] - mid[:-window_v]
    return vel


def compute_acceleration_raw(log_highs, log_lows, window_v=10):
    vel = compute_velocity_raw(log_highs, log_lows, window_v)
    n = len(vel)
    acc = np.full(n, np.nan)
    acc[window_v:] = vel[window_v:] - vel[:-window_v]
    return acc


def compute_volatility_raw(log_highs, log_lows, window_vol=WINDOW_VOL):
    mid = (log_highs + log_lows) / 2.0
    ret = np.diff(mid, prepend=np.nan)
    return pd.Series(ret).rolling(window_vol, min_periods=window_vol).std().to_numpy()


# ── pivot-нативные признаки (T-зависимые) ───────────────────────────────

def causal_pivot_percentile_rank(values: np.ndarray, k: int = K_LEG, min_hist: int = 3) -> np.ndarray:
    """Причинный перцентильный ранг values[i] относительно ПРЕДЫДУЩИХ до
    k значений (i само не входит). values[0] всегда NaN."""
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


def leg_age_raw(confirm_dates: np.ndarray, dates_full: np.ndarray) -> np.ndarray:
    bar_idx = np.searchsorted(dates_full, confirm_dates)
    n = len(bar_idx)
    age = np.full(n, np.nan)
    if n > 1:
        age[1:] = (bar_idx[1:] - bar_idx[:-1]).astype(float)
    return age


# ── пул с признаками (bar+pivot смешанно) ───────────────────────────────

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


def build_causal_pool_with_mixed_ranks(cutoff_date, ticker_data, t_pool, embedding_dim, horizon,
                                       min_bars, bar_feat_names=BAR_FEATURES,
                                       pivot_feat_names=PIVOT_FEATURES, k_leg=K_LEG):
    """ticker_data: {ticker: (lh, ll, dates, {bar_feat: rank_series})}.
    Один build_zigzag на тикер обслуживает и bar-, и pivot-нативные ранги."""
    feats_l, tars_l, dirs_l = [], [], []
    all_feat_names = bar_feat_names + pivot_feat_names
    ranks_l = {f: [] for f in all_feat_names}
    for ticker, (lh, ll, dates, bar_rank_dict) in ticker_data.items():
        mask = dates <= cutoff_date
        if mask.sum() < embedding_dim + horizon + 2:
            continue
        lh_c, ll_c, dt_c = lh[mask], ll[mask], dates[mask]
        bar_rank_c = {f: bar_rank_dict[f][mask] for f in bar_feat_names}
        pivot_prices, confirm_dates, pivot_dirs = build_zigzag(lh_c, ll_c, dt_c, t_pool, min_bars)
        if len(pivot_prices) < 3:
            continue
        bar_idx = np.searchsorted(dt_c, confirm_dates)
        pivot_ranks = {f: bar_rank_c[f][bar_idx] for f in bar_feat_names}
        for pf in pivot_feat_names:
            raw = leg_age_raw(confirm_dates, dt_c) if pf == "leg_age" else None
            pivot_ranks[pf] = causal_pivot_percentile_rank(raw, k_leg)
        feats, tars, dirs, ranks = build_pool_vectors_with_ranks(
            pivot_prices, pivot_dirs, pivot_ranks, embedding_dim, horizon)
        if len(tars):
            feats_l.append(feats); tars_l.append(tars); dirs_l.append(dirs)
            for f in all_feat_names:
                ranks_l[f].append(ranks[f])
    if not tars_l:
        empty_ranks = {f: np.zeros(0) for f in all_feat_names}
        return np.zeros((0, embedding_dim)), np.zeros(0), np.zeros(0, dtype=np.int8), empty_ranks
    out_ranks = {f: np.concatenate(v) for f, v in ranks_l.items()}
    return np.vstack(feats_l), np.concatenate(tars_l), np.concatenate(dirs_l), out_ranks


def pinball(actual: float, band: dict, q_levels: tuple) -> float:
    total = 0.0
    for q in q_levels:
        diff = actual - band[q]
        total += max(q * diff, (q - 1) * diff)
    return total / len(q_levels)


def coverage_at(actual: float, band: dict, level: float) -> int:
    q_lo, q_hi = (1 - level) / 2, (1 + level) / 2
    return int(band[q_lo] <= actual <= band[q_hi])


def prepare_origins(q_lp, min_hist):
    return list(range(min_hist, len(q_lp) - 2))


def evaluate_origins_mixed(target, t, lambdas: dict, origins, q_lp, q_dates, q_dirs,
                           rank_series_target: dict, q_rank_pivot: dict, ticker_data, min_bars):
    q_levels_needed = sorted(set(Q_LEVELS) |
                             {(1 - lv) / 2 for lv in COVERAGE_LEVELS} |
                             {(1 + lv) / 2 for lv in COVERAGE_LEVELS})
    rows_h1, rows_h2 = [], []
    for origin in origins:
        q_dir = int(q_dirs[origin])
        origin_date = str(q_dates[origin])
        origin_bar = int(np.searchsorted(ticker_data[target][2], origin_date))
        origin_bar = min(origin_bar, len(rank_series_target[BAR_FEATURES[0]]) - 1)
        rank_query = {f: rank_series_target[f][origin_bar] for f in BAR_FEATURES}
        for pf in PIVOT_FEATURES:
            rank_query[pf] = q_rank_pivot[pf][origin]
        if any(not np.isfinite(v) for v in rank_query.values()):
            continue

        actual_lr_1 = float(q_lp[origin + 1] - q_lp[origin])
        actual_lr_2 = float(q_lp[origin + 2] - q_lp[origin])

        _f1, ptr1, pdir1, pranks1 = build_causal_pool_with_mixed_ranks(
            origin_date, ticker_data, t, M, 1, min_bars)
        _f2, ptr2, pdir2, pranks2 = build_causal_pool_with_mixed_ranks(
            origin_date, ticker_data, t, M, 2, min_bars)
        mask1 = pdir1 == q_dir
        mask2 = pdir2 == q_dir
        if mask1.sum() < M + 2 or mask2.sum() < M + 2:
            continue

        sq1 = np.zeros(int(mask1.sum()))
        sq2 = np.zeros(int(mask2.sum()))
        for f in FEATURE_ORDER:
            sq1 += lambdas[f] * (pranks1[f][mask1] - rank_query[f]) ** 2
            sq2 += lambdas[f] * (pranks2[f][mask2] - rank_query[f]) ** 2
        w1, w2 = np.exp(-sq1), np.exp(-sq2)
        if w1.sum() < 1e-9 or w2.sum() < 1e-9:
            continue

        band1 = weighted_quantile(ptr1[mask1], w1, q_levels_needed)
        band2 = weighted_quantile(ptr2[mask2], w2, q_levels_needed)

        pb1 = pinball(actual_lr_1, band1, Q_LEVELS)
        pb2 = pinball(actual_lr_2, band2, Q_LEVELS)
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


def calibrate_feature(target, t, feat_name, lambdas_fixed: dict, ticker_data, t0,
                      lambda_lo, lambda_hi, lambda_step, q_lp, q_dates, q_dirs, dev, holdout,
                      rank_series_t, q_rank_pivot):
    lam_grid = np.round(np.arange(lambda_lo, lambda_hi + 1e-9, lambda_step), 3)
    rows_by_split = {frac: [] for frac in DEV_SPLIT_FRACS}
    for lam in lam_grid:
        trial = {**lambdas_fixed, feat_name: float(lam)}
        for frac in DEV_SPLIT_FRACS:
            split_i = int(len(dev) * frac)
            calib_o = dev[:split_i]
            if len(calib_o) < MIN_CALIB_ORIGINS:
                continue
            r = evaluate_origins_mixed(target, t, trial, calib_o, q_lp, q_dates, q_dirs,
                                       rank_series_t, q_rank_pivot, ticker_data, MIN_BARS)
            if r is not None:
                rows_by_split[frac].append({"t": float(lam), "score": r["pinball_norm_avg"]})

    candidates = {}
    for frac, rows in rows_by_split.items():
        res = _grid_argmin(rows, key="score")
        if res is not None:
            candidates[frac] = res
    if not candidates:
        print(f"      [{feat_name}] ПРОПУСК: ни один сплит не дал кандидата")
        return {"feature": feat_name, "skipped": True}

    unique_lams = sorted(set(round(c["t_star"], 3) for c in candidates.values()))

    holdout_results = {}
    for lam in sorted(set(unique_lams) | {0.0}):
        trial = {**lambdas_fixed, feat_name: lam}
        r = evaluate_origins_mixed(target, t, trial, holdout, q_lp, q_dates, q_dirs,
                                   rank_series_t, q_rank_pivot, ticker_data, MIN_BARS)
        holdout_results[lam] = r

    valid = {l: r for l, r in holdout_results.items() if r is not None}
    if not valid or 0.0 not in valid:
        print(f"      [{feat_name}] ПРОПУСК: holdout недоступен")
        return {"feature": feat_name, "skipped": True}

    winner = min(valid, key=lambda l: valid[l]["pinball_norm_avg"])
    baseline = valid[0.0]
    rel = valid[winner]["pinball_norm_avg"] / baseline["pinball_norm_avg"]
    w = valid[winner]
    print(f"      [{feat_name}] λ*={winner:.2f}  pinball_norm={w['pinball_norm_avg']:.4f} "
          f"(baseline={baseline['pinball_norm_avg']:.4f}, rel={rel:.4f})  "
          f"width={w['width_avg']:.4f} (baseline={baseline['width_avg']:.4f})")
    return {
        "feature": feat_name, "skipped": False, "lambda_star": winner,
        "pinball_rel": rel, "pinball_norm": w["pinball_norm_avg"],
        "baseline_pinball_norm": baseline["pinball_norm_avg"],
        "width_avg": w["width_avg"], "baseline_width": baseline["width_avg"],
        "cov_err_avg": w["cov_err_avg"], "baseline_cov_err": baseline["cov_err_avg"],
        "n_holdout": w["n"],
    }


def calibrate_combined_multipass(target, t, ticker_data, t0, lambda_lo=DEFAULT_LAMBDA_LO,
                                 lambda_hi=DEFAULT_LAMBDA_HI, lambda_step=DEFAULT_LAMBDA_STEP,
                                 max_passes=DEFAULT_MAX_PASSES, tol=DEFAULT_CONVERGENCE_TOL):
    """Многопроходный координатный спуск по FEATURE_ORDER — повторяет
    проход, пока max|Δλ| между проходами ≤ tol или не исчерпан max_passes."""
    lh_t, ll_t, dates_t, rank_dict_t = ticker_data[target]
    q_lp, q_dates, q_dirs = build_zigzag(lh_t, ll_t, dates_t, t, MIN_BARS)
    print(f"{target} T={t*100:.0f}%: {len(q_lp)} пивотов")

    q_rank_pivot = {"leg_age": causal_pivot_percentile_rank(leg_age_raw(q_dates, dates_t), K_LEG)}

    min_hist = M + 3
    origins = prepare_origins(q_lp, min_hist)
    holdout_target_n = max(MIN_TEST_ORIGINS, 20)
    if len(origins) <= holdout_target_n:
        print(f"  ПРОПУСК: недостаточно origin'ов ({len(origins)}) для holdout-схемы")
        return {"target": target, "skipped": True}
    holdout_cutoff_date = str(q_dates[origins[-holdout_target_n]])
    dev = [o for o in origins if str(q_dates[o]) < holdout_cutoff_date]
    holdout = [o for o in origins if str(q_dates[o]) >= holdout_cutoff_date]
    print(f"  origin'ов: {len(origins)}  (dev={len(dev)}, holdout={len(holdout)})")

    lambdas = {f: 0.0 for f in FEATURE_ORDER}
    all_passes = []
    for pass_num in range(1, max_passes + 1):
        print(f"  === проход {pass_num}/{max_passes} ===")
        prev_lambdas = dict(lambdas)
        per_feature = []
        for feat_name in FEATURE_ORDER:
            res = calibrate_feature(target, t, feat_name, lambdas, ticker_data, t0,
                                    lambda_lo, lambda_hi, lambda_step, q_lp, q_dates, q_dirs,
                                    dev, holdout, rank_dict_t, q_rank_pivot)
            per_feature.append(res)
            if not res.get("skipped"):
                lambdas[feat_name] = res["lambda_star"]
        all_passes.append({"pass": pass_num, "lambdas": dict(lambdas), "per_feature": per_feature})
        max_delta = max(abs(lambdas[f] - prev_lambdas[f]) for f in FEATURE_ORDER)
        print(f"  проход {pass_num} итог: {lambdas}  (макс. |Δλ|={max_delta:.2f})  [{time.time()-t0:.1f}s]")
        if max_delta <= tol:
            print(f"  сходимость (Δ≤{tol}) достигнута после {pass_num} проход(ов)")
            break
    else:
        print(f"  ⚠ сходимость НЕ достигнута за {max_passes} проход(ов) (Δtol={tol})")

    baseline_zero = {f: 0.0 for f in FEATURE_ORDER}
    final_combo = evaluate_origins_mixed(target, t, lambdas, holdout, q_lp, q_dates, q_dirs,
                                         rank_dict_t, q_rank_pivot, ticker_data, MIN_BARS)
    final_baseline = evaluate_origins_mixed(target, t, baseline_zero, holdout, q_lp, q_dates, q_dirs,
                                            rank_dict_t, q_rank_pivot, ticker_data, MIN_BARS)
    rel_final = (final_combo["pinball_norm_avg"] / final_baseline["pinball_norm_avg"]
                if final_combo and final_baseline else None)

    print(f"  ИТОГ ({len(all_passes)} проход(ов)) {lambdas}:")
    if final_combo and final_baseline:
        print(f"    pinball_norm={final_combo['pinball_norm_avg']:.4f} "
              f"(baseline={final_baseline['pinball_norm_avg']:.4f}, rel={rel_final:.4f})  "
              f"width={final_combo['width_avg']:.4f} (baseline={final_baseline['width_avg']:.4f})  "
              f"cov_err={final_combo['cov_err_avg']:.4f} (baseline={final_baseline['cov_err_avg']:.4f})")
    print(f"  [{time.time()-t0:.1f}s]\n")

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


def mask_ticker_data(ticker_data: dict, cutoff_date: str) -> dict:
    """Обрезает КАЖДОГО тикера (включая целевой, если он там есть) по дате
    <= cutoff_date — rank_dict обрезается той же маской. Причинно корректно:
    rank[i] зависит только от данных ДО i (см. to_percentile_rank/
    compute_density_rank_series), поэтому обрезка ПОСЛЕ i не меняет
    значение в i — используется и в live-прогнозе (обрезка целевого тикера
    под origin_offset), и в causality_check_band_lambda_ref.py."""
    out = {}
    for tk, (lh, ll, dates, rank_dict) in ticker_data.items():
        mask = dates <= cutoff_date
        if mask.sum() < 20:
            continue
        out[tk] = (lh[mask], ll[mask], dates[mask], {f: v[mask] for f, v in rank_dict.items()})
    return out


def load_universe_ticker_data(interval: str = "1d") -> dict:
    """Загружает UNIVERSE + считает все bar-нативные признаки один раз.
    Возвращает {ticker: (lh, ll, dates, {bar_feat: rank_series})}."""
    ticker_data = {}
    for tk in UNIVERSE:
        loaded = load_ticker_candles_with_volume(tk, interval)
        if loaded is None:
            continue
        lh, ll, dates, vol = loaded
        rank_dict = {
            "volume": compute_density_rank_series(lh, ll, vol),
            "trend": to_percentile_rank(compute_trend_raw(lh, ll)),
            "velocity": to_percentile_rank(compute_velocity_raw(lh, ll)),
            "acceleration": to_percentile_rank(compute_acceleration_raw(lh, ll)),
            "volatility": to_percentile_rank(compute_volatility_raw(lh, ll)),
        }
        ticker_data[tk] = (lh, ll, dates, rank_dict)
    return ticker_data


def main():
    parser = argparse.ArgumentParser(
        description="Многопроходный калибратор λ (6 признаков) для S-map полосы, "
                     "objective=pinball_norm_avg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("ticker", nargs="?", help="Целевой тикер (напр. SBER)")
    parser.add_argument("--all", action="store_true", help="Прогнать все 7 целевых тикеров")
    parser.add_argument("--interval", default="1d", metavar="IV")
    parser.add_argument("--t", type=float, default=0.20, metavar="T",
                        help="T_query=T_pool, доля — выбор пользователя, не калибруется")
    parser.add_argument("--lambda-lo", type=float, default=DEFAULT_LAMBDA_LO, metavar="L")
    parser.add_argument("--lambda-hi", type=float, default=DEFAULT_LAMBDA_HI, metavar="L")
    parser.add_argument("--lambda-step", type=float, default=DEFAULT_LAMBDA_STEP, metavar="L")
    parser.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES, metavar="N")
    parser.add_argument("--tol", type=float, default=DEFAULT_CONVERGENCE_TOL, metavar="X",
                        help="Порог сходимости: макс.|Δλ| между проходами")
    args = parser.parse_args()
    if not args.all and not args.ticker:
        parser.error("укажите тикер или --all")
    targets = TARGETS if args.all else [args.ticker]

    t0 = time.time()
    print(f"Загрузка кросс-тикерного пула + признаков ({len(UNIVERSE)} тикеров)…")
    ticker_data = load_universe_ticker_data(args.interval)
    print(f"загружено {len(ticker_data)}/{len(UNIVERSE)} тикеров  [{time.time()-t0:.1f}s]\n")

    results = []
    for target in targets:
        print(f"--- {target} (T={args.t*100:.0f}%, λ∈[{args.lambda_lo},{args.lambda_hi}] "
              f"шаг={args.lambda_step}, max_passes={args.max_passes}, tol={args.tol}) ---")
        results.append(calibrate_combined_multipass(target, args.t, ticker_data, t0,
                                                     args.lambda_lo, args.lambda_hi, args.lambda_step,
                                                     args.max_passes, args.tol))

    ok = [r for r in results if not r.get("skipped")]
    if ok:
        rows = []
        for r in ok:
            rows.append({
                "target": r["target"], "n_passes": r["n_passes"],
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
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

    skipped = [r["target"] for r in results if r.get("skipped")]
    if skipped:
        print(f"Пропущены (недостаточно данных): {skipped}")
    print(f"\nВсего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
