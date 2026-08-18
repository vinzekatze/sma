#!/usr/bin/env python3
"""
29g_lambda_volume_calibrator.py — калибратор λ (объёмный вес пула) для
S-map полосы, отдельно на каждый целевой тикер. ЭТО ЦЕЛЕВОЙ АРТЕФАКТ
исследования (как 29f_t_calibrator.py) — впоследствии переезжает в основное
приложение вместе с калибратором T. Держать в актуальном состоянии.

Концептуальная позиция (зафиксирована пользователем 2026-07-08, по итогам
работы над 29f): T — это ВЫБОР ПОЛЬЗОВАТЕЛЯ под свою торговую задачу
(масштаб движения, время реакции), НЕ статистический оптимум — калибровка
T под «удобную для статистики» точность систематически выбирала мелкий T,
доминируемый лёгкими, но бесполезными для торговли боковыми колебаниями
(см. память project-phase7-uncertainty-field-pivot). Поэтому этот
калибратор принимает T КАК ВХОДНОЙ ПАРАМЕТР (--t) и подбирает ДОПОЛНИТЕЛЬНЫЙ
параметр (λ) под уже выбранный пользователем T — не ищет T заново.

Концепция объёмного веса (по прецеденту эксп.20b/20e фазы 7, точечный
пайплайн): крупные зигзаг-пивоты приземляются в зонах НИЗКОЙ объёмной
плотности (p<0.0001, структурный факт, воспроизведён на 7 тикерах). Вес —
не фильтр и не постфактум-коррекция (оба паттерна систематически
проваливались в фазе 7), а МУЛЬТИПЛИКАТИВНЫЙ фактор ВНУТРИ кернела: события
пула с похожим на запрос объёмным контекстом весят больше.

  rank(бар) — перцентильный ранг цены бара в объёмном профиле window=252
              баров ДО него (причинно, compute_density_rank_series).
  w_vol = exp(-λ · (rank_event − rank_query)²)  — вес пула

λ ищется той же честной схемой, что T в 29f_t_calibrator.py (сетка +
сглаживание по полному окну, несколько dev-сплитов + отдельный honest
holdout) — переиспользует calib29/29f инфраструктуру напрямую. λ=0
(отсутствие объёмного веса, равносильно базовой линии) — ЗАКОННЫЙ исход
калибровки, не брак: побеждает, если ни один кандидат λ>0 не бьёт baseline
на honest holdout (см. SBER в результатах — калибратор корректно выбрал
λ=0, эффекта там нет, и это не нужно «чинить» подгонкой).

θ=0, m=6, min_bars=5, T_pool=T_query=T — зафиксированы (см. 29f/29_smap_
band_calibrator.py). Единственный дополнительный калибруемый параметр
здесь — λ.

⚠️ ИСПРАВЛЕНИЕ OBJECTIVE (2026-07-08, после вопроса пользователя "поля становятся
точнее или просто размываются?"): первый прогон отбирал λ по cov_err_avg (среднее
|покрытие−номинал| по 6 уровням h1/h2×{50,75,90}%). Прямая проверка на GAZP
(λ*=11 из первого прогона) показала: "улучшение" cov_err_avg целиком объяснялось
ОДНИМ из 6 покрытий (cov50_h2: 0.40→0.45, остальные 5 не менялись ВООБЩЕ), а
ПОЛОСА при этом стала ШИРЕ (+2.2%…+3.5% по всем 4 замерам ширины), не уже —
ровно сценарий размазывания из [[feedback-uncertainty-field-width-control]].
Причина: cov_err_avg не штрафует ширину — расширение полосы механически чинит
недопокрытие без всякой связи с реальной точностью. objective был скопирован из
29f (там cov_err_avg обоснован — T меняет физический масштаб плеч, сырой pinball
между разными T не сравним), но здесь T ФИКСИРОВАН при переборе λ — pinball_norm_avg
(pinball/T, уже считался в evaluate_origins, просто не использовался для отбора)
напрямую сравним между λ и структурно (как proper scoring rule) штрафует лишнюю
ширину. С этой версии objective — pinball_norm_avg; cov_err_avg и явная ширина
полосы (width_avg = band[0.9]−band[0.1]) остаются в выводе как диагностика.
Старые числа (GAZP λ*=11 −60%, CHMF λ*=1 −13% и т.д.) недействительны, требуют
пересчёта с новым objective.

Использование:
    python 29g_lambda_volume_calibrator.py SBER --t 0.20
    python 29g_lambda_volume_calibrator.py --all --t 0.20
"""
import json
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REF_DIR = HERE.parents[1] / "reference"
sys.path.insert(0, str(REF_DIR))
from smap_band_ref import (
    build_zigzag, build_pool_vectors, build_query_vector, weighted_quantile,
    UNIVERSE, DATA_DIR,
)

# ── функции этого эксперимента (объём) — НАМЕРЕННО не в smap_band_ref.py:
#    реферный модуль должен оставаться чистым и содержать только проверенную,
#    финальную концепцию, не экспериментальные добавки, которые могут не дать
#    результата (по прямому указанию пользователя, 2026-07-08) ──

def load_ticker_candles_with_volume(ticker: str, interval: str, data_dir: Path = DATA_DIR):
    """Как smap_band_ref.load_ticker_candles, но дополнительно возвращает
    объём. Возвращает (log_highs, log_lows, dates, volumes) или None."""
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


def compute_density_rank_series(log_highs: np.ndarray, log_lows: np.ndarray,
                                volumes: np.ndarray, window: int = 252,
                                n_bins: int = 40) -> np.ndarray:
    """Причинный объёмный профиль: для КАЖДОГО бара i (>= window) строит
    гистограмму объёма по ценам ПРЕДЫДУЩИХ window баров (i САМ не входит —
    строго причинно), затем находит перцентильный ранг цены бара i (мидпоинт
    (log(high)+log(low))/2) в этом профиле.

    rank≈0 — редкая/разрежённая зона по объёму, rank≈1 — плотная/бойкая
    зона. Совпадает по идее с эксп.20b/20e фазы 7 (точечный пайплайн): крупные
    зигзаг-пивоты приземляются в зонах НИЗКОЙ объёмной плотности, p<0.0001.

    Возвращает массив той же длины, что log_highs — NaN там, где window
    ещё не набрано.
    """
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

spec29 = importlib.util.spec_from_file_location("calib29", HERE / "29_smap_band_calibrator.py")
calib29 = importlib.util.module_from_spec(spec29)
spec29.loader.exec_module(calib29)

spec29f = importlib.util.spec_from_file_location("calib29f", HERE / "29f_t_calibrator.py")
calib29f = importlib.util.module_from_spec(spec29f)
spec29f.loader.exec_module(calib29f)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TARGETS = calib29f.TARGETS   # SBER LKOH GAZP MGNT CHMF MTSS PLZL
DEFAULT_T = 0.20        # дефолт для --t, НЕ жёсткая константа — T это выбор
                        # пользователя (см. docstring), передаётся явным
                        # параметром через все функции ниже, не читается из
                        # глобальной переменной
M = calib29.M            # 6
MIN_BARS = calib29f.MIN_BARS   # 5
RANK_WINDOW = 252
N_BINS = 40
# Дефолты сетки λ — НАСТОЯЩИЕ ПАРАМЕТРЫ (--lambda-lo/--lambda-hi/--lambda-step
# в CLI), не глобальные константы, которые надо править в коде. Шаг 1.0 —
# для быстрой сессионной разведки нескольких признаков сразу (по решению
# пользователя 2026-07-08: ищем не один "лучший" признак, а выгодный НАБОР —
# грубого шага достаточно, чтобы сравнить кандидатов). Для точечной
# калибровки под конкретный тикер (в первую очередь — в основном приложении)
# уменьшать шаг через CLI, не редактируя файл.
DEFAULT_LAMBDA_LO, DEFAULT_LAMBDA_HI, DEFAULT_LAMBDA_STEP = 0.0, 20.0, 1.0
COVERAGE_LEVELS = calib29f.COVERAGE_LEVELS
MIN_CALIB_ORIGINS = calib29f.MIN_CALIB_ORIGINS
MIN_TEST_ORIGINS = calib29f.MIN_TEST_ORIGINS
# ШИРЕ, чем 29f's [0.55,0.65,0.75]: dev-объём на T=20% для ОДНОГО тикера
# сильно варьируется по тикерам (SBER dev≈36, другие могут быть заметно
# больше) — держим и низкие, и высокие доли, чтобы для любого тикера
# хватало хотя бы 2-3 рабочих сплитов, не подгоняя вручную под каждый тикер
DEV_SPLIT_FRACS = [0.55, 0.65, 0.75, 0.85]
SMOOTH_WINDOW = calib29f.SMOOTH_WINDOW


def build_pool_vectors_with_rank(pivot_prices, pivot_dirs, pivot_ranks, embedding_dim, horizon):
    n = len(pivot_prices)
    valid = np.arange(embedding_dim, n - horizon)
    if len(valid) == 0:
        return (np.zeros((0, embedding_dim)), np.zeros(0), np.zeros(0, dtype=np.int8), np.zeros(0))
    lag_idx = valid[:, None] - np.arange(embedding_dim)[None, :]
    feats = pivot_prices[lag_idx] - pivot_prices[lag_idx - 1]
    tars = pivot_prices[valid + horizon] - pivot_prices[valid]
    dirs = pivot_dirs[valid]
    ranks = pivot_ranks[valid]     # ранг ПРИВЯЗАН к пивоту-началу плеча (j), не к цели
    finite = np.all(np.isfinite(feats), axis=1) & np.isfinite(tars) & np.isfinite(ranks)
    return feats[finite], tars[finite], dirs[finite], ranks[finite]


def build_causal_pool_with_rank(cutoff_date, ticker_data, t_pool, embedding_dim, horizon, min_bars):
    """ticker_data: {ticker: (log_highs, log_lows, dates, rank_series)}"""
    feats_l, tars_l, dirs_l, ranks_l = [], [], [], []
    for ticker, (lh, ll, dates, rank_series) in ticker_data.items():
        mask = dates <= cutoff_date
        if mask.sum() < embedding_dim + horizon + 2:
            continue
        lh_c, ll_c, dt_c = lh[mask], ll[mask], dates[mask]
        rank_c = rank_series[mask]
        pivot_prices, confirm_dates, pivot_dirs = build_zigzag(lh_c, ll_c, dt_c, t_pool, min_bars)
        if len(pivot_prices) == 0:
            continue
        bar_idx = np.searchsorted(dt_c, confirm_dates)
        pivot_ranks = rank_c[bar_idx]
        feats, tars, dirs, ranks = build_pool_vectors_with_rank(
            pivot_prices, pivot_dirs, pivot_ranks, embedding_dim, horizon)
        if len(tars):
            feats_l.append(feats); tars_l.append(tars); dirs_l.append(dirs); ranks_l.append(ranks)
    if not tars_l:
        return (np.zeros((0, embedding_dim)), np.zeros(0), np.zeros(0, dtype=np.int8), np.zeros(0))
    return (np.vstack(feats_l), np.concatenate(tars_l), np.concatenate(dirs_l), np.concatenate(ranks_l))


def coverage_at(actual, band, level):
    q_lo, q_hi = (1 - level) / 2, (1 + level) / 2
    return int(band[q_lo] <= actual <= band[q_hi])


def evaluate_origins(target, t, lam, origins, q_lp, q_dates, q_dirs, rank_series_target, ticker_data):
    q_levels_needed = sorted(set(calib29.Q_LEVELS) |
                             {(1 - lv) / 2 for lv in COVERAGE_LEVELS} |
                             {(1 + lv) / 2 for lv in COVERAGE_LEVELS})
    rows_h1, rows_h2 = [], []
    for origin in origins:
        q_lp_prefix = q_lp[: origin + 1]
        q_dir = int(q_dirs[origin])
        origin_date = str(q_dates[origin])
        origin_bar = int(np.searchsorted(ticker_data[target][2], origin_date))
        origin_bar = min(origin_bar, len(rank_series_target) - 1)
        rank_query = rank_series_target[origin_bar]
        if not np.isfinite(rank_query):
            continue

        actual_lr_1 = float(q_lp[origin + 1] - q_lp[origin])
        actual_lr_2 = float(q_lp[origin + 2] - q_lp[origin])

        pfm1, ptr1, pdir1, prank1 = build_causal_pool_with_rank(
            origin_date, ticker_data, t, M, 1, MIN_BARS)
        pfm2, ptr2, pdir2, prank2 = build_causal_pool_with_rank(
            origin_date, ticker_data, t, M, 2, MIN_BARS)
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


def calibrate_lambda(target: str, t: float, ticker_data: dict, t0: float,
                     lambda_lo: float = DEFAULT_LAMBDA_LO, lambda_hi: float = DEFAULT_LAMBDA_HI,
                     lambda_step: float = DEFAULT_LAMBDA_STEP) -> dict:
    lh_t, ll_t, dates_t, rank_series_t = ticker_data[target]
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

    lam_grid = np.round(np.arange(lambda_lo, lambda_hi + 1e-9, lambda_step), 3)

    rows_by_split = {frac: [] for frac in DEV_SPLIT_FRACS}
    for lam in lam_grid:
        for frac in DEV_SPLIT_FRACS:
            split_i = int(len(dev) * frac)
            calib_o = dev[:split_i]
            if len(calib_o) < MIN_CALIB_ORIGINS:
                continue
            r = evaluate_origins(target, t, float(lam), calib_o, q_lp, q_dates, q_dirs, rank_series_t, ticker_data)
            if r is not None:
                rows_by_split[frac].append({"t": float(lam), "score": r["pinball_norm_avg"]})

    candidates = {}
    for frac, rows in rows_by_split.items():
        res = calib29f._grid_argmin(rows, key="score")
        if res is not None:
            candidates[frac] = res

    if not candidates:
        print(f"  ПРОПУСК: ни один сплит не дал кандидата λ  [{time.time()-t0:.1f}s]")
        return {"target": target, "skipped": True}

    print("  Кандидаты λ по сплитам:")
    for frac, c in sorted(candidates.items()):
        print(f"    split={frac:.2f}: λ*={c['t_star']:.2f}  calib_score={c['calib_score']:.4f}")

    unique_lams = sorted(set(round(c["t_star"], 3) for c in candidates.values()))
    spread = round(max(unique_lams) - min(unique_lams), 3)
    holdout_results = {}
    for lam in sorted(set(unique_lams) | {0.0}):
        r = evaluate_origins(target, t, lam, holdout, q_lp, q_dates, q_dirs, rank_series_t, ticker_data)
        holdout_results[lam] = r

    valid = {l: r for l, r in holdout_results.items() if r is not None}
    if not valid:
        print(f"  ПРОПУСК: ни один кандидат не прошёл holdout  [{time.time()-t0:.1f}s]")
        return {"target": target, "skipped": True}

    winner = min(valid, key=lambda l: valid[l]["pinball_norm_avg"])
    baseline = valid.get(0.0)
    rel = valid[winner]["pinball_norm_avg"] / baseline["pinball_norm_avg"] if baseline else None
    w = valid[winner]
    if baseline:
        print(f"  Holdout: λ=0 baseline pinball_norm={baseline['pinball_norm_avg']:.4f}  "
              f"cov_err={baseline['cov_err_avg']:.4f}  width={baseline['width_avg']:.4f}  (n={baseline['n']})")
    else:
        print("  Holdout: λ=0 baseline недоступен")
    winner_line = (f"  Победитель λ={winner:.2f}  pinball_norm={w['pinball_norm_avg']:.4f}  "
                   f"cov_err={w['cov_err_avg']:.4f}  width={w['width_avg']:.4f}")
    if rel is not None:
        winner_line += f"  относительно baseline={rel:.4f}  ({'лучше' if rel < 1 else 'не лучше'})"
    print(winner_line)
    print(f"  [{time.time()-t0:.1f}s]\n")

    return {
        "target": target, "skipped": False, "lambda_star": winner,
        "spread": spread, "n_candidates": len(unique_lams),
        "holdout_pinball_norm": w["pinball_norm_avg"],
        "baseline_pinball_norm": baseline["pinball_norm_avg"] if baseline else None,
        "rel_vs_baseline": rel,
        "holdout_cov_err": w["cov_err_avg"],
        "baseline_cov_err": baseline["cov_err_avg"] if baseline else None,
        "holdout_width": w["width_avg"],
        "baseline_width": baseline["width_avg"] if baseline else None,
        "n_holdout": w["n"],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Калибратор λ (объёмный вес) для S-map полосы, T задаётся пользователем")
    parser.add_argument("ticker", nargs="?", help="Целевой тикер (напр. SBER)")
    parser.add_argument("--all", action="store_true", help="Прогнать все 7 целевых тикеров")
    parser.add_argument("--t", type=float, default=DEFAULT_T, metavar="T",
                        help="T_query=T_pool, доля (напр. 0.20 = 20%%) — ВЫБОР ПОЛЬЗОВАТЕЛЯ, "
                             "не калибруется этим скриптом (см. docstring)")
    parser.add_argument("--lambda-lo", type=float, default=DEFAULT_LAMBDA_LO, metavar="L")
    parser.add_argument("--lambda-hi", type=float, default=DEFAULT_LAMBDA_HI, metavar="L")
    parser.add_argument("--lambda-step", type=float, default=DEFAULT_LAMBDA_STEP, metavar="L",
                        help="Шаг сетки λ — грубый (1.0) для разведки нескольких признаков сразу, "
                             "мельче для точечной калибровки под конкретный тикер")
    args = parser.parse_args()
    if not args.all and not args.ticker:
        parser.error("укажите тикер или --all")
    targets = TARGETS if args.all else [args.ticker]
    t = args.t

    t0 = time.time()
    print(f"Загрузка кросс-тикерного пула с объёмом ({len(UNIVERSE)} тикеров)…")
    ticker_data = {}
    for tk in UNIVERSE:
        loaded = load_ticker_candles_with_volume(tk, "1d")
        if loaded is None:
            continue
        lh, ll, dates, vol = loaded
        rank_series = compute_density_rank_series(lh, ll, vol, RANK_WINDOW, N_BINS)
        ticker_data[tk] = (lh, ll, dates, rank_series)
    print(f"загружено {len(ticker_data)}/{len(UNIVERSE)} тикеров  [{time.time()-t0:.1f}s]\n")

    results = []
    for target in targets:
        print(f"--- {target} (T={t*100:.0f}%, λ∈[{args.lambda_lo},{args.lambda_hi}] шаг={args.lambda_step}) ---")
        results.append(calibrate_lambda(target, t, ticker_data, t0,
                                        args.lambda_lo, args.lambda_hi, args.lambda_step))

    rows = [r for r in results if not r.get("skipped")]
    if rows:
        df = pd.DataFrame(rows)
        out_path = RESULTS / "29g_lambda_calibrator.csv"
        df.to_csv(out_path, index=False)
        print(df.to_string(index=False))
        print(f"\nСохранено: {out_path}")
    skipped = [r["target"] for r in results if r.get("skipped")]
    if skipped:
        print(f"Пропущены (недостаточно данных): {skipped}")

    print(f"\nВсего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
