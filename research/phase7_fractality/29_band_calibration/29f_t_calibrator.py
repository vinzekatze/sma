#!/usr/bin/env python3
"""
29f_t_calibrator.py — калибратор единого T (=T_query=T_pool, как в app9.py)
для S-map полосы, отдельно на каждый целевой тикер. ЭТО ЦЕЛЕВОЙ АРТЕФАКТ
исследования (не разовая диагностика) — впоследствии переезжает в основное
приложение для калибровки инструмента под тикер. Держать в актуальном
состоянии при каждой правке.

История версий (коротко, полная — в git/памяти проекта):
  v1: golden-section по T, объектив — отклонение покрытия от номинала
      (масштабно-независимо; сырой pinball между разными T сравнивать
      нельзя — при большем T плечи зигзага физически крупнее).
  v2: golden-section → сетка + скользящее сглаживание (кривая cov_err(T)
      не унимодальна — MGNT промахнулся мимо видимого оптимума).
  v3: исправлен баг сглаживания на краю сетки (min_periods=1 → =SMOOTH_
      WINDOW) — краевая точка с неполным окном соседей нечестно выигрывала
      argmin (MTSS: T=23% дал test_cov_err=0.138 вместо честных ~0.024).
  v4 (эта версия): один calib/test сплит оказался ненадёжен на PLZL — два
      близких T (21.5% и 22%) дали разный test_err просто из-за того, какие
      origin'ы попали в маленький (n≈26-28) test-набор. Добавлен ЧЕСТНЫЙ
      HOLDOUT (последние HOLDOUT_FRAC истории, календарно — НИКОГДА не
      участвует в подборе T) + НЕСКОЛЬКО внутренних calib-сплитов (dev-
      часть истории режется на DEV_SPLIT_FRACS долей) — каждый сплит даёт
      своего кандидата T*, holdout проверяет ВСЕХ кандидатов один раз и
      выбирает победителя. Расхождение кандидатов между собой — само по
      себе диагностика устойчивости калибровки на этом тикере.
  v5 (эта версия): T сам по себе не защищает от «мельчания» — на мелком T
      пивоты могут подтверждаться каждые 1-2 бара, а трейдер физически не
      успевает среагировать (SBER T=13%: 21% промежутков ≤2 бара). Добавлен
      MIN_BARS=5 — фиксированная КОНСТАНТА (не калибруется, по решению
      пользователя: событие мельче ~недели на 1d бессмысленно), Depth-аналог
      в build_zigzag/build_causal_pool — полностью устраняет промежутки
      ≤2 бара, T пересчитывается заново с этим ограничением.

Использование:
    python 29f_t_calibrator.py SBER --interval 1d
    python 29f_t_calibrator.py --all   # все 7 целевых тикеров подряд
"""
import argparse
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
    load_ticker_candles, build_zigzag, build_causal_pool, build_query_vector,
    weighted_quantile, UNIVERSE,
)

spec = importlib.util.spec_from_file_location("calib29", HERE / "29_smap_band_calibrator.py")
calib29 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib29)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TARGETS = ["SBER", "LKOH", "GAZP", "MGNT", "CHMF", "MTSS", "PLZL"]
# T_HI сужен с 0.35 до 0.24: ни один тикер ни в одном прогоне 29-серии не
# выбрал T выше ~22%, а на T=30-35% у SBER на ВСЮ 19-летнюю историю всего
# 14-30 пивотов — заведомо недостаточно для holdout-схемы (нужны и calib,
# и honest test одновременно). Расширять можно, но осознанно.
T_LO, T_HI, T_TOL = 0.12, 0.24, 0.005
MIN_BARS = 5    # фиксированная КОНСТАНТА (не калибруется, по решению пользователя
                # 2026-07-08): минимум баров между пивотами (Depth-аналог), реагировать
                # на события мельче ~недели (5 баров на 1d) практического смысла нет
COVERAGE_LEVELS = [0.50, 0.75, 0.90]
MIN_CALIB_ORIGINS = 25    # защита от шумного argmin на малой выборке
MIN_TEST_ORIGINS  = 15
HOLDOUT_FRAC = 0.20        # ориентир, НЕ используется напрямую — см. holdout_cutoff_date
DEV_SPLIT_FRACS = [0.55, 0.65, 0.75]   # 3 разных calib-среза внутри dev-части → 3 кандидата T*


def coverage_at(actual: float, band: dict, level: float) -> int:
    q_lo, q_hi = (1 - level) / 2, (1 + level) / 2
    return int(band[q_lo] <= actual <= band[q_hi])


def evaluate_origins_at_t(target: str, t: float, origins: list, q_lp, q_dates, q_dirs,
                          ticker_arrays: dict) -> dict | None:
    """Честный расчёт (без калибровки внутри) на заданном списке origin'ов
    и заданном T. Возвращает сводку (pinball_norm, cov_err_avg, детали
    покрытия) или None, если origin'ов не хватает."""
    if len(origins) < 1:
        return None
    q_levels_needed = sorted(set(calib29.Q_LEVELS) |
                             {(1 - lv) / 2 for lv in COVERAGE_LEVELS} |
                             {(1 + lv) / 2 for lv in COVERAGE_LEVELS})
    rows_h1, rows_h2 = [], []
    for origin in origins:
        q_lp_prefix = q_lp[: origin + 1]
        qv0 = build_query_vector(q_lp_prefix, calib29.M)
        q_dir = int(q_dirs[origin])
        origin_date = str(q_dates[origin])

        actual_lr_1 = float(q_lp[origin + 1] - q_lp[origin])
        actual_lr_2 = float(q_lp[origin + 2] - q_lp[origin])   # кумулятивно, как в app9 (horizon=2)

        pfm1, ptr1, pdir1 = build_causal_pool(origin_date, ticker_arrays, t, calib29.M, horizon=1, min_bars=MIN_BARS)
        pfm2, ptr2, pdir2 = build_causal_pool(origin_date, ticker_arrays, t, calib29.M, horizon=2, min_bars=MIN_BARS)
        mask1 = pdir1 == q_dir
        mask2 = pdir2 == q_dir
        if mask1.sum() < calib29.M + 2 or mask2.sum() < calib29.M + 2:
            continue

        band1 = weighted_quantile(ptr1[mask1], np.ones(mask1.sum()), q_levels_needed)
        band2 = weighted_quantile(ptr2[mask2], np.ones(mask2.sum()), q_levels_needed)

        pb1 = calib29.pinball(actual_lr_1, band1, calib29.Q_LEVELS)
        pb2 = calib29.pinball(actual_lr_2, band2, calib29.Q_LEVELS)
        rows_h1.append({"pinball": pb1, **{f"cov{int(lv*100)}": coverage_at(actual_lr_1, band1, lv)
                                           for lv in COVERAGE_LEVELS}})
        rows_h2.append({"pinball": pb2, **{f"cov{int(lv*100)}": coverage_at(actual_lr_2, band2, lv)
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
        "cov50_h1": df1.cov50.mean(), "cov50_h2": df2.cov50.mean(),
        "cov75_h1": df1.cov75.mean(), "cov75_h2": df2.cov75.mean(),
        "cov90_h1": df1.cov90.mean(), "cov90_h2": df2.cov90.mean(),
    }


T_GRID_STEP = 0.01   # сетка вместо golden-section — кривая cov_err(T) не гарантированно
                     # унимодальна (см. MGNT: golden-section промахнулся мимо видимого на
                     # сетке оптимума из-за локального шума при T=20-22%)
SMOOTH_WINDOW = 3    # скользящее среднее по 3 соседним точкам сетки — гасит точечный шум
                     # ПЕРЕД argmin, чтобы не выбирать T просто по случайному провалу


def _grid_argmin(rows: list, key: str = "cov_err") -> dict | None:
    """Сетка → сглаживание (полное окно, без нечестного края) → argmin.
    `key` — имя поля-объектива в rows (по умолчанию "cov_err" для обратной
    совместимости с T-калибровкой; 29g/29j передают key="score", заполненное
    pinball_norm_avg — см. их докстринги про исправление objective 2026-07-08).
    Возвращает {"t_star","calib_score"} или None, если данных не хватает."""
    if len(rows) < SMOOTH_WINDOW:
        return None
    gdf = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    gdf["smooth"] = gdf[key].rolling(
        SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW).mean()
    valid = gdf.dropna(subset=["smooth"])
    if valid.empty:
        return None
    best = valid.loc[valid["smooth"].idxmin()]
    return {"t_star": float(best["t"]), "calib_score": float(best["smooth"]), "grid": gdf}


def calibrate_t(target: str, ticker_arrays: dict) -> dict:
    """Многосплитовая калибровка с честным holdout (v4, см. docstring файла):
      1. holdout_cutoff_date — фиксированная календарная граница (последние
         HOLDOUT_FRAC истории), ОДНА и та же для ЛЮБОГО кандидата T — иначе
         сравнение между T на holdout было бы нечестным (разный T даёт
         разные по времени пивоты, значит "последние 20% origin'ов" не
         совпадали бы по датам между кандидатами).
      2. Для каждого T из сетки: origin'ы делятся на dev (< cutoff) и holdout
         (>= cutoff) по ДАТЕ. Holdout НИКОГДА не используется для выбора T.
      3. Внутри dev — DEV_SPLIT_FRACS разных calib-срезов → 3 независимых
         кандидата T* (сетка+сглаживание на каждом срезе отдельно).
      4. Все уникальные кандидаты проверяются на holdout РОВНО ОДИН РАЗ —
         побеждает лучший по cov_err_avg. Расхождение кандидатов между собой
         — диагностика устойчивости (сильно разошлись → калибровка на этом
         тикере ненадёжна, см. SBER/дрейф режима)."""
    log_highs, log_lows, dates = ticker_arrays[target]

    # holdout_cutoff_date — НЕ доля сырых баров (SBER: 80% баров оставляло
    # последние ~3.5 года, а в них почти не было движений ≥18% — holdout
    # получался пустым на половине сетки). Вместо этого — гарантия
    # минимального числа holdout-origin'ов у САМОГО РАЗРЕЖЁННОГО кандидата
    # (T_HI, наименьшая плотность пивотов). Более низкие T гуще пивотами →
    # автоматически получают ≥ столько же holdout-origin'ов в том же окне.
    q_lp_hi, q_dates_hi, _ = build_zigzag(log_highs, log_lows, dates, T_HI, MIN_BARS)
    origins_hi = calib29.prepare_origins(q_lp_hi, calib29.M + 3)
    holdout_target_n = max(MIN_TEST_ORIGINS, 20)
    if len(origins_hi) <= holdout_target_n:
        raise RuntimeError(f"{target}: на T_HI={T_HI*100:.0f}% всего {len(origins_hi)} origin'ов "
                           f"на всю историю — недостаточно для holdout-схемы, снизьте T_HI")
    holdout_cutoff_date = str(q_dates_hi[origins_hi[-holdout_target_n]])

    def build_dev_holdout_at_t(t):
        q_lp, q_dates, q_dirs = build_zigzag(log_highs, log_lows, dates, t, MIN_BARS)
        min_hist = calib29.M + 3
        origins = calib29.prepare_origins(q_lp, min_hist)
        dev = [o for o in origins if str(q_dates[o]) < holdout_cutoff_date]
        holdout = [o for o in origins if str(q_dates[o]) >= holdout_cutoff_date]
        return q_lp, q_dates, q_dirs, dev, holdout

    t_grid = np.round(np.arange(T_LO, T_HI + 1e-9, T_GRID_STEP), 4)

    # ── таблица (T, split_frac) → cov_err на calib-подмножестве dev ──
    rows_by_split = {frac: [] for frac in DEV_SPLIT_FRACS}
    for t in t_grid:
        q_lp, q_dates, q_dirs, dev, holdout = build_dev_holdout_at_t(float(t))
        if len(holdout) < MIN_TEST_ORIGINS:
            continue   # этот T даёт слишком мало holdout-origin'ов — исключаем сразу
        for frac in DEV_SPLIT_FRACS:
            split_i = int(len(dev) * frac)
            calib_o = dev[:split_i]
            if len(calib_o) < MIN_CALIB_ORIGINS:
                continue
            r = evaluate_origins_at_t(target, float(t), calib_o, q_lp, q_dates, q_dirs, ticker_arrays)
            if r is not None:
                rows_by_split[frac].append({"t": float(t), "cov_err": r["cov_err_avg"]})

    candidates = {}
    for frac, rows in rows_by_split.items():
        res = _grid_argmin(rows)
        if res is not None:
            candidates[frac] = res
    if not candidates:
        raise RuntimeError(f"{target}: ни один из {len(DEV_SPLIT_FRACS)} сплитов не дал кандидата — "
                           f"недостаточно истории для этой схемы")

    unique_t_stars = sorted(set(round(c["t_star"], 4) for c in candidates.values()))

    # ── честная проверка КАЖДОГО уникального кандидата на holdout — один раз ──
    holdout_by_t = {}
    for t_star in unique_t_stars:
        q_lp, q_dates, q_dirs, dev, holdout = build_dev_holdout_at_t(t_star)
        holdout_by_t[t_star] = evaluate_origins_at_t(target, t_star, holdout, q_lp, q_dates, q_dirs, ticker_arrays)

    valid_holdout = {t: r for t, r in holdout_by_t.items() if r is not None}
    if not valid_holdout:
        raise RuntimeError(f"{target}: ни один кандидат не прошёл holdout-проверку")
    winner_t = min(valid_holdout, key=lambda t: valid_holdout[t]["cov_err_avg"])

    return {
        "target": target,
        "candidates": candidates,              # {split_frac: {t_star, calib_score, grid}}
        "holdout_by_t": valid_holdout,          # {t_star: eval_dict}
        "winner_t_pct": round(winner_t * 100, 2),
        "winner_holdout": valid_holdout[winner_t],
        "agreement_pct": round((max(unique_t_stars) - min(unique_t_stars)) * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Калибратор единого T для S-map полосы (per-ticker)")
    parser.add_argument("ticker", nargs="?", help="Целевой тикер (напр. SBER)")
    parser.add_argument("--all", action="store_true", help="Прогнать все 7 целевых тикеров")
    parser.add_argument("--interval", default="1d", metavar="IV")
    args = parser.parse_args()

    if not args.all and not args.ticker:
        parser.error("укажите тикер или --all")

    t0 = time.time()
    print(f"Загрузка кросс-тикерного пула ({len(UNIVERSE)} тикеров)…")
    ticker_arrays = {}
    for tk in UNIVERSE:
        loaded = load_ticker_candles(tk, args.interval)
        if loaded is not None:
            ticker_arrays[tk] = loaded
    print(f"загружено {len(ticker_arrays)}/{len(UNIVERSE)} тикеров\n")

    targets = TARGETS if args.all else [args.ticker]
    results = []
    for target in targets:
        print(f"--- {target} ---")
        r = calibrate_t(target, ticker_arrays)
        results.append(r)

        print(f"  Кандидаты по сплитам (dev-доля={DEV_SPLIT_FRACS}):")
        for frac, c in sorted(r["candidates"].items()):
            print(f"    split={frac:.2f}: T*={c['t_star']*100:.2f}%  calib_score={c['calib_score']:.4f}")
        print(f"  Разброс кандидатов: {r['agreement_pct']:.2f} п.п.")

        print(f"  Holdout-проверка каждого уникального кандидата:")
        for t_star, res in sorted(r["holdout_by_t"].items()):
            mark = " <- ПОБЕДИТЕЛЬ" if round(t_star * 100, 2) == r["winner_t_pct"] else ""
            print(f"    T={t_star*100:.2f}%: cov_err_avg={res['cov_err_avg']:.4f}  n={res['n']}{mark}")

        w = r["winner_holdout"]
        print(f"  ИТОГ: T*={r['winner_t_pct']:.2f}%  holdout cov_err_avg={w['cov_err_avg']:.4f}  "
              f"pinball_norm={w['pinball_norm_avg']:.4f}  n_holdout={w['n']}")
        print(f"    cov50=({w['cov50_h1']:.2f},{w['cov50_h2']:.2f})  "
              f"cov75=({w['cov75_h1']:.2f},{w['cov75_h2']:.2f})  "
              f"cov90=({w['cov90_h1']:.2f},{w['cov90_h2']:.2f})")
        for frac, c in r["candidates"].items():
            c["grid"].to_csv(RESULTS / f"29f_grid_{target}_split{int(frac*100)}.csv", index=False)
        print(f"  [{time.time()-t0:.1f}s]\n")

    rows = []
    for r in results:
        w = r["winner_holdout"]
        rows.append({
            "target": r["target"], "T*_%": r["winner_t_pct"],
            "candidates_spread_pp": r["agreement_pct"], "n_candidates": len(r["holdout_by_t"]),
            "holdout_cov_err": w["cov_err_avg"], "holdout_pinball_norm": w["pinball_norm_avg"],
            "n_holdout": w["n"],
        })
    df = pd.DataFrame(rows)
    out_path = RESULTS / "29f_t_calibrator.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nСохранено: {out_path}")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
