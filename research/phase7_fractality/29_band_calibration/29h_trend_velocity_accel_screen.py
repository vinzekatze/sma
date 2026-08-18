#!/usr/bin/env python3
"""
29h_trend_velocity_accel_screen.py — быстрый скрининг трёх новых кандидатов
веса пула (трендовый контекст, скорость, ускорение) на 2 тикерах (GAZP,
CHMF — там объёмный вес дал самый надёжный сигнал в 29g) грубой сеткой λ,
ПРЕЖДЕ чем тратить время на полный честный прогон по всем 7 тикерам.

Переиспользует ВСЮ калибровочную инфраструктуру 29g_lambda_volume_
calibrator.py (build_causal_pool_with_rank, evaluate_origins,
calibrate_lambda) — она не знает, что взвешивающий признак это именно
объём, просто ожидает {ticker: (log_highs, log_lows, dates, feature_series)}
с ПРИЧИННО посчитанным feature_series (перцентильный ранг в [0,1] —
единый масштаб для сравнимости λ между фичами).

Признаки (все — перцентильный ранг за window=252 бара, причинно):
  trend        — позиция цены относительно скользящей средней (window_ma=100)
  velocity     — лог-доходность за последние 10 баров ДО текущего
  acceleration — изменение velocity за последние 10 баров (вторая производная)

θ=0, m=6, min_bars=5, T_query=T_pool=0.20 — как в 29g. λ ищется ГРУБОЙ
сеткой (шаг 2.0, диапазон 0-16) на 2 СПЛИТАХ (не 4) — это разведка, не
финальная калибровка; для признаков, показавших сигнал, нужен отдельный
полный прогон по образцу 29g на всех 7 тикерах.

Использование:
    python 29h_trend_velocity_accel_screen.py
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
from smap_band_ref import build_zigzag, UNIVERSE

spec29 = importlib.util.spec_from_file_location("calib29", HERE / "29_smap_band_calibrator.py")
calib29 = importlib.util.module_from_spec(spec29)
spec29.loader.exec_module(calib29)

spec29f = importlib.util.spec_from_file_location("calib29f", HERE / "29f_t_calibrator.py")
calib29f = importlib.util.module_from_spec(spec29f)
spec29f.loader.exec_module(calib29f)

spec29g = importlib.util.spec_from_file_location("calib29g", HERE / "29g_lambda_volume_calibrator.py")
calib29g = importlib.util.module_from_spec(spec29g)
spec29g.loader.exec_module(calib29g)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SCREEN_TICKERS = ["GAZP", "CHMF"]
T_SCREEN = 0.20


# ── признаки этого скрининга — НЕ в smap_band_ref.py (реферный модуль чист) ──

def to_percentile_rank(series: np.ndarray, window: int = 252) -> np.ndarray:
    """Причинный перцентильный ранг series[i] относительно ПРЕДЫДУЩИХ window
    значений (i само не входит). Единый масштаб [0,1] для любой фичи —
    делает λ сравнимой между trend/velocity/acceleration/volume."""
    n = len(series)
    ranks = np.full(n, np.nan)
    for i in range(window, n):
        hist = series[i - window:i]
        finite = hist[np.isfinite(hist)]
        if len(finite) < 10 or not np.isfinite(series[i]):
            continue
        ranks[i] = float((finite < series[i]).mean())
    return ranks


def compute_trend_raw(log_highs, log_lows, window_ma=100):
    """Позиция цены относительно скользящей средней (лог-расстояние)."""
    mid = (log_highs + log_lows) / 2.0
    n = len(mid)
    trend = np.full(n, np.nan)
    csum = np.concatenate([[0.0], np.cumsum(mid)])
    for i in range(window_ma, n):
        ma = (csum[i] - csum[i - window_ma]) / window_ma
        trend[i] = mid[i] - ma
    return trend


def compute_velocity_raw(log_highs, log_lows, window_v=10):
    """Лог-доходность за последние window_v баров (скорость)."""
    mid = (log_highs + log_lows) / 2.0
    n = len(mid)
    vel = np.full(n, np.nan)
    vel[window_v:] = mid[window_v:] - mid[:-window_v]
    return vel


def compute_acceleration_raw(log_highs, log_lows, window_v=10):
    """Изменение скорости за window_v баров (вторая производная)."""
    vel = compute_velocity_raw(log_highs, log_lows, window_v)
    n = len(vel)
    acc = np.full(n, np.nan)
    acc[window_v:] = vel[window_v:] - vel[:-window_v]
    return acc


FEATURES = {
    "trend":        lambda lh, ll: to_percentile_rank(compute_trend_raw(lh, ll)),
    "velocity":     lambda lh, ll: to_percentile_rank(compute_velocity_raw(lh, ll)),
    "acceleration": lambda lh, ll: to_percentile_rank(compute_acceleration_raw(lh, ll)),
}


def main():
    t0 = time.time()
    print(f"Загрузка кросс-тикерного пула ({len(UNIVERSE)} тикеров)…")
    base_data = {}
    for tk in UNIVERSE:
        loaded = calib29g.load_ticker_candles_with_volume(tk, "1d")
        if loaded is None:
            continue
        lh, ll, dates, _vol = loaded   # объём тут не нужен, только high/low/dates
        base_data[tk] = (lh, ll, dates)
    print(f"загружено {len(base_data)}/{len(UNIVERSE)} тикеров  [{time.time()-t0:.1f}s]\n")

    # скрининг — грубая сетка (шаг/границы теперь настоящие параметры
    # calibrate_lambda, не глобальные константы), 2 сплита (не 4), 2 тикера
    calib29g.DEV_SPLIT_FRACS = [0.65, 0.80]
    SCREEN_LAMBDA_LO, SCREEN_LAMBDA_HI, SCREEN_LAMBDA_STEP = 0.0, 16.0, 2.0

    all_rows = []
    for feat_name, feat_fn in FEATURES.items():
        print(f"=== Признак: {feat_name} ===")
        ticker_data = {tk: (lh, ll, dates, feat_fn(lh, ll)) for tk, (lh, ll, dates) in base_data.items()}
        for target in SCREEN_TICKERS:
            print(f"--- {target} ---")
            r = calib29g.calibrate_lambda(target, T_SCREEN, ticker_data, t0,
                                          SCREEN_LAMBDA_LO, SCREEN_LAMBDA_HI, SCREEN_LAMBDA_STEP)
            r["feature"] = feat_name
            all_rows.append(r)

    rows = [r for r in all_rows if not r.get("skipped")]
    if rows:
        df = pd.DataFrame(rows)[["feature", "target", "lambda_star", "spread", "n_candidates",
                                 "holdout_cov_err", "baseline_cov_err", "rel_vs_baseline", "n_holdout"]]
        out_path = RESULTS / "29h_screen_summary.csv"
        df.to_csv(out_path, index=False)
        print("\n=== Итог скрининга ===")
        print(df.to_string(index=False))
        print(f"\nСохранено: {out_path}")

    print(f"\nВсего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
