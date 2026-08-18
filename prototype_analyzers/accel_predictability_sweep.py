"""accel_predictability_sweep — предсказуем ли сам ряд «ускорения»
accel[t] = slope[t] - slope[t-1] (вторая производная log-цены)?

Та же сетка и методология, что и slope_predictability_sweep.py (см. его
докстринг за гипотезами/критикой) — переиспользует общий walk-forward
(analyzers/predictability.py) и LWR/каскад (analyzers/lwr_cascade.py) без
изменений алгоритма.

Ожидание (озвучено пользователем до запуска): accel — вероятно шум
(разность уже гладкого ряда slope усиливает шумовую компоненту, а не
убирает её) — проверяем.

Запуск (из prototype_analyzers/): python accel_predictability_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
_ROOT = _HERE.parent

for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from analyzers.trend_variance import rolling_trend_variance
from analyzers.predictability import run_walk_forward

DATA_DIR = _ROOT / "data" / "candles"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INTERVAL = "1d"
TICKERS = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]

WINDOW_GRID = [30, 50, 100]
P_FIT_GRID = [7, 9, 13]
N_LEVELS_GRID = [2, 4]

N_ORIGINS = 40
STEP_WF = 5


def load_close(ticker: str, interval: str) -> np.ndarray:
    path = DATA_DIR / ticker / f"{interval}.json"
    with open(path) as f:
        data = json.load(f)
    return np.array([float(c["close"]) for c in data])


def main() -> None:
    t0 = time.time()
    close_cache = {t: load_close(t, INTERVAL) for t in TICKERS}

    method_cols = ["pred_persist", "pred_delta", "pred_nocascade", "pred_cascade"]
    all_rows = []
    n_combos = len(TICKERS) * len(WINDOW_GRID) * len(P_FIT_GRID) * len(N_LEVELS_GRID)
    done = 0

    for ticker in TICKERS:
        close = close_cache[ticker]
        log_price = np.log(close)
        for window in WINDOW_GRID:
            slope, _ = rolling_trend_variance(log_price, window)
            accel = np.full(len(slope), np.nan)
            accel[1:] = slope[1:] - slope[:-1]
            valid_start = window  # первый небезнановый индекс accel

            for p_fit in P_FIT_GRID:
                xi_lwr = 3 * (p_fit + 1) + 5
                for n_levels in N_LEVELS_GRID:
                    df = run_walk_forward(accel, valid_start, p_fit, n_levels, xi_lwr, N_ORIGINS, STEP_WF)
                    done += 1
                    if df.empty:
                        continue
                    for col in method_cols:
                        df[f"err_{col}"] = np.abs(df[col] - df["true"])
                    mae_persist = df["err_pred_persist"].mean()
                    for col in method_cols:
                        mae = df[f"err_{col}"].mean()
                        rmae = mae / mae_persist if mae_persist > 0 else np.nan
                        all_rows.append(dict(
                            ticker=ticker, window=window, p_fit=p_fit,
                            n_levels=n_levels, xi=xi_lwr, method=col,
                            mae=mae, rmae=rmae, n_origins=len(df),
                            n_nan=int(df[col].isna().sum()),
                        ))
                    print(f"[{done}/{n_combos}] {ticker} w={window} p_fit={p_fit} "
                          f"n_levels={n_levels} — {time.time()-t0:.1f}s")

    result_df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "accel_predictability_sweep.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nсохранено: {out_path} ({len(result_df)} строк, {time.time()-t0:.1f}s всего)")

    print("\n=== средний rMAE по методу ===")
    print(result_df.groupby("method")["rmae"].agg(["mean", "median", "std"]).loc[method_cols])


if __name__ == "__main__":
    main()
