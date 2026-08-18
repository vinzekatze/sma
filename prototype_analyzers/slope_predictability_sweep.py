"""slope_predictability_sweep — сетка (тикер × window × p_fit × N_LEVELS) для
проверки предсказуемости slope методами LWR без каскада / с каскадом,
продолжение slope_predictability_test.py (см. его докстринг — те же
гипотезы H1-H3 и критика).

Сетка:
  тикеры  = CHMF, LKOH, MGNT, MRKP, NLMK, NVTK, SBER, VTBR (1d, уже кэш.)
  window  = 30, 50, 100
  p_fit   = 7, 9, 13
  N_LEVELS = 2, 4     (октавный каскад; 2 уровня — «мягкий» каскад,
                        4 — как в первом тесте)
  xi = 3*(p_fit+1)+5  (правило проекта, без изменений)
  N_ORIGINS=40, STEP_WF=5 — как в первом тесте.

Вопрос: воспроизводится ли (пере-)упорядочение accel > no-cascade > cascade,
или это артефакт одного тикера/сетапа?

Запуск (из prototype_analyzers/): python slope_predictability_sweep.py
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
from analyzers.lwr_cascade import predict_no_cascade, predict_cascade

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


def load_close(ticker: str, interval: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    path = DATA_DIR / ticker / f"{interval}.json"
    with open(path) as f:
        data = json.load(f)
    close = np.array([float(c["close"]) for c in data])
    times = pd.to_datetime([c["begin"] for c in data])
    return close, times


def run_one(slope: np.ndarray, window: int, p_fit: int, n_levels: int, xi_lwr: int) -> list[dict]:
    p_max = p_fit * (2 ** (n_levels - 1))
    valid_start = window - 1
    min_origin = valid_start + p_max + xi_lwr + 5

    last_origin = len(slope) - 2
    origins = list(range(last_origin - (N_ORIGINS - 1) * STEP_WF, last_origin + 1, STEP_WF))
    origins = [o for o in origins if o >= min_origin]

    rows = []
    for origin in origins:
        series = slope[valid_start:origin + 1]
        true_val = slope[origin + 1]

        pred_persist = series[-1]
        pred_accel = series[-1] + (series[-1] - series[-2])
        pred_nocascade = predict_no_cascade(series, p_fit, xi_lwr)
        pred_cascade = predict_cascade(series, p_fit, n_levels, xi_lwr)

        rows.append(dict(
            true=true_val,
            pred_persist=pred_persist, pred_accel=pred_accel,
            pred_nocascade=pred_nocascade, pred_cascade=pred_cascade,
        ))
    return rows


def main() -> None:
    t0 = time.time()
    close_cache = {t: load_close(t, INTERVAL) for t in TICKERS}

    method_cols = ["pred_persist", "pred_accel", "pred_nocascade", "pred_cascade"]
    all_rows = []
    n_combos = len(TICKERS) * len(WINDOW_GRID) * len(P_FIT_GRID) * len(N_LEVELS_GRID)
    done = 0

    for ticker in TICKERS:
        close, _ = close_cache[ticker]
        log_price = np.log(close)
        for window in WINDOW_GRID:
            slope, _ = rolling_trend_variance(log_price, window)
            for p_fit in P_FIT_GRID:
                xi_lwr = 3 * (p_fit + 1) + 5
                for n_levels in N_LEVELS_GRID:
                    rows = run_one(slope, window, p_fit, n_levels, xi_lwr)
                    done += 1
                    if not rows:
                        continue
                    df = pd.DataFrame(rows)
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
    out_path = RESULTS_DIR / "slope_predictability_sweep.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nсохранено: {out_path} ({len(result_df)} строк, {time.time()-t0:.1f}s всего)")

    print("\n=== средний rMAE по методу, усреднённый по всем тикерам/параметрам ===")
    print(result_df.groupby("method")["rmae"].agg(["mean", "median", "std"]).loc[method_cols])

    print("\n=== средний rMAE по методу и N_LEVELS (cascade only) ===")
    casc = result_df[result_df["method"] == "pred_cascade"]
    print(casc.groupby("n_levels")["rmae"].agg(["mean", "median"]))

    print("\n=== доля комбинаций, где cascade точнее nocascade ===")
    piv = result_df.pivot_table(index=["ticker", "window", "p_fit", "n_levels"],
                                 columns="method", values="rmae")
    win_rate = (piv["pred_cascade"] < piv["pred_nocascade"]).mean()
    print(f"cascade < nocascade в {win_rate:.1%} комбинаций (n={len(piv)})")


if __name__ == "__main__":
    main()
