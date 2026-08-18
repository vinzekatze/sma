"""slope_predictability_test — насколько предсказуем slope (наклон скользящего
OLS-тренда из analyzers/trend_variance.py) относительно baseline-персистенции.

Согласовано с пользователем 2026-08-17:
  Гипотезы:
    H1 — LWR (без каскада) точнее персистенции (pred = slope[t]).
    H2 — каскадный поиск соседей (research/cascade_algorithm.md) точнее
         одноуровневого LWR.
    H3 — наивная экстраполяция «по ускорению»
         (pred = slope[t] + (slope[t]-slope[t-1])) как простой baseline.

  ⚠️ Критика: slope[t] сам по себе — выход OLS на окне [t-window+1, t];
  соседние slope[t], slope[t-1] считаются на окнах, перекрывающихся на
  window-1 баров → сильная встроенная инерция ряда. Персистенция здесь —
  заведомо сильный baseline; если LWR/каскад её не побьют, это не значит
  «метод не работает», это может значить «сигнал уже объяснён гладкостью
  конструкции ряда».

  Каузальность: slope[t] уже сам по себе каузален (окно оканчивается в t),
  поэтому срез slope[:origin+1] достаточен — не нужна доп. отсечка.

Параметры: window=50, p_fit=9, N_LEVELS=4 (октавный ×2 → p_lv=[72,36,18,9]),
xi=3*(p_fit+1)+5=35, N_ORIGINS=40, STEP_WF=5, тикер SBER 1d.

Запуск (из prototype_analyzers/): python slope_predictability_test.py
"""
from __future__ import annotations

import json
import sys
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

TICKER = "SBER"
INTERVAL = "1d"
WINDOW = 50  # окно тренда, как дефолт в app.py

P_FIT = 9
N_LEVELS = 4
XI_LWR = 3 * (P_FIT + 1) + 5  # = 35, правило ξ ≥ 3(p+1) + запас

N_ORIGINS = 40
STEP_WF = 5


def load_close(ticker: str, interval: str) -> tuple[np.ndarray, pd.DatetimeIndex]:
    path = DATA_DIR / ticker / f"{interval}.json"
    with open(path) as f:
        data = json.load(f)
    close = np.array([float(c["close"]) for c in data])
    times = pd.to_datetime([c["begin"] for c in data])
    return close, times


def main() -> None:
    close, times = load_close(TICKER, INTERVAL)
    log_price = np.log(close)
    slope, _ = rolling_trend_variance(log_price, WINDOW)

    p_max = P_FIT * (2 ** (N_LEVELS - 1))
    valid_start = WINDOW - 1  # первый небезнановый индекс slope
    min_origin = valid_start + p_max + XI_LWR + 5

    last_origin = len(slope) - 2  # нужен slope[origin+1] как истина
    origins = list(range(last_origin - (N_ORIGINS - 1) * STEP_WF, last_origin + 1, STEP_WF))
    origins = [o for o in origins if o >= min_origin]

    if not origins:
        raise RuntimeError("недостаточно истории для заданных параметров")

    rows = []
    for origin in origins:
        series = slope[valid_start:origin + 1]  # каузальный срез, без NaN-хвоста
        true_val = slope[origin + 1]

        pred_persist = series[-1]
        pred_accel = series[-1] + (series[-1] - series[-2])
        pred_nocascade = predict_no_cascade(series, P_FIT, XI_LWR)
        pred_cascade = predict_cascade(series, P_FIT, N_LEVELS, XI_LWR)

        rows.append(dict(
            origin=origin, time=times[origin],
            true=true_val,
            pred_persist=pred_persist, pred_accel=pred_accel,
            pred_nocascade=pred_nocascade, pred_cascade=pred_cascade,
        ))

    df = pd.DataFrame(rows)
    method_cols = ["pred_persist", "pred_accel", "pred_nocascade", "pred_cascade"]
    for col in method_cols:
        df[f"err_{col}"] = np.abs(df[col] - df["true"])

    mae_persist = df["err_pred_persist"].mean()

    print(f"тикер={TICKER} interval={INTERVAL} window={WINDOW} p_fit={P_FIT} "
          f"n_levels={N_LEVELS} xi={XI_LWR} origins={len(df)}")
    print()
    for col in method_cols:
        mae = df[f"err_{col}"].mean()
        rmae = mae / mae_persist
        n_nan = df[col].isna().sum()
        print(f"{col:>16s}: MAE={mae:.6g}  rMAE={rmae:.4f}  NaN={n_nan}")

    out_path = RESULTS_DIR / f"slope_predictability_{TICKER}_{INTERVAL}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nсохранено: {out_path}")


if __name__ == "__main__":
    main()
