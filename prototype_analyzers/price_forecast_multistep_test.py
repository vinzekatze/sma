"""price_forecast_multistep_test — многошаговый прогноз цены h=1..H.

Урок из price_forecast_test.py: anchor должен быть реальной последней ценой
close[origin], НЕ сглаженным концом линии тренда (тот лагирует, даёт rMAE≈3).
Урок из accel_predictability_sweep.py: accel (Δslope) — шум (rMAE>1 хуже
персистенции) → ожидаем, что квадратичная поправка на «ускорение» будет
только вредить на бОльших горизонтах, а не помогать.

Три метода (все — детерминированная экстраполяция, без LWR/каскада — тот
уже проверен и здесь неприменим напрямую к многошаговости):
  persist   — pred_price[h] = close[origin]                          (baseline)
  linear    — pred_log[h]   = log(close[origin]) + h·slope[origin]
              (постоянный текущий наклон, без поправки на ускорение)
  quadratic — pred_log[h]   = log(close[origin]) + h·slope[origin]
                               + d·h·(h+1)/2,  d = slope[origin]-slope[origin-1]
              (постоянное «ускорение» — та же гипотеза, что дельта-
              экстраполяция slope на 1 шаг, но продолженная на h шагов)

Метрика: rMAE(h) = MAE(метод,h) / MAE(persist,h), отдельно по каждому h —
интересно, как ошибка растёт с горизонтом и меняется ли порядок методов.

Каузальность: slope[origin] использует только log_price[:origin+1] (окно
[origin-window+1, origin]) — прогноз на h шагов вперёд не трогает будущее.

Тикеры/window — как в предыдущих тестах (8 тикеров, 1d, window=50).

Запуск (из prototype_analyzers/): python price_forecast_multistep_test.py
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

DATA_DIR = _ROOT / "data" / "candles"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INTERVAL = "1d"
TICKERS = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
WINDOW = 50

H_GRID = [1, 2, 3, 5, 10, 20]
N_ORIGINS = 200
STEP_WF = 5


def load_close(ticker: str, interval: str) -> np.ndarray:
    path = DATA_DIR / ticker / f"{interval}.json"
    with open(path) as f:
        data = json.load(f)
    return np.array([float(c["close"]) for c in data])


def main() -> None:
    h_max = max(H_GRID)
    rows = []

    for ticker in TICKERS:
        close = load_close(ticker, INTERVAL)
        log_price = np.log(close)
        slope, _ = rolling_trend_variance(log_price, WINDOW)
        n = len(close)

        valid_start = WINDOW  # нужен slope[origin-1] тоже валиден
        last_origin = n - 1 - h_max
        origins = list(range(last_origin - (N_ORIGINS - 1) * STEP_WF, last_origin + 1, STEP_WF))
        origins = [o for o in origins if o >= valid_start]

        for origin in origins:
            s = slope[origin]
            d = slope[origin] - slope[origin - 1]
            log_p0 = log_price[origin]

            for h in H_GRID:
                true_price = close[origin + h]
                pred_persist = close[origin]
                pred_linear = np.exp(log_p0 + h * s)
                pred_quad = np.exp(log_p0 + h * s + d * h * (h + 1) / 2.0)

                rows.append(dict(
                    ticker=ticker, origin=origin, h=h, true=true_price,
                    pred_persist=pred_persist, pred_linear=pred_linear, pred_quad=pred_quad,
                ))

    df = pd.DataFrame(rows)
    for col in ("pred_persist", "pred_linear", "pred_quad"):
        df[f"err_{col}"] = np.abs(df[col] - df["true"]) / df["true"]

    print(f"тикеров={len(TICKERS)} window={WINDOW} всего строк={len(df)}")
    print()
    print(f"{'h':>4s} {'MAPE_persist':>13s} {'MAPE_linear':>12s} {'rMAE_linear':>12s} "
          f"{'MAPE_quad':>10s} {'rMAE_quad':>10s}")
    for h in H_GRID:
        sub = df[df["h"] == h]
        mape_p = sub["err_pred_persist"].mean()
        mape_l = sub["err_pred_linear"].mean()
        mape_q = sub["err_pred_quad"].mean()
        print(f"{h:>4d} {mape_p:>13.5f} {mape_l:>12.5f} {mape_l/mape_p:>12.4f} "
              f"{mape_q:>10.5f} {mape_q/mape_p:>10.4f}")

    out_path = RESULTS_DIR / "price_forecast_multistep_test.csv"
    df.to_csv(out_path, index=False)
    print(f"\nсохранено: {out_path}")


if __name__ == "__main__":
    main()
