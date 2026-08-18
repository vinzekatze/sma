"""price_forecast_test — walk-forward проверка восстановленной цены (h=1) из
slope-прогноза «по ускорению» + доверительного интервала из resid_var.

Продолжение slope_predictability_test.py / _sweep.py — там «ускорение»
оказалось лучшим методом прогноза slope на всех 144 комбинациях. Здесь
проверяем, дают ли реконструкция цены и полоса неопределённости что-то
полезное на уровне ЦЕНЫ (не slope).

Методология:
  - anchor[t] + pred_slope[t] → pred_log_price[t+1] → exp() → pred_price
  - baseline 1: персистенция цены (pred = close[t])
  - baseline 2: наивное продолжение тренда БЕЗ коррекции на ускорение
    (pred_log = anchor[t] + slope[t]) — чтобы отделить эффект самой
    реконструкции (anchor+интеграция тренда) от эффекта ускорения.
  - полоса: pred_price * exp(±k·sqrt(var[t])), k ∈ K_GRID.
  - метрики: rMAE цены (относительно персистенции) + эмпирическое покрытие
    полосы (доля origins, где true_price внутри [lo,hi]) для каждого k,
    сравнивается с номинальным Gaussian-покрытием (чтобы видеть, насколько
    полоса калибрована, а не просто "широкая = хорошо").

Каузальность: anchor/slope/var в момент t используют только log_price[:t+1]
(окно [t-window+1, t]) — уже каузально по построению rolling_trend_variance.

Тикеры/сетап — как в предыдущих тестах (8 тикеров, 1d, window=50).

Запуск (из prototype_analyzers/): python price_forecast_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from scipy import stats

_HERE = Path(__file__).parent
_ROOT = _HERE.parent

for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from analyzers.price_forecast import forecast_next_bar

DATA_DIR = _ROOT / "data" / "candles"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INTERVAL = "1d"
TICKERS = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
WINDOW = 50

N_ORIGINS = 200   # больше, чем в slope-тестах — тут метрика (покрытие) требует выборки побольше
STEP_WF = 5

K_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]


def load_close(ticker: str, interval: str) -> np.ndarray:
    path = DATA_DIR / ticker / f"{interval}.json"
    with open(path) as f:
        data = json.load(f)
    return np.array([float(c["close"]) for c in data])


def main() -> None:
    rows = []
    for ticker in TICKERS:
        close = load_close(ticker, INTERVAL)
        log_price = np.log(close)
        out = forecast_next_bar(log_price, WINDOW)

        n = len(close)
        valid_start = WINDOW  # нужен slope[t-1], т.е. t>=window (не window-1)
        last_origin = n - 2   # нужен close[origin+1] как истина
        origins = list(range(last_origin - (N_ORIGINS - 1) * STEP_WF, last_origin + 1, STEP_WF))
        origins = [o for o in origins if o >= valid_start]

        for origin in origins:
            true_price = close[origin + 1]
            pred_price_accel = np.exp(out["pred_log"][origin])
            pred_price_notrend_accel = np.exp(out["anchor"][origin] + out["slope"][origin])  # без ускорения
            pred_price_persist = close[origin]
            std_log = np.sqrt(out["var"][origin])

            rows.append(dict(
                ticker=ticker, origin=origin,
                true=true_price,
                pred_persist=pred_price_persist,
                pred_trend=pred_price_notrend_accel,
                pred_accel=pred_price_accel,
                std_log=std_log,
                pred_log=out["pred_log"][origin],
            ))

    df = pd.DataFrame(rows)
    for col in ("pred_persist", "pred_trend", "pred_accel"):
        df[f"err_{col}"] = np.abs(df[col] - df["true"]) / df["true"]  # относительная ошибка (MAPE-стиль)

    mae_persist = df["err_pred_persist"].mean()
    print(f"тикеров={len(TICKERS)} window={WINDOW} origins/ticker~{N_ORIGINS} всего={len(df)}")
    print()
    print("=== точность цены (h=1) ===")
    for col in ("pred_persist", "pred_trend", "pred_accel"):
        mape = df[f"err_{col}"].mean()
        rmae = mape / mae_persist
        print(f"{col:>12s}: MAPE={mape:.5f}  rMAE={rmae:.4f}")

    print()
    print("=== покрытие полосы pred_accel ± k·std(resid_var), лог-симметрично ===")
    true_log = np.log(df["true"].values)
    for k in K_GRID:
        lo = df["pred_log"].values - k * df["std_log"].values
        hi = df["pred_log"].values + k * df["std_log"].values
        coverage = np.mean((true_log >= lo) & (true_log <= hi))
        nominal = 2 * stats.norm.cdf(k) - 1
        rel_width = np.mean(np.exp(hi) / np.exp(lo) - 1)  # ширина полосы в % от нижней границы, ориентир
        print(f"k={k:.1f}: покрытие={coverage:.3f}  номинал(Gauss)={nominal:.3f}  "
              f"ширина(hi/lo-1)={rel_width:.4f}")

    out_path = RESULTS_DIR / "price_forecast_test.csv"
    df.to_csv(out_path, index=False)
    print(f"\nсохранено: {out_path}")


if __name__ == "__main__":
    main()
