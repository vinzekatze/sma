"""var_band_test — практическая проверка: даёт ли полоса на ПРОГНОЗНОЙ
var[origin+1] (дельта-экстраполяция, см. var_predictability_sweep.py)
лучшее покрытие/ширину, чем полоса на ТЕКУЩЕЙ var[origin] (то, что уже
используется в app.py)?

Центр полосы в обоих случаях — персистенция цены log(close[origin])
(уже установлено в price_forecast_test.py как лучший практический центр
на h=1 — здесь не переоткрываем это, а фиксируем и меняем только ШИРИНУ).

Два варианта полосы:
  band_current: center ± k·sqrt(var[origin])                (что уже в app.py)
  band_pred:    center ± k·sqrt(var_pred[origin+1])          (дельта-экстраполяция var)
  var_pred = max(var[origin] + (var[origin]-var[origin-1]), 0)

Метрики (вместе, не порознь — см. критику): покрытие + средняя относительная
ширина, для k ∈ K_GRID.

Каузальность: var[origin], var[origin-1] используют только log_price[:origin+1]
— уже каузально по построению rolling_trend_variance.

Тикеры/window — как в предыдущих тестах (8 тикеров, 1d, window=200 — новый дефолт).

Запуск (из prototype_analyzers/): python var_band_test.py
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

from analyzers.trend_variance import rolling_trend_variance

DATA_DIR = _ROOT / "data" / "candles"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INTERVAL = "1d"
TICKERS = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
WINDOW = 200

N_ORIGINS = 200
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
        _, var_series = rolling_trend_variance(log_price, WINDOW)
        n = len(close)

        valid_start = WINDOW  # нужен var[origin-1] тоже валиден
        last_origin = n - 2
        origins = list(range(last_origin - (N_ORIGINS - 1) * STEP_WF, last_origin + 1, STEP_WF))
        origins = [o for o in origins if o >= valid_start]

        for origin in origins:
            center = log_price[origin]
            true_log = log_price[origin + 1]

            var_current = var_series[origin]
            var_pred = max(var_series[origin] + (var_series[origin] - var_series[origin - 1]), 0.0)

            rows.append(dict(
                ticker=ticker, origin=origin, center=center, true_log=true_log,
                std_current=np.sqrt(var_current), std_pred=np.sqrt(var_pred),
            ))

    df = pd.DataFrame(rows)
    print(f"тикеров={len(TICKERS)} window={WINDOW} строк={len(df)}")
    print()
    print(f"{'k':>4s} {'cov_current':>12s} {'width_current':>14s} "
          f"{'cov_pred':>9s} {'width_pred':>11s} {'nominal':>8s}")
    for k in K_GRID:
        lo_c = df["center"] - k * df["std_current"]
        hi_c = df["center"] + k * df["std_current"]
        cov_c = np.mean((df["true_log"] >= lo_c) & (df["true_log"] <= hi_c))
        width_c = np.mean(np.exp(hi_c) / np.exp(lo_c) - 1)

        lo_p = df["center"] - k * df["std_pred"]
        hi_p = df["center"] + k * df["std_pred"]
        cov_p = np.mean((df["true_log"] >= lo_p) & (df["true_log"] <= hi_p))
        width_p = np.mean(np.exp(hi_p) / np.exp(lo_p) - 1)

        nominal = 2 * stats.norm.cdf(k) - 1
        print(f"{k:>4.1f} {cov_c:>12.3f} {width_c:>14.4f} "
              f"{cov_p:>9.3f} {width_p:>11.4f} {nominal:>8.3f}")

    out_path = RESULTS_DIR / "var_band_test.csv"
    df.to_csv(out_path, index=False)
    print(f"\nсохранено: {out_path}")


if __name__ == "__main__":
    main()
