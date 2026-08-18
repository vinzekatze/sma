"""level_methods_detrended — level_methods_compare_rolling.py, но уровни
считаются в ОСТАТКЕ от локального скользящего тренда (log_price - anchor,
та же логика, что и analyzers/trend_variance.py / app.py), а не в
абсолютной log-цене. Мотивация пользователя: на стабильных трендовых
участках цена визуально колеблется в некотором диапазоне ОТ тренда — не
абсолютные уровни, а уровни ОТНОСИТЕЛЬНО локального тренда.

Побочный эффект детрендинга: устраняет проблему смены ценового режима
(баг level_methods_compare.py) по построению — остаток стационарен
относительно СВОЕГО ЖЕ локального тренда, поэтому можно безопасно взять
БОЛЬШЕ окно (второй запрос пользователя): W_PAST=1000 вместо 252.

anchor[t] — значение локального тренда (окно TREND_WINDOW, оканчивается в
t) — переиспользует rolling_trend_variance + rolling_window_mean 1-в-1
(та же формула, что и в analyzers/price_forecast.py).
resid[t] = log_price[t] - anchor[t]; resid_high/resid_low — high/low,
детрендированные ТЕМ ЖЕ anchor (аппроксимация — anchor считался по close).
swing — локальные экстремумы ОСТАТКА (не сырой цены) — колебание вокруг
тренда, не сам тренд.

Запуск (из prototype_analyzers/): python level_methods_detrended.py
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
from analyzers.price_forecast import rolling_window_mean

DATA_DIR = _ROOT / "data" / "candles"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INTERVAL = "1d"
TICKERS = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
DELTA = 0.02
TREND_WINDOW = 100

W_PAST = 1000
W_FUTURE = 250
DECAY_HALFLIVES = {"none": None, "hl250": 250.0}
MIN_TOUCHES = 3

N_ORIGINS = 15
STEP_WF = 150


def load_ohlcv(ticker: str, interval: str) -> dict[str, np.ndarray]:
    path = DATA_DIR / ticker / f"{interval}.json"
    with open(path) as f:
        data = json.load(f)
    return dict(
        high=np.array([float(c["high"]) for c in data]),
        low=np.array([float(c["low"]) for c in data]),
        close=np.array([float(c["close"]) for c in data]),
        vol=np.array([float(c.get("volume", 0.0)) for c in data]),
    )


def bin_index(x: np.ndarray, delta: float) -> np.ndarray:
    return np.round(x / delta).astype(np.int64)


def swing_mask(x: np.ndarray) -> np.ndarray:
    d = np.diff(x)
    sign = np.sign(d)
    is_swing = np.zeros(len(x), dtype=bool)
    chg = sign[1:] != sign[:-1]
    is_swing[1:-1] = chg & (sign[1:] != 0) & (sign[:-1] != 0)
    return is_swing


def decayed_weights(n: int, halflife: float | None) -> np.ndarray:
    if halflife is None:
        return np.ones(n)
    lam = np.log(2) / halflife
    age = np.arange(n)[::-1]
    return np.exp(-lam * age)


def bin_sums(bin_idx: np.ndarray, weights: np.ndarray, bin0: int, n_bins: int) -> np.ndarray:
    out = np.zeros(n_bins)
    order = bin_idx - bin0
    valid = (order >= 0) & (order < n_bins)
    np.add.at(out, order[valid], weights[valid])
    return out


def analyze_origin(resid_hi, resid_lo, resid_close_bin, swings, vol, origin: int) -> list[dict]:
    a, mid, b = origin - W_PAST, origin, origin + W_FUTURE
    hi_bin_all = bin_index(resid_hi, DELTA)
    lo_bin_all = bin_index(resid_lo, DELTA)

    bin0 = int(min(lo_bin_all[a:b].min(), resid_close_bin[a:b].min()))
    bin1 = int(max(hi_bin_all[a:b].max(), resid_close_bin[a:b].max()))
    n_bins = bin1 - bin0 + 1
    if n_bins > 5000:
        return []

    def agg_touch(lo, hi, weights):
        out = np.zeros(n_bins)
        for i in range(len(lo)):
            s = max(lo[i] - bin0, 0)
            e = min(hi[i] - bin0 + 1, n_bins)
            if e > s:
                out[s:e] += weights[i]
        return out

    def agg_dwell(cb, weights):
        return bin_sums(cb, weights, bin0, n_bins)

    def agg_volume(cb, v, weights):
        return bin_sums(cb, v * weights, bin0, n_bins)

    def agg_swing(cb, sw_mask, weights):
        idx = np.where(sw_mask)[0]
        if len(idx) == 0:
            return np.zeros(n_bins)
        return bin_sums(cb[idx], weights[idx], bin0, n_bins)

    fw = np.ones(b - mid)
    future = dict(
        touch=agg_touch(lo_bin_all[mid:b], hi_bin_all[mid:b], fw),
        dwell=agg_dwell(resid_close_bin[mid:b], fw),
        volume=agg_volume(resid_close_bin[mid:b], vol[mid:b], fw),
        swing=agg_swing(resid_close_bin[mid:b], swings[mid:b], fw),
    )

    rows = []
    for decay_name, hl in DECAY_HALFLIVES.items():
        pw = decayed_weights(mid - a, hl)
        pw_flat = np.ones(mid - a)
        past = dict(
            touch=agg_touch(lo_bin_all[a:mid], hi_bin_all[a:mid], pw),
            dwell=agg_dwell(resid_close_bin[a:mid], pw),
            volume=agg_volume(resid_close_bin[a:mid], vol[a:mid], pw),
            swing=agg_swing(resid_close_bin[a:mid], swings[a:mid], pw),
        )
        past_flat = dict(
            touch=agg_touch(lo_bin_all[a:mid], hi_bin_all[a:mid], pw_flat),
            dwell=agg_dwell(resid_close_bin[a:mid], pw_flat),
            volume=agg_volume(resid_close_bin[a:mid], vol[a:mid], pw_flat),
            swing=agg_swing(resid_close_bin[a:mid], swings[a:mid], pw_flat),
        )
        for pm in ("touch", "dwell", "volume", "swing"):
            p_vals = past[pm]
            thr = past_flat[pm]
            for fm in ("touch", "dwell", "volume", "swing"):
                f_vals = future[fm]
                if pm == "swing" or fm == "swing":
                    mask = (thr > 0) & (f_vals > 0)
                else:
                    mask = (thr >= MIN_TOUCHES) & (f_vals >= MIN_TOUCHES)
                n_ok = int(mask.sum())
                corr = float(np.corrcoef(p_vals[mask], f_vals[mask])[0, 1]) if n_ok >= 8 else np.nan
                rows.append(dict(origin=origin, past_metric=pm, decay=decay_name,
                                  future_metric=fm, n_bins=n_ok, corr=corr))
    return rows


def main() -> None:
    all_rows = []
    for ticker in TICKERS:
        d = load_ohlcv(ticker, INTERVAL)
        log_hi, log_lo, log_cl = np.log(d["high"]), np.log(d["low"]), np.log(d["close"])

        slope, _ = rolling_trend_variance(log_cl, TREND_WINDOW)
        mean_y = rolling_window_mean(log_cl, TREND_WINDOW)
        anchor = mean_y + slope * (TREND_WINDOW - 1) / 2.0

        resid_hi = log_hi - anchor
        resid_lo = log_lo - anchor
        resid_cl = log_cl - anchor
        resid_close_bin = bin_index(resid_cl, DELTA)
        swings = swing_mask(resid_cl)

        n = len(d["close"])
        earliest_valid = TREND_WINDOW - 1
        last_origin = n - W_FUTURE
        origins = [o for o in range(last_origin - (N_ORIGINS - 1) * STEP_WF, last_origin + 1, STEP_WF)
                   if o - W_PAST >= earliest_valid]
        for origin in origins:
            for row in analyze_origin(resid_hi, resid_lo, resid_close_bin, swings, d["vol"], origin):
                row["ticker"] = ticker
                all_rows.append(row)
        print(f"{ticker}: {len(origins)} origins done")

    df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "level_methods_detrended.csv"
    df.to_csv(out_path, index=False)

    print("\n=== средняя корреляция (сначала по origin внутри тикера, потом по тикерам), 'та же метрика' ===")
    same = df[df["past_metric"] == df["future_metric"]]
    per_ticker = same.groupby(["ticker", "past_metric", "decay"])["corr"].mean().reset_index()
    print(per_ticker.groupby(["past_metric", "decay"])["corr"].agg(["mean", "std", "median", "count"]))

    print("\n=== volume(прошлое) -> dwell/swing(будущее) ===")
    cross = df[(df["past_metric"] == "volume") & (df["future_metric"].isin(["dwell", "swing"]))]
    per_ticker_c = cross.groupby(["ticker", "future_metric", "decay"])["corr"].mean().reset_index()
    print(per_ticker_c.groupby(["future_metric", "decay"])["corr"].agg(["mean", "std", "median", "count"]))

    print(f"\nсохранено: {out_path}")


if __name__ == "__main__":
    main()
