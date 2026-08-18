"""level_methods_compare_rolling — исправленная версия level_methods_compare.py.

Баг предыдущей версии: фиксированная сетка бинов по АБСОЛЮТНОЙ log-цене на
ВСЕЙ истории (10-20 лет), разбитой пополам статично — из-за смены
ценового режима за такой срок прошлое/будущее почти не пересекаются по
цене, что дало искусственную сильную ОТРИЦАТЕЛЬНУЮ корреляцию по всем
метрикам без исключения (артефакт, не находка). Эксп.20b использовал
именно поэтому СКОЛЬЗЯЩЕЕ окно 252 бара до события — деталь, которую я
упустил при первой попытке.

Здесь — walk-forward: много origin по истории, для каждого:
  прошлое = [origin-W_PAST, origin)   — с decay (none / hl60)
  будущее = [origin, origin+W_FUTURE) — как есть

Корреляция past_metric(бин) vs future_metric(бин) считается ОТДЕЛЬНО на
каждом origin (локальная сетка бинов внутри окна, не глобальная), затем
усредняется по origins — устраняет проблему смены режима, т.к. окно
W_PAST+W_FUTURE=352 бара (~1.4 года на 1d) не успевает уйти в другой
ценовой диапазон, как это было при split по всей истории.

Метрики, декей, тикеры — как в level_methods_compare.py.

Запуск (из prototype_analyzers/): python level_methods_compare_rolling.py
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

DATA_DIR = _ROOT / "data" / "candles"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INTERVAL = "1d"
TICKERS = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
DELTA = 0.02

W_PAST = 252
W_FUTURE = 100
DECAY_HALFLIVES = {"none": None, "hl60": 60.0}
MIN_TOUCHES = 3

N_ORIGINS = 30
STEP_WF = 60


def load_ohlcv(ticker: str, interval: str) -> dict[str, np.ndarray]:
    path = DATA_DIR / ticker / f"{interval}.json"
    with open(path) as f:
        data = json.load(f)
    return dict(
        lh=np.log(np.array([float(c["high"]) for c in data])),
        ll=np.log(np.array([float(c["low"]) for c in data])),
        lc=np.log(np.array([float(c["close"]) for c in data])),
        vol=np.array([float(c.get("volume", 0.0)) for c in data]),
    )


def bin_index(x: np.ndarray, delta: float) -> np.ndarray:
    return np.round(x / delta).astype(np.int64)


def swing_mask(close_log: np.ndarray) -> np.ndarray:
    d = np.diff(close_log)
    sign = np.sign(d)
    is_swing = np.zeros(len(close_log), dtype=bool)
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


def analyze_origin(d: dict, swings: np.ndarray, origin: int) -> list[dict]:
    a, mid, b = origin - W_PAST, origin, origin + W_FUTURE
    lh, ll, lc, vol = d["lh"], d["ll"], d["lc"], d["vol"]

    lo_bin_all = bin_index(ll, DELTA)
    hi_bin_all = bin_index(lh, DELTA)
    close_bin_all = bin_index(lc, DELTA)

    bin0 = int(min(lo_bin_all[a:b].min(), close_bin_all[a:b].min()))
    bin1 = int(max(hi_bin_all[a:b].max(), close_bin_all[a:b].max()))
    n_bins = bin1 - bin0 + 1
    if n_bins > 5000:  # защита от патологии (не должно случаться при разумном DELTA)
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

    # будущее (плоские веса)
    fw = np.ones(b - mid)
    future = dict(
        touch=agg_touch(lo_bin_all[mid:b], hi_bin_all[mid:b], fw),
        dwell=agg_dwell(close_bin_all[mid:b], fw),
        volume=agg_volume(close_bin_all[mid:b], vol[mid:b], fw),
        swing=agg_swing(close_bin_all[mid:b], swings[mid:b], fw),
    )

    rows = []
    for decay_name, hl in DECAY_HALFLIVES.items():
        pw = decayed_weights(mid - a, hl)
        past = dict(
            touch=agg_touch(lo_bin_all[a:mid], hi_bin_all[a:mid], pw),
            dwell=agg_dwell(close_bin_all[a:mid], pw),
            volume=agg_volume(close_bin_all[a:mid], vol[a:mid], pw),
            swing=agg_swing(close_bin_all[a:mid], swings[a:mid], pw),
        )
        past_flat = dict(
            touch=agg_touch(lo_bin_all[a:mid], hi_bin_all[a:mid], np.ones(mid - a)),
            dwell=agg_dwell(close_bin_all[a:mid], np.ones(mid - a)),
            volume=agg_volume(close_bin_all[a:mid], vol[a:mid], np.ones(mid - a)),
            swing=agg_swing(close_bin_all[a:mid], swings[a:mid], np.ones(mid - a)),
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
                if n_ok < 8:
                    corr = np.nan
                else:
                    corr = float(np.corrcoef(p_vals[mask], f_vals[mask])[0, 1])
                rows.append(dict(origin=origin, past_metric=pm, decay=decay_name,
                                  future_metric=fm, n_bins=n_ok, corr=corr))
    return rows


def main() -> None:
    all_rows = []
    for ticker in TICKERS:
        d = load_ohlcv(ticker, INTERVAL)
        swings = swing_mask(d["lc"])
        n = len(d["lc"])
        last_origin = n - W_FUTURE
        origins = [o for o in range(last_origin - (N_ORIGINS - 1) * STEP_WF, last_origin + 1, STEP_WF)
                   if o - W_PAST >= 0]
        for origin in origins:
            for row in analyze_origin(d, swings, origin):
                row["ticker"] = ticker
                all_rows.append(row)
        print(f"{ticker}: {len(origins)} origins done")

    df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "level_methods_compare_rolling.csv"
    df.to_csv(out_path, index=False)

    print("\n=== средняя корреляция по origins (сначала усреднить внутри тикера, потом по тикерам), 'та же метрика' ===")
    same = df[df["past_metric"] == df["future_metric"]]
    per_ticker = same.groupby(["ticker", "past_metric", "decay"])["corr"].mean().reset_index()
    print(per_ticker.groupby(["past_metric", "decay"])["corr"].agg(["mean", "median", "count"]))

    print("\n=== volume(прошлое) -> dwell/swing(будущее) ===")
    cross = df[(df["past_metric"] == "volume") & (df["future_metric"].isin(["dwell", "swing"]))]
    per_ticker_c = cross.groupby(["ticker", "future_metric", "decay"])["corr"].mean().reset_index()
    print(per_ticker_c.groupby(["future_metric", "decay"])["corr"].agg(["mean", "median", "count"]))

    print(f"\nсохранено: {out_path}")


if __name__ == "__main__":
    main()
