"""level_methods_compare — ⚠️ БАГ: статичный split всей истории пополам на
фиксированной сетке абсолютных бинов даёт искусственную сильную
ОТРИЦАТЕЛЬНУЮ корреляцию по всем метрикам (смена ценового режима за
10-20 лет истории, прошлое/будущее почти не пересекаются по цене) —
не находка, артефакт постановки. Используйте
level_methods_compare_rolling.py (скользящее окно, walk-forward) — там
исправлено. Файл оставлен как задокументированный урок, не для повторного
запуска.

level_methods_compare — сравнение методов определения ценовых «уровней»
на аналитическую (не обязательно прогностическую) полезность.

Продолжение level_reversal_test.py: та находка (level-crossing reversal
heterogeneity) держалась только в зоне риска артефакта порядка событий
внутри бара и не пережила split-half. Здесь — другая конструкция: ФИКСИРОВАННЫЙ
бин по log-цене (не последовательность событий, поэтому проблема порядка
пересечений внутри бара не возникает), четыре независимые метрики на бин:

  touch  — бары, чей [low,high] задел бин (сколько раз цена вообще была здесь)
  dwell  — бары, чей close попал в бин ("магнит" — время, проведённое у цены)
  volume — суммарный объём баров с close в бине (полный volume profile,
           расширение эксп.20b с "только пивоты" на "все бары")
  swing  — локальные экстремумы close (смена знака приращения) в бине
           (простейшее определение "разворота", грубее зигзага)

Ключевая проверка — НАПРАВЛЕННАЯ (прошлое→будущее, не симметричный
split-half, как в level_reversal_test.py): история делится на прошлое
[0,T) и будущее [T,N). Прошлое считается с decay по давности (без decay —
baseline, и exponential half-life) — проверяем, предсказывает ли
"уровень" из прошлого то же самое (или другое) поведение в будущем, и
помогает ли decay. Это прямее отвечает на вопрос "устаревают ли уровни",
чем симметричный тест.

⚠️ Прецедент: эксп.20c/20d уже пробовали volume-as-predictor (расстояние
до HVN → величина следующего хода) — провал, r≈0.03-0.08. Здесь вопрос
ДРУГОЙ (volume-at-price → будущее dwell/swing НА ТОЙ ЖЕ цене, не величина
хода) — не строгий повтор, но тот прецедент снижает априорную вероятность
успеха, держим в уме.

Δ=0.02 (log-цена, бины) — без риска артефакта порядка (в отличие от
level_reversal_test.py, здесь просто биннинг, не последовательность
событий). Аналитическая цель — не прогноз, поэтому строгая каузальность
прогнозного пайплайна не требуется, но направленность прошлое→будущее
соблюдена, чтобы не мерить чистую ретроспекцию.

Запуск (из prototype_analyzers/): python level_methods_compare.py
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

DECAY_HALFLIVES = {"none": None, "hl150": 150.0, "hl400": 400.0}
MIN_TOUCHES = 5  # мин. наблюдений метрики в бине (и в прошлом, и в будущем), чтобы бин учитывался


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
    age = np.arange(n)[::-1]  # 0 = самый последний (ближе к T), растёт назад во времени
    return np.exp(-lam * age)


def bin_sums(bin_idx: np.ndarray, weights: np.ndarray, all_bins: np.ndarray) -> np.ndarray:
    """Сумма weights по каждому значению из all_bins (0, если бин не встретился)."""
    order = np.searchsorted(all_bins, bin_idx)
    out = np.zeros(len(all_bins))
    np.add.at(out, order, weights)
    return out


def analyze_ticker(ticker: str) -> list[dict]:
    d = load_ohlcv(ticker, INTERVAL)
    n = len(d["lc"])
    T = n // 2

    lh, ll, lc, vol = d["lh"], d["ll"], d["lc"], d["vol"]
    close_bin = bin_index(lc, DELTA)
    swings = swing_mask(lc)

    # touch: бин задет, если попадает в [bin(low), bin(high)] бара
    lo_bin = bin_index(ll, DELTA)
    hi_bin = bin_index(lh, DELTA)

    all_bins = np.arange(lo_bin.min(), hi_bin.max() + 1)

    def past_future_split(mask_or_weight_fn, is_touch: bool = False):
        """Возвращает (past_by_decay: dict[str,np.ndarray], future: np.ndarray) выровненные по all_bins."""
        future_vals = mask_or_weight_fn(T, n, weights=np.ones(n - T))
        past_by_decay = {}
        for name, hl in DECAY_HALFLIVES.items():
            w = decayed_weights(T, hl)
            past_by_decay[name] = mask_or_weight_fn(0, T, weights=w)
        return past_by_decay, future_vals

    def agg_dwell(a, b, weights):
        return bin_sums(close_bin[a:b], weights, all_bins)

    def agg_volume(a, b, weights):
        return bin_sums(close_bin[a:b], vol[a:b] * weights, all_bins)

    def agg_swing(a, b, weights):
        idx = np.where(swings[a:b])[0]
        if len(idx) == 0:
            return np.zeros(len(all_bins))
        return bin_sums(close_bin[a:b][idx], weights[idx], all_bins)

    def agg_touch(a, b, weights):
        out = np.zeros(len(all_bins))
        lob, hib = lo_bin[a:b], hi_bin[a:b]
        for i in range(len(lob)):
            lo_i, hi_i = lob[i], hib[i]
            sl = slice(lo_i - all_bins[0], hi_i - all_bins[0] + 1)
            out[sl] += weights[i]
        return out

    metrics = {"touch": agg_touch, "dwell": agg_dwell, "volume": agg_volume, "swing": agg_swing}

    past: dict[str, dict[str, np.ndarray]] = {}
    future: dict[str, np.ndarray] = {}
    for mname, fn in metrics.items():
        p, f = past_future_split(fn)
        past[mname] = p
        future[mname] = f

    rows = []
    for past_metric in metrics:
        for decay_name in DECAY_HALFLIVES:
            p_vals = past[past_metric][decay_name]
            for future_metric in metrics:
                f_vals = future[future_metric]
                mask = (past[past_metric]["none"] >= MIN_TOUCHES) & (f_vals >= MIN_TOUCHES) \
                    if past_metric != "swing" and future_metric != "swing" \
                    else (past[past_metric]["none"] > 0) & (f_vals > 0)
                n_bins = int(mask.sum())
                if n_bins < 8:
                    corr = np.nan
                else:
                    corr = float(np.corrcoef(p_vals[mask], f_vals[mask])[0, 1])
                rows.append(dict(
                    ticker=ticker, past_metric=past_metric, decay=decay_name,
                    future_metric=future_metric, n_bins=n_bins, corr=corr,
                ))
    return rows


def main() -> None:
    all_rows = []
    for ticker in TICKERS:
        all_rows.extend(analyze_ticker(ticker))
        print(f"{ticker}: done")

    df = pd.DataFrame(all_rows)
    out_path = RESULTS_DIR / "level_methods_compare.csv"
    df.to_csv(out_path, index=False)

    print("\n=== средняя корреляция (по 8 тикерам) для 'той же метрики' прошлое->будущее ===")
    same = df[df["past_metric"] == df["future_metric"]]
    print(same.groupby(["past_metric", "decay"])["corr"].agg(["mean", "median", "count"]))

    print("\n=== volume(прошлое) -> dwell/swing(будущее), проверка идеи эксп.20b на полном профиле ===")
    cross = df[(df["past_metric"] == "volume") & (df["future_metric"].isin(["dwell", "swing"]))]
    print(cross.groupby(["future_metric", "decay"])["corr"].agg(["mean", "median", "count"]))

    print(f"\nсохранено: {out_path}")


if __name__ == "__main__":
    main()
