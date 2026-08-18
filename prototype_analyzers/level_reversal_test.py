"""level_reversal_test — новый вопрос про level-crossing событийное время
(research/phase7_fractality/26_level_crossing/, app6.py), НЕ пересчёт
эксп.26. Тот проверял предсказуемость НАПРАВЛЕНИЯ следующего события
(S-map/LWR) — отрицательный результат, θ упёрлась в границу поиска.
Здесь — другой, ретроспективный/структурный вопрос: разворачивается ли
цена статистически чаще на ОПРЕДЕЛЁННЫХ уровнях сетки Δ, чем на других
(аналог находки эксп.20b про объёмную разреженность пивотов, только для
частоты разворота вместо объёма).

Гипотезы:
  H1 (неоднородность) — reversal_rate различается между уровнями сетки
     сильнее, чем предсказывает биномиальный шум при общей средней ставке
     (chi-square goodness-of-fit).
  H2 (устойчивость/exploitability) — "прилипчивость" уровня (высокая/низкая
     reversal_rate) сохраняется во времени: если разбить историю на две
     половины, reversal_rate уровня в первой половине коррелирует с
     reversal_rate ТОГО ЖЕ уровня во второй половине. Это ключевая
     проверка — по опыту эксп.20, ретроспективная неоднородность часто
     не переживает попытку сделать из неё что-то применимое.

Событие/разворот — определения 1-в-1 из эксп.26 (build_level_events,
rev_frac = dirs[1:] != dirs[:-1]), логика уровней — 1-в-1 из
_level_events_raw (app6.py), чтобы не размывать сравнение с прошлым
результатом.

Δ-сетка — рабочий диапазон эксп.26 (events/bar<1 был подтверждён для
SBER; здесь заново проверяется events/bar для всех тикеров, на всякий
случай — риск сетка не универсальна между тикерами с разной волатильностью).

Каузальность: НЕ требуется в том же смысле, что для прогноза — это чисто
ретроспективный структурный анализ (как эксп.20b), не оценка прогнозной
модели. Явно помечено, чтобы не путать с прогнозной честностью.

Запуск (из prototype_analyzers/): python level_reversal_test.py
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

DATA_DIR = _ROOT / "data" / "candles"
RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

INTERVAL = "1d"
TICKERS = ["CHMF", "LKOH", "MGNT", "MRKP", "NLMK", "NVTK", "SBER", "VTBR"]
DELTA_GRID = [0.032, 0.05, 0.08, 0.13, 0.20]  # рабочая сетка эксп.26

MIN_TOUCHES_HETERO = 5   # мин. касаний уровня для теста H1
MIN_TOUCHES_SPLIT = 3    # мин. касаний уровня В КАЖДОЙ половине для H2


def load_ohlc(ticker: str, interval: str) -> dict[str, np.ndarray]:
    path = DATA_DIR / ticker / f"{interval}.json"
    with open(path) as f:
        data = json.load(f)
    return dict(
        lh=np.log(np.array([float(c["high"]) for c in data])),
        ll=np.log(np.array([float(c["low"]) for c in data])),
        lo=np.log(np.array([float(c["open"]) for c in data])),
        lc=np.log(np.array([float(c["close"]) for c in data])),
    )


def build_level_events(lh: np.ndarray, ll: np.ndarray, lo: np.ndarray, lc: np.ndarray,
                       delta: float) -> tuple[np.ndarray, np.ndarray]:
    """1-в-1 логика _level_events_raw (app6.py) / build_level_events (эксп.26).
    Возвращает (levels, dirs)."""
    lp: list[float] = []
    dirs: list[int] = []
    anchor = (lh[0] + ll[0]) / 2.0
    cur_level = round(anchor / delta)
    cur_val = cur_level * delta
    n = len(lh)
    for i in range(n):
        hi, low_, cl, op = lh[i], ll[i], lc[i], lo[i]
        if not (np.isfinite(hi) and np.isfinite(low_) and np.isfinite(cl) and np.isfinite(op)):
            continue
        order = (1, -1) if cl >= op else (-1, 1)
        for step_dir in order:
            if step_dir == 1:
                while hi - cur_val >= delta:
                    cur_level += 1
                    cur_val = cur_level * delta
                    lp.append(cur_val); dirs.append(1)
            else:
                while cur_val - low_ >= delta:
                    cur_level -= 1
                    cur_val = cur_level * delta
                    lp.append(cur_val); dirs.append(-1)
    return np.array(lp), np.array(dirs, dtype=np.int8)


def analyze(levels: np.ndarray, dirs: np.ndarray) -> dict:
    n = len(levels)
    if n < 20:
        return dict(status="too_few_events", n_events=n)

    reversed_ = (dirs[1:] != dirs[:-1]).astype(np.float64)
    lev_r = levels[1:]  # уровень, НА КОТОРОМ произошёл (не-)разворот i-го события
    p_hat = reversed_.mean()

    # --- H1: неоднородность reversal_rate по уровням (chi-square) ---
    uniq_levels, inv, counts = np.unique(lev_r, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=reversed_)
    mask = counts >= MIN_TOUCHES_HETERO
    if mask.sum() < 5:
        chi2_stat, chi2_p, n_levels_used = np.nan, np.nan, int(mask.sum())
    else:
        n_i = counts[mask]; obs = sums[mask]
        expected = n_i * p_hat
        var = n_i * p_hat * (1 - p_hat)
        chi2_stat = float(np.sum((obs - expected) ** 2 / var))
        df = mask.sum() - 1
        chi2_p = float(1 - stats.chi2.cdf(chi2_stat, df))
        n_levels_used = int(mask.sum())

    # --- H2: устойчивость во времени (split-half по индексу события) ---
    half = len(lev_r) // 2
    lev1, rev1 = lev_r[:half], reversed_[:half]
    lev2, rev2 = lev_r[half:], reversed_[half:]

    def per_level_rate(lev, rev, min_touch):
        u, inv2, cnt = np.unique(lev, return_inverse=True, return_counts=True)
        s = np.bincount(inv2, weights=rev)
        rate = np.full(len(u), np.nan)
        ok = cnt >= min_touch
        rate[ok] = s[ok] / cnt[ok]
        return dict(zip(u[ok].tolist(), rate[ok].tolist()))

    r1 = per_level_rate(lev1, rev1, MIN_TOUCHES_SPLIT)
    r2 = per_level_rate(lev2, rev2, MIN_TOUCHES_SPLIT)
    common = sorted(set(r1) & set(r2))
    if len(common) >= 5:
        x = np.array([r1[k] for k in common])
        y = np.array([r2[k] for k in common])
        split_corr = float(np.corrcoef(x, y)[0, 1])
    else:
        split_corr = np.nan

    return dict(
        status="ok", n_events=n, p_hat_reversal=p_hat,
        chi2_stat=chi2_stat, chi2_p=chi2_p, n_levels_hetero=n_levels_used,
        n_common_levels_split=len(common), split_half_corr=split_corr,
    )


def main() -> None:
    rows = []
    for ticker in TICKERS:
        ohlc = load_ohlc(ticker, INTERVAL)
        n_bars = len(ohlc["lh"])
        for delta in DELTA_GRID:
            levels, dirs = build_level_events(ohlc["lh"], ohlc["ll"], ohlc["lo"], ohlc["lc"], delta)
            events_per_bar = len(levels) / n_bars
            res = analyze(levels, dirs)
            res.update(ticker=ticker, delta=delta, n_bars=n_bars, events_per_bar=events_per_bar)
            rows.append(res)
            print(f"{ticker:>5s} Δ={delta:.3f}  events/bar={events_per_bar:.3f}  "
                  f"n_ev={res.get('n_events', 0):>5}  "
                  f"chi2_p={res.get('chi2_p', float('nan')):.4f}  "
                  f"split_corr={res.get('split_half_corr', float('nan')):.3f} "
                  f"(n_common={res.get('n_common_levels_split', 0)})")

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "level_reversal_test.csv"
    df.to_csv(out_path, index=False)

    print("\n=== сводка: доля (тикер,Δ) с chi2_p<0.05 (H1: неоднородность) ===")
    ok = df[df["status"] == "ok"]
    print(f"{(ok['chi2_p'] < 0.05).mean():.2%} из {len(ok)} комбинаций")

    print("\n=== сводка: split-half корреляция (H2: устойчивость) ===")
    print(ok.groupby("delta")["split_half_corr"].agg(["mean", "median", "std", "count"]))

    print(f"\nсохранено: {out_path}")


if __name__ == "__main__":
    main()
