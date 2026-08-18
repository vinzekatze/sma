#!/usr/bin/env python3
"""
eval_last_n.py — rMAE на последних N событиях через каузализацию ref-стиля.

Для каждого из последних N пивотов T_BIG:
  1. Обрезаем свечи до confirm_date[step] включительно  ← единственная точка каузальности
  2. Строим зигзаги T_BIG и T_pool на обрезанных данных
  3. Прогноз LWR и S-map с заданными параметрами
  4. actual = следующий T_BIG пивот (из полного зигзага, уже известен)

Вычисляет rMAE для каждого метода по окнам [10, 25, 50].

Использование:
  python eval_last_n.py --config results/calibration_SBER_10m_T001.json
  python eval_last_n.py --ticker SBER --interval 10m --t-big 0.01 \\
      --lwr-m 3 --lwr-k 400 --lwr-tratio 0.8158 \\
      --smap-m 4 --smap-theta 2.825 --smap-tratio 0.8175
"""
import sys
import json
import argparse
import numpy as np
from pathlib import Path

_REF = Path(__file__).parents[3] / "research" / "reference"
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))
from lwr_ref  import build_zigzag, build_pool_vectors, build_query_vector, \
                      find_neighbors, lwr_predict
from smap_ref import filter_direction_pool, smap_predict

DATA_DIR = Path(__file__).parents[3] / "data" / "candles"
WINDOWS  = [10, 25, 50]


def load_candles(ticker, interval):
    path = DATA_DIR / ticker / f"{interval}.json"
    with open(path) as f:
        raw = json.load(f)
    lh = np.log(np.array([c["high"] for c in raw], dtype=np.float64))
    ll = np.log(np.array([c["low"]  for c in raw], dtype=np.float64))
    dt = np.array([c["begin"] for c in raw])
    return lh, ll, dt


def forecast_lwr(lh_cut, ll_cut, dt_cut, t_big, t_pool, m, K):
    """Прогноз LWR на обрезанных данных. Возвращает (pred_price, ok)."""
    qlp, _, qdirs = build_zigzag(lh_cut, ll_cut, dt_cut, t_big)
    plp, _, pdirs = build_zigzag(lh_cut, ll_cut, dt_cut, t_pool)
    if len(qlp) < m + 1:
        return None, False
    pfm, ptgt, pdir = build_pool_vectors(plp, pdirs, m)
    qvec  = build_query_vector(qlp, m)
    qdir  = int(qdirs[-1])
    nn    = find_neighbors(qvec, pfm, pdir, qdir, K)
    if nn is None:
        return None, False
    lr    = lwr_predict(qvec, pfm, ptgt, nn)
    return float(np.exp(qlp[-1] + lr)), True


def forecast_smap(lh_cut, ll_cut, dt_cut, t_big, t_pool, m, theta, min_pool):
    """Прогноз S-map на обрезанных данных. Возвращает (pred_price, ok)."""
    qlp, _, qdirs = build_zigzag(lh_cut, ll_cut, dt_cut, t_big)
    plp, _, pdirs = build_zigzag(lh_cut, ll_cut, dt_cut, t_pool)
    if len(qlp) < m + 1:
        return None, False
    pfm, ptgt, pdir = build_pool_vectors(plp, pdirs, m)
    qvec  = build_query_vector(qlp, m)
    qdir  = int(qdirs[-1])
    df, dt_ = filter_direction_pool(pfm, ptgt, pdir, qdir, min_pool)
    if df is None:
        return None, False
    lr    = smap_predict(qvec, df, dt_, theta)
    return float(np.exp(qlp[-1] + lr)), True


def run_eval(ticker, interval, t_big,
             lwr_m, lwr_K, lwr_tratio,
             smap_m, smap_theta, smap_tratio):

    lh, ll, dt = load_candles(ticker, interval)
    print(f"Свечей: {len(dt)}  ({dt[0][:10]} … {dt[-1][:10]})")

    # Полный зигзаг для получения confirm_dates и actual цен
    qlp_full, qconf_full, _ = build_zigzag(lh, ll, dt, t_big)
    n = len(qlp_full)
    print(f"T_BIG={t_big*100:.1f}%  пивотов: {n}")

    max_win = max(WINDOWS)
    # Шаги i = n-max_win-1 .. n-2  (для i нужен actual = qlp_full[i+1])
    eval_steps = list(range(n - max_win - 1, n - 1))

    t_pool_lwr  = lwr_tratio  * t_big
    t_pool_smap = smap_tratio * t_big
    min_pool_s  = max(smap_m + 2, 4)

    lwr_errors, smap_errors, pers_errors = [], [], []

    for step in eval_steps:
        confirm = qconf_full[step]
        # Обрезка: оставляем свечи с begin <= confirm_date[step]
        cut = int(np.searchsorted(dt, confirm, side="right"))
        lh_c, ll_c, dt_c = lh[:cut], ll[:cut], dt[:cut]

        actual_price = float(np.exp(qlp_full[step + 1]))
        # Persistence naive: следующий пивот = текущий^2 / предыдущий (в лог-пространстве: 2·lp[i] − lp[i−1])
        pers_pred    = float(np.exp(qlp_full[step - 1]))

        pred_lwr, ok_lwr = forecast_lwr(lh_c, ll_c, dt_c,
                                         t_big, t_pool_lwr, lwr_m, lwr_K)
        pred_smap, ok_sm = forecast_smap(lh_c, ll_c, dt_c,
                                          t_big, t_pool_smap,
                                          smap_m, smap_theta, min_pool_s)

        lwr_errors.append(abs(pred_lwr   - actual_price) if ok_lwr else None)
        smap_errors.append(abs(pred_smap  - actual_price) if ok_sm  else None)
        pers_errors.append(abs(pers_pred  - actual_price))

    print(f"\n{'Окно':>6}  {'LWR rMAE':>10}  {'S-map rMAE':>12}  "
          f"{'n_lwr':>6}  {'n_smap':>7}")
    print("-" * 52)

    results = {}
    for win in WINDOWS:
        # последние win ошибок из eval_steps — знаменатель из того же окна
        lwr_w  = [e for e in lwr_errors[-win:]  if e is not None]
        smap_w = [e for e in smap_errors[-win:] if e is not None]
        pers_w = pers_errors[-win:]

        dz = float(np.mean(pers_w)) if pers_w else 1.0
        r_lwr  = float(np.mean(lwr_w))  / dz if lwr_w  else float("nan")
        r_smap = float(np.mean(smap_w)) / dz if smap_w else float("nan")

        print(f"{win:>6}  {r_lwr:>10.4f}  {r_smap:>12.4f}  "
              f"{len(lwr_w):>6}  {len(smap_w):>7}")
        results[win] = {"lwr": round(r_lwr, 4), "smap": round(r_smap, 4),
                        "n_lwr": len(lwr_w), "n_smap": len(smap_w)}

    return results


def main():
    parser = argparse.ArgumentParser(
        description="rMAE на последних N событиях (каузализация ref-стиля)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None,
                        help="JSON калибровки (calibration_*.json); перекрывает ручные параметры")
    parser.add_argument("--ticker",     default="SBER")
    parser.add_argument("--interval",   default="10m")
    parser.add_argument("--t-big",      type=float, default=0.01)
    parser.add_argument("--lwr-m",      type=int,   default=3)
    parser.add_argument("--lwr-k",      type=int,   default=400)
    parser.add_argument("--lwr-tratio", type=float, default=0.8158)
    parser.add_argument("--smap-m",     type=int,   default=4)
    parser.add_argument("--smap-theta", type=float, default=2.825)
    parser.add_argument("--smap-tratio",type=float, default=0.8175)
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        ticker   = cfg["ticker"]
        interval = cfg["interval"]
        t_big    = cfg["T_big"]
        lwr_m    = cfg["lwr"]["optimal"]["m"]
        lwr_K    = cfg["lwr"]["optimal"]["K"]
        lwr_tr   = cfg["lwr"]["optimal"]["T_ratio"]
        sm_m     = cfg["smap"]["optimal"]["m"]
        sm_th    = cfg["smap"]["optimal"]["theta"]
        sm_tr    = cfg["smap"]["optimal"]["T_ratio"]
    else:
        ticker   = args.ticker
        interval = args.interval
        t_big    = args.t_big
        lwr_m    = args.lwr_m
        lwr_K    = args.lwr_k
        lwr_tr   = args.lwr_tratio
        sm_m     = args.smap_m
        sm_th    = args.smap_theta
        sm_tr    = args.smap_tratio

    print(f"=== EVAL LAST N: {ticker} {interval} T={t_big*100:.1f}% ===")
    print(f"LWR:   m={lwr_m}  K={lwr_K}  T_pool={lwr_tr*t_big*100:.4f}%")
    print(f"S-map: m={sm_m}  θ={sm_th}  T_pool={sm_tr*t_big*100:.4f}%")

    results = run_eval(ticker, interval, t_big,
                       lwr_m, lwr_K, lwr_tr,
                       sm_m, sm_th, sm_tr)

    out_path = Path(__file__).parent / "results" / \
               f"eval_last_n_{ticker}_{interval}_T{round(t_big*100):03d}.json"
    with open(out_path, "w") as f:
        json.dump({"ticker": ticker, "interval": interval, "t_big": t_big,
                   "lwr_params": {"m": lwr_m, "K": lwr_K, "T_ratio": lwr_tr},
                   "smap_params": {"m": sm_m, "theta": sm_th, "T_ratio": sm_tr},
                   "windows": results}, f, indent=2)
    print(f"\nСохранено: {out_path}")


if __name__ == "__main__":
    main()
