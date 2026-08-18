"""
Общий код для калибраторов: загрузка, зигзаг, пул, метрика.
"""
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parents[3] / "data" / "candles"


def load_log_candles(ticker, interval):
    path = DATA_DIR / ticker / f"{interval}.json"
    with open(path) as f:
        raw = json.load(f)
    lh = np.log(np.array([c["high"] for c in raw], dtype=np.float64))
    ll = np.log(np.array([c["low"]  for c in raw], dtype=np.float64))
    dt = np.array([c["begin"] for c in raw])
    return lh, ll, dt


def build_zigzag(lh, ll, dates, thr):
    lp, conf, dirs = [], [], []
    cur_dir = 0
    ext = (lh[0] + ll[0]) / 2.0
    for i in range(len(lh)):
        if cur_dir == 0:
            if lh[i] - ext >= thr:
                cur_dir = 1; ext = lh[i]
            elif ext - ll[i] >= thr:
                cur_dir = -1; ext = ll[i]
        elif cur_dir == 1:
            if lh[i] > ext:
                ext = lh[i]
            elif ext - ll[i] >= thr:
                lp.append(ext); conf.append(dates[i]); dirs.append(1)
                cur_dir = -1; ext = ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= thr:
                lp.append(ext); conf.append(dates[i]); dirs.append(-1)
                cur_dir = 1; ext = lh[i]
    return np.array(lp), np.array(conf), np.array(dirs, dtype=np.int8)


_zz_cache   = {}
_pool_cache = {}


def clear_cache():
    """Очищает кэш зигзагов и пулов. Вызывать при смене тикера."""
    _zz_cache.clear()
    _pool_cache.clear()


def get_zigzag(lh, ll, dates, thr):
    key = round(thr, 5)
    if key not in _zz_cache:
        _zz_cache[key] = build_zigzag(lh, ll, dates, thr)
    return _zz_cache[key]


def get_pool(lh, ll, dates, thr, m):
    key = (round(thr, 5), m)
    if key in _pool_cache:
        return _pool_cache[key]
    lp, conf, dirs = get_zigzag(lh, ll, dates, thr)
    n    = len(lp)
    vidx = np.arange(m, n - 1)
    feat = np.zeros((len(vidx), m))
    tgt  = np.zeros(len(vidx))
    dar  = np.zeros(len(vidx), dtype=np.int8)
    for row, j in enumerate(vidx):
        for lag in range(m):
            feat[row, lag] = lp[j - lag] - lp[j - lag - 1]
        tgt[row] = lp[j + 1] - lp[j]
        dar[row] = dirs[j]
    econf  = conf[vidx + 1]
    finite = np.all(np.isfinite(feat), axis=1) & np.isfinite(tgt)
    result = (feat[finite], tgt[finite], dar[finite], econf[finite])
    _pool_cache[key] = result
    return result


def rmae(errors, act_diffs):
    if len(errors) < 2:
        return 1.0
    return float(np.mean(errors) / np.mean(act_diffs))


def golden(func, a, b, tol, max_iter=50):
    phi = (5 ** 0.5 - 1) / 2
    x1 = b - phi * (b - a); x2 = a + phi * (b - a)
    f1 = func(x1);           f2 = func(x2)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a); f1 = func(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a); f2 = func(x2)
    return (a + b) / 2
