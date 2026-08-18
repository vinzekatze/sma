#!/usr/bin/env python3
"""
12_calibration_A_smap.py — То же, что calibration_A, но с S-map вместо LWR.

S-map (Sugihara 1994): использует ВСЕ однонаправленные события пула с весами
  w_j = exp(−θ · d_j / mean(d))
θ=0 → глобальная OLS; θ→∞ → концентрация на ближайшем соседе.

θ — непрерывный параметр: оба варианта (T_ratio и θ) оптимизируются
золотым сечением. Нет целочисленных артефактов K.

Параметры поиска:
  m     : {2, 3, 4, 5}  — перебор
  T_ratio: [0.65, 1.0]  — золотое сечение
  θ      : [0, 20]      — золотое сечение
"""
import json
import csv
import numpy as np
from pathlib import Path

# ── параметры ─────────────────────────────────────────────────────────────────
T_BIG       = 0.04
N_CAL       = 100
R_RECAL     = 50
N_MAX       = None
MIN_HISTORY = 50

M_VALUES      = [2, 3, 4, 5]
T_LO, T_HI   = 0.65, 1.0
THETA_LO      = 0.0
THETA_HI      = 20.0
T_TOL         = 0.005
THETA_TOL     = 0.05
MAX_OUTER     = 5
MIN_POOL      = 4     # минимум однонаправленных событий для S-map

DEFAULT_M       = 2
DEFAULT_T_RATIO = 0.85
DEFAULT_THETA   = 2.0

TICKER   = "SBER"
INTERVAL = "10m"

HERE = Path(__file__).parent
DATA = HERE.parent.parent.parent / "data" / "candles"
OUT  = HERE / "results"

_zigzag_cache: dict = {}
_log_highs_g = _log_lows_g = _dates_g = None


# ── загрузка и зигзаг ────────────────────────────────────────────────────────

def load_log_candles(ticker, interval):
    path = DATA / ticker / f"{interval}.json"
    with open(path) as fh:
        raw = json.load(fh)
    log_highs = np.log(np.array([c["high"] for c in raw], dtype=np.float64))
    log_lows  = np.log(np.array([c["low"]  for c in raw], dtype=np.float64))
    dates     = np.array([c["begin"] for c in raw])
    return log_highs, log_lows, dates


def build_zigzag(log_highs, log_lows, dates, threshold):
    lp, conf, dirs = [], [], []
    direction = 0
    extreme   = (log_highs[0] + log_lows[0]) / 2.0
    for bar in range(len(log_highs)):
        if direction == 0:
            if log_highs[bar] - extreme >= threshold:
                direction, extreme = 1, log_highs[bar]
            elif extreme - log_lows[bar] >= threshold:
                direction, extreme = -1, log_lows[bar]
        elif direction == 1:
            if log_highs[bar] > extreme:
                extreme = log_highs[bar]
            elif extreme - log_lows[bar] >= threshold:
                lp.append(extreme); conf.append(dates[bar]); dirs.append(+1)
                direction, extreme = -1, log_lows[bar]
        else:
            if log_lows[bar] < extreme:
                extreme = log_lows[bar]
            elif log_highs[bar] - extreme >= threshold:
                lp.append(extreme); conf.append(dates[bar]); dirs.append(-1)
                direction, extreme = 1, log_highs[bar]
    return np.array(lp), np.array(conf), np.array(dirs, dtype=np.int8)


def get_pool_zigzag(T_ratio):
    T_pool = round(T_ratio * T_BIG, 4)
    if T_pool not in _zigzag_cache:
        _zigzag_cache[T_pool] = build_zigzag(_log_highs_g, _log_lows_g, _dates_g, T_pool)
    return _zigzag_cache[T_pool]


# ── один шаг S-map ────────────────────────────────────────────────────────────

def smap_step(query_lp, query_dirs, step_i, pool_lp, pool_conf, pool_dirs,
              conf_date, m, theta, n_max):
    """Один S-map прогноз. Возвращает predicted_price или None."""
    j_max = int(np.searchsorted(pool_conf, conf_date, side='left'))
    valid = np.arange(m, j_max - 1)
    if len(valid) == 0:
        return None
    if n_max and len(valid) > n_max:
        valid = valid[-n_max:]

    feat = np.zeros((len(valid), m))
    tgt  = np.zeros(len(valid))
    dirs = np.zeros(len(valid), dtype=np.int8)
    for row, j in enumerate(valid):
        for lag in range(m):
            feat[row, lag] = pool_lp[j - lag] - pool_lp[j - lag - 1]
        tgt[row]  = pool_lp[j + 1] - pool_lp[j]
        dirs[row] = pool_dirs[j]

    ok = np.all(np.isfinite(feat), axis=1) & np.isfinite(tgt)
    feat, tgt, dirs = feat[ok], tgt[ok], dirs[ok]

    qdir = int(query_dirs[step_i])
    qvec = np.array([query_lp[step_i - lag] - query_lp[step_i - lag - 1]
                     for lag in range(m)])

    mask = dirs == qdir
    if mask.sum() < MIN_POOL:
        return None

    cf, ct = feat[mask], tgt[mask]
    dists     = np.linalg.norm(cf - qvec, axis=1)
    mean_dist = dists.mean()

    if mean_dist < 1e-14:
        lr = float(ct.mean())
    elif theta == 0:
        weights = np.ones(len(ct))
        sw  = np.sqrt(weights)
        A   = np.column_stack([np.ones(len(ct)), cf]) * sw[:, None]
        c, *_ = np.linalg.lstsq(A, ct * sw, rcond=None)
        lr  = float(c[0] + c[1:] @ qvec)
    else:
        weights = np.exp(-theta * dists / mean_dist)
        sw  = np.sqrt(weights)
        A   = np.column_stack([np.ones(len(ct)), cf]) * sw[:, None]
        c, *_ = np.linalg.lstsq(A, ct * sw, rcond=None)
        lr  = float(c[0] + c[1:] @ qvec)

    return float(np.exp(query_lp[step_i] + lr))


# ── оценка параметров на окне ─────────────────────────────────────────────────

def eval_params(query_lp, query_conf, query_dirs,
                cal_start, cal_end, m, T_ratio, theta):
    pool_lp, pool_conf, pool_dirs = get_pool_zigzag(T_ratio)
    n = len(query_lp)
    preds, actuals = [], []
    for i in range(cal_start, min(cal_end, n - 1)):
        pred = smap_step(query_lp, query_dirs, i, pool_lp, pool_conf, pool_dirs,
                         query_conf[i], m, theta, N_MAX)
        if pred is None:
            continue
        preds.append(pred)
        actuals.append(float(np.exp(query_lp[i + 1])))
    if len(preds) < 5:
        return np.inf
    errs = np.abs(np.array(preds) - np.array(actuals))
    a    = np.asarray(actuals, dtype=float)
    pers = np.abs(a[2:] - a[:-2]) if len(a) >= 3 else np.abs(np.diff(a))
    dz   = float(np.mean(pers)) if len(pers) > 0 else 1.0
    return float(errs.mean() / dz) if dz > 1e-12 else np.inf


# ── золотое сечение ───────────────────────────────────────────────────────────

def golden_section(func, a, b, tol, max_iter=40):
    phi = (np.sqrt(5) - 1) / 2
    x1, x2 = b - phi * (b - a), a + phi * (b - a)
    f1, f2  = func(x1), func(x2)
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


# ── покоординатный спуск ─────────────────────────────────────────────────────

def coordinate_descent(query_lp, query_conf, query_dirs,
                       cal_start, cal_end,
                       start_m, start_T_ratio, start_theta,
                       verbose=True):
    """
    Порядок: θ → T_ratio → m (повторяем до сходимости).
    Оба непрерывных параметра — золотое сечение.
    """
    m, T_ratio, theta = start_m, float(start_T_ratio), float(start_theta)
    n_evals = 0

    def obj(mm, tr, th):
        nonlocal n_evals
        n_evals += 1
        return eval_params(query_lp, query_conf, query_dirs,
                           cal_start, cal_end, mm, tr, th)

    for outer in range(MAX_OUTER):
        m_prev, T_prev, th_prev = m, T_ratio, theta

        # 1. Оптимизация θ (золотое сечение)
        theta = golden_section(lambda th: obj(m, T_ratio, th),
                               THETA_LO, THETA_HI, THETA_TOL)

        # 2. Оптимизация T_ratio (золотое сечение)
        T_ratio = golden_section(lambda t: obj(m, t, theta),
                                 T_LO, T_HI, T_TOL)

        # 3. Оптимизация m (перебор)
        m = min(M_VALUES, key=lambda mm: obj(mm, T_ratio, theta))

        best_r = obj(m, T_ratio, theta)

        if verbose:
            print(f"    iter {outer+1}: m={m}  T_ratio={T_ratio:.4f}  "
                  f"θ={theta:.3f}  rMAE={best_r:.4f}  (evals={n_evals})")

        converged = (m == m_prev
                     and abs(T_ratio - T_prev) < T_TOL
                     and abs(theta - th_prev) < THETA_TOL)
        if converged:
            break

    return m, T_ratio, theta, n_evals, best_r


# ── walk-forward с перекалибровкой ───────────────────────────────────────────

def walk_forward_adaptive(query_lp, query_conf, query_dirs):
    n = len(query_lp)
    cur_m, cur_T_ratio, cur_theta = DEFAULT_M, DEFAULT_T_RATIO, DEFAULT_THETA

    results_adaptive = []
    results_default  = []
    params_log       = []

    first_cal_end = MIN_HISTORY + N_CAL
    if first_cal_end < n - 1:
        print(f"\n── Калибровка 0: шаги [{MIN_HISTORY}, {first_cal_end}) ──")
        cur_m, cur_T_ratio, cur_theta, evals, rmae_cal = coordinate_descent(
            query_lp, query_conf, query_dirs,
            MIN_HISTORY, first_cal_end,
            cur_m, cur_T_ratio, cur_theta,
        )
        params_log.append(dict(
            recal_idx=0, cal_start=MIN_HISTORY, cal_end=first_cal_end,
            m=cur_m, T_ratio=round(cur_T_ratio, 4), theta=round(cur_theta, 3),
            n_evals=evals, rmae_cal=round(rmae_cal, 4),
        ))

    forecast_start = first_cal_end
    next_recal     = forecast_start + R_RECAL
    recal_idx      = 1

    def_pool_lp, def_pool_conf, def_pool_dirs = get_pool_zigzag(DEFAULT_T_RATIO)

    for i in range(forecast_start, n - 1):
        if i >= next_recal:
            cal_start = max(MIN_HISTORY, i - N_CAL)
            cal_end   = i
            print(f"\n── Калибровка {recal_idx}: шаги [{cal_start}, {cal_end}) ──")
            cur_m, cur_T_ratio, cur_theta, evals, rmae_cal = coordinate_descent(
                query_lp, query_conf, query_dirs,
                cal_start, cal_end,
                cur_m, cur_T_ratio, cur_theta,
            )
            params_log.append(dict(
                recal_idx=recal_idx, cal_start=cal_start, cal_end=cal_end,
                m=cur_m, T_ratio=round(cur_T_ratio, 4), theta=round(cur_theta, 3),
                n_evals=evals, rmae_cal=round(rmae_cal, 4),
            ))
            next_recal = i + R_RECAL
            recal_idx += 1

        conf_date = query_conf[i]
        actual    = float(np.exp(query_lp[i + 1]))

        adp_pool_lp, adp_pool_conf, adp_pool_dirs = get_pool_zigzag(cur_T_ratio)
        pred_adp = smap_step(query_lp, query_dirs, i,
                             adp_pool_lp, adp_pool_conf, adp_pool_dirs,
                             conf_date, cur_m, cur_theta, N_MAX)

        pred_def = smap_step(query_lp, query_dirs, i,
                             def_pool_lp, def_pool_conf, def_pool_dirs,
                             conf_date, DEFAULT_M, DEFAULT_THETA, N_MAX)

        if pred_adp is not None:
            results_adaptive.append(dict(
                step=i, date=conf_date, actual=actual, predicted=pred_adp,
                error=pred_adp - actual,
                m=cur_m, T_ratio=round(cur_T_ratio, 4), theta=round(cur_theta, 3),
            ))
        if pred_def is not None:
            results_default.append(dict(
                step=i, date=conf_date, actual=actual, predicted=pred_def,
                error=pred_def - actual,
            ))

    return results_adaptive, results_default, params_log


def rmae(rows):
    if len(rows) < 5:
        return np.nan
    errs    = np.abs(np.array([r["error"]  for r in rows]))
    actuals = np.array([r["actual"] for r in rows])
    pers = np.abs(actuals[2:] - actuals[:-2]) if len(actuals) >= 3 else np.abs(np.diff(actuals))
    dz = float(np.mean(pers)) if len(pers) > 0 else 1.0
    return float(errs.mean() / dz) if dz > 1e-12 else np.nan


def main():
    global _log_highs_g, _log_lows_g, _dates_g

    print(f"=== 12_calibration_A_smap  SBER {INTERVAL}  T_big={T_BIG*100:.0f}%  "
          f"N_cal={N_CAL}  R={R_RECAL}  θ∈[{THETA_LO},{THETA_HI}] ===\n")

    log_highs, log_lows, dates = load_log_candles(TICKER, INTERVAL)
    _log_highs_g, _log_lows_g, _dates_g = log_highs, log_lows, dates
    print(f"Свечей: {len(dates)}  ({dates[0][:10]} … {dates[-1][:10]})")

    query_lp, query_conf, query_dirs = build_zigzag(log_highs, log_lows, dates, T_BIG)
    print(f"T_query={T_BIG*100:.0f}%: {len(query_lp)} пивотов\n")

    results_adp, results_def, params_log = walk_forward_adaptive(
        query_lp, query_conf, query_dirs
    )

    r_adp = rmae(results_adp)
    r_def = rmae(results_def)

    print(f"\n{'─'*60}")
    print(f"S-map дефолт (m={DEFAULT_M} T={DEFAULT_T_RATIO} θ={DEFAULT_THETA}): "
          f"rMAE={r_def:.4f}  (n={len(results_def)})")
    print(f"S-map адаптив (скользящая калибровка):          "
          f"rMAE={r_adp:.4f}  (n={len(results_adp)})")
    delta = (r_adp / r_def - 1) * 100
    print(f"Изменение: {delta:+.1f}%  ({'улучшение' if delta < 0 else 'ухудшение'})")
    print(f"{'─'*60}")

    print(f"\n{'─'*60}")
    print(f"{'#':>3}  {'cal':>10}  {'m':>2}  {'T_ratio':>7}  {'θ':>6}  "
          f"{'evals':>6}  {'rMAE_cal':>8}")
    for p in params_log:
        print(f"{p['recal_idx']:>3}  "
              f"[{p['cal_start']:>3},{p['cal_end']:>3})  "
              f"{p['m']:>2}  {p['T_ratio']:>7.4f}  {p['theta']:>6.3f}  "
              f"{p['n_evals']:>6}  {p['rmae_cal']:>8.4f}")

    print(f"\nЗигзагов в кэше: {len(_zigzag_cache)}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "smap_adaptive.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results_adp[0].keys())
        w.writeheader(); w.writerows(results_adp)
    with open(OUT / "smap_default.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results_def[0].keys())
        w.writeheader(); w.writerows(results_def)
    with open(OUT / "smap_params_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=params_log[0].keys())
        w.writeheader(); w.writerows(params_log)

    print(f"CSV → {OUT}/  (smap_adaptive, smap_default, smap_params_log)")


if __name__ == "__main__":
    main()
