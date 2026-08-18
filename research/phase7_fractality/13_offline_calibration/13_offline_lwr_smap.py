#!/usr/bin/env python3
"""
13_offline_lwr_smap.py — Офлайн-калибровка и сравнение LWR vs S-map.

Каузальный контракт
───────────────────
На шаге i доступны только события пула где подтверждён следующий пивот:
    pool_conf[j+1] < query_conf[i]
Зигзаги строятся на всех данных один раз, каузальность — через фильтр дат.
"""

import json
import time
import numpy as np
from pathlib import Path

# ── Константы ─────────────────────────────────────────────────────────────────
T_BIG         = 0.04
DATA_PATH     = Path(__file__).parents[3] / "data/candles/SBER/10m.json"
MAX_OUTER     = 5

M_VALUES      = [2, 3, 4, 5]
K_LO, K_HI   = 10, 300
T_LO, T_HI   = 0.65, 1.0
T_TOL         = 0.005

THETA_LO, THETA_HI = 0.0, 20.0
THETA_TOL     = 0.1

DEF_M         = 2
DEF_K         = 75
DEF_THETA     = 2.0
DEF_T_RATIO   = 0.85


# ── Загрузка ──────────────────────────────────────────────────────────────────

def load_log_candles(path):
    with open(path) as f:
        raw = json.load(f)
    lh = np.log(np.array([c["high"] for c in raw], dtype=np.float64))
    ll = np.log(np.array([c["low"]  for c in raw], dtype=np.float64))
    dt = np.array([c["begin"] for c in raw])
    return lh, ll, dt


# ── Зигзаг ────────────────────────────────────────────────────────────────────

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


# ── Кэш ───────────────────────────────────────────────────────────────────────

_zz_cache   = {}  # thr_key → (lp, conf, dirs)
_pool_cache = {}  # (thr_key, m) → (feat, tgt, dir_arr, event_conf)


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
    # event_conf = когда цель lp[j+1] стала известна (подтверждение j+1)
    econf  = conf[vidx + 1]
    finite = np.all(np.isfinite(feat), axis=1) & np.isfinite(tgt)
    result = (feat[finite], tgt[finite], dar[finite], econf[finite])
    _pool_cache[key] = result
    return result


# ── Метрика ───────────────────────────────────────────────────────────────────

def rmae(errors, act_diffs):
    if len(errors) < 2:
        return 1.0
    return float(np.mean(errors) / np.mean(act_diffs))


# ── Walk-forward: LWR ─────────────────────────────────────────────────────────

def eval_lwr(qlp, qconf, qdirs, lh, ll, dates, m, K, T_ratio):
    pf, pt, pd, peconf = get_pool(lh, ll, dates, T_ratio * T_BIG, m)
    errors, act_diffs  = [], []

    for i in range(m, len(qlp) - 1):
        causal = peconf < qconf[i]
        if causal.sum() < K:
            continue
        pfc, ptc, pdc = pf[causal], pt[causal], pd[causal]

        qvec  = np.array([qlp[i - lag] - qlp[i - lag - 1] for lag in range(m)])
        dmask = pdc == int(qdirs[i])
        if dmask.sum() < K:
            continue

        pfd, ptd  = pfc[dmask], ptc[dmask]
        dists     = np.linalg.norm(pfd - qvec, axis=1)
        nn        = np.argpartition(dists, K - 1)[:K]
        d_nn, f_nn, t_nn = dists[nn], pfd[nn], ptd[nn]
        d_max     = d_nn.max()

        if d_max < 1e-12:
            lr = float(t_nn.mean())
        else:
            w   = np.exp(-0.5 * (d_nn / d_max) ** 2)
            sw  = np.sqrt(w)
            A   = np.column_stack([np.ones(K), f_nn]) * sw[:, None]
            b   = t_nn * sw
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
            lr  = float(c[0] + c[1:] @ qvec)

        errors.append(abs(np.exp(qlp[i] + lr) - np.exp(qlp[i + 1])))
        act_diffs.append(abs(np.exp(qlp[i + 1]) - np.exp(qlp[i])))

    return rmae(errors, act_diffs), len(errors)


# ── Walk-forward: S-map ───────────────────────────────────────────────────────

def eval_smap(qlp, qconf, qdirs, lh, ll, dates, m, theta, T_ratio):
    pf, pt, pd, peconf = get_pool(lh, ll, dates, T_ratio * T_BIG, m)
    min_pool = max(m + 2, 4)
    errors, act_diffs  = [], []

    for i in range(m, len(qlp) - 1):
        causal = peconf < qconf[i]
        pfc, ptc, pdc = pf[causal], pt[causal], pd[causal]

        qvec  = np.array([qlp[i - lag] - qlp[i - lag - 1] for lag in range(m)])
        dmask = pdc == int(qdirs[i])
        if dmask.sum() < min_pool:
            continue

        pfd, ptd = pfc[dmask], ptc[dmask]
        dists    = np.linalg.norm(pfd - qvec, axis=1)
        mean_d   = dists.mean()

        if mean_d < 1e-14:
            lr = float(ptd.mean())
        else:
            w  = np.ones(len(ptd)) if theta == 0 else np.exp(-theta * dists / mean_d)
            sw = np.sqrt(w)
            A  = np.column_stack([np.ones(len(ptd)), pfd]) * sw[:, None]
            b  = ptd * sw
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
            lr = float(c[0] + c[1:] @ qvec)

        errors.append(abs(np.exp(qlp[i] + lr) - np.exp(qlp[i + 1])))
        act_diffs.append(abs(np.exp(qlp[i + 1]) - np.exp(qlp[i])))

    return rmae(errors, act_diffs), len(errors)


# ── Оптимизаторы ──────────────────────────────────────────────────────────────

def ternary_int(func, lo, hi, max_iter=14):
    while hi - lo > 2 and max_iter > 0:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3
        if func(m1) <= func(m2):
            hi = m2
        else:
            lo = m1
        max_iter -= 1
    best = lo; best_v = func(lo)
    for k in range(lo + 1, hi + 1):
        v = func(k)
        if v < best_v:
            best_v = v; best = k
    return best


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


# ── Калибровка LWR ────────────────────────────────────────────────────────────

def calibrate_lwr(qlp, qconf, qdirs, lh, ll, dates):
    m, K, T = DEF_M, DEF_K, DEF_T_RATIO
    print(f"\n[LWR] Старт: m={m}, K={K}, T_ratio={T:.3f}")
    trace = []
    prev  = None

    for outer in range(MAX_OUTER):
        best_m = m; best_v = float('inf')
        for mc in M_VALUES:
            v, _ = eval_lwr(qlp, qconf, qdirs, lh, ll, dates, mc, K, T)
            if v < best_v:
                best_v = v; best_m = mc
        m = best_m

        K = ternary_int(
            lambda k: eval_lwr(qlp, qconf, qdirs, lh, ll, dates, m, k, T)[0],
            K_LO, K_HI,
        )

        T = golden(
            lambda t: eval_lwr(qlp, qconf, qdirs, lh, ll, dates, m, K, t)[0],
            T_LO, T_HI, T_TOL,
        )

        v, n = eval_lwr(qlp, qconf, qdirs, lh, ll, dates, m, K, T)
        row  = {'iter': outer + 1, 'm': m, 'K': K,
                'T_ratio': round(T, 4), 'rMAE': round(v, 4)}
        trace.append(row)
        print(f"  Iter {outer+1}: m={m}, K={K}, T_ratio={T:.4f} → rMAE={v:.4f}  (n={n})")

        cur = (m, K, round(T, 3))
        if cur == prev:
            print("  Сошлось.")
            break
        prev = cur

    return m, K, T, v, n, trace


# ── Калибровка S-map ──────────────────────────────────────────────────────────

def calibrate_smap(qlp, qconf, qdirs, lh, ll, dates):
    m, theta, T = DEF_M, DEF_THETA, DEF_T_RATIO
    print(f"\n[S-map] Старт: m={m}, θ={theta}, T_ratio={T:.3f}")
    trace = []
    prev  = None

    for outer in range(MAX_OUTER):
        best_m = m; best_v = float('inf')
        for mc in M_VALUES:
            v, _ = eval_smap(qlp, qconf, qdirs, lh, ll, dates, mc, theta, T)
            if v < best_v:
                best_v = v; best_m = mc
        m = best_m

        theta = golden(
            lambda th: eval_smap(qlp, qconf, qdirs, lh, ll, dates, m, th, T)[0],
            THETA_LO, THETA_HI, THETA_TOL,
        )

        T = golden(
            lambda t: eval_smap(qlp, qconf, qdirs, lh, ll, dates, m, theta, t)[0],
            T_LO, T_HI, T_TOL,
        )

        v, n = eval_smap(qlp, qconf, qdirs, lh, ll, dates, m, theta, T)
        row  = {'iter': outer + 1, 'm': m, 'theta': round(theta, 3),
                'T_ratio': round(T, 4), 'rMAE': round(v, 4)}
        trace.append(row)
        print(f"  Iter {outer+1}: m={m}, θ={theta:.3f}, T_ratio={T:.4f} → rMAE={v:.4f}  (n={n})")

        cur = (m, round(theta, 2), round(T, 3))
        if cur == prev:
            print("  Сошлось.")
            break
        prev = cur

    return m, theta, T, v, n, trace


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global T_BIG, K_LO, K_HI

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--t-big",  type=float, default=T_BIG,  help="Порог зигзага запроса")
    parser.add_argument("--k-hi",   type=int,   default=K_HI,   help="Верхняя граница K для LWR")
    parser.add_argument("--k-lo",   type=int,   default=K_LO,   help="Нижняя граница K для LWR")
    args = parser.parse_args()

    T_BIG = args.t_big
    K_LO  = args.k_lo
    K_HI  = args.k_hi

    print("=== OFFLINE CALIBRATION: LWR vs S-map ===")
    lh, ll, dates = load_log_candles(DATA_PATH)
    print(f"Свечей: {len(dates)}  ({dates[0][:10]} … {dates[-1][:10]})")

    qlp, qconf, qdirs = build_zigzag(lh, ll, dates, T_BIG)
    print(f"T_query={T_BIG*100:.0f}%:  {len(qlp)} пивотов")

    # ── Дефолты ──────────────────────────────────────────────────────────────
    print("\n── Дефолтные параметры ──")
    t0 = time.time()
    lwr_def,  lwr_def_n  = eval_lwr(qlp, qconf, qdirs, lh, ll, dates,
                                     DEF_M, DEF_K, DEF_T_RATIO)
    smap_def, smap_def_n = eval_smap(qlp, qconf, qdirs, lh, ll, dates,
                                      DEF_M, DEF_THETA, DEF_T_RATIO)
    print(f"LWR   m={DEF_M}  K={DEF_K}   T_ratio={DEF_T_RATIO}  → rMAE={lwr_def:.4f}  (n={lwr_def_n})")
    print(f"S-map m={DEF_M}  θ={DEF_THETA}  T_ratio={DEF_T_RATIO}  → rMAE={smap_def:.4f}  (n={smap_def_n})")
    print(f"Время: {time.time()-t0:.1f}s")

    # ── Оптимизация LWR ──────────────────────────────────────────────────────
    print("\n── LWR: координатный спуск ──")
    t0 = time.time()
    lwr_m, lwr_K, lwr_T, lwr_opt, lwr_opt_n, lwr_trace = calibrate_lwr(
        qlp, qconf, qdirs, lh, ll, dates,
    )
    print(f"Время: {time.time()-t0:.1f}s  |  Зигзагов в кэше: {len(_zz_cache)}")

    # ── Оптимизация S-map ────────────────────────────────────────────────────
    print("\n── S-map: координатный спуск ──")
    t0 = time.time()
    sm_m, sm_theta, sm_T, sm_opt, sm_opt_n, sm_trace = calibrate_smap(
        qlp, qconf, qdirs, lh, ll, dates,
    )
    print(f"Время: {time.time()-t0:.1f}s  |  Зигзагов в кэше: {len(_zz_cache)}")

    # ── Итог ─────────────────────────────────────────────────────────────────
    lwr_delta  = (lwr_opt  - lwr_def)  / lwr_def  * 100
    smap_delta = (sm_opt   - smap_def) / smap_def * 100

    print("\n══ ИТОГ ══")
    print(f"{'':22s} {'default':>10} {'optimal':>10} {'Δ%':>8}")
    print(f"{'LWR':22s} {lwr_def:10.4f} {lwr_opt:10.4f} {lwr_delta:8.1f}%")
    print(f"{'S-map':22s} {smap_def:10.4f} {sm_opt:10.4f} {smap_delta:8.1f}%")
    print()
    print(f"LWR   optimal: m={lwr_m},  K={lwr_K},      T_ratio={lwr_T:.4f}")
    print(f"S-map optimal: m={sm_m},   θ={sm_theta:.3f},  T_ratio={sm_T:.4f}")

    # ── Сохранение ───────────────────────────────────────────────────────────
    result = {
        "ticker": "SBER", "interval": "10m", "T_big": T_BIG,
        "n_query_pivots": int(len(qlp)),
        "lwr": {
            "default": {"m": DEF_M, "K": DEF_K, "T_ratio": DEF_T_RATIO,
                        "rMAE": round(lwr_def, 4), "n_steps": lwr_def_n},
            "optimal": {"m": lwr_m, "K": lwr_K, "T_ratio": round(lwr_T, 4),
                        "rMAE": round(lwr_opt, 4), "n_steps": lwr_opt_n,
                        "delta_pct": round(lwr_delta, 2)},
            "trace": lwr_trace,
        },
        "smap": {
            "default": {"m": DEF_M, "theta": DEF_THETA, "T_ratio": DEF_T_RATIO,
                        "rMAE": round(smap_def, 4), "n_steps": smap_def_n},
            "optimal": {"m": sm_m, "theta": round(sm_theta, 3), "T_ratio": round(sm_T, 4),
                        "rMAE": round(sm_opt, 4), "n_steps": sm_opt_n,
                        "delta_pct": round(smap_delta, 2)},
            "trace": sm_trace,
        },
    }
    t_tag = f"T{round(T_BIG * 100):03d}"
    out = Path(__file__).parent / "results" / f"calibration_SBER_10m_{t_tag}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nРезультат: {out}")


if __name__ == "__main__":
    main()
