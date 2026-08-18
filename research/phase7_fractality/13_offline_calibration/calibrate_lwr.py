#!/usr/bin/env python3
"""
calibrate_lwr.py — Офлайн-калибровка LWR для зигзаг-прогноза.

Оптимизирует (m, K, T_ratio) покоординатным спуском на полной истории.

Использование:
  python calibrate_lwr.py --ticker SBER --interval 10m --t-big 0.04
  python calibrate_lwr.py --ticker GAZP --interval 1d  --t-big 0.03 --k-hi 500

Каузальный контракт: событие пула добавляется только когда известна его цель,
т.е. подтверждён следующий пивот: pool_conf[j+1] < query_conf[i].
"""
import json
import time
import argparse
import numpy as np
from pathlib import Path

from _core import load_log_candles, get_pool, rmae, golden

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Дефолты ───────────────────────────────────────────────────────────────────
M_VALUES    = [2, 3, 4, 5]
K_LO        = 10
K_HI_DEF    = 300
T_LO, T_HI  = 0.65, 1.0
T_TOL       = 0.005
MAX_OUTER   = 5

DEF_M       = 2
DEF_K       = 75
DEF_T_RATIO = 0.85


# ── Walk-forward LWR ──────────────────────────────────────────────────────────

def eval_lwr(qlp, qconf, qdirs, lh, ll, dates, t_big, m, K, T_ratio):
    pf, pt, pd, peconf = get_pool(lh, ll, dates, T_ratio * t_big, m)
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

        pfd, ptd = pfc[dmask], ptc[dmask]
        dists    = np.linalg.norm(pfd - qvec, axis=1)
        nn       = np.argpartition(dists, K - 1)[:K]
        d_nn, f_nn, t_nn = dists[nn], pfd[nn], ptd[nn]
        d_max    = d_nn.max()

        if d_max < 1e-12:
            lr = float(t_nn.mean())
        else:
            w  = np.exp(-0.5 * (d_nn / d_max) ** 2)
            sw = np.sqrt(w)
            A  = np.column_stack([np.ones(K), f_nn]) * sw[:, None]
            b  = t_nn * sw
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
            lr = float(c[0] + c[1:] @ qvec)

        errors.append(abs(np.exp(qlp[i] + lr) - np.exp(qlp[i + 1])))
        act_diffs.append(abs(np.exp(qlp[i + 1]) - np.exp(qlp[i - 1])))

    return rmae(errors, act_diffs), len(errors)


# ── Оптимизация K (целочисленный тернарный поиск) ─────────────────────────────

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


# ── Калибровка ────────────────────────────────────────────────────────────────

def calibrate(qlp, qconf, qdirs, lh, ll, dates, t_big, k_hi):
    m, K, T = DEF_M, DEF_K, DEF_T_RATIO
    print(f"  Старт: m={m}, K={K}, T_ratio={T:.3f}")
    trace = []
    prev  = None

    for outer in range(MAX_OUTER):
        best_m = m; best_v = float("inf")
        for mc in M_VALUES:
            v, _ = eval_lwr(qlp, qconf, qdirs, lh, ll, dates, t_big, mc, K, T)
            if v < best_v:
                best_v = v; best_m = mc
        m = best_m

        K = ternary_int(
            lambda k: eval_lwr(qlp, qconf, qdirs, lh, ll, dates, t_big, m, k, T)[0],
            K_LO, k_hi,
        )

        T = golden(
            lambda t: eval_lwr(qlp, qconf, qdirs, lh, ll, dates, t_big, m, K, t)[0],
            T_LO, T_HI, T_TOL,
        )

        v, n = eval_lwr(qlp, qconf, qdirs, lh, ll, dates, t_big, m, K, T)
        row  = {"iter": outer + 1, "m": m, "K": K,
                "T_ratio": round(T, 4), "rMAE": round(v, 4)}
        trace.append(row)
        print(f"  Iter {outer+1}: m={m}, K={K}, T_ratio={T:.4f} → rMAE={v:.4f}  (n={n})")

        cur = (m, K, round(T, 3))
        if cur == prev:
            print("  Сошлось.")
            break
        prev = cur

    return m, K, T, v, n, trace


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Офлайн-калибровка LWR для зигзаг-прогноза",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker",   default="SBER",  help="Тикер")
    parser.add_argument("--interval", default="10m",   help="Интервал свечей")
    parser.add_argument("--t-big",    type=float, default=0.04, help="Порог T_BIG зигзага")
    parser.add_argument("--k-hi",     type=int, default=K_HI_DEF, help="Верхняя граница K")
    args = parser.parse_args()

    from _core import build_zigzag
    print(f"=== LWR CALIBRATION: {args.ticker} {args.interval} T={args.t_big*100:.0f}% ===")
    t0_total = time.time()

    lh, ll, dates = load_log_candles(args.ticker, args.interval)
    print(f"Свечей: {len(dates)}  ({dates[0][:10]} … {dates[-1][:10]})")

    qlp, qconf, qdirs = build_zigzag(lh, ll, dates, args.t_big)
    print(f"Пивотов: {len(qlp)}")

    print("\nДефолт:")
    lwr_def, lwr_def_n = eval_lwr(qlp, qconf, qdirs, lh, ll, dates,
                                   args.t_big, DEF_M, DEF_K, DEF_T_RATIO)
    print(f"  m={DEF_M}  K={DEF_K}  T_ratio={DEF_T_RATIO}  → rMAE={lwr_def:.4f}  (n={lwr_def_n})")

    print("\nКалибровка:")
    t0 = time.time()
    opt_m, opt_K, opt_T, opt_v, opt_n, trace = calibrate(
        qlp, qconf, qdirs, lh, ll, dates, args.t_big, args.k_hi,
    )
    elapsed = time.time() - t0

    delta = (opt_v - lwr_def) / lwr_def * 100
    print(f"\nВремя: {elapsed:.1f}s")
    print(f"Результат: m={opt_m}  K={opt_K}  T_ratio={opt_T:.4f}  → rMAE={opt_v:.4f}  Δ={delta:+.1f}%")

    result = {
        "method":   "LWR",
        "ticker":   args.ticker,
        "interval": args.interval,
        "t_big":    args.t_big,
        "n_pivots": int(len(qlp)),
        "default":  {"m": DEF_M, "K": DEF_K, "T_ratio": DEF_T_RATIO,
                     "rMAE": round(lwr_def, 4), "n_steps": lwr_def_n},
        "optimal":  {"m": opt_m, "K": opt_K, "T_ratio": round(opt_T, 4),
                     "rMAE": round(opt_v, 4), "n_steps": opt_n,
                     "delta_pct": round(delta, 2)},
        "trace":    trace,
        "elapsed_s": round(elapsed, 1),
    }
    t_tag = f"T{round(args.t_big * 100):03d}"
    out = RESULTS_DIR / f"lwr_{args.ticker}_{args.interval}_{t_tag}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Сохранено: {out}")
    print(f"Всего: {time.time()-t0_total:.1f}s")


if __name__ == "__main__":
    main()
