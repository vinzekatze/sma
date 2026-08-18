#!/usr/bin/env python3
"""
calibrate_smap.py — Офлайн-калибровка S-map для зигзаг-прогноза.

Оптимизирует (m, θ, T_ratio) покоординатным спуском на полной истории.

Использование:
  python calibrate_smap.py --ticker SBER --interval 10m --t-big 0.04
  python calibrate_smap.py --ticker GAZP --interval 1d  --t-big 0.03

S-map: w_i = exp(−θ · d_i / mean_d), регрессия по всем однонаправленным событиям пула.
При θ→0 вырождается в OLS. При θ→∞ берёт только ближайшего соседа.

Каузальный контракт: pool_conf[j+1] < query_conf[i].
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
M_VALUES        = [2, 3, 4, 5]
T_LO, T_HI     = 0.65, 1.0
T_TOL           = 0.005
THETA_LO        = 0.0
THETA_HI        = 20.0
THETA_TOL       = 0.1
MAX_OUTER       = 5
MIN_POOL_SMAP   = 4   # max(m+2, 4) применяется динамически

DEF_M       = 2
DEF_THETA   = 2.0
DEF_T_RATIO = 0.85


# ── Walk-forward S-map ────────────────────────────────────────────────────────

def eval_smap(qlp, qconf, qdirs, lh, ll, dates, t_big, m, theta, T_ratio):
    pf, pt, pd, peconf = get_pool(lh, ll, dates, T_ratio * t_big, m)
    min_pool  = max(m + 2, MIN_POOL_SMAP)
    errors, act_diffs = [], []

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
        act_diffs.append(abs(np.exp(qlp[i + 1]) - np.exp(qlp[i - 1])))

    return rmae(errors, act_diffs), len(errors)


# ── Калибровка ────────────────────────────────────────────────────────────────

def calibrate(qlp, qconf, qdirs, lh, ll, dates, t_big):
    m, theta, T = DEF_M, DEF_THETA, DEF_T_RATIO
    print(f"  Старт: m={m}, θ={theta}, T_ratio={T:.3f}")
    trace = []
    prev  = None

    for outer in range(MAX_OUTER):
        best_m = m; best_v = float("inf")
        for mc in M_VALUES:
            v, _ = eval_smap(qlp, qconf, qdirs, lh, ll, dates, t_big, mc, theta, T)
            if v < best_v:
                best_v = v; best_m = mc
        m = best_m

        theta = golden(
            lambda th: eval_smap(qlp, qconf, qdirs, lh, ll, dates, t_big, m, th, T)[0],
            THETA_LO, THETA_HI, THETA_TOL,
        )

        T = golden(
            lambda t: eval_smap(qlp, qconf, qdirs, lh, ll, dates, t_big, m, theta, t)[0],
            T_LO, T_HI, T_TOL,
        )

        v, n = eval_smap(qlp, qconf, qdirs, lh, ll, dates, t_big, m, theta, T)
        row  = {"iter": outer + 1, "m": m, "theta": round(theta, 3),
                "T_ratio": round(T, 4), "rMAE": round(v, 4)}
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
    parser = argparse.ArgumentParser(
        description="Офлайн-калибровка S-map для зигзаг-прогноза",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker",   default="SBER",  help="Тикер")
    parser.add_argument("--interval", default="10m",   help="Интервал свечей")
    parser.add_argument("--t-big",    type=float, default=0.04, help="Порог T_BIG зигзага")
    args = parser.parse_args()

    from _core import build_zigzag
    print(f"=== S-MAP CALIBRATION: {args.ticker} {args.interval} T={args.t_big*100:.0f}% ===")
    t0_total = time.time()

    lh, ll, dates = load_log_candles(args.ticker, args.interval)
    print(f"Свечей: {len(dates)}  ({dates[0][:10]} … {dates[-1][:10]})")

    qlp, qconf, qdirs = build_zigzag(lh, ll, dates, args.t_big)
    print(f"Пивотов: {len(qlp)}")

    print("\nДефолт:")
    smap_def, smap_def_n = eval_smap(qlp, qconf, qdirs, lh, ll, dates,
                                      args.t_big, DEF_M, DEF_THETA, DEF_T_RATIO)
    print(f"  m={DEF_M}  θ={DEF_THETA}  T_ratio={DEF_T_RATIO}  → rMAE={smap_def:.4f}  (n={smap_def_n})")

    print("\nКалибровка:")
    t0 = time.time()
    opt_m, opt_theta, opt_T, opt_v, opt_n, trace = calibrate(
        qlp, qconf, qdirs, lh, ll, dates, args.t_big,
    )
    elapsed = time.time() - t0

    delta = (opt_v - smap_def) / smap_def * 100
    print(f"\nВремя: {elapsed:.1f}s")
    print(f"Результат: m={opt_m}  θ={opt_theta:.3f}  T_ratio={opt_T:.4f}  → rMAE={opt_v:.4f}  Δ={delta:+.1f}%")

    result = {
        "method":   "S-map",
        "ticker":   args.ticker,
        "interval": args.interval,
        "t_big":    args.t_big,
        "n_pivots": int(len(qlp)),
        "default":  {"m": DEF_M, "theta": DEF_THETA, "T_ratio": DEF_T_RATIO,
                     "rMAE": round(smap_def, 4), "n_steps": smap_def_n},
        "optimal":  {"m": opt_m, "theta": round(opt_theta, 3), "T_ratio": round(opt_T, 4),
                     "rMAE": round(opt_v, 4), "n_steps": opt_n,
                     "delta_pct": round(delta, 2)},
        "trace":    trace,
        "elapsed_s": round(elapsed, 1),
    }
    t_tag = f"T{round(args.t_big * 100):03d}"
    out = RESULTS_DIR / f"smap_{args.ticker}_{args.interval}_{t_tag}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Сохранено: {out}")
    print(f"Всего: {time.time()-t0_total:.1f}s")


if __name__ == "__main__":
    main()
