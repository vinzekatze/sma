#!/usr/bin/env python3
"""
20f_individual_calibrator.py — совместная калибровка (m, θ, T_ratio, λ)
ИНДИВИДУАЛЬНО на каждый тикер (не LOO с чужих 6, как в 20e) — по
философии проекта: калибровка всегда персональна для тикера, даже при
общем кросс-тикерном пуле.

⚠️ Методическая оговорка (проговорена с пользователем): λ калибруется
IN-SAMPLE на тех же 42-75 origin, на которых потом оценивается rMAE —
тот же риск переобучения, что убил линейную amplitude-коррекцию в 17i
(in-sample b=1.134 выглядел многообещающе, на честном тесте — шум).
Для m/θ/T_ratio это исторически не оказалось проблемой (17r: калибровка
помогает на ВСЕХ 7 тикеров) — но λ ближе по духу к "коррекции", чем к
структурному параметру модели, поэтому результат этого скрипта сверяется
с уже честным LOO-результатом 20e (λ=2.0 почти everywhere), а не заменяет
его.

Покоординатный спуск: m (грид) → θ (golden) → T_ratio (golden) → λ (golden)
→ повтор до сходимости. Пул перестраивается только при смене m/T_ratio
(дорого — кросс-тикерные zigzag), θ/λ пересчитываются на готовом пуле
(дёшево).
"""
import importlib.util
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
EXP17_DIR = HERE.parents[0] / "17_large_scale_pooled"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

spec17 = importlib.util.spec_from_file_location("exp17", EXP17_DIR / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

spec20e = importlib.util.spec_from_file_location("exp20e", HERE / "20e_volume_weighting_comparison.py")
exp20e = importlib.util.module_from_spec(spec20e)
spec20e.loader.exec_module(exp20e)   # переиспользуем get_rank/get_logvol/build_pool_rows_with_extra/_smap_vol

T_BIG = 0.20
ARM = "D_allpeers"

M_GRID = [2, 3, 4, 5]
THETA_LO, THETA_HI, THETA_TOL = 0.0, 32.0, 0.1
T_LO, T_HI, T_TOL = 0.50, 0.95, 0.01
LAM_LO, LAM_HI, LAM_TOL = 0.0, 20.0, 0.1
MAX_OUTER = 5

CALIB_BASE = {   # старт — уже известная калибровка без λ (эксп.17f/17r)
    "SBER": {"m": 3, "theta": 25.697, "T_ratio": 0.8987},
    "LKOH": {"m": 4, "theta": 24.817, "T_ratio": 0.8331},
    "CHMF": {"m": 3, "theta": 11.492, "T_ratio": 0.9238},
    "NVTK": {"m": 2, "theta": 15.158, "T_ratio": 0.8080},
    "MGNT": {"m": 2, "theta": 11.814, "T_ratio": 0.6110},
    "VTBR": {"m": 3, "theta": 0.731,  "T_ratio": 0.6169},
    "NLMK": {"m": 2, "theta": 2.315,  "T_ratio": 0.8641},
}


def golden(func, a, b, tol, max_iter=50):
    phi = (5 ** 0.5 - 1) / 2
    x1 = b - phi * (b - a); x2 = a + phi * (b - a)
    f1 = func(x1); f2 = func(x2)
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


def build_pools(target, m, t_ratio):
    """Возвращает список кортежей (qvec, feats, tgts, ranks_j, rank_query, cur_lp, actual_price, pers_err)."""
    target_data = exp17.load_ticker(target)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != target]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != target}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    t_frac = t_ratio * T_BIG
    full_lp, full_conf, _ = exp17.build_zigzag(lh, ll, dates, T_BIG)
    n_big = len(full_lp)

    pools = []
    for i in range(exp17.MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
        t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]

        own_big_lp, own_big_conf, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, T_BIG)
        if len(own_big_lp) < m + 1 or own_big_lp[-1] != full_lp[i]:
            continue
        qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue
        q_dir = int(own_big_dir[-1])
        P_log = float(own_big_lp[-1])

        own_bar_idx = cutoff_idx - 1
        rank_query = exp20e.get_rank(target, own_bar_idx, P_log)
        if not np.isfinite(rank_query):
            continue

        def extra_for_chain(ticker, dates_arr, lp_arr):
            dates_full_t = exp20e.get_raw(ticker)[4]
            out = np.zeros((len(lp_arr), 2))
            for k in range(len(lp_arr)):
                bidx = exp20e.bar_idx_of(dates_full_t, dates_arr[k])
                out[k, 0] = exp20e.get_rank(ticker, bidx, lp_arr[k])
                out[k, 1] = exp20e.get_logvol(ticker, bidx)
            return out

        pool_feats, pool_tgts, pool_dirs, pool_extra = [], [], [], []
        ex = extra_for_chain(target, own_big_conf, own_big_lp)
        f, tg, dd, ee = exp20e.build_pool_rows_with_extra(own_big_lp, own_big_dir, ex, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd); pool_extra.append(ee)

        own_frac_lp, own_frac_conf, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
        ex = extra_for_chain(target, own_frac_conf, own_frac_lp)
        f, tg, dd, ee = exp20e.build_pool_rows_with_extra(own_frac_lp, own_frac_dir, ex, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd); pool_extra.append(ee)

        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < m + 2:
                continue
            p_lp, p_conf, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                                      peer_data[peer]["ll"][:p_cutoff],
                                                      p_dates[:p_cutoff], t_frac)
            ex = extra_for_chain(peer, p_conf, p_lp)
            f, tg, dd, ee = exp20e.build_pool_rows_with_extra(p_lp, p_dir, ex, m)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd); pool_extra.append(ee)

        feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts)
        dirs = np.concatenate(pool_dirs); extra = np.concatenate(pool_extra)
        mask = dirs == q_dir
        feats_d, tgts_d, extra_d = feats[mask], tgts[mask], extra[mask]
        if len(feats_d):
            d = np.linalg.norm(feats_d - qvec, axis=1)
            dup = d < exp17.DUP_EPS
            if dup.any():
                feats_d, tgts_d, extra_d = feats_d[~dup], tgts_d[~dup], extra_d[~dup]
        if len(tgts_d) < m + 2:
            continue

        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)

        pools.append((qvec, feats_d, tgts_d, extra_d[:, 0], rank_query, P_log, actual_price, pers_err))
    return pools


def eval_pools(pools, theta, lam, min_pool):
    errs, dz = [], []
    for qvec, feats, tgts, ranks_j, rank_query, P_log, actual_price, pers_err in pools:
        lr = exp20e._smap_vol(qvec, feats, tgts, ranks_j, rank_query, min_pool, theta, lam)
        if np.isfinite(lr):
            errs.append(abs(float(np.exp(P_log + lr)) - actual_price)); dz.append(pers_err)
    if len(errs) < 5:
        return float("inf"), 0
    return float(np.mean(errs) / np.mean(dz)), len(errs)


def calibrate(target, m0, theta0, t_ratio0):
    m, theta, T, lam = m0, theta0, t_ratio0, 2.0
    trace, prev = [], None
    for outer in range(MAX_OUTER):
        best_m, best_v = m, float("inf")
        for mc in M_GRID:
            pools = build_pools(target, mc, T)
            v, _ = eval_pools(pools, theta, lam, mc + 2)
            if v < best_v:
                best_v, best_m = v, mc
        m = best_m

        pools = build_pools(target, m, T)
        theta = golden(lambda th: eval_pools(pools, th, lam, m + 2)[0], THETA_LO, THETA_HI, THETA_TOL)

        T = golden(lambda t: eval_pools(build_pools(target, m, t), theta, lam, m + 2)[0], T_LO, T_HI, T_TOL)

        pools = build_pools(target, m, T)
        lam = golden(lambda l: eval_pools(pools, theta, l, m + 2)[0], LAM_LO, LAM_HI, LAM_TOL)

        v, n = eval_pools(pools, theta, lam, m + 2)
        trace.append({"iter": outer + 1, "m": m, "theta": round(theta, 3),
                      "T_ratio": round(T, 4), "lambda": round(lam, 3), "rMAE": round(v, 4)})
        print(f"    iter {outer+1}: m={m} θ={theta:.3f} T_ratio={T:.4f} λ={lam:.3f} → rMAE={v:.4f} (n={n})")
        cur = (m, round(theta, 2), round(T, 3), round(lam, 2))
        if cur == prev:
            break
        prev = cur
    return m, theta, T, lam, v, n, trace


def main():
    t0 = time.time()
    print("=== 20f_individual_calibrator — совместная калибровка (m,θ,T_ratio,λ) ПЕРСОНАЛЬНО на тикер ===\n")

    results = []
    for target, base in CALIB_BASE.items():
        print(f"--- {target} (старт: m={base['m']} θ={base['theta']} T_ratio={base['T_ratio']}) ---")
        t1 = time.time()

        # без λ (эталон 17f/17r — для сверки)
        pools0 = build_pools(target, base["m"], base["T_ratio"])
        v0, n0 = eval_pools(pools0, base["theta"], 0.0, base["m"] + 2)

        m, theta, T, lam, v, n, trace = calibrate(target, base["m"], base["theta"], base["T_ratio"])
        elapsed = time.time() - t1
        print(f"  без λ (эталон): rMAE={v0:.4f}\n"
              f"  → m={m} θ={theta:.3f} T_ratio={T:.4f} λ={lam:.3f} rMAE={v:.4f} (n={n})  ({elapsed:.1f}s)\n")

        results.append({"ticker": target, "rMAE_no_lambda": round(v0, 4),
                         "m": m, "theta": round(theta, 3), "T_ratio": round(T, 4), "lambda": round(lam, 3),
                         "rMAE_individual": round(v, 4), "n": n, "elapsed_s": round(elapsed, 1)})
        pd.DataFrame(results).to_csv(RESULTS / "individual_calibration.csv", index=False, float_format="%.4f")

    df = pd.DataFrame(results)
    print(f"{'='*90}")
    print(df.to_string(index=False))
    print(f"{'='*90}")
    print(f"Среднее: без λ={df['rMAE_no_lambda'].mean():.4f}  индивидуальная калибровка с λ={df['rMAE_individual'].mean():.4f}")
    print(f"Разброс индивидуальных λ: {sorted(df['lambda'].tolist())}")
    print(f"\nСверка с честным LOO (20e): там λ=2.0 почти everywhere — если индивидуальные λ здесь "
          f"сильно разбегаются в разные стороны, это флаг переобучения in-sample.")

    print(f"\nСохранено: {RESULTS / 'individual_calibration.csv'}")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
