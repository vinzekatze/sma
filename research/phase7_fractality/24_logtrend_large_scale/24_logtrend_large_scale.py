#!/usr/bin/env python3
"""
24_logtrend_large_scale.py — "контрольный тест" пользователя: применить
logtrend-стационаризацию (causal OLS, Direction A — CLAUDE.md "Ключевые
методические решения") к зигзагу на КРУПНОМ масштабе (T_big~20%), не
пробовавшуюся здесь ранее. Ожидание пользователя — вероятно, сломает
результат (крупномасштабный "разворот" может частично СОВПАДАТЬ с
долгосрочным трендом, который logtrend как раз вычитает).

log(ratio) = log(close) − lintrend_causal(log(close))  (причинный OLS,
инкрементальные суммы, без разогрева — copy из research/phase3_filterbank/
41_linear_trend_norm.py). Тренд считается по CLOSE, вычитается из
log(high)/log(low) тоже (единый тренд для полосы).

Четыре условия (SBER): raw+symmetric (эталон 17f, 0.6049), raw+asymmetric
(эксп.23, 0.5900), logtrend+symmetric, logtrend+asymmetric — две последние
считаются здесь, тот же протокол (подбор T_big/P_pct под сопоставимое
число пивотов, независимая калибровка m/θ/T_ratio).
"""
import importlib.util
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
EXP17_DIR = HERE.parents[0] / "17_large_scale_pooled"
EXP23_DIR = HERE.parents[0] / "23_asymmetric_zigzag"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

spec17 = importlib.util.spec_from_file_location("exp17", EXP17_DIR / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

spec23 = importlib.util.spec_from_file_location("exp23", EXP23_DIR / "23_asymmetric_zigzag_calibration.py")
exp23 = importlib.util.module_from_spec(spec23)
spec23.loader.exec_module(exp23)

TARGET = "SBER"
ARM = "D_allpeers"
MIN_HIST = exp17.MIN_HIST
N_TARGET_PIVOTS = 73   # эталон symmetric-raw (17f)

TBIG_SEARCH_GRID = np.round(np.arange(0.03, 0.31, 0.005), 4)   # шире вниз — detrended ряд может требовать меньший T
PPCT_SEARCH_GRID = exp23.PPCT_SEARCH_GRID

M_GRID = [2, 3, 4, 5]
THETA_LO, THETA_HI, THETA_TOL = 0.0, 32.0, 0.1
TRATIO_LO, TRATIO_HI, TRATIO_TOL = 0.50, 0.95, 0.01
MAX_OUTER = 5


_detrend_cache = {}


def lintrend_causal(x):
    """Causal OLS: trend[t]=a+b*t, инкрементальные суммы O(N). Копия
    research/phase3_filterbank/41_linear_trend_norm.py (без EPS-сдвига —
    x уже log(close), конечен по построению load_ticker)."""
    n = len(x)
    t = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n + 1, dtype=np.float64)
    ct = np.cumsum(t)
    ct2 = np.cumsum(t ** 2)
    cy = np.cumsum(x)
    cty = np.cumsum(t * x)
    denom = cn * ct2 - ct ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn * cty - ct * cy) / denom, 0.0)
    a = (cy - b * ct) / cn
    trend = a + b * t
    trend[:2] = x[:2]
    return trend


def compute_detrended(ticker):
    if ticker not in _detrend_cache or _detrend_cache[ticker] is None:
        data = exp17.load_ticker(ticker)
        trend = lintrend_causal(data["lc"])
        lh_d = data["lh"] - trend
        ll_d = data["ll"] - trend
        dates = data["dates"]
        _detrend_cache[ticker] = (lh_d, ll_d, dates)
    return _detrend_cache[ticker]


def pivot_count_sym(lh, ll, dates, t):
    lp, _, _ = exp17.build_zigzag(lh, ll, dates, t)
    return len(lp)


def pivot_count_asym(lh, ll, dates, p_pct):
    t_down, t_up = exp23.pct_to_log_thresholds(p_pct)
    lp, _, _ = exp23.build_zigzag_asym(lh, ll, dates, t_down, t_up)
    return len(lp)


# ── пул на detrended рядах, symmetric ────────────────────────────────────────

def build_all_pools_sym(rankings, checkpoints, t_big, t_ratio, m):
    lh, ll, dates = compute_detrended(TARGET)
    full_lp, full_conf, full_dirs = exp17.build_zigzag(lh, ll, dates, t_big)
    n_big = len(full_lp)
    t_frac = t_ratio * t_big

    pools = []
    for i in range(MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
        t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]

        own_big_lp, _, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_big)
        if len(own_big_lp) < m + 1 or own_big_lp[-1] != full_lp[i]:
            continue
        qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue
        q_dir = int(own_big_dir[-1])

        pool_feats, pool_tgts, pool_dirs = [], [], []
        f, tg, dd = exp17.build_pool_rows(own_big_lp, own_big_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)
        own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
        f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
            p_lh, p_ll, p_dates = compute_detrended(peer)
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < m + 2:
                continue
            p_lp, _, p_dir = exp17.build_zigzag(p_lh[:p_cutoff], p_ll[:p_cutoff], p_dates[:p_cutoff], t_frac)
            f, tg, dd = exp17.build_pool_rows(p_lp, p_dir, m)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts); dirs = np.concatenate(pool_dirs)
        mask = dirs == q_dir
        feats_d, tgts_d = feats[mask], tgts[mask]
        if len(feats_d):
            d = np.linalg.norm(feats_d - qvec, axis=1)
            dup = d < exp17.DUP_EPS
            if dup.any():
                feats_d, tgts_d = feats_d[~dup], tgts_d[~dup]

        cur_lp = float(own_big_lp[-1])
        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)
        pools.append((qvec, feats_d, tgts_d, cur_lp, actual_price, pers_err))
    return pools


def build_all_pools_asym(rankings, checkpoints, p_pct_big, p_ratio, m):
    lh, ll, dates = compute_detrended(TARGET)
    t_down_big, t_up_big = exp23.pct_to_log_thresholds(p_pct_big)
    p_pct_frac = p_ratio * p_pct_big
    t_down_frac, t_up_frac = exp23.pct_to_log_thresholds(p_pct_frac)

    full_lp, full_conf, full_dirs = exp23.build_zigzag_asym(lh, ll, dates, t_down_big, t_up_big)
    n_big = len(full_lp)

    pools = []
    for i in range(MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
        t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]

        own_big_lp, _, own_big_dir = exp23.build_zigzag_asym(t_lh, t_ll, t_dt, t_down_big, t_up_big)
        if len(own_big_lp) < m + 1 or own_big_lp[-1] != full_lp[i]:
            continue
        qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue
        q_dir = int(own_big_dir[-1])

        pool_feats, pool_tgts, pool_dirs = [], [], []
        f, tg, dd = exp17.build_pool_rows(own_big_lp, own_big_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)
        own_frac_lp, _, own_frac_dir = exp23.build_zigzag_asym(t_lh, t_ll, t_dt, t_down_frac, t_up_frac)
        f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
            p_lh, p_ll, p_dates = compute_detrended(peer)
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < m + 2:
                continue
            p_lp, _, p_dir = exp23.build_zigzag_asym(p_lh[:p_cutoff], p_ll[:p_cutoff], p_dates[:p_cutoff],
                                                      t_down_frac, t_up_frac)
            f, tg, dd = exp17.build_pool_rows(p_lp, p_dir, m)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts); dirs = np.concatenate(pool_dirs)
        mask = dirs == q_dir
        feats_d, tgts_d = feats[mask], tgts[mask]
        if len(feats_d):
            d = np.linalg.norm(feats_d - qvec, axis=1)
            dup = d < exp17.DUP_EPS
            if dup.any():
                feats_d, tgts_d = feats_d[~dup], tgts_d[~dup]

        cur_lp = float(own_big_lp[-1])
        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)
        pools.append((qvec, feats_d, tgts_d, cur_lp, actual_price, pers_err))
    return pools


eval_smap_pools = exp23.eval_smap_pools
golden = exp23.golden


def calibrate_sym(rankings, checkpoints, t_big):
    m, theta, T = 3, 5.0, 0.85
    prev = None
    for outer in range(MAX_OUTER):
        best_m, best_v = m, float("inf")
        for mc in M_GRID:
            pools = build_all_pools_sym(rankings, checkpoints, t_big, T, mc)
            v, _ = eval_smap_pools(pools, theta, mc + 2)
            if v < best_v:
                best_v, best_m = v, mc
        m = best_m
        pools = build_all_pools_sym(rankings, checkpoints, t_big, T, m)
        theta = golden(lambda th: eval_smap_pools(pools, th, m + 2)[0], THETA_LO, THETA_HI, THETA_TOL)
        T = golden(lambda t: eval_smap_pools(build_all_pools_sym(rankings, checkpoints, t_big, t, m), theta, m + 2)[0],
                   TRATIO_LO, TRATIO_HI, TRATIO_TOL)
        pools = build_all_pools_sym(rankings, checkpoints, t_big, T, m)
        v, n = eval_smap_pools(pools, theta, m + 2)
        print(f"    iter {outer+1}: m={m} θ={theta:.3f} T_ratio={T:.4f} → rMAE={v:.4f} (n={n})")
        cur = (m, round(theta, 2), round(T, 3))
        if cur == prev:
            break
        prev = cur
    return m, theta, T, v, n


def calibrate_asym(rankings, checkpoints, p_pct_big):
    m, theta, T = 3, 5.0, 0.85
    prev = None
    for outer in range(MAX_OUTER):
        best_m, best_v = m, float("inf")
        for mc in M_GRID:
            pools = build_all_pools_asym(rankings, checkpoints, p_pct_big, T, mc)
            v, _ = eval_smap_pools(pools, theta, mc + 2)
            if v < best_v:
                best_v, best_m = v, mc
        m = best_m
        pools = build_all_pools_asym(rankings, checkpoints, p_pct_big, T, m)
        theta = golden(lambda th: eval_smap_pools(pools, th, m + 2)[0], THETA_LO, THETA_HI, THETA_TOL)
        T = golden(lambda t: eval_smap_pools(build_all_pools_asym(rankings, checkpoints, p_pct_big, t, m), theta, m + 2)[0],
                   TRATIO_LO, TRATIO_HI, TRATIO_TOL)
        pools = build_all_pools_asym(rankings, checkpoints, p_pct_big, T, m)
        v, n = eval_smap_pools(pools, theta, m + 2)
        print(f"    iter {outer+1}: m={m} θ={theta:.3f} T_ratio={T:.4f} → rMAE={v:.4f} (n={n})")
        cur = (m, round(theta, 2), round(T, 3))
        if cur == prev:
            break
        prev = cur
    return m, theta, T, v, n


def main():
    t0 = time.time()
    print("=== 24_logtrend_large_scale — logtrend-detrended зигзаг на крупном масштабе (SBER) ===\n")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != TARGET}
    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = compute_detrended(TARGET)

    counts_sym = {t: pivot_count_sym(lh, ll, dates, t) for t in TBIG_SEARCH_GRID}
    best_t = min(counts_sym, key=lambda t: abs(counts_sym[t] - N_TARGET_PIVOTS))
    print(f"logtrend+symmetric: подобран T_big={best_t} → {counts_sym[best_t]} пивотов "
          f"(цель {N_TARGET_PIVOTS})")

    counts_asym = {p: pivot_count_asym(lh, ll, dates, p) for p in PPCT_SEARCH_GRID}
    best_p = min(counts_asym, key=lambda p: abs(counts_asym[p] - N_TARGET_PIVOTS))
    print(f"logtrend+asymmetric: подобран P_pct={best_p} → {counts_asym[best_p]} пивотов "
          f"(цель {N_TARGET_PIVOTS})\n")

    results = []

    print(f"--- logtrend+symmetric: T_big={best_t} ---")
    t1 = time.time()
    m, theta, T, v, n = calibrate_sym(rankings, checkpoints, best_t)
    el = time.time() - t1
    print(f"  → m={m} θ={theta:.3f} T_ratio={T:.4f} rMAE={v:.4f} (n={n})  ({el:.1f}s)\n")
    results.append({"condition": "logtrend+symmetric", "param": best_t, "n_pivots": counts_sym[best_t],
                     "m": m, "theta": round(theta, 3), "T_ratio": round(T, 4), "rMAE": round(v, 4),
                     "n_eval": n, "elapsed_s": round(el, 1)})
    pd.DataFrame(results).to_csv(RESULTS / "logtrend_large_scale.csv", index=False, float_format="%.4f")

    print(f"--- logtrend+asymmetric: P_pct={best_p} ---")
    t1 = time.time()
    m, theta, T, v, n = calibrate_asym(rankings, checkpoints, best_p)
    el = time.time() - t1
    print(f"  → m={m} θ={theta:.3f} T_ratio={T:.4f} rMAE={v:.4f} (n={n})  ({el:.1f}s)\n")
    results.append({"condition": "logtrend+asymmetric", "param": best_p, "n_pivots": counts_asym[best_p],
                     "m": m, "theta": round(theta, 3), "T_ratio": round(T, 4), "rMAE": round(v, 4),
                     "n_eval": n, "elapsed_s": round(el, 1)})
    pd.DataFrame(results).to_csv(RESULTS / "logtrend_large_scale.csv", index=False, float_format="%.4f")

    results.append({"condition": "raw+symmetric (эталон 17f)", "param": 0.20, "n_pivots": 73,
                     "m": 3, "theta": 25.697, "T_ratio": 0.8987, "rMAE": 0.6049, "n_eval": 42, "elapsed_s": None})
    results.append({"condition": "raw+asymmetric (эксп.23)", "param": 0.21, "n_pivots": 75,
                     "m": 4, "theta": 26.340, "T_ratio": 0.8581, "rMAE": 0.5900, "n_eval": 44, "elapsed_s": None})
    pd.DataFrame(results).to_csv(RESULTS / "logtrend_large_scale.csv", index=False, float_format="%.4f")

    df = pd.DataFrame(results)
    print(f"{'='*100}")
    print(df.to_string(index=False))
    print(f"{'='*100}")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
