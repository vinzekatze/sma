#!/usr/bin/env python3
"""
23_asymmetric_zigzag_calibration.py — направленно-зависимый порог зигзага
(percentage-symmetric), сравнение со стандартным симметричным-в-логах
порогом (весь эксп.17-20).

Мотивация: `ext − price >= T` (вниз) и `price − ext >= T` (вверх) с ОДНИМ и
тем же T симметричны в ЛОГ-единицах, но не в процентах:
  вниз от пика:    падение на 1 − e^(−T)   (T=0.20 → 18.13%)
  вверх от впадины: рост   на e^T − 1      (T=0.20 → 22.14%)
Один и тот же T требует РАЗНОГО процентного движения в разные стороны —
классический зигзаг (на сырых ценах, с процентным порогом) симметричен
в процентах, не в логах. Раз мы работаем в лог-пространстве, порог можно
явно исправить: для целевого процента отката P_pct
  T_down = −log(1 − P_pct)      T_up = log(1 + P_pct)
оба выводятся из ОДНОГО P_pct, но различны в лог-единицах.

Сравнение — как в эксп.21 (SSA): не абсолютные rMAE в лоб, а после
подбора сопоставимого числа пивотов и НЕЗАВИСИМОЙ калибровки (m,θ,T_ratio)
для каждого варианта, rMAE относительно СВОЕГО persistence.

"symmetric" (эталон, уже откалиброван в эксп.17f) — T_big=0.20, high/low,
m=3 θ=25.697 T_ratio=0.8987, rMAE=0.6049.
"asymmetric" — P_pct подбирается под сопоставимое число пивотов, дальше
T_ratio (тот же принцип — доля P_pct для уровня T_frac) калибруется заново.
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

TARGET = "SBER"
ARM = "D_allpeers"
MIN_HIST = exp17.MIN_HIST

REFERENCE_SYMMETRIC = {"T_big": 0.20, "m": 3, "theta": 25.697, "T_ratio": 0.8987, "rMAE": 0.6049}

M_GRID = [2, 3, 4, 5]
THETA_LO, THETA_HI, THETA_TOL = 0.0, 32.0, 0.1
TRATIO_LO, TRATIO_HI, TRATIO_TOL = 0.50, 0.95, 0.01
MAX_OUTER = 5
DEF_M, DEF_THETA, DEF_T_RATIO = 3, 25.697, 0.8987

PPCT_SEARCH_GRID = np.round(np.arange(0.10, 0.31, 0.005), 4)


def pct_to_log_thresholds(p_pct):
    """P_pct (доля отката, одинаковая в обе стороны) -> (T_down, T_up) в лог-единицах."""
    t_down = -np.log(1 - p_pct)
    t_up = np.log(1 + p_pct)
    return t_down, t_up


def build_zigzag_asym(lh, ll, dates, t_down, t_up):
    """Тот же алгоритм, что exp17.build_zigzag, но t_down (подтверждение разворота
    ВНИЗ от пика) и t_up (подтверждение разворота ВВЕРХ от впадины) — разные."""
    lp, conf, dirs = [], [], []
    cur = 0
    ext = (lh[0] + ll[0]) / 2.0
    for i in range(len(lh)):
        if cur == 0:
            if lh[i] - ext >= t_up:
                cur = 1; ext = lh[i]
            elif ext - ll[i] >= t_down:
                cur = -1; ext = ll[i]
        elif cur == 1:
            if lh[i] > ext:
                ext = lh[i]
            elif ext - ll[i] >= t_down:
                lp.append(ext); conf.append(dates[i]); dirs.append(1)
                cur = -1; ext = ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= t_up:
                lp.append(ext); conf.append(dates[i]); dirs.append(-1)
                cur = 1; ext = lh[i]
    return np.array(lp), np.array(conf), np.array(dirs, dtype=np.int8)


def pivot_count_asym(lh, ll, dates, p_pct):
    t_down, t_up = pct_to_log_thresholds(p_pct)
    lp, _, _ = build_zigzag_asym(lh, ll, dates, t_down, t_up)
    return len(lp)


def build_all_pools(target_data, peer_data, rankings, checkpoints, p_pct_big, p_ratio, m):
    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    t_down_big, t_up_big = pct_to_log_thresholds(p_pct_big)
    p_pct_frac = p_ratio * p_pct_big
    t_down_frac, t_up_frac = pct_to_log_thresholds(p_pct_frac)

    full_lp, full_conf, full_dirs = build_zigzag_asym(lh, ll, dates, t_down_big, t_up_big)
    n_big = len(full_lp)

    pools = []
    for i in range(MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
        t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]

        own_big_lp, _, own_big_dir = build_zigzag_asym(t_lh, t_ll, t_dt, t_down_big, t_up_big)
        if len(own_big_lp) < m + 1 or own_big_lp[-1] != full_lp[i]:
            continue
        qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue
        q_dir = int(own_big_dir[-1])

        pool_feats, pool_tgts, pool_dirs = [], [], []
        f, tg, dd = exp17.build_pool_rows(own_big_lp, own_big_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        own_frac_lp, _, own_frac_dir = build_zigzag_asym(t_lh, t_ll, t_dt, t_down_frac, t_up_frac)
        f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < m + 2:
                continue
            p_lp, _, p_dir = build_zigzag_asym(peer_data[peer]["lh"][:p_cutoff],
                                                peer_data[peer]["ll"][:p_cutoff],
                                                p_dates[:p_cutoff], t_down_frac, t_up_frac)
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


def eval_smap_pools(pools, theta, min_pool):
    errs, dz = [], []
    for qvec, feats, tgts, cur_lp, actual_price, pers_err in pools:
        lr = exp17._smap(qvec, feats, tgts, min_pool, theta)
        if np.isfinite(lr):
            errs.append(abs(float(np.exp(cur_lp + lr)) - actual_price)); dz.append(pers_err)
    if len(errs) < 5:
        return float("inf"), 0
    return float(np.mean(errs) / np.mean(dz)), len(errs)


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


def calibrate(target_data, peer_data, rankings, checkpoints, p_pct_big):
    m, theta, T = DEF_M, DEF_THETA, DEF_T_RATIO
    trace, prev = [], None
    for outer in range(MAX_OUTER):
        best_m, best_v = m, float("inf")
        for mc in M_GRID:
            pools = build_all_pools(target_data, peer_data, rankings, checkpoints, p_pct_big, T, mc)
            v, _ = eval_smap_pools(pools, theta, mc + 2)
            if v < best_v:
                best_v, best_m = v, mc
        m = best_m

        pools = build_all_pools(target_data, peer_data, rankings, checkpoints, p_pct_big, T, m)
        theta = golden(lambda th: eval_smap_pools(pools, th, m + 2)[0], THETA_LO, THETA_HI, THETA_TOL)

        T = golden(lambda t: eval_smap_pools(
            build_all_pools(target_data, peer_data, rankings, checkpoints, p_pct_big, t, m), theta, m + 2)[0],
            TRATIO_LO, TRATIO_HI, TRATIO_TOL)

        pools = build_all_pools(target_data, peer_data, rankings, checkpoints, p_pct_big, T, m)
        v, n = eval_smap_pools(pools, theta, m + 2)
        trace.append({"iter": outer + 1, "m": m, "theta": round(theta, 3), "T_ratio": round(T, 4), "rMAE": round(v, 4)})
        print(f"    iter {outer+1}: m={m} θ={theta:.3f} T_ratio={T:.4f} → rMAE={v:.4f} (n={n})")
        cur = (m, round(theta, 2), round(T, 3))
        if cur == prev:
            break
        prev = cur
    return m, theta, T, v, n, trace


def main():
    t0 = time.time()
    print("=== 23_asymmetric_zigzag_calibration — symmetric-log vs percentage-symmetric ===\n")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != TARGET}
    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]

    # ── референс (симметричный-в-логах, уже откалиброван в 17f) ──
    ref_lp, _, _ = exp17.build_zigzag(lh, ll, dates, REFERENCE_SYMMETRIC["T_big"])
    n_ref = len(ref_lp)
    print(f"symmetric (эталон 17f): T_big={REFERENCE_SYMMETRIC['T_big']} → {n_ref} пивотов "
          f"(m={REFERENCE_SYMMETRIC['m']} θ={REFERENCE_SYMMETRIC['theta']} "
          f"T_ratio={REFERENCE_SYMMETRIC['T_ratio']} rMAE={REFERENCE_SYMMETRIC['rMAE']})")

    # ── проверка: сколько пивотов даёт percentage-symmetric на том же номинальном 20%? ──
    n_asym_20 = pivot_count_asym(lh, ll, dates, 0.20)
    print(f"asymmetric @ P_pct=0.20 (без подбора): {n_asym_20} пивотов "
          f"(T_down={pct_to_log_thresholds(0.20)[0]:.4f} T_up={pct_to_log_thresholds(0.20)[1]:.4f})")

    counts = {p: pivot_count_asym(lh, ll, dates, p) for p in PPCT_SEARCH_GRID}
    best_p = min(counts, key=lambda p: abs(counts[p] - n_ref))
    print(f"Подбор P_pct под {n_ref} пивотов (грид {PPCT_SEARCH_GRID[0]}..{PPCT_SEARCH_GRID[-1]}): "
          f"P_pct={best_p} → {counts[best_p]} пивотов\n")

    print(f"--- asymmetric: P_pct={best_p} ({counts[best_p]} пивотов) ---")
    t1 = time.time()
    m, theta, T, v, n, trace = calibrate(target_data, peer_data, rankings, checkpoints, best_p)
    elapsed = time.time() - t1
    print(f"  → m={m} θ={theta:.3f} T_ratio={T:.4f} rMAE={v:.4f} (n={n})  ({elapsed:.1f}s)\n")

    result = {"condition": "asymmetric", "P_pct": best_p, "n_pivots": counts[best_p],
              "m": m, "theta": round(theta, 3), "T_ratio": round(T, 4),
              "rMAE": round(v, 4), "n_eval": n, "elapsed_s": round(elapsed, 1)}
    pd.DataFrame([
        {"condition": "symmetric", "P_pct_or_Tbig": REFERENCE_SYMMETRIC["T_big"], "n_pivots": n_ref,
         "m": REFERENCE_SYMMETRIC["m"], "theta": REFERENCE_SYMMETRIC["theta"],
         "T_ratio": REFERENCE_SYMMETRIC["T_ratio"], "rMAE": REFERENCE_SYMMETRIC["rMAE"], "n_eval": 42},
        {"condition": "asymmetric", "P_pct_or_Tbig": best_p, "n_pivots": counts[best_p],
         "m": m, "theta": round(theta, 3), "T_ratio": round(T, 4), "rMAE": round(v, 4), "n_eval": n},
    ]).to_csv(RESULTS / "asymmetric_vs_symmetric.csv", index=False, float_format="%.4f")

    print(f"{'='*80}")
    print(f"symmetric (17f, эталон):  rMAE={REFERENCE_SYMMETRIC['rMAE']:.4f}  n={42}")
    print(f"asymmetric (пересчитан):  rMAE={v:.4f}  n={n}")
    print(f"{'='*80}")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
