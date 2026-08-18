#!/usr/bin/env python3
"""
17f_calibrator.py — офлайн-калибровка (m, локальность, T_ratio) для
S-map/LWR/Simplex на кросс-тикерном пуле (D_allpeers), покоординатный спуск.

Аналог `13_offline_calibration/calibrate_*.py`, но пул строится через
кросс-тикерную инфраструктуру 17_large_scale_pooled.py (D_allpeers, 45
тикеров, каузальная обрезка + дедуп), а не через одиночный тикер `_core.py`.

Методы: S-map (θ), LWR (K), Simplex (K=m+1 фиксировано — канонический
минимальный симплекс, локальность НЕ калибруется отдельно, только m).
LA0 не калибруется: в 17d показал себя плоским по K, всегда между
LWR/Simplex — отдельного смысла нет.

arm зафиксирован на D_allpeers (уже подтверждён как лучший/сопоставимый с
C_top8 в Stage 1/1b) — выбор arm не входит в сетку калибровки.

Оптимизация: пул перестраивается (дорогая операция — зигзаги по 45
тикерам на каждом шаге) только когда меняются m или T_ratio. Поиск
локальности (θ golden-section / K ternary-search) выполняется на УЖЕ
построенном пуле — пересчёт только финальной взвешенной регрессии, дёшево.

Первый прогон: только SBER, T_big=20% (подтверждённая точка Stage 1e).
"""
import time
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent

spec17 = importlib.util.spec_from_file_location("exp17", HERE / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

specd = importlib.util.spec_from_file_location("ksweep", HERE / "17d_k_sweep.py")
ksweep = importlib.util.module_from_spec(specd)
specd.loader.exec_module(ksweep)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TARGET = "SBER"
T_BIG = 0.20
ARM = "D_allpeers"

M_GRID = [2, 3, 4, 5]
THETA_LO, THETA_HI, THETA_TOL = 0.0, 32.0, 0.1
K_LO, K_HI = 10, 200
T_LO, T_HI, T_TOL = 0.50, 0.95, 0.01
MAX_OUTER = 5

DEF_M, DEF_THETA, DEF_K, DEF_T_RATIO = 3, 1.0, 50, 0.85


# ── golden-section / ternary search (копия из _core.py, самодостаточно) ──────

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


# ── построение пула (m как параметр, не exp17.M) ──────────────────────────────

def build_all_pools(target_data, peer_data, rankings, checkpoints, t_big, m, t_frac, arm, full_lp, full_conf):
    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    n_big = len(full_lp)
    pools = []
    for i in range(exp17.MIN_HIST, n_big - 1):
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

        own_big_feats, own_big_tgts, own_big_dirs = exp17.build_pool_rows(own_big_lp, own_big_dir, m)
        pool_feats = [own_big_feats]; pool_tgts = [own_big_tgts]; pool_dirs = [own_big_dirs]

        if arm != "A_baseline":
            own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
            f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, m)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

            for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, arm):
                p_dates = peer_data[peer]["dates"]
                p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
                if p_cutoff < m + 2:
                    continue
                p_lh = peer_data[peer]["lh"][:p_cutoff]
                p_ll = peer_data[peer]["ll"][:p_cutoff]
                p_dt = p_dates[:p_cutoff]
                p_lp, _, p_dir = exp17.build_zigzag(p_lh, p_ll, p_dt, t_frac)
                f, tg, dd = exp17.build_pool_rows(p_lp, p_dir, m)
                pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        feats = np.concatenate(pool_feats) if pool_feats else np.empty((0, m))
        tgts  = np.concatenate(pool_tgts)  if pool_tgts  else np.empty(0)
        dirs  = np.concatenate(pool_dirs)  if pool_dirs  else np.empty(0, dtype=np.int8)
        mask = dirs == q_dir
        feats_d, tgts_d = feats[mask], tgts[mask]

        if len(feats_d):
            d = np.linalg.norm(feats_d - qvec, axis=1)
            dup_mask = d < exp17.DUP_EPS
            if dup_mask.any():
                feats_d, tgts_d, d = feats_d[~dup_mask], tgts_d[~dup_mask], d[~dup_mask]
        else:
            d = np.empty(0)

        cur_lp = float(own_big_lp[-1])
        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)

        pools.append((qvec, feats_d, tgts_d, d, cur_lp, actual_price, pers_err))
    return pools


# ── eval по уже построенным пулам (дёшево — без перестройки зигзагов) ───────

def eval_smap_pools(pools, theta, min_pool):
    errs, dz = [], []
    for qvec, feats_d, tgts_d, d, cur_lp, actual_price, pers_err in pools:
        lr = exp17._smap(qvec, feats_d, tgts_d, min_pool, theta)
        if np.isfinite(lr):
            errs.append(abs(float(np.exp(cur_lp + lr)) - actual_price)); dz.append(pers_err)
    if len(errs) < 5:
        return float("inf"), 0
    return float(np.mean(errs) / np.mean(dz)), len(errs)


def eval_lwr_pools(pools, K):
    errs, dz = [], []
    for qvec, feats_d, tgts_d, d, cur_lp, actual_price, pers_err in pools:
        lr = ksweep._lwr_k(d, feats_d, tgts_d, qvec, K)
        if np.isfinite(lr):
            errs.append(abs(float(np.exp(cur_lp + lr)) - actual_price)); dz.append(pers_err)
    if len(errs) < 5:
        return float("inf"), 0
    return float(np.mean(errs) / np.mean(dz)), len(errs)


def eval_simplex_pools(pools, K):
    errs, dz = [], []
    for qvec, feats_d, tgts_d, d, cur_lp, actual_price, pers_err in pools:
        lr = ksweep._simplex_k(d, tgts_d, K)
        if np.isfinite(lr):
            errs.append(abs(float(np.exp(cur_lp + lr)) - actual_price)); dz.append(pers_err)
    if len(errs) < 5:
        return float("inf"), 0
    return float(np.mean(errs) / np.mean(dz)), len(errs)


# ── покоординатный спуск ──────────────────────────────────────────────────────

def _rebuild(td, pd_, rk, cps, t_big, m, T, arm, full_lp, full_conf):
    return build_all_pools(td, pd_, rk, cps, t_big, m, T * t_big, arm, full_lp, full_conf)


def calibrate_smap(td, pd_, rk, cps, t_big, arm, full_lp, full_conf):
    m, theta, T = DEF_M, DEF_THETA, DEF_T_RATIO
    trace, prev = [], None
    for outer in range(MAX_OUTER):
        best_m, best_v = m, float("inf")
        for mc in M_GRID:
            pools = _rebuild(td, pd_, rk, cps, t_big, mc, T, arm, full_lp, full_conf)
            v, _ = eval_smap_pools(pools, theta, mc + 2)
            if v < best_v:
                best_v, best_m = v, mc
        m = best_m

        pools = _rebuild(td, pd_, rk, cps, t_big, m, T, arm, full_lp, full_conf)
        theta = golden(lambda th: eval_smap_pools(pools, th, m + 2)[0], THETA_LO, THETA_HI, THETA_TOL)

        T = golden(lambda t: eval_smap_pools(_rebuild(td, pd_, rk, cps, t_big, m, t, arm, full_lp, full_conf), theta, m + 2)[0],
                   T_LO, T_HI, T_TOL)

        pools = _rebuild(td, pd_, rk, cps, t_big, m, T, arm, full_lp, full_conf)
        v, n = eval_smap_pools(pools, theta, m + 2)
        trace.append({"iter": outer + 1, "m": m, "theta": round(theta, 3), "T_ratio": round(T, 4), "rMAE": round(v, 4)})
        print(f"    [S-map] iter {outer+1}: m={m} θ={theta:.3f} T={T:.4f} → rMAE={v:.4f} (n={n})")
        cur = (m, round(theta, 2), round(T, 3))
        if cur == prev:
            break
        prev = cur
    return m, theta, T, v, n, trace


def calibrate_lwr(td, pd_, rk, cps, t_big, arm, full_lp, full_conf):
    m, K, T = DEF_M, DEF_K, DEF_T_RATIO
    trace, prev = [], None
    for outer in range(MAX_OUTER):
        best_m, best_v = m, float("inf")
        for mc in M_GRID:
            pools = _rebuild(td, pd_, rk, cps, t_big, mc, T, arm, full_lp, full_conf)
            v, _ = eval_lwr_pools(pools, K)
            if v < best_v:
                best_v, best_m = v, mc
        m = best_m

        pools = _rebuild(td, pd_, rk, cps, t_big, m, T, arm, full_lp, full_conf)
        K = ternary_int(lambda k: eval_lwr_pools(pools, k)[0], K_LO, K_HI)

        T = golden(lambda t: eval_lwr_pools(_rebuild(td, pd_, rk, cps, t_big, m, t, arm, full_lp, full_conf), K)[0],
                   T_LO, T_HI, T_TOL)

        pools = _rebuild(td, pd_, rk, cps, t_big, m, T, arm, full_lp, full_conf)
        v, n = eval_lwr_pools(pools, K)
        trace.append({"iter": outer + 1, "m": m, "K": K, "T_ratio": round(T, 4), "rMAE": round(v, 4)})
        print(f"    [LWR]   iter {outer+1}: m={m} K={K} T={T:.4f} → rMAE={v:.4f} (n={n})")
        cur = (m, K, round(T, 3))
        if cur == prev:
            break
        prev = cur
    return m, K, T, v, n, trace


def calibrate_simplex(td, pd_, rk, cps, t_big, arm, full_lp, full_conf):
    m, T = DEF_M, DEF_T_RATIO
    trace, prev = [], None
    for outer in range(MAX_OUTER):
        best_m, best_v = m, float("inf")
        for mc in M_GRID:
            pools = _rebuild(td, pd_, rk, cps, t_big, mc, T, arm, full_lp, full_conf)
            v, _ = eval_simplex_pools(pools, mc + 1)  # K = E+1, канонично
            if v < best_v:
                best_v, best_m = v, mc
        m = best_m

        T = golden(lambda t: eval_simplex_pools(_rebuild(td, pd_, rk, cps, t_big, m, t, arm, full_lp, full_conf), m + 1)[0],
                   T_LO, T_HI, T_TOL)

        pools = _rebuild(td, pd_, rk, cps, t_big, m, T, arm, full_lp, full_conf)
        v, n = eval_simplex_pools(pools, m + 1)
        trace.append({"iter": outer + 1, "m": m, "K": m + 1, "T_ratio": round(T, 4), "rMAE": round(v, 4)})
        print(f"    [Simplex] iter {outer+1}: m={m} K={m+1}(=E+1) T={T:.4f} → rMAE={v:.4f} (n={n})")
        cur = (m, round(T, 3))
        if cur == prev:
            break
        prev = cur
    return m, m + 1, T, v, n, trace


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"=== 17f_calibrator === target={TARGET} T_big={T_BIG} arm={ARM}")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, full_dirs = exp17.build_zigzag(lh, ll, dates, T_BIG)
    print(f"Пивотов T_big: {len(full_lp)}")

    # ── дефолты для сравнения ──
    def_pools = build_all_pools(target_data, peer_data, rankings, checkpoints, T_BIG, DEF_M, DEF_T_RATIO * T_BIG, ARM, full_lp, full_conf)
    smap_def, _ = eval_smap_pools(def_pools, DEF_THETA, DEF_M + 2)
    lwr_def, _  = eval_lwr_pools(def_pools, DEF_K)
    sx_def, _   = eval_simplex_pools(def_pools, DEF_M + 1)
    print(f"\nДефолт (m={DEF_M}, T_ratio={DEF_T_RATIO}): "
          f"S-map(θ={DEF_THETA})={smap_def:.4f}  LWR(K={DEF_K})={lwr_def:.4f}  Simplex(K={DEF_M+1})={sx_def:.4f}\n")

    results = {}
    print("Калибровка S-map...")
    t1 = time.time()
    sm = calibrate_smap(target_data, peer_data, rankings, checkpoints, T_BIG, ARM, full_lp, full_conf)
    print(f"  → m={sm[0]} θ={sm[1]:.3f} T_ratio={sm[2]:.4f} rMAE={sm[3]:.4f}  ({time.time()-t1:.1f}s)")
    results["smap"] = sm

    print("Калибровка LWR...")
    t1 = time.time()
    lw = calibrate_lwr(target_data, peer_data, rankings, checkpoints, T_BIG, ARM, full_lp, full_conf)
    print(f"  → m={lw[0]} K={lw[1]} T_ratio={lw[2]:.4f} rMAE={lw[3]:.4f}  ({time.time()-t1:.1f}s)")
    results["lwr"] = lw

    print("Калибровка Simplex...")
    t1 = time.time()
    sx = calibrate_simplex(target_data, peer_data, rankings, checkpoints, T_BIG, ARM, full_lp, full_conf)
    print(f"  → m={sx[0]} K={sx[1]}(=E+1) T_ratio={sx[2]:.4f} rMAE={sx[3]:.4f}  ({time.time()-t1:.1f}s)")
    results["simplex"] = sx

    print(f"\n{'='*60}")
    print(f"{'method':<10} {'default':>10} {'optimal':>10} {'params'}")
    print(f"{'S-map':<10} {smap_def:>10.4f} {sm[3]:>10.4f}  m={sm[0]} θ={sm[1]:.3f} T={sm[2]:.4f}")
    print(f"{'LWR':<10} {lwr_def:>10.4f} {lw[3]:>10.4f}  m={lw[0]} K={lw[1]} T={lw[2]:.4f}")
    print(f"{'Simplex':<10} {sx_def:>10.4f} {sx[3]:>10.4f}  m={sx[0]} K={sx[1]} T={sx[2]:.4f}")
    print(f"{'='*60}")
    print(f"Всего: {time.time()-t0:.1f}s")

    out = {
        "target": TARGET, "t_big": T_BIG, "arm": ARM,
        "defaults": {"m": DEF_M, "T_ratio": DEF_T_RATIO,
                     "smap": {"theta": DEF_THETA, "rMAE": round(smap_def, 4)},
                     "lwr": {"K": DEF_K, "rMAE": round(lwr_def, 4)},
                     "simplex": {"K": DEF_M + 1, "rMAE": round(sx_def, 4)}},
        "optimal": {
            "smap":    {"m": sm[0], "theta": round(sm[1], 3), "T_ratio": round(sm[2], 4), "rMAE": round(sm[3], 4), "n": sm[4], "trace": sm[5]},
            "lwr":     {"m": lw[0], "K": lw[1], "T_ratio": round(lw[2], 4), "rMAE": round(lw[3], 4), "n": lw[4], "trace": lw[5]},
            "simplex": {"m": sx[0], "K": sx[1], "T_ratio": round(sx[2], 4), "rMAE": round(sx[3], 4), "n": sx[4], "trace": sx[5]},
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    import json
    out_path = RESULTS / f"calibrator_{TARGET}_T{int(T_BIG*100):03d}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
