#!/usr/bin/env python3
"""
19c_calibrator.py — Фаза B: совместная калибровка (m, θ, T_ratio_frac,
T_ratio_super) для S-map на пуле с добавленным T_super-уровнем (события
крупнее T_big). Только S-map (LWR/Simplex не калибруются — договорено
ранее, они менее гибкие).

Мотивация: Фаза A (19_quick_test.py) и стратификация (19b_stratified.py)
показали, что на ТЕКУЩЕЙ калибровке эксп.17f (θ=25.697, заточенной под пул
без T_super) добавление T_super-пула в среднем хуже baseline, но точечно
улучшает именно случаи с крупным будущим плечом (до −9% на терции). Гипотеза
пользователя: совместная калибровка (в первую очередь θ) может частично
или полностью снять этот компромисс — вероятно, в сторону МЕНЬШЕГО θ, чтобы
физически далёкие (немасштабированные, как и решили — без нормировки)
T_super-векторы не гасились локализацией S-map.

Дедуп — как и в 19_quick_test.py, единый проход по объединённому пулу.

Структура покоординатного спуска — как в 17f_calibrator.py (m → θ → T),
здесь добавлено четвёртое измерение U = T_ratio_super, с тем же принципом
(golden-section, пул перестраивается только при смене m/T/U).

Точка: SBER, T_big=20%, D_allpeers (тот же arm, что и везде в фазе 7 для
этого масштаба).
"""
import time
import importlib.util
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
T_BIG = 0.20
ARM = "D_allpeers"

M_GRID = [2, 3, 4, 5]
THETA_LO, THETA_HI, THETA_TOL = 0.0, 32.0, 0.1
T_LO, T_HI, T_TOL = 0.50, 0.95, 0.01     # диапазон для T_ratio_frac И T_ratio_super
MAX_OUTER = 5

DEF_M, DEF_THETA = 3, 25.697
DEF_T_RATIO_FRAC = 0.8987
DEF_T_RATIO_SUPER = 0.8987   # старт — «зеркало», но диапазон поиска тот же 0.50-0.95


# ── golden-section (копия 17f_calibrator.py, самодостаточно) ────────────────

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


# ── построение пула с T_super (m как параметр) ───────────────────────────────

def build_all_pools_super(target_data, peer_data, rankings, checkpoints, t_big, m, t_frac, t_super,
                           arm, full_lp, full_conf):
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

        pool_feats, pool_tgts, pool_dirs = [], [], []

        def add(lp, dirs):
            f, tg, dd = exp17.build_pool_rows(lp, dirs, m)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        big_f, big_tg, big_dd = exp17.build_pool_rows(own_big_lp, own_big_dir, m)
        pool_feats.append(big_f); pool_tgts.append(big_tg); pool_dirs.append(big_dd)

        own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
        add(own_frac_lp, own_frac_dir)
        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, arm):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < m + 2:
                continue
            p_lp, _, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                                 peer_data[peer]["ll"][:p_cutoff],
                                                 p_dates[:p_cutoff], t_frac)
            add(p_lp, p_dir)

        own_super_lp, _, own_super_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_super)
        add(own_super_lp, own_super_dir)
        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, arm):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < m + 2:
                continue
            p_lp, _, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                                 peer_data[peer]["ll"][:p_cutoff],
                                                 p_dates[:p_cutoff], t_super)
            add(p_lp, p_dir)

        feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts); dirs = np.concatenate(pool_dirs)
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


def eval_smap_pools(pools, theta, min_pool):
    errs, dz = [], []
    for qvec, feats_d, tgts_d, d, cur_lp, actual_price, pers_err in pools:
        lr = exp17._smap(qvec, feats_d, tgts_d, min_pool, theta)
        if np.isfinite(lr):
            errs.append(abs(float(np.exp(cur_lp + lr)) - actual_price)); dz.append(pers_err)
    if len(errs) < 5:
        return float("inf"), 0
    return float(np.mean(errs) / np.mean(dz)), len(errs)


# ── покоординатный спуск: m → θ → T_ratio_frac → T_ratio_super ──────────────

def _rebuild(td, pd_, rk, cps, t_big, m, T, U, arm, full_lp, full_conf):
    return build_all_pools_super(td, pd_, rk, cps, t_big, m, T * t_big, t_big / U, arm, full_lp, full_conf)


def calibrate_smap_super(td, pd_, rk, cps, t_big, arm, full_lp, full_conf):
    m, theta, T, U = DEF_M, DEF_THETA, DEF_T_RATIO_FRAC, DEF_T_RATIO_SUPER
    trace, prev = [], None
    for outer in range(MAX_OUTER):
        best_m, best_v = m, float("inf")
        for mc in M_GRID:
            pools = _rebuild(td, pd_, rk, cps, t_big, mc, T, U, arm, full_lp, full_conf)
            v, _ = eval_smap_pools(pools, theta, mc + 2)
            if v < best_v:
                best_v, best_m = v, mc
        m = best_m

        pools = _rebuild(td, pd_, rk, cps, t_big, m, T, U, arm, full_lp, full_conf)
        theta = golden(lambda th: eval_smap_pools(pools, th, m + 2)[0], THETA_LO, THETA_HI, THETA_TOL)

        T = golden(lambda t: eval_smap_pools(
            _rebuild(td, pd_, rk, cps, t_big, m, t, U, arm, full_lp, full_conf), theta, m + 2)[0],
            T_LO, T_HI, T_TOL)

        U = golden(lambda u: eval_smap_pools(
            _rebuild(td, pd_, rk, cps, t_big, m, T, u, arm, full_lp, full_conf), theta, m + 2)[0],
            T_LO, T_HI, T_TOL)

        pools = _rebuild(td, pd_, rk, cps, t_big, m, T, U, arm, full_lp, full_conf)
        v, n = eval_smap_pools(pools, theta, m + 2)
        trace.append({"iter": outer + 1, "m": m, "theta": round(theta, 3),
                      "T_ratio_frac": round(T, 4), "T_ratio_super": round(U, 4), "rMAE": round(v, 4)})
        print(f"    iter {outer+1}: m={m} θ={theta:.3f} T_frac={T:.4f} T_super_ratio={U:.4f} "
              f"(T_super={t_big/U*100:.2f}%) → rMAE={v:.4f} (n={n})")
        cur = (m, round(theta, 2), round(T, 3), round(U, 3))
        if cur == prev:
            break
        prev = cur
    return m, theta, T, U, v, n, trace


def main():
    t0 = time.time()
    print(f"=== 19c_calibrator === target={TARGET} T_big={T_BIG} arm={ARM}")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, _ = exp17.build_zigzag(lh, ll, dates, T_BIG)
    print(f"Пивотов T_big: {len(full_lp)}")

    def_pools = build_all_pools_super(target_data, peer_data, rankings, checkpoints, T_BIG, DEF_M,
                                       DEF_T_RATIO_FRAC * T_BIG, T_BIG / DEF_T_RATIO_SUPER, ARM, full_lp, full_conf)
    def_v, def_n = eval_smap_pools(def_pools, DEF_THETA, DEF_M + 2)
    print(f"\nДефолт (m={DEF_M} θ={DEF_THETA} T_frac_ratio={DEF_T_RATIO_FRAC} "
          f"T_super_ratio={DEF_T_RATIO_SUPER}): rMAE={def_v:.4f} (n={def_n})\n")

    print("Калибровка S-map (m, θ, T_ratio_frac, T_ratio_super)...")
    t1 = time.time()
    res = calibrate_smap_super(target_data, peer_data, rankings, checkpoints, T_BIG, ARM, full_lp, full_conf)
    m, theta, T, U, v, n, trace = res
    print(f"\n→ m={m} θ={theta:.3f} T_ratio_frac={T:.4f} T_ratio_super={U:.4f} "
          f"(T_super={T_BIG/U*100:.2f}%) rMAE={v:.4f} (n={n})  ({time.time()-t1:.1f}s)")

    # ── сверка с эксп.17f (без T_super) ──
    print(f"\n{'='*70}")
    print(f"Эксп.17f (без T_super, откалиброван):           rMAE=0.6049 (m=3 θ=25.697 T_ratio=0.8987)")
    print(f"Эксп.19 дефолт (T_super добавлен, старая калибр): rMAE={def_v:.4f}")
    print(f"Эксп.19 калиброван (T_super добавлен, новая калибр.): rMAE={v:.4f}")
    print(f"{'='*70}")

    import json
    out = {
        "target": TARGET, "t_big": T_BIG, "arm": ARM,
        "default": {"m": DEF_M, "theta": DEF_THETA, "T_ratio_frac": DEF_T_RATIO_FRAC,
                    "T_ratio_super": DEF_T_RATIO_SUPER, "rMAE": round(def_v, 4), "n": def_n},
        "optimal": {"m": m, "theta": round(theta, 3), "T_ratio_frac": round(T, 4),
                    "T_ratio_super": round(U, 4), "T_super_pct": round(T_BIG / U * 100, 3),
                    "rMAE": round(v, 4), "n": n, "trace": trace},
        "reference_exp17f_no_super": {"m": 3, "theta": 25.697, "T_ratio": 0.8987, "rMAE": 0.6049},
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_path = RESULTS / f"calibrator_super_{TARGET}_T{int(T_BIG*100):03d}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Сохранено: {out_path}")
    print(f"Всего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
