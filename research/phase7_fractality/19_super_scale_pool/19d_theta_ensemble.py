#!/usr/bin/env python3
"""
19d_theta_ensemble.py — проверка идеи: усреднение прогнозов S-map по
нескольким θ в узком диапазоне вокруг калиброванного оптимума может дать
положительный эффект (bagging-подобная регуляризация против
переобучения θ на тех же 42 origin, на которых он калибровался).

Проверяется на двух конфигурациях (пул строится ОДИН РАЗ на каждую,
дальше θ варьируется дёшево — sm-предсказание пересчитывается по готовому
пулу без перестройки зигзагов):
  A) baseline (эксп.17f, без T_super): m=3, T_ratio_frac=0.8987, θ*=25.697
  B) T_super-конфигурация (эксп.19c): m=2, T_ratio_frac=0.9046,
     T_ratio_super=0.7770, θ*=28.384

Для каждой конфигурации: мелкий свип θ (шаг 1.0) на готовом пуле →
lr(θ) на каждый origin. Дальше для набора полуширин Δ усредняем lr по всем
θ в [θ*-Δ, θ*+Δ] (равномерно, простое среднее) и считаем rMAE усреднённого
прогноза. Δ=0 воспроизводит одноточечный калиброванный результат (сверка).

Дополнительно — разбивка по терциям фактического будущего плеча (как в
19b_stratified.py), чтобы посмотреть, не помогает ли усреднение именно там,
где baseline слабее всего.
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

spec17f = importlib.util.spec_from_file_location("exp17f", EXP17_DIR / "17f_calibrator.py")
exp17f = importlib.util.module_from_spec(spec17f)
spec17f.loader.exec_module(exp17f)

spec19c = importlib.util.spec_from_file_location("exp19c", HERE / "19c_calibrator.py")
exp19c = importlib.util.module_from_spec(spec19c)
spec19c.loader.exec_module(exp19c)

TARGET = "SBER"
T_BIG = 0.20
ARM = "D_allpeers"

THETA_GRID = np.arange(2.0, 51.0, 1.0)   # мелкий свип, шаг 1.0, дёшево на готовом пуле
HALF_WIDTHS = [0, 1, 2, 3, 5, 8, 12, 18, 24]

CONFIG_A = {"name": "A_baseline_no_super", "m": 3, "T_ratio_frac": 0.8987, "T_super_ratio": None, "theta_star": 25.697}
CONFIG_B = {"name": "B_with_super", "m": 2, "T_ratio_frac": 0.9046, "T_super_ratio": 0.7770, "theta_star": 28.384}


def sweep_theta(pools, theta_grid, min_pool):
    """Возвращает DataFrame: строка на origin, столбцы lr@theta, + cur_lp, actual_price, pers_err, actual_leg_abs."""
    rows = []
    for qvec, feats_d, tgts_d, d, cur_lp, actual_price, pers_err in pools:
        row = {"cur_lp": cur_lp, "actual_price": actual_price, "pers_err": pers_err}
        for th in theta_grid:
            lr = exp17._smap(qvec, feats_d, tgts_d, min_pool, th)
            row[f"lr_{th:.1f}"] = lr
        rows.append(row)
    return pd.DataFrame(rows)


def build_pools_for_config(cfg, target_data, peer_data, rankings, checkpoints, full_lp, full_conf):
    if cfg["T_super_ratio"] is None:
        t_frac = cfg["T_ratio_frac"] * T_BIG
        pools = exp17f.build_all_pools(target_data, peer_data, rankings, checkpoints, T_BIG, cfg["m"],
                                        t_frac, ARM, full_lp, full_conf)
    else:
        t_frac = cfg["T_ratio_frac"] * T_BIG
        t_super = T_BIG / cfg["T_super_ratio"]
        pools = exp19c.build_all_pools_super(target_data, peer_data, rankings, checkpoints, T_BIG, cfg["m"],
                                              t_frac, t_super, ARM, full_lp, full_conf)
    return pools


def evaluate_ensemble(df, theta_grid, theta_star, half_width):
    center = theta_grid[np.argmin(np.abs(theta_grid - theta_star))]
    sel = theta_grid[np.abs(theta_grid - center) <= half_width]
    cols = [f"lr_{th:.1f}" for th in sel]
    lr_mean = df[cols].mean(axis=1)
    pred_price = np.exp(df["cur_lp"] + lr_mean)
    abs_err = (pred_price - df["actual_price"]).abs()
    dz = df["pers_err"].mean()
    rmae = float(abs_err.mean() / dz) if dz > 1e-12 else np.nan
    return rmae, len(sel), abs_err


def main():
    t0 = time.time()
    print("=== 19d_theta_ensemble — усреднение прогноза по узкому диапазону θ ===")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, _ = exp17.build_zigzag(lh, ll, dates, T_BIG)

    all_results = []
    for cfg in [CONFIG_A, CONFIG_B]:
        print(f"\n--- {cfg['name']}: строю пул (m={cfg['m']}, T_ratio_frac={cfg['T_ratio_frac']}, "
              f"T_super_ratio={cfg['T_super_ratio']}) ---")
        t1 = time.time()
        pools = build_pools_for_config(cfg, target_data, peer_data, rankings, checkpoints, full_lp, full_conf)
        print(f"  пул готов ({time.time()-t1:.1f}s, n_origins={len(pools)})")

        df = sweep_theta(pools, THETA_GRID, cfg["m"] + 2)
        # актуальное будущее плечо для терций (актуально = |actual_price/exp(cur_lp) в лог|, но проще пересчитать из full_lp)
        df["actual_leg_abs"] = (np.log(df["actual_price"]) - df["cur_lp"]).abs()
        df.to_csv(RESULTS / f"theta_sweep_{cfg['name']}.csv", index=False, float_format="%.6f")

        print(f"  {'Δ':>4} {'n_theta':>7} {'rMAE':>8}")
        for hw in HALF_WIDTHS:
            rmae, n_th, abs_err = evaluate_ensemble(df, THETA_GRID, cfg["theta_star"], hw)
            print(f"  {hw:>4} {n_th:>7} {rmae:>8.4f}")
            all_results.append({"config": cfg["name"], "half_width": hw, "n_theta": n_th, "rMAE": rmae})

        # стратификация по терциям будущего плеча, для нескольких характерных Δ
        df["tercile"] = pd.qcut(df["actual_leg_abs"], 3, labels=["small", "mid", "large"])
        print(f"\n  Стратификация по терции будущего плеча ({cfg['name']}):")
        for hw in [0, 3, 8, 18]:
            rmae, n_th, abs_err = evaluate_ensemble(df, THETA_GRID, cfg["theta_star"], hw)
            df["_abs_err_tmp"] = abs_err
            for terc in ["small", "mid", "large"]:
                g = df[df["tercile"] == terc]
                dz = g["pers_err"].mean()
                r = float(g["_abs_err_tmp"].mean() / dz) if dz > 1e-12 else np.nan
                all_results.append({"config": cfg["name"], "half_width": hw, "n_theta": n_th,
                                     "rMAE": r, "tercile": terc})
                print(f"    Δ={hw:>3} {terc:>5}: rMAE={r:.4f} (n={len(g)})")

    res = pd.DataFrame(all_results)
    res.to_csv(RESULTS / "theta_ensemble_results.csv", index=False, float_format="%.5f")
    print(f"\nСохранено: {RESULTS}/theta_sweep_*.csv, theta_ensemble_results.csv")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
