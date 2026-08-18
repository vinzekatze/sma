#!/usr/bin/env python3
"""
19f_t_ensemble.py — усреднение прогнозов по НЕСКОЛЬКИМ близким T_big
(не по θ, как в 19d — там не сработало; и не по плавающему T_live, как в
19e — там пул разваливается на разреженных крупных T).

Мотивация (пользователь): T резко (порогово) определяет масштаб структуры
зигзага, но в реальности эта структура, вероятно, "плавающая", не имеет
такой резкой границы. Дебаг T_live (19e) прямо показал: зигзаг хаотично
чувствителен к порогу (соседние T дают несовпадающие цепочки пивотов —
"эффект бабочки"). Гипотеза: усреднение прогнозов по узкой сетке соседних
T_big могло бы сгладить именно эту чувствительность/резкость.

Механика: для каждого origin (walk-forward шаг по T_big-зигзагу, тот же
cutoff_idx, что и раньше) для КАЖДОГО T' из сетки строится СВОЙ причинный
зигзаг на ТЕХ ЖЕ обрезанных данных → свой последний подтверждённый пивот
P' (не обязательно совпадает с P исходного T_big — своя дата/цена), свой
вектор запроса, свой пул (own T' legs + T_frac'=0.8987×T' свой+кросс-
тикерный D_allpeers), прогноз той же калиброванной методологией (m=3,
θ=25.697). Результат — pred_price' (абсолютная цена, не лог-доходность от
общего якоря, т.к. у каждого T' свой якорь). Усредняем `log(pred_price')`
по сетке T' (простое среднее), экспонента один раз в конце.

Сравнение — с фактическим следующим T_big-пивотом и его persistence,
как и во всех предыдущих экспериментах ветки.

SBER, T_big=20%, D_allpeers, m=3, θ=25.697, T_ratio_frac=0.8987.
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
T_BIG = 0.20
ARM = "D_allpeers"
M = 3
THETA = 25.697
T_RATIO_FRAC = 0.8987

MIN_HIST = exp17.MIN_HIST

# сетки соседних T' для сравнения — от узкой до пошире
T_GRIDS = {
    "single_T_big":      [1.00],
    "narrow_pm5pct":     [0.95, 1.00, 1.05],
    "medium_pm10pct":    [0.90, 0.95, 1.00, 1.05, 1.10],
    "wide_pm20pct":      [0.80, 0.90, 1.00, 1.10, 1.20],
}


def forecast_at_threshold(target_data, peer_data, rankings, checkpoints, t_lh, t_ll, t_dt,
                           confirm_date, t_prime, m):
    """Прогноз следующего пивота на масштабе t_prime, тот же рецепт, что для T_big.
    Возвращает (log_pred_price, P_log) либо None."""
    own_lp, _, own_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_prime)
    if len(own_lp) < m + 1:
        return None
    qvec = np.array([own_lp[-1 - lag] - own_lp[-2 - lag] for lag in range(m)])
    if not np.all(np.isfinite(qvec)):
        return None
    q_dir = int(own_dir[-1])
    P_log = float(own_lp[-1])

    t_frac_prime = T_RATIO_FRAC * t_prime
    pool_feats, pool_tgts, pool_dirs = [], [], []

    def add(lp, dirs):
        f, tg, dd = exp17.build_pool_rows(lp, dirs, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

    add(own_lp, own_dir)
    own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac_prime)
    add(own_frac_lp, own_frac_dir)
    for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
        p_dates = peer_data[peer]["dates"]
        p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
        if p_cutoff < m + 2:
            continue
        p_lp, _, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                             peer_data[peer]["ll"][:p_cutoff],
                                             p_dates[:p_cutoff], t_frac_prime)
        add(p_lp, p_dir)

    feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts); dirs = np.concatenate(pool_dirs)
    mask = dirs == q_dir
    feats_d, tgts_d = feats[mask], tgts[mask]
    if len(feats_d):
        d = np.linalg.norm(feats_d - qvec, axis=1)
        dup = d < exp17.DUP_EPS
        if dup.any():
            feats_d, tgts_d = feats_d[~dup], tgts_d[~dup]

    lr = exp17._smap(qvec, feats_d, tgts_d, m + 2, THETA)
    if not np.isfinite(lr):
        return None
    return P_log + lr   # лог-цена прогноза (абсолютная, не относительная)


def main():
    t0 = time.time()
    print("=== 19f_t_ensemble — усреднение прогноза по сетке соседних T_big ===")
    print(f"Сетки: {T_GRIDS}")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, _ = exp17.build_zigzag(lh, ll, dates, T_BIG)
    n_big = len(full_lp)

    records = []
    for i in range(MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
        t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]

        own_big_lp, _, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, T_BIG)
        if len(own_big_lp) < M + 1 or own_big_lp[-1] != full_lp[i]:
            continue

        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)

        # прогноз на всех уникальных T' по объединению всех сеток (считаем один раз)
        all_ratios = sorted(set(r for grid in T_GRIDS.values() for r in grid))
        log_preds = {}
        for r in all_ratios:
            t_prime = round(T_BIG * r, 6)
            lp = forecast_at_threshold(target_data, peer_data, rankings, checkpoints,
                                        t_lh, t_ll, t_dt, confirm_date, t_prime, M)
            log_preds[r] = lp

        row = {"step": i, "pers_err": pers_err}
        for grid_name, ratios in T_GRIDS.items():
            vals = [log_preds[r] for r in ratios if log_preds.get(r) is not None]
            if len(vals) == 0:
                row[f"abs_err__{grid_name}"] = np.nan
                continue
            avg_log_pred = float(np.mean(vals))
            pred_price = float(np.exp(avg_log_pred))
            row[f"abs_err__{grid_name}"] = abs(pred_price - actual_price)
            row[f"n_valid__{grid_name}"] = len(vals)
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(RESULTS / "t_ensemble.csv", index=False, float_format="%.6f")

    dz = df["pers_err"].mean()
    print(f"\nn_origins={len(df)}")
    for grid_name in T_GRIDS:
        col = f"abs_err__{grid_name}"
        valid = df[col].dropna()
        rmae = float(valid.mean() / dz) if len(valid) > 3 else np.nan
        print(f"  {grid_name:<18s} rMAE={rmae:.4f}  (n={len(valid)})")

    print(f"\nСохранено: {RESULTS / 't_ensemble.csv'}")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
