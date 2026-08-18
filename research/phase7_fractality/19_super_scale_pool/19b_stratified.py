#!/usr/bin/env python3
"""
19b_stratified.py — уточнение Фазы A: проверка не среднего rMAE по всем
origin, а rMAE ОТДЕЛЬНО для запросов с крупным недавним движением vs типичным.

Мотивация (пользователь, по 3D-визуализации): крупным движениям (запрос с
большим |qvec|) часто не хватает крупных соседей в пуле — систематическая
недооценка (Stage 1g) может быть локализована именно в этом подмножестве,
а средний rMAE по всем 42 origin (Фаза A, 19_quick_test.py) мог смазать
эффект, если для «типичных» origin добавка T_super лишь шумит (раздувает
d_mean в S-map весах: w_j = exp(-θ·d_j/d_mean), нерелевантные далёкие
T_super-строки увеличивают знаменатель для ВСЕХ, включая типичные, запросы).

Никакой нормировки признаков не вводится — сырые лог-диффы, как и везде.

Метод: ||qvec|| (норма вектора последних m известных плечей T_big) как мера
«насколько крупным было последнее движение». Терции по этой норме (n=42 →
~14 на терцию, экспериментально малая выборка, только для ориентира).
rMAE считается отдельно для baseline (без T_super) и каждой точки сетки
T_ratio_super внутри каждой терции.
"""
import importlib.util
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

spec19 = importlib.util.spec_from_file_location("exp19", HERE / "19_quick_test.py")
exp19 = importlib.util.module_from_spec(spec19)
spec19.loader.exec_module(exp19)  # запускает main() НЕ вызывается (guard __main__)

exp17 = exp19.exp17
TARGET, T_BIG, ARM = exp19.TARGET, exp19.T_BIG, exp19.ARM
M, THETA, T_FRAC = exp19.M, exp19.THETA, exp19.T_FRAC
T_RATIO_SUPER_GRID = exp19.T_RATIO_SUPER_GRID
MIN_HIST = exp19.MIN_HIST


def walk_forward_detailed(target_data, peer_data, rankings, checkpoints, full_lp, full_conf, t_frac, t_super):
    n_big = len(full_lp)
    records = []
    for i in range(MIN_HIST, n_big - 1):
        res = exp19.build_pool_with_super(target_data, peer_data, rankings, checkpoints, i, full_lp, full_conf,
                                           t_frac, t_super)
        if res is None:
            continue
        feats_d, tgts_d, qvec, cur_lp, actual_price, pers_price, n_dup_by_src, n_pool_by_src = res
        lr = exp17._smap(qvec, feats_d, tgts_d, M + 2, THETA)
        pers_err = abs(actual_price - pers_price)
        abs_err = abs(float(np.exp(cur_lp + lr)) - actual_price) if np.isfinite(lr) else np.nan
        actual_leg_abs = float(abs(full_lp[i + 1] - full_lp[i]))
        records.append({
            "step": i, "qvec_norm": float(np.linalg.norm(qvec)), "last_leg_abs": float(abs(qvec[0])),
            "actual_leg_abs": actual_leg_abs,
            "pers_err": pers_err, "abs_err": abs_err,
        })
    return pd.DataFrame(records)


def main():
    t0 = time.time()
    print("=== 19b_stratified — rMAE по терциям |qvec| (крупный/типичный запрос) ===")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, _ = exp17.build_zigzag(lh, ll, dates, T_BIG)

    configs = [("baseline_no_super", None)] + [
        (f"T_ratio_super={tr:.4f}", T_BIG / tr) for tr in T_RATIO_SUPER_GRID
    ]

    wide = None
    for label, t_super in configs:
        print(f"\n[{label}]...")
        df = walk_forward_detailed(target_data, peer_data, rankings, checkpoints, full_lp, full_conf, T_FRAC, t_super)
        df = df.rename(columns={"abs_err": f"abs_err__{label}"})
        if wide is None:
            wide = df[["step", "qvec_norm", "last_leg_abs", "actual_leg_abs", "pers_err", f"abs_err__{label}"]]
        else:
            wide = wide.merge(df[["step", f"abs_err__{label}"]], on="step", how="inner")
        print(f"  n={len(df)}")

    wide.to_csv(RESULTS / "phaseA_stratified_wide.csv", index=False, float_format="%.5f")

    def stratify_by(by_col, tag):
        w = wide.copy()
        w["tercile"] = pd.qcut(w[by_col], 3, labels=["small", "mid", "large"])
        print(f"\nГраницы терций {tag}:")
        print(w.groupby("tercile", observed=True)[by_col].agg(["min", "max", "count"]))

        rows = []
        for label, _ in configs:
            col = f"abs_err__{label}"
            for terc in ["small", "mid", "large"]:
                g = w[w["tercile"] == terc]
                valid = g[col].dropna()
                dz = g["pers_err"].mean()
                rmae = float(valid.mean() / dz) if len(valid) > 3 and dz > 1e-12 else np.nan
                rows.append({"config": label, "tercile": terc, "n": len(valid), "rMAE": rmae})
            valid_all = w[col].dropna()
            dz_all = w["pers_err"].mean()
            rows.append({"config": label, "tercile": "ALL", "n": len(valid_all),
                         "rMAE": float(valid_all.mean() / dz_all) if len(valid_all) > 3 else np.nan})

        res = pd.DataFrame(rows)
        res.to_csv(RESULTS / f"phaseA_stratified_rmae_{tag}.csv", index=False, float_format="%.5f")
        print(f"\n=== rMAE по терциям {tag} и конфигурации ===")
        pivot = res.pivot(index="config", columns="tercile", values="rMAE")
        pivot = pivot[["small", "mid", "large", "ALL"]]
        print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
        return res

    stratify_by("qvec_norm", "qvec_norm_PAST")
    stratify_by("actual_leg_abs", "actual_leg_FUTURE")

    print(f"\nСохранено: {RESULTS}/phaseA_stratified_wide.csv, phaseA_stratified_rmae_*.csv")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
