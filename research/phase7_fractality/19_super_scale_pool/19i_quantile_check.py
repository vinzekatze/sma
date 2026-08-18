#!/usr/bin/env python3
"""
19i_quantile_check.py — нелинейная/квантильная проверка смещения амплитуды:
линейная коррекция (actual≈a+b·pred, эксп.17i/17k) дала b≈1.0-1.02 на
большом пуле — глобального ЛИНЕЙНОГО смещения нет. Но если недооценка
концентрируется в определённых квантилях предсказанной величины (а не
размазана линейно по всему диапазону), линейная регрессия её не увидит —
компенсируется противоположным смещением в другом квантиле.

Здесь: бины по |предсказанное плечо| (терции — как и в 19b, для
сопоставимости), в каждом бине — mean(|pred|), mean(|actual|),
mean(actual)/mean(pred) (локальный коэффициент, аналог b, но per-bin,
не один глобальный).

SBER, T_big=20%, D_allpeers, m=3, θ=25.697, T_ratio_frac=0.8987 (эксп.17f).
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


def main():
    t0 = time.time()
    print("=== 19i_quantile_check — недооценка амплитуды по квантилям предсказанного плеча ===")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    T_FRAC = T_RATIO_FRAC * T_BIG
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
        qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(M)])
        if not np.all(np.isfinite(qvec)):
            continue
        q_dir = int(own_big_dir[-1])

        pool_feats, pool_tgts, pool_dirs = [], [], []
        f, tg, dd = exp17.build_pool_rows(own_big_lp, own_big_dir, M)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)
        own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, T_FRAC)
        f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, M)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)
        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < M + 2:
                continue
            p_lp, _, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                                 peer_data[peer]["ll"][:p_cutoff],
                                                 p_dates[:p_cutoff], T_FRAC)
            f, tg, dd = exp17.build_pool_rows(p_lp, p_dir, M)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)
        feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts); dirs = np.concatenate(pool_dirs)
        mask = dirs == q_dir
        feats_d, tgts_d = feats[mask], tgts[mask]
        if len(feats_d):
            d = np.linalg.norm(feats_d - qvec, axis=1)
            dup = d < exp17.DUP_EPS
            if dup.any():
                feats_d, tgts_d = feats_d[~dup], tgts_d[~dup]
        lr = exp17._smap(qvec, feats_d, tgts_d, M + 2, THETA)
        if not np.isfinite(lr):
            continue

        actual_leg = float(full_lp[i + 1] - full_lp[i])
        records.append({"step": i, "pred_leg": lr, "actual_leg": actual_leg,
                         "pred_abs": abs(lr), "actual_abs": abs(actual_leg),
                         "same_sign": int(np.sign(lr) == np.sign(actual_leg))})

    df = pd.DataFrame(records)
    df.to_csv(RESULTS / "quantile_check.csv", index=False, float_format="%.6f")
    print(f"n_origins={len(df)}   доля совпадения знака (перелёт/недолёт направления): "
          f"{df['same_sign'].mean():.3f}\n")

    # терции по ПРЕДСКАЗАННОЙ величине (не по фактической, как в 19b — это новый угол)
    df["tercile"] = pd.qcut(df["pred_abs"], 3, labels=["small_pred", "mid_pred", "large_pred"])
    print("Бины по |предсказанное плечо| (терции):")
    print(f"{'bin':<12} {'n':>3} {'mean|pred|':>11} {'mean|actual|':>13} {'actual/pred':>12} {'median|pred|':>13} {'median|actual|':>15}")
    rows = []
    for terc in ["small_pred", "mid_pred", "large_pred"]:
        g = df[df["tercile"] == terc]
        mp, ma = g["pred_abs"].mean(), g["actual_abs"].mean()
        ratio = ma / mp if mp > 1e-12 else np.nan
        print(f"{terc:<12} {len(g):>3} {mp:>11.5f} {ma:>13.5f} {ratio:>12.3f} "
              f"{g['pred_abs'].median():>13.5f} {g['actual_abs'].median():>15.5f}")
        rows.append({"tercile": terc, "n": len(g), "mean_pred_abs": mp, "mean_actual_abs": ma,
                     "actual_over_pred": ratio})

    # тоже глобально, для сверки с 17i/17k (b≈1.0)
    mp_all, ma_all = df["pred_abs"].mean(), df["actual_abs"].mean()
    print(f"\n{'ALL':<12} {len(df):>3} {mp_all:>11.5f} {ma_all:>13.5f} {ma_all/mp_all:>12.3f}")
    rows.append({"tercile": "ALL", "n": len(df), "mean_pred_abs": mp_all, "mean_actual_abs": ma_all,
                 "actual_over_pred": ma_all / mp_all})

    pd.DataFrame(rows).to_csv(RESULTS / "quantile_check_bins.csv", index=False, float_format="%.5f")
    print(f"\nСохранено: {RESULTS}/quantile_check.csv, quantile_check_bins.csv")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
