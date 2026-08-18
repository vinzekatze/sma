#!/usr/bin/env python3
"""
19h_volatility_check.py — корреляция волатильности (перед origin) с ошибкой
калиброванной S-map (эксп.17f), отдельно с абсолютной ошибкой и со
знаковой недооценкой амплитуды (актуальное плечо больше предсказанного).

Не проверялось ранее в этом контексте (проверялся Efficiency Ratio в 17o
— r≈0.036, ноль; волатильность как предиктор ОПТИМАЛЬНОГО T_frac в ранней
малой фазе7 — r≈-0.11, тоже слабо; но не волатильность vs ошибка текущей
крупномасштабной калиброванной S-map).

Волатильность — причинно (только бары до origin): rolling std дневных
лог-доходностей (окно 20 и 60 баров) и ATR-14 (true range / close).

SBER, T_big=20%, D_allpeers, m=3, θ=25.697, T_ratio_frac=0.8987 (эксп.17f).
"""
import importlib.util
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

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


def causal_vol_features(lc, lh, ll, idx_cutoff):
    """Признаки волатильности, используя ТОЛЬКО бары [:idx_cutoff]."""
    lc_c = lc[:idx_cutoff]
    ret = np.diff(lc_c)
    vol20 = float(np.std(ret[-20:])) if len(ret) >= 20 else np.nan
    vol60 = float(np.std(ret[-60:])) if len(ret) >= 60 else np.nan
    # ATR-14: true range приближённо как (high-low) в лог-пространстве / close
    lh_c, ll_c = lh[:idx_cutoff], ll[:idx_cutoff]
    tr = lh_c[-14:] - ll_c[-14:]
    atr14 = float(np.mean(tr)) if len(tr) >= 14 else np.nan
    return vol20, vol60, atr14


def main():
    t0 = time.time()
    print("=== 19h_volatility_check ===")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, lc, dates = target_data["lh"], target_data["ll"], target_data["lc"], target_data["dates"]
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
        P_log = float(own_big_lp[-1])

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

        actual_leg = float(full_lp[i + 1] - full_lp[i])          # знаковая лог-доходность (факт)
        pred_leg = lr                                             # знаковая лог-доходность (прогноз)
        actual_price = float(np.exp(full_lp[i + 1]))
        pred_price = float(np.exp(P_log + lr))
        abs_err = abs(pred_price - actual_price)
        underest = abs(actual_leg) - abs(pred_leg)   # >0 → недооценили размер плеча

        vol20, vol60, atr14 = causal_vol_features(lc, lh, ll, cutoff_idx)

        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)

        records.append({"step": i, "abs_err": abs_err, "underest": underest, "pers_err": pers_err,
                         "vol20": vol20, "vol60": vol60, "atr14": atr14})

    df = pd.DataFrame(records)
    df.to_csv(RESULTS / "volatility_check.csv", index=False, float_format="%.6f")
    print(f"n_origins={len(df)}\n")

    print("Корреляция волатильности с абсолютной ошибкой (rMAE-компонента) и со знаковой недооценкой:")
    for feat in ["vol20", "vol60", "atr14"]:
        sub = df.dropna(subset=[feat])
        r_abs, p_abs = stats.pearsonr(sub[feat], sub["abs_err"])
        r_und, p_und = stats.pearsonr(sub[feat], sub["underest"])
        rho_abs, _ = stats.spearmanr(sub[feat], sub["abs_err"])
        rho_und, _ = stats.spearmanr(sub[feat], sub["underest"])
        print(f"  {feat:<8s} (n={len(sub)}):  abs_err  r={r_abs:+.3f} (p={p_abs:.3f})  rho={rho_abs:+.3f}   |   "
              f"underest  r={r_und:+.3f} (p={p_und:.3f})  rho={rho_und:+.3f}")

    print(f"\nСохранено: {RESULTS / 'volatility_check.csv'}")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
