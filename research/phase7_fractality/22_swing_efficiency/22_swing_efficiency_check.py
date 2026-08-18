#!/usr/bin/env python3
"""
22_swing_efficiency_check.py — диагностика (ПЕРЕД любым весом, по правилу
[[feedback-kernel-weight-over-filter-correction]]): несёт ли "чистота" пути
конкретного плеча, приведшего к пивоту-запросу, сигнал об ошибке прогноза?

Отличие от уже провалившегося Efficiency Ratio (эксп.17o): там ER считался
на СКОЛЬЗЯЩЕМ окне N=60 баров ПЕРЕД запросом (общая трендовость рынка в
моменте) — здесь ER считается КОНКРЕТНО для того плеча зигзага, которое
только что сформировало текущий пивот-запрос (от предыдущего T_big-пивота
до текущего) — вопрос "насколько монотонно/чисто цена шла именно в этом
конкретном развороте", а не "какой рынок в среднем в последние 60 баров".

ER = |P_log − P_prev_log| / Σ|log_close[t+1] − log_close[t]|  по барам
     этого конкретного плеча (t от бара prev-пивота до бара текущего пивота).
ER→1 — плечо шло почти по прямой; ER→0 — очень "рваный" путь при том же
итоговом смещении.

Проверяем корреляцию ER с abs_err (общая ошибка) и с "недооценкой"
(|факт.плечо| − |predict.плечо|, эксп.19i) — если сигнала нет, останавливаемся
здесь (дёшево), не переходим к весу.

SBER, T_big=20%, D_allpeers, m=3, θ=25.697, T_ratio=0.8987 (эксп.17f).
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
M, THETA, T_RATIO_FRAC = 3, 25.697, 0.8987
MIN_HIST = exp17.MIN_HIST


def swing_efficiency(lc, bar_prev, bar_cur, p_prev_log, p_cur_log):
    """ER конкретного плеча зигзага: |смещение| / Σ|шаг за шагом| по сырому log(close)."""
    if bar_cur <= bar_prev:
        return np.nan
    path = lc[bar_prev:bar_cur + 1]
    path = path[np.isfinite(path)]
    if len(path) < 2:
        return np.nan
    total_path = np.abs(np.diff(path)).sum()
    if total_path < 1e-12:
        return np.nan
    net = abs(p_cur_log - p_prev_log)
    return float(net / total_path)


def main():
    t0 = time.time()
    print("=== 22_swing_efficiency_check — ER конкретного плеча vs ошибка прогноза ===\n")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != TARGET}
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

        own_big_lp, own_big_conf, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, T_BIG)
        if len(own_big_lp) < M + 2 or own_big_lp[-1] != full_lp[i]:
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
        if len(tgts_d) < M + 2:
            continue
        lr = exp17._smap(qvec, feats_d, tgts_d, M + 2, THETA)
        if not np.isfinite(lr):
            continue

        # ── ER плеча, сформировавшего P (от предыдущего T_big-пивота до P) ──
        bar_prev = int(np.searchsorted(dates, own_big_conf[-2], side="right")) - 1
        bar_cur = int(np.searchsorted(dates, own_big_conf[-1], side="right")) - 1
        # net и path — оба по log(close), иначе net (по экстремуму high/low пивота)
        # может превысить path (по close), и ER выйдет за пределы [0,1]
        er = swing_efficiency(lc, bar_prev, bar_cur, lc[bar_prev], lc[bar_cur])
        if not np.isfinite(er):
            continue

        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)
        abs_err = abs(float(np.exp(P_log + lr)) - actual_price)
        actual_leg_abs = abs(full_lp[i + 1] - full_lp[i])
        underest = actual_leg_abs - abs(lr)

        records.append({"step": i, "er": er, "abs_err": abs_err, "pers_err": pers_err, "underest": underest})

    df = pd.DataFrame(records)
    df.to_csv(RESULTS / "swing_efficiency_check.csv", index=False, float_format="%.6f")
    print(f"n_origins={len(df)}  ER: mean={df['er'].mean():.3f} median={df['er'].median():.3f} "
          f"std={df['er'].std():.3f}\n")

    for target_col, label in [("abs_err", "абсолютная ошибка"), ("underest", "недооценка (факт-прогноз)")]:
        r, p = stats.pearsonr(df["er"], df[target_col])
        rho, ps = stats.spearmanr(df["er"], df[target_col])
        print(f"ER vs {label}: r={r:+.3f} (p={p:.4f})  rho={rho:+.3f} (p={ps:.4f})")

    # терции ER — rMAE по бину
    df["tercile"] = pd.qcut(df["er"], 3, labels=["low_ER(рваное)", "mid_ER", "high_ER(чистое)"])
    print("\nrMAE по терциям ER:")
    for terc in ["low_ER(рваное)", "mid_ER", "high_ER(чистое)"]:
        g = df[df["tercile"] == terc]
        dz = g["pers_err"].mean()
        rmae = g["abs_err"].mean() / dz if dz > 1e-12 else np.nan
        print(f"  {terc:<18s} n={len(g):>3d}  mean_ER={g['er'].mean():.3f}  rMAE={rmae:.4f}")

    print(f"\nСохранено: {RESULTS / 'swing_efficiency_check.csv'}")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
