#!/usr/bin/env python3
"""
20h_multistep_volume_weighted.py — итеративный 2-3-шаговый прогноз S-map с
объёмным взвешиванием (λ=2.0 фикс. поверх старых m/θ/T_ratio из
эксп.17f/17r — простой, честно LOO-проверенный вариант, не пересчитанная
совместная калибровка из 20g, где выигрыш сомнителен и вскрылась
нестыковка по LKOH).

Механика — та же, что в каноническом smap_ref.py --steps N / 17m:
сдвиг вектора запроса на предсказанное плечо, разворот направления,
переотбор пула по новому направлению. Отличие с объёмным весом:
rank_query на шаге h считается на ТОМ ЖЕ (реальном, каузальном) объёмном
профиле, что и на шаге 1 (профиль не может "заглянуть в будущее"), но
В НОВОЙ (предсказанной) точке цены — P_log + cumsum(lr_1..lr_{h-1}).
rank_j для соседей пула не пересчитывается между шагами (свойство их
собственных, уже случившихся исторических событий).

Мотивация пользователя: точность шагов h≥2 ощутимо ниже (уже видели в
17m), но зигзаг подтверждает пивот с запаздыванием (Stage 1g/1n) — грубая
многошаговая траектория с объёмным контекстом может быть полезна именно
для этой картины, не как точечный прогноз.

Метрика — как в эксп.18: rMAE по каждому h отдельно, persistence
generalized (pers_h = full_lp[i+h-2]), плюс сравнение λ=2.0 vs λ=0 (без
объёмного веса) на каждом h — видно ли, что объёмный вес помогает
СИЛЬНЕЕ на дальних шагах (гипотеза пользователя) или одинаково слабо.
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

spec20e = importlib.util.spec_from_file_location("exp20e", HERE / "20e_volume_weighting_comparison.py")
exp20e = importlib.util.module_from_spec(spec20e)
spec20e.loader.exec_module(exp20e)

T_BIG = 0.20
ARM = "D_allpeers"
N_STEPS = 3
LAMBDA = 2.0

CALIB = {   # старые (m,θ,T_ratio) — эксп.17f/17r, без пересчёта
    "SBER": {"m": 3, "theta": 25.697, "T_ratio": 0.8987},
    "LKOH": {"m": 4, "theta": 24.817, "T_ratio": 0.8331},
    "CHMF": {"m": 3, "theta": 11.492, "T_ratio": 0.9238},
    "NVTK": {"m": 2, "theta": 15.158, "T_ratio": 0.8080},
    "MGNT": {"m": 2, "theta": 11.814, "T_ratio": 0.6110},
    "VTBR": {"m": 3, "theta": 0.731,  "T_ratio": 0.6169},
    "NLMK": {"m": 2, "theta": 2.315,  "T_ratio": 0.8641},
}


def build_pool_full(target, m, t_ratio, target_data, peer_data, rankings, checkpoints,
                     t_lh, t_ll, t_dt, confirm_date):
    """Пул БЕЗ фильтра по направлению (нужен для многошаговости — на каждом шаге своё направление)."""
    t_frac = t_ratio * T_BIG
    own_big_lp, own_big_conf, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, T_BIG)
    if len(own_big_lp) < m + 1:
        return None

    def extra_for_chain(ticker, dates_arr, lp_arr):
        dates_full_t = exp20e.get_raw(ticker)[4]
        out = np.zeros((len(lp_arr), 2))
        for k in range(len(lp_arr)):
            bidx = exp20e.bar_idx_of(dates_full_t, dates_arr[k])
            out[k, 0] = exp20e.get_rank(ticker, bidx, lp_arr[k])
            out[k, 1] = exp20e.get_logvol(ticker, bidx)
        return out

    pool_feats, pool_tgts, pool_dirs, pool_extra = [], [], [], []
    ex = extra_for_chain(target, own_big_conf, own_big_lp)
    f, tg, dd, ee = exp20e.build_pool_rows_with_extra(own_big_lp, own_big_dir, ex, m)
    pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd); pool_extra.append(ee)

    own_frac_lp, own_frac_conf, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
    ex = extra_for_chain(target, own_frac_conf, own_frac_lp)
    f, tg, dd, ee = exp20e.build_pool_rows_with_extra(own_frac_lp, own_frac_dir, ex, m)
    pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd); pool_extra.append(ee)

    for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
        p_dates = peer_data[peer]["dates"]
        p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
        if p_cutoff < m + 2:
            continue
        p_lp, p_conf, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                                  peer_data[peer]["ll"][:p_cutoff],
                                                  p_dates[:p_cutoff], t_frac)
        ex = extra_for_chain(peer, p_conf, p_lp)
        f, tg, dd, ee = exp20e.build_pool_rows_with_extra(p_lp, p_dir, ex, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd); pool_extra.append(ee)

    feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts)
    dirs = np.concatenate(pool_dirs); extra = np.concatenate(pool_extra)
    return own_big_lp, own_big_dir, feats, tgts, dirs, extra


def iterate_smap_vol(qvec1, qdir1, feats, tgts, dirs, ranks, P_log, edges, profile, lc_win,
                      m, theta, lam, n_steps):
    lrs = []
    qvec = qvec1.copy()
    qdir = qdir1
    cum_lr = 0.0
    for _ in range(n_steps):
        mask = dirs == qdir
        fd, td, rk = feats[mask], tgts[mask], ranks[mask]
        if len(fd):
            dd = np.linalg.norm(fd - qvec, axis=1)
            dup = dd < exp17.DUP_EPS
            if dup.any():
                fd, td, rk = fd[~dup], td[~dup], rk[~dup]
        rank_query = exp20e.density_rank(edges, profile, P_log + cum_lr, lc_win)
        if not np.isfinite(rank_query):
            lrs.append(np.nan)
            break
        lr = exp20e._smap_vol(qvec, fd, td, rk, rank_query, m + 2, theta, lam)
        lrs.append(lr)
        if not np.isfinite(lr):
            break
        cum_lr += lr
        qvec = np.concatenate([[lr], qvec[:-1]])
        qdir = -qdir
    return lrs


def main():
    t0 = time.time()
    print(f"=== 20h_multistep_volume_weighted — N_STEPS={N_STEPS}, λ={LAMBDA} (старые m/θ/T_ratio) ===\n")

    long_rows = []
    for target, params in CALIB.items():
        m, theta, t_ratio = params["m"], params["theta"], params["T_ratio"]
        print(f"--- {target} (m={m} θ={theta} T_ratio={t_ratio}) ---")

        target_data = exp17.load_ticker(target)
        exp17.PEERS = [t for t in exp17.UNIVERSE if t != target]
        peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != target}
        years = sorted(set(int(d[:4]) for d in target_data["dates"]))
        checkpoints = np.array([f"{y}-01-01" for y in years])
        rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

        lh, ll, lc, dates = target_data["lh"], target_data["ll"], target_data["lc"], target_data["dates"]
        full_lp, full_conf, full_dirs = exp17.build_zigzag(lh, ll, dates, T_BIG)
        n_big = len(full_lp)

        n_ok = 0
        for i in range(exp17.MIN_HIST, n_big - 1 - N_STEPS):
            confirm_date = full_conf[i]
            cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
            t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]

            res = build_pool_full(target, m, t_ratio, target_data, peer_data, rankings, checkpoints,
                                   t_lh, t_ll, t_dt, confirm_date)
            if res is None:
                continue
            own_big_lp, own_big_dir, feats, tgts, dirs, extra = res
            if own_big_lp[-1] != full_lp[i]:
                continue
            qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(m)])
            if not np.all(np.isfinite(qvec)):
                continue
            q_dir = int(own_big_dir[-1])
            P_log = float(own_big_lp[-1])

            bar_idx = cutoff_idx - 1
            start = max(0, bar_idx - exp20e.WINDOW)
            if bar_idx - start < exp20e.MIN_HIST_BARS:
                continue
            res_prof = exp20e.volume_profile(lh[start:bar_idx+1], ll[start:bar_idx+1],
                                              exp20e.get_raw(target)[3][start:bar_idx+1], exp20e.N_BINS)
            if res_prof is None:
                continue
            edges, profile = res_prof
            if profile.sum() <= 0:
                continue
            lc_win = lc[start:bar_idx+1]

            lrs_vol = iterate_smap_vol(qvec, q_dir, feats, tgts, dirs, extra[:, 0], P_log,
                                        edges, profile, lc_win, m, theta, LAMBDA, N_STEPS)
            lrs_novol = iterate_smap_vol(qvec, q_dir, feats, tgts, dirs, extra[:, 0], P_log,
                                          edges, profile, lc_win, m, theta, 0.0, N_STEPS)
            if len(lrs_vol) < N_STEPS or not all(np.isfinite(lrs_vol)):
                continue
            if len(lrs_novol) < N_STEPS or not all(np.isfinite(lrs_novol)):
                continue
            n_ok += 1

            cum_vol, cum_novol = 0.0, 0.0
            for h in range(1, N_STEPS + 1):
                cum_vol += lrs_vol[h - 1]
                cum_novol += lrs_novol[h - 1]
                actual_price = float(np.exp(full_lp[i + h]))
                pers_price = float(np.exp(full_lp[i + h - 2])) if i + h - 2 >= 0 else np.nan
                pers_err = abs(actual_price - pers_price)
                long_rows.append({
                    "ticker": target, "step": i, "h": h,
                    "abs_err_vol": abs(float(np.exp(P_log + cum_vol)) - actual_price),
                    "abs_err_novol": abs(float(np.exp(P_log + cum_novol)) - actual_price),
                    "pers_err": pers_err,
                })
        print(f"  n_origins_с_полным_{N_STEPS}-шаговым_прогнозом={n_ok}")

    df = pd.DataFrame(long_rows)
    df.to_csv(RESULTS / "multistep_volume_weighted.csv", index=False, float_format="%.6f")

    print(f"\n{'='*80}\n=== rMAE по горизонту h (λ={LAMBDA} vs λ=0), все тикеры объединённо ===")
    summary_rows = []
    for h in range(1, N_STEPS + 1):
        g = df[df["h"] == h]
        dz = g["pers_err"].mean()
        r_vol = g["abs_err_vol"].mean() / dz
        r_novol = g["abs_err_novol"].mean() / dz
        print(f"  h={h}: n={len(g)}  rMAE(λ=2.0)={r_vol:.4f}  rMAE(λ=0)={r_novol:.4f}  Δ={r_vol-r_novol:+.4f}")
        summary_rows.append({"h": h, "n": len(g), "rMAE_vol": r_vol, "rMAE_novol": r_novol, "delta": r_vol - r_novol})

    print(f"\n=== То же, по тикерам ===")
    for target in CALIB:
        gt = df[df["ticker"] == target]
        for h in range(1, N_STEPS + 1):
            g = gt[gt["h"] == h]
            if len(g) < 3:
                continue
            dz = g["pers_err"].mean()
            r_vol = g["abs_err_vol"].mean() / dz
            r_novol = g["abs_err_novol"].mean() / dz
            print(f"  {target:<6s} h={h}: n={len(g):>3d}  rMAE(λ=2.0)={r_vol:.4f}  rMAE(λ=0)={r_novol:.4f}  Δ={r_vol-r_novol:+.4f}")

    pd.DataFrame(summary_rows).to_csv(RESULTS / "multistep_volume_weighted_summary.csv", index=False, float_format="%.5f")
    print(f"\nСохранено: {RESULTS}/multistep_volume_weighted.csv, multistep_volume_weighted_summary.csv")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
