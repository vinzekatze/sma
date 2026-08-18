#!/usr/bin/env python3
"""
20d_density_correction.py — коррекция прогноза S-map по объёмной плотности
В ТОЧКЕ ПРОГНОЗА (не по отдельно найденному уровню, как в 20c — там сигнала
не было). Идея пользователя: если предсказанная точка попадает в зону с
плотностью объёма ВЫШЕ типичной для настоящего пивота (~0.10-0.16
перцентиля, находка 20b) — вероятно, недолёт; доразгоняем прогноз дальше
в том же направлении пропорционально избытку плотности.

corrected_leg = pred_leg × (1 + k × max(0, density_rank_at_pred − τ))

τ, k — калибруются LEAVE-ONE-TICKER-OUT (как в 19j): для held-out тикера
используется (τ,k), лучшая на ОСТАЛЬНЫХ 6 тикерах (по гриду), применяется
к отложенному. Никакой информации о held-out тикере в выбор (τ,k) не
попадает.

Экстремальные (шоковые) случаи мы, скорее всего, не поймаем — заведомая
оговорка пользователя. Интересно — работает ли на "обычных" случаях.
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

T_BIG = 0.20
ARM = "D_allpeers"
WINDOW = 252
N_BINS = 50
MIN_HIST_BARS = 60

CALIB = {
    "SBER": {"m": 3, "theta": 25.697, "T_ratio": 0.8987},
    "LKOH": {"m": 4, "theta": 24.817, "T_ratio": 0.8331},
    "CHMF": {"m": 3, "theta": 11.492, "T_ratio": 0.9238},
    "NVTK": {"m": 2, "theta": 15.158, "T_ratio": 0.8080},
    "MGNT": {"m": 2, "theta": 11.814, "T_ratio": 0.6110},
    "VTBR": {"m": 3, "theta": 0.731,  "T_ratio": 0.6169},
    "NLMK": {"m": 2, "theta": 2.315,  "T_ratio": 0.8641},
}

TAU_GRID = [0.10, 0.13, 0.16, 0.20, 0.25]
K_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]


def volume_profile(lh_win, ll_win, vol_win, n_bins):
    lo = np.nanmin(ll_win); hi = np.nanmax(lh_win)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    profile = np.zeros(n_bins)
    binw = (hi - lo) / n_bins
    for j in range(len(lh_win)):
        a, b, v = ll_win[j], lh_win[j], vol_win[j]
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(v)) or b <= a:
            continue
        b0 = int(np.floor((a - lo) / binw)); b1 = int(np.floor((b - lo) / binw))
        b0 = max(0, min(b0, n_bins - 1)); b1 = max(0, min(b1, n_bins - 1))
        if b0 == b1:
            profile[b0] += v
        else:
            span = b - a
            for k in range(b0, b1 + 1):
                bin_lo, bin_hi = lo + k * binw, lo + (k + 1) * binw
                overlap = max(0.0, min(b, bin_hi) - max(a, bin_lo))
                profile[k] += v * (overlap / span)
    return edges, profile


def density_rank(edges, profile, q, ref_log_prices):
    n_bins = len(profile)
    binw = (edges[-1] - edges[0]) / n_bins

    def lookup(x):
        idx = int(np.floor((x - edges[0]) / binw))
        return profile[max(0, min(idx, n_bins - 1))]

    dens_q = lookup(q)
    dens_ref = np.array([lookup(x) for x in ref_log_prices if np.isfinite(x)])
    if len(dens_ref) < MIN_HIST_BARS:
        return np.nan
    return float((dens_ref < dens_q).mean() + 0.5 * (dens_ref == dens_q).mean())


def run_walkforward_with_density(target, m, theta, t_ratio):
    target_data = exp17.load_ticker(target)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != target]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != target}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, lc, dates = target_data["lh"], target_data["ll"], target_data["lc"], target_data["dates"]
    # load_ticker в exp17 не хранит volume — читаем отдельно
    import json
    raw = json.load(open(EXP17_DIR.parents[2] / "data" / "candles" / target / "1d.json"))
    volume = np.array([c["volume"] for c in raw], dtype=np.float64)

    t_frac = t_ratio * T_BIG
    full_lp, full_conf, _ = exp17.build_zigzag(lh, ll, dates, T_BIG)
    n_big = len(full_lp)

    records = []
    for i in range(exp17.MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))
        t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]

        own_big_lp, _, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, T_BIG)
        if len(own_big_lp) < m + 1 or own_big_lp[-1] != full_lp[i]:
            continue
        qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue
        q_dir = int(own_big_dir[-1])
        P_log = float(own_big_lp[-1])

        pool_feats, pool_tgts, pool_dirs = [], [], []
        f, tg, dd = exp17.build_pool_rows(own_big_lp, own_big_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)
        own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
        f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)
        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < m + 2:
                continue
            p_lp, _, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                                 peer_data[peer]["ll"][:p_cutoff],
                                                 p_dates[:p_cutoff], t_frac)
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
        lr = exp17._smap(qvec, feats_d, tgts_d, m + 2, theta)
        if not np.isfinite(lr):
            continue

        # ── объёмная плотность в точке прогноза (каузальное окно до cutoff_idx-1) ──
        bar_idx = cutoff_idx - 1
        start = max(0, bar_idx - WINDOW)
        if bar_idx - start < MIN_HIST_BARS:
            continue
        lh_win, ll_win, vol_win = lh[start:bar_idx+1], ll[start:bar_idx+1], volume[start:bar_idx+1]
        res = volume_profile(lh_win, ll_win, vol_win, N_BINS)
        if res is None:
            continue
        edges, profile = res
        if profile.sum() <= 0:
            continue
        pred_price_log = P_log + lr
        lc_win = lc[start:bar_idx+1]
        rank = density_rank(edges, profile, pred_price_log, lc_win)
        rank_origin = density_rank(edges, profile, P_log, lc_win)
        if not np.isfinite(rank) or not np.isfinite(rank_origin):
            continue

        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)
        actual_leg_abs = abs(full_lp[i + 1] - full_lp[i])

        records.append({"ticker": target, "step": i, "P_log": P_log, "pred_leg": lr,
                         "density_rank_at_pred": rank, "density_rank_at_origin": rank_origin,
                         "actual_price": actual_price, "pers_err": pers_err, "actual_leg_abs": actual_leg_abs})
    return pd.DataFrame(records)


def rmae_with_correction(df, tau, k):
    excess = np.maximum(0.0, df["density_rank_at_pred"] - tau)
    corrected_leg = df["pred_leg"] * (1 + k * excess)
    pred_price = np.exp(df["P_log"] + corrected_leg)
    abs_err = (pred_price - df["actual_price"]).abs()
    dz = df["pers_err"].mean()
    return float(abs_err.mean() / dz) if dz > 1e-12 else np.nan


def main():
    t0 = time.time()
    print("=== 20d_density_correction — LOO-проверка коррекции по плотности в точке прогноза ===\n")

    all_df = {}
    for ticker, params in CALIB.items():
        print(f"Walk-forward+density {ticker}...")
        df = run_walkforward_with_density(ticker, params["m"], params["theta"], params["T_ratio"])
        all_df[ticker] = df
        r_unc = rmae_with_correction(df, tau=1.0, k=0.0)  # tau=1 -> excess всегда 0 -> без коррекции
        print(f"  n={len(df)}  rMAE_uncorrected={r_unc:.4f}  "
              f"mean(density_rank_at_pred)={df['density_rank_at_pred'].mean():.3f}")

    combined = pd.concat(all_df.values(), ignore_index=True)
    combined.to_csv(RESULTS / "density_correction_raw.csv", index=False, float_format="%.6f")

    print(f"\n{'='*70}\n=== Диагностика: density_rank_at_origin (плотность В ТОЧКЕ ОТСЧЁТА) "
          f"vs фактическая величина следующего плеча ===")
    from scipy import stats as scipy_stats
    for ticker, df in all_df.items():
        r, p = scipy_stats.pearsonr(df["density_rank_at_origin"], df["actual_leg_abs"])
        rho, ps = scipy_stats.spearmanr(df["density_rank_at_origin"], df["actual_leg_abs"])
        print(f"  {ticker}: n={len(df):>3d}  r={r:+.3f} (p={p:.4f})  rho={rho:+.3f} (p={ps:.4f})  "
              f"mean_rank_at_origin={df['density_rank_at_origin'].mean():.3f}")
    r_all, p_all = scipy_stats.pearsonr(combined["density_rank_at_origin"], combined["actual_leg_abs"])
    rho_all, ps_all = scipy_stats.spearmanr(combined["density_rank_at_origin"], combined["actual_leg_abs"])
    print(f"  ВСЕ ТИКЕРЫ: n={len(combined)}  r={r_all:+.3f} (p={p_all:.5f})  rho={rho_all:+.3f} (p={ps_all:.5f})  "
          f"mean_rank_at_origin={combined['density_rank_at_origin'].mean():.3f}")

    print(f"\n{'='*70}\n=== LOO: (τ,k) выбирается на остальных 6, применяется к отложенному ===")
    loo_rows = []
    for held_out in CALIB:
        other = pd.concat([all_df[t] for t in CALIB if t != held_out], ignore_index=True)
        best_tau, best_k, best_v = None, None, float("inf")
        for tau in TAU_GRID:
            for k in K_GRID:
                v = rmae_with_correction(other, tau, k)
                if v < best_v:
                    best_v, best_tau, best_k = v, tau, k

        target_df = all_df[held_out]
        r_unc = rmae_with_correction(target_df, tau=1.0, k=0.0)
        r_cor = rmae_with_correction(target_df, best_tau, best_k)
        print(f"  {held_out:<6s}  chosen(τ={best_tau}, k={best_k}, rMAE_on_other6={best_v:.4f})  "
              f"rMAE unc={r_unc:.4f}  rMAE corr={r_cor:.4f}  Δ={r_cor-r_unc:+.4f}")
        loo_rows.append({"ticker": held_out, "tau": best_tau, "k": best_k,
                          "rMAE_uncorrected": r_unc, "rMAE_corrected": r_cor, "delta": r_cor - r_unc})

    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(RESULTS / "density_correction_loo_results.csv", index=False, float_format="%.5f")
    print(f"\n{'='*70}")
    print(loo_df.to_string(index=False))
    print(f"\nСреднее rMAE: без коррекции={loo_df['rMAE_uncorrected'].mean():.4f}  "
          f"с LOO-коррекцией={loo_df['rMAE_corrected'].mean():.4f}")
    print(f"Тикеров, где коррекция помогла: {(loo_df['delta']<0).sum()}/7")

    print(f"\nСохранено: {RESULTS}/density_correction_raw.csv, density_correction_loo_results.csv")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
