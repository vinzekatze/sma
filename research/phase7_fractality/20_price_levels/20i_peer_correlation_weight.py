#!/usr/bin/env python3
"""
20i_peer_correlation_weight.py — дешёвый прокси "совместимости пира":
корреляция доходностей (уже считается в exp17.compute_peer_rankings для
отбора C_top8/D_allpeers, здесь используется как МЯГКИЙ ВЕС, не фильтр).

w = exp(-θ·d/d_mean) × exp(λ_peer · corr_j)

corr_j — корреляция пира-источника строки j с целевым тикером на момент
confirm_date (тот же каузальный расчёт, что и для C_top8/D_allpeers, окно
756 дней, ранжирование фиксируется на год вперёд). Для строк own_big/
own_frac (сам целевой тикер) corr=1.0 (максимальная "совместимость" по
определению).

⚠️ Явная оговорка (проговорено с пользователем): это НЕ "историческая
полезность соседа по факту точности прошлых прогнозов" (дороже, не делаем
сегодня) — а обычная корреляция доходностей, тот же прокси, что уже
использовался в H3 эксп.17 (там — как фильтр, слабый эффект). Проверяем
гипотезу "тот же прокси, но как вес, а не фильтр" — по методологическому
выводу сессии ([[feedback-kernel-weight-over-filter-correction]]) вес
должен работать лучше фильтра.

Старые (m,θ,T_ratio) из эксп.17f/17r — НЕ пересчитываются (20g показал,
что пересчёт не окупается). λ_peer калибруется LOO по 7 тикерам, как λ
для объёма (эксп.20e).
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

CALIB = {   # старые (m,θ,T_ratio) — эксп.17f/17r, без пересчёта
    "SBER": {"m": 3, "theta": 25.697, "T_ratio": 0.8987},
    "LKOH": {"m": 4, "theta": 24.817, "T_ratio": 0.8331},
    "CHMF": {"m": 3, "theta": 11.492, "T_ratio": 0.9238},
    "NVTK": {"m": 2, "theta": 15.158, "T_ratio": 0.8080},
    "MGNT": {"m": 2, "theta": 11.814, "T_ratio": 0.6110},
    "VTBR": {"m": 3, "theta": 0.731,  "T_ratio": 0.6169},
    "NLMK": {"m": 2, "theta": 2.315,  "T_ratio": 0.8641},
}
LAMBDA_PEER_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def compute_peer_correlations(target_data, peer_data, checkpoints):
    """Копия причинной части exp17.compute_peer_rankings, но сохраняет
    СЫРЫЕ значения корреляции (там они использовались только для ранжирования
    и отбрасывались)."""
    t_dates, t_lc = target_data["dates"], target_data["lc"]
    corr_by_cp = {}
    for cp in checkpoints:
        cutoff = int(np.searchsorted(t_dates, cp, side="left"))
        if cutoff < exp17.MIN_CORR_HIST_DAYS + 1:
            corr_by_cp[cp] = {}
            continue
        t_dates_slice = t_dates[:cutoff][-exp17.CORR_WINDOW_DAYS:]
        t_lc_slice = t_lc[:cutoff][-exp17.CORR_WINDOW_DAYS:]
        t_ret = np.diff(t_lc_slice)
        t_ret_dates = t_dates_slice[1:]
        corrs = {}
        for peer in exp17.PEERS:
            p_dates, p_lc = peer_data[peer]["dates"], peer_data[peer]["lc"]
            p_cutoff = int(np.searchsorted(p_dates, cp, side="left"))
            if p_cutoff < 2:
                continue
            p_dates_slice = p_dates[:p_cutoff]
            p_lc_slice = p_lc[:p_cutoff]
            p_ret = np.diff(p_lc_slice)
            p_ret_dates = p_dates_slice[1:]
            common, ti, pi = np.intersect1d(t_ret_dates, p_ret_dates, return_indices=True)
            if len(common) < 100:
                continue
            c = np.corrcoef(t_ret[ti], p_ret[pi])[0, 1]
            if np.isfinite(c):
                corrs[peer] = c
        corr_by_cp[cp] = corrs
    return corr_by_cp


def corr_for_date(corr_by_cp, checkpoints, date, peer):
    idx = int(np.searchsorted(checkpoints, date, side="right")) - 1
    if idx < 0:
        return 0.0
    return corr_by_cp[checkpoints[idx]].get(peer, 0.0)


def _smap_corr(qvec, feats, tgts, corrs_j, min_pool, theta, lam_peer):
    if len(tgts) < min_pool:
        return np.nan
    d = np.linalg.norm(feats - qvec, axis=1)
    d_mean = d.mean()
    if d_mean < 1e-14:
        return float(tgts.mean())
    w_price = np.exp(-theta * d / d_mean)
    w_peer = np.exp(lam_peer * corrs_j) if lam_peer != 0 else np.ones_like(d)
    w = w_price * w_peer
    sw = np.sqrt(w)
    A = np.column_stack([np.ones(len(tgts)), feats]) * sw[:, None]
    b = tgts * sw
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(c[0] + c[1:] @ qvec)


def run_walkforward(target, m, theta, t_ratio):
    target_data = exp17.load_ticker(target)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != target]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != target}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)
    corr_by_cp = compute_peer_correlations(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
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

        pool_feats, pool_tgts, pool_dirs, pool_corr = [], [], [], []

        def add(lp, dirs, corr_val):
            f, tg, dd = exp17.build_pool_rows(lp, dirs, m)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)
            pool_corr.append(np.full(len(tg), corr_val))

        add(own_big_lp, own_big_dir, 1.0)
        own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
        add(own_frac_lp, own_frac_dir, 1.0)

        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, ARM):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < m + 2:
                continue
            p_lp, _, p_dir = exp17.build_zigzag(peer_data[peer]["lh"][:p_cutoff],
                                                 peer_data[peer]["ll"][:p_cutoff],
                                                 p_dates[:p_cutoff], t_frac)
            corr_val = corr_for_date(corr_by_cp, checkpoints, confirm_date, peer)
            add(p_lp, p_dir, corr_val)

        feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts)
        dirs = np.concatenate(pool_dirs); corrv = np.concatenate(pool_corr)
        mask = dirs == q_dir
        feats_d, tgts_d, corr_d = feats[mask], tgts[mask], corrv[mask]
        if len(feats_d):
            d = np.linalg.norm(feats_d - qvec, axis=1)
            dup = d < exp17.DUP_EPS
            if dup.any():
                feats_d, tgts_d, corr_d = feats_d[~dup], tgts_d[~dup], corr_d[~dup]
        if len(tgts_d) < m + 2:
            continue

        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)

        row = {"ticker": target, "step": i, "P_log": P_log, "actual_price": actual_price, "pers_err": pers_err}
        for lam in LAMBDA_PEER_GRID:
            row[f"lr_{lam}"] = _smap_corr(qvec, feats_d, tgts_d, corr_d, m + 2, theta, lam)
        records.append(row)
    return pd.DataFrame(records)


def rmae(col, df):
    pred_price = np.exp(df["P_log"] + df[col])
    abs_err = (pred_price - df["actual_price"]).abs()
    dz = df["pers_err"].mean()
    return float(abs_err.mean() / dz) if dz > 1e-12 else np.nan


def main():
    t0 = time.time()
    print("=== 20i_peer_correlation_weight — корреляция как мягкий вес (LOO) ===\n")

    all_df = {}
    for ticker, params in CALIB.items():
        print(f"Walk-forward {ticker}...")
        df = run_walkforward(ticker, params["m"], params["theta"], params["T_ratio"])
        all_df[ticker] = df
        r_base = rmae("lr_0.0", df)
        print(f"  n={len(df)}  rMAE(λ_peer=0, baseline)={r_base:.4f}")

    combined = pd.concat(all_df.values(), ignore_index=True)
    combined.to_csv(RESULTS / "peer_correlation_weight_raw.csv", index=False, float_format="%.6f")

    print(f"\n{'='*70}\n=== LOO: λ_peer выбирается на остальных 6, применяется к отложенному ===")
    loo_rows = []
    for held_out in CALIB:
        other = pd.concat([all_df[t] for t in CALIB if t != held_out], ignore_index=True)
        best_lam, best_v = None, float("inf")
        for lam in LAMBDA_PEER_GRID:
            v = rmae(f"lr_{lam}", other)
            if v < best_v:
                best_v, best_lam = v, lam
        target_df = all_df[held_out]
        r_base = rmae("lr_0.0", target_df)
        r_peer = rmae(f"lr_{best_lam}", target_df)
        print(f"  {held_out:<6s}  chosen λ_peer={best_lam} (rMAE_on_other6={best_v:.4f})  "
              f"base={r_base:.4f}  peer_weighted={r_peer:.4f}  Δ={r_peer-r_base:+.4f}")
        loo_rows.append({"ticker": held_out, "rMAE_base": r_base, "lambda_peer_chosen": best_lam,
                          "rMAE_peer_weighted": r_peer, "delta": r_peer - r_base})

    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(RESULTS / "peer_correlation_weight_loo_results.csv", index=False, float_format="%.5f")
    print(f"\n{'='*70}")
    print(loo_df.to_string(index=False))
    print(f"\nСреднее: base={loo_df['rMAE_base'].mean():.4f}  peer_weighted={loo_df['rMAE_peer_weighted'].mean():.4f}")
    print(f"Улучшение на: {(loo_df['delta']<0).sum()}/7 тикерах")

    print(f"\nСохранено: {RESULTS}/peer_correlation_weight_raw.csv, peer_correlation_weight_loo_results.csv")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
