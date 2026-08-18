#!/usr/bin/env python3
"""
19j_quantile_correction_loo.py — честная каузальная проверка квантильной
(не линейной) коррекции амплитуды, найденной в 19i на SBER: недооценка
сосредоточена в бинах, где сама модель предсказывает МЕЛКОЕ/СРЕДНЕЕ плечо
(ratio actual/pred ≈1.2-1.3), а не в бине "модель предсказывает крупное"
(ratio≈1.0).

Ловушка 17i→17k: in-sample коэффициент на одном тикере выглядел
многообещающе, на честном каузальном тесте (большой пул) свёлся к шуму.
Здесь — то же самое лекарство: 7 тикеров, каждый со своей калибровкой
(эксп.17f/17r), LEAVE-ONE-TICKER-OUT — коррекция (бины + множитель на бин)
считается на ОСТАЛЬНЫХ 6 тикерах, применяется к отложенному 7-му.
Никакой информации о самом отложенном тикере в калибровку коррекции не
попадает.

Дополнительно (по просьбе пользователя) — показываю квантильную таблицу
ОТДЕЛЬНО по каждому тикеру: возможно, коррекция тикер-специфична, а не
универсальна (общая философия проекта — не искать единый рецепт на все
тикеры).

Данные калибровки (эксп.17f + 17r):
  SBER: m=3 θ=25.697  T_ratio=0.8987
  LKOH: m=4 θ=24.817  T_ratio=0.8331
  CHMF: m=3 θ=11.492  T_ratio=0.9238
  NVTK: m=2 θ=15.158  T_ratio=0.8080
  MGNT: m=2 θ=11.814  T_ratio=0.6110
  VTBR: m=3 θ=0.731   T_ratio=0.6169
  NLMK: m=2 θ=2.315   T_ratio=0.8641
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

CALIB = {
    "SBER": {"m": 3, "theta": 25.697, "T_ratio": 0.8987},
    "LKOH": {"m": 4, "theta": 24.817, "T_ratio": 0.8331},
    "CHMF": {"m": 3, "theta": 11.492, "T_ratio": 0.9238},
    "NVTK": {"m": 2, "theta": 15.158, "T_ratio": 0.8080},
    "MGNT": {"m": 2, "theta": 11.814, "T_ratio": 0.6110},
    "VTBR": {"m": 3, "theta": 0.731,  "T_ratio": 0.6169},
    "NLMK": {"m": 2, "theta": 2.315,  "T_ratio": 0.8641},
}
N_BINS = 3


def run_walkforward(target, m, theta, t_ratio):
    target_data = exp17.load_ticker(target)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != target]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE if t != target}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

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

        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)

        records.append({"ticker": target, "step": i, "pred_leg": lr,
                         "actual_leg": float(full_lp[i + 1] - full_lp[i]),
                         "P_log": P_log, "actual_price": actual_price, "pers_err": pers_err})
    return pd.DataFrame(records)


def fit_bin_correction(pool_df, n_bins):
    """Квантильные границы и множитель (mean|actual|/mean|pred|) по бину, на объединённом пуле."""
    pred_abs = pool_df["pred_leg"].abs().values
    actual_abs = pool_df["actual_leg"].abs().values
    edges = np.quantile(pred_abs, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    mult = []
    for b in range(n_bins):
        m_ = (pred_abs >= edges[b]) & (pred_abs < edges[b + 1])
        if m_.sum() < 3:
            mult.append(1.0)
            continue
        mp, ma = pred_abs[m_].mean(), actual_abs[m_].mean()
        mult.append(ma / mp if mp > 1e-12 else 1.0)
    return edges, mult


def apply_correction(pred_abs, edges, mult):
    bins = np.clip(np.searchsorted(edges, pred_abs, side="right") - 1, 0, len(mult) - 1)
    return np.array(mult)[bins]


def rmae(abs_err, pers_err):
    valid = abs_err.dropna()
    dz = pers_err.mean()
    return float(valid.mean() / dz) if len(valid) > 3 and dz > 1e-12 else np.nan


def main():
    t0 = time.time()
    print("=== 19j_quantile_correction_loo — LOO-проверка квантильной коррекции амплитуды ===\n")

    all_df = {}
    for ticker, params in CALIB.items():
        print(f"Walk-forward {ticker} (m={params['m']} θ={params['theta']} T_ratio={params['T_ratio']})...")
        df = run_walkforward(ticker, params["m"], params["theta"], params["T_ratio"])
        df["abs_err_uncorrected"] = (np.exp(df["P_log"] + df["pred_leg"]) - df["actual_price"]).abs()
        all_df[ticker] = df
        print(f"  n={len(df)}  rMAE_uncorrected={rmae(df['abs_err_uncorrected'], df['pers_err']):.4f}")

    combined = pd.concat(all_df.values(), ignore_index=True)
    combined.to_csv(RESULTS / "quantile_loo_raw.csv", index=False, float_format="%.6f")

    # ── описательно: квантильная таблица ОТДЕЛЬНО по каждому тикеру ──
    print("\n=== Квантильная таблица (терции |pred|) по каждому тикеру отдельно ===")
    desc_rows = []
    for ticker, df in all_df.items():
        pred_abs = df["pred_leg"].abs()
        actual_abs = df["actual_leg"].abs()
        terc = pd.qcut(pred_abs, N_BINS, labels=["small", "mid", "large"], duplicates="drop")
        print(f"\n{ticker}:")
        for b in terc.cat.categories:
            m_ = terc == b
            mp, ma = pred_abs[m_].mean(), actual_abs[m_].mean()
            print(f"  {b:<6s} n={m_.sum():>2d}  mean|pred|={mp:.4f}  mean|actual|={ma:.4f}  ratio={ma/mp:.3f}")
            desc_rows.append({"ticker": ticker, "bin": b, "n": int(m_.sum()),
                               "mean_pred": mp, "mean_actual": ma, "ratio": ma / mp})
    pd.DataFrame(desc_rows).to_csv(RESULTS / "quantile_loo_per_ticker_bins.csv", index=False, float_format="%.5f")

    # ── LOO: коррекция калибруется на ОСТАЛЬНЫХ 6, применяется к отложенному ──
    print(f"\n{'='*70}\n=== LOO-коррекция (калибровка на остальных 6 тикерах) ===")
    loo_rows = []
    for held_out in CALIB:
        other = pd.concat([all_df[t] for t in CALIB if t != held_out], ignore_index=True)
        edges, mult = fit_bin_correction(other, N_BINS)

        target_df = all_df[held_out].copy()
        pred_abs = target_df["pred_leg"].abs().values
        corr_factor = apply_correction(pred_abs, edges, mult)
        corrected_pred_leg = target_df["pred_leg"] * corr_factor   # сохраняем знак, растягиваем модуль
        pred_price_corr = np.exp(target_df["P_log"] + corrected_pred_leg)
        abs_err_corr = (pred_price_corr - target_df["actual_price"]).abs()

        r_unc = rmae(target_df["abs_err_uncorrected"], target_df["pers_err"])
        r_cor = rmae(abs_err_corr, target_df["pers_err"])
        print(f"  {held_out:<6s}  edges={np.round(edges,4).tolist()}  mult={np.round(mult,3).tolist()}  "
              f"rMAE unc={r_unc:.4f}  rMAE corr={r_cor:.4f}  Δ={r_cor-r_unc:+.4f}")
        loo_rows.append({"ticker": held_out, "rMAE_uncorrected": r_unc, "rMAE_corrected": r_cor,
                          "delta": r_cor - r_unc, "mult_bins": str(np.round(mult, 3).tolist())})

    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(RESULTS / "quantile_loo_results.csv", index=False, float_format="%.5f")
    print(f"\n{'='*70}")
    print(loo_df.to_string(index=False))
    print(f"\nСреднее rMAE: без коррекции={loo_df['rMAE_uncorrected'].mean():.4f}  "
          f"с LOO-коррекцией={loo_df['rMAE_corrected'].mean():.4f}")
    print(f"Тикеров, где коррекция помогла: {(loo_df['delta']<0).sum()}/{len(loo_df)}")

    print(f"\nСохранено: {RESULTS}/quantile_loo_raw.csv, quantile_loo_per_ticker_bins.csv, quantile_loo_results.csv")
    print(f"Время: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
