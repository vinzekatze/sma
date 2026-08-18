#!/usr/bin/env python3
"""
17l_trend_correction.py — условная (не константная) коррекция амплитуды:
actual_lr ≈ a + (b0 + b1·trend)·smap_lr, trend = qvec[0] (лог-размер
последнего плеча — простейший, наименее переобучаемый прокси моментума).

17k показал: константная поправка (один b) не помогает — на объединённом
пуле b≈1.0-1.02, эффект систематического смещения был в основном шумом
маленькой выборки SBER. Гипотеза здесь другая: нет константного смещения,
но есть УСЛОВНОЕ — сила коррекции зависит от текущего режима (тренд vs
диапазон). Проверяем интеракционный член, не сам факт смещения.

3 параметра вместо 2 в 17k — больше риска переобучения на малых n, поэтому
MIN_FIT увеличен, и подгонка всё так же каузальна и кросс-тикерна (как 17k).
"""
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("blend", HERE / "17h_persistence_blend.py")
blend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blend)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TARGET_LIST = ["SBER", "LKOH", "CHMF", "NVTK", "MGNT", "VTBR", "NLMK"]
MIN_FIT = 20
blend.THETA = 25.697


def build_steps_with_trend(target_data, peer_data, rankings, checkpoints, t_big, m, t_frac, arm, full_lp, full_conf):
    """Как blend.build_steps, но дополнительно хранит qvec[0] (trend proxy)."""
    exp17 = blend.exp17
    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    n_big = len(full_lp)
    steps = []
    for i in range(exp17.MIN_HIST, n_big - 1):
        confirm_date = full_conf[i]
        cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))

        t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]
        own_big_lp, _, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_big)
        if len(own_big_lp) < m + 1 or own_big_lp[-1] != full_lp[i]:
            continue

        qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue
        q_dir = int(own_big_dir[-1])

        own_big_feats, own_big_tgts, own_big_dirs = exp17.build_pool_rows(own_big_lp, own_big_dir, m)
        pool_feats = [own_big_feats]; pool_tgts = [own_big_tgts]; pool_dirs = [own_big_dirs]

        own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
        f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, m)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, arm):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < m + 2:
                continue
            p_lh = peer_data[peer]["lh"][:p_cutoff]
            p_ll = peer_data[peer]["ll"][:p_cutoff]
            p_dt = p_dates[:p_cutoff]
            p_lp, _, p_dir = exp17.build_zigzag(p_lh, p_ll, p_dt, t_frac)
            f, tg, dd = exp17.build_pool_rows(p_lp, p_dir, m)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        feats = np.concatenate(pool_feats); tgts = np.concatenate(pool_tgts); dirs = np.concatenate(pool_dirs)
        mask = dirs == q_dir
        feats_d, tgts_d = feats[mask], tgts[mask]

        d = np.linalg.norm(feats_d - qvec, axis=1) if len(feats_d) else np.empty(0)
        if len(d):
            dup_mask = d < exp17.DUP_EPS
            if dup_mask.any():
                feats_d, tgts_d, d = feats_d[~dup_mask], tgts_d[~dup_mask], d[~dup_mask]

        cur_lp = float(own_big_lp[-1])
        smap_lr = exp17._smap(qvec, feats_d, tgts_d, m + 2, blend.THETA)
        if not np.isfinite(smap_lr):
            continue
        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1]))
        pers_err = abs(pers_price - actual_price)

        steps.append({"i": i, "cur_lp": cur_lp, "smap_lr": smap_lr,
                      "actual_lr": float(np.log(actual_price) - cur_lp),
                      "actual_price": actual_price, "pers_err": pers_err,
                      "trend": float(qvec[0])})
    return steps


def ols(X, y):
    X = np.asarray(X); y = np.asarray(y)
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef  # [a, b0, b1]


def main():
    print("=== 17l_trend_correction ===")
    all_data = {t: blend.exp17.load_ticker(t) for t in blend.exp17.UNIVERSE}

    master = []
    per_ticker_steps = {}
    for target in TARGET_LIST:
        blend.exp17.PEERS = [t for t in blend.exp17.UNIVERSE if t != target]
        target_data = all_data[target]
        years = sorted(set(int(d[:4]) for d in target_data["dates"]))
        checkpoints = np.array([f"{y}-01-01" for y in years])
        rankings = blend.exp17.compute_peer_rankings(target_data, all_data, checkpoints)

        lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
        full_lp, full_conf, full_dirs = blend.exp17.build_zigzag(lh, ll, dates, blend.T_BIG)
        t_frac = blend.T_RATIO * blend.T_BIG

        steps = build_steps_with_trend(target_data, all_data, rankings, checkpoints, blend.T_BIG, blend.M, t_frac, blend.ARM, full_lp, full_conf)
        for s in steps:
            s["confirm_date"] = str(full_conf[s["i"]])
            s["ticker"] = target
        per_ticker_steps[target] = steps
        master.extend(steps)

    master.sort(key=lambda s: s["confirm_date"])
    print(f"Всего в объединённом пуле: {len(master)} шагов\n")

    a0, b0f, b1f = ols([[s["smap_lr"], s["smap_lr"] * s["trend"]] for s in master], [s["actual_lr"] for s in master])
    print(f"Некаузальный fit (in-sample): a={a0:.4f}  b0={b0f:.4f}  b1={b1f:.4f}")
    print(f"  b1 значимо != 0 означало бы условный эффект тренда на амплитуду коррекции\n")

    results = []
    for target in TARGET_LIST:
        steps = sorted(per_ticker_steps[target], key=lambda s: s["confirm_date"])
        errs_raw, errs_corr, dz = [], [], []
        for s in steps:
            prior = [m for m in master if m["confirm_date"] < s["confirm_date"]]
            pred_raw_price = float(np.exp(s["cur_lp"] + s["smap_lr"]))
            err_raw = abs(pred_raw_price - s["actual_price"])
            if len(prior) >= MIN_FIT:
                X = [[p["smap_lr"], p["smap_lr"] * p["trend"]] for p in prior]
                y = [p["actual_lr"] for p in prior]
                a, b0c, b1c = ols(X, y)
                corr_lr = a + (b0c + b1c * s["trend"]) * s["smap_lr"]
                pred_corr_price = float(np.exp(s["cur_lp"] + corr_lr))
                err_corr = abs(pred_corr_price - s["actual_price"])
                errs_raw.append(err_raw); errs_corr.append(err_corr); dz.append(s["pers_err"])

        if not errs_raw:
            continue
        rmae_raw = float(np.mean(errs_raw) / np.mean(dz))
        rmae_corr = float(np.mean(errs_corr) / np.mean(dz))
        results.append({"ticker": target, "n": len(errs_raw), "rMAE_raw": rmae_raw, "rMAE_corr": rmae_corr,
                        "delta_pct": (rmae_corr - rmae_raw) / rmae_raw * 100})
        print(f"  {target:5s} n={len(errs_raw):3d}  raw={rmae_raw:.4f}  corr={rmae_corr:.4f}  "
              f"Δ={(rmae_corr-rmae_raw)/rmae_raw*100:+.1f}%")

    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"Среднее по {len(df)} тикерам: raw={df['rMAE_raw'].mean():.4f}  corr={df['rMAE_corr'].mean():.4f}  "
          f"Δ={(df['rMAE_corr'].mean()-df['rMAE_raw'].mean())/df['rMAE_raw'].mean()*100:+.1f}%")
    wins = (df["rMAE_corr"] < df["rMAE_raw"]).sum()
    print(f"Коррекция лучше raw на {wins}/{len(df)} тикерах")

    out = RESULTS / "trend_correction.csv"
    df.to_csv(out, index=False, float_format="%.5f")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
