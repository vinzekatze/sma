#!/usr/bin/env python3
"""
17h_persistence_blend.py — можно ли использовать силу persistence?
Два варианта подмешивания persistence в калиброванный S-map (m=3, θ=25.697,
T_ratio=0.8987, T_big=20%, D_allpeers, SBER — точка из 17f):

  A) Фиксированный shrinkage: pred_lr = α·smap_lr + (1-α)·persistence_lr,
     α∈[0..1] — свип.
  B) Confidence-gate: α_i = 1/(1 + λ·d_min_i/median(d_min)) — чем дальше
     ближайший сосед на конкретном шаге, тем сильнее давим к persistence.
     λ=0 эквивалентно чистой модели (α=1 везде).

persistence_lr = lp[i-1] - cur_lp (тот же persistence, что в знаменателе rMAE).

Тот же каузальный контракт и дедуп, что и в 17f/17g (build_all_pools).
"""
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
spec17 = importlib.util.spec_from_file_location("exp17", HERE / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec17)
spec17.loader.exec_module(exp17)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TARGET = "SBER"
T_BIG = 0.20
ARM = "D_allpeers"
M, THETA, T_RATIO = 3, 25.697, 0.8987   # калиброванный S-map из 17f

ALPHA_GRID  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LAMBDA_GRID = [0, 1, 2, 5, 10, 20, 50, 100]


def build_steps(target_data, peer_data, rankings, checkpoints, t_big, m, t_frac, arm, full_lp, full_conf):
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
        smap_lr = exp17._smap(qvec, feats_d, tgts_d, m + 2, THETA)
        if not np.isfinite(smap_lr):
            continue
        persistence_lr = float(full_lp[i - 1] - cur_lp)
        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1]))
        pers_err = abs(pers_price - actual_price)
        d_min = float(d.min()) if len(d) else np.nan

        steps.append({"i": i, "cur_lp": cur_lp, "smap_lr": smap_lr, "persistence_lr": persistence_lr,
                      "actual_price": actual_price, "pers_err": pers_err, "d_min": d_min})
    return steps


def rmae_blend(steps, alpha_fn):
    errs, dz = [], []
    for s in steps:
        alpha = alpha_fn(s)
        lr = alpha * s["smap_lr"] + (1 - alpha) * s["persistence_lr"]
        pred = float(np.exp(s["cur_lp"] + lr))
        errs.append(abs(pred - s["actual_price"]))
        dz.append(s["pers_err"])
    return float(np.mean(errs) / np.mean(dz))


def main():
    print("=== 17h_persistence_blend ===")
    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, full_dirs = exp17.build_zigzag(lh, ll, dates, T_BIG)
    t_frac = T_RATIO * T_BIG

    steps = build_steps(target_data, peer_data, rankings, checkpoints, T_BIG, M, t_frac, ARM, full_lp, full_conf)
    print(f"Шагов: {len(steps)}")

    median_dmin = float(np.median([s["d_min"] for s in steps]))
    print(f"median(d_min) = {median_dmin:.4f}\n")

    print("── A) Фиксированный shrinkage α ──")
    rows_a = []
    for alpha in ALPHA_GRID:
        r = rmae_blend(steps, lambda s, a=alpha: a)
        rows_a.append({"alpha": alpha, "rMAE": r})
        print(f"  α={alpha:.1f}  rMAE={r:.4f}")

    print("\n── B) Confidence-gate по d_min (λ) ──")
    rows_b = []
    for lam in LAMBDA_GRID:
        def alpha_fn(s, lam=lam):
            return 1.0 / (1.0 + lam * s["d_min"] / median_dmin)
        r = rmae_blend(steps, alpha_fn)
        alphas = [alpha_fn(s) for s in steps]
        rows_b.append({"lambda": lam, "rMAE": r, "alpha_mean": float(np.mean(alphas)), "alpha_min": float(np.min(alphas))})
        print(f"  λ={lam:<4d}  rMAE={r:.4f}  mean(α)={np.mean(alphas):.3f}  min(α)={np.min(alphas):.3f}")

    pure_model = rmae_blend(steps, lambda s: 1.0)
    pure_pers  = rmae_blend(steps, lambda s: 0.0)
    print(f"\nЧистая модель (α=1 всегда): rMAE={pure_model:.4f}")
    print(f"Чистый persistence (α=0):   rMAE={pure_pers:.4f}  (=1.0000 по определению denom)")

    pd.DataFrame(rows_a).to_csv(RESULTS / "persistence_blend_alpha.csv", index=False, float_format="%.5f")
    pd.DataFrame(rows_b).to_csv(RESULTS / "persistence_blend_gate.csv", index=False, float_format="%.5f")
    print(f"\nСохранено: {RESULTS / 'persistence_blend_alpha.csv'}, {RESULTS / 'persistence_blend_gate.csv'}")


if __name__ == "__main__":
    main()
