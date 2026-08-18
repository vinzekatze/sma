#!/usr/bin/env python3
"""
17d_k_sweep.py — свип числа соседей K на подтверждённой точке (SBER,
T_big=20%, ratio=0.85, arm=D_allpeers). Мотивация: 17c показал, что с
кросс-тикерным пулом (~1674 событий в среднем) K ближайших соседей физически
намного ближе к запросу, чем на own-only пуле (d@4: 0.072 vs 0.30) — значит
можно позволить меньший K, не теряя локальность, и Simplex (K=E+1) снова
осмыслен.

Методы: LA0, LWR — с переменным K∈K_GRID; Simplex — K=E+1 (m+1=4) и, для
сравнения, при тех же K∈K_GRID, что LA0/LWR (не канонично для Sugihara, но
интересно посмотреть тренд); S-map — считается один раз (не зависит от K,
использует весь пул), как референс.

Каузальный контракт и дедуп — те же, что в walk_forward() из
17_large_scale_pooled.py (переиспользуются напрямую, не дублируются).
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("exp17", HERE / "17_large_scale_pooled.py")
exp17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp17)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TARGET = "SBER"
T_BIG = 0.20
RATIO = 0.85
ARM = "D_allpeers"
K_GRID = [exp17.M + 1, 5, 10, 20, 30, 50]  # E+1=4 первым — канонический Simplex


def _la0_k(d, tgts, K):
    if len(d) < K:
        return np.nan
    knn = np.argpartition(d, K - 1)[:K]
    dd = d[knn]; d_max = dd.max()
    if d_max < 1e-12:
        return float(tgts[knn].mean())
    w = np.exp(-0.5 * (dd / d_max) ** 2); w /= w.sum()
    return float((w * tgts[knn]).sum())


def _lwr_k(d, feats, tgts, qvec, K):
    if len(d) < K:
        return np.nan
    knn = np.argpartition(d, K - 1)[:K]
    dd = d[knn]; d_max = dd.max()
    if d_max < 1e-12:
        return float(tgts[knn].mean())
    w = np.exp(-0.5 * (dd / d_max) ** 2); sw = np.sqrt(w)
    A = np.column_stack([np.ones(K), feats[knn]]) * sw[:, None]
    b = tgts[knn] * sw
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(c[0] + c[1:] @ qvec)


def _simplex_k(d, tgts, K):
    """Sugihara & May 1990: K соседей, веса exp(-d_j/d_1), d_1 = ближайший."""
    if len(d) < K:
        return np.nan
    knn = np.argpartition(d, K - 1)[:K]
    ord_ = knn[np.argsort(d[knn])]
    dd = d[ord_]
    d_min = dd[0] + 1e-12
    w = np.exp(-dd / d_min); w /= w.sum()
    return float((w * tgts[ord_]).sum())


def build_pool_for_step(target_data, peer_data, rankings, checkpoints, t_big, t_frac, arm, i, full_lp, full_conf):
    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    confirm_date = full_conf[i]
    cutoff_idx = int(np.searchsorted(dates, confirm_date, side="right"))

    t_lh, t_ll, t_dt = lh[:cutoff_idx], ll[:cutoff_idx], dates[:cutoff_idx]
    own_big_lp, _, own_big_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_big)
    if len(own_big_lp) < exp17.M + 1 or own_big_lp[-1] != full_lp[i]:
        return None

    qvec = np.array([own_big_lp[-1 - lag] - own_big_lp[-2 - lag] for lag in range(exp17.M)])
    if not np.all(np.isfinite(qvec)):
        return None
    q_dir = int(own_big_dir[-1])

    own_big_feats, own_big_tgts, own_big_dirs = exp17.build_pool_rows(own_big_lp, own_big_dir, exp17.M)
    pool_feats = [own_big_feats]; pool_tgts = [own_big_tgts]; pool_dirs = [own_big_dirs]

    if arm != "A_baseline":
        own_frac_lp, _, own_frac_dir = exp17.build_zigzag(t_lh, t_ll, t_dt, t_frac)
        f, tg, dd = exp17.build_pool_rows(own_frac_lp, own_frac_dir, exp17.M)
        pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

        for peer in exp17.peers_for_date(rankings, checkpoints, confirm_date, arm):
            p_dates = peer_data[peer]["dates"]
            p_cutoff = int(np.searchsorted(p_dates, confirm_date, side="right"))
            if p_cutoff < exp17.M + 2:
                continue
            p_lh = peer_data[peer]["lh"][:p_cutoff]
            p_ll = peer_data[peer]["ll"][:p_cutoff]
            p_dt = p_dates[:p_cutoff]
            p_lp, _, p_dir = exp17.build_zigzag(p_lh, p_ll, p_dt, t_frac)
            f, tg, dd = exp17.build_pool_rows(p_lp, p_dir, exp17.M)
            pool_feats.append(f); pool_tgts.append(tg); pool_dirs.append(dd)

    feats = np.concatenate(pool_feats) if pool_feats else np.empty((0, exp17.M))
    tgts  = np.concatenate(pool_tgts)  if pool_tgts  else np.empty(0)
    dirs  = np.concatenate(pool_dirs)  if pool_dirs  else np.empty(0, dtype=np.int8)

    mask = dirs == q_dir
    feats_d, tgts_d = feats[mask], tgts[mask]

    if len(feats_d):
        d = np.linalg.norm(feats_d - qvec, axis=1)
        dup_mask = d < exp17.DUP_EPS
        if dup_mask.any():
            feats_d, tgts_d, d = feats_d[~dup_mask], tgts_d[~dup_mask], d[~dup_mask]
    else:
        d = np.empty(0)

    cur_lp = float(own_big_lp[-1])
    return qvec, feats_d, tgts_d, d, cur_lp


def main():
    print("=== 17d_k_sweep ===")
    print(f"Target={TARGET}  T_big={T_BIG}  ratio={RATIO}  arm={ARM}  K_GRID={K_GRID}")

    target_data = exp17.load_ticker(TARGET)
    exp17.PEERS = [t for t in exp17.UNIVERSE if t != TARGET]
    peer_data = {t: exp17.load_ticker(t) for t in exp17.UNIVERSE}

    years = sorted(set(int(d[:4]) for d in target_data["dates"]))
    checkpoints = np.array([f"{y}-01-01" for y in years])
    rankings = exp17.compute_peer_rankings(target_data, peer_data, checkpoints)

    lh, ll, dates = target_data["lh"], target_data["ll"], target_data["dates"]
    full_lp, full_conf, full_dirs = exp17.build_zigzag(lh, ll, dates, T_BIG)
    n_big = len(full_lp)
    t_frac = RATIO * T_BIG

    records = []
    for i in range(exp17.MIN_HIST, n_big - 1):
        res = build_pool_for_step(target_data, peer_data, rankings, checkpoints, T_BIG, t_frac, ARM, i, full_lp, full_conf)
        if res is None:
            continue
        qvec, feats_d, tgts_d, d, cur_lp = res

        actual_price = float(np.exp(full_lp[i + 1]))
        pers_price = float(np.exp(full_lp[i - 1])) if i - 1 >= 0 else np.nan
        pers_err = abs(actual_price - pers_price)

        row = {"step": i, "n_pool": len(tgts_d), "pers_err": pers_err}

        smap_lr = exp17._smap(qvec, feats_d, tgts_d, exp17.MIN_POOL_SMAP, exp17.THETA_SMAP)
        row["e_Smap"] = abs(float(np.exp(cur_lp + smap_lr)) - actual_price) if np.isfinite(smap_lr) else np.nan

        for K in K_GRID:
            la0_lr = _la0_k(d, tgts_d, K)
            lwr_lr = _lwr_k(d, feats_d, tgts_d, qvec, K)
            sx_lr  = _simplex_k(d, tgts_d, K)
            row[f"e_LA0_K{K}"] = abs(float(np.exp(cur_lp + la0_lr)) - actual_price) if np.isfinite(la0_lr) else np.nan
            row[f"e_LWR_K{K}"] = abs(float(np.exp(cur_lp + lwr_lr)) - actual_price) if np.isfinite(lwr_lr) else np.nan
            row[f"e_Simplex_K{K}"] = abs(float(np.exp(cur_lp + sx_lr)) - actual_price) if np.isfinite(sx_lr) else np.nan

        records.append(row)

    df = pd.DataFrame(records)
    dz = float(df["pers_err"].mean())
    print(f"\nШагов: {len(df)}   persistence denom: {dz:.5f}   pool_avg: {df['n_pool'].mean():.1f}\n")

    summary = {"Smap": float(df["e_Smap"].dropna().mean() / dz)}
    print(f"{'method':<10} " + "  ".join(f"K={K:<3d}" for K in K_GRID))
    for method in ["LA0", "LWR", "Simplex"]:
        vals = []
        for K in K_GRID:
            col = f"e_{method}_K{K}"
            v = df[col].dropna()
            r = float(v.mean() / dz) if len(v) > 5 else np.nan
            vals.append(r)
            summary[f"{method}_K{K}"] = r
        print(f"{method:<10} " + "  ".join(f"{v:.4f}" if not np.isnan(v) else "  nan " for v in vals))
    print(f"{'Smap':<10} {summary['Smap']:.4f} (не зависит от K)")

    out = RESULTS / "k_sweep.csv"
    df.to_csv(out, index=False, float_format="%.5f")
    pd.DataFrame([summary]).to_csv(RESULTS / "k_sweep_summary.csv", index=False, float_format="%.5f")
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
