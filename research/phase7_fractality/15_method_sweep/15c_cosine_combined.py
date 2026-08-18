#!/usr/bin/env python3
"""
15c_cosine_combined.py — Simplex + LA0, косинус vs евклид
Пул = T_big=4% события ∪ T_frac события (как в HTML: оба T объединены).
Без фильтра по направлению — косинус сам разделяет H/L по знаку Δ₁.

Каузальный контракт: событие пула j доступно когда confirm_date(j+1) < confirm_date(query i).
"""
import json, time
import numpy as np
import pandas as pd
from pathlib import Path

HERE    = Path(__file__).parent
DATA    = HERE.parents[2] / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

T_BIG      = 0.04
M          = 3
MIN_HIST   = 20
TFRAC_GRID = [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.012, 0.015, 0.020]
K_GRID     = [15, 20, 25, 30, 40]


def load_candles():
    with open(DATA / "10m.json") as f:
        raw = json.load(f)
    lh = np.log(np.array([c["high"] for c in raw], dtype=np.float64))
    ll = np.log(np.array([c["low"]  for c in raw], dtype=np.float64))
    dt = np.array([c["begin"] for c in raw])
    return lh, ll, dt


def build_zigzag(lh, ll, dt, thr):
    lp, conf, dirs = [], [], []
    cur_dir, ext = 0, (lh[0] + ll[0]) / 2.0
    for i in range(len(lh)):
        if cur_dir == 0:
            if lh[i] - ext >= thr:   cur_dir =  1; ext = lh[i]
            elif ext - ll[i] >= thr:  cur_dir = -1; ext = ll[i]
        elif cur_dir == 1:
            if lh[i] > ext: ext = lh[i]
            elif ext - ll[i] >= thr:
                lp.append(ext); conf.append(dt[i]); dirs.append(1)
                cur_dir = -1; ext = ll[i]
        else:
            if ll[i] < ext: ext = ll[i]
            elif lh[i] - ext >= thr:
                lp.append(ext); conf.append(dt[i]); dirs.append(-1)
                cur_dir = 1; ext = lh[i]
    return np.array(lp), np.array(conf), np.array(dirs, dtype=np.int8)


def build_pool_entries(lp, conf):
    """Возвращает (feats N×M, tgts N, confs N) — confirm_date пивота j+1."""
    rows = []
    for j in range(M, len(lp) - 1):
        feat = np.array([lp[j-lag] - lp[j-lag-1] for lag in range(M)])
        tgt  = lp[j+1] - lp[j]
        if np.all(np.isfinite(feat)) and np.isfinite(tgt):
            rows.append((feat, tgt, conf[j+1]))
    if not rows:
        return None
    return (
        np.array([r[0] for r in rows]),
        np.array([r[1] for r in rows]),
        np.array([r[2] for r in rows]),
    )


def cos_dists(feats, qvec):
    nq = np.linalg.norm(qvec)
    if nq < 1e-12:
        return np.ones(len(feats))
    nf = np.linalg.norm(feats, axis=1)
    dots = feats @ qvec
    with np.errstate(invalid='ignore', divide='ignore'):
        sim = np.where(nf < 1e-12, 0.0, dots / (nf * nq))
    return np.maximum(0.0, 1.0 - sim)


def euc_dists(feats, qvec):
    return np.linalg.norm(feats - qvec, axis=1)


def predict(nd, nt):
    d_min = nd[0] + 1e-12
    w_s = np.exp(-nd / d_min); w_s /= w_s.sum()
    pred_s = float((w_s * nt).sum())

    d_max = nd.max()
    w_l = np.exp(-0.5 * (nd / d_max) ** 2) if d_max > 1e-12 else np.ones(len(nt))
    w_l /= w_l.sum()
    pred_l = float((w_l * nt).sum())
    return pred_s, pred_l


def walk_forward(qlp, qconf, pool_big, pool_frac, K, dist_fn):
    """
    pool_big:  (feats, tgts, confs) — T_big события
    pool_frac: (feats, tgts, confs) — T_frac события (или None)
    Без фильтра по направлению — как в HTML.
    """
    bf, bt, bc = pool_big
    if pool_frac is not None:
        ff, ft, fc = pool_frac
        all_feats = np.vstack([bf, ff])
        all_tgts  = np.concatenate([bt, ft])
        all_confs = np.concatenate([bc, fc])
    else:
        all_feats, all_tgts, all_confs = bf, bt, bc

    records = []
    for i in range(MIN_HIST, len(qlp) - 1):
        qvec = np.array([qlp[i-lag] - qlp[i-lag-1] for lag in range(M)])
        if not np.all(np.isfinite(qvec)):
            continue

        valid = all_confs < qconf[i]
        if valid.sum() < K:
            continue

        fv = all_feats[valid]
        tv = all_tgts[valid]
        dists  = dist_fn(fv, qvec)
        nn_idx = np.argpartition(dists, K-1)[:K]
        nn_idx = nn_idx[np.argsort(dists[nn_idx])]

        nd, nt = dists[nn_idx], tv[nn_idx]
        actual   = float(np.exp(qlp[i+1]))
        cur_lp   = float(qlp[i])
        pers_err = float(abs(actual - np.exp(qlp[i-1])))
        ps, pl   = predict(nd, nt)

        p_avg = (ps + pl) / 2
        records.append({
            "e_Simplex": abs(np.exp(cur_lp + ps)    - actual),
            "e_LA0":     abs(np.exp(cur_lp + pl)    - actual),
            "e_Avg":     abs(np.exp(cur_lp + p_avg) - actual),
            "e_pers":    pers_err,
        })
    return pd.DataFrame(records) if records else pd.DataFrame()


def rmae(df):
    if df.empty: return {}
    dz = df["e_pers"].mean()
    if dz < 1e-12: return {}
    return {"n": len(df),
            "Simplex": float(df["e_Simplex"].mean() / dz),
            "LA0":     float(df["e_LA0"].mean()     / dz),
            "Avg":     float(df["e_Avg"].mean()     / dz)}


def main():
    print(f"=== COMBINED POOL (T_big ∪ T_frac), SBER 10m  T_BIG={T_BIG*100:.0f}% ===\n")
    lh, ll, dt = load_candles()
    print(f"Свечей: {len(dt)}  ({dt[0][:10]} … {dt[-1][:10]})")

    qlp, qconf, _ = build_zigzag(lh, ll, dt, T_BIG)
    print(f"T_BIG пивотов: {len(qlp)}\n")

    pool_big = build_pool_entries(qlp, qconf)
    print(f"Пул T_big: {len(pool_big[0])} строк\n")

    frac_pools = {}
    for tf in TFRAC_GRID:
        lp_f, conf_f, _ = build_zigzag(lh, ll, dt, tf)
        p = build_pool_entries(lp_f, conf_f)
        if p is not None:
            frac_pools[tf] = p
            print(f"  T_frac={tf*100:.1f}%  пивотов={len(lp_f):5d}  строк={len(p[0]):5d}")
    print()

    rows = []
    t0 = time.time()
    for tf in [None] + TFRAC_GRID:
        pool_frac = frac_pools.get(tf) if tf is not None else None
        label_tf  = f"{tf*100:.1f}%" if tf else "—"
        pool_size = (len(pool_big[0]) + (len(pool_frac[0]) if pool_frac else 0))

        for K in K_GRID:
            for metric, fn in [("cos", cos_dists), ("euc", euc_dists)]:
                df = walk_forward(qlp, qconf, pool_big, pool_frac, K, fn)
                m  = rmae(df)
                if not m: continue
                row = {"T_frac": label_tf, "pool_size": pool_size,
                       "K": K, "metric": metric, **m}
                rows.append(row)
                print(f"T_frac={label_tf:5s}  K={K:2d}  [{metric}]  "
                      f"n={m['n']}  Simplex={m['Simplex']:.4f}  LA0={m['LA0']:.4f}  Avg={m['Avg']:.4f}")

    if not rows:
        print("Нет результатов."); return

    df_out = pd.DataFrame(rows)
    out = RESULTS / "cosine_combined_sber_10m_T4.csv"
    df_out.to_csv(out, index=False, float_format="%.5f")
    print(f"\nСохранено: {out}")

    print("\n=== Топ-5 LA0 по метрикам ===")
    for metric in ["cos", "euc"]:
        sub = df_out[df_out.metric == metric]
        if sub.empty: continue
        top = sub.nsmallest(5, "LA0")[["T_frac", "pool_size", "K", "n", "Simplex", "LA0", "Avg"]]
        print(f"\n[{metric}]")
        print(top.to_string(index=False))

    print(f"\nВремя: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
