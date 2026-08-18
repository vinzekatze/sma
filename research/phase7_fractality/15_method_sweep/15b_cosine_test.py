#!/usr/bin/env python3
"""
15b_cosine_test.py — Simplex + LA0 с косинусной метрикой
SBER 10m, T_BIG=4%, T_frac sweep вокруг 0.7%, K sweep вокруг 25.
Для сравнения — те же методы с евклидовой метрикой.

Каузальный контракт: пул события j доступен только когда confirm_date(j+1) < confirm_date(query).
"""
import json
import time
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
TFRAC_GRID = [0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.012, 0.015]
K_GRID     = [15, 20, 25, 30, 40]


# ── данные ─────────────────────────────────────────────────────────────────────

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


def build_pool(lp, conf, dirs):
    rows = []
    for j in range(M, len(lp) - 1):
        feat = np.array([lp[j-lag] - lp[j-lag-1] for lag in range(M)])
        tgt  = lp[j+1] - lp[j]
        if np.all(np.isfinite(feat)) and np.isfinite(tgt):
            rows.append((feat, tgt, dirs[j], conf[j+1]))
    if not rows:
        return None
    return (
        np.array([r[0] for r in rows]),
        np.array([r[1] for r in rows]),
        np.array([r[2] for r in rows], dtype=np.int8),
        np.array([r[3] for r in rows]),
    )


# ── метрики ────────────────────────────────────────────────────────────────────

def euc_dists(feats, qvec):
    return np.linalg.norm(feats - qvec, axis=1)


def cos_dists(feats, qvec):
    nq = np.linalg.norm(qvec)
    if nq < 1e-12:
        return np.ones(len(feats))
    nf = np.linalg.norm(feats, axis=1)
    dots = feats @ qvec
    with np.errstate(invalid='ignore', divide='ignore'):
        sim = np.where(nf < 1e-12, 0.0, dots / (nf * nq))
    return np.maximum(0.0, 1.0 - sim)


# ── методы ─────────────────────────────────────────────────────────────────────

def predict(nd, nt):
    """Возвращает (pred_simplex, pred_la0)."""
    d_min = nd[0] + 1e-12
    w_s = np.exp(-nd / d_min); w_s /= w_s.sum()
    pred_s = float((w_s * nt).sum())

    d_max = nd.max()
    if d_max < 1e-12:
        pred_l = float(nt.mean())
    else:
        w_l = np.exp(-0.5 * (nd / d_max) ** 2); w_l /= w_l.sum()
        pred_l = float((w_l * nt).sum())

    return pred_s, pred_l


# ── walk-forward ───────────────────────────────────────────────────────────────

def walk_forward(qlp, qconf, qdirs, pool, K, dist_fn):
    pf, pt, pdirs, pc = pool
    records = []
    for i in range(max(M, MIN_HIST), len(qlp) - 1):
        qvec = np.array([qlp[i-lag] - qlp[i-lag-1] for lag in range(M)])
        if not np.all(np.isfinite(qvec)):
            continue
        valid = (pc < qconf[i]) & (pdirs == int(qdirs[i]))
        if valid.sum() < K:
            continue

        pf_v = pf[valid]; pt_v = pt[valid]
        dists  = dist_fn(pf_v, qvec)
        nn_idx = np.argpartition(dists, K-1)[:K]
        nn_idx = nn_idx[np.argsort(dists[nn_idx])]
        nd, nt = dists[nn_idx], pt_v[nn_idx]

        actual     = float(np.exp(qlp[i+1]))
        cur_lp     = float(qlp[i])
        pers_err   = float(abs(actual - np.exp(qlp[i-1])))
        pred_s, pred_l = predict(nd, nt)

        records.append({
            "e_Simplex": abs(np.exp(cur_lp + pred_s) - actual),
            "e_LA0":     abs(np.exp(cur_lp + pred_l) - actual),
            "e_pers":    pers_err,
        })
    return pd.DataFrame(records) if records else pd.DataFrame()


def rmae(df):
    if df.empty:
        return {}
    dz = df["e_pers"].mean()
    if dz < 1e-12:
        return {}
    return {
        "n":       len(df),
        "Simplex": float(df["e_Simplex"].mean() / dz),
        "LA0":     float(df["e_LA0"].mean()     / dz),
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"=== COSINE vs EUC: SBER 10m  T_BIG={T_BIG*100:.0f}%  M={M} ===\n")
    lh, ll, dt = load_candles()
    print(f"Свечей: {len(dt)}  ({dt[0][:10]} … {dt[-1][:10]})")

    qlp, qconf, qdirs = build_zigzag(lh, ll, dt, T_BIG)
    print(f"T_BIG пивотов: {len(qlp)}\n")

    pools = {}
    for tf in TFRAC_GRID:
        lp_f, conf_f, dirs_f = build_zigzag(lh, ll, dt, tf)
        p = build_pool(lp_f, conf_f, dirs_f)
        if p is not None:
            pools[tf] = p
            print(f"  T_frac={tf*100:.1f}%  пивотов={len(lp_f):5d}  строк пула={len(p[0]):5d}")
    print()

    rows = []
    t0 = time.time()
    combos = [(tf, K, metric, fn)
              for tf in TFRAC_GRID
              for K  in K_GRID
              for metric, fn in [("cos", cos_dists), ("euc", euc_dists)]]

    for tf, K, metric, fn in combos:
        if tf not in pools:
            continue
        df = walk_forward(qlp, qconf, qdirs, pools[tf], K, fn)
        m  = rmae(df)
        if not m:
            continue
        row = {"T_frac_pct": round(tf*100, 2), "K": K, "metric": metric, **m}
        rows.append(row)
        print(f"T={tf*100:.1f}%  K={K:2d}  [{metric}]  n={m['n']}  "
              f"Simplex={m['Simplex']:.4f}  LA0={m['LA0']:.4f}")

    if not rows:
        print("Нет результатов."); return

    df_out = pd.DataFrame(rows)
    out = RESULTS / "cosine_test_sber_10m_T4.csv"
    df_out.to_csv(out, index=False, float_format="%.5f")
    print(f"\nСохранено: {out}")

    for metric in ["cos", "euc"]:
        sub = df_out[df_out.metric == metric]
        if sub.empty: continue
        best_la0  = sub.loc[sub.LA0.idxmin()]
        best_simp = sub.loc[sub.Simplex.idxmin()]
        print(f"\n[{metric}]  лучший LA0:     T={best_la0.T_frac_pct:.1f}%  K={int(best_la0.K):2d}  "
              f"rMAE={best_la0.LA0:.4f}")
        print(f"[{metric}]  лучший Simplex: T={best_simp.T_frac_pct:.1f}%  K={int(best_simp.K):2d}  "
              f"rMAE={best_simp.Simplex:.4f}")

    print(f"\nВремя: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
