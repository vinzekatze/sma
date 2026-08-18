#!/usr/bin/env python3
"""
15_method_sweep.py — Сравнение методов LA: Simplex / LA0 / LWR (LA1) / RBF
                     SBER 10m, T_BIG=2%

Каузальный контракт: событие пула добавляется только когда известна его цель,
т.е. confirm_date следующего пивота < confirm_date текущего query.

rMAE = mean(|error|) / mean(persistence_errors)
persistence_error[i] = |exp(lp_big[i+1]) − exp(lp_big[i−1])|
"""
import json
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

HERE    = Path(__file__).parent
DATA    = HERE.parents[2] / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# ── Параметры сетки ───────────────────────────────────────────────────────────
T_BIG      = 0.02
TFRAC_GRID = [0.010, 0.013, 0.015, 0.017, 0.019, 0.022]
M_GRID     = [2, 3, 4, 5]
K_GRID     = [5, 10, 20, 30]
MIN_HIST   = 20   # минимум пивотов T_BIG до начала оценки


# ── Данные ────────────────────────────────────────────────────────────────────

def load_candles():
    with open(DATA / "10m.json") as f:
        raw = json.load(f)
    lh = np.log(np.array([c["high"] for c in raw], dtype=np.float64))
    ll = np.log(np.array([c["low"]  for c in raw], dtype=np.float64))
    dt = np.array([c["begin"] for c in raw])
    return lh, ll, dt


def build_zigzag(lh, ll, dt, thr):
    lp, conf, dirs = [], [], []
    cur_dir = 0
    ext = (lh[0] + ll[0]) / 2.0
    for i in range(len(lh)):
        if cur_dir == 0:
            if lh[i] - ext >= thr:
                cur_dir = 1; ext = lh[i]
            elif ext - ll[i] >= thr:
                cur_dir = -1; ext = ll[i]
        elif cur_dir == 1:
            if lh[i] > ext:
                ext = lh[i]
            elif ext - ll[i] >= thr:
                lp.append(ext); conf.append(dt[i]); dirs.append(1)
                cur_dir = -1; ext = ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= thr:
                lp.append(ext); conf.append(dt[i]); dirs.append(-1)
                cur_dir = 1; ext = lh[i]
    return np.array(lp), np.array(conf), np.array(dirs, dtype=np.int8)


def build_pool(lp, conf, dirs, m):
    """
    Возвращает (feats, tgts, pool_dirs, pool_confs) или None.
    feats:       N×M, лог-диффы M лагов
    tgts:        N, log-diff следующего пивота
    pool_dirs:   N, тип пивота j (1=HIGH, -1=LOW)
    pool_confs:  N, confirm_date пивота j+1 (момент, когда событие j известно)
    """
    rows = []
    for j in range(m, len(lp) - 1):
        feat = np.array([lp[j - lag] - lp[j - lag - 1] for lag in range(m)])
        if not np.all(np.isfinite(feat)):
            continue
        tgt = lp[j + 1] - lp[j]
        if not np.isfinite(tgt):
            continue
        rows.append((feat, tgt, dirs[j], conf[j + 1]))
    if not rows:
        return None
    feats  = np.array([r[0] for r in rows], dtype=np.float64)
    tgts   = np.array([r[1] for r in rows], dtype=np.float64)
    pdirs  = np.array([r[2] for r in rows], dtype=np.int8)
    pconfs = np.array([r[3] for r in rows])
    return feats, tgts, pdirs, pconfs


# ── Методы предсказания ───────────────────────────────────────────────────────

def _simplex(qvec, nn_feats, nn_tgts, nn_dists):
    """Sugihara 1990: K соседей, веса exp(−d/d_min), взвешенное среднее."""
    d_min = nn_dists[0] + 1e-12
    w = np.exp(-nn_dists / d_min)
    w /= w.sum()
    return float((w * nn_tgts).sum())


def _la0(qvec, nn_feats, nn_tgts, nn_dists):
    """LA0: взвешенное среднее с Гауссовым bandwidth=d_max."""
    d_max = nn_dists.max()
    if d_max < 1e-12:
        return float(nn_tgts.mean())
    w = np.exp(-0.5 * (nn_dists / d_max) ** 2)
    w /= w.sum()
    return float((w * nn_tgts).sum())


def _lwr(qvec, nn_feats, nn_tgts, nn_dists):
    """LA1 / LWR: взвешенная линейная регрессия, bandwidth=d_max."""
    d_max = nn_dists.max()
    if d_max < 1e-12:
        return float(nn_tgts.mean())
    w = np.exp(-0.5 * (nn_dists / d_max) ** 2)
    sw = np.sqrt(w)
    A = np.column_stack([np.ones(len(nn_tgts)), nn_feats]) * sw[:, None]
    b = nn_tgts * sw
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(c[0] + c[1:] @ qvec)


def _rbf(qvec, nn_feats, nn_tgts, nn_dists):
    """
    RBF-регрессия (kernel ridge): центры = K соседей, σ = d_max.
    Φ_ij = exp(−||x_i − x_j||² / (2σ²))
    Ridge: (Φ + λI) w = y, λ = 1e-3
    ŷ = k_query · w,  k_q[j] = exp(−d_j² / (2σ²))
    """
    d_max = nn_dists.max()
    if d_max < 1e-12:
        return float(nn_tgts.mean())

    sigma2 = 2.0 * d_max ** 2  # σ = d_max → 2σ² для exp

    diff = nn_feats[:, None, :] - nn_feats[None, :, :]  # K×K×M
    Phi  = np.exp(-np.sum(diff ** 2, axis=2) / sigma2)

    K   = len(nn_tgts)
    lam = 1e-3
    w, *_ = np.linalg.lstsq(Phi + lam * np.eye(K), nn_tgts, rcond=None)

    k_q = np.exp(-nn_dists ** 2 / sigma2)
    val = float(k_q @ w)
    return val if np.isfinite(val) else float("nan")


METHODS = {
    "Simplex": _simplex,
    "LA0":     _la0,
    "LWR":     _lwr,
    "RBF":     _rbf,
}


# ── Walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(qlp, qconf, qdirs, pool, K):
    """
    pool = (feats, tgts, pdirs, pconfs)
    Возвращает DataFrame со столбцами: step, e_Simplex, e_LA0, e_LWR, e_RBF, e_pers.
    """
    pf, pt, pdirs, pc = pool
    m = pf.shape[1]
    records = []

    for i in range(max(m, MIN_HIST), len(qlp) - 1):
        qvec = np.array([qlp[i - lag] - qlp[i - lag - 1] for lag in range(m)])
        if not np.all(np.isfinite(qvec)):
            continue

        valid = (pc < qconf[i]) & (pdirs == int(qdirs[i]))
        if valid.sum() < K:
            continue

        pf_v = pf[valid]
        pt_v = pt[valid]

        dists  = np.linalg.norm(pf_v - qvec, axis=1)
        nn_idx = np.argpartition(dists, K - 1)[:K]
        nn_idx = nn_idx[np.argsort(dists[nn_idx])]  # сортируем: [0] = ближайший

        nd = dists[nn_idx]
        nf = pf_v[nn_idx]
        nt = pt_v[nn_idx]

        actual_price = float(np.exp(qlp[i + 1]))
        cur_lp       = float(qlp[i])
        pers_err     = float(abs(actual_price - np.exp(qlp[i - 1])))

        row = {"step": i, "e_pers": pers_err}
        for name, fn in METHODS.items():
            p = fn(qvec, nf, nt, nd)
            if np.isfinite(p):
                row[f"e_{name}"] = abs(np.exp(cur_lp + p) - actual_price)
            else:
                row[f"e_{name}"] = np.nan

        records.append(row)

    return pd.DataFrame(records) if records else pd.DataFrame()


def rmae_from_df(df):
    if df.empty:
        return {}
    dz = float(df["e_pers"].mean())
    if dz < 1e-12:
        return {}
    result = {"n": len(df)}
    for name in METHODS:
        col = f"e_{name}"
        valid = df[col].dropna()
        result[name] = float(valid.mean() / dz) if len(valid) > 0 else float("nan")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== METHOD SWEEP: SBER 10m  T_BIG=2% ===")
    print(f"T_frac: {[f'{t*100:.1f}%' for t in TFRAC_GRID]}")
    print(f"M:      {M_GRID}")
    print(f"K:      {K_GRID}")
    print()

    lh, ll, dt = load_candles()
    print(f"Свечей: {len(dt)}  ({dt[0][:10]} … {dt[-1][:10]})")

    # Каузальный обрез: всё после последней свечи недоступно (обрезаем данные явно)
    # lh, ll, dt уже полный ряд — пул строится только из событий с confirm < query

    qlp, qconf, qdirs = build_zigzag(lh, ll, dt, T_BIG)
    print(f"T_BIG={T_BIG*100:.0f}%  пивотов: {len(qlp)}")
    print()

    # Строим пулы (кэш)
    pools = {}
    for tf in TFRAC_GRID:
        lp_f, conf_f, dirs_f = build_zigzag(lh, ll, dt, tf)
        n_piv = len(lp_f)
        for m in M_GRID:
            p = build_pool(lp_f, conf_f, dirs_f, m)
            if p is not None:
                pools[(tf, m)] = p
        print(f"T_frac={tf*100:.1f}%  пивотов={n_piv}  "
              f"строк пула (M=2): {len(pools.get((tf,2),(([],)*4)[0]))}"
              if (tf, 2) in pools else
              f"T_frac={tf*100:.1f}%  пивотов={n_piv}")
    print()

    # Прогон
    all_rows = []
    combos   = [(tf, m, K) for tf in TFRAC_GRID for m in M_GRID for K in K_GRID]
    total    = len(combos)
    t0       = time.time()

    for idx, (tf, m, K) in enumerate(combos, 1):
        key = (tf, m)
        if key not in pools:
            continue

        df = walk_forward(qlp, qconf, qdirs, pools[key], K)
        metrics = rmae_from_df(df)
        if not metrics:
            continue

        row = {"T_frac_pct": round(tf * 100, 2), "M": m, "K": K, **metrics}
        all_rows.append(row)

        elapsed = time.time() - t0
        eta     = elapsed / idx * (total - idx)
        simp = f"{metrics.get('Simplex', float('nan')):.4f}"
        la0  = f"{metrics.get('LA0',     float('nan')):.4f}"
        lwr  = f"{metrics.get('LWR',     float('nan')):.4f}"
        rbf  = f"{metrics.get('RBF',     float('nan')):.4f}"
        print(
            f"[{idx:3d}/{total}] T={tf*100:.1f}% M={m} K={K:2d}  n={metrics['n']:3d}  "
            f"Simplex={simp}  LA0={la0}  LWR={lwr}  RBF={rbf}  ETA {eta:.0f}s"
        )
        sys.stdout.flush()

    if not all_rows:
        print("Нет результатов.")
        return

    result_df = pd.DataFrame(all_rows)
    out = RESULTS / "method_sweep_sber_10m_T2.csv"
    result_df.to_csv(out, index=False, float_format="%.5f")
    print(f"\nСохранено: {out}")

    # Итоговая таблица: топ-5 по каждому методу
    for method in METHODS:
        valid = result_df[result_df[method].notna()]
        if valid.empty:
            continue
        top = valid.nsmallest(5, method)[["T_frac_pct", "M", "K", "n", method]]
        print(f"\nТоп-5 по {method}:")
        print(top.to_string(index=False))

    # Лучший по каждому методу
    print("\n=== ЛУЧШЕЕ ПО МЕТОДАМ ===")
    for method in METHODS:
        valid = result_df[result_df[method].notna()]
        if valid.empty:
            continue
        best = valid.loc[valid[method].idxmin()]
        print(f"  {method:<8}: rMAE={best[method]:.4f}  "
              f"T_frac={best['T_frac_pct']:.1f}%  M={int(best['M'])}  K={int(best['K'])}")

    print(f"\nВремя: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
