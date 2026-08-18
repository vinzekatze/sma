#!/usr/bin/env python3
"""
02_multiscale_b2.py — Многомасштабное вложение B2 (единая шкала), v2

Embedding (p=5, одинаковый для запроса и пула):

  Пул j (T_frac1 события):
    X[j] = [price_f1[j],
             lr_f1(j → j-1),    lr_f1(j-1 → j-2),
             lr_f2(d2 → d2-1),  lr_f2(d2-1 → d2-2)]
    d2 = последний T_frac2 пивот до T_frac1[j]

  Запрос (T_big событие на шаге step):
    X_q = [price_big[step],                         ← реальный текущий уровень
            log(price_big/p_f1[c1]),                ← T_frac1-шаг, вошедший в T_big
            log(p_f1[c1]/p_f1[c1-1]),               ← предыдущий T_frac1-шаг
            lr_f2(d2_q → d2_q-1),                   ← T_frac2 под-структура
            lr_f2(d2_q-1 → d2_q-2)]
    c1  = последний T_frac1 пивот до T_big[step]
    d2_q = последний T_frac2 пивот до T_big[step]

Cols 1-2 запроса ≈ ±T_frac1% (тот же масштаб что и пул). Col 0 = SBER цена.
Метка y = p_f1[j+1]. Оценка: pred vs p_big[step+1] → rMAE.

T_big = 4% (SBER 10m, фиксирован). Свип: T_frac1 × T_frac2.

Baselines (из 01_frac_only_sweep):
  LWR T_big only          rMAE = 0.4163
  frac-only B1 (3%)       rMAE = 0.4023
"""
import json, csv
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent.parent.parent / "data" / "candles" / "SBER"

T_BIG       = 0.04
H           = 1
MIN_HISTORY = 50
K           = 50

T_FRAC1_GRID = [0.020, 0.025, 0.028, 0.030, 0.032, 0.035]
FRAC2_RATIOS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

B1_BEST = 0.4023   # frac-only 01, T_f1=3%
T4_BASE = 0.4163   # LWR T_big only


# ── данные ───────────────────────────────────────────────────────────────────
def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    vals, dts = [], []
    direction, ext_val, ext_idx = 0, (highs[0] + lows[0]) / 2.0, 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction, ext_val, ext_idx = 1, highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                direction, ext_val, ext_idx = -1, lows[i], i
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                direction, ext_val, ext_idx = -1, lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx])
                direction, ext_val, ext_idx = 1, highs[i], i
    return np.array(vals), np.array(dts)


def rmae(errs, acts):
    acts = np.asarray(acts, dtype=float)
    pers = np.abs(acts[2:] - acts[:-2]) if len(acts) >= 3 else np.abs(np.diff(acts))
    dz   = float(np.mean(pers)) if len(pers) > 0 else 1.0
    return float(np.mean(np.abs(errs)) / dz) if dz > 1e-12 else np.nan


# ── embedding ─────────────────────────────────────────────────────────────────
def build_Xf1(p_f1, dt_f1, lp_f1, n_f1,
              dt_f2=None, lp_f2=None):
    """
    Строит embedding для всех T_frac1 событий.
    use_f2 = dt_f2 is not None.
    Возвращает X (n_f1, 3 или 5), nan где недостаточно истории.
    """
    use_f2 = dt_f2 is not None
    ncols  = 5 if use_f2 else 3
    X = np.full((n_f1, ncols), np.nan)

    j = np.arange(2, n_f1)
    X[j, 0] = p_f1[j]
    X[j, 1] = lp_f1[j]   - lp_f1[j - 1]
    X[j, 2] = lp_f1[j-1] - lp_f1[j - 2]

    if use_f2:
        # для каждого T_frac1[j]: индекс последнего T_frac2 пивота до него
        ce2   = np.searchsorted(dt_f2, dt_f1, side='left') - 1  # (n_f1,)
        valid = (ce2 >= 2) & (np.arange(n_f1) >= 2)
        jj, c2 = np.where(valid)[0], ce2[np.where(valid)[0]]
        X[jj, 3] = lp_f2[c2]     - lp_f2[c2 - 1]
        X[jj, 4] = lp_f2[c2 - 1] - lp_f2[c2 - 2]

    return X


# ── walk-forward ─────────────────────────────────────────────────────────────
def run_wf(p_big, dt_big, n_big,
           p_f1, dt_f1, lp_f1, n_f1, X_f1,
           dt_f2=None, lp_f2=None):
    """
    Walk-forward LWR B2 (исправленный запрос).

    Запрос строится заново на каждом шаге:
      x_q[0]  = price_big[step]           (текущий уровень T_big)
      x_q[1]  = log(price_big/p_f1[c1])  (T_frac1-шаг в T_big)
      x_q[2]  = log(p_f1[c1]/p_f1[c1-1]) (предыдущий T_frac1-шаг)
      x_q[3,4]= lr_frac2 (если use_f2)

    Пул = X_f1[valid j < c1], метка = p_f1[j+1].
    Оценка: pred vs p_big[step+H].
    """
    use_f2 = dt_f2 is not None
    P = X_f1.shape[1]

    valid = (
        ~np.any(np.isnan(X_f1), axis=1)
        & (np.arange(n_f1) + 1 < n_f1)
        & (np.arange(n_f1) >= 2)
    )
    ce1_all = np.searchsorted(dt_f1, dt_big, side='left')
    if use_f2:
        ce2_all = np.searchsorted(dt_f2, dt_big, side='left')

    lp_big = np.log(p_big)

    errs, acts = [], []
    for step in range(MIN_HISTORY, n_big - H):
        c1 = ce1_all[step] - 1       # последний T_frac1 перед T_big[step]
        if c1 < 2 or c1 + 1 >= n_f1:
            continue

        # --- строим запрос ---
        x_q = np.empty(P)
        x_q[0] = p_big[step]
        x_q[1] = lp_big[step] - lp_f1[c1]       # шаг T_frac1 → T_big
        x_q[2] = lp_f1[c1]   - lp_f1[c1 - 1]   # предыдущий T_frac1-шаг

        if use_f2:
            c2_q = ce2_all[step] - 1
            if c2_q < 2:
                continue
            x_q[3] = lp_f2[c2_q]     - lp_f2[c2_q - 1]
            x_q[4] = lp_f2[c2_q - 1] - lp_f2[c2_q - 2]

        # --- каузальный пул j < c1 ---
        idx = np.where(valid[:c1])[0]
        if len(idx) < P + 2:
            continue

        X_pool = X_f1[idx]
        y_abs  = p_f1[idx + 1]

        mu  = X_pool.mean(0)
        sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
        Xn  = (X_pool - mu) / sig
        xn  = (x_q    - mu) / sig

        d     = np.linalg.norm(Xn - xn, axis=1)
        k_eff = min(K, len(d))
        ord_  = np.argsort(d)
        knn   = ord_[:k_eff]
        xi    = d[ord_[k_eff - 1]]

        if xi < 1e-12:
            pred = float(y_abs[knn].mean())
        else:
            w  = np.exp(-0.5 * (d[knn] / xi) ** 2); ws = np.sqrt(w)
            A  = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:, None]
            c, *_ = np.linalg.lstsq(A, y_abs[knn] * ws, rcond=None)
            pred = float(c[0] + c[1:] @ xn)

        if np.isnan(pred):
            continue

        errs.append(pred - float(p_big[step + H]))
        acts.append(float(p_big[step + H]))

    return np.array(errs), np.array(acts)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    h, l, d = load_tf("10m")
    p_big, dt_big = find_pivots(h, l, d, T_BIG)
    n_big = len(p_big)

    print(f"SBER 10m | T_big={T_BIG*100:.0f}% | n_big={n_big} | K={K} | H={H}")
    print(f"Baselines:  LWR T_big={T4_BASE:.4f}   frac-only B1(3%)={B1_BEST:.4f}\n")

    pivot_cache = {}
    def get_pivots(T):
        if T not in pivot_cache:
            pv, dt = find_pivots(h, l, d, T)
            pivot_cache[T] = (pv, dt, np.log(pv), len(pv))
        return pivot_cache[T]

    all_rows = []

    for T_f1 in T_FRAC1_GRID:
        p_f1, dt_f1, lp_f1, n_f1 = get_pivots(T_f1)

        # B2-base: только T_frac1 признаки (p=3, без T_frac2)
        Xb  = build_Xf1(p_f1, dt_f1, lp_f1, n_f1)
        eb, ab = run_wf(p_big, dt_big, n_big,
                        p_f1, dt_f1, lp_f1, n_f1, Xb)
        r_b = rmae(eb, ab)
        tag_b = " ★" if r_b < B1_BEST else ""

        print(f"{'─'*70}")
        print(f"T_f1={T_f1*100:.2f}%  n_f1={n_f1:5d}  "
              f"B2-base(no f2): rMAE={r_b:.4f}  "
              f"Δ B1={( r_b-B1_BEST)/B1_BEST*100:+.1f}%{tag_b}")

        all_rows.append(dict(T_f1=T_f1, T_f2=None, f2_ratio=None,
                             rMAE=r_b, dB1=(r_b-B1_BEST)/B1_BEST*100,
                             dBase=0.0, n=len(eb), tag="base"))

        for ratio in FRAC2_RATIOS:
            T_f2 = round(T_f1 * ratio, 4)
            if T_f2 <= 0.001 or T_f2 >= T_f1:
                continue
            p_f2, dt_f2, lp_f2, n_f2 = get_pivots(T_f2)

            Xf = build_Xf1(p_f1, dt_f1, lp_f1, n_f1,
                            dt_f2=dt_f2, lp_f2=lp_f2)
            ef, af = run_wf(p_big, dt_big, n_big,
                             p_f1, dt_f1, lp_f1, n_f1, Xf,
                             dt_f2=dt_f2, lp_f2=lp_f2)
            r  = rmae(ef, af)
            dB1   = (r - B1_BEST) / B1_BEST * 100
            dBase = (r - r_b)    / r_b      * 100
            tag   = " ★" if r < B1_BEST else ""

            print(f"  T_f2={T_f2*100:.3f}%  ratio={ratio:.2f}  "
                  f"rMAE={r:.4f}  Δ B1={dB1:+.1f}%  Δ base={dBase:+.1f}%  "
                  f"n={len(ef)}{tag}")

            all_rows.append(dict(T_f1=T_f1, T_f2=T_f2, f2_ratio=ratio,
                                 rMAE=r, dB1=dB1, dBase=dBase,
                                 n=len(ef), tag="b2"))

    # сохранить CSV
    out = HERE / "results" / "sweep_b2.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["T_f1","T_f2","f2_ratio",
                                           "rMAE","dB1","dBase","n","tag"])
        w.writeheader(); w.writerows(all_rows)

    valid = [r for r in all_rows if not np.isnan(r["rMAE"])]
    best  = min(valid, key=lambda x: x["rMAE"])

    print(f"\n{'═'*70}")
    print(f"  M0                             rMAE = 1.0000")
    print(f"  LWR T_big only                 rMAE = {T4_BASE:.4f}")
    print(f"  frac-only B1 (T_f1=3%)         rMAE = {B1_BEST:.4f}")
    if best["T_f2"] is not None:
        print(f"  best B2  T_f1={best['T_f1']*100:.2f}%"
              f"  T_f2={best['T_f2']*100:.3f}%"
              f"  rMAE = {best['rMAE']:.4f}  Δ B1={best['dB1']:+.1f}%")
    else:
        print(f"  best B2-base  T_f1={best['T_f1']*100:.2f}%"
              f"  rMAE = {best['rMAE']:.4f}  Δ B1={best['dB1']:+.1f}%")
    print(f"  Результаты: {out}")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
