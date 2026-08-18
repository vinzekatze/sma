#!/usr/bin/env python3
"""
14_ampcos_sweep.py — LWR с метрикой amp_cos на зигзаге (событийное время).

Метрика amp_cos (как в app6):
    d(x, q) = α · |log(‖x‖/‖q‖)| / max_amp  +  (1−α) · (1−cos∠(x,q)) / max_cos
    α=0 → чистый косинус (форма вектора, масштаб игнорируется)
    α=1 → чистая амплитуда (отношение длин)

Та же метрика — и для отбора K соседей, и как bandwidth в ядре LWR.

Каузальный контракт: pool_conf[j+1] < query_conf[i].

Два прогона:
  1. Грубый: T_ratio × α (полная сетка)
  2. Тонкий: T_ratio вблизи оптимума × α вблизи нуля
"""

import json
import time
import numpy as np
import csv
from pathlib import Path

# ── Константы ─────────────────────────────────────────────────────────────────
T_BIG     = 0.04
M         = 2
K         = 162
DATA_PATH = Path(__file__).parents[3] / "data/candles/SBER/10m.json"

# Грубый прогон
T_COARSE  = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 1.00]
A_COARSE  = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

# Тонкий прогон (α вблизи нуля, T_ratio вблизи оптимума — заполняется после грубого)
A_FINE    = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
T_FINE_WIDTH = 0.05   # ±0.05 вокруг оптимального T_ratio из грубого
T_FINE_STEP  = 0.025

# Эталон (лучший Евклид из эксп.13)
EUCL_RMAE    = 0.3770
EUCL_PARAMS  = "Euclidean m=2 K=162 T_ratio=0.885"


# ── Загрузка ──────────────────────────────────────────────────────────────────

def load_log_candles(path):
    with open(path) as f:
        raw = json.load(f)
    lh = np.log(np.array([c["high"] for c in raw], dtype=np.float64))
    ll = np.log(np.array([c["low"]  for c in raw], dtype=np.float64))
    dt = np.array([c["begin"] for c in raw])
    return lh, ll, dt


# ── Зигзаг ────────────────────────────────────────────────────────────────────

def build_zigzag(lh, ll, dates, thr):
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
                lp.append(ext); conf.append(dates[i]); dirs.append(1)
                cur_dir = -1; ext = ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= thr:
                lp.append(ext); conf.append(dates[i]); dirs.append(-1)
                cur_dir = 1; ext = lh[i]
    return np.array(lp), np.array(conf), np.array(dirs, dtype=np.int8)


# ── Кэш ───────────────────────────────────────────────────────────────────────

_zz_cache   = {}
_pool_cache = {}


def get_pool(lh, ll, dates, thr, m):
    key = (round(thr, 5), m)
    if key in _pool_cache:
        return _pool_cache[key]
    zkey = round(thr, 5)
    if zkey not in _zz_cache:
        _zz_cache[zkey] = build_zigzag(lh, ll, dates, thr)
    lp, conf, dirs = _zz_cache[zkey]
    n    = len(lp)
    vidx = np.arange(m, n - 1)
    feat  = np.zeros((len(vidx), m))
    tgt   = np.zeros(len(vidx))
    dar   = np.zeros(len(vidx), dtype=np.int8)
    for row, j in enumerate(vidx):
        for lag in range(m):
            feat[row, lag] = lp[j - lag] - lp[j - lag - 1]
        tgt[row]  = lp[j + 1] - lp[j]
        dar[row]  = dirs[j]
    econf  = conf[vidx + 1]
    finite = np.all(np.isfinite(feat), axis=1) & np.isfinite(tgt)
    result = (feat[finite], tgt[finite], dar[finite], econf[finite])
    _pool_cache[key] = result
    return result


# ── Метрика amp_cos ───────────────────────────────────────────────────────────

def ampcos_dists(X: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    """
    amp_cos расстояние от каждой строки X до вектора q.

    d = alpha · |log(‖x‖/‖q‖)| / max_amp  +  (1−alpha) · (1−cos∠(x,q)) / max_cos
    Нормировка — по максимуму среди всех точек X (как в app6).
    """
    nX = np.linalg.norm(X, axis=1)
    nq = float(np.linalg.norm(q))

    # Амплитудная составляющая
    if nq > 1e-10:
        with np.errstate(invalid="ignore", divide="ignore"):
            d_amp = np.where(nX > 1e-10, np.abs(np.log(nX / nq)), np.abs(nX - nq))
    else:
        d_amp = np.abs(nX - nq)

    # Косинусная составляющая
    if nq > 1e-10:
        with np.errstate(invalid="ignore"):
            sim     = np.where(nX > 1e-10, (X @ q) / (nX * nq), 0.0)
        d_cos = 1.0 - np.clip(sim, -1.0, 1.0)
    else:
        d_cos = np.ones(len(X))

    max_a = max(float(d_amp.max()), 1e-10)
    max_c = max(float(d_cos.max()), 1e-10)

    return alpha * (d_amp / max_a) + (1.0 - alpha) * (d_cos / max_c)


# ── Walk-forward: LWR с amp_cos ───────────────────────────────────────────────

def eval_lwr_ampcos(qlp, qconf, qdirs, lh, ll, dates, m, k, T_ratio, alpha):
    pf, pt, pd, peconf = get_pool(lh, ll, dates, T_ratio * T_BIG, m)
    errors, act_diffs  = [], []

    for i in range(m, len(qlp) - 1):
        causal = peconf < qconf[i]
        if causal.sum() < k:
            continue
        pfc, ptc, pdc = pf[causal], pt[causal], pd[causal]

        qvec  = np.array([qlp[i - lag] - qlp[i - lag - 1] for lag in range(m)])
        dmask = pdc == int(qdirs[i])
        if dmask.sum() < k:
            continue

        pfd, ptd = pfc[dmask], ptc[dmask]

        # Расстояния по amp_cos
        dists = ampcos_dists(pfd, qvec, alpha)

        # K ближайших
        nn       = np.argpartition(dists, k - 1)[:k]
        d_nn     = dists[nn]
        f_nn     = pfd[nn]
        t_nn     = ptd[nn]
        d_max    = d_nn.max()

        if d_max < 1e-12:
            lr = float(t_nn.mean())
        else:
            w   = np.exp(-0.5 * (d_nn / d_max) ** 2)
            sw  = np.sqrt(w)
            A   = np.column_stack([np.ones(k), f_nn]) * sw[:, None]
            b   = t_nn * sw
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
            lr  = float(c[0] + c[1:] @ qvec)

        errors.append(abs(np.exp(qlp[i] + lr) - np.exp(qlp[i + 1])))
        act_diffs.append(abs(np.exp(qlp[i + 1]) - np.exp(qlp[i])))

    if len(errors) < 2:
        return 1.0, 0
    return float(np.mean(errors) / np.mean(act_diffs)), len(errors)


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(label, qlp, qconf, qdirs, lh, ll, dates, t_grid, a_grid):
    print(f"\n── {label} ({len(t_grid)}×{len(a_grid)} = {len(t_grid)*len(a_grid)} комбинаций) ──")
    rows = []
    best_rmae = float('inf')
    best_params = None

    for T_ratio in t_grid:
        for alpha in a_grid:
            v, n = eval_lwr_ampcos(qlp, qconf, qdirs, lh, ll, dates, M, K, T_ratio, alpha)
            delta = (v - EUCL_RMAE) / EUCL_RMAE * 100
            rows.append({'T_ratio': round(T_ratio, 4), 'alpha': round(alpha, 4),
                         'rMAE': round(v, 4), 'vs_eucl_pct': round(delta, 2), 'n': n})
            if v < best_rmae:
                best_rmae = v
                best_params = (T_ratio, alpha)
            print(f"  T_ratio={T_ratio:.3f}  α={alpha:.3f}  → rMAE={v:.4f}  ({delta:+.1f}% vs Eucl)  n={n}")

    print(f"\n  Лучший: T_ratio={best_params[0]:.3f}, α={best_params[1]:.3f} → rMAE={best_rmae:.4f}")
    return rows, best_params, best_rmae


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== 14: LWR amp_cos sweep ===")
    print(f"T_BIG={T_BIG*100:.0f}%  m={M}  K={K}")
    print(f"Эталон (Euclidean): rMAE={EUCL_RMAE}  [{EUCL_PARAMS}]")

    lh, ll, dates = load_log_candles(DATA_PATH)
    print(f"Свечей: {len(dates)}  ({dates[0][:10]} … {dates[-1][:10]})")

    qlp, qconf, qdirs = build_zigzag(lh, ll, dates, T_BIG)
    print(f"T_query={T_BIG*100:.0f}%: {len(qlp)} пивотов")

    t0 = time.time()

    # ── Грубый прогон ─────────────────────────────────────────────────────────
    coarse_rows, (best_T, best_A), best_coarse = run_sweep(
        "Грубый прогон", qlp, qconf, qdirs, lh, ll, dates,
        T_COARSE, A_COARSE,
    )

    # ── Тонкий прогон ─────────────────────────────────────────────────────────
    t_lo = max(0.50, best_T - T_FINE_WIDTH)
    t_hi = min(1.00, best_T + T_FINE_WIDTH)
    n_steps = round((t_hi - t_lo) / T_FINE_STEP) + 1
    t_fine  = [round(t_lo + i * T_FINE_STEP, 4) for i in range(n_steps)]
    # убрать дубликаты с грубой сеткой, добавить best_T если не попал
    t_fine_set = sorted(set(t_fine + [round(best_T, 4)]))

    fine_rows, (fine_T, fine_A), best_fine = run_sweep(
        "Тонкий прогон (α≈0)", qlp, qconf, qdirs, lh, ll, dates,
        t_fine_set, A_FINE,
    )

    elapsed = time.time() - t0

    # ── Итог ─────────────────────────────────────────────────────────────────
    print(f"\n══ ИТОГ ══")
    print(f"Эталон Euclidean:     rMAE={EUCL_RMAE:.4f}")
    print(f"Лучший грубый:        rMAE={best_coarse:.4f}  T_ratio={best_T:.3f}  α={best_A:.3f}  "
          f"({(best_coarse-EUCL_RMAE)/EUCL_RMAE*100:+.1f}%)")
    print(f"Лучший тонкий:        rMAE={best_fine:.4f}  T_ratio={fine_T:.3f}  α={fine_A:.3f}  "
          f"({(best_fine-EUCL_RMAE)/EUCL_RMAE*100:+.1f}%)")
    print(f"Время: {elapsed:.1f}s  |  Зигзагов в кэше: {len(_zz_cache)}")

    # ── Сохранение ────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent / "results"

    all_rows = coarse_rows + fine_rows
    csv_path = out_dir / "ampcos_sweep_SBER_10m_T004.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sweep", "T_ratio", "alpha", "rMAE", "vs_eucl_pct", "n"])
        w.writeheader()
        for row in coarse_rows:
            w.writerow({"sweep": "coarse", **row})
        for row in fine_rows:
            w.writerow({"sweep": "fine", **row})

    # Таблица по грубому прогону: T_ratio × α
    print(f"\nГрубая сетка rMAE (строки=T_ratio, столбцы=α):")
    header = f"{'T_ratio':>8}" + "".join(f"  α={a:.1f}" for a in A_COARSE)
    print(header)
    by_T = {}
    for r in coarse_rows:
        by_T.setdefault(r['T_ratio'], {})[r['alpha']] = r['rMAE']
    for T in T_COARSE:
        line = f"{T:>8.3f}"
        for a in A_COARSE:
            v = by_T.get(round(T, 4), {}).get(round(a, 4), float('nan'))
            mark = " *" if (round(T, 4) == round(best_T, 4) and round(a, 4) == round(best_A, 4)) else "  "
            line += f"  {v:.4f}{mark}"
        print(line)

    print(f"\nРезультаты: {csv_path}")


if __name__ == "__main__":
    main()
