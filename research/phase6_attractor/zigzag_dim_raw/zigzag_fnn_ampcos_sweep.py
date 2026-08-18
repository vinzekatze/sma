#!/usr/bin/env python3
"""
FNN с метрикой amp_cos (app6): свип по T зигзага и alpha (blend).
Только 1d SBER. H и L — раздельно.

МЕТРИКА: d(x,q) = alpha·|log(‖x‖/‖q‖)|_norm + (1−alpha)·(1−cos(x,q))_norm
NN ищется по custom metric. FNN-критерий: Kennel Euclidean ratio + abs.

Признаковый вектор: X[i] = [price[i], log(p[i]/p[i-1]), ..., log(p[i-p+2]/p[i-p+1])]
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT      = BASE_DIR / "results"

T_GRID  = np.round(np.arange(0.010, 0.085, 0.005), 4)   # 1.0%…8.0%, шаг 0.5%
A_GRID  = np.round(np.linspace(0.0, 1.0, 11), 2)         # alpha: 0.0…1.0
P_MAX   = 15
R_THR   = 15.0
A_THR   = 2.0


def load_1d():
    with open(DATA / "1d.json") as f:
        data = json.load(f)
    return (
        np.array([d["high"]  for d in data], dtype=np.float64),
        np.array([d["low"]   for d in data], dtype=np.float64),
    )


def find_pivots_hl(highs, lows, thr):
    """Возвращает (prices_H, prices_L)."""
    pivots    = []
    direction = 0
    ext_val   = (highs[0] + lows[0]) / 2.0
    ext_idx   = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1;  ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val, ext_idx = lows[i], i
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                pivots.append((ext_val, 'H'))
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                pivots.append((ext_val, 'L'))
                direction = 1;  ext_val, ext_idx = highs[i], i
    prices_H = np.array([p for p, d in pivots if d == 'H'])
    prices_L = np.array([p for p, d in pivots if d == 'L'])
    return prices_H, prices_L


def build_embedding(z, p):
    n  = len(z)
    lz = np.log(z)
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = z[i]
        for lag in range(1, p):
            X[i, lag] = lz[i - lag + 1] - lz[i - lag]
    return X


def amp_cos_components(X):
    """
    Returns (d_amp_n, d_shape_n): (m×m) matrices, нормированные в [0,1].
    d_amp   = |log(‖x_i‖ / ‖x_j‖)|
    d_shape = 1 − cos(x_i, x_j)
    """
    nX = np.linalg.norm(X, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln = np.where(nX > 1e-10, np.log(nX), 0.0)
    d_amp = np.abs(ln[:, None] - ln[None, :])

    XXT   = X @ X.T
    denom = nX[:, None] * nX[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_s = np.where(denom > 1e-10, XXT / denom, 0.0)
    d_shape = 1.0 - cos_s.clip(-1.0, 1.0)

    max_a = max(float(d_amp.max()), 1e-10)
    max_s = max(float(d_shape.max()), 1e-10)
    return d_amp / max_a, d_shape / max_s


def pmin_sweep(z, alpha_arr):
    """
    Для каждого alpha: первый p, при котором FNN=0.
    Возвращает float-массив (nan = не достигнуто в P_MAX).
    """
    n     = len(z)
    std_z = float(np.std(z))
    if std_z < 1e-12 or n < P_MAX + 2:
        return np.full(len(alpha_arr), np.nan)

    p_min = np.full(len(alpha_arr), np.nan)
    found = np.zeros(len(alpha_arr), dtype=bool)

    for p in range(1, P_MAX + 1):
        if found.all():
            break

        X_p  = build_embedding(z, p)
        X_p1 = build_embedding(z, p + 1)
        valid = np.where(~np.any(np.isnan(X_p1), axis=1))[0]
        if len(valid) < 5:
            continue

        Xv  = X_p[valid]
        Xv1 = X_p1[valid]

        d_amp_n, d_shape_n = amp_cos_components(Xv)

        for ai, alpha in enumerate(alpha_arr):
            if found[ai]:
                continue

            D = float(alpha) * d_amp_n + (1.0 - float(alpha)) * d_shape_n
            np.fill_diagonal(D, np.inf)
            nn = np.argmin(D, axis=1)

            d_p  = np.linalg.norm(Xv  - Xv [nn], axis=1)
            d_p1 = np.linalg.norm(Xv1 - Xv1[nn], axis=1)

            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(d_p > 1e-12, d_p1 / d_p, np.inf)
            false_nn = (ratio > R_THR) | (d_p1 / std_z > A_THR)

            if float(false_nn.mean()) == 0.0:
                p_min[ai] = p
                found[ai] = True

    return p_min


def run():
    highs, lows = load_1d()
    print(f"SBER 1d: {len(highs)} баров")
    print(f"T_GRID: {len(T_GRID)} значений ({T_GRID[0]*100:.1f}%…{T_GRID[-1]*100:.1f}%, шаг 0.5%)")
    print(f"A_GRID: {len(A_GRID)} значений (α 0.0…1.0)")
    print(f"P_MAX={P_MAX},  R_thr={R_THR},  A_thr={A_THR}")
    print()

    records = []
    pmin_H = np.full((len(T_GRID), len(A_GRID)), np.nan)
    pmin_L = np.full((len(T_GRID), len(A_GRID)), np.nan)

    for ti, T in enumerate(T_GRID):
        pH, pL = find_pivots_hl(highs, lows, T)
        print(f"T={T*100:.1f}%  n_H={len(pH):4d}  n_L={len(pL):4d}", end="  ", flush=True)

        pm_H = pmin_sweep(pH, A_GRID)
        pm_L = pmin_sweep(pL, A_GRID)
        pmin_H[ti] = pm_H
        pmin_L[ti] = pm_L

        h_str = "  ".join(f"{v:.0f}" if not np.isnan(v) else "?" for v in pm_H)
        l_str = "  ".join(f"{v:.0f}" if not np.isnan(v) else "?" for v in pm_L)
        print(f"\n  H: [{h_str}]")
        print(f"  L: [{l_str}]")

        for ai, alpha in enumerate(A_GRID):
            records.append({
                "T_pct"  : round(T * 100, 1),
                "alpha"  : round(float(alpha), 2),
                "n_H"    : len(pH),
                "n_L"    : len(pL),
                "pmin_H" : pm_H[ai],
                "pmin_L" : pm_L[ai],
            })

    df = pd.DataFrame(records)
    df.to_csv(OUT / "fnn_ampcox_sweep.csv", index=False)

    # ── Сводка ───────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    for direction, pm in [("H (хаи)", pmin_H), ("L (лои)", pmin_L)]:
        v = pm[~np.isnan(pm)]
        if len(v) == 0:
            print(f"{direction}: нет данных"); continue
        print(f"{direction}:  p_min ∈ [{int(v.min())}, {int(v.max())}]  "
              f"среднее={v.mean():.1f}  медиана={float(np.median(v)):.1f}")
        r, c = np.unravel_index(np.nanargmin(pm), pm.shape)
        print(f"  мин p_min={int(pm[r,c])}  @ T={T_GRID[r]*100:.1f}% α={A_GRID[c]:.1f}")
        r, c = np.unravel_index(np.nanargmax(pm), pm.shape)
        print(f"  макс p_min={int(pm[r,c])}  @ T={T_GRID[r]*100:.1f}% α={A_GRID[c]:.1f}")

    # ── Тепловые карты ────────────────────────────────────────────────────────
    all_vals = np.concatenate([pmin_H[~np.isnan(pmin_H)], pmin_L[~np.isnan(pmin_L)]])
    vmin, vmax = int(all_vals.min()), int(all_vals.max())

    t_labels = [f"{t*100:.1f}" for t in T_GRID]
    a_labels = [f"{a:.1f}"     for a in A_GRID]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"FNN p_min: метрика amp_cos  SBER 1d\n"
        f"H/L раздельно  |  Kennel R_thr={R_THR} A_thr={A_THR}  |  P_max={P_MAX}",
        fontsize=11
    )

    for ax, pm, title in [(axes[0], pmin_H, "Хаи (H)"),
                           (axes[1], pmin_L, "Лои (L)")]:
        pm_plot = np.where(np.isnan(pm), P_MAX + 1, pm)
        im = ax.imshow(pm_plot, aspect="auto", origin="upper",
                       vmin=vmin, vmax=vmax + 1, cmap="YlOrRd")
        ax.set_xticks(range(len(A_GRID)))
        ax.set_xticklabels(a_labels, fontsize=8)
        ax.set_yticks(range(len(T_GRID)))
        ax.set_yticklabels(t_labels, fontsize=8)
        ax.set_xlabel("α  (0 = косинус, 1 = амплитуда)")
        ax.set_ylabel("T зигзага, %")
        ax.set_title(title)
        mid = (vmin + vmax) / 2
        for ti in range(len(T_GRID)):
            for ai in range(len(A_GRID)):
                v = pm[ti, ai]
                txt = f"{int(v)}" if not np.isnan(v) else "?"
                col = "white" if (not np.isnan(v) and v > mid) else "black"
                ax.text(ai, ti, txt, ha="center", va="center", fontsize=6.5, color=col)
        plt.colorbar(im, ax=ax, label="p_min")

    plt.tight_layout()
    fig.savefig(OUT / "fnn_ampcox_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCSV + график: {OUT}")


if __name__ == "__main__":
    run()
