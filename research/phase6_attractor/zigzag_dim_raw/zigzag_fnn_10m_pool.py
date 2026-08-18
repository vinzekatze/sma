#!/usr/bin/env python3
"""
zigzag_fnn_10m_pool.py

FNN-диагностика согласованного 10m-пула:
  big    = 10m T=4%  (крупные события, 776 пив)
  small  = 10m T=0.4% (мелкие события, 41223 пив)
  combined = big + small, отсортировано по дате

HIGH и LOW анализируются раздельно.

Параметры Kennel 1992: R_thr=15, A_thr=2.0
Вложение: X[i] = [price[i], log(p[i]/p[i-1]), ..., log(p[i-p+2]/p[i-p+1])]
          (то же что в LWR-прогнозе)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT      = BASE_DIR / "results"
OUT.mkdir(exist_ok=True)

T_BIG   = 0.04
T_SMALL = 0.004
P_MAX   = 10
R_THR   = 15.0
A_THR   = 2.0


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots(highs, lows, dates, thr):
    """Возвращает (vals, dts, ptypes): ptypes 1=HIGH, -1=LOW."""
    vals, dts, ptypes = [], [], []
    direction = 0
    ext_val = (highs[0] + lows[0]) / 2.0
    ext_idx = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1; ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val, ext_idx = lows[i], i
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx]); ptypes.append(1)
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(dates[ext_idx]); ptypes.append(-1)
                direction = 1; ext_val, ext_idx = highs[i], i
    return np.array(vals), np.array(dts), np.array(ptypes, dtype=int)


def build_X_full(prices, p_max):
    """Полная матрица вложения до p_max+1 признаков."""
    n  = len(prices)
    lp = np.log(prices)
    X  = np.full((n, p_max + 1), np.nan)
    X[:, 0] = prices
    for lag in range(1, p_max + 1):
        for i in range(lag, n):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def fnn_curve(prices, p_max=P_MAX, r_thr=R_THR, a_thr=A_THR):
    """
    Вычисляет долю False Nearest Neighbors для p=1..p_max.
    Kennel 1992: d_p1 — полная (p+1)-мерная норма; std_z — std ценового ряда.
    Возвращает dict {p: fnn_fraction}.
    """
    X     = build_X_full(prices, p_max)
    std_z = float(np.std(prices))
    if std_z < 1e-12:
        return {p: np.nan for p in range(1, p_max + 1)}

    results = {}
    for p in range(1, p_max + 1):
        valid = np.where(~np.any(np.isnan(X[:, :p + 1]), axis=1))[0]
        if len(valid) < 5:
            results[p] = np.nan
            continue
        Xv   = X[valid, :p]       # p-мерное вложение
        Xv1  = X[valid, :p + 1]   # (p+1)-мерное вложение

        nn = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(Xv)
        dists, idxs = nn.kneighbors(Xv)
        nn_idx = idxs[:, 1]
        d_p    = dists[:, 1]

        d_p1 = np.linalg.norm(Xv1 - Xv1[nn_idx], axis=1)  # полная (p+1)-мерная норма

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(d_p > 1e-12, d_p1 / d_p, np.inf)
        false_nn = (ratio > r_thr) | (d_p1 / std_z > a_thr)
        results[p] = float(false_nn.mean())
    return results


def print_table(label, fnn_h, fnn_l, n_h, n_l):
    print(f"\n{'─'*52}")
    print(f"  {label}   (HIGH n={n_h}, LOW n={n_l})")
    print(f"  {'p':>3}  {'FNN HIGH':>10}  {'FNN LOW':>10}")
    print(f"{'─'*52}")
    for p in range(1, P_MAX + 1):
        fh = fnn_h.get(p, np.nan)
        fl = fnn_l.get(p, np.nan)
        mark_h = " ←" if (not np.isnan(fh) and fh == 0.0) else ""
        mark_l = " ←" if (not np.isnan(fl) and fl == 0.0) else ""
        fh_s = f"{fh:.3f}{mark_h}" if not np.isnan(fh) else "  —"
        fl_s = f"{fl:.3f}{mark_l}" if not np.isnan(fl) else "  —"
        print(f"  {p:>3}  {fh_s:>12}  {fl_s:>12}")


def p_min(fnn_dict):
    for p in range(1, P_MAX + 1):
        if fnn_dict.get(p, 1.0) == 0.0:
            return p
    return f">{P_MAX}"


def main():
    h1d,  l1d,  d1d  = load_tf("1d")
    h10m, l10m, d10m = load_tf("10m")

    p_1d,    dt_1d,    t_1d    = find_pivots(h1d,  l1d,  d1d,  T_BIG)
    p_big,   dt_big,   t_big   = find_pivots(h10m, l10m, d10m, T_BIG)
    p_small, dt_small, t_small = find_pivots(h10m, l10m, d10m, T_SMALL)

    # Объединённый пул 10m: big + small, сортировка по дате
    vals_all  = np.concatenate([p_big,   p_small])
    dates_all = np.concatenate([dt_big,  dt_small])
    types_all = np.concatenate([t_big,   t_small])
    order     = np.argsort(dates_all, kind="stable")
    vals_all  = vals_all[order]
    types_all = types_all[order]

    # Раздельные подмножества
    subsets = {
        "1d T=4% H":  p_1d[t_1d     ==  1],
        "1d T=4% L":  p_1d[t_1d     == -1],
        "big    H":   p_big[t_big   ==  1],
        "big    L":   p_big[t_big   == -1],
        "small  H":   p_small[t_small ==  1],
        "small  L":   p_small[t_small == -1],
        "comb   H":   vals_all[types_all ==  1],
        "comb   L":   vals_all[types_all == -1],
    }

    print("SBER — FNN-диагностика событийного аттрактора")
    print(f"  1d T=4%:      {len(p_1d)} пив  "
          f"(H={( t_1d==1).sum()}, L={(t_1d==-1).sum()})")
    print(f"  10m T=4%:     {len(p_big)} пив  "
          f"(H={( t_big==1).sum()}, L={(t_big==-1).sum()})")
    print(f"  10m T=0.4%:   {len(p_small)} пив  "
          f"(H={(t_small==1).sum()}, L={(t_small==-1).sum()})")
    print(f"  10m combined: {len(vals_all)} пив  "
          f"(H={(types_all==1).sum()}, L={(types_all==-1).sum()})")

    # Вычисляем FNN
    print("\nВычисление FNN...")
    fnn = {}
    for name, prices in subsets.items():
        print(f"  {name} (n={len(prices)}) ...", end=" ", flush=True)
        fnn[name] = fnn_curve(prices)
        pm = p_min(fnn[name])
        print(f"p_min={pm}")

    # Таблицы
    print_table("1d T=4%",
                fnn["1d T=4% H"], fnn["1d T=4% L"],
                len(subsets["1d T=4% H"]), len(subsets["1d T=4% L"]))
    print_table("10m T=4% (big)",
                fnn["big    H"], fnn["big    L"],
                len(subsets["big    H"]), len(subsets["big    L"]))
    print_table("10m T=0.4% (small)",
                fnn["small  H"], fnn["small  L"],
                len(subsets["small  H"]), len(subsets["small  L"]))
    print_table("10m combined (T=4% + T=0.4%)",
                fnn["comb   H"], fnn["comb   L"],
                len(subsets["comb   H"]), len(subsets["comb   L"]))

    # Сводка p_min
    print(f"\n{'═'*45}")
    print("  Сводка p_min:")
    for name, fnn_d in fnn.items():
        print(f"  {name}: p_min={p_min(fnn_d)}")

    # График — 4 строки, 2 столбца (HIGH / LOW)
    ps = list(range(1, P_MAX + 1))
    rows = [
        ("1d  T=4%",               "1d T=4% H", "1d T=4% L"),
        ("10m  T=4%  (big)",        "big    H",  "big    L"),
        ("10m  T=0.4%  (small)",    "small  H",  "small  L"),
        ("10m  combined",           "comb   H",  "comb   L"),
    ]
    COLORS = {"H": "#c0392b", "L": "#27ae60"}

    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)

    for row_i, (row_title, key_h, key_l) in enumerate(rows):
        for col_i, (key, side, color) in enumerate([
            (key_h, "HIGH", COLORS["H"]),
            (key_l, "LOW",  COLORS["L"]),
        ]):
            ax = axes[row_i, col_i]
            vals = [fnn[key].get(p, np.nan) for p in ps]
            pmin_val = p_min(fnn[key])
            n_pts = len(subsets[key])

            ax.plot(ps, vals, "o-", color=color, lw=2, ms=7,
                    label=f"p_min={pmin_val}  (n={n_pts})")
            ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
            ax.fill_between(ps, vals, alpha=0.12, color=color)

            ymax = max(v for v in vals if not np.isnan(v)) + 0.03
            ax.set_ylim(-0.02, max(0.3, ymax))
            ax.set_xticks(ps)
            ax.set_ylabel("доля FNN")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc="upper right")
            ax.set_title(f"{row_title} — {side}", fontsize=10, fontweight="bold")

    for ax in axes[3]:
        ax.set_xlabel("p (размерность вложения)", fontsize=10)

    fig.suptitle("SBER: FNN (R_thr=15, A_thr=2)\n"
                 "1d T=4% / 10m T=4% / 10m T=0.4% / 10m combined",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fnn_10m_pool.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {OUT}/fnn_10m_pool.png")


if __name__ == "__main__":
    main()
