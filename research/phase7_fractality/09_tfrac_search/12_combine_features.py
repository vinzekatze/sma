#!/usr/bin/env python3
"""
12_combine_features.py — Комбинирование amp_curr + дистанционных признаков

Объединяем два источника:
  скр.10: amp_curr, amp_z, amp_ratio  (амплитуда T_big события)
  скр.11: fix_dm, fix_ds, slope_dm, dm_range  (профиль расстояний)

Вопросы:
  1. Независимы ли признаки? (матрица cross-корреляций)
  2. Улучшает ли amp_curr прогноз tf_oracle поверх дистанционных признаков?
  3. Какая линейная комбинация даёт максимальную корреляцию с tf_oracle?
"""
import csv
import sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE    = Path(__file__).parent
RESULTS = HERE / "results"

FEAT10 = RESULTS / "features_10m.csv"
FEAT11 = RESULTS / "dist_features_10m.csv"


def load_csv(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            rows[step] = {k: float(v) for k, v in row.items() if k != "step"}
    return rows


def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return np.nan, mask.sum()
    r, _ = stats.spearmanr(x[mask], y[mask])
    return r, mask.sum()


def main():
    d10 = load_csv(FEAT10)
    d11 = load_csv(FEAT11)

    steps = sorted(set(d10) & set(d11))
    print(f"Совместных шагов: {len(steps)}")
    print()

    tf_or     = np.array([d10[s]["tf_oracle"]  for s in steps])
    amp_curr  = np.array([d10[s]["amp_curr"]   for s in steps])
    amp_z     = np.array([d10[s]["amp_z"]      for s in steps])
    amp_ratio = np.array([d10[s]["amp_ratio"]  for s in steps])
    fix_dm    = np.array([d11[s]["fix_dm"]     for s in steps])
    fix_ds    = np.array([d11[s]["fix_ds"]     for s in steps])
    fix_max   = np.array([d11[s]["fix_max"]    for s in steps])
    slope_dm  = np.array([d11[s]["slope_dm"]   for s in steps])
    dm_range  = np.array([d11[s]["dm_range"]   for s in steps])

    features = {
        "amp_curr":  amp_curr,
        "amp_z":     amp_z,
        "amp_ratio": amp_ratio,
        "fix_dm":    fix_dm,
        "fix_ds":    fix_ds,
        "fix_max":   fix_max,
        "slope_dm":  slope_dm,
        "dm_range":  dm_range,
    }

    # ── 1. Индивидуальные корреляции с tf_oracle ─────────────────────────────
    print("=== Индивидуальные Spearman ρ vs tf_oracle ===")
    for name, arr in features.items():
        r, n = spearman(arr, tf_or)
        print(f"  {name:<12}  ρ={r:+.3f}  n={n}")
    print()

    # ── 2. Матрица взаимных корреляций ────────────────────────────────────────
    print("=== Матрица взаимных корреляций (Spearman) ===")
    names = list(features)
    arrs  = [features[n] for n in names]
    print(f"{'':12}", end="")
    for n in names:
        print(f"  {n[:8]:>9}", end="")
    print()
    corr_mat = np.full((len(names), len(names)), np.nan)
    for i, (ni, ai) in enumerate(zip(names, arrs)):
        print(f"  {ni:<12}", end="")
        for j, (nj, aj) in enumerate(zip(names, arrs)):
            r, _ = spearman(ai, aj)
            corr_mat[i, j] = r
            print(f"  {r:+.2f}   ", end="")
        print()
    print()

    # ── 3. Комбинации двух признаков ─────────────────────────────────────────
    # OLS-прогноз tf_oracle по паре признаков
    print("=== Комбинации (OLS прогноз tf_oracle по 2 признакам) → ρ с tf_oracle ===")
    pairs = [
        ("amp_curr",  "fix_dm"),
        ("amp_curr",  "fix_ds"),
        ("amp_curr",  "slope_dm"),
        ("amp_curr",  "dm_range"),
        ("amp_curr",  "fix_max"),
        ("fix_dm",    "slope_dm"),
        ("fix_dm",    "dm_range"),
        ("fix_dm",    "fix_ds"),
        ("fix_dm",    "fix_max"),
    ]
    best_r, best_pair, best_pred = -np.inf, None, None
    for n1, n2 in pairs:
        a1, a2 = features[n1], features[n2]
        mask = np.isfinite(a1) & np.isfinite(a2) & np.isfinite(tf_or)
        if mask.sum() < 30:
            continue
        X = np.column_stack([np.ones(mask.sum()), a1[mask], a2[mask]])
        c, *_ = np.linalg.lstsq(X, tf_or[mask], rcond=None)
        pred = X @ c
        r, _ = stats.spearmanr(pred, tf_or[mask])
        better = " ◄ best" if r > best_r else ""
        print(f"  {n1:12} + {n2:12}  ρ={r:+.3f}  n={mask.sum()}{better}")
        if r > best_r:
            best_r, best_pair = r, (n1, n2)
            best_pred_full = np.full(len(tf_or), np.nan)
            best_pred_full[mask] = pred
            best_pred = best_pred_full
    print()

    # ── 4. Тройная комбинация: amp_curr + fix_dm + slope_dm ─────────────────
    print("=== Тройная комбинация ===")
    triples = [
        ("amp_curr", "fix_dm",   "slope_dm"),
        ("amp_curr", "fix_dm",   "dm_range"),
        ("amp_curr", "fix_dm",   "fix_ds"),
        ("amp_curr", "fix_dm",   "fix_max"),
        ("amp_curr", "slope_dm", "dm_range"),
        ("fix_dm",   "slope_dm", "dm_range"),
    ]
    best_r3, best_triple, best_pred3 = -np.inf, None, None
    for n1, n2, n3 in triples:
        a1, a2, a3 = features[n1], features[n2], features[n3]
        mask = np.isfinite(a1) & np.isfinite(a2) & np.isfinite(a3) & np.isfinite(tf_or)
        if mask.sum() < 30:
            continue
        X = np.column_stack([np.ones(mask.sum()), a1[mask], a2[mask], a3[mask]])
        c, *_ = np.linalg.lstsq(X, tf_or[mask], rcond=None)
        pred = X @ c
        r, _ = stats.spearmanr(pred, tf_or[mask])
        better = " ◄ best" if r > best_r3 else ""
        print(f"  {n1:12} + {n2:12} + {n3:12}  ρ={r:+.3f}{better}")
        if r > best_r3:
            best_r3, best_triple = r, (n1, n2, n3)
            best_pred3_full = np.full(len(tf_or), np.nan)
            best_pred3_full[mask] = pred
            best_pred3 = best_pred3_full
    print()

    # ── 5. График ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # scatter: лучшая пара vs tf_oracle
    ax = axes[0, 0]
    if best_pred is not None:
        mask = np.isfinite(best_pred) & np.isfinite(tf_or)
        ax.scatter(best_pred[mask] * 100, tf_or[mask] * 100,
                   s=5, alpha=0.3, color="steelblue")
        ax.set_xlabel(f"OLS({best_pair[0]}, {best_pair[1]}) прогноз (%)")
        ax.set_ylabel("tf_oracle (%)")
        ax.set_title(f"Лучшая пара  ρ={best_r:.3f}")

    # scatter: лучшая тройка vs tf_oracle
    ax = axes[0, 1]
    if best_pred3 is not None:
        mask = np.isfinite(best_pred3) & np.isfinite(tf_or)
        ax.scatter(best_pred3[mask] * 100, tf_or[mask] * 100,
                   s=5, alpha=0.3, color="darkorange")
        ax.set_xlabel(f"OLS{best_triple} прогноз (%)")
        ax.set_ylabel("tf_oracle (%)")
        ax.set_title(f"Лучшая тройка  ρ={best_r3:.3f}")

    # тепловая карта взаимных корреляций
    ax = axes[0, 2]
    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            v = corr_mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.5 else "black")
    plt.colorbar(im, ax=ax)
    ax.set_title("Матрица взаимных корреляций")

    # scatter amp_curr vs fix_dm
    ax = axes[1, 0]
    mask = np.isfinite(amp_curr) & np.isfinite(fix_dm)
    ax.scatter(amp_curr[mask] * 100, fix_dm[mask],
               s=5, alpha=0.3, color="mediumseagreen")
    r_cross, _ = stats.spearmanr(amp_curr[mask], fix_dm[mask])
    ax.set_xlabel("amp_curr (%)")
    ax.set_ylabel("dist_mean @ 3.6%")
    ax.set_title(f"amp_curr vs fix_dm  ρ={r_cross:.3f}")

    # scatter amp_curr vs tf_oracle
    ax = axes[1, 1]
    mask = np.isfinite(amp_curr) & np.isfinite(tf_or)
    ax.scatter(amp_curr[mask] * 100, tf_or[mask] * 100,
               s=5, alpha=0.3, color="steelblue")
    r_a, _ = stats.spearmanr(amp_curr[mask], tf_or[mask])
    ax.set_xlabel("amp_curr (%)")
    ax.set_ylabel("tf_oracle (%)")
    ax.set_title(f"amp_curr vs tf_oracle  ρ={r_a:.3f}")

    # scatter fix_dm vs tf_oracle
    ax = axes[1, 2]
    mask = np.isfinite(fix_dm) & np.isfinite(tf_or)
    ax.scatter(fix_dm[mask], tf_or[mask] * 100,
               s=5, alpha=0.3, color="darkorange")
    r_d, _ = stats.spearmanr(fix_dm[mask], tf_or[mask])
    ax.set_xlabel("dist_mean @ 3.6%")
    ax.set_ylabel("tf_oracle (%)")
    ax.set_title(f"fix_dm vs tf_oracle  ρ={r_d:.3f}")

    plt.suptitle("Комбинирование amp_curr + дистанционные признаки\n"
                 f"SBER 10m | T_big={sys.argv[1] if len(sys.argv)>1 else '4'}%",
                 fontsize=11)
    plt.tight_layout()
    out = RESULTS / "combine_features_10m.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out}")


if __name__ == "__main__":
    main()
