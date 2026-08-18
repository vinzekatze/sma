"""
112_analyze.py — анализ результатов эксп.111 (alpha scan).

Читает results/forecast_accuracy.csv, строит rMAE(alpha) по (d, n_iter),
сравнивает с anchor-точками из эксп.101 и эксп.109.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── опорные точки из предыдущих экспериментов ────────────────────────────────
# Эксп.101 (m=3d): лучшая ячейка
REF_101 = {"n_iter": 2, "d": 11, "m": 33, "alpha": 3.00, "rmae": 0.3274}
# Эксп.109 (m=2d+1): лучшая ячейка
REF_109 = {"n_iter": 3, "d": 3,  "m": 7,  "alpha": 2.33, "rmae": 0.3930}

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"


def rmae(group: pd.DataFrame) -> float:
    """rMAE = mean(abs_error) / std(true)."""
    std = group["true"].std()
    if std < 1e-14 or len(group) == 0:
        return float("nan")
    return float(group["abs_error"].mean() / std)


def main(test: bool = False) -> None:
    suffix = "_test" if test else ""
    acc_path = RESULTS_DIR / f"forecast_accuracy{suffix}.csv"
    if not acc_path.exists():
        print(f"Файл не найден: {acc_path}")
        sys.exit(1)

    FIGURES_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(acc_path)
    df = df[df["ok"] == 1].copy()
    df["alpha"] = df["alpha"].round(4)

    print(f"Строк (ok=1): {len(df)}")
    print(f"Тикеры: {sorted(df['ticker'].unique())}")
    print(f"alpha: {sorted(df['alpha'].unique())}")
    print(f"d: {sorted(df['d'].unique())}")
    print(f"n_iter: {sorted(df['n_iter'].unique())}")

    # Дубликаты по m: (alpha1,d) и (alpha2,d) дают одинаковый m
    dup = df.groupby(["d", "n_iter", "m"])["alpha"].nunique()
    dups = dup[dup > 1]
    if len(dups) > 0:
        print("\nПары alpha с одинаковым m (дубликаты по факту):")
        for (d, ni, m), na in dups.items():
            alphas = sorted(df[(df["d"]==d) & (df["n_iter"]==ni) & (df["m"]==m)]["alpha"].unique())
            print(f"  d={d} n_iter={ni} m={m}: alpha={alphas}")

    # ── rMAE(alpha, d, n_iter) ────────────────────────────────────────────────
    table = (df.groupby(["n_iter", "d", "alpha"])
               .apply(rmae, include_groups=False)
               .reset_index(name="rmae"))
    table["m"] = table.apply(lambda r: max(3, round(r["alpha"] * r["d"])), axis=1)
    table = table.sort_values(["n_iter", "d", "alpha"])

    print("\n=== rMAE(n_iter, d, alpha, m) ===")
    print(table.to_string(index=False))

    best_row = table.loc[table["rmae"].idxmin()]
    print(f"\nЛучшая ячейка: n_iter={int(best_row['n_iter'])} "
          f"d={int(best_row['d'])} alpha={best_row['alpha']:.2f} "
          f"m={int(best_row['m'])} rMAE={best_row['rmae']:.4f}")
    print(f"Для сравнения:")
    print(f"  Эксп.101 (m=3d):   n_iter={REF_101['n_iter']} d={REF_101['d']} "
          f"m={REF_101['m']} rMAE={REF_101['rmae']:.4f}")
    print(f"  Эксп.109 (m=2d+1): n_iter={REF_109['n_iter']} d={REF_109['d']} "
          f"m={REF_109['m']} rMAE={REF_109['rmae']:.4f}")

    # ── mean_abs_error (без нормировки) ──────────────────────────────────────
    mae_table = (df.groupby(["n_iter", "d", "alpha"])["abs_error"]
                   .mean()
                   .reset_index(name="mean_abs_error"))
    mae_table["m"] = mae_table.apply(
        lambda r: max(3, round(r["alpha"] * r["d"])), axis=1)

    # ── сохранение summary ───────────────────────────────────────────────────
    summary = {
        "test_mode": test,
        "n_ok": len(df),
        "tickers": sorted(df["ticker"].unique().tolist()),
        "rmae_table": table.to_dict(orient="records"),
        "mae_table": mae_table.to_dict(orient="records"),
        "best_cell": {
            "n_iter": int(best_row["n_iter"]),
            "d": int(best_row["d"]),
            "alpha": float(best_row["alpha"]),
            "m": int(best_row["m"]),
            "rmae": float(best_row["rmae"]),
        },
        "ref_101_rmae": REF_101["rmae"],
        "ref_109_rmae": REF_109["rmae"],
    }
    summary_path = RESULTS_DIR / f"analysis_summary{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                       default=float))
    print(f"\nСводка: {summary_path}")

    if not HAS_MPL:
        print("matplotlib не найден — рисунки пропущены")
        return

    # ── рисунок 1: rMAE(alpha) по (d, n_iter) ────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    fig.suptitle("rMAE(alpha) при m=max(3,round(alpha·d))", fontsize=13)

    colors = {2: "steelblue", 3: "darkorange"}
    markers = {3: "o", 7: "s", 11: "^", 15: "D"}

    for col_idx, d_val in enumerate([3, 7, 11, 15]):
        for row_idx, ni in enumerate([2, 3]):
            ax = axes[row_idx, col_idx]
            sub = table[(table["d"] == d_val) & (table["n_iter"] == ni)]
            if len(sub) == 0:
                ax.set_visible(False); continue
            ax.plot(sub["alpha"], sub["rmae"],
                    color=colors[ni], marker=markers[d_val],
                    linewidth=1.5, markersize=6,
                    label=f"n_iter={ni}, d={d_val}")
            # опорные точки
            if d_val == REF_101["d"] and ni == REF_101["n_iter"]:
                ax.axhline(REF_101["rmae"], color="green", linestyle="--",
                           linewidth=1, alpha=0.7, label=f"exp.101 rMAE={REF_101['rmae']:.3f}")
            if d_val == REF_109["d"] and ni == REF_109["n_iter"]:
                ax.axhline(REF_109["rmae"], color="red", linestyle="--",
                           linewidth=1, alpha=0.7, label=f"exp.109 rMAE={REF_109['rmae']:.3f}")
            ax.set_xlabel("alpha", fontsize=9)
            ax.set_ylabel("rMAE", fontsize=9)
            ax.set_title(f"d={d_val}, n_iter={ni}", fontsize=10)
            ax.set_xticks([2.0, 2.25, 2.5, 2.75, 3.0])
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = FIGURES_DIR / f"rmae_vs_alpha{suffix}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Рисунок: {p}")

    # ── рисунок 2: rMAE(alpha), совмещённые кривые по d, раздельно по n_iter ─
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("rMAE(alpha) — совмещение по d", fontsize=13)
    colors_d = {3: "tab:blue", 7: "tab:orange", 11: "tab:green", 15: "tab:red"}

    for ax, ni in [(ax1, 2), (ax2, 3)]:
        for d_val in [3, 7, 11, 15]:
            sub = table[(table["d"] == d_val) & (table["n_iter"] == ni)]
            if len(sub) == 0:
                continue
            ax.plot(sub["alpha"], sub["rmae"],
                    color=colors_d[d_val], marker="o", linewidth=1.5,
                    markersize=5, label=f"d={d_val}")
        # горизонтальные ref-линии
        ax.axhline(REF_101["rmae"], color="green", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"exp.101 rMAE={REF_101['rmae']:.3f}")
        ax.axhline(REF_109["rmae"], color="red", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"exp.109 rMAE={REF_109['rmae']:.3f}")
        ax.set_xlabel("alpha", fontsize=10)
        ax.set_ylabel("rMAE", fontsize=10)
        ax.set_title(f"n_iter={ni}", fontsize=11)
        ax.set_xticks([2.0, 2.25, 2.5, 2.75, 3.0])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = FIGURES_DIR / f"rmae_vs_alpha_combined{suffix}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Рисунок: {p}")

    # ── рисунок 3: rMAE(m) — ось p_fit для сравнения с exp.101/109 ──────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("rMAE(p_fit=m) — ось сравнения с exp.101/exp.109", fontsize=13)

    for ax, ni in [(ax1, 2), (ax2, 3)]:
        for d_val in [3, 7, 11, 15]:
            sub = table[(table["d"] == d_val) & (table["n_iter"] == ni)].copy()
            if len(sub) == 0:
                continue
            sub = sub.sort_values("m")
            ax.plot(sub["m"], sub["rmae"],
                    color=colors_d[d_val], marker="o", linewidth=1.5,
                    markersize=5, label=f"d={d_val} (×m-axis)")
        ax.axhline(REF_101["rmae"], color="green", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"exp.101 best (m=33)")
        ax.axhline(REF_109["rmae"], color="red", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"exp.109 best (m=7)")
        ax.set_xlabel("m (=p_fit)", fontsize=10)
        ax.set_ylabel("rMAE", fontsize=10)
        ax.set_title(f"n_iter={ni}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = FIGURES_DIR / f"rmae_vs_pfit{suffix}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Рисунок: {p}")

    # ── рисунок 4: heatmap rMAE(alpha, d) для лучшего n_iter ────────────────
    best_ni = int(table.loc[table["rmae"].idxmin(), "n_iter"])
    heat = (table[table["n_iter"] == best_ni]
            .pivot(index="d", columns="alpha", values="rmae"))

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(heat.values, aspect="auto",
                   cmap="RdYlGn_r", origin="lower")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([f"{a:.2f}" for a in heat.columns])
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index.tolist())
    ax.set_xlabel("alpha"); ax.set_ylabel("d")
    ax.set_title(f"rMAE heatmap (n_iter={best_ni})", fontsize=11)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    p = FIGURES_DIR / f"rmae_heatmap{suffix}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Рисунок: {p}")


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    main(test=test_mode)
