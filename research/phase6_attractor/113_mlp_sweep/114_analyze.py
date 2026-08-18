"""
114_analyze.py — анализ результатов эксп.113 (m_lp sweep при d=3, p_fit=7).

Ключевой вопрос: для origins, прошедших гейт d≥LB при m_lp=7,
помогает ли увеличение m_lp (окна LP-фильтра)?
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
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# baseline из эксп.109: n_iter=3, d=3, p_fit=7, все 8 тикеров, 40 origins
REF_109 = {"m_lp": 7, "rmae": 0.3930, "mean_abs_error": 0.003426}


def rmae(g: pd.DataFrame) -> float:
    s = g["true"].std()
    return float(g["abs_error"].mean() / s) if s > 1e-14 else float("nan")


def main(test: bool = False) -> None:
    suffix = "_test" if test else ""
    path = RESULTS_DIR / f"results{suffix}.csv"
    if not path.exists():
        print(f"Файл не найден: {path}"); sys.exit(1)

    FIGURES_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(path)
    df = df[df["ok"] == 1].copy()
    df["gate"] = df["gate"].astype(str).map({"1": True, "0": False, "": None})

    has_gate = df["gate"].notna().any()
    print(f"Строк (ok=1): {len(df)}")
    print(f"Тикеры: {sorted(df['ticker'].unique())}")
    print(f"m_lp:   {sorted(df['m_lp'].unique())}")
    print(f"gate_labels: {has_gate}")
    if has_gate:
        g_count = df.groupby("gate")["m_lp"].count()
        print(f"  gate=True: {g_count.get(True,0)} строк, "
              f"gate=False: {g_count.get(False,0)} строк")

    # ── 1. rMAE(m_lp) — пулом ────────────────────────────────────────────────
    rmae_all = (df.groupby("m_lp")
                  .apply(rmae, include_groups=False)
                  .reset_index(name="rmae"))
    mae_all  = df.groupby("m_lp")["abs_error"].mean().reset_index(name="mean_abs_error")
    table_all = rmae_all.merge(mae_all, on="m_lp")

    print("\n=== rMAE(m_lp) — все origins ===")
    print(table_all.to_string(index=False))

    # ── 2. rMAE(m_lp) по группам gate ────────────────────────────────────────
    if has_gate:
        rmae_gate = {}
        mae_gate  = {}
        for g_val in [True, False]:
            sub = df[df["gate"] == g_val]
            if len(sub) == 0:
                continue
            rmae_gate[g_val] = (sub.groupby("m_lp")
                                   .apply(rmae, include_groups=False)
                                   .reset_index(name="rmae"))
            mae_gate[g_val]  = (sub.groupby("m_lp")["abs_error"]
                                    .mean().reset_index(name="mean_abs_error"))

        print("\n=== mean|error|(m_lp) по гейту ===")
        base_t = df[df["m_lp"] == 7].groupby("gate")["abs_error"].mean()
        for g_val, label in [(True, "gate=True (d≥LB)"), (False, "gate=False (d<LB)")]:
            if g_val not in mae_gate:
                continue
            tbl = mae_gate[g_val]
            base = float(base_t.get(g_val, np.nan))
            print(f"\n  {label}:")
            for _, row in tbl.iterrows():
                delta = (row.mean_abs_error - base) / base * 100 if base > 0 else 0
                bar = ("▼" if delta < 0 else "▲") * min(10, int(abs(delta) / 2))
                print(f"    m_lp={int(row.m_lp):2d}  mae={row.mean_abs_error:.6f}"
                      f"  vs m=7: {delta:+.1f}%  {bar}")

    # ── 3. per-origin анализ: гетерогенность оптимального m_lp ──────────────
    # для каждого origin: какой m_lp даёт минимальный abs_error?
    print("\n=== per-origin оптимальный m_lp (гетерогенность) ===")
    opt_per_origin = (df.sort_values("abs_error")
                        .groupby(["ticker", "origin"])
                        .first()
                        .reset_index()[["ticker", "origin", "m_lp", "abs_error", "gate"]])
    opt_per_origin.rename(columns={"m_lp": "best_m_lp"}, inplace=True)

    dist = opt_per_origin["best_m_lp"].value_counts().sort_index()
    print("  Распределение лучшего m_lp по origins:")
    for m_val, cnt in dist.items():
        pct = cnt / len(opt_per_origin) * 100
        bar = "█" * int(pct / 2)
        print(f"    m_lp={m_val:2d}: {cnt:4d} origins ({pct:.1f}%)  {bar}")

    if has_gate:
        print("\n  Лучший m_lp по группам гейта:")
        for g_val, label in [(True, "gate=True"), (False, "gate=False")]:
            sub = opt_per_origin[opt_per_origin["gate"] == g_val]
            if len(sub) == 0:
                continue
            dist_g = sub["best_m_lp"].value_counts().sort_index()
            vals = [f"m={m}:{c}" for m, c in dist_g.items()]
            print(f"    {label}: {', '.join(vals)}")
            print(f"    median(best_m_lp)={sub['best_m_lp'].median():.0f}  "
                  f"mean={sub['best_m_lp'].mean():.1f}")

    # ── 4. delta анализ: m_lp vs baseline m=7, per-origin ───────────────────
    baseline = df[df["m_lp"] == 7][["ticker", "origin", "abs_error", "gate"]].copy()
    baseline.rename(columns={"abs_error": "base_error"}, inplace=True)

    df_delta = df[df["m_lp"] != 7].merge(
        baseline[["ticker", "origin", "base_error"]], on=["ticker", "origin"])
    df_delta["delta"] = df_delta["abs_error"] - df_delta["base_error"]
    df_delta["improvement"] = df_delta["delta"] < 0

    print("\n=== Доля origins, где m_lp > 7 лучше baseline ===")
    if has_gate:
        for g_val, label in [(True, "gate=True"), (False, "gate=False")]:
            sub = df_delta[df_delta["gate"] == g_val]
            if len(sub) == 0:
                continue
            pct = sub["improvement"].mean() * 100
            print(f"  {label}: {pct:.1f}% origins улучшаются при m_lp>7")
            # по каждому m_lp
            for m_val in sorted(df_delta["m_lp"].unique()):
                s2 = sub[sub["m_lp"] == m_val]
                p = s2["improvement"].mean() * 100 if len(s2) > 0 else np.nan
                med_d = s2["delta"].median() * 1e4 if len(s2) > 0 else np.nan
                print(f"    m_lp={m_val:2d}: {p:.1f}% лучше  медиана delta={med_d:.2f}×10⁻⁴")
    else:
        pct = df_delta["improvement"].mean() * 100
        print(f"  Все origins: {pct:.1f}% улучшаются при m_lp>7")

    # ── 5. сводка ─────────────────────────────────────────────────────────────
    best_row = table_all.loc[table_all["rmae"].idxmin()]
    summary = {
        "test_mode": test,
        "n_ok": len(df),
        "d_fixed": 3, "p_fit": 7, "k_lp": 30, "n_iter_lp": 3,
        "m_lp_grid": sorted(df["m_lp"].unique().tolist()),
        "rmae_table": table_all.to_dict(orient="records"),
        "best_m_lp": int(best_row["m_lp"]),
        "best_rmae": float(best_row["rmae"]),
        "ref_109_rmae": REF_109["rmae"],
        "has_gate": has_gate,
    }
    if has_gate:
        gate_opt = {}
        for g_val in [True, False]:
            sub = opt_per_origin[opt_per_origin["gate"] == g_val]
            if len(sub) > 0:
                key = "gate_true" if g_val else "gate_false"
                gate_opt[key] = {
                    "n": len(sub),
                    "median_best_m_lp": float(sub["best_m_lp"].median()),
                    "mean_best_m_lp":   float(sub["best_m_lp"].mean()),
                }
        summary["gate_opt_m_lp"] = gate_opt

    sp = RESULTS_DIR / f"analysis_summary{suffix}.json"
    sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"\nСводка: {sp}")

    if not HAS_MPL:
        print("matplotlib не найден — рисунки пропущены"); return

    # ── рисунок 1: mean|error|(m_lp) по группам гейта ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Влияние m_lp (окна LP) на точность прогноза\n"
                 f"d={3}, p_fit={7}, k={30}, n_iter={3}", fontsize=12)

    # левый: mean|error|(m_lp)
    ax = axes[0]
    ax.plot(table_all["m_lp"], table_all["mean_abs_error"],
            color="black", marker="o", linewidth=2, label="все origins")
    if has_gate:
        for g_val, col, label in [(True, "steelblue", "gate=True (d≥LB)"),
                                   (False, "tomato",   "gate=False (d<LB)")]:
            if g_val in mae_gate:
                t = mae_gate[g_val]
                ax.plot(t["m_lp"], t["mean_abs_error"],
                        color=col, marker="o", linewidth=1.5,
                        linestyle="--", label=label)
    ax.set_xlabel("m_lp (окно LP-фильтра)"); ax.set_ylabel("mean|error|")
    ax.set_title("mean|error|(m_lp)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # правый: % origins, где m_lp лучше baseline (m=7)
    ax = axes[1]
    if has_gate:
        for g_val, col, label in [(True, "steelblue", "gate=True"),
                                   (False, "tomato",   "gate=False")]:
            sub = df_delta[df_delta["gate"] == g_val]
            if len(sub) == 0:
                continue
            pcts = sub.groupby("m_lp")["improvement"].mean() * 100
            ax.plot(pcts.index, pcts.values,
                    color=col, marker="o", linewidth=1.5, label=label)
    else:
        pcts = df_delta.groupby("m_lp")["improvement"].mean() * 100
        ax.plot(pcts.index, pcts.values, color="black", marker="o", linewidth=2)
    ax.axhline(50, color="gray", linestyle=":", linewidth=1, label="50% (безразличие)")
    ax.set_xlabel("m_lp"); ax.set_ylabel("% origins лучше baseline")
    ax.set_title("% origins, где m_lp > baseline (m=7)")
    ax.set_ylim(0, 100); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = FIGURES_DIR / f"mlp_vs_gate{suffix}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Рисунок: {p}")

    # ── рисунок 2: гетерогенность оптимального m_lp ──────────────────────────
    if has_gate:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle("Гетерогенность оптимального m_lp per-origin", fontsize=12)
        for ax, g_val, label in [(ax1, True, "gate=True (d≥LB)"),
                                   (ax2, False, "gate=False (d<LB)")]:
            sub = opt_per_origin[opt_per_origin["gate"] == g_val]["best_m_lp"]
            if len(sub) == 0:
                ax.set_visible(False); continue
            ax.hist(sub, bins=[5.5,7.5,9.5,11.5,15.5,21.5,27.5],
                    color=("steelblue" if g_val else "tomato"),
                    edgecolor="white", alpha=0.8)
            ax.set_xlabel("best m_lp per origin"); ax.set_ylabel("count")
            ax.set_title(label)
            ax.axvline(sub.median(), color="black", linestyle="--",
                       linewidth=1, label=f"median={sub.median():.0f}")
            ax.legend()
        plt.tight_layout()
        p2 = FIGURES_DIR / f"best_mlp_hist{suffix}.png"
        plt.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Рисунок: {p2}")


if __name__ == "__main__":
    main(test="--test" in sys.argv)
