"""
110_analyze.py — анализ результатов 109_thin_pfit: повторяет диагностику
эксп.101 (rMAE(d), LB(после)~d, корреляции LB↔|error|, гейт d≥LB) на сетке
с «тонким» p_fit=2d+1, плюс прямое сопоставление с зафиксированными числами
эксп.101 (по p_fit, а не по d — см. критику в README §5).

Вход:  results/diag_levels{_test}.csv, results/forecast_accuracy{_test}.csv
       (см. 109_thin_pfit.py — формат и происхождение колонок)

Выход:
  results/analysis_summary.json   — rMAE(n_iter,d), корреляции (пулом и по d),
                                     гейт d≥LB, сравнение с эксп.101 по p_fit
  figures/rmae_vs_d.png
  figures/rmae_vs_pfit.png         — то же, но по оси p_fit (ось сравнения с 101)
  figures/lb_vs_d.png
  figures/lb_after_vs_abs_error.png
  figures/gate_d_ge_lb.png         — mean|error| при гейте d≥LB вкл/выкл, по d

Запуск:
  source /home/kali/.venvs/sma/bin/activate
  python research/phase6_attractor/109_thin_pfit/110_analyze.py [--test]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES_DIR = HERE / "results"
FIG_DIR = HERE / "figures"

TEST = "--test" in sys.argv
SUFFIX = "_test" if TEST else ""

LEVEL_METRICS = ["lb_before_q", "lb_before_n", "lb_after_q", "lb_after_n"]

# Эталонные числа эксп.101 (101_lb_diag_accuracy/README.md §6) для сравнения
# по p_fit (т.к. при разных формулах m одинаковый d => разный p_fit, см. §5
# README этого эксперимента). p_fit_101 = 3*d_101.
REF_101_RMAE_BEST = {"n_iter": 2, "d": 11, "p_fit": 33, "rmae": 0.3274}
REF_101_GATE_IMPROVEMENT = 0.347  # mean|error| d<LB vs d>=LB, финальный уровень, пулом


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    diag = pd.read_csv(RES_DIR / f"diag_levels{SUFFIX}.csv")
    acc = pd.read_csv(RES_DIR / f"forecast_accuracy{SUFFIX}.csv")
    acc = acc[acc["ok"] == 1].copy()
    return diag, acc


def widen_diag(diag: pd.DataFrame) -> pd.DataFrame:
    """Long (1 строка = уровень) → wide (1 строка = origin), колонки <metric>_lvl<k>."""
    key = ["ticker", "n_iter", "d", "origin"]
    pieces = []
    for metric in LEVEL_METRICS:
        piv = diag.pivot_table(index=key, columns="level", values=metric)
        piv.columns = [f"{metric}_lvl{int(c)}" for c in piv.columns]
        pieces.append(piv)
    wide = pd.concat(pieces, axis=1).reset_index()

    n_levels = diag["level"].nunique()
    for metric in LEVEL_METRICS:
        cols = [f"{metric}_lvl{k}" for k in range(n_levels)]
        wide[f"{metric}_spread"] = wide[cols].std(axis=1)
        wide[f"{metric}_mean"] = wide[cols].mean(axis=1)

    for k in range(n_levels):
        wide[f"delta_q_lvl{k}"] = wide[f"lb_after_q_lvl{k}"] - wide[f"lb_before_q_lvl{k}"]
        wide[f"delta_n_lvl{k}"] = wide[f"lb_after_n_lvl{k}"] - wide[f"lb_before_n_lvl{k}"]
    delta_q_cols = [f"delta_q_lvl{k}" for k in range(n_levels)]
    delta_n_cols = [f"delta_n_lvl{k}" for k in range(n_levels)]
    wide["delta_q_mean"] = wide[delta_q_cols].mean(axis=1)
    wide["delta_q_range"] = wide[delta_q_cols].max(axis=1) - wide[delta_q_cols].min(axis=1)
    wide["delta_n_mean"] = wide[delta_n_cols].mean(axis=1)
    wide["delta_n_range"] = wide[delta_n_cols].max(axis=1) - wide[delta_n_cols].min(axis=1)

    # гейт d>=LB на каждом уровне (lb_before_q — финальный найден самым сильным в эксп.101)
    for k in range(n_levels):
        wide[f"gate_lvl{k}"] = wide["d"] >= wide[f"lb_before_q_lvl{k}"]
    return wide


def rmae_table(acc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n_iter, d), g in acc.groupby(["n_iter", "d"]):
        std_true = g["true"].std()
        rmae = g["abs_error"].mean() / std_true if std_true > 0 else np.nan
        p_fit = int(g["p_fit"].iloc[0])
        rows.append({"n_iter": n_iter, "d": d, "p_fit": p_fit, "n_obs": len(g),
                     "rmae": rmae, "mean_abs_error": g["abs_error"].mean()})
    return pd.DataFrame(rows).sort_values(["n_iter", "d"])


def correlations(merged: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    y = merged["abs_error"].values
    for col in feature_cols:
        x = merged[col].values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            rows.append({"feature": col, "n": int(mask.sum()),
                         "pearson_r": np.nan, "pearson_p": np.nan,
                         "spearman_r": np.nan, "spearman_p": np.nan})
            continue
        pr, pp = stats.pearsonr(x[mask], y[mask])
        sr, sp = stats.spearmanr(x[mask], y[mask])
        rows.append({"feature": col, "n": int(mask.sum()),
                     "pearson_r": pr, "pearson_p": pp,
                     "spearman_r": sr, "spearman_p": sp})
    return pd.DataFrame(rows).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)


def gate_analysis(merged: pd.DataFrame, n_levels: int) -> dict:
    """Повтор проверки эксп.101: разбиение по d>=LB(финальный, до очистки),
    mean|error| в каждой страте, по уровням каскада, по тикерам, по d."""
    final = n_levels - 1
    gate_col = f"gate_lvl{final}"
    out: dict = {}

    g_true = merged[merged[gate_col] == True]["abs_error"]
    g_false = merged[merged[gate_col] == False]["abs_error"]
    if len(g_true) > 5 and len(g_false) > 5:
        u, p = stats.mannwhitneyu(g_true, g_false, alternative="two-sided")
        improvement = 1 - g_true.mean() / g_false.mean() if g_false.mean() > 0 else np.nan
        out["pooled"] = {
            "mean_abs_error_gate_true": float(g_true.mean()), "n_gate_true": int(len(g_true)),
            "mean_abs_error_gate_false": float(g_false.mean()), "n_gate_false": int(len(g_false)),
            "improvement_pct": float(improvement * 100), "mannwhitney_p": float(p),
        }
    else:
        out["pooled"] = {"note": "insufficient n for gate split"}

    # по уровню каскада (монотонность найденная в эксп.101: грубый→финальный растёт)
    by_level = {}
    for lvl in range(n_levels):
        col = f"gate_lvl{lvl}"
        gt = merged[merged[col] == True]["abs_error"]
        gf = merged[merged[col] == False]["abs_error"]
        if len(gt) > 5 and len(gf) > 5 and gf.mean() > 0:
            by_level[str(lvl)] = float((1 - gt.mean() / gf.mean()) * 100)
    out["improvement_by_level_pct"] = by_level

    # по тикерам (финальный уровень)
    by_ticker = {}
    for ticker, g in merged.groupby("ticker"):
        gt = g[g[gate_col] == True]["abs_error"]
        gf = g[g[gate_col] == False]["abs_error"]
        if len(gt) > 5 and len(gf) > 5 and gf.mean() > 0:
            by_ticker[ticker] = float((1 - gt.mean() / gf.mean()) * 100)
    out["improvement_by_ticker_pct"] = by_ticker

    # по d (финальный уровень) — устойчивость эффекта внутри фиксированного d
    by_d = {}
    for d, g in merged.groupby("d"):
        gt = g[g[gate_col] == True]["abs_error"]
        gf = g[g[gate_col] == False]["abs_error"]
        if len(gt) > 5 and len(gf) > 5 and gf.mean() > 0:
            by_d[str(int(d))] = float((1 - gt.mean() / gf.mean()) * 100)
    out["improvement_by_d_pct"] = by_d

    return out


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    diag, acc = load()
    if len(acc) == 0:
        print("Нет успешных прогнозов в forecast_accuracy — нечего анализировать.")
        return

    wide = widen_diag(diag)
    merged = wide.merge(acc, on=["ticker", "n_iter", "d", "origin"], how="inner")
    print(f"diag rows={len(diag)}  acc rows={len(acc)}  merged origins={len(merged)}")

    n_levels = diag["level"].nunique()
    final_lvl = n_levels - 1

    rmae_df = rmae_table(acc)
    print("\n=== rMAE(n_iter, d, p_fit) ===")
    print(rmae_df.to_string(index=False))
    best_row = rmae_df.loc[rmae_df["rmae"].idxmin()]
    print(f"\nЛучшая ячейка: n_iter={int(best_row.n_iter)} d={int(best_row.d)} "
          f"p_fit={int(best_row.p_fit)} rMAE={best_row.rmae:.4f}")
    print(f"Для сравнения, эксп.101 лучшая ячейка: n_iter={REF_101_RMAE_BEST['n_iter']} "
          f"d={REF_101_RMAE_BEST['d']} p_fit={REF_101_RMAE_BEST['p_fit']} "
          f"rMAE={REF_101_RMAE_BEST['rmae']}")

    feature_cols = (
        [f"{m}_lvl{k}" for m in LEVEL_METRICS for k in range(n_levels)]
        + [f"{m}_spread" for m in LEVEL_METRICS]
        + [f"{m}_mean" for m in LEVEL_METRICS]
        + ["delta_q_mean", "delta_q_range", "delta_n_mean", "delta_n_range"]
    )
    corr_pooled = correlations(merged, feature_cols)
    print("\n=== Корреляция с |error|, пулом по всем (тикер,n_iter,d,origin) ===")
    print(corr_pooled.to_string(index=False))

    corr_by_d: dict[str, list[dict]] = {}
    for d, g in merged.groupby("d"):
        if len(g) < 20:
            continue
        c = correlations(g, feature_cols)
        corr_by_d[str(d)] = c.to_dict("records")

    gate = gate_analysis(merged, n_levels)
    print("\n=== Гейт d>=LB (финальный уровень), пулом ===")
    print(json.dumps(gate.get("pooled", {}), indent=2, ensure_ascii=False))
    print(f"Для сравнения, эксп.101: improvement_pct={REF_101_GATE_IMPROVEMENT*100:.1f}")

    summary = {
        "test_mode": TEST,
        "n_merged_origins": len(merged),
        "n_levels": n_levels,
        "m_formula": "2*d+1",
        "rmae_table": rmae_df.to_dict("records"),
        "correlations_pooled": corr_pooled.to_dict("records"),
        "correlations_by_d": corr_by_d,
        "gate_d_ge_lb": gate,
        "ref_101_rmae_best": REF_101_RMAE_BEST,
        "ref_101_gate_improvement_pct": REF_101_GATE_IMPROVEMENT * 100,
    }
    out_json = RES_DIR / f"analysis_summary{SUFFIX}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"\nСводка: {out_json}")

    # ── рисунки ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_iter, g in rmae_df.groupby("n_iter"):
        ax.plot(g["d"], g["rmae"], marker="o", label=f"n_iter={n_iter}")
    ax.set_xlabel("d"); ax.set_ylabel("rMAE (h=1, att)")
    ax.set_title("rMAE(d) по итерациям LP-фильтра, p_fit=2d+1")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"rmae_vs_d{SUFFIX}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n_iter, g in rmae_df.groupby("n_iter"):
        ax.plot(g["p_fit"], g["rmae"], marker="o", label=f"n_iter={n_iter}")
    ax.axvline(REF_101_RMAE_BEST["p_fit"], color="k", ls="--", lw=1,
               label=f"эксп.101 best p_fit={REF_101_RMAE_BEST['p_fit']}")
    ax.axhline(REF_101_RMAE_BEST["rmae"], color="r", ls=":", lw=1,
               label=f"эксп.101 best rMAE={REF_101_RMAE_BEST['rmae']}")
    ax.set_xlabel("p_fit"); ax.set_ylabel("rMAE (h=1, att)")
    ax.set_title("rMAE(p_fit) — ось сравнения с эксп.101 (p_fit=3d)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"rmae_vs_pfit{SUFFIX}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    lb_d = merged.groupby(["n_iter", "d"])[f"lb_after_q_lvl{final_lvl}"].mean().reset_index()
    for n_iter, g in lb_d.groupby("n_iter"):
        ax.plot(g["d"], g[f"lb_after_q_lvl{final_lvl}"], marker="o", label=f"n_iter={n_iter}")
    ax.plot(lb_d["d"].unique(), lb_d["d"].unique(), "k--", lw=1, label="LB = d (идеал)")
    ax.set_xlabel("d"); ax.set_ylabel(f"LB после, запрос, финальный уровень (p={final_lvl})")
    ax.set_title("LB(после, финальный уровень) vs d, p_fit=2d+1")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"lb_vs_d{SUFFIX}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(merged[f"lb_after_q_lvl{final_lvl}"], merged["abs_error"],
               s=8, alpha=0.3, c=merged["d"], cmap="viridis")
    ax.set_xlabel(f"LB после, запрос, финальный уровень (p={final_lvl})")
    ax.set_ylabel("|error|")
    ax.set_title("LB(после,финальный) vs |error|  (цвет = d)")
    cb = plt.colorbar(ax.collections[0], ax=ax); cb.set_label("d")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"lb_after_vs_abs_error{SUFFIX}.png", dpi=120)
    plt.close(fig)

    by_d_imp = gate.get("improvement_by_d_pct", {})
    if by_d_imp:
        fig, ax = plt.subplots(figsize=(8, 5))
        ds = sorted(int(k) for k in by_d_imp)
        vals = [by_d_imp[str(d)] for d in ds]
        ax.plot(ds, vals, marker="o")
        ax.axhline(REF_101_GATE_IMPROVEMENT * 100, color="r", ls="--", lw=1,
                   label=f"эксп.101 пулом ({REF_101_GATE_IMPROVEMENT*100:.1f}%)")
        ax.set_xlabel("d"); ax.set_ylabel("улучшение mean|error|, %  (d>=LB vs d<LB)")
        ax.set_title("Гейт d>=LB: устойчивость эффекта по d, p_fit=2d+1")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"gate_d_ge_lb{SUFFIX}.png", dpi=120)
        plt.close(fig)

    print(f"\nРисунки: {FIG_DIR}")


if __name__ == "__main__":
    main()
