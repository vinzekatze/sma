"""
103_level_trajectory.py — углублённый анализ траектории LB по уровням каскада
(0=грубый p_lvl=24d → 3=финальный p_lvl=3d) в эксперименте 101.

Мотивация: 102_analyze.py агрегировал диагностику по уровням только через
mean/spread (std). Визуальный осмотр показал, что для малых d LB заметно
падает от уровня 0 к уровню 3 («стабилизация»), а для d из зоны лучшей
точности (9-13) это падение исчезает или меняет знак. Этот скрипт проверяет,
несёт ли сама направленная динамика (lvl0 - lvl3) дополнительную информацию
о точности прогноза сверх d и финального значения LB.

Вход:  results/diag_levels.csv, results/forecast_accuracy.csv
Выход: results/level_trajectory_summary.json
       figures/trajectory_by_d_bin.png
       figures/drop_vs_error_by_d_range.png

Запуск:
  source /home/kali/.venvs/sma/bin/activate
  python research/phase6_attractor/101_lb_diag_accuracy/103_level_trajectory.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
RES_DIR = HERE / "results"
FIG_DIR = HERE / "figures"

KEY = ["ticker", "n_iter", "d", "origin"]
D_RANGES = {"d<=4": (1, 4), "d=5..8": (5, 8), "d=9..13 (best)": (9, 13), "d>=16": (16, 20)}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    diag = pd.read_csv(RES_DIR / "diag_levels.csv")
    acc = pd.read_csv(RES_DIR / "forecast_accuracy.csv")
    acc = acc[acc["ok"] == 1].copy()
    return diag, acc


def widen_with_drop(diag: pd.DataFrame, metric: str) -> pd.DataFrame:
    piv = diag.pivot_table(index=KEY, columns="level", values=metric).reset_index()
    n_levels = diag["level"].nunique()
    piv.columns = KEY + [f"lvl{c}" for c in range(n_levels)]
    piv[f"{metric}_drop_0to3"] = piv["lvl0"] - piv[f"lvl{n_levels-1}"]
    piv[f"{metric}_lvl0"] = piv["lvl0"]
    piv[f"{metric}_lvlfinal"] = piv[f"lvl{n_levels-1}"]
    return piv[KEY + [f"{metric}_drop_0to3", f"{metric}_lvl0", f"{metric}_lvlfinal"]]


def bin_d(d: int) -> str | None:
    for label, (lo, hi) in D_RANGES.items():
        if lo <= d <= hi:
            return label
    return None


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    diag, acc = load()

    metrics = ["lb_before_q", "lb_after_q", "lb_before_n", "lb_after_n"]
    merged = acc.copy()
    for metric in metrics:
        w = widen_with_drop(diag, metric)
        merged = merged.merge(w, on=KEY, how="inner")
    merged["d_bin"] = merged["d"].apply(bin_d)

    summary: dict = {"d_ranges": D_RANGES, "by_metric": {}}

    # ── 1. корреляция drop с |error|, по диапазонам d, для каждой метрики ──
    for metric in metrics:
        drop_col = f"{metric}_drop_0to3"
        rows = []
        for label, (lo, hi) in D_RANGES.items():
            sub = merged[(merged["d"] >= lo) & (merged["d"] <= hi)]
            r, p = stats.pearsonr(sub[drop_col], sub["abs_error"])
            sr, sp = stats.spearmanr(sub[drop_col], sub["abs_error"])
            rows.append({"d_range": label, "n": len(sub), "mean_drop": float(sub[drop_col].mean()),
                         "pearson_r": r, "pearson_p": p, "spearman_r": sr, "spearman_p": sp})
        summary["by_metric"][metric] = {"drop_vs_error_by_d_range": rows}

    # ── 2. устойчивость по тикерам для d>=16 (главная находка) ─────────────
    sub16 = merged[merged["d"] >= 16]
    per_ticker = []
    for ticker, g in sub16.groupby("ticker"):
        if len(g) < 20:
            continue
        r, p = stats.pearsonr(g["lb_before_q_drop_0to3"], g["abs_error"])
        per_ticker.append({"ticker": ticker, "n": len(g), "pearson_r": r, "pearson_p": p})
    summary["d_ge_16_per_ticker_lb_before_q"] = per_ticker

    # ── 3. drop добавляет информацию сверх lvlfinal и d? частная корреляция ─
    # OLS: abs_error ~ lvlfinal + drop + d  (внутри d>=16, где эффект drop силён)
    import numpy.linalg as la
    for label, (lo, hi) in [("d>=16", (16, 20)), ("d=9..13 (best)", (9, 13))]:
        sub = merged[(merged["d"] >= lo) & (merged["d"] <= hi)].copy()
        X = np.column_stack([np.ones(len(sub)), sub["lb_before_q_lvlfinal"],
                              sub["lb_before_q_drop_0to3"], sub["d"].astype(float)])
        y = sub["abs_error"].values
        beta, *_ = la.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        # сравнение с моделью без drop
        X0 = np.column_stack([np.ones(len(sub)), sub["lb_before_q_lvlfinal"], sub["d"].astype(float)])
        beta0, *_ = la.lstsq(X0, y, rcond=None)
        resid0 = y - X0 @ beta0
        r2_0 = 1 - np.sum(resid0 ** 2) / ss_tot
        summary.setdefault("ols_drop_added_value", {})[label] = {
            "n": len(sub), "beta_const_lvlfinal_drop_d": beta.tolist(),
            "r2_with_drop": float(r2), "r2_without_drop": float(r2_0),
            "delta_r2": float(r2 - r2_0),
        }

    out_json = RES_DIR / "level_trajectory_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"Сводка: {out_json}")
    print(json.dumps(summary["by_metric"]["lb_before_q"], indent=2, ensure_ascii=False, default=float))
    print("\nd>=16, по тикерам (lb_before_q drop vs |error|):")
    for r in per_ticker:
        print(f"  {r['ticker']:6s} n={r['n']:4d} r={r['pearson_r']:+.3f} p={r['pearson_p']:.2e}")
    print("\nДобавленная ценность drop (delta R^2 после lvlfinal+d):")
    for label, v in summary["ols_drop_added_value"].items():
        print(f"  {label}: r2_with={v['r2_with_drop']:.4f} r2_without={v['r2_without_drop']:.4f} delta={v['delta_r2']:.4f}")

    # ── рисунки ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(D_RANGES), figsize=(4 * len(D_RANGES), 4), sharey=True)
    for ax, (label, (lo, hi)) in zip(axes, D_RANGES.items()):
        sub_diag = diag[(diag["d"] >= lo) & (diag["d"] <= hi)]
        traj = sub_diag.groupby("level")["lb_before_q"].mean()
        ax.plot(traj.index, traj.values, marker="o")
        ax.set_title(label)
        ax.set_xlabel("уровень каскада (0=грубый,3=финальный)")
    axes[0].set_ylabel("LB до очистки, запрос (среднее)")
    plt.suptitle("Траектория LB(до, запрос) по уровням каскада, по диапазонам d")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "trajectory_by_d_bin.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, label in zip(axes, ["d<=4", "d=9..13 (best)", "d>=16"]):
        lo, hi = D_RANGES[label]
        sub = merged[(merged["d"] >= lo) & (merged["d"] <= hi)]
        sc = ax.scatter(sub["lb_before_q_drop_0to3"], sub["abs_error"], s=6, alpha=0.3, c=sub["d"], cmap="viridis")
        ax.set_title(label)
        ax.set_xlabel("LB(до,запрос) drop: lvl0 - lvlfinal")
    axes[0].set_ylabel("|error|")
    plt.colorbar(sc, ax=axes[-1], label="d")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "drop_vs_error_by_d_range.png", dpi=120)
    plt.close(fig)
    print(f"\nРисунки: {FIG_DIR}")


if __name__ == "__main__":
    main()
