"""
102_analyze.py — анализ результатов 101_lb_diag_accuracy: связь диагностики
Levina-Bickel (LB) каскада с точностью прогноза (rMAE на h=1).

Вход:  results/diag_levels{_test}.csv, results/forecast_accuracy{_test}.csv
       (см. 101_lb_diag_accuracy.py — формат и происхождение колонок)

Выход:
  results/analysis_summary.json   — rMAE(n_iter,d), корреляции (пулом и по d)
  figures/rmae_vs_d.png           — rMAE как функция d, по n_iter
  figures/lb_vs_d.png             — LB(после, финальный уровень) как функция d
  figures/lb_after_vs_abs_error.png — рассеяние LB(после,финальный) vs |error|

Запуск:
  source /home/kali/.venvs/sma/bin/activate
  python research/phase6_attractor/101_lb_diag_accuracy/102_analyze.py [--test]
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

    # производные признаки: разброс по уровням и дельты (после−до)
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
    return wide


def rmae_table(acc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n_iter, d), g in acc.groupby(["n_iter", "d"]):
        std_true = g["true"].std()
        rmae = g["abs_error"].mean() / std_true if std_true > 0 else np.nan
        rows.append({"n_iter": n_iter, "d": d, "n_obs": len(g),
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
    print("\n=== rMAE(n_iter, d) ===")
    print(rmae_df.to_string(index=False))

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

    summary = {
        "test_mode": TEST,
        "n_merged_origins": len(merged),
        "n_levels": n_levels,
        "rmae_table": rmae_df.to_dict("records"),
        "correlations_pooled": corr_pooled.to_dict("records"),
        "correlations_by_d": corr_by_d,
    }
    out_json = RES_DIR / f"analysis_summary{SUFFIX}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"\nСводка: {out_json}")

    # ── рисунки ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_iter, g in rmae_df.groupby("n_iter"):
        ax.plot(g["d"], g["rmae"], marker="o", label=f"n_iter={n_iter}")
    ax.set_xlabel("d"); ax.set_ylabel("rMAE (h=1, att)")
    ax.set_title("rMAE(d) по итерациям LP-фильтра")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"rmae_vs_d{SUFFIX}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    lb_d = merged.groupby(["n_iter", "d"])[f"lb_after_q_lvl{final_lvl}"].mean().reset_index()
    for n_iter, g in lb_d.groupby("n_iter"):
        ax.plot(g["d"], g[f"lb_after_q_lvl{final_lvl}"], marker="o", label=f"n_iter={n_iter}")
    ax.plot(lb_d["d"].unique(), lb_d["d"].unique(), "k--", lw=1, label="LB = d (идеал)")
    ax.set_xlabel("d"); ax.set_ylabel(f"LB после, запрос, финальный уровень (p={final_lvl})")
    ax.set_title("LB(после, финальный уровень) vs d")
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

    print(f"\nРисунки: {FIG_DIR}")


if __name__ == "__main__":
    main()
