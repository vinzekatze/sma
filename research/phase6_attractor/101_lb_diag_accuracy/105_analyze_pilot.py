"""
105_analyze_pilot.py — анализ результатов пилота 104_lb_greedy_select.py:
сравнение baseline (top-xi_lwr по расстоянию, метод эксп.101) и нового
метода (жадное сужение широкого пула по изоляции, цель — низкий lb_before_n)
на парных origin-наблюдениях (тот же att/origin/d/n_iter для обоих методов).

Вход:  results/pilot_greedy_select.csv (или _test.csv)
Выход: results/pilot_analysis_summary.json
       figures/pilot_rmae_base_vs_new.png   — rMAE(d) для baseline и new, по n_iter
       figures/pilot_paired_diff_hist.png   — гистограмма (|err_new|-|err_base|) по origin

Запуск:
  source /home/kali/.venvs/sma/bin/activate
  python research/phase6_attractor/101_lb_diag_accuracy/105_analyze_pilot.py [--test]
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


def rmae_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n_iter, d), g in df.groupby(["n_iter", "d"]):
        std_true = g["true"].std()
        if std_true <= 0:
            continue
        rows.append({
            "n_iter": n_iter, "d": d, "n_obs": len(g),
            "rmae_base": g["abs_error_base"].mean() / std_true,
            "rmae_new": g["abs_error_new"].mean() / std_true,
            "lb_n_base_mean": g["lb_n_base"].mean(),
            "lb_n_new_mean": g["lb_n_new"].mean(),
        })
    return pd.DataFrame(rows).sort_values(["n_iter", "d"])


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    path = RES_DIR / f"pilot_greedy_select{SUFFIX}.csv"
    df = pd.read_csv(path)
    df = df[df["ok"] == 1].copy()
    print(f"n_rows ok=1: {len(df)}  (всего в файле: {len(pd.read_csv(path))})")

    # ── lb_n действительно ниже у нового метода? (проверка, что отбор работает) ──
    lb_diff = df["lb_n_new"] - df["lb_n_base"]
    print(f"\nlb_n_new - lb_n_base: mean={lb_diff.mean():.4f}  "
          f"median={lb_diff.median():.4f}  "
          f"% строк с lb_n_new < lb_n_base: {100*(lb_diff < 0).mean():.1f}%")

    # ── rMAE: baseline vs new, по (n_iter, d) ──
    rmae_df = rmae_table(df)
    print("\n=== rMAE base vs new, по (n_iter, d) ===")
    print(rmae_df.to_string(index=False))

    rmae_df["improvement_pct"] = 100 * (rmae_df["rmae_base"] - rmae_df["rmae_new"]) / rmae_df["rmae_base"]
    print(f"\nСреднее улучшение (rmae_base-rmae_new)/rmae_base по ячейкам: "
          f"{rmae_df['improvement_pct'].mean():.2f}%  "
          f"(положительное = new лучше)")

    # ── агрегированный paired-тест по всем origin (Wilcoxon signed-rank) ──
    err_base = df["abs_error_base"].values
    err_new = df["abs_error_new"].values
    diff = err_new - err_base  # отрицательное = new лучше (меньше |error|)
    wstat, wp = stats.wilcoxon(err_base, err_new)
    print(f"\nWilcoxon signed-rank (|err_base| vs |err_new|), n={len(diff)}: "
          f"stat={wstat:.1f}  p={wp:.4f}")
    print(f"mean(|err_new| - |err_base|) = {diff.mean():.6f}  "
          f"(отрицательное = new лучше в среднем)")
    print(f"% origins где new лучше (|err_new| < |err_base|): {100*(diff < 0).mean():.1f}%")

    # ── по тикерам — устойчивость результата (см. предупреждение из эксп.101 о d>=16) ──
    per_ticker = []
    for ticker, g in df.groupby("ticker"):
        d_ = g["abs_error_new"].values - g["abs_error_base"].values
        w, p = stats.wilcoxon(g["abs_error_base"], g["abs_error_new"])
        per_ticker.append({"ticker": ticker, "n": len(g),
                            "mean_diff": float(d_.mean()),
                            "pct_new_better": float(100 * (d_ < 0).mean()),
                            "wilcoxon_p": float(p)})
    print("\n=== По тикерам (mean_diff<0 и pct_new_better>50 => new лучше) ===")
    for r in per_ticker:
        print(f"  {r['ticker']:6s} n={r['n']:4d}  mean_diff={r['mean_diff']:+.6f}  "
              f"new_better={r['pct_new_better']:.1f}%  p={r['wilcoxon_p']:.4f}")

    summary = {
        "test_mode": TEST, "n_obs": len(df),
        "lb_n_diff_mean": float(lb_diff.mean()),
        "lb_n_new_lower_pct": float(100 * (lb_diff < 0).mean()),
        "rmae_table": rmae_df.to_dict("records"),
        "wilcoxon_pooled": {"stat": float(wstat), "p": float(wp),
                             "mean_diff": float(diff.mean()),
                             "pct_new_better": float(100 * (diff < 0).mean())},
        "per_ticker": per_ticker,
    }
    out_json = RES_DIR / f"pilot_analysis_summary{SUFFIX}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(f"\nСводка: {out_json}")

    # ── рисунки ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_iter, g in rmae_df.groupby("n_iter"):
        ax.plot(g["d"], g["rmae_base"], marker="o", linestyle="--", label=f"baseline n_iter={n_iter}")
        ax.plot(g["d"], g["rmae_new"], marker="s", label=f"new n_iter={n_iter}")
    ax.set_xlabel("d"); ax.set_ylabel("rMAE (h=1, att)")
    ax.set_title("rMAE: baseline (расстояние) vs new (greedy по изоляции)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"pilot_rmae_base_vs_new{SUFFIX}.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(diff, bins=60)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("|err_new| - |err_base|  (<0 = new лучше)")
    ax.set_ylabel("count")
    ax.set_title(f"Парная разница ошибок по origin (n={len(diff)})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"pilot_paired_diff_hist{SUFFIX}.png", dpi=120)
    plt.close(fig)

    print(f"\nРисунки: {FIG_DIR}")


if __name__ == "__main__":
    main()
