"""
T02_analyze.py — анализ суррогатного теста.

Читает geometry.csv и lwr.csv, вычисляет:
  - Prism G: p-value rank-тест d_L (original vs surrogate distribution)
  - Prism L: Wilcoxon p-value rMAE (original vs surrogate), процент улучшения
  - LB-gate: разрыв gate=True/False для original и каждого типа суррогата

Сохраняет analysis_summary.json и 3 рисунка.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRIPT_DIR / "results"
FIG_DIR    = OUT_DIR / "figures"   # сохраняем рядом с данными (работает и в Docker)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SURR_ORDER = ["shuffle", "ft", "aaft", "block"]
SIGNALS    = ["ratio", "dratio"]
LP_D       = 3   # d LP-фильтра → порог LB-gate


# ── загрузка ──────────────────────────────────────────────────────────────────

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    geom = pd.read_csv(OUT_DIR / "geometry.csv")
    lwr  = pd.read_csv(OUT_DIR / "lwr.csv")
    lwr["ok"] = lwr["ok"].astype(str).str.lower() == "true"
    return geom, lwr


# ── Prism G: rank-тест ────────────────────────────────────────────────────────

def analyze_geometry(geom: pd.DataFrame) -> dict:
    """
    p-value: доля суррогатов с d_L <= d_L(original).
    H1_G: d_L(original) < d_L(surrogates) → p-value мало.
    """
    results = {}
    for sig in SIGNALS:
        results[sig] = {}
        orig_mask = (geom["signal"] == sig) & (geom["surrogate_type"] == "original")
        orig_dL   = geom.loc[orig_mask, "d_L"].dropna().values

        if len(orig_dL) == 0:
            continue
        d_L_orig = float(np.median(orig_dL))

        for st in SURR_ORDER:
            mask    = (geom["signal"] == sig) & (geom["surrogate_type"] == st)
            surr_dL = geom.loc[mask, "d_L"].dropna().values
            if len(surr_dL) == 0:
                continue
            # one-sided: fraction of surrogates with d_L <= original (low = good for H1)
            p_val = float(np.mean(surr_dL <= d_L_orig))
            results[sig][st] = {
                "d_L_original":     round(d_L_orig, 3),
                "d_L_surr_median":  round(float(np.median(surr_dL)), 3),
                "d_L_surr_mean":    round(float(np.mean(surr_dL)), 3),
                "d_L_surr_q05":     round(float(np.percentile(surr_dL, 5)), 3),
                "d_L_surr_q95":     round(float(np.percentile(surr_dL, 95)), 3),
                "n_surr":           len(surr_dL),
                "p_value_H1_dL_lt_surr": round(p_val, 4),
            }
    return results


# ── Prism L: LWR rMAE ─────────────────────────────────────────────────────────

def analyze_lwr(lwr: pd.DataFrame) -> dict:
    """
    Для каждого (signal, surrogate_type) vs original:
      - mean |error| original
      - mean |error| surrogate
      - % improvement
      - Wilcoxon signed-rank p-value (paired by (ticker, origin))
    """
    results = {}

    for sig in SIGNALS:
        results[sig] = {}
        orig = lwr[(lwr["signal"] == sig) & (lwr["surrogate_type"] == "original") & lwr["ok"]].copy()
        if len(orig) == 0:
            continue
        std_true = orig["abs_error"].std()  # глобальная std для нормировки
        mae_orig = float(orig["abs_error"].mean())

        for st in SURR_ORDER:
            surr = lwr[(lwr["signal"] == sig) & (lwr["surrogate_type"] == st) & lwr["ok"]].copy()
            if len(surr) == 0:
                continue
            mae_surr = float(surr["abs_error"].mean())

            # paired Wilcoxon: для каждого (ticker, origin) сравниваем mean error
            orig_agg  = orig.groupby(["ticker", "origin"])["abs_error"].mean().reset_index()
            surr_agg  = surr.groupby(["ticker", "origin"])["abs_error"].mean().reset_index()
            merged    = orig_agg.merge(surr_agg, on=["ticker", "origin"],
                                       suffixes=("_orig", "_surr"))
            if len(merged) >= 3:
                stat, p_wilcox = sp_stats.wilcoxon(
                    merged["abs_error_orig"], merged["abs_error_surr"],
                    alternative="less")  # H1: original < surrogate
                p_wilcox = float(p_wilcox)
            else:
                p_wilcox = float("nan")

            improvement = (mae_surr - mae_orig) / mae_surr * 100 if mae_surr > 0 else 0

            results[sig][st] = {
                "mae_original":     round(mae_orig, 6),
                "mae_surrogate":    round(mae_surr, 6),
                "improvement_pct":  round(improvement, 2),
                "p_wilcoxon_H1_orig_lt_surr": round(p_wilcox, 4),
                "n_orig_origins":   len(orig_agg),
                "n_surr_pairs":     len(merged),
            }
    return results


# ── LB-gate ───────────────────────────────────────────────────────────────────

def analyze_gate(lwr: pd.DataFrame) -> dict:
    """
    Для каждого (signal, surrogate_type): разрыв mean|error| между gate=True/False.
    LB-gate: d ≥ lb_q_final (d = LP_D = 3).
    """
    results = {}
    lwr = lwr[lwr["ok"]].copy()
    lwr["gate"] = lwr["lb_q_final"].apply(
        lambda x: (x <= LP_D) if np.isfinite(x) else None)

    for sig in SIGNALS:
        results[sig] = {}
        for st in ["original"] + SURR_ORDER:
            mask = (lwr["signal"] == sig) & (lwr["surrogate_type"] == st)
            sub  = lwr[mask].dropna(subset=["gate"])
            if len(sub) == 0:
                continue
            gt = sub[sub["gate"] == True]["abs_error"]
            gf = sub[sub["gate"] == False]["abs_error"]
            if len(gt) < 3 or len(gf) < 3:
                continue
            mae_t = float(gt.mean())
            mae_f = float(gf.mean())
            impr  = (mae_f - mae_t) / mae_f * 100 if mae_f > 0 else 0
            _, p_mw = sp_stats.mannwhitneyu(gt, gf, alternative="less")
            results[sig][st] = {
                "mae_gate_true":  round(mae_t, 6),
                "mae_gate_false": round(mae_f, 6),
                "gate_improvement_pct": round(impr, 2),
                "n_true": len(gt), "n_false": len(gf),
                "p_mannwhitney": round(float(p_mw), 4),
            }
    return results


# ── графики ───────────────────────────────────────────────────────────────────

def plot_geometry(geom: pd.DataFrame, geom_res: dict) -> None:
    surr_types_avail = [s for s in SURR_ORDER if s in geom["surrogate_type"].unique()]
    n_types = len(surr_types_avail)
    if n_types == 0:
        return

    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(6 * len(SIGNALS), 5))
    if len(SIGNALS) == 1:
        axes = [axes]

    for ax, sig in zip(axes, SIGNALS):
        orig_dL = geom[(geom["signal"] == sig) & (geom["surrogate_type"] == "original")]["d_L"].dropna().values
        if len(orig_dL) == 0:
            ax.set_title(sig + " (no data)")
            continue

        data   = [geom[(geom["signal"] == sig) & (geom["surrogate_type"] == st)]["d_L"].dropna().values
                  for st in surr_types_avail]
        labels = surr_types_avail

        parts = ax.violinplot(data, positions=range(n_types), showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.5)

        orig_val = float(np.median(orig_dL))
        ax.axhline(orig_val, color="red", linewidth=2, linestyle="--",
                   label=f"original d_L={orig_val:.2f}")

        # p-values as annotations
        for i, st in enumerate(surr_types_avail):
            if sig in geom_res and st in geom_res[sig]:
                pv = geom_res[sig][st]["p_value_H1_dL_lt_surr"]
                ax.text(i, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10,
                        f"p={pv:.3f}", ha="center", fontsize=8)

        ax.set_xticks(range(n_types))
        ax.set_xticklabels(labels)
        ax.set_ylabel("TwoNN d_L")
        ax.set_title(f"{sig} — d_L distribution (red = original)")
        ax.legend(fontsize=8)

    fig.suptitle("Prism G: TwoNN d_L — original vs surrogates", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "d_L_violin.png", dpi=120)
    plt.close(fig)


def plot_lwr(lwr: pd.DataFrame, lwr_res: dict) -> None:
    lwr_ok = lwr[lwr["ok"]].copy()
    surr_types_avail = [s for s in SURR_ORDER if s in lwr_ok["surrogate_type"].unique()]
    if not surr_types_avail:
        return

    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(7 * len(SIGNALS), 5))
    if len(SIGNALS) == 1:
        axes = [axes]

    for ax, sig in zip(axes, SIGNALS):
        orig = lwr_ok[(lwr_ok["signal"] == sig) & (lwr_ok["surrogate_type"] == "original")]
        mae_orig = float(orig["abs_error"].mean()) if len(orig) > 0 else float("nan")

        mae_surr = []
        labels   = []
        for st in surr_types_avail:
            sub = lwr_ok[(lwr_ok["signal"] == sig) & (lwr_ok["surrogate_type"] == st)]
            mae_surr.append(float(sub["abs_error"].mean()) if len(sub) > 0 else float("nan"))
            pv = lwr_res.get(sig, {}).get(st, {}).get("p_wilcoxon_H1_orig_lt_surr", float("nan"))
            labels.append(f"{st}\n(p={pv:.3f})" if np.isfinite(pv) else st)

        x = np.arange(len(surr_types_avail))
        ax.bar(x, mae_surr, color="steelblue", alpha=0.7, label="surrogate")
        ax.axhline(mae_orig, color="red", linewidth=2, linestyle="--",
                   label=f"original={mae_orig:.5f}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("mean |error|")
        ax.set_title(f"{sig} — LWR quality")
        ax.legend(fontsize=8)

    fig.suptitle("Prism L: mean |error| — original vs surrogates", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lwr_improvement.png", dpi=120)
    plt.close(fig)


def plot_gate(gate_res: dict) -> None:
    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(8 * len(SIGNALS), 5))
    if len(SIGNALS) == 1:
        axes = [axes]

    for ax, sig in zip(axes, SIGNALS):
        if sig not in gate_res:
            ax.set_title(sig + " (no data)")
            continue
        all_keys = ["original"] + SURR_ORDER
        labels, gt_vals, gf_vals = [], [], []
        for st in all_keys:
            if st not in gate_res[sig]:
                continue
            r = gate_res[sig][st]
            labels.append(f"{st}\n(impr={r['gate_improvement_pct']:.1f}%)")
            gt_vals.append(r["mae_gate_true"])
            gf_vals.append(r["mae_gate_false"])
        x = np.arange(len(labels))
        ax.bar(x - 0.2, gt_vals, width=0.4, label="gate=True (d≥LB)", color="green", alpha=0.7)
        ax.bar(x + 0.2, gf_vals, width=0.4, label="gate=False (d<LB)", color="orange", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("mean |error|")
        ax.set_title(f"{sig} — LB-gate (d≥LB) effect")
        ax.legend(fontsize=9)

    fig.suptitle("LB-gate improvement: original vs surrogates", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gate_surrogate.png", dpi=120)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading data from {OUT_DIR}...", flush=True)
    geom, lwr = load()
    print(f"  geometry.csv: {len(geom)} rows, lwr.csv: {len(lwr)} rows", flush=True)

    geom_res = analyze_geometry(geom)
    lwr_res  = analyze_lwr(lwr)
    gate_res = analyze_gate(lwr)

    summary = {
        "geometry": geom_res,
        "lwr":      lwr_res,
        "gate":     gate_res,
    }
    (OUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved analysis_summary.json", flush=True)

    plot_geometry(geom, geom_res)
    plot_lwr(lwr, lwr_res)
    plot_gate(gate_res)
    print(f"Figures saved to {FIG_DIR}", flush=True)

    # краткий вывод в консоль
    print("\n=== Prism G: TwoNN d_L ===")
    for sig in SIGNALS:
        if sig not in geom_res:
            continue
        print(f"\n  Signal: {sig}")
        for st, r in geom_res[sig].items():
            print(f"    {st:8s}: original={r['d_L_original']:.2f}  "
                  f"surr_median={r['d_L_surr_median']:.2f}  "
                  f"p={r['p_value_H1_dL_lt_surr']:.4f}")

    print("\n=== Prism L: LWR quality ===")
    for sig in SIGNALS:
        if sig not in lwr_res:
            continue
        print(f"\n  Signal: {sig}")
        for st, r in lwr_res[sig].items():
            print(f"    {st:8s}: orig={r['mae_original']:.5f}  "
                  f"surr={r['mae_surrogate']:.5f}  "
                  f"impr={r['improvement_pct']:+.1f}%  "
                  f"p={r['p_wilcoxon_H1_orig_lt_surr']:.4f}")

    print("\n=== LB-gate ===")
    for sig in SIGNALS:
        if sig not in gate_res:
            continue
        print(f"\n  Signal: {sig}")
        for st, r in gate_res[sig].items():
            print(f"    {st:8s}: gate_impr={r['gate_improvement_pct']:+.1f}%  "
                  f"p={r['p_mannwhitney']:.4f}")


if __name__ == "__main__":
    main()
