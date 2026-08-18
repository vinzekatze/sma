"""
T02_analyze.py — количественный анализ UMAP-геометрии.

Читает umap_points.csv и вычисляет:
  1. LB-статистика по (variant, ticker): median, q05, q95, cv
  2. Временная корреляция LB: r(LB, t) — есть ли тренд локальной размерности
  3. Пространственная кластеризация LB: k-means на UMAP + разрыв LB между кластерами
  4. Устойчивость UMAP: корреляция расстояний между seeds (distance-matrix Spearman)
  5. Сводная таблица по всем вариантам

Выход: results/analysis_T02.json, results/analysis_T02_summary.txt
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "results"
CSV_PATH   = OUT_DIR / "umap_points.csv"

VARIANTS_ORDER = ["A_d3","B_d11","C_dr_p9","D_dr_p33","E_r_p9","F_r_p33",
                  "G_att_p144","H_att_p288","I_d11_p144","J_d11_p288"]

TICKERS  = ["SBER","MRKP","CHMF","NVTK"]
N_SAMPLE = 1000   # для distance-matrix (дорого считать на полном N)
N_CLUST  = 4      # k-means кластеров на UMAP


def load() -> pd.DataFrame:
    """
    CSV смешан по схемам:
      11 col (A, B): ticker,variant,t,att_val,lb_q,umap1_s0,...
      12 col (C-J):  ticker,variant,signal,t,sig_val,lb_q,umap1_s0,...
    Приводим к единой схеме с полями: ticker,variant,signal,t,sig_val,lb_q,umap1_s0,...
    """
    UMAP_COLS = ["umap1_s0","umap2_s0","umap1_s42","umap2_s42","umap1_s123","umap2_s123"]
    COLS_11   = ["ticker","variant","t","att_val","lb_q"] + UMAP_COLS
    COLS_12   = ["ticker","variant","signal","t","sig_val","lb_q"] + UMAP_COLS

    rows: list[dict] = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader)           # пропустить заголовок
        for row in reader:
            if len(row) == 11:
                d = dict(zip(COLS_11, row))
                d["signal"]  = "att"
                d["sig_val"] = d.pop("att_val")
                rows.append(d)
            elif len(row) == 12:
                rows.append(dict(zip(COLS_12, row)))

    df = pd.DataFrame(rows)
    num_cols = ["t", "sig_val", "lb_q"] + UMAP_COLS
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── 1. LB-статистика ─────────────────────────────────────────────────────────

def lb_stats(df: pd.DataFrame) -> dict:
    result = {}
    for var in VARIANTS_ORDER:
        sub = df[df["variant"] == var]["lb_q"].dropna()
        if len(sub) == 0:
            continue
        result[var] = {
            "n":      int(len(sub)),
            "median": round(float(sub.median()), 3),
            "mean":   round(float(sub.mean()),   3),
            "std":    round(float(sub.std()),    3),
            "cv":     round(float(sub.std() / sub.mean()), 3) if sub.mean() > 0 else None,
            "q05":    round(float(sub.quantile(0.05)), 3),
            "q25":    round(float(sub.quantile(0.25)), 3),
            "q75":    round(float(sub.quantile(0.75)), 3),
            "q95":    round(float(sub.quantile(0.95)), 3),
        }
    return result


# ── 2. Временная корреляция LB ────────────────────────────────────────────────

def lb_time_corr(df: pd.DataFrame) -> dict:
    """Spearman r(LB, t) по (variant, ticker)."""
    result = {}
    for var in VARIANTS_ORDER:
        result[var] = {}
        for tk in TICKERS:
            sub = df[(df["variant"] == var) & (df["ticker"] == tk)].dropna(subset=["lb_q"])
            if len(sub) < 10:
                continue
            r, p = sp_stats.spearmanr(sub["t"], sub["lb_q"])
            result[var][tk] = {"r": round(float(r), 4), "p": round(float(p), 4)}
    return result


# ── 3. Пространственная кластеризация LB ─────────────────────────────────────

def lb_cluster_gap(df: pd.DataFrame) -> dict:
    """
    k-means (k=N_CLUST) на UMAP seed=0 координатах.
    Разрыв LB: max(cluster_mean_lb) / min(cluster_mean_lb).
    Если разрыв >> 1 → регионы UMAP имеют разную локальную размерность.
    """
    result = {}
    for var in VARIANTS_ORDER:
        result[var] = {}
        for tk in TICKERS:
            sub = df[(df["variant"] == var) & (df["ticker"] == tk)].dropna(
                subset=["lb_q", "umap1_s0", "umap2_s0"])
            if len(sub) < N_CLUST * 10:
                continue
            xy = sub[["umap1_s0", "umap2_s0"]].values
            km = KMeans(n_clusters=N_CLUST, random_state=0, n_init=5).fit(xy)
            labels = km.labels_
            cluster_lb = [sub["lb_q"].values[labels == c].mean()
                          for c in range(N_CLUST)]
            gap  = max(cluster_lb) / max(min(cluster_lb), 1e-6)
            # Kruskal-Wallis тест: различаются ли LB между кластерами?
            groups = [sub["lb_q"].values[labels == c] for c in range(N_CLUST)
                      if np.sum(labels == c) >= 3]
            if len(groups) >= 2:
                h_stat, p_kw = sp_stats.kruskal(*groups)
            else:
                h_stat, p_kw = float("nan"), float("nan")
            result[var][tk] = {
                "cluster_lb_means": [round(float(x), 3) for x in cluster_lb],
                "gap_max_min":      round(float(gap), 3),
                "kruskal_p":        round(float(p_kw), 4),
            }
    return result


# ── 4. Устойчивость UMAP (distance-matrix Spearman) ──────────────────────────

def umap_stability(df: pd.DataFrame) -> dict:
    """
    Для (variant, ticker): Spearman correlation между попарными расстояниями
    в проекциях seed=0 и seed=42 (на subsample N_SAMPLE точек).
    r → 1: проекции геометрически эквивалентны.
    """
    result = {}
    rng = np.random.default_rng(0)
    for var in VARIANTS_ORDER:
        result[var] = {}
        for tk in TICKERS:
            sub = df[(df["variant"] == var) & (df["ticker"] == tk)].dropna(
                subset=["umap1_s0","umap2_s0","umap1_s42","umap2_s42"])
            if len(sub) < 20:
                continue
            idx = rng.choice(len(sub), size=min(N_SAMPLE, len(sub)), replace=False)
            s0  = sub[["umap1_s0",  "umap2_s0"]].values[idx]
            s42 = sub[["umap1_s42", "umap2_s42"]].values[idx]
            d0  = pdist(s0)
            d42 = pdist(s42)
            r, p = sp_stats.spearmanr(d0, d42)
            result[var][tk] = {"r": round(float(r), 4), "p": round(float(p), 6)}
    return result


# ── 5. Сводная таблица ────────────────────────────────────────────────────────

def summary_table(lb: dict, time_corr: dict,
                  cluster: dict, stability: dict) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append(f"{'Вариант':<14} {'LB med':>7} {'LB q05':>7} {'LB q95':>7} "
                 f"{'CV':>5} {'r(LB,t)':>8} {'gap':>6} {'stab r':>7}")
    lines.append("-" * 90)

    for var in VARIANTS_ORDER:
        if var not in lb:
            continue
        s = lb[var]
        # средняя временная корреляция по тикерам
        tc_vals = [v["r"] for v in time_corr.get(var, {}).values()
                   if "r" in v and np.isfinite(v["r"])]
        tc_mean = round(float(np.mean(tc_vals)), 4) if tc_vals else float("nan")
        # средний gap по тикерам
        gap_vals = [v["gap_max_min"] for v in cluster.get(var, {}).values()
                    if "gap_max_min" in v]
        gap_mean = round(float(np.mean(gap_vals)), 3) if gap_vals else float("nan")
        # средняя устойчивость
        stab_vals = [v["r"] for v in stability.get(var, {}).values()
                     if "r" in v and np.isfinite(v["r"])]
        stab_mean = round(float(np.mean(stab_vals)), 4) if stab_vals else float("nan")

        lines.append(
            f"{var:<14} {s['median']:>7.3f} {s['q05']:>7.3f} {s['q95']:>7.3f} "
            f"{s['cv'] or 0:>5.2f} {tc_mean:>8.4f} {gap_mean:>6.3f} {stab_mean:>7.4f}"
        )
    lines.append("=" * 90)
    lines.append("")
    lines.append("LB med   — медиана локальной размерности (Levina-Bickel)")
    lines.append("CV       — коэфф. вариации LB (гетерогенность по manifold)")
    lines.append("r(LB,t)  — Spearman(LB, time), >0: LB растёт со временем (нестационарность)")
    lines.append("gap      — max/min средней LB по кластерам UMAP (>1: регионы разной размерности)")
    lines.append("stab r   — Spearman дист.матриц seed=0 vs seed=42 (→1: UMAP устойчив)")
    return "\n".join(lines)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Загрузка {CSV_PATH}...", flush=True)
    df = load()
    print(f"  {len(df)} строк, {df['variant'].nunique()} вариантов, "
          f"{df['ticker'].nunique()} тикеров", flush=True)

    print("LB статистика...", flush=True)
    lb = lb_stats(df)

    print("Временная корреляция LB...", flush=True)
    tc = lb_time_corr(df)

    print("Пространственная кластеризация...", flush=True)
    cl = lb_cluster_gap(df)

    print("Устойчивость UMAP...", flush=True)
    st = umap_stability(df)

    result = {"lb_stats": lb, "time_corr": tc, "cluster_gap": cl, "stability": st}
    (OUT_DIR / "analysis_T02.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    print("Сохранено analysis_T02.json", flush=True)

    tbl = summary_table(lb, tc, cl, st)
    (OUT_DIR / "analysis_T02_summary.txt").write_text(tbl)
    print("\n" + tbl, flush=True)


if __name__ == "__main__":
    main()
