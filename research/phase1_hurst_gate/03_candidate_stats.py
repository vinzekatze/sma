"""
Исследование 03: какие статистики val_mape кандидатов лучше предсказывают качество прогноза?

Гипотезы:
  A. val_mape лучшего кандидата (текущий предиктор) — лучшее, что есть
  B. Среднее val_mape по кандидатам (mean) несёт дополнительный сигнал
  C. Разброс val_mape (std, range) — "несогласие моделей" — несёт сигнал
  D. Комбинация признаков лучше одного val_mape

Данные: 02_results.jsonl (candidate_mapes + mape_f5_uniform + val_mape).
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS  = Path(__file__).parent / "results"
IN_FILE  = RESULTS / "02_results.jsonl"


def run():
    records = []
    with open(IN_FILE) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    print(f"Загружено {len(records)} записей")

    import pandas as pd
    from scipy.stats import spearmanr, pearsonr, mannwhitneyu
    from scipy.stats import spearmanr

    rows = []
    for r in records:
        cm = r.get("candidate_mapes")
        if not cm or not isinstance(cm, list) or len(cm) < 2:
            continue
        cm = np.array(cm, dtype=float)
        mape_f5 = r.get("mape_f5_uniform")
        if mape_f5 is None:
            continue
        rows.append({
            "mape_f5":        float(mape_f5),
            "val_best":       float(cm[0]),           # лучший кандидат (текущий)
            "val_worst":      float(cm[-1]),           # худший кандидат
            "val_mean":       float(cm.mean()),        # среднее
            "val_median":     float(np.median(cm)),    # медиана
            "val_std":        float(cm.std()),         # std (несогласие)
            "val_range":      float(cm[-1] - cm[0]),   # max - min
            "val_cv":         float(cm.std() / (cm.mean() + 1e-12)),  # CV
            "val_ratio_bm":   float(cm[0] / (cm.mean() + 1e-12)),    # best / mean
            "val_top2_mean":  float(cm[:2].mean()),    # среднее топ-2
            "val_top3_mean":  float(cm[:3].mean()),    # среднее топ-3
            "n_candidates":   len(cm),
        })

    df = pd.DataFrame(rows)
    df_clean = df[df["mape_f5"] <= 0.20].copy()
    print(f"Без выбросов: n={len(df_clean)}")

    features = [
        "val_best",
        "val_worst",
        "val_mean",
        "val_median",
        "val_std",
        "val_range",
        "val_cv",
        "val_ratio_bm",
        "val_top2_mean",
        "val_top3_mean",
    ]

    print(f"\n{'='*70}")
    print("СПИРМЕН-КОРРЕЛЯЦИИ С mape_f5 (без выбросов)")
    print(f"{'='*70}")
    print(f"  {'Признак':<18} {'r':>7}  {'p':>8}  {'|r| rank'}")
    print(f"  {'-'*50}")

    results_corr = []
    for feat in features:
        col = df_clean[feat].dropna()
        target = df_clean.loc[col.index, "mape_f5"]
        r, p = spearmanr(col, target)
        results_corr.append((feat, r, p))

    results_corr.sort(key=lambda x: abs(x[1]), reverse=True)
    for rank, (feat, r, p) in enumerate(results_corr, 1):
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        marker = " ← текущий" if feat == "val_best" else ""
        print(f"  {feat:<18} {r:>+7.4f}  {p:>8.4f}{stars:<4}  #{rank}{marker}")

    # Сравниваем val_best vs val_mean: есть ли дополнительный сигнал в mean?
    print(f"\n{'='*70}")
    print("ЧАСТИЧНАЯ КОРРЕЛЯЦИЯ: val_mean с mape_f5, контролируя val_best")
    print(f"{'='*70}")
    try:
        from scipy.stats import spearmanr as sp
        # residuals подход: regress val_mean on val_best, взять residuals
        from numpy.polynomial import polynomial as P
        x = df_clean["val_best"].values
        y = df_clean["val_mean"].values
        t = df_clean["mape_f5"].values
        # ранговые residuals
        def rank_residuals(a, b):
            """Ранги a, не объяснённые рангами b."""
            from scipy.stats import rankdata
            ra = rankdata(a)
            rb = rankdata(b)
            coef = np.polyfit(rb, ra, 1)
            return ra - np.polyval(coef, rb)
        res_mean = rank_residuals(y, x)
        res_target = rank_residuals(t, x)
        r_partial, p_partial = sp(res_mean, res_target)
        print(f"  partial_r(val_mean | val_best) = {r_partial:+.4f}  p={p_partial:.4f}")
        if p_partial < 0.05:
            print("  → val_mean несёт дополнительный сигнал поверх val_best!")
        else:
            print("  → val_mean НЕ несёт дополнительного сигнала поверх val_best")

        res_std = rank_residuals(df_clean["val_std"].values, x)
        r_std, p_std = sp(res_std, res_target)
        print(f"  partial_r(val_std  | val_best) = {r_std:+.4f}  p={p_std:.4f}")
        if p_std < 0.05:
            print("  → val_std несёт дополнительный сигнал!")
        else:
            print("  → val_std НЕ несёт дополнительного сигнала")

        res_range = rank_residuals(df_clean["val_range"].values, x)
        r_range, p_range = sp(res_range, res_target)
        print(f"  partial_r(val_range| val_best) = {r_range:+.4f}  p={p_range:.4f}")
        if p_range < 0.05:
            print("  → val_range несёт дополнительный сигнал!")
        else:
            print("  → val_range НЕ несёт дополнительного сигнала")
    except Exception as e:
        print(f"  Ошибка: {e}")

    # Квантильный разрез: если std (несогласие) высокий — хуже ли прогноз?
    print(f"\n{'='*70}")
    print("КВАНТИЛЬНЫЙ РАЗРЕЗ ПО val_std (несогласие кандидатов)")
    print(f"{'='*70}")
    q33, q66 = df_clean["val_std"].quantile([0.33, 0.66])
    for label, mask in [
        (f"val_std < {q33:.5f} (кандидаты согласны)",     df_clean["val_std"] < q33),
        (f"val_std {q33:.5f}–{q66:.5f} (средний разброс)", (df_clean["val_std"] >= q33) & (df_clean["val_std"] < q66)),
        (f"val_std > {q66:.5f} (кандидаты несогласны)",   df_clean["val_std"] >= q66),
    ]:
        grp = df_clean[mask]["mape_f5"]
        print(f"  {label}")
        print(f"    n={len(grp)}  median={grp.median():.4f}  mean={grp.mean():.4f}")

    u_low  = df_clean[df_clean["val_std"] < q33]["mape_f5"].values
    u_high = df_clean[df_clean["val_std"] >= q66]["mape_f5"].values
    stat, p_mw = mannwhitneyu(u_low, u_high, alternative="two-sided")
    stars = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "(ns)"
    print(f"  Mann-Whitney low vs high: p={p_mw:.4f}{stars}")

    # Комбинированный гейт: val_best < thr AND val_std < thr2
    print(f"\n{'='*70}")
    print("КОМБИНИРОВАННЫЙ ГЕЙТ: val_best < median AND val_std < median")
    print(f"{'='*70}")
    med_best = df_clean["val_best"].median()
    med_std  = df_clean["val_std"].median()
    mask_both = (df_clean["val_best"] < med_best) & (df_clean["val_std"] < med_std)
    mask_best_only = (df_clean["val_best"] < med_best)
    grp_both  = df_clean[mask_both]["mape_f5"]
    grp_best  = df_clean[mask_best_only]["mape_f5"]
    grp_other = df_clean[~mask_both]["mape_f5"]
    print(f"  val_best < {med_best:.4f} (текущий гейт):         n={len(grp_best):3d}  median={grp_best.median():.4f}")
    print(f"  val_best < {med_best:.4f} AND val_std < {med_std:.5f}: n={len(grp_both):3d}  median={grp_both.median():.4f}")
    stat2, p2 = mannwhitneyu(grp_both.values, grp_best.values, alternative="two-sided")
    print(f"  Mann-Whitney (оба vs только best): p={p2:.4f}")

    # Регрессия: насколько объясняет дисперсию
    print(f"\n{'='*70}")
    print("ИНФОРМАЦИОННЫЙ ВКЛАД (OLS R²)")
    print(f"{'='*70}")
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        X_single = df_clean[["val_best"]].values
        X_combo  = df_clean[["val_best", "val_std", "val_mean"]].values
        y = df_clean["mape_f5"].values
        sc = StandardScaler()

        def ols_r2(X, y):
            Xs = sc.fit_transform(X)
            m = LinearRegression().fit(Xs, y)
            return m.score(Xs, y)

        r2_single = ols_r2(X_single, y)
        r2_combo  = ols_r2(X_combo, y)
        print(f"  R²(val_best только):                {r2_single:.4f}")
        print(f"  R²(val_best + val_std + val_mean):  {r2_combo:.4f}")
        print(f"  Прирост R²:                         {r2_combo - r2_single:+.4f}")
    except ImportError:
        print("  sklearn не установлен, пропуск OLS")

    print("\nГотово.")


if __name__ == "__main__":
    run()
