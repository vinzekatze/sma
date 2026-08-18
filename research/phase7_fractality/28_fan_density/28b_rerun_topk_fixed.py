#!/usr/bin/env python3
"""Пересчёт эксп.28 с исправленной (top-K) метрикой, БЕЗ повторной калибровки —
переиспользуем уже откалиброванные (m,theta) из первого прогона."""
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "exp28", "/home/kali/workspace/apps/sma/research/phase7_fractality/28_fan_density/28_fan_density.py")
exp28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp28)

CALIBRATED = {
    "SBER": (3, 20.167), "LKOH": (2, 1.232), "GAZP": (2, 7.263),
    "MGNT": (3, 17.183), "CHMF": (3, 16.501), "MTSS": (4, 7.003), "PLZL": (3, 23.474),
}

print("Предзагрузка тикеров…")
all_data = {}
for tk in exp28.UNIVERSE:
    loaded = exp28.load(tk)
    if loaded is not None:
        all_data[tk] = loaded
print(f"загружено {len(all_data)} тикеров\n")

all_rows = []
for target, (m, theta) in CALIBRATED.items():
    lh_t, ll_t, dates_t = all_data[target]
    full_lp, full_idx, full_dirs = exp28.build_zigzag_raw(lh_t, ll_t, exp28.T_QUERY)
    n_full = len(full_lp)
    origins = list(range(exp28.MIN_HIST, n_full - exp28.H_MAX - 1))
    n_ok = 0
    for p in origins:
        res = exp28.run_origin(p, full_lp, full_idx, lh_t, ll_t, dates_t, all_data, m, theta, exp28.H_MAX)
        if res is None:
            continue
        n_ok += 1
        for r in res["rows"]:
            all_rows.append({"target": target, "m": m, "theta": theta,
                             "p_orig": p, "cutoff_date": res["cutoff_date"][:10], **r})
    print(f"{target}: m={m} θ={theta}  origin'ов с результатом: {n_ok}/{len(origins)}")

df = pd.DataFrame(all_rows)
df.to_csv("/home/kali/workspace/apps/sma/research/phase7_fractality/28_fan_density/results/fan_density_raw_topk.csv", index=False)
print(f"\nСохранено {len(df)} строк")

summary_rows = []
for crash_excl in [True, False]:
    sub = df[~df["crash_flag"]] if crash_excl else df
    sub = sub[np.isfinite(sub["actual"])]
    for (target, h), g in sub.groupby(["target", "h"]):
        actual_above = g["actual"] > g["point"]
        signal_above = g["frac_above"] > 0.5
        acc = float((actual_above == signal_above).mean())
        x = g["frac_above"].values - 0.5
        y = actual_above.values.astype(float)
        corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else float("nan")
        summary_rows.append({"crash_excluded": crash_excl, "target": target, "h": h,
                              "n": len(g), "accuracy": round(acc, 3), "corr": round(corr, 3)})
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("/home/kali/workspace/apps/sma/research/phase7_fractality/28_fan_density/results/fan_density_summary_topk.csv", index=False)

print("\n=== Агрегат (top-K=20), крах исключён ===")
print(summary_df[summary_df.crash_excluded].to_string(index=False))
print("\n=== Агрегат (top-K=20), крах включён ===")
print(summary_df[~summary_df.crash_excluded].to_string(index=False))

print("\n=== Пул по всем тикерам (h=1, крах исключён) ===")
sub = df[(~df["crash_flag"]) & (df["h"] == 1) & np.isfinite(df["actual"])]
actual_above = sub["actual"] > sub["point"]
signal_above = sub["frac_above"] > 0.5
acc_pooled = float((actual_above == signal_above).mean())
x = sub["frac_above"].values - 0.5
y = actual_above.values.astype(float)
corr_pooled = float(np.corrcoef(x, y)[0, 1])
print(f"n={len(sub)}  accuracy={acc_pooled:.3f}  corr={corr_pooled:.3f}")
