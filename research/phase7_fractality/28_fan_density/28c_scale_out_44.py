#!/usr/bin/env python3
"""
28c_scale_out_44.py — расширение эксп.28 на все 44 тикера пула, без
калибровки (m=3, θ=25.7 фикс.) — цель: набрать достаточно тикеров (n=44
вместо n=7), чтобы понять, что отличает тикеры с рабочим сигналом
(SBER/LKOH/MGNT на n=7) от тех, где сигнала нет (GAZP/CHMF/MTSS/PLZL).

Компромисс: без per-ticker калибровки (m,θ) — быстро (минуты вместо часа),
но может недооценивать тикеры, которым нужен другой m/θ (как LKOH с θ=1.23
в эксп.28). Явная оговорка, не скрывать при интерпретации.

Переиспользует функции из 28_fan_density.py (build_zigzag_raw,
build_pool_vectors, build_query_vector, smap_predict/weights, run_origin —
уже с TOP_K=20 после исправления в эксп.28).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("exp28", HERE / "28_fan_density.py")
exp28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp28)

RESULTS = HERE / "results"
DATA = HERE.parents[2] / "data" / "candles"

M_FIXED, THETA_FIXED = 3, 25.7
H_MAX = 3


def ticker_stats(ticker):
    path = DATA / ticker / "1d.json"
    raw = json.load(open(path))
    values = np.array([c.get("value", 0) for c in raw], dtype=float)
    closes = np.array([c["close"] for c in raw], dtype=float)
    closes = np.where(closes <= 0, np.nan, closes)
    lr = np.diff(np.log(closes))
    lr = lr[np.isfinite(lr)]
    vol_annual_pct = float(np.std(lr) * np.sqrt(252) * 100)
    mean_value_bln = float(np.nanmean(values[values > 0]) / 1e9) if (values > 0).any() else float("nan")
    return vol_annual_pct, mean_value_bln


def main():
    print("=== 28c_scale_out_44: расширение на все 44 тикера, без калибровки ===\n")
    print("Предзагрузка тикеров…")
    all_data = {}
    for tk in exp28.UNIVERSE:
        loaded = exp28.load(tk)
        if loaded is not None:
            all_data[tk] = loaded
    print(f"загружено {len(all_data)} тикеров\n")

    rows = []
    for target in exp28.UNIVERSE:
        lh_t, ll_t, dates_t = all_data[target]
        full_lp, full_idx, full_dirs = exp28.build_zigzag_raw(lh_t, ll_t, exp28.T_QUERY)
        n_full = len(full_lp)
        origins = list(range(exp28.MIN_HIST, n_full - H_MAX - 1))
        if len(origins) < 15:
            print(f"{target}: недостаточно origin'ов ({len(origins)}), пропуск")
            continue
        for p in origins:
            res = exp28.run_origin(p, full_lp, full_idx, lh_t, ll_t, dates_t, all_data,
                                    M_FIXED, THETA_FIXED, H_MAX)
            if res is None:
                continue
            for r in res["rows"]:
                rows.append({"target": target, "p_orig": p, "cutoff_date": res["cutoff_date"][:10], **r})
        print(f"{target}: пивотов={n_full} origin'ов_возможно={len(origins)}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "scale_out_44_raw.csv", index=False)
    print(f"\nСохранено {len(df)} строк\n")

    summary = []
    for target, g in df.groupby("target"):
        row = {"target": target}
        for h in [1, 2, 3]:
            gh = g[(g.h == h) & (~g.crash_flag) & np.isfinite(g.actual)]
            if len(gh) < 10:
                row[f"acc_h{h}"] = float("nan"); row[f"corr_h{h}"] = float("nan"); row[f"n_h{h}"] = len(gh)
                continue
            actual_above = gh.actual > gh.point
            signal_above = gh.frac_above > 0.5
            acc = float((actual_above == signal_above).mean())
            x = gh.frac_above.values - 0.5
            y = actual_above.values.astype(float)
            corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 else float("nan")
            row[f"acc_h{h}"] = round(acc, 3); row[f"corr_h{h}"] = round(corr, 3); row[f"n_h{h}"] = len(gh)
        vol, liq = ticker_stats(target)
        row["vol_annual_pct"] = round(vol, 1)
        row["mean_value_bln"] = round(liq, 3)
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(RESULTS / "scale_out_44_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    print("\n=== Корреляция силы сигнала (corr_h1, corr_h3) с характеристиками тикера ===")
    for col in ["vol_annual_pct", "mean_value_bln", "n_h1"]:
        for h in [1, 3]:
            sub = summary_df.dropna(subset=[f"corr_h{h}", col])
            if len(sub) < 10:
                continue
            c = np.corrcoef(sub[col], sub[f"corr_h{h}"])[0, 1]
            print(f"  corr({col}, corr_h{h}) = {c:+.3f}  (n={len(sub)})")


if __name__ == "__main__":
    main()
