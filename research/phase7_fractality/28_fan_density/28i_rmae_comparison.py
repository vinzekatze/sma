#!/usr/bin/env python3
"""
28i_rmae_comparison.py — честное сравнение rMAE: S-map (взвешенная регрессия
по расстоянию) vs K-усреднение (равные веса, эксп.28h) на ТЕХ ЖЕ origin'ах
(m=13, все 44 тикера). Эксп.28h показал, что accuracy (сторона относительно
точки) и rMAE (реальное качество прогноза цены) дают ПРОТИВОПОЛОЖНЫЕ
рекомендации по K — нужно теперь честно сравнить лучший K (=8) с S-map при
нескольких θ на идентичных origin'ах.

persistence = q_last_price (цена не меняется) — простейший, наименее спорный
baseline, не завязанный на дискуссионную формулу альтернирующего зигзага
(feedback_rmae_persistence).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("exp28", HERE / "28_fan_density.py")
exp28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp28)

RESULTS = HERE / "results"
M_FIXED = 13
H_MAX = 3
THETA_GRID = [0.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.7, 35.0, 50.0]
MIN_HIST = M_FIXED + 2
BIG_TABLE_CSV = RESULTS / "smap_rmae_big_table.csv"


def eval_origin_smap(qv0, q_direction, q_last_price, pool_rows, actual_prices, theta_grid):
    out = []
    for theta in theta_grid:
        cur_qv, cur_dir, cum_lr = qv0.copy(), q_direction, 0.0
        for h in range(1, H_MAX + 1):
            s_dir = [(f, t) for f, t, d in pool_rows if d == cur_dir]
            if len(s_dir) < M_FIXED + 2:
                break
            feats_h = np.array([x[0] for x in s_dir]); tars_h = np.array([x[1] for x in s_dir])
            lr = exp28.smap_predict(cur_qv, feats_h, tars_h, theta)
            cum_lr += lr
            point_price = q_last_price * np.exp(cum_lr)
            actual_h = float(actual_prices[h - 1]) if h <= len(actual_prices) else float("nan")
            out.append({"theta": theta, "h": h, "point": point_price, "actual": actual_h})
            cur_qv = np.concatenate([[lr], cur_qv[:-1]])
            cur_dir = -cur_dir
    return out


def process_ticker(target, all_data):
    lh_t, ll_t, dates_t = all_data[target]
    full_lp, full_idx, full_dirs = exp28.build_zigzag_raw(lh_t, ll_t, exp28.T_QUERY)
    n_full = len(full_lp)
    origins = list(range(MIN_HIST, n_full - H_MAX - 1))

    rows = []
    for p in origins:
        cutoff_bar = int(full_idx[p])
        cutoff_date = str(dates_t[cutoff_bar])
        lh_c, ll_c = lh_t[:cutoff_bar + 1], ll_t[:cutoff_bar + 1]
        q_lp, q_idx, q_dirs = exp28.build_zigzag_raw(lh_c, ll_c, exp28.T_QUERY)
        if len(q_lp) != p + 1:
            continue
        qv = exp28.build_query_vector(q_lp, M_FIXED)
        if qv is None:
            continue
        q_direction = int(q_dirs[-1])
        q_last_price = float(np.exp(q_lp[-1]))

        pool_rows = exp28.build_causal_pool(cutoff_date, all_data, M_FIXED)

        h_avail = min(H_MAX, n_full - 1 - p)
        actual_prices = np.exp(full_lp[p + 1: p + 1 + h_avail]) if h_avail > 0 else np.array([])
        actual_dates = dates_t[full_idx[p + 1: p + 1 + h_avail]] if h_avail > 0 else np.array([])

        origin_rows = eval_origin_smap(qv, q_direction, q_last_price, pool_rows, actual_prices, THETA_GRID)
        for r in origin_rows:
            h = r["h"]
            actual_date_h = str(actual_dates[h - 1]) if h <= len(actual_dates) else None
            r["crash_flag"] = bool(actual_date_h and exp28.CRASH_LO <= actual_date_h[:10] <= exp28.CRASH_HI)
            r["target"] = target
            r["p_orig"] = p
            r["q_last_price"] = q_last_price
        rows.extend(origin_rows)
        del pool_rows
    return rows


def rmae(df):
    df = df[~df.crash_flag & np.isfinite(df.actual)]
    if len(df) < 20:
        return float("nan"), len(df)
    err = np.abs(df.point - df.actual)
    pers_err = np.abs(df.q_last_price - df.actual)
    return float(err.mean() / pers_err.mean()), len(df)


def main():
    print(f"=== 28i_rmae_comparison: S-map (m={M_FIXED}) по сетке θ, те же origin'ы что эксп.28h ===\n")

    if BIG_TABLE_CSV.exists():
        print(f"Загрузка готовой таблицы {BIG_TABLE_CSV}…")
        big_df = pd.read_csv(BIG_TABLE_CSV)
    else:
        print("Предзагрузка тикеров…")
        all_data = {}
        for tk in exp28.UNIVERSE:
            loaded = exp28.load(tk)
            if loaded is not None:
                all_data[tk] = loaded
        print(f"загружено {len(all_data)} тикеров\n")

        all_rows = []
        for target in exp28.UNIVERSE:
            if target not in all_data:
                continue
            rows = process_ticker(target, all_data)
            all_rows.extend(rows)
            print(f"  {target}: строк={len(rows)} (всего {len(all_rows)})")
        big_df = pd.DataFrame(all_rows)
        big_df.to_csv(BIG_TABLE_CSV, index=False)
        print(f"\nСохранено {len(big_df)} строк в {BIG_TABLE_CSV}")

    print("\n=== rMAE S-map по θ (h=1,2,3) ===")
    smap_summary = []
    for h in [1, 2, 3]:
        print(f"--- h={h} ---")
        for theta in THETA_GRID:
            sub = big_df[(big_df.theta == theta) & (big_df.h == h)]
            r, n = rmae(sub)
            print(f"  θ={theta:6.1f}  rMAE={r:.4f}  n={n}")
            smap_summary.append({"theta": theta, "h": h, "rmae": r, "n": n})
    pd.DataFrame(smap_summary).to_csv(RESULTS / "smap_rmae_summary.csv", index=False)

    # ── сравнение с K-усреднением (эксп.28h), на тех же данных ──────────────
    kavg_path = RESULTS / "kavg_big_table.csv"
    if kavg_path.exists():
        kdf = pd.read_csv(kavg_path)
        kdf = kdf[~kdf.crash_flag & np.isfinite(kdf.actual)]
        print("\n=== Сводное сравнение rMAE (h=1): K-усреднение vs S-map ===")
        for k in sorted(kdf.k.unique()):
            sub = kdf[(kdf.k == k) & (kdf.h == 1)]
            err = np.abs(sub.point - sub.actual)
            pers_err = np.abs(sub.q_last_price - sub.actual)
            r = float(err.mean() / pers_err.mean())
            print(f"  K-avg K={k:3d}:  rMAE={r:.4f}  n={len(sub)}")
        for theta in THETA_GRID:
            sub = big_df[(big_df.theta == theta) & (big_df.h == 1)]
            r, n = rmae(sub)
            print(f"  S-map θ={theta:6.1f}:  rMAE={r:.4f}  n={n}")


if __name__ == "__main__":
    main()
