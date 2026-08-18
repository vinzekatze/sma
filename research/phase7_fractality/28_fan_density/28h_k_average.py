#!/usr/bin/env python3
"""
28h_k_average.py — K ближайших соседей (K=1,2,3,5,8), простое усреднение
target'ов, БЕЗ регрессии (в отличие от S-map) и БЕЗ смешивания разных m
(в отличие от провалившегося ансамбля эксп.28f, где усреднялись РАЗНЫЕ,
несогласующиеся ближайшие соседи от разных m).

Здесь усредняются K ближайших соседей В РАМКАХ ОДНОГО m=13 (лучшее плато из
эксп.28e) — это стандартный kNN-regressor (Надарая-Уотсон без затухания по
расстоянию, равные веса), промежуточный вариант между 1-NN (K=1) и полным
S-map (взвешенная регрессия по всему пулу). Проверяем: помогает ли небольшое
усреднение соседей одного масштаба (в отличие от усреднения РАЗНЫХ
масштабов, которое ухудшило результат), или лучше остаться на K=1.
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
K_GRID = [1, 2, 3, 5, 8, 13, 20]
TOP_K_DENSITY = 20
N_SPLITS = 7
MIN_HIST = M_FIXED + 2
BIG_TABLE_CSV = RESULTS / "kavg_big_table.csv"


def eval_origin_kavg(qv0, q_direction, q_last_price, pool_rows, actual_prices, actual_dates, k_grid):
    """Для одного origin'а — для каждого K в сетке независимая цепочка (K
    влияет только на то, сколько ближайших усредняется в 'точку')."""
    out_by_k = {k: [] for k in k_grid}
    for k in k_grid:
        cur_qv, cur_dir, cum_lr = qv0.copy(), q_direction, 0.0
        for h in range(1, H_MAX + 1):
            s_dir = [(f, t) for f, t, d in pool_rows if d == cur_dir]
            if len(s_dir) < M_FIXED + 2:
                break
            feats_h = np.array([x[0] for x in s_dir]); tars_h = np.array([x[1] for x in s_dir])
            d = np.linalg.norm(feats_h - cur_qv, axis=1)
            k_eff = min(k, len(d))
            top_idx = np.argsort(d)[:k_eff]
            lr = float(np.mean(tars_h[top_idx]))
            cum_lr += lr
            point_price = q_last_price * np.exp(cum_lr)

            dens_idx = np.argsort(d)[:TOP_K_DENSITY]
            target_point_lr = np.log(point_price / q_last_price)
            frac_above = float(np.mean(tars_h[dens_idx] > target_point_lr))

            actual_h = float(actual_prices[h - 1]) if h <= len(actual_prices) else float("nan")
            actual_date_h = str(actual_dates[h - 1]) if h <= len(actual_dates) else None
            crash_flag = bool(actual_date_h and exp28.CRASH_LO <= actual_date_h[:10] <= exp28.CRASH_HI)

            out_by_k[k].append({"h": h, "point": point_price, "frac_above": frac_above,
                                "actual": actual_h, "crash_flag": crash_flag})

            cur_qv = np.concatenate([[lr], cur_qv[:-1]])
            cur_dir = -cur_dir
    return out_by_k


def process_ticker(target, all_data, k_grid):
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

        out_by_k = eval_origin_kavg(qv, q_direction, q_last_price, pool_rows, actual_prices, actual_dates, k_grid)
        for k, kr in out_by_k.items():
            for r in kr:
                r["target"] = target
                r["p_orig"] = p
                r["q_last_price"] = q_last_price
                r["k"] = k
            rows.extend(kr)
        del pool_rows
    return rows


def pooled_metrics(df):
    df = df[~df.crash_flag & np.isfinite(df.actual)]
    if len(df) < 20:
        return float("nan"), float("nan"), len(df)
    actual_above = (df.actual > df.point).values
    signal_above = (df.frac_above > 0.5).values
    acc = float(np.mean(actual_above == signal_above))
    x = df.frac_above.values - 0.5
    y = actual_above.astype(float)
    corr_val = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 else float("nan")
    return acc, corr_val, len(df)


def main():
    print(f"=== 28h_k_average: K ближайших соседей (m={M_FIXED} фикс.) ===\n")

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
            rows = process_ticker(target, all_data, K_GRID)
            all_rows.extend(rows)
            print(f"  {target}: строк={len(rows)} (всего {len(all_rows)})")
        big_df = pd.DataFrame(all_rows)
        big_df.to_csv(BIG_TABLE_CSV, index=False)
        print(f"\nСохранено {len(big_df)} строк в {BIG_TABLE_CSV}")

    print("\n=== accuracy/corr по K (h=1, крах исключён) ===")
    grid_summary = []
    for k in K_GRID:
        sub = big_df[(big_df.k == k) & (big_df.h == 1)]
        acc, corr_val, n = pooled_metrics(sub)
        print(f"  K={k:3d}  acc_h1={acc:.3f} corr_h1={corr_val:+.3f} n={n:6d}")
        grid_summary.append({"k": k, "acc_h1": acc, "corr_h1": corr_val, "n": n})
    pd.DataFrame(grid_summary).to_csv(RESULTS / "kavg_summary.csv", index=False)

    valid_targets = sorted(big_df.target.unique())
    best_k = max(grid_summary, key=lambda r: r["acc_h1"] if np.isfinite(r["acc_h1"]) else -1)["k"]
    print(f"\nЛучшее K: {best_k}")

    print(f"\n=== {N_SPLITS} случайных train/test разбиений при K={best_k} (стабильность) ===")
    split_results = []
    for seed in range(N_SPLITS):
        rng = np.random.default_rng(seed)
        shuffled = list(valid_targets)
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        test_t = shuffled[half:]
        test_df = big_df[(big_df.target.isin(test_t)) & (big_df.h == 1) & (big_df.k == best_k)]
        acc, corr_val, n = pooled_metrics(test_df)
        print(f"  seed={seed}: TEST acc={acc:.3f} corr={corr_val:+.3f} (n={n})")
        split_results.append({"seed": seed, "acc": acc, "corr_val": corr_val, "n": n})

    split_df = pd.DataFrame(split_results)
    split_df.to_csv(RESULTS / "kavg_stability_splits.csv", index=False)
    print(f"\nacc: mean={split_df.acc.mean():.3f} std={split_df.acc.std():.3f}")
    print(f"corr: mean={split_df.corr_val.mean():.3f} std={split_df.corr_val.std():.3f}")


if __name__ == "__main__":
    main()
