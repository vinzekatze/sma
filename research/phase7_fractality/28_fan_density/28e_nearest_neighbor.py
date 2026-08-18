#!/usr/bin/env python3
"""
28e_nearest_neighbor.py — проверка вопроса пользователя: раз при θ→∞ S-map
вырождается почти в "взять единственного ближайшего соседа", нужна ли вообще
взвешенная OLS-регрессия (smap_predict), или можно напрямую скопировать
target ЕДИНСТВЕННОГО ближайшего соседа (1-NN)?

Отличие от S-map при экстремальной θ: OLS-регрессия (intercept+slope) через
почти вырожденную весовую матрицу МОЖЕТ экстраполировать за пределы
наблюдаемых target'ов (это и даёт "overflow"/раздутый p90 хвост в эксп.28d).
Прямой 1-NN lookup — просто копирует РЕАЛЬНО НАБЛЮДАВШЕЕСЯ значение соседа,
экстраполяция невозможна по построению.

Также по комментарию пользователя (из ручных экспериментов в app8) — сетка
по m (размерность вложения): при m=3 у "ближайшего" соседа мало измерений,
чтобы отличаться от прочих (может быть случайное совпадение); с бОльшим m
сравнение специфичнее.

frac_above (топ-20 по расстоянию) остаётся как раньше — меняется способ
вычисления "точки" и добавлена сетка m.
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

M_GRID = [3, 5, 8, 13, 20]
H_MAX = 3
TOP_K = exp28.TOP_K
N_SPLITS = 7
BIG_TABLE_CSV = RESULTS / "nn_mgrid_big_table.csv"


def nn_predict(qv, feats, tars):
    d = np.linalg.norm(feats - qv, axis=1)
    j = int(np.argmin(d))
    return float(tars[j])


def eval_origin_nn(qv0, q_direction, q_last_price, pool_rows, actual_prices, actual_dates, m):
    cur_qv, cur_dir, cum_lr = qv0.copy(), q_direction, 0.0
    rows = []
    for h in range(1, H_MAX + 1):
        s_dir = [(f, t) for f, t, d in pool_rows if d == cur_dir]
        if len(s_dir) < m + 2:
            break
        feats_h = np.array([x[0] for x in s_dir]); tars_h = np.array([x[1] for x in s_dir])

        lr = nn_predict(cur_qv, feats_h, tars_h)
        cum_lr += lr
        point_price = q_last_price * np.exp(cum_lr)

        d = np.linalg.norm(feats_h - cur_qv, axis=1)
        top_idx = np.argsort(d)[:TOP_K]
        target_point_lr = np.log(point_price / q_last_price)
        frac_above = float(np.mean(tars_h[top_idx] > target_point_lr))

        actual_h = float(actual_prices[h - 1]) if h <= len(actual_prices) else float("nan")
        actual_date_h = str(actual_dates[h - 1]) if h <= len(actual_dates) else None
        crash_flag = bool(actual_date_h and exp28.CRASH_LO <= actual_date_h[:10] <= exp28.CRASH_HI)

        rows.append({"h": h, "point": point_price, "q_last_price": q_last_price,
                    "frac_above": frac_above, "actual": actual_h, "crash_flag": crash_flag})

        cur_qv = np.concatenate([[lr], cur_qv[:-1]])
        cur_dir = -cur_dir
    return rows


def process_ticker(target, all_data, m):
    lh_t, ll_t, dates_t = all_data[target]
    full_lp, full_idx, full_dirs = exp28.build_zigzag_raw(lh_t, ll_t, exp28.T_QUERY)
    n_full = len(full_lp)
    origins = list(range(m + 2, n_full - H_MAX - 1))

    rows = []
    for p in origins:
        cutoff_bar = int(full_idx[p])
        cutoff_date = str(dates_t[cutoff_bar])
        lh_c, ll_c = lh_t[:cutoff_bar + 1], ll_t[:cutoff_bar + 1]
        q_lp, q_idx, q_dirs = exp28.build_zigzag_raw(lh_c, ll_c, exp28.T_QUERY)
        if len(q_lp) != p + 1:
            continue
        qv = exp28.build_query_vector(q_lp, m)
        if qv is None:
            continue
        q_direction = int(q_dirs[-1])
        q_last_price = float(np.exp(q_lp[-1]))

        pool_rows = exp28.build_causal_pool(cutoff_date, all_data, m)

        h_avail = min(H_MAX, n_full - 1 - p)
        actual_prices = np.exp(full_lp[p + 1: p + 1 + h_avail]) if h_avail > 0 else np.array([])
        actual_dates = dates_t[full_idx[p + 1: p + 1 + h_avail]] if h_avail > 0 else np.array([])

        origin_rows = eval_origin_nn(qv, q_direction, q_last_price, pool_rows, actual_prices, actual_dates, m)
        for r in origin_rows:
            r["target"] = target
            r["p_orig"] = p
            r["m"] = m
        rows.extend(origin_rows)
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
    print("=== 28e_nearest_neighbor: 1-NN, сетка m ===\n")

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
        for m in M_GRID:
            print(f"--- m={m} ---")
            for target in exp28.UNIVERSE:
                if target not in all_data:
                    continue
                rows = process_ticker(target, all_data, m)
                all_rows.extend(rows)
            print(f"  m={m}: накоплено всего строк {len(all_rows)}")
        big_df = pd.DataFrame(all_rows)
        big_df.to_csv(BIG_TABLE_CSV, index=False)
        print(f"\nСохранено {len(big_df)} строк в {BIG_TABLE_CSV}")

    print("\n=== accuracy/corr по m (h=1, крах исключён) ===")
    grid_summary = []
    for m in M_GRID:
        sub = big_df[(big_df.m == m) & (big_df.h == 1)]
        acc, corr_val, n = pooled_metrics(sub)
        abs_lr = np.abs(np.log(sub.point.values / sub.q_last_price.values))
        med_lr = float(np.median(abs_lr)) if len(abs_lr) else float("nan")
        p90_lr = float(np.percentile(abs_lr, 90)) if len(abs_lr) else float("nan")
        print(f"  m={m:3d}  acc_h1={acc:.3f} corr_h1={corr_val:+.3f} n={n:6d}   "
              f"|lr_точки| median={med_lr:.3f} p90={p90_lr:.3f}")
        grid_summary.append({"m": m, "acc_h1": acc, "corr_h1": corr_val, "n": n,
                             "median_abs_lr": med_lr, "p90_abs_lr": p90_lr})
    pd.DataFrame(grid_summary).to_csv(RESULTS / "nn_mgrid_summary.csv", index=False)

    valid_targets = sorted(big_df.target.unique())
    best_m = max(grid_summary, key=lambda r: r["acc_h1"] if np.isfinite(r["acc_h1"]) else -1)["m"]
    print(f"\nЛучшее m (по общей accuracy h=1): {best_m}")

    print(f"\n=== {N_SPLITS} случайных train/test разбиений тикеров при m={best_m} (стабильность) ===")
    split_results = []
    for seed in range(N_SPLITS):
        rng = np.random.default_rng(seed)
        shuffled = list(valid_targets)
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        test_t = shuffled[half:]
        test_df = big_df[(big_df.target.isin(test_t)) & (big_df.h == 1) & (big_df.m == best_m)]
        acc, corr_val, n = pooled_metrics(test_df)
        print(f"  seed={seed}: TEST acc={acc:.3f} corr={corr_val:+.3f} (n={n})")
        split_results.append({"seed": seed, "acc": acc, "corr_val": corr_val, "n": n})

    split_df = pd.DataFrame(split_results)
    split_df.to_csv(RESULTS / "nn_mgrid_stability_splits.csv", index=False)
    print(f"\nacc: mean={split_df.acc.mean():.3f} std={split_df.acc.std():.3f}")
    print(f"corr: mean={split_df.corr_val.mean():.3f} std={split_df.corr_val.std():.3f}")


if __name__ == "__main__":
    main()
