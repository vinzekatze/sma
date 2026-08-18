#!/usr/bin/env python3
"""
28d_theta_for_density.py — калибровка θ ПОД САМУ метрику плотности
(accuracy/corr frac_above vs actual>point на h=1), а не под rMAE точки
(эксп.28) и не без калибровки вообще (эксп.28c).

Эксп.28c показал: с фиксированной θ=25.7 сигнал почти универсален на 44
тикерах, тогда как per-ticker калибровка (под rMAE точки, эксп.28) местами
ЕГО ОСЛАБЛЯЛА (CHMF, MTSS). Здесь ищем θ, максимизирующую именно сигнал
плотности — сеткой до θ=500 (проверка вырождения в "почти один сосед",
которое пользователь наблюдал раньше как нестабильный эффект в html).

Архитектура (важно, чтобы не упасть в OOM — было на этой машине с 6.8GB RAM
при попытке держать все пулы в памяти/pickle одновременно):
  - для каждого origin'а пул строится ОДИН РАЗ, сразу прогоняется по ВСЕЙ
    сетке θ, в результирующую таблицу попадают только СКАЛЯРЫ (theta, h,
    point, frac_above, actual, crash_flag, target, p_orig) — сам пул
    (numpy-массивы) отбрасывается сразу после origin'а, не накапливается.
  - результирующая таблица (~200К строк скаляров) — лёгкая, дальше на НЕЙ
    делается МНОЖЕСТВО случайных train/test разбиений тикеров (дёшево,
    просто group-by/filter), чтобы проверить СТАБИЛЬНОСТЬ выбора θ и
    итоговой accuracy — не полагаться на одно разбиение.
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

M_FIXED = 3
H_MAX = 3
THETA_GRID = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 21.0,
              25.7, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0, 150.0, 250.0, 500.0]
TOP_K = exp28.TOP_K
N_SPLITS = 7
BIG_TABLE_CSV = RESULTS / "theta_big_table.csv"


def eval_all_thetas_on_origin(qv0, q_direction, q_last_price, pool_rows,
                              actual_prices, actual_dates, theta_grid):
    """Для ОДНОГО построенного пула — прогон по ВСЕЙ сетке θ.
    Возвращает список строк-скаляров (theta, h, point, frac_above, actual, crash_flag)."""
    out = []
    for theta in theta_grid:
        cur_qv, cur_dir, cum_lr = qv0.copy(), q_direction, 0.0
        for h in range(1, H_MAX + 1):
            s_dir = [(f, t) for f, t, d in pool_rows if d == cur_dir]
            if len(s_dir) < M_FIXED + 2:
                break
            feats_h = np.array([x[0] for x in s_dir]); tars_h = np.array([x[1] for x in s_dir])
            weights_h = exp28.smap_weights(cur_qv, feats_h, theta)

            lr = exp28.smap_predict(cur_qv, feats_h, tars_h, theta)
            cum_lr += lr
            point_price = q_last_price * np.exp(cum_lr)

            top_idx = np.argsort(weights_h)[::-1][:TOP_K]
            target_point_lr = np.log(point_price / q_last_price)
            frac_above = float(np.mean(tars_h[top_idx] > target_point_lr))

            actual_h = float(actual_prices[h - 1]) if h <= len(actual_prices) else float("nan")
            actual_date_h = str(actual_dates[h - 1]) if h <= len(actual_dates) else None
            crash_flag = bool(actual_date_h and exp28.CRASH_LO <= actual_date_h[:10] <= exp28.CRASH_HI)

            out.append({"theta": theta, "h": h, "point": point_price, "q_last_price": q_last_price,
                        "frac_above": frac_above, "actual": actual_h, "crash_flag": crash_flag})

            cur_qv = np.concatenate([[lr], cur_qv[:-1]])
            cur_dir = -cur_dir
    return out


def process_ticker(target, all_data, theta_grid):
    lh_t, ll_t, dates_t = all_data[target]
    full_lp, full_idx, full_dirs = exp28.build_zigzag_raw(lh_t, ll_t, exp28.T_QUERY)
    n_full = len(full_lp)
    origins = list(range(exp28.MIN_HIST, n_full - H_MAX - 1))

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

        origin_rows = eval_all_thetas_on_origin(qv, q_direction, q_last_price, pool_rows,
                                                actual_prices, actual_dates, theta_grid)
        for r in origin_rows:
            r["target"] = target
            r["p_orig"] = p
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
    corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 else float("nan")
    return acc, corr, len(df)


def main():
    print("=== 28d_theta_for_density: калибровка θ под метрику плотности ===\n")

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
            rows = process_ticker(target, all_data, THETA_GRID)
            all_rows.extend(rows)
            print(f"  {target}: origin×theta×h строк = {len(rows)}  (всего накоплено {len(all_rows)})")
        big_df = pd.DataFrame(all_rows)
        big_df.to_csv(BIG_TABLE_CSV, index=False)
        print(f"\nСохранено {len(big_df)} строк в {BIG_TABLE_CSV}")

    valid_targets = sorted(big_df.target.unique())
    print(f"\nВсего тикеров с данными: {len(valid_targets)}")

    # ── диагностика вырождения точки (по всей выборке, h=1) ──────────────────
    print("\n=== |lr_точки h=1| по θ (диагностика вырождения) ===")
    for theta in THETA_GRID:
        sub = big_df[(big_df.theta == theta) & (big_df.h == 1)]
        acc, corr, n = pooled_metrics(sub)
        abs_lr = np.abs(np.log(sub.point.values / sub.q_last_price.values))
        med_lr, p90_lr = float(np.median(abs_lr)), float(np.percentile(abs_lr, 90))
        print(f"  θ={theta:6.1f}  acc_h1={acc:.3f} corr_h1={corr:+.3f} n={n:5d}   "
              f"|lr_точки| median={med_lr:.3f} p90={p90_lr:.3f}")

    # ── МНОЖЕСТВЕННЫЕ случайные train/test разбиения тикеров (стабильность) ──
    print(f"\n=== {N_SPLITS} случайных train/test разбиений тикеров (стабильность best θ) ===")
    split_results = []
    for seed in range(N_SPLITS):
        rng = np.random.default_rng(seed)
        shuffled = list(valid_targets)
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        train_t, test_t = shuffled[:half], shuffled[half:]

        train_df = big_df[big_df.target.isin(train_t)]
        best_theta, best_acc = None, -1.0
        for theta in THETA_GRID:
            sub = train_df[(train_df.theta == theta) & (train_df.h == 1)]
            acc, corr, n = pooled_metrics(sub)
            if np.isfinite(acc) and acc > best_acc:
                best_acc, best_theta = acc, theta

        test_df = big_df[big_df.target.isin(test_t)]
        sub_test = test_df[(test_df.theta == best_theta) & (test_df.h == 1)]
        acc_test, corr_test, n_test = pooled_metrics(sub_test)
        sub_test_fixed = test_df[(test_df.theta == 25.7) & (test_df.h == 1)]
        acc_fixed, corr_fixed, n_fixed = pooled_metrics(sub_test_fixed)

        print(f"  seed={seed}: best_theta(train)={best_theta:6.1f} acc_train={best_acc:.3f}  |  "
              f"TEST acc={acc_test:.3f} corr={corr_test:+.3f} (n={n_test})  vs  "
              f"fixed θ=25.7: acc={acc_fixed:.3f} corr={corr_fixed:+.3f}")
        split_results.append({"seed": seed, "best_theta": best_theta, "acc_train": best_acc,
                              "acc_test_best": acc_test, "corr_test_best": corr_test,
                              "acc_test_fixed257": acc_fixed, "corr_test_fixed257": corr_fixed})

    split_df = pd.DataFrame(split_results)
    split_df.to_csv(RESULTS / "theta_stability_splits.csv", index=False)
    print("\nСводка по разбиениям:")
    print(split_df.to_string(index=False))
    print(f"\nbest_theta: mean={split_df.best_theta.mean():.1f} std={split_df.best_theta.std():.1f} "
          f"min={split_df.best_theta.min()} max={split_df.best_theta.max()}")
    print(f"acc_test_best: mean={split_df.acc_test_best.mean():.3f} std={split_df.acc_test_best.std():.3f}")
    print(f"acc_test_fixed257: mean={split_df.acc_test_fixed257.mean():.3f} std={split_df.acc_test_fixed257.std():.3f}")


if __name__ == "__main__":
    main()
