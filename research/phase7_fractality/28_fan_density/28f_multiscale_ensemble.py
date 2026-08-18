#!/usr/bin/env python3
"""
28f_multiscale_ensemble.py — ансамбль 1-NN по НЕСКОЛЬКИМ m одновременно
(усреднение), вместо выбора одного "лучшего" m (эксп.28e).

Мотивация пользователя: раз это применение S-map/KNN здесь чисто
эмпирическое (не строгая реконструкция фазового пространства по Такенсу),
нет причин искать ЕДИНСТВЕННОЕ "правильное" m — логичнее смешать несколько
масштабов вложения сразу, как обычный KNN-ансамбль.

Конструкция: на каждом шаге h для КАЖДОГО m∈ENSEMBLE_M ищем 1-NN (ближайший
по m-мерному вектору запроса), берём его target log-доходность, УСРЕДНЯЕМ
по всем m — это и есть комбинированная точка шага. Для frac_above объединяем
топ-K (по расстоянию) кандидатов от ВСЕХ m в один общий пул.
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

ENSEMBLE_M = [3, 5, 8, 13]
H_MAX = 3
TOP_K_PER_M = 8   # топ-K от КАЖДОГО m в общий пул для frac_above (4*8=32 всего)
N_SPLITS = 7
BIG_TABLE_CSV = RESULTS / "ensemble_big_table.csv"
MIN_HIST = max(ENSEMBLE_M) + 2


def eval_origin_ensemble(q_lp_by_m, q_direction, q_last_price, all_data, cutoff_date, ensemble_m):
    cur_dir = q_direction
    cum_lr = 0.0
    cur_qv_by_m = {m: exp28.build_query_vector(q_lp_by_m[m], m) for m in ensemble_m}

    # пул строится ОДИН РАЗ на origin для каждого m (не зависит от h/direction)
    pools_by_m = {m: exp28.build_causal_pool(cutoff_date, all_data, m) for m in ensemble_m}

    rows = []
    for h in range(1, H_MAX + 1):
        per_m_lr = []
        pooled_candidates = []  # (target_lr,) от топ-K каждого m
        ok = True
        for m in ensemble_m:
            pool_rows = pools_by_m[m]
            s_dir = [(f, t) for f, t, d in pool_rows if d == cur_dir]
            if len(s_dir) < m + 2:
                ok = False
                break
            feats_h = np.array([x[0] for x in s_dir]); tars_h = np.array([x[1] for x in s_dir])
            qv = cur_qv_by_m[m]
            d = np.linalg.norm(feats_h - qv, axis=1)
            j = int(np.argmin(d))
            per_m_lr.append(float(tars_h[j]))
            top_idx = np.argsort(d)[:TOP_K_PER_M]
            pooled_candidates.extend(tars_h[top_idx].tolist())
        if not ok or not per_m_lr:
            break

        lr_combined = float(np.mean(per_m_lr))
        cum_lr += lr_combined
        point_price = q_last_price * np.exp(cum_lr)

        target_point_lr = np.log(point_price / q_last_price)
        frac_above = float(np.mean(np.array(pooled_candidates) > target_point_lr))

        rows.append({"h": h, "point": point_price, "frac_above": frac_above,
                    "lr_combined": lr_combined})

        for m in ensemble_m:
            cur_qv_by_m[m] = np.concatenate([[per_m_lr[ensemble_m.index(m)]], cur_qv_by_m[m][:-1]])
        cur_dir = -cur_dir
    return rows


def process_ticker(target, all_data, ensemble_m):
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
        q_direction = int(q_dirs[-1])
        q_last_price = float(np.exp(q_lp[-1]))
        q_lp_by_m = {m: q_lp for m in ensemble_m}  # один и тот же q_lp, build_query_vector сам берёт нужный хвост

        h_avail = min(H_MAX, n_full - 1 - p)
        actual_prices = np.exp(full_lp[p + 1: p + 1 + h_avail]) if h_avail > 0 else np.array([])
        actual_dates = dates_t[full_idx[p + 1: p + 1 + h_avail]] if h_avail > 0 else np.array([])

        origin_rows = eval_origin_ensemble(q_lp_by_m, q_direction, q_last_price, all_data, cutoff_date, ensemble_m)
        for i, r in enumerate(origin_rows):
            h = r["h"]
            r["actual"] = float(actual_prices[h - 1]) if h <= len(actual_prices) else float("nan")
            actual_date_h = str(actual_dates[h - 1]) if h <= len(actual_dates) else None
            r["crash_flag"] = bool(actual_date_h and exp28.CRASH_LO <= actual_date_h[:10] <= exp28.CRASH_HI)
            r["target"] = target
            r["p_orig"] = p
            r["q_last_price"] = q_last_price
        rows.extend(origin_rows)
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
    print(f"=== 28f_multiscale_ensemble: 1-NN усреднённый по m={ENSEMBLE_M} ===\n")

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
            rows = process_ticker(target, all_data, ENSEMBLE_M)
            all_rows.extend(rows)
            print(f"  {target}: строк={len(rows)} (всего {len(all_rows)})")
        big_df = pd.DataFrame(all_rows)
        big_df.to_csv(BIG_TABLE_CSV, index=False)
        print(f"\nСохранено {len(big_df)} строк в {BIG_TABLE_CSV}")

    print("\n=== accuracy/corr ансамбля по h (крах исключён) ===")
    for h in [1, 2, 3]:
        sub = big_df[big_df.h == h]
        acc, corr_val, n = pooled_metrics(sub)
        abs_lr = np.abs(np.log(sub.point.values / sub.q_last_price.values))
        med_lr = float(np.median(abs_lr)) if len(abs_lr) else float("nan")
        p90_lr = float(np.percentile(abs_lr, 90)) if len(abs_lr) else float("nan")
        print(f"  h={h}  acc={acc:.3f} corr={corr_val:+.3f} n={n:6d}   "
              f"|lr_точки| median={med_lr:.3f} p90={p90_lr:.3f}")

    valid_targets = sorted(big_df.target.unique())
    print(f"\n=== {N_SPLITS} случайных train/test разбиений (стабильность, h=1) ===")
    split_results = []
    for seed in range(N_SPLITS):
        rng = np.random.default_rng(seed)
        shuffled = list(valid_targets)
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        test_t = shuffled[half:]
        test_df = big_df[(big_df.target.isin(test_t)) & (big_df.h == 1)]
        acc, corr_val, n = pooled_metrics(test_df)
        print(f"  seed={seed}: TEST acc={acc:.3f} corr={corr_val:+.3f} (n={n})")
        split_results.append({"seed": seed, "acc": acc, "corr_val": corr_val, "n": n})

    split_df = pd.DataFrame(split_results)
    split_df.to_csv(RESULTS / "ensemble_stability_splits.csv", index=False)
    print(f"\nacc: mean={split_df.acc.mean():.3f} std={split_df.acc.std():.3f}")
    print(f"corr: mean={split_df.corr_val.mean():.3f} std={split_df.corr_val.std():.3f}")


if __name__ == "__main__":
    main()
