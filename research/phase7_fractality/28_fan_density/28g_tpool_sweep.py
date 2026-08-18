#!/usr/bin/env python3
"""
28g_tpool_sweep.py — не фиксировать T_pool=0.8987×T_query (эксп.17f), а
перебирать сетку T_pool (в т.ч. БОЛЬШЕ базового T_query — в этом контексте
это уже не "T_frac" в исходном смысле событийной фрактальности, а просто
ещё одна степень свободы поиска), искать ЕДИНСТВЕННОГО глобально ближайшего
соседа среди кандидатов от ВСЕХ T_pool сразу (не в рамках одного
фиксированного порога).

Идея пользователя, продолжение эксп.28e/28f (1-NN не хуже S-map, ансамбль
по m не сильно выигрывает у одного хорошего m). Сокращённая выборка (15
тикеров, m=8 фиксировано — среднее из найденного плато 8-13 в эксп.28e) —
компромисс по времени, автономный прогон.

⚠️ Автономный прогон без промежуточного подтверждения пользователя (он
попросил "проверяй на автомате" и отошёл) — параметры выбраны разумно по
уже накопленному контексту сессии, но не согласованы явно построчно.
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

TARGETS = ["SBER", "LKOH", "GAZP", "MGNT", "CHMF", "MTSS", "PLZL",
          "NVTK", "VTBR", "NLMK", "ROSN", "TATN", "SNGS", "AFLT", "MOEX"]
M_FIXED = 8
H_MAX = 3
TPOOL_GRID = [0.05, 0.08, 0.12, 0.15, 0.18, 0.1797, 0.22, 0.27, 0.32]
TOP_K = 20
N_SPLITS = 5
MIN_HIST = M_FIXED + 2
BIG_TABLE_CSV = RESULTS / "tpool_sweep_big_table.csv"


def build_pool_at_tpool(cutoff_date, all_data, m, t_pool):
    pool_rows = []
    for tk, (lh_k, ll_k, dates_k) in all_data.items():
        mask_c = dates_k <= cutoff_date
        if mask_c.sum() < 10:
            continue
        lh_kc, ll_kc = lh_k[mask_c], ll_k[mask_c]
        pp_lp, pp_idx, pp_dirs = exp28.build_zigzag_raw(lh_kc, ll_kc, t_pool)
        feats, tars, dirs = exp28.build_pool_vectors(pp_lp, pp_dirs, m)
        for feat, tar, dr in zip(feats, tars, dirs):
            pool_rows.append((feat, tar, dr, t_pool))
    return pool_rows


def eval_origin(qv, q_direction, q_last_price, all_data, cutoff_date, m, tpool_grid, h_max):
    pools_by_tpool = {tp: build_pool_at_tpool(cutoff_date, all_data, m, tp) for tp in tpool_grid}
    cur_qv, cur_dir, cum_lr = qv.copy(), q_direction, 0.0
    rows = []
    for h in range(1, h_max + 1):
        candidates_feat, candidates_tar = [], []
        for tp in tpool_grid:
            s_dir = [(f, t) for f, t, d, _ in pools_by_tpool[tp] if d == cur_dir]
            candidates_feat.extend(x[0] for x in s_dir)
            candidates_tar.extend(x[1] for x in s_dir)
        if len(candidates_tar) < m + 2:
            break
        feats_all = np.array(candidates_feat); tars_all = np.array(candidates_tar)
        d = np.linalg.norm(feats_all - cur_qv, axis=1)
        j_best = int(np.argmin(d))
        lr = float(tars_all[j_best])
        cum_lr += lr
        point_price = q_last_price * np.exp(cum_lr)

        top_idx = np.argsort(d)[:TOP_K]
        target_point_lr = np.log(point_price / q_last_price)
        frac_above = float(np.mean(tars_all[top_idx] > target_point_lr))

        rows.append({"h": h, "point": point_price, "frac_above": frac_above,
                    "best_tpool": tpool_grid[0] if False else None})  # tpool идентичность не хранится по кандидату здесь

        cur_qv = np.concatenate([[lr], cur_qv[:-1]])
        cur_dir = -cur_dir
    return rows


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

        h_avail = min(H_MAX, n_full - 1 - p)
        actual_prices = np.exp(full_lp[p + 1: p + 1 + h_avail]) if h_avail > 0 else np.array([])
        actual_dates = dates_t[full_idx[p + 1: p + 1 + h_avail]] if h_avail > 0 else np.array([])

        origin_rows = eval_origin(qv, q_direction, q_last_price, all_data, cutoff_date, M_FIXED, TPOOL_GRID, H_MAX)
        for r in origin_rows:
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
    print(f"=== 28g_tpool_sweep: перебор T_pool={TPOOL_GRID}, m={M_FIXED} фикс., {len(TARGETS)} тикеров ===\n")

    if BIG_TABLE_CSV.exists():
        print(f"Загрузка готовой таблицы {BIG_TABLE_CSV}…")
        big_df = pd.read_csv(BIG_TABLE_CSV)
    else:
        print("Предзагрузка тикеров пула (все 44, для построения пулов на разных T_pool)…")
        all_data = {}
        for tk in exp28.UNIVERSE:
            loaded = exp28.load(tk)
            if loaded is not None:
                all_data[tk] = loaded
        print(f"загружено {len(all_data)} тикеров\n")

        all_rows = []
        for target in TARGETS:
            if target not in all_data:
                continue
            rows = process_ticker(target, all_data)
            all_rows.extend(rows)
            print(f"  {target}: строк={len(rows)} (всего {len(all_rows)})")
        big_df = pd.DataFrame(all_rows)
        big_df.to_csv(BIG_TABLE_CSV, index=False)
        print(f"\nСохранено {len(big_df)} строк в {BIG_TABLE_CSV}")

    print("\n=== accuracy/corr по h (крах исключён), объединённый пул T_pool ===")
    for h in [1, 2, 3]:
        sub = big_df[big_df.h == h]
        acc, corr_val, n = pooled_metrics(sub)
        print(f"  h={h}  acc={acc:.3f} corr={corr_val:+.3f} n={n:6d}")

    valid_targets = sorted(big_df.target.unique())
    print(f"\n=== {N_SPLITS} случайных train/test разбиений тикеров (стабильность, h=1) ===")
    split_results = []
    for seed in range(N_SPLITS):
        rng = np.random.default_rng(seed)
        shuffled = list(valid_targets)
        rng.shuffle(shuffled)
        half = max(3, len(shuffled) // 2)
        test_t = shuffled[half:]
        test_df = big_df[(big_df.target.isin(test_t)) & (big_df.h == 1)]
        acc, corr_val, n = pooled_metrics(test_df)
        print(f"  seed={seed}: TEST acc={acc:.3f} corr={corr_val:+.3f} (n={n})")
        split_results.append({"seed": seed, "acc": acc, "corr_val": corr_val, "n": n})

    split_df = pd.DataFrame(split_results)
    split_df.to_csv(RESULTS / "tpool_sweep_stability_splits.csv", index=False)
    print(f"\nacc: mean={split_df.acc.mean():.3f} std={split_df.acc.std():.3f}")
    print(f"corr: mean={split_df.corr_val.mean():.3f} std={split_df.corr_val.std():.3f}")


if __name__ == "__main__":
    main()
