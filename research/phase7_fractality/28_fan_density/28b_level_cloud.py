#!/usr/bin/env python3
"""
28b_level_cloud.py — "облако уровней" вместо пошагового прогноза (продолжение
эксп.28, переосмысление после разговора с пользователем).

Идея пользователя (эксп.28 попал в старую парадигму — пошаговое сопоставление
прогноз/факт, хотя ось времени номинальная и порядок событий не гарантирован):
многошаговый S-map-прогноз стоит читать не как последовательность конкретных
будущих событий, а как ОБЛАКО потенциальных уровней. Дополнительно — облако
обогащается за счёт РАЗНЫХ m (разных "масштабов" вложения: короткая история
плеч vs длинная) — пользователь заметил, что модель на некоторых m словно
пытается угадывать крупные события через мелкие, теряя привязку к конкретным
зигзаг-пивотам; объединение уровней с разных m — грубый способ смешать
масштабы без явной событийной иерархии T_frac/T_big.

Конструкция:
  - для набора m (без калибровки θ — фиксировано, чтобы не тратить часы на
    golden-section по каждому m) считаем ОДНУ итеративную S-map-цепочку на
    H=20 шагов (тот же механизм, что в app8: направление чередуется).
  - облако = все m×H прогнозных уровней вместе (не по шагам, одним пулом).
  - cloud_frac_above = доля уровней облака ВЫШЕ цены origin'а.
  - факт: доля РЕАЛЬНЫХ следующих (до H) пивотов, которые оказались выше
    цены origin'а — actual_frac_above. Сравнение "плотность vs плотность",
    не "шаг vs шаг".
  - метрика: корреляция cloud_frac_above vs actual_frac_above по многим
    origin'ам, на каждом из 7 тикеров эксп.28.

Причинность: пул строится по данным, обрезанным по дате origin'а — та же
дисциплина, что везде в фазе 7. "Актуальные" будущие пивоты берутся из
НЕобрезанного ряда только для ex-post сравнения (бэктест), не участвуют в
построении прогноза.
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

TARGETS = ["SBER", "LKOH", "GAZP", "MGNT", "CHMF", "MTSS", "PLZL"]
M_GRID = [2, 3, 5, 8]
THETA = 25.7          # фиксировано (не калибруется — см. докстрок)
H = 20
MIN_HIST = max(M_GRID) + 2
MIN_ACTUAL = 5         # минимум реальных будущих пивотов, чтобы origin считался
ODD_ONLY = True         # только нечётные шаги (противоположный origin'у экстремум)
OUT_RAW = "level_cloud_raw_oddonly.csv" if ODD_ONLY else "level_cloud_raw.csv"
OUT_SUMMARY = "level_cloud_summary_oddonly.csv" if ODD_ONLY else "level_cloud_summary.csv"


def build_causal_pool_multi_m(cutoff_date, all_data, m_grid):
    """Строит СЫРОЙ (не зависящий от m) причинно обрезанный зигзаг один раз на
    тикер, затем для каждого m — дешёвое build_pool_vectors. Возвращает
    {m: [(feat, target, dir), ...]}."""
    raw = {}
    for tk, (lh_k, ll_k, dates_k) in all_data.items():
        mask_c = dates_k <= cutoff_date
        if mask_c.sum() < 10:
            continue
        lh_kc, ll_kc = lh_k[mask_c], ll_k[mask_c]
        pp_lp, pp_idx, pp_dirs = exp28.build_zigzag_raw(lh_kc, ll_kc, exp28.T_POOL)
        raw[tk] = (pp_lp, pp_dirs)

    pools = {m: [] for m in m_grid}
    for m in m_grid:
        for tk, (pp_lp, pp_dirs) in raw.items():
            feats, tars, dirs = exp28.build_pool_vectors(pp_lp, pp_dirs, m)
            for feat, tar, dr in zip(feats, tars, dirs):
                pools[m].append((feat, tar, dr))
    return pools


def iterative_chain(qv0, direction0, q_last_price, pool_rows, theta, h_steps, min_pool):
    """Возвращает список (h, level) — h=1..h_steps (или меньше, если пул иссяк)."""
    cur_qv, cur_dir, cum_lr = qv0.copy(), direction0, 0.0
    levels = []
    for step in range(1, h_steps + 1):
        s_dir = [(f, t) for f, t, d in pool_rows if d == cur_dir]
        if len(s_dir) < min_pool:
            break
        feats = np.array([x[0] for x in s_dir]); tars = np.array([x[1] for x in s_dir])
        lr = exp28.smap_predict(cur_qv, feats, tars, theta)
        cum_lr += lr
        levels.append((step, q_last_price * np.exp(cum_lr)))
        cur_qv = np.concatenate([[lr], cur_qv[:-1]])
        cur_dir = -cur_dir
    return levels


def run_origin_cloud(p_orig, full_lp, full_idx, lh_t, ll_t, dates_t, all_data, m_grid, theta, h, min_actual,
                     odd_only=False):
    cutoff_bar = int(full_idx[p_orig])
    cutoff_date = str(dates_t[cutoff_bar])

    lh_c, ll_c = lh_t[:cutoff_bar + 1], ll_t[:cutoff_bar + 1]
    q_lp, q_idx, q_dirs = exp28.build_zigzag_raw(lh_c, ll_c, exp28.T_QUERY)
    if len(q_lp) != p_orig + 1:
        return None
    q_last_price = float(np.exp(q_lp[-1]))
    q_direction = int(q_dirs[-1])

    h_avail_actual = min(h, len(full_lp) - 1 - p_orig)
    if h_avail_actual < min_actual:
        return None
    actual_prices = np.exp(full_lp[p_orig + 1: p_orig + 1 + h_avail_actual])
    # h=1..h_avail_actual; нечётные h = противоположный origin'у экстремум (сигнал),
    # чётные h = тот же тип экстремума (шум, эксп.28)
    actual_steps = np.arange(1, h_avail_actual + 1)
    if odd_only:
        mask_odd = (actual_steps % 2) == 1
        actual_prices_used = actual_prices[mask_odd]
    else:
        actual_prices_used = actual_prices
    if len(actual_prices_used) < 3:
        return None
    actual_frac_above = float(np.mean(actual_prices_used > q_last_price))

    pools = build_causal_pool_multi_m(cutoff_date, all_data, m_grid)

    cloud = []
    for m in m_grid:
        qv = exp28.build_query_vector(q_lp, m)
        if qv is None:
            continue
        levels = iterative_chain(qv, q_direction, q_last_price, pools[m], theta, h, m + 2)
        if odd_only:
            levels = [(s, lv) for s, lv in levels if s % 2 == 1]
        cloud.extend(lv for s, lv in levels)

    if len(cloud) < 5:
        return None
    cloud = np.array(cloud)
    cloud_frac_above = float(np.mean(cloud > q_last_price))

    return {"cutoff_date": cutoff_date[:10], "cloud_frac_above": cloud_frac_above,
            "actual_frac_above": actual_frac_above, "n_cloud": len(cloud),
            "n_actual": h_avail_actual}


def main():
    print("=== 28b_level_cloud: облако уровней (multi-m) vs плотность факта ===\n")
    print("Предзагрузка тикеров…")
    all_data = {}
    for tk in exp28.UNIVERSE:
        loaded = exp28.load(tk)
        if loaded is not None:
            all_data[tk] = loaded
    print(f"загружено {len(all_data)} тикеров\n")

    rows = []
    for target in TARGETS:
        lh_t, ll_t, dates_t = all_data[target]
        full_lp, full_idx, full_dirs = exp28.build_zigzag_raw(lh_t, ll_t, exp28.T_QUERY)
        n_full = len(full_lp)
        origins = list(range(MIN_HIST, n_full - 1))
        n_ok = 0
        for p in origins:
            res = run_origin_cloud(p, full_lp, full_idx, lh_t, ll_t, dates_t, all_data,
                                   M_GRID, THETA, H, MIN_ACTUAL, odd_only=ODD_ONLY)
            if res is None:
                continue
            n_ok += 1
            rows.append({"target": target, "p_orig": p, **res})
        print(f"{target}: origin'ов с результатом {n_ok}/{len(origins)}")
        pd.DataFrame(rows).to_csv(RESULTS / OUT_RAW, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / OUT_RAW, index=False)
    print(f"\nСохранено {len(df)} строк в results/{OUT_RAW}\n")

    print("=== Корреляция cloud_frac_above vs actual_frac_above, по тикерам ===")
    summary = []
    for target, g in df.groupby("target"):
        x = g["cloud_frac_above"].values - 0.5
        y = g["actual_frac_above"].values - 0.5
        corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else float("nan")
        acc = float(np.mean((x > 0) == (y > 0)))
        print(f"  {target:6s} n={len(g):3d}  corr={corr:+.3f}  "
              f"accuracy(знак)={acc:.3f}  mean_cloud_frac={g['cloud_frac_above'].mean():.3f}")
        summary.append({"target": target, "n": len(g), "corr": round(corr, 3), "accuracy": round(acc, 3)})

    all_x = df["cloud_frac_above"].values - 0.5
    all_y = df["actual_frac_above"].values - 0.5
    corr_all = float(np.corrcoef(all_x, all_y)[0, 1])
    acc_all = float(np.mean((all_x > 0) == (all_y > 0)))
    print(f"\nПул по всем тикерам: n={len(df)}  corr={corr_all:+.3f}  accuracy(знак)={acc_all:.3f}")
    pd.DataFrame(summary).to_csv(RESULTS / OUT_SUMMARY, index=False)


if __name__ == "__main__":
    main()
