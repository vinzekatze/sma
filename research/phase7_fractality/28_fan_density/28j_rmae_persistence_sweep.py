#!/usr/bin/env python3
"""
28j_rmae_persistence_sweep.py — честный m×θ/K свип по rMAE с ИСПРАВЛЕННОЙ
persistence-формулой зигзага (feedback_rmae_persistence), + проверка
плотностного сигнала (frac_above, эксп.28) на лучшей по rMAE конфигурации.

Закрывает два пробела эксп.28d-28i (см. README §"Приложение — эксп.28c-28i"):
  1. persistence там была |q_last_price − actual| (цена не меняется) —
     несовместимо с остальной фазой 7 (эксп.17-25), где
     pers[h] = exp(full_lp[p_orig + h − 2]) (плечо того же размера, что
     плечо за 2 шага до текущего — обобщение формулы h=1 на h>1, взято из
     20h_multistep_volume_weighted.py).
  2. m=13 было зафиксировано по свипу, оптимизировавшему accuracy (28e), не
     rMAE; малые m (2-5, как в исходном калибраторе эксп.28) с сеткой θ по
     rMAE не проверялись вообще.

Согласовано с пользователем (2026-07-08):
  - 7 целевых тикеров основного эксп.28 (TARGETS), пул — кросс-тикерный 44
    (UNIVERSE), как везде в фазе 7.
  - M_GRID = [2,3,5,8,13,20] (объединение сетки 28 и 28e).
  - THETA_GRID / K_GRID — БЕЗ ИЗМЕНЕНИЙ, те же значения, что в 28i/28h,
    чтобы результат был прямо сравним с уже посчитанными (некорректными)
    числами после пересчёта persistence.
  - Плотностный сигнал (frac_above/accuracy) пересчитывается на лучшей по
    rMAE точке СРАЗУ в этом же прогоне — frac_above уже считается на каждом
    шаге для каждой конфигурации (как в 28/28h), доп. прохода не нужно.

Экономия по сравнению с раздельными 28i (только θ, 44 тикера) + 28h (только
K, 44 тикера): пул (build_causal_pool, доминирующая статья затрат) строится
ОДИН РАЗ на (target, m, origin) и переиспользуется и для S-map-цепочек, и
для K-усреднения — вместо двух независимых полных проходов.

Причинность: не меняется относительно 28_fan_density.py — один и тот же
cutoff_date на origin, полный causal rebuild зигзага и пула на каждом origin
(не инкрементально).
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("exp28", HERE / "28_fan_density.py")
exp28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp28)

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

TARGETS = exp28.TARGETS                      # 7 тикеров, согласовано
M_GRID = [2, 3, 5, 8, 13, 20]                 # согласовано
THETA_GRID = [0.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.7, 35.0, 50.0]   # = 28i
K_GRID = [1, 2, 3, 5, 8, 13, 20]                                   # = 28h
H_MAX = exp28.H_MAX          # 3
MIN_HIST = exp28.MIN_HIST    # 8
TOP_K_DENSITY = exp28.TOP_K  # 20

BIG_TABLE_CSV = RESULTS / "28j_big_table.csv"
SUMMARY_CSV = RESULTS / "28j_rmae_summary.csv"
DENSITY_CSV = RESULTS / "28j_density_at_best.csv"


def eval_origin_combined(qv0, q_direction, q_last_price, pool_rows, m,
                          full_lp, p_orig, actual_prices, actual_dates,
                          theta_grid, k_grid):
    """Общий пул (pool_rows) переиспользуется между S-map-цепочками (по θ) и
    K-усреднением (по K). Внутри каждой цепочки состояние (cur_qv, cur_dir)
    расходится по шагам h — поэтому цепочки не могут шарить промежуточные
    результаты между собой, только исходный пул на входе h=1."""
    min_pool = m + 2
    out = []

    def pers_price(h):
        idx = p_orig + h - 2
        return float(np.exp(full_lp[idx])) if idx >= 0 else float("nan")

    def actual_and_date(h):
        a = float(actual_prices[h - 1]) if h <= len(actual_prices) else float("nan")
        dt = str(actual_dates[h - 1]) if h <= len(actual_dates) else None
        return a, dt

    for theta in theta_grid:
        cur_qv, cur_dir, cum_lr = qv0.copy(), q_direction, 0.0
        for h in range(1, H_MAX + 1):
            s_dir = [(f, t) for f, t, d in pool_rows if d == cur_dir]
            if len(s_dir) < min_pool:
                break
            feats_h = np.array([x[0] for x in s_dir]); tars_h = np.array([x[1] for x in s_dir])
            lr = exp28.smap_predict(cur_qv, feats_h, tars_h, theta)
            cum_lr += lr
            point_price = q_last_price * np.exp(cum_lr)

            w = exp28.smap_weights(cur_qv, feats_h, theta)
            top_idx = np.argsort(w)[::-1][:TOP_K_DENSITY]
            target_point_lr = np.log(point_price / q_last_price)
            frac_above = float(np.mean(tars_h[top_idx] > target_point_lr))

            actual_h, dt = actual_and_date(h)
            crash_flag = bool(dt and exp28.CRASH_LO <= dt[:10] <= exp28.CRASH_HI)
            out.append({"method": "smap", "param": theta, "h": h,
                        "point": point_price, "frac_above": frac_above,
                        "actual": actual_h, "pers": pers_price(h),
                        "crash_flag": crash_flag})

            cur_qv = np.concatenate([[lr], cur_qv[:-1]])
            cur_dir = -cur_dir

    for k in k_grid:
        cur_qv, cur_dir, cum_lr = qv0.copy(), q_direction, 0.0
        for h in range(1, H_MAX + 1):
            s_dir = [(f, t) for f, t, d in pool_rows if d == cur_dir]
            if len(s_dir) < min_pool:
                break
            feats_h = np.array([x[0] for x in s_dir]); tars_h = np.array([x[1] for x in s_dir])
            d = np.linalg.norm(feats_h - cur_qv, axis=1)
            k_eff = min(k, len(d))
            top_idx_k = np.argsort(d)[:k_eff]
            lr = float(np.mean(tars_h[top_idx_k]))
            cum_lr += lr
            point_price = q_last_price * np.exp(cum_lr)

            dens_idx = np.argsort(d)[:TOP_K_DENSITY]
            target_point_lr = np.log(point_price / q_last_price)
            frac_above = float(np.mean(tars_h[dens_idx] > target_point_lr))

            actual_h, dt = actual_and_date(h)
            crash_flag = bool(dt and exp28.CRASH_LO <= dt[:10] <= exp28.CRASH_HI)
            out.append({"method": "kavg", "param": k, "h": h,
                        "point": point_price, "frac_above": frac_above,
                        "actual": actual_h, "pers": pers_price(h),
                        "crash_flag": crash_flag})

            cur_qv = np.concatenate([[lr], cur_qv[:-1]])
            cur_dir = -cur_dir

    return out


def process_target_m(target, m, all_data):
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
        qv = exp28.build_query_vector(q_lp, m)
        if qv is None:
            continue
        q_direction = int(q_dirs[-1])
        q_last_price = float(np.exp(q_lp[-1]))

        pool_rows = exp28.build_causal_pool(cutoff_date, all_data, m)

        h_avail = min(H_MAX, n_full - 1 - p)
        actual_prices = np.exp(full_lp[p + 1: p + 1 + h_avail]) if h_avail > 0 else np.array([])
        actual_dates = dates_t[full_idx[p + 1: p + 1 + h_avail]] if h_avail > 0 else np.array([])

        origin_rows = eval_origin_combined(qv, q_direction, q_last_price, pool_rows, m,
                                           full_lp, p, actual_prices, actual_dates,
                                           THETA_GRID, K_GRID)
        for r in origin_rows:
            r["target"] = target
            r["m"] = m
            r["p_orig"] = p
        rows.extend(origin_rows)
        del pool_rows
    return rows


def rmae_of(df):
    df = df[~df.crash_flag & np.isfinite(df.actual) & np.isfinite(df.pers)]
    if len(df) < 20:
        return float("nan"), len(df)
    err = np.abs(df.point - df.actual)
    pers_err = np.abs(df.pers - df.actual)
    return float(err.mean() / max(1e-12, pers_err.mean())), len(df)


def density_metrics(df):
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
    t0 = time.time()
    print("=== 28j_rmae_persistence_sweep: m×θ/K по честному rMAE ===\n")
    print(f"TARGETS={TARGETS}")
    print(f"M_GRID={M_GRID}  THETA_GRID={THETA_GRID}  K_GRID={K_GRID}\n")

    if BIG_TABLE_CSV.exists():
        print(f"Загрузка готовой таблицы {BIG_TABLE_CSV}…")
        big_df = pd.read_csv(BIG_TABLE_CSV)
    else:
        print("Предзагрузка тикеров (44, кросс-тикерный пул)…")
        all_data = {}
        for tk in exp28.UNIVERSE:
            loaded = exp28.load(tk)
            if loaded is not None:
                all_data[tk] = loaded
        print(f"загружено {len(all_data)} тикеров\n")

        all_rows = []
        for target in TARGETS:
            for m in M_GRID:
                rows = process_target_m(target, m, all_data)
                all_rows.extend(rows)
                print(f"  {target} m={m}: строк={len(rows)} (всего {len(all_rows)})  "
                      f"[{time.time()-t0:.1f}s]")
                pd.DataFrame(all_rows).to_csv(BIG_TABLE_CSV, index=False)  # online flush
        big_df = pd.DataFrame(all_rows)
        big_df.to_csv(BIG_TABLE_CSV, index=False)
        print(f"\nСохранено {len(big_df)} строк в {BIG_TABLE_CSV}  [{time.time()-t0:.1f}s]")

    # ── rMAE по (m, method, param, h) ──────────────────────────────────────
    print("\n=== rMAE (persistence исправлена) по (m, method, param, h=1) ===")
    summary_rows = []
    for m in M_GRID:
        for method in ["smap", "kavg"]:
            params = THETA_GRID if method == "smap" else K_GRID
            for param in params:
                for h in [1, 2, 3]:
                    sub = big_df[(big_df.m == m) & (big_df.method == method) &
                                 (big_df.param == param) & (big_df.h == h)]
                    r, n = rmae_of(sub)
                    summary_rows.append({"m": m, "method": method, "param": param,
                                         "h": h, "rmae": r, "n": n})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    h1 = summary_df[(summary_df.h == 1) & np.isfinite(summary_df.rmae)]
    print(h1.sort_values("rmae").head(15).to_string(index=False))

    best_row = h1.loc[h1.rmae.idxmin()]
    best_m, best_method, best_param = best_row["m"], best_row["method"], best_row["param"]
    print(f"\nЛучшая конфигурация по rMAE (h=1): m={best_m} method={best_method} "
          f"param={best_param}  rMAE={best_row['rmae']:.4f}")

    print("\n=== та же лучшая конфигурация на h=1,2,3 ===")
    for h in [1, 2, 3]:
        sub = summary_df[(summary_df.m == best_m) & (summary_df.method == best_method) &
                         (summary_df.param == best_param) & (summary_df.h == h)]
        if len(sub):
            print(f"  h={h}: rMAE={sub.iloc[0]['rmae']:.4f}  n={sub.iloc[0]['n']}")

    # ── плотностный сигнал на лучшей по rMAE конфигурации ──────────────────
    print("\n=== Плотностный сигнал (frac_above) на лучшей по rMAE конфигурации ===")
    dens_rows = []
    for h in [1, 2, 3]:
        sub = big_df[(big_df.m == best_m) & (big_df.method == best_method) &
                     (big_df.param == best_param) & (big_df.h == h)]
        acc, corr, n = density_metrics(sub)
        print(f"  h={h}: accuracy={acc:.3f} corr={corr:+.3f} n={n}")
        dens_rows.append({"m": best_m, "method": best_method, "param": best_param,
                          "h": h, "accuracy": acc, "corr": corr, "n": n})
    pd.DataFrame(dens_rows).to_csv(DENSITY_CSV, index=False)

    print("\n=== для сравнения: плотностный сигнал на исходной калибровке эксп.28 "
          "(m=3, θ=25.7 — дефолт SBER эксп.17f, ближайшее в сетке) ===")
    ref_m, ref_theta = 3, 25.7
    for h in [1, 2, 3]:
        sub = big_df[(big_df.m == ref_m) & (big_df.method == "smap") &
                     (np.isclose(big_df.param, ref_theta)) & (big_df.h == h)]
        acc, corr, n = density_metrics(sub)
        r, nr = rmae_of(sub)
        print(f"  h={h}: accuracy={acc:.3f} corr={corr:+.3f} n={n}   rMAE={r:.4f} (n={nr})")

    print(f"\nВсего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
