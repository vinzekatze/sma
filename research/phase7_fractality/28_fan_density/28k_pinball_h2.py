#!/usr/bin/env python3
"""
28k_pinball_h2.py — качество ПОЛОСЫ неопределённости (не точки) на
двухшаговом прогнозе (H=2, "уход-возврат"), метрика — pinball loss.

Контекст (2026-07-08): в app8 добавлено поле неопределённости (взвешенные
квантили целевых лог-доходностей пула — те же веса, что дают точку S-map/
KNN). Пользователь заметил, что факт чаще попадает в полосу, чем в точку,
и предложил стратегический разворот направления B: полоса — основной
продукт, "траектория" многошагового прогноза не нужна как рамка, фокус на
H=2 (departure leg + return leg). См. память project-phase7-uncertainty-
field-pivot и feedback-uncertainty-field-width-control (последняя — КРИТИЧНО:
оптимизация обязана штрафовать ширину полосы, не только покрытие; pinball
loss — строго честная (proper) метрика, для которой размазывание полосы к
±∞ даёт неограниченный рост штрафа, это structural свойство метрики, не
костыль сверху).

Согласовано с пользователем (2026-07-08):
  - H=2, шаг 2 ветвится от ФАКТИЧЕСКОЙ цены шага 1 (не от точки модели) —
    оценивает качество полосы шага 2 САМОЙ ПО СЕБЕ, без накопления ошибки
    шага 1. В живом app8 так не работает (там факта ещё нет) — это только
    для бэктеста/оценки.
  - 7 целевых тикеров и сетка m×θ/K — те же, что в 28j (переиспользуем для
    сравнимости, экономим время).
  - Baseline для pinball — НЕ persistence-формула зигзага (та даёт только
    точку, не распределение), а "наивная безусловная полоса": квантили
    ТОГО ЖЕ направленно-отфильтрованного пула БЕЗ учёта расстояния до
    запроса (веса=1). Смысл: "какого размера плечи бывают у этого типа
    пивота вообще", без учёта текущего состояния — прямой аналог
    persistence, но для распределения, а не для точки.

Метрика: pinball loss L_q(actual, pred_q) = q·(actual−pred_q) если
actual≥pred_q, иначе (1−q)·(pred_q−actual), в ЛОГ-ДОХОДНОСТЯХ (не в цене —
масштабно-свободно между тикерами). Усредняется по 5 уровням квантилей
[0.1,0.25,0.5,0.75,0.9] → relative_pinball = mean_pinball(модель) /
mean_pinball(наивная полоса). Плюс "сырое" эмпирическое покрытие
(доля факта внутри [q10,q90] и [q25,q75]) — для проверки на глаз, что
полоса не выродилась (см. feedback-uncertainty-field-width-control).

Точка (rMAE) здесь НЕ считается — это отдельный вопрос, уже закрыт в 28j
(и там же зафиксирован урок: публиковать per-ticker, не пуловое среднее).

Причинность: не меняется относительно 28_fan_density.py/28j — один и тот
же cutoff_date на origin, полный causal rebuild зигзага и пула на каждом
origin. Единственное отличие в причинности — шаг 2 использует ФАКТИЧЕСКУЮ
цену шага 1, которая при живом прогнозе (origin_offset=0) не существует;
это чисто оценочный (бэктестовый) приём, не влияет на то, что видит модель
в момент прогноза (пул на шаге 2 по-прежнему строится по direction,
известному ДО факта, только сдвиг query-вектора использует факт вместо
точки — сделано специально, чтобы decouple ошибку шага 1 от оценки шага 2).
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

TARGETS = exp28.TARGETS
M_GRID = [2, 3, 5, 8, 13, 20]
THETA_GRID = [0.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.7, 35.0, 50.0]
K_GRID = [1, 2, 3, 5, 8, 13, 20]
Q_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
MIN_HIST = exp28.MIN_HIST

BIG_TABLE_CSV = RESULTS / "28k_big_table.csv"
SUMMARY_CSV = RESULTS / "28k_pinball_summary.csv"


def pinball(actual: float, preds: list[float], qs: list[float]) -> list[float]:
    out = []
    for pred, q in zip(preds, qs):
        diff = actual - pred
        out.append(max(q * diff, (q - 1) * diff))
    return out


def eval_step(qv, cur_dir, pool_rows, min_pool, actual_lr, theta_grid, k_grid, q_levels):
    """Для одного шага (уже известны qv/cur_dir состояния входа) — считает
    наивную полосу + полосу для каждого θ (smap) и K (kavg), их pinball
    против actual_lr, и покрытие [q10,q90]/[q25,q75]."""
    s_dir = [(f, t) for f, t, d in pool_rows if d == cur_dir]
    if len(s_dir) < min_pool:
        return None
    feats = np.array([x[0] for x in s_dir]); tars = np.array([x[1] for x in s_dir])

    rows = []
    ones = np.ones(len(tars))
    naive_band = np.array([float(x) for x in _weighted_quantile(tars, ones, q_levels)])
    naive_pb = pinball(actual_lr, list(naive_band), q_levels)
    rows.append({"method": "naive", "param": np.nan,
                "pinball": float(np.mean(naive_pb)),
                "cov1090": int(naive_band[0] <= actual_lr <= naive_band[4]),
                "cov2575": int(naive_band[1] <= actual_lr <= naive_band[3])})

    for theta in theta_grid:
        w = exp28.smap_weights(qv, feats, theta)
        band = np.array(_weighted_quantile(tars, w, q_levels))
        pb = pinball(actual_lr, list(band), q_levels)
        rows.append({"method": "smap", "param": theta, "pinball": float(np.mean(pb)),
                    "cov1090": int(band[0] <= actual_lr <= band[4]),
                    "cov2575": int(band[1] <= actual_lr <= band[3])})

    for k in k_grid:
        d = np.linalg.norm(feats - qv, axis=1)
        k_eff = min(k, len(d))
        w = np.zeros(len(d)); w[np.argsort(d)[:k_eff]] = 1.0
        band = np.array(_weighted_quantile(tars, w, q_levels))
        pb = pinball(actual_lr, list(band), q_levels)
        rows.append({"method": "kavg", "param": k, "pinball": float(np.mean(pb)),
                    "cov1090": int(band[0] <= actual_lr <= band[4]),
                    "cov2575": int(band[1] <= actual_lr <= band[3])})

    return rows, len(s_dir)


def _weighted_quantile(values, weights, quantiles):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    if w.sum() < 1e-14:
        return [float(np.median(values))] * len(quantiles)
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return [float(x) for x in np.interp(quantiles, cw, v)]


def process_target_m(target, m, all_data):
    lh_t, ll_t, dates_t = all_data[target]
    full_lp, full_idx, full_dirs = exp28.build_zigzag_raw(lh_t, ll_t, exp28.T_QUERY)
    n_full = len(full_lp)
    min_pool = m + 2
    origins = list(range(MIN_HIST, n_full - 3))  # нужны actual на p+1 И p+2

    rows = []
    for p in origins:
        cutoff_bar = int(full_idx[p])
        cutoff_date = str(dates_t[cutoff_bar])
        lh_c, ll_c = lh_t[:cutoff_bar + 1], ll_t[:cutoff_bar + 1]
        q_lp, q_idx, q_dirs = exp28.build_zigzag_raw(lh_c, ll_c, exp28.T_QUERY)
        if len(q_lp) != p + 1:
            continue
        qv0 = exp28.build_query_vector(q_lp, m)
        if qv0 is None:
            continue
        q_direction = int(q_dirs[-1])

        actual_date_1 = str(dates_t[full_idx[p + 1]])[:10]
        actual_date_2 = str(dates_t[full_idx[p + 2]])[:10]
        crash = bool(exp28.CRASH_LO <= actual_date_1 <= exp28.CRASH_HI or
                    exp28.CRASH_LO <= actual_date_2 <= exp28.CRASH_HI)
        if crash:
            continue

        actual_lr_1 = float(full_lp[p + 1] - full_lp[p])
        actual_lr_2 = float(full_lp[p + 2] - full_lp[p + 1])

        pool_rows = exp28.build_causal_pool(cutoff_date, all_data, m)

        res1 = eval_step(qv0, q_direction, pool_rows, min_pool, actual_lr_1,
                         THETA_GRID, K_GRID, Q_LEVELS)
        if res1 is None:
            del pool_rows
            continue
        rows1, n_pool_1 = res1
        for r in rows1:
            r.update({"h": 1, "target": target, "m": m, "p_orig": p, "n_pool": n_pool_1})
        rows.extend(rows1)

        # шаг 2: ветвим от ФАКТИЧЕСКОЙ цены шага 1, не от точки модели
        qv_fact = np.concatenate([[actual_lr_1], qv0[:-1]])
        cur_dir2 = -q_direction
        res2 = eval_step(qv_fact, cur_dir2, pool_rows, min_pool, actual_lr_2,
                         THETA_GRID, K_GRID, Q_LEVELS)
        if res2 is not None:
            rows2, n_pool_2 = res2
            for r in rows2:
                r.update({"h": 2, "target": target, "m": m, "p_orig": p, "n_pool": n_pool_2})
            rows.extend(rows2)

        del pool_rows
    return rows


def main():
    t0 = time.time()
    print("=== 28k_pinball_h2: качество полосы неопределённости, H=2, ветвление от факта ===\n")
    print(f"TARGETS={TARGETS}\nM_GRID={M_GRID}\nQ_LEVELS={Q_LEVELS}\n")

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
                pd.DataFrame(all_rows).to_csv(BIG_TABLE_CSV, index=False)
        big_df = pd.DataFrame(all_rows)
        big_df.to_csv(BIG_TABLE_CSV, index=False)
        print(f"\nСохранено {len(big_df)} строк в {BIG_TABLE_CSV}  [{time.time()-t0:.1f}s]")

    # ── relative pinball (модель/наивная) + покрытие, PER-TICKER ────────────
    print("\n=== per-ticker: лучшая конфигурация по relative pinball (h=1) ===")
    summary_rows = []
    for target in TARGETS:
        naive = big_df[(big_df.target == target) & (big_df.method == "naive") & (big_df.h == 1)]
        naive_pb = naive.pinball.mean()
        for m in M_GRID:
            for method in ["smap", "kavg"]:
                params = THETA_GRID if method == "smap" else K_GRID
                for param in params:
                    for h in [1, 2]:
                        sub = big_df[(big_df.target == target) & (big_df.m == m) &
                                     (big_df.method == method) & (big_df.param == param) &
                                     (big_df.h == h)]
                        if len(sub) < 10:
                            continue
                        naive_h = big_df[(big_df.target == target) & (big_df.method == "naive") &
                                         (big_df.h == h)].pinball.mean()
                        summary_rows.append({
                            "target": target, "m": m, "method": method, "param": param, "h": h,
                            "pinball": sub.pinball.mean(), "naive_pinball": naive_h,
                            "rel_pinball": sub.pinball.mean() / naive_h,
                            "cov1090": sub.cov1090.mean(), "cov2575": sub.cov2575.mean(),
                            "n": len(sub),
                        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    for target in TARGETS:
        for h in [1, 2]:
            sub = summary_df[(summary_df.target == target) & (summary_df.h == h)]
            if not len(sub):
                continue
            best = sub.loc[sub.rel_pinball.idxmin()]
            print(f"  {target} h={h}: лучшее {best['method']}(param={best['param']:.2f}) m={best['m']:.0f}  "
                  f"rel_pinball={best['rel_pinball']:.4f}  cov1090={best['cov1090']:.2f}  "
                  f"cov2575={best['cov2575']:.2f}  n={best['n']:.0f}")

    print(f"\nВсего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
