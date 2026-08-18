#!/usr/bin/env python3
"""
28_fan_density.py — асимметрия взвешенной плотности пула вокруг точки
S-map-прогноза как индикатор направления (эксп.28).

Продолжение эксп.7 сессии (scratch-диагностика, не публиковалась как
формальный эксп.): на 54 origin'ах SBER (T_query=20%, m=3, θ=25.7) доля
взвешенной массы пула ВЫШЕ/НИЖЕ одноточечного прогноза коррелировала с тем,
куда реально пошла цена (accuracy 0.61-0.74, corr +0.30…+0.45 по h=1..8),
и это НЕ объяснялось календарной близостью аналогов к origin'у (медианный
разрыв 906 дней, только 2.1% аналогов — сам SBER).

Упрощение относительно scratch-версии: там "веер" строился как СОБСТВЕННЫЙ
реализованный путь топ-K аналогов (ex-post, не причинно для h>1). Здесь —
проще и полностью причинно: на КАЖДОМ шаге итеративного S-map-прогноза
(пул на шаге h уже причинно переотфильтрован по направлению) считаем
взвешенное распределение НАПРЯМУЮ по pool.targets этого шага (не только
OLS-точку) — frac_above(h) = доля взвешенной массы пула, чей target
подразумевает цену ВЫШЕ точки прогноза. Не нужно отслеживать identity
аналогов и их "будущее" — везде используются только величины, уже
причинно доступные на данном шаге.

Причинность: пул (кто попадает и с каким весом) строится по данным,
обрезанным по дате confirm ≤ cutoff_date origin'а — единственная точка
обрезки на КАЖДОМ origin'е (полный rebuild зигзага, не инкрементальное
обновление). Проверено эмпирически (verify_causality).

Копия чистой (без Streamlit) логики app8.py: build_zigzag_raw,
build_pool_vectors, build_query_vector, smap_predict/weights — без
изменений алгоритма.
"""
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parents[3] / "data" / "candles"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

UNIVERSE = [
    "SBER", "LKOH", "CHMF", "NVTK", "MGNT", "VTBR", "NLMK", "MRKP", "GAZP", "PLZL",
    "ROSN", "TATN", "MTSS", "ALRS", "MOEX", "SNGS", "IRAO", "RUAL", "MAGN", "PHOR",
    "AFLT", "HYDR", "SIBN", "TRNFP", "RTKM",
    "BANE", "BANEP", "MTLR", "MTLRP", "RASP", "VSMO", "KMAZ", "AKRN", "MSNG", "TGKA",
    "UPRO", "PIKK", "MVID", "LSRG", "CBOM", "GCHE", "SVAV", "FESH", "KZOS", "NKNC",
]
TARGETS = ["SBER", "LKOH", "GAZP", "MGNT", "CHMF", "MTSS", "PLZL"]

INTERVAL = "1d"
T_QUERY = 0.20
T_POOL = T_QUERY * 0.8987
H_MAX = 3
MIN_HIST = 8

M_GRID = [2, 3, 4, 5]
THETA_LO, THETA_HI, THETA_TOL = 0.0, 32.0, 0.3
MAX_OUTER = 3
N_ORIGINS_CALIB = 60
TOP_K = 20

CRASH_LO, CRASH_HI = "2022-01-01", "2022-06-01"


def load(ticker, interval=INTERVAL):
    path = DATA / ticker / f"{interval}.json"
    if not path.exists():
        return None
    raw = json.load(open(path))
    if not raw:
        return None
    high = np.array([c["high"] for c in raw], dtype=np.float64)
    low = np.array([c["low"] for c in raw], dtype=np.float64)
    dates = np.array([c["begin"] for c in raw])
    high = np.where(high <= 0, np.nan, high)
    low = np.where(low <= 0, np.nan, low)
    return np.log(high), np.log(low), dates


def build_zigzag_raw(lh, ll, threshold):
    prices, idxs, dirs = [], [], []
    direction = 0
    ext_val = (lh[0] + ll[0]) / 2.0
    n = len(lh)
    for i in range(n):
        h, l = lh[i], ll[i]
        if not (np.isfinite(h) and np.isfinite(l)):
            continue
        if direction == 0:
            if h - ext_val >= threshold:
                direction = 1; ext_val = h
            elif ext_val - l >= threshold:
                direction = -1; ext_val = l
        elif direction == 1:
            if h > ext_val:
                ext_val = h
            elif ext_val - l >= threshold:
                prices.append(ext_val); idxs.append(i); dirs.append(1)
                direction = -1; ext_val = l
        else:
            if l < ext_val:
                ext_val = l
            elif h - ext_val >= threshold:
                prices.append(ext_val); idxs.append(i); dirs.append(-1)
                direction = 1; ext_val = h
    return (np.array(prices), np.array(idxs, dtype=np.int64),
            np.array(dirs, dtype=np.int8))


def build_pool_vectors(lp, dirs, m):
    n = len(lp)
    valid = np.arange(m, n - 1)
    if len(valid) == 0:
        return np.zeros((0, m)), np.zeros(0), np.zeros(0, dtype=np.int8)
    lag_idx = valid[:, None] - np.arange(m)[None, :]
    feat = lp[lag_idx] - lp[lag_idx - 1]
    target = lp[valid + 1] - lp[valid]
    d = dirs[valid]
    finite = np.all(np.isfinite(feat), axis=1) & np.isfinite(target)
    return feat[finite], target[finite], d[finite]


def build_query_vector(lp, m):
    n = len(lp)
    if n < m + 1:
        return None
    lags = np.arange(m)
    return lp[n - 1 - lags] - lp[n - 2 - lags]


def smap_weights(qv, feats, theta):
    d = np.linalg.norm(feats - qv, axis=1)
    md = d.mean()
    if md < 1e-14:
        return np.ones(len(d))
    return np.ones(len(d)) if theta == 0 else np.exp(-theta * d / md)


def smap_predict(qv, feats, tars, theta):
    w = smap_weights(qv, feats, theta)
    sw = np.sqrt(w)
    design = np.column_stack([np.ones(len(tars)), feats])
    coeffs, *_ = np.linalg.lstsq(design * sw[:, None], tars * sw, rcond=None)
    return float(coeffs[0] + coeffs[1:] @ qv)


def golden(f, lo, hi, tol):
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    while abs(b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a); fd = f(d)
    return (a + b) / 2


def build_causal_pool(cutoff_date, all_data, m):
    """Причинно обрезанный пул по ВСЕМ тикерам (включая целевой), одна точка
    обрезки — дата подтверждения origin'а."""
    pool_rows = []
    for tk, (lh_k, ll_k, dates_k) in all_data.items():
        mask_c = dates_k <= cutoff_date
        if mask_c.sum() < 10:
            continue
        lh_kc, ll_kc = lh_k[mask_c], ll_k[mask_c]
        pp_lp, pp_idx, pp_dirs = build_zigzag_raw(lh_kc, ll_kc, T_POOL)
        feats, tars, dirs = build_pool_vectors(pp_lp, pp_dirs, m)
        for feat, tar, dr in zip(feats, tars, dirs):
            pool_rows.append((feat, tar, dr))
    return pool_rows


def run_origin(p_orig, full_lp, full_idx, lh_t, ll_t, dates_t, all_data, m, theta, h_max):
    """Полностью причинный прогон одного origin'а: возвращает список строк
    (h, point, frac_above, actual, crash_flag) или None если пул мал."""
    cutoff_bar = int(full_idx[p_orig])
    cutoff_date = str(dates_t[cutoff_bar])

    lh_c, ll_c = lh_t[:cutoff_bar + 1], ll_t[:cutoff_bar + 1]
    q_lp, q_idx, q_dirs = build_zigzag_raw(lh_c, ll_c, T_QUERY)
    if len(q_lp) != p_orig + 1:
        return None
    qv = build_query_vector(q_lp, m)
    if qv is None:
        return None
    q_direction = int(q_dirs[-1])
    q_last_price = float(np.exp(q_lp[-1]))
    min_pool = m + 2

    pool_rows = build_causal_pool(cutoff_date, all_data, m)

    h_avail_actual = min(h_max, len(full_lp) - 1 - p_orig)
    actual_prices = (np.exp(full_lp[p_orig + 1: p_orig + 1 + h_avail_actual])
                     if h_avail_actual > 0 else np.array([]))
    actual_dates = (dates_t[full_idx[p_orig + 1: p_orig + 1 + h_avail_actual]]
                   if h_avail_actual > 0 else np.array([]))

    cur_qv, cur_dir, cum_lr = qv.copy(), q_direction, 0.0
    rows = []
    for h in range(1, h_max + 1):
        s_dir = [(f, t) for f, t, d in pool_rows if d == cur_dir]
        if len(s_dir) < min_pool:
            break
        feats_h = np.array([x[0] for x in s_dir])
        tars_h = np.array([x[1] for x in s_dir])
        weights_h = smap_weights(cur_qv, feats_h, theta)

        lr = smap_predict(cur_qv, feats_h, tars_h, theta)
        cum_lr += lr
        point_price = q_last_price * np.exp(cum_lr)

        # top-K по весу (ЛОКАЛЬНАЯ квантиль) — полный взвешенный пул размывает
        # сигнал (проверено: на тех же origin'ах SBER full-pool даёт corr~0.04,
        # top-20 даёт corr~0.26-0.30, близко к исходной scratch-диагностике).
        # Всё ещё полностью причинно: top_tars — те же одношаговые targets,
        # уже входящие в причинно обрезанный pool_rows, без ex-post лукапа.
        top_idx = np.argsort(weights_h)[::-1][:TOP_K]
        target_point_lr = np.log(point_price / q_last_price)
        frac_above = float(np.mean(tars_h[top_idx] > target_point_lr))

        actual_h = float(actual_prices[h - 1]) if h <= len(actual_prices) else float("nan")
        actual_date_h = str(actual_dates[h - 1]) if h <= len(actual_dates) else None
        crash_flag = bool(actual_date_h and CRASH_LO <= actual_date_h[:10] <= CRASH_HI)

        rows.append({"h": h, "point": point_price, "frac_above": frac_above,
                     "actual": actual_h, "crash_flag": crash_flag,
                     "n_pool_h": len(s_dir)})

        cur_qv = np.concatenate([[lr], cur_qv[:-1]])
        cur_dir = -cur_dir

    return {"cutoff_date": cutoff_date, "rows": rows}


def calibrate_target(full_lp, full_idx, lh_t, ll_t, dates_t, all_data, origins_calib):
    """Покоординатный спуск (m, θ) минимизирующий rMAE точки h=1 —
    тот же протокол, что эксп.17f/26/27."""
    m, theta = 3, 25.7
    prev = None
    for outer in range(MAX_OUTER):
        def rmae_for(m_try, th_try):
            errs, pers_errs = [], []
            for p in origins_calib:
                res = run_origin(p, full_lp, full_idx, lh_t, ll_t, dates_t, all_data, m_try, th_try, 1)
                if res is None or not res["rows"]:
                    continue
                r = res["rows"][0]
                if not np.isfinite(r["actual"]):
                    continue
                cutoff_bar = int(full_idx[p])
                q_last_price = float(np.exp(full_lp[p]))
                errs.append(abs(r["point"] - r["actual"]))
                pers_errs.append(abs(q_last_price - r["actual"]))
            if len(errs) < 10:
                return float("inf")
            return float(np.mean(errs) / max(1e-12, np.mean(pers_errs)))

        best_m, best_v = m, float("inf")
        for mc in M_GRID:
            v = rmae_for(mc, theta)
            if v < best_v:
                best_v, best_m = v, mc
        m = best_m
        theta = golden(lambda th: rmae_for(m, th), THETA_LO, THETA_HI, THETA_TOL)
        cur = (m, round(theta, 2))
        print(f"    калибровка iter {outer+1}: m={m} θ={theta:.3f} rMAE={best_v:.4f}")
        if cur == prev:
            break
        prev = cur
    return m, theta


def main():
    t0 = time.time()
    print("=== 28_fan_density: асимметрия взвешенной плотности пула ===\n")
    print("Предзагрузка тикеров пула…")
    all_data = {}
    for tk in UNIVERSE:
        loaded = load(tk)
        if loaded is not None:
            all_data[tk] = loaded
    print(f"загружено {len(all_data)} тикеров\n")

    all_rows = []
    for target in TARGETS:
        print(f"--- {target} ---")
        lh_t, ll_t, dates_t = all_data[target]
        full_lp, full_idx, full_dirs = build_zigzag_raw(lh_t, ll_t, T_QUERY)
        n_full = len(full_lp)
        print(f"  пивотов T_query=20%: {n_full}")

        origins = list(range(MIN_HIST, n_full - H_MAX - 1))
        if len(origins) < 15:
            print(f"  недостаточно origin'ов ({len(origins)}), пропуск"); continue

        rng = np.random.default_rng(42)
        origins_calib = list(rng.choice(origins, size=min(N_ORIGINS_CALIB, len(origins)), replace=False))

        m, theta = calibrate_target(full_lp, full_idx, lh_t, ll_t, dates_t, all_data, origins_calib)
        print(f"  -> m={m} θ={theta:.3f}")

        n_ok = 0
        for p in origins:
            res = run_origin(p, full_lp, full_idx, lh_t, ll_t, dates_t, all_data, m, theta, H_MAX)
            if res is None:
                continue
            n_ok += 1
            for r in res["rows"]:
                all_rows.append({"target": target, "m": m, "theta": round(theta, 3),
                                  "p_orig": p, "cutoff_date": res["cutoff_date"][:10],
                                  **r})
        print(f"  origin'ов с результатом: {n_ok}/{len(origins)}  ({time.time()-t0:.1f}s суммарно)\n")

        pd.DataFrame(all_rows).to_csv(RESULTS / "fan_density_raw.csv", index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "fan_density_raw.csv", index=False)
    print(f"Сохранено {len(df)} строк в results/fan_density_raw.csv")

    # ── агрегат по (target, h), с флагом краха и без ──────────────────────────
    summary_rows = []
    for crash_excl in [True, False]:
        sub = df[~df["crash_flag"]] if crash_excl else df
        sub = sub[np.isfinite(sub["actual"])]
        for (target, h), g in sub.groupby(["target", "h"]):
            actual_above = g["actual"] > g["point"]
            signal_above = g["frac_above"] > 0.5
            acc = float((actual_above == signal_above).mean())
            x = g["frac_above"].values - 0.5
            y = actual_above.values.astype(float)
            corr = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-9 and y.std() > 1e-9 else float("nan")
            summary_rows.append({"crash_excluded": crash_excl, "target": target, "h": h,
                                  "n": len(g), "accuracy": round(acc, 3), "corr": round(corr, 3)})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS / "fan_density_summary.csv", index=False)
    print("\n=== Агрегат по (тикер, h), крах исключён ===")
    print(summary_df[summary_df.crash_excluded].to_string(index=False))
    print("\n=== Агрегат по (тикер, h), крах включён ===")
    print(summary_df[~summary_df.crash_excluded].to_string(index=False))

    print(f"\nВсего: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
