#!/usr/bin/env python3
"""
16_val_window_smap.py — Окно валидации (2 известных шага) для выбора (m, θ)
S-map на НЕПРЕРЫВНОЙ (сцепленной) траектории, продолженной на 1 реальный шаг
прогноза вперёд. SBER 1d, T_BIG=4%, T_FRAC=3.4%.

Схема (3 шага итеративной цепочки от общего якоря A = O-2):
  Шаг 1: A → A+1 (=O-1). Вектор запроса строится из РЕАЛЬНЫХ логдоходностей
         T_BIG зигзага до A. Пул причинно обрезан по confirm_date(A) —
         фиксирован для ВСЕЙ цепочки (как в smap_ref.py --steps: пул строится
         один раз в момент якоря, дальше не обновляется).
  Шаг 2: A+1 → A+2 (=O). Вектор запроса СДВИГАЕТСЯ: самый свежий лаг — это
         ПРЕДСКАЗАННАЯ (не реальная!) логдоходность шага 1. Направление
         зигзага чередуется автоматически (структурное свойство, не
         прогнозируется).
  Шаг 3: A+2 → A+3 (=O+1). Реальный прогноз — ПРОДОЛЖЕНИЕ той же цепочки
         (лаг = предсказанная логдоходность шага 2), а НЕ новый расчёт от
         реальных данных в O. Если шаги 1-2 разошлись с реальностью, шаг 3
         наследует это расхождение ("слетели с траектории").

Валидация: (m, θ) выбираются минимизацией ОБЪЕДИНЁННОЙ ошибки шагов 1 и 2
(предсказанная цена цепочки vs реально известные O-1, O) — "максимальное
попадание в обе известные точки". θ ищется golden-section на [0,50] (логика
калибратора эксп.13 — θ требует тонкой подстройки на больших значениях),
m — полный перебор 1..20.

Итоговая метрика — ТОЛЬКО шаг 3 (единственный настоящий прогноз):
  rMAE = |forecast_error| / |persistence_error|
  persistence_error = |exp(lp[i+1]) − exp(lp[i−1])|  (наивный прогноз:
  следующий пивот = предыдущий пивот ТОГО ЖЕ направления, НЕ "без изменений";
  см. feedback_rmae_persistence.md).

Без сравнения с бейзлайном — по явному указанию пользователя.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE.parents[2] / "data" / "candles"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

TICKER, INTERVAL = "SBER", "1d"
T_BIG, T_FRAC = 0.04, 0.034

M_GRID = list(range(1, 21))
THETA_LO, THETA_HI, THETA_TOL = 0.0, 50.0, 0.1
MAX_OUTER = 5
MIN_POOL_BASE = 4
N_STEPS = 3
START_BUFFER = max(M_GRID) + 10  # запас, чтобы A-m >= 0 для любого m из сетки


# ── Данные и зигзаг ───────────────────────────────────────────────────────────

def load_log_candles(ticker, interval):
    path = DATA / ticker / f"{interval}.json"
    with open(path) as f:
        raw = json.load(f)
    lh = np.log(np.array([c["high"] for c in raw], dtype=np.float64))
    ll = np.log(np.array([c["low"] for c in raw], dtype=np.float64))
    dt = np.array([c["begin"] for c in raw])
    return lh, ll, dt


def build_zigzag(lh, ll, dates, thr):
    lp, conf, dirs = [], [], []
    cur_dir = 0
    ext = (lh[0] + ll[0]) / 2.0
    for i in range(len(lh)):
        if cur_dir == 0:
            if lh[i] - ext >= thr:
                cur_dir = 1; ext = lh[i]
            elif ext - ll[i] >= thr:
                cur_dir = -1; ext = ll[i]
        elif cur_dir == 1:
            if lh[i] > ext:
                ext = lh[i]
            elif ext - ll[i] >= thr:
                lp.append(ext); conf.append(dates[i]); dirs.append(1)
                cur_dir = -1; ext = ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= thr:
                lp.append(ext); conf.append(dates[i]); dirs.append(-1)
                cur_dir = 1; ext = lh[i]
    return np.array(lp), np.array(conf), np.array(dirs, dtype=np.int8)


def build_pool_by_m(lh, ll, dates, thr, m_values):
    """Пул T_FRAC событий, построенный один раз для каждого m из сетки."""
    lp, conf, dirs = build_zigzag(lh, ll, dates, thr)
    n = len(lp)
    pool = {}
    for m in m_values:
        vidx = np.arange(m, n - 1)
        feat = np.zeros((len(vidx), m))
        tgt = np.zeros(len(vidx))
        dar = np.zeros(len(vidx), dtype=np.int8)
        for row, j in enumerate(vidx):
            for lag in range(m):
                feat[row, lag] = lp[j - lag] - lp[j - lag - 1]
            tgt[row] = lp[j + 1] - lp[j]
            dar[row] = dirs[j]
        econf = conf[vidx + 1]
        finite = np.all(np.isfinite(feat), axis=1) & np.isfinite(tgt)
        pool[m] = (feat[finite], tgt[finite], dar[finite], econf[finite])
    return pool, n


def golden(func, a, b, tol, max_iter=60):
    phi = (5 ** 0.5 - 1) / 2
    x1 = b - phi * (b - a); x2 = a + phi * (b - a)
    f1 = func(x1); f2 = func(x2)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a); f1 = func(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a); f2 = func(x2)
    return (a + b) / 2


def rmae_of(errors, acts):
    if not errors:
        return None
    return float(np.mean(errors) / np.mean(acts))


# ── Сцепленная (не сбрасываемая) S-map траектория от якоря ───────────────────

def iterate_chain(qlp, qconf, qdirs, pool_by_m, m, theta, anchor, n_steps=N_STEPS):
    """
    Строит n_steps итеративных прогнозов от anchor, каждый следующий шаг
    использует ПРЕДСКАЗАННУЮ логдоходность предыдущего шага как самый
    свежий лаг вектора запроса (см. smap_ref.py --steps). Пул причинно
    обрезан ОДИН раз по confirm_date(anchor) и не обновляется по ходу цепочки.

    Возвращает список словарей по шагам или None, если данных/пула не хватило.
    """
    pf, pt, pdir, peconf = pool_by_m[m]
    if anchor - m < 0 or anchor - 1 < 0 or anchor + n_steps >= len(qlp):
        return None

    causal = peconf < qconf[anchor]
    pfc, ptc, pdc = pf[causal], pt[causal], pdir[causal]
    min_pool = max(m + 2, MIN_POOL_BASE)

    vec = np.array([qlp[anchor - lag] - qlp[anchor - lag - 1] for lag in range(m)])
    direction = int(qdirs[anchor])
    cum_lr = 0.0
    steps = []

    for step in range(1, n_steps + 1):
        dmask = pdc == direction
        if dmask.sum() < min_pool:
            return None
        pfd, ptd = pfc[dmask], ptc[dmask]
        dists = np.linalg.norm(pfd - vec, axis=1)
        mean_d = dists.mean()

        if mean_d < 1e-14:
            lr = float(ptd.mean())
        else:
            w = np.ones(len(ptd)) if theta == 0 else np.exp(-theta * dists / mean_d)
            sw = np.sqrt(w)
            A = np.column_stack([np.ones(len(ptd)), pfd]) * sw[:, None]
            b = ptd * sw
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
            lr = float(c[0] + c[1:] @ vec)

        cum_lr += lr
        target_idx = anchor + step
        pred_price = float(np.exp(qlp[anchor] + cum_lr))
        actual_price = float(np.exp(qlp[target_idx]))
        pers_idx = target_idx - 2
        pers_price = float(np.exp(qlp[pers_idx])) if pers_idx >= 0 else None
        pers_err = abs(actual_price - pers_price) if pers_price is not None else None

        steps.append({
            "step": step, "lr": lr, "pred_price": pred_price,
            "actual_price": actual_price, "persistence_price": pers_price,
            "error": abs(pred_price - actual_price), "persistence_error": pers_err,
        })

        vec = np.concatenate([[lr], vec[:-1]])
        direction = -direction

    return steps


def validation_objective(qlp, qconf, qdirs, pool_by_m, m, theta, anchor):
    """rMAE цепочки на шагах 1-2 (единственные уже известные точки)."""
    chain = iterate_chain(qlp, qconf, qdirs, pool_by_m, m, theta, anchor, n_steps=2)
    if chain is None:
        return None
    errs = [s["error"] for s in chain]
    acts = [s["persistence_error"] for s in chain]
    return rmae_of(errs, acts)


# ── Калибровка по окну валидации (логика калибратора эксп.13) ────────────────

def calibrate_at_anchor(qlp, qconf, qdirs, pool_by_m, anchor):
    m, theta = 2, 2.0
    prev = None
    for _ in range(MAX_OUTER):
        best_m, best_v = None, float("inf")
        for mc in M_GRID:
            v = validation_objective(qlp, qconf, qdirs, pool_by_m, mc, theta, anchor)
            if v is not None and v < best_v:
                best_v, best_m = v, mc
        if best_m is None:
            return None, None, None
        m = best_m

        def theta_obj(th, m=m):
            v = validation_objective(qlp, qconf, qdirs, pool_by_m, m, th, anchor)
            return v if v is not None else 1e9

        theta = golden(theta_obj, THETA_LO, THETA_HI, THETA_TOL)

        cur = (m, round(theta, 2))
        if cur == prev:
            break
        prev = cur

    val_rmae = validation_objective(qlp, qconf, qdirs, pool_by_m, m, theta, anchor)
    return m, theta, val_rmae


# ── Walk-forward ──────────────────────────────────────────────────────────────

def run(origins, qlp, qconf, qdirs, pool_by_m):
    rows = []
    for O in origins:
        anchor = O - 2
        m, theta, val_rmae = calibrate_at_anchor(qlp, qconf, qdirs, pool_by_m, anchor)
        if m is None:
            continue

        chain = iterate_chain(qlp, qconf, qdirs, pool_by_m, m, theta, anchor, n_steps=3)
        if chain is None:
            continue
        s1, s2, s3 = chain
        if s3["persistence_error"] is None or s3["persistence_error"] < 1e-14:
            continue
        fc_rmae = s3["error"] / s3["persistence_error"]

        rows.append({
            "origin": int(O),
            "anchor": int(anchor),
            "date": str(qconf[O])[:19],
            "m": m,
            "theta": round(theta, 3),
            "val_rmae": round(val_rmae, 4) if val_rmae is not None else None,
            "step1_pred": round(s1["pred_price"], 4),
            "step1_actual": round(s1["actual_price"], 4),
            "step2_pred": round(s2["pred_price"], 4),
            "step2_actual": round(s2["actual_price"], 4),
            "predicted_price": round(s3["pred_price"], 4),
            "actual_price": round(s3["actual_price"], 4),
            "persistence_price": round(s3["persistence_price"], 4),
            "forecast_error": round(s3["error"], 4),
            "persistence_error": round(s3["persistence_error"], 4),
            "rmae": round(fc_rmae, 4),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Тестовый прогон на подвыборке origin")
    parser.add_argument("--n-test", type=int, default=40)
    args = parser.parse_args()

    print(f"=== ЭКСП.16: ОКНО ВАЛИДАЦИИ (2 шага) + СЦЕПЛЕННЫЙ ПРОГНОЗ (шаг 3) "
          f"({'ТЕСТ' if args.test else 'ПОЛНЫЙ ПРОГОН'}) ===")
    print(f"{TICKER} {INTERVAL}  T_BIG={T_BIG*100:.1f}%  T_FRAC={T_FRAC*100:.1f}%  "
          f"m∈[{M_GRID[0]},{M_GRID[-1]}]  θ∈[{THETA_LO},{THETA_HI}] (golden, tol={THETA_TOL})")

    lh, ll, dates = load_log_candles(TICKER, INTERVAL)
    print(f"Свечей: {len(dates)}  ({str(dates[0])[:10]} … {str(dates[-1])[:10]})")

    qlp, qconf, qdirs = build_zigzag(lh, ll, dates, T_BIG)
    print(f"T_BIG пивотов: {len(qlp)}")

    t0 = time.time()
    pool_by_m, n_frac = build_pool_by_m(lh, ll, dates, T_FRAC, M_GRID)
    print(f"T_FRAC пивотов: {n_frac}  (пул построен за {time.time()-t0:.1f}s)")

    end = len(qlp) - N_STEPS - 1  # O такой, что anchor+3 = O+1 существует
    all_origins = list(range(START_BUFFER, end))
    print(f"Диапазон origin: [{START_BUFFER}, {end})  всего {len(all_origins)}")

    if args.test:
        idx = np.linspace(0, len(all_origins) - 1, args.n_test).astype(int)
        origins = [all_origins[i] for i in sorted(set(idx))]
        print(f"Тестовая подвыборка: {len(origins)} origin (равномерно по всему периоду)")
    else:
        origins = all_origins

    t0 = time.time()
    df = run(origins, qlp, qconf, qdirs, pool_by_m)
    elapsed = time.time() - t0
    print(f"Обработано origin: {len(df)}/{len(origins)}  за {elapsed:.1f}s "
          f"({elapsed/max(len(origins),1)*1000:.1f} мс/origin)")

    if len(df) == 0:
        print("Нет ни одного валидного origin — проверить MIN_POOL / диапазон.")
        return

    tag = "test" if args.test else "full"
    out_csv = RESULTS / f"val_window_smap_{TICKER}_{INTERVAL}_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Сохранено: {out_csv}")

    med_rmae = df["rmae"].median()
    mean_rmae = df["rmae"].mean()
    print(f"rMAE (шаг 3, реальный прогноз): median={med_rmae:.4f}  mean={mean_rmae:.4f}  n={len(df)}")
    print(f"m: median={df['m'].median()}  диапазон [{df['m'].min()},{df['m'].max()}]")
    print(f"θ: median={df['theta'].median():.2f}  диапазон [{df['theta'].min():.2f},{df['theta'].max():.2f}]")

    # ── График: rMAE по времени + предсказанное vs фактическое ───────────────
    df["date_dt"] = pd.to_datetime(df["date"])
    df_sorted = df.sort_values("date_dt")
    roll = df_sorted["rmae"].rolling(window=min(30, max(3, len(df_sorted)//3)), min_periods=3).median()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].scatter(df_sorted["date_dt"], df_sorted["rmae"], s=10, alpha=0.4, label="rMAE (шаг 3)")
    axes[0].plot(df_sorted["date_dt"], roll, color="red", lw=1.5, label="скользящая медиана")
    axes[0].axhline(1.0, color="gray", ls="--", lw=1, label="persistence (rMAE=1)")
    axes[0].set_ylabel("rMAE")
    axes[0].set_title(f"{TICKER} {INTERVAL}  T_BIG={T_BIG*100:.0f}%  T_FRAC={T_FRAC*100:.1f}%  "
                       f"сцепленный 3-шаговый прогноз ({tag}, n={len(df)})")
    axes[0].legend(fontsize=8)

    axes[1].plot(df_sorted["date_dt"], df_sorted["actual_price"], label="фактическая цена (шаг 3)", lw=1)
    axes[1].plot(df_sorted["date_dt"], df_sorted["predicted_price"], label="прогноз (сцепленный)", lw=1, alpha=0.8)
    axes[1].plot(df_sorted["date_dt"], df_sorted["persistence_price"], label="persistence (наив.)", lw=1, ls="--")
    axes[1].set_ylabel("цена")
    axes[1].legend(fontsize=8)

    axes[2].plot(df_sorted["date_dt"], df_sorted["m"], label="m (выбран)", lw=1)
    ax2b = axes[2].twinx()
    ax2b.plot(df_sorted["date_dt"], df_sorted["theta"], color="orange", label="θ (выбран)", lw=1)
    axes[2].set_ylabel("m")
    ax2b.set_ylabel("θ")
    axes[2].set_xlabel("дата")
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")

    plt.tight_layout()
    fig_path = FIGURES / f"val_window_smap_{TICKER}_{INTERVAL}_{tag}.png"
    plt.savefig(fig_path, dpi=110)
    print(f"График: {fig_path}")

    # ── Примеры прогнозов (вся цепочка: шаг1, шаг2, шаг3) ─────────────────────
    examples = pd.concat([
        df_sorted.head(3),
        df_sorted.iloc[[len(df_sorted)//2]] if len(df_sorted) > 6 else df_sorted.iloc[[]],
        df_sorted.nsmallest(3, "rmae"),
        df_sorted.nlargest(3, "rmae"),
    ]).drop_duplicates(subset="origin")
    ex_path = RESULTS / f"val_window_smap_{TICKER}_{INTERVAL}_{tag}_examples.csv"
    examples.to_csv(ex_path, index=False)
    print(f"\nПримеры прогнозов ({ex_path}):")
    print(examples[["date", "m", "theta", "val_rmae",
                     "step1_pred", "step1_actual", "step2_pred", "step2_actual",
                     "predicted_price", "actual_price", "persistence_price", "rmae"]].to_string(index=False))


if __name__ == "__main__":
    main()
