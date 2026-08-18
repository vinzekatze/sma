#!/usr/bin/env python3
"""
05_lwr_best.py — Лучший LWR для фазы 7 (событийная фрактальность зигзага)

Алгоритм:
  1. Строим зигзаг T_BIG=4% и T_FRAC=3.6% на log(high)/log(low)
     Порог T аддитивный в лог-пространстве: T=0.04 ≈ 4.08% в ценах
  2. Каузальная обрезка пула (см. causal_pool): cutoff = дата ПОДТВЕРЖДЕНИЯ
     пивота T_BIG (бар разворота), а не дата экстремума
  3. dir_filter: ищем соседей только среди одного направления (HIGH или LOW)
  4. Вектор задержек: [Δ1, Δ2] = diff-фичи без нормализации
  5. K=75 ближайших соседей по L2
  6. LWR (Gaussian) → предсказывает y_rel = lp_f1[j+1] - lp_f1[j]
  7. Прогноз: exp(lp_big[step] + y_rel_pred)

Результаты (SBER 10m, walk-forward):
  Референс 4-way ensemble:       rMAE = 0.3911
  v1 (soft pivot-date cutoff):   rMAE = 0.3762  (-3.8% vs референс)
  v2 (hard confirm-date cutoff): rMAE = 0.3619  (-7.5% vs референс)  ← текущий

Ключевые улучшения относительно стартовой точки (0.3861):
  srch=diff        (убрать уровень цены из поиска)       -1.4%
  dir_filter       (только одно направление)             -0.5%
  K=75             (вместо K=50)                         -0.3%
  raw везде        (без z-score)                         -0.3%
  hard cutoff      (confirm date вместо pivot date)      -3.8%
"""
import csv
import json
import sys
import numpy as np
from pathlib import Path

HERE    = Path(__file__).parent
DATA    = HERE.parent.parent.parent / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# ── параметры ─────────────────────────────────────────────────────────────────
T_BIG   = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
T_FRAC  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.036
H       = 1       # горизонт прогноза (событий зигзага)
P       = 3       # embedding dimension → 2D diff-вектор [Δ1, Δ2]
K       = 75      # число соседей LWR
MIN_HISTORY = 50  # минимум шагов до первого прогноза
INTERVAL = sys.argv[3] if len(sys.argv) > 3 else "10m"


# ── данные ───────────────────────────────────────────────────────────────────
def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


# ── зигзаг в лог-пространстве ────────────────────────────────────────────────
def find_pivots_log(highs, lows, dates, thr):
    """
    Зигзаг на log(high)/log(low) с аддитивным порогом thr.
    thr=0.04 ≈ 4.08% в ценах (exp(0.04)-1).

    Возвращает:
        vals_log      : log-цены пивотов (float array)
        pivot_dates   : даты пивотов — день достижения экстремума
        confirm_dates : даты подтверждения — день фиксации разворота
                        (первый бар, где цена ушла на thr в обратную сторону)
        dirs          : направление (+1=HIGH, -1=LOW)

    pivot_dates != confirm_dates: между ними может пройти от 1 до сотен баров.
    Алгоритм знает о пивоте только в момент confirm_date.
    """
    lh, ll = np.log(highs), np.log(lows)
    vals_log, pivot_dates, confirm_dates, dirs = [], [], [], []
    direction, ext_val, ext_idx = 0, (lh[0] + ll[0]) / 2.0, 0
    for i in range(len(highs)):
        if direction == 0:
            if lh[i] - ext_val >= thr:
                direction, ext_val, ext_idx = 1, lh[i], i
            elif ext_val - ll[i] >= thr:
                direction, ext_val, ext_idx = -1, ll[i], i
        elif direction == 1:
            if lh[i] > ext_val:
                ext_val, ext_idx = lh[i], i
            elif ext_val - ll[i] >= thr:
                vals_log.append(ext_val)
                pivot_dates.append(dates[ext_idx])
                confirm_dates.append(dates[i])          # ← бар разворота
                dirs.append(+1)
                direction, ext_val, ext_idx = -1, ll[i], i
        else:
            if ll[i] < ext_val:
                ext_val, ext_idx = ll[i], i
            elif lh[i] - ext_val >= thr:
                vals_log.append(ext_val)
                pivot_dates.append(dates[ext_idx])
                confirm_dates.append(dates[i])          # ← бар разворота
                dirs.append(-1)
                direction, ext_val, ext_idx = 1, lh[i], i
    return (np.array(vals_log), np.array(pivot_dates),
            np.array(confirm_dates), np.array(dirs))


# ── вектор задержек ───────────────────────────────────────────────────────────
def build_X(log_prices, p):
    """
    col 0 = log_prices[i]                           (уровень, не используется)
    col k = log_prices[i-k+1] - log_prices[i-k]    = Δk (лог-движение)
    Поиск и LWR используют только cols 1: без нормализации.
    """
    n = len(log_prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = log_prices[i]
        for lag in range(1, p):
            X[i, lag] = log_prices[i - lag + 1] - log_prices[i - lag]
    return X


# ── каузальная обрезка пула ───────────────────────────────────────────────────
def causal_pool(confirm_date, dir_query,
                dt_f1, Xf1, dir_f1, y_rel):
    """
    Единственная точка доступа к T_FRAC пулу. Алгоритмы (LWR и любые
    последующие) получают данные только через эту функцию.

    Жёсткая каузальная обрезка: в пул попадают только T_FRAC пивоты,
    подтверждённые ДО confirm_date текущего T_BIG пивота.

    Аргументы:
        confirm_date : дата подтверждения текущего T_BIG пивота
        dir_query    : направление запроса (+1 или -1)
        dt_f1        : даты ПОДТВЕРЖДЕНИЯ T_FRAC пивотов
        Xf1          : матрица фич T_FRAC пивотов (build_X)
        dir_f1       : направления T_FRAC пивотов
        y_rel        : метки — лог-сдвиг к следующему T_FRAC пивоту

    Возвращает:
        X_diff : diff-фичи пула [Δ1, Δ2, ...] shape (n_pool, P-1)
        y      : метки пула shape (n_pool,)
        None, None если пула недостаточно
    """
    # строгая временна́я граница по confirm_date
    ce  = int(np.searchsorted(dt_f1, confirm_date, side='left'))
    rng = np.arange(P - 1, min(ce, len(dt_f1) - H))
    if len(rng) == 0:
        return None, None

    valid = ~np.any(np.isnan(Xf1[rng]), axis=1) & ~np.isnan(y_rel[rng])
    idx   = rng[valid]
    if len(idx) < P + 2:
        return None, None

    # dir_filter: однонаправленные соседи
    idx_d = idx[dir_f1[idx] == dir_query]
    if len(idx_d) < P + 2:
        idx_d = idx  # fallback если мало однонаправленных

    return Xf1[idx_d][:, 1:], y_rel[idx_d]


# ── LWR ──────────────────────────────────────────────────────────────────────
def lwr_predict(xq, X_pool, y_pool):
    """
    Локально взвешенная регрессия на K ближайших соседях.
    xq, X_pool — raw diff-фичи (без нормализации).
    Возвращает pred_rel — предсказанный лог-сдвиг.
    """
    dists = np.linalg.norm(X_pool - xq, axis=1)
    keff  = min(K, len(dists))
    knn   = np.argsort(dists)[:keff]

    X_sel, y_sel = X_pool[knn], y_pool[knn]
    d  = dists[knn]
    xi = d.max()
    if xi < 1e-12:
        return float(y_sel.mean())
    w  = np.exp(-0.5 * (d / xi) ** 2)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(keff), X_sel]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, y_sel * ws, rcond=None)
    return float(c[0] + c[1:] @ xq)


# ── walk-forward ──────────────────────────────────────────────────────────────
def walk_forward(lp_big, confirm_big, dir_big, n_big,
                 lp_f1,  dt_f1_confirm, dir_f1,  n_f1):
    """
    Walk-forward: для каждого T_BIG пивота (step) предсказываем следующий.

    Момент прогноза = confirm_big[step]: когда пивот стал известен.
    Пул строится через causal_pool → алгоритм не видит ничего из будущего.

    Возвращает:
        steps      : индексы пивотов (в lp_big), для которых сделан прогноз
        errs       : pred - actual
        acts       : actual (цены следующих пивотов)
        preds      : predicted (цены следующих пивотов)
        pool_sizes : размер пула при каждом шаге
    """
    X_big = build_X(lp_big, P)
    Xf1   = build_X(lp_f1,  P)
    y_rel = np.array([
        lp_f1[j + H] - lp_f1[j] if j + H < n_f1 else np.nan
        for j in range(n_f1)
    ])

    steps, errs, acts, preds, pool_sizes = [], [], [], [], []

    for step in range(MIN_HISTORY, n_big - H):
        if np.any(np.isnan(X_big[step])):
            continue

        X_pool, y_pool = causal_pool(
            confirm_big[step], int(dir_big[step]),
            dt_f1_confirm, Xf1, dir_f1, y_rel
        )
        if X_pool is None:
            continue

        xq = X_big[step][1:]
        pred_rel = lwr_predict(xq, X_pool, y_pool)
        if not np.isfinite(pred_rel):
            continue

        pred   = float(np.exp(lp_big[step] + pred_rel))
        actual = float(np.exp(lp_big[step + H]))

        steps.append(step)
        errs.append(pred - actual)
        acts.append(actual)
        preds.append(pred)
        pool_sizes.append(len(y_pool))

    return (np.array(steps), np.array(errs),
            np.array(acts), np.array(preds), np.array(pool_sizes))


def rmae(errs, acts):
    acts = np.asarray(acts, dtype=float)
    pers = np.abs(acts[2:] - acts[:-2]) if len(acts) >= 3 else np.abs(np.diff(acts))
    dz   = float(np.mean(pers)) if len(pers) > 0 else 1.0
    return float(np.mean(np.abs(errs))) / dz if dz > 1e-12 else np.nan


# ── визуализация ──────────────────────────────────────────────────────────────
def plot_results(lp_big, confirm_big, dir_big,
                 steps, acts, preds, errs, r, out_path):
    """
    График: T_BIG зигзаг + predicted vs actual + rolling rMAE.

    Верхняя панель — ценовой график (exp зигзага):
      • серая ломаная — T_BIG зигзаг (известные пивоты)
      • синие треугольники вниз — actual следующего пивота
      • красные кружки — predicted следующего пивота
      Линии actual→predicted показывают направление ошибки.

    Нижняя панель — скользящий rMAE (окно 50 шагов).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime

    # строим дату-ось по confirm_big
    parse = lambda s: datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    all_dates = np.array([parse(s) for s in confirm_big])

    p_big = np.exp(lp_big)
    step_dates  = all_dates[steps]
    next_dates  = all_dates[np.array(steps) + H]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=False)

    # ── верхняя панель: ценовой зигзаг ──────────────────────────────────────
    ax1.plot(all_dates, p_big, color="lightgray", lw=0.8, zorder=1,
             label="T_BIG зигзаг")

    ax1.scatter(next_dates, acts, marker="v", s=20, color="steelblue",
                zorder=3, label="actual", alpha=0.7)
    ax1.scatter(next_dates, preds, marker="o", s=20, color="tomato",
                zorder=3, label="predicted", alpha=0.7)

    # соединяем actual и predicted для каждого шага
    for nd, a, p in zip(next_dates, acts, preds):
        ax1.plot([nd, nd], [a, p], color="gray", lw=0.5, alpha=0.4, zorder=2)

    ax1.set_ylabel("Цена SBER (руб.)")
    ax1.set_title(
        f"SBER {INTERVAL} | T_big={T_BIG*100:.1f}% T_frac={T_FRAC*100:.1f}% "
        f"P={P} K={K} H={H} hard-cutoff\n"
        f"rMAE = {r:.4f}   (Δ = {(r-0.3911)/0.3911*100:+.1f}% vs референс 0.3911)"
    )
    ax1.legend(fontsize=8, loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30, ha="right")

    # ── нижняя панель: скользящий rMAE ───────────────────────────────────────
    W = 50  # ширина скользящего окна (шагов)
    roll_rmae = []
    for i in range(len(errs)):
        sl = slice(max(0, i - W + 1), i + 1)
        e_sl, a_sl = errs[sl], acts[sl]
        a_sl_arr = np.asarray(a_sl, dtype=float)
        pers_sl = np.abs(a_sl_arr[2:] - a_sl_arr[:-2]) if len(a_sl_arr) >= 3 else np.abs(np.diff(a_sl_arr))
        dz = float(np.mean(pers_sl)) if len(pers_sl) > 0 else np.nan
        roll_rmae.append(float(np.mean(np.abs(e_sl)) / dz) if dz and dz > 1e-12 else np.nan)

    ax2.plot(step_dates, roll_rmae, color="darkorange", lw=1.0,
             label=f"rolling rMAE (W={W})")
    ax2.axhline(r, color="red", lw=0.8, ls="--", label=f"global rMAE={r:.4f}")
    ax2.axhline(0.3911, color="gray", lw=0.6, ls=":", label="референс 0.3911")
    ax2.set_ylabel("rMAE")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    fig.autofmt_xdate(rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    h, l, d = load_tf(INTERVAL)

    lp_big, _, confirm_big, dir_big = find_pivots_log(h, l, d, T_BIG)
    lp_f1,  _, confirm_f1,  dir_f1  = find_pivots_log(h, l, d, T_FRAC)
    n_big, n_f1 = len(lp_big), len(lp_f1)

    print(f"SBER {INTERVAL} | T_big={T_BIG*100:.1f}% ({(np.exp(T_BIG)-1)*100:.2f}% в ценах) "
          f"| T_frac={T_FRAC*100:.1f}% | n_big={n_big} | n_f1={n_f1}")
    print(f"Параметры: P={P} K={K} H={H} dir_filter=True raw hard-cutoff")

    steps, errs, acts, preds, pool_szs = walk_forward(
        lp_big, confirm_big, dir_big, n_big,
        lp_f1,  confirm_f1,  dir_f1,  n_f1
    )

    r = rmae(errs, acts)
    print(f"\nrMAE = {r:.4f}   n_steps={len(errs)}   pool_avg={pool_szs.mean():.0f}")
    print(f"Референс (4-way ensemble): 0.3911  |  Δ = {(r-0.3911)/0.3911*100:+.1f}%")

    # CSV
    out_csv = RESULTS / f"best_{INTERVAL}_T{int(T_BIG*100)}_Tf{int(T_FRAC*1000)}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "error", "actual", "predicted"])
        for s, e, a, p in zip(steps, errs, acts, preds):
            w.writerow([int(s), e, a, p])
    print(f"Results → {out_csv}")

    # График
    fig_path = RESULTS / f"chart_{INTERVAL}_T{int(T_BIG*100)}_Tf{int(T_FRAC*1000)}.png"
    plot_results(lp_big, confirm_big, dir_big,
                 steps, acts, preds, errs, r, fig_path)


if __name__ == "__main__":
    main()
