"""
Визуализация прогноза на ценовом графике — оптимальные параметры.

Конфигурация:
  Модель:         A_p2  (X = [z_i, y_i, y_{i-1}], ξ=15)
  Первичный ряд:  SBER 1d,  порог 2%
  Аугментация:    SBER 1h,  порог 2%  (лучший результат по sweep)
  Walk-forward:   окно 200 пивотов

График:
  Серый фон    — реальная цена
  Синяя линия  — реальный зигзаг
  Горизонтальные сегменты — предсказанный уровень следующего пивота:
      зелёный — направление предсказано верно
      красный — направление ошибочно
  Интенсивность — точность попадания (тёмный = малая ошибка)
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

_BASE        = os.path.dirname(__file__)
DATA_1D      = os.path.join(_BASE, "../../data/candles/SBER/1d.json")
DATA_1H      = os.path.join(_BASE, "../../prototype/data/candles/SBER/1h.json")
OUT_PATH     = os.path.join(_BASE, "zigzag_optimal_chart.png")

THRESHOLD = 0.02
TRAIN_WIN = 200
HORIZONS  = [1, 2, 3, 4]
N_LAGS    = 2
XI        = 15

# ── данные ────────────────────────────────────────────────────────────────────

def load_df(path):
    with open(path) as f: data = json.load(f)
    df = pd.DataFrame(data)
    df["close"] = pd.to_numeric(df["close"])
    return df

def logtrend_causal(close):
    n = len(close); lc = np.log(np.maximum(close, 1e-10))
    t = np.arange(n, dtype=np.float64); cn = np.arange(1, n+1, dtype=np.float64)
    ct = np.cumsum(t); ct2 = np.cumsum(t**2)
    cy = np.cumsum(lc); cty = np.cumsum(t*lc)
    denom = cn*ct2 - ct**2
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(denom > 0, (cn*cty - ct*cy)/denom, 0.0)
    a = (cy - b*ct)/cn; trend = np.exp(a + b*t); trend[:2] = close[:2]
    return trend

def find_pivots(ratio, thr):
    n = len(ratio); pivots = [0]; direction = 0
    ext_val, ext_idx = ratio[0], 0
    for i in range(1, n):
        v = ratio[i]
        if direction == 0:
            if abs(v - ext_val) >= thr*ext_val:
                direction = 1 if v > ext_val else -1; ext_val, ext_idx = v, i
        elif direction == 1:
            if v > ext_val: ext_val, ext_idx = v, i
            elif (ext_val - v) >= thr*ext_val:
                pivots.append(ext_idx); direction = -1; ext_val, ext_idx = v, i
        else:
            if v < ext_val: ext_val, ext_idx = v, i
            elif (v - ext_val) >= thr*ext_val:
                pivots.append(ext_idx); direction = 1; ext_val, ext_idx = v, i
    return pivots

def build_feature(z, dt, i):
    if i < N_LAGS + 1: return None
    return np.array([z[i], z[i]-z[i-1], z[i-1]-z[i-2]], dtype=np.float64)

def lwr_predict(X_tr, y_tr, x_q):
    N = len(X_tr); xi = min(XI, N - 1)
    mu = X_tr.mean(0); sigma = X_tr.std(0)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)
    X_n = (X_tr - mu)/sigma; x_n = (x_q - mu)/sigma
    dists = np.sqrt(((X_n - x_n)**2).sum(1))
    idx = np.argsort(dists)[:xi]
    bw = dists[idx[-1]] + 1e-10
    w = np.exp(-0.5*(dists[idx]/bw)**2)
    Xw = np.column_stack([np.ones(xi), X_n[idx]]) * np.sqrt(w)[:,None]
    yw = y_tr[idx] * np.sqrt(w)
    coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    return float(coef[0] + coef[1:] @ x_n)

# ── walk-forward A_p2 + 1h ───────────────────────────────────────────────────

def walk_forward(z1d, dt1d, dates1d, z_1h, dt_1h, dates_1h):
    N = len(z1d)
    test_start = TRAIN_WIN + N_LAGS + 2
    test_end   = N - max(HORIZONS)

    preds_a2  = []   # A_p2 + 1h
    preds_m0  = []   # random walk
    actuals_1 = []

    for i in range(test_start, test_end):
        train_s      = i - TRAIN_WIN
        current_date = dates1d[i]

        actuals_1.append(z1d[i+1] if i+1 < N else np.nan)
        preds_m0.append(z1d[i])

        x_q = build_feature(z1d, dt1d, i)
        if x_q is None:
            preds_a2.append(z1d[i]); continue

        # 1d train
        X_list, y_list = [], []
        for j in range(train_s, i):
            xj = build_feature(z1d, dt1d, j)
            if xj is None: continue
            X_list.append(xj)
            y_list.append(z1d[j+1] if j+1 < N else np.nan)

        # 1h aug (causally gated)
        ce = int(np.searchsorted(dates_1h, current_date, side="left"))
        N1h = len(z_1h)
        for k in range(N_LAGS+1, ce):
            xk = build_feature(z_1h, dt_1h, k)
            if xk is None: continue
            X_list.append(xk)
            y_list.append(z_1h[k+1] if k+1 < N1h else np.nan)

        if len(X_list) < XI + 1:
            preds_a2.append(z1d[i]); continue

        X_tr = np.array(X_list)
        y_arr = np.array(y_list)
        mask = ~np.isnan(y_arr)
        if mask.sum() < XI + 1:
            preds_a2.append(z1d[i]); continue

        preds_a2.append(lwr_predict(X_tr[mask], y_arr[mask], x_q))

    return (np.array(preds_a2), np.array(preds_m0),
            np.array(actuals_1), test_start)

# ── сборка шагов для визуализации ────────────────────────────────────────────

def build_steps(close, trend, dates_bars, pivots, z, preds_a2, preds_m0,
                actuals_1, test_start):
    test_end = len(z) - max(HORIZONS)
    steps = []
    for k, i in enumerate(range(test_start, test_end)):
        if k >= len(preds_a2): break
        if i+1 >= len(z): break

        bar_curr = pivots[i]
        bar_next = pivots[i+1]

        z_curr   = z[i]
        z_next   = z[i+1]
        z_pred   = preds_a2[k]
        z_m0     = preds_m0[k]

        price_curr = close[bar_curr]
        price_next = close[bar_next]
        price_pred = z_pred * trend[bar_next]
        price_m0   = z_m0   * trend[bar_next]

        dir_actual = np.sign(z_next - z_curr)
        dir_pred   = np.sign(z_pred - z_curr)
        correct    = (dir_actual == dir_pred)

        actual_move = abs(z_next - z_curr)
        rel_err     = abs(z_pred - z_next) / actual_move if actual_move > 1e-8 else 1.0
        m0_err      = abs(z_m0   - z_next) / actual_move if actual_move > 1e-8 else 1.0
        better_than_m0 = rel_err < m0_err

        steps.append(dict(
            i=i, k=k,
            bar_curr=bar_curr, bar_next=bar_next,
            date_curr=dates_bars[bar_curr], date_next=dates_bars[bar_next],
            price_curr=price_curr, price_next=price_next,
            price_pred=price_pred, price_m0=price_m0,
            correct=correct, rel_err=rel_err, m0_err=m0_err,
            better_than_m0=better_than_m0,
        ))
    return steps

# ── отрисовка одного субплота ─────────────────────────────────────────────────

def draw_panel(ax, close, dates_bars, pivots, steps, bar_lo, bar_hi, title):
    # цена
    sl = slice(bar_lo, bar_hi+1)
    ax.plot(dates_bars[sl], close[sl], color="#dddddd", lw=0.6, zorder=1)

    # реальный зигзаг
    piv_in = [p for p in pivots if bar_lo <= p <= bar_hi]
    if len(piv_in) >= 2:
        ax.plot(dates_bars[piv_in], close[piv_in],
                color="#3a6ea5", lw=1.5, zorder=3)
        ax.scatter(dates_bars[piv_in], close[piv_in],
                   color="#3a6ea5", s=20, zorder=4)

    # горизонтальные сегменты: предсказание следующего пивота
    # Цветовая кодировка по точности относительно M0:
    #   тёмно-зелёный  — значительно лучше M0  (rel_err < 0.5 × m0_err или rel_err < 0.4)
    #   светло-зелёный — немного лучше M0
    #   оранжевый      — немного хуже M0
    #   красный        — значительно хуже M0
    n_better = 0; n_worse = 0
    for s in steps:
        if not (bar_lo <= s["bar_curr"] <= bar_hi): continue
        if not (bar_lo <= s["bar_next"] <= bar_hi): continue

        if s["better_than_m0"]:
            n_better += 1
            # интенсивность: чем меньше ошибка, тем темнее зелёный
            t = min(s["rel_err"], 1.5) / 1.5  # 0=точно, 1=большая ошибка
            g = 0.85 - 0.40 * t               # 0.85 (точно) … 0.45 (хуже)
            color = (0.0, g, 0.15)
        else:
            n_worse += 1
            # интенсивность: насколько хуже M0
            ratio_vs_m0 = min(s["rel_err"] / max(s["m0_err"], 1e-8), 3.0)
            t   = min((ratio_vs_m0 - 1.0) / 2.0, 1.0)  # 0=чуть хуже, 1=намного хуже
            r   = 0.60 + 0.35 * t
            color = (r, 0.10, 0.05)

        lw    = 1.6 if s["better_than_m0"] else 0.9
        alpha = 0.80 if s["better_than_m0"] else 0.55

        ax.hlines(y=s["price_pred"],
                  xmin=s["date_curr"], xmax=s["date_next"],
                  color=color, lw=lw, alpha=alpha, zorder=2)
        ax.scatter([s["date_next"]], [s["price_pred"]],
                   color=color, s=14, alpha=alpha+0.1, zorder=5)

    pct = n_better / max(n_better+n_worse, 1) * 100
    ax.set_title(f"{title}  (лучше M0: {n_better}/{n_better+n_worse} = {pct:.0f}%)",
                 fontsize=9)
    ax.set_ylabel("Цена (руб.)", fontsize=8)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right", fontsize=7)
    ax.grid(alpha=0.18, zorder=0)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Загрузка данных …")
    df_1d = load_df(DATA_1D)
    df_1h = load_df(DATA_1H)

    close = df_1d["close"].values.astype(np.float64)
    trend = logtrend_causal(close)
    ratio = close / trend
    pivots_list = find_pivots(ratio, THRESHOLD)
    pivots = np.array(pivots_list)
    z      = ratio[pivots]
    dt_arr = np.append(np.diff(pivots)[0], np.diff(pivots)).astype(float)
    dates_1d = pd.to_datetime(df_1d["begin"]).dt.normalize().values

    dates_piv_1d = dates_1d[pivots]
    print(f"  1d: {len(close)} баров, {len(pivots)} пивотов")

    # 1h
    close_1h = df_1h["close"].values.astype(np.float64)
    trend_1h = logtrend_causal(close_1h)
    ratio_1h = close_1h / trend_1h
    pivots_1h = np.array(find_pivots(ratio_1h, THRESHOLD))
    z_1h   = ratio_1h[pivots_1h]
    dt_1h  = np.append(np.diff(pivots_1h)[0], np.diff(pivots_1h)).astype(float)
    dates_1h_bars = pd.to_datetime(df_1h["begin"]).dt.normalize().values
    dates_piv_1h  = dates_1h_bars[pivots_1h]
    print(f"  1h: {len(close_1h)} баров, {len(pivots_1h)} пивотов")

    print("Walk-forward A_p2 + 1h …", flush=True)
    preds_a2, preds_m0, actuals_1, test_start = walk_forward(
        z, dt_arr, dates_piv_1d, z_1h, dt_1h, dates_piv_1h)

    mean_y = np.mean(np.abs(np.diff(z)))
    mask   = ~np.isnan(actuals_1)
    rmae_a2 = np.abs(preds_a2[mask] - actuals_1[mask]).mean() / mean_y
    rmae_m0 = np.abs(preds_m0[mask] - actuals_1[mask]).mean() / mean_y
    print(f"  rMAE M0={rmae_m0:.3f}  A_p2+1h={rmae_a2:.3f}")

    dates_bars = pd.to_datetime(df_1d["begin"]).values
    steps = build_steps(close, trend, dates_bars, pivots_list, z,
                        preds_a2, preds_m0, actuals_1, test_start)

    # диапазоны баров
    bar_full_lo = pivots[test_start]
    bar_full_hi = pivots[min(test_start + len(preds_a2) - 1 + 1,
                              len(pivots) - 1)]
    bar_zoom_lo = max(0, bar_full_hi - 300)

    print("Рисую …")
    fig, axes = plt.subplots(2, 1, figsize=(16, 11))
    fig.suptitle(
        f"A_p2 + 1h аугментация (T=2%)  —  SBER 1d  |  rMAE: M0={rmae_m0:.3f}  LWR={rmae_a2:.3f}  "
        f"(−{(rmae_m0-rmae_a2)/rmae_m0*100:.0f}%)",
        fontsize=12, fontweight="bold")

    draw_panel(axes[0], close, dates_bars, pivots_list, steps,
               bar_full_lo, bar_full_hi, "Полный тест-период")
    draw_panel(axes[1], close, dates_bars, pivots_list, steps,
               bar_zoom_lo, bar_full_hi, "Zoom: последние 300 баров")

    # легенда
    legend_handles = [
        mpatches.Patch(color="#3a6ea5", label="Реальный зигзаг"),
        mpatches.Patch(color=(0.0, 0.75, 0.15), label="Прогноз: лучше M0 (тёмнее = точнее)"),
        mpatches.Patch(color=(0.80, 0.10, 0.05), label="Прогноз: хуже M0 (темнее = хуже)"),
        mpatches.Patch(color="white",
                       label="Горизонталь = предсказанный уровень следующего пивота"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, loc="upper left", ncol=2)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    print(f"Сохранено: {OUT_PATH}")

    # ── сводная таблица по годам ────────────────────────────────────────────
    year_stats = {}
    for s in steps:
        yr = pd.Timestamp(s["date_curr"]).year
        if yr not in year_stats:
            year_stats[yr] = {"corr": 0, "total": 0, "err": []}
        year_stats[yr]["total"] += 1
        if s["better_than_m0"]: year_stats[yr]["corr"] += 1
        year_stats[yr]["err"].append(s["rel_err"])

    print("\nСтатистика по годам:")
    print(f"  {'год':>4}  {'всего':>6}  {'лучше_M0%':>10}  {'mean_err':>9}  {'median_err':>11}")
    for yr in sorted(year_stats):
        d = year_stats[yr]
        pct = d["corr"] / d["total"] * 100
        me  = np.mean(d["err"])
        med = np.median(d["err"])
        print(f"  {yr:>4}  {d['total']:>6}  {pct:>9.1f}%  {me:>9.3f}  {med:>11.3f}")

    total_better = sum(d["corr"]  for d in year_stats.values())
    total_steps  = sum(d["total"] for d in year_stats.values())
    all_errs     = [e for d in year_stats.values() for e in d["err"]]
    print(f"\n  Итого: {total_better}/{total_steps} = {total_better/total_steps*100:.1f}% шагов лучше M0")
    print(f"  Mean rel_err: {np.mean(all_errs):.3f}  Median: {np.median(all_errs):.3f}")
    print("Готово.")

if __name__ == "__main__":
    main()
