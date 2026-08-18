"""
LWR-прогноз уровней зигзага в событийном времени.

Семейства признаков:
  A: X = [z_i, y(sign)·p]  — базовая лучшая конфигурация
  E: X = [z_i, |y|·p]      — модуль вместо знака
  F: X = [z_i, |y|·p], цель=|Δz|, реконструкция со знаком

Аугментация: к 1d-обучающей выборке добавляются пивоты SBER 1h.
CAUSALITY GATE: в каждом шаге walk-forward первым делом отсекаются
все 1h-пивоты с датой >= даты текущего 1d тест-пивота.
Алгоритм физически не получает 1h-данных за точкой прогнозирования.

Walk-forward: скользящее окно TRAIN_WIN=200 пивотов (1d) + все
каузальные 1h пивоты (без ограничения окна).
Метрика: rMAE = MAE / mean(|y|)
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

_BASE       = os.path.dirname(__file__)
DATA_PATH    = os.path.join(_BASE, "../../data/candles/SBER/1d.json")
DATA_PATH_1H = os.path.join(_BASE, "../../prototype/data/candles/SBER/1h.json")
DATA_PATH_10M = os.path.join(_BASE, "../../prototype/data/candles/SBER/10m.json")
OUT_DIR      = _BASE

THRESHOLD = 0.02
TRAIN_WIN = 200
HORIZONS  = [1, 2, 3, 4]
P_GRID    = [2, 4, 6]

# ── данные ────────────────────────────────────────────────────────────────────

def load_data(path):
    with open(path) as f:
        candles = json.load(f)
    df = pd.DataFrame(candles)
    df["close"] = pd.to_numeric(df["close"])
    return df

def load_aux(path, threshold):
    """
    Загружает вспомогательный ряд (1h), вычисляет зигзаг и возвращает:
      z_aux      — уровни пивотов (ratio)
      dt_aux     — длительности качаний в барах
      dates_aux  — даты (date-only, np.datetime64[D]) каждого пивота
    Нормализация — та же logtrend_causal, что и для 1d.
    """
    df = load_data(path)
    close = df["close"].values.astype(np.float64)
    trend = logtrend_causal(close)
    ratio = close / trend
    pivots = find_pivots(ratio, threshold)
    piv_arr = np.array(pivots)
    z_aux   = ratio[piv_arr]
    dt_raw  = np.diff(piv_arr).astype(float)
    dt_aux  = np.append(dt_raw[0], dt_raw)
    # даты до уровня дня (нормировка убирает время)
    dates_aux = pd.to_datetime(df["begin"]).dt.normalize().values[piv_arr]
    return z_aux, dt_aux, dates_aux

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
    n = len(ratio); pivots = [0]; direction = 0; ext_val, ext_idx = ratio[0], 0
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

# ── признаки ──────────────────────────────────────────────────────────────────

def build_feature(z, dt, i, n_lags, use_dt, include_level=True, abs_y=False):
    """
    Вектор состояния в пивоте i.
      include_level=True:  [z_i, y_i, ...]
      include_level=False: [y_i, ...]
      abs_y=True:          y заменяются на |y|
      use_dt=True:         + [log(Δt_i), ...]
    Требует i >= n_lags + 1.
    """
    if i < n_lags + 1:
        return None
    feats = ([z[i]] if include_level else [])
    for k in range(n_lags):
        y = z[i - k] - z[i - k - 1]
        feats.append(abs(y) if abs_y else y)
    if use_dt:
        for k in range(n_lags):
            feats.append(np.log(max(float(dt[i - k]), 1.0)))
    return np.array(feats, dtype=np.float64)

def feat_dim(n_lags, use_dt, include_level=True):
    return (1 if include_level else 0) + n_lags + (n_lags if use_dt else 0)

def xi_for(n_lags, use_dt, include_level=True):
    return max(15, 3 * (feat_dim(n_lags, use_dt, include_level) + 1))

# ── LWR ───────────────────────────────────────────────────────────────────────

def lwr_predict(X_tr, y_tr, x_q, xi):
    """
    Локальная взвешенная регрессия (WLS + Gaussian kernel).
    X_tr: (N, d), y_tr: (N,), x_q: (d,)
    """
    N = len(X_tr)
    xi = min(xi, N - 1)

    # нормировка по обучающей выборке
    mu    = X_tr.mean(0)
    sigma = X_tr.std(0)
    sigma = np.where(sigma < 1e-10, 1.0, sigma)

    X_n = (X_tr - mu) / sigma
    x_n = (x_q  - mu) / sigma

    dists  = np.sqrt(((X_n - x_n) ** 2).sum(1))
    idx    = np.argsort(dists)[:xi]
    bw     = dists[idx[-1]] + 1e-10
    w      = np.exp(-0.5 * (dists[idx] / bw) ** 2)
    w_sqrt = np.sqrt(w)

    # WLS: √W · [1 | X] β = √W · y
    Xw = np.column_stack([np.ones(xi), X_n[idx]]) * w_sqrt[:, None]
    yw = y_tr[idx] * w_sqrt
    coef, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)

    return float(coef[0] + coef[1:] @ x_n)

# ── walk-forward ──────────────────────────────────────────────────────────────

# (name, n_lags, use_dt, include_level, delta_target, abs_y)
# A: [z, y(sign)…], цель=z               — базовая лучшая
# E: [z, |y|…],     цель=z               — |y| признаки, цель прямая
# F: [z, |y|…],     цель=|Δz|→ẑ=z-sign·|ŷ| — амплитуда + вшитое направление (h=1)
CONFIGS = []
for _p in P_GRID:
    CONFIGS.append((f"A_p{_p}", _p, False, True, False, False))
    CONFIGS.append((f"E_p{_p}", _p, False, True, False, True))
CONFIGS.append(("F_p2", 2, False, True, True, True))

def walk_forward(z, dt, dates_piv, aux_series=None):
    """
    z, dt, dates_piv  — 1d ряд (первичный)
    aux_series        — список (z_s, dt_s, dates_s) вспомогательных рядов,
                        каждый обрабатывается независимо (пивоты не смешиваются).

    CAUSALITY GATE: для каждого шага i и каждого вспомогательного ряда s —
    causal_end_s = searchsorted(dates_s, current_date, side="left").
    Алгоритм не получает ни одного пивота с датой >= даты текущего 1d-пивота.
    dates_s должны быть отсортированы по возрастанию.
    """
    N = len(z)
    n_lags_max  = max(P_GRID)
    test_start  = TRAIN_WIN + n_lags_max + 2
    test_end    = N - max(HORIZONS)

    use_aux = bool(aux_series)

    all_names = ["M0", "M3"] + [name for name, *_ in CONFIGS]
    results   = {name: {h: [] for h in HORIZONS} for name in all_names}
    actuals   = {h: [] for h in HORIZONS}
    xi_cache  = {name: xi_for(p, ud, il)
                 for name, p, ud, il, *_ in CONFIGS}

    for i in range(test_start, test_end):
        train_s = i - TRAIN_WIN

        current_date = dates_piv[i]   # дата текущего 1d-пивота (causality boundary)

        for h in HORIZONS:
            actuals[h].append(z[i + h] if i + h < N else np.nan)

        # ── M0: random walk ────────────────────────────────────────────────
        for h in HORIZONS:
            results["M0"][h].append(z[i])

        # ── M3: OLS 6D (baseline) ──────────────────────────────────────────
        yi   = z[i]   - z[i-1]
        yim1 = z[i-1] - z[i-2]
        x_m3 = np.array([z[i], z[i-1], abs(yi), abs(yim1),
                         float(dt[i]), float(dt[i-1])])
        for h in HORIZONS:
            rows_x, rows_y = [], []
            for j in range(max(train_s, 2), i):
                yj = z[j] - z[j-1]; yjm = z[j-1] - z[j-2]
                rows_x.append([z[j], z[j-1], abs(yj), abs(yjm),
                                float(dt[j]), float(dt[j-1])])
                rows_y.append(z[j + h] if j + h < N else np.nan)
            X_m3 = np.array(rows_x); y_m3 = np.array(rows_y)
            mask = ~np.isnan(y_m3)
            if mask.sum() > 10:
                Xw = np.column_stack([np.ones(mask.sum()), X_m3[mask]])
                coef, _, _, _ = np.linalg.lstsq(Xw, y_m3[mask], rcond=None)
                pred = float(coef[0] + coef[1:] @ x_m3)
            else:
                pred = z[i]
            results["M3"][h].append(pred)

        # ── LWR configs ───────────────────────────────────────────────────
        sign_i = np.sign(z[i] - z[i-1])   # +1 вершина, -1 впадина

        for name, n_lags, use_dt, include_level, delta_target, abs_y in CONFIGS:
            xi  = xi_cache[name]
            x_q = build_feature(z, dt, i, n_lags, use_dt, include_level, abs_y)
            if x_q is None:
                for h in HORIZONS:
                    results[name][h].append(z[i])
                continue

            # обучающая выборка
            X_tr_list = []
            y_tr_dict = {h: [] for h in HORIZONS}
            for j in range(train_s, i):
                xj = build_feature(z, dt, j, n_lags, use_dt, include_level, abs_y)
                if xj is None:
                    continue
                X_tr_list.append(xj)
                for h in HORIZONS:
                    if j + h >= N:
                        y_tr_dict[h].append(np.nan)
                        continue
                    if delta_target and abs_y:
                        # F-вариант: цель = |следующее качание|
                        tgt = abs(z[j + 1] - z[j]) if h == 1 else abs(z[j + h] - z[j])
                    elif delta_target:
                        tgt = z[j + h] - z[j]
                    else:
                        tgt = z[j + h]
                    y_tr_dict[h].append(tgt)

            if len(X_tr_list) < xi + 1:
                for h in HORIZONS:
                    results[name][h].append(z[i])
                continue

            # ── вспомогательные ряды (causally gated каждый отдельно) ────
            if use_aux:
                for z_s, dt_s, dates_s in aux_series:
                    ce = int(np.searchsorted(dates_s, current_date, side="left"))
                    if ce <= n_lags + 1:
                        continue
                    N_s = len(z_s)
                    for k in range(n_lags + 1, ce):
                        xk = build_feature(z_s, dt_s, k, n_lags,
                                           use_dt, include_level, abs_y)
                        if xk is None:
                            continue
                        X_tr_list.append(xk)
                        for h in HORIZONS:
                            if k + h < N_s:
                                if delta_target and abs_y:
                                    tgt = abs(z_s[k+1] - z_s[k]) if h == 1 \
                                          else abs(z_s[k+h] - z_s[k])
                                elif delta_target:
                                    tgt = z_s[k+h] - z_s[k]
                                else:
                                    tgt = z_s[k+h]
                            else:
                                tgt = np.nan
                            y_tr_dict[h].append(tgt)
            # ──────────────────────────────────────────────────────────────

            X_tr = np.array(X_tr_list)
            for h in HORIZONS:
                y_arr = np.array(y_tr_dict[h])
                mask  = ~np.isnan(y_arr)
                if mask.sum() < xi + 1:
                    results[name][h].append(z[i])
                    continue
                raw = lwr_predict(X_tr[mask], y_arr[mask], x_q, xi)
                if delta_target and abs_y:
                    pred = z[i] - sign_i * abs(raw)
                elif delta_target:
                    pred = z[i] + raw
                else:
                    pred = raw
                results[name][h].append(pred)

    for name in results:
        for h in HORIZONS:
            results[name][h] = np.array(results[name][h])
    for h in HORIZONS:
        actuals[h] = np.array(actuals[h])

    return results, actuals

# ── метрики ───────────────────────────────────────────────────────────────────

def compute_metrics(results, actuals, z):
    mean_y = np.mean(np.abs(np.diff(z)))
    rows = []
    for name in results:
        for h in HORIZONS:
            pred = results[name][h]; act = actuals[h]
            mask = ~np.isnan(act) & ~np.isnan(pred)
            if mask.sum() == 0:
                continue
            mae = np.abs(pred[mask] - act[mask]).mean()
            rows.append({"model": name, "h": h, "rMAE": mae / mean_y})
    return pd.DataFrame(rows)

# ── графики ───────────────────────────────────────────────────────────────────

PALETTE_A = {
    "M0":   ("#aaaaaa", "--", 1.8),
    "M3":   ("#333333", "-",  1.8),
    "A_p2": ("#1f77b4", "-",  1.4),
    "A_p4": ("#ff7f0e", "-",  1.4),
    "A_p6": ("#2ca02c", "-",  1.4),
}
PALETTE_E = {
    "M0":   ("#aaaaaa", "--", 1.8),
    "A_p2": ("#1f77b4", "-",  1.6),   # лучшая A — ориентир
    "E_p2": ("#d62728", "-",  1.4),
    "E_p4": ("#e377c2", "-",  1.4),
    "E_p6": ("#8c564b", "-",  1.4),
    "F_p2": ("#17becf", "-",  1.6),   # амплитуда + знак
}

def _plot_panel(ax, df, models, palette, title):
    for name in models:
        sub = df[df.model == name].sort_values("h")
        if sub.empty:
            continue
        color, ls, lw = palette[name]
        vals = sub.rMAE.values
        ax.plot(HORIZONS[:len(vals)], vals, color=color, ls=ls, lw=lw,
                marker="o", ms=5, label=name)
        # аннотация h=1
        ax.annotate(f"{vals[0]:.3f}", (1, vals[0]),
                    textcoords="offset points", xytext=(4, 2),
                    fontsize=7, color=color)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("горизонт h (пивотов)", fontsize=9)
    ax.set_ylabel("rMAE", fontsize=9)
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

def plot_results(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LWR зигзаг в событийном времени — SBER 1d  (порог 2%, окно 200)",
                 fontsize=12)

    _plot_panel(axes[0], df,
                ["M0", "M3", "A_p2", "A_p4", "A_p6"], PALETTE_A,
                "A — X=[z, y(sign)·p], цель=z")
    _plot_panel(axes[1], df,
                ["M0", "A_p2", "E_p2", "E_p4", "E_p6", "F_p2"], PALETTE_E,
                "E — X=[z, |y|·p], цель=z  |  F — цель=|Δz|+sign")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)

# ── ценовой график ────────────────────────────────────────────────────────────

def plot_price_chart(df_raw, trend, pivots, z, results, actuals, test_start, out_path,
                     zoom_last=200):
    """
    Два субплота: полный тест-период и zoom последних zoom_last баров.
    Синяя линия — реальный зигзаг.
    Красные кружки — A_p2 h=1 (предсказанный уровень, x = реальный бар пивота).
    Серые треугольники — M0 h=1 (random walk).
    """
    close = df_raw["close"].values
    dates = pd.to_datetime(df_raw["begin"]).values

    # test_indices в pivot-координатах
    test_end_piv = len(z) - max(HORIZONS)
    test_indices = list(range(test_start, test_end_piv))

    # bar-диапазон тест-периода
    bar_start_full = pivots[test_indices[0]]
    bar_end_full   = pivots[min(test_indices[-1] + 1, len(pivots) - 1)]
    bar_start_zoom = max(0, bar_end_full - zoom_last)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("LWR зигзаг h=1 — SBER 1d  (порог 2%, A_p2 vs M0)", fontsize=12)

    for ax_idx, (bs, be, title) in enumerate([
        (bar_start_full, bar_end_full,
         f"Полный тест-период (пивоты {test_indices[0]}–{test_indices[-1]})"),
        (bar_start_zoom, bar_end_full,
         f"Zoom: последние {zoom_last} баров"),
    ]):
        ax = axes[ax_idx]

        # цена
        sl = slice(bs, be + 1)
        ax.plot(dates[sl], close[sl], color="#cccccc", lw=0.7, zorder=1)

        # реальный зигзаг
        piv_range = [p for p in pivots if bs <= p <= be]
        if len(piv_range) >= 2:
            ax.plot(dates[piv_range], close[piv_range],
                    color="steelblue", lw=1.3, zorder=2, label="Реальный зигзаг")
            ax.scatter(dates[piv_range], close[piv_range],
                       color="steelblue", s=18, zorder=3)

        # предсказания A_p2 и M0 на h=1
        for name, marker, ms, color, zord, label_str in [
            ("M0",   "^", 45, "#aaaaaa", 4, "M0 (random walk)"),
            ("A_p2", "o", 55, "crimson",  5, "A_p2 LWR h=1"),
        ]:
            xs, ys = [], []
            pred_arr = results[name][1]   # h=1
            for k, i in enumerate(test_indices):
                if k >= len(pred_arr):
                    break
                bar_target = pivots[i + 1] if i + 1 < len(pivots) else None
                if bar_target is None or not (bs <= bar_target <= be):
                    continue
                price_pred = pred_arr[k] * trend[bar_target]
                xs.append(dates[bar_target])
                ys.append(price_pred)
            if xs:
                lbl = label_str if ax_idx == 0 else None
                ax.scatter(xs, ys, marker=marker, s=ms, color=color,
                           alpha=0.75, zorder=zord, label=lbl,
                           edgecolors="white", linewidths=0.4)

        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Цена (руб.)", fontsize=9)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.grid(alpha=0.2)
        if ax_idx == 0:
            ax.legend(fontsize=9, ncol=3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


# ── сравнительный график 1h-аугментации ──────────────────────────────────────

def plot_aux_comparison(df_no_aux, df_1h, df_10m, out_path):
    """M0 / A_p2 / A_p2+1h / A_p2+1h+10m по горизонтам."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Фрактальная аугментация LWR A_p2 — SBER 1d  (порог 2%)",
                 fontsize=12)

    specs = [
        ("M0",          df_no_aux, "#aaaaaa", "--", 1.8),
        ("A_p2",        df_no_aux, "#1f77b4", "-",  1.6),
        ("A_p2+1h",     df_1h,     "#ff7f0e", "-",  1.6),
        ("A_p2+1h+10m", df_10m,    "#d62728", "-",  1.6),
    ]
    for name, src, color, ls, lw in specs:
        sub = src[src.model == name].sort_values("h")
        if sub.empty:
            continue
        vals = sub.rMAE.values
        ax.plot(HORIZONS[:len(vals)], vals, color=color, ls=ls, lw=lw,
                marker="o", ms=5, label=name)
        ax.annotate(f"{vals[0]:.3f}", (1, vals[0]),
                    textcoords="offset points", xytext=(4, 2),
                    fontsize=8, color=color)

    ax.set_xlabel("горизонт h (пивотов)", fontsize=10)
    ax.set_ylabel("rMAE", fontsize=10)
    ax.set_xticks(HORIZONS)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1d ────────────────────────────────────────────────────────────────────
    print("Загрузка SBER 1d …")
    df_raw = load_data(DATA_PATH)
    close  = df_raw["close"].values.astype(np.float64)
    trend  = logtrend_causal(close)
    ratio  = close / trend
    pivots = find_pivots(ratio, THRESHOLD)
    z      = ratio[np.array(pivots)]
    dt_arr = np.diff(np.array(pivots)).astype(float)
    dt_arr = np.append(dt_arr[0], dt_arr)
    dates_1d     = pd.to_datetime(df_raw["begin"]).dt.normalize().values
    dates_piv_1d = dates_1d[np.array(pivots)]

    print(f"  баров: {len(close)}, пивотов: {len(pivots)}")
    mean_y = np.mean(np.abs(np.diff(z)))
    print(f"  mean|y|={mean_y:.4f}, mean Δt={dt_arr[1:].mean():.1f} баров")

    # ── 1h ────────────────────────────────────────────────────────────────────
    print("Загрузка SBER 1h …")
    z_1h, dt_1h, dates_piv_1h = load_aux(DATA_PATH_1H, THRESHOLD)
    print(f"  пивотов 1h: {len(z_1h)}")
    print(f"  диапазон дат 1h: {dates_piv_1h[0]} … {dates_piv_1h[-1]}")

    # ── 10m ───────────────────────────────────────────────────────────────────
    print("Загрузка SBER 10m …")
    z_10m, dt_10m, dates_piv_10m = load_aux(DATA_PATH_10M, THRESHOLD)
    print(f"  пивотов 10m: {len(z_10m)}")
    print(f"  диапазон дат 10m: {dates_piv_10m[0]} … {dates_piv_10m[-1]}")
    print("  CAUSALITY GATE: searchsorted отдельно для каждого ряда")

    print("\nКонфигурации LWR:")
    for name, p, use_dt, incl, delta, abs_y in CONFIGS:
        d  = feat_dim(p, use_dt, incl)
        xi = xi_for(p, use_dt, incl)
        print(f"  {name:6s}: dim={d:2d}, ξ={xi:3d}  abs_y={abs_y}  delta={delta}")

    # ── прогон 1: без аугментации (все конфиги) ───────────────────────────────
    print("\nWalk-forward (без aux) …", flush=True)
    res_no_aux, actuals = walk_forward(z, dt_arr, dates_piv_1d)
    df_no_aux = compute_metrics(res_no_aux, actuals, z)

    print("\nМетрики rMAE — без aux:")
    order = ["M0", "M3"] + [name for name, *_ in CONFIGS]
    order = [m for m in order if m in df_no_aux.model.values]
    tbl = df_no_aux.pivot(index="model", columns="h", values="rMAE").reindex(order)
    print(tbl.round(3).to_string())

    # ── прогон 2: с 1h ────────────────────────────────────────────────────────
    print("\nWalk-forward (+ 1h) …", flush=True)
    res_1h, _ = walk_forward(z, dt_arr, dates_piv_1d,
                             aux_series=[(z_1h, dt_1h, dates_piv_1h)])
    df_1h = compute_metrics({"M0": res_1h["M0"], "A_p2": res_1h["A_p2"]},
                             actuals, z)
    df_1h["model"] = df_1h["model"].replace({"A_p2": "A_p2+1h"})

    # ── прогон 3: с 1h + 10m ──────────────────────────────────────────────────
    print("Walk-forward (+ 1h + 10m) …", flush=True)
    res_10m, _ = walk_forward(z, dt_arr, dates_piv_1d,
                              aux_series=[(z_1h, dt_1h, dates_piv_1h),
                                          (z_10m, dt_10m, dates_piv_10m)])
    df_10m = compute_metrics({"M0": res_10m["M0"], "A_p2": res_10m["A_p2"]},
                              actuals, z)
    df_10m["model"] = df_10m["model"].replace({"A_p2": "A_p2+1h+10m"})

    # ── сравнение ─────────────────────────────────────────────────────────────
    print("\nСравнение A_p2 по уровням аугментации (rMAE):")
    rows = []
    for label, src, mname in [
        ("M0",          df_no_aux, "M0"),
        ("A_p2",        df_no_aux, "A_p2"),
        ("A_p2+1h",     df_1h,     "A_p2+1h"),
        ("A_p2+1h+10m", df_10m,    "A_p2+1h+10m"),
    ]:
        row = {"модель": label}
        for h in HORIZONS:
            sub = src[(src.model == mname) & (src.h == h)]
            row[f"h={h}"] = sub.rMAE.values[0] if not sub.empty else float("nan")
        rows.append(row)
    cmp = pd.DataFrame(rows).set_index("модель")
    print(cmp.round(3).to_string())

    v_base = cmp.loc["A_p2",        "h=1"]
    v_1h   = cmp.loc["A_p2+1h",     "h=1"]
    v_10m  = cmp.loc["A_p2+1h+10m", "h=1"]
    print(f"\n  +1h:       {v_base:.3f} → {v_1h:.3f}  "
          f"({'↓' if v_1h < v_base else '↑'}{abs((v_1h-v_base)/v_base*100):.1f}%)")
    print(f"  +1h+10m:   {v_base:.3f} → {v_10m:.3f}  "
          f"({'↓' if v_10m < v_base else '↑'}{abs((v_10m-v_base)/v_base*100):.1f}%)")

    n_lags_max     = max(P_GRID)
    test_start_piv = TRAIN_WIN + n_lags_max + 2

    print("\nРисую метрики (базовые конфиги) …")
    plot_results(df_no_aux, os.path.join(OUT_DIR, "zigzag_lwr_results.png"))

    print("Рисую сравнение фрактальной аугментации …")
    plot_aux_comparison(df_no_aux, df_1h, df_10m,
                        os.path.join(OUT_DIR, "zigzag_lwr_aux_comparison.png"))

    print("Рисую ценовой график …")
    plot_price_chart(df_raw, trend, pivots, z, res_no_aux, actuals,
                     test_start_piv,
                     os.path.join(OUT_DIR, "zigzag_lwr_price.png"))
    print("Готово.")

if __name__ == "__main__":
    main()
