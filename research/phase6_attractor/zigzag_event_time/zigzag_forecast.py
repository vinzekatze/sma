"""
Baseline прогноз следующих 1-4 пивотов зигзага в событийном времени.

Модели:
  M0 — random walk:         ẑ_{i+h} = z_i
  M1 — mean-reversion:      ẑ_{i+h} = 1.0 (прогноз = logtrend)
  M2 — OLS(z_i):            линейная регрессия только на уровень
  M3 — OLS(z_i,z_{i-1},|y_i|,|y_{i-1}|,Δt_i,Δt_{i-1})

Walk-forward: скользящее окно 200 пивотов.
Метрика: rMAE_h = MAE / mean(|y|) — нормировано на среднее качание.

Запуск: python research/phase6_attractor/zigzag_forecast.py
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import stats
import pandas as pd

# ── пути ──────────────────────────────────────────────────────────────────────

_BASE     = os.path.dirname(__file__)
DATA_PATH = os.path.join(_BASE, "../../data/candles/SBER/1d.json")
OUT_DIR   = _BASE

THRESHOLD  = 0.02
TRAIN_WIN  = 200   # пивотов в обучающем окне
HORIZONS   = [1, 2, 3, 4]

# ── загрузка и нормализация ───────────────────────────────────────────────────

def load_data(path):
    with open(path) as f:
        candles = json.load(f)
    df = pd.DataFrame(candles)
    df["date"]  = pd.to_datetime(df["begin"])
    df["close"] = pd.to_numeric(df["close"])
    return df

def logtrend_causal(close):
    n  = len(close); lc = np.log(np.maximum(close, 1e-10))
    t  = np.arange(n, dtype=np.float64)
    cn = np.arange(1, n+1, dtype=np.float64)
    ct = np.cumsum(t);  ct2 = np.cumsum(t**2)
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

# ── признаки для OLS ──────────────────────────────────────────────────────────

def build_features(z, dt, start, end):
    """
    Для каждого пивота i в [start..end] возвращает вектор признаков.
    Требует i >= 2 (нужен лаг 2).
    """
    rows = []
    for i in range(max(start, 2), end):
        y_i   = z[i]   - z[i-1]
        y_im1 = z[i-1] - z[i-2]
        rows.append([
            z[i],          # z_i
            z[i-1],        # z_{i-1}
            abs(y_i),      # |y_i|
            abs(y_im1),    # |y_{i-1}|
            float(dt[i]),  # Δt_i
            float(dt[i-1]) # Δt_{i-1}
        ])
    return np.array(rows)

def build_targets(z, start, end):
    """Цели z_{i+h} для каждого i и каждого h в HORIZONS."""
    targets = {}
    for h in HORIZONS:
        t = []
        for i in range(max(start, 2), end):
            if i + h < len(z):
                t.append(z[i + h])
            else:
                t.append(np.nan)
        targets[h] = np.array(t)
    return targets

# ── walk-forward прогноз ──────────────────────────────────────────────────────

def walk_forward(z, dt):
    """
    Возвращает dict: predictions[model][h] = array предсказаний
    и test_indices — список i (индексов пивотов), для которых есть прогноз.
    """
    N = len(z)
    test_start = TRAIN_WIN + 2
    test_end   = N - max(HORIZONS)

    predictions = {m: {h: [] for h in HORIZONS} for m in ["M0","M1","M2","M3"]}
    actuals     = {h: [] for h in HORIZONS}
    test_indices = []

    for i in range(test_start, test_end):
        train_start = i - TRAIN_WIN
        # признаки и цели на тренировочном окне
        X_tr = build_features(z, dt, train_start, i)
        T_tr = build_targets(z, train_start, i)

        # признаки текущего пивота
        y_i   = z[i]   - z[i-1]
        y_im1 = z[i-1] - z[i-2]
        x_cur = np.array([z[i], z[i-1], abs(y_i), abs(y_im1),
                          float(dt[i]), float(dt[i-1])])

        test_indices.append(i)

        for h in HORIZONS:
            actual = z[i + h] if i + h < N else np.nan
            actuals[h].append(actual)

            # M0: random walk
            predictions["M0"][h].append(z[i])

            # M1: mean-reversion → 1.0
            predictions["M1"][h].append(1.0)

            # M2: OLS(z_i)
            y_tr = T_tr[h]
            mask = ~np.isnan(y_tr)
            if mask.sum() > 5:
                x_m2 = X_tr[mask, 0:1]  # только z_i
                y_m2 = y_tr[mask]
                sl, ic, _, _, _ = stats.linregress(x_m2[:, 0], y_m2)
                pred_m2 = ic + sl * x_cur[0]
            else:
                pred_m2 = z[i]
            predictions["M2"][h].append(pred_m2)

            # M3: OLS(все признаки) через lstsq
            if mask.sum() > 10:
                Xm = np.column_stack([np.ones(mask.sum()), X_tr[mask]])
                coef, _, _, _ = np.linalg.lstsq(Xm, y_tr[mask], rcond=None)
                pred_m3 = coef[0] + coef[1:].dot(x_cur)
            else:
                pred_m3 = z[i]
            predictions["M3"][h].append(pred_m3)

    # в numpy
    for m in predictions:
        for h in HORIZONS:
            predictions[m][h] = np.array(predictions[m][h])
    for h in HORIZONS:
        actuals[h] = np.array(actuals[h])

    return predictions, actuals, test_indices

# ── метрики ───────────────────────────────────────────────────────────────────

def compute_metrics(predictions, actuals, z):
    mean_y = np.mean(np.abs(np.diff(z)))  # среднее |y| для нормировки
    rows = []
    for m in ["M0","M1","M2","M3"]:
        for h in HORIZONS:
            pred = predictions[m][h]
            act  = actuals[h]
            mask = ~np.isnan(act) & ~np.isnan(pred)
            mae  = np.abs(pred[mask] - act[mask]).mean()
            rmae = mae / mean_y
            rows.append({"model": m, "h": h, "MAE": mae, "rMAE": rmae})
    return pd.DataFrame(rows)

# ── графики ───────────────────────────────────────────────────────────────────

COLORS = {"M0": "gray", "M1": "royalblue", "M2": "darkorange", "M3": "crimson"}

def plot_metrics(df_metrics, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Baseline прогноз пивотов зигзага — SBER 1d  (порог 2%, окно 200)",
                 fontsize=12)

    models = ["M0","M1","M2","M3"]
    model_labels = {
        "M0": "M0: random walk",
        "M1": "M1: к тренду (1.0)",
        "M2": "M2: OLS(z_i)",
        "M3": "M3: OLS(все признаки)",
    }
    x = np.arange(len(HORIZONS))
    width = 0.2

    for ax_idx, metric in enumerate(["MAE", "rMAE"]):
        ax = axes[ax_idx]
        for k, m in enumerate(models):
            vals = [df_metrics[(df_metrics.model==m)&(df_metrics.h==h)][metric].values[0]
                    for h in HORIZONS]
            ax.bar(x + k*width, vals, width=width, color=COLORS[m],
                   alpha=0.85, label=model_labels[m])
        ax.set_xticks(x + 1.5*width)
        ax.set_xticklabels([f"h={h}" for h in HORIZONS])
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} по горизонтам")
        if ax_idx == 0:
            ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


def plot_price_chart(df, trend, ratio, z, pivots, predictions, actuals,
                     test_indices, out_path, zoom_last=150):
    """
    Два графика:
      1. Полный тест-период: цена + зигзаг + предсказания M2 и M3
      2. Zoom последних zoom_last баров
    """
    close = df["close"].values
    dates = df["date"].values

    # тест-период в барах
    test_bar_start = pivots[test_indices[0]]
    test_bar_end   = pivots[min(test_indices[-1] + max(HORIZONS), len(pivots)-1)]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("Прогноз пивотов зигзага на цене — SBER 1d  (порог 2%)", fontsize=12)

    for ax_idx, (bar_start, bar_end, title) in enumerate([
        (test_bar_start, test_bar_end,
         f"Тест-период (пивоты {test_indices[0]}–{test_indices[-1]})"),
        (max(0, test_bar_end - zoom_last), test_bar_end,
         f"Zoom: последние {zoom_last} баров"),
    ]):
        ax = axes[ax_idx]
        sl = slice(bar_start, bar_end + 1)
        ax.plot(dates[sl], close[sl], color="lightgray", lw=0.8, zorder=1)

        # реальный зигзаг на тест-периоде
        piv_in_range = [p for p in pivots if bar_start <= p <= bar_end]
        if len(piv_in_range) >= 2:
            ax.plot(dates[piv_in_range], close[piv_in_range],
                    color="steelblue", lw=1.2, zorder=2, label="Реальный зигзаг")
            ax.scatter(dates[piv_in_range], close[piv_in_range],
                       color="steelblue", s=20, zorder=3)

        # предсказания M2 и M3 для h=1 и h=2
        for m, marker, ms in [("M2", "^", 60), ("M3", "o", 60)]:
            for h in [1, 2]:
                xs, ys = [], []
                for k, i in enumerate(test_indices):
                    if i + h >= len(pivots):
                        continue
                    bar_actual = pivots[i + h]
                    if not (bar_start <= bar_actual <= bar_end):
                        continue
                    ratio_pred = predictions[m][h][k]
                    price_pred = ratio_pred * trend[bar_actual]
                    xs.append(dates[bar_actual])
                    ys.append(price_pred)
                if xs:
                    label = f"{m} h={h}" if ax_idx == 0 else None
                    ax.scatter(xs, ys, marker=marker, s=ms, color=COLORS[m],
                               alpha=0.7, zorder=4, label=label,
                               edgecolors="white", linewidths=0.5)

        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Цена (руб.)", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.grid(alpha=0.2)
        if ax_idx == 0:
            ax.legend(fontsize=8, ncol=3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


def plot_price_chart_detail(df, trend, ratio, z, pivots, predictions, actuals,
                             test_indices, out_path):
    """
    4 субплота: по одному для каждого горизонта h=1..4.
    Scatter: ось X = реальный уровень z_{i+h}, ось Y = предсказанный.
    Показывает разброс ошибок M2 vs M3.
    """
    close = df["close"].values
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Предсказанный vs реальный уровень пивота — SBER 1d", fontsize=12)

    for ax, h in zip(axes.ravel(), HORIZONS):
        act = actuals[h]
        mask = ~np.isnan(act)
        for m in ["M0","M1","M2","M3"]:
            pred = predictions[m][h]
            ax.scatter(act[mask], pred[mask], s=6, alpha=0.4,
                       color=COLORS[m], label=m)
        # идеальная диагональ
        lo = np.nanmin(act); hi = np.nanmax(act)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_title(f"h = {h} пивота вперёд", fontsize=10)
        ax.set_xlabel("реальный z_{i+h}", fontsize=8)
        ax.set_ylabel("предсказанный z_{i+h}", fontsize=8)
        ax.legend(fontsize=7, markerscale=2)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  сохранено: {out_path}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Загрузка SBER 1d …")
    df    = load_data(DATA_PATH)
    close = df["close"].values.astype(np.float64)
    trend = logtrend_causal(close)
    ratio = close / trend

    pivots = find_pivots(ratio, THRESHOLD)
    z      = ratio[np.array(pivots)]
    dt     = np.diff(np.array(pivots))
    dt     = np.append(dt[0], dt)  # Δt[0] = Δt[1] для первого

    print(f"  баров: {len(close)}, пивотов: {len(pivots)}")
    print(f"  train={TRAIN_WIN}, test~{len(pivots)-TRAIN_WIN-max(HORIZONS)-2} пивотов")
    mean_y = np.mean(np.abs(np.diff(z)))
    print(f"  mean|y|={mean_y:.4f}  mean Δt={np.mean(dt[1:]):.1f} баров")

    print("\nWalk-forward …")
    predictions, actuals, test_indices = walk_forward(z, dt)

    print("\nМетрики:")
    df_metrics = compute_metrics(predictions, actuals, z)
    pivot_tbl = df_metrics.pivot(index="model", columns="h", values="rMAE")
    print(pivot_tbl.round(3).to_string())

    print("\nРисую метрики …")
    plot_metrics(df_metrics, os.path.join(OUT_DIR, "zigzag_forecast_metrics.png"))

    print("Рисую ценовой график …")
    plot_price_chart(df, trend, ratio, z, pivots, predictions, actuals,
                     test_indices,
                     os.path.join(OUT_DIR, "zigzag_forecast_price.png"))

    print("Рисую scatter pred vs actual …")
    plot_price_chart_detail(df, trend, ratio, z, pivots, predictions, actuals,
                             test_indices,
                             os.path.join(OUT_DIR, "zigzag_forecast_scatter.png"))

    print("\nГотово.")


if __name__ == "__main__":
    main()
