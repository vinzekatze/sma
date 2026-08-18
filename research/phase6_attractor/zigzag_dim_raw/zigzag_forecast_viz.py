#!/usr/bin/env python3
"""
zigzag_forecast_viz.py

Визуализация прогнозов 4-way ансамбля на фоне реальных цен SBER.

Два панели:
  Верхняя (обзор): последние ~800 сырых баров + MA(50/100/200) + зигзаг
  Нижняя (детали): последние N_DETAIL баров + последние N_SHOW прогнозов

Прогнозы показываются как:
  ● начало прогноза (origin pivot)
  ▲/▼ предсказание следующего пивота (▲ HIGH, ▼ LOW)
  | вертикальная линия pred → actual (ошибка)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA     = BASE_DIR.parent.parent.parent / "data" / "candles" / "SBER"
OUT      = BASE_DIR / "results"
OUT.mkdir(exist_ok=True)

T_1D        = 0.04
T_10M       = 0.004
H           = 1
MIN_HISTORY = 50

P_LWR  = 3;  K_LWR = 50
P_SX   = 8;  K_SX  = P_SX + 1
THETA  = 1.0
P_RBF  = 3;  K_RBF = 12

A_LWR = 0.05; A_SX = 0.35; A_SM = 0.20; A_RBF = 0.40

N_SHOW   = 25    # последних прогнозов в детальной панели
N_DETAIL = 250   # сырых баров в детальной панели
MA_WINDOWS = [50, 100, 200]
MA_COLORS  = ["#2196F3", "#FF9800", "#E91E63"]


def parse_dt(s):
    s = str(s)[:10]
    return datetime.strptime(s, "%Y-%m-%d")


def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    highs  = np.array([d["high"]  for d in raw], dtype=np.float64)
    lows   = np.array([d["low"]   for d in raw], dtype=np.float64)
    begins = np.array([d["begin"] for d in raw])
    dates  = np.array([parse_dt(b) for b in begins])
    return highs, lows, begins, dates


def find_pivots(highs, lows, begins, thr):
    vals, dts, bidxs, ptypes = [], [], [], []
    direction = 0
    ext_val = (highs[0] + lows[0]) / 2.0
    ext_idx = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1;  ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val, ext_idx = lows[i], i
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val, ext_idx = highs[i], i
            elif ext_val - lows[i] >= thr * ext_val:
                vals.append(ext_val); dts.append(begins[ext_idx])
                bidxs.append(ext_idx); ptypes.append(1)
                direction = -1; ext_val, ext_idx = lows[i], i
        else:
            if lows[i] < ext_val:
                ext_val, ext_idx = lows[i], i
            elif highs[i] - ext_val >= thr * ext_val:
                vals.append(ext_val); dts.append(begins[ext_idx])
                bidxs.append(ext_idx); ptypes.append(-1)
                direction = 1; ext_val, ext_idx = highs[i], i
    return (np.array(vals), np.array(dts),
            np.array(bidxs, int), np.array(ptypes, int))


def build_X(prices, p):
    n = len(prices); lp = np.log(prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


def causal_ma(series, W):
    cs = np.concatenate([[0.0], np.cumsum(series)])
    counts = np.minimum(np.arange(1, len(series) + 1), W)
    start  = np.maximum(0, np.arange(len(series)) - W + 1)
    return (cs[np.arange(1, len(series) + 1)] - cs[start]) / counts


def run():
    h1d, l1d, beg1d, dates1d = load_tf("1d")
    h10m, l10m, beg10m, _    = load_tf("10m")
    p1d,  dt1d,  bidxs1d,  types1d  = find_pivots(h1d,  l1d,  beg1d,  T_1D)
    p10m, dt10m, _,         _        = find_pivots(h10m, l10m, beg10m, T_10M)

    n1d = len(p1d); na = len(p10m)
    ce10m_all = np.searchsorted(dt10m, dt1d, side="left")

    Xs = {p: (build_X(p1d, p), build_X(p10m, p))
          for p in set([P_LWR, P_SX, P_RBF])}

    mid1d = (h1d + l1d) / 2.0

    print(f"SBER 1d {n1d} пив  10m {na} пив")
    print("Прогон ансамбля...", flush=True)

    # ── Ансамблевый прогон ───────────────────────────────────────────────────
    records = []   # (step, origin_bidx, target_bidx, p_cur, pred, actual, ttype)

    for step in range(MIN_HISTORY, n1d - H):
        skip = any(np.any(np.isnan(Xs[p][0][step])) for p in [P_LWR, P_SX, P_RBF])
        if skip:
            continue

        ce    = int(ce10m_all[step])
        p_cur = float(p1d[step])
        ttype = int(types1d[step + H])

        def make_pool(p):
            X1d_, X10m_ = Xs[p]
            j1d = np.arange(p - 1, step)
            v1d = ~np.any(np.isnan(X1d_[j1d]), axis=1) & (j1d + H < n1d)
            idx1 = j1d[v1d]
            Xp     = list(X1d_[idx1]);   yp_abs = list(p1d[idx1 + H])
            yp_src = list(p1d[idx1])
            j10 = np.arange(p - 1, min(ce, na - H))
            if len(j10):
                v10   = ~np.any(np.isnan(X10m_[j10]), axis=1)
                idx10 = j10[v10]
                Xp.extend(X10m_[idx10]); yp_abs.extend(p10m[idx10 + H])
                yp_src.extend(p10m[idx10])
            if len(Xp) < p + 2:
                return None
            X_pool = np.array(Xp)
            y_abs  = np.array(yp_abs, dtype=float)
            y_lr   = np.log(y_abs / np.array(yp_src, dtype=float))
            x_q    = X1d_[step]
            mu  = X_pool.mean(0); sig = np.where(X_pool.std(0) < 1e-10, 1.0, X_pool.std(0))
            Xn  = (X_pool - mu) / sig; xn = (x_q - mu) / sig
            d   = np.linalg.norm(Xn - xn, axis=1)
            return Xn, xn, d, y_abs, y_lr, X_pool.shape[0]

        res3 = make_pool(P_LWR)
        y_lwr = y_smap = np.nan
        if res3:
            Xn, xn, d, y_abs, y_lr, N = res3
            ord_ = np.argsort(d)
            k_eff = min(K_LWR, N); knn = ord_[:k_eff]; xi = d[ord_[k_eff-1]]
            if xi < 1e-12:
                y_lwr = float(y_abs[knn].mean())
            else:
                w = np.exp(-0.5*(d[knn]/xi)**2); ws = np.sqrt(w)
                A_ = np.column_stack([np.ones(k_eff), Xn[knn]]) * ws[:,None]
                c, *_ = np.linalg.lstsq(A_, y_abs[knn]*ws, rcond=None)
                y_lwr = float(c[0] + c[1:] @ xn)
            mean_d = d.mean() + 1e-12
            w_sm   = np.exp(-THETA * d / mean_d); ws_sm = np.sqrt(w_sm)
            A_sm   = np.column_stack([np.ones(N), Xn]) * ws_sm[:,None]
            c_sm, *_ = np.linalg.lstsq(A_sm, y_abs*ws_sm, rcond=None)
            y_smap = float(c_sm[0] + c_sm[1:] @ xn)

        res8 = make_pool(P_SX)
        y_sx = np.nan
        if res8 and res8[5] >= K_SX:
            Xn, xn, d, y_abs, y_lr, N = res8
            ords = np.argsort(d); knn = ords[:K_SX]; d1 = d[ords[0]]
            if d1 < 1e-12:
                y_sx = float(p_cur * np.exp(y_lr[ords[0]]))
            else:
                w = np.exp(-d[knn]/d1); w /= w.sum()
                y_sx = float(p_cur * np.exp(w @ y_lr[knn]))

        y_rbf = np.nan
        if res3 and res3[5] >= K_RBF:
            Xn, xn, d, y_abs, y_lr, N = res3
            ord_ = np.argsort(d); k = min(K_RBF, N); knn = ord_[:k]
            xi = d[ord_[k-1]]
            if xi < 1e-12:
                y_rbf = float(p_cur * np.exp(y_lr[knn].mean()))
            else:
                w = np.exp(-0.5*(d[knn]/xi)**2)
                y_rbf = float(p_cur * np.exp((w @ y_lr[knn]) / w.sum()))

        pred = A_LWR*y_lwr + A_SX*y_sx + A_SM*y_smap + A_RBF*y_rbf
        if np.isnan(pred):
            continue

        records.append((
            step,
            int(bidxs1d[step]),       # bar idx origin
            int(bidxs1d[step + H]),   # bar idx target
            p_cur,
            float(pred),
            float(p1d[step + H]),     # actual
            ttype,
        ))

    print(f"Готово. Прогнозов: {len(records)}\n")

    # ── Берём последние N_SHOW ────────────────────────────────────────────────
    shown = records[-N_SHOW:]

    # ── Рисуем ───────────────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [1, 1.6]},
    )

    # Общая функция: рисуем сырые цены + MA + зигзаг на ax в диапазоне баров
    def draw_base(ax, bar_lo, bar_hi):
        sl = slice(bar_lo, bar_hi + 1)
        xs = dates1d[sl]

        # Mid-price
        ax.plot(xs, mid1d[sl], color="#cccccc", lw=0.8, zorder=1)

        # MA
        for W, color in zip(MA_WINDOWS, MA_COLORS):
            ma = causal_ma(mid1d, W)
            ax.plot(xs, ma[sl], color=color, lw=1.2, ls="--",
                    label=f"MA{W}", zorder=2, alpha=0.85)

        # Зигзаг (пивоты в этом диапазоне)
        piv_mask = (bidxs1d >= bar_lo) & (bidxs1d <= bar_hi)
        piv_x    = np.array([parse_dt(dt1d[i]) for i in np.where(piv_mask)[0]])
        piv_y    = p1d[piv_mask]
        if len(piv_x) > 1:
            ax.plot(piv_x, piv_y, color="black", lw=1.4, zorder=3,
                    label="зигзаг T=4%")

    # ── Верхняя панель: обзор (последние ~800 баров) ─────────────────────────
    bar_lo_top = max(0, len(h1d) - 800)
    bar_hi_top = len(h1d) - 1
    draw_base(ax_top, bar_lo_top, bar_hi_top)
    ax_top.set_title("SBER 1d — обзор (последние ~800 баров)", fontsize=11)
    ax_top.legend(fontsize=8, loc="upper left"); ax_top.grid(True, alpha=0.3)
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_top.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Подсветить диапазон нижней панели
    bar_lo_bot = max(0, len(h1d) - N_DETAIL)
    ax_top.axvspan(dates1d[bar_lo_bot], dates1d[bar_hi_top],
                   alpha=0.08, color="steelblue", zorder=0)

    # ── Нижняя панель: детали (последние N_DETAIL баров) ─────────────────────
    bar_lo_bot = max(0, len(h1d) - N_DETAIL)
    bar_hi_bot = len(h1d) - 1
    draw_base(ax_bot, bar_lo_bot, bar_hi_bot)

    # Последние N_SHOW прогнозов
    for rec in shown:
        step, bidx_orig, bidx_tgt, p_cur, pred, actual, ttype = rec

        # Рисуем только если origin и target в видимом диапазоне
        if bidx_orig < bar_lo_bot:
            continue

        x_orig = parse_dt(beg1d[bidx_orig])
        x_tgt  = parse_dt(beg1d[bidx_tgt])

        # Origin
        ax_bot.plot(x_orig, p_cur, "o", color="black", ms=5, zorder=6)

        # Линия origin → predicted
        is_high = (ttype == 1)
        col = "#c0392b" if is_high else "#2980b9"   # красный=HIGH, синий=LOW
        ax_bot.plot([x_orig, x_tgt], [p_cur, pred],
                    color=col, lw=1.0, ls=":", alpha=0.7, zorder=4)

        # Predicted point
        marker = "^" if is_high else "v"
        ax_bot.plot(x_tgt, pred, marker=marker, color=col,
                    ms=8, zorder=7, markeredgecolor="white", markeredgewidth=0.5)

        # Вертикаль pred → actual (ошибка)
        ax_bot.plot([x_tgt, x_tgt], [pred, actual],
                    color="gray", lw=1.5, alpha=0.6, zorder=5)

    # Легенда для прогнозов
    from matplotlib.lines import Line2D
    leg_extra = [
        Line2D([0], [0], marker="o",  color="black",   ls="None", ms=5,  label="origin"),
        Line2D([0], [0], marker="^",  color="#c0392b", ls="None", ms=8,  label="pred HIGH"),
        Line2D([0], [0], marker="v",  color="#2980b9", ls="None", ms=8,  label="pred LOW"),
        Line2D([0], [0], color="gray", lw=1.5, alpha=0.6,                label="ошибка (pred→actual)"),
    ]
    handles, labels = ax_bot.get_legend_handles_labels()
    ax_bot.legend(handles=handles + leg_extra, fontsize=8,
                  loc="upper left", ncol=2)

    ax_bot.set_title(f"Детали: последние {N_DETAIL} баров + последние {N_SHOW} прогнозов",
                     fontsize=11)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_bot.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
    plt.setp(ax_bot.xaxis.get_majorticklabels(), rotation=35, ha="right")

    fig.suptitle(
        f"4-way ансамбль: прогнозы зигзага SBER | T=4% 1d+10m | последние {N_SHOW}",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = OUT / "forecast_viz.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"График → {out_path}")

    # ── Краткая таблица последних N_SHOW прогнозов ───────────────────────────
    acts_shown  = np.array([r[5] for r in shown])
    preds_shown = np.array([r[4] for r in shown])
    errs_shown  = preds_shown - acts_shown
    dz = np.mean(np.abs(np.diff(acts_shown))) if len(acts_shown) > 1 else 1.0

    print(f"\nПоследние {N_SHOW} прогнозов:")
    print(f"  {'#':>3}  {'тип':>5}  {'origin':>8}  {'pred':>8}  "
          f"{'actual':>8}  {'err':>7}  {'err%':>6}")
    print("  " + "─" * 55)
    for i, rec in enumerate(shown):
        step, bidx_o, bidx_t, p_cur, pred, actual, ttype = rec
        err   = pred - actual
        err_p = err / actual * 100
        typ   = "HIGH" if ttype == 1 else "LOW "
        print(f"  {i+1:>3}  {typ}  {p_cur:>8.2f}  {pred:>8.2f}  "
              f"{actual:>8.2f}  {err:>+7.2f}  {err_p:>+5.2f}%")

    rmae_shown = float(np.mean(np.abs(errs_shown)) / dz)
    print(f"\n  rMAE (last {N_SHOW}): {rmae_shown:.4f}  "
          f"(global: 0.3963)")


if __name__ == "__main__":
    run()
