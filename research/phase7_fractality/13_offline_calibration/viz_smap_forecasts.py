#!/usr/bin/env python3
"""
viz_smap_forecasts.py — Визуализация прогнозов S-map vs наивный на сырых ценах.

Выбирает N случайных пивотов T_BIG зигзага из последних RECENT пивотов,
для каждого строит прогноз S-map через строгую обрезку данных (ref-стиль),
показывает predicted vs naive vs actual на графике сырых цен.

Наивный прогноз (persistence): следующий пивот = текущий^2 / предыдущий (то же плечо).
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "reference"))
from smap_ref import (build_pool_vectors, build_query_vector,
                      filter_direction_pool, smap_predict)

DATA_FILE = HERE.parents[2] / "data" / "candles" / "SBER" / "10m.json"
RESULTS   = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# ── Параметры S-map (оптимальные для T=2%) ───────────────────────────────────
T_BIG    = 0.02
T_POOL   = 0.02 * 0.8129   # T_ratio=0.8129
M        = 4
THETA    = 1.710
MIN_POOL = max(M + 2, 4)

N_SAMPLES = 7    # сколько случайных пивотов показать
RECENT    = 150  # брать из последних RECENT пивотов
SEED      = 42

# для отображения: сколько свечей вокруг каждого события
BAR_BEFORE = 300
BAR_AFTER  = 300


def load(path):
    with open(path) as f:
        raw = json.load(f)
    close = np.array([c["close"] for c in raw], dtype=np.float64)
    lh    = np.log(np.array([c["high"]  for c in raw], dtype=np.float64))
    ll    = np.log(np.array([c["low"]   for c in raw], dtype=np.float64))
    dt    = np.array([c["begin"] for c in raw])
    return close, lh, ll, dt


def parse_dt(s):
    return datetime.strptime(s[:16], "%Y-%m-%d %H:%M")


def build_zigzag_full(lh, ll, dates, thr):
    """Зигзаг с датами как экстремума, так и подтверждения."""
    lp, ext_dates, conf_dates, dirs = [], [], [], []
    cur_dir = 0
    ext = (lh[0] + ll[0]) / 2.0
    ext_bar = 0
    for i in range(len(lh)):
        if cur_dir == 0:
            if lh[i] - ext >= thr:
                cur_dir, ext, ext_bar = 1, lh[i], i
            elif ext - ll[i] >= thr:
                cur_dir, ext, ext_bar = -1, ll[i], i
        elif cur_dir == 1:
            if lh[i] > ext:
                ext, ext_bar = lh[i], i
            elif ext - ll[i] >= thr:
                lp.append(ext)
                ext_dates.append(dates[ext_bar])
                conf_dates.append(dates[i])
                dirs.append(+1)
                cur_dir, ext, ext_bar = -1, ll[i], i
        else:
            if ll[i] < ext:
                ext, ext_bar = ll[i], i
            elif lh[i] - ext >= thr:
                lp.append(ext)
                ext_dates.append(dates[ext_bar])
                conf_dates.append(dates[i])
                dirs.append(-1)
                cur_dir, ext, ext_bar = 1, lh[i], i
    return (np.array(lp), np.array(ext_dates),
            np.array(conf_dates), np.array(dirs, dtype=np.int8))


def forecast_step(lh, ll, dt, bar_cut):
    """Прогноз S-map на данных [0..bar_cut]. Возвращает pred_price или None."""
    lh_c, ll_c, dt_c = lh[:bar_cut], ll[:bar_cut], dt[:bar_cut]
    lp_q, _, cd_q, qdirs = build_zigzag_full(lh_c, ll_c, dt_c, T_BIG)
    lp_p, _, cd_p, pdirs = build_zigzag_full(lh_c, ll_c, dt_c, T_POOL)
    if len(lp_q) < M + 1:
        return None
    pfm, ptgt, pdir = build_pool_vectors(lp_p, pdirs, M)
    qvec = build_query_vector(lp_q, M)
    qdir = int(qdirs[-1])
    df, dt_ = filter_direction_pool(pfm, ptgt, pdir, qdir, MIN_POOL)
    if df is None:
        return None
    lr = smap_predict(qvec, df, dt_, THETA)
    return float(np.exp(lp_q[-1] + lr))


def main():
    print("Загрузка данных...")
    close, lh, ll, dt = load(DATA_FILE)
    print(f"  {len(dt)} свечей  ({dt[0][:10]} … {dt[-1][:10]})")

    # Полный зигзаг с датами экстремумов и подтверждений
    lp_full, ed_full, cd_full, _ = build_zigzag_full(lh, ll, dt, T_BIG)
    n = len(lp_full)
    print(f"  T={T_BIG*100:.0f}%  пивотов: {n}")

    # Выбираем случайные пивоты из последних RECENT (исключая последний — нет actual)
    pool_idx = list(range(n - RECENT, n - 1))
    rng      = np.random.default_rng(SEED)
    selected = sorted(rng.choice(pool_idx, size=N_SAMPLES, replace=False))
    print(f"  Выбрано пивотов: {selected}")

    # Строим прогнозы
    results = []
    for step in selected:
        confirm    = cd_full[step]
        ext_date   = ed_full[step]
        bar_cut    = int(np.searchsorted(dt, confirm, side="right"))
        cur_price  = float(np.exp(lp_full[step]))
        pers_price = float(np.exp(lp_full[step - 1]))
        act_price  = float(np.exp(lp_full[step + 1]))
        pred_price = forecast_step(lh, ll, dt, bar_cut)

        ext_dt     = parse_dt(ext_date)
        confirm_dt = parse_dt(confirm)
        actual_dt  = parse_dt(ed_full[step + 1])   # дата экстремума следующего пивота

        err_pers  = abs(pers_price - act_price)
        err_pred  = abs(pred_price - act_price) if pred_price else float("nan")

        results.append({
            "step":       step,
            "ext_dt":     ext_dt,
            "confirm_dt": confirm_dt,
            "actual_dt":  actual_dt,
            "bar_cut":    bar_cut,
            "cur_price":  cur_price,
            "pers_price": pers_price,
            "act_price":  act_price,
            "pred_price": pred_price,
            "err_pers":   err_pers,
            "err_pred":   err_pred,
        })
        status = f"{pred_price:.4f}" if pred_price else "N/A"
        print(f"  step={step}  pers={pers_price:.4f}  pred={status}  actual={act_price:.4f}"
              f"  |Δpers|={err_pers:.4f}  |Δpred|={err_pred:.4f}")

    # ── График ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(N_SAMPLES, 1,
                             figsize=(14, 2.8 * N_SAMPLES),
                             squeeze=False)

    for ax, r in zip(axes[:, 0], results):
        bar_cut = r["bar_cut"]
        lo = max(0, bar_cut - BAR_BEFORE)
        hi = min(len(dt), bar_cut + BAR_AFTER)

        xs      = [parse_dt(d) for d in dt[lo:hi]]
        ys      = close[lo:hi]
        ax.plot(xs, ys, color="#888888", lw=0.7, alpha=0.8)

        # Зигзаг-линия в окне [lo..hi]
        zz_dt_lo = dt[lo]; zz_dt_hi = dt[hi - 1]
        zz_xs, zz_ys = [], []
        for k in range(len(lp_full)):
            if ed_full[k] >= zz_dt_lo and ed_full[k] <= zz_dt_hi:
                zz_xs.append(parse_dt(ed_full[k]))
                zz_ys.append(np.exp(lp_full[k]))
        if zz_xs:
            ax.plot(zz_xs, zz_ys, color="steelblue", lw=1.2, alpha=0.5, zorder=3)

        # Экстремум пивота (где реально был max/min)
        ax.scatter([r["ext_dt"]], [r["cur_price"]],
                   color="steelblue", s=70, zorder=6, label="пивот (экстремум)")
        # Вертикаль подтверждения: момент когда алгоритм узнал о пивоте
        ax.axvline(r["confirm_dt"], color="steelblue", lw=0.9, ls=":", alpha=0.5,
                   label="подтверждение пивота")
        # Горизонтальная связка: уровень пивота от экстремума до подтверждения
        ax.plot([r["ext_dt"], r["confirm_dt"]],
                [r["cur_price"], r["cur_price"]],
                color="steelblue", lw=0.8, ls="--", alpha=0.4)

        # Actual следующий пивот (на дате его экстремума)
        ax.scatter([r["actual_dt"]], [r["act_price"]],
                   color="black", marker="*", s=140, zorder=7, label="actual (след. пивот)")

        # Persistence наивный: следующий пивот = предыдущий^2 / за-предыдущий
        ax.plot([r["confirm_dt"], r["actual_dt"]],
                [r["cur_price"], r["pers_price"]],
                color="tomato", lw=1.5, ls="--", label="persistence naive")
        ax.scatter([r["actual_dt"]], [r["pers_price"]],
                   color="tomato", s=50, zorder=5, marker="D")

        # S-map прогноз: стрела от confirm_date к actual времени
        if r["pred_price"]:
            ax.annotate("",
                xy=(r["actual_dt"], r["pred_price"]),
                xytext=(r["confirm_dt"], r["cur_price"]),
                arrowprops=dict(arrowstyle="->", color="seagreen", lw=1.8))
            ax.scatter([r["actual_dt"]], [r["pred_price"]],
                       color="seagreen", s=70, zorder=6, marker="^",
                       label="S-map прогноз")

        # Аннотация ошибок
        label = (f"|Δ| persistence={r['err_pers']:.4f}  "
                 f"pred={r['err_pred']:.4f}  "
                 f"{'✓ лучше' if r['err_pred'] < r['err_pers'] else '✗ хуже'}")
        ax.set_title(f"step={r['step']}  "
                     f"пивот: {r['ext_dt'].strftime('%Y-%m-%d %H:%M')} "
                     f"(подтв: {r['confirm_dt'].strftime('%H:%M')})  "
                     f"→  actual: {r['actual_dt'].strftime('%Y-%m-%d %H:%M')}\n{label}",
                     fontsize=9, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=8))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Цена (руб.)", fontsize=8)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="upper left", ncol=4)

    fig.suptitle(
        f"S-map прогноз vs наивный  |  SBER 10m  T={T_BIG*100:.0f}%  "
        f"m={M}  θ={THETA}  T_pool={T_POOL*100:.4f}%",
        fontsize=11, y=1.001,
    )
    fig.tight_layout()
    out = RESULTS / "viz_smap_forecasts_SBER_10m_T002.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nГрафик → {out}")

    # Итог
    preds_ok = [r for r in results if r["pred_price"] is not None]
    n_better = sum(1 for r in preds_ok if r["err_pred"] < r["err_pers"])
    print(f"Лучше persistence: {n_better}/{len(preds_ok)}")


if __name__ == "__main__":
    main()
