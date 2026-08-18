#!/usr/bin/env python3
"""
lwr_visualize.py — Два графика:
  1. Цены SBER 10m: зигзаг T_query + LWR прогнозы следующего пивота
  2. Событийное пространство (lr1, lr2): точки запросов, цвет = |ошибка|/dz
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path

T_QUERY     = 0.04
T_POOL      = 0.036
M_EMBED     = 2
K_LWR       = 75
H           = 1
MIN_HISTORY = 50
MIN_DIR     = M_EMBED + 2

DATA = Path(__file__).parent.parent.parent / "data" / "candles"
OUT  = Path(__file__).parent


# ── загрузка ──────────────────────────────────────────────────────────────────

def load_candles(ticker: str, interval: str) -> tuple:
    path = DATA / ticker / f"{interval}.json"
    with open(path) as fh:
        raw = json.load(fh)
    log_highs = np.log(np.array([c["high"]  for c in raw], dtype=np.float64))
    log_lows  = np.log(np.array([c["low"]   for c in raw], dtype=np.float64))
    closes    = np.array([c["close"] for c in raw], dtype=np.float64)
    dates     = np.array([c["begin"] for c in raw])
    return log_highs, log_lows, closes, dates


# ── зигзаг ────────────────────────────────────────────────────────────────────

def build_zigzag(log_highs, log_lows, dates, threshold):
    log_pivot_prices, confirm_dates, pivot_directions = [], [], []
    direction = 0
    extreme   = (log_highs[0] + log_lows[0]) / 2.0
    for bar in range(len(log_highs)):
        if direction == 0:
            if log_highs[bar] - extreme >= threshold:
                direction, extreme = 1, log_highs[bar]
            elif extreme - log_lows[bar] >= threshold:
                direction, extreme = -1, log_lows[bar]
        elif direction == 1:
            if log_highs[bar] > extreme:
                extreme = log_highs[bar]
            elif extreme - log_lows[bar] >= threshold:
                log_pivot_prices.append(extreme)
                confirm_dates.append(dates[bar])
                pivot_directions.append(+1)
                direction, extreme = -1, log_lows[bar]
        else:
            if log_lows[bar] < extreme:
                extreme = log_lows[bar]
            elif log_highs[bar] - extreme >= threshold:
                log_pivot_prices.append(extreme)
                confirm_dates.append(dates[bar])
                pivot_directions.append(-1)
                direction, extreme = 1, log_highs[bar]
    return (
        np.array(log_pivot_prices),
        np.array(confirm_dates),
        np.array(pivot_directions, dtype=np.int8),
    )


# ── один шаг LWR ──────────────────────────────────────────────────────────────

def lwr_step(query_lp, query_dirs, step_i,
             pool_lp, pool_conf, pool_dirs, conf_date):
    j_max = int(np.searchsorted(pool_conf, conf_date, side='left'))
    valid  = np.arange(M_EMBED, j_max - 1)
    if len(valid) == 0:
        return None

    feat = np.array([
        [pool_lp[j - lag] - pool_lp[j - lag - 1] for lag in range(M_EMBED)]
        for j in valid
    ])
    tgt  = pool_lp[valid + 1] - pool_lp[valid]
    dirs = pool_dirs[valid]

    finite = np.all(np.isfinite(feat), axis=1) & np.isfinite(tgt)
    feat, tgt, dirs = feat[finite], tgt[finite], dirs[finite]

    qdir = int(query_dirs[step_i])
    qvec = np.array([
        query_lp[step_i - lag] - query_lp[step_i - lag - 1]
        for lag in range(M_EMBED)
    ])

    mask = dirs == qdir
    if mask.sum() < max(MIN_DIR, K_LWR):
        return None

    cand_feat = feat[mask]
    cand_tgt  = tgt[mask]
    dists = np.linalg.norm(cand_feat - qvec, axis=1)
    nn    = np.argpartition(dists, K_LWR - 1)[:K_LWR]

    nb_feat = cand_feat[nn]
    nb_tgt  = cand_tgt[nn]
    nb_d    = dists[nn]
    dmax    = nb_d.max()

    if dmax < 1e-12:
        return float(nb_tgt.mean())

    w  = np.exp(-0.5 * (nb_d / dmax) ** 2)
    sw = np.sqrt(w)
    A  = np.column_stack([np.ones(K_LWR), nb_feat]) * sw[:, None]
    c, *_ = np.linalg.lstsq(A, nb_tgt * sw, rcond=None)
    return float(c[0] + c[1:] @ qvec)


# ── walk-forward ──────────────────────────────────────────────────────────────

def run_walkforward(query_lp, query_conf, query_dirs,
                    pool_lp, pool_conf, pool_dirs):
    steps, preds, actuals, qvecs = [], [], [], []
    n = len(query_lp)
    for i in range(MIN_HISTORY, n - H):
        lr = lwr_step(query_lp, query_dirs, i,
                      pool_lp, pool_conf, pool_dirs, query_conf[i])
        if lr is None:
            continue
        steps.append(i)
        preds.append(float(np.exp(query_lp[i] + lr)))
        actuals.append(float(np.exp(query_lp[i + H])))
        qvecs.append([
            query_lp[i] - query_lp[i - 1],
            query_lp[i - 1] - query_lp[i - 2],
        ])
    return (np.array(steps), np.array(preds),
            np.array(actuals), np.array(qvecs))


# ── вспомогательная: парсинг дат ──────────────────────────────────────────────

def parse_dates(date_strings):
    return [datetime.strptime(s[:16], "%Y-%m-%d %H:%M") for s in date_strings]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ticker, interval = "SBER", "10m"
    log_highs, log_lows, closes, dates = load_candles(ticker, interval)
    print(f"Свечей: {len(dates)}")

    query_lp, query_conf, query_dirs = build_zigzag(
        log_highs, log_lows, dates, T_QUERY)
    pool_lp, pool_conf, pool_dirs = build_zigzag(
        log_highs, log_lows, dates, T_POOL)
    print(f"T_query={T_QUERY*100:.0f}%: {len(query_lp)} пивотов")
    print(f"T_pool ={T_POOL*100:.1f}%:  {len(pool_lp)} пивотов")

    steps, preds, actuals, qvecs = run_walkforward(
        query_lp, query_conf, query_dirs,
        pool_lp, pool_conf, pool_dirs,
    )
    errors  = np.abs(preds - actuals)
    dz      = float(np.mean(np.abs(np.diff(actuals))))
    rel_err = errors / dz
    rmae    = errors.mean() / dz
    print(f"Прогнозов: {len(steps)}  rMAE={rmae:.4f}")

    # даты confirm для шагов walk-forward
    step_dates  = parse_dates(query_conf[steps])
    # даты actual (следующий пивот)
    target_dates = parse_dates(query_conf[steps + H])

    # ── рисунок ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(f"LWR зигзаг-прогноз  SBER 10m  T={T_QUERY*100:.0f}%  "
                 f"m={M_EMBED} K={K_LWR}  rMAE={rmae:.4f}", fontsize=13)

    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=0.38, wspace=0.3)
    ax_price  = fig.add_subplot(gs[0, :])   # верхняя панель — всю ширину
    ax_space  = fig.add_subplot(gs[1, 0])   # нижний левый — событийное пространство
    ax_err    = fig.add_subplot(gs[1, 1])   # нижний правый — ошибка по времени

    # ── 1. Цены: зигзаг + прогнозы ───────────────────────────────────────────
    # Окно: 2018-2026 (тут достаточно событий и читаемо)
    WIN_START = "2018-01-01"
    WIN_END   = "2026-07-01"

    all_pivot_dates = parse_dates(query_conf)
    all_pivot_prices = np.exp(query_lp)

    # маска зигзага в окне
    zz_mask = np.array([WIN_START <= d <= WIN_END for d in query_conf])
    zz_dates  = [all_pivot_dates[i] for i in range(len(all_pivot_dates)) if zz_mask[i]]
    zz_prices = all_pivot_prices[zz_mask]

    ax_price.plot(zz_dates, zz_prices,
                  'o-', color='steelblue', lw=1.0, ms=3.5,
                  label=f'Зигзаг T={T_QUERY*100:.0f}% (пивоты)', zorder=3)

    # прогнозные точки: рисуем в дате target (следующий пивот) с цветом по ошибке
    wf_mask = np.array([WIN_START <= d.strftime("%Y-%m-%d") <= WIN_END
                        for d in target_dates])
    td_win  = [target_dates[i] for i in range(len(target_dates)) if wf_mask[i]]
    pr_win  = preds[wf_mask]
    er_win  = rel_err[wf_mask]
    ac_win  = actuals[wf_mask]

    sc = ax_price.scatter(td_win, pr_win, c=er_win, cmap='RdYlGn_r',
                          vmin=0, vmax=3, s=18, zorder=4,
                          label='LWR прогноз (цвет=|ош|/dz)')
    plt.colorbar(sc, ax=ax_price, shrink=0.7, label='|ошибка|/dz')

    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_price.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax_price.set_ylabel("Цена, ₽")
    ax_price.set_title("Зигзаг T_query=4%  и  LWR прогнозы следующего пивота")
    ax_price.legend(fontsize=8, loc='upper left')
    ax_price.grid(True, alpha=0.2)

    # ── 2. Событийное пространство ────────────────────────────────────────────
    qv = np.array(qvecs)  # (N, 2): [lr1, lr2]

    # пул (T_pool) — фон: все доступные события в финальном состоянии
    pool_feat_all = np.array([
        [pool_lp[j] - pool_lp[j - 1],
         pool_lp[j - 1] - pool_lp[j - 2]]
        for j in range(2, len(pool_lp) - 1)
        if np.isfinite(pool_lp[j]) and np.isfinite(pool_lp[j - 1]) and np.isfinite(pool_lp[j - 2])
    ])

    ax_space.scatter(pool_feat_all[:, 0], pool_feat_all[:, 1],
                     s=4, color='lightgrey', alpha=0.5, zorder=1,
                     label=f'Пул T_pool={T_POOL*100:.1f}% ({len(pool_feat_all)} событий)')

    sc2 = ax_space.scatter(qv[:, 0], qv[:, 1], c=rel_err,
                           cmap='RdYlGn_r', vmin=0, vmax=3,
                           s=20, alpha=0.85, zorder=2,
                           label=f'Запросы T_query=4% ({len(qv)} шагов)')
    plt.colorbar(sc2, ax=ax_space, shrink=0.85, label='|ошибка|/dz')

    ax_space.set_xlabel("lr₁ = log(P[i]/P[i-1])")
    ax_space.set_ylabel("lr₂ = log(P[i-1]/P[i-2])")
    ax_space.set_title("Событийное пространство (m=2)")
    ax_space.legend(fontsize=7, loc='upper right')
    ax_space.grid(True, alpha=0.2)

    # ── 3. Ошибка по времени ─────────────────────────────────────────────────
    ax_err.scatter(step_dates, rel_err, s=6, alpha=0.5, color='steelblue')
    ax_err.axhline(rmae, color='red', lw=1.2, ls='--',
                   label=f'rMAE={rmae:.4f}')
    ax_err.axhline(1.0, color='orange', lw=0.8, ls=':',
                   label='RW baseline (1.0)')
    ax_err.set_ylim(0, None)
    ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_err.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax_err.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax_err.set_ylabel("|ошибка|/dz")
    ax_err.set_title("Относительная ошибка по времени")
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.2)

    out_path = OUT / "lwr_visualize.png"
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Рисунок → {out_path}")


if __name__ == "__main__":
    main()
