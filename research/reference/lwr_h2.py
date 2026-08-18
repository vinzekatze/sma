#!/usr/bin/env python3
"""
lwr_h2.py — Двухшаговый прогноз зигзага (H=2 пивота вперёд).

Четыре метода:
  H1   — базовый LWR H=1 (воспроизведение)
  ITER — итеративный: H=1 модель применяется дважды;
         второй шаг ищет соседей в сдвинутом пуле с [lr1_pred, lr1_cur]
  DIR  — прямой: те же признаки [lr1, lr2], таргет = lp[j+2]-lp[j]
  AUG  — augmented vector: первый шаг (H=1) предсказывает lr1_pred,
         затем строим 3D запрос [lr1_pred, lr1_cur, lr2_cur] и ищем
         в 3D пуле [lp[j+1]-lp[j], lp[j]-lp[j-1], lp[j-1]-lp[j-2]]
         c таргетом lp[j+2]-lp[j+1].

Идея AUG: не сдвигать вектор задержки (как в ITER), а добавить
спрогнозированную точку как новое измерение, сохранив текущий контекст.

Каузальность:
  H=1  : таргет j+1 подтверждён → j < j_max-1
  H=2  : таргет j+2 подтверждён → j < j_max-2
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
MIN_HISTORY = 50
MIN_DIR     = M_EMBED + 2

DATA = Path(__file__).parent.parent.parent / "data" / "candles"
OUT  = Path(__file__).parent


# ── загрузка ──────────────────────────────────────────────────────────────────

def load_candles(ticker, interval):
    path = DATA / ticker / f"{interval}.json"
    with open(path) as fh:
        raw = json.load(fh)
    log_highs = np.log(np.array([c["high"]  for c in raw], dtype=np.float64))
    log_lows  = np.log(np.array([c["low"]   for c in raw], dtype=np.float64))
    dates     = np.array([c["begin"] for c in raw])
    return log_highs, log_lows, dates


def build_zigzag(log_highs, log_lows, dates, threshold):
    lp, conf, dirs = [], [], []
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
                lp.append(extreme); conf.append(dates[bar]); dirs.append(+1)
                direction, extreme = -1, log_lows[bar]
        else:
            if log_lows[bar] < extreme:
                extreme = log_lows[bar]
            elif log_highs[bar] - extreme >= threshold:
                lp.append(extreme); conf.append(dates[bar]); dirs.append(-1)
                direction, extreme = 1, log_highs[bar]
    return np.array(lp), np.array(conf), np.array(dirs, dtype=np.int8)


# ── LWR ядро ─────────────────────────────────────────────────────────────────

def lwr_fit(query_vec, feat, tgt, k):
    """Взвешенная OLS по k ближайшим соседям. Возвращает lr или None."""
    dists = np.linalg.norm(feat - query_vec, axis=1)
    if len(dists) < k:
        return None
    nn   = np.argpartition(dists, k - 1)[:k]
    d    = dists[nn]
    dmax = d.max()
    if dmax < 1e-12:
        return float(tgt[nn].mean())
    w  = np.exp(-0.5 * (d / dmax) ** 2)
    sw = np.sqrt(w)
    A  = np.column_stack([np.ones(k), feat[nn]]) * sw[:, None]
    c, *_ = np.linalg.lstsq(A, tgt[nn] * sw, rcond=None)
    return float(c[0] + c[1:] @ query_vec)


def pool_cutoff(pool_lp, pool_conf, pool_dirs, conf_date,
                m, h, target_offset=1):
    """
    Строит пул признаков и таргетов с каузальной отсечкой.

    h           — горизонт для C3 (j+h < j_max)
    target_offset — откуда берётся таргет:
        1  → target = lp[j+target_offset] - lp[j]
        но для ITER-шаг2 нам нужен lp[j+1]-lp[j] как признак,
        а lp[j+2]-lp[j+1] как таргет → используем shift=True
    """
    j_max = int(np.searchsorted(pool_conf, conf_date, side='left'))
    valid = np.arange(m, j_max - h)
    if len(valid) == 0:
        return None, None, None

    feat = np.array([
        [pool_lp[j - lag] - pool_lp[j - lag - 1] for lag in range(m)]
        for j in valid
    ])
    tgt  = pool_lp[valid + h] - pool_lp[valid]
    dirs = pool_dirs[valid]

    ok = np.all(np.isfinite(feat), axis=1) & np.isfinite(tgt)
    return feat[ok], tgt[ok], dirs[ok]


def pool_cutoff_shifted(pool_lp, pool_conf, pool_dirs, conf_date, m):
    """
    Пул для ITER шага 2: признаки = [lp[j+1]-lp[j], lp[j]-lp[j-1], ...],
    таргет = lp[j+2] - lp[j+1].
    Каузальность: j+2 < j_max → j < j_max-2.
    """
    j_max = int(np.searchsorted(pool_conf, conf_date, side='left'))
    # j+1 нужен как признак, j+2 как таргет → j <= j_max-3
    valid = np.arange(m - 1, j_max - 2)
    if len(valid) == 0:
        return None, None, None

    feat = np.array([
        [pool_lp[j + 1 - lag] - pool_lp[j - lag] for lag in range(m)]
        for j in valid
    ])
    tgt  = pool_lp[valid + 2] - pool_lp[valid + 1]
    dirs = pool_dirs[valid + 1]  # направление в точке j+1

    ok = np.all(np.isfinite(feat), axis=1) & np.isfinite(tgt)
    return feat[ok], tgt[ok], dirs[ok]


def pool_cutoff_aug(pool_lp, pool_conf, pool_dirs, conf_date, m):
    """
    Пул для AUG: признаки = [lp[j+1]-lp[j], lp[j]-lp[j-1], ..., lp[j-m+2]-lp[j-m+1]]
    (m+1 измерений: 1 "будущее" + m текущих),
    таргет = lp[j+2] - lp[j+1].
    Каузальность: j+2 < j_max.
    """
    aug_m = m + 1
    j_max = int(np.searchsorted(pool_conf, conf_date, side='left'))
    valid = np.arange(m, j_max - 2)
    if len(valid) == 0:
        return None, None, None

    # feat[0] = lp[j+1]-lp[j]  (фактический следующий шаг — "будущее" пула)
    # feat[1..m] = lp[j]-lp[j-1], ..., lp[j-m+2]-lp[j-m+1]
    feat = np.zeros((len(valid), aug_m))
    for row, j in enumerate(valid):
        feat[row, 0] = pool_lp[j + 1] - pool_lp[j]
        for lag in range(m):
            feat[row, lag + 1] = pool_lp[j - lag] - pool_lp[j - lag - 1]
    tgt  = pool_lp[valid + 2] - pool_lp[valid + 1]
    dirs = pool_dirs[valid]   # направление в точке j (как в H=1 запросе)

    ok = np.all(np.isfinite(feat), axis=1) & np.isfinite(tgt)
    return feat[ok], tgt[ok], dirs[ok]


# ── метрика ───────────────────────────────────────────────────────────────────

def rmae(preds, actuals):
    e = np.abs(np.array(preds) - np.array(actuals))
    dz = float(np.mean(np.abs(np.diff(actuals))))
    return float(e.mean() / dz) if dz > 1e-12 else np.nan


# ── walk-forward ──────────────────────────────────────────────────────────────

def walk_forward(query_lp, query_conf, query_dirs,
                 pool_lp, pool_conf, pool_dirs):
    n = len(query_lp)

    h1_preds, h1_actuals       = [], []
    iter_preds, iter_actuals   = [], []
    dir_preds,  dir_actuals    = [], []
    aug_preds,  aug_actuals    = [], []

    for i in range(MIN_HISTORY, n - 2):  # нужен i+2
        conf_date = query_conf[i]
        qdir      = int(query_dirs[i])

        # Вектор запроса H=1: [lr1_cur, lr2_cur]
        q1 = np.array([
            query_lp[i - lag] - query_lp[i - lag - 1]
            for lag in range(M_EMBED)
        ])

        # ── H=1 базовый ──────────────────────────────────────────────────────
        f1, t1, d1 = pool_cutoff(pool_lp, pool_conf, pool_dirs,
                                  conf_date, M_EMBED, h=1)
        if f1 is None:
            continue
        mask1 = d1 == qdir
        if mask1.sum() < max(MIN_DIR, K_LWR):
            continue
        lr1 = lwr_fit(q1, f1[mask1], t1[mask1], K_LWR)
        if lr1 is None:
            continue

        actual_h1 = float(np.exp(query_lp[i + 1]))
        h1_preds.append(float(np.exp(query_lp[i] + lr1)))
        h1_actuals.append(actual_h1)

        actual_h2 = float(np.exp(query_lp[i + 2]))

        # ── ITER: второй шаг со сдвинутым пулом ─────────────────────────────
        # Признак: [lr1_pred, lr1_cur]; направление на i+1 = -qdir
        q_iter = np.array([lr1, q1[0]])   # [lr1_pred, lr1_cur]
        dir_iter = -qdir                   # следующий пивот противоположного знака

        fs, ts, ds = pool_cutoff_shifted(pool_lp, pool_conf, pool_dirs,
                                          conf_date, M_EMBED)
        if fs is not None:
            mask_s = ds == dir_iter
            if mask_s.sum() >= max(MIN_DIR, K_LWR):
                lr2_iter = lwr_fit(q_iter, fs[mask_s], ts[mask_s], K_LWR)
                if lr2_iter is not None:
                    pred_h2_iter = float(np.exp(query_lp[i] + lr1 + lr2_iter))
                    iter_preds.append(pred_h2_iter)
                    iter_actuals.append(actual_h2)

        # ── DIR: прямой H=2 ──────────────────────────────────────────────────
        # Признаки те же [lr1, lr2], таргет = lp[j+2]-lp[j]
        f2, t2, d2 = pool_cutoff(pool_lp, pool_conf, pool_dirs,
                                   conf_date, M_EMBED, h=2)
        if f2 is not None:
            mask2 = d2 == qdir
            if mask2.sum() >= max(MIN_DIR, K_LWR):
                lr_dir = lwr_fit(q1, f2[mask2], t2[mask2], K_LWR)
                if lr_dir is not None:
                    dir_preds.append(float(np.exp(query_lp[i] + lr_dir)))
                    dir_actuals.append(actual_h2)

        # ── AUG: расширенный вектор [lr1_pred, lr1_cur, lr2_cur] ─────────────
        q_aug = np.array([lr1, q1[0], q1[1]])   # 3D запрос
        fa, ta, da = pool_cutoff_aug(pool_lp, pool_conf, pool_dirs,
                                      conf_date, M_EMBED)
        if fa is not None:
            mask_a = da == qdir
            if mask_a.sum() >= max(MIN_DIR, K_LWR):
                lr_aug = lwr_fit(q_aug, fa[mask_a], ta[mask_a], K_LWR)
                if lr_aug is not None:
                    # AUG предсказывает lp[j+2]-lp[j+1]; добавляем lr1 (шаг 1)
                    aug_preds.append(float(np.exp(query_lp[i] + lr1 + lr_aug)))
                    aug_actuals.append(actual_h2)

    return (h1_preds,   h1_actuals,
            iter_preds, iter_actuals,
            dir_preds,  dir_actuals,
            aug_preds,  aug_actuals)


# ── визуализация ──────────────────────────────────────────────────────────────

def plot_results(results, out_path):
    (h1p, h1a, itp, ita, drp, dra, agp, aga) = results

    methods = [
        ("H=1 базовый",    np.array(h1p),  np.array(h1a),  "steelblue"),
        ("H=2 итерат.",    np.array(itp),  np.array(ita),  "darkorange"),
        ("H=2 прямой",     np.array(drp),  np.array(dra),  "green"),
        ("H=2 augmented",  np.array(agp),  np.array(aga),  "purple"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LWR двухшаговый прогноз  SBER 10m  T=4%  m=2  K=75", fontsize=12)

    # rMAE bar chart
    ax = axes[0]
    names, rmae_vals, colors = [], [], []
    for label, preds, acts, color in methods:
        if len(preds) < 5:
            continue
        r = rmae(preds, acts)
        names.append(f"{label}\n(n={len(preds)})")
        rmae_vals.append(r)
        colors.append(color)
        print(f"{label:20s}  n={len(preds):4d}  rMAE={r:.4f}")

    bars = ax.bar(names, rmae_vals, color=colors, alpha=0.8, width=0.5)
    ax.axhline(1.0, color='black', lw=0.8, ls=':', label='RW baseline')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
    ax.set_ylabel("rMAE")
    ax.set_title("rMAE по методу")
    ax.set_ylim(0, max(rmae_vals) * 1.15)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # predicted vs actual scatter (H=2 методы)
    ax2 = axes[1]
    for label, preds, acts, color in methods[1:]:  # пропускаем H=1
        if len(preds) < 5:
            continue
        ax2.scatter(acts, preds, s=6, alpha=0.35, color=color, label=label)

    lims = [
        min(np.concatenate([np.array(dra), np.array(aga), np.array(ita)])),
        max(np.concatenate([np.array(dra), np.array(aga), np.array(ita)])),
    ]
    ax2.plot(lims, lims, 'k--', lw=0.8, label='Идеал')
    ax2.set_xlabel("Факт H=2")
    ax2.set_ylabel("Прогноз H=2")
    ax2.set_title("Факт vs прогноз (H=2)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Рисунок → {out_path}")


def main():
    log_highs, log_lows, dates = load_candles("SBER", "10m")
    print(f"Свечей: {len(dates)}")

    query_lp, query_conf, query_dirs = build_zigzag(
        log_highs, log_lows, dates, T_QUERY)
    pool_lp, pool_conf, pool_dirs = build_zigzag(
        log_highs, log_lows, dates, T_POOL)
    print(f"T_query={T_QUERY*100:.0f}%: {len(query_lp)} пивотов")
    print(f"T_pool ={T_POOL*100:.1f}%:  {len(pool_lp)} пивотов\n")

    results = walk_forward(
        query_lp, query_conf, query_dirs,
        pool_lp, pool_conf, pool_dirs,
    )
    plot_results(results, OUT / "lwr_h2.png")


if __name__ == "__main__":
    main()
