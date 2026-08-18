#!/usr/bin/env python3
"""
13_new_features.py — Поиск новых признаков для tf_oracle

Пять групп, ещё не проверявшихся:

  A. Эмпирический N_frac / N_big
       n_frac_w5   — число T_frac-событий в окне последних 5 T_big-событий
       n_frac_w10  — то же, окно 10
       nratio_w5   — n_frac_w5 / 5
       nratio_w10  — n_frac_w10 / 10
       (вычисляется при фиксированном T_frac=3.6% и при T_frac=best-mean≈3.6%)

  B. Разброс прогнозов по сетке T_frac
       pred_std    — std предсказаний (pred_rel) по всем T_frac
       pred_range  — max - min предсказаний
       pred_cv     — pred_std / |pred_mean|
       (высокий разброс = T_frac-выбор критичен; низкий = безразличен)

  C. Позиция цены
       price_pos_w50  — (lp[step] - min50) / (max50 - min50)  в лог-пространстве
       price_pos_w100 — то же, окно 100 баров

  D. Временны́е признаки (из confirm_date)
       hour        — час подтверждения (0-23)
       dow         — день недели (0=Пн, 4=Пт)

  E. Query-вектор xq
       xq_norm     — L2-норма xq
       xq0, xq1    — компоненты (signed diff-ы)
       xq_sign_agree — совпадают ли знаки xq[0] и xq[1] (momentum vs reversion)
"""
import csv
import json
import sys
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE    = Path(__file__).parent
DATA    = HERE.parent.parent.parent / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
ORACLE  = HERE.parent / "08_adaptive_fd" / "results" / "kappa_oracle_10m.csv"

T_BIG    = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
INTERVAL = sys.argv[2]        if len(sys.argv) > 2 else "10m"
T_REF    = 0.036   # фиксированный T_frac для группы A

H           = 1
P           = 3
K           = 75
MIN_HISTORY = 50
T_FRAC_GRID = np.round(np.arange(0.005, T_BIG, 0.001), 4)


# ── загрузка ─────────────────────────────────────────────────────────────────
def load_tf(name):
    with open(DATA / f"{name}.json") as f:
        raw = json.load(f)
    return (np.array([d["high"]  for d in raw], dtype=np.float64),
            np.array([d["low"]   for d in raw], dtype=np.float64),
            np.array([d["close"] for d in raw], dtype=np.float64),
            np.array([d["begin"] for d in raw]))


def find_pivots_log(highs, lows, dates, thr):
    lh, ll = np.log(highs), np.log(lows)
    vals_log, confirm_dates, dirs, confirm_bars = [], [], [], []
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
                confirm_dates.append(dates[i]); dirs.append(+1); confirm_bars.append(i)
                direction, ext_val, ext_idx = -1, ll[i], i
        else:
            if ll[i] < ext_val:
                ext_val, ext_idx = ll[i], i
            elif lh[i] - ext_val >= thr:
                vals_log.append(ext_val)
                confirm_dates.append(dates[i]); dirs.append(-1); confirm_bars.append(i)
                direction, ext_val, ext_idx = 1, lh[i], i
    return (np.array(vals_log), np.array(confirm_dates),
            np.array(dirs), np.array(confirm_bars))


def build_X(log_prices, p):
    n = len(log_prices)
    X = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = log_prices[i]
        for lag in range(1, p):
            X[i, lag] = log_prices[i - lag + 1] - log_prices[i - lag]
    return X


def load_oracle(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if k != "step" else int(v)
                         for k, v in row.items()})
    return rows


def causal_pool(confirm_date, dir_query, dt_f1, Xf1, dir_f1, y_rel):
    ce    = int(np.searchsorted(dt_f1, confirm_date, side='left'))
    rng   = np.arange(P - 1, min(ce, len(dt_f1) - H))
    if len(rng) == 0:
        return None, None
    valid = ~np.any(np.isnan(Xf1[rng]), axis=1) & ~np.isnan(y_rel[rng])
    idx   = rng[valid]
    if len(idx) < P + 2:
        return None, None
    idx_d = idx[dir_f1[idx] == dir_query]
    if len(idx_d) < P + 2:
        idx_d = idx
    return Xf1[idx_d][:, 1:], y_rel[idx_d]


def lwr_predict(xq, X_pool, y_pool):
    dists = np.linalg.norm(X_pool - xq, axis=1)
    keff  = min(K, len(dists))
    knn   = np.argsort(dists)[:keff]
    X_sel, y_sel, d = X_pool[knn], y_pool[knn], dists[knn]
    xi = d.max()
    if xi < 1e-12:
        return float(y_sel.mean())
    w  = np.exp(-0.5 * (d / xi) ** 2)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(keff), X_sel]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, y_sel * ws, rcond=None)
    return float(c[0] + c[1:] @ xq)


# ── основной проход ───────────────────────────────────────────────────────────
def run(oracle_rows, lp_big, confirm_big, dir_big, confirm_bars_big,
        lh, ll, lc, pool_cache, pool_ref):
    grid  = np.array(sorted(pool_cache.keys()))
    X_big = build_X(lp_big, P)

    lp_lc = np.log(lc)   # log(close) для позиции цены

    records = []
    n_done  = 0

    # подготовим confirm_bars для T_frac_ref (группа A)
    lp_ref, conf_ref, dir_ref, Xref, yref = pool_ref

    for rec in oracle_rows:
        step = rec["step"]
        if step < MIN_HISTORY or np.any(np.isnan(X_big[step])):
            continue

        bar = int(confirm_bars_big[step])
        xq  = X_big[step][1:]

        # ── A. N_frac / N_big ─────────────────────────────────────────────
        # число T_ref событий в окне [confirm_bar[step-W]..confirm_bar[step]]
        def nfrac_in_window(W):
            if step < W:
                return np.nan
            bar_from = int(confirm_bars_big[step - W])
            bar_to   = int(confirm_bars_big[step])
            # сколько подтверждённых T_ref событий попало в этот диапазон баров
            # ищем по confirm_bars T_ref (нет confirm_bars в pool_ref — строим)
            n = np.sum((conf_ref < confirm_big[step]) &
                       (conf_ref >= confirm_big[step - W]))
            return float(n)

        nfrac_w5  = nfrac_in_window(5)
        nfrac_w10 = nfrac_in_window(10)

        # ── B. Разброс прогнозов по сетке T_frac ─────────────────────────
        preds = []
        for tf in grid:
            lp_f1, conf_f1, dir_f1, Xf1, y_rel = pool_cache[tf]
            X_pool, y_pool = causal_pool(
                confirm_big[step], int(dir_big[step]),
                conf_f1, Xf1, dir_f1, y_rel
            )
            if X_pool is None:
                continue
            p_rel = lwr_predict(xq, X_pool, y_pool)
            if np.isfinite(p_rel):
                preds.append(p_rel)

        if len(preds) < 3:
            continue
        preds_arr = np.array(preds)
        pred_std   = float(preds_arr.std())
        pred_range = float(preds_arr.max() - preds_arr.min())
        pred_mean  = float(preds_arr.mean())
        pred_cv    = pred_std / abs(pred_mean) if abs(pred_mean) > 1e-8 else np.nan

        # ── C. Позиция цены ───────────────────────────────────────────────
        def price_pos(W):
            sl = lp_lc[max(0, bar - W):bar + 1]
            rng = sl.max() - sl.min()
            return float((lp_lc[bar] - sl.min()) / rng) if rng > 1e-8 else np.nan

        pos_50  = price_pos(50)
        pos_100 = price_pos(100)

        # ── D. Временны́е признаки ────────────────────────────────────────
        dt = confirm_big[step]
        try:
            dtp  = datetime.strptime(dt[:16], "%Y-%m-%d %H:%M")
            hour = dtp.hour
            dow  = dtp.weekday()   # 0=Пн, 4=Пт
        except Exception:
            hour, dow = np.nan, np.nan

        # ── E. Query-вектор xq ───────────────────────────────────────────
        xq_norm  = float(np.linalg.norm(xq))
        xq0      = float(xq[0])        # signed diff текущего события
        xq1      = float(xq[1]) if len(xq) > 1 else np.nan
        # знаки совпадают = тренд; противоположные = разворот
        sign_agree = float(np.sign(xq0) == np.sign(xq1)) if len(xq) > 1 else np.nan

        records.append(dict(
            step        = step,
            tf_oracle   = rec["tf_oracle"],
            kappa_ora   = rec["kappa_oracle"],
            # A
            nfrac_w5    = nfrac_w5,
            nfrac_w10   = nfrac_w10,
            nratio_w5   = nfrac_w5  / 5  if np.isfinite(nfrac_w5)  else np.nan,
            nratio_w10  = nfrac_w10 / 10 if np.isfinite(nfrac_w10) else np.nan,
            # B
            pred_std    = pred_std,
            pred_range  = pred_range,
            pred_cv     = pred_cv,
            # C
            price_pos50 = pos_50,
            price_pos100= pos_100,
            # D
            hour        = float(hour),
            dow         = float(dow),
            # E
            xq_norm     = xq_norm,
            xq0         = xq0,
            xq1         = xq1,
            sign_agree  = sign_agree,
        ))

        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{len(oracle_rows)} ...", flush=True)

    return records


FEAT_GROUPS = {
    "A — N_frac/N_big": ["nfrac_w5", "nfrac_w10", "nratio_w5", "nratio_w10"],
    "B — Разброс прогнозов": ["pred_std", "pred_range", "pred_cv"],
    "C — Позиция цены": ["price_pos50", "price_pos100"],
    "D — Время": ["hour", "dow"],
    "E — Query xq": ["xq_norm", "xq0", "xq1", "sign_agree"],
}
ALL_FEATS = [f for fs in FEAT_GROUPS.values() for f in fs]


def spearman(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 20:
        return np.nan, mask.sum()
    r, _ = stats.spearmanr(x[mask], y[mask])
    return r, mask.sum()


# ── визуализация ──────────────────────────────────────────────────────────────
def plot(records, corr_results, out_path):
    tf_or = np.array([r["tf_oracle"] for r in records])

    # топ-6 по |ρ|
    top6 = sorted(corr_results, key=lambda x: abs(x[1]) if np.isfinite(x[1]) else 0,
                  reverse=True)[:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, (feat, rho, n) in enumerate(top6):
        vals = np.array([r[feat] for r in records])
        mask = np.isfinite(vals) & np.isfinite(tf_or)
        ax = axes[i]
        ax.scatter(vals[mask], tf_or[mask] * 100, s=5, alpha=0.3, color="steelblue")
        if mask.sum() > 5:
            z  = np.polyfit(vals[mask], tf_or[mask] * 100, 1)
            xr = np.linspace(vals[mask].min(), vals[mask].max(), 50)
            ax.plot(xr, np.polyval(z, xr), color="tomato", lw=1.5)
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("tf_oracle (%)", fontsize=9)
        ax.set_title(f"{feat}  ρ={rho:+.3f}  n={n}", fontsize=9)

    plt.suptitle(f"Топ-6 новых признаков vs tf_oracle\n"
                 f"SBER {INTERVAL} | T_big={T_BIG*100:.1f}%", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"T_BIG={T_BIG*100:.1f}%  INTERVAL={INTERVAL}  T_REF={T_REF*100:.1f}%")
    print()

    h, l, c, d = load_tf(INTERVAL)
    lh, ll, lc = np.log(h), np.log(l), np.log(c)

    lp_big, confirm_big, dir_big, confirm_bars_big = find_pivots_log(h, l, d, T_BIG)
    print(f"T_BIG: {len(lp_big)} пивотов")

    oracle_rows = load_oracle(ORACLE)
    print(f"Oracle записей: {len(oracle_rows)}")

    print(f"Предвычисляем {len(T_FRAC_GRID)} зигзагов ...", flush=True)
    pool_cache = {}
    for tf in T_FRAC_GRID:
        lp_f, conf_f, dir_f, cb_f = find_pivots_log(h, l, d, tf)
        Xf1   = build_X(lp_f, P)
        y_rel = np.array([lp_f[j+H] - lp_f[j] if j+H < len(lp_f) else np.nan
                          for j in range(len(lp_f))])
        pool_cache[tf] = (lp_f, conf_f, dir_f, Xf1, y_rel)

    tf_ref_key = float(T_FRAC_GRID[np.argmin(np.abs(T_FRAC_GRID - T_REF))])
    pool_ref   = pool_cache[tf_ref_key]
    print(f"T_ref для группы A: {tf_ref_key*100:.1f}%")
    print()

    print("Считаем признаки ...", flush=True)
    records = run(oracle_rows, lp_big, confirm_big, dir_big, confirm_bars_big,
                  lh, ll, lc, pool_cache, pool_ref)
    print(f"Готово: {len(records)} записей")
    print()

    # ── корреляции ────────────────────────────────────────────────────────────
    tf_or = np.array([r["tf_oracle"] for r in records])
    corr_results = []

    for group, feats in FEAT_GROUPS.items():
        print(f"=== {group} ===")
        print(f"  {'Признак':<16}  {'Spearman ρ':>10}  {'n':>6}")
        for feat in feats:
            vals = np.array([r[feat] for r in records])
            rho, n = spearman(vals, tf_or)
            flag = " ◄" if np.isfinite(rho) and abs(rho) > 0.10 else ""
            print(f"  {feat:<16}  {rho:>+10.3f}  {n:>6}{flag}")
            corr_results.append((feat, rho, n))
        print()

    # ── итог: топ по |ρ| ─────────────────────────────────────────────────────
    print("=== ТОП признаков по |Spearman ρ| ===")
    sorted_c = sorted(corr_results, key=lambda x: abs(x[1]) if np.isfinite(x[1]) else 0,
                      reverse=True)
    for feat, rho, n in sorted_c[:10]:
        print(f"  {feat:<16}  ρ={rho:+.3f}")

    # сравнение с уже известными
    print()
    print("Для справки (предыдущие скрипты):")
    print("  fix_dm (dist)    ρ=-0.243")
    print("  amp_curr         ρ=-0.202")

    # ── CSV ──────────────────────────────────────────────────────────────────
    out_csv = RESULTS / f"new_features_{INTERVAL}.csv"
    keys_out = ["step", "tf_oracle"] + ALL_FEATS
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys_out)
        for r in records:
            w.writerow([r[k] for k in keys_out])
    print(f"\nCSV → {out_csv}")

    plot(records, corr_results,
         RESULTS / f"new_features_top6_{INTERVAL}.png")


if __name__ == "__main__":
    main()
