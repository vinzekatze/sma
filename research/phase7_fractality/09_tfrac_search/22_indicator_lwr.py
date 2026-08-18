#!/usr/bin/env python3
"""
22_indicator_lwr.py — Технические индикаторы как дополнительные признаки поиска соседей.

Идея: расширяем вектор запроса LWR тремя каузальными индикаторами:
  vol(50)  : std лог-доходностей close за 50 свечей до пивота
  rsi14    : RSI(14), нормированный: (rsi - 50) / 50
  macd_sig : MACD signal line (EMA12−EMA26 сигнал), нормированный на std

Поиск K соседей — по расширенному вектору [M_zigzag | α × 3_ind].
LWR-регрессия — только по M лог-доходностям плечей зигзага.

Оптимизация: dist²(α) = dist²_mv + α² × dist²_ind — не пересчитываем
             расстояния при смене α.

Сетка:
  α      : [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
  T_frac : [2.0%, 2.5%, 3.0%, 3.5%, 4.0%]

Данные: SBER 10m, T_BIG=4%, M=2, K=50, H=1.

Каузальность: индикаторы вычисляются только по свечам с begin <= confirm_date пивота.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

HERE    = Path(__file__).parent
DATA    = HERE.parent.parent.parent / "data" / "candles" / "SBER"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

T_BIG  = 0.04
M      = 2
K      = 50
H      = 1
MIN_HISTORY = 20

ALPHA_GRID  = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
TFRAC_GRID  = [0.020, 0.025, 0.030, 0.035, 0.040]

RSI_PERIOD  = 14
VOL_PERIOD  = 50
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIG    = 9


# ── загрузка свечей ───────────────────────────────────────────────────────────

def load_candles(path):
    with open(path) as f:
        raw = json.load(f)
    lh = np.log(np.array([c["high"]  for c in raw], dtype=np.float64))
    ll = np.log(np.array([c["low"]   for c in raw], dtype=np.float64))
    lc = np.log(np.array([c["close"] for c in raw], dtype=np.float64))
    dt = np.array([c["begin"] for c in raw])
    return lh, ll, lc, dt


# ── зигзаг ────────────────────────────────────────────────────────────────────

def build_zigzag(lh, ll, dt, thr):
    lp, cd, dirs = [], [], []
    cur_dir = 0
    ext = (lh[0] + ll[0]) / 2.0
    for i in range(len(lh)):
        if cur_dir == 0:
            if lh[i] - ext >= thr:
                cur_dir, ext = 1, lh[i]
            elif ext - ll[i] >= thr:
                cur_dir, ext = -1, ll[i]
        elif cur_dir == 1:
            if lh[i] > ext:
                ext = lh[i]
            elif ext - ll[i] >= thr:
                lp.append(ext); cd.append(dt[i]); dirs.append(+1)
                cur_dir, ext = -1, ll[i]
        else:
            if ll[i] < ext:
                ext = ll[i]
            elif lh[i] - ext >= thr:
                lp.append(ext); cd.append(dt[i]); dirs.append(-1)
                cur_dir, ext = 1, lh[i]
    return np.array(lp), np.array(cd), np.array(dirs, dtype=np.int8)


# ── технические индикаторы ────────────────────────────────────────────────────

def _ema(x, span):
    k = 2.0 / (span + 1)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1.0 - k)
    return out


def compute_vol(lc_slice):
    if len(lc_slice) < 2:
        return 0.0
    n = min(VOL_PERIOD + 1, len(lc_slice))
    return float(np.std(np.diff(lc_slice[-n:])))


def compute_rsi(c_slice):
    if len(c_slice) < RSI_PERIOD + 1:
        return 50.0
    delta = np.diff(c_slice)
    gains  = np.where(delta > 0,  delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    ag = gains[:RSI_PERIOD].mean()
    al = losses[:RSI_PERIOD].mean()
    for i in range(RSI_PERIOD, len(delta)):
        ag = (ag * (RSI_PERIOD - 1) + gains[i])  / RSI_PERIOD
        al = (al * (RSI_PERIOD - 1) + losses[i]) / RSI_PERIOD
    if al < 1e-12:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def compute_macd_signal(c_slice):
    need = MACD_SLOW + MACD_SIG
    if len(c_slice) < need:
        return 0.0
    e12  = _ema(c_slice, MACD_FAST)
    e26  = _ema(c_slice, MACD_SLOW)
    macd = e12 - e26
    sig  = _ema(macd, MACD_SIG)
    return float(sig[-1])


def compute_indicators(bar_idx, lc):
    """Три каузальных индикатора по свечам [0..bar_idx] включительно."""
    c = np.exp(lc[:bar_idx + 1])
    v = compute_vol(lc[:bar_idx + 1])
    r = compute_rsi(c)
    ms = compute_macd_signal(c)
    return np.array([v, r, ms], dtype=np.float64)


def find_bar_idx(dt_candles, confirm_dt):
    """Индекс последней свечи с begin <= confirm_dt."""
    return max(0, int(np.searchsorted(dt_candles, confirm_dt, side="right")) - 1)


# ── пул и LWR ─────────────────────────────────────────────────────────────────

def build_pool(lp, cd, dirs, lc, dt_candles, m):
    """
    Возвращает (features_mv, indicators_raw, targets, dirs, dates).
    features_mv   : M лог-доходностей плечей [n, M]
    indicators_raw: 3 индикатора [n, 3] (ненормированные)
    """
    rows, inds, tgts, row_dirs, row_dates = [], [], [], [], []
    for j in range(m, len(lp) - 1):
        feat = np.array([lp[j - lag] - lp[j - lag - 1] for lag in range(m)])
        if not np.all(np.isfinite(feat)):
            continue
        tgt = lp[j + 1] - lp[j]
        if not np.isfinite(tgt):
            continue
        bidx = find_bar_idx(dt_candles, cd[j])
        ind  = compute_indicators(bidx, lc)
        rows.append(feat)
        inds.append(ind)
        tgts.append(tgt)
        row_dirs.append(dirs[j])
        row_dates.append(cd[j])
    if not rows:
        return None
    return (np.array(rows, dtype=np.float64),
            np.array(inds, dtype=np.float64),
            np.array(tgts, dtype=np.float64),
            np.array(row_dirs, dtype=np.int8),
            np.array(row_dates))


def normalize_indicators(ind_big, ind_frac):
    """Нормируем по std T_BIG пивотов. Возвращает (big_norm, frac_norm, scale)."""
    scale = np.std(ind_big, axis=0)
    scale[scale < 1e-12] = 1.0
    return ind_big / scale, ind_frac / scale, scale


def lwr_predict(qv, fm_mv, tgt, indices):
    feats = fm_mv[indices]
    dists = np.linalg.norm(feats - qv, axis=1)
    d_max = dists.max()
    if d_max < 1e-12:
        return float(tgt[indices].mean())
    w  = np.exp(-0.5 * (dists / d_max) ** 2)
    ws = np.sqrt(w)
    A  = np.column_stack([np.ones(len(indices)), feats]) * ws[:, None]
    c, *_ = np.linalg.lstsq(A, tgt[indices] * ws, rcond=None)
    return float(c[0] + c[1:] @ qv)


# ── walk-forward ──────────────────────────────────────────────────────────────

def run_walk_forward(lp_big, cd_big, dir_big, qv_all, qind_all,
                     pool_mv, pool_ind_norm, pool_tgt, pool_dirs, pool_dates):
    """
    Для каждого (T_frac) выполняет walk-forward по всем α одновременно.

    Возвращает dict alpha → list of (err, actual).
    """
    n_big = len(lp_big)
    results = {a: [] for a in ALPHA_GRID}

    for step in range(max(MIN_HISTORY, M), n_big - H):
        qv   = qv_all[step]
        qind = qind_all[step]
        if qv is None:
            continue

        qdir      = int(dir_big[step])
        actual_lp = lp_big[step + H]

        ce = int(np.searchsorted(pool_dates, cd_big[step], side="left"))
        if ce < K + 1:
            continue

        mv_s   = pool_mv[:ce]
        ind_s  = pool_ind_norm[:ce]
        tgt_s  = pool_tgt[:ce]
        dirs_s = pool_dirs[:ce]

        mask     = dirs_s == qdir
        cand_idx = np.where(mask)[0]
        if len(cand_idx) < K:
            continue

        # dist² по M-части и 3-части — один раз
        diff_mv  = mv_s[cand_idx] - qv        # [n_cand, M]
        diff_ind = ind_s[cand_idx] - qind      # [n_cand, 3]
        d2_mv  = np.einsum("ij,ij->i", diff_mv,  diff_mv)
        d2_ind = np.einsum("ij,ij->i", diff_ind, diff_ind)

        for alpha in ALPHA_GRID:
            d2 = d2_mv + alpha ** 2 * d2_ind
            top_local = np.argpartition(d2, K - 1)[:K]
            top_k     = cand_idx[top_local]

            pred_lr = lwr_predict(qv, mv_s, tgt_s, top_k)
            err     = np.exp(lp_big[step] + pred_lr) - np.exp(actual_lp)
            results[alpha].append((float(err), float(np.exp(actual_lp))))

    return results


# ── метрика ───────────────────────────────────────────────────────────────────

def rmae(records, dz):
    if not records:
        return np.nan
    errs = np.abs([r[0] for r in records])
    return float(np.mean(errs) / dz)


# ── графики ───────────────────────────────────────────────────────────────────

def plot_heatmap(df_pivot, baseline_rmae):
    fig, ax = plt.subplots(figsize=(9, 5))
    data = df_pivot.values
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                   vmin=data.min() * 0.97, vmax=data.max() * 1.01)
    ax.set_xticks(range(len(ALPHA_GRID)))
    ax.set_xticklabels([str(a) for a in ALPHA_GRID])
    ax.set_yticks(range(len(TFRAC_GRID)))
    ax.set_yticklabels([f"{t*100:.1f}%" for t in TFRAC_GRID])
    ax.set_xlabel("α (вес индикаторов)")
    ax.set_ylabel("T_frac")
    ax.set_title(f"rMAE: технические индикаторы в поиске соседей  (SBER 10m, T_big=4%)\n"
                 f"baseline (α=0, T_frac=3%) = {baseline_rmae:.4f}")
    for i in range(len(TFRAC_GRID)):
        for j in range(len(ALPHA_GRID)):
            v = data[i, j]
            delta = (v - baseline_rmae) / baseline_rmae * 100
            clr = "white" if abs(v - data.mean()) > 0.5 * data.std() else "black"
            ax.text(j, i, f"{v:.4f}\n{delta:+.1f}%", ha="center", va="center",
                    fontsize=7.5, color=clr)
    plt.colorbar(im, ax=ax, label="rMAE")
    fig.tight_layout()
    fig.savefig(RESULTS / "indicator_lwr_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → indicator_lwr_heatmap.png")


def plot_alpha_curves(df_all, baseline_rmae):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(TFRAC_GRID)))
    for (tf, color) in zip(TFRAC_GRID, colors):
        sub = df_all[df_all["T_frac"] == tf].sort_values("alpha")
        ax.plot(sub["alpha"], sub["rMAE"], marker="o", color=color,
                label=f"T_frac={tf*100:.1f}%")
    ax.axhline(baseline_rmae, color="black", ls="--", lw=1.2,
               label=f"baseline {baseline_rmae:.4f}")
    ax.set_xlabel("α (вес индикаторов в поиске соседей)")
    ax.set_ylabel("rMAE")
    ax.set_title("rMAE vs α при разных T_frac  (SBER 10m, T_big=4%)", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "indicator_lwr_alpha.png", dpi=150)
    plt.close(fig)
    print("  → indicator_lwr_alpha.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    lh, ll, lc, dt_candles = load_candles(DATA / "10m.json")
    print(f"Свечей: {len(dt_candles)}  ({dt_candles[0][:10]} … {dt_candles[-1][:10]})")

    # T_BIG зигзаг
    lp_big, cd_big, dir_big = build_zigzag(lh, ll, dt_candles, T_BIG)
    print(f"T_BIG={T_BIG*100:.0f}%  пивотов: {len(lp_big)}")

    # Предвычисляем qv и qind для T_BIG (кэш)
    print("Вычисляем векторы запроса T_BIG...")
    qv_all   = [None] * len(lp_big)
    qind_all = [None] * len(lp_big)
    for step in range(max(MIN_HISTORY, M), len(lp_big) - H):
        qv = np.array([lp_big[step - lag] - lp_big[step - lag - 1]
                       for lag in range(M)])
        if not np.all(np.isfinite(qv)):
            continue
        bidx = find_bar_idx(dt_candles, cd_big[step])
        qind_all[step] = compute_indicators(bidx, lc)
        qv_all[step]   = qv

    # Нормировка: считаем scale по T_BIG (для последовательности ind_big)
    ind_big_all = np.array([q for q in qind_all if q is not None])
    ind_scale   = np.std(ind_big_all, axis=0)
    ind_scale[ind_scale < 1e-12] = 1.0
    # Нормируем qind_all
    for step in range(len(qind_all)):
        if qind_all[step] is not None:
            qind_all[step] = qind_all[step] / ind_scale

    print(f"  Масштабы индикаторов: vol={ind_scale[0]:.6f}  rsi={ind_scale[1]:.4f}  macd={ind_scale[2]:.8f}")

    # Пулы T_frac
    all_results = []

    for tf in TFRAC_GRID:
        print(f"\nT_frac={tf*100:.1f}%  строим пул...")
        lp_f, cd_f, dir_f = build_zigzag(lh, ll, dt_candles, tf)
        pool = build_pool(lp_f, cd_f, dir_f, lc, dt_candles, M)
        if pool is None:
            print("  пул пуст, пропускаем")
            continue

        pool_mv, pool_ind_raw, pool_tgt, pool_dirs, pool_dates = pool
        # Нормируем индикаторы пула на те же scale
        pool_ind_norm = pool_ind_raw / ind_scale
        print(f"  строк пула: {len(pool_mv)}")

        res = run_walk_forward(
            lp_big, cd_big, dir_big,
            qv_all, qind_all,
            pool_mv, pool_ind_norm, pool_tgt, pool_dirs, pool_dates
        )

        # Единый dz от T_BIG actuals
        all_acts = []
        for records in res.values():
            if records:
                all_acts = [r[1] for r in records]
                break
        a = np.asarray(all_acts, dtype=float)
        pers = np.abs(a[2:] - a[:-2]) if len(a) >= 3 else np.abs(np.diff(a))
        dz = float(np.mean(pers)) if len(pers) > 0 else 1.0

        for alpha, records in res.items():
            r = rmae(records, dz)
            all_results.append({
                "T_frac": tf,
                "alpha":  alpha,
                "rMAE":   r,
                "n":      len(records),
            })
            mark = " ←" if tf == 0.030 and alpha == 0.0 else ""
            print(f"  α={alpha:.1f}  rMAE={r:.4f}  n={len(records)}{mark}")

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(RESULTS / "indicator_lwr_results.csv", index=False)

    # baseline (α=0, T_frac=3%)
    baseline_row = df_all[(df_all["T_frac"] == 0.030) & (df_all["alpha"] == 0.0)]
    baseline_rmae = float(baseline_row["rMAE"].values[0]) if len(baseline_row) else np.nan
    print(f"\nBaseline (α=0, T_frac=3%): rMAE={baseline_rmae:.4f}")

    # Лучшая комбинация
    best_row = df_all.loc[df_all["rMAE"].idxmin()]
    print(f"Лучшая: α={best_row['alpha']:.1f}  T_frac={best_row['T_frac']*100:.1f}%  "
          f"rMAE={best_row['rMAE']:.4f}  "
          f"Δ={(best_row['rMAE'] - baseline_rmae) / baseline_rmae * 100:+.2f}%")

    # Тепловая карта
    df_pivot = df_all.pivot(index="T_frac", columns="alpha", values="rMAE")
    plot_heatmap(df_pivot, baseline_rmae)
    plot_alpha_curves(df_all, baseline_rmae)

    print("\nГотово.")


if __name__ == "__main__":
    main()
