#!/usr/bin/env python3
"""
d2_causal_walk.py — D₂ в каузальном walk-forward режиме.

Для каждого шага t: берём пул [0..t-1], нормируем per-step z-score
(как в прогнозе), считаем D₂ по нормированному облаку.

Показывает, как фрактальная размерность меняется со временем —
то есть как меняется геометрия пространства поиска соседей.
"""

import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial.distance import pdist

HERE     = os.path.dirname(__file__)
DATA_10M = os.path.join(HERE, "../../../data/candles/SBER/10m.json")
RES_DIR  = os.path.join(HERE, "results")
FIG_DIR  = os.path.join(HERE, "figures")

P        = 3          # размерность вложения (оптимум по D₂)
MIN_POOL = 80         # минимальный пул для надёжного D₂
D2_N_R   = 30         # число r-точек для C(r)


# ── данные ────────────────────────────────────────────────────────────────────

def load_10m():
    with open(DATA_10M) as f:
        raw = json.load(f)
    highs  = np.array([d["high"]  for d in raw], dtype=np.float64)
    lows   = np.array([d["low"]   for d in raw], dtype=np.float64)
    dates  = np.array([d["begin"] for d in raw])
    return highs, lows, dates


def find_pivots(highs, lows, thr):
    prices, idxs, types = [], [], []
    direction = 0
    ext_val = (highs[0] + lows[0]) / 2.0
    ext_idx = 0; ext_type = 0
    for i in range(len(highs)):
        if direction == 0:
            if highs[i] - ext_val >= thr * ext_val:
                direction = 1; ext_val = highs[i]; ext_idx = i; ext_type = 1
            elif ext_val - lows[i] >= thr * ext_val:
                direction = -1; ext_val = lows[i]; ext_idx = i; ext_type = -1
        elif direction == 1:
            if highs[i] > ext_val:
                ext_val = highs[i]; ext_idx = i; ext_type = 1
            elif ext_val - lows[i] >= thr * ext_val:
                prices.append(ext_val); idxs.append(ext_idx); types.append(ext_type)
                direction = -1; ext_val = lows[i]; ext_idx = i; ext_type = -1
        else:
            if lows[i] < ext_val:
                ext_val = lows[i]; ext_idx = i; ext_type = -1
            elif highs[i] - ext_val >= thr * ext_val:
                prices.append(ext_val); idxs.append(ext_idx); types.append(ext_type)
                direction = 1; ext_val = highs[i]; ext_idx = i; ext_type = 1
    return (np.array(prices, dtype=np.float64),
            np.array(idxs, dtype=np.int64),
            np.array(types, dtype=np.int8))


# ── вложение ──────────────────────────────────────────────────────────────────

def build_hybrid(prices, p):
    n  = len(prices)
    lp = np.log(np.maximum(prices, 1e-12))
    X  = np.full((n, p), np.nan)
    for i in range(p - 1, n):
        X[i, 0] = prices[i]
        for lag in range(1, p):
            X[i, lag] = lp[i - lag + 1] - lp[i - lag]
    return X


# ── D₂ для одного облака ──────────────────────────────────────────────────────

def d2_from_pool(X_norm, n_r=D2_N_R):
    """
    D₂ по нормированному облаку X_norm.
    Возвращает (D2, качество) — качество: число точек в линейном участке.
    """
    if len(X_norm) < 20:
        return np.nan, 0

    d_flat = pdist(X_norm).astype(np.float32)
    d_pos  = d_flat[d_flat > 1e-6]
    if len(d_pos) < 10:
        return np.nan, 0

    r_min = float(np.percentile(d_pos, 5))
    r_max = float(np.percentile(d_pos, 80))
    if r_min >= r_max:
        return np.nan, 0

    r_arr = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    C_arr = np.array([np.mean(d_flat < r) for r in r_arr], dtype=np.float32)

    mask = C_arr > 0
    if mask.sum() < 4:
        return np.nan, 0

    lr = np.log10(r_arr[mask])
    lc = np.log10(C_arr[mask])

    lo = int(len(lr) * 0.2)
    hi = int(len(lr) * 0.8)
    if hi - lo < 3:
        lo, hi = 0, len(lr)

    coef = np.polyfit(lr[lo:hi], lc[lo:hi], 1)
    return float(coef[0]), hi - lo


# ── walk-forward ──────────────────────────────────────────────────────────────

def run_causal_d2(prices, bar_dates, p=P, min_pool=MIN_POOL):
    """
    Для каждого шага t: берём пул X[0..t-1], нормируем per-step z-score,
    вычисляем D₂. Возвращает DataFrame.
    """
    X_full = build_hybrid(prices, p)
    # убираем NaN-строки (первые p-1)
    valid  = np.where(~np.any(np.isnan(X_full), axis=1))[0]
    X_v    = X_full[valid]
    dates_v = bar_dates[valid]
    prices_v = prices[valid]
    N = len(X_v)

    rows = []
    for t in range(min_pool, N):
        pool = X_v[:t]                          # только история
        mu   = pool.mean(0)
        sig  = pool.std(0)
        sig  = np.where(sig < 1e-10, 1.0, sig)
        Xn   = (pool - mu) / sig                # per-step нормировка

        d2, quality = d2_from_pool(Xn)
        rows.append({
            "t":        t,
            "date":     dates_v[t],
            "price":    prices_v[t],
            "pool_n":   t,
            "D2":       d2,
            "quality":  quality,
            # нормировка пула для диагностики
            "mu_price": mu[0],
            "sig_price": sig[0],
            "mu_lr1":   mu[1] if p > 1 else np.nan,
            "sig_lr1":  sig[1] if p > 1 else np.nan,
        })

    return pd.DataFrame(rows)


# ── графики ───────────────────────────────────────────────────────────────────

def plot_d2_vs_price(df, title="SBER 10m  T=4%  p=3"):
    """
    Два subplot: цена пивотов сверху, D₂ снизу.
    Подсвечиваем зоны где D₂ резко меняется.
    """
    df = df.dropna(subset=["D2"]).copy()
    dates = pd.to_datetime(df["date"])

    # скользящее среднее D₂ (окно 20) для плавной линии
    d2_smooth = df["D2"].rolling(20, center=True, min_periods=5).mean()

    # зоны значительного изменения: D₂ > 75-й перцентиль или < 25-й
    q25, q75 = df["D2"].quantile(0.25), df["D2"].quantile(0.75)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2, 2, 1]})
    fig.suptitle(f"D₂ каузальный walk-forward  |  {title}", fontsize=11)

    # ── subplot 1: цена пивотов ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(dates, df["price"], color="#1f77b4", lw=1.0, alpha=0.8)
    # маркеры где D₂ высокая (сложная геометрия)
    mask_hi = df["D2"] > q75
    ax.scatter(dates[mask_hi], df["price"][mask_hi],
               color="#d62728", s=20, zorder=3, label=f"D₂ > {q75:.2f}", alpha=0.7)
    # маркеры где D₂ низкая (простая геометрия)
    mask_lo = df["D2"] < q25
    ax.scatter(dates[mask_lo], df["price"][mask_lo],
               color="#2ca02c", s=20, zorder=3, label=f"D₂ < {q25:.2f}", alpha=0.7)
    ax.set_ylabel("Цена пивота", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.2)

    # ── subplot 2: D₂ во времени ─────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(dates, df["D2"], alpha=0.15, color="#1f77b4")
    ax.plot(dates, df["D2"], color="#aaaaaa", lw=0.6, alpha=0.6)
    ax.plot(dates, d2_smooth, color="#1f77b4", lw=1.8, label="D₂ (скол. среднее 20)")
    ax.axhline(q25, color="#2ca02c", ls="--", lw=1.0, label=f"Q25={q25:.2f}")
    ax.axhline(q75, color="#d62728", ls="--", lw=1.0, label=f"Q75={q75:.2f}")
    ax.axhline(df["D2"].median(), color="#ff7f0e", ls="-.", lw=1.2,
               label=f"медиана={df['D2'].median():.2f}")
    ax.set_ylabel("D₂", fontsize=8)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.2)

    # ── subplot 3: sig_lr1 (нормировочная нестабильность) ────────────────────
    ax = axes[2]
    ax.plot(dates, df["sig_lr1"], color="#9467bd", lw=1.2, alpha=0.85)
    ax.set_ylabel("σ(lr₁) пула", fontsize=8)
    ax.set_xlabel("Дата пивота", fontsize=8)
    ax.grid(alpha=0.2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=7, rotation=30)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "d2_causal_walk.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def plot_c_curves(df, prices_all, bar_dates_all, p=P,
                  n_samples=6, title="SBER 10m  T=4%  p=3"):
    """
    C(r) кривые для нескольких точек: 3 с низким D₂ и 3 с высоким.
    Наглядно показывает разницу геометрии облака.
    """
    df_valid = df.dropna(subset=["D2"]).copy()
    q25 = df_valid["D2"].quantile(0.25)
    q75 = df_valid["D2"].quantile(0.75)

    lo_idx = df_valid[df_valid["D2"] < q25].sample(
        min(n_samples // 2, (df_valid["D2"] < q25).sum()),
        random_state=42).index.tolist()
    hi_idx = df_valid[df_valid["D2"] > q75].sample(
        min(n_samples // 2, (df_valid["D2"] > q75).sum()),
        random_state=42).index.tolist()

    X_full   = build_hybrid(prices_all, p)
    valid    = np.where(~np.any(np.isnan(X_full), axis=1))[0]
    X_v      = X_full[valid]
    dates_v  = bar_dates_all[valid]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"C(r) кривые — низкий vs высокий D₂  |  {title}", fontsize=10)

    for ax, idxs, label, color in [
        (axes[0], lo_idx, f"D₂ < {q25:.2f}  (простая геометрия)", "#2ca02c"),
        (axes[1], hi_idx, f"D₂ > {q75:.2f}  (сложная геометрия)",  "#d62728"),
    ]:
        for row_idx in idxs:
            t   = int(df_valid.loc[row_idx, "t"])
            pool = X_v[:t]
            mu   = pool.mean(0); sig = pool.std(0)
            sig  = np.where(sig < 1e-10, 1.0, sig)
            Xn   = (pool - mu) / sig

            d_flat = pdist(Xn).astype(np.float32)
            d_pos  = d_flat[d_flat > 1e-6]
            if len(d_pos) < 10:
                continue

            r_min = float(np.percentile(d_pos, 5))
            r_max = float(np.percentile(d_pos, 80))
            if r_min >= r_max:
                continue

            r_arr  = np.logspace(np.log10(r_min), np.log10(r_max), 30)
            C_arr  = np.array([np.mean(d_flat < r) for r in r_arr])
            mask_c = C_arr > 0

            date_str = str(dates_v[t])[:10]
            d2_val   = df_valid.loc[row_idx, "D2"]
            ax.plot(np.log10(r_arr[mask_c]), np.log10(C_arr[mask_c]),
                    color=color, lw=1.3, alpha=0.7,
                    label=f"{date_str}  D₂={d2_val:.2f}  N={t}")

        ax.set_title(label, fontsize=9)
        ax.set_xlabel("log₁₀ r", fontsize=8)
        ax.set_ylabel("log₁₀ C(r)", fontsize=8)
        ax.legend(fontsize=6.5)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, "d2_cr_curves.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Загрузка данных …")
    highs, lows, dates = load_10m()
    prices, bar_idxs, types = find_pivots(highs, lows, 0.04)
    bar_dates = dates[bar_idxs]

    print(f"  T=4%: {len(prices)} пивотов")
    print(f"  Период: {bar_dates[0][:10]} .. {bar_dates[-1][:10]}")

    print(f"\nWalk-forward D₂  p={P}  min_pool={MIN_POOL} …")
    df = run_causal_d2(prices, bar_dates, p=P, min_pool=MIN_POOL)

    df.to_csv(os.path.join(RES_DIR, "d2_causal_walk.csv"), index=False)

    df_valid = df.dropna(subset=["D2"])
    print(f"\n  Шагов с D₂: {len(df_valid)} / {len(df)}")
    print(f"  D₂: min={df_valid['D2'].min():.3f}  "
          f"median={df_valid['D2'].median():.3f}  "
          f"max={df_valid['D2'].max():.3f}  "
          f"std={df_valid['D2'].std():.3f}")

    # топ-5 минимумов и максимумов D₂
    print("\n  Точки с минимальным D₂ (простая геометрия):")
    low5 = df_valid.nsmallest(5, "D2")[["date", "price", "pool_n", "D2", "sig_lr1"]]
    print(low5.to_string(index=False))

    print("\n  Точки с максимальным D₂ (сложная геометрия):")
    hi5  = df_valid.nlargest(5, "D2")[["date", "price", "pool_n", "D2", "sig_lr1"]]
    print(hi5.to_string(index=False))

    # корреляция D₂ с sigma пула
    corr_sig = df_valid[["D2", "sig_price", "sig_lr1", "pool_n"]].corr()
    print("\n  Корреляция D₂ с параметрами пула:")
    print(corr_sig["D2"].drop("D2").to_string())

    print("\nГрафики …")
    plot_d2_vs_price(df, title="SBER 10m  T=4%  p=3")
    plot_c_curves(df, prices, bar_dates, p=P,
                  n_samples=6, title="SBER 10m  T=4%  p=3")

    print("\nГотово.")


if __name__ == "__main__":
    main()
